from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

import midi_cleaner.drums.extract_audio as drums_extract_audio
from midi_cleaner.cli import app


runner = CliRunner()


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32), sample_rate)


def _build_drum_like_wav(path: Path, sample_rate: int = 44100) -> list[float]:
    duration_sec = 1.4
    total = int(round(duration_sec * sample_rate))
    signal = np.zeros(total, dtype=np.float64)

    onset_specs = [
        (0.10, 90.0),
        (0.28, 220.0),
        (0.46, 1400.0),
        (0.64, 4200.0),
        (0.88, 260.0),
        (1.08, 6000.0),
    ]

    burst_len = int(round(0.045 * sample_rate))
    envelope = np.exp(-np.linspace(0.0, 5.0, burst_len, endpoint=False))

    onset_times: list[float] = []
    for onset_sec, frequency in onset_specs:
        start = int(round(onset_sec * sample_rate))
        stop = min(total, start + burst_len)
        count = stop - start
        if count <= 0:
            continue

        t = np.arange(count, dtype=np.float64) / sample_rate
        burst = np.sin(2.0 * np.pi * frequency * t) * envelope[:count]
        signal[start:stop] += burst
        onset_times.append(onset_sec)

    signal = np.clip(signal, -0.98, 0.98)
    _write_wav(path, signal, sample_rate)
    return onset_times


def _note_on_events(midi_path: Path) -> list[tuple[int, int, int, int]]:
    midi = mido.MidiFile(str(midi_path))
    events: list[tuple[int, int, int, int]] = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if message.is_meta:
                continue
            if message.type == "note_on" and message.velocity > 0:
                events.append((tick, int(message.note), int(message.channel), int(message.velocity)))
    return sorted(events, key=lambda item: (item[0], item[1], item[2]))


def _max_tick(midi_path: Path) -> int:
    midi = mido.MidiFile(str(midi_path))
    max_tick = 0
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
        max_tick = max(max_tick, tick)
    return max_tick


def _patch_detected_hits(
    monkeypatch: pytest.MonkeyPatch,
    *,
    onset_times: list[float],
    class_names: list[str],
    onset_strengths: list[float] | None = None,
    confidences: list[float] | None = None,
    low_ratio: float = 0.7,
    mid_ratio: float = 0.2,
    high_ratio: float = 0.1,
) -> None:
    strengths = onset_strengths or [1.0 for _ in onset_times]
    confidence_values = confidences or [0.90 for _ in onset_times]
    frame_count = max(8, int(round((max(onset_times) if onset_times else 0.0) * 44100 / 256.0)) + 8)

    candidate_id = {"value": 0}

    def _build_candidates(detector_name: str) -> list[drums_extract_audio._HitCandidate]:
        candidates: list[drums_extract_audio._HitCandidate] = []
        for idx, onset_sec in enumerate(onset_times):
            class_name = class_names[idx] if idx < len(class_names) else class_names[-1]
            confidence = confidence_values[idx] if idx < len(confidence_values) else confidence_values[-1]
            strength = strengths[idx] if idx < len(strengths) else strengths[-1]
            cid = candidate_id["value"]
            candidate_id["value"] += 1
            candidates.append(
                drums_extract_audio._HitCandidate(
                    candidate_id=cid,
                    detector_name=detector_name,
                    onset_sec=float(onset_sec),
                    onset_strength=float(strength),
                    class_name=class_name,
                    low_peak_strength=0.85 if class_name == "kick" else 0.25,
                    mid_peak_strength=0.78 if class_name in {"snare_or_clap", "tom_or_perc"} else 0.30,
                    high_peak_strength=0.86 if class_name in {"hat", "cymbal"} else 0.20,
                    attack_score=0.75,
                    decay_score=0.35 if class_name != "cymbal" else 0.80,
                    band_dominance_score=0.72,
                    confidence=float(confidence),
                    competing_class="hat" if class_name == "cymbal" else "cymbal",
                    competing_class_score=0.30,
                    low_energy_ratio=low_ratio,
                    mid_energy_ratio=mid_ratio,
                    high_energy_ratio=high_ratio,
                    spectral_centroid=1500.0,
                )
            )
        return candidates

    monkeypatch.setattr(
        drums_extract_audio,
        "_onset_strength_envelopes",
        lambda audio, sample_rate: (
            {
                "full": np.ones(frame_count, dtype=np.float64),
                "low": np.full(frame_count, 0.85, dtype=np.float64),
                "mid": np.full(frame_count, 0.35, dtype=np.float64),
                "high": np.full(frame_count, 0.25, dtype=np.float64),
                "upper": np.full(frame_count, 0.22, dtype=np.float64),
                "kick": np.full(frame_count, 0.85, dtype=np.float64),
                "snare": np.full(frame_count, 0.50, dtype=np.float64),
                "hat": np.full(frame_count, 0.55, dtype=np.float64),
                "cymbal": np.full(frame_count, 0.48, dtype=np.float64),
                "tom": np.full(frame_count, 0.45, dtype=np.float64),
            },
            1024,
            256,
        ),
    )
    monkeypatch.setattr(
        drums_extract_audio,
        "_detect_onsets",
        lambda onset_strength, sample_rate, hop_size, min_onset_strength, **kwargs: (
            np.array(onset_times, dtype=np.float64),
            np.array(strengths, dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        drums_extract_audio,
        "_collect_multidetector_candidates",
        lambda **kwargs: _build_candidates("kick"),
    )
    monkeypatch.setattr(
        drums_extract_audio,
        "_collect_global_candidates",
        lambda **kwargs: _build_candidates("global"),
    )
    monkeypatch.setattr(
        drums_extract_audio,
        "_extract_hit_spectral_features",
        lambda audio, sample_rate, onset_sec: (
            low_ratio,
            mid_ratio,
            high_ratio,
            1500.0,
            4500.0,
            0.20,
            0.70,
        ),
    )


def test_audio_onset_extraction_creates_midi_notes(tmp_path: Path) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "drums_from_audio.mid"
    report_path = tmp_path / "drums_from_audio_report.json"
    _build_drum_like_wav(wav_path)

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "ujam-candy",
            "--c1-midi-note",
            "36",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert output_midi.exists()
    assert len(_note_on_events(output_midi)) > 0
    assert payload["onset_count"] > 0
    assert sum(payload["output_note_counts"].values()) > 0
    assert payload["output_pitch_counts"] == payload["output_note_counts"]


def test_drums_extract_command_is_registered() -> None:
    result = runner.invoke(
        app,
        ["drums", "extract-from-audio", "--help"],
        env={"COLUMNS": "240", "LINES": "120"},
    )

    assert result.exit_code == 0
    assert "--wav" in result.stdout
    assert "--output" in result.stdout
    assert "--target-map" in result.stdout
    assert "--detection-mode" in result.stdout
    assert "--min-class-confidence" in result.stdout
    assert "--emit-unknown" in result.stdout


def test_output_note_timing_follows_detected_onset_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "timing.mid"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.137, 0.509],
        class_names=["kick", "kick"],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--bpm",
            "120",
            "--min-onset-strength",
            "0.05",
        ],
    )

    note_on_ticks = [tick for tick, _note, _channel, _velocity in _note_on_events(output_midi)]

    assert result.exit_code == 0
    assert note_on_ticks == [132, 489]


def test_target_map_ujam_candy_maps_classes_to_expected_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "candy_map.mid"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.10, 0.30, 0.50, 0.70, 0.90],
        class_names=[
            "kick",
            "snare_or_clap",
            "hat",
            "cymbal",
            "tom_or_perc",
        ],
        low_ratio=0.7,
        mid_ratio=0.2,
        high_ratio=0.1,
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "ujam-candy",
            "--c1-midi-note",
            "36",
            "--snare-target",
            "clap",
            "--bpm",
            "120",
        ],
    )

    note_numbers = [note for _tick, note, _channel, _velocity in _note_on_events(output_midi)]

    assert result.exit_code == 0
    assert note_numbers == [36, 43, 48, 60, 50]


def test_forced_bpm_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "forced_bpm.mid"
    report_path = tmp_path / "forced_bpm_report.json"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.20],
        class_names=["kick"],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--bpm",
            "123",
            "--report",
            str(report_path),
        ],
    )

    midi = mido.MidiFile(str(output_midi))
    tempo_events = [
        msg
        for track in midi.tracks
        for msg in track
        if msg.is_meta and msg.type == "set_tempo"
    ]
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert len(tempo_events) == 1
    assert int(tempo_events[0].tempo) == int(round(60_000_000.0 / 123.0))
    assert payload["bpm_used"] == 123.0
    assert payload["bpm_source"] == "forced"


def test_auto_bpm_path_is_used_when_bpm_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "auto_bpm.mid"
    report_path = tmp_path / "auto_bpm_report.json"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.00, 0.50, 1.00],
        class_names=["kick", "kick", "kick"],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    tempo_events = [
        msg
        for track in mido.MidiFile(str(output_midi)).tracks
        for msg in track
        if msg.is_meta and msg.type == "set_tempo"
    ]

    assert result.exit_code == 0
    assert payload["bpm_source"] == "detected"
    assert payload["detected_bpm"] == 120.0
    assert payload["bpm_used"] == 120.0
    assert len(tempo_events) == 1
    assert int(tempo_events[0].tempo) == int(round(60_000_000.0 / 120.0))


def test_multi_detector_is_default_mode(tmp_path: Path) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "default_mode.mid"
    report_path = tmp_path / "default_mode_report.json"
    _build_drum_like_wav(wav_path)

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert payload["detection_mode"] == "multi-detector"


def test_global_mode_still_works_as_fallback(tmp_path: Path) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "global_mode.mid"
    report_path = tmp_path / "global_mode_report.json"
    _build_drum_like_wav(wav_path)

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--detection-mode",
            "global",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert payload["detection_mode"] == "global"
    assert output_midi.exists()


def test_report_includes_bpm_fields(tmp_path: Path) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "bpm_fields.mid"
    report_path = tmp_path / "bpm_fields_report.json"
    _build_drum_like_wav(wav_path)

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert "detected_bpm" in payload
    assert "bpm_used" in payload
    assert "bpm_source" in payload


def test_no_quantization_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "no_quantize.mid"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.133],
        class_names=["kick"],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--bpm",
            "120",
        ],
    )

    note_on_ticks = [tick for tick, _note, _channel, _velocity in _note_on_events(output_midi)]

    assert result.exit_code == 0
    assert note_on_ticks == [128]
    assert note_on_ticks[0] not in {120, 144}


def test_output_midi_uses_channel_10_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "channel10.mid"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.20, 0.40],
        class_names=["kick", "snare_or_clap"],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--bpm",
            "120",
        ],
    )

    channels = {channel for _tick, _note, channel, _velocity in _note_on_events(output_midi)}

    assert result.exit_code == 0
    assert channels == {9}


def test_dry_run_writes_report_but_not_midi(tmp_path: Path) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "dry.mid"
    report_path = tmp_path / "dry_report.json"
    _build_drum_like_wav(wav_path)

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--dry-run",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert report_path.exists()
    assert not output_midi.exists()
    assert payload["output_file"] is None


def test_report_and_debug_csv_are_written(tmp_path: Path) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "drums.mid"
    report_path = tmp_path / "report.json"
    debug_csv = tmp_path / "hits.csv"
    _build_drum_like_wav(wav_path)

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "ujam-candy",
            "--report",
            str(report_path),
            "--debug-csv",
            str(debug_csv),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    header = debug_csv.read_text(encoding="utf-8").splitlines()[0]

    assert result.exit_code == 0
    assert report_path.exists()
    assert debug_csv.exists()
    assert "output_pitch_counts" in payload
    assert "per_hit_summary" in payload
    assert len(payload["per_hit_summary"]) > 0
    assert all(
        key in payload["per_hit_summary"][0]
        for key in [
            "raw_onset_sec",
            "accepted_onset_sec",
            "tick",
            "class",
            "target_note",
            "velocity",
            "confidence",
            "low_energy_ratio",
            "mid_energy_ratio",
            "high_energy_ratio",
            "spectral_centroid",
            "onset_strength",
            "suppressed",
            "suppression_reason",
            "grouped_transient_id",
            "class_refractory_ms",
            "nearest_previous_same_class_ms",
        ]
    )
    assert "raw_onset_count" in payload
    assert "accepted_onset_count" in payload
    assert "suppressed_duplicate_count" in payload
    assert "suppressed_by_class" in payload
    assert "class_refractory_ms" in payload
    assert "notes_per_second" in payload
    assert "class_notes_per_second" in payload
    assert "velocity_summary" in payload
    assert "too_dense_warning" in payload
    assert "duplicate_interval_summary" in payload
    assert "raw_onset_sec" in header
    assert "accepted_onset_sec" in header
    assert "suppressed" in header
    assert "suppression_reason" in header
    assert "grouped_transient_id" in header
    assert "class_refractory_ms" in header
    assert "nearest_previous_same_class_ms" in header
    assert "detection_mode" in header
    assert "detector_name" in header
    assert "candidate_class" in header
    assert "accepted_class" in header
    assert "class_confidence" in header
    assert "competing_class" in header
    assert "competing_class_score" in header
    assert "accepted" in header
    assert "rejection_reason" in header
    assert "merged_with_transient_id" in header
    assert "detector_candidate_counts" in payload
    assert "detector_accepted_counts" in payload
    assert "detector_rejected_counts" in payload
    assert "low_confidence_rejected_count" in payload
    assert "rejected_by_reason" in payload
    assert "multi_detector_merge_conflicts" in payload
    assert "tick" in header
    assert "spectral_centroid" in header
    assert "onset_strength" in header


def test_kick_detector_detects_low_band_synthetic_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "kick_detector.mid"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.10, 0.35],
        class_names=["kick", "kick"],
        onset_strengths=[0.75, 0.78],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "ujam-candy",
            "--c1-midi-note",
            "36",
            "--detection-mode",
            "multi-detector",
            "--bpm",
            "120",
        ],
    )

    notes = [note for _tick, note, _channel, _velocity in _note_on_events(output_midi)]

    assert result.exit_code == 0
    assert notes == [36, 36]


def test_snare_detector_detects_broadband_snare_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "snare_detector.mid"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.12, 0.36],
        class_names=["snare_or_clap", "snare_or_clap"],
        onset_strengths=[0.72, 0.74],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "ujam-candy",
            "--c1-midi-note",
            "36",
            "--detection-mode",
            "multi-detector",
            "--bpm",
            "120",
        ],
    )

    notes = [note for _tick, note, _channel, _velocity in _note_on_events(output_midi)]

    assert result.exit_code == 0
    assert notes == [43, 43]


def test_hat_detector_detects_repeated_short_high_ticks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "hat_detector.mid"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.10, 0.18, 0.26],
        class_names=["hat", "hat", "hat"],
        onset_strengths=[0.70, 0.68, 0.66],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "ujam-candy",
            "--c1-midi-note",
            "36",
            "--detection-mode",
            "multi-detector",
            "--bpm",
            "120",
        ],
    )

    notes = [note for _tick, note, _channel, _velocity in _note_on_events(output_midi)]

    assert result.exit_code == 0
    assert notes == [48, 48, 48]


def test_tom_is_not_catch_all_when_confidence_is_low(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "tom_not_catchall.mid"
    report_path = tmp_path / "tom_not_catchall_report.json"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.10, 0.24, 0.39],
        class_names=["tom_or_perc", "tom_or_perc", "tom_or_perc"],
        onset_strengths=[0.25, 0.22, 0.20],
        confidences=[0.42, 0.40, 0.38],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--detection-mode",
            "multi-detector",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert payload["class_counts"]["tom_or_perc"] == 0
    assert payload["low_confidence_rejected_count"] >= 1


def test_low_confidence_candidates_are_skipped_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "low_conf_skip.mid"
    report_path = tmp_path / "low_conf_skip_report.json"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.14, 0.33],
        class_names=["kick", "snare_or_clap"],
        onset_strengths=[0.18, 0.21],
        confidences=[0.45, 0.47],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--detection-mode",
            "multi-detector",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    events = _note_on_events(output_midi)

    assert result.exit_code == 0
    assert len(events) == 0
    assert payload["accepted_onset_count"] == 0
    assert payload["low_confidence_rejected_count"] >= 2


def test_same_transient_allows_kick_hat_layering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "layer_kick_hat.mid"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.10, 0.12],
        class_names=["kick", "hat"],
        onset_strengths=[0.78, 0.70],
        confidences=[0.88, 0.82],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "ujam-candy",
            "--c1-midi-note",
            "36",
            "--detection-mode",
            "multi-detector",
            "--bpm",
            "120",
        ],
    )

    notes = sorted(note for _tick, note, _channel, _velocity in _note_on_events(output_midi))

    assert result.exit_code == 0
    assert notes == [36, 48]


def test_same_transient_does_not_create_duplicate_same_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "no_dupe_same_class.mid"
    report_path = tmp_path / "no_dupe_same_class_report.json"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.10, 0.12],
        class_names=["snare_or_clap", "snare_or_clap"],
        onset_strengths=[0.74, 0.72],
        confidences=[0.84, 0.82],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--detection-mode",
            "multi-detector",
            "--bpm",
            "120",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    notes = _note_on_events(output_midi)

    assert result.exit_code == 0
    assert len(notes) == 1
    assert payload["suppressed_duplicate_count"] >= 1


def test_duplicate_kick_hits_inside_refractory_window_are_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "kick_refractory.mid"
    report_path = tmp_path / "kick_refractory_report.json"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.10, 0.19],
        onset_strengths=[0.70, 0.65],
        class_names=["kick", "kick"],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--bpm",
            "120",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    note_events = _note_on_events(output_midi)

    assert result.exit_code == 0
    assert len(note_events) == 1
    assert payload["raw_onset_count"] == 2
    assert payload["accepted_onset_count"] == 1
    assert payload["suppressed_duplicate_count"] == 1
    assert payload["suppressed_by_class"]["kick"] >= 1


def test_stronger_hit_is_kept_when_class_duplicates_collide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "stronger_duplicate.mid"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.10, 0.19],
        onset_strengths=[0.35, 0.95],
        class_names=["kick", "kick"],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--bpm",
            "120",
        ],
    )

    note_on_ticks = [tick for tick, _note, _channel, _velocity in _note_on_events(output_midi)]

    assert result.exit_code == 0
    assert note_on_ticks == [182]


def test_cymbal_tail_chatter_is_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "cymbal_tail.mid"
    report_path = tmp_path / "cymbal_tail_report.json"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.10, 0.20, 0.30, 0.40],
        onset_strengths=[0.85, 0.55, 0.80, 0.50],
        class_names=["cymbal", "cymbal", "cymbal", "cymbal"],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--bpm",
            "120",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cymbal_count = payload["class_counts"]["cymbal"]

    assert result.exit_code == 0
    assert cymbal_count <= 2
    assert payload["suppressed_by_class"]["cymbal"] >= 2


def test_same_transient_grouping_prevents_multiple_same_class_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "same_transient.mid"
    report_path = tmp_path / "same_transient_report.json"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.100, 0.118, 0.130],
        onset_strengths=[0.72, 0.61, 0.52],
        class_names=["kick", "kick", "kick"],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--bpm",
            "120",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    reasons = [
        item.get("suppression_reason")
        for item in payload["per_hit_summary"]
        if item.get("suppressed")
    ]

    assert result.exit_code == 0
    assert payload["class_counts"]["kick"] == 1
    assert reasons.count("same_transient_group") >= 1


def test_velocity_scaling_is_not_saturated_to_maximum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "velocity_scaling.mid"
    report_path = tmp_path / "velocity_scaling_report.json"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.10, 0.30, 0.50, 0.70, 0.90, 1.10],
        onset_strengths=[0.14, 0.22, 0.35, 0.50, 0.72, 0.98],
        class_names=["kick", "kick", "kick", "kick", "kick", "kick"],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
            "--bpm",
            "120",
            "--report",
            str(report_path),
        ],
    )

    velocities = [velocity for _tick, _note, _channel, velocity in _note_on_events(output_midi)]
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert len(set(velocities)) > 1
    assert max(velocities) < 125
    assert payload["velocity_summary"]["max"] < 125
    assert payload["velocity_summary"]["p90"] < 125


def test_conservative_profile_reduces_dense_repeated_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_balanced = tmp_path / "balanced.mid"
    output_conservative = tmp_path / "conservative.mid"
    _build_drum_like_wav(wav_path)

    onset_times = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
    class_names = ["kick", "kick", "kick", "kick", "kick", "kick"]

    _patch_detected_hits(
        monkeypatch,
        onset_times=onset_times,
        onset_strengths=[0.80, 0.77, 0.75, 0.73, 0.70, 0.68],
        class_names=class_names,
    )
    balanced_result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_balanced),
            "--target-map",
            "gm",
            "--bpm",
            "120",
            "--profile",
            "balanced",
        ],
    )

    _patch_detected_hits(
        monkeypatch,
        onset_times=onset_times,
        onset_strengths=[0.80, 0.77, 0.75, 0.73, 0.70, 0.68],
        class_names=class_names,
    )
    conservative_result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_conservative),
            "--target-map",
            "gm",
            "--bpm",
            "120",
            "--profile",
            "conservative",
        ],
    )

    balanced_count = len(_note_on_events(output_balanced))
    conservative_count = len(_note_on_events(output_conservative))

    assert balanced_result.exit_code == 0
    assert conservative_result.exit_code == 0
    assert conservative_count < balanced_count


def test_separate_files_mode_creates_synchronized_class_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "main.mid"
    _build_drum_like_wav(wav_path)

    _patch_detected_hits(
        monkeypatch,
        onset_times=[0.10, 0.25, 0.40, 0.55, 0.70, 0.85],
        class_names=[
            "kick",
            "snare_or_clap",
            "hat",
            "cymbal",
            "tom_or_perc",
            "unknown",
        ],
    )

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "ujam-candy",
            "--separate-files",
            "--bpm",
            "120",
        ],
    )

    expected_files = [
        output_midi.parent / "kick.mid",
        output_midi.parent / "snare_clap.mid",
        output_midi.parent / "hat.mid",
        output_midi.parent / "cymbal.mid",
        output_midi.parent / "tom_perc.mid",
    ]

    assert result.exit_code == 0
    assert output_midi.exists()
    for path in expected_files:
        assert path.exists()
        assert _max_tick(path) == _max_tick(output_midi)


def test_source_wav_is_not_modified(tmp_path: Path) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "preserve_source.mid"
    _build_drum_like_wav(wav_path)
    before_hash = hashlib.sha256(wav_path.read_bytes()).hexdigest()

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
        ],
    )

    after_hash = hashlib.sha256(wav_path.read_bytes()).hexdigest()

    assert result.exit_code == 0
    assert before_hash == after_hash


def test_no_ai_is_invoked_during_audio_drum_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "no_ai.mid"
    _build_drum_like_wav(wav_path)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("AI completion should not be called for drums extract-from-audio")

    monkeypatch.setattr("midi_cleaner.cli.complete_ai_pattern_completion", _raise_if_called)

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
        ],
    )

    assert result.exit_code == 0


def test_extract_from_audio_does_not_call_remap_drums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "Drums.wav"
    output_midi = tmp_path / "no_remap.mid"
    _build_drum_like_wav(wav_path)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("remap_drums_file must not be called by extract-from-audio")

    monkeypatch.setattr("midi_cleaner.cli.remap_drums_file", _raise_if_called)

    result = runner.invoke(
        app,
        [
            "drums",
            "extract-from-audio",
            "--wav",
            str(wav_path),
            "--output",
            str(output_midi),
            "--target-map",
            "gm",
        ],
    )

    assert result.exit_code == 0
