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
    low_ratio: float = 0.7,
    mid_ratio: float = 0.2,
    high_ratio: float = 0.1,
) -> None:
    monkeypatch.setattr(
        drums_extract_audio,
        "_detect_onsets",
        lambda onset_strength, sample_rate, hop_size, min_onset_strength: (
            np.array(onset_times, dtype=np.float64),
            np.ones(len(onset_times), dtype=np.float64),
        ),
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

    state = {"index": 0}

    def _classify(
        onset_strength: float,
        low_ratio: float,
        mid_ratio: float,
        high_ratio: float,
        centroid_hz: float,
        rolloff_hz: float,
    ) -> tuple[str, float]:
        idx = state["index"]
        state["index"] += 1
        if idx >= len(class_names):
            idx = len(class_names) - 1
        return class_names[idx], 0.90

    monkeypatch.setattr(drums_extract_audio, "_classify_hit", _classify)


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
        ]
    )
    assert "tick" in header
    assert "spectral_centroid" in header
    assert "onset_strength" in header


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
