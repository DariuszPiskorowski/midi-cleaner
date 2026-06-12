from __future__ import annotations

import json
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from midi_cleaner.alignment.audio_time import AudioTimeAlignmentParameters, align_notes_to_audio_time
from midi_cleaner.audio.analyzer import analyze_stem
from midi_cleaner.cli import app
from midi_cleaner.midi.importer import import_midi_candidate


runner = CliRunner()


def _write_pulse_wav(
    path: Path,
    pulses_sec: list[float],
    duration_sec: float = 2.0,
    sample_rate: int = 44100,
) -> None:
    samples = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
    pulse_len = int(0.015 * sample_rate)
    for pulse_sec in pulses_sec:
        start = int(pulse_sec * sample_rate)
        end = min(len(samples), start + pulse_len)
        samples[start:end] = 0.9
    sf.write(str(path), samples, sample_rate)


def _write_single_note_midi(
    path: Path,
    start_sec: float,
    end_sec: float,
    ticks_per_beat: int = 480,
    tempo_us_per_beat: int = 500000,
) -> None:
    ticks_per_second = (ticks_per_beat * 1_000_000.0) / float(tempo_us_per_beat)
    start_tick = int(round(start_sec * ticks_per_second))
    end_tick = int(round(end_sec * ticks_per_second))
    if end_tick <= start_tick:
        end_tick = start_tick + 1

    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us_per_beat, time=0))
    track.append(mido.Message("note_on", note=45, velocity=100, time=start_tick, channel=0))
    track.append(mido.Message("note_off", note=45, velocity=0, time=end_tick - start_tick, channel=0))

    midi.save(path)


def test_audio_onset_alignment_moves_note_start_close_to_pulse(tmp_path: Path) -> None:
    wav_path = tmp_path / "pulse.wav"
    midi_path = tmp_path / "candidate.mid"
    notes_path = tmp_path / "note_events.json"
    audio_features_path = tmp_path / "audio_features.json"

    _write_pulse_wav(wav_path, pulses_sec=[1.0])
    _write_single_note_midi(midi_path, start_sec=1.5, end_sec=1.8)

    notes_doc, _ = import_midi_candidate(midi_path, source="ripx", layer="bass")
    audio_doc, _ = analyze_stem(wav_path, layer="bass")

    notes_path.write_text(notes_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    audio_features_path.write_text(audio_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")

    aligned_doc, aligned_report = align_notes_to_audio_time(
        notes_file=notes_path,
        audio_features_file=audio_features_path,
        params=AudioTimeAlignmentParameters(
            onset_search_window_ms=600.0,
            max_start_correction_ms=600.0,
        ),
    )

    assert aligned_report.note_count == 1
    assert aligned_report.aligned_count == 1

    aligned_note = aligned_doc.notes[0]
    assert aligned_note.alignment_action == "ALIGNED"
    assert abs(aligned_note.aligned_start_sec - 1.0) <= 0.04
    assert abs(aligned_note.aligned_start_sec - aligned_note.original_start_sec) >= 0.45


def test_global_offset_alignment_recovers_large_shift_before_local_snap(tmp_path: Path) -> None:
    wav_path = tmp_path / "pulse_global.wav"
    midi_path = tmp_path / "candidate_global.mid"
    notes_path = tmp_path / "note_events.json"
    audio_features_path = tmp_path / "audio_features.json"

    _write_pulse_wav(wav_path, pulses_sec=[1.0], duration_sec=4.0)
    _write_single_note_midi(midi_path, start_sec=3.0, end_sec=3.25)

    notes_doc, _ = import_midi_candidate(midi_path, source="ripx", layer="bass")
    audio_doc, _ = analyze_stem(wav_path, layer="bass")

    notes_path.write_text(notes_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    audio_features_path.write_text(audio_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")

    aligned_doc, aligned_report = align_notes_to_audio_time(
        notes_file=notes_path,
        audio_features_file=audio_features_path,
        params=AudioTimeAlignmentParameters(
            onset_search_window_ms=120.0,
            max_start_correction_ms=120.0,
            global_max_search_offset_ms=3000.0,
            global_search_step_ms=10.0,
            global_min_confidence=0.05,
        ),
    )

    aligned_note = aligned_doc.notes[0]
    assert aligned_note.alignment_action == "ALIGNED"
    assert abs(aligned_note.aligned_start_sec - 1.0) <= 0.04
    assert aligned_report.global_offset_applied is True
    assert abs(aligned_report.global_offset_sec + 2.0) <= 0.12
    assert aligned_report.global_confidence >= 0.05


def test_no_audio_evidence_keeps_original_timing(tmp_path: Path) -> None:
    wav_path = tmp_path / "silence.wav"
    midi_path = tmp_path / "candidate.mid"
    notes_path = tmp_path / "note_events.json"
    audio_features_path = tmp_path / "audio_features.json"

    _write_pulse_wav(wav_path, pulses_sec=[], duration_sec=1.5)
    _write_single_note_midi(midi_path, start_sec=0.5, end_sec=0.7)

    notes_doc, _ = import_midi_candidate(midi_path, source="ripx", layer="bass")
    audio_doc, _ = analyze_stem(wav_path, layer="bass")

    notes_path.write_text(notes_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    audio_features_path.write_text(audio_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")

    aligned_doc, aligned_report = align_notes_to_audio_time(
        notes_file=notes_path,
        audio_features_file=audio_features_path,
        params=AudioTimeAlignmentParameters(),
    )

    assert aligned_report.note_count == 1
    assert aligned_report.no_audio_evidence_count == 1

    aligned_note = aligned_doc.notes[0]
    assert aligned_note.alignment_action == "NO_AUDIO_EVIDENCE"
    assert abs(aligned_note.aligned_start_sec - aligned_note.original_start_sec) <= 1e-9


def test_cli_align_audio_time_writes_outputs(tmp_path: Path) -> None:
    wav_path = tmp_path / "pulse_cli.wav"
    midi_path = tmp_path / "candidate_cli.mid"
    notes_path = tmp_path / "note_events.json"
    audio_features_path = tmp_path / "audio_features.json"
    aligned_output_path = tmp_path / "out" / "audio_aligned_note_events.json"
    report_path = tmp_path / "out" / "audio_alignment_report.json"

    _write_pulse_wav(wav_path, pulses_sec=[0.8])
    _write_single_note_midi(midi_path, start_sec=0.95, end_sec=1.2)

    notes_doc, _ = import_midi_candidate(midi_path, source="ripx", layer="bass")
    audio_doc, _ = analyze_stem(wav_path, layer="bass")

    notes_path.write_text(notes_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    audio_features_path.write_text(audio_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "validate",
            "align-audio-time",
            "--notes",
            str(notes_path),
            "--audio-features",
            str(audio_features_path),
            "--output",
            str(aligned_output_path),
            "--report",
            str(report_path),
            "--onset-search-window-ms",
            "300",
        ],
    )

    assert result.exit_code == 0
    assert aligned_output_path.exists()
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["output_file"] == str(aligned_output_path)
