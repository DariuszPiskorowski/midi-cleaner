from __future__ import annotations

import json
from pathlib import Path

import mido
from typer.testing import CliRunner

from midi_cleaner.alignment.models import AudioAlignedNoteDocument, AudioAlignedNoteEvent
from midi_cleaner.cleanup.cleaned_exporter import CleanedMidiExportParameters, export_cleaned_midi
from midi_cleaner.cleanup.models import CleanupAction, CleanupPlanDocument
from midi_cleaner.cli import app
from midi_cleaner.midi.models import NoteEvent, NoteEventDocument, TempoEvent


runner = CliRunner()


def _note(
    note_id: str,
    pitch: int,
    velocity: int,
    start_tick: int,
    end_tick: int,
    channel: int = 0,
    layer: str = "bass",
) -> NoteEvent:
    return NoteEvent(
        note_id=note_id,
        source="ripx",
        layer=layer,
        track_index=0,
        track_name="Source",
        channel=channel,
        pitch_midi=pitch,
        pitch_name="C4",
        velocity=velocity,
        start_tick=start_tick,
        end_tick=end_tick,
        duration_ticks=end_tick - start_tick,
        start_sec=0.0,
        end_sec=0.0,
        duration_sec=0.0,
    )


def _action(note_id: str, plan_action: str, confidence: float = 0.5) -> CleanupAction:
    return CleanupAction(
        note_id=note_id,
        original_recommended_action="REVIEW",
        plan_action=plan_action,
        confidence=confidence,
        reasons=["reason"],
        source_validation={"recommended_action": "REVIEW"},
    )


def _write_inputs(
    tmp_path: Path,
    notes: list[NoteEvent],
    actions: list[CleanupAction],
) -> tuple[Path, Path]:
    notes_doc = NoteEventDocument(
        schema_version="0.1.0",
        source_file="candidate.mid",
        source="ripx",
        layer="bass",
        ticks_per_beat=480,
        tempo_map=[TempoEvent(tick=0, tempo_us_per_beat=500000, sec=0.0)],
        notes=notes,
    )
    cleanup_doc = CleanupPlanDocument(
        schema_version="0.1.0",
        validation_file="note_validation.json",
        layer="bass",
        planner_parameters={
            "mute_threshold": 0.45,
            "review_threshold": 0.70,
            "delete_threshold": 0.20,
            "allow_delete_candidates": False,
        },
        actions=actions,
    )

    notes_path = tmp_path / "note_events.json"
    plan_path = tmp_path / "cleanup_plan.json"
    notes_path.write_text(notes_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    plan_path.write_text(cleanup_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return notes_path, plan_path


def _extract_note_pairs_with_channel(midi_path: Path) -> list[tuple[int, int, int, int]]:
    midi_file = mido.MidiFile(str(midi_path))
    track = midi_file.tracks[0]

    absolute_tick = 0
    active: dict[tuple[int, int], list[tuple[int, int]]] = {}
    pairs: list[tuple[int, int, int, int]] = []

    for message in track:
        absolute_tick += message.time
        if message.type == "note_on" and message.velocity > 0:
            key = (message.channel, message.note)
            active.setdefault(key, []).append((absolute_tick, message.velocity))
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            key = (message.channel, message.note)
            if key in active and active[key]:
                start_tick, velocity = active[key].pop(0)
                pairs.append((message.note, velocity, message.channel, absolute_tick - start_tick))

    return pairs


def _extract_note_on_times_sec(midi_path: Path) -> tuple[list[float], int, int]:
    midi_file = mido.MidiFile(str(midi_path))
    track = midi_file.tracks[0]

    absolute_tick = 0
    tempo_us_per_beat = 500000
    note_on_times: list[float] = []

    for message in track:
        absolute_tick += message.time
        if message.type == "set_tempo":
            tempo_us_per_beat = int(message.tempo)
        if message.type == "note_on" and message.velocity > 0:
            note_on_times.append(
                (absolute_tick / midi_file.ticks_per_beat) * (tempo_us_per_beat / 1_000_000)
            )

    return note_on_times, tempo_us_per_beat, midi_file.ticks_per_beat


def _write_audio_aligned_notes(
    tmp_path: Path,
    note_id: str,
    aligned_start_sec: float,
    aligned_end_sec: float,
) -> Path:
    aligned_doc = AudioAlignedNoteDocument(
        schema_version="0.1.0",
        notes_file="note_events.json",
        audio_features_file="audio_features.json",
        layer="bass",
        sample_rate=44100,
        audio_duration_sec=2.0,
        alignment_parameters={
            "onset_search_window_ms": 250.0,
            "offset_search_window_ms": 350.0,
            "min_onset_score": 0.005,
            "min_rms": 0.001,
            "snap_start_to_audio_onset": True,
            "snap_end_to_energy_offset": True,
            "max_start_correction_ms": 500.0,
            "max_end_correction_ms": 800.0,
            "low_confidence_action": "KEEP_ORIGINAL_LOW_CONFIDENCE",
        },
        notes=[
            AudioAlignedNoteEvent(
                note_id=note_id,
                source="ripx",
                layer="bass",
                pitch_midi=72,
                pitch_name="C5",
                velocity=88,
                channel=3,
                original_start_sec=0.0,
                original_end_sec=0.0,
                original_duration_sec=0.0,
                original_start_tick=1000,
                original_end_tick=1200,
                aligned_start_sec=aligned_start_sec,
                aligned_end_sec=aligned_end_sec,
                aligned_duration_sec=aligned_end_sec - aligned_start_sec,
                start_correction_ms=0.0,
                end_correction_ms=0.0,
                duration_correction_ms=0.0,
                nearest_audio_onset_sec=aligned_start_sec,
                nearest_audio_offset_sec=aligned_end_sec,
                onset_error_before_ms=0.0,
                onset_error_after_ms=0.0,
                local_rms=0.01,
                local_onset_score=0.02,
                sustained_energy_ratio=0.8,
                alignment_confidence=0.9,
                alignment_action="ALIGNED",
                reasons=["test"],
            )
        ],
    )

    aligned_path = tmp_path / "audio_aligned_note_events.json"
    aligned_path.write_text(aligned_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return aligned_path


def test_cleaned_contains_only_keep_by_default(tmp_path: Path) -> None:
    notes = [_note("k", 60, 90, 0, 120), _note("r", 62, 70, 120, 240)]
    actions = [_action("k", "KEEP"), _action("r", "REVIEW")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    report = export_cleaned_midi(
        notes_file=notes_path,
        cleanup_plan_file=plan_path,
        output_dir=tmp_path / "out",
        params=CleanedMidiExportParameters(),
    )

    assert _extract_note_pairs_with_channel(tmp_path / "out" / "cleaned.mid") == [(60, 90, 0, 120)]
    assert report.cleaned_note_count == 1


def test_review_not_in_cleaned_by_default(tmp_path: Path) -> None:
    notes = [_note("r", 62, 70, 120, 240)]
    actions = [_action("r", "REVIEW")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    export_cleaned_midi(notes_path, plan_path, tmp_path / "out", CleanedMidiExportParameters())

    assert _extract_note_pairs_with_channel(tmp_path / "out" / "cleaned.mid") == []


def test_review_included_in_cleaned_when_enabled(tmp_path: Path) -> None:
    notes = [_note("r", 62, 70, 120, 240)]
    actions = [_action("r", "REVIEW")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    export_cleaned_midi(
        notes_path,
        plan_path,
        tmp_path / "out",
        CleanedMidiExportParameters(include_review_in_cleaned=True),
    )

    assert _extract_note_pairs_with_channel(tmp_path / "out" / "cleaned.mid") == [(62, 70, 0, 120)]


def test_rejected_contains_mute_and_delete_candidate(tmp_path: Path) -> None:
    notes = [_note("m", 64, 50, 0, 100), _note("d", 65, 80, 200, 260)]
    actions = [_action("m", "MUTE"), _action("d", "DELETE_CANDIDATE")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    report = export_cleaned_midi(notes_path, plan_path, tmp_path / "out", CleanedMidiExportParameters())

    rejected = _extract_note_pairs_with_channel(tmp_path / "out" / "rejected.mid")
    assert (64, 50, 0, 100) in rejected
    assert (65, 80, 0, 60) in rejected
    assert report.rejected_note_count == 2


def test_review_mid_contains_review_notes(tmp_path: Path) -> None:
    notes = [_note("r", 67, 77, 10, 210)]
    actions = [_action("r", "REVIEW")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    report = export_cleaned_midi(notes_path, plan_path, tmp_path / "out", CleanedMidiExportParameters())

    assert _extract_note_pairs_with_channel(tmp_path / "out" / "review.mid") == [(67, 77, 0, 200)]
    assert report.review_note_count == 1


def test_export_preserves_pitch_velocity_channel_and_duration(tmp_path: Path) -> None:
    notes = [_note("k", 72, 88, 500, 860, channel=3)]
    actions = [_action("k", "KEEP")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    export_cleaned_midi(notes_path, plan_path, tmp_path / "out", CleanedMidiExportParameters())

    pairs = _extract_note_pairs_with_channel(tmp_path / "out" / "cleaned.mid")
    assert pairs == [(72, 88, 3, 360)]


def test_missing_plan_action_creates_warning(tmp_path: Path) -> None:
    notes = [_note("k", 60, 90, 0, 120), _note("missing", 61, 90, 120, 240)]
    actions = [_action("k", "KEEP")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    report = export_cleaned_midi(notes_path, plan_path, tmp_path / "out", CleanedMidiExportParameters())

    assert report.warning_count >= 1
    assert any("No plan action" in warning for warning in report.warnings)


def test_unknown_note_id_in_plan_creates_warning(tmp_path: Path) -> None:
    notes = [_note("k", 60, 90, 0, 120)]
    actions = [_action("k", "KEEP"), _action("unknown", "REVIEW")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    report = export_cleaned_midi(notes_path, plan_path, tmp_path / "out", CleanedMidiExportParameters())

    assert report.warning_count >= 1
    assert any("unknown note_id" in warning for warning in report.warnings)


def test_cli_writes_cleaned_review_rejected_and_report(tmp_path: Path) -> None:
    notes = [
        _note("k", 60, 90, 0, 120),
        _note("r", 62, 70, 120, 240),
        _note("m", 64, 50, 240, 360),
        _note("d", 65, 80, 360, 480),
    ]
    actions = [
        _action("k", "KEEP"),
        _action("r", "REVIEW"),
        _action("m", "MUTE"),
        _action("d", "DELETE_CANDIDATE"),
    ]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    out_dir = tmp_path / "cleaned_midi"
    report_path = out_dir / "cleaned_export_report.json"

    result = runner.invoke(
        app,
        [
            "cleanup",
            "export-cleaned-midi",
            "--notes",
            str(notes_path),
            "--plan",
            str(plan_path),
            "--output-dir",
            str(out_dir),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert (out_dir / "cleaned.mid").exists()
    assert (out_dir / "review.mid").exists()
    assert (out_dir / "rejected.mid").exists()
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["cleaned_note_count"] == 1
    assert report["review_note_count"] == 1
    assert report["rejected_note_count"] == 2


def test_audio_aligned_seconds_drive_export_timing_with_strict_tolerance(tmp_path: Path) -> None:
    note = _note("k", 72, 88, 1000, 1200, channel=3)
    notes_path, plan_path = _write_inputs(tmp_path, [note], [_action("k", "KEEP")])
    aligned_path = _write_audio_aligned_notes(
        tmp_path,
        note_id="k",
        aligned_start_sec=0.123,
        aligned_end_sec=0.287,
    )

    report = export_cleaned_midi(
        notes_file=notes_path,
        cleanup_plan_file=plan_path,
        output_dir=tmp_path / "out",
        params=CleanedMidiExportParameters(
            ticks_per_beat=960,
            audio_aligned_notes_file=aligned_path,
        ),
    )

    note_on_times_sec, tempo_us_per_beat, exported_tpb = _extract_note_on_times_sec(
        tmp_path / "out" / "cleaned.mid"
    )

    assert note_on_times_sec
    assert abs(note_on_times_sec[0] - 0.123) <= 0.002
    assert report.timing_source == "audio_aligned_seconds"
    assert report.max_export_time_error_ms <= 2.0
    assert report.mean_export_time_error_ms <= 2.0
    assert tempo_us_per_beat == report.tempo_us_per_beat
    assert exported_tpb == report.exported_ticks_per_beat
