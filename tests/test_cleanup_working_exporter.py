from __future__ import annotations

import json
from pathlib import Path

import mido
from typer.testing import CliRunner

from midi_cleaner.cleanup.models import CleanupAction, CleanupPlanDocument
from midi_cleaner.cleanup.working_exporter import (
    WorkingMidiExportParameters,
    export_working_midi,
)
from midi_cleaner.cli import app
from midi_cleaner.midi.models import NoteEvent, NoteEventDocument, TempoEvent
from midi_cleaner.refinement.models import RefinedNoteDocument, RefinedNoteEvent


runner = CliRunner()


def _note(note_id: str, pitch: int, velocity: int, start_tick: int, end_tick: int) -> NoteEvent:
    return NoteEvent(
        note_id=note_id,
        source="ripx",
        layer="bass",
        track_index=0,
        track_name="Source",
        channel=0,
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


def _action(note_id: str, plan_action: str) -> CleanupAction:
    return CleanupAction(
        note_id=note_id,
        original_recommended_action="REVIEW",
        plan_action=plan_action,
        confidence=0.5,
        reasons=["reason"],
        source_validation={"recommended_action": "REVIEW"},
    )


def _refined_note(
    note_id: str,
    pitch: int,
    velocity: int,
    start_sec: float,
    end_sec: float,
) -> RefinedNoteEvent:
    return RefinedNoteEvent(
        note_id=note_id,
        source="ripx",
        layer="bass",
        pitch_midi=pitch,
        pitch_name="C4",
        velocity=velocity,
        channel=0,
        original_start_sec=start_sec,
        original_end_sec=end_sec,
        aligned_start_sec=start_sec,
        aligned_end_sec=end_sec,
        refined_start_sec=start_sec,
        refined_end_sec=end_sec,
        refined_duration_sec=end_sec - start_sec,
        start_refinement_ms=0.0,
        end_refinement_ms=0.0,
        merged_note_ids=[],
        refinement_actions=["UNCHANGED"],
        refinement_confidence=0.9,
        reasons=["test"],
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    notes_doc = NoteEventDocument(
        schema_version="0.1.0",
        source_file="candidate.mid",
        source="ripx",
        layer="bass",
        ticks_per_beat=480,
        tempo_map=[TempoEvent(tick=0, tempo_us_per_beat=500000, sec=0.0)],
        notes=[
            _note("k", 60, 90, 100, 200),
            _note("r", 62, 70, 220, 300),
            _note("m", 64, 65, 320, 420),
            _note("d", 65, 65, 440, 520),
        ],
    )
    plan_doc = CleanupPlanDocument(
        schema_version="0.1.0",
        validation_file="note_validation.json",
        layer="bass",
        planner_parameters={
            "mute_threshold": 0.45,
            "review_threshold": 0.70,
            "delete_threshold": 0.20,
            "allow_delete_candidates": True,
        },
        actions=[
            _action("k", "KEEP"),
            _action("r", "REVIEW"),
            _action("m", "MUTE"),
            _action("d", "DELETE_CANDIDATE"),
        ],
    )
    refined_doc = RefinedNoteDocument(
        schema_version="0.1.0",
        aligned_notes_file="audio_aligned_note_events.json",
        audio_features_file="audio_features.json",
        validation_file="note_validation.json",
        layer="bass",
        sample_rate=44100,
        audio_duration_sec=2.0,
        refinement_parameters={
            "attack_lookback_ms": 80.0,
            "max_attack_advance_ms": 80.0,
            "merge_gap_ms": 160.0,
            "minimum_silence_ms": 80.0,
            "tail_rms_ratio": 0.2,
            "tail_silence_hold_ms": 120.0,
            "max_tail_extension_ms": 900.0,
            "minimum_note_duration_ms": 80.0,
            "monophonic": True,
            "allow_pitch_overlap": False,
        },
        notes=[
            _refined_note("k", 60, 90, 0.100, 0.200),
            _refined_note("r", 62, 70, 0.220, 0.340),
            _refined_note("m", 64, 65, 0.360, 0.500),
            _refined_note("d", 65, 65, 0.520, 0.660),
        ],
    )

    notes_path = tmp_path / "note_events.json"
    plan_path = tmp_path / "cleanup_plan.json"
    refined_path = tmp_path / "refined_note_events.json"

    notes_path.write_text(notes_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    plan_path.write_text(plan_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    refined_path.write_text(refined_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return notes_path, plan_path, refined_path


def _extract_note_on_times_sec(midi_path: Path) -> list[float]:
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

    return note_on_times


def test_working_export_includes_keep_review_and_rejected_sets(tmp_path: Path) -> None:
    notes_path, plan_path, refined_path = _write_inputs(tmp_path)

    report = export_working_midi(
        notes_file=notes_path,
        cleanup_plan_file=plan_path,
        output_dir=tmp_path / "out",
        params=WorkingMidiExportParameters(
            ticks_per_beat=960,
            refined_notes_file=refined_path,
            include_diagnostic=False,
        ),
    )

    working_path = tmp_path / "out" / "working.mid"
    rejected_path = tmp_path / "out" / "rejected.mid"

    assert working_path.exists()
    assert rejected_path.exists()

    working_onsets = _extract_note_on_times_sec(working_path)
    rejected_onsets = _extract_note_on_times_sec(rejected_path)

    assert len(working_onsets) == 2
    assert len(rejected_onsets) == 2

    assert abs(working_onsets[0] - 0.100) <= 0.002
    assert abs(working_onsets[1] - 0.220) <= 0.002
    assert abs(rejected_onsets[0] - 0.360) <= 0.002
    assert abs(rejected_onsets[1] - 0.520) <= 0.002

    assert report.timing_source == "refined_audio_seconds"
    assert report.working_note_count == 2
    assert report.rejected_note_count == 2
    assert report.max_export_time_error_ms <= 2.0
    assert report.mean_export_time_error_ms <= 2.0


def test_cli_export_working_midi_writes_outputs_and_report(tmp_path: Path) -> None:
    notes_path, plan_path, refined_path = _write_inputs(tmp_path)

    out_dir = tmp_path / "working"
    report_path = out_dir / "working_export_report.json"

    result = runner.invoke(
        app,
        [
            "cleanup",
            "export-working-midi",
            "--notes",
            str(notes_path),
            "--plan",
            str(plan_path),
            "--refined-notes",
            str(refined_path),
            "--output-dir",
            str(out_dir),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert (out_dir / "working.mid").exists()
    assert (out_dir / "rejected.mid").exists()
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["timing_source"] == "refined_audio_seconds"
    assert payload["working_note_count"] == 2
    assert payload["rejected_note_count"] == 2


def test_working_export_supports_custom_variant_filenames(tmp_path: Path) -> None:
    notes_path, plan_path, refined_path = _write_inputs(tmp_path)

    report = export_working_midi(
        notes_file=notes_path,
        cleanup_plan_file=plan_path,
        output_dir=tmp_path / "out",
        params=WorkingMidiExportParameters(
            refined_notes_file=refined_path,
            working_filename="working_iter2.mid",
            rejected_filename="rejected_iter2.mid",
            diagnostic_filename="diagnostic_iter2.mid",
            include_diagnostic=True,
        ),
    )

    assert (tmp_path / "out" / "working_iter2.mid").exists()
    assert (tmp_path / "out" / "rejected_iter2.mid").exists()
    assert (tmp_path / "out" / "diagnostic_iter2.mid").exists()
    exported_paths = {item.path for item in report.exported_files}
    assert str(tmp_path / "out" / "working_iter2.mid") in exported_paths


def test_working_export_keeps_repair_generated_notes_without_plan_action(tmp_path: Path) -> None:
    notes_path, plan_path, refined_path = _write_inputs(tmp_path)
    refined_doc = RefinedNoteDocument.model_validate_json(refined_path.read_text(encoding="utf-8"))
    generated = _refined_note("repair_missing_000001", 60, 90, 0.700, 0.820)
    generated = generated.model_copy(update={"refinement_actions": ["ACTIVITY_REPAIR_INSERTED"]})
    refined_doc = refined_doc.model_copy(update={"notes": [*refined_doc.notes, generated]})
    refined_path.write_text(refined_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")

    report = export_working_midi(
        notes_file=notes_path,
        cleanup_plan_file=plan_path,
        output_dir=tmp_path / "out_generated",
        params=WorkingMidiExportParameters(refined_notes_file=refined_path),
    )

    assert report.working_note_count == 3
    assert any("defaulted to KEEP" in warning for warning in report.warnings)
