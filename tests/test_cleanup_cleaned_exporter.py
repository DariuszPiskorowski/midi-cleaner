from __future__ import annotations

import json
from pathlib import Path

import mido
from typer.testing import CliRunner

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
