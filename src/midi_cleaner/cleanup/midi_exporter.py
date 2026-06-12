from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mido

from midi_cleaner.cleanup.models import (
    CleanupPlanDocument,
    ReviewMidiExportFile,
    ReviewMidiExportReport,
)
from midi_cleaner.midi.models import NoteEvent, NoteEventDocument


class ReviewMidiExportError(Exception):
    """Raised when review MIDI export cannot be completed."""


@dataclass(frozen=True)
class ReviewMidiExportParameters:
    ticks_per_beat: int = 960
    track_name_prefix: str = "Hermes"
    include_delete_candidates: bool = True


def _load_notes_document(path: Path) -> NoteEventDocument:
    try:
        return NoteEventDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise ReviewMidiExportError(f"Invalid notes JSON: {path}") from exc


def _load_cleanup_plan(path: Path) -> CleanupPlanDocument:
    try:
        return CleanupPlanDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise ReviewMidiExportError(f"Invalid cleanup plan JSON: {path}") from exc


def _absolute_events(note: NoteEvent) -> list[tuple[int, int, mido.Message]]:
    channel = 0 if note.channel is None else int(note.channel)
    start_tick = int(note.start_tick)
    end_tick = int(note.end_tick)
    if end_tick < start_tick:
        end_tick = start_tick

    # Sort key uses order so note_off is emitted before note_on at equal ticks.
    return [
        (
            start_tick,
            1,
            mido.Message(
                "note_on",
                note=int(note.pitch_midi),
                velocity=int(note.velocity),
                channel=channel,
                time=0,
            ),
        ),
        (
            end_tick,
            0,
            mido.Message(
                "note_off",
                note=int(note.pitch_midi),
                velocity=0,
                channel=channel,
                time=0,
            ),
        ),
    ]


def _write_review_midi(
    notes: list[NoteEvent],
    output_path: Path,
    ticks_per_beat: int,
    track_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))

    absolute_events: list[tuple[int, int, mido.Message]] = []
    for note in notes:
        absolute_events.extend(_absolute_events(note))

    absolute_events.sort(key=lambda item: (item[0], item[1]))

    previous_tick = 0
    for tick, _order, message in absolute_events:
        delta = tick - previous_tick
        previous_tick = tick
        message.time = delta
        track.append(message)

    midi_file.save(str(output_path))


def export_review_midi(
    notes_file: Path,
    cleanup_plan_file: Path,
    output_dir: Path,
    params: ReviewMidiExportParameters,
) -> ReviewMidiExportReport:
    if not notes_file.exists() or not notes_file.is_file():
        raise ReviewMidiExportError(f"Notes file does not exist: {notes_file}")
    if not cleanup_plan_file.exists() or not cleanup_plan_file.is_file():
        raise ReviewMidiExportError(f"Cleanup plan file does not exist: {cleanup_plan_file}")

    note_document = _load_notes_document(notes_file)
    cleanup_plan = _load_cleanup_plan(cleanup_plan_file)

    warnings: list[str] = []
    if cleanup_plan.layer != note_document.layer:
        warnings.append(
            "Layer mismatch between note events and cleanup plan: "
            f"{note_document.layer} vs {cleanup_plan.layer}."
        )

    notes_by_id = {note.note_id: note for note in note_document.notes}
    action_by_note_id: dict[str, str] = {}

    for action in cleanup_plan.actions:
        if action.note_id not in notes_by_id:
            warnings.append(f"Plan references unknown note_id: {action.note_id}")
            continue
        action_by_note_id[action.note_id] = action.plan_action

    for note in note_document.notes:
        if note.note_id not in action_by_note_id:
            warnings.append(f"No plan action for note_id: {note.note_id}")

    grouped: dict[str, list[NoteEvent]] = defaultdict(list)
    for note in note_document.notes:
        plan_action = action_by_note_id.get(note.note_id)
        if plan_action is None:
            continue
        grouped[plan_action].append(note)

    output_dir.mkdir(parents=True, exist_ok=True)

    export_specs: list[tuple[str, str, bool]] = [
        ("KEEP", "keep.mid", True),
        ("REVIEW", "review.mid", True),
        ("MUTE", "muted.mid", True),
        (
            "DELETE_CANDIDATE",
            "delete_candidates.mid",
            params.include_delete_candidates and bool(grouped.get("DELETE_CANDIDATE")),
        ),
    ]

    exported_files: list[ReviewMidiExportFile] = []
    for action, filename, should_export in export_specs:
        if not should_export:
            continue

        action_notes = grouped.get(action, [])
        output_path = output_dir / filename
        track_name = f"{params.track_name_prefix} {action} {note_document.layer}"
        _write_review_midi(
            notes=action_notes,
            output_path=output_path,
            ticks_per_beat=params.ticks_per_beat,
            track_name=track_name,
        )

        exported_files.append(
            ReviewMidiExportFile(
                action=action,
                path=str(output_path),
                note_count=len(action_notes),
            )
        )

    return ReviewMidiExportReport(
        notes_file=str(notes_file),
        cleanup_plan_file=str(cleanup_plan_file),
        status="ok",
        layer=note_document.layer,
        ticks_per_beat=params.ticks_per_beat,
        exported_files=exported_files,
        warning_count=len(warnings),
        warnings=warnings,
    )
