from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from midi_cleaner.cleanup.models import (
    CleanedMidiExportFile,
    CleanedMidiExportReport,
    CleanupPlanDocument,
)
from midi_cleaner.midi.models import NoteEvent, NoteEventDocument


class CleanedMidiExportError(Exception):
    """Raised when cleaned MIDI export cannot be completed."""


@dataclass(frozen=True)
class CleanedMidiExportParameters:
    ticks_per_beat: int = 960
    track_name_prefix: str = "Hermes"
    include_review_in_cleaned: bool = False
    write_empty_files: bool = True


def _load_notes_document(path: Path) -> NoteEventDocument:
    try:
        return NoteEventDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise CleanedMidiExportError(f"Invalid notes JSON: {path}") from exc


def _load_cleanup_plan(path: Path) -> CleanupPlanDocument:
    try:
        return CleanupPlanDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise CleanedMidiExportError(f"Invalid cleanup plan JSON: {path}") from exc


def _absolute_events(note: NoteEvent) -> list[tuple[int, int, mido.Message]]:
    channel = 0 if note.channel is None else int(note.channel)
    start_tick = int(note.start_tick)
    end_tick = int(note.end_tick)
    if end_tick < start_tick:
        end_tick = start_tick

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


def _write_midi_track(
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


def export_cleaned_midi(
    notes_file: Path,
    cleanup_plan_file: Path,
    output_dir: Path,
    params: CleanedMidiExportParameters,
) -> CleanedMidiExportReport:
    if not notes_file.exists() or not notes_file.is_file():
        raise CleanedMidiExportError(f"Notes file does not exist: {notes_file}")
    if not cleanup_plan_file.exists() or not cleanup_plan_file.is_file():
        raise CleanedMidiExportError(f"Cleanup plan file does not exist: {cleanup_plan_file}")

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

    cleaned_actions = {"KEEP"}
    if params.include_review_in_cleaned:
        cleaned_actions.add("REVIEW")

    cleaned_notes: list[NoteEvent] = []
    review_notes: list[NoteEvent] = []
    rejected_notes: list[NoteEvent] = []

    for note in note_document.notes:
        action = action_by_note_id.get(note.note_id)
        if action is None:
            continue
        if action in cleaned_actions:
            cleaned_notes.append(note)
        if action == "REVIEW":
            review_notes.append(note)
        if action in {"MUTE", "DELETE_CANDIDATE"}:
            rejected_notes.append(note)

    output_dir.mkdir(parents=True, exist_ok=True)

    export_specs = [
        (
            "CLEANED",
            "cleaned.mid",
            cleaned_notes,
            sorted(cleaned_actions),
        ),
        (
            "REVIEW",
            "review.mid",
            review_notes,
            ["REVIEW"],
        ),
        (
            "REJECTED",
            "rejected.mid",
            rejected_notes,
            ["DELETE_CANDIDATE", "MUTE"],
        ),
    ]

    exported_files: list[CleanedMidiExportFile] = []
    for role, filename, notes, included_actions in export_specs:
        output_path = output_dir / filename
        if not notes and not params.write_empty_files:
            warnings.append(f"Skipped empty MIDI file for role {role.lower()}: {output_path}")
            continue

        track_name = f"{params.track_name_prefix} {role} {note_document.layer}"
        _write_midi_track(
            notes=notes,
            output_path=output_path,
            ticks_per_beat=params.ticks_per_beat,
            track_name=track_name,
        )
        exported_files.append(
            CleanedMidiExportFile(
                role=role,
                path=str(output_path),
                note_count=len(notes),
                included_plan_actions=included_actions,
            )
        )

    return CleanedMidiExportReport(
        notes_file=str(notes_file),
        cleanup_plan_file=str(cleanup_plan_file),
        status="ok",
        layer=note_document.layer,
        ticks_per_beat=params.ticks_per_beat,
        cleaned_note_count=len(cleaned_notes),
        review_note_count=len(review_notes),
        rejected_note_count=len(rejected_notes),
        exported_files=exported_files,
        warning_count=len(warnings),
        warnings=warnings,
    )
