from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from midi_cleaner.alignment.models import AudioAlignedNoteDocument
from midi_cleaner.cleanup.models import (
    CleanupPlanDocument,
    WorkingMidiExportFile,
    WorkingMidiExportReport,
)
from midi_cleaner.midi.models import NoteEventDocument
from midi_cleaner.refinement.models import RefinedNoteDocument, RefinedNoteEvent

DEFAULT_TEMPO_US_PER_BEAT = 500000


class WorkingMidiExportError(Exception):
    """Raised when working MIDI export cannot be completed."""


@dataclass(frozen=True)
class WorkingMidiExportParameters:
    ticks_per_beat: int | None = None
    track_name_prefix: str = "Hermes"
    include_diagnostic: bool = False
    write_empty_files: bool = True
    refined_notes_file: Path | None = None
    repair_plan_file: Path | None = None
    audio_aligned_notes_file: Path | None = None
    repair_extend_count: int = 0
    repair_shorten_count: int = 0
    repair_insert_missing_count: int = 0
    repair_split_count: int = 0
    repair_close_gap_count: int = 0
    repair_review_manual_count: int = 0
    working_filename: str = "working.mid"
    rejected_filename: str = "rejected.mid"
    diagnostic_filename: str = "diagnostic.mid"


@dataclass(frozen=True)
class _ExportNote:
    note_id: str
    pitch_midi: int
    velocity: int
    channel: int | None
    start_tick: int
    end_tick: int
    start_sec: float | None
    end_sec: float | None


def _load_notes_document(path: Path) -> NoteEventDocument:
    try:
        return NoteEventDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise WorkingMidiExportError(f"Invalid notes JSON: {path}") from exc


def _load_cleanup_plan(path: Path) -> CleanupPlanDocument:
    try:
        return CleanupPlanDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise WorkingMidiExportError(f"Invalid cleanup plan JSON: {path}") from exc


def _load_refined_document(path: Path) -> RefinedNoteDocument:
    try:
        return RefinedNoteDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise WorkingMidiExportError(f"Invalid refined notes JSON: {path}") from exc


def _load_audio_aligned_document(path: Path) -> AudioAlignedNoteDocument:
    try:
        return AudioAlignedNoteDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise WorkingMidiExportError(f"Invalid audio aligned notes JSON: {path}") from exc


def _tempo_us_per_beat_from_document(note_document: NoteEventDocument) -> int:
    if note_document.tempo_map:
        return int(note_document.tempo_map[0].tempo_us_per_beat)
    return DEFAULT_TEMPO_US_PER_BEAT


def _ticks_per_second(ticks_per_beat: int, tempo_us_per_beat: int) -> float:
    return (ticks_per_beat * 1_000_000.0) / float(tempo_us_per_beat)


def _seconds_to_tick(seconds: float, ticks_per_second: float) -> int:
    return max(0, int(round(seconds * ticks_per_second)))


def _tick_to_seconds(tick: int, ticks_per_second: float) -> float:
    return float(tick) / ticks_per_second


def _absolute_events(note: _ExportNote, start_tick: int, end_tick: int) -> list[tuple[int, int, mido.Message]]:
    channel = 0 if note.channel is None else int(note.channel)
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
    note_events: list[tuple[_ExportNote, int, int]],
    output_path: Path,
    ticks_per_beat: int,
    track_name: str,
    tempo_us_per_beat: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us_per_beat, time=0))

    absolute: list[tuple[int, int, mido.Message]] = []
    for note, start_tick, end_tick in note_events:
        absolute.extend(_absolute_events(note, start_tick=start_tick, end_tick=end_tick))

    absolute.sort(key=lambda item: (item[0], item[1]))

    previous_tick = 0
    for tick, _order, message in absolute:
        message.time = tick - previous_tick
        previous_tick = tick
        track.append(message)

    midi_file.save(str(output_path))


def _resolve_timing(
    note: _ExportNote,
    ticks_per_second: float,
) -> tuple[int, int, list[float]]:
    if note.start_sec is not None and note.end_sec is not None:
        start_sec = max(0.0, float(note.start_sec))
        end_sec = max(start_sec, float(note.end_sec))

        start_tick = _seconds_to_tick(start_sec, ticks_per_second)
        end_tick = _seconds_to_tick(end_sec, ticks_per_second)
        if end_tick <= start_tick:
            end_tick = start_tick + 1

        start_error_ms = abs(_tick_to_seconds(start_tick, ticks_per_second) - start_sec) * 1000.0
        end_error_ms = abs(_tick_to_seconds(end_tick, ticks_per_second) - end_sec) * 1000.0
        return start_tick, end_tick, [start_error_ms, end_error_ms]

    start_tick = int(note.start_tick)
    end_tick = int(note.end_tick)
    if end_tick < start_tick:
        end_tick = start_tick
    return start_tick, end_tick, []


def _action_priority(action: str) -> int:
    priorities = {
        "KEEP": 4,
        "REVIEW": 3,
        "MUTE": 2,
        "DELETE_CANDIDATE": 1,
    }
    return priorities.get(action, 0)


def _resolve_action_from_ids(note_ids: list[str], action_by_note_id: dict[str, str]) -> str | None:
    actions = [action_by_note_id[note_id] for note_id in note_ids if note_id in action_by_note_id]
    if not actions:
        return None
    return max(actions, key=_action_priority)


def _build_refined_export_notes(
    refined_document: RefinedNoteDocument,
    note_document: NoteEventDocument,
    action_by_note_id: dict[str, str],
    warnings: list[str],
) -> tuple[list[_ExportNote], dict[str, str]]:
    note_by_id = {note.note_id: note for note in note_document.notes}
    export_notes: list[_ExportNote] = []
    action_by_export_note_id: dict[str, str] = {}

    for refined in refined_document.notes:
        note_ids = [refined.note_id, *refined.merged_note_ids]
        resolved_action = _resolve_action_from_ids(note_ids, action_by_note_id)
        if resolved_action is None:
            # Keep iterative/repair-generated notes in working exports by default.
            if refined.note_id.startswith(("repair_missing_", "repair_split_")):
                resolved_action = "KEEP"
                warnings.append(
                    "No cleanup action for repair-generated note; defaulted to KEEP: "
                    f"{refined.note_id}"
                )
            else:
                warnings.append(f"No cleanup action for refined note and merges: {refined.note_id}")
                continue

        source_note = note_by_id.get(refined.note_id)
        if source_note is None:
            channel = refined.channel
            velocity = refined.velocity
            pitch_midi = refined.pitch_midi
            start_tick = 0
            end_tick = 1
        else:
            channel = source_note.channel
            velocity = source_note.velocity
            pitch_midi = source_note.pitch_midi
            start_tick = source_note.start_tick
            end_tick = source_note.end_tick

        export_note = _ExportNote(
            note_id=refined.note_id,
            pitch_midi=pitch_midi,
            velocity=velocity,
            channel=channel,
            start_tick=start_tick,
            end_tick=end_tick,
            start_sec=max(0.0, float(refined.refined_start_sec)),
            end_sec=max(float(refined.refined_start_sec), float(refined.refined_end_sec)),
        )
        export_notes.append(export_note)
        action_by_export_note_id[export_note.note_id] = resolved_action

    return export_notes, action_by_export_note_id


def _build_aligned_export_notes(
    note_document: NoteEventDocument,
    aligned_document: AudioAlignedNoteDocument,
    action_by_note_id: dict[str, str],
    warnings: list[str],
) -> tuple[list[_ExportNote], dict[str, str]]:
    aligned_by_note_id = {item.note_id: item for item in aligned_document.notes}
    export_notes: list[_ExportNote] = []
    action_by_export_note_id: dict[str, str] = {}

    for note in note_document.notes:
        action = action_by_note_id.get(note.note_id)
        if action is None:
            warnings.append(f"No plan action for note_id: {note.note_id}")
            continue

        aligned = aligned_by_note_id.get(note.note_id)
        if aligned is None:
            warnings.append(f"Audio alignment missing for note_id: {note.note_id}; used original ticks")
            start_sec = None
            end_sec = None
        else:
            start_sec = max(0.0, float(aligned.aligned_start_sec))
            end_sec = max(start_sec, float(aligned.aligned_end_sec))

        export_note = _ExportNote(
            note_id=note.note_id,
            pitch_midi=note.pitch_midi,
            velocity=note.velocity,
            channel=note.channel,
            start_tick=note.start_tick,
            end_tick=note.end_tick,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        export_notes.append(export_note)
        action_by_export_note_id[export_note.note_id] = action

    return export_notes, action_by_export_note_id


def _build_original_export_notes(
    note_document: NoteEventDocument,
    action_by_note_id: dict[str, str],
    warnings: list[str],
) -> tuple[list[_ExportNote], dict[str, str]]:
    export_notes: list[_ExportNote] = []
    action_by_export_note_id: dict[str, str] = {}

    for note in note_document.notes:
        action = action_by_note_id.get(note.note_id)
        if action is None:
            warnings.append(f"No plan action for note_id: {note.note_id}")
            continue

        export_note = _ExportNote(
            note_id=note.note_id,
            pitch_midi=note.pitch_midi,
            velocity=note.velocity,
            channel=note.channel,
            start_tick=note.start_tick,
            end_tick=note.end_tick,
            start_sec=None,
            end_sec=None,
        )
        export_notes.append(export_note)
        action_by_export_note_id[export_note.note_id] = action

    return export_notes, action_by_export_note_id


def export_working_midi(
    notes_file: Path,
    cleanup_plan_file: Path,
    output_dir: Path,
    params: WorkingMidiExportParameters,
) -> WorkingMidiExportReport:
    if not notes_file.exists() or not notes_file.is_file():
        raise WorkingMidiExportError(f"Notes file does not exist: {notes_file}")
    if not cleanup_plan_file.exists() or not cleanup_plan_file.is_file():
        raise WorkingMidiExportError(f"Cleanup plan file does not exist: {cleanup_plan_file}")

    note_document = _load_notes_document(notes_file)
    cleanup_plan = _load_cleanup_plan(cleanup_plan_file)

    refined_document: RefinedNoteDocument | None = None
    aligned_document: AudioAlignedNoteDocument | None = None
    if params.refined_notes_file is not None:
        if not params.refined_notes_file.exists() or not params.refined_notes_file.is_file():
            raise WorkingMidiExportError(f"Refined notes file does not exist: {params.refined_notes_file}")
        refined_document = _load_refined_document(params.refined_notes_file)
    if params.audio_aligned_notes_file is not None:
        if not params.audio_aligned_notes_file.exists() or not params.audio_aligned_notes_file.is_file():
            raise WorkingMidiExportError(
                f"Audio aligned notes file does not exist: {params.audio_aligned_notes_file}"
            )
        aligned_document = _load_audio_aligned_document(params.audio_aligned_notes_file)

    warnings: list[str] = []
    if cleanup_plan.layer != note_document.layer:
        warnings.append(
            "Layer mismatch between note events and cleanup plan: "
            f"{note_document.layer} vs {cleanup_plan.layer}."
        )
    if refined_document is not None and refined_document.layer != note_document.layer:
        warnings.append(
            "Layer mismatch between note events and refined notes: "
            f"{note_document.layer} vs {refined_document.layer}."
        )
    if aligned_document is not None and aligned_document.layer != note_document.layer:
        warnings.append(
            "Layer mismatch between note events and audio alignment: "
            f"{note_document.layer} vs {aligned_document.layer}."
        )

    source_ticks_per_beat = int(note_document.ticks_per_beat)
    if params.ticks_per_beat is None:
        exported_ticks_per_beat = source_ticks_per_beat
        ticks_per_beat_source = "auto_from_note_events"
    else:
        exported_ticks_per_beat = int(params.ticks_per_beat)
        ticks_per_beat_source = "user_override"
        if exported_ticks_per_beat <= 0:
            warnings.append("Invalid ticks_per_beat override; using note_events ticks_per_beat instead.")
            exported_ticks_per_beat = source_ticks_per_beat
            ticks_per_beat_source = "auto_from_note_events"

    tempo_us_per_beat = _tempo_us_per_beat_from_document(note_document)
    ticks_per_second = _ticks_per_second(exported_ticks_per_beat, tempo_us_per_beat)

    action_by_note_id: dict[str, str] = {}
    note_ids = {note.note_id for note in note_document.notes}
    for item in cleanup_plan.actions:
        if item.note_id not in note_ids:
            warnings.append(f"Plan references unknown note_id: {item.note_id}")
            continue
        action_by_note_id[item.note_id] = item.plan_action

    if refined_document is not None:
        timing_source = "refined_audio_seconds"
        export_notes, action_by_export_note_id = _build_refined_export_notes(
            refined_document=refined_document,
            note_document=note_document,
            action_by_note_id=action_by_note_id,
            warnings=warnings,
        )
    elif aligned_document is not None:
        timing_source = "audio_aligned_seconds"
        export_notes, action_by_export_note_id = _build_aligned_export_notes(
            note_document=note_document,
            aligned_document=aligned_document,
            action_by_note_id=action_by_note_id,
            warnings=warnings,
        )
    else:
        timing_source = "original_midi_ticks"
        export_notes, action_by_export_note_id = _build_original_export_notes(
            note_document=note_document,
            action_by_note_id=action_by_note_id,
            warnings=warnings,
        )

    working_notes = [
        note
        for note in export_notes
        if action_by_export_note_id.get(note.note_id) in {"KEEP", "REVIEW"}
    ]
    rejected_notes = [
        note
        for note in export_notes
        if action_by_export_note_id.get(note.note_id) in {"MUTE", "DELETE_CANDIDATE"}
    ]
    diagnostic_notes = list(export_notes)

    output_dir.mkdir(parents=True, exist_ok=True)

    export_specs: list[tuple[str, str, list[_ExportNote], list[str], bool]] = [
        (
            "WORKING",
            params.working_filename,
            working_notes,
            ["KEEP", "REVIEW"],
            True,
        ),
        (
            "REJECTED",
            params.rejected_filename,
            rejected_notes,
            ["DELETE_CANDIDATE", "MUTE"],
            True,
        ),
        (
            "DIAGNOSTIC",
            params.diagnostic_filename,
            diagnostic_notes,
            ["KEEP", "REVIEW", "MUTE", "DELETE_CANDIDATE"],
            params.include_diagnostic,
        ),
    ]

    exported_files: list[WorkingMidiExportFile] = []
    export_time_errors_ms: list[float] = []
    for role, filename, notes, included_actions, should_export in export_specs:
        if not should_export:
            continue

        output_path = output_dir / filename
        if not notes and not params.write_empty_files:
            warnings.append(f"Skipped empty MIDI file for role {role.lower()}: {output_path}")
            continue

        note_events: list[tuple[_ExportNote, int, int]] = []
        for note in notes:
            start_tick, end_tick, errors_ms = _resolve_timing(note=note, ticks_per_second=ticks_per_second)
            export_time_errors_ms.extend(errors_ms)
            note_events.append((note, start_tick, end_tick))

        track_name = f"{params.track_name_prefix} {role} {note_document.layer}"
        _write_midi_track(
            note_events=note_events,
            output_path=output_path,
            ticks_per_beat=exported_ticks_per_beat,
            track_name=track_name,
            tempo_us_per_beat=tempo_us_per_beat,
        )
        exported_files.append(
            WorkingMidiExportFile(
                role=role,
                path=str(output_path),
                note_count=len(notes),
                included_plan_actions=included_actions,
            )
        )

    if export_time_errors_ms:
        mean_export_time_error_ms = float(sum(export_time_errors_ms) / len(export_time_errors_ms))
        max_export_time_error_ms = float(max(export_time_errors_ms))
    else:
        mean_export_time_error_ms = 0.0
        max_export_time_error_ms = 0.0

    return WorkingMidiExportReport(
        notes_file=str(notes_file),
        cleanup_plan_file=str(cleanup_plan_file),
        refined_notes_file=(str(params.refined_notes_file) if params.refined_notes_file is not None else None),
        repair_plan_file=(str(params.repair_plan_file) if params.repair_plan_file is not None else None),
        audio_aligned_notes_file=(
            str(params.audio_aligned_notes_file)
            if params.audio_aligned_notes_file is not None
            else None
        ),
        status="ok",
        layer=note_document.layer,
        ticks_per_beat=exported_ticks_per_beat,
        ticks_per_beat_source=ticks_per_beat_source,
        timing_source=timing_source,
        max_export_time_error_ms=max_export_time_error_ms,
        mean_export_time_error_ms=mean_export_time_error_ms,
        source_ticks_per_beat=source_ticks_per_beat,
        exported_ticks_per_beat=exported_ticks_per_beat,
        tempo_us_per_beat=tempo_us_per_beat,
        bpm=float(60_000_000.0 / tempo_us_per_beat),
        working_note_count=len(working_notes),
        rejected_note_count=len(rejected_notes),
        diagnostic_note_count=len(diagnostic_notes) if params.include_diagnostic else 0,
        repair_extend_count=int(params.repair_extend_count),
        repair_shorten_count=int(params.repair_shorten_count),
        repair_insert_missing_count=int(params.repair_insert_missing_count),
        repair_split_count=int(params.repair_split_count),
        repair_close_gap_count=int(params.repair_close_gap_count),
        repair_review_manual_count=int(params.repair_review_manual_count),
        exported_files=exported_files,
        warning_count=len(warnings),
        warnings=warnings,
    )