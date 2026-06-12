from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from midi_cleaner.alignment.models import AudioAlignedNoteDocument, AudioAlignedNoteEvent
from midi_cleaner.cleanup.models import (
    CleanedMidiExportFile,
    CleanedMidiExportReport,
    CleanupPlanDocument,
)
from midi_cleaner.midi.models import NoteEvent, NoteEventDocument

DEFAULT_TEMPO_US_PER_BEAT = 500000


class CleanedMidiExportError(Exception):
    """Raised when cleaned MIDI export cannot be completed."""


@dataclass(frozen=True)
class CleanedMidiExportParameters:
    ticks_per_beat: int = 960
    track_name_prefix: str = "Hermes"
    include_review_in_cleaned: bool = False
    write_empty_files: bool = True
    audio_aligned_notes_file: Path | None = None


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


def _load_audio_aligned_document(path: Path) -> AudioAlignedNoteDocument:
    try:
        return AudioAlignedNoteDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise CleanedMidiExportError(f"Invalid audio aligned notes JSON: {path}") from exc


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


def _absolute_events(note: NoteEvent, start_tick: int, end_tick: int) -> list[tuple[int, int, mido.Message]]:
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
    note_events: list[tuple[NoteEvent, int, int]],
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

    absolute_events: list[tuple[int, int, mido.Message]] = []
    for note, start_tick, end_tick in note_events:
        absolute_events.extend(_absolute_events(note, start_tick=start_tick, end_tick=end_tick))

    absolute_events.sort(key=lambda item: (item[0], item[1]))

    previous_tick = 0
    for tick, _order, message in absolute_events:
        delta = tick - previous_tick
        previous_tick = tick
        message.time = delta
        track.append(message)

    midi_file.save(str(output_path))


def _resolve_timing(
    note: NoteEvent,
    aligned_by_note_id: dict[str, AudioAlignedNoteEvent],
    use_audio_aligned_seconds: bool,
    ticks_per_second: float,
    warnings: list[str],
) -> tuple[int, int, list[float]]:
    if use_audio_aligned_seconds:
        aligned = aligned_by_note_id.get(note.note_id)
        if aligned is not None:
            start_sec = max(0.0, float(aligned.aligned_start_sec))
            end_sec = max(start_sec, float(aligned.aligned_end_sec))

            start_tick = _seconds_to_tick(start_sec, ticks_per_second)
            end_tick = _seconds_to_tick(end_sec, ticks_per_second)
            if end_tick <= start_tick:
                end_tick = start_tick + 1

            start_error_ms = abs(_tick_to_seconds(start_tick, ticks_per_second) - start_sec) * 1000.0
            end_error_ms = abs(_tick_to_seconds(end_tick, ticks_per_second) - end_sec) * 1000.0
            return start_tick, end_tick, [start_error_ms, end_error_ms]

        warnings.append(f"Audio alignment missing for note_id: {note.note_id}; used original MIDI ticks")

    start_tick = int(note.start_tick)
    end_tick = int(note.end_tick)
    if end_tick < start_tick:
        end_tick = start_tick
    return start_tick, end_tick, []


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

    aligned_document: AudioAlignedNoteDocument | None = None
    if params.audio_aligned_notes_file is not None:
        if not params.audio_aligned_notes_file.exists() or not params.audio_aligned_notes_file.is_file():
            raise CleanedMidiExportError(
                f"Audio aligned notes file does not exist: {params.audio_aligned_notes_file}"
            )
        aligned_document = _load_audio_aligned_document(params.audio_aligned_notes_file)

    warnings: list[str] = []
    if cleanup_plan.layer != note_document.layer:
        warnings.append(
            "Layer mismatch between note events and cleanup plan: "
            f"{note_document.layer} vs {cleanup_plan.layer}."
        )
    if aligned_document is not None and aligned_document.layer != note_document.layer:
        warnings.append(
            "Layer mismatch between note events and audio alignment: "
            f"{note_document.layer} vs {aligned_document.layer}."
        )

    use_audio_aligned_seconds = aligned_document is not None
    aligned_by_note_id = (
        {item.note_id: item for item in aligned_document.notes}
        if aligned_document is not None
        else {}
    )

    source_ticks_per_beat = int(note_document.ticks_per_beat)
    exported_ticks_per_beat = int(params.ticks_per_beat)
    if exported_ticks_per_beat <= 0:
        exported_ticks_per_beat = source_ticks_per_beat

    tempo_us_per_beat = _tempo_us_per_beat_from_document(note_document)
    ticks_per_second = _ticks_per_second(exported_ticks_per_beat, tempo_us_per_beat)

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
    export_time_errors_ms: list[float] = []
    for role, filename, notes, included_actions in export_specs:
        output_path = output_dir / filename
        if not notes and not params.write_empty_files:
            warnings.append(f"Skipped empty MIDI file for role {role.lower()}: {output_path}")
            continue

        note_events: list[tuple[NoteEvent, int, int]] = []
        for note in notes:
            start_tick, end_tick, errors_ms = _resolve_timing(
                note=note,
                aligned_by_note_id=aligned_by_note_id,
                use_audio_aligned_seconds=use_audio_aligned_seconds,
                ticks_per_second=ticks_per_second,
                warnings=warnings,
            )
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
            CleanedMidiExportFile(
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

    return CleanedMidiExportReport(
        notes_file=str(notes_file),
        cleanup_plan_file=str(cleanup_plan_file),
        audio_aligned_notes_file=(
            str(params.audio_aligned_notes_file)
            if params.audio_aligned_notes_file is not None
            else None
        ),
        status="ok",
        layer=note_document.layer,
        ticks_per_beat=exported_ticks_per_beat,
        timing_source=(
            "audio_aligned_seconds" if use_audio_aligned_seconds else "original_midi_ticks"
        ),
        max_export_time_error_ms=max_export_time_error_ms,
        mean_export_time_error_ms=mean_export_time_error_ms,
        source_ticks_per_beat=source_ticks_per_beat,
        exported_ticks_per_beat=exported_ticks_per_beat,
        tempo_us_per_beat=tempo_us_per_beat,
        bpm=float(60_000_000.0 / tempo_us_per_beat),
        cleaned_note_count=len(cleaned_notes),
        review_note_count=len(review_notes),
        rejected_note_count=len(rejected_notes),
        exported_files=exported_files,
        warning_count=len(warnings),
        warnings=warnings,
    )
