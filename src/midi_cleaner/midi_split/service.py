from __future__ import annotations

from pathlib import Path

from midi_cleaner.midi.importer import MidiImportError, import_midi_candidate
from midi_cleaner.midi_split.models import MidiSplitSession, SplitNote, SplitTrack

SCHEMA_VERSION = "0.1.0"
DEFAULT_MAX_TRACKS = 12


class MidiSplitSessionError(Exception):
    """Raised when a MIDI split session operation cannot be completed."""


def _default_track_name(editable_track_index: int, source_track_name: str | None) -> str:
    if source_track_name is not None and source_track_name.strip():
        return source_track_name.strip()
    return f"Track {editable_track_index:02d}"


def _sorted_tracks(tracks: list[SplitTrack]) -> list[SplitTrack]:
    return sorted(tracks, key=lambda track: track.editable_track_index)


def _sorted_notes(notes: list[SplitNote]) -> list[SplitNote]:
    return sorted(
        notes,
        key=lambda note: (
            note.start_tick,
            note.end_tick,
            note.editable_track_index,
            note.channel if note.channel is not None else -1,
            note.pitch_midi,
            note.note_id,
        ),
    )


def _rebuild_track_sources(tracks: list[SplitTrack], notes: list[SplitNote]) -> list[SplitTrack]:
    source_by_track: dict[int, set[int]] = {track.editable_track_index: set() for track in tracks}
    for note in notes:
        source_by_track.setdefault(note.editable_track_index, set()).add(note.source_track_index)

    rebuilt: list[SplitTrack] = []
    for track in tracks:
        rebuilt.append(
            track.model_copy(
                update={
                    "source_track_indices": sorted(source_by_track.get(track.editable_track_index, set()))
                }
            )
        )

    return _sorted_tracks(rebuilt)


def create_split_session(
    input_midi: Path,
    *,
    source: str = "manual",
    layer: str = "midi",
) -> MidiSplitSession:
    try:
        document, _report = import_midi_candidate(input_midi=input_midi, source=source, layer=layer)
    except MidiImportError as exc:
        raise MidiSplitSessionError(f"Failed to create split session from MIDI: {input_midi}") from exc

    source_track_indices = sorted({int(note.track_index) for note in document.notes})
    source_track_names: dict[int, str | None] = {}
    for note in document.notes:
        if note.track_index not in source_track_names and note.track_name is not None:
            source_track_names[note.track_index] = note.track_name

    editable_track_by_source_track: dict[int, SplitTrack] = {}
    tracks: list[SplitTrack] = []
    for editable_index, source_track_index in enumerate(source_track_indices, start=1):
        track_name = _default_track_name(
            editable_track_index=editable_index,
            source_track_name=source_track_names.get(source_track_index),
        )
        track = SplitTrack(
            editable_track_index=editable_index,
            name=track_name,
            source_track_indices=[source_track_index],
        )
        editable_track_by_source_track[source_track_index] = track
        tracks.append(track)

    notes: list[SplitNote] = []
    for note in document.notes:
        editable_track = editable_track_by_source_track.get(int(note.track_index))
        if editable_track is None:
            raise MidiSplitSessionError(
                f"Cannot map source track_index={note.track_index} for note_id={note.note_id}"
            )

        notes.append(
            SplitNote(
                note_id=note.note_id,
                source_track_index=int(note.track_index),
                source_track_name=note.track_name,
                editable_track_index=editable_track.editable_track_index,
                editable_track_name=editable_track.name,
                channel=note.channel,
                pitch_midi=int(note.pitch_midi),
                pitch_name=note.pitch_name,
                velocity=int(note.velocity),
                start_tick=int(note.start_tick),
                end_tick=int(note.end_tick),
                duration_ticks=int(note.duration_ticks),
                start_sec=float(note.start_sec),
                end_sec=float(note.end_sec),
                duration_sec=float(note.duration_sec),
                metadata=dict(note.metadata),
            )
        )

    return MidiSplitSession(
        schema_version=SCHEMA_VERSION,
        source_midi=str(input_midi),
        source=source,
        layer=layer,
        ticks_per_beat=int(document.ticks_per_beat),
        tempo_map=list(document.tempo_map),
        tracks=_sorted_tracks(tracks),
        notes=_sorted_notes(notes),
    )


def add_empty_track(
    session: MidiSplitSession,
    name: str | None = None,
    *,
    max_tracks: int = DEFAULT_MAX_TRACKS,
) -> MidiSplitSession:
    if max_tracks <= 0:
        raise MidiSplitSessionError(f"Invalid max_tracks value: {max_tracks}")

    existing_track_indices = {int(track.editable_track_index) for track in session.tracks}
    if len(existing_track_indices) >= max_tracks:
        raise MidiSplitSessionError(f"Maximum editable track count reached ({max_tracks}).")

    next_track_index = 1
    while next_track_index in existing_track_indices:
        next_track_index += 1

    if next_track_index > max_tracks:
        raise MidiSplitSessionError(f"Maximum editable track count reached ({max_tracks}).")

    track_name = name.strip() if name is not None and name.strip() else f"Layer {next_track_index:02d}"

    updated_tracks = [
        *session.tracks,
        SplitTrack(
            editable_track_index=next_track_index,
            name=track_name,
            source_track_indices=[],
        ),
    ]

    return session.model_copy(update={"tracks": _sorted_tracks(updated_tracks)})


def move_notes_to_track(
    session: MidiSplitSession,
    note_ids: list[str],
    target_track_index: int,
) -> MidiSplitSession:
    target_track = next(
        (track for track in session.tracks if int(track.editable_track_index) == int(target_track_index)),
        None,
    )
    if target_track is None:
        raise MidiSplitSessionError(f"Target editable track does not exist: {target_track_index}")

    normalized_note_ids = {note_id.strip() for note_id in note_ids if note_id.strip()}
    if not normalized_note_ids:
        return session

    notes_by_id = {note.note_id: note for note in session.notes}
    missing_note_ids = sorted(normalized_note_ids - set(notes_by_id.keys()))
    if missing_note_ids:
        missing_label = ", ".join(missing_note_ids)
        raise MidiSplitSessionError(f"Unknown note_id values: {missing_label}")

    moved_notes: list[SplitNote] = []
    for note in session.notes:
        if note.note_id in normalized_note_ids:
            moved_notes.append(
                note.model_copy(
                    update={
                        "editable_track_index": int(target_track_index),
                        "editable_track_name": target_track.name,
                    }
                )
            )
        else:
            moved_notes.append(note)

    sorted_notes = _sorted_notes(moved_notes)
    rebuilt_tracks = _rebuild_track_sources(session.tracks, sorted_notes)

    return session.model_copy(update={"tracks": rebuilt_tracks, "notes": sorted_notes})


def merge_tracks(
    session: MidiSplitSession,
    editable_track_indices: list[int],
) -> MidiSplitSession:
    selected_track_indices = sorted({int(index) for index in editable_track_indices})
    if not selected_track_indices:
        return session

    tracks_by_index = {int(track.editable_track_index): track for track in session.tracks}
    missing = [index for index in selected_track_indices if index not in tracks_by_index]
    if missing:
        missing_label = ", ".join(str(index) for index in missing)
        raise MidiSplitSessionError(f"Cannot merge unknown editable track indices: {missing_label}")

    target_track_index = selected_track_indices[0]
    target_track = tracks_by_index[target_track_index]
    selected_set = set(selected_track_indices)

    merged_notes: list[SplitNote] = []
    for note in session.notes:
        if note.editable_track_index in selected_set and note.editable_track_index != target_track_index:
            merged_notes.append(
                note.model_copy(
                    update={
                        "editable_track_index": target_track_index,
                        "editable_track_name": target_track.name,
                    }
                )
            )
        else:
            merged_notes.append(note)

    remaining_tracks = [
        track
        for track in session.tracks
        if track.editable_track_index == target_track_index
        or track.editable_track_index not in selected_set
    ]

    sorted_notes = _sorted_notes(merged_notes)
    rebuilt_tracks = _rebuild_track_sources(remaining_tracks, sorted_notes)

    return session.model_copy(update={"tracks": rebuilt_tracks, "notes": sorted_notes})


def save_session(session: MidiSplitSession, output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(session.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_session(input_json: Path) -> MidiSplitSession:
    if not input_json.exists() or not input_json.is_file():
        raise MidiSplitSessionError(f"Split session file does not exist: {input_json}")

    try:
        return MidiSplitSession.model_validate_json(input_json.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise MidiSplitSessionError(f"Invalid split session JSON: {input_json}") from exc
