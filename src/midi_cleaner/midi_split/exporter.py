from __future__ import annotations

from pathlib import Path

import mido

from midi_cleaner.midi_split.models import MidiSplitSession, SplitNote, SplitTrack

DEFAULT_TEMPO_US_PER_BEAT = 500000


class MidiSplitExportError(Exception):
    """Raised when split MIDI export cannot be completed."""


def _normalized_tempo_map(session: MidiSplitSession) -> list[tuple[int, int]]:
    tick_to_tempo: dict[int, int] = {
        int(event.tick): int(event.tempo_us_per_beat) for event in session.tempo_map
    }
    if 0 not in tick_to_tempo:
        tick_to_tempo[0] = DEFAULT_TEMPO_US_PER_BEAT

    normalized = sorted(tick_to_tempo.items(), key=lambda item: item[0])
    return normalized


def _note_sort_key(note: SplitNote) -> tuple[int, int, int, int, int, str]:
    return (
        int(note.start_tick),
        int(note.end_tick),
        int(note.channel) if note.channel is not None else -1,
        int(note.pitch_midi),
        int(note.velocity),
        note.note_id,
    )


def _note_absolute_events(note: SplitNote) -> list[tuple[int, int, mido.Message]]:
    start_tick = int(note.start_tick)
    end_tick = max(start_tick, int(note.end_tick))
    channel = int(note.channel) if note.channel is not None else 0

    return [
        (
            start_tick,
            1,
            mido.Message(
                "note_on",
                channel=channel,
                note=int(note.pitch_midi),
                velocity=int(note.velocity),
                time=0,
            ),
        ),
        (
            end_tick,
            0,
            mido.Message(
                "note_off",
                channel=channel,
                note=int(note.pitch_midi),
                velocity=0,
                time=0,
            ),
        ),
    ]


def _append_sorted_absolute_events(
    track: mido.MidiTrack,
    absolute_events: list[tuple[int, int, mido.Message]],
) -> None:
    absolute_events.sort(
        key=lambda event: (
            int(event[0]),
            int(event[1]),
            getattr(event[2], "channel", -1),
            getattr(event[2], "note", -1),
            getattr(event[2], "velocity", -1),
        )
    )

    previous_tick = 0
    for tick, _order, message in absolute_events:
        delta = int(tick) - previous_tick
        previous_tick = int(tick)
        message.time = delta
        track.append(message)


def _sanitize_track_name(value: str) -> str:
    text = value.strip()
    if not text:
        return ""

    safe = []
    for char in text:
        if char.isalnum() or char in {"_", "-"}:
            safe.append(char)
        else:
            safe.append("_")

    collapsed = "".join(safe)
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")

    return collapsed.strip("_")


def _track_file_name(track: SplitTrack) -> str:
    safe_name = _sanitize_track_name(track.name)
    if safe_name:
        return f"{track.editable_track_index:02d}_{safe_name}.mid"
    return f"{track.editable_track_index:02d}_.mid"


def _notes_for_track(session: MidiSplitSession, editable_track_index: int) -> list[SplitNote]:
    notes = [
        note for note in session.notes if int(note.editable_track_index) == int(editable_track_index)
    ]
    notes.sort(key=_note_sort_key)
    return notes


def export_split_multitrack_midi(session: MidiSplitSession, output_midi: Path) -> None:
    output_midi.parent.mkdir(parents=True, exist_ok=True)

    midi_file = mido.MidiFile(type=1, ticks_per_beat=int(session.ticks_per_beat))

    conductor = mido.MidiTrack()
    midi_file.tracks.append(conductor)
    conductor.append(mido.MetaMessage("track_name", name="Conductor", time=0))

    conductor_events: list[tuple[int, int, mido.Message]] = []
    for tick, tempo in _normalized_tempo_map(session):
        conductor_events.append(
            (
                int(tick),
                -1,
                mido.MetaMessage("set_tempo", tempo=int(tempo), time=0),
            )
        )
    _append_sorted_absolute_events(conductor, conductor_events)

    for track in sorted(session.tracks, key=lambda item: int(item.editable_track_index)):
        midi_track = mido.MidiTrack()
        midi_file.tracks.append(midi_track)
        midi_track.append(mido.MetaMessage("track_name", name=track.name, time=0))

        absolute_events: list[tuple[int, int, mido.Message]] = []
        for note in _notes_for_track(session, track.editable_track_index):
            absolute_events.extend(_note_absolute_events(note))

        _append_sorted_absolute_events(midi_track, absolute_events)

    try:
        midi_file.save(str(output_midi))
    except Exception as exc:  # pragma: no cover - mido backend specific
        raise MidiSplitExportError(f"Failed to write multitrack split MIDI: {output_midi}") from exc


def export_split_separate_midi_files(
    session: MidiSplitSession,
    output_dir: Path,
    *,
    skip_empty: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files: list[Path] = []
    tempo_map = _normalized_tempo_map(session)

    for track in sorted(session.tracks, key=lambda item: int(item.editable_track_index)):
        notes = _notes_for_track(session, track.editable_track_index)
        if skip_empty and not notes:
            continue

        output_path = output_dir / _track_file_name(track)
        midi_file = mido.MidiFile(type=0, ticks_per_beat=int(session.ticks_per_beat))
        midi_track = mido.MidiTrack()
        midi_file.tracks.append(midi_track)

        absolute_events: list[tuple[int, int, mido.Message]] = [
            (0, -2, mido.MetaMessage("track_name", name=track.name, time=0))
        ]
        for tick, tempo in tempo_map:
            absolute_events.append(
                (
                    int(tick),
                    -1,
                    mido.MetaMessage("set_tempo", tempo=int(tempo), time=0),
                )
            )
        for note in notes:
            absolute_events.extend(_note_absolute_events(note))

        _append_sorted_absolute_events(midi_track, absolute_events)

        try:
            midi_file.save(str(output_path))
        except Exception as exc:  # pragma: no cover - mido backend specific
            raise MidiSplitExportError(f"Failed to write split track MIDI: {output_path}") from exc

        exported_files.append(output_path)

    return exported_files
