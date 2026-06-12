from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict, deque
from pathlib import Path

import mido

from midi_cleaner.midi.models import (
    MidiImportReport,
    NoteEvent,
    NoteEventDocument,
    TempoEvent,
)

DEFAULT_TEMPO_US_PER_BEAT = 500000
SCHEMA_VERSION = "0.1.0"


class MidiImportError(Exception):
    """Raised when a MIDI candidate cannot be imported."""


def pitch_name_from_midi(note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (note // 12) - 1
    return f"{names[note % 12]}{octave}"


def _normalize_tempo_events(
    ticks_per_beat: int,
    tempo_events: list[tuple[int, int, int, int]],
) -> list[TempoEvent]:
    sorted_events = sorted(tempo_events, key=lambda item: (item[0], item[1], item[2]))
    tick_to_tempo: dict[int, int] = {}
    for tick, _track_index, _order, tempo_us in sorted_events:
        tick_to_tempo[tick] = tempo_us

    if 0 not in tick_to_tempo:
        tick_to_tempo[0] = DEFAULT_TEMPO_US_PER_BEAT

    events = sorted(tick_to_tempo.items(), key=lambda item: item[0])

    tempo_map: list[TempoEvent] = []
    current_sec = 0.0
    for index, (tick, tempo_us_per_beat) in enumerate(events):
        if index > 0:
            prev_tick, prev_tempo = events[index - 1]
            delta_ticks = tick - prev_tick
            current_sec += (delta_ticks / ticks_per_beat) * (prev_tempo / 1_000_000)

        tempo_map.append(
            TempoEvent(
                tick=tick,
                tempo_us_per_beat=tempo_us_per_beat,
                sec=current_sec,
            )
        )

    return tempo_map


def tick_to_seconds(tick: int, tempo_map: list[TempoEvent], ticks_per_beat: int) -> float:
    ticks = [event.tick for event in tempo_map]
    index = bisect_right(ticks, tick) - 1
    if index < 0:
        index = 0
    event = tempo_map[index]
    delta_ticks = tick - event.tick
    delta_sec = (delta_ticks / ticks_per_beat) * (event.tempo_us_per_beat / 1_000_000)
    return event.sec + delta_sec


def _build_note_id(
    layer: str,
    track_index: int,
    channel: int | None,
    pitch: int,
    counters: dict[tuple[int, int | None, int], int],
) -> str:
    key = (track_index, channel, pitch)
    counters[key] += 1
    ordinal = counters[key]
    channel_token = "NA" if channel is None else f"{channel:02d}"
    return (
        f"{layer}_t{track_index + 1:02d}_ch{channel_token}_"
        f"p{pitch:03d}_n{ordinal:06d}"
    )


def import_midi_candidate(
    input_midi: Path,
    source: str,
    layer: str,
) -> tuple[NoteEventDocument, MidiImportReport]:
    if not input_midi.exists() or not input_midi.is_file():
        raise MidiImportError(f"Input MIDI file does not exist: {input_midi}")

    try:
        midi_file = mido.MidiFile(str(input_midi))
    except Exception as exc:  # pragma: no cover - library exception type varies
        raise MidiImportError(f"Failed to parse MIDI file: {input_midi}") from exc

    ticks_per_beat = midi_file.ticks_per_beat
    warnings: list[str] = []

    tempo_events_raw: list[tuple[int, int, int, int]] = []
    for track_index, track in enumerate(midi_file.tracks):
        absolute_tick = 0
        for order, message in enumerate(track):
            absolute_tick += message.time
            if message.type == "set_tempo":
                tempo_events_raw.append((absolute_tick, track_index, order, int(message.tempo)))

    tempo_map = _normalize_tempo_events(ticks_per_beat, tempo_events_raw)

    note_counters: dict[tuple[int, int | None, int], int] = defaultdict(int)
    notes: list[NoteEvent] = []

    for track_index, track in enumerate(midi_file.tracks):
        absolute_tick = 0
        track_name: str | None = None
        active_notes: dict[tuple[int | None, int], deque[tuple[int, int]]] = defaultdict(deque)

        for message in track:
            absolute_tick += message.time
            if message.type == "track_name" and track_name is None:
                track_name = message.name

            is_note_on = message.type == "note_on" and message.velocity > 0
            is_note_off = message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            )

            if is_note_on:
                key = (getattr(message, "channel", None), int(message.note))
                active_notes[key].append((absolute_tick, int(message.velocity)))
                continue

            if is_note_off:
                key = (getattr(message, "channel", None), int(message.note))
                if not active_notes[key]:
                    warnings.append(
                        "Unmatched note_off encountered on "
                        f"track {track_index}, note {message.note}."
                    )
                    continue

                start_tick, velocity = active_notes[key].popleft()
                end_tick = absolute_tick
                duration_ticks = end_tick - start_tick
                channel = getattr(message, "channel", None)
                note_id = _build_note_id(
                    layer=layer,
                    track_index=track_index,
                    channel=channel,
                    pitch=int(message.note),
                    counters=note_counters,
                )

                start_sec = tick_to_seconds(start_tick, tempo_map, ticks_per_beat)
                end_sec = tick_to_seconds(end_tick, tempo_map, ticks_per_beat)

                notes.append(
                    NoteEvent(
                        note_id=note_id,
                        source=source,
                        layer=layer,
                        track_index=track_index,
                        track_name=track_name,
                        channel=channel,
                        pitch_midi=int(message.note),
                        pitch_name=pitch_name_from_midi(int(message.note)),
                        velocity=velocity,
                        start_tick=start_tick,
                        end_tick=end_tick,
                        duration_ticks=duration_ticks,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        duration_sec=end_sec - start_sec,
                    )
                )

        for (channel, pitch), queue in active_notes.items():
            if queue:
                warnings.append(
                    "Unclosed note_on encountered on "
                    f"track {track_index}, channel {channel}, note {pitch}."
                )

    report = MidiImportReport(
        input_file=str(input_midi),
        source=source,
        layer=layer,
        status="ok",
        ticks_per_beat=ticks_per_beat,
        track_count=len(midi_file.tracks),
        note_count=len(notes),
        tempo_event_count=len(tempo_map),
        warning_count=len(warnings),
        warnings=warnings,
        output_file=None,
    )

    document = NoteEventDocument(
        schema_version=SCHEMA_VERSION,
        source_file=str(input_midi),
        source=source,
        layer=layer,
        ticks_per_beat=ticks_per_beat,
        tempo_map=tempo_map,
        notes=notes,
    )

    return document, report
