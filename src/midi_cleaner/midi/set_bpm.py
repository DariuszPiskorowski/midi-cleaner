from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mido
from pydantic import BaseModel


class MidiSetBpmError(Exception):
    """Raised when MIDI BPM update fails."""


class MidiSetBpmReport(BaseModel):
    input_file: str
    output_file: str
    status: Literal["ok", "error"]
    bpm: float
    tempo_us_per_beat: int
    source_track_count: int
    output_track_count: int
    removed_tempo_event_count: int
    inserted_tempo_event_count: int
    warning_count: int
    warnings: list[str]


@dataclass(frozen=True)
class _AbsoluteEvent:
    tick: int
    original_index: int
    message: mido.Message | mido.MetaMessage


def _absolute_events(track: mido.MidiTrack) -> list[_AbsoluteEvent]:
    absolute_tick = 0
    events: list[_AbsoluteEvent] = []
    for index, message in enumerate(track):
        absolute_tick += int(message.time)
        events.append(
            _AbsoluteEvent(
                tick=absolute_tick,
                original_index=index,
                message=message.copy(time=0),
            )
        )
    return events


def _event_priority(message: mido.Message | mido.MetaMessage) -> int:
    if message.type == "track_name":
        return 0
    if message.type == "set_tempo":
        return 1
    if message.type == "end_of_track":
        return 3
    return 2


def _rebuild_track(events: list[_AbsoluteEvent]) -> mido.MidiTrack:
    ordered = sorted(
        events,
        key=lambda item: (item.tick, _event_priority(item.message), item.original_index),
    )

    track = mido.MidiTrack()
    previous_tick = 0
    for event in ordered:
        message = event.message.copy(time=max(0, event.tick - previous_tick))
        previous_tick = event.tick
        track.append(message)
    return track


def set_midi_bpm(input_file: Path, output_file: Path, bpm: float) -> MidiSetBpmReport:
    if not input_file.exists() or not input_file.is_file():
        raise MidiSetBpmError(f"Input MIDI file does not exist: {input_file}")
    if bpm <= 0:
        raise MidiSetBpmError("BPM must be greater than 0.")

    try:
        midi = mido.MidiFile(str(input_file))
    except Exception as exc:  # pragma: no cover - library exception type varies
        raise MidiSetBpmError(f"Failed to parse MIDI file: {input_file}") from exc

    output = mido.MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat)
    tempo_us_per_beat = int(round(60_000_000.0 / float(bpm)))

    warnings: list[str] = []
    removed_tempo_event_count = 0

    for track_index, track in enumerate(midi.tracks):
        absolute = _absolute_events(track)
        filtered: list[_AbsoluteEvent] = []

        for event in absolute:
            if event.message.type == "set_tempo":
                removed_tempo_event_count += 1
                continue
            filtered.append(event)

        if track_index == 0:
            filtered.append(
                _AbsoluteEvent(
                    tick=0,
                    original_index=-1,
                    message=mido.MetaMessage("set_tempo", tempo=tempo_us_per_beat, time=0),
                )
            )

        output.tracks.append(_rebuild_track(filtered))

    if not output.tracks:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("set_tempo", tempo=tempo_us_per_beat, time=0))
        output.tracks.append(track)
        warnings.append("Input MIDI had no tracks; created a single output track.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.save(str(output_file))

    return MidiSetBpmReport(
        input_file=str(input_file),
        output_file=str(output_file),
        status="ok",
        bpm=float(bpm),
        tempo_us_per_beat=tempo_us_per_beat,
        source_track_count=len(midi.tracks),
        output_track_count=len(output.tracks),
        removed_tempo_event_count=removed_tempo_event_count,
        inserted_tempo_event_count=1,
        warning_count=len(warnings),
        warnings=warnings,
    )
