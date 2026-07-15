from __future__ import annotations

from pydantic import BaseModel, Field

from midi_cleaner.midi.models import TempoEvent


class SplitNote(BaseModel):
    note_id: str
    source_track_index: int
    source_track_name: str | None
    editable_track_index: int
    editable_track_name: str
    channel: int | None
    pitch_midi: int
    pitch_name: str
    velocity: int
    start_tick: int
    end_tick: int
    duration_ticks: int
    start_sec: float
    end_sec: float
    duration_sec: float
    muted: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)


class SplitTrack(BaseModel):
    editable_track_index: int
    name: str
    source_track_indices: list[int] = Field(default_factory=list)
    muted: bool | None = None
    color: str | None = None


class MidiSplitSession(BaseModel):
    schema_version: str
    source_midi: str
    source: str = "manual"
    layer: str = "midi"
    ticks_per_beat: int
    tempo_map: list[TempoEvent] = Field(default_factory=list)
    tracks: list[SplitTrack] = Field(default_factory=list)
    notes: list[SplitNote] = Field(default_factory=list)
