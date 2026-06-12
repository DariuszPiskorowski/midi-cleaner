from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TempoEvent(BaseModel):
    tick: int
    tempo_us_per_beat: int
    sec: float


class NoteEvent(BaseModel):
    note_id: str
    source: str
    layer: str
    track_index: int
    track_name: str | None
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
    action: str = "UNCLASSIFIED"
    confidence: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class NoteEventDocument(BaseModel):
    schema_version: str
    source_file: str
    source: str
    layer: str
    ticks_per_beat: int
    tempo_map: list[TempoEvent]
    notes: list[NoteEvent]


class MidiImportReport(BaseModel):
    input_file: str
    source: str
    layer: str
    status: Literal["ok", "error"]
    ticks_per_beat: int | None
    track_count: int
    note_count: int
    tempo_event_count: int
    warning_count: int
    warnings: list[str]
    output_file: str | None
