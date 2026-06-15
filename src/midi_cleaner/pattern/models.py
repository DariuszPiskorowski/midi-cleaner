from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PatternBlockNote(BaseModel):
    note_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    pitch_midi: int
    pitch_name: str
    velocity: int
    channel: int | None


class PatternBlock(BaseModel):
    block_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    note_count: int
    notes: list[PatternBlockNote] = Field(default_factory=list)
    relative_onsets_sec: list[float] = Field(default_factory=list)
    relative_durations_sec: list[float] = Field(default_factory=list)
    pitch_sequence: list[int] = Field(default_factory=list)
    pitch_names: list[str] = Field(default_factory=list)
    interval_sequence: list[int] = Field(default_factory=list)
    rhythm_signature: list[float] = Field(default_factory=list)
    pitch_set: list[int] = Field(default_factory=list)
    assigned_pattern_family_id: str | None
    status: Literal["complete", "incomplete", "unknown"]


class PatternFamily(BaseModel):
    pattern_family_id: str
    representative_pitch_sequence: list[int] = Field(default_factory=list)
    representative_interval_sequence: list[int] = Field(default_factory=list)
    representative_relative_onsets_sec: list[float] = Field(default_factory=list)
    representative_durations_sec: list[float] = Field(default_factory=list)
    representative_pitch_set: list[int] = Field(default_factory=list)
    occurrence_count: int
    occurrences: list[str] = Field(default_factory=list)
    first_seen_sec: float
    last_seen_sec: float


class ProposedCompletionNote(BaseModel):
    source_pattern_family_id: str
    source_block_id: str
    source_note_index: int
    note_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    pitch_midi: int
    pitch_name: str
    velocity: int
    channel: int | None


class IncompleteBlockMatch(BaseModel):
    pattern_family_id: str
    score: float
    reason: str


class IncompleteBlockReport(BaseModel):
    incomplete_block_id: str
    start_sec: float
    end_sec: float
    observed_pitch_sequence: list[int] = Field(default_factory=list)
    observed_relative_onsets_sec: list[float] = Field(default_factory=list)
    possible_matches: list[IncompleteBlockMatch] = Field(default_factory=list)
    best_match_pattern_family_id: str | None
    match_reason: str
    missing_notes_to_insert: list[ProposedCompletionNote] = Field(default_factory=list)
    confidence_level: Literal["high", "medium", "low"]
    action: Literal["completed", "skipped"]


class PatternCompletionReport(BaseModel):
    status: Literal["ok", "error"]
    layer: str
    project_dir: str
    base_midi_path: str | None
    pattern_block_count: int
    pattern_family_count: int
    incomplete_block_count: int
    completed_block_count: int
    skipped_block_count: int
    inserted_note_count: int
    output_midi_path: str | None
    pattern_blocks_file: str | None
    pattern_families_file: str | None
    incomplete_blocks_file: str | None
    debug_midi_path: str | None
    warnings: list[str] = Field(default_factory=list)
    warning_count: int = 0
    error: str | None = None
