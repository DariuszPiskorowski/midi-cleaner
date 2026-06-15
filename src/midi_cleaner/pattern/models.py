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


class MissingExpectedBlock(BaseModel):
    missing_block_id: str
    expected_pattern_family_id: str | None
    write_start_sec: float
    write_end_sec: float
    expected_duration_sec: float
    evidence_before_occurrences: list[str] = Field(default_factory=list)
    evidence_after_occurrences: list[str] = Field(default_factory=list)
    detected_note_count_in_region: int
    confidence_score: float
    possible_matches: list[IncompleteBlockMatch] = Field(default_factory=list)


class IncompleteBlockReport(BaseModel):
    block_type: Literal["incomplete_existing_block", "missing_expected_block"]
    incomplete_block_id: str | None = None
    missing_block_id: str | None = None
    expected_pattern_family_id: str | None = None
    start_sec: float
    end_sec: float
    write_start_sec: float | None = None
    write_end_sec: float | None = None
    expected_duration_sec: float | None = None
    evidence_before_occurrences: list[str] = Field(default_factory=list)
    evidence_after_occurrences: list[str] = Field(default_factory=list)
    observed_note_count_in_region: int | None = None
    observed_pitch_sequence: list[int] = Field(default_factory=list)
    observed_relative_onsets_sec: list[float] = Field(default_factory=list)
    possible_matches: list[IncompleteBlockMatch] = Field(default_factory=list)
    best_match_pattern_family_id: str | None
    reason: str | None = None
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
    incomplete_existing_block_count: int
    missing_expected_block_count: int
    incomplete_block_count: int
    completed_incomplete_existing_block_count: int
    completed_missing_expected_block_count: int
    completed_block_count: int
    skipped_block_count: int
    skipped_ambiguous_count: int
    skipped_no_clear_family_count: int
    inserted_note_count: int
    output_midi_path: str | None
    pattern_blocks_file: str | None
    pattern_families_file: str | None
    incomplete_blocks_file: str | None
    missing_expected_blocks_file: str | None
    debug_midi_path: str | None
    warnings: list[str] = Field(default_factory=list)
    warning_count: int = 0
    error: str | None = None
