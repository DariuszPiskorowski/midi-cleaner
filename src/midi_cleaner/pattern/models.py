from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PatternBlockNote(BaseModel):
    note_id: str
    start_tick: int | None = None
    end_tick: int | None = None
    start_sec: float
    end_sec: float
    duration_sec: float
    start_beat: float | None = None
    end_beat: float | None = None
    duration_beat: float | None = None
    onset_slot: int | None = None
    duration_slots: int | None = None
    pitch_midi: int
    pitch_name: str
    velocity: int
    channel: int | None


class PatternBlock(BaseModel):
    block_id: str
    bar_index: int
    start_beat: float
    end_beat: float
    block_length_beats: float
    start_sec: float
    end_sec: float
    duration_sec: float
    time_signature: str = "4/4"
    grid_resolution: str = "1/16"
    onset_slots: list[int] = Field(default_factory=list)
    occupied_slots: list[int] = Field(default_factory=list)
    empty_slots: list[int] = Field(default_factory=list)
    note_count: int
    notes: list[PatternBlockNote] = Field(default_factory=list)
    relative_onsets_beat: list[float] = Field(default_factory=list)
    relative_durations_beat: list[float] = Field(default_factory=list)
    relative_onsets_sec: list[float] = Field(default_factory=list)
    relative_durations_sec: list[float] = Field(default_factory=list)
    pitch_sequence: list[int] = Field(default_factory=list)
    pitch_names: list[str] = Field(default_factory=list)
    interval_sequence: list[int] = Field(default_factory=list)
    rhythm_signature: list[float] = Field(default_factory=list)
    pitch_set: list[int] = Field(default_factory=list)
    assigned_pattern_family_id: str | None
    status: Literal["complete", "incomplete", "empty", "unknown"]


class PatternFamily(BaseModel):
    pattern_family_id: str
    block_length_beats: float
    time_signature: str = "4/4"
    grid_resolution: str = "1/16"
    representative_onset_slots: list[int] = Field(default_factory=list)
    representative_duration_slots: list[int] = Field(default_factory=list)
    representative_relative_onsets_beat: list[float] = Field(default_factory=list)
    representative_relative_durations_beat: list[float] = Field(default_factory=list)
    representative_pitch_sequence: list[int] = Field(default_factory=list)
    representative_interval_sequence: list[int] = Field(default_factory=list)
    representative_relative_onsets_sec: list[float] = Field(default_factory=list)
    representative_durations_sec: list[float] = Field(default_factory=list)
    representative_pitch_set: list[int] = Field(default_factory=list)
    representative_note_count: int
    occurrence_count: int
    occurrence_bars: list[int] = Field(default_factory=list)
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
    target_bar_index: int
    expected_pattern_family_id: str | None
    write_start_sec: float
    write_end_sec: float
    write_start_beat: float
    write_end_beat: float
    expected_duration_sec: float
    expected_duration_beat: float
    observed_slots: list[int] = Field(default_factory=list)
    missing_slots: list[int] = Field(default_factory=list)
    evidence_before_occurrences: list[str] = Field(default_factory=list)
    evidence_after_occurrences: list[str] = Field(default_factory=list)
    detected_note_count_in_region: int
    confidence_score: float
    possible_matches: list[IncompleteBlockMatch] = Field(default_factory=list)


class IncompleteBlockReport(BaseModel):
    block_type: Literal["incomplete_existing_block", "missing_expected_block"]
    incomplete_block_id: str | None = None
    missing_block_id: str | None = None
    target_bar_index: int | None = None
    target_start_sec: float | None = None
    target_end_sec: float | None = None
    expected_pattern_family_id: str | None = None
    source_pattern_family_id: str | None = None
    source_family_occurrence_count: int | None = None
    start_sec: float
    end_sec: float
    start_beat: float | None = None
    end_beat: float | None = None
    write_start_sec: float | None = None
    write_end_sec: float | None = None
    write_start_beat: float | None = None
    write_end_beat: float | None = None
    expected_duration_sec: float | None = None
    expected_duration_beat: float | None = None
    evidence_before_occurrences: list[str] = Field(default_factory=list)
    evidence_after_occurrences: list[str] = Field(default_factory=list)
    observed_note_count_in_region: int | None = None
    onset_slots_observed: list[int] = Field(default_factory=list)
    onset_slots_expected: list[int] = Field(default_factory=list)
    onset_slots_missing: list[int] = Field(default_factory=list)
    observed_slots: list[int] = Field(default_factory=list)
    missing_slots: list[int] = Field(default_factory=list)
    observed_pitch_sequence: list[int] = Field(default_factory=list)
    observed_relative_onsets_sec: list[float] = Field(default_factory=list)
    possible_matches: list[IncompleteBlockMatch] = Field(default_factory=list)
    best_match_pattern_family_id: str | None
    reason: str | None = None
    match_reason: str
    missing_notes_to_insert: list[ProposedCompletionNote] = Field(default_factory=list)
    inserted_notes: list[ProposedCompletionNote] = Field(default_factory=list)
    rejected_candidate_notes: list[dict[str, object]] = Field(default_factory=list)
    confidence_level: Literal["high", "medium", "low"]
    action: Literal["completed", "skipped"]


class PatternCompletionReport(BaseModel):
    status: Literal["ok", "error"]
    layer: str
    project_dir: str
    base_midi_path: str | None
    bar_aligned_block_count: int
    pattern_block_count: int
    complete_block_count: int
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
    rejected_micro_note_count: int
    rejected_polyphonic_stack_count: int
    rejected_low_confidence_count: int
    rejected_tiny_gap_count: int
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
