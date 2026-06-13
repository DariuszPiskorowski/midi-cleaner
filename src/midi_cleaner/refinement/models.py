from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RefinedNoteEvent(BaseModel):
    note_id: str
    source: str
    layer: str
    pitch_midi: int
    pitch_name: str
    velocity: int
    channel: int | None
    original_start_sec: float
    original_end_sec: float
    aligned_start_sec: float
    aligned_end_sec: float
    refined_start_sec: float
    refined_end_sec: float
    refined_duration_sec: float
    start_refinement_ms: float
    end_refinement_ms: float
    merged_note_ids: list[str] = Field(default_factory=list)
    refinement_actions: list[str] = Field(default_factory=list)
    refinement_confidence: float
    reasons: list[str] = Field(default_factory=list)


class RefinedNoteDocument(BaseModel):
    schema_version: str
    aligned_notes_file: str
    audio_features_file: str
    validation_file: str
    layer: str
    sample_rate: int
    audio_duration_sec: float
    timing_source: Literal["refined_audio_seconds"] = "refined_audio_seconds"
    refinement_parameters: dict[str, float | bool]
    notes: list[RefinedNoteEvent]


class BassRefinementReport(BaseModel):
    aligned_notes_file: str
    audio_features_file: str
    validation_file: str
    status: Literal["ok", "error"]
    layer: str
    input_note_count: int
    output_note_count: int
    merged_count: int
    false_retrigger_merge_count: int
    tail_extended_count: int
    short_note_extended_count: int
    overlap_resolved_count: int
    median_start_refinement_ms: float | None
    median_end_refinement_ms: float | None
    max_tail_extension_ms: float
    warning_count: int
    warnings: list[str]
    output_file: str | None