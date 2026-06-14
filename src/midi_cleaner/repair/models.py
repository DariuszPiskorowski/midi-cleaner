from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RepairAction(BaseModel):
    action_id: str
    action_type: Literal[
        "EXTEND_NOTE",
        "SHORTEN_NOTE",
        "INSERT_MISSING_NOTE",
        "SPLIT_NOTE",
        "CLOSE_GAP",
        "KEEP",
        "REVIEW_MANUAL",
    ]
    target_note_id: str | None
    new_note_id: str | None
    start_sec: float
    end_sec: float
    old_start_sec: float | None
    old_end_sec: float | None
    new_start_sec: float | None
    new_end_sec: float | None
    pitch_midi: int | None
    confidence: float
    reasons: list[str] = Field(default_factory=list)
    evidence: dict[str, object] = Field(default_factory=dict)


class ActivityRepairPlan(BaseModel):
    schema_version: str
    refined_notes_file: str
    audio_features_file: str
    dsp_features_file: str | None
    pitch_contour_file: str | None
    cleanup_plan_file: str
    layer: str
    actions: list[RepairAction]


class ActivityRepairReport(BaseModel):
    refined_notes_file: str
    audio_features_file: str
    dsp_features_file: str | None
    pitch_contour_file: str | None
    cleanup_plan_file: str
    status: Literal["ok", "error"]
    layer: str
    input_note_count: int
    output_note_count: int
    extend_count: int
    shorten_count: int
    insert_missing_count: int
    split_count: int
    close_gap_count: int
    review_manual_count: int
    keep_count: int
    sustain_protected_count: int
    pitch_protected_count: int
    legato_protected_count: int
    shorten_candidate_count: int
    shorten_applied_count: int
    shorten_rejected_count: int
    audio_active_region_count: int
    midi_active_region_count: int
    audio_gap_count: int
    midi_overhang_count: int
    warning_count: int
    warnings: list[str]
    output_file: str | None
    plan_file: str | None


class RepairIterationSummary(BaseModel):
    iteration_index: int
    input_note_count: int
    output_note_count: int
    applied_action_count: int
    candidate_action_count: int
    extend_count: int
    shorten_count: int
    insert_count: int
    split_count: int
    merge_count: int
    close_gap_count: int
    protected_count: int
    review_manual_count: int
    audio_gap_count: int
    midi_overhang_count: int
    unresolved_error_count: int
    coverage_score: float
    overhang_score: float
    continuity_score: float
    pitch_consistency_score: float
    total_score: float
    improvement_from_previous: float
    stopped_reason: str | None


class IterativeRepairReport(BaseModel):
    status: Literal["ok", "error"]
    layer: str
    input_refined_notes_file: str
    final_repaired_notes_file: str | None
    iterations_requested: int
    iterations_completed: int
    convergence_reached: bool
    best_iteration_index: int
    final_score: float
    initial_score: float
    total_improvement: float
    warning_count: int
    warnings: list[str]
    iterations: list[RepairIterationSummary] = Field(default_factory=list)
    output_file: str | None


class IterationScoringReport(BaseModel):
    total_score: float
    coverage_score: float
    overhang_score: float
    continuity_score: float
    pitch_consistency_score: float
    unresolved_error_count: int
    audio_gap_count: int
    midi_overhang_count: int
    error_regions: list[dict[str, object]] = Field(default_factory=list)
