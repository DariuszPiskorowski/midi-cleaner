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
    cleanup_plan_file: str
    layer: str
    actions: list[RepairAction]


class ActivityRepairReport(BaseModel):
    refined_notes_file: str
    audio_features_file: str
    dsp_features_file: str | None
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
    audio_active_region_count: int
    midi_active_region_count: int
    audio_gap_count: int
    midi_overhang_count: int
    warning_count: int
    warnings: list[str]
    output_file: str | None
    plan_file: str | None
