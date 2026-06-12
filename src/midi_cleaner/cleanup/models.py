from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CleanupAction(BaseModel):
    note_id: str
    original_recommended_action: str
    plan_action: Literal["KEEP", "REVIEW", "MUTE", "DELETE_CANDIDATE"]
    confidence: float
    reasons: list[str]
    source_validation: dict[str, object]
    metadata: dict[str, object] = Field(default_factory=dict)


class CleanupPlanDocument(BaseModel):
    schema_version: str
    validation_file: str
    layer: str
    planner_parameters: dict[str, float | bool]
    actions: list[CleanupAction]


class CleanupPlanReport(BaseModel):
    validation_file: str
    status: Literal["ok", "error"]
    layer: str
    action_count: int
    keep_count: int
    review_count: int
    mute_count: int
    delete_candidate_count: int
    warning_count: int
    warnings: list[str]
    output_file: str | None
