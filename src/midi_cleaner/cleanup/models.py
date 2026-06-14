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


class ReviewMidiExportFile(BaseModel):
    action: str
    path: str
    note_count: int


class ReviewMidiExportReport(BaseModel):
    notes_file: str
    cleanup_plan_file: str
    audio_aligned_notes_file: str | None
    status: Literal["ok", "error"]
    layer: str
    ticks_per_beat: int
    ticks_per_beat_source: Literal["auto_from_note_events", "user_override"] = (
        "auto_from_note_events"
    )
    timing_source: Literal[
        "refined_audio_seconds",
        "audio_aligned_seconds",
        "original_midi_ticks",
    ]
    max_export_time_error_ms: float
    mean_export_time_error_ms: float
    source_ticks_per_beat: int
    exported_ticks_per_beat: int
    tempo_us_per_beat: int
    bpm: float
    exported_files: list[ReviewMidiExportFile]
    warning_count: int
    warnings: list[str]


class CleanedMidiExportFile(BaseModel):
    role: str
    path: str
    note_count: int
    included_plan_actions: list[str]


class CleanedMidiExportReport(BaseModel):
    notes_file: str
    cleanup_plan_file: str
    audio_aligned_notes_file: str | None
    status: Literal["ok", "error"]
    layer: str
    ticks_per_beat: int
    ticks_per_beat_source: Literal["auto_from_note_events", "user_override"] = (
        "auto_from_note_events"
    )
    timing_source: Literal[
        "refined_audio_seconds",
        "audio_aligned_seconds",
        "original_midi_ticks",
    ]
    max_export_time_error_ms: float
    mean_export_time_error_ms: float
    source_ticks_per_beat: int
    exported_ticks_per_beat: int
    tempo_us_per_beat: int
    bpm: float
    cleaned_note_count: int
    review_note_count: int
    rejected_note_count: int
    exported_files: list[CleanedMidiExportFile]
    warning_count: int
    warnings: list[str]


class WorkingMidiExportFile(BaseModel):
    role: str
    path: str
    note_count: int
    included_plan_actions: list[str]


class WorkingMidiExportReport(BaseModel):
    notes_file: str
    cleanup_plan_file: str
    refined_notes_file: str | None
    repair_plan_file: str | None = None
    audio_aligned_notes_file: str | None
    status: Literal["ok", "error"]
    layer: str
    ticks_per_beat: int
    ticks_per_beat_source: Literal["auto_from_note_events", "user_override"] = (
        "auto_from_note_events"
    )
    timing_source: Literal[
        "refined_audio_seconds",
        "audio_aligned_seconds",
        "original_midi_ticks",
    ]
    max_export_time_error_ms: float
    mean_export_time_error_ms: float
    source_ticks_per_beat: int
    exported_ticks_per_beat: int
    tempo_us_per_beat: int
    bpm: float
    working_note_count: int
    rejected_note_count: int
    diagnostic_note_count: int
    repair_extend_count: int = 0
    repair_shorten_count: int = 0
    repair_insert_missing_count: int = 0
    repair_split_count: int = 0
    repair_close_gap_count: int = 0
    repair_review_manual_count: int = 0
    exported_files: list[WorkingMidiExportFile]
    warning_count: int
    warnings: list[str]
