from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PipelineStageReport(BaseModel):
    name: str
    status: Literal["ok", "error"]
    output_files: list[str]
    warning_count: int
    warnings: list[str]


class PipelineReport(BaseModel):
    status: Literal["ok", "error"]
    input_midi: str
    input_wav: str
    source: str
    layer: str
    project_dir: str
    stages: list[PipelineStageReport]
    output_files: dict[str, str]
    warning_count: int
    warnings: list[str]


class QANoteRow(BaseModel):
    note_id: str
    pitch_midi: int
    pitch_name: str
    start_sec: float
    end_sec: float
    duration_sec: float
    original_start_sec: float | None
    aligned_start_sec: float | None
    start_correction_ms: float | None
    alignment_action: str | None
    alignment_confidence: float | None
    confidence: float
    validation_action: str
    plan_action: str | None
    onset_score: float
    mean_rms_during_note: float
    sustained_energy_ratio: float
    reasons: str


class QASummary(BaseModel):
    status: Literal["ok", "error"]
    project_dir: str
    layer: str | None
    total_notes: int
    keep_count: int
    review_count: int
    mute_count: int
    delete_candidate_count: int
    cleaned_note_count: int
    rejected_note_count: int
    aligned_count: int
    keep_original_count: int
    review_timing_count: int
    no_audio_evidence_count: int
    median_abs_start_correction_ms: float | None
    p95_abs_start_correction_ms: float | None
    max_abs_start_correction_ms: float | None
    mean_confidence: float | None
    min_confidence: float | None
    max_confidence: float | None
    low_confidence_count: int
    weak_onset_count: int
    low_rms_count: int
    output_files: dict[str, str]
    warning_count: int
    warnings: list[str]
