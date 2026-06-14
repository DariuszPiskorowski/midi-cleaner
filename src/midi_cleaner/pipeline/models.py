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
    refined_start_sec: float | None = None
    refined_end_sec: float | None = None
    start_refinement_ms: float | None = None
    end_refinement_ms: float | None = None
    refinement_actions: str | None = None
    merged_note_ids: str | None = None
    repair_actions: str | None = None
    repair_reason_summary: str | None = None
    was_inserted_by_repair: bool = False
    was_split_by_repair: bool = False
    was_extended_by_repair: bool = False
    was_shortened_by_repair: bool = False
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
    refined_note_count: int = 0
    merged_count: int = 0
    false_retrigger_merge_count: int = 0
    tail_extended_count: int = 0
    short_note_extended_count: int = 0
    overlap_resolved_count: int = 0
    median_start_refinement_ms: float | None = None
    median_end_refinement_ms: float | None = None
    dsp_backend_name: str | None = None
    dsp_backend_available: bool | None = None
    dsp_frame_count: int = 0
    dsp_attack_rise_count: int = 0
    dsp_sustain_count: int = 0
    dsp_tail_count: int = 0
    dsp_silence_count: int = 0
    dsp_debug_csv_file: str | None = None
    activity_repair_enabled: bool = False
    repaired_note_count: int = 0
    repair_extend_count: int = 0
    repair_shorten_count: int = 0
    repair_insert_missing_count: int = 0
    repair_split_count: int = 0
    repair_close_gap_count: int = 0
    repair_review_manual_count: int = 0
    audio_active_region_count: int = 0
    midi_active_region_count: int = 0
    audio_gap_count: int = 0
    midi_overhang_count: int = 0
    working_midi_note_count: int = 0
    working_export_time_error_ms: float | None = None
    validation_timing_source: str | None
    review_export_timing_source: str | None
    cleaned_export_timing_source: str | None
    global_offset_ms: float | None
    global_confidence: float | None
    global_offset_applied: bool | None
    aligned_count: int
    keep_original_count: int
    review_timing_count: int
    no_audio_evidence_count: int
    median_abs_start_correction_ms: float | None
    p95_abs_start_correction_ms: float | None
    max_abs_start_correction_ms: float | None
    max_export_time_error_ms: float | None
    mean_export_time_error_ms: float | None
    mean_confidence: float | None
    min_confidence: float | None
    max_confidence: float | None
    low_confidence_count: int
    weak_onset_count: int
    low_rms_count: int
    output_files: dict[str, str]
    warning_count: int
    warnings: list[str]
