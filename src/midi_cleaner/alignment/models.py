from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AudioAlignedNoteEvent(BaseModel):
    note_id: str
    source: str
    layer: str
    pitch_midi: int
    pitch_name: str
    velocity: int
    channel: int | None

    original_start_sec: float
    original_end_sec: float
    original_duration_sec: float
    original_start_tick: int
    original_end_tick: int

    aligned_start_sec: float
    aligned_end_sec: float
    aligned_duration_sec: float

    start_correction_ms: float
    end_correction_ms: float
    duration_correction_ms: float

    nearest_audio_onset_sec: float | None
    nearest_audio_offset_sec: float | None
    onset_error_before_ms: float | None
    onset_error_after_ms: float | None
    local_rms: float
    local_onset_score: float
    sustained_energy_ratio: float
    alignment_confidence: float
    alignment_action: Literal[
        "ALIGNED",
        "KEEP_ORIGINAL_LOW_CONFIDENCE",
        "REVIEW_TIMING",
        "NO_AUDIO_EVIDENCE",
    ]
    reasons: list[str] = Field(default_factory=list)


class AudioAlignedNoteDocument(BaseModel):
    schema_version: str
    notes_file: str
    audio_features_file: str
    layer: str
    sample_rate: int
    audio_duration_sec: float
    alignment_parameters: dict[str, float | bool | str]
    notes: list[AudioAlignedNoteEvent]


class AudioAlignmentReport(BaseModel):
    notes_file: str
    audio_features_file: str
    status: Literal["ok", "error"]
    layer: str
    note_count: int
    aligned_count: int
    keep_original_count: int
    review_timing_count: int
    no_audio_evidence_count: int
    median_abs_start_correction_ms: float | None
    p95_abs_start_correction_ms: float | None
    max_abs_start_correction_ms: float | None
    warning_count: int
    warnings: list[str]
    output_file: str | None
