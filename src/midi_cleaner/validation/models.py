from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class NoteValidation(BaseModel):
    note_id: str
    pitch_midi: int
    pitch_name: str
    layer: str
    source: str
    start_sec: float
    end_sec: float
    duration_sec: float
    nearest_onset_sec: float | None
    onset_error_ms: float | None
    onset_score: float
    max_rms_during_note: float
    mean_rms_during_note: float
    sustained_energy_ratio: float
    energy_match_score: float
    duration_match_score: float
    confidence: float
    recommended_action: str
    reasons: list[str]


class NoteValidationDocument(BaseModel):
    schema_version: str
    notes_file: str
    audio_features_file: str
    layer: str
    validation_parameters: dict[str, float]
    validations: list[NoteValidation]


class MidiAudioValidationReport(BaseModel):
    notes_file: str
    audio_features_file: str
    timing_source: Literal["audio_aligned_seconds", "original_note_events_seconds"] = (
        "original_note_events_seconds"
    )
    audio_aligned_notes_file: str | None = None
    status: Literal["ok", "error"]
    layer: str
    note_count: int
    keep_count: int
    review_count: int
    mute_candidate_count: int
    mean_confidence: float
    warning_count: int
    warnings: list[str]
    output_file: str | None
