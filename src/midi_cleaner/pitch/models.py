from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class BassPitchFrame(BaseModel):
    frame_index: int
    start_sec: float
    end_sec: float
    f0_hz: float | None
    pitch_midi_float: float | None
    pitch_midi_rounded: int | None
    pitch_confidence: float
    voiced: bool
    low_band_energy: float | None
    harmonic_energy: float | None


class BassPitchContourDocument(BaseModel):
    schema_version: str
    wav_file: str
    layer: str
    backend_name: str
    backend_available: bool
    sample_rate: int
    duration_sec: float
    hop_length: int
    frame_length: int
    min_hz: float
    max_hz: float
    frames: list[BassPitchFrame]


class BassPitchContourReport(BaseModel):
    wav_file: str
    status: Literal["ok", "error"]
    layer: str
    backend_name: str
    backend_available: bool
    frame_count: int
    voiced_frame_count: int
    voiced_ratio: float
    mean_pitch_confidence: float
    min_detected_hz: float | None
    max_detected_hz: float | None
    warning_count: int
    warnings: list[str]
    output_file: str | None
