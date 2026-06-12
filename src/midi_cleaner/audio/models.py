from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class AudioFrameFeature(BaseModel):
    frame_index: int
    start_sec: float
    end_sec: float
    rms: float
    peak: float
    zero_crossing_rate: float
    spectral_centroid_hz: float | None
    spectral_rolloff_hz: float | None
    is_silent: bool
    onset_score: float


class AudioGlobalFeatures(BaseModel):
    peak: float
    rms: float
    duration_sec: float
    estimated_silence_ratio: float
    frame_count: int
    onset_count: int
    mean_spectral_centroid_hz: float | None
    mean_spectral_rolloff_hz: float | None


class AudioFeatureDocument(BaseModel):
    schema_version: str
    source_file: str
    layer: str
    sample_rate: int
    channels: int
    duration_sec: float
    frame_size: int
    hop_size: int
    frames: list[AudioFrameFeature]
    global_features: AudioGlobalFeatures


class AudioAnalysisReport(BaseModel):
    input_file: str
    layer: str
    status: Literal["ok", "error"]
    sample_rate: int | None
    channels: int | None
    duration_sec: float | None
    frame_count: int
    onset_count: int
    warning_count: int
    warnings: list[str]
    output_file: str | None
