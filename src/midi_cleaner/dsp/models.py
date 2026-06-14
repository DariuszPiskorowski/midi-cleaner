from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DspAudioFrame(BaseModel):
    frame_index: int
    start_sec: float
    end_sec: float
    rms: float
    rms_smooth: float
    rms_delta: float
    envelope: float
    envelope_smooth: float
    envelope_delta: float
    low_band_rms: float
    low_band_envelope: float
    low_band_envelope_smooth: float
    low_band_delta: float
    spectral_flux: float
    onset_strength: float
    harmonic_rms: float
    percussive_rms: float
    is_attack_rise: bool
    is_sustain: bool
    is_tail: bool
    is_silence: bool


class DspAudioFeatureDocument(BaseModel):
    schema_version: str
    wav_file: str
    layer: str
    sample_rate: int
    duration_sec: float
    backend_name: str
    backend_available: bool
    hop_length: int
    frame_length: int
    low_band_hz: list[float] = Field(min_length=2, max_length=2)
    frames: list[DspAudioFrame]


class DspAnalysisReport(BaseModel):
    wav_file: str
    status: Literal["ok", "error"]
    layer: str
    backend_name: str
    backend_available: bool
    frame_count: int
    duration_sec: float
    low_band_hz: list[float] = Field(min_length=2, max_length=2)
    mean_rms: float
    max_rms: float
    mean_low_band_rms: float
    max_low_band_rms: float
    attack_rise_count: int
    sustain_count: int
    tail_count: int
    silence_count: int
    warning_count: int
    warnings: list[str]
    output_file: str | None
    debug_csv_file: str | None
