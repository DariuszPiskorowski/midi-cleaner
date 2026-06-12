from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal.windows import hann

from midi_cleaner.audio.models import (
    AudioAnalysisReport,
    AudioFeatureDocument,
    AudioFrameFeature,
    AudioGlobalFeatures,
)

SCHEMA_VERSION = "0.1.0"
DEFAULT_FRAME_SIZE = 2048
DEFAULT_HOP_SIZE = 512
DEFAULT_SILENCE_RMS_THRESHOLD = 1e-3
DEFAULT_ONSET_THRESHOLD = 1e-4


class AudioAnalysisError(Exception):
    """Raised when WAV analysis cannot be performed."""


def _ensure_mono(audio: np.ndarray) -> tuple[np.ndarray, int]:
    if audio.ndim == 1:
        return audio.astype(np.float64), 1

    channels = int(audio.shape[1])
    mono = np.mean(audio.astype(np.float64), axis=1)
    return mono, channels


def _compute_spectral_features(frame: np.ndarray, sample_rate: int) -> tuple[float | None, float | None]:
    window = hann(len(frame), sym=False)
    spectrum = np.abs(np.fft.rfft(frame * window))
    energy = float(np.sum(spectrum))
    if energy <= 1e-12:
        return None, None

    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sample_rate)
    centroid = float(np.sum(freqs * spectrum) / energy)

    cumsum = np.cumsum(spectrum)
    rolloff_threshold = 0.85 * cumsum[-1]
    rolloff_index = int(np.searchsorted(cumsum, rolloff_threshold, side="left"))
    rolloff_index = min(rolloff_index, len(freqs) - 1)
    rolloff = float(freqs[rolloff_index])

    return centroid, rolloff


def _frame_signal(audio: np.ndarray, frame_size: int, hop_size: int) -> list[np.ndarray]:
    if len(audio) == 0:
        return [np.zeros(frame_size, dtype=np.float64)]

    frames: list[np.ndarray] = []
    for start in range(0, len(audio), hop_size):
        stop = start + frame_size
        frame = audio[start:stop]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), mode="constant")
        frames.append(frame)

        if stop >= len(audio):
            break

    return frames


def analyze_stem(
    input_wav: Path,
    layer: str,
    frame_size: int = DEFAULT_FRAME_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
) -> tuple[AudioFeatureDocument, AudioAnalysisReport]:
    if not input_wav.exists() or not input_wav.is_file():
        raise AudioAnalysisError(f"Input WAV file does not exist: {input_wav}")

    try:
        audio, sample_rate = sf.read(str(input_wav), always_2d=False)
    except Exception as exc:  # pragma: no cover - soundfile backend varies
        raise AudioAnalysisError(f"Failed to read WAV file: {input_wav}") from exc

    audio_array = np.asarray(audio)
    mono, channels = _ensure_mono(audio_array)

    duration_sec = float(len(mono) / sample_rate) if sample_rate > 0 else 0.0
    warnings: list[str] = []
    if len(mono) < frame_size:
        warnings.append("Input shorter than one frame; analysis uses zero-padded frame.")

    raw_frames = _frame_signal(mono, frame_size=frame_size, hop_size=hop_size)
    energy = np.array([float(np.mean(frame * frame)) for frame in raw_frames], dtype=np.float64)

    frame_features: list[AudioFrameFeature] = []
    previous_energy = 0.0

    for index, frame in enumerate(raw_frames):
        frame_rms = float(np.sqrt(energy[index]))
        frame_peak = float(np.max(np.abs(frame)))

        signs = np.signbit(frame)
        zcr = float(np.mean(signs[1:] != signs[:-1])) if len(frame) > 1 else 0.0

        centroid, rolloff = _compute_spectral_features(frame, sample_rate=sample_rate)

        onset_score = max(0.0, float(energy[index] - previous_energy))
        previous_energy = float(energy[index])

        start_sec = (index * hop_size) / sample_rate
        end_sec = min((index * hop_size + frame_size) / sample_rate, duration_sec)

        frame_features.append(
            AudioFrameFeature(
                frame_index=index,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                rms=frame_rms,
                peak=frame_peak,
                zero_crossing_rate=zcr,
                spectral_centroid_hz=centroid,
                spectral_rolloff_hz=rolloff,
                is_silent=frame_rms < DEFAULT_SILENCE_RMS_THRESHOLD,
                onset_score=onset_score,
            )
        )

    centroid_values = [f.spectral_centroid_hz for f in frame_features if f.spectral_centroid_hz is not None]
    rolloff_values = [f.spectral_rolloff_hz for f in frame_features if f.spectral_rolloff_hz is not None]

    onset_count = sum(1 for feature in frame_features if feature.onset_score > DEFAULT_ONSET_THRESHOLD)
    silence_ratio = (
        sum(1 for feature in frame_features if feature.is_silent) / len(frame_features)
        if frame_features
        else 1.0
    )

    global_features = AudioGlobalFeatures(
        peak=float(np.max(np.abs(mono))) if len(mono) else 0.0,
        rms=float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0,
        duration_sec=duration_sec,
        estimated_silence_ratio=float(silence_ratio),
        frame_count=len(frame_features),
        onset_count=onset_count,
        mean_spectral_centroid_hz=float(np.mean(centroid_values)) if centroid_values else None,
        mean_spectral_rolloff_hz=float(np.mean(rolloff_values)) if rolloff_values else None,
    )

    document = AudioFeatureDocument(
        schema_version=SCHEMA_VERSION,
        source_file=str(input_wav),
        layer=layer,
        sample_rate=int(sample_rate),
        channels=channels,
        duration_sec=duration_sec,
        frame_size=frame_size,
        hop_size=hop_size,
        frames=frame_features,
        global_features=global_features,
    )

    report = AudioAnalysisReport(
        input_file=str(input_wav),
        layer=layer,
        status="ok",
        sample_rate=int(sample_rate),
        channels=channels,
        duration_sec=duration_sec,
        frame_count=len(frame_features),
        onset_count=onset_count,
        warning_count=len(warnings),
        warnings=warnings,
        output_file=None,
    )

    return document, report
