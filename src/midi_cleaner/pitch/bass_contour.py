from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import math

import numpy as np
import soundfile as sf

from midi_cleaner.dsp.analyzer import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    _fit_length,
    _frame_signal,
    _frame_times,
    _safe_smooth,
)
from midi_cleaner.pitch.models import (
    BassPitchContourDocument,
    BassPitchContourReport,
    BassPitchFrame,
)

SCHEMA_VERSION = "0.1.0"


class PitchContourError(Exception):
    """Raised when bass pitch contour analysis cannot be completed."""


@dataclass(frozen=True)
class PitchContourParameters:
    backend: str = "auto"
    min_hz: float = 35.0
    max_hz: float = 400.0
    confidence_threshold: float = 0.60
    frame_length: int = DEFAULT_FRAME_LENGTH
    hop_length: int = DEFAULT_HOP_LENGTH


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64)
    return np.mean(audio.astype(np.float64), axis=1)


def _backend_available(name: str) -> bool:
    value = name.strip().lower()
    if value == "basic":
        return True
    if value == "librosa":
        return importlib.util.find_spec("librosa") is not None
    return False


def _resolve_backend(requested: str, allow_fallback: bool = True) -> tuple[str, bool, list[str]]:
    value = requested.strip().lower()
    warnings: list[str] = []

    if value not in {"auto", "librosa", "basic"}:
        raise PitchContourError(f"Unsupported pitch backend: {requested}")

    if value == "auto":
        if _backend_available("librosa"):
            return "librosa", True, warnings
        return "basic", True, warnings

    if _backend_available(value):
        return value, True, warnings

    if not allow_fallback:
        return value, False, warnings

    warnings.append(f"Requested pitch backend '{value}' unavailable; using 'basic' instead.")
    return "basic", True, warnings


def _frequency_to_midi(f0_hz: float) -> float:
    return 69.0 + (12.0 * math.log2(f0_hz / 440.0))


def _estimate_low_band_energy(
    frame_signal: np.ndarray,
    sample_rate: int,
    min_hz: float,
    max_hz: float,
) -> float:
    if len(frame_signal) == 0:
        return 0.0
    spectrum = np.fft.rfft(frame_signal)
    freqs = np.fft.rfftfreq(len(frame_signal), d=1.0 / float(sample_rate))
    band_mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(band_mask):
        return 0.0
    power = np.abs(spectrum[band_mask]) ** 2
    return float(np.mean(power))


def _estimate_harmonic_energy(
    frame_signal: np.ndarray,
    sample_rate: int,
    f0_hz: float | None,
) -> float | None:
    if f0_hz is None or f0_hz <= 0.0:
        return None
    spectrum = np.abs(np.fft.rfft(frame_signal))
    freqs = np.fft.rfftfreq(len(frame_signal), d=1.0 / float(sample_rate))
    if len(freqs) == 0:
        return None

    bins: list[float] = []
    for multiple in (1, 2, 3):
        target = f0_hz * multiple
        if target <= 0.0 or target > freqs[-1]:
            continue
        idx = int(np.argmin(np.abs(freqs - target)))
        bins.append(float(spectrum[idx]))

    if not bins:
        return None
    return float(sum(bins) / len(bins))


def _basic_pitch_from_frame(
    frame_signal: np.ndarray,
    sample_rate: int,
    min_hz: float,
    max_hz: float,
) -> tuple[float | None, float]:
    if len(frame_signal) == 0:
        return None, 0.0

    window = np.hanning(len(frame_signal))
    spectrum = np.abs(np.fft.rfft(frame_signal * window))
    freqs = np.fft.rfftfreq(len(frame_signal), d=1.0 / float(sample_rate))

    band_mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(band_mask):
        return None, 0.0

    band_spec = spectrum[band_mask]
    if len(band_spec) == 0:
        return None, 0.0

    idx = int(np.argmax(band_spec))
    peak_mag = float(band_spec[idx])
    if peak_mag <= 0.0:
        return None, 0.0

    total_mag = float(np.sum(band_spec))
    if total_mag <= 0.0:
        return None, 0.0

    freq_values = freqs[band_mask]
    f0_hz = float(freq_values[idx])
    confidence = max(0.0, min(1.0, peak_mag / total_mag * 8.0))
    return f0_hz, confidence


def _librosa_pitch(
    mono: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    min_hz: float,
    max_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    f0_hz, voiced_flag, voiced_prob = librosa.pyin(
        mono.astype(np.float32),
        fmin=float(min_hz),
        fmax=float(max_hz),
        sr=int(sample_rate),
        frame_length=int(frame_length),
        hop_length=int(hop_length),
    )

    hz = np.asarray(f0_hz, dtype=np.float64)
    conf = np.asarray(voiced_prob, dtype=np.float64)

    if voiced_flag is not None:
        voiced = np.asarray(voiced_flag, dtype=bool)
        conf = np.where(voiced, conf, 0.0)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    hz = np.nan_to_num(hz, nan=0.0, posinf=0.0, neginf=0.0)
    hz = np.where(hz > 0.0, hz, 0.0)
    return hz, conf


def analyze_bass_pitch_contour(
    wav_file: Path,
    layer: str,
    params: PitchContourParameters,
) -> tuple[BassPitchContourDocument, BassPitchContourReport]:
    if not wav_file.exists() or not wav_file.is_file():
        raise PitchContourError(f"Input WAV file does not exist: {wav_file}")

    if params.min_hz <= 0.0 or params.max_hz <= params.min_hz:
        raise PitchContourError("Invalid pitch frequency bounds")

    try:
        audio, sample_rate = sf.read(str(wav_file), always_2d=False)
    except Exception as exc:  # pragma: no cover - soundfile backend varies
        raise PitchContourError(f"Failed to read WAV file: {wav_file}") from exc

    mono = _ensure_mono(np.asarray(audio))
    duration_sec = float(len(mono) / sample_rate) if sample_rate > 0 else 0.0
    frame_length = int(params.frame_length)
    hop_length = int(params.hop_length)

    backend_name, backend_available, warnings = _resolve_backend(params.backend, allow_fallback=True)
    if not backend_available:
        raise PitchContourError(f"Pitch backend unavailable: {params.backend}")

    frames = _frame_signal(mono, frame_length=frame_length, hop_length=hop_length)
    frame_count = len(frames)
    start_times, end_times = _frame_times(
        frame_count=frame_count,
        hop_length=hop_length,
        frame_length=frame_length,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
    )

    if backend_name == "librosa":
        try:
            f0_hz_arr, confidence_arr = _librosa_pitch(
                mono=mono,
                sample_rate=sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
                min_hz=params.min_hz,
                max_hz=params.max_hz,
            )
            f0_hz_arr = _fit_length(f0_hz_arr, frame_count=frame_count)
            confidence_arr = _fit_length(confidence_arr, frame_count=frame_count)
        except Exception as exc:
            warnings.append("librosa pitch extraction failed; falling back to basic backend.")
            backend_name = "basic"
            f0_vals: list[float] = []
            conf_vals: list[float] = []
            for frame in frames:
                f0_hz, confidence = _basic_pitch_from_frame(
                    frame,
                    sample_rate=sample_rate,
                    min_hz=params.min_hz,
                    max_hz=params.max_hz,
                )
                f0_vals.append(float(f0_hz) if f0_hz is not None else 0.0)
                conf_vals.append(confidence)
            f0_hz_arr = np.asarray(f0_vals, dtype=np.float64)
            confidence_arr = np.asarray(conf_vals, dtype=np.float64)
            if str(exc):
                warnings.append(f"librosa error detail: {exc}")
    else:
        f0_vals = []
        conf_vals = []
        for frame in frames:
            f0_hz, confidence = _basic_pitch_from_frame(
                frame,
                sample_rate=sample_rate,
                min_hz=params.min_hz,
                max_hz=params.max_hz,
            )
            f0_vals.append(float(f0_hz) if f0_hz is not None else 0.0)
            conf_vals.append(confidence)
        f0_hz_arr = np.asarray(f0_vals, dtype=np.float64)
        confidence_arr = np.asarray(conf_vals, dtype=np.float64)

    low_band_energy = np.asarray(
        [
            _estimate_low_band_energy(
                frame,
                sample_rate=sample_rate,
                min_hz=params.min_hz,
                max_hz=params.max_hz,
            )
            for frame in frames
        ],
        dtype=np.float64,
    )
    low_band_energy_smooth = _safe_smooth(low_band_energy, window=5)

    pitch_frames: list[BassPitchFrame] = []
    detected_hz: list[float] = []
    voiced_count = 0

    for idx in range(frame_count):
        f0_hz = float(f0_hz_arr[idx]) if idx < len(f0_hz_arr) else 0.0
        pitch_conf = float(confidence_arr[idx]) if idx < len(confidence_arr) else 0.0
        f0_val = f0_hz if f0_hz > 0.0 else None

        low_band_val = float(low_band_energy_smooth[idx]) if idx < len(low_band_energy_smooth) else 0.0
        harmonic = _estimate_harmonic_energy(frames[idx], sample_rate=sample_rate, f0_hz=f0_val)

        if f0_val is None:
            voiced = False
            pitch_midi_float = None
            pitch_midi_rounded = None
        else:
            pitch_midi_float = _frequency_to_midi(f0_val)
            pitch_midi_rounded = int(round(pitch_midi_float))
            voiced = pitch_conf >= params.confidence_threshold
            if voiced:
                voiced_count += 1
                detected_hz.append(f0_val)

        pitch_frames.append(
            BassPitchFrame(
                frame_index=idx,
                start_sec=float(start_times[idx]),
                end_sec=float(end_times[idx]),
                f0_hz=f0_val,
                pitch_midi_float=pitch_midi_float,
                pitch_midi_rounded=pitch_midi_rounded,
                pitch_confidence=max(0.0, min(1.0, pitch_conf)),
                voiced=voiced,
                low_band_energy=low_band_val,
                harmonic_energy=harmonic,
            )
        )

    frame_count_safe = max(1, frame_count)
    report = BassPitchContourReport(
        wav_file=str(wav_file),
        status="ok",
        layer=layer,
        backend_name=backend_name,
        backend_available=backend_available,
        frame_count=frame_count,
        voiced_frame_count=voiced_count,
        voiced_ratio=float(voiced_count / frame_count_safe),
        mean_pitch_confidence=float(np.mean(confidence_arr)) if len(confidence_arr) else 0.0,
        min_detected_hz=min(detected_hz) if detected_hz else None,
        max_detected_hz=max(detected_hz) if detected_hz else None,
        warning_count=len(warnings),
        warnings=warnings,
        output_file=None,
    )

    document = BassPitchContourDocument(
        schema_version=SCHEMA_VERSION,
        wav_file=str(wav_file),
        layer=layer,
        backend_name=backend_name,
        backend_available=backend_available,
        sample_rate=int(sample_rate),
        duration_sec=duration_sec,
        hop_length=hop_length,
        frame_length=frame_length,
        min_hz=float(params.min_hz),
        max_hz=float(params.max_hz),
        frames=pitch_frames,
    )

    return document, report
