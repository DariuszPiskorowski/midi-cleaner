from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.signal.windows import hann

from midi_cleaner.dsp.backends import BackendResolution, resolve_backend
from midi_cleaner.dsp.models import DspAnalysisReport, DspAudioFeatureDocument, DspAudioFrame

SCHEMA_VERSION = "0.1.0"
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 512


class DspAnalysisError(Exception):
    """Raised when DSP feature analysis cannot be performed."""


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64)
    return np.mean(audio.astype(np.float64), axis=1)


def _frame_signal(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros((1, frame_length), dtype=np.float64)

    frames: list[np.ndarray] = []
    for start in range(0, len(audio), hop_length):
        stop = start + frame_length
        frame = audio[start:stop]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode="constant")
        frames.append(frame)
        if stop >= len(audio):
            break

    return np.stack(frames, axis=0)


def _frame_times(frame_count: int, hop_length: int, frame_length: int, sample_rate: int, duration_sec: float) -> tuple[np.ndarray, np.ndarray]:
    starts = (np.arange(frame_count, dtype=np.float64) * hop_length) / float(sample_rate)
    ends = np.minimum(starts + (frame_length / float(sample_rate)), duration_sec)
    return starts, ends


def _safe_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) == 0:
        return values
    size = max(1, int(window))
    if size == 1:
        return values
    kernel = np.ones(size, dtype=np.float64) / float(size)
    return np.convolve(values, kernel, mode="same")


def _delta(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.zeros_like(values)
    if len(values) > 1:
        out[1:] = values[1:] - values[:-1]
    return out


def _bandpass_filter(signal: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = max(1.0, sample_rate * 0.5)
    low = max(1.0, low_hz)
    high = min(high_hz, nyquist * 0.98)
    if high <= low:
        return signal.copy()

    sos = butter(N=4, Wn=[low, high], btype="bandpass", fs=float(sample_rate), output="sos")
    if len(signal) > 64:
        try:
            return sosfiltfilt(sos, signal)
        except Exception:
            return signal.copy()
    return signal.copy()


def _spectral_flux(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    frames = _frame_signal(signal, frame_length=frame_length, hop_length=hop_length)
    if len(frames) == 0:
        return np.zeros(0, dtype=np.float64)

    window = hann(frame_length, sym=False)
    magnitudes = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
    if len(magnitudes) == 1:
        return np.zeros(1, dtype=np.float64)

    diff = np.diff(magnitudes, axis=0)
    flux = np.sum(np.maximum(0.0, diff), axis=1)
    return np.concatenate(([0.0], flux.astype(np.float64)))


def _fit_length(values: np.ndarray, frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float64)
    if len(values) == frame_count:
        return values.astype(np.float64)
    if len(values) == 0:
        return np.zeros(frame_count, dtype=np.float64)

    source_x = np.linspace(0.0, 1.0, len(values), endpoint=True)
    target_x = np.linspace(0.0, 1.0, frame_count, endpoint=True)
    return np.interp(target_x, source_x, values.astype(np.float64))


def _frame_rms(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    frames = _frame_signal(signal, frame_length=frame_length, hop_length=hop_length)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _librosa_features(
    signal: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import librosa

    y = signal.astype(np.float32)
    onset_strength = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)

    stft_mag = np.abs(
        librosa.stft(y=y, n_fft=frame_length, hop_length=hop_length, center=False)
    )
    if stft_mag.shape[1] <= 1:
        spectral_flux = np.zeros(stft_mag.shape[1], dtype=np.float64)
    else:
        flux = np.sum(np.maximum(0.0, np.diff(stft_mag, axis=1)), axis=0)
        spectral_flux = np.concatenate(([0.0], flux.astype(np.float64)))

    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_rms = librosa.feature.rms(
        y=harmonic,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]
    percussive_rms = librosa.feature.rms(
        y=percussive,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]

    return (
        onset_strength.astype(np.float64),
        spectral_flux.astype(np.float64),
        harmonic_rms.astype(np.float64),
        percussive_rms.astype(np.float64),
    )


def _classify_frames(
    rms_smooth: np.ndarray,
    envelope_delta: np.ndarray,
    low_band_smooth: np.ndarray,
    low_band_delta: np.ndarray,
    onset_strength: np.ndarray,
    harmonic_rms: np.ndarray,
    percussive_rms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nonzero = rms_smooth[rms_smooth > 0.0]
    if len(nonzero):
        silence_threshold = max(1e-5, float(np.percentile(nonzero, 20) * 0.35))
    else:
        silence_threshold = 1e-5

    onset_threshold = float(np.percentile(onset_strength, 75) * 0.6) if len(onset_strength) else 0.0
    onset_threshold = max(1e-7, onset_threshold)

    positive_delta = envelope_delta[envelope_delta > 0.0]
    attack_delta_threshold = (
        float(np.percentile(positive_delta, 60)) if len(positive_delta) else 1e-6
    )

    negative_low_band_delta = low_band_delta[low_band_delta < 0.0]
    tail_delta_threshold = (
        float(np.percentile(negative_low_band_delta, 40)) if len(negative_low_band_delta) else -1e-6
    )

    is_silence = rms_smooth <= silence_threshold
    is_attack = (
        (onset_strength >= onset_threshold)
        & (envelope_delta >= attack_delta_threshold)
        & (~is_silence)
    )
    is_tail = (
        (~is_silence)
        & (~is_attack)
        & (low_band_delta <= tail_delta_threshold)
        & (harmonic_rms >= percussive_rms * 0.75)
    )
    is_sustain = (~is_silence) & (~is_attack) & (~is_tail)
    return is_attack, is_sustain, is_tail, is_silence


def _write_debug_csv(path: Path, frames: list[DspAudioFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "start_sec",
                "rms_smooth",
                "envelope_smooth",
                "low_band_envelope_smooth",
                "onset_strength",
                "spectral_flux",
                "harmonic_rms",
                "percussive_rms",
                "is_attack_rise",
                "is_sustain",
                "is_tail",
                "is_silence",
            ],
        )
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "start_sec": frame.start_sec,
                    "rms_smooth": frame.rms_smooth,
                    "envelope_smooth": frame.envelope_smooth,
                    "low_band_envelope_smooth": frame.low_band_envelope_smooth,
                    "onset_strength": frame.onset_strength,
                    "spectral_flux": frame.spectral_flux,
                    "harmonic_rms": frame.harmonic_rms,
                    "percussive_rms": frame.percussive_rms,
                    "is_attack_rise": frame.is_attack_rise,
                    "is_sustain": frame.is_sustain,
                    "is_tail": frame.is_tail,
                    "is_silence": frame.is_silence,
                }
            )


def analyze_dsp_stem(
    wav_file: Path,
    layer: str,
    backend: str = "auto",
    allow_backend_fallback: bool = True,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    low_band_hz: tuple[float, float] | None = None,
    debug_csv_path: Path | None = None,
) -> tuple[DspAudioFeatureDocument, DspAnalysisReport]:
    if not wav_file.exists() or not wav_file.is_file():
        raise DspAnalysisError(f"Input WAV file does not exist: {wav_file}")

    try:
        audio, sample_rate = sf.read(str(wav_file), always_2d=False)
    except Exception as exc:  # pragma: no cover - soundfile backend varies
        raise DspAnalysisError(f"Failed to read WAV file: {wav_file}") from exc

    mono = _ensure_mono(np.asarray(audio))
    duration_sec = float(len(mono) / sample_rate) if sample_rate > 0 else 0.0

    if low_band_hz is None:
        low_band_hz = (40.0, 500.0) if layer.lower() == "bass" else (80.0, 2000.0)

    try:
        resolution: BackendResolution = resolve_backend(
            requested=backend,
            allow_fallback=allow_backend_fallback,
        )
    except ValueError as exc:
        raise DspAnalysisError(str(exc)) from exc

    warnings = list(resolution.warnings)
    if not resolution.backend_available:
        raise DspAnalysisError(f"DSP backend unavailable: {backend}")

    frames = _frame_signal(mono, frame_length=frame_length, hop_length=hop_length)
    frame_count = len(frames)
    start_times, end_times = _frame_times(
        frame_count=frame_count,
        hop_length=hop_length,
        frame_length=frame_length,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
    )

    rms = np.sqrt(np.mean(frames * frames, axis=1))
    rms_smooth = _safe_smooth(rms, window=5)
    rms_delta = _delta(rms_smooth)

    envelope_signal = np.abs(hilbert(mono)) if len(mono) else np.zeros(0, dtype=np.float64)
    envelope = _fit_length(
        _frame_rms(envelope_signal, frame_length=frame_length, hop_length=hop_length),
        frame_count=frame_count,
    )
    envelope_smooth = _safe_smooth(envelope, window=5)
    envelope_delta = _delta(envelope_smooth)

    if resolution.backend_name in {"scipy", "librosa"}:
        low_band_signal = _bandpass_filter(
            mono,
            sample_rate=sample_rate,
            low_hz=float(low_band_hz[0]),
            high_hz=float(low_band_hz[1]),
        )
    else:
        low_band_signal = mono.copy()

    low_band_rms = _frame_rms(low_band_signal, frame_length=frame_length, hop_length=hop_length)
    low_band_envelope_signal = np.abs(hilbert(low_band_signal)) if len(low_band_signal) else np.zeros(0)
    low_band_envelope = _frame_rms(
        low_band_envelope_signal,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    low_band_envelope_smooth = _safe_smooth(low_band_envelope, window=5)
    low_band_delta = _delta(low_band_envelope_smooth)

    if resolution.backend_name == "librosa":
        try:
            onset_strength, spectral_flux, harmonic_rms, percussive_rms = _librosa_features(
                mono,
                sample_rate=sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
            )
        except Exception as exc:
            if not allow_backend_fallback:
                raise DspAnalysisError("librosa backend failed during feature extraction") from exc
            fallback = resolve_backend(requested="scipy", allow_fallback=True)
            warnings.append("librosa feature extraction failed; falling back to scipy-style metrics.")
            resolution = BackendResolution(
                backend_name=fallback.backend_name,
                backend_available=fallback.backend_available,
                warnings=warnings,
            )
            spectral_flux = _spectral_flux(mono, frame_length=frame_length, hop_length=hop_length)
            onset_strength = spectral_flux.copy()
            harmonic_rms = _safe_smooth(rms, window=7)
            percussive_rms = np.maximum(0.0, np.abs(_delta(rms)) * 2.0)
    else:
        spectral_flux = _spectral_flux(mono, frame_length=frame_length, hop_length=hop_length)
        onset_strength = spectral_flux.copy()
        if resolution.backend_name == "scipy":
            harmonic_rms = _safe_smooth(low_band_rms, window=7)
            percussive_rms = np.maximum(0.0, np.abs(_delta(rms)) * 2.0)
        else:
            harmonic_rms = rms.copy()
            percussive_rms = np.maximum(0.0, spectral_flux * 0.5)

    spectral_flux = _fit_length(spectral_flux, frame_count=frame_count)
    onset_strength = _fit_length(onset_strength, frame_count=frame_count)
    harmonic_rms = _fit_length(harmonic_rms, frame_count=frame_count)
    percussive_rms = _fit_length(percussive_rms, frame_count=frame_count)

    is_attack, is_sustain, is_tail, is_silence = _classify_frames(
        rms_smooth=rms_smooth,
        envelope_delta=envelope_delta,
        low_band_smooth=low_band_envelope_smooth,
        low_band_delta=low_band_delta,
        onset_strength=onset_strength,
        harmonic_rms=harmonic_rms,
        percussive_rms=percussive_rms,
    )

    dsp_frames: list[DspAudioFrame] = []
    for idx in range(frame_count):
        dsp_frames.append(
            DspAudioFrame(
                frame_index=idx,
                start_sec=float(start_times[idx]),
                end_sec=float(end_times[idx]),
                rms=float(rms[idx]),
                rms_smooth=float(rms_smooth[idx]),
                rms_delta=float(rms_delta[idx]),
                envelope=float(envelope[idx]),
                envelope_smooth=float(envelope_smooth[idx]),
                envelope_delta=float(envelope_delta[idx]),
                low_band_rms=float(low_band_rms[idx]),
                low_band_envelope=float(low_band_envelope[idx]),
                low_band_envelope_smooth=float(low_band_envelope_smooth[idx]),
                low_band_delta=float(low_band_delta[idx]),
                spectral_flux=float(spectral_flux[idx]),
                onset_strength=float(onset_strength[idx]),
                harmonic_rms=float(harmonic_rms[idx]),
                percussive_rms=float(percussive_rms[idx]),
                is_attack_rise=bool(is_attack[idx]),
                is_sustain=bool(is_sustain[idx]),
                is_tail=bool(is_tail[idx]),
                is_silence=bool(is_silence[idx]),
            )
        )

    debug_csv_file = None
    if debug_csv_path is not None:
        _write_debug_csv(debug_csv_path, dsp_frames)
        debug_csv_file = str(debug_csv_path)

    document = DspAudioFeatureDocument(
        schema_version=SCHEMA_VERSION,
        wav_file=str(wav_file),
        layer=layer,
        sample_rate=int(sample_rate),
        duration_sec=float(duration_sec),
        backend_name=resolution.backend_name,
        backend_available=resolution.backend_available,
        hop_length=hop_length,
        frame_length=frame_length,
        low_band_hz=[float(low_band_hz[0]), float(low_band_hz[1])],
        frames=dsp_frames,
    )

    report = DspAnalysisReport(
        wav_file=str(wav_file),
        status="ok",
        layer=layer,
        backend_name=resolution.backend_name,
        backend_available=resolution.backend_available,
        frame_count=len(dsp_frames),
        duration_sec=float(duration_sec),
        low_band_hz=[float(low_band_hz[0]), float(low_band_hz[1])],
        mean_rms=float(np.mean(rms)) if len(rms) else 0.0,
        max_rms=float(np.max(rms)) if len(rms) else 0.0,
        mean_low_band_rms=float(np.mean(low_band_rms)) if len(low_band_rms) else 0.0,
        max_low_band_rms=float(np.max(low_band_rms)) if len(low_band_rms) else 0.0,
        attack_rise_count=int(np.sum(is_attack)),
        sustain_count=int(np.sum(is_sustain)),
        tail_count=int(np.sum(is_tail)),
        silence_count=int(np.sum(is_silence)),
        warning_count=len(warnings),
        warnings=warnings,
        output_file=None,
        debug_csv_file=debug_csv_file,
    )

    return document, report
