from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import mido
import numpy as np
import soundfile as sf
from scipy.signal.windows import hann

from midi_cleaner.midi.drum_maps import (
    DrumMapDefinition,
    DrumMapError,
    load_custom_drum_map,
    load_preset_drum_map,
    resolve_ujam_candy_layout_notes,
)

DrumClass = Literal[
    "kick",
    "snare_or_clap",
    "hat",
    "cymbal",
    "crash_or_cymbal",
    "tom_or_perc",
    "unknown",
]
SnareTarget = Literal["sn1", "sn2", "clap"]
ExtractionProfile = Literal["conservative", "balanced", "sensitive"]

DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_MIN_ONSET_STRENGTH = 0.20
DEFAULT_BPM = 120.0

_CLASS_FILE_NAMES: dict[str, str] = {
    "kick": "kick.mid",
    "snare_or_clap": "snare_clap.mid",
    "hat": "hat.mid",
    "cymbal": "cymbal.mid",
    "tom_or_perc": "tom_perc.mid",
}

_CLASS_NOTE_DURATIONS_SEC: dict[DrumClass, float] = {
    "kick": 0.10,
    "snare_or_clap": 0.10,
    "hat": 0.06,
    "cymbal": 0.32,
    "crash_or_cymbal": 0.32,
    "tom_or_perc": 0.14,
    "unknown": 0.08,
}

_CANONICAL_CLASSES: tuple[str, ...] = (
    "kick",
    "snare_or_clap",
    "hat",
    "cymbal",
    "tom_or_perc",
    "unknown",
)


_PROFILE_DEFAULTS: dict[str, dict[str, float | int]] = {
    "conservative": {
        "onset_pre_max": 1,
        "onset_post_max": 1,
        "onset_pre_avg": 6,
        "onset_post_avg": 6,
        "onset_delta": 0.10,
        "onset_wait": 2,
        "min_hit_spacing_ms": 80.0,
        "kick_refractory_ms": 160.0,
        "snare_refractory_ms": 140.0,
        "hat_refractory_ms": 70.0,
        "cymbal_refractory_ms": 280.0,
        "tom_refractory_ms": 140.0,
        "same_transient_window_ms": 35.0,
    },
    "balanced": {
        "onset_pre_max": 1,
        "onset_post_max": 1,
        "onset_pre_avg": 5,
        "onset_post_avg": 5,
        "onset_delta": 0.08,
        "onset_wait": 1,
        "min_hit_spacing_ms": 60.0,
        "kick_refractory_ms": 120.0,
        "snare_refractory_ms": 120.0,
        "hat_refractory_ms": 55.0,
        "cymbal_refractory_ms": 180.0,
        "tom_refractory_ms": 100.0,
        "same_transient_window_ms": 35.0,
    },
    "sensitive": {
        "onset_pre_max": 1,
        "onset_post_max": 1,
        "onset_pre_avg": 4,
        "onset_post_avg": 4,
        "onset_delta": 0.05,
        "onset_wait": 1,
        "min_hit_spacing_ms": 45.0,
        "kick_refractory_ms": 90.0,
        "snare_refractory_ms": 90.0,
        "hat_refractory_ms": 40.0,
        "cymbal_refractory_ms": 130.0,
        "tom_refractory_ms": 80.0,
        "same_transient_window_ms": 30.0,
    },
}


class AudioDrumExtractionError(Exception):
    """Raised when drum extraction from audio cannot continue."""


@dataclass(frozen=True)
class AudioDrumExtractionParameters:
    output_file: Path
    target_map: str
    map_file: Path | None = None
    c1_midi_note: int = 36
    bpm: float | None = None
    channel: int = 9
    min_onset_strength: float = DEFAULT_MIN_ONSET_STRENGTH
    dry_run: bool = False
    separate_files: bool = False
    debug_csv: Path | None = None
    report_file: Path | None = None
    snare_target: SnareTarget = "clap"
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT
    profile: ExtractionProfile = "balanced"
    onset_pre_max: int | None = None
    onset_post_max: int | None = None
    onset_pre_avg: int | None = None
    onset_post_avg: int | None = None
    onset_delta: float | None = None
    onset_wait: int | None = None
    min_hit_spacing_ms: float | None = None
    kick_refractory_ms: float | None = None
    snare_refractory_ms: float | None = None
    hat_refractory_ms: float | None = None
    cymbal_refractory_ms: float | None = None
    tom_refractory_ms: float | None = None
    same_transient_window_ms: float | None = None


@dataclass
class PerHitSummary:
    onset_sec: float
    raw_onset_sec: float
    accepted_onset_sec: float | None
    tick: int
    class_name: DrumClass
    target_note: int
    velocity: int
    confidence: float
    low_energy_ratio: float
    mid_energy_ratio: float
    high_energy_ratio: float
    spectral_centroid: float
    onset_strength: float
    suppressed: bool
    suppression_reason: str | None
    grouped_transient_id: int | None
    class_refractory_ms: float
    nearest_previous_same_class_ms: float | None


@dataclass
class AudioDrumExtractionReport:
    wav_file: str
    output_file: str | None
    duration_sec: float
    sample_rate: int
    detected_bpm: float | None
    bpm_used: float
    bpm_source: Literal["detected", "forced"]
    onset_count: int
    raw_onset_count: int
    accepted_onset_count: int
    suppressed_duplicate_count: int
    suppressed_by_class: dict[str, int]
    min_hit_spacing_ms: float
    class_refractory_ms: dict[str, float]
    same_transient_window_ms: float
    class_counts: dict[str, int]
    output_note_counts: dict[str, int]
    output_pitch_counts: dict[str, int]
    notes_per_second: float
    class_notes_per_second: dict[str, float]
    velocity_summary: dict[str, float]
    too_dense_warning: bool
    duplicate_interval_summary: dict[str, dict[str, float | int | None]]
    target_map: str
    c1_midi_note: int
    synchronization_preserved: bool
    warnings: list[str] = field(default_factory=list)
    per_hit_summary: list[PerHitSummary] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["per_hit_summary"] = [
            {
                "onset_sec": item.onset_sec,
                "raw_onset_sec": item.raw_onset_sec,
                "accepted_onset_sec": item.accepted_onset_sec,
                "tick": item.tick,
                "class": item.class_name,
                "target_note": item.target_note,
                "velocity": item.velocity,
                "confidence": item.confidence,
                "low_energy_ratio": item.low_energy_ratio,
                "mid_energy_ratio": item.mid_energy_ratio,
                "high_energy_ratio": item.high_energy_ratio,
                "spectral_centroid": item.spectral_centroid,
                "onset_strength": item.onset_strength,
                "suppressed": item.suppressed,
                "suppression_reason": item.suppression_reason,
                "grouped_transient_id": item.grouped_transient_id,
                "class_refractory_ms": item.class_refractory_ms,
                "nearest_previous_same_class_ms": item.nearest_previous_same_class_ms,
            }
            for item in self.per_hit_summary
        ]
        return payload


@dataclass
class _DetectedHit:
    hit_id: int
    onset_sec: float
    accepted_onset_sec: float | None
    tick: int
    onset_strength: float
    low_band_onset_strength: float
    mid_band_onset_strength: float
    high_band_onset_strength: float
    class_name: DrumClass
    low_energy_ratio: float
    mid_energy_ratio: float
    high_energy_ratio: float
    spectral_centroid: float
    confidence: float
    target_note: int
    velocity: int
    suppressed: bool
    suppression_reason: str | None
    grouped_transient_id: int | None
    class_refractory_ms: float
    nearest_previous_same_class_ms: float | None

    def score(self) -> float:
        return (
            0.62 * float(self.onset_strength)
            + 0.30 * float(self.confidence)
            + 0.08 * float(self.high_band_onset_strength)
        )


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64)
    return np.mean(audio.astype(np.float64), axis=1)


def _frame_signal(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros((1, frame_size), dtype=np.float64)

    frames: list[np.ndarray] = []
    for start in range(0, len(audio), hop_size):
        stop = start + frame_size
        frame = audio[start:stop]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), mode="constant")
        frames.append(frame)
        if stop >= len(audio):
            break

    return np.stack(frames, axis=0)


def _normalize_envelope(raw: np.ndarray) -> np.ndarray:
    if len(raw) == 0:
        return np.zeros(0, dtype=np.float64)
    if not np.any(raw > 0.0):
        return np.zeros_like(raw, dtype=np.float64)

    floor = float(np.percentile(raw, 5))
    shifted = np.maximum(0.0, raw - floor)
    p95 = float(np.percentile(shifted, 95))
    scale = max(p95, 1e-9)
    return np.clip(shifted / scale, 0.0, 2.5)


def _onset_strength_envelopes(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[dict[str, np.ndarray], int, int]:
    frame_size = 1024
    hop_size = 256
    frames = _frame_signal(audio, frame_size=frame_size, hop_size=hop_size)
    window = hann(frame_size, sym=False)

    magnitudes = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
    if len(magnitudes) <= 1:
        zeros = np.zeros(len(magnitudes), dtype=np.float64)
        return {
            "full": zeros,
            "low": zeros,
            "mid": zeros,
            "high": zeros,
        }, frame_size, hop_size

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sample_rate))
    diff = np.maximum(0.0, np.diff(magnitudes, axis=0))

    low_mask = (freqs >= 20.0) & (freqs < 180.0)
    mid_mask = (freqs >= 180.0) & (freqs < 3000.0)
    high_mask = (freqs >= 3000.0) & (freqs < 12000.0)

    full_raw = np.concatenate(([0.0], np.sum(diff, axis=1).astype(np.float64)))
    low_raw = np.concatenate(([0.0], np.sum(diff[:, low_mask], axis=1).astype(np.float64)))
    mid_raw = np.concatenate(([0.0], np.sum(diff[:, mid_mask], axis=1).astype(np.float64)))
    high_raw = np.concatenate(([0.0], np.sum(diff[:, high_mask], axis=1).astype(np.float64)))

    return {
        "full": _normalize_envelope(full_raw),
        "low": _normalize_envelope(low_raw),
        "mid": _normalize_envelope(mid_raw),
        "high": _normalize_envelope(high_raw),
    }, frame_size, hop_size


def _peak_pick_indices(
    onset_strength: np.ndarray,
    *,
    pre_max: int,
    post_max: int,
    pre_avg: int,
    post_avg: int,
    delta: float,
    wait: int,
    min_onset_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(onset_strength) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    peaks: list[int] = []
    heights: list[float] = []

    for idx in range(len(onset_strength)):
        value = float(onset_strength[idx])
        if value < float(min_onset_strength):
            continue

        max_start = max(0, idx - pre_max)
        max_stop = min(len(onset_strength), idx + post_max + 1)
        if value < float(np.max(onset_strength[max_start:max_stop])):
            continue

        avg_start = max(0, idx - pre_avg)
        avg_stop = min(len(onset_strength), idx + post_avg + 1)
        local_avg = float(np.mean(onset_strength[avg_start:avg_stop]))
        if value < local_avg + float(delta):
            continue

        if peaks and (idx - peaks[-1]) <= wait:
            if value > heights[-1]:
                peaks[-1] = idx
                heights[-1] = value
            continue

        peaks.append(idx)
        heights.append(value)

    return np.asarray(peaks, dtype=np.int64), np.asarray(heights, dtype=np.float64)


def _detect_onsets(
    onset_strength: np.ndarray,
    sample_rate: int,
    hop_size: int,
    min_onset_strength: float,
    *,
    onset_pre_max: int,
    onset_post_max: int,
    onset_pre_avg: int,
    onset_post_avg: int,
    onset_delta: float,
    onset_wait: int,
) -> tuple[np.ndarray, np.ndarray]:
    peaks, strengths = _peak_pick_indices(
        onset_strength,
        pre_max=onset_pre_max,
        post_max=onset_post_max,
        pre_avg=onset_pre_avg,
        post_avg=onset_post_avg,
        delta=onset_delta,
        wait=onset_wait,
        min_onset_strength=min_onset_strength,
    )

    if len(peaks) == 0:
        peaks, strengths = _peak_pick_indices(
            onset_strength,
            pre_max=onset_pre_max,
            post_max=onset_post_max,
            pre_avg=onset_pre_avg,
            post_avg=onset_post_avg,
            delta=max(0.0, onset_delta * 0.85),
            wait=onset_wait,
            min_onset_strength=max(0.03, min_onset_strength * 0.75),
        )

    times = peaks.astype(np.float64) * (hop_size / float(sample_rate))
    return times, strengths.astype(np.float64)


def _estimate_bpm(onset_times_sec: np.ndarray) -> float | None:
    if len(onset_times_sec) < 2:
        return None

    intervals = np.diff(onset_times_sec)
    intervals = intervals[(intervals >= 0.06) & (intervals <= 1.5)]
    if len(intervals) == 0:
        return None

    median_interval = float(np.median(intervals))
    if median_interval <= 0.0:
        return None

    bpm = 60.0 / median_interval
    while bpm < 70.0:
        bpm *= 2.0
    while bpm > 190.0:
        bpm /= 2.0
    return bpm


def _band_energy(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    low = max(0.0, low_hz)
    high = max(low, high_hz)
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(spectrum[mask] * spectrum[mask]))


def _extract_hit_spectral_features(
    audio: np.ndarray,
    sample_rate: int,
    onset_sec: float,
) -> tuple[float, float, float, float, float, float, float]:
    onset_sample = int(round(onset_sec * sample_rate))
    frame_size = max(256, int(round(0.10 * sample_rate)))

    frame = audio[onset_sample : onset_sample + frame_size]
    if len(frame) < frame_size:
        frame = np.pad(frame, (0, frame_size - len(frame)), mode="constant")

    window = hann(frame_size, sym=False)
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sample_rate))

    low_energy = _band_energy(spectrum, freqs, 20.0, 160.0)
    mid_energy = _band_energy(spectrum, freqs, 160.0, 3000.0)
    high_energy = _band_energy(spectrum, freqs, 3000.0, 12000.0)
    total = max(low_energy + mid_energy + high_energy, 1e-12)

    low_ratio = low_energy / total
    mid_ratio = mid_energy / total
    high_ratio = high_energy / total

    weighted = float(np.sum(freqs * spectrum))
    spec_sum = float(np.sum(spectrum))
    centroid_hz = weighted / max(spec_sum, 1e-12)

    cumsum = np.cumsum(spectrum)
    rolloff_threshold = 0.85 * cumsum[-1] if len(cumsum) else 0.0
    rolloff_index = int(np.searchsorted(cumsum, rolloff_threshold, side="left")) if len(cumsum) else 0
    rolloff_index = min(rolloff_index, len(freqs) - 1)
    rolloff_hz = float(freqs[rolloff_index]) if len(freqs) else 0.0

    rms = float(np.sqrt(np.mean(frame * frame)))
    peak = float(np.max(np.abs(frame)))

    return low_ratio, mid_ratio, high_ratio, centroid_hz, rolloff_hz, rms, peak


def _classify_hit(
    onset_strength: float,
    low_ratio: float,
    mid_ratio: float,
    high_ratio: float,
    centroid_hz: float,
    rolloff_hz: float,
) -> tuple[DrumClass, float]:
    dominant = max(low_ratio, mid_ratio, high_ratio)
    secondary = sorted([low_ratio, mid_ratio, high_ratio], reverse=True)[1]
    separation = max(0.0, dominant - secondary)

    if low_ratio >= 0.52 and low_ratio > mid_ratio * 1.1:
        base_confidence = 0.70 + (low_ratio - 0.52)
        return "kick", min(1.0, base_confidence + 0.20 * onset_strength)

    if high_ratio >= 0.62 and (centroid_hz >= 5600.0 or rolloff_hz >= 9300.0):
        if high_ratio >= 0.76 or rolloff_hz >= 11200.0:
            base_confidence = 0.66 + (high_ratio - 0.62)
            return "cymbal", min(1.0, base_confidence + 0.20 * onset_strength)
        base_confidence = 0.62 + (high_ratio - 0.62)
        return "hat", min(1.0, base_confidence + 0.20 * onset_strength)

    if high_ratio >= 0.42 and centroid_hz >= 3200.0:
        base_confidence = 0.58 + (high_ratio - 0.42)
        return "hat", min(1.0, base_confidence + 0.20 * onset_strength)

    if mid_ratio >= 0.45:
        if low_ratio >= 0.30 and centroid_hz < 1700.0:
            base_confidence = 0.57 + 0.4 * separation
            return "tom_or_perc", min(1.0, base_confidence + 0.18 * onset_strength)
        base_confidence = 0.60 + 0.4 * separation
        return "snare_or_clap", min(1.0, base_confidence + 0.20 * onset_strength)

    if low_ratio >= 0.34 and mid_ratio >= 0.28:
        base_confidence = 0.50 + 0.3 * separation
        return "tom_or_perc", min(1.0, base_confidence + 0.15 * onset_strength)

    return "unknown", min(0.55, 0.25 + 0.25 * onset_strength + 0.25 * separation)


def _resolve_target_map_definition(params: AudioDrumExtractionParameters) -> DrumMapDefinition:
    if params.target_map not in {"gm", "sitala", "ujam-candy", "custom"}:
        raise AudioDrumExtractionError(
            "Invalid --target-map. Use gm, sitala, ujam-candy, or custom."
        )

    try:
        if params.target_map == "custom":
            if params.map_file is None:
                raise AudioDrumExtractionError(
                    "--map-file is required when --target-map custom."
                )
            return load_custom_drum_map(params.map_file)

        return load_preset_drum_map(params.target_map, c1_midi_note=params.c1_midi_note)
    except DrumMapError as exc:
        raise AudioDrumExtractionError(str(exc)) from exc


def _infer_custom_class_targets(
    map_definition: DrumMapDefinition,
    *,
    snare_target: SnareTarget,
) -> dict[DrumClass, int]:
    by_keyword: dict[str, list[int]] = {
        "kick": [],
        "sn1": [],
        "sn2": [],
        "clap": [],
        "snare": [],
        "hat": [],
        "cym": [],
        "tom": [],
        "perc": [],
    }

    for source_note, label in map_definition.labels.items():
        target_note = int(map_definition.notes.get(source_note, source_note))
        normalized = str(label).lower()
        if "kick" in normalized:
            by_keyword["kick"].append(target_note)
        if "sn1" in normalized:
            by_keyword["sn1"].append(target_note)
        if "sn2" in normalized:
            by_keyword["sn2"].append(target_note)
        if "clap" in normalized:
            by_keyword["clap"].append(target_note)
        if "snare" in normalized:
            by_keyword["snare"].append(target_note)
        if "hat" in normalized or "hh" in normalized:
            by_keyword["hat"].append(target_note)
        if "cym" in normalized or "crash" in normalized:
            by_keyword["cym"].append(target_note)
        if "tom" in normalized:
            by_keyword["tom"].append(target_note)
        if "perc" in normalized:
            by_keyword["perc"].append(target_note)

    fallback_note = 36
    if map_definition.notes:
        fallback_note = int(next(iter(sorted(map_definition.notes.values()))))

    def _first(*groups: str) -> int:
        for group in groups:
            if by_keyword[group]:
                return int(by_keyword[group][0])
        return fallback_note

    if snare_target == "sn1":
        snare_note = _first("sn1", "snare", "clap")
    elif snare_target == "sn2":
        snare_note = _first("sn2", "snare", "clap")
    else:
        snare_note = _first("clap", "snare", "sn1", "sn2")

    return {
        "kick": _first("kick"),
        "snare_or_clap": snare_note,
        "hat": _first("hat"),
        "cymbal": _first("cym"),
        "crash_or_cymbal": _first("cym"),
        "tom_or_perc": _first("tom", "perc"),
        "unknown": fallback_note,
    }


def _resolve_class_targets(
    map_definition: DrumMapDefinition,
    params: AudioDrumExtractionParameters,
) -> dict[DrumClass, int]:
    if params.target_map == "ujam-candy":
        candy_layout = resolve_ujam_candy_layout_notes(params.c1_midi_note)
        snare_note_name = {
            "sn1": "E1",
            "sn2": "F1",
            "clap": "G1",
        }[params.snare_target]
        return {
            "kick": candy_layout["C1"],
            "snare_or_clap": candy_layout[snare_note_name],
            "hat": candy_layout["C2"],
            "cymbal": candy_layout["C3"],
            "crash_or_cymbal": candy_layout["C3"],
            "tom_or_perc": candy_layout["D2"],
            "unknown": candy_layout["D1"],
        }

    if params.target_map in {"gm", "sitala"}:
        snare_note = {
            "sn1": 38,
            "sn2": 40,
            "clap": 39,
        }[params.snare_target]
        return {
            "kick": 36,
            "snare_or_clap": snare_note,
            "hat": 42,
            "cymbal": 49,
            "crash_or_cymbal": 49,
            "tom_or_perc": 45,
            "unknown": 39,
        }

    return _infer_custom_class_targets(map_definition, snare_target=params.snare_target)


def _normalize_class_name(class_name: DrumClass) -> DrumClass:
    if class_name == "crash_or_cymbal":
        return "cymbal"
    return class_name


def _canonical_class(class_name: DrumClass) -> str:
    normalized = _normalize_class_name(class_name)
    if normalized not in _CANONICAL_CLASSES:
        return "unknown"
    return normalized


def _hit_to_tick(onset_sec: float, ticks_per_second: float) -> int:
    return max(0, int(round(onset_sec * ticks_per_second)))


def _resolve_profile_settings(params: AudioDrumExtractionParameters) -> dict[str, float | int]:
    defaults = dict(_PROFILE_DEFAULTS[params.profile])

    overrides: dict[str, float | int | None] = {
        "onset_pre_max": params.onset_pre_max,
        "onset_post_max": params.onset_post_max,
        "onset_pre_avg": params.onset_pre_avg,
        "onset_post_avg": params.onset_post_avg,
        "onset_delta": params.onset_delta,
        "onset_wait": params.onset_wait,
        "min_hit_spacing_ms": params.min_hit_spacing_ms,
        "kick_refractory_ms": params.kick_refractory_ms,
        "snare_refractory_ms": params.snare_refractory_ms,
        "hat_refractory_ms": params.hat_refractory_ms,
        "cymbal_refractory_ms": params.cymbal_refractory_ms,
        "tom_refractory_ms": params.tom_refractory_ms,
        "same_transient_window_ms": params.same_transient_window_ms,
    }

    for key, value in overrides.items():
        if value is not None:
            defaults[key] = value

    return defaults


def _resolve_refractory_ms(settings: dict[str, float | int]) -> dict[str, float]:
    return {
        "kick": float(settings["kick_refractory_ms"]),
        "snare_or_clap": float(settings["snare_refractory_ms"]),
        "hat": float(settings["hat_refractory_ms"]),
        "cymbal": float(settings["cymbal_refractory_ms"]),
        "tom_or_perc": float(settings["tom_refractory_ms"]),
        "unknown": float(settings["tom_refractory_ms"]),
    }


def _is_clear_layered_high_hit(primary: _DetectedHit, candidate: _DetectedHit) -> bool:
    if _canonical_class(primary.class_name) not in {"hat", "cymbal"}:
        return False
    if _canonical_class(candidate.class_name) not in {"hat", "cymbal"}:
        return False
    if _canonical_class(primary.class_name) == _canonical_class(candidate.class_name):
        return False

    delta_sec = abs(candidate.onset_sec - primary.onset_sec)
    if delta_sec < 0.010:
        return False

    strength_ratio = candidate.score() / max(primary.score(), 1e-9)
    if strength_ratio < 0.78:
        return False

    if min(primary.high_band_onset_strength, candidate.high_band_onset_strength) < 0.45:
        return False

    return min(primary.confidence, candidate.confidence) >= 0.65


def _suppress_hit(
    hit: _DetectedHit,
    reason: str,
    *,
    duplicate_suppressed_by_class: dict[str, int],
) -> None:
    if hit.suppressed:
        return
    hit.suppressed = True
    hit.accepted_onset_sec = None
    hit.suppression_reason = reason
    if reason in {"same_transient_group", "global_min_spacing", "class_refractory"}:
        canonical = _canonical_class(hit.class_name)
        duplicate_suppressed_by_class[canonical] = duplicate_suppressed_by_class.get(canonical, 0) + 1


def _group_same_transients(
    hits: list[_DetectedHit],
    *,
    window_sec: float,
    duplicate_suppressed_by_class: dict[str, int],
) -> list[_DetectedHit]:
    if not hits:
        return []

    groups: list[list[_DetectedHit]] = []
    current_group: list[_DetectedHit] = [hits[0]]

    for hit in hits[1:]:
        if (hit.onset_sec - current_group[-1].onset_sec) <= window_sec:
            current_group.append(hit)
        else:
            groups.append(current_group)
            current_group = [hit]
    groups.append(current_group)

    accepted: list[_DetectedHit] = []
    for group_id, group in enumerate(groups):
        for hit in group:
            hit.grouped_transient_id = group_id

        for class_name in ("kick", "snare_or_clap", "tom_or_perc", "unknown"):
            members = [hit for hit in group if _canonical_class(hit.class_name) == class_name]
            if not members:
                continue
            strongest = max(members, key=lambda item: item.score())
            for item in members:
                if item.hit_id != strongest.hit_id:
                    _suppress_hit(
                        item,
                        "same_transient_group",
                        duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                    )

        high_members = [
            hit
            for hit in group
            if _canonical_class(hit.class_name) in {"hat", "cymbal"} and not hit.suppressed
        ]
        if high_members:
            high_members = sorted(high_members, key=lambda item: item.score(), reverse=True)
            allowed_high: list[_DetectedHit] = [high_members[0]]
            for candidate in high_members[1:]:
                if len(allowed_high) >= 2:
                    _suppress_hit(
                        candidate,
                        "same_transient_group",
                        duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                    )
                    continue
                if _is_clear_layered_high_hit(allowed_high[0], candidate):
                    allowed_high.append(candidate)
                    continue
                _suppress_hit(
                    candidate,
                    "same_transient_group",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                )

        accepted.extend([item for item in group if not item.suppressed])

    return sorted(accepted, key=lambda item: item.onset_sec)


def _apply_global_min_spacing(
    hits: list[_DetectedHit],
    *,
    min_spacing_sec: float,
    duplicate_suppressed_by_class: dict[str, int],
) -> list[_DetectedHit]:
    if not hits:
        return []

    accepted: list[_DetectedHit] = []
    for hit in sorted(hits, key=lambda item: item.onset_sec):
        if not accepted:
            accepted.append(hit)
            continue

        previous = accepted[-1]
        if (hit.onset_sec - previous.onset_sec) < min_spacing_sec:
            if hit.score() > previous.score():
                _suppress_hit(
                    previous,
                    "global_min_spacing",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                )
                accepted[-1] = hit
            else:
                _suppress_hit(
                    hit,
                    "global_min_spacing",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                )
            continue

        accepted.append(hit)

    return sorted([item for item in accepted if not item.suppressed], key=lambda item: item.onset_sec)


def _apply_class_refractory(
    hits: list[_DetectedHit],
    *,
    class_refractory_sec: dict[str, float],
    duplicate_suppressed_by_class: dict[str, int],
) -> list[_DetectedHit]:
    accepted: list[_DetectedHit] = []
    last_by_class_index: dict[str, int] = {}

    for hit in sorted(hits, key=lambda item: item.onset_sec):
        canonical = _canonical_class(hit.class_name)
        refractory_sec = float(class_refractory_sec.get(canonical, 0.0))
        hit.class_refractory_ms = refractory_sec * 1000.0

        last_index = last_by_class_index.get(canonical)
        if last_index is None:
            accepted.append(hit)
            last_by_class_index[canonical] = len(accepted) - 1
            continue

        previous = accepted[last_index]
        if (hit.onset_sec - previous.onset_sec) < refractory_sec:
            if hit.score() > previous.score():
                _suppress_hit(
                    previous,
                    "class_refractory",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                )
                accepted[last_index] = hit
            else:
                _suppress_hit(
                    hit,
                    "class_refractory",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                )
            continue

        accepted.append(hit)
        last_by_class_index[canonical] = len(accepted) - 1

    return sorted([item for item in accepted if not item.suppressed], key=lambda item: item.onset_sec)


def _assign_output_velocities(hits: list[_DetectedHit]) -> None:
    if not hits:
        return

    strengths = np.asarray(
        [hit.onset_strength * (0.75 + 0.25 * hit.confidence) for hit in hits],
        dtype=np.float64,
    )
    low = float(np.percentile(strengths, 15))
    high = float(np.percentile(strengths, 92))
    span = max(high - low, 1e-9)

    for hit, value in zip(hits, strengths.tolist()):
        normalized = np.clip((value - low) / span, 0.0, 1.0)
        confidence_gain = np.clip((float(hit.confidence) - 0.45) / 0.55, 0.0, 1.0)
        velocity = int(round(22.0 + (84.0 * normalized) + (16.0 * confidence_gain)))
        hit.velocity = max(1, min(127, velocity))


def _build_midi(
    hits: list[_DetectedHit],
    *,
    output_path: Path,
    ticks_per_beat: int,
    bpm_used: float,
    duration_sec: float,
    channel: int,
    track_name: str,
) -> tuple[int, int]:
    tempo_us_per_beat = int(round(60_000_000.0 / max(bpm_used, 1e-9)))
    ticks_per_second = (ticks_per_beat * bpm_used) / 60.0
    source_length_ticks = max(0, int(round(duration_sec * ticks_per_second)))

    midi_file = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us_per_beat, time=0))

    absolute_events: list[tuple[int, int, mido.Message]] = []
    for hit in hits:
        start_tick = _hit_to_tick(hit.onset_sec, ticks_per_second)
        duration_ticks = max(
            1,
            int(round(_CLASS_NOTE_DURATIONS_SEC[hit.class_name] * ticks_per_second)),
        )
        end_tick = min(source_length_ticks, start_tick + duration_ticks)
        if end_tick <= start_tick:
            if start_tick >= source_length_ticks:
                continue
            end_tick = min(source_length_ticks, start_tick + 1)

        absolute_events.append(
            (
                start_tick,
                1,
                mido.Message(
                    "note_on",
                    note=int(hit.target_note),
                    velocity=int(hit.velocity),
                    channel=channel,
                    time=0,
                ),
            )
        )
        absolute_events.append(
            (
                end_tick,
                0,
                mido.Message(
                    "note_off",
                    note=int(hit.target_note),
                    velocity=0,
                    channel=channel,
                    time=0,
                ),
            )
        )

    absolute_events.sort(key=lambda item: (item[0], item[1]))

    previous_tick = 0
    for tick, _priority, message in absolute_events:
        track.append(message.copy(time=tick - previous_tick))
        previous_tick = tick

    final_tick = max(source_length_ticks, previous_tick)
    track.append(mido.MetaMessage("end_of_track", time=final_tick - previous_tick))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_file.save(str(output_path))

    return source_length_ticks, final_tick


def _write_debug_csv(path: Path, hits: list[_DetectedHit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "onset_sec",
                "raw_onset_sec",
                "accepted_onset_sec",
                "suppressed",
                "suppression_reason",
                "grouped_transient_id",
                "class_refractory_ms",
                "nearest_previous_same_class_ms",
                "tick",
                "class",
                "target_note",
                "velocity",
                "confidence",
                "low_energy_ratio",
                "mid_energy_ratio",
                "high_energy_ratio",
                "spectral_centroid",
                "onset_strength",
            ],
        )
        writer.writeheader()
        for hit in hits:
            writer.writerow(
                {
                    "onset_sec": hit.onset_sec,
                    "raw_onset_sec": hit.onset_sec,
                    "accepted_onset_sec": hit.accepted_onset_sec,
                    "suppressed": str(hit.suppressed).lower(),
                    "suppression_reason": hit.suppression_reason,
                    "grouped_transient_id": hit.grouped_transient_id,
                    "class_refractory_ms": hit.class_refractory_ms,
                    "nearest_previous_same_class_ms": hit.nearest_previous_same_class_ms,
                    "tick": hit.tick,
                    "onset_strength": hit.onset_strength,
                    "class": hit.class_name,
                    "target_note": hit.target_note,
                    "velocity": hit.velocity,
                    "confidence": hit.confidence,
                    "low_energy_ratio": hit.low_energy_ratio,
                    "mid_energy_ratio": hit.mid_energy_ratio,
                    "high_energy_ratio": hit.high_energy_ratio,
                    "spectral_centroid": hit.spectral_centroid,
                }
            )


def _validate_params(params: AudioDrumExtractionParameters) -> None:
    if params.channel < 0 or params.channel > 15:
        raise AudioDrumExtractionError("channel must be in range 0..15.")

    if params.min_onset_strength < 0.0:
        raise AudioDrumExtractionError("--min-onset-strength must be >= 0.")

    if params.bpm is not None and params.bpm <= 0.0:
        raise AudioDrumExtractionError("--bpm must be > 0.")

    if params.snare_target not in {"sn1", "sn2", "clap"}:
        raise AudioDrumExtractionError("--snare-target must be one of: sn1, sn2, clap.")

    if params.c1_midi_note < 0 or params.c1_midi_note > 127:
        raise AudioDrumExtractionError("--c1-midi-note must be in range 0..127.")

    if params.profile not in {"conservative", "balanced", "sensitive"}:
        raise AudioDrumExtractionError("--profile must be one of: conservative, balanced, sensitive.")

    settings = _resolve_profile_settings(params)
    int_fields = (
        "onset_pre_max",
        "onset_post_max",
        "onset_pre_avg",
        "onset_post_avg",
        "onset_wait",
    )
    for name in int_fields:
        if int(settings[name]) < 0:
            raise AudioDrumExtractionError(f"--{name.replace('_', '-')} must be >= 0.")

    float_fields = (
        "onset_delta",
        "min_hit_spacing_ms",
        "kick_refractory_ms",
        "snare_refractory_ms",
        "hat_refractory_ms",
        "cymbal_refractory_ms",
        "tom_refractory_ms",
        "same_transient_window_ms",
    )
    for name in float_fields:
        if float(settings[name]) < 0.0:
            raise AudioDrumExtractionError(f"--{name.replace('_', '-')} must be >= 0.")


def _count_output_notes(hits: list[_DetectedHit]) -> dict[str, int]:
    counts: dict[int, int] = {}
    for hit in hits:
        if int(hit.target_note) < 0:
            continue
        note = int(hit.target_note)
        counts[note] = counts.get(note, 0) + 1
    return {str(note): count for note, count in sorted(counts.items())}


def _count_classes(hits: list[_DetectedHit]) -> dict[str, int]:
    counts: dict[str, int] = {
        "kick": 0,
        "snare_or_clap": 0,
        "hat": 0,
        "cymbal": 0,
        "tom_or_perc": 0,
        "unknown": 0,
    }
    for hit in hits:
        normalized = _canonical_class(hit.class_name)
        counts[normalized] += 1
    return counts


def _class_notes_per_second(class_counts: dict[str, int], duration_sec: float) -> dict[str, float]:
    safe_duration = max(duration_sec, 1e-9)
    return {
        name: float(count) / safe_duration
        for name, count in class_counts.items()
    }


def _velocity_summary(hits: list[_DetectedHit]) -> dict[str, float]:
    if not hits:
        return {
            "min": 0.0,
            "median": 0.0,
            "max": 0.0,
            "p90": 0.0,
        }

    velocities = np.asarray([int(hit.velocity) for hit in hits], dtype=np.float64)
    return {
        "min": float(np.min(velocities)),
        "median": float(np.median(velocities)),
        "max": float(np.max(velocities)),
        "p90": float(np.percentile(velocities, 90)),
    }


def _duplicate_interval_summary(
    hits: list[_DetectedHit],
    class_refractory_ms: dict[str, float],
) -> dict[str, dict[str, float | int | None]]:
    by_class: dict[str, list[float]] = {
        "kick": [],
        "snare_or_clap": [],
        "hat": [],
        "cymbal": [],
        "tom_or_perc": [],
        "unknown": [],
    }

    for hit in sorted(hits, key=lambda item: item.onset_sec):
        by_class[_canonical_class(hit.class_name)].append(float(hit.onset_sec))

    summary: dict[str, dict[str, float | int | None]] = {}
    for class_name, onsets in by_class.items():
        if len(onsets) < 2:
            summary[class_name] = {
                "min_interval_ms": None,
                "median_interval_ms": None,
                "count_under_80ms": 0,
                "count_under_class_refractory": 0,
                "interval_count": 0,
            }
            continue

        intervals_ms = np.diff(np.asarray(onsets, dtype=np.float64)) * 1000.0
        refractory = float(class_refractory_ms.get(class_name, 0.0))
        summary[class_name] = {
            "min_interval_ms": float(np.min(intervals_ms)),
            "median_interval_ms": float(np.median(intervals_ms)),
            "count_under_80ms": int(np.sum(intervals_ms < 80.0)),
            "count_under_class_refractory": int(np.sum(intervals_ms < refractory)),
            "interval_count": int(len(intervals_ms)),
        }

    return summary


def _evaluate_density_warnings(
    *,
    duration_sec: float,
    class_counts: dict[str, int],
    duplicate_interval_summary: dict[str, dict[str, float | int | None]],
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    too_dense = False

    if class_counts.get("kick", 0) > 700:
        too_dense = True
        warnings.append(
            "Strong warning: kick count exceeds 700; output is likely too dense."
        )

    if class_counts.get("cymbal", 0) > 400:
        too_dense = True
        warnings.append(
            "Strong warning: cymbal count exceeds 400; output is likely too dense."
        )

    for class_name, item in duplicate_interval_summary.items():
        interval_count = int(item.get("interval_count") or 0)
        under_ref = int(item.get("count_under_class_refractory") or 0)
        if interval_count <= 0:
            continue
        ratio = under_ref / float(interval_count)
        if ratio > 0.10:
            too_dense = True
            warnings.append(
                "Strong warning: more than 10% of "
                f"{class_name} intervals are under class refractory."
            )

    notes_total = sum(class_counts.values())
    notes_per_second = notes_total / max(duration_sec, 1e-9)
    if notes_per_second > 14.0:
        warnings.append(
            "Warning: overall notes_per_second is unusually high for drum extraction."
        )

    return too_dense, warnings


def _to_per_hit_summary(hits: list[_DetectedHit]) -> list[PerHitSummary]:
    return [
        PerHitSummary(
            onset_sec=float(hit.onset_sec),
            raw_onset_sec=float(hit.onset_sec),
            accepted_onset_sec=(
                None if hit.accepted_onset_sec is None else float(hit.accepted_onset_sec)
            ),
            tick=int(hit.tick),
            class_name=_normalize_class_name(hit.class_name),
            target_note=int(hit.target_note),
            velocity=int(hit.velocity),
            confidence=float(hit.confidence),
            low_energy_ratio=float(hit.low_energy_ratio),
            mid_energy_ratio=float(hit.mid_energy_ratio),
            high_energy_ratio=float(hit.high_energy_ratio),
            spectral_centroid=float(hit.spectral_centroid),
            onset_strength=float(hit.onset_strength),
            suppressed=bool(hit.suppressed),
            suppression_reason=hit.suppression_reason,
            grouped_transient_id=hit.grouped_transient_id,
            class_refractory_ms=float(hit.class_refractory_ms),
            nearest_previous_same_class_ms=(
                None
                if hit.nearest_previous_same_class_ms is None
                else float(hit.nearest_previous_same_class_ms)
            ),
        )
        for hit in hits
    ]


def _adjust_class_with_band_evidence(
    *,
    class_name: DrumClass,
    onset_strength: float,
    low_band_onset_strength: float,
    mid_band_onset_strength: float,
    high_band_onset_strength: float,
    low_ratio: float,
    mid_ratio: float,
    high_ratio: float,
    centroid_hz: float,
    previous_high_hit: _DetectedHit | None,
    onset_sec: float,
) -> DrumClass:
    normalized = _normalize_class_name(class_name)

    if normalized == "kick":
        if low_band_onset_strength < 0.20 and mid_band_onset_strength > (low_band_onset_strength + 0.08):
            return "tom_or_perc"

    if normalized in {"hat", "cymbal"}:
        if high_band_onset_strength < 0.16 and mid_ratio >= 0.45:
            return "snare_or_clap"

        if normalized == "cymbal":
            if high_band_onset_strength < 0.30 and onset_strength < 0.65:
                return "hat"
            if centroid_hz < 5200.0 and high_ratio < 0.62:
                return "hat"
            if previous_high_hit is not None:
                delta_sec = onset_sec - previous_high_hit.onset_sec
                if 0.0 < delta_sec < 0.14 and onset_strength < (previous_high_hit.onset_strength * 0.82):
                    return "hat"

    if normalized == "tom_or_perc":
        if low_ratio > 0.55 and low_band_onset_strength > (mid_band_onset_strength + 0.10):
            return "kick"

    return normalized


def extract_drums_from_audio(
    *,
    wav_file: Path,
    params: AudioDrumExtractionParameters,
) -> AudioDrumExtractionReport:
    _validate_params(params)

    if not wav_file.exists() or not wav_file.is_file():
        raise AudioDrumExtractionError(f"Input WAV file does not exist: {wav_file}")

    source_audio_bytes = wav_file.read_bytes()
    try:
        audio, sample_rate = sf.read(str(wav_file), always_2d=False)
    except Exception as exc:  # pragma: no cover
        raise AudioDrumExtractionError(f"Failed to read WAV file: {wav_file}") from exc

    mono = _ensure_mono(np.asarray(audio))
    duration_sec = float(len(mono) / sample_rate) if sample_rate > 0 else 0.0

    map_definition = _resolve_target_map_definition(params)
    class_targets = _resolve_class_targets(map_definition, params)
    settings = _resolve_profile_settings(params)

    onset_envelopes, _frame_size, hop_size = _onset_strength_envelopes(
        mono,
        sample_rate,
    )
    onset_times, onset_strengths = _detect_onsets(
        onset_envelopes["full"],
        sample_rate,
        hop_size,
        params.min_onset_strength,
        onset_pre_max=int(settings["onset_pre_max"]),
        onset_post_max=int(settings["onset_post_max"]),
        onset_pre_avg=int(settings["onset_pre_avg"]),
        onset_post_avg=int(settings["onset_post_avg"]),
        onset_delta=float(settings["onset_delta"]),
        onset_wait=int(settings["onset_wait"]),
    )

    detected_bpm = _estimate_bpm(onset_times)
    bpm_used = float(params.bpm) if params.bpm is not None else float(detected_bpm or DEFAULT_BPM)
    bpm_source: Literal["detected", "forced"] = "forced" if params.bpm is not None else "detected"
    ticks_per_second = (params.ticks_per_beat * bpm_used) / 60.0

    warnings: list[str] = []
    if detected_bpm is None:
        warnings.append("Could not confidently estimate BPM from onsets; defaulted to stable fallback.")
    if len(onset_times) == 0:
        warnings.append("No onsets detected above threshold.")

    class_refractory_ms = _resolve_refractory_ms(settings)
    class_refractory_sec = {
        class_name: value / 1000.0 for class_name, value in class_refractory_ms.items()
    }
    min_hit_spacing_sec = float(settings["min_hit_spacing_ms"]) / 1000.0
    same_transient_window_sec = float(settings["same_transient_window_ms"]) / 1000.0

    raw_hits: list[_DetectedHit] = []
    duplicate_suppressed_by_class: dict[str, int] = {
        "kick": 0,
        "snare_or_clap": 0,
        "hat": 0,
        "cymbal": 0,
        "tom_or_perc": 0,
        "unknown": 0,
    }

    last_seen_same_class_time: dict[str, float] = {}
    previous_high_hit: _DetectedHit | None = None

    for hit_id, (onset_sec, onset_strength) in enumerate(
        zip(onset_times.tolist(), onset_strengths.tolist())
    ):
        frame_idx = int(round((float(onset_sec) * sample_rate) / hop_size))
        frame_idx = max(0, min(frame_idx, len(onset_envelopes["full"]) - 1))

        low_onset_strength = float(onset_envelopes["low"][frame_idx])
        mid_onset_strength = float(onset_envelopes["mid"][frame_idx])
        high_onset_strength = float(onset_envelopes["high"][frame_idx])

        (
            low_ratio,
            mid_ratio,
            high_ratio,
            centroid_hz,
            rolloff_hz,
            _rms,
            _peak,
        ) = _extract_hit_spectral_features(
            mono,
            sample_rate,
            float(onset_sec),
        )

        class_name, confidence = _classify_hit(
            onset_strength=float(onset_strength),
            low_ratio=low_ratio,
            mid_ratio=mid_ratio,
            high_ratio=high_ratio,
            centroid_hz=centroid_hz,
            rolloff_hz=rolloff_hz,
        )

        class_name = _adjust_class_with_band_evidence(
            class_name=class_name,
            onset_strength=float(onset_strength),
            low_band_onset_strength=low_onset_strength,
            mid_band_onset_strength=mid_onset_strength,
            high_band_onset_strength=high_onset_strength,
            low_ratio=low_ratio,
            mid_ratio=mid_ratio,
            high_ratio=high_ratio,
            centroid_hz=centroid_hz,
            previous_high_hit=previous_high_hit,
            onset_sec=float(onset_sec),
        )

        class_name = _normalize_class_name(class_name)

        target_note = int(class_targets[class_name])
        if class_name == "tom_or_perc" and params.target_map == "ujam-candy" and high_ratio > 0.45:
            candy_layout = resolve_ujam_candy_layout_notes(params.c1_midi_note)
            target_note = candy_layout["F2"]

        canonical_class = _canonical_class(class_name)
        previous_time = last_seen_same_class_time.get(canonical_class)
        nearest_previous_same_class_ms = (
            None
            if previous_time is None
            else (float(onset_sec) - previous_time) * 1000.0
        )
        last_seen_same_class_time[canonical_class] = float(onset_sec)

        hit = _DetectedHit(
            hit_id=int(hit_id),
            onset_sec=float(onset_sec),
            accepted_onset_sec=float(onset_sec),
            tick=_hit_to_tick(float(onset_sec), ticks_per_second),
            onset_strength=float(onset_strength),
            low_band_onset_strength=low_onset_strength,
            mid_band_onset_strength=mid_onset_strength,
            high_band_onset_strength=high_onset_strength,
            class_name=class_name,
            low_energy_ratio=float(low_ratio),
            mid_energy_ratio=float(mid_ratio),
            high_energy_ratio=float(high_ratio),
            spectral_centroid=float(centroid_hz),
            confidence=float(confidence),
            target_note=target_note,
            velocity=1,
            suppressed=False,
            suppression_reason=None,
            grouped_transient_id=None,
            class_refractory_ms=float(class_refractory_ms[canonical_class]),
            nearest_previous_same_class_ms=nearest_previous_same_class_ms,
        )

        raw_hits.append(hit)

        if canonical_class in {"hat", "cymbal"}:
            previous_high_hit = hit

    grouped_hits = _group_same_transients(
        raw_hits,
        window_sec=same_transient_window_sec,
        duplicate_suppressed_by_class=duplicate_suppressed_by_class,
    )
    spacing_filtered_hits = _apply_global_min_spacing(
        grouped_hits,
        min_spacing_sec=min_hit_spacing_sec,
        duplicate_suppressed_by_class=duplicate_suppressed_by_class,
    )
    refractory_filtered_hits = _apply_class_refractory(
        spacing_filtered_hits,
        class_refractory_sec=class_refractory_sec,
        duplicate_suppressed_by_class=duplicate_suppressed_by_class,
    )

    emitted_hits: list[_DetectedHit] = []
    skipped_unknown_count = 0
    for hit in sorted(refractory_filtered_hits, key=lambda item: item.onset_sec):
        if _canonical_class(hit.class_name) == "unknown" and float(hit.confidence) < 0.80:
            skipped_unknown_count += 1
            hit.target_note = -1
            _suppress_hit(
                hit,
                "low_confidence_unknown",
                duplicate_suppressed_by_class=duplicate_suppressed_by_class,
            )
            continue
        emitted_hits.append(hit)

    _assign_output_velocities(emitted_hits)

    suppressed_duplicate_count = int(sum(duplicate_suppressed_by_class.values()))

    if skipped_unknown_count > 0:
        warnings.append(
            f"Skipped {skipped_unknown_count} low-confidence unknown hits from MIDI output."
        )

    class_counts = _count_classes(emitted_hits)
    duplicate_interval_summary = _duplicate_interval_summary(emitted_hits, class_refractory_ms)
    too_dense_warning, density_warnings = _evaluate_density_warnings(
        duration_sec=duration_sec,
        class_counts=class_counts,
        duplicate_interval_summary=duplicate_interval_summary,
    )
    warnings.extend(density_warnings)

    if params.debug_csv is not None:
        _write_debug_csv(params.debug_csv, sorted(raw_hits, key=lambda item: item.onset_sec))

    output_file = params.output_file
    synchronization_preserved = True

    if not params.dry_run:
        track_name = f"drums_from_audio_{params.target_map.replace('-', '_')}"
        source_length_ticks, output_length_ticks = _build_midi(
            emitted_hits,
            output_path=output_file,
            ticks_per_beat=params.ticks_per_beat,
            bpm_used=bpm_used,
            duration_sec=duration_sec,
            channel=params.channel,
            track_name=track_name,
        )
        synchronization_preserved = source_length_ticks == output_length_ticks

        if params.separate_files:
            by_class: dict[str, list[_DetectedHit]] = {
                "kick": [],
                "snare_or_clap": [],
                "hat": [],
                "cymbal": [],
                "tom_or_perc": [],
            }
            for hit in emitted_hits:
                bucket = _canonical_class(hit.class_name)
                if bucket == "unknown":
                    bucket = "tom_or_perc"
                by_class[bucket].append(hit)

            for class_name, file_name in _CLASS_FILE_NAMES.items():
                class_track_name = (
                    f"drums_from_audio_{params.target_map.replace('-', '_')}_{class_name}"
                )
                class_output = output_file.parent / file_name
                class_source_ticks, class_output_ticks = _build_midi(
                    by_class[class_name],
                    output_path=class_output,
                    ticks_per_beat=params.ticks_per_beat,
                    bpm_used=bpm_used,
                    duration_sec=duration_sec,
                    channel=params.channel,
                    track_name=class_track_name,
                )
                synchronization_preserved = (
                    synchronization_preserved
                    and class_source_ticks == source_length_ticks
                    and class_output_ticks == output_length_ticks
                )
    else:
        ticks_per_second = (params.ticks_per_beat * bpm_used) / 60.0
        source_length_ticks = max(0, int(round(duration_sec * ticks_per_second)))
        output_length_ticks = source_length_ticks
        synchronization_preserved = source_length_ticks == output_length_ticks

    safe_duration = max(duration_sec, 1e-9)
    notes_per_second = float(sum(class_counts.values()) / safe_duration)

    report = AudioDrumExtractionReport(
        wav_file=str(wav_file),
        output_file=None if params.dry_run else str(output_file),
        duration_sec=float(duration_sec),
        sample_rate=int(sample_rate),
        detected_bpm=float(detected_bpm) if detected_bpm is not None else None,
        bpm_used=float(bpm_used),
        bpm_source=bpm_source,
        onset_count=len(onset_times),
        raw_onset_count=len(raw_hits),
        accepted_onset_count=len(emitted_hits),
        suppressed_duplicate_count=suppressed_duplicate_count,
        suppressed_by_class=duplicate_suppressed_by_class,
        min_hit_spacing_ms=float(settings["min_hit_spacing_ms"]),
        class_refractory_ms={
            "kick": float(class_refractory_ms["kick"]),
            "snare_or_clap": float(class_refractory_ms["snare_or_clap"]),
            "hat": float(class_refractory_ms["hat"]),
            "cymbal": float(class_refractory_ms["cymbal"]),
            "tom_or_perc": float(class_refractory_ms["tom_or_perc"]),
        },
        same_transient_window_ms=float(settings["same_transient_window_ms"]),
        class_counts=class_counts,
        output_note_counts=_count_output_notes(emitted_hits),
        output_pitch_counts=_count_output_notes(emitted_hits),
        notes_per_second=notes_per_second,
        class_notes_per_second=_class_notes_per_second(class_counts, duration_sec),
        velocity_summary=_velocity_summary(emitted_hits),
        too_dense_warning=too_dense_warning,
        duplicate_interval_summary=duplicate_interval_summary,
        target_map=map_definition.name,
        c1_midi_note=int(params.c1_midi_note),
        synchronization_preserved=bool(synchronization_preserved),
        warnings=warnings,
        per_hit_summary=_to_per_hit_summary(sorted(raw_hits, key=lambda item: item.onset_sec)),
    )

    if params.report_file is not None:
        params.report_file.parent.mkdir(parents=True, exist_ok=True)
        params.report_file.write_text(
            json.dumps(report.to_json_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    if wav_file.read_bytes() != source_audio_bytes:
        raise AudioDrumExtractionError("Source WAV file was modified unexpectedly.")

    return report
