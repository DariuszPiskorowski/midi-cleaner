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
DetectionMode = Literal["global", "multi-detector"]
DetectorName = Literal["global", "kick", "snare", "hat", "cymbal", "tom"]

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
        "cymbal_refractory_ms": 320.0,
        "tom_refractory_ms": 140.0,
        "same_transient_window_ms": 35.0,
        "min_class_confidence": 0.60,
    },
    "balanced": {
        "onset_pre_max": 1,
        "onset_post_max": 1,
        "onset_pre_avg": 5,
        "onset_post_avg": 5,
        "onset_delta": 0.08,
        "onset_wait": 1,
        "min_hit_spacing_ms": 60.0,
        "kick_refractory_ms": 140.0,
        "snare_refractory_ms": 120.0,
        "hat_refractory_ms": 55.0,
        "cymbal_refractory_ms": 240.0,
        "tom_refractory_ms": 100.0,
        "same_transient_window_ms": 35.0,
        "min_class_confidence": 0.55,
    },
    "sensitive": {
        "onset_pre_max": 1,
        "onset_post_max": 1,
        "onset_pre_avg": 4,
        "onset_post_avg": 4,
        "onset_delta": 0.05,
        "onset_wait": 1,
        "min_hit_spacing_ms": 45.0,
        "kick_refractory_ms": 110.0,
        "snare_refractory_ms": 95.0,
        "hat_refractory_ms": 40.0,
        "cymbal_refractory_ms": 180.0,
        "tom_refractory_ms": 80.0,
        "same_transient_window_ms": 30.0,
        "min_class_confidence": 0.50,
    },
}


def _default_detector_thresholds() -> dict[str, float]:
    return {
        "kick": 0.24,
        "snare": 0.22,
        "hat": 0.22,
        "cymbal": 0.28,
        "tom": 0.24,
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
    detection_mode: DetectionMode = "multi-detector"
    min_class_confidence: float | None = None
    emit_unknown: bool = False
    unknown_target_note: int | None = None
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
    detection_mode: str
    detector_name: str
    candidate_class: str
    accepted_class: str | None
    class_confidence: float
    competing_class: str | None
    competing_class_score: float
    low_peak_strength: float
    mid_peak_strength: float
    high_peak_strength: float
    attack_score: float
    decay_score: float
    band_dominance_score: float
    accepted: bool
    rejection_reason: str | None
    merged_with_transient_id: int | None


@dataclass
class AudioDrumExtractionReport:
    wav_file: str
    output_file: str | None
    duration_sec: float
    sample_rate: int
    detected_bpm: float | None
    bpm_used: float
    bpm_source: Literal["detected", "forced"]
    detection_mode: DetectionMode
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
    detector_candidate_counts: dict[str, int]
    detector_accepted_counts: dict[str, int]
    detector_rejected_counts: dict[str, int]
    low_confidence_rejected_count: int
    rejected_by_reason: dict[str, int]
    multi_detector_merge_conflicts: int
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
                "detection_mode": item.detection_mode,
                "detector_name": item.detector_name,
                "candidate_class": item.candidate_class,
                "accepted_class": item.accepted_class,
                "class_confidence": item.class_confidence,
                "competing_class": item.competing_class,
                "competing_class_score": item.competing_class_score,
                "low_peak_strength": item.low_peak_strength,
                "mid_peak_strength": item.mid_peak_strength,
                "high_peak_strength": item.high_peak_strength,
                "attack_score": item.attack_score,
                "decay_score": item.decay_score,
                "band_dominance_score": item.band_dominance_score,
                "accepted": item.accepted,
                "rejection_reason": item.rejection_reason,
                "merged_with_transient_id": item.merged_with_transient_id,
            }
            for item in self.per_hit_summary
        ]
        return payload


@dataclass
class _HitCandidate:
    candidate_id: int
    detector_name: DetectorName
    onset_sec: float
    onset_strength: float
    class_name: DrumClass
    low_peak_strength: float
    mid_peak_strength: float
    high_peak_strength: float
    attack_score: float
    decay_score: float
    band_dominance_score: float
    confidence: float
    competing_class: DrumClass | None
    competing_class_score: float
    low_energy_ratio: float
    mid_energy_ratio: float
    high_energy_ratio: float
    spectral_centroid: float


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
    detection_mode: DetectionMode
    detector_name: DetectorName
    candidate_class: DrumClass
    accepted_class: DrumClass | None
    class_confidence: float
    competing_class: DrumClass | None
    competing_class_score: float
    low_peak_strength: float
    mid_peak_strength: float
    high_peak_strength: float
    attack_score: float
    decay_score: float
    band_dominance_score: float
    accepted: bool
    rejection_reason: str | None
    merged_with_transient_id: int | None

    def score(self) -> float:
        return (
            0.52 * float(self.onset_strength)
            + 0.26 * float(self.class_confidence)
            + 0.10 * float(self.attack_score)
            + 0.12 * float(self.band_dominance_score)
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
            "upper": zeros,
            "kick": zeros,
            "snare": zeros,
            "hat": zeros,
            "cymbal": zeros,
            "tom": zeros,
        }, frame_size, hop_size

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sample_rate))
    diff = np.maximum(0.0, np.diff(magnitudes, axis=0))

    low_mask = (freqs >= 20.0) & (freqs < 180.0)
    mid_mask = (freqs >= 180.0) & (freqs < 3000.0)
    upper_mask = (freqs >= 3500.0) & (freqs < 9000.0)
    high_mask = (freqs >= 5000.0) & (freqs < 12000.0)
    kick_mask = (freqs >= 30.0) & (freqs < 180.0)
    snare_mask = (freqs >= 180.0) & (freqs < 9000.0)
    hat_mask = (freqs >= 5000.0) & (freqs < 12000.0)
    cymbal_mask = (freqs >= 4500.0) & (freqs < 12000.0)
    tom_mask = (freqs >= 120.0) & (freqs < 2200.0)

    def _raw(mask: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.sum(diff[:, mask], axis=1).astype(np.float64)))

    return {
        "full": _normalize_envelope(_raw(np.ones_like(freqs, dtype=bool))),
        "low": _normalize_envelope(_raw(low_mask)),
        "mid": _normalize_envelope(_raw(mid_mask)),
        "upper": _normalize_envelope(_raw(upper_mask)),
        "high": _normalize_envelope(_raw(high_mask)),
        "kick": _normalize_envelope(_raw(kick_mask)),
        "snare": _normalize_envelope(_raw(snare_mask)),
        "hat": _normalize_envelope(_raw(hat_mask)),
        "cymbal": _normalize_envelope(_raw(cymbal_mask)),
        "tom": _normalize_envelope(_raw(tom_mask)),
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
    mid_energy = _band_energy(spectrum, freqs, 160.0, 3500.0)
    high_energy = _band_energy(spectrum, freqs, 3500.0, 12000.0)
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
        "min_class_confidence": params.min_class_confidence,
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

    if params.detection_mode not in {"global", "multi-detector"}:
        raise AudioDrumExtractionError("--detection-mode must be one of: global, multi-detector.")

    if params.unknown_target_note is not None and (params.unknown_target_note < 0 or params.unknown_target_note > 127):
        raise AudioDrumExtractionError("--unknown-target-note must be in range 0..127.")

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
        "min_class_confidence",
    )
    for name in float_fields:
        if float(settings[name]) < 0.0:
            raise AudioDrumExtractionError(f"--{name.replace('_', '-')} must be >= 0.")


def _attack_decay_scores(envelope: np.ndarray, idx: int) -> tuple[float, float]:
    pre = float(np.mean(envelope[max(0, idx - 3): idx + 1]))
    post_short = float(np.mean(envelope[idx: min(len(envelope), idx + 3)]))
    post_long = float(np.mean(envelope[idx: min(len(envelope), idx + 10)]))

    attack = np.clip((post_short - pre) / max(post_short + 1e-9, 1e-9), 0.0, 1.0)
    decay = np.clip(post_long / max(post_short + 1e-9, 1e-9), 0.0, 1.0)
    return float(attack), float(decay)


def _collect_detector_candidates(
    *,
    detector_name: DetectorName,
    class_name: DrumClass,
    envelope: np.ndarray,
    all_env: dict[str, np.ndarray],
    sample_rate: int,
    hop_size: int,
    settings: dict[str, float | int],
    threshold: float,
    candidate_id_start: int,
) -> tuple[list[_HitCandidate], int]:
    times, strengths = _detect_onsets(
        envelope,
        sample_rate,
        hop_size,
        min_onset_strength=threshold,
        onset_pre_max=int(settings["onset_pre_max"]),
        onset_post_max=int(settings["onset_post_max"]),
        onset_pre_avg=int(settings["onset_pre_avg"]),
        onset_post_avg=int(settings["onset_post_avg"]),
        onset_delta=float(settings["onset_delta"]),
        onset_wait=int(settings["onset_wait"]),
    )

    candidates: list[_HitCandidate] = []
    next_id = candidate_id_start
    for onset_sec, onset_strength in zip(times.tolist(), strengths.tolist()):
        idx = int(round((float(onset_sec) * sample_rate) / hop_size))
        idx = max(0, min(idx, len(all_env["full"]) - 1))

        low_peak = float(all_env["low"][idx])
        mid_peak = float(all_env["mid"][idx])
        high_peak = float(all_env["high"][idx])
        upper_peak = float(all_env["upper"][idx])

        attack_score, decay_score = _attack_decay_scores(envelope, idx)

        band_total = low_peak + mid_peak + high_peak + 1e-9
        if class_name == "kick":
            dominance = low_peak / band_total
            comp_class: DrumClass | None = "tom_or_perc"
            comp_score = mid_peak * 0.65
            confidence = 0.48 * float(onset_strength) + 0.34 * dominance + 0.18 * attack_score
            confidence -= 0.10 * decay_score if decay_score > 0.92 else 0.0
        elif class_name == "snare_or_clap":
            snare_band = min(1.5, (mid_peak + 0.7 * upper_peak))
            dominance = snare_band / max(snare_band + low_peak + 0.5 * high_peak, 1e-9)
            comp_class = "cymbal"
            comp_score = 0.55 * high_peak
            confidence = 0.46 * float(onset_strength) + 0.30 * dominance + 0.24 * attack_score
        elif class_name == "hat":
            dominance = high_peak / max(mid_peak + high_peak + 1e-9, 1e-9)
            comp_class = "cymbal"
            comp_score = 0.55 * high_peak + 0.25 * decay_score
            confidence = 0.44 * float(onset_strength) + 0.32 * dominance + 0.24 * attack_score
        elif class_name == "cymbal":
            dominance = (high_peak + 0.4 * upper_peak) / max(low_peak + mid_peak + high_peak + 1e-9, 1e-9)
            comp_class = "hat"
            comp_score = 0.55 * high_peak + 0.25 * attack_score
            confidence = 0.42 * float(onset_strength) + 0.28 * dominance + 0.30 * decay_score
        else:
            tom_dom = (0.6 * low_peak + mid_peak) / max(low_peak + mid_peak + high_peak + 1e-9, 1e-9)
            dominance = tom_dom
            comp_class = "kick"
            comp_score = 0.65 * low_peak
            confidence = 0.44 * float(onset_strength) + 0.34 * dominance + 0.22 * attack_score

        if class_name == "tom_or_perc" and high_peak > 0.70:
            confidence -= 0.12
        if class_name == "snare_or_clap" and high_peak > 0.85 and mid_peak < 0.30:
            confidence -= 0.09
        if class_name == "cymbal" and attack_score > 0.70 and decay_score < 0.35:
            confidence -= 0.12

        confidence = float(np.clip(confidence, 0.0, 0.99))

        low_ratio = low_peak / band_total
        mid_ratio = mid_peak / band_total
        high_ratio = high_peak / band_total

        centroid_proxy = (
            low_ratio * 120.0
            + mid_ratio * 1600.0
            + high_ratio * 7000.0
        )

        candidates.append(
            _HitCandidate(
                candidate_id=int(next_id),
                detector_name=detector_name,
                onset_sec=float(onset_sec),
                onset_strength=float(onset_strength),
                class_name=class_name,
                low_peak_strength=low_peak,
                mid_peak_strength=mid_peak,
                high_peak_strength=high_peak,
                attack_score=float(attack_score),
                decay_score=float(decay_score),
                band_dominance_score=float(np.clip(dominance, 0.0, 1.0)),
                confidence=confidence,
                competing_class=comp_class,
                competing_class_score=float(np.clip(comp_score, 0.0, 1.0)),
                low_energy_ratio=float(np.clip(low_ratio, 0.0, 1.0)),
                mid_energy_ratio=float(np.clip(mid_ratio, 0.0, 1.0)),
                high_energy_ratio=float(np.clip(high_ratio, 0.0, 1.0)),
                spectral_centroid=float(centroid_proxy),
            )
        )
        next_id += 1

    return candidates, next_id


def _collect_multidetector_candidates(
    *,
    envelopes: dict[str, np.ndarray],
    sample_rate: int,
    hop_size: int,
    settings: dict[str, float | int],
) -> list[_HitCandidate]:
    thresholds = _default_detector_thresholds()
    detector_inputs: list[tuple[DetectorName, DrumClass, str]] = [
        ("kick", "kick", "kick"),
        ("snare", "snare_or_clap", "snare"),
        ("hat", "hat", "hat"),
        ("cymbal", "cymbal", "cymbal"),
        ("tom", "tom_or_perc", "tom"),
    ]

    candidates: list[_HitCandidate] = []
    next_id = 0
    for detector_name, class_name, envelope_key in detector_inputs:
        detector_candidates, next_id = _collect_detector_candidates(
            detector_name=detector_name,
            class_name=class_name,
            envelope=envelopes[envelope_key],
            all_env=envelopes,
            sample_rate=sample_rate,
            hop_size=hop_size,
            settings=settings,
            threshold=max(float(settings["onset_delta"]), thresholds[detector_name], 0.04),
            candidate_id_start=next_id,
        )
        candidates.extend(detector_candidates)

    return sorted(candidates, key=lambda item: (item.onset_sec, -item.confidence))


def _collect_global_candidates(
    *,
    envelopes: dict[str, np.ndarray],
    sample_rate: int,
    hop_size: int,
    settings: dict[str, float | int],
    min_onset_strength: float,
) -> list[_HitCandidate]:
    onset_times, onset_strengths = _detect_onsets(
        envelopes["full"],
        sample_rate,
        hop_size,
        min_onset_strength=min_onset_strength,
        onset_pre_max=int(settings["onset_pre_max"]),
        onset_post_max=int(settings["onset_post_max"]),
        onset_pre_avg=int(settings["onset_pre_avg"]),
        onset_post_avg=int(settings["onset_post_avg"]),
        onset_delta=float(settings["onset_delta"]),
        onset_wait=int(settings["onset_wait"]),
    )

    candidates: list[_HitCandidate] = []
    for idx, (onset_sec, onset_strength) in enumerate(zip(onset_times.tolist(), onset_strengths.tolist())):
        frame_idx = int(round((float(onset_sec) * sample_rate) / hop_size))
        frame_idx = max(0, min(frame_idx, len(envelopes["full"]) - 1))

        low = float(envelopes["low"][frame_idx])
        mid = float(envelopes["mid"][frame_idx])
        high = float(envelopes["high"][frame_idx])

        class_scores: list[tuple[DrumClass, float]] = [
            ("kick", 0.55 * low + 0.25 * float(onset_strength) + 0.20 * (low / (mid + 1e-9))),
            ("snare_or_clap", 0.45 * mid + 0.28 * float(onset_strength) + 0.27 * np.minimum(1.0, (mid + 0.6 * high))),
            ("hat", 0.46 * high + 0.24 * float(onset_strength) + 0.30 * np.minimum(1.0, high / (mid + 1e-9))),
            ("cymbal", 0.38 * high + 0.20 * float(onset_strength) + 0.42 * np.minimum(1.0, high + 0.35 * mid)),
            ("tom_or_perc", 0.42 * mid + 0.30 * low + 0.28 * float(onset_strength)),
        ]
        class_scores = [(name, float(np.clip(score, 0.0, 1.25))) for name, score in class_scores]
        class_scores.sort(key=lambda item: item[1], reverse=True)

        best_class, best_score = class_scores[0]
        competitor_class, competitor_score = class_scores[1]
        confidence = float(np.clip(best_score - 0.35 * competitor_score, 0.0, 0.98))

        band_total = low + mid + high + 1e-9
        attack_score, decay_score = _attack_decay_scores(envelopes["full"], frame_idx)

        candidates.append(
            _HitCandidate(
                candidate_id=idx,
                detector_name="global",
                onset_sec=float(onset_sec),
                onset_strength=float(onset_strength),
                class_name=best_class,
                low_peak_strength=low,
                mid_peak_strength=mid,
                high_peak_strength=high,
                attack_score=float(attack_score),
                decay_score=float(decay_score),
                band_dominance_score=float(max(low, mid, high) / band_total),
                confidence=confidence,
                competing_class=competitor_class,
                competing_class_score=float(np.clip(competitor_score, 0.0, 1.0)),
                low_energy_ratio=float(low / band_total),
                mid_energy_ratio=float(mid / band_total),
                high_energy_ratio=float(high / band_total),
                spectral_centroid=float((low / band_total) * 120 + (mid / band_total) * 1600 + (high / band_total) * 7000),
            )
        )

    return sorted(candidates, key=lambda item: item.onset_sec)


def _candidate_to_hit(
    *,
    candidate: _HitCandidate,
    hit_id: int,
    ticks_per_second: float,
    class_targets: dict[DrumClass, int],
    class_refractory_ms: dict[str, float],
    detection_mode: DetectionMode,
    unknown_target_note: int,
) -> _DetectedHit:
    class_name = _normalize_class_name(candidate.class_name)
    target = int(class_targets.get(class_name, unknown_target_note))
    if class_name == "unknown":
        target = int(unknown_target_note)

    canonical = _canonical_class(class_name)
    return _DetectedHit(
        hit_id=hit_id,
        onset_sec=float(candidate.onset_sec),
        accepted_onset_sec=float(candidate.onset_sec),
        tick=_hit_to_tick(float(candidate.onset_sec), ticks_per_second),
        onset_strength=float(candidate.onset_strength),
        low_band_onset_strength=float(candidate.low_peak_strength),
        mid_band_onset_strength=float(candidate.mid_peak_strength),
        high_band_onset_strength=float(candidate.high_peak_strength),
        class_name=class_name,
        low_energy_ratio=float(candidate.low_energy_ratio),
        mid_energy_ratio=float(candidate.mid_energy_ratio),
        high_energy_ratio=float(candidate.high_energy_ratio),
        spectral_centroid=float(candidate.spectral_centroid),
        confidence=float(candidate.confidence),
        target_note=target,
        velocity=1,
        suppressed=False,
        suppression_reason=None,
        grouped_transient_id=None,
        class_refractory_ms=float(class_refractory_ms.get(canonical, 0.0)),
        nearest_previous_same_class_ms=None,
        detection_mode=detection_mode,
        detector_name=candidate.detector_name,
        candidate_class=class_name,
        accepted_class=class_name,
        class_confidence=float(candidate.confidence),
        competing_class=candidate.competing_class,
        competing_class_score=float(candidate.competing_class_score),
        low_peak_strength=float(candidate.low_peak_strength),
        mid_peak_strength=float(candidate.mid_peak_strength),
        high_peak_strength=float(candidate.high_peak_strength),
        attack_score=float(candidate.attack_score),
        decay_score=float(candidate.decay_score),
        band_dominance_score=float(candidate.band_dominance_score),
        accepted=True,
        rejection_reason=None,
        merged_with_transient_id=None,
    )


def _increment_counter(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _suppress_hit(
    hit: _DetectedHit,
    reason: str,
    *,
    duplicate_suppressed_by_class: dict[str, int],
    rejected_by_reason: dict[str, int],
) -> None:
    if hit.suppressed:
        return
    hit.suppressed = True
    hit.accepted_onset_sec = None
    hit.suppression_reason = reason
    hit.rejection_reason = reason
    hit.accepted = False
    hit.accepted_class = None
    _increment_counter(rejected_by_reason, reason)
    if reason in {"same_transient_group", "global_min_spacing", "class_refractory", "merge_conflict"}:
        canonical = _canonical_class(hit.class_name)
        duplicate_suppressed_by_class[canonical] = duplicate_suppressed_by_class.get(canonical, 0) + 1


def _allow_layering(class_a: str, class_b: str) -> bool:
    pair = {class_a, class_b}
    return pair in (
        {"kick", "snare_or_clap"},
        {"kick", "hat"},
        {"kick", "cymbal"},
        {"snare_or_clap", "hat"},
    )


def _merge_same_transient_candidates(
    hits: list[_DetectedHit],
    *,
    window_sec: float,
    duplicate_suppressed_by_class: dict[str, int],
    rejected_by_reason: dict[str, int],
) -> tuple[list[_DetectedHit], int]:
    if not hits:
        return [], 0

    sorted_hits = sorted(hits, key=lambda item: (item.onset_sec, -item.score()))
    groups: list[list[_DetectedHit]] = []
    current: list[_DetectedHit] = [sorted_hits[0]]
    for hit in sorted_hits[1:]:
        if (hit.onset_sec - current[-1].onset_sec) <= window_sec:
            current.append(hit)
        else:
            groups.append(current)
            current = [hit]
    groups.append(current)

    conflicts = 0
    accepted_all: list[_DetectedHit] = []
    for group_id, group in enumerate(groups):
        for hit in group:
            hit.grouped_transient_id = group_id
            hit.merged_with_transient_id = group_id

        chosen: list[_DetectedHit] = []
        for candidate in sorted(group, key=lambda item: item.score(), reverse=True):
            if candidate.suppressed:
                continue
            candidate_class = _canonical_class(candidate.class_name)

            duplicate_same = next(
                (item for item in chosen if _canonical_class(item.class_name) == candidate_class),
                None,
            )
            if duplicate_same is not None:
                conflicts += 1
                _suppress_hit(
                    candidate,
                    "same_transient_group",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                    rejected_by_reason=rejected_by_reason,
                )
                continue

            blocked = False
            for existing in chosen:
                existing_class = _canonical_class(existing.class_name)
                if candidate_class == existing_class:
                    continue
                if _allow_layering(candidate_class, existing_class):
                    continue
                if {candidate_class, existing_class} == {"hat", "cymbal"}:
                    # Allow one high-class event only unless cymbal is clearly stronger.
                    if candidate_class == "cymbal" and candidate.score() > existing.score() * 1.05:
                        _suppress_hit(
                            existing,
                            "merge_conflict",
                            duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                            rejected_by_reason=rejected_by_reason,
                        )
                        chosen.remove(existing)
                        break
                    blocked = True
                    break
                blocked = True
                break

            if blocked:
                conflicts += 1
                _suppress_hit(
                    candidate,
                    "merge_conflict",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                    rejected_by_reason=rejected_by_reason,
                )
                continue

            chosen.append(candidate)

        accepted_all.extend([item for item in chosen if not item.suppressed])

    return sorted(accepted_all, key=lambda item: item.onset_sec), conflicts


def _apply_global_min_spacing(
    hits: list[_DetectedHit],
    *,
    min_spacing_sec: float,
    duplicate_suppressed_by_class: dict[str, int],
    rejected_by_reason: dict[str, int],
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
            prev_class = _canonical_class(previous.class_name)
            curr_class = _canonical_class(hit.class_name)
            if (
                previous.grouped_transient_id is not None
                and hit.grouped_transient_id is not None
                and previous.grouped_transient_id == hit.grouped_transient_id
                and prev_class != curr_class
                and _allow_layering(prev_class, curr_class)
            ):
                accepted.append(hit)
                continue

            if hit.score() > previous.score():
                _suppress_hit(
                    previous,
                    "global_min_spacing",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                    rejected_by_reason=rejected_by_reason,
                )
                accepted[-1] = hit
            else:
                _suppress_hit(
                    hit,
                    "global_min_spacing",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                    rejected_by_reason=rejected_by_reason,
                )
            continue

        accepted.append(hit)

    return sorted([item for item in accepted if not item.suppressed], key=lambda item: item.onset_sec)


def _apply_class_refractory(
    hits: list[_DetectedHit],
    *,
    class_refractory_sec: dict[str, float],
    duplicate_suppressed_by_class: dict[str, int],
    rejected_by_reason: dict[str, int],
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
                    rejected_by_reason=rejected_by_reason,
                )
                accepted[last_index] = hit
            else:
                _suppress_hit(
                    hit,
                    "class_refractory",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                    rejected_by_reason=rejected_by_reason,
                )
            continue

        accepted.append(hit)
        last_by_class_index[canonical] = len(accepted) - 1

    return sorted([item for item in accepted if not item.suppressed], key=lambda item: item.onset_sec)


def _assign_nearest_same_class_ms(hits: list[_DetectedHit]) -> None:
    last_seen: dict[str, float] = {}
    for hit in sorted(hits, key=lambda item: item.onset_sec):
        canonical = _canonical_class(hit.class_name)
        previous = last_seen.get(canonical)
        hit.nearest_previous_same_class_ms = (
            None if previous is None else (hit.onset_sec - previous) * 1000.0
        )
        if not hit.suppressed:
            last_seen[canonical] = hit.onset_sec


def _assign_output_velocities(hits: list[_DetectedHit]) -> None:
    if not hits:
        return

    strengths = np.asarray(
        [
            hit.onset_strength
            * (0.70 + 0.20 * hit.class_confidence + 0.10 * hit.attack_score)
            for hit in hits
        ],
        dtype=np.float64,
    )
    low = float(np.percentile(strengths, 10))
    high = float(np.percentile(strengths, 90))
    span = max(high - low, 1e-9)

    for hit, value in zip(hits, strengths.tolist()):
        normalized = np.clip((value - low) / span, 0.0, 1.0)
        confidence_gain = np.clip(hit.class_confidence, 0.0, 1.0)
        velocity = int(round(20.0 + (88.0 * normalized) + (12.0 * confidence_gain)))
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
    return {name: float(count) / safe_duration for name, count in class_counts.items()}


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
        warnings.append("Strong warning: kick count exceeds 700; output is likely too dense.")

    if class_counts.get("cymbal", 0) > 400:
        too_dense = True
        warnings.append("Strong warning: cymbal count exceeds 400; output is likely too dense.")

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
        warnings.append("Warning: overall notes_per_second is unusually high for drum extraction.")

    return too_dense, warnings


def _to_per_hit_summary(hits: list[_DetectedHit]) -> list[PerHitSummary]:
    return [
        PerHitSummary(
            onset_sec=float(hit.onset_sec),
            raw_onset_sec=float(hit.onset_sec),
            accepted_onset_sec=(None if hit.accepted_onset_sec is None else float(hit.accepted_onset_sec)),
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
                None if hit.nearest_previous_same_class_ms is None else float(hit.nearest_previous_same_class_ms)
            ),
            detection_mode=hit.detection_mode,
            detector_name=hit.detector_name,
            candidate_class=hit.candidate_class,
            accepted_class=hit.accepted_class,
            class_confidence=float(hit.class_confidence),
            competing_class=hit.competing_class,
            competing_class_score=float(hit.competing_class_score),
            low_peak_strength=float(hit.low_peak_strength),
            mid_peak_strength=float(hit.mid_peak_strength),
            high_peak_strength=float(hit.high_peak_strength),
            attack_score=float(hit.attack_score),
            decay_score=float(hit.decay_score),
            band_dominance_score=float(hit.band_dominance_score),
            accepted=bool(hit.accepted),
            rejection_reason=hit.rejection_reason,
            merged_with_transient_id=hit.merged_with_transient_id,
        )
        for hit in hits
    ]


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
                "detection_mode",
                "detector_name",
                "candidate_class",
                "accepted_class",
                "class_confidence",
                "competing_class",
                "competing_class_score",
                "accepted",
                "rejection_reason",
                "merged_with_transient_id",
                "low_peak_strength",
                "mid_peak_strength",
                "high_peak_strength",
                "attack_score",
                "decay_score",
                "band_dominance_score",
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
                    "detection_mode": hit.detection_mode,
                    "detector_name": hit.detector_name,
                    "candidate_class": hit.candidate_class,
                    "accepted_class": hit.accepted_class,
                    "class_confidence": hit.class_confidence,
                    "competing_class": hit.competing_class,
                    "competing_class_score": hit.competing_class_score,
                    "accepted": str(hit.accepted).lower(),
                    "rejection_reason": hit.rejection_reason,
                    "merged_with_transient_id": hit.merged_with_transient_id,
                    "low_peak_strength": hit.low_peak_strength,
                    "mid_peak_strength": hit.mid_peak_strength,
                    "high_peak_strength": hit.high_peak_strength,
                    "attack_score": hit.attack_score,
                    "decay_score": hit.decay_score,
                    "band_dominance_score": hit.band_dominance_score,
                }
            )


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

    envelopes, _frame_size, hop_size = _onset_strength_envelopes(mono, sample_rate)

    raw_onset_times, _raw_onset_strengths = _detect_onsets(
        envelopes["full"],
        sample_rate,
        hop_size,
        min_onset_strength=params.min_onset_strength,
        onset_pre_max=int(settings["onset_pre_max"]),
        onset_post_max=int(settings["onset_post_max"]),
        onset_pre_avg=int(settings["onset_pre_avg"]),
        onset_post_avg=int(settings["onset_post_avg"]),
        onset_delta=float(settings["onset_delta"]),
        onset_wait=int(settings["onset_wait"]),
    )

    if params.detection_mode == "multi-detector":
        candidates = _collect_multidetector_candidates(
            envelopes=envelopes,
            sample_rate=sample_rate,
            hop_size=hop_size,
            settings=settings,
        )
    else:
        candidates = _collect_global_candidates(
            envelopes=envelopes,
            sample_rate=sample_rate,
            hop_size=hop_size,
            settings=settings,
            min_onset_strength=params.min_onset_strength,
        )

    candidate_times = np.asarray([item.onset_sec for item in candidates], dtype=np.float64)
    detected_bpm = _estimate_bpm(candidate_times if len(candidate_times) > 1 else raw_onset_times)
    bpm_used = float(params.bpm) if params.bpm is not None else float(detected_bpm or DEFAULT_BPM)
    bpm_source: Literal["detected", "forced"] = "forced" if params.bpm is not None else "detected"
    ticks_per_second = (params.ticks_per_beat * bpm_used) / 60.0

    warnings: list[str] = []
    if detected_bpm is None:
        warnings.append("Could not confidently estimate BPM from onsets; defaulted to stable fallback.")
    if len(candidates) == 0:
        warnings.append("No detector candidates found above threshold.")

    class_refractory_ms = _resolve_refractory_ms(settings)
    class_refractory_sec = {
        class_name: value / 1000.0 for class_name, value in class_refractory_ms.items()
    }
    min_hit_spacing_sec = float(settings["min_hit_spacing_ms"]) / 1000.0
    same_transient_window_sec = float(settings["same_transient_window_ms"]) / 1000.0
    min_class_confidence = float(settings["min_class_confidence"])

    detector_candidate_counts: dict[str, int] = {
        "global": 0,
        "kick": 0,
        "snare": 0,
        "hat": 0,
        "cymbal": 0,
        "tom": 0,
    }
    detector_rejected_counts: dict[str, int] = {
        "global": 0,
        "kick": 0,
        "snare": 0,
        "hat": 0,
        "cymbal": 0,
        "tom": 0,
    }
    detector_accepted_counts: dict[str, int] = {
        "global": 0,
        "kick": 0,
        "snare": 0,
        "hat": 0,
        "cymbal": 0,
        "tom": 0,
    }

    duplicate_suppressed_by_class: dict[str, int] = {
        "kick": 0,
        "snare_or_clap": 0,
        "hat": 0,
        "cymbal": 0,
        "tom_or_perc": 0,
        "unknown": 0,
    }
    rejected_by_reason: dict[str, int] = {}

    unknown_target = (
        int(params.unknown_target_note)
        if params.unknown_target_note is not None
        else int(class_targets.get("unknown", 39))
    )

    raw_hits: list[_DetectedHit] = []
    low_confidence_rejected_count = 0

    for hit_id, candidate in enumerate(candidates):
        _increment_counter(detector_candidate_counts, candidate.detector_name)
        hit = _candidate_to_hit(
            candidate=candidate,
            hit_id=hit_id,
            ticks_per_second=ticks_per_second,
            class_targets=class_targets,
            class_refractory_ms=class_refractory_ms,
            detection_mode=params.detection_mode,
            unknown_target_note=unknown_target,
        )

        if hit.class_confidence < min_class_confidence:
            low_confidence_rejected_count += 1
            _increment_counter(detector_rejected_counts, candidate.detector_name)
            _suppress_hit(
                hit,
                "low_confidence",
                duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                rejected_by_reason=rejected_by_reason,
            )
            raw_hits.append(hit)
            continue

        if _canonical_class(hit.class_name) == "tom_or_perc" and (
            hit.competing_class in {"kick", "snare_or_clap"}
            and hit.competing_class_score >= hit.class_confidence * 0.92
        ):
            _increment_counter(detector_rejected_counts, candidate.detector_name)
            _suppress_hit(
                hit,
                "tom_not_clear",
                duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                rejected_by_reason=rejected_by_reason,
            )
            raw_hits.append(hit)
            continue

        if _canonical_class(hit.class_name) == "unknown" and not params.emit_unknown:
            _increment_counter(detector_rejected_counts, candidate.detector_name)
            _suppress_hit(
                hit,
                "unknown_skipped",
                duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                rejected_by_reason=rejected_by_reason,
            )
            raw_hits.append(hit)
            continue

        raw_hits.append(hit)

    premerge_hits = [item for item in raw_hits if not item.suppressed]
    merged_hits, merge_conflicts = _merge_same_transient_candidates(
        premerge_hits,
        window_sec=same_transient_window_sec,
        duplicate_suppressed_by_class=duplicate_suppressed_by_class,
        rejected_by_reason=rejected_by_reason,
    )
    spaced_hits = _apply_global_min_spacing(
        merged_hits,
        min_spacing_sec=min_hit_spacing_sec,
        duplicate_suppressed_by_class=duplicate_suppressed_by_class,
        rejected_by_reason=rejected_by_reason,
    )
    accepted_hits = _apply_class_refractory(
        spaced_hits,
        class_refractory_sec=class_refractory_sec,
        duplicate_suppressed_by_class=duplicate_suppressed_by_class,
        rejected_by_reason=rejected_by_reason,
    )

    _assign_nearest_same_class_ms(sorted(raw_hits, key=lambda item: item.onset_sec))

    emitted_hits: list[_DetectedHit] = []
    for hit in accepted_hits:
        if _canonical_class(hit.class_name) == "unknown" and not params.emit_unknown:
            _suppress_hit(
                hit,
                "unknown_skipped",
                duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                rejected_by_reason=rejected_by_reason,
            )
            continue
        emitted_hits.append(hit)

    for hit in raw_hits:
        if not hit.suppressed:
            _increment_counter(detector_accepted_counts, hit.detector_name)
        else:
            _increment_counter(detector_rejected_counts, hit.detector_name)

    _assign_output_velocities(emitted_hits)

    suppressed_duplicate_count = int(sum(duplicate_suppressed_by_class.values()))

    class_counts = _count_classes(emitted_hits)
    duplicate_interval_summary = _duplicate_interval_summary(emitted_hits, class_refractory_ms)
    too_dense_warning, density_warnings = _evaluate_density_warnings(
        duration_sec=duration_sec,
        class_counts=class_counts,
        duplicate_interval_summary=duplicate_interval_summary,
    )
    warnings.extend(density_warnings)

    if not params.emit_unknown:
        warnings.append("Unknown hits are skipped by default; use --emit-unknown to include them.")

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
                class_track_name = f"drums_from_audio_{params.target_map.replace('-', '_')}_{class_name}"
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
        detection_mode=params.detection_mode,
        onset_count=len(raw_onset_times),
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
        detector_candidate_counts=detector_candidate_counts,
        detector_accepted_counts=detector_accepted_counts,
        detector_rejected_counts=detector_rejected_counts,
        low_confidence_rejected_count=low_confidence_rejected_count,
        rejected_by_reason=rejected_by_reason,
        multi_detector_merge_conflicts=merge_conflicts,
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
