from __future__ import annotations

import csv
import json
import re
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
from midi_cleaner.drums.layer_mapping import (
    DrumLayerMapping,
    DrumLayerMappingError,
    build_default_layer_mapping,
    duplicate_target_notes,
    load_layer_mapping,
    ordered_layers,
    save_layer_mapping,
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
OutputLayout = Literal["separate-files", "multitrack", "single-track"]
DetectorName = Literal["global", "kick", "snare", "hat", "cymbal", "tom"]
LayerName = Literal["kick", "snare_clap", "hat", "tom_perc", "cymbal", "unknown"]

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

_LAYER_ORDER: tuple[str, ...] = (
    "kick",
    "snare_clap",
    "hat",
    "tom_perc",
    "cymbal",
)

_LAYER_TRACK_NAMES: dict[str, str] = {
    "kick": "Kick",
    "snare_clap": "SnareClap",
    "hat": "Hat",
    "tom_perc": "TomPerc",
    "cymbal": "Cymbal",
    "unknown": "TomPerc",
}

_LAYER_TO_CLASS: dict[str, DrumClass] = {
    "kick": "kick",
    "snare_clap": "snare_or_clap",
    "hat": "hat",
    "tom_perc": "tom_or_perc",
    "cymbal": "cymbal",
    "unknown": "unknown",
}

_CLASS_TO_LAYER: dict[str, str] = {
    "kick": "kick",
    "snare_or_clap": "snare_clap",
    "hat": "hat",
    "cymbal": "cymbal",
    "tom_or_perc": "tom_perc",
    "unknown": "unknown",
}

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
    output_file: Path | None
    target_map: str
    map_file: Path | None = None
    output_dir: Path | None = None
    mapping_file: Path | None = None
    save_mapping_file: Path | None = None
    write_empty_layers: bool = False
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
    output_layout: OutputLayout = "separate-files"
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
    layer_name: str
    semantic_layer: str
    target_track_name: str
    target_note_name: str
    output_file: str | None
    mapping_name: str | None
    primary_slot_assignment: bool
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
    same_layer_duplicate: bool
    cross_layer_allowed: bool
    nearest_previous_same_layer_ms: float | None
    simultaneous_layers_at_time: str


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
    output_layout: OutputLayout
    output_dir: str | None
    mapping_file: str | None
    mapping_name: str
    track_order: list[str]
    created_files: list[str]
    skipped_layers: dict[str, str]
    disabled_layers: list[str]
    write_empty_layers: bool
    duplicate_target_notes: dict[str, list[str]]
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
    layer_counts: dict[str, int]
    layer_target_notes: dict[str, int]
    layer_target_note_names: dict[str, str]
    layer_track_names: dict[str, str]
    layer_output_pitch_counts: dict[str, dict[str, int]]
    populated_semantic_layers: list[str]
    unpopulated_enabled_layers: list[str]
    unpopulated_disabled_layers: list[str]
    primary_slot_assignment_used: bool
    cross_layer_simultaneous_hit_count: int
    same_layer_suppressed_count: int
    cross_layer_suppressed_count: int
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
                "layer_name": item.layer_name,
                "semantic_layer": item.semantic_layer,
                "target_track_name": item.target_track_name,
                "target_note_name": item.target_note_name,
                "output_file": item.output_file,
                "mapping_name": item.mapping_name,
                "primary_slot_assignment": item.primary_slot_assignment,
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
                "same_layer_duplicate": item.same_layer_duplicate,
                "cross_layer_allowed": item.cross_layer_allowed,
                "nearest_previous_same_layer_ms": item.nearest_previous_same_layer_ms,
                "simultaneous_layers_at_time": item.simultaneous_layers_at_time,
            }
            for item in self.per_hit_summary
        ]
        return payload


@dataclass
class DrumLayerHit:
    onset_sec: float
    tick: int
    instrument: str
    target_note: int
    velocity: int
    confidence: float
    detector_name: str
    evidence: dict[str, object]


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
    layer_name: LayerName
    semantic_layer: str
    target_track_name: str
    target_note_name: str
    output_file: str | None
    mapping_name: str | None
    primary_slot_assignment: bool
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
    same_layer_duplicate: bool
    cross_layer_allowed: bool
    nearest_previous_same_layer_ms: float | None
    simultaneous_layers_at_time: str

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


def _layer_from_class(class_name: DrumClass) -> LayerName:
    canonical = _canonical_class(class_name)
    return _CLASS_TO_LAYER.get(canonical, "unknown")  # type: ignore[return-value]


def _track_name_from_layer(layer_name: LayerName) -> str:
    return _LAYER_TRACK_NAMES.get(layer_name, "TomPerc")


def _layer_sort_key(layer_name: str) -> int:
    if layer_name in _LAYER_ORDER:
        return _LAYER_ORDER.index(layer_name)
    return len(_LAYER_ORDER)


def _semantic_slot_for_hit(hit: _DetectedHit, *, snare_target: SnareTarget) -> str | None:
    canonical = _canonical_class(hit.class_name)
    if canonical == "kick":
        return "kick_1"

    if canonical == "snare_or_clap":
        if snare_target == "clap":
            return "clap_1"
        if hit.high_peak_strength >= 0.65 and hit.attack_score >= 0.58 and hit.decay_score <= 0.60:
            return "clap_1"
        return "snare_1"

    if canonical == "hat":
        if hit.decay_score >= 0.62 and hit.high_peak_strength >= 0.30:
            return "hh_open_1"
        return "hh_1"

    if canonical == "cymbal":
        return "cym_1"

    if canonical == "tom_or_perc":
        if hit.high_peak_strength >= 0.82:
            return "perc_1"
        if hit.spectral_centroid < 600.0 or hit.low_peak_strength > (hit.mid_peak_strength * 1.35):
            return "tom_l_1"
        if hit.spectral_centroid < 1400.0 and hit.mid_peak_strength > 0.90:
            return "tom_m_1"
        if hit.spectral_centroid < 3000.0 and hit.mid_peak_strength > 1.10:
            return "tom_h_1"
        return "tom_l_1"

    return None


def _resolve_layer_mapping(
    params: AudioDrumExtractionParameters,
) -> tuple[DrumLayerMapping, dict[str, list[str]], list[str]]:
    mapping_warnings: list[str] = []
    if params.mapping_file is not None:
        try:
            mapping = load_layer_mapping(
                params.mapping_file,
                fallback_c1_midi_note=params.c1_midi_note,
            )
        except DrumLayerMappingError as exc:
            raise AudioDrumExtractionError(str(exc)) from exc
    else:
        mapping = build_default_layer_mapping(
            target_map=params.target_map,
            c1_midi_note=params.c1_midi_note,
            name=f"{params.target_map}_expanded_default_mapping",
        )
        mapping_warnings.append(
            "No mapping file was supplied; using expanded default semantic layer mapping."
        )

    if params.save_mapping_file is not None:
        save_layer_mapping(mapping, params.save_mapping_file)

    duplicate_notes = duplicate_target_notes(mapping)
    if duplicate_notes:
        mapping_warnings.append(
            "Enabled semantic layers map to duplicate target notes; review duplicate_target_notes in report."
        )

    return mapping, duplicate_notes, mapping_warnings


def _resolve_output_target(
    params: AudioDrumExtractionParameters,
    wav_file: Path,
) -> tuple[Path, Path]:
    default_output = wav_file.with_suffix(".mid")
    output_file = params.output_file if params.output_file is not None else default_output

    if params.output_dir is not None:
        output_dir = params.output_dir
    elif params.output_layout == "separate-files":
        output_dir = output_file.parent
    else:
        output_dir = output_file.parent

    return output_file, output_dir


def _sanitize_file_token(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]+", "", value.strip())
    return compact or "Layer"


def _slot_to_class_for_duration(slot_name: str) -> DrumClass:
    family = slot_name.rsplit("_", 1)[0]
    if family == "kick":
        return "kick"
    if family in {"snare", "clap"}:
        return "snare_or_clap"
    if family in {"hh", "hh_open"}:
        return "hat"
    if family in {"tom_l", "tom_m", "tom_h", "perc"}:
        return "tom_or_perc"
    if family == "cym":
        return "cymbal"
    return "unknown"


def _slot_export_file_name(index: int, track_name: str, note_name: str) -> str:
    return f"{index:02d}_{_sanitize_file_token(track_name)}_{_sanitize_file_token(note_name)}.mid"


def _layer_statistics(
    mapping: DrumLayerMapping,
    layer_hits: dict[str, list[DrumLayerHit]],
) -> tuple[list[str], list[str], list[str]]:
    populated = sorted([layer for layer, hits in layer_hits.items() if len(hits) > 0])

    unpopulated_enabled: list[str] = []
    unpopulated_disabled: list[str] = []
    for layer_name, slot in mapping.layers.items():
        if layer_hits.get(layer_name):
            continue
        if slot.enabled:
            unpopulated_enabled.append(layer_name)
        else:
            unpopulated_disabled.append(layer_name)

    return populated, sorted(unpopulated_enabled), sorted(unpopulated_disabled)


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

    if params.output_layout not in {"separate-files", "multitrack", "single-track"}:
        raise AudioDrumExtractionError(
            "--output-layout must be one of: separate-files, multitrack, single-track."
        )

    if params.mapping_file is not None and params.mapping_file.suffix.lower() != ".json":
        raise AudioDrumExtractionError("--mapping-file must point to a .json file.")

    if params.save_mapping_file is not None and params.save_mapping_file.suffix.lower() != ".json":
        raise AudioDrumExtractionError("--save-mapping-file must point to a .json file.")

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
    layer_name = _layer_from_class(class_name)
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
        layer_name=layer_name,
        semantic_layer="",
        target_track_name=_track_name_from_layer(layer_name),
        target_note_name="",
        output_file=None,
        mapping_name=None,
        primary_slot_assignment=True,
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
        same_layer_duplicate=False,
        cross_layer_allowed=False,
        nearest_previous_same_layer_ms=None,
        simultaneous_layers_at_time=layer_name,
    )


def _increment_counter(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _suppress_hit(
    hit: _DetectedHit,
    reason: str,
    *,
    duplicate_suppressed_by_class: dict[str, int],
    rejected_by_reason: dict[str, int],
    same_layer_duplicate: bool = False,
) -> None:
    if hit.suppressed:
        return
    hit.suppressed = True
    hit.accepted_onset_sec = None
    hit.suppression_reason = reason
    hit.rejection_reason = reason
    hit.accepted = False
    hit.accepted_class = None
    if same_layer_duplicate:
        hit.same_layer_duplicate = True
    _increment_counter(rejected_by_reason, reason)
    if reason in {"same_transient_group", "global_min_spacing", "class_refractory", "merge_conflict"}:
        canonical = _canonical_class(hit.class_name)
        duplicate_suppressed_by_class[canonical] = duplicate_suppressed_by_class.get(canonical, 0) + 1


def _group_by_transient_window(hits: list[_DetectedHit], window_sec: float) -> list[list[_DetectedHit]]:
    if not hits:
        return []

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
    return groups


def _merge_same_transient_candidates(
    hits: list[_DetectedHit],
    *,
    window_sec: float,
    duplicate_suppressed_by_class: dict[str, int],
    rejected_by_reason: dict[str, int],
) -> tuple[list[_DetectedHit], int]:
    if not hits:
        return [], 0

    groups = _group_by_transient_window(hits, window_sec)
    suppressed_count = 0
    accepted_all: list[_DetectedHit] = []
    for group_id, group in enumerate(groups):
        for hit in group:
            hit.grouped_transient_id = group_id
            hit.merged_with_transient_id = group_id

        ranked = sorted(group, key=lambda item: item.score(), reverse=True)
        chosen = ranked[0]
        accepted_all.append(chosen)
        for candidate in ranked[1:]:
            suppressed_count += 1
            _suppress_hit(
                candidate,
                "same_transient_group",
                duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                rejected_by_reason=rejected_by_reason,
                same_layer_duplicate=True,
            )

    return sorted([item for item in accepted_all if not item.suppressed], key=lambda item: item.onset_sec), suppressed_count


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
            if hit.score() > previous.score():
                _suppress_hit(
                    previous,
                    "global_min_spacing",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                    rejected_by_reason=rejected_by_reason,
                    same_layer_duplicate=True,
                )
                accepted[-1] = hit
            else:
                _suppress_hit(
                    hit,
                    "global_min_spacing",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                    rejected_by_reason=rejected_by_reason,
                    same_layer_duplicate=True,
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
                    same_layer_duplicate=True,
                )
                accepted[last_index] = hit
            else:
                _suppress_hit(
                    hit,
                    "class_refractory",
                    duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                    rejected_by_reason=rejected_by_reason,
                    same_layer_duplicate=True,
                )
            continue

        accepted.append(hit)
        last_by_class_index[canonical] = len(accepted) - 1

    return sorted([item for item in accepted if not item.suppressed], key=lambda item: item.onset_sec)


def _assign_nearest_same_class_ms(hits: list[_DetectedHit]) -> None:
    last_seen: dict[str, float] = {}
    for hit in sorted(hits, key=lambda item: item.onset_sec):
        key = str(hit.layer_name)
        previous = last_seen.get(key)
        interval_ms = (
            None if previous is None else (hit.onset_sec - previous) * 1000.0
        )
        hit.nearest_previous_same_class_ms = interval_ms
        hit.nearest_previous_same_layer_ms = interval_ms
        if not hit.suppressed:
            last_seen[key] = hit.onset_sec


def _split_hits_by_layer(hits: list[_DetectedHit]) -> dict[str, list[_DetectedHit]]:
    by_layer: dict[str, list[_DetectedHit]] = {layer: [] for layer in (*_LAYER_ORDER, "unknown")}
    for hit in hits:
        by_layer.setdefault(str(hit.layer_name), []).append(hit)
    return by_layer


def _annotate_simultaneous_layers(hits: list[_DetectedHit], *, window_sec: float) -> int:
    if not hits:
        return 0

    cross_layer_group_count = 0
    for group in _group_by_transient_window(hits, window_sec):
        layers = sorted({str(hit.layer_name) for hit in group}, key=_layer_sort_key)
        layer_token = "|".join(layers) if layers else "unknown"
        is_cross = len(layers) > 1
        if is_cross:
            cross_layer_group_count += 1

        for hit in group:
            hit.cross_layer_allowed = bool(is_cross and hit.detection_mode == "multi-detector")
            hit.simultaneous_layers_at_time = layer_token

    return cross_layer_group_count


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


def _to_layered_hits(
    hits: list[_DetectedHit],
    *,
    mapping: DrumLayerMapping,
    snare_target: SnareTarget,
    warnings: list[str],
) -> dict[str, list[DrumLayerHit]]:
    layered_hits: dict[str, list[DrumLayerHit]] = {
        layer_name: [] for layer_name in ordered_layers(mapping)
    }
    warned_missing_layers: set[str] = set()

    for hit in sorted(hits, key=lambda item: item.onset_sec):
        semantic_layer = _semantic_slot_for_hit(hit, snare_target=snare_target)
        if semantic_layer is None:
            continue

        slot = mapping.layers.get(semantic_layer)
        if slot is None:
            if semantic_layer not in warned_missing_layers:
                warnings.append(
                    f"Detected semantic layer '{semantic_layer}' is not present in mapping and was skipped."
                )
                warned_missing_layers.add(semantic_layer)
            continue

        hit.semantic_layer = semantic_layer
        hit.primary_slot_assignment = True
        hit.target_note = int(slot.note)
        hit.target_note_name = str(slot.note_name)
        hit.target_track_name = str(slot.track_name)
        hit.mapping_name = mapping.name

        layered_hits.setdefault(semantic_layer, []).append(
            DrumLayerHit(
                onset_sec=float(hit.onset_sec),
                tick=int(hit.tick),
                instrument=semantic_layer,
                target_note=int(slot.note),
                velocity=int(hit.velocity),
                confidence=float(hit.class_confidence),
                detector_name=str(hit.detector_name),
                evidence={
                    "class_name": str(hit.class_name),
                    "onset_strength": float(hit.onset_strength),
                    "attack_score": float(hit.attack_score),
                    "decay_score": float(hit.decay_score),
                    "band_dominance_score": float(hit.band_dominance_score),
                },
            )
        )

    return layered_hits


def _layer_duration_sec(layer_name: str) -> float:
    mapped_class = _slot_to_class_for_duration(layer_name)
    return float(_CLASS_NOTE_DURATIONS_SEC.get(mapped_class, 0.10))


def _layer_absolute_events(
    *,
    layer_hits: list[DrumLayerHit],
    ticks_per_second: float,
    source_length_ticks: int,
    channel: int,
) -> list[tuple[int, int, mido.Message]]:
    absolute_events: list[tuple[int, int, mido.Message]] = []
    for hit in layer_hits:
        start_tick = _hit_to_tick(hit.onset_sec, ticks_per_second)
        duration_ticks = max(1, int(round(_layer_duration_sec(hit.instrument) * ticks_per_second)))
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
    return absolute_events


def _flatten_layer_hits(
    layered_hits: dict[str, list[DrumLayerHit]],
    layer_order: list[str],
) -> list[DrumLayerHit]:
    flattened: list[DrumLayerHit] = []
    for layer_name in layer_order:
        flattened.extend(layered_hits.get(layer_name, []))
    return sorted(flattened, key=lambda item: (item.onset_sec, item.target_note))


def _build_single_track_midi(
    *,
    layered_hits: dict[str, list[DrumLayerHit]],
    layer_order: list[str],
    output_path: Path,
    ticks_per_beat: int,
    bpm_used: float,
    duration_sec: float,
    channel: int,
    track_name: str,
) -> tuple[int, int, list[str]]:
    tempo_us_per_beat = int(round(60_000_000.0 / max(bpm_used, 1e-9)))
    ticks_per_second = (ticks_per_beat * bpm_used) / 60.0
    source_length_ticks = max(0, int(round(duration_sec * ticks_per_second)))

    midi_file = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us_per_beat, time=0))

    layer_hits = _flatten_layer_hits(layered_hits, layer_order)
    absolute_events = _layer_absolute_events(
        layer_hits=layer_hits,
        ticks_per_second=ticks_per_second,
        source_length_ticks=source_length_ticks,
        channel=channel,
    )

    previous_tick = 0
    for tick, _priority, message in absolute_events:
        track.append(message.copy(time=tick - previous_tick))
        previous_tick = tick

    final_tick = max(source_length_ticks, previous_tick)
    track.append(mido.MetaMessage("end_of_track", time=final_tick - previous_tick))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_file.save(str(output_path))

    return source_length_ticks, final_tick, [track_name]


def _build_multitrack_midi(
    *,
    layered_hits: dict[str, list[DrumLayerHit]],
    layer_order: list[str],
    mapping: DrumLayerMapping,
    output_path: Path,
    ticks_per_beat: int,
    bpm_used: float,
    duration_sec: float,
    channel: int,
) -> tuple[int, int, list[str]]:
    tempo_us_per_beat = int(round(60_000_000.0 / max(bpm_used, 1e-9)))
    ticks_per_second = (ticks_per_beat * bpm_used) / 60.0
    source_length_ticks = max(0, int(round(duration_sec * ticks_per_second)))

    midi_file = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    output_tracks: list[str] = []
    final_tick = source_length_ticks

    for layer_name in layer_order:
        slot = mapping.layers[layer_name]
        track_name = slot.track_name
        output_tracks.append(track_name)

        track = mido.MidiTrack()
        midi_file.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=track_name, time=0))
        track.append(mido.MetaMessage("set_tempo", tempo=tempo_us_per_beat, time=0))

        absolute_events = _layer_absolute_events(
            layer_hits=layered_hits.get(layer_name, []),
            ticks_per_second=ticks_per_second,
            source_length_ticks=source_length_ticks,
            channel=channel,
        )

        previous_tick = 0
        for tick, _priority, message in absolute_events:
            track.append(message.copy(time=tick - previous_tick))
            previous_tick = tick

        track.append(mido.MetaMessage("end_of_track", time=max(0, source_length_ticks - previous_tick)))
        final_tick = max(final_tick, previous_tick)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_file.save(str(output_path))
    return source_length_ticks, max(final_tick, source_length_ticks), output_tracks


def _select_export_layers(
    *,
    mapping: DrumLayerMapping,
    layered_hits: dict[str, list[DrumLayerHit]],
    write_empty_layers: bool,
) -> tuple[list[str], list[str], dict[str, str]]:
    selected_layers: list[str] = []
    disabled_layers: list[str] = []
    skipped_layers: dict[str, str] = {}

    for layer_name in ordered_layers(mapping):
        slot = mapping.layers[layer_name]
        hit_count = len(layered_hits.get(layer_name, []))

        if not slot.enabled:
            disabled_layers.append(layer_name)
            if write_empty_layers:
                selected_layers.append(layer_name)
            else:
                skipped_layers[layer_name] = "disabled"
            continue

        if hit_count == 0 and not write_empty_layers:
            skipped_layers[layer_name] = "empty"
            continue

        selected_layers.append(layer_name)

    return selected_layers, sorted(disabled_layers), skipped_layers


def _build_separate_files_midi(
    *,
    layered_hits: dict[str, list[DrumLayerHit]],
    layer_order: list[str],
    mapping: DrumLayerMapping,
    output_dir: Path,
    ticks_per_beat: int,
    bpm_used: float,
    duration_sec: float,
    channel: int,
) -> tuple[int, int, list[str], list[Path], dict[str, Path]]:
    created_files: list[Path] = []
    tracks: list[str] = []
    source_length_ticks = 0
    output_length_ticks = 0
    by_layer_output: dict[str, Path] = {}

    for index, layer_name in enumerate(layer_order, start=1):
        slot = mapping.layers[layer_name]
        layer_file = output_dir / _slot_export_file_name(index, slot.track_name, slot.note_name)
        layer_track_name = slot.track_name
        layer_hits_only = {layer_name: layered_hits.get(layer_name, [])}

        src_ticks, out_ticks, _tracks = _build_single_track_midi(
            layered_hits=layer_hits_only,
            layer_order=[layer_name],
            output_path=layer_file,
            ticks_per_beat=ticks_per_beat,
            bpm_used=bpm_used,
            duration_sec=duration_sec,
            channel=channel,
            track_name=layer_track_name,
        )
        source_length_ticks = max(source_length_ticks, src_ticks)
        output_length_ticks = max(output_length_ticks, out_ticks)
        created_files.append(layer_file)
        tracks.append(layer_track_name)
        by_layer_output[layer_name] = layer_file

    return source_length_ticks, output_length_ticks, tracks, created_files, by_layer_output


def _build_midi(
    *,
    layered_hits: dict[str, list[DrumLayerHit]],
    layer_order: list[str],
    mapping: DrumLayerMapping,
    output_layout: OutputLayout,
    output_path: Path,
    output_dir: Path,
    ticks_per_beat: int,
    bpm_used: float,
    duration_sec: float,
    channel: int,
    track_name: str,
) -> tuple[int, int, list[str], list[Path], dict[str, Path]]:
    if output_layout == "single-track":
        src, out, tracks = _build_single_track_midi(
            layered_hits=layered_hits,
            layer_order=layer_order,
            output_path=output_path,
            ticks_per_beat=ticks_per_beat,
            bpm_used=bpm_used,
            duration_sec=duration_sec,
            channel=channel,
            track_name=track_name,
        )
        return src, out, tracks, [output_path], {}

    if output_layout == "multitrack":
        src, out, tracks = _build_multitrack_midi(
            layered_hits=layered_hits,
            layer_order=layer_order,
            mapping=mapping,
            output_path=output_path,
            ticks_per_beat=ticks_per_beat,
            bpm_used=bpm_used,
            duration_sec=duration_sec,
            channel=channel,
        )
        return src, out, tracks, [output_path], {}

    return _build_separate_files_midi(
        layered_hits=layered_hits,
        layer_order=layer_order,
        mapping=mapping,
        output_dir=output_dir,
        ticks_per_beat=ticks_per_beat,
        bpm_used=bpm_used,
        duration_sec=duration_sec,
        channel=channel,
    )


def _count_layer_hits(layered_hits: dict[str, list[DrumLayerHit]]) -> dict[str, int]:
    return {layer: len(hits) for layer, hits in layered_hits.items()}


def _count_layer_output_pitch_counts(
    layered_hits: dict[str, list[DrumLayerHit]],
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for layer in sorted(layered_hits):
        notes: dict[str, int] = {}
        for hit in layered_hits.get(layer, []):
            note = str(int(hit.target_note))
            notes[note] = notes.get(note, 0) + 1
        summary[layer] = dict(sorted(notes.items(), key=lambda item: int(item[0])))
    return summary


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
            layer_name=hit.layer_name,
            semantic_layer=hit.semantic_layer,
            target_track_name=hit.target_track_name,
            target_note_name=hit.target_note_name,
            output_file=hit.output_file,
            mapping_name=hit.mapping_name,
            primary_slot_assignment=bool(hit.primary_slot_assignment),
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
            same_layer_duplicate=bool(hit.same_layer_duplicate),
            cross_layer_allowed=bool(hit.cross_layer_allowed),
            nearest_previous_same_layer_ms=(
                None if hit.nearest_previous_same_layer_ms is None else float(hit.nearest_previous_same_layer_ms)
            ),
            simultaneous_layers_at_time=str(hit.simultaneous_layers_at_time),
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
                "detector_family",
                "layer_name",
                "semantic_layer",
                "target_track_name",
                "target_note_name",
                "candidate_class",
                "accepted_class",
                "class_confidence",
                "competing_class",
                "competing_class_score",
                "accepted",
                "rejection_reason",
                "same_layer_duplicate",
                "cross_layer_allowed",
                "nearest_previous_same_layer_ms",
                "simultaneous_layers_at_time",
                "merged_with_transient_id",
                "low_peak_strength",
                "mid_peak_strength",
                "high_peak_strength",
                "attack_score",
                "decay_score",
                "band_dominance_score",
                "mapping_name",
                "primary_slot_assignment",
                "output_file",
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
                    "detector_family": hit.detector_name,
                    "layer_name": hit.layer_name,
                    "semantic_layer": hit.semantic_layer,
                    "target_track_name": hit.target_track_name,
                    "target_note_name": hit.target_note_name,
                    "candidate_class": hit.candidate_class,
                    "accepted_class": hit.accepted_class,
                    "class_confidence": hit.class_confidence,
                    "competing_class": hit.competing_class,
                    "competing_class_score": hit.competing_class_score,
                    "accepted": str(hit.accepted).lower(),
                    "rejection_reason": hit.rejection_reason,
                    "same_layer_duplicate": str(hit.same_layer_duplicate).lower(),
                    "cross_layer_allowed": str(hit.cross_layer_allowed).lower(),
                    "nearest_previous_same_layer_ms": hit.nearest_previous_same_layer_ms,
                    "simultaneous_layers_at_time": hit.simultaneous_layers_at_time,
                    "merged_with_transient_id": hit.merged_with_transient_id,
                    "low_peak_strength": hit.low_peak_strength,
                    "mid_peak_strength": hit.mid_peak_strength,
                    "high_peak_strength": hit.high_peak_strength,
                    "attack_score": hit.attack_score,
                    "decay_score": hit.decay_score,
                    "band_dominance_score": hit.band_dominance_score,
                    "mapping_name": hit.mapping_name,
                    "primary_slot_assignment": str(hit.primary_slot_assignment).lower(),
                    "output_file": hit.output_file,
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

    output_file, output_dir = _resolve_output_target(params, wav_file)

    map_definition = _resolve_target_map_definition(params)
    class_targets = _resolve_class_targets(map_definition, params)
    settings = _resolve_profile_settings(params)

    layer_mapping, duplicate_target_notes_by_note, mapping_warnings = _resolve_layer_mapping(params)

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
    warnings.extend(mapping_warnings)
    if params.mapping_file is not None and layer_mapping.c1_midi_note != params.c1_midi_note:
        warnings.append(
            "Mapping c1_midi_note differs from CLI --c1-midi-note; mapping-local value is used for note-name conversion."
        )
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
            _suppress_hit(
                hit,
                "tom_not_clear",
                duplicate_suppressed_by_class=duplicate_suppressed_by_class,
                rejected_by_reason=rejected_by_reason,
            )
            raw_hits.append(hit)
            continue

        if _canonical_class(hit.class_name) == "unknown" and not params.emit_unknown:
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
    merge_conflicts = 0
    accepted_hits: list[_DetectedHit] = []
    for layer_name, layer_hits in _split_hits_by_layer(premerge_hits).items():
        if not layer_hits:
            continue

        merged_hits, layer_conflicts = _merge_same_transient_candidates(
            layer_hits,
            window_sec=same_transient_window_sec,
            duplicate_suppressed_by_class=duplicate_suppressed_by_class,
            rejected_by_reason=rejected_by_reason,
        )
        merge_conflicts += layer_conflicts

        spaced_hits = _apply_global_min_spacing(
            merged_hits,
            min_spacing_sec=min_hit_spacing_sec,
            duplicate_suppressed_by_class=duplicate_suppressed_by_class,
            rejected_by_reason=rejected_by_reason,
        )
        filtered_hits = _apply_class_refractory(
            spaced_hits,
            class_refractory_sec=class_refractory_sec,
            duplicate_suppressed_by_class=duplicate_suppressed_by_class,
            rejected_by_reason=rejected_by_reason,
        )
        accepted_hits.extend(filtered_hits)

    accepted_hits = sorted(accepted_hits, key=lambda item: item.onset_sec)
    _assign_nearest_same_class_ms(sorted(raw_hits, key=lambda item: item.onset_sec))
    _annotate_simultaneous_layers(sorted(raw_hits, key=lambda item: item.onset_sec), window_sec=same_transient_window_sec)

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

    emitted_hits = sorted(emitted_hits, key=lambda item: item.onset_sec)
    cross_layer_simultaneous_hit_count = _annotate_simultaneous_layers(
        emitted_hits,
        window_sec=same_transient_window_sec,
    )

    for hit in raw_hits:
        if not hit.suppressed:
            _increment_counter(detector_accepted_counts, hit.detector_name)
        else:
            _increment_counter(detector_rejected_counts, hit.detector_name)

    _assign_output_velocities(emitted_hits)
    layered_hits = _to_layered_hits(
        emitted_hits,
        mapping=layer_mapping,
        snare_target=params.snare_target,
        warnings=warnings,
    )

    selected_layers, disabled_layers, skipped_layers = _select_export_layers(
        mapping=layer_mapping,
        layered_hits=layered_hits,
        write_empty_layers=params.write_empty_layers,
    )

    effective_output_layout: OutputLayout = (
        "separate-files" if params.separate_files else params.output_layout
    )
    if params.separate_files and params.output_layout != "separate-files":
        warnings.append("--separate-files is deprecated; using output layout 'separate-files'.")

    if effective_output_layout in {"single-track", "multitrack"} and not selected_layers:
        ordered = ordered_layers(layer_mapping)
        if ordered:
            selected_layers = [ordered[0]]
            skipped_layers.pop(ordered[0], None)
            warnings.append(
                "No populated enabled semantic layer was selected for export; wrote synchronized empty container MIDI."
            )

    export_layered_hits: dict[str, list[DrumLayerHit]] = {}
    for layer_name in selected_layers:
        slot = layer_mapping.layers[layer_name]
        if slot.enabled:
            export_layered_hits[layer_name] = list(layered_hits.get(layer_name, []))
        else:
            export_layered_hits[layer_name] = []

    suppressed_duplicate_count = int(sum(duplicate_suppressed_by_class.values()))
    same_layer_suppressed_count = int(
        sum(1 for hit in raw_hits if hit.suppressed and bool(hit.same_layer_duplicate))
    )
    cross_layer_suppressed_count = int(
        sum(
            1
            for hit in raw_hits
            if hit.suppressed
            and not bool(hit.same_layer_duplicate)
            and hit.suppression_reason in {"merge_conflict", "global_min_spacing"}
        )
    )

    class_counts = _count_classes(emitted_hits)
    layer_counts = _count_layer_hits(layered_hits)
    layer_output_pitch_counts = _count_layer_output_pitch_counts(layered_hits)
    populated_semantic_layers, unpopulated_enabled_layers, unpopulated_disabled_layers = _layer_statistics(
        layer_mapping,
        layered_hits,
    )
    duplicate_interval_summary = _duplicate_interval_summary(emitted_hits, class_refractory_ms)
    too_dense_warning, density_warnings = _evaluate_density_warnings(
        duration_sec=duration_sec,
        class_counts=class_counts,
        duplicate_interval_summary=duplicate_interval_summary,
    )
    warnings.extend(density_warnings)

    if cross_layer_suppressed_count > 0:
        warnings.append(
            "Cross-layer suppression detected; inspect debug CSV for reasons because layered hits should normally coexist."
        )

    if not params.emit_unknown:
        warnings.append("Unknown hits are skipped by default; use --emit-unknown to include them.")

    output_tracks_created: list[str] = [layer_mapping.layers[layer].track_name for layer in selected_layers]
    created_files: list[Path] = []
    layer_output_files: dict[str, Path] = {}

    synchronization_preserved = True
    if effective_output_layout == "separate-files":
        output_tracks_created = [layer_mapping.layers[layer].track_name for layer in selected_layers]

    if not params.dry_run:
        track_name = f"drums_from_audio_{params.target_map.replace('-', '_')}"
        source_length_ticks, output_length_ticks, output_tracks_created, created_files, layer_output_files = _build_midi(
            layered_hits=export_layered_hits,
            layer_order=selected_layers,
            mapping=layer_mapping,
            output_layout=effective_output_layout,
            output_path=output_file,
            output_dir=output_dir,
            ticks_per_beat=params.ticks_per_beat,
            bpm_used=bpm_used,
            duration_sec=duration_sec,
            channel=params.channel,
            track_name=track_name,
        )
        synchronization_preserved = source_length_ticks == output_length_ticks

        if effective_output_layout == "separate-files":
            for hit in emitted_hits:
                if hit.semantic_layer in layer_output_files:
                    hit.output_file = str(layer_output_files[hit.semantic_layer])
        else:
            for hit in emitted_hits:
                if hit.semantic_layer in selected_layers:
                    hit.output_file = str(output_file)
    else:
        ticks_per_second = (params.ticks_per_beat * bpm_used) / 60.0
        source_length_ticks = max(0, int(round(duration_sec * ticks_per_second)))
        output_length_ticks = source_length_ticks
        synchronization_preserved = source_length_ticks == output_length_ticks

    exported_hits: list[_DetectedHit] = [
        hit
        for hit in emitted_hits
        if hit.semantic_layer in export_layered_hits
        and hit.semantic_layer in layer_mapping.layers
        and layer_mapping.layers[hit.semantic_layer].enabled
    ]

    if params.debug_csv is not None:
        _write_debug_csv(params.debug_csv, sorted(raw_hits, key=lambda item: item.onset_sec))

    safe_duration = max(duration_sec, 1e-9)
    notes_per_second = float(sum(class_counts.values()) / safe_duration)

    ordered_layer_names = ordered_layers(layer_mapping)
    layer_target_notes = {
        layer: int(layer_mapping.layers[layer].note)
        for layer in ordered_layer_names
    }
    layer_target_note_names = {
        layer: str(layer_mapping.layers[layer].note_name)
        for layer in ordered_layer_names
    }
    layer_track_names = {
        layer: str(layer_mapping.layers[layer].track_name)
        for layer in ordered_layer_names
    }
    primary_slot_assignment_used = all(
        (not hit.semantic_layer) or hit.semantic_layer.endswith("_1")
        for hit in emitted_hits
    )
    if primary_slot_assignment_used:
        warnings.append("Primary slot assignment used: detector populated only primary semantic slots.")

    report = AudioDrumExtractionReport(
        wav_file=str(wav_file),
        output_file=(
            None
            if params.dry_run or effective_output_layout == "separate-files"
            else str(output_file)
        ),
        duration_sec=float(duration_sec),
        sample_rate=int(sample_rate),
        detected_bpm=float(detected_bpm) if detected_bpm is not None else None,
        bpm_used=float(bpm_used),
        bpm_source=bpm_source,
        detection_mode=params.detection_mode,
        output_layout=effective_output_layout,
        output_dir=str(output_dir),
        mapping_file=(None if params.mapping_file is None else str(params.mapping_file)),
        mapping_name=layer_mapping.name,
        track_order=output_tracks_created,
        created_files=[str(path) for path in created_files],
        skipped_layers=dict(sorted(skipped_layers.items())),
        disabled_layers=disabled_layers,
        write_empty_layers=bool(params.write_empty_layers),
        duplicate_target_notes=duplicate_target_notes_by_note,
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
        output_note_counts=_count_output_notes(exported_hits),
        output_pitch_counts=_count_output_notes(exported_hits),
        layer_counts=layer_counts,
        layer_target_notes=layer_target_notes,
        layer_target_note_names=layer_target_note_names,
        layer_track_names=layer_track_names,
        layer_output_pitch_counts=layer_output_pitch_counts,
        populated_semantic_layers=populated_semantic_layers,
        unpopulated_enabled_layers=unpopulated_enabled_layers,
        unpopulated_disabled_layers=unpopulated_disabled_layers,
        primary_slot_assignment_used=bool(primary_slot_assignment_used),
        cross_layer_simultaneous_hit_count=int(cross_layer_simultaneous_hit_count),
        same_layer_suppressed_count=same_layer_suppressed_count,
        cross_layer_suppressed_count=cross_layer_suppressed_count,
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
        c1_midi_note=int(layer_mapping.c1_midi_note),
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
