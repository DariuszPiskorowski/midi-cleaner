from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import mido
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
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
    "crash_or_cymbal",
    "tom_or_perc",
    "unknown",
]
SnareTarget = Literal["sn1", "sn2", "clap"]

DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_MIN_ONSET_STRENGTH = 0.20
DEFAULT_BPM = 120.0

_CLASS_FILE_NAMES: dict[str, str] = {
    "kick": "kick.mid",
    "snare_or_clap": "snare_clap.mid",
    "hat": "hat.mid",
    "crash_or_cymbal": "cymbal.mid",
    "tom_or_perc": "tom_perc.mid",
}

_CLASS_NOTE_DURATIONS_SEC: dict[DrumClass, float] = {
    "kick": 0.10,
    "snare_or_clap": 0.10,
    "hat": 0.06,
    "crash_or_cymbal": 0.32,
    "tom_or_perc": 0.14,
    "unknown": 0.08,
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


@dataclass
class PerHitSummary:
    onset_sec: float
    class_name: DrumClass
    target_note: int
    velocity: int
    confidence: float
    low_energy_ratio: float
    mid_energy_ratio: float
    high_energy_ratio: float


@dataclass
class AudioDrumExtractionReport:
    wav_file: str
    output_file: str | None
    duration_sec: float
    sample_rate: int
    detected_bpm: float | None
    bpm_used: float
    onset_count: int
    class_counts: dict[str, int]
    output_note_counts: dict[str, int]
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
                "class": item.class_name,
                "target_note": item.target_note,
                "velocity": item.velocity,
                "confidence": item.confidence,
                "low_energy_ratio": item.low_energy_ratio,
                "mid_energy_ratio": item.mid_energy_ratio,
                "high_energy_ratio": item.high_energy_ratio,
            }
            for item in self.per_hit_summary
        ]
        return payload


@dataclass
class _DetectedHit:
    onset_sec: float
    onset_strength: float
    class_name: DrumClass
    low_energy_ratio: float
    mid_energy_ratio: float
    high_energy_ratio: float
    confidence: float
    target_note: int
    velocity: int


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


def _onset_strength_envelope(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int]:
    frame_size = 1024
    hop_size = 256
    frames = _frame_signal(audio, frame_size=frame_size, hop_size=hop_size)
    window = hann(frame_size, sym=False)

    magnitudes = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
    if len(magnitudes) <= 1:
        return np.zeros(len(magnitudes), dtype=np.float64), frame_size, hop_size

    diff = np.diff(magnitudes, axis=0)
    flux = np.sum(np.maximum(0.0, diff), axis=1)
    raw = np.concatenate(([0.0], flux.astype(np.float64)))

    p95 = float(np.percentile(raw, 95)) if np.any(raw > 0.0) else 1.0
    p95 = max(p95, 1e-9)
    normalized = np.clip(raw / p95, 0.0, 2.0)

    return normalized, frame_size, hop_size


def _detect_onsets(
    onset_strength: np.ndarray,
    sample_rate: int,
    hop_size: int,
    min_onset_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(onset_strength) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    distance_frames = max(1, int(round((0.035 * sample_rate) / hop_size)))
    peaks, peak_props = find_peaks(
        onset_strength,
        height=min_onset_strength,
        prominence=max(0.03, min_onset_strength * 0.40),
        distance=distance_frames,
    )

    if len(peaks) == 0:
        relaxed = max(0.03, min_onset_strength * 0.65)
        peaks, peak_props = find_peaks(
            onset_strength,
            height=relaxed,
            prominence=max(0.02, relaxed * 0.35),
            distance=distance_frames,
        )

    times = peaks.astype(np.float64) * (hop_size / float(sample_rate))
    strengths = peak_props.get("peak_heights", np.zeros(len(peaks), dtype=np.float64))
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
) -> tuple[float, float, float, float, float]:
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

    return low_ratio, mid_ratio, high_ratio, centroid_hz, rolloff_hz


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

    if high_ratio >= 0.58 and (centroid_hz >= 5200.0 or rolloff_hz >= 9000.0):
        if high_ratio >= 0.75 or rolloff_hz >= 11000.0:
            base_confidence = 0.65 + (high_ratio - 0.58)
            return "crash_or_cymbal", min(1.0, base_confidence + 0.20 * onset_strength)
        base_confidence = 0.62 + (high_ratio - 0.58)
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


def _velocity_from_hit(onset_strength: float, confidence: float) -> int:
    scaled = 30.0 + (70.0 * min(1.0, onset_strength)) + (25.0 * min(1.0, confidence))
    return max(1, min(127, int(round(scaled))))


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
            "crash_or_cymbal": 49,
            "tom_or_perc": 45,
            "unknown": 39,
        }

    return _infer_custom_class_targets(map_definition, snare_target=params.snare_target)


def _hit_to_tick(onset_sec: float, ticks_per_second: float) -> int:
    return max(0, int(round(onset_sec * ticks_per_second)))


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
                "onset_strength",
                "class",
                "target_note",
                "velocity",
                "confidence",
                "low_energy_ratio",
                "mid_energy_ratio",
                "high_energy_ratio",
            ],
        )
        writer.writeheader()
        for hit in hits:
            writer.writerow(
                {
                    "onset_sec": hit.onset_sec,
                    "onset_strength": hit.onset_strength,
                    "class": hit.class_name,
                    "target_note": hit.target_note,
                    "velocity": hit.velocity,
                    "confidence": hit.confidence,
                    "low_energy_ratio": hit.low_energy_ratio,
                    "mid_energy_ratio": hit.mid_energy_ratio,
                    "high_energy_ratio": hit.high_energy_ratio,
                }
            )


def _validate_params(params: AudioDrumExtractionParameters) -> None:
    if params.channel < 0 or params.channel > 15:
        raise AudioDrumExtractionError("channel must be in range 0..15.")

    if params.min_onset_strength <= 0.0:
        raise AudioDrumExtractionError("--min-onset-strength must be > 0.")

    if params.bpm is not None and params.bpm <= 0.0:
        raise AudioDrumExtractionError("--bpm must be > 0.")

    if params.snare_target not in {"sn1", "sn2", "clap"}:
        raise AudioDrumExtractionError("--snare-target must be one of: sn1, sn2, clap.")

    if params.c1_midi_note < 0 or params.c1_midi_note > 127:
        raise AudioDrumExtractionError("--c1-midi-note must be in range 0..127.")


def _count_output_notes(hits: list[_DetectedHit]) -> dict[str, int]:
    counts: dict[int, int] = {}
    for hit in hits:
        note = int(hit.target_note)
        counts[note] = counts.get(note, 0) + 1
    return {str(note): count for note, count in sorted(counts.items())}


def _count_classes(hits: list[_DetectedHit]) -> dict[str, int]:
    counts: dict[str, int] = {
        "kick": 0,
        "snare_or_clap": 0,
        "hat": 0,
        "crash_or_cymbal": 0,
        "tom_or_perc": 0,
        "unknown": 0,
    }
    for hit in hits:
        counts[hit.class_name] += 1
    return counts


def _to_per_hit_summary(hits: list[_DetectedHit]) -> list[PerHitSummary]:
    return [
        PerHitSummary(
            onset_sec=float(hit.onset_sec),
            class_name=hit.class_name,
            target_note=int(hit.target_note),
            velocity=int(hit.velocity),
            confidence=float(hit.confidence),
            low_energy_ratio=float(hit.low_energy_ratio),
            mid_energy_ratio=float(hit.mid_energy_ratio),
            high_energy_ratio=float(hit.high_energy_ratio),
        )
        for hit in hits
    ]


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

    onset_strength_envelope, _frame_size, hop_size = _onset_strength_envelope(
        mono,
        sample_rate,
    )
    onset_times, onset_strengths = _detect_onsets(
        onset_strength_envelope,
        sample_rate,
        hop_size,
        params.min_onset_strength,
    )

    detected_bpm = _estimate_bpm(onset_times)
    bpm_used = float(params.bpm) if params.bpm is not None else float(detected_bpm or DEFAULT_BPM)

    warnings: list[str] = []
    if detected_bpm is None:
        warnings.append("Could not confidently estimate BPM from onsets; defaulted to stable fallback.")
    if len(onset_times) == 0:
        warnings.append("No onsets detected above threshold.")

    hits: list[_DetectedHit] = []
    for onset_sec, onset_strength in zip(onset_times.tolist(), onset_strengths.tolist()):
        low_ratio, mid_ratio, high_ratio, centroid_hz, rolloff_hz = _extract_hit_spectral_features(
            mono,
            sample_rate,
            onset_sec,
        )
        class_name, confidence = _classify_hit(
            onset_strength=float(onset_strength),
            low_ratio=low_ratio,
            mid_ratio=mid_ratio,
            high_ratio=high_ratio,
            centroid_hz=centroid_hz,
            rolloff_hz=rolloff_hz,
        )

        target_note = int(class_targets[class_name])
        if class_name == "tom_or_perc" and params.target_map == "ujam-candy" and high_ratio > 0.45:
            candy_layout = resolve_ujam_candy_layout_notes(params.c1_midi_note)
            target_note = candy_layout["F2"]

        hits.append(
            _DetectedHit(
                onset_sec=float(onset_sec),
                onset_strength=float(onset_strength),
                class_name=class_name,
                low_energy_ratio=float(low_ratio),
                mid_energy_ratio=float(mid_ratio),
                high_energy_ratio=float(high_ratio),
                confidence=float(confidence),
                target_note=target_note,
                velocity=_velocity_from_hit(float(onset_strength), confidence),
            )
        )

    hits.sort(key=lambda item: item.onset_sec)

    if params.debug_csv is not None:
        _write_debug_csv(params.debug_csv, hits)

    output_file = params.output_file
    synchronization_preserved = True

    if not params.dry_run:
        track_name = f"drums_from_audio_{params.target_map.replace('-', '_')}"
        source_length_ticks, output_length_ticks = _build_midi(
            hits,
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
                "crash_or_cymbal": [],
                "tom_or_perc": [],
            }
            for hit in hits:
                bucket = hit.class_name
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

    report = AudioDrumExtractionReport(
        wav_file=str(wav_file),
        output_file=None if params.dry_run else str(output_file),
        duration_sec=float(duration_sec),
        sample_rate=int(sample_rate),
        detected_bpm=float(detected_bpm) if detected_bpm is not None else None,
        bpm_used=float(bpm_used),
        onset_count=len(hits),
        class_counts=_count_classes(hits),
        output_note_counts=_count_output_notes(hits),
        target_map=map_definition.name,
        c1_midi_note=int(params.c1_midi_note),
        synchronization_preserved=bool(synchronization_preserved),
        warnings=warnings,
        per_hit_summary=_to_per_hit_summary(hits),
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
