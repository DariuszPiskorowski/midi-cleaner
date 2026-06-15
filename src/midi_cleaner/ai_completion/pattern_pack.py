from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import math
import statistics

import mido

from midi_cleaner.alignment.models import AudioAlignedNoteDocument
from midi_cleaner.audio.models import AudioFeatureDocument
from midi_cleaner.dsp.models import DspAudioFeatureDocument
from midi_cleaner.midi.models import NoteEventDocument
from midi_cleaner.pitch.models import BassPitchContourDocument
from midi_cleaner.refinement.models import RefinedNoteDocument


class PatternPackBuildError(Exception):
    """Raised when pattern pack inputs are invalid or missing."""


@dataclass(frozen=True)
class BasePatternNote:
    note_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    pitch_midi: int
    velocity: int
    confidence: float | None
    source: str | None
    reasons: list[str]


@dataclass(frozen=True)
class AllowedCompletionRegion:
    region_id: str
    start_sec: float
    end_sec: float
    write_start_sec: float
    write_end_sec: float
    reason: str
    context_before_start_sec: float
    context_after_end_sec: float
    context_window_before_sec: float
    context_window_after_sec: float
    reference_notes_before: list[str]
    reference_notes_after: list[str]
    notes_before: list[dict[str, object]]
    notes_after: list[dict[str, object]]
    local_pitch_set: list[int]
    local_pitch_names: list[str]
    local_pitch_range: dict[str, int]
    allowed_pitch_range: dict[str, int]
    preferred_pitches: list[int]
    forbidden_pitches: list[int]
    allow_pitch_outside_local_set: bool
    estimated_key_or_scale: str
    rhythmic_pattern_summary: dict[str, object]
    local_rhythm_intervals_sec: list[float]
    detected_local_motif: dict[str, object]
    motif_confidence: float
    optional_region: bool
    expected_note_count_min: int
    expected_note_count_max: int
    density_limit_notes_per_sec: float
    min_note_duration_sec: float
    max_note_duration_sec: float
    no_notes_outside_region: bool
    instruction: str


@dataclass(frozen=True)
class PatternPackBuildResult:
    pattern_pack: dict[str, object]
    base_notes: list[BasePatternNote]
    duration_sec: float
    ticks_per_beat: int
    tempo_us_per_beat: int
    base_note_source: str
    allowed_completion_regions: list[AllowedCompletionRegion]
    warnings: list[str]


def build_pattern_pack(project_dir: Path, layer: str = "bass") -> PatternPackBuildResult:
    analysis_dir = project_dir / "analysis"
    working_midi_path = project_dir / "midi" / "working" / "working.mid"
    audio_features_path = analysis_dir / "audio_features.json"

    if not working_midi_path.exists() or not working_midi_path.is_file():
        raise PatternPackBuildError(f"Missing working MIDI: {working_midi_path}")
    if not audio_features_path.exists() or not audio_features_path.is_file():
        raise PatternPackBuildError(f"Missing audio features: {audio_features_path}")

    base_notes, base_note_source = _load_base_notes(analysis_dir=analysis_dir, working_midi_path=working_midi_path)
    if not base_notes:
        raise PatternPackBuildError("Base note set is empty. Cannot build pattern_pack.json.")

    audio_features = AudioFeatureDocument.model_validate_json(
        audio_features_path.read_text(encoding="utf-8")
    )

    dsp_doc = _load_optional_dsp(analysis_dir / "audio_features_dsp.json")
    pitch_doc = _load_optional_pitch(analysis_dir / "bass_pitch_contour.json")

    ticks_per_beat, tempo_us_per_beat, tempo_bpm = _read_timing_from_working_midi(working_midi_path)
    duration_sec = float(audio_features.duration_sec)

    warnings: list[str] = []
    if base_note_source.endswith("note_events.json"):
        warnings.append(
            "Pattern pack used note_events timing source because refined/repaired notes were unavailable."
        )
    elif base_note_source.endswith("working.mid"):
        warnings.append("Pattern pack used working.mid as canonical base note source.")

    base_notes_records = [_base_note_record(item) for item in base_notes]
    base_notes_records, trunc_warnings = _truncate_base_notes_if_needed(base_notes_records)
    warnings.extend(trunc_warnings)

    audio_activity_regions = _build_audio_activity_regions(
        audio_features=audio_features,
        dsp_doc=dsp_doc,
        pitch_doc=pitch_doc,
    )
    pitch_contour_summary = _build_pitch_contour_summary(
        pitch_doc=pitch_doc,
        duration_sec=duration_sec,
    )
    pattern_windows = _build_pattern_windows(
        base_notes=base_notes,
        duration_sec=duration_sec,
        audio_activity_regions=audio_activity_regions,
        pitch_contour_summary=pitch_contour_summary,
    )
    allowed_completion_regions = _detect_allowed_completion_regions(
        base_notes=base_notes,
        audio_activity_regions=audio_activity_regions,
        pitch_contour_summary=pitch_contour_summary,
        duration_sec=duration_sec,
    )

    if not allowed_completion_regions:
        warnings.append(
            "No allowed completion regions were detected. AI completion should return zero notes."
        )

    pattern_pack: dict[str, object] = {
        "version": "1.0",
        "track_role": layer,
        "base_note_source": base_note_source,
        "timeline": {
            "duration_sec": round(duration_sec, 6),
            "time_origin": "wav_seconds",
            "ticks_per_beat": int(ticks_per_beat),
            "tempo_bpm": (round(tempo_bpm, 6) if tempo_bpm is not None else None),
            "midi_source": "working.mid",
        },
        "base_midi_summary": _build_base_midi_summary(base_notes=base_notes, duration_sec=duration_sec),
        "base_notes": base_notes_records,
        "audio_activity_regions": audio_activity_regions,
        "pitch_contour_summary": pitch_contour_summary,
        "pattern_windows": pattern_windows,
        "allowed_completion_regions": [
            _allowed_region_record(region) for region in allowed_completion_regions
        ],
        "instructions_for_ai": {
            "goal": (
                "Generate an additional synchronized bass MIDI completion track that adds "
                "missing/continuation fragments of the existing pattern. "
                "Do not rewrite or duplicate the base MIDI."
            ),
            "output_time_unit": "seconds",
            "output_file_expected": "bass_ai_completion.mid",
        },
    }

    if warnings:
        pattern_pack["pack_warnings"] = warnings

    return PatternPackBuildResult(
        pattern_pack=pattern_pack,
        base_notes=base_notes,
        duration_sec=duration_sec,
        ticks_per_beat=ticks_per_beat,
        tempo_us_per_beat=tempo_us_per_beat,
        base_note_source=base_note_source,
        allowed_completion_regions=allowed_completion_regions,
        warnings=warnings,
    )


def _load_base_notes(
    *,
    analysis_dir: Path,
    working_midi_path: Path,
) -> tuple[list[BasePatternNote], str]:
    source_candidates = [
        working_midi_path,
        analysis_dir / "final_repaired_note_events.json",
        analysis_dir / "repaired_refined_note_events.json",
        analysis_dir / "refined_note_events.json",
        analysis_dir / "audio_aligned_note_events.json",
        analysis_dir / "note_events.json",
    ]

    for path in source_candidates:
        if not path.exists() or not path.is_file():
            continue

        if path.name == "working.mid":
            notes = _load_base_notes_from_working_midi(path)
            if notes:
                return notes, str(path)
            continue

        if path.name.endswith("repaired_note_events.json") or path.name == "refined_note_events.json":
            document = RefinedNoteDocument.model_validate_json(path.read_text(encoding="utf-8"))
            notes = [
                BasePatternNote(
                    note_id=item.note_id,
                    start_sec=float(item.refined_start_sec),
                    end_sec=float(item.refined_end_sec),
                    duration_sec=float(item.refined_duration_sec),
                    pitch_midi=int(item.pitch_midi),
                    velocity=int(item.velocity),
                    confidence=float(item.refinement_confidence),
                    source=item.source,
                    reasons=list(item.reasons),
                )
                for item in document.notes
            ]
            return notes, str(path)

        if path.name == "audio_aligned_note_events.json":
            document = AudioAlignedNoteDocument.model_validate_json(path.read_text(encoding="utf-8"))
            notes = [
                BasePatternNote(
                    note_id=item.note_id,
                    start_sec=float(item.aligned_start_sec),
                    end_sec=float(item.aligned_end_sec),
                    duration_sec=float(item.aligned_duration_sec),
                    pitch_midi=int(item.pitch_midi),
                    velocity=int(item.velocity),
                    confidence=float(item.alignment_confidence),
                    source=item.source,
                    reasons=list(item.reasons),
                )
                for item in document.notes
            ]
            return notes, str(path)

        if path.name == "note_events.json":
            document = NoteEventDocument.model_validate_json(path.read_text(encoding="utf-8"))
            notes = [
                BasePatternNote(
                    note_id=item.note_id,
                    start_sec=float(item.start_sec),
                    end_sec=float(item.end_sec),
                    duration_sec=float(item.duration_sec),
                    pitch_midi=int(item.pitch_midi),
                    velocity=int(item.velocity),
                    confidence=(float(item.confidence) if item.confidence is not None else None),
                    source=item.source,
                    reasons=[],
                )
                for item in document.notes
            ]
            return notes, str(path)

    raise PatternPackBuildError("Missing note source in analysis directory.")


def _load_base_notes_from_working_midi(path: Path) -> list[BasePatternNote]:
    midi = mido.MidiFile(str(path))
    ticks_per_beat = int(midi.ticks_per_beat)

    tempo_us_per_beat = 500000
    found_tempo = False
    for track in midi.tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo_us_per_beat = int(message.tempo)
                found_tempo = True
                break
        if found_tempo:
            break

    ticks_per_second = (float(ticks_per_beat) * 1_000_000.0) / float(tempo_us_per_beat)

    active_notes: dict[tuple[int, int], list[tuple[float, int]]] = {}
    emitted: list[BasePatternNote] = []
    note_counter = 0

    for track_index, track in enumerate(midi.tracks):
        abs_tick = 0
        for message in track:
            abs_tick += int(message.time)

            if message.type not in {"note_on", "note_off"}:
                continue

            note_value = int(getattr(message, "note", 0))
            channel = int(getattr(message, "channel", 0))
            velocity = int(getattr(message, "velocity", 0))
            key = (note_value, channel)
            current_sec = abs_tick / max(1e-9, ticks_per_second)

            is_on = message.type == "note_on" and velocity > 0
            if is_on:
                active_notes.setdefault(key, []).append((current_sec, velocity))
                continue

            starts = active_notes.get(key)
            if not starts:
                continue

            start_sec, start_velocity = starts.pop(0)
            end_sec = max(start_sec + 1e-6, current_sec)
            duration_sec = max(1e-6, end_sec - start_sec)
            note_counter += 1
            emitted.append(
                BasePatternNote(
                    note_id=(
                        f"working_t{track_index:02d}"
                        f"_ch{channel:02d}_p{note_value:03d}_n{note_counter:06d}"
                    ),
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    duration_sec=float(duration_sec),
                    pitch_midi=note_value,
                    velocity=max(1, min(127, int(start_velocity))),
                    confidence=None,
                    source="working.mid",
                    reasons=[],
                )
            )

    emitted.sort(key=lambda item: (item.start_sec, item.end_sec, item.pitch_midi))
    return emitted


def _load_optional_dsp(path: Path) -> DspAudioFeatureDocument | None:
    if not path.exists() or not path.is_file():
        return None
    return DspAudioFeatureDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _load_optional_pitch(path: Path) -> BassPitchContourDocument | None:
    if not path.exists() or not path.is_file():
        return None
    return BassPitchContourDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _read_timing_from_working_midi(path: Path) -> tuple[int, int, float | None]:
    midi = mido.MidiFile(str(path))
    ticks_per_beat = int(midi.ticks_per_beat)
    tempo_us_per_beat: int | None = None

    for track in midi.tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo_us_per_beat = int(message.tempo)
                break
        if tempo_us_per_beat is not None:
            break

    if tempo_us_per_beat is None:
        return ticks_per_beat, 500000, None

    tempo_bpm = 60_000_000.0 / float(tempo_us_per_beat)
    return ticks_per_beat, tempo_us_per_beat, tempo_bpm


def _base_note_record(note: BasePatternNote) -> dict[str, object]:
    return {
        "note_id": note.note_id,
        "start_sec": round(note.start_sec, 6),
        "end_sec": round(note.end_sec, 6),
        "duration_sec": round(note.duration_sec, 6),
        "pitch_midi": int(note.pitch_midi),
        "velocity": int(note.velocity),
        "confidence": (round(note.confidence, 6) if note.confidence is not None else None),
        "source": note.source,
        "reasons": list(note.reasons),
    }


def _truncate_base_notes_if_needed(
    base_notes_records: list[dict[str, object]],
    limit: int = 2000,
) -> tuple[list[dict[str, object]], list[str]]:
    if len(base_notes_records) <= limit:
        return base_notes_records, []

    step = max(1, math.ceil(len(base_notes_records) / float(limit)))
    truncated = base_notes_records[::step][:limit]
    warnings = [
        "base_notes were truncated for compactness; pattern_windows and summaries preserve context.",
        f"base_notes_total={len(base_notes_records)}, base_notes_included={len(truncated)}",
    ]
    return truncated, warnings


def _build_base_midi_summary(base_notes: list[BasePatternNote], duration_sec: float) -> dict[str, object]:
    pitches = [item.pitch_midi for item in base_notes]
    durations = sorted(item.duration_sec for item in base_notes)

    common_pitches_counter = Counter(pitches)
    common_pitches = [
        {"pitch_midi": int(pitch), "count": int(count)}
        for pitch, count in common_pitches_counter.most_common(8)
    ]

    density_by_section = []
    section_len = max(2.0, duration_sec / 8.0 if duration_sec > 0 else 2.0)
    start = 0.0
    while start < duration_sec + 1e-9:
        end = min(duration_sec, start + section_len)
        note_count = sum(1 for item in base_notes if item.start_sec < end and item.end_sec > start)
        seconds = max(1e-6, end - start)
        density_by_section.append(
            {
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "note_count": int(note_count),
                "notes_per_sec": round(note_count / seconds, 6),
            }
        )
        if end >= duration_sec:
            break
        start = end

    median_duration = 0.0
    if durations:
        mid = len(durations) // 2
        if len(durations) % 2 == 1:
            median_duration = durations[mid]
        else:
            median_duration = (durations[mid - 1] + durations[mid]) / 2.0

    return {
        "note_count": len(base_notes),
        "pitch_range": {
            "min": int(min(pitches)) if pitches else 0,
            "max": int(max(pitches)) if pitches else 0,
        },
        "median_note_duration_sec": round(median_duration, 6),
        "common_pitches": common_pitches,
        "density_by_section": density_by_section,
    }


def _build_audio_activity_regions(
    audio_features: AudioFeatureDocument,
    dsp_doc: DspAudioFeatureDocument | None,
    pitch_doc: BassPitchContourDocument | None,
) -> list[dict[str, object]]:
    threshold = max(1e-6, float(audio_features.global_features.rms) * 0.18)
    onset_threshold = 0.015

    regions: list[tuple[int, int]] = []
    start_idx: int | None = None
    frames = audio_features.frames
    for index, frame in enumerate(frames):
        is_active = frame.rms >= threshold or frame.onset_score >= onset_threshold
        if is_active and start_idx is None:
            start_idx = index
        if not is_active and start_idx is not None:
            regions.append((start_idx, index - 1))
            start_idx = None
    if start_idx is not None:
        regions.append((start_idx, len(frames) - 1))

    if not regions:
        return []

    dsp_frames = dsp_doc.frames if dsp_doc is not None else []
    pitch_frames = pitch_doc.frames if pitch_doc is not None else []

    payload: list[dict[str, object]] = []
    for start_idx, end_idx in regions:
        section = frames[start_idx : end_idx + 1]
        start_sec = float(section[0].start_sec)
        end_sec = float(section[-1].end_sec)
        rms_values = [float(item.rms) for item in section]
        onset_count = sum(1 for item in section if item.onset_score >= onset_threshold)

        region_dsp = [
            item
            for item in dsp_frames
            if item.start_sec < end_sec and item.end_sec > start_sec
        ]
        low_band_mean = (
            sum(float(item.low_band_rms) for item in region_dsp) / len(region_dsp)
            if region_dsp
            else None
        )
        harmonic_mean = (
            sum(float(item.harmonic_rms) for item in region_dsp) / len(region_dsp)
            if region_dsp
            else None
        )

        region_pitch = [
            item
            for item in pitch_frames
            if item.start_sec < end_sec and item.end_sec > start_sec
        ]
        pitch_candidates = [
            int(item.pitch_midi_rounded)
            for item in region_pitch
            if item.voiced and item.pitch_midi_rounded is not None
        ]
        dominant_pitch = None
        if pitch_candidates:
            dominant_pitch = Counter(pitch_candidates).most_common(1)[0][0]

        pitch_confidence = None
        if region_pitch:
            pitch_confidence = sum(float(item.pitch_confidence) for item in region_pitch) / len(region_pitch)

        payload.append(
            {
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
                "duration_sec": round(end_sec - start_sec, 6),
                "rms_mean": round(sum(rms_values) / len(rms_values), 8),
                "rms_peak": round(max(rms_values), 8),
                "low_band_mean": (
                    round(float(low_band_mean), 8) if low_band_mean is not None else None
                ),
                "harmonic_mean": (
                    round(float(harmonic_mean), 8) if harmonic_mean is not None else None
                ),
                "onset_count": int(onset_count),
                "dominant_pitch_midi": (int(dominant_pitch) if dominant_pitch is not None else None),
                "pitch_confidence": (
                    round(float(pitch_confidence), 6) if pitch_confidence is not None else None
                ),
            }
        )

    return payload


def _build_pitch_contour_summary(
    pitch_doc: BassPitchContourDocument | None,
    duration_sec: float,
) -> list[dict[str, object]]:
    if pitch_doc is None or not pitch_doc.frames:
        return []

    summary: list[dict[str, object]] = []
    window_sec = 1.0
    index = 0
    start_sec = 0.0

    while start_sec < duration_sec + 1e-9:
        end_sec = min(duration_sec, start_sec + window_sec)
        section = [
            frame
            for frame in pitch_doc.frames
            if frame.start_sec < end_sec and frame.end_sec > start_sec
        ]
        if section:
            voiced_frames = [item for item in section if item.voiced and item.pitch_midi_rounded is not None]
            dominant_pitch = None
            pitch_mean = None
            if voiced_frames:
                dominant_pitch = Counter(
                    int(item.pitch_midi_rounded) for item in voiced_frames
                ).most_common(1)[0][0]
                pitch_mean = sum(float(item.pitch_midi_float) for item in voiced_frames if item.pitch_midi_float is not None)
                pitch_mean = pitch_mean / len(voiced_frames)

            voiced_ratio = len(voiced_frames) / float(len(section))
            mean_conf = sum(float(item.pitch_confidence) for item in section) / float(len(section))

            summary.append(
                {
                    "start_sec": round(start_sec, 6),
                    "end_sec": round(end_sec, 6),
                    "dominant_pitch_midi": (
                        int(dominant_pitch) if dominant_pitch is not None else None
                    ),
                    "pitch_midi_mean": (
                        round(float(pitch_mean), 6) if pitch_mean is not None else None
                    ),
                    "voiced_ratio": round(float(voiced_ratio), 6),
                    "mean_confidence": round(float(mean_conf), 6),
                }
            )

        if end_sec >= duration_sec:
            break
        start_sec = end_sec
        index += 1
        _ = index

    return summary


def _build_pattern_windows(
    base_notes: list[BasePatternNote],
    duration_sec: float,
    audio_activity_regions: list[dict[str, object]],
    pitch_contour_summary: list[dict[str, object]],
) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    window_sec = max(2.0, duration_sec / 12.0 if duration_sec > 0 else 2.0)
    start_sec = 0.0
    window_index = 0

    while start_sec < duration_sec + 1e-9:
        end_sec = min(duration_sec, start_sec + window_sec)
        section_notes = [
            item
            for item in base_notes
            if item.start_sec < end_sec and item.end_sec > start_sec
        ]
        section_notes = sorted(section_notes, key=lambda item: item.start_sec)

        onsets = [round(item.start_sec, 6) for item in section_notes][:64]
        intervals: list[float] = []
        for idx in range(1, len(onsets)):
            intervals.append(round(onsets[idx] - onsets[idx - 1], 6))
        durations = [round(item.duration_sec, 6) for item in section_notes]
        common_durations = [
            float(value)
            for value, _count in Counter(durations).most_common(8)
        ]

        region_indices = [
            idx
            for idx, region in enumerate(audio_activity_regions)
            if float(region["start_sec"]) < end_sec and float(region["end_sec"]) > start_sec
        ]
        pitch_indices = [
            idx
            for idx, section in enumerate(pitch_contour_summary)
            if float(section["start_sec"]) < end_sec and float(section["end_sec"]) > start_sec
        ]

        windows.append(
            {
                "window_index": int(window_index),
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
                "base_notes": [item.note_id for item in section_notes],
                "audio_activity_region_indices": region_indices,
                "pitch_summary_indices": pitch_indices,
                "rhythmic_summary": {
                    "note_onsets_sec": onsets,
                    "intervals_sec": intervals[:64],
                    "common_durations_sec": common_durations,
                },
            }
        )

        if end_sec >= duration_sec:
            break
        start_sec = end_sec
        window_index += 1

    return windows


def _allowed_region_record(region: AllowedCompletionRegion) -> dict[str, object]:
    return {
        "region_id": region.region_id,
        "start_sec": round(region.start_sec, 6),
        "end_sec": round(region.end_sec, 6),
        "write_start_sec": round(region.write_start_sec, 6),
        "write_end_sec": round(region.write_end_sec, 6),
        "reason": region.reason,
        "context_before_start_sec": round(region.context_before_start_sec, 6),
        "context_after_end_sec": round(region.context_after_end_sec, 6),
        "context_window_before_sec": round(region.context_window_before_sec, 6),
        "context_window_after_sec": round(region.context_window_after_sec, 6),
        "reference_notes_before": list(region.reference_notes_before),
        "reference_notes_after": list(region.reference_notes_after),
        "notes_before": [dict(item) for item in region.notes_before],
        "notes_after": [dict(item) for item in region.notes_after],
        "local_pitch_set": [int(value) for value in region.local_pitch_set],
        "local_pitch_names": [str(value) for value in region.local_pitch_names],
        "local_pitch_range": {
            "min": int(region.local_pitch_range["min"]),
            "max": int(region.local_pitch_range["max"]),
        },
        "allowed_pitch_range": {
            "min": int(region.allowed_pitch_range["min"]),
            "max": int(region.allowed_pitch_range["max"]),
        },
        "preferred_pitches": [int(value) for value in region.preferred_pitches],
        "forbidden_pitches": [int(value) for value in region.forbidden_pitches],
        "allow_pitch_outside_local_set": bool(region.allow_pitch_outside_local_set),
        "estimated_key_or_scale": region.estimated_key_or_scale,
        "rhythmic_pattern_summary": dict(region.rhythmic_pattern_summary),
        "local_rhythm_intervals_sec": [
            round(float(value), 6) for value in region.local_rhythm_intervals_sec
        ],
        "detected_local_motif": dict(region.detected_local_motif),
        "motif_confidence": round(float(region.motif_confidence), 6),
        "optional_region": bool(region.optional_region),
        "expected_note_count_min": int(region.expected_note_count_min),
        "expected_note_count_max": int(region.expected_note_count_max),
        "density_limit_notes_per_sec": round(region.density_limit_notes_per_sec, 6),
        "min_note_duration_sec": round(region.min_note_duration_sec, 6),
        "max_note_duration_sec": round(region.max_note_duration_sec, 6),
        "no_notes_outside_region": bool(region.no_notes_outside_region),
        "instruction": region.instruction,
    }


def _detect_allowed_completion_regions(
    *,
    base_notes: list[BasePatternNote],
    audio_activity_regions: list[dict[str, object]],
    pitch_contour_summary: list[dict[str, object]],
    duration_sec: float,
) -> list[AllowedCompletionRegion]:
    if len(base_notes) < 2:
        return []

    ordered_notes = sorted(
        base_notes,
        key=lambda item: (float(item.start_sec), float(item.end_sec), int(item.pitch_midi)),
    )

    note_durations = [float(item.duration_sec) for item in ordered_notes if float(item.duration_sec) > 0.0]
    onset_deltas = [
        float(ordered_notes[index + 1].start_sec) - float(ordered_notes[index].start_sec)
        for index in range(len(ordered_notes) - 1)
        if float(ordered_notes[index + 1].start_sec) > float(ordered_notes[index].start_sec)
    ]

    median_duration = statistics.median(note_durations) if note_durations else 0.25
    median_onset_delta = statistics.median(onset_deltas) if onset_deltas else median_duration
    long_gap_threshold = max(0.14, median_duration * 0.75, median_onset_delta * 0.9)
    max_region_duration = max(0.75, min(10.0, median_onset_delta * 12.0))

    candidate_regions: list[dict[str, float | int | str]] = []
    for index in range(len(ordered_notes) - 1):
        previous_note = ordered_notes[index]
        next_note = ordered_notes[index + 1]

        gap_start = float(previous_note.end_sec)
        gap_end = float(next_note.start_sec)
        gap_duration = gap_end - gap_start

        if gap_duration < long_gap_threshold:
            continue

        evidence = _activity_gap_evidence(
            gap_start=gap_start,
            gap_end=gap_end,
            audio_activity_regions=audio_activity_regions,
            pitch_contour_summary=pitch_contour_summary,
        )
        if not evidence["has_evidence"]:
            continue

        evidence_start = float(evidence["active_start"])
        evidence_end = float(evidence["active_end"])

        region_start = max(gap_start, evidence_start)
        region_end = min(gap_end, evidence_end)
        if region_end <= region_start + 0.04:
            region_start = gap_start
            region_end = gap_end

        region_end = min(region_end, region_start + max_region_duration)
        if region_end <= region_start + 0.08:
            continue

        candidate_regions.append(
            {
                "start_sec": region_start,
                "end_sec": region_end,
                "reason": str(evidence["reason"]),
                "onset_count": int(evidence["onset_count"]),
            }
        )

    # Consider trailing missing sections after the last base note where audio/pitch still indicate bass activity.
    last_note_end = float(ordered_notes[-1].end_sec)
    tail_gap_start = last_note_end
    tail_gap_end = float(duration_sec)
    tail_gap_duration = tail_gap_end - tail_gap_start
    if tail_gap_duration >= long_gap_threshold:
        tail_evidence = _activity_gap_evidence(
            gap_start=tail_gap_start,
            gap_end=tail_gap_end,
            audio_activity_regions=audio_activity_regions,
            pitch_contour_summary=pitch_contour_summary,
        )
        if tail_evidence["has_evidence"]:
            tail_region_start = max(tail_gap_start, float(tail_evidence["active_start"]))
            tail_region_end = min(tail_gap_end, float(tail_evidence["active_end"]))
            if tail_region_end <= tail_region_start + 0.04:
                tail_region_start = tail_gap_start
                tail_region_end = tail_gap_end

            tail_region_end = min(tail_region_end, tail_region_start + max_region_duration)
            if tail_region_end > tail_region_start + 0.08:
                candidate_regions.append(
                    {
                        "start_sec": tail_region_start,
                        "end_sec": tail_region_end,
                        "reason": str(tail_evidence["reason"]),
                        "onset_count": int(tail_evidence["onset_count"]),
                    }
                )

    merged_candidates = _merge_candidate_regions(candidate_regions)
    if not merged_candidates:
        return []

    global_pitch_min = min(int(item.pitch_midi) for item in ordered_notes)
    global_pitch_max = max(int(item.pitch_midi) for item in ordered_notes)

    detected_regions: list[AllowedCompletionRegion] = []
    for region_index, candidate in enumerate(merged_candidates, start=1):
        start_sec = max(0.0, float(candidate["start_sec"]))
        end_sec = min(float(duration_sec), float(candidate["end_sec"]))
        if end_sec <= start_sec + 0.08:
            continue

        context_before_start_sec = max(0.0, start_sec - 5.0)
        context_after_end_sec = min(float(duration_sec), end_sec + 5.0)

        before_notes = [
            note
            for note in ordered_notes
            if float(note.end_sec) <= start_sec and float(note.end_sec) >= context_before_start_sec
        ]
        after_notes = [
            note
            for note in ordered_notes
            if float(note.start_sec) >= end_sec and float(note.start_sec) <= context_after_end_sec
        ]

        before_notes = sorted(before_notes, key=lambda note: float(note.end_sec))[-24:]
        after_notes = sorted(after_notes, key=lambda note: float(note.start_sec))[:24]

        local_notes = before_notes[-32:] + after_notes[:32]
        if not local_notes:
            local_notes = _nearest_context_notes(
                ordered_notes=ordered_notes,
                region_start=start_sec,
                region_end=end_sec,
                limit=24,
            )

        local_pitches = [int(item.pitch_midi) for item in local_notes]
        local_pitch_set = sorted(set(local_pitches))
        if local_pitches:
            local_pitch_min = min(local_pitches)
            local_pitch_max = max(local_pitches)
        else:
            local_pitch_min = global_pitch_min
            local_pitch_max = global_pitch_max

        allowed_pitch_min = max(0, local_pitch_min)
        allowed_pitch_max = min(127, local_pitch_max)

        preferred_pitches = [
            int(pitch)
            for pitch, _count in Counter(local_pitches).most_common(8)
        ]

        forbidden_pitches: list[int] = []
        if allowed_pitch_min > 0:
            forbidden_pitches.append(allowed_pitch_min - 1)
        if allowed_pitch_max < 127:
            forbidden_pitches.append(allowed_pitch_max + 1)

        estimated_key_or_scale = _estimate_local_key_or_scale(local_pitches)

        rhythmic_summary = _build_local_rhythmic_summary(local_notes)
        local_rhythm_intervals_sec = _build_local_rhythm_intervals(
            before_notes=before_notes,
            after_notes=after_notes,
        )
        detected_local_motif = _detect_local_motif(
            before_notes=before_notes,
            after_notes=after_notes,
            local_rhythm_intervals_sec=local_rhythm_intervals_sec,
        )
        motif_confidence = float(detected_local_motif.get("confidence", 0.0))
        optional_region = (
            motif_confidence < 0.45
            or len(before_notes) < 2
            or len(after_notes) < 1
        )

        local_durations = [float(item.duration_sec) for item in local_notes if float(item.duration_sec) > 0.0]
        local_duration_median = statistics.median(local_durations) if local_durations else median_duration

        min_note_duration_sec = max(0.05, min(0.5, local_duration_median * 0.45))
        max_note_duration_sec = min(max(1.2, end_sec - start_sec), max(0.3, local_duration_median * 3.2))
        if max_note_duration_sec <= min_note_duration_sec:
            max_note_duration_sec = min_note_duration_sec + 0.1

        density_window_start = max(0.0, start_sec - 3.0)
        density_window_end = min(float(duration_sec), end_sec + 3.0)
        density_window_sec = max(0.25, density_window_end - density_window_start)
        local_density_notes = sum(
            1
            for note in ordered_notes
            if float(note.start_sec) < density_window_end and float(note.end_sec) > density_window_start
        )
        local_density = float(local_density_notes) / density_window_sec

        region_duration_sec = max(0.1, end_sec - start_sec)
        expected_center = local_density * region_duration_sec
        onset_hint = int(candidate.get("onset_count", 0))
        if onset_hint > 0:
            expected_center = max(expected_center, min(8.0, float(onset_hint)))

        expected_note_count_min = int(max(0, math.floor(expected_center * 0.4)))
        if optional_region:
            expected_note_count_min = 0
        elif expected_center >= 1.0 and expected_note_count_min == 0:
            expected_note_count_min = 1
        expected_note_count_max = int(
            max(expected_note_count_min, min(24, math.ceil(expected_center * 1.8) + 1))
        )

        expected_density_ceiling = float(expected_note_count_max) / region_duration_sec
        median_rhythm_interval = (
            statistics.median(local_rhythm_intervals_sec)
            if local_rhythm_intervals_sec
            else 0.0
        )
        rhythm_density_limit = (
            (1.0 / max(0.05, median_rhythm_interval)) * 1.2
            if median_rhythm_interval > 0.0
            else 0.0
        )
        density_limit_notes_per_sec = max(
            0.35,
            min(
                8.0,
                max(
                    (local_density * 1.7) if local_density > 0.0 else 0.0,
                    rhythm_density_limit,
                    expected_density_ceiling,
                ),
            ),
        )

        notes_before_payload = [_context_note_record(item) for item in before_notes]
        notes_after_payload = [_context_note_record(item) for item in after_notes]
        local_pitch_names = [_midi_pitch_name(pitch) for pitch in local_pitch_set]

        instruction = (
            "Fill only this gap by continuing the local motif. "
            "Do not write outside write_start_sec/write_end_sec."
        )
        if optional_region:
            instruction += " Motif context is low confidence; prefer zero notes unless continuation is clear."

        detected_regions.append(
            AllowedCompletionRegion(
                region_id=f"acr_{region_index:04d}",
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                write_start_sec=float(start_sec),
                write_end_sec=float(end_sec),
                reason=str(candidate["reason"]),
                context_before_start_sec=float(context_before_start_sec),
                context_after_end_sec=float(context_after_end_sec),
                context_window_before_sec=float(start_sec - context_before_start_sec),
                context_window_after_sec=float(context_after_end_sec - end_sec),
                reference_notes_before=[item.note_id for item in before_notes],
                reference_notes_after=[item.note_id for item in after_notes],
                notes_before=notes_before_payload,
                notes_after=notes_after_payload,
                local_pitch_set=[int(value) for value in local_pitch_set],
                local_pitch_names=local_pitch_names,
                local_pitch_range={"min": int(local_pitch_min), "max": int(local_pitch_max)},
                allowed_pitch_range={"min": int(allowed_pitch_min), "max": int(allowed_pitch_max)},
                preferred_pitches=preferred_pitches,
                forbidden_pitches=forbidden_pitches,
                allow_pitch_outside_local_set=False,
                estimated_key_or_scale=estimated_key_or_scale,
                rhythmic_pattern_summary=rhythmic_summary,
                local_rhythm_intervals_sec=local_rhythm_intervals_sec,
                detected_local_motif=detected_local_motif,
                motif_confidence=float(motif_confidence),
                optional_region=optional_region,
                expected_note_count_min=expected_note_count_min,
                expected_note_count_max=expected_note_count_max,
                density_limit_notes_per_sec=float(density_limit_notes_per_sec),
                min_note_duration_sec=float(min_note_duration_sec),
                max_note_duration_sec=float(max_note_duration_sec),
                no_notes_outside_region=True,
                instruction=instruction,
            )
        )

    return detected_regions


def _activity_gap_evidence(
    *,
    gap_start: float,
    gap_end: float,
    audio_activity_regions: list[dict[str, object]],
    pitch_contour_summary: list[dict[str, object]],
) -> dict[str, object]:
    gap_duration = max(1e-6, gap_end - gap_start)
    overlapping_regions = [
        region
        for region in audio_activity_regions
        if float(region.get("start_sec", 0.0)) < gap_end
        and float(region.get("end_sec", 0.0)) > gap_start
    ]

    overlap_duration = 0.0
    onset_count = 0
    active_start = gap_start
    active_end = gap_end
    if overlapping_regions:
        start_values = []
        end_values = []
        for region in overlapping_regions:
            region_start = max(gap_start, float(region.get("start_sec", gap_start)))
            region_end = min(gap_end, float(region.get("end_sec", gap_end)))
            if region_end <= region_start:
                continue
            overlap_duration += region_end - region_start
            onset_count += int(region.get("onset_count", 0))
            start_values.append(region_start)
            end_values.append(region_end)
        if start_values and end_values:
            active_start = min(start_values)
            active_end = max(end_values)

    pitch_hits = sum(
        1
        for section in pitch_contour_summary
        if float(section.get("start_sec", 0.0)) < gap_end
        and float(section.get("end_sec", 0.0)) > gap_start
        and section.get("dominant_pitch_midi") is not None
        and float(section.get("voiced_ratio", 0.0)) >= 0.2
    )

    active_ratio = overlap_duration / gap_duration
    has_evidence = active_ratio >= 0.18 or onset_count >= 1 or pitch_hits >= 2

    reason = "long_gap_with_activity_evidence"
    if active_ratio >= 0.4 and onset_count >= 1:
        reason = "long_gap_with_strong_audio_activity"
    elif onset_count >= 1:
        reason = "long_gap_with_onset_evidence"
    elif pitch_hits >= 2:
        reason = "long_gap_with_pitch_contour_evidence"

    return {
        "has_evidence": has_evidence,
        "active_start": active_start,
        "active_end": active_end,
        "onset_count": onset_count,
        "reason": reason,
    }


def _merge_candidate_regions(candidates: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda item: float(item["start_sec"]))
    merged: list[dict[str, float | int | str]] = []

    for candidate in sorted_candidates:
        if not merged:
            merged.append(dict(candidate))
            continue

        previous = merged[-1]
        previous_end = float(previous["end_sec"])
        current_start = float(candidate["start_sec"])
        if current_start - previous_end > 0.1:
            merged.append(dict(candidate))
            continue

        previous["end_sec"] = max(float(previous["end_sec"]), float(candidate["end_sec"]))
        previous["onset_count"] = int(previous.get("onset_count", 0)) + int(candidate.get("onset_count", 0))

        previous_reason = str(previous.get("reason", ""))
        current_reason = str(candidate.get("reason", ""))
        if "strong" in current_reason and "strong" not in previous_reason:
            previous["reason"] = current_reason

    return merged


def _nearest_context_notes(
    *,
    ordered_notes: list[BasePatternNote],
    region_start: float,
    region_end: float,
    limit: int,
) -> list[BasePatternNote]:
    ranked = sorted(
        ordered_notes,
        key=lambda note: min(
            abs(float(note.start_sec) - region_start),
            abs(float(note.end_sec) - region_end),
        ),
    )
    return ranked[: max(1, limit)]


def _context_note_record(note: BasePatternNote) -> dict[str, object]:
    pitch_midi = int(note.pitch_midi)
    return {
        "note_id": note.note_id,
        "start_sec": round(float(note.start_sec), 6),
        "end_sec": round(float(note.end_sec), 6),
        "pitch_midi": pitch_midi,
        "pitch_name": _midi_pitch_name(pitch_midi),
    }


def _build_local_rhythm_intervals(
    *,
    before_notes: list[BasePatternNote],
    after_notes: list[BasePatternNote],
) -> list[float]:
    ordered = sorted(
        before_notes + after_notes,
        key=lambda note: (float(note.start_sec), float(note.end_sec)),
    )
    onsets = [float(note.start_sec) for note in ordered]
    intervals: list[float] = []
    for index in range(1, len(onsets)):
        delta = onsets[index] - onsets[index - 1]
        if delta > 0.0:
            intervals.append(round(delta, 6))
    return intervals[:24]


def _detect_local_motif(
    *,
    before_notes: list[BasePatternNote],
    after_notes: list[BasePatternNote],
    local_rhythm_intervals_sec: list[float],
) -> dict[str, object]:
    ordered_before = sorted(before_notes, key=lambda note: float(note.start_sec))
    ordered_after = sorted(after_notes, key=lambda note: float(note.start_sec))

    motif_notes: list[BasePatternNote]
    if ordered_before and ordered_after:
        motif_notes = (ordered_before[-4:] + ordered_after[:4])[:8]
    elif ordered_before:
        motif_notes = ordered_before[-6:]
    else:
        motif_notes = ordered_after[:6]

    pitch_sequence = [int(note.pitch_midi) for note in motif_notes]
    interval_sequence = [
        int(pitch_sequence[index + 1] - pitch_sequence[index])
        for index in range(len(pitch_sequence) - 1)
    ]

    motif_rhythm_sequence: list[float] = []
    motif_onsets = [float(note.start_sec) for note in motif_notes]
    for index in range(1, len(motif_onsets)):
        delta = motif_onsets[index] - motif_onsets[index - 1]
        if delta > 0.0:
            motif_rhythm_sequence.append(round(delta, 6))
    if not motif_rhythm_sequence:
        motif_rhythm_sequence = [float(value) for value in local_rhythm_intervals_sec[:8]]

    confidence = 0.0
    if len(ordered_before) >= 4:
        confidence += 0.35
    elif len(ordered_before) >= 2:
        confidence += 0.2

    if len(ordered_after) >= 3:
        confidence += 0.35
    elif len(ordered_after) >= 1:
        confidence += 0.2

    if interval_sequence:
        dominant_interval_count = Counter(interval_sequence).most_common(1)[0][1]
        confidence += 0.2 if dominant_interval_count >= 2 else 0.1

    if len(motif_rhythm_sequence) >= 2:
        if statistics.pstdev(motif_rhythm_sequence) <= 0.25:
            confidence += 0.1
    elif motif_rhythm_sequence:
        confidence += 0.05

    confidence = max(0.0, min(1.0, round(confidence, 3)))

    return {
        "pitch_sequence": pitch_sequence,
        "interval_sequence": interval_sequence,
        "rhythm_sequence_sec": motif_rhythm_sequence,
        "confidence": confidence,
    }


def _midi_pitch_name(pitch_midi: int) -> str:
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pitch = max(0, min(127, int(pitch_midi)))
    octave = (pitch // 12) - 1
    return f"{pitch_names[pitch % 12]}{octave}"


def _build_local_rhythmic_summary(local_notes: list[BasePatternNote]) -> dict[str, object]:
    ordered = sorted(local_notes, key=lambda note: (float(note.start_sec), float(note.end_sec)))
    onsets = [round(float(note.start_sec), 6) for note in ordered[:24]]
    intervals: list[float] = []
    for index in range(1, len(onsets)):
        intervals.append(round(onsets[index] - onsets[index - 1], 6))

    durations = [round(float(note.duration_sec), 6) for note in ordered]
    common_durations = [
        float(value)
        for value, _count in Counter(durations).most_common(8)
    ]

    return {
        "note_onsets_sec": onsets,
        "intervals_sec": intervals[:24],
        "common_durations_sec": common_durations,
    }


def _estimate_local_key_or_scale(local_pitches: list[int]) -> str:
    if len(local_pitches) < 3:
        return "unknown"

    pitch_classes = [int(value) % 12 for value in local_pitches]
    root_pc = Counter(pitch_classes).most_common(1)[0][0]

    major_scale = {(root_pc + step) % 12 for step in (0, 2, 4, 5, 7, 9, 11)}
    minor_scale = {(root_pc + step) % 12 for step in (0, 2, 3, 5, 7, 8, 10)}

    unique_classes = set(pitch_classes)
    major_score = len(unique_classes.intersection(major_scale))
    minor_score = len(unique_classes.intersection(minor_scale))

    root_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][root_pc]
    if minor_score > major_score:
        return f"{root_name} minor (estimated)"
    if major_score > minor_score:
        return f"{root_name} major (estimated)"
    return f"{root_name} modal (estimated)"
