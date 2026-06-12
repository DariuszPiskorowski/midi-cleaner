from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median

from midi_cleaner.alignment.models import (
    AudioAlignedNoteDocument,
    AudioAlignedNoteEvent,
    AudioAlignmentReport,
)
from midi_cleaner.audio.models import AudioFeatureDocument, AudioFrameFeature
from midi_cleaner.midi.models import NoteEventDocument

SCHEMA_VERSION = "0.1.0"


class AudioTimeAlignmentError(Exception):
    """Raised when audio-time alignment cannot be completed."""


@dataclass(frozen=True)
class AudioTimeAlignmentParameters:
    onset_search_window_ms: float = 250.0
    offset_search_window_ms: float = 350.0
    min_onset_score: float = 0.005
    min_rms: float = 0.001
    snap_start_to_audio_onset: bool = True
    snap_end_to_energy_offset: bool = True
    max_start_correction_ms: float = 500.0
    max_end_correction_ms: float = 800.0
    low_confidence_action: str = "KEEP_ORIGINAL_LOW_CONFIDENCE"


def _load_note_document(path: Path) -> NoteEventDocument:
    try:
        return NoteEventDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise AudioTimeAlignmentError(f"Invalid notes JSON: {path}") from exc


def _load_audio_document(path: Path) -> AudioFeatureDocument:
    try:
        return AudioFeatureDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise AudioTimeAlignmentError(f"Invalid audio features JSON: {path}") from exc


def _window_frames(
    frames: list[AudioFrameFeature],
    anchor_sec: float,
    window_sec: float,
) -> list[AudioFrameFeature]:
    low = anchor_sec - window_sec
    high = anchor_sec + window_sec
    return [frame for frame in frames if low <= frame.start_sec <= high]


def _nearest_frame(frames: list[AudioFrameFeature], anchor_sec: float) -> AudioFrameFeature | None:
    if not frames:
        return None
    return min(frames, key=lambda frame: abs(frame.start_sec - anchor_sec))


def _build_prev_rms(frames: list[AudioFrameFeature]) -> dict[int, float]:
    previous = 0.0
    mapping: dict[int, float] = {}
    for frame in frames:
        mapping[frame.frame_index] = previous
        previous = frame.rms
    return mapping


def _best_onset_candidate(
    onset_candidates: list[AudioFrameFeature],
    prev_rms_by_index: dict[int, float],
    note_start_sec: float,
    params: AudioTimeAlignmentParameters,
) -> tuple[AudioFrameFeature | None, AudioFrameFeature | None]:
    if not onset_candidates:
        return None, None

    weak_best = max(
        onset_candidates,
        key=lambda frame: (frame.onset_score, -abs(frame.start_sec - note_start_sec)),
    )

    strong_candidates = [
        frame
        for frame in onset_candidates
        if frame.onset_score >= params.min_onset_score
        and frame.rms >= params.min_rms
        and (frame.rms - prev_rms_by_index.get(frame.frame_index, 0.0)) > 0.0
    ]

    if not strong_candidates:
        return None, weak_best

    strong_best = max(
        strong_candidates,
        key=lambda frame: (
            frame.onset_score,
            frame.rms - prev_rms_by_index.get(frame.frame_index, 0.0),
            -abs(frame.start_sec - note_start_sec),
        ),
    )
    return strong_best, weak_best


def _best_offset_candidate(
    offset_candidates: list[AudioFrameFeature],
    next_rms_by_index: dict[int, float],
    aligned_start_sec: float,
    original_end_sec: float,
    local_rms: float,
    params: AudioTimeAlignmentParameters,
) -> AudioFrameFeature | None:
    if not offset_candidates:
        return None

    threshold = max(params.min_rms, local_rms * 0.35)

    valid_candidates = [frame for frame in offset_candidates if frame.start_sec >= aligned_start_sec]
    if not valid_candidates:
        return None

    scored = []
    for frame in valid_candidates:
        next_rms = next_rms_by_index.get(frame.frame_index, frame.rms)
        drop = max(0.0, frame.rms - next_rms)
        low_energy_flag = 1 if frame.rms <= threshold else 0
        scored.append((
            low_energy_flag,
            drop,
            -abs(frame.start_sec - original_end_sec),
            -frame.start_sec,
            frame,
        ))

    scored.sort(reverse=True)
    best = scored[0][4]
    next_rms = next_rms_by_index.get(best.frame_index, best.rms)
    has_energy_drop_evidence = best.rms <= threshold or (best.rms - next_rms) > 0.0

    if has_energy_drop_evidence:
        return best
    return None


def _build_next_rms(frames: list[AudioFrameFeature]) -> dict[int, float]:
    mapping: dict[int, float] = {}
    for index, frame in enumerate(frames):
        if index + 1 < len(frames):
            mapping[frame.frame_index] = frames[index + 1].rms
        else:
            mapping[frame.frame_index] = frame.rms
    return mapping


def _interval_frames(
    frames: list[AudioFrameFeature],
    start_sec: float,
    end_sec: float,
) -> list[AudioFrameFeature]:
    return [
        frame
        for frame in frames
        if frame.end_sec > start_sec and frame.start_sec < end_sec
    ]


def _safe_interval(
    start_sec: float,
    end_sec: float,
    audio_duration_sec: float,
    sample_rate: int,
) -> tuple[float, float]:
    minimum_duration = 1.0 / sample_rate if sample_rate > 0 else 1e-4
    minimum_duration = max(minimum_duration, 1e-4)

    max_start = max(0.0, audio_duration_sec - minimum_duration)
    clamped_start = min(max(0.0, start_sec), max_start)

    clamped_end = max(end_sec, clamped_start + minimum_duration)
    if audio_duration_sec > 0.0:
        clamped_end = min(clamped_end, audio_duration_sec)
        if clamped_end <= clamped_start:
            clamped_end = min(audio_duration_sec, clamped_start + minimum_duration)
            if clamped_end <= clamped_start:
                clamped_end = clamped_start + minimum_duration

    return clamped_start, clamped_end


def _onset_side_errors(reference_sec: float, detected_sec: float | None) -> tuple[float | None, float | None]:
    if detected_sec is None:
        return None, None

    before = max(0.0, (reference_sec - detected_sec) * 1000.0)
    after = max(0.0, (detected_sec - reference_sec) * 1000.0)
    return before, after


def _percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * percentile_value
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]

    weight = rank - lower
    return (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)


def _alignment_confidence(
    action: str,
    onset_score: float,
    min_onset_score: float,
    local_rms: float,
    min_rms: float,
    sustained_energy_ratio: float,
) -> float:
    onset_norm = 1.0 if min_onset_score <= 0 else min(1.0, onset_score / min_onset_score)
    rms_norm = 1.0 if min_rms <= 0 else min(1.0, local_rms / min_rms)
    sustain_norm = max(0.0, min(1.0, sustained_energy_ratio))

    if action == "ALIGNED":
        confidence = (0.55 * onset_norm) + (0.30 * rms_norm) + (0.15 * sustain_norm)
    elif action == "REVIEW_TIMING":
        confidence = (0.30 * onset_norm) + (0.30 * rms_norm) + (0.20 * sustain_norm)
    elif action == "KEEP_ORIGINAL_LOW_CONFIDENCE":
        confidence = (0.20 * onset_norm) + (0.20 * rms_norm) + (0.20 * sustain_norm)
    else:
        confidence = 0.0

    return max(0.0, min(1.0, confidence))


def align_notes_to_audio_time(
    notes_file: Path,
    audio_features_file: Path,
    params: AudioTimeAlignmentParameters,
) -> tuple[AudioAlignedNoteDocument, AudioAlignmentReport]:
    if not notes_file.exists() or not notes_file.is_file():
        raise AudioTimeAlignmentError(f"Notes file does not exist: {notes_file}")
    if not audio_features_file.exists() or not audio_features_file.is_file():
        raise AudioTimeAlignmentError(f"Audio features file does not exist: {audio_features_file}")

    note_document = _load_note_document(notes_file)
    audio_document = _load_audio_document(audio_features_file)

    warnings: list[str] = []
    if note_document.layer != audio_document.layer:
        warnings.append(
            "Layer mismatch between notes and audio features: "
            f"{note_document.layer} vs {audio_document.layer}."
        )

    frames = audio_document.frames
    prev_rms_by_index = _build_prev_rms(frames)
    next_rms_by_index = _build_next_rms(frames)

    onset_window_sec = params.onset_search_window_ms / 1000.0
    offset_window_sec = params.offset_search_window_ms / 1000.0

    aligned_notes: list[AudioAlignedNoteEvent] = []

    for note in note_document.notes:
        reasons: list[str] = []

        original_start_sec = float(note.start_sec)
        original_end_sec = float(note.end_sec)
        original_duration_sec = max(0.0, float(note.duration_sec))

        onset_candidates = _window_frames(frames, original_start_sec, onset_window_sec)
        strong_onset, weak_onset = _best_onset_candidate(
            onset_candidates=onset_candidates,
            prev_rms_by_index=prev_rms_by_index,
            note_start_sec=original_start_sec,
            params=params,
        )

        local_frame = _nearest_frame(onset_candidates, original_start_sec)
        if local_frame is None:
            local_frame = _nearest_frame(frames, original_start_sec)

        local_rms = float(local_frame.rms) if local_frame is not None else 0.0
        local_onset_score = float(local_frame.onset_score) if local_frame is not None else 0.0

        nearest_audio_onset_sec = None
        if strong_onset is not None:
            nearest_audio_onset_sec = float(strong_onset.start_sec)
        elif weak_onset is not None and weak_onset.onset_score > 0.0:
            nearest_audio_onset_sec = float(weak_onset.start_sec)

        aligned_start_sec = original_start_sec
        alignment_action = params.low_confidence_action

        if strong_onset is not None:
            start_shift_ms = (strong_onset.start_sec - original_start_sec) * 1000.0
            if not params.snap_start_to_audio_onset:
                alignment_action = params.low_confidence_action
                reasons.append("snap_start_to_audio_onset disabled; kept original start")
            elif abs(start_shift_ms) <= params.max_start_correction_ms:
                aligned_start_sec = float(strong_onset.start_sec)
                alignment_action = "ALIGNED"
                reasons.append("snapped start to strongest local audio onset")
            else:
                alignment_action = "REVIEW_TIMING"
                reasons.append("start correction exceeds max_start_correction_ms; kept original start")
        else:
            if local_rms < params.min_rms and local_onset_score < params.min_onset_score:
                alignment_action = "NO_AUDIO_EVIDENCE"
                reasons.append("no audio onset or RMS evidence near note start")
            else:
                alignment_action = params.low_confidence_action
                reasons.append("no strong onset in search window; kept original start")

        fallback_end_sec = aligned_start_sec + original_duration_sec
        offset_candidates = _window_frames(frames, original_end_sec, offset_window_sec)
        offset_frame = _best_offset_candidate(
            offset_candidates=offset_candidates,
            next_rms_by_index=next_rms_by_index,
            aligned_start_sec=aligned_start_sec,
            original_end_sec=original_end_sec,
            local_rms=local_rms,
            params=params,
        )

        nearest_audio_offset_sec = None
        aligned_end_sec = fallback_end_sec

        if params.snap_end_to_energy_offset and offset_frame is not None:
            end_shift_ms = (offset_frame.start_sec - original_end_sec) * 1000.0
            if abs(end_shift_ms) <= params.max_end_correction_ms:
                aligned_end_sec = float(offset_frame.start_sec)
                nearest_audio_offset_sec = float(offset_frame.start_sec)
                reasons.append("snapped end to local audio energy drop")
                if alignment_action == params.low_confidence_action:
                    alignment_action = "ALIGNED"
            else:
                reasons.append("end correction exceeds max_end_correction_ms; preserved duration")
        else:
            reasons.append("preserved original duration for end timing")

        aligned_start_sec, aligned_end_sec = _safe_interval(
            start_sec=aligned_start_sec,
            end_sec=aligned_end_sec,
            audio_duration_sec=float(audio_document.duration_sec),
            sample_rate=int(audio_document.sample_rate),
        )

        aligned_duration_sec = max(0.0, aligned_end_sec - aligned_start_sec)
        overlapping = _interval_frames(frames, aligned_start_sec, aligned_end_sec)
        if overlapping:
            sustained_energy_ratio = float(
                sum(1 for frame in overlapping if frame.rms >= params.min_rms) / len(overlapping)
            )
        else:
            sustained_energy_ratio = 0.0

        start_correction_ms = (aligned_start_sec - original_start_sec) * 1000.0
        end_correction_ms = (aligned_end_sec - original_end_sec) * 1000.0
        duration_correction_ms = (aligned_duration_sec - original_duration_sec) * 1000.0

        onset_error_before_ms, onset_error_after_ms = _onset_side_errors(
            reference_sec=original_start_sec,
            detected_sec=nearest_audio_onset_sec,
        )

        alignment_confidence = _alignment_confidence(
            action=alignment_action,
            onset_score=local_onset_score,
            min_onset_score=params.min_onset_score,
            local_rms=local_rms,
            min_rms=params.min_rms,
            sustained_energy_ratio=sustained_energy_ratio,
        )

        aligned_notes.append(
            AudioAlignedNoteEvent(
                note_id=note.note_id,
                source=note.source,
                layer=note.layer,
                pitch_midi=note.pitch_midi,
                pitch_name=note.pitch_name,
                velocity=note.velocity,
                channel=note.channel,
                original_start_sec=original_start_sec,
                original_end_sec=original_end_sec,
                original_duration_sec=original_duration_sec,
                original_start_tick=note.start_tick,
                original_end_tick=note.end_tick,
                aligned_start_sec=aligned_start_sec,
                aligned_end_sec=aligned_end_sec,
                aligned_duration_sec=aligned_duration_sec,
                start_correction_ms=start_correction_ms,
                end_correction_ms=end_correction_ms,
                duration_correction_ms=duration_correction_ms,
                nearest_audio_onset_sec=nearest_audio_onset_sec,
                nearest_audio_offset_sec=nearest_audio_offset_sec,
                onset_error_before_ms=onset_error_before_ms,
                onset_error_after_ms=onset_error_after_ms,
                local_rms=local_rms,
                local_onset_score=local_onset_score,
                sustained_energy_ratio=sustained_energy_ratio,
                alignment_confidence=alignment_confidence,
                alignment_action=alignment_action,
                reasons=reasons,
            )
        )

    aligned_count = sum(1 for note in aligned_notes if note.alignment_action == "ALIGNED")
    keep_original_count = sum(
        1 for note in aligned_notes if note.alignment_action == "KEEP_ORIGINAL_LOW_CONFIDENCE"
    )
    review_timing_count = sum(1 for note in aligned_notes if note.alignment_action == "REVIEW_TIMING")
    no_audio_evidence_count = sum(1 for note in aligned_notes if note.alignment_action == "NO_AUDIO_EVIDENCE")

    aligned_start_shifts = [
        abs(note.start_correction_ms)
        for note in aligned_notes
        if note.alignment_action == "ALIGNED"
    ]

    if aligned_start_shifts:
        median_abs_start_correction_ms = float(median(aligned_start_shifts))
        p95_abs_start_correction_ms = float(_percentile(aligned_start_shifts, 0.95))
        max_abs_start_correction_ms = float(max(aligned_start_shifts))
    else:
        median_abs_start_correction_ms = None
        p95_abs_start_correction_ms = None
        max_abs_start_correction_ms = None

    if no_audio_evidence_count > 0:
        warnings.append(f"No audio evidence for {no_audio_evidence_count} notes.")

    document = AudioAlignedNoteDocument(
        schema_version=SCHEMA_VERSION,
        notes_file=str(notes_file),
        audio_features_file=str(audio_features_file),
        layer=note_document.layer,
        sample_rate=audio_document.sample_rate,
        audio_duration_sec=audio_document.duration_sec,
        alignment_parameters={
            "onset_search_window_ms": params.onset_search_window_ms,
            "offset_search_window_ms": params.offset_search_window_ms,
            "min_onset_score": params.min_onset_score,
            "min_rms": params.min_rms,
            "snap_start_to_audio_onset": params.snap_start_to_audio_onset,
            "snap_end_to_energy_offset": params.snap_end_to_energy_offset,
            "max_start_correction_ms": params.max_start_correction_ms,
            "max_end_correction_ms": params.max_end_correction_ms,
            "low_confidence_action": params.low_confidence_action,
        },
        notes=aligned_notes,
    )

    report = AudioAlignmentReport(
        notes_file=str(notes_file),
        audio_features_file=str(audio_features_file),
        status="ok",
        layer=note_document.layer,
        note_count=len(aligned_notes),
        aligned_count=aligned_count,
        keep_original_count=keep_original_count,
        review_timing_count=review_timing_count,
        no_audio_evidence_count=no_audio_evidence_count,
        median_abs_start_correction_ms=median_abs_start_correction_ms,
        p95_abs_start_correction_ms=p95_abs_start_correction_ms,
        max_abs_start_correction_ms=max_abs_start_correction_ms,
        warning_count=len(warnings),
        warnings=warnings,
        output_file=None,
    )

    return document, report
