from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from midi_cleaner.audio.models import AudioFeatureDocument, AudioFrameFeature
from midi_cleaner.cleanup.models import CleanupPlanDocument
from midi_cleaner.dsp.models import DspAudioFeatureDocument, DspAudioFrame
from midi_cleaner.pitch.models import BassPitchContourDocument, BassPitchFrame
from midi_cleaner.refinement.models import RefinedNoteDocument, RefinedNoteEvent
from midi_cleaner.repair.models import ActivityRepairPlan, ActivityRepairReport, RepairAction

SCHEMA_VERSION = "0.1.0"


class ActivityRepairError(Exception):
    """Raised when activity repair cannot be completed."""


@dataclass(frozen=True)
class ActivityRepairParameters:
    audio_active_threshold_ratio: float = 0.18
    audio_silence_hold_ms: float = 120.0
    min_audio_region_ms: float = 50.0
    merge_audio_region_gap_ms: float = 60.0
    low_band_weight: float = 0.55
    harmonic_weight: float = 0.25
    rms_weight: float = 0.20

    missing_gap_min_ms: float = 80.0
    max_extend_for_gap_ms: float = 500.0
    max_insert_missing_ms: float = 700.0
    context_pitch_search_ms: float = 800.0

    overhang_min_ms: float = 220.0
    tail_padding_ms: float = 90.0
    minimum_repaired_note_duration_ms: float = 100.0

    sustain_protect_ratio: float = 0.16
    sustain_protect_hold_ms: float = 180.0
    pitch_sustain_hold_ms: float = 160.0
    legato_neighbor_window_ms: float = 220.0
    legato_min_silence_ms: float = 100.0

    split_min_note_duration_ms: float = 500.0
    split_min_distance_from_edges_ms: float = 120.0
    split_onset_strength_ratio: float = 0.55
    split_valley_ratio: float = 0.35

    close_gap_ms: float = 50.0
    near_pitch_tolerance_semitones: int = 1

    insert_auto_confidence: float = 0.80
    split_auto_confidence: float = 0.75
    split_pitch_change_semitones: float = 0.75
    insert_from_pitch_contour_confidence: float = 0.75


@dataclass(frozen=True)
class _ActivityFrame:
    start_sec: float
    end_sec: float
    energy: float
    onset: float
    is_active_hint: bool


@dataclass(frozen=True)
class _ActivityRegion:
    start_sec: float
    end_sec: float
    peak_energy: float
    mean_energy: float
    onset_count: int
    confidence: float


@dataclass(frozen=True)
class _MidiRegion:
    start_sec: float
    end_sec: float
    note_ids: list[str]


@dataclass(frozen=True)
class _PitchWindowSummary:
    voiced_ratio: float
    mean_confidence: float
    dominant_pitch_midi: int | None


@dataclass
class _MutableNote:
    note_id: str
    source: str
    layer: str
    pitch_midi: int
    pitch_name: str
    velocity: int
    channel: int | None
    original_start_sec: float
    original_end_sec: float
    aligned_start_sec: float
    aligned_end_sec: float
    refined_start_sec: float
    refined_end_sec: float
    merged_note_ids: list[str]
    refinement_actions: list[str]
    refinement_confidence: float
    reasons: list[str]

    def duration_sec(self) -> float:
        return max(0.0, self.refined_end_sec - self.refined_start_sec)


def _load_audio_document(path: Path) -> AudioFeatureDocument:
    try:
        return AudioFeatureDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise ActivityRepairError(f"Invalid audio features JSON: {path}") from exc


def _load_dsp_document(path: Path) -> DspAudioFeatureDocument:
    try:
        return DspAudioFeatureDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise ActivityRepairError(f"Invalid DSP features JSON: {path}") from exc


def _load_pitch_document(path: Path) -> BassPitchContourDocument:
    try:
        return BassPitchContourDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise ActivityRepairError(f"Invalid bass pitch contour JSON: {path}") from exc


def _load_refined_document(path: Path) -> RefinedNoteDocument:
    try:
        return RefinedNoteDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise ActivityRepairError(f"Invalid refined notes JSON: {path}") from exc


def _load_cleanup_plan(path: Path) -> CleanupPlanDocument:
    try:
        return CleanupPlanDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise ActivityRepairError(f"Invalid cleanup plan JSON: {path}") from exc


def _frames_from_dsp(
    dsp_document: DspAudioFeatureDocument,
    audio_document: AudioFeatureDocument,
    params: ActivityRepairParameters,
) -> list[_ActivityFrame]:
    frames: list[_ActivityFrame] = []
    low_w = max(0.0, params.low_band_weight)
    harm_w = max(0.0, params.harmonic_weight)
    rms_w = max(0.0, params.rms_weight)
    total_w = max(1e-6, low_w + harm_w + rms_w)

    fallback_rms = audio_document.global_features.rms
    for frame in dsp_document.frames:
        energy = (
            (low_w * float(frame.low_band_envelope_smooth))
            + (harm_w * float(frame.harmonic_rms))
            + (rms_w * float(frame.rms_smooth))
        ) / total_w
        onset = max(float(frame.onset_strength), float(frame.spectral_flux))
        is_active_hint = (not frame.is_silence) and (
            frame.is_attack_rise or frame.is_sustain or frame.is_tail or energy > fallback_rms * 0.25
        )
        frames.append(
            _ActivityFrame(
                start_sec=float(frame.start_sec),
                end_sec=float(frame.end_sec),
                energy=max(0.0, float(energy)),
                onset=max(0.0, float(onset)),
                is_active_hint=is_active_hint,
            )
        )
    return frames


def _frames_from_audio(audio_document: AudioFeatureDocument) -> list[_ActivityFrame]:
    frames: list[_ActivityFrame] = []
    for frame in audio_document.frames:
        frames.append(
            _ActivityFrame(
                start_sec=float(frame.start_sec),
                end_sec=float(frame.end_sec),
                energy=max(0.0, float(frame.rms)),
                onset=max(0.0, float(frame.onset_score)),
                is_active_hint=not bool(frame.is_silent),
            )
        )
    return frames


def _build_audio_activity_regions(
    frames: list[_ActivityFrame],
    params: ActivityRepairParameters,
) -> list[_ActivityRegion]:
    if not frames:
        return []

    energies = [frame.energy for frame in frames]
    peak_energy = max(energies) if energies else 0.0
    mean_energy = sum(energies) / max(1, len(energies))
    base_threshold = max(1e-6, peak_energy * params.audio_active_threshold_ratio)
    threshold = max(base_threshold, mean_energy * 0.6)

    frame_step = max(1e-6, frames[0].end_sec - frames[0].start_sec)
    hold_sec = max(0.0, params.audio_silence_hold_ms / 1000.0)
    min_region_sec = max(0.0, params.min_audio_region_ms / 1000.0)
    merge_gap_sec = max(0.0, params.merge_audio_region_gap_ms / 1000.0)

    regions: list[_ActivityRegion] = []
    current_start: float | None = None
    current_end = 0.0
    current_peak = 0.0
    current_sum = 0.0
    current_count = 0
    current_onsets = 0
    below_run_start: float | None = None

    for frame in frames:
        active_now = frame.energy >= threshold or frame.is_active_hint
        if current_start is None:
            if active_now:
                current_start = frame.start_sec
                current_end = frame.end_sec
                current_peak = frame.energy
                current_sum = frame.energy
                current_count = 1
                current_onsets = 1 if frame.onset > 0.0 else 0
                below_run_start = None
            continue

        if active_now:
            current_end = frame.end_sec
            current_peak = max(current_peak, frame.energy)
            current_sum += frame.energy
            current_count += 1
            if frame.onset > 0.0:
                current_onsets += 1
            below_run_start = None
            continue

        if below_run_start is None:
            below_run_start = frame.start_sec

        silence_len = max(0.0, frame.end_sec - below_run_start)
        if silence_len < hold_sec:
            current_end = frame.end_sec
            current_sum += frame.energy
            current_count += 1
            if frame.onset > 0.0:
                current_onsets += 1
            continue

        duration = max(0.0, current_end - current_start)
        mean_e = current_sum / max(1, current_count)
        confidence = min(1.0, max(0.0, (mean_e / max(threshold, 1e-6)) * 0.6 + (0.1 * current_onsets)))
        if duration >= min_region_sec or current_onsets > 0:
            regions.append(
                _ActivityRegion(
                    start_sec=current_start,
                    end_sec=current_end,
                    peak_energy=current_peak,
                    mean_energy=mean_e,
                    onset_count=current_onsets,
                    confidence=confidence,
                )
            )

        current_start = None
        current_end = 0.0
        current_peak = 0.0
        current_sum = 0.0
        current_count = 0
        current_onsets = 0
        below_run_start = None

    if current_start is not None:
        duration = max(0.0, current_end - current_start)
        mean_e = current_sum / max(1, current_count)
        confidence = min(1.0, max(0.0, (mean_e / max(threshold, 1e-6)) * 0.6 + (0.1 * current_onsets)))
        if duration >= min_region_sec or current_onsets > 0:
            regions.append(
                _ActivityRegion(
                    start_sec=current_start,
                    end_sec=current_end,
                    peak_energy=current_peak,
                    mean_energy=mean_e,
                    onset_count=current_onsets,
                    confidence=confidence,
                )
            )

    if not regions:
        return []

    merged: list[_ActivityRegion] = [regions[0]]
    for region in regions[1:]:
        prev = merged[-1]
        gap = max(0.0, region.start_sec - prev.end_sec)
        if gap <= merge_gap_sec:
            total_dur = max(1e-6, (prev.end_sec - prev.start_sec) + (region.end_sec - region.start_sec))
            weighted_mean = (
                prev.mean_energy * (prev.end_sec - prev.start_sec)
                + region.mean_energy * (region.end_sec - region.start_sec)
            ) / total_dur
            merged[-1] = _ActivityRegion(
                start_sec=prev.start_sec,
                end_sec=max(prev.end_sec, region.end_sec),
                peak_energy=max(prev.peak_energy, region.peak_energy),
                mean_energy=weighted_mean,
                onset_count=prev.onset_count + region.onset_count,
                confidence=max(prev.confidence, region.confidence),
            )
            continue
        merged.append(region)

    return merged


def _selected_note_ids_from_plan(cleanup_plan: CleanupPlanDocument) -> set[str]:
    selected: set[str] = set()
    for action in cleanup_plan.actions:
        if action.plan_action in {"KEEP", "REVIEW"}:
            selected.add(action.note_id)
    return selected


def _build_midi_regions(notes: list[_MutableNote], params: ActivityRepairParameters) -> list[_MidiRegion]:
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: (n.refined_start_sec, n.refined_end_sec, n.note_id))
    merge_gap_sec = max(0.0, params.close_gap_ms / 1000.0)
    regions: list[_MidiRegion] = []

    current_start = sorted_notes[0].refined_start_sec
    current_end = sorted_notes[0].refined_end_sec
    current_ids = [sorted_notes[0].note_id]

    for note in sorted_notes[1:]:
        gap = note.refined_start_sec - current_end
        if gap <= merge_gap_sec:
            current_end = max(current_end, note.refined_end_sec)
            current_ids.append(note.note_id)
            continue

        regions.append(_MidiRegion(start_sec=current_start, end_sec=current_end, note_ids=current_ids))
        current_start = note.refined_start_sec
        current_end = note.refined_end_sec
        current_ids = [note.note_id]

    regions.append(_MidiRegion(start_sec=current_start, end_sec=current_end, note_ids=current_ids))
    return regions


def _region_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def _next_action_id(counter: int) -> str:
    return f"repair_action_{counter:06d}"


def _next_missing_note_id(counter: int) -> str:
    return f"repair_missing_{counter:06d}"


def _next_split_note_id(counter: int) -> str:
    return f"repair_split_{counter:06d}"


def _find_prev_note(notes: list[_MutableNote], start_sec: float) -> _MutableNote | None:
    candidates = [note for note in notes if note.refined_end_sec <= start_sec]
    if not candidates:
        return None
    return max(candidates, key=lambda note: note.refined_end_sec)


def _find_next_note(notes: list[_MutableNote], end_sec: float) -> _MutableNote | None:
    candidates = [note for note in notes if note.refined_start_sec >= end_sec]
    if not candidates:
        return None
    return min(candidates, key=lambda note: note.refined_start_sec)


def _find_activity_region_for_note(
    note: _MutableNote,
    regions: list[_ActivityRegion],
) -> _ActivityRegion | None:
    best: _ActivityRegion | None = None
    best_overlap = 0.0
    for region in regions:
        overlap = _region_overlap(note.refined_start_sec, note.refined_end_sec, region.start_sec, region.end_sec)
        if overlap > best_overlap:
            best_overlap = overlap
            best = region
    return best


def _build_onset_signal_from_dsp(frames: list[DspAudioFrame]) -> list[tuple[float, float]]:
    return [(float(frame.start_sec), max(float(frame.onset_strength), float(frame.spectral_flux))) for frame in frames]


def _build_onset_signal_from_audio(frames: list[AudioFrameFeature]) -> list[tuple[float, float]]:
    return [(float(frame.start_sec), float(frame.onset_score)) for frame in frames]


def _mean_signal_in_window(signal: list[tuple[float, float]], start_sec: float, end_sec: float) -> float:
    values = [value for sec, value in signal if start_sec <= sec <= end_sec]
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _max_signal_in_window(signal: list[tuple[float, float]], start_sec: float, end_sec: float) -> tuple[float, float] | None:
    items = [(sec, value) for sec, value in signal if start_sec <= sec <= end_sec]
    if not items:
        return None
    return max(items, key=lambda pair: pair[1])


def _closest_note(notes: list[_MutableNote], anchor_sec: float, search_sec: float) -> _MutableNote | None:
    candidates = [note for note in notes if abs(note.refined_start_sec - anchor_sec) <= search_sec or abs(note.refined_end_sec - anchor_sec) <= search_sec]
    if not candidates:
        return None
    return min(candidates, key=lambda note: min(abs(note.refined_start_sec - anchor_sec), abs(note.refined_end_sec - anchor_sec)))


def _window_signal_values(
    signal: list[tuple[float, float]],
    start_sec: float,
    end_sec: float,
) -> list[float]:
    return [value for sec, value in signal if start_sec <= sec <= end_sec]


def _pitch_frames_in_window(
    pitch_frames: list[BassPitchFrame],
    start_sec: float,
    end_sec: float,
) -> list[BassPitchFrame]:
    return [frame for frame in pitch_frames if frame.start_sec <= end_sec and frame.end_sec >= start_sec]


def _pitch_window_summary(
    pitch_frames: list[BassPitchFrame],
    start_sec: float,
    end_sec: float,
) -> _PitchWindowSummary:
    frames = _pitch_frames_in_window(pitch_frames, start_sec, end_sec)
    if not frames:
        return _PitchWindowSummary(voiced_ratio=0.0, mean_confidence=0.0, dominant_pitch_midi=None)

    voiced = [frame for frame in frames if frame.voiced]
    voiced_ratio = len(voiced) / float(len(frames))
    mean_conf = sum(frame.pitch_confidence for frame in frames) / float(len(frames))
    pitch_counts: dict[int, int] = {}
    for frame in voiced:
        if frame.pitch_midi_rounded is None:
            continue
        pitch_counts[frame.pitch_midi_rounded] = pitch_counts.get(frame.pitch_midi_rounded, 0) + 1
    dominant_pitch = None
    if pitch_counts:
        dominant_pitch = max(pitch_counts.items(), key=lambda item: item[1])[0]
    return _PitchWindowSummary(
        voiced_ratio=voiced_ratio,
        mean_confidence=mean_conf,
        dominant_pitch_midi=dominant_pitch,
    )


def _find_pitch_note_for_insert(
    pitch_frames: list[BassPitchFrame],
    start_sec: float,
    end_sec: float,
    min_confidence: float,
) -> int | None:
    candidates = _pitch_frames_in_window(pitch_frames, start_sec, end_sec)
    if not candidates:
        return None
    ranked = sorted(
        (
            frame
            for frame in candidates
            if frame.voiced
            and frame.pitch_midi_rounded is not None
            and frame.pitch_confidence >= min_confidence
        ),
        key=lambda frame: frame.pitch_confidence,
        reverse=True,
    )
    if not ranked:
        return None
    return ranked[0].pitch_midi_rounded


def _is_legato_transition(
    note: _MutableNote,
    notes: list[_MutableNote],
    audio_regions: list[_ActivityRegion],
    params: ActivityRepairParameters,
) -> tuple[bool, dict[str, float]]:
    window_sec = params.legato_neighbor_window_ms / 1000.0
    silence_need_sec = params.legato_min_silence_ms / 1000.0

    next_notes = [candidate for candidate in notes if candidate.refined_start_sec >= note.refined_end_sec and candidate.note_id != note.note_id]
    if not next_notes:
        return False, {}

    next_note = min(next_notes, key=lambda candidate: candidate.refined_start_sec)
    gap_sec = max(0.0, next_note.refined_start_sec - note.refined_end_sec)
    if gap_sec > window_sec:
        return False, {"next_gap_sec": gap_sec}

    overlap = any(
        _region_overlap(note.refined_end_sec, next_note.refined_start_sec, region.start_sec, region.end_sec) > 0.0
        for region in audio_regions
    )
    if overlap:
        return True, {
            "next_note_start_sec": next_note.refined_start_sec,
            "next_gap_sec": gap_sec,
            "required_silence_sec": silence_need_sec,
        }

    return gap_sec < silence_need_sec, {
        "next_note_start_sec": next_note.refined_start_sec,
        "next_gap_sec": gap_sec,
        "required_silence_sec": silence_need_sec,
    }


def _sustain_signal_guard(
    note: _MutableNote,
    candidate_end: float,
    signal: list[tuple[float, float]],
    params: ActivityRepairParameters,
) -> tuple[bool, float]:
    hold_sec = params.sustain_protect_hold_ms / 1000.0
    start = max(candidate_end, note.refined_start_sec)
    end = max(start, min(note.refined_end_sec, candidate_end + hold_sec))
    values = _window_signal_values(signal, start, end)
    if not values:
        return False, 0.0

    note_values = _window_signal_values(signal, note.refined_start_sec, note.refined_end_sec)
    if not note_values:
        note_values = values
    baseline = max(1e-6, max(note_values))
    mean_after = sum(values) / float(len(values))
    ratio = mean_after / baseline
    return ratio >= params.sustain_protect_ratio, ratio


def _pitch_sustain_guard(
    note: _MutableNote,
    candidate_end: float,
    pitch_frames: list[BassPitchFrame] | None,
    params: ActivityRepairParameters,
) -> tuple[bool, _PitchWindowSummary]:
    if not pitch_frames:
        return False, _PitchWindowSummary(voiced_ratio=0.0, mean_confidence=0.0, dominant_pitch_midi=None)

    hold_sec = params.pitch_sustain_hold_ms / 1000.0
    start = max(candidate_end, note.refined_start_sec)
    end = max(start, min(note.refined_end_sec, candidate_end + hold_sec))
    summary = _pitch_window_summary(pitch_frames, start, end)
    return summary.voiced_ratio > 0.0 and summary.mean_confidence >= 0.6, summary


def _to_mutable_note(note: RefinedNoteEvent) -> _MutableNote:
    return _MutableNote(
        note_id=note.note_id,
        source=note.source,
        layer=note.layer,
        pitch_midi=note.pitch_midi,
        pitch_name=note.pitch_name,
        velocity=note.velocity,
        channel=note.channel,
        original_start_sec=note.original_start_sec,
        original_end_sec=note.original_end_sec,
        aligned_start_sec=note.aligned_start_sec,
        aligned_end_sec=note.aligned_end_sec,
        refined_start_sec=max(0.0, float(note.refined_start_sec)),
        refined_end_sec=max(float(note.refined_start_sec), float(note.refined_end_sec)),
        merged_note_ids=list(note.merged_note_ids),
        refinement_actions=list(note.refinement_actions),
        refinement_confidence=float(note.refinement_confidence),
        reasons=list(note.reasons),
    )


def _to_refined_event(note: _MutableNote) -> RefinedNoteEvent:
    start = max(0.0, note.refined_start_sec)
    end = max(start, note.refined_end_sec)
    return RefinedNoteEvent(
        note_id=note.note_id,
        source=note.source,
        layer=note.layer,
        pitch_midi=note.pitch_midi,
        pitch_name=note.pitch_name,
        velocity=note.velocity,
        channel=note.channel,
        original_start_sec=note.original_start_sec,
        original_end_sec=note.original_end_sec,
        aligned_start_sec=note.aligned_start_sec,
        aligned_end_sec=note.aligned_end_sec,
        refined_start_sec=start,
        refined_end_sec=end,
        refined_duration_sec=max(0.0, end - start),
        start_refinement_ms=(start - note.aligned_start_sec) * 1000.0,
        end_refinement_ms=(end - note.aligned_end_sec) * 1000.0,
        merged_note_ids=list(note.merged_note_ids),
        refinement_actions=list(note.refinement_actions),
        refinement_confidence=max(0.0, min(1.0, float(note.refinement_confidence))),
        reasons=list(note.reasons),
    )


def _resolve_monophonic_overlaps(notes: list[_MutableNote]) -> int:
    if len(notes) < 2:
        return 0
    notes.sort(key=lambda n: (n.refined_start_sec, n.refined_end_sec, n.note_id))
    resolved = 0
    for idx in range(len(notes) - 1):
        current = notes[idx]
        nxt = notes[idx + 1]
        if current.refined_end_sec <= nxt.refined_start_sec:
            continue
        current.refined_end_sec = nxt.refined_start_sec
        if "ACTIVITY_REPAIR_GAP_CLOSED" not in current.refinement_actions:
            current.refinement_actions.append("ACTIVITY_REPAIR_GAP_CLOSED")
        current.reasons.append("trimmed overlap after activity repair")
        resolved += 1
    return resolved


def repair_activity(
    refined_notes_file: Path,
    audio_features_file: Path,
    cleanup_plan_file: Path,
    params: ActivityRepairParameters,
    dsp_features_file: Path | None = None,
    pitch_contour_file: Path | None = None,
) -> tuple[RefinedNoteDocument, ActivityRepairPlan, ActivityRepairReport]:
    if not refined_notes_file.exists() or not refined_notes_file.is_file():
        raise ActivityRepairError(f"Refined notes file does not exist: {refined_notes_file}")
    if not audio_features_file.exists() or not audio_features_file.is_file():
        raise ActivityRepairError(f"Audio features file does not exist: {audio_features_file}")
    if not cleanup_plan_file.exists() or not cleanup_plan_file.is_file():
        raise ActivityRepairError(f"Cleanup plan file does not exist: {cleanup_plan_file}")
    if dsp_features_file is not None and (not dsp_features_file.exists() or not dsp_features_file.is_file()):
        raise ActivityRepairError(f"DSP features file does not exist: {dsp_features_file}")
    if pitch_contour_file is not None and (
        not pitch_contour_file.exists() or not pitch_contour_file.is_file()
    ):
        raise ActivityRepairError(f"Pitch contour file does not exist: {pitch_contour_file}")

    refined_document = _load_refined_document(refined_notes_file)
    audio_document = _load_audio_document(audio_features_file)
    cleanup_plan = _load_cleanup_plan(cleanup_plan_file)
    dsp_document = _load_dsp_document(dsp_features_file) if dsp_features_file is not None else None
    pitch_document = _load_pitch_document(pitch_contour_file) if pitch_contour_file is not None else None

    warnings: list[str] = []
    if refined_document.layer != audio_document.layer:
        warnings.append(
            "Layer mismatch between refined notes and audio features: "
            f"{refined_document.layer} vs {audio_document.layer}."
        )
    if cleanup_plan.layer != refined_document.layer:
        warnings.append(
            "Layer mismatch between cleanup plan and refined notes: "
            f"{cleanup_plan.layer} vs {refined_document.layer}."
        )
    if dsp_document is not None and dsp_document.layer != refined_document.layer:
        warnings.append(
            "Layer mismatch between DSP features and refined notes: "
            f"{dsp_document.layer} vs {refined_document.layer}."
        )
    if pitch_document is not None and pitch_document.layer != refined_document.layer:
        warnings.append(
            "Layer mismatch between pitch contour and refined notes: "
            f"{pitch_document.layer} vs {refined_document.layer}."
        )

    selected_ids = _selected_note_ids_from_plan(cleanup_plan)
    selected_notes = [_to_mutable_note(note) for note in refined_document.notes if note.note_id in selected_ids]
    selected_notes.sort(key=lambda n: (n.refined_start_sec, n.refined_end_sec, n.note_id))

    if not selected_notes:
        raise ActivityRepairError("No KEEP/REVIEW notes available for activity repair")

    if dsp_document is not None:
        activity_frames = _frames_from_dsp(dsp_document=dsp_document, audio_document=audio_document, params=params)
        onset_signal = _build_onset_signal_from_dsp(dsp_document.frames)
        sustain_signal = [
            (float(frame.start_sec), float(max(frame.low_band_envelope_smooth, frame.harmonic_rms)))
            for frame in dsp_document.frames
        ]
    else:
        activity_frames = _frames_from_audio(audio_document)
        onset_signal = _build_onset_signal_from_audio(audio_document.frames)
        sustain_signal = [(float(frame.start_sec), float(frame.rms)) for frame in audio_document.frames]

    pitch_frames = pitch_document.frames if pitch_document is not None else None

    audio_regions = _build_audio_activity_regions(activity_frames, params)
    midi_regions = _build_midi_regions(selected_notes, params)

    actions: list[RepairAction] = []
    action_counter = 1
    missing_counter = 1
    split_counter = 1

    extend_count = 0
    shorten_count = 0
    insert_missing_count = 0
    split_count = 0
    close_gap_count = 0
    review_manual_count = 0
    keep_count = 0
    sustain_protected_count = 0
    pitch_protected_count = 0
    legato_protected_count = 0
    shorten_candidate_count = 0
    shorten_applied_count = 0
    shorten_rejected_count = 0
    audio_gap_count = 0
    midi_overhang_count = 0
    conflict_resolved_count = 0
    suppressed_conflict_action_count = 0

    missing_min_sec = params.missing_gap_min_ms / 1000.0
    max_extend_sec = params.max_extend_for_gap_ms / 1000.0
    max_insert_sec = params.max_insert_missing_ms / 1000.0
    context_sec = params.context_pitch_search_ms / 1000.0
    extended_note_ids_this_pass: set[str] = set()
    inserted_note_ids_this_pass: set[str] = set()

    for region in audio_regions:
        overlap = 0.0
        for midi_region in midi_regions:
            overlap += _region_overlap(region.start_sec, region.end_sec, midi_region.start_sec, midi_region.end_sec)

        uncovered = max(0.0, (region.end_sec - region.start_sec) - overlap)
        if uncovered < missing_min_sec:
            continue

        audio_gap_count += 1
        overlap_notes = [
            note
            for note in selected_notes
            if _region_overlap(
                region.start_sec,
                region.end_sec,
                note.refined_start_sec,
                note.refined_end_sec,
            )
            > 0.0
        ]
        if overlap_notes:
            prev_note = max(overlap_notes, key=lambda note: note.refined_end_sec)
        else:
            prev_note = _find_prev_note(selected_notes, region.start_sec)
        next_note = _find_next_note(selected_notes, region.end_sec)

        if prev_note is not None and prev_note.refined_end_sec < region.end_sec:
            extend_by = region.end_sec - prev_note.refined_end_sec
            if 0.0 < extend_by <= max_extend_sec:
                old_end = prev_note.refined_end_sec
                prev_note.refined_end_sec = max(prev_note.refined_end_sec, region.end_sec)
                if "ACTIVITY_REPAIR_EXTENDED" not in prev_note.refinement_actions:
                    prev_note.refinement_actions.append("ACTIVITY_REPAIR_EXTENDED")
                prev_note.reasons.append("extended note to cover audio-only activity")
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="EXTEND_NOTE",
                        target_note_id=prev_note.note_id,
                        new_note_id=None,
                        start_sec=old_end,
                        end_sec=prev_note.refined_end_sec,
                        old_start_sec=prev_note.refined_start_sec,
                        old_end_sec=old_end,
                        new_start_sec=prev_note.refined_start_sec,
                        new_end_sec=prev_note.refined_end_sec,
                        pitch_midi=prev_note.pitch_midi,
                        confidence=min(1.0, max(0.0, 0.7 + region.confidence * 0.2)),
                        reasons=["audio region extends past note end"],
                        evidence={
                            "audio_region_start_sec": region.start_sec,
                            "audio_region_end_sec": region.end_sec,
                            "uncovered_sec": uncovered,
                        },
                    )
                )
                action_counter += 1
                extend_count += 1
                extended_note_ids_this_pass.add(prev_note.note_id)
                continue

        if next_note is not None and region.start_sec < next_note.refined_start_sec:
            extend_by = next_note.refined_start_sec - region.start_sec
            if 0.0 < extend_by <= max_extend_sec:
                old_start = next_note.refined_start_sec
                next_note.refined_start_sec = max(0.0, region.start_sec)
                if "ACTIVITY_REPAIR_GAP_CLOSED" not in next_note.refinement_actions:
                    next_note.refinement_actions.append("ACTIVITY_REPAIR_GAP_CLOSED")
                next_note.reasons.append("closed front gap against active audio")
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="CLOSE_GAP",
                        target_note_id=next_note.note_id,
                        new_note_id=None,
                        start_sec=next_note.refined_start_sec,
                        end_sec=old_start,
                        old_start_sec=old_start,
                        old_end_sec=next_note.refined_end_sec,
                        new_start_sec=next_note.refined_start_sec,
                        new_end_sec=next_note.refined_end_sec,
                        pitch_midi=next_note.pitch_midi,
                        confidence=min(1.0, max(0.0, 0.65 + region.confidence * 0.2)),
                        reasons=["audio region starts before next note"],
                        evidence={
                            "audio_region_start_sec": region.start_sec,
                            "audio_region_end_sec": region.end_sec,
                            "uncovered_sec": uncovered,
                        },
                    )
                )
                action_counter += 1
                close_gap_count += 1
                continue

        region_len = region.end_sec - region.start_sec
        nearest = _closest_note(selected_notes, (region.start_sec + region.end_sec) * 0.5, context_sec)
        contour_pitch = (
            _find_pitch_note_for_insert(
                pitch_frames=pitch_frames if pitch_frames is not None else [],
                start_sec=region.start_sec,
                end_sec=region.end_sec,
                min_confidence=params.insert_from_pitch_contour_confidence,
            )
            if pitch_frames is not None
            else None
        )
        if nearest is not None and region_len <= max_insert_sec:
            confidence = min(1.0, max(0.0, 0.55 + (region.confidence * 0.4)))
            if confidence >= params.insert_auto_confidence:
                new_note_id = _next_missing_note_id(missing_counter)
                missing_counter += 1
                selected_pitch = contour_pitch if contour_pitch is not None else nearest.pitch_midi
                new_note = _MutableNote(
                    note_id=new_note_id,
                    source=nearest.source if nearest.source else "hermes_repair",
                    layer=refined_document.layer,
                    pitch_midi=selected_pitch,
                    pitch_name=nearest.pitch_name,
                    velocity=nearest.velocity if nearest.velocity > 0 else 80,
                    channel=nearest.channel,
                    original_start_sec=region.start_sec,
                    original_end_sec=region.end_sec,
                    aligned_start_sec=region.start_sec,
                    aligned_end_sec=region.end_sec,
                    refined_start_sec=region.start_sec,
                    refined_end_sec=region.end_sec,
                    merged_note_ids=[],
                    refinement_actions=["ACTIVITY_REPAIR_INSERTED"],
                    refinement_confidence=confidence,
                    reasons=["inserted missing note from audio activity"],
                )
                selected_notes.append(new_note)
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="INSERT_MISSING_NOTE",
                        target_note_id=None,
                        new_note_id=new_note_id,
                        start_sec=region.start_sec,
                        end_sec=region.end_sec,
                        old_start_sec=None,
                        old_end_sec=None,
                        new_start_sec=region.start_sec,
                        new_end_sec=region.end_sec,
                        pitch_midi=new_note.pitch_midi,
                        confidence=confidence,
                        reasons=["audio-only region with nearby pitch context"],
                        evidence={
                            "context_note_id": nearest.note_id,
                            "pitch_from_contour": contour_pitch,
                            "audio_region_start_sec": region.start_sec,
                            "audio_region_end_sec": region.end_sec,
                            "region_confidence": region.confidence,
                        },
                    )
                )
                action_counter += 1
                insert_missing_count += 1
                inserted_note_ids_this_pass.add(new_note_id)
            else:
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="REVIEW_MANUAL",
                        target_note_id=None,
                        new_note_id=None,
                        start_sec=region.start_sec,
                        end_sec=region.end_sec,
                        old_start_sec=None,
                        old_end_sec=None,
                        new_start_sec=None,
                        new_end_sec=None,
                        pitch_midi=None,
                        confidence=confidence,
                        reasons=["insufficient confidence for auto insert"],
                        evidence={
                            "context_note_id": nearest.note_id,
                            "audio_region_start_sec": region.start_sec,
                            "audio_region_end_sec": region.end_sec,
                        },
                    )
                )
                action_counter += 1
                review_manual_count += 1
        elif contour_pitch is not None and region_len <= max_insert_sec:
            confidence = min(1.0, max(0.0, 0.50 + (region.confidence * 0.35)))
            if confidence >= params.insert_auto_confidence:
                new_note_id = _next_missing_note_id(missing_counter)
                missing_counter += 1
                new_note = _MutableNote(
                    note_id=new_note_id,
                    source="hermes_repair",
                    layer=refined_document.layer,
                    pitch_midi=contour_pitch,
                    pitch_name=f"MIDI_{contour_pitch}",
                    velocity=80,
                    channel=0,
                    original_start_sec=region.start_sec,
                    original_end_sec=region.end_sec,
                    aligned_start_sec=region.start_sec,
                    aligned_end_sec=region.end_sec,
                    refined_start_sec=region.start_sec,
                    refined_end_sec=region.end_sec,
                    merged_note_ids=[],
                    refinement_actions=["ACTIVITY_REPAIR_INSERTED"],
                    refinement_confidence=confidence,
                    reasons=["inserted missing note from pitch contour"],
                )
                selected_notes.append(new_note)
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="INSERT_MISSING_NOTE",
                        target_note_id=None,
                        new_note_id=new_note_id,
                        start_sec=region.start_sec,
                        end_sec=region.end_sec,
                        old_start_sec=None,
                        old_end_sec=None,
                        new_start_sec=region.start_sec,
                        new_end_sec=region.end_sec,
                        pitch_midi=contour_pitch,
                        confidence=confidence,
                        reasons=["audio-only region with pitch contour evidence"],
                        evidence={
                            "audio_region_start_sec": region.start_sec,
                            "audio_region_end_sec": region.end_sec,
                            "pitch_from_contour": contour_pitch,
                            "region_confidence": region.confidence,
                        },
                    )
                )
                action_counter += 1
                insert_missing_count += 1
                inserted_note_ids_this_pass.add(new_note_id)
            else:
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="REVIEW_MANUAL",
                        target_note_id=None,
                        new_note_id=None,
                        start_sec=region.start_sec,
                        end_sec=region.end_sec,
                        old_start_sec=None,
                        old_end_sec=None,
                        new_start_sec=None,
                        new_end_sec=None,
                        pitch_midi=contour_pitch,
                        confidence=confidence,
                        reasons=["insufficient confidence for contour-driven insert"],
                        evidence={
                            "audio_region_start_sec": region.start_sec,
                            "audio_region_end_sec": region.end_sec,
                            "pitch_from_contour": contour_pitch,
                        },
                    )
                )
                action_counter += 1
                review_manual_count += 1
        else:
            actions.append(
                RepairAction(
                    action_id=_next_action_id(action_counter),
                    action_type="REVIEW_MANUAL",
                    target_note_id=None,
                    new_note_id=None,
                    start_sec=region.start_sec,
                    end_sec=region.end_sec,
                    old_start_sec=None,
                    old_end_sec=None,
                    new_start_sec=None,
                    new_end_sec=None,
                    pitch_midi=None,
                    confidence=0.4,
                    reasons=["audio-only region without usable pitch context"],
                    evidence={
                        "audio_region_start_sec": region.start_sec,
                        "audio_region_end_sec": region.end_sec,
                        "region_duration_sec": region_len,
                    },
                )
            )
            action_counter += 1
            review_manual_count += 1

    selected_notes.sort(key=lambda n: (n.refined_start_sec, n.refined_end_sec, n.note_id))
    min_repaired_sec = params.minimum_repaired_note_duration_ms / 1000.0
    tail_padding_sec = params.tail_padding_ms / 1000.0
    overhang_min_sec = params.overhang_min_ms / 1000.0

    for note in selected_notes:
        region = _find_activity_region_for_note(note, audio_regions)
        if region is None:
            continue

        overhang = note.refined_end_sec - region.end_sec
        if overhang >= overhang_min_sec:
            midi_overhang_count += 1
            shorten_candidate_count += 1
            old_end = note.refined_end_sec
            candidate_end = min(note.refined_end_sec, region.end_sec + tail_padding_sec)

            is_legato, legato_evidence = _is_legato_transition(
                note=note,
                notes=selected_notes,
                audio_regions=audio_regions,
                params=params,
            )
            sustain_protect, sustain_ratio = _sustain_signal_guard(
                note=note,
                candidate_end=candidate_end,
                signal=sustain_signal,
                params=params,
            )
            pitch_protect, pitch_summary = _pitch_sustain_guard(
                note=note,
                candidate_end=candidate_end,
                pitch_frames=pitch_frames,
                params=params,
            )

            if is_legato:
                if "ACTIVITY_REPAIR_LEGATO_PROTECTED" not in note.refinement_actions:
                    note.refinement_actions.append("ACTIVITY_REPAIR_LEGATO_PROTECTED")
                note.reasons.append("legato transition protection prevented shorten")
                legato_protected_count += 1
                shorten_rejected_count += 1
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="REVIEW_MANUAL",
                        target_note_id=note.note_id,
                        new_note_id=None,
                        start_sec=candidate_end,
                        end_sec=old_end,
                        old_start_sec=note.refined_start_sec,
                        old_end_sec=old_end,
                        new_start_sec=note.refined_start_sec,
                        new_end_sec=None,
                        pitch_midi=note.pitch_midi,
                        confidence=0.55,
                        reasons=["LEGATO_PROTECTED_FROM_SHORTEN"],
                        evidence={
                            "audio_region_end_sec": region.end_sec,
                            **legato_evidence,
                        },
                    )
                )
                action_counter += 1
                continue

            if sustain_protect:
                if "ACTIVITY_REPAIR_SUSTAIN_PROTECTED" not in note.refinement_actions:
                    note.refinement_actions.append("ACTIVITY_REPAIR_SUSTAIN_PROTECTED")
                note.reasons.append("sustain energy protection prevented shorten")
                sustain_protected_count += 1
                shorten_rejected_count += 1
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="REVIEW_MANUAL",
                        target_note_id=note.note_id,
                        new_note_id=None,
                        start_sec=candidate_end,
                        end_sec=old_end,
                        old_start_sec=note.refined_start_sec,
                        old_end_sec=old_end,
                        new_start_sec=note.refined_start_sec,
                        new_end_sec=None,
                        pitch_midi=note.pitch_midi,
                        confidence=0.6,
                        reasons=["SUSTAIN_PROTECTED_FROM_SHORTEN"],
                        evidence={
                            "audio_region_end_sec": region.end_sec,
                            "sustain_ratio": sustain_ratio,
                        },
                    )
                )
                action_counter += 1
                continue

            if pitch_protect:
                if "ACTIVITY_REPAIR_PITCH_PROTECTED" not in note.refinement_actions:
                    note.refinement_actions.append("ACTIVITY_REPAIR_PITCH_PROTECTED")
                note.reasons.append("pitch contour protection prevented shorten")
                pitch_protected_count += 1
                shorten_rejected_count += 1
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="REVIEW_MANUAL",
                        target_note_id=note.note_id,
                        new_note_id=None,
                        start_sec=candidate_end,
                        end_sec=old_end,
                        old_start_sec=note.refined_start_sec,
                        old_end_sec=old_end,
                        new_start_sec=note.refined_start_sec,
                        new_end_sec=None,
                        pitch_midi=note.pitch_midi,
                        confidence=0.62,
                        reasons=["PITCH_CONTOUR_PROTECTED_FROM_SHORTEN"],
                        evidence={
                            "audio_region_end_sec": region.end_sec,
                            "pitch_voiced_ratio": pitch_summary.voiced_ratio,
                            "pitch_mean_confidence": pitch_summary.mean_confidence,
                            "dominant_pitch_midi": pitch_summary.dominant_pitch_midi,
                        },
                    )
                )
                action_counter += 1
                continue

            if candidate_end - note.refined_start_sec >= min_repaired_sec:
                note.refined_end_sec = candidate_end
                if "ACTIVITY_REPAIR_SHORTENED" not in note.refinement_actions:
                    note.refinement_actions.append("ACTIVITY_REPAIR_SHORTENED")
                note.reasons.append("shortened note to audio activity boundary")
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="SHORTEN_NOTE",
                        target_note_id=note.note_id,
                        new_note_id=None,
                        start_sec=candidate_end,
                        end_sec=old_end,
                        old_start_sec=note.refined_start_sec,
                        old_end_sec=old_end,
                        new_start_sec=note.refined_start_sec,
                        new_end_sec=candidate_end,
                        pitch_midi=note.pitch_midi,
                        confidence=0.82,
                        reasons=["note overhang past audio-active region"],
                        evidence={
                            "audio_region_end_sec": region.end_sec,
                            "old_end_sec": old_end,
                            "new_end_sec": candidate_end,
                            "shorten_candidate": True,
                        },
                    )
                )
                action_counter += 1
                shorten_count += 1
                shorten_applied_count += 1
            else:
                shorten_rejected_count += 1
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="REVIEW_MANUAL",
                        target_note_id=note.note_id,
                        new_note_id=None,
                        start_sec=candidate_end,
                        end_sec=old_end,
                        old_start_sec=note.refined_start_sec,
                        old_end_sec=old_end,
                        new_start_sec=note.refined_start_sec,
                        new_end_sec=None,
                        pitch_midi=note.pitch_midi,
                        confidence=0.5,
                        reasons=["minimum_repaired_note_duration guard"],
                        evidence={
                            "audio_region_end_sec": region.end_sec,
                            "old_end_sec": old_end,
                            "proposed_end_sec": candidate_end,
                            "minimum_repaired_note_duration_ms": params.minimum_repaired_note_duration_ms,
                        },
                    )
                )
                action_counter += 1

    if dsp_document is not None:
        split_signal = _build_onset_signal_from_dsp(dsp_document.frames)
    else:
        split_signal = _build_onset_signal_from_audio(audio_document.frames)

    split_min_dur_sec = params.split_min_note_duration_ms / 1000.0
    split_edge_sec = params.split_min_distance_from_edges_ms / 1000.0

    split_candidates: list[tuple[_MutableNote, float, float, float, bool]] = []
    for note in selected_notes:
        if note.duration_sec() < split_min_dur_sec:
            continue

        inner_start = note.refined_start_sec + split_edge_sec
        inner_end = note.refined_end_sec - split_edge_sec
        if inner_end <= inner_start:
            continue

        onset_peak = _max_signal_in_window(split_signal, inner_start, inner_end)
        if onset_peak is None:
            continue

        local_mean = _mean_signal_in_window(split_signal, inner_start, inner_end)
        onset_sec, onset_value = onset_peak
        if onset_value < max(1e-6, local_mean * params.split_onset_strength_ratio):
            continue

        valley_start = max(note.refined_start_sec, onset_sec - 0.12)
        valley_end = onset_sec
        valley_mean = _mean_signal_in_window(split_signal, valley_start, valley_end)
        if valley_mean > max(1e-6, onset_value * params.split_valley_ratio):
            continue

        pitch_change = 0.0
        strong_independent_pitch_change = False
        if pitch_frames is not None:
            before = _pitch_window_summary(pitch_frames, inner_start, max(inner_start, onset_sec - 0.04))
            after = _pitch_window_summary(pitch_frames, min(inner_end, onset_sec + 0.04), inner_end)
            if before.dominant_pitch_midi is not None and after.dominant_pitch_midi is not None:
                pitch_change = abs(float(after.dominant_pitch_midi - before.dominant_pitch_midi))
            strong_independent_pitch_change = (
                pitch_change >= max(1.5, params.split_pitch_change_semitones + 0.5)
                and before.voiced_ratio >= 0.5
                and after.voiced_ratio >= 0.5
                and before.mean_confidence >= 0.7
                and after.mean_confidence >= 0.7
            )
            if pitch_change < params.split_pitch_change_semitones and valley_mean >= max(1e-6, onset_value * 0.02):
                actions.append(
                    RepairAction(
                        action_id=_next_action_id(action_counter),
                        action_type="REVIEW_MANUAL",
                        target_note_id=note.note_id,
                        new_note_id=None,
                        start_sec=note.refined_start_sec,
                        end_sec=note.refined_end_sec,
                        old_start_sec=note.refined_start_sec,
                        old_end_sec=note.refined_end_sec,
                        new_start_sec=None,
                        new_end_sec=None,
                        pitch_midi=note.pitch_midi,
                        confidence=0.5,
                        reasons=["split candidate has no meaningful pitch contour change"],
                        evidence={
                            "split_onset_sec": onset_sec,
                            "pitch_change_semitones": pitch_change,
                            "required_pitch_change_semitones": params.split_pitch_change_semitones,
                        },
                    )
                )
                action_counter += 1
                review_manual_count += 1
                continue

        split_confidence = min(1.0, max(0.0, 0.65 + (onset_value / max(1e-6, local_mean)) * 0.1))
        if pitch_change >= params.split_pitch_change_semitones:
            split_confidence = min(1.0, split_confidence + 0.12)
        if split_confidence < params.split_auto_confidence:
            actions.append(
                RepairAction(
                    action_id=_next_action_id(action_counter),
                    action_type="REVIEW_MANUAL",
                    target_note_id=note.note_id,
                    new_note_id=None,
                    start_sec=note.refined_start_sec,
                    end_sec=note.refined_end_sec,
                    old_start_sec=note.refined_start_sec,
                    old_end_sec=note.refined_end_sec,
                    new_start_sec=None,
                    new_end_sec=None,
                    pitch_midi=note.pitch_midi,
                    confidence=split_confidence,
                    reasons=["split candidate below auto threshold"],
                    evidence={
                        "split_onset_sec": onset_sec,
                        "split_onset_value": onset_value,
                        "local_mean": local_mean,
                        "pitch_change_semitones": pitch_change,
                    },
                )
            )
            action_counter += 1
            review_manual_count += 1
            continue

        split_candidates.append(
            (note, onset_sec, split_confidence, pitch_change, strong_independent_pitch_change)
        )

    is_bass_layer = refined_document.layer.lower() == "bass"
    for note, split_sec, split_confidence, pitch_change, strong_pitch in split_candidates:
        if split_sec <= note.refined_start_sec or split_sec >= note.refined_end_sec:
            continue

        if note.note_id in inserted_note_ids_this_pass:
            warnings.append(
                "conflict-suppressed action: "
                f"SPLIT_NOTE target={note.note_id} reason=inserted_note_same_pass"
            )
            conflict_resolved_count += 1
            suppressed_conflict_action_count += 1
            continue

        if note.note_id in extended_note_ids_this_pass and (is_bass_layer and not strong_pitch):
            warnings.append(
                "conflict-suppressed action: "
                f"SPLIT_NOTE target={note.note_id} reason=extend_split_conflict "
                f"pitch_change={pitch_change:.3f}"
            )
            conflict_resolved_count += 1
            suppressed_conflict_action_count += 1
            continue

        old_end = note.refined_end_sec
        new_note_id = _next_split_note_id(split_counter)
        split_counter += 1

        note.refined_end_sec = split_sec
        if "ACTIVITY_REPAIR_SPLIT" not in note.refinement_actions:
            note.refinement_actions.append("ACTIVITY_REPAIR_SPLIT")
        note.reasons.append("split note at internal attack event")

        second = _MutableNote(
            note_id=new_note_id,
            source=note.source,
            layer=note.layer,
            pitch_midi=note.pitch_midi,
            pitch_name=note.pitch_name,
            velocity=note.velocity,
            channel=note.channel,
            original_start_sec=split_sec,
            original_end_sec=old_end,
            aligned_start_sec=split_sec,
            aligned_end_sec=old_end,
            refined_start_sec=split_sec,
            refined_end_sec=old_end,
            merged_note_ids=[],
            refinement_actions=["ACTIVITY_REPAIR_SPLIT"],
            refinement_confidence=split_confidence,
            reasons=["created by activity repair split"],
        )
        selected_notes.append(second)

        actions.append(
            RepairAction(
                action_id=_next_action_id(action_counter),
                action_type="SPLIT_NOTE",
                target_note_id=note.note_id,
                new_note_id=new_note_id,
                start_sec=note.refined_start_sec,
                end_sec=old_end,
                old_start_sec=note.refined_start_sec,
                old_end_sec=old_end,
                new_start_sec=note.refined_start_sec,
                new_end_sec=split_sec,
                pitch_midi=note.pitch_midi,
                confidence=split_confidence,
                reasons=["strong internal onset and valley detected"],
                evidence={
                    "split_sec": split_sec,
                    "old_end_sec": old_end,
                },
            )
        )
        action_counter += 1
        split_count += 1

    if suppressed_conflict_action_count > 0:
        warnings.append(
            "activity repair conflict resolution: "
            f"conflict_resolved_count={conflict_resolved_count}, "
            f"suppressed_conflict_action_count={suppressed_conflict_action_count}"
        )

    selected_notes.sort(key=lambda n: (n.refined_start_sec, n.refined_end_sec, n.note_id))
    close_gap_sec = params.close_gap_ms / 1000.0
    for idx in range(len(selected_notes) - 1):
        current = selected_notes[idx]
        nxt = selected_notes[idx + 1]
        if abs(current.pitch_midi - nxt.pitch_midi) > params.near_pitch_tolerance_semitones:
            continue
        gap = nxt.refined_start_sec - current.refined_end_sec
        if gap <= 0.0 or gap > close_gap_sec:
            continue
        overlap = any(
            _region_overlap(current.refined_end_sec, nxt.refined_start_sec, region.start_sec, region.end_sec) > 0.0
            for region in audio_regions
        )
        if not overlap:
            continue

        old_end = current.refined_end_sec
        current.refined_end_sec = nxt.refined_start_sec
        if "ACTIVITY_REPAIR_GAP_CLOSED" not in current.refinement_actions:
            current.refinement_actions.append("ACTIVITY_REPAIR_GAP_CLOSED")
        current.reasons.append("closed tiny same-pitch gap with active audio")
        actions.append(
            RepairAction(
                action_id=_next_action_id(action_counter),
                action_type="CLOSE_GAP",
                target_note_id=current.note_id,
                new_note_id=None,
                start_sec=old_end,
                end_sec=current.refined_end_sec,
                old_start_sec=current.refined_start_sec,
                old_end_sec=old_end,
                new_start_sec=current.refined_start_sec,
                new_end_sec=current.refined_end_sec,
                pitch_midi=current.pitch_midi,
                confidence=0.86,
                reasons=["tiny same-pitch gap while audio stays active"],
                evidence={
                    "next_note_id": nxt.note_id,
                    "gap_sec": gap,
                },
            )
        )
        action_counter += 1
        close_gap_count += 1

    _resolve_monophonic_overlaps(selected_notes)

    untouched_ids = {note.note_id for note in selected_notes}
    repaired_notes = list(selected_notes)
    for note in refined_document.notes:
        if note.note_id in untouched_ids:
            continue
        repaired_notes.append(_to_mutable_note(note))

    repaired_notes.sort(key=lambda n: (n.refined_start_sec, n.refined_end_sec, n.note_id))

    action_target_ids = {action.target_note_id for action in actions if action.target_note_id is not None}
    for note in repaired_notes:
        if note.note_id in action_target_ids:
            continue
        actions.append(
            RepairAction(
                action_id=_next_action_id(action_counter),
                action_type="KEEP",
                target_note_id=note.note_id,
                new_note_id=None,
                start_sec=note.refined_start_sec,
                end_sec=note.refined_end_sec,
                old_start_sec=note.refined_start_sec,
                old_end_sec=note.refined_end_sec,
                new_start_sec=note.refined_start_sec,
                new_end_sec=note.refined_end_sec,
                pitch_midi=note.pitch_midi,
                confidence=max(0.0, min(1.0, note.refinement_confidence)),
                reasons=["no activity repair needed"],
                evidence={},
            )
        )
        action_counter += 1
        keep_count += 1

    repaired_document = RefinedNoteDocument(
        schema_version=refined_document.schema_version,
        aligned_notes_file=refined_document.aligned_notes_file,
        audio_features_file=refined_document.audio_features_file,
        validation_file=refined_document.validation_file,
        layer=refined_document.layer,
        sample_rate=refined_document.sample_rate,
        audio_duration_sec=refined_document.audio_duration_sec,
        timing_source=refined_document.timing_source,
        refinement_parameters=dict(refined_document.refinement_parameters),
        notes=[_to_refined_event(note) for note in repaired_notes],
    )

    repair_plan = ActivityRepairPlan(
        schema_version=SCHEMA_VERSION,
        refined_notes_file=str(refined_notes_file),
        audio_features_file=str(audio_features_file),
        dsp_features_file=str(dsp_features_file) if dsp_features_file is not None else None,
        pitch_contour_file=str(pitch_contour_file) if pitch_contour_file is not None else None,
        cleanup_plan_file=str(cleanup_plan_file),
        layer=refined_document.layer,
        actions=actions,
    )

    report = ActivityRepairReport(
        refined_notes_file=str(refined_notes_file),
        audio_features_file=str(audio_features_file),
        dsp_features_file=str(dsp_features_file) if dsp_features_file is not None else None,
        pitch_contour_file=str(pitch_contour_file) if pitch_contour_file is not None else None,
        cleanup_plan_file=str(cleanup_plan_file),
        status="ok",
        layer=refined_document.layer,
        input_note_count=len(refined_document.notes),
        output_note_count=len(repaired_document.notes),
        extend_count=extend_count,
        shorten_count=shorten_count,
        insert_missing_count=insert_missing_count,
        split_count=split_count,
        close_gap_count=close_gap_count,
        review_manual_count=review_manual_count,
        keep_count=keep_count,
        sustain_protected_count=sustain_protected_count,
        pitch_protected_count=pitch_protected_count,
        legato_protected_count=legato_protected_count,
        shorten_candidate_count=shorten_candidate_count,
        shorten_applied_count=shorten_applied_count,
        shorten_rejected_count=shorten_rejected_count,
        audio_active_region_count=len(audio_regions),
        midi_active_region_count=len(midi_regions),
        audio_gap_count=audio_gap_count,
        midi_overhang_count=midi_overhang_count,
        warning_count=len(warnings),
        warnings=warnings,
        output_file=None,
        plan_file=None,
    )

    return repaired_document, repair_plan, report
