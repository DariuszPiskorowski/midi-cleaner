from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

from midi_cleaner.audio.models import AudioFeatureDocument
from midi_cleaner.cleanup.models import CleanupPlanDocument
from midi_cleaner.dsp.models import DspAudioFeatureDocument
from midi_cleaner.pitch.models import BassPitchContourDocument, BassPitchFrame
from midi_cleaner.refinement.models import RefinedNoteDocument, RefinedNoteEvent
from midi_cleaner.repair.activity import (
    ActivityRepairError,
    ActivityRepairParameters,
    _ActivityFrame,
    _build_audio_activity_regions,
    _frames_from_audio,
    _frames_from_dsp,
    _selected_note_ids_from_plan,
    repair_activity,
)
from midi_cleaner.repair.models import (
    ActivityRepairPlan,
    ActivityRepairReport,
    IterativeRepairReport,
    IterationScoringReport,
    RepairIterationSummary,
)


class IterativeRepairError(Exception):
    """Raised when iterative activity repair cannot be completed."""


@dataclass(frozen=True)
class IterativeRepairParameters:
    max_iterations: int = 3
    min_improvement: float = 0.005
    allow_regression: bool = False
    conservative_final_pass: bool = True
    max_actions_per_iteration: int | None = None
    freeze_stable_notes: bool = True
    stable_note_tolerance_ms: float = 20.0
    protect_previous_good_regions: bool = True
    pass1_profile: str = "balanced"
    pass2_profile: str = "sustain_legato"
    pass3_profile: str = "conservative"


@dataclass(frozen=True)
class IterationArtifacts:
    iteration_index: int
    repaired_document: RefinedNoteDocument
    repair_plan: ActivityRepairPlan
    repair_report: ActivityRepairReport
    score_report: IterationScoringReport
    summary: RepairIterationSummary


@dataclass(frozen=True)
class _RepairErrorRegion:
    region_type: str
    start_sec: float
    end_sec: float
    severity: float
    note_id: str | None


@dataclass(frozen=True)
class _CandidateScore:
    coverage_score: float
    overhang_score: float
    continuity_score: float
    pitch_consistency_score: float
    total_score: float
    audio_gap_count: int
    midi_overhang_count: int
    unresolved_error_count: int
    error_regions: list[_RepairErrorRegion]


def _load_audio_document(path: Path) -> AudioFeatureDocument:
    try:
        return AudioFeatureDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise IterativeRepairError(f"Invalid audio features JSON: {path}") from exc


def _load_dsp_document(path: Path) -> DspAudioFeatureDocument:
    try:
        return DspAudioFeatureDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise IterativeRepairError(f"Invalid DSP features JSON: {path}") from exc


def _load_pitch_document(path: Path) -> BassPitchContourDocument:
    try:
        return BassPitchContourDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise IterativeRepairError(f"Invalid bass pitch contour JSON: {path}") from exc


def _load_refined_document(path: Path) -> RefinedNoteDocument:
    try:
        return RefinedNoteDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise IterativeRepairError(f"Invalid refined notes JSON: {path}") from exc


def _load_cleanup_plan(path: Path) -> CleanupPlanDocument:
    try:
        return CleanupPlanDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise IterativeRepairError(f"Invalid cleanup plan JSON: {path}") from exc


def _write_refined_document(path: Path, document: RefinedNoteDocument) -> None:
    path.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _build_activity_frames(
    audio_document: AudioFeatureDocument,
    dsp_document: DspAudioFeatureDocument | None,
    params: ActivityRepairParameters,
) -> list[_ActivityFrame]:
    if dsp_document is not None:
        return _frames_from_dsp(
            dsp_document=dsp_document,
            audio_document=audio_document,
            params=params,
        )
    return _frames_from_audio(audio_document)


def _build_candidate_ranges(
    notes: list[RefinedNoteEvent],
    selected_ids: set[str],
) -> list[tuple[float, float, str]]:
    ranges: list[tuple[float, float, str]] = []
    for note in notes:
        if note.note_id not in selected_ids and not note.note_id.startswith(
            ("repair_missing_", "repair_split_")
        ):
            continue
        start_sec = max(0.0, float(note.refined_start_sec))
        end_sec = max(start_sec, float(note.refined_end_sec))
        ranges.append((start_sec, end_sec, note.note_id))
    ranges.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranges


def _total_span_coverage(
    start_sec: float,
    end_sec: float,
    spans: list[tuple[float, float]],
) -> float:
    if end_sec <= start_sec:
        return 0.0
    covered = 0.0
    for span_start, span_end in spans:
        overlap_start = max(start_sec, span_start)
        overlap_end = min(end_sec, span_end)
        if overlap_end > overlap_start:
            covered += overlap_end - overlap_start
    return max(0.0, covered)


def _merged_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []
    sorted_spans = sorted((max(0.0, start), max(0.0, end)) for start, end in spans)
    merged: list[tuple[float, float]] = []
    current_start, current_end = sorted_spans[0]
    for start, end in sorted_spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _activity_audio_spans(
    frames: list[_ActivityFrame],
    params: ActivityRepairParameters,
) -> list[tuple[float, float]]:
    regions = _build_audio_activity_regions(frames=frames, params=params)
    return [(region.start_sec, region.end_sec) for region in regions]


def _extract_pitch_for_window(
    frames: list[BassPitchFrame],
    start_sec: float,
    end_sec: float,
) -> int | None:
    candidates = [
        frame
        for frame in frames
        if frame.start_sec <= end_sec
        and frame.end_sec >= start_sec
        and frame.voiced
        and frame.pitch_midi_rounded is not None
    ]
    if not candidates:
        return None

    weighted_counts: dict[int, float] = {}
    for frame in candidates:
        pitch = int(frame.pitch_midi_rounded)
        weighted_counts[pitch] = weighted_counts.get(pitch, 0.0) + float(frame.pitch_confidence)

    if not weighted_counts:
        return None
    return max(weighted_counts.items(), key=lambda item: item[1])[0]


def _build_error_regions(
    audio_spans: list[tuple[float, float]],
    midi_spans: list[tuple[float, float, str]],
    pitch_errors: list[_RepairErrorRegion],
) -> list[_RepairErrorRegion]:
    regions: list[_RepairErrorRegion] = []
    midi_ranges = [(start, end) for start, end, _ in midi_spans]
    merged_midi = _merged_spans(midi_ranges)

    for audio_start, audio_end in audio_spans:
        overlap = _total_span_coverage(audio_start, audio_end, merged_midi)
        length = max(1e-9, audio_end - audio_start)
        coverage = overlap / length
        if coverage < 0.9:
            regions.append(
                _RepairErrorRegion(
                    region_type="AUDIO_GAP",
                    start_sec=audio_start,
                    end_sec=audio_end,
                    severity=min(1.0, 1.0 - coverage),
                    note_id=None,
                )
            )

    if audio_spans:
        merged_audio = _merged_spans(audio_spans)
        for start_sec, end_sec, note_id in midi_spans:
            length = max(1e-9, end_sec - start_sec)
            overlap = _total_span_coverage(start_sec, end_sec, merged_audio)
            ratio = overlap / length
            if ratio < 0.85:
                regions.append(
                    _RepairErrorRegion(
                        region_type="MIDI_OVERHANG",
                        start_sec=start_sec,
                        end_sec=end_sec,
                        severity=min(1.0, 1.0 - ratio),
                        note_id=note_id,
                    )
                )

    for index in range(len(midi_spans) - 1):
        current = midi_spans[index]
        nxt = midi_spans[index + 1]
        gap = nxt[0] - current[1]
        if 0.0 < gap <= 0.06:
            regions.append(
                _RepairErrorRegion(
                    region_type="MICRO_GAP",
                    start_sec=current[1],
                    end_sec=nxt[0],
                    severity=min(1.0, gap / 0.06),
                    note_id=current[2],
                )
            )

    regions.extend(pitch_errors)
    return regions


def _score_candidate(
    candidate_document: RefinedNoteDocument,
    cleanup_plan: CleanupPlanDocument,
    activity_frames: list[_ActivityFrame],
    activity_params: ActivityRepairParameters,
    pitch_document: BassPitchContourDocument | None,
) -> _CandidateScore:
    selected_ids = _selected_note_ids_from_plan(cleanup_plan)
    midi_spans = _build_candidate_ranges(candidate_document.notes, selected_ids)
    audio_spans = _activity_audio_spans(frames=activity_frames, params=activity_params)

    pitch_matches = 0.0
    pitch_total = 0.0
    pitch_errors: list[_RepairErrorRegion] = []
    if pitch_document is not None:
        for note in candidate_document.notes:
            if note.note_id not in selected_ids and not note.note_id.startswith(
                ("repair_missing_", "repair_split_")
            ):
                continue
            expected_pitch = _extract_pitch_for_window(
                frames=pitch_document.frames,
                start_sec=max(0.0, float(note.refined_start_sec)),
                end_sec=max(float(note.refined_start_sec), float(note.refined_end_sec)),
            )
            if expected_pitch is None:
                continue
            pitch_total += 1.0
            if abs(int(note.pitch_midi) - int(expected_pitch)) <= 1:
                pitch_matches += 1.0
            else:
                pitch_errors.append(
                    _RepairErrorRegion(
                        region_type="PITCH_MISMATCH",
                        start_sec=max(0.0, float(note.refined_start_sec)),
                        end_sec=max(float(note.refined_start_sec), float(note.refined_end_sec)),
                        severity=0.6,
                        note_id=note.note_id,
                    )
                )

    error_regions = _build_error_regions(
        audio_spans=audio_spans,
        midi_spans=midi_spans,
        pitch_errors=pitch_errors,
    )

    merged_audio = _merged_spans(audio_spans)
    merged_midi = _merged_spans([(start, end) for start, end, _ in midi_spans])

    total_audio = sum(max(0.0, end - start) for start, end in merged_audio)
    total_midi = sum(max(0.0, end - start) for start, end in merged_midi)

    overlap_total = 0.0
    for audio_start, audio_end in merged_audio:
        overlap_total += _total_span_coverage(audio_start, audio_end, merged_midi)

    coverage_score = (
        1.0
        if total_audio <= 1e-9
        else max(0.0, min(1.0, overlap_total / total_audio))
    )

    midi_no_audio = max(0.0, total_midi - overlap_total)
    overhang_penalty = 0.0 if total_midi <= 1e-9 else min(1.0, midi_no_audio / total_midi)
    overhang_score = max(0.0, 1.0 - overhang_penalty)

    micro_gap_count = len([region for region in error_regions if region.region_type == "MICRO_GAP"])
    splitish_count = len(
        [
            region
            for region in error_regions
            if region.region_type in {"POSSIBLE_FALSE_SPLIT", "POSSIBLE_MISSING_SPLIT"}
        ]
    )
    continuity_penalty = min(1.0, (micro_gap_count * 0.06) + (splitish_count * 0.04))
    continuity_score = max(0.0, 1.0 - continuity_penalty)

    pitch_consistency_score = 0.9
    if pitch_total > 0.0:
        pitch_consistency_score = max(0.0, min(1.0, pitch_matches / pitch_total))

    total_score = (
        (coverage_score * 0.35)
        + (overhang_score * 0.20)
        + (continuity_score * 0.30)
        + (pitch_consistency_score * 0.15)
    )

    unresolved_error_count = len(
        [
            region
            for region in error_regions
            if region.region_type
            in {
                "AUDIO_GAP",
                "MIDI_OVERHANG",
                "MICRO_GAP",
                "POSSIBLE_FALSE_SPLIT",
                "POSSIBLE_MISSING_SPLIT",
                "PITCH_MISMATCH",
                "REVIEW_MANUAL",
            }
        ]
    )

    audio_gap_count = len([region for region in error_regions if region.region_type == "AUDIO_GAP"])
    midi_overhang_count = len(
        [region for region in error_regions if region.region_type == "MIDI_OVERHANG"]
    )

    return _CandidateScore(
        coverage_score=coverage_score,
        overhang_score=overhang_score,
        continuity_score=continuity_score,
        pitch_consistency_score=pitch_consistency_score,
        total_score=total_score,
        audio_gap_count=audio_gap_count,
        midi_overhang_count=midi_overhang_count,
        unresolved_error_count=unresolved_error_count,
        error_regions=error_regions,
    )


def _profile_to_activity_params(
    profile: str,
    base_params: ActivityRepairParameters,
    conservative_final_pass: bool,
) -> ActivityRepairParameters:
    profile_key = profile.strip().lower()
    if conservative_final_pass:
        profile_key = "conservative"

    baseline = base_params.__dict__.copy()
    if profile_key == "aggressive":
        baseline.update(
            {
                "overhang_min_ms": 160.0,
                "missing_gap_min_ms": 60.0,
                "insert_auto_confidence": max(0.6, base_params.insert_auto_confidence - 0.10),
                "split_auto_confidence": max(0.6, base_params.split_auto_confidence - 0.08),
                "split_pitch_change_semitones": max(
                    0.4,
                    base_params.split_pitch_change_semitones - 0.20,
                ),
            }
        )
    elif profile_key == "sustain_legato":
        baseline.update(
            {
                "overhang_min_ms": max(260.0, base_params.overhang_min_ms),
                "close_gap_ms": max(60.0, base_params.close_gap_ms),
                "insert_auto_confidence": max(0.84, base_params.insert_auto_confidence),
                "split_auto_confidence": max(0.82, base_params.split_auto_confidence),
                "sustain_protect_ratio": max(0.20, base_params.sustain_protect_ratio),
                "legato_neighbor_window_ms": max(260.0, base_params.legato_neighbor_window_ms),
                "split_pitch_change_semitones": max(
                    0.9,
                    base_params.split_pitch_change_semitones,
                ),
            }
        )
    elif profile_key == "conservative":
        baseline.update(
            {
                "overhang_min_ms": max(320.0, base_params.overhang_min_ms),
                "missing_gap_min_ms": max(90.0, base_params.missing_gap_min_ms),
                "close_gap_ms": max(55.0, base_params.close_gap_ms),
                "insert_auto_confidence": max(0.95, base_params.insert_auto_confidence),
                "split_auto_confidence": max(0.95, base_params.split_auto_confidence),
                "split_pitch_change_semitones": max(
                    1.4,
                    base_params.split_pitch_change_semitones,
                ),
            }
        )

    return ActivityRepairParameters(**baseline)


def _note_fingerprint(note: RefinedNoteEvent) -> tuple[int, int]:
    return (
        int(round(float(note.refined_start_sec) * 1000.0)),
        int(round(float(note.refined_end_sec) * 1000.0)),
    )


def _stable_note_ids(
    previous_doc: RefinedNoteDocument,
    current_doc: RefinedNoteDocument,
    tolerance_ms: float,
    current_score: _CandidateScore,
) -> set[str]:
    prev_by_id = {note.note_id: note for note in previous_doc.notes}
    curr_by_id = {note.note_id: note for note in current_doc.notes}
    stable: set[str] = set()

    tolerance = max(0.0, tolerance_ms)
    for note_id, previous_note in prev_by_id.items():
        current_note = curr_by_id.get(note_id)
        if current_note is None:
            continue
        prev_start, prev_end = _note_fingerprint(previous_note)
        curr_start, curr_end = _note_fingerprint(current_note)
        if abs(curr_start - prev_start) <= tolerance and abs(curr_end - prev_end) <= tolerance:
            stable.add(note_id)

    unresolved_types = {
        "AUDIO_GAP",
        "MIDI_OVERHANG",
        "MICRO_GAP",
        "POSSIBLE_FALSE_SPLIT",
        "POSSIBLE_MISSING_SPLIT",
        "PITCH_MISMATCH",
        "REVIEW_MANUAL",
    }
    impacted_notes: set[str] = set()
    for region in current_score.error_regions:
        if region.region_type not in unresolved_types:
            continue
        if region.note_id is not None:
            impacted_notes.add(region.note_id)
            continue

        region_start = max(0.0, float(region.start_sec))
        region_end = max(region_start, float(region.end_sec))
        for note in current_doc.notes:
            note_start = max(0.0, float(note.refined_start_sec))
            note_end = max(note_start, float(note.refined_end_sec))
            if max(note_start, region_start) < min(note_end, region_end):
                impacted_notes.add(note.note_id)

    stable.difference_update(impacted_notes)

    if (
        current_score.coverage_score >= 0.98
        and current_score.overhang_score >= 0.98
        and current_score.unresolved_error_count == 0
    ):
        stable.update(curr_by_id.keys())

    return stable


def _apply_stable_freeze(
    previous_doc: RefinedNoteDocument,
    candidate_doc: RefinedNoteDocument,
    stable_note_ids: set[str],
) -> RefinedNoteDocument:
    if not stable_note_ids:
        return candidate_doc

    previous_by_id = {note.note_id: note for note in previous_doc.notes}
    frozen_notes: list[RefinedNoteEvent] = []
    for note in candidate_doc.notes:
        previous_note = previous_by_id.get(note.note_id)
        if previous_note is None or note.note_id not in stable_note_ids:
            frozen_notes.append(note)
            continue

        actions = list(previous_note.refinement_actions)
        if "ITERATIVE_REPAIR_STABLE" not in actions:
            actions.append("ITERATIVE_REPAIR_STABLE")
        frozen_notes.append(previous_note.model_copy(update={"refinement_actions": actions}))

    return candidate_doc.model_copy(update={"notes": frozen_notes})


def _cap_candidate_changes(
    previous_doc: RefinedNoteDocument,
    candidate_doc: RefinedNoteDocument,
    max_changes: int,
) -> tuple[RefinedNoteDocument, int]:
    if max_changes < 0:
        return candidate_doc, 0

    previous_by_id = {note.note_id: note for note in previous_doc.notes}
    changed_ids: list[str] = []
    for note in candidate_doc.notes:
        previous_note = previous_by_id.get(note.note_id)
        if previous_note is None:
            changed_ids.append(note.note_id)
            continue
        if _note_fingerprint(previous_note) != _note_fingerprint(note):
            changed_ids.append(note.note_id)

    if len(changed_ids) <= max_changes:
        return candidate_doc, 0

    candidate_by_id = {note.note_id: note for note in candidate_doc.notes}
    ranked_changes: list[tuple[float, str]] = []
    for note_id in changed_ids:
        candidate_note = candidate_by_id[note_id]
        previous_note = previous_by_id.get(note_id)
        if previous_note is None:
            note_start = max(0.0, float(candidate_note.refined_start_sec))
            note_end = max(note_start, float(candidate_note.refined_end_sec))
            magnitude = note_end - note_start
        else:
            prev_start, prev_end = _note_fingerprint(previous_note)
            curr_start, curr_end = _note_fingerprint(candidate_note)
            magnitude = (abs(curr_start - prev_start) + abs(curr_end - prev_end)) / 1000.0
        ranked_changes.append((magnitude, note_id))

    allowed_ids = {
        note_id
        for _magnitude, note_id in sorted(
            ranked_changes,
            key=lambda item: (-item[0], item[1]),
        )[:max_changes]
    }
    blocked_count = len(changed_ids) - len(allowed_ids)

    capped_notes: list[RefinedNoteEvent] = []
    for note in candidate_doc.notes:
        previous_note = previous_by_id.get(note.note_id)
        if note.note_id in allowed_ids:
            capped_notes.append(note)
            continue
        if previous_note is not None and note.note_id in changed_ids:
            capped_notes.append(previous_note)
            continue
        if previous_note is None and note.note_id in changed_ids:
            continue
        capped_notes.append(note)

    return candidate_doc.model_copy(update={"notes": capped_notes}), blocked_count


def _restrict_changes_to_error_regions(
    previous_doc: RefinedNoteDocument,
    candidate_doc: RefinedNoteDocument,
    error_regions: list[_RepairErrorRegion],
) -> tuple[RefinedNoteDocument, int]:
    if not error_regions:
        return candidate_doc, 0

    permitted_regions = [
        (region.start_sec, region.end_sec)
        for region in error_regions
        if region.severity >= 0.20
    ]
    if not permitted_regions:
        return candidate_doc, 0

    previous_by_id = {note.note_id: note for note in previous_doc.notes}
    constrained_notes: list[RefinedNoteEvent] = []
    blocked_count = 0

    for note in candidate_doc.notes:
        previous_note = previous_by_id.get(note.note_id)
        note_start = max(0.0, float(note.refined_start_sec))
        note_end = max(note_start, float(note.refined_end_sec))
        overlaps_error = any(
            max(note_start, region_start) < min(note_end, region_end)
            for region_start, region_end in permitted_regions
        )

        if previous_note is None and not overlaps_error:
            blocked_count += 1
            continue

        if previous_note is not None and _note_fingerprint(previous_note) != _note_fingerprint(note):
            if not overlaps_error:
                constrained_notes.append(previous_note)
                blocked_count += 1
                continue

        constrained_notes.append(note)

    return candidate_doc.model_copy(update={"notes": constrained_notes}), blocked_count


def _annotate_iterative_changes(
    previous_doc: RefinedNoteDocument,
    candidate_doc: RefinedNoteDocument,
    iteration_index: int,
) -> RefinedNoteDocument:
    previous_by_id = {note.note_id: note for note in previous_doc.notes}

    annotated: list[RefinedNoteEvent] = []
    for note in candidate_doc.notes:
        previous_note = previous_by_id.get(note.note_id)
        action_set = list(note.refinement_actions)

        changed = False
        if previous_note is None:
            changed = True
            if "ITERATIVE_REPAIR_NEW" not in action_set:
                action_set.append("ITERATIVE_REPAIR_NEW")
        elif _note_fingerprint(previous_note) != _note_fingerprint(note):
            changed = True

        if changed and "ITERATIVE_REPAIR_CHANGED" not in action_set:
            action_set.append("ITERATIVE_REPAIR_CHANGED")
        if changed:
            token = f"ITERATIVE_REPAIR_ITERATION_{iteration_index}"
            if token not in action_set:
                action_set.append(token)

        annotated.append(note.model_copy(update={"refinement_actions": action_set}))

    return candidate_doc.model_copy(update={"notes": annotated})


def _derive_iteration_summary(
    iteration_index: int,
    input_doc: RefinedNoteDocument,
    output_doc: RefinedNoteDocument,
    score: _CandidateScore,
    previous_total_score: float,
    repair_plan: ActivityRepairPlan,
    repair_report: ActivityRepairReport,
    stopped_reason: str | None,
    *,
    accepted: bool,
    rejected_reason: str | None,
    candidate_score: _CandidateScore | None,
    candidate_note_count: int | None,
) -> RepairIterationSummary:
    candidate_action_count = len(
        [action for action in repair_plan.actions if action.action_type != "KEEP"]
    )
    applied_action_count = len(
        [
            action
            for action in repair_plan.actions
            if action.action_type not in {"KEEP", "REVIEW_MANUAL"}
        ]
    )
    if not accepted:
        applied_action_count = 0

    rejected_action_count = 0 if accepted else candidate_action_count

    protected_count = (
        int(repair_report.sustain_protected_count)
        + int(repair_report.pitch_protected_count)
        + int(repair_report.legato_protected_count)
    )

    return RepairIterationSummary(
        iteration_index=iteration_index,
        input_note_count=len(input_doc.notes),
        output_note_count=len(output_doc.notes),
        applied_action_count=applied_action_count,
        candidate_action_count=candidate_action_count,
        rejected_action_count=rejected_action_count,
        extend_count=int(repair_report.extend_count),
        shorten_count=int(repair_report.shorten_count),
        insert_count=int(repair_report.insert_missing_count),
        split_count=int(repair_report.split_count),
        merge_count=0,
        close_gap_count=int(repair_report.close_gap_count),
        protected_count=protected_count,
        review_manual_count=int(repair_report.review_manual_count),
        audio_gap_count=int(score.audio_gap_count),
        midi_overhang_count=int(score.midi_overhang_count),
        unresolved_error_count=int(score.unresolved_error_count),
        coverage_score=float(score.coverage_score),
        overhang_score=float(score.overhang_score),
        continuity_score=float(score.continuity_score),
        pitch_consistency_score=float(score.pitch_consistency_score),
        total_score=float(score.total_score),
        improvement_from_previous=float(score.total_score - previous_total_score),
        accepted=accepted,
        rejected_reason=rejected_reason,
        candidate_score=(None if candidate_score is None else float(candidate_score.total_score)),
        accepted_score=float(score.total_score),
        candidate_note_count=candidate_note_count,
        accepted_note_count=len(output_doc.notes),
        stopped_reason=stopped_reason,
    )


def _profile_for_iteration(params: IterativeRepairParameters, iteration_index: int) -> str:
    if iteration_index <= 1:
        return params.pass1_profile
    if iteration_index == 2:
        return params.pass2_profile
    return params.pass3_profile


def _build_scoring_report(score: _CandidateScore) -> IterationScoringReport:
    return IterationScoringReport(
        total_score=float(score.total_score),
        coverage_score=float(score.coverage_score),
        overhang_score=float(score.overhang_score),
        continuity_score=float(score.continuity_score),
        pitch_consistency_score=float(score.pitch_consistency_score),
        unresolved_error_count=int(score.unresolved_error_count),
        audio_gap_count=int(score.audio_gap_count),
        midi_overhang_count=int(score.midi_overhang_count),
        error_regions=[
            {
                "region_type": region.region_type,
                "start_sec": region.start_sec,
                "end_sec": region.end_sec,
                "severity": region.severity,
                "note_id": region.note_id,
            }
            for region in score.error_regions
        ],
    )


def run_iterative_activity_repair(
    refined_notes_file: Path,
    audio_features_file: Path,
    cleanup_plan_file: Path,
    params: IterativeRepairParameters,
    activity_params: ActivityRepairParameters,
    dsp_features_file: Path | None = None,
    pitch_contour_file: Path | None = None,
) -> tuple[RefinedNoteDocument, IterativeRepairReport, list[IterationArtifacts]]:
    if not refined_notes_file.exists() or not refined_notes_file.is_file():
        raise IterativeRepairError(f"Refined notes file does not exist: {refined_notes_file}")
    if not audio_features_file.exists() or not audio_features_file.is_file():
        raise IterativeRepairError(f"Audio features file does not exist: {audio_features_file}")
    if not cleanup_plan_file.exists() or not cleanup_plan_file.is_file():
        raise IterativeRepairError(f"Cleanup plan file does not exist: {cleanup_plan_file}")
    if dsp_features_file is not None and (not dsp_features_file.exists() or not dsp_features_file.is_file()):
        raise IterativeRepairError(f"DSP features file does not exist: {dsp_features_file}")
    if pitch_contour_file is not None and (
        not pitch_contour_file.exists() or not pitch_contour_file.is_file()
    ):
        raise IterativeRepairError(f"Pitch contour file does not exist: {pitch_contour_file}")

    if params.max_iterations <= 0:
        raise IterativeRepairError("max_iterations must be >= 1")
    if params.max_iterations > 10:
        raise IterativeRepairError("max_iterations exceeds hard safety cap (10)")

    audio_document = _load_audio_document(audio_features_file)
    cleanup_plan = _load_cleanup_plan(cleanup_plan_file)
    initial_document = _load_refined_document(refined_notes_file)
    dsp_document = _load_dsp_document(dsp_features_file) if dsp_features_file is not None else None
    pitch_document = _load_pitch_document(pitch_contour_file) if pitch_contour_file is not None else None

    warnings: list[str] = []
    if initial_document.layer != audio_document.layer:
        warnings.append(
            "Layer mismatch between refined notes and audio features: "
            f"{initial_document.layer} vs {audio_document.layer}."
        )
    if cleanup_plan.layer != initial_document.layer:
        warnings.append(
            "Layer mismatch between cleanup plan and refined notes: "
            f"{cleanup_plan.layer} vs {initial_document.layer}."
        )
    if dsp_document is not None and dsp_document.layer != initial_document.layer:
        warnings.append(
            "Layer mismatch between DSP features and refined notes: "
            f"{dsp_document.layer} vs {initial_document.layer}."
        )
    if pitch_document is not None and pitch_document.layer != initial_document.layer:
        warnings.append(
            "Layer mismatch between pitch contour and refined notes: "
            f"{pitch_document.layer} vs {initial_document.layer}."
        )

    activity_frames = _build_activity_frames(
        audio_document=audio_document,
        dsp_document=dsp_document,
        params=activity_params,
    )

    initial_score = _score_candidate(
        candidate_document=initial_document,
        cleanup_plan=cleanup_plan,
        activity_frames=activity_frames,
        activity_params=activity_params,
        pitch_document=pitch_document,
    )

    current_document = initial_document

    best_document = initial_document
    best_score = initial_score
    best_iteration_index = 0

    artifacts: list[IterationArtifacts] = []
    stable_note_ids: set[str] = set()
    convergence_reached = False
    stop_reason: str | None = None
    accepted_any = False
    rejected_count = 0
    all_candidate_actions_zero = True

    for iteration_index in range(1, params.max_iterations + 1):
        profile = _profile_for_iteration(params, iteration_index)
        is_final_iteration = iteration_index == params.max_iterations
        iteration_activity_params = _profile_to_activity_params(
            profile=profile,
            base_params=activity_params,
            conservative_final_pass=(params.conservative_final_pass and is_final_iteration),
        )

        with tempfile.TemporaryDirectory(prefix="midi_cleaner_iterative_") as temp_dir:
            iteration_input = Path(temp_dir) / "refined_iter_input.json"
            _write_refined_document(iteration_input, current_document)

            try:
                repaired_document, repair_plan, repair_report = repair_activity(
                    refined_notes_file=iteration_input,
                    audio_features_file=audio_features_file,
                    cleanup_plan_file=cleanup_plan_file,
                    params=iteration_activity_params,
                    dsp_features_file=dsp_features_file,
                    pitch_contour_file=pitch_contour_file,
                )
            except ActivityRepairError as exc:
                raise IterativeRepairError(str(exc)) from exc

        current_iteration_score = _score_candidate(
            candidate_document=current_document,
            cleanup_plan=cleanup_plan,
            activity_frames=activity_frames,
            activity_params=iteration_activity_params,
            pitch_document=pitch_document,
        )

        candidate_document = _annotate_iterative_changes(
            previous_doc=current_document,
            candidate_doc=repaired_document,
            iteration_index=iteration_index,
        )

        if params.protect_previous_good_regions and iteration_index > 1:
            candidate_document, blocked_outside_errors = _restrict_changes_to_error_regions(
                previous_doc=current_document,
                candidate_doc=candidate_document,
                error_regions=current_iteration_score.error_regions,
            )
            if blocked_outside_errors > 0:
                warnings.append(
                    "changes restricted to error regions: "
                    f"iteration={iteration_index}, blocked_changes={blocked_outside_errors}"
                )

        if params.max_actions_per_iteration is not None:
            candidate_document, blocked_count = _cap_candidate_changes(
                previous_doc=current_document,
                candidate_doc=candidate_document,
                max_changes=int(params.max_actions_per_iteration),
            )
            if blocked_count > 0:
                warnings.append(
                    "iteration action cap applied: "
                    f"iteration={iteration_index}, blocked_changes={blocked_count}"
                )

        if params.freeze_stable_notes and stable_note_ids:
            candidate_document = _apply_stable_freeze(
                previous_doc=current_document,
                candidate_doc=candidate_document,
                stable_note_ids=stable_note_ids,
            )

        candidate_score = _score_candidate(
            candidate_document=candidate_document,
            cleanup_plan=cleanup_plan,
            activity_frames=activity_frames,
            activity_params=iteration_activity_params,
            pitch_document=pitch_document,
        )

        improvement = candidate_score.total_score - current_iteration_score.total_score

        candidate_action_count = len(
            [action for action in repair_plan.actions if action.action_type != "KEEP"]
        )
        if candidate_action_count > 0:
            all_candidate_actions_zero = False

        reject_reason: str | None = None
        should_reject = False
        if improvement < 0.0 and not params.allow_regression:
            should_reject = True
            reject_reason = "regression_rejected"
        elif improvement == 0.0:
            should_reject = True
            reject_reason = "no_improvement"

        if should_reject:
            rejected_count += 1
            summary = _derive_iteration_summary(
                iteration_index=iteration_index,
                input_doc=current_document,
                output_doc=current_document,
                score=current_iteration_score,
                previous_total_score=current_iteration_score.total_score,
                repair_plan=repair_plan,
                repair_report=repair_report,
                stopped_reason=reject_reason,
                accepted=False,
                rejected_reason=reject_reason,
                candidate_score=candidate_score,
                candidate_note_count=len(candidate_document.notes),
            )
            artifacts.append(
                IterationArtifacts(
                    iteration_index=iteration_index,
                    repaired_document=current_document,
                    repair_plan=repair_plan,
                    repair_report=repair_report,
                    score_report=_build_scoring_report(current_iteration_score),
                    summary=summary,
                )
            )
            continue

        summary = _derive_iteration_summary(
            iteration_index=iteration_index,
            input_doc=current_document,
            output_doc=candidate_document,
            score=candidate_score,
            previous_total_score=current_iteration_score.total_score,
            repair_plan=repair_plan,
            repair_report=repair_report,
            stopped_reason=None,
            accepted=True,
            rejected_reason=None,
            candidate_score=candidate_score,
            candidate_note_count=len(candidate_document.notes),
        )

        artifacts.append(
            IterationArtifacts(
                iteration_index=iteration_index,
                repaired_document=candidate_document,
                repair_plan=repair_plan,
                repair_report=repair_report,
                score_report=_build_scoring_report(candidate_score),
                summary=summary,
            )
        )

        accepted_any = True
        previous_document = current_document
        current_document = candidate_document

        candidate_baseline_score = _score_candidate(
            candidate_document=candidate_document,
            cleanup_plan=cleanup_plan,
            activity_frames=activity_frames,
            activity_params=activity_params,
            pitch_document=pitch_document,
        )

        if candidate_baseline_score.total_score > best_score.total_score:
            best_document = candidate_document
            best_score = candidate_baseline_score
            best_iteration_index = iteration_index

        if params.freeze_stable_notes and params.protect_previous_good_regions:
            stable_note_ids = _stable_note_ids(
                previous_doc=previous_document,
                current_doc=current_document,
                tolerance_ms=params.stable_note_tolerance_ms,
                current_score=candidate_score,
            )

        if improvement < params.min_improvement:
            stop_reason = "min_improvement_not_met"
            convergence_reached = True
            break

        if candidate_score.unresolved_error_count == 0:
            stop_reason = "no_unresolved_errors"
            convergence_reached = True
            break

    if stop_reason is None:
        if not accepted_any and rejected_count == len(artifacts):
            stop_reason = "no_actions_available" if all_candidate_actions_zero else "all_candidates_rejected"
        else:
            stop_reason = "max_iterations_reached"

    if artifacts:
        last = artifacts[-1]
        artifacts[-1] = IterationArtifacts(
            iteration_index=last.iteration_index,
            repaired_document=last.repaired_document,
            repair_plan=last.repair_plan,
            repair_report=last.repair_report,
            score_report=last.score_report,
            summary=last.summary.model_copy(update={"stopped_reason": stop_reason}),
        )

    if best_iteration_index == 0:
        best_document = initial_document
        best_score = initial_score

    final_score = float(best_score.total_score)
    total_improvement = float(max(0.0, final_score - initial_score.total_score))

    report = IterativeRepairReport(
        status="ok",
        layer=initial_document.layer,
        input_refined_notes_file=str(refined_notes_file),
        final_repaired_notes_file=None,
        iterations_requested=int(params.max_iterations),
        iterations_completed=len(artifacts),
        convergence_reached=bool(convergence_reached),
        best_iteration_index=int(best_iteration_index),
        final_score=final_score,
        initial_score=float(initial_score.total_score),
        total_improvement=total_improvement,
        warning_count=len(warnings),
        warnings=warnings,
        iterations=[artifact.summary for artifact in artifacts],
        output_file=None,
    )

    return best_document, report, artifacts
