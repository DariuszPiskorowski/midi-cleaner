from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

from midi_cleaner.alignment.models import AudioAlignedNoteDocument, AudioAlignedNoteEvent
from midi_cleaner.audio.models import AudioFeatureDocument, AudioFrameFeature
from midi_cleaner.dsp.models import DspAudioFeatureDocument, DspAudioFrame
from midi_cleaner.refinement.models import BassRefinementReport, RefinedNoteDocument, RefinedNoteEvent
from midi_cleaner.validation.models import NoteValidationDocument

SCHEMA_VERSION = "0.1.0"


class BassRefinementError(Exception):
    """Raised when bass refinement cannot be completed."""


@dataclass(frozen=True)
class BassRefinementParameters:
    attack_lookback_ms: float = 80.0
    max_attack_advance_ms: float = 80.0
    attack_rms_ratio: float = 0.25
    min_attack_rise: float = 0.0005

    merge_gap_ms: float = 160.0
    minimum_silence_ms: float = 80.0
    silence_rms_ratio: float = 0.18
    same_pitch_tolerance_semitones: int = 1
    max_merge_window_ms: float = 600.0

    tail_rms_ratio: float = 0.20
    tail_silence_hold_ms: float = 120.0
    max_tail_extension_ms: float = 900.0
    protect_next_onset_ms: float = 80.0

    minimum_note_duration_ms: float = 80.0
    gap_close_ms: float = 30.0

    monophonic: bool = True
    allow_pitch_overlap: bool = False


@dataclass
class _MutableRefinedNote:
    event: AudioAlignedNoteEvent
    refined_start_sec: float
    refined_end_sec: float
    merged_note_ids: list[str]
    refinement_actions: set[str]
    reasons: list[str]
    validation_action: str
    confidence: float

    def add_action(self, action: str) -> None:
        self.refinement_actions.add(action)

    @property
    def pitch_midi(self) -> int:
        return int(self.event.pitch_midi)

    @property
    def note_id(self) -> str:
        return self.event.note_id


def _load_aligned_document(path: Path) -> AudioAlignedNoteDocument:
    try:
        return AudioAlignedNoteDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise BassRefinementError(f"Invalid aligned notes JSON: {path}") from exc


def _load_audio_document(path: Path) -> AudioFeatureDocument:
    try:
        return AudioFeatureDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise BassRefinementError(f"Invalid audio features JSON: {path}") from exc


def _load_validation_document(path: Path) -> NoteValidationDocument:
    try:
        return NoteValidationDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise BassRefinementError(f"Invalid validation JSON: {path}") from exc


def _load_dsp_document(path: Path) -> DspAudioFeatureDocument:
    try:
        return DspAudioFeatureDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise BassRefinementError(f"Invalid DSP features JSON: {path}") from exc


def _nearest_frame(frames: list[AudioFrameFeature], anchor_sec: float) -> AudioFrameFeature | None:
    if not frames:
        return None
    return min(frames, key=lambda frame: abs(frame.start_sec - anchor_sec))


def _frame_span_ms(frame: AudioFrameFeature) -> float:
    return max(1e-6, (frame.end_sec - frame.start_sec) * 1000.0)


def _median_frame_span_ms(frames: list[AudioFrameFeature]) -> float:
    if not frames:
        return 10.0
    spans = [_frame_span_ms(frame) for frame in frames]
    return max(1.0, float(median(spans)))


def _window_frames(frames: list[AudioFrameFeature], low_sec: float, high_sec: float) -> list[AudioFrameFeature]:
    return [frame for frame in frames if frame.end_sec > low_sec and frame.start_sec < high_sec]


def _window_dsp_frames(
    frames: list[DspAudioFrame],
    low_sec: float,
    high_sec: float,
) -> list[DspAudioFrame]:
    return [frame for frame in frames if frame.end_sec > low_sec and frame.start_sec < high_sec]


def _rms_threshold_for_anchor(frames: list[AudioFrameFeature], anchor_sec: float, ratio: float) -> float:
    anchor = _nearest_frame(frames, anchor_sec)
    anchor_rms = float(anchor.rms) if anchor is not None else 0.0
    return max(0.0, anchor_rms * ratio)


def _max_local_onset(frames: list[AudioFrameFeature], start_sec: float, end_sec: float) -> float:
    candidates = _window_frames(frames, start_sec, end_sec)
    if not candidates:
        return 0.0
    return max(float(item.onset_score) for item in candidates)


def _is_silence_between(
    frames: list[AudioFrameFeature],
    start_sec: float,
    end_sec: float,
    silence_threshold: float,
    minimum_silence_ms: float,
) -> bool:
    if end_sec <= start_sec:
        return False

    interval = _window_frames(frames, start_sec, end_sec)
    if not interval:
        return True

    run_start = None
    for frame in interval:
        if float(frame.rms) <= silence_threshold:
            if run_start is None:
                run_start = frame.start_sec
            run_end = frame.end_sec
            if (run_end - run_start) * 1000.0 >= minimum_silence_ms:
                return True
        else:
            run_start = None
    return False


def _is_dsp_silence_between(
    frames: list[DspAudioFrame],
    start_sec: float,
    end_sec: float,
    minimum_silence_ms: float,
) -> bool:
    if end_sec <= start_sec:
        return False

    interval = _window_dsp_frames(frames, start_sec, end_sec)
    if not interval:
        return True

    run_start = None
    for frame in interval:
        if frame.is_silence:
            if run_start is None:
                run_start = float(frame.start_sec)
            run_end = float(frame.end_sec)
            if (run_end - run_start) * 1000.0 >= minimum_silence_ms:
                return True
        else:
            run_start = None
    return False


def _find_attack_start(
    note: _MutableRefinedNote,
    frames: list[AudioFrameFeature],
    params: BassRefinementParameters,
    audio_duration_sec: float,
    dsp_frames: list[DspAudioFrame] | None = None,
) -> float:
    _ = audio_duration_sec
    lookback_sec = max(0.0, params.attack_lookback_ms / 1000.0)
    max_advance_sec = max(0.0, params.max_attack_advance_ms / 1000.0)

    current_start = max(0.0, note.refined_start_sec)

    if dsp_frames:
        low = max(0.0, current_start - lookback_sec)
        dsp_candidates = sorted(
            [
                frame
                for frame in _window_dsp_frames(dsp_frames, low, current_start + 1e-6)
                if frame.start_sec <= current_start and frame.is_attack_rise
            ],
            key=lambda frame: frame.start_sec,
        )
        if dsp_candidates:
            dsp_start = max(current_start - max_advance_sec, float(dsp_candidates[0].start_sec))
            if dsp_start < current_start:
                note.add_action("ATTACK_START_ADJUSTED")
                note.reasons.append("moved start earlier to DSP attack evidence")
                return dsp_start

    low = max(0.0, current_start - lookback_sec)
    candidates = [
        frame
        for frame in _window_frames(frames, low, current_start + 1e-6)
        if frame.start_sec <= current_start
    ]
    if not candidates:
        return current_start

    candidates = sorted(candidates, key=lambda frame: frame.start_sec)
    attack_threshold = _rms_threshold_for_anchor(frames, current_start, params.attack_rms_ratio)

    onset_idx = len(candidates) - 1
    for idx in range(len(candidates) - 1, -1, -1):
        frame = candidates[idx]
        if float(frame.rms) > attack_threshold:
            onset_idx = idx
            break

    earliest_idx = onset_idx
    for idx in range(onset_idx - 1, -1, -1):
        current = float(candidates[idx].rms)
        nxt = float(candidates[idx + 1].rms)
        if nxt - current >= params.min_attack_rise:
            earliest_idx = idx
        else:
            break

    candidate_start = max(0.0, float(candidates[earliest_idx].start_sec))
    candidate_start = max(current_start - max_advance_sec, candidate_start)
    if candidate_start >= current_start:
        return current_start

    has_rise_evidence = False
    for idx in range(earliest_idx, onset_idx):
        current = float(candidates[idx].rms)
        nxt = float(candidates[idx + 1].rms)
        if (nxt - current) >= params.min_attack_rise and nxt > attack_threshold:
            has_rise_evidence = True
            break

    if not has_rise_evidence:
        return current_start

    note.add_action("ATTACK_START_ADJUSTED")
    note.reasons.append("moved start earlier to audible attack rise")
    return candidate_start


def _merge_false_retriggers(
    notes: list[_MutableRefinedNote],
    frames: list[AudioFrameFeature],
    params: BassRefinementParameters,
    dsp_frames: list[DspAudioFrame] | None = None,
) -> tuple[list[_MutableRefinedNote], int]:
    if not notes:
        return [], 0

    merged: list[_MutableRefinedNote] = [notes[0]]
    merge_count = 0

    for note in notes[1:]:
        previous = merged[-1]

        pitch_distance = abs(previous.pitch_midi - note.pitch_midi)
        near_pitch = pitch_distance <= int(params.same_pitch_tolerance_semitones)

        gap_sec = note.refined_start_sec - previous.refined_end_sec
        merge_gap_sec = params.merge_gap_ms / 1000.0
        if gap_sec < 0:
            gap_sec = 0.0
        in_gap_window = gap_sec <= merge_gap_sec

        merge_window_ok = (
            (note.refined_end_sec - previous.refined_start_sec) * 1000.0
            <= params.max_merge_window_ms
        )

        silence_threshold = _rms_threshold_for_anchor(
            frames,
            previous.refined_end_sec,
            params.silence_rms_ratio,
        )
        has_real_silence = _is_silence_between(
            frames=frames,
            start_sec=previous.refined_end_sec,
            end_sec=note.refined_start_sec,
            silence_threshold=silence_threshold,
            minimum_silence_ms=params.minimum_silence_ms,
        )

        if dsp_frames is not None:
            has_dsp_silence = _is_dsp_silence_between(
                frames=dsp_frames,
                start_sec=previous.refined_end_sec,
                end_sec=note.refined_start_sec,
                minimum_silence_ms=params.minimum_silence_ms,
            )
            has_dsp_sustain = any(
                (frame.is_sustain or frame.is_tail) and (not frame.is_silence)
                for frame in _window_dsp_frames(
                    dsp_frames,
                    previous.refined_end_sec,
                    note.refined_start_sec,
                )
            )
            if has_dsp_silence:
                has_real_silence = True
            elif has_dsp_sustain:
                has_real_silence = False

        if near_pitch and in_gap_window and merge_window_ok and not has_real_silence:
            previous.refined_end_sec = max(previous.refined_end_sec, note.refined_end_sec)
            previous.merged_note_ids.extend([note.note_id, *note.merged_note_ids])
            previous.add_action("FALSE_RETRIGGER_MERGED")
            previous.reasons.append(
                f"merged near-pitch retrigger {note.note_id} without real silence"
            )
            if note.validation_action == "KEEP" or previous.validation_action == "KEEP":
                previous.validation_action = "KEEP"
            elif note.validation_action == "REVIEW" or previous.validation_action == "REVIEW":
                previous.validation_action = "REVIEW"
            previous.confidence = max(previous.confidence, note.confidence)
            merge_count += 1
            continue

        merged.append(note)

    return merged, merge_count


def _find_tail_end(
    note: _MutableRefinedNote,
    next_note_start_sec: float | None,
    frames: list[AudioFrameFeature],
    params: BassRefinementParameters,
    audio_duration_sec: float,
    dsp_frames: list[DspAudioFrame] | None = None,
) -> float:
    current_end = note.refined_end_sec
    max_tail_end = min(audio_duration_sec, current_end + (params.max_tail_extension_ms / 1000.0))
    if max_tail_end <= current_end:
        return current_end

    protect_onset_sec = params.protect_next_onset_ms / 1000.0
    if next_note_start_sec is not None:
        max_tail_end = min(max_tail_end, max(0.0, next_note_start_sec - protect_onset_sec))
        if max_tail_end <= current_end:
            return current_end

    threshold = _rms_threshold_for_anchor(frames, current_end, params.tail_rms_ratio)
    tail_frames = sorted(_window_frames(frames, current_end, max_tail_end), key=lambda item: item.start_sec)
    if not tail_frames:
        return current_end

    hold_sec = params.tail_silence_hold_ms / 1000.0
    silence_run_start = None
    candidate_end = current_end

    for frame in tail_frames:
        frame_rms = float(frame.rms)
        frame_start = float(frame.start_sec)
        frame_end = float(frame.end_sec)

        if frame_rms > threshold:
            candidate_end = max(candidate_end, frame_end)
            silence_run_start = None
            continue

        if silence_run_start is None:
            silence_run_start = frame_start

        silence_len = max(0.0, frame_end - silence_run_start)
        if silence_len >= hold_sec:
            break

    if dsp_frames:
        dsp_tail_frames = sorted(
            _window_dsp_frames(dsp_frames, current_end, max_tail_end),
            key=lambda item: item.start_sec,
        )
        dsp_silence_run_start = None
        dsp_candidate_end = current_end
        for frame in dsp_tail_frames:
            frame_start = float(frame.start_sec)
            frame_end = float(frame.end_sec)
            has_tail_energy = (frame.is_sustain or frame.is_tail) and (not frame.is_silence)
            if has_tail_energy:
                dsp_candidate_end = max(dsp_candidate_end, frame_end)
                dsp_silence_run_start = None
                continue

            if dsp_silence_run_start is None:
                dsp_silence_run_start = frame_start
            silence_len = max(0.0, frame_end - dsp_silence_run_start)
            if silence_len >= hold_sec:
                break

        candidate_end = max(candidate_end, min(dsp_candidate_end, max_tail_end))

    candidate_end = min(candidate_end, max_tail_end)
    if candidate_end <= current_end:
        return current_end

    note.add_action("SUSTAIN_TAIL_EXTENDED")
    note.reasons.append("extended note through audible sustain tail")
    return candidate_end


def _ensure_minimum_duration(
    note: _MutableRefinedNote,
    next_note_start_sec: float | None,
    frames: list[AudioFrameFeature],
    params: BassRefinementParameters,
    audio_duration_sec: float,
    dsp_frames: list[DspAudioFrame] | None = None,
) -> None:
    min_duration_sec = params.minimum_note_duration_ms / 1000.0
    duration = max(0.0, note.refined_end_sec - note.refined_start_sec)
    if duration >= min_duration_sec:
        return

    local_frames = _window_frames(frames, note.refined_start_sec, note.refined_end_sec + min_duration_sec)
    has_energy = any(float(frame.rms) > 0.0 for frame in local_frames)
    if not has_energy and dsp_frames is not None:
        has_energy = any(
            not frame.is_silence
            for frame in _window_dsp_frames(
                dsp_frames,
                note.refined_start_sec,
                note.refined_end_sec + min_duration_sec,
            )
        )
    if not has_energy:
        return

    target_end = note.refined_start_sec + min_duration_sec
    if next_note_start_sec is not None:
        target_end = min(target_end, next_note_start_sec)
    target_end = min(target_end, audio_duration_sec)

    if target_end > note.refined_end_sec:
        note.refined_end_sec = target_end
        note.add_action("SHORT_NOTE_EXTENDED")
        note.reasons.append("extended short note to minimum musical duration")


def _close_tiny_gaps(notes: list[_MutableRefinedNote], params: BassRefinementParameters) -> None:
    if len(notes) < 2:
        return

    max_gap_sec = params.gap_close_ms / 1000.0
    for idx in range(len(notes) - 1):
        current = notes[idx]
        nxt = notes[idx + 1]
        if abs(current.pitch_midi - nxt.pitch_midi) > int(params.same_pitch_tolerance_semitones):
            continue

        gap = nxt.refined_start_sec - current.refined_end_sec
        if 0.0 < gap <= max_gap_sec:
            current.refined_end_sec = nxt.refined_start_sec
            current.add_action("GAP_CLOSED")
            current.reasons.append("closed tiny same-pitch gap")


def _resolve_monophonic_overlaps(
    notes: list[_MutableRefinedNote],
    params: BassRefinementParameters,
) -> int:
    if not params.monophonic:
        return 0
    if len(notes) < 2:
        return 0

    resolved = 0
    for idx in range(len(notes) - 1):
        current = notes[idx]
        nxt = notes[idx + 1]
        if current.refined_end_sec <= nxt.refined_start_sec:
            continue

        near_pitch = abs(current.pitch_midi - nxt.pitch_midi) <= int(params.same_pitch_tolerance_semitones)
        if params.allow_pitch_overlap and not near_pitch:
            continue

        current.refined_end_sec = nxt.refined_start_sec
        current.add_action("MONOPHONIC_OVERLAP_RESOLVED")
        current.reasons.append("trimmed overlap for monophonic bass lane")
        resolved += 1

    return resolved


def _to_refined_event(note: _MutableRefinedNote) -> RefinedNoteEvent:
    refined_start_sec = max(0.0, float(note.refined_start_sec))
    refined_end_sec = max(refined_start_sec, float(note.refined_end_sec))
    refined_duration_sec = max(0.0, refined_end_sec - refined_start_sec)

    actions = sorted(note.refinement_actions)
    if not actions:
        actions = ["UNCHANGED"]

    return RefinedNoteEvent(
        note_id=note.note_id,
        source=note.event.source,
        layer=note.event.layer,
        pitch_midi=note.event.pitch_midi,
        pitch_name=note.event.pitch_name,
        velocity=note.event.velocity,
        channel=note.event.channel,
        original_start_sec=float(note.event.original_start_sec),
        original_end_sec=float(note.event.original_end_sec),
        aligned_start_sec=float(note.event.aligned_start_sec),
        aligned_end_sec=float(note.event.aligned_end_sec),
        refined_start_sec=refined_start_sec,
        refined_end_sec=refined_end_sec,
        refined_duration_sec=refined_duration_sec,
        start_refinement_ms=(refined_start_sec - float(note.event.aligned_start_sec)) * 1000.0,
        end_refinement_ms=(refined_end_sec - float(note.event.aligned_end_sec)) * 1000.0,
        merged_note_ids=note.merged_note_ids,
        refinement_actions=actions,
        refinement_confidence=max(0.0, min(1.0, float(note.confidence))),
        reasons=note.reasons,
    )


def refine_bass_notes(
    aligned_notes_file: Path,
    audio_features_file: Path,
    validation_file: Path,
    params: BassRefinementParameters,
    dsp_features_file: Path | None = None,
) -> tuple[RefinedNoteDocument, BassRefinementReport]:
    if not aligned_notes_file.exists() or not aligned_notes_file.is_file():
        raise BassRefinementError(f"Aligned notes file does not exist: {aligned_notes_file}")
    if not audio_features_file.exists() or not audio_features_file.is_file():
        raise BassRefinementError(f"Audio features file does not exist: {audio_features_file}")
    if not validation_file.exists() or not validation_file.is_file():
        raise BassRefinementError(f"Validation file does not exist: {validation_file}")
    if dsp_features_file is not None and (not dsp_features_file.exists() or not dsp_features_file.is_file()):
        raise BassRefinementError(f"DSP features file does not exist: {dsp_features_file}")

    aligned_document = _load_aligned_document(aligned_notes_file)
    audio_document = _load_audio_document(audio_features_file)
    validation_document = _load_validation_document(validation_file)
    dsp_document = _load_dsp_document(dsp_features_file) if dsp_features_file is not None else None

    warnings: list[str] = []
    if aligned_document.layer != audio_document.layer:
        warnings.append(
            "Layer mismatch between aligned notes and audio features: "
            f"{aligned_document.layer} vs {audio_document.layer}."
        )
    if validation_document.layer != aligned_document.layer:
        warnings.append(
            "Layer mismatch between validation and aligned notes: "
            f"{validation_document.layer} vs {aligned_document.layer}."
        )
    if dsp_document is not None and dsp_document.layer != aligned_document.layer:
        warnings.append(
            "Layer mismatch between DSP features and aligned notes: "
            f"{dsp_document.layer} vs {aligned_document.layer}."
        )

    validation_by_note_id = {item.note_id: item for item in validation_document.validations}
    dsp_frames = dsp_document.frames if dsp_document is not None else None

    mutable_notes: list[_MutableRefinedNote] = []
    for note in aligned_document.notes:
        validation = validation_by_note_id.get(note.note_id)
        if validation is None:
            warnings.append(f"Validation missing for aligned note_id: {note.note_id}; defaulting action REVIEW")
            validation_action = "REVIEW"
            confidence = 0.5
        else:
            validation_action = str(validation.recommended_action)
            confidence = float(validation.confidence)

        mutable_notes.append(
            _MutableRefinedNote(
                event=note,
                refined_start_sec=max(0.0, float(note.aligned_start_sec)),
                refined_end_sec=max(float(note.aligned_start_sec), float(note.aligned_end_sec)),
                merged_note_ids=[],
                refinement_actions=set(),
                reasons=[],
                validation_action=validation_action,
                confidence=confidence,
            )
        )

    mutable_notes.sort(key=lambda item: (item.refined_start_sec, item.refined_end_sec, item.note_id))

    for note in mutable_notes:
        note.refined_start_sec = _find_attack_start(
            note=note,
            frames=audio_document.frames,
            params=params,
            audio_duration_sec=float(audio_document.duration_sec),
            dsp_frames=dsp_frames,
        )

    mutable_notes, false_merge_count = _merge_false_retriggers(
        notes=mutable_notes,
        frames=audio_document.frames,
        params=params,
        dsp_frames=dsp_frames,
    )

    mutable_notes.sort(key=lambda item: (item.refined_start_sec, item.refined_end_sec, item.note_id))

    tail_extended_count = 0
    short_note_extended_count = 0
    minimum_duration_sec = params.minimum_note_duration_ms / 1000.0
    for idx, note in enumerate(mutable_notes):
        next_note_start_sec = None
        if idx + 1 < len(mutable_notes):
            next_note_start_sec = mutable_notes[idx + 1].refined_start_sec

        initial_duration = note.refined_end_sec - note.refined_start_sec
        before_end = note.refined_end_sec
        note.refined_end_sec = _find_tail_end(
            note=note,
            next_note_start_sec=next_note_start_sec,
            frames=audio_document.frames,
            params=params,
            audio_duration_sec=float(audio_document.duration_sec),
            dsp_frames=dsp_frames,
        )
        if note.refined_end_sec > before_end:
            tail_extended_count += 1

        before_duration = note.refined_end_sec - note.refined_start_sec
        _ensure_minimum_duration(
            note=note,
            next_note_start_sec=next_note_start_sec,
            frames=audio_document.frames,
            params=params,
            audio_duration_sec=float(audio_document.duration_sec),
            dsp_frames=dsp_frames,
        )
        after_duration = note.refined_end_sec - note.refined_start_sec
        if initial_duration < minimum_duration_sec and after_duration >= minimum_duration_sec:
            if "SHORT_NOTE_EXTENDED" not in note.refinement_actions:
                note.add_action("SHORT_NOTE_EXTENDED")
                note.reasons.append("extended short note to minimum musical duration")
            short_note_extended_count += 1
        elif after_duration > before_duration and "SHORT_NOTE_EXTENDED" in note.refinement_actions:
            short_note_extended_count += 1

    _close_tiny_gaps(mutable_notes, params)
    overlap_resolved_count = _resolve_monophonic_overlaps(mutable_notes, params)

    refined_events = [_to_refined_event(note) for note in mutable_notes]

    start_shifts = [event.start_refinement_ms for event in refined_events]
    end_shifts = [event.end_refinement_ms for event in refined_events]
    median_start_refinement_ms = float(median(start_shifts)) if start_shifts else None
    median_end_refinement_ms = float(median(end_shifts)) if end_shifts else None

    max_tail_extension_ms = 0.0
    for event in refined_events:
        tail_extension_ms = max(0.0, event.end_refinement_ms)
        if tail_extension_ms > max_tail_extension_ms:
            max_tail_extension_ms = tail_extension_ms

    document = RefinedNoteDocument(
        schema_version=SCHEMA_VERSION,
        aligned_notes_file=str(aligned_notes_file),
        audio_features_file=str(audio_features_file),
        validation_file=str(validation_file),
        layer=aligned_document.layer,
        sample_rate=audio_document.sample_rate,
        audio_duration_sec=audio_document.duration_sec,
        timing_source="refined_audio_seconds",
        refinement_parameters={
            "attack_lookback_ms": params.attack_lookback_ms,
            "max_attack_advance_ms": params.max_attack_advance_ms,
            "attack_rms_ratio": params.attack_rms_ratio,
            "min_attack_rise": params.min_attack_rise,
            "merge_gap_ms": params.merge_gap_ms,
            "minimum_silence_ms": params.minimum_silence_ms,
            "silence_rms_ratio": params.silence_rms_ratio,
            "same_pitch_tolerance_semitones": float(params.same_pitch_tolerance_semitones),
            "max_merge_window_ms": params.max_merge_window_ms,
            "tail_rms_ratio": params.tail_rms_ratio,
            "tail_silence_hold_ms": params.tail_silence_hold_ms,
            "max_tail_extension_ms": params.max_tail_extension_ms,
            "protect_next_onset_ms": params.protect_next_onset_ms,
            "minimum_note_duration_ms": params.minimum_note_duration_ms,
            "gap_close_ms": params.gap_close_ms,
            "monophonic": params.monophonic,
            "allow_pitch_overlap": params.allow_pitch_overlap,
            "uses_dsp_features": dsp_document is not None,
        },
        notes=refined_events,
    )

    report = BassRefinementReport(
        aligned_notes_file=str(aligned_notes_file),
        audio_features_file=str(audio_features_file),
        validation_file=str(validation_file),
        status="ok",
        layer=aligned_document.layer,
        input_note_count=len(aligned_document.notes),
        output_note_count=len(refined_events),
        merged_count=false_merge_count,
        false_retrigger_merge_count=false_merge_count,
        tail_extended_count=tail_extended_count,
        short_note_extended_count=short_note_extended_count,
        overlap_resolved_count=overlap_resolved_count,
        median_start_refinement_ms=median_start_refinement_ms,
        median_end_refinement_ms=median_end_refinement_ms,
        max_tail_extension_ms=float(max_tail_extension_ms),
        warning_count=len(warnings),
        warnings=warnings,
        output_file=None,
    )

    return document, report