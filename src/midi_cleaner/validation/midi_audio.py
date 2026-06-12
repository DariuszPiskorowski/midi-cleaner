from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from midi_cleaner.alignment.models import AudioAlignedNoteDocument
from midi_cleaner.audio.models import AudioFeatureDocument, AudioFrameFeature
from midi_cleaner.midi.models import NoteEvent, NoteEventDocument
from midi_cleaner.validation.models import (
    MidiAudioValidationReport,
    NoteValidation,
    NoteValidationDocument,
)

SCHEMA_VERSION = "0.1.0"


class MidiAudioValidationError(Exception):
    """Raised when MIDI-vs-audio validation cannot be completed."""


@dataclass(frozen=True)
class ValidationParameters:
    onset_window_ms: float = 50.0
    minimum_rms: float = 0.001
    minimum_onset_score: float = 0.01
    review_threshold: float = 0.45
    keep_threshold: float = 0.70


@dataclass(frozen=True)
class _ValidationNoteContext:
    note: NoteEvent
    start_sec: float
    end_sec: float
    duration_sec: float


def _load_note_document(path: Path) -> NoteEventDocument:
    try:
        return NoteEventDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise MidiAudioValidationError(f"Invalid notes JSON: {path}") from exc


def _load_audio_document(path: Path) -> AudioFeatureDocument:
    try:
        return AudioFeatureDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise MidiAudioValidationError(f"Invalid audio features JSON: {path}") from exc


def _load_audio_aligned_document(path: Path) -> AudioAlignedNoteDocument:
    try:
        return AudioAlignedNoteDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise MidiAudioValidationError(f"Invalid audio aligned notes JSON: {path}") from exc


def _overlapping_frames(
    note_context: _ValidationNoteContext,
    frames: list[AudioFrameFeature],
) -> list[AudioFrameFeature]:
    return [
        frame
        for frame in frames
        if frame.end_sec > note_context.start_sec and frame.start_sec < note_context.end_sec
    ]


def _select_onset_frame(
    note_context: _ValidationNoteContext,
    frames: list[AudioFrameFeature],
    onset_window_sec: float,
) -> AudioFrameFeature | None:
    candidates = [
        frame
        for frame in frames
        if abs(frame.start_sec - note_context.start_sec) <= onset_window_sec
    ]
    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda frame: (-frame.onset_score, abs(frame.start_sec - note_context.start_sec)),
    )[0]


def _action_from_confidence(confidence: float, params: ValidationParameters) -> str:
    if confidence >= params.keep_threshold:
        return "KEEP"
    if confidence >= params.review_threshold:
        return "REVIEW"
    return "MUTE_CANDIDATE"


def _score_energy(mean_rms: float, max_rms: float, minimum_rms: float) -> float:
    if minimum_rms <= 0:
        return 1.0

    mean_component = min(1.0, mean_rms / minimum_rms)
    max_component = min(1.0, max_rms / (minimum_rms * 2.0))
    return (0.6 * mean_component) + (0.4 * max_component)


def _score_onset(onset_score: float, minimum_onset_score: float) -> float:
    if minimum_onset_score <= 0:
        return 1.0 if onset_score > 0 else 0.0
    return min(1.0, onset_score / minimum_onset_score)


def _validate_note(
    note_context: _ValidationNoteContext,
    frames: list[AudioFrameFeature],
    params: ValidationParameters,
) -> NoteValidation:
    overlapping = _overlapping_frames(note_context, frames)
    onset_frame = _select_onset_frame(note_context, frames, params.onset_window_ms / 1000.0)

    onset_score = float(onset_frame.onset_score) if onset_frame else 0.0
    nearest_onset_sec = float(onset_frame.start_sec) if onset_frame else None
    onset_error_ms = (
        abs(nearest_onset_sec - note_context.start_sec) * 1000.0
        if nearest_onset_sec is not None
        else None
    )

    if not overlapping:
        max_rms = 0.0
        mean_rms = 0.0
        sustained_ratio = 0.0
    else:
        rms_values = [frame.rms for frame in overlapping]
        max_rms = float(max(rms_values))
        mean_rms = float(sum(rms_values) / len(rms_values))
        sustained_ratio = float(
            sum(1 for frame in overlapping if not frame.is_silent) / len(overlapping)
        )

    energy_match_score = _score_energy(mean_rms, max_rms, params.minimum_rms)
    duration_match_score = sustained_ratio
    onset_match_score = _score_onset(onset_score, params.minimum_onset_score)

    confidence = (0.35 * onset_match_score) + (0.40 * energy_match_score) + (0.25 * duration_match_score)
    confidence = max(0.0, min(1.0, confidence))

    recommended_action = _action_from_confidence(confidence, params)

    reasons: list[str] = []
    if not overlapping:
        reasons.append("no overlapping audio frames")
    if onset_score < params.minimum_onset_score:
        reasons.append("weak or missing onset near note start")
    if mean_rms < params.minimum_rms:
        reasons.append("low RMS during note")
    if sustained_ratio < 0.5:
        reasons.append("short/weak sustained energy")
    if confidence >= params.keep_threshold:
        reasons.append("high confidence match")
    if not reasons:
        reasons.append("mixed evidence for note-to-audio match")

    note = note_context.note
    return NoteValidation(
        note_id=note.note_id,
        pitch_midi=note.pitch_midi,
        pitch_name=note.pitch_name,
        layer=note.layer,
        source=note.source,
        start_sec=note_context.start_sec,
        end_sec=note_context.end_sec,
        duration_sec=note_context.duration_sec,
        nearest_onset_sec=nearest_onset_sec,
        onset_error_ms=onset_error_ms,
        onset_score=onset_score,
        max_rms_during_note=max_rms,
        mean_rms_during_note=mean_rms,
        sustained_energy_ratio=sustained_ratio,
        energy_match_score=energy_match_score,
        duration_match_score=duration_match_score,
        confidence=confidence,
        recommended_action=recommended_action,
        reasons=reasons,
    )


def validate_midi_vs_audio(
    notes_file: Path,
    audio_features_file: Path,
    params: ValidationParameters,
    audio_aligned_notes_file: Path | None = None,
) -> tuple[NoteValidationDocument, MidiAudioValidationReport]:
    if not notes_file.exists() or not notes_file.is_file():
        raise MidiAudioValidationError(f"Notes file does not exist: {notes_file}")
    if not audio_features_file.exists() or not audio_features_file.is_file():
        raise MidiAudioValidationError(f"Audio features file does not exist: {audio_features_file}")
    if audio_aligned_notes_file is not None:
        if not audio_aligned_notes_file.exists() or not audio_aligned_notes_file.is_file():
            raise MidiAudioValidationError(
                f"Audio aligned notes file does not exist: {audio_aligned_notes_file}"
            )

    note_document = _load_note_document(notes_file)
    audio_document = _load_audio_document(audio_features_file)
    aligned_document: AudioAlignedNoteDocument | None = None
    if audio_aligned_notes_file is not None:
        aligned_document = _load_audio_aligned_document(audio_aligned_notes_file)

    warnings: list[str] = []
    if note_document.layer != audio_document.layer:
        warnings.append(
            "Layer mismatch between notes and audio features: "
            f"{note_document.layer} vs {audio_document.layer}."
        )
    if aligned_document is not None and aligned_document.layer != note_document.layer:
        warnings.append(
            "Layer mismatch between notes and audio alignment: "
            f"{note_document.layer} vs {aligned_document.layer}."
        )

    note_contexts: list[_ValidationNoteContext] = []
    timing_source = "original_note_events_seconds"
    if aligned_document is not None:
        timing_source = "audio_aligned_seconds"
        aligned_by_note_id = {item.note_id: item for item in aligned_document.notes}
        for note in note_document.notes:
            aligned = aligned_by_note_id.get(note.note_id)
            if aligned is None:
                warnings.append(
                    f"Audio alignment missing for note_id: {note.note_id}; used original note-event timing"
                )
                start_sec = float(note.start_sec)
                end_sec = float(note.end_sec)
            else:
                start_sec = max(0.0, float(aligned.aligned_start_sec))
                end_sec = max(start_sec, float(aligned.aligned_end_sec))

            note_contexts.append(
                _ValidationNoteContext(
                    note=note,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=max(0.0, end_sec - start_sec),
                )
            )
    else:
        note_contexts = [
            _ValidationNoteContext(
                note=note,
                start_sec=float(note.start_sec),
                end_sec=float(note.end_sec),
                duration_sec=max(0.0, float(note.duration_sec)),
            )
            for note in note_document.notes
        ]

    validations = [
        _validate_note(note_context, audio_document.frames, params)
        for note_context in note_contexts
    ]

    keep_count = sum(1 for item in validations if item.recommended_action == "KEEP")
    review_count = sum(1 for item in validations if item.recommended_action == "REVIEW")
    mute_candidate_count = sum(1 for item in validations if item.recommended_action == "MUTE_CANDIDATE")

    mean_confidence = (
        sum(item.confidence for item in validations) / len(validations)
        if validations
        else 0.0
    )

    document = NoteValidationDocument(
        schema_version=SCHEMA_VERSION,
        notes_file=str(notes_file),
        audio_features_file=str(audio_features_file),
        layer=note_document.layer,
        validation_parameters={
            "onset_window_ms": params.onset_window_ms,
            "minimum_rms": params.minimum_rms,
            "minimum_onset_score": params.minimum_onset_score,
            "review_threshold": params.review_threshold,
            "keep_threshold": params.keep_threshold,
        },
        validations=validations,
    )

    report = MidiAudioValidationReport(
        notes_file=str(notes_file),
        audio_features_file=str(audio_features_file),
        timing_source=timing_source,
        audio_aligned_notes_file=(
            str(audio_aligned_notes_file)
            if audio_aligned_notes_file is not None
            else None
        ),
        status="ok",
        layer=note_document.layer,
        note_count=len(validations),
        keep_count=keep_count,
        review_count=review_count,
        mute_candidate_count=mute_candidate_count,
        mean_confidence=float(mean_confidence),
        warning_count=len(warnings),
        warnings=warnings,
        output_file=None,
    )

    return document, report
