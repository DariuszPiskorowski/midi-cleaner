from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from midi_cleaner.cleanup.models import CleanupAction, CleanupPlanDocument, CleanupPlanReport
from midi_cleaner.validation.models import NoteValidation, NoteValidationDocument

SCHEMA_VERSION = "0.1.0"


class CleanupPlanError(Exception):
    """Raised when cleanup plan generation cannot be completed."""


@dataclass(frozen=True)
class CleanupPlannerParameters:
    mute_threshold: float = 0.45
    review_threshold: float = 0.70
    delete_threshold: float = 0.20
    allow_delete_candidates: bool = False


def _load_validation_document(path: Path) -> NoteValidationDocument:
    try:
        return NoteValidationDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - pydantic internals vary
        raise CleanupPlanError(f"Invalid validation JSON: {path}") from exc


def _source_validation_payload(item: NoteValidation) -> dict[str, object]:
    return {
        "recommended_action": item.recommended_action,
        "onset_score": item.onset_score,
        "energy_match_score": item.energy_match_score,
        "duration_match_score": item.duration_match_score,
        "nearest_onset_sec": item.nearest_onset_sec,
        "onset_error_ms": item.onset_error_ms,
        "max_rms_during_note": item.max_rms_during_note,
        "mean_rms_during_note": item.mean_rms_during_note,
        "sustained_energy_ratio": item.sustained_energy_ratio,
    }


def _plan_action_for_validation(
    validation: NoteValidation,
    params: CleanupPlannerParameters,
) -> tuple[str, list[str]]:
    confidence = validation.confidence
    reasons = list(validation.reasons)

    if validation.recommended_action == "KEEP" and confidence >= params.review_threshold:
        reasons.append("kept because confidence is above review threshold")
        return "KEEP", reasons

    if confidence >= params.review_threshold:
        reasons.append("kept because confidence is above review threshold")
        return "KEEP", reasons

    if confidence >= params.mute_threshold:
        reasons.append("manual review recommended")
        return "REVIEW", reasons

    if params.allow_delete_candidates and confidence < params.delete_threshold:
        reasons.append("delete candidate only; no destructive action applied")
        return "DELETE_CANDIDATE", reasons

    reasons.append("muted non-destructively because confidence is low")
    return "MUTE", reasons


def build_cleanup_plan(
    validation_file: Path,
    params: CleanupPlannerParameters,
) -> tuple[CleanupPlanDocument, CleanupPlanReport]:
    if not validation_file.exists() or not validation_file.is_file():
        raise CleanupPlanError(f"Validation file does not exist: {validation_file}")

    validation_document = _load_validation_document(validation_file)

    warnings: list[str] = []
    if params.delete_threshold > params.mute_threshold:
        warnings.append("delete_threshold is above mute_threshold; delete candidates may never be selected.")
    if params.mute_threshold > params.review_threshold:
        warnings.append("mute_threshold is above review_threshold; review band is inverted.")

    actions: list[CleanupAction] = []
    for item in validation_document.validations:
        plan_action, reasons = _plan_action_for_validation(item, params)
        actions.append(
            CleanupAction(
                note_id=item.note_id,
                original_recommended_action=item.recommended_action,
                plan_action=plan_action,
                confidence=item.confidence,
                reasons=reasons,
                source_validation=_source_validation_payload(item),
            )
        )

    keep_count = sum(1 for action in actions if action.plan_action == "KEEP")
    review_count = sum(1 for action in actions if action.plan_action == "REVIEW")
    mute_count = sum(1 for action in actions if action.plan_action == "MUTE")
    delete_candidate_count = sum(
        1 for action in actions if action.plan_action == "DELETE_CANDIDATE"
    )

    plan_document = CleanupPlanDocument(
        schema_version=SCHEMA_VERSION,
        validation_file=str(validation_file),
        layer=validation_document.layer,
        planner_parameters={
            "mute_threshold": params.mute_threshold,
            "review_threshold": params.review_threshold,
            "delete_threshold": params.delete_threshold,
            "allow_delete_candidates": params.allow_delete_candidates,
        },
        actions=actions,
    )

    report = CleanupPlanReport(
        validation_file=str(validation_file),
        status="ok",
        layer=validation_document.layer,
        action_count=len(actions),
        keep_count=keep_count,
        review_count=review_count,
        mute_count=mute_count,
        delete_candidate_count=delete_candidate_count,
        warning_count=len(warnings),
        warnings=warnings,
        output_file=None,
    )

    return plan_document, report
