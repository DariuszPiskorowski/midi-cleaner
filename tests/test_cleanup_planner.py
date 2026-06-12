from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from midi_cleaner.cli import app
from midi_cleaner.cleanup.planner import CleanupPlannerParameters, build_cleanup_plan
from midi_cleaner.validation.models import NoteValidation, NoteValidationDocument


runner = CliRunner()


def _validation(note_id: str, confidence: float, recommended_action: str) -> NoteValidation:
    return NoteValidation(
        note_id=note_id,
        pitch_midi=45,
        pitch_name="A2",
        layer="bass",
        source="ripx",
        start_sec=0.0,
        end_sec=0.25,
        duration_sec=0.25,
        nearest_onset_sec=0.0,
        onset_error_ms=0.0,
        onset_score=0.2,
        max_rms_during_note=0.02,
        mean_rms_during_note=0.01,
        sustained_energy_ratio=0.8,
        energy_match_score=0.7,
        duration_match_score=0.8,
        confidence=confidence,
        recommended_action=recommended_action,
        reasons=["base reason"],
    )


def _write_validation_doc(tmp_path: Path, items: list[NoteValidation]) -> Path:
    doc = NoteValidationDocument(
        schema_version="0.1.0",
        notes_file="note_events.json",
        audio_features_file="audio_features.json",
        layer="bass",
        validation_parameters={
            "onset_window_ms": 50.0,
            "minimum_rms": 0.001,
            "minimum_onset_score": 0.01,
            "review_threshold": 0.45,
            "keep_threshold": 0.70,
        },
        validations=items,
    )
    path = tmp_path / "note_validation.json"
    path.write_text(doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def test_keep_validation_becomes_keep(tmp_path: Path) -> None:
    validation_path = _write_validation_doc(tmp_path, [_validation("n1", 0.9, "KEEP")])

    doc, report = build_cleanup_plan(validation_path, CleanupPlannerParameters())

    assert report.keep_count == 1
    assert doc.actions[0].plan_action == "KEEP"


def test_review_range_confidence_becomes_review(tmp_path: Path) -> None:
    validation_path = _write_validation_doc(tmp_path, [_validation("n2", 0.55, "REVIEW")])

    doc, report = build_cleanup_plan(validation_path, CleanupPlannerParameters())

    assert report.review_count == 1
    assert doc.actions[0].plan_action == "REVIEW"


def test_low_confidence_becomes_mute_by_default(tmp_path: Path) -> None:
    validation_path = _write_validation_doc(tmp_path, [_validation("n3", 0.10, "MUTE_CANDIDATE")])

    doc, report = build_cleanup_plan(validation_path, CleanupPlannerParameters())

    assert report.mute_count == 1
    assert report.delete_candidate_count == 0
    assert doc.actions[0].plan_action == "MUTE"


def test_very_low_confidence_can_be_delete_candidate(tmp_path: Path) -> None:
    validation_path = _write_validation_doc(tmp_path, [_validation("n4", 0.10, "MUTE_CANDIDATE")])

    doc, report = build_cleanup_plan(
        validation_path,
        CleanupPlannerParameters(allow_delete_candidates=True),
    )

    assert report.delete_candidate_count == 1
    assert doc.actions[0].plan_action == "DELETE_CANDIDATE"


def test_cli_writes_cleanup_plan_and_report_with_correct_counts(tmp_path: Path) -> None:
    validation_path = _write_validation_doc(
        tmp_path,
        [
            _validation("n_keep", 0.9, "KEEP"),
            _validation("n_review", 0.6, "REVIEW"),
            _validation("n_mute", 0.4, "MUTE_CANDIDATE"),
            _validation("n_delete", 0.1, "MUTE_CANDIDATE"),
        ],
    )

    output_path = tmp_path / "out" / "cleanup_plan.json"
    report_path = tmp_path / "out" / "cleanup_plan_report.json"

    result = runner.invoke(
        app,
        [
            "cleanup",
            "plan",
            "--validation",
            str(validation_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--allow-delete-candidates",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert report_path.exists()

    plan_doc = json.loads(output_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(plan_doc["actions"]) == 4
    assert report["action_count"] == 4
    assert report["keep_count"] == 1
    assert report["review_count"] == 1
    assert report["mute_count"] == 1
    assert report["delete_candidate_count"] == 1
