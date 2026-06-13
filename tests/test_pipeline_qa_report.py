from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from midi_cleaner.alignment.models import (
    AudioAlignedNoteDocument,
    AudioAlignedNoteEvent,
    AudioAlignmentReport,
)
from midi_cleaner.cleanup.models import (
    CleanedMidiExportFile,
    CleanedMidiExportReport,
    CleanupAction,
    CleanupPlanDocument,
    ReviewMidiExportFile,
    ReviewMidiExportReport,
    WorkingMidiExportFile,
    WorkingMidiExportReport,
)
from midi_cleaner.cli import app
from midi_cleaner.pipeline.models import PipelineReport, PipelineStageReport
from midi_cleaner.pipeline.qa_report import QAReportError, QAReportParameters, generate_qa_report
from midi_cleaner.refinement.models import BassRefinementReport, RefinedNoteDocument, RefinedNoteEvent
from midi_cleaner.validation.models import NoteValidation, NoteValidationDocument


runner = CliRunner()


def _validation(
    note_id: str,
    confidence: float,
    action: str,
    onset_score: float,
    mean_rms: float,
    reasons: list[str],
) -> NoteValidation:
    return NoteValidation(
        note_id=note_id,
        pitch_midi=60,
        pitch_name="C4",
        layer="bass",
        source="ripx",
        start_sec=0.0,
        end_sec=0.5,
        duration_sec=0.5,
        nearest_onset_sec=0.0,
        onset_error_ms=0.0,
        onset_score=onset_score,
        max_rms_during_note=0.02,
        mean_rms_during_note=mean_rms,
        sustained_energy_ratio=0.8,
        energy_match_score=0.7,
        duration_match_score=0.8,
        confidence=confidence,
        recommended_action=action,
        reasons=reasons,
    )


def _write_pipeline_like_project(tmp_path: Path, include_optional: bool = True) -> Path:
    project_dir = tmp_path / "pipeline_project"
    (project_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (project_dir / "cleanup").mkdir(parents=True, exist_ok=True)
    (project_dir / "midi" / "cleaned").mkdir(parents=True, exist_ok=True)
    (project_dir / "midi" / "review").mkdir(parents=True, exist_ok=True)
    (project_dir / "midi" / "working").mkdir(parents=True, exist_ok=True)
    (project_dir / "reports").mkdir(parents=True, exist_ok=True)

    note_validation = NoteValidationDocument(
        schema_version="0.1.0",
        notes_file="analysis/note_events.json",
        audio_features_file="analysis/audio_features.json",
        layer="bass",
        validation_parameters={
            "onset_window_ms": 50.0,
            "minimum_rms": 0.001,
            "minimum_onset_score": 0.01,
            "review_threshold": 0.45,
            "keep_threshold": 0.70,
        },
        validations=[
            _validation("k", 0.9, "KEEP", 0.2, 0.01, ["high confidence match"]),
            _validation("r", 0.6, "REVIEW", 0.02, 0.005, ["manual review", "<unsafe>"]),
            _validation("m", 0.2, "MUTE_CANDIDATE", 0.001, 0.0005, ["low RMS"]),
            _validation("d", 0.1, "MUTE_CANDIDATE", 0.0, 0.0001, ["no overlap"]),
        ],
    )
    (project_dir / "analysis" / "note_validation.json").write_text(
        note_validation.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (project_dir / "analysis" / "midi_audio_validation_report.json").write_text(
        json.dumps(
            {
                "notes_file": "analysis/note_events.json",
                "audio_features_file": "analysis/audio_features.json",
                "timing_source": "audio_aligned_seconds",
                "audio_aligned_notes_file": "analysis/audio_aligned_note_events.json",
                "status": "ok",
                "layer": "bass",
                "note_count": 4,
                "keep_count": 1,
                "review_count": 1,
                "mute_candidate_count": 2,
                "mean_confidence": 0.45,
                "warning_count": 0,
                "warnings": [],
                "output_file": "analysis/note_validation.json",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cleanup_plan = CleanupPlanDocument(
        schema_version="0.1.0",
        validation_file="analysis/note_validation.json",
        layer="bass",
        planner_parameters={
            "mute_threshold": 0.45,
            "review_threshold": 0.70,
            "delete_threshold": 0.20,
            "allow_delete_candidates": True,
        },
        actions=[
            CleanupAction(
                note_id="k",
                original_recommended_action="KEEP",
                plan_action="KEEP",
                confidence=0.9,
                reasons=["keep"],
                source_validation={"recommended_action": "KEEP"},
            ),
            CleanupAction(
                note_id="r",
                original_recommended_action="REVIEW",
                plan_action="REVIEW",
                confidence=0.6,
                reasons=["review"],
                source_validation={"recommended_action": "REVIEW"},
            ),
            CleanupAction(
                note_id="m",
                original_recommended_action="MUTE_CANDIDATE",
                plan_action="MUTE",
                confidence=0.2,
                reasons=["mute"],
                source_validation={"recommended_action": "MUTE_CANDIDATE"},
            ),
            CleanupAction(
                note_id="d",
                original_recommended_action="MUTE_CANDIDATE",
                plan_action="DELETE_CANDIDATE",
                confidence=0.1,
                reasons=["delete"],
                source_validation={"recommended_action": "MUTE_CANDIDATE"},
            ),
        ],
    )
    (project_dir / "cleanup" / "cleanup_plan.json").write_text(
        cleanup_plan.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    if include_optional:
        aligned_doc = AudioAlignedNoteDocument(
            schema_version="0.1.0",
            notes_file="analysis/note_events.json",
            audio_features_file="analysis/audio_features.json",
            layer="bass",
            sample_rate=44100,
            audio_duration_sec=1.0,
            alignment_parameters={
                "onset_search_window_ms": 250.0,
                "offset_search_window_ms": 350.0,
                "min_onset_score": 0.005,
                "min_rms": 0.001,
                "snap_start_to_audio_onset": True,
                "snap_end_to_energy_offset": True,
                "max_start_correction_ms": 500.0,
                "max_end_correction_ms": 800.0,
                "low_confidence_action": "KEEP_ORIGINAL_LOW_CONFIDENCE",
            },
            notes=[
                AudioAlignedNoteEvent(
                    note_id="k",
                    source="ripx",
                    layer="bass",
                    pitch_midi=60,
                    pitch_name="C4",
                    velocity=90,
                    channel=0,
                    original_start_sec=0.0,
                    original_end_sec=0.5,
                    original_duration_sec=0.5,
                    original_start_tick=0,
                    original_end_tick=240,
                    aligned_start_sec=0.01,
                    aligned_end_sec=0.51,
                    aligned_duration_sec=0.5,
                    start_correction_ms=10.0,
                    end_correction_ms=10.0,
                    duration_correction_ms=0.0,
                    nearest_audio_onset_sec=0.01,
                    nearest_audio_offset_sec=0.51,
                    onset_error_before_ms=0.0,
                    onset_error_after_ms=10.0,
                    local_rms=0.01,
                    local_onset_score=0.2,
                    sustained_energy_ratio=0.8,
                    alignment_confidence=0.9,
                    alignment_action="ALIGNED",
                    reasons=["test"],
                )
            ],
        )
        (project_dir / "analysis" / "audio_aligned_note_events.json").write_text(
            aligned_doc.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        refined_doc = RefinedNoteDocument(
            schema_version="0.1.0",
            aligned_notes_file="analysis/audio_aligned_note_events.json",
            audio_features_file="analysis/audio_features.json",
            validation_file="analysis/note_validation.json",
            layer="bass",
            sample_rate=44100,
            audio_duration_sec=1.0,
            refinement_parameters={
                "attack_lookback_ms": 80.0,
                "max_attack_advance_ms": 80.0,
                "merge_gap_ms": 160.0,
                "minimum_silence_ms": 80.0,
                "tail_rms_ratio": 0.2,
                "tail_silence_hold_ms": 120.0,
                "max_tail_extension_ms": 900.0,
                "minimum_note_duration_ms": 80.0,
                "monophonic": True,
                "allow_pitch_overlap": False,
            },
            notes=[
                RefinedNoteEvent(
                    note_id="k",
                    source="ripx",
                    layer="bass",
                    pitch_midi=60,
                    pitch_name="C4",
                    velocity=90,
                    channel=0,
                    original_start_sec=0.0,
                    original_end_sec=0.5,
                    aligned_start_sec=0.01,
                    aligned_end_sec=0.51,
                    refined_start_sec=0.0,
                    refined_end_sec=0.56,
                    refined_duration_sec=0.56,
                    start_refinement_ms=-10.0,
                    end_refinement_ms=50.0,
                    merged_note_ids=["r"],
                    refinement_actions=["ATTACK_START_ADJUSTED", "SUSTAIN_TAIL_EXTENDED"],
                    refinement_confidence=0.9,
                    reasons=["test"],
                )
            ],
        )
        (project_dir / "analysis" / "refined_note_events.json").write_text(
            refined_doc.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        refinement_report = BassRefinementReport(
            aligned_notes_file="analysis/audio_aligned_note_events.json",
            audio_features_file="analysis/audio_features.json",
            validation_file="analysis/note_validation.json",
            status="ok",
            layer="bass",
            input_note_count=4,
            output_note_count=3,
            merged_count=1,
            false_retrigger_merge_count=1,
            tail_extended_count=1,
            short_note_extended_count=0,
            overlap_resolved_count=1,
            median_start_refinement_ms=-5.0,
            median_end_refinement_ms=25.0,
            max_tail_extension_ms=150.0,
            warning_count=0,
            warnings=[],
            output_file="analysis/refined_note_events.json",
        )
        (project_dir / "analysis" / "bass_refinement_report.json").write_text(
            refinement_report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        alignment_report = AudioAlignmentReport(
            notes_file="analysis/note_events.json",
            audio_features_file="analysis/audio_features.json",
            status="ok",
            layer="bass",
            note_count=4,
            aligned_count=2,
            keep_original_count=1,
            review_timing_count=1,
            no_audio_evidence_count=0,
            median_abs_start_correction_ms=10.0,
            p95_abs_start_correction_ms=18.0,
            max_abs_start_correction_ms=20.0,
            warning_count=0,
            warnings=[],
            output_file="analysis/audio_aligned_note_events.json",
        )
        (project_dir / "analysis" / "audio_alignment_report.json").write_text(
            alignment_report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        cleaned_report = CleanedMidiExportReport(
            notes_file="analysis/note_events.json",
            cleanup_plan_file="cleanup/cleanup_plan.json",
            audio_aligned_notes_file="analysis/audio_aligned_note_events.json",
            status="ok",
            layer="bass",
            ticks_per_beat=960,
            timing_source="audio_aligned_seconds",
            max_export_time_error_ms=0.6,
            mean_export_time_error_ms=0.2,
            source_ticks_per_beat=480,
            exported_ticks_per_beat=960,
            tempo_us_per_beat=500000,
            bpm=120.0,
            cleaned_note_count=1,
            review_note_count=1,
            rejected_note_count=2,
            exported_files=[
                CleanedMidiExportFile(
                    role="CLEANED",
                    path="midi/cleaned/cleaned.mid",
                    note_count=1,
                    included_plan_actions=["KEEP"],
                )
            ],
            warning_count=0,
            warnings=[],
        )
        (project_dir / "midi" / "cleaned" / "cleaned_export_report.json").write_text(
            cleaned_report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        review_report = ReviewMidiExportReport(
            notes_file="analysis/note_events.json",
            cleanup_plan_file="cleanup/cleanup_plan.json",
            audio_aligned_notes_file="analysis/audio_aligned_note_events.json",
            status="ok",
            layer="bass",
            ticks_per_beat=960,
            timing_source="audio_aligned_seconds",
            max_export_time_error_ms=0.5,
            mean_export_time_error_ms=0.2,
            source_ticks_per_beat=480,
            exported_ticks_per_beat=960,
            tempo_us_per_beat=500000,
            bpm=120.0,
            exported_files=[
                ReviewMidiExportFile(action="KEEP", path="midi/review/keep.mid", note_count=1)
            ],
            warning_count=0,
            warnings=[],
        )
        (project_dir / "midi" / "review" / "export_report.json").write_text(
            review_report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        working_report = WorkingMidiExportReport(
            notes_file="analysis/note_events.json",
            cleanup_plan_file="cleanup/cleanup_plan.json",
            refined_notes_file="analysis/refined_note_events.json",
            audio_aligned_notes_file="analysis/audio_aligned_note_events.json",
            status="ok",
            layer="bass",
            ticks_per_beat=960,
            timing_source="refined_audio_seconds",
            max_export_time_error_ms=0.4,
            mean_export_time_error_ms=0.15,
            source_ticks_per_beat=480,
            exported_ticks_per_beat=960,
            tempo_us_per_beat=500000,
            bpm=120.0,
            working_note_count=2,
            rejected_note_count=1,
            diagnostic_note_count=0,
            exported_files=[
                WorkingMidiExportFile(
                    role="WORKING",
                    path="midi/working/working.mid",
                    note_count=2,
                    included_plan_actions=["KEEP", "REVIEW"],
                ),
                WorkingMidiExportFile(
                    role="REJECTED",
                    path="midi/working/rejected.mid",
                    note_count=1,
                    included_plan_actions=["MUTE", "DELETE_CANDIDATE"],
                ),
            ],
            warning_count=0,
            warnings=[],
        )
        (project_dir / "midi" / "working" / "working_export_report.json").write_text(
            working_report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        pipeline_report = PipelineReport(
            status="ok",
            input_midi="candidate.mid",
            input_wav="stem.wav",
            source="ripx",
            layer="bass",
            project_dir=str(project_dir),
            stages=[
                PipelineStageReport(
                    name="dummy",
                    status="ok",
                    output_files=[],
                    warning_count=1,
                    warnings=["stage warning"],
                )
            ],
            output_files={},
            warning_count=1,
            warnings=["pipeline warning"],
        )
        (project_dir / "reports" / "pipeline_report.json").write_text(
            pipeline_report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    return project_dir


def test_qa_report_creates_summary_csv_and_html(tmp_path: Path) -> None:
    project_dir = _write_pipeline_like_project(tmp_path)

    summary = generate_qa_report(project_dir, None, QAReportParameters())

    assert summary.status == "ok"
    assert (project_dir / "reports" / "qa_summary.json").exists()
    assert (project_dir / "reports" / "qa_notes.csv").exists()
    assert (project_dir / "reports" / "qa_report.html").exists()


def test_summary_counts_actions_correctly(tmp_path: Path) -> None:
    project_dir = _write_pipeline_like_project(tmp_path)

    summary = generate_qa_report(project_dir, None, QAReportParameters())

    assert summary.keep_count == 1
    assert summary.review_count == 1
    assert summary.mute_count == 1
    assert summary.delete_candidate_count == 1
    assert summary.aligned_count == 2
    assert summary.keep_original_count == 1
    assert summary.review_timing_count == 1


def test_csv_contains_expected_rows(tmp_path: Path) -> None:
    project_dir = _write_pipeline_like_project(tmp_path)

    generate_qa_report(project_dir, None, QAReportParameters())

    with (project_dir / "reports" / "qa_notes.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert any(row["note_id"] == "k" for row in rows)
    assert "original_start_sec" in rows[0]
    assert "aligned_start_sec" in rows[0]
    assert "start_correction_ms" in rows[0]
    assert "refined_start_sec" in rows[0]
    assert "refined_end_sec" in rows[0]
    assert "start_refinement_ms" in rows[0]
    assert "end_refinement_ms" in rows[0]
    assert "refinement_actions" in rows[0]
    assert "merged_note_ids" in rows[0]
    assert "alignment_action" in rows[0]
    assert "alignment_confidence" in rows[0]
    assert "validation_action" in rows[0]
    assert "plan_action" in rows[0]


def test_html_contains_sections_and_escaped_content(tmp_path: Path) -> None:
    project_dir = _write_pipeline_like_project(tmp_path)

    generate_qa_report(project_dir, None, QAReportParameters())

    html_text = (project_dir / "reports" / "qa_report.html").read_text(encoding="utf-8")
    assert "Hermes Static QA Report" in html_text
    assert "Audio-Time Alignment / Sync" in html_text
    assert "validation_timing_source" in html_text
    assert "review_export_timing_source" in html_text
    assert "cleaned_export_timing_source" in html_text
    assert "global_offset_ms" in html_text
    assert "max_export_time_error_ms" in html_text
    assert "working_export_time_error_ms" in html_text
    assert "refined_note_count" in html_text
    assert "Top 25 Lowest-Confidence Notes" in html_text
    assert "&lt;unsafe&gt;" in html_text


def test_summary_contains_audio_sync_fields(tmp_path: Path) -> None:
    project_dir = _write_pipeline_like_project(tmp_path)

    summary = generate_qa_report(project_dir, None, QAReportParameters())

    assert summary.validation_timing_source == "audio_aligned_seconds"
    assert summary.review_export_timing_source == "audio_aligned_seconds"
    assert summary.cleaned_export_timing_source == "audio_aligned_seconds"
    assert summary.global_offset_ms == 0.0
    assert summary.global_confidence == 0.0
    assert summary.global_offset_applied is False
    assert summary.max_export_time_error_ms == 0.6
    assert summary.mean_export_time_error_ms == 0.2
    assert summary.refined_note_count == 3
    assert summary.merged_count == 1
    assert summary.false_retrigger_merge_count == 1
    assert summary.tail_extended_count == 1
    assert summary.overlap_resolved_count == 1
    assert summary.working_midi_note_count == 2
    assert summary.working_export_time_error_ms == 0.4


def test_missing_optional_reports_warns_but_succeeds(tmp_path: Path) -> None:
    project_dir = _write_pipeline_like_project(tmp_path, include_optional=False)

    summary = generate_qa_report(project_dir, None, QAReportParameters())

    assert summary.status == "ok"
    assert summary.warning_count >= 1


def test_missing_project_dir_fails(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does_not_exist"

    try:
        generate_qa_report(missing_dir, None, QAReportParameters())
        assert False, "Expected QAReportError"
    except QAReportError:
        assert True


def test_cli_qa_report_command_works(tmp_path: Path) -> None:
    project_dir = _write_pipeline_like_project(tmp_path)

    result = runner.invoke(
        app,
        [
            "pipeline",
            "qa-report",
            "--project-dir",
            str(project_dir),
        ],
    )

    assert result.exit_code == 0
    assert (project_dir / "reports" / "qa_summary.json").exists()
    assert (project_dir / "reports" / "qa_notes.csv").exists()
    assert (project_dir / "reports" / "qa_report.html").exists()

    summary = json.loads((project_dir / "reports" / "qa_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
