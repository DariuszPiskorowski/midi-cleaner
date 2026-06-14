from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import median

from midi_cleaner.alignment.models import (
    AudioAlignedNoteDocument,
    AudioAlignedNoteEvent,
    AudioAlignmentReport,
)
from midi_cleaner.cleanup.models import CleanupPlanDocument, WorkingMidiExportReport
from midi_cleaner.pipeline.models import PipelineReport, QANoteRow, QASummary
from midi_cleaner.repair.models import ActivityRepairReport
from midi_cleaner.refinement.models import BassRefinementReport, RefinedNoteDocument, RefinedNoteEvent
from midi_cleaner.validation.models import NoteValidationDocument


class QAReportError(Exception):
    """Raised when QA report generation fails."""


@dataclass(frozen=True)
class QAReportParameters:
    top_n: int = 25
    include_csv: bool = True
    include_html: bool = True


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_note_validation(path: Path) -> NoteValidationDocument:
    return NoteValidationDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _load_cleanup_plan(path: Path) -> CleanupPlanDocument:
    return CleanupPlanDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _load_audio_aligned_document(path: Path) -> AudioAlignedNoteDocument:
    return AudioAlignedNoteDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _load_audio_alignment_report(path: Path) -> AudioAlignmentReport:
    return AudioAlignmentReport.model_validate_json(path.read_text(encoding="utf-8"))


def _load_pipeline_report(path: Path) -> PipelineReport:
    return PipelineReport.model_validate_json(path.read_text(encoding="utf-8"))


def _load_refined_document(path: Path) -> RefinedNoteDocument:
    return RefinedNoteDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _load_refinement_report(path: Path) -> BassRefinementReport:
    return BassRefinementReport.model_validate_json(path.read_text(encoding="utf-8"))


def _load_working_export_report(path: Path) -> WorkingMidiExportReport:
    return WorkingMidiExportReport.model_validate_json(path.read_text(encoding="utf-8"))


def _load_activity_repair_report(path: Path) -> ActivityRepairReport:
    return ActivityRepairReport.model_validate_json(path.read_text(encoding="utf-8"))


def _build_rows(
    note_validation: NoteValidationDocument | None,
    cleanup_plan: CleanupPlanDocument | None,
    aligned_notes: AudioAlignedNoteDocument | None,
    refined_notes: RefinedNoteDocument | None,
    warnings: list[str],
) -> list[QANoteRow]:
    if note_validation is None:
        return []

    plan_action_by_note_id: dict[str, str] = {}
    if cleanup_plan is not None:
        plan_action_by_note_id = {
            item.note_id: item.plan_action for item in cleanup_plan.actions
        }

    aligned_by_note_id: dict[str, AudioAlignedNoteEvent] = {}
    if aligned_notes is not None:
        aligned_by_note_id = {item.note_id: item for item in aligned_notes.notes}

    refined_by_note_id: dict[str, RefinedNoteEvent] = {}
    if refined_notes is not None:
        refined_by_note_id = {item.note_id: item for item in refined_notes.notes}

    validation_note_ids = {item.note_id for item in note_validation.validations}
    for note_id in plan_action_by_note_id:
        if note_id not in validation_note_ids:
            warnings.append(f"cleanup plan contains unknown note_id: {note_id}")

    for note_id in aligned_by_note_id:
        if note_id not in validation_note_ids:
            warnings.append(f"audio alignment contains unknown note_id: {note_id}")

    for note_id, refined in refined_by_note_id.items():
        if note_id in validation_note_ids:
            continue
        if any(merged_id in validation_note_ids for merged_id in refined.merged_note_ids):
            continue
        if any(action.startswith("ACTIVITY_REPAIR_") for action in refined.refinement_actions):
            continue
        warnings.append(f"refinement contains unknown note_id: {note_id}")

    rows: list[QANoteRow] = []
    missing_alignment_count = 0
    for item in note_validation.validations:
        plan_action = plan_action_by_note_id.get(item.note_id)
        if cleanup_plan is not None and plan_action is None:
            warnings.append(f"validation note has no cleanup plan action: {item.note_id}")

        aligned = aligned_by_note_id.get(item.note_id)
        if aligned_notes is not None and aligned is None:
            missing_alignment_count += 1

        original_start_sec = float(item.start_sec)
        aligned_start_sec = float(item.start_sec)
        start_correction_ms = 0.0
        alignment_action: str | None = None
        alignment_confidence: float | None = None
        refined_start_sec: float | None = None
        refined_end_sec: float | None = None
        start_refinement_ms: float | None = None
        end_refinement_ms: float | None = None
        refinement_actions: str | None = None
        merged_note_ids: str | None = None
        repair_actions: str | None = None
        repair_reason_summary: str | None = None
        was_inserted_by_repair = False
        was_split_by_repair = False
        was_extended_by_repair = False
        was_shortened_by_repair = False

        if aligned is not None:
            original_start_sec = float(aligned.original_start_sec)
            aligned_start_sec = float(aligned.aligned_start_sec)
            start_correction_ms = float(aligned.start_correction_ms)
            alignment_action = aligned.alignment_action
            alignment_confidence = float(aligned.alignment_confidence)

        refined = refined_by_note_id.get(item.note_id)
        if refined is not None:
            refined_start_sec = float(refined.refined_start_sec)
            refined_end_sec = float(refined.refined_end_sec)
            start_refinement_ms = float(refined.start_refinement_ms)
            end_refinement_ms = float(refined.end_refinement_ms)
            refinement_actions = "; ".join(refined.refinement_actions)
            merged_note_ids = "; ".join(refined.merged_note_ids)

            repair_tokens = [
                action
                for action in refined.refinement_actions
                if action.startswith("ACTIVITY_REPAIR_")
            ]
            if repair_tokens:
                repair_actions = "; ".join(repair_tokens)
                repair_reason_summary = "; ".join(refined.reasons)
            token_set = set(repair_tokens)
            was_inserted_by_repair = "ACTIVITY_REPAIR_INSERTED" in token_set
            was_split_by_repair = "ACTIVITY_REPAIR_SPLIT" in token_set
            was_extended_by_repair = "ACTIVITY_REPAIR_EXTENDED" in token_set
            was_shortened_by_repair = "ACTIVITY_REPAIR_SHORTENED" in token_set

        rows.append(
            QANoteRow(
                note_id=item.note_id,
                pitch_midi=item.pitch_midi,
                pitch_name=item.pitch_name,
                start_sec=item.start_sec,
                end_sec=item.end_sec,
                duration_sec=item.duration_sec,
                original_start_sec=original_start_sec,
                aligned_start_sec=aligned_start_sec,
                start_correction_ms=start_correction_ms,
                refined_start_sec=refined_start_sec,
                refined_end_sec=refined_end_sec,
                start_refinement_ms=start_refinement_ms,
                end_refinement_ms=end_refinement_ms,
                refinement_actions=refinement_actions,
                merged_note_ids=merged_note_ids,
                repair_actions=repair_actions,
                repair_reason_summary=repair_reason_summary,
                was_inserted_by_repair=was_inserted_by_repair,
                was_split_by_repair=was_split_by_repair,
                was_extended_by_repair=was_extended_by_repair,
                was_shortened_by_repair=was_shortened_by_repair,
                alignment_action=alignment_action,
                alignment_confidence=alignment_confidence,
                confidence=item.confidence,
                validation_action=item.recommended_action,
                plan_action=plan_action,
                onset_score=item.onset_score,
                mean_rms_during_note=item.mean_rms_during_note,
                sustained_energy_ratio=item.sustained_energy_ratio,
                reasons="; ".join(item.reasons),
            )
        )

    if missing_alignment_count > 0:
        warnings.append(
            f"audio alignment missing for {missing_alignment_count} validation notes"
        )

    return rows


def _write_csv(path: Path, rows: list[QANoteRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "note_id",
                "pitch_midi",
                "pitch_name",
                "start_sec",
                "end_sec",
                "duration_sec",
                "original_start_sec",
                "aligned_start_sec",
                "start_correction_ms",
                "refined_start_sec",
                "refined_end_sec",
                "start_refinement_ms",
                "end_refinement_ms",
                "refinement_actions",
                "merged_note_ids",
                "repair_actions",
                "repair_reason_summary",
                "was_inserted_by_repair",
                "was_split_by_repair",
                "was_extended_by_repair",
                "was_shortened_by_repair",
                "alignment_action",
                "alignment_confidence",
                "confidence",
                "validation_action",
                "plan_action",
                "onset_score",
                "mean_rms_during_note",
                "sustained_energy_ratio",
                "reasons",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def _rows_table_html(title: str, rows: list[QANoteRow]) -> str:
    if not rows:
        return f"<h3>{escape(title)}</h3><p>No rows.</p>"

    html_rows = []
    for row in rows:
        alignment_confidence_cell = (
            f"<td>{row.alignment_confidence:.4f}</td>"
            if row.alignment_confidence is not None
            else "<td></td>"
        )
        refined_start_cell = (
            f"<td>{row.refined_start_sec:.6f}</td>"
            if row.refined_start_sec is not None
            else "<td></td>"
        )
        refined_end_cell = (
            f"<td>{row.refined_end_sec:.6f}</td>"
            if row.refined_end_sec is not None
            else "<td></td>"
        )
        start_refinement_cell = (
            f"<td>{row.start_refinement_ms:.2f}</td>"
            if row.start_refinement_ms is not None
            else "<td></td>"
        )
        end_refinement_cell = (
            f"<td>{row.end_refinement_ms:.2f}</td>"
            if row.end_refinement_ms is not None
            else "<td></td>"
        )
        html_rows.append(
            "<tr>"
            f"<td>{escape(row.note_id)}</td>"
            f"<td>{row.pitch_midi}</td>"
            f"<td>{escape(row.pitch_name)}</td>"
            f"<td>{row.confidence:.4f}</td>"
            f"<td>{escape(row.validation_action)}</td>"
            f"<td>{escape(str(row.plan_action))}</td>"
            f"<td>{escape(str(row.alignment_action))}</td>"
            f"<td>{row.start_correction_ms:.2f}</td>"
            f"{alignment_confidence_cell}"
            f"{refined_start_cell}"
            f"{refined_end_cell}"
            f"{start_refinement_cell}"
            f"{end_refinement_cell}"
            f"<td>{escape(str(row.refinement_actions))}</td>"
            f"<td>{escape(str(row.merged_note_ids))}</td>"
            f"<td>{escape(str(row.repair_actions))}</td>"
            f"<td>{escape(str(row.repair_reason_summary))}</td>"
            f"<td>{str(row.was_inserted_by_repair)}</td>"
            f"<td>{str(row.was_split_by_repair)}</td>"
            f"<td>{str(row.was_extended_by_repair)}</td>"
            f"<td>{str(row.was_shortened_by_repair)}</td>"
            f"<td>{row.onset_score:.6f}</td>"
            f"<td>{row.mean_rms_during_note:.6f}</td>"
            f"<td>{row.sustained_energy_ratio:.4f}</td>"
            f"<td>{escape(row.reasons)}</td>"
            "</tr>"
        )

    return (
        f"<h3>{escape(title)}</h3>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<thead><tr>"
        "<th>note_id</th><th>pitch_midi</th><th>pitch_name</th><th>confidence</th>"
        "<th>validation_action</th><th>plan_action</th><th>alignment_action</th>"
        "<th>start_correction_ms</th><th>alignment_confidence</th>"
        "<th>refined_start_sec</th><th>refined_end_sec</th>"
        "<th>start_refinement_ms</th><th>end_refinement_ms</th>"
        "<th>refinement_actions</th><th>merged_note_ids</th>"
        "<th>repair_actions</th><th>repair_reason_summary</th>"
        "<th>repair_inserted</th><th>repair_split</th>"
        "<th>repair_extended</th><th>repair_shortened</th>"
        "<th>onset_score</th><th>mean_rms</th><th>sustained_ratio</th><th>reasons</th>"
        "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
    )


def _write_html(
    path: Path,
    summary: QASummary,
    rows: list[QANoteRow],
    top_n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lowest_confidence = sorted(rows, key=lambda item: item.confidence)[:top_n]
    weakest_onset = sorted(rows, key=lambda item: item.onset_score)[:top_n]
    lowest_rms = sorted(rows, key=lambda item: item.mean_rms_during_note)[:top_n]

    warning_items = "".join(f"<li>{escape(item)}</li>" for item in summary.warnings)
    output_items = "".join(
        f"<li>{escape(key)}: {escape(value)}</li>" for key, value in summary.output_files.items()
    )

    sync_rows = [
        ("validation_timing_source", str(summary.validation_timing_source)),
        ("review_export_timing_source", str(summary.review_export_timing_source)),
        ("cleaned_export_timing_source", str(summary.cleaned_export_timing_source)),
        ("global_offset_ms", str(summary.global_offset_ms)),
        ("global_confidence", str(summary.global_confidence)),
        ("global_offset_applied", str(summary.global_offset_applied)),
        ("aligned_count", str(summary.aligned_count)),
        ("keep_original_count", str(summary.keep_original_count)),
        ("review_timing_count", str(summary.review_timing_count)),
        ("no_audio_evidence_count", str(summary.no_audio_evidence_count)),
        ("median_abs_start_correction_ms", str(summary.median_abs_start_correction_ms)),
        ("p95_abs_start_correction_ms", str(summary.p95_abs_start_correction_ms)),
        ("max_abs_start_correction_ms", str(summary.max_abs_start_correction_ms)),
        ("max_export_time_error_ms", str(summary.max_export_time_error_ms)),
        ("mean_export_time_error_ms", str(summary.mean_export_time_error_ms)),
        ("working_export_time_error_ms", str(summary.working_export_time_error_ms)),
        ("median_start_refinement_ms", str(summary.median_start_refinement_ms)),
        ("median_end_refinement_ms", str(summary.median_end_refinement_ms)),
    ]
    sync_rows_html = "".join(
        f"<tr><th>{escape(name)}</th><td>{escape(value)}</td></tr>" for name, value in sync_rows
    )

    html_doc = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Hermes QA Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; line-height: 1.45; }}
    h1, h2, h3 {{ margin-top: 1.2em; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th {{ background: #f3f3f3; text-align: left; }}
    td, th {{ font-size: 13px; }}
    code {{ background: #f7f7f7; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>Hermes Static QA Report</h1>
  <p>This report is a heuristic QA artifact generated from pipeline outputs. It is informative, not final truth.</p>

  <h2>Summary</h2>
  <table border='1' cellpadding='6' cellspacing='0'>
    <tbody>
      <tr><th>project_dir</th><td>{escape(summary.project_dir)}</td></tr>
      <tr><th>layer</th><td>{escape(str(summary.layer))}</td></tr>
      <tr><th>total_notes</th><td>{summary.total_notes}</td></tr>
      <tr><th>keep_count</th><td>{summary.keep_count}</td></tr>
      <tr><th>review_count</th><td>{summary.review_count}</td></tr>
      <tr><th>mute_count</th><td>{summary.mute_count}</td></tr>
      <tr><th>delete_candidate_count</th><td>{summary.delete_candidate_count}</td></tr>
      <tr><th>cleaned_note_count</th><td>{summary.cleaned_note_count}</td></tr>
      <tr><th>rejected_note_count</th><td>{summary.rejected_note_count}</td></tr>
            <tr><th>refined_note_count</th><td>{summary.refined_note_count}</td></tr>
            <tr><th>merged_count</th><td>{summary.merged_count}</td></tr>
            <tr><th>false_retrigger_merge_count</th><td>{summary.false_retrigger_merge_count}</td></tr>
            <tr><th>tail_extended_count</th><td>{summary.tail_extended_count}</td></tr>
            <tr><th>short_note_extended_count</th><td>{summary.short_note_extended_count}</td></tr>
            <tr><th>overlap_resolved_count</th><td>{summary.overlap_resolved_count}</td></tr>
            <tr><th>dsp_backend_name</th><td>{summary.dsp_backend_name}</td></tr>
            <tr><th>dsp_backend_available</th><td>{summary.dsp_backend_available}</td></tr>
            <tr><th>dsp_frame_count</th><td>{summary.dsp_frame_count}</td></tr>
            <tr><th>dsp_attack_rise_count</th><td>{summary.dsp_attack_rise_count}</td></tr>
            <tr><th>dsp_sustain_count</th><td>{summary.dsp_sustain_count}</td></tr>
            <tr><th>dsp_tail_count</th><td>{summary.dsp_tail_count}</td></tr>
            <tr><th>dsp_silence_count</th><td>{summary.dsp_silence_count}</td></tr>
            <tr><th>dsp_debug_csv_file</th><td>{summary.dsp_debug_csv_file}</td></tr>
            <tr><th>activity_repair_enabled</th><td>{summary.activity_repair_enabled}</td></tr>
            <tr><th>repaired_note_count</th><td>{summary.repaired_note_count}</td></tr>
            <tr><th>repair_extend_count</th><td>{summary.repair_extend_count}</td></tr>
            <tr><th>repair_shorten_count</th><td>{summary.repair_shorten_count}</td></tr>
            <tr><th>repair_insert_missing_count</th><td>{summary.repair_insert_missing_count}</td></tr>
            <tr><th>repair_split_count</th><td>{summary.repair_split_count}</td></tr>
            <tr><th>repair_close_gap_count</th><td>{summary.repair_close_gap_count}</td></tr>
            <tr><th>repair_review_manual_count</th><td>{summary.repair_review_manual_count}</td></tr>
            <tr><th>repair_sustain_protected_count</th><td>{summary.repair_sustain_protected_count}</td></tr>
            <tr><th>repair_pitch_protected_count</th><td>{summary.repair_pitch_protected_count}</td></tr>
            <tr><th>repair_legato_protected_count</th><td>{summary.repair_legato_protected_count}</td></tr>
            <tr><th>repair_shorten_candidate_count</th><td>{summary.repair_shorten_candidate_count}</td></tr>
            <tr><th>repair_shorten_applied_count</th><td>{summary.repair_shorten_applied_count}</td></tr>
            <tr><th>repair_shorten_rejected_count</th><td>{summary.repair_shorten_rejected_count}</td></tr>
            <tr><th>audio_active_region_count</th><td>{summary.audio_active_region_count}</td></tr>
            <tr><th>midi_active_region_count</th><td>{summary.midi_active_region_count}</td></tr>
            <tr><th>audio_gap_count</th><td>{summary.audio_gap_count}</td></tr>
            <tr><th>midi_overhang_count</th><td>{summary.midi_overhang_count}</td></tr>
            <tr><th>working_midi_note_count</th><td>{summary.working_midi_note_count}</td></tr>
    <tr><th>aligned_count</th><td>{summary.aligned_count}</td></tr>
    <tr><th>keep_original_count</th><td>{summary.keep_original_count}</td></tr>
    <tr><th>review_timing_count</th><td>{summary.review_timing_count}</td></tr>
    <tr><th>no_audio_evidence_count</th><td>{summary.no_audio_evidence_count}</td></tr>
    <tr><th>median_abs_start_correction_ms</th><td>{summary.median_abs_start_correction_ms}</td></tr>
    <tr><th>p95_abs_start_correction_ms</th><td>{summary.p95_abs_start_correction_ms}</td></tr>
    <tr><th>max_abs_start_correction_ms</th><td>{summary.max_abs_start_correction_ms}</td></tr>
      <tr><th>mean_confidence</th><td>{summary.mean_confidence}</td></tr>
      <tr><th>min_confidence</th><td>{summary.min_confidence}</td></tr>
      <tr><th>max_confidence</th><td>{summary.max_confidence}</td></tr>
      <tr><th>low_confidence_count</th><td>{summary.low_confidence_count}</td></tr>
      <tr><th>weak_onset_count</th><td>{summary.weak_onset_count}</td></tr>
      <tr><th>low_rms_count</th><td>{summary.low_rms_count}</td></tr>
      <tr><th>warning_count</th><td>{summary.warning_count}</td></tr>
    </tbody>
  </table>

  <h2>Output Files</h2>
  <ul>{output_items}</ul>

    <h2>Counts by Action</h2>
    <p>Validation and cleanup action counts are summarized in the table above and in <code>qa_summary.json</code>.</p>

    <h2>Audio-Time Alignment / Sync</h2>
    <p>
        This section reports whether validation and export stages are using audio-aligned seconds,
        plus global offset and export timing precision metrics.
    </p>
    <table border='1' cellpadding='6' cellspacing='0'>
        <tbody>
            {sync_rows_html}
        </tbody>
    </table>

  {_rows_table_html(f"Top {top_n} Lowest-Confidence Notes", lowest_confidence)}
  {_rows_table_html(f"Top {top_n} Weak-Onset Notes", weakest_onset)}
  {_rows_table_html(f"Top {top_n} Low-RMS Notes", lowest_rms)}

  <h2>Warnings</h2>
  <ul>{warning_items}</ul>
</body>
</html>
"""

    path.write_text(html_doc, encoding="utf-8")


def generate_qa_report(
    project_dir: Path,
    output_dir: Path | None,
    params: QAReportParameters,
) -> QASummary:
    if not project_dir.exists() or not project_dir.is_dir():
        raise QAReportError(f"Project directory does not exist: {project_dir}")

    out_dir = output_dir if output_dir is not None else (project_dir / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    note_validation_path = project_dir / "analysis" / "note_validation.json"
    audio_aligned_notes_path = project_dir / "analysis" / "audio_aligned_note_events.json"
    audio_alignment_report_path = project_dir / "analysis" / "audio_alignment_report.json"
    refined_notes_path = project_dir / "analysis" / "refined_note_events.json"
    repaired_refined_notes_path = project_dir / "analysis" / "repaired_refined_note_events.json"
    bass_refinement_report_path = project_dir / "analysis" / "bass_refinement_report.json"
    dsp_analysis_report_path = project_dir / "analysis" / "audio_analysis_dsp_report.json"
    activity_repair_report_path = project_dir / "analysis" / "activity_repair_report.json"
    midi_audio_validation_report_path = project_dir / "analysis" / "midi_audio_validation_report.json"
    cleanup_plan_path = project_dir / "cleanup" / "cleanup_plan.json"
    cleaned_export_report_path = project_dir / "midi" / "cleaned" / "cleaned_export_report.json"
    review_export_report_path = project_dir / "midi" / "review" / "export_report.json"
    working_export_report_path = project_dir / "midi" / "working" / "working_export_report.json"
    pipeline_report_path = project_dir / "reports" / "pipeline_report.json"

    warnings: list[str] = []

    note_validation: NoteValidationDocument | None = None
    if note_validation_path.exists():
        note_validation = _load_note_validation(note_validation_path)
    else:
        warnings.append(f"Missing note validation report: {note_validation_path}")

    cleanup_plan: CleanupPlanDocument | None = None
    if cleanup_plan_path.exists():
        cleanup_plan = _load_cleanup_plan(cleanup_plan_path)
    else:
        warnings.append(f"Missing cleanup plan report: {cleanup_plan_path}")

    aligned_notes: AudioAlignedNoteDocument | None = None
    if audio_aligned_notes_path.exists():
        aligned_notes = _load_audio_aligned_document(audio_aligned_notes_path)
    else:
        warnings.append(f"Missing audio aligned notes: {audio_aligned_notes_path}")

    refined_notes: RefinedNoteDocument | None = None
    if refined_notes_path.exists():
        refined_notes = _load_refined_document(refined_notes_path)
    else:
        warnings.append(f"Missing refined notes: {refined_notes_path}")

    repaired_notes: RefinedNoteDocument | None = None
    if repaired_refined_notes_path.exists():
        repaired_notes = _load_refined_document(repaired_refined_notes_path)

    alignment_report: AudioAlignmentReport | None = None
    if audio_alignment_report_path.exists():
        alignment_report = _load_audio_alignment_report(audio_alignment_report_path)
    else:
        warnings.append(f"Missing audio alignment report: {audio_alignment_report_path}")

    refinement_report: BassRefinementReport | None = None
    if bass_refinement_report_path.exists():
        refinement_report = _load_refinement_report(bass_refinement_report_path)
    else:
        warnings.append(f"Missing bass refinement report: {bass_refinement_report_path}")

    if note_validation is None and cleanup_plan is None:
        raise QAReportError("Missing both required inputs: analysis/note_validation.json and cleanup/cleanup_plan.json")

    validation_timing_source: str | None = None
    if midi_audio_validation_report_path.exists():
        validation_report_payload = _read_json(midi_audio_validation_report_path)
        raw_validation_timing_source = validation_report_payload.get("timing_source")
        if isinstance(raw_validation_timing_source, str):
            validation_timing_source = raw_validation_timing_source
    else:
        warnings.append(f"Missing MIDI-audio validation report: {midi_audio_validation_report_path}")

    cleaned_note_count = 0
    rejected_note_count = 0
    working_midi_note_count = 0
    activity_repair_enabled = False
    repaired_note_count = 0
    repair_extend_count = 0
    repair_shorten_count = 0
    repair_insert_missing_count = 0
    repair_split_count = 0
    repair_close_gap_count = 0
    repair_review_manual_count = 0
    repair_sustain_protected_count = 0
    repair_pitch_protected_count = 0
    repair_legato_protected_count = 0
    repair_shorten_candidate_count = 0
    repair_shorten_applied_count = 0
    repair_shorten_rejected_count = 0
    audio_active_region_count = 0
    midi_active_region_count = 0
    audio_gap_count = 0
    midi_overhang_count = 0
    dsp_backend_name: str | None = None
    dsp_backend_available: bool | None = None
    dsp_frame_count = 0
    dsp_attack_rise_count = 0
    dsp_sustain_count = 0
    dsp_tail_count = 0
    dsp_silence_count = 0
    dsp_debug_csv_file: str | None = None
    cleaned_export_timing_source: str | None = None
    review_export_timing_source: str | None = None
    working_export_time_error_ms: float | None = None
    export_max_errors_ms: list[float] = []
    export_mean_errors_ms: list[float] = []

    if cleaned_export_report_path.exists():
        cleaned_export_report = _read_json(cleaned_export_report_path)
        cleaned_note_count = int(cleaned_export_report.get("cleaned_note_count", 0))
        rejected_note_count = int(cleaned_export_report.get("rejected_note_count", 0))

        raw_cleaned_timing_source = cleaned_export_report.get("timing_source")
        if isinstance(raw_cleaned_timing_source, str):
            cleaned_export_timing_source = raw_cleaned_timing_source

        raw_cleaned_max_error = cleaned_export_report.get("max_export_time_error_ms")
        if isinstance(raw_cleaned_max_error, (int, float)):
            export_max_errors_ms.append(float(raw_cleaned_max_error))

        raw_cleaned_mean_error = cleaned_export_report.get("mean_export_time_error_ms")
        if isinstance(raw_cleaned_mean_error, (int, float)):
            export_mean_errors_ms.append(float(raw_cleaned_mean_error))
    else:
        warnings.append(f"Missing cleaned export report: {cleaned_export_report_path}")

    if review_export_report_path.exists():
        review_export_report = _read_json(review_export_report_path)

        raw_review_timing_source = review_export_report.get("timing_source")
        if isinstance(raw_review_timing_source, str):
            review_export_timing_source = raw_review_timing_source

        raw_review_max_error = review_export_report.get("max_export_time_error_ms")
        if isinstance(raw_review_max_error, (int, float)):
            export_max_errors_ms.append(float(raw_review_max_error))

        raw_review_mean_error = review_export_report.get("mean_export_time_error_ms")
        if isinstance(raw_review_mean_error, (int, float)):
            export_mean_errors_ms.append(float(raw_review_mean_error))
    else:
        warnings.append(f"Missing review export report: {review_export_report_path}")

    if working_export_report_path.exists():
        working_export_report = _load_working_export_report(working_export_report_path)
        working_midi_note_count = int(working_export_report.working_note_count)
        working_export_time_error_ms = float(working_export_report.max_export_time_error_ms)
    else:
        warnings.append(f"Missing working export report: {working_export_report_path}")

    if activity_repair_report_path.exists():
        activity_repair_report = _load_activity_repair_report(activity_repair_report_path)
        activity_repair_enabled = True
        repaired_note_count = int(activity_repair_report.output_note_count)
        repair_extend_count = int(activity_repair_report.extend_count)
        repair_shorten_count = int(activity_repair_report.shorten_count)
        repair_insert_missing_count = int(activity_repair_report.insert_missing_count)
        repair_split_count = int(activity_repair_report.split_count)
        repair_close_gap_count = int(activity_repair_report.close_gap_count)
        repair_review_manual_count = int(activity_repair_report.review_manual_count)
        repair_sustain_protected_count = int(activity_repair_report.sustain_protected_count)
        repair_pitch_protected_count = int(activity_repair_report.pitch_protected_count)
        repair_legato_protected_count = int(activity_repair_report.legato_protected_count)
        repair_shorten_candidate_count = int(activity_repair_report.shorten_candidate_count)
        repair_shorten_applied_count = int(activity_repair_report.shorten_applied_count)
        repair_shorten_rejected_count = int(activity_repair_report.shorten_rejected_count)
        audio_active_region_count = int(activity_repair_report.audio_active_region_count)
        midi_active_region_count = int(activity_repair_report.midi_active_region_count)
        audio_gap_count = int(activity_repair_report.audio_gap_count)
        midi_overhang_count = int(activity_repair_report.midi_overhang_count)

    if dsp_analysis_report_path.exists():
        dsp_report_payload = _read_json(dsp_analysis_report_path)
        raw_backend_name = dsp_report_payload.get("backend_name")
        raw_backend_available = dsp_report_payload.get("backend_available")
        raw_frame_count = dsp_report_payload.get("frame_count")
        raw_attack_count = dsp_report_payload.get("attack_rise_count")
        raw_sustain_count = dsp_report_payload.get("sustain_count")
        raw_tail_count = dsp_report_payload.get("tail_count")
        raw_silence_count = dsp_report_payload.get("silence_count")
        raw_debug_csv = dsp_report_payload.get("debug_csv_file")

        if isinstance(raw_backend_name, str):
            dsp_backend_name = raw_backend_name
        if isinstance(raw_backend_available, bool):
            dsp_backend_available = raw_backend_available
        if isinstance(raw_frame_count, int):
            dsp_frame_count = raw_frame_count
        if isinstance(raw_attack_count, int):
            dsp_attack_rise_count = raw_attack_count
        if isinstance(raw_sustain_count, int):
            dsp_sustain_count = raw_sustain_count
        if isinstance(raw_tail_count, int):
            dsp_tail_count = raw_tail_count
        if isinstance(raw_silence_count, int):
            dsp_silence_count = raw_silence_count
        if isinstance(raw_debug_csv, str):
            dsp_debug_csv_file = raw_debug_csv

    if not pipeline_report_path.exists():
        warnings.append(f"Missing pipeline report: {pipeline_report_path}")
    else:
        pipeline_report = _load_pipeline_report(pipeline_report_path)
        warnings.extend([f"pipeline: {item}" for item in pipeline_report.warnings])

    rows = _build_rows(
        note_validation=note_validation,
        cleanup_plan=cleanup_plan,
        aligned_notes=aligned_notes,
        refined_notes=(repaired_notes if repaired_notes is not None else refined_notes),
        warnings=warnings,
    )

    keep_count = 0
    review_count = 0
    mute_count = 0
    delete_candidate_count = 0
    if cleanup_plan is not None:
        keep_count = sum(1 for item in cleanup_plan.actions if item.plan_action == "KEEP")
        review_count = sum(1 for item in cleanup_plan.actions if item.plan_action == "REVIEW")
        mute_count = sum(1 for item in cleanup_plan.actions if item.plan_action == "MUTE")
        delete_candidate_count = sum(
            1 for item in cleanup_plan.actions if item.plan_action == "DELETE_CANDIDATE"
        )

    aligned_count = 0
    keep_original_count = 0
    review_timing_count = 0
    no_audio_evidence_count = 0
    median_abs_start_correction_ms: float | None = None
    p95_abs_start_correction_ms: float | None = None
    max_abs_start_correction_ms: float | None = None
    global_offset_ms: float | None = None
    global_confidence: float | None = None
    global_offset_applied: bool | None = None

    if alignment_report is not None:
        aligned_count = alignment_report.aligned_count
        keep_original_count = alignment_report.keep_original_count
        review_timing_count = alignment_report.review_timing_count
        no_audio_evidence_count = alignment_report.no_audio_evidence_count
        median_abs_start_correction_ms = alignment_report.median_abs_start_correction_ms
        p95_abs_start_correction_ms = alignment_report.p95_abs_start_correction_ms
        max_abs_start_correction_ms = alignment_report.max_abs_start_correction_ms
        global_offset_ms = alignment_report.global_offset_ms
        global_confidence = alignment_report.global_confidence
        global_offset_applied = alignment_report.global_offset_applied
    elif aligned_notes is not None:
        aligned_count = sum(1 for item in aligned_notes.notes if item.alignment_action == "ALIGNED")
        keep_original_count = sum(
            1
            for item in aligned_notes.notes
            if item.alignment_action == "KEEP_ORIGINAL_LOW_CONFIDENCE"
        )
        review_timing_count = sum(
            1 for item in aligned_notes.notes if item.alignment_action == "REVIEW_TIMING"
        )
        no_audio_evidence_count = sum(
            1 for item in aligned_notes.notes if item.alignment_action == "NO_AUDIO_EVIDENCE"
        )
        abs_start_corrections = [
            abs(item.start_correction_ms)
            for item in aligned_notes.notes
            if item.alignment_action == "ALIGNED"
        ]
        if abs_start_corrections:
            sorted_corrections = sorted(abs_start_corrections)
            median_abs_start_correction_ms = sorted_corrections[len(sorted_corrections) // 2]
            p95_index = int((len(sorted_corrections) - 1) * 0.95)
            p95_abs_start_correction_ms = sorted_corrections[p95_index]
            max_abs_start_correction_ms = sorted_corrections[-1]

    refined_note_count = 0
    merged_count = 0
    false_retrigger_merge_count = 0
    tail_extended_count = 0
    short_note_extended_count = 0
    overlap_resolved_count = 0
    median_start_refinement_ms: float | None = None
    median_end_refinement_ms: float | None = None

    if refinement_report is not None:
        refined_note_count = refinement_report.output_note_count
        merged_count = refinement_report.merged_count
        false_retrigger_merge_count = refinement_report.false_retrigger_merge_count
        tail_extended_count = refinement_report.tail_extended_count
        short_note_extended_count = refinement_report.short_note_extended_count
        overlap_resolved_count = refinement_report.overlap_resolved_count
        median_start_refinement_ms = refinement_report.median_start_refinement_ms
        median_end_refinement_ms = refinement_report.median_end_refinement_ms
    elif refined_notes is not None:
        refined_note_count = len(refined_notes.notes)
        for item in refined_notes.notes:
            actions = set(item.refinement_actions)
            if "FALSE_RETRIGGER_MERGED" in actions:
                false_retrigger_merge_count += 1
            if "SUSTAIN_TAIL_EXTENDED" in actions:
                tail_extended_count += 1
            if "SHORT_NOTE_EXTENDED" in actions:
                short_note_extended_count += 1
            if "MONOPHONIC_OVERLAP_RESOLVED" in actions:
                overlap_resolved_count += 1
            merged_count += len(item.merged_note_ids)

        if refined_notes.notes:
            median_start_refinement_ms = float(
                median(item.start_refinement_ms for item in refined_notes.notes)
            )
            median_end_refinement_ms = float(
                median(item.end_refinement_ms for item in refined_notes.notes)
            )

    if repaired_notes is not None and repaired_note_count == 0:
        repaired_note_count = len(repaired_notes.notes)
        activity_repair_enabled = True

    max_export_time_error_ms = max(export_max_errors_ms) if export_max_errors_ms else None
    mean_export_time_error_ms = (
        (sum(export_mean_errors_ms) / len(export_mean_errors_ms))
        if export_mean_errors_ms
        else None
    )

    confidences = [row.confidence for row in rows]
    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    min_confidence = min(confidences) if confidences else None
    max_confidence = max(confidences) if confidences else None

    review_threshold = 0.45
    minimum_onset_score = 0.01
    minimum_rms = 0.001
    if note_validation is not None:
        review_threshold = float(note_validation.validation_parameters.get("review_threshold", 0.45))
        minimum_onset_score = float(note_validation.validation_parameters.get("minimum_onset_score", 0.01))
        minimum_rms = float(note_validation.validation_parameters.get("minimum_rms", 0.001))

    low_confidence_count = sum(1 for row in rows if row.confidence < review_threshold)
    weak_onset_count = sum(1 for row in rows if row.onset_score < minimum_onset_score)
    low_rms_count = sum(1 for row in rows if row.mean_rms_during_note < minimum_rms)

    layer = None
    if note_validation is not None:
        layer = note_validation.layer
    elif cleanup_plan is not None:
        layer = cleanup_plan.layer

    output_files: dict[str, str] = {}

    qa_summary_path = out_dir / "qa_summary.json"
    qa_csv_path = out_dir / "qa_notes.csv"
    qa_html_path = out_dir / "qa_report.html"

    summary = QASummary(
        status="ok",
        project_dir=str(project_dir),
        layer=layer,
        total_notes=len(rows) if rows else (len(cleanup_plan.actions) if cleanup_plan is not None else 0),
        keep_count=keep_count,
        review_count=review_count,
        mute_count=mute_count,
        delete_candidate_count=delete_candidate_count,
        cleaned_note_count=cleaned_note_count,
        rejected_note_count=rejected_note_count,
        refined_note_count=refined_note_count,
        merged_count=merged_count,
        false_retrigger_merge_count=false_retrigger_merge_count,
        tail_extended_count=tail_extended_count,
        short_note_extended_count=short_note_extended_count,
        overlap_resolved_count=overlap_resolved_count,
        median_start_refinement_ms=median_start_refinement_ms,
        median_end_refinement_ms=median_end_refinement_ms,
        dsp_backend_name=dsp_backend_name,
        dsp_backend_available=dsp_backend_available,
        dsp_frame_count=dsp_frame_count,
        dsp_attack_rise_count=dsp_attack_rise_count,
        dsp_sustain_count=dsp_sustain_count,
        dsp_tail_count=dsp_tail_count,
        dsp_silence_count=dsp_silence_count,
        dsp_debug_csv_file=dsp_debug_csv_file,
        activity_repair_enabled=activity_repair_enabled,
        repaired_note_count=repaired_note_count,
        repair_extend_count=repair_extend_count,
        repair_shorten_count=repair_shorten_count,
        repair_insert_missing_count=repair_insert_missing_count,
        repair_split_count=repair_split_count,
        repair_close_gap_count=repair_close_gap_count,
        repair_review_manual_count=repair_review_manual_count,
        repair_sustain_protected_count=repair_sustain_protected_count,
        repair_pitch_protected_count=repair_pitch_protected_count,
        repair_legato_protected_count=repair_legato_protected_count,
        repair_shorten_candidate_count=repair_shorten_candidate_count,
        repair_shorten_applied_count=repair_shorten_applied_count,
        repair_shorten_rejected_count=repair_shorten_rejected_count,
        audio_active_region_count=audio_active_region_count,
        midi_active_region_count=midi_active_region_count,
        audio_gap_count=audio_gap_count,
        midi_overhang_count=midi_overhang_count,
        working_midi_note_count=working_midi_note_count,
        working_export_time_error_ms=working_export_time_error_ms,
        validation_timing_source=validation_timing_source,
        review_export_timing_source=review_export_timing_source,
        cleaned_export_timing_source=cleaned_export_timing_source,
        global_offset_ms=global_offset_ms,
        global_confidence=global_confidence,
        global_offset_applied=global_offset_applied,
        aligned_count=aligned_count,
        keep_original_count=keep_original_count,
        review_timing_count=review_timing_count,
        no_audio_evidence_count=no_audio_evidence_count,
        median_abs_start_correction_ms=median_abs_start_correction_ms,
        p95_abs_start_correction_ms=p95_abs_start_correction_ms,
        max_abs_start_correction_ms=max_abs_start_correction_ms,
        max_export_time_error_ms=max_export_time_error_ms,
        mean_export_time_error_ms=mean_export_time_error_ms,
        mean_confidence=mean_confidence,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        low_confidence_count=low_confidence_count,
        weak_onset_count=weak_onset_count,
        low_rms_count=low_rms_count,
        output_files=output_files,
        warning_count=len(warnings),
        warnings=warnings,
    )

    qa_summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    output_files["qa_summary"] = str(qa_summary_path)

    if params.include_csv:
        _write_csv(qa_csv_path, rows)
        output_files["qa_notes_csv"] = str(qa_csv_path)

    if params.include_html:
        _write_html(qa_html_path, summary=summary, rows=rows, top_n=params.top_n)
        output_files["qa_report_html"] = str(qa_html_path)

    summary.output_files = output_files
    qa_summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")

    return summary
