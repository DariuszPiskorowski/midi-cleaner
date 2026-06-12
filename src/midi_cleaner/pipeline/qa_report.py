from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

from midi_cleaner.cleanup.models import CleanupPlanDocument
from midi_cleaner.pipeline.models import PipelineReport, QANoteRow, QASummary
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


def _load_pipeline_report(path: Path) -> PipelineReport:
    return PipelineReport.model_validate_json(path.read_text(encoding="utf-8"))


def _build_rows(
    note_validation: NoteValidationDocument | None,
    cleanup_plan: CleanupPlanDocument | None,
    warnings: list[str],
) -> list[QANoteRow]:
    if note_validation is None:
        return []

    plan_action_by_note_id: dict[str, str] = {}
    if cleanup_plan is not None:
        plan_action_by_note_id = {
            item.note_id: item.plan_action for item in cleanup_plan.actions
        }

    validation_note_ids = {item.note_id for item in note_validation.validations}
    for note_id in plan_action_by_note_id:
        if note_id not in validation_note_ids:
            warnings.append(f"cleanup plan contains unknown note_id: {note_id}")

    rows: list[QANoteRow] = []
    for item in note_validation.validations:
        plan_action = plan_action_by_note_id.get(item.note_id)
        if cleanup_plan is not None and plan_action is None:
            warnings.append(f"validation note has no cleanup plan action: {item.note_id}")

        rows.append(
            QANoteRow(
                note_id=item.note_id,
                pitch_midi=item.pitch_midi,
                pitch_name=item.pitch_name,
                start_sec=item.start_sec,
                end_sec=item.end_sec,
                duration_sec=item.duration_sec,
                confidence=item.confidence,
                validation_action=item.recommended_action,
                plan_action=plan_action,
                onset_score=item.onset_score,
                mean_rms_during_note=item.mean_rms_during_note,
                sustained_energy_ratio=item.sustained_energy_ratio,
                reasons="; ".join(item.reasons),
            )
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
        html_rows.append(
            "<tr>"
            f"<td>{escape(row.note_id)}</td>"
            f"<td>{row.pitch_midi}</td>"
            f"<td>{escape(row.pitch_name)}</td>"
            f"<td>{row.confidence:.4f}</td>"
            f"<td>{escape(row.validation_action)}</td>"
            f"<td>{escape(str(row.plan_action))}</td>"
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
        "<th>validation_action</th><th>plan_action</th><th>onset_score</th>"
        "<th>mean_rms</th><th>sustained_ratio</th><th>reasons</th>"
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
    cleanup_plan_path = project_dir / "cleanup" / "cleanup_plan.json"
    cleaned_export_report_path = project_dir / "midi" / "cleaned" / "cleaned_export_report.json"
    review_export_report_path = project_dir / "midi" / "review" / "export_report.json"
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

    if note_validation is None and cleanup_plan is None:
        raise QAReportError("Missing both required inputs: analysis/note_validation.json and cleanup/cleanup_plan.json")

    cleaned_note_count = 0
    rejected_note_count = 0
    if cleaned_export_report_path.exists():
        cleaned_export_report = _read_json(cleaned_export_report_path)
        cleaned_note_count = int(cleaned_export_report.get("cleaned_note_count", 0))
        rejected_note_count = int(cleaned_export_report.get("rejected_note_count", 0))
    else:
        warnings.append(f"Missing cleaned export report: {cleaned_export_report_path}")

    if not review_export_report_path.exists():
        warnings.append(f"Missing review export report: {review_export_report_path}")

    if not pipeline_report_path.exists():
        warnings.append(f"Missing pipeline report: {pipeline_report_path}")
    else:
        pipeline_report = _load_pipeline_report(pipeline_report_path)
        warnings.extend([f"pipeline: {item}" for item in pipeline_report.warnings])

    rows = _build_rows(note_validation=note_validation, cleanup_plan=cleanup_plan, warnings=warnings)

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
