from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from pydantic import BaseModel

from midi_cleaner.alignment.audio_time import (
    AudioTimeAlignmentParameters,
    align_notes_to_audio_time,
)
from midi_cleaner.audio.analyzer import analyze_stem
from midi_cleaner.cleanup.cleaned_exporter import (
    CleanedMidiExportParameters,
    export_cleaned_midi,
)
from midi_cleaner.cleanup.midi_exporter import ReviewMidiExportParameters, export_review_midi
from midi_cleaner.cleanup.planner import CleanupPlannerParameters, build_cleanup_plan
from midi_cleaner.midi.importer import import_midi_candidate
from midi_cleaner.pipeline.models import PipelineReport, PipelineStageReport
from midi_cleaner.validation.midi_audio import ValidationParameters, validate_midi_vs_audio


class PipelineProcessError(Exception):
    """Raised when process-stem pipeline fails."""


@dataclass(frozen=True)
class PipelineProcessParameters:
    onset_window_ms: float = 50.0
    minimum_rms: float = 0.001
    minimum_onset_score: float = 0.01
    review_threshold: float = 0.45
    keep_threshold: float = 0.70
    onset_search_window_ms: float = 250.0
    offset_search_window_ms: float = 350.0
    alignment_min_onset_score: float = 0.005
    alignment_min_rms: float = 0.001
    snap_start_to_audio_onset: bool = True
    snap_end_to_energy_offset: bool = True
    max_start_correction_ms: float = 500.0
    max_end_correction_ms: float = 800.0
    low_confidence_action: str = "KEEP_ORIGINAL_LOW_CONFIDENCE"
    mute_threshold: float = 0.45
    cleanup_review_threshold: float = 0.70
    delete_threshold: float = 0.20
    allow_delete_candidates: bool = False
    ticks_per_beat: int | None = None
    track_name_prefix: str = "Hermes"
    include_review_in_cleaned: bool = False
    write_empty_files: bool = True
    include_delete_candidates: bool = True


def _write_json(path: Path, payload: BaseModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        path.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_json_dict(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def process_stem_pipeline(
    input_midi: Path,
    input_wav: Path,
    source: str,
    layer: str,
    project_dir: Path,
    params: PipelineProcessParameters,
) -> PipelineReport:
    input_dir = project_dir / "input"
    analysis_dir = project_dir / "analysis"
    cleanup_dir = project_dir / "cleanup"
    review_midi_dir = project_dir / "midi" / "review"
    cleaned_midi_dir = project_dir / "midi" / "cleaned"
    reports_dir = project_dir / "reports"
    pipeline_report_path = reports_dir / "pipeline_report.json"

    for directory in [
        input_dir,
        analysis_dir,
        cleanup_dir,
        review_midi_dir,
        cleaned_midi_dir,
        reports_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    source_paths_file = input_dir / "source_paths.json"
    _write_json_dict(
        source_paths_file,
        {
            "input_midi": str(input_midi.resolve()),
            "input_wav": str(input_wav.resolve()),
            "source": source,
            "layer": layer,
            "parameters": {
                "validation": {
                    "onset_window_ms": params.onset_window_ms,
                    "minimum_rms": params.minimum_rms,
                    "minimum_onset_score": params.minimum_onset_score,
                    "review_threshold": params.review_threshold,
                    "keep_threshold": params.keep_threshold,
                    "timing_source": "audio_aligned_seconds",
                    "audio_aligned_notes_file": str(analysis_dir / "audio_aligned_note_events.json"),
                },
                "cleanup": {
                    "mute_threshold": params.mute_threshold,
                    "review_threshold": params.cleanup_review_threshold,
                    "delete_threshold": params.delete_threshold,
                    "allow_delete_candidates": params.allow_delete_candidates,
                },
                "alignment": {
                    "onset_search_window_ms": params.onset_search_window_ms,
                    "offset_search_window_ms": params.offset_search_window_ms,
                    "min_onset_score": params.alignment_min_onset_score,
                    "min_rms": params.alignment_min_rms,
                    "snap_start_to_audio_onset": params.snap_start_to_audio_onset,
                    "snap_end_to_energy_offset": params.snap_end_to_energy_offset,
                    "max_start_correction_ms": params.max_start_correction_ms,
                    "max_end_correction_ms": params.max_end_correction_ms,
                    "low_confidence_action": params.low_confidence_action,
                },
                "midi_export": {
                    "ticks_per_beat": params.ticks_per_beat,
                    "track_name_prefix": params.track_name_prefix,
                    "include_review_in_cleaned": params.include_review_in_cleaned,
                    "write_empty_files": params.write_empty_files,
                    "include_delete_candidates": params.include_delete_candidates,
                },
            },
        },
    )

    output_files: dict[str, str] = {
        "source_paths": str(source_paths_file),
        "pipeline_report": str(pipeline_report_path),
    }
    stages: list[PipelineStageReport] = []
    warnings: list[str] = []
    current_stage = "initialization"

    try:
        # Stage 1: MIDI import
        current_stage = "midi_import"
        note_events_path = analysis_dir / "note_events.json"
        midi_import_report_path = analysis_dir / "midi_import_report.json"
        note_document, midi_import_report = import_midi_candidate(input_midi, source=source, layer=layer)
        midi_import_report.output_file = str(note_events_path)
        _write_json(note_events_path, note_document)
        _write_json(midi_import_report_path, midi_import_report)
        output_files["note_events"] = str(note_events_path)
        output_files["midi_import_report"] = str(midi_import_report_path)
        stages.append(
            PipelineStageReport(
                name="midi_import",
                status="ok",
                output_files=[str(note_events_path), str(midi_import_report_path)],
                warning_count=midi_import_report.warning_count,
                warnings=midi_import_report.warnings,
            )
        )
        warnings.extend([f"midi_import: {item}" for item in midi_import_report.warnings])

        # Stage 2: WAV analysis
        current_stage = "audio_analysis"
        audio_features_path = analysis_dir / "audio_features.json"
        audio_analysis_report_path = analysis_dir / "audio_analysis_report.json"
        audio_document, audio_report = analyze_stem(input_wav=input_wav, layer=layer)
        audio_report.output_file = str(audio_features_path)
        _write_json(audio_features_path, audio_document)
        _write_json(audio_analysis_report_path, audio_report)
        output_files["audio_features"] = str(audio_features_path)
        output_files["audio_analysis_report"] = str(audio_analysis_report_path)
        stages.append(
            PipelineStageReport(
                name="audio_analysis",
                status="ok",
                output_files=[str(audio_features_path), str(audio_analysis_report_path)],
                warning_count=audio_report.warning_count,
                warnings=audio_report.warnings,
            )
        )
        warnings.extend([f"audio_analysis: {item}" for item in audio_report.warnings])

        # Stage 3: Audio-time note alignment
        current_stage = "audio_time_alignment"
        audio_aligned_note_events_path = analysis_dir / "audio_aligned_note_events.json"
        audio_alignment_report_path = analysis_dir / "audio_alignment_report.json"
        aligned_document, alignment_report = align_notes_to_audio_time(
            notes_file=note_events_path,
            audio_features_file=audio_features_path,
            params=AudioTimeAlignmentParameters(
                onset_search_window_ms=params.onset_search_window_ms,
                offset_search_window_ms=params.offset_search_window_ms,
                min_onset_score=params.alignment_min_onset_score,
                min_rms=params.alignment_min_rms,
                snap_start_to_audio_onset=params.snap_start_to_audio_onset,
                snap_end_to_energy_offset=params.snap_end_to_energy_offset,
                max_start_correction_ms=params.max_start_correction_ms,
                max_end_correction_ms=params.max_end_correction_ms,
                low_confidence_action=params.low_confidence_action,
            ),
        )
        alignment_report.output_file = str(audio_aligned_note_events_path)
        _write_json(audio_aligned_note_events_path, aligned_document)
        _write_json(audio_alignment_report_path, alignment_report)
        output_files["audio_aligned_note_events"] = str(audio_aligned_note_events_path)
        output_files["audio_alignment_report"] = str(audio_alignment_report_path)
        stages.append(
            PipelineStageReport(
                name="audio_time_alignment",
                status="ok",
                output_files=[str(audio_aligned_note_events_path), str(audio_alignment_report_path)],
                warning_count=alignment_report.warning_count,
                warnings=alignment_report.warnings,
            )
        )
        warnings.extend([f"audio_time_alignment: {item}" for item in alignment_report.warnings])

        # Stage 4: MIDI-vs-audio validation
        current_stage = "midi_audio_validation"
        note_validation_path = analysis_dir / "note_validation.json"
        midi_audio_validation_report_path = analysis_dir / "midi_audio_validation_report.json"
        validation_document, validation_report = validate_midi_vs_audio(
            notes_file=note_events_path,
            audio_features_file=audio_features_path,
            audio_aligned_notes_file=audio_aligned_note_events_path,
            params=ValidationParameters(
                onset_window_ms=params.onset_window_ms,
                minimum_rms=params.minimum_rms,
                minimum_onset_score=params.minimum_onset_score,
                review_threshold=params.review_threshold,
                keep_threshold=params.keep_threshold,
            ),
        )
        validation_report.output_file = str(note_validation_path)
        _write_json(note_validation_path, validation_document)
        _write_json(midi_audio_validation_report_path, validation_report)
        output_files["note_validation"] = str(note_validation_path)
        output_files["midi_audio_validation_report"] = str(midi_audio_validation_report_path)
        output_files["validation_timing_source"] = validation_report.timing_source
        if validation_report.audio_aligned_notes_file is not None:
            output_files["validation_audio_aligned_notes_file"] = (
                validation_report.audio_aligned_notes_file
            )
        stages.append(
            PipelineStageReport(
                name="midi_audio_validation",
                status="ok",
                output_files=[str(note_validation_path), str(midi_audio_validation_report_path)],
                warning_count=validation_report.warning_count,
                warnings=validation_report.warnings,
            )
        )
        warnings.extend([f"midi_audio_validation: {item}" for item in validation_report.warnings])

        # Stage 5: Cleanup planning
        current_stage = "cleanup_plan"
        cleanup_plan_path = cleanup_dir / "cleanup_plan.json"
        cleanup_plan_report_path = cleanup_dir / "cleanup_plan_report.json"
        cleanup_document, cleanup_report = build_cleanup_plan(
            validation_file=note_validation_path,
            params=CleanupPlannerParameters(
                mute_threshold=params.mute_threshold,
                review_threshold=params.cleanup_review_threshold,
                delete_threshold=params.delete_threshold,
                allow_delete_candidates=params.allow_delete_candidates,
            ),
        )
        cleanup_report.output_file = str(cleanup_plan_path)
        _write_json(cleanup_plan_path, cleanup_document)
        _write_json(cleanup_plan_report_path, cleanup_report)
        output_files["cleanup_plan"] = str(cleanup_plan_path)
        output_files["cleanup_plan_report"] = str(cleanup_plan_report_path)
        stages.append(
            PipelineStageReport(
                name="cleanup_plan",
                status="ok",
                output_files=[str(cleanup_plan_path), str(cleanup_plan_report_path)],
                warning_count=cleanup_report.warning_count,
                warnings=cleanup_report.warnings,
            )
        )
        warnings.extend([f"cleanup_plan: {item}" for item in cleanup_report.warnings])

        # Stage 6: Review MIDI export
        current_stage = "review_midi_export"
        review_export_report_path = review_midi_dir / "export_report.json"
        review_export_report = export_review_midi(
            notes_file=note_events_path,
            cleanup_plan_file=cleanup_plan_path,
            output_dir=review_midi_dir,
            params=ReviewMidiExportParameters(
                ticks_per_beat=params.ticks_per_beat,
                track_name_prefix=params.track_name_prefix,
                include_delete_candidates=params.include_delete_candidates,
                audio_aligned_notes_file=audio_aligned_note_events_path,
            ),
        )
        _write_json(review_export_report_path, review_export_report)
        output_files["review_export_report"] = str(review_export_report_path)
        review_outputs = [item.path for item in review_export_report.exported_files] + [
            str(review_export_report_path)
        ]
        stages.append(
            PipelineStageReport(
                name="review_midi_export",
                status="ok",
                output_files=review_outputs,
                warning_count=review_export_report.warning_count,
                warnings=review_export_report.warnings,
            )
        )
        warnings.extend([f"review_midi_export: {item}" for item in review_export_report.warnings])

        # Stage 7: Cleaned MIDI export
        current_stage = "cleaned_midi_export"
        cleaned_export_report_path = cleaned_midi_dir / "cleaned_export_report.json"
        cleaned_export_report = export_cleaned_midi(
            notes_file=note_events_path,
            cleanup_plan_file=cleanup_plan_path,
            output_dir=cleaned_midi_dir,
            params=CleanedMidiExportParameters(
                ticks_per_beat=params.ticks_per_beat,
                track_name_prefix=params.track_name_prefix,
                include_review_in_cleaned=params.include_review_in_cleaned,
                write_empty_files=params.write_empty_files,
                audio_aligned_notes_file=audio_aligned_note_events_path,
            ),
        )
        _write_json(cleaned_export_report_path, cleaned_export_report)
        output_files["cleaned_export_report"] = str(cleaned_export_report_path)
        cleaned_outputs = [item.path for item in cleaned_export_report.exported_files] + [
            str(cleaned_export_report_path)
        ]
        stages.append(
            PipelineStageReport(
                name="cleaned_midi_export",
                status="ok",
                output_files=cleaned_outputs,
                warning_count=cleaned_export_report.warning_count,
                warnings=cleaned_export_report.warnings,
            )
        )
        warnings.extend([f"cleaned_midi_export: {item}" for item in cleaned_export_report.warnings])

        pipeline_report = PipelineReport(
            status="ok",
            input_midi=str(input_midi),
            input_wav=str(input_wav),
            source=source,
            layer=layer,
            project_dir=str(project_dir),
            stages=stages,
            output_files=output_files,
            warning_count=len(warnings),
            warnings=warnings,
        )
        _write_json(pipeline_report_path, pipeline_report)
        return pipeline_report

    except Exception as exc:
        error_stage = PipelineStageReport(
            name=current_stage,
            status="error",
            output_files=[],
            warning_count=1,
            warnings=[str(exc)],
        )
        stages.append(error_stage)
        warnings.append(f"pipeline_error: {exc}")

        error_report = PipelineReport(
            status="error",
            input_midi=str(input_midi),
            input_wav=str(input_wav),
            source=source,
            layer=layer,
            project_dir=str(project_dir),
            stages=stages,
            output_files=output_files,
            warning_count=len(warnings),
            warnings=warnings,
        )
        _write_json(pipeline_report_path, error_report)
        raise PipelineProcessError(str(exc)) from exc
