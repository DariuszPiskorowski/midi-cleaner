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
from midi_cleaner.cleanup.working_exporter import (
    WorkingMidiExportParameters,
    export_working_midi,
)
from midi_cleaner.cleanup.planner import CleanupPlannerParameters, build_cleanup_plan
from midi_cleaner.dsp.analyzer import DspAnalysisError, analyze_dsp_stem
from midi_cleaner.midi.importer import import_midi_candidate
from midi_cleaner.pipeline.models import PipelineReport, PipelineStageReport
from midi_cleaner.repair.activity import (
    ActivityRepairParameters,
    repair_activity,
)
from midi_cleaner.refinement.bass import BassRefinementParameters, refine_bass_notes
from midi_cleaner.refinement.models import BassRefinementReport, RefinedNoteDocument, RefinedNoteEvent
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
    enable_bass_refinement: bool = True
    attack_lookback_ms: float = 80.0
    max_attack_advance_ms: float = 80.0
    merge_gap_ms: float = 160.0
    minimum_silence_ms: float = 80.0
    tail_rms_ratio: float = 0.20
    tail_silence_hold_ms: float = 120.0
    max_tail_extension_ms: float = 900.0
    minimum_note_duration_ms: float = 80.0
    monophonic: bool = True
    include_diagnostic_working_midi: bool = False
    enable_dsp_analysis: bool = True
    require_dsp_analysis: bool = False
    dsp_backend: str = "auto"
    dsp_debug_csv: bool = True
    enable_activity_repair: bool = True
    audio_active_threshold_ratio: float = 0.18
    audio_silence_hold_ms: float = 120.0
    missing_gap_min_ms: float = 80.0
    overhang_min_ms: float = 120.0
    split_min_note_duration_ms: float = 500.0
    close_gap_ms: float = 50.0
    insert_auto_confidence: float = 0.80
    split_auto_confidence: float = 0.75


def _write_json(path: Path, payload: BaseModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        path.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_json_dict(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_passthrough_refinement(
    aligned_document,
    aligned_notes_file: Path,
    audio_features_file: Path,
    validation_file: Path,
    reason: str,
    params: PipelineProcessParameters,
) -> tuple[RefinedNoteDocument, BassRefinementReport]:
    refined_notes: list[RefinedNoteEvent] = []
    for note in aligned_document.notes:
        refined_notes.append(
            RefinedNoteEvent(
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
                refined_start_sec=note.aligned_start_sec,
                refined_end_sec=note.aligned_end_sec,
                refined_duration_sec=note.aligned_duration_sec,
                start_refinement_ms=0.0,
                end_refinement_ms=0.0,
                merged_note_ids=[],
                refinement_actions=["UNCHANGED"],
                refinement_confidence=note.alignment_confidence,
                reasons=[reason],
            )
        )

    document = RefinedNoteDocument(
        schema_version="0.1.0",
        aligned_notes_file=str(aligned_notes_file),
        audio_features_file=str(audio_features_file),
        validation_file=str(validation_file),
        layer=aligned_document.layer,
        sample_rate=aligned_document.sample_rate,
        audio_duration_sec=aligned_document.audio_duration_sec,
        timing_source="refined_audio_seconds",
        refinement_parameters={
            "attack_lookback_ms": params.attack_lookback_ms,
            "max_attack_advance_ms": params.max_attack_advance_ms,
            "merge_gap_ms": params.merge_gap_ms,
            "minimum_silence_ms": params.minimum_silence_ms,
            "tail_rms_ratio": params.tail_rms_ratio,
            "tail_silence_hold_ms": params.tail_silence_hold_ms,
            "max_tail_extension_ms": params.max_tail_extension_ms,
            "minimum_note_duration_ms": params.minimum_note_duration_ms,
            "monophonic": params.monophonic,
            "allow_pitch_overlap": False,
        },
        notes=refined_notes,
    )

    report = BassRefinementReport(
        aligned_notes_file=str(aligned_notes_file),
        audio_features_file=str(audio_features_file),
        validation_file=str(validation_file),
        status="ok",
        layer=aligned_document.layer,
        input_note_count=len(aligned_document.notes),
        output_note_count=len(refined_notes),
        merged_count=0,
        false_retrigger_merge_count=0,
        tail_extended_count=0,
        short_note_extended_count=0,
        overlap_resolved_count=0,
        median_start_refinement_ms=0.0,
        median_end_refinement_ms=0.0,
        max_tail_extension_ms=0.0,
        warning_count=1,
        warnings=[reason],
        output_file=None,
    )
    return document, report


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
    working_midi_dir = project_dir / "midi" / "working"
    reports_dir = project_dir / "reports"
    pipeline_report_path = reports_dir / "pipeline_report.json"

    for directory in [
        input_dir,
        analysis_dir,
        cleanup_dir,
        review_midi_dir,
        cleaned_midi_dir,
        working_midi_dir,
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
                "dsp_analysis": {
                    "enable_dsp_analysis": params.enable_dsp_analysis,
                    "require_dsp_analysis": params.require_dsp_analysis,
                    "dsp_backend": params.dsp_backend,
                    "dsp_debug_csv": params.dsp_debug_csv,
                    "dsp_features_file": str(analysis_dir / "audio_features_dsp.json"),
                },
                "refinement": {
                    "enable_bass_refinement": params.enable_bass_refinement,
                    "attack_lookback_ms": params.attack_lookback_ms,
                    "max_attack_advance_ms": params.max_attack_advance_ms,
                    "merge_gap_ms": params.merge_gap_ms,
                    "minimum_silence_ms": params.minimum_silence_ms,
                    "tail_rms_ratio": params.tail_rms_ratio,
                    "tail_silence_hold_ms": params.tail_silence_hold_ms,
                    "max_tail_extension_ms": params.max_tail_extension_ms,
                    "minimum_note_duration_ms": params.minimum_note_duration_ms,
                    "monophonic": params.monophonic,
                },
                "activity_repair": {
                    "enable_activity_repair": params.enable_activity_repair,
                    "audio_active_threshold_ratio": params.audio_active_threshold_ratio,
                    "audio_silence_hold_ms": params.audio_silence_hold_ms,
                    "missing_gap_min_ms": params.missing_gap_min_ms,
                    "overhang_min_ms": params.overhang_min_ms,
                    "split_min_note_duration_ms": params.split_min_note_duration_ms,
                    "close_gap_ms": params.close_gap_ms,
                    "insert_auto_confidence": params.insert_auto_confidence,
                    "split_auto_confidence": params.split_auto_confidence,
                    "repaired_refined_notes_file": str(
                        analysis_dir / "repaired_refined_note_events.json"
                    ),
                    "activity_repair_plan_file": str(analysis_dir / "activity_repair_plan.json"),
                    "activity_repair_report_file": str(
                        analysis_dir / "activity_repair_report.json"
                    ),
                },
                "midi_export": {
                    "ticks_per_beat": params.ticks_per_beat,
                    "track_name_prefix": params.track_name_prefix,
                    "include_review_in_cleaned": params.include_review_in_cleaned,
                    "write_empty_files": params.write_empty_files,
                    "include_delete_candidates": params.include_delete_candidates,
                    "include_diagnostic_working_midi": params.include_diagnostic_working_midi,
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
    dsp_features_path: Path | None = None
    repaired_refined_note_events_path: Path | None = None
    activity_repair_plan_path: Path | None = None
    activity_repair_report_path: Path | None = None
    activity_repair_summary: dict[str, int] = {
        "extend_count": 0,
        "shorten_count": 0,
        "insert_missing_count": 0,
        "split_count": 0,
        "close_gap_count": 0,
        "review_manual_count": 0,
    }

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

        # Stage 3: DSP-backed audio analysis (optional, with graceful fallback)
        current_stage = "dsp_analysis"
        dsp_features_path = analysis_dir / "audio_features_dsp.json"
        dsp_analysis_report_path = analysis_dir / "audio_analysis_dsp_report.json"
        dsp_debug_csv_path = (
            analysis_dir / "audio_features_dsp_debug.csv" if params.dsp_debug_csv else None
        )

        if params.enable_dsp_analysis:
            try:
                dsp_document, dsp_report = analyze_dsp_stem(
                    wav_file=input_wav,
                    layer=layer,
                    backend=params.dsp_backend,
                    allow_backend_fallback=not params.require_dsp_analysis,
                    debug_csv_path=dsp_debug_csv_path,
                )
            except DspAnalysisError as exc:
                if params.require_dsp_analysis:
                    raise
                dsp_features_path = None
                warnings.append(f"dsp_analysis: {exc}")
                stages.append(
                    PipelineStageReport(
                        name="dsp_analysis",
                        status="ok",
                        output_files=[],
                        warning_count=1,
                        warnings=[str(exc)],
                    )
                )
            else:
                dsp_report.output_file = str(dsp_features_path)
                if dsp_debug_csv_path is not None:
                    dsp_report.debug_csv_file = str(dsp_debug_csv_path)
                _write_json(dsp_features_path, dsp_document)
                _write_json(dsp_analysis_report_path, dsp_report)
                output_files["audio_features_dsp"] = str(dsp_features_path)
                output_files["audio_analysis_dsp_report"] = str(dsp_analysis_report_path)
                if dsp_debug_csv_path is not None:
                    output_files["audio_features_dsp_debug_csv"] = str(dsp_debug_csv_path)
                stages.append(
                    PipelineStageReport(
                        name="dsp_analysis",
                        status="ok",
                        output_files=[
                            str(dsp_features_path),
                            str(dsp_analysis_report_path),
                            *([str(dsp_debug_csv_path)] if dsp_debug_csv_path is not None else []),
                        ],
                        warning_count=dsp_report.warning_count,
                        warnings=dsp_report.warnings,
                    )
                )
                warnings.extend([f"dsp_analysis: {item}" for item in dsp_report.warnings])
        else:
            dsp_features_path = None
            stages.append(
                PipelineStageReport(
                    name="dsp_analysis",
                    status="ok",
                    output_files=[],
                    warning_count=0,
                    warnings=[],
                )
            )

        # Stage 4: Audio-time note alignment
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

        # Stage 5: MIDI-vs-audio validation
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

        # Stage 6: Bass refinement (or passthrough for non-bass/disabled)
        current_stage = "bass_refinement"
        refined_note_events_path = analysis_dir / "refined_note_events.json"
        bass_refinement_report_path = analysis_dir / "bass_refinement_report.json"
        refinement_enabled = params.enable_bass_refinement and layer.lower() == "bass"

        if refinement_enabled:
            refined_document, refinement_report = refine_bass_notes(
                aligned_notes_file=audio_aligned_note_events_path,
                audio_features_file=audio_features_path,
                validation_file=note_validation_path,
                params=BassRefinementParameters(
                    attack_lookback_ms=params.attack_lookback_ms,
                    max_attack_advance_ms=params.max_attack_advance_ms,
                    merge_gap_ms=params.merge_gap_ms,
                    minimum_silence_ms=params.minimum_silence_ms,
                    tail_rms_ratio=params.tail_rms_ratio,
                    tail_silence_hold_ms=params.tail_silence_hold_ms,
                    max_tail_extension_ms=params.max_tail_extension_ms,
                    minimum_note_duration_ms=params.minimum_note_duration_ms,
                    monophonic=params.monophonic,
                    allow_pitch_overlap=False,
                ),
                dsp_features_file=dsp_features_path,
            )
        else:
            reason = (
                "bass refinement disabled by option"
                if not params.enable_bass_refinement
                else f"refinement not applied for non-bass layer: {layer}"
            )
            refined_document, refinement_report = _build_passthrough_refinement(
                aligned_document=aligned_document,
                aligned_notes_file=audio_aligned_note_events_path,
                audio_features_file=audio_features_path,
                validation_file=note_validation_path,
                reason=reason,
                params=params,
            )

        refinement_report.output_file = str(refined_note_events_path)
        _write_json(refined_note_events_path, refined_document)
        _write_json(bass_refinement_report_path, refinement_report)
        output_files["refined_note_events"] = str(refined_note_events_path)
        output_files["bass_refinement_report"] = str(bass_refinement_report_path)
        stages.append(
            PipelineStageReport(
                name="bass_refinement",
                status="ok",
                output_files=[str(refined_note_events_path), str(bass_refinement_report_path)],
                warning_count=refinement_report.warning_count,
                warnings=refinement_report.warnings,
            )
        )
        warnings.extend([f"bass_refinement: {item}" for item in refinement_report.warnings])

        # Precompute cleanup plan so activity repair can use KEEP/REVIEW masking.
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

        # Stage 7: Activity repair
        current_stage = "activity_repair"
        repaired_refined_note_events_path = analysis_dir / "repaired_refined_note_events.json"
        activity_repair_plan_path = analysis_dir / "activity_repair_plan.json"
        activity_repair_report_path = analysis_dir / "activity_repair_report.json"

        activity_repair_enabled = params.enable_activity_repair and layer.lower() == "bass"
        if activity_repair_enabled:
            repaired_document, repair_plan, repair_report = repair_activity(
                refined_notes_file=refined_note_events_path,
                audio_features_file=audio_features_path,
                cleanup_plan_file=cleanup_plan_path,
                params=ActivityRepairParameters(
                    audio_active_threshold_ratio=params.audio_active_threshold_ratio,
                    audio_silence_hold_ms=params.audio_silence_hold_ms,
                    missing_gap_min_ms=params.missing_gap_min_ms,
                    overhang_min_ms=params.overhang_min_ms,
                    split_min_note_duration_ms=params.split_min_note_duration_ms,
                    close_gap_ms=params.close_gap_ms,
                    insert_auto_confidence=params.insert_auto_confidence,
                    split_auto_confidence=params.split_auto_confidence,
                ),
                dsp_features_file=dsp_features_path,
            )
            repair_report.output_file = str(repaired_refined_note_events_path)
            repair_report.plan_file = str(activity_repair_plan_path)
            _write_json(repaired_refined_note_events_path, repaired_document)
            _write_json(activity_repair_plan_path, repair_plan)
            _write_json(activity_repair_report_path, repair_report)

            output_files["repaired_refined_note_events"] = str(repaired_refined_note_events_path)
            output_files["activity_repair_plan"] = str(activity_repair_plan_path)
            output_files["activity_repair_report"] = str(activity_repair_report_path)

            activity_repair_summary = {
                "extend_count": repair_report.extend_count,
                "shorten_count": repair_report.shorten_count,
                "insert_missing_count": repair_report.insert_missing_count,
                "split_count": repair_report.split_count,
                "close_gap_count": repair_report.close_gap_count,
                "review_manual_count": repair_report.review_manual_count,
            }

            stages.append(
                PipelineStageReport(
                    name="activity_repair",
                    status="ok",
                    output_files=[
                        str(repaired_refined_note_events_path),
                        str(activity_repair_plan_path),
                        str(activity_repair_report_path),
                    ],
                    warning_count=repair_report.warning_count,
                    warnings=repair_report.warnings,
                )
            )
            warnings.extend([f"activity_repair: {item}" for item in repair_report.warnings])
        else:
            repaired_refined_note_events_path = None
            activity_repair_plan_path = None
            activity_repair_report_path = None
            stages.append(
                PipelineStageReport(
                    name="activity_repair",
                    status="ok",
                    output_files=[],
                    warning_count=0,
                    warnings=[],
                )
            )

        # Stage 8: Cleanup planning
        current_stage = "cleanup_plan"
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

        # Stage 9: Review MIDI export (backward compatible)
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

        # Stage 9: Cleaned MIDI export (backward compatible)
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

        # Stage 10: Working MIDI export
        current_stage = "working_midi_export"
        working_export_report_path = working_midi_dir / "working_export_report.json"
        working_refined_notes_file = (
            repaired_refined_note_events_path
            if repaired_refined_note_events_path is not None
            else refined_note_events_path
        )
        working_export_report = export_working_midi(
            notes_file=note_events_path,
            cleanup_plan_file=cleanup_plan_path,
            output_dir=working_midi_dir,
            params=WorkingMidiExportParameters(
                ticks_per_beat=params.ticks_per_beat,
                track_name_prefix=params.track_name_prefix,
                include_diagnostic=params.include_diagnostic_working_midi,
                write_empty_files=params.write_empty_files,
                refined_notes_file=working_refined_notes_file,
                repair_plan_file=activity_repair_plan_path,
                audio_aligned_notes_file=audio_aligned_note_events_path,
                repair_extend_count=activity_repair_summary["extend_count"],
                repair_shorten_count=activity_repair_summary["shorten_count"],
                repair_insert_missing_count=activity_repair_summary["insert_missing_count"],
                repair_split_count=activity_repair_summary["split_count"],
                repair_close_gap_count=activity_repair_summary["close_gap_count"],
                repair_review_manual_count=activity_repair_summary["review_manual_count"],
            ),
        )
        _write_json(working_export_report_path, working_export_report)
        output_files["working_export_report"] = str(working_export_report_path)
        working_outputs = [item.path for item in working_export_report.exported_files] + [
            str(working_export_report_path)
        ]
        stages.append(
            PipelineStageReport(
                name="working_midi_export",
                status="ok",
                output_files=working_outputs,
                warning_count=working_export_report.warning_count,
                warnings=working_export_report.warnings,
            )
        )
        warnings.extend([f"working_midi_export: {item}" for item in working_export_report.warnings])

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
