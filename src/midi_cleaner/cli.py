from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from midi_cleaner import __version__
from midi_cleaner.alignment.audio_time import (
    AudioTimeAlignmentError,
    AudioTimeAlignmentParameters,
    align_notes_to_audio_time,
)
from midi_cleaner.alignment.models import AudioAlignmentReport
from midi_cleaner.audio.analyzer import AudioAnalysisError, analyze_stem
from midi_cleaner.dsp.analyzer import DspAnalysisError, analyze_dsp_stem
from midi_cleaner.cleanup.cleaned_exporter import (
    CleanedMidiExportError,
    CleanedMidiExportParameters,
    export_cleaned_midi,
)
from midi_cleaner.cleanup.models import CleanedMidiExportReport, WorkingMidiExportReport
from midi_cleaner.cleanup.midi_exporter import (
    ReviewMidiExportError,
    ReviewMidiExportParameters,
    export_review_midi,
)
from midi_cleaner.cleanup.working_exporter import (
    WorkingMidiExportError,
    WorkingMidiExportParameters,
    export_working_midi,
)
from midi_cleaner.cleanup.planner import (
    CleanupPlanError,
    CleanupPlannerParameters,
    build_cleanup_plan,
)
from midi_cleaner.midi.importer import MidiImportError, import_midi_candidate
from midi_cleaner.pipeline.process_stem import (
    PipelineProcessError,
    PipelineProcessParameters,
    process_stem_pipeline,
)
from midi_cleaner.pipeline.qa_report import (
    QAReportError,
    QAReportParameters,
    generate_qa_report,
)
from midi_cleaner.pitch.bass_contour import (
    PitchContourError,
    PitchContourParameters,
    analyze_bass_pitch_contour,
)
from midi_cleaner.repair.activity import (
    ActivityRepairError,
    ActivityRepairParameters,
    repair_activity,
)
from midi_cleaner.repair.iterative import (
    IterativeRepairError,
    IterativeRepairParameters,
    run_iterative_activity_repair,
)
from midi_cleaner.refinement.bass import (
    BassRefinementError,
    BassRefinementParameters,
    refine_bass_notes,
)
from midi_cleaner.runtime.report import RuntimeReport, build_runtime_report
from midi_cleaner.validation.models import MidiAudioValidationReport
from midi_cleaner.validation.midi_audio import (
    MidiAudioValidationError,
    ValidationParameters,
    validate_midi_vs_audio,
)

app = typer.Typer(add_completion=False, help="Hermes MIDI Fidelity Engine CLI")
midi_app = typer.Typer(help="MIDI candidate import tools.")
audio_app = typer.Typer(help="Audio stem analysis tools.")
validate_app = typer.Typer(help="MIDI-vs-audio validation tools.")
cleanup_app = typer.Typer(help="Non-destructive MIDI cleanup planning tools.")
refine_app = typer.Typer(help="MIDI timing quality refinement tools.")
repair_app = typer.Typer(help="Audio/MIDI activity repair tools.")
pitch_app = typer.Typer(help="Bass pitch contour analysis tools.")
pipeline_app = typer.Typer(help="End-to-end pipeline tools.")
console = Console()

app.add_typer(midi_app, name="midi")
app.add_typer(audio_app, name="audio")
app.add_typer(validate_app, name="validate")
app.add_typer(cleanup_app, name="cleanup")
app.add_typer(refine_app, name="refine")
app.add_typer(repair_app, name="repair")
app.add_typer(pitch_app, name="pitch")
app.add_typer(pipeline_app, name="pipeline")


def version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit(code=0)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show package version and exit.",
    ),
) -> None:
    _ = version


def _render_human_report(report: RuntimeReport) -> None:
    console.print(f"[bold]Project:[/bold] {report.project_name}")
    console.print(f"[bold]Version:[/bold] {report.package_version}")
    console.print(f"[bold]Timestamp (UTC):[/bold] {report.timestamp_utc}")
    console.print(f"[bold]Status:[/bold] {report.status}")
    console.print(
        "[bold]Python:[/bold] "
        f"{report.python.version} (required {report.python.required_major_minor}.x)"
    )
    console.print(f"[bold]Executable:[/bold] {report.python.executable}")
    console.print(f"[bold]CWD:[/bold] {report.cwd}")

    env_table = Table(title="Python Environment")
    env_table.add_column("Variable")
    env_table.add_column("Value")
    env_table.add_row("VIRTUAL_ENV", str(report.python_environment.VIRTUAL_ENV))
    env_table.add_row(
        "UV_PROJECT_ENVIRONMENT", str(report.python_environment.UV_PROJECT_ENVIRONMENT)
    )
    env_table.add_row("PYTHONPATH", str(report.python_environment.PYTHONPATH))
    console.print(env_table)

    tools_table = Table(title="External Tools")
    tools_table.add_column("Tool")
    tools_table.add_column("Available")
    tools_table.add_column("Required")
    tools_table.add_column("Path")
    tools_table.add_column("Version")
    for tool in report.external_tools:
        tools_table.add_row(
            tool.name,
            "yes" if tool.available else "no",
            "yes" if tool.required else "no",
            str(tool.path),
            str(tool.version),
        )
    console.print(tools_table)

    if report.problems:
        console.print("[bold]Notes:[/bold]")
        for problem in report.problems:
            console.print(f"- {problem}")


@app.command()
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON only."),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write report JSON to a file.",
    ),
) -> None:
    report = build_runtime_report(__version__)
    report_json = report.model_dump_json(indent=2)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report_json + "\n", encoding="utf-8")

    if json_output:
        typer.echo(report_json)
    else:
        _render_human_report(report)
        if output is not None:
            console.print(f"[bold]Saved JSON report:[/bold] {output}")

    if report.status == "error":
        raise typer.Exit(code=1)


@midi_app.command("import-candidate")
def import_candidate(
    input_midi: Path = typer.Argument(..., help="Path to input candidate .mid file."),
    source: str = typer.Option(..., "--source", help="Candidate source label, e.g. ripx."),
    layer: str = typer.Option(..., "--layer", help="Logical instrument layer, e.g. bass."),
    output: Path = typer.Option(..., "--output", help="Output path for note-event JSON."),
    report: Path = typer.Option(..., "--report", help="Output path for import report JSON."),
) -> None:
    try:
        document, import_report = import_midi_candidate(input_midi, source=source, layer=layer)
    except MidiImportError as exc:
        typer.echo(f"Import failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")

    import_report.output_file = str(output)
    report.write_text(import_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "Imported MIDI candidate: "
        f"notes={import_report.note_count}, "
        f"tracks={import_report.track_count}, "
        f"warnings={import_report.warning_count}"
    )


@audio_app.command("analyze-stem")
def analyze_stem_command(
    input_wav: Path = typer.Argument(..., help="Path to input WAV stem file."),
    layer: str = typer.Option(..., "--layer", help="Logical instrument layer, e.g. bass."),
    output: Path = typer.Option(..., "--output", help="Output path for audio feature JSON."),
    report: Path = typer.Option(..., "--report", help="Output path for analysis report JSON."),
) -> None:
    try:
        document, analysis_report = analyze_stem(input_wav=input_wav, layer=layer)
    except AudioAnalysisError as exc:
        typer.echo(f"Audio analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")

    analysis_report.output_file = str(output)
    report.write_text(analysis_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "Analyzed WAV stem: "
        f"frames={analysis_report.frame_count}, "
        f"onsets={analysis_report.onset_count}, "
        f"warnings={analysis_report.warning_count}"
    )


@audio_app.command("analyze-dsp")
def analyze_dsp_command(
    wav: Path = typer.Option(..., "--wav", help="Path to input WAV stem file."),
    layer: str = typer.Option(..., "--layer", help="Logical instrument layer, e.g. bass."),
    output: Path = typer.Option(..., "--output", help="Output path for DSP feature JSON."),
    report: Path = typer.Option(..., "--report", help="Output path for DSP analysis report JSON."),
    debug_csv: Path | None = typer.Option(
        None,
        "--debug-csv",
        help="Optional output path for DSP debug frame CSV.",
    ),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="DSP backend: auto|librosa|scipy|basic.",
    ),
) -> None:
    try:
        document, analysis_report = analyze_dsp_stem(
            wav_file=wav,
            layer=layer,
            backend=backend,
            allow_backend_fallback=True,
            debug_csv_path=debug_csv,
        )
    except DspAnalysisError as exc:
        typer.echo(f"DSP analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    if debug_csv is not None:
        debug_csv.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")

    analysis_report.output_file = str(output)
    if debug_csv is not None:
        analysis_report.debug_csv_file = str(debug_csv)
    report.write_text(analysis_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "Analyzed DSP stem: "
        f"backend={analysis_report.backend_name}, "
        f"frames={analysis_report.frame_count}, "
        f"attacks={analysis_report.attack_rise_count}, "
        f"sustain={analysis_report.sustain_count}, "
        f"tail={analysis_report.tail_count}, "
        f"warnings={analysis_report.warning_count}"
    )


@validate_app.command("midi-vs-audio")
def validate_midi_vs_audio_command(
    notes: Path = typer.Option(..., "--notes", help="Path to note_events.json."),
    audio_features: Path = typer.Option(
        ..., "--audio-features", help="Path to audio_features.json."
    ),
    audio_aligned_notes: Path | None = typer.Option(
        None,
        "--audio-aligned-notes",
        help="Optional path to audio_aligned_note_events.json.",
    ),
    output: Path = typer.Option(..., "--output", help="Output path for note validation JSON."),
    report: Path = typer.Option(..., "--report", help="Output path for validation report JSON."),
    onset_window_ms: float = typer.Option(50.0, "--onset-window-ms"),
    minimum_rms: float = typer.Option(0.001, "--minimum-rms"),
    minimum_onset_score: float = typer.Option(0.01, "--minimum-onset-score"),
    review_threshold: float = typer.Option(0.45, "--review-threshold"),
    keep_threshold: float = typer.Option(0.70, "--keep-threshold"),
) -> None:
    params = ValidationParameters(
        onset_window_ms=onset_window_ms,
        minimum_rms=minimum_rms,
        minimum_onset_score=minimum_onset_score,
        review_threshold=review_threshold,
        keep_threshold=keep_threshold,
    )

    try:
        document, validation_report = validate_midi_vs_audio(
            notes_file=notes,
            audio_features_file=audio_features,
            params=params,
            audio_aligned_notes_file=audio_aligned_notes,
        )
    except MidiAudioValidationError as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")

    validation_report.output_file = str(output)
    report.write_text(validation_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "Validated MIDI vs audio: "
        f"notes={validation_report.note_count}, "
        f"keep={validation_report.keep_count}, "
        f"review={validation_report.review_count}, "
        f"mute_candidates={validation_report.mute_candidate_count}, "
        f"warnings={validation_report.warning_count}"
    )


@validate_app.command("align-audio-time")
def align_audio_time_command(
    notes: Path = typer.Option(..., "--notes", help="Path to note_events.json."),
    audio_features: Path = typer.Option(
        ..., "--audio-features", help="Path to audio_features.json."
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output path for audio aligned note events JSON."
    ),
    report: Path = typer.Option(
        ..., "--report", help="Output path for audio alignment report JSON."
    ),
    onset_search_window_ms: float = typer.Option(250.0, "--onset-search-window-ms"),
    offset_search_window_ms: float = typer.Option(350.0, "--offset-search-window-ms"),
    min_onset_score: float = typer.Option(0.005, "--min-onset-score"),
    min_rms: float = typer.Option(0.001, "--min-rms"),
    snap_start_to_audio_onset: bool = typer.Option(
        True,
        "--snap-start-to-audio-onset/--no-snap-start-to-audio-onset",
    ),
    snap_end_to_energy_offset: bool = typer.Option(
        True,
        "--snap-end-to-energy-offset/--no-snap-end-to-energy-offset",
    ),
    max_start_correction_ms: float = typer.Option(500.0, "--max-start-correction-ms"),
    max_end_correction_ms: float = typer.Option(800.0, "--max-end-correction-ms"),
    low_confidence_action: str = typer.Option(
        "KEEP_ORIGINAL_LOW_CONFIDENCE", "--low-confidence-action"
    ),
) -> None:
    params = AudioTimeAlignmentParameters(
        onset_search_window_ms=onset_search_window_ms,
        offset_search_window_ms=offset_search_window_ms,
        min_onset_score=min_onset_score,
        min_rms=min_rms,
        snap_start_to_audio_onset=snap_start_to_audio_onset,
        snap_end_to_energy_offset=snap_end_to_energy_offset,
        max_start_correction_ms=max_start_correction_ms,
        max_end_correction_ms=max_end_correction_ms,
        low_confidence_action=low_confidence_action,
    )

    try:
        document, alignment_report = align_notes_to_audio_time(
            notes_file=notes,
            audio_features_file=audio_features,
            params=params,
        )
    except AudioTimeAlignmentError as exc:
        typer.echo(f"Audio-time alignment failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")

    alignment_report.output_file = str(output)
    report.write_text(alignment_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "Audio-time note alignment complete: "
        f"notes={alignment_report.note_count}, "
        f"aligned={alignment_report.aligned_count}, "
        f"keep_original={alignment_report.keep_original_count}, "
        f"review={alignment_report.review_timing_count}, "
        f"no_audio_evidence={alignment_report.no_audio_evidence_count}, "
        f"warnings={alignment_report.warning_count}"
    )


@refine_app.command("bass")
def refine_bass_command(
    aligned_notes: Path = typer.Option(
        ..., "--aligned-notes", help="Path to audio_aligned_note_events.json."
    ),
    audio_features: Path = typer.Option(
        ..., "--audio-features", help="Path to audio_features.json."
    ),
    dsp_features: Path | None = typer.Option(
        None,
        "--dsp-features",
        help="Optional path to audio_features_dsp.json.",
    ),
    validation: Path = typer.Option(..., "--validation", help="Path to note_validation.json."),
    output: Path = typer.Option(..., "--output", help="Output path for refined notes JSON."),
    report: Path = typer.Option(..., "--report", help="Output path for refinement report JSON."),
    attack_lookback_ms: float = typer.Option(80.0, "--attack-lookback-ms"),
    max_attack_advance_ms: float = typer.Option(80.0, "--max-attack-advance-ms"),
    attack_rms_ratio: float = typer.Option(0.25, "--attack-rms-ratio"),
    min_attack_rise: float = typer.Option(0.0005, "--min-attack-rise"),
    merge_gap_ms: float = typer.Option(160.0, "--merge-gap-ms"),
    minimum_silence_ms: float = typer.Option(80.0, "--minimum-silence-ms"),
    silence_rms_ratio: float = typer.Option(0.18, "--silence-rms-ratio"),
    same_pitch_tolerance_semitones: int = typer.Option(1, "--same-pitch-tolerance-semitones"),
    max_merge_window_ms: float = typer.Option(600.0, "--max-merge-window-ms"),
    tail_rms_ratio: float = typer.Option(0.20, "--tail-rms-ratio"),
    tail_silence_hold_ms: float = typer.Option(120.0, "--tail-silence-hold-ms"),
    max_tail_extension_ms: float = typer.Option(900.0, "--max-tail-extension-ms"),
    protect_next_onset_ms: float = typer.Option(80.0, "--protect-next-onset-ms"),
    minimum_note_duration_ms: float = typer.Option(80.0, "--minimum-note-duration-ms"),
    monophonic: bool = typer.Option(True, "--monophonic/--polyphonic"),
    allow_pitch_overlap: bool = typer.Option(False, "--allow-pitch-overlap/--no-allow-pitch-overlap"),
) -> None:
    params = BassRefinementParameters(
        attack_lookback_ms=attack_lookback_ms,
        max_attack_advance_ms=max_attack_advance_ms,
        attack_rms_ratio=attack_rms_ratio,
        min_attack_rise=min_attack_rise,
        merge_gap_ms=merge_gap_ms,
        minimum_silence_ms=minimum_silence_ms,
        silence_rms_ratio=silence_rms_ratio,
        same_pitch_tolerance_semitones=same_pitch_tolerance_semitones,
        max_merge_window_ms=max_merge_window_ms,
        tail_rms_ratio=tail_rms_ratio,
        tail_silence_hold_ms=tail_silence_hold_ms,
        max_tail_extension_ms=max_tail_extension_ms,
        protect_next_onset_ms=protect_next_onset_ms,
        minimum_note_duration_ms=minimum_note_duration_ms,
        monophonic=monophonic,
        allow_pitch_overlap=allow_pitch_overlap,
    )

    try:
        refined_document, refinement_report = refine_bass_notes(
            aligned_notes_file=aligned_notes,
            audio_features_file=audio_features,
            validation_file=validation,
            params=params,
            dsp_features_file=dsp_features,
        )
    except BassRefinementError as exc:
        typer.echo(f"Bass refinement failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(refined_document.model_dump_json(indent=2) + "\n", encoding="utf-8")
    refinement_report.output_file = str(output)
    report.write_text(refinement_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "Refined bass notes: "
        f"input={refinement_report.input_note_count}, "
        f"output={refinement_report.output_note_count}, "
        f"merged={refinement_report.false_retrigger_merge_count}, "
        f"tail_extended={refinement_report.tail_extended_count}, "
        f"short_extended={refinement_report.short_note_extended_count}, "
        f"warnings={refinement_report.warning_count}"
    )


@cleanup_app.command("plan")
def cleanup_plan_command(
    validation: Path = typer.Option(..., "--validation", help="Path to note_validation.json."),
    output: Path = typer.Option(..., "--output", help="Output path for cleanup plan JSON."),
    report: Path = typer.Option(..., "--report", help="Output path for cleanup plan report JSON."),
    mute_threshold: float = typer.Option(0.45, "--mute-threshold"),
    review_threshold: float = typer.Option(0.70, "--review-threshold"),
    delete_threshold: float = typer.Option(0.20, "--delete-threshold"),
    allow_delete_candidates: bool = typer.Option(
        False,
        "--allow-delete-candidates/--no-allow-delete-candidates",
    ),
) -> None:
    params = CleanupPlannerParameters(
        mute_threshold=mute_threshold,
        review_threshold=review_threshold,
        delete_threshold=delete_threshold,
        allow_delete_candidates=allow_delete_candidates,
    )

    try:
        plan_document, plan_report = build_cleanup_plan(validation_file=validation, params=params)
    except CleanupPlanError as exc:
        typer.echo(f"Cleanup planning failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(plan_document.model_dump_json(indent=2) + "\n", encoding="utf-8")

    plan_report.output_file = str(output)
    report.write_text(plan_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "planned actions: "
        f"keep={plan_report.keep_count}, "
        f"review={plan_report.review_count}, "
        f"mute={plan_report.mute_count}, "
        f"delete_candidates={plan_report.delete_candidate_count}"
    )


@cleanup_app.command("export-review-midi")
def cleanup_export_review_midi_command(
    notes: Path = typer.Option(..., "--notes", help="Path to note_events.json."),
    plan: Path = typer.Option(..., "--plan", help="Path to cleanup_plan.json."),
    audio_aligned_notes: Path | None = typer.Option(
        None,
        "--audio-aligned-notes",
        help="Optional path to audio_aligned_note_events.json.",
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for exported review MIDI files."),
    report: Path = typer.Option(..., "--report", help="Output path for export report JSON."),
    ticks_per_beat: int | None = typer.Option(
        None,
        "--ticks-per-beat",
        help="Override output ticks-per-beat (default: note_events ticks_per_beat).",
    ),
    track_name_prefix: str = typer.Option("Hermes", "--track-name-prefix"),
    include_delete_candidates: bool = typer.Option(
        True,
        "--include-delete-candidates/--no-include-delete-candidates",
    ),
) -> None:
    params = ReviewMidiExportParameters(
        ticks_per_beat=ticks_per_beat,
        track_name_prefix=track_name_prefix,
        include_delete_candidates=include_delete_candidates,
        audio_aligned_notes_file=audio_aligned_notes,
    )

    try:
        export_report = export_review_midi(
            notes_file=notes,
            cleanup_plan_file=plan,
            output_dir=output_dir,
            params=params,
        )
    except ReviewMidiExportError as exc:
        typer.echo(f"Review MIDI export failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(export_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    counts = {item.action: item.note_count for item in export_report.exported_files}
    typer.echo(
        "exported review MIDI: "
        f"keep={counts.get('KEEP', 0)}, "
        f"review={counts.get('REVIEW', 0)}, "
        f"muted={counts.get('MUTE', 0)}, "
        f"delete_candidates={counts.get('DELETE_CANDIDATE', 0)}"
    )


@cleanup_app.command("export-cleaned-midi")
def cleanup_export_cleaned_midi_command(
    notes: Path = typer.Option(..., "--notes", help="Path to note_events.json."),
    plan: Path = typer.Option(..., "--plan", help="Path to cleanup_plan.json."),
    audio_aligned_notes: Path | None = typer.Option(
        None,
        "--audio-aligned-notes",
        help="Optional path to audio_aligned_note_events.json.",
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for exported cleaned MIDI files."),
    report: Path = typer.Option(..., "--report", help="Output path for cleaned export report JSON."),
    ticks_per_beat: int | None = typer.Option(
        None,
        "--ticks-per-beat",
        help="Override output ticks-per-beat (default: note_events ticks_per_beat).",
    ),
    track_name_prefix: str = typer.Option("Hermes", "--track-name-prefix"),
    include_review_in_cleaned: bool = typer.Option(
        False,
        "--include-review-in-cleaned/--no-include-review-in-cleaned",
    ),
    write_empty_files: bool = typer.Option(
        True,
        "--write-empty-files/--no-write-empty-files",
    ),
) -> None:
    params = CleanedMidiExportParameters(
        ticks_per_beat=ticks_per_beat,
        track_name_prefix=track_name_prefix,
        include_review_in_cleaned=include_review_in_cleaned,
        write_empty_files=write_empty_files,
        audio_aligned_notes_file=audio_aligned_notes,
    )

    try:
        export_report = export_cleaned_midi(
            notes_file=notes,
            cleanup_plan_file=plan,
            output_dir=output_dir,
            params=params,
        )
    except CleanedMidiExportError as exc:
        typer.echo(f"Cleaned MIDI export failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(export_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "exported cleaned MIDI: "
        f"cleaned={export_report.cleaned_note_count}, "
        f"review={export_report.review_note_count}, "
        f"rejected={export_report.rejected_note_count}"
    )


@cleanup_app.command("export-working-midi")
def cleanup_export_working_midi_command(
    notes: Path = typer.Option(..., "--notes", help="Path to note_events.json."),
    plan: Path = typer.Option(..., "--plan", help="Path to cleanup_plan.json."),
    refined_notes: Path | None = typer.Option(
        None,
        "--refined-notes",
        help="Optional path to refined_note_events.json.",
    ),
    repair_plan: Path | None = typer.Option(
        None,
        "--repair-plan",
        help="Optional path to activity_repair_plan.json.",
    ),
    audio_aligned_notes: Path | None = typer.Option(
        None,
        "--audio-aligned-notes",
        help="Optional path to audio_aligned_note_events.json when refined notes are unavailable.",
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for working MIDI files."),
    report: Path = typer.Option(..., "--report", help="Output path for working export report JSON."),
    ticks_per_beat: int | None = typer.Option(
        None,
        "--ticks-per-beat",
        help="Override output ticks-per-beat (default: note_events ticks_per_beat).",
    ),
    track_name_prefix: str = typer.Option("Hermes", "--track-name-prefix"),
    include_diagnostic: bool = typer.Option(
        False,
        "--include-diagnostic/--no-include-diagnostic",
    ),
    write_empty_files: bool = typer.Option(
        True,
        "--write-empty-files/--no-write-empty-files",
    ),
) -> None:
    params = WorkingMidiExportParameters(
        ticks_per_beat=ticks_per_beat,
        track_name_prefix=track_name_prefix,
        include_diagnostic=include_diagnostic,
        write_empty_files=write_empty_files,
        refined_notes_file=refined_notes,
        repair_plan_file=repair_plan,
        audio_aligned_notes_file=audio_aligned_notes,
    )

    try:
        export_report = export_working_midi(
            notes_file=notes,
            cleanup_plan_file=plan,
            output_dir=output_dir,
            params=params,
        )
    except WorkingMidiExportError as exc:
        typer.echo(f"Working MIDI export failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(export_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "exported working MIDI: "
        f"working={export_report.working_note_count}, "
        f"rejected={export_report.rejected_note_count}, "
        f"diagnostic={export_report.diagnostic_note_count}, "
        f"timing_source={export_report.timing_source}"
    )


@repair_app.command("activity")
def repair_activity_command(
    refined_notes: Path = typer.Option(
        ..., "--refined-notes", help="Path to refined_note_events.json."
    ),
    audio_features: Path = typer.Option(
        ..., "--audio-features", help="Path to audio_features.json."
    ),
    cleanup_plan: Path = typer.Option(
        ..., "--cleanup-plan", help="Path to cleanup_plan.json."
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output path for repaired refined notes JSON."
    ),
    plan: Path = typer.Option(
        ..., "--plan", help="Output path for activity repair plan JSON."
    ),
    report: Path = typer.Option(
        ..., "--report", help="Output path for activity repair report JSON."
    ),
    dsp_features: Path | None = typer.Option(
        None,
        "--dsp-features",
        help="Optional path to audio_features_dsp.json.",
    ),
    pitch_contour: Path | None = typer.Option(
        None,
        "--pitch-contour",
        help="Optional path to bass_pitch_contour.json.",
    ),
    audio_active_threshold_ratio: float = typer.Option(
        0.18, "--audio-active-threshold-ratio"
    ),
    audio_silence_hold_ms: float = typer.Option(120.0, "--audio-silence-hold-ms"),
    missing_gap_min_ms: float = typer.Option(80.0, "--missing-gap-min-ms"),
    overhang_min_ms: float = typer.Option(220.0, "--overhang-min-ms"),
    split_min_note_duration_ms: float = typer.Option(500.0, "--split-min-note-duration-ms"),
    close_gap_ms: float = typer.Option(50.0, "--close-gap-ms"),
    insert_auto_confidence: float = typer.Option(0.80, "--insert-auto-confidence"),
    split_auto_confidence: float = typer.Option(0.75, "--split-auto-confidence"),
    split_pitch_change_semitones: float = typer.Option(0.75, "--split-pitch-change-semitones"),
    insert_from_pitch_contour_confidence: float = typer.Option(
        0.75,
        "--insert-from-pitch-contour-confidence",
    ),
    enable_iterative_repair: bool = typer.Option(
        True,
        "--enable-iterative-repair/--no-enable-iterative-repair",
    ),
    repair_iterations: int = typer.Option(3, "--repair-iterations"),
    repair_min_improvement: float = typer.Option(0.005, "--repair-min-improvement"),
    freeze_stable_notes: bool = typer.Option(
        True,
        "--freeze-stable-notes/--no-freeze-stable-notes",
    ),
    conservative_final_pass: bool = typer.Option(
        True,
        "--conservative-final-pass/--no-conservative-final-pass",
    ),
    export_iteration_variants: bool = typer.Option(
        True,
        "--export-iteration-variants/--no-export-iteration-variants",
    ),
) -> None:
    params = ActivityRepairParameters(
        audio_active_threshold_ratio=audio_active_threshold_ratio,
        audio_silence_hold_ms=audio_silence_hold_ms,
        missing_gap_min_ms=missing_gap_min_ms,
        overhang_min_ms=overhang_min_ms,
        split_min_note_duration_ms=split_min_note_duration_ms,
        close_gap_ms=close_gap_ms,
        insert_auto_confidence=insert_auto_confidence,
        split_auto_confidence=split_auto_confidence,
        split_pitch_change_semitones=split_pitch_change_semitones,
        insert_from_pitch_contour_confidence=insert_from_pitch_contour_confidence,
        enable_iterative_repair=enable_iterative_repair,
        repair_iterations=repair_iterations,
        repair_min_improvement=repair_min_improvement,
        freeze_stable_notes=freeze_stable_notes,
        conservative_final_pass=conservative_final_pass,
        export_iteration_variants=export_iteration_variants,
    )

    try:
        repaired_document, repair_plan, repair_report = repair_activity(
            refined_notes_file=refined_notes,
            audio_features_file=audio_features,
            cleanup_plan_file=cleanup_plan,
            params=params,
            dsp_features_file=dsp_features,
            pitch_contour_file=pitch_contour,
        )
    except ActivityRepairError as exc:
        typer.echo(f"Activity repair failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    plan.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(repaired_document.model_dump_json(indent=2) + "\n", encoding="utf-8")
    plan.write_text(repair_plan.model_dump_json(indent=2) + "\n", encoding="utf-8")

    repair_report.output_file = str(output)
    repair_report.plan_file = str(plan)
    report.write_text(repair_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "Activity repair complete: "
        f"input={repair_report.input_note_count}, "
        f"output={repair_report.output_note_count}, "
        f"extend={repair_report.extend_count}, "
        f"shorten={repair_report.shorten_count}, "
        f"insert={repair_report.insert_missing_count}, "
        f"split={repair_report.split_count}, "
        f"review={repair_report.review_manual_count}, "
        f"warnings={repair_report.warning_count}"
    )


@repair_app.command("iterative")
def repair_iterative_command(
    refined_notes: Path = typer.Option(
        ..., "--refined-notes", help="Path to refined_note_events.json."
    ),
    audio_features: Path = typer.Option(
        ..., "--audio-features", help="Path to audio_features.json."
    ),
    cleanup_plan: Path = typer.Option(
        ..., "--cleanup-plan", help="Path to cleanup_plan.json."
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output path for final repaired refined notes JSON."
    ),
    report: Path = typer.Option(
        ..., "--report", help="Output path for iterative repair report JSON."
    ),
    dsp_features: Path | None = typer.Option(
        None,
        "--dsp-features",
        help="Optional path to audio_features_dsp.json.",
    ),
    pitch_contour: Path | None = typer.Option(
        None,
        "--pitch-contour",
        help="Optional path to bass_pitch_contour.json.",
    ),
    max_iterations: int = typer.Option(3, "--max-iterations"),
    min_improvement: float = typer.Option(0.005, "--min-improvement"),
    conservative_final_pass: bool = typer.Option(
        True,
        "--conservative-final-pass/--no-conservative-final-pass",
    ),
    freeze_stable_notes: bool = typer.Option(
        True,
        "--freeze-stable-notes/--no-freeze-stable-notes",
    ),
    pass1_profile: str = typer.Option(
        "balanced",
        "--pass1-profile",
        help="Pass-1 profile: balanced|sustain_legato|aggressive.",
    ),
    pass2_profile: str = typer.Option(
        "sustain_legato",
        "--pass2-profile",
        help="Pass-2 profile: balanced|sustain_legato|aggressive.",
    ),
    pass3_profile: str = typer.Option(
        "conservative",
        "--pass3-profile",
        help="Pass-3 profile: conservative|sustain_legato|balanced.",
    ),
) -> None:
    pass12_allowed = {"balanced", "sustain_legato", "aggressive"}
    pass3_allowed = {"conservative", "sustain_legato", "balanced"}
    if pass1_profile not in pass12_allowed:
        typer.echo(f"Invalid --pass1-profile: {pass1_profile}", err=True)
        raise typer.Exit(code=1)
    if pass2_profile not in pass12_allowed:
        typer.echo(f"Invalid --pass2-profile: {pass2_profile}", err=True)
        raise typer.Exit(code=1)
    if pass3_profile not in pass3_allowed:
        typer.echo(f"Invalid --pass3-profile: {pass3_profile}", err=True)
        raise typer.Exit(code=1)

    iterative_params = IterativeRepairParameters(
        max_iterations=max_iterations,
        min_improvement=min_improvement,
        conservative_final_pass=conservative_final_pass,
        freeze_stable_notes=freeze_stable_notes,
        pass1_profile=pass1_profile,
        pass2_profile=pass2_profile,
        pass3_profile=pass3_profile,
    )

    try:
        final_document, iterative_report, _artifacts = run_iterative_activity_repair(
            refined_notes_file=refined_notes,
            audio_features_file=audio_features,
            cleanup_plan_file=cleanup_plan,
            params=iterative_params,
            activity_params=ActivityRepairParameters(),
            dsp_features_file=dsp_features,
            pitch_contour_file=pitch_contour,
        )
    except IterativeRepairError as exc:
        typer.echo(f"Iterative repair failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(final_document.model_dump_json(indent=2) + "\n", encoding="utf-8")
    iterative_report.final_repaired_notes_file = str(output)
    iterative_report.output_file = str(report)
    report.write_text(iterative_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "Iterative repair complete: "
        f"iterations={iterative_report.iterations_completed}/{iterative_report.iterations_requested}, "
        f"best_iteration={iterative_report.best_iteration_index}, "
        f"initial_score={iterative_report.initial_score:.4f}, "
        f"final_score={iterative_report.final_score:.4f}, "
        f"improvement={iterative_report.total_improvement:.4f}, "
        f"warnings={iterative_report.warning_count}"
    )


@pitch_app.command("bass-contour")
def pitch_bass_contour_command(
    wav: Path = typer.Option(..., "--wav", help="Path to input WAV stem file."),
    layer: str = typer.Option(..., "--layer", help="Logical instrument layer, e.g. bass."),
    output: Path = typer.Option(..., "--output", help="Output path for bass contour JSON."),
    report: Path = typer.Option(..., "--report", help="Output path for bass contour report JSON."),
    backend: str = typer.Option("auto", "--pitch-backend", help="Pitch backend: auto|librosa|basic."),
    pitch_min_hz: float = typer.Option(35.0, "--pitch-min-hz"),
    pitch_max_hz: float = typer.Option(400.0, "--pitch-max-hz"),
    pitch_confidence_threshold: float = typer.Option(0.60, "--pitch-confidence-threshold"),
) -> None:
    params = PitchContourParameters(
        backend=backend,
        min_hz=pitch_min_hz,
        max_hz=pitch_max_hz,
        confidence_threshold=pitch_confidence_threshold,
    )

    try:
        document, contour_report = analyze_bass_pitch_contour(
            wav_file=wav,
            layer=layer,
            params=params,
        )
    except PitchContourError as exc:
        typer.echo(f"Bass contour analysis failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")
    contour_report.output_file = str(output)
    report.write_text(contour_report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.echo(
        "Bass pitch contour complete: "
        f"backend={contour_report.backend_name}, "
        f"frames={contour_report.frame_count}, "
        f"voiced={contour_report.voiced_frame_count}, "
        f"warnings={contour_report.warning_count}"
    )


@pipeline_app.command("process-stem")
def process_stem_command(
    midi: Path = typer.Option(..., "--midi", help="Path to candidate MIDI file."),
    wav: Path = typer.Option(..., "--wav", help="Path to stem WAV file."),
    source: str = typer.Option(..., "--source", help="Source label, e.g. ripx."),
    layer: str = typer.Option(..., "--layer", help="Layer label, e.g. bass."),
    project_dir: Path = typer.Option(..., "--project-dir", help="Pipeline project output directory."),
    onset_window_ms: float = typer.Option(50.0, "--onset-window-ms"),
    minimum_rms: float = typer.Option(0.001, "--minimum-rms"),
    minimum_onset_score: float = typer.Option(0.01, "--minimum-onset-score"),
    review_threshold: float = typer.Option(0.45, "--review-threshold"),
    keep_threshold: float = typer.Option(0.70, "--keep-threshold"),
    onset_search_window_ms: float = typer.Option(250.0, "--onset-search-window-ms"),
    offset_search_window_ms: float = typer.Option(350.0, "--offset-search-window-ms"),
    alignment_min_onset_score: float = typer.Option(0.005, "--alignment-min-onset-score"),
    alignment_min_rms: float = typer.Option(0.001, "--alignment-min-rms"),
    snap_start_to_audio_onset: bool = typer.Option(
        True,
        "--snap-start-to-audio-onset/--no-snap-start-to-audio-onset",
    ),
    snap_end_to_energy_offset: bool = typer.Option(
        True,
        "--snap-end-to-energy-offset/--no-snap-end-to-energy-offset",
    ),
    max_start_correction_ms: float = typer.Option(500.0, "--max-start-correction-ms"),
    max_end_correction_ms: float = typer.Option(800.0, "--max-end-correction-ms"),
    low_confidence_action: str = typer.Option(
        "KEEP_ORIGINAL_LOW_CONFIDENCE", "--low-confidence-action"
    ),
    mute_threshold: float = typer.Option(0.45, "--mute-threshold"),
    cleanup_review_threshold: float = typer.Option(0.70, "--cleanup-review-threshold"),
    delete_threshold: float = typer.Option(0.20, "--delete-threshold"),
    allow_delete_candidates: bool = typer.Option(
        False,
        "--allow-delete-candidates/--no-allow-delete-candidates",
    ),
    ticks_per_beat: int | None = typer.Option(
        None,
        "--ticks-per-beat",
        help="Override output ticks-per-beat (default: note_events ticks_per_beat).",
    ),
    track_name_prefix: str = typer.Option("Hermes", "--track-name-prefix"),
    include_review_in_cleaned: bool = typer.Option(
        False,
        "--include-review-in-cleaned/--no-include-review-in-cleaned",
    ),
    write_empty_files: bool = typer.Option(True, "--write-empty-files/--no-write-empty-files"),
    include_delete_candidates: bool = typer.Option(
        True,
        "--include-delete-candidates/--no-include-delete-candidates",
    ),
    enable_bass_refinement: bool = typer.Option(
        True,
        "--enable-bass-refinement/--no-enable-bass-refinement",
    ),
    attack_lookback_ms: float = typer.Option(80.0, "--attack-lookback-ms"),
    max_attack_advance_ms: float = typer.Option(80.0, "--max-attack-advance-ms"),
    merge_gap_ms: float = typer.Option(160.0, "--merge-gap-ms"),
    minimum_silence_ms: float = typer.Option(80.0, "--minimum-silence-ms"),
    tail_rms_ratio: float = typer.Option(0.20, "--tail-rms-ratio"),
    tail_silence_hold_ms: float = typer.Option(120.0, "--tail-silence-hold-ms"),
    max_tail_extension_ms: float = typer.Option(900.0, "--max-tail-extension-ms"),
    minimum_note_duration_ms: float = typer.Option(80.0, "--minimum-note-duration-ms"),
    monophonic: bool = typer.Option(True, "--monophonic/--polyphonic"),
    enable_dsp_analysis: bool = typer.Option(
        True,
        "--enable-dsp-analysis/--no-enable-dsp-analysis",
    ),
    require_dsp_analysis: bool = typer.Option(
        False,
        "--require-dsp-analysis/--no-require-dsp-analysis",
    ),
    dsp_backend: str = typer.Option(
        "auto",
        "--dsp-backend",
        help="DSP backend: auto|librosa|scipy|basic.",
    ),
    dsp_debug_csv: bool = typer.Option(
        True,
        "--dsp-debug-csv/--no-dsp-debug-csv",
    ),
    enable_pitch_contour: bool = typer.Option(
        True,
        "--enable-pitch-contour/--no-enable-pitch-contour",
    ),
    require_pitch_contour: bool = typer.Option(
        False,
        "--require-pitch-contour/--no-require-pitch-contour",
    ),
    pitch_backend: str = typer.Option(
        "auto",
        "--pitch-backend",
        help="Pitch backend: auto|librosa|basic.",
    ),
    pitch_min_hz: float = typer.Option(35.0, "--pitch-min-hz"),
    pitch_max_hz: float = typer.Option(400.0, "--pitch-max-hz"),
    pitch_confidence_threshold: float = typer.Option(0.60, "--pitch-confidence-threshold"),
    enable_activity_repair: bool = typer.Option(
        True,
        "--enable-activity-repair/--no-enable-activity-repair",
    ),
    audio_active_threshold_ratio: float = typer.Option(
        0.18,
        "--audio-active-threshold-ratio",
    ),
    audio_silence_hold_ms: float = typer.Option(120.0, "--audio-silence-hold-ms"),
    missing_gap_min_ms: float = typer.Option(80.0, "--missing-gap-min-ms"),
    overhang_min_ms: float = typer.Option(220.0, "--overhang-min-ms"),
    split_min_note_duration_ms: float = typer.Option(500.0, "--split-min-note-duration-ms"),
    close_gap_ms: float = typer.Option(50.0, "--close-gap-ms"),
    insert_auto_confidence: float = typer.Option(0.80, "--insert-auto-confidence"),
    split_auto_confidence: float = typer.Option(0.75, "--split-auto-confidence"),
    split_pitch_change_semitones: float = typer.Option(0.75, "--split-pitch-change-semitones"),
    insert_from_pitch_contour_confidence: float = typer.Option(
        0.75,
        "--insert-from-pitch-contour-confidence",
    ),
) -> None:
    params = PipelineProcessParameters(
        onset_window_ms=onset_window_ms,
        minimum_rms=minimum_rms,
        minimum_onset_score=minimum_onset_score,
        review_threshold=review_threshold,
        keep_threshold=keep_threshold,
        onset_search_window_ms=onset_search_window_ms,
        offset_search_window_ms=offset_search_window_ms,
        alignment_min_onset_score=alignment_min_onset_score,
        alignment_min_rms=alignment_min_rms,
        snap_start_to_audio_onset=snap_start_to_audio_onset,
        snap_end_to_energy_offset=snap_end_to_energy_offset,
        max_start_correction_ms=max_start_correction_ms,
        max_end_correction_ms=max_end_correction_ms,
        low_confidence_action=low_confidence_action,
        mute_threshold=mute_threshold,
        cleanup_review_threshold=cleanup_review_threshold,
        delete_threshold=delete_threshold,
        allow_delete_candidates=allow_delete_candidates,
        ticks_per_beat=ticks_per_beat,
        track_name_prefix=track_name_prefix,
        include_review_in_cleaned=include_review_in_cleaned,
        write_empty_files=write_empty_files,
        include_delete_candidates=include_delete_candidates,
        enable_bass_refinement=enable_bass_refinement,
        attack_lookback_ms=attack_lookback_ms,
        max_attack_advance_ms=max_attack_advance_ms,
        merge_gap_ms=merge_gap_ms,
        minimum_silence_ms=minimum_silence_ms,
        tail_rms_ratio=tail_rms_ratio,
        tail_silence_hold_ms=tail_silence_hold_ms,
        max_tail_extension_ms=max_tail_extension_ms,
        minimum_note_duration_ms=minimum_note_duration_ms,
        monophonic=monophonic,
        enable_dsp_analysis=enable_dsp_analysis,
        require_dsp_analysis=require_dsp_analysis,
        dsp_backend=dsp_backend,
        dsp_debug_csv=dsp_debug_csv,
        enable_pitch_contour=enable_pitch_contour,
        require_pitch_contour=require_pitch_contour,
        pitch_backend=pitch_backend,
        pitch_min_hz=pitch_min_hz,
        pitch_max_hz=pitch_max_hz,
        pitch_confidence_threshold=pitch_confidence_threshold,
        enable_activity_repair=enable_activity_repair,
        audio_active_threshold_ratio=audio_active_threshold_ratio,
        audio_silence_hold_ms=audio_silence_hold_ms,
        missing_gap_min_ms=missing_gap_min_ms,
        overhang_min_ms=overhang_min_ms,
        split_min_note_duration_ms=split_min_note_duration_ms,
        close_gap_ms=close_gap_ms,
        insert_auto_confidence=insert_auto_confidence,
        split_auto_confidence=split_auto_confidence,
        split_pitch_change_semitones=split_pitch_change_semitones,
        insert_from_pitch_contour_confidence=insert_from_pitch_contour_confidence,
    )

    try:
        report = process_stem_pipeline(
            input_midi=midi,
            input_wav=wav,
            source=source,
            layer=layer,
            project_dir=project_dir,
            params=params,
        )
    except PipelineProcessError as exc:
        typer.echo(f"process-stem failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    note_count = 0
    aligned_count = 0
    keep_original_timing_count = 0
    review_timing_count = 0
    cleaned_count = 0
    review_count = 0
    rejected_count = 0
    working_count = 0
    alignment_report_path = report.output_files.get("audio_alignment_report")
    if alignment_report_path:
        alignment_report = AudioAlignmentReport.model_validate_json(
            Path(alignment_report_path).read_text(encoding="utf-8")
        )
        aligned_count = alignment_report.aligned_count
        keep_original_timing_count = alignment_report.keep_original_count
        review_timing_count = alignment_report.review_timing_count

    validation_report_path = report.output_files.get("midi_audio_validation_report")
    if validation_report_path:
        validation_report = MidiAudioValidationReport.model_validate_json(
            Path(validation_report_path).read_text(encoding="utf-8")
        )
        note_count = validation_report.note_count

    cleaned_report_path = report.output_files.get("cleaned_export_report")
    if cleaned_report_path:
        cleaned_report = CleanedMidiExportReport.model_validate_json(
            Path(cleaned_report_path).read_text(encoding="utf-8")
        )
        cleaned_count = cleaned_report.cleaned_note_count
        review_count = cleaned_report.review_note_count
        rejected_count = cleaned_report.rejected_note_count

    working_report_path = report.output_files.get("working_export_report")
    if working_report_path:
        working_report = WorkingMidiExportReport.model_validate_json(
            Path(working_report_path).read_text(encoding="utf-8")
        )
        working_count = working_report.working_note_count

    typer.echo(
        "process-stem complete: "
        f"notes={note_count}, "
        f"aligned={aligned_count}, "
        f"keep_original_timing={keep_original_timing_count}, "
        f"review_timing={review_timing_count}, "
        f"keep={cleaned_count}, "
        f"review={review_count}, "
        f"rejected={rejected_count}, "
        f"working={working_count}"
    )


@pipeline_app.command("qa-report")
def pipeline_qa_report_command(
    project_dir: Path = typer.Option(..., "--project-dir", help="Pipeline project directory."),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Output directory for QA artifacts (default: PROJECT_DIR/reports).",
    ),
    top_n: int = typer.Option(25, "--top-n"),
    include_csv: bool = typer.Option(True, "--include-csv/--no-include-csv"),
    include_html: bool = typer.Option(True, "--include-html/--no-include-html"),
) -> None:
    params = QAReportParameters(top_n=top_n, include_csv=include_csv, include_html=include_html)

    try:
        summary = generate_qa_report(project_dir=project_dir, output_dir=output_dir, params=params)
    except QAReportError as exc:
        typer.echo(f"qa-report failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        "QA report complete: "
        f"notes={summary.total_notes}, "
        f"keep={summary.keep_count}, "
        f"review={summary.review_count}, "
        f"mute={summary.mute_count}, "
        f"warnings={summary.warning_count}"
    )


if __name__ == "__main__":
    app()
