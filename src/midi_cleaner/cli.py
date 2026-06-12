from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from midi_cleaner import __version__
from midi_cleaner.audio.analyzer import AudioAnalysisError, analyze_stem
from midi_cleaner.cleanup.planner import (
    CleanupPlanError,
    CleanupPlannerParameters,
    build_cleanup_plan,
)
from midi_cleaner.midi.importer import MidiImportError, import_midi_candidate
from midi_cleaner.runtime.report import RuntimeReport, build_runtime_report
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
console = Console()

app.add_typer(midi_app, name="midi")
app.add_typer(audio_app, name="audio")
app.add_typer(validate_app, name="validate")
app.add_typer(cleanup_app, name="cleanup")


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


@validate_app.command("midi-vs-audio")
def validate_midi_vs_audio_command(
    notes: Path = typer.Option(..., "--notes", help="Path to note_events.json."),
    audio_features: Path = typer.Option(
        ..., "--audio-features", help="Path to audio_features.json."
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


if __name__ == "__main__":
    app()
