from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from midi_cleaner import __version__
from midi_cleaner.runtime.report import RuntimeReport, build_runtime_report

app = typer.Typer(add_completion=False, help="Hermes MIDI Fidelity Engine CLI")
console = Console()


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


if __name__ == "__main__":
    app()
