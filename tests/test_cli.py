import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from midi_cleaner import __version__
from midi_cleaner.cli import app


runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_doctor_json_is_parseable() -> None:
    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code in {0, 1}
    payload = json.loads(result.stdout)
    assert payload["project_name"] == "midi-cleaner"
    assert payload["python"]["required_major_minor"] == "3.11"
    assert "external_tools" in payload


def test_pipeline_process_stem_help_lists_iterative_options() -> None:
    result = runner.invoke(
        app,
        ["pipeline", "process-stem", "--help"],
        env={"COLUMNS": "240", "LINES": "120"},
    )

    assert result.exit_code == 0
    assert "--enable-iterative-repair" in result.stdout
    assert "--no-enable-iterative-repair" in result.stdout
    assert "--repair-iterations" in result.stdout
    assert "--repair-min-improvement" in result.stdout
    assert "--freeze-stable-notes" in result.stdout
    assert "--no-freeze-stable-notes" in result.stdout
    assert "--conservative-final-pass" in result.stdout
    assert "--no-conservative-final-pass" in result.stdout
    assert "--export-iteration-variants" in result.stdout
    assert "--no-export-iteration-variants" in result.stdout


def test_pipeline_process_stem_passes_iterative_cli_values(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_process_stem_pipeline(*, input_midi, input_wav, source, layer, project_dir, params):
        captured["input_midi"] = input_midi
        captured["input_wav"] = input_wav
        captured["source"] = source
        captured["layer"] = layer
        captured["project_dir"] = project_dir
        captured["params"] = params
        return SimpleNamespace(output_files={})

    monkeypatch.setattr("midi_cleaner.cli.process_stem_pipeline", _fake_process_stem_pipeline)

    result = runner.invoke(
        app,
        [
            "pipeline",
            "process-stem",
            "--midi",
            "candidate.mid",
            "--wav",
            "stem.wav",
            "--source",
            "ripx",
            "--layer",
            "bass",
            "--project-dir",
            "projects/demo",
            "--no-enable-iterative-repair",
            "--repair-iterations",
            "2",
            "--repair-min-improvement",
            "0.01",
            "--no-freeze-stable-notes",
            "--no-conservative-final-pass",
            "--no-export-iteration-variants",
        ],
    )

    assert result.exit_code == 0
    params = captured["params"]
    assert params.enable_iterative_repair is False
    assert params.repair_iterations == 2
    assert params.repair_min_improvement == 0.01
    assert params.freeze_stable_notes is False
    assert params.conservative_final_pass is False
    assert params.export_iteration_variants is False
