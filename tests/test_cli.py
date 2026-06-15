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
    assert "--enable-ai-pattern-completion" in result.stdout
    assert "--no-enable-ai-pattern-completion" in result.stdout


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
            "--enable-ai-pattern-completion",
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
    assert params.enable_ai_pattern_completion is True


def test_ai_complete_pattern_help_lists_required_options() -> None:
    result = runner.invoke(
        app,
        ["ai", "complete-pattern", "--help"],
        env={"COLUMNS": "240", "LINES": "120"},
    )

    assert result.exit_code == 0
    assert "--project-dir" in result.stdout
    assert "--layer" in result.stdout
    assert "--model" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--max-completion-notes" in result.stdout
    assert "--temperature" in result.stdout
    assert "--keep-ai-json" in result.stdout


def test_pattern_complete_blocks_help_lists_required_options() -> None:
    result = runner.invoke(
        app,
        ["pattern", "complete-blocks", "--help"],
        env={"COLUMNS": "240", "LINES": "120"},
    )

    assert result.exit_code == 0
    assert "--project-dir" in result.stdout
    assert "--layer" in result.stdout
    assert "--write-debug-midi" in result.stdout
    assert "--no-write-debug-midi" in result.stdout


def test_pattern_complete_blocks_calls_deterministic_service(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_complete_pattern_blocks(*, project_dir, params):
        captured["project_dir"] = project_dir
        captured["params"] = params
        return SimpleNamespace(
            bar_aligned_block_count=3,
            pattern_block_count=3,
            complete_block_count=2,
            pattern_family_count=1,
            incomplete_existing_block_count=1,
            missing_expected_block_count=0,
            incomplete_block_count=1,
            completed_incomplete_existing_block_count=1,
            completed_missing_expected_block_count=0,
            completed_block_count=1,
            skipped_block_count=0,
            skipped_ambiguous_count=0,
            skipped_no_clear_family_count=0,
            rejected_micro_note_count=0,
            rejected_polyphonic_stack_count=0,
            rejected_low_confidence_count=0,
            rejected_tiny_gap_count=0,
            bar_gap_candidate_count=2,
            inserted_note_count=2,
            output_midi_path="projects/demo/midi/uzupelnienie.mid",
            bar_gap_candidates_file="projects/demo/analysis/pattern_blocks/bar_gap_candidates.json",
            warning_count=0,
        )

    monkeypatch.setattr("midi_cleaner.cli.complete_pattern_blocks", _fake_complete_pattern_blocks)

    result = runner.invoke(
        app,
        [
            "pattern",
            "complete-blocks",
            "--project-dir",
            "projects/demo",
            "--layer",
            "bass",
            "--no-write-debug-midi",
        ],
    )

    assert result.exit_code == 0
    params = captured["params"]
    assert params.layer == "bass"
    assert params.write_debug_midi is False
    assert "inserted_note_count=2" in result.stdout
    assert "missing_expected_block_count=0" in result.stdout
    assert "bar_aligned_block_count=3" in result.stdout
    assert "rejected_low_confidence_count=0" in result.stdout
    assert "bar_gap_candidate_count=2" in result.stdout
