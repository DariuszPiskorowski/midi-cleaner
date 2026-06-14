from __future__ import annotations

import json
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from midi_cleaner.cli import app
from midi_cleaner.pipeline.process_stem import (
    DspAnalysisError,
    PipelineProcessError,
    PipelineProcessParameters,
    process_stem_pipeline,
)


runner = CliRunner()


def _write_candidate_midi(path: Path, with_unmatched_note_off: bool = False) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    if with_unmatched_note_off:
        track.append(mido.Message("note_off", note=60, velocity=0, time=0, channel=0))
    track.append(mido.Message("note_on", note=45, velocity=100, time=0, channel=0))
    track.append(mido.Message("note_off", note=45, velocity=0, time=480, channel=0))
    midi.save(path)


def _write_stem_wav(path: Path) -> None:
    sr = 44100
    t = np.linspace(0.0, 0.5, int(sr * 0.5), endpoint=False)
    wave = 0.2 * np.sin(2.0 * np.pi * 110.0 * t)
    sf.write(str(path), wave.astype(np.float32), sr)


def _expected_paths(project_dir: Path) -> list[Path]:
    return [
        project_dir / "input" / "source_paths.json",
        project_dir / "analysis" / "note_events.json",
        project_dir / "analysis" / "midi_import_report.json",
        project_dir / "analysis" / "audio_features.json",
        project_dir / "analysis" / "audio_analysis_report.json",
        project_dir / "analysis" / "audio_features_dsp.json",
        project_dir / "analysis" / "audio_analysis_dsp_report.json",
        project_dir / "analysis" / "audio_features_dsp_debug.csv",
        project_dir / "analysis" / "audio_aligned_note_events.json",
        project_dir / "analysis" / "audio_alignment_report.json",
        project_dir / "analysis" / "note_validation.json",
        project_dir / "analysis" / "midi_audio_validation_report.json",
        project_dir / "analysis" / "refined_note_events.json",
        project_dir / "analysis" / "bass_refinement_report.json",
        project_dir / "cleanup" / "cleanup_plan.json",
        project_dir / "cleanup" / "cleanup_plan_report.json",
        project_dir / "midi" / "review" / "keep.mid",
        project_dir / "midi" / "review" / "review.mid",
        project_dir / "midi" / "review" / "muted.mid",
        project_dir / "midi" / "review" / "export_report.json",
        project_dir / "midi" / "cleaned" / "cleaned.mid",
        project_dir / "midi" / "cleaned" / "review.mid",
        project_dir / "midi" / "cleaned" / "rejected.mid",
        project_dir / "midi" / "cleaned" / "cleaned_export_report.json",
        project_dir / "midi" / "working" / "working.mid",
        project_dir / "midi" / "working" / "rejected.mid",
        project_dir / "midi" / "working" / "working_export_report.json",
        project_dir / "reports" / "pipeline_report.json",
    ]


def test_process_stem_creates_expected_structure_and_reports(tmp_path: Path) -> None:
    midi_path = tmp_path / "candidate.mid"
    wav_path = tmp_path / "stem.wav"
    project_dir = tmp_path / "pipeline_project"

    _write_candidate_midi(midi_path)
    _write_stem_wav(wav_path)

    report = process_stem_pipeline(
        input_midi=midi_path,
        input_wav=wav_path,
        source="ripx",
        layer="bass",
        project_dir=project_dir,
        params=PipelineProcessParameters(),
    )

    assert report.status == "ok"
    for path in _expected_paths(project_dir):
        assert path.exists()

    pipeline_report = json.loads((project_dir / "reports" / "pipeline_report.json").read_text(encoding="utf-8"))
    assert pipeline_report["status"] == "ok"

    validation_report = json.loads(
        (project_dir / "analysis" / "midi_audio_validation_report.json").read_text(encoding="utf-8")
    )
    assert validation_report["timing_source"] == "audio_aligned_seconds"
    assert validation_report["audio_aligned_notes_file"] == str(
        project_dir / "analysis" / "audio_aligned_note_events.json"
    )


def test_pipeline_report_aggregates_warnings(tmp_path: Path) -> None:
    midi_path = tmp_path / "candidate_warn.mid"
    wav_path = tmp_path / "stem.wav"
    project_dir = tmp_path / "pipeline_project_warn"

    _write_candidate_midi(midi_path, with_unmatched_note_off=True)
    _write_stem_wav(wav_path)

    report = process_stem_pipeline(
        input_midi=midi_path,
        input_wav=wav_path,
        source="ripx",
        layer="bass",
        project_dir=project_dir,
        params=PipelineProcessParameters(),
    )

    assert report.warning_count >= 1
    assert any("midi_import:" in warning for warning in report.warnings)


def test_cli_process_stem_end_to_end(tmp_path: Path) -> None:
    midi_path = tmp_path / "candidate_cli.mid"
    wav_path = tmp_path / "stem_cli.wav"
    project_dir = tmp_path / "pipeline_project_cli"

    _write_candidate_midi(midi_path)
    _write_stem_wav(wav_path)

    result = runner.invoke(
        app,
        [
            "pipeline",
            "process-stem",
            "--midi",
            str(midi_path),
            "--wav",
            str(wav_path),
            "--source",
            "ripx",
            "--layer",
            "bass",
            "--project-dir",
            str(project_dir),
        ],
    )

    assert result.exit_code == 0
    assert (project_dir / "reports" / "pipeline_report.json").exists()
    assert (project_dir / "analysis" / "audio_aligned_note_events.json").exists()
    assert (project_dir / "analysis" / "audio_features_dsp.json").exists()
    assert (project_dir / "analysis" / "refined_note_events.json").exists()
    assert (project_dir / "midi" / "review" / "export_report.json").exists()
    assert (project_dir / "midi" / "cleaned" / "cleaned_export_report.json").exists()
    assert (project_dir / "midi" / "working" / "working_export_report.json").exists()


def test_pipeline_allows_non_required_dsp_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    midi_path = tmp_path / "candidate_non_required.mid"
    wav_path = tmp_path / "stem_non_required.wav"
    project_dir = tmp_path / "pipeline_non_required"

    _write_candidate_midi(midi_path)
    _write_stem_wav(wav_path)

    def _raise_dsp(*args, **kwargs):
        raise DspAnalysisError("simulated dsp failure")

    monkeypatch.setattr("midi_cleaner.pipeline.process_stem.analyze_dsp_stem", _raise_dsp)

    report = process_stem_pipeline(
        input_midi=midi_path,
        input_wav=wav_path,
        source="ripx",
        layer="bass",
        project_dir=project_dir,
        params=PipelineProcessParameters(require_dsp_analysis=False),
    )

    assert report.status == "ok"
    assert any(stage.name == "dsp_analysis" for stage in report.stages)
    assert any("dsp_analysis:" in warning for warning in report.warnings)
    assert not (project_dir / "analysis" / "audio_features_dsp.json").exists()


def test_pipeline_requires_dsp_when_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    midi_path = tmp_path / "candidate_required.mid"
    wav_path = tmp_path / "stem_required.wav"
    project_dir = tmp_path / "pipeline_required"

    _write_candidate_midi(midi_path)
    _write_stem_wav(wav_path)

    def _raise_dsp(*args, **kwargs):
        raise DspAnalysisError("simulated required dsp failure")

    monkeypatch.setattr("midi_cleaner.pipeline.process_stem.analyze_dsp_stem", _raise_dsp)

    with pytest.raises(PipelineProcessError):
        process_stem_pipeline(
            input_midi=midi_path,
            input_wav=wav_path,
            source="ripx",
            layer="bass",
            project_dir=project_dir,
            params=PipelineProcessParameters(require_dsp_analysis=True),
        )
