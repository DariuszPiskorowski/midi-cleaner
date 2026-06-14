from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from midi_cleaner.cli import app
from midi_cleaner.pitch.bass_contour import (
    PitchContourParameters,
    analyze_bass_pitch_contour,
)


runner = CliRunner()


def _write_bass_wav(path: Path, sample_rate: int = 44100) -> None:
    duration_sec = 1.0
    t = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    wave = (
        0.25 * np.sin(2.0 * np.pi * 55.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * 110.0 * t)
    ).astype(np.float32)
    sf.write(str(path), wave, sample_rate)


def test_analyze_bass_pitch_contour_basic_backend(tmp_path: Path) -> None:
    wav_path = tmp_path / "bass.wav"
    _write_bass_wav(wav_path)

    document, report = analyze_bass_pitch_contour(
        wav_file=wav_path,
        layer="bass",
        params=PitchContourParameters(backend="basic", confidence_threshold=0.2),
    )

    assert report.status == "ok"
    assert document.backend_name == "basic"
    assert report.frame_count == len(document.frames)
    assert report.frame_count > 0
    assert report.voiced_frame_count > 0
    assert report.voiced_ratio > 0.0


def test_cli_pitch_bass_contour_writes_outputs(tmp_path: Path) -> None:
    wav_path = tmp_path / "bass_cli.wav"
    output_path = tmp_path / "analysis" / "bass_pitch_contour.json"
    report_path = tmp_path / "analysis" / "bass_pitch_contour_report.json"
    _write_bass_wav(wav_path)

    result = runner.invoke(
        app,
        [
            "pitch",
            "bass-contour",
            "--wav",
            str(wav_path),
            "--layer",
            "bass",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--pitch-backend",
            "basic",
            "--pitch-min-hz",
            "35",
            "--pitch-max-hz",
            "400",
            "--pitch-confidence-threshold",
            "0.5",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert report_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["layer"] == "bass"
    assert payload["backend_name"] == "basic"
    assert report["status"] == "ok"
    assert report["frame_count"] > 0
