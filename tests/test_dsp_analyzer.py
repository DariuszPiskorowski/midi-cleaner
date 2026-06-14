from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from midi_cleaner.cli import app
from midi_cleaner.dsp.analyzer import DspAnalysisError, analyze_dsp_stem
from midi_cleaner.dsp.backends import resolve_backend


runner = CliRunner()


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate)


def test_analyze_dsp_basic_backend_outputs_feature_document(tmp_path: Path) -> None:
    sr = 44100
    t = np.linspace(0.0, 0.5, int(sr * 0.5), endpoint=False)
    wave = (0.25 * np.sin(2.0 * np.pi * 110.0 * t)).astype(np.float32)

    wav_path = tmp_path / "bass.wav"
    debug_csv = tmp_path / "analysis" / "audio_features_dsp_debug.csv"
    _write_wav(wav_path, wave, sr)

    document, report = analyze_dsp_stem(
        wav_file=wav_path,
        layer="bass",
        backend="basic",
        allow_backend_fallback=True,
        debug_csv_path=debug_csv,
    )

    assert report.status == "ok"
    assert report.backend_name == "basic"
    assert document.backend_name == "basic"
    assert report.frame_count == len(document.frames)
    assert report.frame_count > 0
    assert debug_csv.exists()


def test_analyze_dsp_cli_writes_output_and_report(tmp_path: Path) -> None:
    sr = 32000
    t = np.linspace(0.0, 0.25, int(sr * 0.25), endpoint=False)
    wave = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    wav_path = tmp_path / "cli_stem.wav"
    output_path = tmp_path / "out" / "audio_features_dsp.json"
    report_path = tmp_path / "out" / "audio_analysis_dsp_report.json"
    debug_csv_path = tmp_path / "out" / "audio_features_dsp_debug.csv"
    _write_wav(wav_path, wave, sr)

    result = runner.invoke(
        app,
        [
            "audio",
            "analyze-dsp",
            "--wav",
            str(wav_path),
            "--layer",
            "bass",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--backend",
            "basic",
            "--debug-csv",
            str(debug_csv_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert report_path.exists()
    assert debug_csv_path.exists()

    feature_doc = json.loads(output_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert feature_doc["layer"] == "bass"
    assert feature_doc["backend_name"] == "basic"
    assert report["status"] == "ok"
    assert report["backend_name"] == "basic"


def test_analyze_dsp_missing_wav_raises_error(tmp_path: Path) -> None:
    with np.testing.assert_raises(DspAnalysisError):
        analyze_dsp_stem(
            wav_file=tmp_path / "missing.wav",
            layer="bass",
            backend="basic",
        )


def test_resolve_backend_fallback_prefers_scipy_over_basic(monkeypatch) -> None:
    def _mock_available(name: str) -> bool:
        return name == "scipy" or name == "basic"

    monkeypatch.setattr("midi_cleaner.dsp.backends.backend_is_available", _mock_available)
    resolved = resolve_backend("librosa", allow_fallback=True)

    assert resolved.backend_available is True
    assert resolved.backend_name == "scipy"
    assert len(resolved.warnings) == 1
