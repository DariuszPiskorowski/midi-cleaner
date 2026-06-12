from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from midi_cleaner.audio.analyzer import analyze_stem
from midi_cleaner.cli import app


runner = CliRunner()


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate)


def test_analyze_simple_sine_wave(tmp_path: Path) -> None:
    sr = 44100
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    sine = 0.5 * np.sin(2.0 * np.pi * 220.0 * t)

    input_path = tmp_path / "sine.wav"
    _write_wav(input_path, sine, sr)

    document, report = analyze_stem(input_wav=input_path, layer="bass")

    assert report.status == "ok"
    assert document.sample_rate == sr
    assert document.channels == 1
    assert len(document.frames) > 0


def test_analyze_accepts_mono_and_stereo(tmp_path: Path) -> None:
    sr = 22050
    mono = np.zeros(sr, dtype=np.float32)
    stereo = np.stack([mono, mono], axis=1)

    mono_path = tmp_path / "mono.wav"
    stereo_path = tmp_path / "stereo.wav"
    _write_wav(mono_path, mono, sr)
    _write_wav(stereo_path, stereo, sr)

    mono_doc, _ = analyze_stem(input_wav=mono_path, layer="pad")
    stereo_doc, _ = analyze_stem(input_wav=stereo_path, layer="pad")

    assert mono_doc.channels == 1
    assert stereo_doc.channels == 2


def test_frame_features_and_global_metadata(tmp_path: Path) -> None:
    sr = 16000
    t = np.linspace(0.0, 0.5, int(sr * 0.5), endpoint=False)
    wave = 0.3 * np.sin(2.0 * np.pi * 440.0 * t)

    input_path = tmp_path / "meta.wav"
    _write_wav(input_path, wave, sr)

    document, report = analyze_stem(input_wav=input_path, layer="lead")

    assert report.sample_rate == sr
    assert report.channels == 1
    assert 0.49 <= report.duration_sec <= 0.51
    assert document.global_features.frame_count == len(document.frames)
    assert document.frames[0].rms >= 0.0


def test_silence_ratio_high_for_silent_wav(tmp_path: Path) -> None:
    sr = 44100
    silence = np.zeros(sr, dtype=np.float32)

    input_path = tmp_path / "silence.wav"
    _write_wav(input_path, silence, sr)

    document, _report = analyze_stem(input_wav=input_path, layer="pad")

    assert document.global_features.estimated_silence_ratio > 0.95


def test_onset_count_nonzero_for_pulse_wav(tmp_path: Path) -> None:
    sr = 44100
    signal = np.zeros(sr, dtype=np.float32)
    pulse_len = int(0.01 * sr)
    signal[1000 : 1000 + pulse_len] = 0.9
    signal[20000 : 20000 + pulse_len] = 0.9

    input_path = tmp_path / "pulse.wav"
    _write_wav(input_path, signal, sr)

    document, _report = analyze_stem(input_wav=input_path, layer="drums")

    assert document.global_features.onset_count > 0


def test_cli_audio_analyze_writes_output_and_report(tmp_path: Path) -> None:
    sr = 32000
    t = np.linspace(0.0, 0.25, int(sr * 0.25), endpoint=False)
    wave = 0.2 * np.sin(2.0 * np.pi * 330.0 * t)

    input_path = tmp_path / "cli_stem.wav"
    output_path = tmp_path / "out" / "audio_features.json"
    report_path = tmp_path / "out" / "audio_analysis_report.json"
    _write_wav(input_path, wave, sr)

    result = runner.invoke(
        app,
        [
            "audio",
            "analyze-stem",
            str(input_path),
            "--layer",
            "bass",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert report_path.exists()

    feature_doc = json.loads(output_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert feature_doc["layer"] == "bass"
    assert "frames" in feature_doc
    assert report["status"] == "ok"
    assert report["output_file"] == str(output_path)
