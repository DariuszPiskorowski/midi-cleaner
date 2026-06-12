from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from midi_cleaner.audio.models import AudioFeatureDocument, AudioFrameFeature, AudioGlobalFeatures
from midi_cleaner.cli import app
from midi_cleaner.midi.models import NoteEvent, NoteEventDocument, TempoEvent
from midi_cleaner.validation.midi_audio import ValidationParameters, validate_midi_vs_audio


runner = CliRunner()


def _make_note(
    note_id: str,
    start_sec: float,
    end_sec: float,
    pitch_midi: int = 45,
    layer: str = "bass",
    source: str = "ripx",
) -> NoteEvent:
    return NoteEvent(
        note_id=note_id,
        source=source,
        layer=layer,
        track_index=0,
        track_name="Test",
        channel=0,
        pitch_midi=pitch_midi,
        pitch_name="A2",
        velocity=100,
        start_tick=0,
        end_tick=0,
        duration_ticks=0,
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=end_sec - start_sec,
    )


def _write_documents(
    tmp_path: Path,
    notes: list[NoteEvent],
    frames: list[AudioFrameFeature],
    note_layer: str = "bass",
    audio_layer: str = "bass",
) -> tuple[Path, Path]:
    notes_doc = NoteEventDocument(
        schema_version="0.1.0",
        source_file="candidate.mid",
        source="ripx",
        layer=note_layer,
        ticks_per_beat=480,
        tempo_map=[TempoEvent(tick=0, tempo_us_per_beat=500000, sec=0.0)],
        notes=notes,
    )

    audio_doc = AudioFeatureDocument(
        schema_version="0.1.0",
        source_file="stem.wav",
        layer=audio_layer,
        sample_rate=44100,
        channels=1,
        duration_sec=1.0,
        frame_size=2048,
        hop_size=512,
        frames=frames,
        global_features=AudioGlobalFeatures(
            peak=0.8,
            rms=0.2,
            duration_sec=1.0,
            estimated_silence_ratio=0.2,
            frame_count=len(frames),
            onset_count=sum(1 for frame in frames if frame.onset_score > 0.01),
            mean_spectral_centroid_hz=350.0,
            mean_spectral_rolloff_hz=1200.0,
        ),
    )

    notes_path = tmp_path / "note_events.json"
    audio_path = tmp_path / "audio_features.json"
    notes_path.write_text(notes_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    audio_path.write_text(audio_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")

    return notes_path, audio_path


def test_aligned_note_gets_keep_recommendation(tmp_path: Path) -> None:
    notes = [_make_note("n1", 0.10, 0.30)]
    frames = [
        AudioFrameFeature(
            frame_index=0,
            start_sec=0.10,
            end_sec=0.15,
            rms=0.02,
            peak=0.7,
            zero_crossing_rate=0.2,
            spectral_centroid_hz=300.0,
            spectral_rolloff_hz=1000.0,
            is_silent=False,
            onset_score=0.3,
        ),
        AudioFrameFeature(
            frame_index=1,
            start_sec=0.15,
            end_sec=0.20,
            rms=0.015,
            peak=0.6,
            zero_crossing_rate=0.2,
            spectral_centroid_hz=320.0,
            spectral_rolloff_hz=1050.0,
            is_silent=False,
            onset_score=0.02,
        ),
    ]
    notes_path, audio_path = _write_documents(tmp_path, notes, frames)

    document, report = validate_midi_vs_audio(
        notes_file=notes_path,
        audio_features_file=audio_path,
        params=ValidationParameters(),
    )

    assert report.keep_count == 1
    assert document.validations[0].recommended_action == "KEEP"


def test_note_in_silence_gets_mute_candidate(tmp_path: Path) -> None:
    notes = [_make_note("n2", 0.40, 0.60)]
    frames = [
        AudioFrameFeature(
            frame_index=0,
            start_sec=0.40,
            end_sec=0.45,
            rms=0.0,
            peak=0.0,
            zero_crossing_rate=0.0,
            spectral_centroid_hz=None,
            spectral_rolloff_hz=None,
            is_silent=True,
            onset_score=0.0,
        )
    ]
    notes_path, audio_path = _write_documents(tmp_path, notes, frames)

    document, report = validate_midi_vs_audio(
        notes_file=notes_path,
        audio_features_file=audio_path,
        params=ValidationParameters(),
    )

    assert report.mute_candidate_count == 1
    assert document.validations[0].recommended_action == "MUTE_CANDIDATE"


def test_uncertain_note_gets_review(tmp_path: Path) -> None:
    notes = [_make_note("n3", 0.20, 0.50)]
    frames = [
        AudioFrameFeature(
            frame_index=0,
            start_sec=0.20,
            end_sec=0.30,
            rms=0.0006,
            peak=0.0012,
            zero_crossing_rate=0.1,
            spectral_centroid_hz=250.0,
            spectral_rolloff_hz=900.0,
            is_silent=False,
            onset_score=0.006,
        ),
        AudioFrameFeature(
            frame_index=1,
            start_sec=0.30,
            end_sec=0.40,
            rms=0.0006,
            peak=0.0012,
            zero_crossing_rate=0.1,
            spectral_centroid_hz=250.0,
            spectral_rolloff_hz=900.0,
            is_silent=True,
            onset_score=0.002,
        ),
    ]
    notes_path, audio_path = _write_documents(tmp_path, notes, frames)

    document, report = validate_midi_vs_audio(
        notes_file=notes_path,
        audio_features_file=audio_path,
        params=ValidationParameters(),
    )

    assert report.review_count == 1
    assert document.validations[0].recommended_action == "REVIEW"


def test_layer_mismatch_adds_warning_but_succeeds(tmp_path: Path) -> None:
    notes = [_make_note("n4", 0.10, 0.20, layer="bass")]
    frames = [
        AudioFrameFeature(
            frame_index=0,
            start_sec=0.10,
            end_sec=0.20,
            rms=0.02,
            peak=0.2,
            zero_crossing_rate=0.1,
            spectral_centroid_hz=200.0,
            spectral_rolloff_hz=700.0,
            is_silent=False,
            onset_score=0.2,
        )
    ]
    notes_path, audio_path = _write_documents(
        tmp_path, notes, frames, note_layer="bass", audio_layer="lead"
    )

    _document, report = validate_midi_vs_audio(
        notes_file=notes_path,
        audio_features_file=audio_path,
        params=ValidationParameters(),
    )

    assert report.status == "ok"
    assert report.warning_count == 1
    assert "Layer mismatch" in report.warnings[0]


def test_cli_writes_validation_and_report_and_counts(tmp_path: Path) -> None:
    notes = [
        _make_note("n_keep", 0.10, 0.30),
        _make_note("n_review", 0.40, 0.60),
        _make_note("n_mute", 0.75, 0.90),
    ]
    frames = [
        AudioFrameFeature(
            frame_index=0,
            start_sec=0.10,
            end_sec=0.20,
            rms=0.03,
            peak=0.8,
            zero_crossing_rate=0.2,
            spectral_centroid_hz=300.0,
            spectral_rolloff_hz=1100.0,
            is_silent=False,
            onset_score=0.4,
        ),
        AudioFrameFeature(
            frame_index=1,
            start_sec=0.40,
            end_sec=0.50,
            rms=0.0006,
            peak=0.0012,
            zero_crossing_rate=0.1,
            spectral_centroid_hz=250.0,
            spectral_rolloff_hz=900.0,
            is_silent=False,
            onset_score=0.006,
        ),
        AudioFrameFeature(
            frame_index=2,
            start_sec=0.50,
            end_sec=0.60,
            rms=0.0006,
            peak=0.0012,
            zero_crossing_rate=0.1,
            spectral_centroid_hz=250.0,
            spectral_rolloff_hz=900.0,
            is_silent=True,
            onset_score=0.002,
        ),
        AudioFrameFeature(
            frame_index=3,
            start_sec=0.75,
            end_sec=0.85,
            rms=0.0,
            peak=0.0,
            zero_crossing_rate=0.0,
            spectral_centroid_hz=None,
            spectral_rolloff_hz=None,
            is_silent=True,
            onset_score=0.0,
        ),
    ]

    notes_path, audio_path = _write_documents(tmp_path, notes, frames)
    output_path = tmp_path / "out" / "note_validation.json"
    report_path = tmp_path / "out" / "midi_audio_validation_report.json"

    result = runner.invoke(
        app,
        [
            "validate",
            "midi-vs-audio",
            "--notes",
            str(notes_path),
            "--audio-features",
            str(audio_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert report_path.exists()

    validation_doc = json.loads(output_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    actions = [item["recommended_action"] for item in validation_doc["validations"]]
    assert "KEEP" in actions
    assert "REVIEW" in actions
    assert "MUTE_CANDIDATE" in actions

    assert report["keep_count"] == 1
    assert report["review_count"] == 1
    assert report["mute_candidate_count"] == 1
