from __future__ import annotations

import json
from pathlib import Path

import mido
from typer.testing import CliRunner

from midi_cleaner.alignment.models import AudioAlignedNoteDocument, AudioAlignedNoteEvent
from midi_cleaner.cleanup.midi_exporter import ReviewMidiExportParameters, export_review_midi
from midi_cleaner.cleanup.models import CleanupAction, CleanupPlanDocument
from midi_cleaner.cli import app
from midi_cleaner.midi.models import NoteEvent, NoteEventDocument, TempoEvent


runner = CliRunner()


def _note(
    note_id: str,
    pitch: int,
    velocity: int,
    start_tick: int,
    end_tick: int,
    layer: str = "bass",
) -> NoteEvent:
    return NoteEvent(
        note_id=note_id,
        source="ripx",
        layer=layer,
        track_index=0,
        track_name="Source",
        channel=0,
        pitch_midi=pitch,
        pitch_name="C4",
        velocity=velocity,
        start_tick=start_tick,
        end_tick=end_tick,
        duration_ticks=end_tick - start_tick,
        start_sec=0.0,
        end_sec=0.0,
        duration_sec=0.0,
    )


def _action(note_id: str, plan_action: str, confidence: float = 0.5) -> CleanupAction:
    return CleanupAction(
        note_id=note_id,
        original_recommended_action="REVIEW",
        plan_action=plan_action,
        confidence=confidence,
        reasons=["reason"],
        source_validation={"recommended_action": "REVIEW"},
    )


def _write_inputs(
    tmp_path: Path,
    notes: list[NoteEvent],
    actions: list[CleanupAction],
) -> tuple[Path, Path]:
    notes_doc = NoteEventDocument(
        schema_version="0.1.0",
        source_file="candidate.mid",
        source="ripx",
        layer="bass",
        ticks_per_beat=480,
        tempo_map=[TempoEvent(tick=0, tempo_us_per_beat=500000, sec=0.0)],
        notes=notes,
    )
    cleanup_doc = CleanupPlanDocument(
        schema_version="0.1.0",
        validation_file="note_validation.json",
        layer="bass",
        planner_parameters={
            "mute_threshold": 0.45,
            "review_threshold": 0.70,
            "delete_threshold": 0.20,
            "allow_delete_candidates": False,
        },
        actions=actions,
    )

    notes_path = tmp_path / "note_events.json"
    plan_path = tmp_path / "cleanup_plan.json"
    notes_path.write_text(notes_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    plan_path.write_text(cleanup_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return notes_path, plan_path


def _extract_note_pairs(midi_path: Path) -> list[tuple[int, int, int]]:
    midi_file = mido.MidiFile(str(midi_path))
    track = midi_file.tracks[0]

    absolute_tick = 0
    active: dict[tuple[int, int], list[tuple[int, int]]] = {}
    pairs: list[tuple[int, int, int]] = []

    for message in track:
        absolute_tick += message.time
        if message.type == "note_on" and message.velocity > 0:
            key = (message.channel, message.note)
            active.setdefault(key, []).append((absolute_tick, message.velocity))
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            key = (message.channel, message.note)
            if key in active and active[key]:
                start_tick, velocity = active[key].pop(0)
                pairs.append((message.note, velocity, absolute_tick - start_tick))

    return pairs


def _extract_note_on_times_sec(midi_path: Path) -> list[float]:
    midi_file = mido.MidiFile(str(midi_path))
    track = midi_file.tracks[0]

    absolute_tick = 0
    tempo_us_per_beat = 500000
    note_on_times: list[float] = []

    for message in track:
        absolute_tick += message.time
        if message.type == "set_tempo":
            tempo_us_per_beat = int(message.tempo)
        if message.type == "note_on" and message.velocity > 0:
            note_on_times.append(
                (absolute_tick / midi_file.ticks_per_beat) * (tempo_us_per_beat / 1_000_000)
            )

    return note_on_times


def _write_audio_aligned_notes(
    tmp_path: Path,
    note_id: str,
    aligned_start_sec: float,
    aligned_end_sec: float,
) -> Path:
    aligned_doc = AudioAlignedNoteDocument(
        schema_version="0.1.0",
        notes_file="note_events.json",
        audio_features_file="audio_features.json",
        layer="bass",
        sample_rate=44100,
        audio_duration_sec=2.0,
        alignment_parameters={
            "onset_search_window_ms": 250.0,
            "offset_search_window_ms": 350.0,
            "min_onset_score": 0.005,
            "min_rms": 0.001,
            "snap_start_to_audio_onset": True,
            "snap_end_to_energy_offset": True,
            "max_start_correction_ms": 500.0,
            "max_end_correction_ms": 800.0,
            "low_confidence_action": "KEEP_ORIGINAL_LOW_CONFIDENCE",
        },
        notes=[
            AudioAlignedNoteEvent(
                note_id=note_id,
                source="ripx",
                layer="bass",
                pitch_midi=67,
                pitch_name="G4",
                velocity=100,
                channel=0,
                original_start_sec=0.0,
                original_end_sec=0.0,
                original_duration_sec=0.0,
                original_start_tick=500,
                original_end_tick=700,
                aligned_start_sec=aligned_start_sec,
                aligned_end_sec=aligned_end_sec,
                aligned_duration_sec=aligned_end_sec - aligned_start_sec,
                start_correction_ms=0.0,
                end_correction_ms=0.0,
                duration_correction_ms=0.0,
                nearest_audio_onset_sec=aligned_start_sec,
                nearest_audio_offset_sec=aligned_end_sec,
                onset_error_before_ms=0.0,
                onset_error_after_ms=0.0,
                local_rms=0.02,
                local_onset_score=0.03,
                sustained_energy_ratio=0.8,
                alignment_confidence=0.9,
                alignment_action="ALIGNED",
                reasons=["test"],
            )
        ],
    )

    aligned_path = tmp_path / "audio_aligned_note_events.json"
    aligned_path.write_text(aligned_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return aligned_path


def test_keep_notes_export_only_to_keep_mid(tmp_path: Path) -> None:
    notes = [_note("n_keep", 60, 90, 120, 360)]
    actions = [_action("n_keep", "KEEP")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    report = export_review_midi(
        notes_file=notes_path,
        cleanup_plan_file=plan_path,
        output_dir=tmp_path / "out",
        params=ReviewMidiExportParameters(),
    )

    keep_path = tmp_path / "out" / "keep.mid"
    review_path = tmp_path / "out" / "review.mid"
    muted_path = tmp_path / "out" / "muted.mid"

    assert keep_path.exists()
    assert review_path.exists()
    assert muted_path.exists()
    assert _extract_note_pairs(keep_path) == [(60, 90, 240)]
    assert _extract_note_pairs(review_path) == []
    assert _extract_note_pairs(muted_path) == []
    assert any(item.action == "KEEP" and item.note_count == 1 for item in report.exported_files)


def test_review_notes_export_only_to_review_mid(tmp_path: Path) -> None:
    notes = [_note("n_review", 62, 70, 10, 90)]
    actions = [_action("n_review", "REVIEW")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    export_review_midi(notes_path, plan_path, tmp_path / "out", ReviewMidiExportParameters())

    assert _extract_note_pairs(tmp_path / "out" / "keep.mid") == []
    assert _extract_note_pairs(tmp_path / "out" / "review.mid") == [(62, 70, 80)]
    assert _extract_note_pairs(tmp_path / "out" / "muted.mid") == []


def test_mute_notes_export_only_to_muted_mid(tmp_path: Path) -> None:
    notes = [_note("n_mute", 64, 50, 0, 100)]
    actions = [_action("n_mute", "MUTE")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    export_review_midi(notes_path, plan_path, tmp_path / "out", ReviewMidiExportParameters())

    assert _extract_note_pairs(tmp_path / "out" / "keep.mid") == []
    assert _extract_note_pairs(tmp_path / "out" / "review.mid") == []
    assert _extract_note_pairs(tmp_path / "out" / "muted.mid") == [(64, 50, 100)]


def test_delete_candidates_exported_when_enabled(tmp_path: Path) -> None:
    notes = [_note("n_delete", 67, 100, 30, 230)]
    actions = [_action("n_delete", "DELETE_CANDIDATE")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    report = export_review_midi(
        notes_path,
        plan_path,
        tmp_path / "out",
        ReviewMidiExportParameters(include_delete_candidates=True),
    )

    delete_path = tmp_path / "out" / "delete_candidates.mid"
    assert delete_path.exists()
    assert _extract_note_pairs(delete_path) == [(67, 100, 200)]
    assert any(item.action == "DELETE_CANDIDATE" and item.note_count == 1 for item in report.exported_files)


def test_delete_candidates_skipped_when_disabled(tmp_path: Path) -> None:
    notes = [_note("n_delete", 69, 64, 40, 100)]
    actions = [_action("n_delete", "DELETE_CANDIDATE")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    report = export_review_midi(
        notes_path,
        plan_path,
        tmp_path / "out",
        ReviewMidiExportParameters(include_delete_candidates=False),
    )

    assert not (tmp_path / "out" / "delete_candidates.mid").exists()
    assert all(item.action != "DELETE_CANDIDATE" for item in report.exported_files)


def test_unknown_note_id_in_plan_creates_warning(tmp_path: Path) -> None:
    notes = [_note("n1", 60, 90, 0, 120)]
    actions = [_action("unknown_note", "KEEP")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    report = export_review_midi(notes_path, plan_path, tmp_path / "out", ReviewMidiExportParameters())

    assert report.warning_count >= 1
    assert any("unknown note_id" in warning for warning in report.warnings)


def test_cli_writes_midi_files_and_report(tmp_path: Path) -> None:
    notes = [
        _note("k", 60, 90, 0, 120),
        _note("r", 62, 70, 120, 240),
        _note("m", 64, 50, 240, 360),
        _note("d", 65, 80, 360, 480),
    ]
    actions = [
        _action("k", "KEEP"),
        _action("r", "REVIEW"),
        _action("m", "MUTE"),
        _action("d", "DELETE_CANDIDATE"),
    ]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)

    out_dir = tmp_path / "review_midi"
    report_path = out_dir / "export_report.json"

    result = runner.invoke(
        app,
        [
            "cleanup",
            "export-review-midi",
            "--notes",
            str(notes_path),
            "--plan",
            str(plan_path),
            "--output-dir",
            str(out_dir),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert (out_dir / "keep.mid").exists()
    assert (out_dir / "review.mid").exists()
    assert (out_dir / "muted.mid").exists()
    assert (out_dir / "delete_candidates.mid").exists()
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    by_action = {item["action"]: item["note_count"] for item in report["exported_files"]}
    assert by_action["KEEP"] == 1
    assert by_action["REVIEW"] == 1
    assert by_action["MUTE"] == 1
    assert by_action["DELETE_CANDIDATE"] == 1


def test_review_export_uses_audio_aligned_seconds_when_provided(tmp_path: Path) -> None:
    notes = [_note("n_keep", 67, 100, 500, 700)]
    actions = [_action("n_keep", "KEEP")]
    notes_path, plan_path = _write_inputs(tmp_path, notes, actions)
    aligned_path = _write_audio_aligned_notes(
        tmp_path,
        note_id="n_keep",
        aligned_start_sec=0.215,
        aligned_end_sec=0.341,
    )

    report = export_review_midi(
        notes_file=notes_path,
        cleanup_plan_file=plan_path,
        output_dir=tmp_path / "out",
        params=ReviewMidiExportParameters(
            ticks_per_beat=960,
            audio_aligned_notes_file=aligned_path,
        ),
    )

    note_on_times = _extract_note_on_times_sec(tmp_path / "out" / "keep.mid")

    assert note_on_times
    assert abs(note_on_times[0] - 0.215) <= 0.002
    assert report.timing_source == "audio_aligned_seconds"
    assert report.max_export_time_error_ms <= 2.0
    assert report.mean_export_time_error_ms <= 2.0
