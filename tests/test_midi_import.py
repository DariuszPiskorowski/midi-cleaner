from __future__ import annotations

import json
from pathlib import Path

import mido
from typer.testing import CliRunner

from midi_cleaner.cli import app
from midi_cleaner.midi.importer import import_midi_candidate


runner = CliRunner()


def _write_midi_file(path: Path, tracks: list[mido.MidiTrack], ticks_per_beat: int = 480) -> None:
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in tracks:
        midi.tracks.append(track)
    midi.save(path)


def _write_long_invalid_key_signature_midi(path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi.tracks.append(tempo_track)

    note_track = mido.MidiTrack()
    note_track.append(mido.MetaMessage("track_name", name="Played With Fire Synth", time=0))
    note_track.append(mido.MetaMessage("key_signature", key="C", time=0))

    events = [
        (0, 480, 60, 100),
        (38400, 960, 62, 96),
        (60000, 960, 64, 94),
    ]

    current_tick = 0
    for start_tick, duration_ticks, note, velocity in events:
        delta = max(0, int(start_tick) - int(current_tick))
        note_track.append(
            mido.Message("note_on", note=int(note), velocity=int(velocity), channel=0, time=delta)
        )
        note_track.append(
            mido.Message("note_off", note=int(note), velocity=0, channel=0, time=int(duration_ticks))
        )
        current_tick = int(start_tick) + int(duration_ticks)

    midi.tracks.append(note_track)
    midi.save(path)

    raw = bytearray(path.read_bytes())
    marker = bytes([0xFF, 0x59, 0x02])
    marker_index = raw.find(marker)
    assert marker_index >= 0
    raw[marker_index + 3] = 0x0E  # 14 sharps forces strict mido key-signature decode failure.
    raw[marker_index + 4] = 0x01
    path.write_bytes(raw)


def test_import_simple_single_note(tmp_path: Path) -> None:
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=100, time=0, channel=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480, channel=0))

    input_path = tmp_path / "single_note.mid"
    _write_midi_file(input_path, [track])

    document, report = import_midi_candidate(input_path, source="ripx", layer="bass")

    assert report.status == "ok"
    assert report.note_count == 1
    note = document.notes[0]
    assert note.pitch_midi == 60
    assert note.pitch_name == "C4"
    assert note.duration_ticks == 480


def test_note_on_velocity_zero_is_note_off(tmp_path: Path) -> None:
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=64, velocity=90, time=0, channel=0))
    track.append(mido.Message("note_on", note=64, velocity=0, time=240, channel=0))

    input_path = tmp_path / "velocity_zero.mid"
    _write_midi_file(input_path, [track])

    document, report = import_midi_candidate(input_path, source="suno", layer="lead")

    assert report.note_count == 1
    note = document.notes[0]
    assert note.end_tick == 240


def test_track_name_is_preserved(tmp_path: Path) -> None:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Bassline", time=0))
    track.append(mido.Message("note_on", note=45, velocity=100, time=0, channel=0))
    track.append(mido.Message("note_off", note=45, velocity=0, time=120, channel=0))

    input_path = tmp_path / "track_name.mid"
    _write_midi_file(input_path, [track])

    document, _report = import_midi_candidate(input_path, source="manual", layer="bass")

    assert document.notes[0].track_name == "Bassline"


def test_default_tempo_used_without_tempo_event(tmp_path: Path) -> None:
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=100, time=0, channel=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480, channel=0))

    input_path = tmp_path / "default_tempo.mid"
    _write_midi_file(input_path, [track], ticks_per_beat=480)

    document, _report = import_midi_candidate(input_path, source="ripx", layer="lead")

    assert document.tempo_map[0].tempo_us_per_beat == 500000
    assert abs(document.notes[0].duration_sec - 0.5) < 1e-9


def test_tempo_change_maps_seconds_deterministically(tmp_path: Path) -> None:
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=1000000, time=480))

    note_track = mido.MidiTrack()
    note_track.append(mido.Message("note_on", note=67, velocity=100, time=0, channel=0))
    note_track.append(mido.Message("note_off", note=67, velocity=0, time=960, channel=0))

    input_path = tmp_path / "tempo_change.mid"
    _write_midi_file(input_path, [tempo_track, note_track], ticks_per_beat=480)

    document, _report = import_midi_candidate(input_path, source="ripx", layer="lead")

    assert abs(document.notes[0].duration_sec - 1.5) < 1e-9


def test_import_leniently_handles_invalid_key_signature_metadata(tmp_path: Path) -> None:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("key_signature", key="C", time=0))
    track.append(mido.Message("note_on", note=60, velocity=100, time=0, channel=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480, channel=0))

    input_path = tmp_path / "invalid_key_signature.mid"
    _write_midi_file(input_path, [track])

    raw = bytearray(input_path.read_bytes())
    marker = bytes([0xFF, 0x59, 0x02])
    marker_index = raw.find(marker)
    assert marker_index >= 0

    raw[marker_index + 3] = 0x0E  # 14 sharps is unsupported by strict mido decoding.
    raw[marker_index + 4] = 0x01
    input_path.write_bytes(raw)

    document, report = import_midi_candidate(input_path, source="manual", layer="midi")

    assert report.status == "ok"
    assert report.note_count == 1
    assert len(document.notes) == 1
    assert report.warning_count >= 1
    assert any("unsupported key signature metadata" in warning for warning in report.warnings)


def test_import_lenient_key_signature_preserves_notes_after_40_seconds(tmp_path: Path) -> None:
    input_path = tmp_path / "Played_With_Fire_-_Deep_House__Synth___Synth_.mid"
    _write_long_invalid_key_signature_midi(input_path)

    document, report = import_midi_candidate(input_path, source="manual", layer="midi")

    max_end_tick = max((int(note.end_tick) for note in document.notes), default=0)
    max_end_sec = max((float(note.end_sec) for note in document.notes), default=0.0)
    notes_after_40 = [note for note in document.notes if float(note.end_sec) >= 40.0]

    assert report.status == "ok"
    assert report.warning_count >= 1
    assert any("unsupported key signature metadata" in warning for warning in report.warnings)
    assert len(document.notes) == 3
    assert max_end_tick > 38400
    assert max_end_sec > 40.0
    assert len(notes_after_40) >= 1


def test_cli_import_writes_output_and_report(tmp_path: Path) -> None:
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=50, velocity=90, time=0, channel=0))
    track.append(mido.Message("note_off", note=50, velocity=0, time=240, channel=0))

    input_path = tmp_path / "cli_input.mid"
    output_path = tmp_path / "out" / "note_events.json"
    report_path = tmp_path / "out" / "midi_import_report.json"
    _write_midi_file(input_path, [track])

    result = runner.invoke(
        app,
        [
            "midi",
            "import-candidate",
            str(input_path),
            "--source",
            "ripx",
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

    note_doc = json.loads(output_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert note_doc["source"] == "ripx"
    assert note_doc["layer"] == "bass"
    assert len(note_doc["notes"]) == 1
    assert report["status"] == "ok"
    assert report["output_file"] == str(output_path)
