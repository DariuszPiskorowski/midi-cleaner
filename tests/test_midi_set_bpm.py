from __future__ import annotations

from pathlib import Path

import mido

from midi_cleaner.midi.set_bpm import set_midi_bpm


def _write_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("track_name", name="tempo", time=0))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=480000, time=240))
    tempo_track.append(mido.MetaMessage("end_of_track", time=240))

    note_track = mido.MidiTrack()
    note_track.append(mido.MetaMessage("track_name", name="notes", time=0))
    note_track.append(mido.Message("note_on", note=40, velocity=90, channel=1, time=0))
    note_track.append(mido.Message("note_off", note=40, velocity=0, channel=1, time=120))
    note_track.append(mido.Message("note_on", note=43, velocity=85, channel=1, time=120))
    note_track.append(mido.Message("note_off", note=43, velocity=0, channel=1, time=240))
    note_track.append(mido.MetaMessage("end_of_track", time=0))

    midi.tracks.extend([tempo_track, note_track])
    midi.save(path)


def _note_events_with_ticks(path: Path) -> list[tuple[int, str, int, int, int]]:
    midi = mido.MidiFile(str(path))
    events: list[tuple[int, str, int, int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for message in track:
            tick += int(message.time)
            if message.is_meta:
                continue
            if message.type == "note_on" and message.velocity > 0:
                events.append((track_index, "on", int(message.note), int(message.channel), tick))
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                events.append((track_index, "off", int(message.note), int(message.channel), tick))
    return events


def test_set_bpm_preserves_note_ticks_and_track_count(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    output = tmp_path / "set_bpm.mid"
    _write_midi(source)

    source_events = _note_events_with_ticks(source)
    source_track_count = len(mido.MidiFile(str(source)).tracks)

    report = set_midi_bpm(input_file=source, output_file=output, bpm=124.529)

    output_midi = mido.MidiFile(str(output))
    output_events = _note_events_with_ticks(output)

    assert output.exists()
    assert report.output_file == str(output)
    assert len(output_midi.tracks) == source_track_count
    assert source_events == output_events

    tempo_events = [
        msg
        for track in output_midi.tracks
        for msg in track
        if msg.is_meta and msg.type == "set_tempo"
    ]
    assert len(tempo_events) == 1
    assert int(tempo_events[0].tempo) == int(round(60_000_000.0 / 124.529))
