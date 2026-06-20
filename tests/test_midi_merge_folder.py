from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mido
from typer.testing import CliRunner

from midi_cleaner.cli import app


runner = CliRunner()


def _write_midi_file(path: Path, tracks: list[mido.MidiTrack], ticks_per_beat: int = 480) -> None:
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for track in tracks:
        midi.tracks.append(track)
    midi.save(path)


def _single_track_midi(path: Path) -> None:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="single", time=0))
    track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=240))
    track.append(mido.MetaMessage("end_of_track", time=0))
    _write_midi_file(path, [track])


def _multi_track_midi(path: Path) -> None:
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("track_name", name="tempo", time=0))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    tempo_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    tempo_track.append(mido.MetaMessage("marker", text="A", time=120))
    tempo_track.append(mido.MetaMessage("end_of_track", time=840))

    bass_track = mido.MidiTrack()
    bass_track.append(mido.MetaMessage("track_name", name="bass", time=0))
    bass_track.append(mido.Message("program_change", program=33, channel=2, time=0))
    bass_track.append(mido.Message("note_on", note=40, velocity=100, channel=2, time=120))
    bass_track.append(mido.Message("note_off", note=40, velocity=0, channel=2, time=240))
    bass_track.append(mido.MetaMessage("end_of_track", time=600))

    lead_track = mido.MidiTrack()
    lead_track.append(mido.MetaMessage("track_name", name="lead", time=0))
    lead_track.append(mido.Message("program_change", program=81, channel=7, time=0))
    lead_track.append(mido.Message("note_on", note=64, velocity=96, channel=7, time=360))
    lead_track.append(mido.Message("note_off", note=64, velocity=0, channel=7, time=240))
    lead_track.append(mido.MetaMessage("end_of_track", time=360))

    _write_midi_file(path, [tempo_track, bass_track, lead_track], ticks_per_beat=480)


def _absolute_note_events(midi_path: Path) -> list[tuple[str, int, int, int]]:
    midi = mido.MidiFile(str(midi_path))
    events: list[tuple[str, int, int, int]] = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if message.is_meta:
                continue
            if message.type == "note_on" and message.velocity > 0:
                events.append(("on", int(message.note), int(message.channel), tick))
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                events.append(("off", int(message.note), int(message.channel), tick))
    return sorted(events)


def _max_tick(midi_path: Path) -> int:
    midi = mido.MidiFile(str(midi_path))
    max_tick = 0
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
        max_tick = max(max_tick, tick)
    return max_tick


def test_merge_folder_finds_mid_and_midi_files(tmp_path: Path) -> None:
    _single_track_midi(tmp_path / "one.mid")
    _single_track_midi(tmp_path / "two.midi")

    result = runner.invoke(
        app,
        [
            "midi",
            "merge-folder",
            "--folder",
            str(tmp_path),
            "--dry-run",
            "--yes",
        ],
    )

    assert result.exit_code == 0
    assert "midi_file_count=2" in result.stdout


def test_merge_folder_skips_single_track_midi(tmp_path: Path) -> None:
    _single_track_midi(tmp_path / "single.mid")

    result = runner.invoke(
        app,
        ["midi", "merge-folder", "--folder", str(tmp_path), "--yes"],
    )

    assert result.exit_code == 0
    assert not (tmp_path / "single_merge.mid").exists()
    assert "multitrack_file_count=0" in result.stdout


def test_merge_folder_interactive_prompt_merges_when_yes(tmp_path: Path) -> None:
    source = tmp_path / "layered.mid"
    _multi_track_midi(source)

    result = runner.invoke(
        app,
        ["midi", "merge-folder", "--folder", str(tmp_path)],
        input="y\n",
    )

    assert result.exit_code == 0
    assert "Detected multi-track MIDI: layered.mid (tracks=3)" in result.stdout
    assert (tmp_path / "layered_merge.mid").exists()


def test_merge_folder_yes_merges_all_without_prompt(tmp_path: Path) -> None:
    _multi_track_midi(tmp_path / "a.mid")
    _multi_track_midi(tmp_path / "b.mid")

    result = runner.invoke(
        app,
        ["midi", "merge-folder", "--folder", str(tmp_path), "--yes"],
    )

    assert result.exit_code == 0
    assert "Merge this MIDI file?" not in result.stdout
    assert (tmp_path / "a_merge.mid").exists()
    assert (tmp_path / "b_merge.mid").exists()


def test_merge_folder_dry_run_writes_no_files(tmp_path: Path) -> None:
    _multi_track_midi(tmp_path / "dry.mid")

    result = runner.invoke(
        app,
        ["midi", "merge-folder", "--folder", str(tmp_path), "--dry-run", "--yes"],
    )

    assert result.exit_code == 0
    assert not (tmp_path / "dry_merge.mid").exists()
    assert "dry_run=true" in result.stdout


def test_merge_folder_uses_unique_name_when_output_exists(tmp_path: Path) -> None:
    _multi_track_midi(tmp_path / "song.mid")
    _single_track_midi(tmp_path / "song_merge.mid")

    result = runner.invoke(
        app,
        ["midi", "merge-folder", "--folder", str(tmp_path), "--yes"],
    )

    assert result.exit_code == 0
    existing = mido.MidiFile(str(tmp_path / "song_merge.mid"))
    assert len(existing.tracks) == 1
    assert (tmp_path / "song_merge_2.mid").exists()


def test_merge_folder_does_not_modify_source_file(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    _multi_track_midi(source)
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    result = runner.invoke(
        app,
        ["midi", "merge-folder", "--folder", str(tmp_path), "--yes"],
    )

    after_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    assert result.exit_code == 0
    assert before_hash == after_hash


def test_merge_output_has_single_track_and_type0(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    _multi_track_midi(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "merge-folder",
            "--folder",
            str(tmp_path),
            "--yes",
            "--format",
            "type0",
        ],
    )

    merged = mido.MidiFile(str(tmp_path / "source_merge.mid"))

    assert result.exit_code == 0
    assert merged.type == 0
    assert len(merged.tracks) == 1


def test_merge_preserves_absolute_note_timing_and_meta_and_duration(tmp_path: Path) -> None:
    source = tmp_path / "timing.mid"
    _multi_track_midi(source)

    result = runner.invoke(
        app,
        ["midi", "merge-folder", "--folder", str(tmp_path), "--yes"],
    )

    merged = tmp_path / "timing_merge.mid"
    source_midi = mido.MidiFile(str(source))
    merged_midi = mido.MidiFile(str(merged))

    source_meta = [msg for track in source_midi.tracks for msg in track if msg.is_meta]
    merged_meta = [msg for track in merged_midi.tracks for msg in track if msg.is_meta]

    assert result.exit_code == 0
    assert _absolute_note_events(source) == _absolute_note_events(merged)
    assert sum(1 for msg in source_meta if msg.type == "set_tempo") == sum(
        1 for msg in merged_meta if msg.type == "set_tempo"
    )
    assert sum(1 for msg in source_meta if msg.type == "time_signature") == sum(
        1 for msg in merged_meta if msg.type == "time_signature"
    )
    assert _max_tick(source) == _max_tick(merged)


def test_merge_channel_policy_preserve_keeps_source_channels(tmp_path: Path) -> None:
    source = tmp_path / "channels.mid"
    _multi_track_midi(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "merge-folder",
            "--folder",
            str(tmp_path),
            "--yes",
            "--channel-policy",
            "preserve",
        ],
    )

    merged = mido.MidiFile(str(tmp_path / "channels_merge.mid"))
    channels = {
        int(msg.channel)
        for track in merged.tracks
        for msg in track
        if not msg.is_meta and hasattr(msg, "channel")
    }

    assert result.exit_code == 0
    assert 2 in channels
    assert 7 in channels


def test_merge_channel_policy_single_forces_channel_zero(tmp_path: Path) -> None:
    source = tmp_path / "single_channel.mid"
    _multi_track_midi(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "merge-folder",
            "--folder",
            str(tmp_path),
            "--yes",
            "--channel-policy",
            "single",
        ],
    )

    merged = mido.MidiFile(str(tmp_path / "single_channel_merge.mid"))
    channels = {
        int(msg.channel)
        for track in merged.tracks
        for msg in track
        if not msg.is_meta and hasattr(msg, "channel")
    }

    assert result.exit_code == 0
    assert channels == {0}


def test_merge_folder_writes_json_report_when_requested(tmp_path: Path) -> None:
    source = tmp_path / "report.mid"
    _multi_track_midi(source)
    report_path = tmp_path / "merge_report.json"

    result = runner.invoke(
        app,
        [
            "midi",
            "merge-folder",
            "--folder",
            str(tmp_path),
            "--yes",
            "--report",
            str(report_path),
        ],
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert payload["midi_file_count"] == 1
    assert payload["multitrack_file_count"] == 1
    assert payload["merged_file_count"] == 1
    assert payload["files"][0]["source_file"].endswith("report.mid")
    assert payload["files"][0]["action"] == "merged"
