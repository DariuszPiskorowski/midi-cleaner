from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mido
from typer.testing import CliRunner

from midi_cleaner.cli import app


runner = CliRunner()


def _write_midi_file(
    path: Path,
    tracks: list[mido.MidiTrack],
    *,
    ticks_per_beat: int = 480,
    midi_type: int = 1,
) -> None:
    midi = mido.MidiFile(type=midi_type, ticks_per_beat=ticks_per_beat)
    for track in tracks:
        midi.tracks.append(track)
    midi.save(path)


def _build_multitrack_drum_source(
    path: Path,
    *,
    include_unmapped: bool = False,
    include_other_channel_control: bool = False,
) -> None:
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("track_name", name="tempo", time=0))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=487805, time=0))
    tempo_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    tempo_track.append(mido.MetaMessage("key_signature", key="C", time=0))
    tempo_track.append(mido.MetaMessage("marker", text="A", time=240))
    tempo_track.append(mido.MetaMessage("end_of_track", time=720))

    drum_a = mido.MidiTrack()
    drum_a.append(mido.MetaMessage("track_name", name="drum_a", time=0))
    drum_a.append(mido.Message("program_change", program=118, channel=9, time=0))
    drum_a.append(mido.Message("note_on", note=36, velocity=100, channel=9, time=0))
    drum_a.append(mido.Message("note_off", note=36, velocity=0, channel=9, time=120))
    drum_a.append(mido.Message("note_on", note=39, velocity=98, channel=9, time=120))
    drum_a.append(mido.Message("note_off", note=39, velocity=0, channel=9, time=120))
    if include_unmapped:
        drum_a.append(mido.Message("note_on", note=70, velocity=90, channel=9, time=120))
        drum_a.append(mido.Message("note_off", note=70, velocity=0, channel=9, time=120))
        drum_a.append(mido.MetaMessage("end_of_track", time=360))
    else:
        drum_a.append(mido.MetaMessage("end_of_track", time=600))

    drum_b = mido.MidiTrack()
    drum_b.append(mido.MetaMessage("track_name", name="drum_b", time=0))
    drum_b.append(mido.Message("program_change", program=118, channel=9, time=0))
    if include_other_channel_control:
        drum_b.append(mido.Message("control_change", control=1, value=64, channel=2, time=0))
    drum_b.append(mido.Message("note_on", note=46, velocity=96, channel=9, time=60))
    drum_b.append(mido.Message("note_off", note=46, velocity=0, channel=9, time=120))
    drum_b.append(mido.Message("note_on", note=57, velocity=96, channel=9, time=120))
    drum_b.append(mido.Message("note_off", note=57, velocity=0, channel=9, time=120))
    drum_b.append(mido.Message("note_on", note=32, velocity=96, channel=9, time=120))
    drum_b.append(mido.Message("note_off", note=32, velocity=0, channel=9, time=120))
    drum_b.append(mido.MetaMessage("end_of_track", time=300))

    _write_midi_file(path, [tempo_track, drum_a, drum_b], ticks_per_beat=480, midi_type=1)


def _note_events(path: Path) -> list[tuple[str, int, int, int]]:
    midi = mido.MidiFile(str(path))
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
    return sorted(events, key=lambda item: (item[3], item[0], item[1], item[2]))


def _max_tick(path: Path) -> int:
    midi = mido.MidiFile(str(path))
    max_tick = 0
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
        max_tick = max(max_tick, tick)
    return max_tick


def _meta_count(path: Path, event_type: str) -> int:
    midi = mido.MidiFile(str(path))
    return sum(1 for track in midi.tracks for msg in track if msg.is_meta and msg.type == event_type)


def test_remap_audit_detects_source_pitches_from_multitrack_midi(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    report = tmp_path / "drum_remap_report.json"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "ujam-candy",
            "--dry-run",
            "--report",
            str(report),
        ],
    )

    payload = json.loads(report.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert payload["source_track_count"] == 3
    assert payload["source_pitch_counts"]["32"] > 0
    assert payload["source_pitch_counts"]["36"] > 0
    assert payload["source_pitch_counts"]["39"] > 0
    assert payload["source_pitch_counts"]["46"] > 0
    assert payload["source_pitch_counts"]["57"] > 0


def test_remap_changes_note_numbers_and_preserves_timing(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_ujam_candy.mid"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "ujam-candy",
            "--output",
            str(output),
        ],
    )

    source_events = _note_events(source)
    output_events = _note_events(output)
    mapping = {
        32: 36,
        36: 36,
        39: 43,
        45: 50,
        46: 48,
        50: 53,
        57: 60,
    }
    expected_events = [
        (kind, mapping.get(note, note), tick)
        for kind, note, _channel, tick in source_events
    ]
    actual_events = [(kind, note, tick) for kind, note, _channel, tick in output_events]

    assert result.exit_code == 0
    assert expected_events == actual_events


def test_ujam_candy_maps_39_to_g1_midi_43_with_c1_36(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_ujam_candy.mid"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "ujam-candy",
            "--c1-midi-note",
            "36",
            "--output",
            str(output),
        ],
    )

    note_ons = [
        note for kind, note, _channel, _tick in _note_events(output) if kind == "on"
    ]

    assert result.exit_code == 0
    assert 43 in note_ons
    assert 39 not in note_ons


def test_ujam_candy_maps_46_to_c2_midi_48_with_c1_36(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_ujam_candy.mid"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "ujam-candy",
            "--c1-midi-note",
            "36",
            "--output",
            str(output),
        ],
    )

    note_ons = [
        note for kind, note, _channel, _tick in _note_events(output) if kind == "on"
    ]

    assert result.exit_code == 0
    assert 48 in note_ons
    assert 46 not in note_ons


def test_ujam_candy_maps_57_to_c3_midi_60_with_c1_36(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_ujam_candy.mid"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "ujam-candy",
            "--c1-midi-note",
            "36",
            "--output",
            str(output),
        ],
    )

    note_ons = [
        note for kind, note, _channel, _tick in _note_events(output) if kind == "on"
    ]

    assert result.exit_code == 0
    assert 60 in note_ons
    assert 57 not in note_ons


def test_ujam_candy_c1_midi_note_24_shifts_resolved_notes_down_by_12(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    report_c1_36 = tmp_path / "report_c1_36.json"
    report_c1_24 = tmp_path / "report_c1_24.json"
    _build_multitrack_drum_source(source)

    result_c1_36 = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "ujam-candy",
            "--c1-midi-note",
            "36",
            "--dry-run",
            "--report",
            str(report_c1_36),
        ],
    )
    result_c1_24 = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "ujam-candy",
            "--c1-midi-note",
            "24",
            "--dry-run",
            "--report",
            str(report_c1_24),
        ],
    )

    mapping_36 = json.loads(report_c1_36.read_text(encoding="utf-8"))[
        "resolved_mapping_note_numbers"
    ]
    mapping_24 = json.loads(report_c1_24.read_text(encoding="utf-8"))[
        "resolved_mapping_note_numbers"
    ]

    assert result_c1_36.exit_code == 0
    assert result_c1_24.exit_code == 0
    for source_note in ("32", "36", "39", "45", "46", "50", "57"):
        assert mapping_24[source_note] == mapping_36[source_note] - 12


def test_remap_preserves_tempo_and_time_signature(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_gm.mid"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "gm",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert _meta_count(source, "set_tempo") == _meta_count(output, "set_tempo")
    assert _meta_count(source, "time_signature") == _meta_count(output, "time_signature")


def test_remap_does_not_modify_source_file(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_sitala.mid"
    _build_multitrack_drum_source(source)
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "sitala",
            "--output",
            str(output),
        ],
    )

    after_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    assert result.exit_code == 0
    assert before_hash == after_hash


def test_remap_merge_tracks_outputs_single_internal_track(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_gm.mid"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "gm",
            "--output",
            str(output),
            "--merge-tracks",
        ],
    )

    midi = mido.MidiFile(str(output))

    assert result.exit_code == 0
    assert len(midi.tracks) == 1
    assert midi.tracks[0][0].type == "track_name"
    assert midi.tracks[0][0].name == "drums_gm"


def test_remap_type0_output_has_one_track(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_type0.mid"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "gm",
            "--no-merge-tracks",
            "--format",
            "type0",
            "--output",
            str(output),
        ],
    )

    midi = mido.MidiFile(str(output))

    assert result.exit_code == 0
    assert midi.type == 0
    assert len(midi.tracks) == 1


def test_remap_channel_policy_single_forces_selected_channel(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_channel.mid"
    _build_multitrack_drum_source(source, include_other_channel_control=True)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "gm",
            "--output",
            str(output),
            "--channel-policy",
            "single",
            "--force-channel",
            "10",
        ],
    )

    midi = mido.MidiFile(str(output))
    channels = {
        int(msg.channel)
        for track in midi.tracks
        for msg in track
        if not msg.is_meta and hasattr(msg, "channel")
    }

    assert result.exit_code == 0
    assert channels == {9}


def test_remap_strips_program_changes_by_default(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_gm.mid"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "gm",
            "--output",
            str(output),
        ],
    )

    midi = mido.MidiFile(str(output))
    program_change_count = sum(
        1 for track in midi.tracks for msg in track if not msg.is_meta and msg.type == "program_change"
    )

    assert result.exit_code == 0
    assert program_change_count == 0


def test_remap_unmapped_keep_warns_and_keeps_notes_by_default(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_ujam_candy.mid"
    report = tmp_path / "report.json"
    _build_multitrack_drum_source(source, include_unmapped=True)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "ujam-candy",
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    notes = [event[1] for event in _note_events(output)]

    assert result.exit_code == 0
    assert 70 in notes
    assert payload["unmapped_pitches"] == [70]
    assert any("kept" in warning for warning in payload["warnings"])


def test_remap_custom_map_json_is_applied(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_custom.mid"
    map_file = tmp_path / "custom_map.json"
    _build_multitrack_drum_source(source)

    custom_map = {
        "name": "ujam_candy_custom",
        "output_channel": 7,
        "notes": {
            "36": 36,
            "39": 40,
            "46": 46,
            "57": 49,
            "45": 45,
            "50": 50,
            "32": 36,
        },
        "labels": {
            "36": "kick",
            "39": "clap_or_snare",
            "46": "open_hat",
            "57": "crash",
            "45": "low_tom",
            "50": "high_tom",
            "32": "low_kick_or_artifact",
        },
    }
    map_file.write_text(json.dumps(custom_map, indent=2) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "custom",
            "--map-file",
            str(map_file),
            "--output",
            str(output),
        ],
    )

    events = _note_events(output)
    note_numbers = [event[1] for event in events]
    channels = [event[2] for event in events]

    assert result.exit_code == 0
    assert 40 in note_numbers
    assert 39 not in note_numbers
    assert set(channels) == {7}


def test_ujam_candy_preserves_timing_and_source_length_ticks(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_ujam_candy.mid"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "ujam-candy",
            "--output",
            str(output),
        ],
    )

    midi = mido.MidiFile(str(output))

    assert result.exit_code == 0
    assert _max_tick(source) == _max_tick(output)
    for track in midi.tracks:
        end_of_track_count = sum(1 for msg in track if msg.is_meta and msg.type == "end_of_track")
        assert end_of_track_count == 1
        assert track[-1].is_meta and track[-1].type == "end_of_track"


def test_remap_dry_run_writes_no_output_midi(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_gm.mid"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "gm",
            "--output",
            str(output),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert not output.exists()


def test_remap_report_contains_source_and_remapped_pitch_counts(tmp_path: Path) -> None:
    source = tmp_path / "drums.mid"
    output = tmp_path / "drums_ujam_candy.mid"
    report = tmp_path / "drum_remap_report.json"
    _build_multitrack_drum_source(source)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "ujam-candy",
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )

    payload = json.loads(report.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert "source_pitch_counts" in payload
    assert "remapped_pitch_counts" in payload
    assert "target_key_layout_name" in payload
    assert "c1_midi_note" in payload
    assert "resolved_target_note_names" in payload
    assert "resolved_mapping_note_numbers" in payload
    assert payload["source_pitch_counts"]["39"] > 0
    assert payload["remapped_pitch_counts"]["43"] > 0
    assert payload["target_key_layout_name"] == "ujam-candy-observed-ui"
    assert payload["c1_midi_note"] == 36
    assert payload["resolved_target_note_names"]["39"] == "G1"
    assert payload["resolved_mapping_note_numbers"]["39"] == 43


def test_remap_report_counts_only_real_note_on_events(tmp_path: Path) -> None:
    source = tmp_path / "note_on_only_count.mid"
    output = tmp_path / "note_on_only_count_out.mid"
    report = tmp_path / "note_on_only_count_report.json"

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    tempo_track.append(mido.MetaMessage("end_of_track", time=480))

    drum_track = mido.MidiTrack()
    drum_track.append(mido.Message("note_on", note=36, velocity=100, channel=9, time=0))
    drum_track.append(mido.Message("note_off", note=36, velocity=0, channel=9, time=120))
    drum_track.append(mido.Message("note_on", note=36, velocity=0, channel=9, time=60))
    drum_track.append(mido.Message("note_off", note=36, velocity=0, channel=9, time=60))
    drum_track.append(mido.Message("note_on", note=39, velocity=110, channel=9, time=60))
    drum_track.append(mido.Message("note_off", note=39, velocity=0, channel=9, time=120))
    drum_track.append(mido.MetaMessage("end_of_track", time=240))

    _write_midi_file(source, [tempo_track, drum_track], ticks_per_beat=480, midi_type=1)

    result = runner.invoke(
        app,
        [
            "midi",
            "remap-drums",
            "--input",
            str(source),
            "--target-map",
            "ujam-candy",
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )

    payload = json.loads(report.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert payload["source_pitch_counts"] == {"36": 1, "39": 1}
    assert payload["remapped_pitch_counts"] == {"36": 1, "43": 1}