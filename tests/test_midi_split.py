from __future__ import annotations

import re
from pathlib import Path

import mido
import pytest

from midi_cleaner.midi.importer import import_midi_candidate
from midi_cleaner.midi_split import (
    MidiSplitSessionError,
    add_empty_track,
    create_split_session,
    export_split_multitrack_midi,
    export_split_separate_midi_files,
    generate_piano_roll_preview,
    merge_tracks,
    move_notes_to_track,
)


def _write_split_source_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("track_name", name="Conductor", time=0))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi.tracks.append(tempo_track)

    lead_track = mido.MidiTrack()
    lead_track.append(mido.MetaMessage("track_name", name="Lead", time=0))
    lead_track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
    lead_track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=240))
    lead_track.append(mido.Message("note_on", note=64, velocity=92, channel=0, time=120))
    lead_track.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=240))
    midi.tracks.append(lead_track)

    pad_track = mido.MidiTrack()
    pad_track.append(mido.MetaMessage("track_name", name="Pad", time=0))
    pad_track.append(mido.Message("note_on", note=48, velocity=88, channel=1, time=120))
    pad_track.append(mido.Message("note_off", note=48, velocity=0, channel=1, time=360))
    midi.tracks.append(pad_track)

    midi.save(path)


def _write_long_invalid_key_signature_split_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi.tracks.append(tempo_track)

    synth_track = mido.MidiTrack()
    synth_track.append(mido.MetaMessage("track_name", name="Played With Fire - Deep House (Synth)", time=0))
    synth_track.append(mido.MetaMessage("key_signature", key="C", time=0))

    events = [
        (0, 480, 60, 100),
        (38400, 960, 62, 96),
        (60000, 960, 64, 94),
    ]

    current_tick = 0
    for start_tick, duration_ticks, pitch, velocity in events:
        delta = max(0, int(start_tick) - int(current_tick))
        synth_track.append(
            mido.Message("note_on", note=int(pitch), velocity=int(velocity), channel=0, time=delta)
        )
        synth_track.append(
            mido.Message("note_off", note=int(pitch), velocity=0, channel=0, time=int(duration_ticks))
        )
        current_tick = int(start_tick) + int(duration_ticks)

    midi.tracks.append(synth_track)
    midi.save(path)

    raw = bytearray(path.read_bytes())
    marker = bytes([0xFF, 0x59, 0x02])
    marker_index = raw.find(marker)
    assert marker_index >= 0
    raw[marker_index + 3] = 0x0E
    raw[marker_index + 4] = 0x01
    path.write_bytes(raw)


def _extract_note_spans_by_track(midi_path: Path) -> dict[int, list[tuple[int, int, int, int, int]]]:
    midi = mido.MidiFile(str(midi_path))
    by_track: dict[int, list[tuple[int, int, int, int, int]]] = {}

    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        spans: list[tuple[int, int, int, int, int]] = []

        for message in track:
            absolute_tick += int(message.time)

            is_note_on = message.type == "note_on" and message.velocity > 0
            is_note_off = message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            )

            if is_note_on:
                key = (int(message.channel), int(message.note))
                active.setdefault(key, []).append((absolute_tick, int(message.velocity)))
                continue

            if is_note_off:
                key = (int(message.channel), int(message.note))
                if key in active and active[key]:
                    start_tick, velocity = active[key].pop(0)
                    spans.append(
                        (
                            int(message.note),
                            int(velocity),
                            int(message.channel),
                            int(start_tick),
                            int(absolute_tick),
                        )
                    )

        spans.sort(key=lambda item: (item[3], item[4], item[2], item[0], item[1]))
        by_track[track_index] = spans

    return by_track


def _create_session(tmp_path: Path):
    midi_path = tmp_path / "source.mid"
    _write_split_source_midi(midi_path)
    return create_split_session(midi_path, source="manual", layer="midi")


def test_create_split_session_preserves_underlying_import_error_detail(tmp_path: Path) -> None:
    bad_midi = tmp_path / "bad.mid"
    bad_midi.write_bytes(b"not-a-midi")

    with pytest.raises(MidiSplitSessionError) as exc_info:
        create_split_session(bad_midi, source="manual", layer="midi")

    message = str(exc_info.value)
    assert "Failed to create split session from MIDI 'bad.mid':" in message
    assert "Failed to parse MIDI file 'bad.mid':" in message


def test_create_split_session_preserves_max_end_over_40_seconds_for_invalid_key_signature(
    tmp_path: Path,
) -> None:
    midi_path = tmp_path / "Played_With_Fire_-_Deep_House__Synth___Synth_.mid"
    _write_long_invalid_key_signature_split_midi(midi_path)

    document, _report = import_midi_candidate(midi_path, source="manual", layer="midi")
    session = create_split_session(midi_path, source="manual", layer="midi", display_name=midi_path.name)

    document_max_end_tick = max((int(note.end_tick) for note in document.notes), default=0)
    document_max_end_sec = max((float(note.end_sec) for note in document.notes), default=0.0)
    session_max_end_tick = max((int(note.end_tick) for note in session.notes), default=0)
    session_max_end_sec = max((float(note.end_sec) for note in session.notes), default=0.0)

    assert len(session.notes) == len(document.notes)
    assert session_max_end_tick == document_max_end_tick
    assert session_max_end_sec == pytest.approx(document_max_end_sec, abs=1e-9)
    assert session_max_end_tick > 38400
    assert session_max_end_sec > 40.0
    assert any(float(note.end_sec) >= 40.0 for note in session.notes)


def _extract_editor_script(html: str) -> str:
    match = re.search(r"<script>\s*\(function \(\) \{(?P<script>.*)\}\)\(\);\s*</script>", html, re.DOTALL)
    if match is None:
        raise AssertionError("Could not extract inline editor script")
    return match.group("script")


def test_split_init_imports_notes_and_creates_tracks_from_note_tracks(tmp_path: Path) -> None:
    session = _create_session(tmp_path)

    assert len(session.notes) == 3
    assert len(session.tracks) == 2
    assert [track.source_track_indices for track in session.tracks] == [[1], [2]]
    assert [track.name for track in session.tracks] == ["Lead", "Pad"]
    assert {note.editable_track_index for note in session.notes} == {1, 2}
    assert {note.muted for note in session.notes} == {False}


def test_add_empty_track_creates_new_track_up_to_max_limit(tmp_path: Path) -> None:
    session = _create_session(tmp_path)

    updated = add_empty_track(session, name="Pluck")

    assert len(updated.tracks) == 3
    track = next(track for track in updated.tracks if track.editable_track_index == 3)
    assert track.name == "Pluck"
    assert track.source_track_indices == []


def test_add_empty_track_fails_after_max_track_count(tmp_path: Path) -> None:
    session = _create_session(tmp_path)

    current = session
    for index in range(3, 13):
        current = add_empty_track(current, name=f"Layer {index}")

    with pytest.raises(MidiSplitSessionError, match="Maximum editable track count reached"):
        add_empty_track(current, name="Overflow")


def test_move_notes_to_track_preserves_musical_note_values(tmp_path: Path) -> None:
    session = add_empty_track(_create_session(tmp_path), name="Layer 3")
    target_note = session.notes[0]
    original_dump = target_note.model_dump()

    moved = move_notes_to_track(session, note_ids=[target_note.note_id], target_track_index=3)
    moved_note = next(note for note in moved.notes if note.note_id == target_note.note_id)

    assert moved_note.editable_track_index == 3
    assert moved_note.editable_track_name == "Layer 3"
    assert moved_note.start_tick == original_dump["start_tick"]
    assert moved_note.end_tick == original_dump["end_tick"]
    assert moved_note.duration_ticks == original_dump["duration_ticks"]
    assert moved_note.pitch_midi == original_dump["pitch_midi"]
    assert moved_note.velocity == original_dump["velocity"]
    assert moved_note.channel == original_dump["channel"]


def test_merge_tracks_moves_notes_to_lowest_selected_track(tmp_path: Path) -> None:
    session = add_empty_track(_create_session(tmp_path), name="Layer 3")
    source_note = next(note for note in session.notes if note.editable_track_index == 2)

    moved = move_notes_to_track(session, note_ids=[source_note.note_id], target_track_index=3)
    merged = merge_tracks(moved, editable_track_indices=[2, 3])
    merged_note = next(note for note in merged.notes if note.note_id == source_note.note_id)

    assert merged_note.editable_track_index == 2
    assert all(track.editable_track_index != 3 for track in merged.tracks)


def test_merge_tracks_preserves_note_data(tmp_path: Path) -> None:
    session = add_empty_track(_create_session(tmp_path), name="Layer 3")
    note_to_reassign = next(note for note in session.notes if note.editable_track_index == 2)
    moved = move_notes_to_track(session, note_ids=[note_to_reassign.note_id], target_track_index=3)

    before = {
        note.note_id: (
            note.start_tick,
            note.end_tick,
            note.duration_ticks,
            note.pitch_midi,
            note.velocity,
            note.channel,
        )
        for note in moved.notes
    }

    merged = merge_tracks(moved, editable_track_indices=[2, 3])

    after = {
        note.note_id: (
            note.start_tick,
            note.end_tick,
            note.duration_ticks,
            note.pitch_midi,
            note.velocity,
            note.channel,
        )
        for note in merged.notes
    }

    assert before == after


def test_export_split_multitrack_midi_preserves_note_timing_and_count(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_midi = tmp_path / "split_multitrack.mid"

    export_split_multitrack_midi(session, output_midi)

    assert output_midi.exists()
    midi = mido.MidiFile(str(output_midi))
    assert midi.ticks_per_beat == session.ticks_per_beat
    assert any(message.type == "set_tempo" for message in midi.tracks[0])

    notes_by_track = _extract_note_spans_by_track(output_midi)
    exported_notes = sorted(
        span for track_index, spans in notes_by_track.items() if track_index > 0 for span in spans
    )
    expected_notes = sorted(
        (
            int(note.pitch_midi),
            int(note.velocity),
            int(note.channel) if note.channel is not None else 0,
            int(note.start_tick),
            int(note.end_tick),
        )
        for note in session.notes
    )

    assert len(exported_notes) == len(session.notes)
    assert exported_notes == expected_notes


def test_export_split_separate_midi_files_writes_non_empty_tracks(tmp_path: Path) -> None:
    session = _create_session(tmp_path)

    exported = export_split_separate_midi_files(session, tmp_path / "split_tracks")

    assert len(exported) == 2
    assert all(path.exists() for path in exported)
    assert {path.name[:3] for path in exported} == {"01_", "02_"}


def test_export_split_separate_midi_files_skips_empty_tracks_by_default(tmp_path: Path) -> None:
    session = add_empty_track(_create_session(tmp_path), name="Layer 3")

    exported = export_split_separate_midi_files(session, tmp_path / "split_tracks")

    assert len(exported) == 2
    assert all(not path.name.startswith("03_") for path in exported)


def test_generate_piano_roll_preview_creates_html_with_embedded_note_data(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_preview.html"

    generate_piano_roll_preview(session, output_html)

    assert output_html.exists()
    html = output_html.read_text(encoding="utf-8")
    assert "<canvas" in html
    assert "session-json" in html
    assert '"notes"' in html
    assert '"tracks"' in html
    assert session.notes[0].note_id in html


def test_generate_piano_roll_preview_contains_editor_toolbar_actions(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    html = output_html.read_text(encoding="utf-8")
    assert "Move selected to track" in html
    assert "Add track" in html
    assert "Merge selected tracks" in html
    assert 'id="import-midi-label"' in html
    assert 'for="import-midi-input"' in html
    assert 'id="export-multitrack-btn"' in html
    assert 'id="export-separate-btn"' in html
    assert 'id="undo-btn"' in html
    assert 'id="redo-btn"' in html
    assert 'id="merge-notes-btn"' in html
    assert 'id="delete-notes-btn"' in html
    assert 'id="mute-notes-btn"' in html
    assert 'id="copy-notes-btn"' in html
    assert 'id="paste-notes-btn"' in html
    assert 'id="loop-notes-btn"' in html
    assert 'id="loop-repeats"' in html
    assert 'id="midi-out-enable-btn"' in html
    assert 'id="midi-out-port"' in html
    assert 'id="audition-selected-btn"' in html
    assert 'id="play-region-btn"' in html
    assert 'id="play-all-btn"' in html
    assert 'id="stop-midi-btn"' in html
    assert 'id="panic-midi-btn"' in html
    assert 'id="follow-playhead"' in html
    assert 'id="audition-on-click"' in html
    assert "MIDI Out:" in html
    assert "Play Selected" in html
    assert "Play Region" in html
    assert "Play All" in html
    assert "Follow playhead" in html
    assert "Panic" in html
    assert "Repeats:" in html
    assert 'id="save-session-btn"' not in html
    assert 'id="download-session-btn"' not in html
    assert "Server: checking" in html
    assert "midi-split-editor-build" in html
    assert "Select" in html
    assert "Draw" in html
    assert "Zoom" in html
    assert "Hand" in html
    assert 'id="tool-draw-btn"' in html
    assert 'id="snap-enabled"' in html
    assert 'id="snap-grid"' in html
    assert "Velocity lane" in html
    assert 'id="velocity-lane-visible"' in html
    assert "Velocity values" in html
    assert 'id="velocity-values-visible"' in html


def test_generate_piano_roll_preview_contains_interactive_editor_js_functions(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    html = output_html.read_text(encoding="utf-8")
    script = _extract_editor_script(html)
    assert "function selectNoteById" in html
    assert "function selectNotesInRect" in html
    assert "function moveSelectedToTrack" in html
    assert "function addTrack" in html
    assert "function mergeSelectedTracks" in html
    assert "function mergeSelectedNotes" in html
    assert "function deleteSelectedNotes" in html
    assert "function setMuteStateForSelected" in html
    assert "function toggleMuteSelectedNotes" in html
    assert "function undoHistory" in html
    assert "function redoHistory" in html
    assert "function importMidi" in html
    assert "function exportMultitrackMidi" in html
    assert "function exportSeparateTracks" in html
    assert "let currentTool" in html
    assert "function drawDrawPreview" in html
    assert "function velocityValue(note)" in html
    assert "function velocityRatio(note)" in html
    assert "function clampEditedVelocity(value)" in html
    assert "function velocityFromLaneY(y)" in html
    assert "function velocityGroupKeyForNote(note)" in html
    assert "function sortedNotesForVelocityGroup(notes)" in html
    assert "function buildVelocityGroups()" in html
    assert "function getVelocityGroups()" in html
    assert "function hitTestVelocityGroupHandle(x, y)" in html
    assert "function velocityBarForNote(note)" in html
    assert "function getVelocityBars()" in html
    assert "function velocityBarForNoteId(noteId)" in html
    assert "function drawVelocityLane()" in html
    assert "function drawPlaybackPlayhead(context)" in html
    assert "function hitTestVelocityBar(x, y)" in html
    assert "function hitTestVelocityBarForApi(x, y)" in html
    assert "function setPasteCursorTick(tick, options)" in html
    assert "function getPasteCursorTick()" in html
    assert "function setKeyboardFocusMode(mode, options)" in html
    assert "function getKeyboardFocusMode()" in html
    assert "function viewportTickStart()" in html
    assert "function viewportTickEnd()" in html
    assert "function isTickRangeVisible(startTick, endTick)" in html
    assert "function ensureTickRangeVisible(startTick, endTick)" in html
    assert "function panViewportByTicks(deltaTicks)" in html
    assert "function panViewportBySemitones(deltaRows)" in html
    assert "function normalizedNoteEndTick(note)" in html
    assert "function getSessionMaxTick()" in html
    assert "function getViewportMaxOffsetTicks()" in html
    assert "function getViewportState()" in html
    assert "function getVisibleTickRange()" in html
    assert "function centerViewportOnTick(tick)" in html
    assert "function setFollowPlayhead(enabled)" in html
    assert "function setFollowPlayheadForTest(enabled)" in html
    assert "function getFollowPlayheadForTest()" in html
    assert "function isNoteActiveForPlayback(noteId)" in html
    assert "function setActivePlaybackNotesForTest(noteIds)" in html
    assert "function getPlaybackVisualState()" in html
    assert "function getPlayheadTickForElapsedMs(elapsedMs)" in html
    assert "function followPlaybackTickIfNeeded(tick)" in html
    assert "function stepPlaybackVisualFrame(timestampMs)" in html
    assert "function startPlaybackVisualFrameLoop()" in html
    assert "function startPlaybackVisuals(options)" in html
    assert "function stopPlaybackVisuals(options)" in html
    assert "function clearPlaybackVisualState()" in html
    assert "function setPlaybackVisualStateForTest(nextState)" in html
    assert "function getClipboardSummary()" in html
    assert "function clampMidiChannel(value)" in html
    assert "function clampMidiVelocityOn(value)" in html
    assert "function requestMidiOutAccess()" in html
    assert "function getMidiOutputPorts()" in html
    assert "function hasPlayableNotes()" in html
    assert "function hasPlayableSelectedNotes()" in html
    assert "function hasPlayableRegionOrSelection()" in html
    assert "function hasSelectedMidiOutput()" in html
    assert "function canSendMidiOut()" in html
    assert "function canUseMidiPlayback()" in html
    assert "function isVisualPlaybackAvailableForMode(mode)" in html
    assert "function getPlaybackControlState()" in html
    assert "function updatePlaybackControls()" in html
    assert "function updatePlaybackButtonState()" in html
    assert "function enableMidiOut()" in html
    assert "function selectMidiOutputPort(portId)" in html
    assert "function sendMidiNoteOn(note)" in html
    assert "function sendMidiNoteOff(note)" in html
    assert "function refreshMidiOutPortList()" in html
    assert "function buildPlaybackEventsForNotes(notes, anchorTick)" in html
    assert "function buildPlaybackEventsForRegion()" in html
    assert "function buildPlaybackEventsForAll()" in html
    assert "function playPlaybackEvents(events, statusLabel, options)" in html
    assert "function playSelectedNotes()" in html
    assert "function panicMidiOut()" in html
    assert "function stopMidiPlayback(options)" in html
    assert "function getMidiOutState()" in html
    assert "function setMidiOutEnabledForTest(enabled)" in html
    assert "function setSelectedMidiOutputForTest(outputId)" in html
    assert "function normalizeRepeatCount(value, options)" in html
    assert "function getLoopRepeatCount()" in html
    assert "function setLoopRepeatCount(value, options)" in html
    assert "function getSelectedRegionForLoop()" in html
    assert "function generateUniqueLoopedNoteId()" in html
    assert "function loopSelectedNotes(repeatCount)" in html
    assert "function keyboardHorizontalNudgeTicks(event)" in html
    assert "function keyboardPitchNudgeSemitones(event)" in html
    assert "function viewportHorizontalArrowStep(event)" in html
    assert "function viewportVerticalArrowStep(event)" in html
    assert "function moveSelectedNotesByKeyboard(deltaTicks, deltaPitch, label)" in html
    assert "function adjustSelectedVelocityByKeyboard(delta)" in html
    assert "function handleArrowKey(event)" in html
    assert "function copySelectedNotes()" in html
    assert "function pasteCopiedNotes()" in html
    assert "function setVelocityForNotes(noteIds, options)" in html
    assert "function startVelocityDrag(params)" in html
    assert "function velocityDragPxPerStep(sensitivityMode)" in html
    assert "function sensitivityModeFromEvent(event)" in html
    assert "function applyVelocityDragPreview(drag, point, event)" in html
    assert "function finalizeVelocityDrag(drag)" in html
    assert "function startDrawDrag" in html
    assert "function finalizeDrawDrag" in html
    assert "function applyMovePreview" in html
    assert "function applyResizePreview" in html
    assert "function handleCanvasWheel" in html
    assert "currentTool === \"draw\"" in html
    assert "currentTool === \"zoom\"" in html
    assert "currentTool === \"pan\"" in html
    assert "function checkServerConnection" in html
    assert "function applyImportedSession" in html
    assert "function fitImportedNotesToView" in html
    assert "function renderAll" in html
    assert "function buildTimelineMarkers" in html
    assert "function drawTimelineRuler" in html
    assert "navigator.requestMIDIAccess({ sysex: false })" in script
    assert "default basic app loopback" in script.lower()
    assert "Web MIDI not available. Use Chrome or Edge." in script
    assert "No MIDI outputs found. Enable Default Basic App Loopback or another virtual MIDI output." in script
    assert "sendMidiMessage([0xB0 | channel, 123, 0]);" in script
    assert "sendMidiMessage([0xB0 | channel, 120, 0]);" in script
    assert "window.requestAnimationFrame(stepPlaybackVisualFrame);" in script
    assert "window.cancelAnimationFrame(visuals.animationFrameId);" in script
    assert "drawPlaybackPlayhead(ctx);" in script
    assert "function setErrorStatus" in html
    assert "console.error(\"MIDI split editor error:\", details);" in script


def test_generate_piano_roll_preview_exposes_snap_controls_in_test_api(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert "setSnapEnabled: function (enabled)" in script
    assert "isSnapEnabled: isSnapEnabled" in script
    assert "setSnapDivision: function (division)" in script
    assert "getSnapDivision: currentSnapDivision" in script
    assert "setVelocityLaneVisible: setVelocityLaneVisible" in script
    assert "getVelocityLaneVisible: function () { return state.velocityLaneVisible; }" in script
    assert "selectedVelocitySummary: selectedVelocitySummary" in script
    assert "getVelocityBars: getVelocityBars" in script
    assert "getVelocityGroups: getVelocityGroups" in script
    assert "velocityBarForNoteId: velocityBarForNoteId" in script
    assert "hitTestVelocityBar: hitTestVelocityBarForApi" in script
    assert "setVelocityForNoteIds: function (noteIds, options)" in script
    assert "copySelectedNotes: copySelectedNotes" in script
    assert "pasteCopiedNotes: pasteCopiedNotes" in script
    assert "loopSelectedNotes: loopSelectedNotes" in script
    assert "getClipboardSummary: getClipboardSummary" in script
    assert "getSelectedRegionForLoop: getSelectedRegionForLoop" in script
    assert "getLoopRepeatCount: getLoopRepeatCount" in script
    assert "setLoopRepeatCount: function (value)" in script
    assert "getMidiOutState: getMidiOutState" in script
    assert "setMidiOutEnabledForTest: setMidiOutEnabledForTest" in script
    assert "setSelectedMidiOutputForTest: setSelectedMidiOutputForTest" in script
    assert "getMidiOutputPorts: getMidiOutputPorts" in script
    assert "enableMidiOut: enableMidiOut" in script
    assert "selectMidiOutputPort: selectMidiOutputPort" in script
    assert "sendMidiNoteOn: sendMidiNoteOn" in script
    assert "sendMidiNoteOff: sendMidiNoteOff" in script
    assert "buildPlaybackEventsForNotes: buildPlaybackEventsForNotes" in script
    assert "buildPlaybackEventsForRegion: buildPlaybackEventsForRegion" in script
    assert "buildPlaybackEventsForAll: buildPlaybackEventsForAll" in script
    assert "playPlaybackEvents: playPlaybackEvents" in script
    assert "playSelectedNotes: playSelectedNotes" in script
    assert "playSelectedRegion: playSelectedRegion" in script
    assert "playAllNotes: playAllNotes" in script
    assert "auditionNote: auditionNote" in script
    assert "panicMidiOut: panicMidiOut" in script
    assert "stopMidiPlayback: stopMidiPlayback" in script
    assert "hasPlayableNotes: hasPlayableNotes" in script
    assert "hasPlayableSelectedNotes: hasPlayableSelectedNotes" in script
    assert "hasPlayableRegionOrSelection: hasPlayableRegionOrSelection" in script
    assert "hasSelectedMidiOutput: hasSelectedMidiOutput" in script
    assert "canSendMidiOut: canSendMidiOut" in script
    assert "canUseMidiPlayback: canUseMidiPlayback" in script
    assert "isVisualPlaybackAvailableForMode: isVisualPlaybackAvailableForMode" in script
    assert "getPlaybackControlState: getPlaybackControlState" in script
    assert "updatePlaybackControls: updatePlaybackControls" in script
    assert "updatePlaybackButtonState: updatePlaybackButtonState" in script
    assert "getPlaybackVisualState: getPlaybackVisualState" in script
    assert "setFollowPlayheadForTest: setFollowPlayheadForTest" in script
    assert "getFollowPlayheadForTest: getFollowPlayheadForTest" in script
    assert "setPlaybackVisualStateForTest: setPlaybackVisualStateForTest" in script
    assert "clearPlaybackVisualState: clearPlaybackVisualState" in script
    assert "startPlaybackVisuals: startPlaybackVisuals" in script
    assert "stopPlaybackVisuals: stopPlaybackVisuals" in script
    assert "setActivePlaybackNotesForTest: setActivePlaybackNotesForTest" in script
    assert "isNoteActiveForPlayback: isNoteActiveForPlayback" in script
    assert "getPlayheadTickForElapsedMs: getPlayheadTickForElapsedMs" in script
    assert "setPasteCursorTick: function (tick)" in script
    assert "getPasteCursorTick: getPasteCursorTick" in script
    assert "getSelectedNoteIds: function ()" in script
    assert "setKeyboardFocusMode: setKeyboardFocusMode" in script
    assert "getKeyboardFocusMode: getKeyboardFocusMode" in script
    assert "getSessionMaxTick: getSessionMaxTick" in script
    assert "getViewportMaxOffsetTicks: getViewportMaxOffsetTicks" in script
    assert "panViewportByTicks: function (deltaTicks)" in script
    assert "setXOffsetTicksForTest: function (tick)" in script
    assert "panViewportBySemitones: function (deltaRows)" in script
    assert "centerViewportOnTick: function (tick)" in script
    assert "getVisibleTickRange: getVisibleTickRange" in script
    assert "getViewportState: getViewportState" in script
    assert "getVisibleNoteIds: function ()" in script
    assert "moveSelectedNotesByKeyboard: moveSelectedNotesByKeyboard" in script
    assert "adjustSelectedVelocityByKeyboard: adjustSelectedVelocityByKeyboard" in script


def test_generate_piano_roll_preview_web_midi_playback_builder_behavior_markers(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert "return note && note.muted !== true;" in script
    assert "const resolvedAnchorTick = Number.isFinite(Number(anchorTick))" in script
    assert "const startMs = Math.max(0, Math.round((startSec - anchorSec) * 1000));" in script
    assert "const allNotes = session.notes.filter(function (note) {" in script
    assert "const earliestTick = Math.min.apply(null, allNotes.map(function (note) {" in script
    assert "velocity: clampMidiVelocityOn(velocityValue(note))" in script
    assert "channel: clampMidiChannel(note.channel)" in script
    assert "setStatus(\"No notes selected to play.\", false);" in script
    assert "setStatus(\"No region or notes selected to play.\", false);" in script
    assert "setStatus(\"No notes to play.\", false);" in script
    assert "setStatus(\"Visual playback only. Enable MIDI Out for external sound.\", false);" in script
    assert "setStatus(\"Enable MIDI Out first.\", true);" not in script
    assert "play_all_enabled: playableNotes," in script
    assert "play_selected_enabled: playableSelectedNotes," in script
    assert "play_region_enabled: playableRegionOrSelection," in script
    assert "stop_enabled: playbackRunning," in script
    assert "const shouldSendMidi = canSendMidiOut();" in script
    assert "if (shouldSendMidi) {" in script
    assert "sendMidiNoteOn(event);" in script
    assert "sendMidiNoteOff(event);" in script
    assert "test_sent_message_count:" in script
    assert "setStatus(String(statusLabel || \"Playing notes.\"), false);" in script
    assert "function getPlaybackTickRangeFromEvents(events)" in script
    assert "function getPlaybackEndTickForEvents(events)" in script
    assert "schedulePlaybackTimer(function () {" in script
    assert "state.playbackVisualState.activeNoteIds.add(noteId);" in script
    assert "state.playbackVisualState.activeNoteIds.delete(noteId);" in script
    assert "startPlaybackVisuals({" in script
    assert "setStatus(\"Playback finished.\", false);" in script


def test_generate_piano_roll_preview_web_midi_stop_and_panic_behavior_markers(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert "for (const timerId of state.midiPlaybackTimerIds)" in script
    assert "state.midiPlaybackTimerIds = [];" in script
    assert "state.midiPlaybackRunning = false;" in script
    assert "for (let channel = 0; channel < 16; channel += 1) {" in script
    assert "for (let pitch = 0; pitch < 128; pitch += 1) {" in script
    assert "sendMidiNoteOff({ channel: channel, pitch: pitch });" in script
    assert "sendMidiMessage([0xB0 | channel, 123, 0]);" in script
    assert "sendMidiMessage([0xB0 | channel, 120, 0]);" in script
    assert "state.midiActiveNotes.clear();" in script
    assert "stopPlaybackVisuals({ redraw: opts.redrawVisuals !== false, preserveFollowPlayhead: true });" in script
    assert "stopMidiPlayback({ sendPanic: true });" in script
    assert "setStatus(\"Playback stopped.\", false);" in script
    assert "setStatus(\"MIDI panic sent.\", false);" in script


def test_generate_piano_roll_preview_follow_playhead_visual_markers(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert 'const followPlayheadEl = document.getElementById("follow-playhead");' in script
    assert "followPlayheadEl.addEventListener(\"change\", function () {" in script
    assert "setFollowPlayhead(Boolean(followPlayheadEl.checked));" in script
    assert "if (active && !muted) {" in script
    assert "const deadZone = Math.max(1, Number(range.span_ticks) * 0.14);" in script
    assert "return centerViewportOnTick(tick);" in script
    assert "const topY = TOP_PAD;" in script
    assert "const bottomY = state.velocityLaneVisible ? velocityLaneBottomY() : pianoRollBottomY();" in script
    assert "drawPlaybackPlayhead(ctx);" in script


def test_generate_piano_roll_preview_velocity_bar_geometry_and_hit_zone_logic(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert "const VELOCITY_BAR_DRAW_WIDTH = 6;" in script
    assert "const VELOCITY_BAR_HIT_WIDTH = 10;" in script
    assert "const VELOCITY_GROUP_HANDLE_HEIGHT = 8;" in script
    assert "const VELOCITY_FAN_SPACING = 3;" in script
    assert "const VELOCITY_DRAG_PX_PER_STEP_NORMAL = 4;" in script
    assert "const VELOCITY_DRAG_PX_PER_STEP_FINE = 8;" in script
    assert "const VELOCITY_DRAG_PX_PER_STEP_COARSE = 2;" in script
    assert "const barHeight = Math.max(1, Math.round(ratio * VELOCITY_LANE_HEIGHT));" in script
    assert "const hitInset = Math.max(0, (VELOCITY_BAR_HIT_WIDTH - VELOCITY_BAR_DRAW_WIDTH) / 2);" in script
    assert "if (x >= bar.hitX && x <= bar.hitX + bar.hitW && y >= bar.y && y <= bar.y + bar.h)" in script
    assert "if (group.noteIds.length < 2)" in script


def test_generate_piano_roll_preview_velocity_bar_muted_visual_is_strongly_dimmed(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert "ctx.globalAlpha = muted ? 0.22 : 0.95;" in script
    assert "ctx.fillStyle = \"rgba(0, 0, 0, 0.35)\";" in script
    assert "ctx.strokeStyle = muted ? \"#ffd29a\" : \"#ffffff\";" in script


def test_generate_piano_roll_preview_wires_history_shortcuts_without_editable_interception(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert "function isEditableTypingTarget" in script
    assert "if (isEditableTypingTarget(event.target))" in script
    assert "if (!isModifierDown && String(event.key || \"\").startsWith(\"Arrow\"))" in script
    assert "const handledArrow = handleArrowKey(event);" in script
    assert "if (isModifierDown && key === \"z\" && !event.shiftKey)" in script
    assert "undoHistory();" in script
    assert "if (isModifierDown && (key === \"y\" || (key === \"z\" && event.shiftKey)))" in script
    assert "redoHistory();" in script
    assert "if (isModifierDown && key === \"c\")" in script
    assert "copySelectedNotes();" in script
    assert "if (isModifierDown && key === \"v\")" in script
    assert "pasteCopiedNotes();" in script
    assert "if (key === \"delete\" || key === \"backspace\")" in script
    assert "deleteSelectedNotes();" in script


def test_generate_piano_roll_preview_wires_copy_paste_toolbar_and_paste_cursor(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert "const copyNotesButton = document.getElementById(\"copy-notes-btn\");" in script
    assert "const pasteNotesButton = document.getElementById(\"paste-notes-btn\");" in script
    assert "const loopNotesButton = document.getElementById(\"loop-notes-btn\");" in script
    assert "const loopRepeatsEl = document.getElementById(\"loop-repeats\");" in script
    assert "copyNotesButton.addEventListener(\"click\", copySelectedNotes);" in script
    assert "pasteNotesButton.addEventListener(\"click\", pasteCopiedNotes);" in script
    assert "loopNotesButton.addEventListener(\"click\", function () {" in script
    assert "loopSelectedNotes(getLoopRepeatCount());" in script
    assert "const PASTE_CURSOR_COLOR = \"#8ed1ff\";" in script
    assert "ctx.setLineDash([4, 3]);" in script
    assert "Selection cleared. Paste cursor set to tick " in script
    assert "ensureTickRangeVisible(pastedStartTick, pastedEndTick);" in script


def test_generate_piano_roll_preview_muted_notes_have_distinct_visual_style(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert "const muted = note.muted === true;" in script
    assert "ctx.globalAlpha = muted ? 0.35 : 1.0;" in script
    assert "ctx.setLineDash([4, 2]);" in script
    assert "ctx.fillText(\"M\"" in script


def test_generate_piano_roll_preview_import_flow_wires_file_input_and_resets_value(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert 'const importLabel = document.getElementById("import-midi-label");' in script
    assert 'const importInput = document.getElementById("import-midi-input");' in script
    assert re.search(
        r'importLabel\.addEventListener\("click",\s*function \(event\) \{.*?setStatus\("Import control activated"',
        script,
        re.DOTALL,
    )
    assert "importInput.click()" not in script
    assert re.search(
        r'importInput\.addEventListener\("change",\s*function \(event\) \{.*?const input = event\.target;.*?importMidi\(file\)\.finally\(function \(\) \{\s*if \(input\) \{\s*input\.value = "";',
        script,
        re.DOTALL,
    )
    assert "applyImportedSession(payload);" in script
    assert "setStatus(\"Replacing editor session\", false);" in script
    assert "Imported " in script and " notes from " in script


def test_generate_piano_roll_preview_export_flow_has_blob_download_and_error_guards(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert "async function exportSessionToDownload(options)" in script
    assert "if (!response.ok)" in script
    assert "data = await response.arrayBuffer();" in script
    assert "downloadBlob(data, contentType, fileName);" in script
    assert "URL.revokeObjectURL(href);" in script
    assert "setErrorStatus(operationLabel + \" failed: server unavailable.\", error);" in script
    assert "const requestBody = JSON.stringify(session);" in script


def test_generate_piano_roll_preview_disables_server_actions_in_static_mode(tmp_path: Path) -> None:
    session = _create_session(tmp_path)
    output_html = tmp_path / "split_editor.html"

    generate_piano_roll_preview(session, output_html)

    script = _extract_editor_script(output_html.read_text(encoding="utf-8"))
    assert "function setServerActionControlsEnabled(enabled, disabledReason)" in script
    assert "fetch(\"/api/session\", {" in script
    assert "cache: \"no-store\"" in script
    assert "Server connection unavailable. Start the editor with: midi-cleaner midi split-editor" in script
