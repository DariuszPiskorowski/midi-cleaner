from __future__ import annotations

import io
from pathlib import Path

import mido
import pytest

from midi_cleaner.midi.models import TempoEvent
from midi_cleaner.midi_split.editor_logic import NoteEditTransaction
from midi_cleaner.midi_split.editor_logic import NoteHistory
from midi_cleaner.midi_split.editor_logic import apply_note_transaction
from midi_cleaner.midi_split.editor_logic import build_timeline_layout
from midi_cleaner.midi_split.editor_logic import clone_note_payload
from midi_cleaner.midi_split.editor_logic import clamp_velocity
from midi_cleaner.midi_split.editor_logic import delete_selected_notes
from midi_cleaner.midi_split.editor_logic import draw_note
from midi_cleaner.midi_split.editor_logic import move_selected_notes
from midi_cleaner.midi_split.editor_logic import resize_note_edge
from midi_cleaner.midi_split.editor_logic import resolve_selected_mute_action
from midi_cleaner.midi_split.editor_logic import merge_selected_notes
from midi_cleaner.midi_split.editor_logic import selected_velocity_summary
from midi_cleaner.midi_split.editor_logic import set_selected_notes_muted
from midi_cleaner.midi_split.editor_logic import tick_to_bar_beat
from midi_cleaner.midi_split.editor_logic import tick_to_seconds
from midi_cleaner.midi_split.exporter import export_split_separate_midi_files
from midi_cleaner.midi_split.exporter import export_split_multitrack_midi
from midi_cleaner.midi_split.models import MidiSplitSession
from midi_cleaner.midi_split.models import SplitNote
from midi_cleaner.midi_split.models import SplitTrack


def _note(
    note_id: str,
    start_tick: int,
    end_tick: int,
    *,
    pitch: int = 40,
    velocity: int = 96,
    channel: int | None = 0,
    track_index: int = 1,
    track_name: str = "Bass",
    muted: bool = False,
) -> dict[str, object]:
    duration_ticks = max(0, end_tick - start_tick)
    return {
        "note_id": note_id,
        "source_track_index": 1,
        "source_track_name": "Source",
        "editable_track_index": track_index,
        "editable_track_name": track_name,
        "channel": channel,
        "pitch_midi": pitch,
        "pitch_name": "E2",
        "velocity": velocity,
        "start_tick": start_tick,
        "end_tick": end_tick,
        "duration_ticks": duration_ticks,
        "start_sec": start_tick / 960.0,
        "end_sec": end_tick / 960.0,
        "duration_sec": duration_ticks / 960.0,
        "muted": muted,
        "metadata": {"tag": "x"},
    }


def _note_by_id(notes: list[dict[str, object]], note_id: str) -> dict[str, object]:
    for note in notes:
        if str(note["note_id"]) == note_id:
            return note
    raise AssertionError(f"Note not found: {note_id}")


def _tempo_map() -> list[dict[str, int]]:
    return [{"tick": 0, "tempo_us_per_beat": 500000}]


def _session_from_notes(notes: list[dict[str, object]]) -> MidiSplitSession:
    track_indices = sorted({int(note["editable_track_index"]) for note in notes})
    tracks = [
        SplitTrack(editable_track_index=index, name=f"Track {index:02d}", source_track_indices=[1])
        for index in track_indices
    ]
    split_notes = [SplitNote.model_validate(note) for note in notes]
    return MidiSplitSession(
        schema_version="0.1.0",
        source_midi="test.mid",
        source="manual",
        layer="midi",
        ticks_per_beat=480,
        tempo_map=[TempoEvent(tick=0, tempo_us_per_beat=500000, sec=0.0)],
        tracks=tracks,
        notes=split_notes,
    )


def _collect_exported_spans(midi_path: Path) -> list[tuple[int, int, int, int, int, int]]:
    midi = mido.MidiFile(str(midi_path))
    spans: list[tuple[int, int, int, int, int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
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
                            track_index,
                            int(message.note),
                            velocity,
                            int(message.channel),
                            start_tick,
                            absolute_tick,
                        )
                    )
    spans.sort(key=lambda item: (item[0], item[4], item[5], item[3], item[1], item[2]))
    return spans


def test_merge_two_adjacent_notes_merges_to_one_contiguous_note() -> None:
    notes = [_note("n1", 100, 140), _note("n2", 140, 190)]

    result = merge_selected_notes(
        all_notes=notes,
        selected_note_ids=["n1", "n2"],
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
    )

    merged = result["merged_note"]
    assert merged["start_tick"] == 100
    assert merged["end_tick"] == 190
    assert merged["duration_ticks"] == 90
    assert len(result["notes_after"]) == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1, 0),
        (0, 0),
        (1, 1),
        (64, 64),
        (127, 127),
        (200, 127),
    ],
)
def test_clamp_velocity_handles_boundaries(value: int, expected: int) -> None:
    assert clamp_velocity(value) == expected


def test_selected_velocity_summary_returns_min_avg_max_for_selection() -> None:
    notes = [
        _note("n1", 0, 120, velocity=48),
        _note("n2", 120, 240, velocity=82),
        _note("n3", 240, 360, velocity=117),
    ]

    summary = selected_velocity_summary(notes)

    assert summary == {"count": 3, "min": 48, "avg": 82, "max": 117}


def test_selected_velocity_summary_is_safe_for_empty_selection() -> None:
    summary = selected_velocity_summary([])

    assert summary == {"count": 0, "min": None, "avg": None, "max": None}


def test_merge_notes_with_gaps_fills_full_span() -> None:
    notes = [_note("n1", 100, 140), _note("n2", 150, 190), _note("n3", 200, 250)]

    result = merge_selected_notes(
        all_notes=notes,
        selected_note_ids=["n1", "n2", "n3"],
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
    )

    merged = result["merged_note"]
    assert merged["start_tick"] == 100
    assert merged["end_tick"] == 250


def test_merge_overlapping_notes_uses_earliest_start_and_latest_end() -> None:
    notes = [_note("n1", 100, 170), _note("n2", 140, 210), _note("n3", 205, 240)]

    result = merge_selected_notes(
        all_notes=notes,
        selected_note_ids=["n1", "n2", "n3"],
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
    )

    merged = result["merged_note"]
    assert merged["start_tick"] == 100
    assert merged["end_tick"] == 240


def test_merge_preserves_first_note_velocity_pitch_track_and_channel() -> None:
    notes = [
        _note("n1", 100, 130, pitch=48, velocity=111, channel=2, track_index=3, track_name="T3"),
        _note("n2", 140, 200, pitch=48, velocity=65, channel=2, track_index=3, track_name="T3"),
    ]

    merged = merge_selected_notes(
        all_notes=notes,
        selected_note_ids=["n1", "n2"],
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
    )["merged_note"]

    assert merged["velocity"] == 111
    assert merged["pitch_midi"] == 48
    assert merged["editable_track_index"] == 3
    assert merged["editable_track_name"] == "T3"
    assert merged["channel"] == 2


def test_merge_removes_original_selected_notes_and_keeps_unrelated_notes() -> None:
    notes = [_note("n1", 100, 140), _note("n2", 150, 190), _note("keep", 400, 460, pitch=50)]

    result = merge_selected_notes(
        all_notes=notes,
        selected_note_ids=["n1", "n2"],
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
    )

    ids = {str(note["note_id"]) for note in result["notes_after"]}
    assert "n1" not in ids
    assert "n2" not in ids
    assert "keep" in ids


def test_merge_generates_unique_note_id() -> None:
    notes = [_note("n1", 100, 140), _note("n2", 150, 190), _note("n1_merged", 500, 600)]

    merged = merge_selected_notes(
        all_notes=notes,
        selected_note_ids=["n1", "n2"],
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
    )["merged_note"]

    assert merged["note_id"] not in {"n1", "n2", "n1_merged"}


def test_move_selected_notes_applies_tick_and_pitch_delta() -> None:
    notes = [
        _note("n1", 100, 160, pitch=40),
        _note("n2", 240, 300, pitch=43),
        _note("keep", 400, 480, pitch=55),
    ]

    result = move_selected_notes(
        all_notes=notes,
        selected_note_ids=["n1", "n2"],
        delta_ticks=120,
        delta_pitch=3,
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
    )

    assert result["moved_count"] == 2
    assert result["applied_delta_ticks"] == 120
    assert result["applied_delta_pitch"] == 3

    by_id = {str(note["note_id"]): note for note in result["notes_after"]}
    assert by_id["n1"]["start_tick"] == 220
    assert by_id["n1"]["end_tick"] == 280
    assert by_id["n1"]["duration_ticks"] == 60
    assert by_id["n1"]["pitch_midi"] == 43
    assert by_id["n2"]["start_tick"] == 360
    assert by_id["n2"]["end_tick"] == 420
    assert by_id["n2"]["duration_ticks"] == 60
    assert by_id["n2"]["pitch_midi"] == 46
    assert by_id["keep"]["start_tick"] == 400
    assert by_id["keep"]["pitch_midi"] == 55


def test_move_selected_notes_clamps_tick_and_pitch_deltas() -> None:
    notes = [
        _note("n1", 10, 70, pitch=1),
        _note("n2", 100, 160, pitch=126),
    ]

    result = move_selected_notes(
        all_notes=notes,
        selected_note_ids=["n1", "n2"],
        delta_ticks=-50,
        delta_pitch=10,
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
    )

    assert result["applied_delta_ticks"] == -10
    assert result["applied_delta_pitch"] == 1

    by_id = {str(note["note_id"]): note for note in result["notes_after"]}
    assert by_id["n1"]["start_tick"] == 0
    assert by_id["n1"]["end_tick"] == 60
    assert by_id["n1"]["pitch_midi"] == 2
    assert by_id["n2"]["start_tick"] == 90
    assert by_id["n2"]["end_tick"] == 150
    assert by_id["n2"]["pitch_midi"] == 127


def test_resize_note_edge_respects_bounds_and_min_duration() -> None:
    notes = [_note("n1", 100, 200, pitch=40)]

    left = resize_note_edge(
        all_notes=notes,
        note_id="n1",
        edge="left",
        target_tick=190,
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
        min_duration_ticks=30,
    )
    assert left["after_note"]["start_tick"] == 170
    assert left["after_note"]["end_tick"] == 200
    assert left["after_note"]["duration_ticks"] == 30

    right = resize_note_edge(
        all_notes=notes,
        note_id="n1",
        edge="right",
        target_tick=110,
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
        min_duration_ticks=40,
    )
    assert right["after_note"]["start_tick"] == 100
    assert right["after_note"]["end_tick"] == 140
    assert right["after_note"]["duration_ticks"] == 40


def test_draw_note_generates_unique_id_and_normalizes_payload() -> None:
    notes = [_note("drawn_000001", 100, 180, pitch=40, track_index=2, track_name="Track 2")]

    result = draw_note(
        all_notes=notes,
        start_tick=480,
        end_tick=480,
        pitch_midi=130,
        editable_track_index=2,
        editable_track_name="Track 2",
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
        min_duration_ticks=30,
        velocity=200,
        metadata={"origin": "draw"},
    )

    drawn = result["drawn_note"]
    assert drawn["note_id"] == "drawn_000002"
    assert drawn["start_tick"] == 480
    assert drawn["end_tick"] == 510
    assert drawn["duration_ticks"] == 30
    assert drawn["pitch_midi"] == 127
    assert drawn["velocity"] == 127
    assert drawn["editable_track_index"] == 2
    assert drawn["source_track_index"] == 2
    assert drawn["metadata"] == {"origin": "draw"}
    assert result["selection_after"] == ["drawn_000002"]
    assert len(result["notes_after"]) == 2


def test_delete_selected_notes_returns_remaining_and_deleted() -> None:
    notes = [
        _note("n1", 100, 140),
        _note("n2", 150, 190),
        _note("n3", 200, 250),
    ]

    result = delete_selected_notes(all_notes=notes, selected_note_ids=["n2", "n3", "missing"])

    assert [str(note["note_id"]) for note in result["notes_after"]] == ["n1"]
    assert [str(note["note_id"]) for note in result["deleted_notes"]] == ["n2", "n3"]


def test_delete_selected_notes_with_empty_selection_leaves_all_notes() -> None:
    notes = [
        _note("n1", 100, 140),
        _note("n2", 150, 190),
    ]

    result = delete_selected_notes(all_notes=notes, selected_note_ids=[])

    assert [str(note["note_id"]) for note in result["notes_after"]] == ["n1", "n2"]
    assert result["deleted_notes"] == []


def test_resolve_selected_mute_action_prefers_mute_for_empty_mixed_or_unmuted() -> None:
    assert resolve_selected_mute_action([]) is None
    assert resolve_selected_mute_action([_note("n1", 0, 120, muted=False)]) == "mute"
    assert (
        resolve_selected_mute_action(
            [
                _note("n1", 0, 120, muted=True),
                _note("n2", 140, 260, muted=False),
            ]
        )
        == "mute"
    )
    assert (
        resolve_selected_mute_action(
            [
                _note("n1", 0, 120, muted=True),
                _note("n2", 140, 260, muted=True),
            ]
        )
        == "unmute"
    )


def test_set_selected_notes_muted_updates_only_selected_notes() -> None:
    notes = [
        _note("n1", 100, 140, muted=False),
        _note("n2", 150, 190, muted=False),
        _note("n3", 200, 250, muted=True),
    ]

    result = set_selected_notes_muted(all_notes=notes, selected_note_ids=["n1", "n2"], mute=True)
    by_id = {str(note["note_id"]): note for note in result["notes_after"]}

    assert by_id["n1"]["muted"] is True
    assert by_id["n2"]["muted"] is True
    assert by_id["n3"]["muted"] is True
    assert result["action"] == "mute"
    assert result["changed_count"] == 2


@pytest.mark.parametrize(
    ("selected_ids", "expected_message"),
    [
        (["n1"], "Select at least two notes to merge."),
        (["n1", "n_pitch"], "Selected notes must have the same pitch."),
        (["n1", "n_track"], "Selected notes must belong to the same track."),
        (["n1", "n_channel"], "Selected notes must use the same MIDI channel."),
    ],
)
def test_merge_validation_rejects_invalid_selection(
    selected_ids: list[str], expected_message: str
) -> None:
    notes = [
        _note("n1", 100, 140, pitch=40, track_index=1, channel=0),
        _note("n_pitch", 150, 190, pitch=41, track_index=1, channel=0),
        _note("n_track", 150, 190, pitch=40, track_index=2, channel=0),
        _note("n_channel", 150, 190, pitch=40, track_index=1, channel=1),
    ]

    with pytest.raises(ValueError, match=expected_message):
        merge_selected_notes(
            all_notes=notes,
            selected_note_ids=selected_ids,
            ticks_per_beat=480,
            tempo_events=_tempo_map(),
        )


def test_undo_reverses_track_assignment_and_redo_reapplies() -> None:
    notes = [_note("n1", 100, 160, track_index=1), _note("n2", 200, 280, track_index=1)]

    before = clone_note_payload(_note_by_id(notes, "n1"))
    after = clone_note_payload(before)
    after["editable_track_index"] = 2
    after["editable_track_name"] = "Track 02"

    transaction = NoteEditTransaction(
        label="move-selected-to-track",
        before_notes=[before],
        after_notes=[after],
        selection_before=["n1"],
        selection_after=["n1"],
    )

    history = NoteHistory(limit=100)
    history.push(transaction)

    moved_notes = apply_note_transaction(notes, transaction, use_after=True)
    assert _note_by_id(moved_notes, "n1")["editable_track_index"] == 2

    undone_notes, undone_selection = history.undo(moved_notes)
    assert _note_by_id(undone_notes, "n1")["editable_track_index"] == 1
    assert undone_selection == ["n1"]

    redone_notes, redone_selection = history.redo(undone_notes)
    assert _note_by_id(redone_notes, "n1")["editable_track_index"] == 2
    assert redone_selection == ["n1"]


def test_undo_restores_original_notes_after_merge_and_redo_recreates_merged_note() -> None:
    notes = [_note("n1", 100, 140), _note("n2", 150, 190)]
    merge_result = merge_selected_notes(
        all_notes=notes,
        selected_note_ids=["n1", "n2"],
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
    )

    transaction = NoteEditTransaction(
        label="merge-notes",
        before_notes=merge_result["selected_before"],
        after_notes=[merge_result["merged_note"]],
        selection_before=["n1", "n2"],
        selection_after=[str(merge_result["merged_note"]["note_id"])],
    )

    history = NoteHistory(limit=100)
    history.push(transaction)

    merged_notes = apply_note_transaction(notes, transaction, use_after=True)
    merged_ids = {str(note["note_id"]) for note in merged_notes}
    assert "n1" not in merged_ids and "n2" not in merged_ids

    undone_notes, _selection = history.undo(merged_notes)
    undone_ids = {str(note["note_id"]) for note in undone_notes}
    assert "n1" in undone_ids and "n2" in undone_ids

    redone_notes, _selection = history.redo(undone_notes)
    redone_ids = {str(note["note_id"]) for note in redone_notes}
    assert "n1" not in redone_ids and "n2" not in redone_ids


def test_new_edit_clears_redo_stack() -> None:
    notes = [_note("n1", 100, 140)]
    tx1 = NoteEditTransaction(
        label="first",
        before_notes=[clone_note_payload(notes[0])],
        after_notes=[{**clone_note_payload(notes[0]), "editable_track_index": 2}],
    )
    tx2 = NoteEditTransaction(
        label="second",
        before_notes=[clone_note_payload(notes[0])],
        after_notes=[{**clone_note_payload(notes[0]), "editable_track_index": 3}],
    )

    history = NoteHistory(limit=100)
    history.push(tx1)
    moved = apply_note_transaction(notes, tx1, use_after=True)
    _undone, _selection = history.undo(moved)
    assert history.can_redo()

    history.push(tx2)
    assert history.can_undo()
    assert not history.can_redo()


def test_import_like_reset_clears_history() -> None:
    notes = [_note("n1", 100, 140)]
    tx = NoteEditTransaction(before_notes=[clone_note_payload(notes[0])], after_notes=[clone_note_payload(notes[0])])

    history = NoteHistory(limit=100)
    history.push(tx)
    history.clear()

    assert not history.can_undo()
    assert not history.can_redo()


def test_history_limit_is_enforced_to_100_actions() -> None:
    history = NoteHistory(limit=100)
    for index in range(120):
        payload = _note(f"n{index}", 100, 140)
        history.push(
            NoteEditTransaction(
                label=f"tx-{index}",
                before_notes=[clone_note_payload(payload)],
                after_notes=[clone_note_payload(payload)],
            )
        )

    assert len(history.undo_stack) == 100
    assert history.undo_stack[0].label == "tx-20"
    assert history.undo_stack[-1].label == "tx-119"


def test_export_reflects_current_undo_redo_state(tmp_path: Path) -> None:
    notes = [_note("n1", 100, 140), _note("n2", 150, 190)]
    merge_result = merge_selected_notes(
        all_notes=notes,
        selected_note_ids=["n1", "n2"],
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
    )

    transaction = NoteEditTransaction(
        label="merge-notes",
        before_notes=merge_result["selected_before"],
        after_notes=[merge_result["merged_note"]],
        selection_before=["n1", "n2"],
        selection_after=[str(merge_result["merged_note"]["note_id"])],
    )

    history = NoteHistory(limit=100)
    history.push(transaction)

    merged_notes = apply_note_transaction(notes, transaction, use_after=True)
    merged_session = _session_from_notes(merged_notes)
    merged_path = tmp_path / "merged.mid"
    export_split_multitrack_midi(merged_session, merged_path)

    merged_spans = [span for span in _collect_exported_spans(merged_path) if span[0] > 0]
    assert len(merged_spans) == 1
    assert merged_spans[0][4] == 100
    assert merged_spans[0][5] == 190

    undone_notes, _selection = history.undo(merged_notes)
    undone_session = _session_from_notes(undone_notes)
    undone_path = tmp_path / "undone.mid"
    export_split_multitrack_midi(undone_session, undone_path)
    undone_spans = [span for span in _collect_exported_spans(undone_path) if span[0] > 0]
    assert len(undone_spans) == 2


def test_export_skips_muted_notes_in_multitrack_and_separate_exports(tmp_path: Path) -> None:
    notes = [
        _note("audible", 100, 180, pitch=40, velocity=100, track_index=1, muted=False),
        _note("muted", 220, 300, pitch=43, velocity=90, track_index=1, muted=True),
    ]
    session = _session_from_notes(notes)

    multitrack_path = tmp_path / "multitrack.mid"
    export_split_multitrack_midi(session, multitrack_path)
    multitrack_spans = [span for span in _collect_exported_spans(multitrack_path) if span[0] > 0]
    assert len(multitrack_spans) == 1
    assert multitrack_spans[0][1] == 40
    assert multitrack_spans[0][4] == 100
    assert multitrack_spans[0][5] == 180

    separate_dir = tmp_path / "separate"
    exported = export_split_separate_midi_files(session, separate_dir, skip_empty=False)
    assert len(exported) == 1
    separate_spans = _collect_exported_spans(exported[0])
    assert len(separate_spans) == 1
    assert separate_spans[0][1] == 40
    assert separate_spans[0][4] == 100
    assert separate_spans[0][5] == 180


def test_tick_to_bar_conversion_for_4_4_boundaries() -> None:
    assert tick_to_bar_beat(0, ticks_per_beat=480, time_signatures=[])["bar"] == 1
    assert tick_to_bar_beat(480, ticks_per_beat=480, time_signatures=[])["beat"] == 2
    assert tick_to_bar_beat(1920, ticks_per_beat=480, time_signatures=[])["bar"] == 2
    assert tick_to_bar_beat(1920, ticks_per_beat=480, time_signatures=[])["beat"] == 1


def test_tick_to_bar_conversion_supports_non_4_4() -> None:
    three_four = [{"tick": 0, "numerator": 3, "denominator": 4}]
    assert tick_to_bar_beat(0, ticks_per_beat=480, time_signatures=three_four)["bar"] == 1
    assert tick_to_bar_beat(480, ticks_per_beat=480, time_signatures=three_four)["beat"] == 2
    assert tick_to_bar_beat(1440, ticks_per_beat=480, time_signatures=three_four)["bar"] == 2


def test_tick_to_time_conversion_handles_tempo_changes() -> None:
    tempo_map = [
        {"tick": 0, "tempo_us_per_beat": 500000},
        {"tick": 480, "tempo_us_per_beat": 1000000},
    ]

    assert tick_to_seconds(480, tempo_map, ticks_per_beat=480) == pytest.approx(0.5, abs=1e-6)
    assert tick_to_seconds(960, tempo_map, ticks_per_beat=480) == pytest.approx(1.5, abs=1e-6)


def test_ruler_layout_aligns_ticks_with_viewport_mapping() -> None:
    layout = build_timeline_layout(
        x_offset_ticks=240,
        pixels_per_tick=0.5,
        viewport_width_px=900,
        left_pad_px=72,
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
        time_signatures=[],
    )

    assert layout.bar_lines
    marker = layout.bar_lines[0]
    expected_x = 72 + (marker.tick - 240) * 0.5
    assert marker.x == pytest.approx(expected_x, abs=1e-6)


def test_ruler_label_density_prevents_overlapping_labels() -> None:
    layout = build_timeline_layout(
        x_offset_ticks=0,
        pixels_per_tick=0.04,
        viewport_width_px=1200,
        left_pad_px=72,
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
        time_signatures=[],
    )

    bar_label_x = [label.x for label in layout.bar_labels]
    time_label_x = [label.x for label in layout.time_labels]

    assert all((b - a) >= 56 - 1e-6 for a, b in zip(bar_label_x, bar_label_x[1:]))
    assert all((b - a) >= 72 - 1e-6 for a, b in zip(time_label_x, time_label_x[1:]))


def test_pan_and_zoom_update_ruler_positions() -> None:
    layout_base = build_timeline_layout(
        x_offset_ticks=0,
        pixels_per_tick=0.2,
        viewport_width_px=1000,
        left_pad_px=72,
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
        time_signatures=[],
    )
    base_x_by_tick = {int(marker.tick): marker.x for marker in layout_base.bar_lines}
    assert 1920 in base_x_by_tick

    layout_pan = build_timeline_layout(
        x_offset_ticks=480,
        pixels_per_tick=0.2,
        viewport_width_px=1000,
        left_pad_px=72,
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
        time_signatures=[],
    )
    pan_x_by_tick = {int(marker.tick): marker.x for marker in layout_pan.bar_lines}
    assert 1920 in pan_x_by_tick

    layout_zoom = build_timeline_layout(
        x_offset_ticks=0,
        pixels_per_tick=0.4,
        viewport_width_px=1000,
        left_pad_px=72,
        ticks_per_beat=480,
        tempo_events=_tempo_map(),
        time_signatures=[],
    )
    zoom_x_by_tick = {int(marker.tick): marker.x for marker in layout_zoom.bar_lines}
    assert 1920 in zoom_x_by_tick

    assert pan_x_by_tick[1920] == pytest.approx(base_x_by_tick[1920] - 96.0, abs=1e-6)
    assert zoom_x_by_tick[1920] == pytest.approx(72 + (1920 * 0.4), abs=1e-6)
    assert zoom_x_by_tick[1920] > base_x_by_tick[1920]
