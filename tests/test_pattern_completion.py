from __future__ import annotations

import json
from pathlib import Path

import mido

from midi_cleaner.pattern import (
    PatternCompletionParameters,
    complete_pattern_blocks,
)


def _write_working_midi(path: Path, *, ambiguous: bool = False, with_overlap_case: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

    # Pattern A (4-note block): C2 A1 D2 G1 (0.5 beat each)
    block_a = [36, 33, 38, 31]

    # Block 1: complete
    for pitch in block_a:
        track.append(mido.Message("note_on", note=pitch, velocity=96, time=0, channel=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=240, channel=0))

    # Gap between blocks
    track.append(mido.MetaMessage("text", text="gap", time=480))

    # Block 2: complete repeated
    for pitch in block_a:
        track.append(mido.Message("note_on", note=pitch, velocity=92, time=0, channel=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=240, channel=0))

    # Gap
    track.append(mido.MetaMessage("text", text="gap", time=480))

    # Block 3: incomplete prefix (first two notes only)
    incomplete_notes = block_a[:2]
    if with_overlap_case:
        # Insert a base note where the completion would be placed to force rejection.
        track.append(mido.Message("note_on", note=38, velocity=84, time=0, channel=0))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240, channel=0))

    for pitch in incomplete_notes:
        track.append(mido.Message("note_on", note=pitch, velocity=90, time=0, channel=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=240, channel=0))

    if ambiguous:
        # Add competing family B to trigger ambiguity.
        track.append(mido.MetaMessage("text", text="gap", time=960))
        family_b = [36, 33, 34, 31]
        for pitch in family_b:
            track.append(mido.Message("note_on", note=pitch, velocity=88, time=0, channel=0))
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=240, channel=0))

    midi.save(path)


def _build_project(tmp_path: Path, *, ambiguous: bool = False, with_overlap_case: bool = False) -> Path:
    project_dir = tmp_path / "project_pattern"
    working_midi_path = project_dir / "midi" / "working" / "working.mid"
    _write_working_midi(working_midi_path, ambiguous=ambiguous, with_overlap_case=with_overlap_case)
    return project_dir


def _read_midi_note_count(path: Path) -> int:
    midi = mido.MidiFile(str(path))
    count = 0
    for track in midi.tracks:
        for msg in track:
            if msg.type == "note_on" and int(msg.velocity) > 0:
                count += 1
    return count


def test_pattern_completion_splits_blocks_and_groups_families(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path)

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.pattern_block_count >= 3
    assert report.pattern_family_count >= 1

    blocks = json.loads(Path(report.pattern_blocks_file).read_text(encoding="utf-8"))
    families = json.loads(Path(report.pattern_families_file).read_text(encoding="utf-8"))

    assert all("block_id" in item for item in blocks)
    assert any(item.get("status") == "incomplete" for item in blocks)
    assert all("pattern_family_id" in item for item in families)


def test_pattern_completion_detects_incomplete_block_and_matches_family(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path)

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    incomplete = json.loads(Path(report.incomplete_blocks_file).read_text(encoding="utf-8"))
    assert incomplete

    first = incomplete[0]
    assert first["incomplete_block_id".lower() if False else "incomplete_block_id"]
    assert first["possible_matches"]
    assert first["best_match_pattern_family_id"] is not None


def test_pattern_completion_inserts_missing_tail_notes_to_uzupelnienie_only(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path)
    working_path = project_dir / "midi" / "working" / "working.mid"
    before_bytes = working_path.read_bytes()

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.inserted_note_count >= 1

    output_midi = Path(report.output_midi_path)
    assert output_midi.exists()
    assert output_midi.name == "uzupelnienie.mid"

    inserted_note_count = _read_midi_note_count(output_midi)
    assert inserted_note_count == report.inserted_note_count

    # Base MIDI remains unchanged.
    assert working_path.read_bytes() == before_bytes


def test_pattern_completion_skips_ambiguous_incomplete_block(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, ambiguous=True)

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    incomplete = json.loads(Path(report.incomplete_blocks_file).read_text(encoding="utf-8"))
    assert incomplete
    assert any(item.get("action") == "skipped" for item in incomplete)
    assert any("Ambiguous" in item.get("match_reason", "") for item in incomplete)


def test_pattern_completion_rejects_overlapping_inserted_notes(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, with_overlap_case=True)

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    incomplete = json.loads(Path(report.incomplete_blocks_file).read_text(encoding="utf-8"))
    if incomplete:
        assert any(
            item.get("action") == "skipped"
            or len(item.get("missing_notes_to_insert", [])) == 0
            for item in incomplete
        )

    # Export remains valid and contains only accepted inserted notes.
    output_midi = Path(report.output_midi_path)
    assert output_midi.exists()
    assert _read_midi_note_count(output_midi) == report.inserted_note_count


def test_pattern_completion_is_deterministic_and_does_not_invoke_ai(tmp_path: Path, monkeypatch) -> None:
    project_dir = _build_project(tmp_path)

    called = {"value": False}

    def _fake_ai(*args, **kwargs):
        _ = (args, kwargs)
        called["value"] = True
        raise AssertionError("AI should not be called in pattern completion workflow")

    monkeypatch.setattr("midi_cleaner.ai_completion.service.complete_ai_pattern_completion", _fake_ai)

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))
    assert report.status == "ok"
    assert called["value"] is False
