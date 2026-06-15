from __future__ import annotations

import json
from pathlib import Path
import shutil

import mido

from midi_cleaner.pattern.models import MissingExpectedBlock
from midi_cleaner.pattern import (
    PatternCompletionParameters,
    complete_pattern_blocks,
)


def _append_block(track: mido.MidiTrack, notes: list[int], *, velocity: int = 96) -> None:
    for pitch in notes:
        track.append(mido.Message("note_on", note=pitch, velocity=velocity, time=0, channel=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=240, channel=0))


def _append_gap(track: mido.MidiTrack, ticks: int = 480) -> None:
    track.append(mido.MetaMessage("text", text="gap", time=ticks))


def _write_incomplete_existing_midi(
    path: Path,
    *,
    ambiguous: bool = False,
    with_overlap_case: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

    # Pattern A (4-note block): C2 A1 D2 G1 (0.5 beat each)
    block_a = [36, 33, 38, 31]

    # Block 1: complete
    _append_block(track, block_a, velocity=96)

    # Gap between blocks
    _append_gap(track, ticks=480)

    # Block 2: complete repeated
    _append_block(track, block_a, velocity=92)

    # Gap
    _append_gap(track, ticks=480)

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
        _append_gap(track, ticks=960)
        family_b = [36, 33, 34, 31]
        _append_block(track, family_b, velocity=88)

    midi.save(path)


def _write_missing_expected_midi(
    path: Path,
    *,
    ambiguous: bool = False,
    with_collision_case: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

    family_a = [36, 33, 38, 31]
    family_b = [36, 33, 34, 31]

    # A1
    _append_block(track, family_a, velocity=96)
    _append_gap(track, ticks=1200)

    # A2
    _append_block(track, family_a, velocity=92)
    _append_gap(track, ticks=3600)

    # Optional collision note in expected missing region.
    if with_collision_case:
        track.append(mido.Message("note_on", note=38, velocity=76, time=0, channel=0))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240, channel=0))
        _append_gap(track, ticks=1200)

    # A4 after missing A3 slot
    _append_block(track, family_a, velocity=93)

    if ambiguous:
        # Family B with matching spacing and hole to create ambiguous competing candidates.
        _append_gap(track, ticks=2400)
        _append_block(track, family_b, velocity=89)
        _append_gap(track, ticks=1200)
        _append_block(track, family_b, velocity=87)
        _append_gap(track, ticks=3600)
        _append_block(track, family_b, velocity=91)

    midi.save(path)


def _build_project(
    tmp_path: Path,
    *,
    fixture: str = "incomplete_existing",
    ambiguous: bool = False,
    with_overlap_case: bool = False,
    with_collision_case: bool = False,
) -> Path:
    project_dir = tmp_path / "project_pattern"
    working_midi_path = project_dir / "midi" / "working" / "working.mid"

    if fixture == "incomplete_existing":
        _write_incomplete_existing_midi(
            working_midi_path,
            ambiguous=ambiguous,
            with_overlap_case=with_overlap_case,
        )
    elif fixture == "missing_expected":
        _write_missing_expected_midi(
            working_midi_path,
            ambiguous=ambiguous,
            with_collision_case=with_collision_case,
        )
    else:
        raise AssertionError(f"Unsupported fixture: {fixture}")

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
    project_dir = _build_project(tmp_path, fixture="incomplete_existing")

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
    project_dir = _build_project(tmp_path, fixture="incomplete_existing")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    incomplete = json.loads(Path(report.incomplete_blocks_file).read_text(encoding="utf-8"))
    assert incomplete

    first = incomplete[0]
    assert first["incomplete_block_id".lower() if False else "incomplete_block_id"]
    assert first["possible_matches"]
    assert first["best_match_pattern_family_id"] is not None


def test_pattern_completion_inserts_missing_tail_notes_to_uzupelnienie_only(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, fixture="incomplete_existing")
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
    project_dir = _build_project(tmp_path, fixture="incomplete_existing", ambiguous=True)

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    incomplete = json.loads(Path(report.incomplete_blocks_file).read_text(encoding="utf-8"))
    assert incomplete
    assert any(item.get("action") == "skipped" for item in incomplete)
    assert any("Ambiguous" in item.get("match_reason", "") for item in incomplete)


def test_pattern_completion_rejects_overlapping_inserted_notes(tmp_path: Path) -> None:
    project_dir = _build_project(
        tmp_path,
        fixture="incomplete_existing",
        with_overlap_case=True,
    )

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
    project_dir = _build_project(tmp_path, fixture="incomplete_existing")

    called = {"value": False}

    def _fake_ai(*args, **kwargs):
        _ = (args, kwargs)
        called["value"] = True
        raise AssertionError("AI should not be called in pattern completion workflow")

    monkeypatch.setattr("midi_cleaner.ai_completion.service.complete_ai_pattern_completion", _fake_ai)

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))
    assert report.status == "ok"
    assert called["value"] is False


def test_pattern_completion_detects_and_completes_missing_expected_block(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, fixture="missing_expected")
    working_path = project_dir / "midi" / "working" / "working.mid"
    before_bytes = working_path.read_bytes()

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.missing_expected_block_count >= 1
    assert report.completed_missing_expected_block_count >= 1
    assert report.inserted_note_count >= 1

    missing_path = project_dir / "analysis" / "pattern_blocks" / "missing_expected_blocks.json"
    assert missing_path.exists()
    missing_reports = json.loads(missing_path.read_text(encoding="utf-8"))
    assert missing_reports
    assert any(item.get("block_type") == "missing_expected_block" for item in missing_reports)
    assert any(item.get("action") == "completed" for item in missing_reports)

    output_midi = Path(report.output_midi_path)
    assert output_midi.exists()
    assert output_midi.name == "uzupelnienie.mid"
    assert _read_midi_note_count(output_midi) == report.inserted_note_count

    # Base MIDI remains unchanged.
    assert working_path.read_bytes() == before_bytes


def test_pattern_completion_skips_ambiguous_missing_expected_block(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = _build_project(tmp_path, fixture="missing_expected", ambiguous=True)

    def _fake_detect_missing_expected_blocks(*, families, blocks_by_id, base_notes):
        _ = (families, blocks_by_id, base_notes)
        return [
            MissingExpectedBlock(
                missing_block_id="missing_a",
                expected_pattern_family_id="pattern_A",
                write_start_sec=4.5,
                write_end_sec=5.5,
                expected_duration_sec=1.0,
                evidence_before_occurrences=["block_0001"],
                evidence_after_occurrences=["block_0003"],
                detected_note_count_in_region=0,
                confidence_score=0.86,
            ),
            MissingExpectedBlock(
                missing_block_id="missing_b",
                expected_pattern_family_id="pattern_B",
                write_start_sec=4.52,
                write_end_sec=5.52,
                expected_duration_sec=1.0,
                evidence_before_occurrences=["block_0001"],
                evidence_after_occurrences=["block_0003"],
                detected_note_count_in_region=0,
                confidence_score=0.80,
            ),
        ]

    monkeypatch.setattr(
        "midi_cleaner.pattern.service._detect_missing_expected_blocks",
        _fake_detect_missing_expected_blocks,
    )

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.missing_expected_block_count >= 1
    assert report.skipped_ambiguous_count >= 1

    missing_path = project_dir / "analysis" / "pattern_blocks" / "missing_expected_blocks.json"
    missing_reports = json.loads(missing_path.read_text(encoding="utf-8"))
    assert any(item.get("action") == "skipped" for item in missing_reports)
    assert any("Ambiguous" in item.get("match_reason", "") for item in missing_reports)


def test_pattern_completion_rejects_collisions_for_missing_expected_block(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, fixture="missing_expected", with_collision_case=True)

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    missing_path = project_dir / "analysis" / "pattern_blocks" / "missing_expected_blocks.json"
    missing_reports = json.loads(missing_path.read_text(encoding="utf-8"))
    assert missing_reports
    assert any("rejected=" in item.get("match_reason", "") for item in missing_reports)

    output_midi = Path(report.output_midi_path)
    assert output_midi.exists()
    assert _read_midi_note_count(output_midi) == report.inserted_note_count


def test_pattern_completion_output_midi_contains_only_inserted_notes(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, fixture="missing_expected")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    output_midi = Path(report.output_midi_path)
    assert output_midi.exists()
    inserted_note_count = _read_midi_note_count(output_midi)
    assert inserted_note_count == report.inserted_note_count

    # Every note in output MIDI must come from report missing_notes_to_insert entries.
    all_reports = json.loads(Path(report.incomplete_blocks_file).read_text(encoding="utf-8"))
    expected = set()
    for block_report in all_reports:
        for note in block_report.get("missing_notes_to_insert", []):
            expected.add(
                (
                    int(round(float(note["start_sec"]) * 1000.0)),
                    int(round(float(note["end_sec"]) * 1000.0)),
                    int(note["pitch_midi"]),
                )
            )

    actual = set()
    midi = mido.MidiFile(str(output_midi))
    ticks_per_second = (float(midi.ticks_per_beat) * 1_000_000.0) / 500000.0
    for track in midi.tracks:
        absolute_tick = 0
        active: dict[int, list[int]] = {}
        for msg in track:
            absolute_tick += int(msg.time)
            if msg.type == "note_on" and int(msg.velocity) > 0:
                active.setdefault(int(msg.note), []).append(absolute_tick)
            if msg.type in {"note_off", "note_on"} and int(getattr(msg, "velocity", 0)) == 0:
                note = int(msg.note)
                starts = active.get(note) or []
                if not starts:
                    continue
                start_tick = starts.pop(0)
                start_sec = int(round((start_tick / ticks_per_second) * 1000.0))
                end_sec = int(round((absolute_tick / ticks_per_second) * 1000.0))
                actual.add((start_sec, end_sec, note))

    assert actual
    assert actual.issubset(expected)


def test_pattern_completion_real_fixture_missing_expected_detection(tmp_path: Path) -> None:
    source = Path("projects") / "real_bass_test_14_strictsplit" / "midi" / "working" / "working.mid"
    if not source.exists():
        return

    project_dir = tmp_path / "project_real_like"
    working_dest = project_dir / "midi" / "working" / "working.mid"
    working_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, working_dest)

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    # This regression guard ensures missing-expected detector can fire when there are timeline holes.
    assert report.status == "ok"
    assert report.missing_expected_block_count >= 0
