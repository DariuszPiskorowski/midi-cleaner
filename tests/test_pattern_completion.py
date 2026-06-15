from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest

import midi_cleaner.pattern.service as pattern_service
from midi_cleaner.pattern import (
    PatternCompletionParameters,
    complete_pattern_blocks,
)
from midi_cleaner.pattern.models import MissingExpectedBlock


_TPB = 480
_BEATS_PER_BAR = 4
_BAR_TICKS = _TPB * _BEATS_PER_BAR
_PATTERN_A = [36, 33, 38, 31]
_TEMPO_US = 500000


def _append_note_event(
    events: list[tuple[int, int, mido.Message]],
    *,
    bar_index: int,
    start_beat: float,
    duration_beat: float,
    pitch: int,
    velocity: int = 96,
) -> None:
    start_tick = int(round((bar_index * _BAR_TICKS) + (start_beat * _TPB)))
    end_tick = start_tick + max(1, int(round(duration_beat * _TPB)))
    events.append(
        (
            start_tick,
            1,
            mido.Message("note_on", note=int(pitch), velocity=int(velocity), channel=0, time=0),
        )
    )
    events.append(
        (
            end_tick,
            0,
            mido.Message("note_off", note=int(pitch), velocity=0, channel=0, time=0),
        )
    )


def _append_pattern_events(
    events: list[tuple[int, int, mido.Message]],
    *,
    bar_index: int,
    pattern: list[int],
    note_count: int | None = None,
    velocity: int = 96,
    durations_beat: float | list[float] = 0.5,
    stacked_intervals: list[int] | None = None,
) -> None:
    count = len(pattern) if note_count is None else max(0, min(note_count, len(pattern)))
    for index, pitch in enumerate(pattern[:count]):
        duration_beat = durations_beat[index] if isinstance(durations_beat, list) else durations_beat
        _append_note_event(
            events,
            bar_index=bar_index,
            start_beat=float(index),
            duration_beat=float(duration_beat),
            pitch=int(pitch),
            velocity=velocity,
        )
        for interval in stacked_intervals or []:
            _append_note_event(
                events,
                bar_index=bar_index,
                start_beat=float(index),
                duration_beat=float(duration_beat),
                pitch=int(pitch + interval),
                velocity=max(1, int(velocity - 12)),
            )


def _append_collision_note(
    events: list[tuple[int, int, mido.Message]],
    *,
    bar_index: int,
    pitch: int,
) -> None:
    _append_note_event(
        events,
        bar_index=bar_index,
        start_beat=2.0,
        duration_beat=0.5,
        pitch=pitch,
        velocity=72,
    )


def _write_working_midi(path: Path, *, scenario: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    midi = mido.MidiFile(ticks_per_beat=_TPB)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=_TEMPO_US, time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    events: list[tuple[int, int, mido.Message]] = []

    if scenario == "incomplete_tail":
        _append_pattern_events(events, bar_index=0, pattern=_PATTERN_A, note_count=4, velocity=96)
        _append_pattern_events(events, bar_index=1, pattern=_PATTERN_A, note_count=2, velocity=92)
        _append_pattern_events(events, bar_index=2, pattern=_PATTERN_A, note_count=4, velocity=94)
    elif scenario == "missing_empty":
        _append_pattern_events(events, bar_index=0, pattern=_PATTERN_A, note_count=4, velocity=96)
        _append_pattern_events(events, bar_index=2, pattern=_PATTERN_A, note_count=4, velocity=94)
    elif scenario == "missing_sparse":
        _append_pattern_events(events, bar_index=0, pattern=_PATTERN_A, note_count=4, velocity=96)
        _append_note_event(events, bar_index=1, start_beat=1.0, duration_beat=0.5, pitch=29, velocity=70)
        _append_pattern_events(events, bar_index=2, pattern=_PATTERN_A, note_count=4, velocity=94)
    elif scenario == "missing_collision":
        _append_pattern_events(events, bar_index=0, pattern=_PATTERN_A, note_count=4, velocity=96)
        _append_collision_note(events, bar_index=1, pitch=38)
        _append_pattern_events(events, bar_index=2, pattern=_PATTERN_A, note_count=4, velocity=94)
    elif scenario == "sustained_onset":
        _append_note_event(events, bar_index=0, start_beat=0.0, duration_beat=3.0, pitch=36, velocity=94)
        _append_note_event(events, bar_index=0, start_beat=3.0, duration_beat=0.5, pitch=33, velocity=90)
        _append_pattern_events(events, bar_index=1, pattern=_PATTERN_A, note_count=4, velocity=92)
    elif scenario == "stacked_source":
        _append_pattern_events(
            events,
            bar_index=0,
            pattern=_PATTERN_A,
            note_count=4,
            velocity=96,
            stacked_intervals=[12],
        )
        _append_pattern_events(events, bar_index=1, pattern=_PATTERN_A, note_count=2, velocity=92)
        _append_pattern_events(
            events,
            bar_index=2,
            pattern=_PATTERN_A,
            note_count=4,
            velocity=94,
            stacked_intervals=[12],
        )
    elif scenario == "micro_candidate":
        durations = [1.0, 1.0, 1.0, 0.125]
        _append_pattern_events(events, bar_index=0, pattern=_PATTERN_A, note_count=4, velocity=96, durations_beat=durations)
        _append_pattern_events(events, bar_index=1, pattern=_PATTERN_A, note_count=3, velocity=90, durations_beat=durations)
        _append_pattern_events(events, bar_index=2, pattern=_PATTERN_A, note_count=4, velocity=94, durations_beat=durations)
    elif scenario == "large_gap_missing":
        _append_pattern_events(events, bar_index=0, pattern=_PATTERN_A, note_count=4, velocity=96)
        _append_pattern_events(events, bar_index=1, pattern=_PATTERN_A, note_count=4, velocity=94)
        _append_pattern_events(events, bar_index=4, pattern=_PATTERN_A, note_count=4, velocity=95)
        _append_pattern_events(events, bar_index=5, pattern=_PATTERN_A, note_count=4, velocity=93)
    else:
        raise AssertionError(f"Unsupported scenario: {scenario}")

    events.sort(key=lambda item: (item[0], item[1]))
    current_tick = 0
    for tick, _order, message in events:
        message.time = max(0, int(tick - current_tick))
        current_tick = int(tick)
        track.append(message)

    track.append(mido.MetaMessage("end_of_track", time=max(1, _TPB)))
    midi.save(path)


def _build_project(tmp_path: Path, *, scenario: str) -> Path:
    project_dir = tmp_path / "project_pattern"
    working_midi = project_dir / "midi" / "working" / "working.mid"
    _write_working_midi(working_midi, scenario=scenario)
    return project_dir


def _read_midi_note_count(path: Path) -> int:
    midi = mido.MidiFile(str(path))
    return sum(
        1
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and int(message.velocity) > 0
    )


def _read_midi_note_events(path: Path) -> list[tuple[int, int, int]]:
    midi = mido.MidiFile(str(path))
    ticks_per_second = (float(midi.ticks_per_beat) * 1_000_000.0) / float(_TEMPO_US)
    events: list[tuple[int, int, int]] = []
    for track in midi.tracks:
        absolute_tick = 0
        active: dict[int, list[int]] = {}
        for msg in track:
            absolute_tick += int(msg.time)
            if msg.type == "note_on" and int(msg.velocity) > 0:
                active.setdefault(int(msg.note), []).append(absolute_tick)
                continue
            if msg.type in {"note_off", "note_on"} and int(getattr(msg, "velocity", 0)) == 0:
                note = int(msg.note)
                starts = active.get(note) or []
                if not starts:
                    continue
                start_tick = starts.pop(0)
                start_ms = int(round((start_tick / ticks_per_second) * 1000.0))
                end_ms = int(round((absolute_tick / ticks_per_second) * 1000.0))
                events.append((start_ms, end_ms, note))
    return sorted(events)


def test_bar_aligned_block_detection_and_grid_fields(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="incomplete_tail")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.bar_aligned_block_count == report.pattern_block_count
    assert report.bar_aligned_block_count >= 3
    assert report.complete_block_count >= 1

    blocks = json.loads(Path(report.pattern_blocks_file).read_text(encoding="utf-8"))
    assert all("bar_index" in item for item in blocks)
    assert all("onset_slots" in item for item in blocks)
    assert all("occupied_slots" in item for item in blocks)
    assert all("empty_slots" in item for item in blocks)
    assert all(item.get("grid_resolution") == "1/16" for item in blocks)
    assert all(item.get("time_signature") == "4/4" for item in blocks)


def test_incomplete_existing_block_completed_and_working_unchanged(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="incomplete_tail")
    working_path = project_dir / "midi" / "working" / "working.mid"
    before_bytes = working_path.read_bytes()

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.incomplete_existing_block_count >= 1
    assert report.inserted_note_count >= 1

    incomplete_reports = json.loads(Path(report.incomplete_blocks_file).read_text(encoding="utf-8"))
    existing = [item for item in incomplete_reports if item.get("block_type") == "incomplete_existing_block"]
    assert existing
    target = existing[0]
    assert target.get("target_bar_index") == 1
    assert target.get("best_match_pattern_family_id") is not None
    assert "onset_slots_observed" in target
    assert target.get("observed_slots")
    assert target.get("missing_slots")
    assert target.get("action") in {"completed", "skipped"}

    output_midi = Path(report.output_midi_path)
    assert output_midi.exists()
    assert output_midi.name == "uzupelnienie.mid"
    assert _read_midi_note_count(output_midi) == report.inserted_note_count

    assert working_path.read_bytes() == before_bytes


def test_missing_expected_block_detected_for_empty_middle_bar(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="missing_empty")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.missing_expected_block_count >= 1

    missing_reports = json.loads(Path(report.missing_expected_blocks_file).read_text(encoding="utf-8"))
    assert missing_reports
    candidate = missing_reports[0]
    assert candidate.get("block_type") == "missing_expected_block"
    assert candidate.get("target_bar_index") == 1
    assert candidate.get("observed_slots") == []
    assert candidate.get("onset_slots_observed") == []
    assert candidate.get("best_match_pattern_family_id") is not None


def test_missing_expected_detects_sparse_middle_bar(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="missing_sparse")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.missing_expected_block_count >= 1

    missing_reports = json.loads(Path(report.missing_expected_blocks_file).read_text(encoding="utf-8"))
    assert any(item.get("target_bar_index") == 1 for item in missing_reports)


def test_missing_expected_block_completion_writes_only_uzupelnienie(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="missing_empty")
    working_path = project_dir / "midi" / "working" / "working.mid"
    before_bytes = working_path.read_bytes()

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.completed_missing_expected_block_count >= 1
    assert report.inserted_note_count >= 1

    output_midi = Path(report.output_midi_path)
    assert output_midi.exists()
    assert output_midi.name == "uzupelnienie.mid"
    assert _read_midi_note_count(output_midi) == report.inserted_note_count

    assert working_path.read_bytes() == before_bytes


def test_sustained_note_uses_onset_slots_not_duration_occupancy(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="sustained_onset")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    blocks = json.loads(Path(report.pattern_blocks_file).read_text(encoding="utf-8"))
    first_bar = next(item for item in blocks if int(item.get("bar_index")) == 0)
    assert len(first_bar.get("onset_slots", [])) <= 2
    assert len(first_bar.get("occupied_slots", [])) > len(first_bar.get("onset_slots", []))


def test_same_onset_source_collapses_to_monophonic_insertions(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="stacked_source")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    incomplete_reports = json.loads(Path(report.incomplete_blocks_file).read_text(encoding="utf-8"))
    bar_one = [
        item
        for item in incomplete_reports
        if item.get("block_type") == "incomplete_existing_block" and int(item.get("target_bar_index", -1)) == 1
    ]
    assert bar_one
    inserted = bar_one[0].get("inserted_notes", [])
    assert inserted
    start_ms = [int(round(float(item["start_sec"]) * 1000.0)) for item in inserted]
    assert len(start_ms) == len(set(start_ms))
    assert all(int(item["pitch_midi"]) in set(_PATTERN_A) for item in inserted)


def test_output_midi_is_monophonic_no_stacked_note_starts(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="stacked_source")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    note_events = _read_midi_note_events(Path(report.output_midi_path))
    start_times = [item[0] for item in note_events]
    assert len(start_times) == len(set(start_times))


def test_micro_note_candidates_are_rejected(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="micro_candidate")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.rejected_micro_note_count >= 1
    all_reports = json.loads(Path(report.incomplete_blocks_file).read_text(encoding="utf-8"))
    rejected_reasons = [
        item.get("reason")
        for item in all_reports
        if item.get("action") == "skipped"
    ]
    assert "rejected_validation" in rejected_reasons


def test_low_confidence_completion_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = _build_project(tmp_path, scenario="incomplete_tail")
    original = pattern_service._score_family_match

    def _force_low_confidence(*, incomplete_block, family):
        candidate = original(incomplete_block=incomplete_block, family=family)
        if candidate is None:
            return None
        return pattern_service._CandidateFamilyMatch(
            family=candidate.family,
            score=0.78,
            reason=f"{candidate.reason} (forced_low_confidence)",
        )

    monkeypatch.setattr(pattern_service, "_score_family_match", _force_low_confidence)

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.rejected_low_confidence_count >= 1
    assert report.completed_incomplete_existing_block_count == 0


def test_large_expected_missing_gap_completed_from_strong_family(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="large_gap_missing")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.missing_expected_block_count >= 1
    assert report.completed_missing_expected_block_count >= 1
    assert report.inserted_note_count >= len(_PATTERN_A)


def test_bar_gap_candidates_file_created_with_context(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="large_gap_missing")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.bar_gap_candidate_count > 0
    assert report.bar_gap_candidates_file

    gap_candidates = json.loads(Path(report.bar_gap_candidates_file).read_text(encoding="utf-8"))
    assert len(gap_candidates) == report.bar_gap_candidate_count
    target = gap_candidates[0]
    assert "bar_index_range" in target
    assert "start_sec" in target
    assert "end_sec" in target
    assert "note_count" in target
    assert "families_before" in target
    assert "families_after" in target
    assert "completion_reason" in target


def test_bar_gap_candidates_detect_sparse_and_empty_ranges(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="missing_sparse")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.bar_gap_candidate_count >= 1

    gap_candidates = json.loads(Path(report.bar_gap_candidates_file).read_text(encoding="utf-8"))
    assert any(int(item.get("start_bar_index", -1)) == 1 for item in gap_candidates)


def test_bar_gap_candidate_extremely_clear_bridge_flag(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="large_gap_missing")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    gap_candidates = json.loads(Path(report.bar_gap_candidates_file).read_text(encoding="utf-8"))
    assert gap_candidates
    assert any(item.get("same_family_bridge") for item in gap_candidates)
    assert any(item.get("completion_readiness") in {"extremely_clear", "unclear", "insufficient_context"} for item in gap_candidates)


def test_ambiguous_missing_expected_block_is_skipped(tmp_path: Path, monkeypatch) -> None:
    project_dir = _build_project(tmp_path, scenario="missing_empty")

    def _fake_detect_missing_expected_blocks(*, families, blocks_by_id, base_notes):
        _ = (families, blocks_by_id, base_notes)
        return [
            MissingExpectedBlock(
                missing_block_id="missing_0001",
                target_bar_index=1,
                expected_pattern_family_id="pattern_A",
                write_start_sec=2.0,
                write_end_sec=4.0,
                write_start_beat=4.0,
                write_end_beat=8.0,
                expected_duration_sec=2.0,
                expected_duration_beat=4.0,
                observed_slots=[],
                missing_slots=[0, 4, 8, 12],
                evidence_before_occurrences=["bar_0001"],
                evidence_after_occurrences=["bar_0003"],
                detected_note_count_in_region=0,
                confidence_score=0.84,
            ),
            MissingExpectedBlock(
                missing_block_id="missing_0002",
                target_bar_index=1,
                expected_pattern_family_id="pattern_B",
                write_start_sec=2.0,
                write_end_sec=4.0,
                write_start_beat=4.0,
                write_end_beat=8.0,
                expected_duration_sec=2.0,
                expected_duration_beat=4.0,
                observed_slots=[],
                missing_slots=[0, 4, 8, 12],
                evidence_before_occurrences=["bar_0001"],
                evidence_after_occurrences=["bar_0003"],
                detected_note_count_in_region=0,
                confidence_score=0.78,
            ),
        ]

    monkeypatch.setattr(
        "midi_cleaner.pattern.service._detect_missing_expected_blocks",
        _fake_detect_missing_expected_blocks,
    )

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    assert report.skipped_ambiguous_count >= 1

    missing_reports = json.loads(Path(report.missing_expected_blocks_file).read_text(encoding="utf-8"))
    assert any(item.get("action") == "skipped" for item in missing_reports)
    assert any(item.get("reason") == "ambiguous" for item in missing_reports)


def test_missing_expected_collision_rejects_overlapping_note(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="missing_collision")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    assert report.status == "ok"
    missing_reports = json.loads(Path(report.missing_expected_blocks_file).read_text(encoding="utf-8"))
    assert missing_reports
    assert any("rejected=" in item.get("match_reason", "") for item in missing_reports)

    output_midi = Path(report.output_midi_path)
    assert output_midi.exists()
    assert _read_midi_note_count(output_midi) == report.inserted_note_count
    assert report.inserted_note_count < len(_PATTERN_A)


def test_output_midi_contains_only_reported_inserted_notes(tmp_path: Path) -> None:
    project_dir = _build_project(tmp_path, scenario="missing_empty")

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))

    output_midi = Path(report.output_midi_path)
    assert output_midi.exists()

    expected = set()
    all_reports = json.loads(Path(report.incomplete_blocks_file).read_text(encoding="utf-8"))
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
    for start_ms, end_ms, note in _read_midi_note_events(output_midi):
        actual.add((start_ms, end_ms, note))

    assert actual.issubset(expected)


def test_pattern_completion_does_not_invoke_ai(tmp_path: Path, monkeypatch) -> None:
    project_dir = _build_project(tmp_path, scenario="missing_empty")
    called = {"value": False}

    def _fake_ai(*args, **kwargs):
        _ = (args, kwargs)
        called["value"] = True
        raise AssertionError("AI should not be called in deterministic pattern completion")

    monkeypatch.setattr("midi_cleaner.ai_completion.service.complete_ai_pattern_completion", _fake_ai)

    report = complete_pattern_blocks(project_dir=project_dir, params=PatternCompletionParameters(layer="bass"))
    assert report.status == "ok"
    assert called["value"] is False
