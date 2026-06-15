from __future__ import annotations

from collections import Counter
from typing import Any


def build_ai_request_pack(
    pattern_pack: dict[str, object],
    max_notes: int = 180,
    max_activity_regions: int = 180,
    max_pitch_sections: int = 240,
) -> dict[str, object]:
    base_notes = _as_dict_list(pattern_pack.get("base_notes"))
    audio_activity_regions = _as_dict_list(pattern_pack.get("audio_activity_regions"))
    pitch_contour_summary = _as_dict_list(pattern_pack.get("pitch_contour_summary"))
    pattern_windows = _as_dict_list(pattern_pack.get("pattern_windows"))
    allowed_completion_regions = _as_dict_list(pattern_pack.get("allowed_completion_regions"))

    selected_base_notes = _select_base_notes(
        base_notes=base_notes,
        activity_regions=audio_activity_regions,
        pitch_sections=pitch_contour_summary,
        limit=max_notes,
    )
    selected_activity_regions = _select_activity_regions(
        regions=audio_activity_regions,
        base_notes=base_notes,
        limit=max_activity_regions,
    )
    selected_pitch_sections = _select_pitch_sections(
        sections=pitch_contour_summary,
        limit=max_pitch_sections,
    )
    summarized_windows = _summarize_pattern_windows(
        windows=pattern_windows,
        base_notes=base_notes,
    )

    compact_pack: dict[str, object] = {
        "version": pattern_pack.get("version", "1.0"),
        "track_role": pattern_pack.get("track_role", "bass"),
        "timeline": pattern_pack.get("timeline", {}),
        "base_midi_summary": pattern_pack.get("base_midi_summary", {}),
        "instructions_for_ai": pattern_pack.get("instructions_for_ai", {}),
        "completion_scope": "target_regions_only",
        "max_completion_notes_is_upper_bound_not_target": True,
        "reject_notes_outside_allowed_regions": True,
        "timeline_sync": {
            "time_origin": "wav_seconds",
            "must_align_with": "working.mid",
            "output_is_additive_layer": True,
        },
        "base_notes_are_occupied": True,
        "base_occupancy_rules": {
            "do_not_place_ai_note_on_base_onset_within_ms": 30,
            "do_not_overlap_same_or_near_pitch_base_note_ratio": 0.7,
            "completion_track_role": "additive_missing_pattern_only",
        },
        "base_notes": selected_base_notes,
        "audio_activity_regions": selected_activity_regions,
        "pitch_contour_summary": selected_pitch_sections,
        "pattern_windows": summarized_windows,
        "allowed_completion_regions": _compact_allowed_completion_regions(
            allowed_completion_regions=allowed_completion_regions
        ),
        "compact_request": True,
        "source_pack_was_compacted": True,
        "original_counts": {
            "base_notes": len(base_notes),
            "audio_activity_regions": len(audio_activity_regions),
            "pitch_contour_summary": len(pitch_contour_summary),
            "pattern_windows": len(pattern_windows),
            "allowed_completion_regions": len(allowed_completion_regions),
        },
        "included_counts": {
            "base_notes": len(selected_base_notes),
            "audio_activity_regions": len(selected_activity_regions),
            "pitch_contour_summary": len(selected_pitch_sections),
            "pattern_windows": len(summarized_windows),
            "allowed_completion_regions": len(allowed_completion_regions),
        },
    }

    pack_warnings = pattern_pack.get("pack_warnings")
    if isinstance(pack_warnings, list):
        compact_pack["pack_warnings"] = [str(item) for item in pack_warnings[:16]]

    return compact_pack


def _select_base_notes(
    *,
    base_notes: list[dict[str, Any]],
    activity_regions: list[dict[str, Any]],
    pitch_sections: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, object]]:
    if not base_notes:
        return []

    note_count = len(base_notes)
    if note_count <= max(1, limit):
        return [_compact_note(note) for note in base_notes]

    priorities: dict[int, int] = {}

    def _mark(index: int, score: int) -> None:
        if 0 <= index < note_count:
            previous = priorities.get(index)
            if previous is None or score > previous:
                priorities[index] = score

    # Keep beginning and ending context to preserve phrase boundaries.
    first_keep = min(24, note_count)
    last_start = max(0, note_count - 24)
    for index in range(first_keep):
        _mark(index, 1000)
    for index in range(last_start, note_count):
        _mark(index, 1000)

    region_pick_count = max(6, min(len(activity_regions), max(1, limit // 6)))
    top_regions = _top_activity_regions(activity_regions, region_pick_count)
    for region in top_regions:
        region_start = _to_float(region.get("start_sec"), 0.0)
        region_end = _to_float(region.get("end_sec"), region_start)
        overlap_indices = _note_indices_overlapping(base_notes, region_start, region_end)
        if overlap_indices:
            _mark(overlap_indices[0], 900)
            _mark(overlap_indices[len(overlap_indices) // 2], 920)
            _mark(overlap_indices[-1], 900)
            continue

        center_sec = (region_start + region_end) / 2.0
        nearest = _nearest_note_index(base_notes, center_sec)
        _mark(nearest, 880)
        _mark(nearest - 1, 820)
        _mark(nearest + 1, 820)

    for change_sec in _pitch_change_times(pitch_sections, max(8, limit // 8)):
        nearest = _nearest_note_index(base_notes, change_sec)
        _mark(nearest, 860)
        _mark(nearest - 1, 810)
        _mark(nearest + 1, 810)

    even_target = min(note_count, max(1, limit))
    for index in _downsample_indices(note_count, even_target):
        _mark(index, 600)

    ranked = sorted(priorities.items(), key=lambda item: (-item[1], item[0]))
    selected_indices = [index for index, _score in ranked[: max(1, limit)]]
    selected_indices.sort()

    return [_compact_note(base_notes[index]) for index in selected_indices]


def _top_activity_regions(regions: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not regions:
        return []

    ranked = sorted(
        regions,
        key=lambda region: (
            -_to_float(region.get("rms_peak"), _to_float(region.get("rms_mean"), 0.0)),
            -_to_int(region.get("onset_count"), 0),
            _to_float(region.get("start_sec"), 0.0),
        ),
    )
    return ranked[:count]


def _pitch_change_times(pitch_sections: list[dict[str, Any]], limit: int) -> list[float]:
    if limit <= 0 or len(pitch_sections) < 2:
        return []

    changes: list[tuple[float, float]] = []
    previous = pitch_sections[0]
    for section in pitch_sections[1:]:
        prev_pitch = previous.get("dominant_pitch_midi")
        curr_pitch = section.get("dominant_pitch_midi")
        prev_conf = _to_float(previous.get("mean_confidence"), 0.0)
        curr_conf = _to_float(section.get("mean_confidence"), 0.0)

        delta = 0.0
        if isinstance(prev_pitch, (int, float)) and isinstance(curr_pitch, (int, float)):
            delta = abs(float(curr_pitch) - float(prev_pitch))

        confidence_delta = abs(curr_conf - prev_conf)
        change_score = delta + (confidence_delta * 2.0)

        if change_score >= 0.75:
            anchor = _to_float(section.get("start_sec"), _to_float(previous.get("end_sec"), 0.0))
            changes.append((anchor, change_score))

        previous = section

    changes.sort(key=lambda item: -item[1])
    selected = changes[:limit]
    selected.sort(key=lambda item: item[0])
    return [item[0] for item in selected]


def _select_activity_regions(
    *,
    regions: list[dict[str, Any]],
    base_notes: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, object]]:
    if not regions:
        return []

    if len(regions) <= max(1, limit):
        return [_compact_activity_region(region) for region in regions]

    notes_spans = [
        (
            _to_float(note.get("start_sec"), 0.0),
            _to_float(note.get("end_sec"), _to_float(note.get("start_sec"), 0.0)),
        )
        for note in base_notes
    ]

    scored: list[dict[str, object]] = []
    for index, region in enumerate(regions):
        start_sec = _to_float(region.get("start_sec"), 0.0)
        end_sec = _to_float(region.get("end_sec"), start_sec)
        rms_peak = _to_float(region.get("rms_peak"), _to_float(region.get("rms_mean"), 0.0))
        onset_count = _to_int(region.get("onset_count"), 0)
        overlap_count = sum(
            1
            for note_start, note_end in notes_spans
            if note_start < end_sec and note_end > start_sec
        )
        scored.append(
            {
                "index": index,
                "start_sec": start_sec,
                "rms_peak": rms_peak,
                "onset_count": onset_count,
                "overlap_count": overlap_count,
            }
        )

    selected_indices: set[int] = set()
    bucket = max(8, max(1, limit // 3))

    for item in sorted(scored, key=lambda row: (-float(row["rms_peak"]), int(row["index"])))[
        :bucket
    ]:
        selected_indices.add(int(item["index"]))

    for item in sorted(
        scored,
        key=lambda row: (-int(row["onset_count"]), -float(row["rms_peak"]), int(row["index"])),
    )[:bucket]:
        selected_indices.add(int(item["index"]))

    for item in sorted(
        scored,
        key=lambda row: (int(row["overlap_count"]), -float(row["rms_peak"]), int(row["index"])),
    )[:bucket]:
        selected_indices.add(int(item["index"]))

    if len(selected_indices) < max(1, limit):
        ranked = sorted(
            scored,
            key=lambda row: (
                -((float(row["rms_peak"]) * 1.5) + float(row["onset_count"])),
                int(row["overlap_count"]),
                int(row["index"]),
            ),
        )
        for item in ranked:
            selected_indices.add(int(item["index"]))
            if len(selected_indices) >= max(1, limit):
                break

    selected = sorted(
        selected_indices,
        key=lambda idx: _to_float(regions[idx].get("start_sec"), float(idx)),
    )[: max(1, limit)]
    return [_compact_activity_region(regions[index]) for index in selected]


def _select_pitch_sections(
    *,
    sections: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, object]]:
    if not sections:
        return []

    indices = _downsample_indices(len(sections), max(1, limit))
    return [_compact_pitch_section(sections[index]) for index in indices]


def _summarize_pattern_windows(
    *,
    windows: list[dict[str, Any]],
    base_notes: list[dict[str, Any]],
) -> list[dict[str, object]]:
    if not windows:
        return []

    pitch_by_note_id: dict[str, int] = {}
    for note in base_notes:
        note_id = str(note.get("note_id", ""))
        if not note_id:
            continue
        pitch = note.get("pitch_midi")
        if isinstance(pitch, (int, float)):
            pitch_by_note_id[note_id] = int(pitch)

    summarized: list[dict[str, object]] = []
    for fallback_index, window in enumerate(windows):
        note_ids = [str(item) for item in _as_list(window.get("base_notes")) if isinstance(item, str)]
        pitches = [pitch_by_note_id[note_id] for note_id in note_ids if note_id in pitch_by_note_id]

        rhythmic = window.get("rhythmic_summary")
        rhythmic_dict = rhythmic if isinstance(rhythmic, dict) else {}

        common_pitches = [
            {"pitch_midi": int(pitch), "count": int(count)}
            for pitch, count in Counter(pitches).most_common(8)
        ]

        summarized.append(
            {
                "window_index": _to_int(window.get("window_index"), fallback_index),
                "start_sec": _rounded(window.get("start_sec")),
                "end_sec": _rounded(window.get("end_sec")),
                "note_count": (
                    len(note_ids)
                    if note_ids
                    else _to_int(window.get("note_count"), 0)
                ),
                "pitch_min": (
                    int(min(pitches))
                    if pitches
                    else _to_optional_int(window.get("pitch_min"))
                ),
                "pitch_max": (
                    int(max(pitches))
                    if pitches
                    else _to_optional_int(window.get("pitch_max"))
                ),
                "common_pitches": common_pitches,
                "first_note_ids": note_ids[:8],
                "rhythmic_summary": {
                    "note_onsets_sec": _float_list(rhythmic_dict.get("note_onsets_sec"), 16),
                    "intervals_sec": _float_list(rhythmic_dict.get("intervals_sec"), 16),
                    "common_durations_sec": _float_list(
                        rhythmic_dict.get("common_durations_sec"),
                        16,
                    ),
                },
            }
        )

    return summarized


def _compact_note(note: dict[str, Any]) -> dict[str, object]:
    start_sec = _to_float(note.get("start_sec"), 0.0)
    end_sec = _to_float(note.get("end_sec"), start_sec)
    compact = {
        "note_id": str(note.get("note_id", "")),
        "start_sec": _rounded(start_sec),
        "end_sec": _rounded(end_sec),
        "duration_sec": _rounded(max(0.0, end_sec - start_sec)),
        "pitch_midi": _to_int(note.get("pitch_midi"), 0),
        "velocity": _to_int(note.get("velocity"), 1),
        "confidence": _rounded(_to_float(note.get("confidence"), 0.0)),
        "occupied": True,
    }
    source = note.get("source")
    if isinstance(source, str) and source:
        compact["source"] = source
    return compact


def _compact_activity_region(region: dict[str, Any]) -> dict[str, object]:
    start_sec = _to_float(region.get("start_sec"), 0.0)
    end_sec = _to_float(region.get("end_sec"), start_sec)
    duration_sec = region.get("duration_sec")
    if not isinstance(duration_sec, (int, float)):
        duration_sec = max(0.0, end_sec - start_sec)

    return {
        "start_sec": _rounded(start_sec),
        "end_sec": _rounded(end_sec),
        "duration_sec": _rounded(_to_float(duration_sec, max(0.0, end_sec - start_sec))),
        "rms_peak": _rounded(_to_float(region.get("rms_peak"), _to_float(region.get("rms_mean"), 0.0)), 8),
        "rms_mean": _rounded(_to_float(region.get("rms_mean"), 0.0), 8),
        "onset_count": _to_int(region.get("onset_count"), 0),
        "dominant_pitch_midi": _to_optional_int(region.get("dominant_pitch_midi")),
        "pitch_confidence": _rounded(_to_float(region.get("pitch_confidence"), 0.0)),
    }


def _compact_pitch_section(section: dict[str, Any]) -> dict[str, object]:
    return {
        "start_sec": _rounded(section.get("start_sec")),
        "end_sec": _rounded(section.get("end_sec")),
        "dominant_pitch_midi": _to_optional_int(section.get("dominant_pitch_midi")),
        "pitch_midi_mean": _rounded(section.get("pitch_midi_mean")),
        "voiced_ratio": _rounded(section.get("voiced_ratio")),
        "mean_confidence": _rounded(section.get("mean_confidence")),
    }


def _compact_allowed_completion_regions(
    *,
    allowed_completion_regions: list[dict[str, Any]],
) -> list[dict[str, object]]:
    compacted: list[dict[str, object]] = []
    for region in allowed_completion_regions:
        allowed_pitch_range = region.get("allowed_pitch_range")
        local_pitch_range = region.get("local_pitch_range")
        rhythmic_summary = region.get("rhythmic_pattern_summary")

        compacted.append(
            {
                "region_id": str(region.get("region_id", "")),
                "start_sec": _rounded(region.get("start_sec")),
                "end_sec": _rounded(region.get("end_sec")),
                "write_start_sec": _rounded(region.get("write_start_sec") or region.get("start_sec")),
                "write_end_sec": _rounded(region.get("write_end_sec") or region.get("end_sec")),
                "reason": str(region.get("reason", "")),
                "context_before_start_sec": _rounded(region.get("context_before_start_sec")),
                "context_after_end_sec": _rounded(region.get("context_after_end_sec")),
                "context_window_before_sec": _rounded(region.get("context_window_before_sec")),
                "context_window_after_sec": _rounded(region.get("context_window_after_sec")),
                "reference_notes_before": [
                    str(item)
                    for item in _as_list(region.get("reference_notes_before"))[:48]
                    if isinstance(item, str)
                ],
                "reference_notes_after": [
                    str(item)
                    for item in _as_list(region.get("reference_notes_after"))[:48]
                    if isinstance(item, str)
                ],
                "notes_before": [
                    _compact_context_note(item)
                    for item in _as_list(region.get("notes_before"))[:64]
                    if isinstance(item, dict)
                ],
                "notes_after": [
                    _compact_context_note(item)
                    for item in _as_list(region.get("notes_after"))[:64]
                    if isinstance(item, dict)
                ],
                "local_pitch_set": [
                    _to_int(value, 0)
                    for value in _as_list(region.get("local_pitch_set"))[:24]
                    if isinstance(value, (int, float, str))
                ],
                "local_pitch_names": [
                    str(value)
                    for value in _as_list(region.get("local_pitch_names"))[:24]
                    if isinstance(value, str)
                ],
                "local_pitch_range": {
                    "min": _to_int(
                        (local_pitch_range.get("min") if isinstance(local_pitch_range, dict) else None),
                        0,
                    ),
                    "max": _to_int(
                        (local_pitch_range.get("max") if isinstance(local_pitch_range, dict) else None),
                        127,
                    ),
                },
                "allowed_pitch_range": {
                    "min": _to_int(
                        (allowed_pitch_range.get("min") if isinstance(allowed_pitch_range, dict) else None),
                        0,
                    ),
                    "max": _to_int(
                        (allowed_pitch_range.get("max") if isinstance(allowed_pitch_range, dict) else None),
                        127,
                    ),
                },
                "preferred_pitches": [
                    _to_int(value, 0)
                    for value in _as_list(region.get("preferred_pitches"))[:16]
                    if isinstance(value, (int, float, str))
                ],
                "forbidden_pitches": [
                    _to_int(value, 0)
                    for value in _as_list(region.get("forbidden_pitches"))[:16]
                    if isinstance(value, (int, float, str))
                ],
                "allow_pitch_outside_local_set": bool(
                    region.get("allow_pitch_outside_local_set", False)
                ),
                "estimated_key_or_scale": str(region.get("estimated_key_or_scale", "unknown")),
                "rhythmic_pattern_summary": {
                    "note_onsets_sec": _float_list(
                        rhythmic_summary.get("note_onsets_sec") if isinstance(rhythmic_summary, dict) else None,
                        24,
                    ),
                    "intervals_sec": _float_list(
                        rhythmic_summary.get("intervals_sec") if isinstance(rhythmic_summary, dict) else None,
                        24,
                    ),
                    "common_durations_sec": _float_list(
                        rhythmic_summary.get("common_durations_sec") if isinstance(rhythmic_summary, dict) else None,
                        16,
                    ),
                },
                "local_rhythm_intervals_sec": _float_list(
                    region.get("local_rhythm_intervals_sec"),
                    24,
                ),
                "detected_local_motif": {
                    "pitch_sequence": [
                        _to_int(value, 0)
                        for value in _as_list(
                            (region.get("detected_local_motif") or {}).get("pitch_sequence")
                            if isinstance(region.get("detected_local_motif"), dict)
                            else None
                        )[:12]
                        if isinstance(value, (int, float, str))
                    ],
                    "interval_sequence": [
                        _to_int(value, 0)
                        for value in _as_list(
                            (region.get("detected_local_motif") or {}).get("interval_sequence")
                            if isinstance(region.get("detected_local_motif"), dict)
                            else None
                        )[:12]
                        if isinstance(value, (int, float, str))
                    ],
                    "rhythm_sequence_sec": _float_list(
                        (region.get("detected_local_motif") or {}).get("rhythm_sequence_sec")
                        if isinstance(region.get("detected_local_motif"), dict)
                        else None,
                        12,
                    ),
                    "confidence": _rounded(
                        (region.get("detected_local_motif") or {}).get("confidence")
                        if isinstance(region.get("detected_local_motif"), dict)
                        else None
                    ),
                },
                "motif_confidence": _rounded(region.get("motif_confidence")),
                "optional_region": bool(region.get("optional_region", False)),
                "expected_note_count_min": _to_int(region.get("expected_note_count_min"), 0),
                "expected_note_count_max": _to_int(region.get("expected_note_count_max"), 0),
                "density_limit_notes_per_sec": _rounded(region.get("density_limit_notes_per_sec")),
                "min_note_duration_sec": _rounded(region.get("min_note_duration_sec")),
                "max_note_duration_sec": _rounded(region.get("max_note_duration_sec")),
                "no_notes_outside_region": bool(region.get("no_notes_outside_region", True)),
                "instruction": str(region.get("instruction", "")),
            }
        )

    return compacted


def _compact_context_note(note: dict[str, Any]) -> dict[str, object]:
    return {
        "note_id": str(note.get("note_id", "")),
        "start_sec": _rounded(note.get("start_sec")),
        "end_sec": _rounded(note.get("end_sec")),
        "pitch_midi": _to_int(note.get("pitch_midi"), 0),
        "pitch_name": str(note.get("pitch_name", "")),
    }


def _note_indices_overlapping(
    notes: list[dict[str, Any]],
    region_start: float,
    region_end: float,
) -> list[int]:
    overlaps: list[int] = []
    for index, note in enumerate(notes):
        note_start = _to_float(note.get("start_sec"), 0.0)
        note_end = _to_float(note.get("end_sec"), note_start)
        if note_start < region_end and note_end > region_start:
            overlaps.append(index)
    return overlaps


def _nearest_note_index(notes: list[dict[str, Any]], target_sec: float) -> int:
    if not notes:
        return 0

    best_index = 0
    best_distance = float("inf")
    for index, note in enumerate(notes):
        note_start = _to_float(note.get("start_sec"), 0.0)
        distance = abs(note_start - target_sec)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _downsample_indices(size: int, limit: int) -> list[int]:
    if size <= 0 or limit <= 0:
        return []
    if size <= limit:
        return list(range(size))
    if limit == 1:
        return [0]

    indices = [0]
    for sample_index in range(1, limit - 1):
        ratio = sample_index / float(limit - 1)
        candidate = int(round(ratio * (size - 1)))
        if candidate <= indices[-1]:
            candidate = indices[-1] + 1
        if candidate >= size - 1:
            candidate = size - 2
        indices.append(candidate)
    indices.append(size - 1)
    return indices


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return []


def _as_dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _float_list(value: object, limit: int) -> list[float]:
    if not isinstance(value, list):
        return []
    output: list[float] = []
    for item in value[:limit]:
        if isinstance(item, (int, float)):
            output.append(_rounded(float(item)))
    return output


def _to_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _to_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _to_optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _rounded(value: object, precision: int = 6) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value), precision)
    if isinstance(value, str):
        try:
            return round(float(value), precision)
        except ValueError:
            return None
    return None
