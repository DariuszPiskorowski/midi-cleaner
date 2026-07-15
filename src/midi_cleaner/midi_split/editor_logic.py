from __future__ import annotations

import copy
import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from midi_cleaner.midi.importer import pitch_name_from_midi

DEFAULT_TEMPO_US_PER_BEAT = 500_000
DEFAULT_TIME_SIGNATURE_NUMERATOR = 4
DEFAULT_TIME_SIGNATURE_DENOMINATOR = 4
DEFAULT_HISTORY_LIMIT = 100


def _as_int(value: object, default: int = 0) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except Exception:
        return default
    return parsed


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except Exception:
        return default
    return parsed


def note_to_payload(note: object) -> dict[str, Any]:
    if isinstance(note, Mapping):
        return copy.deepcopy(dict(note))
    if hasattr(note, "model_dump"):
        return copy.deepcopy(getattr(note, "model_dump")())
    if hasattr(note, "__dict__"):
        return copy.deepcopy(dict(getattr(note, "__dict__")))
    raise TypeError(f"Unsupported note payload type: {type(note)!r}")


def clone_note_payload(note: object) -> dict[str, Any]:
    return note_to_payload(note)


def sort_note_payloads(notes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payloads = [note_to_payload(note) for note in notes]
    payloads.sort(
        key=lambda note: (
            _as_int(note.get("start_tick"), 0),
            _as_int(note.get("end_tick"), 0),
            _as_int(note.get("editable_track_index"), 0),
            _as_int(note.get("channel"), -1),
            _as_int(note.get("pitch_midi"), 0),
            str(note.get("note_id", "")),
        )
    )
    return payloads


def _note_id(note: Mapping[str, Any]) -> str:
    return str(note.get("note_id", "")).strip()


def _clamp_pitch_midi(value: int) -> int:
    return max(0, min(127, int(value)))


def _normalize_note_timing_and_pitch(
    note: dict[str, Any],
    *,
    ticks_per_beat: int,
    tempo_events: Sequence[Mapping[str, Any]] | None,
    min_duration_ticks: int = 0,
) -> None:
    min_duration = max(0, int(min_duration_ticks))
    start_tick = max(0, _as_int(note.get("start_tick"), 0))
    end_tick = _as_int(note.get("end_tick"), start_tick)
    end_tick = max(start_tick + min_duration, end_tick)

    pitch_midi = _clamp_pitch_midi(_as_int(note.get("pitch_midi"), 0))

    note["start_tick"] = int(start_tick)
    note["end_tick"] = int(end_tick)
    note["duration_ticks"] = int(max(0, end_tick - start_tick))
    note["pitch_midi"] = int(pitch_midi)
    note["pitch_name"] = pitch_name_from_midi(pitch_midi)
    note["start_sec"] = float(
        tick_to_seconds(start_tick, tempo_events, ticks_per_beat=ticks_per_beat)
    )
    note["end_sec"] = float(
        tick_to_seconds(end_tick, tempo_events, ticks_per_beat=ticks_per_beat)
    )
    note["duration_sec"] = float(max(0.0, note["end_sec"] - note["start_sec"]))


def _replacement_notes(
    *,
    payloads: Sequence[Mapping[str, Any]],
    after_notes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    after_by_id = {_note_id(note): note_to_payload(note) for note in after_notes}

    merged: list[dict[str, Any]] = []
    for note in payloads:
        note_id = _note_id(note)
        if note_id in after_by_id:
            merged.append(clone_note_payload(after_by_id[note_id]))
        else:
            merged.append(clone_note_payload(note))

    replaced_ids = {_note_id(note) for note in payloads}
    for note in after_notes:
        note_id = _note_id(note)
        if note_id not in replaced_ids:
            merged.append(clone_note_payload(note))

    return sort_note_payloads(merged)


def move_selected_notes(
    *,
    all_notes: Sequence[object],
    selected_note_ids: Sequence[str],
    delta_ticks: int,
    delta_pitch: int,
    ticks_per_beat: int,
    tempo_events: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    payloads = [note_to_payload(note) for note in all_notes]
    selected_set = {str(note_id) for note_id in selected_note_ids}
    selected_notes = [note for note in payloads if _note_id(note) in selected_set]
    before_notes = sort_note_payloads(selected_notes)

    if not before_notes:
        return {
            "before_notes": [],
            "after_notes": [],
            "notes_after": sort_note_payloads(payloads),
            "selection_after": [],
            "moved_count": 0,
            "applied_delta_ticks": 0,
            "applied_delta_pitch": 0,
        }

    requested_delta_ticks = _as_int(delta_ticks, 0)
    min_start_tick = min(_as_int(note.get("start_tick"), 0) for note in before_notes)
    applied_delta_ticks = requested_delta_ticks
    if min_start_tick + applied_delta_ticks < 0:
        applied_delta_ticks = -min_start_tick

    requested_delta_pitch = _as_int(delta_pitch, 0)
    min_pitch = min(_as_int(note.get("pitch_midi"), 0) for note in before_notes)
    max_pitch = max(_as_int(note.get("pitch_midi"), 0) for note in before_notes)
    applied_delta_pitch = requested_delta_pitch
    if min_pitch + applied_delta_pitch < 0:
        applied_delta_pitch = -min_pitch
    if max_pitch + applied_delta_pitch > 127:
        applied_delta_pitch = 127 - max_pitch

    after_notes: list[dict[str, Any]] = []
    for note in before_notes:
        updated = clone_note_payload(note)

        start_tick = _as_int(note.get("start_tick"), 0) + applied_delta_ticks
        end_tick = _as_int(note.get("end_tick"), start_tick) + applied_delta_ticks
        duration_ticks = max(0, _as_int(note.get("end_tick"), start_tick) - _as_int(note.get("start_tick"), 0))
        updated["start_tick"] = int(max(0, start_tick))
        updated["end_tick"] = int(max(updated["start_tick"], updated["start_tick"] + duration_ticks))
        updated["pitch_midi"] = int(_as_int(note.get("pitch_midi"), 0) + applied_delta_pitch)
        _normalize_note_timing_and_pitch(
            updated,
            ticks_per_beat=ticks_per_beat,
            tempo_events=tempo_events,
        )
        after_notes.append(updated)

    notes_after = _replacement_notes(payloads=payloads, after_notes=after_notes)

    selection_after = [str(note["note_id"]) for note in before_notes]
    return {
        "before_notes": before_notes,
        "after_notes": sort_note_payloads(after_notes),
        "notes_after": notes_after,
        "selection_after": selection_after,
        "moved_count": len(before_notes),
        "applied_delta_ticks": int(applied_delta_ticks),
        "applied_delta_pitch": int(applied_delta_pitch),
    }


def resize_note_edge(
    *,
    all_notes: Sequence[object],
    note_id: str,
    edge: str,
    target_tick: int,
    ticks_per_beat: int,
    tempo_events: Sequence[Mapping[str, Any]] | None,
    min_duration_ticks: int = 1,
) -> dict[str, Any]:
    if edge not in {"left", "right"}:
        raise ValueError("edge must be 'left' or 'right'.")

    payloads = [note_to_payload(note) for note in all_notes]
    note_map = {_note_id(note): note for note in payloads}
    normalized_note_id = str(note_id)
    if normalized_note_id not in note_map:
        raise ValueError(f"Unknown note_id: {normalized_note_id}")

    before_note = clone_note_payload(note_map[normalized_note_id])
    updated_note = clone_note_payload(before_note)

    min_duration = max(1, _as_int(min_duration_ticks, 1))
    requested_tick = max(0, _as_int(target_tick, 0))
    start_tick = _as_int(before_note.get("start_tick"), 0)
    end_tick = _as_int(before_note.get("end_tick"), start_tick)

    if edge == "left":
        max_start_tick = max(0, end_tick - min_duration)
        updated_note["start_tick"] = int(min(requested_tick, max_start_tick))
        updated_note["end_tick"] = int(end_tick)
    else:
        min_end_tick = start_tick + min_duration
        updated_note["start_tick"] = int(start_tick)
        updated_note["end_tick"] = int(max(requested_tick, min_end_tick))

    _normalize_note_timing_and_pitch(
        updated_note,
        ticks_per_beat=ticks_per_beat,
        tempo_events=tempo_events,
        min_duration_ticks=min_duration,
    )

    notes_after = _replacement_notes(payloads=payloads, after_notes=[updated_note])

    return {
        "before_note": before_note,
        "after_note": clone_note_payload(updated_note),
        "before_notes": [before_note],
        "after_notes": [clone_note_payload(updated_note)],
        "notes_after": notes_after,
        "selection_after": [normalized_note_id],
    }


def _generate_unique_draw_note_id(existing_note_ids: set[str], note_id_prefix: str) -> str:
    prefix = note_id_prefix.strip() or "drawn"
    ordinal = 1
    while True:
        candidate = f"{prefix}_{ordinal:06d}"
        if candidate not in existing_note_ids:
            return candidate
        ordinal += 1


def draw_note(
    *,
    all_notes: Sequence[object],
    start_tick: int,
    end_tick: int,
    pitch_midi: int,
    editable_track_index: int,
    editable_track_name: str,
    ticks_per_beat: int,
    tempo_events: Sequence[Mapping[str, Any]] | None,
    channel: int = 0,
    velocity: int = 100,
    source_track_index: int | None = None,
    source_track_name: str | None = None,
    muted: bool = False,
    metadata: Mapping[str, Any] | None = None,
    min_duration_ticks: int = 1,
    note_id_prefix: str = "drawn",
) -> dict[str, Any]:
    payloads = [note_to_payload(note) for note in all_notes]
    existing_note_ids = {_note_id(note) for note in payloads}
    generated_note_id = _generate_unique_draw_note_id(existing_note_ids, note_id_prefix)

    min_duration = max(1, _as_int(min_duration_ticks, 1))
    normalized_start = max(0, _as_int(start_tick, 0))
    normalized_end = max(0, _as_int(end_tick, normalized_start))
    if normalized_end < normalized_start:
        normalized_start, normalized_end = normalized_end, normalized_start
    if normalized_end <= normalized_start:
        normalized_end = normalized_start + min_duration

    note_payload: dict[str, Any] = {
        "note_id": generated_note_id,
        "source_track_index": int(
            _as_int(source_track_index, _as_int(editable_track_index, 0))
            if source_track_index is not None
            else _as_int(editable_track_index, 0)
        ),
        "source_track_name": source_track_name,
        "editable_track_index": int(_as_int(editable_track_index, 0)),
        "editable_track_name": str(editable_track_name),
        "channel": int(_as_int(channel, 0)),
        "pitch_midi": int(_clamp_pitch_midi(_as_int(pitch_midi, 0))),
        "pitch_name": pitch_name_from_midi(_clamp_pitch_midi(_as_int(pitch_midi, 0))),
        "velocity": int(max(1, min(127, _as_int(velocity, 100)))),
        "start_tick": int(normalized_start),
        "end_tick": int(normalized_end),
        "duration_ticks": int(max(0, normalized_end - normalized_start)),
        "start_sec": 0.0,
        "end_sec": 0.0,
        "duration_sec": 0.0,
        "muted": bool(muted),
        "metadata": dict(metadata or {}),
    }

    _normalize_note_timing_and_pitch(
        note_payload,
        ticks_per_beat=ticks_per_beat,
        tempo_events=tempo_events,
        min_duration_ticks=min_duration,
    )

    notes_after = sort_note_payloads([*payloads, note_payload])

    return {
        "drawn_note": clone_note_payload(note_payload),
        "before_notes": [],
        "after_notes": [clone_note_payload(note_payload)],
        "notes_after": notes_after,
        "selection_after": [generated_note_id],
    }


def validate_merge_note_candidates(selected_notes: Sequence[object]) -> str | None:
    payloads = [note_to_payload(note) for note in selected_notes]
    if len(payloads) < 2:
        return "Select at least two notes to merge."

    first_pitch = _as_int(payloads[0].get("pitch_midi"), -1)
    first_track = _as_int(payloads[0].get("editable_track_index"), -1)
    first_channel = payloads[0].get("channel")

    for note in payloads[1:]:
        if _as_int(note.get("pitch_midi"), -1) != first_pitch:
            return "Selected notes must have the same pitch."
        if _as_int(note.get("editable_track_index"), -1) != first_track:
            return "Selected notes must belong to the same track."
        if note.get("channel") != first_channel:
            return "Selected notes must use the same MIDI channel."

    return None


def generate_unique_merged_note_id(existing_note_ids: set[str], base_note_id: str) -> str:
    token = base_note_id.strip() or "note"
    candidate = f"{token}_merged"
    ordinal = 1
    while candidate in existing_note_ids:
        candidate = f"{token}_merged_{ordinal:03d}"
        ordinal += 1
    return candidate


def normalize_tempo_map(
    tempo_events: Sequence[Mapping[str, Any]] | None,
    *,
    ticks_per_beat: int,
) -> list[dict[str, float | int]]:
    sanitized_ticks_per_beat = max(1, int(ticks_per_beat))
    tick_to_tempo: dict[int, int] = {}

    for raw_event in list(tempo_events or []):
        tick = max(0, _as_int(raw_event.get("tick"), 0))
        tempo = _as_int(raw_event.get("tempo_us_per_beat"), DEFAULT_TEMPO_US_PER_BEAT)
        if tempo <= 0:
            tempo = DEFAULT_TEMPO_US_PER_BEAT
        tick_to_tempo[tick] = tempo

    if 0 not in tick_to_tempo:
        tick_to_tempo[0] = DEFAULT_TEMPO_US_PER_BEAT

    sorted_events = sorted(tick_to_tempo.items(), key=lambda item: item[0])
    normalized: list[dict[str, float | int]] = []
    current_sec = 0.0

    for index, (tick, tempo) in enumerate(sorted_events):
        if index > 0:
            prev_tick, prev_tempo = sorted_events[index - 1]
            delta_ticks = tick - prev_tick
            current_sec += (delta_ticks / sanitized_ticks_per_beat) * (prev_tempo / 1_000_000.0)

        normalized.append(
            {
                "tick": int(tick),
                "tempo_us_per_beat": int(tempo),
                "sec": float(current_sec),
            }
        )

    return normalized


def tick_to_seconds(
    tick: float | int,
    tempo_events: Sequence[Mapping[str, Any]] | None,
    *,
    ticks_per_beat: int,
) -> float:
    normalized = normalize_tempo_map(tempo_events, ticks_per_beat=ticks_per_beat)
    target_tick = max(0.0, float(tick))
    ticks = [float(event["tick"]) for event in normalized]

    index = bisect_right(ticks, target_tick) - 1
    if index < 0:
        index = 0

    event = normalized[index]
    event_tick = float(event["tick"])
    event_sec = float(event["sec"])
    tempo = max(1.0, float(event["tempo_us_per_beat"]))

    delta_ticks = target_tick - event_tick
    delta_sec = (delta_ticks / max(1.0, float(ticks_per_beat))) * (tempo / 1_000_000.0)
    return event_sec + delta_sec


def seconds_to_tick(
    seconds: float,
    tempo_events: Sequence[Mapping[str, Any]] | None,
    *,
    ticks_per_beat: int,
) -> float:
    normalized = normalize_tempo_map(tempo_events, ticks_per_beat=ticks_per_beat)
    target_sec = max(0.0, float(seconds))

    secs = [float(event["sec"]) for event in normalized]
    index = bisect_right(secs, target_sec) - 1
    if index < 0:
        index = 0

    event = normalized[index]
    event_tick = float(event["tick"])
    event_sec = float(event["sec"])
    tempo = max(1.0, float(event["tempo_us_per_beat"]))

    ticks_per_second = (max(1.0, float(ticks_per_beat)) * 1_000_000.0) / tempo
    delta_seconds = target_sec - event_sec
    return event_tick + delta_seconds * ticks_per_second


def normalize_time_signature_map(
    time_signatures: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, int]]:
    tick_to_signature: dict[int, tuple[int, int]] = {}

    for raw_event in list(time_signatures or []):
        tick = max(0, _as_int(raw_event.get("tick"), 0))
        numerator = max(1, _as_int(raw_event.get("numerator"), DEFAULT_TIME_SIGNATURE_NUMERATOR))
        denominator = max(1, _as_int(raw_event.get("denominator"), DEFAULT_TIME_SIGNATURE_DENOMINATOR))
        tick_to_signature[tick] = (numerator, denominator)

    if 0 not in tick_to_signature:
        tick_to_signature[0] = (
            DEFAULT_TIME_SIGNATURE_NUMERATOR,
            DEFAULT_TIME_SIGNATURE_DENOMINATOR,
        )

    return [
        {
            "tick": int(tick),
            "numerator": int(signature[0]),
            "denominator": int(signature[1]),
        }
        for tick, signature in sorted(tick_to_signature.items(), key=lambda item: item[0])
    ]


def _ticks_per_signature_beat(ticks_per_beat: int, denominator: int) -> float:
    return max(1.0, float(ticks_per_beat) * 4.0 / float(max(1, denominator)))


def tick_to_bar_beat(
    tick: float | int,
    *,
    ticks_per_beat: int,
    time_signatures: Sequence[Mapping[str, Any]] | None,
) -> dict[str, int | float]:
    signatures = normalize_time_signature_map(time_signatures)
    target_tick = max(0.0, float(tick))

    bar_number = 1
    bar_start_tick = 0.0
    signature_index = 0

    for _ in range(100_000):
        while (
            signature_index + 1 < len(signatures)
            and float(signatures[signature_index + 1]["tick"]) <= bar_start_tick + 1e-9
        ):
            signature_index += 1

        signature = signatures[signature_index]
        numerator = int(signature["numerator"])
        denominator = int(signature["denominator"])
        ticks_per_sig_beat = _ticks_per_signature_beat(ticks_per_beat, denominator)
        ticks_per_bar = ticks_per_sig_beat * float(max(1, numerator))

        next_signature_tick = (
            float(signatures[signature_index + 1]["tick"])
            if signature_index + 1 < len(signatures)
            else math.inf
        )
        bar_end_tick = bar_start_tick + ticks_per_bar
        if next_signature_tick > bar_start_tick and next_signature_tick < bar_end_tick:
            bar_end_tick = next_signature_tick

        if target_tick < bar_end_tick - 1e-9:
            beat_float = (target_tick - bar_start_tick) / ticks_per_sig_beat
            beat_number = max(1, int(math.floor(max(0.0, beat_float))) + 1)
            return {
                "bar": int(bar_number),
                "beat": int(beat_number),
                "bar_start_tick": float(bar_start_tick),
                "numerator": int(numerator),
                "denominator": int(denominator),
            }

        bar_start_tick = bar_end_tick
        bar_number += 1

    raise RuntimeError("Failed to resolve bar/beat position.")


@dataclass(slots=True)
class TimelineMarker:
    kind: str
    tick: float
    x: float
    text: str | None = None


@dataclass(slots=True)
class TimelineLayout:
    bar_lines: list[TimelineMarker] = field(default_factory=list)
    beat_lines: list[TimelineMarker] = field(default_factory=list)
    bar_labels: list[TimelineMarker] = field(default_factory=list)
    beat_labels: list[TimelineMarker] = field(default_factory=list)
    time_labels: list[TimelineMarker] = field(default_factory=list)


def _format_mm_ss(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def _time_label_step_seconds(px_per_second: float, min_spacing_px: float) -> int:
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300):
        if step * px_per_second >= min_spacing_px:
            return int(step)
    return 600


def build_timeline_layout(
    *,
    x_offset_ticks: float,
    pixels_per_tick: float,
    viewport_width_px: float,
    left_pad_px: float,
    ticks_per_beat: int,
    tempo_events: Sequence[Mapping[str, Any]] | None,
    time_signatures: Sequence[Mapping[str, Any]] | None,
    min_bar_label_px: float = 56.0,
    min_beat_label_px: float = 36.0,
    min_time_label_px: float = 72.0,
) -> TimelineLayout:
    safe_pixels_per_tick = max(1e-9, float(pixels_per_tick))
    safe_width = max(1.0, float(viewport_width_px))
    start_tick = max(0.0, float(x_offset_ticks))
    end_tick = start_tick + (safe_width / safe_pixels_per_tick)
    right_edge = left_pad_px + safe_width

    signatures = normalize_time_signature_map(time_signatures)
    layout = TimelineLayout()

    def tick_to_x(tick_value: float) -> float:
        return float(left_pad_px) + (float(tick_value) - start_tick) * safe_pixels_per_tick

    bar_number = 1
    bar_start_tick = 0.0
    signature_index = 0
    last_bar_label_x = -math.inf
    last_beat_label_x = -math.inf

    for _ in range(100_000):
        while (
            signature_index + 1 < len(signatures)
            and float(signatures[signature_index + 1]["tick"]) <= bar_start_tick + 1e-9
        ):
            signature_index += 1

        signature = signatures[signature_index]
        numerator = max(1, int(signature["numerator"]))
        denominator = max(1, int(signature["denominator"]))
        ticks_per_sig_beat = _ticks_per_signature_beat(ticks_per_beat, denominator)
        ticks_per_bar = ticks_per_sig_beat * float(numerator)

        next_signature_tick = (
            float(signatures[signature_index + 1]["tick"])
            if signature_index + 1 < len(signatures)
            else math.inf
        )
        bar_end_tick = bar_start_tick + ticks_per_bar
        if next_signature_tick > bar_start_tick and next_signature_tick < bar_end_tick:
            bar_end_tick = next_signature_tick

        if bar_start_tick > end_tick + ticks_per_bar:
            break

        bar_x = tick_to_x(bar_start_tick)
        if left_pad_px <= bar_x <= right_edge:
            layout.bar_lines.append(TimelineMarker(kind="bar", tick=bar_start_tick, x=bar_x))
            if bar_x - last_bar_label_x >= min_bar_label_px:
                layout.bar_labels.append(
                    TimelineMarker(
                        kind="bar_label",
                        tick=bar_start_tick,
                        x=bar_x,
                        text=str(bar_number),
                    )
                )
                last_bar_label_x = bar_x

        beat_tick = bar_start_tick
        beat_number = 1
        while beat_tick < bar_end_tick - 1e-9:
            if beat_number > 1:
                beat_x = tick_to_x(beat_tick)
                if left_pad_px <= beat_x <= right_edge:
                    layout.beat_lines.append(TimelineMarker(kind="beat", tick=beat_tick, x=beat_x))
                    beat_spacing = ticks_per_sig_beat * safe_pixels_per_tick
                    if (
                        beat_spacing >= min_beat_label_px
                        and beat_x - last_beat_label_x >= min_beat_label_px
                    ):
                        layout.beat_labels.append(
                            TimelineMarker(
                                kind="beat_label",
                                tick=beat_tick,
                                x=beat_x,
                                text=str(beat_number),
                            )
                        )
                        last_beat_label_x = beat_x

            beat_tick += ticks_per_sig_beat
            beat_number += 1

        bar_start_tick = bar_end_tick
        bar_number += 1

    start_sec = tick_to_seconds(start_tick, tempo_events, ticks_per_beat=ticks_per_beat)
    end_sec = tick_to_seconds(end_tick, tempo_events, ticks_per_beat=ticks_per_beat)
    span_seconds = max(1e-6, end_sec - start_sec)
    px_per_second = safe_width / span_seconds
    step_seconds = _time_label_step_seconds(px_per_second, min_time_label_px)

    last_time_label_x = -math.inf
    first_second = math.floor(start_sec / step_seconds) * step_seconds
    current_second = float(first_second)

    for _ in range(5000):
        if current_second > end_sec + step_seconds:
            break
        tick_value = seconds_to_tick(current_second, tempo_events, ticks_per_beat=ticks_per_beat)
        x_value = tick_to_x(tick_value)
        if left_pad_px <= x_value <= right_edge and x_value - last_time_label_x >= min_time_label_px:
            layout.time_labels.append(
                TimelineMarker(
                    kind="time_label",
                    tick=float(tick_value),
                    x=float(x_value),
                    text=_format_mm_ss(current_second),
                )
            )
            last_time_label_x = x_value
        current_second += float(step_seconds)

    return layout


def merge_selected_notes(
    *,
    all_notes: Sequence[object],
    selected_note_ids: Sequence[str],
    ticks_per_beat: int,
    tempo_events: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    payloads = [note_to_payload(note) for note in all_notes]
    selected_set = {str(note_id) for note_id in selected_note_ids}
    selected_notes = [note for note in payloads if _note_id(note) in selected_set]

    message = validate_merge_note_candidates(selected_notes)
    if message is not None:
        raise ValueError(message)

    selected_sorted = sort_note_payloads(selected_notes)
    template = clone_note_payload(selected_sorted[0])

    merged_start_tick = min(_as_int(note.get("start_tick"), 0) for note in selected_sorted)
    merged_end_tick = max(_as_int(note.get("end_tick"), merged_start_tick) for note in selected_sorted)

    existing_ids = {_note_id(note) for note in payloads}
    merged_note_id = generate_unique_merged_note_id(existing_ids, _note_id(template))

    template["note_id"] = merged_note_id
    template["start_tick"] = int(merged_start_tick)
    template["end_tick"] = int(max(merged_start_tick, merged_end_tick))
    template["duration_ticks"] = int(max(0, template["end_tick"] - template["start_tick"]))
    template["start_sec"] = float(
        tick_to_seconds(template["start_tick"], tempo_events, ticks_per_beat=ticks_per_beat)
    )
    template["end_sec"] = float(
        tick_to_seconds(template["end_tick"], tempo_events, ticks_per_beat=ticks_per_beat)
    )
    template["duration_sec"] = float(max(0.0, template["end_sec"] - template["start_sec"]))

    selected_ids = {_note_id(note) for note in selected_sorted}
    remaining = [note for note in payloads if _note_id(note) not in selected_ids]
    notes_after = sort_note_payloads([*remaining, template])

    return {
        "merged_note": clone_note_payload(template),
        "selected_before": selected_sorted,
        "notes_after": notes_after,
    }


def is_note_muted(note: Mapping[str, Any]) -> bool:
    return bool(note.get("muted", False))


def delete_selected_notes(
    *,
    all_notes: Sequence[object],
    selected_note_ids: Sequence[str],
) -> dict[str, Any]:
    payloads = [note_to_payload(note) for note in all_notes]
    selected_set = {str(note_id) for note_id in selected_note_ids}

    deleted_notes = [note for note in payloads if _note_id(note) in selected_set]
    deleted_sorted = sort_note_payloads(deleted_notes)

    remaining_notes = [note for note in payloads if _note_id(note) not in selected_set]
    notes_after = sort_note_payloads(remaining_notes)

    return {
        "deleted_notes": deleted_sorted,
        "notes_after": notes_after,
        "selection_after": [],
    }


def resolve_selected_mute_action(selected_notes: Sequence[object]) -> str | None:
    payloads = [note_to_payload(note) for note in selected_notes]
    if not payloads:
        return None

    all_muted = all(is_note_muted(note) for note in payloads)
    return "unmute" if all_muted else "mute"


def set_selected_notes_muted(
    *,
    all_notes: Sequence[object],
    selected_note_ids: Sequence[str],
    mute: bool | None = None,
) -> dict[str, Any]:
    payloads = [note_to_payload(note) for note in all_notes]
    selected_set = {str(note_id) for note_id in selected_note_ids}
    selected_notes = [note for note in payloads if _note_id(note) in selected_set]

    resolved_action = "mute" if bool(mute) else "unmute"
    if mute is None:
        dynamic_action = resolve_selected_mute_action(selected_notes)
        if dynamic_action is None:
            resolved_action = "mute"
        else:
            resolved_action = dynamic_action

    should_mute = resolved_action == "mute"
    before_notes = sort_note_payloads(selected_notes)

    after_notes: list[dict[str, Any]] = []
    changed_count = 0
    for note in before_notes:
        updated = clone_note_payload(note)
        prior_muted = is_note_muted(note)
        updated["muted"] = should_mute
        if prior_muted != should_mute:
            changed_count += 1
        after_notes.append(updated)

    by_after_id = {_note_id(note): note for note in after_notes}
    notes_after: list[dict[str, Any]] = []
    for note in payloads:
        note_id = _note_id(note)
        if note_id in by_after_id:
            notes_after.append(clone_note_payload(by_after_id[note_id]))
        else:
            notes_after.append(clone_note_payload(note))

    return {
        "action": resolved_action,
        "before_notes": before_notes,
        "after_notes": sort_note_payloads(after_notes),
        "notes_after": sort_note_payloads(notes_after),
        "changed_count": int(changed_count),
    }


@dataclass(slots=True)
class NoteEditTransaction:
    before_notes: list[dict[str, Any]]
    after_notes: list[dict[str, Any]]
    selection_before: list[str] = field(default_factory=list)
    selection_after: list[str] = field(default_factory=list)
    label: str = ""


def apply_note_transaction(
    notes: Sequence[object],
    transaction: NoteEditTransaction,
    *,
    use_after: bool,
) -> list[dict[str, Any]]:
    payloads = [note_to_payload(note) for note in notes]
    replacement = transaction.after_notes if use_after else transaction.before_notes

    affected_ids = {
        _note_id(note)
        for note in [*transaction.before_notes, *transaction.after_notes]
        if _note_id(note)
    }

    kept = [note for note in payloads if _note_id(note) not in affected_ids]
    merged = [*kept, *[clone_note_payload(note) for note in replacement]]
    return sort_note_payloads(merged)


@dataclass(slots=True)
class NoteHistory:
    limit: int = DEFAULT_HISTORY_LIMIT
    undo_stack: list[NoteEditTransaction] = field(default_factory=list)
    redo_stack: list[NoteEditTransaction] = field(default_factory=list)

    def push(self, transaction: NoteEditTransaction) -> None:
        self.undo_stack.append(transaction)
        if len(self.undo_stack) > max(1, int(self.limit)):
            overflow = len(self.undo_stack) - max(1, int(self.limit))
            del self.undo_stack[:overflow]
        self.redo_stack.clear()

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()

    def undo(self, notes: Sequence[object]) -> tuple[list[dict[str, Any]], list[str]]:
        if not self.undo_stack:
            return [note_to_payload(note) for note in notes], []

        transaction = self.undo_stack.pop()
        self.redo_stack.append(transaction)
        updated_notes = apply_note_transaction(notes, transaction, use_after=False)
        return updated_notes, list(transaction.selection_before)

    def redo(self, notes: Sequence[object]) -> tuple[list[dict[str, Any]], list[str]]:
        if not self.redo_stack:
            return [note_to_payload(note) for note in notes], []

        transaction = self.redo_stack.pop()
        self.undo_stack.append(transaction)
        updated_notes = apply_note_transaction(notes, transaction, use_after=True)
        return updated_notes, list(transaction.selection_after)
