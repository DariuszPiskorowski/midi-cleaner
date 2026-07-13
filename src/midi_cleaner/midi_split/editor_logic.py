from __future__ import annotations

import copy
import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

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
