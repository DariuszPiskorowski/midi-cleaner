from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class DrumMapError(ValueError):
    """Raised when a drum map preset or custom map is invalid."""


@dataclass(frozen=True)
class DrumMapDefinition:
    name: str
    output_channel: int
    notes: dict[int, int]
    labels: dict[int, str] = field(default_factory=dict)


_GM_MAP_NOTES: dict[int, int] = {
    32: 36,
    35: 36,
    36: 36,
    37: 38,
    38: 38,
    39: 38,
    40: 38,
    41: 41,
    43: 43,
    45: 45,
    47: 47,
    48: 48,
    50: 50,
    42: 42,
    44: 44,
    46: 46,
    49: 49,
    51: 49,
    52: 49,
    55: 49,
    57: 49,
    59: 49,
}

_GM_MAP_LABELS: dict[int, str] = {
    32: "kick_or_artifact",
    35: "kick",
    36: "kick",
    38: "snare",
    39: "clap_to_snare",
    42: "closed_hat",
    45: "low_tom",
    46: "open_hat",
    49: "crash",
    50: "high_tom",
    57: "crash_2",
}

# These presets are intentionally compact and editable for real-world kit adjustments.
PRESET_DRUM_MAPS: dict[str, DrumMapDefinition] = {
    "gm": DrumMapDefinition(
        name="gm",
        output_channel=9,
        notes=_GM_MAP_NOTES,
        labels=_GM_MAP_LABELS,
    ),
    "sitala": DrumMapDefinition(
        name="sitala",
        output_channel=9,
        notes=_GM_MAP_NOTES,
        labels=_GM_MAP_LABELS,
    ),
    "ujam-candy": DrumMapDefinition(
        name="ujam_candy",
        output_channel=9,
        notes={
            32: 36,
            36: 36,
            39: 38,
            45: 45,
            46: 46,
            50: 50,
            57: 49,
        },
        labels={
            36: "kick",
            39: "clap_or_snare",
            46: "open_hat",
            57: "crash",
            45: "low_tom",
            50: "high_tom",
            32: "low_kick_or_artifact",
        },
    ),
}


def _coerce_note_value(value: object, *, field_name: str) -> int:
    try:
        note = int(value)
    except (TypeError, ValueError) as exc:
        raise DrumMapError(f"Invalid {field_name}: {value}") from exc

    if note < 0 or note > 127:
        raise DrumMapError(f"{field_name} must be in range 0..127, got {note}")

    return note


def _coerce_output_channel(value: object) -> int:
    try:
        channel = int(value)
    except (TypeError, ValueError) as exc:
        raise DrumMapError(f"Invalid output_channel: {value}") from exc

    if channel < 0 or channel > 15:
        raise DrumMapError(f"output_channel must be in range 0..15, got {channel}")

    return channel


def _coerce_notes(raw_notes: object) -> dict[int, int]:
    if not isinstance(raw_notes, dict):
        raise DrumMapError("Custom map 'notes' must be an object.")

    notes: dict[int, int] = {}
    for source_note, target_note in raw_notes.items():
        source = _coerce_note_value(source_note, field_name="source note")
        target = _coerce_note_value(target_note, field_name="target note")
        notes[source] = target

    return notes


def _coerce_labels(raw_labels: object) -> dict[int, str]:
    if raw_labels is None:
        return {}

    if not isinstance(raw_labels, dict):
        raise DrumMapError("Custom map 'labels' must be an object when provided.")

    labels: dict[int, str] = {}
    for source_note, label in raw_labels.items():
        source = _coerce_note_value(source_note, field_name="label note")
        labels[source] = str(label)

    return labels


def _clone_map(map_definition: DrumMapDefinition) -> DrumMapDefinition:
    return DrumMapDefinition(
        name=map_definition.name,
        output_channel=map_definition.output_channel,
        notes=dict(map_definition.notes),
        labels=dict(map_definition.labels),
    )


def load_preset_drum_map(name: str) -> DrumMapDefinition:
    normalized = name.strip().lower()
    if normalized not in PRESET_DRUM_MAPS:
        allowed = ", ".join(sorted(PRESET_DRUM_MAPS))
        raise DrumMapError(f"Unknown target map: {name}. Available presets: {allowed}")
    return _clone_map(PRESET_DRUM_MAPS[normalized])


def load_custom_drum_map(map_file: Path) -> DrumMapDefinition:
    if not map_file.exists() or not map_file.is_file():
        raise DrumMapError(f"Custom map file does not exist: {map_file}")

    try:
        payload = json.loads(map_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DrumMapError(f"Failed to parse custom map JSON: {map_file}") from exc

    if not isinstance(payload, dict):
        raise DrumMapError("Custom map JSON must be an object.")

    if "name" not in payload:
        raise DrumMapError("Custom map JSON must include 'name'.")
    if "output_channel" not in payload:
        raise DrumMapError("Custom map JSON must include 'output_channel'.")
    if "notes" not in payload:
        raise DrumMapError("Custom map JSON must include 'notes'.")

    name = str(payload["name"])
    output_channel = _coerce_output_channel(payload["output_channel"])
    notes = _coerce_notes(payload["notes"])
    labels = _coerce_labels(payload.get("labels"))

    return DrumMapDefinition(
        name=name,
        output_channel=output_channel,
        notes=notes,
        labels=labels,
    )