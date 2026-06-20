from __future__ import annotations

import json
import re
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
    key_layout_name: str = ""
    target_note_names: dict[int, str] = field(default_factory=dict)


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

_NATURAL_SEMITONE_BY_NOTE_NAME: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_UJAM_CANDY_LAYOUT_LABELS: dict[str, str] = {
    "C1": "Kick",
    "D1": "Acc",
    "E1": "Sn1",
    "F1": "Sn2",
    "G1": "Clap",
    "A1": "HH1",
    "B1": "HH2",
    "C2": "HH0",
    "D2": "Tom L",
    "E2": "Tom M",
    "F2": "Tom H",
    "G2": "Perc 1",
    "A2": "Perc 2",
    "B2": "Perc 3",
    "C3": "Cym 1",
    "D3": "Cym 2",
}

_UJAM_CANDY_SOURCE_TO_LAYOUT_NOTE: dict[int, str] = {
    32: "C1",
    36: "C1",
    39: "G1",
    45: "D2",
    46: "C2",
    50: "F2",
    57: "C3",
}

_UJAM_CANDY_LAYOUT_NAME = "ujam-candy-observed-ui"

# These presets are intentionally compact and editable for real-world kit adjustments.
PRESET_DRUM_MAPS: dict[str, DrumMapDefinition] = {
    "gm": DrumMapDefinition(
        name="gm",
        output_channel=9,
        notes=_GM_MAP_NOTES,
        labels=_GM_MAP_LABELS,
        key_layout_name="gm",
    ),
    "sitala": DrumMapDefinition(
        name="sitala",
        output_channel=9,
        notes=_GM_MAP_NOTES,
        labels=_GM_MAP_LABELS,
        key_layout_name="sitala-gm-compatible",
    ),
}


def _note_name_to_midi(note_name: str, *, c1_midi_note: int) -> int:
    matched = re.fullmatch(r"([A-G])([0-9]+)", note_name)
    if matched is None:
        raise DrumMapError(f"Invalid note name in key layout: {note_name}")

    note_token = matched.group(1)
    octave = int(matched.group(2))

    semitone = _NATURAL_SEMITONE_BY_NOTE_NAME[note_token]
    midi_note = c1_midi_note + ((octave - 1) * 12) + semitone
    if midi_note < 0 or midi_note > 127:
        raise DrumMapError(
            f"Resolved note {note_name} from c1_midi_note={c1_midi_note} is out of MIDI range."
        )

    return midi_note


def _build_ujam_candy_map(c1_midi_note: int) -> DrumMapDefinition:
    if c1_midi_note < 0 or c1_midi_note > 127:
        raise DrumMapError(f"c1_midi_note must be in range 0..127, got {c1_midi_note}")

    notes: dict[int, int] = {}
    labels: dict[int, str] = {}
    target_note_names: dict[int, str] = {}

    for source_note, target_layout_note in _UJAM_CANDY_SOURCE_TO_LAYOUT_NOTE.items():
        target_midi_note = _note_name_to_midi(target_layout_note, c1_midi_note=c1_midi_note)
        notes[source_note] = target_midi_note
        labels[source_note] = _UJAM_CANDY_LAYOUT_LABELS[target_layout_note]
        target_note_names[source_note] = target_layout_note

    return DrumMapDefinition(
        name="ujam-candy",
        output_channel=9,
        notes=notes,
        labels=labels,
        key_layout_name=_UJAM_CANDY_LAYOUT_NAME,
        target_note_names=target_note_names,
    )


def resolve_ujam_candy_layout_notes(c1_midi_note: int) -> dict[str, int]:
    resolved: dict[str, int] = {}
    for note_name in sorted(_UJAM_CANDY_LAYOUT_LABELS, key=lambda item: (_note_name_to_midi(item, c1_midi_note=c1_midi_note), item)):
        resolved[note_name] = _note_name_to_midi(note_name, c1_midi_note=c1_midi_note)
    return resolved


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
        key_layout_name=map_definition.key_layout_name,
        target_note_names=dict(map_definition.target_note_names),
    )


def load_preset_drum_map(name: str, *, c1_midi_note: int = 36) -> DrumMapDefinition:
    normalized = name.strip().lower()
    if normalized == "ujam-candy":
        return _build_ujam_candy_map(c1_midi_note=c1_midi_note)

    if normalized not in PRESET_DRUM_MAPS:
        allowed = ", ".join(sorted(PRESET_DRUM_MAPS))
        allowed = f"{allowed}, ujam-candy"
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
    key_layout_name = str(payload.get("key_layout_name", "custom"))
    raw_target_note_names = payload.get("target_note_names")
    target_note_names: dict[int, str] = {}
    if raw_target_note_names is not None:
        if not isinstance(raw_target_note_names, dict):
            raise DrumMapError("Custom map 'target_note_names' must be an object when provided.")
        for source_note, note_name in raw_target_note_names.items():
            source = _coerce_note_value(source_note, field_name="target_note_names note")
            target_note_names[source] = str(note_name)

    return DrumMapDefinition(
        name=name,
        output_channel=output_channel,
        notes=notes,
        labels=labels,
        key_layout_name=key_layout_name,
        target_note_names=target_note_names,
    )