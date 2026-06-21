from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


class DrumLayerMappingError(ValueError):
    """Raised when drum layer mapping JSON is invalid."""


@dataclass(frozen=True)
class SemanticLayerSlot:
    enabled: bool
    note: int
    note_name: str
    track_name: str


@dataclass(frozen=True)
class DrumLayerMapping:
    name: str
    c1_midi_note: int
    layers: dict[str, SemanticLayerSlot]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "c1_midi_note": self.c1_midi_note,
            "layers": {
                layer_name: {
                    "enabled": slot.enabled,
                    "note_name": slot.note_name,
                    "note": slot.note,
                    "track_name": slot.track_name,
                }
                for layer_name, slot in self.layers.items()
            },
        }


_NATURAL_SEMITONES: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_SHARP_NAMES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

_DEFAULT_SLOT_ORDER: tuple[str, ...] = (
    "kick_1",
    "kick_2",
    "kick_3",
    "snare_1",
    "snare_2",
    "snare_3",
    "clap_1",
    "clap_2",
    "clap_3",
    "hh_1",
    "hh_2",
    "hh_3",
    "hh_open_1",
    "hh_open_2",
    "hh_open_3",
    "tom_l_1",
    "tom_l_2",
    "tom_l_3",
    "tom_m_1",
    "tom_m_2",
    "tom_m_3",
    "tom_h_1",
    "tom_h_2",
    "tom_h_3",
    "perc_1",
    "perc_2",
    "perc_3",
    "cym_1",
    "cym_2",
    "cym_3",
    "fx_1",
    "fx_2",
    "fx_3",
    "acc_1",
    "acc_2",
    "acc_3",
)

# Defaults intentionally keep primary slots aligned with existing extractor behavior.
_DEFAULT_NOTES_BY_TARGET_MAP: dict[str, dict[str, int]] = {
    "gm": {
        "kick_1": 36,
        "kick_2": 38,
        "kick_3": 40,
        "snare_1": 38,
        "snare_2": 40,
        "snare_3": 41,
        "clap_1": 39,
        "clap_2": 41,
        "clap_3": 43,
        "hh_1": 42,
        "hh_2": 44,
        "hh_3": 46,
        "hh_open_1": 46,
        "hh_open_2": 48,
        "hh_open_3": 50,
        "tom_l_1": 45,
        "tom_l_2": 47,
        "tom_l_3": 48,
        "tom_m_1": 47,
        "tom_m_2": 48,
        "tom_m_3": 50,
        "tom_h_1": 50,
        "tom_h_2": 52,
        "tom_h_3": 53,
        "perc_1": 51,
        "perc_2": 53,
        "perc_3": 55,
        "cym_1": 49,
        "cym_2": 51,
        "cym_3": 53,
        "fx_1": 47,
        "fx_2": 48,
        "fx_3": 50,
        "acc_1": 38,
        "acc_2": 40,
        "acc_3": 41,
    },
    "sitala": {
        "kick_1": 36,
        "kick_2": 38,
        "kick_3": 40,
        "snare_1": 38,
        "snare_2": 40,
        "snare_3": 41,
        "clap_1": 39,
        "clap_2": 41,
        "clap_3": 43,
        "hh_1": 42,
        "hh_2": 44,
        "hh_3": 46,
        "hh_open_1": 46,
        "hh_open_2": 48,
        "hh_open_3": 50,
        "tom_l_1": 45,
        "tom_l_2": 47,
        "tom_l_3": 48,
        "tom_m_1": 47,
        "tom_m_2": 48,
        "tom_m_3": 50,
        "tom_h_1": 50,
        "tom_h_2": 52,
        "tom_h_3": 53,
        "perc_1": 51,
        "perc_2": 53,
        "perc_3": 55,
        "cym_1": 49,
        "cym_2": 51,
        "cym_3": 53,
        "fx_1": 47,
        "fx_2": 48,
        "fx_3": 50,
        "acc_1": 38,
        "acc_2": 40,
        "acc_3": 41,
    },
    "ujam-candy": {
        "kick_1": 36,
        "kick_2": 38,
        "kick_3": 40,
        "snare_1": 40,
        "snare_2": 41,
        "snare_3": 43,
        "clap_1": 43,
        "clap_2": 45,
        "clap_3": 47,
        "hh_1": 48,
        "hh_2": 49,
        "hh_3": 50,
        "hh_open_1": 48,
        "hh_open_2": 50,
        "hh_open_3": 52,
        "tom_l_1": 50,
        "tom_l_2": 52,
        "tom_l_3": 53,
        "tom_m_1": 52,
        "tom_m_2": 53,
        "tom_m_3": 55,
        "tom_h_1": 53,
        "tom_h_2": 55,
        "tom_h_3": 57,
        "perc_1": 55,
        "perc_2": 57,
        "perc_3": 59,
        "cym_1": 60,
        "cym_2": 62,
        "cym_3": 64,
        "fx_1": 47,
        "fx_2": 48,
        "fx_3": 50,
        "acc_1": 38,
        "acc_2": 40,
        "acc_3": 41,
    },
}

_ENABLED_DEFAULT_SLOTS: set[str] = {
    "kick_1",
    "snare_1",
    "clap_1",
    "hh_1",
    "hh_open_1",
    "tom_l_1",
    "tom_m_1",
    "tom_h_1",
    "perc_1",
    "cym_1",
}


def note_name_to_midi(note_name: str, *, c1_midi_note: int) -> int:
    if c1_midi_note < 0 or c1_midi_note > 127:
        raise DrumLayerMappingError(f"c1_midi_note must be in range 0..127, got {c1_midi_note}")

    matched = re.fullmatch(r"\s*([A-Ga-g])([#b]?)(-?\d+)\s*", note_name)
    if matched is None:
        raise DrumLayerMappingError(f"Invalid note_name: {note_name}")

    letter = matched.group(1).upper()
    accidental = matched.group(2)
    octave = int(matched.group(3))

    semitone = _NATURAL_SEMITONES[letter]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1

    midi_note = int(c1_midi_note + ((octave - 1) * 12) + semitone)
    if midi_note < 0 or midi_note > 127:
        raise DrumLayerMappingError(
            f"Resolved note {note_name} with c1_midi_note={c1_midi_note} is out of range 0..127"
        )

    return midi_note


def midi_to_note_name(note: int, *, c1_midi_note: int) -> str:
    if note < 0 or note > 127:
        raise DrumLayerMappingError(f"MIDI note must be in range 0..127, got {note}")
    if c1_midi_note < 0 or c1_midi_note > 127:
        raise DrumLayerMappingError(f"c1_midi_note must be in range 0..127, got {c1_midi_note}")

    semitone_offset = note - c1_midi_note
    octave = 1 + (semitone_offset // 12)
    semitone = semitone_offset % 12
    return f"{_SHARP_NAMES[semitone]}{octave}"


def default_layer_order() -> list[str]:
    return list(_DEFAULT_SLOT_ORDER)


def _default_track_name(slot_name: str) -> str:
    family, _, variant = slot_name.rpartition("_")
    if not family:
        return slot_name

    token = family
    replacements = {
        "kick": "Kick",
        "snare": "Snare",
        "clap": "Clap",
        "hh": "HH",
        "hh_open": "HHOpen",
        "tom_l": "TomL",
        "tom_m": "TomM",
        "tom_h": "TomH",
        "perc": "Perc",
        "cym": "Cym",
        "fx": "FX",
        "acc": "Acc",
    }
    head = replacements.get(token, token.title().replace("_", ""))
    return f"{head}{variant}"


def _coerce_note_value(raw_value: object, *, field_name: str) -> int:
    try:
        note = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise DrumLayerMappingError(f"{field_name} must be an integer MIDI note") from exc

    if note < 0 or note > 127:
        raise DrumLayerMappingError(f"{field_name} must be in range 0..127, got {note}")

    return note


def _parse_layer_slot(
    layer_name: str,
    raw_slot: object,
    *,
    c1_midi_note: int,
) -> SemanticLayerSlot:
    if not isinstance(raw_slot, dict):
        raise DrumLayerMappingError(f"Layer '{layer_name}' must be an object")

    enabled = bool(raw_slot.get("enabled", True))
    raw_note_name = raw_slot.get("note_name")
    raw_note = raw_slot.get("note")
    track_name = str(raw_slot.get("track_name", _default_track_name(layer_name))).strip()
    if not track_name:
        raise DrumLayerMappingError(f"Layer '{layer_name}' has an empty track_name")

    parsed_note: int | None = None
    parsed_note_name: str | None = None

    if raw_note_name is not None:
        parsed_note_name = str(raw_note_name).strip()
        if not parsed_note_name:
            raise DrumLayerMappingError(f"Layer '{layer_name}' has an empty note_name")
        parsed_note = note_name_to_midi(parsed_note_name, c1_midi_note=c1_midi_note)

    if raw_note is not None:
        note_value = _coerce_note_value(raw_note, field_name=f"Layer '{layer_name}' note")
        if parsed_note is not None and note_value != parsed_note:
            expected = midi_to_note_name(note_value, c1_midi_note=c1_midi_note)
            raise DrumLayerMappingError(
                f"Layer '{layer_name}' note_name and note do not match for c1_midi_note={c1_midi_note} "
                f"(note_name -> {parsed_note}, note -> {note_value}/{expected})"
            )
        parsed_note = note_value

    if parsed_note is None:
        raise DrumLayerMappingError(
            f"Layer '{layer_name}' must provide note_name, note, or both"
        )

    if parsed_note_name is None:
        parsed_note_name = midi_to_note_name(parsed_note, c1_midi_note=c1_midi_note)

    normalized_note_name = midi_to_note_name(parsed_note, c1_midi_note=c1_midi_note)

    return SemanticLayerSlot(
        enabled=enabled,
        note=parsed_note,
        note_name=normalized_note_name,
        track_name=track_name,
    )


def build_default_layer_mapping(
    *,
    target_map: str,
    c1_midi_note: int,
    name: str = "expanded_default_drum_mapping",
) -> DrumLayerMapping:
    normalized_target = target_map.strip().lower()
    template = _DEFAULT_NOTES_BY_TARGET_MAP.get(normalized_target)
    if template is None:
        template = _DEFAULT_NOTES_BY_TARGET_MAP["gm"]

    layers: dict[str, SemanticLayerSlot] = {}
    for slot_name in _DEFAULT_SLOT_ORDER:
        note = template[slot_name]
        note_name = midi_to_note_name(note, c1_midi_note=c1_midi_note)
        layers[slot_name] = SemanticLayerSlot(
            enabled=slot_name in _ENABLED_DEFAULT_SLOTS,
            note=note,
            note_name=note_name,
            track_name=_default_track_name(slot_name),
        )

    return DrumLayerMapping(
        name=name,
        c1_midi_note=c1_midi_note,
        layers=layers,
    )


def load_layer_mapping(mapping_file: Path, *, fallback_c1_midi_note: int) -> DrumLayerMapping:
    if not mapping_file.exists() or not mapping_file.is_file():
        raise DrumLayerMappingError(f"Mapping file does not exist: {mapping_file}")

    try:
        payload = json.loads(mapping_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DrumLayerMappingError(f"Failed to parse mapping JSON: {mapping_file}") from exc

    if not isinstance(payload, dict):
        raise DrumLayerMappingError("Mapping JSON must be an object")

    name = str(payload.get("name", "custom_drum_mapping"))
    c1_midi_note = _coerce_note_value(
        payload.get("c1_midi_note", fallback_c1_midi_note),
        field_name="c1_midi_note",
    )

    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, dict):
        raise DrumLayerMappingError("Mapping JSON must contain a 'layers' object")

    layers: dict[str, SemanticLayerSlot] = {}
    for layer_name, raw_slot in raw_layers.items():
        normalized_name = str(layer_name).strip()
        if not normalized_name:
            raise DrumLayerMappingError("Layer names must be non-empty")
        layers[normalized_name] = _parse_layer_slot(
            normalized_name,
            raw_slot,
            c1_midi_note=c1_midi_note,
        )

    if not layers:
        raise DrumLayerMappingError("Mapping JSON must include at least one layer")

    return DrumLayerMapping(name=name, c1_midi_note=c1_midi_note, layers=layers)


def save_layer_mapping(mapping: DrumLayerMapping, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(mapping.to_json_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def duplicate_target_notes(mapping: DrumLayerMapping) -> dict[str, list[str]]:
    by_note: dict[int, list[str]] = {}
    for layer_name, slot in mapping.layers.items():
        if not slot.enabled:
            continue
        by_note.setdefault(slot.note, []).append(layer_name)

    duplicates: dict[str, list[str]] = {}
    for note, layers in sorted(by_note.items()):
        if len(layers) > 1:
            duplicates[str(note)] = sorted(layers)

    return duplicates


def ordered_layers(mapping: DrumLayerMapping) -> list[str]:
    default_index = {name: index for index, name in enumerate(_DEFAULT_SLOT_ORDER)}

    def _key(layer_name: str) -> tuple[int, int, str]:
        if layer_name in default_index:
            return (0, default_index[layer_name], layer_name)
        return (1, len(default_index), layer_name)

    return sorted(mapping.layers, key=_key)
