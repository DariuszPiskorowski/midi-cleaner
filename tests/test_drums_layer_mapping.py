from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from midi_cleaner.drums.layer_mapping import (
    DrumLayerMappingError,
    build_default_layer_mapping,
    duplicate_target_notes,
    load_layer_mapping,
    midi_to_note_name,
    note_name_to_midi,
    ordered_layers,
)


def test_default_mapping_includes_expanded_slots() -> None:
    mapping = build_default_layer_mapping(target_map="ujam-candy", c1_midi_note=36)
    layers = mapping.layers

    assert "kick_1" in layers
    assert "kick_2" in layers
    assert "kick_3" in layers
    assert "snare_1" in layers
    assert "clap_1" in layers
    assert "hh_open_1" in layers
    assert "tom_l_1" in layers
    assert "tom_m_1" in layers
    assert "tom_h_1" in layers
    assert "perc_1" in layers
    assert "cym_1" in layers
    assert "fx_1" in layers
    assert "acc_1" in layers
    assert len(layers) >= 36


def test_note_name_to_midi_uses_c1_36() -> None:
    assert note_name_to_midi("C1", c1_midi_note=36) == 36
    assert note_name_to_midi("D1", c1_midi_note=36) == 38
    assert note_name_to_midi("C2", c1_midi_note=36) == 48


def test_midi_to_note_name_uses_c1_36() -> None:
    assert midi_to_note_name(36, c1_midi_note=36) == "C1"
    assert midi_to_note_name(38, c1_midi_note=36) == "D1"
    assert midi_to_note_name(48, c1_midi_note=36) == "C2"


def test_invalid_note_mapping_raises_clear_error(tmp_path: Path) -> None:
    payload = {
        "name": "bad",
        "c1_midi_note": 36,
        "layers": {
            "kick_1": {
                "enabled": True,
                "note_name": "C1",
                "note": 40,
                "track_name": "Kick1",
            }
        },
    }
    mapping_file = tmp_path / "bad_mapping.json"
    mapping_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DrumLayerMappingError) as exc:
        load_layer_mapping(mapping_file, fallback_c1_midi_note=36)

    assert "do not match" in str(exc.value)


def test_duplicate_target_notes_warn_only() -> None:
    mapping = build_default_layer_mapping(target_map="gm", c1_midi_note=36)
    layers = dict(mapping.layers)
    layers["kick_2"] = replace(
        layers["kick_2"],
        enabled=True,
        note=layers["kick_1"].note,
        note_name=layers["kick_1"].note_name,
    )
    duplicate = duplicate_target_notes(
        replace(mapping, layers=layers)
    )

    assert str(layers["kick_1"].note) in duplicate
    assert "kick_1" in duplicate[str(layers["kick_1"].note)]
    assert "kick_2" in duplicate[str(layers["kick_1"].note)]


def test_unknown_custom_slot_is_preserved(tmp_path: Path) -> None:
    payload = {
        "name": "custom_with_extra",
        "c1_midi_note": 36,
        "layers": {
            "kick_1": {
                "enabled": True,
                "note_name": "C1",
                "track_name": "Kick1",
            },
            "noise_blip_7": {
                "enabled": True,
                "note": 74,
                "track_name": "NoiseBlip7",
            },
        },
    }
    mapping_file = tmp_path / "custom_mapping.json"
    mapping_file.write_text(json.dumps(payload), encoding="utf-8")

    mapping = load_layer_mapping(mapping_file, fallback_c1_midi_note=36)

    assert "noise_blip_7" in mapping.layers
    assert mapping.layers["noise_blip_7"].note == 74
    assert "noise_blip_7" in ordered_layers(mapping)
