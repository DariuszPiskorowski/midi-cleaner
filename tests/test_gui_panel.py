from __future__ import annotations

import json
import re
import time
import tkinter as tk
from pathlib import Path

import pytest

from midi_cleaner.gui.controller import (
    ACTION_MAKE_MIDI_FROM_WAV,
    ROLE_BASS,
    ROLE_DRUMS,
    HermesActionRequest,
    HermesActionResult,
    HermesDrumsRequest,
)
from midi_cleaner.gui.panel import HermesGuiPanel


class _FakePanelController:
    def __init__(self, desktop_dir: Path) -> None:
        self.desktop_dir = desktop_dir
        self.calls: list[HermesActionRequest] = []
        self.saved_mapping_payload: dict[str, object] | None = None

    @staticmethod
    def requires_midi(role: str, action: str) -> bool:
        return role == ROLE_BASS and action == ACTION_MAKE_MIDI_FROM_WAV

    @staticmethod
    def supports_bpm(role: str, action: str) -> bool:
        return False

    @staticmethod
    def default_drums_mapping(*, target_map: str, c1_midi_note: int) -> dict[str, object]:
        return {
            "name": f"{target_map}_expanded_default_mapping",
            "c1_midi_note": c1_midi_note,
            "layers": {
                "kick_1": {
                    "enabled": True,
                    "note_name": "C1",
                    "note": 36,
                    "track_name": "Kick1",
                },
                "snare_1": {
                    "enabled": True,
                    "note_name": "E1",
                    "note": 40,
                    "track_name": "Snare1",
                },
                "clap_1": {
                    "enabled": True,
                    "note_name": "G1",
                    "note": 43,
                    "track_name": "Clap1",
                },
                "hh_1": {
                    "enabled": True,
                    "note_name": "C2",
                    "note": 48,
                    "track_name": "HH1",
                },
                "tom_l_1": {
                    "enabled": True,
                    "note_name": "D2",
                    "note": 50,
                    "track_name": "TomL1",
                },
                "cym_1": {
                    "enabled": True,
                    "note_name": "C3",
                    "note": 60,
                    "track_name": "Cym1",
                },
            },
        }

    @staticmethod
    def load_drums_mapping(*, mapping_file: Path, fallback_c1_midi_note: int) -> dict[str, object]:
        payload = json.loads(mapping_file.read_text(encoding="utf-8"))
        if "c1_midi_note" not in payload:
            payload["c1_midi_note"] = fallback_c1_midi_note
        return payload

    def save_drums_mapping(
        self,
        *,
        mapping_payload: dict[str, object],
        destination_file: Path,
        fallback_c1_midi_note: int,
    ) -> Path:
        self.saved_mapping_payload = mapping_payload
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        destination_file.write_text(json.dumps(mapping_payload, indent=2) + "\n", encoding="utf-8")
        return destination_file

    def execute(self, request: HermesActionRequest, log) -> HermesActionResult:
        self.calls.append(request)

        if request.role == ROLE_DRUMS and request.action == ACTION_MAKE_MIDI_FROM_WAV:
            assert request.drums is not None
            mapping_payload = request.drums.mapping_payload
            if isinstance(mapping_payload, dict):
                layers = mapping_payload.get("layers")
                if isinstance(layers, dict):
                    for layer_name, layer in layers.items():
                        if not isinstance(layer, dict):
                            continue
                        note_name = str(layer.get("note_name", "")).strip()
                        if note_name and re.fullmatch(r"[A-G][#b]?\d+", note_name) is None:
                            return HermesActionResult(
                                success=False,
                                message=f"Invalid note name for layer {layer_name}: {note_name}",
                                output_file=None,
                                report_file=None,
                            )

            output_dir = request.drums.output_dir if request.drums.output_dir is not None else (self.desktop_dir / "drums")
            report_file = output_dir / "drums_layers_report.json"
            debug_csv = output_dir / "drums_layers_hits.csv"
            created_files = (
                output_dir / "01_Kick1_C1.mid",
                output_dir / "02_Clap1_G1.mid",
                output_dir / "03_HH1_C2.mid",
            )
            return HermesActionResult(
                success=True,
                message="Drums extracted from WAV.",
                output_file=None,
                report_file=report_file,
                output_dir=output_dir,
                debug_csv_file=debug_csv,
                created_files=created_files,
                warnings=("Primary slot assignment used.",),
                mapping_name="gui_test_mapping",
                duplicate_target_notes={"48": ["hh_1", "hh_open_1"]},
                layer_counts={"kick_1": 10, "clap_1": 4, "hh_1": 8},
                populated_semantic_layers=("kick_1", "clap_1", "hh_1"),
                unpopulated_enabled_layers=("snare_1",),
                disabled_layers=("kick_2",),
                output_layout="separate-files",
            )

        output_file = self.desktop_dir / "hermes_bass_working.mid"
        report_file = self.desktop_dir / "hermes_bass_working_report.json"
        return HermesActionResult(
            success=True,
            message="Bass make-MIDI workflow complete.",
            output_file=output_file,
            report_file=report_file,
        )


def _new_panel(tmp_path: Path) -> tuple[HermesGuiPanel, _FakePanelController]:
    controller = _FakePanelController(desktop_dir=tmp_path)
    try:
        panel = HermesGuiPanel(controller=controller)
    except tk.TclError as exc:  # pragma: no cover
        pytest.skip(f"tk unavailable in test environment: {exc}")
    panel._root.withdraw()
    panel._root.update_idletasks()
    return panel, controller


def _wait_for_worker(panel: HermesGuiPanel, *, timeout_sec: float = 3.0) -> None:
    deadline = time.time() + timeout_sec
    while panel._running and time.time() < deadline:
        panel._root.update()
        time.sleep(0.01)
    panel._root.update()
    assert panel._running is False


def _run_action_sync(panel: HermesGuiPanel, monkeypatch: pytest.MonkeyPatch) -> None:
    class _ImmediateThread:
        def __init__(self, *, target=None, daemon=None):
            self._target = target
            self.daemon = daemon

        def start(self) -> None:
            if self._target is not None:
                self._target()

    monkeypatch.setattr("midi_cleaner.gui.panel.threading.Thread", _ImmediateThread)
    panel._run_action()
    panel._root.update()
    panel._root.update_idletasks()


def test_drums_panel_appears_for_role_drums(tmp_path: Path) -> None:
    panel, _controller = _new_panel(tmp_path)
    try:
        panel._role_var.set("drums")
        panel._action_var.set("make MIDI from WAV")
        panel._refresh_field_visibility()
        assert panel._drums_panel.winfo_manager() == "grid"

        panel._role_var.set("bass")
        panel._refresh_field_visibility()
        assert panel._drums_panel.winfo_manager() == ""
    finally:
        panel.close()


def test_write_empty_layers_default_false(tmp_path: Path) -> None:
    panel, _controller = _new_panel(tmp_path)
    try:
        assert panel._write_empty_layers_var.get() is False
    finally:
        panel.close()


def test_mapping_table_loads_default_expanded_mapping(tmp_path: Path) -> None:
    panel, _controller = _new_panel(tmp_path)
    try:
        children = panel._mapping_table.get_children()
        assert "kick_1" in children
        assert "snare_1" in children
        assert "clap_1" in children
        assert "hh_1" in children
        assert "cym_1" in children
    finally:
        panel.close()


def test_edited_mapping_can_be_saved_to_json(tmp_path: Path) -> None:
    panel, controller = _new_panel(tmp_path)
    try:
        panel._mapping_selected_layer_var.set("kick_1")
        panel._load_selected_layer_into_editor()
        panel._mapping_enabled_edit_var.set(False)
        panel._mapping_track_edit_var.set("KickOne")
        panel._mapping_note_edit_var.set("D1")
        panel._apply_mapping_editor_changes()

        mapping_path = tmp_path / "saved_mapping.json"
        panel._mapping_file_var.set(str(mapping_path))
        panel._save_mapping_to_file()

        assert mapping_path.exists()
        assert controller.saved_mapping_payload is not None
        layers = controller.saved_mapping_payload["layers"]
        assert isinstance(layers, dict)
        kick = layers["kick_1"]
        assert isinstance(kick, dict)
        assert kick["enabled"] is False
        assert kick["track_name"] == "KickOne"
        assert kick["note_name"] == "D1"
    finally:
        panel.close()


def test_separate_files_run_shows_created_files_report_debug_and_output_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel, _controller = _new_panel(tmp_path)
    try:
        wav = tmp_path / "Drums.wav"
        wav.write_bytes(b"wav")

        panel._role_var.set("drums")
        panel._action_var.set("make MIDI from WAV")
        panel._refresh_field_visibility()

        output_dir = tmp_path / "drums_layers"
        panel._wav_var.set(str(wav))
        panel._output_dir_var.set(str(output_dir))
        panel._output_layout_var.set("separate-files")
        _run_action_sync(panel, monkeypatch)

        assert panel._output_var.get() == ""
        assert panel._output_dir_var.get() == str(output_dir)
        assert panel._report_var.get().endswith("drums_layers_report.json")
        assert panel._debug_csv_var.get().endswith("drums_layers_hits.csv")
        assert panel._created_files_count_var.get() == "3"

        created_text = panel._created_files_text.get("1.0", tk.END)
        assert "01_Kick1_C1.mid" in created_text
        assert "02_Clap1_G1.mid" in created_text
        assert "03_HH1_C2.mid" in created_text
    finally:
        panel.close()


def test_old_non_drums_gui_workflow_still_works(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel, _controller = _new_panel(tmp_path)
    try:
        wav = tmp_path / "Bass.wav"
        midi = tmp_path / "Input.mid"
        wav.write_bytes(b"wav")
        midi.write_bytes(b"midi")

        panel._role_var.set("bass")
        panel._action_var.set("make MIDI from WAV")
        panel._refresh_field_visibility()

        panel._wav_var.set(str(wav))
        panel._midi_var.set(str(midi))
        _run_action_sync(panel, monkeypatch)

        assert panel._output_var.get().endswith("hermes_bass_working.mid")
        assert panel._report_var.get().endswith("hermes_bass_working_report.json")
        assert panel._created_files_count_var.get() == "0"
    finally:
        panel.close()
