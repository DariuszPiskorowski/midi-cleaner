from __future__ import annotations

from pathlib import Path

from midi_cleaner.gui.controller import (
    ACTION_MAKE_MIDI_FROM_WAV,
    ACTION_SET_BPM,
    ACTION_SYNCHRONIZE_MIDI_WITH_WAV,
    HermesDrumsRequest,
    ROLE_BASS,
    ROLE_DRUMS,
    HermesActionRequest,
    HermesGuiController,
)
from midi_cleaner.gui.service import HermesWorkflowResult
from midi_cleaner.gui.split_editor_launcher import SplitEditorLaunchResult


class _FakeWorkflowService:
    def __init__(self) -> None:
        self.drums_extraction_available = True
        self.calls: list[tuple[str, dict[str, object]]] = []

    def make_bass_midi_from_wav(
        self,
        wav_file: Path,
        midi_file: Path,
        output_file: Path,
        report_file: Path | None,
        bpm_override: float | None,
        log,
    ) -> HermesWorkflowResult:
        self.calls.append(
            (
                "make_bass",
                {
                    "wav_file": wav_file,
                    "midi_file": midi_file,
                    "output_file": output_file,
                    "report_file": report_file,
                    "bpm_override": bpm_override,
                },
            )
        )
        return HermesWorkflowResult(
            output_file=output_file,
            report_file=report_file,
            message="ok",
        )

    def make_drums_midi_from_wav(
        self,
        wav_file: Path,
        output_file: Path,
        report_file: Path | None,
        bpm_override: float | None,
        output_dir: Path | None,
        debug_csv_file: Path | None,
        output_layout: str,
        profile: str,
        detection_mode: str,
        mapping_file: Path | None,
        mapping_payload: dict[str, object] | None,
        write_empty_layers: bool,
        clean_output_folder: bool,
        c1_midi_note: int,
        target_map: str,
        log,
    ) -> HermesWorkflowResult:
        self.calls.append(
            (
                "make_drums",
                {
                    "wav_file": wav_file,
                    "output_file": output_file,
                    "report_file": report_file,
                    "bpm_override": bpm_override,
                    "output_dir": output_dir,
                    "debug_csv_file": debug_csv_file,
                    "output_layout": output_layout,
                    "profile": profile,
                    "detection_mode": detection_mode,
                    "mapping_file": mapping_file,
                    "mapping_payload": mapping_payload,
                    "write_empty_layers": write_empty_layers,
                    "clean_output_folder": clean_output_folder,
                    "c1_midi_note": c1_midi_note,
                    "target_map": target_map,
                },
            )
        )
        return HermesWorkflowResult(
            output_file=None,
            report_file=report_file,
            output_dir=output_dir,
            debug_csv_file=debug_csv_file,
            created_files=(
                (output_dir / "01_Kick1_C1.mid") if output_dir is not None else Path("01_Kick1_C1.mid"),
                (output_dir / "02_Clap1_G1.mid") if output_dir is not None else Path("02_Clap1_G1.mid"),
            ),
            warnings=("duplicate target note",),
            mapping_name="test_mapping",
            duplicate_target_notes={"48": ["hh_1", "hh_open_1"]},
            layer_counts={"kick_1": 1, "clap_1": 1},
            populated_semantic_layers=("kick_1", "clap_1"),
            unpopulated_enabled_layers=("snare_1",),
            disabled_layers=("kick_2",),
            output_layout="separate-files",
            message="ok",
        )

    def synchronize_midi_with_wav(
        self,
        wav_file: Path,
        midi_file: Path,
        role: str,
        output_file: Path,
        report_file: Path | None,
        bpm_override: float | None,
        log,
    ) -> HermesWorkflowResult:
        self.calls.append(
            (
                "sync_midi",
                {
                    "wav_file": wav_file,
                    "midi_file": midi_file,
                    "role": role,
                    "output_file": output_file,
                    "report_file": report_file,
                    "bpm_override": bpm_override,
                },
            )
        )
        return HermesWorkflowResult(
            output_file=output_file,
            report_file=report_file,
            message="ok",
        )

    def set_midi_bpm(
        self,
        midi_file: Path,
        bpm: float,
        output_file: Path,
        report_file: Path | None,
        log,
    ) -> HermesWorkflowResult:
        self.calls.append(
            (
                "set_bpm",
                {
                    "midi_file": midi_file,
                    "bpm": bpm,
                    "output_file": output_file,
                    "report_file": report_file,
                },
            )
        )
        return HermesWorkflowResult(
            output_file=output_file,
            report_file=report_file,
            message="ok",
        )

    def default_drums_mapping(self, *, target_map: str, c1_midi_note: int) -> dict[str, object]:
        return {
            "name": "default_map",
            "c1_midi_note": c1_midi_note,
            "layers": {
                "kick_1": {
                    "enabled": True,
                    "note_name": "C1",
                    "note": 36,
                    "track_name": "Kick1",
                }
            },
        }

    def load_drums_mapping(self, *, mapping_file: Path, fallback_c1_midi_note: int) -> dict[str, object]:
        self.calls.append(("load_mapping", {"mapping_file": mapping_file, "fallback": fallback_c1_midi_note}))
        return self.default_drums_mapping(target_map="ujam-candy", c1_midi_note=fallback_c1_midi_note)

    def save_drums_mapping(
        self,
        *,
        mapping_payload: dict[str, object],
        destination_file: Path,
        fallback_c1_midi_note: int,
    ) -> Path:
        self.calls.append(
            (
                "save_mapping",
                {
                    "mapping_payload": mapping_payload,
                    "destination_file": destination_file,
                    "fallback": fallback_c1_midi_note,
                },
            )
        )
        return destination_file


class _FakeSplitEditorLauncher:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def open_split_editor(
        self,
        midi_file: Path | None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> SplitEditorLaunchResult:
        self.calls.append(
            {
                "midi_file": midi_file,
                "host": host,
                "port": port,
            }
        )
        return SplitEditorLaunchResult(
            success=True,
            url=f"http://{host}:{port}/",
            message=f"MIDI Split Editor opened at http://{host}:{port}/",
            reused_existing_server=False,
            started_new_server=True,
        )


def test_unique_output_path_suffixing(tmp_path: Path) -> None:
    controller = HermesGuiController(desktop_dir=tmp_path)

    first = controller.build_unique_output_path(tmp_path, "hermes_drums_from_audio.mid")
    assert first == tmp_path / "hermes_drums_from_audio.mid"

    (tmp_path / "hermes_drums_from_audio.mid").write_text("x", encoding="utf-8")
    second = controller.build_unique_output_path(tmp_path, "hermes_drums_from_audio.mid")
    assert second == tmp_path / "hermes_drums_from_audio_2.mid"

    (tmp_path / "hermes_drums_from_audio_2.mid").write_text("x", encoding="utf-8")
    third = controller.build_unique_output_path(tmp_path, "hermes_drums_from_audio.mid")
    assert third == tmp_path / "hermes_drums_from_audio_3.mid"


def test_bass_make_midi_calls_bass_workflow(tmp_path: Path) -> None:
    wav = tmp_path / "stem.wav"
    midi = tmp_path / "seed.mid"
    wav.write_bytes(b"wav")
    midi.write_bytes(b"midi")

    service = _FakeWorkflowService()
    controller = HermesGuiController(service=service, desktop_dir=tmp_path)

    request = HermesActionRequest(
        role=ROLE_BASS,
        action=ACTION_MAKE_MIDI_FROM_WAV,
        wav_file=wav,
        midi_file=midi,
        bpm_text=None,
    )

    result = controller.execute(request=request, log=lambda _message: None)

    assert result.success is True
    assert len(service.calls) == 1
    name, payload = service.calls[0]
    assert name == "make_bass"
    assert payload["wav_file"] == wav
    assert payload["midi_file"] == midi
    assert payload["output_file"] == tmp_path / "hermes_bass_working.mid"


def test_drums_make_midi_uses_output_folder_and_returns_created_files(tmp_path: Path) -> None:
    wav = tmp_path / "drums.wav"
    wav.write_bytes(b"wav")

    output_dir = tmp_path / "layers"
    service = _FakeWorkflowService()
    controller = HermesGuiController(service=service, desktop_dir=tmp_path)

    request = HermesActionRequest(
        role=ROLE_DRUMS,
        action=ACTION_MAKE_MIDI_FROM_WAV,
        wav_file=wav,
        midi_file=None,
        bpm_text=None,
        drums=HermesDrumsRequest(
            output_dir=output_dir,
            output_layout="separate-files",
            profile="conservative",
            detection_mode="multi-detector",
            write_empty_layers=False,
            clean_output_folder=False,
            c1_midi_note=36,
            target_map="ujam-candy",
        ),
    )

    result = controller.execute(request=request, log=lambda _message: None)

    assert result.success is True
    assert result.output_dir == output_dir
    assert result.output_file is None
    assert result.report_file == output_dir / "drums_layers_report.json"
    assert result.debug_csv_file == output_dir / "drums_layers_hits.csv"
    assert len(result.created_files) == 2
    assert result.output_layout == "separate-files"


def test_drums_make_midi_returns_not_implemented_without_fallback(tmp_path: Path) -> None:
    wav = tmp_path / "drums.wav"
    wav.write_bytes(b"wav")

    service = _FakeWorkflowService()
    service.drums_extraction_available = False

    controller = HermesGuiController(service=service, desktop_dir=tmp_path)
    request = HermesActionRequest(
        role=ROLE_DRUMS,
        action=ACTION_MAKE_MIDI_FROM_WAV,
        wav_file=wav,
        midi_file=None,
        bpm_text=None,
    )

    result = controller.execute(request=request, log=lambda _message: None)

    assert result.success is False
    assert result.message == "Audio-driven drums extraction is not implemented yet."
    assert service.calls == []


def test_sync_requires_wav_and_midi_inputs(tmp_path: Path) -> None:
    midi = tmp_path / "input.mid"
    wav = tmp_path / "stem.wav"
    midi.write_bytes(b"midi")
    wav.write_bytes(b"wav")

    controller = HermesGuiController(service=_FakeWorkflowService(), desktop_dir=tmp_path)

    missing_wav = HermesActionRequest(
        role=ROLE_BASS,
        action=ACTION_SYNCHRONIZE_MIDI_WITH_WAV,
        wav_file=None,
        midi_file=midi,
        bpm_text=None,
    )
    missing_midi = HermesActionRequest(
        role=ROLE_BASS,
        action=ACTION_SYNCHRONIZE_MIDI_WITH_WAV,
        wav_file=wav,
        midi_file=None,
        bpm_text=None,
    )

    result_missing_wav = controller.execute(request=missing_wav, log=lambda _message: None)
    result_missing_midi = controller.execute(request=missing_midi, log=lambda _message: None)

    assert result_missing_wav.success is False
    assert result_missing_wav.message == "WAV input is required for this action."
    assert result_missing_midi.success is False
    assert result_missing_midi.message == "MIDI input is required for this action."


def test_set_bpm_requires_midi_and_bpm(tmp_path: Path) -> None:
    midi = tmp_path / "input.mid"
    midi.write_bytes(b"midi")

    controller = HermesGuiController(service=_FakeWorkflowService(), desktop_dir=tmp_path)

    missing_midi = HermesActionRequest(
        role=ROLE_BASS,
        action=ACTION_SET_BPM,
        wav_file=None,
        midi_file=None,
        bpm_text="124.529",
    )
    missing_bpm = HermesActionRequest(
        role=ROLE_BASS,
        action=ACTION_SET_BPM,
        wav_file=None,
        midi_file=midi,
        bpm_text="",
    )

    result_missing_midi = controller.execute(request=missing_midi, log=lambda _message: None)
    result_missing_bpm = controller.execute(request=missing_bpm, log=lambda _message: None)

    assert result_missing_midi.success is False
    assert result_missing_midi.message == "MIDI input is required for this action."
    assert result_missing_bpm.success is False
    assert result_missing_bpm.message == "BPM value is required for this action."


def test_set_bpm_filename_uses_decimal_token(tmp_path: Path) -> None:
    midi = tmp_path / "input.mid"
    midi.write_bytes(b"midi")

    controller = HermesGuiController(service=_FakeWorkflowService(), desktop_dir=tmp_path)
    request = HermesActionRequest(
        role=ROLE_BASS,
        action=ACTION_SET_BPM,
        wav_file=None,
        midi_file=midi,
        bpm_text="124.529",
    )

    plan = controller.build_action_plan(request)
    assert plan.output_file.name == "hermes_set_bpm_124_529.mid"


def test_controller_exposes_drums_mapping_helpers(tmp_path: Path) -> None:
    service = _FakeWorkflowService()
    controller = HermesGuiController(service=service, desktop_dir=tmp_path)

    payload = controller.default_drums_mapping(target_map="ujam-candy", c1_midi_note=36)
    assert payload["name"] == "default_map"

    loaded = controller.load_drums_mapping(
        mapping_file=tmp_path / "mapping.json",
        fallback_c1_midi_note=36,
    )
    assert loaded["c1_midi_note"] == 36

    saved_path = controller.save_drums_mapping(
        mapping_payload=payload,
        destination_file=tmp_path / "saved_mapping.json",
        fallback_c1_midi_note=36,
    )
    assert saved_path == tmp_path / "saved_mapping.json"
    assert any(call[0] == "save_mapping" for call in service.calls)


def test_controller_exposes_split_editor_launch_method(tmp_path: Path) -> None:
    launcher = _FakeSplitEditorLauncher()
    controller = HermesGuiController(
        service=_FakeWorkflowService(),
        desktop_dir=tmp_path,
        split_editor_launcher=launcher,
    )

    midi = tmp_path / "selected.mid"
    midi.write_bytes(b"midi")

    result = controller.open_split_editor(midi_file=midi)

    assert result.success is True
    assert result.url == "http://127.0.0.1:8765/"
    assert len(launcher.calls) == 1
    assert launcher.calls[0]["midi_file"] == midi
    assert launcher.calls[0]["host"] == "127.0.0.1"
    assert launcher.calls[0]["port"] == 8765
