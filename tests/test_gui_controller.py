from __future__ import annotations

from pathlib import Path

from midi_cleaner.gui.controller import (
    ACTION_MAKE_MIDI_FROM_WAV,
    ACTION_SET_BPM,
    ACTION_SYNCHRONIZE_MIDI_WITH_WAV,
    ROLE_BASS,
    ROLE_DRUMS,
    HermesActionRequest,
    HermesGuiController,
)
from midi_cleaner.gui.service import HermesWorkflowResult


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
                },
            )
        )
        return HermesWorkflowResult(
            output_file=output_file,
            report_file=report_file,
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
