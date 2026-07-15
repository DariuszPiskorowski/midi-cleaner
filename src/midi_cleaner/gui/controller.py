from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from midi_cleaner.gui.service import HermesGuiWorkflowError, HermesWorkflowResult, HermesWorkflowService
from midi_cleaner.gui.split_editor_launcher import SplitEditorLaunchResult, SplitEditorLauncher

ROLE_DRUMS = "drums"
ROLE_BASS = "bass"
ROLE_SYNTH = "synth"
ROLE_GUITAR = "guitar"
ROLE_OTHER = "other"

ACTION_MAKE_MIDI_FROM_WAV = "make_midi_from_wav"
ACTION_SYNCHRONIZE_MIDI_WITH_WAV = "synchronize_midi_with_wav"
ACTION_SET_BPM = "set_bpm"

_COMING_SOON_ROLES = {ROLE_SYNTH, ROLE_GUITAR, ROLE_OTHER}

_DRUMS_OUTPUT_LAYOUTS = {"separate-files", "multitrack", "single-track"}
_DRUMS_PROFILES = {"conservative", "balanced", "sensitive"}
_DRUMS_DETECTION_MODES = {"multi-detector", "global"}
_TARGET_MAPS = {"gm", "sitala", "ujam-candy", "custom"}


@dataclass(frozen=True)
class HermesDrumsRequest:
    output_dir: Path | None = None
    output_layout: str = "separate-files"
    profile: str = "conservative"
    detection_mode: str = "multi-detector"
    mapping_file: Path | None = None
    mapping_payload: dict[str, object] | None = None
    write_empty_layers: bool = False
    clean_output_folder: bool = False
    c1_midi_note: int = 36
    target_map: str = "ujam-candy"


@dataclass(frozen=True)
class HermesActionRequest:
    role: str
    action: str
    wav_file: Path | None
    midi_file: Path | None
    bpm_text: str | None = None
    drums: HermesDrumsRequest | None = None


@dataclass(frozen=True)
class HermesActionPlan:
    workflow: str
    role: str
    action: str
    wav_file: Path | None
    midi_file: Path | None
    bpm: float | None
    output_file: Path
    report_file: Path | None
    output_dir: Path | None = None
    debug_csv_file: Path | None = None
    drums: HermesDrumsRequest | None = None


@dataclass(frozen=True)
class HermesActionResult:
    success: bool
    message: str
    output_file: Path | None
    report_file: Path | None
    output_dir: Path | None = None
    debug_csv_file: Path | None = None
    created_files: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
    mapping_name: str | None = None
    duplicate_target_notes: dict[str, list[str]] = field(default_factory=dict)
    layer_counts: dict[str, int] = field(default_factory=dict)
    populated_semantic_layers: tuple[str, ...] = ()
    unpopulated_enabled_layers: tuple[str, ...] = ()
    disabled_layers: tuple[str, ...] = ()
    output_layout: str | None = None


class HermesGuiController:
    def __init__(
        self,
        service: HermesWorkflowService | None = None,
        desktop_dir: Path | None = None,
        split_editor_launcher: SplitEditorLauncher | None = None,
    ) -> None:
        self._service = service if service is not None else HermesWorkflowService()
        self._desktop_dir = desktop_dir if desktop_dir is not None else Path.home() / "Desktop"
        self._split_editor_launcher = (
            split_editor_launcher if split_editor_launcher is not None else SplitEditorLauncher()
        )

    @property
    def desktop_dir(self) -> Path:
        return self._desktop_dir

    def default_drums_mapping(self, *, target_map: str, c1_midi_note: int) -> dict[str, object]:
        return self._service.default_drums_mapping(target_map=target_map, c1_midi_note=c1_midi_note)

    def load_drums_mapping(
        self,
        *,
        mapping_file: Path,
        fallback_c1_midi_note: int,
    ) -> dict[str, object]:
        return self._service.load_drums_mapping(
            mapping_file=mapping_file,
            fallback_c1_midi_note=fallback_c1_midi_note,
        )

    def save_drums_mapping(
        self,
        *,
        mapping_payload: dict[str, object],
        destination_file: Path,
        fallback_c1_midi_note: int,
    ) -> Path:
        return self._service.save_drums_mapping(
            mapping_payload=mapping_payload,
            destination_file=destination_file,
            fallback_c1_midi_note=fallback_c1_midi_note,
        )

    @staticmethod
    def _parse_bpm_text(bpm_text: str | None) -> float | None:
        if bpm_text is None:
            return None
        normalized = bpm_text.strip()
        if not normalized:
            return None
        try:
            value = float(normalized)
        except ValueError as exc:
            raise ValueError("BPM must be a valid decimal number.") from exc
        if value <= 0:
            raise ValueError("BPM must be greater than 0.")
        return value

    @staticmethod
    def _bpm_token(bpm: float) -> str:
        rendered = f"{bpm:.3f}".rstrip("0").rstrip(".")
        return rendered.replace(".", "_")

    @staticmethod
    def build_unique_output_path(directory: Path, filename: str) -> Path:
        base = directory / filename
        if not base.exists():
            return base

        stem = base.stem
        suffix = base.suffix
        index = 2
        while True:
            candidate = directory / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def open_split_editor(
        self,
        midi_file: Path | None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> SplitEditorLaunchResult:
        return self._split_editor_launcher.open_split_editor(
            midi_file=midi_file,
            host=host,
            port=port,
        )

    def requires_midi(self, role: str, action: str) -> bool:
        if action == ACTION_SET_BPM:
            return True
        if action == ACTION_SYNCHRONIZE_MIDI_WITH_WAV:
            return True
        if action == ACTION_MAKE_MIDI_FROM_WAV and role == ROLE_BASS:
            return True
        return False

    def requires_bpm(self, action: str) -> bool:
        return action == ACTION_SET_BPM

    def supports_bpm(self, role: str, action: str) -> bool:
        if action == ACTION_SET_BPM:
            return True
        if action == ACTION_SYNCHRONIZE_MIDI_WITH_WAV:
            return True
        return False

    @staticmethod
    def _validate_drums_request(drums: HermesDrumsRequest) -> None:
        if drums.output_layout not in _DRUMS_OUTPUT_LAYOUTS:
            raise ValueError("Output layout must be separate-files, multitrack, or single-track.")

        if drums.profile not in _DRUMS_PROFILES:
            raise ValueError("Drums profile must be conservative, balanced, or sensitive.")

        if drums.detection_mode not in _DRUMS_DETECTION_MODES:
            raise ValueError("Detection mode must be multi-detector or global.")

        if drums.c1_midi_note < 0 or drums.c1_midi_note > 127:
            raise ValueError("C1 MIDI note must be in range 0..127.")

        if drums.target_map not in _TARGET_MAPS:
            raise ValueError("Target map must be one of: gm, sitala, ujam-candy, custom.")

        if drums.mapping_file is not None and (
            not drums.mapping_file.exists() or not drums.mapping_file.is_file()
        ):
            raise ValueError(f"Mapping file does not exist: {drums.mapping_file}")

        if drums.target_map == "custom" and drums.mapping_file is None and drums.mapping_payload is None:
            raise ValueError("Custom target map requires a mapping file or edited mapping payload.")

        if drums.output_dir is not None and drums.output_dir.exists() and not drums.output_dir.is_dir():
            raise ValueError(f"Output folder is not a directory: {drums.output_dir}")

    def _validate_inputs(self, request: HermesActionRequest) -> float | None:
        if request.role in _COMING_SOON_ROLES:
            raise ValueError("This workflow will be added later.")

        if request.action not in {
            ACTION_MAKE_MIDI_FROM_WAV,
            ACTION_SYNCHRONIZE_MIDI_WITH_WAV,
            ACTION_SET_BPM,
        }:
            raise ValueError(f"Unsupported action: {request.action}")

        if request.role not in {
            ROLE_DRUMS,
            ROLE_BASS,
            ROLE_SYNTH,
            ROLE_GUITAR,
            ROLE_OTHER,
        }:
            raise ValueError(f"Unsupported role: {request.role}")

        if request.wav_file is None and request.action != ACTION_SET_BPM:
            raise ValueError("WAV input is required for this action.")

        if request.wav_file is not None and (not request.wav_file.exists() or not request.wav_file.is_file()):
            raise ValueError(f"WAV file does not exist: {request.wav_file}")

        if self.requires_midi(request.role, request.action):
            if request.midi_file is None:
                raise ValueError("MIDI input is required for this action.")
            if not request.midi_file.exists() or not request.midi_file.is_file():
                raise ValueError(f"MIDI file does not exist: {request.midi_file}")

        bpm = self._parse_bpm_text(request.bpm_text)
        if self.requires_bpm(request.action) and bpm is None:
            raise ValueError("BPM value is required for this action.")
        if bpm is not None and not self.supports_bpm(request.role, request.action):
            raise ValueError("BPM override is not supported for the selected role/action.")

        if request.role == ROLE_DRUMS and request.action == ACTION_MAKE_MIDI_FROM_WAV:
            self._validate_drums_request(request.drums if request.drums is not None else HermesDrumsRequest())

        return bpm

    def _build_filenames(self, role: str, action: str, bpm: float | None) -> tuple[str, str]:
        if action == ACTION_MAKE_MIDI_FROM_WAV:
            if role == ROLE_DRUMS:
                stem = "hermes_drums_from_audio"
            elif role == ROLE_BASS:
                stem = "hermes_bass_working"
            else:
                raise ValueError("This workflow will be added later.")
        elif action == ACTION_SYNCHRONIZE_MIDI_WITH_WAV:
            stem = f"hermes_synced_{role}"
        elif action == ACTION_SET_BPM:
            if bpm is None:
                raise ValueError("BPM value is required for set BPM action.")
            stem = f"hermes_set_bpm_{self._bpm_token(bpm)}"
        else:
            raise ValueError(f"Unsupported action: {action}")

        return f"{stem}.mid", f"{stem}_report.json"

    def build_action_plan(self, request: HermesActionRequest) -> HermesActionPlan:
        bpm = self._validate_inputs(request)

        if request.role == ROLE_DRUMS and request.action == ACTION_MAKE_MIDI_FROM_WAV:
            drums = request.drums if request.drums is not None else HermesDrumsRequest()
            output_dir = drums.output_dir if drums.output_dir is not None else (self._desktop_dir / "hermes_drums_layers")
            output_file = output_dir / "hermes_drums_from_audio.mid"
            report_file = output_dir / "drums_layers_report.json"
            debug_csv = output_dir / "drums_layers_hits.csv"
            return HermesActionPlan(
                workflow="make_drums",
                role=request.role,
                action=request.action,
                wav_file=request.wav_file,
                midi_file=request.midi_file,
                bpm=bpm,
                output_file=output_file,
                report_file=report_file,
                output_dir=output_dir,
                debug_csv_file=debug_csv,
                drums=drums,
            )

        output_name, report_name = self._build_filenames(
            role=request.role,
            action=request.action,
            bpm=bpm,
        )
        output_file = self.build_unique_output_path(self._desktop_dir, output_name)
        report_file = self.build_unique_output_path(self._desktop_dir, report_name)

        if request.action == ACTION_MAKE_MIDI_FROM_WAV:
            workflow = "make_drums" if request.role == ROLE_DRUMS else "make_bass"
        elif request.action == ACTION_SYNCHRONIZE_MIDI_WITH_WAV:
            workflow = "sync_midi"
        else:
            workflow = "set_bpm"

        return HermesActionPlan(
            workflow=workflow,
            role=request.role,
            action=request.action,
            wav_file=request.wav_file,
            midi_file=request.midi_file,
            bpm=bpm,
            output_file=output_file,
            report_file=report_file,
        )

    def execute(
        self,
        request: HermesActionRequest,
        log: Callable[[str], None],
    ) -> HermesActionResult:
        try:
            plan = self.build_action_plan(request)
        except ValueError as exc:
            return HermesActionResult(
                success=False,
                message=str(exc),
                output_file=None,
                report_file=None,
            )

        try:
            workflow_result: HermesWorkflowResult
            if plan.workflow == "make_drums":
                if not self._service.drums_extraction_available:
                    return HermesActionResult(
                        success=False,
                        message="Audio-driven drums extraction is not implemented yet.",
                        output_file=None,
                        report_file=None,
                    )
                workflow_result = self._service.make_drums_midi_from_wav(
                    wav_file=plan.wav_file,
                    output_file=plan.output_file,
                    report_file=plan.report_file,
                    bpm_override=plan.bpm,
                    output_dir=plan.output_dir,
                    debug_csv_file=plan.debug_csv_file,
                    output_layout=(plan.drums.output_layout if plan.drums is not None else "separate-files"),
                    profile=(plan.drums.profile if plan.drums is not None else "conservative"),
                    detection_mode=(plan.drums.detection_mode if plan.drums is not None else "multi-detector"),
                    mapping_file=(plan.drums.mapping_file if plan.drums is not None else None),
                    mapping_payload=(plan.drums.mapping_payload if plan.drums is not None else None),
                    write_empty_layers=(plan.drums.write_empty_layers if plan.drums is not None else False),
                    clean_output_folder=(plan.drums.clean_output_folder if plan.drums is not None else False),
                    c1_midi_note=(plan.drums.c1_midi_note if plan.drums is not None else 36),
                    target_map=(plan.drums.target_map if plan.drums is not None else "ujam-candy"),
                    log=log,
                )
            elif plan.workflow == "make_bass":
                workflow_result = self._service.make_bass_midi_from_wav(
                    wav_file=plan.wav_file,
                    midi_file=plan.midi_file,
                    output_file=plan.output_file,
                    report_file=plan.report_file,
                    bpm_override=plan.bpm,
                    log=log,
                )
            elif plan.workflow == "sync_midi":
                workflow_result = self._service.synchronize_midi_with_wav(
                    wav_file=plan.wav_file,
                    midi_file=plan.midi_file,
                    role=plan.role,
                    output_file=plan.output_file,
                    report_file=plan.report_file,
                    bpm_override=plan.bpm,
                    log=log,
                )
            elif plan.workflow == "set_bpm":
                workflow_result = self._service.set_midi_bpm(
                    midi_file=plan.midi_file,
                    bpm=float(plan.bpm),
                    output_file=plan.output_file,
                    report_file=plan.report_file,
                    log=log,
                )
            else:
                raise HermesGuiWorkflowError(f"Unsupported workflow: {plan.workflow}")
        except HermesGuiWorkflowError as exc:
            return HermesActionResult(
                success=False,
                message=str(exc),
                output_file=None,
                report_file=None,
            )

        return HermesActionResult(
            success=True,
            message=workflow_result.message,
            output_file=workflow_result.output_file,
            report_file=workflow_result.report_file,
            output_dir=workflow_result.output_dir,
            debug_csv_file=workflow_result.debug_csv_file,
            created_files=tuple(workflow_result.created_files),
            warnings=tuple(workflow_result.warnings),
            mapping_name=workflow_result.mapping_name,
            duplicate_target_notes=dict(workflow_result.duplicate_target_notes),
            layer_counts=dict(workflow_result.layer_counts),
            populated_semantic_layers=tuple(workflow_result.populated_semantic_layers),
            unpopulated_enabled_layers=tuple(workflow_result.unpopulated_enabled_layers),
            disabled_layers=tuple(workflow_result.disabled_layers),
            output_layout=workflow_result.output_layout,
        )
