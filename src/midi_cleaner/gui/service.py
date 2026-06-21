from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from midi_cleaner.drums.extract_audio import (
    AudioDrumExtractionError,
    AudioDrumExtractionParameters,
    extract_drums_from_audio,
)
from midi_cleaner.drums.layer_mapping import (
    DrumLayerMapping,
    DrumLayerMappingError,
    build_default_layer_mapping,
    load_layer_mapping,
    save_layer_mapping,
)
from midi_cleaner.midi.set_bpm import MidiSetBpmError, set_midi_bpm as set_midi_bpm_file
from midi_cleaner.midi.sync_with_audio import (
    MidiSyncWithAudioError,
    MidiSyncWithAudioParameters,
    sync_midi_with_wav,
)
from midi_cleaner.pipeline.process_stem import (
    PipelineProcessError,
    PipelineProcessParameters,
    process_stem_pipeline,
)


class HermesGuiWorkflowError(Exception):
    """Raised when a GUI-launched workflow fails."""


@dataclass(frozen=True)
class HermesWorkflowResult:
    output_file: Path | None
    report_file: Path | None
    message: str
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


class HermesWorkflowService:
    def __init__(self) -> None:
        self.drums_extraction_available = True

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _safe_clean_drums_output_dir(output_dir: Path, log: Callable[[str], None]) -> None:
        if not output_dir.exists() or not output_dir.is_dir():
            return

        removed = 0
        for child in output_dir.iterdir():
            if not child.is_file():
                continue
            name = child.name.lower()
            if name == "drums_layers_report.json" or name == "drums_layers_hits.csv":
                child.unlink(missing_ok=True)
                removed += 1
                continue

            if name.endswith(".mid") and len(name) >= 3 and name[:2].isdigit() and name[2] == "_":
                child.unlink(missing_ok=True)
                removed += 1

        if removed > 0:
            log(f"Cleaned {removed} Hermes-generated file(s) from output folder.")

    @staticmethod
    def _load_or_build_mapping(
        *,
        mapping_file: Path | None,
        mapping_payload: dict[str, object] | None,
        target_map: str,
        c1_midi_note: int,
    ) -> DrumLayerMapping:
        if mapping_payload is not None:
            parsed = mapping_payload
            if not isinstance(parsed, dict):
                raise DrumLayerMappingError("Mapping payload must be an object.")

            temp_path = Path.cwd() / ".hermes_gui_mapping_tmp.json"
            temp_path.write_text(json.dumps(parsed), encoding="utf-8")
            try:
                return load_layer_mapping(temp_path, fallback_c1_midi_note=c1_midi_note)
            finally:
                temp_path.unlink(missing_ok=True)

        if mapping_file is not None:
            return load_layer_mapping(mapping_file, fallback_c1_midi_note=c1_midi_note)

        return build_default_layer_mapping(target_map=target_map, c1_midi_note=c1_midi_note)

    def save_drums_mapping(
        self,
        *,
        mapping_payload: dict[str, object],
        destination_file: Path,
        fallback_c1_midi_note: int,
    ) -> Path:
        try:
            temp_path = destination_file.parent / ".hermes_gui_mapping_validate.json"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(json.dumps(mapping_payload), encoding="utf-8")
            mapping = load_layer_mapping(temp_path, fallback_c1_midi_note=fallback_c1_midi_note)
            temp_path.unlink(missing_ok=True)
        except DrumLayerMappingError as exc:
            raise HermesGuiWorkflowError(str(exc)) from exc

        save_layer_mapping(mapping, destination_file)
        return destination_file

    def load_drums_mapping(
        self,
        *,
        mapping_file: Path,
        fallback_c1_midi_note: int,
    ) -> dict[str, object]:
        try:
            mapping = load_layer_mapping(mapping_file, fallback_c1_midi_note=fallback_c1_midi_note)
        except DrumLayerMappingError as exc:
            raise HermesGuiWorkflowError(str(exc)) from exc
        return mapping.to_json_dict()

    def default_drums_mapping(
        self,
        *,
        target_map: str,
        c1_midi_note: int,
    ) -> dict[str, object]:
        mapping = build_default_layer_mapping(target_map=target_map, c1_midi_note=c1_midi_note)
        return mapping.to_json_dict()

    def make_bass_midi_from_wav(
        self,
        wav_file: Path,
        midi_file: Path,
        output_file: Path,
        report_file: Path | None,
        bpm_override: float | None,
        log: Callable[[str], None],
    ) -> HermesWorkflowResult:
        if not midi_file.exists() or not midi_file.is_file():
            raise HermesGuiWorkflowError(f"Input MIDI file does not exist: {midi_file}")
        if bpm_override is not None:
            log("BPM override is ignored for bass make-MIDI workflow.")

        with TemporaryDirectory(prefix="hermes_gui_bass_") as temp_dir:
            project_dir = Path(temp_dir) / "project"
            log("Running Hermes bass process-stem workflow.")
            try:
                report = process_stem_pipeline(
                    input_midi=midi_file,
                    input_wav=wav_file,
                    source="gui",
                    layer="bass",
                    project_dir=project_dir,
                    params=PipelineProcessParameters(),
                )
            except PipelineProcessError as exc:
                raise HermesGuiWorkflowError(str(exc)) from exc

            working_midi = project_dir / "midi" / "working" / "working.mid"
            if not working_midi.exists() or not working_midi.is_file():
                raise HermesGuiWorkflowError("Bass workflow did not produce working.mid.")

            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(working_midi, output_file)

            if report_file is not None:
                self._write_json(
                    report_file,
                    report.model_dump(mode="json"),
                )

        return HermesWorkflowResult(
            output_file=output_file,
            report_file=report_file,
            message="Bass make-MIDI workflow complete.",
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
        log: Callable[[str], None],
    ) -> HermesWorkflowResult:
        if not self.drums_extraction_available:
            raise HermesGuiWorkflowError("Audio-driven drums extraction is not implemented yet.")

        resolved_output_dir = output_dir if output_dir is not None else output_file.parent
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        if clean_output_folder:
            self._safe_clean_drums_output_dir(resolved_output_dir, log)

        resolved_report = report_file if report_file is not None else (resolved_output_dir / "drums_layers_report.json")
        resolved_debug_csv = debug_csv_file if debug_csv_file is not None else (resolved_output_dir / "drums_layers_hits.csv")

        resolved_mapping_file = mapping_file
        temp_mapping_file: Path | None = None
        if mapping_payload is not None:
            temp_mapping_file = resolved_output_dir / ".hermes_gui_current_mapping.json"
            temp_mapping_file.write_text(json.dumps(mapping_payload, indent=2) + "\n", encoding="utf-8")
            resolved_mapping_file = temp_mapping_file

        try:
            resolved_mapping = self._load_or_build_mapping(
                mapping_file=resolved_mapping_file,
                mapping_payload=None,
                target_map=target_map,
                c1_midi_note=c1_midi_note,
            )
        except DrumLayerMappingError as exc:
            raise HermesGuiWorkflowError(str(exc)) from exc

        if resolved_mapping_file is not None and temp_mapping_file is None:
            save_layer_mapping(resolved_mapping, resolved_mapping_file)

        if resolved_mapping_file is None:
            temp_mapping_file = resolved_output_dir / ".hermes_gui_current_mapping.json"
            save_layer_mapping(resolved_mapping, temp_mapping_file)
            resolved_mapping_file = temp_mapping_file

        params = AudioDrumExtractionParameters(
            output_file=output_file,
            output_dir=resolved_output_dir,
            target_map=target_map,
            bpm=bpm_override,
            channel=9,
            min_onset_strength=0.20,
            profile=profile,
            detection_mode=detection_mode,
            output_layout=output_layout,
            c1_midi_note=c1_midi_note,
            mapping_file=resolved_mapping_file,
            dry_run=False,
            separate_files=False,
            write_empty_layers=write_empty_layers,
            report_file=resolved_report,
            debug_csv=resolved_debug_csv,
            snare_target="clap",
        )

        log("Running audio-driven drums extraction workflow.")
        try:
            report = extract_drums_from_audio(wav_file=wav_file, params=params)
        except AudioDrumExtractionError as exc:
            raise HermesGuiWorkflowError(str(exc)) from exc
        finally:
            if temp_mapping_file is not None:
                temp_mapping_file.unlink(missing_ok=True)

        created_files = tuple(Path(path) for path in report.created_files)
        disabled_layers = tuple(str(layer) for layer in report.disabled_layers)

        return HermesWorkflowResult(
            output_file=(None if report.output_file is None else Path(report.output_file)),
            report_file=(None if resolved_report is None else resolved_report),
            output_dir=resolved_output_dir,
            debug_csv_file=resolved_debug_csv,
            created_files=created_files,
            warnings=tuple(report.warnings),
            mapping_name=report.mapping_name,
            duplicate_target_notes=dict(report.duplicate_target_notes),
            layer_counts=dict(report.layer_counts),
            populated_semantic_layers=tuple(report.populated_semantic_layers),
            unpopulated_enabled_layers=tuple(report.unpopulated_enabled_layers),
            disabled_layers=disabled_layers,
            output_layout=report.output_layout,
            message="Drums extracted from WAV.",
        )

    def synchronize_midi_with_wav(
        self,
        wav_file: Path,
        midi_file: Path,
        role: str,
        output_file: Path,
        report_file: Path | None,
        bpm_override: float | None,
        log: Callable[[str], None],
    ) -> HermesWorkflowResult:
        log("Running Hermes MIDI synchronization workflow.")
        params = MidiSyncWithAudioParameters(
            source="gui",
            layer=role,
            bpm_override=bpm_override,
        )
        try:
            sync_report, _aligned_document, alignment_report = sync_midi_with_wav(
                input_midi=midi_file,
                input_wav=wav_file,
                output_midi=output_file,
                params=params,
            )
        except MidiSyncWithAudioError as exc:
            raise HermesGuiWorkflowError(str(exc)) from exc

        if report_file is not None:
            self._write_json(
                report_file,
                {
                    "sync_report": sync_report.model_dump(mode="json"),
                    "alignment_report": alignment_report,
                },
            )

        return HermesWorkflowResult(
            output_file=output_file,
            report_file=report_file,
            message="MIDI synchronized with WAV.",
        )

    def set_midi_bpm(
        self,
        midi_file: Path,
        bpm: float,
        output_file: Path,
        report_file: Path | None,
        log: Callable[[str], None],
    ) -> HermesWorkflowResult:
        log(f"Setting MIDI BPM to {bpm:.3f}.")
        try:
            report = set_midi_bpm_file(input_file=midi_file, output_file=output_file, bpm=bpm)
        except MidiSetBpmError as exc:
            raise HermesGuiWorkflowError(str(exc)) from exc

        if report_file is not None:
            self._write_json(report_file, report.model_dump(mode="json"))

        return HermesWorkflowResult(
            output_file=output_file,
            report_file=report_file,
            message="MIDI BPM update complete.",
        )
