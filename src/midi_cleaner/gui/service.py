from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from midi_cleaner.drums.extract_audio import (
    AudioDrumExtractionError,
    AudioDrumExtractionParameters,
    extract_drums_from_audio,
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
    output_file: Path
    report_file: Path | None
    message: str


class HermesWorkflowService:
    def __init__(self) -> None:
        self.drums_extraction_available = True

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

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
        log: Callable[[str], None],
    ) -> HermesWorkflowResult:
        if not self.drums_extraction_available:
            raise HermesGuiWorkflowError("Audio-driven drums extraction is not implemented yet.")

        params = AudioDrumExtractionParameters(
            output_file=output_file,
            output_dir=output_file.parent,
            target_map="gm",
            bpm=bpm_override,
            channel=9,
            min_onset_strength=0.20,
            profile="conservative",
            dry_run=False,
            separate_files=False,
            output_layout="separate-files",
            write_empty_layers=False,
            report_file=report_file,
            snare_target="clap",
        )

        log("Running audio-driven drums extraction workflow.")
        try:
            extract_drums_from_audio(wav_file=wav_file, params=params)
        except AudioDrumExtractionError as exc:
            raise HermesGuiWorkflowError(str(exc)) from exc

        return HermesWorkflowResult(
            output_file=output_file,
            report_file=report_file,
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
