from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import mido
from pydantic import BaseModel

from midi_cleaner.alignment.audio_time import (
    AudioTimeAlignmentParameters,
    align_notes_to_audio_time,
)
from midi_cleaner.alignment.models import AudioAlignedNoteDocument
from midi_cleaner.audio.analyzer import analyze_stem
from midi_cleaner.midi.importer import import_midi_candidate
from midi_cleaner.midi.models import TempoEvent


class MidiSyncWithAudioError(Exception):
    """Raised when MIDI synchronization with WAV fails."""


class MidiSyncWithAudioReport(BaseModel):
    input_midi_file: str
    input_wav_file: str
    output_midi_file: str
    status: Literal["ok", "error"]
    source: str
    layer: str
    bpm_override: float | None
    tempo_preserved: bool
    note_count: int
    aligned_count: int
    keep_original_count: int
    review_timing_count: int
    no_audio_evidence_count: int
    alignment_report_file: str | None
    warning_count: int
    warnings: list[str]


@dataclass(frozen=True)
class MidiSyncWithAudioParameters:
    source: str = "ripx"
    layer: str = "bass"
    bpm_override: float | None = None
    onset_search_window_ms: float = 250.0
    offset_search_window_ms: float = 350.0
    min_onset_score: float = 0.005
    min_rms: float = 0.001
    snap_start_to_audio_onset: bool = True
    snap_end_to_energy_offset: bool = True
    max_start_correction_ms: float = 500.0
    max_end_correction_ms: float = 800.0
    low_confidence_action: str = "KEEP_ORIGINAL_LOW_CONFIDENCE"


def _seconds_to_tick(
    seconds: float,
    tempo_map: list[TempoEvent],
    ticks_per_beat: int,
) -> int:
    if not tempo_map:
        return max(0, int(round(seconds * ticks_per_beat * 2.0)))

    target_sec = max(0.0, float(seconds))
    candidate = tempo_map[0]
    for event in tempo_map:
        if float(event.sec) <= target_sec:
            candidate = event
        else:
            break

    delta_sec = max(0.0, target_sec - float(candidate.sec))
    ticks_per_second = (float(ticks_per_beat) * 1_000_000.0) / float(candidate.tempo_us_per_beat)
    tick = int(round(float(candidate.tick) + (delta_sec * ticks_per_second)))
    return max(0, tick)


def _resolve_output_tempo_map(
    source_tempo_map: list[TempoEvent],
    bpm_override: float | None,
) -> tuple[list[TempoEvent], bool]:
    if bpm_override is None:
        return list(source_tempo_map), True

    if bpm_override <= 0:
        raise MidiSyncWithAudioError("BPM override must be greater than 0.")

    tempo_us_per_beat = int(round(60_000_000.0 / float(bpm_override)))
    return [TempoEvent(tick=0, tempo_us_per_beat=tempo_us_per_beat, sec=0.0)], False


def _build_synchronized_midi(
    aligned_document: AudioAlignedNoteDocument,
    output_path: Path,
    ticks_per_beat: int,
    tempo_map: list[TempoEvent],
) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=int(ticks_per_beat))
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="Hermes Synced", time=0))

    absolute_events: list[tuple[int, int, mido.Message | mido.MetaMessage]] = []
    for event in tempo_map:
        absolute_events.append(
            (
                int(event.tick),
                0,
                mido.MetaMessage("set_tempo", tempo=int(event.tempo_us_per_beat), time=0),
            )
        )

    for note in aligned_document.notes:
        start_tick = _seconds_to_tick(
            seconds=float(note.aligned_start_sec),
            tempo_map=tempo_map,
            ticks_per_beat=ticks_per_beat,
        )
        end_tick = _seconds_to_tick(
            seconds=float(note.aligned_end_sec),
            tempo_map=tempo_map,
            ticks_per_beat=ticks_per_beat,
        )
        if end_tick <= start_tick:
            end_tick = start_tick + 1

        channel = 0 if note.channel is None else int(note.channel)

        absolute_events.append(
            (
                start_tick,
                2,
                mido.Message(
                    "note_on",
                    note=int(note.pitch_midi),
                    velocity=int(note.velocity),
                    channel=channel,
                    time=0,
                ),
            )
        )
        absolute_events.append(
            (
                end_tick,
                1,
                mido.Message(
                    "note_off",
                    note=int(note.pitch_midi),
                    velocity=0,
                    channel=channel,
                    time=0,
                ),
            )
        )

    absolute_events.sort(key=lambda item: (item[0], item[1]))

    previous_tick = 0
    for tick, _order, message in absolute_events:
        message.time = max(0, int(tick) - previous_tick)
        previous_tick = int(tick)
        track.append(message)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(output_path))


def sync_midi_with_wav(
    input_midi: Path,
    input_wav: Path,
    output_midi: Path,
    params: MidiSyncWithAudioParameters,
) -> tuple[MidiSyncWithAudioReport, AudioAlignedNoteDocument, dict[str, object]]:
    if not input_midi.exists() or not input_midi.is_file():
        raise MidiSyncWithAudioError(f"Input MIDI file does not exist: {input_midi}")
    if not input_wav.exists() or not input_wav.is_file():
        raise MidiSyncWithAudioError(f"Input WAV file does not exist: {input_wav}")

    try:
        note_document, _import_report = import_midi_candidate(
            input_midi,
            source=params.source,
            layer=params.layer,
        )
    except Exception as exc:
        raise MidiSyncWithAudioError(f"MIDI import failed: {exc}") from exc

    try:
        audio_document, _audio_report = analyze_stem(input_wav=input_wav, layer=params.layer)
    except Exception as exc:
        raise MidiSyncWithAudioError(f"WAV analysis failed: {exc}") from exc

    with TemporaryDirectory(prefix="hermes_sync_") as temp_dir:
        temp_root = Path(temp_dir)
        notes_path = temp_root / "note_events.json"
        audio_path = temp_root / "audio_features.json"

        notes_path.write_text(note_document.model_dump_json(indent=2) + "\n", encoding="utf-8")
        audio_path.write_text(audio_document.model_dump_json(indent=2) + "\n", encoding="utf-8")

        try:
            aligned_document, alignment_report = align_notes_to_audio_time(
                notes_file=notes_path,
                audio_features_file=audio_path,
                params=AudioTimeAlignmentParameters(
                    onset_search_window_ms=params.onset_search_window_ms,
                    offset_search_window_ms=params.offset_search_window_ms,
                    min_onset_score=params.min_onset_score,
                    min_rms=params.min_rms,
                    snap_start_to_audio_onset=params.snap_start_to_audio_onset,
                    snap_end_to_energy_offset=params.snap_end_to_energy_offset,
                    max_start_correction_ms=params.max_start_correction_ms,
                    max_end_correction_ms=params.max_end_correction_ms,
                    low_confidence_action=params.low_confidence_action,
                ),
            )
        except Exception as exc:
            raise MidiSyncWithAudioError(f"Audio-time alignment failed: {exc}") from exc

    output_tempo_map, tempo_preserved = _resolve_output_tempo_map(
        source_tempo_map=note_document.tempo_map,
        bpm_override=params.bpm_override,
    )

    _build_synchronized_midi(
        aligned_document=aligned_document,
        output_path=output_midi,
        ticks_per_beat=int(note_document.ticks_per_beat),
        tempo_map=output_tempo_map,
    )

    report = MidiSyncWithAudioReport(
        input_midi_file=str(input_midi),
        input_wav_file=str(input_wav),
        output_midi_file=str(output_midi),
        status="ok",
        source=params.source,
        layer=params.layer,
        bpm_override=params.bpm_override,
        tempo_preserved=tempo_preserved,
        note_count=int(alignment_report.note_count),
        aligned_count=int(alignment_report.aligned_count),
        keep_original_count=int(alignment_report.keep_original_count),
        review_timing_count=int(alignment_report.review_timing_count),
        no_audio_evidence_count=int(alignment_report.no_audio_evidence_count),
        alignment_report_file=None,
        warning_count=int(alignment_report.warning_count),
        warnings=list(alignment_report.warnings),
    )

    return report, aligned_document, alignment_report.model_dump(mode="json")
