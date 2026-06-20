from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

import mido

ChannelPolicy = Literal["preserve", "single"]
OutputFormat = Literal["type0", "single-track-type1"]
MergeAction = Literal["merged", "skipped_single_track", "skipped_user", "dry_run"]


class MidiMergeFolderError(Exception):
    """Raised when merge-folder processing cannot continue."""


@dataclass
class MergeFolderFileReport:
    source_file: str
    output_file: str | None
    source_track_count: int
    output_track_count: int
    source_ticks_per_beat: int
    output_ticks_per_beat: int
    source_length_ticks: int
    output_length_ticks: int
    note_on_count: int
    note_off_count: int
    tempo_event_count: int
    time_signature_event_count: int
    channel_policy: ChannelPolicy
    action: MergeAction
    warnings: list[str] = field(default_factory=list)


@dataclass
class MergeFolderRunReport:
    scanned_folder: str
    recursive: bool
    midi_file_count: int
    multitrack_file_count: int
    merged_file_count: int
    skipped_file_count: int
    dry_run: bool
    files: list[MergeFolderFileReport]

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["files"] = [asdict(item) for item in self.files]
        return payload


@dataclass
class _EventRecord:
    absolute_tick: int
    priority: int
    track_index: int
    order_index: int
    message: mido.Message | mido.MetaMessage


def _iter_midi_files(folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        candidates = list(folder.rglob("*"))
    else:
        candidates = list(folder.glob("*"))

    midi_files = [
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    ]
    return sorted(midi_files)


def _message_priority(message: mido.Message | mido.MetaMessage) -> int:
    if message.is_meta:
        if message.type in {"set_tempo", "time_signature", "key_signature"}:
            return 1
        return 2

    if message.type in {
        "program_change",
        "control_change",
        "polytouch",
        "aftertouch",
        "pitchwheel",
        "sysex",
    }:
        return 3

    if message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
        return 4

    if message.type == "note_on":
        return 5

    return 3


def _source_length_ticks(midi_file: mido.MidiFile) -> int:
    max_tick = 0
    for track in midi_file.tracks:
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
        max_tick = max(max_tick, absolute_tick)
    return max_tick


def _count_note_events(messages: list[mido.Message | mido.MetaMessage]) -> tuple[int, int]:
    note_on_count = 0
    note_off_count = 0
    for message in messages:
        if message.is_meta:
            continue
        if message.type == "note_on" and message.velocity > 0:
            note_on_count += 1
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            note_off_count += 1
    return note_on_count, note_off_count


def _count_meta_events(messages: list[mido.Message | mido.MetaMessage]) -> tuple[int, int]:
    tempo = 0
    time_sig = 0
    for message in messages:
        if not message.is_meta:
            continue
        if message.type == "set_tempo":
            tempo += 1
        elif message.type == "time_signature":
            time_sig += 1
    return tempo, time_sig


def _flatten_source_events(midi_file: mido.MidiFile) -> list[mido.Message | mido.MetaMessage]:
    flattened: list[mido.Message | mido.MetaMessage] = []
    for track in midi_file.tracks:
        for message in track:
            flattened.append(message)
    return flattened


def _merged_output_path(source_file: Path, suffix: str) -> Path:
    if not suffix:
        suffix = "_merge"

    base = source_file.stem
    extension = source_file.suffix
    candidate = source_file.with_name(f"{base}{suffix}{extension}")
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = source_file.with_name(f"{base}{suffix}_{index}{extension}")
        if not candidate.exists():
            return candidate
        index += 1


def merge_multitrack_midi(
    midi_file: mido.MidiFile,
    *,
    channel_policy: ChannelPolicy,
    output_format: OutputFormat,
) -> tuple[mido.MidiFile, int]:
    events: list[_EventRecord] = []

    for track_index, track in enumerate(midi_file.tracks):
        absolute_tick = 0
        for order_index, message in enumerate(track):
            absolute_tick += int(message.time)

            if message.is_meta and message.type in {"end_of_track", "track_name"}:
                continue

            copied = message.copy(time=0)
            if (
                channel_policy == "single"
                and not copied.is_meta
                and hasattr(copied, "channel")
            ):
                copied.channel = 0

            events.append(
                _EventRecord(
                    absolute_tick=absolute_tick,
                    priority=_message_priority(copied),
                    track_index=track_index,
                    order_index=order_index,
                    message=copied,
                )
            )

    events.sort(
        key=lambda item: (
            item.absolute_tick,
            item.priority,
            item.track_index,
            item.order_index,
        )
    )

    merged_track = mido.MidiTrack()
    merged_track.append(mido.MetaMessage("track_name", name="merged", time=0))

    previous_tick = 0
    for record in events:
        delta = record.absolute_tick - previous_tick
        merged_track.append(record.message.copy(time=delta))
        previous_tick = record.absolute_tick

    source_length_ticks = _source_length_ticks(midi_file)
    final_tick = max(source_length_ticks, previous_tick)
    merged_track.append(mido.MetaMessage("end_of_track", time=final_tick - previous_tick))

    midi_type = 0 if output_format == "type0" else 1
    merged_file = mido.MidiFile(type=midi_type, ticks_per_beat=midi_file.ticks_per_beat)
    merged_file.tracks.append(merged_track)

    return merged_file, source_length_ticks


def merge_midi_folder(
    *,
    folder: Path,
    recursive: bool,
    yes: bool,
    dry_run: bool,
    channel_policy: ChannelPolicy,
    output_suffix: str,
    output_format: OutputFormat,
    on_detect_multitrack: Callable[[Path, int], None] | None = None,
    prompt_merge: Callable[[Path, int], bool] | None = None,
) -> MergeFolderRunReport:
    if not folder.exists() or not folder.is_dir():
        raise MidiMergeFolderError(f"Folder does not exist or is not a directory: {folder}")

    midi_files = _iter_midi_files(folder, recursive=recursive)
    multitrack_file_count = 0
    merged_file_count = 0
    skipped_file_count = 0
    file_reports: list[MergeFolderFileReport] = []

    for source_file in midi_files:
        try:
            source_midi = mido.MidiFile(str(source_file))
        except Exception as exc:  # pragma: no cover
            raise MidiMergeFolderError(f"Failed to parse MIDI file: {source_file}") from exc

        source_track_count = len(source_midi.tracks)
        source_ticks_per_beat = int(source_midi.ticks_per_beat)
        source_flattened = _flatten_source_events(source_midi)
        source_note_on_count, source_note_off_count = _count_note_events(source_flattened)
        source_tempo_count, source_time_sig_count = _count_meta_events(source_flattened)
        source_duration_ticks = _source_length_ticks(source_midi)

        if source_track_count <= 1:
            skipped_file_count += 1
            file_reports.append(
                MergeFolderFileReport(
                    source_file=str(source_file),
                    output_file=None,
                    source_track_count=source_track_count,
                    output_track_count=source_track_count,
                    source_ticks_per_beat=source_ticks_per_beat,
                    output_ticks_per_beat=source_ticks_per_beat,
                    source_length_ticks=source_duration_ticks,
                    output_length_ticks=source_duration_ticks,
                    note_on_count=source_note_on_count,
                    note_off_count=source_note_off_count,
                    tempo_event_count=source_tempo_count,
                    time_signature_event_count=source_time_sig_count,
                    channel_policy=channel_policy,
                    action="skipped_single_track",
                )
            )
            continue

        multitrack_file_count += 1
        if on_detect_multitrack is not None:
            on_detect_multitrack(source_file, source_track_count)

        should_merge = False
        if dry_run:
            skipped_file_count += 1
            action: MergeAction = "dry_run"
        elif yes:
            should_merge = True
            action = "merged"
        elif prompt_merge is not None and prompt_merge(source_file, source_track_count):
            should_merge = True
            action = "merged"
        else:
            skipped_file_count += 1
            action = "skipped_user"

        output_file: Path | None = None
        output_track_count = source_track_count
        output_ticks_per_beat = source_ticks_per_beat
        output_length_ticks = source_duration_ticks
        output_note_on_count = source_note_on_count
        output_note_off_count = source_note_off_count
        output_tempo_count = source_tempo_count
        output_time_sig_count = source_time_sig_count
        warnings: list[str] = []

        if should_merge:
            merged_midi, source_length_ticks = merge_multitrack_midi(
                source_midi,
                channel_policy=channel_policy,
                output_format=output_format,
            )
            output_file = _merged_output_path(source_file, output_suffix)
            merged_midi.save(str(output_file))
            merged_file_count += 1

            merged_flattened = _flatten_source_events(merged_midi)
            output_track_count = len(merged_midi.tracks)
            output_ticks_per_beat = int(merged_midi.ticks_per_beat)
            output_length_ticks = _source_length_ticks(merged_midi)
            output_note_on_count, output_note_off_count = _count_note_events(merged_flattened)
            output_tempo_count, output_time_sig_count = _count_meta_events(merged_flattened)

            if source_length_ticks != output_length_ticks:
                warnings.append(
                    "Merged output duration ticks differ from source duration ticks."
                )

        file_reports.append(
            MergeFolderFileReport(
                source_file=str(source_file),
                output_file=str(output_file) if output_file is not None else None,
                source_track_count=source_track_count,
                output_track_count=output_track_count,
                source_ticks_per_beat=source_ticks_per_beat,
                output_ticks_per_beat=output_ticks_per_beat,
                source_length_ticks=source_duration_ticks,
                output_length_ticks=output_length_ticks,
                note_on_count=output_note_on_count,
                note_off_count=output_note_off_count,
                tempo_event_count=output_tempo_count,
                time_signature_event_count=output_time_sig_count,
                channel_policy=channel_policy,
                action=action,
                warnings=warnings,
            )
        )

    return MergeFolderRunReport(
        scanned_folder=str(folder),
        recursive=recursive,
        midi_file_count=len(midi_files),
        multitrack_file_count=multitrack_file_count,
        merged_file_count=merged_file_count,
        skipped_file_count=skipped_file_count,
        dry_run=dry_run,
        files=file_reports,
    )
