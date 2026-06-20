from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import mido

from midi_cleaner.midi.drum_maps import (
    DrumMapDefinition,
    DrumMapError,
    load_custom_drum_map,
    load_preset_drum_map,
)

ChannelPolicy = Literal["preserve", "single"]
OutputFormat = Literal["type0", "single-track-type1"]
UnmappedPolicy = Literal["keep", "drop", "nearest"]

DRUM_CHANNEL_ZERO_BASED = 9


class MidiRemapDrumsError(Exception):
    """Raised when drum remap processing cannot continue."""


@dataclass(frozen=True)
class DrumRemapParameters:
    target_map: str
    output_file: Path | None = None
    map_file: Path | None = None
    merge_tracks: bool = True
    channel_policy: ChannelPolicy = "single"
    force_channel: int | None = None
    unmapped_policy: UnmappedPolicy = "keep"
    strip_program_changes: bool = True
    strip_track_names: bool = True
    c1_midi_note: int = 36
    output_format: OutputFormat = "type0"
    dry_run: bool = False
    report_file: Path | None = None


@dataclass
class DrumRemapReport:
    input_file: str
    output_file: str | None
    input_type: int
    output_type: int
    source_track_count: int
    output_track_count: int
    ticks_per_beat: int
    source_length_ticks: int
    output_length_ticks: int
    tempo_event_count: int
    time_signature_event_count: int
    source_pitch_counts: dict[str, int]
    remapped_pitch_counts: dict[str, int]
    source_channels: list[int]
    output_channels: list[int]
    target_map: str
    target_key_layout_name: str
    c1_midi_note: int
    resolved_target_note_names: dict[str, str]
    resolved_mapping_note_numbers: dict[str, int]
    map_file: str | None
    unmapped_policy: UnmappedPolicy
    unmapped_pitches: list[int]
    stripped_program_changes: int
    merged_tracks: bool
    synchronization_preserved: bool
    warnings: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _EventRecord:
    absolute_tick: int
    priority: int
    track_index: int
    order_index: int
    message: mido.Message | mido.MetaMessage


@dataclass
class _RemapContext:
    source_pitch_counts: dict[int, int] = field(default_factory=dict)
    remapped_pitch_counts: dict[int, int] = field(default_factory=dict)
    source_channels: set[int] = field(default_factory=set)
    output_channels: set[int] = field(default_factory=set)
    unmapped_pitches: set[int] = field(default_factory=set)
    stripped_program_changes: int = 0
    warnings: list[str] = field(default_factory=list)


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


def _is_drum_note_message(message: mido.Message) -> bool:
    if message.type not in {"note_on", "note_off"}:
        return False
    return int(getattr(message, "channel", -1)) == DRUM_CHANNEL_ZERO_BASED


def _is_real_note_on(message: mido.Message) -> bool:
    return message.type == "note_on" and int(getattr(message, "velocity", 0)) > 0


def _nearest_note(source_note: int, known_notes: set[int]) -> int:
    if not known_notes:
        return source_note

    def _sort_key(candidate: int) -> tuple[int, int]:
        return (abs(candidate - source_note), candidate)

    return min(known_notes, key=_sort_key)


def _effective_output_path(input_file: Path, explicit_output: Path | None, target_map: str) -> Path:
    if explicit_output is not None:
        return explicit_output

    suffix = target_map.replace("-", "_")
    return input_file.with_name(f"{input_file.stem}_{suffix}{input_file.suffix}")


def _resolve_output_channel(
    *,
    params: DrumRemapParameters,
    map_definition: DrumMapDefinition,
) -> int | None:
    if params.channel_policy == "preserve":
        if params.force_channel is None:
            return None
        return params.force_channel

    if params.force_channel is not None:
        return params.force_channel

    return map_definition.output_channel


def _validate_params(params: DrumRemapParameters) -> None:
    if params.target_map not in {"gm", "sitala", "ujam-candy", "custom"}:
        raise MidiRemapDrumsError(
            "Invalid --target-map. Use gm, sitala, ujam-candy, or custom."
        )

    if params.channel_policy not in {"preserve", "single"}:
        raise MidiRemapDrumsError("Invalid --channel-policy. Use preserve or single.")

    if params.unmapped_policy not in {"keep", "drop", "nearest"}:
        raise MidiRemapDrumsError("Invalid --unmapped. Use keep, drop, or nearest.")

    if params.output_format not in {"type0", "single-track-type1"}:
        raise MidiRemapDrumsError("Invalid --format. Use type0 or single-track-type1.")

    if params.force_channel is not None and (params.force_channel < 0 or params.force_channel > 15):
        raise MidiRemapDrumsError("--force-channel must be in range 0..15.")

    if params.target_map == "custom" and params.map_file is None:
        raise MidiRemapDrumsError("--map-file is required when --target-map custom.")

    if params.c1_midi_note < 0 or params.c1_midi_note > 127:
        raise MidiRemapDrumsError("--c1-midi-note must be in range 0..127.")


def _load_map_definition(params: DrumRemapParameters) -> DrumMapDefinition:
    try:
        if params.target_map == "custom":
            assert params.map_file is not None
            return load_custom_drum_map(params.map_file)
        return load_preset_drum_map(params.target_map, c1_midi_note=params.c1_midi_note)
    except DrumMapError as exc:
        raise MidiRemapDrumsError(str(exc)) from exc


def _accumulate_source_audit(
    source_midi: mido.MidiFile,
    context: _RemapContext,
) -> None:
    for track in source_midi.tracks:
        for message in track:
            if message.is_meta:
                continue
            if hasattr(message, "channel"):
                context.source_channels.add(int(message.channel))
            if _is_drum_note_message(message) and _is_real_note_on(message):
                note = int(message.note)
                context.source_pitch_counts[note] = context.source_pitch_counts.get(note, 0) + 1


def _remap_note(
    note: int,
    *,
    map_definition: DrumMapDefinition,
    unmapped_policy: UnmappedPolicy,
    context: _RemapContext,
) -> int | None:
    if note in map_definition.notes:
        return int(map_definition.notes[note])

    context.unmapped_pitches.add(note)
    if unmapped_policy == "drop":
        return None

    if unmapped_policy == "keep":
        return note

    nearest_source = _nearest_note(note, set(map_definition.notes))
    return int(map_definition.notes.get(nearest_source, note))


def _copy_message_with_policy(
    message: mido.Message | mido.MetaMessage,
    *,
    map_definition: DrumMapDefinition,
    params: DrumRemapParameters,
    output_channel: int | None,
    context: _RemapContext,
) -> mido.Message | mido.MetaMessage | None:
    copied = message.copy(time=0)

    if copied.is_meta:
        if params.strip_track_names and copied.type == "track_name":
            return None
        if copied.type == "end_of_track":
            return None
        return copied

    if copied.type == "program_change" and params.strip_program_changes:
        context.stripped_program_changes += 1
        return None

    if _is_drum_note_message(copied):
        mapped_note = _remap_note(
            int(copied.note),
            map_definition=map_definition,
            unmapped_policy=params.unmapped_policy,
            context=context,
        )
        if mapped_note is None:
            return None
        copied.note = mapped_note
        if _is_real_note_on(copied):
            context.remapped_pitch_counts[mapped_note] = (
                context.remapped_pitch_counts.get(mapped_note, 0) + 1
            )

    if output_channel is not None and hasattr(copied, "channel"):
        copied.channel = output_channel

    if hasattr(copied, "channel"):
        context.output_channels.add(int(copied.channel))

    return copied


def _flatten_events(
    source_midi: mido.MidiFile,
    *,
    map_definition: DrumMapDefinition,
    params: DrumRemapParameters,
    output_channel: int | None,
    context: _RemapContext,
) -> list[_EventRecord]:
    events: list[_EventRecord] = []

    for track_index, track in enumerate(source_midi.tracks):
        absolute_tick = 0
        for order_index, message in enumerate(track):
            absolute_tick += int(message.time)

            copied = _copy_message_with_policy(
                message,
                map_definition=map_definition,
                params=params,
                output_channel=output_channel,
                context=context,
            )
            if copied is None:
                continue

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
    return events


def _build_single_track_output(
    events: list[_EventRecord],
    *,
    source_midi: mido.MidiFile,
    track_name: str,
    output_format: OutputFormat,
) -> mido.MidiFile:
    output_track = mido.MidiTrack()
    output_track.append(mido.MetaMessage("track_name", name=track_name, time=0))

    previous_tick = 0
    for record in events:
        delta = record.absolute_tick - previous_tick
        output_track.append(record.message.copy(time=delta))
        previous_tick = record.absolute_tick

    source_length = _source_length_ticks(source_midi)
    final_tick = max(source_length, previous_tick)
    output_track.append(mido.MetaMessage("end_of_track", time=final_tick - previous_tick))

    midi_type = 0 if output_format == "type0" else 1
    output_midi = mido.MidiFile(type=midi_type, ticks_per_beat=source_midi.ticks_per_beat)
    output_midi.tracks.append(output_track)
    return output_midi


def _build_preserved_tracks_output(
    source_midi: mido.MidiFile,
    *,
    map_definition: DrumMapDefinition,
    params: DrumRemapParameters,
    output_channel: int | None,
    context: _RemapContext,
) -> mido.MidiFile:
    output_type = 1 if source_midi.type == 1 else 0
    output_midi = mido.MidiFile(type=output_type, ticks_per_beat=source_midi.ticks_per_beat)

    for source_track in source_midi.tracks:
        out_track = mido.MidiTrack()
        absolute_tick = 0
        kept_messages: list[tuple[int, mido.Message | mido.MetaMessage]] = []

        for message in source_track:
            absolute_tick += int(message.time)
            copied = _copy_message_with_policy(
                message,
                map_definition=map_definition,
                params=params,
                output_channel=output_channel,
                context=context,
            )
            if copied is None:
                continue
            kept_messages.append((absolute_tick, copied))

        previous_tick = 0
        for absolute_message_tick, copied in kept_messages:
            delta = absolute_message_tick - previous_tick
            out_track.append(copied.copy(time=delta))
            previous_tick = absolute_message_tick

        out_track.append(mido.MetaMessage("end_of_track", time=absolute_tick - previous_tick))
        output_midi.tracks.append(out_track)

    if not output_midi.tracks:
        output_track = mido.MidiTrack()
        output_track.append(mido.MetaMessage("track_name", name="drums", time=0))
        output_track.append(mido.MetaMessage("end_of_track", time=0))
        output_midi.tracks.append(output_track)

    return output_midi


def _count_meta_events(midi_file: mido.MidiFile) -> tuple[int, int]:
    tempo_count = 0
    time_signature_count = 0
    for track in midi_file.tracks:
        for message in track:
            if not message.is_meta:
                continue
            if message.type == "set_tempo":
                tempo_count += 1
            elif message.type == "time_signature":
                time_signature_count += 1
    return tempo_count, time_signature_count


def _max_track_tick(track: mido.MidiTrack) -> int:
    tick = 0
    for message in track:
        tick += int(message.time)
    return tick


def _build_report(
    *,
    input_file: Path,
    output_file: Path | None,
    source_midi: mido.MidiFile,
    output_midi: mido.MidiFile,
    params: DrumRemapParameters,
    map_definition: DrumMapDefinition,
    context: _RemapContext,
) -> DrumRemapReport:
    source_length = _source_length_ticks(source_midi)
    output_length = _source_length_ticks(output_midi)
    tempo_count, time_signature_count = _count_meta_events(source_midi)
    sync_preserved = (
        int(source_midi.ticks_per_beat) == int(output_midi.ticks_per_beat)
        and source_length == output_length
    )

    warnings = list(context.warnings)
    if context.unmapped_pitches and params.unmapped_policy == "keep":
        warnings.append(
            "Unmapped pitches were kept."
        )
    if context.unmapped_pitches and params.unmapped_policy == "nearest":
        warnings.append(
            "Unmapped pitches were remapped with nearest policy."
        )

    if not sync_preserved:
        warnings.append("Output MIDI timing differs from source timing.")

    resolved_target_note_names = {
        str(source_note): target_note_name
        for source_note, target_note_name in sorted(map_definition.target_note_names.items())
    }
    resolved_mapping_note_numbers = {
        str(source_note): target_note
        for source_note, target_note in sorted(map_definition.notes.items())
    }

    return DrumRemapReport(
        input_file=str(input_file),
        output_file=str(output_file) if output_file is not None else None,
        input_type=int(source_midi.type),
        output_type=int(output_midi.type),
        source_track_count=len(source_midi.tracks),
        output_track_count=len(output_midi.tracks),
        ticks_per_beat=int(source_midi.ticks_per_beat),
        source_length_ticks=source_length,
        output_length_ticks=output_length,
        tempo_event_count=tempo_count,
        time_signature_event_count=time_signature_count,
        source_pitch_counts={str(k): v for k, v in sorted(context.source_pitch_counts.items())},
        remapped_pitch_counts={str(k): v for k, v in sorted(context.remapped_pitch_counts.items())},
        source_channels=sorted(context.source_channels),
        output_channels=sorted(context.output_channels),
        target_map=map_definition.name,
        target_key_layout_name=map_definition.key_layout_name or map_definition.name,
        c1_midi_note=params.c1_midi_note,
        resolved_target_note_names=resolved_target_note_names,
        resolved_mapping_note_numbers=resolved_mapping_note_numbers,
        map_file=str(params.map_file) if params.map_file is not None else None,
        unmapped_policy=params.unmapped_policy,
        unmapped_pitches=sorted(context.unmapped_pitches),
        stripped_program_changes=context.stripped_program_changes,
        merged_tracks=params.merge_tracks,
        synchronization_preserved=sync_preserved,
        warnings=warnings,
    )


def _ensure_final_end_of_track_sanity(midi_file: mido.MidiFile) -> None:
    for track in midi_file.tracks:
        if not track:
            track.append(mido.MetaMessage("end_of_track", time=0))
            continue

        # Ensure exactly one terminal end_of_track event.
        while (
            len(track) > 1
            and track[-1].is_meta
            and track[-1].type == "end_of_track"
            and track[-2].is_meta
            and track[-2].type == "end_of_track"
        ):
            track.pop()

        if not (track[-1].is_meta and track[-1].type == "end_of_track"):
            track.append(mido.MetaMessage("end_of_track", time=0))


def remap_drums_file(
    *,
    input_file: Path,
    params: DrumRemapParameters,
) -> DrumRemapReport:
    _validate_params(params)

    if not input_file.exists() or not input_file.is_file():
        raise MidiRemapDrumsError(f"Input MIDI file does not exist: {input_file}")

    try:
        source_midi = mido.MidiFile(str(input_file))
    except Exception as exc:  # pragma: no cover
        raise MidiRemapDrumsError(f"Failed to parse MIDI file: {input_file}") from exc

    map_definition = _load_map_definition(params)
    output_channel = _resolve_output_channel(params=params, map_definition=map_definition)

    context = _RemapContext()
    _accumulate_source_audit(source_midi, context)

    use_single_track_output = params.merge_tracks or params.output_format == "type0"

    if use_single_track_output:
        events = _flatten_events(
            source_midi,
            map_definition=map_definition,
            params=params,
            output_channel=output_channel,
            context=context,
        )
        if not params.merge_tracks and params.output_format == "type0" and len(source_midi.tracks) > 1:
            context.warnings.append(
                "Output format type0 requires a single output track; tracks were merged."
            )
        output_track_name = f"drums_{params.target_map.replace('-', '_')}"
        output_midi = _build_single_track_output(
            events,
            source_midi=source_midi,
            track_name=output_track_name,
            output_format=params.output_format,
        )
    else:
        output_midi = _build_preserved_tracks_output(
            source_midi,
            map_definition=map_definition,
            params=params,
            output_channel=output_channel,
            context=context,
        )

    _ensure_final_end_of_track_sanity(output_midi)

    output_file = _effective_output_path(input_file, params.output_file, params.target_map)
    report = _build_report(
        input_file=input_file,
        output_file=output_file,
        source_midi=source_midi,
        output_midi=output_midi,
        params=params,
        map_definition=map_definition,
        context=context,
    )

    if params.dry_run:
        report.output_file = None
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_midi.save(str(output_file))

    if params.report_file is not None:
        params.report_file.parent.mkdir(parents=True, exist_ok=True)
        params.report_file.write_text(
            json.dumps(report.to_json_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    return report