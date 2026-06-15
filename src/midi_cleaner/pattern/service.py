from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import math
import statistics

import mido

from midi_cleaner.midi.importer import import_midi_candidate
from midi_cleaner.pattern.models import (
    BarGapCandidate,
    BarGapFamilyContext,
    IncompleteBlockMatch,
    IncompleteBlockReport,
    MissingExpectedBlock,
    PatternBlock,
    PatternBlockNote,
    PatternCompletionReport,
    PatternFamily,
    ProposedCompletionNote,
)


class PatternCompletionError(Exception):
    """Raised when deterministic pattern block completion cannot run."""


@dataclass(frozen=True)
class PatternCompletionParameters:
    layer: str = "bass"
    write_debug_midi: bool = True


@dataclass(frozen=True)
class _BaseNote:
    note_id: str
    start_tick: int
    end_tick: int
    start_sec: float
    end_sec: float
    duration_sec: float
    start_beat: float
    end_beat: float
    duration_beat: float
    pitch_midi: int
    pitch_name: str
    velocity: int
    channel: int | None


@dataclass(frozen=True)
class _ParsedMidiTiming:
    ticks_per_beat: int
    tempo_us_per_beat: int
    beats_per_bar: int
    beat_unit: int
    beat_length_sec: float
    duration_sec: float


@dataclass(frozen=True)
class _CandidateFamilyMatch:
    family: PatternFamily
    score: float
    reason: str


@dataclass(frozen=True)
class _FamilyPeriodEstimate:
    period_sec: float
    stability: float


@dataclass
class _CompletionStats:
    rejected_micro_note_count: int = 0
    rejected_polyphonic_stack_count: int = 0
    rejected_low_confidence_count: int = 0
    rejected_tiny_gap_count: int = 0


@dataclass(frozen=True)
class _LocalNoteThresholds:
    min_duration_sec: float
    min_velocity: int
    median_velocity: int


def complete_pattern_blocks(
    project_dir: Path,
    params: PatternCompletionParameters,
) -> PatternCompletionReport:
    project_dir = project_dir.resolve()
    analysis_dir = project_dir / "analysis" / "pattern_blocks"
    midi_dir = project_dir / "midi"
    debug_dir = midi_dir / "debug"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    pattern_blocks_path = analysis_dir / "pattern_blocks.json"
    pattern_families_path = analysis_dir / "pattern_families.json"
    incomplete_blocks_path = analysis_dir / "incomplete_blocks.json"
    missing_expected_blocks_path = analysis_dir / "missing_expected_blocks.json"
    bar_gap_candidates_path = analysis_dir / "bar_gap_candidates.json"
    report_path = analysis_dir / "pattern_completion_report.json"
    output_midi_path = midi_dir / "uzupelnienie.mid"
    debug_midi_path = debug_dir / "pattern_blocks_debug.mid"

    warnings: list[str] = []

    try:
        base_midi_path = _resolve_base_midi(project_dir)
        timing = _parse_midi_timing(base_midi_path)
        base_notes = _load_base_notes(base_midi_path, layer=params.layer)
        stats = _CompletionStats()

        if not base_notes:
            raise PatternCompletionError(f"No base notes found in {base_midi_path}")

        local_thresholds = _compute_local_note_thresholds(base_notes)

        blocks = _split_into_blocks(base_notes, timing=timing)
        families = _build_pattern_families(blocks)

        enriched_blocks: list[PatternBlock] = [
            _classify_block(block=block, families=families)
            for block in blocks
        ]

        blocks_by_id = {block.block_id: block for block in enriched_blocks}
        incomplete_existing_reports: list[IncompleteBlockReport] = []
        missing_expected_reports: list[IncompleteBlockReport] = []
        inserted_notes: list[ProposedCompletionNote] = []

        for block in enriched_blocks:
            if block.status != "incomplete":
                continue

            report, new_notes = _complete_incomplete_block(
                incomplete_block=block,
                blocks_by_id=blocks_by_id,
                families=families,
                base_notes=base_notes,
                local_thresholds=local_thresholds,
                stats=stats,
            )
            incomplete_existing_reports.append(report)
            inserted_notes.extend(new_notes)

        missing_expected_blocks = _detect_missing_expected_blocks(
            families=families,
            blocks_by_id=blocks_by_id,
            base_notes=base_notes,
        )
        bar_gap_candidates = _detect_bar_gap_candidates(
            project_dir=project_dir,
            blocks=enriched_blocks,
            families=families,
        )
        missing_reports, missing_inserted_notes = _complete_missing_expected_blocks(
            missing_expected_blocks=missing_expected_blocks,
            blocks_by_id=blocks_by_id,
            families=families,
            base_notes=base_notes,
            local_thresholds=local_thresholds,
            stats=stats,
        )
        missing_expected_reports.extend(missing_reports)
        inserted_notes.extend(missing_inserted_notes)

        all_reports = list(incomplete_existing_reports) + list(missing_expected_reports)

        deduped_inserted_notes = _dedupe_inserted_notes(inserted_notes, stats=stats)

        _write_json(pattern_blocks_path, [item.model_dump(mode="json") for item in enriched_blocks])
        _write_json(pattern_families_path, [item.model_dump(mode="json") for item in families])
        _write_json(incomplete_blocks_path, [item.model_dump(mode="json") for item in all_reports])
        _write_json(
            missing_expected_blocks_path,
            [item.model_dump(mode="json") for item in missing_expected_reports],
        )
        _write_json(
            bar_gap_candidates_path,
            [item.model_dump(mode="json") for item in bar_gap_candidates],
        )

        _write_completion_midi(
            output_path=output_midi_path,
            notes=deduped_inserted_notes,
            ticks_per_beat=timing.ticks_per_beat,
            tempo_us_per_beat=timing.tempo_us_per_beat,
            project_duration_sec=timing.duration_sec,
        )

        debug_path_for_report: str | None = None
        if params.write_debug_midi:
            _write_debug_midi(
                output_path=debug_midi_path,
                blocks=enriched_blocks,
                inserted_notes=deduped_inserted_notes,
                ticks_per_beat=timing.ticks_per_beat,
                tempo_us_per_beat=timing.tempo_us_per_beat,
                project_duration_sec=timing.duration_sec,
            )
            debug_path_for_report = str(debug_midi_path)

        incomplete_existing_block_count = len(incomplete_existing_reports)
        missing_expected_block_count = len(missing_expected_reports)
        complete_block_count = sum(1 for item in enriched_blocks if item.status == "complete")
        completed_incomplete_existing_block_count = sum(
            1
            for item in incomplete_existing_reports
            if item.action == "completed"
        )
        completed_missing_expected_block_count = sum(
            1
            for item in missing_expected_reports
            if item.action == "completed"
        )
        completed_block_count = completed_incomplete_existing_block_count + completed_missing_expected_block_count
        skipped_block_count = sum(1 for item in all_reports if item.action == "skipped")
        skipped_ambiguous_count = sum(
            1
            for item in all_reports
            if item.action == "skipped" and item.reason == "ambiguous"
        )
        skipped_no_clear_family_count = sum(
            1
            for item in all_reports
            if item.action == "skipped"
            and item.reason == "no_clear_family"
        )

        report = PatternCompletionReport(
            status="ok",
            layer=params.layer,
            project_dir=str(project_dir),
            base_midi_path=str(base_midi_path),
            bar_aligned_block_count=len(enriched_blocks),
            pattern_block_count=len(enriched_blocks),
            complete_block_count=complete_block_count,
            pattern_family_count=len(families),
            incomplete_existing_block_count=incomplete_existing_block_count,
            missing_expected_block_count=missing_expected_block_count,
            incomplete_block_count=len(all_reports),
            completed_incomplete_existing_block_count=completed_incomplete_existing_block_count,
            completed_missing_expected_block_count=completed_missing_expected_block_count,
            completed_block_count=completed_block_count,
            skipped_block_count=skipped_block_count,
            skipped_ambiguous_count=skipped_ambiguous_count,
            skipped_no_clear_family_count=skipped_no_clear_family_count,
            rejected_micro_note_count=stats.rejected_micro_note_count,
            rejected_polyphonic_stack_count=stats.rejected_polyphonic_stack_count,
            rejected_low_confidence_count=stats.rejected_low_confidence_count,
            rejected_tiny_gap_count=stats.rejected_tiny_gap_count,
            inserted_note_count=len(deduped_inserted_notes),
            output_midi_path=str(output_midi_path),
            pattern_blocks_file=str(pattern_blocks_path),
            pattern_families_file=str(pattern_families_path),
            incomplete_blocks_file=str(incomplete_blocks_path),
            missing_expected_blocks_file=str(missing_expected_blocks_path),
            debug_midi_path=debug_path_for_report,
            bar_gap_candidate_count=len(bar_gap_candidates),
            bar_gap_candidates_file=str(bar_gap_candidates_path),
            warnings=warnings,
            warning_count=len(warnings),
            error=None,
        )
        _write_json(report_path, report.model_dump(mode="json"))
        return report

    except PatternCompletionError as exc:
        error_report = PatternCompletionReport(
            status="error",
            layer=params.layer,
            project_dir=str(project_dir),
            base_midi_path=None,
            bar_aligned_block_count=0,
            pattern_block_count=0,
            complete_block_count=0,
            pattern_family_count=0,
            incomplete_existing_block_count=0,
            missing_expected_block_count=0,
            incomplete_block_count=0,
            completed_incomplete_existing_block_count=0,
            completed_missing_expected_block_count=0,
            completed_block_count=0,
            skipped_block_count=0,
            skipped_ambiguous_count=0,
            skipped_no_clear_family_count=0,
            rejected_micro_note_count=0,
            rejected_polyphonic_stack_count=0,
            rejected_low_confidence_count=0,
            rejected_tiny_gap_count=0,
            inserted_note_count=0,
            output_midi_path=None,
            pattern_blocks_file=str(pattern_blocks_path),
            pattern_families_file=str(pattern_families_path),
            incomplete_blocks_file=str(incomplete_blocks_path),
            missing_expected_blocks_file=str(missing_expected_blocks_path),
            debug_midi_path=None,
            bar_gap_candidate_count=0,
            bar_gap_candidates_file=str(bar_gap_candidates_path),
            warnings=[str(exc)],
            warning_count=1,
            error=str(exc),
        )
        _write_json(report_path, error_report.model_dump(mode="json"))
        raise


def _resolve_base_midi(project_dir: Path) -> Path:
    candidate_paths = [
        project_dir / "midi" / "working" / "working.mid",
        project_dir / "midi" / "working.mid",
    ]
    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path
    raise PatternCompletionError(
        f"Could not find base MIDI. Expected one of: {', '.join(str(item) for item in candidate_paths)}"
    )


def _parse_midi_timing(path: Path) -> _ParsedMidiTiming:
    midi_file = mido.MidiFile(str(path))
    ticks_per_beat = int(midi_file.ticks_per_beat)
    tempo_us_per_beat = 500000
    beats_per_bar = 4
    beat_unit = 4
    duration_sec = float(midi_file.length)

    for track in midi_file.tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo_us_per_beat = int(message.tempo)
            elif message.type == "time_signature":
                beats_per_bar = int(getattr(message, "numerator", 4) or 4)
                denominator = int(getattr(message, "denominator", 4) or 4)
                beat_unit = denominator if denominator > 0 else 4
            if message.type in {"set_tempo", "time_signature"}:
                continue
        else:
            continue
        break

    beat_length_sec = (tempo_us_per_beat / 1_000_000.0) * (4.0 / float(beat_unit))

    return _ParsedMidiTiming(
        ticks_per_beat=ticks_per_beat,
        tempo_us_per_beat=tempo_us_per_beat,
        beats_per_bar=beats_per_bar,
        beat_unit=beat_unit,
        beat_length_sec=beat_length_sec,
        duration_sec=duration_sec,
    )


def _load_base_notes(base_midi_path: Path, layer: str) -> list[_BaseNote]:
    note_document, _report = import_midi_candidate(base_midi_path, source="working", layer=layer)
    notes: list[_BaseNote] = []

    for note in sorted(note_document.notes, key=lambda item: (item.start_sec, item.end_sec, item.pitch_midi)):
        notes.append(
            _BaseNote(
                note_id=note.note_id,
                start_tick=int(note.start_tick),
                end_tick=int(note.end_tick),
                start_sec=float(note.start_sec),
                end_sec=float(note.end_sec),
                duration_sec=float(note.duration_sec),
                start_beat=float(note.start_tick) / float(max(1, note_document.ticks_per_beat)),
                end_beat=float(note.end_tick) / float(max(1, note_document.ticks_per_beat)),
                duration_beat=float(note.duration_ticks) / float(max(1, note_document.ticks_per_beat)),
                pitch_midi=int(note.pitch_midi),
                pitch_name=note.pitch_name,
                velocity=int(note.velocity),
                channel=note.channel,
            )
        )

    return notes


def _compute_local_note_thresholds(base_notes: list[_BaseNote]) -> _LocalNoteThresholds:
    if not base_notes:
        return _LocalNoteThresholds(
            min_duration_sec=0.12,
            min_velocity=48,
            median_velocity=90,
        )

    durations = [max(1e-6, float(note.duration_sec)) for note in base_notes]
    velocities = [int(note.velocity) for note in base_notes]

    median_duration = float(statistics.median(durations))
    median_velocity = int(round(float(statistics.median(velocities))))

    min_duration_sec = max(0.10, min(0.15, median_duration * 0.45))
    min_velocity = max(35, int(round(median_velocity * 0.45)))

    return _LocalNoteThresholds(
        min_duration_sec=round(min_duration_sec, 6),
        min_velocity=min_velocity,
        median_velocity=median_velocity,
    )


def _split_into_blocks(
    notes: list[_BaseNote],
    *,
    timing: _ParsedMidiTiming,
    bars_per_block: int = 1,
    grid_division: int = 16,
) -> list[PatternBlock]:
    if not notes:
        return []

    beats_per_bar = max(1, int(timing.beats_per_bar))
    block_length_beats = float(beats_per_bar * max(1, bars_per_block))
    block_length_sec = float(block_length_beats * timing.beat_length_sec)
    total_slots = max(1, int(block_length_beats * grid_division))
    bar_count = int(math.ceil(max(note.end_beat for note in notes) / float(beats_per_bar)))

    blocks: list[PatternBlock] = []
    time_signature = f"{beats_per_bar}/{timing.beat_unit}"
    grid_resolution = f"1/{grid_division}"

    for bar_index in range(bar_count):
        start_beat = float(bar_index * beats_per_bar)
        end_beat = start_beat + block_length_beats
        start_sec = float(start_beat * timing.beat_length_sec)
        end_sec = start_sec + block_length_sec

        block_notes = [
            note
            for note in notes
            if note.start_beat >= start_beat - 1e-9 and note.start_beat < end_beat - 1e-9
        ]
        block_notes.sort(key=lambda item: (item.start_beat, item.end_beat, item.pitch_midi))

        note_records: list[PatternBlockNote] = []
        relative_onsets_beat: list[float] = []
        relative_durations_beat: list[float] = []
        relative_onsets_sec: list[float] = []
        relative_durations_sec: list[float] = []
        pitches: list[int] = []
        pitch_names: list[str] = []
        occupied_slots: set[int] = set()
        onset_slots: list[int] = []

        for note in block_notes:
            onset_rel_beat = max(0.0, float(note.start_beat - start_beat))
            duration_beat = max(1e-6, float(note.duration_beat))
            duration_slots = max(1, int(round(duration_beat * grid_division)))
            onset_slot = int(round(onset_rel_beat * grid_division))
            onset_slot = max(0, min(total_slots - 1, onset_slot))
            end_slot = min(total_slots, onset_slot + duration_slots)

            for slot in range(onset_slot, end_slot):
                occupied_slots.add(slot)

            note_records.append(
                PatternBlockNote(
                    note_id=note.note_id,
                    start_tick=note.start_tick,
                    end_tick=note.end_tick,
                    start_sec=round(note.start_sec, 6),
                    end_sec=round(note.end_sec, 6),
                    duration_sec=round(note.duration_sec, 6),
                    start_beat=round(note.start_beat, 6),
                    end_beat=round(note.end_beat, 6),
                    duration_beat=round(duration_beat, 6),
                    onset_slot=onset_slot,
                    duration_slots=duration_slots,
                    pitch_midi=note.pitch_midi,
                    pitch_name=note.pitch_name,
                    velocity=note.velocity,
                    channel=note.channel,
                )
            )
            onset_slots.append(onset_slot)
            relative_onsets_beat.append(round(onset_rel_beat, 6))
            relative_durations_beat.append(round(duration_beat, 6))
            relative_onsets_sec.append(round(note.start_sec - start_sec, 6))
            relative_durations_sec.append(round(note.duration_sec, 6))
            pitches.append(note.pitch_midi)
            pitch_names.append(note.pitch_name)

        occupied_slots_sorted = sorted(occupied_slots)
        onset_slots_sorted = sorted(set(onset_slots))
        empty_slots = [slot for slot in range(total_slots) if slot not in occupied_slots]
        intervals = [
            int(pitches[item + 1] - pitches[item])
            for item in range(len(pitches) - 1)
        ]
        rhythm_signature = [
            round(relative_onsets_beat[item + 1] - relative_onsets_beat[item], 6)
            for item in range(len(relative_onsets_beat) - 1)
        ]

        blocks.append(
            PatternBlock(
                block_id=f"bar_{bar_index + 1:04d}",
                bar_index=bar_index,
                start_beat=round(start_beat, 6),
                end_beat=round(end_beat, 6),
                block_length_beats=round(block_length_beats, 6),
                start_sec=round(start_sec, 6),
                end_sec=round(end_sec, 6),
                duration_sec=round(max(0.0, end_sec - start_sec), 6),
                time_signature=time_signature,
                grid_resolution=grid_resolution,
                onset_slots=onset_slots_sorted,
                occupied_slots=occupied_slots_sorted,
                empty_slots=empty_slots,
                note_count=len(block_notes),
                notes=note_records,
                relative_onsets_beat=relative_onsets_beat,
                relative_durations_beat=relative_durations_beat,
                relative_onsets_sec=relative_onsets_sec,
                relative_durations_sec=relative_durations_sec,
                pitch_sequence=pitches,
                pitch_names=pitch_names,
                interval_sequence=intervals,
                rhythm_signature=rhythm_signature,
                pitch_set=sorted(set(pitches)),
                assigned_pattern_family_id=None,
                status="empty" if len(block_notes) == 0 else "unknown",
            )
        )

    return blocks


def _build_pattern_families(
    blocks: list[PatternBlock],
) -> list[PatternFamily]:
    if not blocks:
        return []

    non_empty_blocks = [block for block in blocks if block.note_count > 0]
    if not non_empty_blocks:
        return []

    note_count_hist = Counter(block.note_count for block in non_empty_blocks)
    reference_note_count = sorted(
        note_count_hist.items(),
        key=lambda item: (-item[1], -item[0]),
    )[0][0]

    complete_candidates = [
        block
        for block in non_empty_blocks
        if block.note_count >= max(2, reference_note_count)
    ]
    if not complete_candidates:
        complete_candidates = [block for block in non_empty_blocks if block.note_count >= 2]
    if not complete_candidates:
        return []

    families_by_signature: dict[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        list[PatternBlock],
    ] = defaultdict(list)

    normalized_by_block_id: dict[str, PatternBlock] = {}

    for block in complete_candidates:
        normalized = _monophonic_block_view(block)
        normalized_by_block_id[block.block_id] = normalized

        onset_slots = tuple(int(note.onset_slot or 0) for note in normalized.notes)
        duration_slots = tuple(int(note.duration_slots or 1) for note in normalized.notes)
        rhythm_quantized = tuple(int(round(value * 1000.0)) for value in normalized.rhythm_signature)
        pitch_signature = tuple(normalized.pitch_sequence)
        interval_signature = tuple(normalized.interval_sequence)
        pitch_set_signature = tuple(normalized.pitch_set)
        signature = (
            onset_slots,
            duration_slots,
            pitch_signature,
            interval_signature,
            pitch_set_signature,
        )
        families_by_signature[signature].append(block)

    families: list[PatternFamily] = []

    grouped_source = list(families_by_signature.values())

    family_counter = 0
    for grouped_blocks in sorted(
        grouped_source,
        key=lambda items: (-(len(items)), float(items[0].start_sec)),
    ):
        family_counter += 1
        family_id = (
            f"pattern_{chr(64 + family_counter)}"
            if family_counter <= 26
            else f"pattern_{family_counter:03d}"
        )

        representative = max(
            grouped_blocks,
            key=lambda item: (normalized_by_block_id[item.block_id].note_count, item.duration_sec),
        )
        normalized_representative = normalized_by_block_id[representative.block_id]

        families.append(
            PatternFamily(
                pattern_family_id=family_id,
                block_length_beats=float(normalized_representative.block_length_beats),
                time_signature=normalized_representative.time_signature,
                grid_resolution=normalized_representative.grid_resolution,
                representative_onset_slots=[int(note.onset_slot or 0) for note in normalized_representative.notes],
                representative_duration_slots=[int(note.duration_slots or 1) for note in normalized_representative.notes],
                representative_relative_onsets_beat=list(normalized_representative.relative_onsets_beat),
                representative_relative_durations_beat=list(normalized_representative.relative_durations_beat),
                representative_pitch_sequence=list(normalized_representative.pitch_sequence),
                representative_interval_sequence=list(normalized_representative.interval_sequence),
                representative_relative_onsets_sec=list(normalized_representative.relative_onsets_sec),
                representative_durations_sec=list(normalized_representative.relative_durations_sec),
                representative_pitch_set=list(normalized_representative.pitch_set),
                representative_note_count=normalized_representative.note_count,
                occurrence_count=len(grouped_blocks),
                occurrence_bars=[item.bar_index for item in sorted(grouped_blocks, key=lambda item: item.bar_index)],
                occurrences=[item.block_id for item in sorted(grouped_blocks, key=lambda item: item.start_sec)],
                first_seen_sec=min(item.start_sec for item in grouped_blocks),
                last_seen_sec=max(item.end_sec for item in grouped_blocks),
            )
        )

    families.sort(key=lambda item: item.pattern_family_id)
    return families


def _classify_block(
    *,
    block: PatternBlock,
    families: list[PatternFamily],
) -> PatternBlock:
    if block.note_count == 0:
        return block.model_copy(
            update={
                "assigned_pattern_family_id": None,
                "status": "empty",
            }
        )

    if not families:
        return block.model_copy(
            update={
                "assigned_pattern_family_id": None,
                "status": "unknown",
            }
        )

    exact_family_id: str | None = None
    for family in families:
        if _is_exact_family_match(block=block, family=family):
            exact_family_id = family.pattern_family_id
            break

    if exact_family_id is not None:
        return block.model_copy(
            update={
                "assigned_pattern_family_id": exact_family_id,
                "status": "complete",
            }
        )

    candidates: list[_CandidateFamilyMatch] = []
    for family in families:
        candidate = _score_family_match(incomplete_block=block, family=family)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return block.model_copy(
            update={
                "assigned_pattern_family_id": None,
                "status": "unknown",
            }
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return block.model_copy(
        update={
            "assigned_pattern_family_id": candidates[0].family.pattern_family_id,
            "status": "incomplete",
        }
    )


def _monophonic_block_view(block: PatternBlock) -> PatternBlock:
    if not block.notes:
        return block

    by_slot: dict[int, list[PatternBlockNote]] = defaultdict(list)
    for note in block.notes:
        by_slot[int(note.onset_slot or 0)].append(note)

    collapsed: list[PatternBlockNote] = []
    for slot in sorted(by_slot.keys()):
        candidates = by_slot[slot]
        selected = min(
            candidates,
            key=lambda item: (
                int(item.pitch_midi),
                -int(item.duration_slots or 1),
                -int(item.velocity),
            ),
        )
        collapsed.append(selected)

    relative_onsets_beat = [
        round(float(int(note.onset_slot or 0) / max(1, _grid_division_from_text(block.grid_resolution))), 6)
        for note in collapsed
    ]
    relative_durations_beat = [
        round(float(int(note.duration_slots or 1) / max(1, _grid_division_from_text(block.grid_resolution))), 6)
        for note in collapsed
    ]
    relative_onsets_sec = [
        round(float(value * (block.duration_sec / max(1e-6, block.block_length_beats))), 6)
        for value in relative_onsets_beat
    ]
    relative_durations_sec = [
        round(float(value * (block.duration_sec / max(1e-6, block.block_length_beats))), 6)
        for value in relative_durations_beat
    ]
    pitches = [int(note.pitch_midi) for note in collapsed]
    intervals = [int(pitches[index + 1] - pitches[index]) for index in range(len(pitches) - 1)]
    rhythm_signature = [
        round(relative_onsets_beat[index + 1] - relative_onsets_beat[index], 6)
        for index in range(len(relative_onsets_beat) - 1)
    ]

    return block.model_copy(
        update={
            "notes": collapsed,
            "note_count": len(collapsed),
            "onset_slots": sorted(set(int(note.onset_slot or 0) for note in collapsed)),
            "relative_onsets_beat": relative_onsets_beat,
            "relative_durations_beat": relative_durations_beat,
            "relative_onsets_sec": relative_onsets_sec,
            "relative_durations_sec": relative_durations_sec,
            "pitch_sequence": pitches,
            "pitch_names": [str(note.pitch_name) for note in collapsed],
            "interval_sequence": intervals,
            "rhythm_signature": rhythm_signature,
            "pitch_set": sorted(set(pitches)),
        }
    )


def _is_exact_family_match(*, block: PatternBlock, family: PatternFamily) -> bool:
    if len(block.notes) != len(family.representative_onset_slots):
        return False

    if block.pitch_sequence != family.representative_pitch_sequence:
        return False

    block_slots = [int(note.onset_slot or 0) for note in block.notes]
    if block_slots != list(family.representative_onset_slots):
        return False

    block_duration_slots = [int(note.duration_slots or 1) for note in block.notes]
    if block_duration_slots != list(family.representative_duration_slots):
        return False

    return True


def _complete_incomplete_block(
    *,
    incomplete_block: PatternBlock,
    blocks_by_id: dict[str, PatternBlock],
    families: list[PatternFamily],
    base_notes: list[_BaseNote],
    local_thresholds: _LocalNoteThresholds,
    stats: _CompletionStats,
) -> tuple[IncompleteBlockReport, list[ProposedCompletionNote]]:
    monophonic_block = _monophonic_block_view(incomplete_block)
    possible_matches: list[_CandidateFamilyMatch] = []

    for family in families:
        candidate = _score_family_match(incomplete_block=monophonic_block, family=family)
        if candidate is not None:
            possible_matches.append(candidate)

    possible_matches.sort(key=lambda item: item.score, reverse=True)

    match_models = [
        IncompleteBlockMatch(
            pattern_family_id=item.family.pattern_family_id,
            score=round(item.score, 6),
            reason=item.reason,
        )
        for item in possible_matches[:5]
    ]

    if not possible_matches:
        report = IncompleteBlockReport(
            block_type="incomplete_existing_block",
            incomplete_block_id=incomplete_block.block_id,
            target_bar_index=incomplete_block.bar_index,
            target_start_sec=incomplete_block.start_sec,
            target_end_sec=incomplete_block.end_sec,
            start_sec=incomplete_block.start_sec,
            end_sec=incomplete_block.end_sec,
            start_beat=incomplete_block.start_beat,
            end_beat=incomplete_block.end_beat,
            observed_slots=list(incomplete_block.occupied_slots),
            missing_slots=[],
            onset_slots_observed=list(monophonic_block.onset_slots),
            onset_slots_expected=[],
            onset_slots_missing=[],
            observed_pitch_sequence=list(incomplete_block.pitch_sequence),
            observed_relative_onsets_sec=list(incomplete_block.relative_onsets_sec),
            possible_matches=[],
            best_match_pattern_family_id=None,
            reason="no_clear_family",
            match_reason="No clear deterministic family match found.",
            missing_notes_to_insert=[],
            confidence_level="low",
            action="skipped",
        )
        return report, []

    best = possible_matches[0]
    second = possible_matches[1] if len(possible_matches) > 1 else None

    if second is not None and (best.score - second.score) < 0.08:
        report = IncompleteBlockReport(
            block_type="incomplete_existing_block",
            incomplete_block_id=incomplete_block.block_id,
            target_bar_index=incomplete_block.bar_index,
            target_start_sec=incomplete_block.start_sec,
            target_end_sec=incomplete_block.end_sec,
            start_sec=incomplete_block.start_sec,
            end_sec=incomplete_block.end_sec,
            start_beat=incomplete_block.start_beat,
            end_beat=incomplete_block.end_beat,
            observed_slots=list(incomplete_block.occupied_slots),
            missing_slots=[],
            onset_slots_observed=list(monophonic_block.onset_slots),
            onset_slots_expected=[int(slot) for slot in best.family.representative_onset_slots],
            onset_slots_missing=[],
            observed_pitch_sequence=list(incomplete_block.pitch_sequence),
            observed_relative_onsets_sec=list(incomplete_block.relative_onsets_sec),
            possible_matches=match_models,
            best_match_pattern_family_id=None,
            reason="ambiguous",
            match_reason=(
                "Ambiguous deterministic match. Top families are too close in score: "
                f"{best.family.pattern_family_id}={best.score:.3f}, {second.family.pattern_family_id}={second.score:.3f}"
            ),
            missing_notes_to_insert=[],
            confidence_level="low",
            action="skipped",
        )
        return report, []

    if best.score < 0.85:
        stats.rejected_low_confidence_count += 1
        report = IncompleteBlockReport(
            block_type="incomplete_existing_block",
            incomplete_block_id=incomplete_block.block_id,
            target_bar_index=incomplete_block.bar_index,
            target_start_sec=incomplete_block.start_sec,
            target_end_sec=incomplete_block.end_sec,
            start_sec=incomplete_block.start_sec,
            end_sec=incomplete_block.end_sec,
            start_beat=incomplete_block.start_beat,
            end_beat=incomplete_block.end_beat,
            observed_slots=list(incomplete_block.occupied_slots),
            missing_slots=[
                slot
                for slot in best.family.representative_onset_slots
                if int(slot) not in set(monophonic_block.onset_slots)
            ],
            onset_slots_observed=list(monophonic_block.onset_slots),
            onset_slots_expected=[int(slot) for slot in best.family.representative_onset_slots],
            onset_slots_missing=[
                int(slot)
                for slot in best.family.representative_onset_slots
                if int(slot) not in set(monophonic_block.onset_slots)
            ],
            observed_pitch_sequence=list(incomplete_block.pitch_sequence),
            observed_relative_onsets_sec=list(incomplete_block.relative_onsets_sec),
            possible_matches=match_models,
            best_match_pattern_family_id=best.family.pattern_family_id,
            source_pattern_family_id=best.family.pattern_family_id,
            source_family_occurrence_count=best.family.occurrence_count,
            reason="no_clear_family",
            match_reason="Low confidence deterministic match; only high confidence completions are allowed.",
            missing_notes_to_insert=[],
            inserted_notes=[],
            rejected_candidate_notes=[{"note_id": incomplete_block.block_id, "reason": "low_confidence"}],
            confidence_level=_confidence_from_score(best.score),
            action="skipped",
        )
        return report, []

    exemplar_block = _find_exemplar_block(best.family, blocks_by_id)
    if exemplar_block is None:
        report = IncompleteBlockReport(
            block_type="incomplete_existing_block",
            incomplete_block_id=incomplete_block.block_id,
            target_bar_index=incomplete_block.bar_index,
            target_start_sec=incomplete_block.start_sec,
            target_end_sec=incomplete_block.end_sec,
            start_sec=incomplete_block.start_sec,
            end_sec=incomplete_block.end_sec,
            start_beat=incomplete_block.start_beat,
            end_beat=incomplete_block.end_beat,
            observed_slots=list(incomplete_block.occupied_slots),
            missing_slots=[],
            onset_slots_observed=list(monophonic_block.onset_slots),
            onset_slots_expected=[int(slot) for slot in best.family.representative_onset_slots],
            onset_slots_missing=[],
            observed_pitch_sequence=list(incomplete_block.pitch_sequence),
            observed_relative_onsets_sec=list(incomplete_block.relative_onsets_sec),
            possible_matches=match_models,
            best_match_pattern_family_id=best.family.pattern_family_id,
            reason="no_clear_family",
            match_reason="Matched family has no exemplar block payload.",
            missing_notes_to_insert=[],
            confidence_level="low",
            action="skipped",
        )
        return report, []

    proposed = _propose_missing_notes(
        incomplete_block=monophonic_block,
        family=best.family,
        exemplar_block=exemplar_block,
        local_thresholds=local_thresholds,
    )

    validated: list[ProposedCompletionNote] = []
    rejected_count = 0
    rejected_candidates: list[dict[str, object]] = []
    for note in proposed:
        reason = _validate_proposed_note(
            note=note,
            incomplete_block=monophonic_block,
            family=best.family,
            base_notes=base_notes,
            local_thresholds=local_thresholds,
            stats=stats,
            accepted_notes=validated,
        )
        if reason is None:
            validated.append(note)
        else:
            rejected_count += 1
            rejected_candidates.append(
                {
                    "note_id": note.note_id,
                    "pitch_midi": note.pitch_midi,
                    "start_sec": note.start_sec,
                    "end_sec": note.end_sec,
                    "reason": reason,
                }
            )

    if not validated:
        report = IncompleteBlockReport(
            block_type="incomplete_existing_block",
            incomplete_block_id=incomplete_block.block_id,
            target_bar_index=incomplete_block.bar_index,
            target_start_sec=incomplete_block.start_sec,
            target_end_sec=incomplete_block.end_sec,
            start_sec=incomplete_block.start_sec,
            end_sec=incomplete_block.end_sec,
            start_beat=incomplete_block.start_beat,
            end_beat=incomplete_block.end_beat,
            observed_slots=list(incomplete_block.occupied_slots),
            missing_slots=[
                slot
                for slot in best.family.representative_onset_slots
                if slot not in set(monophonic_block.onset_slots)
            ],
            onset_slots_observed=list(monophonic_block.onset_slots),
            onset_slots_expected=[int(slot) for slot in best.family.representative_onset_slots],
            onset_slots_missing=[
                int(slot)
                for slot in best.family.representative_onset_slots
                if int(slot) not in set(monophonic_block.onset_slots)
            ],
            observed_pitch_sequence=list(incomplete_block.pitch_sequence),
            observed_relative_onsets_sec=list(incomplete_block.relative_onsets_sec),
            possible_matches=match_models,
            best_match_pattern_family_id=best.family.pattern_family_id,
            reason="rejected_validation",
            match_reason=f"{best.reason}. All proposed notes rejected during validation.",
            missing_notes_to_insert=[],
            inserted_notes=[],
            rejected_candidate_notes=rejected_candidates,
            confidence_level=_confidence_from_score(best.score),
            action="skipped",
        )
        return report, []

    report = IncompleteBlockReport(
        block_type="incomplete_existing_block",
        incomplete_block_id=incomplete_block.block_id,
        target_bar_index=incomplete_block.bar_index,
        target_start_sec=incomplete_block.start_sec,
        target_end_sec=incomplete_block.end_sec,
        start_sec=incomplete_block.start_sec,
        end_sec=incomplete_block.end_sec,
        start_beat=incomplete_block.start_beat,
        end_beat=incomplete_block.end_beat,
        observed_slots=list(incomplete_block.occupied_slots),
        missing_slots=[
            slot
            for slot in best.family.representative_onset_slots
            if slot not in set(monophonic_block.onset_slots)
        ],
        onset_slots_observed=list(monophonic_block.onset_slots),
        onset_slots_expected=[int(slot) for slot in best.family.representative_onset_slots],
        onset_slots_missing=[
            int(slot)
            for slot in best.family.representative_onset_slots
            if int(slot) not in set(monophonic_block.onset_slots)
        ],
        observed_pitch_sequence=list(incomplete_block.pitch_sequence),
        observed_relative_onsets_sec=list(incomplete_block.relative_onsets_sec),
        possible_matches=match_models,
        best_match_pattern_family_id=best.family.pattern_family_id,
        source_pattern_family_id=best.family.pattern_family_id,
        source_family_occurrence_count=best.family.occurrence_count,
        reason="completed",
        match_reason=(
            f"{best.reason}. Proposed={len(proposed)} accepted={len(validated)} rejected={rejected_count}."
        ),
        missing_notes_to_insert=validated,
        inserted_notes=validated,
        rejected_candidate_notes=rejected_candidates,
        confidence_level=_confidence_from_score(best.score),
        action="completed",
    )
    return report, validated


def _detect_missing_expected_blocks(
    *,
    families: list[PatternFamily],
    blocks_by_id: dict[str, PatternBlock],
    base_notes: list[_BaseNote],
) -> list[MissingExpectedBlock]:
    _ = base_notes
    candidates: list[MissingExpectedBlock] = []
    candidate_counter = 0

    blocks_by_bar = {block.bar_index: block for block in blocks_by_id.values()}

    for family in families:
        if family.occurrence_count < 2:
            continue

        occurrence_bars = sorted(set(int(item) for item in family.occurrence_bars))
        if len(occurrence_bars) < 2:
            continue

        bar_diffs = [
            occurrence_bars[index + 1] - occurrence_bars[index]
            for index in range(len(occurrence_bars) - 1)
            if occurrence_bars[index + 1] > occurrence_bars[index]
        ]
        if not bar_diffs:
            continue

        typical_period = max(1.0, float(statistics.median(bar_diffs)))
        variation = statistics.pstdev(bar_diffs) if len(bar_diffs) > 1 else 0.0
        stability = max(0.0, min(1.0, 1.0 - (variation / max(1.0, typical_period))))

        sparse_threshold = max(1, int(math.floor(float(family.representative_note_count) * 0.35)))

        for index in range(len(occurrence_bars) - 1):
            before_bar = occurrence_bars[index]
            after_bar = occurrence_bars[index + 1]
            bar_gap = after_bar - before_bar
            if bar_gap <= 1:
                continue

            approx_multiple = max(1, int(round(bar_gap / typical_period)))
            if abs(bar_gap - (approx_multiple * typical_period)) > max(1.0, typical_period * 0.35):
                continue

            for missing_bar in range(before_bar + 1, after_bar):
                block = blocks_by_bar.get(missing_bar)
                if block is None:
                    continue

                if block.note_count > sparse_threshold:
                    continue

                observed_slots = list(block.onset_slots)
                expected_slots = [int(slot) for slot in family.representative_onset_slots]
                missing_slots = [
                    slot
                    for slot in expected_slots
                    if slot not in set(observed_slots)
                ]
                if not missing_slots:
                    continue

                required_missing = max(1, int(math.ceil(len(expected_slots) * 0.5)))
                if len(missing_slots) < required_missing:
                    continue

                sparsity_score = 1.0 - min(1.0, block.note_count / float(sparse_threshold + 1))
                missing_coverage = len(missing_slots) / float(max(1, len(expected_slots)))
                gap_strength = min(1.0, float(max(0, bar_gap - 1)) / max(1.0, typical_period))
                empty_bonus = 0.10 if block.note_count == 0 else 0.0
                confidence_score = max(
                    0.0,
                    min(
                        1.0,
                        0.45
                        + (0.20 * min(1.0, family.occurrence_count / 4.0))
                        + (0.15 * stability)
                        + (0.15 * missing_coverage)
                        + (0.10 * sparsity_score)
                        + (0.10 * gap_strength)
                        + empty_bonus,
                    ),
                )

                before_block = blocks_by_bar.get(before_bar)
                after_block = blocks_by_bar.get(after_bar)

                candidate_counter += 1
                reason = (
                    "missing_expected_pattern_occurrence; "
                    f"bar_gap={bar_gap} period={typical_period:.2f} stability={stability:.2f} "
                    f"notes_in_bar={block.note_count} missing_onsets={len(missing_slots)}/{len(expected_slots)}"
                )
                candidates.append(
                    MissingExpectedBlock(
                        missing_block_id=f"missing_{candidate_counter:04d}",
                        target_bar_index=missing_bar,
                        expected_pattern_family_id=family.pattern_family_id,
                        write_start_sec=round(block.start_sec, 6),
                        write_end_sec=round(block.end_sec, 6),
                        write_start_beat=round(block.start_beat, 6),
                        write_end_beat=round(block.end_beat, 6),
                        expected_duration_sec=round(block.duration_sec, 6),
                        expected_duration_beat=round(block.block_length_beats, 6),
                        observed_slots=observed_slots,
                        missing_slots=missing_slots,
                        evidence_before_occurrences=[before_block.block_id] if before_block is not None else [],
                        evidence_after_occurrences=[after_block.block_id] if after_block is not None else [],
                        detected_note_count_in_region=block.note_count,
                        confidence_score=round(confidence_score, 6),
                        possible_matches=[
                            IncompleteBlockMatch(
                                pattern_family_id=family.pattern_family_id,
                                score=round(confidence_score, 6),
                                reason=reason,
                            )
                        ],
                    )
                )

    deduped: dict[tuple[int, str], MissingExpectedBlock] = {}
    for candidate in candidates:
        family_id = candidate.expected_pattern_family_id or "unknown"
        key = (int(candidate.target_bar_index), family_id)
        existing = deduped.get(key)
        if existing is None or candidate.confidence_score > existing.confidence_score:
            deduped[key] = candidate

    return sorted(
        deduped.values(),
        key=lambda item: (item.write_start_sec, item.write_end_sec, -(item.confidence_score)),
    )


def _detect_bar_gap_candidates(
    *,
    project_dir: Path,
    blocks: list[PatternBlock],
    families: list[PatternFamily],
) -> list[BarGapCandidate]:
    if not blocks:
        return []

    blocks_sorted = sorted(blocks, key=lambda item: item.bar_index)
    audio_frames = _load_audio_frames(project_dir=project_dir)
    pitch_frames = _load_pitch_frames(project_dir=project_dir)
    audio_available = bool(audio_frames)
    pitch_available = bool(pitch_frames)

    family_by_id = {family.pattern_family_id: family for family in families}
    family_by_bar: dict[int, str] = {}
    for block in blocks_sorted:
        family_id = block.assigned_pattern_family_id
        if family_id is None:
            continue
        family_by_bar[int(block.bar_index)] = str(family_id)

    sparse_threshold = max(1, int(math.ceil(statistics.median([item.note_count for item in blocks_sorted]) * 0.35)))

    candidates: list[BarGapCandidate] = []
    current_start: int | None = None
    current_note_count = 0
    candidate_counter = 0

    def finalize(end_index: int) -> None:
        nonlocal current_start
        nonlocal current_note_count
        nonlocal candidate_counter

        if current_start is None:
            return

        gap_blocks = [
            block
            for block in blocks_sorted
            if current_start <= int(block.bar_index) <= end_index
        ]
        if not gap_blocks:
            current_start = None
            current_note_count = 0
            return

        start_sec = float(gap_blocks[0].start_sec)
        end_sec = float(gap_blocks[-1].end_sec)
        bar_count = (end_index - current_start) + 1

        before_context = _collect_family_context(
            anchor_bar=current_start - 1,
            direction=-1,
            family_by_bar=family_by_bar,
            family_by_id=family_by_id,
            gap_start_bar=current_start,
            gap_end_bar=end_index,
        )
        after_context = _collect_family_context(
            anchor_bar=end_index + 1,
            direction=1,
            family_by_bar=family_by_bar,
            family_by_id=family_by_id,
            gap_start_bar=current_start,
            gap_end_bar=end_index,
        )

        audio_ratio = _activity_ratio_in_frames(audio_frames, start_sec=start_sec, end_sec=end_sec, key="onset_score")
        pitch_ratio = _activity_ratio_in_frames(pitch_frames, start_sec=start_sec, end_sec=end_sec, key="voiced")

        before_ids = {item.pattern_family_id for item in before_context}
        after_ids = {item.pattern_family_id for item in after_context}
        raw_bridge_ids = sorted(before_ids.intersection(after_ids))
        bridge_ids: list[str] = []
        weak_bridge_ids: list[str] = []

        audio_strong = audio_ratio is not None and audio_ratio >= 0.35
        pitch_strong = pitch_ratio is not None and pitch_ratio >= 0.35

        for family_id in raw_bridge_ids:
            family = family_by_id.get(family_id)
            occurrence_count = int(family.occurrence_count) if family is not None else 0
            before_distances = [item.distance_bars for item in before_context if item.pattern_family_id == family_id]
            after_distances = [item.distance_bars for item in after_context if item.pattern_family_id == family_id]
            near_gap = (
                bool(before_distances)
                and bool(after_distances)
                and (min(before_distances) <= 1 or min(after_distances) <= 1)
            )

            if occurrence_count >= 2:
                bridge_ids.append(family_id)
                continue

            if audio_strong and pitch_strong and not near_gap:
                bridge_ids.append(family_id)
                continue

            weak_bridge_ids.append(family_id)

        same_bridge = bool(bridge_ids)

        compatible_bridge = False
        if not same_bridge and before_context and after_context:
            compatible_bridge = _has_compatible_bridge(before_context, after_context, family_by_id)

        completion_readiness = "insufficient_context"
        completion_possible = False
        reason = "No deterministic bridge context around gap."

        if same_bridge and bar_count >= 1:
            completion_readiness = "extremely_clear"
            completion_possible = True
            reason = (
                "Same pattern family appears before and after gap; deterministic bridge is extremely clear."
            )
        elif weak_bridge_ids:
            completion_readiness = "unclear"
            reason = (
                "Bridge family overlap exists but evidence is weak (occurrence too low or only near-gap support): "
                f"{', '.join(weak_bridge_ids)}."
            )
        elif compatible_bridge:
            completion_readiness = "unclear"
            reason = "Compatible but non-identical families appear before and after gap; diagnostic only."
        elif before_context or after_context:
            completion_readiness = "unclear"
            reason = "Partial family context around gap; deterministic fill not yet allowed."

        candidate_counter += 1
        candidates.append(
            BarGapCandidate(
                gap_id=f"gap_{candidate_counter:04d}",
                start_bar_index=int(current_start),
                end_bar_index=int(end_index),
                bar_index_range=f"{current_start + 1}-{end_index + 1}",
                bar_count=int(bar_count),
                start_sec=round(start_sec, 6),
                end_sec=round(end_sec, 6),
                note_count=int(current_note_count),
                sparse_threshold=int(sparse_threshold),
                audio_features_available=audio_available,
                audio_evidence_exists=(audio_ratio is not None and audio_ratio >= 0.10) if audio_available else None,
                audio_active_frame_ratio=round(audio_ratio, 6) if audio_ratio is not None else None,
                pitch_contour_available=pitch_available,
                pitch_contour_evidence_exists=(pitch_ratio is not None and pitch_ratio >= 0.10) if pitch_available else None,
                pitch_voiced_frame_ratio=round(pitch_ratio, 6) if pitch_ratio is not None else None,
                families_before=before_context,
                families_after=after_context,
                same_family_bridge=same_bridge,
                compatible_family_bridge=compatible_bridge,
                bridge_family_ids=bridge_ids,
                completion_possible=completion_possible,
                completion_readiness=completion_readiness,
                completion_reason=reason,
            )
        )

        current_start = None
        current_note_count = 0

    for block in blocks_sorted:
        is_sparse = int(block.note_count) <= sparse_threshold
        if is_sparse:
            if current_start is None:
                current_start = int(block.bar_index)
                current_note_count = int(block.note_count)
            else:
                current_note_count += int(block.note_count)
            continue

        if current_start is not None:
            finalize(int(block.bar_index) - 1)

    if current_start is not None:
        finalize(int(blocks_sorted[-1].bar_index))

    return sorted(candidates, key=lambda item: (item.start_sec, item.end_sec, item.bar_count), reverse=False)


def _collect_family_context(
    *,
    anchor_bar: int,
    direction: int,
    family_by_bar: dict[int, str],
    family_by_id: dict[str, PatternFamily],
    gap_start_bar: int,
    gap_end_bar: int,
    max_scan: int = 8,
) -> list[BarGapFamilyContext]:
    contexts: list[BarGapFamilyContext] = []
    seen: set[str] = set()
    edge_bar = gap_start_bar if direction < 0 else gap_end_bar
    for offset in range(0, max_scan):
        bar_index = anchor_bar + (direction * offset)
        if gap_start_bar <= int(bar_index) <= gap_end_bar:
            continue
        family_id = family_by_bar.get(int(bar_index))
        if family_id is None or family_id in seen:
            continue
        seen.add(family_id)
        family = family_by_id.get(family_id)
        occurrence_count = family.occurrence_count if family is not None else None
        distance_bars = abs(int(bar_index) - int(edge_bar))
        contexts.append(
            BarGapFamilyContext(
                pattern_family_id=family_id,
                bar_index=int(bar_index),
                distance_bars=int(distance_bars),
                occurrence_count=int(occurrence_count) if occurrence_count is not None else None,
            )
        )
        if len(contexts) >= 5:
            break
    return contexts


def _has_compatible_bridge(
    before: list[BarGapFamilyContext],
    after: list[BarGapFamilyContext],
    family_by_id: dict[str, PatternFamily],
) -> bool:
    for left in before:
        for right in after:
            fam_left = family_by_id.get(left.pattern_family_id)
            fam_right = family_by_id.get(right.pattern_family_id)
            if fam_left is None or fam_right is None:
                continue
            if fam_left.representative_note_count != fam_right.representative_note_count:
                continue
            if fam_left.representative_interval_sequence == fam_right.representative_interval_sequence:
                return True
            left_rhythm = tuple(int(round(item * 1000.0)) for item in fam_left.representative_relative_onsets_beat)
            right_rhythm = tuple(int(round(item * 1000.0)) for item in fam_right.representative_relative_onsets_beat)
            if left_rhythm == right_rhythm:
                return True
    return False


def _load_audio_frames(*, project_dir: Path) -> list[dict[str, object]]:
    path = project_dir / "analysis" / "audio_features.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    frames = payload.get("frames")
    if isinstance(frames, list):
        return [item for item in frames if isinstance(item, dict)]
    return []


def _load_pitch_frames(*, project_dir: Path) -> list[dict[str, object]]:
    path = project_dir / "analysis" / "bass_pitch_contour.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    frames = payload.get("frames")
    if isinstance(frames, list):
        return [item for item in frames if isinstance(item, dict)]
    return []


def _activity_ratio_in_frames(
    frames: list[dict[str, object]],
    *,
    start_sec: float,
    end_sec: float,
    key: str,
) -> float | None:
    if not frames:
        return None

    selected = [
        item
        for item in frames
        if float(item.get("end_sec", 0.0)) > float(start_sec)
        and float(item.get("start_sec", 0.0)) < float(end_sec)
    ]
    if not selected:
        return None

    active = 0
    for frame in selected:
        if key == "onset_score":
            if float(frame.get("onset_score", 0.0)) >= 0.01 or bool(frame.get("is_silent", False)) is False:
                active += 1
            continue
        if key == "voiced":
            if bool(frame.get("voiced", False)):
                active += 1
            continue

    return active / float(len(selected))


def _complete_missing_expected_blocks(
    *,
    missing_expected_blocks: list[MissingExpectedBlock],
    blocks_by_id: dict[str, PatternBlock],
    families: list[PatternFamily],
    base_notes: list[_BaseNote],
    local_thresholds: _LocalNoteThresholds,
    stats: _CompletionStats,
) -> tuple[list[IncompleteBlockReport], list[ProposedCompletionNote]]:
    if not missing_expected_blocks:
        return [], []

    families_by_id = {family.pattern_family_id: family for family in families}
    region_groups: dict[int, list[MissingExpectedBlock]] = defaultdict(list)

    for block in missing_expected_blocks:
        key = int(block.target_bar_index)
        region_groups[key].append(block)

    reports: list[IncompleteBlockReport] = []
    inserted_notes: list[ProposedCompletionNote] = []

    for grouped in sorted(
        region_groups.values(),
        key=lambda items: min(item.write_start_sec for item in items),
    ):
        best_by_family: dict[str, MissingExpectedBlock] = {}
        for item in grouped:
            family_id = item.expected_pattern_family_id or "unknown"
            existing = best_by_family.get(family_id)
            if existing is None or item.confidence_score > existing.confidence_score:
                best_by_family[family_id] = item

        candidates = sorted(best_by_family.values(), key=lambda item: item.confidence_score, reverse=True)
        if not candidates:
            continue

        possible_matches = [
            IncompleteBlockMatch(
                pattern_family_id=item.expected_pattern_family_id or "unknown",
                score=round(item.confidence_score, 6),
                reason=(
                    "missing_expected_pattern_occurrence; "
                    f"before={item.evidence_before_occurrences} after={item.evidence_after_occurrences} "
                    f"region_notes={item.detected_note_count_in_region}"
                ),
            )
            for item in candidates[:5]
        ]

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        if (
            second is not None
            and best.expected_pattern_family_id != second.expected_pattern_family_id
            and (best.confidence_score - second.confidence_score) < 0.12
        ):
            reports.append(
                IncompleteBlockReport(
                    block_type="missing_expected_block",
                    missing_block_id=best.missing_block_id,
                    target_bar_index=best.target_bar_index,
                    target_start_sec=best.write_start_sec,
                    target_end_sec=best.write_end_sec,
                    expected_pattern_family_id=None,
                    start_sec=best.write_start_sec,
                    end_sec=best.write_end_sec,
                    start_beat=best.write_start_beat,
                    end_beat=best.write_end_beat,
                    write_start_sec=best.write_start_sec,
                    write_end_sec=best.write_end_sec,
                    write_start_beat=best.write_start_beat,
                    write_end_beat=best.write_end_beat,
                    expected_duration_sec=best.expected_duration_sec,
                    expected_duration_beat=best.expected_duration_beat,
                    evidence_before_occurrences=list(best.evidence_before_occurrences),
                    evidence_after_occurrences=list(best.evidence_after_occurrences),
                    observed_note_count_in_region=best.detected_note_count_in_region,
                    onset_slots_observed=list(best.observed_slots),
                    onset_slots_expected=[],
                    onset_slots_missing=list(best.missing_slots),
                    observed_slots=list(best.observed_slots),
                    missing_slots=list(best.missing_slots),
                    possible_matches=possible_matches,
                    best_match_pattern_family_id=None,
                    reason="ambiguous",
                    match_reason=(
                        "Ambiguous deterministic match for missing expected block. "
                        f"Top families are too close: {best.expected_pattern_family_id}={best.confidence_score:.3f}, "
                        f"{second.expected_pattern_family_id}={second.confidence_score:.3f}"
                    ),
                    missing_notes_to_insert=[],
                    inserted_notes=[],
                    rejected_candidate_notes=[],
                    confidence_level="low",
                    action="skipped",
                )
            )
            continue

        if best.expected_pattern_family_id is None or best.confidence_score < 0.85:
            stats.rejected_low_confidence_count += 1
            reports.append(
                IncompleteBlockReport(
                    block_type="missing_expected_block",
                    missing_block_id=best.missing_block_id,
                    target_bar_index=best.target_bar_index,
                    target_start_sec=best.write_start_sec,
                    target_end_sec=best.write_end_sec,
                    expected_pattern_family_id=best.expected_pattern_family_id,
                    start_sec=best.write_start_sec,
                    end_sec=best.write_end_sec,
                    start_beat=best.write_start_beat,
                    end_beat=best.write_end_beat,
                    write_start_sec=best.write_start_sec,
                    write_end_sec=best.write_end_sec,
                    write_start_beat=best.write_start_beat,
                    write_end_beat=best.write_end_beat,
                    expected_duration_sec=best.expected_duration_sec,
                    expected_duration_beat=best.expected_duration_beat,
                    evidence_before_occurrences=list(best.evidence_before_occurrences),
                    evidence_after_occurrences=list(best.evidence_after_occurrences),
                    observed_note_count_in_region=best.detected_note_count_in_region,
                    onset_slots_observed=list(best.observed_slots),
                    onset_slots_expected=[],
                    onset_slots_missing=list(best.missing_slots),
                    observed_slots=list(best.observed_slots),
                    missing_slots=list(best.missing_slots),
                    possible_matches=possible_matches,
                    best_match_pattern_family_id=best.expected_pattern_family_id,
                    source_pattern_family_id=best.expected_pattern_family_id,
                    reason="no_clear_family",
                    match_reason="Low confidence deterministic candidate; only high confidence completions are allowed.",
                    missing_notes_to_insert=[],
                    inserted_notes=[],
                    rejected_candidate_notes=[
                        {
                            "note_id": best.missing_block_id,
                            "reason": "low_confidence",
                        }
                    ],
                    confidence_level="low",
                    action="skipped",
                )
            )
            continue

        family = families_by_id.get(best.expected_pattern_family_id)
        if (
            family is None
            and best.expected_pattern_family_id is not None
            and best.expected_pattern_family_id.startswith("synthetic::")
        ):
            synthetic_block_id = best.expected_pattern_family_id.split("::", 1)[1]
            synthetic_block = blocks_by_id.get(synthetic_block_id)
            if synthetic_block is not None:
                family = _family_from_block(
                    block=synthetic_block,
                    family_id=best.expected_pattern_family_id,
                )

        if family is None and best.evidence_before_occurrences:
            evidence_block = blocks_by_id.get(best.evidence_before_occurrences[0])
            if evidence_block is not None:
                family = _family_from_block(
                    block=evidence_block,
                    family_id=best.expected_pattern_family_id or f"synthetic::{evidence_block.block_id}",
                )

        if family is None:
            reports.append(
                IncompleteBlockReport(
                    block_type="missing_expected_block",
                    missing_block_id=best.missing_block_id,
                    target_bar_index=best.target_bar_index,
                    target_start_sec=best.write_start_sec,
                    target_end_sec=best.write_end_sec,
                    expected_pattern_family_id=best.expected_pattern_family_id,
                    start_sec=best.write_start_sec,
                    end_sec=best.write_end_sec,
                    start_beat=best.write_start_beat,
                    end_beat=best.write_end_beat,
                    write_start_sec=best.write_start_sec,
                    write_end_sec=best.write_end_sec,
                    write_start_beat=best.write_start_beat,
                    write_end_beat=best.write_end_beat,
                    expected_duration_sec=best.expected_duration_sec,
                    expected_duration_beat=best.expected_duration_beat,
                    evidence_before_occurrences=list(best.evidence_before_occurrences),
                    evidence_after_occurrences=list(best.evidence_after_occurrences),
                    observed_note_count_in_region=best.detected_note_count_in_region,
                    onset_slots_observed=list(best.observed_slots),
                    onset_slots_expected=[],
                    onset_slots_missing=list(best.missing_slots),
                    observed_slots=list(best.observed_slots),
                    missing_slots=list(best.missing_slots),
                    possible_matches=possible_matches,
                    best_match_pattern_family_id=best.expected_pattern_family_id,
                    source_pattern_family_id=best.expected_pattern_family_id,
                    reason="no_clear_family",
                    match_reason="No clear deterministic family payload for missing expected block.",
                    missing_notes_to_insert=[],
                    inserted_notes=[],
                    rejected_candidate_notes=[],
                    confidence_level="low",
                    action="skipped",
                )
            )
            continue

        if family.occurrence_count < 2:
            stats.rejected_low_confidence_count += 1
            reports.append(
                IncompleteBlockReport(
                    block_type="missing_expected_block",
                    missing_block_id=best.missing_block_id,
                    target_bar_index=best.target_bar_index,
                    target_start_sec=best.write_start_sec,
                    target_end_sec=best.write_end_sec,
                    expected_pattern_family_id=best.expected_pattern_family_id,
                    source_pattern_family_id=family.pattern_family_id,
                    source_family_occurrence_count=family.occurrence_count,
                    start_sec=best.write_start_sec,
                    end_sec=best.write_end_sec,
                    start_beat=best.write_start_beat,
                    end_beat=best.write_end_beat,
                    write_start_sec=best.write_start_sec,
                    write_end_sec=best.write_end_sec,
                    write_start_beat=best.write_start_beat,
                    write_end_beat=best.write_end_beat,
                    expected_duration_sec=best.expected_duration_sec,
                    expected_duration_beat=best.expected_duration_beat,
                    evidence_before_occurrences=list(best.evidence_before_occurrences),
                    evidence_after_occurrences=list(best.evidence_after_occurrences),
                    observed_note_count_in_region=best.detected_note_count_in_region,
                    onset_slots_observed=list(best.observed_slots),
                    onset_slots_expected=[int(slot) for slot in family.representative_onset_slots],
                    onset_slots_missing=list(best.missing_slots),
                    observed_slots=list(best.observed_slots),
                    missing_slots=list(best.missing_slots),
                    possible_matches=possible_matches,
                    best_match_pattern_family_id=family.pattern_family_id,
                    reason="no_clear_family",
                    match_reason="Family occurrence evidence too weak for missing-expected completion.",
                    missing_notes_to_insert=[],
                    inserted_notes=[],
                    rejected_candidate_notes=[{"note_id": best.missing_block_id, "reason": "insufficient_family_evidence"}],
                    confidence_level="low",
                    action="skipped",
                )
            )
            continue

        if (best.write_end_sec - best.write_start_sec) < max(0.25, local_thresholds.min_duration_sec * 1.5):
            stats.rejected_tiny_gap_count += 1
            reports.append(
                IncompleteBlockReport(
                    block_type="missing_expected_block",
                    missing_block_id=best.missing_block_id,
                    target_bar_index=best.target_bar_index,
                    target_start_sec=best.write_start_sec,
                    target_end_sec=best.write_end_sec,
                    expected_pattern_family_id=best.expected_pattern_family_id,
                    source_pattern_family_id=family.pattern_family_id,
                    source_family_occurrence_count=family.occurrence_count,
                    start_sec=best.write_start_sec,
                    end_sec=best.write_end_sec,
                    start_beat=best.write_start_beat,
                    end_beat=best.write_end_beat,
                    write_start_sec=best.write_start_sec,
                    write_end_sec=best.write_end_sec,
                    write_start_beat=best.write_start_beat,
                    write_end_beat=best.write_end_beat,
                    expected_duration_sec=best.expected_duration_sec,
                    expected_duration_beat=best.expected_duration_beat,
                    evidence_before_occurrences=list(best.evidence_before_occurrences),
                    evidence_after_occurrences=list(best.evidence_after_occurrences),
                    observed_note_count_in_region=best.detected_note_count_in_region,
                    onset_slots_observed=list(best.observed_slots),
                    onset_slots_expected=[int(slot) for slot in family.representative_onset_slots],
                    onset_slots_missing=list(best.missing_slots),
                    observed_slots=list(best.observed_slots),
                    missing_slots=list(best.missing_slots),
                    possible_matches=possible_matches,
                    best_match_pattern_family_id=family.pattern_family_id,
                    reason="rejected_tiny_gap",
                    match_reason="Target write region too tiny for meaningful completion.",
                    missing_notes_to_insert=[],
                    inserted_notes=[],
                    rejected_candidate_notes=[{"note_id": best.missing_block_id, "reason": "tiny_gap"}],
                    confidence_level="low",
                    action="skipped",
                )
            )
            continue

        report, notes = _complete_missing_expected_block(
            missing_expected_block=best,
            family=family,
            blocks_by_id=blocks_by_id,
            base_notes=base_notes,
            possible_matches=possible_matches,
            local_thresholds=local_thresholds,
            stats=stats,
        )
        reports.append(report)
        inserted_notes.extend(notes)

    return reports, inserted_notes


def _complete_missing_expected_block(
    *,
    missing_expected_block: MissingExpectedBlock,
    family: PatternFamily,
    blocks_by_id: dict[str, PatternBlock],
    base_notes: list[_BaseNote],
    possible_matches: list[IncompleteBlockMatch],
    local_thresholds: _LocalNoteThresholds,
    stats: _CompletionStats,
) -> tuple[IncompleteBlockReport, list[ProposedCompletionNote]]:
    exemplar_block = _find_exemplar_block(family, blocks_by_id)
    if exemplar_block is None:
        report = IncompleteBlockReport(
            block_type="missing_expected_block",
            missing_block_id=missing_expected_block.missing_block_id,
            target_bar_index=missing_expected_block.target_bar_index,
            target_start_sec=missing_expected_block.write_start_sec,
            target_end_sec=missing_expected_block.write_end_sec,
            expected_pattern_family_id=family.pattern_family_id,
            source_pattern_family_id=family.pattern_family_id,
            source_family_occurrence_count=family.occurrence_count,
            start_sec=missing_expected_block.write_start_sec,
            end_sec=missing_expected_block.write_end_sec,
            start_beat=missing_expected_block.write_start_beat,
            end_beat=missing_expected_block.write_end_beat,
            write_start_sec=missing_expected_block.write_start_sec,
            write_end_sec=missing_expected_block.write_end_sec,
            write_start_beat=missing_expected_block.write_start_beat,
            write_end_beat=missing_expected_block.write_end_beat,
            expected_duration_sec=missing_expected_block.expected_duration_sec,
            expected_duration_beat=missing_expected_block.expected_duration_beat,
            evidence_before_occurrences=list(missing_expected_block.evidence_before_occurrences),
            evidence_after_occurrences=list(missing_expected_block.evidence_after_occurrences),
            observed_note_count_in_region=missing_expected_block.detected_note_count_in_region,
            onset_slots_observed=list(missing_expected_block.observed_slots),
            onset_slots_expected=[int(slot) for slot in family.representative_onset_slots],
            onset_slots_missing=list(missing_expected_block.missing_slots),
            observed_slots=list(missing_expected_block.observed_slots),
            missing_slots=list(missing_expected_block.missing_slots),
            possible_matches=possible_matches,
            best_match_pattern_family_id=family.pattern_family_id,
            reason="no_clear_family",
            match_reason="No clear deterministic family payload for missing expected block.",
            missing_notes_to_insert=[],
            inserted_notes=[],
            rejected_candidate_notes=[],
            confidence_level="low",
            action="skipped",
        )
        return report, []

    proposed = _propose_missing_expected_notes(
        missing_expected_block=missing_expected_block,
        family=family,
        exemplar_block=exemplar_block,
        local_thresholds=local_thresholds,
    )

    validated: list[ProposedCompletionNote] = []
    rejected_count = 0
    rejected_candidates: list[dict[str, object]] = []
    for note in proposed:
        reason = _validate_missing_expected_note(
            note=note,
            missing_expected_block=missing_expected_block,
            family=family,
            base_notes=base_notes,
            local_thresholds=local_thresholds,
            stats=stats,
            accepted_notes=validated,
        )
        if reason is None:
            validated.append(note)
        else:
            rejected_count += 1
            rejected_candidates.append(
                {
                    "note_id": note.note_id,
                    "pitch_midi": note.pitch_midi,
                    "start_sec": note.start_sec,
                    "end_sec": note.end_sec,
                    "reason": reason,
                }
            )

    if not validated:
        report = IncompleteBlockReport(
            block_type="missing_expected_block",
            missing_block_id=missing_expected_block.missing_block_id,
            target_bar_index=missing_expected_block.target_bar_index,
            target_start_sec=missing_expected_block.write_start_sec,
            target_end_sec=missing_expected_block.write_end_sec,
            expected_pattern_family_id=family.pattern_family_id,
            source_pattern_family_id=family.pattern_family_id,
            source_family_occurrence_count=family.occurrence_count,
            start_sec=missing_expected_block.write_start_sec,
            end_sec=missing_expected_block.write_end_sec,
            start_beat=missing_expected_block.write_start_beat,
            end_beat=missing_expected_block.write_end_beat,
            write_start_sec=missing_expected_block.write_start_sec,
            write_end_sec=missing_expected_block.write_end_sec,
            write_start_beat=missing_expected_block.write_start_beat,
            write_end_beat=missing_expected_block.write_end_beat,
            expected_duration_sec=missing_expected_block.expected_duration_sec,
            expected_duration_beat=missing_expected_block.expected_duration_beat,
            evidence_before_occurrences=list(missing_expected_block.evidence_before_occurrences),
            evidence_after_occurrences=list(missing_expected_block.evidence_after_occurrences),
            observed_note_count_in_region=missing_expected_block.detected_note_count_in_region,
            onset_slots_observed=list(missing_expected_block.observed_slots),
            onset_slots_expected=[int(slot) for slot in family.representative_onset_slots],
            onset_slots_missing=list(missing_expected_block.missing_slots),
            observed_slots=list(missing_expected_block.observed_slots),
            missing_slots=list(missing_expected_block.missing_slots),
            possible_matches=possible_matches,
            best_match_pattern_family_id=family.pattern_family_id,
            reason="rejected_validation",
            match_reason=(
                "missing_expected_pattern_occurrence. "
                f"All proposed notes rejected during validation (rejected={rejected_count})."
            ),
            missing_notes_to_insert=[],
            inserted_notes=[],
            rejected_candidate_notes=rejected_candidates,
            confidence_level=_confidence_from_score(missing_expected_block.confidence_score),
            action="skipped",
        )
        return report, []

    report = IncompleteBlockReport(
        block_type="missing_expected_block",
        missing_block_id=missing_expected_block.missing_block_id,
        target_bar_index=missing_expected_block.target_bar_index,
        target_start_sec=missing_expected_block.write_start_sec,
        target_end_sec=missing_expected_block.write_end_sec,
        expected_pattern_family_id=family.pattern_family_id,
        source_pattern_family_id=family.pattern_family_id,
        source_family_occurrence_count=family.occurrence_count,
        start_sec=missing_expected_block.write_start_sec,
        end_sec=missing_expected_block.write_end_sec,
        start_beat=missing_expected_block.write_start_beat,
        end_beat=missing_expected_block.write_end_beat,
        write_start_sec=missing_expected_block.write_start_sec,
        write_end_sec=missing_expected_block.write_end_sec,
        write_start_beat=missing_expected_block.write_start_beat,
        write_end_beat=missing_expected_block.write_end_beat,
        expected_duration_sec=missing_expected_block.expected_duration_sec,
        expected_duration_beat=missing_expected_block.expected_duration_beat,
        evidence_before_occurrences=list(missing_expected_block.evidence_before_occurrences),
        evidence_after_occurrences=list(missing_expected_block.evidence_after_occurrences),
        observed_note_count_in_region=missing_expected_block.detected_note_count_in_region,
        onset_slots_observed=list(missing_expected_block.observed_slots),
        onset_slots_expected=[int(slot) for slot in family.representative_onset_slots],
        onset_slots_missing=list(missing_expected_block.missing_slots),
        observed_slots=list(missing_expected_block.observed_slots),
        missing_slots=list(missing_expected_block.missing_slots),
        possible_matches=possible_matches,
        best_match_pattern_family_id=family.pattern_family_id,
        reason="missing_expected_pattern_occurrence",
        match_reason=(
            "missing_expected_pattern_occurrence. "
            f"Proposed={len(proposed)} accepted={len(validated)} rejected={rejected_count}."
        ),
        missing_notes_to_insert=validated,
        inserted_notes=validated,
        rejected_candidate_notes=rejected_candidates,
        confidence_level=_confidence_from_score(missing_expected_block.confidence_score),
        action="completed",
    )
    return report, validated


def _propose_missing_expected_notes(
    *,
    missing_expected_block: MissingExpectedBlock,
    family: PatternFamily,
    exemplar_block: PatternBlock,
    local_thresholds: _LocalNoteThresholds,
) -> list[ProposedCompletionNote]:
    exemplar_notes = _monophonic_block_view(exemplar_block).notes
    if not exemplar_notes:
        return []

    reference_velocity = int(statistics.median(note.velocity for note in exemplar_notes)) if exemplar_notes else 90
    normalized_velocity = int(round((reference_velocity + local_thresholds.median_velocity) / 2.0))
    normalized_velocity = max(local_thresholds.min_velocity, min(127, normalized_velocity))
    reference_channel = exemplar_notes[0].channel if exemplar_notes else 0

    grid_division = _grid_division_from_text(family.grid_resolution)
    sec_per_beat = (
        (missing_expected_block.write_end_sec - missing_expected_block.write_start_sec)
        / max(1e-6, missing_expected_block.write_end_beat - missing_expected_block.write_start_beat)
    )

    proposals: list[ProposedCompletionNote] = []
    for index, _rel_start in enumerate(family.representative_relative_onsets_sec):
        if index >= len(exemplar_notes):
            continue

        exemplar_note = exemplar_notes[index]
        onset_slot = (
            int(family.representative_onset_slots[index])
            if index < len(family.representative_onset_slots)
            else int(round(family.representative_relative_onsets_beat[index] * grid_division))
        )
        duration_slots = (
            int(family.representative_duration_slots[index])
            if index < len(family.representative_duration_slots)
            else max(1, int(round(family.representative_relative_durations_beat[index] * grid_division)))
        )

        rel_start_beat = float(onset_slot) / float(grid_division)
        rel_duration_beat = max(1.0 / float(grid_division), float(duration_slots) / float(grid_division))
        raw_start_beat = float(missing_expected_block.write_start_beat) + rel_start_beat
        raw_end_beat = raw_start_beat + rel_duration_beat
        if raw_end_beat <= missing_expected_block.write_start_beat:
            continue
        if raw_start_beat >= missing_expected_block.write_end_beat:
            continue

        start_beat = max(raw_start_beat, float(missing_expected_block.write_start_beat))
        end_beat = min(raw_end_beat, float(missing_expected_block.write_end_beat))
        if end_beat - start_beat <= (0.25 / float(grid_division)):
            continue

        start_sec = float(missing_expected_block.write_start_sec) + (
            (start_beat - float(missing_expected_block.write_start_beat)) * sec_per_beat
        )
        end_sec = float(missing_expected_block.write_start_sec) + (
            (end_beat - float(missing_expected_block.write_start_beat)) * sec_per_beat
        )

        proposals.append(
            ProposedCompletionNote(
                source_pattern_family_id=family.pattern_family_id,
                source_block_id=exemplar_block.block_id,
                source_note_index=index,
                note_id=f"mx_{missing_expected_block.missing_block_id}_{index:02d}",
                start_sec=round(start_sec, 6),
                end_sec=round(end_sec, 6),
                duration_sec=round(max(1e-6, end_sec - start_sec), 6),
                pitch_midi=int(exemplar_note.pitch_midi),
                pitch_name=exemplar_note.pitch_name,
                velocity=int(normalized_velocity),
                channel=reference_channel,
            )
        )

    proposals.sort(key=lambda item: (item.start_sec, item.end_sec, item.pitch_midi))
    return proposals


def _validate_missing_expected_note(
    *,
    note: ProposedCompletionNote,
    missing_expected_block: MissingExpectedBlock,
    family: PatternFamily,
    base_notes: list[_BaseNote],
    local_thresholds: _LocalNoteThresholds,
    stats: _CompletionStats,
    accepted_notes: list[ProposedCompletionNote],
) -> str | None:
    if note.end_sec <= note.start_sec:
        return "invalid_duration"

    if note.start_sec < (missing_expected_block.write_start_sec - 1e-6):
        return "outside_expected_write_region"

    if note.end_sec > (missing_expected_block.write_end_sec + 1e-6):
        return "outside_expected_write_region"

    sec_per_beat = (
        (missing_expected_block.write_end_sec - missing_expected_block.write_start_sec)
        / max(1e-6, missing_expected_block.write_end_beat - missing_expected_block.write_start_beat)
    )
    grid_division = _grid_division_from_text(family.grid_resolution)
    rel_start_beat = (note.start_sec - missing_expected_block.write_start_sec) / max(1e-6, sec_per_beat)
    onset_slot = int(round(rel_start_beat * grid_division))
    expected_slots = set(int(slot) for slot in family.representative_onset_slots)
    if onset_slot not in expected_slots:
        return "not_aligned_to_expected_slot"

    if note.pitch_midi < min(family.representative_pitch_set) or note.pitch_midi > max(family.representative_pitch_set):
        return "outside_pattern_pitch_range"

    if note.duration_sec < local_thresholds.min_duration_sec:
        stats.rejected_micro_note_count += 1
        return "micro_note"

    if note.velocity < local_thresholds.min_velocity:
        return "artifact_low_velocity"

    family_durations = family.representative_durations_sec
    if family_durations:
        median_duration = statistics.median(family_durations)
        if note.duration_sec > max(0.35, median_duration * 1.8):
            return "timing_not_matching_pattern"

    for existing in accepted_notes:
        onset_delta = abs(existing.start_sec - note.start_sec)
        overlap = min(existing.end_sec, note.end_sec) - max(existing.start_sec, note.start_sec)
        if onset_delta <= 0.03:
            stats.rejected_polyphonic_stack_count += 1
            return "polyphonic_stack"
        if overlap > 0.0:
            stats.rejected_polyphonic_stack_count += 1
            return "polyphonic_overlap"

    return _collision_with_base_notes(note=note, base_notes=base_notes)


def _family_duration_sec(family: PatternFamily) -> float:
    max_end = 0.0
    for index, onset in enumerate(family.representative_relative_onsets_sec):
        duration = (
            family.representative_durations_sec[index]
            if index < len(family.representative_durations_sec)
            else 0.0
        )
        max_end = max(max_end, float(onset) + float(duration))
    return max(0.05, float(max_end))


def _compatible_group_key(block: PatternBlock) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    # Quantize rhythm to 80 ms bins to keep deterministic grouping robust to small timing drift.
    rhythm_bins = tuple(int(round((value * 1000.0) / 80.0)) for value in block.rhythm_signature)
    return (
        int(block.note_count),
        tuple(int(item) for item in block.interval_sequence),
        rhythm_bins,
    )


def _family_from_block(*, block: PatternBlock, family_id: str) -> PatternFamily:
    mono = _monophonic_block_view(block)
    return PatternFamily(
        pattern_family_id=family_id,
        block_length_beats=float(mono.block_length_beats),
        time_signature=mono.time_signature,
        grid_resolution=mono.grid_resolution,
        representative_onset_slots=[int(note.onset_slot or 0) for note in mono.notes],
        representative_duration_slots=[int(note.duration_slots or 1) for note in mono.notes],
        representative_relative_onsets_beat=list(mono.relative_onsets_beat),
        representative_relative_durations_beat=list(mono.relative_durations_beat),
        representative_pitch_sequence=list(mono.pitch_sequence),
        representative_interval_sequence=list(mono.interval_sequence),
        representative_relative_onsets_sec=list(mono.relative_onsets_sec),
        representative_durations_sec=list(mono.relative_durations_sec),
        representative_pitch_set=list(mono.pitch_set),
        representative_note_count=mono.note_count,
        occurrence_count=1,
        occurrence_bars=[mono.bar_index],
        occurrences=[mono.block_id],
        first_seen_sec=mono.start_sec,
        last_seen_sec=mono.end_sec,
    )


def _infer_family_period(occurrence_blocks: list[PatternBlock]) -> _FamilyPeriodEstimate | None:
    if len(occurrence_blocks) < 2:
        return None

    diffs = [
        occurrence_blocks[index + 1].start_sec - occurrence_blocks[index].start_sec
        for index in range(len(occurrence_blocks) - 1)
        if occurrence_blocks[index + 1].start_sec > occurrence_blocks[index].start_sec
    ]
    if not diffs:
        return None

    base_period = min(diffs)
    normalized_periods: list[float] = []
    for diff in diffs:
        multiple = max(1, int(round(diff / max(1e-6, base_period))))
        normalized_periods.append(diff / float(multiple))

    period_sec = statistics.median(normalized_periods)
    if period_sec <= 0.03:
        return None

    variation = statistics.pstdev(normalized_periods) if len(normalized_periods) > 1 else 0.0
    stability = max(0.0, min(1.0, 1.0 - (variation / max(1e-6, period_sec))))
    return _FamilyPeriodEstimate(period_sec=float(period_sec), stability=float(stability))


def _count_notes_in_region(*, base_notes: list[_BaseNote], start_sec: float, end_sec: float) -> int:
    count = 0
    for note in base_notes:
        overlap = min(end_sec, note.end_sec) - max(start_sec, note.start_sec)
        if overlap > 0.0:
            count += 1
    return count


def _score_family_match(
    *,
    incomplete_block: PatternBlock,
    family: PatternFamily,
) -> _CandidateFamilyMatch | None:
    observed_pitch = list(incomplete_block.pitch_sequence)
    observed_slots = [int(note.onset_slot or 0) for note in incomplete_block.notes]
    observed_duration_slots = [int(note.duration_slots or 1) for note in incomplete_block.notes]
    family_pitch = list(family.representative_pitch_sequence)
    family_slots = list(family.representative_onset_slots)
    family_duration_slots = list(family.representative_duration_slots)

    if not observed_pitch or not family_pitch:
        return None
    if len(observed_pitch) >= len(family_pitch):
        return None

    pitch_prefix_len = _matching_prefix_len(observed_pitch, family_pitch)
    pitch_suffix_len = _matching_suffix_len(observed_pitch, family_pitch)

    slot_prefix_len = _matching_prefix_len(observed_slots, family_slots, tolerance=1)
    slot_suffix_len = _matching_suffix_len(observed_slots, family_slots, tolerance=1)
    duration_prefix_len = _matching_prefix_len(observed_duration_slots, family_duration_slots, tolerance=1)
    duration_suffix_len = _matching_suffix_len(observed_duration_slots, family_duration_slots, tolerance=1)

    interval_prefix_len = _matching_prefix_len(
        incomplete_block.interval_sequence,
        family.representative_interval_sequence,
        tolerance=1,
    )
    interval_suffix_len = _matching_suffix_len(
        incomplete_block.interval_sequence,
        family.representative_interval_sequence,
        tolerance=1,
    )

    best_pitch_match = max(pitch_prefix_len, pitch_suffix_len)
    if best_pitch_match < max(1, len(observed_pitch) - 1):
        internal = _matching_internal_fragment_len(observed_pitch, family_pitch)
        if internal < max(1, len(observed_pitch) - 1):
            return None
        best_pitch_match = internal

    pitch_score = best_pitch_match / float(len(observed_pitch))
    slot_score = max(slot_prefix_len, slot_suffix_len) / float(max(1, len(observed_slots)))
    duration_score = max(duration_prefix_len, duration_suffix_len) / float(max(1, len(observed_duration_slots)))
    interval_score = max(interval_prefix_len, interval_suffix_len) / float(
        max(1, len(incomplete_block.interval_sequence))
    )

    total = (pitch_score * 0.45) + (slot_score * 0.30) + (interval_score * 0.15) + (duration_score * 0.10)
    total = max(0.0, min(1.0, total))

    if total < 0.52:
        return None

    reason = (
        "Matched by deterministic pattern similarity: "
        f"pitch={pitch_score:.2f}, slots={slot_score:.2f}, interval={interval_score:.2f}, duration={duration_score:.2f}"
    )
    return _CandidateFamilyMatch(family=family, score=total, reason=reason)


def _matching_prefix_len(observed: list[int], full: list[int], tolerance: int = 0) -> int:
    count = 0
    for index, value in enumerate(observed):
        if index >= len(full):
            break
        if abs(int(value) - int(full[index])) <= tolerance:
            count += 1
        else:
            break
    return count


def _matching_suffix_len(observed: list[int], full: list[int], tolerance: int = 0) -> int:
    count = 0
    for index in range(1, len(observed) + 1):
        if index > len(full):
            break
        if abs(int(observed[-index]) - int(full[-index])) <= tolerance:
            count += 1
        else:
            break
    return count


def _matching_internal_fragment_len(observed: list[int], full: list[int]) -> int:
    if len(observed) > len(full):
        return 0
    best = 0
    for start in range(0, len(full) - len(observed) + 1):
        score = 0
        for offset, value in enumerate(observed):
            if int(value) == int(full[start + offset]):
                score += 1
            else:
                break
        best = max(best, score)
    return best


def _find_exemplar_block(family: PatternFamily, blocks_by_id: dict[str, PatternBlock]) -> PatternBlock | None:
    for block_id in family.occurrences:
        block = blocks_by_id.get(block_id)
        if block is not None and block.note_count >= len(family.representative_pitch_sequence):
            return block
    for block_id in family.occurrences:
        block = blocks_by_id.get(block_id)
        if block is not None:
            return block
    return None


def _propose_missing_notes(
    *,
    incomplete_block: PatternBlock,
    family: PatternFamily,
    exemplar_block: PatternBlock,
    local_thresholds: _LocalNoteThresholds,
) -> list[ProposedCompletionNote]:
    observed_count = len(incomplete_block.pitch_sequence)
    family_count = len(family.representative_pitch_sequence)

    proposals: list[ProposedCompletionNote] = []
    if observed_count >= family_count:
        return proposals

    block_start_sec = incomplete_block.start_sec
    block_end_sec = incomplete_block.end_sec
    block_start_beat = incomplete_block.start_beat
    block_end_beat = incomplete_block.end_beat
    sec_per_beat = incomplete_block.duration_sec / max(1e-6, incomplete_block.block_length_beats)
    grid_division = _grid_division_from_text(incomplete_block.grid_resolution)

    # Prefix completion is the primary deterministic path; suffix recovery is second.
    prefix_match = _matching_prefix_len(incomplete_block.pitch_sequence, family.representative_pitch_sequence)
    suffix_match = _matching_suffix_len(incomplete_block.pitch_sequence, family.representative_pitch_sequence)

    missing_indices: list[int] = []
    if prefix_match >= max(1, observed_count - 1):
        missing_indices = list(range(observed_count, family_count))
    elif suffix_match >= max(1, observed_count - 1):
        missing_indices = list(range(0, family_count - observed_count))
    else:
        internal_start = _internal_alignment_start(incomplete_block.pitch_sequence, family.representative_pitch_sequence)
        if internal_start is None:
            return proposals
        for index in range(family_count):
            aligned = internal_start <= index < (internal_start + observed_count)
            if not aligned:
                missing_indices.append(index)

    exemplar_notes = _monophonic_block_view(exemplar_block).notes
    if not exemplar_notes:
        return proposals

    reference_velocity = (
        int(statistics.median(note.velocity for note in incomplete_block.notes))
        if incomplete_block.notes
        else int(statistics.median(note.velocity for note in exemplar_notes))
    )
    normalized_velocity = int(round((reference_velocity + local_thresholds.median_velocity) / 2.0))
    normalized_velocity = max(local_thresholds.min_velocity, min(127, normalized_velocity))
    reference_channel = incomplete_block.notes[0].channel if incomplete_block.notes else 0

    for missing_index in missing_indices:
        if missing_index >= len(exemplar_notes):
            continue

        exemplar_note = exemplar_notes[missing_index]
        onset_slot = (
            int(family.representative_onset_slots[missing_index])
            if missing_index < len(family.representative_onset_slots)
            else int(round(family.representative_relative_onsets_beat[missing_index] * grid_division))
        )
        duration_slots = (
            int(family.representative_duration_slots[missing_index])
            if missing_index < len(family.representative_duration_slots)
            else max(1, int(round(family.representative_relative_durations_beat[missing_index] * grid_division)))
        )

        rel_start_beat = float(onset_slot) / float(grid_division)
        rel_duration_beat = max(1.0 / float(grid_division), float(duration_slots) / float(grid_division))
        start_beat = block_start_beat + rel_start_beat
        end_beat = min(block_end_beat, start_beat + rel_duration_beat)
        if end_beat <= start_beat + (0.25 / float(grid_division)):
            continue

        start_sec = block_start_sec + ((start_beat - block_start_beat) * sec_per_beat)
        end_sec = block_start_sec + ((end_beat - block_start_beat) * sec_per_beat)
        if start_sec < (block_start_sec - 1e-6) or end_sec > (block_end_sec + 1e-6):
            continue

        proposals.append(
            ProposedCompletionNote(
                source_pattern_family_id=family.pattern_family_id,
                source_block_id=exemplar_block.block_id,
                source_note_index=int(missing_index),
                note_id=f"pb_{incomplete_block.block_id}_{missing_index:02d}",
                start_sec=round(start_sec, 6),
                end_sec=round(end_sec, 6),
                duration_sec=round(max(1e-6, end_sec - start_sec), 6),
                pitch_midi=int(exemplar_note.pitch_midi),
                pitch_name=exemplar_note.pitch_name,
                velocity=int(normalized_velocity),
                channel=reference_channel,
            )
        )

    proposals.sort(key=lambda item: (item.start_sec, item.end_sec, item.pitch_midi))
    return proposals


def _internal_alignment_start(observed: list[int], full: list[int]) -> int | None:
    if len(observed) > len(full):
        return None
    best_start: int | None = None
    best_score = -1
    for start in range(0, len(full) - len(observed) + 1):
        score = 0
        for offset, value in enumerate(observed):
            if int(value) == int(full[start + offset]):
                score += 1
        if score > best_score:
            best_score = score
            best_start = start
    if best_score <= 0:
        return None
    return best_start


def _grid_division_from_text(value: str) -> int:
    text = str(value).strip()
    if "/" in text:
        tail = text.split("/", 1)[1]
    else:
        tail = text
    try:
        parsed = int(tail)
    except ValueError:
        return 16
    return max(1, parsed)


def _validate_proposed_note(
    *,
    note: ProposedCompletionNote,
    incomplete_block: PatternBlock,
    family: PatternFamily,
    base_notes: list[_BaseNote],
    local_thresholds: _LocalNoteThresholds,
    stats: _CompletionStats,
    accepted_notes: list[ProposedCompletionNote],
) -> str | None:
    if note.end_sec <= note.start_sec:
        return "invalid_duration"

    if note.start_sec < (incomplete_block.start_sec - 1e-6):
        return "outside_incomplete_block"

    if note.end_sec > (incomplete_block.end_sec + 1e-6):
        return "outside_incomplete_block"

    sec_per_beat = incomplete_block.duration_sec / max(1e-6, incomplete_block.block_length_beats)
    grid_division = _grid_division_from_text(incomplete_block.grid_resolution)
    rel_start_beat = (note.start_sec - incomplete_block.start_sec) / max(1e-6, sec_per_beat)
    onset_slot = int(round(rel_start_beat * grid_division))
    expected_slots = set(int(slot) for slot in family.representative_onset_slots)
    if onset_slot not in expected_slots:
        return "not_aligned_to_expected_slot"

    if note.pitch_midi < min(family.representative_pitch_set) or note.pitch_midi > max(family.representative_pitch_set):
        return "outside_pattern_pitch_range"

    if note.duration_sec < local_thresholds.min_duration_sec:
        stats.rejected_micro_note_count += 1
        return "micro_note"

    if note.velocity < local_thresholds.min_velocity:
        return "artifact_low_velocity"

    family_durations = family.representative_durations_sec
    if family_durations:
        median_duration = statistics.median(family_durations)
        if note.duration_sec <= 0.01 or abs(note.duration_sec - median_duration) > max(0.25, median_duration * 0.85):
            return "timing_not_matching_pattern"

    for existing in accepted_notes:
        onset_delta = abs(existing.start_sec - note.start_sec)
        overlap = min(existing.end_sec, note.end_sec) - max(existing.start_sec, note.start_sec)
        if onset_delta <= 0.03:
            stats.rejected_polyphonic_stack_count += 1
            return "polyphonic_stack"
        if overlap > 0.0:
            stats.rejected_polyphonic_stack_count += 1
            return "polyphonic_overlap"

    return _collision_with_base_notes(note=note, base_notes=base_notes)


def _collision_with_base_notes(*, note: ProposedCompletionNote, base_notes: list[_BaseNote]) -> str | None:
    for base_note in base_notes:
        onset_delta = abs(note.start_sec - base_note.start_sec)
        overlap = min(note.end_sec, base_note.end_sec) - max(note.start_sec, base_note.start_sec)
        if overlap <= 0.0:
            continue

        if note.pitch_midi == base_note.pitch_midi and onset_delta <= 0.02:
            return "duplicate_existing_base_note"

        overlap_ratio = overlap / max(1e-6, min(note.duration_sec, base_note.duration_sec))
        if overlap_ratio >= 0.65:
            return "collision_with_base_note"

    return None


def _confidence_from_score(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.68:
        return "medium"
    return "low"


def _prefer_monophonic_note(
    left: ProposedCompletionNote,
    right: ProposedCompletionNote,
) -> ProposedCompletionNote:
    left_key = (
        int(left.pitch_midi),
        -float(left.duration_sec),
        -int(left.velocity),
        str(left.source_pattern_family_id),
    )
    right_key = (
        int(right.pitch_midi),
        -float(right.duration_sec),
        -int(right.velocity),
        str(right.source_pattern_family_id),
    )
    return left if left_key <= right_key else right


def _dedupe_inserted_notes(
    notes: list[ProposedCompletionNote],
    *,
    stats: _CompletionStats | None = None,
) -> list[ProposedCompletionNote]:
    deduped: dict[tuple[int, int, int], ProposedCompletionNote] = {}
    for note in notes:
        key = (
            int(round(note.start_sec * 1000.0)),
            int(round(note.end_sec * 1000.0)),
            int(note.pitch_midi),
        )
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = note
            continue

        if note.source_pattern_family_id < existing.source_pattern_family_id:
            deduped[key] = note

    ordered = sorted(deduped.values(), key=lambda item: (item.start_sec, item.end_sec, item.pitch_midi))
    monophonic: list[ProposedCompletionNote] = []
    for note in ordered:
        replacement_index: int | None = None
        for index, existing in enumerate(monophonic):
            onset_delta = abs(float(existing.start_sec) - float(note.start_sec))
            overlap = min(float(existing.end_sec), float(note.end_sec)) - max(float(existing.start_sec), float(note.start_sec))
            if onset_delta <= 0.03 or overlap > 0.0:
                replacement_index = index
                break

        if replacement_index is None:
            monophonic.append(note)
            continue

        chosen = _prefer_monophonic_note(monophonic[replacement_index], note)
        if chosen is not monophonic[replacement_index] and stats is not None:
            stats.rejected_polyphonic_stack_count += 1
        elif stats is not None:
            stats.rejected_polyphonic_stack_count += 1
        monophonic[replacement_index] = chosen

    return sorted(monophonic, key=lambda item: (item.start_sec, item.end_sec, item.pitch_midi))


def _write_completion_midi(
    *,
    output_path: Path,
    notes: list[ProposedCompletionNote],
    ticks_per_beat: int,
    tempo_us_per_beat: int,
    project_duration_sec: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    midi_file = mido.MidiFile(ticks_per_beat=int(ticks_per_beat))
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="uzupelnienie", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=int(tempo_us_per_beat), time=0))

    ticks_per_second = (float(ticks_per_beat) * 1_000_000.0) / float(tempo_us_per_beat)

    absolute_events: list[tuple[int, int, mido.Message]] = []
    for note in notes:
        start_tick = max(0, int(round(float(note.start_sec) * ticks_per_second)))
        end_tick = max(start_tick + 1, int(round(float(note.end_sec) * ticks_per_second)))

        absolute_events.append(
            (
                start_tick,
                1,
                mido.Message(
                    "note_on",
                    note=int(note.pitch_midi),
                    velocity=max(1, min(127, int(note.velocity))),
                    channel=0 if note.channel is None else int(note.channel),
                    time=0,
                ),
            )
        )
        absolute_events.append(
            (
                end_tick,
                0,
                mido.Message(
                    "note_off",
                    note=int(note.pitch_midi),
                    velocity=0,
                    channel=0 if note.channel is None else int(note.channel),
                    time=0,
                ),
            )
        )

    project_end_tick = max(1, int(round(float(project_duration_sec) * ticks_per_second)))
    absolute_events.append((project_end_tick, 2, mido.MetaMessage("end_of_track", time=0)))

    absolute_events.sort(key=lambda item: (item[0], item[1]))
    previous_tick = 0
    for tick, _order, message in absolute_events:
        message.time = tick - previous_tick
        previous_tick = tick
        track.append(message)

    midi_file.save(str(output_path))


def _write_debug_midi(
    *,
    output_path: Path,
    blocks: list[PatternBlock],
    inserted_notes: list[ProposedCompletionNote],
    ticks_per_beat: int,
    tempo_us_per_beat: int,
    project_duration_sec: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    midi_file = mido.MidiFile(ticks_per_beat=int(ticks_per_beat))
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="pattern_blocks_debug", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=int(tempo_us_per_beat), time=0))

    ticks_per_second = (float(ticks_per_beat) * 1_000_000.0) / float(tempo_us_per_beat)

    family_channel_map: dict[str, int] = {}
    next_channel = 0

    absolute_events: list[tuple[int, int, mido.Message]] = []
    for block in blocks:
        family_id = block.assigned_pattern_family_id or "unknown"
        if family_id not in family_channel_map:
            family_channel_map[family_id] = next_channel % 12
            next_channel += 1
        channel = 12 if block.status == "incomplete" else family_channel_map[family_id]

        for note in block.notes:
            start_tick = max(0, int(round(float(note.start_sec) * ticks_per_second)))
            end_tick = max(start_tick + 1, int(round(float(note.end_sec) * ticks_per_second)))
            absolute_events.append(
                (
                    start_tick,
                    1,
                    mido.Message(
                        "note_on",
                        note=int(note.pitch_midi),
                        velocity=max(1, min(127, int(note.velocity))),
                        channel=int(channel),
                        time=0,
                    ),
                )
            )
            absolute_events.append(
                (
                    end_tick,
                    0,
                    mido.Message(
                        "note_off",
                        note=int(note.pitch_midi),
                        velocity=0,
                        channel=int(channel),
                        time=0,
                    ),
                )
            )

    for note in inserted_notes:
        start_tick = max(0, int(round(float(note.start_sec) * ticks_per_second)))
        end_tick = max(start_tick + 1, int(round(float(note.end_sec) * ticks_per_second)))
        absolute_events.append(
            (
                start_tick,
                1,
                mido.Message(
                    "note_on",
                    note=int(note.pitch_midi),
                    velocity=max(1, min(127, int(note.velocity))),
                    channel=15,
                    time=0,
                ),
            )
        )
        absolute_events.append(
            (
                end_tick,
                0,
                mido.Message(
                    "note_off",
                    note=int(note.pitch_midi),
                    velocity=0,
                    channel=15,
                    time=0,
                ),
            )
        )

    project_end_tick = max(1, int(round(float(project_duration_sec) * ticks_per_second)))
    absolute_events.append((project_end_tick, 2, mido.MetaMessage("end_of_track", time=0)))

    absolute_events.sort(key=lambda item: (item[0], item[1]))
    previous_tick = 0
    for tick, _order, message in absolute_events:
        message.time = tick - previous_tick
        previous_tick = tick
        track.append(message)

    midi_file.save(str(output_path))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
