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
    start_sec: float
    end_sec: float
    duration_sec: float
    pitch_midi: int
    pitch_name: str
    velocity: int
    channel: int | None


@dataclass(frozen=True)
class _ParsedMidiTiming:
    ticks_per_beat: int
    tempo_us_per_beat: int
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
    report_path = analysis_dir / "pattern_completion_report.json"
    output_midi_path = midi_dir / "uzupelnienie.mid"
    debug_midi_path = debug_dir / "pattern_blocks_debug.mid"

    warnings: list[str] = []

    try:
        base_midi_path = _resolve_base_midi(project_dir)
        timing = _parse_midi_timing(base_midi_path)
        base_notes = _load_base_notes(base_midi_path, layer=params.layer)

        if not base_notes:
            raise PatternCompletionError(f"No base notes found in {base_midi_path}")

        blocks = _split_into_blocks(base_notes)
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
            )
            incomplete_existing_reports.append(report)
            inserted_notes.extend(new_notes)

        missing_expected_blocks = _detect_missing_expected_blocks(
            families=families,
            blocks_by_id=blocks_by_id,
            base_notes=base_notes,
        )
        missing_reports, missing_inserted_notes = _complete_missing_expected_blocks(
            missing_expected_blocks=missing_expected_blocks,
            blocks_by_id=blocks_by_id,
            families=families,
            base_notes=base_notes,
        )
        missing_expected_reports.extend(missing_reports)
        inserted_notes.extend(missing_inserted_notes)

        all_reports = list(incomplete_existing_reports) + list(missing_expected_reports)

        deduped_inserted_notes = _dedupe_inserted_notes(inserted_notes)

        _write_json(pattern_blocks_path, [item.model_dump(mode="json") for item in enriched_blocks])
        _write_json(pattern_families_path, [item.model_dump(mode="json") for item in families])
        _write_json(incomplete_blocks_path, [item.model_dump(mode="json") for item in all_reports])
        _write_json(
            missing_expected_blocks_path,
            [item.model_dump(mode="json") for item in missing_expected_reports],
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
            pattern_block_count=len(enriched_blocks),
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
            inserted_note_count=len(deduped_inserted_notes),
            output_midi_path=str(output_midi_path),
            pattern_blocks_file=str(pattern_blocks_path),
            pattern_families_file=str(pattern_families_path),
            incomplete_blocks_file=str(incomplete_blocks_path),
            missing_expected_blocks_file=str(missing_expected_blocks_path),
            debug_midi_path=debug_path_for_report,
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
            pattern_block_count=0,
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
            inserted_note_count=0,
            output_midi_path=None,
            pattern_blocks_file=str(pattern_blocks_path),
            pattern_families_file=str(pattern_families_path),
            incomplete_blocks_file=str(incomplete_blocks_path),
            missing_expected_blocks_file=str(missing_expected_blocks_path),
            debug_midi_path=None,
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
    duration_sec = float(midi_file.length)

    for track in midi_file.tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo_us_per_beat = int(message.tempo)
                break
        else:
            continue
        break

    return _ParsedMidiTiming(
        ticks_per_beat=ticks_per_beat,
        tempo_us_per_beat=tempo_us_per_beat,
        duration_sec=duration_sec,
    )


def _load_base_notes(base_midi_path: Path, layer: str) -> list[_BaseNote]:
    note_document, _report = import_midi_candidate(base_midi_path, source="working", layer=layer)
    notes: list[_BaseNote] = []

    for note in sorted(note_document.notes, key=lambda item: (item.start_sec, item.end_sec, item.pitch_midi)):
        notes.append(
            _BaseNote(
                note_id=note.note_id,
                start_sec=float(note.start_sec),
                end_sec=float(note.end_sec),
                duration_sec=float(note.duration_sec),
                pitch_midi=int(note.pitch_midi),
                pitch_name=note.pitch_name,
                velocity=int(note.velocity),
                channel=note.channel,
            )
        )

    return notes


def _split_into_blocks(notes: list[_BaseNote]) -> list[PatternBlock]:
    if not notes:
        return []

    onset_intervals = [
        notes[index + 1].start_sec - notes[index].start_sec
        for index in range(len(notes) - 1)
        if notes[index + 1].start_sec > notes[index].start_sec
    ]
    median_interval = statistics.median(onset_intervals) if onset_intervals else 0.35
    boundary_gap = max(0.5, median_interval * 2.4)

    groups: list[list[_BaseNote]] = []
    current: list[_BaseNote] = [notes[0]]
    for note in notes[1:]:
        previous = current[-1]
        if (note.start_sec - previous.start_sec) >= boundary_gap:
            groups.append(current)
            current = [note]
        else:
            current.append(note)
    groups.append(current)

    blocks: list[PatternBlock] = []
    for index, group in enumerate(groups, start=1):
        block_start = group[0].start_sec
        block_end = group[-1].end_sec
        note_records: list[PatternBlockNote] = []
        relative_onsets: list[float] = []
        relative_durations: list[float] = []
        pitches: list[int] = []
        pitch_names: list[str] = []

        for note in group:
            note_records.append(
                PatternBlockNote(
                    note_id=note.note_id,
                    start_sec=round(note.start_sec, 6),
                    end_sec=round(note.end_sec, 6),
                    duration_sec=round(note.duration_sec, 6),
                    pitch_midi=note.pitch_midi,
                    pitch_name=note.pitch_name,
                    velocity=note.velocity,
                    channel=note.channel,
                )
            )
            relative_onsets.append(round(note.start_sec - block_start, 6))
            relative_durations.append(round(note.duration_sec, 6))
            pitches.append(note.pitch_midi)
            pitch_names.append(note.pitch_name)

        intervals = [
            int(pitches[item + 1] - pitches[item])
            for item in range(len(pitches) - 1)
        ]
        rhythm_signature = [
            round(relative_onsets[item + 1] - relative_onsets[item], 6)
            for item in range(len(relative_onsets) - 1)
        ]

        blocks.append(
            PatternBlock(
                block_id=f"block_{index:04d}",
                start_sec=round(block_start, 6),
                end_sec=round(block_end, 6),
                duration_sec=round(block_end - block_start, 6),
                note_count=len(group),
                notes=note_records,
                relative_onsets_sec=relative_onsets,
                relative_durations_sec=relative_durations,
                pitch_sequence=pitches,
                pitch_names=pitch_names,
                interval_sequence=intervals,
                rhythm_signature=rhythm_signature,
                pitch_set=sorted(set(pitches)),
                assigned_pattern_family_id=None,
                status="unknown",
            )
        )

    return blocks


def _build_pattern_families(
    blocks: list[PatternBlock],
) -> list[PatternFamily]:
    if not blocks:
        return []

    note_count_hist = Counter(block.note_count for block in blocks if block.note_count > 0)
    if not note_count_hist:
        return []

    complete_note_count = sorted(
        note_count_hist.items(),
        key=lambda item: (-item[1], -item[0]),
    )[0][0]

    complete_candidates = [
        block
        for block in blocks
        if block.note_count == complete_note_count
    ]

    families_by_signature: dict[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        list[PatternBlock],
    ] = defaultdict(list)

    for block in complete_candidates:
        duration_quantized = tuple(
            int(round(value * 1000.0)) for value in block.relative_durations_sec
        )
        rhythm_quantized = tuple(int(round(value * 1000.0)) for value in block.rhythm_signature)
        pitch_signature = tuple(block.pitch_sequence)
        signature = (pitch_signature, rhythm_quantized, duration_quantized)
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
            key=lambda item: (item.note_count, item.duration_sec),
        )

        families.append(
            PatternFamily(
                pattern_family_id=family_id,
                representative_pitch_sequence=list(representative.pitch_sequence),
                representative_interval_sequence=list(representative.interval_sequence),
                representative_relative_onsets_sec=list(representative.relative_onsets_sec),
                representative_durations_sec=list(representative.relative_durations_sec),
                representative_pitch_set=list(representative.pitch_set),
                occurrence_count=len(grouped_blocks),
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


def _is_exact_family_match(*, block: PatternBlock, family: PatternFamily) -> bool:
    if len(block.pitch_sequence) != len(family.representative_pitch_sequence):
        return False

    if block.pitch_sequence != family.representative_pitch_sequence:
        return False

    if len(block.rhythm_signature) != len(family.representative_relative_onsets_sec) - 1:
        return False

    block_rhythm = [
        int(round(value * 1000.0))
        for value in block.rhythm_signature
    ]
    family_rhythm = [
        int(round(family.representative_relative_onsets_sec[index + 1] * 1000.0))
        - int(round(family.representative_relative_onsets_sec[index] * 1000.0))
        for index in range(len(family.representative_relative_onsets_sec) - 1)
    ]
    if block_rhythm != family_rhythm:
        return False

    return True


def _complete_incomplete_block(
    *,
    incomplete_block: PatternBlock,
    blocks_by_id: dict[str, PatternBlock],
    families: list[PatternFamily],
    base_notes: list[_BaseNote],
) -> tuple[IncompleteBlockReport, list[ProposedCompletionNote]]:
    possible_matches: list[_CandidateFamilyMatch] = []

    for family in families:
        candidate = _score_family_match(incomplete_block=incomplete_block, family=family)
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
            start_sec=incomplete_block.start_sec,
            end_sec=incomplete_block.end_sec,
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
            start_sec=incomplete_block.start_sec,
            end_sec=incomplete_block.end_sec,
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

    exemplar_block = _find_exemplar_block(best.family, blocks_by_id)
    if exemplar_block is None:
        report = IncompleteBlockReport(
            block_type="incomplete_existing_block",
            incomplete_block_id=incomplete_block.block_id,
            start_sec=incomplete_block.start_sec,
            end_sec=incomplete_block.end_sec,
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
        incomplete_block=incomplete_block,
        family=best.family,
        exemplar_block=exemplar_block,
    )

    validated: list[ProposedCompletionNote] = []
    rejected_count = 0
    for note in proposed:
        reason = _validate_proposed_note(
            note=note,
            incomplete_block=incomplete_block,
            family=best.family,
            base_notes=base_notes,
        )
        if reason is None:
            validated.append(note)
        else:
            rejected_count += 1

    if not validated:
        report = IncompleteBlockReport(
            block_type="incomplete_existing_block",
            incomplete_block_id=incomplete_block.block_id,
            start_sec=incomplete_block.start_sec,
            end_sec=incomplete_block.end_sec,
            observed_pitch_sequence=list(incomplete_block.pitch_sequence),
            observed_relative_onsets_sec=list(incomplete_block.relative_onsets_sec),
            possible_matches=match_models,
            best_match_pattern_family_id=best.family.pattern_family_id,
            reason="rejected_validation",
            match_reason=f"{best.reason}. All proposed notes rejected during validation.",
            missing_notes_to_insert=[],
            confidence_level=_confidence_from_score(best.score),
            action="skipped",
        )
        return report, []

    report = IncompleteBlockReport(
        block_type="incomplete_existing_block",
        incomplete_block_id=incomplete_block.block_id,
        start_sec=incomplete_block.start_sec,
        end_sec=incomplete_block.end_sec,
        observed_pitch_sequence=list(incomplete_block.pitch_sequence),
        observed_relative_onsets_sec=list(incomplete_block.relative_onsets_sec),
        possible_matches=match_models,
        best_match_pattern_family_id=best.family.pattern_family_id,
        reason="completed",
        match_reason=(
            f"{best.reason}. Proposed={len(proposed)} accepted={len(validated)} rejected={rejected_count}."
        ),
        missing_notes_to_insert=validated,
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
    candidates: list[MissingExpectedBlock] = []
    candidate_counter = 0

    compatible_groups: dict[tuple[int, tuple[int, ...], tuple[int, ...]], list[PatternBlock]] = defaultdict(list)
    all_blocks = sorted(blocks_by_id.values(), key=lambda item: item.start_sec)
    for block in all_blocks:
        if block.note_count < 2:
            continue
        compatible_groups[_compatible_group_key(block)].append(block)

    for grouped_blocks in compatible_groups.values():
        if len(grouped_blocks) < 2:
            continue

        occurrence_blocks = list(grouped_blocks)
        occurrence_blocks.sort(key=lambda item: item.start_sec)

        period_estimate = _infer_family_period(occurrence_blocks)
        if period_estimate is None:
            continue

        expected_duration_sec = statistics.median(
            block.duration_sec for block in occurrence_blocks
        )
        sparse_threshold = max(
            1,
            int(math.floor(statistics.median(block.note_count for block in occurrence_blocks) * 0.80)),
        )

        for index in range(len(occurrence_blocks) - 1):
            before = occurrence_blocks[index]
            after = occurrence_blocks[index + 1]
            gap_sec = after.start_sec - before.start_sec
            if gap_sec <= period_estimate.period_sec * 1.45:
                continue

            estimated_step_count = int(round(gap_sec / period_estimate.period_sec))
            missing_count = estimated_step_count - 1
            if missing_count <= 0:
                continue

            expected_gap_sec = (missing_count + 1) * period_estimate.period_sec
            gap_error_sec = abs(gap_sec - expected_gap_sec)
            if gap_error_sec > max(0.22, period_estimate.period_sec * 0.22):
                continue

            for step in range(1, missing_count + 1):
                write_start_sec = before.start_sec + (period_estimate.period_sec * step)
                write_end_sec = min(write_start_sec + expected_duration_sec, after.start_sec)
                if write_end_sec <= write_start_sec + 0.03:
                    continue

                detected_note_count = _count_notes_in_region(
                    base_notes=base_notes,
                    start_sec=write_start_sec,
                    end_sec=write_end_sec,
                )
                if detected_note_count > sparse_threshold:
                    continue

                sparsity_score = 1.0 - min(1.0, detected_note_count / float(sparse_threshold + 1))
                gap_fit_score = max(
                    0.0,
                    1.0
                    - (
                        gap_error_sec
                        / max(1e-6, max(0.22, period_estimate.period_sec * 0.22))
                    ),
                )
                confidence_score = max(
                    0.0,
                    min(
                        1.0,
                        0.22
                        + (0.35 * period_estimate.stability)
                        + (0.25 * gap_fit_score)
                        + (0.15 * sparsity_score)
                        + (0.03 if len(occurrence_blocks) >= 3 else 0.0),
                    ),
                )

                candidate_family_ids: list[tuple[str, float]] = []
                before_family = before.assigned_pattern_family_id
                after_family = after.assigned_pattern_family_id

                if before_family and after_family and before_family == after_family:
                    candidate_family_ids.append((before_family, min(1.0, confidence_score + 0.08)))
                else:
                    if before_family:
                        candidate_family_ids.append((before_family, confidence_score))
                    else:
                        candidate_family_ids.append((f"synthetic::{before.block_id}", confidence_score - 0.02))

                    if after_family and after_family != before_family:
                        candidate_family_ids.append((after_family, max(0.0, confidence_score - 0.03)))
                    elif not after_family:
                        candidate_family_ids.append((f"synthetic::{after.block_id}", max(0.0, confidence_score - 0.03)))

                deduped_families: dict[str, float] = {}
                for family_id, score in candidate_family_ids:
                    existing = deduped_families.get(family_id)
                    if existing is None or score > existing:
                        deduped_families[family_id] = score

                for family_id, family_score in sorted(deduped_families.items()):
                    candidate_counter += 1
                    reason = (
                        "missing_expected_pattern_occurrence; "
                        f"period={period_estimate.period_sec:.3f}s stability={period_estimate.stability:.2f} "
                        f"gap={gap_sec:.3f}s estimated_missing={missing_count} "
                        f"region_notes={detected_note_count} compatible_group_size={len(occurrence_blocks)}"
                    )
                    candidates.append(
                        MissingExpectedBlock(
                            missing_block_id=f"missing_{candidate_counter:04d}",
                            expected_pattern_family_id=family_id,
                            write_start_sec=round(write_start_sec, 6),
                            write_end_sec=round(write_end_sec, 6),
                            expected_duration_sec=round(expected_duration_sec, 6),
                            evidence_before_occurrences=[before.block_id],
                            evidence_after_occurrences=[after.block_id],
                            detected_note_count_in_region=detected_note_count,
                            confidence_score=round(max(0.0, min(1.0, family_score)), 6),
                            possible_matches=[
                                IncompleteBlockMatch(
                                    pattern_family_id=family_id,
                                    score=round(max(0.0, min(1.0, family_score)), 6),
                                    reason=reason,
                                )
                            ],
                        )
                    )

    deduped: dict[tuple[str, int, int], MissingExpectedBlock] = {}
    for candidate in candidates:
        family_id = candidate.expected_pattern_family_id or "unknown"
        key = (
            family_id,
            int(round(candidate.write_start_sec * 100.0)),
            int(round(candidate.write_end_sec * 100.0)),
        )
        existing = deduped.get(key)
        if existing is None or candidate.confidence_score > existing.confidence_score:
            deduped[key] = candidate

    return sorted(
        deduped.values(),
        key=lambda item: (item.write_start_sec, item.write_end_sec, -(item.confidence_score)),
    )


def _complete_missing_expected_blocks(
    *,
    missing_expected_blocks: list[MissingExpectedBlock],
    blocks_by_id: dict[str, PatternBlock],
    families: list[PatternFamily],
    base_notes: list[_BaseNote],
) -> tuple[list[IncompleteBlockReport], list[ProposedCompletionNote]]:
    if not missing_expected_blocks:
        return [], []

    families_by_id = {family.pattern_family_id: family for family in families}
    region_groups: dict[tuple[int, int], list[MissingExpectedBlock]] = defaultdict(list)

    for block in missing_expected_blocks:
        key = (
            int(round(block.write_start_sec * 1000.0 / 120.0)),
            int(round(block.write_end_sec * 1000.0 / 120.0)),
        )
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
                    expected_pattern_family_id=None,
                    start_sec=best.write_start_sec,
                    end_sec=best.write_end_sec,
                    write_start_sec=best.write_start_sec,
                    write_end_sec=best.write_end_sec,
                    expected_duration_sec=best.expected_duration_sec,
                    evidence_before_occurrences=list(best.evidence_before_occurrences),
                    evidence_after_occurrences=list(best.evidence_after_occurrences),
                    observed_note_count_in_region=best.detected_note_count_in_region,
                    possible_matches=possible_matches,
                    best_match_pattern_family_id=None,
                    reason="ambiguous",
                    match_reason=(
                        "Ambiguous deterministic match for missing expected block. "
                        f"Top families are too close: {best.expected_pattern_family_id}={best.confidence_score:.3f}, "
                        f"{second.expected_pattern_family_id}={second.confidence_score:.3f}"
                    ),
                    missing_notes_to_insert=[],
                    confidence_level="low",
                    action="skipped",
                )
            )
            continue

        if best.expected_pattern_family_id is None or best.confidence_score < 0.62:
            reports.append(
                IncompleteBlockReport(
                    block_type="missing_expected_block",
                    missing_block_id=best.missing_block_id,
                    expected_pattern_family_id=best.expected_pattern_family_id,
                    start_sec=best.write_start_sec,
                    end_sec=best.write_end_sec,
                    write_start_sec=best.write_start_sec,
                    write_end_sec=best.write_end_sec,
                    expected_duration_sec=best.expected_duration_sec,
                    evidence_before_occurrences=list(best.evidence_before_occurrences),
                    evidence_after_occurrences=list(best.evidence_after_occurrences),
                    observed_note_count_in_region=best.detected_note_count_in_region,
                    possible_matches=possible_matches,
                    best_match_pattern_family_id=best.expected_pattern_family_id,
                    reason="no_clear_family",
                    match_reason="No clear deterministic family evidence for missing expected block.",
                    missing_notes_to_insert=[],
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
                    expected_pattern_family_id=best.expected_pattern_family_id,
                    start_sec=best.write_start_sec,
                    end_sec=best.write_end_sec,
                    write_start_sec=best.write_start_sec,
                    write_end_sec=best.write_end_sec,
                    expected_duration_sec=best.expected_duration_sec,
                    evidence_before_occurrences=list(best.evidence_before_occurrences),
                    evidence_after_occurrences=list(best.evidence_after_occurrences),
                    observed_note_count_in_region=best.detected_note_count_in_region,
                    possible_matches=possible_matches,
                    best_match_pattern_family_id=best.expected_pattern_family_id,
                    reason="no_clear_family",
                    match_reason="No clear deterministic family payload for missing expected block.",
                    missing_notes_to_insert=[],
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
) -> tuple[IncompleteBlockReport, list[ProposedCompletionNote]]:
    exemplar_block = _find_exemplar_block(family, blocks_by_id)
    if exemplar_block is None:
        report = IncompleteBlockReport(
            block_type="missing_expected_block",
            missing_block_id=missing_expected_block.missing_block_id,
            expected_pattern_family_id=family.pattern_family_id,
            start_sec=missing_expected_block.write_start_sec,
            end_sec=missing_expected_block.write_end_sec,
            write_start_sec=missing_expected_block.write_start_sec,
            write_end_sec=missing_expected_block.write_end_sec,
            expected_duration_sec=missing_expected_block.expected_duration_sec,
            evidence_before_occurrences=list(missing_expected_block.evidence_before_occurrences),
            evidence_after_occurrences=list(missing_expected_block.evidence_after_occurrences),
            observed_note_count_in_region=missing_expected_block.detected_note_count_in_region,
            possible_matches=possible_matches,
            best_match_pattern_family_id=family.pattern_family_id,
            reason="no_clear_family",
            match_reason="No clear deterministic family payload for missing expected block.",
            missing_notes_to_insert=[],
            confidence_level="low",
            action="skipped",
        )
        return report, []

    proposed = _propose_missing_expected_notes(
        missing_expected_block=missing_expected_block,
        family=family,
        exemplar_block=exemplar_block,
    )

    validated: list[ProposedCompletionNote] = []
    rejected_count = 0
    for note in proposed:
        reason = _validate_missing_expected_note(
            note=note,
            missing_expected_block=missing_expected_block,
            family=family,
            base_notes=base_notes,
        )
        if reason is None:
            validated.append(note)
        else:
            rejected_count += 1

    if not validated:
        report = IncompleteBlockReport(
            block_type="missing_expected_block",
            missing_block_id=missing_expected_block.missing_block_id,
            expected_pattern_family_id=family.pattern_family_id,
            start_sec=missing_expected_block.write_start_sec,
            end_sec=missing_expected_block.write_end_sec,
            write_start_sec=missing_expected_block.write_start_sec,
            write_end_sec=missing_expected_block.write_end_sec,
            expected_duration_sec=missing_expected_block.expected_duration_sec,
            evidence_before_occurrences=list(missing_expected_block.evidence_before_occurrences),
            evidence_after_occurrences=list(missing_expected_block.evidence_after_occurrences),
            observed_note_count_in_region=missing_expected_block.detected_note_count_in_region,
            possible_matches=possible_matches,
            best_match_pattern_family_id=family.pattern_family_id,
            reason="rejected_validation",
            match_reason=(
                "missing_expected_pattern_occurrence. "
                f"All proposed notes rejected during validation (rejected={rejected_count})."
            ),
            missing_notes_to_insert=[],
            confidence_level=_confidence_from_score(missing_expected_block.confidence_score),
            action="skipped",
        )
        return report, []

    report = IncompleteBlockReport(
        block_type="missing_expected_block",
        missing_block_id=missing_expected_block.missing_block_id,
        expected_pattern_family_id=family.pattern_family_id,
        start_sec=missing_expected_block.write_start_sec,
        end_sec=missing_expected_block.write_end_sec,
        write_start_sec=missing_expected_block.write_start_sec,
        write_end_sec=missing_expected_block.write_end_sec,
        expected_duration_sec=missing_expected_block.expected_duration_sec,
        evidence_before_occurrences=list(missing_expected_block.evidence_before_occurrences),
        evidence_after_occurrences=list(missing_expected_block.evidence_after_occurrences),
        observed_note_count_in_region=missing_expected_block.detected_note_count_in_region,
        possible_matches=possible_matches,
        best_match_pattern_family_id=family.pattern_family_id,
        reason="missing_expected_pattern_occurrence",
        match_reason=(
            "missing_expected_pattern_occurrence. "
            f"Proposed={len(proposed)} accepted={len(validated)} rejected={rejected_count}."
        ),
        missing_notes_to_insert=validated,
        confidence_level=_confidence_from_score(missing_expected_block.confidence_score),
        action="completed",
    )
    return report, validated


def _propose_missing_expected_notes(
    *,
    missing_expected_block: MissingExpectedBlock,
    family: PatternFamily,
    exemplar_block: PatternBlock,
) -> list[ProposedCompletionNote]:
    exemplar_notes = exemplar_block.notes
    if not exemplar_notes:
        return []

    reference_velocity = int(statistics.median(note.velocity for note in exemplar_notes)) if exemplar_notes else 90
    reference_channel = exemplar_notes[0].channel if exemplar_notes else 0

    proposals: list[ProposedCompletionNote] = []
    for index, rel_start in enumerate(family.representative_relative_onsets_sec):
        if index >= len(exemplar_notes):
            continue

        exemplar_note = exemplar_notes[index]
        rel_duration = (
            family.representative_durations_sec[index]
            if index < len(family.representative_durations_sec)
            else exemplar_note.duration_sec
        )

        raw_start = float(missing_expected_block.write_start_sec) + float(rel_start)
        raw_end = raw_start + float(rel_duration)
        if raw_end <= missing_expected_block.write_start_sec:
            continue
        if raw_start >= missing_expected_block.write_end_sec:
            continue

        start_sec = max(raw_start, float(missing_expected_block.write_start_sec))
        end_sec = min(raw_end, float(missing_expected_block.write_end_sec))
        if end_sec - start_sec <= 0.01:
            continue

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
                velocity=int(reference_velocity),
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
) -> str | None:
    if note.end_sec <= note.start_sec:
        return "invalid_duration"

    if note.start_sec < (missing_expected_block.write_start_sec - 1e-6):
        return "outside_expected_write_region"

    if note.end_sec > (missing_expected_block.write_end_sec + 1e-6):
        return "outside_expected_write_region"

    if note.pitch_midi < min(family.representative_pitch_set) or note.pitch_midi > max(family.representative_pitch_set):
        return "outside_pattern_pitch_range"

    if note.duration_sec <= 0.01:
        return "timing_not_matching_pattern"

    family_durations = family.representative_durations_sec
    if family_durations:
        median_duration = statistics.median(family_durations)
        if note.duration_sec > max(0.35, median_duration * 1.8):
            return "timing_not_matching_pattern"

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
    return PatternFamily(
        pattern_family_id=family_id,
        representative_pitch_sequence=list(block.pitch_sequence),
        representative_interval_sequence=list(block.interval_sequence),
        representative_relative_onsets_sec=list(block.relative_onsets_sec),
        representative_durations_sec=list(block.relative_durations_sec),
        representative_pitch_set=list(block.pitch_set),
        occurrence_count=1,
        occurrences=[block.block_id],
        first_seen_sec=block.start_sec,
        last_seen_sec=block.end_sec,
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
    observed_onsets = list(incomplete_block.relative_onsets_sec)
    family_pitch = list(family.representative_pitch_sequence)
    family_onsets = list(family.representative_relative_onsets_sec)

    if not observed_pitch or not family_pitch:
        return None
    if len(observed_pitch) >= len(family_pitch):
        return None

    pitch_prefix_len = _matching_prefix_len(observed_pitch, family_pitch)
    pitch_suffix_len = _matching_suffix_len(observed_pitch, family_pitch)

    rhythm_prefix_len = _matching_prefix_len(
        [int(round(item * 1000.0)) for item in observed_onsets],
        [int(round(item * 1000.0)) for item in family_onsets],
        tolerance=35,
    )
    rhythm_suffix_len = _matching_suffix_len(
        [int(round(item * 1000.0)) for item in observed_onsets],
        [int(round(item * 1000.0)) for item in family_onsets],
        tolerance=35,
    )

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
    rhythm_score = max(rhythm_prefix_len, rhythm_suffix_len) / float(max(1, len(observed_onsets)))
    interval_score = max(interval_prefix_len, interval_suffix_len) / float(
        max(1, len(incomplete_block.interval_sequence))
    )

    total = (pitch_score * 0.55) + (rhythm_score * 0.30) + (interval_score * 0.15)
    total = max(0.0, min(1.0, total))

    if total < 0.55:
        return None

    reason = (
        "Matched by deterministic pattern similarity: "
        f"pitch={pitch_score:.2f}, rhythm={rhythm_score:.2f}, interval={interval_score:.2f}"
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
) -> list[ProposedCompletionNote]:
    observed_count = len(incomplete_block.pitch_sequence)
    family_count = len(family.representative_pitch_sequence)

    proposals: list[ProposedCompletionNote] = []
    if observed_count >= family_count:
        return proposals

    block_start = incomplete_block.start_sec
    block_end = incomplete_block.end_sec

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

    exemplar_notes = exemplar_block.notes
    if not exemplar_notes:
        return proposals

    reference_velocity = int(statistics.median(note.velocity for note in incomplete_block.notes)) if incomplete_block.notes else 90
    reference_channel = incomplete_block.notes[0].channel if incomplete_block.notes else 0

    for missing_index in missing_indices:
        if missing_index >= len(exemplar_notes):
            continue

        exemplar_note = exemplar_notes[missing_index]
        rel_start = family.representative_relative_onsets_sec[missing_index]
        rel_duration = family.representative_durations_sec[missing_index]

        start_sec = block_start + float(rel_start)
        end_sec = start_sec + float(rel_duration)

        # Allow tail-note completion immediately after the observed incomplete fragment.
        if end_sec < block_start:
            continue
        if start_sec > (block_end + max(0.35, incomplete_block.duration_sec * 0.6)):
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
                velocity=int(reference_velocity),
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


def _validate_proposed_note(
    *,
    note: ProposedCompletionNote,
    incomplete_block: PatternBlock,
    family: PatternFamily,
    base_notes: list[_BaseNote],
) -> str | None:
    if note.end_sec <= note.start_sec:
        return "invalid_duration"

    if note.start_sec < (incomplete_block.start_sec - 1e-6):
        return "outside_incomplete_block"

    family_span_end = 0.0
    for index, onset in enumerate(family.representative_relative_onsets_sec):
        duration = (
            family.representative_durations_sec[index]
            if index < len(family.representative_durations_sec)
            else 0.0
        )
        family_span_end = max(family_span_end, float(onset) + float(duration))

    allowed_write_end = max(
        incomplete_block.end_sec + max(0.35, incomplete_block.duration_sec * 0.6),
        incomplete_block.start_sec + family_span_end + 0.05,
    )
    if note.end_sec > (allowed_write_end + 1e-6):
        return "outside_incomplete_block"

    if note.pitch_midi < min(family.representative_pitch_set) or note.pitch_midi > max(family.representative_pitch_set):
        return "outside_pattern_pitch_range"

    family_durations = family.representative_durations_sec
    if family_durations:
        median_duration = statistics.median(family_durations)
        if note.duration_sec <= 0.01 or abs(note.duration_sec - median_duration) > max(0.25, median_duration * 0.85):
            return "timing_not_matching_pattern"

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


def _dedupe_inserted_notes(notes: list[ProposedCompletionNote]) -> list[ProposedCompletionNote]:
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

    return sorted(deduped.values(), key=lambda item: (item.start_sec, item.end_sec, item.pitch_midi))


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
