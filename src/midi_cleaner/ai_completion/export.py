from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import mido

from midi_cleaner.ai_completion.models import (
    AIPatternCompletionNote,
    AIPatternCompletionOutput,
    AIPatternCompletionRejectedNote,
)
from midi_cleaner.ai_completion.pattern_pack import AllowedCompletionRegion, BasePatternNote


class AIPatternCompletionExportError(Exception):
    """Raised when AI completion notes cannot be validated or exported."""


@dataclass(frozen=True)
class AICompletionValidationResult:
    accepted_notes: list[AIPatternCompletionNote]
    rejected_notes: list[AIPatternCompletionRejectedNote]
    rejected_reason_counts: dict[str, int]
    pitch_range_used: dict[str, int | None]
    accepted_note_count_by_region: dict[str, int]
    warnings: list[str]


def validate_ai_completion_notes(
    ai_output: AIPatternCompletionOutput,
    *,
    base_notes: list[BasePatternNote],
    project_duration_sec: float,
    max_completion_notes: int,
    allowed_completion_regions: list[AllowedCompletionRegion],
) -> AICompletionValidationResult:
    if max_completion_notes <= 0:
        raise AIPatternCompletionExportError("max_completion_notes must be > 0")

    accepted: list[AIPatternCompletionNote] = []
    rejected: list[AIPatternCompletionRejectedNote] = []
    rejected_reason_counts: dict[str, int] = {}
    accepted_note_count_by_region: dict[str, int] = defaultdict(int)
    warnings: list[str] = []

    base_pitch_min = min(item.pitch_midi for item in base_notes)
    base_pitch_max = max(item.pitch_midi for item in base_notes)
    allowed_pitch_min = base_pitch_min - 12
    allowed_pitch_max = base_pitch_max + 12

    if not allowed_completion_regions:
        for note in sorted(ai_output.notes, key=lambda item: (item.start_sec, item.end_sec)):
            reason = "outside_allowed_completion_region"
            rejected.append(AIPatternCompletionRejectedNote(note_id=note.note_id, reason=reason))
            rejected_reason_counts[reason] = rejected_reason_counts.get(reason, 0) + 1
        warnings.append("No allowed completion regions were detected. AI notes were rejected.")
        return AICompletionValidationResult(
            accepted_notes=[],
            rejected_notes=rejected,
            rejected_reason_counts=rejected_reason_counts,
            pitch_range_used={"min": allowed_pitch_min, "max": allowed_pitch_max},
            accepted_note_count_by_region={},
            warnings=warnings,
        )

    regions_by_id = {region.region_id: region for region in allowed_completion_regions}
    candidate_region_counts: dict[str, int] = defaultdict(int)

    candidate_notes = sorted(ai_output.notes, key=lambda item: (item.start_sec, item.end_sec))

    for note in candidate_notes:
        reject_reason = _validate_note(
            note=note,
            base_notes=base_notes,
            project_duration_sec=project_duration_sec,
            allowed_pitch_min=allowed_pitch_min,
            allowed_pitch_max=allowed_pitch_max,
            allowed_completion_regions=allowed_completion_regions,
            candidate_region_counts=candidate_region_counts,
            regions_by_id=regions_by_id,
        )
        if reject_reason is not None:
            rejected.append(AIPatternCompletionRejectedNote(note_id=note.note_id, reason=reject_reason))
            rejected_reason_counts[reject_reason] = rejected_reason_counts.get(reject_reason, 0) + 1
            continue

        region_id = _find_note_region_id(note=note, allowed_completion_regions=allowed_completion_regions)
        if region_id is None:
            reason = "outside_allowed_completion_region"
            rejected.append(AIPatternCompletionRejectedNote(note_id=note.note_id, reason=reason))
            rejected_reason_counts[reason] = rejected_reason_counts.get(reason, 0) + 1
            continue

        if len(accepted) >= max_completion_notes:
            reason = "max_completion_notes_exceeded"
            rejected.append(AIPatternCompletionRejectedNote(note_id=note.note_id, reason=reason))
            rejected_reason_counts[reason] = rejected_reason_counts.get(reason, 0) + 1
            continue

        accepted.append(
            AIPatternCompletionNote(
                note_id=note.note_id,
                start_sec=float(note.start_sec),
                end_sec=float(note.end_sec),
                pitch_midi=int(note.pitch_midi),
                velocity=max(1, min(127, int(note.velocity))),
                confidence=float(note.confidence),
                reason=note.reason,
                pattern_reference_note_ids=list(note.pattern_reference_note_ids),
                risk=note.risk,
            )
        )
        candidate_region_counts[region_id] += 1
        accepted_note_count_by_region[region_id] += 1

    if not accepted:
        warnings.append("No AI completion notes passed validation.")

    return AICompletionValidationResult(
        accepted_notes=accepted,
        rejected_notes=rejected,
        rejected_reason_counts=rejected_reason_counts,
        pitch_range_used={"min": allowed_pitch_min, "max": allowed_pitch_max},
        accepted_note_count_by_region=dict(accepted_note_count_by_region),
        warnings=warnings,
    )


def _validate_note(
    *,
    note: AIPatternCompletionNote,
    base_notes: list[BasePatternNote],
    project_duration_sec: float,
    allowed_pitch_min: int,
    allowed_pitch_max: int,
    allowed_completion_regions: list[AllowedCompletionRegion],
    candidate_region_counts: dict[str, int],
    regions_by_id: dict[str, AllowedCompletionRegion],
) -> str | None:
    start_sec = float(note.start_sec)
    end_sec = float(note.end_sec)
    duration_sec = end_sec - start_sec

    if start_sec < 0.0 or end_sec < 0.0:
        return "negative_time"
    if end_sec <= start_sec:
        return "end_before_or_equal_start"
    if start_sec > project_duration_sec or end_sec > project_duration_sec:
        return "outside_project_duration"
    if duration_sec < 0.025:
        return "note_too_short"
    if duration_sec > 8.0 and float(note.confidence) < 0.9:
        return "note_too_long_low_confidence"
    if int(note.pitch_midi) < allowed_pitch_min or int(note.pitch_midi) > allowed_pitch_max:
        return "pitch_outside_allowed_range"

    for base_note in base_notes:
        if abs(int(note.pitch_midi) - int(base_note.pitch_midi)) > 1:
            continue

        onset_delta = abs(start_sec - float(base_note.start_sec))
        overlap_sec = min(end_sec, float(base_note.end_sec)) - max(start_sec, float(base_note.start_sec))
        if overlap_sec <= 0.0:
            continue

        overlap_ratio = overlap_sec / max(1e-6, min(duration_sec, float(base_note.duration_sec)))
        if onset_delta <= 0.03:
            return "duplicate_base_note_onset"
        if overlap_ratio >= 0.7:
            return "duplicate_base_note_overlap"

    region_id = _find_note_region_id(note=note, allowed_completion_regions=allowed_completion_regions)
    if region_id is None:
        return "outside_allowed_completion_region"

    region = regions_by_id.get(region_id)
    if region is None:
        return "outside_allowed_completion_region"

    if int(note.pitch_midi) < int(region.allowed_pitch_range["min"]):
        return "pitch_below_region_range"
    if int(note.pitch_midi) > int(region.allowed_pitch_range["max"]):
        return "pitch_above_region_range"
    if region.preferred_pitches and int(note.pitch_midi) not in set(int(value) for value in region.preferred_pitches):
        return "pitch_not_in_preferred_pitch_set"

    if duration_sec < float(region.min_note_duration_sec):
        return "note_too_short"

    projected_region_count = int(candidate_region_counts.get(region_id, 0)) + 1
    if projected_region_count > int(region.expected_note_count_max):
        return "region_expected_count_exceeded"

    region_duration_sec = max(1e-6, float(region.end_sec) - float(region.start_sec))
    projected_density = projected_region_count / region_duration_sec
    if projected_density > float(region.density_limit_notes_per_sec):
        return "region_note_density_too_high"

    return None


def _find_note_region_id(
    *,
    note: AIPatternCompletionNote,
    allowed_completion_regions: list[AllowedCompletionRegion],
) -> str | None:
    start_sec = float(note.start_sec)
    end_sec = float(note.end_sec)

    for region in allowed_completion_regions:
        if start_sec >= float(region.start_sec) and end_sec <= float(region.end_sec):
            return region.region_id
    return None


def export_ai_completion_midi(
    *,
    notes: list[AIPatternCompletionNote],
    output_midi_path: Path,
    ticks_per_beat: int,
    tempo_us_per_beat: int,
    track_name: str = "Hermes AI COMPLETION bass",
) -> None:
    output_midi_path.parent.mkdir(parents=True, exist_ok=True)

    midi = mido.MidiFile(ticks_per_beat=int(ticks_per_beat))
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=int(tempo_us_per_beat), time=0))

    ticks_per_second = (float(ticks_per_beat) * 1_000_000.0) / float(tempo_us_per_beat)
    absolute_events: list[tuple[int, int, mido.Message]] = []

    for note in sorted(notes, key=lambda item: (item.start_sec, item.end_sec, item.pitch_midi)):
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
                    channel=0,
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
                    channel=0,
                    time=0,
                ),
            )
        )

    absolute_events.sort(key=lambda item: (item[0], item[1]))
    prev_tick = 0
    for tick, _order, message in absolute_events:
        message.time = tick - prev_tick
        prev_tick = tick
        track.append(message)

    midi.save(str(output_midi_path))
