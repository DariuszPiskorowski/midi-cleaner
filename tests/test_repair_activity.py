from __future__ import annotations

from pathlib import Path

from midi_cleaner.audio.models import AudioFeatureDocument, AudioFrameFeature, AudioGlobalFeatures
from midi_cleaner.cleanup.models import CleanupAction, CleanupPlanDocument
from midi_cleaner.repair.activity import (
    ActivityRepairParameters,
    _ActivityFrame,
    _build_audio_activity_regions,
    repair_activity,
)
from midi_cleaner.refinement.models import RefinedNoteDocument, RefinedNoteEvent


def _build_audio_document(
    duration_sec: float,
    frame_step_sec: float,
    rms_fn,
    onset_fn,
) -> AudioFeatureDocument:
    frames: list[AudioFrameFeature] = []
    frame_count = int(duration_sec / frame_step_sec)
    for idx in range(frame_count):
        start = idx * frame_step_sec
        end = start + frame_step_sec
        rms = float(rms_fn(start))
        onset = float(onset_fn(start))
        frames.append(
            AudioFrameFeature(
                frame_index=idx,
                start_sec=start,
                end_sec=end,
                rms=rms,
                peak=rms,
                zero_crossing_rate=0.0,
                spectral_centroid_hz=None,
                spectral_rolloff_hz=None,
                is_silent=rms < 1e-6,
                onset_score=onset,
            )
        )

    return AudioFeatureDocument(
        schema_version="0.1.0",
        source_file="stem.wav",
        layer="bass",
        sample_rate=44100,
        channels=1,
        duration_sec=duration_sec,
        frame_size=1024,
        hop_size=441,
        frames=frames,
        global_features=AudioGlobalFeatures(
            peak=max((frame.peak for frame in frames), default=0.0),
            rms=sum(frame.rms for frame in frames) / max(1, len(frames)),
            duration_sec=duration_sec,
            estimated_silence_ratio=0.0,
            frame_count=len(frames),
            onset_count=sum(1 for frame in frames if frame.onset_score > 0.0),
            mean_spectral_centroid_hz=None,
            mean_spectral_rolloff_hz=None,
        ),
    )


def _refined_note(note_id: str, pitch: int, start_sec: float, end_sec: float) -> RefinedNoteEvent:
    return RefinedNoteEvent(
        note_id=note_id,
        source="ripx",
        layer="bass",
        pitch_midi=pitch,
        pitch_name="C3",
        velocity=90,
        channel=0,
        original_start_sec=start_sec,
        original_end_sec=end_sec,
        aligned_start_sec=start_sec,
        aligned_end_sec=end_sec,
        refined_start_sec=start_sec,
        refined_end_sec=end_sec,
        refined_duration_sec=end_sec - start_sec,
        start_refinement_ms=0.0,
        end_refinement_ms=0.0,
        merged_note_ids=[],
        refinement_actions=["UNCHANGED"],
        refinement_confidence=0.9,
        reasons=["test"],
    )


def _cleanup_action(note_id: str, action: str = "KEEP") -> CleanupAction:
    return CleanupAction(
        note_id=note_id,
        original_recommended_action="KEEP",
        plan_action=action,
        confidence=0.9,
        reasons=["test"],
        source_validation={"recommended_action": "KEEP"},
    )


def _write_inputs(
    tmp_path: Path,
    refined_notes: list[RefinedNoteEvent],
    audio_doc: AudioFeatureDocument,
    cleanup_actions: list[CleanupAction],
) -> tuple[Path, Path, Path]:
    refined_doc = RefinedNoteDocument(
        schema_version="0.1.0",
        aligned_notes_file="audio_aligned_note_events.json",
        audio_features_file="audio_features.json",
        validation_file="note_validation.json",
        layer="bass",
        sample_rate=audio_doc.sample_rate,
        audio_duration_sec=audio_doc.duration_sec,
        timing_source="refined_audio_seconds",
        refinement_parameters={
            "attack_lookback_ms": 80.0,
            "max_attack_advance_ms": 80.0,
            "merge_gap_ms": 160.0,
            "minimum_silence_ms": 80.0,
            "tail_rms_ratio": 0.2,
            "tail_silence_hold_ms": 120.0,
            "max_tail_extension_ms": 900.0,
            "minimum_note_duration_ms": 80.0,
            "monophonic": True,
            "allow_pitch_overlap": False,
        },
        notes=refined_notes,
    )

    cleanup_doc = CleanupPlanDocument(
        schema_version="0.1.0",
        validation_file="note_validation.json",
        layer="bass",
        planner_parameters={
            "mute_threshold": 0.45,
            "review_threshold": 0.70,
            "delete_threshold": 0.20,
            "allow_delete_candidates": False,
        },
        actions=cleanup_actions,
    )

    refined_path = tmp_path / "refined_note_events.json"
    audio_path = tmp_path / "audio_features.json"
    cleanup_path = tmp_path / "cleanup_plan.json"

    refined_path.write_text(refined_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    audio_path.write_text(audio_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    cleanup_path.write_text(cleanup_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")

    return refined_path, audio_path, cleanup_path


def test_audio_activity_regions_merge_tiny_gap() -> None:
    frames = [
        _ActivityFrame(start_sec=i * 0.02, end_sec=(i + 1) * 0.02, energy=0.2, onset=0.0, is_active_hint=True)
        for i in range(5)
    ]
    frames.extend(
        [
            _ActivityFrame(start_sec=0.10, end_sec=0.12, energy=0.0, onset=0.0, is_active_hint=False),
            _ActivityFrame(start_sec=0.12, end_sec=0.14, energy=0.0, onset=0.0, is_active_hint=False),
        ]
    )
    frames.extend(
        [
            _ActivityFrame(start_sec=0.14 + i * 0.02, end_sec=0.16 + i * 0.02, energy=0.2, onset=0.0, is_active_hint=True)
            for i in range(5)
        ]
    )

    regions = _build_audio_activity_regions(
        frames,
        ActivityRepairParameters(audio_silence_hold_ms=0.0, merge_audio_region_gap_ms=60.0),
    )

    assert len(regions) == 1


def test_audio_activity_regions_split_on_long_silence() -> None:
    frames = [
        _ActivityFrame(start_sec=i * 0.02, end_sec=(i + 1) * 0.02, energy=0.2, onset=0.0, is_active_hint=True)
        for i in range(5)
    ]
    frames.extend(
        [
            _ActivityFrame(start_sec=0.10 + i * 0.02, end_sec=0.12 + i * 0.02, energy=0.0, onset=0.0, is_active_hint=False)
            for i in range(10)
        ]
    )
    frames.extend(
        [
            _ActivityFrame(start_sec=0.30 + i * 0.02, end_sec=0.32 + i * 0.02, energy=0.2, onset=0.0, is_active_hint=True)
            for i in range(5)
        ]
    )

    regions = _build_audio_activity_regions(
        frames,
        ActivityRepairParameters(audio_silence_hold_ms=0.0, merge_audio_region_gap_ms=60.0),
    )

    assert len(regions) == 2


def test_missing_gap_extension_extends_note(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=1.5,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.09 if 0.2 <= t <= 0.55 else 0.0,
        onset_fn=lambda t: 0.1 if abs(t - 0.2) < 1e-6 else 0.0,
    )
    notes = [_refined_note("n1", 45, 0.2, 0.4)]
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes,
        audio_doc,
        [_cleanup_action("n1", "KEEP")],
    )

    repaired, _plan, report = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
    )

    assert report.extend_count >= 1
    assert repaired.notes[0].refined_end_sec > 0.5


def test_missing_gap_insert_with_context_pitch(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.5,
        frame_step_sec=0.01,
        rms_fn=lambda t: (
            0.08 if (0.2 <= t <= 0.35 or 1.0 <= t <= 1.05 or 1.62 <= t <= 1.70) else 0.0
        ),
        onset_fn=lambda t: 0.12 if abs(t - 1.62) < 1e-6 else 0.0,
    )
    notes = [
        _refined_note("n1", 45, 0.2, 0.35),
        _refined_note("n2", 45, 1.0, 1.05),
    ]
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes,
        audio_doc,
        [_cleanup_action("n1", "KEEP"), _cleanup_action("n2", "KEEP")],
    )

    repaired, _plan, report = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
    )

    inserted = [note for note in repaired.notes if note.note_id.startswith("repair_missing_")]
    assert report.insert_missing_count >= 1
    assert len(inserted) >= 1


def test_low_confidence_missing_gap_marks_review_manual(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.5,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if (0.2 <= t <= 0.35 or 1.6 <= t <= 1.70) else 0.0,
        onset_fn=lambda t: 0.0,
    )
    notes = [_refined_note("n1", 45, 0.2, 0.35)]
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes,
        audio_doc,
        [_cleanup_action("n1", "KEEP")],
    )

    _repaired, plan, report = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(
            audio_silence_hold_ms=0.0,
            context_pitch_search_ms=50.0,
        ),
    )

    assert report.review_manual_count >= 1
    assert all(action.action_type != "INSERT_MISSING_NOTE" for action in plan.actions)


def test_overhang_shortening_shortens_note(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=1.5,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 0.45 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    notes = [_refined_note("n1", 45, 0.2, 0.8)]
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes,
        audio_doc,
        [_cleanup_action("n1", "KEEP")],
    )

    repaired, _plan, report = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
    )

    assert report.shorten_count >= 1
    assert repaired.notes[0].refined_end_sec <= 0.5


def test_split_note_on_strong_internal_onset(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 1.0 else 0.0,
        onset_fn=lambda t: (
            0.0 if 0.48 <= t <= 0.58 else (1.0 if abs(t - 0.6) < 1e-6 else 0.02)
        ),
    )
    notes = [_refined_note("n1", 45, 0.2, 1.0)]
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes,
        audio_doc,
        [_cleanup_action("n1", "KEEP")],
    )

    repaired, _plan, report = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
    )

    assert report.split_count >= 1
    assert len(repaired.notes) >= 2
    assert any("ACTIVITY_REPAIR_SPLIT" in note.refinement_actions for note in repaired.notes)


def test_weak_internal_onset_does_not_split(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 1.0 else 0.0,
        onset_fn=lambda t: 0.2 if 0.2 <= t <= 1.0 else 0.0,
    )
    notes = [_refined_note("n1", 45, 0.2, 1.0)]
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes,
        audio_doc,
        [_cleanup_action("n1", "KEEP")],
    )

    repaired, _plan, report = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
    )

    assert report.split_count == 0
    assert len(repaired.notes) == 1


def test_close_tiny_gap_for_same_pitch_notes(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=1.5,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 0.6 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    notes = [
        _refined_note("n1", 45, 0.2, 0.4),
        _refined_note("n2", 45, 0.43, 0.6),
    ]
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes,
        audio_doc,
        [_cleanup_action("n1", "KEEP"), _cleanup_action("n2", "KEEP")],
    )

    repaired, _plan, report = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(audio_silence_hold_ms=0.0, close_gap_ms=50.0),
    )

    assert report.close_gap_count >= 1
    repaired_by_id = {note.note_id: note for note in repaired.notes}
    assert repaired_by_id["n1"].refined_end_sec >= 0.43
