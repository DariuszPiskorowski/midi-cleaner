from __future__ import annotations

from pathlib import Path

import numpy as np

from midi_cleaner.audio.models import AudioFeatureDocument, AudioFrameFeature, AudioGlobalFeatures
from midi_cleaner.cleanup.models import CleanupAction, CleanupPlanDocument
from midi_cleaner.dsp.models import DspAudioFeatureDocument, DspAudioFrame
from midi_cleaner.pitch.models import BassPitchContourDocument, BassPitchContourReport, BassPitchFrame
from midi_cleaner.repair.activity import (
    ActivityRepairParameters,
    _MutableNote,
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


def _write_dsp_features(
    tmp_path: Path,
    duration_sec: float,
    frame_step_sec: float,
    low_band_fn,
    harmonic_fn,
    onset_fn,
) -> Path:
    frames: list[DspAudioFrame] = []
    frame_count = int(duration_sec / frame_step_sec)
    for idx in range(frame_count):
        start = idx * frame_step_sec
        end = start + frame_step_sec
        low_band = float(low_band_fn(start))
        harmonic = float(harmonic_fn(start))
        onset = float(onset_fn(start))
        rms = max(1e-6, low_band * 0.8)
        frames.append(
            DspAudioFrame(
                frame_index=idx,
                start_sec=start,
                end_sec=end,
                rms=rms,
                rms_smooth=rms,
                rms_delta=0.0,
                envelope=rms,
                envelope_smooth=rms,
                envelope_delta=0.0,
                low_band_rms=low_band,
                low_band_envelope=low_band,
                low_band_envelope_smooth=low_band,
                low_band_delta=0.0,
                spectral_flux=onset,
                onset_strength=onset,
                harmonic_rms=harmonic,
                percussive_rms=0.0,
                is_attack_rise=onset > 0.0,
                is_sustain=(low_band > 0.0),
                is_tail=False,
                is_silence=(low_band <= 0.0),
            )
        )

    doc = DspAudioFeatureDocument(
        schema_version="0.1.0",
        wav_file="stem.wav",
        layer="bass",
        sample_rate=44100,
        duration_sec=duration_sec,
        backend_name="basic",
        backend_available=True,
        hop_length=512,
        frame_length=2048,
        low_band_hz=[40.0, 500.0],
        frames=frames,
    )
    path = tmp_path / "audio_features_dsp.json"
    path.write_text(doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _write_pitch_contour(
    tmp_path: Path,
    duration_sec: float,
    frame_step_sec: float,
    pitch_fn,
    confidence_fn,
) -> Path:
    frames: list[BassPitchFrame] = []
    frame_count = int(duration_sec / frame_step_sec)
    for idx in range(frame_count):
        start = idx * frame_step_sec
        end = start + frame_step_sec
        pitch = pitch_fn(start)
        confidence = float(confidence_fn(start))
        if pitch is None:
            frames.append(
                BassPitchFrame(
                    frame_index=idx,
                    start_sec=start,
                    end_sec=end,
                    f0_hz=None,
                    pitch_midi_float=None,
                    pitch_midi_rounded=None,
                    pitch_confidence=confidence,
                    voiced=False,
                    low_band_energy=0.0,
                    harmonic_energy=0.0,
                )
            )
            continue

        midi_float = 69.0 + 12.0 * np.log2(float(pitch) / 440.0)
        frames.append(
            BassPitchFrame(
                frame_index=idx,
                start_sec=start,
                end_sec=end,
                f0_hz=float(pitch),
                pitch_midi_float=float(midi_float),
                pitch_midi_rounded=int(round(midi_float)),
                pitch_confidence=confidence,
                voiced=confidence >= 0.6,
                low_band_energy=0.01,
                harmonic_energy=0.01,
            )
        )

    doc = BassPitchContourDocument(
        schema_version="0.1.0",
        wav_file="stem.wav",
        layer="bass",
        backend_name="basic",
        backend_available=True,
        sample_rate=44100,
        duration_sec=duration_sec,
        hop_length=512,
        frame_length=2048,
        min_hz=35.0,
        max_hz=400.0,
        frames=frames,
    )
    _ = BassPitchContourReport(
        wav_file="stem.wav",
        status="ok",
        layer="bass",
        backend_name="basic",
        backend_available=True,
        frame_count=frame_count,
        voiced_frame_count=sum(1 for f in frames if f.voiced),
        voiced_ratio=sum(1 for f in frames if f.voiced) / max(1, frame_count),
        mean_pitch_confidence=sum(f.pitch_confidence for f in frames) / max(1, frame_count),
        min_detected_hz=min((f.f0_hz for f in frames if f.f0_hz is not None), default=None),
        max_detected_hz=max((f.f0_hz for f in frames if f.f0_hz is not None), default=None),
        warning_count=0,
        warnings=[],
        output_file=None,
    )
    path = tmp_path / "bass_pitch_contour.json"
    path.write_text(doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


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
    assert repaired.notes[0].refined_end_sec <= 0.56


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


def test_extend_split_same_note_conflict_prefers_extend_for_bass(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 1.2 else 0.0,
        onset_fn=lambda t: (0.0 if 0.52 <= t <= 0.60 else (1.0 if abs(t - 0.62) < 1e-6 else 0.02)),
    )
    notes = [_refined_note("n1", 45, 0.2, 1.0)]
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes,
        audio_doc,
        [_cleanup_action("n1", "KEEP")],
    )

    repaired, plan, report = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
    )

    by_target: dict[str, set[str]] = {}
    for action in plan.actions:
        if action.target_note_id is None:
            continue
        if action.action_type in {"KEEP", "REVIEW_MANUAL"}:
            continue
        by_target.setdefault(action.target_note_id, set()).add(action.action_type)

    assert "EXTEND_NOTE" in by_target.get("n1", set())
    assert "SPLIT_NOTE" not in by_target.get("n1", set())
    assert report.extend_count >= 1
    assert report.split_count == 0
    assert any("conflict-suppressed action" in warning for warning in report.warnings)
    assert repaired.notes[0].refined_end_sec > 1.0


def test_insert_split_same_pass_conflict_suppresses_split_on_inserted_note(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.6,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if (0.2 <= t <= 0.35 or 1.62 <= t <= 2.28) else 0.0,
        onset_fn=lambda t: (0.0 if 1.96 <= t <= 2.04 else (1.0 if abs(t - 2.06) < 1e-6 else 0.02)),
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

    def _fake_closest_note(notes: list[_MutableNote], anchor_sec: float, search_sec: float) -> _MutableNote | None:
        _ = anchor_sec
        _ = search_sec
        if not notes:
            return None
        return notes[0]

    def _fake_prev_note(notes: list[_MutableNote], start_sec: float) -> _MutableNote | None:
        _ = start_sec
        return None

    monkeypatch.setattr("midi_cleaner.repair.activity._closest_note", _fake_closest_note)
    monkeypatch.setattr("midi_cleaner.repair.activity._find_prev_note", _fake_prev_note)

    _repaired, plan, report = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(
            audio_silence_hold_ms=0.0,
            context_pitch_search_ms=800.0,
            max_extend_for_gap_ms=150.0,
        ),
    )

    inserted_ids = {
        action.new_note_id
        for action in plan.actions
        if action.action_type == "INSERT_MISSING_NOTE" and action.new_note_id is not None
    }
    split_targets = {
        action.target_note_id
        for action in plan.actions
        if action.action_type == "SPLIT_NOTE" and action.target_note_id is not None
    }

    assert inserted_ids
    assert inserted_ids.isdisjoint(split_targets)
    assert any("inserted_note_same_pass" in warning for warning in report.warnings)


def test_split_count_excludes_conflict_suppressed_splits(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 1.2 else 0.0,
        onset_fn=lambda t: (0.0 if 0.52 <= t <= 0.60 else (1.0 if abs(t - 0.62) < 1e-6 else 0.02)),
    )
    notes = [_refined_note("n1", 45, 0.2, 1.0)]
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes,
        audio_doc,
        [_cleanup_action("n1", "KEEP")],
    )

    repaired, plan, report = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
    )

    applied_splits = [action for action in plan.actions if action.action_type == "SPLIT_NOTE"]
    assert len(applied_splits) == report.split_count
    assert report.split_count == 0
    assert any("suppressed_conflict_action_count=" in warning for warning in report.warnings)
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


def test_do_not_shorten_sustained_low_band_note(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=1.5,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 0.42 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    dsp_path = _write_dsp_features(
        tmp_path,
        duration_sec=1.5,
        frame_step_sec=0.01,
        low_band_fn=lambda t: 0.06 if 0.2 <= t <= 0.45 else 0.0,
        harmonic_fn=lambda t: (0.05 if 0.2 <= t <= 0.45 else (0.03 if 0.45 < t <= 0.72 else 0.0)),
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
        dsp_features_file=dsp_path,
    )

    assert report.shorten_count == 0
    assert report.sustain_protected_count >= 1
    assert repaired.notes[0].refined_end_sec >= 0.75


def test_do_not_shorten_voiced_pitch_contour(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=1.5,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 0.42 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    pitch_path = _write_pitch_contour(
        tmp_path,
        duration_sec=1.5,
        frame_step_sec=0.01,
        pitch_fn=lambda t: 55.0 if 0.2 <= t <= 0.76 else None,
        confidence_fn=lambda t: 0.9 if 0.2 <= t <= 0.76 else 0.0,
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
        pitch_contour_file=pitch_path,
    )

    assert report.shorten_count == 0
    assert report.pitch_protected_count >= 1
    assert repaired.notes[0].refined_end_sec >= 0.75


def test_shorten_true_overhang_when_unvoiced_and_no_sustain(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=1.5,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 0.42 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    dsp_path = _write_dsp_features(
        tmp_path,
        duration_sec=1.5,
        frame_step_sec=0.01,
        low_band_fn=lambda t: 0.06 if 0.2 <= t <= 0.42 else 0.0,
        harmonic_fn=lambda t: 0.04 if 0.2 <= t <= 0.42 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    pitch_path = _write_pitch_contour(
        tmp_path,
        duration_sec=1.5,
        frame_step_sec=0.01,
        pitch_fn=lambda t: 55.0 if 0.2 <= t <= 0.42 else None,
        confidence_fn=lambda t: 0.9 if 0.2 <= t <= 0.42 else 0.0,
    )
    notes = [_refined_note("n1", 45, 0.2, 0.82)]
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
        dsp_features_file=dsp_path,
        pitch_contour_file=pitch_path,
    )

    assert report.shorten_count >= 1
    assert report.shorten_applied_count >= 1
    assert repaired.notes[0].refined_end_sec < 0.7


def test_legato_protection_blocks_aggressive_shorten(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=1.8,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if (0.2 <= t <= 0.86 or 0.88 <= t <= 1.2) else 0.0,
        onset_fn=lambda t: 0.0,
    )
    notes = [
        _refined_note("n1", 45, 0.2, 1.00),
        _refined_note("n2", 45, 1.05, 1.2),
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
        params=ActivityRepairParameters(
            audio_silence_hold_ms=0.0,
            merge_audio_region_gap_ms=5.0,
            overhang_min_ms=100.0,
        ),
    )

    assert report.legato_protected_count >= 1
    assert report.shorten_count == 0
    repaired_by_id = {note.note_id: note for note in repaired.notes}
    assert repaired_by_id["n1"].refined_end_sec >= 0.95


def test_pitch_aware_split_requires_pitch_change_evidence(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 1.0 else 0.0,
        onset_fn=lambda t: (1.0 if abs(t - 0.6) < 1e-6 else 0.02),
    )
    notes = [_refined_note("n1", 45, 0.2, 1.0)]
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes,
        audio_doc,
        [_cleanup_action("n1", "KEEP")],
    )

    no_change_pitch = _write_pitch_contour(
        tmp_path,
        duration_sec=2.0,
        frame_step_sec=0.01,
        pitch_fn=lambda t: 55.0 if 0.2 <= t <= 1.0 else None,
        confidence_fn=lambda t: 0.9 if 0.2 <= t <= 1.0 else 0.0,
    )
    repaired_a, _plan_a, report_a = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
        pitch_contour_file=no_change_pitch,
    )
    assert report_a.split_count == 0
    assert len(repaired_a.notes) == 1

    change_pitch = _write_pitch_contour(
        tmp_path,
        duration_sec=2.0,
        frame_step_sec=0.01,
        pitch_fn=lambda t: (55.0 if t < 0.6 else (62.0 if 0.6 <= t <= 1.0 else None)),
        confidence_fn=lambda t: 0.9 if 0.2 <= t <= 1.0 else 0.0,
    )
    repaired_b, _plan_b, report_b = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
        pitch_contour_file=change_pitch,
    )
    assert report_b.split_count >= 1
    assert len(repaired_b.notes) >= 2


def test_insert_missing_note_uses_pitch_contour_when_no_context(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.9 <= t <= 1.0 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    pitch_path = _write_pitch_contour(
        tmp_path,
        duration_sec=2.0,
        frame_step_sec=0.01,
        pitch_fn=lambda t: 65.4 if 0.9 <= t <= 1.0 else None,
        confidence_fn=lambda t: 0.95 if 0.9 <= t <= 1.0 else 0.0,
    )

    notes: list[RefinedNoteEvent] = []
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes,
        audio_doc,
        [],
    )

    # Keep one distant note in plan context so repair stage can run.
    distant_note = _refined_note("n1", 45, 0.2, 0.3)
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        [distant_note],
        audio_doc,
        [_cleanup_action("n1", "KEEP")],
    )

    repaired, _plan, report = repair_activity(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
        pitch_contour_file=pitch_path,
    )

    inserted = [note for note in repaired.notes if note.note_id.startswith("repair_missing_")]
    assert report.insert_missing_count >= 1
    assert inserted
    assert inserted[0].pitch_midi == 36
