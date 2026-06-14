from __future__ import annotations

from pathlib import Path

from midi_cleaner.audio.models import AudioFeatureDocument, AudioFrameFeature, AudioGlobalFeatures
from midi_cleaner.cleanup.models import CleanupAction, CleanupPlanDocument
from midi_cleaner.repair.activity import ActivityRepairParameters
from midi_cleaner.repair.iterative import (
    IterativeRepairParameters,
    run_iterative_activity_repair,
)
from midi_cleaner.repair.models import ActivityRepairPlan, ActivityRepairReport
from midi_cleaner.refinement.models import RefinedNoteDocument, RefinedNoteEvent


def _build_audio_document(
    duration_sec: float,
    frame_step_sec: float,
    rms_fn,
    onset_fn,
) -> AudioFeatureDocument:
    frames: list[AudioFrameFeature] = []
    frame_count = int(duration_sec / frame_step_sec)
    for index in range(frame_count):
        start_sec = index * frame_step_sec
        end_sec = start_sec + frame_step_sec
        rms = float(rms_fn(start_sec))
        onset = float(onset_fn(start_sec))
        frames.append(
            AudioFrameFeature(
                frame_index=index,
                start_sec=start_sec,
                end_sec=end_sec,
                rms=rms,
                peak=rms,
                zero_crossing_rate=0.0,
                spectral_centroid_hz=None,
                spectral_rolloff_hz=None,
                is_silent=rms < 1e-9,
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


def _refined_note(note_id: str, pitch_midi: int, start_sec: float, end_sec: float) -> RefinedNoteEvent:
    return RefinedNoteEvent(
        note_id=note_id,
        source="ripx",
        layer="bass",
        pitch_midi=pitch_midi,
        pitch_name="C3",
        velocity=90,
        channel=0,
        original_start_sec=start_sec,
        original_end_sec=end_sec,
        aligned_start_sec=start_sec,
        aligned_end_sec=end_sec,
        refined_start_sec=start_sec,
        refined_end_sec=end_sec,
        refined_duration_sec=max(0.0, end_sec - start_sec),
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
    notes: list[RefinedNoteEvent],
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
        notes=notes,
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


def test_iterative_loop_improves_coverage(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=1.5,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 0.58 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes=[_refined_note("n1", 45, 0.2, 0.36)],
        audio_doc=audio_doc,
        cleanup_actions=[_cleanup_action("n1")],
    )

    final_doc, report, artifacts = run_iterative_activity_repair(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=IterativeRepairParameters(max_iterations=3, min_improvement=0.0),
        activity_params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
    )

    assert report.final_score >= report.initial_score
    assert report.iterations_completed >= 1
    assert any(
        item.extend_count > 0 or item.close_gap_count > 0 for item in report.iterations
    )
    by_id = {note.note_id: note for note in final_doc.notes}
    assert by_id["n1"].refined_end_sec > 0.5


def test_second_pass_repairs_remaining_overhang(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if (0.2 <= t <= 0.50 or 0.7 <= t <= 0.82) else 0.0,
        onset_fn=lambda t: 0.0,
    )
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes=[
            _refined_note("n1", 45, 0.2, 0.30),
            _refined_note("n2", 45, 0.7, 1.2),
        ],
        audio_doc=audio_doc,
        cleanup_actions=[_cleanup_action("n1"), _cleanup_action("n2")],
    )

    final_doc, report, artifacts = run_iterative_activity_repair(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=IterativeRepairParameters(
            max_iterations=3,
            min_improvement=0.0,
            max_actions_per_iteration=1,
        ),
        activity_params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
    )

    assert report.iterations_completed >= 2
    assert report.iterations[1].total_score >= report.iterations[0].total_score
    by_id = {note.note_id: note for note in final_doc.notes}
    assert by_id["n2"].refined_end_sec < 1.05


def test_third_pass_conservative_avoids_aggressive_insert_split(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 1.2 <= t <= 1.28 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes=[_refined_note("n1", 45, 0.2, 0.3)],
        audio_doc=audio_doc,
        cleanup_actions=[_cleanup_action("n1")],
    )

    _final_doc, report, _artifacts = run_iterative_activity_repair(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=IterativeRepairParameters(max_iterations=3, min_improvement=-1.0),
        activity_params=ActivityRepairParameters(
            audio_silence_hold_ms=0.0,
            context_pitch_search_ms=40.0,
        ),
    )

    assert report.iterations_completed == 3
    assert report.iterations[-1].insert_count == 0
    assert report.iterations[-1].split_count == 0


def test_regression_guard_keeps_previous_best(monkeypatch, tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=1.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 0.5 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes=[_refined_note("n1", 45, 0.2, 0.5)],
        audio_doc=audio_doc,
        cleanup_actions=[_cleanup_action("n1")],
    )

    def _force_regression(**kwargs):
        input_path = Path(kwargs["refined_notes_file"])
        doc = RefinedNoteDocument.model_validate_json(input_path.read_text(encoding="utf-8"))
        shifted = [
            note.model_copy(
                update={
                    "refined_start_sec": float(note.refined_start_sec) + 5.0,
                    "refined_end_sec": float(note.refined_end_sec) + 5.0,
                }
            )
            for note in doc.notes
        ]
        out_doc = doc.model_copy(update={"notes": shifted})
        plan = ActivityRepairPlan(
            schema_version="0.1.0",
            refined_notes_file=str(input_path),
            audio_features_file=str(kwargs["audio_features_file"]),
            dsp_features_file=None,
            pitch_contour_file=None,
            cleanup_plan_file=str(kwargs["cleanup_plan_file"]),
            layer=doc.layer,
            actions=[],
        )
        report = ActivityRepairReport(
            refined_notes_file=str(input_path),
            audio_features_file=str(kwargs["audio_features_file"]),
            dsp_features_file=None,
            pitch_contour_file=None,
            cleanup_plan_file=str(kwargs["cleanup_plan_file"]),
            status="ok",
            layer=doc.layer,
            input_note_count=len(doc.notes),
            output_note_count=len(out_doc.notes),
            extend_count=0,
            shorten_count=0,
            insert_missing_count=0,
            split_count=0,
            close_gap_count=0,
            review_manual_count=0,
            keep_count=0,
            sustain_protected_count=0,
            pitch_protected_count=0,
            legato_protected_count=0,
            shorten_candidate_count=0,
            shorten_applied_count=0,
            shorten_rejected_count=0,
            audio_active_region_count=0,
            midi_active_region_count=0,
            audio_gap_count=0,
            midi_overhang_count=0,
            warning_count=0,
            warnings=[],
            output_file=None,
            plan_file=None,
        )
        return out_doc, plan, report

    monkeypatch.setattr("midi_cleaner.repair.iterative.repair_activity", _force_regression)

    final_doc, report, artifacts = run_iterative_activity_repair(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=IterativeRepairParameters(max_iterations=3, allow_regression=False),
        activity_params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
    )

    assert report.iterations_completed == 1
    assert report.iterations[-1].stopped_reason == "regression_rejected"
    assert report.total_improvement <= 0.000001
    assert abs(final_doc.notes[0].refined_start_sec - 0.2) < 0.000001


def test_stable_region_freeze_marks_stable_notes(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if (0.2 <= t <= 0.42 or 1.2 <= t <= 1.28) else 0.0,
        onset_fn=lambda t: 0.0,
    )
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes=[_refined_note("stable", 45, 0.2, 0.42)],
        audio_doc=audio_doc,
        cleanup_actions=[_cleanup_action("stable")],
    )

    final_doc, report, artifacts = run_iterative_activity_repair(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=IterativeRepairParameters(
            max_iterations=3,
            min_improvement=-1.0,
            freeze_stable_notes=True,
        ),
        activity_params=ActivityRepairParameters(
            audio_silence_hold_ms=0.0,
            context_pitch_search_ms=40.0,
        ),
    )

    assert report.iterations_completed == 3
    stable_note = {note.note_id: note for note in final_doc.notes}["stable"]
    assert abs(stable_note.refined_end_sec - 0.42) < 0.001
    last_iter_note = {
        note.note_id: note for note in artifacts[-1].repaired_document.notes
    }["stable"]
    assert "ITERATIVE_REPAIR_STABLE" in last_iter_note.refinement_actions


def test_sustain_legato_profile_avoids_aggressive_shorten(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=1.6,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.2 <= t <= 0.9 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes=[_refined_note("n1", 45, 0.2, 1.0)],
        audio_doc=audio_doc,
        cleanup_actions=[_cleanup_action("n1")],
    )

    _final_doc, report, _artifacts = run_iterative_activity_repair(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=IterativeRepairParameters(max_iterations=1, pass1_profile="sustain_legato"),
        activity_params=ActivityRepairParameters(audio_silence_hold_ms=0.0),
    )

    assert report.iterations[0].shorten_count == 0


def test_conservative_final_pass_keeps_uncertain_insert_manual(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 1.1 <= t <= 1.2 else 0.0,
        onset_fn=lambda t: 0.0,
    )
    refined_path, audio_path, cleanup_path = _write_inputs(
        tmp_path,
        notes=[_refined_note("n1", 45, 0.2, 0.3)],
        audio_doc=audio_doc,
        cleanup_actions=[_cleanup_action("n1")],
    )

    _final_doc, report, _artifacts = run_iterative_activity_repair(
        refined_notes_file=refined_path,
        audio_features_file=audio_path,
        cleanup_plan_file=cleanup_path,
        params=IterativeRepairParameters(
            max_iterations=3,
            min_improvement=-1.0,
            conservative_final_pass=True,
        ),
        activity_params=ActivityRepairParameters(
            audio_silence_hold_ms=0.0,
            context_pitch_search_ms=40.0,
        ),
    )

    assert report.iterations[-1].insert_count == 0
    assert report.iterations[-1].review_manual_count >= 1
