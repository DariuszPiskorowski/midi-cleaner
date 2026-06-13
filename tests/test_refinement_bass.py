from __future__ import annotations

from pathlib import Path

from midi_cleaner.alignment.models import AudioAlignedNoteDocument, AudioAlignedNoteEvent
from midi_cleaner.audio.models import AudioFeatureDocument, AudioFrameFeature, AudioGlobalFeatures
from midi_cleaner.refinement.bass import (
    BassRefinementParameters,
    refine_bass_notes,
)
from midi_cleaner.validation.models import NoteValidation, NoteValidationDocument


def _build_audio_document(
    layer: str,
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
        layer=layer,
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


def _aligned_note(
    note_id: str,
    pitch: int,
    start_sec: float,
    end_sec: float,
    layer: str = "bass",
) -> AudioAlignedNoteEvent:
    return AudioAlignedNoteEvent(
        note_id=note_id,
        source="ripx",
        layer=layer,
        pitch_midi=pitch,
        pitch_name="C4",
        velocity=90,
        channel=0,
        original_start_sec=start_sec,
        original_end_sec=end_sec,
        original_duration_sec=end_sec - start_sec,
        original_start_tick=0,
        original_end_tick=0,
        aligned_start_sec=start_sec,
        aligned_end_sec=end_sec,
        aligned_duration_sec=end_sec - start_sec,
        start_correction_ms=0.0,
        end_correction_ms=0.0,
        duration_correction_ms=0.0,
        nearest_audio_onset_sec=start_sec,
        nearest_audio_offset_sec=end_sec,
        onset_error_before_ms=0.0,
        onset_error_after_ms=0.0,
        local_rms=0.02,
        local_onset_score=0.05,
        sustained_energy_ratio=0.8,
        alignment_confidence=0.9,
        alignment_action="ALIGNED",
        reasons=["test"],
    )


def _validation(note_id: str, layer: str = "bass") -> NoteValidation:
    return NoteValidation(
        note_id=note_id,
        pitch_midi=60,
        pitch_name="C4",
        layer=layer,
        source="ripx",
        start_sec=0.0,
        end_sec=0.2,
        duration_sec=0.2,
        nearest_onset_sec=0.0,
        onset_error_ms=0.0,
        onset_score=0.05,
        max_rms_during_note=0.1,
        mean_rms_during_note=0.05,
        sustained_energy_ratio=0.8,
        energy_match_score=0.8,
        duration_match_score=0.8,
        confidence=0.8,
        recommended_action="KEEP",
        reasons=["test"],
    )


def _write_inputs(
    tmp_path: Path,
    aligned_notes: list[AudioAlignedNoteEvent],
    audio_doc: AudioFeatureDocument,
) -> tuple[Path, Path, Path]:
    aligned_doc = AudioAlignedNoteDocument(
        schema_version="0.1.0",
        notes_file="note_events.json",
        audio_features_file="audio_features.json",
        layer="bass",
        sample_rate=audio_doc.sample_rate,
        audio_duration_sec=audio_doc.duration_sec,
        alignment_parameters={
            "onset_search_window_ms": 250.0,
            "offset_search_window_ms": 350.0,
            "min_onset_score": 0.005,
            "min_rms": 0.001,
            "snap_start_to_audio_onset": True,
            "snap_end_to_energy_offset": True,
            "max_start_correction_ms": 500.0,
            "max_end_correction_ms": 800.0,
            "low_confidence_action": "KEEP_ORIGINAL_LOW_CONFIDENCE",
        },
        notes=aligned_notes,
    )

    validation_doc = NoteValidationDocument(
        schema_version="0.1.0",
        notes_file="note_events.json",
        audio_features_file="audio_features.json",
        layer="bass",
        validation_parameters={
            "onset_window_ms": 50.0,
            "minimum_rms": 0.001,
            "minimum_onset_score": 0.01,
            "review_threshold": 0.45,
            "keep_threshold": 0.70,
        },
        validations=[_validation(note.note_id) for note in aligned_notes],
    )

    aligned_path = tmp_path / "audio_aligned_note_events.json"
    audio_path = tmp_path / "audio_features.json"
    validation_path = tmp_path / "note_validation.json"

    aligned_path.write_text(aligned_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    audio_path.write_text(audio_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    validation_path.write_text(validation_doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return aligned_path, audio_path, validation_path


def test_attack_adjustment_moves_start_earlier_toward_attack_rise(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        layer="bass",
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.01 + ((t - 0.95) * 2.0) if 0.95 <= t <= 1.00 else (0.2 if 1.00 < t < 1.3 else 0.0),
        onset_fn=lambda t: 0.2 if abs(t - 1.0) < 1e-6 else 0.0,
    )
    aligned_notes = [_aligned_note("n1", pitch=45, start_sec=1.0, end_sec=1.2)]
    aligned_path, audio_path, validation_path = _write_inputs(tmp_path, aligned_notes, audio_doc)

    refined_doc, _ = refine_bass_notes(
        aligned_notes_file=aligned_path,
        audio_features_file=audio_path,
        validation_file=validation_path,
        params=BassRefinementParameters(),
    )

    note = refined_doc.notes[0]
    assert note.refined_start_sec <= 0.96
    assert note.refined_start_sec >= 0.92
    assert "ATTACK_START_ADJUSTED" in note.refinement_actions


def test_false_retrigger_merge_combines_same_pitch_without_real_silence(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        layer="bass",
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 0.95 <= t <= 1.45 else 0.0,
        onset_fn=lambda t: 0.05 if abs(t - 1.0) < 1e-6 or abs(t - 1.24) < 1e-6 else 0.0,
    )
    aligned_notes = [
        _aligned_note("a", pitch=45, start_sec=1.0, end_sec=1.2),
        _aligned_note("b", pitch=45, start_sec=1.24, end_sec=1.4),
    ]
    aligned_path, audio_path, validation_path = _write_inputs(tmp_path, aligned_notes, audio_doc)

    refined_doc, report = refine_bass_notes(
        aligned_notes_file=aligned_path,
        audio_features_file=audio_path,
        validation_file=validation_path,
        params=BassRefinementParameters(),
    )

    assert report.false_retrigger_merge_count >= 1
    assert len(refined_doc.notes) == 1
    assert "FALSE_RETRIGGER_MERGED" in refined_doc.notes[0].refinement_actions


def test_real_silence_prevents_false_retrigger_merge(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        layer="bass",
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: (
            0.08
            if (0.95 <= t <= 1.2 or 1.35 <= t <= 1.6)
            else 0.0
        ),
        onset_fn=lambda t: 0.05 if abs(t - 1.0) < 1e-6 or abs(t - 1.35) < 1e-6 else 0.0,
    )
    aligned_notes = [
        _aligned_note("a", pitch=45, start_sec=1.0, end_sec=1.2),
        _aligned_note("b", pitch=45, start_sec=1.35, end_sec=1.5),
    ]
    aligned_path, audio_path, validation_path = _write_inputs(tmp_path, aligned_notes, audio_doc)

    refined_doc, report = refine_bass_notes(
        aligned_notes_file=aligned_path,
        audio_features_file=audio_path,
        validation_file=validation_path,
        params=BassRefinementParameters(),
    )

    assert report.false_retrigger_merge_count == 0
    assert len(refined_doc.notes) == 2


def test_tail_extension_extends_note_through_sustain(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        layer="bass",
        duration_sec=2.5,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.09 if 1.0 <= t <= 1.6 else 0.0,
        onset_fn=lambda t: 0.05 if abs(t - 1.0) < 1e-6 else 0.0,
    )
    aligned_notes = [_aligned_note("n1", pitch=45, start_sec=1.0, end_sec=1.2)]
    aligned_path, audio_path, validation_path = _write_inputs(tmp_path, aligned_notes, audio_doc)

    refined_doc, report = refine_bass_notes(
        aligned_notes_file=aligned_path,
        audio_features_file=audio_path,
        validation_file=validation_path,
        params=BassRefinementParameters(),
    )

    note = refined_doc.notes[0]
    assert note.refined_end_sec >= 1.5
    assert report.tail_extended_count >= 1


def test_tail_extension_respects_max_cap(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        layer="bass",
        duration_sec=4.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.08 if 1.0 <= t <= 3.5 else 0.0,
        onset_fn=lambda t: 0.05 if abs(t - 1.0) < 1e-6 else 0.0,
    )
    aligned_notes = [_aligned_note("n1", pitch=45, start_sec=1.0, end_sec=1.2)]
    aligned_path, audio_path, validation_path = _write_inputs(tmp_path, aligned_notes, audio_doc)

    refined_doc, _ = refine_bass_notes(
        aligned_notes_file=aligned_path,
        audio_features_file=audio_path,
        validation_file=validation_path,
        params=BassRefinementParameters(max_tail_extension_ms=300.0),
    )

    note = refined_doc.notes[0]
    assert note.refined_end_sec <= 1.5 + 1e-9


def test_minimum_duration_extends_short_energetic_note(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        layer="bass",
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.09 if 1.0 <= t <= 1.12 else 0.0,
        onset_fn=lambda t: 0.05 if abs(t - 1.0) < 1e-6 else 0.0,
    )
    aligned_notes = [_aligned_note("n1", pitch=45, start_sec=1.0, end_sec=1.03)]
    aligned_path, audio_path, validation_path = _write_inputs(tmp_path, aligned_notes, audio_doc)

    refined_doc, report = refine_bass_notes(
        aligned_notes_file=aligned_path,
        audio_features_file=audio_path,
        validation_file=validation_path,
        params=BassRefinementParameters(minimum_note_duration_ms=80.0),
    )

    note = refined_doc.notes[0]
    assert note.refined_duration_sec >= 0.08
    assert report.short_note_extended_count >= 1


def test_monophonic_overlap_cleanup_resolves_overlapping_notes(tmp_path: Path) -> None:
    audio_doc = _build_audio_document(
        layer="bass",
        duration_sec=2.0,
        frame_step_sec=0.01,
        rms_fn=lambda t: 0.09 if 0.95 <= t <= 1.6 else 0.0,
        onset_fn=lambda t: 0.05 if abs(t - 1.0) < 1e-6 or abs(t - 1.2) < 1e-6 else 0.0,
    )
    aligned_notes = [
        _aligned_note("a", pitch=45, start_sec=1.0, end_sec=1.3),
        _aligned_note("b", pitch=47, start_sec=1.2, end_sec=1.5),
    ]
    aligned_path, audio_path, validation_path = _write_inputs(tmp_path, aligned_notes, audio_doc)

    refined_doc, report = refine_bass_notes(
        aligned_notes_file=aligned_path,
        audio_features_file=audio_path,
        validation_file=validation_path,
        params=BassRefinementParameters(monophonic=True),
    )

    assert len(refined_doc.notes) == 2
    assert refined_doc.notes[0].refined_end_sec <= refined_doc.notes[1].refined_start_sec
    assert report.overlap_resolved_count >= 1
