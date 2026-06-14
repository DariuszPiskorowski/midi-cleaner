from __future__ import annotations

import json
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from midi_cleaner.ai_completion.models import AIPatternCompletionReport
from midi_cleaner.ai_completion.pattern_pack import build_pattern_pack
from midi_cleaner.ai_completion.prompt import build_ai_completion_prompts
from midi_cleaner.ai_completion.service import (
    AIPatternCompletionParameters,
    complete_ai_pattern_completion,
)
from midi_cleaner.audio.models import (
    AudioFeatureDocument,
    AudioFrameFeature,
    AudioGlobalFeatures,
)
from midi_cleaner.pipeline.process_stem import (
    PipelineProcessParameters,
    process_stem_pipeline,
)
from midi_cleaner.refinement.models import RefinedNoteDocument, RefinedNoteEvent


class _FakeAIClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.called = False

    def complete_pattern(
        self,
        *,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_completion_notes: int,
        max_output_tokens: int = 4000,
    ) -> tuple[str, dict[str, object]]:
        _ = (
            api_key,
            model,
            system_prompt,
            user_prompt,
            temperature,
            max_completion_notes,
            max_output_tokens,
        )
        self.called = True
        return json.dumps(self.payload), self.payload


def _write_candidate_midi(path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("note_on", note=45, velocity=100, time=0, channel=0))
    track.append(mido.Message("note_off", note=45, velocity=0, time=480, channel=0))
    midi.save(path)


def _write_stem_wav(path: Path) -> None:
    sample_rate = 44100
    timeline = np.linspace(0.0, 0.5, int(sample_rate * 0.5), endpoint=False)
    waveform = 0.2 * np.sin(2.0 * np.pi * 110.0 * timeline)
    sf.write(str(path), waveform.astype(np.float32), sample_rate)


def _write_working_midi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("note_on", note=40, velocity=100, time=0, channel=0))
    track.append(mido.Message("note_off", note=40, velocity=0, time=480, channel=0))
    midi.save(path)


def _write_audio_features(path: Path, duration_sec: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 20
    frame_duration = duration_sec / frame_count

    frames: list[AudioFrameFeature] = []
    for index in range(frame_count):
        start_sec = index * frame_duration
        end_sec = min(duration_sec, (index + 1) * frame_duration)
        rms = 0.08 if 0.2 <= start_sec <= 1.6 else 0.005
        onset = 0.05 if index in {2, 6, 10, 14} else 0.0
        frames.append(
            AudioFrameFeature(
                frame_index=index,
                start_sec=start_sec,
                end_sec=end_sec,
                rms=rms,
                peak=rms + 0.03,
                zero_crossing_rate=0.12,
                spectral_centroid_hz=150.0,
                spectral_rolloff_hz=350.0,
                is_silent=rms < 0.01,
                onset_score=onset,
            )
        )

    document = AudioFeatureDocument(
        schema_version="0.1.0",
        source_file="stem.wav",
        layer="bass",
        sample_rate=44100,
        channels=1,
        duration_sec=duration_sec,
        frame_size=2048,
        hop_size=512,
        frames=frames,
        global_features=AudioGlobalFeatures(
            peak=0.11,
            rms=0.03,
            duration_sec=duration_sec,
            estimated_silence_ratio=0.2,
            frame_count=frame_count,
            onset_count=4,
            mean_spectral_centroid_hz=150.0,
            mean_spectral_rolloff_hz=350.0,
        ),
    )
    path.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _write_refined_notes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = [
        RefinedNoteEvent(
            note_id="base_001",
            source="ripx",
            layer="bass",
            pitch_midi=40,
            pitch_name="E2",
            velocity=96,
            channel=0,
            original_start_sec=0.2,
            original_end_sec=0.6,
            aligned_start_sec=0.2,
            aligned_end_sec=0.6,
            refined_start_sec=0.2,
            refined_end_sec=0.6,
            refined_duration_sec=0.4,
            start_refinement_ms=0.0,
            end_refinement_ms=0.0,
            merged_note_ids=[],
            refinement_actions=["UNCHANGED"],
            refinement_confidence=0.92,
            reasons=[],
        ),
        RefinedNoteEvent(
            note_id="base_002",
            source="ripx",
            layer="bass",
            pitch_midi=43,
            pitch_name="G2",
            velocity=94,
            channel=0,
            original_start_sec=0.8,
            original_end_sec=1.2,
            aligned_start_sec=0.8,
            aligned_end_sec=1.2,
            refined_start_sec=0.8,
            refined_end_sec=1.2,
            refined_duration_sec=0.4,
            start_refinement_ms=0.0,
            end_refinement_ms=0.0,
            merged_note_ids=[],
            refinement_actions=["UNCHANGED"],
            refinement_confidence=0.93,
            reasons=[],
        ),
    ]

    document = RefinedNoteDocument(
        schema_version="0.1.0",
        aligned_notes_file="analysis/audio_aligned_note_events.json",
        audio_features_file="analysis/audio_features.json",
        validation_file="analysis/note_validation.json",
        layer="bass",
        sample_rate=44100,
        audio_duration_sec=2.0,
        timing_source="refined_audio_seconds",
        refinement_parameters={"monophonic": True},
        notes=notes,
    )
    path.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _build_project_layout(project_dir: Path) -> None:
    _write_working_midi(project_dir / "midi" / "working" / "working.mid")
    _write_audio_features(project_dir / "analysis" / "audio_features.json")
    _write_refined_notes(project_dir / "analysis" / "refined_note_events.json")


def _valid_ai_payload() -> dict[str, object]:
    return {
        "version": "1.0",
        "track_role": "bass_ai_completion",
        "timeline_reference": "wav_seconds",
        "global_confidence": 0.88,
        "notes": [
            {
                "note_id": "ai_bass_000001",
                "start_sec": 1.3,
                "end_sec": 1.6,
                "pitch_midi": 45,
                "velocity": 92,
                "confidence": 0.84,
                "reason": "continuation after repeated motif",
                "pattern_reference_note_ids": ["base_002"],
                "risk": "low",
            }
        ],
        "uncertain_regions": [],
        "summary": "Added a short continuation note after the local phrase ending.",
    }


def test_env_example_exists_and_dotenv_ignored() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_example = repo_root / ".env.example"
    gitignore = repo_root / ".gitignore"

    assert env_example.exists()
    assert env_example.read_text(encoding="utf-8").strip().startswith("OPENAI_API_KEY=")
    assert ".env" in gitignore.read_text(encoding="utf-8")


def test_prompt_builder_has_json_only_and_no_base_rewrite_instruction() -> None:
    pattern_pack = {
        "version": "1.0",
        "track_role": "bass",
        "timeline": {"duration_sec": 1.0},
        "base_midi_summary": {},
        "base_notes": [],
        "audio_activity_regions": [],
        "pitch_contour_summary": [],
        "pattern_windows": [],
        "instructions_for_ai": {},
    }
    system_prompt, user_prompt, _combined = build_ai_completion_prompts(pattern_pack, 64)

    assert "JSON only" in system_prompt
    assert "do not output a full bass transcription" in system_prompt
    assert "do not modify, delete, shorten, extend, or copy base MIDI notes" in system_prompt
    assert "Do not add more than 64 notes" in user_prompt


def test_pattern_pack_builder_has_required_top_level_fields(tmp_path: Path) -> None:
    project_dir = tmp_path / "project_pack"
    _build_project_layout(project_dir)

    result = build_pattern_pack(project_dir=project_dir, layer="bass")
    pack = result.pattern_pack

    assert pack["version"] == "1.0"
    assert pack["track_role"] == "bass"
    assert "timeline" in pack
    assert "base_midi_summary" in pack
    assert "base_notes" in pack
    assert "audio_activity_regions" in pack
    assert "pitch_contour_summary" in pack
    assert "pattern_windows" in pack
    assert "instructions_for_ai" in pack


def test_ai_completion_valid_notes_export_bass_ai_completion_midi(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "project_valid"
    _build_project_layout(project_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    params = AIPatternCompletionParameters(layer="bass", model="gpt-4o-mini")
    client = _FakeAIClient(_valid_ai_payload())
    report = complete_ai_pattern_completion(
        project_dir=project_dir,
        params=params,
        ai_client=client,
    )

    assert report.status == "ok"
    assert client.called is True
    assert report.accepted_note_count == 1
    assert report.output_midi_file is not None
    assert Path(report.output_midi_file).exists()


def test_ai_completion_rejects_invalid_notes_and_reports_reasons(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "project_invalid"
    _build_project_layout(project_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    payload = _valid_ai_payload()
    payload["notes"] = [
        {
            "note_id": "bad_negative",
            "start_sec": -0.1,
            "end_sec": 0.2,
            "pitch_midi": 45,
            "velocity": 100,
            "confidence": 0.8,
            "reason": "invalid",
            "pattern_reference_note_ids": ["base_001"],
            "risk": "low",
        },
        {
            "note_id": "bad_short",
            "start_sec": 1.21,
            "end_sec": 1.22,
            "pitch_midi": 45,
            "velocity": 100,
            "confidence": 0.8,
            "reason": "too short",
            "pattern_reference_note_ids": ["base_002"],
            "risk": "low",
        },
        {
            "note_id": "bad_pitch",
            "start_sec": 1.3,
            "end_sec": 1.6,
            "pitch_midi": 80,
            "velocity": 100,
            "confidence": 0.8,
            "reason": "too high",
            "pattern_reference_note_ids": ["base_002"],
            "risk": "medium",
        },
    ]

    report = complete_ai_pattern_completion(
        project_dir=project_dir,
        params=AIPatternCompletionParameters(layer="bass", model="gpt-4o-mini"),
        ai_client=_FakeAIClient(payload),
    )

    assert report.status == "ok"
    assert report.accepted_note_count == 0
    assert report.rejected_note_count == 3
    assert report.rejected_reasons.get("negative_time", 0) == 1
    assert report.rejected_reasons.get("note_too_short", 0) == 1
    assert report.rejected_reasons.get("pitch_outside_allowed_range", 0) == 1


def test_ai_completion_rejects_duplicate_overlap_with_base(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "project_duplicate"
    _build_project_layout(project_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    payload = _valid_ai_payload()
    payload["notes"] = [
        {
            "note_id": "dup_001",
            "start_sec": 0.2,
            "end_sec": 0.6,
            "pitch_midi": 40,
            "velocity": 90,
            "confidence": 0.7,
            "reason": "duplicate onset",
            "pattern_reference_note_ids": ["base_001"],
            "risk": "medium",
        }
    ]

    report = complete_ai_pattern_completion(
        project_dir=project_dir,
        params=AIPatternCompletionParameters(layer="bass", model="gpt-4o-mini"),
        ai_client=_FakeAIClient(payload),
    )

    assert report.status == "ok"
    assert report.accepted_note_count == 0
    assert report.rejected_reasons.get("duplicate_base_note_onset", 0) == 1


def test_ai_completion_dry_run_writes_pattern_and_prompt_without_api_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "project_dry"
    _build_project_layout(project_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = _FakeAIClient(_valid_ai_payload())
    report = complete_ai_pattern_completion(
        project_dir=project_dir,
        params=AIPatternCompletionParameters(layer="bass", dry_run=True),
        ai_client=client,
    )

    assert report.status == "ok"
    assert report.api_called is False
    assert client.called is False
    assert (project_dir / "analysis" / "ai_pattern_completion" / "pattern_pack.json").exists()
    assert (project_dir / "analysis" / "ai_pattern_completion" / "ai_prompt.txt").exists()
    assert not (project_dir / "midi" / "ai" / "bass_ai_completion.mid").exists()


def test_pipeline_process_stem_flag_runs_ai_pattern_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    midi_path = tmp_path / "candidate.mid"
    wav_path = tmp_path / "stem.wav"
    project_dir = tmp_path / "pipeline_project"

    _write_candidate_midi(midi_path)
    _write_stem_wav(wav_path)

    calls = {"count": 0}

    def _fake_complete_ai_pattern_completion(project_dir: Path, params, ai_client=None):
        _ = ai_client
        calls["count"] += 1
        assert params.layer == "bass"

        analysis_dir = project_dir / "analysis" / "ai_pattern_completion"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "pattern_pack.json").write_text("{}\n", encoding="utf-8")
        (analysis_dir / "ai_prompt.txt").write_text("prompt\n", encoding="utf-8")
        (analysis_dir / "bass_ai_completion.json").write_text("{}\n", encoding="utf-8")
        (analysis_dir / "bass_ai_completion_report.json").write_text("{}\n", encoding="utf-8")

        ai_midi_path = project_dir / "midi" / "ai" / "bass_ai_completion.mid"
        ai_midi_path.parent.mkdir(parents=True, exist_ok=True)
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        midi.save(ai_midi_path)

        return AIPatternCompletionReport(
            status="ok",
            project_dir=str(project_dir),
            layer="bass",
            model="gpt-4o-mini",
            api_called=False,
            dry_run=False,
            pattern_pack_file=str(analysis_dir / "pattern_pack.json"),
            ai_prompt_file=str(analysis_dir / "ai_prompt.txt"),
            ai_json_file=str(analysis_dir / "bass_ai_completion.json"),
            output_midi_file=str(ai_midi_path),
            proposed_note_count=1,
            accepted_note_count=1,
            rejected_note_count=0,
            rejected_reasons={},
            pitch_range_used={"min": 28, "max": 55},
            warning_count=0,
            warnings=[],
            error=None,
            raw_response_text="{}",
        )

    monkeypatch.setattr(
        "midi_cleaner.pipeline.process_stem.complete_ai_pattern_completion",
        _fake_complete_ai_pattern_completion,
    )

    report = process_stem_pipeline(
        input_midi=midi_path,
        input_wav=wav_path,
        source="ripx",
        layer="bass",
        project_dir=project_dir,
        params=PipelineProcessParameters(enable_ai_pattern_completion=True),
    )

    assert report.status == "ok"
    assert calls["count"] == 1
    assert "bass_ai_completion_midi" in report.output_files
