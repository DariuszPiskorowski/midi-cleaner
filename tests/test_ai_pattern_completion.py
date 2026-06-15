from __future__ import annotations

import json
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

from midi_cleaner.ai_completion.compact_pack import build_ai_request_pack
from midi_cleaner.ai_completion.models import AIPatternCompletionReport
from midi_cleaner.ai_completion.openai_client import (
    OpenAIPatternCompletionResult,
    calculate_max_output_tokens,
)
from midi_cleaner.ai_completion.pattern_pack import BasePatternNote, PatternPackBuildResult
from midi_cleaner.ai_completion.pattern_pack import build_pattern_pack
from midi_cleaner.ai_completion.prompt import build_ai_completion_prompts
from midi_cleaner.ai_completion.service import (
    AIPatternCompletionError,
    AIPatternCompletionParameters,
    _build_json_retry_feedback_context,
    _resolve_openai_api_key,
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
        max_output_tokens: int | None = None,
    ) -> OpenAIPatternCompletionResult:
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
        return OpenAIPatternCompletionResult(
            raw_response_text=json.dumps(self.payload),
            parsed_payload=self.payload,
            response_debug={
                "status": "completed",
                "finish_reason": "stop",
                "max_output_tokens": int(max_output_tokens or 4000),
            },
            max_output_tokens_used=int(max_output_tokens or 4000),
        )


class _SequenceAIClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.call_count = 0

    def complete_pattern(
        self,
        *,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_completion_notes: int,
        max_output_tokens: int | None = None,
    ) -> OpenAIPatternCompletionResult:
        _ = (
            api_key,
            model,
            system_prompt,
            user_prompt,
            temperature,
            max_completion_notes,
            max_output_tokens,
        )
        payload = self.payloads[min(self.call_count, len(self.payloads) - 1)]
        self.call_count += 1
        return OpenAIPatternCompletionResult(
            raw_response_text=json.dumps(payload),
            parsed_payload=payload,
            response_debug={
                "status": "completed",
                "finish_reason": "stop",
                "max_output_tokens": int(max_output_tokens or 4000),
            },
            max_output_tokens_used=int(max_output_tokens or 4000),
        )


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
    ai_request_pack = {
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
    system_prompt, user_prompt, _combined = build_ai_completion_prompts(ai_request_pack, 64)

    assert "JSON only" in system_prompt
    assert "do not output a full bass transcription" in system_prompt
    assert "do not modify, delete, shorten, extend, or copy base MIDI notes" in system_prompt
    assert "pattern_reference_note_ids are evidence/examples only" in system_prompt
    assert "never create an AI completion note at the same start_sec as a base note" in system_prompt
    assert "Do not add more than 64 notes" in user_prompt
    assert "Bad example (reject):" in user_prompt
    assert "Good example (allowed):" in user_prompt

    payload = user_prompt.split("Pattern pack JSON:\n", maxsplit=1)[1]
    assert payload == json.dumps(ai_request_pack, separators=(",", ":"), ensure_ascii=False)


def test_calculate_max_output_tokens_dynamic_budget() -> None:
    assert calculate_max_output_tokens(1) == 4000
    assert calculate_max_output_tokens(16) == 4520
    assert calculate_max_output_tokens(64) == 12000
    assert calculate_max_output_tokens(120) == 12000


def test_json_retry_feedback_context_is_strict_and_truncated() -> None:
    invalid_raw = "{" + ("x" * 2100)
    context = _build_json_retry_feedback_context(raw_response_text=invalid_raw)

    assert (
        "Your previous response was not valid JSON. Return JSON only matching the schema. "
        "No markdown, no comments, no prose."
    ) in context
    excerpt = context.split("Invalid response excerpt (max 2000 chars):\n", maxsplit=1)[1]
    assert len(excerpt) <= 2000


def test_compact_request_pack_limits_and_metadata() -> None:
    base_notes = [
        {
            "note_id": f"base_{index:04d}",
            "start_sec": float(index) * 0.5,
            "end_sec": (float(index) * 0.5) + 0.25,
            "duration_sec": 0.25,
            "pitch_midi": 36 + (index % 6),
            "velocity": 90,
            "confidence": 0.9,
            "source": "ripx",
            "reasons": [],
        }
        for index in range(420)
    ]
    activity_regions = [
        {
            "start_sec": float(index),
            "end_sec": float(index) + 0.9,
            "duration_sec": 0.9,
            "rms_peak": 0.03 + (index * 0.0002),
            "rms_mean": 0.01,
            "onset_count": (index % 10),
            "dominant_pitch_midi": 36 + (index % 7),
            "pitch_confidence": 0.7,
        }
        for index in range(300)
    ]
    pitch_sections = [
        {
            "start_sec": float(index) * 0.5,
            "end_sec": (float(index) * 0.5) + 0.5,
            "dominant_pitch_midi": 36 + (index % 12),
            "pitch_midi_mean": 40.0,
            "voiced_ratio": 0.8,
            "mean_confidence": 0.75,
        }
        for index in range(320)
    ]
    pattern_windows = [
        {
            "window_index": index,
            "start_sec": float(index) * 4.0,
            "end_sec": (float(index) * 4.0) + 4.0,
            "base_notes": [f"base_{note:04d}" for note in range(index * 8, (index * 8) + 24)],
            "rhythmic_summary": {
                "note_onsets_sec": [float(i) / 8.0 for i in range(32)],
                "intervals_sec": [0.125 for _ in range(32)],
                "common_durations_sec": [0.25, 0.5],
            },
        }
        for index in range(40)
    ]

    pattern_pack = {
        "version": "1.0",
        "track_role": "bass",
        "timeline": {"duration_sec": 200.0},
        "base_midi_summary": {"note_count": len(base_notes)},
        "base_notes": base_notes,
        "audio_activity_regions": activity_regions,
        "pitch_contour_summary": pitch_sections,
        "pattern_windows": pattern_windows,
        "instructions_for_ai": {"goal": "fill missing patterns"},
    }

    compact = build_ai_request_pack(pattern_pack)

    assert compact["compact_request"] is True
    assert compact["source_pack_was_compacted"] is True
    assert compact["original_counts"]["base_notes"] == 420
    assert compact["included_counts"]["base_notes"] <= 180
    assert compact["included_counts"]["audio_activity_regions"] <= 180
    assert compact["included_counts"]["pitch_contour_summary"] <= 240
    assert compact["base_occupancy_rules"]["do_not_place_ai_note_on_base_onset_within_ms"] == 30
    assert compact["base_occupancy_rules"]["do_not_overlap_same_or_near_pitch_base_note_ratio"] == 0.7
    assert compact["base_occupancy_rules"]["completion_track_role"] == "additive_missing_pattern_only"
    assert all(bool(item.get("occupied")) for item in compact["base_notes"])


def test_service_uses_compact_pack_for_prompt_and_writes_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "project_compact"
    project_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    base_notes_records = [
        {
            "note_id": f"base_{index:04d}",
            "start_sec": float(index) * 0.5,
            "end_sec": (float(index) * 0.5) + 0.25,
            "duration_sec": 0.25,
            "pitch_midi": 36 + (index % 6),
            "velocity": 96,
            "confidence": 0.9,
            "source": "ripx",
            "reasons": [],
        }
        for index in range(360)
    ]

    pattern_windows = [
        {
            "window_index": index,
            "start_sec": float(index) * 4.0,
            "end_sec": (float(index) * 4.0) + 4.0,
            "base_notes": [f"base_{note:04d}" for note in range(index * 6, (index * 6) + 24)],
            "rhythmic_summary": {
                "note_onsets_sec": [float(i) / 8.0 for i in range(32)],
                "intervals_sec": [0.125 for _ in range(32)],
                "common_durations_sec": [0.25, 0.5],
            },
        }
        for index in range(32)
    ]

    pattern_pack = {
        "version": "1.0",
        "track_role": "bass",
        "timeline": {
            "duration_sec": 240.0,
            "time_origin": "wav_seconds",
            "ticks_per_beat": 480,
            "tempo_bpm": 120.0,
            "midi_source": "working.mid",
        },
        "base_midi_summary": {"note_count": len(base_notes_records)},
        "base_notes": base_notes_records,
        "audio_activity_regions": [
            {
                "start_sec": float(index),
                "end_sec": float(index) + 0.75,
                "duration_sec": 0.75,
                "rms_peak": 0.03 + (index * 0.0002),
                "rms_mean": 0.01,
                "onset_count": index % 8,
                "dominant_pitch_midi": 36 + (index % 8),
                "pitch_confidence": 0.7,
            }
            for index in range(220)
        ],
        "pitch_contour_summary": [
            {
                "start_sec": float(index) * 0.5,
                "end_sec": (float(index) * 0.5) + 0.5,
                "dominant_pitch_midi": 36 + (index % 12),
                "pitch_midi_mean": 40.0,
                "voiced_ratio": 0.85,
                "mean_confidence": 0.74,
            }
            for index in range(300)
        ],
        "pattern_windows": pattern_windows,
        "instructions_for_ai": {"goal": "complete bass continuity"},
    }

    base_notes_for_validation = [
        BasePatternNote(
            note_id=str(item["note_id"]),
            start_sec=float(item["start_sec"]),
            end_sec=float(item["end_sec"]),
            duration_sec=float(item["duration_sec"]),
            pitch_midi=int(item["pitch_midi"]),
            velocity=int(item["velocity"]),
            confidence=float(item["confidence"]),
            source="ripx",
            reasons=[],
        )
        for item in base_notes_records
    ]

    monkeypatch.setattr(
        "midi_cleaner.ai_completion.service.build_pattern_pack",
        lambda project_dir, layer: PatternPackBuildResult(
            pattern_pack=pattern_pack,
            base_notes=base_notes_for_validation,
            duration_sec=240.0,
            ticks_per_beat=480,
            tempo_us_per_beat=500000,
            base_note_source="working.mid",
            warnings=[],
        ),
    )

    report = complete_ai_pattern_completion(
        project_dir=project_dir,
        params=AIPatternCompletionParameters(layer="bass", model="gpt-4o-mini"),
        ai_client=_FakeAIClient(_valid_ai_payload()),
    )

    analysis_dir = project_dir / "analysis" / "ai_pattern_completion"
    pattern_pack_path = analysis_dir / "pattern_pack.json"
    ai_request_pack_path = analysis_dir / "ai_request_pack.json"
    ai_prompt_path = analysis_dir / "ai_prompt.txt"

    assert report.status == "ok"
    assert ai_request_pack_path.exists()
    assert pattern_pack_path.exists()
    assert ai_prompt_path.exists()

    full_size = pattern_pack_path.stat().st_size
    compact_size = ai_request_pack_path.stat().st_size
    assert compact_size < full_size
    assert report.full_pattern_pack_size_bytes == full_size
    assert report.ai_request_pack_size_bytes == compact_size

    full_pack = json.loads(pattern_pack_path.read_text(encoding="utf-8"))
    request_pack = json.loads(ai_request_pack_path.read_text(encoding="utf-8"))
    prompt_text = ai_prompt_path.read_text(encoding="utf-8")
    compact_payload = json.dumps(request_pack, separators=(",", ":"), ensure_ascii=False)
    full_pretty_payload = json.dumps(full_pack, indent=2)

    assert compact_payload in prompt_text
    assert full_pretty_payload not in prompt_text
    assert request_pack["included_counts"]["base_notes"] < request_pack["original_counts"]["base_notes"]

    full_ids = {str(item["note_id"]) for item in full_pack["base_notes"]}
    compact_ids = {str(item["note_id"]) for item in request_pack["base_notes"]}
    removed_ids = sorted(full_ids - compact_ids)
    assert removed_ids


def test_dotenv_overrides_stale_process_key_for_ai_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cwd_dir = tmp_path / "cwd"
    project_dir = tmp_path / "project"
    cwd_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    (cwd_dir / ".env").write_text('OPENAI_API_KEY="cwd-key"\n', encoding="utf-8")
    (project_dir / ".env").write_text('OPENAI_API_KEY="project-key"\n', encoding="utf-8")

    monkeypatch.chdir(cwd_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "stale-process-key")

    api_key, source = _resolve_openai_api_key(project_dir)

    assert api_key == "project-key"
    assert source == "dotenv"


def test_ai_completion_fails_fast_when_compact_prompt_is_too_large(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "project_large_prompt"
    _build_project_layout(project_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    huge_prompt = "x" * 250_001
    monkeypatch.setattr(
        "midi_cleaner.ai_completion.service.build_ai_completion_prompts",
        lambda ai_request_pack, max_completion_notes: ("system", huge_prompt, huge_prompt),
    )

    client = _FakeAIClient(_valid_ai_payload())
    with pytest.raises(AIPatternCompletionError) as exc_info:
        complete_ai_pattern_completion(
            project_dir=project_dir,
            params=AIPatternCompletionParameters(layer="bass", model="gpt-4o-mini"),
            ai_client=client,
        )

    assert "AI request pack is too large for model context" in str(exc_info.value)
    assert client.called is False

    report_path = project_dir / "analysis" / "ai_pattern_completion" / "bass_ai_completion_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "AI request pack is too large for model context" in payload["error"]


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
    assert result.base_note_source.endswith("working.mid")


def test_ai_completion_retries_once_when_first_pass_is_mostly_duplicate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "project_retry"
    _build_project_layout(project_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    first_payload = {
        "version": "1.0",
        "track_role": "bass_ai_completion",
        "timeline_reference": "wav_seconds",
        "global_confidence": 0.7,
        "notes": [
            {
                "note_id": "ai_bass_dup_000001",
                "start_sec": 0.0,
                "end_sec": 0.5,
                "pitch_midi": 40,
                "velocity": 80,
                "confidence": 0.8,
                "reason": "reinforcing existing motif",
                "pattern_reference_note_ids": ["working_t00_ch00_p040_n000001"],
                "risk": "low",
            }
        ],
        "uncertain_regions": [],
        "summary": "duplicate first pass",
    }
    second_payload = {
        "version": "1.0",
        "track_role": "bass_ai_completion",
        "timeline_reference": "wav_seconds",
        "global_confidence": 0.82,
        "notes": [
            {
                "note_id": "ai_bass_new_000001",
                "start_sec": 0.62,
                "end_sec": 0.92,
                "pitch_midi": 43,
                "velocity": 92,
                "confidence": 0.84,
                "reason": "continuation after base note ending",
                "pattern_reference_note_ids": ["working_t00_ch00_p040_n000001"],
                "risk": "low",
            }
        ],
        "uncertain_regions": [],
        "summary": "new additive note",
    }

    sequence_client = _SequenceAIClient([first_payload, second_payload])
    report = complete_ai_pattern_completion(
        project_dir=project_dir,
        params=AIPatternCompletionParameters(layer="bass", model="gpt-4o-mini"),
        ai_client=sequence_client,
    )

    assert sequence_client.call_count == 2
    assert report.retry_count == 1
    assert report.retry_reason is not None
    assert report.first_pass_proposed_note_count == 1
    assert report.first_pass_rejected_reasons.get("duplicate_base_note_onset", 0) == 1
    assert report.final_proposed_note_count == 1
    assert report.accepted_note_count == 1


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
    duplicate_reject_count = report.rejected_reasons.get("duplicate_base_note_onset", 0)
    duplicate_reject_count += report.rejected_reasons.get("duplicate_base_note_overlap", 0)
    assert duplicate_reject_count == 1


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
    assert (project_dir / "analysis" / "ai_pattern_completion" / "ai_request_pack.json").exists()
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
        (analysis_dir / "ai_request_pack.json").write_text("{}\n", encoding="utf-8")
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
            api_key_source="env",
            dry_run=False,
            pattern_pack_file=str(analysis_dir / "pattern_pack.json"),
            full_pattern_pack_file=str(analysis_dir / "pattern_pack.json"),
            ai_request_pack_file=str(analysis_dir / "ai_request_pack.json"),
            ai_prompt_file=str(analysis_dir / "ai_prompt.txt"),
            full_pattern_pack_size_bytes=2,
            ai_request_pack_size_bytes=2,
            ai_prompt_size_bytes=7,
            ai_json_file=str(analysis_dir / "bass_ai_completion.json"),
            output_midi_file=str(ai_midi_path),
            output_midi_path=str(ai_midi_path),
            base_note_source="working.mid",
            json_retry_count=0,
            json_retry_reason=None,
            retry_count=0,
            retry_reason=None,
            first_pass_proposed_note_count=1,
            first_pass_rejected_reasons={},
            final_proposed_note_count=1,
            raw_response_file=str(analysis_dir / "openai_raw_response_first_pass.txt"),
            retry_raw_response_file=None,
            openai_response_status="completed",
            openai_finish_reason="stop",
            max_output_tokens_used=4000,
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
