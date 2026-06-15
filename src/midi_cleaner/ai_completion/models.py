from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AIPatternCompletionNote(BaseModel):
    note_id: str
    start_sec: float
    end_sec: float
    pitch_midi: int
    velocity: int
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    pattern_reference_note_ids: list[str] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"]


class AICompletionUncertainRegion(BaseModel):
    start_sec: float
    end_sec: float
    reason: str


class AIPatternCompletionOutput(BaseModel):
    version: Literal["1.0"]
    track_role: Literal["bass_ai_completion"]
    timeline_reference: Literal["wav_seconds"]
    global_confidence: float = Field(ge=0.0, le=1.0)
    notes: list[AIPatternCompletionNote] = Field(default_factory=list)
    uncertain_regions: list[AICompletionUncertainRegion] = Field(default_factory=list)
    summary: str


class AIPatternCompletionRejectedNote(BaseModel):
    note_id: str
    reason: str


class AIPatternCompletionReport(BaseModel):
    status: Literal["ok", "error"]
    feature: Literal["ai_pattern_completion"] = "ai_pattern_completion"
    project_dir: str
    layer: str
    model: str
    api_called: bool
    api_key_source: Literal["env", "dotenv"]
    dry_run: bool
    pattern_pack_file: str
    full_pattern_pack_file: str
    ai_request_pack_file: str
    ai_prompt_file: str
    full_pattern_pack_size_bytes: int
    ai_request_pack_size_bytes: int
    ai_prompt_size_bytes: int
    ai_json_file: str | None
    output_midi_file: str | None
    output_midi_path: str | None = None
    base_note_source: str | None = None
    json_retry_count: int = 0
    json_retry_reason: str | None = None
    retry_count: int = 0
    retry_reason: str | None = None
    first_pass_proposed_note_count: int = 0
    first_pass_rejected_reasons: dict[str, int] = Field(default_factory=dict)
    final_proposed_note_count: int = 0
    raw_response_file: str | None = None
    retry_raw_response_file: str | None = None
    openai_response_status: str | None = None
    openai_finish_reason: str | None = None
    max_output_tokens_used: int = 0
    proposed_note_count: int
    accepted_note_count: int
    rejected_note_count: int
    rejected_reasons: dict[str, int] = Field(default_factory=dict)
    pitch_range_used: dict[str, int | None] = Field(
        default_factory=lambda: {"min": None, "max": None}
    )
    warning_count: int
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    raw_response_text: str | None = None


def ai_output_json_schema(max_completion_notes: int) -> dict[str, object]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "version",
            "track_role",
            "timeline_reference",
            "global_confidence",
            "notes",
            "uncertain_regions",
            "summary",
        ],
        "properties": {
            "version": {"type": "string", "enum": ["1.0"]},
            "track_role": {"type": "string", "enum": ["bass_ai_completion"]},
            "timeline_reference": {"type": "string", "enum": ["wav_seconds"]},
            "global_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "notes": {
                "type": "array",
                "maxItems": int(max_completion_notes),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "note_id",
                        "start_sec",
                        "end_sec",
                        "pitch_midi",
                        "velocity",
                        "confidence",
                        "reason",
                        "pattern_reference_note_ids",
                        "risk",
                    ],
                    "properties": {
                        "note_id": {"type": "string"},
                        "start_sec": {"type": "number"},
                        "end_sec": {"type": "number"},
                        "pitch_midi": {"type": "integer"},
                        "velocity": {"type": "integer", "minimum": 1, "maximum": 127},
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "reason": {"type": "string"},
                        "pattern_reference_note_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                },
            },
            "uncertain_regions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start_sec", "end_sec", "reason"],
                    "properties": {
                        "start_sec": {"type": "number"},
                        "end_sec": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                },
            },
            "summary": {"type": "string"},
        },
    }
    return schema
