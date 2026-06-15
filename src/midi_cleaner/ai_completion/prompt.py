from __future__ import annotations

import json


def build_ai_completion_prompts(
    ai_request_pack: dict[str, object],
    max_completion_notes: int,
) -> tuple[str, str, str]:
    system_prompt = (
        "You are an expert MIDI bass pattern completion engine.\n\n"
        "You receive a JSON pattern pack generated from:\n"
        "- the original bass WAV stem,\n"
        "- a synchronized base MIDI created by Hermes,\n"
        "- pitch contour,\n"
        "- energy envelope,\n"
        "- onset/transient information,\n"
        "- timeline data.\n\n"
        "Your job:\n"
        "- infer the musical bass pattern,\n"
        "- add only missing or musically implied continuation notes,\n"
        "- return a JSON patch for a separate synchronized AI completion MIDI track,\n"
        "- do not modify, delete, shorten, extend, or copy base MIDI notes,\n"
        "- do not output a full bass transcription,\n"
        "- do not duplicate existing notes unless there is a clear musical reason and it will not "
        "cause doubled/flammed bass,\n"
        "- use the same WAV-second timeline,\n"
        "- keep timing tight with the base MIDI,\n"
        "- prefer the same pitch register/range as the base bass pattern,\n"
        "- prefer local pattern consistency over creative variation,\n"
        "- if uncertain, output fewer notes, not more,\n"
        "- the completion track must sound like one bass line when played together with the base MIDI.\n\n"
        "The AI response must be JSON only, no markdown, no comments.\n"
    )

    user_prompt = (
        "Return JSON only and follow this schema exactly:\n"
        "{\n"
        '  "version": "1.0",\n'
        '  "track_role": "bass_ai_completion",\n'
        '  "timeline_reference": "wav_seconds",\n'
        '  "global_confidence": 0.0,\n'
        '  "notes": [\n'
        "    {\n"
        '      "note_id": "ai_bass_000001",\n'
        '      "start_sec": 0.0,\n'
        '      "end_sec": 0.0,\n'
        '      "pitch_midi": 36,\n'
        '      "velocity": 90,\n'
        '      "confidence": 0.0,\n'
        '      "reason": "short explanation",\n'
        '      "pattern_reference_note_ids": ["..."],\n'
        '      "risk": "low"\n'
        "    }\n"
        "  ],\n"
        '  "uncertain_regions": [\n'
        "    {\n"
        '      "start_sec": 0.0,\n'
        '      "end_sec": 0.0,\n'
        '      "reason": "why AI avoided adding notes"\n'
        "    }\n"
        "  ],\n"
        '  "summary": "short summary"\n'
        "}\n\n"
        "Rules:\n"
        "- notes may be empty.\n"
        "- start_sec and end_sec are absolute seconds in WAV timeline.\n"
        "- end_sec must be greater than start_sec.\n"
        "- pitch_midi must be integer.\n"
        "- velocity must be 1-127.\n"
        "- confidence must be 0-1.\n"
        "- risk must be one of low, medium, high.\n"
        f"- Do not add more than {max_completion_notes} notes.\n\n"
        "Pattern pack JSON:\n"
        + json.dumps(ai_request_pack, separators=(",", ":"), ensure_ascii=False)
    )

    combined_prompt = (
        "=== SYSTEM PROMPT ===\n"
        + system_prompt
        + "\n=== USER PROMPT ===\n"
        + user_prompt
    )
    return system_prompt, user_prompt, combined_prompt
