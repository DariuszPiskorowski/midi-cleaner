from __future__ import annotations

import json


def build_ai_completion_prompts(
    ai_request_pack: dict[str, object],
    max_completion_notes: int,
    feedback_context: str | None = None,
) -> tuple[str, str, str]:
    feedback_section = ""
    if feedback_context:
        feedback_section = f"Validation feedback from previous attempt:\n{feedback_context}\n\n"

    allowed_regions_count = 0
    allowed_regions_raw = ai_request_pack.get("allowed_completion_regions")
    if isinstance(allowed_regions_raw, list):
        allowed_regions_count = len(allowed_regions_raw)

    system_prompt = (
        "You are Hermes bass MIDI repair, not a free music generator.\n\n"
        "You are not composing a new bassline. You are filling local gaps. "
        "For each allowed region, read notes_before and notes_after. "
        "Continue the local motif only inside write_start_sec/write_end_sec. "
        "The context notes before/after are examples of the existing pattern "
        "and must not be copied outside the writable region.\n\n"
        "Core mission:\n"
        "- complete missing bass pattern fragments only inside explicit allowed_completion_regions,\n"
        "- preserve synchronization to the original WAV timeline and Hermes working.mid,\n"
        "- keep output additive to base MIDI (do not replace/rewrite the base line).\n\n"
        "Hard constraints:\n"
        "- do not compose a new bassline for the whole song,\n"
        "- do not fill the full timeline,\n"
        "- every generated note must belong to exactly one allowed_completion_region,\n"
        "- any note outside allowed_completion_regions is invalid,\n"
        "- write only inside write_start_sec/write_end_sec for each region,\n"
        "- do not place notes earlier than the first allowed region,\n"
        "- do not continue filling after the last allowed region,\n"
        "- context before/after each region is for inference only, not writable space,\n"
        "- max_completion_notes is an absolute upper bound, not a target,\n"
        "- output fewer notes when uncertain,\n"
        "- do not force notes for optional/low-confidence regions,\n"
        "- attempt continuation for required regions with clear motif evidence,\n"
        "- output zero notes when evidence is insufficient.\n\n"
        "Musical constraints:\n"
        "- do not modify, delete, shorten, extend, or copy base MIDI notes,\n"
        "- never create an AI note at same onset as a base note,\n"
        "- never copy exact base note start/end/pitch,\n"
        "- pattern_reference_note_ids are evidence only, never notes to recreate,\n"
        "- respect local_pitch_set, local_pitch_range, local motif rhythm, and region density,\n"
        "- do not invent pitches outside local_pitch_set unless explicitly allowed by region,\n"
        "- avoid random register jumps (especially out-of-range high bass notes),\n"
        "- keep local phrase consistency and timeline sync in wav_seconds with working.mid.\n\n"
        "The AI response must be JSON only, no markdown, no comments, no prose.\n"
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
        "- pattern_reference_note_ids are references only; do not copy those notes.\n"
        "- never place a completion note on the same onset as a base note.\n"
        "- never output a note that matches base note start/end/pitch.\n"
        "- only write notes inside allowed_completion_regions.\n"
        "- for each region, write only inside write_start_sec/write_end_sec.\n"
        "- infer continuation from ordered notes_before and notes_after, not from full-song composition.\n"
        "- for optional/low-confidence regions, prefer zero notes unless motif continuation is clear.\n"
        "- max_completion_notes is an upper bound, not a target.\n"
        "- it is valid and preferred to output fewer notes than the limit.\n"
        "- it is valid to output zero notes if listed regions lack evidence.\n"
        "- do not use reasons like 'reinforcing existing motif' or 'maintaining harmonic structure' "
        "when a note overlaps/copies base MIDI.\n"
        f"- allowed_completion_regions_count is {allowed_regions_count}.\n"
        f"- Do not add more than {max_completion_notes} notes.\n\n"
        "Region-scope examples:\n"
        "Bad example (reject):\n"
        "Allowed region: 28.000-31.500 sec.\n"
        "AI notes at 2.700, 10.300, 18.100 sec.\n"
        "Reason: outside allowed completion region.\n\n"
        "Bad example (reject):\n"
        "Allowed pitch range: MIDI 24-36.\n"
        "AI note pitch_midi=52.\n"
        "Reason: outside bass register/local pitch constraint.\n\n"
        "Bad example (reject):\n"
        "Region expected note count: 2-6.\n"
        "AI outputs 30 notes because max_completion_notes is 64.\n"
        "Reason: max_completion_notes is not a target.\n\n"
        "Good example (allowed):\n"
        "Allowed region: 28.000-31.500 sec.\n"
        "AI uses up to 5 sec before/after context only to infer rhythm and pitch.\n"
        "AI outputs 3 notes fully inside 28.000-31.500 sec, matching local key/pitch/rhythm.\n\n"
        "Anti-duplicate examples:\n"
        "Bad example (reject):\n"
        "Base note: start=2.774785 end=3.240499 pitch=36\n"
        "AI note: start=2.774785 end=3.240499 pitch=36\n"
        "Reason: duplicate copy of occupied base note.\n\n"
        "Good example (allowed):\n"
        "Base pattern implies continuation after a base note ends.\n"
        "AI note starts after base note end or inside a true gap, not at the same onset.\n\n"
        + feedback_section
        + "Pattern pack JSON:\n"
        + json.dumps(ai_request_pack, separators=(",", ":"), ensure_ascii=False)
    )

    combined_prompt = (
        "=== SYSTEM PROMPT ===\n"
        + system_prompt
        + "\n=== USER PROMPT ===\n"
        + user_prompt
    )
    return system_prompt, user_prompt, combined_prompt
