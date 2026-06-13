# midi-cleaner

Hermes MIDI Fidelity Engine project scaffold.

This repository cleans and validates MIDI extracted from Suno/RipX against original WAV stems. The current scope is **Milestone 11**: bass MIDI quality refinement on top of canonical audio-time alignment.

Milestone 10.1 integration and reporting hardening is now in place:
- Validation consumes audio-aligned timing when alignment data exists.
- Cleanup exports consume audio-aligned timing when alignment data exists.
- WAV/audio seconds are canonical for synchronization.
- BPM/bar calculations are not trusted for synchronization.
- A coarse global offset search runs before local per-note alignment.
- Export ticks-per-beat defaults to source note_events ticks-per-beat unless explicitly overridden.
- Generated projects outputs under `projects/*` are ignored and should not be committed.

Milestone 11 quality refinement is now in place:
- Bass attack-aware start adjustment can move note starts earlier toward audible attack rise.
- False retrigger suppression can merge near-pitch retriggers when there is no real silence.
- Sustain-tail extension can preserve audible decay while respecting extension caps.
- Minimum-duration and monophonic overlap cleanup improve practical bass playback.
- Practical DAW output adds `midi/working/working.mid` and `midi/working/rejected.mid` with optional diagnostic MIDI.

## Current Milestone

Milestone 11 currently implements:
- Python package scaffold
- Strict Python 3.11 guard
- Runtime/environment report via CLI doctor command
- MIDI candidate import from `.mid` using `mido`
- Hermes note-event JSON export and import report JSON
- WAV stem feature extraction using `numpy`, `scipy`, and `soundfile`
- Audio feature JSON export and audio analysis report JSON
- MIDI-vs-audio heuristic validation into note validation JSON and report JSON
- Non-destructive cleanup plan generation from note validation JSON
- Non-destructive review MIDI export grouped by cleanup action
- Conservative cleaned/review/rejected MIDI export from cleanup plan
- One-command process-stem pipeline that orchestrates existing stages
- Static QA artifacts: qa_summary.json, qa_notes.csv, qa_report.html
- Audio-time canonical note alignment into analysis/audio_aligned_note_events.json
- Audio alignment report in analysis/audio_alignment_report.json
- Validation uses aligned timing via audio_aligned_note_events.json when available
- Review/cleaned MIDI export using aligned_start_sec/aligned_end_sec when alignment data is provided
- Bass refinement stage output in analysis/refined_note_events.json
- Bass refinement report in analysis/bass_refinement_report.json
- Attack-aware onset refinement and false-retrigger merge suppression for bass
- Sustain-tail extension, minimum-duration extension, and optional monophonic overlap resolution
- Practical working export in midi/working/working.mid and midi/working/rejected.mid
- Optional midi/working/diagnostic.mid for quick inspection
- QA report includes explicit Audio-Time Alignment / Sync metrics and timing sources
- QA report includes refinement summary metrics and per-note refinement columns
- Tests for guard behavior, MIDI import, audio analysis, validation, cleanup planning, review MIDI export, cleaned MIDI export, process-stem pipeline, and QA reports

Destructive note deletion, rendering, UI, and ML are not implemented yet.

## Requirements

- Python 3.11.x
- [uv](https://docs.astral.sh/uv/) for dependency and environment management

## Install uv

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternative methods are available in the uv docs.

## Project Setup

```powershell
uv sync
```

## Run Tests

```powershell
uv run pytest
```

## Runtime Guard Check

```powershell
uv run midi-cleaner doctor --json
```

Human-readable mode:

```powershell
uv run midi-cleaner doctor
```

Write JSON report to file:

```powershell
uv run midi-cleaner doctor --output runtime_report.json
```

## MIDI Candidate Import

```powershell
uv run midi-cleaner midi import-candidate INPUT_MIDI --source ripx --layer bass --output OUTPUT_JSON --report REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner midi import-candidate .\candidate.mid --source ripx --layer bass --output .\artifacts\note_events.json --report .\artifacts\midi_import_report.json
```

## WAV Stem Analysis

```powershell
uv run midi-cleaner audio analyze-stem INPUT_WAV --layer bass --output OUTPUT_JSON --report REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner audio analyze-stem .\stem.wav --layer bass --output .\artifacts\audio_features.json --report .\artifacts\audio_analysis_report.json
```

## MIDI-vs-Audio Validation

```powershell
uv run midi-cleaner validate midi-vs-audio --notes NOTE_EVENTS_JSON --audio-features AUDIO_FEATURES_JSON --output OUTPUT_JSON --report REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner validate midi-vs-audio --notes .\artifacts\note_events.json --audio-features .\artifacts\audio_features.json --output .\artifacts\note_validation.json --report .\artifacts\midi_audio_validation_report.json
```

## Audio-Time Note Alignment

```powershell
uv run midi-cleaner validate align-audio-time --notes NOTE_EVENTS_JSON --audio-features AUDIO_FEATURES_JSON --output OUTPUT_JSON --report REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner validate align-audio-time --notes .\artifacts\note_events.json --audio-features .\artifacts\audio_features.json --output .\artifacts\audio_aligned_note_events.json --report .\artifacts\audio_alignment_report.json
```

## Bass Timing Refinement

```powershell
uv run midi-cleaner refine bass --aligned-notes AUDIO_ALIGNED_NOTE_EVENTS_JSON --audio-features AUDIO_FEATURES_JSON --validation NOTE_VALIDATION_JSON --output REFINED_NOTE_EVENTS_JSON --report REFINEMENT_REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner refine bass --aligned-notes .\artifacts\audio_aligned_note_events.json --audio-features .\artifacts\audio_features.json --validation .\artifacts\note_validation.json --output .\artifacts\refined_note_events.json --report .\artifacts\bass_refinement_report.json
```

## Cleanup Plan (Non-Destructive)

```powershell
uv run midi-cleaner cleanup plan --validation NOTE_VALIDATION_JSON --output OUTPUT_JSON --report REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner cleanup plan --validation .\artifacts\note_validation.json --output .\artifacts\cleanup_plan.json --report .\artifacts\cleanup_plan_report.json
```

## Review MIDI Export (Non-Destructive)

```powershell
uv run midi-cleaner cleanup export-review-midi --notes NOTE_EVENTS_JSON --plan CLEANUP_PLAN_JSON --output-dir OUTPUT_DIR --report REPORT_JSON
```

Optional aligned timing input:

```powershell
uv run midi-cleaner cleanup export-review-midi --notes NOTE_EVENTS_JSON --plan CLEANUP_PLAN_JSON --audio-aligned-notes AUDIO_ALIGNED_NOTE_EVENTS_JSON --output-dir OUTPUT_DIR --report REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner cleanup export-review-midi --notes .\artifacts\note_events.json --plan .\artifacts\cleanup_plan.json --output-dir .\artifacts\review_midi --report .\artifacts\review_midi\export_report.json
```

## Conservative Cleaned MIDI Export

```powershell
uv run midi-cleaner cleanup export-cleaned-midi --notes NOTE_EVENTS_JSON --plan CLEANUP_PLAN_JSON --output-dir OUTPUT_DIR --report REPORT_JSON
```

Optional aligned timing input:

```powershell
uv run midi-cleaner cleanup export-cleaned-midi --notes NOTE_EVENTS_JSON --plan CLEANUP_PLAN_JSON --audio-aligned-notes AUDIO_ALIGNED_NOTE_EVENTS_JSON --output-dir OUTPUT_DIR --report REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner cleanup export-cleaned-midi --notes .\artifacts\note_events.json --plan .\artifacts\cleanup_plan.json --output-dir .\artifacts\cleaned_midi --report .\artifacts\cleaned_midi\cleaned_export_report.json
```

## Working MIDI Export (Practical DAW Output)

```powershell
uv run midi-cleaner cleanup export-working-midi --notes NOTE_EVENTS_JSON --plan CLEANUP_PLAN_JSON --refined-notes REFINED_NOTE_EVENTS_JSON --output-dir OUTPUT_DIR --report REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner cleanup export-working-midi --notes .\artifacts\note_events.json --plan .\artifacts\cleanup_plan.json --refined-notes .\artifacts\refined_note_events.json --output-dir .\artifacts\working_midi --report .\artifacts\working_midi\working_export_report.json
```

## End-to-End Process Stem Pipeline

```powershell
uv run midi-cleaner pipeline process-stem --midi INPUT_MIDI --wav INPUT_WAV --source ripx --layer bass --project-dir PROJECT_DIR
```

Example:

```powershell
uv run midi-cleaner pipeline process-stem --midi .\candidate.mid --wav .\stem.wav --source ripx --layer bass --project-dir .\artifacts\pipeline_run
```

## Static QA Report

```powershell
uv run midi-cleaner pipeline qa-report --project-dir PROJECT_DIR
```

Example:

```powershell
uv run midi-cleaner pipeline qa-report --project-dir .\artifacts\pipeline_run
```

## Planned Pipeline

Suno/RipX stem WAV -> RipX MIDI candidate -> Hermes note-event JSON -> WAV comparison -> audio-time alignment -> validation -> bass refinement -> cleanup planning -> working/rejected (plus backward-compatible review/cleaned exports) -> QA artifacts