# midi-cleaner

Hermes MIDI Fidelity Engine project scaffold.

This repository will eventually clean and validate MIDI extracted from Suno/RipX against original WAV stems. The current scope is **Milestone 10**: audio-time canonical note alignment and aligned-second MIDI export.

## Current Milestone

Milestone 10 implements:
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
- Review/cleaned MIDI export using aligned_start_sec/aligned_end_sec when alignment data is provided
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

Suno/RipX stem WAV -> RipX MIDI candidate -> Hermes note-event JSON -> WAV comparison -> actions KEEP/MUTE/MERGE/SHORTEN/QUANTIZE/REASSIGN/DELETE -> cleaned MIDI and reports