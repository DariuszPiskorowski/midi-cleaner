# midi-cleaner

Hermes MIDI Fidelity Engine project scaffold.

This repository will eventually clean and validate MIDI extracted from Suno/RipX against original WAV stems. The current scope is **Milestone 4**: environment guard, MIDI candidate import, lightweight WAV stem analysis, and first deterministic MIDI-vs-audio validation pass.

## Current Milestone

Milestone 4 implements:
- Python package scaffold
- Strict Python 3.11 guard
- Runtime/environment report via CLI doctor command
- MIDI candidate import from `.mid` using `mido`
- Hermes note-event JSON export and import report JSON
- WAV stem feature extraction using `numpy`, `scipy`, and `soundfile`
- Audio feature JSON export and audio analysis report JSON
- MIDI-vs-audio heuristic validation into note validation JSON and report JSON
- Tests for guard behavior, MIDI import, audio analysis, and validation behavior

MIDI cleaning, MIDI rewriting/export, rendering, UI, and ML are not implemented yet.

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

## Planned Pipeline

Suno/RipX stem WAV -> RipX MIDI candidate -> Hermes note-event JSON -> WAV comparison -> actions KEEP/MUTE/MERGE/SHORTEN/QUANTIZE/REASSIGN/DELETE -> cleaned MIDI and reports