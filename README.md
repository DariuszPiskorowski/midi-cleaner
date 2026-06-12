# midi-cleaner

Hermes MIDI Fidelity Engine project scaffold.

This repository will eventually clean and validate MIDI extracted from Suno/RipX against original WAV stems. The current scope is **Milestone 1** only: project skeleton and strict runtime environment guard.

## Current Milestone

Milestone 1 implements:
- Python package scaffold
- Strict Python 3.11 guard
- Runtime/environment report via CLI doctor command
- Basic tests for guard behavior and CLI JSON output

No MIDI cleaning or audio processing is implemented yet.

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

## Planned Pipeline

Suno/RipX stem WAV -> RipX MIDI candidate -> Hermes note-event JSON -> WAV comparison -> actions KEEP/MUTE/MERGE/SHORTEN/QUANTIZE/REASSIGN/DELETE -> cleaned MIDI and reports