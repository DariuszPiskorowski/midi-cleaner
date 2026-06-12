# Milestones

## Milestone 1 - Environment Guard and Project Skeleton
- Initialize Python package structure.
- Pin runtime to Python 3.11.
- Add `uv` workflow (`uv sync`, `uv run ...`).
- Implement `midi-cleaner doctor` runtime report.
- Validate strict Python 3.11 guard behavior.
- Keep all external tools optional.

## Future Milestones (Planned)
- Parse candidate MIDI events extracted from RipX workflows.
- Build Hermes note-event JSON representation.
- Compare note-event candidates against WAV stems.
- Classify actions: KEEP, MUTE, MERGE, SHORTEN, QUANTIZE, REASSIGN, DELETE.
- Export cleaned MIDI and machine-readable QA reports.
