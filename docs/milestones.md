# Milestones

## Milestone 1 - Environment Guard and Project Skeleton
- Initialize Python package structure.
- Pin runtime to Python 3.11.
- Add `uv` workflow (`uv sync`, `uv run ...`).
- Implement `midi-cleaner doctor` runtime report.
- Validate strict Python 3.11 guard behavior.
- Keep all external tools optional.

## Milestone 2 - MIDI Candidate Import to Hermes Note Events
- Add lightweight MIDI parsing with `mido`.
- Add CLI command `midi-cleaner midi import-candidate`.
- Import candidate notes from `.mid` into Hermes note-event JSON.
- Preserve factual notes only: no quantize, no merge, no delete, no cleanup actions.
- Export import report JSON with note/track/tempo/warning counts.
- Keep WAV analysis and cleaner logic out of this milestone.

## Future Milestones (Planned)
- Compare note-event candidates against WAV stems.
- Classify actions: KEEP, MUTE, MERGE, SHORTEN, QUANTIZE, REASSIGN, DELETE.
- Export cleaned MIDI and machine-readable QA reports.
