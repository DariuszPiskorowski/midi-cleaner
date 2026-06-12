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

## Milestone 3 - Lightweight WAV Stem Analyzer
- Add lightweight WAV analysis with `numpy`, `scipy`, and `soundfile`.
- Add CLI command `midi-cleaner audio analyze-stem`.
- Extract per-frame and global deterministic audio features.
- Export feature document JSON and analysis report JSON.
- Keep MIDI-vs-WAV validation and cleaner actions out of this milestone.

## Milestone 4 - First MIDI-vs-Audio Validation Pass
- Add CLI command `midi-cleaner validate midi-vs-audio`.
- Validate imported note events against analyzed audio frame features.
- Compute deterministic per-note confidence and recommended action.
- Export note validation JSON and validation report JSON.
- Keep cleaning, rewriting, and pitch/harmonic validation out of this milestone.

## Future Milestones (Planned)
- Classify actions: KEEP, MUTE, MERGE, SHORTEN, QUANTIZE, REASSIGN, DELETE.
- Export cleaned MIDI and machine-readable QA reports.
