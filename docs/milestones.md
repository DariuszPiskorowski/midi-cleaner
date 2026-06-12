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

## Milestone 5 - Non-Destructive MIDI Cleanup Plan
- Add CLI command `midi-cleaner cleanup plan`.
- Convert note validation results into a machine-readable cleanup plan.
- Keep plan non-destructive: no MIDI rewriting, no note deletion.
- Allow optional `DELETE_CANDIDATE` planning only as metadata.
- Export cleanup plan JSON and cleanup plan report JSON.

## Milestone 6 - Non-Destructive Review MIDI Export
- Add CLI command `midi-cleaner cleanup export-review-midi`.
- Export grouped review MIDI files: KEEP, REVIEW, MUTE, optional DELETE_CANDIDATE.
- Preserve note pitch, velocity, and timing from note events.
- Keep export non-destructive: no final cleaned MIDI and no note deletion.
- Export machine-readable MIDI export report JSON.

## Future Milestones (Planned)
- Classify actions: KEEP, MUTE, MERGE, SHORTEN, QUANTIZE, REASSIGN, DELETE.
- Export cleaned MIDI and machine-readable QA reports.
