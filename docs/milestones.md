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

## Milestone 7 - Conservative Cleaned MIDI Export
- Add CLI command `midi-cleaner cleanup export-cleaned-midi`.
- Export grouped files: `cleaned.mid`, `review.mid`, `rejected.mid`.
- Keep default conservative behavior: only KEEP goes to cleaned by default.
- Keep export non-destructive and deterministic.
- Export machine-readable cleaned export report JSON.

## Milestone 8 - One-Command Process Stem Pipeline
- Add CLI command `midi-cleaner pipeline process-stem`.
- Orchestrate import, analysis, validation, planning, and exports end-to-end.
- Produce a structured project directory with stage outputs.
- Aggregate stage warnings and statuses into `pipeline_report.json`.
- Keep orchestration non-destructive and call internal functions directly.

## Milestone 9 - Static QA Report Generation
- Add CLI command `midi-cleaner pipeline qa-report`.
- Generate static QA artifacts from completed pipeline outputs.
- Export `qa_summary.json`, optional `qa_notes.csv`, optional `qa_report.html`.
- Keep reporting deterministic and dependency-light using Python standard library.
- Do not add new heuristics; summarize existing stage outputs.

## Milestone 10 - Audio-Time Canonical Note Alignment
- Add CLI command `midi-cleaner validate align-audio-time`.
- Produce `analysis/audio_aligned_note_events.json` using WAV/audio seconds as canonical time.
- Produce `analysis/audio_alignment_report.json` with alignment quality metrics.
- Keep synchronization independent from BPM/bar assumptions.

## Milestone 10.1 - Alignment Integration Hardening
- Fix Milestone 10 integration so validation uses aligned timing when alignment data exists.
- Ensure cleanup review/cleaned exports use aligned timing when alignment data exists.
- Use coarse global offset search before local per-note snapping.
- Keep synchronization anchored to WAV/audio seconds; do not rely on BPM/bar calculations.
- Default export ticks-per-beat to source note_events ticks-per-beat unless user override is provided.
- Expand QA artifacts to show Audio-Time Alignment / Sync timing sources and precision metrics.
- Keep generated `projects/*` outputs ignored and out of commits.

## Milestone 11 - Bass MIDI Quality Refinement
- Add `midi-cleaner refine bass` to refine note start/end timing from aligned audio seconds.
- Implement attack-aware start adjustment so note-on can lead the local energy peak when audio attack ramps in.
- Suppress false retriggers by merging near-pitch notes when no real silence exists between them.
- Extend note tails through audible sustain/decay while capping maximum extension.
- Enforce minimum note duration for short energetic notes and close tiny same-pitch gaps.
- Apply optional monophonic overlap cleanup for bass layers.
- Export practical `midi/working/working.mid` + `midi/working/rejected.mid` with optional `diagnostic.mid`.
- Keep legacy cleaned/review/rejected exports for backward compatibility.
- Expand QA artifacts with refinement and working-export summary fields and per-note refinement columns.

## Milestone 12 - DSP-Backed Audio Feature Analyzer
- Add `midi-cleaner audio analyze-dsp` to generate DSP-oriented per-frame features and report JSON.
- Keep Milestone 3 lightweight analyzer intact; DSP analysis is additive and does not replace existing outputs.
- Support backend routing and fallback for `auto|librosa|scipy|basic` with deterministic warning behavior.
- Add optional debug CSV export for DSP frame evidence.
- Insert DSP stage into `pipeline process-stem` between lightweight WAV analysis and audio-time alignment.
- Add pipeline controls: enable/disable DSP stage, require DSP success, backend selection, debug CSV toggle.
- Allow bass refinement to consume DSP features when available for attack placement, retrigger suppression, and tail extension decisions.
- Keep refinement behavior backward compatible when DSP features are unavailable.
- Expand QA artifacts with DSP backend/frame-classification summary fields.

## Future Milestones (Planned)
- Classify actions: KEEP, MUTE, MERGE, SHORTEN, QUANTIZE, REASSIGN, DELETE.
- Export cleaned MIDI and machine-readable QA reports.
