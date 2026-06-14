# midi-cleaner

Hermes MIDI Fidelity Engine project scaffold.

This repository cleans and validates MIDI extracted from Suno/RipX against original WAV stems. The current scope is **Milestone 14**: bounded iterative MIDI self-repair on top of Milestones 11/12/13A/13B.

AI pattern completion is now available as an additive feature:
- Hermes provides a rich `pattern_pack.json` and synchronized base MIDI context.
- OpenAI infers musically implied bass continuation/missing fragments without editing base MIDI.
- Output is a separate synchronized completion track: `midi/ai/bass_ai_completion.mid`.

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

Milestone 12 DSP analysis is now in place:
- A new additive DSP analyzer command `audio analyze-dsp` exports frame-level DSP evidence and report JSON.
- Existing lightweight analyzer `audio analyze-stem` remains unchanged and is still used by legacy flows.
- DSP backend selection supports `auto|librosa|scipy|basic` with fallback warnings when optional backends are unavailable.
- `process-stem` runs DSP analysis between WAV analysis and alignment, with strict (`--require-dsp-analysis`) and permissive modes.
- Bass refinement optionally consumes DSP features to improve attack/tail/retrigger decisions when DSP evidence is present.
- QA summary and HTML include DSP backend/classification metrics.

Milestone 13A activity repair is now in place:
- Additive post-refinement repair stage compares audio activity regions against MIDI activity regions.
- Detects and repairs obvious bass issues: missing gaps, overhang tails, split candidates, and tiny gaps.
- Uses DSP activity evidence when available; falls back to lightweight audio features when DSP data is absent.
- Keeps WAV/audio seconds canonical and avoids BPM/bar timing assumptions.
- Exports `analysis/repaired_refined_note_events.json`, `analysis/activity_repair_plan.json`, and `analysis/activity_repair_report.json`.
- Working export prefers repaired refined notes when activity repair is enabled.
- QA summary/CSV/HTML include activity-repair counts and per-note repair flags.

Milestone 13B sustain-aware repair guards are now in place:
- New additive pitch analysis command: `pitch bass-contour`.
- Process-stem can run pitch contour stage after DSP and before refinement/repair.
- Activity repair now applies conservative shorten guards for sustained low-band/harmonic energy, voiced pitch continuation, and legato transitions.
- Pitch contour is used as repair evidence, not treated as final truth.
- Overhang shortening is conservative and uncertainty is routed to `REVIEW_MANUAL`.

Future render-back work remains planned:
- Milestone 15 (13C): render-back audio comparison.

Milestone 14 iterative self-repair is now in place:
- Adds a bounded multi-pass repair controller after activity repair and before final working export.
- Each pass re-scores candidate MIDI activity against WAV/DSP/pitch evidence.
- Pass profiles support `balanced`, `sustain_legato`, `aggressive`, and final-pass `conservative` behavior.
- Stable-region freezing can lock good regions across passes to prevent unnecessary re-edits.
- Regression guard can reject score regressions and keep the best prior candidate.
- Pipeline outputs include per-iteration plans/notes, iterative report, and `analysis/final_repaired_note_events.json`.
- Working export uses final repaired notes and can export `working_iter1.mid`, `working_iter2.mid`, `working_iter3.mid`, and `working_best.mid` for REAPER comparison.

## Current Milestone

Milestone 14 currently implements:
- Python package scaffold
- Strict Python 3.11 guard
- Runtime/environment report via CLI doctor command
- MIDI candidate import from `.mid` using `mido`
- Hermes note-event JSON export and import report JSON
- WAV stem feature extraction using `numpy`, `scipy`, and `soundfile`
- Audio feature JSON export and audio analysis report JSON
- DSP-backed WAV feature extraction via `audio analyze-dsp`
- DSP feature JSON export, DSP analysis report JSON, and optional DSP debug CSV
- MIDI-vs-audio heuristic validation into note validation JSON and report JSON
- Non-destructive cleanup plan generation from note validation JSON
- Non-destructive review MIDI export grouped by cleanup action
- Conservative cleaned/review/rejected MIDI export from cleanup plan
- One-command process-stem pipeline that orchestrates existing stages
- Optional DSP stage in process-stem with fallback/strict controls
- Optional pitch contour stage in process-stem with fallback/strict controls
- Activity-repair stage between bass refinement and working export
- Repair plan/report artifacts and repaired refined note output
- Activity-repair CLI command and process-stem repair controls
- Bass pitch contour CLI command and contour/report artifacts
- Sustain-aware repair guards to protect long sustained/legato bass from aggressive shortening
- Working export preference for repaired refined notes
- Static QA artifacts: qa_summary.json, qa_notes.csv, qa_report.html
- Audio-time canonical note alignment into analysis/audio_aligned_note_events.json
- Audio alignment report in analysis/audio_alignment_report.json
- Validation uses aligned timing via audio_aligned_note_events.json when available
- Review/cleaned MIDI export using aligned_start_sec/aligned_end_sec when alignment data is provided
- Bass refinement stage output in analysis/refined_note_events.json
- Bass refinement report in analysis/bass_refinement_report.json
- Attack-aware onset refinement and false-retrigger merge suppression for bass
- Sustain-tail extension, minimum-duration extension, and optional monophonic overlap resolution
- Optional DSP-aware bass refinement behavior when DSP features are provided
- Practical working export in midi/working/working.mid and midi/working/rejected.mid
- Optional midi/working/diagnostic.mid for quick inspection
- QA report includes explicit Audio-Time Alignment / Sync metrics and timing sources
- QA report includes refinement summary metrics and per-note refinement columns
- QA report includes DSP backend/frame classification summary metrics
- QA report includes activity-repair summary fields and per-note repair columns
- QA report includes sustain/pitch/legato protection and shorten decision counters
- Bounded iterative repair loop with pass-specific profiles and deterministic stopping
- Iterative scoring with coverage/overhang/continuity/pitch metrics and error-region counting
- Stable note/region freezing and optional conservative final pass
- Iteration artifact exports and optional working MIDI variants for visual comparison
- OpenAI-powered `ai_pattern_completion` feature with JSON-validated bass completion export
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

## AI Pattern Completion Environment

Create a local `.env` from `.env.example`:

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

If `OPENAI_MODEL` is omitted, `gpt-4o-mini` is used by default.

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

## DSP WAV Stem Analysis (Additive)

```powershell
uv run midi-cleaner audio analyze-dsp --wav INPUT_WAV --layer bass --output OUTPUT_JSON --report REPORT_JSON --backend auto
```

Example:

```powershell
uv run midi-cleaner audio analyze-dsp --wav .\stem.wav --layer bass --output .\artifacts\audio_features_dsp.json --report .\artifacts\audio_analysis_dsp_report.json --backend auto --debug-csv .\artifacts\audio_features_dsp_debug.csv
```

## Bass Pitch Contour Analysis

```powershell
uv run midi-cleaner pitch bass-contour --wav INPUT_WAV --layer bass --output BASS_PITCH_CONTOUR_JSON --report BASS_PITCH_CONTOUR_REPORT_JSON --pitch-backend auto
```

Example:

```powershell
uv run midi-cleaner pitch bass-contour --wav .\stem.wav --layer bass --output .\artifacts\bass_pitch_contour.json --report .\artifacts\bass_pitch_contour_report.json --pitch-backend auto --pitch-min-hz 35 --pitch-max-hz 400 --pitch-confidence-threshold 0.60
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

Optional DSP feature input:

```powershell
uv run midi-cleaner refine bass --aligned-notes AUDIO_ALIGNED_NOTE_EVENTS_JSON --audio-features AUDIO_FEATURES_JSON --dsp-features AUDIO_FEATURES_DSP_JSON --validation NOTE_VALIDATION_JSON --output REFINED_NOTE_EVENTS_JSON --report REFINEMENT_REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner refine bass --aligned-notes .\artifacts\audio_aligned_note_events.json --audio-features .\artifacts\audio_features.json --validation .\artifacts\note_validation.json --output .\artifacts\refined_note_events.json --report .\artifacts\bass_refinement_report.json
```

## Audio/MIDI Activity Repair

```powershell
uv run midi-cleaner repair activity --refined-notes REFINED_NOTE_EVENTS_JSON --audio-features AUDIO_FEATURES_JSON --cleanup-plan CLEANUP_PLAN_JSON --output REPAIRED_REFINED_NOTE_EVENTS_JSON --plan ACTIVITY_REPAIR_PLAN_JSON --report ACTIVITY_REPAIR_REPORT_JSON
```

Optional DSP feature input:

```powershell
uv run midi-cleaner repair activity --refined-notes REFINED_NOTE_EVENTS_JSON --audio-features AUDIO_FEATURES_JSON --dsp-features AUDIO_FEATURES_DSP_JSON --cleanup-plan CLEANUP_PLAN_JSON --output REPAIRED_REFINED_NOTE_EVENTS_JSON --plan ACTIVITY_REPAIR_PLAN_JSON --report ACTIVITY_REPAIR_REPORT_JSON
```

Optional pitch contour input:

```powershell
uv run midi-cleaner repair activity --refined-notes REFINED_NOTE_EVENTS_JSON --audio-features AUDIO_FEATURES_JSON --pitch-contour BASS_PITCH_CONTOUR_JSON --cleanup-plan CLEANUP_PLAN_JSON --output REPAIRED_REFINED_NOTE_EVENTS_JSON --plan ACTIVITY_REPAIR_PLAN_JSON --report ACTIVITY_REPAIR_REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner repair activity --refined-notes .\artifacts\refined_note_events.json --audio-features .\artifacts\audio_features.json --cleanup-plan .\artifacts\cleanup_plan.json --output .\artifacts\repaired_refined_note_events.json --plan .\artifacts\activity_repair_plan.json --report .\artifacts\activity_repair_report.json
```

## Iterative MIDI Self-Repair

```powershell
uv run midi-cleaner repair iterative --refined-notes REFINED_NOTE_EVENTS_JSON --audio-features AUDIO_FEATURES_JSON --cleanup-plan CLEANUP_PLAN_JSON --output FINAL_REPAIRED_REFINED_NOTES_JSON --report ITERATIVE_REPAIR_REPORT_JSON
```

Optional DSP and pitch inputs:

```powershell
uv run midi-cleaner repair iterative --refined-notes REFINED_NOTE_EVENTS_JSON --audio-features AUDIO_FEATURES_JSON --cleanup-plan CLEANUP_PLAN_JSON --dsp-features AUDIO_FEATURES_DSP_JSON --pitch-contour BASS_PITCH_CONTOUR_JSON --output FINAL_REPAIRED_REFINED_NOTES_JSON --report ITERATIVE_REPAIR_REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner repair iterative --refined-notes .\artifacts\repaired_refined_note_events.json --audio-features .\artifacts\audio_features.json --cleanup-plan .\artifacts\cleanup_plan.json --dsp-features .\artifacts\audio_features_dsp.json --pitch-contour .\artifacts\bass_pitch_contour.json --output .\artifacts\final_repaired_note_events.json --report .\artifacts\iterative_repair_report.json
```

## AI Pattern Completion

```powershell
uv run midi-cleaner ai complete-pattern --project-dir PROJECT_DIR --layer bass
```

Common options:
- `--model` optional override (default `OPENAI_MODEL` or `gpt-4o-mini`)
- `--output-dir` optional output directory (default `midi/ai` under project)
- `--dry-run` build `pattern_pack.json` + `ai_prompt.txt` without API call
- `--max-completion-notes` default `64`
- `--temperature` default `0.2`
- `--keep-ai-json/--no-keep-ai-json` default keep enabled

Default artifacts:
- `analysis/ai_pattern_completion/pattern_pack.json`
- `analysis/ai_pattern_completion/ai_prompt.txt`
- `analysis/ai_pattern_completion/bass_ai_completion.json`
- `analysis/ai_pattern_completion/bass_ai_completion_report.json`
- `midi/ai/bass_ai_completion.mid`

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

Optional repair plan metadata:

```powershell
uv run midi-cleaner cleanup export-working-midi --notes NOTE_EVENTS_JSON --plan CLEANUP_PLAN_JSON --refined-notes REPAIRED_REFINED_NOTE_EVENTS_JSON --repair-plan ACTIVITY_REPAIR_PLAN_JSON --output-dir OUTPUT_DIR --report REPORT_JSON
```

Example:

```powershell
uv run midi-cleaner cleanup export-working-midi --notes .\artifacts\note_events.json --plan .\artifacts\cleanup_plan.json --refined-notes .\artifacts\refined_note_events.json --output-dir .\artifacts\working_midi --report .\artifacts\working_midi\working_export_report.json
```

## End-to-End Process Stem Pipeline

```powershell
uv run midi-cleaner pipeline process-stem --midi INPUT_MIDI --wav INPUT_WAV --source ripx --layer bass --project-dir PROJECT_DIR
```

DSP controls (defaults shown):
- `--enable-dsp-analysis/--no-enable-dsp-analysis` (default enabled)
- `--require-dsp-analysis/--no-require-dsp-analysis` (default permissive)
- `--dsp-backend auto|librosa|scipy|basic` (default `auto`)
- `--dsp-debug-csv/--no-dsp-debug-csv` (default enabled)

Pitch contour controls (defaults shown):
- `--enable-pitch-contour/--no-enable-pitch-contour` (default enabled for bass)
- `--require-pitch-contour/--no-require-pitch-contour` (default permissive)
- `--pitch-backend auto|librosa|basic` (default `auto`)
- `--pitch-min-hz` (default `35`)
- `--pitch-max-hz` (default `400`)
- `--pitch-confidence-threshold` (default `0.60`)

Activity repair controls (defaults shown):
- `--enable-activity-repair/--no-enable-activity-repair` (default enabled for bass)
- `--audio-active-threshold-ratio` (default `0.18`)
- `--audio-silence-hold-ms` (default `120`)
- `--missing-gap-min-ms` (default `80`)
- `--overhang-min-ms` (default `220`)
- `--split-min-note-duration-ms` (default `500`)
- `--close-gap-ms` (default `50`)
- `--insert-auto-confidence` (default `0.80`)
- `--split-auto-confidence` (default `0.75`)
- `--split-pitch-change-semitones` (default `0.75`)
- `--insert-from-pitch-contour-confidence` (default `0.75`)

Iterative repair controls (defaults shown):
- `--enable-iterative-repair/--no-enable-iterative-repair` (default enabled for bass)
- `--repair-iterations` (default `3`)
- `--repair-min-improvement` (default `0.005`)
- `--freeze-stable-notes/--no-freeze-stable-notes` (default enabled)
- `--conservative-final-pass/--no-conservative-final-pass` (default enabled)
- `--export-iteration-variants/--no-export-iteration-variants` (default enabled)

AI completion control:
- `--enable-ai-pattern-completion/--no-enable-ai-pattern-completion` (default disabled)

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

Suno/RipX stem WAV -> RipX MIDI candidate -> Hermes note-event JSON -> WAV comparison -> DSP analysis (additive) -> bass pitch contour (additive) -> audio-time alignment -> validation -> bass refinement -> activity repair (sustain-aware guards) -> iterative repair loop (pass1/pass2/pass3 with scoring and freeze guards) -> cleanup planning/exports -> working/rejected (uses final repaired notes) -> QA artifacts