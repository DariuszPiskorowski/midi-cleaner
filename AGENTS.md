# AGENTS.md

Rules for AI coding agents in `midi-cleaner` during Milestone 13A:

- Keep the Python environment guard strict for Python 3.11.x.
- Run all commands through `uv run` when executing project tools.
- Milestone 2 parsing with `mido` is allowed.
- Milestone 3 analysis may use `numpy`, `scipy`, and `soundfile`.
- Milestone 4 validates MIDI events against audio features.
- Milestone 5 creates a non-destructive cleanup plan.
- Milestone 6 exports non-destructive review MIDI files.
- Milestone 7 exports conservative cleaned/review/rejected MIDI files.
- Milestone 8 orchestrates existing stages.
- Milestone 9 generates static QA artifacts from existing pipeline outputs.
- Milestone 10 aligns note timing to canonical WAV/audio seconds.
- Milestone 10.1 ensures validation and cleanup consume aligned timing when available.
- Milestone 11 refines bass note quality after audio-time alignment.
- Milestone 11 must support attack-aware start adjustment, false-retrigger suppression, and sustain-tail extension.
- Milestone 11 exports practical `working.mid` plus `rejected.mid` (optional diagnostic output).
- Milestone 12 adds DSP-backed audio feature analysis as an additional stage; do not remove or replace the existing lightweight analyzer.
- Milestone 12 must support backend selection (`auto|librosa|scipy|basic`) with deterministic fallback behavior.
- Milestone 12 must allow strict mode (`require_dsp_analysis`) and permissive mode (warning + continue when DSP fails).
- Milestone 12 refinement may use DSP evidence when available but must preserve Milestone 11 behavior when DSP data is absent.
- Milestone 12 QA outputs must include DSP backend and frame/classification summary metrics.
- Milestone 13A adds audio/MIDI activity repair as an additional stage after bass refinement.
- Milestone 13A must preserve Milestone 11 and 12 behavior when repair is disabled or inapplicable.
- Milestone 13A may use DSP evidence when available but must fall back deterministically to lightweight audio features.
- Milestone 13A must support confidence-gated repair actions and route low-confidence insert/split cases to review-manual.
- Milestone 13A outputs must include repaired refined notes, activity repair plan/report, and QA repair summary metrics.
- Synchronization is critical.
- Global offset search must run before local per-note alignment.
- Do not rely on BPM/bar calculations for timing alignment.
- Keep export ticks_per_beat defaulted from source note_events unless explicitly overridden.
- Do not commit generated project outputs under `projects/*`.
- Do not add dependencies.
- Do not implement web UI.
- Do not implement new validation heuristics, pitch validation, rendering, ML, or DAW integration.
- Update and maintain tests with every behavior change.
