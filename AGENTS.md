# AGENTS.md

Rules for AI coding agents in `midi-cleaner` during Milestone 11:

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
- Synchronization is critical.
- Global offset search must run before local per-note alignment.
- Do not rely on BPM/bar calculations for timing alignment.
- Keep export ticks_per_beat defaulted from source note_events unless explicitly overridden.
- Do not commit generated project outputs under `projects/*`.
- Do not add dependencies.
- Do not implement web UI.
- Do not implement new validation heuristics, pitch validation, rendering, ML, or DAW integration.
- Update and maintain tests with every behavior change.
