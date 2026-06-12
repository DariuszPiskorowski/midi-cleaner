# AGENTS.md

Rules for AI coding agents in `midi-cleaner` during Milestone 9:

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
- Do not add dependencies.
- Do not implement web UI.
- Do not implement new validation heuristics, pitch validation, rendering, ML, or DAW integration.
- Update and maintain tests with every behavior change.
