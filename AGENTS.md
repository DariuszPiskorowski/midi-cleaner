# AGENTS.md

Rules for AI coding agents in `midi-cleaner` during Milestone 5:

- Keep the Python environment guard strict for Python 3.11.x.
- Run all commands through `uv run` when executing project tools.
- Milestone 2 parsing with `mido` is allowed.
- Milestone 3 analysis may use `numpy`, `scipy`, and `soundfile`.
- Milestone 4 validates MIDI events against audio features.
- Milestone 5 creates a non-destructive cleanup plan.
- Do not implement MIDI export/rewriting yet.
- Do not delete notes.
- Do not add dependencies.
- Do not implement UI or DAW integration.
- Update and maintain tests with every behavior change.
