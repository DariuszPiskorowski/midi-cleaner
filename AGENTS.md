# AGENTS.md

Rules for AI coding agents in `midi-cleaner` during Milestone 4:

- Keep the Python environment guard strict for Python 3.11.x.
- Run all commands through `uv run` when executing project tools.
- Milestone 2 parsing with `mido` is allowed.
- Milestone 3 analysis may use `numpy`, `scipy`, and `soundfile`.
- Milestone 4 validates MIDI events against audio features.
- Do not implement cleaner/export yet.
- Do not add heavy audio/ML dependencies yet.
- Do not implement pitch/harmonic validation yet.
- Do not create a UI yet.
- Do not assume RipX, Cubase, or Reaper APIs are available.
- Update and maintain tests with every behavior change.
