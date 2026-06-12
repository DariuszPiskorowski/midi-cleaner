# AGENTS.md

Rules for AI coding agents in `midi-cleaner` during Milestone 2:

- Keep the Python environment guard strict for Python 3.11.x.
- Run all commands through `uv run` when executing project tools.
- Milestone 2 may use `mido` for MIDI parsing.
- Do not add heavy audio, ML, or rendering dependencies yet.
- Do not implement MIDI cleaning actions yet.
- Do not implement WAV analysis yet and do not assume WAV files exist.
- Do not create a UI yet.
- Do not assume RipX, Cubase, or Reaper APIs are available.
- Update and maintain tests with every behavior change.
