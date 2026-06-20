"""Simple Hermes desktop GUI launcher."""

from midi_cleaner.gui.controller import (
    ACTION_MAKE_MIDI_FROM_WAV,
    ACTION_SET_BPM,
    ACTION_SYNCHRONIZE_MIDI_WITH_WAV,
    ROLE_BASS,
    ROLE_DRUMS,
    ROLE_GUITAR,
    ROLE_OTHER,
    ROLE_SYNTH,
    HermesActionRequest,
    HermesActionResult,
    HermesGuiController,
)


def launch_hermes_gui(controller: HermesGuiController | None = None) -> None:
    from midi_cleaner.gui.panel import launch_hermes_gui as _launch

    _launch(controller=controller)

__all__ = [
    "ACTION_MAKE_MIDI_FROM_WAV",
    "ACTION_SET_BPM",
    "ACTION_SYNCHRONIZE_MIDI_WITH_WAV",
    "ROLE_BASS",
    "ROLE_DRUMS",
    "ROLE_GUITAR",
    "ROLE_OTHER",
    "ROLE_SYNTH",
    "HermesActionRequest",
    "HermesActionResult",
    "HermesGuiController",
    "launch_hermes_gui",
]
