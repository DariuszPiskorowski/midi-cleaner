"""MIDI split editor backend and preview helpers."""

from midi_cleaner.midi_split.exporter import (
    MidiSplitExportError,
    export_split_multitrack_midi,
    export_split_separate_midi_files,
)
from midi_cleaner.midi_split.html_preview import generate_piano_roll_preview
from midi_cleaner.midi_split.models import MidiSplitSession, SplitNote, SplitTrack
from midi_cleaner.midi_split.service import (
    DEFAULT_MAX_TRACKS,
    MidiSplitSessionError,
    add_empty_track,
    create_split_session,
    load_session,
    merge_tracks,
    move_notes_to_track,
    save_session,
)

__all__ = [
    "DEFAULT_MAX_TRACKS",
    "MidiSplitExportError",
    "MidiSplitSession",
    "MidiSplitSessionError",
    "SplitNote",
    "SplitTrack",
    "add_empty_track",
    "create_split_session",
    "export_split_multitrack_midi",
    "export_split_separate_midi_files",
    "generate_piano_roll_preview",
    "load_session",
    "merge_tracks",
    "move_notes_to_track",
    "save_session",
]
