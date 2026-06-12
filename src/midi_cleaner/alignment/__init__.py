"""Audio-time canonical note alignment."""

from midi_cleaner.alignment.audio_time import (
    AudioTimeAlignmentError,
    AudioTimeAlignmentParameters,
    align_notes_to_audio_time,
)
from midi_cleaner.alignment.models import (
    AudioAlignedNoteDocument,
    AudioAlignedNoteEvent,
    AudioAlignmentReport,
)

__all__ = [
    "AudioAlignedNoteDocument",
    "AudioAlignedNoteEvent",
    "AudioAlignmentReport",
    "AudioTimeAlignmentError",
    "AudioTimeAlignmentParameters",
    "align_notes_to_audio_time",
]
