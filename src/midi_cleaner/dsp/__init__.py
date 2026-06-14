"""DSP-backed audio feature analysis with backend fallbacks."""

from midi_cleaner.dsp.analyzer import DspAnalysisError, analyze_dsp_stem
from midi_cleaner.dsp.backends import BackendResolution, resolve_backend
from midi_cleaner.dsp.models import DspAnalysisReport, DspAudioFeatureDocument, DspAudioFrame

__all__ = [
    "BackendResolution",
    "DspAnalysisError",
    "DspAnalysisReport",
    "DspAudioFeatureDocument",
    "DspAudioFrame",
    "analyze_dsp_stem",
    "resolve_backend",
]
