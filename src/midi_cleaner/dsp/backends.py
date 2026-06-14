from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Literal

DspBackendName = Literal["auto", "librosa", "scipy", "basic"]


@dataclass(frozen=True)
class BackendResolution:
    backend_name: str
    backend_available: bool
    warnings: list[str]


def backend_is_available(name: str) -> bool:
    normalized = name.strip().lower()
    if normalized == "basic":
        return True
    if normalized in {"librosa", "scipy", "aubio", "essentia"}:
        return importlib.util.find_spec(normalized) is not None
    return False


def optional_backend_status() -> dict[str, bool]:
    return {
        "librosa": backend_is_available("librosa"),
        "scipy": backend_is_available("scipy"),
        "basic": backend_is_available("basic"),
        "aubio": backend_is_available("aubio"),
        "essentia": backend_is_available("essentia"),
    }


def resolve_backend(requested: str, allow_fallback: bool = True) -> BackendResolution:
    value = requested.strip().lower()
    warnings: list[str] = []

    if value not in {"auto", "librosa", "scipy", "basic"}:
        raise ValueError(f"Unsupported DSP backend: {requested}")

    if value == "auto":
        for candidate in ["librosa", "scipy", "basic"]:
            if backend_is_available(candidate):
                return BackendResolution(
                    backend_name=candidate,
                    backend_available=True,
                    warnings=warnings,
                )
        return BackendResolution(backend_name="basic", backend_available=True, warnings=warnings)

    if backend_is_available(value):
        return BackendResolution(backend_name=value, backend_available=True, warnings=warnings)

    if not allow_fallback:
        return BackendResolution(backend_name=value, backend_available=False, warnings=warnings)

    for fallback in ["scipy", "basic"]:
        if backend_is_available(fallback):
            warnings.append(
                f"Requested DSP backend '{value}' unavailable; using '{fallback}' instead."
            )
            return BackendResolution(
                backend_name=fallback,
                backend_available=True,
                warnings=warnings,
            )

    warnings.append(
        f"Requested DSP backend '{value}' unavailable and no fallback backend found."
    )
    return BackendResolution(backend_name=value, backend_available=False, warnings=warnings)
