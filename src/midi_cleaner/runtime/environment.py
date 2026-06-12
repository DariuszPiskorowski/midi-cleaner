from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from packaging.version import Version

REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 11
PROJECT_NAME = "midi-cleaner"


@dataclass(frozen=True)
class ToolStatus:
    name: str
    available: bool
    required: bool
    path: str | None
    version: str | None


def required_python_string() -> str:
    return f"{REQUIRED_PYTHON_MAJOR}.{REQUIRED_PYTHON_MINOR}"


def python_version_matches(version: Version) -> bool:
    return (
        version.major == REQUIRED_PYTHON_MAJOR
        and version.minor == REQUIRED_PYTHON_MINOR
    )


def current_python_version() -> Version:
    raw = platform.python_version()
    return Version(raw)


def detect_tool(tool_name: str, version_args: tuple[str, ...] = ("--version",)) -> ToolStatus:
    tool_path = shutil.which(tool_name)
    if not tool_path:
        return ToolStatus(
            name=tool_name,
            available=False,
            required=False,
            path=None,
            version=None,
        )

    version_text: str | None = None
    try:
        completed = subprocess.run(
            [tool_name, *version_args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (completed.stdout or completed.stderr).strip()
        version_text = output.splitlines()[0] if output else None
    except (OSError, subprocess.SubprocessError):
        version_text = None

    return ToolStatus(
        name=tool_name,
        available=True,
        required=False,
        path=tool_path,
        version=version_text,
    )


def gather_optional_tools() -> list[ToolStatus]:
    return [
        detect_tool("git"),
        detect_tool("ffmpeg"),
        detect_tool("node"),
        detect_tool("nvidia-smi"),
    ]


def gather_runtime_context() -> dict[str, object]:
    py_version = current_python_version()
    tools = gather_optional_tools()

    return {
        "project_name": PROJECT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "version": str(py_version),
            "executable": sys.executable,
            "required_major_minor": required_python_string(),
            "matches_requirement": python_version_matches(py_version),
        },
        "cwd": str(Path.cwd()),
        "python_environment": {
            "VIRTUAL_ENV": os.getenv("VIRTUAL_ENV"),
            "UV_PROJECT_ENVIRONMENT": os.getenv("UV_PROJECT_ENVIRONMENT"),
            "PYTHONPATH": os.getenv("PYTHONPATH"),
        },
        "external_tools": [
            {
                "name": tool.name,
                "available": tool.available,
                "required": tool.required,
                "path": tool.path,
                "version": tool.version,
            }
            for tool in tools
        ],
    }
