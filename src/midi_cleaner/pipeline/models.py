from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PipelineStageReport(BaseModel):
    name: str
    status: Literal["ok", "error"]
    output_files: list[str]
    warning_count: int
    warnings: list[str]


class PipelineReport(BaseModel):
    status: Literal["ok", "error"]
    input_midi: str
    input_wav: str
    source: str
    layer: str
    project_dir: str
    stages: list[PipelineStageReport]
    output_files: dict[str, str]
    warning_count: int
    warnings: list[str]
