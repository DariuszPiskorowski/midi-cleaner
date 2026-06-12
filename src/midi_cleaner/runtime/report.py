from __future__ import annotations

from pydantic import BaseModel, Field

from midi_cleaner.runtime.environment import gather_runtime_context


class OSInfo(BaseModel):
    platform: str
    system: str
    release: str
    machine: str


class PythonInfo(BaseModel):
    version: str
    executable: str
    required_major_minor: str
    matches_requirement: bool


class PythonEnvironmentInfo(BaseModel):
    VIRTUAL_ENV: str | None
    UV_PROJECT_ENVIRONMENT: str | None
    PYTHONPATH: str | None


class ToolInfo(BaseModel):
    name: str
    available: bool
    required: bool = Field(default=False)
    path: str | None
    version: str | None


class RuntimeReport(BaseModel):
    project_name: str
    package_version: str
    timestamp_utc: str
    os: OSInfo
    python: PythonInfo
    cwd: str
    python_environment: PythonEnvironmentInfo
    external_tools: list[ToolInfo]
    status: str
    problems: list[str]


def build_runtime_report(package_version: str) -> RuntimeReport:
    context = gather_runtime_context()
    tools = [ToolInfo.model_validate(item) for item in context["external_tools"]]

    problems: list[str] = []
    status = "ok"

    python_info = PythonInfo.model_validate(context["python"])
    if not python_info.matches_requirement:
        status = "error"
        problems.append(
            "Python version mismatch: required 3.11.x, "
            f"detected {python_info.version}."
        )

    missing_optional = [tool.name for tool in tools if not tool.available]
    if status != "error" and missing_optional:
        status = "warning"

    if missing_optional:
        problems.append(
            "Optional tools not found: " + ", ".join(sorted(missing_optional))
        )

    return RuntimeReport(
        project_name=str(context["project_name"]),
        package_version=package_version,
        timestamp_utc=str(context["timestamp_utc"]),
        os=OSInfo.model_validate(context["os"]),
        python=python_info,
        cwd=str(context["cwd"]),
        python_environment=PythonEnvironmentInfo.model_validate(
            context["python_environment"]
        ),
        external_tools=tools,
        status=status,
        problems=problems,
    )
