from packaging.version import Version

from midi_cleaner.runtime.environment import (
    gather_runtime_context,
    python_version_matches,
    required_python_string,
)


def test_required_python_is_311() -> None:
    assert required_python_string() == "3.11"


def test_python_version_check_passes_for_311() -> None:
    assert python_version_matches(Version("3.11.0"))
    assert python_version_matches(Version("3.11.9"))


def test_python_version_check_fails_for_310_or_312() -> None:
    assert not python_version_matches(Version("3.10.14"))
    assert not python_version_matches(Version("3.12.1"))


def test_optional_missing_tools_do_not_set_error(monkeypatch) -> None:
    monkeypatch.setattr("midi_cleaner.runtime.environment.shutil.which", lambda _name: None)
    report = gather_runtime_context()

    external_tools = report["external_tools"]
    assert external_tools
    assert all(not tool["available"] for tool in external_tools)
