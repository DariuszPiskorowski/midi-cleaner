from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _script_text(file_name: str) -> str:
    return (SCRIPTS_DIR / file_name).read_text(encoding="utf-8")


def test_windows_launcher_files_exist() -> None:
    assert (SCRIPTS_DIR / "start-hermes.cmd").exists()
    assert (SCRIPTS_DIR / "start-hermes.ps1").exists()
    assert (SCRIPTS_DIR / "start-hermes-midi-editor.cmd").exists()
    assert (SCRIPTS_DIR / "start-hermes-midi-editor.ps1").exists()
    assert (SCRIPTS_DIR / "install-desktop-shortcut.ps1").exists()
    assert (SCRIPTS_DIR / "README.md").exists()


def test_main_cmd_launcher_has_repo_relative_and_uv_guard() -> None:
    script = _script_text("start-hermes.cmd")

    assert 'set "SCRIPT_DIR=%~dp0"' in script
    assert 'for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"' in script
    assert 'where uv >nul 2>nul' in script
    assert "uv is not installed or not on PATH." in script
    assert "Starting Hermes main app..." in script
    assert (
        'powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "'
        '%SCRIPT_DIR%start-hermes.ps1"'
    ) in script
    assert "midi split-editor" not in script
    assert "pause" in script


def test_main_powershell_launcher_starts_full_hermes_app() -> None:
    script = _script_text("start-hermes.ps1")

    assert "Starting Hermes main app..." in script
    assert "uv is not installed or not on PATH." in script
    assert "& uv run midi-cleaner gui" in script
    assert "midi split-editor" not in script
    assert "/api/session" not in script
    assert "Start-Process" not in script


def test_shortcut_installer_targets_full_launcher_cmd() -> None:
    script = _script_text("install-desktop-shortcut.ps1")

    assert '[string]$ShortcutName = "Hermes"' in script
    assert '$launcherPath = Join-Path $scriptDir "start-hermes.cmd"' in script
    assert '$desktopPath = [Environment]::GetFolderPath("Desktop")' in script
    assert '$shortcutPath = Join-Path $desktopPath ($ShortcutName + ".lnk")' in script
    assert "$wshShell = New-Object -ComObject WScript.Shell" in script
    assert "$shortcut.TargetPath = $launcherPath" in script
    assert "$shortcut.WorkingDirectory = $repoRoot" in script
    assert "start-hermes-midi-editor.cmd" not in script


def test_midi_editor_launcher_remains_optional_and_separate() -> None:
    shortcut_script = _script_text("install-desktop-shortcut.ps1")
    midi_editor_script = _script_text("start-hermes-midi-editor.ps1")

    assert "start-hermes-midi-editor.cmd" not in shortcut_script
    assert "& uv run midi-cleaner midi split-editor --host $BindHost --port $Port" in midi_editor_script


def test_scripts_readme_mentions_full_app_default_and_optional_midi_editor() -> None:
    readme = _script_text("README.md")

    assert "Start Hermes main app" in readme
    assert "scripts\\start-hermes.cmd" in readme
    assert "Hermes" in readme
    assert "optional" in readme.lower()
    assert "start only the browser MIDI Editor directly" in readme
    assert "scripts\\start-hermes-midi-editor.cmd" in readme
    assert "scripts\\install-desktop-shortcut.ps1" in readme
    assert "powershell -ExecutionPolicy Bypass -File scripts\\install-desktop-shortcut.ps1" in readme


def test_no_lnk_files_are_committed_in_repository_tree() -> None:
    lnk_files = list(REPO_ROOT.rglob("*.lnk"))
    assert lnk_files == []