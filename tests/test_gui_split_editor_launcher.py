from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from midi_cleaner.gui.split_editor_launcher import SplitEditorLauncher


class _FakeThread:
    def __init__(self, alive: bool = True) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class _FakeServerHandle:
    def __init__(self, url: str) -> None:
        self.url = url
        self.thread = _FakeThread(alive=True)
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


def test_open_split_editor_without_midi_starts_server_and_opens_browser(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _ = tmp_path
    launcher = SplitEditorLauncher()

    started: dict[str, object] = {}
    browser_urls: list[str] = []

    monkeypatch.setattr(
        launcher,
        "_probe_split_editor",
        lambda _url: SimpleNamespace(responding=False, is_split_editor=False),
    )
    monkeypatch.setattr(launcher, "_is_port_available", lambda _host, _port: True)
    monkeypatch.setattr(launcher, "_wait_until_split_editor_ready", lambda **_kwargs: None)
    monkeypatch.setattr(
        "midi_cleaner.gui.split_editor_launcher.start_split_editor_server",
        lambda *, input_midi, host, port: (
            started.update({"input_midi": input_midi, "host": host, "port": port})
            or _FakeServerHandle(f"http://{host}:{port}/")
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_open_browser",
        lambda url: browser_urls.append(url) or True,
    )

    result = launcher.open_split_editor(midi_file=None, host="127.0.0.1", port=8765)

    assert result.success is True
    assert result.url == "http://127.0.0.1:8765/"
    assert result.started_new_server is True
    assert result.reused_existing_server is False
    assert started["input_midi"] is None
    assert browser_urls == ["http://127.0.0.1:8765/"]


def test_open_split_editor_with_selected_midi_passes_input_to_new_server(
    tmp_path: Path,
    monkeypatch,
) -> None:
    midi_path = tmp_path / "selected.mid"
    midi_path.write_bytes(b"midi")

    launcher = SplitEditorLauncher()
    started: dict[str, object] = {}

    monkeypatch.setattr(
        launcher,
        "_probe_split_editor",
        lambda _url: SimpleNamespace(responding=False, is_split_editor=False),
    )
    monkeypatch.setattr(launcher, "_is_port_available", lambda _host, _port: True)
    monkeypatch.setattr(launcher, "_wait_until_split_editor_ready", lambda **_kwargs: None)
    monkeypatch.setattr(
        "midi_cleaner.gui.split_editor_launcher.start_split_editor_server",
        lambda *, input_midi, host, port: (
            started.update({"input_midi": input_midi, "host": host, "port": port})
            or _FakeServerHandle(f"http://{host}:{port}/")
        ),
    )
    monkeypatch.setattr(launcher, "_open_browser", lambda _url: True)

    result = launcher.open_split_editor(midi_file=midi_path, host="127.0.0.1", port=8765)

    assert result.success is True
    assert result.started_new_server is True
    assert started["input_midi"] == midi_path


def test_open_split_editor_reuses_existing_server_and_imports_midi(
    tmp_path: Path,
    monkeypatch,
) -> None:
    midi_path = tmp_path / "selected.mid"
    midi_path.write_bytes(b"midi")

    launcher = SplitEditorLauncher()
    imported: dict[str, object] = {}
    browser_urls: list[str] = []

    monkeypatch.setattr(
        launcher,
        "_probe_split_editor",
        lambda _url: SimpleNamespace(responding=True, is_split_editor=True),
    )
    monkeypatch.setattr(
        launcher,
        "_import_midi_into_server",
        lambda *, url, midi_file: imported.update({"url": url, "midi_file": midi_file}) or None,
    )
    monkeypatch.setattr(
        launcher,
        "_open_browser",
        lambda url: browser_urls.append(url) or True,
    )
    monkeypatch.setattr(
        "midi_cleaner.gui.split_editor_launcher.start_split_editor_server",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("server should not start")),
    )

    result = launcher.open_split_editor(midi_file=midi_path, host="127.0.0.1", port=8765)

    assert result.success is True
    assert result.reused_existing_server is True
    assert result.started_new_server is False
    assert imported["url"] == "http://127.0.0.1:8765/"
    assert imported["midi_file"] == midi_path
    assert browser_urls == ["http://127.0.0.1:8765/"]


def test_open_split_editor_fails_when_port_serves_other_http_service(monkeypatch) -> None:
    launcher = SplitEditorLauncher()

    monkeypatch.setattr(
        launcher,
        "_probe_split_editor",
        lambda _url: SimpleNamespace(responding=True, is_split_editor=False),
    )
    browser_urls: list[str] = []
    monkeypatch.setattr(launcher, "_open_browser", lambda url: browser_urls.append(url) or True)

    result = launcher.open_split_editor(midi_file=None, host="127.0.0.1", port=8765)

    assert result.success is False
    assert "another HTTP service" in result.message
    assert browser_urls == []


def test_open_split_editor_missing_midi_path_opens_without_preload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing_midi = tmp_path / "missing.mid"

    launcher = SplitEditorLauncher()
    started: dict[str, object] = {}

    monkeypatch.setattr(
        launcher,
        "_probe_split_editor",
        lambda _url: SimpleNamespace(responding=False, is_split_editor=False),
    )
    monkeypatch.setattr(launcher, "_is_port_available", lambda _host, _port: True)
    monkeypatch.setattr(launcher, "_wait_until_split_editor_ready", lambda **_kwargs: None)
    monkeypatch.setattr(
        "midi_cleaner.gui.split_editor_launcher.start_split_editor_server",
        lambda *, input_midi, host, port: (
            started.update({"input_midi": input_midi, "host": host, "port": port})
            or _FakeServerHandle(f"http://{host}:{port}/")
        ),
    )
    monkeypatch.setattr(launcher, "_open_browser", lambda _url: True)

    result = launcher.open_split_editor(midi_file=missing_midi, host="127.0.0.1", port=8765)

    assert result.success is True
    assert result.started_new_server is True
    assert started["input_midi"] is None
    assert "does not exist" in result.message
