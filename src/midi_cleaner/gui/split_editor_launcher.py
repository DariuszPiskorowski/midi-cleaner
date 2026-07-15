from __future__ import annotations

import json
import socket
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from midi_cleaner.midi_split.server import (
    MidiSplitEditorServerError,
    MidiSplitEditorServerHandle,
    start_split_editor_server,
)


@dataclass(frozen=True)
class SplitEditorLaunchResult:
    success: bool
    url: str
    message: str
    reused_existing_server: bool
    started_new_server: bool


@dataclass(frozen=True)
class _ServerProbeResult:
    responding: bool
    is_split_editor: bool


@dataclass
class _ManagedServerState:
    host: str
    port: int
    handle: MidiSplitEditorServerHandle


class SplitEditorLauncher:
    def __init__(self, request_timeout_sec: float = 1.5) -> None:
        self._request_timeout_sec = max(0.1, float(request_timeout_sec))
        self._lock = threading.Lock()
        self._managed_server: _ManagedServerState | None = None

    def open_split_editor(
        self,
        midi_file: Path | None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> SplitEditorLaunchResult:
        normalized_host = host.strip()
        if not normalized_host:
            return SplitEditorLaunchResult(
                success=False,
                url="",
                message="Split editor host cannot be empty.",
                reused_existing_server=False,
                started_new_server=False,
            )

        if port < 0 or port > 65535:
            return SplitEditorLaunchResult(
                success=False,
                url="",
                message="Split editor port must be in range 0..65535.",
                reused_existing_server=False,
                started_new_server=False,
            )

        requested_url = self._build_url(normalized_host, port)
        midi_path, midi_warning = self._normalize_midi_file(midi_file)

        with self._lock:
            url = requested_url
            probe = self._probe_split_editor(url)
            reused_existing_server = False
            started_new_server = False

            if probe.is_split_editor:
                reused_existing_server = True
            elif probe.responding:
                return SplitEditorLaunchResult(
                    success=False,
                    url=url,
                    message=(
                        f"Port {port} is already used by another HTTP service. "
                        "Stop that service or run split editor on another port."
                    ),
                    reused_existing_server=False,
                    started_new_server=False,
                )
            else:
                if port != 0 and not self._is_port_available(normalized_host, port):
                    return SplitEditorLaunchResult(
                        success=False,
                        url=url,
                        message=(
                            f"Port {port} is busy and is not responding as Hermes MIDI Editor. "
                            "Stop the conflicting process and try again."
                        ),
                        reused_existing_server=False,
                        started_new_server=False,
                    )

                managed = self._managed_server
                if managed is not None and managed.host == normalized_host and managed.port == port:
                    if managed.handle.thread.is_alive():
                        wait_error = self._wait_until_split_editor_ready(
                            url=managed.handle.url,
                            handle=managed.handle,
                        )
                        if wait_error is None:
                            url = managed.handle.url
                            reused_existing_server = True
                        else:
                            self._managed_server = None
                    else:
                        self._managed_server = None

                if not reused_existing_server:
                    try:
                        handle = start_split_editor_server(
                            input_midi=midi_path,
                            host=normalized_host,
                            port=port,
                        )
                    except (MidiSplitEditorServerError, OSError) as exc:
                        return SplitEditorLaunchResult(
                            success=False,
                            url=url,
                            message=f"Failed to start MIDI Editor server: {exc}",
                            reused_existing_server=False,
                            started_new_server=False,
                        )

                    self._managed_server = _ManagedServerState(
                        host=normalized_host,
                        port=port,
                        handle=handle,
                    )

                    wait_error = self._wait_until_split_editor_ready(url=handle.url, handle=handle)
                    if wait_error is not None:
                        self._managed_server = None
                        handle.stop()
                        return SplitEditorLaunchResult(
                            success=False,
                            url=handle.url,
                            message=wait_error,
                            reused_existing_server=False,
                            started_new_server=False,
                        )

                    url = handle.url
                    started_new_server = True

            preload_message = ""
            if midi_path is not None:
                if reused_existing_server:
                    import_error = self._import_midi_into_server(url=url, midi_file=midi_path)
                    if import_error is None:
                        preload_message = f"Loaded MIDI into editor: {midi_path.name}."
                    else:
                        preload_message = (
                            f"Could not preload MIDI into running editor ({midi_path.name}): {import_error}"
                        )
                elif started_new_server:
                    preload_message = f"Loaded MIDI into editor: {midi_path.name}."

        browser_opened = self._open_browser(url)

        message_parts = [f"MIDI Editor opened at {url}"]
        if reused_existing_server:
            message_parts.append("Reused existing MIDI Editor server.")
        if started_new_server:
            message_parts.append("Started new MIDI Editor server.")
        if preload_message:
            message_parts.append(preload_message)
        if midi_warning:
            message_parts.append(midi_warning)
        if not browser_opened:
            message_parts.append("Could not confirm browser launch; open the URL manually.")

        return SplitEditorLaunchResult(
            success=True,
            url=url,
            message=" ".join(message_parts),
            reused_existing_server=reused_existing_server,
            started_new_server=started_new_server,
        )

    @staticmethod
    def _build_url(host: str, port: int) -> str:
        return f"http://{host}:{port}/"

    @staticmethod
    def _normalize_midi_file(midi_file: Path | None) -> tuple[Path | None, str]:
        if midi_file is None:
            return None, ""

        candidate = Path(midi_file)
        if candidate.exists() and candidate.is_file():
            return candidate, ""

        return (
            None,
            (
                f"MIDI file does not exist: {candidate}. "
                "Opened editor without preloaded MIDI; use Import MIDI inside the editor."
            ),
        )

    def _open_browser(self, url: str) -> bool:
        try:
            return bool(webbrowser.open_new_tab(url))
        except Exception:
            return False

    def _probe_split_editor(self, base_url: str) -> _ServerProbeResult:
        endpoint = urllib.parse.urljoin(base_url, "api/session")
        request = Request(endpoint, method="GET")

        try:
            with urlopen(request, timeout=self._request_timeout_sec) as response:
                status = int(response.status)
                payload_bytes = response.read()
        except HTTPError:
            return _ServerProbeResult(responding=True, is_split_editor=False)
        except (URLError, TimeoutError, OSError):
            return _ServerProbeResult(responding=False, is_split_editor=False)

        if status != 200:
            return _ServerProbeResult(responding=True, is_split_editor=False)

        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            return _ServerProbeResult(responding=True, is_split_editor=False)

        is_split_editor = (
            isinstance(payload, dict)
            and "schema_version" in payload
            and isinstance(payload.get("tracks"), list)
            and isinstance(payload.get("notes"), list)
        )
        return _ServerProbeResult(responding=True, is_split_editor=is_split_editor)

    @staticmethod
    def _is_port_available(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                return False
        return True

    def _wait_until_split_editor_ready(
        self,
        *,
        url: str,
        handle: MidiSplitEditorServerHandle,
        timeout_sec: float = 4.0,
    ) -> str | None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            probe = self._probe_split_editor(url)
            if probe.is_split_editor:
                return None

            if not handle.thread.is_alive():
                return "MIDI Editor server stopped before becoming ready."

            time.sleep(0.1)

        return f"MIDI Editor server did not become ready at {url}"

    def _import_midi_into_server(self, *, url: str, midi_file: Path) -> str | None:
        payload = midi_file.read_bytes()
        endpoint = urllib.parse.urljoin(
            url,
            f"api/import-midi?filename={urllib.parse.quote(midi_file.name)}",
        )

        request = Request(
            endpoint,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/octet-stream",
                "X-File-Name": midi_file.name,
            },
        )

        try:
            with urlopen(request, timeout=self._request_timeout_sec) as response:
                if int(response.status) != 200:
                    return f"HTTP {response.status}"
        except HTTPError as exc:
            return f"HTTP {exc.code}"
        except (URLError, TimeoutError, OSError) as exc:
            return str(exc)

        return None
