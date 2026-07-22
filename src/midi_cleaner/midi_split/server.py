from __future__ import annotations

import argparse
import io
import json
import tempfile
import threading
import webbrowser
import zipfile
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from midi_cleaner.midi_split.exporter import (
    export_split_multitrack_midi,
    export_split_separate_midi_files,
)
from midi_cleaner.midi_split.html_preview import render_piano_roll_preview_html
from midi_cleaner.midi_split.models import MidiSplitSession
from midi_cleaner.midi_split.service import (
    MidiSplitSessionError,
    create_split_session,
    load_session,
)


class MidiSplitEditorServerError(Exception):
    """Raised when MIDI split editor server operations fail."""


@dataclass
class MidiSplitEditorServerHandle:
    server: ThreadingHTTPServer
    thread: threading.Thread
    host: str
    port: int
    url: str

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()


def _session_to_html(session: MidiSplitSession) -> str:
    return render_piano_roll_preview_html(session)


def _default_session() -> MidiSplitSession:
    return MidiSplitSession(
        schema_version="0.1.0",
        source_midi="",
        source="manual",
        layer="midi",
        ticks_per_beat=480,
        tempo_map=[],
        tracks=[],
        notes=[],
    )


def _sanitize_filename(value: str, fallback: str) -> str:
    if not value:
        return fallback

    name = Path(value).name.strip()
    if not name:
        return fallback

    safe = []
    for char in name:
        if char.isalnum() or char in {"_", "-", "."}:
            safe.append(char)
        else:
            safe.append("_")

    result = "".join(safe).strip("._")
    if not result:
        return fallback
    return result


class _EditorState:
    def __init__(self, initial_session: MidiSplitSession) -> None:
        self._lock = threading.Lock()
        self._session = initial_session

    def get_session(self) -> MidiSplitSession:
        with self._lock:
            return self._session.model_copy(deep=True)

    def set_session(self, session: MidiSplitSession) -> None:
        with self._lock:
            self._session = session


class _MidiSplitEditorRequestHandler(BaseHTTPRequestHandler):
    server_version = "MidiSplitEditor/0.1"

    @property
    def editor_state(self) -> _EditorState:
        return self.server.editor_state  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: object) -> None:  # pragma: no cover
        return

    def _read_body(self) -> bytes:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            return b""
        try:
            length = int(content_length)
        except ValueError:
            return b""
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8") + b"\n"
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        payload: bytes,
        *,
        content_type: str,
        status: int = HTTPStatus.OK,
        download_name: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        if download_name is not None:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"status": "error", "message": message}, status=status)

    def _parse_session_json(self) -> MidiSplitSession:
        body = self._read_body()
        if not body:
            raise MidiSplitEditorServerError("Request body is empty.")

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise MidiSplitEditorServerError("Invalid JSON body.") from exc

        try:
            return MidiSplitSession.model_validate(payload)
        except Exception as exc:
            raise MidiSplitEditorServerError("Invalid split session payload.") from exc

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/":
            session = self.editor_state.get_session()
            html = _session_to_html(session)
            self._send_bytes(html.encode("utf-8"), content_type="text/html; charset=utf-8")
            return

        if parsed.path == "/api/session":
            session = self.editor_state.get_session()
            self._send_json(session.model_dump(mode="json"), status=HTTPStatus.OK)
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {parsed.path}")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/api/import-midi":
            self._handle_import_midi(parsed)
            return

        if parsed.path == "/api/export-multitrack":
            self._handle_export_multitrack()
            return

        if parsed.path == "/api/export-separate":
            self._handle_export_separate()
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {parsed.path}")

    def _handle_import_midi(self, parsed_url) -> None:
        query = parse_qs(parsed_url.query)
        filename = query.get("filename", [""])[0]
        if not filename:
            filename = self.headers.get("X-File-Name", "")

        raw = self._read_body()
        payload_size = len(raw)
        safe_name = _sanitize_filename(filename, "uploaded.mid")
        display_name = filename.strip() if isinstance(filename, str) and filename.strip() else safe_name

        if not raw:
            self._send_json(
                {
                    "status": "error",
                    "message": "Empty MIDI payload.",
                    "filename": display_name,
                    "size_bytes": payload_size,
                },
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        with tempfile.TemporaryDirectory(prefix="midi_split_editor_import_") as temp_dir:
            midi_path = Path(temp_dir) / safe_name
            midi_path.write_bytes(raw)
            try:
                session = create_split_session(
                    midi_path,
                    source="manual",
                    layer="midi",
                    display_name=display_name,
                )
            except MidiSplitSessionError as exc:
                self._send_json(
                    {
                        "status": "error",
                        "message": str(exc),
                        "filename": display_name,
                        "size_bytes": payload_size,
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

        session = session.model_copy(update={"source_midi": safe_name})
        self.editor_state.set_session(session)
        self._send_json(session.model_dump(mode="json"), status=HTTPStatus.OK)

    def _handle_export_multitrack(self) -> None:
        try:
            session = self._parse_session_json()
        except MidiSplitEditorServerError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return

        self.editor_state.set_session(session)

        with tempfile.TemporaryDirectory(prefix="midi_split_editor_export_multi_") as temp_dir:
            output_path = Path(temp_dir) / "split.mid"
            export_split_multitrack_midi(session, output_path)
            payload = output_path.read_bytes()

        base = _sanitize_filename(Path(session.source_midi).stem or "split", "split")
        download_name = f"{base}_split.mid"
        self._send_bytes(
            payload,
            content_type="audio/midi",
            status=HTTPStatus.OK,
            download_name=download_name,
        )

    def _handle_export_separate(self) -> None:
        try:
            session = self._parse_session_json()
        except MidiSplitEditorServerError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return

        self.editor_state.set_session(session)

        with tempfile.TemporaryDirectory(prefix="midi_split_editor_export_sep_") as temp_dir:
            output_dir = Path(temp_dir) / "tracks"
            output_dir.mkdir(parents=True, exist_ok=True)
            export_split_separate_midi_files(session, output_dir=output_dir, skip_empty=True)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                for midi_file in sorted(output_dir.glob("*.mid")):
                    archive.write(midi_file, arcname=midi_file.name)

            payload = zip_buffer.getvalue()

        base = _sanitize_filename(Path(session.source_midi).stem or "split", "split")
        download_name = f"{base}_split_tracks.zip"
        self._send_bytes(
            payload,
            content_type="application/zip",
            status=HTTPStatus.OK,
            download_name=download_name,
        )


def _build_initial_session(
    *,
    input_midi: Path | None,
    session_path: Path | None,
) -> MidiSplitSession:
    if session_path is not None:
        return load_session(session_path)

    if input_midi is not None:
        return create_split_session(input_midi, source="manual", layer="midi")

    return _default_session()


def start_split_editor_server(
    *,
    input_midi: Path | None = None,
    session_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> MidiSplitEditorServerHandle:
    try:
        initial_session = _build_initial_session(input_midi=input_midi, session_path=session_path)
    except MidiSplitSessionError as exc:
        raise MidiSplitEditorServerError(str(exc)) from exc

    state = _EditorState(initial_session)

    server = ThreadingHTTPServer((host, port), _MidiSplitEditorRequestHandler)
    server.editor_state = state  # type: ignore[attr-defined]

    actual_host, actual_port = server.server_address
    url = f"http://{actual_host}:{actual_port}/"

    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.2},
        daemon=True,
        name=f"midi-split-editor-{actual_host}:{actual_port}",
    )
    thread.start()

    return MidiSplitEditorServerHandle(
        server=server,
        thread=thread,
        host=actual_host,
        port=int(actual_port),
        url=url,
    )


def run_split_editor_server(
    *,
    input_midi: Path | None = None,
    session_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> str:
    handle = start_split_editor_server(
        input_midi=input_midi,
        session_path=session_path,
        host=host,
        port=port,
    )
    url = handle.url

    if open_browser:
        webbrowser.open(url)

    print(f"MIDI split editor server running at {url}")
    print("Press Ctrl+C to stop.")

    try:
        while handle.thread.is_alive():
            handle.thread.join(timeout=0.2)
    except KeyboardInterrupt:  # pragma: no cover - interactive path
        pass
    finally:
        handle.stop()

    return url


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MIDI Split Editor local server.")
    parser.add_argument("--input", type=Path, default=None, help="Optional input MIDI path.")
    parser.add_argument("--session", type=Path, default=None, help="Optional input split session JSON path.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=0, help="Bind port (0 = auto).")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically.")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - exercised through CLI tests instead
    args = _parse_args()
    run_split_editor_server(
        input_midi=args.input,
        session_path=args.session,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    main()
