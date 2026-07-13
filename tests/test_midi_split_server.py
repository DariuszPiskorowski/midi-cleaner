from __future__ import annotations

import json
import threading
import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import mido

from midi_cleaner.midi_split.models import MidiSplitSession
from midi_cleaner.midi_split.server import _default_session
from midi_cleaner.midi_split.server import _EditorState
from midi_cleaner.midi_split.server import _MidiSplitEditorRequestHandler
from midi_cleaner.midi_split.server import _session_to_html
from midi_cleaner.midi_split.service import create_split_session
from http.server import ThreadingHTTPServer


def _write_test_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi.tracks.append(tempo_track)

    notes_track = mido.MidiTrack()
    notes_track.append(mido.MetaMessage("track_name", name="Combo", time=0))
    notes_track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
    notes_track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=240))
    notes_track.append(mido.Message("note_on", note=67, velocity=90, channel=0, time=120))
    notes_track.append(mido.Message("note_off", note=67, velocity=0, channel=0, time=240))
    midi.tracks.append(notes_track)

    midi.save(path)


def _build_server(initial_session: MidiSplitSession):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MidiSplitEditorRequestHandler)
    server.editor_state = _EditorState(initial_session)  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    return server, thread, base_url


def _request_json(method: str, url: str, payload: bytes | None = None, content_type: str = "application/json"):
    headers = {}
    if payload is not None:
        headers["Content-Type"] = content_type
    request = Request(url, data=payload, method=method, headers=headers)
    with urlopen(request, timeout=10) as response:
        body = response.read()
        return response.status, response.headers, json.loads(body.decode("utf-8"))


def _request_bytes(method: str, url: str, payload: bytes | None = None, content_type: str = "application/json"):
    headers = {}
    if payload is not None:
        headers["Content-Type"] = content_type
    request = Request(url, data=payload, method=method, headers=headers)
    with urlopen(request, timeout=10) as response:
        body = response.read()
        return response.status, response.headers, body


def _repo_entries() -> set[str]:
    root = Path.cwd()
    return {entry.name for entry in root.iterdir()}


def test_session_to_html_contains_editor_document() -> None:
    html = _session_to_html(_default_session())

    assert "MIDI Split Editor Preview" in html
    assert "session-json" in html
    assert "Import MIDI" in html


def test_api_import_midi_returns_session_json_with_notes(tmp_path: Path) -> None:
    midi_path = tmp_path / "input.mid"
    _write_test_midi(midi_path)

    server, thread, base_url = _build_server(_default_session())
    try:
        payload = midi_path.read_bytes()
        status, _headers, body = _request_json(
            "POST",
            f"{base_url}/api/import-midi?filename=input.mid",
            payload=payload,
            content_type="application/octet-stream",
        )

        assert status == 200
        assert body["source_midi"] == "input.mid"
        assert len(body["notes"]) == 2
        assert len(body["tracks"]) == 1
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_api_export_multitrack_returns_non_empty_midi_bytes(tmp_path: Path) -> None:
    midi_path = tmp_path / "input.mid"
    _write_test_midi(midi_path)
    session = create_split_session(midi_path)

    server, thread, base_url = _build_server(session)
    try:
        payload = json.dumps(session.model_dump(mode="json")).encode("utf-8")
        status, headers, body = _request_bytes(
            "POST",
            f"{base_url}/api/export-multitrack",
            payload=payload,
            content_type="application/json",
        )

        assert status == 200
        assert len(body) > 0
        assert headers.get("Content-Type", "").startswith("audio/midi")

        exported_midi = mido.MidiFile(file=io.BytesIO(body))
        assert len(exported_midi.tracks) >= 1
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_api_export_separate_returns_zip_with_mid_entries(tmp_path: Path) -> None:
    midi_path = tmp_path / "input.mid"
    _write_test_midi(midi_path)
    session = create_split_session(midi_path)

    server, thread, base_url = _build_server(session)
    try:
        payload = json.dumps(session.model_dump(mode="json")).encode("utf-8")
        status, headers, body = _request_bytes(
            "POST",
            f"{base_url}/api/export-separate",
            payload=payload,
            content_type="application/json",
        )

        assert status == 200
        assert len(body) > 0
        assert headers.get("Content-Type", "").startswith("application/zip")

        with zipfile.ZipFile(io.BytesIO(body), mode="r") as archive:
            names = archive.namelist()
            assert names
            assert any(name.lower().endswith(".mid") for name in names)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_server_endpoints_do_not_leave_artifacts_in_repo(tmp_path: Path) -> None:
    midi_path = tmp_path / "input.mid"
    _write_test_midi(midi_path)
    baseline_entries = _repo_entries()

    server, thread, base_url = _build_server(_default_session())
    try:
        status, _headers, _body = _request_json(
            "POST",
            f"{base_url}/api/import-midi?filename=input.mid",
            payload=midi_path.read_bytes(),
            content_type="application/octet-stream",
        )
        assert status == 200

        current_session_status, _session_headers, current_session = _request_json(
            "GET",
            f"{base_url}/api/session",
        )
        assert current_session_status == 200

        payload = json.dumps(current_session).encode("utf-8")
        export_status, _export_headers, _export_bytes = _request_bytes(
            "POST",
            f"{base_url}/api/export-separate",
            payload=payload,
            content_type="application/json",
        )
        assert export_status == 200
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert _repo_entries() == baseline_entries
