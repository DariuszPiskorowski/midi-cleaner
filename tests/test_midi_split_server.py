from __future__ import annotations

import io
import json
import threading
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import mido

from midi_cleaner.midi_split.models import MidiSplitSession
from midi_cleaner.midi_split.server import _MidiSplitEditorRequestHandler
from midi_cleaner.midi_split.server import _EditorState
from midi_cleaner.midi_split.server import _default_session
from midi_cleaner.midi_split.server import _session_to_html
from midi_cleaner.midi_split.service import create_split_session


def _write_test_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi.tracks.append(tempo_track)

    notes_track_a = mido.MidiTrack()
    notes_track_a.append(mido.MetaMessage("track_name", name="Combo A", time=0))
    notes_track_a.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
    notes_track_a.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=240))
    notes_track_a.append(mido.Message("note_on", note=67, velocity=90, channel=0, time=120))
    notes_track_a.append(mido.Message("note_off", note=67, velocity=0, channel=0, time=240))
    midi.tracks.append(notes_track_a)

    notes_track_b = mido.MidiTrack()
    notes_track_b.append(mido.MetaMessage("track_name", name="Combo B", time=0))
    notes_track_b.append(mido.Message("note_on", note=43, velocity=88, channel=1, time=60))
    notes_track_b.append(mido.Message("note_off", note=43, velocity=0, channel=1, time=300))
    midi.tracks.append(notes_track_b)

    midi.save(path)


def _build_server(initial_session: MidiSplitSession):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MidiSplitEditorRequestHandler)
    server.editor_state = _EditorState(initial_session)  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    return server, thread, base_url


def _request_bytes(
    method: str,
    url: str,
    payload: bytes | None = None,
    content_type: str = "application/json",
    allow_error: bool = False,
):
    headers = {}
    if payload is not None:
        headers["Content-Type"] = content_type
    request = Request(url, data=payload, method=method, headers=headers)
    try:
        with urlopen(request, timeout=10) as response:
            body = response.read()
            return response.status, response.headers, body
    except HTTPError as exc:
        if not allow_error:
            raise
        body = exc.read()
        return exc.code, exc.headers, body


def _request_json(
    method: str,
    url: str,
    payload: bytes | None = None,
    content_type: str = "application/json",
    allow_error: bool = False,
):
    status, headers, body = _request_bytes(
        method,
        url,
        payload=payload,
        content_type=content_type,
        allow_error=allow_error,
    )
    return status, headers, json.loads(body.decode("utf-8"))


def _repo_entries() -> set[str]:
    root = Path.cwd()
    return {entry.name for entry in root.iterdir()}


def _collect_note_spans(midi: mido.MidiFile) -> list[tuple[int, int, int, int, int, int]]:
    spans: list[tuple[int, int, int, int, int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}

        for message in track:
            absolute_tick += int(message.time)

            is_note_on = message.type == "note_on" and message.velocity > 0
            is_note_off = message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            )

            if is_note_on:
                key = (int(message.channel), int(message.note))
                active.setdefault(key, []).append((absolute_tick, int(message.velocity)))
                continue

            if is_note_off:
                key = (int(message.channel), int(message.note))
                if key in active and active[key]:
                    start_tick, velocity = active[key].pop(0)
                    spans.append(
                        (
                            int(track_index),
                            int(message.note),
                            int(velocity),
                            int(message.channel),
                            int(start_tick),
                            int(absolute_tick),
                        )
                    )

    spans.sort(key=lambda item: (item[0], item[4], item[5], item[3], item[1], item[2]))
    return spans


def _expected_session_spans(session: MidiSplitSession) -> list[tuple[int, int, int, int, int, int]]:
    spans = [
        (
            int(note.editable_track_index),
            int(note.pitch_midi),
            int(note.velocity),
            int(note.channel) if note.channel is not None else 0,
            int(note.start_tick),
            int(note.end_tick),
        )
        for note in session.notes
    ]
    spans.sort(key=lambda item: (item[0], item[4], item[5], item[3], item[1], item[2]))
    return spans


def test_session_to_html_contains_editor_document() -> None:
    html = _session_to_html(_default_session())

    assert "MIDI Split Editor Preview" in html
    assert "session-json" in html
    assert "Import MIDI" in html
    assert "Save Session JSON" not in html
    assert "Download updated session JSON" not in html


def test_api_get_endpoints_include_no_store_headers() -> None:
    server, thread, base_url = _build_server(_default_session())
    try:
        status_root, headers_root, body_root = _request_bytes("GET", f"{base_url}/")
        assert status_root == 200
        assert "Import MIDI" in body_root.decode("utf-8")
        assert headers_root.get("Cache-Control", "").startswith("no-store")

        status_session, headers_session, body_session = _request_bytes("GET", f"{base_url}/api/session")
        assert status_session == 200
        assert json.loads(body_session.decode("utf-8"))["schema_version"] == "0.1.0"
        assert headers_session.get("Cache-Control", "").startswith("no-store")
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


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
        assert len(body["notes"]) == 3
        assert len(body["tracks"]) == 2
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_api_import_midi_rejects_invalid_payload(tmp_path: Path) -> None:
    _ = tmp_path
    server, thread, base_url = _build_server(_default_session())
    try:
        status, _headers, body = _request_json(
            "POST",
            f"{base_url}/api/import-midi?filename=bad.mid",
            payload=b"not-a-midi",
            content_type="application/octet-stream",
            allow_error=True,
        )

        assert status == 400
        assert body["status"] == "error"
        assert isinstance(body["message"], str)
        assert body["message"].strip()
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
        assert "_split.mid" in headers.get("Content-Disposition", "")

        exported_midi = mido.MidiFile(file=io.BytesIO(body))
        assert exported_midi.ticks_per_beat == session.ticks_per_beat

        exported_spans = [span for span in _collect_note_spans(exported_midi) if span[0] > 0]
        expected_spans = _expected_session_spans(session)
        assert exported_spans == expected_spans
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
        assert "_split_tracks.zip" in headers.get("Content-Disposition", "")

        with zipfile.ZipFile(io.BytesIO(body), mode="r") as archive:
            names = archive.namelist()
            assert names
            assert len(names) == 2
            assert all(name.lower().endswith(".mid") for name in names)
            assert all("/" not in name and "\\" not in name for name in names)

            for name in names:
                with archive.open(name) as midi_file:
                    midi_bytes = midi_file.read()
                parsed = mido.MidiFile(file=io.BytesIO(midi_bytes))
                spans = _collect_note_spans(parsed)
                assert spans
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
