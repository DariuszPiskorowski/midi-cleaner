from __future__ import annotations

import io
import json
import shutil
import subprocess
import threading
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import mido
import pytest

from midi_cleaner.midi_split.server import _default_session
from midi_cleaner.midi_split.server import _EditorState
from midi_cleaner.midi_split.server import _MidiSplitEditorRequestHandler


def _playwright_available() -> bool:
    node = shutil.which("node")
    if node is None:
        return False

    try:
        result = subprocess.run(
            [node, "-e", "try { require.resolve('playwright'); process.exit(0); } catch (e) { process.exit(1); }"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return False

    return result.returncode == 0


PLAYWRIGHT_AVAILABLE = _playwright_available()

pytestmark = pytest.mark.skipif(
    not PLAYWRIGHT_AVAILABLE,
    reason="Node Playwright is not available in this environment",
)


def _write_browser_test_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi.tracks.append(tempo_track)

    track_a = mido.MidiTrack()
    track_a.append(mido.MetaMessage("track_name", name="Track A", time=0))
    track_a.append(mido.Message("note_on", note=40, velocity=100, channel=0, time=0))
    track_a.append(mido.Message("note_off", note=40, velocity=0, channel=0, time=240))
    track_a.append(mido.Message("note_on", note=43, velocity=96, channel=0, time=120))
    track_a.append(mido.Message("note_off", note=43, velocity=0, channel=0, time=240))
    midi.tracks.append(track_a)

    track_b = mido.MidiTrack()
    track_b.append(mido.MetaMessage("track_name", name="Track B", time=0))
    track_b.append(mido.Message("note_on", note=52, velocity=92, channel=1, time=60))
    track_b.append(mido.Message("note_off", note=52, velocity=0, channel=1, time=300))
    midi.tracks.append(track_b)

    midi.save(path)


def _build_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MidiSplitEditorRequestHandler)
    server.editor_state = _EditorState(_default_session())  # type: ignore[attr-defined]

    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()

    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"


def _node_script_content() -> str:
    return """
const fs = require("fs");
const { chromium } = require("playwright");

const [baseUrl, midiPath, outMidiPath, outZipPath, outResultPath] = process.argv.slice(2);

async function main() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();

  const consoleErrors = [];
  const pageErrors = [];

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      consoleErrors.push(msg.text());
    }
  });

  page.on("pageerror", (error) => {
    pageErrors.push(String(error));
  });

  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("#server-status");
  await page.waitForFunction(() => {
    const el = document.getElementById("server-status");
    return !!el && /connected/i.test(el.textContent || "");
  }, { timeout: 15000 });

  const hasSaveJsonButton = await page.locator("#save-session-btn").count();
  const hasDownloadJsonButton = await page.locator("#download-session-btn").count();

  await page.setInputFiles("#import-midi-input", midiPath);
  await page.waitForFunction(() => {
    const editor = window.__midiSplitEditor;
    return !!editor && editor.getSession().notes.length > 0;
  }, { timeout: 15000 });

  const firstStatus = await page.locator("#status-line").innerText();
  const firstInputValue = await page.$eval("#import-midi-input", (el) => el.value);
  const firstSession = await page.evaluate(() => window.__midiSplitEditor.getSession());
  const firstNoteBoxCount = await page.evaluate(() => window.__midiSplitEditor.getNoteBoxes().length);

  await page.evaluate(() => {
    const editor = window.__midiSplitEditor;
    const current = editor.getSession();
    if (!current.notes.length || current.tracks.length < 2) {
      return;
    }

    const targetTrack = current.tracks[current.tracks.length - 1].editable_track_index;
    editor.selectNotesByIds([current.notes[0].note_id]);
    const select = document.getElementById("target-track");
    if (select) {
      select.value = String(targetTrack);
    }
    const moveButton = document.getElementById("move-selected-btn");
    if (moveButton) {
      moveButton.click();
    }
  });

  await page.setInputFiles("#import-midi-input", midiPath);
  await page.waitForFunction(() => {
    const editor = window.__midiSplitEditor;
    return !!editor && editor.getSession().notes.length > 0;
  }, { timeout: 15000 });

  const secondStatus = await page.locator("#status-line").innerText();
  const secondInputValue = await page.$eval("#import-midi-input", (el) => el.value);
  const secondSession = await page.evaluate(() => window.__midiSplitEditor.getSession());

  const [multitrackDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.click("#export-multitrack-btn"),
  ]);
  await multitrackDownload.saveAs(outMidiPath);

  const [separateDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.click("#export-separate-btn"),
  ]);
  await separateDownload.saveAs(outZipPath);

  const serverStatus = await page.locator("#server-status").innerText();

  await browser.close();

  const result = {
    serverStatus,
    firstStatus,
    secondStatus,
    firstInputValue,
    secondInputValue,
    firstSessionNotes: firstSession.notes.length,
    firstSessionTracks: firstSession.tracks.length,
    secondSessionNotes: secondSession.notes.length,
    secondSessionTracks: secondSession.tracks.length,
    firstNoteBoxCount,
    hasSaveJsonButton,
    hasDownloadJsonButton,
    multitrackSuggestedFilename: multitrackDownload.suggestedFilename(),
    separateSuggestedFilename: separateDownload.suggestedFilename(),
    consoleErrors,
    pageErrors,
  };

  fs.writeFileSync(outResultPath, JSON.stringify(result, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
""".strip()


def test_browser_import_and_exports_via_real_controls(tmp_path: Path) -> None:
    midi_path = tmp_path / "browser_flow.mid"
    _write_browser_test_midi(midi_path)

    node_path = shutil.which("node")
    assert node_path is not None

    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    multitrack_path = downloads_dir / "multitrack.mid"
    zip_path = downloads_dir / "tracks.zip"
    result_path = tmp_path / "browser_result.json"

    script_path = tmp_path / "browser_flow.js"
    script_path.write_text(_node_script_content() + "\n", encoding="utf-8")

    server, thread, base_url = _build_server()
    try:
        completed = subprocess.run(
            [
                node_path,
                str(script_path),
                base_url,
                str(midi_path),
                str(multitrack_path),
                str(zip_path),
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert completed.returncode == 0, (
        "Browser automation script failed\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert "connected" in result["serverStatus"].lower()
    assert result["hasSaveJsonButton"] == 0
    assert result["hasDownloadJsonButton"] == 0

    assert result["firstSessionNotes"] > 0
    assert result["firstSessionTracks"] >= 2
    assert result["secondSessionNotes"] > 0
    assert result["secondSessionTracks"] >= 2
    assert result["firstNoteBoxCount"] > 0

    assert "Imported" in result["firstStatus"]
    assert "Imported" in result["secondStatus"]
    assert result["firstInputValue"] == ""
    assert result["secondInputValue"] == ""

    assert result["consoleErrors"] == []
    assert result["pageErrors"] == []

    assert multitrack_path.exists()
    assert multitrack_path.stat().st_size > 0
    parsed_multitrack = mido.MidiFile(str(multitrack_path))
    assert len(parsed_multitrack.tracks) > 0

    assert zip_path.exists()
    assert zip_path.stat().st_size > 0

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        names = archive.namelist()
        assert names
        assert all(name.lower().endswith(".mid") for name in names)

        for name in names:
            midi_data = archive.read(name)
            parsed = mido.MidiFile(file=io.BytesIO(midi_data))
            assert len(parsed.tracks) > 0
