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


def _write_long_browser_test_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi.tracks.append(tempo_track)

    long_track = mido.MidiTrack()
    long_track.append(mido.MetaMessage("track_name", name="Long Browser Track", time=0))

    events = [
        (0, 240, 40, 100),
        (960, 240, 43, 98),
        (28800, 480, 45, 96),
        (57600, 480, 47, 95),
        (115200, 960, 48, 94),
        (230400, 960, 50, 92),
    ]

    current_tick = 0
    for start_tick, duration_tick, pitch, velocity in events:
        delta = max(0, int(start_tick) - int(current_tick))
        long_track.append(mido.Message("note_on", note=int(pitch), velocity=int(velocity), channel=0, time=delta))
        long_track.append(mido.Message("note_off", note=int(pitch), velocity=0, channel=0, time=int(duration_tick)))
        current_tick = int(start_tick) + int(duration_tick)

    midi.tracks.append(long_track)
    midi.save(path)


def _write_long_invalid_key_signature_browser_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi.tracks.append(tempo_track)

    synth_track = mido.MidiTrack()
    synth_track.append(mido.MetaMessage("track_name", name="Played With Fire - Deep House (Synth)", time=0))
    synth_track.append(mido.MetaMessage("key_signature", key="C", time=0))

    events = [
        (2400, 960, 36, 100),
        (4800, 480, 48, 96),
        (38400, 960, 52, 95),
        (76800, 960, 55, 94),
        (153600, 960, 57, 92),
        (230400, 960, 59, 90),
        (300000, 960, 62, 88),
        (309550, 130, 65, 86),
    ]

    current_tick = 0
    for start_tick, duration_tick, pitch, velocity in events:
        delta = max(0, int(start_tick) - int(current_tick))
        synth_track.append(
            mido.Message("note_on", note=int(pitch), velocity=int(velocity), channel=0, time=delta)
        )
        synth_track.append(
            mido.Message("note_off", note=int(pitch), velocity=0, channel=0, time=int(duration_tick))
        )
        current_tick = int(start_tick) + int(duration_tick)

    midi.tracks.append(synth_track)
    midi.save(path)

    raw = bytearray(path.read_bytes())
    marker = bytes([0xFF, 0x59, 0x02])
    marker_index = raw.find(marker)
    assert marker_index >= 0
    raw[marker_index + 3] = 0x0E
    raw[marker_index + 4] = 0x01
    path.write_bytes(raw)


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

  const playbackReport = await page.evaluate(async () => {
    const editor = window.__midiSplitEditor;
    const statusEl = document.getElementById("status-line");
    const playSelectedButton = document.getElementById("audition-selected-btn");
    const playRegionButton = document.getElementById("play-region-btn");
    const playAllButton = document.getElementById("play-all-btn");
    const stopButton = document.getElementById("stop-midi-btn");

    const notes = editor.getSession().notes || [];
    const playableNotes = notes.filter((note) => note && note.muted !== true);
    if (!playableNotes.length) {
      return null;
    }

    editor.setMidiOutEnabledForTest(false);
    editor.selectNotesByIds([String(playableNotes[0].note_id)]);

    const selectedState = editor.getPlaybackControlState();
    const playAllEnabledWithoutMidi = !playAllButton.disabled;
    const playSelectedEnabledWithoutMidi = !playSelectedButton.disabled;

    const regionStart = Math.max(0, Math.round(Number(playableNotes[0].start_tick || 0)));
    const regionEnd = Math.max(
      regionStart + 1,
      Math.round(Number(playableNotes[0].end_tick || playableNotes[0].start_tick || 0))
    );
    editor.setSelectionRegionForTest(regionStart, regionEnd);
    const regionState = editor.getPlaybackControlState();
    const playRegionEnabledWithoutMidi = !playRegionButton.disabled;

    editor.setTool("zoom");
    const canvas = document.getElementById("roll-canvas");
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      for (let i = 0; i < 10; i += 1) {
        const zoomEvent = new WheelEvent("wheel", {
          deltaY: -120,
          clientX: rect.left + rect.width * 0.72,
          clientY: rect.top + rect.height * 0.3,
          bubbles: true,
          cancelable: true,
        });
        canvas.dispatchEvent(zoomEvent);
      }
    }
    editor.setTool("select");

    editor.panViewportByTicks(6000);
    const viewportBeforeFollowOn = editor.getViewState();
    editor.setFollowPlayheadForTest(true);
    const startedFollowOn = editor.playAllNotes();
    const statusAfterPlayAllNoMidi = String(statusEl?.textContent || "");
    const visualAtStartFollowOn = editor.getPlaybackVisualState();
    const stopEnabledDuringFollowOn = !stopButton.disabled;

    let sawActiveHighlight = false;
    const activeStart = Date.now();
    while (Date.now() - activeStart < 1500) {
      const visualState = editor.getPlaybackVisualState();
      if (Array.isArray(visualState.activeNoteIds) && visualState.activeNoteIds.length > 0) {
        sawActiveHighlight = true;
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 20));
    }

    const viewportDuringFollowOn = editor.getViewState();
    editor.stopMidiPlayback({ sendPanic: true });
    const visualAfterStopNoMidi = editor.getPlaybackVisualState();

    editor.panViewportByTicks(5000);
    const viewportBeforeFollowOff = editor.getViewState();
    editor.setFollowPlayheadForTest(false);
    const startedFollowOff = editor.playSelectedRegion();
    const statusAfterRegionNoMidi = String(statusEl?.textContent || "");
    await new Promise((resolve) => setTimeout(resolve, 120));
    const viewportDuringFollowOff = editor.getViewState();
    editor.stopMidiPlayback({ sendPanic: true });
    const visualAfterFollowOffStop = editor.getPlaybackVisualState();

    editor.setMidiOutEnabledForTest(true);
    editor.setSelectedMidiOutputForTest("__test__");
    editor.selectNotesByIds([String(playableNotes[0].note_id)]);
    const midiStateBefore = editor.getMidiOutState();
    const startedWithMidi = editor.playSelectedNotes();
    await new Promise((resolve) => setTimeout(resolve, 220));
    editor.stopMidiPlayback({ sendPanic: true });
    const midiStateAfter = editor.getMidiOutState();

    return {
      selectedState,
      regionState,
      playAllEnabledWithoutMidi,
      playSelectedEnabledWithoutMidi,
      playRegionEnabledWithoutMidi,
      startedFollowOn,
      statusAfterPlayAllNoMidi,
      visualAtStartFollowOn,
      stopEnabledDuringFollowOn,
      sawActiveHighlight,
      viewportBeforeFollowOn,
      viewportDuringFollowOn,
      visualAfterStopNoMidi,
      startedFollowOff,
      statusAfterRegionNoMidi,
      viewportBeforeFollowOff,
      viewportDuringFollowOff,
      visualAfterFollowOffStop,
      midiStateBefore,
      startedWithMidi,
      midiStateAfter,
    };
  });

  const loopReport = await page.evaluate(() => {
    const editor = window.__midiSplitEditor;
    const before = editor.getSession();
    const sourceSelection = before.notes.slice(0, Math.min(2, before.notes.length)).map((n) => String(n.note_id));
    if (!sourceSelection.length) {
      return null;
    }

    editor.selectNotesByIds(sourceSelection);
    const region = editor.getSelectedRegionForLoop();
    editor.setLoopRepeatCount(1);
    const repeatOneResult = editor.loopSelectedNotes();
    const afterRepeatOne = editor.getSession();
    const selectedAfterRepeatOne = editor.getSelectedNoteIds();
    const noteBoxesAfterRepeatOne = editor.getNoteBoxes();
    const velocityBarsAfterRepeatOne = editor.getVelocityBars();
    const statusAfterRepeatOne = String(document.getElementById("status-line")?.textContent || "");

    editor.undo();
    const afterUndoCount = editor.getSession().notes.length;
    editor.redo();
    const afterRedoCount = editor.getSession().notes.length;

    editor.selectNotesByIds(sourceSelection);
    editor.setLoopRepeatCount(3);
    const repeatThreeResult = editor.loopSelectedNotes();
    const afterRepeatThree = editor.getSession();
    const statusAfterRepeatThree = String(document.getElementById("status-line")?.textContent || "");

    const requestedTooHigh = editor.setLoopRepeatCount(999);
    const clampedRepeat = editor.getLoopRepeatCount();

    const createdByOne = repeatOneResult && Array.isArray(repeatOneResult.created_note_ids)
      ? repeatOneResult.created_note_ids
      : [];
    const createdByThree = repeatThreeResult && Array.isArray(repeatThreeResult.created_note_ids)
      ? repeatThreeResult.created_note_ids
      : [];

    const createdNotesOne = afterRepeatOne.notes.filter((n) => createdByOne.includes(String(n.note_id)));
    const createdNotesThree = afterRepeatThree.notes.filter((n) => createdByThree.includes(String(n.note_id)));

    return {
      region,
      statusAfterRepeatOne,
      statusAfterRepeatThree,
      beforeCount: before.notes.length,
      afterRepeatOneCount: afterRepeatOne.notes.length,
      afterUndoCount,
      afterRedoCount,
      afterRepeatThreeCount: afterRepeatThree.notes.length,
      repeatOneResult,
      repeatThreeResult,
      selectedAfterRepeatOne,
      noteBoxesAfterRepeatOne,
      velocityBarsAfterRepeatOne,
      createdByOne,
      createdByThree,
      createdNotesOne,
      createdNotesThree,
      requestedTooHigh,
      clampedRepeat,
    };
  });

  const copyPasteReport = await page.evaluate(() => {
    const editor = window.__midiSplitEditor;
    const before = editor.getSession();
    const preIds = new Set(before.notes.map((n) => String(n.note_id)));
    const selectedIds = before.notes.slice(0, Math.min(2, before.notes.length)).map((n) => String(n.note_id));
    editor.selectNotesByIds(selectedIds);
    editor.copySelectedNotes();
    const statusAfterCopy = String(document.getElementById("status-line")?.textContent || "");
    const clipboard = editor.getClipboardSummary();

    const beforePasteCount = editor.getSession().notes.length;
    editor.pasteCopiedNotes();
    const afterFirstPaste = editor.getSession();
    const addedAfterFirstPaste = afterFirstPaste.notes.filter((n) => !preIds.has(String(n.note_id)));
    const selectedAfterFirstPaste = editor.getSelectedNoteIds();
    const noteBoxesAfterFirstPaste = editor.getNoteBoxes();
    const velocityBarsAfterFirstPaste = editor.getVelocityBars();
    const statusAfterFirstPaste = String(document.getElementById("status-line")?.textContent || "");
    const cursorAfterFirstPaste = editor.getPasteCursorTick();

    const firstAddedIdSet = new Set(addedAfterFirstPaste.map((n) => String(n.note_id)));
    editor.pasteCopiedNotes();
    const afterSecondPaste = editor.getSession();
    const addedAfterSecondPaste = afterSecondPaste.notes.filter(
      (n) => !preIds.has(String(n.note_id)) && !firstAddedIdSet.has(String(n.note_id))
    );
    const cursorAfterSecondPaste = editor.getPasteCursorTick();

    const beforeFarPasteView = editor.getViewState();
    editor.setPasteCursorTick(Math.round(beforeFarPasteView.xOffsetTicks + 6000));
    editor.pasteCopiedNotes();
    const afterFarPasteView = editor.getViewState();

    const getIdSet = (items) => new Set(items.map((item) => String(item.note_id)));
    const noteBoxIdSet = getIdSet(noteBoxesAfterFirstPaste);
    const velocityBarIdSet = getIdSet(velocityBarsAfterFirstPaste);

    return {
      statusAfterCopy,
      statusAfterFirstPaste,
      clipboard,
      beforePasteCount,
      afterFirstPasteCount: afterFirstPaste.notes.length,
      afterSecondPasteCount: afterSecondPaste.notes.length,
      addedAfterFirstPaste,
      addedAfterSecondPaste,
      selectedAfterFirstPaste,
      noteBoxContainsAllFirstPasteIds: addedAfterFirstPaste.every((n) => noteBoxIdSet.has(String(n.note_id))),
      velocityBarContainsAllFirstPasteIds: addedAfterFirstPaste.every((n) => velocityBarIdSet.has(String(n.note_id))),
      firstPasteIdsUnique: (new Set(addedAfterFirstPaste.map((n) => String(n.note_id)))).size === addedAfterFirstPaste.length,
      cursorAfterFirstPaste,
      cursorAfterSecondPaste,
      beforeFarPasteOffset: beforeFarPasteView.xOffsetTicks,
      afterFarPasteOffset: afterFarPasteView.xOffsetTicks,
    };
  });

  const velocityTargetId = secondSession.notes[0]?.note_id;
  if (!velocityTargetId) {
    throw new Error("No note available for velocity drag test");
  }

  await page.evaluate((noteId) => {
    const editor = window.__midiSplitEditor;
    editor.selectNotesByIds([noteId]);
    editor.setVelocityForNoteIds([noteId], { targetVelocity: 60 });
  }, velocityTargetId);

  async function dragVelocityBar(noteId, deltaY, useShift) {
    const point = await page.evaluate((id) => {
      const editor = window.__midiSplitEditor;
      const bar = editor.velocityBarForNoteId(id);
      if (!bar) {
        return null;
      }
      const canvas = document.getElementById("roll-canvas");
      if (!canvas) {
        return null;
      }
      const rect = canvas.getBoundingClientRect();
      return {
        x: rect.left + bar.x + Math.max(1, bar.w / 2),
        y: rect.top + bar.y + Math.max(1, Math.min(bar.h - 1, 4)),
      };
    }, noteId);

    if (!point) {
      throw new Error("Velocity bar point not found for note " + String(noteId));
    }

    if (useShift) {
      await page.keyboard.down("Shift");
    }
    await page.mouse.move(point.x, point.y);
    await page.mouse.down();
    await page.mouse.move(point.x, point.y - deltaY);
    await page.mouse.up();
    if (useShift) {
      await page.keyboard.up("Shift");
    }
  }

  function getVelocity(noteId) {
    return page.evaluate((id) => {
      const editor = window.__midiSplitEditor;
      const note = editor.getSession().notes.find((n) => String(n.note_id) === String(id));
      return note ? Number(note.velocity) : null;
    }, noteId);
  }

  const velocityBefore = await getVelocity(velocityTargetId);
  await dragVelocityBar(velocityTargetId, 1, false);
  const velocityAfter1px = await getVelocity(velocityTargetId);

  await page.evaluate((noteId) => {
    const editor = window.__midiSplitEditor;
    editor.setVelocityForNoteIds([noteId], { targetVelocity: 60 });
  }, velocityTargetId);
  await dragVelocityBar(velocityTargetId, 20, false);
  const velocityAfter20pxUp = await getVelocity(velocityTargetId);

  await page.evaluate((noteId) => {
    const editor = window.__midiSplitEditor;
    editor.setVelocityForNoteIds([noteId], { targetVelocity: 60 });
  }, velocityTargetId);
  await dragVelocityBar(velocityTargetId, -20, false);
  const velocityAfter20pxDown = await getVelocity(velocityTargetId);

  await page.evaluate((noteId) => {
    const editor = window.__midiSplitEditor;
    editor.setVelocityForNoteIds([noteId], { targetVelocity: 60 });
  }, velocityTargetId);
  await dragVelocityBar(velocityTargetId, 20, true);
  const velocityAfter20pxUpShift = await getVelocity(velocityTargetId);

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
    playbackReport,
    firstNoteBoxCount,
    hasSaveJsonButton,
    hasDownloadJsonButton,
    multitrackSuggestedFilename: multitrackDownload.suggestedFilename(),
    separateSuggestedFilename: separateDownload.suggestedFilename(),
    copyPasteReport,
    loopReport,
    velocityBefore,
    velocityAfter1px,
    velocityAfter20pxUp,
    velocityAfter20pxDown,
    velocityAfter20pxUpShift,
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


def _node_long_viewport_script_content() -> str:
    return """
const fs = require("fs");
const { chromium } = require("playwright");

const [baseUrl, midiPath, outResultPath] = process.argv.slice(2);

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

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

  await page.setInputFiles("#import-midi-input", midiPath);
  await page.waitForFunction(() => {
    const editor = window.__midiSplitEditor;
    return !!editor && editor.getSession().notes.length >= 6;
  }, { timeout: 15000 });

  const report = await page.evaluate(async () => {
    const editor = window.__midiSplitEditor;
    const session = editor.getSession();
    const notes = Array.isArray(session.notes) ? session.notes.slice() : [];

    const sessionMaxByData = notes.reduce((maxTick, note) => {
      const startTick = Number(note?.start_tick || 0);
      const endTick = Number(note?.end_tick);
      if (Number.isFinite(endTick)) {
        return Math.max(maxTick, Math.max(startTick, endTick));
      }
      const durationTick = Number(note?.duration_ticks || 0);
      return Math.max(maxTick, Math.max(startTick, startTick + Math.max(0, durationTick)));
    }, 0);

    const sessionMaxByApi = Number(editor.getSessionMaxTick());
    const viewportStart = editor.getViewState();
    const viewportMaxOffset = Number(editor.getViewportMaxOffsetTicks());
    const expectedMinOffset = Math.max(0, sessionMaxByApi - Number(viewportStart.visibleTickSpan || 0));

    const lastNote = notes.reduce((best, note) => {
      if (!best) {
        return note;
      }
      const currentEnd = Number(note?.end_tick || note?.start_tick || 0);
      const bestEnd = Number(best?.end_tick || best?.start_tick || 0);
      return currentEnd > bestEnd ? note : best;
    }, null);
    const lastNoteId = lastNote ? String(lastNote.note_id) : null;

    const nearEndRequest = Math.max(0, sessionMaxByApi - Number(viewportStart.visibleTickSpan || 0) * 0.5);
    const offsetNearEnd = Number(editor.setXOffsetTicksForTest(nearEndRequest));
    const viewportNearEnd = editor.getViewState();
    const visibleNoteIdsNearEnd = editor.getVisibleNoteIds();
    const noteBoxesNearEnd = editor.getNoteBoxes();
    const nearEndContainsLast = Boolean(lastNoteId && visibleNoteIdsNearEnd.includes(lastNoteId));

    const playAllEvents = editor.buildPlaybackEventsForAll();
    const playAllMaxEndTick = playAllEvents.reduce((maxTick, event) => {
      return Math.max(maxTick, Number(event.end_tick || event.start_tick || 0));
    }, 0);

    editor.setMidiOutEnabledForTest(false);
    const statusEl = document.getElementById("status-line");

    const waitForActiveHighlight = async (timeoutMs) => {
      const startedAt = Date.now();
      while (Date.now() - startedAt < Number(timeoutMs || 0)) {
        const visualState = editor.getPlaybackVisualState();
        if (Array.isArray(visualState.activeNoteIds) && visualState.activeNoteIds.length > 0) {
          return true;
        }
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
      return false;
    };

    editor.setTool("zoom");
    const canvas = document.getElementById("roll-canvas");
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      for (let i = 0; i < 10; i += 1) {
        const zoomEvent = new WheelEvent("wheel", {
          deltaY: -120,
          clientX: rect.left + rect.width * 0.68,
          clientY: rect.top + rect.height * 0.3,
          bubbles: true,
          cancelable: true,
        });
        canvas.dispatchEvent(zoomEvent);
      }
    }
    editor.setTool("select");

    editor.setXOffsetTicksForTest(0);
    const viewportBeforeFollowOn = editor.getViewState();
    const globalMaxBeforeFollowOn = Number(editor.getSessionMaxTick());
    editor.setFollowPlayheadForTest(true);
    const startedFollowOn = editor.playAllNotes();
    const statusAfterFollowOnPlay = String(statusEl?.textContent || "");
    const sawActiveFollowOn = await waitForActiveHighlight(1800);
    await new Promise((resolve) => setTimeout(resolve, 4500));
    const viewportAfterFollowOn = editor.getViewState();
    const visualDuringFollowOn = editor.getPlaybackVisualState();
    const globalMaxDuringFollowOn = Number(editor.getSessionMaxTick());
    editor.stopMidiPlayback({ sendPanic: true });
    const visualAfterFollowOnStop = editor.getPlaybackVisualState();
    const globalMaxAfterFollowOnStop = Number(editor.getSessionMaxTick());

    editor.setXOffsetTicksForTest(6000);
    const viewportBeforeFollowOff = editor.getViewState();
    const globalMaxBeforeFollowOff = Number(editor.getSessionMaxTick());
    editor.setFollowPlayheadForTest(false);
    const startedFollowOff = editor.playAllNotes();
    const statusAfterFollowOffPlay = String(statusEl?.textContent || "");
    const sawActiveFollowOff = await waitForActiveHighlight(1800);
    const viewportBeforeFollowOffManualPan = editor.getViewState();
    const manualPanChanged = editor.panViewportByTicks(4000);
    const viewportAfterFollowOffManualPan = editor.getViewState();
    await new Promise((resolve) => setTimeout(resolve, 1200));
    const viewportAfterFollowOffWait = editor.getViewState();
    const visualDuringFollowOff = editor.getPlaybackVisualState();
    const globalMaxDuringFollowOff = Number(editor.getSessionMaxTick());
    editor.stopMidiPlayback({ sendPanic: true });
    const visualAfterFollowOffStop = editor.getPlaybackVisualState();
    const globalMaxAfterFollowOffStop = Number(editor.getSessionMaxTick());

    const firstNote = notes.reduce((best, note) => {
      if (!best) {
        return note;
      }
      return Number(note.start_tick || 0) < Number(best.start_tick || 0) ? note : best;
    }, null);
    const regionStart = firstNote ? Math.max(0, Math.round(Number(firstNote.start_tick || 0))) : 0;
    const regionEnd = firstNote
      ? Math.max(regionStart + 1, Math.round(Number(firstNote.end_tick || regionStart + 1)))
      : regionStart + 1;
    editor.setSelectionRegionForTest(regionStart, regionEnd);
    const regionEvents = editor.buildPlaybackEventsForRegion();
    const regionEventMaxEndTick = regionEvents.reduce((maxTick, event) => {
      return Math.max(maxTick, Number(event.end_tick || event.start_tick || 0));
    }, 0);

    const globalMaxBeforeRegionPlay = Number(editor.getSessionMaxTick());
    const startedRegion = editor.playSelectedRegion();
    const visualAtRegionStart = editor.getPlaybackVisualState();
    await new Promise((resolve) => setTimeout(resolve, 120));
    const globalMaxDuringRegionPlay = Number(editor.getSessionMaxTick());
    editor.stopMidiPlayback({ sendPanic: true });

    return {
      sessionNoteCount: notes.length,
      sessionMaxByData,
      sessionMaxByApi,
      viewportStart,
      viewportMaxOffset,
      expectedMinOffset,
      lastNoteId,
      offsetNearEnd,
      viewportNearEnd,
      visibleNoteIdsNearEnd,
      noteBoxesNearEndCount: noteBoxesNearEnd.length,
      nearEndContainsLast,
      playAllEventCount: playAllEvents.length,
      playAllMaxEndTick,
      startedFollowOn,
      statusAfterFollowOnPlay,
      sawActiveFollowOn,
      viewportBeforeFollowOn,
      viewportAfterFollowOn,
      followOnOffsetDelta: Number(viewportAfterFollowOn.xOffsetTicks || 0) - Number(viewportBeforeFollowOn.xOffsetTicks || 0),
      visualDuringFollowOn,
      visualAfterFollowOnStop,
      globalMaxBeforeFollowOn,
      globalMaxDuringFollowOn,
      globalMaxAfterFollowOnStop,
      startedFollowOff,
      statusAfterFollowOffPlay,
      sawActiveFollowOff,
      viewportBeforeFollowOff,
      viewportBeforeFollowOffManualPan,
      viewportAfterFollowOffManualPan,
      viewportAfterFollowOffWait,
      manualPanChanged,
      manualPanDelta: Number(viewportAfterFollowOffManualPan.xOffsetTicks || 0) - Number(viewportBeforeFollowOffManualPan.xOffsetTicks || 0),
      followOffOffsetDrift: Number(viewportAfterFollowOffWait.xOffsetTicks || 0) - Number(viewportAfterFollowOffManualPan.xOffsetTicks || 0),
      visualDuringFollowOff,
      visualAfterFollowOffStop,
      globalMaxBeforeFollowOff,
      globalMaxDuringFollowOff,
      globalMaxAfterFollowOffStop,
      startedRegion,
      visualAtRegionStart,
      regionStart,
      regionEnd,
      regionEventMaxEndTick,
      globalMaxBeforeRegionPlay,
      globalMaxDuringRegionPlay,
    };
  });

  await browser.close();

  fs.writeFileSync(
    outResultPath,
    JSON.stringify({ report, consoleErrors, pageErrors }, null, 2)
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
""".strip()


def _node_long_user_file_script_content() -> str:
    return """
const fs = require("fs");
const { chromium } = require("playwright");

const [baseUrl, midiPath, outResultPath] = process.argv.slice(2);

function tickToSeconds(tick, tempoMap, ticksPerBeat) {
  const targetTick = Math.max(0, Number(tick || 0));
  if (!Array.isArray(tempoMap) || !tempoMap.length) {
    return (targetTick / Math.max(1, Number(ticksPerBeat || 480))) * 0.5;
  }

  const sorted = tempoMap
    .map((event) => ({
      tick: Number(event.tick || 0),
      sec: Number(event.sec || 0),
      tempo: Number(event.tempo_us_per_beat || 500000),
    }))
    .sort((a, b) => a.tick - b.tick);

  let current = sorted[0];
  let sec = Number(current.sec || 0);
  for (let index = 1; index < sorted.length; index += 1) {
    const next = sorted[index];
    if (targetTick <= next.tick) {
      return sec + ((targetTick - current.tick) / ticksPerBeat) * (current.tempo / 1_000_000);
    }
    sec += ((next.tick - current.tick) / ticksPerBeat) * (current.tempo / 1_000_000);
    current = next;
  }

  return sec + ((targetTick - current.tick) / ticksPerBeat) * (current.tempo / 1_000_000);
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

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

  await page.setInputFiles("#import-midi-input", midiPath);
  await page.waitForFunction(() => {
    const editor = window.__midiSplitEditor;
    return !!editor && editor.getSession().notes.length > 0;
  }, { timeout: 15000 });

  const report = await page.evaluate(async () => {
    const editor = window.__midiSplitEditor;
    const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    const session = editor.getSession();
    const notes = Array.isArray(session.notes) ? session.notes.slice() : [];
    const ticksPerBeat = Math.max(1, Number(session.ticks_per_beat || 480));

    const tickToSeconds = (tick, tempoMap, ticksPerBeatValue) => {
      const targetTick = Math.max(0, Number(tick || 0));
      if (!Array.isArray(tempoMap) || !tempoMap.length) {
        return (targetTick / Math.max(1, Number(ticksPerBeatValue || 480))) * 0.5;
      }

      const sorted = tempoMap
        .map((event) => ({
          tick: Number(event.tick || 0),
          sec: Number(event.sec || 0),
          tempo: Number(event.tempo_us_per_beat || 500000),
        }))
        .sort((a, b) => a.tick - b.tick);

      let current = sorted[0];
      let sec = Number(current.sec || 0);
      for (let index = 1; index < sorted.length; index += 1) {
        const next = sorted[index];
        if (targetTick <= next.tick) {
          return sec + ((targetTick - current.tick) / ticksPerBeatValue) * (current.tempo / 1_000_000);
        }
        sec += ((next.tick - current.tick) / ticksPerBeatValue) * (current.tempo / 1_000_000);
        current = next;
      }

      return sec + ((targetTick - current.tick) / ticksPerBeatValue) * (current.tempo / 1_000_000);
    };

    const maxEndTick = notes.reduce((maxTick, note) => {
      return Math.max(maxTick, Number(note?.end_tick || note?.start_tick || 0));
    }, 0);
    const maxEndSec = notes.reduce((maxSec, note) => {
      return Math.max(maxSec, Number(note?.end_sec || note?.start_sec || 0));
    }, 0);
    const notesAfter40 = notes.filter((note) => Number(note?.end_sec || 0) >= 40).length;

    editor.setXOffsetTicksForTest(maxEndTick);
    const viewAfterEndPan = editor.getViewState();
    const visibleIds = editor.getVisibleNoteIds ? editor.getVisibleNoteIds() : [];

    const lastTen = notes
      .slice()
      .sort((a, b) => Number(a?.end_tick || 0) - Number(b?.end_tick || 0))
      .slice(-10)
      .map((note) => String(note.note_id));
    const lastSet = new Set(lastTen);
    const visibleLastIds = visibleIds.filter((noteId) => lastSet.has(String(noteId)));

    editor.setMidiOutEnabledForTest(false);
    editor.setFollowPlayheadForTest(true);

    const beforeMax = Number(editor.getSessionMaxTick());
    const started = editor.playAllNotes();
    await wait(2500);

    const stateAfterWait = editor.getPlaybackVisualState();
    const playheadTickAfterWait = Number(stateAfterWait.currentTick || 0);
    const playheadSecAfterWait = tickToSeconds(
      playheadTickAfterWait,
      session.tempo_map,
      ticksPerBeat,
    );

    const ffTick = Number(editor.getPlayheadTickForElapsedMs(45000));
    const ffSec = tickToSeconds(ffTick, session.tempo_map, ticksPerBeat);

    editor.setPlaybackVisualStateForTest({
      isPlaying: true,
      timingStartTick: Number(stateAfterWait.timingStartTick || 0),
      playbackEndTick: Number(stateAfterWait.playbackEndTick || beforeMax),
      currentTick: ffTick,
      durationMs: Number(stateAfterWait.durationMs || 0),
      followPlayhead: true,
    });
    editor.centerViewportOnTick(ffTick);
    const viewAfterFastForward = editor.getViewState();

    editor.stopMidiPlayback({ sendPanic: true });

    return {
      sourceMidi: String(session.source_midi || ""),
      noteCount: notes.length,
      maxEndTick,
      maxEndSec,
      notesAfter40,
      viewAfterEndPan,
      visibleIdsCount: visibleIds.length,
      visibleLastIdsCount: visibleLastIds.length,
      beforeMax,
      started,
      playheadTickAfterWait,
      playheadSecAfterWait,
      ffTick,
      ffSec,
      viewAfterFastForward,
    };
  });

  await browser.close();

  fs.writeFileSync(
    outResultPath,
    JSON.stringify({ report, consoleErrors, pageErrors }, null, 2)
  );
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

    playback = result["playbackReport"]
    assert playback is not None
    assert playback["playAllEnabledWithoutMidi"] is True
    assert playback["playSelectedEnabledWithoutMidi"] is True
    assert playback["playRegionEnabledWithoutMidi"] is True
    assert playback["selectedState"]["play_selected_enabled"] is True
    assert playback["regionState"]["play_region_enabled"] is True
    assert playback["selectedState"]["can_send_midi_out"] is False
    assert playback["startedFollowOn"] is True
    assert playback["visualAtStartFollowOn"]["isPlaying"] is True
    assert playback["stopEnabledDuringFollowOn"] is True
    assert playback["sawActiveHighlight"] is True
    assert "Visual playback only. Enable MIDI Out for external sound." in playback["statusAfterPlayAllNoMidi"]

    if playback["viewportBeforeFollowOn"]["xOffsetTicks"] > 1:
      assert playback["viewportDuringFollowOn"]["xOffsetTicks"] < playback["viewportBeforeFollowOn"]["xOffsetTicks"]

    assert playback["startedFollowOff"] is True
    assert "Visual playback only. Enable MIDI Out for external sound." in playback["statusAfterRegionNoMidi"]
    assert abs(
      playback["viewportDuringFollowOff"]["xOffsetTicks"]
      - playback["viewportBeforeFollowOff"]["xOffsetTicks"]
    ) < 1
    assert playback["visualAfterStopNoMidi"]["isPlaying"] is False
    assert playback["visualAfterStopNoMidi"]["activeNoteIds"] == []
    assert playback["visualAfterFollowOffStop"]["isPlaying"] is False
    assert playback["visualAfterFollowOffStop"]["activeNoteIds"] == []
    assert playback["midiStateBefore"]["can_send_midi_out"] is True
    assert playback["startedWithMidi"] is True
    assert playback["midiStateAfter"]["test_sent_message_count"] > playback["midiStateBefore"]["test_sent_message_count"]

    copy_paste = result["copyPasteReport"]
    assert "Copied" in copy_paste["statusAfterCopy"]
    assert "Pasted" in copy_paste["statusAfterFirstPaste"]
    assert copy_paste["clipboard"]["note_count"] >= 1
    assert copy_paste["afterFirstPasteCount"] > copy_paste["beforePasteCount"]
    assert copy_paste["firstPasteIdsUnique"] is True
    assert copy_paste["noteBoxContainsAllFirstPasteIds"] is True
    assert copy_paste["velocityBarContainsAllFirstPasteIds"] is True
    assert len(copy_paste["selectedAfterFirstPaste"]) == len(copy_paste["addedAfterFirstPaste"])
    assert copy_paste["cursorAfterSecondPaste"] == (
      copy_paste["cursorAfterFirstPaste"] + copy_paste["clipboard"]["region_duration_ticks"]
    )
    assert copy_paste["afterFarPasteOffset"] >= copy_paste["beforeFarPasteOffset"]

    added_first = copy_paste["addedAfterFirstPaste"]
    assert added_first
    assert all(int(note["start_tick"]) >= 0 for note in added_first)
    assert all(int(note["end_tick"]) >= int(note["start_tick"]) for note in added_first)
    assert all(int(note["duration_ticks"]) == int(note["end_tick"]) - int(note["start_tick"]) for note in added_first)

    assert result["velocityBefore"] == 60
    assert result["velocityAfter1px"] == 60
    assert 64 <= result["velocityAfter20pxUp"] <= 66
    assert 54 <= result["velocityAfter20pxDown"] <= 56
    assert 62 <= result["velocityAfter20pxUpShift"] <= 63

    loop_report = result["loopReport"]
    assert loop_report is not None
    assert loop_report["region"] is not None
    assert "Looped" in loop_report["statusAfterRepeatOne"]
    assert "Looped" in loop_report["statusAfterRepeatThree"]

    repeat_one = loop_report["repeatOneResult"]
    repeat_three = loop_report["repeatThreeResult"]
    assert repeat_one["repeats"] == 1
    assert repeat_three["repeats"] == 3
    assert repeat_one["created_count"] > 0
    assert repeat_three["created_count"] > repeat_one["created_count"]

    assert loop_report["afterRepeatOneCount"] > loop_report["beforeCount"]
    assert loop_report["afterUndoCount"] == loop_report["beforeCount"]
    assert loop_report["afterRedoCount"] == loop_report["afterRepeatOneCount"]
    assert loop_report["afterRepeatThreeCount"] > loop_report["afterRedoCount"]

    created_one = loop_report["createdNotesOne"]
    assert created_one
    created_one_ids = [str(note["note_id"]) for note in created_one]
    assert len(created_one_ids) == len(set(created_one_ids))

    note_box_ids = {str(item["note_id"]) for item in loop_report["noteBoxesAfterRepeatOne"]}
    velocity_bar_ids = {str(item["note_id"]) for item in loop_report["velocityBarsAfterRepeatOne"]}
    assert all(note_id in note_box_ids for note_id in created_one_ids)
    assert all(note_id in velocity_bar_ids for note_id in created_one_ids)

    selected_after_repeat_one = {str(note_id) for note_id in loop_report["selectedAfterRepeatOne"]}
    assert all(note_id in selected_after_repeat_one for note_id in created_one_ids)

    region = loop_report["region"]
    region_duration = int(region["duration_ticks"])
    assert region_duration >= 1

    source_session = result["secondSessionNotes"]
    assert source_session > 0
    source_notes = []
    # Pull first two source notes from runtime session result through loop report relationship.
    # Repeat one should place notes exactly one region duration after their source offset.
    for note in created_one:
      assert int(note["duration_ticks"]) == int(note["end_tick"]) - int(note["start_tick"])

    assert loop_report["requestedTooHigh"] == 64
    assert loop_report["clampedRepeat"] == 64

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


def test_browser_long_session_viewport_uses_full_session_range(tmp_path: Path) -> None:
    midi_path = tmp_path / "browser_long_viewport.mid"
    _write_long_browser_test_midi(midi_path)

    node_path = shutil.which("node")
    assert node_path is not None

    result_path = tmp_path / "browser_long_viewport_result.json"
    script_path = tmp_path / "browser_long_viewport.js"
    script_path.write_text(_node_long_viewport_script_content() + "\n", encoding="utf-8")

    server, thread, base_url = _build_server()
    try:
        completed = subprocess.run(
            [
                node_path,
                str(script_path),
                base_url,
                str(midi_path),
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
        "Browser long-session viewport script failed\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["consoleErrors"] == []
    assert payload["pageErrors"] == []

    report = payload["report"]
    assert report["sessionNoteCount"] >= 6
    assert report["sessionMaxByData"] >= 231360
    assert report["sessionMaxByApi"] >= report["sessionMaxByData"]
    assert report["viewportMaxOffset"] > 0
    assert report["viewportMaxOffset"] >= report["expectedMinOffset"] - 1
    assert report["offsetNearEnd"] > 0
    assert report["offsetNearEnd"] <= report["viewportMaxOffset"] + 1
    assert report["nearEndContainsLast"] is True
    assert report["noteBoxesNearEndCount"] >= 1

    assert report["playAllEventCount"] >= report["sessionNoteCount"]
    assert report["playAllMaxEndTick"] >= report["sessionMaxByData"]

    assert report["startedFollowOn"] is True
    assert "Visual playback only. Enable MIDI Out for external sound." in report["statusAfterFollowOnPlay"]
    assert report["sawActiveFollowOn"] is True
    assert report["followOnOffsetDelta"] > 100
    assert report["visualDuringFollowOn"]["isPlaying"] is True
    assert report["visualDuringFollowOn"]["followPlayhead"] is True
    assert report["visualDuringFollowOn"]["currentTick"] is not None
    assert report["globalMaxBeforeFollowOn"] == pytest.approx(report["globalMaxDuringFollowOn"], abs=1e-6)
    assert report["globalMaxBeforeFollowOn"] == pytest.approx(report["globalMaxAfterFollowOnStop"], abs=1e-6)
    assert report["visualAfterFollowOnStop"]["isPlaying"] is False
    assert report["visualAfterFollowOnStop"]["currentTick"] is None
    assert report["visualAfterFollowOnStop"]["activeNoteIds"] == []
    assert report["visualAfterFollowOnStop"]["animationFrameActive"] is False

    assert report["startedFollowOff"] is True
    assert "Visual playback only. Enable MIDI Out for external sound." in report["statusAfterFollowOffPlay"]
    assert report["sawActiveFollowOff"] is True
    assert report["manualPanChanged"] is True
    assert report["manualPanDelta"] == pytest.approx(4000, abs=1)
    assert abs(report["followOffOffsetDrift"]) < 1
    assert report["visualDuringFollowOff"]["isPlaying"] is True
    assert report["visualDuringFollowOff"]["followPlayhead"] is False
    assert report["visualDuringFollowOff"]["currentTick"] is not None
    assert report["globalMaxBeforeFollowOff"] == pytest.approx(report["globalMaxDuringFollowOff"], abs=1e-6)
    assert report["globalMaxBeforeFollowOff"] == pytest.approx(report["globalMaxAfterFollowOffStop"], abs=1e-6)
    assert report["visualAfterFollowOffStop"]["isPlaying"] is False
    assert report["visualAfterFollowOffStop"]["currentTick"] is None
    assert report["visualAfterFollowOffStop"]["activeNoteIds"] == []
    assert report["visualAfterFollowOffStop"]["animationFrameActive"] is False

    assert report["startedRegion"] is True
    assert report["regionEventMaxEndTick"] >= report["regionStart"]
    assert report["regionEventMaxEndTick"] <= report["sessionMaxByApi"]
    assert report["visualAtRegionStart"]["playbackEndTick"] <= report["sessionMaxByApi"]
    assert report["globalMaxBeforeRegionPlay"] == pytest.approx(report["globalMaxDuringRegionPlay"], abs=1e-6)


def test_browser_import_preserves_long_invalid_key_signature_over_40_seconds(tmp_path: Path) -> None:
    midi_path = tmp_path / "Played_With_Fire_-_Deep_House__Synth___Synth_.mid"
    _write_long_invalid_key_signature_browser_midi(midi_path)

    node_path = shutil.which("node")
    assert node_path is not None

    result_path = tmp_path / "browser_long_user_file_result.json"
    script_path = tmp_path / "browser_long_user_file.js"
    script_path.write_text(_node_long_user_file_script_content() + "\n", encoding="utf-8")

    server, thread, base_url = _build_server()
    try:
        completed = subprocess.run(
            [
                node_path,
                str(script_path),
                base_url,
                str(midi_path),
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
        "Browser long malformed-key-signature script failed\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["consoleErrors"] == []
    assert payload["pageErrors"] == []

    report = payload["report"]
    assert report["sourceMidi"] == "Played_With_Fire_-_Deep_House__Synth___Synth_.mid"
    assert report["noteCount"] >= 8
    assert report["maxEndTick"] > 38400
    assert report["maxEndSec"] > 40.0
    assert report["notesAfter40"] >= 1
    assert report["visibleIdsCount"] >= 1
    assert report["visibleLastIdsCount"] >= 1
    assert report["viewAfterEndPan"]["xOffsetTicks"] > 0

    assert report["started"] is True
    assert report["beforeMax"] == pytest.approx(report["maxEndTick"], abs=1e-6)
    assert report["ffTick"] > 0
    assert report["ffSec"] > 40.0
    assert report["viewAfterFastForward"]["xOffsetTicks"] > 0
