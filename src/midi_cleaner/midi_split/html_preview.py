from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from midi_cleaner.midi_split.models import MidiSplitSession


def render_piano_roll_preview_html(session: MidiSplitSession) -> str:
    payload = session.model_dump(mode="json")
    payload_json = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")
    build_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="midi-split-editor-build" content="__BUILD_ID__" />
  <title>MIDI Split Editor Preview</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #161616;
      color: #f5f5f5;
    }
    #toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 8px;
      background: #222;
      border-bottom: 1px solid #444;
      font-size: 13px;
    }
    #toolbar button,
    #toolbar .toolbar-button,
    #toolbar select {
      padding: 4px 8px;
      font-size: 12px;
      border: 1px solid #666;
      background: #2e2e2e;
      color: #f5f5f5;
      cursor: pointer;
      text-decoration: none;
    }
    #toolbar .toolbar-button {
      display: inline-block;
      line-height: 1.2;
    }
    #toolbar button.active-tool {
      border-color: #9ab8ff;
      background: #3b4d7d;
    }
    #toolbar button:disabled,
    #toolbar .toolbar-button.disabled-control {
      opacity: 0.45;
      cursor: not-allowed;
      pointer-events: none;
    }
    #server-status {
      font-size: 12px;
      color: #b9d0ff;
    }
    #build-id {
      margin-left: auto;
      font-size: 11px;
      color: #aaaaaa;
    }
    #status-line {
      padding: 4px 8px;
      border-bottom: 1px solid #333;
      min-height: 24px;
      font-size: 12px;
      color: #d9d9d9;
      background: #1d1d1d;
    }
    #status-line.error {
      color: #ffb3b3;
    }
    #layout {
      display: flex;
      height: calc(100vh - 88px);
      min-height: 420px;
    }
    #track-panel {
      width: 280px;
      min-width: 220px;
      max-width: 340px;
      overflow: auto;
      border-right: 1px solid #333;
      background: #1e1e1e;
      padding: 8px;
      font-size: 12px;
    }
    .track-row {
      display: grid;
      grid-template-columns: 24px 1fr auto;
      gap: 6px;
      align-items: center;
      margin-bottom: 6px;
      padding: 6px;
      border: 1px solid #333;
      background: #242424;
    }
    .track-swatch {
      width: 10px;
      height: 10px;
      border: 1px solid #000;
      margin-right: 6px;
      display: inline-block;
      vertical-align: middle;
    }
    .track-label {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    #roll-wrap {
      flex: 1;
      overflow: auto;
      background: #111;
      position: relative;
    }
    #roll-canvas {
      display: block;
      background: #111;
      cursor: crosshair;
    }
  </style>
</head>
<body>
  <div id="toolbar">
    <strong>Source:</strong>
    <span id="source-name">-</span>
    <span>|</span>
    <span>Selected notes: <strong id="selected-count">0</strong></span>
    <span>|</span>
    <span id="server-status">Server: checking</span>

    <label id="import-midi-label" class="toolbar-button" for="import-midi-input">Import MIDI</label>
    <button id="export-multitrack-btn" type="button">Export Multitrack MIDI</button>
    <button id="export-separate-btn" type="button">Export Separate Tracks ZIP</button>
    <button id="undo-btn" type="button" disabled>Undo</button>
    <button id="redo-btn" type="button" disabled>Redo</button>

    <span>|</span>
    <button id="tool-select-btn" type="button">Select</button>
    <button id="tool-zoom-btn" type="button">Zoom</button>
    <button id="tool-pan-btn" type="button">Hand</button>

    <span>|</span>
    <label for="target-track">Target track:</label>
    <select id="target-track"></select>
    <button id="move-selected-btn" type="button">Move selected to track</button>
    <button id="merge-notes-btn" type="button">Merge Notes</button>
    <button id="add-track-btn" type="button">Add track</button>
    <button id="merge-tracks-btn" type="button">Merge selected tracks</button>
    <button id="clear-selection-btn" type="button">Clear selection</button>
    <span id="build-id">Build: __BUILD_ID__</span>
    <input id="import-midi-input" type="file" accept=".mid,.midi,audio/midi,audio/x-midi" hidden />
  </div>
  <div id="status-line">Ready.</div>
  <div id="layout">
    <aside id="track-panel"></aside>
    <main id="roll-wrap">
      <canvas id="roll-canvas"></canvas>
    </main>
  </div>

  <script id="session-json" type="application/json">__SESSION_JSON__</script>
  <script>
    (function () {
      "use strict";

      const MAX_TRACKS = 12;
      const HISTORY_LIMIT = 100;
      const LEFT_PAD = 72;
      const TOP_PAD = 52;
      const RIGHT_PAD = 24;
      const BOTTOM_PAD = 24;
      const RULER_BAR_TEXT_Y = 12;
      const RULER_BEAT_TEXT_Y = 24;
      const RULER_TIME_TEXT_Y = 36;
      const RULER_DIVIDER_Y = 42;
      const MIN_CANVAS_WIDTH = 1000;
      const NOTE_ROW_HEIGHT = 10;
      const DEFAULT_PIXELS_PER_TICK = 0.18;
      const MIN_PIXELS_PER_TICK = 0.03;
      const MAX_PIXELS_PER_TICK = 2.50;
      const ZOOM_IN_FACTOR = 1.20;
      const ZOOM_OUT_FACTOR = 1 / 1.20;
      const DRAG_THRESHOLD_PX = 4;
      const DEFAULT_TEMPO_US_PER_BEAT = 500000;
      const DEFAULT_TIME_SIGNATURE_NUMERATOR = 4;
      const DEFAULT_TIME_SIGNATURE_DENOMINATOR = 4;
      const palette = [
        "#ff6f61", "#6fcf97", "#56ccf2", "#f2c94c", "#bb6bd9", "#f2994a",
        "#2d9cdb", "#9b51e0", "#27ae60", "#eb5757", "#219ebc", "#f77f00"
      ];

      function normalizeSession(value) {
        const session = value && typeof value === "object" ? value : {};
        if (!Array.isArray(session.tracks)) {
          session.tracks = [];
        }
        if (!Array.isArray(session.notes)) {
          session.notes = [];
        }
        if (!Array.isArray(session.tempo_map)) {
          session.tempo_map = [];
        }
        if (Array.isArray(session.time_signature_map)) {
          session.time_signature_map = session.time_signature_map;
        } else if (Array.isArray(session.time_signatures)) {
          session.time_signature_map = session.time_signatures;
        } else {
          session.time_signature_map = [];
        }
        if (typeof session.schema_version !== "string") {
          session.schema_version = "0.1.0";
        }
        if (typeof session.source_midi !== "string") {
          session.source_midi = "";
        }
        if (typeof session.source !== "string") {
          session.source = "manual";
        }
        if (typeof session.layer !== "string") {
          session.layer = "midi";
        }
        if (!Number.isFinite(Number(session.ticks_per_beat)) || Number(session.ticks_per_beat) <= 0) {
          session.ticks_per_beat = 480;
        } else {
          session.ticks_per_beat = Number(session.ticks_per_beat);
        }
        return session;
      }

      let session = normalizeSession(JSON.parse(document.getElementById("session-json").textContent));
      let currentTool = "select";

      const sourceNameEl = document.getElementById("source-name");
      const selectedCountEl = document.getElementById("selected-count");
      const targetTrackEl = document.getElementById("target-track");
      const statusEl = document.getElementById("status-line");
      const serverStatusEl = document.getElementById("server-status");
      const trackPanelEl = document.getElementById("track-panel");
      const rollWrapEl = document.getElementById("roll-wrap");
      const canvas = document.getElementById("roll-canvas");
      const ctx = canvas.getContext("2d");
      const importInput = document.getElementById("import-midi-input");
      const importLabel = document.getElementById("import-midi-label");
      const exportMultitrackButton = document.getElementById("export-multitrack-btn");
      const exportSeparateButton = document.getElementById("export-separate-btn");
      const undoButton = document.getElementById("undo-btn");
      const redoButton = document.getElementById("redo-btn");
      const mergeNotesButton = document.getElementById("merge-notes-btn");

      const toolButtons = {
        select: document.getElementById("tool-select-btn"),
        zoom: document.getElementById("tool-zoom-btn"),
        pan: document.getElementById("tool-pan-btn"),
      };

      const serverActionControls = [
        importLabel,
        exportMultitrackButton,
        exportSeparateButton,
      ];

      const state = {
        selectedNoteIds: new Set(),
        mergeTrackIndices: new Set(),
        noteById: new Map(),
        noteBoxes: [],
        dragSelect: null,
        panDrag: null,
        pitchMin: 24,
        pitchMax: 108,
        pixelsPerTick: DEFAULT_PIXELS_PER_TICK,
        xOffsetTicks: 0,
        serverConnected: false,
        undoStack: [],
        redoStack: [],
        normalizedTempoMap: [],
        normalizedTimeSignatureMap: [],
      };

      function setStatus(text, isError) {
        const hasError = Boolean(isError);
        statusEl.classList.toggle("error", hasError);
        statusEl.textContent = text;
      }

      function setErrorStatus(text, details) {
        if (details !== undefined) {
          console.error("MIDI split editor error:", details);
        }
        setStatus(text, true);
      }

      function setServerStatus(connected, label) {
        state.serverConnected = Boolean(connected);
        if (state.serverConnected) {
          serverStatusEl.textContent = "Server: connected";
          serverStatusEl.style.color = "#8be0a5";
          return;
        }
        const text = String(label || "Server: unavailable");
        serverStatusEl.textContent = text;
        serverStatusEl.style.color = "#ffb3b3";
      }

      function setServerActionControlsEnabled(enabled, disabledReason) {
        const isEnabled = Boolean(enabled);
        const reason = String(disabledReason || "");

        for (const control of serverActionControls) {
          if (control instanceof HTMLButtonElement) {
            control.disabled = !isEnabled;
          } else {
            control.classList.toggle("disabled-control", !isEnabled);
            control.setAttribute("aria-disabled", String(!isEnabled));
          }

          if (reason) {
            control.title = reason;
          } else {
            control.removeAttribute("title");
          }
        }

        importInput.disabled = !isEnabled;
      }

      function cloneJsonValue(value) {
        return JSON.parse(JSON.stringify(value));
      }

      function cloneNoteSnapshot(note) {
        return cloneJsonValue(note);
      }

      function normalizeTempoMapEvents() {
        const tickToTempo = new Map();
        for (const event of Array.isArray(session.tempo_map) ? session.tempo_map : []) {
          const tick = Math.max(0, Number(event && event.tick));
          let tempo = Number(event && event.tempo_us_per_beat);
          if (!Number.isFinite(tempo) || tempo <= 0) {
            tempo = DEFAULT_TEMPO_US_PER_BEAT;
          }
          if (Number.isFinite(tick)) {
            tickToTempo.set(Math.floor(tick), Math.floor(tempo));
          }
        }

        if (!tickToTempo.has(0)) {
          tickToTempo.set(0, DEFAULT_TEMPO_US_PER_BEAT);
        }

        const sorted = Array.from(tickToTempo.entries())
          .map(function (item) {
            return { tick: Number(item[0]), tempo_us_per_beat: Number(item[1]), sec: 0 };
          })
          .sort(function (a, b) {
            return a.tick - b.tick;
          });

        let currentSec = 0;
        const ticksPerBeat = Math.max(1, Number(session.ticks_per_beat || 480));
        for (let index = 0; index < sorted.length; index += 1) {
          if (index > 0) {
            const prev = sorted[index - 1];
            const deltaTicks = sorted[index].tick - prev.tick;
            currentSec += (deltaTicks / ticksPerBeat) * (prev.tempo_us_per_beat / 1000000);
          }
          sorted[index].sec = currentSec;
        }

        return sorted;
      }

      function normalizeTimeSignatureEvents() {
        const tickToSignature = new Map();
        const rawMap = Array.isArray(session.time_signature_map) ? session.time_signature_map : [];
        for (const event of rawMap) {
          const tick = Math.max(0, Number(event && event.tick));
          let numerator = Number(event && event.numerator);
          let denominator = Number(event && event.denominator);
          if (!Number.isFinite(numerator) || numerator <= 0) {
            numerator = DEFAULT_TIME_SIGNATURE_NUMERATOR;
          }
          if (!Number.isFinite(denominator) || denominator <= 0) {
            denominator = DEFAULT_TIME_SIGNATURE_DENOMINATOR;
          }
          if (Number.isFinite(tick)) {
            tickToSignature.set(Math.floor(tick), {
              tick: Math.floor(tick),
              numerator: Math.max(1, Math.floor(numerator)),
              denominator: Math.max(1, Math.floor(denominator)),
            });
          }
        }

        if (!tickToSignature.has(0)) {
          tickToSignature.set(0, {
            tick: 0,
            numerator: DEFAULT_TIME_SIGNATURE_NUMERATOR,
            denominator: DEFAULT_TIME_SIGNATURE_DENOMINATOR,
          });
        }

        return Array.from(tickToSignature.values()).sort(function (a, b) {
          return a.tick - b.tick;
        });
      }

      function refreshTimingCaches() {
        state.normalizedTempoMap = normalizeTempoMapEvents();
        state.normalizedTimeSignatureMap = normalizeTimeSignatureEvents();
      }

      function tickToSeconds(tick) {
        const targetTick = Math.max(0, Number(tick || 0));
        const tempoMap = state.normalizedTempoMap;
        if (!tempoMap.length) {
          return (targetTick / Math.max(1, Number(session.ticks_per_beat || 480))) * 0.5;
        }

        let selected = tempoMap[0];
        for (const event of tempoMap) {
          if (event.tick <= targetTick) {
            selected = event;
          } else {
            break;
          }
        }

        const deltaTicks = targetTick - selected.tick;
        return selected.sec + (deltaTicks / Math.max(1, Number(session.ticks_per_beat || 480))) * (selected.tempo_us_per_beat / 1000000);
      }

      function secondsToTick(seconds) {
        const targetSec = Math.max(0, Number(seconds || 0));
        const tempoMap = state.normalizedTempoMap;
        if (!tempoMap.length) {
          return targetSec * Math.max(1, Number(session.ticks_per_beat || 480)) * 2;
        }

        let selected = tempoMap[0];
        for (const event of tempoMap) {
          if (event.sec <= targetSec) {
            selected = event;
          } else {
            break;
          }
        }

        const ticksPerSecond = (Math.max(1, Number(session.ticks_per_beat || 480)) * 1000000) / Math.max(1, selected.tempo_us_per_beat);
        return selected.tick + (targetSec - selected.sec) * ticksPerSecond;
      }

      function formatElapsedTime(seconds) {
        const totalSeconds = Math.max(0, Math.round(Number(seconds || 0)));
        const minutes = Math.floor(totalSeconds / 60);
        const secs = totalSeconds % 60;
        return String(minutes).padStart(2, "0") + ":" + String(secs).padStart(2, "0");
      }

      function chooseTimeLabelStepSeconds(pxPerSecond) {
        const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300];
        for (const step of steps) {
          if (step * pxPerSecond >= 72) {
            return step;
          }
        }
        return 600;
      }

      function rebuildNoteLookup() {
        state.noteById.clear();
        for (const note of session.notes) {
          state.noteById.set(String(note.note_id), note);
        }
      }

      function sanitizeSelection() {
        for (const noteId of Array.from(state.selectedNoteIds)) {
          if (!state.noteById.has(String(noteId))) {
            state.selectedNoteIds.delete(String(noteId));
          }
        }
      }

      function setSelectionFromList(noteIds) {
        state.selectedNoteIds.clear();
        for (const noteId of noteIds || []) {
          const normalized = String(noteId);
          if (state.noteById.has(normalized)) {
            state.selectedNoteIds.add(normalized);
          }
        }
        updateSelectionUi();
      }

      function updateHistoryControls() {
        undoButton.disabled = state.undoStack.length === 0;
        redoButton.disabled = state.redoStack.length === 0;
      }

      function updateMergeButtonState() {
        mergeNotesButton.disabled = state.selectedNoteIds.size < 2;
      }

      function updateEditorActionButtons() {
        updateHistoryControls();
        updateMergeButtonState();
      }

      function pushHistoryTransaction(transaction) {
        state.undoStack.push(transaction);
        if (state.undoStack.length > HISTORY_LIMIT) {
          state.undoStack.shift();
        }
        state.redoStack = [];
        updateHistoryControls();
      }

      function clearHistory() {
        state.undoStack = [];
        state.redoStack = [];
        updateHistoryControls();
      }

      function applyHistoryTransaction(transaction, useAfter) {
        const beforeNotes = Array.isArray(transaction.beforeNotes) ? transaction.beforeNotes : [];
        const afterNotes = Array.isArray(transaction.afterNotes) ? transaction.afterNotes : [];
        const targetNotes = useAfter ? afterNotes : beforeNotes;
        const affectedIds = new Set();

        for (const note of beforeNotes) {
          affectedIds.add(String(note.note_id));
        }
        for (const note of afterNotes) {
          affectedIds.add(String(note.note_id));
        }

        session.notes = session.notes.filter(function (note) {
          return !affectedIds.has(String(note.note_id));
        });

        for (const snapshot of targetNotes) {
          session.notes.push(cloneNoteSnapshot(snapshot));
        }

        sortNotes();
        rebuildTrackSources();
        rebuildNoteLookup();

        const desiredSelection = useAfter
          ? (Array.isArray(transaction.selectionAfter) ? transaction.selectionAfter : [])
          : (Array.isArray(transaction.selectionBefore) ? transaction.selectionBefore : []);
        setSelectionFromList(desiredSelection);

        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        updateEditorActionButtons();
      }

      function undoHistory() {
        if (state.undoStack.length === 0) {
          return;
        }
        const transaction = state.undoStack.pop();
        state.redoStack.push(transaction);
        applyHistoryTransaction(transaction, false);
        updateHistoryControls();
        setStatus("Undo complete.", false);
      }

      function redoHistory() {
        if (state.redoStack.length === 0) {
          return;
        }
        const transaction = state.redoStack.pop();
        state.undoStack.push(transaction);
        applyHistoryTransaction(transaction, true);
        updateHistoryControls();
        setStatus("Redo complete.", false);
      }

      function generateUniqueMergedNoteId(baseNoteId) {
        const existing = new Set(session.notes.map(function (note) {
          return String(note.note_id);
        }));
        const token = String(baseNoteId || "note");
        let candidate = token + "_merged";
        let ordinal = 1;
        while (existing.has(candidate)) {
          candidate = token + "_merged_" + String(ordinal).padStart(3, "0");
          ordinal += 1;
        }
        return candidate;
      }

      function syncNoteTimingFromTicks(note) {
        note.start_tick = Math.max(0, Math.round(Number(note.start_tick || 0)));
        note.end_tick = Math.max(note.start_tick, Math.round(Number(note.end_tick || note.start_tick)));
        note.duration_ticks = Math.max(0, note.end_tick - note.start_tick);
        note.start_sec = tickToSeconds(note.start_tick);
        note.end_sec = tickToSeconds(note.end_tick);
        note.duration_sec = Math.max(0, note.end_sec - note.start_sec);
      }

      function validateMergeSelection(orderedNotes) {
        if (orderedNotes.length < 2) {
          return "Select at least two notes to merge.";
        }

        const first = orderedNotes[0];
        const firstPitch = Number(first.pitch_midi);
        const firstTrack = Number(first.editable_track_index);
        const firstChannel = first.channel;

        for (let index = 1; index < orderedNotes.length; index += 1) {
          const note = orderedNotes[index];
          if (Number(note.pitch_midi) !== firstPitch) {
            return "Selected notes must have the same pitch.";
          }
          if (Number(note.editable_track_index) !== firstTrack) {
            return "Selected notes must belong to the same track.";
          }
          if (note.channel !== firstChannel) {
            return "Selected notes must use the same MIDI channel.";
          }
        }

        return "";
      }

      function mergeSelectedNotes() {
        const orderedSelection = Array.from(state.selectedNoteIds)
          .map(function (noteId) {
            return state.noteById.get(String(noteId));
          })
          .filter(function (note) { return Boolean(note); })
          .sort(function (a, b) {
            if (Number(a.start_tick) !== Number(b.start_tick)) {
              return Number(a.start_tick) - Number(b.start_tick);
            }
            if (Number(a.end_tick) !== Number(b.end_tick)) {
              return Number(a.end_tick) - Number(b.end_tick);
            }
            return String(a.note_id).localeCompare(String(b.note_id));
          });

        const validationMessage = validateMergeSelection(orderedSelection);
        if (validationMessage) {
          setStatus(validationMessage);
          return;
        }

        const selectionBefore = orderedSelection.map(function (note) {
          return String(note.note_id);
        });
        const beforeNotes = orderedSelection.map(function (note) {
          return cloneNoteSnapshot(note);
        });

        const earliest = cloneNoteSnapshot(orderedSelection[0]);
        const mergedStartTick = Math.min.apply(null, orderedSelection.map(function (note) { return Number(note.start_tick || 0); }));
        const mergedEndTick = Math.max.apply(null, orderedSelection.map(function (note) { return Number(note.end_tick || 0); }));

        earliest.note_id = generateUniqueMergedNoteId(String(orderedSelection[0].note_id));
        earliest.start_tick = mergedStartTick;
        earliest.end_tick = mergedEndTick;
        syncNoteTimingFromTicks(earliest);

        const selectedIdSet = new Set(selectionBefore);
        session.notes = session.notes.filter(function (note) {
          return !selectedIdSet.has(String(note.note_id));
        });
        session.notes.push(earliest);

        pushHistoryTransaction({
          label: "merge-notes",
          beforeNotes: beforeNotes,
          afterNotes: [cloneNoteSnapshot(earliest)],
          selectionBefore: selectionBefore,
          selectionAfter: [String(earliest.note_id)],
        });

        sortNotes();
        rebuildTrackSources();
        rebuildNoteLookup();
        setSelectionFromList([String(earliest.note_id)]);
        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        updateEditorActionButtons();
        setStatus("Merged " + String(orderedSelection.length) + " notes into one note.", false);
      }

      function isEditableTypingTarget(target) {
        if (!(target instanceof Element)) {
          return false;
        }
        if (target instanceof HTMLInputElement) {
          return true;
        }
        if (target instanceof HTMLTextAreaElement) {
          return true;
        }
        if (target instanceof HTMLSelectElement) {
          return true;
        }
        return Boolean(target.isContentEditable);
      }

      function ensureUsableSessionPayload(payload, operationLabel) {
        if (!payload || typeof payload !== "object") {
          throw new Error(operationLabel + " failed: empty JSON response.");
        }
        if (!Array.isArray(payload.tracks) || !Array.isArray(payload.notes)) {
          throw new Error(operationLabel + " failed: response does not contain tracks/notes.");
        }
      }

      async function checkServerConnection() {
        setServerStatus(false, "Server: checking");
        setStatus("Checking server connection...", false);

        try {
          const response = await fetch("/api/session", {
            method: "GET",
            cache: "no-store",
          });
          const payload = await parseJsonResponse(response, "Server session check");
          ensureUsableSessionPayload(payload, "Server session check");
          setServerActionControlsEnabled(true, "");
          setServerStatus(true);
          setStatus("Server connected.", false);
          return payload;
        } catch (error) {
          const message = "Server connection unavailable. Start the editor with: midi-cleaner midi split-editor";
          setServerActionControlsEnabled(false, message);
          setServerStatus(false);
          setErrorStatus(message, error);
          return null;
        }
      }

      async function parseJsonResponse(response, operationLabel) {
        const raw = await response.text();
        let payload = null;
        if (raw.trim()) {
          try {
            payload = JSON.parse(raw);
          } catch (error) {
            if (response.ok) {
              throw new Error(operationLabel + " failed: response is not valid JSON.");
            }
          }
        }

        if (!response.ok) {
          const detail = payload && typeof payload.message === "string" && payload.message.trim()
            ? payload.message.trim()
            : (raw.trim() || "unknown server error");
          throw new Error(operationLabel + " failed (HTTP " + String(response.status) + "): " + detail);
        }

        ensureUsableSessionPayload(payload, operationLabel);

        return payload;
      }

      function colorForTrack(trackIndex) {
        const normalized = Math.max(1, Number(trackIndex));
        return palette[(normalized - 1) % palette.length];
      }

      function sortTracks() {
        session.tracks.sort(function (a, b) {
          return Number(a.editable_track_index) - Number(b.editable_track_index);
        });
      }

      function sortNotes() {
        session.notes.sort(function (a, b) {
          if (Number(a.start_tick) !== Number(b.start_tick)) {
            return Number(a.start_tick) - Number(b.start_tick);
          }
          if (Number(a.end_tick) !== Number(b.end_tick)) {
            return Number(a.end_tick) - Number(b.end_tick);
          }
          if (Number(a.editable_track_index) !== Number(b.editable_track_index)) {
            return Number(a.editable_track_index) - Number(b.editable_track_index);
          }
          return String(a.note_id).localeCompare(String(b.note_id));
        });
      }

      function getTrackByIndex(index) {
        for (const track of session.tracks) {
          if (Number(track.editable_track_index) === Number(index)) {
            return track;
          }
        }
        return null;
      }

      function getMaxTick() {
        let maxTick = Math.max(0, Number(session.ticks_per_beat) * 8);
        for (const note of session.notes) {
          maxTick = Math.max(maxTick, Number(note.end_tick || 0));
        }
        return maxTick;
      }

      function getVisibleTickSpan() {
        const widthPx = Math.max(1, canvas.width - LEFT_PAD - RIGHT_PAD);
        return widthPx / state.pixelsPerTick;
      }

      function clampXOffsetTicks() {
        const maxTick = getMaxTick();
        const maxOffset = Math.max(0, maxTick - getVisibleTickSpan() + Number(session.ticks_per_beat || 480) * 2);
        state.xOffsetTicks = Math.max(0, Math.min(state.xOffsetTicks, maxOffset));
      }

      function updatePitchRange() {
        if (session.notes.length === 0) {
          state.pitchMin = 24;
          state.pitchMax = 108;
          return;
        }

        let minPitch = 127;
        let maxPitch = 0;
        for (const note of session.notes) {
          const pitch = Number(note.pitch_midi || 0);
          minPitch = Math.min(minPitch, pitch);
          maxPitch = Math.max(maxPitch, pitch);
        }

        state.pitchMin = Math.max(0, minPitch - 2);
        state.pitchMax = Math.min(127, maxPitch + 2);
      }

      function updateCanvasSize() {
        const width = Math.max(MIN_CANVAS_WIDTH, Number(rollWrapEl.clientWidth || MIN_CANVAS_WIDTH));
        const rows = state.pitchMax - state.pitchMin + 1;
        const height = Math.max(420, Math.ceil(TOP_PAD + rows * NOTE_ROW_HEIGHT + BOTTOM_PAD));
        canvas.width = Math.ceil(width);
        canvas.height = height;
      }

      function yForPitch(pitch) {
        return TOP_PAD + (state.pitchMax - pitch) * NOTE_ROW_HEIGHT;
      }

      function xForTick(tick) {
        return LEFT_PAD + (Number(tick) - state.xOffsetTicks) * state.pixelsPerTick;
      }

      function updateSelectionUi() {
        sanitizeSelection();
        selectedCountEl.textContent = String(state.selectedNoteIds.size);
        updateEditorActionButtons();
      }

      function clearSelection() {
        state.selectedNoteIds.clear();
        updateSelectionUi();
      }

      function buildTrackNoteCounts() {
        const counts = new Map();
        for (const track of session.tracks) {
          counts.set(Number(track.editable_track_index), 0);
        }
        for (const note of session.notes) {
          const index = Number(note.editable_track_index);
          counts.set(index, (counts.get(index) || 0) + 1);
        }
        return counts;
      }

      function rebuildTrackSources() {
        const sourceByTrack = new Map();
        for (const track of session.tracks) {
          sourceByTrack.set(Number(track.editable_track_index), new Set());
        }

        for (const note of session.notes) {
          const trackIndex = Number(note.editable_track_index);
          if (!sourceByTrack.has(trackIndex)) {
            sourceByTrack.set(trackIndex, new Set());
          }
          sourceByTrack.get(trackIndex).add(Number(note.source_track_index));
        }

        for (const track of session.tracks) {
          const values = Array.from(sourceByTrack.get(Number(track.editable_track_index)) || []);
          values.sort(function (a, b) { return a - b; });
          track.source_track_indices = values;
        }
      }

      function renderTrackPanel() {
        const counts = buildTrackNoteCounts();
        trackPanelEl.innerHTML = "";

        for (const track of session.tracks) {
          const row = document.createElement("div");
          row.className = "track-row";

          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.title = "Select track for merge";
          checkbox.checked = state.mergeTrackIndices.has(Number(track.editable_track_index));
          checkbox.addEventListener("change", function () {
            const index = Number(track.editable_track_index);
            if (checkbox.checked) {
              state.mergeTrackIndices.add(index);
            } else {
              state.mergeTrackIndices.delete(index);
            }
          });

          const label = document.createElement("div");
          label.className = "track-label";
          const swatch = document.createElement("span");
          swatch.className = "track-swatch";
          swatch.style.background = colorForTrack(track.editable_track_index);
          label.appendChild(swatch);
          label.appendChild(
            document.createTextNode(
              String(track.editable_track_index).padStart(2, "0") + " - " + String(track.name || "Track")
            )
          );

          const count = document.createElement("div");
          count.textContent = String(counts.get(Number(track.editable_track_index)) || 0) + " notes";

          row.appendChild(checkbox);
          row.appendChild(label);
          row.appendChild(count);
          trackPanelEl.appendChild(row);
        }
      }

      function updateTargetTrackDropdown() {
        const previousValue = Number(targetTrackEl.value || 0);
        targetTrackEl.innerHTML = "";

        for (const track of session.tracks) {
          const option = document.createElement("option");
          option.value = String(track.editable_track_index);
          option.textContent = String(track.editable_track_index).padStart(2, "0") + " - " + String(track.name);
          targetTrackEl.appendChild(option);
        }

        if (getTrackByIndex(previousValue) !== null) {
          targetTrackEl.value = String(previousValue);
        } else if (session.tracks.length > 0) {
          targetTrackEl.value = String(session.tracks[0].editable_track_index);
        }
      }

      function setTool(toolName) {
        currentTool = toolName;
        for (const name of Object.keys(toolButtons)) {
          if (name === toolName) {
            toolButtons[name].classList.add("active-tool");
          } else {
            toolButtons[name].classList.remove("active-tool");
          }
        }

        if (toolName === "select") {
          canvas.style.cursor = "crosshair";
        } else if (toolName === "zoom") {
          canvas.style.cursor = "zoom-in";
        } else {
          canvas.style.cursor = "grab";
        }
      }

      function resetDerivedSessionState() {
        state.mergeTrackIndices.clear();
        state.dragSelect = null;
        state.panDrag = null;
        state.noteBoxes = [];
        state.pixelsPerTick = DEFAULT_PIXELS_PER_TICK;
        state.xOffsetTicks = 0;
        clearSelection();
      }

      function renderAll() {
        sourceNameEl.textContent = String(session.source_midi || "-");
        updateTargetTrackDropdown();
        renderTrackPanel();
        updateSelectionUi();
        redraw();
      }

      function setSessionData(newSession) {
        session = normalizeSession(newSession);
        sortTracks();
        sortNotes();
        rebuildTrackSources();
        refreshTimingCaches();
        rebuildNoteLookup();
        clearHistory();
        resetDerivedSessionState();
        updateEditorActionButtons();
      }

      function fitImportedNotesToView() {
        if (!session.notes.length) {
          state.pixelsPerTick = DEFAULT_PIXELS_PER_TICK;
          state.xOffsetTicks = 0;
          return;
        }

        let minStartTick = Number.POSITIVE_INFINITY;
        let maxEndTick = 0;
        for (const note of session.notes) {
          minStartTick = Math.min(minStartTick, Number(note.start_tick || 0));
          maxEndTick = Math.max(maxEndTick, Number(note.end_tick || 0));
        }

        const ticksPerBeat = Math.max(1, Number(session.ticks_per_beat || 480));
        const widthPx = Math.max(120, Number(rollWrapEl.clientWidth || MIN_CANVAS_WIDTH) - LEFT_PAD - RIGHT_PAD);
        const visibleSpanTicks = Math.max(ticksPerBeat, maxEndTick - minStartTick + ticksPerBeat);
        const suggestedPixelsPerTick = widthPx / visibleSpanTicks;

        state.pixelsPerTick = Math.max(
          MIN_PIXELS_PER_TICK,
          Math.min(MAX_PIXELS_PER_TICK, suggestedPixelsPerTick)
        );
        state.xOffsetTicks = Math.max(0, minStartTick - ticksPerBeat * 0.5);
        clampXOffsetTicks();
      }

      function applyImportedSession(payload) {
        setStatus("Replacing editor session", false);
        setSessionData(payload);
        fitImportedNotesToView();
        renderAll();
      }

      function rebuildNoteBoxes() {
        state.noteBoxes = [];
        for (const note of session.notes) {
          const pitch = Number(note.pitch_midi || 0);
          if (pitch < state.pitchMin || pitch > state.pitchMax) {
            continue;
          }

          const startTick = Number(note.start_tick || 0);
          const endTick = Math.max(startTick, Number(note.end_tick || startTick));
          const x = xForTick(startTick);
          const endX = xForTick(endTick);
          const w = Math.max(1, endX - x);
          const h = Math.max(3, NOTE_ROW_HEIGHT - 2);
          const y = yForPitch(pitch) + 1;

          if (x + w < LEFT_PAD - 2 || x > canvas.width - RIGHT_PAD + 2) {
            continue;
          }

          state.noteBoxes.push({
            x: x,
            y: y,
            w: w,
            h: h,
            note: note,
          });
        }
      }

      function ticksPerSignatureBeat(denominator) {
        return Math.max(1, Number(session.ticks_per_beat || 480) * (4 / Math.max(1, Number(denominator || 4))));
      }

      function buildTimelineMarkers(startTick, endTick) {
        const bars = [];
        const beats = [];
        const signatures = state.normalizedTimeSignatureMap;

        let barNumber = 1;
        let barStartTick = 0;
        let signatureIndex = 0;

        for (let iteration = 0; iteration < 100000; iteration += 1) {
          while (
            signatureIndex + 1 < signatures.length
            && Number(signatures[signatureIndex + 1].tick) <= barStartTick + 0.0001
          ) {
            signatureIndex += 1;
          }

          const signature = signatures[signatureIndex] || {
            tick: 0,
            numerator: DEFAULT_TIME_SIGNATURE_NUMERATOR,
            denominator: DEFAULT_TIME_SIGNATURE_DENOMINATOR,
          };

          const numerator = Math.max(1, Number(signature.numerator || DEFAULT_TIME_SIGNATURE_NUMERATOR));
          const ticksPerBeat = ticksPerSignatureBeat(signature.denominator);
          const ticksPerBar = Math.max(1, ticksPerBeat * numerator);

          const nextSignatureTick = signatureIndex + 1 < signatures.length
            ? Number(signatures[signatureIndex + 1].tick)
            : Number.POSITIVE_INFINITY;

          let barEndTick = barStartTick + ticksPerBar;
          if (nextSignatureTick > barStartTick && nextSignatureTick < barEndTick) {
            barEndTick = nextSignatureTick;
          }

          if (barStartTick > endTick + ticksPerBar) {
            break;
          }

          if (barEndTick >= startTick - ticksPerBeat * 2 && barStartTick <= endTick + ticksPerBeat * 2) {
            bars.push({
              tick: barStartTick,
              barNumber: barNumber,
              ticksPerBeat: ticksPerBeat,
              barEndTick: barEndTick,
            });
          }

          let beatTick = barStartTick;
          let beatInBar = 1;
          while (beatTick < barEndTick - 0.0001) {
            if (beatTick >= startTick - ticksPerBeat * 2 && beatTick <= endTick + ticksPerBeat * 2) {
              beats.push({
                tick: beatTick,
                barNumber: barNumber,
                beatInBar: beatInBar,
                isBarStart: beatInBar === 1,
                ticksPerBeat: ticksPerBeat,
              });
            }
            beatTick += ticksPerBeat;
            beatInBar += 1;
            if (beatInBar > 128) {
              break;
            }
          }

          barStartTick = barEndTick;
          barNumber += 1;
        }

        return {
          bars: bars,
          beats: beats,
        };
      }

      function drawTimelineRuler(markers, startTick, endTick) {
        const noteAreaWidth = Math.max(0, canvas.width - LEFT_PAD - RIGHT_PAD);

        ctx.fillStyle = "#121212";
        ctx.fillRect(0, 0, canvas.width, TOP_PAD);
        ctx.fillStyle = "#181818";
        ctx.fillRect(LEFT_PAD, 0, noteAreaWidth, TOP_PAD);

        ctx.strokeStyle = "#2f2f2f";
        ctx.beginPath();
        ctx.moveTo(LEFT_PAD + 0.5, RULER_DIVIDER_Y + 0.5);
        ctx.lineTo(canvas.width - RIGHT_PAD + 0.5, RULER_DIVIDER_Y + 0.5);
        ctx.stroke();

        ctx.fillStyle = "#f0f0f0";
        ctx.font = "11px Arial";

        let lastBarLabelX = Number.NEGATIVE_INFINITY;
        for (const bar of markers.bars) {
          const x = Math.round(xForTick(bar.tick)) + 0.5;
          if (x < LEFT_PAD || x > canvas.width - RIGHT_PAD) {
            continue;
          }
          if (x - lastBarLabelX >= 56) {
            ctx.fillText(String(bar.barNumber), x + 2, RULER_BAR_TEXT_Y);
            lastBarLabelX = x;
          }
        }

        let lastBeatLabelX = Number.NEGATIVE_INFINITY;
        ctx.fillStyle = "#b6b6b6";
        for (const beat of markers.beats) {
          if (beat.isBarStart) {
            continue;
          }
          const x = Math.round(xForTick(beat.tick)) + 0.5;
          if (x < LEFT_PAD || x > canvas.width - RIGHT_PAD) {
            continue;
          }
          const beatSpacingPx = beat.ticksPerBeat * state.pixelsPerTick;
          if (beatSpacingPx < 36) {
            continue;
          }
          if (x - lastBeatLabelX >= 36) {
            ctx.fillText(String(beat.beatInBar), x + 2, RULER_BEAT_TEXT_Y);
            lastBeatLabelX = x;
          }
        }

        const startSec = tickToSeconds(startTick);
        const endSec = tickToSeconds(endTick);
        const spanSec = Math.max(0.001, endSec - startSec);
        const pxPerSecond = noteAreaWidth / spanSec;
        const stepSec = chooseTimeLabelStepSeconds(pxPerSecond);

        let currentSec = Math.floor(startSec / stepSec) * stepSec;
        let lastTimeLabelX = Number.NEGATIVE_INFINITY;
        ctx.fillStyle = "#9fd3ff";
        for (let index = 0; index < 5000; index += 1) {
          if (currentSec > endSec + stepSec) {
            break;
          }
          const tick = secondsToTick(currentSec);
          const x = Math.round(xForTick(tick)) + 0.5;
          if (x >= LEFT_PAD && x <= canvas.width - RIGHT_PAD && x - lastTimeLabelX >= 72) {
            ctx.fillText(formatElapsedTime(currentSec), x + 2, RULER_TIME_TEXT_Y);
            lastTimeLabelX = x;
          }
          currentSec += stepSec;
        }
      }

      function drawGrid() {
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const startTick = state.xOffsetTicks;
        const endTick = state.xOffsetTicks + getVisibleTickSpan();
        const timeline = buildTimelineMarkers(startTick, endTick);

        drawTimelineRuler(timeline, startTick, endTick);

        for (let pitch = state.pitchMin; pitch <= state.pitchMax; pitch += 1) {
          const y = yForPitch(pitch);
          ctx.fillStyle = pitch % 12 === 0 ? "#1a1a1a" : "#141414";
          ctx.fillRect(LEFT_PAD, y, canvas.width - LEFT_PAD - RIGHT_PAD, NOTE_ROW_HEIGHT);
        }

        for (const marker of timeline.beats) {
          const x = Math.round(xForTick(marker.tick)) + 0.5;
          if (x < LEFT_PAD || x > canvas.width - RIGHT_PAD) {
            continue;
          }
          if (!marker.isBarStart && marker.ticksPerBeat * state.pixelsPerTick < 8) {
            continue;
          }
          ctx.strokeStyle = marker.isBarStart ? "#4b4b4b" : "#2f2f2f";
          ctx.beginPath();
          ctx.moveTo(x, TOP_PAD);
          ctx.lineTo(x, canvas.height - BOTTOM_PAD);
          ctx.stroke();
        }

        ctx.fillStyle = "#d8d8d8";
        ctx.font = "11px Arial";
        for (let pitch = state.pitchMin; pitch <= state.pitchMax; pitch += 12) {
          const y = yForPitch(pitch) + NOTE_ROW_HEIGHT - 1;
          ctx.fillText(String(pitch), 8, y);
        }
      }

      function drawNotes() {
        for (const box of state.noteBoxes) {
          const note = box.note;
          const trackIndex = Number(note.editable_track_index || 0);
          const color = colorForTrack(trackIndex);
          const selected = state.selectedNoteIds.has(String(note.note_id));

          ctx.fillStyle = color;
          ctx.fillRect(box.x, box.y, box.w, box.h);

          ctx.strokeStyle = selected ? "#ffffff" : "#000000";
          ctx.lineWidth = selected ? 2 : 1;
          ctx.strokeRect(box.x + 0.5, box.y + 0.5, Math.max(0, box.w - 1), Math.max(0, box.h - 1));
        }
        ctx.lineWidth = 1;
      }

      function drawSelectionRectangle() {
        if (!state.dragSelect || !state.dragSelect.active) {
          return;
        }

        const x = Math.min(state.dragSelect.startX, state.dragSelect.currentX);
        const y = Math.min(state.dragSelect.startY, state.dragSelect.currentY);
        const w = Math.abs(state.dragSelect.currentX - state.dragSelect.startX);
        const h = Math.abs(state.dragSelect.currentY - state.dragSelect.startY);

        ctx.fillStyle = "rgba(120, 180, 255, 0.18)";
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = "#78b4ff";
        ctx.setLineDash([5, 3]);
        ctx.strokeRect(x + 0.5, y + 0.5, Math.max(0, w - 1), Math.max(0, h - 1));
        ctx.setLineDash([]);
      }

      function redraw() {
        updatePitchRange();
        updateCanvasSize();
        clampXOffsetTicks();
        rebuildNoteBoxes();
        drawGrid();
        drawNotes();
        drawSelectionRectangle();
      }

      function getNoteBoxAt(x, y) {
        for (let index = state.noteBoxes.length - 1; index >= 0; index -= 1) {
          const box = state.noteBoxes[index];
          if (x >= box.x && x <= box.x + box.w && y >= box.y && y <= box.y + box.h) {
            return box;
          }
        }
        return null;
      }

      function selectNoteById(noteId, additive) {
        const id = String(noteId);
        if (!additive) {
          state.selectedNoteIds.clear();
        }
        if (additive && state.selectedNoteIds.has(id)) {
          state.selectedNoteIds.delete(id);
        } else {
          state.selectedNoteIds.add(id);
        }
        updateSelectionUi();
      }

      function selectNotesInRect(rect, additive) {
        if (!additive) {
          state.selectedNoteIds.clear();
        }

        for (const box of state.noteBoxes) {
          const intersects =
            box.x < rect.x + rect.w &&
            box.x + box.w > rect.x &&
            box.y < rect.y + rect.h &&
            box.y + box.h > rect.y;
          if (intersects) {
            state.selectedNoteIds.add(String(box.note.note_id));
          }
        }
        updateSelectionUi();
      }

      function canvasPointFromEvent(event) {
        const rect = canvas.getBoundingClientRect();
        return {
          x: Math.max(0, Math.min(canvas.width, event.clientX - rect.left)),
          y: Math.max(0, Math.min(canvas.height, event.clientY - rect.top)),
        };
      }

      function handleCanvasMouseDown(event) {
        if (event.button !== 0) {
          return;
        }

        const point = canvasPointFromEvent(event);
        if (currentTool === "pan") {
          state.panDrag = {
            startX: point.x,
            startOffsetTicks: state.xOffsetTicks,
          };
          canvas.style.cursor = "grabbing";
          return;
        }

        if (point.y < TOP_PAD || point.x < LEFT_PAD || point.x > canvas.width - RIGHT_PAD) {
          return;
        }

        if (currentTool !== "select") {
          return;
        }

        const hit = getNoteBoxAt(point.x, point.y);
        state.dragSelect = {
          startX: point.x,
          startY: point.y,
          currentX: point.x,
          currentY: point.y,
          active: false,
          hitNoteId: hit ? String(hit.note.note_id) : null,
          additive: Boolean(event.ctrlKey || event.metaKey),
        };
      }

      function handleCanvasMouseMove(event) {
        const point = canvasPointFromEvent(event);

        if (state.panDrag) {
          const dx = point.x - state.panDrag.startX;
          state.xOffsetTicks = state.panDrag.startOffsetTicks - dx / state.pixelsPerTick;
          clampXOffsetTicks();
          redraw();
          return;
        }

        if (!state.dragSelect) {
          return;
        }

        state.dragSelect.currentX = point.x;
        state.dragSelect.currentY = point.y;

        const dx = Math.abs(state.dragSelect.currentX - state.dragSelect.startX);
        const dy = Math.abs(state.dragSelect.currentY - state.dragSelect.startY);
        if (dx >= DRAG_THRESHOLD_PX || dy >= DRAG_THRESHOLD_PX) {
          state.dragSelect.active = true;
        }

        redraw();
      }

      function handleCanvasMouseUp() {
        if (state.panDrag) {
          state.panDrag = null;
          if (currentTool === "pan") {
            canvas.style.cursor = "grab";
          }
          return;
        }

        if (!state.dragSelect) {
          return;
        }

        if (state.dragSelect.active) {
          const rect = {
            x: Math.min(state.dragSelect.startX, state.dragSelect.currentX),
            y: Math.min(state.dragSelect.startY, state.dragSelect.currentY),
            w: Math.abs(state.dragSelect.currentX - state.dragSelect.startX),
            h: Math.abs(state.dragSelect.currentY - state.dragSelect.startY),
          };
          selectNotesInRect(rect, state.dragSelect.additive);
        } else if (state.dragSelect.hitNoteId) {
          selectNoteById(state.dragSelect.hitNoteId, state.dragSelect.additive);
        } else if (!state.dragSelect.additive) {
          clearSelection();
          setStatus("Selection cleared.");
        }

        state.dragSelect = null;
        redraw();
      }

      function handleCanvasWheel(event) {
        if (currentTool !== "zoom" && currentTool !== "pan") {
          return;
        }

        event.preventDefault();
        const point = canvasPointFromEvent(event);

        if (currentTool === "zoom") {
          const cursorTick = state.xOffsetTicks + (point.x - LEFT_PAD) / state.pixelsPerTick;
          const factor = event.deltaY < 0 ? ZOOM_IN_FACTOR : ZOOM_OUT_FACTOR;
          state.pixelsPerTick = Math.max(
            MIN_PIXELS_PER_TICK,
            Math.min(MAX_PIXELS_PER_TICK, state.pixelsPerTick * factor)
          );
          state.xOffsetTicks = cursorTick - (point.x - LEFT_PAD) / state.pixelsPerTick;
          clampXOffsetTicks();
          setStatus("Zoom: " + state.pixelsPerTick.toFixed(3) + " px/tick");
          redraw();
          return;
        }

        const deltaTicks = event.deltaY / state.pixelsPerTick;
        state.xOffsetTicks += deltaTicks;
        clampXOffsetTicks();
        redraw();
      }

      function moveSelectedToTrack() {
        if (state.selectedNoteIds.size === 0) {
          setStatus("No notes selected.");
          return;
        }

        const targetIndex = Number(targetTrackEl.value || 0);
        const targetTrack = getTrackByIndex(targetIndex);
        if (!targetTrack) {
          setStatus("Target track is invalid.");
          return;
        }

        const selectionBefore = Array.from(state.selectedNoteIds).filter(function (noteId) {
          return state.noteById.has(String(noteId));
        });
        const selectedSet = new Set(selectionBefore);
        const beforeNotes = [];
        const afterNotes = [];

        let movedCount = 0;
        for (const note of session.notes) {
          if (selectedSet.has(String(note.note_id))) {
            if (Number(note.editable_track_index) === Number(targetTrack.editable_track_index)) {
              continue;
            }
            beforeNotes.push(cloneNoteSnapshot(note));
            note.editable_track_index = Number(targetTrack.editable_track_index);
            note.editable_track_name = String(targetTrack.name);
            afterNotes.push(cloneNoteSnapshot(note));
            movedCount += 1;
          }
        }

        if (movedCount === 0) {
          setStatus("Selected notes are already on the target track.");
          return;
        }

        pushHistoryTransaction({
          label: "move-selected-to-track",
          beforeNotes: beforeNotes,
          afterNotes: afterNotes,
          selectionBefore: selectionBefore,
          selectionAfter: selectionBefore,
        });

        sortNotes();
        rebuildTrackSources();
        rebuildNoteLookup();
        setSelectionFromList(selectionBefore);
        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        setStatus(
          "Moved " + String(movedCount) + " notes to track " + String(targetTrack.editable_track_index) + "."
        );
      }

      function nextAvailableTrackIndex() {
        const used = new Set();
        for (const track of session.tracks) {
          used.add(Number(track.editable_track_index));
        }
        for (let index = 1; index <= MAX_TRACKS; index += 1) {
          if (!used.has(index)) {
            return index;
          }
        }
        return null;
      }

      function addTrack() {
        const nextIndex = nextAvailableTrackIndex();
        if (nextIndex === null) {
          setStatus("Cannot add track: maximum editable track count reached (12).");
          return;
        }

        const defaultName = "Track " + String(nextIndex);
        const entered = window.prompt("Track name:", defaultName);
        if (entered === null) {
          setStatus("Add track cancelled.");
          return;
        }

        const name = entered.trim() ? entered.trim() : defaultName;
        session.tracks.push({
          editable_track_index: nextIndex,
          name: name,
          source_track_indices: [],
          muted: null,
          color: null,
        });

        sortTracks();
        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        setStatus("Added track " + String(nextIndex) + ".");
      }

      function mergeSelectedTracks() {
        const selected = Array.from(state.mergeTrackIndices).sort(function (a, b) { return a - b; });
        if (selected.length < 2) {
          setStatus("Select at least two tracks in the sidebar to merge.");
          return;
        }

        const targetIndex = selected[0];
        const selectedSet = new Set(selected);
        const targetTrack = getTrackByIndex(targetIndex);
        if (!targetTrack) {
          setStatus("Merge target track is invalid.");
          return;
        }

        let moved = 0;
        for (const note of session.notes) {
          const idx = Number(note.editable_track_index);
          if (selectedSet.has(idx) && idx !== targetIndex) {
            note.editable_track_index = targetIndex;
            note.editable_track_name = String(targetTrack.name);
            moved += 1;
          }
        }

        session.tracks = session.tracks.filter(function (track) {
          const index = Number(track.editable_track_index);
          return index === targetIndex || !selectedSet.has(index);
        });

        sortTracks();
        sortNotes();
        rebuildTrackSources();
        rebuildNoteLookup();
        updateSelectionUi();
        state.mergeTrackIndices.clear();
        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        setStatus("Merged tracks into track " + String(targetIndex) + ". Moved " + String(moved) + " notes.");
      }

      function downloadBlob(data, mimeType, fileName) {
        const blob = new Blob([data], { type: mimeType });
        const href = URL.createObjectURL(blob);
        const link = document.createElement("a");
        try {
          link.href = href;
          link.download = fileName;
          document.body.appendChild(link);
          link.click();
          link.remove();
        } finally {
          URL.revokeObjectURL(href);
        }
      }

      function sessionBaseName() {
        const sourceName = String(session.source_midi || "split_session").split(/[/\\\\]/).pop() || "split_session";
        return sourceName.replace(/[.][^.]+$/, "") || "split_session";
      }

      function filenameFromDisposition(contentDisposition, fallback) {
        if (!contentDisposition) {
          return fallback;
        }
        const match = /filename=\"?([^\";]+)\"?/i.exec(contentDisposition);
        if (!match) {
          return fallback;
        }
        const candidate = match[1].trim();
        if (!candidate) {
          return fallback;
        }
        return candidate.replace(/[\\/]/g, "_");
      }

      async function importMidi(file) {
        if (!state.serverConnected) {
          setErrorStatus("Server connection unavailable. Start the editor with: midi-cleaner midi split-editor");
          return;
        }
        if (!file) {
          setErrorStatus("No MIDI file selected for import.");
          return;
        }

        const fileName = String(file.name || "uploaded.mid");
        console.info("Import control activated");
        setStatus("File selected: " + fileName, false);

        let fileBytes;
        try {
          console.info("Reading file");
          setStatus("Reading file", false);
          fileBytes = await file.arrayBuffer();
        } catch (error) {
          setErrorStatus("Could not read selected MIDI file.", error);
          return;
        }

        const url = "/api/import-midi?filename=" + encodeURIComponent(fileName);
        let response;
        try {
          console.info("Uploading MIDI");
          setStatus("Uploading MIDI", false);
          response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: fileBytes,
          });
        } catch (error) {
          setErrorStatus("MIDI import failed: server unavailable.", error);
          return;
        }

        let payload;
        try {
          console.info("Import response received");
          setStatus("Import response received", false);
          payload = await parseJsonResponse(response, "MIDI import");
          ensureUsableSessionPayload(payload, "MIDI import");
        } catch (error) {
          setErrorStatus(String(error.message || error), error);
          return;
        }

        try {
          console.info("Replacing editor session");
          applyImportedSession(payload);
        } catch (error) {
          setErrorStatus("MIDI import succeeded but UI refresh failed.", error);
          return;
        }

        const noteCount = Array.isArray(payload.notes) ? payload.notes.length : 0;
        const trackCount = Array.isArray(payload.tracks) ? payload.tracks.length : 0;
        console.info("Imported " + String(noteCount) + " notes from " + String(trackCount) + " tracks");
        setStatus(
          "Imported " + String(noteCount) + " notes from " + String(trackCount) + " tracks: " + fileName,
          false
        );
      }

      async function exportSessionToDownload(options) {
        if (!state.serverConnected) {
          setErrorStatus("Server connection unavailable. Start the editor with: midi-cleaner midi split-editor");
          return;
        }

        const endpoint = String(options.endpoint);
        const operationLabel = String(options.operationLabel);
        const fallbackFileName = String(options.fallbackFileName);
        const expectedMimePrefix = String(options.expectedMimePrefix);
        const successLabel = String(options.successLabel);
        const requestBody = JSON.stringify(session);

        setStatus(operationLabel + " in progress...", false);

        let response;
        try {
          response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: requestBody,
          });
        } catch (error) {
          setErrorStatus(operationLabel + " failed: server unavailable.", error);
          return;
        }

        if (!response.ok) {
          try {
            await parseJsonResponse(response, operationLabel);
          } catch (error) {
            setErrorStatus(String(error.message || error), error);
          }
          return;
        }

        const contentType = String(response.headers.get("Content-Type") || "").toLowerCase();
        if (!contentType.startsWith(expectedMimePrefix)) {
          setErrorStatus(
            operationLabel + " failed: unexpected response type '" + contentType + "'."
          );
          return;
        }

        let data;
        try {
          data = await response.arrayBuffer();
        } catch (error) {
          setErrorStatus(operationLabel + " failed: could not read response body.", error);
          return;
        }

        if (!data || data.byteLength <= 0) {
          setErrorStatus(operationLabel + " failed: exported file is empty.");
          return;
        }

        const fileName = filenameFromDisposition(
          response.headers.get("Content-Disposition"),
          fallbackFileName
        );

        try {
          downloadBlob(data, contentType, fileName);
        } catch (error) {
          setErrorStatus(operationLabel + " failed while preparing browser download.", error);
          return;
        }

        console.info(successLabel + ": " + fileName);
        setStatus(successLabel + ": " + fileName + ".", false);
      }

      async function exportMultitrackMidi() {
        await exportSessionToDownload({
          endpoint: "/api/export-multitrack",
          operationLabel: "Export multitrack MIDI",
          fallbackFileName: sessionBaseName() + "_split.mid",
          expectedMimePrefix: "audio/midi",
          successLabel: "Exported multitrack MIDI",
        });
      }

      async function exportSeparateTracks() {
        await exportSessionToDownload({
          endpoint: "/api/export-separate",
          operationLabel: "Export separate tracks ZIP",
          fallbackFileName: sessionBaseName() + "_split_tracks.zip",
          expectedMimePrefix: "application/zip",
          successLabel: "Exported separate track ZIP",
        });
      }

      function bindToolbarActions() {
        document.getElementById("move-selected-btn").addEventListener("click", moveSelectedToTrack);
        document.getElementById("merge-notes-btn").addEventListener("click", mergeSelectedNotes);
        undoButton.addEventListener("click", undoHistory);
        redoButton.addEventListener("click", redoHistory);
        document.getElementById("add-track-btn").addEventListener("click", addTrack);
        document.getElementById("merge-tracks-btn").addEventListener("click", mergeSelectedTracks);

        importLabel.addEventListener("click", function (event) {
          setStatus("Import control activated", false);
          console.info("Import control activated");
          if (importInput.disabled || !state.serverConnected) {
            event.preventDefault();
            setErrorStatus("Server connection unavailable. Start the editor with: midi-cleaner midi split-editor");
            return;
          }
        });

        importInput.addEventListener("change", function (event) {
          const input = event.target;
          const file = input && input.files && input.files[0] ? input.files[0] : null;
          if (!file) {
            setStatus("No file selected.", false);
            return;
          }
          importMidi(file).finally(function () {
            if (input) {
              input.value = "";
            }
          });
        });

        exportMultitrackButton.addEventListener("click", function () {
          exportMultitrackMidi();
        });
        exportSeparateButton.addEventListener("click", function () {
          exportSeparateTracks();
        });

        document.getElementById("clear-selection-btn").addEventListener("click", function () {
          clearSelection();
          redraw();
          setStatus("Selection cleared.");
        });

        toolButtons.select.addEventListener("click", function () { setTool("select"); });
        toolButtons.zoom.addEventListener("click", function () { setTool("zoom"); });
        toolButtons.pan.addEventListener("click", function () { setTool("pan"); });
      }

      function bindCanvasActions() {
        canvas.addEventListener("mousedown", handleCanvasMouseDown);
        window.addEventListener("mousemove", handleCanvasMouseMove);
        window.addEventListener("mouseup", handleCanvasMouseUp);
        canvas.addEventListener("wheel", handleCanvasWheel, { passive: false });
        window.addEventListener("resize", redraw);
      }

      async function initialize() {
        setServerActionControlsEnabled(
          false,
          "Server connection unavailable. Start the editor with: midi-cleaner midi split-editor"
        );
        setServerStatus(false, "Server: checking");
        setSessionData(session);
        renderAll();
        setTool("select");
        bindToolbarActions();
        bindCanvasActions();

        window.addEventListener("keydown", function (event) {
          if (isEditableTypingTarget(event.target)) {
            return;
          }

          const isModifierDown = Boolean(event.ctrlKey || event.metaKey);
          const key = String(event.key || "").toLowerCase();

          if (isModifierDown && key === "z" && !event.shiftKey) {
            event.preventDefault();
            undoHistory();
            return;
          }

          if (isModifierDown && (key === "y" || (key === "z" && event.shiftKey))) {
            event.preventDefault();
            redoHistory();
            return;
          }

          if (event.key === "Escape") {
            clearSelection();
            redraw();
            setStatus("Selection cleared.");
          }
        });

        window.__midiSplitEditor = {
          getSession: function () { return JSON.parse(JSON.stringify(session)); },
          getSelectedCount: function () { return state.selectedNoteIds.size; },
          selectNotesByIds: function (ids) {
            setSelectionFromList(ids || []);
            redraw();
          },
          setTool: setTool,
          getTool: function () { return currentTool; },
          moveSelectedToTrack: moveSelectedToTrack,
          mergeSelectedNotes: mergeSelectedNotes,
          undo: undoHistory,
          redo: redoHistory,
          addTrack: addTrack,
          mergeSelectedTracks: mergeSelectedTracks,
          importMidi: importMidi,
          exportMultitrackMidi: exportMultitrackMidi,
          exportSeparateTracks: exportSeparateTracks,
          clearSelection: function () {
            clearSelection();
            redraw();
          },
          getViewState: function () {
            return {
              pixelsPerTick: state.pixelsPerTick,
              xOffsetTicks: state.xOffsetTicks,
            };
          },
          isServerConnected: function () {
            return state.serverConnected;
          },
          getNoteBoxes: function () {
            return state.noteBoxes.map(function (box) {
              return {
                note_id: box.note.note_id,
                x: box.x,
                y: box.y,
                w: box.w,
                h: box.h,
              };
            });
          },
        };

        const serverSession = await checkServerConnection();
        if (serverSession) {
          applyImportedSession(serverSession);
        }
      }

      initialize().catch(function (error) {
        setErrorStatus("Editor initialization failed.", error);
      });
    })();
  </script>
</body>
</html>
"""

    return template.replace("__SESSION_JSON__", payload_json).replace("__BUILD_ID__", build_id)


def generate_piano_roll_preview(session: MidiSplitSession, output_html: Path) -> None:
    html = render_piano_roll_preview_html(session)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
