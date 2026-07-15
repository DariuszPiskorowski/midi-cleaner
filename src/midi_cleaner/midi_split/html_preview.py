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
    <button id="copy-notes-btn" type="button" disabled>Copy</button>
    <button id="paste-notes-btn" type="button" disabled>Paste</button>

    <span>|</span>
    <button id="tool-select-btn" type="button">Select</button>
    <button id="tool-draw-btn" type="button">Draw</button>
    <button id="tool-zoom-btn" type="button">Zoom</button>
    <button id="tool-pan-btn" type="button">Hand</button>

    <label for="snap-enabled">Snap</label>
    <input id="snap-enabled" type="checkbox" checked />
    <select id="snap-grid">
      <option value="4">1/4</option>
      <option value="8">1/8</option>
      <option value="16" selected>1/16</option>
      <option value="32">1/32</option>
    </select>
    <label for="velocity-lane-visible">Velocity lane</label>
    <input id="velocity-lane-visible" type="checkbox" checked />
    <label for="velocity-values-visible">Velocity values</label>
    <input id="velocity-values-visible" type="checkbox" />

    <span>|</span>
    <label for="target-track">Target track:</label>
    <select id="target-track"></select>
    <button id="move-selected-btn" type="button">Move selected to track</button>
    <button id="merge-notes-btn" type="button">Merge Notes</button>
    <button id="delete-notes-btn" type="button" disabled>Delete Notes</button>
    <button id="mute-notes-btn" type="button" disabled>Mute Notes</button>
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
      const NOTE_EDGE_RESIZE_HIT_PX = 6;
      const RULER_BAR_TEXT_Y = 12;
      const RULER_BEAT_TEXT_Y = 24;
      const RULER_TIME_TEXT_Y = 36;
      const RULER_DIVIDER_Y = 42;
      const MIN_CANVAS_WIDTH = 1000;
      const NOTE_ROW_HEIGHT = 10;
      const VELOCITY_LANE_HEIGHT = 124;
      const VELOCITY_LANE_GAP = 8;
      const VELOCITY_AXIS_TEXT_PAD = 4;
      const VELOCITY_GROUP_HANDLE_HEIGHT = 8;
      const VELOCITY_FAN_SPACING = 3;
      const VELOCITY_BAR_DRAW_WIDTH = 6;
      const VELOCITY_BAR_HIT_WIDTH = 10;
      const PASTE_CURSOR_COLOR = "#8ed1ff";
      const VELOCITY_DRAG_PX_PER_STEP_NORMAL = 4;
      const VELOCITY_DRAG_PX_PER_STEP_FINE = 8;
      const VELOCITY_DRAG_PX_PER_STEP_COARSE = 2;
      const KEYBOARD_FOCUS_VIEWPORT = "viewport";
      const KEYBOARD_FOCUS_NOTES = "notes";
      const KEYBOARD_FOCUS_VELOCITY = "velocity";
      const DEFAULT_PIXELS_PER_TICK = 0.18;
      const DEFAULT_DRAW_VELOCITY = 100;
      const MIN_PIXELS_PER_TICK = 0.03;
      const MAX_PIXELS_PER_TICK = 2.50;
      const ZOOM_IN_FACTOR = 1.20;
      const ZOOM_OUT_FACTOR = 1 / 1.20;
      const DRAG_THRESHOLD_PX = 4;
      const DRAW_NOTE_ID_PREFIX = "drawn";
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
        for (const note of session.notes) {
          if (!note || typeof note !== "object") {
            continue;
          }
          note.muted = note.muted === true;
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
      const deleteNotesButton = document.getElementById("delete-notes-btn");
      const muteNotesButton = document.getElementById("mute-notes-btn");
      const copyNotesButton = document.getElementById("copy-notes-btn");
      const pasteNotesButton = document.getElementById("paste-notes-btn");
      const snapEnabledEl = document.getElementById("snap-enabled");
      const snapGridEl = document.getElementById("snap-grid");
      const velocityLaneVisibleEl = document.getElementById("velocity-lane-visible");
      const velocityValuesVisibleEl = document.getElementById("velocity-values-visible");

      const toolButtons = {
        select: document.getElementById("tool-select-btn"),
        draw: document.getElementById("tool-draw-btn"),
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
        velocityBars: [],
        velocityGroups: [],
        velocityGroupMetaByNoteId: new Map(),
        dragSelect: null,
        noteEditDrag: null,
        velocityDrag: null,
        drawDrag: null,
        panDrag: null,
        selectionRegion: null,
        clipboard: null,
        pasteCursorTick: null,
        pasteIdCounter: 1,
        keyboardFocusMode: KEYBOARD_FOCUS_VIEWPORT,
        focusedVelocityNoteId: null,
        focusedVelocityGroupNoteIds: null,
        pitchViewportShift: 0,
        pitchMin: 24,
        pitchMax: 108,
        velocityLaneVisible: true,
        velocityValuesVisible: false,
        pixelsPerTick: DEFAULT_PIXELS_PER_TICK,
        xOffsetTicks: 0,
        serverConnected: false,
        undoStack: [],
        redoStack: [],
        drawNoteCounter: 1,
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

      function pitchNameFromMidi(pitch) {
        const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        const normalized = Math.max(0, Math.min(127, Math.round(Number(pitch || 0))));
        const octave = Math.floor(normalized / 12) - 1;
        return names[normalized % 12] + String(octave);
      }

      function clampPitchMidi(value) {
        return Math.max(0, Math.min(127, Math.round(Number(value || 0))));
      }

      function velocityValue(note) {
        const raw = Number(note && note.velocity);
        if (!Number.isFinite(raw)) {
          return 0;
        }
        return Math.max(0, Math.min(127, Math.round(raw)));
      }

      function clampEditedVelocity(value) {
        return Math.max(1, Math.min(127, Math.round(Number(value || 0))));
      }

      function velocityRatio(note) {
        return velocityValue(note) / 127;
      }

      function velocityFromLaneY(y) {
        const laneTop = velocityLaneTopY();
        const laneBottom = velocityLaneBottomY();
        const clampedY = Math.max(laneTop, Math.min(laneBottom, Number(y || laneBottom)));
        const ratio = (laneBottom - clampedY) / Math.max(1, VELOCITY_LANE_HEIGHT);
        return clampEditedVelocity(Math.round(ratio * 127));
      }

      function velocityGroupKeyForNote(note) {
        const startTick = Math.round(Number(note && note.start_tick || 0));
        const trackIndex = Math.round(Number(note && note.editable_track_index || 0));
        const channel = Number(note && note.channel);
        const hasChannel = Number.isFinite(channel);
        return String(startTick) + "|" + String(trackIndex) + "|" + (hasChannel ? String(Math.round(channel)) : "any");
      }

      function sortedNotesForVelocityGroup(notes) {
        const list = notes.slice();
        list.sort(function (a, b) {
          if (Number(a.pitch_midi) !== Number(b.pitch_midi)) {
            return Number(a.pitch_midi) - Number(b.pitch_midi);
          }
          return String(a.note_id).localeCompare(String(b.note_id));
        });
        return list;
      }

      function generateUniquePastedNoteId() {
        const existing = new Set(session.notes.map(function (note) {
          return String(note.note_id);
        }));

        for (let guard = 0; guard < 200000; guard += 1) {
          const candidate = "pasted_" + String(Date.now()) + "_" + String(state.pasteIdCounter).padStart(6, "0");
          state.pasteIdCounter += 1;
          if (!existing.has(candidate)) {
            return candidate;
          }
        }

        return "pasted_" + String(Date.now()) + "_" + String(Math.floor(Math.random() * 1000000));
      }

      function currentSnapDivision() {
        const parsed = Number(snapGridEl && snapGridEl.value);
        if (!Number.isFinite(parsed) || parsed <= 0) {
          return 16;
        }
        return Math.max(1, Math.round(parsed));
      }

      function isSnapEnabled() {
        return Boolean(snapEnabledEl && snapEnabledEl.checked);
      }

      function currentSnapTicks() {
        const ticksPerBeat = Math.max(1, Number(session.ticks_per_beat || 480));
        const denominator = currentSnapDivision();
        return Math.max(1, Math.round((ticksPerBeat * 4) / denominator));
      }

      function minimumDurationTicks() {
        if (!isSnapEnabled()) {
          return 1;
        }
        return Math.max(1, currentSnapTicks());
      }

      function snapAbsoluteTick(tick) {
        const rounded = Math.round(Number(tick || 0));
        if (!isSnapEnabled()) {
          return rounded;
        }
        const grid = currentSnapTicks();
        return Math.round(rounded / grid) * grid;
      }

      function snapDeltaTicks(deltaTicks) {
        const rounded = Math.round(Number(deltaTicks || 0));
        if (!isSnapEnabled()) {
          return rounded;
        }
        const grid = currentSnapTicks();
        return Math.round(rounded / grid) * grid;
      }

      function noteDurationTicks(note) {
        const start = Math.round(Number(note && note.start_tick || 0));
        const end = Math.round(Number(note && note.end_tick || start));
        return Math.max(0, end - start);
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

      function setKeyboardFocusMode(mode, options) {
        const opts = options && typeof options === "object" ? options : {};
        const normalized = String(mode || "").toLowerCase();
        if (
          normalized !== KEYBOARD_FOCUS_VIEWPORT
          && normalized !== KEYBOARD_FOCUS_NOTES
          && normalized !== KEYBOARD_FOCUS_VELOCITY
        ) {
          return;
        }

        state.keyboardFocusMode = normalized;
        if (normalized !== KEYBOARD_FOCUS_VELOCITY && !opts.preserveVelocityTarget) {
          state.focusedVelocityGroupNoteIds = null;
          state.focusedVelocityNoteId = null;
        }
      }

      function getKeyboardFocusMode() {
        return String(state.keyboardFocusMode || KEYBOARD_FOCUS_VIEWPORT);
      }

      function setSelectionFromList(noteIds) {
        state.selectionRegion = null;
        state.selectedNoteIds.clear();
        for (const noteId of noteIds || []) {
          const normalized = String(noteId);
          if (state.noteById.has(normalized)) {
            state.selectedNoteIds.add(normalized);
          }
        }
        updateSelectionUi();
      }

      function getSelectedNotes() {
        return Array.from(state.selectedNoteIds)
          .map(function (noteId) {
            return state.noteById.get(String(noteId));
          })
          .filter(function (note) { return Boolean(note); });
      }

      function resolveSelectedMuteAction(selectedNotes) {
        if (!selectedNotes.length) {
          return "mute";
        }
        const allMuted = selectedNotes.every(function (note) {
          return note.muted === true;
        });
        return allMuted ? "unmute" : "mute";
      }

      function updateHistoryControls() {
        undoButton.disabled = state.undoStack.length === 0;
        redoButton.disabled = state.redoStack.length === 0;
      }

      function updateMergeButtonState() {
        mergeNotesButton.disabled = state.selectedNoteIds.size < 2;
      }

      function updateDeleteButtonState() {
        deleteNotesButton.disabled = state.selectedNoteIds.size === 0;
      }

      function updateMuteButtonState() {
        const selectedNotes = getSelectedNotes();
        if (!selectedNotes.length) {
          muteNotesButton.disabled = true;
          muteNotesButton.textContent = "Mute Notes";
          return;
        }

        muteNotesButton.disabled = false;
        const mode = resolveSelectedMuteAction(selectedNotes);
        muteNotesButton.textContent = mode === "unmute" ? "Unmute Notes" : "Mute Notes";
      }

      function updateClipboardButtons() {
        copyNotesButton.disabled = state.selectedNoteIds.size === 0;
        pasteNotesButton.disabled = !state.clipboard || !Array.isArray(state.clipboard.notes) || state.clipboard.notes.length === 0;
      }

      function updateEditorActionButtons() {
        updateHistoryControls();
        updateMergeButtonState();
        updateDeleteButtonState();
        updateMuteButtonState();
        updateClipboardButtons();
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
        note.pitch_midi = clampPitchMidi(note.pitch_midi);
        note.pitch_name = pitchNameFromMidi(note.pitch_midi);
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

      function deleteSelectedNotes() {
        const selectionBefore = Array.from(state.selectedNoteIds).filter(function (noteId) {
          return state.noteById.has(String(noteId));
        });
        if (!selectionBefore.length) {
          setStatus("No notes selected.");
          return;
        }

        const selectedSet = new Set(selectionBefore);
        const deletedNotes = session.notes
          .filter(function (note) {
            return selectedSet.has(String(note.note_id));
          })
          .map(function (note) {
            return cloneNoteSnapshot(note);
          });

        if (!deletedNotes.length) {
          setStatus("No notes selected.");
          return;
        }

        session.notes = session.notes.filter(function (note) {
          return !selectedSet.has(String(note.note_id));
        });

        pushHistoryTransaction({
          label: "delete-notes",
          beforeNotes: deletedNotes,
          afterNotes: [],
          selectionBefore: selectionBefore,
          selectionAfter: [],
        });

        sortNotes();
        rebuildTrackSources();
        rebuildNoteLookup();
        setSelectionFromList([]);
        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        updateEditorActionButtons();
        setStatus("Deleted " + String(deletedNotes.length) + " notes.", false);
      }

      function setMuteStateForSelected(shouldMute) {
        const selectionBefore = Array.from(state.selectedNoteIds).filter(function (noteId) {
          return state.noteById.has(String(noteId));
        });
        if (!selectionBefore.length) {
          setStatus("No notes selected.");
          return;
        }

        const selectedSet = new Set(selectionBefore);
        const beforeNotes = [];
        const afterNotes = [];

        for (const note of session.notes) {
          if (!selectedSet.has(String(note.note_id))) {
            continue;
          }
          beforeNotes.push(cloneNoteSnapshot(note));
          note.muted = Boolean(shouldMute);
          afterNotes.push(cloneNoteSnapshot(note));
        }

        if (!beforeNotes.length) {
          setStatus("No notes selected.");
          return;
        }

        pushHistoryTransaction({
          label: Boolean(shouldMute) ? "mute-notes" : "unmute-notes",
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
        updateEditorActionButtons();
        if (Boolean(shouldMute)) {
          setStatus("Muted " + String(beforeNotes.length) + " notes.", false);
        } else {
          setStatus("Unmuted " + String(beforeNotes.length) + " notes.", false);
        }
      }

      function toggleMuteSelectedNotes() {
        const selectedNotes = getSelectedNotes();
        if (!selectedNotes.length) {
          setStatus("No notes selected.");
          return;
        }
        const mode = resolveSelectedMuteAction(selectedNotes);
        setMuteStateForSelected(mode !== "unmute");
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
        applyPitchViewportShift(state.pitchViewportShift);
      }

      function updateCanvasSize() {
        const width = Math.max(MIN_CANVAS_WIDTH, Number(rollWrapEl.clientWidth || MIN_CANVAS_WIDTH));
        const rows = state.pitchMax - state.pitchMin + 1;
        const pianoHeight = TOP_PAD + rows * NOTE_ROW_HEIGHT;
        const velocityHeight = state.velocityLaneVisible ? (VELOCITY_LANE_GAP + VELOCITY_LANE_HEIGHT) : 0;
        const height = Math.max(420, Math.ceil(pianoHeight + velocityHeight + BOTTOM_PAD));
        canvas.width = Math.ceil(width);
        canvas.height = height;
      }

      function pianoRollBottomY() {
        const rows = state.pitchMax - state.pitchMin + 1;
        return TOP_PAD + rows * NOTE_ROW_HEIGHT;
      }

      function velocityLaneTopY() {
        return pianoRollBottomY() + VELOCITY_LANE_GAP;
      }

      function velocityLaneBottomY() {
        return velocityLaneTopY() + VELOCITY_LANE_HEIGHT;
      }

      function isPointInPianoRollArea(point) {
        if (!point) {
          return false;
        }
        return point.y >= TOP_PAD && point.y <= pianoRollBottomY();
      }

      function isPointInVelocityLane(point) {
        if (!point || !state.velocityLaneVisible) {
          return false;
        }
        return point.y >= velocityLaneTopY() && point.y <= velocityLaneBottomY();
      }

      function yForPitch(pitch) {
        return TOP_PAD + (state.pitchMax - pitch) * NOTE_ROW_HEIGHT;
      }

      function pitchFromCanvasY(y) {
        const relative = Math.floor((Number(y) - TOP_PAD) / NOTE_ROW_HEIGHT);
        const pitch = state.pitchMax - relative;
        return clampPitchMidi(pitch);
      }

      function xForTick(tick) {
        return LEFT_PAD + (Number(tick) - state.xOffsetTicks) * state.pixelsPerTick;
      }

      function tickFromCanvasX(x) {
        return state.xOffsetTicks + (Number(x) - LEFT_PAD) / state.pixelsPerTick;
      }

      function updateSelectionUi() {
        sanitizeSelection();
        selectedCountEl.textContent = String(state.selectedNoteIds.size);
        updateEditorActionButtons();
        updateSelectionStatusLine();
      }

      function clearSelection() {
        state.selectionRegion = null;
        state.selectedNoteIds.clear();
        setKeyboardFocusMode(KEYBOARD_FOCUS_VIEWPORT);
        updateSelectionUi();
      }

      function normalizedPitchBounds() {
        if (session.notes.length === 0) {
          return {
            min: 24,
            max: 108,
          };
        }

        let minPitch = 127;
        let maxPitch = 0;
        for (const note of session.notes) {
          const pitch = Number(note.pitch_midi || 0);
          minPitch = Math.min(minPitch, pitch);
          maxPitch = Math.max(maxPitch, pitch);
        }

        return {
          min: Math.max(0, minPitch - 2),
          max: Math.min(127, maxPitch + 2),
        };
      }

      function applyPitchViewportShift(requestedShift) {
        const bounds = normalizedPitchBounds();
        const minAllowed = -bounds.min;
        const maxAllowed = 127 - bounds.max;
        const clampedShift = Math.max(minAllowed, Math.min(maxAllowed, Math.round(Number(requestedShift || 0))));
        state.pitchViewportShift = clampedShift;
        state.pitchMin = bounds.min + clampedShift;
        state.pitchMax = bounds.max + clampedShift;
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

        if (toolName === "draw") {
          canvas.style.cursor = "crosshair";
          return;
        }
        if (toolName === "zoom") {
          canvas.style.cursor = "zoom-in";
          return;
        }
        if (toolName === "pan") {
          canvas.style.cursor = state.panDrag ? "grabbing" : "grab";
          return;
        }

        canvas.style.cursor = "default";
      }

      function resetDerivedSessionState() {
        state.mergeTrackIndices.clear();
        state.dragSelect = null;
        state.noteEditDrag = null;
        state.velocityDrag = null;
        state.drawDrag = null;
        state.panDrag = null;
        state.noteBoxes = [];
        state.velocityBars = [];
        state.velocityGroups = [];
        state.velocityGroupMetaByNoteId = new Map();
        state.selectionRegion = null;
        state.clipboard = null;
        state.pasteCursorTick = null;
        state.pasteIdCounter = 1;
        state.keyboardFocusMode = KEYBOARD_FOCUS_VIEWPORT;
        state.focusedVelocityNoteId = null;
        state.focusedVelocityGroupNoteIds = null;
        state.pitchViewportShift = 0;
        state.velocityLaneVisible = true;
        state.velocityValuesVisible = false;
        state.pixelsPerTick = DEFAULT_PIXELS_PER_TICK;
        state.xOffsetTicks = 0;
        clearSelection();
        if (velocityLaneVisibleEl) {
          velocityLaneVisibleEl.checked = true;
        }
        if (velocityValuesVisibleEl) {
          velocityValuesVisibleEl.checked = false;
          velocityValuesVisibleEl.disabled = false;
        }
      }

      function resetDrawNoteCounter() {
        let nextOrdinal = 1;
        for (const note of session.notes) {
          const noteId = String(note.note_id || "");
          const match = /^drawn_(\\d{6})$/i.exec(noteId);
          if (!match) {
            continue;
          }
          const parsed = Number(match[1]);
          if (Number.isFinite(parsed)) {
            nextOrdinal = Math.max(nextOrdinal, Math.floor(parsed) + 1);
          }
        }
        state.drawNoteCounter = nextOrdinal;
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
        resetDrawNoteCounter();
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
        state.pitchViewportShift = 0;
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
          const gridBottom = state.velocityLaneVisible ? velocityLaneBottomY() : canvas.height - BOTTOM_PAD;
          ctx.lineTo(x, gridBottom);
          ctx.stroke();
        }

        ctx.fillStyle = "#d8d8d8";
        ctx.font = "11px Arial";
        for (let pitch = state.pitchMin; pitch <= state.pitchMax; pitch += 12) {
          const y = yForPitch(pitch) + NOTE_ROW_HEIGHT - 1;
          ctx.fillText(String(pitch), 8, y);
        }

        if (Number.isFinite(Number(state.pasteCursorTick))) {
          const pasteX = Math.round(xForTick(Number(state.pasteCursorTick))) + 0.5;
          if (pasteX >= LEFT_PAD && pasteX <= canvas.width - RIGHT_PAD) {
            ctx.strokeStyle = PASTE_CURSOR_COLOR;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(pasteX, TOP_PAD);
            const cursorBottom = state.velocityLaneVisible ? velocityLaneBottomY() : (canvas.height - BOTTOM_PAD);
            ctx.lineTo(pasteX, cursorBottom);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.lineWidth = 1;
          }
        }
      }

      function buildVelocityGroups() {
        const groupsByKey = new Map();

        for (const note of session.notes) {
          const key = velocityGroupKeyForNote(note);
          if (!groupsByKey.has(key)) {
            groupsByKey.set(key, []);
          }
          groupsByKey.get(key).push(note);
        }

        const groupMetaByNoteId = new Map();
        const groups = [];

        for (const notes of groupsByKey.values()) {
          const ordered = sortedNotesForVelocityGroup(notes);
          const first = ordered[0];
          const groupSize = ordered.length;
          const startTick = Number(first.start_tick || 0);
          const baseX = xForTick(startTick);

          let minFanX = Number.POSITIVE_INFINITY;
          let maxFanX = Number.NEGATIVE_INFINITY;

          for (let index = 0; index < ordered.length; index += 1) {
            const fanOffset = (index - (groupSize - 1) / 2) * VELOCITY_FAN_SPACING;
            const fanX = baseX + fanOffset;
            const noteId = String(ordered[index].note_id);
            groupMetaByNoteId.set(noteId, {
              groupSize: groupSize,
              fanOffset: fanOffset,
              fanIndex: index,
              baseX: baseX,
            });
            minFanX = Math.min(minFanX, fanX);
            maxFanX = Math.max(maxFanX, fanX + VELOCITY_BAR_DRAW_WIDTH);
          }

          if (groupSize >= 2) {
            const laneTop = velocityLaneTopY();
            groups.push({
              key: velocityGroupKeyForNote(first),
              noteIds: ordered.map(function (note) { return String(note.note_id); }),
              startTick: startTick,
              editableTrackIndex: Number(first.editable_track_index || 0),
              channel: first.channel,
              handleX: minFanX - 1,
              handleY: laneTop + 4,
              handleW: Math.max(8, maxFanX - minFanX + 2),
              handleH: VELOCITY_GROUP_HANDLE_HEIGHT,
            });
          }
        }

        return {
          groups: groups,
          groupMetaByNoteId: groupMetaByNoteId,
        };
      }

      function velocityBarForNote(note) {
        if (!state.velocityLaneVisible) {
          return null;
        }

        const noteId = String(note.note_id);
        const groupMeta = state.velocityGroupMetaByNoteId ? state.velocityGroupMetaByNoteId.get(noteId) : null;
        const startTick = Number(note.start_tick || 0);
        const baseX = xForTick(startTick);
        const x = baseX + Number(groupMeta && groupMeta.fanOffset || 0);
        const laneTop = velocityLaneTopY();
        const laneBottom = velocityLaneBottomY();
        const ratio = velocityRatio(note);
        const barHeight = Math.max(1, Math.round(ratio * VELOCITY_LANE_HEIGHT));
        const barTop = laneBottom - barHeight;
        const hitInset = Math.max(0, (VELOCITY_BAR_HIT_WIDTH - VELOCITY_BAR_DRAW_WIDTH) / 2);
        const hitX = x - hitInset;

        if (x + VELOCITY_BAR_DRAW_WIDTH < LEFT_PAD - 2 || x > canvas.width - RIGHT_PAD + 2) {
          return null;
        }

        return {
          noteId: noteId,
          startTick: startTick,
          velocity: velocityValue(note),
          muted: note.muted === true,
          selected: state.selectedNoteIds.has(noteId),
          groupSize: Number(groupMeta && groupMeta.groupSize || 1),
          fanIndex: Number(groupMeta && groupMeta.fanIndex || 0),
          baseX: baseX,
          x: x,
          y: barTop,
          w: VELOCITY_BAR_DRAW_WIDTH,
          hitX: hitX,
          hitW: VELOCITY_BAR_HIT_WIDTH,
          h: Math.max(1, laneBottom - barTop),
          laneTop: laneTop,
          laneBottom: laneBottom,
          note: note,
        };
      }

      function rebuildVelocityBars() {
        const grouping = buildVelocityGroups();
        state.velocityGroups = grouping.groups;
        state.velocityGroupMetaByNoteId = grouping.groupMetaByNoteId;
        state.velocityBars = [];
        if (!state.velocityLaneVisible) {
          return;
        }

        for (const note of session.notes) {
          const bar = velocityBarForNote(note);
          if (bar) {
            state.velocityBars.push(bar);
          }
        }
      }

      function getVelocityBars() {
        return state.velocityBars.map(function (bar) {
          return {
            note_id: bar.noteId,
            start_tick: bar.startTick,
            velocity: bar.velocity,
            muted: bar.muted,
            selected: bar.selected,
            group_size: bar.groupSize,
            fan_index: bar.fanIndex,
            base_x: bar.baseX,
            x: bar.x,
            y: bar.y,
            w: bar.w,
            h: bar.h,
            hit_x: bar.hitX,
            hit_w: bar.hitW,
            lane_top: bar.laneTop,
            lane_bottom: bar.laneBottom,
          };
        });
      }

      function velocityBarForNoteId(noteId) {
        const normalized = String(noteId);
        for (const bar of state.velocityBars) {
          if (String(bar.noteId) === normalized) {
            return {
              note_id: bar.noteId,
              start_tick: bar.startTick,
              velocity: bar.velocity,
              muted: bar.muted,
              selected: bar.selected,
              group_size: bar.groupSize,
              fan_index: bar.fanIndex,
              base_x: bar.baseX,
              x: bar.x,
              y: bar.y,
              w: bar.w,
              h: bar.h,
              hit_x: bar.hitX,
              hit_w: bar.hitW,
              lane_top: bar.laneTop,
              lane_bottom: bar.laneBottom,
            };
          }
        }
        return null;
      }

      function getVelocityGroups() {
        return state.velocityGroups.map(function (group) {
          return {
            key: group.key,
            note_ids: group.noteIds.slice(),
            start_tick: group.startTick,
            editable_track_index: group.editableTrackIndex,
            channel: group.channel,
            handle_x: group.handleX,
            handle_y: group.handleY,
            handle_w: group.handleW,
            handle_h: group.handleH,
          };
        });
      }

      function velocityGroupForBar(bar) {
        if (!bar) {
          return null;
        }
        for (const group of state.velocityGroups) {
          if (group.noteIds.indexOf(String(bar.noteId)) >= 0) {
            return group;
          }
        }
        return null;
      }

      function hitTestVelocityGroupHandle(x, y) {
        if (!state.velocityLaneVisible) {
          return null;
        }
        for (let index = state.velocityGroups.length - 1; index >= 0; index -= 1) {
          const group = state.velocityGroups[index];
          if (
            Number(x) >= Number(group.handleX)
            && Number(x) <= Number(group.handleX) + Number(group.handleW)
            && Number(y) >= Number(group.handleY)
            && Number(y) <= Number(group.handleY) + Number(group.handleH)
          ) {
            return group;
          }
        }
        return null;
      }

      function drawVelocityLane() {
        if (!state.velocityLaneVisible) {
          return;
        }

        const laneTop = velocityLaneTopY();
        const laneBottom = velocityLaneBottomY();

        ctx.fillStyle = "#0f0f0f";
        ctx.fillRect(LEFT_PAD, laneTop, canvas.width - LEFT_PAD - RIGHT_PAD, VELOCITY_LANE_HEIGHT);

        ctx.strokeStyle = "#2f2f2f";
        ctx.beginPath();
        ctx.moveTo(LEFT_PAD + 0.5, laneTop + 0.5);
        ctx.lineTo(canvas.width - RIGHT_PAD + 0.5, laneTop + 0.5);
        ctx.moveTo(LEFT_PAD + 0.5, laneBottom + 0.5);
        ctx.lineTo(canvas.width - RIGHT_PAD + 0.5, laneBottom + 0.5);
        ctx.stroke();

        ctx.fillStyle = "#a9a9a9";
        ctx.font = "10px Arial";
        ctx.fillText("127", VELOCITY_AXIS_TEXT_PAD, laneTop + 10);
        ctx.fillText("64", VELOCITY_AXIS_TEXT_PAD + 8, laneTop + Math.round(VELOCITY_LANE_HEIGHT / 2));
        ctx.fillText("0", VELOCITY_AXIS_TEXT_PAD + 18, laneBottom - 2);
        ctx.fillStyle = "#d8d8d8";
        ctx.fillText("Velocity", VELOCITY_AXIS_TEXT_PAD, laneTop - 2);

        for (const bar of state.velocityBars) {
          const note = bar.note;
          const selected = state.selectedNoteIds.has(String(note.note_id));
          const muted = note.muted === true;
          const color = colorForTrack(Number(note.editable_track_index || 0));

          ctx.fillStyle = color;
          ctx.globalAlpha = muted ? 0.22 : 0.95;
          ctx.fillRect(bar.x, bar.y, bar.w, bar.h);
          ctx.globalAlpha = 1.0;

          if (muted) {
            ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
            ctx.fillRect(bar.x, bar.y, bar.w, bar.h);
          }

          if (selected) {
            ctx.strokeStyle = muted ? "#ffd29a" : "#ffffff";
            ctx.lineWidth = 2;
            ctx.strokeRect(bar.x + 0.5, bar.y + 0.5, Math.max(1, bar.w - 1), Math.max(1, bar.h - 1));
            ctx.lineWidth = 1;
          }

          if (state.velocityValuesVisible && state.pixelsPerTick >= 0.18 && bar.h >= 18 && bar.w >= 6) {
            ctx.fillStyle = "#f3f3f3";
            ctx.font = "10px Arial";
            ctx.fillText(String(velocityValue(note)), bar.x - 2, Math.max(laneTop + 10, bar.y - 2));
          }
        }

        for (const group of state.velocityGroups) {
          if (group.noteIds.length < 2) {
            continue;
          }
          ctx.strokeStyle = "#d9e9ff";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(group.handleX + 0.5, group.handleY + group.handleH * 0.5 + 0.5);
          ctx.lineTo(group.handleX + group.handleW + 0.5, group.handleY + group.handleH * 0.5 + 0.5);
          ctx.stroke();
          ctx.lineWidth = 1;
        }

        ctx.globalAlpha = 1.0;
      }

      function drawNotes() {
        for (const box of state.noteBoxes) {
          const note = box.note;
          const trackIndex = Number(note.editable_track_index || 0);
          const color = colorForTrack(trackIndex);
          const selected = state.selectedNoteIds.has(String(note.note_id));
          const muted = note.muted === true;

          ctx.fillStyle = color;
          ctx.globalAlpha = muted ? 0.35 : 1.0;
          ctx.fillRect(box.x, box.y, box.w, box.h);
          ctx.globalAlpha = 1.0;

          if (muted) {
            ctx.strokeStyle = selected ? "#ffffff" : "#8a8a8a";
            ctx.lineWidth = selected ? 2 : 1;
            ctx.setLineDash([4, 2]);
            ctx.strokeRect(box.x + 0.5, box.y + 0.5, Math.max(0, box.w - 1), Math.max(0, box.h - 1));
            ctx.setLineDash([]);
            if (box.w >= 10) {
              ctx.fillStyle = "#d0d0d0";
              ctx.font = "9px Arial";
              ctx.fillText("M", box.x + 2, box.y + Math.max(8, box.h - 1));
            }
          } else {
            ctx.strokeStyle = selected ? "#ffffff" : "#000000";
            ctx.lineWidth = selected ? 2 : 1;
            ctx.strokeRect(box.x + 0.5, box.y + 0.5, Math.max(0, box.w - 1), Math.max(0, box.h - 1));
          }
        }
        ctx.globalAlpha = 1.0;
        ctx.setLineDash([]);
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
        rebuildVelocityBars();
        drawGrid();
        drawNotes();
        drawVelocityLane();
        drawSelectionRectangle();
        drawDrawPreview();
      }

      function hitTestVelocityBar(x, y) {
        if (!state.velocityLaneVisible) {
          return null;
        }

        // Priority 1: selected/active bars first for precise single-note editing.
        for (let index = state.velocityBars.length - 1; index >= 0; index -= 1) {
          const bar = state.velocityBars[index];
          if (!bar.selected) {
            continue;
          }
          if (x >= bar.hitX && x <= bar.hitX + bar.hitW && y >= bar.y && y <= bar.y + bar.h) {
            return bar;
          }
        }

        // Priority 2: any individual bar under cursor.
        for (let index = state.velocityBars.length - 1; index >= 0; index -= 1) {
          const bar = state.velocityBars[index];
          if (x >= bar.hitX && x <= bar.hitX + bar.hitW && y >= bar.y && y <= bar.y + bar.h) {
            return bar;
          }
        }
        return null;
      }

      function hitTestVelocityBarForApi(x, y) {
        const hit = hitTestVelocityBar(Number(x), Number(y));
        if (!hit) {
          return null;
        }
        return String(hit.noteId);
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

      function getNoteHitRegion(box, x) {
        const threshold = Math.max(3, Math.min(NOTE_EDGE_RESIZE_HIT_PX, box.w / 2));
        const rightEdge = box.x + box.w;

        if (Math.abs(Number(x) - box.x) <= threshold) {
          return "left";
        }
        if (Math.abs(Number(x) - rightEdge) <= threshold) {
          return "right";
        }
        return "body";
      }

      function updateCanvasCursorForPoint(point) {
        if (currentTool === "draw") {
          canvas.style.cursor = "crosshair";
          return;
        }
        if (currentTool === "zoom") {
          canvas.style.cursor = "zoom-in";
          return;
        }
        if (currentTool === "pan") {
          canvas.style.cursor = state.panDrag ? "grabbing" : "grab";
          return;
        }

        if (state.noteEditDrag) {
          if (state.noteEditDrag.mode === "move") {
            canvas.style.cursor = "grabbing";
          } else {
            canvas.style.cursor = "ew-resize";
          }
          return;
        }

        if (state.velocityDrag) {
          canvas.style.cursor = "ns-resize";
          return;
        }

        if (!point || point.x < LEFT_PAD || point.x > canvas.width - RIGHT_PAD) {
          canvas.style.cursor = "default";
          return;
        }

        if (isPointInVelocityLane(point)) {
          const groupHandleHit = hitTestVelocityGroupHandle(point.x, point.y);
          if (groupHandleHit) {
            canvas.style.cursor = "ns-resize";
            return;
          }
          const velocityHit = hitTestVelocityBar(point.x, point.y);
          canvas.style.cursor = velocityHit ? "ns-resize" : "default";
          return;
        }

        if (!isPointInPianoRollArea(point)) {
          canvas.style.cursor = "default";
          return;
        }

        const hit = getNoteBoxAt(point.x, point.y);
        if (!hit) {
          canvas.style.cursor = "default";
          return;
        }

        const region = getNoteHitRegion(hit, point.x);
        if (region === "left" || region === "right") {
          canvas.style.cursor = "ew-resize";
          return;
        }

        canvas.style.cursor = "move";
      }

      function selectNoteById(noteId, additive) {
        state.selectionRegion = null;
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

      function selectedVelocitySummary() {
        const selectedNotes = getSelectedNotes();
        if (!selectedNotes.length) {
          return {
            count: 0,
            min: null,
            avg: null,
            max: null,
            text: "Ready.",
          };
        }

        if (selectedNotes.length === 1) {
          const note = selectedNotes[0];
          return {
            count: 1,
            min: velocityValue(note),
            avg: velocityValue(note),
            max: velocityValue(note),
            text: "Selected: 1 note | pitch: "
              + String(pitchNameFromMidi(Number(note.pitch_midi || 0)))
              + " | velocity: " + String(velocityValue(note))
              + " | start: " + String(Math.round(Number(note.start_tick || 0)))
              + " | duration: " + String(Math.max(0, Math.round(Number(note.duration_ticks || 0))))
              + " ticks",
          };
        }

        let minVelocity = 127;
        let maxVelocity = 0;
        let sumVelocity = 0;
        for (const note of selectedNotes) {
          const value = velocityValue(note);
          minVelocity = Math.min(minVelocity, value);
          maxVelocity = Math.max(maxVelocity, value);
          sumVelocity += value;
        }
        const avgVelocity = Math.round(sumVelocity / selectedNotes.length);

        return {
          count: selectedNotes.length,
          min: minVelocity,
          avg: avgVelocity,
          max: maxVelocity,
          text: "Selected: " + String(selectedNotes.length)
            + " notes | velocity min/avg/max: "
            + String(minVelocity) + "/" + String(avgVelocity) + "/" + String(maxVelocity),
        };
      }

      function updateSelectionStatusLine() {
        const summary = selectedVelocitySummary();
        setStatus(summary.text, false);
      }

      function setVelocityLaneVisible(visible) {
        state.velocityLaneVisible = Boolean(visible);
        if (velocityLaneVisibleEl) {
          velocityLaneVisibleEl.checked = state.velocityLaneVisible;
        }
        if (velocityValuesVisibleEl) {
          velocityValuesVisibleEl.disabled = !state.velocityLaneVisible;
          if (!state.velocityLaneVisible) {
            velocityValuesVisibleEl.checked = false;
            state.velocityValuesVisible = false;
          }
        }
        redraw();
      }

      function setPasteCursorTick(tick, options) {
        const opts = options && typeof options === "object" ? options : {};
        if (tick === null || tick === undefined || !Number.isFinite(Number(tick))) {
          state.pasteCursorTick = null;
          return;
        }
        let normalized = Math.max(0, Math.round(Number(tick)));
        if (opts.snap !== false && isSnapEnabled()) {
          normalized = Math.max(0, snapAbsoluteTick(normalized));
        }
        state.pasteCursorTick = normalized;
      }

      function viewportTickStart() {
        return Number(state.xOffsetTicks || 0);
      }

      function viewportTickEnd() {
        return viewportTickStart() + getVisibleTickSpan();
      }

      function isTickRangeVisible(startTick, endTick) {
        const start = Number(startTick || 0);
        const end = Math.max(start, Number(endTick || start));
        const viewStart = viewportTickStart();
        const viewEnd = viewportTickEnd();
        return start >= viewStart && end <= viewEnd;
      }

      function ensureTickRangeVisible(startTick, endTick) {
        const start = Math.max(0, Math.round(Number(startTick || 0)));
        const end = Math.max(start, Math.round(Number(endTick || start)));
        const margin = Math.max(1, Math.round(Number(session.ticks_per_beat || 480) * 0.25));
        const desiredStart = Math.max(0, start - margin);
        const desiredEnd = end + margin;
        const span = Math.max(1, getVisibleTickSpan());

        let nextOffset = Number(state.xOffsetTicks || 0);
        if (desiredStart < viewportTickStart()) {
          nextOffset = desiredStart;
        } else if (desiredEnd > viewportTickEnd()) {
          nextOffset = Math.max(0, desiredEnd - span);
        }

        if (nextOffset !== Number(state.xOffsetTicks || 0)) {
          state.xOffsetTicks = nextOffset;
          clampXOffsetTicks();
          return true;
        }
        return false;
      }

      function panViewportByTicks(deltaTicks) {
        const parsed = Number(deltaTicks || 0);
        if (!Number.isFinite(parsed) || parsed === 0) {
          return false;
        }
        const before = Number(state.xOffsetTicks || 0);
        state.xOffsetTicks = before + parsed;
        clampXOffsetTicks();
        return Number(state.xOffsetTicks || 0) !== before;
      }

      function panViewportBySemitones(deltaRows) {
        const parsed = Math.round(Number(deltaRows || 0));
        if (!Number.isFinite(parsed) || parsed === 0) {
          return false;
        }
        const before = Number(state.pitchViewportShift || 0);
        applyPitchViewportShift(before + parsed);
        return Number(state.pitchViewportShift || 0) !== before;
      }

      function getViewportState() {
        return {
          pixelsPerTick: state.pixelsPerTick,
          xOffsetTicks: state.xOffsetTicks,
          pitchMin: state.pitchMin,
          pitchMax: state.pitchMax,
          pitchViewportShift: state.pitchViewportShift,
          visibleTickSpan: getVisibleTickSpan(),
        };
      }

      function getPasteCursorTick() {
        if (!Number.isFinite(Number(state.pasteCursorTick))) {
          return null;
        }
        return Number(state.pasteCursorTick);
      }

      function getClipboardSummary() {
        if (!state.clipboard) {
          return {
            note_count: 0,
            region_start_tick: null,
            region_end_tick: null,
            region_duration_ticks: null,
          };
        }
        return {
          note_count: Array.isArray(state.clipboard.notes) ? state.clipboard.notes.length : 0,
          region_start_tick: Number(state.clipboard.regionStartTick),
          region_end_tick: Number(state.clipboard.regionEndTick),
          region_duration_ticks: Number(state.clipboard.regionDurationTicks),
        };
      }

      function keyboardHorizontalNudgeTicks(event) {
        const ticksPerBeat = Math.max(1, Number(session.ticks_per_beat || 480));
        const snapTicks = isSnapEnabled() ? currentSnapTicks() : Math.max(1, Math.round(ticksPerBeat / 16));
        const base = Math.max(1, Math.round(snapTicks));
        if (event && event.shiftKey) {
          return Math.max(1, base * 4);
        }
        if (event && event.altKey) {
          return Math.max(1, Math.round(base / 4));
        }
        return base;
      }

      function keyboardPitchNudgeSemitones(event) {
        if (event && event.shiftKey) {
          return 12;
        }
        return 1;
      }

      function viewportHorizontalArrowStep(event) {
        const visible = Math.max(1, getVisibleTickSpan());
        let ratio = 0.10;
        if (event && event.shiftKey) {
          ratio = 0.25;
        } else if (event && event.altKey) {
          ratio = 0.03;
        }
        return Math.max(1, Math.round(visible * ratio));
      }

      function viewportVerticalArrowStep(event) {
        if (event && event.shiftKey) {
          return 12;
        }
        if (event && event.altKey) {
          return 1;
        }
        return 3;
      }

      function moveSelectedNotesByKeyboard(deltaTicks, deltaPitch, label) {
        const selection = Array.from(state.selectedNoteIds).filter(function (noteId) {
          return state.noteById.has(String(noteId));
        });
        if (!selection.length) {
          return {
            changedCount: 0,
            appliedDeltaTicks: 0,
            appliedDeltaPitch: 0,
          };
        }

        const selectedNotes = selection.map(function (noteId) {
          return state.noteById.get(String(noteId));
        }).filter(function (note) { return Boolean(note); });

        const beforeNotes = selectedNotes.map(function (note) {
          return cloneNoteSnapshot(note);
        });
        const beforeById = new Map(beforeNotes.map(function (note) {
          return [String(note.note_id), note];
        }));

        let appliedTicks = Math.round(Number(deltaTicks || 0));
        if (appliedTicks !== 0) {
          const minStart = Math.min.apply(null, beforeNotes.map(function (note) {
            return Number(note.start_tick || 0);
          }));
          if (minStart + appliedTicks < 0) {
            appliedTicks = -Math.round(minStart);
          }
        }

        let appliedPitch = Math.round(Number(deltaPitch || 0));
        if (appliedPitch !== 0) {
          const minPitch = Math.min.apply(null, beforeNotes.map(function (note) {
            return Number(note.pitch_midi || 0);
          }));
          const maxPitch = Math.max.apply(null, beforeNotes.map(function (note) {
            return Number(note.pitch_midi || 0);
          }));
          if (minPitch + appliedPitch < 0) {
            appliedPitch = -minPitch;
          }
          if (maxPitch + appliedPitch > 127) {
            appliedPitch = 127 - maxPitch;
          }
        }

        if (appliedTicks === 0 && appliedPitch === 0) {
          return {
            changedCount: 0,
            appliedDeltaTicks: 0,
            appliedDeltaPitch: 0,
          };
        }

        for (const before of beforeNotes) {
          const note = state.noteById.get(String(before.note_id));
          if (!note) {
            continue;
          }
          const duration = noteDurationTicks(before);
          note.start_tick = Math.max(0, Math.round(Number(before.start_tick || 0) + appliedTicks));
          note.end_tick = note.start_tick + duration;
          note.pitch_midi = clampPitchMidi(Number(before.pitch_midi || 0) + appliedPitch);
          syncNoteTimingFromTicks(note);
        }

        const changedBefore = [];
        const changedAfter = [];
        for (const noteId of selection) {
          const before = beforeById.get(String(noteId));
          const after = state.noteById.get(String(noteId));
          if (!before || !after) {
            continue;
          }
          const afterSnapshot = cloneNoteSnapshot(after);
          if (noteSnapshotsChanged(before, afterSnapshot)) {
            changedBefore.push(cloneNoteSnapshot(before));
            changedAfter.push(afterSnapshot);
          }
        }

        if (!changedAfter.length) {
          return {
            changedCount: 0,
            appliedDeltaTicks: 0,
            appliedDeltaPitch: 0,
          };
        }

        pushHistoryTransaction({
          label: String(label || "keyboard-note-move"),
          beforeNotes: changedBefore,
          afterNotes: changedAfter,
          selectionBefore: selection.slice(),
          selectionAfter: selection.slice(),
        });

        sortNotes();
        rebuildTrackSources();
        rebuildNoteLookup();
        setSelectionFromList(selection.slice());
        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        updateEditorActionButtons();

        return {
          changedCount: changedAfter.length,
          appliedDeltaTicks: appliedTicks,
          appliedDeltaPitch: appliedPitch,
        };
      }

      function focusedVelocityTargetNoteIds() {
        if (Array.isArray(state.focusedVelocityGroupNoteIds) && state.focusedVelocityGroupNoteIds.length) {
          const groupIds = state.focusedVelocityGroupNoteIds.map(function (noteId) {
            return String(noteId);
          }).filter(function (noteId) {
            return state.noteById.has(noteId);
          });
          if (groupIds.length) {
            return groupIds;
          }
        }

        const selected = Array.from(state.selectedNoteIds).filter(function (noteId) {
          return state.noteById.has(String(noteId));
        });
        if (selected.length) {
          return selected;
        }

        if (state.focusedVelocityNoteId && state.noteById.has(String(state.focusedVelocityNoteId))) {
          return [String(state.focusedVelocityNoteId)];
        }

        return [];
      }

      function adjustSelectedVelocityByKeyboard(delta) {
        const noteIds = focusedVelocityTargetNoteIds();
        if (!noteIds.length) {
          return {
            changedCount: 0,
            noteCount: 0,
          };
        }

        const selectionBefore = Array.from(state.selectedNoteIds).filter(function (noteId) {
          return state.noteById.has(String(noteId));
        });
        const effectiveSelectionBefore = selectionBefore.length ? selectionBefore : noteIds.slice();

        const beforeNotes = noteIds.map(function (noteId) {
          const note = state.noteById.get(String(noteId));
          return note ? cloneNoteSnapshot(note) : null;
        }).filter(function (note) { return Boolean(note); });

        const result = setVelocityForNotes(noteIds, { delta: Number(delta || 0) });
        if (result.changedCount <= 0) {
          return {
            changedCount: 0,
            noteCount: noteIds.length,
          };
        }

        const afterNotes = noteIds.map(function (noteId) {
          const note = state.noteById.get(String(noteId));
          return note ? cloneNoteSnapshot(note) : null;
        }).filter(function (note) { return Boolean(note); });

        pushHistoryTransaction({
          label: "keyboard-velocity-edit",
          beforeNotes: beforeNotes,
          afterNotes: afterNotes,
          selectionBefore: effectiveSelectionBefore,
          selectionAfter: noteIds.slice(),
        });

        sortNotes();
        rebuildTrackSources();
        rebuildNoteLookup();
        setSelectionFromList(noteIds.slice());
        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        updateEditorActionButtons();

        return {
          changedCount: result.changedCount,
          noteCount: noteIds.length,
        };
      }

      function handleArrowKey(event) {
        const key = String(event.key || "");
        if (key !== "ArrowLeft" && key !== "ArrowRight" && key !== "ArrowUp" && key !== "ArrowDown") {
          return false;
        }

        const mode = getKeyboardFocusMode();

        if (mode === KEYBOARD_FOCUS_VIEWPORT) {
          let handled = false;
          if (key === "ArrowLeft") {
            handled = panViewportByTicks(-viewportHorizontalArrowStep(event));
          } else if (key === "ArrowRight") {
            handled = panViewportByTicks(viewportHorizontalArrowStep(event));
          } else if (key === "ArrowUp") {
            handled = panViewportBySemitones(viewportVerticalArrowStep(event));
          } else if (key === "ArrowDown") {
            handled = panViewportBySemitones(-viewportVerticalArrowStep(event));
          }

          if (handled) {
            redraw();
            updateCanvasCursorForPoint(null);
            setStatus("Viewport moved.", false);
          }
          return true;
        }

        if (mode === KEYBOARD_FOCUS_NOTES && state.selectedNoteIds.size > 0) {
          let deltaTicks = 0;
          let deltaPitch = 0;

          if (key === "ArrowLeft") {
            deltaTicks = -keyboardHorizontalNudgeTicks(event);
          } else if (key === "ArrowRight") {
            deltaTicks = keyboardHorizontalNudgeTicks(event);
          } else if (key === "ArrowUp") {
            deltaPitch = keyboardPitchNudgeSemitones(event);
          } else if (key === "ArrowDown") {
            deltaPitch = -keyboardPitchNudgeSemitones(event);
          }

          const result = moveSelectedNotesByKeyboard(deltaTicks, deltaPitch, "keyboard-note-move");
          if (result.changedCount > 0) {
            if (deltaTicks > 0) {
              setStatus("Moved " + String(result.changedCount) + " note(s) right.", false);
            } else if (deltaTicks < 0) {
              setStatus("Moved " + String(result.changedCount) + " note(s) left.", false);
            } else if (deltaPitch > 0) {
              setStatus("Moved " + String(result.changedCount) + " note(s) up.", false);
            } else if (deltaPitch < 0) {
              setStatus("Moved " + String(result.changedCount) + " note(s) down.", false);
            }
          }
          return true;
        }

        if (mode === KEYBOARD_FOCUS_VELOCITY) {
          if (key === "ArrowLeft") {
            const movedLeft = panViewportByTicks(-viewportHorizontalArrowStep(event));
            if (movedLeft) {
              redraw();
              setStatus("Viewport moved.", false);
            }
            return true;
          }
          if (key === "ArrowRight") {
            const movedRight = panViewportByTicks(viewportHorizontalArrowStep(event));
            if (movedRight) {
              redraw();
              setStatus("Viewport moved.", false);
            }
            return true;
          }
          if (key === "ArrowUp" || key === "ArrowDown") {
            const step = event.shiftKey ? 10 : 1;
            const delta = key === "ArrowUp" ? step : -step;
            const result = adjustSelectedVelocityByKeyboard(delta);
            if (result.changedCount > 0) {
              const prefix = delta > 0 ? "+" : "";
              setStatus(
                "Velocity " + prefix + String(delta) + " for " + String(result.noteCount) + " note(s).",
                false
              );
            }
            return true;
          }
        }

        return false;
      }

      function setVelocityForNotes(noteIds, options) {
        const normalizedNoteIds = (noteIds || []).map(function (noteId) {
          return String(noteId);
        });
        const uniqueIds = Array.from(new Set(normalizedNoteIds)).filter(function (noteId) {
          return state.noteById.has(noteId);
        });

        const opts = options && typeof options === "object" ? options : {};
        const hasTarget = Number.isFinite(Number(opts.targetVelocity));
        const hasDelta = Number.isFinite(Number(opts.delta));
        let changedCount = 0;

        for (const noteId of uniqueIds) {
          const note = state.noteById.get(noteId);
          if (!note) {
            continue;
          }

          const beforeVelocity = clampEditedVelocity(velocityValue(note));
          let nextVelocity = beforeVelocity;
          if (hasTarget) {
            nextVelocity = clampEditedVelocity(Number(opts.targetVelocity));
          } else if (hasDelta) {
            nextVelocity = clampEditedVelocity(beforeVelocity + Number(opts.delta));
          }

          if (beforeVelocity !== nextVelocity) {
            note.velocity = nextVelocity;
            changedCount += 1;
          }
        }

        return {
          noteIds: uniqueIds,
          changedCount: changedCount,
        };
      }

      function startVelocityDrag(params) {
        if (!params || !Array.isArray(params.noteIds) || !params.noteIds.length) {
          return;
        }
        const normalizedIds = params.noteIds
          .map(function (noteId) { return String(noteId); })
          .filter(function (noteId) { return state.noteById.has(noteId); });
        if (!normalizedIds.length) {
          return;
        }

        const anchorId = state.noteById.has(String(params.anchorNoteId))
          ? String(params.anchorNoteId)
          : normalizedIds[0];
        const anchorNote = state.noteById.get(anchorId);
        if (!anchorNote) {
          return;
        }

        const beforeById = new Map();
        const beforeNotes = [];
        for (const noteId of normalizedIds) {
          const note = state.noteById.get(noteId);
          if (!note) {
            continue;
          }
          const snapshot = cloneNoteSnapshot(note);
          beforeById.set(noteId, snapshot);
          beforeNotes.push(snapshot);
        }
        if (!beforeNotes.length) {
          return;
        }

        state.velocityDrag = {
          noteIds: normalizedIds,
          anchorNoteId: anchorId,
          beforeById: beforeById,
          beforeNotes: beforeNotes,
          selectionBefore: Array.from(state.selectedNoteIds),
          startY: Number(params.startY || 0),
          startVelocity: clampEditedVelocity(velocityValue(anchorNote)),
          appliedDelta: 0,
          active: false,
          mode: String(params.mode || "bar"),
          sensitivityMode: "normal",
        };
      }

      function velocityDragPxPerStep(sensitivityMode) {
        if (sensitivityMode === "fine") {
          return VELOCITY_DRAG_PX_PER_STEP_FINE;
        }
        if (sensitivityMode === "coarse") {
          return VELOCITY_DRAG_PX_PER_STEP_COARSE;
        }
        return VELOCITY_DRAG_PX_PER_STEP_NORMAL;
      }

      function sensitivityModeFromEvent(event) {
        if (event && event.shiftKey) {
          return "fine";
        }
        if (event && (event.altKey || event.ctrlKey || event.metaKey)) {
          return "coarse";
        }
        return "normal";
      }

      function applyVelocityDragPreview(drag, point, event) {
        if (!drag || !point) {
          return;
        }

        const mode = sensitivityModeFromEvent(event);
        drag.sensitivityMode = mode;
        const pxPerStep = velocityDragPxPerStep(mode);
        const deltaPixels = Number(drag.startY || 0) - Number(point.y || 0);
        const delta = Math.round(deltaPixels / Math.max(1, pxPerStep));

        if (drag.appliedDelta === delta && drag.active) {
          return;
        }

        let changedCount = 0;
        for (const noteId of drag.noteIds) {
          const before = drag.beforeById.get(String(noteId));
          const note = state.noteById.get(String(noteId));
          if (!before || !note) {
            continue;
          }
          const baseline = clampEditedVelocity(velocityValue(before));
          const nextVelocity = clampEditedVelocity(baseline + delta);
          if (clampEditedVelocity(velocityValue(note)) !== nextVelocity) {
            note.velocity = nextVelocity;
            changedCount += 1;
          }
        }

        drag.appliedDelta = delta;
        if (changedCount > 0) {
          drag.active = true;
        }

        if (drag.noteIds.length === 1) {
          const note = state.noteById.get(String(drag.noteIds[0]));
          const prefix = delta >= 0 ? "+" : "";
          setStatus(
            "Velocity " + prefix + String(delta) + " for 1 note(s): "
              + String(note ? clampEditedVelocity(velocityValue(note)) : clampEditedVelocity(drag.startVelocity + delta)),
            false
          );
        } else {
          const prefix = delta >= 0 ? "+" : "";
          setStatus(
            "Velocity " + prefix + String(delta) + " for " + String(drag.noteIds.length) + " note(s).",
            false
          );
        }
      }

      function finalizeVelocityDrag(drag) {
        if (!drag) {
          return;
        }

        const changedBefore = [];
        const changedAfter = [];
        for (const noteId of drag.noteIds) {
          const before = drag.beforeById.get(String(noteId));
          const note = state.noteById.get(String(noteId));
          if (!before || !note) {
            continue;
          }
          const beforeVelocity = clampEditedVelocity(velocityValue(before));
          const afterVelocity = clampEditedVelocity(velocityValue(note));
          if (beforeVelocity !== afterVelocity) {
            changedBefore.push(cloneNoteSnapshot(before));
            changedAfter.push(cloneNoteSnapshot(note));
          }
        }

        if (!changedAfter.length) {
          return;
        }

        pushHistoryTransaction({
          label: drag.mode === "group" ? "velocity-group-edit" : "velocity-edit",
          beforeNotes: changedBefore,
          afterNotes: changedAfter,
          selectionBefore: Array.isArray(drag.selectionBefore) ? drag.selectionBefore.slice() : [],
          selectionAfter: drag.noteIds.slice(),
        });

        sortNotes();
        rebuildTrackSources();
        rebuildNoteLookup();
        setSelectionFromList(drag.noteIds.slice());
        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        updateEditorActionButtons();

        if (drag.noteIds.length === 1) {
          setStatus("Changed velocity for 1 note.", false);
        } else {
          setStatus("Changed velocity for " + String(drag.noteIds.length) + " notes.", false);
        }
      }

      function copySelectedNotes() {
        const selectedNotes = getSelectedNotes();
        if (!selectedNotes.length) {
          setStatus("No notes selected to copy.", false);
          return;
        }

        let regionStart = null;
        let regionEnd = null;
        if (state.selectionRegion && Number.isFinite(Number(state.selectionRegion.startTick)) && Number.isFinite(Number(state.selectionRegion.endTick))) {
          regionStart = Math.max(0, Math.round(Number(state.selectionRegion.startTick)));
          regionEnd = Math.max(regionStart + 1, Math.round(Number(state.selectionRegion.endTick)));
        } else {
          regionStart = Math.min.apply(null, selectedNotes.map(function (note) {
            return Math.round(Number(note.start_tick || 0));
          }));
          regionEnd = Math.max.apply(null, selectedNotes.map(function (note) {
            return Math.round(Number(note.end_tick || note.start_tick || 0));
          }));
          regionEnd = Math.max(regionStart + 1, regionEnd);
        }

        const clipboardNotes = selectedNotes.map(function (note) {
          const snapshot = cloneNoteSnapshot(note);
          return {
            sourceNoteId: String(snapshot.note_id),
            payload: snapshot,
            offsetStartTicks: Math.round(Number(snapshot.start_tick || 0)) - regionStart,
            offsetEndTicks: Math.round(Number(snapshot.end_tick || snapshot.start_tick || 0)) - regionStart,
            durationTicks: Math.max(0, Math.round(Number(snapshot.duration_ticks || 0))),
          };
        });

        state.clipboard = {
          notes: clipboardNotes,
          regionStartTick: regionStart,
          regionEndTick: regionEnd,
          regionDurationTicks: Math.max(1, regionEnd - regionStart),
        };

        if (!Number.isFinite(Number(state.pasteCursorTick))) {
          setPasteCursorTick(regionEnd, { snap: true });
        }

        updateEditorActionButtons();
        redraw();
        setStatus(
          "Copied " + String(clipboardNotes.length) + " note(s), region "
            + String(state.clipboard.regionDurationTicks) + " ticks.",
          false
        );
      }

      function pasteCopiedNotes() {
        if (!state.clipboard || !Array.isArray(state.clipboard.notes) || !state.clipboard.notes.length) {
          setStatus("Clipboard is empty.", false);
          return;
        }

        const clipboard = state.clipboard;
        const durationTicks = Math.max(1, Math.round(Number(clipboard.regionDurationTicks || 1)));
        const defaultVisibleTick = Math.max(0, snapAbsoluteTick(viewportTickStart()));
        const defaultClipboardTick = Math.max(0, Math.round(Number(clipboard.regionEndTick || 0)));
        const requestedCursor = Number.isFinite(Number(state.pasteCursorTick))
          ? Number(state.pasteCursorTick)
          : (isTickRangeVisible(defaultClipboardTick, defaultClipboardTick) ? defaultClipboardTick : defaultVisibleTick);
        const baseTick = isSnapEnabled() ? Math.max(0, snapAbsoluteTick(requestedCursor)) : Math.max(0, Math.round(requestedCursor));

        const selectionBefore = Array.from(state.selectedNoteIds);
        const pastedSnapshots = [];
        const pastedIds = [];
        let fallbackTrackCount = 0;

        for (const item of clipboard.notes) {
          const sourcePayload = cloneNoteSnapshot(item.payload || {});
          const note = cloneNoteSnapshot(sourcePayload);
          note.note_id = generateUniquePastedNoteId();

          const offsetStart = Math.round(Number(item.offsetStartTicks || 0));
          const offsetEnd = Math.round(Number(item.offsetEndTicks || 0));
          const duration = Math.max(0, Math.round(Number(item.durationTicks || (offsetEnd - offsetStart))));

          note.start_tick = Math.max(0, Math.round(baseTick + offsetStart));
          note.end_tick = Math.max(note.start_tick, Math.round(baseTick + offsetEnd));
          if (note.end_tick <= note.start_tick) {
            note.end_tick = note.start_tick + Math.max(1, duration);
          }
          note.duration_ticks = Math.max(0, note.end_tick - note.start_tick);

          const existingTrack = getTrackByIndex(Number(note.editable_track_index || 0));
          if (!existingTrack) {
            const fallbackTrack = getTrackByIndex(Number(targetTrackEl.value || 0)) || (session.tracks.length ? session.tracks[0] : null);
            if (fallbackTrack) {
              note.editable_track_index = Number(fallbackTrack.editable_track_index);
              note.editable_track_name = String(fallbackTrack.name || "Track");
              fallbackTrackCount += 1;
            }
          }

          note.velocity = clampEditedVelocity(velocityValue(note));
          syncNoteTimingFromTicks(note);
          session.notes.push(note);
          pastedSnapshots.push(cloneNoteSnapshot(note));
          pastedIds.push(String(note.note_id));
        }

        if (!pastedSnapshots.length) {
          setStatus("Clipboard is empty.", false);
          return;
        }

        pushHistoryTransaction({
          label: "paste-notes",
          beforeNotes: [],
          afterNotes: pastedSnapshots,
          selectionBefore: selectionBefore,
          selectionAfter: pastedIds.slice(),
        });

        sortNotes();
        rebuildTrackSources();
        rebuildNoteLookup();
        setSelectionFromList(pastedIds.slice());
        updateTargetTrackDropdown();
        renderTrackPanel();

        const pastedStartTick = Math.min.apply(null, pastedSnapshots.map(function (note) {
          return Math.round(Number(note.start_tick || 0));
        }));
        const pastedEndTick = Math.max.apply(null, pastedSnapshots.map(function (note) {
          return Math.round(Number(note.end_tick || note.start_tick || 0));
        }));
        ensureTickRangeVisible(pastedStartTick, pastedEndTick);

        setPasteCursorTick(baseTick + durationTicks, { snap: true });
        ensureTickRangeVisible(Number(state.pasteCursorTick || 0), Number(state.pasteCursorTick || 0));
        redraw();
        updateEditorActionButtons();

        if (fallbackTrackCount > 0) {
          setStatus("Pasted " + String(pastedSnapshots.length) + " note(s). " + String(fallbackTrackCount) + " note(s) used fallback track.", false);
        } else {
          setStatus("Pasted " + String(pastedSnapshots.length) + " note(s).", false);
        }
      }

      function drawDrawPreview() {
        if (!state.drawDrag) {
          return;
        }

        const drag = state.drawDrag;
        const minDuration = minimumDurationTicks();
        const startTick = Number(drag.startTick);
        const currentTick = Number(drag.currentTick);

        let leftTick = Math.min(startTick, currentTick);
        let rightTick = Math.max(startTick, currentTick);
        if (!drag.active || rightTick <= leftTick) {
          rightTick = leftTick + minDuration;
        }

        const x = xForTick(leftTick);
        const rightX = xForTick(rightTick);
        const w = Math.max(1, rightX - x);
        const pitch = clampPitchMidi(drag.currentPitch);
        const y = yForPitch(pitch) + 1;
        const h = Math.max(3, NOTE_ROW_HEIGHT - 2);

        ctx.fillStyle = "rgba(154, 184, 255, 0.45)";
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = "#d6e5ff";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(x + 0.5, y + 0.5, Math.max(0, w - 1), Math.max(0, h - 1));
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
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

      function startNoteEditDrag(hitBox, point, additive) {
        const hitNoteId = String(hitBox.note.note_id);
        const hitRegion = getNoteHitRegion(hitBox, point.x);
        const mode = hitRegion === "left"
          ? "resize-left"
          : (hitRegion === "right" ? "resize-right" : "move");

        if (mode === "move") {
          if (!state.selectedNoteIds.has(hitNoteId)) {
            if (additive) {
              state.selectedNoteIds.add(hitNoteId);
            } else {
              state.selectedNoteIds.clear();
              state.selectedNoteIds.add(hitNoteId);
            }
            updateSelectionUi();
          }
        } else {
          setSelectionFromList([hitNoteId]);
        }

        const selection = mode === "move"
          ? Array.from(state.selectedNoteIds).filter(function (noteId) {
              return state.noteById.has(String(noteId));
            })
          : [hitNoteId];

        const beforeNotes = selection
          .map(function (noteId) {
            const note = state.noteById.get(String(noteId));
            return note ? cloneNoteSnapshot(note) : null;
          })
          .filter(function (note) { return Boolean(note); });

        const beforeById = new Map(
          beforeNotes.map(function (note) {
            return [String(note.note_id), note];
          })
        );

        state.noteEditDrag = {
          mode: mode,
          startX: point.x,
          startY: point.y,
          active: false,
          noteIds: selection,
          targetNoteId: hitNoteId,
          selectionBefore: selection.slice(),
          beforeNotes: beforeNotes,
          beforeById: beforeById,
          appliedDeltaTicks: 0,
          appliedDeltaPitch: 0,
          appliedTick: null,
        };
      }

      function applyMovePreview(drag, point) {
        const rawDeltaTicks = (point.x - drag.startX) / state.pixelsPerTick;
        let deltaTicks = snapDeltaTicks(rawDeltaTicks);

        const minStart = Math.min.apply(
          null,
          drag.beforeNotes.map(function (note) {
            return Number(note.start_tick || 0);
          })
        );
        if (minStart + deltaTicks < 0) {
          deltaTicks = -minStart;
        }

        const rawDeltaPitch = Math.round((drag.startY - point.y) / NOTE_ROW_HEIGHT);
        let deltaPitch = rawDeltaPitch;
        const minPitch = Math.min.apply(
          null,
          drag.beforeNotes.map(function (note) {
            return Number(note.pitch_midi || 0);
          })
        );
        const maxPitch = Math.max.apply(
          null,
          drag.beforeNotes.map(function (note) {
            return Number(note.pitch_midi || 0);
          })
        );
        if (minPitch + deltaPitch < 0) {
          deltaPitch = -minPitch;
        }
        if (maxPitch + deltaPitch > 127) {
          deltaPitch = 127 - maxPitch;
        }

        if (drag.appliedDeltaTicks === deltaTicks && drag.appliedDeltaPitch === deltaPitch) {
          return;
        }

        drag.appliedDeltaTicks = deltaTicks;
        drag.appliedDeltaPitch = deltaPitch;

        for (const before of drag.beforeNotes) {
          const note = state.noteById.get(String(before.note_id));
          if (!note) {
            continue;
          }

          const duration = noteDurationTicks(before);
          note.start_tick = Math.max(0, Math.round(Number(before.start_tick || 0) + deltaTicks));
          note.end_tick = note.start_tick + duration;
          note.pitch_midi = clampPitchMidi(Number(before.pitch_midi || 0) + deltaPitch);
          syncNoteTimingFromTicks(note);
        }
      }

      function applyResizePreview(drag, point) {
        const before = drag.beforeById.get(String(drag.targetNoteId));
        const note = state.noteById.get(String(drag.targetNoteId));
        if (!before || !note) {
          return;
        }

        const requestedTick = Math.max(0, snapAbsoluteTick(tickFromCanvasX(point.x)));
        if (drag.appliedTick === requestedTick) {
          return;
        }
        drag.appliedTick = requestedTick;

        const beforeStart = Math.round(Number(before.start_tick || 0));
        const beforeEnd = Math.max(beforeStart, Math.round(Number(before.end_tick || beforeStart)));
        const minDuration = Math.max(1, Math.min(minimumDurationTicks(), Math.max(1, beforeEnd)));

        if (drag.mode === "resize-left") {
          const maxStart = Math.max(0, beforeEnd - minDuration);
          note.start_tick = Math.max(0, Math.min(requestedTick, maxStart));
          note.end_tick = beforeEnd;
        } else {
          const minEnd = beforeStart + minDuration;
          note.start_tick = beforeStart;
          note.end_tick = Math.max(requestedTick, minEnd);
        }

        syncNoteTimingFromTicks(note);
      }

      function noteSnapshotsChanged(beforeNote, afterNote) {
        return JSON.stringify(beforeNote) !== JSON.stringify(afterNote);
      }

      function finalizeNoteEditDrag(drag) {
        if (!drag.active) {
          return;
        }

        const afterNotes = drag.noteIds
          .map(function (noteId) {
            const note = state.noteById.get(String(noteId));
            return note ? cloneNoteSnapshot(note) : null;
          })
          .filter(function (note) { return Boolean(note); });

        const changedBefore = [];
        const changedAfter = [];
        for (const after of afterNotes) {
          const before = drag.beforeById.get(String(after.note_id));
          if (!before) {
            continue;
          }
          if (noteSnapshotsChanged(before, after)) {
            changedBefore.push(cloneNoteSnapshot(before));
            changedAfter.push(cloneNoteSnapshot(after));
          }
        }

        if (!changedAfter.length) {
          return;
        }

        const selectionAfter = drag.mode === "move"
          ? drag.noteIds.slice()
          : [String(drag.targetNoteId)];

        pushHistoryTransaction({
          label: drag.mode === "move" ? "drag-move-notes" : "resize-note",
          beforeNotes: changedBefore,
          afterNotes: changedAfter,
          selectionBefore: drag.selectionBefore,
          selectionAfter: selectionAfter,
        });

        sortNotes();
        rebuildTrackSources();
        rebuildNoteLookup();
        setSelectionFromList(selectionAfter);
        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        updateEditorActionButtons();

        if (drag.mode === "move") {
          setStatus("Moved " + String(changedAfter.length) + " note(s).", false);
        } else {
          setStatus("Resized note.", false);
        }
      }

      function generateUniqueDrawnNoteId() {
        const existing = new Set(
          session.notes.map(function (note) {
            return String(note.note_id);
          })
        );

        for (let guard = 0; guard < 100000; guard += 1) {
          const candidate = DRAW_NOTE_ID_PREFIX + "_" + String(state.drawNoteCounter).padStart(6, "0");
          state.drawNoteCounter += 1;
          if (!existing.has(candidate)) {
            return candidate;
          }
        }

        return DRAW_NOTE_ID_PREFIX + "_" + String(Date.now());
      }

      function resolveTargetTrackForDraw() {
        const targetIndex = Number(targetTrackEl.value || 0);
        const selectedTrack = getTrackByIndex(targetIndex);
        if (selectedTrack) {
          return selectedTrack;
        }
        if (session.tracks.length > 0) {
          return session.tracks[0];
        }
        return null;
      }

      function resolveDefaultChannelForTrack(editableTrackIndex) {
        for (const note of session.notes) {
          if (Number(note.editable_track_index) !== Number(editableTrackIndex)) {
            continue;
          }
          if (Number.isFinite(Number(note.channel))) {
            return Math.max(0, Math.min(15, Math.round(Number(note.channel))));
          }
        }
        return 0;
      }

      function startDrawDrag(point) {
        const targetTrack = resolveTargetTrackForDraw();
        if (!targetTrack) {
          setStatus("No target track available for drawing.", true);
          return;
        }

        const snappedTick = Math.max(0, snapAbsoluteTick(tickFromCanvasX(point.x)));
        const pitch = pitchFromCanvasY(point.y);
        state.drawDrag = {
          startX: point.x,
          startY: point.y,
          currentX: point.x,
          currentY: point.y,
          startTick: snappedTick,
          currentTick: snappedTick,
          currentPitch: pitch,
          active: false,
          selectionBefore: Array.from(state.selectedNoteIds),
          trackIndex: Number(targetTrack.editable_track_index),
          trackName: String(targetTrack.name || "Track"),
          channel: resolveDefaultChannelForTrack(targetTrack.editable_track_index),
        };
      }

      function finalizeDrawDrag(drag) {
        const minDuration = minimumDurationTicks();
        let startTick = Math.round(Number(drag.startTick || 0));
        let endTick = Math.round(Number(drag.currentTick || startTick));

        if (!drag.active || endTick === startTick) {
          endTick = startTick + minDuration;
        }

        if (endTick < startTick) {
          const temp = startTick;
          startTick = endTick;
          endTick = temp;
        }

        if (endTick <= startTick) {
          endTick = startTick + minDuration;
        }

        const noteId = generateUniqueDrawnNoteId();
        const newNote = {
          note_id: noteId,
          source_track_index: Number(drag.trackIndex),
          source_track_name: String(drag.trackName),
          editable_track_index: Number(drag.trackIndex),
          editable_track_name: String(drag.trackName),
          channel: Number(drag.channel),
          pitch_midi: clampPitchMidi(drag.currentPitch),
          pitch_name: pitchNameFromMidi(clampPitchMidi(drag.currentPitch)),
          velocity: DEFAULT_DRAW_VELOCITY,
          start_tick: Math.max(0, startTick),
          end_tick: Math.max(0, endTick),
          duration_ticks: 0,
          start_sec: 0,
          end_sec: 0,
          duration_sec: 0,
          muted: false,
          metadata: {},
        };

        syncNoteTimingFromTicks(newNote);
        if (Number(newNote.end_tick) <= Number(newNote.start_tick)) {
          newNote.end_tick = Number(newNote.start_tick) + minDuration;
          syncNoteTimingFromTicks(newNote);
        }

        session.notes.push(newNote);

        pushHistoryTransaction({
          label: "draw-note",
          beforeNotes: [],
          afterNotes: [cloneNoteSnapshot(newNote)],
          selectionBefore: Array.isArray(drag.selectionBefore) ? drag.selectionBefore.slice() : [],
          selectionAfter: [String(noteId)],
        });

        sortNotes();
        rebuildTrackSources();
        rebuildNoteLookup();
        setSelectionFromList([String(noteId)]);
        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        updateEditorActionButtons();
        setStatus("Added note.", false);
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

        if (point.x < LEFT_PAD || point.x > canvas.width - RIGHT_PAD) {
          return;
        }

        if (isPointInVelocityLane(point)) {
          if (currentTool !== "select") {
            return;
          }
          const additive = Boolean(event.ctrlKey || event.metaKey);
          const hitGroup = hitTestVelocityGroupHandle(point.x, point.y);
          if (hitGroup && hitGroup.noteIds && hitGroup.noteIds.length) {
            const selectionBefore = Array.from(state.selectedNoteIds);
            if (additive) {
              const merged = Array.from(new Set(selectionBefore.concat(hitGroup.noteIds)));
              setSelectionFromList(merged);
            } else {
              setSelectionFromList(hitGroup.noteIds.slice());
            }
            const selectionAfter = Array.from(state.selectedNoteIds);
            state.focusedVelocityGroupNoteIds = hitGroup.noteIds.slice();
            state.focusedVelocityNoteId = String(hitGroup.noteIds[0]);
            setKeyboardFocusMode(KEYBOARD_FOCUS_VELOCITY, { preserveVelocityTarget: true });
            startVelocityDrag({
              mode: "group",
              noteIds: selectionAfter,
              anchorNoteId: String(hitGroup.noteIds[0]),
              startY: point.y,
            });
            redraw();
            updateEditorActionButtons();
            return;
          }

          const hitVelocityBar = hitTestVelocityBar(point.x, point.y);
          if (hitVelocityBar) {
            const noteId = String(hitVelocityBar.note.note_id);
            const wasSelected = state.selectedNoteIds.has(noteId);
            if (!(wasSelected && !additive && state.selectedNoteIds.size > 1)) {
              selectNoteById(noteId, additive);
            }

            state.focusedVelocityGroupNoteIds = null;
            state.focusedVelocityNoteId = noteId;
            setKeyboardFocusMode(KEYBOARD_FOCUS_VELOCITY, { preserveVelocityTarget: true });

            let dragNoteIds = [];
            if (additive && wasSelected) {
              dragNoteIds = [noteId];
            } else {
              const selectedInGroup = getSelectedNotes().filter(function (note) {
                return velocityGroupKeyForNote(note) === velocityGroupKeyForNote(hitVelocityBar.note);
              });
              if (selectedInGroup.length >= 2) {
                dragNoteIds = selectedInGroup.map(function (note) {
                  return String(note.note_id);
                });
              } else if (state.selectedNoteIds.size >= 2 && state.selectedNoteIds.has(noteId)) {
                dragNoteIds = Array.from(state.selectedNoteIds);
              } else {
                dragNoteIds = [noteId];
              }
            }

            startVelocityDrag({
              mode: dragNoteIds.length >= 2 ? "group" : "bar",
              noteIds: dragNoteIds,
              anchorNoteId: noteId,
              startY: point.y,
            });
            redraw();
            updateEditorActionButtons();
          } else if (!additive) {
            clearSelection();
            redraw();
            updateEditorActionButtons();
          }
          return;
        }

        if (!isPointInPianoRollArea(point)) {
          return;
        }

        if (currentTool === "draw") {
          startDrawDrag(point);
          redraw();
          return;
        }

        if (currentTool !== "select") {
          return;
        }

        const additive = Boolean(event.ctrlKey || event.metaKey);
        const hit = getNoteBoxAt(point.x, point.y);
        if (hit) {
          setKeyboardFocusMode(KEYBOARD_FOCUS_NOTES);
          startNoteEditDrag(hit, point, additive);
          updateCanvasCursorForPoint(point);
          redraw();
          return;
        }

        state.dragSelect = {
          startX: point.x,
          startY: point.y,
          currentX: point.x,
          currentY: point.y,
          active: false,
          hitNoteId: null,
          additive: additive,
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

        if (state.noteEditDrag) {
          const drag = state.noteEditDrag;
          const dx = Math.abs(point.x - drag.startX);
          const dy = Math.abs(point.y - drag.startY);
          if (!drag.active && (dx >= DRAG_THRESHOLD_PX || dy >= DRAG_THRESHOLD_PX)) {
            drag.active = true;
          }

          if (drag.active) {
            if (drag.mode === "move") {
              applyMovePreview(drag, point);
            } else {
              applyResizePreview(drag, point);
            }
            redraw();
          }
          updateCanvasCursorForPoint(point);
          return;
        }

        if (state.velocityDrag) {
          const drag = state.velocityDrag;
          const anchorNote = state.noteById.get(String(drag.anchorNoteId));
          const anchorX = anchorNote ? xForTick(Number(anchorNote.start_tick || 0)) : point.x;
          const dx = Math.abs(point.x - anchorX);
          const dy = Math.abs(point.y - drag.startY);
          if (!drag.active && (dx >= DRAG_THRESHOLD_PX || dy >= DRAG_THRESHOLD_PX)) {
            drag.active = true;
          }
          applyVelocityDragPreview(drag, point, event);
          redraw();
          updateCanvasCursorForPoint(point);
          return;
        }

        if (state.drawDrag) {
          const drag = state.drawDrag;
          drag.currentX = point.x;
          drag.currentY = point.y;
          const snappedTick = Math.max(0, snapAbsoluteTick(tickFromCanvasX(point.x)));
          drag.currentTick = snappedTick;
          drag.currentPitch = pitchFromCanvasY(point.y);

          const dx = Math.abs(drag.currentX - drag.startX);
          const dy = Math.abs(drag.currentY - drag.startY);
          if (!drag.active && (dx >= DRAG_THRESHOLD_PX || dy >= DRAG_THRESHOLD_PX)) {
            drag.active = true;
          }

          redraw();
          updateCanvasCursorForPoint(point);
          return;
        }

        if (!state.dragSelect) {
          updateCanvasCursorForPoint(point);
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

      function handleCanvasMouseUp(event) {
        const point = event ? canvasPointFromEvent(event) : null;

        if (state.panDrag) {
          state.panDrag = null;
          updateCanvasCursorForPoint(point);
          return;
        }

        if (state.noteEditDrag) {
          const drag = state.noteEditDrag;
          finalizeNoteEditDrag(drag);
          state.noteEditDrag = null;
          redraw();
          updateCanvasCursorForPoint(point);
          return;
        }

        if (state.velocityDrag) {
          const drag = state.velocityDrag;
          finalizeVelocityDrag(drag);
          state.velocityDrag = null;
          redraw();
          updateCanvasCursorForPoint(point);
          return;
        }

        if (state.drawDrag) {
          const drag = state.drawDrag;
          finalizeDrawDrag(drag);
          state.drawDrag = null;
          redraw();
          updateCanvasCursorForPoint(point);
          return;
        }

        if (!state.dragSelect) {
          updateCanvasCursorForPoint(point);
          return;
        }

        if (state.dragSelect.active) {
          const rect = {
            x: Math.min(state.dragSelect.startX, state.dragSelect.currentX),
            y: Math.min(state.dragSelect.startY, state.dragSelect.currentY),
            w: Math.abs(state.dragSelect.currentX - state.dragSelect.startX),
            h: Math.abs(state.dragSelect.currentY - state.dragSelect.startY),
          };
          const startTick = Math.max(0, snapAbsoluteTick(tickFromCanvasX(rect.x)));
          const endTick = Math.max(startTick + 1, snapAbsoluteTick(tickFromCanvasX(rect.x + rect.w)));
          state.selectionRegion = {
            startTick: Math.min(startTick, endTick),
            endTick: Math.max(startTick, endTick),
          };
          selectNotesInRect(rect, state.dragSelect.additive);
          if (state.selectedNoteIds.size > 0) {
            setKeyboardFocusMode(KEYBOARD_FOCUS_NOTES);
          } else {
            setKeyboardFocusMode(KEYBOARD_FOCUS_VIEWPORT);
          }
        } else if (state.dragSelect.hitNoteId) {
          selectNoteById(state.dragSelect.hitNoteId, state.dragSelect.additive);
          setKeyboardFocusMode(KEYBOARD_FOCUS_NOTES);
        } else if (!state.dragSelect.additive) {
          const cursorTick = Math.max(0, snapAbsoluteTick(tickFromCanvasX(state.dragSelect.currentX)));
          setPasteCursorTick(cursorTick, { snap: true });
          clearSelection();
          setKeyboardFocusMode(KEYBOARD_FOCUS_VIEWPORT);
          setStatus("Selection cleared. Paste cursor set to tick " + String(cursorTick) + ".", false);
        }

        state.dragSelect = null;
        redraw();
        updateCanvasCursorForPoint(point);
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
        document.getElementById("delete-notes-btn").addEventListener("click", deleteSelectedNotes);
        document.getElementById("mute-notes-btn").addEventListener("click", toggleMuteSelectedNotes);
        copyNotesButton.addEventListener("click", copySelectedNotes);
        pasteNotesButton.addEventListener("click", pasteCopiedNotes);
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
        toolButtons.draw.addEventListener("click", function () { setTool("draw"); });
        toolButtons.zoom.addEventListener("click", function () { setTool("zoom"); });
        toolButtons.pan.addEventListener("click", function () { setTool("pan"); });

        if (snapEnabledEl) {
          snapEnabledEl.addEventListener("change", function () {
            redraw();
            setStatus(
              "Snap " + (isSnapEnabled() ? "on" : "off") + " (" + "1/" + String(currentSnapDivision()) + ").",
              false
            );
          });
        }

        if (snapGridEl) {
          snapGridEl.addEventListener("change", function () {
            redraw();
            setStatus(
              "Snap grid: 1/" + String(currentSnapDivision()) + (isSnapEnabled() ? "" : " (snap disabled)"),
              false
            );
          });
        }

        if (velocityLaneVisibleEl) {
          velocityLaneVisibleEl.addEventListener("change", function () {
            setVelocityLaneVisible(Boolean(velocityLaneVisibleEl.checked));
            setStatus("Velocity lane " + (state.velocityLaneVisible ? "on." : "off."), false);
          });
        }

        if (velocityValuesVisibleEl) {
          velocityValuesVisibleEl.addEventListener("change", function () {
            state.velocityValuesVisible = Boolean(velocityValuesVisibleEl.checked);
            redraw();
            setStatus("Velocity values " + (state.velocityValuesVisible ? "on." : "off."), false);
          });
        }
      }

      function bindCanvasActions() {
        canvas.addEventListener("mousedown", handleCanvasMouseDown);
        window.addEventListener("mousemove", handleCanvasMouseMove);
        window.addEventListener("mouseup", handleCanvasMouseUp);
        canvas.addEventListener("mouseleave", function () {
          if (state.panDrag || state.noteEditDrag || state.velocityDrag || state.drawDrag || state.dragSelect) {
            return;
          }
          updateCanvasCursorForPoint(null);
        });
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
        state.velocityValuesVisible = false;
        if (velocityValuesVisibleEl) {
          velocityValuesVisibleEl.checked = false;
        }
        setTool("select");
        bindToolbarActions();
        bindCanvasActions();

        window.addEventListener("keydown", function (event) {
          if (isEditableTypingTarget(event.target)) {
            return;
          }

          const isModifierDown = Boolean(event.ctrlKey || event.metaKey);
          const key = String(event.key || "").toLowerCase();

          if (!isModifierDown && String(event.key || "").startsWith("Arrow")) {
            const handledArrow = handleArrowKey(event);
            if (handledArrow) {
              event.preventDefault();
              return;
            }
          }

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

          if (isModifierDown && key === "c") {
            event.preventDefault();
            copySelectedNotes();
            return;
          }

          if (isModifierDown && key === "v") {
            event.preventDefault();
            pasteCopiedNotes();
            return;
          }

          if (key === "delete" || key === "backspace") {
            event.preventDefault();
            deleteSelectedNotes();
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
          setSnapEnabled: function (enabled) {
            if (!snapEnabledEl) {
              return;
            }
            snapEnabledEl.checked = Boolean(enabled);
            redraw();
          },
          isSnapEnabled: isSnapEnabled,
          setSnapDivision: function (division) {
            if (!snapGridEl) {
              return;
            }
            snapGridEl.value = String(Math.max(1, Math.round(Number(division || 16))));
            redraw();
          },
          getSnapDivision: currentSnapDivision,
          setVelocityLaneVisible: setVelocityLaneVisible,
          getVelocityLaneVisible: function () { return state.velocityLaneVisible; },
          selectedVelocitySummary: selectedVelocitySummary,
          getVelocityBars: getVelocityBars,
          getVelocityGroups: getVelocityGroups,
          velocityBarForNoteId: velocityBarForNoteId,
          hitTestVelocityBar: hitTestVelocityBarForApi,
          setVelocityForNoteIds: function (noteIds, options) {
            const result = setVelocityForNotes(noteIds || [], options || {});
            sortNotes();
            rebuildTrackSources();
            rebuildNoteLookup();
            redraw();
            updateEditorActionButtons();
            return {
              note_ids: result.noteIds.slice(),
              changed_count: result.changedCount,
            };
          },
          copySelectedNotes: copySelectedNotes,
          pasteCopiedNotes: pasteCopiedNotes,
          getClipboardSummary: getClipboardSummary,
          setPasteCursorTick: function (tick) {
            setPasteCursorTick(tick, { snap: true });
            redraw();
          },
          getPasteCursorTick: getPasteCursorTick,
          getSelectedNoteIds: function () {
            return Array.from(state.selectedNoteIds);
          },
          setKeyboardFocusMode: setKeyboardFocusMode,
          getKeyboardFocusMode: getKeyboardFocusMode,
          panViewportByTicks: function (deltaTicks) {
            const changed = panViewportByTicks(deltaTicks);
            if (changed) {
              redraw();
            }
            return changed;
          },
          panViewportBySemitones: function (deltaRows) {
            const changed = panViewportBySemitones(deltaRows);
            if (changed) {
              redraw();
            }
            return changed;
          },
          getViewportState: getViewportState,
          moveSelectedNotesByKeyboard: moveSelectedNotesByKeyboard,
          adjustSelectedVelocityByKeyboard: adjustSelectedVelocityByKeyboard,
          moveSelectedToTrack: moveSelectedToTrack,
          mergeSelectedNotes: mergeSelectedNotes,
          deleteSelectedNotes: deleteSelectedNotes,
          toggleMuteSelectedNotes: toggleMuteSelectedNotes,
          muteSelectedNotes: function () { setMuteStateForSelected(true); },
          unmuteSelectedNotes: function () { setMuteStateForSelected(false); },
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
            return getViewportState();
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
