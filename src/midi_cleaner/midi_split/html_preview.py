from __future__ import annotations

import json
from pathlib import Path

from midi_cleaner.midi_split.models import MidiSplitSession


def generate_piano_roll_preview(session: MidiSplitSession, output_html: Path) -> None:
    payload = {
        "schema_version": session.schema_version,
        "source_midi": session.source_midi,
        "ticks_per_beat": int(session.ticks_per_beat),
        "tracks": [track.model_dump(mode="json") for track in session.tracks],
        "notes": [note.model_dump(mode="json") for note in session.notes],
    }
    payload_json = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
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
    #toolbar select {
      padding: 4px 8px;
      font-size: 12px;
    }
    #status-line {
      padding: 4px 8px;
      border-bottom: 1px solid #333;
      min-height: 24px;
      font-size: 12px;
      color: #d9d9d9;
      background: #1d1d1d;
    }
    #layout {
      display: flex;
      height: calc(100vh - 78px);
      min-height: 420px;
    }
    #track-panel {
      width: 260px;
      min-width: 220px;
      max-width: 320px;
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
    }
    #roll-canvas {
      display: block;
      background: #111;
    }
  </style>
</head>
<body>
  <div id="toolbar">
    <strong>Source:</strong>
    <span id="source-name">-</span>
    <span>|</span>
    <span>Selected notes: <strong id="selected-count">0</strong></span>
    <label for="target-track">Target track:</label>
    <select id="target-track"></select>
    <button id="move-selected-btn" type="button">Move selected to track</button>
    <button id="add-track-btn" type="button">Add track</button>
    <button id="merge-tracks-btn" type="button">Merge selected tracks</button>
    <button id="download-session-btn" type="button">Download updated session JSON</button>
    <button id="clear-selection-btn" type="button">Clear selection</button>
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
      const LEFT_PAD = 72;
      const TOP_PAD = 24;
      const RIGHT_PAD = 24;
      const BOTTOM_PAD = 24;
      const MIN_CANVAS_WIDTH = 1000;
      const NOTE_ROW_HEIGHT = 10;
      const TICK_SCALE = 0.18;
      const DRAG_THRESHOLD_PX = 4;
      const palette = [
        "#ff6f61", "#6fcf97", "#56ccf2", "#f2c94c", "#bb6bd9", "#f2994a",
        "#2d9cdb", "#9b51e0", "#27ae60", "#eb5757", "#219ebc", "#f77f00"
      ];

      const session = JSON.parse(document.getElementById("session-json").textContent);
      if (!Array.isArray(session.tracks)) {
        session.tracks = [];
      }
      if (!Array.isArray(session.notes)) {
        session.notes = [];
      }

      const sourceNameEl = document.getElementById("source-name");
      const selectedCountEl = document.getElementById("selected-count");
      const targetTrackEl = document.getElementById("target-track");
      const statusEl = document.getElementById("status-line");
      const trackPanelEl = document.getElementById("track-panel");
      const canvas = document.getElementById("roll-canvas");
      const ctx = canvas.getContext("2d");

      const state = {
        selectedNoteIds: new Set(),
        mergeTrackIndices: new Set(),
        noteBoxes: [],
        drag: null,
        pitchMin: 24,
        pitchMax: 108,
      };

      function setStatus(text) {
        statusEl.textContent = text;
      }

      function sortTracks() {
        session.tracks.sort(function (a, b) {
          return Number(a.editable_track_index) - Number(b.editable_track_index);
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

      function colorForTrack(trackIndex) {
        const normalized = Math.max(1, Number(trackIndex));
        return palette[(normalized - 1) % palette.length];
      }

      function clearSelection() {
        state.selectedNoteIds.clear();
        updateSelectionUi();
        redraw();
      }

      function updateSelectionUi() {
        selectedCountEl.textContent = String(state.selectedNoteIds.size);
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

      function updateTargetTrackDropdown() {
        const previousValue = Number(targetTrackEl.value || 0);
        targetTrackEl.innerHTML = "";

        for (const track of session.tracks) {
          const option = document.createElement("option");
          option.value = String(track.editable_track_index);
          option.textContent = String(track.editable_track_index).padStart(2, "0") + " - " + track.name;
          targetTrackEl.appendChild(option);
        }

        if (getTrackByIndex(previousValue) !== null) {
          targetTrackEl.value = String(previousValue);
        } else if (session.tracks.length > 0) {
          targetTrackEl.value = String(session.tracks[0].editable_track_index);
        }
      }

      function rebuildTrackSources() {
        const sourcesByTrack = new Map();
        for (const track of session.tracks) {
          sourcesByTrack.set(Number(track.editable_track_index), new Set());
        }

        for (const note of session.notes) {
          const trackIndex = Number(note.editable_track_index);
          if (!sourcesByTrack.has(trackIndex)) {
            sourcesByTrack.set(trackIndex, new Set());
          }
          sourcesByTrack.get(trackIndex).add(Number(note.source_track_index));
        }

        for (const track of session.tracks) {
          const values = Array.from(sourcesByTrack.get(Number(track.editable_track_index)) || []);
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
            const idx = Number(track.editable_track_index);
            if (checkbox.checked) {
              state.mergeTrackIndices.add(idx);
            } else {
              state.mergeTrackIndices.delete(idx);
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

      function yForPitch(pitch) {
        return TOP_PAD + (state.pitchMax - pitch) * NOTE_ROW_HEIGHT;
      }

      function xForTick(tick) {
        return LEFT_PAD + Number(tick) * TICK_SCALE;
      }

      function updateCanvasSize() {
        const ticksPerBeat = Math.max(1, Number(session.ticks_per_beat || 480));
        let maxTick = ticksPerBeat * 8;
        for (const note of session.notes) {
          maxTick = Math.max(maxTick, Number(note.end_tick || 0));
        }

        const width = Math.max(MIN_CANVAS_WIDTH, Math.ceil(LEFT_PAD + maxTick * TICK_SCALE + RIGHT_PAD));
        const rows = state.pitchMax - state.pitchMin + 1;
        const height = Math.max(420, Math.ceil(TOP_PAD + rows * NOTE_ROW_HEIGHT + BOTTOM_PAD));
        canvas.width = width;
        canvas.height = height;
      }

      function rebuildNoteBoxes() {
        state.noteBoxes = [];
        const notes = session.notes.slice();
        notes.sort(function (a, b) {
          if (Number(a.start_tick) !== Number(b.start_tick)) {
            return Number(a.start_tick) - Number(b.start_tick);
          }
          if (Number(a.pitch_midi) !== Number(b.pitch_midi)) {
            return Number(a.pitch_midi) - Number(b.pitch_midi);
          }
          return String(a.note_id).localeCompare(String(b.note_id));
        });

        for (const note of notes) {
          const pitch = Number(note.pitch_midi || 0);
          if (pitch < state.pitchMin || pitch > state.pitchMax) {
            continue;
          }

          const startTick = Number(note.start_tick || 0);
          const endTick = Math.max(startTick, Number(note.end_tick || startTick));
          const x = xForTick(startTick);
          const y = yForPitch(pitch) + 1;
          const w = Math.max(1, (endTick - startTick) * TICK_SCALE);
          const h = Math.max(3, NOTE_ROW_HEIGHT - 2);
          state.noteBoxes.push({ x: x, y: y, w: w, h: h, note: note });
        }
      }

      function drawGrid() {
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        for (let pitch = state.pitchMin; pitch <= state.pitchMax; pitch += 1) {
          const y = yForPitch(pitch);
          ctx.fillStyle = pitch % 12 === 0 ? "#1a1a1a" : "#141414";
          ctx.fillRect(LEFT_PAD, y, canvas.width - LEFT_PAD - RIGHT_PAD, NOTE_ROW_HEIGHT);
        }

        const ticksPerBeat = Math.max(1, Number(session.ticks_per_beat || 480));
        const barTicks = ticksPerBeat * 4;
        const maxTicks = Math.max(0, Math.floor((canvas.width - LEFT_PAD - RIGHT_PAD) / TICK_SCALE));
        for (let tick = 0; tick <= maxTicks; tick += ticksPerBeat) {
          const x = Math.round(xForTick(tick)) + 0.5;
          const isBar = tick % barTicks === 0;
          ctx.strokeStyle = isBar ? "#4b4b4b" : "#2f2f2f";
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
        if (!state.drag || !state.drag.active) {
          return;
        }

        const x = Math.min(state.drag.startX, state.drag.currentX);
        const y = Math.min(state.drag.startY, state.drag.currentY);
        const w = Math.abs(state.drag.currentX - state.drag.startX);
        const h = Math.abs(state.drag.currentY - state.drag.startY);

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
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        return {
          x: Math.max(0, Math.min(canvas.width, x)),
          y: Math.max(0, Math.min(canvas.height, y)),
        };
      }

      function handleCanvasMouseDown(event) {
        if (event.button !== 0) {
          return;
        }

        const point = canvasPointFromEvent(event);
        const hit = getNoteBoxAt(point.x, point.y);
        state.drag = {
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
        if (!state.drag) {
          return;
        }

        const point = canvasPointFromEvent(event);
        state.drag.currentX = point.x;
        state.drag.currentY = point.y;

        const dx = Math.abs(state.drag.currentX - state.drag.startX);
        const dy = Math.abs(state.drag.currentY - state.drag.startY);
        if (dx >= DRAG_THRESHOLD_PX || dy >= DRAG_THRESHOLD_PX) {
          state.drag.active = true;
        }

        redraw();
      }

      function handleCanvasMouseUp() {
        if (!state.drag) {
          return;
        }

        if (state.drag.active) {
          const rect = {
            x: Math.min(state.drag.startX, state.drag.currentX),
            y: Math.min(state.drag.startY, state.drag.currentY),
            w: Math.abs(state.drag.currentX - state.drag.startX),
            h: Math.abs(state.drag.currentY - state.drag.startY),
          };
          selectNotesInRect(rect, state.drag.additive);
        } else if (state.drag.hitNoteId) {
          selectNoteById(state.drag.hitNoteId, state.drag.additive);
        } else if (!state.drag.additive) {
          clearSelection();
        }

        state.drag = null;
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

        let movedCount = 0;
        for (const note of session.notes) {
          if (state.selectedNoteIds.has(String(note.note_id))) {
            note.editable_track_index = Number(targetTrack.editable_track_index);
            note.editable_track_name = String(targetTrack.name);
            movedCount += 1;
          }
        }

        rebuildTrackSources();
        renderTrackPanel();
        redraw();
        setStatus("Moved " + String(movedCount) + " notes to track " + String(targetTrack.editable_track_index) + ".");
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
          setStatus("Cannot add track: maximum track count reached (12).");
          alert("Maximum editable track count reached (12).");
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
          const idx = Number(track.editable_track_index);
          return idx === targetIndex || !selectedSet.has(idx);
        });

        sortTracks();
        rebuildTrackSources();
        state.mergeTrackIndices.clear();
        updateTargetTrackDropdown();
        renderTrackPanel();
        redraw();
        setStatus("Merged tracks into track " + String(targetIndex) + ". Moved " + String(moved) + " notes.");
      }

      function downloadSessionJson() {
        const pretty = JSON.stringify(session, null, 2) + "\\n";
        const blob = new Blob([pretty], { type: "application/json" });
        const href = URL.createObjectURL(blob);

        const sourceName = String(session.source_midi || "split_session").split(/[/\\\\]/).pop() || "split_session";
        const baseName = sourceName.replace(/\\.[^.]+$/, "") || "split_session";
        const filename = baseName + "_split_session_updated.json";

        const link = document.createElement("a");
        link.href = href;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(href);

        setStatus("Downloaded " + filename + ".");
      }

      function initialize() {
        sortTracks();
        sourceNameEl.textContent = String(session.source_midi || "-");
        updateTargetTrackDropdown();
        renderTrackPanel();
        updateSelectionUi();
        redraw();

        document.getElementById("move-selected-btn").addEventListener("click", moveSelectedToTrack);
        document.getElementById("add-track-btn").addEventListener("click", addTrack);
        document.getElementById("merge-tracks-btn").addEventListener("click", mergeSelectedTracks);
        document.getElementById("download-session-btn").addEventListener("click", downloadSessionJson);
        document.getElementById("clear-selection-btn").addEventListener("click", function () {
          clearSelection();
          setStatus("Selection cleared.");
        });

        canvas.addEventListener("mousedown", handleCanvasMouseDown);
        window.addEventListener("mousemove", handleCanvasMouseMove);
        window.addEventListener("mouseup", handleCanvasMouseUp);

        window.addEventListener("keydown", function (event) {
          if (event.key === "Escape") {
            clearSelection();
            setStatus("Selection cleared.");
          }
        });

        window.__midiSplitEditor = {
          getSession: function () { return JSON.parse(JSON.stringify(session)); },
          getSelectedCount: function () { return state.selectedNoteIds.size; },
          selectNotesByIds: function (ids) {
            state.selectedNoteIds.clear();
            for (const id of ids) {
              state.selectedNoteIds.add(String(id));
            }
            updateSelectionUi();
            redraw();
          },
          moveSelectedToTrack: moveSelectedToTrack,
          addTrack: addTrack,
          mergeSelectedTracks: mergeSelectedTracks,
          clearSelection: clearSelection,
          downloadSessionJson: downloadSessionJson,
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
      }

      initialize();
    })();
  </script>
</body>
</html>
"""
    html = template.replace("__SESSION_JSON__", payload_json)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
