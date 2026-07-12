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
    payload_json = json.dumps(payload, ensure_ascii=True)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MIDI Split Preview</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #141b34;
      --grid-major: rgba(255, 255, 255, 0.20);
      --grid-minor: rgba(255, 255, 255, 0.08);
      --text: #ecf1ff;
      --muted: #9aa6d1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(1200px 600px at 20% -20%, #274074 0%, transparent 60%),
        radial-gradient(1000px 500px at 90% -10%, #1b284c 0%, transparent 65%),
        var(--bg);
      color: var(--text);
    }}
    .wrap {{
      padding: 16px;
      max-width: 1500px;
      margin: 0 auto;
    }}
    .meta {{
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.10);
      border-radius: 12px;
      padding: 12px 14px;
      margin-bottom: 12px;
      line-height: 1.5;
      font-size: 14px;
    }}
    .meta strong {{ color: var(--text); }}
    .meta span {{ color: var(--muted); }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 10px;
      font-size: 12px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 999px;
      padding: 4px 10px;
      border: 1px solid rgba(255, 255, 255, 0.10);
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
      border: 1px solid rgba(255, 255, 255, 0.35);
    }}
    .canvas-wrap {{
      overflow: auto;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      background: #0a0f1d;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
    }}
    canvas {{ display: block; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="meta" id="meta"></div>
    <div class="legend" id="legend"></div>
    <div class="canvas-wrap">
      <canvas id="roll"></canvas>
    </div>
  </div>
  <script id="session-json" type="application/json">{payload_json}</script>
  <script>
    (function () {{
      const payload = JSON.parse(document.getElementById("session-json").textContent);
      const notes = Array.isArray(payload.notes) ? payload.notes : [];
      const tracks = Array.isArray(payload.tracks) ? payload.tracks : [];
      const ticksPerBeat = Math.max(1, Number(payload.ticks_per_beat || 480));

      const pitchMin = 24;
      const pitchMax = 108;
      const rowHeight = 9;
      const leftPad = 70;
      const topPad = 24;
      const rightPad = 30;
      const bottomPad = 24;
      const minWidth = 1000;
      const tickScale = 0.20;

      const maxTick = notes.reduce((acc, note) => Math.max(acc, Number(note.end_tick || 0)), 0);
      const totalTicks = Math.max(maxTick + ticksPerBeat * 4, ticksPerBeat * 8);
      const gridRows = pitchMax - pitchMin + 1;

      const canvas = document.getElementById("roll");
      canvas.width = Math.max(minWidth, Math.ceil(leftPad + totalTicks * tickScale + rightPad));
      canvas.height = Math.ceil(topPad + gridRows * rowHeight + bottomPad);

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const palette = [
        "#f6bd60", "#84a59d", "#f28482", "#8ecae6", "#90be6d", "#b388eb",
        "#f4a261", "#4cc9f0", "#ffafcc", "#95d5b2", "#ff9f1c", "#a8dadc"
      ];

      const colorByTrack = new Map();
      tracks.forEach((track, index) => {{
        const key = Number(track.editable_track_index);
        const color = palette[index % palette.length];
        colorByTrack.set(key, color);
      }});

      const meta = document.getElementById("meta");
      meta.innerHTML =
        `<strong>Source MIDI:</strong> <span>${{payload.source_midi || "-"}}</span><br/>` +
        `<strong>Tracks:</strong> <span>${{tracks.length}}</span> &nbsp; ` +
        `<strong>Notes:</strong> <span>${{notes.length}}</span> &nbsp; ` +
        `<strong>Ticks/beat:</strong> <span>${{ticksPerBeat}}</span>`;

      const legend = document.getElementById("legend");
      tracks
        .slice()
        .sort((a, b) => Number(a.editable_track_index) - Number(b.editable_track_index))
        .forEach((track) => {{
          const index = Number(track.editable_track_index);
          const item = document.createElement("div");
          item.className = "legend-item";
          item.innerHTML =
            `<span class="swatch" style="background:${{colorByTrack.get(index)}}"></span>` +
            `<span>${{String(index).padStart(2, "0")}} - ${{track.name || "Track"}}</span>`;
          legend.appendChild(item);
        }});

      function yForPitch(pitch) {{
        return topPad + (pitchMax - pitch) * rowHeight;
      }}

      function xForTick(tick) {{
        return leftPad + tick * tickScale;
      }}

      ctx.fillStyle = "#0a0f1d";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (let pitch = pitchMin; pitch <= pitchMax; pitch += 1) {{
        const y = yForPitch(pitch);
        ctx.fillStyle = pitch % 12 === 0 ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.03)";
        ctx.fillRect(leftPad, y, canvas.width - leftPad - rightPad, rowHeight);
      }}

      const beatTicks = ticksPerBeat;
      const barTicks = ticksPerBeat * 4;
      for (let tick = 0; tick <= totalTicks; tick += beatTicks) {{
        const x = Math.round(xForTick(tick)) + 0.5;
        const isBar = tick % barTicks === 0;
        ctx.strokeStyle = isBar ? "var(--grid-major)" : "var(--grid-minor)";
        ctx.beginPath();
        ctx.moveTo(x, topPad);
        ctx.lineTo(x, canvas.height - bottomPad);
        ctx.stroke();
      }}

      ctx.fillStyle = "#d4dcff";
      ctx.font = "11px Segoe UI";
      for (let pitch = pitchMin; pitch <= pitchMax; pitch += 12) {{
        const y = yForPitch(pitch) + rowHeight - 1;
        ctx.fillText(String(pitch), 10, y);
      }}

      const sortedNotes = notes.slice().sort((a, b) => Number(a.start_tick) - Number(b.start_tick));
      sortedNotes.forEach((note) => {{
        const startTick = Number(note.start_tick || 0);
        const endTick = Math.max(startTick, Number(note.end_tick || startTick));
        const pitch = Number(note.pitch_midi || 0);
        if (pitch < pitchMin || pitch > pitchMax) return;

        const x = xForTick(startTick);
        const y = yForPitch(pitch);
        const width = Math.max(1, (endTick - startTick) * tickScale);
        const color = colorByTrack.get(Number(note.editable_track_index)) || "#f6bd60";

        ctx.fillStyle = color;
        ctx.globalAlpha = 0.88;
        ctx.fillRect(x, y + 1, width, Math.max(3, rowHeight - 2));

        ctx.strokeStyle = "rgba(0,0,0,0.45)";
        ctx.globalAlpha = 1;
        ctx.strokeRect(x + 0.5, y + 1.5, Math.max(0, width - 1), Math.max(2, rowHeight - 3));
      }});
    }})();
  </script>
</body>
</html>
"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
