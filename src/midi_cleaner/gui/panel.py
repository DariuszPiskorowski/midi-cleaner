from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
import json
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from midi_cleaner.gui.controller import (
    ACTION_MAKE_MIDI_FROM_WAV,
    _DRUMS_DETECTION_MODES,
    _DRUMS_OUTPUT_LAYOUTS,
    _DRUMS_PROFILES,
    ACTION_SET_BPM,
    ACTION_SYNCHRONIZE_MIDI_WITH_WAV,
    HermesDrumsRequest,
    ROLE_BASS,
    ROLE_DRUMS,
    ROLE_GUITAR,
    ROLE_OTHER,
    ROLE_SYNTH,
    HermesActionRequest,
    HermesGuiController,
)

_ROLE_OPTIONS: list[tuple[str, str]] = [
    ("drums", ROLE_DRUMS),
    ("bass", ROLE_BASS),
    ("synth (coming soon)", ROLE_SYNTH),
    ("guitar (coming soon)", ROLE_GUITAR),
    ("other (coming soon)", ROLE_OTHER),
]

_ACTION_OPTIONS: list[tuple[str, str]] = [
    ("make MIDI from WAV", ACTION_MAKE_MIDI_FROM_WAV),
    ("synchronize existing MIDI with WAV", ACTION_SYNCHRONIZE_MIDI_WITH_WAV),
    ("set/fix BPM", ACTION_SET_BPM),
]


class HermesGuiPanel:
    def __init__(self, controller: HermesGuiController) -> None:
        self._controller = controller
        self._root = tk.Tk()
        self._root.title("Hermes Workflow Panel")
        self._root.geometry("860x600")

        self._role_by_label = {label: value for label, value in _ROLE_OPTIONS}
        self._action_by_label = {label: value for label, value in _ACTION_OPTIONS}

        self._wav_var = tk.StringVar()
        self._midi_var = tk.StringVar()
        self._role_var = tk.StringVar(value=_ROLE_OPTIONS[0][0])
        self._action_var = tk.StringVar(value=_ACTION_OPTIONS[0][0])
        self._bpm_var = tk.StringVar()
        self._status_var = tk.StringVar(value="Ready")
        self._output_var = tk.StringVar(value="")
        self._report_var = tk.StringVar(value="")
        self._debug_csv_var = tk.StringVar(value="")
        self._output_dir_var = tk.StringVar(value="")
        self._mapping_file_var = tk.StringVar(value="")
        self._mapping_name_var = tk.StringVar(value="")
        self._created_files_count_var = tk.StringVar(value="0")
        self._output_layout_var = tk.StringVar(value="separate-files")
        self._profile_var = tk.StringVar(value="conservative")
        self._detection_mode_var = tk.StringVar(value="multi-detector")
        self._write_empty_layers_var = tk.BooleanVar(value=False)
        self._clean_output_folder_var = tk.BooleanVar(value=False)
        self._c1_midi_note_var = tk.StringVar(value="36")

        self._drums_mapping_payload: dict[str, object] | None = None
        self._mapping_table_rows: list[dict[str, object]] = []
        self._mapping_table_row_vars: dict[str, dict[str, tk.Variable]] = {}

        self._running = False
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._log_poll_after_id: str | None = None

        self._build_ui()
        self._bind_events()
        self._refresh_field_visibility()
        self._schedule_log_poll()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self._root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(frame, text="WAV file").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self._wav_var).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse", command=self._browse_wav).grid(row=row, column=2, sticky="ew")

        row += 1
        self._midi_label = ttk.Label(frame, text="MIDI file")
        self._midi_label.grid(row=row, column=0, sticky="w")
        self._midi_entry = ttk.Entry(frame, textvariable=self._midi_var)
        self._midi_entry.grid(row=row, column=1, sticky="ew", padx=8)
        self._midi_button = ttk.Button(frame, text="Browse", command=self._browse_midi)
        self._midi_button.grid(row=row, column=2, sticky="ew")

        row += 1
        ttk.Label(frame, text="Role").grid(row=row, column=0, sticky="w")
        self._role_combo = ttk.Combobox(
            frame,
            values=[label for label, _ in _ROLE_OPTIONS],
            textvariable=self._role_var,
            state="readonly",
        )
        self._role_combo.grid(row=row, column=1, sticky="ew", padx=8)

        row += 1
        ttk.Label(frame, text="Action").grid(row=row, column=0, sticky="w")
        self._action_combo = ttk.Combobox(
            frame,
            values=[label for label, _ in _ACTION_OPTIONS],
            textvariable=self._action_var,
            state="readonly",
        )
        self._action_combo.grid(row=row, column=1, sticky="ew", padx=8)

        row += 1
        self._bpm_label = ttk.Label(frame, text="BPM override")
        self._bpm_label.grid(row=row, column=0, sticky="w")
        self._bpm_entry = ttk.Entry(frame, textvariable=self._bpm_var)
        self._bpm_entry.grid(row=row, column=1, sticky="ew", padx=8)

        row += 1
        button_row = ttk.Frame(frame)
        button_row.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self._run_button = ttk.Button(button_row, text="Run", command=self._run_action)
        self._run_button.pack(side=tk.LEFT)
        ttk.Button(
            button_row,
            text="Open Desktop/output folder",
            command=self._open_output_folder,
        ).pack(side=tk.LEFT, padx=(8, 0))

        row += 1
        ttk.Label(frame, text="Status").grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Label(frame, textvariable=self._status_var).grid(
            row=row,
            column=1,
            columnspan=2,
            sticky="w",
            pady=(10, 0),
        )

        row += 1
        ttk.Label(frame, text="Output MIDI").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self._output_var, state="readonly").grid(
            row=row,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=8,
        )

        row += 1
        ttk.Label(frame, text="Report JSON").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self._report_var, state="readonly").grid(
            row=row,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=8,
        )

        row += 1
        self._debug_csv_label = ttk.Label(frame, text="Debug CSV")
        self._debug_csv_label.grid(row=row, column=0, sticky="w")
        self._debug_csv_entry = ttk.Entry(frame, textvariable=self._debug_csv_var, state="readonly")
        self._debug_csv_entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=8)

        row += 1
        self._created_count_label = ttk.Label(frame, text="Created MIDI files")
        self._created_count_label.grid(row=row, column=0, sticky="w")
        self._created_count_entry = ttk.Entry(frame, textvariable=self._created_files_count_var, state="readonly")
        self._created_count_entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=8)

        row += 1
        self._created_files_label = ttk.Label(frame, text="Created files list")
        self._created_files_label.grid(row=row, column=0, sticky="nw")
        self._created_files_text = ScrolledText(frame, height=5, wrap=tk.WORD)
        self._created_files_text.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=8)

        row += 1
        self._drums_panel = ttk.LabelFrame(frame, text="Drums")
        self._drums_panel.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        self._build_drums_panel(self._drums_panel)

        row += 1
        ttk.Label(frame, text="Log").grid(row=row, column=0, sticky="w", pady=(10, 0))

        row += 1
        self._log_text = ScrolledText(frame, height=16, wrap=tk.WORD)
        self._log_text.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(2, 0))

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row - 2, weight=1)
        frame.rowconfigure(row, weight=1)

    def _build_drums_panel(self, parent: ttk.LabelFrame) -> None:
        parent.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(parent, text="Output folder").grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self._output_dir_var).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Browse", command=self._browse_output_dir).grid(row=row, column=2, sticky="ew")

        row += 1
        ttk.Label(parent, text="Output layout").grid(row=row, column=0, sticky="w")
        layout_frame = ttk.Frame(parent)
        layout_frame.grid(row=row, column=1, columnspan=2, sticky="w")
        for value in ["separate-files", "multitrack", "single-track"]:
            label = {
                "separate-files": "Separate MIDI per instrument/layer",
                "multitrack": "One multitrack MIDI",
                "single-track": "One single-track MIDI",
            }[value]
            ttk.Radiobutton(
                layout_frame,
                text=label,
                value=value,
                variable=self._output_layout_var,
            ).pack(anchor="w")

        row += 1
        ttk.Label(parent, text="Profile").grid(row=row, column=0, sticky="w")
        self._drums_profile_combo = ttk.Combobox(
            parent,
            values=sorted(_DRUMS_PROFILES),
            textvariable=self._profile_var,
            state="readonly",
            width=18,
        )
        self._drums_profile_combo.grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        ttk.Label(parent, text="Detection mode").grid(row=row, column=0, sticky="w")
        self._drums_detection_combo = ttk.Combobox(
            parent,
            values=sorted(_DRUMS_DETECTION_MODES),
            textvariable=self._detection_mode_var,
            state="readonly",
            width=18,
        )
        self._drums_detection_combo.grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        ttk.Label(parent, text="C1 MIDI note").grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self._c1_midi_note_var, width=8).grid(row=row, column=1, sticky="w", padx=8)

        row += 1
        ttk.Checkbutton(
            parent,
            text="Write empty layers",
            variable=self._write_empty_layers_var,
        ).grid(row=row, column=0, sticky="w")
        ttk.Checkbutton(
            parent,
            text="Clean output folder before run",
            variable=self._clean_output_folder_var,
        ).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(parent, text="Mapping file").grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self._mapping_file_var).grid(row=row, column=1, sticky="ew", padx=8)
        mapping_buttons = ttk.Frame(parent)
        mapping_buttons.grid(row=row, column=2, sticky="ew")
        ttk.Button(mapping_buttons, text="Browse", command=self._browse_mapping_file).pack(fill=tk.X)

        row += 1
        mapping_action_row = ttk.Frame(parent)
        mapping_action_row.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(mapping_action_row, text="Load Mapping JSON", command=self._load_mapping_from_file).pack(side=tk.LEFT)
        ttk.Button(mapping_action_row, text="Save Mapping JSON", command=self._save_mapping_to_file).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(mapping_action_row, text="Use default expanded mapping", command=self._load_default_mapping).pack(
            side=tk.LEFT,
            padx=(8, 0),
        )

        row += 1
        ttk.Label(parent, text="Mapping name").grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self._mapping_name_var, state="readonly").grid(
            row=row,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=8,
        )

        row += 1
        ttk.Label(parent, text="Mapping table").grid(row=row, column=0, sticky="nw")
        self._mapping_table = ttk.Treeview(
            parent,
            columns=("enabled", "track_name", "target_note"),
            show="tree headings",
            height=10,
        )
        self._mapping_table.heading("#0", text="Semantic Layer")
        self._mapping_table.heading("enabled", text="Enabled")
        self._mapping_table.heading("track_name", text="Track Name")
        self._mapping_table.heading("target_note", text="Target Note")
        self._mapping_table.column("#0", width=190)
        self._mapping_table.column("enabled", width=90)
        self._mapping_table.column("track_name", width=130)
        self._mapping_table.column("target_note", width=120)
        self._mapping_table.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=8)

        row += 1
        editor_row = ttk.Frame(parent)
        editor_row.grid(row=row, column=1, columnspan=2, sticky="ew", padx=8, pady=(4, 0))
        ttk.Label(editor_row, text="Layer").grid(row=0, column=0, sticky="w")
        self._mapping_selected_layer_var = tk.StringVar(value="")
        self._mapping_layer_combo = ttk.Combobox(editor_row, textvariable=self._mapping_selected_layer_var, state="readonly", width=26)
        self._mapping_layer_combo.grid(row=0, column=1, sticky="w", padx=(4, 12))

        self._mapping_enabled_edit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(editor_row, text="Enabled", variable=self._mapping_enabled_edit_var).grid(row=0, column=2, sticky="w")

        ttk.Label(editor_row, text="Track").grid(row=0, column=3, sticky="w", padx=(12, 0))
        self._mapping_track_edit_var = tk.StringVar(value="")
        ttk.Entry(editor_row, textvariable=self._mapping_track_edit_var, width=14).grid(row=0, column=4, sticky="w", padx=(4, 0))

        ttk.Label(editor_row, text="Note").grid(row=0, column=5, sticky="w", padx=(12, 0))
        self._mapping_note_edit_var = tk.StringVar(value="")
        ttk.Entry(editor_row, textvariable=self._mapping_note_edit_var, width=10).grid(row=0, column=6, sticky="w", padx=(4, 0))

        ttk.Button(editor_row, text="Apply", command=self._apply_mapping_editor_changes).grid(row=0, column=7, padx=(12, 0), sticky="w")

        row += 1
        drums_run_row = ttk.Frame(parent)
        drums_run_row.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Button(drums_run_row, text="Run Drums Extraction", command=self._run_action).pack(side=tk.LEFT)
        ttk.Button(drums_run_row, text="Open Output Folder", command=self._open_output_folder).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(drums_run_row, text="Open Report JSON", command=self._open_report_json).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(drums_run_row, text="Open Debug CSV", command=self._open_debug_csv).pack(side=tk.LEFT, padx=(8, 0))

        parent.rowconfigure(row - 1, weight=1)

        self._load_default_mapping(silent=True)

    def _bind_events(self) -> None:
        self._role_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_field_visibility())
        self._action_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_field_visibility())
        self._mapping_layer_combo.bind("<<ComboboxSelected>>", lambda _event: self._load_selected_layer_into_editor())

    def _selected_role(self) -> str:
        return self._role_by_label[self._role_var.get()]

    def _selected_action(self) -> str:
        return self._action_by_label[self._action_var.get()]

    def _refresh_field_visibility(self) -> None:
        role = self._selected_role()
        action = self._selected_action()
        is_drums_make = role == ROLE_DRUMS and action == ACTION_MAKE_MIDI_FROM_WAV

        midi_required = self._controller.requires_midi(role=role, action=action)
        if midi_required:
            self._midi_label.grid()
            self._midi_entry.grid()
            self._midi_button.grid()
        else:
            self._midi_label.grid_remove()
            self._midi_entry.grid_remove()
            self._midi_button.grid_remove()

        bpm_supported = self._controller.supports_bpm(role=role, action=action)
        if bpm_supported:
            self._bpm_label.grid()
            self._bpm_entry.grid()
        else:
            self._bpm_label.grid_remove()
            self._bpm_entry.grid_remove()
            self._bpm_var.set("")

        if is_drums_make:
            self._drums_panel.grid()
            self._output_var.set("")
        else:
            self._drums_panel.grid_remove()

        if is_drums_make:
            self._created_count_label.grid()
            self._created_count_entry.grid()
            self._created_files_label.grid()
            self._created_files_text.grid()
            self._debug_csv_label.grid()
            self._debug_csv_entry.grid()
        else:
            self._created_count_label.grid_remove()
            self._created_count_entry.grid_remove()
            self._created_files_label.grid_remove()
            self._created_files_text.grid_remove()
            self._debug_csv_label.grid_remove()
            self._debug_csv_entry.grid_remove()

    def _append_log(self, message: str) -> None:
        self._log_text.insert(tk.END, message + "\n")
        self._log_text.see(tk.END)

    def _log(self, message: str) -> None:
        self._log_queue.put(message)

    def _schedule_log_poll(self) -> None:
        self._poll_log_queue()
        self._log_poll_after_id = self._root.after(150, self._schedule_log_poll)

    def _poll_log_queue(self) -> None:
        while True:
            try:
                message = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(message)

    def _browse_wav(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if selected:
            self._wav_var.set(selected)

    def _browse_midi(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select MIDI file",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
        )
        if selected:
            self._midi_var.set(selected)

    def _browse_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="Select output folder")
        if selected:
            self._output_dir_var.set(selected)

    def _browse_mapping_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select mapping JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self._mapping_file_var.set(selected)

    def _populate_mapping_table(self) -> None:
        self._mapping_table.delete(*self._mapping_table.get_children())
        self._mapping_table_rows = []
        self._mapping_table_row_vars.clear()

        payload = self._drums_mapping_payload
        if payload is None:
            return

        layers_obj = payload.get("layers")
        if not isinstance(layers_obj, dict):
            return

        for layer_name in sorted(layers_obj):
            layer = layers_obj[layer_name]
            if not isinstance(layer, dict):
                continue

            enabled = bool(layer.get("enabled", True))
            track_name = str(layer.get("track_name", ""))
            note_name = str(layer.get("note_name", layer.get("note", "")))
            self._mapping_table.insert(
                "",
                tk.END,
                iid=layer_name,
                text=layer_name,
                values=("yes" if enabled else "no", track_name, note_name),
            )
            self._mapping_table_rows.append({
                "layer": layer_name,
                "enabled": enabled,
                "track_name": track_name,
                "note_name": note_name,
            })

        self._mapping_layer_combo.configure(values=[row["layer"] for row in self._mapping_table_rows])
        if self._mapping_table_rows:
            first_layer = str(self._mapping_table_rows[0]["layer"])
            self._mapping_selected_layer_var.set(first_layer)
            self._load_selected_layer_into_editor()

    def _load_selected_layer_into_editor(self) -> None:
        payload = self._drums_mapping_payload
        if payload is None:
            return
        layers_obj = payload.get("layers")
        if not isinstance(layers_obj, dict):
            return

        layer_name = self._mapping_selected_layer_var.get().strip()
        layer = layers_obj.get(layer_name)
        if not isinstance(layer, dict):
            return

        self._mapping_enabled_edit_var.set(bool(layer.get("enabled", True)))
        self._mapping_track_edit_var.set(str(layer.get("track_name", "")))
        self._mapping_note_edit_var.set(str(layer.get("note_name", layer.get("note", ""))))

    def _apply_mapping_editor_changes(self) -> None:
        payload = self._drums_mapping_payload
        if payload is None:
            return

        layers_obj = payload.get("layers")
        if not isinstance(layers_obj, dict):
            return

        layer_name = self._mapping_selected_layer_var.get().strip()
        layer = layers_obj.get(layer_name)
        if not isinstance(layer, dict):
            return

        track_name = self._mapping_track_edit_var.get().strip()
        note_name = self._mapping_note_edit_var.get().strip()

        if not track_name:
            self._status_var.set("Track name cannot be empty.")
            return
        if not note_name:
            self._status_var.set("Target note cannot be empty.")
            return

        layer["enabled"] = bool(self._mapping_enabled_edit_var.get())
        layer["track_name"] = track_name
        layer["note_name"] = note_name
        layer.pop("note", None)
        self._populate_mapping_table()
        self._status_var.set(f"Updated mapping layer: {layer_name}")

    def _load_default_mapping(self, silent: bool = False) -> None:
        try:
            c1_note = int(self._c1_midi_note_var.get().strip() or "36")
        except ValueError:
            self._status_var.set("C1 MIDI note must be a valid integer.")
            return

        mapping_payload = self._controller.default_drums_mapping(
            target_map="ujam-candy",
            c1_midi_note=c1_note,
        )
        self._drums_mapping_payload = mapping_payload
        self._mapping_name_var.set(str(mapping_payload.get("name", "")))
        self._populate_mapping_table()
        if not silent:
            self._status_var.set("Loaded default expanded mapping.")

    def _load_mapping_from_file(self) -> None:
        mapping_path_text = self._mapping_file_var.get().strip()
        if not mapping_path_text:
            self._status_var.set("Select a mapping file first.")
            return

        try:
            c1_note = int(self._c1_midi_note_var.get().strip() or "36")
        except ValueError:
            self._status_var.set("C1 MIDI note must be a valid integer.")
            return

        try:
            payload = self._controller.load_drums_mapping(
                mapping_file=Path(mapping_path_text),
                fallback_c1_midi_note=c1_note,
            )
        except Exception as exc:
            self._status_var.set(str(exc))
            return

        self._drums_mapping_payload = payload
        self._mapping_name_var.set(str(payload.get("name", "")))
        self._populate_mapping_table()
        self._status_var.set("Loaded mapping JSON.")

    def _save_mapping_to_file(self) -> None:
        if self._drums_mapping_payload is None:
            self._status_var.set("No mapping is loaded.")
            return

        target = self._mapping_file_var.get().strip()
        if not target:
            selected = filedialog.asksaveasfilename(
                title="Save mapping JSON",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not selected:
                return
            target = selected
            self._mapping_file_var.set(target)

        try:
            c1_note = int(self._c1_midi_note_var.get().strip() or "36")
        except ValueError:
            self._status_var.set("C1 MIDI note must be a valid integer.")
            return

        try:
            destination = self._controller.save_drums_mapping(
                mapping_payload=self._drums_mapping_payload,
                destination_file=Path(target),
                fallback_c1_midi_note=c1_note,
            )
        except Exception as exc:
            self._status_var.set(str(exc))
            return

        self._status_var.set(f"Saved mapping JSON: {destination}")

    def _render_drums_result(self, result) -> None:
        self._output_var.set(str(result.output_file) if result.output_file else "")
        self._report_var.set(str(result.report_file) if result.report_file else "")
        self._debug_csv_var.set(str(result.debug_csv_file) if result.debug_csv_file else "")
        self._output_dir_var.set(str(result.output_dir) if result.output_dir else self._output_dir_var.get())
        self._created_files_count_var.set(str(len(result.created_files)))

        self._created_files_text.delete("1.0", tk.END)
        for path in result.created_files:
            self._created_files_text.insert(tk.END, f"{Path(path).name}\n")

        if result.mapping_name:
            self._mapping_name_var.set(result.mapping_name)

        summary_parts: list[str] = []
        if result.output_layout:
            summary_parts.append(f"layout={result.output_layout}")
        if result.mapping_name:
            summary_parts.append(f"mapping={result.mapping_name}")
        if result.populated_semantic_layers:
            summary_parts.append("populated=" + ",".join(result.populated_semantic_layers))
        if result.unpopulated_enabled_layers:
            summary_parts.append("unpopulated_enabled=" + ",".join(result.unpopulated_enabled_layers))
        if result.disabled_layers:
            summary_parts.append("disabled=" + ",".join(result.disabled_layers))
        if result.duplicate_target_notes:
            summary_parts.append("duplicate_notes=" + json.dumps(result.duplicate_target_notes, sort_keys=True))

        if summary_parts:
            self._append_log("Drums summary: " + " | ".join(summary_parts))
        for warning in result.warnings:
            self._append_log("Warning: " + str(warning))

    def _collect_request(self) -> HermesActionRequest:
        wav_text = self._wav_var.get().strip()
        midi_text = self._midi_var.get().strip()

        wav_path = Path(wav_text) if wav_text else None
        midi_path = Path(midi_text) if midi_text else None

        drums_request: HermesDrumsRequest | None = None
        if self._selected_role() == ROLE_DRUMS and self._selected_action() == ACTION_MAKE_MIDI_FROM_WAV:
            try:
                c1_note = int(self._c1_midi_note_var.get().strip() or "36")
            except ValueError:
                c1_note = 36

            mapping_file_value = self._mapping_file_var.get().strip()
            drums_request = HermesDrumsRequest(
                output_dir=(Path(self._output_dir_var.get().strip()) if self._output_dir_var.get().strip() else None),
                output_layout=self._output_layout_var.get().strip() or "separate-files",
                profile=self._profile_var.get().strip() or "conservative",
                detection_mode=self._detection_mode_var.get().strip() or "multi-detector",
                mapping_file=(Path(mapping_file_value) if mapping_file_value else None),
                mapping_payload=self._drums_mapping_payload,
                write_empty_layers=bool(self._write_empty_layers_var.get()),
                clean_output_folder=bool(self._clean_output_folder_var.get()),
                c1_midi_note=c1_note,
                target_map="ujam-candy",
            )

        return HermesActionRequest(
            role=self._selected_role(),
            action=self._selected_action(),
            wav_file=wav_path,
            midi_file=midi_path,
            bpm_text=self._bpm_var.get(),
            drums=drums_request,
        )

    def _run_action(self) -> None:
        if self._running:
            return

        request = self._collect_request()
        self._status_var.set("Running...")
        self._run_button.configure(state=tk.DISABLED)
        self._running = True
        self._output_var.set("")
        self._report_var.set("")
        self._debug_csv_var.set("")
        self._created_files_count_var.set("0")
        self._created_files_text.delete("1.0", tk.END)

        self._append_log("Starting Hermes workflow.")

        def _on_worker_done(result) -> None:
            if result.success:
                self._status_var.set(result.message)
                if self._selected_role() == ROLE_DRUMS and self._selected_action() == ACTION_MAKE_MIDI_FROM_WAV:
                    self._render_drums_result(result)
                else:
                    self._output_var.set(str(result.output_file) if result.output_file else "")
                    self._report_var.set(str(result.report_file) if result.report_file else "")
                self._log_queue.put(f"Done: {result.message}")
            else:
                self._status_var.set(result.message)
                self._log_queue.put(f"Error: {result.message}")

            self._running = False
            self._run_button.configure(state=tk.NORMAL)

        def _worker() -> None:
            result = self._controller.execute(request=request, log=self._log)
            self._root.after(0, lambda: _on_worker_done(result))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _open_output_folder(self) -> None:
        selected_dir = self._output_dir_var.get().strip()
        desktop = Path(selected_dir) if selected_dir else self._controller.desktop_dir
        desktop.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(desktop))  # type: ignore[attr-defined]
        except Exception:
            self._append_log(f"Output folder: {desktop}")

    def _open_report_json(self) -> None:
        report = self._report_var.get().strip()
        if not report:
            self._append_log("Report JSON path is empty.")
            return
        report_path = Path(report)
        if not report_path.exists():
            self._append_log(f"Report JSON not found: {report_path}")
            return
        try:
            os.startfile(str(report_path))  # type: ignore[attr-defined]
        except Exception:
            self._append_log(f"Report JSON: {report_path}")

    def _open_debug_csv(self) -> None:
        debug_path_text = self._debug_csv_var.get().strip()
        if not debug_path_text:
            self._append_log("Debug CSV path is empty.")
            return
        debug_path = Path(debug_path_text)
        if not debug_path.exists():
            self._append_log(f"Debug CSV not found: {debug_path}")
            return
        try:
            os.startfile(str(debug_path))  # type: ignore[attr-defined]
        except Exception:
            self._append_log(f"Debug CSV: {debug_path}")

    def run(self) -> None:
        self._root.mainloop()

    def close(self) -> None:
        if self._log_poll_after_id is not None:
            try:
                self._root.after_cancel(self._log_poll_after_id)
            except Exception:
                pass
            self._log_poll_after_id = None
        self._root.destroy()


def launch_hermes_gui(controller: HermesGuiController | None = None) -> None:
    panel = HermesGuiPanel(controller if controller is not None else HermesGuiController())
    panel.run()
