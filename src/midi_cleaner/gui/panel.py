from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from midi_cleaner.gui.controller import (
    ACTION_MAKE_MIDI_FROM_WAV,
    ACTION_SET_BPM,
    ACTION_SYNCHRONIZE_MIDI_WITH_WAV,
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

        self._running = False
        self._log_queue: queue.Queue[str] = queue.Queue()

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
        ttk.Label(frame, text="Log").grid(row=row, column=0, sticky="w", pady=(10, 0))

        row += 1
        self._log_text = ScrolledText(frame, height=16, wrap=tk.WORD)
        self._log_text.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(2, 0))

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row, weight=1)

    def _bind_events(self) -> None:
        self._role_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_field_visibility())
        self._action_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_field_visibility())

    def _selected_role(self) -> str:
        return self._role_by_label[self._role_var.get()]

    def _selected_action(self) -> str:
        return self._action_by_label[self._action_var.get()]

    def _refresh_field_visibility(self) -> None:
        role = self._selected_role()
        action = self._selected_action()

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

    def _append_log(self, message: str) -> None:
        self._log_text.insert(tk.END, message + "\n")
        self._log_text.see(tk.END)

    def _log(self, message: str) -> None:
        self._log_queue.put(message)

    def _schedule_log_poll(self) -> None:
        self._poll_log_queue()
        self._root.after(150, self._schedule_log_poll)

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

    def _collect_request(self) -> HermesActionRequest:
        wav_text = self._wav_var.get().strip()
        midi_text = self._midi_var.get().strip()

        wav_path = Path(wav_text) if wav_text else None
        midi_path = Path(midi_text) if midi_text else None

        return HermesActionRequest(
            role=self._selected_role(),
            action=self._selected_action(),
            wav_file=wav_path,
            midi_file=midi_path,
            bpm_text=self._bpm_var.get(),
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

        self._append_log("Starting Hermes workflow.")

        def _on_worker_done(result) -> None:
            if result.success:
                self._status_var.set(result.message)
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
        desktop = self._controller.desktop_dir
        desktop.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(desktop))  # type: ignore[attr-defined]
        except Exception:
            self._append_log(f"Output folder: {desktop}")

    def run(self) -> None:
        self._root.mainloop()


def launch_hermes_gui(controller: HermesGuiController | None = None) -> None:
    panel = HermesGuiPanel(controller if controller is not None else HermesGuiController())
    panel.run()
