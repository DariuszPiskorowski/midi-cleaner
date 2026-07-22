# Hermes Windows Launchers

This folder contains Windows launch scripts for Hermes.

## Start Hermes main app

Double-click this file:

scripts\start-hermes.cmd

This launches the full Hermes GUI app.

## Create Desktop Shortcut

Option 1:

Right-click scripts\install-desktop-shortcut.ps1 and choose Run with PowerShell.

Option 2:

powershell -ExecutionPolicy Bypass -File scripts\install-desktop-shortcut.ps1

The script creates or updates a desktop shortcut named:

Hermes

The shortcut targets the full-app launcher:

scripts\start-hermes.cmd

## Optional advanced helper: direct MIDI Editor launch

If you specifically want to start only the browser MIDI Editor directly, use:

scripts\start-hermes-midi-editor.cmd

For normal workflow, start Hermes first and open MIDI Editor from the Hermes button.