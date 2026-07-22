@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

cd /d "%REPO_ROOT%"
if errorlevel 1 (
  echo Failed to switch to repository root: "%REPO_ROOT%"
  pause
  exit /b 1
)

where uv >nul 2>nul
if errorlevel 1 (
  echo uv is not installed or not on PATH.
  echo Install uv and try again.
  pause
  exit /b 1
)

echo Starting Hermes main app...
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start-hermes.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Hermes launcher exited with code %EXIT_CODE%.
  pause
)

exit /b %EXIT_CODE%