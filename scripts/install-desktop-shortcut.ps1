[CmdletBinding()]
param(
    [string]$ShortcutName = "Hermes"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir ".." )).Path
$launcherPath = Join-Path $scriptDir "start-hermes.cmd"

if (-not (Test-Path -LiteralPath $launcherPath -PathType Leaf)) {
    Write-Host "Launcher script not found: $launcherPath" -ForegroundColor Red
    exit 1
}

$desktopPath = [Environment]::GetFolderPath("Desktop")
if ([string]::IsNullOrWhiteSpace($desktopPath)) {
    Write-Host "Could not resolve the Desktop folder." -ForegroundColor Red
    exit 1
}

$shortcutPath = Join-Path $desktopPath ($ShortcutName + ".lnk")

$wshShell = New-Object -ComObject WScript.Shell
$shortcut = $wshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcherPath
$shortcut.WorkingDirectory = $repoRoot
$shortcut.Description = "Launch Hermes main app"
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,220"
$shortcut.Save()

Write-Host "Desktop shortcut created or updated:" -ForegroundColor Green
Write-Host $shortcutPath

exit 0