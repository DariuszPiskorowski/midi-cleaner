[CmdletBinding()]
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir ".." )).Path
Set-Location $repoRoot

$baseUrl = "http://$BindHost`:$Port/"
$sessionUrl = "$baseUrl" + "api/session"

function Test-HermesServer {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    try {
        $response = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 2
        if ($null -eq $response) {
            return $false
        }
        $propertyNames = $response.PSObject.Properties.Name
        return ($propertyNames -contains "tracks") -and ($propertyNames -contains "notes")
    }
    catch {
        return $false
    }
}

function Test-PortListening {
    param(
        [Parameter(Mandatory = $true)]
        [int]$PortNumber
    )

    try {
        $listener = Get-NetTCPConnection -LocalPort $PortNumber -State Listen -ErrorAction Stop | Select-Object -First 1
        if ($null -ne $listener) {
            return $true
        }
    }
    catch {
        # Fallback for environments where Get-NetTCPConnection is unavailable.
    }

    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $asyncResult = $client.BeginConnect("127.0.0.1", $PortNumber, $null, $null)
        $connected = $asyncResult.AsyncWaitHandle.WaitOne(200)
        if ($connected -and $client.Connected) {
            $client.EndConnect($asyncResult) | Out-Null
            $client.Close()
            return $true
        }
        $client.Close()
        return $false
    }
    catch {
        return $false
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed or not on PATH." -ForegroundColor Red
    exit 1
}

if (Test-HermesServer -Url $sessionUrl) {
    Write-Host "Hermes MIDI Editor is already running at $baseUrl"
    Start-Process $baseUrl | Out-Null
    exit 0
}

if (Test-PortListening -PortNumber $Port) {
    Write-Host "Port $Port is already in use, but Hermes did not respond at $baseUrl" -ForegroundColor Red
    Write-Host "Close the process using port $Port and try again." -ForegroundColor Red
    exit 1
}

Write-Host "Starting Hermes MIDI Editor at $baseUrl"
& uv run midi-cleaner midi split-editor --host $BindHost --port $Port

$exitCode = $LASTEXITCODE
if ($null -eq $exitCode) {
    $exitCode = 0
}

exit $exitCode