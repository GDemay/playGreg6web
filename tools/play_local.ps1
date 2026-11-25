<#
.SYNOPSIS
  Build (if stale) and watch the browser build in a headed Chrome, locally.
  Thin wrapper around tools/run_browser_boot.py (local COOP/COEP server + Chrome with its own profile).
.EXAMPLE
  .\tools\play_local.ps1 -Exe D:\game\game.exe -GameDir D:\game\package -InputScript calibration-skip
.NOTES
  Stop: close the Chrome window or press Ctrl+C here; the server stops with this script.
  See docs/PLAY_LOCALLY.md. Game files stay outside the repo.
#>
param(
    [string]$Exe = $env:PLAYGTA6_EXE,
    [string]$GameDir = $env:GTA6_GAME_DIR,
    [int]$Port = 8141,
    [string]$BuildDir = "work/browser-boot",
    [string]$Chrome = $env:CHROME_PATH,
    [ValidateSet("", "start-screen", "menu-flow", "calibration-skip", "intro-watch", "legal-notices", "chrome-menu-flow")]
    [string]$InputScript = "",
    [int]$DurationS = 7200,
    [switch]$NoBuild,
    [switch]$TrustBuild,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if (-not $Exe -or -not $GameDir) { throw "Pass -Exe and -GameDir (or set PLAYGTA6_EXE / GTA6_GAME_DIR)." }
$argv = @("tools/run_browser_boot.py", "--preset", "long", "--name", "play-local", "--headed",
          "--exe", $Exe, "--game-dir", $GameDir, "--port", $Port, "--build-dir", $BuildDir,
          "--duration-s", $DurationS, "--report-to-log")
if ($Chrome) { $argv += @("--chrome", $Chrome) }
if ($InputScript) { $argv += @("--input-script", $InputScript) }
if ($NoBuild) { $argv += "--no-build" }
if ($TrustBuild) { $argv += "--trust-build" }
if ($DryRun) { $argv += "--dry-run" }
Write-Host "Serving on http://127.0.0.1:$Port/ ; stop with Ctrl+C or by closing Chrome (run lasts up to $DurationS s)."
Push-Location $repo
try { & python @argv; exit $LASTEXITCODE } finally { Pop-Location }
