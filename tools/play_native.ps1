<#
.SYNOPSIS
  Build (if stale, with the Dawn native executor on) and play natively: a live window, audio out, keyboard/pad.
  Thin wrapper around tools/run_native_boot.py --play (uses the object/lift/recomp caches).
.EXAMPLE
  .\tools\play_native.ps1 -MaxMs 120000
.NOTES
  Needs the shader pack (-ShaderPack or env PLAYGTA6_NATIVE_SHADER_PACK). Stop: close the window or Ctrl+C;
  without -MaxMs the run lasts 2 h. See docs/PLAY_LOCALLY.md. Game files stay outside the repo.
#>
param(
    [string]$Exe = $env:PLAYGTA6_EXE,
    [string]$GameDir = $env:GTA6_GAME_DIR,
    [string]$ShaderPack = $env:PLAYGTA6_NATIVE_SHADER_PACK,
    [string]$SaveDir = "",
    [ValidateSet("none", "start-screen", "menu-flow", "menu-flow-2", "menu-flow-3", "menu-flow-4", "menu-to-ingame")]
    [string]$InputScript = "none",
    [string]$BuildDir = "work/native-boot",
    [string]$Name = "",
    [int]$MaxMs = 7200000,
    [switch]$NoBuild,
    [switch]$TrustBuild,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if (-not $Exe -or -not $GameDir) { throw "Pass -Exe and -GameDir (or set PLAYGTA6_EXE / GTA6_GAME_DIR)." }
if (-not $ShaderPack -or -not (Test-Path -LiteralPath $ShaderPack)) {
    throw "No shader pack: pass -ShaderPack <std.wgsl.pack> or set PLAYGTA6_NATIVE_SHADER_PACK (without one nothing is drawn)."
}
if (-not $Name) { $Name = "play-native-" + (Get-Date -Format "yyyyMMdd-HHmmss") }
$preset = if ($InputScript -eq "none") { "long" } else { $InputScript }
$argv = @("tools/run_native_boot.py", "--preset", $preset, "--name", $Name, "--play",
          "--exe", $Exe, "--game-dir", $GameDir, "--build-dir", $BuildDir,
          "--max-ms", $MaxMs, "--watchdog-ms", ($MaxMs + 30000), "--shader-pack", $ShaderPack)
if ($NoBuild) { $argv += "--no-build" }
if ($TrustBuild) { $argv += "--trust-build" }
if ($DryRun) { $argv += "--dry-run" }
if ($SaveDir) { $argv += @("--", "--save-dir", $SaveDir) }
Push-Location $repo
try { & python @argv; exit $LASTEXITCODE } finally { Pop-Location }
