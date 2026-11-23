[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string]$BuildType = 'Release',
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$distDir = Join-Path $repoRoot "web\dist"
$buildDir = Join-Path $PSScriptRoot "build-wasm64"
$cmakeLists = Join-Path $PSScriptRoot "CMakeLists.txt"

$emcmake = Get-Command emcmake -ErrorAction SilentlyContinue
$emcc = Get-Command emcc -ErrorAction SilentlyContinue

if (-not $emcmake -or -not $emcc) {
    $missing = @()
    if (-not $emcmake) { $missing += 'emcmake' }
    if (-not $emcc) { $missing += 'emcc' }
    Write-Host "Emscripten tool(s) not found on PATH: $($missing -join ', ')" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install and activate emsdk first:"
    Write-Host "  1. git clone https://github.com/emscripten-core/emsdk.git"
    Write-Host "  2. cd emsdk"
    Write-Host "  3. .\emsdk.ps1 install latest"
    Write-Host "  4. .\emsdk.ps1 activate latest"
    Write-Host "  5. .\emsdk_env.ps1"
    Write-Host ""
    Write-Host "Then re-run: .\runtime\build_wasm.ps1"
    exit 1
}

$emccVersion = (& emcc --version | Select-Object -First 1)
Write-Host "Found emcmake: $($emcmake.Source)"
Write-Host "Found emcc:    $($emcc.Source) ($emccVersion)"

$cmakeContent = Get-Content -Path $cmakeLists -Raw
$requiredFlags = @('-sMEMORY64=1', '-pthread')
foreach ($flag in $requiredFlags) {
    if ($cmakeContent -notmatch [regex]::Escape($flag)) {
        Write-Error "runtime/CMakeLists.txt is missing required Wasm64 build flag: $flag"
    }
}
Write-Host "Verified Wasm64 build flags in runtime/CMakeLists.txt: $($requiredFlags -join ', ')"

if (-not (Test-Path -Path $distDir)) {
    New-Item -ItemType Directory -Path $distDir -Force | Out-Null
    Write-Host "Created output directory: $distDir"
} else {
    Write-Host "Output directory ready: $distDir"
}

if ($CheckOnly) {
    Write-Host "Emscripten Wasm64 toolchain check passed (-CheckOnly)."
    exit 0
}

if ($Clean -and (Test-Path -Path $buildDir)) {
    Remove-Item -Path $buildDir -Recurse -Force
}

$wasmFlags = "-sMEMORY64=1 -pthread"
& emcmake cmake -S $PSScriptRoot -B $buildDir `
    "-DCMAKE_BUILD_TYPE=$BuildType" `
    "-DCMAKE_CXX_FLAGS=$wasmFlags"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& cmake --build $buildDir --parallel
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$expectedOutputs = @(
    (Join-Path $distDir "runtime.js"),
    (Join-Path $distDir "runtime.wasm")
)
foreach ($artifact in $expectedOutputs) {
    if (-not (Test-Path -Path $artifact)) {
        Write-Error "Expected Wasm64 build artifact not found: $artifact"
    }
}

Write-Host "Built Wasm64 runtime into web/dist/runtime.js + web/dist/runtime.wasm"
