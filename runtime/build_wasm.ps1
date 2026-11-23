$ErrorActionPreference = 'Stop'

if (-not (Get-Command emcmake -ErrorAction SilentlyContinue)) {
    Write-Error "Emscripten (emcmake) not found on PATH. Install/activate emsdk first."
}

$buildDir = Join-Path $PSScriptRoot "build-wasm64"
emcmake cmake -S $PSScriptRoot -B $buildDir -DCMAKE_BUILD_TYPE=Release
cmake --build $buildDir --parallel
Write-Host "Built Wasm64 runtime into web/dist/runtime.js + web/dist/runtime.wasm"
