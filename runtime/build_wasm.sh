#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${REPO_ROOT}/web/dist"
CMAKE_LISTS="${SCRIPT_DIR}/CMakeLists.txt"

CHECK_ONLY=0
CLEAN=0
BUILD_TYPE="Release"
BUILD_DIR="${SCRIPT_DIR}/build-wasm64"

usage() {
  cat <<'EOF'
Usage: ./runtime/build_wasm.sh [options]

Build the playgta6web C++20 runtime for WebAssembly memory64 (Wasm64) with Emscripten,
emitting web/dist/runtime.js and web/dist/runtime.wasm.

Options:
  --check-only                   Verify emsdk toolchain, CMake Wasm64 flags, and output dir without building
  --build-type <TYPE>            CMake build type: Release (default), Debug, or RelWithDebInfo
  --build-dir <PATH>             CMake build directory (default: runtime/build-wasm64)
  --clean                        Remove existing build directory before configuring
  -h, --help                     Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --build-type)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Error: --build-type requires a value (Release, Debug, or RelWithDebInfo)." >&2
        exit 2
      fi
      BUILD_TYPE="$2"
      shift 2
      ;;
    --build-type=*)
      BUILD_TYPE="${1#*=}"
      if [[ -z "${BUILD_TYPE}" ]]; then
        echo "Error: --build-type requires a value (Release, Debug, or RelWithDebInfo)." >&2
        exit 2
      fi
      shift
      ;;
    --build-dir)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Error: --build-dir requires a path." >&2
        exit 2
      fi
      BUILD_DIR="$2"
      shift 2
      ;;
    --build-dir=*)
      BUILD_DIR="${1#*=}"
      if [[ -z "${BUILD_DIR}" ]]; then
        echo "Error: --build-dir requires a path." >&2
        exit 2
      fi
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      echo "Run './runtime/build_wasm.sh --help' for usage." >&2
      exit 2
      ;;
  esac
done

case "${BUILD_TYPE}" in
  Release|Debug|RelWithDebInfo) ;;
  *)
    echo "Error: invalid --build-type '${BUILD_TYPE}'. Expected Release, Debug, or RelWithDebInfo." >&2
    exit 2
    ;;
esac

EMCMAKE_PATH="$(command -v emcmake 2>/dev/null || true)"
EMCC_PATH="$(command -v emcc 2>/dev/null || true)"

if [[ -z "${EMCMAKE_PATH}" || -z "${EMCC_PATH}" ]]; then
  MISSING=()
  [[ -z "${EMCMAKE_PATH}" ]] && MISSING+=("emcmake")
  [[ -z "${EMCC_PATH}" ]] && MISSING+=("emcc")
  MISSING_LIST="$(IFS=", "; echo "${MISSING[*]}")"

  echo "Emscripten tool(s) not found on PATH: ${MISSING_LIST}" >&2
  echo "" >&2
  echo "Install and activate emsdk first:" >&2
  echo "  1. git clone https://github.com/emscripten-core/emsdk.git" >&2
  echo "  2. cd emsdk" >&2
  echo "  3. ./emsdk install latest" >&2
  echo "  4. ./emsdk activate latest" >&2
  echo "  5. source ./emsdk_env.sh" >&2
  echo "" >&2
  echo "Then re-run: ./runtime/build_wasm.sh" >&2
  exit 1
fi

EMCC_VERSION_RAW="$(emcc --version 2>&1 || true)"
EMCC_VERSION="${EMCC_VERSION_RAW%%$'\n'*}"
EMCC_VERSION="${EMCC_VERSION%$'\r'}"
echo "Found emcmake: ${EMCMAKE_PATH}"
echo "Found emcc:    ${EMCC_PATH} (${EMCC_VERSION})"

if [[ ! -f "${CMAKE_LISTS}" ]]; then
  echo "Error: runtime/CMakeLists.txt not found at ${CMAKE_LISTS}" >&2
  exit 1
fi

REQUIRED_FLAGS=("-sMEMORY64=1" "-pthread")
for flag in "${REQUIRED_FLAGS[@]}"; do
  if ! grep -Fq -- "${flag}" "${CMAKE_LISTS}"; then
    echo "Error: runtime/CMakeLists.txt is missing required Wasm64 build flag: ${flag}" >&2
    exit 1
  fi
done
echo "Verified Wasm64 build flags in runtime/CMakeLists.txt: -sMEMORY64=1, -pthread"

if [[ ! -d "${DIST_DIR}" ]]; then
  mkdir -p "${DIST_DIR}"
  echo "Created output directory: ${DIST_DIR}"
else
  echo "Output directory ready: ${DIST_DIR}"
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "Emscripten Wasm64 toolchain check passed (--check-only)."
  exit 0
fi

if [[ "${CLEAN}" -eq 1 && -d "${BUILD_DIR}" ]]; then
  rm -rf "${BUILD_DIR}"
fi

WASM_FLAGS="-sMEMORY64=1 -pthread"
emcmake cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" \
  "-DCMAKE_CXX_FLAGS=${WASM_FLAGS}"

cmake --build "${BUILD_DIR}" --parallel

JS_ARTIFACT="${DIST_DIR}/runtime.js"
WASM_ARTIFACT="${DIST_DIR}/runtime.wasm"
for artifact in "${JS_ARTIFACT}" "${WASM_ARTIFACT}"; do
  if [[ ! -f "${artifact}" ]]; then
    echo "Error: Expected Wasm64 build artifact not found: ${artifact}" >&2
    exit 1
  fi
done

WASM_MAGIC="$(LC_ALL=C od -An -tx1 -N4 "${WASM_ARTIFACT}" | tr -d '[:space:]')"
if [[ "${WASM_MAGIC}" != "0061736d" ]]; then
  echo "Error: ${WASM_ARTIFACT} does not start with valid WebAssembly magic bytes (\\0asm); got '${WASM_MAGIC}'" >&2
  exit 1
fi

echo "Built Wasm64 runtime into web/dist/runtime.js + web/dist/runtime.wasm"
