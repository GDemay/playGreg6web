# Wasm64 ↔ JS Host ABI (`runtime/` ↔ `web/loader.js`)

ABI version is defined in `runtime/abi/playgta6_abi.h` (`playgta6::kAbiVersion = 1`).

## Required Browser Capabilities
- `self.crossOriginIsolated === true` (`SharedArrayBuffer` enabled)
- `navigator.gpu` (WebGPU on `OffscreenCanvas`)
- WebAssembly `memory64` (Wasm64, built via Emscripten `-sMEMORY64=1 -pthread`)

## Host Imports (`env`)
Declared in `runtime/os/os_shim.cpp`:

| Function | Signature | Purpose |
| :--- | :--- | :--- |
| `host_log` | `(ptr: i64, len: i64) -> ()` | UTF-8 log line to console / progress channel |
| `host_httpfs_open` | `(path_ptr: i64, path_len: i64) -> i32` | Open relative `/data/` path; returns handle `>= 0` or `-1` |
| `host_httpfs_read` | `(handle: i32, offset: i64, dst_ptr: i64, len: i64) -> i64` | Range read via `io_worker.js` |
| `host_httpfs_close` | `(handle: i32) -> ()` | Close virtual file handle |
| `host_wgpu_submit` | `(cmd_ptr: i64, cmd_len: i64) -> ()` | Forward binary command buffer to `wgpu_worker.js` |

## Shared Input Block (`InputState`, 40 bytes)
Written into a `SharedArrayBuffer` by `web/input/input_bridge.js` (`Int32Array` / `Float32Array`) matching `playgta6::InputState`:
- `0..4`: `seq` (`u32`, incremented via `Atomics.add` after each update)
- `4..8`: `buttons` (`u32` bitmask)
- `8..12`: `mouse_dx` (`i32`)
- `12..16`: `mouse_dy` (`i32`)
- `16..40`: `left_stick_x/y`, `right_stick_x/y`, `left_trigger`, `right_trigger` (`6 × f32`)
