# GPU Command Stream & Shader Index (`runtime/gpu` ↔ `web/workers/wgpu_worker.js`)

## Command Packet Layout
Little-endian binary packets matching `playgta6::GpuPacketHeader` (`runtime/abi/playgta6_abi.h`, 8 bytes) followed by `payload_bytes`:

| Offset | Type | Field |
| :--- | :--- | :--- |
| `0..2` | `u16` | `opcode` (`GpuOpcode`) |
| `2..4` | `u16` | `flags` |
| `4..8` | `u32` | `payload_bytes` |
| `8..8+N` | `[u8; N]` | Opcode payload |

### Opcodes (v1)
All multi-byte integers and IEEE-754 floats are **little-endian**. Every struct below matches its exact `static_assert` size in `runtime/abi/playgta6_abi.h`.

| Opcode | Name | Payload Size | Payload Layout |
| :--- | :--- | :--- | :--- |
| `0` | `Nop` | `0` | Empty |
| `1` | `BeginFrame` | `0` | Empty — resets per-frame clear color and queued render-pass commands |
| `2` | `ClearColor` | `16` | `{ f32 r, f32 g, f32 b, f32 a }` (`GpuClearColorPayload`) |
| `3` | `CreateBuffer` | `16` | `{ u32 id, u32 usage_flags, u64 byte_size }` (`GpuCreateBufferPayload`) |
| `4` | `WriteBuffer` | `16 + data_len` | `{ u32 id, u32 dst_offset, u32 data_len, u32 reserved }` (`GpuWriteBufferPayload`) followed by `data_len` raw bytes |
| `5` | `CreateShaderModule` | `16 + code_len` | `{ u32 id, u32 stage, u32 code_len, u32 reserved }` (`GpuCreateShaderModulePayload`) followed by `code_len` UTF-8 WGSL bytes |
| `6` | `Draw` | `16` | `{ u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance }` (`GpuDrawPayload`) |
| `7` | `Present` | `0` | Empty — encodes and submits the current frame's WebGPU passes (`ClearColor` + render/compute passes) |
| `8` | `CreateRenderPipeline` | `24` | `{ u32 id, u32 vs_shader_id, u32 fs_shader_id, u32 vertex_stride, u32 topology, u32 flags }` (`GpuCreateRenderPipelinePayload`) |
| `9` | `SetPipeline` | `4` | `{ u32 pipeline_id }` (`GpuSetPipelinePayload`) |
| `10` | `SetVertexBuffer` | `16` | `{ u32 slot, u32 buffer_id, u64 offset }` (`GpuSetVertexBufferPayload`) |
| `11` | `DestroyResource` | `4` | `{ u32 id }` (`GpuDestroyResourcePayload`) — destroys buffer/texture (if present) and removes `id` from all resource tables |
| `12` | `CreateTexture` | `32` | `{ u32 id, u32 width, u32 height, u32 depth_or_layers, u32 mip_levels, u32 format, u32 usage_flags, u32 dimension }` (`GpuCreateTexturePayload`) |
| `13` | `WriteTexture` | `48 + data_len` | `{ u32 texture_id, u32 mip_level, u32 origin_x, u32 origin_y, u32 origin_z, u32 width, u32 height, u32 depth_or_layers, u32 bytes_per_row, u32 rows_per_image, u32 data_len, u32 reserved }` (`GpuWriteTexturePayload`) followed by `data_len` raw texel bytes |
| `14` | `CreateSampler` | `32` | `{ u32 id, u32 address_u, u32 address_v, u32 address_w, u32 mag_filter, u32 min_filter, u32 mipmap_filter, u32 max_anisotropy }` (`GpuCreateSamplerPayload`) |
| `15` | `CreateBindGroup` | `16 + 32 * entry_count` | `{ u32 id, u32 pipeline_id, u32 group_index, u32 entry_count }` (`GpuCreateBindGroupPayload`) followed by `entry_count` packed `GpuBindGroupEntry` structs (`{ u32 binding, u32 resource_type, u32 resource_id, u32 reserved, u64 offset, u64 size }`, 32 bytes each; for `Buffer` entries, `size == 0` binds from `offset` to buffer end) |
| `16` | `SetBindGroup` | `8` | `{ u32 group_index, u32 bind_group_id }` (`GpuSetBindGroupPayload`) — binds to the currently active render or compute pass |
| `17` | `SetIndexBuffer` | `16` | `{ u32 buffer_id, u32 index_format, u64 offset }` (`GpuSetIndexBufferPayload`) |
| `18` | `DrawIndexed` | `20` | `{ u32 index_count, u32 instance_count, u32 first_index, i32 base_vertex, u32 first_instance }` (`GpuDrawIndexedPayload`) |
| `19` | `CreateComputePipeline` | `16` | `{ u32 id, u32 cs_shader_id, u32 flags, u32 reserved }` (`GpuCreateComputePipelinePayload`) |
| `20` | `SetComputePipeline` | `4` | `{ u32 pipeline_id }` (`GpuSetComputePipelinePayload`) — ends any open render pass and begins a compute pass |
| `21` | `DispatchCompute` | `12` | `{ u32 workgroup_x, u32 workgroup_y, u32 workgroup_z }` (`GpuDispatchComputePayload`) |

### Enum Values

#### `usage_flags` (`GpuBufferUsage`, matches WebGPU `GPUBufferUsage`)
- `0x0001`: `MAP_READ`
- `0x0002`: `MAP_WRITE`
- `0x0004`: `COPY_SRC`
- `0x0008`: `COPY_DST`
- `0x0010`: `INDEX`
- `0x0020`: `VERTEX`
- `0x0040`: `UNIFORM`
- `0x0080`: `STORAGE`
- `0x0100`: `INDIRECT`
- `0x0200`: `QUERY_RESOLVE`

#### `usage_flags` (`GpuTextureUsage`, matches WebGPU `GPUTextureUsage`)
- `0x0001`: `COPY_SRC`
- `0x0002`: `COPY_DST`
- `0x0004`: `TEXTURE_BINDING`
- `0x0008`: `STORAGE_BINDING`
- `0x0010`: `RENDER_ATTACHMENT`

#### `format` (`GpuTextureFormat`)
- `0`: `Rgba8Unorm` (`"rgba8unorm"`)
- `1`: `Bgra8Unorm` (`"bgra8unorm"`)
- `2`: `Rgba16Float` (`"rgba16float"`)
- `3`: `Depth24Plus` (`"depth24plus"`)
- `4`: `Depth32Float` (`"depth32float"`)
- `5`: `Bc7RgbaUnorm` (`"bc7-rgba-unorm"`)

#### `dimension` (`GpuTextureDimension`)
- `0`: `Dim1D` (`"1d"`)
- `1`: `Dim2D` (`"2d"`)
- `2`: `Dim3D` (`"3d"`)

#### `address_u` / `address_v` / `address_w` (`GpuAddressMode`)
- `0`: `ClampToEdge` (`"clamp-to-edge"`)
- `1`: `Repeat` (`"repeat"`)
- `2`: `MirrorRepeat` (`"mirror-repeat"`)

#### `mag_filter` / `min_filter` / `mipmap_filter` (`GpuFilterMode`)
- `0`: `Nearest` (`"nearest"`)
- `1`: `Linear` (`"linear"`)

#### `resource_type` (`GpuBindResourceType`)
- `0`: `Buffer` (`GPUBufferBinding { buffer, offset, size? }`)
- `1`: `TextureView` (`GPUTextureView` created from texture `resource_id`)
- `2`: `Sampler` (`GPUSampler`)

#### `index_format` (`GpuIndexFormat`)
- `0`: `Uint16` (`"uint16"`)
- `1`: `Uint32` (`"uint32"`)

#### `stage` (`GpuShaderStage`)
- `0`: `Vertex` (`vs_main`)
- `1`: `Fragment` (`fs_main`)
- `2`: `Compute` (`cs_main`)

#### `topology` (`GpuPrimitiveTopology`)
- `0`: `TriangleList` (`"triangle-list"`)
- `1`: `TriangleStrip` (`"triangle-strip"`)
- `2`: `LineList` (`"line-list"`)
- `3`: `LineStrip` (`"line-strip"`)
- `4`: `PointList` (`"point-list"`)

### Pass Switching Semantics
Within a frame (between `BeginFrame` and `Present`), the WebGPU worker lazily opens render and compute passes on a single `GPUCommandEncoder`:
- Any compute command (`SetComputePipeline`, `DispatchCompute`) ends an open render pass before calling `beginComputePass()`.
- Any render command (`SetPipeline`, `SetVertexBuffer`, `SetIndexBuffer`, `Draw`, `DrawIndexed`) ends an open compute pass before calling `beginRenderPass()`. The first render pass in a frame uses `loadOp: "clear"` with the active `ClearColor`; any resumed render pass in the same frame uses `loadOp: "load"` to preserve prior drawing.
- `Present` ends any open pass (opening a clearing render pass if no render pass ran yet) and submits the finished `GPUCommandBuffer`.

## `shader-index.json` (Generated by `shaders/`)
Maps Xbox shader hashes to lowered WGSL modules:

```json
{
  "abi_version": 1,
  "shaders": {
    "0xdeadbeef": {
      "stage": "compute",
      "wgsl": "shaders/deadbeef.wgsl"
    }
  }
}
```
