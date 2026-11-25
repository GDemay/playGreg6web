#pragma once

// Shader pack index (G47): which shader identities a WGSL pack (.wgsl.pack, "WGPK", read on the
// WebGPU side by web/workers/wgsl_pack_loader.js) can resolve. The native layer never compiles
// WGSL: it only needs the keys, to decide which pipeline stages are resolved and to emit
// CreateShaderModule packets whose code is the hash-only string "0x%016x", which the WebGPU worker
// resolves from its loaded pack.
//
// WGPK layout (little endian): header {char magic[4] "WGPK", u32 version (1), u32 entry_count,
// u32 flags}, then entry_count entries {u64 hash, u8 stage (1 vertex, 2 fragment, 3 compute),
// u8 reserved[3], u32 offset, u32 length} (20 bytes each); payloads are UTF-8 WGSL.

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace playgta6::gpu {

inline constexpr size_t kShaderPackHeaderBytes = 16;
inline constexpr size_t kShaderPackEntryBytes = 20;

enum class PackStage : uint8_t { Vertex = 1, Fragment = 2, Compute = 3 };

// Pack key -> stage. A key present for several stages keeps the first.
using ShaderPackIndex = std::unordered_map<uint64_t, PackStage>;

// Parses and validates the 16-byte WGPK header ("WGPK" magic, version == 1) and writes the entry
// count to `out_count`. False + message when malformed.
bool parse_shader_pack_header(const uint8_t* data,
                              size_t size,
                              uint32_t* out_count,
                              std::string* error);

// Parses the header and entry table from a TOC-only (or full-file) buffer without reading WGSL
// payloads. Validates magic, version 1, `toc_size >= 16 + count * 20`, entry stages in 1..3,
// non-overflowing 32-bit `offset + length`, and `offset >= 16 + count * 20` (unless `length == 0`).
// When `total_file_size > 0`, also validates `uint64_t{offset} + length <= total_file_size`.
bool parse_shader_pack_index_toc(const uint8_t* toc_data,
                                 size_t toc_size,
                                 uint64_t total_file_size,
                                 ShaderPackIndex* out,
                                 std::string* error);

// Parses the header and entry table of a complete in-memory pack (payloads are not read).
// Equivalent to `parse_shader_pack_index_toc(data, size, size, out, error)`.
bool parse_shader_pack_index(const uint8_t* data, size_t size, ShaderPackIndex* out, std::string* error);

// The hash-only CreateShaderModule code for a pack key ("0x" + 16 lowercase hex digits).
std::string pack_key_code(uint64_t key);

}  // namespace playgta6::gpu
