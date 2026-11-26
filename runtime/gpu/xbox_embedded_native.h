#pragma once

// Console-native code embedded in a DXIL container (G66, #663 / #691).
//
// The console compiler stores the GFX10 machine code of a shader inside the container's `DXIL`
// part, next to the bitcode. A pipeline created from DXIL (slots 86, 10, 11, 62) therefore
// carries its native stages even when the title never creates a slot-57 native pipeline for it
// (the legal / privacy screen text and its UI quad: their digests are in no slot-57 metadata).
// The data is zlib-packed, not encrypted. Layout as observed in the title's containers (offsets
// from the start of the `DXIL` part data):
//
//   +0x18 "XDAT"  u32 entry count, u32, then per entry: u32 kind, u32 offset, u32 size
//   kind 0x0d     "XBCH" u32 version (3), u64 packed size, u64 unpacked size, zlib stream
//   unpacked      "FSSM" u32 version (0x101), u64 payload size, payload
//   payload +0x00 u64 graphics slot count (6); +0x08.. u64 record offsets LS HS ES GS VS PS
//                 (0x08..0x30), mesh 0x48, compute 0x60; 0xDEADBEEF_DEADBEEF when absent
//   record  +0x00 u64 code offset, +0x08 u64 code size (payload-relative); 0x54 bytes, then the
//                 stage data. Pixel stage data (E = record + 0x54): E+0x148 u32 export patch
//                 size (dwords), E+0x14c 8 patches (one per MRT) of u32 code offset (dwords, -1
//                 none) and 10 variants of 10 dwords (variant 9: every channel as raw 32 bits).
//
// A pixel program's colour exports are left for pipeline creation to patch for the render
// target formats; the extracted code has the raw 32-bit variant patched in (the values the
// shader computes, unconverted), which is what a slot-57 stage hands the translator.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace playgta6::gpu::d3d12_guest::xbox {

enum class EmbeddedStage : uint8_t { Ls, Hs, Es, Gs, Vs, Ps, Ms, Cs };

struct EmbeddedProgram {
    EmbeddedStage stage = EmbeddedStage::Vs;
    std::vector<uint8_t> code;  // GFX10 machine code (pixel: export patches applied)
};

struct EmbeddedNative {
    std::vector<uint8_t> fssm;  // the unpacked blob of the first native entry (dump metadata)
    std::vector<EmbeddedProgram> programs;
};

// Extracts the native programs of a DXBC container. True with no programs when the container
// has no XDAT table or no native entry; false (error set) when one is malformed.
bool extract_embedded_native(const uint8_t* container, size_t size, EmbeddedNative* out, std::string* error);

// The first program of `stage`, or null.
const EmbeddedProgram* find_embedded(const EmbeddedNative& native, EmbeddedStage stage);

}  // namespace playgta6::gpu::d3d12_guest::xbox
