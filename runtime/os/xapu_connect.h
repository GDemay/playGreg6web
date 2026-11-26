// XApuConnect parameter struct (xapu.dll guest model, acp_hal_guest.cpp).
//
// Six little-endian u32 words, verified against the title's own call (PG6_AUDIO_TRACE=1 raw dump,
// menu run): {1, 256, 2, 0x104000, 0, 1} = process kind 1 (Opus decode), 256 streams, 2 commands
// per stream, 0x104000 bytes of shared memory (= 128 * 0x280 + 128 * 0x1E00, exactly the voice
// slot layout the title indexes), 0 frames per output, options 1. Option bit 3 (value 8) asks for
// an auto-reset result-ready event, written to the second qword of the output struct; the title
// does not set it.

#pragma once

#include <cstddef>
#include <cstdint>

namespace playgta6::os::xapu_connect {

constexpr uint32_t kProcessDecodeOpus = 1;
constexpr uint32_t kOptionResultEvent = 8;
constexpr uint32_t kMaxStreams = 4096;
constexpr uint32_t kMaxCommandsPerStream = 4096;
constexpr uint32_t kMaxMemoryBytes = 64u << 20;
constexpr uint64_t kVoiceSlots = 0x80;
constexpr uint64_t kFixedLayoutBytes = kVoiceSlots * 0x280 + kVoiceSlots * 0x1E00;

struct Params {
    uint64_t memory_bytes = kFixedLayoutBytes;
    size_t outstanding_limit = 0;  // streams * commands per stream; 0 = unlimited
    bool result_event = false;
};

// `w` are the first six words of the struct. Implausible values fall back to the fixed layout; the
// voice slots sit at fixed offsets, so memory never goes below it.
inline Params parse(const uint32_t (&w)[6]) {
    Params r;
    if (w[1] != 0 && w[1] <= kMaxStreams && w[2] != 0 && w[2] <= kMaxCommandsPerStream) {
        r.outstanding_limit = size_t{w[1]} * w[2];
    }
    if (w[3] > kFixedLayoutBytes && w[3] <= kMaxMemoryBytes) {
        r.memory_bytes = w[3];
    }
    r.result_event = (w[5] & kOptionResultEvent) != 0;
    return r;
}

}  // namespace playgta6::os::xapu_connect
