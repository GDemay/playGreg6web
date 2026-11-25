// Opus packet decoding for the xapu.dll guest model (acp_hal_guest.cpp).
//
// The title's stream decoder hands the audio processing unit one Opus packet per decode command
// (960 samples per channel, at most 0x280 bytes) and reads back float32 PCM. This wraps libopus
// (BSD-3-Clause, Xiph.Org Foundation; fetched by runtime/CMakeLists.txt) behind a per-voice
// decoder. Without libopus (PLAYGTA6_OPUS=OFF) `available()` is false and decoding yields silence.

#pragma once

#include <cstddef>
#include <cstdint>

namespace playgta6::os::xapu_opus {

// Whether the runtime was built with libopus.
bool available();

// One voice's decoder state; packets of a stream are decoded in order on the same object.
class VoiceDecoder {
public:
    VoiceDecoder() = default;
    ~VoiceDecoder();
    VoiceDecoder(const VoiceDecoder&) = delete;
    VoiceDecoder& operator=(const VoiceDecoder&) = delete;

    // Forget the stream state (a new stream starts on the voice).
    void reset();

    // Decodes one packet to interleaved float32 frames, at most `max_frames` frames of `channels`
    // samples into `out`. `sample_rate` is rounded to a rate Opus decodes to (8, 12, 16, 24 or
    // 48 kHz). Returns the frames written, or -1 when the packet or parameters are invalid.
    int decode(const uint8_t* packet, size_t bytes, uint32_t sample_rate, uint32_t channels, float* out,
               size_t max_frames);

private:
    void* decoder_ = nullptr;  // OpusDecoder*
    uint32_t sample_rate_ = 0;
    uint32_t channels_ = 0;
};

}  // namespace playgta6::os::xapu_opus
