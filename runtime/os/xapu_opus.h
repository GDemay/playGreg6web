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

// Maximum frames per Opus packet (120 ms at 48 kHz, RFC 6716).
constexpr size_t kMaxOpusFrames = 48000u * 120u / 1000u;

// Whether the runtime was built with libopus.
bool available();

// RFC 6716 TOC & packet framing metadata.
struct PacketInfo {
    uint8_t config = 0;
    bool stereo = false;
    uint8_t frame_count_code = 0;
    uint32_t frame_samples_48k = 0;
    bool valid = false;
};

// Inspects an Opus packet's RFC 6716 TOC byte and packet length framing without decoding.
// Returns true iff the packet header/framing is valid and populates *out (if non-null).
bool inspect_packet(const uint8_t* packet, size_t bytes, PacketInfo* out);

// Multi-voice float32 mixing & soft-limiting helpers (for combining Opus float32 and XMA2 int16).
void mix_pcm_additive(float* dst, const float* src, size_t samples, float gain);
void mix_int16_to_float_additive(float* dst, const int16_t* src, size_t samples, float gain);
void sanitize_and_soft_clip(float* pcm, size_t samples);

// One voice's decoder state; packets of a stream are decoded in order on the same object.
class VoiceDecoder {
public:
    VoiceDecoder() = default;
    ~VoiceDecoder();
    VoiceDecoder(const VoiceDecoder&) = delete;
    VoiceDecoder& operator=(const VoiceDecoder&) = delete;

    // Reset stream state (reuses existing OpusDecoder via OPUS_RESET_STATE without heap churn).
    void reset();
    // Alias for state reset without deallocation.
    void flush();
    // Release underlying OpusDecoder resources (also invoked by destructor).
    void release();

    // Decodes one packet to interleaved float32 frames, at most `max_frames` frames of `channels`
    // samples into `out`. `sample_rate` is rounded to a rate Opus decodes to (8, 12, 16, 24 or
    // 48 kHz). Returns the frames written, or -1 when the packet or parameters are invalid.
    int decode(const uint8_t* packet, size_t bytes, uint32_t sample_rate, uint32_t channels, float* out,
               size_t max_frames);

    uint64_t decoded_packets() const { return decoded_packets_; }
    uint64_t failed_packets() const { return failed_packets_; }
    uint64_t reset_count() const { return reset_count_; }

private:
    void* decoder_ = nullptr;  // OpusDecoder*
    uint32_t sample_rate_ = 0;
    uint32_t channels_ = 0;
    uint64_t decoded_packets_ = 0;
    uint64_t failed_packets_ = 0;
    uint64_t reset_count_ = 0;
};

}  // namespace playgta6::os::xapu_opus
