#include "os/xapu_opus.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(PLAYGTA6_HAVE_OPUS)
#include <opus.h>
#endif

namespace playgta6::os::xapu_opus {

namespace {

constexpr float kDenormalThreshold = 1e-30f;

// Opus decodes to these rates only; anything else is taken as the nearest one at or above it.
uint32_t opus_rate(uint32_t rate) {
    static constexpr uint32_t kRates[] = {8000u, 12000u, 16000u, 24000u};
    for (uint32_t r : kRates) {
        if (rate <= r) {
            return r;
        }
    }
    return 48000u;
}

float sanitize_sample(float v) {
    if (!std::isfinite(v) || std::fabs(v) < kDenormalThreshold) {
        return 0.0f;
    }
    return v;
}

uint32_t config_frame_samples_48k(uint8_t config) {
    if (config < 12u) {
        // SILK-only: 10, 20, 40, 60 ms
        static constexpr uint32_t kSilk[4] = {480u, 960u, 1920u, 2880u};
        return kSilk[config & 3u];
    }
    if (config < 16u) {
        // Hybrid: 10, 20 ms
        return (config & 1u) != 0u ? 960u : 480u;
    }
    // CELT-only (16..31): 2.5, 5, 10, 20 ms
    static constexpr uint32_t kCelt[4] = {120u, 240u, 480u, 960u};
    return kCelt[config & 3u];
}

}  // namespace

bool available() {
#if defined(PLAYGTA6_HAVE_OPUS)
    return true;
#else
    return false;
#endif
}

bool inspect_packet(const uint8_t* packet, size_t bytes, PacketInfo* out) {
    PacketInfo info{};
    if (packet == nullptr || bytes == 0u) {
        if (out != nullptr) {
            *out = info;
        }
        return false;
    }
    const uint8_t toc = packet[0];
    info.config = static_cast<uint8_t>(toc >> 3u);
    info.stereo = ((toc >> 2u) & 1u) != 0u;
    info.frame_count_code = static_cast<uint8_t>(toc & 3u);
    info.frame_samples_48k = config_frame_samples_48k(info.config);

    bool ok = false;
    switch (info.frame_count_code) {
        case 0u:
            // Single frame: TOC byte + 0 or more payload bytes.
            ok = true;
            break;
        case 1u:
            // Two equal-size frames: (bytes - 1) must be even.
            ok = ((bytes - 1u) & 1u) == 0u && (2u * info.frame_samples_48k <= kMaxOpusFrames);
            break;
        case 2u:
            // Two variable-size frames: 1- or 2-byte length prefix for frame 1.
            if (bytes >= 2u && (2u * info.frame_samples_48k <= kMaxOpusFrames)) {
                const uint8_t b1 = packet[1];
                if (b1 < 252u) {
                    ok = (bytes - 2u) >= size_t{b1};
                } else if (bytes >= 3u) {
                    const size_t len1 = size_t{b1} + 4u * size_t{packet[2]};
                    ok = (bytes - 3u) >= len1;
                }
            }
            break;
        case 3u:
            // Arbitrary number of frames M (1..48), total duration <= 120 ms (5760 samples @ 48k).
            if (bytes >= 2u) {
                const uint32_t m = packet[1] & 0x3Fu;
                ok = m >= 1u && m <= 48u && (uint64_t{m} * info.frame_samples_48k <= kMaxOpusFrames);
            }
            break;
        default:
            ok = false;
            break;
    }

    info.valid = ok;
    if (out != nullptr) {
        *out = info;
    }
    return ok;
}

void mix_pcm_additive(float* dst, const float* src, size_t samples, float gain) {
    if (dst == nullptr || src == nullptr || samples == 0u || !std::isfinite(gain) || gain == 0.0f) {
        return;
    }
    for (size_t i = 0u; i < samples; ++i) {
        const float d = sanitize_sample(dst[i]);
        const float s = sanitize_sample(src[i]);
        dst[i] = sanitize_sample(d + s * gain);
    }
}

void mix_int16_to_float_additive(float* dst, const int16_t* src, size_t samples, float gain) {
    if (dst == nullptr || src == nullptr || samples == 0u || !std::isfinite(gain) || gain == 0.0f) {
        return;
    }
    const float scale = gain * (1.0f / 32768.0f);
    for (size_t i = 0u; i < samples; ++i) {
        const float d = sanitize_sample(dst[i]);
        dst[i] = sanitize_sample(d + static_cast<float>(src[i]) * scale);
    }
}

void sanitize_and_soft_clip(float* pcm, size_t samples) {
    if (pcm == nullptr || samples == 0u) {
        return;
    }
    constexpr float kKnee = 0.85f;
    constexpr float kHeadroom = 1.0f - kKnee;
    for (size_t i = 0u; i < samples; ++i) {
        const float v = sanitize_sample(pcm[i]);
        const float mag = std::fabs(v);
        if (mag <= kKnee) {
            pcm[i] = v;
        } else {
            const float limited = kKnee + kHeadroom * std::tanh((mag - kKnee) / kHeadroom);
            const float clamped = std::min(limited, 1.0f);
            pcm[i] = v < 0.0f ? -clamped : clamped;
        }
    }
}

VoiceDecoder::~VoiceDecoder() { release(); }

void VoiceDecoder::reset() {
    ++reset_count_;
#if defined(PLAYGTA6_HAVE_OPUS)
    if (decoder_ != nullptr) {
        opus_decoder_ctl(static_cast<OpusDecoder*>(decoder_), OPUS_RESET_STATE);
    }
#endif
}

void VoiceDecoder::flush() { reset(); }

void VoiceDecoder::release() {
#if defined(PLAYGTA6_HAVE_OPUS)
    if (decoder_ != nullptr) {
        opus_decoder_destroy(static_cast<OpusDecoder*>(decoder_));
    }
#endif
    decoder_ = nullptr;
    sample_rate_ = 0;
    channels_ = 0;
}

int VoiceDecoder::decode(const uint8_t* packet, size_t bytes, uint32_t sample_rate, uint32_t channels, float* out,
                         size_t max_frames) {
    if (packet == nullptr || bytes == 0u || bytes > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        out == nullptr || max_frames == 0u || channels == 0u || channels > 2u) {
        ++failed_packets_;
        return -1;
    }
    const size_t safe_max_frames = std::min(max_frames, kMaxOpusFrames);
#if defined(PLAYGTA6_HAVE_OPUS)
    const uint32_t rate = opus_rate(sample_rate);
    if (decoder_ == nullptr || rate != sample_rate_ || channels != channels_) {
        release();
        int error = OPUS_OK;
        OpusDecoder* d = opus_decoder_create(static_cast<opus_int32>(rate), static_cast<int>(channels), &error);
        if (d == nullptr || error != OPUS_OK) {
            ++failed_packets_;
            return -1;
        }
        decoder_ = d;
        sample_rate_ = rate;
        channels_ = channels;
    }
    // A packet holds at most 120 ms; decode in full (Opus fails when the buffer is short), then copy.
    static thread_local float pcm[kMaxOpusFrames * 2u];  // 45 KiB: off the (small wasm) stack
    const int frames = opus_decode_float(static_cast<OpusDecoder*>(decoder_), packet,
                                         static_cast<opus_int32>(bytes), pcm, static_cast<int>(kMaxOpusFrames), 0);
    if (frames < 0) {
        ++failed_packets_;
        return -1;
    }
    const size_t n = std::min(static_cast<size_t>(frames), safe_max_frames);
    const size_t sample_count = n * channels;
    for (size_t i = 0u; i < sample_count; ++i) {
        out[i] = sanitize_sample(pcm[i]);
    }
    ++decoded_packets_;
    return static_cast<int>(n);
#else
    (void)sample_rate;
    std::memset(out, 0, safe_max_frames * channels * sizeof(float));
    ++decoded_packets_;
    return static_cast<int>(safe_max_frames);
#endif
}

}  // namespace playgta6::os::xapu_opus
