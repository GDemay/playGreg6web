#include "os/xapu_opus.h"

#include <cstring>

#if defined(PLAYGTA6_HAVE_OPUS)
#include <opus.h>
#endif

namespace playgta6::os::xapu_opus {

namespace {

// Opus decodes to these rates only; anything else is taken as the nearest one at or above it.
uint32_t opus_rate(uint32_t rate) {
    static constexpr uint32_t kRates[] = {8000u, 12000u, 16000u, 24000u};
    for (uint32_t r : kRates) {
        if (rate <= r) {
            return r;
        }
    }
    return 48000;
}

}  // namespace

bool available() {
#if defined(PLAYGTA6_HAVE_OPUS)
    return true;
#else
    return false;
#endif
}

VoiceDecoder::~VoiceDecoder() { reset(); }

void VoiceDecoder::reset() {
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
    if (packet == nullptr || bytes == 0 || out == nullptr || max_frames == 0 || channels == 0 || channels > 2) {
        return -1;
    }
#if defined(PLAYGTA6_HAVE_OPUS)
    const uint32_t rate = opus_rate(sample_rate);
    if (decoder_ == nullptr || rate != sample_rate_ || channels != channels_) {
        reset();
        int error = OPUS_OK;
        OpusDecoder* d = opus_decoder_create(static_cast<opus_int32>(rate), static_cast<int>(channels), &error);
        if (d == nullptr || error != OPUS_OK) {
            return -1;
        }
        decoder_ = d;
        sample_rate_ = rate;
        channels_ = channels;
    }
    // A packet holds at most 120 ms; decode in full (Opus fails when the buffer is short), then copy.
    constexpr size_t kMaxFrames = 48000 * 120 / 1000;
    static thread_local float pcm[kMaxFrames * 2];  // 45 KiB: off the (small wasm) stack
    const int frames = opus_decode_float(static_cast<OpusDecoder*>(decoder_), packet,
                                         static_cast<opus_int32>(bytes), pcm, static_cast<int>(kMaxFrames), 0);
    if (frames < 0) {
        return -1;
    }
    const size_t n = static_cast<size_t>(frames) < max_frames ? static_cast<size_t>(frames) : max_frames;
    std::memcpy(out, pcm, n * channels * sizeof(float));
    return static_cast<int>(n);
#else
    (void)sample_rate;
    std::memset(out, 0, max_frames * channels * sizeof(float));
    return static_cast<int>(max_frames);
#endif
}

}  // namespace playgta6::os::xapu_opus
