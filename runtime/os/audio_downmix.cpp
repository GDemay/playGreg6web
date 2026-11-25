#include "os/audio_downmix.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <vector>

namespace playgta6::os::audio_downmix {

namespace {

constexpr float kM3dB = 0.70710678f;
constexpr float kM6dB = 0.5f;
constexpr size_t kMaxStagedFrames = 48000;  // one second per source; older audio is dropped

struct RingStorage {
    float data[kRingSamples] = {};
    uint32_t write = 0;
    uint32_t read = 0;
};
RingStorage& storage() {
    static RingStorage s;
    return s;
}

struct Staged {
    std::mutex mutex;
    std::vector<float> spatial;  // interleaved stereo, already downmixed
    std::vector<float> render;   // interleaved stereo
    StereoSink sink = nullptr;
    void* sink_user = nullptr;
};
Staged& staged() {
    static Staged s;
    return s;
}

void append_capped(std::vector<float>& dst, const float* src, size_t count) {
    dst.insert(dst.end(), src, src + count);
    const size_t cap = kMaxStagedFrames * kChannels;
    if (dst.size() > cap) {
        dst.erase(dst.begin(), dst.begin() + static_cast<std::ptrdiff_t>(dst.size() - cap));
    }
}

uint32_t load(uint32_t* p, std::memory_order o) { return std::atomic_ref<uint32_t>(*p).load(o); }
void store(uint32_t* p, uint32_t v, std::memory_order o) { std::atomic_ref<uint32_t>(*p).store(v, o); }

}  // namespace

StereoGain downmix_gain(uint32_t t) {
    switch (t) {
        case kFrontLeft: return {1.0f, 0.0f};
        case kFrontRight: return {0.0f, 1.0f};
        case kFrontCenter: return {kM3dB, kM3dB};
        case kLowFrequency: return {0.0f, 0.0f};
        case kSideLeft: return {kM3dB, 0.0f};
        case kSideRight: return {0.0f, kM3dB};
        case kBackLeft: return {kM3dB, 0.0f};
        case kBackRight: return {0.0f, kM3dB};
        case kTopFrontLeft: return {kM3dB, 0.0f};
        case kTopFrontRight: return {0.0f, kM3dB};
        case kTopBackLeft: return {kM6dB, 0.0f};
        case kTopBackRight: return {0.0f, kM6dB};
        default: return {0.0f, 0.0f};
    }
}

void mix_object_into_stereo(uint32_t object_type, const float* mono, uint32_t frames, float* out) {
    const StereoGain g = downmix_gain(object_type);
    if (mono == nullptr || out == nullptr || (g.left == 0.0f && g.right == 0.0f)) {
        return;
    }
    for (uint32_t i = 0; i < frames; ++i) {
        out[2 * i] += mono[i] * g.left;
        out[2 * i + 1] += mono[i] * g.right;
    }
}

void mix_stereo_into_stereo(const float* in, uint32_t frames, float* out) {
    if (in == nullptr || out == nullptr) {
        return;
    }
    for (size_t i = 0; i < static_cast<size_t>(frames) * 2; ++i) {
        out[i] += in[i];
    }
}

PcmRing& runtime_ring() {
    static PcmRing ring = [] {
        RingStorage& s = storage();
        return PcmRing{s.data, kRingSamples, &s.write, &s.read};
    }();
    return ring;
}

size_t ring_readable(const PcmRing& ring) {
    const uint32_t w = load(ring.write_index, std::memory_order_acquire);
    const uint32_t r = load(ring.read_index, std::memory_order_acquire);
    return static_cast<uint32_t>(w - r);
}

size_t ring_push(PcmRing& ring, const float* samples, size_t count) {
    const uint32_t w = load(ring.write_index, std::memory_order_relaxed);
    const uint32_t r = load(ring.read_index, std::memory_order_acquire);
    const size_t free_space = ring.capacity - static_cast<uint32_t>(w - r);
    const size_t n = std::min(count, free_space);
    const uint32_t mask = ring.capacity - 1;
    for (size_t i = 0; i < n; ++i) {
        ring.data[(w + static_cast<uint32_t>(i)) & mask] = samples[i];
    }
    store(ring.write_index, w + static_cast<uint32_t>(n), std::memory_order_release);
    return n;
}

size_t ring_pop(PcmRing& ring, float* out, size_t count) {
    const uint32_t r = load(ring.read_index, std::memory_order_relaxed);
    const uint32_t w = load(ring.write_index, std::memory_order_acquire);
    const size_t n = std::min<size_t>(count, static_cast<uint32_t>(w - r));
    const uint32_t mask = ring.capacity - 1;
    for (size_t i = 0; i < n; ++i) {
        out[i] = ring.data[(r + static_cast<uint32_t>(i)) & mask];
    }
    store(ring.read_index, r + static_cast<uint32_t>(n), std::memory_order_release);
    return n;
}

void push_spatial_pass(const uint32_t* types, const float* const* buffers, uint32_t objects, uint32_t frames) {
    if (frames == 0 || objects == 0) {
        return;
    }
    std::vector<float> mix(static_cast<size_t>(frames) * 2, 0.0f);
    for (uint32_t i = 0; i < objects; ++i) {
        mix_object_into_stereo(types[i], buffers[i], frames, mix.data());
    }
    Staged& s = staged();
    std::lock_guard<std::mutex> lock(s.mutex);
    append_capped(s.spatial, mix.data(), mix.size());
}

void push_render_stereo(const float* interleaved, uint32_t frames) {
    if (interleaved == nullptr || frames == 0) {
        return;
    }
    Staged& s = staged();
    std::lock_guard<std::mutex> lock(s.mutex);
    append_capped(s.render, interleaved, static_cast<size_t>(frames) * 2);
}

uint32_t flush_period(uint32_t period_frames) {
    Staged& s = staged();
    std::vector<float> mix;
    StereoSink sink;
    void* user;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        const size_t want = static_cast<size_t>(period_frames) * 2;
        const size_t a = std::min(want, s.spatial.size());
        const size_t b = std::min(want, s.render.size());
        const size_t n = std::max(a, b);
        if (n == 0) {
            return 0;
        }
        mix.assign(n, 0.0f);
        for (size_t i = 0; i < a; ++i) mix[i] += s.spatial[i];
        for (size_t i = 0; i < b; ++i) mix[i] += s.render[i];
        s.spatial.erase(s.spatial.begin(), s.spatial.begin() + static_cast<std::ptrdiff_t>(a));
        s.render.erase(s.render.begin(), s.render.begin() + static_cast<std::ptrdiff_t>(b));
        sink = s.sink;
        user = s.sink_user;
    }
    for (float& v : mix) {
        v = std::isfinite(v) ? std::clamp(v, -1.0f, 1.0f) : 0.0f;
    }
    ring_push(runtime_ring(), mix.data(), mix.size());
    const uint32_t frames = static_cast<uint32_t>(mix.size() / 2);
    if (sink != nullptr) {
        sink(user, mix.data(), frames, kSampleRate);
    }
    return frames;
}

void reset() {
    Staged& s = staged();
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.spatial.clear();
        s.render.clear();
        s.sink = nullptr;
        s.sink_user = nullptr;
    }
    PcmRing& r = runtime_ring();
    store(r.read_index, 0, std::memory_order_seq_cst);
    store(r.write_index, 0, std::memory_order_seq_cst);
}

void set_stereo_sink(StereoSink sink, void* user) {
    Staged& s = staged();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.sink = sink;
    s.sink_user = user;
}

}  // namespace playgta6::os::audio_downmix

extern "C" const Playgta6AudioRing* playgta6_audio_ring(void) {
    using namespace playgta6::os::audio_downmix;
    static const Playgta6AudioRing view = [] {
        PcmRing& r = runtime_ring();
        return Playgta6AudioRing{reinterpret_cast<uintptr_t>(r.data), r.capacity, kChannels, kSampleRate,
                                 reinterpret_cast<uintptr_t>(r.read_index),
                                 reinterpret_cast<uintptr_t>(r.write_index)};
    }();
    return &view;
}
