#include "os/audio_downmix.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <vector>

namespace playgta6::os::audio_downmix {

namespace {

constexpr float kM3dB = 0.70710678f;
constexpr float kM6dB = 0.5f;
constexpr size_t kMaxStagedFrames = 48000;  // one second per source; older audio is dropped
constexpr size_t kMaxStagedSamples = kMaxStagedFrames * kChannels;
constexpr size_t kStackMixFrames = 2048;    // stack scratch covers all normal periods (<= 42.6 ms)

// Keep producer `write` and consumer `read` on separate 64-byte cache lines to eliminate SPSC
// false sharing between the runtime engine thread and the host/browser audio worker.
struct RingStorage {
    alignas(64) float data[kRingSamples] = {};
    alignas(64) uint32_t write = 0;
    alignas(64) uint32_t read = 0;
};
static_assert(offsetof(RingStorage, read) - offsetof(RingStorage, write) >= 64,
              "SPSC write and read indices must reside on separate 64-byte cache lines");

RingStorage& storage() {
    static RingStorage s;
    return s;
}

// Fixed-capacity circular buffer for interleaved stereo staging (O(1) push/pop, zero heap
// allocations). Overflow drops the oldest stereo frames and returns how many frames were dropped.
class StagedRing {
public:
    size_t size() const { return count_; }
    void clear() {
        head_ = 0;
        count_ = 0;
    }

    // Appends `sample_count` interleaved stereo floats (must be a multiple of kChannels).
    // Returns the number of oldest frames dropped due to capacity overflow.
    uint64_t push(const float* src, size_t sample_count) {
        if (src == nullptr || sample_count == 0) {
            return 0;
        }
        if (sample_count >= kMaxStagedSamples) {
            const size_t keep_offset = sample_count - kMaxStagedSamples;
            const uint64_t dropped_samples = static_cast<uint64_t>(count_) + static_cast<uint64_t>(keep_offset);
            std::memcpy(buf_, src + keep_offset, kMaxStagedSamples * sizeof(float));
            head_ = 0;
            count_ = kMaxStagedSamples;
            return dropped_samples / kChannels;
        }
        uint64_t dropped_samples = 0;
        if (count_ + sample_count > kMaxStagedSamples) {
            const size_t overflow = (count_ + sample_count) - kMaxStagedSamples;
            head_ = (head_ + overflow) % kMaxStagedSamples;
            count_ -= overflow;
            dropped_samples = static_cast<uint64_t>(overflow);
        }
        size_t tail = head_ + count_;
        if (tail >= kMaxStagedSamples) {
            tail -= kMaxStagedSamples;
        }
        const size_t first = std::min(sample_count, kMaxStagedSamples - tail);
        std::memcpy(buf_ + tail, src, first * sizeof(float));
        const size_t second = sample_count - first;
        if (second != 0) {
            std::memcpy(buf_, src + first, second * sizeof(float));
        }
        count_ += sample_count;
        return dropped_samples / kChannels;
    }

    // Copies up to `max_samples` out into `dst` and advances the head.
    size_t pop_copy(float* dst, size_t max_samples) {
        const size_t n = std::min(max_samples, count_);
        if (n == 0) {
            return 0;
        }
        const size_t first = std::min(n, kMaxStagedSamples - head_);
        std::memcpy(dst, buf_ + head_, first * sizeof(float));
        const size_t second = n - first;
        if (second != 0) {
            std::memcpy(dst + first, buf_, second * sizeof(float));
        }
        head_ += n;
        if (head_ >= kMaxStagedSamples) {
            head_ -= kMaxStagedSamples;
        }
        count_ -= n;
        return n;
    }

    // Accumulates up to `max_samples` into `dst` and advances the head.
    size_t pop_add(float* dst, size_t max_samples) {
        const size_t n = std::min(max_samples, count_);
        if (n == 0) {
            return 0;
        }
        const size_t first = std::min(n, kMaxStagedSamples - head_);
        const float* p1 = buf_ + head_;
        for (size_t i = 0; i < first; ++i) {
            dst[i] += p1[i];
        }
        const size_t second = n - first;
        if (second != 0) {
            const float* p2 = buf_;
            for (size_t i = 0; i < second; ++i) {
                dst[first + i] += p2[i];
            }
        }
        head_ += n;
        if (head_ >= kMaxStagedSamples) {
            head_ -= kMaxStagedSamples;
        }
        count_ -= n;
        return n;
    }

private:
    float buf_[kMaxStagedSamples] = {};
    size_t head_ = 0;
    size_t count_ = 0;
};

struct Staged {
    std::mutex mutex;
    StagedRing spatial;  // interleaved stereo, already downmixed
    StagedRing render;   // interleaved stereo
    StereoSink sink = nullptr;
    void* sink_user = nullptr;
};
Staged& staged() {
    static Staged s;
    return s;
}

struct Counters {
    std::atomic<uint64_t> periods_flushed{0};
    std::atomic<uint64_t> frames_pushed{0};
    std::atomic<uint64_t> frames_dropped{0};
    std::atomic<uint64_t> staged_spatial_dropped{0};
    std::atomic<uint64_t> staged_render_dropped{0};
    std::atomic<uint64_t> non_finite_clamped{0};
};
Counters& counters() {
    static Counters c;
    return c;
}

bool valid_ring(const PcmRing& ring) {
    return ring.data != nullptr && ring.capacity != 0 && (ring.capacity & (ring.capacity - 1)) == 0 &&
           ring.write_index != nullptr && ring.read_index != nullptr;
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
    if (mono == nullptr || out == nullptr || frames == 0 || (g.left == 0.0f && g.right == 0.0f)) {
        return;
    }
    if (g.right == 0.0f) {
        if (g.left == 1.0f) {
            for (uint32_t i = 0; i < frames; ++i) {
                out[static_cast<size_t>(i) * 2] += mono[i];
            }
        } else {
            const float gl = g.left;
            for (uint32_t i = 0; i < frames; ++i) {
                out[static_cast<size_t>(i) * 2] += mono[i] * gl;
            }
        }
        return;
    }
    if (g.left == 0.0f) {
        if (g.right == 1.0f) {
            for (uint32_t i = 0; i < frames; ++i) {
                out[static_cast<size_t>(i) * 2 + 1] += mono[i];
            }
        } else {
            const float gr = g.right;
            for (uint32_t i = 0; i < frames; ++i) {
                out[static_cast<size_t>(i) * 2 + 1] += mono[i] * gr;
            }
        }
        return;
    }
    const float gl = g.left;
    const float gr = g.right;
    for (uint32_t i = 0; i < frames; ++i) {
        const float s = mono[i];
        out[static_cast<size_t>(i) * 2] += s * gl;
        out[static_cast<size_t>(i) * 2 + 1] += s * gr;
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
    if (!valid_ring(ring)) {
        return 0;
    }
    const uint32_t w = load(ring.write_index, std::memory_order_acquire);
    const uint32_t r = load(ring.read_index, std::memory_order_acquire);
    const uint32_t used = static_cast<uint32_t>(w - r);
    return std::min<size_t>(used, ring.capacity);
}

size_t ring_push(PcmRing& ring, const float* samples, size_t count) {
    if (!valid_ring(ring) || samples == nullptr || count == 0) {
        return 0;
    }
    const uint32_t w = load(ring.write_index, std::memory_order_relaxed);
    const uint32_t r = load(ring.read_index, std::memory_order_acquire);
    const uint32_t used = static_cast<uint32_t>(w - r);
    if (used >= ring.capacity) {
        return 0;
    }
    const size_t free_space = static_cast<size_t>(ring.capacity - used);
    const size_t n = std::min(count, free_space);
    if (n == 0) {
        return 0;
    }
    const uint32_t mask = ring.capacity - 1;
    const uint32_t start = w & mask;
    const size_t first = std::min<size_t>(n, ring.capacity - start);
    std::memcpy(ring.data + start, samples, first * sizeof(float));
    const size_t second = n - first;
    if (second != 0) {
        std::memcpy(ring.data, samples + first, second * sizeof(float));
    }
    store(ring.write_index, w + static_cast<uint32_t>(n), std::memory_order_release);
    return n;
}

size_t ring_push_frames(PcmRing& ring, const float* samples, size_t frame_count) {
    if (!valid_ring(ring) || ring.capacity < kChannels || samples == nullptr || frame_count == 0) {
        return 0;
    }
    const uint32_t w = load(ring.write_index, std::memory_order_relaxed);
    const uint32_t r = load(ring.read_index, std::memory_order_acquire);
    const uint32_t used = static_cast<uint32_t>(w - r);
    if (used >= ring.capacity) {
        return 0;
    }
    const size_t free_frames = static_cast<size_t>(ring.capacity - used) / kChannels;
    const size_t n_frames = std::min(frame_count, free_frames);
    if (n_frames == 0) {
        return 0;
    }
    const size_t n = n_frames * kChannels;
    const uint32_t mask = ring.capacity - 1;
    const uint32_t start = w & mask;
    const size_t first = std::min<size_t>(n, ring.capacity - start);
    std::memcpy(ring.data + start, samples, first * sizeof(float));
    const size_t second = n - first;
    if (second != 0) {
        std::memcpy(ring.data, samples + first, second * sizeof(float));
    }
    store(ring.write_index, w + static_cast<uint32_t>(n), std::memory_order_release);
    return n_frames;
}

size_t ring_pop(PcmRing& ring, float* out, size_t count) {
    if (!valid_ring(ring) || out == nullptr || count == 0) {
        return 0;
    }
    const uint32_t r = load(ring.read_index, std::memory_order_relaxed);
    const uint32_t w = load(ring.write_index, std::memory_order_acquire);
    const uint32_t used = static_cast<uint32_t>(w - r);
    const size_t avail = std::min<size_t>(used, ring.capacity);
    const size_t n = std::min(count, avail);
    if (n == 0) {
        return 0;
    }
    const uint32_t mask = ring.capacity - 1;
    const uint32_t start = r & mask;
    const size_t first = std::min<size_t>(n, ring.capacity - start);
    std::memcpy(out, ring.data + start, first * sizeof(float));
    const size_t second = n - first;
    if (second != 0) {
        std::memcpy(out + first, ring.data, second * sizeof(float));
    }
    store(ring.read_index, r + static_cast<uint32_t>(n), std::memory_order_release);
    return n;
}

void push_spatial_pass(const uint32_t* types, const float* const* buffers, uint32_t objects, uint32_t frames) {
    if (types == nullptr || buffers == nullptr || frames == 0 || objects == 0) {
        return;
    }
    const size_t total_samples = static_cast<size_t>(frames) * kChannels;
    float stack_mix[kStackMixFrames * kChannels];
    float* mix_ptr = stack_mix;
    if (frames <= kStackMixFrames) {
        std::memset(stack_mix, 0, total_samples * sizeof(float));
    } else {
        thread_local std::vector<float> fallback_mix;
        fallback_mix.assign(total_samples, 0.0f);
        mix_ptr = fallback_mix.data();
    }
    for (uint32_t i = 0; i < objects; ++i) {
        mix_object_into_stereo(types[i], buffers[i], frames, mix_ptr);
    }
    uint64_t dropped = 0;
    {
        Staged& s = staged();
        std::lock_guard<std::mutex> lock(s.mutex);
        dropped = s.spatial.push(mix_ptr, total_samples);
    }
    if (dropped != 0) {
        counters().staged_spatial_dropped.fetch_add(dropped, std::memory_order_relaxed);
    }
}

void push_render_stereo(const float* interleaved, uint32_t frames) {
    if (interleaved == nullptr || frames == 0) {
        return;
    }
    uint64_t dropped = 0;
    {
        Staged& s = staged();
        std::lock_guard<std::mutex> lock(s.mutex);
        dropped = s.render.push(interleaved, static_cast<size_t>(frames) * kChannels);
    }
    if (dropped != 0) {
        counters().staged_render_dropped.fetch_add(dropped, std::memory_order_relaxed);
    }
}

uint32_t flush_period(uint32_t period_frames) {
    if (period_frames == 0) {
        return 0;
    }
    Staged& s = staged();
    float stack_mix[kStackMixFrames * kChannels];
    float* mix_ptr = stack_mix;
    size_t n = 0;
    StereoSink sink = nullptr;
    void* user = nullptr;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        const size_t want = static_cast<size_t>(period_frames) * kChannels;
        const size_t a = std::min(want, s.spatial.size());
        const size_t b = std::min(want, s.render.size());
        n = std::max(a, b);
        if (n == 0) {
            return 0;
        }
        if (n > kStackMixFrames * kChannels) {
            thread_local std::vector<float> fallback_mix;
            if (fallback_mix.size() < n) {
                fallback_mix.resize(n);
            }
            mix_ptr = fallback_mix.data();
        }
        if (a >= b) {
            s.spatial.pop_copy(mix_ptr, a);
            if (b != 0) {
                s.render.pop_add(mix_ptr, b);
            }
        } else {
            s.render.pop_copy(mix_ptr, b);
            if (a != 0) {
                s.spatial.pop_add(mix_ptr, a);
            }
        }
        sink = s.sink;
        user = s.sink_user;
    }

    uint64_t non_finite = 0;
    for (size_t i = 0; i < n; ++i) {
        const float v = mix_ptr[i];
        if (std::isfinite(v)) {
            mix_ptr[i] = std::clamp(v, -1.0f, 1.0f);
        } else {
            mix_ptr[i] = 0.0f;
            ++non_finite;
        }
    }

    const uint32_t frames = static_cast<uint32_t>(n / kChannels);
    const size_t pushed_frames = ring_push_frames(runtime_ring(), mix_ptr, frames);
    Counters& c = counters();
    c.periods_flushed.fetch_add(1, std::memory_order_relaxed);
    c.frames_pushed.fetch_add(static_cast<uint64_t>(pushed_frames), std::memory_order_relaxed);
    if (pushed_frames < frames) {
        c.frames_dropped.fetch_add(static_cast<uint64_t>(frames - pushed_frames), std::memory_order_relaxed);
    }
    if (non_finite != 0) {
        c.non_finite_clamped.fetch_add(non_finite, std::memory_order_relaxed);
    }
    if (sink != nullptr) {
        sink(user, mix_ptr, frames, kSampleRate);
    }
    return frames;
}

AudioDownmixStats stats() {
    const Counters& c = counters();
    AudioDownmixStats out;
    out.periods_flushed = c.periods_flushed.load(std::memory_order_relaxed);
    out.frames_pushed = c.frames_pushed.load(std::memory_order_relaxed);
    out.frames_dropped = c.frames_dropped.load(std::memory_order_relaxed);
    out.staged_spatial_dropped = c.staged_spatial_dropped.load(std::memory_order_relaxed);
    out.staged_render_dropped = c.staged_render_dropped.load(std::memory_order_relaxed);
    out.non_finite_clamped = c.non_finite_clamped.load(std::memory_order_relaxed);
    return out;
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
    Counters& c = counters();
    c.periods_flushed.store(0, std::memory_order_relaxed);
    c.frames_pushed.store(0, std::memory_order_relaxed);
    c.frames_dropped.store(0, std::memory_order_relaxed);
    c.staged_spatial_dropped.store(0, std::memory_order_relaxed);
    c.staged_render_dropped.store(0, std::memory_order_relaxed);
    c.non_finite_clamped.store(0, std::memory_order_relaxed);
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
