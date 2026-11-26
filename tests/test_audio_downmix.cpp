// Stereo downmix matrix of the 7.1.4 static objects, the PCM ring (wraparound, full ring, SPSC
// threads, cache-line separation, bounds guards, stereo-frame alignment), the period mixer (circular
// staging overflow, NaN/Inf clamping, lock-free stats) and the host accessor playgta6_audio_ring().

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>

#include "os/audio_downmix.h"

namespace {

int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_audio_downmix.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace ad = playgta6::os::audio_downmix;

bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

void test_matrix() {
    struct Row {
        uint32_t type;
        float l, r;
    };
    const float h = 0.70710678f;
    const Row rows[] = {
        {ad::kFrontLeft, 1, 0},    {ad::kFrontRight, 0, 1},    {ad::kFrontCenter, h, h},
        {ad::kLowFrequency, 0, 0}, {ad::kSideLeft, h, 0},      {ad::kSideRight, 0, h},
        {ad::kBackLeft, h, 0},     {ad::kBackRight, 0, h},     {ad::kTopFrontLeft, h, 0},
        {ad::kTopFrontRight, 0, h}, {ad::kTopBackLeft, 0.5f, 0}, {ad::kTopBackRight, 0, 0.5f},
        {0, 0, 0},                 {1 /*dynamic*/, 0, 0},      {0x2000, 0, 0},
    };
    for (const Row& row : rows) {
        const ad::StereoGain g = ad::downmix_gain(row.type);
        CHECK(near(g.left, row.l));
        CHECK(near(g.right, row.r));
    }
    // A unit impulse on each object lands at exactly its gains, interleaved L/R, accumulating on
    // non-zero output without disturbing untouched channels on single-sided paths.
    const float mono[4] = {1.0f, 0.5f, 0.0f, -1.0f};
    for (const Row& row : rows) {
        float out[8] = {0.25f, -0.25f, 0.25f, -0.25f, 0.25f, -0.25f, 0.25f, -0.25f};
        ad::mix_object_into_stereo(row.type, mono, 4, out);
        for (int i = 0; i < 4; ++i) {
            CHECK(near(out[2 * i], 0.25f + mono[i] * row.l));
            CHECK(near(out[2 * i + 1], -0.25f + mono[i] * row.r));
        }
    }
    // Null and zero-frame guards.
    float untouched[2] = {0.5f, -0.5f};
    ad::mix_object_into_stereo(ad::kFrontLeft, mono, 0, untouched);
    ad::mix_object_into_stereo(ad::kFrontLeft, nullptr, 1, untouched);
    ad::mix_object_into_stereo(ad::kFrontRight, mono, 1, nullptr);
    CHECK(untouched[0] == 0.5f && untouched[1] == -0.5f);

    // All twelve objects at unit level: L and R are symmetric.
    float sum[2] = {};
    for (uint32_t bit = 1; bit < 13; ++bit) {
        const float one = 1.0f;
        ad::mix_object_into_stereo(1u << bit, &one, 1, sum);
    }
    CHECK(near(sum[0], sum[1]));
    CHECK(sum[0] > 4.0f);
}

void test_abi_and_cacheline_separation() {
    static_assert(ad::kSampleRate == 48000);
    static_assert(ad::kChannels == 2);
    static_assert(ad::kRingSamples == (1u << 17));
    static_assert(offsetof(Playgta6AudioRing, base) == 0);
    static_assert(offsetof(Playgta6AudioRing, capacity) == sizeof(uintptr_t));
    static_assert(offsetof(Playgta6AudioRing, channels) == sizeof(uintptr_t) + 4);
    static_assert(offsetof(Playgta6AudioRing, sample_rate) == sizeof(uintptr_t) + 8);

    const Playgta6AudioRing* view = playgta6_audio_ring();
    CHECK(view != nullptr);
    CHECK(view->base == reinterpret_cast<uintptr_t>(ad::runtime_ring().data));
    CHECK(view->capacity == ad::kRingSamples);
    CHECK(view->channels == ad::kChannels);
    CHECK(view->sample_rate == ad::kSampleRate);
    CHECK(view->write_index_ptr == reinterpret_cast<uintptr_t>(ad::runtime_ring().write_index));
    CHECK(view->read_index_ptr == reinterpret_cast<uintptr_t>(ad::runtime_ring().read_index));

    const std::ptrdiff_t idx_delta =
        static_cast<std::ptrdiff_t>(view->write_index_ptr) - static_cast<std::ptrdiff_t>(view->read_index_ptr);
    CHECK(std::abs(idx_delta) >= 64);
    CHECK((view->write_index_ptr % 64) == 0);
    CHECK((view->read_index_ptr % 64) == 0);
    CHECK((view->base % 64) == 0);
}

void test_ring() {
    ad::reset();
    ad::PcmRing& ring = ad::runtime_ring();
    CHECK(ring.capacity == ad::kRingSamples);
    // Wraparound: push/pop in odd chunks well past the capacity, order and values preserved.
    uint32_t next_in = 0, next_out = 0;
    std::vector<float> in, out;
    for (int round = 0; round < 40; ++round) {
        in.resize(5000 + round * 37);
        for (float& v : in) v = static_cast<float>(next_in++);
        CHECK(ad::ring_push(ring, in.data(), in.size()) == in.size());
        out.resize(in.size());
        CHECK(ad::ring_pop(ring, out.data(), out.size()) == out.size());
        for (float v : out) {
            CHECK(v == static_cast<float>(next_out++));
        }
    }
    CHECK(next_in > ring.capacity);  // it really wrapped
    CHECK(ad::ring_readable(ring) == 0);

    // Full ring: extra samples are dropped, unread ones intact.
    std::vector<float> big(ring.capacity + 10, 1.0f);
    CHECK(ad::ring_push(ring, big.data(), big.size()) == ring.capacity);
    CHECK(ad::ring_readable(ring) == ring.capacity);
    float one = 2.0f;
    CHECK(ad::ring_push(ring, &one, 1) == 0);
    float got[3];
    CHECK(ad::ring_pop(ring, got, 3) == 3 && got[0] == 1.0f);
    CHECK(ad::ring_push(ring, &one, 1) == 1);

    // Index wrap at 2^32: start both indices just below it.
    ad::reset();
    *ring.write_index = 0xFFFFFFF0u;
    *ring.read_index = 0xFFFFFFF0u;
    std::vector<float> a(100);
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i);
    CHECK(ad::ring_push(ring, a.data(), a.size()) == 100);
    CHECK(ad::ring_readable(ring) == 100);
    std::vector<float> b(100);
    CHECK(ad::ring_pop(ring, b.data(), 100) == 100);
    CHECK(a == b);
    CHECK(*ring.write_index == 84u);
    ad::reset();
}

void test_bounds_and_corrupted_index_guards() {
    float buf[16] = {};
    float io[32] = {};
    uint32_t w = 0, r = 0;

    // Null pointer and invalid capacity guards.
    ad::PcmRing bad_data{nullptr, 16, &w, &r};
    ad::PcmRing bad_cap0{buf, 0, &w, &r};
    ad::PcmRing bad_cap_np2{buf, 12, &w, &r};
    ad::PcmRing bad_w{buf, 16, nullptr, &r};
    ad::PcmRing bad_r{buf, 16, &w, nullptr};
    ad::PcmRing good{buf, 16, &w, &r};
    for (ad::PcmRing* p : {&bad_data, &bad_cap0, &bad_cap_np2, &bad_w, &bad_r}) {
        CHECK(ad::ring_readable(*p) == 0);
        CHECK(ad::ring_push(*p, io, 4) == 0);
        CHECK(ad::ring_push_frames(*p, io, 2) == 0);
        CHECK(ad::ring_pop(*p, io, 4) == 0);
    }
    CHECK(ad::ring_push(good, nullptr, 4) == 0 && w == 0);
    CHECK(ad::ring_push(good, io, 0) == 0 && w == 0);
    CHECK(ad::ring_push_frames(good, nullptr, 2) == 0 && w == 0);
    CHECK(ad::ring_push_frames(good, io, 0) == 0 && w == 0);
    CHECK(ad::ring_pop(good, nullptr, 4) == 0 && r == 0);
    CHECK(ad::ring_pop(good, io, 0) == 0 && r == 0);

    // Corrupted / desynced index delta (w - r > capacity): no unsigned underflow or OOB read.
    w = 100;
    r = 0;
    CHECK(ad::ring_push(good, io, 4) == 0 && w == 100);
    CHECK(ad::ring_push_frames(good, io, 2) == 0 && w == 100);
    CHECK(ad::ring_readable(good) == 16);
    CHECK(ad::ring_pop(good, io, 32) == 16 && r == 16);

    // Wrapped corrupted delta (w - r = 21 > 16).
    w = 5;
    r = 0xFFFFFFF0u;
    CHECK(ad::ring_push(good, io, 4) == 0 && w == 5);
    CHECK(ad::ring_push_frames(good, io, 2) == 0 && w == 5);
    CHECK(ad::ring_readable(good) == 16);
    CHECK(ad::ring_pop(good, io, 32) == 16 && r == 0u);

    // NaN / +Inf / -Inf clamping and lock-free stats.
    ad::reset();
    const ad::AudioDownmixStats s0 = ad::stats();
    CHECK(s0.periods_flushed == 0 && s0.frames_pushed == 0 && s0.frames_dropped == 0 &&
          s0.staged_spatial_dropped == 0 && s0.staged_render_dropped == 0 && s0.non_finite_clamped == 0);

    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float dirty[8] = {qnan, inf, -inf, 2.5f, -2.5f, 0.25f, -0.5f, qnan};
    ad::push_render_stereo(dirty, 4);
    CHECK(ad::flush_period(4) == 4);

    float out[8] = {};
    CHECK(ad::ring_pop(ad::runtime_ring(), out, 8) == 8);
    CHECK(out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f && out[3] == 1.0f);
    CHECK(out[4] == -1.0f && near(out[5], 0.25f) && near(out[6], -0.5f) && out[7] == 0.0f);

    const ad::AudioDownmixStats s1 = ad::stats();
    CHECK(s1.periods_flushed == 1);
    CHECK(s1.frames_pushed == 4);
    CHECK(s1.frames_dropped == 0);
    CHECK(s1.non_finite_clamped == 4);

    ad::reset();
    const ad::AudioDownmixStats s2 = ad::stats();
    CHECK(s2.periods_flushed == 0 && s2.frames_pushed == 0 && s2.non_finite_clamped == 0);
}

void test_stereo_frame_aligned_overrun() {
    ad::reset();
    ad::PcmRing& ring = ad::runtime_ring();

    // Fill the ring completely, then pop an ODD number of floats (5) so free_space == 5
    // (2 whole stereo frames + 1 stray float), with write_index even and read_index odd.
    std::vector<float> fill(ring.capacity, 0.0f);
    CHECK(ad::ring_push(ring, fill.data(), fill.size()) == ring.capacity);
    float discard[5] = {};
    CHECK(ad::ring_pop(ring, discard, 5) == 5);
    CHECK(ad::ring_readable(ring) == ring.capacity - 5);
    const uint32_t w_before = *ring.write_index;
    CHECK((w_before & 1u) == 0u);

    // Flush 4 stereo frames into the ring with only 5 floats free: exactly 2 whole frames (4
    // floats) must be pushed, leaving write_index even and recording 2 dropped frames.
    const float frames4[8] = {0.1f, -0.1f, 0.2f, -0.2f, 0.3f, -0.3f, 0.4f, -0.4f};
    ad::push_render_stereo(frames4, 4);
    CHECK(ad::flush_period(4) == 4);
    CHECK((*ring.write_index - w_before) == 4u);
    CHECK((*ring.write_index & 1u) == 0u);

    const ad::AudioDownmixStats st = ad::stats();
    CHECK(st.periods_flushed == 1);
    CHECK(st.frames_pushed == 2);
    CHECK(st.frames_dropped == 2);

    // Drain the pre-existing filler floats and verify the 2 pushed frames kept L/R alignment.
    std::vector<float> drain(ring.capacity - 5);
    CHECK(ad::ring_pop(ring, drain.data(), drain.size()) == drain.size());
    float got4[4] = {};
    CHECK(ad::ring_pop(ring, got4, 4) == 4);
    CHECK(near(got4[0], 0.1f) && near(got4[1], -0.1f) && near(got4[2], 0.2f) && near(got4[3], -0.2f));

    // A subsequent stereo frame still lands with L in slot 0 and R in slot 1 (no channel swap).
    const float next1[2] = {0.9f, -0.9f};
    ad::push_render_stereo(next1, 1);
    CHECK(ad::flush_period(1) == 1);
    float got1[2] = {};
    CHECK(ad::ring_pop(ring, got1, 2) == 2);
    CHECK(near(got1[0], 0.9f) && near(got1[1], -0.9f));

    // Direct ring_push_frames check: free_space == 1 fits 0 frames; free_space == 3 fits 1 frame.
    float small_buf[8] = {};
    uint32_t w = 7, r = 0;
    ad::PcmRing small_ring{small_buf, 8, &w, &r};
    CHECK(ad::ring_push_frames(small_ring, next1, 1) == 0 && w == 7);
    r = 2;  // free_space = 8 - 5 = 3 floats -> 1 whole frame
    CHECK(ad::ring_push_frames(small_ring, frames4, 2) == 1 && w == 9);
    ad::reset();
}

void test_circular_staging_overflow() {
    ad::reset();

    // 1. Render staging overflow past 48,000 frames: push 30,000 + 30,000 = 60,000 frames.
    //    Oldest 12,000 frames must be dropped; newest 48,000 frames [12000..59999] survive.
    std::vector<float> batch(30000 * 2);
    for (uint32_t i = 0; i < 30000; ++i) {
        batch[2 * i] = static_cast<float>(i) / 100000.0f;
        batch[2 * i + 1] = -static_cast<float>(i) / 100000.0f;
    }
    ad::push_render_stereo(batch.data(), 30000);
    for (uint32_t i = 0; i < 30000; ++i) {
        const uint32_t idx = 30000 + i;
        batch[2 * i] = static_cast<float>(idx) / 100000.0f;
        batch[2 * i + 1] = -static_cast<float>(idx) / 100000.0f;
    }
    ad::push_render_stereo(batch.data(), 30000);
    CHECK(ad::stats().staged_render_dropped == 12000);

    bool render_ok = true;
    uint32_t expected_idx = 12000;
    std::vector<float> pop_buf(480 * 2);
    for (int p = 0; p < 100; ++p) {
        CHECK(ad::flush_period(480) == 480);
        CHECK(ad::ring_pop(ad::runtime_ring(), pop_buf.data(), pop_buf.size()) == pop_buf.size());
        for (uint32_t i = 0; i < 480; ++i) {
            const float exp = static_cast<float>(expected_idx++) / 100000.0f;
            if (!near(pop_buf[2 * i], exp) || !near(pop_buf[2 * i + 1], -exp)) {
                render_ok = false;
            }
        }
    }
    CHECK(render_ok && expected_idx == 60000);
    CHECK(ad::flush_period(480) == 0);

    // 2. Spatial staging overflow (also exercises >2048-frame fallback scratch and >48000 single push).
    ad::reset();
    std::vector<float> fl(50000), fr(50000);
    for (uint32_t i = 0; i < 50000; ++i) {
        fl[i] = static_cast<float>(i) / 100000.0f;
        fr[i] = -static_cast<float>(i) / 100000.0f;
    }
    const uint32_t types[] = {ad::kFrontLeft, ad::kFrontRight};
    const float* bufs[] = {fl.data(), fr.data()};
    // Push 500 frames first (will all be dropped by the subsequent 50,000-frame push).
    ad::push_spatial_pass(types, bufs, 2, 500);
    ad::push_spatial_pass(types, bufs, 2, 50000);
    // Dropped: 500 previously staged + 2000 leading frames of the 50,000 push = 2500.
    CHECK(ad::stats().staged_spatial_dropped == 2500);

    bool spatial_ok = true;
    expected_idx = 2000;
    // Flush using a >2048-frame period as well to exercise flush_period's fallback path.
    std::vector<float> big_pop(4000 * 2);
    for (int p = 0; p < 12; ++p) {
        CHECK(ad::flush_period(4000) == 4000);
        CHECK(ad::ring_pop(ad::runtime_ring(), big_pop.data(), big_pop.size()) == big_pop.size());
        for (uint32_t i = 0; i < 4000; ++i) {
            const float exp = static_cast<float>(expected_idx++) / 100000.0f;
            if (!near(big_pop[2 * i], exp) || !near(big_pop[2 * i + 1], -exp)) {
                spatial_ok = false;
            }
        }
    }
    CHECK(spatial_ok && expected_idx == 50000);
    CHECK(ad::flush_period(480) == 0);
    ad::reset();
}

void test_spsc_threads() {
    ad::reset();
    ad::PcmRing& ring = ad::runtime_ring();
    constexpr uint32_t kTotal = 1u << 20;
    std::thread producer([&] {
        uint32_t i = 0;
        float v;
        while (i < kTotal) {
            v = static_cast<float>(i);
            if (ad::ring_push(ring, &v, 1) == 1) ++i;
        }
    });
    uint32_t seen = 0;
    bool ordered = true;
    while (seen < kTotal) {
        float v;
        if (ad::ring_pop(ring, &v, 1) == 1) {
            ordered = ordered && v == static_cast<float>(seen);
            ++seen;
        }
    }
    producer.join();
    CHECK(ordered);
    ad::reset();
}

void test_spsc_multithread_variable_chunks_wrap32() {
    ad::reset();
    ad::PcmRing& ring = ad::runtime_ring();
    // Start 65,536 floats below 2^32 so both indices cross 0xFFFFFFFF -> 0 during the stress run.
    constexpr uint32_t kStartIdx = 0xFFFF0000u;
    constexpr uint32_t kTotal = 1u << 20;  // 1,048,576 floats (~8 full ring laps)
    *ring.write_index = kStartIdx;
    *ring.read_index = kStartIdx;

    std::thread producer([&] {
        static constexpr size_t kPushChunks[] = {1, 2, 7, 13, 64, 255, 960, 1023, 4097};
        std::vector<float> buf(4097);
        uint32_t sent = 0;
        size_t step = 0;
        while (sent < kTotal) {
            const size_t want = std::min<size_t>(kPushChunks[step++ % 9], kTotal - sent);
            for (size_t i = 0; i < want; ++i) {
                buf[i] = static_cast<float>((sent + static_cast<uint32_t>(i)) & 0xFFFFu);
            }
            size_t offset = 0;
            while (offset < want) {
                const size_t pushed = ad::ring_push(ring, buf.data() + offset, want - offset);
                offset += pushed;
                if (pushed == 0) {
                    std::this_thread::yield();
                }
            }
            sent += static_cast<uint32_t>(want);
        }
    });

    static constexpr size_t kPopChunks[] = {3, 5, 16, 127, 480, 1000, 3001};
    std::vector<float> out(3001);
    uint32_t seen = 0;
    size_t step = 0;
    bool ordered = true;
    while (seen < kTotal) {
        const size_t want = std::min<size_t>(kPopChunks[step++ % 7], kTotal - seen);
        const size_t got = ad::ring_pop(ring, out.data(), want);
        if (got == 0) {
            std::this_thread::yield();
            continue;
        }
        for (size_t i = 0; i < got; ++i) {
            if (out[i] != static_cast<float>((seen + static_cast<uint32_t>(i)) & 0xFFFFu)) {
                ordered = false;
            }
        }
        seen += static_cast<uint32_t>(got);
    }
    producer.join();
    CHECK(ordered);
    CHECK(ad::ring_readable(ring) == 0);
    CHECK(*ring.write_index == static_cast<uint32_t>(kStartIdx + kTotal));
    CHECK(*ring.read_index == static_cast<uint32_t>(kStartIdx + kTotal));
    ad::reset();
}

struct Capture {
    std::vector<float> pcm;
    uint32_t rate = 0;
};
void capture_sink(void* user, const float* stereo, uint32_t frames, uint32_t rate) {
    auto* c = static_cast<Capture*>(user);
    c->pcm.insert(c->pcm.end(), stereo, stereo + frames * 2);
    c->rate = rate;
}

void test_mixer_and_accessor() {
    ad::reset();
    Capture cap;
    ad::set_stereo_sink(&capture_sink, &cap);
    CHECK(ad::flush_period(480) == 0);  // nothing staged: nothing delivered

    std::vector<float> fl(480, 0.5f), fc(480, 0.25f);
    const uint32_t types[] = {ad::kFrontLeft, ad::kFrontCenter};
    const float* bufs[] = {fl.data(), fc.data()};
    ad::push_spatial_pass(types, bufs, 2, 480);
    std::vector<float> render(480 * 2, 0.1f);
    ad::push_render_stereo(render.data(), 480);

    CHECK(ad::flush_period(480) == 480);
    CHECK(cap.pcm.size() == 960 && cap.rate == 48000);
    CHECK(near(cap.pcm[0], 0.5f + 0.25f * 0.70710678f + 0.1f));
    CHECK(near(cap.pcm[1], 0.25f * 0.70710678f + 0.1f));

    const Playgta6AudioRing* view = playgta6_audio_ring();
    CHECK(view->base == reinterpret_cast<uintptr_t>(ad::runtime_ring().data));
    CHECK(view->capacity == ad::kRingSamples && (view->capacity & (view->capacity - 1)) == 0);
    CHECK(view->channels == 2 && view->sample_rate == 48000);
    CHECK(view->write_index_ptr != view->read_index_ptr);
    CHECK(*reinterpret_cast<uint32_t*>(view->write_index_ptr) == 960u);
    CHECK(*reinterpret_cast<uint32_t*>(view->read_index_ptr) == 0u);
    const float* data = reinterpret_cast<const float*>(view->base);
    CHECK(near(data[0], cap.pcm[0]));

    // Clamped output.
    std::vector<float> loud(480, 1.0f);
    const uint32_t t1[] = {ad::kFrontLeft};
    const float* b1[] = {loud.data()};
    ad::push_spatial_pass(t1, b1, 1, 480);
    ad::push_render_stereo(render.data(), 480);
    ad::set_stereo_sink(nullptr, nullptr);
    CHECK(ad::flush_period(480) == 480);
    float peak[2];
    std::vector<float> all(960 + 960);
    CHECK(ad::ring_pop(ad::runtime_ring(), all.data(), all.size()) == 1920);
    peak[0] = all[960];
    CHECK(peak[0] == 1.0f);
    ad::reset();
}

void test_nan_isolation_and_clamping() {
    ad::reset();
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    // Object 0 (FrontLeft) carries NaN/+Inf/-Inf, while Object 1 (FrontCenter) carries valid 0.5f.
    // Non-finite samples in one object must never poison valid samples from other objects.
    const float bad_fl[4] = {qnan, inf, -inf, 0.25f};
    const float good_fc[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    const uint32_t types[2] = {ad::kFrontLeft, ad::kFrontCenter};
    const float* bufs[2] = {bad_fl, good_fc};
    ad::push_spatial_pass(types, bufs, 2, 4);
    CHECK(ad::flush_period(4) == 4);

    float out[8] = {};
    CHECK(ad::ring_pop(ad::runtime_ring(), out, 8) == 8);
    const float fc_gain = 0.5f * 0.70710678f;
    CHECK(near(out[0], fc_gain) && near(out[1], fc_gain));
    CHECK(near(out[2], fc_gain) && near(out[3], fc_gain));
    CHECK(near(out[4], fc_gain) && near(out[5], fc_gain));
    CHECK(near(out[6], 0.25f + fc_gain) && near(out[7], fc_gain));
    CHECK(ad::stats().non_finite_clamped == 3);

    // Direct mix_stereo_into_stereo also ignores non-finite samples without poisoning accumulator.
    float acc[4] = {0.25f, -0.25f, 0.5f, -0.5f};
    const float dirty_stereo[4] = {qnan, 0.5f, inf, -inf};
    ad::mix_stereo_into_stereo(dirty_stereo, 2, acc);
    CHECK(near(acc[0], 0.25f) && near(acc[1], 0.25f) && near(acc[2], 0.5f) && near(acc[3], -0.5f));
    ad::reset();
}

void test_flush_pending_catchup_and_wraparound() {
    ad::reset();
    Capture cap;
    ad::set_stereo_sink(&capture_sink, &cap);

    // 1. Single-source backlog catch-up: stage 6 periods of 480 frames, flush_pending(480, 4)
    //    must drain 4 periods (1920 frames) immediately and leave 2 periods (960 frames) staged.
    std::vector<float> fl(480), fr(480);
    const uint32_t types[2] = {ad::kFrontLeft, ad::kFrontRight};
    const float* bufs[2] = {fl.data(), fr.data()};
    for (uint32_t p = 0; p < 6; ++p) {
        for (uint32_t i = 0; i < 480; ++i) {
            const float v = static_cast<float>(p * 480 + i) / 10000.0f;
            fl[i] = v;
            fr[i] = -v;
        }
        ad::push_spatial_pass(types, bufs, 2, 480);
    }
    CHECK(ad::staged_frames() == 6 * 480);
    CHECK(ad::flush_pending(480, 4) == 4 * 480);
    CHECK(ad::staged_frames() == 2 * 480);
    CHECK(ad::flush_pending(480, 4) == 2 * 480);
    CHECK(ad::staged_frames() == 0);
    CHECK(cap.pcm.size() == 6 * 480 * 2);

    std::vector<float> got(6 * 480 * 2);
    CHECK(ad::ring_pop(ad::runtime_ring(), got.data(), got.size()) == got.size());
    bool ramp_ok = true;
    for (uint32_t i = 0; i < 6 * 480; ++i) {
        const float exp = static_cast<float>(i) / 10000.0f;
        if (!near(got[2 * i], exp) || !near(got[2 * i + 1], -exp)) {
            ramp_ok = false;
        }
    }
    CHECK(ramp_ok);

    // 2. Dual-source lockstep: 2 spatial periods + 1 render period -> flush_pending flushes only
    //    the 1 paired period so spatial[480..959] stays aligned for render's 2nd period.
    ad::reset();
    std::vector<float> sp1(480, 0.25f), sp2(480, 0.5f), rd1(480 * 2, 0.1f), rd2(480 * 2, 0.2f);
    const uint32_t t1[1] = {ad::kFrontLeft};
    const float* b1[1] = {sp1.data()};
    const float* b2[1] = {sp2.data()};
    ad::push_spatial_pass(t1, b1, 1, 480);
    ad::push_spatial_pass(t1, b2, 1, 480);
    ad::push_render_stereo(rd1.data(), 480);
    CHECK(ad::flush_pending(480, 4) == 480);
    CHECK(ad::staged_frames() == 480);
    ad::push_render_stereo(rd2.data(), 480);
    CHECK(ad::flush_pending(480, 4) == 480);
    CHECK(ad::staged_frames() == 0);
    std::vector<float> paired(960 * 2);
    CHECK(ad::ring_pop(ad::runtime_ring(), paired.data(), paired.size()) == paired.size());
    CHECK(near(paired[0], 0.35f) && near(paired[1], 0.1f));
    CHECK(near(paired[960], 0.7f) && near(paired[961], 0.2f));

    // 3. Ring backpressure: when the ring only has room for 1 period + 100 frames, flush_pending
    //    pushes the 1st period and leaves extra catch-up periods staged rather than dropping them.
    ad::reset();
    ad::PcmRing& ring = ad::runtime_ring();
    const size_t cap_frames = ring.capacity / 2;
    std::vector<float> filler((cap_frames - 580) * 2, 0.0f);
    CHECK(ad::ring_push(ring, filler.data(), filler.size()) == filler.size());
    for (int p = 0; p < 3; ++p) {
        ad::push_render_stereo(rd1.data(), 480);
    }
    CHECK(ad::flush_pending(480, 4) == 480);
    CHECK(ad::staged_frames() == 2 * 480);
    CHECK(ad::stats().frames_dropped == 0);

    // 4. Multi-lap continuous wraparound + concurrent flush serialization (> 2 full ring laps).
    ad::reset();
    constexpr uint32_t kTotalPeriods = 300;  // 300 * 480 = 144,000 frames (> 2.19x 65536-frame ring)
    std::thread producer([&] {
        std::vector<float> chunk(480 * 2);
        for (uint32_t p = 0; p < kTotalPeriods; ++p) {
            for (uint32_t i = 0; i < 480; ++i) {
                const float v = static_cast<float>((p * 480 + i) & 0x3FFu) / 1024.0f;
                chunk[2 * i] = v;
                chunk[2 * i + 1] = -v;
            }
            ad::push_render_stereo(chunk.data(), 480);
            // Alternate between single-period flush and multi-period catch-up flush.
            if ((p & 3u) == 3u) {
                while (ad::staged_frames() >= 480) {
                    const size_t free_f = (ring.capacity - ad::ring_readable(ring)) / 2;
                    if (free_f < 480) {
                        std::this_thread::yield();
                        continue;
                    }
                    ad::flush_pending(480, 4);
                }
            }
        }
        while (ad::staged_frames() > 0) {
            const size_t free_f = (ring.capacity - ad::ring_readable(ring)) / 2;
            if (free_f < 480) {
                std::this_thread::yield();
                continue;
            }
            ad::flush_pending(480, 4);
        }
    });

    uint32_t frames_read = 0;
    bool continuous = true;
    std::vector<float> out(512 * 2);
    while (frames_read < kTotalPeriods * 480) {
        const size_t got_samples = ad::ring_pop(ring, out.data(), out.size());
        if (got_samples == 0) {
            std::this_thread::yield();
            continue;
        }
        CHECK((got_samples & 1u) == 0u);
        const uint32_t got_frames = static_cast<uint32_t>(got_samples / 2);
        for (uint32_t i = 0; i < got_frames; ++i) {
            const float exp = static_cast<float>((frames_read + i) & 0x3FFu) / 1024.0f;
            if (!near(out[2 * i], exp) || !near(out[2 * i + 1], -exp)) {
                continuous = false;
            }
        }
        frames_read += got_frames;
    }
    producer.join();
    CHECK(continuous);
    CHECK(ad::stats().frames_dropped == 0);
    ad::reset();
}

}  // namespace

int main() {
    test_matrix();
    test_abi_and_cacheline_separation();
    test_ring();
    test_bounds_and_corrupted_index_guards();
    test_stereo_frame_aligned_overrun();
    test_circular_staging_overflow();
    test_spsc_threads();
    test_spsc_multithread_variable_chunks_wrap32();
    test_mixer_and_accessor();
    test_nan_isolation_and_clamping();
    test_flush_pending_catchup_and_wraparound();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_audio_downmix failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_audio_downmix: all checks passed\n");
    return 0;
}
