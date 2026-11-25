// Stereo downmix matrix of the 7.1.4 static objects, the PCM ring (wraparound, full ring, SPSC
// threads), the period mixer and the host accessor playgta6_audio_ring().

#include <cmath>
#include <cstdint>
#include <cstdio>
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
    // A unit impulse on each object lands at exactly its gains, interleaved L/R.
    const float mono[4] = {1.0f, 0.5f, 0.0f, -1.0f};
    for (const Row& row : rows) {
        float out[8] = {};
        ad::mix_object_into_stereo(row.type, mono, 4, out);
        for (int i = 0; i < 4; ++i) {
            CHECK(near(out[2 * i], mono[i] * row.l));
            CHECK(near(out[2 * i + 1], mono[i] * row.r));
        }
    }
    // All twelve objects at unit level: L and R are symmetric.
    float sum[2] = {};
    for (uint32_t bit = 1; bit < 13; ++bit) {
        const float one = 1.0f;
        ad::mix_object_into_stereo(1u << bit, &one, 1, sum);
    }
    CHECK(near(sum[0], sum[1]));
    CHECK(sum[0] > 4.0f);
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

}  // namespace

int main() {
    test_matrix();
    test_ring();
    test_spsc_threads();
    test_mixer_and_accessor();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_audio_downmix failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_audio_downmix: all checks passed\n");
    return 0;
}
