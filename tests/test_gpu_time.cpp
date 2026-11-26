// Synthetic tests for per-pipeline GPU time aggregation and the query-set ring (gpu/gpu_time.h).
#include <cstdio>
#include <string>

#include "gpu/gpu_time.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_gpu_time.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
using namespace playgta6::gpu::d3d12_guest;
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
    // Nothing fed: unavailable, no invented numbers.
    GpuTimeTable none;
    const std::string j0 = none.json();
    CHECK(has(j0, "\"available\": false"));
    CHECK(has(j0, "\"reason\""));
    CHECK(has(j0, "\"top\": []"));
    none.set_support(GpuTimeTable::Support::Unavailable, "adapter lacks timestamp-query");
    CHECK(has(none.json(), "adapter lacks timestamp-query"));

    // Aggregation per (hash, kind), sorted by total.
    GpuTimeTable t;
    t.record(0xa, GpuPassKind::Render, 100);
    t.record(0xa, GpuPassKind::Render, 300);
    t.record(0xa, GpuPassKind::Compute, 50);
    t.record(0xb, GpuPassKind::Render, 1000);
    CHECK(t.support() == GpuTimeTable::Support::Available);
    CHECK(t.tracked() == 3);
    auto top = t.top(10);
    CHECK(top.size() == 3);
    CHECK(top[0].hash == 0xb && top[0].stats.total_ns == 1000);
    CHECK(top[1].hash == 0xa && top[1].kind == GpuPassKind::Render);
    CHECK(top[1].stats.passes == 2 && top[1].stats.total_ns == 400 && top[1].stats.max_ns == 300);
    CHECK(t.top(1).size() == 1);
    const std::string j = t.json(2);
    CHECK(has(j, "\"available\": true"));
    CHECK(has(j, "\"hash\": \"0x000000000000000b\", \"kind\": \"render\", \"passes\": 1, \"total_ns\": 1000"));
    CHECK(has(j, "\"avg_ns\": 200"));
    CHECK(!has(j, "\"compute\""));  // cut by top_n=2
    CHECK(has(j, "\"total_ns\": 1450"));

    // Bounded: keys past the cap land in "other".
    GpuTimeTable b;
    for (uint64_t i = 1; i <= GpuTimeTable::kMaxKeys + 5; ++i) b.record(i, GpuPassKind::Render, 10);
    CHECK(b.tracked() == GpuTimeTable::kMaxKeys);
    CHECK(has(b.json(1), "\"other\": {\"passes\": 5, \"total_ns\": 50}"));

    // Ring: two slots, a slot that has not resolved is skipped instead of stalling.
    GpuTimeTable rt;
    GpuTimeRing ring(2, 2);
    uint32_t qb = 99, qe = 99;
    int s0 = ring.begin_frame();
    CHECK(s0 == 0);
    CHECK(ring.add_pass(0x1, GpuPassKind::Render, &qb, &qe) && qb == 0 && qe == 1);
    CHECK(ring.add_pass(0x2, GpuPassKind::Compute, &qb, &qe) && qb == 2 && qe == 3);
    CHECK(!ring.add_pass(0x3, GpuPassKind::Render, &qb, &qe));  // set full
    CHECK(ring.overflow_passes() == 1);
    CHECK(ring.end_frame(s0) == 2);
    int s1 = ring.begin_frame();
    CHECK(s1 == 1);
    ring.end_frame(s1);
    CHECK(ring.begin_frame() == GpuTimeRing::kNoSlot);  // slot 0 still in flight
    CHECK(!ring.add_pass(0x4, GpuPassKind::Render, &qb, &qe));
    // Resolve slot 0: tick period 2 ns; the second pass has an invalid (zero) start.
    const uint64_t ticks[4] = {100, 150, 0, 40};
    ring.complete(s0, ticks, 4, 2.0, rt);
    auto rtop = rt.top(5);
    CHECK(rtop.size() == 1 && rtop[0].hash == 0x1 && rtop[0].stats.total_ns == 100);
    CHECK(has(rt.json(), "\"passes_skipped\": 1"));
    CHECK(has(rt.json(), "\"frames_measured\": 1"));

    // Abandon counts the lost frame.
    GpuTimeTable at;
    GpuTimeRing r2(1, 4);
    int a = r2.begin_frame();
    r2.add_pass(0x9, GpuPassKind::Copy, nullptr, nullptr);
    r2.end_frame(a);
    r2.abandon(a, at);
    CHECK(has(at.json(), "\"frames_skipped\": 1"));
    CHECK(has(at.json(), "\"passes_skipped\": 1"));
    CHECK(r2.begin_frame() == 0);

    if (g_failures == 0) std::printf("test_gpu_time: ok\n");
    return g_failures == 0 ? 0 : 1;
}
