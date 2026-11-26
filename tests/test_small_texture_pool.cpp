#include <cstdint>
#include <cstdio>
#include <vector>

#include "gpu/small_texture_pool.h"
#include "gpu/texture_budget.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_small_texture_pool.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu;

SmallTextureShape shape(uint32_t w, uint32_t h, uint32_t fmt = 28, uint64_t owner = 1) {
    SmallTextureShape s;
    s.owner = owner;
    s.dxgi_format = fmt;
    s.width = w;
    s.height = h;
    return s;
}

void test_hit_miss() {
    SmallTexturePool p;
    CHECK(p.acquire(shape(8, 8), 256) == 0);  // miss
    CHECK(p.release(shape(8, 8), 10, 256, nullptr, nullptr));
    CHECK(p.acquire(shape(16, 8), 512) == 0);  // other extent: miss
    CHECK(p.acquire(shape(8, 8, 29), 256) == 0);  // other format: miss
    CHECK(p.acquire(shape(8, 8, 28, 2), 256) == 0);  // other device: miss
    CHECK(p.acquire(shape(8, 8), 256) == 10);  // hit
    CHECK(p.acquire(shape(8, 8), 256) == 0);   // taken once only
    const SmallTexturePoolStats s = p.stats();
    CHECK(s.hits == 1 && s.misses == 5 && s.bytes_reused == 256 && s.parked_count == 0 && s.parked_bytes == 0);
}

void test_not_poolable() {
    SmallTexturePool p;
    CHECK(!p.release(shape(512, 512), 1, 1u << 20, nullptr, nullptr));
    CHECK(p.acquire(shape(512, 512), 1u << 20) == 0);
    CHECK(p.stats().misses == 0 && p.stats().rejected == 1);  // big textures are not counted as misses
    p.set_enabled(false);
    CHECK(!p.release(shape(8, 8), 2, 256, nullptr, nullptr));
    CHECK(p.acquire(shape(8, 8), 256) == 0);
}

void test_eviction_and_drain() {
    SmallTexturePool p;
    p.set_limits(2, 1000);
    std::vector<uint32_t> ev;
    std::vector<uint64_t> evb;
    CHECK(p.release(shape(1, 1), 1, 100, &ev, &evb));
    CHECK(p.release(shape(2, 1), 2, 100, &ev, &evb));
    CHECK(ev.empty());
    CHECK(p.release(shape(3, 1), 3, 100, &ev, &evb));  // count limit: oldest goes
    CHECK(ev.size() == 1 && ev[0] == 1 && evb[0] == 100);
    ev.clear();
    evb.clear();
    p.set_limits(8, 250);
    CHECK(p.release(shape(4, 1), 4, 200, &ev, &evb));  // byte limit: both older ones go
    CHECK(ev.size() == 2 && ev[0] == 2 && ev[1] == 3);
    CHECK(p.stats().parked_bytes == 200 && p.stats().parked_count == 1 && p.stats().evicted == 3);
    CHECK(p.release(shape(5, 1, 28, 2), 5, 40, nullptr, nullptr));
    std::vector<uint32_t> ids;
    std::vector<uint64_t> bytes;
    p.drain(1, &ids, &bytes);  // only this owner's textures
    CHECK(ids.size() == 1 && ids[0] == 4 && bytes[0] == 200);
    CHECK(p.stats().parked_count == 1 && p.stats().parked_bytes == 40);
}

void test_zero_fill_covers_everything() {
    // 8x8 RGBA8 with 4 mips: 8x8, 4x4, 2x2, 1x1, every texel written.
    const auto w = zero_fill_writes(8, 8, 1, 4, false, 1, 1, 4);
    CHECK(w.size() == 4);
    uint64_t total = 0;
    for (const ZeroWrite& z : w) {
        total += z.bytes;
    }
    CHECK(total == texture_chain_bytes(8, 8, 1, 4, 1, 1, 4));
    CHECK(w[0].width == 8 && w[0].bytes_per_row == 32 && w[3].width == 1 && w[3].height == 1);
    // BC1 2x2 mip is one 4x4 block: the physical extent is rounded up.
    const auto bc = zero_fill_writes(8, 8, 1, 4, false, 4, 4, 8);
    CHECK(bc.size() == 4 && bc[2].width == 4 && bc[2].height == 4 && bc[2].bytes == 8);
    CHECK(bc[3].width == 4 && bc[3].bytes == 8);
    // Array layers keep their count on every mip, 3D depth halves.
    const auto arr = zero_fill_writes(4, 4, 6, 3, false, 1, 1, 4);
    CHECK(arr[0].depth_or_layers == 6 && arr[2].depth_or_layers == 6 && arr[2].bytes == 6 * 4);
    const auto vol = zero_fill_writes(4, 4, 4, 3, true, 1, 1, 4);
    CHECK(vol[0].depth_or_layers == 4 && vol[1].depth_or_layers == 2 && vol[2].depth_or_layers == 1);
    CHECK(zero_fill_writes(4, 4, 1, 1, false, 0, 4, 8).empty());  // unknown block: not poolable
}

void test_budget_counting() {
    // The owner counts a texture once at creation and once at real destruction: parking in the pool
    // keeps it in the budget.
    TextureBudget b;
    SmallTexturePool p;
    p.set_limits(1, 1000);
    b.note_created(100);
    CHECK(p.release(shape(1, 1), 1, 100, nullptr, nullptr));
    CHECK(b.stats().used_bytes == 100);  // parked: still counted
    CHECK(p.acquire(shape(1, 1), 100) == 1);
    CHECK(b.stats().used_bytes == 100);  // reused: no second count
    std::vector<uint32_t> ev;
    std::vector<uint64_t> evb;
    b.note_created(50);
    CHECK(p.release(shape(1, 1), 1, 100, &ev, &evb));
    CHECK(p.release(shape(2, 2), 2, 50, &ev, &evb));
    for (uint64_t x : evb) {
        b.note_destroyed(x);
    }
    CHECK(ev.size() == 1 && b.stats().used_bytes == 50);
}
}  // namespace

int main() {
    test_hit_miss();
    test_not_poolable();
    test_eviction_and_drain();
    test_zero_fill_covers_everything();
    test_budget_counting();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_small_texture_pool: ok");
    return 0;
}
