#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <unordered_set>
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

void test_lifo_per_shape_interleaved() {
    SmallTexturePool p;
    const SmallTextureShape a = shape(8, 8, 28, 1);
    const SmallTextureShape b = shape(16, 16, 28, 1);
    CHECK(p.distinct_shapes() == 0);
    CHECK(p.release(a, 101, 256, nullptr, nullptr));
    CHECK(p.distinct_shapes() == 1);
    CHECK(p.release(b, 201, 512, nullptr, nullptr));
    CHECK(p.distinct_shapes() == 2);
    CHECK(p.release(a, 102, 256, nullptr, nullptr));
    CHECK(p.release(b, 202, 512, nullptr, nullptr));
    CHECK(p.release(a, 103, 256, nullptr, nullptr));
    CHECK(p.distinct_shapes() == 2);
    CHECK(p.stats().parked_count == 5);

    // LIFO within shape A despite interleaved B entries.
    CHECK(p.acquire(a, 256) == 103);
    CHECK(p.acquire(a, 256) == 102);
    CHECK(p.distinct_shapes() == 2);
    CHECK(p.acquire(a, 256) == 101);
    CHECK(p.distinct_shapes() == 1);
    CHECK(p.acquire(a, 256) == 0);

    // LIFO within shape B.
    CHECK(p.acquire(b, 512) == 202);
    CHECK(p.acquire(b, 512) == 201);
    CHECK(p.distinct_shapes() == 0);
    CHECK(p.stats().parked_count == 0 && p.stats().parked_bytes == 0);
}

void test_global_oldest_first_eviction_mixed_shapes() {
    SmallTexturePool p;
    p.set_limits(3, 10000);
    const SmallTextureShape a = shape(4, 4, 28, 1);
    const SmallTextureShape b = shape(8, 4, 28, 1);
    const SmallTextureShape c = shape(12, 4, 28, 1);
    const SmallTextureShape d = shape(16, 4, 28, 1);

    std::vector<uint32_t> ev;
    std::vector<uint64_t> evb;
    CHECK(p.release(a, 1, 100, &ev, &evb));  // global: [A1]
    CHECK(p.release(b, 2, 100, &ev, &evb));  // global: [A1, B2]
    CHECK(p.release(a, 3, 100, &ev, &evb));  // global: [A1, B2, A3]
    CHECK(p.distinct_shapes() == 2);
    CHECK(ev.empty());

    // Evicts oldest globally (A1), leaving A3 still intact in bucket A.
    CHECK(p.release(c, 4, 100, &ev, &evb));  // global: [B2, A3, C4]
    CHECK(ev.size() == 1 && ev[0] == 1 && evb[0] == 100);
    CHECK(p.distinct_shapes() == 3);

    // Evicts oldest globally (B2), removing bucket B entirely.
    ev.clear();
    evb.clear();
    CHECK(p.release(d, 5, 100, &ev, &evb));  // global: [A3, C4, D5]
    CHECK(ev.size() == 1 && ev[0] == 2 && evb[0] == 100);
    CHECK(p.distinct_shapes() == 3);
    CHECK(p.acquire(b, 100) == 0);
    CHECK(p.acquire(a, 100) == 3);
    CHECK(p.distinct_shapes() == 2);
}

void test_trim_and_distinct_shapes() {
    SmallTexturePool p;
    CHECK(p.release(shape(1, 1), 10, 100, nullptr, nullptr));
    CHECK(p.release(shape(2, 1), 20, 100, nullptr, nullptr));
    CHECK(p.release(shape(1, 1), 30, 100, nullptr, nullptr));
    CHECK(p.release(shape(3, 1), 40, 100, nullptr, nullptr));
    CHECK(p.release(shape(2, 1), 50, 100, nullptr, nullptr));
    CHECK(p.distinct_shapes() == 3 && p.stats().parked_count == 5 && p.stats().parked_bytes == 500);

    std::vector<uint32_t> ev;
    std::vector<uint64_t> evb;
    // No-op trim when already within limits.
    CHECK(p.trim(5, 500, &ev, &evb) == 0 && ev.empty());
    // Trim by count: oldest two (10, 20) evicted.
    CHECK(p.trim(3, 1000, &ev, &evb) == 2);
    CHECK(ev == (std::vector<uint32_t>{10, 20}) && evb == (std::vector<uint64_t>{100, 100}));
    CHECK(p.distinct_shapes() == 3 && p.stats().parked_count == 3 && p.stats().parked_bytes == 300);

    // Trim by bytes: evict oldest two (30, 40) to get <= 150 bytes.
    ev.clear();
    evb.clear();
    CHECK(p.trim(10, 150, &ev, &evb) == 2);
    CHECK(ev == (std::vector<uint32_t>{30, 40}));
    CHECK(p.distinct_shapes() == 1 && p.stats().parked_count == 1 && p.stats().parked_bytes == 100);

    // Trim to zero.
    ev.clear();
    evb.clear();
    CHECK(p.trim(0, 0, &ev, &evb) == 1);
    CHECK(ev == (std::vector<uint32_t>{50}));
    CHECK(p.distinct_shapes() == 0 && p.stats().parked_count == 0 && p.stats().parked_bytes == 0);
    CHECK(p.stats().evicted == 5);
}

void test_disabled_fast_path() {
    SmallTexturePool p;
    const SmallTextureShape s = shape(8, 8);
    CHECK(p.release(s, 77, 256, nullptr, nullptr));
    p.set_enabled(false);
    // Disabled acquire must return 0 without incrementing misses.
    CHECK(p.acquire(s, 256) == 0);
    CHECK(p.acquire(shape(16, 16), 512) == 0);
    CHECK(p.stats().misses == 0 && p.stats().hits == 0 && p.stats().parked_count == 1);
    CHECK(!p.release(s, 78, 256, nullptr, nullptr));
    CHECK(p.stats().rejected == 1);
    // Re-enabling makes parked entries reusable again.
    p.set_enabled(true);
    CHECK(p.acquire(s, 256) == 77);
    CHECK(p.stats().hits == 1 && p.stats().misses == 0);
}

void test_address_counters() {
    SmallTexturePool p;
    p.note_address_miss(false);
    p.note_address_miss(true);
    p.note_address_eviction();
    const SmallTexturePoolStats st = p.stats();
    CHECK(st.address_misses == 2 && st.address_recreates == 1 && st.address_evictions == 1);
    // The address counters are separate from the pool's own hit/miss counters.
    CHECK(st.hits == 0 && st.misses == 0 && st.evicted == 0);
}

void test_stress_2000_ops_matches_reference() {
    struct RefEntry {
        SmallTextureShape shape;
        uint32_t id;
        uint64_t bytes;
    };
    SmallTexturePool pool;
    std::deque<RefEntry> ref;
    uint32_t max_count = 32;
    uint64_t max_bytes = 32768;
    pool.set_limits(max_count, max_bytes);

    uint64_t ref_parked_bytes = 0;
    uint64_t ref_hits = 0;
    uint64_t ref_misses = 0;
    uint64_t ref_evicted = 0;
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    auto next_u32 = [&]() -> uint32_t {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        return static_cast<uint32_t>((rng * 0x2545F4914F6CDD1Dull) >> 32);
    };

    uint32_t next_id = 1;
    for (int step = 0; step < 2500; ++step) {
        const uint32_t op = next_u32() % 100;
        const uint64_t owner = 1 + (next_u32() % 3);
        const uint32_t w = 4u << (next_u32() % 4);
        const uint32_t h = 4u << (next_u32() % 3);
        const uint32_t fmt = 28u + (next_u32() % 3);
        SmallTextureShape sh = shape(w, h, fmt, owner);
        sh.mip_levels = 1u + (next_u32() % 2);
        sh.storage = (next_u32() % 4) == 0;
        const uint64_t bytes = uint64_t{w} * h * 4u;

        if (op < 48) {
            // Release
            std::vector<uint32_t> ev, ref_ev;
            std::vector<uint64_t> evb, ref_evb;
            const uint32_t id = next_id++;
            const bool ok = pool.release(sh, id, bytes, &ev, &evb);
            while (!ref.empty() && (ref.size() >= max_count || ref_parked_bytes + bytes > max_bytes)) {
                ref_ev.push_back(ref.front().id);
                ref_evb.push_back(ref.front().bytes);
                ref_parked_bytes -= ref.front().bytes;
                ref.pop_front();
                ++ref_evicted;
            }
            ref.push_back(RefEntry{sh, id, bytes});
            ref_parked_bytes += bytes;
            CHECK(ok);
            CHECK(ev == ref_ev && evb == ref_evb);
        } else if (op < 90) {
            // Acquire
            const uint32_t got = pool.acquire(sh, bytes);
            uint32_t ref_got = 0;
            for (auto it = ref.rbegin(); it != ref.rend(); ++it) {
                if (it->shape == sh) {
                    ref_got = it->id;
                    ref_parked_bytes -= it->bytes;
                    ref.erase(std::next(it).base());
                    ++ref_hits;
                    break;
                }
            }
            if (ref_got == 0) ++ref_misses;
            CHECK(got == ref_got);
        } else if (op < 95) {
            // Trim
            const uint32_t tc = next_u32() % 20;
            const uint64_t tb = uint64_t{next_u32() % 16384};
            std::vector<uint32_t> ev, ref_ev;
            std::vector<uint64_t> evb, ref_evb;
            const size_t n = pool.trim(tc, tb, &ev, &evb);
            while (!ref.empty() && (ref.size() > tc || ref_parked_bytes > tb)) {
                ref_ev.push_back(ref.front().id);
                ref_evb.push_back(ref.front().bytes);
                ref_parked_bytes -= ref.front().bytes;
                ref.pop_front();
                ++ref_evicted;
            }
            CHECK(n == ref_ev.size());
            CHECK(ev == ref_ev && evb == ref_evb);
        } else {
            // Drain one owner
            std::vector<uint32_t> ids, ref_ids;
            std::vector<uint64_t> bts, ref_bts;
            pool.drain(owner, &ids, &bts);
            for (auto it = ref.begin(); it != ref.end();) {
                if (it->shape.owner == owner) {
                    ref_ids.push_back(it->id);
                    ref_bts.push_back(it->bytes);
                    ref_parked_bytes -= it->bytes;
                    it = ref.erase(it);
                } else {
                    ++it;
                }
            }
            CHECK(ids == ref_ids && bts == ref_bts);
        }

        if ((step % 50) == 49) {
            const SmallTexturePoolStats st = pool.stats();
            std::unordered_set<SmallTextureShape, SmallTextureShapeHash> distinct;
            for (const RefEntry& e : ref) distinct.insert(e.shape);
            CHECK(st.parked_count == ref.size());
            CHECK(st.parked_bytes == ref_parked_bytes);
            CHECK(st.hits == ref_hits);
            CHECK(st.misses == ref_misses);
            CHECK(st.evicted == ref_evicted);
            CHECK(pool.distinct_shapes() == distinct.size());
        }
    }
}
// The large pool takes chains over 64 KiB up to its own cap, with its own limits, and only when the
// consumer clears on the GPU; texture_pool_for routes a chain to the right pool.
void test_large_pool() {
    SmallTexturePool p(kSmallTextureMaxBytes, kLargeTextureMaxBytes, 2, 3u << 20);
    CHECK(!p.accepts(kSmallTextureMaxBytes) && p.accepts(kSmallTextureMaxBytes + 1) && p.accepts(kLargeTextureMaxBytes));
    CHECK(!p.accepts(kLargeTextureMaxBytes + 1) && !p.accepts(0));
    CHECK(!p.release(shape(8, 8), 1, 256, nullptr, nullptr));  // small chains stay in the small pool
    CHECK(p.stats().rejected == 1);
    CHECK(p.acquire(shape(512, 512), 1u << 20) == 0 && p.stats().misses == 1);
    CHECK(p.release(shape(512, 512), 7, 1u << 20, nullptr, nullptr));
    CHECK(p.acquire(shape(512, 512, 29), 1u << 20) == 0);  // other format
    CHECK(p.acquire(shape(512, 512), 1u << 20) == 7);      // same shape: reused
    // Own byte limit: a third 1 MiB texture pushes out the oldest.
    std::vector<uint32_t> ev;
    std::vector<uint64_t> evb;
    CHECK(p.release(shape(512, 512), 1, 1u << 20, &ev, &evb));
    CHECK(p.release(shape(256, 1024), 2, 1u << 20, &ev, &evb));
    CHECK(ev.empty());
    CHECK(p.release(shape(1024, 256), 3, 1u << 20, &ev, &evb));  // count limit 2
    CHECK(ev.size() == 1 && ev[0] == 1 && evb[0] == (1u << 20));
    CHECK(!p.release(shape(2048, 2048), 4, 4u << 20, &ev, &evb));  // over the pool's byte limit
    // reset() restores this pool's own limits, not the small pool's.
    p.set_limits(0, 0);
    p.reset();
    CHECK(p.release(shape(512, 512), 5, 2u << 20, nullptr, nullptr));
    CHECK(p.release(shape(512, 512), 6, 1u << 20, nullptr, nullptr));
    CHECK(p.stats().parked_count == 2 && p.stats().parked_bytes == (3u << 20));

    // Routing: without the GPU clear every chain goes to the small pool (which declines big ones).
    set_gpu_texture_clear(false);
    CHECK(&texture_pool_for(256) == &small_texture_pool());
    CHECK(&texture_pool_for(1u << 20) == &small_texture_pool());
    set_gpu_texture_clear(true);
    CHECK(gpu_texture_clear());
    CHECK(&texture_pool_for(256) == &small_texture_pool());
    CHECK(&texture_pool_for(kSmallTextureMaxBytes) == &small_texture_pool());
    CHECK(&texture_pool_for(kSmallTextureMaxBytes + 1) == &large_texture_pool());
    CHECK(&texture_pool_for(kLargeTextureMaxBytes + 1) == &small_texture_pool());
    set_gpu_texture_clear(false);
}
}  // namespace

int main() {
    test_large_pool();
    test_hit_miss();
    test_not_poolable();
    test_eviction_and_drain();
    test_zero_fill_covers_everything();
    test_budget_counting();
    test_lifo_per_shape_interleaved();
    test_global_oldest_first_eviction_mixed_shapes();
    test_trim_and_distinct_shapes();
    test_disabled_fast_path();
    test_address_counters();
    test_stress_2000_ops_matches_reference();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_small_texture_pool: ok");
    return 0;
}
