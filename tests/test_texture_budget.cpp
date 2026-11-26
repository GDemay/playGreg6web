#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "gpu/texture_budget.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_texture_budget.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu;

void test_cap_floor() {
    CHECK(cap_floor_for(4096, 4096, 13, 0) == 0);      // no cap
    CHECK(cap_floor_for(4096, 2048, 13, 1024) == 2);   // 4096 -> 1024 drops two levels
    CHECK(cap_floor_for(4096, 4096, 13, 256) == 4);
    CHECK(cap_floor_for(512, 512, 10, 1024) == 0);     // already fits
    CHECK(cap_floor_for(300, 100, 9, 256) == 1);       // the larger side decides
    CHECK(cap_floor_for(4096, 4096, 2, 256) == 1);     // never drops the last level
    CHECK(cap_floor_for(4096, 4096, 1, 256) == 0);
}

void test_chain_bytes() {
    // BC1 (4x4 blocks, 8 B): 256x256 mip 0 = 64*64*8, mip 1 = 32*32*8, ...
    CHECK(texture_chain_bytes(256, 256, 1, 1, 4, 4, 8) == 64u * 64u * 8u);
    CHECK(texture_chain_bytes(256, 256, 1, 2, 4, 4, 8, 1) == 32u * 32u * 8u);
    CHECK(texture_chain_bytes(8, 8, 1, 4, 0, 4, 8) == 0);
}

void test_pressure_tiers() {
    constexpr uint64_t b = 1000;
    CHECK(cap_for_pressure(0, 0, 0) == 0);              // no budget, no base cap
    CHECK(cap_for_pressure(999, 0, 1024) == 1024);      // base cap only
    CHECK(cap_for_pressure(690, b, 0) == 0);            // below 70 percent
    CHECK(cap_for_pressure(700, b, 0) == kTextureCapMedium);
    CHECK(cap_for_pressure(840, b, 0) == kTextureCapMedium);
    CHECK(cap_for_pressure(850, b, 0) == kTextureCapHigh);
    CHECK(cap_for_pressure(700, b, 256) == 256);        // the tighter of base and tier wins
    CHECK(cap_for_pressure(850, b, 512) == kTextureCapHigh);
    CHECK(cap_for_pressure(100, b, 1024) == 1024);
    CHECK(cap_for_pressure(UINT64_MAX / 10 * 8, UINT64_MAX, 0) == kTextureCapMedium);  // no overflow
}

void test_tracker() {
    TextureBudget t;
    t.set_budget_bytes(1000);
    CHECK(t.current_cap() == 0);
    t.note_created(720);
    CHECK(t.current_cap() == kTextureCapMedium);
    t.note_created(200);
    CHECK(t.current_cap() == kTextureCapHigh);
    CHECK(t.stats().used_bytes == 920 && t.stats().peak_used_bytes == 920);
    t.note_destroyed(500);  // memory frees: the cap loosens
    CHECK(t.current_cap() == 0);
    CHECK(t.stats().cap_changes == 3 && t.stats().peak_used_bytes == 920);
    t.note_destroyed(100000);  // over-release clamps at zero
    CHECK(t.stats().used_bytes == 0);
    t.note_capped(2, 4096);
    t.note_capped(1, 100);
    t.note_restored();
    const TextureBudgetStats s = t.stats();
    CHECK(s.textures_capped == 2 && s.mips_dropped == 3 && s.bytes_not_uploaded == 4196 && s.textures_restored == 1);
    t.set_base_cap(1024);
    CHECK(t.current_cap() == 1024 && t.stats().base_cap == 1024);
    t.reset();
    CHECK(t.stats().cap_changes == 0 && t.stats().budget_bytes == 0);
}

void test_host_realloc() {
    TextureBudget t;
    t.set_budget_bytes(1000);
    t.note_created(900);  // 90%: cap 256
    CHECK(t.current_cap() == kTextureCapHigh);
    // A 1024x1024 BC1-like texture (4x4 blocks of 8 bytes): shrinking it by 2 levels saves its top two.
    const uint64_t full = texture_chain_bytes(1024, 1024, 1, 11, 4, 4, 8);
    const uint64_t capped = texture_chain_bytes(1024, 1024, 1, 11, 4, 4, 8, 2);
    CHECK(full - capped == texture_chain_bytes(1024, 1024, 1, 2, 4, 4, 8));
    // The capped host texture has the chain of a 256x256 texture with 9 levels.
    CHECK(capped == texture_chain_bytes(256, 256, 1, 9, 4, 4, 8));
    t.note_host_realloc(static_cast<int64_t>(300));
    CHECK(t.stats().used_bytes == 600 && t.stats().host_bytes_saved == 300 && t.stats().host_reallocs == 1);
    CHECK(t.stats().peak_used_bytes == 900);
    CHECK(t.current_cap() == 0);
    // Restore hysteresis: growing back by 300 would return to 90%, so it does not fit a 0-level floor.
    CHECK(!t.restore_fits(1024, 1024, 11, 0, 300));
    CHECK(t.cap_with_extra(300) == kTextureCapHigh);
    CHECK(t.restore_fits(1024, 1024, 11, 0, 50));  // 65% stays below the medium tier
    CHECK(t.restore_fits(1024, 1024, 11, 2, 300));  // the capped size itself is allowed
    t.note_host_realloc(-300);  // restored
    CHECK(t.stats().used_bytes == 900 && t.stats().host_bytes_saved == 0 && t.stats().host_reallocs == 2);
    // A shrunk texture destroyed: released bytes are counted back, then its full size leaves.
    t.note_host_realloc(300);
    t.note_shrunk_released(300);
    t.note_destroyed(300 + 0);
    CHECK(t.stats().used_bytes == 600 && t.stats().host_bytes_saved == 0);
    t.reset();
    CHECK(t.stats().host_reallocs == 0 && t.stats().host_bytes_saved == 0);
}

void test_3d_and_array_chain_bytes() {
    // 2D cubemap (6 constant array layers) vs 3D volume (depth halves per mip):
    const uint64_t face = texture_chain_bytes(256, 256, 1, 9, 4, 4, 16, 0, 1);
    const uint64_t cube = texture_chain_bytes(256, 256, 1, 9, 4, 4, 16, 0, 6);
    CHECK(cube == face * 6u);
    const uint64_t vol = texture_chain_bytes(32, 32, 32, 6, 1, 1, 4, 0, 1);
    const uint64_t expected_vol = (32u * 32u * 32u + 16u * 16u * 16u + 8u * 8u * 8u +
                                   4u * 4u * 4u + 2u * 2u * 2u + 1u) * 4u;
    CHECK(vol == expected_vol);
    CHECK(texture_chain_bytes(32, 32, 1, 4, 1, 1, 4, 0, 0) == 0u);
    CHECK(cap_floor_for_3d(128, 128, 512, 10, 256) == 1u);  // depth decides
    CHECK(cap_floor_for_3d(128, 128, 128, 8, 256) == 0u);
}

void test_pressure_tiers_and_reclaim_targets() {
    constexpr uint64_t b = 1000;
    CHECK(pressure_tier_for(0, 0) == BudgetPressureTier::Normal);
    CHECK(pressure_tier_for(699, b) == BudgetPressureTier::Normal);
    CHECK(pressure_tier_for(700, b) == BudgetPressureTier::Medium);
    CHECK(pressure_tier_for(799, b) == BudgetPressureTier::Medium);
    CHECK(pressure_tier_for(800, b) == BudgetPressureTier::Reclaim);
    CHECK(pressure_tier_for(849, b) == BudgetPressureTier::Reclaim);
    CHECK(pressure_tier_for(850, b) == BudgetPressureTier::High);
    CHECK(pressure_tier_for(919, b) == BudgetPressureTier::High);
    CHECK(pressure_tier_for(920, b) == BudgetPressureTier::Critical);

    // Reclaim target with hysteresis:
    CHECK(reclaim_target_bytes(790, b, /*reclaim_active=*/false) == 0u);
    CHECK(reclaim_target_bytes(800, b, /*reclaim_active=*/false) == 120u);  // 800 - 680
    // While reclaim_active is latched, usage between 68% and 80% still reclaims down to 68%.
    CHECK(reclaim_target_bytes(750, b, /*reclaim_active=*/true) == 70u);    // 750 - 680
    CHECK(reclaim_target_bytes(680, b, /*reclaim_active=*/true) == 0u);
    CHECK(reclaim_target_bytes(600, b, /*reclaim_active=*/true) == 0u);
    // Incoming allocation lookahead triggers reclamation before exceeding 80%.
    CHECK(reclaim_target_bytes(750, b, /*reclaim_active=*/false, /*incoming=*/60) == 130u);  // 810 - 680
    CHECK(reclaim_target_bytes(UINT64_MAX - 10, UINT64_MAX, false, 100) > 0u);
}

void test_lru_ordering_pinning_and_age_protection() {
    TextureBudget t;
    t.set_budget_bytes(1000);

    // Register 4 textures at frame 1 (ids 1..4, 200B each = 800B -> 80% Reclaim tier).
    // Pin id 1 (back buffer), mark id 2 non-evictable, leave ids 3 and 4 evictable.
    t.register_texture(1, 200, /*frame=*/1, /*pinned=*/true, /*evictable=*/true);
    t.register_texture(2, 200, /*frame=*/1, /*pinned=*/false, /*evictable=*/false);
    t.register_texture(3, 200, /*frame=*/1, /*pinned=*/false, /*evictable=*/true);
    t.register_texture(4, 200, /*frame=*/1, /*pinned=*/false, /*evictable=*/true);
    t.note_created(800);

    TextureBudgetStats s = t.stats();
    CHECK(s.lru_tracked_textures == 4u && s.lru_tracked_bytes == 800u && s.lru_pinned_textures == 1u);
    CHECK(s.pressure_tier == static_cast<uint8_t>(BudgetPressureTier::Reclaim));

    // Touch id 3 at frame 2 so LRU order of evictable entries is [4 (older), 3 (newer)].
    t.touch_texture(3, /*frame=*/2);

    // At frame 2 with default min_age_frames = 2: id 4 has age 1 (< 2) and id 3 has age 0 -> both protected!
    auto cands_f2 = t.select_eviction_candidates(/*current_frame=*/2);
    CHECK(cands_f2.empty());
    CHECK(t.is_reclaim_active());

    // At frame 3 with min_age_frames = 2: id 4 has age 2 (eligible!), id 3 has age 1 (protected),
    // id 1 pinned, id 2 non-evictable -> only id 4 selected!
    auto cands_f3 = t.select_eviction_candidates(/*current_frame=*/3);
    CHECK(cands_f3.size() == 1u && cands_f3[0].resource_id == 4u && cands_f3[0].bytes == 200u);

    // Push into Critical tier (>= 92%): id 3 (age 1, NOT current frame 3) becomes eligible in Critical,
    // while a current-frame texture (id 5 at frame 3) and pinned id 1 remain protected!
    t.register_texture(5, 150, /*frame=*/3, /*pinned=*/false, /*evictable=*/true);
    t.note_created(150);  // 950 / 1000 = 95% Critical
    CHECK(t.current_pressure_tier() == BudgetPressureTier::Critical);
    auto cands_crit = t.select_eviction_candidates(/*current_frame=*/3, /*target_bytes=*/0, 64, 2);
    // Target is 950 - 680 = 270 -> selects id 4 (200B) + id 3 (200B) = 400B >= 270B, never id 1, 2, or 5!
    CHECK(cands_crit.size() == 2u);
    CHECK(cands_crit[0].resource_id == 4u);
    CHECK(cands_crit[1].resource_id == 3u);

    // Complete evictions and verify hysteresis clears once usage drops below 680B.
    t.note_destroyed(200);
    t.note_lru_evicted(4, 200);
    CHECK(t.is_reclaim_active());  // 750 > 680 -> hysteresis stays latched!
    CHECK(t.bytes_to_reclaim() == 70u);

    t.note_destroyed(200);
    t.note_lru_evicted(3, 200);
    CHECK(!t.is_reclaim_active());  // 550 <= 680 -> hysteresis cleared!
    CHECK(t.bytes_to_reclaim() == 0u);

    s = t.stats();
    CHECK(s.lru_tracked_textures == 3u && s.lru_tracked_bytes == 550u);
    CHECK(s.lru_evictions == 2u && s.lru_bytes_reclaimed == 400u);

    // Dynamic pin / unpin and byte size update.
    t.set_texture_pinned(5, true);
    CHECK(t.stats().lru_pinned_textures == 2u);
    t.update_texture_bytes(5, 100);
    CHECK(t.stats().lru_tracked_bytes == 500u);
    t.set_texture_pinned(5, false);
    CHECK(t.stats().lru_pinned_textures == 1u);
    t.unregister_texture(5);
    CHECK(t.stats().lru_tracked_textures == 2u && t.stats().lru_tracked_bytes == 400u);

    t.reset();
    s = t.stats();
    CHECK(s.lru_tracked_textures == 0u && s.lru_tracked_bytes == 0u && s.lru_evictions == 0u &&
          s.lru_bytes_reclaimed == 0u && s.lru_pinned_textures == 0u && !s.reclaim_active);
}

void test_streaming_gigabytes_hysteresis_and_concurrency() {
    // Simulate streaming 4 GiB of 3D open-world textures within a 512 MiB Chrome VRAM budget
    // across 120 frames, verifying usage stays bounded without thrashing active-frame textures.
    TextureBudget t;
    constexpr uint64_t kBudget = 512ull << 20;
    constexpr uint64_t kTexSize = 16ull << 20;  // 16 MiB per streamed 3D/BC7 texture
    t.set_budget_bytes(kBudget);
    // Keep 2 pinned render targets (32 MiB).
    t.register_texture(1, 16ull << 20, 0, /*pinned=*/true, /*evictable=*/false);
    t.register_texture(2, 16ull << 20, 0, /*pinned=*/true, /*evictable=*/false);
    t.note_created(32ull << 20);

    uint32_t next_id = 100;
    for (uint64_t frame = 1; frame <= 120; ++frame) {
        // Stream 2 new 16 MiB textures per frame (3.84 GiB total streamed).
        for (int k = 0; k < 2; ++k) {
            const uint64_t goal = t.bytes_to_reclaim(kTexSize);
            if (goal > 0) {
                for (const auto& c : t.select_eviction_candidates(frame, goal)) {
                    t.note_destroyed(c.bytes);
                    t.note_lru_evicted(c.resource_id, c.bytes);
                }
            }
            const uint32_t id = next_id++;
            t.register_texture(id, kTexSize, frame, /*pinned=*/false, /*evictable=*/true);
            t.note_created(kTexSize);
        }
        // Usage must stay well below the 512 MiB hard ceiling.
        CHECK(t.stats().used_bytes <= kBudget);
    }
    const TextureBudgetStats st = t.stats();
    CHECK(st.lru_evictions >= 200u);
    CHECK(st.lru_bytes_reclaimed >= (3ull << 30));
    CHECK(st.lru_pinned_textures == 2u);

    // Multi-threaded stress test: concurrent register, touch, select, unregister, stats.
    t.reset();
    t.set_budget_bytes(64ull << 20);
    std::vector<std::thread> workers;
    for (uint32_t w = 0; w < 4; ++w) {
        workers.emplace_back([&, w] {
            const uint32_t base = 10000u + w * 200u;
            for (uint32_t i = 0; i < 100u; ++i) {
                t.register_texture(base + i, 64u << 10, i, (i % 16u) == 0u, true);
                t.touch_texture(base + (i / 2u), i + 1u);
                if ((i % 10u) == 0u) {
                    (void)t.select_eviction_candidates(i + 5u, 128u << 10, 8u, 2u);
                    (void)t.stats();
                }
            }
            for (uint32_t i = 0; i < 100u; ++i) {
                t.unregister_texture(base + i);
            }
        });
    }
    for (auto& th : workers) {
        th.join();
    }
    CHECK(t.stats().lru_tracked_textures == 0u && t.stats().lru_tracked_bytes == 0u &&
          t.stats().lru_pinned_textures == 0u);
}
}  // namespace

int main() {
    test_cap_floor();
    test_chain_bytes();
    test_pressure_tiers();
    test_tracker();
    test_host_realloc();
    test_3d_and_array_chain_bytes();
    test_pressure_tiers_and_reclaim_targets();
    test_lru_ordering_pinning_and_age_protection();
    test_streaming_gigabytes_hysteresis_and_concurrency();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_texture_budget: ok");
    return 0;
}
