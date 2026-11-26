#include <cstdint>
#include <cstdio>

#include "gpu/depth_twin.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_depth_twin.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using playgta6::gpu::bridge::DepthTwinTracker;
using Side = DepthTwinTracker::Side;

// A fresh texture has no twin contents: copy in once, then not again until the depth side changes.
void test_copy_in_once() {
    DepthTwinTracker t;
    CHECK(t.side(7) == Side::Stale);
    CHECK(t.needs_copy_in(7));
    t.note_copied_in(7);
    CHECK(t.side(7) == Side::Synced);
    CHECK(!t.needs_copy_in(7));
    CHECK(!t.needs_copy_out(7));
    CHECK(t.needs_copy_in(8));  // other textures are independent
}

// A write dispatch leaves the depth plane behind; the copy back makes both equal again (no copy in).
void test_write_then_copy_out() {
    DepthTwinTracker t;
    t.note_copied_in(7);
    t.note_twin_written(7);
    CHECK(t.needs_copy_out(7));
    CHECK(!t.needs_copy_in(7));
    t.note_copied_out(7);
    CHECK(t.side(7) == Side::Synced);
    CHECK(!t.needs_copy_out(7));
    CHECK(!t.needs_copy_in(7));
}

// A render pass writing the depth plane makes the twin stale; an untracked texture stays untracked.
void test_depth_write_invalidates() {
    DepthTwinTracker t;
    t.note_depth_written(3);  // never twinned: nothing to track
    CHECK(t.tracked() == 0);
    t.note_copied_in(3);
    t.note_depth_written(3);
    CHECK(t.needs_copy_in(3));
    CHECK(!t.needs_copy_out(3));
    t.note_copied_in(3);
    CHECK(!t.needs_copy_in(3));
}

// Pending twin writes are not forgotten by a depth write (the caller flushes them first).
void test_pending_write_kept() {
    DepthTwinTracker t;
    t.note_copied_in(5);
    t.note_twin_written(5);
    t.note_depth_written(5);
    CHECK(t.needs_copy_out(5));
}

void test_forget_and_reset() {
    DepthTwinTracker t;
    t.note_copied_in(1);
    t.note_copied_in(2);
    t.forget(1);
    CHECK(t.needs_copy_in(1) && !t.needs_copy_in(2));
    t.reset();
    CHECK(t.needs_copy_in(2) && t.tracked() == 0);
}
// Subresources (mip x layer) are independent: copying one in leaves the others stale.
void test_subresources_independent() {
    DepthTwinTracker t;
    t.note_copied_in(9, 2, 3);
    CHECK(!t.needs_copy_in(9, 2, 3));
    CHECK(t.needs_copy_in(9, 2, 4));
    CHECK(t.needs_copy_in(9, 3, 3));
    CHECK(t.needs_copy_in(9));  // mip 0 / layer 0 untouched
    t.note_twin_written(9, 2, 3);
    CHECK(t.needs_copy_out(9, 2, 3) && !t.needs_copy_out(9, 2, 4));
    const auto pending = t.pending_out(9);
    CHECK(pending.size() == 1 && pending[0].mip == 2 && pending[0].layer == 3);
    t.note_copied_out(9, 2, 3);
    CHECK(t.pending_out(9).empty());
    CHECK(t.tracked() == 1);
}

// A depth write (render pass view, clear, copy or upload) stales only the subresources it covers;
// a count of 0 is "all remaining".
void test_range_invalidation() {
    DepthTwinTracker t;
    for (uint32_t mip = 0; mip < 3; ++mip) {
        for (uint32_t layer = 0; layer < 4; ++layer) {
            t.note_copied_in(4, mip, layer);
        }
    }
    t.note_depth_written(4, 1, 1, 2, 1);  // mip 1, layer 2
    CHECK(t.needs_copy_in(4, 1, 2));
    CHECK(!t.needs_copy_in(4, 1, 1) && !t.needs_copy_in(4, 0, 2) && !t.needs_copy_in(4, 2, 2));
    t.note_copied_in(4, 1, 2);
    t.note_depth_written(4, 0, 1, 1, 0);  // mip 0, layers 1..end
    CHECK(t.needs_copy_in(4, 0, 1) && t.needs_copy_in(4, 0, 3));
    CHECK(!t.needs_copy_in(4, 0, 0) && !t.needs_copy_in(4, 1, 1));
    t.note_depth_written(4, 0, 0, 0, 0);  // everything
    for (uint32_t mip = 0; mip < 3; ++mip) {
        for (uint32_t layer = 0; layer < 4; ++layer) {
            CHECK(t.needs_copy_in(4, mip, layer));
        }
    }
}

// Pending twin writes survive a range write, and stay listed for the flush before a copy out.
void test_pending_survives_range_write() {
    DepthTwinTracker t;
    t.note_copied_in(6, 0, 0);
    t.note_copied_in(6, 0, 1);
    t.note_twin_written(6, 0, 1);
    t.note_depth_written_all(6);
    CHECK(t.needs_copy_in(6, 0, 0));
    CHECK(t.needs_copy_out(6, 0, 1));
    CHECK(t.pending_out(6).size() == 1);
    CHECK(t.pending_out(77).empty());
    t.forget(6);
    CHECK(t.tracked() == 0);
}

// G79: Range-filtered pending_out and deterministic (layer, mip) ordering.
void test_pending_out_range_and_order() {
    DepthTwinTracker t;
    t.note_twin_written(10, 2, 3);
    t.note_twin_written(10, 0, 1);
    t.note_twin_written(10, 1, 1);
    t.note_twin_written(10, 0, 3);
    const auto all = t.pending_out(10);
    CHECK(all.size() == 4);
    CHECK(all[0].layer == 1 && all[0].mip == 0);
    CHECK(all[1].layer == 1 && all[1].mip == 1);
    CHECK(all[2].layer == 3 && all[2].mip == 0);
    CHECK(all[3].layer == 3 && all[3].mip == 2);

    CHECK(t.has_pending_out(10));
    CHECK(t.has_pending_out(10, 1, 1, 1, 1));
    CHECK(!t.has_pending_out(10, 0, 1, 2, 1));
    const auto slice1 = t.pending_out(10, 0, 0, 1, 1);
    CHECK(slice1.size() == 2 && slice1[0].mip == 0 && slice1[1].mip == 1);
}

// G79: ClearDepthStencilView synchronization (depth vs stencil-only, full vs partial clear).
void test_clear_depth_stencil_sync() {
    DepthTwinTracker t;
    t.note_copied_in(11, 0, 0);
    t.note_copied_in(11, 0, 1);
    t.note_twin_written(11, 0, 1);

    // Stencil-only clear must NOT touch depth twin or SRV state.
    t.note_depth_cleared(11, DepthTwinTracker::kClearFlagStencil, 0, 1, 0, 2, true);
    CHECK(t.side(11, 0, 0) == Side::Synced);
    CHECK(t.side(11, 0, 1) == Side::TwinNewer);
    CHECK(!t.needs_srv_refresh(11, 0, 0));

    // Partial-rect depth clear preserves pending TwinNewer (must flush first), stales Synced.
    t.note_depth_cleared(11, DepthTwinTracker::kClearFlagDepth, 0, 1, 0, 2, false);
    CHECK(t.side(11, 0, 0) == Side::Stale);
    CHECK(t.needs_srv_refresh(11, 0, 0));
    CHECK(t.side(11, 0, 1) == Side::TwinNewer);

    // Full-subresource depth clear overwrites every texel -> even TwinNewer becomes Stale.
    t.note_depth_cleared(11, DepthTwinTracker::kClearFlagDepth, 0, 1, 1, 1, true);
    CHECK(t.side(11, 0, 1) == Side::Stale);
    CHECK(!t.needs_copy_out(11, 0, 1));
    CHECK(t.needs_srv_refresh(11, 0, 1));
}

// G79: Depth-as-SRV twin tracking & synchronization across mip levels and array layers
// (SSAO/SSR/decals/shadows reading depth after render-pass writes, clears, copies, or compute UAV writes).
void test_depth_as_srv_sync_plan() {
    DepthTwinTracker t;
    // Layer 0: compute wrote twin (TwinNewer).
    // Layer 1: already synced.
    // Layer 2: depth written by render pass (Stale).
    t.note_copied_in(20, 0, 0);
    t.note_twin_written(20, 0, 0);
    t.note_copied_in(20, 0, 1);
    t.note_copied_in(20, 0, 2);
    t.note_depth_written(20, 0, 1, 2, 1);

    const auto plan = t.plan_srv_read(20, 0, 1, 0, 3);
    CHECK(plan.any());
    CHECK(plan.flush_twin_to_depth.size() == 1 && plan.flush_twin_to_depth[0].layer == 0 &&
          plan.flush_twin_to_depth[0].mip == 0);
    CHECK(plan.refresh_depth_to_srv.size() == 1 && plan.refresh_depth_to_srv[0].layer == 2 &&
          plan.refresh_depth_to_srv[0].mip == 0);

    t.apply_srv_sync(20, plan);
    CHECK(!t.needs_copy_out(20, 0, 0));
    CHECK(!t.needs_srv_refresh(20, 0, 0));
    CHECK(!t.needs_srv_refresh(20, 0, 1));
    CHECK(!t.needs_srv_refresh(20, 0, 2));

    // Re-planning with no intervening depth write needs zero work.
    const auto plan2 = t.plan_srv_read(20, 0, 1, 0, 3);
    CHECK(!plan2.any());

    // Copy or resolve into layer 1 stales only layer 1 and bumps write generation.
    const uint64_t g0 = t.write_generation(20, 0, 1);
    t.note_depth_copied(20, 0, 1, 1, 1);
    CHECK(t.write_generation(20, 0, 1) > g0);
    const auto plan3 = t.plan_srv_read(20, 0, 0, 0, 0);  // wildcard range
    CHECK(plan3.flush_twin_to_depth.empty());
    CHECK(plan3.refresh_depth_to_srv.size() == 1 && plan3.refresh_depth_to_srv[0].layer == 1);
    t.apply_srv_sync(20, plan3);
    CHECK(!t.plan_srv_read(20, 0, 0, 0, 0).any());

    CHECK(t.tracked_srv() >= 3);
    t.forget(20);
    CHECK(t.tracked() == 0 && t.tracked_srv() == 0);
}
}  // namespace

int main() {
    test_copy_in_once();
    test_write_then_copy_out();
    test_depth_write_invalidates();
    test_pending_write_kept();
    test_forget_and_reset();
    test_subresources_independent();
    test_range_invalidation();
    test_pending_survives_range_write();
    test_pending_out_range_and_order();
    test_clear_depth_stencil_sync();
    test_depth_as_srv_sync_plan();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_depth_twin: ok");
    return 0;
}
