// CPU twin of the BVH node test against the shared vectors (contract C5), prelude contract,
// plus G80 tests for 3D RT BVH BLAS/TLAS build, in-place refit, 3x4 instance packing, bounded
// scratch recycling, and group-3 arena traversal.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gpu/rt_arena.h"
#include "gpu/rt_bvh.h"
#include "rt_bvh_vectors.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const std::string& what, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_rt_bvh.cpp:%d]: %s\n", line, what.c_str());
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
using namespace playgta6::gpu::rt_bvh;
namespace rt_arena = playgta6::gpu::rt_arena;

bool near_f(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

// Reference brute-force ray-triangle closest hit using rt_bvh::intersect() directly on packed leaves.
TraceHit brute_force_blas(const std::vector<Triangle>& tris, Vec3 origin, Vec3 dir, float t_min, float t_max) {
    TraceHit best{};
    best.t = t_max;
    uint32_t desc[4];
    make_bvh_desc(0x1000, false, desc);
    const Vec3 inv = {rt_bvh_vectors::inv_of(dir.x), rt_bvh_vectors::inv_of(dir.y), rt_bvh_vectors::inv_of(dir.z)};
    uint32_t node_words[16];
    for (uint32_t i = 0; i < tris.size(); ++i) {
        pack_triangle_node(tris[i], node_words);
        const auto load = [&](uint64_t a) -> uint32_t {
            const uint64_t off = (a - 0x1000ull) / 4ull;
            return off < 16u ? node_words[off] : 0u;
        };
        const Result r = intersect(desc, 0u, best.t, origin, dir, inv, load);
        float t_num = 0.0f, t_den = 0.0f, u_num = 0.0f, v_num = 0.0f;
        std::memcpy(&t_num, &r.w[0], 4);
        std::memcpy(&t_den, &r.w[1], 4);
        std::memcpy(&u_num, &r.w[2], 4);
        std::memcpy(&v_num, &r.w[3], 4);
        if (std::isinf(t_num) || t_den == 0.0f) continue;
        const float t = t_num / t_den;
        if (t >= t_min && t <= best.t) {
            best.hit = true;
            best.t = t;
            best.u = u_num / t_den;
            best.v = v_num / t_den;
            best.front_face = t_den > 0.0f;
            best.primitive_index = tris[i].primitive_id;
        }
    }
    return best;
}

std::vector<Triangle> make_test_scene(uint32_t grid_n, float z_offset) {
    std::vector<Triangle> tris;
    uint32_t pid = 0;
    for (uint32_t gx = 0; gx < grid_n; ++gx) {
        for (uint32_t gy = 0; gy < grid_n; ++gy) {
            const float x0 = static_cast<float>(gx) * 2.0f - static_cast<float>(grid_n);
            const float y0 = static_cast<float>(gy) * 2.0f - static_cast<float>(grid_n);
            const float z = z_offset + 0.25f * static_cast<float>((gx + gy) % 5u);
            Triangle t0{};
            t0.v0 = {x0, y0, z};
            t0.v1 = {x0, y0 + 1.5f, z + 0.1f};
            t0.v2 = {x0 + 1.5f, y0, z - 0.1f};
            t0.primitive_id = pid++;
            t0.geometry_id = 1u;
            tris.push_back(t0);

            Triangle t1{};
            t1.v0 = {x0 + 1.5f, y0, z - 0.1f};
            t1.v1 = {x0, y0 + 1.5f, z + 0.1f};
            t1.v2 = {x0 + 1.5f, y0 + 1.5f, z + 0.2f};
            t1.primitive_id = pid++;
            t1.geometry_id = 1u;
            tris.push_back(t1);
        }
    }
    return tris;
}

void test_half_outward_rounding() {
    const float samples[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.33333334f, -0.33333334f, 1.0001f, -1.0001f,
        1e-10f, -1e-10f, 6.0e-5f, -6.0e-5f, 1234.5678f, -1234.5678f,
        65504.0f, -65504.0f, 70000.0f, -70000.0f, 1e20f, -1e20f,
    };
    for (float x : samples) {
        const float lo = half_to_float(float_to_half_floor(x));
        const float hi = half_to_float(float_to_half_ceil(x));
        CHECK(lo <= x);
        CHECK(hi >= x);
    }
    CHECK(std::isnan(half_to_float(float_to_half_floor(NAN))));
    CHECK(std::isnan(half_to_float(float_to_half_ceil(NAN))));
}

void test_affine_3x4_and_instance_packing() {
    // Rotation around Z by 90 deg + non-uniform scale (2, 3, 4) + translation (10, -5, 2).
    const float m[3][4] = {
        {0.0f, -3.0f, 0.0f, 10.0f},
        {2.0f, 0.0f, 0.0f, -5.0f},
        {0.0f, 0.0f, 4.0f, 2.0f},
    };
    float inv[3][4];
    CHECK(invert_affine_3x4(m, inv));
    const Vec3 p{1.5f, -2.25f, 3.75f};
    const Vec3 wp = transform_point_3x4(m, p);
    const Vec3 back = transform_point_3x4(inv, wp);
    CHECK(near_f(back.x, p.x) && near_f(back.y, p.y) && near_f(back.z, p.z));

    // Singular matrix must fail cleanly.
    const float singular[3][4] = {{1, 0, 0, 0}, {2, 0, 0, 1}, {0, 0, 1, 0}};
    CHECK(!invert_affine_3x4(singular, inv));

    // Tight transformed AABB encloses all 8 corners.
    const Aabb local{{-1.0f, -2.0f, -3.0f}, {4.0f, 5.0f, 6.0f}};
    const Aabb world = transform_aabb_3x4(local, m);
    for (int i = 0; i < 8; ++i) {
        const Vec3 corner{
            (i & 1) ? local.max.x : local.min.x,
            (i & 2) ? local.max.y : local.min.y,
            (i & 4) ? local.max.z : local.min.z,
        };
        const Vec3 wc = transform_point_3x4(m, corner);
        CHECK(wc.x >= world.min.x - 1e-4f && wc.x <= world.max.x + 1e-4f);
        CHECK(wc.y >= world.min.y - 1e-4f && wc.y <= world.max.y + 1e-4f);
        CHECK(wc.z >= world.min.z - 1e-4f && wc.z <= world.max.z + 1e-4f);
    }

    // 64-byte kind-6 instance node round-trip with 40-bit 256-byte aligned BLAS address.
    InstanceDesc inst{};
    std::memcpy(inst.transform, m, sizeof(m));
    inst.instance_id = 0xABC123u;
    inst.instance_mask = 0x5Au;
    inst.hit_group_contribution = 0x0F1E2Du;
    inst.flags = InstanceFlags::TriangleCullDisable | InstanceFlags::ForceOpaque;
    inst.blas_address = 0x8123456700ull;  // 40-bit, 256-byte aligned
    inst.blas_root_node = make_node_ptr(1u, kNodeKindBox32);

    uint32_t packed[16];
    CHECK(pack_instance_node(inst, packed));
    const PackedInstanceView view = unpack_instance_node(packed);
    CHECK(view.instance_id == inst.instance_id);
    CHECK(view.instance_mask == inst.instance_mask);
    CHECK(view.hit_group_contribution == inst.hit_group_contribution);
    CHECK(view.flags == inst.flags);
    CHECK(view.blas_address == inst.blas_address);
    CHECK(view.blas_root_node == inst.blas_root_node);
    const Vec3 roundtrip = transform_point_3x4(view.inv_transform, wp);
    CHECK(near_f(roundtrip.x, p.x) && near_f(roundtrip.y, p.y) && near_f(roundtrip.z, p.z));

    // Misaligned or >40-bit BLAS address must fail.
    InstanceDesc bad = inst;
    bad.blas_address = 0x10040ull;  // not 256-byte aligned
    CHECK(!pack_instance_node(bad, packed));
    bad.blas_address = 1ull << 40;
    CHECK(!pack_instance_node(bad, packed));
}

void test_blas_build_and_refit(BoxFormat format) {
    auto tris = make_test_scene(6, 5.0f);  // 72 triangles
    BuildScratch bscratch;
    bscratch.reserve(static_cast<uint32_t>(tris.size()));
    BvhStructure blas;
    CHECK(build_blas(tris.data(), static_cast<uint32_t>(tris.size()), format, bscratch, blas));
    CHECK(!blas.empty());
    CHECK(!blas.is_tlas);
    CHECK(blas.element_count == tris.size());
    CHECK(blas.byte_size() % kStructureAlignment == 0);

    const PrebuildInfo pb = compute_prebuild_info(static_cast<uint32_t>(tris.size()), format, false);
    CHECK(blas.byte_size() <= pb.result_bytes);

    const StructureHeader hdr = blas.header();
    CHECK((hdr.version_flags & 0xFFFFu) == kHeaderVersion);
    CHECK((hdr.version_flags & kHeaderFlagBottomLevel) != 0);
    CHECK(hdr.element_count == tris.size());
    CHECK(hdr.root_node == blas.root_node);
    CHECK(hdr.generation == 1u);

    constexpr uint64_t kBlasBase = 0x20000ull;
    const auto make_loader = [&](const BvhStructure& s) -> LoadU32 {
        return [&s](uint64_t a) -> uint32_t {
            if (a < kBlasBase) return 0u;
            const uint64_t idx = (a - kBlasBase) / 4ull;
            return idx < s.dwords.size() ? s.dwords[static_cast<size_t>(idx)] : 0u;
        };
    };

    // Compare 81 rays against brute force.
    uint32_t hit_count = 0;
    for (int ix = -4; ix <= 4; ++ix) {
        for (int iy = -4; iy <= 4; ++iy) {
            const Vec3 o{static_cast<float>(ix) * 1.35f, static_cast<float>(iy) * 1.35f, 0.0f};
            const Vec3 d{0.05f * static_cast<float>(ix), -0.04f * static_cast<float>(iy), 1.0f};
            const TraceHit got = trace_blas(kBlasBase, blas.root_node, o, d, 0.001f, 100.0f, make_loader(blas));
            const TraceHit ref = brute_force_blas(tris, o, d, 0.001f, 100.0f);
            CHECK(got.hit == ref.hit);
            if (ref.hit) {
                ++hit_count;
                CHECK(near_f(got.t, ref.t, 1e-3f));
                CHECK(got.primitive_index == ref.primitive_index);
                CHECK(near_f(got.u, ref.u, 1e-3f) && near_f(got.v, ref.v, 1e-3f));
                CHECK(got.front_face == ref.front_face);
            }
        }
    }
    CHECK(hit_count >= 25u);

    // Dynamic in-place refit: translate & deform all triangles outside their original bounds.
    for (auto& t : tris) {
        t.v0.x += 12.0f; t.v1.x += 12.0f; t.v2.x += 12.0f;
        t.v0.y -= 4.0f;  t.v1.y -= 4.0f;  t.v2.y -= 4.0f;
        t.v0.z += 3.0f;  t.v1.z += 3.5f;  t.v2.z += 2.5f;
    }
    RefitScratch rscratch;
    rscratch.reserve(static_cast<uint32_t>(tris.size()), blas.box_count);
    const uint32_t* ptr_before = blas.dwords.data();
    const size_t size_before = blas.dwords.size();
    CHECK(refit_blas(tris.data(), static_cast<uint32_t>(tris.size()), rscratch, blas));
    CHECK(blas.dwords.data() == ptr_before && blas.dwords.size() == size_before);
    CHECK(blas.generation == 2u && blas.header().generation == 2u);

    // Verify hits moved to the new region and match brute force.
    uint32_t refit_hits = 0;
    for (int ix = -3; ix <= 3; ++ix) {
        for (int iy = -3; iy <= 3; ++iy) {
            const Vec3 o{12.0f + static_cast<float>(ix) * 1.35f, -4.0f + static_cast<float>(iy) * 1.35f, 0.0f};
            const Vec3 d{0.0f, 0.0f, 1.0f};
            const TraceHit got = trace_blas(kBlasBase, blas.root_node, o, d, 0.001f, 100.0f, make_loader(blas));
            const TraceHit ref = brute_force_blas(tris, o, d, 0.001f, 100.0f);
            CHECK(got.hit == ref.hit);
            if (ref.hit) {
                ++refit_hits;
                CHECK(near_f(got.t, ref.t, 1e-3f));
                CHECK(got.primitive_index == ref.primitive_index);
            }
        }
    }
    CHECK(refit_hits >= 15u);

    // Refit rejects count mismatch.
    CHECK(!refit_blas(tris.data(), static_cast<uint32_t>(tris.size() - 1), rscratch, blas));

    // Edge cases: empty BLAS (0 triangles) and single-triangle BLAS.
    BvhStructure empty_blas;
    CHECK(build_blas(nullptr, 0u, format, bscratch, empty_blas));
    CHECK(empty_blas.empty() && empty_blas.byte_size() == kStructureAlignment);
    CHECK(refit_blas(nullptr, 0u, rscratch, empty_blas) && empty_blas.generation == 2u);

    Triangle one{};
    one.v0 = {-1, -1, 2}; one.v1 = {0, 1, 2}; one.v2 = {1, -1, 2}; one.primitive_id = 42;
    BvhStructure single;
    CHECK(build_blas(&one, 1u, format, bscratch, single));
    const TraceHit h1 = trace_blas(kBlasBase, single.root_node, {0, 0, 0}, {0, 0, 1}, 0.0f, 10.0f, make_loader(single));
    CHECK(h1.hit && near_f(h1.t, 2.0f) && h1.primitive_index == 42u);
}

void test_tlas_arena_integration_and_refit() {
    // Build two distinct BLASes (one Box32, one Box16) and stage them + TLAS into rt_arena::Arena.
    Triangle tri_a[2]{};
    tri_a[0] = {{-1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, 101u, 0u};
    tri_a[1] = {{-1.0f, 1.0f, 0.5f}, {1.0f, 1.0f, 0.5f}, {0.0f, -1.0f, 0.5f}, 102u, 0u};

    Triangle tri_b[1]{};
    tri_b[0] = {{-2.0f, -2.0f, 0.0f}, {0.0f, 2.0f, 0.0f}, {2.0f, -2.0f, 0.0f}, 201u, 1u};

    BuildScratch bscratch;
    bscratch.reserve(16);
    BvhStructure blas_a, blas_b;
    CHECK(build_blas(tri_a, 2u, BoxFormat::Box32, bscratch, blas_a));
    CHECK(build_blas(tri_b, 1u, BoxFormat::Box16, bscratch, blas_b));

    // Place structures at 40-bit 256-byte aligned addresses (including > 4 GiB!).
    constexpr uint64_t kBlasAAddr = 0x0010000000ull;
    constexpr uint64_t kBlasBAddr = 0x0120000000ull;
    constexpr uint64_t kTlasAddr  = 0x0240000000ull;

    std::vector<InstanceDesc> instances(3);
    // Instance 0: BLAS A at z = 5, mask 0x01
    instances[0].transform[0][3] = 0.0f;
    instances[0].transform[1][3] = 0.0f;
    instances[0].transform[2][3] = 5.0f;
    instances[0].instance_id = 0x111u;
    instances[0].instance_mask = 0x01u;
    instances[0].hit_group_contribution = 0x10u;
    instances[0].blas_address = kBlasAAddr;
    instances[0].blas_root_node = blas_a.root_node;
    instances[0].local_bounds = blas_a.root_bounds;

    // Instance 1: BLAS B at (10, 0, 8), scaled 0.5x, mask 0x02
    instances[1].transform[0][0] = 0.5f;
    instances[1].transform[1][1] = 0.5f;
    instances[1].transform[2][2] = 0.5f;
    instances[1].transform[0][3] = 10.0f;
    instances[1].transform[2][3] = 8.0f;
    instances[1].instance_id = 0x222u;
    instances[1].instance_mask = 0x02u;
    instances[1].hit_group_contribution = 0x20u;
    instances[1].flags = InstanceFlags::TriangleFrontCounterclockwise;
    instances[1].blas_address = kBlasBAddr;
    instances[1].blas_root_node = blas_b.root_node;
    instances[1].local_bounds = blas_b.root_bounds;

    // Instance 2: BLAS A behind Instance 0 at z = 12, mask 0x04
    instances[2].transform[2][3] = 12.0f;
    instances[2].instance_id = 0x333u;
    instances[2].instance_mask = 0x04u;
    instances[2].hit_group_contribution = 0x30u;
    instances[2].blas_address = kBlasAAddr;
    instances[2].blas_root_node = blas_a.root_node;
    instances[2].local_bounds = blas_a.root_bounds;

    BvhStructure tlas;
    CHECK(build_tlas(instances.data(), static_cast<uint32_t>(instances.size()), BoxFormat::Box32, bscratch, tlas));
    CHECK(tlas.is_tlas && tlas.element_count == 3u);

    // Stage BLAS A, BLAS B, and TLAS into group-3 rt_arena::Arena backing memory.
    constexpr uint32_t kPagesPerArena = 8;
    rt_arena::Arena arena(rt_arena::kArenaCount, kPagesPerArena);
    std::vector<uint32_t> backing[rt_arena::kArenaCount];
    const uint32_t* backing_ptrs[rt_arena::kArenaCount];
    for (uint32_t i = 0; i < rt_arena::kArenaCount; ++i) {
        backing[i].assign(static_cast<size_t>(kPagesPerArena) * (rt_arena::kPageBytes / 4u), 0u);
        backing_ptrs[i] = backing[i].data();
    }

    std::vector<rt_arena::StagedWrite> writes;
    const auto upload_structure = [&](uint64_t id, uint64_t gpu_addr, const BvhStructure& s) {
        const auto res = arena.stage_structure(id, gpu_addr, s.byte_size(), true, &writes);
        CHECK(res == rt_arena::Ensure::Uploaded || res == rt_arena::Ensure::Resident);
        const auto* src_bytes = reinterpret_cast<const uint8_t*>(s.dwords.data());
        for (const auto& w : writes) {
            std::memcpy(reinterpret_cast<uint8_t*>(backing[w.arena].data()) + w.arena_byte_offset,
                        src_bytes + w.src_byte_offset, w.byte_size);
        }
    };
    upload_structure(1, kBlasAAddr, blas_a);
    upload_structure(2, kBlasBAddr, blas_b);
    upload_structure(3, kTlasAddr, tlas);

    uint64_t faults = 0;
    const LoadU32 arena_load = [&](uint64_t addr) -> uint32_t {
        return arena.load32_simulated(addr, backing_ptrs, &faults);
    };

    // Ray along +Z at x=0 hits instance 0 at t=5 (mask 0xFF).
    const TraceHit h0 = trace_tlas(kTlasAddr, tlas.root_node, {0, 0, 0}, {0, 0, 1}, 0.01f, 50.0f, 0xFFu, arena_load);
    CHECK(h0.hit && near_f(h0.t, 5.0f) && h0.instance_id == 0x111u && h0.hit_group_contribution == 0x10u);
    CHECK(h0.instance_index == 0u && h0.primitive_index == 101u);

    // Mask filtering: mask 0x04 skips instance 0 (0x01) and hits instance 2 at t=12!
    const TraceHit h2 = trace_tlas(kTlasAddr, tlas.root_node, {0, 0, 0}, {0, 0, 1}, 0.01f, 50.0f, 0x04u, arena_load);
    CHECK(h2.hit && near_f(h2.t, 12.0f) && h2.instance_id == 0x333u && h2.instance_index == 2u);

    // Ray at x=10 hits scaled instance 1 at t=8, with flipped front_face due to TriangleFrontCounterclockwise.
    const TraceHit h1 = trace_tlas(kTlasAddr, tlas.root_node, {10, 0, 0}, {0, 0, 1}, 0.01f, 50.0f, 0x02u, arena_load);
    CHECK(h1.hit && near_f(h1.t, 8.0f) && h1.instance_id == 0x222u && h1.primitive_index == 201u && !h1.front_face);

    // Dynamic TLAS refit: move instance 1 from x=10 to x=-20, z=4 and disable instance 0 (mask=0).
    instances[0].instance_mask = 0u;
    instances[1].transform[0][3] = -20.0f;
    instances[1].transform[2][3] = 4.0f;

    RefitScratch rscratch;
    rscratch.reserve(3u, tlas.box_count);
    const uint32_t* tlas_ptr = tlas.dwords.data();
    CHECK(refit_tlas(instances.data(), static_cast<uint32_t>(instances.size()), rscratch, tlas));
    CHECK(tlas.dwords.data() == tlas_ptr && tlas.generation == 2u);
    upload_structure(3, kTlasAddr, tlas);

    // Old position x=10 now misses; new position x=-20 hits at t=4; x=0 hits instance 2 at t=12.
    const TraceHit old_h1 = trace_tlas(kTlasAddr, tlas.root_node, {10, 0, 0}, {0, 0, 1}, 0.01f, 50.0f, 0xFFu, arena_load);
    CHECK(!old_h1.hit);
    const TraceHit new_h1 = trace_tlas(kTlasAddr, tlas.root_node, {-20, 0, 0}, {0, 0, 1}, 0.01f, 50.0f, 0xFFu, arena_load);
    CHECK(new_h1.hit && near_f(new_h1.t, 4.0f) && new_h1.instance_id == 0x222u);
    const TraceHit h_after = trace_tlas(kTlasAddr, tlas.root_node, {0, 0, 0}, {0, 0, 1}, 0.01f, 50.0f, 0xFFu, arena_load);
    CHECK(h_after.hit && near_f(h_after.t, 12.0f) && h_after.instance_id == 0x333u);
    CHECK(faults == 0u);
}

void test_bounded_scratch_zero_alloc_steady_state() {
    auto tris = make_test_scene(5, 2.0f);  // 50 triangles
    BuildScratch bscratch;
    bscratch.reserve(64);
    BvhStructure blas;
    const PrebuildInfo pb = compute_prebuild_info(64, BoxFormat::Box32, false);
    blas.dwords.reserve(static_cast<size_t>(pb.result_bytes / 4u));

    CHECK(build_blas(tris.data(), static_cast<uint32_t>(tris.size()), BoxFormat::Box32, bscratch, blas));
    const size_t steady_bytes = bscratch.bytes_allocated();
    const uint32_t* steady_out_ptr = blas.dwords.data();

    for (int iter = 0; iter < 10; ++iter) {
        tris[0].v0.x += 0.1f;
        CHECK(build_blas(tris.data(), static_cast<uint32_t>(tris.size()), BoxFormat::Box32, bscratch, blas));
        CHECK(bscratch.bytes_allocated() == steady_bytes);
        CHECK(blas.dwords.data() == steady_out_ptr);
    }
    CHECK(bscratch.reuse_count == 11u);

    RefitScratch rscratch;
    rscratch.reserve(64, blas.box_count);
    const size_t steady_refit_bytes = rscratch.bytes_allocated();
    for (int iter = 0; iter < 20; ++iter) {
        tris[1].v1.y += 0.05f;
        CHECK(refit_blas(tris.data(), static_cast<uint32_t>(tris.size()), rscratch, blas));
        CHECK(rscratch.bytes_allocated() == steady_refit_bytes);
        CHECK(blas.dwords.data() == steady_out_ptr);
    }
    CHECK(rscratch.reuse_count == 20u);

    // Budget cap trims oversized scratch buffers.
    bscratch.set_budget_bytes(128);
    CHECK(bscratch.budget_trims >= 1u && bscratch.bytes_allocated() <= 128u);
    rscratch.set_budget_bytes(64);
    CHECK(rscratch.budget_trims >= 1u && rscratch.bytes_allocated() <= 64u);
}

}  // namespace

int main() {
    const auto vectors = rt_bvh_vectors::all();
    CHECK(vectors.size() >= 18);
    for (const auto& v : vectors) {
        std::map<uint64_t, uint32_t> mem;
        for (const auto& m : v.memory) {
            for (size_t i = 0; i < m.dwords.size(); ++i) mem[m.addr + 4 * i] = m.dwords[i];
        }
        const auto load = [&](uint64_t a) -> uint32_t {
            const auto it = mem.find(a);
            return it == mem.end() ? 0u : it->second;
        };
        const float inv[3] = {rt_bvh_vectors::inv_of(v.dir[0]), rt_bvh_vectors::inv_of(v.dir[1]),
                              rt_bvh_vectors::inv_of(v.dir[2])};
        const Result r = intersect(v.desc, v.node, v.extent, {v.origin[0], v.origin[1], v.origin[2]},
                                   {v.dir[0], v.dir[1], v.dir[2]}, {inv[0], inv[1], inv[2]}, load);
        char got[160];
        std::snprintf(got, sizeof got, " got %08x %08x %08x %08x want %08x %08x %08x %08x", r.w[0], r.w[1], r.w[2],
                      r.w[3], v.expected[0], v.expected[1], v.expected[2], v.expected[3]);
        check_impl(std::memcmp(r.w, v.expected, 16) == 0, v.name + got, __LINE__);
    }
    // The prelude exposes the C4 entry point and the C3 bindings, and carries the version.
    const std::string wgsl = prelude_wgsl();
    CHECK(wgsl.find("fn pg6_bvh_intersect(desc: vec4<u32>, node: vec2<u32>, extent: f32, origin: vec3<f32>, "
                    "dir: vec3<f32>,") != std::string::npos);
    CHECK(wgsl.find("-> vec4<u32>") != std::string::npos);
    for (int b = 16; b <= 21; ++b) {
        CHECK(wgsl.find("@group(3) @binding(" + std::to_string(b) + ")") != std::string::npos);
    }
    CHECK(wgsl.find("prelude, version " + std::to_string(kPreludeVersion)) != std::string::npos);

    test_half_outward_rounding();
    test_affine_3x4_and_instance_packing();
    test_blas_build_and_refit(BoxFormat::Box32);
    test_blas_build_and_refit(BoxFormat::Box16);
    test_tlas_arena_integration_and_refit();
    test_bounded_scratch_zero_alloc_steady_state();

    if (g_failures == 0) std::printf("test_rt_bvh: OK (%zu vectors + BLAS/TLAS/refit/arena)\n", vectors.size());
    return g_failures == 0 ? 0 : 1;
}
