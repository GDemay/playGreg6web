// Software image_bvh_intersect_ray / image_bvh64_intersect_ray (docs/RAYTRACING_DESIGN.md sections 2
// and 6, contract C4/C5): the per-node test on the console's native node format, as a CPU twin of
// the runtime-owned WGSL prelude function pg6_bvh_intersect. One implementation of the semantics,
// pinned by shared test vectors (tests/rt_bvh_vectors.h) that run against both.
//
// Choices the design leaves open ("unknown on the console, assumed") are made here and pinned by the
// vectors: box slab tests are inclusive; a NaN slab (0 * inf) constrains nothing on that axis; any
// NaN in the ray gives an all-miss; an empty box (min > max on an axis) is never entered; triangle
// edge functions use single precision with no double fallback; zero edge functions count as inside.
//
// Also provides 3D RT BVH BLAS & TLAS 4-wide builders, in-place bottom-up dynamic refit, 64-byte
// kind-6 instance node packing, bounded reusable scratch buffers, and reference CPU traversal.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace playgta6::gpu::rt_bvh {

// Version of the WGSL prelude (prelude_wgsl). Bump on any change to its semantics or interface; the
// translator reports the version it was written against.
constexpr uint32_t kPreludeVersion = 1;

constexpr uint32_t kNoChild = 0xFFFFFFFFu;
constexpr uint64_t kAddressMask = (1ull << 40) - 1;

// Native console node / structure constants (docs/RAYTRACING_DESIGN.md section 2).
constexpr uint32_t kNodeGranularityBytes = 64;
constexpr uint32_t kNodeGranularityDwords = 16;
constexpr uint32_t kStructureAlignment = 256;
constexpr uint32_t kHeaderVersion = 6;
constexpr uint32_t kHeaderFlagBottomLevel = 0x80000000u;
constexpr uint32_t kHeaderFlagCompacted = 0x00040000u;
constexpr uint32_t kHeaderFlagRehydrated = 0x00020000u;
constexpr uint32_t kDescriptorTypeBvh = 0x80000000u;  // type 8 in dw3 bits 31:28

constexpr uint32_t kNodeKindTri0 = 0;
constexpr uint32_t kNodeKindTri1 = 1;
constexpr uint32_t kNodeKindTri2 = 2;
constexpr uint32_t kNodeKindTri3 = 3;
constexpr uint32_t kNodeKindBox16 = 4;
constexpr uint32_t kNodeKindBox32 = 5;
constexpr uint32_t kNodeKindInstance = 6;
constexpr uint32_t kNodeKindProcedural = 7;

constexpr uint32_t kTriNodeDwords = 16;       // 64 bytes (1 slot)
constexpr uint32_t kBox16NodeDwords = 16;     // 64 bytes (1 slot)
constexpr uint32_t kBox32NodeDwords = 32;     // 128 bytes (2 slots)
constexpr uint32_t kInstanceNodeDwords = 16;  // 64 bytes (1 slot)
constexpr uint32_t kDefaultTriSelector = 0x09090909u;  // I = v1, J = v2 for all 4 triangle slots

struct Vec3 {
    float x, y, z;
};

// Reads the dword at a 40-bit GPU address (what pg6_load32 does on the GPU).
using LoadU32 = std::function<uint32_t(uint64_t addr)>;

struct Result {
    uint32_t w[4];
    bool operator==(const Result& o) const {
        return w[0] == o.w[0] && w[1] == o.w[1] && w[2] == o.w[2] && w[3] == o.w[3];
    }
};

// `desc`: the 128-bit BVH descriptor (base in 256-byte units in dw0 and the low byte of dw1, sort in
// dw1 bit 31). `node`: the 32- or 64-bit node pointer (kind in the low 3 bits; the node is at
// base + ((ptr >> 3) << 6)). Box nodes (kind 4 f16, kind 5 f32) return the four child pointers the
// ray enters within [0, extent] (sorted nearest first when sort is set; 0xFFFFFFFF for a miss);
// triangle kinds 0-3 return t_num, t_den, i_num, j_num as f32 bits (a miss is +inf, 1, 0, 0);
// instances and procedural nodes (6, 7) answer a triangle miss.
Result intersect(const uint32_t desc[4], uint64_t node, float extent, Vec3 origin, Vec3 dir, Vec3 inv_dir,
                 const LoadU32& load);

// The WGSL prelude: arena bindings (group 3, 16-21), pg6_load32 (directory walk), and
// pg6_bvh_intersect(desc: vec4<u32>, node: vec2<u32>, extent: f32, origin: vec3<f32>,
// dir: vec3<f32>, inv_dir: vec3<f32>) -> vec4<u32>.
const char* prelude_wgsl();

// -----------------------------------------------------------------------------
// 3D RT BVH BLAS / TLAS builder, dynamic refit, instance packing, and traversal
// -----------------------------------------------------------------------------

enum class BoxFormat : uint8_t {
    Box16 = kNodeKindBox16,  // 64-byte f16 box nodes (outward-rounded bounds)
    Box32 = kNodeKindBox32,  // 128-byte f32 box nodes (exact bounds)
};

constexpr uint32_t make_node_ptr(uint32_t slot64, uint32_t kind) {
    return (slot64 << 3) | (kind & 7u);
}
constexpr uint32_t node_ptr_slot(uint32_t ptr) { return ptr >> 3; }
constexpr uint32_t node_ptr_kind(uint32_t ptr) { return ptr & 7u; }

struct Aabb {
    Vec3 min;
    Vec3 max;

    static Aabb empty() {
        const float inf = std::numeric_limits<float>::infinity();
        return {{inf, inf, inf}, {-inf, -inf, -inf}};
    }

    static Aabb from_point(Vec3 p) { return {p, p}; }

    bool is_empty() const { return min.x > max.x || min.y > max.y || min.z > max.z; }

    void expand(Vec3 p) {
        if (p.x < min.x) min.x = p.x;
        if (p.y < min.y) min.y = p.y;
        if (p.z < min.z) min.z = p.z;
        if (p.x > max.x) max.x = p.x;
        if (p.y > max.y) max.y = p.y;
        if (p.z > max.z) max.z = p.z;
    }

    void expand(const Aabb& o) {
        if (o.is_empty()) return;
        if (o.min.x < min.x) min.x = o.min.x;
        if (o.min.y < min.y) min.y = o.min.y;
        if (o.min.z < min.z) min.z = o.min.z;
        if (o.max.x > max.x) max.x = o.max.x;
        if (o.max.y > max.y) max.y = o.max.y;
        if (o.max.z > max.z) max.z = o.max.z;
    }

    Vec3 centroid() const {
        return {0.5f * (min.x + max.x), 0.5f * (min.y + max.y), 0.5f * (min.z + max.z)};
    }

    Vec3 extent() const {
        if (is_empty()) return {0.0f, 0.0f, 0.0f};
        return {max.x - min.x, max.y - min.y, max.z - min.z};
    }

    float surface_area() const {
        if (is_empty()) return 0.0f;
        const float dx = max.x - min.x;
        const float dy = max.y - min.y;
        const float dz = max.z - min.z;
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }

    int longest_axis() const {
        const Vec3 e = extent();
        if (e.y > e.x && e.y >= e.z) return 1;
        if (e.z > e.x && e.z > e.y) return 2;
        return 0;
    }
};

// Exact binary16 <-> binary32 conversions with outward rounding for conservative f16 box bounds.
uint16_t float_to_half_floor(float x);
uint16_t float_to_half_ceil(float x);
float half_to_float(uint16_t h);

// Builds a 128-bit BVH descriptor for a 256-byte aligned 40-bit base address.
void make_bvh_desc(uint64_t base_addr, bool sort_boxes, uint32_t out_desc[4]);

// 3x4 row-major affine matrix helpers.
bool invert_affine_3x4(const float m[3][4], float out_inv[3][4]);
Vec3 transform_point_3x4(const float m[3][4], Vec3 p);
Vec3 transform_vector_3x4(const float m[3][4], Vec3 v);
Aabb transform_aabb_3x4(const Aabb& local, const float m[3][4]);

// Triangle input for BLAS build / refit.
struct Triangle {
    Vec3 v0{0.0f, 0.0f, 0.0f};
    Vec3 v1{0.0f, 0.0f, 0.0f};
    Vec3 v2{0.0f, 0.0f, 0.0f};
    uint32_t primitive_id = 0;
    uint32_t geometry_id = 0;
};

namespace InstanceFlags {
constexpr uint8_t None = 0u;
constexpr uint8_t TriangleCullDisable = 1u << 0;
constexpr uint8_t TriangleFrontCounterclockwise = 1u << 1;
constexpr uint8_t ForceOpaque = 1u << 2;
constexpr uint8_t ForceNonOpaque = 1u << 3;
}  // namespace InstanceFlags

// Top-level instance descriptor for TLAS build / refit and kind-6 64-byte node packing.
struct InstanceDesc {
    float transform[3][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    };
    float inv_transform[3][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    };
    bool has_inv_transform = false;
    uint32_t instance_id = 0;             // 24-bit
    uint8_t instance_mask = 0xFFu;        // 8-bit
    uint32_t hit_group_contribution = 0;  // 24-bit
    uint8_t flags = 0;                    // 8-bit (InstanceFlags)
    uint64_t blas_address = 0;            // 40-bit, 256-byte aligned
    uint32_t blas_root_node = kNoChild;   // root node pointer of the referenced BLAS
    Aabb local_bounds = Aabb::empty();    // BLAS local-space AABB
    Aabb world_bounds = Aabb::empty();    // optional precomputed world AABB (derived if empty)
};

// Decoded view of a 64-byte kind-6 instance node.
struct PackedInstanceView {
    float inv_transform[3][4] = {};
    uint32_t instance_id = 0;
    uint8_t instance_mask = 0;
    uint32_t hit_group_contribution = 0;
    uint8_t flags = 0;
    uint64_t blas_address = 0;
    uint32_t blas_root_node = kNoChild;
};

// Native node encoders / decoders.
void pack_triangle_node(const Triangle& tri, uint32_t out_dwords[16]);
void pack_triangle_node_multi(const Vec3 verts[5], uint32_t selector, uint32_t out_dwords[16]);
void pack_box32_node(const uint32_t child_ptrs[4], const Aabb child_bounds[4], uint32_t out_dwords[32]);
void pack_box16_node(const uint32_t child_ptrs[4], const Aabb child_bounds[4], uint32_t out_dwords[16],
                     Aabb quantized_out[4] = nullptr);
bool pack_instance_node(const InstanceDesc& inst, uint32_t out_dwords[16]);
PackedInstanceView unpack_instance_node(const uint32_t in_dwords[16]);

// 64-byte acceleration structure header placed at slot 0 (byte offset 0) of every built BLAS/TLAS.
struct StructureHeader {
    uint32_t version_flags = 0;
    uint32_t total_slots64 = 0;
    uint32_t box_kind = 0;
    uint32_t element_count = 0;
    float min_xyz[3] = {0.0f, 0.0f, 0.0f};
    float max_xyz[3] = {0.0f, 0.0f, 0.0f};
    uint32_t box_offset_bytes = 0;
    uint32_t box_count = 0;
    uint32_t root_node = kNoChild;
    uint32_t elements_offset_bytes = 0;
    uint32_t parent_table_offset_bytes = 0;
    uint32_t generation = 0;
};
static_assert(sizeof(StructureHeader) == 64, "StructureHeader must stay 64 bytes");

struct PrebuildInfo {
    uint64_t result_bytes = kStructureAlignment;
    uint64_t build_scratch_bytes = kStructureAlignment;
    uint64_t update_scratch_bytes = kStructureAlignment;
    uint32_t max_box_nodes = 0;
    uint32_t max_leaf_nodes = 0;
};

PrebuildInfo compute_prebuild_info(uint32_t primitive_count, BoxFormat format = BoxFormat::Box32,
                                   bool is_tlas = false);

// Self-contained relocatable BLAS or TLAS in the console's native node format.
struct BvhStructure {
    std::vector<uint32_t> dwords;
    bool is_tlas = false;
    BoxFormat box_format = BoxFormat::Box32;
    uint32_t root_node = kNoChild;
    uint32_t element_count = 0;
    uint32_t box_count = 0;
    uint32_t box_offset_dwords = 0;
    uint32_t elements_offset_dwords = 0;
    uint32_t parent_table_offset_dwords = 0;
    uint32_t generation = 0;
    Aabb root_bounds = Aabb::empty();

    size_t byte_size() const { return dwords.size() * sizeof(uint32_t); }
    bool empty() const { return element_count == 0 || root_node == kNoChild; }
    StructureHeader header() const;
};

// Bounded reusable scratch memory for BLAS/TLAS builds (zero steady-state heap allocations).
struct BuildScratch {
    struct BinNode {
        Aabb bounds = Aabb::empty();
        uint32_t left = kNoChild;
        uint32_t right = kNoChild;
        uint32_t prim_index = kNoChild;
        uint32_t prim_count = 0;
    };
    struct CollapseTask {
        uint32_t bin_node = 0;
        uint32_t box_index = 0;
    };

    std::vector<Aabb> prim_aabbs;
    std::vector<Vec3> prim_centroids;
    std::vector<uint32_t> prim_indices;
    std::vector<BinNode> bin_nodes;
    std::vector<CollapseTask> collapse_stack;
    std::vector<uint32_t> parent_ptrs;

    size_t budget_bytes = 0;  // 0 = unbounded
    size_t peak_bytes = 0;
    uint64_t reuse_count = 0;
    uint64_t budget_trims = 0;

    void reserve(uint32_t max_primitives);
    void set_budget_bytes(size_t bytes);
    size_t bytes_allocated() const;
    void trim_to_budget();
};

// Bounded reusable scratch memory for in-place bottom-up BLAS/TLAS refit (zero steady-state allocs).
struct RefitScratch {
    std::vector<Aabb> leaf_aabbs;
    std::vector<Aabb> box_aabbs;

    size_t budget_bytes = 0;  // 0 = unbounded
    size_t peak_bytes = 0;
    uint64_t reuse_count = 0;
    uint64_t budget_trims = 0;

    void reserve(uint32_t max_primitives, uint32_t max_boxes);
    void set_budget_bytes(size_t bytes);
    size_t bytes_allocated() const;
    void trim_to_budget();
};

// Builds a 4-wide BLAS over `triangles` into `out` using `scratch`. Reuses `out.dwords` capacity.
bool build_blas(const Triangle* triangles, uint32_t count, BoxFormat format, BuildScratch& scratch,
                BvhStructure& out);

// Refits an existing BLAS in place over updated triangle positions (same primitive count & topology).
bool refit_blas(const Triangle* triangles, uint32_t count, RefitScratch& scratch, BvhStructure& blas);

// Builds a 4-wide TLAS over 3x4 affine `instances` into `out` using `scratch`.
bool build_tlas(const InstanceDesc* instances, uint32_t count, BoxFormat format, BuildScratch& scratch,
                BvhStructure& out);

// Refits an existing TLAS in place over updated instance transforms/bounds (same count & topology).
bool refit_tlas(const InstanceDesc* instances, uint32_t count, RefitScratch& scratch, BvhStructure& tlas);

// CPU reference closest-hit traversal built directly on `rt_bvh::intersect(...)`.
struct TraceHit {
    bool hit = false;
    float t = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    bool front_face = true;
    uint32_t primitive_index = 0;
    uint32_t instance_index = 0;
    uint32_t instance_id = 0;
    uint32_t hit_group_contribution = 0;
    uint32_t nodes_visited = 0;
};

TraceHit trace_blas(uint64_t base_addr, uint32_t root_node, Vec3 origin, Vec3 dir, float t_min, float t_max,
                    const LoadU32& load, bool sort_boxes = true, bool cull_back_faces = false);

TraceHit trace_tlas(uint64_t tlas_base_addr, uint32_t tlas_root_node, Vec3 origin, Vec3 dir, float t_min,
                    float t_max, uint8_t ray_mask, const LoadU32& load, bool sort_boxes = true,
                    bool cull_back_faces = false);

}  // namespace playgta6::gpu::rt_bvh
