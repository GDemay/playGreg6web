#include "gpu/rt_bvh.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace playgta6::gpu::rt_bvh {

namespace {

float bits_f(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint32_t f_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
bool isnan_(float f) { return f != f; }

// Exact binary16 -> binary32 (what unpack2x16float does).
float half_f(uint32_t h) {
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t out;
    if (e == 0) {
        if (m == 0) {
            out = s << 31;
        } else {  // subnormal: normalise
            uint32_t ee = 127 - 15 + 1;
            uint32_t mm = m;
            while ((mm & 0x400u) == 0) {
                mm <<= 1;
                --ee;
            }
            out = (s << 31) | (ee << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31) {
        out = (s << 31) | 0x7f800000u | (m << 13);
    } else {
        out = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
    }
    return bits_f(out);
}

// Round-to-nearest-even binary32 -> binary16.
uint16_t f32_to_f16_rne(float f) {
    const uint32_t x = f_bits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xffu;
    const uint32_t mant = x & 0x7fffffu;

    if (exp == 0xffu) {
        if (mant != 0) {
            return static_cast<uint16_t>(sign | 0x7e00u | ((mant >> 13) & 0x3ffu));
        }
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    const int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;
    if (new_exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (new_exp <= 0) {
        if (new_exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        const uint32_t full_mant = (exp == 0) ? mant : (mant | 0x800000u);
        const uint32_t shift = static_cast<uint32_t>(14 - new_exp);
        const uint32_t half_bit = 1u << (shift - 1u);
        uint32_t rounded = full_mant >> shift;
        const uint32_t rem = full_mant & ((1u << shift) - 1u);
        if (rem > half_bit || (rem == half_bit && (rounded & 1u) != 0u)) {
            ++rounded;
        }
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t rounded = mant >> 13;
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (rounded & 1u) != 0u)) {
        ++rounded;
        if (rounded == 0x400u) {
            rounded = 0;
            if (new_exp + 1 >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(new_exp + 1) << 10));
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(new_exp) << 10) | rounded);
}

// Slab test; true when entered. *near_t = entry distance (0 when the ray starts inside).
bool slab(const float mn[3], const float mx[3], const float o[3], const float inv[3], float extent, float* near_t) {
    float lo = 0.0f, hi = extent;
    for (int a = 0; a < 3; ++a) {
        if (mn[a] > mx[a]) {
            return false;  // empty slot
        }
        const float t0 = (mn[a] - o[a]) * inv[a];
        const float t1 = (mx[a] - o[a]) * inv[a];
        if (isnan_(t0) || isnan_(t1)) {
            continue;
        }
        const float a0 = t0 < t1 ? t0 : t1;
        const float a1 = t0 < t1 ? t1 : t0;
        if (a0 > lo) lo = a0;
        if (a1 < hi) hi = a1;
    }
    *near_t = lo;
    return lo <= hi;
}

Result miss_triangle() { return {{f_bits(INFINITY), f_bits(1.0f), 0u, 0u}}; }

Result triangle(const uint32_t n[16], uint32_t kind, const float o[3], const float d[3]) {
    static const uint32_t kVerts[4][3] = {{0, 1, 2}, {1, 3, 2}, {2, 3, 4}, {2, 4, 0}};
    float v[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int c = 0; c < 3; ++c) {
            v[i][c] = bits_f(n[kVerts[kind][i] * 3 + c]) - o[c];
        }
    }
    int kz = 0;
    float az = std::fabs(d[0]);
    if (std::fabs(d[1]) > az) {
        kz = 1;
        az = std::fabs(d[1]);
    }
    if (std::fabs(d[2]) > az) {
        kz = 2;
    }
    if (d[kz] == 0.0f) {
        return miss_triangle();
    }
    int kx = (kz + 1) % 3, ky = (kx + 1) % 3;
    if (d[kz] < 0.0f) {
        const int t = kx;
        kx = ky;
        ky = t;
    }
    const float sx = d[kx] / d[kz], sy = d[ky] / d[kz], sz = 1.0f / d[kz];
    float px[3], py[3], pz[3];
    for (int i = 0; i < 3; ++i) {
        px[i] = v[i][kx] - sx * v[i][kz];
        py[i] = v[i][ky] - sy * v[i][kz];
        pz[i] = sz * v[i][kz];
    }
    const float u = px[2] * py[1] - py[2] * px[1];
    const float vv = px[0] * py[2] - py[0] * px[2];
    const float w = px[1] * py[0] - py[1] * px[0];
    if ((u < 0.0f || vv < 0.0f || w < 0.0f) && (u > 0.0f || vv > 0.0f || w > 0.0f)) {
        return miss_triangle();
    }
    const float det = u + vv + w;
    const float t_num = u * pz[0] + vv * pz[1] + w * pz[2];
    if (det == 0.0f || isnan_(det) || isnan_(t_num) || isnan_(u) || isnan_(vv) || isnan_(w)) {
        return miss_triangle();
    }
    // dword 15: one byte per triangle; bits 1:0 the vertex weighted by I, bits 3:2 the one by J
    // (an index 3 selects nothing: weight 0).
    const uint32_t sel = (n[15] >> (8 * kind)) & 0xffu;
    const float wt[4] = {u, vv, w, 0.0f};
    return {{f_bits(t_num), f_bits(det), f_bits(wt[sel & 3u]), f_bits(wt[(sel >> 2) & 3u])}};
}

inline float axis_of(Vec3 v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); }

inline uint64_t align_up(uint64_t v, uint64_t align) {
    return (v + align - 1u) & ~(align - 1u);
}

inline float safe_inv(float d) { return d == 0.0f ? INFINITY : 1.0f / d; }

inline Aabb triangle_aabb(const Triangle& t) {
    Aabb b = Aabb::from_point(t.v0);
    b.expand(t.v1);
    b.expand(t.v2);
    return b;
}

inline Aabb instance_world_aabb(const InstanceDesc& inst) {
    if (inst.instance_mask == 0 || inst.blas_root_node == kNoChild) {
        return Aabb::empty();
    }
    if (!inst.world_bounds.is_empty()) {
        return inst.world_bounds;
    }
    if (inst.local_bounds.is_empty()) {
        return Aabb::empty();
    }
    return transform_aabb_3x4(inst.local_bounds, inst.transform);
}

// Recursive binned-SAH binary BVH builder operating purely on BuildScratch buffers.
constexpr uint32_t kSahBins = 12;

uint32_t build_binary_node(BuildScratch& s, uint32_t begin, uint32_t end) {
    const uint32_t node_idx = static_cast<uint32_t>(s.bin_nodes.size());
    s.bin_nodes.push_back({});

    Aabb bounds = Aabb::empty();
    Aabb centroid_bounds = Aabb::empty();
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t p = s.prim_indices[i];
        bounds.expand(s.prim_aabbs[p]);
        centroid_bounds.expand(s.prim_centroids[p]);
    }
    const uint32_t count = end - begin;
    if (count == 1u) {
        BuildScratch::BinNode& leaf = s.bin_nodes[node_idx];
        leaf.bounds = bounds;
        leaf.left = kNoChild;
        leaf.right = kNoChild;
        leaf.prim_index = s.prim_indices[begin];
        leaf.prim_count = 1u;
        return node_idx;
    }

    int best_axis = -1;
    uint32_t best_split_bin = 0;
    float best_cost = std::numeric_limits<float>::infinity();
    float best_min_c = 0.0f;
    float best_scale = 0.0f;

    struct Bin {
        Aabb bounds = Aabb::empty();
        uint32_t count = 0;
    };

    const float parent_area = bounds.surface_area();
    if (parent_area > 0.0f) {
        for (int axis = 0; axis < 3; ++axis) {
            const float c_min = axis_of(centroid_bounds.min, axis);
            const float c_max = axis_of(centroid_bounds.max, axis);
            const float span = c_max - c_min;
            if (!(span > 1e-7f)) continue;

            Bin bins[kSahBins];
            const float scale = static_cast<float>(kSahBins) / span;
            for (uint32_t i = begin; i < end; ++i) {
                const uint32_t p = s.prim_indices[i];
                const float c = axis_of(s.prim_centroids[p], axis);
                int b = static_cast<int>((c - c_min) * scale);
                if (b < 0) b = 0;
                if (b >= static_cast<int>(kSahBins)) b = static_cast<int>(kSahBins) - 1;
                bins[b].bounds.expand(s.prim_aabbs[p]);
                ++bins[b].count;
            }

            Aabb right_bounds[kSahBins];
            uint32_t right_counts[kSahBins];
            Aabb acc_b = Aabb::empty();
            uint32_t acc_c = 0;
            for (int b = static_cast<int>(kSahBins) - 1; b >= 0; --b) {
                acc_b.expand(bins[b].bounds);
                acc_c += bins[b].count;
                right_bounds[b] = acc_b;
                right_counts[b] = acc_c;
            }

            Aabb left_b = Aabb::empty();
            uint32_t left_c = 0;
            for (uint32_t b = 0; b + 1u < kSahBins; ++b) {
                left_b.expand(bins[b].bounds);
                left_c += bins[b].count;
                const uint32_t rc = right_counts[b + 1u];
                if (left_c == 0 || rc == 0) continue;
                const float cost =
                    left_b.surface_area() * static_cast<float>(left_c) +
                    right_bounds[b + 1u].surface_area() * static_cast<float>(rc);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_split_bin = b + 1u;
                    best_min_c = c_min;
                    best_scale = scale;
                }
            }
        }
    }

    uint32_t mid = begin + count / 2u;
    if (best_axis >= 0) {
        auto it = std::partition(
            s.prim_indices.begin() + static_cast<ptrdiff_t>(begin),
            s.prim_indices.begin() + static_cast<ptrdiff_t>(end),
            [&](uint32_t p) {
                const float c = axis_of(s.prim_centroids[p], best_axis);
                int b = static_cast<int>((c - best_min_c) * best_scale);
                if (b < 0) b = 0;
                if (b >= static_cast<int>(kSahBins)) b = static_cast<int>(kSahBins) - 1;
                return static_cast<uint32_t>(b) < best_split_bin;
            });
        const uint32_t part = static_cast<uint32_t>(it - s.prim_indices.begin());
        if (part > begin && part < end) {
            mid = part;
        }
    } else {
        const int axis = centroid_bounds.longest_axis();
        std::nth_element(
            s.prim_indices.begin() + static_cast<ptrdiff_t>(begin),
            s.prim_indices.begin() + static_cast<ptrdiff_t>(mid),
            s.prim_indices.begin() + static_cast<ptrdiff_t>(end),
            [&](uint32_t a, uint32_t b) {
                return axis_of(s.prim_centroids[a], axis) < axis_of(s.prim_centroids[b], axis);
            });
    }

    const uint32_t left = build_binary_node(s, begin, mid);
    const uint32_t right = build_binary_node(s, mid, end);

    BuildScratch::BinNode& node = s.bin_nodes[node_idx];
    node.bounds = bounds;
    node.left = left;
    node.right = right;
    node.prim_index = kNoChild;
    node.prim_count = count;
    return node_idx;
}

void write_header(BvhStructure& out) {
    StructureHeader h{};
    h.version_flags = kHeaderVersion | (out.is_tlas ? 0u : kHeaderFlagBottomLevel);
    h.total_slots64 = static_cast<uint32_t>(out.dwords.size() / kNodeGranularityDwords);
    h.box_kind = static_cast<uint32_t>(out.box_format);
    h.element_count = out.element_count;
    h.min_xyz[0] = out.root_bounds.min.x;
    h.min_xyz[1] = out.root_bounds.min.y;
    h.min_xyz[2] = out.root_bounds.min.z;
    h.max_xyz[0] = out.root_bounds.max.x;
    h.max_xyz[1] = out.root_bounds.max.y;
    h.max_xyz[2] = out.root_bounds.max.z;
    h.box_offset_bytes = out.box_offset_dwords * 4u;
    h.box_count = out.box_count;
    h.root_node = out.root_node;
    h.elements_offset_bytes = out.elements_offset_dwords * 4u;
    h.parent_table_offset_bytes = out.parent_table_offset_dwords * 4u;
    h.generation = out.generation;
    std::memcpy(out.dwords.data(), &h, sizeof(h));
}

template <typename PackLeaf>
bool build_4wide_common(uint32_t count, bool is_tlas, BoxFormat format, uint32_t leaf_kind,
                        BuildScratch& scratch, BvhStructure& out, PackLeaf&& pack_leaf) {
    ++scratch.reuse_count;
    out.is_tlas = is_tlas;
    out.box_format = format;
    out.element_count = count;
    out.generation = 1u;

    if (count == 0u) {
        const size_t min_dwords = kStructureAlignment / 4u;
        out.dwords.assign(min_dwords, 0u);
        out.root_node = kNoChild;
        out.box_count = 0u;
        out.box_offset_dwords = kNodeGranularityDwords;
        out.elements_offset_dwords = kNodeGranularityDwords;
        out.parent_table_offset_dwords = 0u;
        out.root_bounds = Aabb::empty();
        write_header(out);
        scratch.peak_bytes = std::max(scratch.peak_bytes, scratch.bytes_allocated());
        scratch.trim_to_budget();
        return true;
    }

    scratch.prim_indices.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        scratch.prim_indices[i] = i;
    }
    scratch.bin_nodes.clear();
    if (scratch.bin_nodes.capacity() < static_cast<size_t>(count) * 2u) {
        scratch.bin_nodes.reserve(static_cast<size_t>(count) * 2u);
    }

    const uint32_t bin_root = build_binary_node(scratch, 0u, count);

    // First pass: collapse binary BVH into 4-wide topology (BFS / queue order in collapse_stack
    // guarantees every parent box index < all of its child box indices for O(N) reverse refit!).
    const uint32_t box_kind = static_cast<uint32_t>(format);
    const uint32_t box_dwords = (format == BoxFormat::Box32) ? kBox32NodeDwords : kBox16NodeDwords;
    const uint32_t box_slots = box_dwords / kNodeGranularityDwords;  // 2 for Box32, 1 for Box16

    // Worst-case 4-wide box count is `count` (exact count computed during collapse).
    struct QBox {
        uint32_t child_bin[4] = {kNoChild, kNoChild, kNoChild, kNoChild};
        uint32_t child_box[4] = {kNoChild, kNoChild, kNoChild, kNoChild};
        bool child_is_leaf[4] = {false, false, false, false};
        uint32_t child_prim[4] = {kNoChild, kNoChild, kNoChild, kNoChild};
        uint32_t parent_box = kNoChild;
    };

    // Reuse collapse_stack as a flat BFS queue of binary roots -> 4-wide box indices.
    scratch.collapse_stack.clear();
    scratch.collapse_stack.push_back({bin_root, 0u});
    // Temporarily store collapsed 4-wide descriptors in a local vector? NO — keep steady-state
    // allocation-free! We can determine exact box_count first, lay out `out.dwords` directly, and
    // fill box nodes in a single BFS pass using `collapse_stack` + `parent_ptrs`!
    // Wait: let's do a 2-step pass using ONLY `scratch.collapse_stack` & `scratch.parent_ptrs`!
    // Step A: count total 4-wide box nodes by running the exact same BFS expansion.
    uint32_t q_head = 0;
    while (q_head < static_cast<uint32_t>(scratch.collapse_stack.size())) {
        const uint32_t bn_idx = scratch.collapse_stack[q_head].bin_node;
        ++q_head;
        const BuildScratch::BinNode& bn = scratch.bin_nodes[bn_idx];
        if (bn.left == kNoChild) {
            continue;  // single-primitive root box holding 1 leaf
        }
        uint32_t ch[4] = {bn.left, bn.right, kNoChild, kNoChild};
        uint32_t n_ch = 2u;
        while (n_ch < 4u) {
            int expand_slot = -1;
            float best_area = -1.0f;
            for (uint32_t k = 0; k < n_ch; ++k) {
                const BuildScratch::BinNode& cb = scratch.bin_nodes[ch[k]];
                if (cb.left != kNoChild) {
                    const float a = cb.bounds.surface_area();
                    if (a > best_area || (a == best_area && expand_slot < 0)) {
                        best_area = a;
                        expand_slot = static_cast<int>(k);
                    }
                }
            }
            if (expand_slot < 0) break;
            const BuildScratch::BinNode& exp = scratch.bin_nodes[ch[expand_slot]];
            ch[expand_slot] = exp.left;
            ch[n_ch++] = exp.right;
        }
        for (uint32_t k = 0; k < n_ch; ++k) {
            if (scratch.bin_nodes[ch[k]].left != kNoChild) {
                const uint32_t next_box = static_cast<uint32_t>(scratch.collapse_stack.size());
                scratch.collapse_stack.push_back({ch[k], next_box});
            }
        }
    }

    const uint32_t box_count = static_cast<uint32_t>(scratch.collapse_stack.size());
    const uint32_t box_slot_base = 1u;  // slot 0 = 64-byte header
    const uint32_t elements_slot_base = box_slot_base + box_count * box_slots;
    const uint32_t parent_slot_base = elements_slot_base + count;
    const uint32_t total_node_slots = parent_slot_base;  // slots 0..parent_slot_base-1
    const uint32_t parent_table_dwords = total_node_slots;
    const uint64_t raw_bytes =
        static_cast<uint64_t>(parent_slot_base) * kNodeGranularityBytes +
        static_cast<uint64_t>(parent_table_dwords) * 4u;
    const uint64_t aligned_bytes = align_up(raw_bytes, kStructureAlignment);
    const size_t total_dwords = static_cast<size_t>(aligned_bytes / 4u);

    out.dwords.resize(total_dwords);
    std::memset(out.dwords.data(), 0, total_dwords * sizeof(uint32_t));
    out.box_count = box_count;
    out.box_offset_dwords = box_slot_base * kNodeGranularityDwords;
    out.elements_offset_dwords = elements_slot_base * kNodeGranularityDwords;
    out.parent_table_offset_dwords = parent_slot_base * kNodeGranularityDwords;
    out.root_node = make_node_ptr(box_slot_base, box_kind);

    // Initialize parent table with kNoChild.
    uint32_t* parent_table = out.dwords.data() + out.parent_table_offset_dwords;
    for (uint32_t i = 0; i < parent_table_dwords; ++i) {
        parent_table[i] = kNoChild;
    }

    // Pack every leaf / instance element at its deterministic original index slot.
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t* dst = out.dwords.data() + out.elements_offset_dwords + static_cast<size_t>(i) * kNodeGranularityDwords;
        pack_leaf(i, dst);
    }

    // Step B: emit all 4-wide box nodes directly into `out.dwords` in exact BFS order.
    // Re-running the deterministic child expansion yields the exact same child box indices.
    uint32_t next_child_box = 1u;
    for (uint32_t bi = 0; bi < box_count; ++bi) {
        const uint32_t bn_idx = scratch.collapse_stack[bi].bin_node;
        const BuildScratch::BinNode& bn = scratch.bin_nodes[bn_idx];
        const uint32_t this_slot = box_slot_base + bi * box_slots;
        const uint32_t this_ptr = make_node_ptr(this_slot, box_kind);

        uint32_t child_ptrs[4] = {kNoChild, kNoChild, kNoChild, kNoChild};
        Aabb child_bounds[4] = {Aabb::empty(), Aabb::empty(), Aabb::empty(), Aabb::empty()};

        if (bn.left == kNoChild) {
            const uint32_t elem_slot = elements_slot_base + bn.prim_index;
            child_ptrs[0] = make_node_ptr(elem_slot, leaf_kind);
            child_bounds[0] = scratch.prim_aabbs[bn.prim_index];
            parent_table[elem_slot] = this_ptr;
        } else {
            uint32_t ch[4] = {bn.left, bn.right, kNoChild, kNoChild};
            uint32_t n_ch = 2u;
            while (n_ch < 4u) {
                int expand_slot = -1;
                float best_area = -1.0f;
                for (uint32_t k = 0; k < n_ch; ++k) {
                    const BuildScratch::BinNode& cb = scratch.bin_nodes[ch[k]];
                    if (cb.left != kNoChild) {
                        const float a = cb.bounds.surface_area();
                        if (a > best_area || (a == best_area && expand_slot < 0)) {
                            best_area = a;
                            expand_slot = static_cast<int>(k);
                        }
                    }
                }
                if (expand_slot < 0) break;
                const BuildScratch::BinNode& exp = scratch.bin_nodes[ch[expand_slot]];
                ch[expand_slot] = exp.left;
                ch[n_ch++] = exp.right;
            }
            for (uint32_t k = 0; k < n_ch; ++k) {
                const BuildScratch::BinNode& cb = scratch.bin_nodes[ch[k]];
                child_bounds[k] = cb.bounds;
                if (cb.left == kNoChild) {
                    const uint32_t elem_slot = elements_slot_base + cb.prim_index;
                    child_ptrs[k] = make_node_ptr(elem_slot, leaf_kind);
                    parent_table[elem_slot] = this_ptr;
                } else {
                    const uint32_t c_box = next_child_box++;
                    const uint32_t c_slot = box_slot_base + c_box * box_slots;
                    child_ptrs[k] = make_node_ptr(c_slot, box_kind);
                    parent_table[c_slot] = this_ptr;
                }
            }
        }

        uint32_t* box_dst = out.dwords.data() + out.box_offset_dwords + static_cast<size_t>(bi) * box_dwords;
        if (format == BoxFormat::Box32) {
            pack_box32_node(child_ptrs, child_bounds, box_dst);
        } else {
            pack_box16_node(child_ptrs, child_bounds, box_dst);
        }
    }

    // For Box16, tighten/propagate outward-quantized bounds bottom-up so parent f16 boxes strictly
    // enclose their children's quantized f16 boxes (guaranteeing conservative hierarchy!).
    if (format == BoxFormat::Box16) {
        // Bottom-up pass reading child boxes directly from `out.dwords` (zero extra memory!).
        for (int bi = static_cast<int>(box_count) - 1; bi >= 0; --bi) {
            uint32_t* box_dst =
                out.dwords.data() + out.box_offset_dwords + static_cast<size_t>(bi) * kBox16NodeDwords;
            uint32_t child_ptrs[4];
            Aabb child_bounds[4];
            for (uint32_t k = 0; k < 4u; ++k) {
                child_ptrs[k] = box_dst[k];
                if (child_ptrs[k] == kNoChild) {
                    child_bounds[k] = Aabb::empty();
                    continue;
                }
                const uint32_t c_kind = node_ptr_kind(child_ptrs[k]);
                const uint32_t c_slot = node_ptr_slot(child_ptrs[k]);
                if (c_kind == kNodeKindBox16) {
                    const uint32_t* cb_words = out.dwords.data() + static_cast<size_t>(c_slot) * kNodeGranularityDwords;
                    Aabb merged = Aabb::empty();
                    for (uint32_t m = 0; m < 4u; ++m) {
                        if (cb_words[m] == kNoChild) continue;
                        const uint32_t w0 = cb_words[4u + 3u * m + 0u];
                        const uint32_t w1 = cb_words[4u + 3u * m + 1u];
                        const uint32_t w2 = cb_words[4u + 3u * m + 2u];
                        Aabb b;
                        b.min = {half_f(w0 & 0xffffu), half_f(w0 >> 16), half_f(w1 & 0xffffu)};
                        b.max = {half_f(w1 >> 16), half_f(w2 & 0xffffu), half_f(w2 >> 16)};
                        merged.expand(b);
                    }
                    child_bounds[k] = merged;
                } else {
                    const uint32_t orig_idx = c_slot - elements_slot_base;
                    child_bounds[k] = scratch.prim_aabbs[orig_idx];
                }
            }
            Aabb quantized[4];
            pack_box16_node(child_ptrs, child_bounds, box_dst, quantized);
            if (bi == 0) {
                Aabb root_b = Aabb::empty();
                for (uint32_t k = 0; k < 4u; ++k) {
                    if (child_ptrs[k] != kNoChild) root_b.expand(quantized[k]);
                }
                out.root_bounds = root_b;
            }
        }
    } else {
        out.root_bounds = scratch.bin_nodes[bin_root].bounds;
    }

    write_header(out);
    scratch.peak_bytes = std::max(scratch.peak_bytes, scratch.bytes_allocated());
    scratch.trim_to_budget();
    return true;
}

bool refit_4wide_common(uint32_t count, bool is_tlas, RefitScratch& scratch, BvhStructure& bvh) {
    if (bvh.is_tlas != is_tlas || bvh.element_count != count) {
        return false;
    }
    ++scratch.reuse_count;
    if (count == 0u) {
        ++bvh.generation;
        bvh.root_bounds = Aabb::empty();
        if (bvh.dwords.size() >= kNodeGranularityDwords) {
            write_header(bvh);
        }
        scratch.peak_bytes = std::max(scratch.peak_bytes, scratch.bytes_allocated());
        scratch.trim_to_budget();
        return true;
    }
    if (bvh.box_count == 0u || bvh.dwords.empty()) {
        return false;
    }

    const uint32_t box_kind = static_cast<uint32_t>(bvh.box_format);
    const uint32_t box_dwords = (bvh.box_format == BoxFormat::Box32) ? kBox32NodeDwords : kBox16NodeDwords;
    const uint32_t box_slots = box_dwords / kNodeGranularityDwords;
    const uint32_t box_slot_base = bvh.box_offset_dwords / kNodeGranularityDwords;
    const uint32_t elements_slot_base = bvh.elements_offset_dwords / kNodeGranularityDwords;

    scratch.box_aabbs.resize(bvh.box_count);

    // Reverse topological pass: since parent index < child index for every box node, visiting
    // `box_count - 1` down to `0` computes exact bottom-up bounds in a single linear pass.
    for (int bi = static_cast<int>(bvh.box_count) - 1; bi >= 0; --bi) {
        uint32_t* box_dst =
            bvh.dwords.data() + bvh.box_offset_dwords + static_cast<size_t>(bi) * box_dwords;
        uint32_t child_ptrs[4];
        Aabb child_bounds[4];
        Aabb merged = Aabb::empty();

        for (uint32_t k = 0; k < 4u; ++k) {
            child_ptrs[k] = box_dst[k];
            if (child_ptrs[k] == kNoChild) {
                child_bounds[k] = Aabb::empty();
                continue;
            }
            const uint32_t c_kind = node_ptr_kind(child_ptrs[k]);
            const uint32_t c_slot = node_ptr_slot(child_ptrs[k]);
            if (c_kind == box_kind) {
                const uint32_t c_box = (c_slot - box_slot_base) / box_slots;
                if (c_box >= bvh.box_count) return false;
                child_bounds[k] = scratch.box_aabbs[c_box];
            } else {
                const uint32_t orig_idx = c_slot - elements_slot_base;
                if (orig_idx >= count) return false;
                child_bounds[k] = scratch.leaf_aabbs[orig_idx];
            }
        }

        if (bvh.box_format == BoxFormat::Box32) {
            pack_box32_node(child_ptrs, child_bounds, box_dst);
            for (uint32_t k = 0; k < 4u; ++k) {
                if (child_ptrs[k] != kNoChild) merged.expand(child_bounds[k]);
            }
        } else {
            Aabb quantized[4];
            pack_box16_node(child_ptrs, child_bounds, box_dst, quantized);
            for (uint32_t k = 0; k < 4u; ++k) {
                if (child_ptrs[k] != kNoChild) merged.expand(quantized[k]);
            }
        }
        scratch.box_aabbs[static_cast<size_t>(bi)] = merged;
    }

    bvh.root_bounds = scratch.box_aabbs[0];
    ++bvh.generation;
    write_header(bvh);
    scratch.peak_bytes = std::max(scratch.peak_bytes, scratch.bytes_allocated());
    scratch.trim_to_budget();
    return true;
}

}  // namespace

Result intersect(const uint32_t desc[4], uint64_t node, float extent, Vec3 origin, Vec3 dir, Vec3 inv_dir,
                 const LoadU32& load) {
    const uint32_t kind = static_cast<uint32_t>(node & 7u);
    const float o[3] = {origin.x, origin.y, origin.z};
    const float d[3] = {dir.x, dir.y, dir.z};
    const float inv[3] = {inv_dir.x, inv_dir.y, inv_dir.z};
    const bool nan_ray = isnan_(extent) || isnan_(o[0]) || isnan_(o[1]) || isnan_(o[2]) || isnan_(d[0]) ||
                         isnan_(d[1]) || isnan_(d[2]) || isnan_(inv[0]) || isnan_(inv[1]) || isnan_(inv[2]);
    const uint64_t base = ((static_cast<uint64_t>(desc[1] & 0xffu) << 32) | desc[0]) << 8;
    const uint64_t addr = (base + ((node >> 3) << 6)) & kAddressMask;
    const bool sort = (desc[1] >> 31) != 0;

    if (kind >= 6) {
        return miss_triangle();
    }
    if (kind < 4) {
        if (nan_ray) {
            return miss_triangle();
        }
        uint32_t n[16];
        for (uint32_t i = 0; i < 16; ++i) {
            n[i] = load((addr + 4ull * i) & kAddressMask);
        }
        return triangle(n, kind, o, d);
    }
    Result out = {{kNoChild, kNoChild, kNoChild, kNoChild}};
    if (nan_ray) {
        return out;
    }
    uint32_t ptr[4];
    float key[4];
    bool hit[4];
    for (uint32_t i = 0; i < 4; ++i) {
        ptr[i] = load((addr + 4ull * i) & kAddressMask);
        float mn[3], mx[3];
        if (kind == 5) {
            for (uint32_t c = 0; c < 3; ++c) {
                mn[c] = bits_f(load((addr + 16 + 24ull * i + 4ull * c) & kAddressMask));
                mx[c] = bits_f(load((addr + 16 + 24ull * i + 12 + 4ull * c) & kAddressMask));
            }
        } else {
            const uint32_t a = load((addr + 16 + 12ull * i) & kAddressMask);
            const uint32_t b = load((addr + 16 + 12ull * i + 4) & kAddressMask);
            const uint32_t c = load((addr + 16 + 12ull * i + 8) & kAddressMask);
            mn[0] = half_f(a & 0xffffu);
            mn[1] = half_f(a >> 16);
            mn[2] = half_f(b & 0xffffu);
            mx[0] = half_f(b >> 16);
            mx[1] = half_f(c & 0xffffu);
            mx[2] = half_f(c >> 16);
        }
        key[i] = 0.0f;
        hit[i] = ptr[i] != kNoChild && slab(mn, mx, o, inv, extent, &key[i]);
    }
    for (uint32_t i = 0; i < 4; ++i) {
        if (!hit[i]) continue;
        uint32_t rank = i;
        if (sort) {
            rank = 0;
            for (uint32_t j = 0; j < 4; ++j) {
                if (hit[j] && (key[j] < key[i] || (key[j] == key[i] && j < i))) ++rank;
            }
        }
        out.w[rank] = ptr[i];
    }
    return out;
}

const char* prelude_wgsl() {
    return R"WGSL(// pg6 runtime prelude, version 1 (rt_bvh.h kPreludeVersion). Group 3, bindings 16-21: contract C3.
@group(3) @binding(16) var<storage, read_write> pg6_arena0: array<u32>;
@group(3) @binding(17) var<storage, read_write> pg6_arena1: array<u32>;
@group(3) @binding(18) var<storage, read_write> pg6_arena2: array<u32>;
@group(3) @binding(19) var<storage, read_write> pg6_arena3: array<u32>;
@group(3) @binding(20) var<storage, read> pg6_arena_dir: array<u32>;
@group(3) @binding(21) var<storage, read_write> pg6_rt_stats: array<atomic<u32>>;

// A non-resident access: bumps rt_fault (word 0) and logs the page for the runtime to make resident
// (word 1 = log attempts, words 2..31 = faulting page + 1, first 30 only). rt_residency.h FaultTracker.
fn pg6_fault(page: u32) {
    atomicAdd(&pg6_rt_stats[0], 1u);
    let i = atomicAdd(&pg6_rt_stats[1], 1u);
    if (i < 30u) {
        atomicStore(&pg6_rt_stats[2u + i], page + 1u);
    }
}

// Dword at a 40-bit address (x = low 32 bits, y = bits 39:32). Unmapped: counts a fault, reads 0.
fn pg6_load32(addr: vec2<u32>) -> u32 {
    let page = (addr.x >> 16u) | ((addr.y & 0xffu) << 16u);
    let t = pg6_arena_dir[page >> 12u];
    if (t == 0u) {
        pg6_fault(page);
        return 0u;
    }
    let e = pg6_arena_dir[4096u + (t - 1u) * 4096u + (page & 0xfffu)];
    if ((e & 0x80000000u) == 0u) {
        pg6_fault(page);
        return 0u;
    }
    let w = (e & 0x1fffffffu) * 16384u + ((addr.x & 0xffffu) >> 2u);
    switch ((e >> 29u) & 3u) {
        case 0u: { return pg6_arena0[w]; }
        case 1u: { return pg6_arena1[w]; }
        case 2u: { return pg6_arena2[w]; }
        default: { return pg6_arena3[w]; }
    }
}

fn pg6_add40(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    var carry = 0u;
    if (lo < a.x) { carry = 1u; }
    return vec2<u32>(lo, (a.y + b.y + carry) & 0xffu);
}

fn pg6_addr_off(a: vec2<u32>, off: u32) -> vec2<u32> {
    return pg6_add40(a, vec2<u32>(off, 0u));
}

fn pg6_f(a: vec2<u32>, off: u32) -> f32 {
    return bitcast<f32>(pg6_load32(pg6_addr_off(a, off)));
}

// Slab test of one box: .x = 1.0 when entered, .y = the entry distance.
fn pg6_slab(mn: vec3<f32>, mx: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, extent: f32) -> vec2<f32> {
    var lo = 0.0;
    var hi = extent;
    for (var a = 0u; a < 3u; a = a + 1u) {
        if (mn[a] > mx[a]) {
            return vec2<f32>(0.0, 0.0);
        }
        let t0 = (mn[a] - o[a]) * inv[a];
        let t1 = (mx[a] - o[a]) * inv[a];
        if (t0 != t0 || t1 != t1) {
            continue;
        }
        var a0 = t1;
        var a1 = t0;
        if (t0 < t1) {
            a0 = t0;
            a1 = t1;
        }
        if (a0 > lo) { lo = a0; }
        if (a1 < hi) { hi = a1; }
    }
    if (lo <= hi) {
        return vec2<f32>(1.0, lo);
    }
    return vec2<f32>(0.0, 0.0);
}

fn pg6_tri_miss() -> vec4<u32> {
    return vec4<u32>(0x7f800000u, 0x3f800000u, 0u, 0u);
}

fn pg6_vec_nan(v: vec3<f32>) -> bool {
    return v.x != v.x || v.y != v.y || v.z != v.z;
}

fn pg6_triangle(addr: vec2<u32>, kind: u32, o: vec3<f32>, d: vec3<f32>) -> vec4<u32> {
    var vi = vec3<u32>(0u, 1u, 2u);
    if (kind == 1u) { vi = vec3<u32>(1u, 3u, 2u); }
    if (kind == 2u) { vi = vec3<u32>(2u, 3u, 4u); }
    if (kind == 3u) { vi = vec3<u32>(2u, 4u, 0u); }
    var v: array<vec3<f32>, 3>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let b = vi[i] * 12u;
        v[i] = vec3<f32>(pg6_f(addr, b), pg6_f(addr, b + 4u), pg6_f(addr, b + 8u)) - o;
    }
    var kz = 0u;
    var az = abs(d.x);
    if (abs(d.y) > az) { kz = 1u; az = abs(d.y); }
    if (abs(d.z) > az) { kz = 2u; }
    let dz = d[kz];
    if (dz == 0.0) {
        return pg6_tri_miss();
    }
    var kx = (kz + 1u) % 3u;
    var ky = (kx + 1u) % 3u;
    if (dz < 0.0) {
        let t = kx;
        kx = ky;
        ky = t;
    }
    let sx = d[kx] / dz;
    let sy = d[ky] / dz;
    let sz = 1.0 / dz;
    var px: array<f32, 3>;
    var py: array<f32, 3>;
    var pz: array<f32, 3>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        px[i] = v[i][kx] - sx * v[i][kz];
        py[i] = v[i][ky] - sy * v[i][kz];
        pz[i] = sz * v[i][kz];
    }
    let u = px[2] * py[1] - py[2] * px[1];
    let vv = px[0] * py[2] - py[0] * px[2];
    let w = px[1] * py[0] - py[1] * px[0];
    if ((u < 0.0 || vv < 0.0 || w < 0.0) && (u > 0.0 || vv > 0.0 || w > 0.0)) {
        return pg6_tri_miss();
    }
    let det = u + vv + w;
    let t_num = u * pz[0] + vv * pz[1] + w * pz[2];
    if (det == 0.0 || det != det || t_num != t_num || u != u || vv != vv || w != w) {
        return pg6_tri_miss();
    }
    let sel = (pg6_load32(pg6_addr_off(addr, 60u)) >> (8u * kind)) & 0xffu;
    var wt = array<f32, 4>(u, vv, w, 0.0);
    return vec4<u32>(bitcast<u32>(t_num), bitcast<u32>(det), bitcast<u32>(wt[sel & 3u]),
                     bitcast<u32>(wt[(sel >> 2u) & 3u]));
}

// C4: the console's image_bvh(64)_intersect_ray node test. desc = the 128-bit BVH descriptor,
// node = the node pointer (x low, y high; high 0 for the 32-bit form).
fn pg6_bvh_intersect(desc: vec4<u32>, node: vec2<u32>, extent: f32, origin: vec3<f32>, dir: vec3<f32>,
                     inv_dir: vec3<f32>) -> vec4<u32> {
    let kind = node.x & 7u;
    if (kind >= 6u) {
        return pg6_tri_miss();
    }
    let nan_ray = extent != extent || pg6_vec_nan(origin) || pg6_vec_nan(dir) || pg6_vec_nan(inv_dir);
    // address = (base256 << 8) + ((ptr >> 3) << 6), 40 bits
    let base = vec2<u32>(desc.x << 8u, ((desc.y & 0xffu) << 8u) | (desc.x >> 24u));
    let idx = vec2<u32>((node.x >> 3u) | (node.y << 29u), node.y >> 3u);
    let off = vec2<u32>(idx.x << 6u, (idx.y << 6u) | (idx.x >> 26u));
    let addr = pg6_add40(base, off);
    if (kind < 4u) {
        if (nan_ray) {
            return pg6_tri_miss();
        }
        return pg6_triangle(addr, kind, origin, dir);
    }
    if (nan_ray) {
        return vec4<u32>(0xffffffffu);
    }
    var ptr: array<u32, 4>;
    var key: array<f32, 4>;
    var hit: array<bool, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        ptr[i] = pg6_load32(pg6_addr_off(addr, 4u * i));
        var mn: vec3<f32>;
        var mx: vec3<f32>;
        if (kind == 5u) {
            let b = 16u + 24u * i;
            mn = vec3<f32>(pg6_f(addr, b), pg6_f(addr, b + 4u), pg6_f(addr, b + 8u));
            mx = vec3<f32>(pg6_f(addr, b + 12u), pg6_f(addr, b + 16u), pg6_f(addr, b + 20u));
        } else {
            let b = 16u + 12u * i;
            let a = unpack2x16float(pg6_load32(pg6_addr_off(addr, b)));
            let bb = unpack2x16float(pg6_load32(pg6_addr_off(addr, b + 4u)));
            let c = unpack2x16float(pg6_load32(pg6_addr_off(addr, b + 8u)));
            mn = vec3<f32>(a.x, a.y, bb.x);
            mx = vec3<f32>(bb.y, c.x, c.y);
        }
        key[i] = 0.0;
        hit[i] = false;
        if (ptr[i] != 0xffffffffu) {
            let s = pg6_slab(mn, mx, origin, inv_dir, extent);
            hit[i] = s.x > 0.5;
            key[i] = s.y;
        }
    }
    var out = vec4<u32>(0xffffffffu);
    let sort = (desc.y >> 31u) != 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (!hit[i]) {
            continue;
        }
        var rank = i;
        if (sort) {
            rank = 0u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                if (hit[j] && (key[j] < key[i] || (key[j] == key[i] && j < i))) {
                    rank = rank + 1u;
                }
            }
        }
        out[rank] = ptr[i];
    }
    return out;
}
)WGSL";
}

// -----------------------------------------------------------------------------
// Outward-rounded f16 helpers, 3x4 affine math, node packing, BLAS/TLAS builders
// -----------------------------------------------------------------------------

float half_to_float(uint16_t h) { return half_f(h); }

uint16_t float_to_half_floor(float x) {
    if (isnan_(x)) return 0x7e00u;
    if (std::isinf(x)) return x < 0.0f ? 0xfc00u : 0x7c00u;
    if (x < -65504.0f) return 0xfc00u;  // -inf
    if (x > 65504.0f) return 0x7bffu;   // +65504.0 (largest finite f16 <= x)
    uint16_t h = f32_to_f16_rne(x);
    if (half_f(h) > x) {
        if ((h & 0x7fffu) == 0u) {
            h = 0x8001u;  // next smaller below +/-0 is -smallest_subnormal
        } else if ((h & 0x8000u) == 0u) {
            --h;
        } else {
            ++h;
        }
    }
    return h;
}

uint16_t float_to_half_ceil(float x) {
    if (isnan_(x)) return 0x7e00u;
    return static_cast<uint16_t>(float_to_half_floor(-x) ^ 0x8000u);
}

void make_bvh_desc(uint64_t base_addr, bool sort_boxes, uint32_t out_desc[4]) {
    const uint64_t masked = base_addr & kAddressMask;
    const uint64_t base256 = masked >> 8;
    out_desc[0] = static_cast<uint32_t>(base256 & 0xFFFFFFFFull);
    out_desc[1] = static_cast<uint32_t>((base256 >> 32) & 0xFFull) | (sort_boxes ? 0x80000000u : 0u);
    out_desc[2] = 0u;
    out_desc[3] = kDescriptorTypeBvh | (1u << 24);  // type 8, return barycentrics
}

bool invert_affine_3x4(const float m[3][4], float out_inv[3][4]) {
    const double a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], tx = m[0][3];
    const double a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], ty = m[1][3];
    const double a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], tz = m[2][3];

    const double c00 = a11 * a22 - a12 * a21;
    const double c01 = a12 * a20 - a10 * a22;
    const double c02 = a10 * a21 - a11 * a20;
    const double det = a00 * c00 + a01 * c01 + a02 * c02;
    if (!(std::fabs(det) > 1e-12) || std::isnan(det) || std::isinf(det)) {
        return false;
    }
    const double inv_det = 1.0 / det;
    const double r00 = c00 * inv_det;
    const double r01 = (a02 * a21 - a01 * a22) * inv_det;
    const double r02 = (a01 * a12 - a02 * a11) * inv_det;

    const double r10 = c01 * inv_det;
    const double r11 = (a00 * a22 - a02 * a20) * inv_det;
    const double r12 = (a02 * a10 - a00 * a12) * inv_det;

    const double r20 = c02 * inv_det;
    const double r21 = (a01 * a20 - a00 * a21) * inv_det;
    const double r22 = (a00 * a11 - a01 * a10) * inv_det;

    out_inv[0][0] = static_cast<float>(r00);
    out_inv[0][1] = static_cast<float>(r01);
    out_inv[0][2] = static_cast<float>(r02);
    out_inv[0][3] = static_cast<float>(-(r00 * tx + r01 * ty + r02 * tz));

    out_inv[1][0] = static_cast<float>(r10);
    out_inv[1][1] = static_cast<float>(r11);
    out_inv[1][2] = static_cast<float>(r12);
    out_inv[1][3] = static_cast<float>(-(r10 * tx + r11 * ty + r12 * tz));

    out_inv[2][0] = static_cast<float>(r20);
    out_inv[2][1] = static_cast<float>(r21);
    out_inv[2][2] = static_cast<float>(r22);
    out_inv[2][3] = static_cast<float>(-(r20 * tx + r21 * ty + r22 * tz));
    return true;
}

Vec3 transform_point_3x4(const float m[3][4], Vec3 p) {
    return {
        m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
        m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
        m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3],
    };
}

Vec3 transform_vector_3x4(const float m[3][4], Vec3 v) {
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    };
}

Aabb transform_aabb_3x4(const Aabb& local, const float m[3][4]) {
    if (local.is_empty()) return Aabb::empty();
    const float mn[3] = {local.min.x, local.min.y, local.min.z};
    const float mx[3] = {local.max.x, local.max.y, local.max.z};
    float out_mn[3] = {m[0][3], m[1][3], m[2][3]};
    float out_mx[3] = {m[0][3], m[1][3], m[2][3]};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            const float a = m[r][c] * mn[c];
            const float b = m[r][c] * mx[c];
            out_mn[r] += a < b ? a : b;
            out_mx[r] += a < b ? b : a;
        }
    }
    return {{out_mn[0], out_mn[1], out_mn[2]}, {out_mx[0], out_mx[1], out_mx[2]}};
}

void pack_triangle_node(const Triangle& tri, uint32_t out_dwords[16]) {
    out_dwords[0] = f_bits(tri.v0.x);
    out_dwords[1] = f_bits(tri.v0.y);
    out_dwords[2] = f_bits(tri.v0.z);
    out_dwords[3] = f_bits(tri.v1.x);
    out_dwords[4] = f_bits(tri.v1.y);
    out_dwords[5] = f_bits(tri.v1.z);
    out_dwords[6] = f_bits(tri.v2.x);
    out_dwords[7] = f_bits(tri.v2.y);
    out_dwords[8] = f_bits(tri.v2.z);
    // Unused vertices 3 and 4 carry primitive/geometry metadata (docs/RAYTRACING_DESIGN.md §2).
    out_dwords[9] = tri.primitive_id;
    out_dwords[10] = tri.geometry_id;
    out_dwords[11] = 0u;
    out_dwords[12] = 0u;
    out_dwords[13] = 0u;
    out_dwords[14] = 0u;
    out_dwords[15] = kDefaultTriSelector;
}

void pack_triangle_node_multi(const Vec3 verts[5], uint32_t selector, uint32_t out_dwords[16]) {
    for (uint32_t i = 0; i < 5u; ++i) {
        out_dwords[i * 3u + 0u] = f_bits(verts[i].x);
        out_dwords[i * 3u + 1u] = f_bits(verts[i].y);
        out_dwords[i * 3u + 2u] = f_bits(verts[i].z);
    }
    out_dwords[15] = selector;
}

void pack_box32_node(const uint32_t child_ptrs[4], const Aabb child_bounds[4], uint32_t out_dwords[32]) {
    std::memset(out_dwords, 0, 32u * sizeof(uint32_t));
    for (uint32_t i = 0; i < 4u; ++i) {
        out_dwords[i] = child_ptrs[i];
        const bool active = child_ptrs[i] != kNoChild && !child_bounds[i].is_empty();
        const float mn[3] = {
            active ? child_bounds[i].min.x : 1.0f,
            active ? child_bounds[i].min.y : 1.0f,
            active ? child_bounds[i].min.z : 1.0f,
        };
        const float mx[3] = {
            active ? child_bounds[i].max.x : -1.0f,
            active ? child_bounds[i].max.y : -1.0f,
            active ? child_bounds[i].max.z : -1.0f,
        };
        for (uint32_t c = 0; c < 3u; ++c) {
            out_dwords[4u + 6u * i + c] = f_bits(mn[c]);
            out_dwords[4u + 6u * i + 3u + c] = f_bits(mx[c]);
        }
    }
}

void pack_box16_node(const uint32_t child_ptrs[4], const Aabb child_bounds[4], uint32_t out_dwords[16],
                     Aabb quantized_out[4]) {
    for (uint32_t i = 0; i < 4u; ++i) {
        out_dwords[i] = child_ptrs[i];
        const bool active = child_ptrs[i] != kNoChild && !child_bounds[i].is_empty();
        uint16_t h_mn[3];
        uint16_t h_mx[3];
        if (active) {
            h_mn[0] = float_to_half_floor(child_bounds[i].min.x);
            h_mn[1] = float_to_half_floor(child_bounds[i].min.y);
            h_mn[2] = float_to_half_floor(child_bounds[i].min.z);
            h_mx[0] = float_to_half_ceil(child_bounds[i].max.x);
            h_mx[1] = float_to_half_ceil(child_bounds[i].max.y);
            h_mx[2] = float_to_half_ceil(child_bounds[i].max.z);
            if (quantized_out != nullptr) {
                quantized_out[i].min = {half_f(h_mn[0]), half_f(h_mn[1]), half_f(h_mn[2])};
                quantized_out[i].max = {half_f(h_mx[0]), half_f(h_mx[1]), half_f(h_mx[2])};
            }
        } else {
            // Finite inverted bounds (+1 > -1) so slab() immediately rejects empty slots.
            h_mn[0] = h_mn[1] = h_mn[2] = 0x3c00u;  // +1.0
            h_mx[0] = h_mx[1] = h_mx[2] = 0xbc00u;  // -1.0
            if (quantized_out != nullptr) {
                quantized_out[i] = Aabb::empty();
            }
        }
        out_dwords[4u + 3u * i + 0u] = static_cast<uint32_t>(h_mn[0]) | (static_cast<uint32_t>(h_mn[1]) << 16);
        out_dwords[4u + 3u * i + 1u] = static_cast<uint32_t>(h_mn[2]) | (static_cast<uint32_t>(h_mx[0]) << 16);
        out_dwords[4u + 3u * i + 2u] = static_cast<uint32_t>(h_mx[1]) | (static_cast<uint32_t>(h_mx[2]) << 16);
    }
}

bool pack_instance_node(const InstanceDesc& inst, uint32_t out_dwords[16]) {
    if ((inst.blas_address & ~kAddressMask) != 0 || (inst.blas_address & 0xFFull) != 0) {
        return false;
    }
    float inv[3][4];
    if (inst.has_inv_transform) {
        std::memcpy(inv, inst.inv_transform, sizeof(inv));
    } else if (!invert_affine_3x4(inst.transform, inv)) {
        // Singular transform: pack zeroed inactive instance (mask = 0, root = kNoChild).
        std::memset(out_dwords, 0, 16u * sizeof(uint32_t));
        out_dwords[15] = kNoChild;
        return false;
    }
    for (uint32_t r = 0; r < 3u; ++r) {
        for (uint32_t c = 0; c < 4u; ++c) {
            out_dwords[r * 4u + c] = f_bits(inv[r][c]);
        }
    }
    out_dwords[12] = (inst.instance_id & 0xFFFFFFu) | (static_cast<uint32_t>(inst.instance_mask) << 24);
    out_dwords[13] = (inst.hit_group_contribution & 0xFFFFFFu) | (static_cast<uint32_t>(inst.flags) << 24);
    out_dwords[14] = static_cast<uint32_t>((inst.blas_address & kAddressMask) >> 8);
    out_dwords[15] = inst.blas_root_node;
    return true;
}

PackedInstanceView unpack_instance_node(const uint32_t in_dwords[16]) {
    PackedInstanceView v{};
    for (uint32_t r = 0; r < 3u; ++r) {
        for (uint32_t c = 0; c < 4u; ++c) {
            v.inv_transform[r][c] = bits_f(in_dwords[r * 4u + c]);
        }
    }
    v.instance_id = in_dwords[12] & 0xFFFFFFu;
    v.instance_mask = static_cast<uint8_t>((in_dwords[12] >> 24) & 0xFFu);
    v.hit_group_contribution = in_dwords[13] & 0xFFFFFFu;
    v.flags = static_cast<uint8_t>((in_dwords[13] >> 24) & 0xFFu);
    v.blas_address = (static_cast<uint64_t>(in_dwords[14]) << 8) & kAddressMask;
    v.blas_root_node = in_dwords[15];
    return v;
}

PrebuildInfo compute_prebuild_info(uint32_t primitive_count, BoxFormat format, bool /*is_tlas*/) {
    PrebuildInfo info{};
    if (primitive_count == 0u) {
        info.result_bytes = kStructureAlignment;
        info.build_scratch_bytes = kStructureAlignment;
        info.update_scratch_bytes = kStructureAlignment;
        info.max_box_nodes = 0u;
        info.max_leaf_nodes = 0u;
        return info;
    }
    const uint32_t max_boxes = primitive_count == 1u ? 1u : primitive_count - 1u;
    const uint64_t box_bytes = (format == BoxFormat::Box32) ? 128ull : 64ull;
    const uint64_t box_slots = box_bytes / kNodeGranularityBytes;
    const uint64_t total_slots = 1ull + static_cast<uint64_t>(max_boxes) * box_slots + primitive_count;
    const uint64_t parent_bytes = total_slots * 4ull;
    const uint64_t raw_result = total_slots * kNodeGranularityBytes + parent_bytes;

    info.max_box_nodes = max_boxes;
    info.max_leaf_nodes = primitive_count;
    info.result_bytes = std::max<uint64_t>(kStructureAlignment, align_up(raw_result, kStructureAlignment));

    const uint64_t build_raw =
        static_cast<uint64_t>(primitive_count) * (sizeof(Aabb) + sizeof(Vec3) + sizeof(uint32_t)) +
        static_cast<uint64_t>(primitive_count) * 2ull * sizeof(BuildScratch::BinNode) +
        static_cast<uint64_t>(max_boxes) * sizeof(BuildScratch::CollapseTask);
    const uint64_t update_raw = (static_cast<uint64_t>(primitive_count) + max_boxes) * sizeof(Aabb);
    info.build_scratch_bytes = std::max<uint64_t>(kStructureAlignment, align_up(build_raw, kStructureAlignment));
    info.update_scratch_bytes = std::max<uint64_t>(kStructureAlignment, align_up(update_raw, kStructureAlignment));
    return info;
}

StructureHeader BvhStructure::header() const {
    StructureHeader h{};
    if (dwords.size() >= kNodeGranularityDwords) {
        std::memcpy(&h, dwords.data(), sizeof(h));
    }
    return h;
}

void BuildScratch::reserve(uint32_t max_primitives) {
    prim_aabbs.reserve(max_primitives);
    prim_centroids.reserve(max_primitives);
    prim_indices.reserve(max_primitives);
    bin_nodes.reserve(static_cast<size_t>(max_primitives) * 2u);
    collapse_stack.reserve(max_primitives > 0u ? max_primitives : 1u);
    peak_bytes = std::max(peak_bytes, bytes_allocated());
}

void BuildScratch::set_budget_bytes(size_t bytes) {
    budget_bytes = bytes;
    trim_to_budget();
}

size_t BuildScratch::bytes_allocated() const {
    return prim_aabbs.capacity() * sizeof(Aabb) + prim_centroids.capacity() * sizeof(Vec3) +
           prim_indices.capacity() * sizeof(uint32_t) + bin_nodes.capacity() * sizeof(BinNode) +
           collapse_stack.capacity() * sizeof(CollapseTask) + parent_ptrs.capacity() * sizeof(uint32_t);
}

void BuildScratch::trim_to_budget() {
    if (budget_bytes == 0 || bytes_allocated() <= budget_bytes) return;
    prim_aabbs.clear();
    prim_aabbs.shrink_to_fit();
    prim_centroids.clear();
    prim_centroids.shrink_to_fit();
    prim_indices.clear();
    prim_indices.shrink_to_fit();
    bin_nodes.clear();
    bin_nodes.shrink_to_fit();
    collapse_stack.clear();
    collapse_stack.shrink_to_fit();
    parent_ptrs.clear();
    parent_ptrs.shrink_to_fit();
    ++budget_trims;
}

void RefitScratch::reserve(uint32_t max_primitives, uint32_t max_boxes) {
    leaf_aabbs.reserve(max_primitives);
    box_aabbs.reserve(max_boxes);
    peak_bytes = std::max(peak_bytes, bytes_allocated());
}

void RefitScratch::set_budget_bytes(size_t bytes) {
    budget_bytes = bytes;
    trim_to_budget();
}

size_t RefitScratch::bytes_allocated() const {
    return leaf_aabbs.capacity() * sizeof(Aabb) + box_aabbs.capacity() * sizeof(Aabb);
}

void RefitScratch::trim_to_budget() {
    if (budget_bytes == 0 || bytes_allocated() <= budget_bytes) return;
    leaf_aabbs.clear();
    leaf_aabbs.shrink_to_fit();
    box_aabbs.clear();
    box_aabbs.shrink_to_fit();
    ++budget_trims;
}

bool build_blas(const Triangle* triangles, uint32_t count, BoxFormat format, BuildScratch& scratch,
                BvhStructure& out) {
    if (count > 0u && triangles == nullptr) return false;
    scratch.prim_aabbs.resize(count);
    scratch.prim_centroids.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        const Aabb b = triangle_aabb(triangles[i]);
        scratch.prim_aabbs[i] = b;
        scratch.prim_centroids[i] = b.centroid();
    }
    return build_4wide_common(count, false, format, kNodeKindTri0, scratch, out,
                              [triangles](uint32_t i, uint32_t* dst16) { pack_triangle_node(triangles[i], dst16); });
}

bool refit_blas(const Triangle* triangles, uint32_t count, RefitScratch& scratch, BvhStructure& blas) {
    if (count > 0u && triangles == nullptr) return false;
    if (blas.is_tlas || blas.element_count != count) return false;
    scratch.leaf_aabbs.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        scratch.leaf_aabbs[i] = triangle_aabb(triangles[i]);
        uint32_t* dst =
            blas.dwords.data() + blas.elements_offset_dwords + static_cast<size_t>(i) * kNodeGranularityDwords;
        pack_triangle_node(triangles[i], dst);
    }
    return refit_4wide_common(count, false, scratch, blas);
}

bool build_tlas(const InstanceDesc* instances, uint32_t count, BoxFormat format, BuildScratch& scratch,
                BvhStructure& out) {
    if (count > 0u && instances == nullptr) return false;
    scratch.prim_aabbs.resize(count);
    scratch.prim_centroids.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        float inv_check[3][4];
        const bool transform_ok =
            instances[i].has_inv_transform || invert_affine_3x4(instances[i].transform, inv_check);
        const bool addr_ok =
            (instances[i].blas_address & ~kAddressMask) == 0 && (instances[i].blas_address & 0xFFull) == 0;
        const Aabb b = (transform_ok && addr_ok) ? instance_world_aabb(instances[i]) : Aabb::empty();
        scratch.prim_aabbs[i] = b;
        scratch.prim_centroids[i] = b.is_empty() ? Vec3{0.0f, 0.0f, 0.0f} : b.centroid();
    }
    return build_4wide_common(count, true, format, kNodeKindInstance, scratch, out,
                              [instances](uint32_t i, uint32_t* dst16) {
                                  if (!pack_instance_node(instances[i], dst16)) {
                                      std::memset(dst16, 0, 16u * sizeof(uint32_t));
                                      dst16[15] = kNoChild;
                                  }
                              });
}

bool refit_tlas(const InstanceDesc* instances, uint32_t count, RefitScratch& scratch, BvhStructure& tlas) {
    if (count > 0u && instances == nullptr) return false;
    if (!tlas.is_tlas || tlas.element_count != count) return false;
    scratch.leaf_aabbs.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t* dst =
            tlas.dwords.data() + tlas.elements_offset_dwords + static_cast<size_t>(i) * kNodeGranularityDwords;
        if (pack_instance_node(instances[i], dst)) {
            scratch.leaf_aabbs[i] = instance_world_aabb(instances[i]);
        } else {
            std::memset(dst, 0, 16u * sizeof(uint32_t));
            dst[15] = kNoChild;
            scratch.leaf_aabbs[i] = Aabb::empty();
        }
    }
    return refit_4wide_common(count, true, scratch, tlas);
}

TraceHit trace_blas(uint64_t base_addr, uint32_t root_node, Vec3 origin, Vec3 dir, float t_min, float t_max,
                    const LoadU32& load, bool sort_boxes, bool cull_back_faces) {
    TraceHit best{};
    best.t = t_max;
    if (root_node == kNoChild || !(t_min <= t_max)) {
        return best;
    }

    uint32_t desc[4];
    make_bvh_desc(base_addr, sort_boxes, desc);
    const Vec3 inv_dir = {safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z)};

    constexpr uint32_t kStackCap = 64;
    uint32_t stack[kStackCap];
    uint32_t sp = 0;
    stack[sp++] = root_node;

    while (sp > 0u) {
        const uint32_t node = stack[--sp];
        ++best.nodes_visited;
        const uint32_t kind = node_ptr_kind(node);
        const Result r = intersect(desc, node, best.t, origin, dir, inv_dir, load);
        if (kind == kNodeKindBox16 || kind == kNodeKindBox32) {
            // Push far-to-near so nearest child is popped first when sort_boxes is true.
            for (int i = 3; i >= 0; --i) {
                if (r.w[i] != kNoChild && sp < kStackCap) {
                    stack[sp++] = r.w[i];
                }
            }
        } else if (kind < 4u) {
            const float t_num = bits_f(r.w[0]);
            const float t_den = bits_f(r.w[1]);
            if (std::isinf(t_num) || t_den == 0.0f) continue;
            const bool front = t_den > 0.0f;
            if (cull_back_faces && !front) continue;
            const float t = t_num / t_den;
            if (t >= t_min && (t < best.t || (!best.hit && t <= best.t))) {
                const uint64_t node_addr =
                    ((base_addr & kAddressMask) + (static_cast<uint64_t>(node_ptr_slot(node)) << 6)) & kAddressMask;
                best.hit = true;
                best.t = t;
                best.u = bits_f(r.w[2]) / t_den;
                best.v = bits_f(r.w[3]) / t_den;
                best.front_face = front;
                best.primitive_index = load((node_addr + 9ull * 4ull) & kAddressMask);
            }
        }
    }
    return best;
}

TraceHit trace_tlas(uint64_t tlas_base_addr, uint32_t tlas_root_node, Vec3 origin, Vec3 dir, float t_min,
                    float t_max, uint8_t ray_mask, const LoadU32& load, bool sort_boxes,
                    bool cull_back_faces) {
    TraceHit best{};
    best.t = t_max;
    if (tlas_root_node == kNoChild || !(t_min <= t_max)) {
        return best;
    }

    uint32_t desc[4];
    make_bvh_desc(tlas_base_addr, sort_boxes, desc);
    const Vec3 inv_dir = {safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z)};

    // Read elements_offset_bytes from TLAS header dw13 for instance_index calculation.
    const uint32_t elements_offset_bytes = load(((tlas_base_addr & kAddressMask) + 13ull * 4ull) & kAddressMask);
    const uint32_t elements_slot_base = elements_offset_bytes / kNodeGranularityBytes;

    constexpr uint32_t kStackCap = 64;
    uint32_t stack[kStackCap];
    uint32_t sp = 0;
    stack[sp++] = tlas_root_node;

    while (sp > 0u) {
        const uint32_t node = stack[--sp];
        ++best.nodes_visited;
        const uint32_t kind = node_ptr_kind(node);
        if (kind == kNodeKindBox16 || kind == kNodeKindBox32) {
            const Result r = intersect(desc, node, best.t, origin, dir, inv_dir, load);
            for (int i = 3; i >= 0; --i) {
                if (r.w[i] != kNoChild && sp < kStackCap) {
                    stack[sp++] = r.w[i];
                }
            }
        } else if (kind == kNodeKindInstance) {
            const uint32_t slot = node_ptr_slot(node);
            const uint64_t inst_addr =
                ((tlas_base_addr & kAddressMask) + (static_cast<uint64_t>(slot) << 6)) & kAddressMask;
            uint32_t words[16];
            for (uint32_t i = 0; i < 16u; ++i) {
                words[i] = load((inst_addr + 4ull * i) & kAddressMask);
            }
            const PackedInstanceView inst = unpack_instance_node(words);
            if ((inst.instance_mask & ray_mask) == 0u || inst.blas_root_node == kNoChild) {
                continue;
            }
            const Vec3 local_o = transform_point_3x4(inst.inv_transform, origin);
            const Vec3 local_d = transform_vector_3x4(inst.inv_transform, dir);
            const bool cull_back = cull_back_faces && ((inst.flags & InstanceFlags::TriangleCullDisable) == 0u);
            const TraceHit sub =
                trace_blas(inst.blas_address, inst.blas_root_node, local_o, local_d, t_min, best.t, load,
                           sort_boxes, cull_back);
            best.nodes_visited += sub.nodes_visited;
            if (sub.hit && (sub.t < best.t || (!best.hit && sub.t <= best.t))) {
                best.hit = true;
                best.t = sub.t;
                best.u = sub.u;
                best.v = sub.v;
                best.front_face =
                    (inst.flags & InstanceFlags::TriangleFrontCounterclockwise) ? !sub.front_face : sub.front_face;
                best.primitive_index = sub.primitive_index;
                best.instance_index = slot >= elements_slot_base ? (slot - elements_slot_base) : 0u;
                best.instance_id = inst.instance_id;
                best.hit_group_contribution = inst.hit_group_contribution;
            }
        }
    }
    return best;
}

}  // namespace playgta6::gpu::rt_bvh
