#pragma once

// ClearTexture (opcode 63): the GPU-side zeroing of a reused (pooled) texture, without Dawn.
//
// The runtime recycles host textures (gpu/small_texture_pool.h) and a reused texture must read as
// zero exactly like a freshly created one. The executor does not zero it at once: it records the
// texture's subresources as pending and encodes copies from a shared zero buffer at the start of the
// next command encoder it submits (the frame's, or an early read-back flush), so a clear costs no
// extra submit. WriteTexture packets run on the queue timeline before that encoder, so a write that
// lands on a pending subresource is handled here:
//  - a write that covers the next rows of a 2D subresource (full width, starting at or above the
//    first still-pending row) or the next slices of a 3D mip moves the pending start past it; the
//    covered part is never cleared later, so the write is not wiped;
//  - a write inside the already covered part needs nothing;
//  - any other write onto a pending part needs the clear first: the executor submits the pending
//    clears (a flush) before the write.
// The copies of what stays pending are planned by `zero_copies`.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace playgta6::gpu::native_exec {

struct ClearGeometry {
    uint32_t width = 1, height = 1, depth_or_layers = 1, mips = 1;
    uint8_t dimension = 1;  // effective: 1 = 2D (a promoted 1D too), 2 = 3D
    uint32_t block_w = 1, block_h = 1, block_bytes = 4;
};

// One buffer -> texture copy from the zero buffer (offset 0).
struct ZeroCopy {
    uint32_t mip = 0;
    uint32_t y = 0;  // texels (a block row boundary)
    uint32_t z = 0;  // array layer (2D) or depth slice (3D)
    uint32_t width = 0, height = 0, depth = 1;  // texels (whole blocks) and layers / slices
    uint32_t bytes_per_row = 0;                  // 256-aligned
    uint32_t rows_per_image = 0;                 // block rows
    uint64_t bytes = 0;                          // texel bytes the copy zeroes (tight)
};

inline constexpr uint32_t kZeroCopyRowAlign = 256;

class PendingClear {
public:
    void reset(const ClearGeometry& g) {
        g_ = g;
        g_.width = std::max<uint32_t>(1, g_.width);
        g_.height = std::max<uint32_t>(1, g_.height);
        g_.depth_or_layers = std::max<uint32_t>(1, g_.depth_or_layers);
        g_.mips = std::max<uint32_t>(1, g_.mips);
        g_.block_w = std::max<uint32_t>(1, g_.block_w);
        g_.block_h = std::max<uint32_t>(1, g_.block_h);
        covered_.assign(subresource_count(), 0u);
    }

    const ClearGeometry& geometry() const { return g_; }

    // Physical (block-rounded) extent of a mip.
    uint32_t mip_width(uint32_t m) const { return round_up(level(g_.width, m), g_.block_w); }
    uint32_t mip_height(uint32_t m) const { return round_up(level(g_.height, m), g_.block_h); }
    uint32_t mip_depth(uint32_t m) const { return is_3d() ? level(g_.depth_or_layers, m) : g_.depth_or_layers; }

    // True while some subresource still needs zeroing.
    bool pending() const {
        for (size_t i = 0; i < covered_.size(); ++i) {
            if (covered_[i] < extent_of(i)) return true;
        }
        return false;
    }

    uint32_t pending_subresources() const {
        uint32_t n = 0;
        for (size_t i = 0; i < covered_.size(); ++i) n += covered_[i] < extent_of(i) ? 1u : 0u;
        return n;
    }

    // A write of the (block-aligned, mip-clamped) region about to land on the queue. Returns false
    // when it overlaps a still-pending part it does not extend: the pending clears must be submitted
    // first. On true the covered prefixes have moved past the write where it extended them.
    bool note_write(uint32_t mip, uint32_t ox, uint32_t oy, uint32_t oz, uint32_t w, uint32_t h, uint32_t d) {
        if (mip >= g_.mips || w == 0 || h == 0 || d == 0) return true;
        const uint32_t mw = mip_width(mip), mh = mip_height(mip), md = mip_depth(mip);
        const bool full_width = ox == 0 && uint64_t{w} >= mw;
        if (is_3d()) {
            uint32_t& cov = covered_[mip];
            if (cov >= md) return true;
            const uint64_t end = uint64_t{oz} + d;
            if (full_width && oy == 0 && uint64_t{h} >= mh && oz <= cov) {
                cov = static_cast<uint32_t>(std::min<uint64_t>(md, std::max<uint64_t>(cov, end)));
                return true;
            }
            return end <= cov;
        }
        const uint32_t layers = g_.depth_or_layers;
        const uint32_t last = static_cast<uint32_t>(std::min<uint64_t>(layers, uint64_t{oz} + d));
        bool ok = true;
        for (uint32_t layer = oz; layer < last; ++layer) {
            uint32_t& cov = covered_[size_t{mip} * layers + layer];
            if (cov >= mh) continue;
            const uint64_t end = uint64_t{oy} + h;
            if (full_width && oy <= cov) {
                cov = static_cast<uint32_t>(std::min<uint64_t>(mh, std::max<uint64_t>(cov, end)));
            } else if (end > cov) {
                ok = false;
            }
        }
        return ok;
    }

    // The copies that zero every still-pending part, each at most `zero_bytes` of buffer (the zero
    // buffer's size; it must hold at least one 256-aligned row of the widest mip).
    void zero_copies(uint64_t zero_bytes, std::vector<ZeroCopy>* out) const {
        for (uint32_t m = 0; m < g_.mips; ++m) {
            const uint32_t mw = mip_width(m), mh = mip_height(m);
            const uint32_t bx = mw / g_.block_w;
            const uint64_t tight_row = uint64_t{bx} * g_.block_bytes;
            const uint32_t bpr = static_cast<uint32_t>(round_up64(tight_row, kZeroCopyRowAlign));
            if (bpr == 0 || bpr > zero_bytes) continue;
            const uint32_t rows_full = mh / g_.block_h;
            if (is_3d()) {
                const uint32_t md = mip_depth(m);
                const uint32_t start = covered_[m];
                if (start >= md) continue;
                emit_slices(m, 0, start, md - start, mw, rows_full, bpr, tight_row, zero_bytes, out);
                continue;
            }
            const uint32_t layers = g_.depth_or_layers;
            uint32_t layer = 0;
            while (layer < layers) {
                const uint32_t cov = covered_[size_t{m} * layers + layer];
                if (cov >= mh) {
                    ++layer;
                    continue;
                }
                if (cov == 0) {
                    // A run of untouched layers: one copy covers several when they fit.
                    uint32_t run = 1;
                    while (layer + run < layers && covered_[size_t{m} * layers + layer + run] == 0) ++run;
                    emit_slices(m, 0, layer, run, mw, rows_full, bpr, tight_row, zero_bytes, out);
                    layer += run;
                    continue;
                }
                const uint32_t start_row = cov / g_.block_h;
                emit_slices(m, start_row, layer, 1, mw, rows_full - start_row, bpr, tight_row, zero_bytes, out);
                ++layer;
            }
        }
    }

    // Largest 256-aligned row of any mip (the zero buffer must hold at least this).
    uint64_t widest_row() const {
        const uint64_t tight = uint64_t{mip_width(0) / g_.block_w} * g_.block_bytes;
        return round_up64(tight, kZeroCopyRowAlign);
    }

private:
    static uint32_t level(uint32_t base, uint32_t m) { return m >= 32 ? 1u : std::max<uint32_t>(1u, base >> m); }
    static uint32_t round_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }
    static uint64_t round_up64(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
    bool is_3d() const { return g_.dimension == 2; }
    size_t subresource_count() const {
        return is_3d() ? size_t{g_.mips} : size_t{g_.mips} * g_.depth_or_layers;
    }
    uint32_t extent_of(size_t i) const {
        if (is_3d()) return mip_depth(static_cast<uint32_t>(i));
        return mip_height(static_cast<uint32_t>(i / g_.depth_or_layers));
    }

    // Copies of `count` slices from `first_slice`, block rows start_row .. start_row + rows.
    void emit_slices(uint32_t mip, uint32_t start_row, uint32_t first_slice, uint32_t count, uint32_t mw,
                     uint32_t rows, uint32_t bpr, uint64_t tight_row, uint64_t zero_bytes,
                     std::vector<ZeroCopy>* out) const {
        if (rows == 0 || count == 0) return;
        const uint64_t slice_bytes = uint64_t{rows} * bpr;
        if (slice_bytes <= zero_bytes) {
            const uint32_t per = static_cast<uint32_t>(std::min<uint64_t>(count, zero_bytes / slice_bytes));
            for (uint32_t s = 0; s < count; s += per) {
                const uint32_t n = std::min(per, count - s);
                ZeroCopy c;
                c.mip = mip;
                c.y = start_row * g_.block_h;
                c.z = first_slice + s;
                c.width = mw;
                c.height = rows * g_.block_h;
                c.depth = n;
                c.bytes_per_row = bpr;
                c.rows_per_image = rows;
                c.bytes = tight_row * rows * n;
                out->push_back(c);
            }
            return;
        }
        const uint32_t per_rows = static_cast<uint32_t>(std::max<uint64_t>(1, zero_bytes / bpr));
        for (uint32_t s = 0; s < count; ++s) {
            for (uint32_t r = 0; r < rows; r += per_rows) {
                const uint32_t n = std::min(per_rows, rows - r);
                ZeroCopy c;
                c.mip = mip;
                c.y = (start_row + r) * g_.block_h;
                c.z = first_slice + s;
                c.width = mw;
                c.height = n * g_.block_h;
                c.depth = 1;
                c.bytes_per_row = bpr;
                c.rows_per_image = n;
                c.bytes = tight_row * n;
                out->push_back(c);
            }
        }
    }

    ClearGeometry g_;
    std::vector<uint32_t> covered_;  // per subresource: texel rows (2D) / slices (3D) already written
};

}  // namespace playgta6::gpu::native_exec
