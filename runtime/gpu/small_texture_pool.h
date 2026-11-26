#pragma once

// Recycler for small host textures (backlog 38).
//
// A host texture is a driver allocation of at least 64 KiB however few bytes it holds, and the title
// creates and drops thousands of tiny ones while it streams. Instead of destroying a small texture
// that is let go of and creating an identical one a moment later, the owner parks its host id here
// and takes it back when a texture of the same shape (format, extent, mips, dimension, storage use)
// is needed. The pool is bookkeeping only: it holds ids and byte counts and emits no GPU packets.
//
// A parked texture still holds its old texels. The owner must overwrite every texel of every mip on
// reuse (zero_fill_writes lists the writes) so no stale content is ever visible.
//
// Parked textures stay counted in the texture budget (texture_budget.h): the owner calls
// note_created once when the host texture is really created and note_destroyed only when it is
// really destroyed, including when the pool evicts or drains it.
//
// Thread-safe.

#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace playgta6::gpu {

inline constexpr uint64_t kSmallTextureMaxBytes = 64u * 1024u;      // largest mip chain that is pooled
inline constexpr uint32_t kSmallTexturePoolMaxCount = 512;          // parked textures at most
inline constexpr uint64_t kSmallTexturePoolMaxBytes = 16ull << 20;  // parked bytes at most

struct SmallTextureShape {
    uint64_t owner = 0;  // the device that owns the host texture ids (ids are per device)
    uint32_t dxgi_format = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mip_levels = 1;
    uint8_t dim = 0;
    bool storage = false;

    bool operator==(const SmallTextureShape& o) const noexcept {
        return owner == o.owner && dxgi_format == o.dxgi_format && width == o.width && height == o.height && depth == o.depth &&
               mip_levels == o.mip_levels && dim == o.dim && storage == o.storage;
    }
};

struct SmallTexturePoolStats {
    uint64_t creates = 0;        // host textures really created by the owner
    uint64_t hits = 0;           // acquires served by a parked texture
    uint64_t misses = 0;         // acquires of a poolable shape that found none
    uint64_t bytes_reused = 0;   // texture bytes served by hits
    uint64_t bytes_created = 0;  // texture bytes of the creates
    uint64_t parked_total = 0;   // textures parked over the run
    uint64_t evicted = 0;        // parked textures destroyed for room
    uint64_t rejected = 0;       // releases not parked (too big, or pool disabled)
    uint64_t create_ns = 0;      // time the owner spent creating host textures
    uint32_t parked_count = 0;
    uint64_t parked_bytes = 0;
    uint64_t peak_parked_bytes = 0;
};

class SmallTexturePool {
public:
    static bool poolable(uint64_t chain_bytes) noexcept {
        return chain_bytes != 0 && chain_bytes <= kSmallTextureMaxBytes;
    }

    void set_limits(uint32_t max_count, uint64_t max_bytes) noexcept;
    void set_enabled(bool enabled) noexcept;
    void reset() noexcept;

    // A parked texture of this shape, or 0 (counts a hit or, for a poolable shape, a miss).
    uint32_t acquire(const SmallTextureShape& shape, uint64_t chain_bytes) noexcept;

    // Parks `id`. Returns false when it is not poolable or the pool is off: the owner destroys it.
    // Ids pushed out to make room are appended to `evicted` with their byte counts: the owner
    // destroys those and notes the bytes destroyed in the budget.
    bool release(const SmallTextureShape& shape, uint32_t id, uint64_t chain_bytes, std::vector<uint32_t>* evicted,
                 std::vector<uint64_t>* evicted_bytes);

    // Removes every parked texture of `owner` (device teardown, memory pressure).
    void drain(uint64_t owner, std::vector<uint32_t>* ids, std::vector<uint64_t>* bytes);

    // The owner reports a real host texture creation.
    void note_create(uint64_t chain_bytes, uint64_t ns) noexcept;

    SmallTexturePoolStats stats() const;

private:
    struct Entry {
        SmallTextureShape shape;
        uint32_t id = 0;
        uint64_t bytes = 0;
    };
    mutable std::mutex mutex_;
    std::deque<Entry> entries_;  // oldest first
    uint32_t max_count_ = kSmallTexturePoolMaxCount;
    uint64_t max_bytes_ = kSmallTexturePoolMaxBytes;
    bool enabled_ = true;
    SmallTexturePoolStats stats_;
};

SmallTexturePool& small_texture_pool() noexcept;

// One zero write: a region of one mip level (depth_or_layers is slices of a 3D mip or array layers).
struct ZeroWrite {
    uint32_t mip = 0;
    uint32_t width = 0;  // texels, rounded up to whole blocks
    uint32_t height = 0;
    uint32_t depth_or_layers = 1;
    uint32_t bytes_per_row = 0;
    uint32_t rows_per_image = 0;  // block rows
    uint64_t bytes = 0;
};

// The writes that overwrite every texel of every mip of a texture with zeros. Empty when the
// block description is unknown (block_w/h/bytes == 0): such a texture must not be pooled.
std::vector<ZeroWrite> zero_fill_writes(uint32_t width, uint32_t height, uint32_t depth_or_layers, uint32_t mips,
                                        bool is_3d, uint32_t block_w, uint32_t block_h, uint32_t block_bytes);

}  // namespace playgta6::gpu
