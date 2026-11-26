#include "gpu/small_texture_pool.h"

#include <algorithm>
#include <iterator>

namespace playgta6::gpu {

void SmallTexturePool::set_limits(uint32_t max_count, uint64_t max_bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    max_count_ = max_count;
    max_bytes_ = max_bytes;
}

void SmallTexturePool::set_enabled(bool enabled) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

void SmallTexturePool::reset() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    stats_ = SmallTexturePoolStats{};
    max_count_ = kSmallTexturePoolMaxCount;
    max_bytes_ = kSmallTexturePoolMaxBytes;
    enabled_ = true;
}

uint32_t SmallTexturePool::acquire(const SmallTextureShape& shape, uint64_t chain_bytes) noexcept {
    if (!poolable(chain_bytes)) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) {
        return 0;
    }
    // Newest first: the most recently parked texture is the most likely still resident.
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        if (it->shape == shape) {
            const uint32_t id = it->id;
            stats_.hits += 1;
            stats_.bytes_reused += it->bytes;
            stats_.parked_bytes -= it->bytes;
            entries_.erase(std::next(it).base());
            stats_.parked_count = static_cast<uint32_t>(entries_.size());
            return id;
        }
    }
    stats_.misses += 1;
    return 0;
}

bool SmallTexturePool::release(const SmallTextureShape& shape, uint32_t id, uint64_t chain_bytes,
                               std::vector<uint32_t>* evicted, std::vector<uint64_t>* evicted_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || id == 0 || !poolable(chain_bytes) || max_count_ == 0 || chain_bytes > max_bytes_) {
        stats_.rejected += 1;
        return false;
    }
    while (!entries_.empty() && (entries_.size() >= max_count_ || stats_.parked_bytes + chain_bytes > max_bytes_)) {
        const Entry& old = entries_.front();
        if (evicted != nullptr) {
            evicted->push_back(old.id);
        }
        if (evicted_bytes != nullptr) {
            evicted_bytes->push_back(old.bytes);
        }
        stats_.parked_bytes -= old.bytes;
        stats_.evicted += 1;
        entries_.pop_front();
    }
    entries_.push_back(Entry{shape, id, chain_bytes});
    stats_.parked_total += 1;
    stats_.parked_bytes += chain_bytes;
    stats_.peak_parked_bytes = std::max(stats_.peak_parked_bytes, stats_.parked_bytes);
    stats_.parked_count = static_cast<uint32_t>(entries_.size());
    return true;
}

void SmallTexturePool::drain(uint64_t owner, std::vector<uint32_t>* ids, std::vector<uint64_t>* bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->shape.owner != owner) {
            ++it;
            continue;
        }
        if (ids != nullptr) {
            ids->push_back(it->id);
        }
        if (bytes != nullptr) {
            bytes->push_back(it->bytes);
        }
        stats_.parked_bytes -= it->bytes;
        it = entries_.erase(it);
    }
    stats_.parked_count = static_cast<uint32_t>(entries_.size());
}

void SmallTexturePool::note_create(uint64_t chain_bytes, uint64_t ns) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.creates += 1;
    stats_.bytes_created += chain_bytes;
    stats_.create_ns += ns;
}

SmallTexturePoolStats SmallTexturePool::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

SmallTexturePool& small_texture_pool() noexcept {
    static SmallTexturePool pool;
    return pool;
}

std::vector<ZeroWrite> zero_fill_writes(uint32_t width, uint32_t height, uint32_t depth_or_layers, uint32_t mips,
                                        bool is_3d, uint32_t block_w, uint32_t block_h, uint32_t block_bytes) {
    std::vector<ZeroWrite> out;
    if (block_w == 0 || block_h == 0 || block_bytes == 0 || width == 0 || height == 0 || mips == 0) {
        return out;
    }
    const uint32_t depth = std::max<uint32_t>(1u, depth_or_layers);
    for (uint32_t m = 0; m < mips; ++m) {
        const uint32_t w = std::max<uint32_t>(1u, width >> m);
        const uint32_t h = std::max<uint32_t>(1u, height >> m);
        const uint32_t d = is_3d ? std::max<uint32_t>(1u, depth >> m) : depth;
        ZeroWrite z;
        z.mip = m;
        const uint32_t bx = (w + block_w - 1u) / block_w;
        const uint32_t by = (h + block_h - 1u) / block_h;
        z.width = bx * block_w;
        z.height = by * block_h;
        z.depth_or_layers = d;
        z.bytes_per_row = bx * block_bytes;
        z.rows_per_image = by;
        z.bytes = uint64_t{z.bytes_per_row} * by * d;
        out.push_back(z);
    }
    return out;
}

}  // namespace playgta6::gpu
