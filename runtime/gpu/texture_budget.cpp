#include "gpu/texture_budget.h"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace playgta6::gpu {

namespace {

bool pct_at_least(uint64_t used_bytes, uint64_t budget_bytes, uint32_t pct) noexcept {
    if (budget_bytes == 0) {
        return false;
    }
    return static_cast<long double>(used_bytes) * 100.0L >= static_cast<long double>(budget_bytes) * pct;
}

uint64_t pct_of(uint64_t budget_bytes, uint32_t pct) noexcept {
    if (budget_bytes == 0) {
        return 0;
    }
    const uint64_t q = budget_bytes / 100u;
    const uint64_t r = budget_bytes % 100u;
    return q * pct + (r * pct) / 100u;
}

uint64_t saturating_add(uint64_t a, uint64_t b) noexcept {
    return a > std::numeric_limits<uint64_t>::max() - b ? std::numeric_limits<uint64_t>::max() : a + b;
}

}  // namespace

uint64_t texture_chain_bytes(uint32_t width, uint32_t height, uint32_t depth, uint32_t mips, uint32_t block_w,
                             uint32_t block_h, uint32_t block_bytes, uint32_t start_mip,
                             uint32_t array_layers) noexcept {
    if (block_w == 0 || block_h == 0 || block_bytes == 0 || array_layers == 0) {
        return 0;
    }
    const uint32_t eff_mips = std::max(mips, 1u);
    uint64_t total = 0;
    for (uint32_t m = start_mip; m < eff_mips; ++m) {
        const uint64_t w = m >= 32 ? 1u : std::max<uint64_t>(1u, width >> m);
        const uint64_t h = m >= 32 ? 1u : std::max<uint64_t>(1u, height >> m);
        const uint64_t d = m >= 32 ? 1u : std::max<uint64_t>(1u, depth >> m);
        const uint64_t bx = (w + block_w - 1u) / block_w;
        const uint64_t by = (h + block_h - 1u) / block_h;
        total += bx * by * d * uint64_t{array_layers} * block_bytes;
    }
    return total;
}

uint32_t cap_floor_for(uint32_t width, uint32_t height, uint32_t mips, uint32_t cap) noexcept {
    if (cap == 0 || mips <= 1) {
        return 0;
    }
    const uint32_t side = std::max(width, height);
    uint32_t drop = 0;
    while (drop + 1 < mips && (drop >= 32 ? 1u : std::max(1u, side >> drop)) > cap) {
        ++drop;
    }
    return drop;
}

uint32_t cap_floor_for_3d(uint32_t width, uint32_t height, uint32_t depth, uint32_t mips, uint32_t cap) noexcept {
    if (cap == 0 || mips <= 1) {
        return 0;
    }
    const uint32_t side = std::max({width, height, depth});
    uint32_t drop = 0;
    while (drop + 1 < mips && (drop >= 32 ? 1u : std::max(1u, side >> drop)) > cap) {
        ++drop;
    }
    return drop;
}

uint32_t cap_for_pressure(uint64_t used_bytes, uint64_t budget_bytes, uint32_t base_cap) noexcept {
    uint32_t tier = 0;
    if (budget_bytes != 0) {
        if (pct_at_least(used_bytes, budget_bytes, kTextureCapHighPercent)) {
            tier = kTextureCapHigh;
        } else if (pct_at_least(used_bytes, budget_bytes, kTextureCapMediumPercent)) {
            tier = kTextureCapMedium;
        }
    }
    if (tier == 0) {
        return base_cap;
    }
    return base_cap == 0 ? tier : std::min(base_cap, tier);
}

BudgetPressureTier pressure_tier_for(uint64_t used_bytes, uint64_t budget_bytes) noexcept {
    if (budget_bytes == 0) {
        return BudgetPressureTier::Normal;
    }
    if (pct_at_least(used_bytes, budget_bytes, kTextureReclaimCriticalPercent)) {
        return BudgetPressureTier::Critical;
    }
    if (pct_at_least(used_bytes, budget_bytes, kTextureCapHighPercent)) {
        return BudgetPressureTier::High;
    }
    if (pct_at_least(used_bytes, budget_bytes, kTextureReclaimTriggerPercent)) {
        return BudgetPressureTier::Reclaim;
    }
    if (pct_at_least(used_bytes, budget_bytes, kTextureCapMediumPercent)) {
        return BudgetPressureTier::Medium;
    }
    return BudgetPressureTier::Normal;
}

uint64_t reclaim_target_bytes(uint64_t used_bytes, uint64_t budget_bytes, bool reclaim_active,
                              uint64_t incoming_bytes) noexcept {
    if (budget_bytes == 0) {
        return 0;
    }
    const uint64_t projected = saturating_add(used_bytes, incoming_bytes);
    const uint64_t target = pct_of(budget_bytes, kTextureReclaimTargetPercent);
    const bool triggered = pct_at_least(projected, budget_bytes, kTextureReclaimTriggerPercent);
    if (!triggered && !reclaim_active) {
        return 0;
    }
    if (projected <= target) {
        return 0;
    }
    return projected - target;
}

void TextureBudget::set_budget_bytes(uint64_t bytes) noexcept { budget_.store(bytes, std::memory_order_relaxed); }

void TextureBudget::set_base_cap(uint32_t cap) noexcept { base_cap_.store(cap, std::memory_order_relaxed); }

void TextureBudget::reset() noexcept {
    budget_ = 0;
    used_ = 0;
    peak_ = 0;
    base_cap_ = 0;
    last_cap_ = 0;
    cap_changes_ = 0;
    capped_ = 0;
    mips_dropped_ = 0;
    bytes_skipped_ = 0;
    restored_ = 0;
    reallocs_ = 0;
    saved_ = 0;
    reclaim_active_.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(lru_mutex_);
    lru_index_.clear();
    lru_list_.clear();
    lru_tick_ = 0;
    lru_tracked_bytes_ = 0;
    lru_pinned_count_ = 0;
    lru_evictions_ = 0;
    lru_bytes_reclaimed_ = 0;
}

void TextureBudget::note_created(uint64_t bytes) noexcept {
    const uint64_t now = used_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    uint64_t peak = peak_.load(std::memory_order_relaxed);
    while (now > peak && !peak_.compare_exchange_weak(peak, now, std::memory_order_relaxed)) {
    }
}

void TextureBudget::note_destroyed(uint64_t bytes) noexcept {
    uint64_t cur = used_.load(std::memory_order_relaxed);
    uint64_t next = cur > bytes ? cur - bytes : 0;
    while (!used_.compare_exchange_weak(cur, next, std::memory_order_relaxed)) {
        next = cur > bytes ? cur - bytes : 0;
    }
    const uint64_t budget = budget_.load(std::memory_order_relaxed);
    if (budget == 0 || next <= pct_of(budget, kTextureReclaimTargetPercent)) {
        reclaim_active_.store(false, std::memory_order_relaxed);
    }
}

uint32_t TextureBudget::current_cap() noexcept {
    const uint32_t cap = cap_for_pressure(used_.load(std::memory_order_relaxed), budget_.load(std::memory_order_relaxed),
                                          base_cap_.load(std::memory_order_relaxed));
    const uint32_t prev = last_cap_.exchange(cap, std::memory_order_relaxed);
    if (prev != cap) {
        const uint64_t n = cap_changes_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 64) {  // bounded log: a use hovering at a threshold must not flood
            std::fprintf(stderr, "[gpu] texture size cap %u -> %u px (%llu of %llu MiB in use)\n", prev, cap,
                         static_cast<unsigned long long>(used_.load(std::memory_order_relaxed) >> 20),
                         static_cast<unsigned long long>(budget_.load(std::memory_order_relaxed) >> 20));
        }
    }
    return cap;
}

void TextureBudget::note_capped(uint32_t mips, uint64_t bytes) noexcept {
    capped_.fetch_add(1, std::memory_order_relaxed);
    mips_dropped_.fetch_add(mips, std::memory_order_relaxed);
    bytes_skipped_.fetch_add(bytes, std::memory_order_relaxed);
}

uint32_t TextureBudget::cap_with_extra(uint64_t extra_bytes) const noexcept {
    return cap_for_pressure(saturating_add(used_.load(std::memory_order_relaxed), extra_bytes),
                            budget_.load(std::memory_order_relaxed), base_cap_.load(std::memory_order_relaxed));
}

bool TextureBudget::restore_fits(uint32_t width, uint32_t height, uint32_t mips, uint32_t target_floor,
                                 uint64_t extra_bytes) const noexcept {
    return cap_floor_for(width, height, mips, cap_with_extra(extra_bytes)) <= target_floor;
}

void TextureBudget::note_host_realloc(int64_t saved_delta) noexcept {
    reallocs_.fetch_add(1, std::memory_order_relaxed);
    if (saved_delta > 0) {
        const uint64_t d = static_cast<uint64_t>(saved_delta);
        note_destroyed(d);
        saved_.fetch_add(d, std::memory_order_relaxed);
    } else if (saved_delta < 0) {
        const uint64_t d = static_cast<uint64_t>(-saved_delta);
        note_created(d);
        uint64_t cur = saved_.load(std::memory_order_relaxed);
        while (!saved_.compare_exchange_weak(cur, cur > d ? cur - d : 0, std::memory_order_relaxed)) {
        }
    }
}

void TextureBudget::note_shrunk_released(uint64_t saved_bytes) noexcept {
    if (saved_bytes == 0) {
        return;
    }
    used_.fetch_add(saved_bytes, std::memory_order_relaxed);  // no peak: the full size leaves right after
    uint64_t cur = saved_.load(std::memory_order_relaxed);
    while (!saved_.compare_exchange_weak(cur, cur > saved_bytes ? cur - saved_bytes : 0, std::memory_order_relaxed)) {
    }
}

void TextureBudget::update_hysteresis_after_free_locked() noexcept {
    const uint64_t budget = budget_.load(std::memory_order_relaxed);
    const uint64_t used = used_.load(std::memory_order_relaxed);
    if (budget == 0 || used <= pct_of(budget, kTextureReclaimTargetPercent)) {
        reclaim_active_.store(false, std::memory_order_relaxed);
    }
}

void TextureBudget::register_texture(uint32_t resource_id, uint64_t bytes, uint64_t frame, bool pinned,
                                     bool evictable) {
    if (resource_id == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(lru_mutex_);
    const uint64_t tick = ++lru_tick_;
    if (auto it = lru_index_.find(resource_id); it != lru_index_.end()) {
        LruEntry& e = *it->second;
        lru_tracked_bytes_ = lru_tracked_bytes_ >= e.bytes ? (lru_tracked_bytes_ - e.bytes) + bytes : bytes;
        if (e.pinned != pinned) {
            if (pinned) {
                ++lru_pinned_count_;
            } else if (lru_pinned_count_ > 0) {
                --lru_pinned_count_;
            }
        }
        e.bytes = bytes;
        e.last_used_tick = tick;
        e.last_used_frame = std::max(e.last_used_frame, frame);
        e.pinned = pinned;
        e.evictable = evictable;
        if (it->second != lru_list_.begin()) {
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        }
        return;
    }
    lru_list_.push_front(LruEntry{resource_id, bytes, tick, frame, pinned, evictable});
    lru_index_.emplace(resource_id, lru_list_.begin());
    lru_tracked_bytes_ += bytes;
    if (pinned) {
        ++lru_pinned_count_;
    }
}

void TextureBudget::touch_texture(uint32_t resource_id, uint64_t frame) {
    if (resource_id == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(lru_mutex_);
    const auto it = lru_index_.find(resource_id);
    if (it == lru_index_.end()) {
        return;
    }
    LruEntry& e = *it->second;
    e.last_used_tick = ++lru_tick_;
    if (frame > e.last_used_frame) {
        e.last_used_frame = frame;
    }
    if (it->second != lru_list_.begin()) {
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
    }
}

void TextureBudget::update_texture_bytes(uint32_t resource_id, uint64_t new_bytes) {
    if (resource_id == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(lru_mutex_);
    const auto it = lru_index_.find(resource_id);
    if (it == lru_index_.end()) {
        return;
    }
    LruEntry& e = *it->second;
    lru_tracked_bytes_ = lru_tracked_bytes_ >= e.bytes ? (lru_tracked_bytes_ - e.bytes) + new_bytes : new_bytes;
    e.bytes = new_bytes;
}

void TextureBudget::set_texture_pinned(uint32_t resource_id, bool pinned) {
    if (resource_id == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(lru_mutex_);
    const auto it = lru_index_.find(resource_id);
    if (it == lru_index_.end()) {
        return;
    }
    LruEntry& e = *it->second;
    if (e.pinned == pinned) {
        return;
    }
    e.pinned = pinned;
    if (pinned) {
        ++lru_pinned_count_;
    } else if (lru_pinned_count_ > 0) {
        --lru_pinned_count_;
    }
}

void TextureBudget::unregister_texture(uint32_t resource_id) {
    if (resource_id == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(lru_mutex_);
    const auto it = lru_index_.find(resource_id);
    if (it == lru_index_.end()) {
        return;
    }
    const LruEntry& e = *it->second;
    lru_tracked_bytes_ = lru_tracked_bytes_ >= e.bytes ? lru_tracked_bytes_ - e.bytes : 0;
    if (e.pinned && lru_pinned_count_ > 0) {
        --lru_pinned_count_;
    }
    lru_list_.erase(it->second);
    lru_index_.erase(it);
}

BudgetPressureTier TextureBudget::current_pressure_tier() const noexcept {
    return pressure_tier_for(used_.load(std::memory_order_relaxed), budget_.load(std::memory_order_relaxed));
}

bool TextureBudget::is_reclaim_active() const noexcept {
    return reclaim_active_.load(std::memory_order_relaxed);
}

uint64_t TextureBudget::bytes_to_reclaim(uint64_t incoming_bytes) noexcept {
    const uint64_t used = used_.load(std::memory_order_relaxed);
    const uint64_t budget = budget_.load(std::memory_order_relaxed);
    if (budget == 0) {
        reclaim_active_.store(false, std::memory_order_relaxed);
        return 0;
    }
    const uint64_t projected = saturating_add(used, incoming_bytes);
    const uint64_t target = pct_of(budget, kTextureReclaimTargetPercent);
    bool active = reclaim_active_.load(std::memory_order_relaxed);
    if (pct_at_least(projected, budget, kTextureReclaimTriggerPercent)) {
        active = true;
        reclaim_active_.store(true, std::memory_order_relaxed);
    } else if (projected <= target) {
        active = false;
        reclaim_active_.store(false, std::memory_order_relaxed);
    }
    return reclaim_target_bytes(used, budget, active, incoming_bytes);
}

std::vector<LruEvictCandidate> TextureBudget::select_eviction_candidates(uint64_t current_frame, uint64_t target_bytes,
                                                                         uint32_t max_candidates,
                                                                         uint64_t min_age_frames) {
    std::vector<LruEvictCandidate> out;
    if (max_candidates == 0) {
        return out;
    }
    const uint64_t goal = target_bytes != 0 ? target_bytes : bytes_to_reclaim(0);
    if (goal == 0) {
        return out;
    }
    const bool critical = current_pressure_tier() == BudgetPressureTier::Critical;

    std::lock_guard<std::mutex> lock(lru_mutex_);
    uint64_t accumulated = 0;
    for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
        if (out.size() >= max_candidates || accumulated >= goal) {
            break;
        }
        const LruEntry& e = *it;
        if (e.pinned || !e.evictable) {
            continue;
        }
        if (min_age_frames > 0) {
            // Current-frame textures are always protected (even under Critical pressure).
            if (e.last_used_frame >= current_frame) {
                continue;
            }
            // Under non-critical pressure, also protect textures touched within min_age_frames.
            if (!critical && (current_frame - e.last_used_frame) < min_age_frames) {
                continue;
            }
        }
        out.push_back(LruEvictCandidate{e.resource_id, e.bytes, e.last_used_tick, e.last_used_frame});
        accumulated = saturating_add(accumulated, e.bytes);
    }
    return out;
}

void TextureBudget::note_lru_evicted(uint32_t resource_id, uint64_t bytes_reclaimed) {
    std::lock_guard<std::mutex> lock(lru_mutex_);
    uint64_t reclaimed = bytes_reclaimed;
    if (resource_id != 0) {
        if (const auto it = lru_index_.find(resource_id); it != lru_index_.end()) {
            const LruEntry& e = *it->second;
            if (reclaimed == 0) {
                reclaimed = e.bytes;
            }
            lru_tracked_bytes_ = lru_tracked_bytes_ >= e.bytes ? lru_tracked_bytes_ - e.bytes : 0;
            if (e.pinned && lru_pinned_count_ > 0) {
                --lru_pinned_count_;
            }
            lru_list_.erase(it->second);
            lru_index_.erase(it);
        }
    }
    ++lru_evictions_;
    lru_bytes_reclaimed_ += reclaimed;
    update_hysteresis_after_free_locked();
}

TextureBudgetStats TextureBudget::stats() const noexcept {
    TextureBudgetStats s;
    s.budget_bytes = budget_.load(std::memory_order_relaxed);
    s.used_bytes = used_.load(std::memory_order_relaxed);
    s.peak_used_bytes = peak_.load(std::memory_order_relaxed);
    s.base_cap = base_cap_.load(std::memory_order_relaxed);
    s.current_cap = last_cap_.load(std::memory_order_relaxed);
    s.cap_changes = cap_changes_.load(std::memory_order_relaxed);
    s.textures_capped = capped_.load(std::memory_order_relaxed);
    s.mips_dropped = mips_dropped_.load(std::memory_order_relaxed);
    s.bytes_not_uploaded = bytes_skipped_.load(std::memory_order_relaxed);
    s.textures_restored = restored_.load(std::memory_order_relaxed);
    s.host_reallocs = reallocs_.load(std::memory_order_relaxed);
    s.host_bytes_saved = saved_.load(std::memory_order_relaxed);
    s.pressure_tier = static_cast<uint8_t>(pressure_tier_for(s.used_bytes, s.budget_bytes));
    s.reclaim_active = reclaim_active_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(lru_mutex_);
        s.lru_tracked_textures = lru_list_.size();
        s.lru_tracked_bytes = lru_tracked_bytes_;
        s.lru_evictions = lru_evictions_;
        s.lru_bytes_reclaimed = lru_bytes_reclaimed_;
        s.lru_pinned_textures = lru_pinned_count_;
    }
    return s;
}

TextureBudget& texture_budget() noexcept {
    static TextureBudget instance;
    return instance;
}

}  // namespace playgta6::gpu
