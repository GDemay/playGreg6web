#include "gpu/rt_arena.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <utility>

namespace playgta6::gpu::rt_arena {

namespace {

constexpr size_t kMaxRecycledPageLists = 64;

bool range_overflows_40bit(uint64_t addr, uint64_t size) {
    const uint64_t span = size ? size : 1ull;
    const uint64_t masked = addr & kAddressMask;
    return (span - 1ull) > (kAddressMask - masked);
}

uint32_t f_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

bool invert_3x4(const float m[3][4], float out[3][4]) {
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

    out[0][0] = static_cast<float>(r00);
    out[0][1] = static_cast<float>(r01);
    out[0][2] = static_cast<float>(r02);
    out[0][3] = static_cast<float>(-(r00 * tx + r01 * ty + r02 * tz));

    out[1][0] = static_cast<float>(r10);
    out[1][1] = static_cast<float>(r11);
    out[1][2] = static_cast<float>(r12);
    out[1][3] = static_cast<float>(-(r10 * tx + r11 * ty + r12 * tz));

    out[2][0] = static_cast<float>(r20);
    out[2][1] = static_cast<float>(r21);
    out[2][2] = static_cast<float>(r22);
    out[2][3] = static_cast<float>(-(r20 * tx + r21 * ty + r22 * tz));
    return true;
}

}  // namespace

const char* skip_reason_name(SkipReason r) {
    switch (r) {
    case SkipReason::RtOff: return "rt_off";
    case SkipReason::ArenaFull: return "arena_full";
    case SkipReason::NotResident: return "not_resident";
    case SkipReason::Atomic64: return "atomic64";
    case SkipReason::TranslatorPending: return "translator_pending";
    case SkipReason::Count: break;
    }
    return "?";
}

bool pack_instance_3x4(const Instance3x4& desc, uint32_t out_dwords[kInstance3x4Dwords]) {
    if (out_dwords == nullptr) return false;
    if ((desc.blas_address & ~kAddressMask) != 0 || (desc.blas_address & 0xFFull) != 0) {
        return false;
    }
    float inv[3][4];
    if (desc.transform_is_inverse) {
        std::memcpy(inv, desc.transform, sizeof(inv));
    } else if (!invert_3x4(desc.transform, inv)) {
        std::memset(out_dwords, 0, kInstance3x4Dwords * sizeof(uint32_t));
        out_dwords[15] = 0xFFFFFFFFu;
        return false;
    }
    for (uint32_t r = 0; r < 3u; ++r) {
        for (uint32_t c = 0; c < 4u; ++c) {
            out_dwords[r * 4u + c] = f_bits(inv[r][c]);
        }
    }
    out_dwords[12] = (desc.instance_id & 0xFFFFFFu) | (static_cast<uint32_t>(desc.instance_mask) << 24);
    out_dwords[13] = (desc.hit_group_contribution & 0xFFFFFFu) | (static_cast<uint32_t>(desc.flags) << 24);
    out_dwords[14] = static_cast<uint32_t>((desc.blas_address & kAddressMask) >> 8);
    out_dwords[15] = desc.blas_root_node;
    return true;
}

size_t pack_instances_3x4(const Instance3x4* instances, size_t count, uint32_t* out_dwords,
                          size_t out_dword_capacity) {
    if (instances == nullptr || out_dwords == nullptr) return 0;
    const size_t max_count = std::min(count, out_dword_capacity / kInstance3x4Dwords);
    size_t packed = 0;
    for (size_t i = 0; i < max_count; ++i) {
        if (pack_instance_3x4(instances[i], out_dwords + i * kInstance3x4Dwords)) {
            ++packed;
        }
    }
    return packed;
}

Arena::Arena(uint32_t arenas, uint32_t pages_per_arena)
    : arenas_(std::min(std::max(arenas, 1u), kArenaCount)),
      pages_per_arena_(std::max(pages_per_arena, 1u)),
      capacity_(static_cast<uint64_t>(arenas_) * pages_per_arena_),
      dir_(kTopEntries, 0),
      scratch_budget_pages_(static_cast<uint32_t>(
          std::min<uint64_t>(kDefaultScratchBudgetPages, static_cast<uint64_t>(arenas_) * pages_per_arena_))) {}

void Arena::touch_dir(uint32_t word) {
    dirty_first_ = std::min(dirty_first_, word);
    dirty_last_ = std::max(dirty_last_, word);
}

bool Arena::take_dir_dirty(uint32_t* first, uint32_t* last) {
    if (dirty_first_ == 0xffffffffu) return false;
    *first = dirty_first_;
    *last = dirty_last_;
    dirty_first_ = 0xffffffffu;
    dirty_last_ = 0;
    return true;
}

bool Arena::set_entry(uint64_t page, uint32_t entry) {
    const uint32_t top = top_index(page);
    uint32_t t = dir_[top];
    if (t == 0) {
        if (entry == 0) return true;
        if (l2_used_ >= kMaxL2Tables) return false;
        t = ++l2_used_;  // 1-based table number
        dir_.resize(static_cast<size_t>(kTopEntries) + static_cast<size_t>(t) * kL2Entries, 0);
        dir_[top] = t;
        touch_dir(top);
    }
    const uint32_t word = kTopEntries + (t - 1) * kL2Entries + l2_index(page);
    dir_[word] = entry;
    touch_dir(word);
    return true;
}

uint32_t Arena::lookup_entry(uint64_t addr) const {
    const uint64_t page = page_of(addr);
    const uint32_t t = dir_[top_index(page)];
    if (t == 0) return 0;
    return dir_[kTopEntries + (t - 1) * kL2Entries + l2_index(page)];
}

uint32_t Arena::load32_simulated(uint64_t addr, const uint32_t* const arenas[kArenaCount],
                                 uint64_t* fault_counter) const {
    const uint32_t e = lookup_entry(addr);
    if (!entry_valid(e)) {
        if (fault_counter != nullptr) ++(*fault_counter);
        return 0u;
    }
    const uint32_t a = entry_arena(e);
    if (arenas == nullptr || a >= arenas_ || arenas[a] == nullptr) {
        if (fault_counter != nullptr) ++(*fault_counter);
        return 0u;
    }
    const uint64_t w = static_cast<uint64_t>(entry_slot(e)) * (kPageBytes / 4u) + ((addr & (kPageBytes - 1u)) >> 2);
    return arenas[a][w];
}

bool Arena::alloc_slot(uint32_t* arena, uint32_t* slot) {
    uint64_t packed;
    if (!free_slots_.empty()) {
        packed = free_slots_.back();
        free_slots_.pop_back();
    } else if (next_fresh_ < capacity_) {
        packed = ((next_fresh_ / pages_per_arena_) << 32) | (next_fresh_ % pages_per_arena_);
        ++next_fresh_;
    } else {
        return false;
    }
    *arena = static_cast<uint32_t>(packed >> 32);
    *slot = static_cast<uint32_t>(packed);
    return true;
}

bool Arena::evict_one() {
    for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
        auto p = pages_.find(*it);
        Info& i = p->second;
        if (i.pins > 0 || i.epoch == epoch_ || i.gpu_owned) continue;
        free_slots_.push_back((static_cast<uint64_t>(entry_arena(i.entry)) << 32) | entry_slot(i.entry));
        set_entry(p->first, 0);
        lru_.erase(i.lru);
        pages_.erase(p);
        ++stats_.evictions;
        return true;
    }
    return false;
}

void Arena::unpin_pages(const std::vector<uint64_t>& pages) {
    for (uint64_t p : pages) {
        auto pi = pages_.find(p);
        if (pi != pages_.end() && pi->second.pins > 0) --pi->second.pins;
    }
}

Ensure Arena::ensure_page(uint64_t page, bool pin) {
    if (page >> (kAddressBits - kPageShift)) {
        ++stats_.out_of_range;
        return Ensure::OutOfRange;
    }
    auto it = pages_.find(page);
    if (it != pages_.end()) {
        Info& i = it->second;
        i.last_frame = frame_;
        i.epoch = epoch_;
        lru_.splice(lru_.begin(), lru_, i.lru);
        if (pin) ++i.pins;
        if (i.dirty) {
            i.dirty = false;
            if (i.gpu_owned) {
                ++stats_.skipped_gpu_owned;
            } else {
                uploads_.push_back({page, entry_arena(i.entry), entry_slot(i.entry)});
                ++stats_.uploads;
                return Ensure::Uploaded;
            }
        }
        return Ensure::Resident;
    }
    // A fresh page may need a directory table; refuse before consuming a slot.
    if (dir_[top_index(page)] == 0 && l2_used_ >= kMaxL2Tables) {
        ++stats_.table_full;
        return Ensure::TableFull;
    }
    uint32_t arena = 0, slot = 0;
    while (!alloc_slot(&arena, &slot)) {
        if (!evict_one()) {
            ++stats_.arena_full;
            return Ensure::ArenaFull;
        }
    }
    Info info;
    info.entry = pack_entry(arena, slot);
    info.last_frame = frame_;
    info.epoch = epoch_;
    info.pins = pin ? 1 : 0;
    lru_.push_front(page);
    info.lru = lru_.begin();
    set_entry(page, info.entry);
    pages_.emplace(page, info);
    uploads_.push_back({page, arena, slot});
    ++stats_.uploads;
    return Ensure::Uploaded;
}

Ensure Arena::ensure_range(uint64_t addr, uint64_t size, bool pin) {
    if (range_overflows_40bit(addr, size)) {
        ++stats_.out_of_range;
        return Ensure::OutOfRange;
    }
    const uint64_t span = size ? size : 1ull;
    const uint64_t masked = addr & kAddressMask;
    const uint64_t first = page_of(masked);
    const uint64_t last = page_of(masked + span - 1ull);
    Ensure worst = Ensure::Resident;
    for (uint64_t p = first; p <= last; ++p) {
        const Ensure e = ensure_page(p, pin);
        if (e == Ensure::ArenaFull || e == Ensure::TableFull || e == Ensure::OutOfRange) return e;
        if (e == Ensure::Uploaded) worst = e;
    }
    return worst;
}

Ensure Arena::pin_structure(uint64_t id, uint64_t addr, uint64_t size) {
    if (range_overflows_40bit(addr, size)) {
        ++stats_.out_of_range;
        return Ensure::OutOfRange;
    }
    release_structure(id);
    const uint64_t span = size ? size : 1ull;
    const uint64_t masked = addr & kAddressMask;
    const uint64_t first = page_of(masked);
    const uint64_t last = page_of(masked + span - 1ull);

    std::vector<uint64_t> done;
    if (!recycled_page_lists_.empty()) {
        done = std::move(recycled_page_lists_.back());
        recycled_page_lists_.pop_back();
        done.clear();
    }
    const uint64_t count = last - first + 1ull;
    if (done.capacity() < count && count <= capacity_) {
        done.reserve(static_cast<size_t>(count));
    }

    Ensure worst = Ensure::Resident;
    for (uint64_t p = first; p <= last; ++p) {
        const Ensure e = ensure_page(p, true);
        if (e == Ensure::ArenaFull || e == Ensure::TableFull || e == Ensure::OutOfRange) {
            unpin_pages(done);
            if (recycled_page_lists_.size() < kMaxRecycledPageLists) {
                done.clear();
                recycled_page_lists_.push_back(std::move(done));
            }
            return e;
        }
        if (e == Ensure::Uploaded) worst = e;
        done.push_back(p);
    }
    structures_[id] = std::move(done);
    return worst;
}

void Arena::release_structure(uint64_t id) {
    auto it = structures_.find(id);
    if (it == structures_.end()) return;
    unpin_pages(it->second);
    if (recycled_page_lists_.size() < kMaxRecycledPageLists) {
        it->second.clear();
        recycled_page_lists_.push_back(std::move(it->second));
    }
    structures_.erase(it);
}

Ensure Arena::stage_structure(uint64_t structure_id, uint64_t gpu_addr, uint64_t byte_size,
                              bool mark_gpu_owned, std::vector<StagedWrite>* out_writes) {
    if (out_writes != nullptr) {
        out_writes->clear();
    }
    if (range_overflows_40bit(gpu_addr, byte_size)) {
        ++stats_.out_of_range;
        return Ensure::OutOfRange;
    }
    const Ensure res = pin_structure(structure_id, gpu_addr, byte_size);
    if (res == Ensure::ArenaFull || res == Ensure::TableFull || res == Ensure::OutOfRange) {
        return res;
    }
    if (mark_gpu_owned && byte_size > 0) {
        mark_gpu_written_range(gpu_addr, byte_size);
    }
    if (out_writes != nullptr && byte_size > 0) {
        const uint64_t masked = gpu_addr & kAddressMask;
        uint64_t src_off = 0;
        while (src_off < byte_size) {
            const uint64_t cur_addr = masked + src_off;
            const uint64_t page_off = cur_addr & (kPageBytes - 1u);
            const uint64_t chunk = std::min<uint64_t>(byte_size - src_off, kPageBytes - page_off);
            const uint32_t e = lookup_entry(cur_addr);
            if (!entry_valid(e)) {
                return Ensure::ArenaFull;
            }
            out_writes->push_back({
                entry_arena(e),
                slot_byte_offset(entry_slot(e)) + page_off,
                src_off,
                static_cast<uint32_t>(chunk),
            });
            src_off += chunk;
        }
    }
    ++stats_.staged_structures;
    stats_.staged_bytes += byte_size;
    return res;
}

void Arena::set_scratch_budget_pages(uint32_t max_pages) {
    scratch_budget_pages_ = static_cast<uint32_t>(std::min<uint64_t>(max_pages, capacity_));
    trim_recycled_scratch_to_budget(0u);
}

void Arena::set_scratch_base(uint64_t base_addr) {
    scratch_base_ = (base_addr & kAddressMask) & ~(kPageBytes - 1u);
    scratch_next_page_ = scratch_base_ >> kPageShift;
}

void Arena::trim_recycled_scratch_to_budget(uint32_t reserve_for_new) {
    while (!recycled_scratch_.empty() &&
           static_cast<uint64_t>(scratch_live_pages_) + scratch_recycled_pages_ + reserve_for_new >
               scratch_budget_pages_) {
        const RecycledScratch old = recycled_scratch_.front();
        recycled_scratch_.erase(recycled_scratch_.begin());
        scratch_recycled_pages_ -= old.page_count;
        // Release GPU ownership on discarded scratch pages so normal LRU eviction can reclaim them.
        for (uint32_t i = 0; i < old.page_count; ++i) {
            auto it = pages_.find(old.first_page + i);
            if (it != pages_.end()) {
                it->second.gpu_owned = false;
                it->second.dirty = false;
            }
        }
        ++stats_.scratch_evictions;
    }
}

Ensure Arena::acquire_scratch(uint64_t byte_size, ScratchAllocation* out) {
    if (out != nullptr) {
        *out = {};
    }
    const uint64_t span = byte_size ? byte_size : 1ull;
    const uint64_t needed64 = (span + kPageBytes - 1u) / kPageBytes;
    if (needed64 > scratch_budget_pages_ || needed64 > capacity_) {
        ++stats_.scratch_budget_exceeded;
        ++stats_.arena_full;
        return Ensure::ArenaFull;
    }
    const uint32_t needed = static_cast<uint32_t>(needed64);

    // 1. Try best-fit recycle from previously released scratch blocks (zero new pages / zero allocs).
    size_t best_idx = recycled_scratch_.size();
    uint32_t best_pages = 0xFFFFFFFFu;
    for (size_t i = 0; i < recycled_scratch_.size(); ++i) {
        const auto& r = recycled_scratch_[i];
        if (r.page_count >= needed && r.page_count < best_pages) {
            best_pages = r.page_count;
            best_idx = i;
            if (r.page_count == needed) break;
        }
    }
    if (best_idx < recycled_scratch_.size()) {
        const RecycledScratch chosen = recycled_scratch_[best_idx];
        recycled_scratch_.erase(recycled_scratch_.begin() + static_cast<ptrdiff_t>(best_idx));
        scratch_recycled_pages_ -= chosen.page_count;

        Ensure worst = Ensure::Resident;
        for (uint32_t i = 0; i < chosen.page_count; ++i) {
            const Ensure e = ensure_page(chosen.first_page + i, true);
            if (e == Ensure::ArenaFull || e == Ensure::TableFull || e == Ensure::OutOfRange) {
                for (uint32_t j = 0; j < i; ++j) {
                    auto it = pages_.find(chosen.first_page + j);
                    if (it != pages_.end() && it->second.pins > 0) --it->second.pins;
                }
                return e;
            }
            mark_gpu_written(chosen.first_page + i);
            if (e == Ensure::Uploaded) worst = e;
        }
        scratch_live_pages_ += chosen.page_count;
        active_scratch_[chosen.pool_id] = chosen;
        ++stats_.scratch_reuses;
        if (out != nullptr) {
            out->gpu_addr = chosen.first_page << kPageShift;
            out->size_bytes = static_cast<uint64_t>(chosen.page_count) * kPageBytes;
            out->first_page = chosen.first_page;
            out->page_count = chosen.page_count;
            out->pool_id = chosen.pool_id;
        }
        return worst;
    }

    // 2. Evict unpinned recycled blocks if needed to fit within bounded scratch budget.
    if (static_cast<uint64_t>(scratch_live_pages_) + needed > scratch_budget_pages_) {
        ++stats_.scratch_budget_exceeded;
        ++stats_.arena_full;
        return Ensure::ArenaFull;
    }
    trim_recycled_scratch_to_budget(needed);

    const uint64_t max_page = kAddressMask >> kPageShift;
    if (scratch_next_page_ > max_page || needed - 1u > max_page - scratch_next_page_) {
        ++stats_.out_of_range;
        return Ensure::OutOfRange;
    }
    const uint64_t first_page = scratch_next_page_;
    Ensure worst = Ensure::Resident;
    for (uint32_t i = 0; i < needed; ++i) {
        const Ensure e = ensure_page(first_page + i, true);
        if (e == Ensure::ArenaFull || e == Ensure::TableFull || e == Ensure::OutOfRange) {
            for (uint32_t j = 0; j < i; ++j) {
                auto it = pages_.find(first_page + j);
                if (it != pages_.end() && it->second.pins > 0) --it->second.pins;
            }
            return e;
        }
        mark_gpu_written(first_page + i);
        if (e == Ensure::Uploaded) worst = e;
    }

    scratch_next_page_ += needed;
    scratch_live_pages_ += needed;
    const uint32_t pool_id = scratch_next_id_++;
    const RecycledScratch entry{first_page, needed, pool_id};
    active_scratch_[pool_id] = entry;
    if (out != nullptr) {
        out->gpu_addr = first_page << kPageShift;
        out->size_bytes = static_cast<uint64_t>(needed) * kPageBytes;
        out->first_page = first_page;
        out->page_count = needed;
        out->pool_id = pool_id;
    }
    return worst;
}

void Arena::release_scratch(const ScratchAllocation& alloc) {
    if (!alloc.valid()) return;
    auto it = active_scratch_.find(alloc.pool_id);
    if (it == active_scratch_.end()) return;
    const RecycledScratch entry = it->second;
    active_scratch_.erase(it);

    for (uint32_t i = 0; i < entry.page_count; ++i) {
        auto pi = pages_.find(entry.first_page + i);
        if (pi != pages_.end() && pi->second.pins > 0) {
            --pi->second.pins;
        }
    }
    scratch_live_pages_ -= entry.page_count;
    recycled_scratch_.push_back(entry);
    scratch_recycled_pages_ += entry.page_count;
    trim_recycled_scratch_to_budget(0u);
}

void Arena::mark_gpu_written(uint64_t page) {
    auto it = pages_.find(page);
    if (it != pages_.end()) {
        it->second.gpu_owned = true;
        it->second.dirty = false;
    }
}

void Arena::mark_gpu_written_range(uint64_t addr, uint64_t size) {
    if (size == 0) return;
    const uint64_t masked = addr & kAddressMask;
    const uint64_t end_addr = (size - 1ull > kAddressMask - masked) ? kAddressMask : (masked + size - 1ull);
    for (uint64_t p = page_of(masked); p <= page_of(end_addr); ++p) {
        mark_gpu_written(p);
    }
}

void Arena::mark_cpu_remapped(uint64_t addr, uint64_t size) {
    if (size == 0) return;
    const uint64_t masked = addr & kAddressMask;
    const uint64_t end_addr = (size - 1ull > kAddressMask - masked) ? kAddressMask : (masked + size - 1ull);
    for (uint64_t p = page_of(masked); p <= page_of(end_addr); ++p) {
        auto it = pages_.find(p);
        if (it == pages_.end()) continue;
        it->second.gpu_owned = false;
        it->second.dirty = true;
    }
}

void Arena::mark_cpu_dirty(uint64_t page) {
    auto it = pages_.find(page);
    if (it != pages_.end()) it->second.dirty = true;
}

bool Arena::is_resident(uint64_t page) const { return pages_.count(page) != 0; }

std::vector<Upload> Arena::take_uploads() {
    std::vector<Upload> out;
    out.swap(uploads_);
    return out;
}

void Arena::take_uploads_into(std::vector<Upload>& out) {
    out.clear();
    if (out.capacity() < uploads_.size()) {
        out.reserve(uploads_.size());
    }
    out.insert(out.end(), uploads_.begin(), uploads_.end());
    uploads_.clear();
}

void Arena::reserve_upload_capacity(size_t count) {
    uploads_.reserve(count);
    recycled_scratch_.reserve(std::min<size_t>(count, 64u));
}

Stats Arena::stats() const {
    Stats s = stats_;
    s.resident = pages_.size();
    s.scratch_live_pages = scratch_live_pages_;
    s.scratch_recycled_pages = scratch_recycled_pages_;
    for (const auto& kv : pages_) {
        if (kv.second.pins) ++s.pinned;
        if (kv.second.gpu_owned) ++s.gpu_owned;
    }
    return s;
}

std::string Arena::json() const {
    const Stats s = stats();
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "{\"arena_bytes\": %llu, \"pages_resident\": %llu, \"pages_pinned\": %llu, "
                  "\"pages_gpu_owned\": %llu, \"uploads\": %llu, \"evictions\": %llu, "
                  "\"skipped_gpu_owned\": %llu, \"arena_full\": %llu, \"table_full\": %llu, "
                  "\"out_of_range\": %llu, \"rt_fault\": %llu, \"dispatches\": %llu, \"unreadable_pages\": %llu, "
                  "\"dir_words\": %zu, \"scratch_live_pages\": %llu, \"scratch_recycled_pages\": %llu, "
                  "\"scratch_reuses\": %llu, \"scratch_evictions\": %llu, \"scratch_budget_exceeded\": %llu, "
                  "\"staged_structures\": %llu, \"staged_bytes\": %llu, \"skips\": {",
                  (unsigned long long)capacity_bytes(), (unsigned long long)s.resident,
                  (unsigned long long)s.pinned, (unsigned long long)s.gpu_owned,
                  (unsigned long long)s.uploads, (unsigned long long)s.evictions,
                  (unsigned long long)s.skipped_gpu_owned, (unsigned long long)s.arena_full,
                  (unsigned long long)s.table_full, (unsigned long long)s.out_of_range,
                  (unsigned long long)s.faults, (unsigned long long)s.dispatches,
                  (unsigned long long)s.unreadable_pages, dir_.size(),
                  (unsigned long long)s.scratch_live_pages, (unsigned long long)s.scratch_recycled_pages,
                  (unsigned long long)s.scratch_reuses, (unsigned long long)s.scratch_evictions,
                  (unsigned long long)s.scratch_budget_exceeded, (unsigned long long)s.staged_structures,
                  (unsigned long long)s.staged_bytes);
    std::string out = buf;
    for (size_t i = 0; i < static_cast<size_t>(SkipReason::Count); ++i) {
        if (i) out += ", ";
        std::snprintf(buf, sizeof buf, "\"%s\": %llu", skip_reason_name(static_cast<SkipReason>(i)),
                      (unsigned long long)s.skips[i]);
        out += buf;
    }
    return out + "}}";
}

namespace {
std::mutex g_report_mutex;
const void* g_report_owner = nullptr;
std::function<std::string()> g_report_provider;
}  // namespace

void set_report_provider(const void* owner, std::function<std::string()> provider) {
    std::lock_guard<std::mutex> lock(g_report_mutex);
    g_report_owner = owner;
    g_report_provider = std::move(provider);
}

void clear_report_provider(const void* owner) {
    std::lock_guard<std::mutex> lock(g_report_mutex);
    if (g_report_owner == owner) {
        g_report_owner = nullptr;
        g_report_provider = nullptr;
    }
}

std::string report_json() {
    std::function<std::string()> provider;
    {
        std::lock_guard<std::mutex> lock(g_report_mutex);
        provider = g_report_provider;
    }
    return provider ? provider() : std::string("{}");
}

}  // namespace playgta6::gpu::rt_arena
