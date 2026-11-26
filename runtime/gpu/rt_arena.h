// Paged GPU address arena for ray tracing / raw-address compute (docs/RAYTRACING_DESIGN.md section 3,
// stage R0/R2 runtime half). Pure bookkeeping: the 40-bit address -> page mapping, the two-level
// page directory packed exactly as the WGSL prelude reads it, residency with LRU eviction and
// pinning, GPU-ownership of written pages, bounded scratch page recycling, 3x4 instance packing &
// structure staging, and counters. The caller owns the actual buffers and performs the uploads this
// class lists. Contract C3 bindings: group 3, bindings 16-21.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace playgta6::gpu::rt_arena {

constexpr uint32_t kPageShift = 16;
constexpr uint64_t kPageBytes = 1ull << kPageShift;  // 64 KiB
constexpr uint32_t kArenaCount = 4;
constexpr uint64_t kArenaBytes = 512ull << 20;  // 512 MiB each
constexpr uint32_t kPagesPerArena = static_cast<uint32_t>(kArenaBytes / kPageBytes);  // 8192
constexpr uint32_t kAddressBits = 40;
constexpr uint64_t kAddressMask = (1ull << kAddressBits) - 1;
constexpr uint32_t kTopEntries = 4096;  // one per 256 MiB
constexpr uint32_t kL2Entries = 4096;   // one per 64 KiB page
constexpr uint32_t kMaxL2Tables = 56;   // keeps the directory under 1 MiB

// Default high virtual window for bounded RT build/refit scratch pages (40-bit space, page-aligned).
constexpr uint64_t kDefaultScratchBase = 0xFE00000000ull;
constexpr uint32_t kDefaultScratchBudgetPages = 256;  // 16 MiB default cap
constexpr uint32_t kInstance3x4Dwords = 16;           // 64-byte kind-6 instance node

// Contract C3 binding slots (group 3).
constexpr uint32_t kBindGroup = 3;
constexpr uint32_t kBindArena0 = 16;  // 16..19 = pg6_arena0..3
constexpr uint32_t kBindDirectory = 20;  // pg6_arena_dir
constexpr uint32_t kBindStats = 21;      // pg6_rt_stats
constexpr uint32_t kStatFaults = 0;      // index 0 of pg6_rt_stats

// Second-level entry: bit 31 valid, bits 30:29 arena, bits 28:0 page slot. 0 = not resident.
constexpr uint32_t kEntryValid = 0x80000000u;
constexpr uint32_t pack_entry(uint32_t arena, uint32_t slot) {
    return kEntryValid | ((arena & 3u) << 29) | (slot & 0x1fffffffu);
}
constexpr bool entry_valid(uint32_t e) { return (e & kEntryValid) != 0; }
constexpr uint32_t entry_arena(uint32_t e) { return (e >> 29) & 3u; }
constexpr uint32_t entry_slot(uint32_t e) { return e & 0x1fffffffu; }

constexpr uint64_t page_of(uint64_t addr) { return (addr & kAddressMask) >> kPageShift; }
constexpr uint32_t top_index(uint64_t page) { return static_cast<uint32_t>(page >> 12); }
constexpr uint32_t l2_index(uint64_t page) { return static_cast<uint32_t>(page & 0xfff); }
// Byte offset of a page slot inside its arena buffer.
constexpr uint64_t slot_byte_offset(uint32_t slot) { return static_cast<uint64_t>(slot) * kPageBytes; }

enum class SkipReason : uint8_t { RtOff, ArenaFull, NotResident, Atomic64, TranslatorPending, Count };
const char* skip_reason_name(SkipReason r);

enum class Ensure : uint8_t { Resident, Uploaded, ArenaFull, TableFull, OutOfRange };

struct Upload {
    uint64_t page;
    uint32_t arena;
    uint32_t slot;
};

// Contiguous arena slice produced by stage_structure() for zero-alloc batched uploads into
// pg6_arena0..3 matching pg6_load32 page offsets.
struct StagedWrite {
    uint32_t arena = 0;
    uint64_t arena_byte_offset = 0;
    uint64_t src_byte_offset = 0;
    uint32_t byte_size = 0;
};

// Bounded recyclable scratch page range in 40-bit GPU address space.
struct ScratchAllocation {
    uint64_t gpu_addr = 0;
    uint64_t size_bytes = 0;
    uint64_t first_page = 0;
    uint32_t page_count = 0;
    uint32_t pool_id = 0;

    bool valid() const { return page_count != 0; }
};

// 3x4 row-major affine instance descriptor packed into the 64-byte kind-6 instance node layout
// matching runtime/gpu/rt_bvh.h and S65 WGSL group-3 arena layout.
struct Instance3x4 {
    float transform[3][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    };
    uint32_t instance_id = 0;             // 24-bit
    uint8_t instance_mask = 0xFFu;        // 8-bit
    uint32_t hit_group_contribution = 0;  // 24-bit
    uint8_t flags = 0;                    // 8-bit
    uint64_t blas_address = 0;            // 40-bit, 256-byte aligned
    uint32_t blas_root_node = 0xFFFFFFFFu;
    bool transform_is_inverse = false;    // false => invert transform into world->object 3x4
};

// Packs one 3x4 instance descriptor into a 64-byte (16-dword) kind-6 instance node.
// Returns false if blas_address exceeds 40 bits / is misaligned or transform is singular.
bool pack_instance_3x4(const Instance3x4& desc, uint32_t out_dwords[kInstance3x4Dwords]);

// Batch-packs `count` instances into `out_dwords` (zero heap allocations). Returns number packed.
size_t pack_instances_3x4(const Instance3x4* instances, size_t count, uint32_t* out_dwords,
                          size_t out_dword_capacity);

struct Stats {
    uint64_t resident = 0, pinned = 0, gpu_owned = 0;
    uint64_t uploads = 0, evictions = 0, skipped_gpu_owned = 0;
    uint64_t arena_full = 0, table_full = 0, out_of_range = 0;
    uint64_t faults = 0;  // last value read back from pg6_rt_stats[0]
    uint64_t dispatches = 0;  // RT-class dispatches recorded
    uint64_t unreadable_pages = 0;  // uploads whose guest memory could not be read (zero page sent)
    uint64_t skips[static_cast<size_t>(SkipReason::Count)] = {};
    // Bounded scratch recycling & structure staging counters (additive).
    uint64_t scratch_live_pages = 0;
    uint64_t scratch_recycled_pages = 0;
    uint64_t scratch_reuses = 0;
    uint64_t scratch_evictions = 0;
    uint64_t scratch_budget_exceeded = 0;
    uint64_t staged_structures = 0;
    uint64_t staged_bytes = 0;
};

class Arena {
public:
    // `arenas` and `pages_per_arena` are overridable for tests / smaller adapters.
    explicit Arena(uint32_t arenas = kArenaCount, uint32_t pages_per_arena = kPagesPerArena);

    uint64_t capacity_pages() const { return capacity_; }
    uint64_t capacity_bytes() const { return capacity_ * kPageBytes; }

    // Frame boundary: advances the LRU clock.
    void begin_frame() { ++frame_; }
    // Starts one RT-class dispatch: pages ensured since are protected from eviction until the next one.
    void begin_dispatch() { ++epoch_; }

    // Makes `page` resident (pinned when `pin`). New or re-dirtied pages are appended to the pending
    // upload list unless the page is GPU-authoritative.
    Ensure ensure_page(uint64_t page, bool pin = false);
    // Makes every page overlapping [addr, addr+size) resident. Returns the first failure (pages made
    // resident before it stay). size 0 touches one page.
    Ensure ensure_range(uint64_t addr, uint64_t size, bool pin = false);

    // Structures: pin a page set while the structure is live; release unpins it.
    Ensure pin_structure(uint64_t id, uint64_t addr, uint64_t size);
    void release_structure(uint64_t id);

    // Stages a BLAS/TLAS/instance buffer into pinned arena pages and appends page-aligned upload
    // slices into `out_writes` (reusing caller capacity). Optionally marks pages GPU-owned.
    Ensure stage_structure(uint64_t structure_id, uint64_t gpu_addr, uint64_t byte_size,
                           bool mark_gpu_owned, std::vector<StagedWrite>* out_writes);

    // Bounded scratch page pool for dynamic BLAS/TLAS builds and refits.
    void set_scratch_budget_pages(uint32_t max_pages);
    uint32_t scratch_budget_pages() const { return scratch_budget_pages_; }
    void set_scratch_base(uint64_t base_addr);
    uint64_t scratch_base() const { return scratch_base_; }
    Ensure acquire_scratch(uint64_t byte_size, ScratchAllocation* out);
    void release_scratch(const ScratchAllocation& alloc);

    // Ownership. A builder/trace program wrote the page: CPU uploads skip it until freed/remapped.
    void mark_gpu_written(uint64_t page);
    void mark_gpu_written_range(uint64_t addr, uint64_t size);
    // Title freed or remapped [addr, addr+size): drops GPU ownership and marks pages dirty.
    void mark_cpu_remapped(uint64_t addr, uint64_t size);
    // Guest memory changed under a page (write-mark): reupload if resident and not GPU-owned.
    void mark_cpu_dirty(uint64_t page);

    bool is_resident(uint64_t page) const;
    // 0 when not resident. Mirrors exactly what the WGSL prelude computes.
    uint32_t lookup_entry(uint64_t addr) const;

    // CPU mirror of WGSL `pg6_load32(addr)` against caller-owned arena backing buffers.
    uint32_t load32_simulated(uint64_t addr, const uint32_t* const arenas[kArenaCount],
                              uint64_t* fault_counter = nullptr) const;

    // Pending uploads (clears the list).
    std::vector<Upload> take_uploads();
    // Zero-alloc steady-state variant: moves pending uploads into caller's reserved vector.
    void take_uploads_into(std::vector<Upload>& out);
    void reserve_upload_capacity(size_t count);

    // Directory words (top level, then L2 tables) and the [first,last] word range changed since the
    // last take_dir_dirty().
    const std::vector<uint32_t>& directory() const { return dir_; }
    bool take_dir_dirty(uint32_t* first, uint32_t* last);

    void note_skip(SkipReason r) { ++stats_.skips[static_cast<size_t>(r)]; }
    void note_dispatch() { ++stats_.dispatches; }
    void note_unreadable() { ++stats_.unreadable_pages; }
    void set_fault_count(uint64_t f) { stats_.faults = f; }
    Stats stats() const;
    std::string json() const;

private:
    struct Info {
        uint32_t entry = 0;
        uint64_t last_frame = 0, epoch = 0;
        uint32_t pins = 0;
        bool gpu_owned = false, dirty = false;
        std::list<uint64_t>::iterator lru;
    };
    struct RecycledScratch {
        uint64_t first_page = 0;
        uint32_t page_count = 0;
        uint32_t pool_id = 0;
    };

    bool alloc_slot(uint32_t* arena, uint32_t* slot);
    bool evict_one();
    bool set_entry(uint64_t page, uint32_t entry);  // false: L2 table limit
    void touch_dir(uint32_t word);
    void trim_recycled_scratch_to_budget(uint32_t reserve_for_new);
    void unpin_pages(const std::vector<uint64_t>& pages);

    uint32_t arenas_, pages_per_arena_;
    uint64_t capacity_, frame_ = 1, epoch_ = 1, next_fresh_ = 0;
    std::vector<uint32_t> dir_;
    uint32_t l2_used_ = 0;
    uint32_t dirty_first_ = 0xffffffffu, dirty_last_ = 0;
    std::unordered_map<uint64_t, Info> pages_;
    std::list<uint64_t> lru_;           // front = most recent
    std::vector<uint64_t> free_slots_;  // packed arena<<32|slot
    std::unordered_map<uint64_t, std::vector<uint64_t>> structures_;
    std::vector<std::vector<uint64_t>> recycled_page_lists_;
    std::vector<Upload> uploads_;

    // Bounded scratch page recycling state.
    uint64_t scratch_base_ = kDefaultScratchBase;
    uint32_t scratch_budget_pages_ = kDefaultScratchBudgetPages;
    uint64_t scratch_next_page_ = kDefaultScratchBase >> kPageShift;
    uint32_t scratch_next_id_ = 1;
    uint32_t scratch_live_pages_ = 0;
    uint32_t scratch_recycled_pages_ = 0;
    std::vector<RecycledScratch> recycled_scratch_;
    std::unordered_map<uint32_t, RecycledScratch> active_scratch_;

    Stats stats_;
};

// Boot-report plumbing: a bridge device registers a provider for its arena's json() (owner = the
// device, replaced on re-register, removed on destruction); the report asks for the current one.
// "{}" when none is registered (no RT-class dispatch has happened yet).
void set_report_provider(const void* owner, std::function<std::string()> provider);
void clear_report_provider(const void* owner);
std::string report_json();

}  // namespace playgta6::gpu::rt_arena
