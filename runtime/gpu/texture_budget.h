#pragma once

// Host GPU memory budget and streamed-texture size cap (backlog 37).
//
// All textures of the title together do not fit a typical host GPU; paging then costs seconds per
// frame or the device is lost. The budget tracker counts the bytes of live host textures against a
// configured budget and derives a size cap from the share in use: below kTextureCapMediumPercent of
// the budget only the configured base cap applies (0 = none), from there up to
// kTextureCapHighPercent the cap is 512 px, above that 256 px. A texture whose largest side is over
// the cap uploads only the mip levels at or below it (the top levels are dropped and the texture's
// views start at the first held level: a lower resolution, not invented content). When memory
// frees and the cap loosens, the dropped levels are uploaded again the next time the texture is
// bound.
//
// A capped texture whose host allocation can be replaced is also *reallocated* at the capped size
// (fewer levels, smaller base; the bridge maps guest mip indices onto the host levels), so the
// dropped levels cost no GPU memory; the budget counts the smaller chain. Loosening reallocates at
// the larger size and re-uploads from guest memory, but only when the restored use still stays
// below the pressure tier the cap came from (cap_with_extra / restore_fits), so a texture does not
// flap between the sizes as its own shrinking moves the use across a threshold.
//
// Pure bookkeeping, thread-safe, no GPU packets.

#include <atomic>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace playgta6::gpu {

inline constexpr uint32_t kTextureCapMediumPercent = 70;
inline constexpr uint32_t kTextureCapHighPercent = 85;
inline constexpr uint32_t kTextureCapMedium = 512;
inline constexpr uint32_t kTextureCapHigh = 256;

// VRAM LRU reclamation pressure tiers & hysteresis watermarks.
// Soft reclamation begins at 80% of budget and continues until usage drops to the 68% target
// watermark (hysteresis prevents thrashing around a single threshold). Above 92% (Critical),
// aggressive reclamation evicts younger non-current-frame textures while still protecting pinned
// resources (back buffers, render targets, depth/stencil) and textures used in the current frame.
inline constexpr uint32_t kTextureReclaimTriggerPercent = 80;
inline constexpr uint32_t kTextureReclaimTargetPercent = 68;
inline constexpr uint32_t kTextureReclaimCriticalPercent = 92;
inline constexpr uint64_t kTextureLruDefaultProtectFrames = 2;

enum class BudgetPressureTier : uint8_t {
    Normal = 0,    // < 70%: base cap only, no forced reclamation
    Medium = 1,    // [70%, 80%): 512 px cap, no forced reclamation
    Reclaim = 2,   // [80%, 85%): 512 px cap + soft LRU reclamation toward 68%
    High = 3,      // [85%, 92%): 256 px cap + LRU reclamation toward 68%
    Critical = 4,  // >= 92%: 256 px cap + aggressive LRU reclamation (only pinned & current-frame protected)
};

// Bytes of the mip chain (levels start_mip..mips-1) of a width x height x depth (x array_layers)
// texture whose blocks are block_w x block_h texels of block_bytes bytes.
// For 3D volumes, pass depth > 1 and array_layers = 1 (depth halves each mip).
// For 2D arrays / cubemaps, pass depth = 1 and array_layers = slices (constant across mips).
uint64_t texture_chain_bytes(uint32_t width, uint32_t height, uint32_t depth, uint32_t mips, uint32_t block_w,
                             uint32_t block_h, uint32_t block_bytes, uint32_t start_mip = 0,
                             uint32_t array_layers = 1) noexcept;

// Number of top mip levels to drop so no side of the first level that remains exceeds `cap`
// texels. 0 when cap is 0 (no cap) or the texture already fits; never drops the last level.
uint32_t cap_floor_for(uint32_t width, uint32_t height, uint32_t mips, uint32_t cap) noexcept;

// 3D variant of cap_floor_for: considers width, height, and depth.
uint32_t cap_floor_for_3d(uint32_t width, uint32_t height, uint32_t depth, uint32_t mips, uint32_t cap) noexcept;

// The cap for a given memory use: base_cap (0 = none) below the medium threshold, then the
// tighter of base_cap and the pressure tier. budget_bytes == 0 means no budget (base cap only).
uint32_t cap_for_pressure(uint64_t used_bytes, uint64_t budget_bytes, uint32_t base_cap) noexcept;

// Pressure tier derived from `used_bytes` / `budget_bytes` (Normal when budget_bytes == 0).
BudgetPressureTier pressure_tier_for(uint64_t used_bytes, uint64_t budget_bytes) noexcept;

// Bytes that should be reclaimed via LRU eviction given current usage, budget, hysteresis latch
// `reclaim_active`, and optional `incoming_bytes` about to be allocated. Returns 0 when no
// reclamation is needed (budget_bytes == 0 or usage below trigger and not already reclaiming).
uint64_t reclaim_target_bytes(uint64_t used_bytes, uint64_t budget_bytes, bool reclaim_active = false,
                              uint64_t incoming_bytes = 0) noexcept;

struct LruEvictCandidate {
    uint32_t resource_id = 0;
    uint64_t bytes = 0;
    uint64_t last_used_tick = 0;
    uint64_t last_used_frame = 0;
};

struct TextureBudgetStats {
    uint64_t budget_bytes = 0;
    uint64_t used_bytes = 0;
    uint64_t peak_used_bytes = 0;
    uint32_t base_cap = 0;
    uint32_t current_cap = 0;         // 0 = none
    uint64_t cap_changes = 0;         // times the effective cap moved
    uint64_t textures_capped = 0;     // upload passes that dropped at least one top level
    uint64_t mips_dropped = 0;        // top levels not uploaded, summed over those passes
    uint64_t bytes_not_uploaded = 0;  // texel bytes those levels would have taken
    uint64_t textures_restored = 0;   // textures that got dropped levels back after the cap loosened
    uint64_t host_reallocs = 0;       // host textures re-created at another size by the cap
    uint64_t host_bytes_saved = 0;    // bytes the currently shrunk host textures are below full size
    // LRU residency & reclamation telemetry:
    uint64_t lru_tracked_textures = 0;  // textures currently tracked in the LRU residency table
    uint64_t lru_tracked_bytes = 0;     // total bytes of currently tracked LRU textures
    uint64_t lru_evictions = 0;         // textures evicted via LRU reclamation
    uint64_t lru_bytes_reclaimed = 0;   // total bytes reclaimed via LRU eviction
    uint64_t lru_pinned_textures = 0;   // currently pinned (protected) textures
    uint8_t pressure_tier = 0;          // static_cast<uint8_t>(BudgetPressureTier)
    bool reclaim_active = false;        // hysteresis latch: reclaiming down to low watermark
};

class TextureBudget {
public:
    void set_budget_bytes(uint64_t bytes) noexcept;
    void set_base_cap(uint32_t cap) noexcept;
    void reset() noexcept;  // tests and per-run reports: counters, use, LRU state and configuration

    // A host texture of `bytes` was created / destroyed.
    void note_created(uint64_t bytes) noexcept;
    void note_destroyed(uint64_t bytes) noexcept;

    // The cap in force now (re-derived from current use; logs when it moves).
    uint32_t current_cap() noexcept;

    // Records that an upload pass dropped `mips` top levels (`bytes` of texels) of one texture.
    void note_capped(uint32_t mips, uint64_t bytes) noexcept;
    void note_restored() noexcept { restored_.fetch_add(1, std::memory_order_relaxed); }

    // The cap that would be in force if `extra_bytes` more were in use (a restore that grows a texture).
    uint32_t cap_with_extra(uint64_t extra_bytes) const noexcept;
    // True when growing a width x height x mips texture to start at level `target_floor`, costing
    // `extra_bytes` more, still leaves a cap that allows that floor (hysteresis for restores).
    bool restore_fits(uint32_t width, uint32_t height, uint32_t mips, uint32_t target_floor,
                      uint64_t extra_bytes) const noexcept;

    // A host texture was re-created at another size: `saved_delta` is the change of its bytes below
    // full size (positive = shrunk further). The used bytes move by the same amount.
    void note_host_realloc(int64_t saved_delta) noexcept;

    // A shrunk texture goes away: its caller then note_destroyed()s the full size, so the saved bytes
    // are counted back in first (net: the shrunk bytes leave the use).
    void note_shrunk_released(uint64_t saved_bytes) noexcept;

    // ---- Thread-safe O(1) LRU texture residency & VRAM reclamation ----
    // Registers (or updates) a live texture in the LRU residency table.
    // `pinned` protects back buffers, render targets, depth/stencil, etc. from LRU eviction.
    // `evictable` marks whether the caller can destroy/evict this host texture under VRAM pressure.
    void register_texture(uint32_t resource_id, uint64_t bytes, uint64_t frame = 0, bool pinned = false,
                          bool evictable = true);
    // Moves `resource_id` to MRU front and records its latest access frame.
    void touch_texture(uint32_t resource_id, uint64_t frame = 0);
    // Updates the tracked host byte size of a registered texture (e.g. after cap shrink/restore).
    void update_texture_bytes(uint32_t resource_id, uint64_t new_bytes);
    // Updates pin state (e.g. when a texture is designated as the swapchain back buffer).
    void set_texture_pinned(uint32_t resource_id, bool pinned);
    // Removes a texture from LRU tracking (no-op if not registered).
    void unregister_texture(uint32_t resource_id);

    // Current pressure tier and hysteresis reclamation state.
    BudgetPressureTier current_pressure_tier() const noexcept;
    bool is_reclaim_active() const noexcept;
    // Evaluates pressure + hysteresis (latching reclaim_active on trigger, clearing at low watermark)
    // and returns how many bytes should be reclaimed (0 when below trigger).
    uint64_t bytes_to_reclaim(uint64_t incoming_bytes = 0) noexcept;

    // Selects least-recently-used evictable, unpinned textures to satisfy `target_bytes` (or, when
    // `target_bytes == 0`, the pressure-tiered hysteresis reclaim target). Protects textures touched
    // within `min_age_frames` of `current_frame` under soft/high pressure, and always protects
    // current-frame textures (`last_used_frame >= current_frame`) when `min_age_frames > 0`.
    std::vector<LruEvictCandidate> select_eviction_candidates(
        uint64_t current_frame, uint64_t target_bytes = 0, uint32_t max_candidates = 64,
        uint64_t min_age_frames = kTextureLruDefaultProtectFrames);

    // Records completion of an LRU eviction: unregisters `resource_id`, increments eviction stats,
    // and refreshes the hysteresis latch. If `bytes_reclaimed == 0`, uses the entry's tracked bytes.
    void note_lru_evicted(uint32_t resource_id, uint64_t bytes_reclaimed = 0);

    TextureBudgetStats stats() const noexcept;

private:
    struct LruEntry {
        uint32_t resource_id = 0;
        uint64_t bytes = 0;
        uint64_t last_used_tick = 0;
        uint64_t last_used_frame = 0;
        bool pinned = false;
        bool evictable = true;
    };

    void update_hysteresis_after_free_locked() noexcept;

    std::atomic<uint64_t> budget_{0};
    std::atomic<uint64_t> used_{0};
    std::atomic<uint64_t> peak_{0};
    std::atomic<uint32_t> base_cap_{0};
    std::atomic<uint32_t> last_cap_{0};
    std::atomic<uint64_t> cap_changes_{0};
    std::atomic<uint64_t> capped_{0};
    std::atomic<uint64_t> mips_dropped_{0};
    std::atomic<uint64_t> bytes_skipped_{0};
    std::atomic<uint64_t> restored_{0};
    std::atomic<uint64_t> reallocs_{0};
    std::atomic<uint64_t> saved_{0};

    mutable std::mutex lru_mutex_;
    std::list<LruEntry> lru_list_;  // front = MRU, back = LRU
    std::unordered_map<uint32_t, std::list<LruEntry>::iterator> lru_index_;
    uint64_t lru_tick_ = 0;
    uint64_t lru_tracked_bytes_ = 0;
    uint64_t lru_pinned_count_ = 0;
    uint64_t lru_evictions_ = 0;
    uint64_t lru_bytes_reclaimed_ = 0;
    std::atomic<bool> reclaim_active_{false};
};

TextureBudget& texture_budget() noexcept;

}  // namespace playgta6::gpu
