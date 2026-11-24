#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::guest {

// Resolves guest virtual addresses to shard-level dispatcher invocations
// (`Pg6ShardDispatchFn(local_idx, state)`).
//
// Multiple PE modules (e.g. `game.exe`, `vcruntime140.dll`, `msvcp140.dll`) can each register
// their own `Pg6ShardDispatchTable` via `add_table()`. Per-table `shard_idx` indices are remapped
// into a unified dispatcher vector, and entries are merged in ascending `guest_va` order.
//
// After registration completes, `lookup()` and `dispatch()` perform O(log N) binary search with
// zero allocations and are safe to call concurrently from any number of guest threads.
class ShardDispatchRegistry {
public:
    struct ResolvedShardTarget {
        Pg6ShardDispatchFn dispatcher = nullptr;
        uint32_t local_idx = 0;
    };

    // Validates and merges `table` into this registry.
    // On failure, writes a diagnostic message into `*error` (if non-null), leaves existing state
    // unmodified, and returns false.
    bool add_table(const Pg6ShardDispatchTable& table, std::string* error = nullptr);

    // Resolves `guest_va` via O(log N) binary search without allocating.
    // Returns true and populates `out` if found; otherwise resets `out` and returns false.
    bool lookup(uint64_t guest_va, ResolvedShardTarget& out) const noexcept;

    // Resolves `guest_va` and invokes `target.dispatcher(target.local_idx, state)`.
    // Returns false if `guest_va` is not registered.
    bool dispatch(uint64_t guest_va, Pg6GuestState* state) const;

    // Finds the highest registered entry with `entry.guest_va <= guest_va` via O(log N) binary
    // search without allocating (for fault diagnostics / `sub_XXXX+0xYY` formatting).
    // Returns true and writes `entry_va` if found; otherwise writes 0 and returns false.
    bool nearest_entry(uint64_t guest_va, uint64_t& entry_va) const noexcept;

    size_t entry_count() const noexcept { return entries_.size(); }
    size_t shard_count() const noexcept { return shards_.size(); }

    void clear() noexcept;

private:
    std::vector<Pg6ShardDispatchFn> shards_;
    std::vector<Pg6ShardFunctionEntry> entries_;
};

using ResolvedShardTarget = ShardDispatchRegistry::ResolvedShardTarget;

}  // namespace playgta6::guest
