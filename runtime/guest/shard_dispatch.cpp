#include "guest/shard_dispatch.h"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace playgta6::guest {

namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

std::string format_hex(uint64_t value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(value));
    return std::string(buf);
}

}  // namespace

bool ShardDispatchRegistry::add_table(const Pg6ShardDispatchTable& table, std::string* error) {
    if (table.version != kPg6ShardDispatchVersion) {
        set_error(error, "unsupported shard dispatch table version " +
                             std::to_string(table.version) + " (expected " +
                             std::to_string(kPg6ShardDispatchVersion) + ")");
        return false;
    }

    if (table.shard_count > 0 && table.shards == nullptr) {
        set_error(error, "shard dispatch table has non-zero shard_count with null shards pointer");
        return false;
    }

    if (table.entry_count > 0 && table.entries == nullptr) {
        set_error(error, "shard dispatch table has non-zero entry_count with null entries pointer");
        return false;
    }

    if (table.entry_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        set_error(error, "shard dispatch table entry_count exceeds addressable size");
        return false;
    }

    const size_t base_shard_offset = shards_.size();
    const uint64_t total_shards =
        static_cast<uint64_t>(base_shard_offset) + static_cast<uint64_t>(table.shard_count);
    if (total_shards > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        set_error(error, "combined shard count exceeds uint32_t capacity");
        return false;
    }

    for (uint32_t i = 0; i < table.shard_count; ++i) {
        if (table.shards[i] == nullptr) {
            set_error(error, "null shard dispatcher at shard_idx " + std::to_string(i));
            return false;
        }
    }

    const size_t incoming_count = static_cast<size_t>(table.entry_count);
    std::vector<Pg6ShardFunctionEntry> incoming;
    incoming.reserve(incoming_count);

    for (size_t i = 0; i < incoming_count; ++i) {
        const Pg6ShardFunctionEntry& src = table.entries[i];
        if (src.shard_idx >= table.shard_count) {
            set_error(error, "entry " + std::to_string(i) + " (guest_va " +
                                 format_hex(src.guest_va) + ") has out-of-range shard_idx " +
                                 std::to_string(src.shard_idx) + " (shard_count " +
                                 std::to_string(table.shard_count) + ")");
            return false;
        }
        if (table.shards[src.shard_idx] == nullptr) {
            set_error(error, "entry " + std::to_string(i) + " (guest_va " +
                                 format_hex(src.guest_va) +
                                 ") references null shard dispatcher at shard_idx " +
                                 std::to_string(src.shard_idx));
            return false;
        }
        Pg6ShardFunctionEntry remapped{};
        remapped.guest_va = src.guest_va;
        remapped.shard_idx = static_cast<uint32_t>(base_shard_offset + src.shard_idx);
        remapped.local_idx = src.local_idx;
        incoming.push_back(remapped);
    }

    std::sort(incoming.begin(), incoming.end(),
              [](const Pg6ShardFunctionEntry& a, const Pg6ShardFunctionEntry& b) {
                  return a.guest_va < b.guest_va;
              });

    for (size_t i = 1; i < incoming.size(); ++i) {
        if (incoming[i].guest_va == incoming[i - 1].guest_va) {
            set_error(error, "duplicate guest_va " + format_hex(incoming[i].guest_va) +
                                 " within shard dispatch table");
            return false;
        }
    }

    std::vector<Pg6ShardFunctionEntry> merged;
    merged.reserve(entries_.size() + incoming.size());

    size_t existing_idx = 0;
    size_t incoming_idx = 0;
    while (existing_idx < entries_.size() && incoming_idx < incoming.size()) {
        const Pg6ShardFunctionEntry& existing_entry = entries_[existing_idx];
        const Pg6ShardFunctionEntry& incoming_entry = incoming[incoming_idx];
        if (existing_entry.guest_va < incoming_entry.guest_va) {
            merged.push_back(existing_entry);
            ++existing_idx;
        } else if (incoming_entry.guest_va < existing_entry.guest_va) {
            merged.push_back(incoming_entry);
            ++incoming_idx;
        } else {
            set_error(error, "duplicate guest_va " + format_hex(incoming_entry.guest_va) +
                                 " already registered");
            return false;
        }
    }
    while (existing_idx < entries_.size()) {
        merged.push_back(entries_[existing_idx++]);
    }
    while (incoming_idx < incoming.size()) {
        merged.push_back(incoming[incoming_idx++]);
    }

    std::vector<Pg6ShardDispatchFn> new_shards;
    new_shards.reserve(base_shard_offset + table.shard_count);
    new_shards.insert(new_shards.end(), shards_.begin(), shards_.end());
    if (table.shard_count > 0) {
        new_shards.insert(new_shards.end(), table.shards, table.shards + table.shard_count);
    }

    shards_.swap(new_shards);
    entries_.swap(merged);

    if (error != nullptr) {
        error->clear();
    }
    return true;
}

bool ShardDispatchRegistry::lookup(uint64_t guest_va, ResolvedShardTarget& out) const noexcept {
    const auto it = std::lower_bound(
        entries_.begin(), entries_.end(), guest_va,
        [](const Pg6ShardFunctionEntry& entry, uint64_t va) { return entry.guest_va < va; });
    if (it == entries_.end() || it->guest_va != guest_va) {
        out = ResolvedShardTarget{};
        return false;
    }
    out.dispatcher = shards_[it->shard_idx];
    out.local_idx = it->local_idx;
    return true;
}

bool ShardDispatchRegistry::dispatch(uint64_t guest_va, Pg6GuestState* state) const {
    ResolvedShardTarget target{};
    if (!lookup(guest_va, target) || target.dispatcher == nullptr) {
        return false;
    }
    target.dispatcher(target.local_idx, state);
    return true;
}

void ShardDispatchRegistry::clear() noexcept {
    entries_.clear();
    shards_.clear();
}

}  // namespace playgta6::guest
