#pragma once
// Debug watch of guest address ranges (`playgta6_boot --watch-guest-va <addr>[:len]`): which
// runtime-side operations touch a range (commit, decommit, release, MEM_RESET, checked writes
// through VirtualMemoryArena::write), and what the range holds at each entry of a function
// listed in the build's `--probe-functions` file, at each present that finds it changed, and at a
// guest fault. Each event is one
// "[watch] t=<ms> thread=<guest thread> ..." log line, so the order of events across threads can
// be read off the log. Guest code's own stores are not seen (lifted code writes memory directly):
// the probe snapshots bracket them. Addresses come from the command line; no game data here.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace playgta6::guest {

struct WatchRange {
    uint64_t va = 0;   // first byte; for an indirect range the qword-sized pointer's address
    uint64_t len = 0;
    bool indirect = false;  // "*ptr+off": the range starts at (*va + offset), read at each snapshot
    uint64_t offset = 0;
};

// Parses "<addr>[:len]" or "*<ptr>[+<off>][:len]" (a heap object reached through a global
// pointer): addr/ptr/off hexadecimal (optional 0x), len decimal or 0x-hex, default 8, at most
// kMaxWatchLen. False on malformed text, a zero address or length, or an overflowing range.
// Indirect ranges are only snapshotted (runtime-side operations are matched against direct ones).
bool parse_watch_spec(std::string_view text, WatchRange* out);

class GuestWatch {
public:
    static constexpr uint64_t kMaxWatchLen = 4096;
    static constexpr size_t kMaxRanges = 16;
    static constexpr size_t kDumpBytes = 64;          // bytes shown per range in a snapshot
    static constexpr uint64_t kDefaultLineBudget = 20000;

    void configure(std::vector<WatchRange> ranges, uint64_t line_budget = kDefaultLineBudget);
    void clear() { configure({}); }
    bool active() const { return !ranges_.empty(); }
    const std::vector<WatchRange>& ranges() const { return ranges_; }

    // True if [va, va + len) intersects a watched range (len 0 counts as 1 byte).
    bool overlaps(uint64_t va, uint64_t len) const;

    // " [0x...:n]=<hex bytes>" per range (at most kDumpBytes each, qword-grouped, low address
    // first), or "=unreadable" where `read` fails.
    std::string snapshot(const std::function<bool(uint64_t va, void* dst, size_t len)>& read) const;

    // False once the run has logged `line_budget` watch lines.
    bool take_line();
    // Entry of a probed function: true for its first kEntriesPerFunction entries (a hot function
    // listed for its first calls does not flood the log), then false. Lock-free; once
    // kMaxFunctions distinct functions were seen, others are not logged.
    static constexpr uint64_t kEntriesPerFunction = 16;
    static constexpr size_t kMaxFunctions = 256;
    bool take_entry(uint64_t va);
    uint64_t lines() const { return lines_.load(std::memory_order_relaxed); }

private:
    std::vector<WatchRange> ranges_;  // sorted by va
    uint64_t budget_ = kDefaultLineBudget;
    std::atomic<uint64_t> lines_{0};
    std::atomic<uint64_t> entry_va_[kMaxFunctions] = {};
    std::atomic<uint64_t> entry_count_[kMaxFunctions] = {};
};

}  // namespace playgta6::guest
