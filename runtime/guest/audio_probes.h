#pragma once
// Music-path probes (backlog A4): counts calls to a few named guest addresses of the title's
// audio stack (decoder factory, music controller init, stream-decoder create/consume) and tallies
// the small integer values of the four register arguments, so the boot report shows which codec
// the factory was asked for (11 Opus / 5 XMA / 0..3 PCM) without knowing the argument order.
// Fed from the dispatch observer: a range check, a binary search and a few relaxed increments on a
// hit, nothing otherwise. Addresses are supplied by the caller; no game data lives in this file.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace playgta6::guest {

struct AudioProbe {
    std::string name;
    uint64_t address = 0;  // guest VA
};

class AudioProbes {
public:
    static constexpr size_t kArgs = 4;      // rcx, rdx, r8, r9
    static constexpr size_t kSmall = 16;    // values 0..15 are tallied per argument

    void configure(std::vector<AudioProbe> probes);
    void clear();
    bool active() const { return count_ != 0; }

    // Counts a call to `target`; returns the probe index or -1. `first` is set on its first call.
    long hit(uint64_t target, const uint64_t args[kArgs], bool* first = nullptr) noexcept;

    // Counts one execution of the function at `target` from the entry counter the lifter emits for
    // `--probe-functions` addresses (direct calls included). Same arguments tally as `hit`, kept in
    // separate counters so a call that arrives through dispatch is not counted twice.
    long entry_hit(uint64_t target, const uint64_t args[kArgs], bool* first = nullptr) noexcept;
    uint64_t entry_calls(size_t i) const { return entry_calls_[i].load(std::memory_order_relaxed); }
    uint64_t entry_small_value(size_t i, size_t arg, size_t v) const {
        return entry_small_[(i * kArgs + arg) * kSmall + v].load(std::memory_order_relaxed);
    }

    size_t size() const { return count_; }
    const std::string& name(size_t i) const { return probes_[i].name; }
    uint64_t address(size_t i) const { return probes_[i].address; }
    uint64_t calls(size_t i) const { return calls_[i].load(std::memory_order_relaxed); }
    // Calls whose argument `arg` (0..3) held the value `v` (0..15) in its low 32 bits, upper bits clear.
    uint64_t small_value(size_t i, size_t arg, size_t v) const {
        return small_[(i * kArgs + arg) * kSmall + v].load(std::memory_order_relaxed);
    }
    long find(const std::string& name) const;

    // `"audio_probes": {...}` body: per probe the address, calls and, per argument, the non-zero tallies.
    std::string to_json() const;

private:
    std::vector<AudioProbe> probes_;  // sorted by address
    std::unique_ptr<std::atomic<uint64_t>[]> calls_;
    std::unique_ptr<std::atomic<uint64_t>[]> small_;
    std::unique_ptr<std::atomic<uint64_t>[]> entry_calls_;
    std::unique_ptr<std::atomic<uint64_t>[]> entry_small_;
    size_t count_ = 0;
    uint64_t min_ = 0;
    uint64_t max_ = 0;
};

}  // namespace playgta6::guest
