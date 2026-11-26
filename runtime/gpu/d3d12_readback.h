#pragma once

// GPU -> CPU read-back ordering (G66).
//
// Copies into CPU-visible memory (readback-heap buffers, texture regions copied into a placed
// footprint) become ReadbackBuffer / ReadbackTexture packets (playgta6_abi.h). The consumer runs
// them in stream order, writes the bytes into guest memory and then adds 1 to a u32 completion
// counter this gate owns. Fences are signalled on the CPU at submission (CommandQueue::Signal), so
// without a hold the title would see a fence complete before its read-back data arrived.
//
// The gate counts the read-backs it issued. A queue signal issued while some of them are still
// outstanding (or while earlier signals are held) is parked with the issue count at that moment and
// released, in order, once the completion counter reaches it. Release calls the signal itself
// (Fence::signal_from_queue), so the fence publish rules (DeviceOptions::fence_publish_lag) apply
// unchanged afterwards: this is an extra hold in front of them, not a change to them.
//
// Without a consumer that answers (native builds, a worker without guest memory), a hold would
// never end: a held signal is released after `timeout` (counted), and after `kMaxTimeoutsInARow`
// timeouts in a row the gate stops holding signals (read-backs are still issued and counted).

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <algorithm>

namespace playgta6::gpu::d3d12_guest {

class ReadbackGate {
public:
    static constexpr uint32_t kMaxTimeoutsInARow = 4;

    struct Stats {
        uint64_t issued = 0;
        uint64_t bytes = 0;
        uint64_t completed = 0;       // counter value seen (completions the consumer reported)
        uint64_t timeouts = 0;        // held signals released by the timeout
        uint64_t signals_held = 0;    // queue signals parked behind a read-back
        uint64_t signals_pending = 0; // parked now
        bool holding = false;         // signals are held (enabled and not given up)
    };

    // `hold`: park queue signals behind outstanding read-backs (a consumer answers).
    ReadbackGate(bool hold, std::chrono::milliseconds timeout);
    ~ReadbackGate();
    ReadbackGate(const ReadbackGate&) = delete;
    ReadbackGate& operator=(const ReadbackGate&) = delete;

    // Host address of the u32 completion counter (the packets' done_addr).
    uint64_t done_address() const { return reinterpret_cast<uintptr_t>(&done_); }

    // Issues one read-back that will write `bytes` at guest address `guest_va`; returns its 1-based
    // sequence number. The range is marked written (guest_window) when it completes.
    uint32_t issue(uint64_t guest_va, uint64_t bytes);

    // A queue signal. If read-backs are outstanding or signals are already held, `release` is
    // parked (run later by poll / the poller thread, in order) and true is returned; otherwise false
    // and the caller signals now.
    bool park(std::function<void()> release);

    // Releases the parked signals whose read-backs completed (or timed out); returns how many.
    size_t poll();

    Stats stats() const;

private:
    struct Parked {
        uint64_t needed;  // completion count that releases it
        std::chrono::steady_clock::time_point at;
        std::function<void()> release;
    };
    struct Range {
        uint64_t va;
        uint64_t bytes;
    };
    uint64_t completed_locked();
    void start_poller_locked();

    const std::chrono::milliseconds timeout_;
    alignas(8) std::atomic<uint32_t> done_{0};
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool hold_;
    bool stop_ = false;
    uint64_t issued_ = 0;
    uint64_t bytes_ = 0;
    uint64_t forced_ = 0;      // completion count assumed after a timeout
    uint64_t seen_ = 0;        // completions whose ranges were marked written
    uint64_t reported_seen_ = 0;  // consumer completions already counted
    uint64_t done_wraps_ = 0;  // 64-bit view of the u32 counter
    uint32_t done_last_ = 0;
    uint64_t timeouts_ = 0;
    uint32_t timeouts_in_a_row_ = 0;
    uint64_t signals_held_ = 0;
    std::deque<Range> ranges_;  // issued, not yet marked written
    std::deque<Parked> parked_;
    std::thread poller_;
};

}  // namespace playgta6::gpu::d3d12_guest
