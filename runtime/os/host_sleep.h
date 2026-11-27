#pragma once

// Precise host sleeps (R40). On Windows a plain Sleep / std::this_thread::sleep_until wakes on the
// system timer tick, 15.6 ms by default: a guest Sleep(1) poll took ~16 ms instead of ~1 ms, and a
// paced present that should wake at its vblank woke up to a tick late and missed it. The console's
// timers are fine-grained, so these waits use a high-resolution waitable timer (Windows 10 1803+),
// which ignores the timer tick; elsewhere (and if that timer cannot be created) the standard sleep.

#include <chrono>
#include <cstdint>
#include <string>

namespace playgta6::os::host_sleep {

// Blocks until `deadline` (steady clock).
void sleep_until(std::chrono::steady_clock::time_point deadline);

inline void sleep_for(std::chrono::steady_clock::duration duration) {
    sleep_until(std::chrono::steady_clock::now() + duration);
}

// Raises the process timer resolution (Windows: 1 ms instead of the 15.6 ms default tick; no-op
// elsewhere). Without it every timed guest wait (WaitForSingleObject / SleepConditionVariable with a
// few ms timeout, the title's own frame limiter) wakes on the next 15.6 ms tick, so a 16.7 ms frame
// limit becomes 16-31 ms frames and the title settles near 40 fps. Idempotent; the OS restores the
// resolution at process exit.
void raise_timer_resolution();

// --timer-resolution 1ms|default. `default` keeps the OS tick: raise_timer_resolution() does nothing
// and sleep_until uses the plain standard sleep (the A/B baseline). Set before the first use.
enum class Mode { OneMs, Default };
void set_mode(Mode mode);
Mode mode();

// Guest Sleep(ms) latency: requested vs actual wall time, as a histogram of the overshoot
// (actual - requested). Recorded by the guest Sleep thunk; thread-safe, lock-free.
void record_guest_sleep(std::chrono::steady_clock::duration requested, std::chrono::steady_clock::duration actual);
void reset_guest_sleep_stats();
// JSON object for the boot report: mode, count, mean/max requested and actual ms, overshoot buckets.
std::string guest_sleep_stats_json();

// Guest timed waits (WaitForSingleObject[Ex] / WaitForMultipleObjects[Ex] / SleepConditionVariable*) with a
// finite non-zero timeout, split by outcome. Timed-out waits get the overshoot histogram (actual - requested);
// signalled waits get count / mean requested / mean actual. Infinite and zero timeouts are not recorded.
void record_guest_timed_wait(uint32_t requested_ms, std::chrono::steady_clock::duration actual, bool timed_out);
void reset_guest_timed_wait_stats();
std::string guest_timed_wait_stats_json();

// A window in which every thread of the process is suspended on purpose (the minidump helper). Sleeps and
// timed waits overlapping it are counted separately ("frozen_by_dump") and kept out of the overshoot stats.
void frozen_window_begin();
void frozen_window_end();

// Construct before the wait, call done() with the outcome afterwards.
class GuestTimedWait {
public:
    explicit GuestTimedWait(uint32_t timeout_ms)
        : timeout_ms_(timeout_ms), active_(timeout_ms != 0 && timeout_ms != 0xFFFFFFFFu) {
        if (active_) t0_ = std::chrono::steady_clock::now();
    }
    void done(bool timed_out) const {
        if (active_) record_guest_timed_wait(timeout_ms_, std::chrono::steady_clock::now() - t0_, timed_out);
    }

private:
    uint32_t timeout_ms_;
    bool active_;
    std::chrono::steady_clock::time_point t0_{};
};

}  // namespace playgta6::os::host_sleep
