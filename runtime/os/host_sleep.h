#pragma once

// Precise host sleeps (R40). On Windows a plain Sleep / std::this_thread::sleep_until wakes on the
// system timer tick, 15.6 ms by default: a guest Sleep(1) poll took ~16 ms instead of ~1 ms, and a
// paced present that should wake at its vblank woke up to a tick late and missed it. The console's
// timers are fine-grained, so these waits use a high-resolution waitable timer (Windows 10 1803+),
// which ignores the timer tick; elsewhere (and if that timer cannot be created) the standard sleep.

#include <chrono>

namespace playgta6::os::host_sleep {

// Blocks until `deadline` (steady clock).
void sleep_until(std::chrono::steady_clock::time_point deadline);

inline void sleep_for(std::chrono::steady_clock::duration duration) {
    sleep_until(std::chrono::steady_clock::now() + duration);
}

}  // namespace playgta6::os::host_sleep
