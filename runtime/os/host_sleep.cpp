// Precise host sleeps (R40). Design: host_sleep.h.

#include "os/host_sleep.h"

#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace playgta6::os::host_sleep {

namespace {

#ifdef _WIN32
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

// One timer per thread (a waitable timer is waited on by the thread that set it); null when the
// OS has no high-resolution timers.
HANDLE thread_timer() {
    struct Timer {
        HANDLE handle = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                               TIMER_ALL_ACCESS);
        ~Timer() {
            if (handle != nullptr) {
                CloseHandle(handle);
            }
        }
    };
    thread_local Timer timer;
    return timer.handle;
}
#endif

}  // namespace

void sleep_until(std::chrono::steady_clock::time_point deadline) {
#ifdef _WIN32
    if (HANDLE timer = thread_timer()) {
        for (;;) {
            const auto left = deadline - std::chrono::steady_clock::now();
            if (left <= std::chrono::steady_clock::duration::zero()) {
                return;
            }
            LARGE_INTEGER due;
            // Relative due time in 100 ns units (negative), rounded up.
            due.QuadPart = -static_cast<LONGLONG>(
                (std::chrono::duration_cast<std::chrono::nanoseconds>(left).count() + 99) / 100);
            if (!SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE) ||
                WaitForSingleObject(timer, INFINITE) != WAIT_OBJECT_0) {
                break;  // fall back to the standard sleep
            }
        }
    }
#endif
    std::this_thread::sleep_until(deadline);
}

}  // namespace playgta6::os::host_sleep
