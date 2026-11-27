// Precise host sleeps (R40). Design: host_sleep.h.

#include "os/host_sleep.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "os/thread_name.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>
#ifdef _MSC_VER
#pragma comment(lib, "winmm.lib")
#endif
#endif

namespace playgta6::os::host_sleep {

namespace {

std::atomic<Mode> g_mode{Mode::OneMs};

constexpr int kBuckets = 8;  // overshoot < 0.5, 1, 2, 4, 8, 16, 32 ms, >= 32 ms
constexpr double kBucketMs[kBuckets - 1] = {0.5, 1, 2, 4, 8, 16, 32};
std::atomic<uint64_t> g_count{0}, g_req_us{0}, g_act_us{0}, g_max_act_us{0}, g_max_over_us{0};
std::atomic<uint64_t> g_hist[kBuckets];

struct TimedWaitAgg {
    std::atomic<uint64_t> count{0}, req_us{0}, act_us{0}, max_act_us{0}, max_over_us{0};
    std::atomic<uint64_t> hist[kBuckets];
    void reset() {
        count = 0;
        req_us = 0;
        act_us = 0;
        max_act_us = 0;
        max_over_us = 0;
        for (auto& h : hist) h = 0;
    }
};
TimedWaitAgg g_tw_timed_out, g_tw_signalled;

void atomic_max(std::atomic<uint64_t>& m, uint64_t v) {
    uint64_t cur = m.load(std::memory_order_relaxed);
    while (v > cur && !m.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
}

// Outlier capture: every guest Sleep / timed wait overshooting >= 32 ms is logged (the first 24: kind, thread,
// end time since process start). Waits overlapping the minidump helper's freeze are excluded (frozen_by_dump).
constexpr int64_t kOutlierOverUs = 32000;
constexpr size_t kMaxOutliers = 24;
const std::chrono::steady_clock::time_point g_t0 = std::chrono::steady_clock::now();
std::mutex g_out_mu;
std::vector<std::string> g_outliers;
std::atomic<uint64_t> g_outlier_total{0};
// Frozen window (steady_clock ns since g_t0): start of the open window (-1 = none), last closed [start, end].
std::atomic<int64_t> g_frozen_open{-1}, g_frozen_last_start{-1}, g_frozen_last_end{-1};
std::atomic<uint64_t> g_frozen_count{0};

int64_t ns_since_start(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t - g_t0).count();
}

// True when [end - actual, end] overlaps a frozen window (open or the last closed one).
bool overlaps_frozen(std::chrono::steady_clock::duration actual) {
    const int64_t end = ns_since_start(std::chrono::steady_clock::now());
    const int64_t start = end - std::chrono::duration_cast<std::chrono::nanoseconds>(actual).count();
    const int64_t open = g_frozen_open.load(std::memory_order_acquire);
    if (open >= 0 && end >= open) return true;
    const int64_t ls = g_frozen_last_start.load(std::memory_order_acquire);
    const int64_t le = g_frozen_last_end.load(std::memory_order_acquire);
    return ls >= 0 && start <= le && end >= ls;
}

double since_start_ms(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t - g_t0).count() / 1000.0;
}

void note_outlier(const char* kind, int64_t req_us, int64_t act_us, int64_t over_us, bool timed_out) {
    g_outlier_total.fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(g_out_mu);
    if (g_outliers.size() >= kMaxOutliers) return;
    char b[320];
    std::snprintf(b, sizeof(b),
                  "{\"kind\": \"%s\", \"thread\": %lu, \"end_ms\": %.1f, \"requested_ms\": %.1f, \"actual_ms\": %.1f, "
                  "\"overshoot_ms\": %.1f, \"timed_out\": %s}",
                  kind,
#ifdef _WIN32
                  static_cast<unsigned long>(GetCurrentThreadId()),
#else
                  0ul,
#endif
                  since_start_ms(now), req_us / 1000.0, act_us / 1000.0, over_us / 1000.0, timed_out ? "true" : "false");
    g_outliers.emplace_back(b);
}

std::string outliers_json() {
    std::lock_guard<std::mutex> lk(g_out_mu);
    auto join = [](const std::vector<std::string>& v) {
        std::string o = "[";
        for (size_t i = 0; i < v.size(); ++i) o += (i ? ", " : "") + v[i];
        return o + "]";
    };
    return "{\"frozen_by_dump\": " + std::to_string(g_frozen_count.load()) + ", \"total\": " + std::to_string(g_outlier_total.load()) + ", \"list\": " + join(g_outliers) +
           "}";
}

std::string agg_json(const TimedWaitAgg& a, bool with_hist) {
    const uint64_t n = a.count.load();
    char buf[512];
    const int len = std::snprintf(buf, sizeof(buf),
                                  "{\"count\": %llu, \"mean_requested_ms\": %.3f, \"mean_actual_ms\": %.3f, "
                                  "\"max_actual_ms\": %.3f",
                                  static_cast<unsigned long long>(n),
                                  n ? a.req_us.load() / 1000.0 / static_cast<double>(n) : 0.0,
                                  n ? a.act_us.load() / 1000.0 / static_cast<double>(n) : 0.0,
                                  a.max_act_us.load() / 1000.0);
    std::string o(buf, static_cast<size_t>(len));
    if (with_hist) {
        std::snprintf(buf, sizeof(buf),
                      ", \"max_overshoot_ms\": %.3f, \"overshoot_ms_buckets\": {\"lt0.5\": %llu, \"lt1\": %llu, "
                      "\"lt2\": %llu, \"lt4\": %llu, \"lt8\": %llu, \"lt16\": %llu, \"lt32\": %llu, \"ge32\": %llu}",
                      a.max_over_us.load() / 1000.0, static_cast<unsigned long long>(a.hist[0].load()),
                      static_cast<unsigned long long>(a.hist[1].load()), static_cast<unsigned long long>(a.hist[2].load()),
                      static_cast<unsigned long long>(a.hist[3].load()), static_cast<unsigned long long>(a.hist[4].load()),
                      static_cast<unsigned long long>(a.hist[5].load()), static_cast<unsigned long long>(a.hist[6].load()),
                      static_cast<unsigned long long>(a.hist[7].load()));
        o += buf;
    }
    return o + "}";
}

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

void set_mode(Mode m) { g_mode.store(m); }
Mode mode() { return g_mode.load(); }

void frozen_window_begin() { g_frozen_open.store(ns_since_start(std::chrono::steady_clock::now()), std::memory_order_release); }

void frozen_window_end() {
    const int64_t open = g_frozen_open.load(std::memory_order_acquire);
    if (open < 0) return;
    g_frozen_last_start.store(open, std::memory_order_release);
    g_frozen_last_end.store(ns_since_start(std::chrono::steady_clock::now()), std::memory_order_release);
    g_frozen_open.store(-1, std::memory_order_release);
}

void record_guest_sleep(std::chrono::steady_clock::duration requested, std::chrono::steady_clock::duration actual) {
    if (overlaps_frozen(actual)) {
        g_frozen_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    const int64_t req = duration_cast<microseconds>(requested).count();
    const int64_t act = duration_cast<microseconds>(actual).count();
    const int64_t over = act > req ? act - req : 0;
    int b = 0;
    while (b < kBuckets - 1 && over >= static_cast<int64_t>(kBucketMs[b] * 1000.0)) ++b;
    g_hist[b].fetch_add(1, std::memory_order_relaxed);
    if (over >= kOutlierOverUs) note_outlier("sleep", req, act, over, true);
    g_count.fetch_add(1, std::memory_order_relaxed);
    g_req_us.fetch_add(static_cast<uint64_t>(req), std::memory_order_relaxed);
    g_act_us.fetch_add(static_cast<uint64_t>(act), std::memory_order_relaxed);
    for (auto* m : {&g_max_act_us, &g_max_over_us}) {
        const uint64_t v = static_cast<uint64_t>(m == &g_max_act_us ? act : over);
        uint64_t cur = m->load(std::memory_order_relaxed);
        while (v > cur && !m->compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
        }
    }
}

void reset_guest_sleep_stats() {
    g_count = 0;
    g_req_us = 0;
    g_act_us = 0;
    g_max_act_us = 0;
    g_max_over_us = 0;
    for (auto& h : g_hist) h = 0;
}

std::string guest_sleep_stats_json() {
    const uint64_t n = g_count.load();
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "{\"timer_resolution\": \"%s\", \"count\": %llu, \"mean_requested_ms\": %.3f, "
                  "\"mean_actual_ms\": %.3f, \"max_actual_ms\": %.3f, \"max_overshoot_ms\": %.3f, "
                  "\"overshoot_ms_buckets\": {\"lt0.5\": %llu, \"lt1\": %llu, \"lt2\": %llu, \"lt4\": %llu, "
                  "\"lt8\": %llu, \"lt16\": %llu, \"lt32\": %llu, \"ge32\": %llu}}",
                  mode() == Mode::OneMs ? "1ms" : "default", static_cast<unsigned long long>(n),
                  n ? g_req_us.load() / 1000.0 / static_cast<double>(n) : 0.0,
                  n ? g_act_us.load() / 1000.0 / static_cast<double>(n) : 0.0, g_max_act_us.load() / 1000.0,
                  g_max_over_us.load() / 1000.0, static_cast<unsigned long long>(g_hist[0].load()),
                  static_cast<unsigned long long>(g_hist[1].load()), static_cast<unsigned long long>(g_hist[2].load()),
                  static_cast<unsigned long long>(g_hist[3].load()), static_cast<unsigned long long>(g_hist[4].load()),
                  static_cast<unsigned long long>(g_hist[5].load()), static_cast<unsigned long long>(g_hist[6].load()),
                  static_cast<unsigned long long>(g_hist[7].load()));
    return buf;
}

void record_guest_timed_wait(uint32_t requested_ms, std::chrono::steady_clock::duration actual, bool timed_out) {
    if (overlaps_frozen(actual)) {
        g_frozen_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const int64_t req = static_cast<int64_t>(requested_ms) * 1000;
    int64_t act = std::chrono::duration_cast<std::chrono::microseconds>(actual).count();
    if (act < 0) act = 0;
    const int64_t over = act > req ? act - req : 0;
    TimedWaitAgg& a = timed_out ? g_tw_timed_out : g_tw_signalled;
    int b = 0;
    while (b < kBuckets - 1 && over >= static_cast<int64_t>(kBucketMs[b] * 1000.0)) ++b;
    if (over >= kOutlierOverUs) note_outlier(timed_out ? "timed_wait" : "signalled_wait", req, act, over, timed_out);
    if (timed_out) a.hist[b].fetch_add(1, std::memory_order_relaxed);
    a.count.fetch_add(1, std::memory_order_relaxed);
    a.req_us.fetch_add(static_cast<uint64_t>(req), std::memory_order_relaxed);
    a.act_us.fetch_add(static_cast<uint64_t>(act), std::memory_order_relaxed);
    atomic_max(a.max_act_us, static_cast<uint64_t>(act));
    atomic_max(a.max_over_us, static_cast<uint64_t>(over));
}

void reset_guest_timed_wait_stats() {
    g_tw_timed_out.reset();
    g_tw_signalled.reset();
}

std::string guest_timed_wait_stats_json() {
    return "{\"timed_out\": " + agg_json(g_tw_timed_out, true) + ", \"signalled\": " + agg_json(g_tw_signalled, false) +
           ", \"outliers\": " + outliers_json() + "}";
}

void raise_timer_resolution() {
    if (mode() == Mode::Default) return;
#ifdef _WIN32
    static const bool done = [] {
        timeBeginPeriod(1);  // 1 ms tick; the OS restores it at process exit
        return true;
    }();
    (void)done;
    // Windows 11: do not let a windowless / occluded process have its timer resolution ignored.
    PROCESS_POWER_THROTTLING_STATE st{};
    st.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    st.ControlMask = PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
    st.StateMask = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st, sizeof(st));
#endif
}

void sleep_until(std::chrono::steady_clock::time_point deadline) {
#ifdef _WIN32
    if (mode() == Mode::OneMs)
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
