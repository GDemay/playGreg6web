#include "gpu/d3d12_readback.h"

#include <utility>
#include <vector>

#include "gpu/d3d12_guest_objects.h"
#include "os/guest_window.h"
#include "os/os_shim.h"

namespace playgta6::gpu::d3d12_guest {

ReadbackGate::ReadbackGate(bool hold, std::chrono::milliseconds timeout) : timeout_(timeout), hold_(hold) {}

ReadbackGate::~ReadbackGate() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        parked_.clear();  // the device is going away: held signals are dropped with it
    }
    cv_.notify_all();
    if (poller_.joinable()) {
        poller_.join();
    }
}

uint32_t ReadbackGate::issue(uint64_t guest_va, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++issued_;
    bytes_ += bytes;
    ranges_.push_back({guest_va, bytes});
    count_work(Work::ReadbacksIssued);
    count_work(Work::ReadbackBytes, bytes);
    return static_cast<uint32_t>(issued_);
}

uint64_t ReadbackGate::completed_locked() {
    const uint32_t v = done_.load(std::memory_order_acquire);
    if (v < done_last_) {
        ++done_wraps_;
    }
    done_last_ = v;
    const uint64_t reported = (done_wraps_ << 32) | v;
    if (reported > reported_seen_) {
        count_work(Work::ReadbacksCompleted, reported - reported_seen_);
        reported_seen_ = reported;
    }
    const uint64_t done = std::max(reported, forced_);
    // The consumer wrote these ranges: lifted code tracking window writes must see them (O37).
    while (seen_ < done && !ranges_.empty()) {
        os::guest_window::mark_written(ranges_.front().va, static_cast<size_t>(ranges_.front().bytes));
        ranges_.pop_front();
        ++seen_;
    }
    seen_ = std::max(seen_, done);
    return done;
}

bool ReadbackGate::park(std::function<void()> release) {
    poll();  // signals already due go first (keeps the order)
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hold_ || stop_) {
        return false;
    }
    if (parked_.empty() && completed_locked() >= issued_) {
        return false;
    }
    parked_.push_back({issued_, std::chrono::steady_clock::now(), std::move(release)});
    ++signals_held_;
    count_work(Work::ReadbackSignalsHeld);
    start_poller_locked();
    cv_.notify_all();
    return true;
}

size_t ReadbackGate::poll() {
    std::vector<std::function<void()>> ready;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t done = completed_locked();
        const auto now = std::chrono::steady_clock::now();
        while (!parked_.empty()) {
            Parked& p = parked_.front();
            if (done >= p.needed) {
                timeouts_in_a_row_ = 0;
            } else if (now - p.at >= timeout_) {
                // No answer: release it anyway rather than hang the title (counted).
                ++timeouts_;
                count_work(Work::ReadbackTimeouts);
                forced_ = std::max(forced_, p.needed);
                done = completed_locked();
                if (++timeouts_in_a_row_ >= kMaxTimeoutsInARow && hold_) {
                    hold_ = false;
                    os::log("d3d12_xs: GPU read-back consumer does not answer; queue signals are no longer held");
                }
            } else {
                break;
            }
            ready.push_back(std::move(p.release));
            parked_.pop_front();
        }
        if (!hold_) {
            // Given up: everything still parked goes now, in order.
            for (Parked& p : parked_) {
                forced_ = std::max(forced_, p.needed);
                ready.push_back(std::move(p.release));
            }
            parked_.clear();
        }
    }
    for (auto& r : ready) {
        r();
    }
    return ready.size();
}

void ReadbackGate::start_poller_locked() {
    if (poller_.joinable()) {
        return;
    }
    // Titles poll the fence's completed-value word directly, so releases cannot wait for an API call.
    poller_ = std::thread([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stop_) {
            if (parked_.empty()) {
                cv_.wait(lock);
                continue;
            }
            cv_.wait_for(lock, std::chrono::microseconds(500));
            if (stop_) {
                break;
            }
            lock.unlock();
            poll();
            lock.lock();
        }
    });
}

ReadbackGate::Stats ReadbackGate::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats s;
    s.issued = issued_;
    s.bytes = bytes_;
    s.completed = (done_wraps_ << 32) | done_.load(std::memory_order_acquire);
    s.timeouts = timeouts_;
    s.signals_held = signals_held_;
    s.signals_pending = parked_.size();
    s.holding = hold_;
    return s;
}

}  // namespace playgta6::gpu::d3d12_guest
