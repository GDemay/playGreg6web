#include "gpu/d3d12_readback.h"
#include "os/thread_name.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "gpu/d3d12_guest_objects.h"
#include "os/guest_window.h"
#include "os/os_shim.h"

namespace playgta6::gpu::d3d12_guest {

namespace {

size_t clamp_ring_capacity(size_t requested) {
    if (requested == 0) {
        return ReadbackGate::kDefaultRingCapacity;
    }
    return std::clamp(requested, ReadbackGate::kMinRingCapacity, ReadbackGate::kMaxRingCapacity);
}

size_t kind_index(ReadbackKind kind) {
    const size_t idx = static_cast<size_t>(kind);
    return idx < ReadbackGate::kKindCount ? idx : 0u;
}

bool timeout_elapsed(std::chrono::steady_clock::time_point now,
                     std::chrono::steady_clock::time_point at,
                     std::chrono::milliseconds timeout) {
    if (now < at) {
        return timeout.count() <= 0;
    }
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - at);
    return elapsed_ms >= timeout;
}

}  // namespace

ReadbackGate::ReadbackGate(bool hold, std::chrono::milliseconds timeout, size_t ring_capacity)
    : hold_(hold),
      state_(std::make_shared<State>(
          hold,
          timeout.count() < 0 ? std::chrono::milliseconds(0) : timeout,
          clamp_ring_capacity(ring_capacity))) {}

ReadbackGate::~ReadbackGate() {
    // Move parked callbacks out before unlocking so any destructor invoked by dropping a parked
    // closure cannot deadlock if it re-enters stats() / poll() / issue(). If destruction occurs on
    // the poller thread itself (e.g. a parked callback dropped the last reference to the gate's
    // owner), detach rather than self-joining (which would call std::terminate); the poller holds
    // a shared_ptr<State> so it safely observes stop_ and exits without UAF.
    std::deque<Parked> dropped;
    std::thread poller_to_finish;
    {
        std::lock_guard<std::mutex> lock(state_->mutex_);
        state_->stop_ = true;
        dropped.swap(state_->parked_);
        if (poller_.joinable()) {
            poller_to_finish = std::move(poller_);
        }
    }
    state_->cv_.notify_all();
    if (poller_to_finish.joinable()) {
        if (poller_to_finish.get_id() == std::this_thread::get_id()) {
            poller_to_finish.detach();
        } else {
            poller_to_finish.join();
        }
    }
    dropped.clear();
}

ReadbackSlot ReadbackGate::issue_slot_impl(uint64_t guest_va, uint64_t bytes, ReadbackKind kind, uint32_t tag,
                                           std::function<void()> on_complete) {
    State& st = *state_;
    std::lock_guard<std::mutex> lock(st.mutex_);
    // Advance completion tracking first so already-completed ring slots are not miscounted as
    // overflows when reusing the ring.
    const uint64_t done = completed_locked(st);
    const uint64_t seq = ++st.issued_;
    st.bytes_ += bytes;
    const size_t kidx = kind_index(kind);
    ++st.issued_by_kind_[kidx];
    st.bytes_by_kind_[kidx] += bytes;

    const uint64_t in_flight = seq > done ? (seq - done) : 0u;
    if (in_flight > st.pending_high_water_) {
        st.pending_high_water_ = in_flight;
    }

    const uint32_t ridx = static_cast<uint32_t>(seq % st.ring_capacity_);
    ReadbackSlot& slot = st.ring_[ridx];
    if (slot.seq != 0 && !slot.completed && slot.seq > done) {
        ++st.ring_overflows_;
    }
    slot.seq = seq;
    slot.guest_va = guest_va;
    slot.bytes = bytes;
    slot.kind = kind;
    slot.tag = tag;
    slot.ring_index = ridx;
    slot.completed = (done >= seq);

    // In non-holding builds (no active readback consumer), cap ranges_ so long runs cannot leak
    // memory unboundedly while still retaining recent window entries for synthetic/poll callers.
    if (!st.hold_ && st.ranges_.size() >= st.ring_capacity_ * 2u) {
        st.ranges_.pop_front();
    }
    st.ranges_.push_back(PendingRange{seq, guest_va, bytes, kind, tag, std::move(on_complete)});
    count_work(Work::ReadbacksIssued);
    if (bytes != 0) {
        count_work(Work::ReadbackBytes, bytes);
    }
    // If consumer already ran ahead of seq (rare synthetic case), reconcile immediately.
    if (done >= seq) {
        completed_locked(st);
    }
    return slot;
}

uint32_t ReadbackGate::issue(uint64_t guest_va, uint64_t bytes, ReadbackKind kind, uint32_t tag) {
    return static_cast<uint32_t>(issue_slot_impl(guest_va, bytes, kind, tag).seq);
}

uint32_t ReadbackGate::issue_with_completion(uint64_t guest_va, uint64_t bytes, ReadbackKind kind, uint32_t tag,
                                             std::function<void()> on_complete) {
    return static_cast<uint32_t>(issue_slot_impl(guest_va, bytes, kind, tag, std::move(on_complete)).seq);
}

uint64_t ReadbackGate::issue64(uint64_t guest_va, uint64_t bytes, ReadbackKind kind, uint32_t tag) {
    return issue_slot_impl(guest_va, bytes, kind, tag).seq;
}

ReadbackSlot ReadbackGate::issue_occlusion_query(uint64_t guest_va, uint64_t bytes, uint32_t query_index) {
    return issue_slot_impl(guest_va, bytes, ReadbackKind::OcclusionQuery, query_index);
}

ReadbackSlot ReadbackGate::issue_timestamp_query(uint64_t guest_va, uint64_t bytes, uint32_t pass_id) {
    return issue_slot_impl(guest_va, bytes, ReadbackKind::TimestampQuery, pass_id);
}

ReadbackSlot ReadbackGate::issue_compute_readback(uint64_t guest_va, uint64_t bytes, uint32_t tag) {
    return issue_slot_impl(guest_va, bytes, ReadbackKind::ComputeReadback, tag);
}

uint64_t ReadbackGate::peek_completed_locked(const State& st) {
    const uint32_t v = st.done_.load(std::memory_order_acquire);
    const uint64_t wraps = st.done_wraps_ + (v < st.done_last_ ? 1ull : 0ull);
    const uint64_t raw = (wraps << 32) | static_cast<uint64_t>(v);
    return std::max(raw, st.reported_seen_);
}

uint64_t ReadbackGate::completed_locked(State& st) {
    const uint32_t v = st.done_.load(std::memory_order_acquire);
    if (v < st.done_last_) {
        ++st.done_wraps_;
    }
    st.done_last_ = v;
    const uint64_t reported = std::max((st.done_wraps_ << 32) | static_cast<uint64_t>(v), st.reported_seen_);
    if (reported > st.reported_seen_) {
        count_work(Work::ReadbacksCompleted, reported - st.reported_seen_);
        st.reported_seen_ = reported;
    }
    const uint64_t done = reported;
    // The consumer wrote these ranges: lifted code tracking window writes must see them (O37).
    while (!st.ranges_.empty() && st.ranges_.front().seq <= done) {
        PendingRange r = std::move(st.ranges_.front());
        st.ranges_.pop_front();
        if (r.on_complete) {
            r.on_complete();
        }
        if (r.va != 0 && r.bytes != 0) {
            const size_t clamped = static_cast<size_t>(
                std::min<uint64_t>(r.bytes, static_cast<uint64_t>(std::numeric_limits<size_t>::max())));
            os::guest_window::mark_written(r.va, clamped);
        }
        ++st.completed_by_kind_[kind_index(r.kind)];
        const uint32_t ridx = static_cast<uint32_t>(r.seq % st.ring_capacity_);
        ReadbackSlot& slot = st.ring_[ridx];
        if (slot.seq == r.seq) {
            slot.completed = true;
        }
        ReadbackSlot finished{};
        finished.seq = r.seq;
        finished.guest_va = r.va;
        finished.bytes = r.bytes;
        finished.kind = r.kind;
        finished.tag = r.tag;
        finished.ring_index = ridx;
        finished.completed = true;
        if (st.completed_ring_.size() >= st.ring_capacity_) {
            st.completed_ring_.pop_front();
        }
        st.completed_ring_.push_back(finished);
        st.seen_ = std::max(st.seen_, r.seq);
    }
    st.seen_ = std::max(st.seen_, done);
    return done;
}

bool ReadbackGate::is_complete(uint64_t seq) const {
    if (seq == 0) {
        return true;
    }
    const State& st = *state_;
    std::lock_guard<std::mutex> lock(st.mutex_);
    return seq <= st.issued_ && peek_completed_locked(st) >= seq;
}

bool ReadbackGate::try_poll_slot(uint64_t seq, ReadbackSlot* out) {
    if (seq == 0) {
        if (out != nullptr) {
            *out = ReadbackSlot{};
            out->completed = true;
        }
        return true;
    }
    State& st = *state_;
    std::lock_guard<std::mutex> lock(st.mutex_);
    const uint64_t done = completed_locked(st);
    const bool ready = (seq <= st.issued_) && (done >= seq);
    if (out != nullptr) {
        const uint32_t ridx = static_cast<uint32_t>(seq % st.ring_capacity_);
        const ReadbackSlot& slot = st.ring_[ridx];
        if (slot.seq == seq) {
            *out = slot;
            out->completed = ready;
        } else {
            ReadbackSlot summary{};
            summary.seq = seq;
            summary.ring_index = ridx;
            summary.completed = ready;
            *out = summary;
        }
    }
    return ready;
}

size_t ReadbackGate::drain_completed_slots(std::vector<ReadbackSlot>* out, size_t max_slots) {
    State& st = *state_;
    std::lock_guard<std::mutex> lock(st.mutex_);
    completed_locked(st);
    const size_t available = st.completed_ring_.size();
    const size_t take = (max_slots == 0 || max_slots > available) ? available : max_slots;
    if (out != nullptr) {
        out->reserve(out->size() + take);
        for (size_t i = 0; i < take; ++i) {
            out->push_back(st.completed_ring_.front());
            st.completed_ring_.pop_front();
        }
    } else {
        for (size_t i = 0; i < take; ++i) {
            st.completed_ring_.pop_front();
        }
    }
    return take;
}

uint64_t ReadbackGate::pending_count() const {
    const State& st = *state_;
    std::lock_guard<std::mutex> lock(st.mutex_);
    const uint64_t done = peek_completed_locked(st);
    return st.issued_ > done ? (st.issued_ - done) : 0u;
}

uint64_t ReadbackGate::completed_count() const {
    const State& st = *state_;
    std::lock_guard<std::mutex> lock(st.mutex_);
    return peek_completed_locked(st);
}

void ReadbackGate::notify() {
    State& st = *state_;
    {
        std::lock_guard<std::mutex> lock(st.mutex_);
        ++st.wake_gen_;
    }
    st.cv_.notify_one();
}

bool ReadbackGate::park(std::function<void()> release) {
    if (!hold_) {
        return false;
    }
    poll();  // signals already due go first (keeps the order)
    State& st = *state_;
    std::lock_guard<std::mutex> lock(st.mutex_);
    if (st.stop_) {
        return false;
    }
    // Preserve strict global FIFO signal order: if another thread (or an outer frame) is still
    // invoking earlier released callbacks, park behind it so the active drainer runs this signal
    // strictly after the earlier callbacks complete.
    if (!st.releasing_ && st.parked_.empty() && completed_locked(st) >= st.issued_) {
        return false;
    }
    st.parked_.push_back({st.issued_, std::chrono::steady_clock::now(), std::move(release)});
    ++st.signals_held_;
    ++st.wake_gen_;
    count_work(Work::ReadbackSignalsHeld);
    start_poller_locked();
    st.cv_.notify_all();
    return true;
}

size_t ReadbackGate::poll_state(State& st) {
    std::vector<std::function<void()>> ready;
    size_t total_released = 0;
    std::unique_lock<std::mutex> lock(st.mutex_);
    if (st.releasing_) {
        // Another thread is actively draining parked callbacks (or this is a re-entrant call from
        // inside a callback); advance completion tracking only so FIFO order and deadlock freedom
        // are both preserved.
        completed_locked(st);
        return 0;
    }
    st.releasing_ = true;
    st.releasing_thread_ = std::this_thread::get_id();
    for (;;) {
        ready.clear();
        bool log_timeout = false;
        const uint64_t done = completed_locked(st);
        const auto now = std::chrono::steady_clock::now();
        while (!st.parked_.empty()) {
            Parked& p = st.parked_.front();
            if (done < p.needed) {
                // Never publish a completion before the bytes arrived: the signal stays held. The
                // timeout is a diagnostic only (once per held signal, logged loudly outside lock).
                if (!p.warned && timeout_elapsed(now, p.at, st.timeout_)) {
                    p.warned = true;
                    ++st.timeouts_;
                    count_work(Work::ReadbackTimeouts);
                    log_timeout = true;
                }
                break;
            }
            ready.push_back(std::move(p.release));
            st.parked_.pop_front();
        }
        if (ready.empty() && !log_timeout) {
            st.releasing_ = false;
            st.releasing_thread_ = {};
            break;
        }
        const int64_t timeout_ms = st.timeout_.count();
        lock.unlock();
        if (log_timeout) {
            os::log("d3d12_xs: WARNING GPU read-back not completed after " + std::to_string(timeout_ms) +
                    " ms; the guarding fence stays unsignalled (consumer stalled or present-bound?)");
        }
        for (auto& r : ready) {
            if (r) {
                r();
            }
        }
        total_released += ready.size();
        lock.lock();
        if (st.stop_) {
            st.releasing_ = false;
            st.releasing_thread_ = {};
            break;
        }
        // If no callbacks ran (only a timeout warning was logged), avoid re-checking immediately.
        if (ready.empty()) {
            st.releasing_ = false;
            st.releasing_thread_ = {};
            break;
        }
    }
    return total_released;
}

size_t ReadbackGate::poll() {
    return poll_state(*state_);
}

void ReadbackGate::start_poller_locked() {
    if (poller_.joinable()) {
        return;
    }
    // Titles poll the fence's completed-value word directly, so releases cannot wait for an API call.
    // Use adaptive backoff so active completions wake quickly (~250 us) while idle or stalled
    // readbacks never busy-spin or waste CPU.
    std::shared_ptr<State> st = state_;
    poller_ = std::thread([st = std::move(st)] {
        playgta6::os::name_this_thread(L"pg6-readback-poller");
        constexpr int64_t kMinPollUs = 250;
        constexpr int64_t kMaxPollUs = 8000;
        constexpr int64_t kMaxTimeoutMsForUs = std::numeric_limits<int64_t>::max() / 1000;
        int64_t poll_us = kMinPollUs;
        uint64_t last_gen = 0;
        uint64_t last_done = 0;
        std::unique_lock<std::mutex> lock(st->mutex_);
        while (!st->stop_) {
            if (st->parked_.empty()) {
                poll_us = kMinPollUs;
                st->cv_.wait(lock, [&st] { return st->stop_ || !st->parked_.empty(); });
                if (st->stop_) {
                    break;
                }
            }
            // If notify() or park() bumped wake_gen_, poll immediately before sleeping so consumer
            // hints wake with minimum latency instead of eating a forced 250 us delay.
            if (st->wake_gen_ == last_gen) {
                auto wait_dur = std::chrono::microseconds(poll_us);
                const Parked& front = st->parked_.front();
                if (!front.warned && st->timeout_.count() <= kMaxTimeoutMsForUs) {
                    const auto now = std::chrono::steady_clock::now();
                    const auto elapsed_us =
                        now >= front.at ? std::chrono::duration_cast<std::chrono::microseconds>(now - front.at)
                                        : std::chrono::microseconds(0);
                    const auto timeout_us = std::chrono::microseconds(st->timeout_.count() * 1000);
                    if (elapsed_us >= timeout_us) {
                        wait_dur = std::chrono::microseconds(0);
                    } else {
                        const auto rem = std::max(timeout_us - elapsed_us, std::chrono::microseconds(1));
                        if (rem < wait_dur) {
                            wait_dur = rem;
                        }
                    }
                }
                if (wait_dur.count() > 0) {
                    st->cv_.wait_for(lock, wait_dur, [&st, last_gen] { return st->stop_ || st->wake_gen_ != last_gen; });
                }
                if (st->stop_) {
                    break;
                }
            } else {
                poll_us = kMinPollUs;
            }
            last_gen = st->wake_gen_;
            const uint64_t prev_done = last_done;
            lock.unlock();
            const size_t released = poll_state(*st);
            lock.lock();
            const uint64_t cur_done = st->reported_seen_;
            last_done = cur_done;
            if (released > 0 || cur_done > prev_done || st->wake_gen_ != last_gen) {
                poll_us = kMinPollUs;
            } else {
                poll_us = std::min<int64_t>(poll_us * 2, kMaxPollUs);
            }
        }
    });
}

ReadbackGate::Stats ReadbackGate::stats() const {
    const State& st = *state_;
    std::lock_guard<std::mutex> lock(st.mutex_);
    const uint64_t done = peek_completed_locked(st);
    Stats s;
    s.issued = st.issued_;
    s.bytes = st.bytes_;
    s.completed = done;
    s.timeouts = st.timeouts_;
    s.signals_held = st.signals_held_;
    s.signals_pending = st.parked_.size();
    s.holding = st.hold_;
    s.pending = st.issued_ > done ? (st.issued_ - done) : 0u;
    s.pending_high_water = st.pending_high_water_;
    s.ring_capacity = st.ring_capacity_;
    s.ring_overflows = st.ring_overflows_;
    s.issued_by_kind = st.issued_by_kind_;
    s.completed_by_kind = st.completed_by_kind_;
    s.bytes_by_kind = st.bytes_by_kind_;
    return s;
}

bool ReadbackGate::seed_sequence_for_testing(uint64_t base_seq) {
    State& st = *state_;
    std::lock_guard<std::mutex> lock(st.mutex_);
    if (!st.parked_.empty() || !st.ranges_.empty()) {
        return false;
    }
    const uint32_t low = static_cast<uint32_t>(base_seq & 0xFFFFFFFFull);
    st.done_.store(low, std::memory_order_release);
    st.done_last_ = low;
    st.done_wraps_ = base_seq >> 32;
    st.issued_ = base_seq;
    st.seen_ = base_seq;
    st.reported_seen_ = base_seq;
    st.completed_ring_.clear();
    for (ReadbackSlot& slot : st.ring_) {
        slot = ReadbackSlot{};
    }
    return true;
}

}  // namespace playgta6::gpu::d3d12_guest
