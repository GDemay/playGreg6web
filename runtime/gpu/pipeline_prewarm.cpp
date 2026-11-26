#include "gpu/pipeline_prewarm.h"

#include <algorithm>
#include <cstring>
#include <unordered_set>

namespace playgta6::gpu {

namespace {
void put32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
void put64(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
uint32_t get32(const uint8_t* p) {
    return uint32_t{p[0]} | uint32_t{p[1]} << 8 | uint32_t{p[2]} << 16 | uint32_t{p[3]} << 24;
}
uint64_t get64(const uint8_t* p) { return uint64_t{get32(p)} | uint64_t{get32(p + 4)} << 32; }
// FNV-1a over the detail bytes, never 0 (0 marks a module-only entry).
uint64_t hash_detail(const std::vector<uint8_t>& d) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint8_t b : d) {
        h ^= b;
        h *= 0x100000001b3ull;
    }
    return h == 0 ? 1 : h;
}

struct ShaderTriple {
    uint32_t kind;
    uint64_t k0;
    uint64_t k1;
    bool operator==(const ShaderTriple& o) const noexcept {
        return kind == o.kind && k0 == o.k0 && k1 == o.k1;
    }
};
struct ShaderTripleHash {
    size_t operator()(const ShaderTriple& t) const noexcept {
        uint64_t h = t.k0 * 0x9E3779B97F4A7C15ull;
        h ^= (t.k1 + 0x7F4A7C15ull + (h << 6) + (h >> 2));
        return static_cast<size_t>(h ^ t.kind);
    }
};

size_t latency_bucket_for_ns(uint64_t ns) noexcept {
    if (ns < 1'000'000ull) return 0;
    if (ns < 10'000'000ull) return 1;
    if (ns < 50'000'000ull) return 2;
    if (ns < 200'000'000ull) return 3;
    return 4;
}
}  // namespace

void PipelinePrewarm::adjust_state_counters(const Slot& s, State state, int32_t delta) noexcept {
    const bool full = s.id.k2 != 0;
    switch (state) {
        case State::Queued:
            queued_count_ = static_cast<uint32_t>(static_cast<int32_t>(queued_count_) + delta);
            break;
        case State::InFlight:
            in_flight_ = static_cast<uint32_t>(static_cast<int32_t>(in_flight_) + delta);
            break;
        case State::Done:
            done_count_ = static_cast<uint32_t>(static_cast<int32_t>(done_count_) + delta);
            if (full) {
                full_done_ = static_cast<uint32_t>(static_cast<int32_t>(full_done_) + delta);
            } else {
                mod_done_ = static_cast<uint32_t>(static_cast<int32_t>(mod_done_) + delta);
            }
            break;
        case State::Used:
            done_count_ = static_cast<uint32_t>(static_cast<int32_t>(done_count_) + delta);
            if (full) {
                full_used_ = static_cast<uint32_t>(static_cast<int32_t>(full_used_) + delta);
            } else {
                mod_used_ = static_cast<uint32_t>(static_cast<int32_t>(mod_used_) + delta);
            }
            break;
        case State::Failed:
            failed_count_ = static_cast<uint32_t>(static_cast<int32_t>(failed_count_) + delta);
            break;
        case State::Skipped:
            skipped_count_ = static_cast<uint32_t>(static_cast<int32_t>(skipped_count_) + delta);
            break;
    }
}

void PipelinePrewarm::set_slot_state(Slot& s, State next) noexcept {
    if (s.state == next) return;
    adjust_state_counters(s, s.state, -1);
    s.state = next;
    adjust_state_counters(s, next, +1);
}

void PipelinePrewarm::set_pack_index(std::shared_ptr<const ShaderPackIndex> index) {
    std::lock_guard<std::mutex> lock(mu_);
    pack_ = std::move(index);
}

void PipelinePrewarm::set_max_in_flight(uint32_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    max_in_flight_ = std::max<uint32_t>(n, 1);
}

bool PipelinePrewarm::resolvable(const Identity& i) const {
    if (!pack_) return true;
    if (pack_->find(i.k0) == pack_->end()) return false;
    return i.k1 == 0 || pack_->find(i.k1) != pack_->end();
}

bool PipelinePrewarm::load_warm_list(const uint8_t* data, size_t size, std::string* error, uint32_t* queued) {
    auto fail = [&](const char* msg) {
        if (error) *error = msg;
        return false;
    };
    if (queued) *queued = 0;
    if (data == nullptr || size < kPrewarmListHeaderBytes) return fail("warm list too short");
    if (std::memcmp(data, "PG6W", 4) != 0) return fail("warm list bad magic");
    const uint32_t version = get32(data + 4);
    if (version != kPrewarmListVersionModules && version != kPrewarmListVersionPipelines) {
        return fail("warm list unsupported version");
    }
    const uint32_t count = get32(data + 8);
    if (count > kPrewarmMaxEntries) return fail("warm list too many entries");
    if (size < kPrewarmListHeaderBytes + uint64_t{count} * kPrewarmListEntryBytes) return fail("warm list truncated");
    struct Parsed {
        Identity id;
        PrewarmDetail detail;
    };
    std::vector<Parsed> parsed;
    parsed.reserve(count);
    size_t pos = kPrewarmListHeaderBytes;
    for (uint32_t i = 0; i < count; ++i) {
        if (size - pos < kPrewarmListEntryBytes) return fail("warm list truncated");
        const uint8_t* e = data + pos;
        pos += kPrewarmListEntryBytes;
        Parsed p{Identity{e[0], get64(e + 8), get64(e + 16), 0}, nullptr};
        if (p.id.kind > kPrewarmKindCompute || p.id.k0 == 0) return fail("warm list bad entry");
        if (version == kPrewarmListVersionPipelines) {
            const uint32_t detail_len = get32(e + 4);
            if (detail_len > kPrewarmMaxDetailBytes) return fail("warm list bad entry");
            if (size - pos < detail_len) return fail("warm list truncated");
            if (detail_len != 0) {
                auto d = std::make_shared<std::vector<uint8_t>>(data + pos, data + pos + detail_len);
                p.id.k2 = hash_detail(*d);
                p.detail = std::move(d);
            }
            pos += detail_len;
        }
        parsed.push_back(std::move(p));
    }
    std::lock_guard<std::mutex> lock(mu_);
    slots_.reserve(slots_.size() + parsed.size());
    by_identity_.reserve(by_identity_.size() + parsed.size());
    uint32_t added = 0;
    for (Parsed& p : parsed) {
        const Identity id = p.id;
        if (by_identity_.count(id) != 0) continue;
        const bool seen = observed_set_.count(id) != 0;
        const bool can_resolve = seen || resolvable(id);
        const State initial_state = (seen || !can_resolve) ? State::Skipped : State::Queued;
        slots_.push_back({id, initial_state, std::move(p.detail), false});
        adjust_state_counters(slots_.back(), initial_state, +1);
        const uint32_t task = static_cast<uint32_t>(slots_.size());
        by_identity_.emplace(id, task);
        if (!can_resolve) {
            ++pack_skipped_;
            continue;
        }
        if (seen) continue;
        queue_.push_back(task);
        ++added;
    }
    if (queued) *queued = added;
    return true;
}

std::vector<uint8_t> PipelinePrewarm::save_warm_list() const {
    std::lock_guard<std::mutex> lock(mu_);
    // A module-only entry is redundant when the list carries a full pipeline for the same shaders
    // (building the pipeline builds the modules).
    std::unordered_set<ShaderTriple, ShaderTripleHash> full;
    full.reserve(observed_.size());
    for (const Identity& id : observed_) {
        if (id.k2 != 0) full.insert(ShaderTriple{id.kind, id.k0, id.k1});
    }
    std::vector<size_t> keep;
    keep.reserve(observed_.size());
    bool any_detail = false;
    size_t detail_bytes_total = 0;
    for (size_t i = 0; i < observed_.size(); ++i) {
        const Identity& id = observed_[i];
        if (id.k2 == 0 && full.find(ShaderTriple{id.kind, id.k0, id.k1}) != full.end()) continue;
        if (id.k2 != 0) {
            any_detail = true;
            if (observed_detail_[i]) detail_bytes_total += observed_detail_[i]->size();
        }
        keep.push_back(i);
    }
    const size_t total_bytes =
        kPrewarmListHeaderBytes + keep.size() * kPrewarmListEntryBytes + (any_detail ? detail_bytes_total : 0);
    std::vector<uint8_t> out;
    out.reserve(total_bytes);
    out.push_back('P');
    out.push_back('G');
    out.push_back('6');
    out.push_back('W');
    put32(out, any_detail ? kPrewarmListVersionPipelines : kPrewarmListVersionModules);
    put32(out, static_cast<uint32_t>(keep.size()));
    put32(out, 0);
    for (size_t i : keep) {
        const Identity& id = observed_[i];
        const PrewarmDetail& detail = observed_detail_[i];
        out.push_back(static_cast<uint8_t>(id.kind));
        if (any_detail) {
            for (int j = 0; j < 3; ++j) out.push_back(0);
            put32(out, detail ? static_cast<uint32_t>(detail->size()) : 0);
        } else {
            for (int j = 0; j < 7; ++j) out.push_back(0);
        }
        put64(out, id.k0);
        put64(out, id.k1);
        if (any_detail && detail) out.insert(out.end(), detail->begin(), detail->end());
    }
    return out;
}

bool PipelinePrewarm::note_locked(const Identity& id, PrewarmDetail detail) {
    if (!observed_set_.emplace(id, static_cast<uint32_t>(observed_.size())).second) return false;
    if (observed_.size() < kPrewarmMaxEntries) {
        observed_.push_back(id);
        observed_detail_.push_back(std::move(detail));
    }
    auto it = by_identity_.find(id);
    if (it != by_identity_.end()) {
        Slot& s = slots_[it->second - 1];
        if (s.state == State::Queued) {
            set_slot_state(s, State::Skipped);  // the title got there first; dropped from the queue on take()
            ++retired_queued_;
        } else if (s.state == State::Done) {
            set_slot_state(s, State::Used);  // built ahead of the title's first use
        }
    }
    return true;
}

bool PipelinePrewarm::note_title_pipeline(uint32_t kind, uint64_t key0, uint64_t key1) {
    if (key0 == 0 || kind > kPrewarmKindCompute) return false;
    std::lock_guard<std::mutex> lock(mu_);
    return note_locked(Identity{kind, key0, key1, 0}, nullptr);
}

bool PipelinePrewarm::note_title_pipeline_full(uint32_t kind, uint64_t key0, uint64_t key1,
                                               std::vector<uint8_t> detail) {
    if (detail.empty()) return note_title_pipeline(kind, key0, key1);
    if (key0 == 0 || kind > kPrewarmKindCompute || detail.size() > kPrewarmMaxDetailBytes) return false;
    const Identity id{kind, key0, key1, hash_detail(detail)};
    std::lock_guard<std::mutex> lock(mu_);
    if (observed_set_.find(id) != observed_set_.end()) return false;
    auto d = std::make_shared<const std::vector<uint8_t>>(std::move(detail));
    return note_locked(id, std::move(d));
}

bool PipelinePrewarm::note_task_used(uint32_t task_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (task_id == 0 || task_id > slots_.size()) return false;
    const Slot& s = slots_[task_id - 1];
    return note_locked(s.id, s.detail);
}

PrewarmDetail PipelinePrewarm::detail_of(uint32_t task_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (task_id == 0 || task_id > slots_.size()) return nullptr;
    return slots_[task_id - 1].detail;
}

bool PipelinePrewarm::enqueue_priority(uint32_t kind, uint64_t key0, uint64_t key1, std::vector<uint8_t> detail) {
    if (key0 == 0 || kind > kPrewarmKindCompute || detail.size() > kPrewarmMaxDetailBytes) return false;
    const uint64_t k2 = detail.empty() ? 0 : hash_detail(detail);
    const Identity id{kind, key0, key1, k2};
    std::lock_guard<std::mutex> lock(mu_);
    if (observed_set_.find(id) != observed_set_.end()) return false;
    auto it = by_identity_.find(id);
    if (it != by_identity_.end()) {
        const uint32_t task_id = it->second;
        Slot& s = slots_[task_id - 1];
        if (s.state != State::Queued) return false;
        if (!s.in_priority) {
            s.in_priority = true;
            priority_queue_.push_back(task_id);
            ++priority_queued_;
        }
        return true;
    }
    if (!resolvable(id) || slots_.size() >= kPrewarmMaxEntries) return false;
    PrewarmDetail d =
        detail.empty() ? nullptr : std::make_shared<const std::vector<uint8_t>>(std::move(detail));
    slots_.push_back({id, State::Queued, std::move(d), true});
    adjust_state_counters(slots_.back(), State::Queued, +1);
    const uint32_t task_id = static_cast<uint32_t>(slots_.size());
    by_identity_.emplace(id, task_id);
    priority_queue_.push_back(task_id);
    ++priority_queued_;
    return true;
}

size_t PipelinePrewarm::take(PrewarmTask* out, size_t max) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t n = 0;
    auto drain_queue = [&](std::deque<uint32_t>& q) {
        while (n < max && in_flight_ < max_in_flight_ && !q.empty()) {
            const uint32_t task = q.front();
            q.pop_front();
            Slot& s = slots_[task - 1];
            if (s.state != State::Queued) continue;
            set_slot_state(s, State::InFlight);
            out[n++] = PrewarmTask{task, s.id.kind, s.id.k0, s.id.k1};
        }
    };
    drain_queue(priority_queue_);
    drain_queue(queue_);
    return n;
}

bool PipelinePrewarm::complete(uint32_t id, bool ok, uint64_t build_ns) {
    std::lock_guard<std::mutex> lock(mu_);
    if (id == 0 || id > slots_.size()) return false;
    Slot& s = slots_[id - 1];
    if (s.state != State::InFlight) return false;
    const State next = !ok ? State::Failed : (observed_set_.count(s.id) != 0 ? State::Used : State::Done);
    set_slot_state(s, next);
    longest_ns_ = std::max(longest_ns_, build_ns);
    total_ns_ += build_ns;
    ++builds_completed_;
    ++latency_buckets_[latency_bucket_for_ns(build_ns)];
    return true;
}

bool PipelinePrewarm::complete_lost(uint32_t id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (id == 0 || id > slots_.size()) return false;
    Slot& s = slots_[id - 1];
    if (s.state != State::InFlight) return false;
    set_slot_state(s, State::Skipped);
    ++races_lost_;
    return true;
}

PrewarmProgress PipelinePrewarm::progress() const {
    std::lock_guard<std::mutex> lock(mu_);
    PrewarmProgress p;
    p.total = static_cast<uint32_t>(slots_.size());
    p.queued = queued_count_;
    p.in_flight = in_flight_;
    p.done = done_count_;
    p.failed = failed_count_;
    p.skipped = skipped_count_;
    p.observed = static_cast<uint32_t>(observed_.size());
    p.longest_ns = longest_ns_;
    p.total_ns = total_ns_;
    return p;
}

PrewarmUse PipelinePrewarm::use_stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    PrewarmUse u;
    u.wasted = mod_done_;
    u.hit = mod_used_;
    u.prewarmed = mod_used_ + mod_done_;
    u.pipelines_wasted = full_done_;
    u.pipelines_hit = full_used_;
    u.pipelines_prewarmed = full_used_ + full_done_;
    return u;
}

PrewarmDiagnostics PipelinePrewarm::diagnostics() const {
    std::lock_guard<std::mutex> lock(mu_);
    PrewarmDiagnostics d;
    d.races_lost = races_lost_;
    d.retired_queued = retired_queued_;
    d.pack_skipped = pack_skipped_;
    d.priority_queued = priority_queued_;
    d.builds_completed = builds_completed_;
    d.avg_build_ns = builds_completed_ != 0 ? (total_ns_ / builds_completed_) : 0;
    d.longest_build_ns = longest_ns_;
    for (size_t i = 0; i < 5; ++i) d.latency_buckets[i] = latency_buckets_[i];
    return d;
}

void PipelinePrewarm::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    slots_.clear();
    by_identity_.clear();
    priority_queue_.clear();
    queue_.clear();
    observed_.clear();
    observed_detail_.clear();
    observed_set_.clear();
    queued_count_ = 0;
    in_flight_ = 0;
    done_count_ = 0;
    failed_count_ = 0;
    skipped_count_ = 0;
    mod_done_ = 0;
    mod_used_ = 0;
    full_done_ = 0;
    full_used_ = 0;
    races_lost_ = 0;
    retired_queued_ = 0;
    pack_skipped_ = 0;
    priority_queued_ = 0;
    builds_completed_ = 0;
    for (size_t i = 0; i < 5; ++i) latency_buckets_[i] = 0;
    longest_ns_ = 0;
    total_ns_ = 0;
}

PipelinePrewarm& pipeline_prewarm() {
    static PipelinePrewarm instance;
    return instance;
}

}  // namespace playgta6::gpu
