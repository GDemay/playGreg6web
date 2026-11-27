#include "guest/guest_watch.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace playgta6::guest {

namespace {

bool parse_number(std::string_view text, int base, uint64_t* out) {
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        text.remove_prefix(2);
        base = 16;
    }
    if (text.empty() || text.size() > 16) {
        return false;
    }
    uint64_t v = 0;
    for (char c : text) {
        unsigned d = 0;
        if (c >= '0' && c <= '9') {
            d = static_cast<unsigned>(c - '0');
        } else if (base == 16 && c >= 'a' && c <= 'f') {
            d = static_cast<unsigned>(c - 'a' + 10);
        } else if (base == 16 && c >= 'A' && c <= 'F') {
            d = static_cast<unsigned>(c - 'A' + 10);
        } else {
            return false;
        }
        if (v > (UINT64_MAX - d) / static_cast<unsigned>(base)) {
            return false;
        }
        v = v * static_cast<unsigned>(base) + d;
    }
    *out = v;
    return true;
}

}  // namespace

bool parse_watch_spec(std::string_view text, WatchRange* out) {
    WatchRange r;
    r.len = 8;
    if (!text.empty() && text[0] == '*') {
        r.indirect = true;
        text.remove_prefix(1);
    }
    const size_t colon = text.find(':');
    std::string_view addr = text.substr(0, colon);
    if (r.indirect) {
        const size_t plus = addr.find('+');
        if (plus != std::string_view::npos) {
            if (!parse_number(addr.substr(plus + 1), 16, &r.offset)) {
                return false;
            }
            addr = addr.substr(0, plus);
        }
    }
    if (!parse_number(addr, 16, &r.va)) {
        return false;
    }
    if (colon != std::string_view::npos && !parse_number(text.substr(colon + 1), 10, &r.len)) {
        return false;
    }
    if (r.va == 0 || r.len == 0 || r.len > GuestWatch::kMaxWatchLen || r.va > UINT64_MAX - r.len) {
        return false;
    }
    *out = r;
    return true;
}

void GuestWatch::configure(std::vector<WatchRange> ranges, uint64_t line_budget) {
    ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [](const WatchRange& r) { return r.len == 0; }),
                 ranges.end());
    if (ranges.size() > kMaxRanges) {
        ranges.resize(kMaxRanges);
    }
    std::sort(ranges.begin(), ranges.end(), [](const WatchRange& a, const WatchRange& b) { return a.va < b.va; });
    ranges_ = std::move(ranges);
    budget_ = line_budget;
    lines_.store(0, std::memory_order_relaxed);
    for (size_t i = 0; i < kMaxFunctions; ++i) {
        entry_va_[i].store(0, std::memory_order_relaxed);
        entry_count_[i].store(0, std::memory_order_relaxed);
    }
}

bool GuestWatch::overlaps(uint64_t va, uint64_t len) const {
    const uint64_t end = len == 0 || va > UINT64_MAX - len ? (len == 0 ? va + 1 : UINT64_MAX) : va + len;
    for (const WatchRange& r : ranges_) {
        if (!r.indirect && va < r.va + r.len && r.va < end) {
            return true;
        }
    }
    return false;
}

std::string GuestWatch::snapshot(const std::function<bool(uint64_t, void*, size_t)>& read) const {
    std::string o;
    for (const WatchRange& r : ranges_) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(r.len, kDumpBytes));
        uint8_t bytes[kDumpBytes];
        char head[96];
        uint64_t at = r.va;
        if (r.indirect) {
            uint64_t base = 0;
            const bool ok = read && read(r.va, &base, sizeof(base));
            std::snprintf(head, sizeof(head), " [*0x%" PRIx64 "+0x%" PRIx64 ":%" PRIu64 "]", r.va, r.offset, r.len);
            o += head;
            if (!ok || base == 0) {
                o += ok ? "=null" : "=unreadable";
                continue;
            }
            at = base + r.offset;
            std::snprintf(head, sizeof(head), "@0x%" PRIx64 "=", at);
        } else {
            std::snprintf(head, sizeof(head), " [0x%" PRIx64 ":%" PRIu64 "]=", r.va, r.len);
        }
        o += head;
        if (!read || !read(at, bytes, n)) {
            o += "unreadable";
            continue;
        }
        // Little-endian qwords (the natural view of the counters and pointers usually watched);
        // a tail shorter than 8 bytes is printed as bytes.
        size_t k = 0;
        for (; k + 8 <= n; k += 8) {
            uint64_t q = 0;
            std::memcpy(&q, bytes + k, 8);
            char word[24];
            std::snprintf(word, sizeof(word), "%s%016" PRIx64, k == 0 ? "" : ",", q);
            o += word;
        }
        for (; k < n; ++k) {
            char b[8];
            std::snprintf(b, sizeof(b), "%s%02x", k == 0 ? "" : (k % 8 == 0 ? "," : ""), bytes[k]);
            o += b;
        }
        if (n < r.len) {
            o += "...";
        }
    }
    return o;
}

bool GuestWatch::take_entry(uint64_t va) {
    if (va == 0) {
        return false;
    }
    for (size_t i = 0; i < kMaxFunctions; ++i) {
        uint64_t cur = entry_va_[i].load(std::memory_order_acquire);
        if (cur == 0) {
            uint64_t expected = 0;
            if (entry_va_[i].compare_exchange_strong(expected, va, std::memory_order_acq_rel)) {
                cur = va;
            } else {
                cur = expected;
            }
        }
        if (cur == va) {
            return entry_count_[i].fetch_add(1, std::memory_order_relaxed) < kEntriesPerFunction;
        }
    }
    return false;
}

bool GuestWatch::take_line() {
    return lines_.fetch_add(1, std::memory_order_relaxed) < budget_;
}

}  // namespace playgta6::guest
