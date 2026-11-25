// Per-reason log rate limiter (O36): the first kFirst occurrences of a reason are logged, then one
// line each time the running count reaches a power of two (16, 32, 64, ...). Counts stay exact.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace playgta6::gpu {

class LogRateLimiter {
public:
    static constexpr uint64_t kFirst = 8;
    static constexpr size_t kMaxReasons = 512;  // bounds memory when messages vary without limit

    struct Decision {
        bool log = false;
        uint64_t count = 0;  // running count of this reason, this occurrence included
    };

    // Not thread-safe: the caller serializes.
    Decision note(const std::string& message) {
        std::string key = reason_of(message);
        auto it = counts_.find(key);
        if (it == counts_.end()) {
            if (counts_.size() >= kMaxReasons) {
                it = counts_.try_emplace("<other reasons>").first;
            } else {
                it = counts_.try_emplace(std::move(key)).first;
            }
        }
        const uint64_t n = ++it->second;
        Decision d;
        d.count = n;
        d.log = n <= kFirst || (n & (n - 1)) == 0;
        return d;
    }

    uint64_t count_of(const std::string& message) const {
        auto it = counts_.find(reason_of(message));
        return it == counts_.end() ? 0 : it->second;
    }

    // The reason key: the message with numbers (decimal and 0x hex) folded to '#', so the same
    // complaint about different addresses or sizes counts as one reason.
    static std::string reason_of(const std::string& message) {
        std::string out;
        out.reserve(message.size());
        for (size_t i = 0; i < message.size();) {
            const char c = message[i];
            if (c == '0' && i + 1 < message.size() && (message[i + 1] == 'x' || message[i + 1] == 'X')) {
                i += 2;
                while (i < message.size() && is_hex(message[i])) ++i;
                out.push_back('#');
            } else if (c >= '0' && c <= '9') {
                while (i < message.size() && message[i] >= '0' && message[i] <= '9') ++i;
                out.push_back('#');
            } else {
                out.push_back(c);
                ++i;
            }
        }
        return out;
    }

private:
    static bool is_hex(char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }
    std::unordered_map<std::string, uint64_t> counts_;
};

}  // namespace playgta6::gpu
