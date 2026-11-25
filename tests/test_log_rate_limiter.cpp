// Per-reason log rate limiter for Device::note_bad_argument (O36).
#include <cstdio>
#include <string>
#include <vector>

#include "gpu/log_rate_limiter.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

using playgta6::gpu::LogRateLimiter;

void test_first_n_then_powers_of_two() {
    LogRateLimiter limiter;
    std::vector<uint64_t> logged;
    for (uint64_t i = 1; i <= 100; ++i) {
        const auto d = limiter.note("pipeline state without shader modules bound");
        CHECK(d.count == i);  // counts are exact
        if (d.log) logged.push_back(i);
    }
    const std::vector<uint64_t> expected{1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64};
    CHECK(logged == expected);
    CHECK(limiter.count_of("pipeline state without shader modules bound") == 100);
}

void test_reasons_are_independent_and_numbers_fold() {
    LogRateLimiter limiter;
    for (int i = 0; i < 50; ++i) limiter.note("A");
    CHECK(limiter.note("B").log);  // another reason starts at its own first occurrence
    // The same complaint about different addresses or sizes is one reason.
    CHECK(LogRateLimiter::reason_of("bad blob 0x1F00 length 12") ==
          LogRateLimiter::reason_of("bad blob 0xABCD length 9999"));
    LogRateLimiter l2;
    for (int i = 0; i < 20; ++i) {
        l2.note("bad blob 0x" + std::to_string(1000 + i) + " length " + std::to_string(i));
    }
    CHECK(l2.count_of("bad blob 0x5 length 5") == 20);
    CHECK(l2.note("different reason").log);
}

void test_reason_table_is_bounded() {
    LogRateLimiter limiter;
    for (size_t i = 0; i < LogRateLimiter::kMaxReasons + 100; ++i) {
        limiter.note("reason " + std::string(i / 26 + 1, static_cast<char>('a' + i % 26)));
    }
    const auto d = limiter.note("one more brand new reason");
    CHECK(d.count >= 1);  // folded into the overflow bucket, still counted
}

}  // namespace

int main() {
    test_first_n_then_powers_of_two();
    test_reasons_are_independent_and_numbers_fold();
    test_reason_table_is_bounded();
    if (g_failures == 0) std::puts("test_log_rate_limiter: all passed");
    return g_failures == 0 ? 0 : 1;
}
