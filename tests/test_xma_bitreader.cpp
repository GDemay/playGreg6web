// Include the implementation to exercise its private reader without a public test API.
#include "../runtime/os/xma_decoder.cpp"
#include <cstdio>
#include <cstdlib>

namespace {
struct Reference {
    const uint8_t* data;
    uint32_t pos, limit;
    bool error = false;
    Reference(const uint8_t* d, uint32_t p, uint32_t l) : data(d), pos(p), limit(l) {
        if (!data || pos > limit) { pos = limit; error = true; }
    }
    uint32_t bits_left() const { return pos < limit ? limit - pos : 0; }
    bool read_bits(uint32_t n, uint32_t& out) {
        if (!n) { out = 0; return !error; }
        if (error || n > 32 || n > bits_left()) { error = true; out = 0; return false; }
        uint32_t value = 0;
        for (uint32_t i = 0; i < n; ++i) {
            value = (value << 1) | ((data[pos >> 3] >> (7 - (pos & 7))) & 1);
            ++pos;
        }
        out = value; return true;
    }
};
void check(const uint8_t* data, uint32_t start, uint32_t limit, uint32_t width) {
    playgta6::os::xma::BitReader actual(data, start, limit);
    Reference expected(data, start, limit);
    // Continue after success/failure to check cursor and sticky-error semantics.
    for (uint32_t n : {width, 0u, 1u, 32u, 33u, 0u}) {
        uint32_t a = 0xdeadbeef, b = a;
        const bool ar = actual.read_bits(n, a), br = expected.read_bits(n, b);
        if (ar != br || a != b || actual.bits_left() != expected.bits_left() || actual.ok() == expected.error) {
            std::fprintf(stderr, "reader mismatch start=%u limit=%u width=%u n=%u\n", start, limit, width, n);
            std::abort();
        }
    }
}
}
int main() {
    std::array<uint8_t, 32> bytes{};
    uint32_t state = 0x12345678;
    for (unsigned round = 0; round < 4; ++round) {
        for (auto& byte : bytes) { state ^= state << 13; state ^= state >> 17; state ^= state << 5; byte = uint8_t(state); }
        for (uint32_t limit = 0; limit <= bytes.size() * 8; ++limit)
            for (uint32_t start = 0; start <= limit + 1; ++start)
                for (uint32_t width = 0; width <= 40; ++width) check(bytes.data(), start, limit, width);
    }
    for (uint32_t n = 0; n <= 40; ++n) check(nullptr, 0, 256, n);
    std::puts("XMA bitreader differential checks passed");
}
