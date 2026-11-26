// BCPack decoder (os/bcpack.h) on SYNTHETIC data only: made-up table files (bcpack_synthetic.h),
// streams from a test-only encoder, and hand-built block records. No real tables, no game data.
// Decoded BC blocks are checked against the public BC1/BC4/BC7 block layouts, read back here.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

#include "bcpack_synthetic.h"
#include "os/bcpack.h"
#include "os/surface_blocks.h"

namespace {

int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_bcpack.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace bp = playgta6::os::bcpack;
namespace sy = bcpack_synthetic;
using Record = std::vector<uint8_t>;

const sy::LaneSpec& spec_for(uint32_t bc) {
    static const std::vector<sy::LaneSpec> specs = sy::default_layouts();
    for (const sy::LaneSpec& s : specs) {
        if (s.bc == bc) {
            return s;
        }
    }
    return specs[0];
}

std::vector<uint8_t> decode_records(const bp::Tables& t, uint32_t bc, const std::vector<Record>& records,
                                    bp::DecodeStatus* status = nullptr) {
    const std::vector<uint8_t> src = sy::stream(t, spec_for(bc), records);
    CHECK(!src.empty());
    std::vector<uint8_t> dst(records.size() * bp::block_bytes(bc));
    const bp::DecodeStatus st = bp::decode(t, bc, src.data(), src.size(), dst.data(), dst.size());
    if (status != nullptr) {
        *status = st;
    } else {
        CHECK(st == bp::DecodeStatus::Ok);
    }
    return dst;
}

// Signed equivalent of `v` modulo 2^bits, zigzagged (what a residual symbol holds).
uint8_t residual(uint32_t v, uint32_t bits) {
    const int half = 1 << (bits - 1);
    const int s = static_cast<int>(v & ((1u << bits) - 1)) >= half ? static_cast<int>(v & ((1u << bits) - 1)) - (1 << bits)
                                                                    : static_cast<int>(v & ((1u << bits) - 1));
    return sy::zigzag(s);
}

uint64_t le64(const uint8_t* p) {
    uint64_t v = 0;
    for (int k = 7; k >= 0; --k) {
        v = (v << 8) | p[k];
    }
    return v;
}

// Public BC1 (four-colour mode) / BC4 index codes -> position along the endpoint line.
uint32_t bc1_position(uint32_t code) {
    const uint32_t pos[4] = {0, 3, 1, 2};
    return pos[code];
}
uint32_t bc4_position(uint32_t code) { return code == 0 ? 0 : code == 1 ? 7 : code - 1; }

// ---------------------------------------------------------------------------
// Table files
// ---------------------------------------------------------------------------

void test_table_parsing() {
    const sy::Builder full = sy::Builder::full();
    std::string error;
    const std::vector<uint8_t> file = full.file();
    const auto t = bp::parse_tables(file.data(), file.size(), &error);
    CHECK(t != nullptr);
    if (!t) {
        std::fprintf(stderr, "  parse error: %s\n", error.c_str());
        return;
    }
    for (uint32_t bc : {1u, 3u, 4u, 5u, 6u, 7u}) {
        CHECK(bp::supports(*t, bc));
    }
    CHECK(!bp::supports(*t, 2) && !bp::supports(*t, 0) && !bp::supports(*t, 15));
    CHECK(bp::supported_formats(*t) == "BC1, BC3, BC4, BC5, BC6H, BC7");

    auto parses = [](const sy::Builder& b, std::string* why = nullptr) {
        const std::vector<uint8_t> f = b.file();
        std::string e;
        const bool ok = bp::parse_tables(f.data(), f.size(), &e) != nullptr;
        if (why != nullptr) {
            *why = e;
        }
        return ok;
    };
    // Optional sections: without DIC7 BC7 is off, the rest still works; unknown tags are ignored.
    sy::Builder no_dic7 = full;
    no_dic7.sections.erase("DIC7");
    no_dic7.sections["ZZZZ"] = {1, 2, 3};
    {
        const std::vector<uint8_t> f = no_dic7.file();
        const auto t2 = bp::parse_tables(f.data(), f.size(), nullptr);
        CHECK(t2 != nullptr && !bp::supports(*t2, 7) && bp::supports(*t2, 4));
        CHECK(t2 && bp::supported_formats(*t2) == "BC1, BC3, BC4, BC5, BC6H");
    }
    sy::Builder bad_dic6 = full;
    bad_dic6.sections["DIC6"].pop_back();
    CHECK(!parses(bad_dic6, &error) && error.find("DIC6") != std::string::npos);
    bad_dic6 = full;
    bad_dic6.sections["DIC6"][25] = 15;  // entry 0's index-half mode
    CHECK(!parses(bad_dic6, &error) && error.find("mode 15") != std::string::npos);
    // Required sections, sizes, model tiling, lane widths.
    sy::Builder no_lane = full;
    no_lane.sections.erase("LANE");
    CHECK(!parses(no_lane, &error) && error.find("LANE") != std::string::npos);
    sy::Builder bad_model = full;
    bad_model.sections["MODL"][0] ^= 1;  // model 0 no longer covers 65536 slots
    CHECK(!parses(bad_model, &error) && error.find("model 0") != std::string::npos);
    sy::Builder empty_bucket = full;
    empty_bucket.sections["MODL"][2] = empty_bucket.sections["MODL"][3] = 0;
    CHECK(!parses(empty_bucket, &error) && error.find("empty bucket") != std::string::npos);
    sy::Builder short_rank = full;
    short_rank.sections["RANK"].pop_back();
    CHECK(!parses(short_rank));
    sy::Builder narrow = full;
    std::vector<sy::LaneSpec> layouts = sy::default_layouts();
    layouts[2].widths[5] = 2;  // below 3: the header could select a negative group
    narrow.set_layouts(layouts);
    CHECK(!parses(narrow, &error) && error.find("width") != std::string::npos);
    sy::Builder wrong_record = full;
    layouts = sy::default_layouts();
    layouts[4].lanes = 12;  // 12 x 2 symbols is not BC7's 32-symbol record
    wrong_record.set_layouts(layouts);
    CHECK(!parses(wrong_record));
    // Header problems.
    std::vector<uint8_t> f = file;
    f[0] = 'X';
    CHECK(bp::parse_tables(f.data(), f.size(), &error) == nullptr && error.find("magic") != std::string::npos);
    f = file;
    f[8] = 2;  // file version
    CHECK(bp::parse_tables(f.data(), f.size(), nullptr) == nullptr);
    f = file;
    f.resize(f.size() - 1);  // the last section runs past the end
    CHECK(bp::parse_tables(f.data(), f.size(), nullptr) == nullptr);
    CHECK(bp::parse_tables(nullptr, 0, nullptr) == nullptr);
    CHECK(bp::parse_tables(file.data(), 15, nullptr) == nullptr);
}

void test_tables_file_and_active() {
    bp::set_active_tables(nullptr);
    CHECK(bp::active_tables() == nullptr);
    std::string error;
    CHECK(!bp::load_tables_file("this/file/does/not/exist.bin", &error));
    CHECK(error.find("cannot open") != std::string::npos);
    const std::vector<uint8_t> file = sy::Builder::full().file();
    const std::string path = "test_bcpack_synthetic_tables.tmp";
    std::FILE* out = std::fopen(path.c_str(), "wb");
    CHECK(out != nullptr);
    if (out != nullptr) {
        std::fwrite(file.data(), 1, file.size(), out);
        std::fclose(out);
        CHECK(bp::load_tables_file(path, &error));
        CHECK(bp::active_tables() != nullptr && bp::supports(*bp::active_tables(), 7));
        std::remove(path.c_str());
    }
    bp::set_active_tables(nullptr);
    CHECK(bp::active_tables() == nullptr);
}

// ---------------------------------------------------------------------------
// Lanes: the rANS fragments and the byte schedule round-trip
// ---------------------------------------------------------------------------

void test_lane_round_trip() {
    // Mixed made-up widths: symbols stay below 2^(width + 1).
    std::vector<sy::LaneSpec> layouts = {
        {1, 12, 1, {7, 6, 5, 4, 3, 7, 6, 5, 4, 3, 7, 7}},
        {3, 12, 2, {3, 4, 5, 6, 7, 7, 7, 6, 5, 4, 3, 7}},
        {4, 12, 1, std::vector<uint8_t>(12, 7)},
        {5, 12, 2, {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}},
        {7, 16, 2, {3, 3, 7, 7, 6, 6, 5, 5, 4, 4, 7, 7, 6, 5, 4, 3}},
    };
    const auto t = sy::Builder::full(layouts).tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    uint64_t seed = 0x243F6A8885A308D3ull;
    auto next = [&seed] {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    for (const sy::LaneSpec& spec : layouts) {
        for (size_t blocks : {size_t{1}, size_t{2}, size_t{17}, size_t{300}, size_t{1500}}) {
            std::vector<Record> records(blocks, Record(spec.lanes * spec.per_block));
            for (Record& r : records) {
                for (size_t k = 0; k < r.size(); ++k) {
                    const uint32_t alphabet = 1u << (spec.widths[k / spec.per_block] + 1);
                    // Mostly small symbols (cheap), some anywhere in the alphabet.
                    const uint64_t v = next();
                    r[k] = static_cast<uint8_t>((v & 3) == 0 ? v % alphabet : (v >> 8) % 4);
                }
            }
            const std::vector<uint8_t> src = sy::stream(*t, spec, records);
            CHECK(!src.empty());
            struct Cursor {
                const std::vector<uint8_t>* src;
                size_t at;
            } cursor{&src, 0};
            auto fetch = [](void* ctx, size_t, uint8_t* byte) {
                auto* c = static_cast<Cursor*>(ctx);
                if (c->at >= c->src->size()) {
                    return false;
                }
                *byte = (*c->src)[c->at++];
                return true;
            };
            std::vector<uint8_t> got(blocks * spec.lanes * spec.per_block);
            CHECK(bp::decode_lanes(*t, spec.bc, blocks, fetch, &cursor, got.data()));
            std::vector<uint8_t> want;
            for (const Record& r : records) {
                want.insert(want.end(), r.begin(), r.end());
            }
            CHECK(got == want);
            if (blocks == 1500) {
                CHECK(src.size() > 12 * 260);  // long (two-byte length) fragments were exercised
            }
            // Cut short: the lane stage reports it instead of looping or reading past the end.
            const std::vector<uint8_t> half(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(src.size() / 2));
            Cursor short_cursor{&half, 0};
            CHECK(!bp::decode_lanes(*t, spec.bc, blocks, fetch, &short_cursor, got.data()));
        }
    }
}

// ---------------------------------------------------------------------------
// Plain formats
// ---------------------------------------------------------------------------

// BC1 symbol for one 2x2 quad: the four 2-bit residuals of pixels (top-left, top-right,
// bottom-left, bottom-right) with the synthetic identity rank table.
uint8_t quad_symbol(const uint32_t ix[16], uint32_t q) {
    const uint32_t corner = (q / 2) * 8 + (q % 2) * 2;
    const uint32_t px[4] = {corner, corner + 1, corner + 4, corner + 5};
    uint32_t v = 0;
    for (uint32_t k = 0; k < 4; ++k) {
        v |= (ix[px[k]] & 3u) << (2 * k);
    }
    return static_cast<uint8_t>(v);
}

// Checks a BC1 colour half against endpoints and line positions.
void check_color_half(const uint8_t* half, const uint32_t c0[3], const uint32_t c1[3], const uint32_t pos[16]) {
    const uint64_t v = le64(half);
    CHECK((v & 0xFFFF) == ((c0[0] << 11) | (c0[1] << 5) | c0[2]));
    CHECK(((v >> 16) & 0xFFFF) == ((c1[0] << 11) | (c1[1] << 5) | c1[2]));
    for (uint32_t p = 0; p < 16; ++p) {
        CHECK(bc1_position((v >> (32 + 2 * p)) & 3) == pos[p]);
    }
}

// Checks a BC4 half: endpoints are stored with the top bit flipped.
void check_alpha_half(const uint8_t* half, uint32_t e0, uint32_t e1, const uint32_t pos[16]) {
    const uint64_t v = le64(half);
    CHECK((v & 0xFF) == (e0 ^ 0x80));
    CHECK(((v >> 8) & 0xFF) == (e1 ^ 0x80));
    for (uint32_t p = 0; p < 16; ++p) {
        CHECK(bc4_position((v >> (16 + 3 * p)) & 7) == pos[p]);
    }
}

void test_bc1_references() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    const uint32_t c0[3] = {31, 40, 7};
    const uint32_t c1[3] = {2, 10, 30};
    uint32_t pos0[16];
    for (uint32_t p = 0; p < 16; ++p) {
        pos0[p] = (p * 3 + 1) % 4;
    }
    // Block 0: no references.
    Record b0(12, 0);
    for (uint32_t k = 0; k < 3; ++k) {
        b0[2 + k] = residual(c0[k], k == 1 ? 6 : 5);
        b0[5 + k] = residual(c1[k], k == 1 ? 6 : 5);
    }
    for (uint32_t q = 0; q < 4; ++q) {
        b0[8 + q] = quad_symbol(pos0, q);
    }
    // Block 1: both references to block 0 (index reference 1, endpoint delta 0); residuals +1.
    Record b1(12, 0);
    b1[1] = 1;
    for (uint32_t k = 0; k < 6; ++k) {
        b1[2 + k] = sy::zigzag(1);
    }
    const uint32_t ones[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    for (uint32_t q = 0; q < 4; ++q) {
        b1[8 + q] = quad_symbol(ones, q);
    }
    // Block 2: indices from dictionary entry 0 (reference 3 > block 2), endpoints from block 1
    // (delta -2 from the index reference); no residuals.
    Record b2(12, 0);
    b2[0] = sy::zigzag(-2);
    b2[1] = 3;
    const std::vector<uint8_t> dst = decode_records(*t, 1, {b0, b1, b2});
    check_color_half(dst.data(), c0, c1, pos0);
    const uint32_t c0b[3] = {0, 41, 8};  // 31 + 1 wraps at 5 bits
    const uint32_t c1b[3] = {3, 11, 31};
    uint32_t pos1[16];
    for (uint32_t p = 0; p < 16; ++p) {
        pos1[p] = (pos0[p] + 1) % 4;
    }
    check_color_half(dst.data() + 8, c0b, c1b, pos1);
    uint32_t dict_pos[16];
    for (uint32_t p = 0; p < 16; ++p) {
        dict_pos[p] = (p + 6) & 3;  // synthetic DIC1 entry 0: index e (6..21) holds e & 3
    }
    check_color_half(dst.data() + 16, c0b, c1b, dict_pos);
}

void test_bc4_bc5_bc3() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    uint32_t pos[16];
    uint32_t pos_next[16];
    for (uint32_t p = 0; p < 16; ++p) {
        pos[p] = p & 7;
        pos_next[p] = (p + 1) & 7;
    }
    auto pair_symbols = [](const uint32_t* ix, uint8_t* out, size_t pairs) {
        for (size_t k = 0; k < pairs; ++k) {
            out[k] = static_cast<uint8_t>((ix[2 * k] & 7) | ((ix[2 * k + 1] & 7) << 3));
        }
    };
    // BC4: endpoints 10 and -3 (253), then a block predicting from it with +1 everywhere.
    Record b0(12, 0);
    b0[2] = sy::zigzag(10);
    b0[3] = sy::zigzag(-3);
    pair_symbols(pos, &b0[4], 8);
    Record b1(12, 0);
    b1[1] = 1;
    b1[2] = b1[3] = sy::zigzag(1);
    for (size_t k = 0; k < 8; ++k) {
        b1[4 + k] = 1 | (1 << 3);
    }
    std::vector<uint8_t> dst = decode_records(*t, 4, {b0, b1});
    check_alpha_half(dst.data(), 10, 253, pos);
    check_alpha_half(dst.data() + 8, 11, 254, pos_next);

    // BC5: two BC4 halves; references use four symbols (two nibble-interleaved bytes).
    Record c(24, 0);
    for (uint32_t k = 0; k < 4; ++k) {
        c[4 + k] = residual(40 * k + 5, 8);
    }
    uint32_t pos32[32];
    for (uint32_t p = 0; p < 32; ++p) {
        pos32[p] = (p * 5) & 7;
    }
    pair_symbols(pos32, &c[8], 16);
    dst = decode_records(*t, 5, {c});
    check_alpha_half(dst.data(), 5, 45, pos32);
    check_alpha_half(dst.data() + 8, 85, 125, pos32 + 16);

    // BC3: alpha (symbols 4-13) in the first half, colour (14-23) in the second.
    Record d(24, 0);
    d[4] = residual(200, 8);
    d[5] = residual(17, 8);
    pair_symbols(pos, &d[6], 8);
    const uint32_t c0[3] = {20, 50, 3};
    const uint32_t c1[3] = {9, 1, 28};
    for (uint32_t ch = 0; ch < 3; ++ch) {
        d[14 + 2 * ch] = residual(c0[ch], ch == 1 ? 6 : 5);
        d[15 + 2 * ch] = residual(c1[ch], ch == 1 ? 6 : 5);
    }
    uint32_t cpos[16];
    for (uint32_t p = 0; p < 16; ++p) {
        cpos[p] = (p / 4) & 3;
    }
    for (uint32_t q = 0; q < 4; ++q) {
        d[20 + q] = quad_symbol(cpos, q);
    }
    dst = decode_records(*t, 3, {d});
    check_alpha_half(dst.data(), 200, 17, pos);
    check_color_half(dst.data() + 8, c0, c1, cpos);

    // BC3 with both references into the dictionary via the nibble-interleaved symbols: index
    // symbols (2, 0) merge to 4 (even nibble bit 1 -> bit 2), endpoint delta 0. Reference 4 of
    // block 0 is dictionary entry 3. Synthetic DIC3 entry 3: endpoint e (0..7) = 9 + e, index j
    // (0..31) = (17 + j) & 3.
    Record e(24, 0);
    e[2] = 2;
    dst = decode_records(*t, 3, {e});
    uint32_t ea[8];
    for (uint32_t k = 0; k < 8; ++k) {
        ea[k] = (9 + k) & 0x1F;
    }
    const uint32_t ec0[3] = {ea[0], ea[1], ea[2]};
    const uint32_t ec1[3] = {ea[3], ea[4], ea[5]};
    uint32_t apos[16];
    uint32_t cpos2[16];
    for (uint32_t p = 0; p < 16; ++p) {
        apos[p] = ((9 + 8 + p) & 3) & 7;
        cpos2[p] = (9 + 24 + p) & 3;
    }
    check_alpha_half(dst.data(), ea[6], ea[7], apos);
    check_color_half(dst.data() + 8, ec0, ec1, cpos2);
}

// ---------------------------------------------------------------------------
// BC7
// ---------------------------------------------------------------------------

class BitReader {
public:
    explicit BitReader(const uint8_t* b) : b_(b) {}
    uint32_t get(uint32_t n) {
        uint32_t v = 0;
        for (uint32_t k = 0; k < n; ++k, ++at_) {
            v |= static_cast<uint32_t>((b_[at_ / 8] >> (at_ % 8)) & 1u) << k;
        }
        return v;
    }
    uint32_t at() const { return at_; }

private:
    const uint8_t* b_;
    uint32_t at_ = 0;
};

// A BC7 block read with the public layout.
struct Bc7Fields {
    uint32_t mode = 99, partition = 0, rotation = 0, index_select = 0;
    uint32_t ep[3][2][4] = {};  // subset, endpoint, R G B A (field values, without p-bits)
    uint32_t pbit[3][2] = {};   // mode 1: pbit[s][0] is the subset's shared bit
    uint32_t index[2][16] = {};  // in block order: the first index block, then the second
    uint32_t bits = 0;
};

Bc7Fields read_bc7(const uint8_t* block, std::initializer_list<uint32_t> anchors) {
    static const uint32_t kColor[8] = {4, 6, 5, 7, 5, 7, 7, 5};
    static const uint32_t kAlpha[8] = {0, 0, 0, 0, 6, 8, 7, 5};
    static const uint32_t kSubsets[8] = {3, 2, 3, 2, 1, 1, 1, 2};
    static const uint32_t kPartBits[8] = {4, 6, 6, 6, 0, 0, 0, 6};
    Bc7Fields f;
    BitReader r(block);
    uint32_t mode = 0;
    while (mode < 8 && r.get(1) == 0) {
        ++mode;
    }
    f.mode = mode;
    if (mode >= 8) {
        return f;
    }
    f.partition = r.get(kPartBits[mode]);
    if (mode == 4 || mode == 5) {
        f.rotation = r.get(2);
    }
    if (mode == 4) {
        f.index_select = r.get(1);
    }
    const uint32_t subsets = kSubsets[mode];
    for (uint32_t ch = 0; ch < 4; ++ch) {
        const uint32_t n = ch < 3 ? kColor[mode] : kAlpha[mode];
        for (uint32_t s = 0; s < subsets; ++s) {
            for (uint32_t e = 0; e < 2; ++e) {
                f.ep[s][e][ch] = r.get(n);
            }
        }
    }
    if (mode == 1) {
        for (uint32_t s = 0; s < subsets; ++s) {
            f.pbit[s][0] = r.get(1);
        }
    } else if (mode == 0 || mode == 3 || mode == 6 || mode == 7) {
        for (uint32_t s = 0; s < subsets; ++s) {
            for (uint32_t e = 0; e < 2; ++e) {
                f.pbit[s][e] = r.get(1);
            }
        }
    }
    static const uint32_t kFirstIndex[8] = {3, 3, 2, 2, 2, 2, 4, 2};
    static const uint32_t kSecondIndex[8] = {0, 0, 0, 0, 3, 2, 0, 0};
    auto read_set = [&](uint32_t set, uint32_t bits) {
        for (uint32_t p = 0; p < 16; ++p) {
            bool anchor = false;
            for (uint32_t a : anchors) {
                anchor = anchor || a == p;
            }
            f.index[set][p] = r.get(bits - (anchor ? 1 : 0));
        }
    };
    read_set(0, kFirstIndex[mode]);
    if (kSecondIndex[mode] != 0) {
        read_set(1, kSecondIndex[mode]);
    }
    f.bits = r.at();
    return f;
}

// A BC7 record for control code `code` with endpoint column values (R0 R1 G0 G1 B0 B1 A0 A1 per
// subset, at the column's stored precision) and index values of both sets.
Record bc7_record(uint32_t code, const uint32_t ep[3][8], const uint32_t prec[4], const uint32_t ix0[16],
                  const uint32_t ix1[16], bool three) {
    Record s(32, 0);
    s[30] = static_cast<uint8_t>(code & 31);
    s[31] = static_cast<uint8_t>(code >> 5);
    auto put = [&](size_t at, uint32_t row, uint32_t col) { s[at] = residual(ep[row][col], prec[col / 2]); };
    for (uint32_t col = 0; col < 6; ++col) {
        put(4 + col, 0, col);
        put(12 + col, 1, col);
    }
    if (three) {
        put(20, 2, 0);
        put(21, 2, 1);
        put(10, 2, 2);
        put(11, 2, 3);
        put(18, 2, 4);
        put(19, 2, 5);
    } else if (prec[3] != 0) {
        put(10, 0, 6);
        put(11, 0, 7);
        put(18, 1, 6);
        put(19, 1, 7);
    }
    auto nib = [](uint32_t v) {  // a 4-bit signed residual holding v modulo 16
        return static_cast<uint32_t>(sy::zigzag(v >= 8 ? static_cast<int>(v) - 16 : static_cast<int>(v)));
    };
    auto spread = [](uint32_t even, uint32_t odd) {
        uint32_t b = 0;
        for (uint32_t k = 0; k < 4; ++k) {
            b |= ((even >> k) & 1u) << (2 * k);
            b |= ((odd >> k) & 1u) << (2 * k + 1);
        }
        return static_cast<uint8_t>(b);
    };
    const uint32_t offsets[2][4] = {{22, 24, 26, 28}, {12, 14, 16, 20}};
    for (uint32_t set = 0; set < 2; ++set) {
        const uint32_t* ix = set == 0 ? ix0 : ix1;
        if (ix == nullptr) {
            continue;
        }
        for (uint32_t k = 0; k < 4; ++k) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t p = 4 * k + 2 * half;
                const uint32_t first = nib(ix[p]);
                const uint32_t second = nib(ix[p + 1]);
                s[offsets[set][k] + half] = k < 3 ? spread(second, first) : spread(first, second);
            }
        }
    }
    return s;
}

void test_bc7_modes() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    // Mode 6 (code 0): R keeps its p-bit as the lowest of 8 bits; G B A 7 bits.
    {
        const uint32_t prec[4] = {8, 7, 7, 7};
        const uint32_t ep[3][8] = {{0x55, 0xFF, 3, 0, 100, 123, 63, 64}, {}, {}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = p == 0 ? 5 : (p * 7) % 16;  // anchor 0 below 8
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(0, ep, prec, ix, nullptr, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0});
        CHECK(f.mode == 6 && f.bits == 128);
        CHECK(f.ep[0][0][0] == 0x2A && f.ep[0][1][0] == 0x7F);
        CHECK(f.pbit[0][0] == 1 && f.pbit[0][1] == 1);
        CHECK(f.ep[0][0][1] == 3 && f.ep[0][1][1] == 0 && f.ep[0][0][2] == 100 && f.ep[0][1][2] == 123);
        CHECK(f.ep[0][0][3] == 63 && f.ep[0][1][3] == 64);
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix[p]);
        }
    }
    // Mode 1 (code 1, partition 17: second-subset anchor at pixel 2): one p-bit per subset.
    {
        const uint32_t prec[4] = {7, 6, 6, 0};
        const uint32_t ep[3][8] = {{0x41, 0x10, 1, 2, 3, 4, 0, 0}, {0x22, 0x7F, 60, 61, 62, 63, 0, 0}, {}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = (p == 0 || p == 2) ? p / 2 + 1 : (p * 3 + 2) % 8;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(1, ep, prec, ix, nullptr, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0, 2});
        CHECK(f.mode == 1 && f.partition == 17 && f.bits == 128);
        CHECK(f.ep[0][0][0] == 0x20 && f.ep[0][1][0] == 0x08 && f.pbit[0][0] == 1);
        CHECK(f.ep[1][0][0] == 0x11 && f.ep[1][1][0] == 0x3F && f.pbit[1][0] == 0);
        CHECK(f.ep[0][0][1] == 1 && f.ep[0][1][2] == 4 && f.ep[1][0][1] == 60 && f.ep[1][1][2] == 63);
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix[p]);
        }
    }
    // Mode 0 (code 3, partition 0: anchors 0, 3, 15): three subsets, the third's G and B come from
    // the symbols alpha uses elsewhere.
    {
        const uint32_t prec[4] = {5, 4, 4, 0};
        const uint32_t ep[3][8] = {
            {1, 2, 3, 4, 5, 6, 0, 0}, {7, 8, 9, 10, 11, 12, 0, 0}, {31, 30, 15, 14, 13, 1, 0, 0}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = (p == 0 || p == 3 || p == 15) ? 3 : (p + 4) % 8;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(3, ep, prec, ix, nullptr, true)});
        const Bc7Fields f = read_bc7(dst.data(), {0, 3, 15});
        CHECK(f.mode == 0 && f.partition == 0 && f.bits == 128);
        CHECK(f.ep[2][0][0] == 15 && f.pbit[2][0] == 1 && f.ep[2][1][0] == 15 && f.pbit[2][1] == 0);
        CHECK(f.ep[2][0][1] == 15 && f.ep[2][1][1] == 14 && f.ep[2][0][2] == 13 && f.ep[2][1][2] == 1);
        CHECK(f.ep[1][0][0] == 3 && f.pbit[1][0] == 1 && f.ep[1][1][2] == 12);
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix[p]);
        }
    }
    // Mode 5 (code 2, rotation 2): green and alpha trade places, so the 8-bit column pair is the
    // one written in alpha's place.
    {
        const uint32_t prec[4] = {7, 8, 7, 7};
        const uint32_t ep[3][8] = {{10, 20, 200, 201, 30, 40, 50, 60}, {}, {}};
        uint32_t ix0[16];
        uint32_t ix1[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix0[p] = p == 0 ? 1 : p % 4;
            ix1[p] = p == 0 ? 0 : (p + 1) % 4;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(2, ep, prec, ix0, ix1, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0});
        CHECK(f.mode == 5 && f.rotation == 2 && f.bits == 128);
        CHECK(f.ep[0][0][0] == 10 && f.ep[0][1][0] == 20);
        CHECK(f.ep[0][0][1] == 50 && f.ep[0][1][1] == 60);    // the A columns in G's place
        CHECK(f.ep[0][0][3] == 200 && f.ep[0][1][3] == 201);  // the G columns in A's place
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix0[p] && f.index[1][p] == ix1[p]);
        }
    }
    // Mode 4 (code 4, rotation 1, index selection 1): the 2-bit index set is written first.
    {
        const uint32_t prec[4] = {6, 5, 5, 5};
        const uint32_t ep[3][8] = {{40, 41, 1, 2, 3, 4, 30, 31}, {}, {}};
        uint32_t ix0[16];
        uint32_t ix1[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix0[p] = p == 0 ? 3 : (p * 5) % 8;
            ix1[p] = p == 0 ? 1 : (p * 3) % 4;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(4, ep, prec, ix0, ix1, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0});
        CHECK(f.mode == 4 && f.rotation == 1 && f.index_select == 1 && f.bits == 128);
        CHECK(f.ep[0][0][0] == 30 && f.ep[0][1][0] == 31);  // A columns in R's place
        CHECK(f.ep[0][0][3] == 40 && f.ep[0][1][3] == 41);  // R columns (6 bits) in A's place
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix1[p]);  // first in the block: the 2-bit set
            CHECK(f.index[1][p] == ix0[p]);
        }
    }
}

void test_bc7_references() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    // Block 0: mode 6 against dictionary entry 0 for both (references 1). Synthetic DIC7 entry 0:
    // rows[r][c] = ((r * 8 + c) & 0x7F) << 9, index bits 4, primary index p = p % 8.
    const uint32_t prec6[4] = {8, 7, 7, 7};
    const uint32_t zero_ep[3][8] = {};
    const uint32_t zero_ix[16] = {};
    Record b0 = bc7_record(0, zero_ep, prec6, zero_ix, nullptr, false);
    b0[2] = 1;  // index reference: merged nibbles (1, 0) = 1; endpoint delta 0 -> 1
    // Block 1: mode 1 (two subsets) with endpoints from block 0: its second subset predicts from
    // block 0's first; residual +1 on G0 of both subsets; indices fresh.
    const uint32_t prec1[4] = {7, 6, 6, 0};
    uint32_t ep1[3][8] = {};
    ep1[0][2] = ep1[1][2] = 1;
    uint32_t ix1[16];
    for (uint32_t p = 0; p < 16; ++p) {
        ix1[p] = (p == 0 || p == 2) ? 0 : p % 8;
    }
    Record b1 = bc7_record(1, ep1, prec1, ix1, nullptr, false);
    b1[1] = 1;  // odd nibble bit 0: merged delta symbol 2 = +1 -> endpoint reference 1 (block 0)
    const std::vector<uint8_t> dst = decode_records(*t, 7, {b0, b1});
    const Bc7Fields f0 = read_bc7(dst.data(), {0});
    CHECK(f0.mode == 6);
    // Dictionary row 0: column c holds c << 9 as a 16-bit left-aligned value: R (8 bits) = c << 1.
    CHECK(f0.ep[0][0][0] == 0 && f0.ep[0][1][0] == 1 && f0.pbit[0][1] == 0);  // R1 = 2: field 1, p 0
    CHECK(f0.ep[0][0][1] == 2 && f0.ep[0][1][1] == 3);                         // G (7 bits) = c
    for (uint32_t p = 0; p < 16; ++p) {
        // Dictionary index p % 8 at 4 bits, re-read at 4 bits (anchor 0 holds 0).
        CHECK(f0.index[0][p] == p % 8);
    }
    const Bc7Fields f1 = read_bc7(dst.data() + 16, {0, 2});
    CHECK(f1.mode == 1 && f1.partition == 17);
    // Block 0's R0 = 0 at 8 bits -> mode 1 R (7 bits incl. p-bit) 0; G0 = 2 (7 bits) cut to 6
    // bits = 1, + 1 = 2; both subsets.
    CHECK(f1.ep[0][0][1] == 2 && f1.ep[1][0][1] == 2);
    CHECK(f1.ep[0][1][1] == 1 && f1.ep[1][1][1] == 1);  // G1 = 3 cut to 6 bits
    for (uint32_t p = 0; p < 16; ++p) {
        CHECK(f1.index[0][p] == ix1[p]);
    }
}

// ---------------------------------------------------------------------------
// BC6H
// ---------------------------------------------------------------------------

// Reads `n` bits at bit `at` of a 128-bit block (little-endian bit order).
uint32_t block_bits(const uint8_t* blk, uint32_t at, uint32_t n) {
    uint32_t v = 0;
    for (uint32_t k = 0; k < n; ++k) {
        v |= static_cast<uint32_t>((blk[(at + k) / 8] >> ((at + k) % 8)) & 1u) << k;
    }
    return v;
}

// A block record for a BC6H block with no references: endpoint fields f[endpoint w x y z][R G B]
// at the mode's precision, stored indices (0 .. 2^bits-1) and the partition (two-subset modes).
Record bc6_record(uint32_t mode, const uint32_t f[4][3], const uint32_t* idx, uint32_t partition) {
    static const uint8_t bits[15][4] = {{0, 0, 0, 0},  {10, 5, 5, 5}, {7, 6, 6, 6},   {11, 5, 4, 4},
                                        {11, 4, 5, 4}, {11, 4, 4, 5}, {9, 5, 5, 5},   {8, 6, 5, 5},
                                        {8, 5, 6, 5},  {8, 5, 5, 6},  {6, 6, 6, 6},   {10, 10, 10, 10},
                                        {11, 9, 9, 9}, {12, 8, 8, 8}, {16, 4, 4, 4}};
    const bool two = mode < 11;
    // Residual symbol: the working value (field, top bit flipped, left-aligned) taken as a signed
    // 16-bit number, zigzagged.
    auto symbol = [](uint32_t field, uint32_t sh) {
        const uint32_t ep = ((field << sh) ^ 0x8000u) & 0xFFFFu;
        const int32_t s = static_cast<int16_t>(ep >> sh);
        return static_cast<uint32_t>(s >= 0 ? 2 * s : -2 * s - 1) & 0xFFFFu;
    };
    auto split = [](uint32_t v, uint8_t* even, uint8_t* odd) {  // the inverse of merging two bytes
        uint32_t a = 0;
        uint32_t b = 0;
        for (uint32_t k = 0; k < 8; ++k) {
            a |= ((v >> (2 * k)) & 1u) << k;
            b |= ((v >> (2 * k + 1)) & 1u) << k;
        }
        *even = static_cast<uint8_t>(a);
        *odd = static_cast<uint8_t>(b);
    };
    Record r(32, 0);
    for (uint32_t c = 0; c < 3; ++c) {
        split(symbol(f[0][c], 16 - bits[mode][0]), &r[4 + 2 * c], &r[5 + 2 * c]);
        split(symbol(f[1][c], 16 - bits[mode][1 + c]), &r[10 + 2 * c], &r[11 + 2 * c]);
        if (two) {
            r[16 + 2 * c] = static_cast<uint8_t>(symbol(f[2][c], 16 - bits[mode][1 + c]));
            r[17 + 2 * c] = static_cast<uint8_t>(symbol(f[3][c], 16 - bits[mode][1 + c]));
        }
    }
    const int ib = mode < 11 ? 3 : 4;
    for (uint32_t k = 0; k < 8; ++k) {
        uint8_t half[2];  // zigzagged signed residuals of pixels 2k and 2k + 1
        for (uint32_t h = 0; h < 2; ++h) {
            half[h] = sy::zigzag(static_cast<int>(idx[2 * k + h]) - (1 << (ib - 1)));
        }
        // Odd bits carry pixel 2k in the first six bytes, even bits in the last two.
        const uint8_t even = k < 6 ? half[1] : half[0];
        const uint8_t odd = k < 6 ? half[0] : half[1];
        uint8_t byte = 0;
        for (uint32_t b = 0; b < 4; ++b) {
            byte = static_cast<uint8_t>(byte | (((even >> b) & 1u) << (2 * b)) | (((odd >> b) & 1u) << (2 * b + 1)));
        }
        r[22 + k] = byte;
    }
    r[30] = static_cast<uint8_t>(mode - 1);
    r[31] = static_cast<uint8_t>(partition);
    return r;
}

void test_bc6h() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    // Mode 11 (one subset, 10-bit endpoints, 4-bit indices): mode bits 00011, then rw gw bw rx gx
    // bx at 10 bits each from bit 5, then the indices from bit 65 (the first one in 3 bits).
    {
        const uint32_t f[4][3] = {{0x155, 0x2AA, 0x0F0}, {0x3FF, 0, 0x201}, {}, {}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = p == 0 ? 5 : (p * 5 + 1) % 16;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 6, {bc6_record(11, f, ix, 0)});
        CHECK(block_bits(dst.data(), 0, 5) == 3);
        for (uint32_t k = 0; k < 6; ++k) {
            CHECK(block_bits(dst.data(), 5 + 10 * k, 10) == f[k / 3][k % 3]);
        }
        uint32_t at = 65;
        for (uint32_t p = 0; p < 16; ++p) {
            const uint32_t n = p == 0 ? 3 : 4;
            CHECK(block_bits(dst.data(), at, n) == (ix[p] & ((1u << n) - 1)));
            at += n;
        }
        CHECK(at == 128);
    }
    // Mode 1 (two subsets, 10/5/5/5 bits), partition 17 (second anchor at pixel 2): the scattered
    // header fields at their public positions, then 3-bit indices (2 bits at the anchors).
    {
        const uint32_t f[4][3] = {{0x155, 0x2AA, 0x0F0}, {17, 3, 30}, {1, 2, 3}, {31, 0, 16}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = (p == 0 || p == 2) ? p + 1 : (p * 3 + 2) % 8;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 6, {bc6_record(1, f, ix, 17)});
        const uint8_t* b = dst.data();
        auto bit = [&](uint32_t at) { return block_bits(b, at, 1); };
        CHECK(block_bits(b, 0, 2) == 0);
        CHECK(bit(2) == (f[2][1] >> 4) && bit(3) == (f[2][2] >> 4) && bit(4) == (f[3][2] >> 4));
        CHECK(block_bits(b, 5, 10) == f[0][0] && block_bits(b, 15, 10) == f[0][1] && block_bits(b, 25, 10) == f[0][2]);
        CHECK(block_bits(b, 35, 5) == f[1][0] && bit(40) == (f[3][1] >> 4) && block_bits(b, 41, 4) == (f[2][1] & 15));
        CHECK(block_bits(b, 45, 5) == f[1][1] && bit(50) == (f[3][2] & 1) && block_bits(b, 51, 4) == (f[3][1] & 15));
        CHECK(block_bits(b, 55, 5) == f[1][2] && bit(60) == ((f[3][2] >> 1) & 1) &&
              block_bits(b, 61, 4) == (f[2][2] & 15));
        CHECK(block_bits(b, 65, 5) == f[2][0] && bit(70) == ((f[3][2] >> 2) & 1) && block_bits(b, 71, 5) == f[3][0]);
        CHECK(bit(76) == ((f[3][2] >> 3) & 1) && block_bits(b, 77, 5) == 17);
        uint32_t at = 82;
        for (uint32_t p = 0; p < 16; ++p) {
            const uint32_t n = (p == 0 || p == 2) ? 2 : 3;
            CHECK(block_bits(b, at, n) == ix[p]);
            at += n;
        }
        CHECK(at == 128);
    }
    // Mode 14 (one subset, 16-bit base, 4-bit deltas): the base's high bits follow the delta.
    {
        const uint32_t f[4][3] = {{0xABCD, 0x1234, 0xFFFF}, {9, 15, 0}, {}, {}};
        uint32_t ix[16] = {};
        for (uint32_t p = 1; p < 16; ++p) {
            ix[p] = p;
        }
        const std::vector<uint8_t> dst = decode_records(*t, 6, {bc6_record(14, f, ix, 0)});
        const uint8_t* b = dst.data();
        CHECK(block_bits(b, 0, 5) == 15);
        CHECK(block_bits(b, 5, 10) == (f[0][0] & 0x3FF) && block_bits(b, 15, 10) == (f[0][1] & 0x3FF) &&
              block_bits(b, 25, 10) == (f[0][2] & 0x3FF));
        CHECK(block_bits(b, 35, 4) == f[1][0] && block_bits(b, 39, 6) == (f[0][0] >> 10));
        CHECK(block_bits(b, 45, 4) == f[1][1] && block_bits(b, 49, 6) == (f[0][1] >> 10));
        CHECK(block_bits(b, 55, 4) == f[1][2] && block_bits(b, 59, 6) == (f[0][2] >> 10));
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(block_bits(b, 65 + 4 * p - (p > 0 ? 1 : 0), p == 0 ? 3 : 4) == ix[p]);
        }
    }
    // References. Block 0 (mode 11) takes both references from dictionary entry 0 (reference 1);
    // block 1 (mode 1) predicts its endpoints from block 0 and adds +1 to the R of endpoint 1.
    // Dictionary entry 0: endpoint j channel c = (3 j + c) << 10, indices p % 8 - 4 at 4 bits.
    {
        const uint32_t zero_f[4][3] = {};
        const uint32_t zero_ix[16] = {};
        Record b0 = bc6_record(11, zero_f, zero_ix, 0);
        std::fill(b0.begin() + 4, b0.begin() + 30, uint8_t{0});  // no residuals
        b0[2] = 1;  // index reference 1 (merged nibbles 1, 0); endpoint reference = 1 + delta 0
        Record b1(32, 0);
        b1[1] = 1;   // endpoint reference: delta symbol 2 = +1 on index reference 0 -> block 0
        b1[11] = 1;  // R of endpoint 1: residual symbol 2 = +1 (odd byte's bit 0)
        b1[30] = 0;  // mode 1
        b1[31] = 17;
        const std::vector<uint8_t> dst = decode_records(*t, 6, {b0, b1});
        const uint8_t* b = dst.data();
        // Block 0: mode 11, 10-bit fields ((raw ^ 0x8000) >> 6) of the entry's endpoints 0 and 1,
        // where endpoint j channel c holds (3 j + c) << 10, i.e. k << 10 for field k.
        CHECK(block_bits(b, 0, 5) == 3);
        for (uint32_t k = 0; k < 6; ++k) {
            CHECK(block_bits(b, 5 + 10 * k, 10) == (((k << 10) ^ 0x8000u) >> 6));
        }
        // Dictionary index p % 8 - 4 at 4 bits, stored as p % 8 + 4 (3 bits at the anchor).
        uint32_t at = 65;
        for (uint32_t p = 0; p < 16; ++p) {
            const uint32_t n = p == 0 ? 3 : 4;
            CHECK(block_bits(b, at, n) == (((p % 8) + 4) & ((1u << n) - 1)));
            at += n;
        }
        // Block 1: mode 1. Endpoint 0 as block 0's; the others at 5 bits from the previous
        // block's endpoints (cut to the 5-bit field) plus the residual.
        const uint8_t* c = dst.data() + 16;
        CHECK(block_bits(c, 0, 2) == 0 && block_bits(c, 77, 5) == 17);
        CHECK(block_bits(c, 5, 10) == ((0u ^ 0x8000u) >> 6) && block_bits(c, 15, 10) == (((1u << 10) ^ 0x8000u) >> 6));
        auto five = [](uint32_t raw, uint32_t plus) {
            return ((((raw & 0xF800u) + (plus << 11)) & 0xFFFFu) ^ 0x8000u) >> 11;
        };
        CHECK(block_bits(c, 35, 5) == five(3u << 10, 1));  // rx
        CHECK(block_bits(c, 45, 5) == five(4u << 10, 0));  // gx
        CHECK(block_bits(c, 55, 5) == five(5u << 10, 0));  // bx
        // Block 0 (one subset) left ep2 = ep0 (re-expressed) and ep3 = ep1.
        CHECK(block_bits(c, 65, 5) == five(0u << 10, 0));  // ry
        CHECK(block_bits(c, 71, 5) == five(3u << 10, 0));  // rz
    }
    // Modes outside 1-14 are corrupt.
    for (uint32_t s30 : {14u, 15u, 255u}) {
        Record r(32, 0);
        r[30] = static_cast<uint8_t>(s30);
        bp::DecodeStatus st = bp::DecodeStatus::Ok;
        decode_records(*t, 6, {r}, &st);
        CHECK(st == bp::DecodeStatus::Corrupt);
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

void test_errors() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    std::vector<Record> records;
    for (uint32_t i = 0; i < 64; ++i) {
        Record r(12, 0);
        r[2] = static_cast<uint8_t>(i);
        r[3] = static_cast<uint8_t>(i * 3);
        for (uint32_t k = 4; k < 12; ++k) {
            r[k] = static_cast<uint8_t>(k + i);
        }
        records.push_back(r);
    }
    const std::vector<uint8_t> src = sy::stream(*t, spec_for(4), records);
    std::vector<uint8_t> dst(64 * 8);
    CHECK(bp::decode(*t, 4, src.data(), src.size(), dst.data(), dst.size()) == bp::DecodeStatus::Ok);
    CHECK(bp::decode(*t, 4, src.data(), src.size() / 2, dst.data(), dst.size()) == bp::DecodeStatus::Truncated);
    CHECK(bp::decode(*t, 4, nullptr, 0, dst.data(), dst.size()) == bp::DecodeStatus::Truncated);
    std::vector<uint8_t> more(65 * 8);
    CHECK(bp::decode(*t, 4, src.data(), src.size(), more.data(), more.size()) == bp::DecodeStatus::Truncated);
    CHECK(bp::decode(*t, 4, src.data(), src.size(), dst.data(), 12) == bp::DecodeStatus::BadSize);
    CHECK(bp::decode(*t, 4, src.data(), src.size(), dst.data(), 0) == bp::DecodeStatus::BadSize);
    CHECK(bp::decode(*t, 2, src.data(), src.size(), dst.data(), dst.size()) == bp::DecodeStatus::Unsupported);
    CHECK(bp::decode(*t, 2, src.data(), src.size(), dst.data(), dst.size()) == bp::DecodeStatus::Unsupported);
    // Fragment headers promising more payload than there is.
    const std::vector<uint8_t> headers(16, 0x30);
    CHECK(bp::decode(*t, 4, headers.data(), headers.size(), dst.data(), 8) == bp::DecodeStatus::Truncated);
    // BC7 control codes the table leaves unused (0) or makes inconsistent.
    for (uint32_t code : {6u, 5u, 511u}) {
        Record r(32, 0);
        r[30] = static_cast<uint8_t>(code & 31);
        r[31] = static_cast<uint8_t>(code >> 5);
        bp::DecodeStatus st = bp::DecodeStatus::Ok;
        decode_records(*t, 7, {r}, &st);
        CHECK(st == bp::DecodeStatus::Corrupt);
    }
    // Noise never crashes and never writes past the destination.
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    for (uint32_t round = 0; round < 200; ++round) {
        std::vector<uint8_t> noise(round * 7 % 900);
        for (uint8_t& b : noise) {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            b = static_cast<uint8_t>(seed);
        }
        for (uint32_t bc : {1u, 3u, 4u, 5u, 6u, 7u}) {
            std::vector<uint8_t> out(16 * (1 + round % 40) + 16, 0xA5);
            const size_t n = out.size() - 16;
            (void)bp::decode(*t, bc, noise.data(), noise.size(), out.data(), n);
            for (size_t k = n; k < out.size(); ++k) {
                CHECK(out[k] == 0xA5);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Extended tests: streaming multi-chunk, all BC7/BC6H modes, hardening, surface_blocks
// ---------------------------------------------------------------------------

void split_nibbles(uint8_t v, uint8_t* even, uint8_t* odd) {
    uint8_t e = 0;
    uint8_t o = 0;
    for (uint32_t k = 0; k < 4; ++k) {
        e = static_cast<uint8_t>(e | (((v >> (2 * k)) & 1u) << k));
        o = static_cast<uint8_t>(o | (((v >> (2 * k + 1)) & 1u) << k));
    }
    *even = e;
    *odd = o;
}

void set_refs(Record& r, bool four_symbols, uint8_t ep_ref, uint8_t ix_ref) {
    const int8_t delta = static_cast<int8_t>(static_cast<uint8_t>(ep_ref - ix_ref));
    const uint8_t z = sy::zigzag(delta);
    if (four_symbols) {
        split_nibbles(z, &r[0], &r[1]);
        split_nibbles(ix_ref, &r[2], &r[3]);
    } else {
        r[0] = z;
        r[1] = ix_ref;
    }
}

void test_streaming_multichunk_and_history_wrap() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    // 605 blocks: spans 4 full 128-block chunks + 93-block partial tail AND wraps the 256-block
    // sliding history ring more than twice.
    constexpr size_t kBlocks = 605;

    auto pair_symbols = [](const uint32_t* ix, uint8_t* out, size_t pairs) {
        for (size_t k = 0; k < pairs; ++k) {
            out[k] = static_cast<uint8_t>((ix[2 * k] & 7u) | ((ix[2 * k + 1] & 7u) << 3));
        }
    };

    // Helper: choose back-reference exercising chunk boundaries (128, 256, 384, 512) and full
    // 255-block history reach across ring wraps.
    auto pick_ref = [](size_t i, uint32_t salt) -> uint8_t {
        if (i == 0) {
            return 0;
        }
        if (i == 128 || i == 256 || i == 384 || i == 512) {
            return salt == 0 ? 1u : static_cast<uint8_t>(std::min<size_t>(i, 255));
        }
        if (i >= 256 && (i % 7 == 0)) {
            return 255;  // maximum sliding-window distance across ring wrap
        }
        if (i >= 128 && (i % 5 == 0)) {
            return 128;  // crosses chunk boundary
        }
        const size_t max_r = std::min<size_t>(i, 255);
        return static_cast<uint8_t>(1 + ((i * 13 + salt * 29) % max_r));
    };

    // --- BC1 (605 blocks) ---
    {
        struct State {
            uint32_t c0[3], c1[3], pos[16];
        };
        std::vector<State> truth(kBlocks);
        std::vector<Record> records(kBlocks, Record(12, 0));
        for (size_t i = 0; i < kBlocks; ++i) {
            const uint8_t ep_ref = pick_ref(i, 0);
            const uint8_t ix_ref = pick_ref(i, 1);
            set_refs(records[i], false, ep_ref, ix_ref);
            const State zero{};
            const State& er = ep_ref == 0 ? zero : truth[i - ep_ref];
            const State& ir = ix_ref == 0 ? zero : truth[i - ix_ref];
            uint32_t dpos[16];
            for (uint32_t c = 0; c < 3; ++c) {
                const uint32_t bits = c == 1 ? 6u : 5u;
                const uint32_t mask = (1u << bits) - 1u;
                const uint32_t d0 = static_cast<uint32_t>((i + c * 3 + 1) & mask);
                const uint32_t d1 = static_cast<uint32_t>((i * 2 + c + 3) & mask);
                records[i][2 + c] = residual(d0, bits);
                records[i][5 + c] = residual(d1, bits);
                truth[i].c0[c] = (er.c0[c] + d0) & mask;
                truth[i].c1[c] = (er.c1[c] + d1) & mask;
            }
            for (uint32_t p = 0; p < 16; ++p) {
                dpos[p] = static_cast<uint32_t>((i + p) & 3u);
                truth[i].pos[p] = (ir.pos[p] + dpos[p]) & 3u;
            }
            for (uint32_t q = 0; q < 4; ++q) {
                records[i][8 + q] = quad_symbol(dpos, q);
            }
        }
        const std::vector<uint8_t> dst = decode_records(*t, 1, records);
        for (size_t i = 0; i < kBlocks; ++i) {
            check_color_half(dst.data() + i * 8, truth[i].c0, truth[i].c1, truth[i].pos);
        }
    }

    // --- BC4 (605 blocks) ---
    {
        struct State {
            uint32_t e0, e1, pos[16];
        };
        std::vector<State> truth(kBlocks);
        std::vector<Record> records(kBlocks, Record(12, 0));
        for (size_t i = 0; i < kBlocks; ++i) {
            const uint8_t ep_ref = pick_ref(i, 2);
            const uint8_t ix_ref = pick_ref(i, 3);
            set_refs(records[i], false, ep_ref, ix_ref);
            const State zero{};
            const State& er = ep_ref == 0 ? zero : truth[i - ep_ref];
            const State& ir = ix_ref == 0 ? zero : truth[i - ix_ref];
            const uint32_t d0 = static_cast<uint32_t>((i * 7 + 5) & 0xFFu);
            const uint32_t d1 = static_cast<uint32_t>((i * 11 + 13) & 0xFFu);
            records[i][2] = residual(d0, 8);
            records[i][3] = residual(d1, 8);
            truth[i].e0 = (er.e0 + d0) & 0xFFu;
            truth[i].e1 = (er.e1 + d1) & 0xFFu;
            uint32_t dpos[16];
            for (uint32_t p = 0; p < 16; ++p) {
                dpos[p] = static_cast<uint32_t>((i + p * 3) & 7u);
                truth[i].pos[p] = (ir.pos[p] + dpos[p]) & 7u;
            }
            pair_symbols(dpos, &records[i][4], 8);
        }
        const std::vector<uint8_t> dst = decode_records(*t, 4, records);
        for (size_t i = 0; i < kBlocks; ++i) {
            check_alpha_half(dst.data() + i * 8, truth[i].e0, truth[i].e1, truth[i].pos);
        }
    }

    // --- BC3 (605 blocks) ---
    {
        struct State {
            uint32_t a0, a1, c0[3], c1[3], apos[16], cpos[16];
        };
        std::vector<State> truth(kBlocks);
        std::vector<Record> records(kBlocks, Record(24, 0));
        for (size_t i = 0; i < kBlocks; ++i) {
            const uint8_t ep_ref = pick_ref(i, 4);
            const uint8_t ix_ref = pick_ref(i, 5);
            set_refs(records[i], true, ep_ref, ix_ref);
            const State zero{};
            const State& er = ep_ref == 0 ? zero : truth[i - ep_ref];
            const State& ir = ix_ref == 0 ? zero : truth[i - ix_ref];
            const uint32_t da0 = static_cast<uint32_t>((i * 3 + 9) & 0xFFu);
            const uint32_t da1 = static_cast<uint32_t>((i * 5 + 1) & 0xFFu);
            records[i][4] = residual(da0, 8);
            records[i][5] = residual(da1, 8);
            truth[i].a0 = (er.a0 + da0) & 0xFFu;
            truth[i].a1 = (er.a1 + da1) & 0xFFu;
            uint32_t dapos[16], dcpos[16];
            for (uint32_t p = 0; p < 16; ++p) {
                dapos[p] = static_cast<uint32_t>((i + p) & 7u);
                dcpos[p] = static_cast<uint32_t>((i * 2 + p) & 3u);
                truth[i].apos[p] = (ir.apos[p] + dapos[p]) & 7u;
                truth[i].cpos[p] = (ir.cpos[p] + dcpos[p]) & 3u;
            }
            pair_symbols(dapos, &records[i][6], 8);
            for (uint32_t c = 0; c < 3; ++c) {
                const uint32_t bits = c == 1 ? 6u : 5u;
                const uint32_t mask = (1u << bits) - 1u;
                const uint32_t d0 = static_cast<uint32_t>((i + c * 5) & mask);
                const uint32_t d1 = static_cast<uint32_t>((i * 3 + c) & mask);
                records[i][14 + 2 * c] = residual(d0, bits);
                records[i][15 + 2 * c] = residual(d1, bits);
                truth[i].c0[c] = (er.c0[c] + d0) & mask;
                truth[i].c1[c] = (er.c1[c] + d1) & mask;
            }
            for (uint32_t q = 0; q < 4; ++q) {
                records[i][20 + q] = quad_symbol(dcpos, q);
            }
        }
        const std::vector<uint8_t> dst = decode_records(*t, 3, records);
        for (size_t i = 0; i < kBlocks; ++i) {
            check_alpha_half(dst.data() + i * 16, truth[i].a0, truth[i].a1, truth[i].apos);
            check_color_half(dst.data() + i * 16 + 8, truth[i].c0, truth[i].c1, truth[i].cpos);
        }
    }

    // --- BC5 (605 blocks) ---
    {
        struct State {
            uint32_t e[4], pos[32];
        };
        std::vector<State> truth(kBlocks);
        std::vector<Record> records(kBlocks, Record(24, 0));
        for (size_t i = 0; i < kBlocks; ++i) {
            const uint8_t ep_ref = pick_ref(i, 6);
            const uint8_t ix_ref = pick_ref(i, 7);
            set_refs(records[i], true, ep_ref, ix_ref);
            const State zero{};
            const State& er = ep_ref == 0 ? zero : truth[i - ep_ref];
            const State& ir = ix_ref == 0 ? zero : truth[i - ix_ref];
            for (uint32_t k = 0; k < 4; ++k) {
                const uint32_t d = static_cast<uint32_t>((i * (k + 3) + k * 19) & 0xFFu);
                records[i][4 + k] = residual(d, 8);
                truth[i].e[k] = (er.e[k] + d) & 0xFFu;
            }
            uint32_t dpos[32];
            for (uint32_t p = 0; p < 32; ++p) {
                dpos[p] = static_cast<uint32_t>((i + p * 3) & 7u);
                truth[i].pos[p] = (ir.pos[p] + dpos[p]) & 7u;
            }
            pair_symbols(dpos, &records[i][8], 16);
        }
        const std::vector<uint8_t> dst = decode_records(*t, 5, records);
        for (size_t i = 0; i < kBlocks; ++i) {
            check_alpha_half(dst.data() + i * 16, truth[i].e[0], truth[i].e[1], truth[i].pos);
            check_alpha_half(dst.data() + i * 16 + 8, truth[i].e[2], truth[i].e[3], truth[i].pos + 16);
        }
    }

    // --- BC6H (605 blocks, mode 11 with sliding history back-references across chunks & ring wrap) ---
    {
        struct State {
            uint32_t f[2][3];  // 10-bit endpoint fields
            uint32_t ix[16];   // 4-bit stored indices
        };
        std::vector<State> truth(kBlocks);
        std::vector<Record> records(kBlocks, Record(32, 0));
        for (size_t i = 0; i < kBlocks; ++i) {
            if (i < 4 || (i % 3 != 0)) {
                // Fresh block (ref = 0) with unique pattern
                uint32_t f[4][3] = {};
                for (uint32_t e = 0; e < 2; ++e) {
                    for (uint32_t c = 0; c < 3; ++c) {
                        f[e][c] = static_cast<uint32_t>((i * 17 + e * 101 + c * 37) & 0x3FFu);
                        truth[i].f[e][c] = f[e][c];
                    }
                }
                uint32_t ix[16];
                for (uint32_t p = 0; p < 16; ++p) {
                    ix[p] = p == 0 ? static_cast<uint32_t>(i & 7u) : static_cast<uint32_t>((i + p * 5) & 15u);
                    truth[i].ix[p] = ix[p];
                }
                records[i] = bc6_record(11, f, ix, 0);
            } else {
                // Pure copy from an earlier block across chunk / ring boundary (zero residuals)
                const uint8_t ep_ref = pick_ref(i, 8);
                const uint8_t ix_ref = pick_ref(i, 9);
                records[i][30] = 10;  // mode 11
                set_refs(records[i], true, ep_ref, ix_ref);
                const State& er = truth[i - ep_ref];
                const State& ir = truth[i - ix_ref];
                for (uint32_t e = 0; e < 2; ++e) {
                    for (uint32_t c = 0; c < 3; ++c) {
                        truth[i].f[e][c] = er.f[e][c];
                    }
                }
                for (uint32_t p = 0; p < 16; ++p) {
                    truth[i].ix[p] = ir.ix[p];
                }
            }
        }
        const std::vector<uint8_t> dst = decode_records(*t, 6, records);
        for (size_t i = 0; i < kBlocks; ++i) {
            const uint8_t* blk = dst.data() + i * 16;
            CHECK(block_bits(blk, 0, 5) == 3);
            for (uint32_t k = 0; k < 6; ++k) {
                CHECK(block_bits(blk, 5 + 10 * k, 10) == truth[i].f[k / 3][k % 3]);
            }
            uint32_t at = 65;
            for (uint32_t p = 0; p < 16; ++p) {
                const uint32_t n = p == 0 ? 3u : 4u;
                CHECK(block_bits(blk, at, n) == (truth[i].ix[p] & ((1u << n) - 1u)));
                at += n;
            }
        }
    }

    // --- BC7 (605 blocks, mode 6 with sliding history back-references across chunks & ring wrap) ---
    {
        struct State {
            uint32_t ep[8];  // R0 R1 (8-bit incl pbit), G0 G1 B0 B1 A0 A1 (7-bit)
            uint32_t ix[16];
        };
        const uint32_t prec6[4] = {8, 7, 7, 7};
        std::vector<State> truth(kBlocks);
        std::vector<Record> records(kBlocks, Record(32, 0));
        for (size_t i = 0; i < kBlocks; ++i) {
            if (i < 4 || (i % 3 != 0)) {
                uint32_t ep[3][8] = {};
                for (uint32_t col = 0; col < 8; ++col) {
                    const uint32_t mask = (1u << prec6[col / 2]) - 1u;
                    ep[0][col] = static_cast<uint32_t>((i * 19 + col * 11 + 3) & mask);
                    truth[i].ep[col] = ep[0][col];
                }
                uint32_t ix[16];
                for (uint32_t p = 0; p < 16; ++p) {
                    ix[p] = p == 0 ? static_cast<uint32_t>(i & 7u) : static_cast<uint32_t>((i + p * 3) & 15u);
                    truth[i].ix[p] = ix[p];
                }
                records[i] = bc7_record(0, ep, prec6, ix, nullptr, false);
            } else {
                const uint8_t ep_ref = pick_ref(i, 10);
                const uint8_t ix_ref = pick_ref(i, 11);
                // Mode 6 is code 0 (s30 = 0, s31 = 0), zero residuals copies references identically.
                set_refs(records[i], true, ep_ref, ix_ref);
                const State& er = truth[i - ep_ref];
                const State& ir = truth[i - ix_ref];
                for (uint32_t col = 0; col < 8; ++col) {
                    truth[i].ep[col] = er.ep[col];
                }
                for (uint32_t p = 0; p < 16; ++p) {
                    truth[i].ix[p] = ir.ix[p];
                }
            }
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, records);
        for (size_t i = 0; i < kBlocks; ++i) {
            const Bc7Fields f = read_bc7(dst.data() + i * 16, {0});
            CHECK(f.mode == 6 && f.bits == 128);
            CHECK(f.ep[0][0][0] == (truth[i].ep[0] >> 1) && f.pbit[0][0] == (truth[i].ep[0] & 1u));
            CHECK(f.ep[0][1][0] == (truth[i].ep[1] >> 1) && f.pbit[0][1] == (truth[i].ep[1] & 1u));
            CHECK(f.ep[0][0][1] == truth[i].ep[2] && f.ep[0][1][1] == truth[i].ep[3]);
            CHECK(f.ep[0][0][2] == truth[i].ep[4] && f.ep[0][1][2] == truth[i].ep[5]);
            CHECK(f.ep[0][0][3] == truth[i].ep[6] && f.ep[0][1][3] == truth[i].ep[7]);
            for (uint32_t p = 0; p < 16; ++p) {
                CHECK(f.index[0][p] == (truth[i].ix[p] & (p == 0 ? 7u : 15u)));
            }
        }
    }
}

void test_bc7_all_modes() {
    sy::Builder b = sy::Builder::full();
    std::vector<uint16_t> ctrl(512, 0);
    // Cover every BC7 mode 0..7, all rotations 0..3 on modes 4 and 5, both index_select on mode 4.
    ctrl[0] = 0 | (3 << 6) | (11 << 8);   // mode 0, 3 subsets, partition 11 (anchors 0, 6, 15)
    ctrl[1] = 1 | (2 << 6) | (34 << 8);   // mode 1, 2 subsets, partition 34 (anchors 0, 6)
    ctrl[2] = 2 | (3 << 6) | (42 << 8);   // mode 2, 3 subsets, partition 42 (anchors 0, 15, 8)
    ctrl[3] = 3 | (2 << 6) | (50 << 8);   // mode 3, 2 subsets, partition 50 (anchors 0, 6)
    ctrl[6] = 6 | (1 << 6);               // mode 6, 1 subset
    ctrl[7] = 7 | (2 << 6) | (61 << 8);   // mode 7, 2 subsets, partition 61 (anchors 0, 2)
    for (uint32_t rot = 0; rot < 4; ++rot) {
        for (uint32_t isel = 0; isel < 2; ++isel) {
            ctrl[16 + rot * 2 + isel] = static_cast<uint16_t>(4 | (rot << 3) | (isel << 5) | (1 << 6));
        }
        ctrl[32 + rot] = static_cast<uint16_t>(5 | (rot << 3) | (1 << 6));
    }
    std::vector<uint8_t>& bc7c = b.sections["BC7C"];
    bc7c.clear();
    for (uint16_t w : ctrl) {
        sy::put16(bc7c, w);
    }
    const auto t = b.tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }

    // Mode 0 (3 subsets, partition 11: anchors 0, 6, 15; per-endpoint p-bits)
    {
        const uint32_t prec[4] = {5, 4, 4, 0};
        const uint32_t ep[3][8] = {
            {0x13, 0x0A, 1, 2, 3, 4, 0, 0}, {0x1E, 0x05, 5, 6, 7, 8, 0, 0}, {0x1F, 0x10, 9, 10, 11, 12, 0, 0}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = (p == 0 || p == 6 || p == 15) ? (p % 4) : ((p * 5 + 1) % 8);
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(0, ep, prec, ix, nullptr, true)});
        const Bc7Fields f = read_bc7(dst.data(), {0, 6, 15});
        CHECK(f.mode == 0 && f.partition == 11 && f.bits == 128);
        for (uint32_t s = 0; s < 3; ++s) {
            CHECK(f.ep[s][0][0] == (ep[s][0] >> 1) && f.pbit[s][0] == (ep[s][0] & 1u));
            CHECK(f.ep[s][1][0] == (ep[s][1] >> 1) && f.pbit[s][1] == (ep[s][1] & 1u));
            CHECK(f.ep[s][0][1] == ep[s][2] && f.ep[s][1][1] == ep[s][3]);
            CHECK(f.ep[s][0][2] == ep[s][4] && f.ep[s][1][2] == ep[s][5]);
        }
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix[p]);
        }
    }

    // Mode 1 (2 subsets, partition 34: anchors 0, 6; shared per-subset p-bits)
    {
        const uint32_t prec[4] = {7, 6, 6, 0};
        const uint32_t ep[3][8] = {{0x7B, 0x24, 11, 22, 33, 44, 0, 0}, {0x00, 0x7E, 55, 63, 7, 19, 0, 0}, {}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = (p == 0 || p == 6) ? (p == 0 ? 2 : 3) : ((p + 5) % 8);
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(1, ep, prec, ix, nullptr, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0, 6});
        CHECK(f.mode == 1 && f.partition == 34 && f.bits == 128);
        for (uint32_t s = 0; s < 2; ++s) {
            CHECK(f.ep[s][0][0] == (ep[s][0] >> 1) && f.ep[s][1][0] == (ep[s][1] >> 1));
            CHECK(f.pbit[s][0] == (ep[s][0] & 1u));
            CHECK(f.ep[s][0][1] == ep[s][2] && f.ep[s][1][1] == ep[s][3]);
            CHECK(f.ep[s][0][2] == ep[s][4] && f.ep[s][1][2] == ep[s][5]);
        }
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix[p]);
        }
    }

    // Mode 2 (3 subsets, partition 42: anchors 0, 15, 8; no p-bits, 5-bit RGB, 2-bit indices)
    {
        const uint32_t prec[4] = {5, 5, 5, 0};
        const uint32_t ep[3][8] = {
            {3, 19, 7, 29, 11, 31, 0, 0}, {1, 2, 3, 4, 5, 6, 0, 0}, {30, 28, 26, 24, 22, 20, 0, 0}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = (p == 0 || p == 15 || p == 8) ? (p & 1u) : ((p * 3 + 1) & 3u);
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(2, ep, prec, ix, nullptr, true)});
        const Bc7Fields f = read_bc7(dst.data(), {0, 15, 8});
        CHECK(f.mode == 2 && f.partition == 42 && f.bits == 128);
        for (uint32_t s = 0; s < 3; ++s) {
            for (uint32_t ch = 0; ch < 3; ++ch) {
                CHECK(f.ep[s][0][ch] == ep[s][2 * ch]);
                CHECK(f.ep[s][1][ch] == ep[s][2 * ch + 1]);
            }
        }
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix[p]);
        }
    }

    // Mode 3 (2 subsets, partition 50: anchors 0, 6; 7-bit RGB + per-endpoint p-bits, 2-bit indices)
    {
        const uint32_t prec[4] = {8, 7, 7, 0};
        const uint32_t ep[3][8] = {{0xAA, 0x55, 10, 20, 30, 40, 0, 0}, {0xFF, 0x01, 100, 101, 126, 127, 0, 0}, {}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = (p == 0 || p == 6) ? (p == 0 ? 1 : 0) : ((p + 2) & 3u);
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(3, ep, prec, ix, nullptr, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0, 6});
        CHECK(f.mode == 3 && f.partition == 50 && f.bits == 128);
        for (uint32_t s = 0; s < 2; ++s) {
            CHECK(f.ep[s][0][0] == (ep[s][0] >> 1) && f.pbit[s][0] == (ep[s][0] & 1u));
            CHECK(f.ep[s][1][0] == (ep[s][1] >> 1) && f.pbit[s][1] == (ep[s][1] & 1u));
            CHECK(f.ep[s][0][1] == ep[s][2] && f.ep[s][1][1] == ep[s][3]);
            CHECK(f.ep[s][0][2] == ep[s][4] && f.ep[s][1][2] == ep[s][5]);
        }
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix[p]);
        }
    }

    // Mode 4: all 4 rotations x both index_select values
    for (uint32_t rot = 0; rot < 4; ++rot) {
        for (uint32_t isel = 0; isel < 2; ++isel) {
            uint32_t prec[4] = {5, 5, 5, 6};
            if (rot != 0) {
                std::swap(prec[3], prec[rot - 1]);
            }
            uint32_t ep[3][8] = {};
            for (uint32_t col = 0; col < 8; ++col) {
                ep[0][col] = (17 + col * 5 + rot * 3 + isel) & ((1u << prec[col / 2]) - 1u);
            }
            uint32_t ix0[16], ix1[16];
            for (uint32_t p = 0; p < 16; ++p) {
                ix0[p] = p == 0 ? ((rot + isel) & 3u) : ((p * 3 + rot) & 7u);
                ix1[p] = p == 0 ? (isel & 1u) : ((p + rot) & 3u);
            }
            const uint32_t code = 16 + rot * 2 + isel;
            const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(code, ep, prec, ix0, ix1, false)});
            const Bc7Fields f = read_bc7(dst.data(), {0});
            CHECK(f.mode == 4 && f.rotation == rot && f.index_select == isel && f.bits == 128);
            uint32_t order[4] = {0, 1, 2, 3};
            if (rot != 0) {
                std::swap(order[3], order[rot - 1]);
            }
            for (uint32_t ch = 0; ch < 4; ++ch) {
                CHECK(f.ep[0][0][ch] == ep[0][2 * order[ch]]);
                CHECK(f.ep[0][1][ch] == ep[0][2 * order[ch] + 1]);
            }
            for (uint32_t p = 0; p < 16; ++p) {
                CHECK(f.index[0][p] == ix1[p]);  // 2-bit set first
                CHECK(f.index[1][p] == ix0[p]);  // 3-bit set second
            }
        }
    }

    // Mode 5: all 4 rotations
    for (uint32_t rot = 0; rot < 4; ++rot) {
        uint32_t prec[4] = {7, 7, 7, 8};
        if (rot != 0) {
            std::swap(prec[3], prec[rot - 1]);
        }
        uint32_t ep[3][8] = {};
        for (uint32_t col = 0; col < 8; ++col) {
            ep[0][col] = (41 + col * 23 + rot * 7) & ((1u << prec[col / 2]) - 1u);
        }
        uint32_t ix0[16], ix1[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix0[p] = p == 0 ? (rot & 1u) : ((p * 3 + rot) & 3u);
            ix1[p] = p == 0 ? ((rot >> 1) & 1u) : ((p + rot + 1) & 3u);
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(32 + rot, ep, prec, ix0, ix1, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0});
        CHECK(f.mode == 5 && f.rotation == rot && f.bits == 128);
        uint32_t order[4] = {0, 1, 2, 3};
        if (rot != 0) {
            std::swap(order[3], order[rot - 1]);
        }
        for (uint32_t ch = 0; ch < 4; ++ch) {
            CHECK(f.ep[0][0][ch] == ep[0][2 * order[ch]]);
            CHECK(f.ep[0][1][ch] == ep[0][2 * order[ch] + 1]);
        }
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix0[p]);
            CHECK(f.index[1][p] == ix1[p]);
        }
    }

    // Mode 7 (2 subsets, partition 61: anchors 0, 2; 5-bit RGBA + per-endpoint p-bits, 2-bit indices)
    {
        const uint32_t prec[4] = {6, 5, 5, 5};
        const uint32_t ep[3][8] = {
            {0x3D, 0x12, 3, 29, 7, 25, 11, 21}, {0x01, 0x3E, 31, 0, 15, 16, 8, 24}, {}};
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            ix[p] = (p == 0 || p == 2) ? (p == 0 ? 1 : 0) : ((p * 3) & 3u);
        }
        const std::vector<uint8_t> dst = decode_records(*t, 7, {bc7_record(7, ep, prec, ix, nullptr, false)});
        const Bc7Fields f = read_bc7(dst.data(), {0, 2});
        CHECK(f.mode == 7 && f.partition == 61 && f.bits == 128);
        for (uint32_t s = 0; s < 2; ++s) {
            CHECK(f.ep[s][0][0] == (ep[s][0] >> 1) && f.pbit[s][0] == (ep[s][0] & 1u));
            CHECK(f.ep[s][1][0] == (ep[s][1] >> 1) && f.pbit[s][1] == (ep[s][1] & 1u));
            CHECK(f.ep[s][0][1] == ep[s][2] && f.ep[s][1][1] == ep[s][3]);
            CHECK(f.ep[s][0][2] == ep[s][4] && f.ep[s][1][2] == ep[s][5]);
            CHECK(f.ep[s][0][3] == ep[s][6] && f.ep[s][1][3] == ep[s][7]);
        }
        for (uint32_t p = 0; p < 16; ++p) {
            CHECK(f.index[0][p] == ix[p]);
        }
    }
}

void test_bc6h_all_modes() {
    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }
    static const uint8_t kBits[15][4] = {{0, 0, 0, 0},  {10, 5, 5, 5}, {7, 6, 6, 6},   {11, 5, 4, 4},
                                         {11, 4, 5, 4}, {11, 4, 4, 5}, {9, 5, 5, 5},   {8, 6, 5, 5},
                                         {8, 5, 6, 5},  {8, 5, 5, 6},  {6, 6, 6, 6},   {10, 10, 10, 10},
                                         {11, 9, 9, 9}, {12, 8, 8, 8}, {16, 4, 4, 4}};
    static const uint8_t kModeCode[15][2] = {{0, 0},  {0, 2},  {1, 2},  {2, 5},  {6, 5},  {10, 5}, {14, 5}, {18, 5},
                                             {22, 5}, {26, 5}, {30, 5}, {3, 5},  {7, 5},  {11, 5}, {15, 5}};
    // Public BC6H second-subset anchor indices for partitions 0..31.
    static const uint8_t kAnchor2[32] = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
                                         15, 2,  8,  2,  2,  8,  8,  15, 2,  8,  2,  2,  8,  8,  2,  2};

    // Unpacks a 128-bit BC6H block according to the public spec for `mode` (1..14) and verifies
    // every endpoint field, partition, and index bit, plus total width == 128.
    auto check_mode = [&](uint32_t mode) {
        const bool two = mode < 11;
        uint32_t ep[4][3] = {};
        for (uint32_t e = 0; e < (two ? 4u : 2u); ++e) {
            for (uint32_t c = 0; c < 3; ++c) {
                const uint32_t nbits = e == 0 ? kBits[mode][0] : kBits[mode][1 + c];
                const uint32_t mask = nbits == 16 ? 0xFFFFu : ((1u << nbits) - 1u);
                // Distinct bit pattern exercising both LSBs and MSBs
                ep[e][c] = (0xA53Cu ^ (mode * 0x1111u) ^ (e * 0x05A3u) ^ (c * 0x0137u)) & mask;
            }
        }
        const uint32_t partition = two ? ((mode * 3 + 5) & 31u) : 0u;
        const uint32_t anchor2 = two ? kAnchor2[partition] : 0u;
        const uint32_t ibits = two ? 3u : 4u;
        uint32_t ix[16];
        for (uint32_t p = 0; p < 16; ++p) {
            const bool is_anchor = (p == 0) || (two && p == anchor2);
            const uint32_t mod = 1u << (ibits - (is_anchor ? 1u : 0u));
            ix[p] = (p * 5 + mode) % mod;
        }

        const std::vector<uint8_t> dst = decode_records(*t, 6, {bc6_record(mode, ep, ix, partition)});
        BitReader r(dst.data());
        CHECK(r.get(kModeCode[mode][1]) == kModeCode[mode][0]);

        uint32_t got[4][3] = {};
        uint32_t got_part = 0;
        auto read_bits = [&](uint32_t e, uint32_t c, uint32_t lo, uint32_t count) {
            got[e][c] |= r.get(count) << lo;
        };
        auto read_bit = [&](uint32_t e, uint32_t c, uint32_t bit) { read_bits(e, c, bit, 1); };
        switch (mode) {
            case 1:
                read_bit(2, 1, 4); read_bit(2, 2, 4); read_bit(3, 2, 4);
                read_bits(0, 0, 0, 10); read_bits(0, 1, 0, 10); read_bits(0, 2, 0, 10);
                read_bits(1, 0, 0, 5); read_bit(3, 1, 4); read_bits(2, 1, 0, 4);
                read_bits(1, 1, 0, 5); read_bit(3, 2, 0); read_bits(3, 1, 0, 4);
                read_bits(1, 2, 0, 5); read_bit(3, 2, 1); read_bits(2, 2, 0, 4);
                read_bits(2, 0, 0, 5); read_bit(3, 2, 2); read_bits(3, 0, 0, 5);
                read_bit(3, 2, 3); got_part = r.get(5);
                break;
            case 2:
                read_bit(2, 1, 5); read_bit(3, 1, 4); read_bit(3, 1, 5);
                read_bits(0, 0, 0, 7); read_bit(3, 2, 0); read_bit(3, 2, 1); read_bit(2, 2, 4);
                read_bits(0, 1, 0, 7); read_bit(2, 2, 5); read_bit(3, 2, 2); read_bit(2, 1, 4);
                read_bits(0, 2, 0, 7); read_bit(3, 2, 3); read_bit(3, 2, 5); read_bit(3, 2, 4);
                read_bits(1, 0, 0, 6); read_bits(2, 1, 0, 4); read_bits(1, 1, 0, 6);
                read_bits(3, 1, 0, 4); read_bits(1, 2, 0, 6); read_bits(2, 2, 0, 4);
                read_bits(2, 0, 0, 6); read_bits(3, 0, 0, 6); got_part = r.get(5);
                break;
            case 3:
                read_bits(0, 0, 0, 10); read_bits(0, 1, 0, 10); read_bits(0, 2, 0, 10);
                read_bits(1, 0, 0, 5); read_bit(0, 0, 10); read_bits(2, 1, 0, 4);
                read_bits(1, 1, 0, 4); read_bit(0, 1, 10); read_bit(3, 2, 0); read_bits(3, 1, 0, 4);
                read_bits(1, 2, 0, 4); read_bit(0, 2, 10); read_bit(3, 2, 1); read_bits(2, 2, 0, 4);
                read_bits(2, 0, 0, 5); read_bit(3, 2, 2); read_bits(3, 0, 0, 5);
                read_bit(3, 2, 3); got_part = r.get(5);
                break;
            case 4:
                read_bits(0, 0, 0, 10); read_bits(0, 1, 0, 10); read_bits(0, 2, 0, 10);
                read_bits(1, 0, 0, 4); read_bit(0, 0, 10); read_bit(3, 1, 4); read_bits(2, 1, 0, 4);
                read_bits(1, 1, 0, 5); read_bit(0, 1, 10); read_bits(3, 1, 0, 4);
                read_bits(1, 2, 0, 4); read_bit(0, 2, 10); read_bit(3, 2, 1); read_bits(2, 2, 0, 4);
                read_bits(2, 0, 0, 4); read_bit(3, 2, 0); read_bit(3, 2, 2);
                read_bits(3, 0, 0, 4); read_bit(2, 1, 4); read_bit(3, 2, 3); got_part = r.get(5);
                break;
            case 5:
                read_bits(0, 0, 0, 10); read_bits(0, 1, 0, 10); read_bits(0, 2, 0, 10);
                read_bits(1, 0, 0, 4); read_bit(0, 0, 10); read_bit(2, 2, 4); read_bits(2, 1, 0, 4);
                read_bits(1, 1, 0, 4); read_bit(0, 1, 10); read_bit(3, 2, 0); read_bits(3, 1, 0, 4);
                read_bits(1, 2, 0, 5); read_bit(0, 2, 10); read_bits(2, 2, 0, 4);
                read_bits(2, 0, 0, 4); read_bit(3, 2, 1); read_bit(3, 2, 2);
                read_bits(3, 0, 0, 4); read_bit(3, 2, 4); read_bit(3, 2, 3); got_part = r.get(5);
                break;
            case 6:
                read_bits(0, 0, 0, 9); read_bit(2, 2, 4); read_bits(0, 1, 0, 9); read_bit(2, 1, 4);
                read_bits(0, 2, 0, 9); read_bit(3, 2, 4);
                read_bits(1, 0, 0, 5); read_bit(3, 1, 4); read_bits(2, 1, 0, 4);
                read_bits(1, 1, 0, 5); read_bit(3, 2, 0); read_bits(3, 1, 0, 4);
                read_bits(1, 2, 0, 5); read_bit(3, 2, 1); read_bits(2, 2, 0, 4);
                read_bits(2, 0, 0, 5); read_bit(3, 2, 2); read_bits(3, 0, 0, 5);
                read_bit(3, 2, 3); got_part = r.get(5);
                break;
            case 7:
                read_bits(0, 0, 0, 8); read_bit(3, 1, 4); read_bit(2, 2, 4);
                read_bits(0, 1, 0, 8); read_bit(3, 2, 2); read_bit(2, 1, 4);
                read_bits(0, 2, 0, 8); read_bit(3, 2, 3); read_bit(3, 2, 4);
                read_bits(1, 0, 0, 6); read_bits(2, 1, 0, 4);
                read_bits(1, 1, 0, 5); read_bit(3, 2, 0); read_bits(3, 1, 0, 4);
                read_bits(1, 2, 0, 5); read_bit(3, 2, 1); read_bits(2, 2, 0, 4);
                read_bits(2, 0, 0, 6); read_bits(3, 0, 0, 6); got_part = r.get(5);
                break;
            case 8:
                read_bits(0, 0, 0, 8); read_bit(3, 2, 0); read_bit(2, 2, 4);
                read_bits(0, 1, 0, 8); read_bit(2, 1, 5); read_bit(2, 1, 4);
                read_bits(0, 2, 0, 8); read_bit(3, 1, 5); read_bit(3, 2, 4);
                read_bits(1, 0, 0, 5); read_bit(3, 1, 4); read_bits(2, 1, 0, 4);
                read_bits(1, 1, 0, 6); read_bits(3, 1, 0, 4);
                read_bits(1, 2, 0, 5); read_bit(3, 2, 1); read_bits(2, 2, 0, 4);
                read_bits(2, 0, 0, 5); read_bit(3, 2, 2); read_bits(3, 0, 0, 5);
                read_bit(3, 2, 3); got_part = r.get(5);
                break;
            case 9:
                read_bits(0, 0, 0, 8); read_bit(3, 2, 1); read_bit(2, 2, 4);
                read_bits(0, 1, 0, 8); read_bit(2, 2, 5); read_bit(2, 1, 4);
                read_bits(0, 2, 0, 8); read_bit(3, 2, 5); read_bit(3, 2, 4);
                read_bits(1, 0, 0, 5); read_bit(3, 1, 4); read_bits(2, 1, 0, 4);
                read_bits(1, 1, 0, 5); read_bit(3, 2, 0); read_bits(3, 1, 0, 4);
                read_bits(1, 2, 0, 6); read_bits(2, 2, 0, 4);
                read_bits(2, 0, 0, 5); read_bit(3, 2, 2); read_bits(3, 0, 0, 5);
                read_bit(3, 2, 3); got_part = r.get(5);
                break;
            case 10:
                read_bits(0, 0, 0, 6); read_bit(3, 1, 4); read_bit(3, 2, 0); read_bit(3, 2, 1); read_bit(2, 2, 4);
                read_bits(0, 1, 0, 6); read_bit(2, 1, 5); read_bit(2, 2, 5); read_bit(3, 2, 2); read_bit(2, 1, 4);
                read_bits(0, 2, 0, 6); read_bit(3, 1, 5); read_bit(3, 2, 3); read_bit(3, 2, 5); read_bit(3, 2, 4);
                read_bits(1, 0, 0, 6); read_bits(2, 1, 0, 4); read_bits(1, 1, 0, 6);
                read_bits(3, 1, 0, 4); read_bits(1, 2, 0, 6); read_bits(2, 2, 0, 4);
                read_bits(2, 0, 0, 6); read_bits(3, 0, 0, 6); got_part = r.get(5);
                break;
            case 11:
                read_bits(0, 0, 0, 10); read_bits(0, 1, 0, 10); read_bits(0, 2, 0, 10);
                read_bits(1, 0, 0, 10); read_bits(1, 1, 0, 10); read_bits(1, 2, 0, 10);
                break;
            case 12:
                read_bits(0, 0, 0, 10); read_bits(0, 1, 0, 10); read_bits(0, 2, 0, 10);
                read_bits(1, 0, 0, 9); read_bit(0, 0, 10);
                read_bits(1, 1, 0, 9); read_bit(0, 1, 10);
                read_bits(1, 2, 0, 9); read_bit(0, 2, 10);
                break;
            case 13:
                read_bits(0, 0, 0, 10); read_bits(0, 1, 0, 10); read_bits(0, 2, 0, 10);
                read_bits(1, 0, 0, 8); read_bits(0, 0, 10, 2);
                read_bits(1, 1, 0, 8); read_bits(0, 1, 10, 2);
                read_bits(1, 2, 0, 8); read_bits(0, 2, 10, 2);
                break;
            case 14:
                read_bits(0, 0, 0, 10); read_bits(0, 1, 0, 10); read_bits(0, 2, 0, 10);
                read_bits(1, 0, 0, 4); read_bits(0, 0, 10, 6);
                read_bits(1, 1, 0, 4); read_bits(0, 1, 10, 6);
                read_bits(1, 2, 0, 4); read_bits(0, 2, 10, 6);
                break;
        }
        for (uint32_t e = 0; e < (two ? 4u : 2u); ++e) {
            for (uint32_t c = 0; c < 3; ++c) {
                CHECK(got[e][c] == ep[e][c]);
            }
        }
        if (two) {
            CHECK(got_part == partition);
        }
        for (uint32_t p = 0; p < 16; ++p) {
            const bool is_anchor = (p == 0) || (two && p == anchor2);
            CHECK(r.get(ibits - (is_anchor ? 1u : 0u)) == ix[p]);
        }
        CHECK(r.at() == 128);
    };

    for (uint32_t mode = 1; mode <= 14; ++mode) {
        check_mode(mode);
    }
}

void test_hardening_and_fast_div() {
    // 1. Exactness verification of multiply-shift division across boundary frequencies.
    for (uint32_t freq : {1u, 2u, 3u, 7u, 17u, 255u, 1024u, 32767u, 65535u}) {
        const uint64_t recip = ((uint64_t{1} << 32) + freq - 1u) / freq;
        for (uint32_t into = 0; into <= 65535u; ++into) {
            const uint32_t fast_q = static_cast<uint32_t>((static_cast<uint64_t>(into) * recip) >> 32);
            const uint32_t exact_q = into / freq;
            if (fast_q != exact_q) {
                CHECK(fast_q == exact_q);
                break;
            }
        }
    }

    // 2. End-to-end rANS round-trip with boundary bucket frequencies (including freq=1,2,3,7,17,
    //    255,1024,32767,65535) directly through parse_tables + decode_lanes.
    {
        sy::Builder b = sy::Builder::full();
        std::vector<uint8_t>& modl = b.sections["MODL"];
        // Overwrite group 0 models (m = 0..8, used = 2 buckets: bucket 0 + bucket 1 (symbol 1))
        // and group 1 models (m = 16..23, used = 3 buckets: bucket 0, bucket 1 (1 sym), bucket 2 (2 syms)).
        auto set_model = [&](uint32_t m, const uint32_t freqs[9]) {
            for (uint32_t k = 0; k < 9; ++k) {
                const size_t off = (static_cast<size_t>(m) * 9 + k) * 2;
                modl[off] = static_cast<uint8_t>(freqs[k] & 0xFFu);
                modl[off + 1] = static_cast<uint8_t>((freqs[k] >> 8) & 0xFFu);
            }
        };
        const uint32_t boundary_freqs[9] = {1u, 2u, 3u, 7u, 17u, 255u, 1024u, 32767u, 65535u};
        for (uint32_t idx = 0; idx < 9; ++idx) {
            const uint32_t f1 = boundary_freqs[idx];
            const uint32_t freqs[9] = {65536u - f1, f1, 0, 0, 0, 0, 0, 0, 0};
            set_model(idx, freqs);
        }
        // Group 1 models (width 3, header bits 6-7 = 2 -> group 1): 2 symbols in bucket 2 (symbols 2, 3)
        // with boundary frequencies so `into / freq` hits q = 0 and q = 1.
        const uint32_t b2_freqs[8] = {1u, 2u, 3u, 7u, 17u, 255u, 1024u, 32767u};
        for (uint32_t idx = 0; idx < 8; ++idx) {
            const uint32_t f2 = b2_freqs[idx];
            const uint32_t f1 = 1u;
            const uint32_t f0 = 65536u - f1 - (f2 << 1);
            const uint32_t freqs[9] = {f0, f1, f2, 0, 0, 0, 0, 0, 0};
            set_model(16 + idx, freqs);
        }
        // Use a BC1 layout with all lane widths = 3 so header (3<<6)|low picks group 0 (m=low)
        // and header (2<<6)|low picks group 1 (m=16+low).
        std::vector<sy::LaneSpec> layouts = sy::default_layouts();
        layouts[0].widths.assign(12, 3);
        b.set_layouts(layouts);
        const auto t_edge = b.tables();
        CHECK(t_edge != nullptr);
        if (t_edge) {
            auto encode_lane_custom = [](uint8_t header_byte, const uint32_t f[9], const std::vector<uint8_t>& syms) {
                uint32_t first[9] = {};
                for (uint32_t k = 1; k < 9; ++k) {
                    first[k] = first[k - 1] + (f[k - 1] << (k == 1 ? 0 : k - 2));
                }
                uint64_t x = 1u << 24;
                std::vector<uint8_t> emitted;
                for (size_t k = syms.size(); k-- > 0;) {
                    const uint32_t s = syms[k];
                    uint32_t freq = 0;
                    uint32_t start = 0;
                    if (s == 0) {
                        freq = f[0] - 1;
                        start = 1;
                    } else {
                        uint32_t bk = 0;
                        while ((s >> bk) != 0) {
                            ++bk;
                        }
                        freq = f[bk];
                        start = first[bk] + (s - (1u << (bk - 1))) * freq;
                    }
                    while (x >= (static_cast<uint64_t>(freq) << 16)) {
                        emitted.push_back(static_cast<uint8_t>(x));
                        x >>= 8;
                    }
                    x = ((x / freq) << 16) + (x % freq) + start;
                }
                for (int k = 0; k < 4; ++k) {
                    emitted.push_back(static_cast<uint8_t>(x >> (8 * k)));
                }
                std::vector<uint8_t> payload(emitted.rbegin(), emitted.rend());
                std::vector<uint8_t> out = {header_byte, static_cast<uint8_t>(payload.size() - 1)};
                out.insert(out.end(), payload.begin(), payload.end());
                const uint8_t filler[] = {0x00, 0x01, 0x00, 0x00};
                out.insert(out.end(), std::begin(filler), std::end(filler));
                return out;
            };
            // Decode 16 blocks across 12 lanes: lanes 0..8 test group 0 (including freq[1] == 65535!),
            // lanes 9..11 test group 1 (bucket 2 with freq 255, 1024, 32767).
            constexpr size_t kEdgeBlocks = 16;
            struct LaneQueues {
                std::vector<std::deque<uint8_t>> q;
            } lq;
            std::vector<uint8_t> expected(kEdgeBlocks * 12, 0);
            for (uint32_t lane = 0; lane < 12; ++lane) {
                std::vector<uint8_t> syms(kEdgeBlocks, 1);
                uint8_t hdr = 0;
                uint32_t f[9] = {};
                if (lane < 9) {
                    hdr = static_cast<uint8_t>((3u << 6) | lane);  // group 0, model `lane`
                    f[0] = 65536u - boundary_freqs[lane];
                    f[1] = boundary_freqs[lane];
                    for (size_t i = 0; i < kEdgeBlocks; ++i) {
                        // When f[0] == 1 (lane 8, freq[1] == 65535), symbol 0 has 0 slots so only symbol 1 is valid.
                        syms[i] = (lane == 8) ? uint8_t{1} : static_cast<uint8_t>((i + lane) & 1u);
                    }
                } else {
                    const uint32_t m_idx = lane - 4;  // 5, 6, 7 -> b2_freqs = 255, 1024, 32767
                    hdr = static_cast<uint8_t>((2u << 6) | m_idx);  // group 1, model `16 + m_idx`
                    f[1] = 1u;
                    f[2] = b2_freqs[m_idx];
                    f[0] = 65536u - f[1] - (f[2] << 1);
                    for (size_t i = 0; i < kEdgeBlocks; ++i) {
                        // Cycle through 1, 2, 3 (avoiding 0 if f[0] == 1 for f[2] == 32767)
                        syms[i] = (m_idx == 7) ? static_cast<uint8_t>(1 + (i % 3)) : static_cast<uint8_t>((i + lane) % 4);
                    }
                }
                for (size_t i = 0; i < kEdgeBlocks; ++i) {
                    expected[i * 12 + lane] = syms[i];
                }
                const std::vector<uint8_t> bytes = encode_lane_custom(hdr, f, syms);
                lq.q.emplace_back(bytes.begin(), bytes.end());
            }
            auto fetch_q = [](void* ctx, size_t lane, uint8_t* byte) {
                auto* qs = static_cast<LaneQueues*>(ctx);
                if (qs->q[lane].empty()) {
                    return false;
                }
                *byte = qs->q[lane].front();
                qs->q[lane].pop_front();
                return true;
            };
            std::vector<uint8_t> got(kEdgeBlocks * 12, 0);
            CHECK(bp::decode_lanes(*t_edge, 1, kEdgeBlocks, fetch_q, &lq, got.data()));
            CHECK(got == expected);
        }
    }

    const auto t = sy::Builder::full().tables();
    CHECK(t != nullptr);
    if (!t) {
        return;
    }

    // 3. `slot == 0` rANS state test (x > 2^24 with (x & 0xFFFF) == 0): must abort cleanly without
    //    unsigned underflow or crash.
    {
        for (uint8_t hi_byte : {uint8_t{0x02}, uint8_t{0x01}}) {
            const uint8_t second_byte = hi_byte == 0x01 ? uint8_t{0x01} : uint8_t{0x00};
            // Lane 0 loads 4-byte state 0x02000000 or 0x01010000 (> kRansFloor, slot == 0).
            struct SlotZeroCtx {
                uint8_t bad_lane[6];
                size_t at[12] = {};
            } ctx{{0x00, 0x03, hi_byte, second_byte, 0x00, 0x00}};
            auto fetch_slot0 = [](void* c, size_t lane, uint8_t* byte) {
                auto* s = static_cast<SlotZeroCtx*>(c);
                if (lane == 0 && s->at[0] < sizeof(s->bad_lane)) {
                    *byte = s->bad_lane[s->at[0]++];
                    return true;
                }
                return false;
            };
            uint8_t rec[12] = {};
            CHECK(!bp::decode_lanes(*t, 1, 1, fetch_slot0, &ctx, rec));
        }
        // Also via decode() with an interleaved byte buffer feeding lane 0 a slot==0 state.
        // Round 0 feeds header 0x00 to lanes 0..7; round 1 feeds length 0x03 to lanes 0..7;
        // subsequent rounds feed 0x02, 0x00, 0x00, 0x00 into lane 0's payload.
        std::vector<uint8_t> bad_stream;
        for (int i = 0; i < 12; ++i) bad_stream.push_back(0x00);  // headers
        for (int i = 0; i < 12; ++i) bad_stream.push_back(0x03);  // 4-byte payloads
        for (int i = 0; i < 12; ++i) bad_stream.push_back(0x02);  // byte 0 of state -> 0x02......
        for (int i = 0; i < 36; ++i) bad_stream.push_back(0x00);  // bytes 1..3 -> 0x02000000 (slot == 0)
        uint8_t out_blk[8] = {};
        CHECK(bp::decode(*t, 1, bad_stream.data(), bad_stream.size(), out_blk, sizeof(out_blk)) ==
              bp::DecodeStatus::Truncated);
    }

    // 4. Invalid BC7C control words (rotation/index_select/partition/subsets violations) -> Corrupt.
    {
        sy::Builder b = sy::Builder::full();
        std::vector<uint16_t> ctrl(512, 0);
        ctrl[10] = 6 | (1 << 3) | (1 << 6);            // mode 6 with non-zero rotation
        ctrl[11] = 0 | (2 << 3) | (3 << 6);            // mode 0 with non-zero rotation
        ctrl[12] = 5 | (1 << 5) | (1 << 6);            // mode 5 with non-zero index_select
        ctrl[13] = 6 | (1 << 5) | (1 << 6);            // mode 6 with non-zero index_select
        ctrl[14] = 0 | (3 << 6) | (16 << 8);           // mode 0 with partition 16 (>= 16)
        ctrl[15] = 0 | (3 << 6) | (63 << 8);           // mode 0 with partition 63 (>= 16)
        ctrl[16] = 4 | (1 << 6) | (1 << 8);            // mode 4 with non-zero partition
        ctrl[17] = 5 | (1 << 6) | (5 << 8);            // mode 5 with non-zero partition
        ctrl[18] = 6 | (1 << 6) | (1 << 8);            // mode 6 with non-zero partition
        ctrl[19] = 2 | (2 << 6);                       // mode 2 with 2 subsets (needs 3)
        ctrl[20] = 7 | (3 << 3) | (2 << 6);            // mode 7 with non-zero rotation
        std::vector<uint8_t>& bc7c = b.sections["BC7C"];
        bc7c.clear();
        for (uint16_t w : ctrl) {
            sy::put16(bc7c, w);
        }
        const auto t_bad_ctrl = b.tables();
        CHECK(t_bad_ctrl != nullptr);
        if (t_bad_ctrl) {
            for (uint32_t code = 10; code <= 20; ++code) {
                Record r(32, 0);
                r[30] = static_cast<uint8_t>(code & 31u);
                r[31] = static_cast<uint8_t>(code >> 5);
                bp::DecodeStatus st = bp::DecodeStatus::Ok;
                decode_records(*t_bad_ctrl, 7, {r}, &st);
                CHECK(st == bp::DecodeStatus::Corrupt);
            }
        }
    }

    // 5. decode_lanes() and decode() null / zero / overflow / BC2 guards.
    {
        auto dummy_fetch = [](void*, size_t, uint8_t* byte) {
            *byte = 0;
            return true;
        };
        uint8_t rec[32] = {};
        CHECK(!bp::decode_lanes(*t, 1, 0, dummy_fetch, nullptr, rec));
        CHECK(!bp::decode_lanes(*t, 1, 1, dummy_fetch, nullptr, nullptr));
        CHECK(!bp::decode_lanes(*t, 1, 1, nullptr, nullptr, rec));
        CHECK(!bp::decode_lanes(*t, 0, 1, dummy_fetch, nullptr, rec));
        CHECK(!bp::decode_lanes(*t, 2, 1, dummy_fetch, nullptr, rec));
        CHECK(!bp::decode_lanes(*t, 8, 1, dummy_fetch, nullptr, rec));
        CHECK(!bp::decode_lanes(*t, 7, std::numeric_limits<size_t>::max(), dummy_fetch, nullptr, rec));
        CHECK(!bp::decode_lanes(*t, 7, std::numeric_limits<size_t>::max() / 2, dummy_fetch, nullptr, rec));

        uint8_t dst[32] = {};
        uint8_t src[16] = {};
        CHECK(bp::decode(*t, 1, nullptr, 16, dst, 8) == bp::DecodeStatus::Truncated);
        CHECK(bp::decode(*t, 7, nullptr, 1, dst, 16) == bp::DecodeStatus::Truncated);
        CHECK(bp::decode(*t, 1, src, sizeof(src), nullptr, 8) == bp::DecodeStatus::BadSize);
        CHECK(bp::decode(*t, 7, src, sizeof(src), dst, 8) == bp::DecodeStatus::BadSize);
        CHECK(bp::decode(*t, 7, src, sizeof(src), dst, 15) == bp::DecodeStatus::BadSize);
        CHECK(bp::decode(*t, 1, src, sizeof(src), dst, 7) == bp::DecodeStatus::BadSize);
        CHECK(bp::decode(*t, 2, src, sizeof(src), dst, 16) == bp::DecodeStatus::Unsupported);
        CHECK(!bp::supports(*t, 2));
        CHECK(bp::block_bytes(2) == 0);
    }

    // 6. load_tables_file() rejects oversized (> 64 MiB) files cleanly instead of truncating.
    {
        const std::string path = "test_bcpack_oversized.tmp";
        std::FILE* f = std::fopen(path.c_str(), "wb");
        CHECK(f != nullptr);
        if (f != nullptr) {
            std::vector<uint8_t> zeros(1u << 20, 0);
            for (int mb = 0; mb < 64; ++mb) {
                std::fwrite(zeros.data(), 1, zeros.size(), f);
            }
            const uint8_t extra = 0x42;
            std::fwrite(&extra, 1, 1, f);
            std::fclose(f);
            std::string error;
            CHECK(!bp::load_tables_file(path, &error));
            CHECK(error.find("64 MiB") != std::string::npos || error.find("exceeds") != std::string::npos);
            std::remove(path.c_str());
        }
    }
}

// ---------------------------------------------------------------------------
// Surface block layout helpers (os/surface_blocks.h)
// ---------------------------------------------------------------------------

void test_surface_blocks() {
    namespace sf = playgta6::os::surface;

    // Compile-time constexpr checks.
    static_assert(sf::kBlockBytes == 65536u);
    static_assert(sf::kBlock4kBytes == 4096u);
    static_assert(sf::bc_block_bytes(static_cast<uint32_t>(sf::BcFormat::BC1)) == 8u);
    static_assert(sf::bc_block_bytes(static_cast<uint32_t>(sf::BcFormat::BC2)) == 16u);
    static_assert(sf::bc_block_bytes(static_cast<uint32_t>(sf::BcFormat::BC3)) == 16u);
    static_assert(sf::bc_block_bytes(static_cast<uint32_t>(sf::BcFormat::BC4)) == 8u);
    static_assert(sf::bc_block_bytes(static_cast<uint32_t>(sf::BcFormat::BC5)) == 16u);
    static_assert(sf::bc_block_bytes(static_cast<uint32_t>(sf::BcFormat::BC6H)) == 16u);
    static_assert(sf::bc_block_bytes(static_cast<uint32_t>(sf::BcFormat::BC7)) == 16u);
    static_assert(sf::bc_block_bytes(0u) == 0u && sf::bc_block_bytes(8u) == 0u);
    static_assert(sf::mip_extent(4096u, 0u) == 4096u);
    static_assert(sf::mip_extent(4096u, 12u) == 1u);
    static_assert(sf::mip_extent(4096u, 32u) == 1u);
    static_assert(sf::mip_extent(4096u, 99u) == 1u);

    // Existing block64k_dims / block64k_layout_2d invariants.
    {
        const struct {
            uint32_t bpe, w, h;
        } cases[5] = {{1, 256, 256}, {2, 256, 128}, {4, 128, 128}, {8, 128, 64}, {16, 64, 64}};
        for (const auto& c : cases) {
            sf::BlockDims d{};
            CHECK(sf::block64k_dims(c.bpe, &d));
            CHECK(d.width == c.w && d.height == c.h);
            CHECK(static_cast<uint64_t>(d.width) * d.height * c.bpe == sf::kBlockBytes);
            sf::Block64kLayout l{};
            CHECK(!sf::block64k_layout_2d(c.w - 1, c.h, c.bpe, &l));
            CHECK(!sf::block64k_layout_2d(c.w, c.h - 1, c.bpe, &l));
            CHECK(sf::block64k_layout_2d(c.w, c.h, c.bpe, &l));
            CHECK(l.padded_width == c.w && l.padded_height == c.h);
            CHECK(l.size_bytes == sf::kBlockBytes && l.alignment == sf::kBlockBytes);
            CHECK(sf::block64k_layout_2d(c.w + 1, c.h + 1, c.bpe, &l));
            CHECK(l.padded_width == 2 * c.w && l.padded_height == 2 * c.h);
            CHECK(l.size_bytes == 4 * sf::kBlockBytes);
        }
        CHECK(!sf::block64k_dims(0, nullptr) && !sf::block64k_dims(3, nullptr) && !sf::block64k_dims(32, nullptr));
    }

    // bc_block_bytes
    for (uint32_t bc = 1; bc <= 7; ++bc) {
        const uint32_t expect = (bc == 1 || bc == 4) ? 8u : 16u;
        CHECK(sf::bc_block_bytes(bc) == expect);
    }
    CHECK(sf::bc_block_bytes(0) == 0 && sf::bc_block_bytes(8) == 0 && sf::bc_block_bytes(255) == 0);

    // bc_surface_grid (non-multiples, 1x1, 4K, UINT32_MAX overflow safety, zero dims)
    {
        uint32_t bx = 0, by = 0;
        CHECK(!sf::bc_surface_grid(0, 4, &bx, &by));
        CHECK(!sf::bc_surface_grid(4, 0, &bx, &by));
        CHECK(!sf::bc_surface_grid(0, 0, &bx, &by));
        CHECK(sf::bc_surface_grid(1, 1, &bx, &by) && bx == 1 && by == 1);
        CHECK(sf::bc_surface_grid(3, 4, &bx, &by) && bx == 1 && by == 1);
        CHECK(sf::bc_surface_grid(5, 9, &bx, &by) && bx == 2 && by == 3);
        CHECK(sf::bc_surface_grid(3840, 2160, &bx, &by) && bx == 960 && by == 540);
        CHECK(sf::bc_surface_grid(4096, 4096, &bx, &by) && bx == 1024 && by == 1024);
        CHECK(sf::bc_surface_grid(UINT32_MAX, UINT32_MAX, &bx, &by));
        CHECK(bx == 0x40000000u && by == 0x40000000u);
        CHECK(sf::bc_surface_grid(13, 29, nullptr, nullptr));
    }

    // bc_linear_size_bytes
    {
        uint64_t bytes = 0;
        CHECK(!sf::bc_linear_size_bytes(0, 16, 7, &bytes));
        CHECK(!sf::bc_linear_size_bytes(16, 0, 7, &bytes));
        CHECK(!sf::bc_linear_size_bytes(16, 16, 0, &bytes));
        CHECK(!sf::bc_linear_size_bytes(16, 16, 8, &bytes));
        CHECK(sf::bc_linear_size_bytes(1, 1, 1, &bytes) && bytes == 8);
        CHECK(sf::bc_linear_size_bytes(1, 1, 2, &bytes) && bytes == 16);
        CHECK(sf::bc_linear_size_bytes(5, 9, 4, &bytes) && bytes == 2ull * 3ull * 8ull);
        CHECK(sf::bc_linear_size_bytes(4096, 4096, 7, &bytes) && bytes == 1024ull * 1024ull * 16ull);
        CHECK(sf::bc_linear_size_bytes(4096, 4096, 1, &bytes) && bytes == 1024ull * 1024ull * 8ull);
        // UINT32_MAX x UINT32_MAX: 2^30 x 2^30 = 2^60 blocks -> * 16 overflows uint64_t, * 8 = 2^63 fits.
        CHECK(!sf::bc_linear_size_bytes(UINT32_MAX, UINT32_MAX, 7, &bytes));
        CHECK(sf::bc_linear_size_bytes(UINT32_MAX, UINT32_MAX, 1, &bytes) && bytes == (uint64_t{1} << 63));
    }

    // block4k_dims / block4k_layout_2d
    {
        const struct {
            uint32_t bpe, w, h;
        } cases[5] = {{1, 64, 64}, {2, 64, 32}, {4, 32, 32}, {8, 32, 16}, {16, 16, 16}};
        for (const auto& c : cases) {
            sf::BlockDims d{};
            CHECK(sf::block4k_dims(c.bpe, &d));
            CHECK(d.width == c.w && d.height == c.h);
            CHECK(static_cast<uint64_t>(d.width) * d.height * c.bpe == sf::kBlock4kBytes);
            sf::Block64kLayout l{};
            CHECK(sf::block4k_layout_2d(1, 1, c.bpe, &l));
            CHECK(l.padded_width == c.w && l.padded_height == c.h);
            CHECK(l.row_pitch_bytes == static_cast<uint64_t>(c.w) * c.bpe);
            CHECK(l.size_bytes == sf::kBlock4kBytes && l.alignment == sf::kBlock4kBytes);
            CHECK(sf::block4k_layout_2d(c.w + 1, c.h * 2 + 3, c.bpe, &l));
            CHECK(l.padded_width == 2 * c.w && l.padded_height == 3 * c.h);
            CHECK(l.size_bytes == 6 * sf::kBlock4kBytes && l.alignment == sf::kBlock4kBytes);
            CHECK(!sf::block4k_layout_2d(0, 1, c.bpe, &l));
            CHECK(!sf::block4k_layout_2d(1, 0, c.bpe, &l));
            CHECK(!sf::block4k_layout_2d(UINT32_MAX, 1, c.bpe, &l));  // padded_width overflows u32
        }
        CHECK(!sf::block4k_dims(0, nullptr) && !sf::block4k_dims(3, nullptr) && !sf::block4k_dims(32, nullptr));
    }

    // block64k_bc_dims / block64k_bc_layout_2d
    {
        sf::BlockDims d{};
        CHECK(sf::block64k_bc_dims(1, &d) && d.width == 128 && d.height == 64);
        CHECK(sf::block64k_bc_dims(4, &d) && d.width == 128 && d.height == 64);
        for (uint32_t bc : {2u, 3u, 5u, 6u, 7u}) {
            CHECK(sf::block64k_bc_dims(bc, &d) && d.width == 64 && d.height == 64);
        }
        CHECK(!sf::block64k_bc_dims(0, &d) && !sf::block64k_bc_dims(8, &d));

        sf::Block64kLayout l{};
        // 1x1 BC7 -> 1x1 blocks -> padded to 64x64 blocks = 64 KiB
        CHECK(sf::block64k_bc_layout_2d(1, 1, 7, &l));
        CHECK(l.padded_width == 64 && l.padded_height == 64);
        CHECK(l.row_pitch_bytes == 64u * 16u && l.size_bytes == sf::kBlockBytes && l.alignment == sf::kBlockBytes);
        // 513x257 BC1 -> 129x65 blocks -> padded to 256x128 blocks = 4 x 64 KiB
        CHECK(sf::block64k_bc_layout_2d(513, 257, 1, &l));
        CHECK(l.padded_width == 256 && l.padded_height == 128);
        CHECK(l.row_pitch_bytes == 256u * 8u && l.size_bytes == 4 * sf::kBlockBytes && l.alignment == sf::kBlockBytes);
        // 4096x4096 BC7 -> 1024x1024 blocks -> exact 16x16 tiles = 16 MiB
        CHECK(sf::block64k_bc_layout_2d(4096, 4096, 7, &l));
        CHECK(l.padded_width == 1024 && l.padded_height == 1024);
        CHECK(l.row_pitch_bytes == 1024u * 16u && l.size_bytes == 16ull * 1024ull * 1024ull);
        CHECK(l.alignment == sf::kBlockBytes);
        CHECK(!sf::block64k_bc_layout_2d(0, 64, 7, &l));
        CHECK(!sf::block64k_bc_layout_2d(64, 64, 0, &l));
        CHECK(!sf::block64k_bc_layout_2d(UINT32_MAX, UINT32_MAX, 7, &l));
    }

    // block4k_bc_dims / block4k_bc_layout_2d
    {
        sf::BlockDims d{};
        CHECK(sf::block4k_bc_dims(1, &d) && d.width == 32 && d.height == 16);
        CHECK(sf::block4k_bc_dims(4, &d) && d.width == 32 && d.height == 16);
        for (uint32_t bc : {2u, 3u, 5u, 6u, 7u}) {
            CHECK(sf::block4k_bc_dims(bc, &d) && d.width == 16 && d.height == 16);
        }
        CHECK(!sf::block4k_bc_dims(0, &d) && !sf::block4k_bc_dims(9, &d));

        sf::Block64kLayout l{};
        CHECK(sf::block4k_bc_layout_2d(1, 1, 7, &l));
        CHECK(l.padded_width == 16 && l.padded_height == 16);
        CHECK(l.row_pitch_bytes == 16u * 16u && l.size_bytes == sf::kBlock4kBytes && l.alignment == sf::kBlock4kBytes);
        CHECK(sf::block4k_bc_layout_2d(129, 65, 1, &l));
        CHECK(l.padded_width == 64 && l.padded_height == 32);
        CHECK(l.row_pitch_bytes == 64u * 8u && l.size_bytes == 4 * sf::kBlock4kBytes &&
              l.alignment == sf::kBlock4kBytes);
        CHECK(!sf::block4k_bc_layout_2d(0, 64, 1, &l));
        CHECK(!sf::block4k_bc_layout_2d(64, 64, 99, &l));
        CHECK(!sf::block4k_bc_layout_2d(UINT32_MAX, UINT32_MAX, 7, &l));
    }

    // mip_extent
    CHECK(sf::mip_extent(4096, 0) == 4096);
    CHECK(sf::mip_extent(4096, 1) == 2048);
    CHECK(sf::mip_extent(4096, 11) == 2);
    CHECK(sf::mip_extent(4096, 12) == 1);
    CHECK(sf::mip_extent(4096, 13) == 1);
    CHECK(sf::mip_extent(4096, 31) == 1);
    CHECK(sf::mip_extent(4096, 32) == 1);
    CHECK(sf::mip_extent(4096, 100) == 1);
    CHECK(sf::mip_extent(300, 1) == 150);
    CHECK(sf::mip_extent(300, 2) == 75);
    CHECK(sf::mip_extent(300, 3) == 37);
    CHECK(sf::mip_extent(0, 0) == 1);
    CHECK(sf::mip_extent(1, 0) == 1);
    CHECK(sf::mip_extent(UINT32_MAX, 0) == UINT32_MAX);
    CHECK(sf::mip_extent(UINT32_MAX, 31) == 1);
    CHECK(sf::mip_extent(UINT32_MAX, 32) == 1);
}

}  // namespace

int main() {
    test_table_parsing();
    test_tables_file_and_active();
    test_lane_round_trip();
    test_bc1_references();
    test_bc4_bc5_bc3();
    test_bc7_modes();
    test_bc7_references();
    test_bc6h();
    test_errors();
    test_streaming_multichunk_and_history_wrap();
    test_bc7_all_modes();
    test_bc6h_all_modes();
    test_hardening_and_fast_div();
    test_surface_blocks();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("test_bcpack: all checks passed\n");
    return 0;
}
