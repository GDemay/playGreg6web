// Synthetic tests for native code embedded in DXIL containers (gpu/xbox_embedded_native.h, G66).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu/xbox_embedded_native.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_xbox_embedded_native.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace x = playgta6::gpu::d3d12_guest::xbox;

using Bytes = std::vector<uint8_t>;

void put32(Bytes& b, size_t at, uint32_t v) {
    if (b.size() < at + 4) {
        b.resize(at + 4);
    }
    std::memcpy(b.data() + at, &v, 4);
}
void put64(Bytes& b, size_t at, uint64_t v) {
    if (b.size() < at + 8) {
        b.resize(at + 8);
    }
    std::memcpy(b.data() + at, &v, 8);
}
void append(Bytes& b, const void* p, size_t n) {
    const auto* c = static_cast<const uint8_t*>(p);
    b.insert(b.end(), c, c + n);
}

// zlib stream of one stored DEFLATE block.
Bytes zlib_stored(const Bytes& data) {
    Bytes z = {0x78, 0x01, 0x01};
    const uint16_t len = static_cast<uint16_t>(data.size());
    const uint16_t nlen = static_cast<uint16_t>(~len);
    append(z, &len, 2);
    append(z, &nlen, 2);
    append(z, data.data(), data.size());
    uint32_t a = 1, b = 0;
    for (uint8_t c : data) {
        a = (a + c) % 65521;
        b = (b + a) % 65521;
    }
    const uint32_t adler = (b << 16) | a;
    for (int s = 24; s >= 0; s -= 8) {
        z.push_back(static_cast<uint8_t>(adler >> s));
    }
    return z;
}

constexpr uint64_t kAbsent = 0xDEADBEEFDEADBEEFull;
constexpr size_t kPatchBytes = 4 + 10 * 10 * 4;

// FSSM with a vertex program (2 dwords) and a pixel program (4 dwords, MRT0 export patch of
// 2 dwords at dword 1).
Bytes make_fssm() {
    Bytes pl(0x68, 0);
    put64(pl, 0, 6);
    for (size_t at = 0x08; at < 0x68; at += 8) {
        put64(pl, at, kAbsent);
    }
    const size_t vs_rec = pl.size();
    pl.resize(vs_rec + 0x54 + 0x24, 0);
    const size_t vs_code = pl.size();
    put32(pl, vs_code, 0x11111111);
    put32(pl, vs_code + 4, 0xBF810000);  // s_endpgm
    put64(pl, 0x28, vs_rec);
    put64(pl, vs_rec, vs_code);
    put64(pl, vs_rec + 8, 8);

    const size_t ps_rec = pl.size();
    const size_t e = ps_rec + 0x54;
    pl.resize(e + 0x14c + 8 * kPatchBytes, 0);
    put32(pl, e + 0x148, 2);
    for (size_t mrt = 0; mrt < 8; ++mrt) {
        put32(pl, e + 0x14c + mrt * kPatchBytes, mrt == 0 ? 1u : 0xFFFFFFFFu);
    }
    const size_t full = e + 0x14c + 4 + 9 * 40;
    put32(pl, full, 0xF800180F);  // exp mrt0 (raw 32-bit variant)
    put32(pl, full + 4, 0x03020100);
    const size_t ps_code = pl.size();
    put32(pl, ps_code, 0x22222222);
    put32(pl, ps_code + 4, 0xBF800000);  // placeholder (s_nop) to be patched
    put32(pl, ps_code + 8, 0xBF800000);
    put32(pl, ps_code + 12, 0xBF810000);
    put64(pl, 0x30, ps_rec);
    put64(pl, ps_rec, ps_code);
    put64(pl, ps_rec + 8, 16);

    Bytes f;
    append(f, "FSSM", 4);
    put32(f, 4, 0x101);
    put64(f, 8, pl.size());
    append(f, pl.data(), pl.size());
    return f;
}

// DXBC container whose DXIL part carries XDAT {0x12 stub, 0x0d native}.
Bytes make_container(const Bytes& fssm, uint32_t xbch_version = 3) {
    const Bytes z = zlib_stored(fssm);
    Bytes xbch;
    append(xbch, "XBCH", 4);
    put32(xbch, 4, xbch_version);
    put64(xbch, 8, z.size());
    put64(xbch, 16, fssm.size());
    append(xbch, z.data(), z.size());

    Bytes d(0x18, 0);
    put32(d, 0, (1u << 16) | 0x65);
    std::memcpy(d.data() + 8, "DXIL", 4);
    append(d, "XDAT", 4);
    put32(d, 0x1c, 2);
    put32(d, 0x20, 0x24);
    const size_t stub = 0x18 + 12 + 24;
    put32(d, 0x24, 0x12);
    put32(d, 0x28, static_cast<uint32_t>(stub));
    put32(d, 0x2c, 16);
    put32(d, 0x30, 0x0d);
    put32(d, 0x34, static_cast<uint32_t>(stub + 16));
    put32(d, 0x38, static_cast<uint32_t>(xbch.size()));
    d.resize(stub + 16, 0);
    append(d, xbch.data(), xbch.size());

    Bytes c(32 + 8, 0);
    std::memcpy(c.data(), "DXBC", 4);
    put32(c, 28, 2);
    put32(c, 32, 40);  // HASH part
    put32(c, 36, 40 + 8 + 20);
    append(c, "HASH", 4);
    put32(c, c.size(), 20);
    c.resize(c.size() + 20, 0xAB);
    append(c, "DXIL", 4);
    put32(c, c.size(), static_cast<uint32_t>(d.size()));
    append(c, d.data(), d.size());
    put32(c, 24, static_cast<uint32_t>(c.size()));
    return c;
}

void test_extracts_vertex_and_patched_pixel() {
    const Bytes fssm = make_fssm();
    const Bytes c = make_container(fssm);
    x::EmbeddedNative n;
    std::string error;
    CHECK(x::extract_embedded_native(c.data(), c.size(), &n, &error));
    CHECK(n.programs.size() == 2);
    CHECK(n.fssm == fssm);
    const x::EmbeddedProgram* vs = x::find_embedded(n, x::EmbeddedStage::Vs);
    const x::EmbeddedProgram* ps = x::find_embedded(n, x::EmbeddedStage::Ps);
    CHECK(vs != nullptr && vs->code.size() == 8);
    CHECK(ps != nullptr && ps->code.size() == 16);
    CHECK(x::find_embedded(n, x::EmbeddedStage::Cs) == nullptr);
    if (ps != nullptr && ps->code.size() == 16) {
        uint32_t w[4];
        std::memcpy(w, ps->code.data(), 16);
        CHECK(w[0] == 0x22222222 && w[1] == 0xF800180F && w[2] == 0x03020100 && w[3] == 0xBF810000);
    }
}

void test_plain_containers_carry_nothing() {
    x::EmbeddedNative n;
    std::string error;
    const Bytes junk(64, 0);
    CHECK(x::extract_embedded_native(junk.data(), junk.size(), &n, &error) && n.programs.empty());
    Bytes c = make_container(make_fssm());
    constexpr size_t kDxilData = 40 + 8 + 20 + 8;  // part offsets, HASH part, DXIL part header
    std::memcpy(c.data() + kDxilData + 0x18, "NONE", 4);
    CHECK(x::extract_embedded_native(c.data(), c.size(), &n, &error) && n.programs.empty());
}

void test_rejects_malformed() {
    x::EmbeddedNative n;
    std::string error;
    const Bytes bad_version = make_container(make_fssm(), 4);
    CHECK(!x::extract_embedded_native(bad_version.data(), bad_version.size(), &n, &error) &&
          error.find("XBCH version") != std::string::npos);
    Bytes fssm = make_fssm();
    put64(fssm, 16 + 0x28, 0xFFFFFF);  // vertex record past the payload
    const Bytes bad_record = make_container(fssm);
    CHECK(!x::extract_embedded_native(bad_record.data(), bad_record.size(), &n, &error) &&
          error.find("record out of bounds") != std::string::npos);
    Bytes truncated = make_container(make_fssm());
    truncated[truncated.size() - 6] ^= 0xFF;  // corrupt the stored data (Adler-32 mismatch)
    CHECK(!x::extract_embedded_native(truncated.data(), truncated.size(), &n, &error));
}

}  // namespace

int main() {
    test_extracts_vertex_and_patched_pixel();
    test_plain_containers_carry_nothing();
    test_rejects_malformed();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_xbox_embedded_native: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_xbox_embedded_native: all checks passed\n");
    return 0;
}
