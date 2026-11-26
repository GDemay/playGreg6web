#include "gpu/xbox_embedded_native.h"

#include <cstring>

#include "os/xbox_scarlett_guest.h"

namespace playgta6::gpu::d3d12_guest::xbox {

namespace {

constexpr uint32_t kDxbcMagic = 0x43425844u;  // "DXBC"
constexpr uint32_t kDxilPart = 0x4C495844u;   // "DXIL"
constexpr size_t kXdatOffset = 0x18;
constexpr uint32_t kMaxXdatEntries = 64;
constexpr uint32_t kKindNative = 0x0d;
constexpr uint32_t kXbchVersion = 3;
constexpr size_t kXbchHeader = 0x18;
constexpr uint32_t kFssmVersion = 0x101;
constexpr size_t kFssmHeader = 0x10;
constexpr uint64_t kGraphicsSlots = 6;
constexpr uint64_t kAbsent = 0xDEADBEEFDEADBEEFull;
constexpr uint64_t kRecordBytes = 0x54;
constexpr uint64_t kMaxUnpacked = 16u << 20;
constexpr size_t kPsPatchSize = 0x148;
constexpr size_t kPsPatches = 0x14c;
constexpr size_t kPatchVariantDwords = 10;
constexpr size_t kPatchFull = 9;
constexpr size_t kPatchBytes = 4 + 10 * kPatchVariantDwords * 4;
constexpr size_t kMrts = 8;

struct Slot {
    EmbeddedStage stage;
    size_t at;
};
constexpr Slot kSlots[] = {{EmbeddedStage::Ls, 0x08}, {EmbeddedStage::Hs, 0x10}, {EmbeddedStage::Es, 0x18},
                           {EmbeddedStage::Gs, 0x20}, {EmbeddedStage::Vs, 0x28}, {EmbeddedStage::Ps, 0x30},
                           {EmbeddedStage::Ms, 0x48}, {EmbeddedStage::Cs, 0x60}};

bool in(size_t len, uint64_t at, uint64_t n) {
    return at <= len && n <= len - at;
}

uint32_t u32(const uint8_t* p, size_t at) {
    uint32_t v;
    std::memcpy(&v, p + at, 4);
    return v;
}

uint64_t u64(const uint8_t* p, size_t at) {
    uint64_t v;
    std::memcpy(&v, p + at, 8);
    return v;
}

bool programs_of(const std::vector<uint8_t>& fssm, std::vector<EmbeddedProgram>* out, std::string* error) {
    auto fail = [&](const char* m) {
        *error = m;
        return false;
    };
    if (fssm.size() < kFssmHeader + 0x68 || std::memcmp(fssm.data(), "FSSM", 4) != 0) {
        return fail("native blob lacks FSSM magic");
    }
    if (u32(fssm.data(), 4) != kFssmVersion) {
        return fail("unknown FSSM version");
    }
    const uint8_t* pl = fssm.data() + kFssmHeader;
    const size_t len = fssm.size() - kFssmHeader;
    if (u64(fssm.data(), 8) != len) {
        return fail("FSSM payload size disagrees");
    }
    if (u64(pl, 0) != kGraphicsSlots) {
        return fail("unexpected FSSM slot count");
    }
    for (const Slot& s : kSlots) {
        const uint64_t start = u64(pl, s.at);
        if (start == kAbsent) {
            continue;
        }
        if (!in(len, start, kRecordBytes)) {
            return fail("FSSM record out of bounds");
        }
        const uint64_t code_at = u64(pl, start);
        const uint64_t code_size = u64(pl, start + 8);
        if (code_size == 0 || code_size % 4 != 0 || !in(len, code_at, code_size)) {
            return fail("native code out of bounds");
        }
        EmbeddedProgram p;
        p.stage = s.stage;
        p.code.assign(pl + code_at, pl + code_at + code_size);
        if (s.stage == EmbeddedStage::Ps) {
            const uint64_t e = start + kRecordBytes;
            if (!in(len, e, kPsPatches + kMrts * kPatchBytes)) {
                return fail("pixel stage data out of bounds");
            }
            const uint32_t dwords = u32(pl, e + kPsPatchSize);
            if (dwords > kPatchVariantDwords) {
                return fail("implausible export patch size");
            }
            for (size_t mrt = 0; mrt < kMrts; ++mrt) {
                const size_t at = e + kPsPatches + mrt * kPatchBytes;
                const uint32_t offset = u32(pl, at);
                if (offset == 0xFFFFFFFFu) {
                    continue;
                }
                if ((uint64_t{offset} + dwords) * 4 > code_size) {
                    return fail("export patch outside the code");
                }
                std::memcpy(p.code.data() + size_t{offset} * 4, pl + at + 4 + kPatchFull * kPatchVariantDwords * 4,
                            size_t{dwords} * 4);
            }
        }
        out->push_back(std::move(p));
    }
    return true;
}

}  // namespace

bool extract_embedded_native(const uint8_t* c, size_t n, EmbeddedNative* out, std::string* error) {
    *out = EmbeddedNative{};
    auto fail = [&](const char* m) {
        *error = m;
        return false;
    };
    if (n < 32 || u32(c, 0) != kDxbcMagic) {
        return true;
    }
    const uint32_t parts = u32(c, 28);
    if (parts > 64 || !in(n, 32, uint64_t{parts} * 4)) {
        return true;
    }
    const uint8_t* d = nullptr;
    size_t dn = 0;
    for (uint32_t i = 0; i < parts; ++i) {
        const uint32_t off = u32(c, 32 + i * 4);
        if (!in(n, off, 8)) {
            return true;
        }
        const uint32_t size = u32(c, off + 4);
        if (u32(c, off) == kDxilPart && in(n, uint64_t{off} + 8, size)) {
            d = c + off + 8;
            dn = size;
            break;
        }
    }
    if (d == nullptr || dn < kXdatOffset + 12 || std::memcmp(d + kXdatOffset, "XDAT", 4) != 0) {
        return true;
    }
    const uint32_t count = u32(d, kXdatOffset + 4);
    if (count > kMaxXdatEntries || !in(dn, kXdatOffset + 12, uint64_t{count} * 12)) {
        return fail("implausible XDAT entry count");
    }
    for (uint32_t i = 0; i < count; ++i) {
        const size_t at = kXdatOffset + 12 + 12 * size_t{i};
        if (u32(d, at) != kKindNative) {
            continue;
        }
        const uint32_t off = u32(d, at + 4);
        const uint32_t size = u32(d, at + 8);
        if (!in(dn, off, size) || size < kXbchHeader || std::memcmp(d + off, "XBCH", 4) != 0) {
            return fail("native XDAT entry lacks XBCH");
        }
        const uint8_t* blob = d + off;
        if (u32(blob, 4) != kXbchVersion) {
            return fail("unknown XBCH version");
        }
        const uint64_t packed = u64(blob, 8);
        const uint64_t unpacked = u64(blob, 16);
        if (unpacked > kMaxUnpacked || !in(size, kXbchHeader, packed)) {
            return fail("implausible XBCH sizes");
        }
        std::vector<uint8_t> fssm(static_cast<size_t>(unpacked));
        size_t produced = 0;
        namespace g = playgta6::os::scarlett::guest_api;
        if (g::inflate_stream(blob + kXbchHeader, static_cast<size_t>(packed), fssm.data(), fssm.size(), &produced) !=
                g::InflateStatus::Ok ||
            produced != fssm.size()) {
            return fail("XBCH stream does not inflate to its stated size");
        }
        if (!programs_of(fssm, &out->programs, error)) {
            return false;
        }
        if (out->fssm.empty()) {
            out->fssm = std::move(fssm);
        }
    }
    return true;
}

const EmbeddedProgram* find_embedded(const EmbeddedNative& native, EmbeddedStage stage) {
    for (const EmbeddedProgram& p : native.programs) {
        if (p.stage == stage) {
            return &p;
        }
    }
    return nullptr;
}

}  // namespace playgta6::gpu::d3d12_guest::xbox
