// XMA2 voices (runtime/os/xma_decoder.h) on synthetic streams: packet headers, the frame walk
// across packet headers, padding, skipped packets and buffers, the output ring arithmetic, and
// pump() filling a ring: block order, start skip, input switch, loops and running dry.

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <vector>

#include "os/xapu_opus.h"
#include "os/xma_decoder.h"

namespace {

int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_xma_decoder.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace x = playgta6::os::xma;

constexpr uint32_t kPayload = x::kPacketBits - x::kPacketHeaderBits;

// The test pattern of frame `f`, bit `j` of its body (after the length field).
uint32_t body_bit(uint32_t f, uint32_t j) { return ((f * 7 + j) % 3) == 0 ? 1u : 0u; }

// A synthetic XMA2 stream: frames of the given lengths laid out back to back over the payload of
// the packets listed in `order` (other packets are foreign: a header and junk). A frame never
// starts with 15 bits or fewer left in a packet; those bits stay padding (all ones).
struct Stream {
    std::vector<uint8_t> data;
    std::vector<uint32_t> starts;  // absolute bit positions of the frames
    uint32_t packets = 0;
};

Stream build(const std::vector<uint32_t>& lengths, uint32_t packets, std::vector<uint32_t> order = {}) {
    if (order.empty()) {
        for (uint32_t i = 0; i < packets; ++i) {
            order.push_back(i);
        }
    }
    Stream s;
    s.packets = packets;
    s.data.assign(size_t{packets} * x::kPacketBytes, 0xFF);
    auto abs_of = [&](uint32_t p) { return order[p / kPayload] * x::kPacketBits + 32 + p % kPayload; };
    auto put = [&](uint32_t p, uint32_t b) {
        const uint32_t a = abs_of(p);
        uint8_t& byte = s.data[a / 8];
        const uint8_t m = static_cast<uint8_t>(0x80u >> (a % 8));
        byte = b != 0 ? (byte | m) : (byte & ~m);
    };
    std::vector<uint32_t> logical;
    uint32_t p = 0;
    for (uint32_t f = 0; f < lengths.size(); ++f) {
        if (kPayload - p % kPayload <= x::kFrameLengthBits) {
            p += kPayload - p % kPayload;
        }
        logical.push_back(p);
        s.starts.push_back(abs_of(p));
        for (uint32_t j = 0; j < x::kFrameLengthBits; ++j) {
            put(p + j, (lengths[f] >> (x::kFrameLengthBits - 1 - j)) & 1u);
        }
        for (uint32_t j = x::kFrameLengthBits; j < lengths[f]; ++j) {
            put(p + j, body_bit(f, j - x::kFrameLengthBits));
        }
        p += lengths[f];
    }
    for (uint32_t k = 0; k < packets; ++k) {
        x::PacketHeader h;
        h.first_frame_bit = x::kNoFrame;
        h.skip_packets = 0;
        uint8_t* pkt = s.data.data() + size_t{k} * x::kPacketBytes;
        uint32_t idx = 0;
        while (idx < order.size() && order[idx] != k) {
            ++idx;
        }
        if (idx == order.size()) {  // foreign packet
            h.frame_count = 1;
            h.first_frame_bit = 0;
            std::memset(pkt + 4, 0x5A, x::kPacketBytes - 4);
            x::write_packet_header(pkt, h);
            continue;
        }
        for (uint32_t l : logical) {
            if (l / kPayload == idx) {
                if (h.frame_count == 0) {
                    h.first_frame_bit = l % kPayload;
                }
                ++h.frame_count;
            }
        }
        if (idx + 1 < order.size()) {
            h.skip_packets = order[idx + 1] - order[idx] - 1;
        }
        x::write_packet_header(pkt, h);
    }
    return s;
}

bool frame_bits_match(const x::Frame& f, uint32_t index) {
    if (f.bits_gathered != f.length) {
        return false;
    }
    for (uint32_t j = 0; j < f.length; ++j) {
        const uint32_t b = (f.bits[j / 8] >> (7 - j % 8)) & 1u;
        const uint32_t want = j < x::kFrameLengthBits ? (f.length >> (x::kFrameLengthBits - 1 - j)) & 1u
                                                      : body_bit(index, j - x::kFrameLengthBits);
        if (b != want) {
            return false;
        }
    }
    return true;
}

void test_packet_header() {
    uint8_t p[4] = {};
    x::PacketHeader h{5, 1234, 3, 17};
    x::write_packet_header(p, h);
    // frame count 6 bits, first-frame bit 15, metadata 3, skip 8, big-endian.
    const uint32_t v = (5u << 26) | (1234u << 11) | (3u << 8) | 17u;
    CHECK(p[0] == (v >> 24) && p[1] == ((v >> 16) & 0xFF) && p[2] == ((v >> 8) & 0xFF) && p[3] == (v & 0xFF));
    const x::PacketHeader r = x::parse_packet_header(p);
    CHECK(r.frame_count == 5 && r.first_frame_bit == 1234 && r.metadata == 3 && r.skip_packets == 17);
}

void test_frame_walk() {
    // 9 frames of 4000 bits over 3 packets: frames 4 and 8 cross packet headers.
    const std::vector<uint32_t> lengths(9, 4000);
    Stream s = build(lengths, 3);
    CHECK(x::parse_packet_header(s.data.data()).first_frame_bit == 0);
    CHECK(x::parse_packet_header(s.data.data()).frame_count == 5);
    // Packet 1's first frame starts after the tail of frame 4 (16352 - 4 * 4000 = 352 bits in p0).
    CHECK(x::parse_packet_header(s.data.data() + 2048).first_frame_bit == 4000 - 352);

    auto f = x::read_frame(s.data.data(), s.packets, 0);  // below the payload: the first frame
    uint32_t n = 0;
    while (f) {
        CHECK(f->start == s.starts[n]);
        CHECK(f->length == 4000 && f->complete);
        CHECK(frame_bits_match(*f, n));
        ++n;
        if (!f->has_next) {
            break;
        }
        CHECK(n < s.starts.size() && f->next == s.starts[n]);
        f = x::read_frame(s.data.data(), s.packets, f->next);
    }
    CHECK(n == 9);
}

void test_padding_and_skip() {
    // A frame that would start with <= 15 bits left in packet 0 starts in packet 1 instead.
    Stream s = build({16352 - 10, 300}, 2);
    auto f = x::read_frame(s.data.data(), s.packets, 0);
    CHECK(f && f->has_next && f->next == x::kPacketBits + 32);
    f = x::read_frame(s.data.data(), s.packets, f->next);
    CHECK(f && f->length == 300 && !f->has_next && frame_bits_match(*f, 1));

    // An explicit 0x7FFF after a frame: the rest of the packet is padding.
    Stream q = build({1000}, 1);
    std::vector<uint8_t> two(2 * x::kPacketBytes, 0xFF);
    std::memcpy(two.data(), q.data.data(), x::kPacketBytes);
    Stream r = build({700}, 1);
    std::memcpy(two.data() + x::kPacketBytes, r.data.data(), x::kPacketBytes);
    f = x::read_frame(two.data(), 2, 0);
    CHECK(f && f->length == 1000 && f->has_next && f->next == x::kPacketBits + 32);
    f = x::read_frame(two.data(), 2, f->next);
    CHECK(f && f->length == 700 && !f->has_next);

    // Packet 0 says skip one packet: packet 1 belongs to another stream.
    Stream k = build(std::vector<uint32_t>(5, 4000), 3, {0, 2});
    CHECK(x::parse_packet_header(k.data.data()).skip_packets == 1);
    f = x::read_frame(k.data.data(), k.packets, 0);
    uint32_t n = 0;
    while (f && f->has_next) {
        ++n;
        CHECK(f->next == k.starts[n]);
        f = x::read_frame(k.data.data(), k.packets, f->next);
    }
    CHECK(n == 4 && f && frame_bits_match(*f, 4));
    CHECK(k.starts[4] / x::kPacketBits == 0);  // frame 4 runs on over the foreign packet

    // A length of 15 bits or less is not a frame.
    Stream bad = build({16}, 1);
    bad.data[4] = 0x00;
    bad.data[5] = 0x08;  // length 4
    CHECK(!x::read_frame(bad.data.data(), 1, 0));
}

void test_frame_across_buffers() {
    // Buffer A: frame 0 complete, frame 1 starts 2000 bits before A ends. Buffer B opens with
    // frame 1's remaining bits, then frame 2 at its packet header's first-frame offset.
    Stream whole = build({16352 - 2000, 5000, 3000}, 2);
    std::vector<uint8_t> a(whole.data.begin(), whole.data.begin() + x::kPacketBytes);
    std::vector<uint8_t> b(whole.data.begin() + x::kPacketBytes, whole.data.end());
    auto f0 = x::read_frame(a.data(), 1, 0);
    CHECK(f0 && f0->complete && f0->has_next);
    auto f1 = x::read_frame(a.data(), 1, f0->next);
    CHECK(f1 && !f1->complete && f1->bits_gathered == 2000);
    CHECK(x::continue_frame(b.data(), 1, f1->length - f1->bits_gathered, *f1));
    CHECK(f1->complete && frame_bits_match(*f1, 1));
    const auto first = x::first_frame_from(b.data(), 1, 0);
    CHECK(first && *first == 32 + 3000);
    auto f2 = x::read_frame(b.data(), 1, *first);
    CHECK(f2 && f2->length == 3000 && frame_bits_match(*f2, 2));
}

void test_ring_arithmetic() {
    CHECK(x::ring_step(3, 2, 4) == 1);
    CHECK(x::ring_step(31, 1, 32) == 0);
    CHECK(x::ring_step(7, 70, 5) == (7 % 5 + 70 % 5) % 5);  // indices past the end are reduced
    CHECK(x::ring_room(0, 0, 8, true) == 8);
    CHECK(x::ring_room(0, 0, 8, false) == 0);
    CHECK(x::ring_room(6, 2, 8, true) == 4);
    CHECK(x::ring_room(2, 6, 8, true) == 4);

    uint32_t w[16] = {};
    x::Context c(w);
    CHECK(c.ring_blocks() == 32);  // 0 encodes 32
    w[0] = (4u << 22) | (6u << 27);  // ring of 4, write block left at 6
    CHECK(c.ring_blocks() == 4 && c.write_block() == 2);
    w[9] = 0xABCD0000u | 3u;
    x::advance_read(c, 2);
    CHECK(w[9] == (0xABCD0000u | 1u));
    CHECK(c.output_writable());
    w[1] = 0;
    x::advance_read(c, 0);
    CHECK(!c.output_writable());
}

// Each frame decodes to samples numbered frame * 10000 + sample (per channel, interleaved).
class CountingDecoder final : public x::FrameDecoder {
public:
    bool decode(const x::Frame&, uint32_t channels, uint32_t, int16_t* pcm) override {
        for (uint32_t i = 0; i < x::kFrameSamples; ++i) {
            for (uint32_t ch = 0; ch < channels; ++ch) {
                pcm[i * channels + ch] = static_cast<int16_t>((frame_ % 3) * 10000 + i);
            }
        }
        ++frame_;
        return true;
    }
    uint32_t frame_ = 0;
};

// A fake APU address space: buffers at fixed addresses.
struct Apu {
    std::map<uint32_t, std::vector<uint8_t>*> regions;
    x::ApuMap map() {
        return [this](uint32_t apu, size_t bytes) -> uint8_t* {
            for (auto& [base, v] : regions) {
                if (apu >= base && uint64_t{apu} + bytes <= uint64_t{base} + v->size()) {
                    return v->data() + (apu - base);
                }
            }
            return nullptr;
        };
    }
};

int16_t sample_at(const std::vector<uint8_t>& out, uint32_t block, uint32_t i) {
    const uint32_t o = block * x::kBlockBytes + 2 * i;
    return static_cast<int16_t>(out[o] | (out[o + 1] << 8));
}

void test_pump_mono() {
    Stream s = build(std::vector<uint32_t>(3, 4000), 1);  // 3 frames, 1536 samples, 12 blocks
    std::vector<uint8_t> out(4 * x::kBlockBytes, 0xEE);
    Apu apu;
    apu.regions[0x10000] = &s.data;
    apu.regions[0x20000] = &out;
    uint32_t w[16] = {};
    x::Context c(w);
    w[0] = 1u | (1u << 20) | (4u << 22);  // 1 packet, input 0 valid, ring of 4
    w[1] = (1u << 24) | (3u << 27) | (1u << 31);  // skip 1 subframe at the start, 48 kHz, writable
    w[5] = 0x10000 + 0x10;  // a packet address is aligned down
    w[7] = 0x20000;
    x::Voice v;
    x::enable(c, v, std::make_unique<CountingDecoder>());
    CHECK(c.sample_rate() == 48000 && c.channels() == 1);

    CHECK(x::pump(c, v, apu.map()) == 4);
    CHECK(c.write_block() == 0 && !c.output_writable());
    CHECK(sample_at(out, 0, 0) == 128);  // the first subframe was skipped
    CHECK(sample_at(out, 2, 127) == 511 && sample_at(out, 3, 0) == 10000);  // frame 1 in block 3
    CHECK(x::pump(c, v, apu.map()) == 0);  // full until the title takes blocks

    x::advance_read(c, 2);
    CHECK(x::pump(c, v, apu.map()) == 2);
    CHECK(sample_at(out, 0, 0) == 10000 + 128);
    uint32_t total = 6;
    for (int i = 0; i < 10; ++i) {
        x::advance_read(c, 4);
        total += x::pump(c, v, apu.map());
    }
    // 1536 samples - 128 skipped = 11 blocks.
    CHECK(total == 11);
    CHECK(v.ended && v.stats.frames == 3 && v.stats.frames_decoded == 3);
    CHECK(!c.input_valid(0));
    CHECK(c.field(2, 26, 5) == x::kErrorInputExhausted && c.field(2, 31, 1) == 1);

    // More input revives the voice; enable clears the error.
    w[0] |= 1u << 20;
    x::enable(c, v, std::make_unique<CountingDecoder>());
    CHECK(c.field(2, 26, 6) == 0);
    x::advance_read(c, 4);
    CHECK(x::pump(c, v, apu.map()) > 0);
}

void test_pump_stereo_switch_and_span() {
    // Buffer 0: frame 0 and the head of frame 1; buffer 1: the tail of frame 1 and frame 2.
    Stream whole = build({16352 - 2000, 5000, 3000}, 2);
    std::vector<uint8_t> a(whole.data.begin(), whole.data.begin() + x::kPacketBytes);
    std::vector<uint8_t> b(whole.data.begin() + x::kPacketBytes, whole.data.end());
    std::vector<uint8_t> out(32 * x::kBlockBytes, 0);
    Apu apu;
    apu.regions[0x10000] = &a;
    apu.regions[0x30000] = &b;
    apu.regions[0x50000] = &out;
    uint32_t w[16] = {};
    x::Context c(w);
    w[0] = 1u | (3u << 20);  // 1 packet, both inputs valid, ring 0 = 32 blocks
    w[1] = 1u | (1u << 29) | (1u << 31);  // input 1: 1 packet, stereo, writable
    w[5] = 0x10000;
    w[6] = 0x30000;
    w[7] = 0x50000;
    x::Voice v;
    x::enable(c, v, std::make_unique<CountingDecoder>());
    // 3 frames * 512 * 2 channels = 3072 samples = 24 blocks.
    CHECK(x::pump(c, v, apu.map()) == 24);
    CHECK(v.stats.frames == 3 && v.stats.bad_frames == 0 && v.stats.input_switches == 1);
    CHECK(c.current_input() == 1 && !c.input_valid(0) && !c.input_valid(1));
    CHECK(sample_at(out, 8, 0) == 10000 && sample_at(out, 8, 1) == 10000);  // frame 1, both channels
    CHECK(c.write_block() == 24 && c.output_writable());
}

void test_pump_loop() {
    // 4 frames; loop once from the end of frame 2 (keep 2 subframes) back to frame 1, whose first
    // subframe is skipped after the jump.
    Stream s = build(std::vector<uint32_t>(4, 3000), 1);
    std::vector<uint8_t> out(32 * x::kBlockBytes, 0);
    Apu apu;
    apu.regions[0x10000] = &s.data;
    apu.regions[0x20000] = &out;
    uint32_t w[16] = {};
    x::Context c(w);
    w[0] = 1u | (1u << 12) | (1u << 20);  // 1 packet, loop count 1, input 0 valid, ring of 32
    w[1] = (1u << 12) | (1u << 17) | (1u << 31);  // loop end subframe 1 (keep 2), loop skip 1
    w[3] = s.starts[1];
    w[4] = s.starts[2] + 100;  // inside frame 2
    w[5] = 0x10000;
    w[7] = 0x20000;
    x::Voice v;
    x::enable(c, v, std::make_unique<CountingDecoder>());
    const uint32_t blocks = x::pump(c, v, apu.map());
    // frames 0, 1, 2 (256 kept), then 1 (384 after the skip), 2, 3 (full: loop count is 0).
    const uint32_t samples = 512 + 512 + 256 + 384 + 512 + 512;
    CHECK(v.stats.loops == 1 && v.stats.frames == 6);
    CHECK(c.loop_count() == 0);
    CHECK(blocks == (samples + 127) / 128);
    // After frame 2's cut (block 10 begins at sample 1280), frame 1 again from sample 128.
    CHECK(sample_at(out, 10, 0) == 10000 * (3 % 3) + 128);
}

// ---------------------------------------------------------------------------
// O50: Synthetic WMA Pro / XMA2 test encoder & round-trip / safety tests
// ---------------------------------------------------------------------------

constexpr double kTestPi = 3.14159265358979323846;

class BitWriter {
public:
    void write_bit(uint32_t b) {
        if (bit_count_ % 8u == 0u) {
            bytes_.push_back(0u);
        }
        if (b != 0u) {
            bytes_[bit_count_ / 8u] |= static_cast<uint8_t>(0x80u >> (bit_count_ % 8u));
        }
        ++bit_count_;
    }

    void write_bits(uint32_t v, uint32_t n) {
        for (uint32_t i = 0u; i < n; ++i) {
            write_bit((v >> (n - 1u - i)) & 1u);
        }
    }

    uint32_t bit_count() const { return bit_count_; }
    const std::vector<uint8_t>& bytes() const { return bytes_; }

    // Wraps payload with the leading 15-bit XMA2 frame length field.
    x::Frame finalize_frame() const {
        BitWriter full;
        const uint32_t total_len = x::kFrameLengthBits + bit_count_;
        full.write_bits(total_len, x::kFrameLengthBits);
        for (uint32_t i = 0u; i < bit_count_; ++i) {
            const uint32_t b = (bytes_[i / 8u] >> (7u - (i % 8u))) & 1u;
            full.write_bit(b);
        }
        x::Frame f;
        f.length = total_len;
        f.bits = full.bytes_;
        f.bits_gathered = total_len;
        f.complete = true;
        return f;
    }

private:
    std::vector<uint8_t> bytes_;
    uint32_t bit_count_ = 0u;
};

// Packs encoded x::Frame payloads back-to-back into valid 2048-byte XMA2 packets.
Stream pack_frames_to_stream(const std::vector<x::Frame>& frames) {
    uint64_t total_bits = 0u;
    for (const auto& f : frames) {
        total_bits += f.length + x::kFrameLengthBits;
    }
    const uint32_t packets = static_cast<uint32_t>(std::max<uint64_t>(1u, (total_bits + kPayload - 1u) / kPayload + 1u));
    Stream s;
    s.packets = packets;
    s.data.assign(size_t{packets} * x::kPacketBytes, 0xFF);
    auto abs_of = [](uint32_t p) { return (p / kPayload) * x::kPacketBits + 32u + (p % kPayload); };
    auto put = [&](uint32_t p, uint32_t b) {
        const uint32_t a = abs_of(p);
        uint8_t& byte = s.data[a / 8u];
        const uint8_t m = static_cast<uint8_t>(0x80u >> (a % 8u));
        byte = b != 0u ? (byte | m) : (byte & ~m);
    };
    std::vector<uint32_t> logical;
    uint32_t p = 0u;
    for (const auto& f : frames) {
        if (kPayload - (p % kPayload) <= x::kFrameLengthBits) {
            p += kPayload - (p % kPayload);
        }
        logical.push_back(p);
        s.starts.push_back(abs_of(p));
        for (uint32_t j = 0u; j < f.length; ++j) {
            const uint32_t b = (f.bits[j / 8u] >> (7u - (j % 8u))) & 1u;
            put(p + j, b);
        }
        p += f.length;
    }
    for (uint32_t k = 0u; k < packets; ++k) {
        x::PacketHeader h;
        h.first_frame_bit = x::kNoFrame;
        h.skip_packets = 0u;
        for (uint32_t l : logical) {
            if (l / kPayload == k) {
                if (h.frame_count == 0u) {
                    h.first_frame_bit = l % kPayload;
                }
                ++h.frame_count;
            }
        }
        x::write_packet_header(s.data.data() + size_t{k} * x::kPacketBytes, h);
    }
    return s;
}

uint32_t test_sfb_count(uint32_t sample_rate, uint32_t subframe_len, std::vector<uint32_t>& offsets) {
    static constexpr uint32_t kCrit[25] = {
        100u,  200u,  300u,  400u,  510u,  630u,  770u,  920u,  1080u,
        1270u, 1480u, 1720u, 2000u, 2320u, 2700u, 3150u, 3700u, 4400u,
        5300u, 6400u, 7700u, 9500u, 12000u, 15500u, 24500u,
    };
    offsets.clear();
    offsets.push_back(0u);
    const uint32_t step = subframe_len >= 128u ? 4u : 2u;
    uint32_t prev = 0u;
    for (uint32_t freq : kCrit) {
        uint32_t bin = static_cast<uint32_t>(
            (uint64_t{freq} * 2u * subframe_len + (sample_rate / 2u)) / sample_rate);
        bin = ((bin + step / 2u) / step) * step;
        if (bin <= prev) {
            continue;
        }
        if (bin >= subframe_len) {
            break;
        }
        offsets.push_back(bin);
        prev = bin;
        if (offsets.size() >= 26u) {
            break;
        }
    }
    if (prev < subframe_len) {
        offsets.push_back(subframe_len);
    }
    return static_cast<uint32_t>(offsets.size() - 1u);
}

void write_sf_delta(BitWriter& bw, int32_t delta) {
    if (delta == 0) {
        bw.write_bit(0u);
        return;
    }
    const uint32_t sign = delta < 0 ? 1u : 0u;
    const uint32_t mag = static_cast<uint32_t>(delta < 0 ? -delta : delta);
    if (mag == 1u) {
        bw.write_bits(0b10u, 2u);
        bw.write_bit(sign);
    } else if (mag == 2u) {
        bw.write_bits(0b110u, 3u);
        bw.write_bit(sign);
    } else if (mag <= 6u) {
        bw.write_bits(0b1110u, 4u);
        bw.write_bits(mag - 3u, 2u);
        bw.write_bit(sign);
    } else if (mag <= 14u) {
        bw.write_bits(0b11110u, 5u);
        bw.write_bits(mag - 7u, 3u);
        bw.write_bit(sign);
    } else {
        bw.write_bits(0b11111u, 5u);
        bw.write_bit(sign);
        bw.write_bits(std::min(mag, 127u), 7u);
    }
}

void write_quant_step(BitWriter& bw, int32_t q) {
    int32_t rem = std::max(0, q);
    while (rem >= 127) {
        bw.write_bits(127u, 7u);
        rem -= 127;
    }
    bw.write_bits(static_cast<uint32_t>(rem), 7u);
}

void write_spectral_coeffs(BitWriter& bw, const std::vector<int32_t>& coeffs, bool exercise_vec_stages) {
    const uint32_t n = static_cast<uint32_t>(coeffs.size());
    uint32_t pos = 0u;

    // Stage 1: optionally encode leading 4-tuples with vector-4 tokens (while |v| <= 130).
    if (exercise_vec_stages && n >= 8u) {
        bw.write_bit(1u);  // use_vec4
        for (uint32_t g = 0u; g < 2u; ++g) {
            bool fits = true;
            for (uint32_t lane = 0u; lane < 4u; ++lane) {
                if (std::abs(coeffs[pos + lane]) > 130) {
                    fits = false;
                }
            }
            if (!fits) {
                break;
            }
            bw.write_bits(0b1111u, 4u);  // general 4-tuple
            for (uint32_t lane = 0u; lane < 4u; ++lane) {
                const int32_t v = coeffs[pos + lane];
                const uint32_t mag = static_cast<uint32_t>(v < 0 ? -v : v);
                if (mag < 3u) {
                    bw.write_bits(mag, 2u);
                } else {
                    bw.write_bits(3u, 2u);
                    bw.write_bits(mag - 3u, 7u);
                }
                if (mag != 0u) {
                    bw.write_bit(v < 0 ? 1u : 0u);
                }
            }
            pos += 4u;
        }
        bw.write_bit(0u);  // end of vec4
    } else {
        bw.write_bit(0u);
    }

    // Stage 2: optionally encode next 2-tuples with vector-2 tokens (while |v| <= 130).
    if (exercise_vec_stages && pos + 4u <= n) {
        bw.write_bit(1u);  // use_vec2
        for (uint32_t g = 0u; g < 2u; ++g) {
            bool fits = true;
            for (uint32_t lane = 0u; lane < 2u; ++lane) {
                if (std::abs(coeffs[pos + lane]) > 130) {
                    fits = false;
                }
            }
            if (!fits) {
                break;
            }
            bw.write_bits(0b1111u, 4u);  // general 2-tuple
            for (uint32_t lane = 0u; lane < 2u; ++lane) {
                const int32_t v = coeffs[pos + lane];
                const uint32_t mag = static_cast<uint32_t>(v < 0 ? -v : v);
                if (mag < 3u) {
                    bw.write_bits(mag, 2u);
                } else {
                    bw.write_bits(3u, 2u);
                    bw.write_bits(mag - 3u, 7u);
                }
                if (mag != 0u) {
                    bw.write_bit(v < 0 ? 1u : 0u);
                }
            }
            pos += 2u;
        }
        bw.write_bit(0u);  // end of vec2
    } else {
        bw.write_bit(0u);
    }

    // Stage 3: run-level + escape coding for remaining nonzero bins.
    uint32_t run = 0u;
    for (uint32_t i = pos; i < n; ++i) {
        const int32_t v = coeffs[i];
        if (v == 0) {
            ++run;
            continue;
        }
        const uint32_t sign = v < 0 ? 1u : 0u;
        const uint32_t mag = static_cast<uint32_t>(v < 0 ? -v : v);
        if (mag == 1u && run == 0u) {
            bw.write_bits(0b10u, 2u);
            bw.write_bit(sign);
        } else if (mag == 1u && run >= 1u && run <= 4u) {
            bw.write_bits(0b110u, 3u);
            bw.write_bits(run - 1u, 2u);
            bw.write_bit(sign);
        } else if (mag >= 2u && mag <= 5u && run <= 7u) {
            bw.write_bits(0b1110u, 4u);
            bw.write_bits(run, 3u);
            bw.write_bits(mag - 2u, 2u);
            bw.write_bit(sign);
        } else if (mag == 1u && run <= 63u) {
            bw.write_bits(0b11110u, 5u);
            bw.write_bits(run, 6u);
            bw.write_bit(sign);
        } else {
            bw.write_bits(0b11111u, 5u);
            bw.write_bits(run, 9u);
            const uint32_t clamped = std::min(mag, 16384u);
            if (clamped <= 128u) {
                bw.write_bit(0u);
                bw.write_bits(clamped - 1u, 7u);
            } else {
                bw.write_bit(1u);
                bw.write_bits(clamped - 1u, 14u);
            }
            bw.write_bit(sign);
        }
        run = 0u;
    }
    if (run > 0u || pos < n) {
        // Emit EOB if the last bin was zero (decoder loop exits automatically when pos == n).
        if (run > 0u) {
            bw.write_bit(0u);
        }
    }
}

void mdct_forward(const std::vector<float>& signal, int64_t center_pos, uint32_t prev_len,
                  uint32_t cur_len, uint32_t next_len, std::vector<float>& out_spec) {
    const uint32_t w_left = std::min(prev_len, cur_len);
    const uint32_t w_right = std::min(cur_len, next_len);
    std::vector<double> xw(2u * cur_len, 0.0);
    const int64_t base = center_pos - static_cast<int64_t>(cur_len / 2u);

    auto sample = [&](int64_t idx) -> double {
        if (idx < 0 || static_cast<size_t>(idx) >= signal.size()) {
            return 0.0;
        }
        return static_cast<double>(signal[static_cast<size_t>(idx)]);
    };

    const uint32_t lead_z = (cur_len - w_left) / 2u;
    for (uint32_t j = 0u; j < w_left; ++j) {
        const double w = std::sin((static_cast<double>(j) + 0.5) * kTestPi / (2.0 * w_left));
        xw[lead_z + j] = sample(base + lead_z + j) * w;
    }
    for (uint32_t i = lead_z + w_left; i < cur_len; ++i) {
        xw[i] = sample(base + i);
    }

    const uint32_t flat_r = (cur_len - w_right) / 2u;
    for (uint32_t i = 0u; i < flat_r; ++i) {
        xw[cur_len + i] = sample(base + cur_len + i);
    }
    for (uint32_t j = 0u; j < w_right; ++j) {
        const double w = std::sin((static_cast<double>(w_right - 1u - j) + 0.5) * kTestPi / (2.0 * w_right));
        xw[cur_len + flat_r + j] = sample(base + cur_len + flat_r + j) * w;
    }

    out_spec.assign(cur_len, 0.0f);
    const double ang_scale = kTestPi / static_cast<double>(cur_len);
    for (uint32_t k = 0u; k < cur_len; ++k) {
        double sum = 0.0;
        const double kf = static_cast<double>(k) + 0.5;
        for (uint32_t n = 0u; n < 2u * cur_len; ++n) {
            const double nf = static_cast<double>(n) + 0.5 + static_cast<double>(cur_len) * 0.5;
            sum += xw[n] * std::cos(ang_scale * nf * kf);
        }
        out_spec[k] = static_cast<float>(sum);
    }
}

struct EncodeFrameOpts {
    uint32_t channels = 1u;
    uint32_t sample_rate = 48000u;
    std::vector<uint32_t> subframe_lens = {512u};
    uint32_t prev_subframe_len = 512u;
    uint32_t next_frame_first_len = 512u;
    uint32_t start_skip = 0u;
    uint32_t end_skip = 0u;
    bool use_decorrelation = false;
    bool use_default_ms = true;
    bool givens_negate = false;
    uint32_t givens_angle_idx = 12u;
    int32_t quant_step = 48;
    int32_t sf_base = 30;
    bool reuse_sf_after_first = false;
    bool exercise_vec_stages = false;
};

x::Frame encode_wmapro_frame(const std::array<std::vector<float>, 2>& signals, int64_t frame_start,
                             const EncodeFrameOpts& opts) {
    BitWriter bw;
    // 1. Optional skip prefix.
    if (opts.start_skip != 0u || opts.end_skip != 0u) {
        bw.write_bit(1u);
        bw.write_bit(opts.start_skip != 0u ? 1u : 0u);
        if (opts.start_skip != 0u) {
            bw.write_bits(opts.start_skip, 10u);
        }
        bw.write_bit(opts.end_skip != 0u ? 1u : 0u);
        if (opts.end_skip != 0u) {
            bw.write_bits(opts.end_skip, 10u);
        }
    } else {
        bw.write_bit(0u);
    }

    auto len_to_code = [](uint32_t len) -> uint32_t {
        switch (len) {
            case 512u:
                return 0u;
            case 256u:
                return 1u;
            case 128u:
                return 2u;
            default:
                return 3u;
        }
    };

    // 2. Subframe layout.
    bool all_same = true;
    for (uint32_t l : opts.subframe_lens) {
        if (l != opts.subframe_lens[0]) {
            all_same = false;
        }
    }
    if (all_same) {
        bw.write_bit(1u);  // fixed length
        bw.write_bits(len_to_code(opts.subframe_lens[0]), 2u);
    } else {
        bw.write_bit(0u);  // variable length
        if (opts.channels > 1u) {
            bw.write_bit(1u);  // tile_len_same
        }
        for (uint32_t l : opts.subframe_lens) {
            bw.write_bits(len_to_code(l), 2u);
        }
    }

    // 3. Per-subframe payload.
    uint32_t offset = 0u;
    uint32_t prev_len = opts.prev_subframe_len;
    for (size_t s = 0u; s < opts.subframe_lens.size(); ++s) {
        const uint32_t cur_len = opts.subframe_lens[s];
        const uint32_t next_len = (s + 1u < opts.subframe_lens.size()) ? opts.subframe_lens[s + 1u]
                                                                       : opts.next_frame_first_len;

        // Forward MDCT per channel.
        std::array<std::vector<float>, 2> spec;
        for (uint32_t c = 0u; c < opts.channels; ++c) {
            mdct_forward(signals[c], frame_start + static_cast<int64_t>(offset), prev_len, cur_len, next_len,
                         spec[c]);
        }

        // Coded-channel bitmask (all coded).
        for (uint32_t c = 0u; c < opts.channels; ++c) {
            bw.write_bit(1u);
        }

        // Stereo decorrelation.
        if (opts.channels == 2u) {
            bw.write_bit(opts.use_decorrelation ? 1u : 0u);
            if (opts.use_decorrelation) {
                bw.write_bit(opts.use_default_ms ? 1u : 0u);
                if (opts.use_default_ms) {
                    constexpr float kInvSqrt2 = 0.7071067811865475244f;
                    for (uint32_t k = 0u; k < cur_len; ++k) {
                        const float l = spec[0][k];
                        const float r = spec[1][k];
                        spec[0][k] = (l + r) * kInvSqrt2;
                        spec[1][k] = (l - r) * kInvSqrt2;
                    }
                } else {
                    bw.write_bit(opts.givens_negate ? 1u : 0u);
                    bw.write_bits(opts.givens_angle_idx, 6u);
                    const double theta = (static_cast<double>(opts.givens_angle_idx) + 0.5) * (kTestPi / 64.0);
                    const float cs = static_cast<float>(std::cos(theta));
                    const float sn = static_cast<float>(std::sin(theta));
                    for (uint32_t k = 0u; k < cur_len; ++k) {
                        const float l = spec[0][k];
                        const float r = spec[1][k];
                        const float u = cs * l + sn * r;
                        const float v = -sn * l + cs * r;
                        spec[0][k] = u;
                        spec[1][k] = opts.givens_negate ? -v : v;
                    }
                }
            }
        }

        // Auto-select quantization step so the subframe peak fits cleanly inside [-8192, 8192]
        // without clipping (exercising large quant_step values when needed).
        float peak_abs = 0.0f;
        for (uint32_t c = 0u; c < opts.channels; ++c) {
            for (float v : spec[c]) {
                peak_abs = std::max(peak_abs, std::fabs(v));
            }
        }
        int32_t q_step = opts.quant_step;
        if (peak_abs > 1.0f) {
            const float needed_step = peak_abs / 4096.0f;
            const int32_t needed_q = opts.sf_base + static_cast<int32_t>(std::ceil(20.0f * std::log10(needed_step)));
            q_step = std::max(q_step, needed_q);
        }

        // Quantization step.
        write_quant_step(bw, q_step);
        if (opts.channels > 1u) {
            for (uint32_t c = 0u; c < opts.channels; ++c) {
                bw.write_bit(0u);  // no per-channel modifier
            }
        }

        std::vector<uint32_t> sfb_offsets;
        const uint32_t num_bands = test_sfb_count(opts.sample_rate, cur_len, sfb_offsets);
        const float step_size = std::pow(10.0f, static_cast<float>(q_step - opts.sf_base) * 0.05f);

        for (uint32_t c = 0u; c < opts.channels; ++c) {
            const bool transmit = !(opts.reuse_sf_after_first && s > 0u);
            bw.write_bit(transmit ? 1u : 0u);
            if (transmit) {
                bw.write_bit(0u);  // absolute + Huffman delta
                bw.write_bits(static_cast<uint32_t>(opts.sf_base), 7u);
                for (uint32_t b = 1u; b < num_bands; ++b) {
                    write_sf_delta(bw, 0);
                }
            }
            std::vector<int32_t> q(cur_len, 0);
            for (uint32_t k = 0u; k < cur_len; ++k) {
                const float scaled = spec[c][k] / step_size;
                q[k] = static_cast<int32_t>(std::lround(std::clamp(scaled, -16384.0f, 16384.0f)));
            }
            write_spectral_coeffs(bw, q, opts.exercise_vec_stages);
        }

        prev_len = cur_len;
        offset += cur_len;
    }

    // 4. End-of-frame marker.
    bw.write_bit(1u);
    return bw.finalize_frame();
}

double cosine_similarity(const std::vector<float>& ref, const std::vector<int16_t>& got,
                         size_t start_ref, size_t start_got, size_t count, uint32_t stride, uint32_t ch) {
    double dot = 0.0, nr = 0.0, ng = 0.0;
    for (size_t i = 0u; i < count; ++i) {
        const double r = static_cast<double>(ref[start_ref + i]);
        const double g = static_cast<double>(got[(start_got + i) * stride + ch]);
        dot += r * g;
        nr += r * r;
        ng += g * g;
    }
    if (nr == 0.0 || ng == 0.0) {
        return 0.0;
    }
    return dot / (std::sqrt(nr) * std::sqrt(ng));
}

void test_wmapro_mono_and_subframe_layouts() {
    constexpr uint32_t kFrames = 6u;
    constexpr uint32_t kTotalSamples = (kFrames + 1u) * x::kFrameSamples;
    std::array<std::vector<float>, 2> sig;
    sig[0].resize(kTotalSamples);
    for (uint32_t i = 0u; i < kTotalSamples; ++i) {
        sig[0][i] = 14000.0f * static_cast<float>(std::sin(2.0 * kTestPi * 1000.0 * i / 48000.0));
    }

    // Test 1: Single 512-sample subframe across 6 consecutive frames (with vec4/vec2 stages exercised).
    {
        auto dec = x::make_wmapro_decoder();
        std::vector<int16_t> decoded(kFrames * x::kFrameSamples, 0);
        EncodeFrameOpts opts;
        opts.channels = 1u;
        opts.sample_rate = 48000u;
        opts.subframe_lens = {512u};
        opts.exercise_vec_stages = true;
        for (uint32_t f = 0u; f < kFrames; ++f) {
            const x::Frame frame = encode_wmapro_frame(sig, static_cast<int64_t>(f * x::kFrameSamples), opts);
            CHECK(dec->decode(frame, 1u, 48000u, decoded.data() + f * x::kFrameSamples));
        }
        // After 256-sample overlap-add ramp-in, output at [512..2559] matches sig[256..2303].
        const double corr = cosine_similarity(sig[0], decoded, 256u, 512u, 2048u, 1u, 0u);
        CHECK(corr > 0.995);
        // Check frame boundary continuity: max step at 511->512 and 1023->1024 matches sine slope.
        CHECK(std::abs(static_cast<int>(decoded[1024]) - static_cast<int>(decoded[1023])) < 3500);
    }

    // Test 2: 4x128 subframes (with scale-factor reuse across subframes).
    {
        auto dec = x::make_wmapro_decoder();
        std::vector<int16_t> decoded(kFrames * x::kFrameSamples, 0);
        EncodeFrameOpts opts;
        opts.channels = 1u;
        opts.sample_rate = 44100u;
        opts.subframe_lens = {128u, 128u, 128u, 128u};
        opts.prev_subframe_len = 128u;
        opts.next_frame_first_len = 128u;
        opts.reuse_sf_after_first = true;
        for (uint32_t f = 0u; f < kFrames; ++f) {
            const x::Frame frame = encode_wmapro_frame(sig, static_cast<int64_t>(f * x::kFrameSamples), opts);
            CHECK(dec->decode(frame, 1u, 44100u, decoded.data() + f * x::kFrameSamples));
        }
        const double corr = cosine_similarity(sig[0], decoded, 256u, 512u, 2048u, 1u, 0u);
        CHECK(corr > 0.99);
    }

    // Test 3: Mixed variable subframes (256 + 128 + 128) + transition from 512, with SF resampling.
    {
        auto dec = x::make_wmapro_decoder();
        std::vector<int16_t> decoded(kFrames * x::kFrameSamples, 0);
        for (uint32_t f = 0u; f < kFrames; ++f) {
            EncodeFrameOpts opts;
            opts.channels = 1u;
            opts.sample_rate = 32000u;
            if (f == 0u) {
                opts.subframe_lens = {512u};
                opts.prev_subframe_len = 512u;
                opts.next_frame_first_len = 256u;
            } else {
                opts.subframe_lens = {256u, 128u, 128u};
                opts.prev_subframe_len = (f == 1u) ? 512u : 128u;
                opts.next_frame_first_len = 256u;
                opts.reuse_sf_after_first = true;  // forces 256 -> 128 SF resampling
            }
            const x::Frame frame = encode_wmapro_frame(sig, static_cast<int64_t>(f * x::kFrameSamples), opts);
            CHECK(dec->decode(frame, 1u, 32000u, decoded.data() + f * x::kFrameSamples));
        }
        const double corr = cosine_similarity(sig[0], decoded, 256u, 512u, 2048u, 1u, 0u);
        CHECK(corr > 0.99);
    }

    // Test 4: Start/end skip raw samples prefix on a 24 kHz frame.
    {
        auto dec = x::make_wmapro_decoder();
        std::vector<int16_t> pcm(x::kFrameSamples, 0);
        EncodeFrameOpts opts;
        opts.channels = 1u;
        opts.sample_rate = 24000u;
        opts.subframe_lens = {64u, 64u, 64u, 64u, 64u, 64u, 64u, 64u};
        opts.prev_subframe_len = 64u;
        opts.next_frame_first_len = 64u;
        // Prime overlap on frame 0, then apply start/end skip on frame 1.
        const x::Frame f0 = encode_wmapro_frame(sig, 0, opts);
        CHECK(dec->decode(f0, 1u, 24000u, pcm.data()));
        opts.start_skip = 64u;
        opts.end_skip = 48u;
        const x::Frame f1 = encode_wmapro_frame(sig, x::kFrameSamples, opts);
        CHECK(dec->decode(f1, 1u, 24000u, pcm.data()));
        for (uint32_t i = 0u; i < 64u; ++i) {
            CHECK(pcm[i] == 0);
        }
        for (uint32_t i = x::kFrameSamples - 48u; i < x::kFrameSamples; ++i) {
            CHECK(pcm[i] == 0);
        }
        int32_t mid_peak = 0;
        for (uint32_t i = 64u; i < x::kFrameSamples - 48u; ++i) {
            mid_peak = std::max<int32_t>(mid_peak, std::abs(static_cast<int32_t>(pcm[i])));
        }
        CHECK(mid_peak > 4000);
    }
}

void test_wmapro_stereo_decorrelation_and_pump() {
    constexpr uint32_t kFrames = 5u;
    constexpr uint32_t kTotalSamples = (kFrames + 1u) * x::kFrameSamples;
    std::array<std::vector<float>, 2> sig;
    sig[0].resize(kTotalSamples);
    sig[1].resize(kTotalSamples);
    for (uint32_t i = 0u; i < kTotalSamples; ++i) {
        sig[0][i] = 12000.0f * static_cast<float>(std::sin(2.0 * kTestPi * 500.0 * i / 48000.0));
        sig[1][i] = 12000.0f * static_cast<float>(std::sin(2.0 * kTestPi * 1500.0 * i / 48000.0));
    }

    // Default M/S orthogonal decorrelation and custom Givens rotation both preserve stereo separation.
    for (int mode = 0; mode < 2; ++mode) {
        auto dec = x::make_wmapro_decoder();
        std::vector<int16_t> decoded(kFrames * x::kFrameSamples * 2u, 0);
        EncodeFrameOpts opts;
        opts.channels = 2u;
        opts.sample_rate = 48000u;
        opts.subframe_lens = {256u, 256u};
        opts.prev_subframe_len = 256u;
        opts.next_frame_first_len = 256u;
        opts.use_decorrelation = true;
        opts.use_default_ms = (mode == 0);
        opts.givens_negate = (mode == 1);
        opts.givens_angle_idx = 19u;
        for (uint32_t f = 0u; f < kFrames; ++f) {
            const x::Frame frame = encode_wmapro_frame(sig, static_cast<int64_t>(f * x::kFrameSamples), opts);
            CHECK(dec->decode(frame, 2u, 48000u, decoded.data() + f * x::kFrameSamples * 2u));
        }
        // Left matches 500 Hz (not 1500 Hz); Right matches 1500 Hz (not 500 Hz).
        CHECK(cosine_similarity(sig[0], decoded, 256u, 512u, 1536u, 2u, 0u) > 0.99);
        CHECK(std::fabs(cosine_similarity(sig[1], decoded, 256u, 512u, 1536u, 2u, 0u)) < 0.05);
        CHECK(cosine_similarity(sig[1], decoded, 256u, 512u, 1536u, 2u, 1u) > 0.99);
        CHECK(std::fabs(cosine_similarity(sig[0], decoded, 256u, 512u, 1536u, 2u, 1u)) < 0.05);
    }

    // End-to-end pump() with enable(ctx, voice, nullptr) defaulting to make_wmapro_decoder() + loop reset.
    std::vector<x::Frame> frames;
    EncodeFrameOpts opts;
    opts.channels = 2u;
    opts.sample_rate = 48000u;
    opts.subframe_lens = {512u};
    opts.use_decorrelation = true;
    opts.use_default_ms = true;
    for (uint32_t f = 0u; f < 3u; ++f) {
        frames.push_back(encode_wmapro_frame(sig, static_cast<int64_t>(f * x::kFrameSamples), opts));
    }
    Stream s = pack_frames_to_stream(frames);
    std::vector<uint8_t> out(32u * x::kBlockBytes, 0u);
    Apu apu;
    apu.regions[0x10000] = &s.data;
    apu.regions[0x40000] = &out;

    uint32_t w[16] = {};
    x::Context c(w);
    w[0] = s.packets | (1u << 12) | (1u << 20);  // loop once, input 0 valid, ring 32
    w[1] = (3u << 12) | (3u << 27) | (1u << 29) | (1u << 31);  // loop end sf 3, 48 kHz, stereo, writable
    w[3] = s.starts[0];
    w[4] = s.starts[1] + 16u;
    w[5] = 0x10000;
    w[7] = 0x40000;

    x::Voice v;
    x::enable(c, v, nullptr);  // nullptr defaults to make_wmapro_decoder()
    uint32_t total_blocks = 0u;
    for (int iter = 0; iter < 8; ++iter) {
        total_blocks += x::pump(c, v, apu.map());
        x::advance_read(c, 32u);
    }
    CHECK(total_blocks > 0u);
    CHECK(v.stats.loops == 1u);
    CHECK(v.stats.frames == 5u && v.stats.frames_decoded == 5u);
    CHECK(v.stats.peak > 4000);

    // Explicit make_silence_decoder() still returns false and writes zeros.
    auto sil = x::make_silence_decoder();
    int16_t sil_pcm[x::kFrameSamples * 2u];
    std::fill(std::begin(sil_pcm), std::end(sil_pcm), int16_t{1234});
    CHECK(!sil->decode(frames[0], 2u, 48000u, sil_pcm));
    CHECK(std::all_of(std::begin(sil_pcm), std::end(sil_pcm), [](int16_t v) { return v == 0; }));
}

void test_wmapro_malformed_fuzz_and_overflow_safety() {
    auto dec = x::make_wmapro_decoder();
    int16_t pcm[x::kFrameSamples * 2u];

    auto expect_fail_silence = [&](const x::Frame& f, uint32_t ch, uint32_t rate) {
        std::fill(std::begin(pcm), std::end(pcm), int16_t{0x5A5A});
        CHECK(!dec->decode(f, ch, rate, pcm));
        if (ch >= 1u && ch <= 2u) {
            CHECK(std::all_of(pcm, pcm + size_t{x::kFrameSamples} * ch, [](int16_t s) { return s == 0; }));
        }
    };

    // Invalid parameters (channel count, sample rate, nullptr).
    x::Frame empty{};
    CHECK(!dec->decode(empty, 1u, 48000u, nullptr));
    CHECK(!dec->decode(empty, 0u, 48000u, pcm));
    CHECK(!dec->decode(empty, 3u, 48000u, pcm));
    expect_fail_silence(empty, 1u, 11025u);  // unsupported sample rate

    // Zero-length & truncated frames.
    expect_fail_silence(empty, 1u, 48000u);
    {
        x::Frame trunc;
        trunc.length = 15u;
        trunc.bits_gathered = 15u;
        trunc.bits = {0x00, 0x1E};
        expect_fail_silence(trunc, 1u, 48000u);
    }
    {
        x::Frame incomplete;
        incomplete.length = 256u;
        incomplete.bits_gathered = 128u;
        incomplete.complete = false;
        incomplete.bits.assign(32u, 0xAAu);
        expect_fail_silence(incomplete, 2u, 48000u);
    }

    // All-0xFF and all-0x00 frames.
    {
        x::Frame all_ff;
        all_ff.length = 512u;
        all_ff.bits_gathered = 512u;
        all_ff.bits.assign(64u, 0xFFu);
        expect_fail_silence(all_ff, 1u, 48000u);
        expect_fail_silence(all_ff, 2u, 48000u);

        x::Frame all_zero;
        all_zero.length = 512u;
        all_zero.bits_gathered = 512u;
        all_zero.bits.assign(64u, 0x00u);
        expect_fail_silence(all_zero, 1u, 48000u);
        expect_fail_silence(all_zero, 2u, 48000u);
    }

    // Seeded pseudo-random fuzz frames (256 frames, mono & stereo, lengths 16..3000 bits).
    uint32_t rng = 0x12345678u;
    auto next_u32 = [&]() {
        rng ^= rng << 13u;
        rng ^= rng >> 17u;
        rng ^= rng << 5u;
        return rng;
    };
    for (uint32_t i = 0u; i < 256u; ++i) {
        const uint32_t ch = (i & 1u) + 1u;
        const uint32_t rate = x::kSampleRates[i % 4u];
        const uint32_t bit_len = 64u + (next_u32() % 2048u);
        x::Frame fuzz;
        fuzz.length = bit_len;
        fuzz.bits_gathered = bit_len;
        fuzz.complete = true;
        fuzz.bits.resize((bit_len + 7u) / 8u);
        for (auto& byte : fuzz.bits) {
            byte = static_cast<uint8_t>(next_u32() & 0xFFu);
        }
        expect_fail_silence(fuzz, ch, rate);
    }

    // Overflow guards in read_frame / first_frame_from / continue_frame.
    uint8_t one_pkt[x::kPacketBytes];
    std::memset(one_pkt, 0xFF, sizeof(one_pkt));
    x::PacketHeader skip_max{0u, x::kNoFrame, 0u, 255u};
    x::write_packet_header(one_pkt, skip_max);
    CHECK(!x::first_frame_from(one_pkt, 0xFFFFFFFFu, 0u).has_value());
    CHECK(!x::read_frame(one_pkt, 0xFFFFFFFFu, 0u).has_value());
    CHECK(!x::read_frame(one_pkt, 1u, 0xFFFFFFF0u).has_value());
    x::Frame undersized;
    undersized.length = 1000u;
    undersized.bits_gathered = 15u;
    undersized.bits.resize(2u, 0u);  // too small for length=1000
    CHECK(!x::continue_frame(one_pkt, 1u, 100u, undersized));
}

void test_xapu_opus_lifecycle_and_mixing() {
    namespace op = playgta6::os::xapu_opus;

    // 1. RFC 6716 TOC inspection.
    op::PacketInfo info{};
    CHECK(!op::inspect_packet(nullptr, 4u, &info) && !info.valid);
    const uint8_t dummy = 0xFCu;
    CHECK(!op::inspect_packet(&dummy, 0u, &info) && !info.valid);

    // CELT 20 ms stereo, code 0 (1 frame): config 31, stereo=true, 960 samples @ 48 kHz.
    const uint8_t celt20_st[] = {0xFCu, 0xFFu, 0xFEu};
    CHECK(op::inspect_packet(celt20_st, sizeof(celt20_st), &info));
    CHECK(info.valid && info.config == 31u && info.stereo && info.frame_count_code == 0u &&
          info.frame_samples_48k == 960u);

    // SILK 60 ms mono, code 0: config 3 -> 2880 samples.
    const uint8_t silk60_mono[] = {static_cast<uint8_t>((3u << 3u) | 0u), 0x11u};
    CHECK(op::inspect_packet(silk60_mono, sizeof(silk60_mono), &info));
    CHECK(info.valid && info.config == 3u && !info.stereo && info.frame_samples_48k == 2880u);

    // Hybrid 10 ms stereo, code 1 (2 equal frames): even payload passes, odd payload fails.
    const uint8_t hyb10_c1_ok[] = {static_cast<uint8_t>((12u << 3u) | 0x04u | 1u), 0xAAu, 0xBBu};
    const uint8_t hyb10_c1_odd[] = {static_cast<uint8_t>((12u << 3u) | 0x04u | 1u), 0xAAu};
    CHECK(op::inspect_packet(hyb10_c1_ok, sizeof(hyb10_c1_ok), &info) && info.frame_samples_48k == 480u);
    CHECK(!op::inspect_packet(hyb10_c1_odd, sizeof(hyb10_c1_odd), &info));

    // Code 2 (2 variable frames): valid length vs truncated.
    const uint8_t celt20_c2_ok[] = {static_cast<uint8_t>((31u << 3u) | 2u), 2u, 0x10u, 0x20u, 0x30u};
    const uint8_t celt20_c2_bad[] = {static_cast<uint8_t>((31u << 3u) | 2u), 10u, 0x10u};
    CHECK(op::inspect_packet(celt20_c2_ok, sizeof(celt20_c2_ok), &info));
    CHECK(!op::inspect_packet(celt20_c2_bad, sizeof(celt20_c2_bad), &info));

    // Code 3 (arbitrary frame count): missing count byte, M=0, >120ms rejected; valid M=3 passes.
    const uint8_t c3_trunc[] = {0xFFu};
    const uint8_t c3_zero[] = {0xFFu, 0x00u};
    const uint8_t c3_over120ms[] = {0xFFu, 7u};  // 7 * 20ms = 140ms > 120ms
    const uint8_t c3_valid[] = {0xFFu, 3u, 0x00u};
    CHECK(!op::inspect_packet(c3_trunc, sizeof(c3_trunc), &info));
    CHECK(!op::inspect_packet(c3_zero, sizeof(c3_zero), &info));
    CHECK(!op::inspect_packet(c3_over120ms, sizeof(c3_over120ms), &info));
    CHECK(op::inspect_packet(c3_valid, sizeof(c3_valid), &info));

    // 2. VoiceDecoder lifecycle, stats, invalid args, PLAYGTA6_OPUS=OFF silence & max_frames cap.
    op::VoiceDecoder vd;
    CHECK(vd.decoded_packets() == 0u && vd.failed_packets() == 0u && vd.reset_count() == 0u);
    std::vector<float> pcm(op::kMaxOpusFrames * 2u, 1.0f);
    CHECK(vd.decode(nullptr, 3u, 48000u, 1u, pcm.data(), 960u) == -1);
    CHECK(vd.decode(celt20_st, 0u, 48000u, 1u, pcm.data(), 960u) == -1);
    CHECK(vd.decode(celt20_st, sizeof(celt20_st), 48000u, 0u, pcm.data(), 960u) == -1);
    CHECK(vd.decode(celt20_st, sizeof(celt20_st), 48000u, 3u, pcm.data(), 960u) == -1);
    CHECK(vd.decode(celt20_st, sizeof(celt20_st), 48000u, 1u, nullptr, 960u) == -1);
    CHECK(vd.decode(celt20_st, sizeof(celt20_st), 48000u, 1u, pcm.data(), 0u) == -1);
    CHECK(vd.failed_packets() == 6u);

    if (!op::available()) {
        std::fill(pcm.begin(), pcm.end(), 0.75f);
        CHECK(vd.decode(celt20_st, sizeof(celt20_st), 48000u, 2u, pcm.data(), 960u) == 960);
        CHECK(std::all_of(pcm.begin(), pcm.begin() + 960u * 2u, [](float v) { return v == 0.0f; }));
        CHECK(vd.decoded_packets() == 1u);

        // Oversized max_frames (> 5760) must be clamped safely to kMaxOpusFrames without OOB write.
        CHECK(vd.decode(celt20_st, sizeof(celt20_st), 48000u, 1u, pcm.data(), 20000u) ==
              static_cast<int>(op::kMaxOpusFrames));
        CHECK(vd.decoded_packets() == 2u);
    }
    vd.reset();
    vd.flush();
    CHECK(vd.reset_count() == 2u);
    vd.release();

    // 3. Multi-voice additive mixing & soft-clipping with NaN/Inf/overload sanitization.
    float mix[8] = {0.25f, -0.25f, std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(), 0.5f, -0.5f, 1e-35f, 0.0f};
    const float opus_src[8] = {0.25f, std::numeric_limits<float>::quiet_NaN(), 0.5f, 0.25f,
                               1.5f, -2.0f, 0.1f, -std::numeric_limits<float>::infinity()};
    const int16_t xma_src[8] = {16384, -16384, 0, 8192, 32767, -32768, 0, 16384};

    op::mix_pcm_additive(mix, opus_src, 8u, 1.0f);
    op::mix_int16_to_float_additive(mix, xma_src, 8u, 0.5f);
    // Index 0: 0.25 + 0.25 + 0.25 = 0.75 (below soft-clip knee 0.85 -> exact passthrough).
    // Index 1: -0.25 + 0 (NaN) - 0.25 = -0.50.
    // Index 2: 0 (NaN) + 0.5 + 0 = 0.50.
    // Index 3: 0 (Inf) + 0.25 + 0.125 = 0.375.
    // Index 4: 0.5 + 1.5 + ~0.5 = ~2.5 -> soft-clipped into (0.85, 1.0].
    // Index 5: -0.5 - 2.0 - 0.5 = -3.0 -> soft-clipped into [-1.0, -0.85).
    op::sanitize_and_soft_clip(mix, 8u);
    CHECK(std::fabs(mix[0] - 0.75f) < 1e-5f);
    CHECK(std::fabs(mix[1] - (-0.50f)) < 1e-5f);
    CHECK(std::fabs(mix[2] - 0.50f) < 1e-5f);
    CHECK(std::fabs(mix[3] - 0.375f) < 1e-5f);
    CHECK(mix[4] > 0.85f && mix[4] <= 1.0f);
    CHECK(mix[5] < -0.85f && mix[5] >= -1.0f);
    CHECK(std::fabs(mix[6] - 0.1f) < 1e-5f);
    CHECK(std::fabs(mix[7] - 0.25f) < 1e-5f);
    for (float v : mix) {
        CHECK(std::isfinite(v) && v >= -1.0f && v <= 1.0f);
    }

    // Extreme values & non-finite sanitize directly.
    float extreme[6] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        1e-38f,
        1e10f,
        -1e10f,
    };
    op::sanitize_and_soft_clip(extreme, 6u);
    CHECK(extreme[0] == 0.0f && extreme[1] == 0.0f && extreme[2] == 0.0f && extreme[3] == 0.0f);
    CHECK(extreme[4] == 1.0f && extreme[5] == -1.0f);
}

}  // namespace

// Public decoder API benchmark; fixture encoding and PCM hashing are outside timing.
int benchmark_wmapro(int argc, char** argv) {
    if (argc != 6) return 2;
    const uint32_t n = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
    const uint32_t channels = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
    const uint32_t frames = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
    const std::string kind = argv[5];
    if ((n != 64 && n != 128 && n != 256 && n != 512) || channels < 1 || channels > 2 ||
        frames < 1 || frames > 100000 || (kind != "nonzero" && kind != "silence" && kind != "malformed")) return 2;
    std::array<std::vector<float>, 2> signal;
    uint32_t seed = 0x4817u;
    for (uint32_t c = 0; c < channels; ++c) {
        signal[c].resize(3 * x::kFrameSamples);
        for (float& v : signal[c]) {
            seed = seed * 1664525u + 1013904223u;
            v = static_cast<float>(static_cast<int32_t>(seed >> 16) - 32768) * 0.25f;
        }
    }
    EncodeFrameOpts opts;
    opts.channels = channels;
    opts.subframe_lens.assign(x::kFrameSamples / n, n);
    opts.prev_subframe_len = n;
    opts.next_frame_first_len = n;
    opts.exercise_vec_stages = true;
    const x::Frame frame = kind == "malformed" ? x::Frame{} : encode_wmapro_frame(signal, x::kFrameSamples, opts);
    auto decoder = kind == "silence" ? x::make_silence_decoder() : x::make_wmapro_decoder();
    const bool expected_ok = kind == "nonzero";
    std::vector<int16_t> first(x::kFrameSamples * channels);
    std::vector<int16_t> pcm(size_t{frames} * x::kFrameSamples * channels);
    const auto cold_cpu_start = std::clock();
    const auto cold_start = std::chrono::steady_clock::now();
    const bool first_ok = decoder->decode(frame, channels, 48000, first.data());
    const double cold_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - cold_start).count();
    const double cold_cpu_us = 1000000.0 * (std::clock() - cold_cpu_start) / CLOCKS_PER_SEC;
    CHECK(first_ok == expected_ok);
    const auto first_decode_pcm = first;
    for (int i = 0; i < 3; ++i) CHECK(decoder->decode(frame, channels, 48000, first.data()) == expected_ok);
    decoder->reset();
    const auto cpu_start = std::clock();
    const auto start = std::chrono::steady_clock::now();
    bool flags_ok = true;
    for (uint32_t f = 0; f < frames; ++f) {
        flags_ok = (decoder->decode(frame, channels, 48000, pcm.data() + size_t{f} * x::kFrameSamples * channels) == expected_ok) && flags_ok;
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const double cpu_ms = 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
    const auto hash_pcm = [](const std::vector<int16_t>& data) {
        uint64_t hash = 0xcbf29ce484222325ull;
        for (int16_t s : data) {
            hash = (hash ^ static_cast<uint8_t>(s)) * 0x100000001b3ull;
            hash = (hash ^ static_cast<uint8_t>(static_cast<uint16_t>(s) >> 8)) * 0x100000001b3ull;
        }
        return hash;
    };
    const bool nonzero = std::any_of(pcm.begin(), pcm.end(), [](int16_t v) { return v != 0; });
    CHECK(flags_ok && nonzero == expected_ok);
    std::printf("{\"subframe\":%u,\"channels\":%u,\"frames\":%u,\"kind\":\"%s\",\"decoded_seconds\":%.9f,"
                "\"wall_ms\":%.6f,\"cpu_ms\":%.6f,\"first_decode_us\":%.6f,\"first_decode_cpu_us\":%.6f,"
                "\"pcm_hash\":\"%016llx\",\"first_pcm_hash\":\"%016llx\",\"correct\":%s}\n",
                n, channels, frames, kind.c_str(), static_cast<double>(frames) * x::kFrameSamples / 48000,
                wall_ms, cpu_ms, cold_us, cold_cpu_us, static_cast<unsigned long long>(hash_pcm(pcm)),
                static_cast<unsigned long long>(hash_pcm(first_decode_pcm)), g_failures == 0 ? "true" : "false");
    return g_failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench-wmapro") return benchmark_wmapro(argc, argv);
    test_packet_header();
    test_frame_walk();
    test_padding_and_skip();
    test_frame_across_buffers();
    test_ring_arithmetic();
    test_pump_mono();
    test_pump_stereo_switch_and_span();
    test_pump_loop();
    test_wmapro_mono_and_subframe_layouts();
    test_wmapro_stereo_decorrelation_and_pump();
    test_wmapro_malformed_fuzz_and_overflow_safety();
    test_xapu_opus_lifecycle_and_mixing();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_xma_decoder: ok\n");
    return 0;
}
