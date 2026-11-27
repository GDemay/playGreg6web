// XMA2 voices: see xma_decoder.h.

#include "os/xma_decoder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>

namespace playgta6::os::xma {

namespace {

constexpr uint32_t kPayloadBits = kPacketBits - kPacketHeaderBits;
constexpr uint32_t kPacketAlignMask = kPacketBytes - 1;
constexpr uint32_t kMaxAddressablePackets = std::numeric_limits<uint32_t>::max() / kPacketBits;

uint32_t load_be32(const uint8_t* p) {
    return (uint32_t{p[0]} << 24) | (uint32_t{p[1]} << 16) | (uint32_t{p[2]} << 8) | uint32_t{p[3]};
}

// Walks the payload bits of a buffer's packets in stream order: past each packet's header, and
// past the packets of other streams a header says to skip.
class PacketWalker {
public:
    PacketWalker(const uint8_t* data, uint32_t packets, uint32_t at) : data_(data), packets_(packets) {
        packet_ = at / kPacketBits;
        offset_ = std::max(at % kPacketBits, kPacketHeaderBits);
    }

    // Moves off the end of a finished packet; false when the buffer has no packet left.
    bool settle() {
        while (packet_ < packets_ && offset_ >= kPacketBits) {
            const uint32_t skip = parse_packet_header(data_ + size_t{packet_} * kPacketBytes).skip_packets;
            const uint64_t next = uint64_t{packet_} + 1u + uint64_t{skip};
            if (next >= packets_) {
                packet_ = packets_;
                offset_ = kPacketHeaderBits;
                return false;
            }
            packet_ = static_cast<uint32_t>(next);
            offset_ = kPacketHeaderBits;
        }
        return packet_ < packets_;
    }
    bool has_bits() { return settle(); }
    uint32_t left_in_packet() const { return offset_ < kPacketBits ? (kPacketBits - offset_) : 0u; }
    bool position_fits_u32() const {
        return uint64_t{packet_} * kPacketBits + offset_ <= std::numeric_limits<uint32_t>::max();
    }
    uint32_t position() const { return packet_ * kPacketBits + offset_; }
    uint32_t packet() const { return packet_; }

    // One bit, MSB first; call only after has_bits().
    uint32_t bit() {
        const uint8_t byte = data_[size_t{packet_} * kPacketBytes + offset_ / 8];
        const uint32_t b = (byte >> (7 - offset_ % 8)) & 1u;
        ++offset_;
        return b;
    }
    // Up to 32 bits within the current packet, without moving.
    uint32_t peek(uint32_t count) const {
        uint32_t v = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t o = offset_ + i;
            const uint8_t byte = data_[size_t{packet_} * kPacketBytes + o / 8];
            v = (v << 1) | ((byte >> (7 - o % 8)) & 1u);
        }
        return v;
    }

private:
    const uint8_t* data_;
    uint32_t packets_;
    uint32_t packet_ = 0;
    uint32_t offset_ = 0;
};

void put_bit(Frame& f, uint32_t b) {
    const size_t byte_idx = f.bits_gathered / 8;
    if (byte_idx < f.bits.size()) {
        if (b != 0) {
            f.bits[byte_idx] |= static_cast<uint8_t>(0x80u >> (f.bits_gathered % 8));
        }
    }
    ++f.bits_gathered;
}

class SilenceDecoder final : public FrameDecoder {
public:
    bool decode(const Frame&, uint32_t channels, uint32_t, int16_t* pcm) override {
        if (pcm != nullptr && channels >= 1u && channels <= 2u) {
            std::fill(pcm, pcm + size_t{kFrameSamples} * channels, int16_t{0});
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Dependency-free WMA Pro / XMA2 subframe decoder
// ---------------------------------------------------------------------------

constexpr double kPi = 3.14159265358979323846;
constexpr uint32_t kMaxChannels = 2u;
constexpr uint32_t kMinSubframeLen = 64u;
constexpr uint32_t kMaxSubframeLen = kFrameSamples;  // 512
constexpr uint32_t kMaxSubframes = kMaxSubframeLen / kMinSubframeLen;  // 8
constexpr uint32_t kMaxSfBands = 26u;
constexpr uint32_t kNumSampleRates = 4u;
constexpr uint32_t kNumSubframeSizes = 4u;  // 64, 128, 256, 512

// Bounds-checked MSB-first bitstream reader over Frame::bits.
class BitReader {
public:
    BitReader(const uint8_t* data, uint32_t start_bit, uint32_t bit_limit)
        : data_(data), pos_(start_bit), limit_(bit_limit) {
        if (data_ == nullptr || pos_ > limit_) {
            pos_ = limit_;
            error_ = true;
        }
    }

    uint32_t bits_left() const { return pos_ < limit_ ? (limit_ - pos_) : 0u; }
    bool ok() const { return !error_; }

    bool read_bit(uint32_t& out) {
        if (error_ || pos_ >= limit_) {
            error_ = true;
            out = 0u;
            return false;
        }
        const uint8_t byte = data_[pos_ >> 3u];
        out = (byte >> (7u - (pos_ & 7u))) & 1u;
        ++pos_;
        return true;
    }

    bool read_bits(uint32_t n, uint32_t& out) {
        if (n == 0u) {
            out = 0u;
            return !error_;
        }
        if (error_ || n > 32u || n > bits_left()) {
            error_ = true;
            out = 0u;
            return false;
        }
        uint32_t v = 0u;
        for (uint32_t i = 0u; i < n; ++i) {
            const uint8_t byte = data_[pos_ >> 3u];
            v = (v << 1u) | ((byte >> (7u - (pos_ & 7u))) & 1u);
            ++pos_;
        }
        out = v;
        return true;
    }

private:
    const uint8_t* data_ = nullptr;
    uint32_t pos_ = 0u;
    uint32_t limit_ = 0u;
    bool error_ = false;
};

int sample_rate_index(uint32_t sample_rate) {
    for (uint32_t i = 0u; i < kNumSampleRates; ++i) {
        if (kSampleRates[i] == sample_rate) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int subframe_size_index(uint32_t len) {
    switch (len) {
        case 64u:
            return 0;
        case 128u:
            return 1;
        case 256u:
            return 2;
        case 512u:
            return 3;
        default:
            return -1;
    }
}

struct ScaleFactorBandTable {
    uint32_t num_bands = 0u;
    std::array<uint16_t, kMaxSfBands + 1u> offsets{};
};

// Critical-band (Bark-scale) cutoffs in Hz matching WMA Pro (25 bands).
constexpr uint32_t kCriticalFreqsHz[25] = {
    100u,  200u,  300u,  400u,  510u,  630u,  770u,  920u,  1080u,
    1270u, 1480u, 1720u, 2000u, 2320u, 2700u, 3150u, 3700u, 4400u,
    5300u, 6400u, 7700u, 9500u, 12000u, 15500u, 24500u,
};

ScaleFactorBandTable build_sfb_table(uint32_t sample_rate, uint32_t subframe_len) {
    ScaleFactorBandTable t{};
    t.offsets[0] = 0u;
    uint32_t band = 0u;
    const uint32_t step = subframe_len >= 128u ? 4u : 2u;
    uint32_t prev = 0u;
    for (uint32_t freq : kCriticalFreqsHz) {
        // Bin index = round(freq * 2 * subframe_len / sample_rate), quantized to `step`.
        uint32_t bin = static_cast<uint32_t>(
            (uint64_t{freq} * 2u * subframe_len + (sample_rate / 2u)) / sample_rate);
        bin = ((bin + step / 2u) / step) * step;
        if (bin <= prev) {
            continue;
        }
        if (bin >= subframe_len) {
            break;
        }
        t.offsets[++band] = static_cast<uint16_t>(bin);
        prev = bin;
        if (band + 1u >= kMaxSfBands) {
            break;
        }
    }
    if (prev < subframe_len) {
        t.offsets[++band] = static_cast<uint16_t>(subframe_len);
    }
    t.num_bands = band;
    return t;
}

const ScaleFactorBandTable& get_sfb_table(uint32_t sample_rate, uint32_t subframe_len) {
    static const auto kTables = [] {
        std::array<std::array<ScaleFactorBandTable, kNumSubframeSizes>, kNumSampleRates> tables{};
        constexpr uint32_t kSizes[kNumSubframeSizes] = {64u, 128u, 256u, 512u};
        for (uint32_t r = 0u; r < kNumSampleRates; ++r) {
            for (uint32_t s = 0u; s < kNumSubframeSizes; ++s) {
                tables[r][s] = build_sfb_table(kSampleRates[r], kSizes[s]);
            }
        }
        return tables;
    }();
    const int ri = sample_rate_index(sample_rate);
    const int si = subframe_size_index(subframe_len);
    return kTables[ri >= 0 ? ri : 3][si >= 0 ? si : 3];
}

// Precomputed sine transition windows: sin(pi * (n + 0.5) / (2 * W)) for W in {64, 128, 256, 512}.
const float* get_sine_window(uint32_t w) {
    static const auto kWindows = [] {
        struct Windows {
            std::array<float, 64> w64{};
            std::array<float, 128> w128{};
            std::array<float, 256> w256{};
            std::array<float, 512> w512{};
        } win;
        auto fill = [](float* dst, uint32_t n) {
            const double scale = kPi / (2.0 * static_cast<double>(n));
            for (uint32_t i = 0u; i < n; ++i) {
                dst[i] = static_cast<float>(std::sin((static_cast<double>(i) + 0.5) * scale));
            }
        };
        fill(win.w64.data(), 64u);
        fill(win.w128.data(), 128u);
        fill(win.w256.data(), 256u);
        fill(win.w512.data(), 512u);
        return win;
    }();
    switch (w) {
        case 64u:
            return kWindows.w64.data();
        case 128u:
            return kWindows.w128.data();
        case 256u:
            return kWindows.w256.data();
        default:
            return kWindows.w512.data();
    }
}

// In-place radix-2 complex FFT of length M (power of 2, M <= 256), computing:
//   Y[m] = sum_{p=0}^{M-1} X[p] * exp(+i * 2 * pi * m * p / M)
void fft_forward_positive(float* re, float* im, uint32_t m) {
    // Bit-reversal permutation.
    uint32_t j = 0u;
    for (uint32_t i = 1u; i < m; ++i) {
        uint32_t bit = m >> 1u;
        while ((j & bit) != 0u) {
            j ^= bit;
            bit >>= 1u;
        }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    // Cooley-Tukey butterflies.
    for (uint32_t len = 2u; len <= m; len <<= 1u) {
        const uint32_t half = len >> 1u;
        const double ang = (2.0 * kPi) / static_cast<double>(len);
        const float wlen_r = static_cast<float>(std::cos(ang));
        const float wlen_i = static_cast<float>(std::sin(ang));
        for (uint32_t i = 0u; i < m; i += len) {
            float wr = 1.0f;
            float wi = 0.0f;
            for (uint32_t k = 0u; k < half; ++k) {
                const uint32_t u = i + k;
                const uint32_t v = u + half;
                const float vr = re[v] * wr - im[v] * wi;
                const float vi = re[v] * wi + im[v] * wr;
                re[v] = re[u] - vr;
                im[v] = im[u] - vi;
                re[u] = re[u] + vr;
                im[u] = im[u] + vi;
                const float next_wr = wr * wlen_r - wi * wlen_i;
                wi = wr * wlen_i + wi * wlen_r;
                wr = next_wr;
            }
        }
    }
}

// Fast IMDCT of length N in {64, 128, 256, 512} via N/2-point complex FFT DCT-IV:
//   y[n] = (2/N) * sum_{k=0}^{N-1} X[k] * cos((pi/N) * (n + 0.5 + N/2) * (k + 0.5)), n = 0..2N-1.
void imdct_fast(const float* in, float* out, uint32_t n) {
    const uint32_t m = n >> 1u;  // N / 2 <= 256
    float re[kMaxSubframeLen / 2u];
    float im[kMaxSubframeLen / 2u];
    // The angles depend only on the four supported lengths. Preserve the original
    // double expressions and float rounding, once per process (thread-safe initialization).
    static const auto tables = [] {
        std::array<std::array<std::array<float, 2>, kMaxSubframeLen>, 4> result{};
        for (uint32_t index = 0; index < 4; ++index) {
            const uint32_t size = 64u << index;
            const double step = kPi / static_cast<double>(size);
            for (uint32_t p = 0; p < size / 2; ++p) {
                const double pre = step * static_cast<double>(p);
                const double post = step * (static_cast<double>(p) + 0.25);
                result[index][p] = {static_cast<float>(std::cos(pre)), static_cast<float>(std::sin(pre))};
                result[index][size / 2 + p] = {static_cast<float>(std::cos(post)), static_cast<float>(std::sin(post))};
            }
        }
        return result;
    }();
    const auto& twiddles = tables[n == 64 ? 0 : n == 128 ? 1 : n == 256 ? 2 : 3];

    // Pre-twiddle: u[p] = (X[2p] - i * X[N-1-2p]) * exp(+i * pi * p / N).
    for (uint32_t p = 0u; p < m; ++p) {
        const float xr = in[2u * p];
        const float xi = -in[n - 1u - 2u * p];
        const float c = twiddles[p][0];
        const float s = twiddles[p][1];
        re[p] = xr * c - xi * s;
        im[p] = xr * s + xi * c;
    }

    fft_forward_positive(re, im, m);

    // Post-twiddle to obtain DCT-IV D[0..N-1]:
    //   V[k] = exp(+i * pi * (k + 0.25) / N) * FFT[k]
    //   D[2k] = Re(V[k]), D[N - 1 - 2k] = Im(V[k]).
    float dct[kMaxSubframeLen];
    const float scale = 2.0f / static_cast<float>(n);
    for (uint32_t k = 0u; k < m; ++k) {
        const float c = twiddles[m + k][0];
        const float s = twiddles[m + k][1];
        const float vr = (re[k] * c - im[k] * s) * scale;
        const float vi = (re[k] * s + im[k] * c) * scale;
        dct[2u * k] = vr;
        dct[n - 1u - 2u * k] = vi;
    }

    // Fold DCT-IV into full 2*N IMDCT output using half-period symmetries.
    const uint32_t half = n >> 1u;
    for (uint32_t j = 0u; j < half; ++j) {
        const float a = dct[half + j];
        const float b = -dct[half - 1u - j];
        out[j] = a;
        out[n - 1u - j] = -a;
        out[n + j] = b;
        out[2u * n - 1u - j] = b;
    }
}

// Canonical Huffman delta decoder for scale factors (signed delta in [-127, 127]).
bool decode_sf_delta(BitReader& br, int32_t& delta) {
    uint32_t bit = 0u;
    if (!br.read_bit(bit)) {
        return false;
    }
    if (bit == 0u) {
        delta = 0;
        return true;
    }
    if (!br.read_bit(bit)) {
        return false;
    }
    if (bit == 0u) {
        // 10 + sign -> +/- 1
        uint32_t sign = 0u;
        if (!br.read_bit(sign)) {
            return false;
        }
        delta = sign != 0u ? -1 : 1;
        return true;
    }
    if (!br.read_bit(bit)) {
        return false;
    }
    if (bit == 0u) {
        // 110 + sign -> +/- 2
        uint32_t sign = 0u;
        if (!br.read_bit(sign)) {
            return false;
        }
        delta = sign != 0u ? -2 : 2;
        return true;
    }
    if (!br.read_bit(bit)) {
        return false;
    }
    if (bit == 0u) {
        // 1110 + 2-bit (mag 3..6) + sign
        uint32_t idx = 0u;
        uint32_t sign = 0u;
        if (!br.read_bits(2u, idx) || !br.read_bit(sign)) {
            return false;
        }
        const int32_t mag = static_cast<int32_t>(3u + idx);
        delta = sign != 0u ? -mag : mag;
        return true;
    }
    if (!br.read_bit(bit)) {
        return false;
    }
    if (bit == 0u) {
        // 11110 + 3-bit (mag 7..14) + sign
        uint32_t idx = 0u;
        uint32_t sign = 0u;
        if (!br.read_bits(3u, idx) || !br.read_bit(sign)) {
            return false;
        }
        const int32_t mag = static_cast<int32_t>(7u + idx);
        delta = sign != 0u ? -mag : mag;
        return true;
    }
    // 11111 + sign + 7-bit mag (1..127)
    uint32_t sign = 0u;
    uint32_t mag = 0u;
    if (!br.read_bit(sign) || !br.read_bits(7u, mag) || mag == 0u) {
        return false;
    }
    delta = sign != 0u ? -static_cast<int32_t>(mag) : static_cast<int32_t>(mag);
    return true;
}

bool decode_quant_step(BitReader& br, int32_t& out_q) {
    uint32_t q = 0u;
    if (!br.read_bits(7u, q)) {
        return false;
    }
    int32_t total = static_cast<int32_t>(q);
    uint32_t escapes = 0u;
    while (q == 127u) {
        if (escapes >= 2u || !br.read_bits(7u, q)) {
            return false;
        }
        total += static_cast<int32_t>(q);
        ++escapes;
    }
    out_q = total;
    return true;
}

// Decode quantized spectral coefficients into coeffs[0..subframe_len-1] via vector-4, vector-2,
// and run-level + escape VLC codes.
bool decode_spectral_coeffs(BitReader& br, int32_t* coeffs, uint32_t subframe_len) {
    std::fill(coeffs, coeffs + subframe_len, 0);
    uint32_t pos = 0u;

    // Stage 1: Vector-4 coding.
    uint32_t use_vec4 = 0u;
    if (!br.read_bit(use_vec4)) {
        return false;
    }
    if (use_vec4 != 0u) {
        for (;;) {
            uint32_t b0 = 0u;
            if (!br.read_bit(b0)) {
                return false;
            }
            if (b0 == 0u) {
                break;  // end of vec4
            }
            if (pos + 4u > subframe_len) {
                return false;
            }
            uint32_t b1 = 0u;
            if (!br.read_bit(b1)) {
                return false;
            }
            if (b1 == 0u) {
                // 10 -> (0, 0, 0, 0)
                pos += 4u;
                continue;
            }
            uint32_t b2 = 0u;
            if (!br.read_bit(b2)) {
                return false;
            }
            if (b2 == 0u) {
                // 110 + 2-bit lane + 1-bit sign -> single +/-1
                uint32_t lane = 0u;
                uint32_t sign = 0u;
                if (!br.read_bits(2u, lane) || !br.read_bit(sign)) {
                    return false;
                }
                coeffs[pos + lane] = sign != 0u ? -1 : 1;
                pos += 4u;
                continue;
            }
            uint32_t b3 = 0u;
            if (!br.read_bit(b3)) {
                return false;
            }
            if (b3 == 0u) {
                // 1110 + 4-bit nonzero mask + sign bits -> +/-1 at masked lanes
                uint32_t mask = 0u;
                if (!br.read_bits(4u, mask) || mask == 0u) {
                    return false;
                }
                for (uint32_t lane = 0u; lane < 4u; ++lane) {
                    if (((mask >> (3u - lane)) & 1u) != 0u) {
                        uint32_t sign = 0u;
                        if (!br.read_bit(sign)) {
                            return false;
                        }
                        coeffs[pos + lane] = sign != 0u ? -1 : 1;
                    }
                }
                pos += 4u;
            } else {
                // 1111 + 4 lanes of (2-bit mag, 3=7-bit escape+3) + sign if nonzero
                for (uint32_t lane = 0u; lane < 4u; ++lane) {
                    uint32_t mag = 0u;
                    if (!br.read_bits(2u, mag)) {
                        return false;
                    }
                    if (mag == 3u) {
                        uint32_t ext = 0u;
                        if (!br.read_bits(7u, ext)) {
                            return false;
                        }
                        mag = 3u + ext;
                    }
                    if (mag != 0u) {
                        uint32_t sign = 0u;
                        if (!br.read_bit(sign)) {
                            return false;
                        }
                        coeffs[pos + lane] = sign != 0u ? -static_cast<int32_t>(mag)
                                                        : static_cast<int32_t>(mag);
                    }
                }
                pos += 4u;
            }
        }
    }

    // Stage 2: Vector-2 coding.
    uint32_t use_vec2 = 0u;
    if (!br.read_bit(use_vec2)) {
        return false;
    }
    if (use_vec2 != 0u) {
        for (;;) {
            uint32_t b0 = 0u;
            if (!br.read_bit(b0)) {
                return false;
            }
            if (b0 == 0u) {
                break;  // end of vec2
            }
            if (pos + 2u > subframe_len) {
                return false;
            }
            uint32_t b1 = 0u;
            if (!br.read_bit(b1)) {
                return false;
            }
            if (b1 == 0u) {
                // 10 -> (0, 0)
                pos += 2u;
                continue;
            }
            uint32_t b2 = 0u;
            if (!br.read_bit(b2)) {
                return false;
            }
            if (b2 == 0u) {
                // 110 + 1-bit lane + 1-bit sign
                uint32_t lane = 0u;
                uint32_t sign = 0u;
                if (!br.read_bit(lane) || !br.read_bit(sign)) {
                    return false;
                }
                coeffs[pos + lane] = sign != 0u ? -1 : 1;
                pos += 2u;
                continue;
            }
            uint32_t b3 = 0u;
            if (!br.read_bit(b3)) {
                return false;
            }
            if (b3 == 0u) {
                // 1110 + 2 signs -> (+/-1, +/-1)
                uint32_t s0 = 0u;
                uint32_t s1 = 0u;
                if (!br.read_bit(s0) || !br.read_bit(s1)) {
                    return false;
                }
                coeffs[pos] = s0 != 0u ? -1 : 1;
                coeffs[pos + 1u] = s1 != 0u ? -1 : 1;
                pos += 2u;
            } else {
                // 1111 + 2 lanes of (2-bit mag, 3=7-bit escape+3) + sign if nonzero
                for (uint32_t lane = 0u; lane < 2u; ++lane) {
                    uint32_t mag = 0u;
                    if (!br.read_bits(2u, mag)) {
                        return false;
                    }
                    if (mag == 3u) {
                        uint32_t ext = 0u;
                        if (!br.read_bits(7u, ext)) {
                            return false;
                        }
                        mag = 3u + ext;
                    }
                    if (mag != 0u) {
                        uint32_t sign = 0u;
                        if (!br.read_bit(sign)) {
                            return false;
                        }
                        coeffs[pos + lane] = sign != 0u ? -static_cast<int32_t>(mag)
                                                        : static_cast<int32_t>(mag);
                    }
                }
                pos += 2u;
            }
        }
    }

    // Stage 3: Run-level + escape coding until EOB or subframe_len.
    while (pos < subframe_len) {
        uint32_t b0 = 0u;
        if (!br.read_bit(b0)) {
            return false;
        }
        if (b0 == 0u) {
            break;  // End-of-block: remaining bins are zero.
        }
        uint32_t run = 0u;
        uint32_t level = 1u;
        uint32_t sign = 0u;

        uint32_t b1 = 0u;
        if (!br.read_bit(b1)) {
            return false;
        }
        if (b1 == 0u) {
            // 10 + sign -> run=0, level=+/-1
            if (!br.read_bit(sign)) {
                return false;
            }
        } else {
            uint32_t b2 = 0u;
            if (!br.read_bit(b2)) {
                return false;
            }
            if (b2 == 0u) {
                // 110 + 2-bit (run-1) + sign -> run=1..4, level=+/-1
                uint32_t r = 0u;
                if (!br.read_bits(2u, r) || !br.read_bit(sign)) {
                    return false;
                }
                run = 1u + r;
            } else {
                uint32_t b3 = 0u;
                if (!br.read_bit(b3)) {
                    return false;
                }
                if (b3 == 0u) {
                    // 1110 + 3-bit run + 2-bit (level-2) + sign -> run=0..7, level=2..5
                    uint32_t r = 0u;
                    uint32_t l = 0u;
                    if (!br.read_bits(3u, r) || !br.read_bits(2u, l) || !br.read_bit(sign)) {
                        return false;
                    }
                    run = r;
                    level = 2u + l;
                } else {
                    uint32_t b4 = 0u;
                    if (!br.read_bit(b4)) {
                        return false;
                    }
                    if (b4 == 0u) {
                        // 11110 + 6-bit run + sign -> run=0..63, level=+/-1
                        if (!br.read_bits(6u, run) || !br.read_bit(sign)) {
                            return false;
                        }
                    } else {
                        // 11111 (escape): 9-bit run + 1-bit large + (7 or 14)-bit (level-1) + sign
                        uint32_t large = 0u;
                        uint32_t lvl_m1 = 0u;
                        if (!br.read_bits(9u, run) || !br.read_bit(large)) {
                            return false;
                        }
                        if (!br.read_bits(large != 0u ? 14u : 7u, lvl_m1) || !br.read_bit(sign)) {
                            return false;
                        }
                        level = 1u + lvl_m1;
                    }
                }
            }
        }
        if (run >= subframe_len - pos) {
            return false;
        }
        pos += run;
        coeffs[pos] = sign != 0u ? -static_cast<int32_t>(level) : static_cast<int32_t>(level);
        ++pos;
    }
    return true;
}

int16_t float_to_int16_safe(float v) {
    if (!std::isfinite(v)) {
        return 0;
    }
    const float clamped = std::clamp(v, -32768.0f, 32767.0f);
    const long rounded = std::lround(clamped);
    if (rounded < -32768L) {
        return INT16_MIN;
    }
    if (rounded > 32767L) {
        return INT16_MAX;
    }
    return static_cast<int16_t>(rounded);
}

class WmaProDecoder final : public FrameDecoder {
public:
    WmaProDecoder() { reset(); }

    void reset() override {
        for (auto& ch : channels_) {
            ch.overlap_buf.fill(0.0f);
            ch.prev_right.fill(0.0f);
            ch.prev_len = 0u;
            ch.saved_sf.fill(0);
            ch.saved_sf_len = 0u;
            ch.saved_sf_rate = 0u;
            ch.has_saved_sf = false;
        }
    }

    bool decode(const Frame& frame, uint32_t channels, uint32_t sample_rate, int16_t* pcm) override {
        if (pcm == nullptr) {
            return false;
        }
        if (channels < 1u || channels > kMaxChannels) {
            return false;
        }
        auto fail_silence = [&]() -> bool {
            std::fill(pcm, pcm + size_t{kFrameSamples} * channels, int16_t{0});
            return false;
        };

        if (sample_rate_index(sample_rate) < 0) {
            return fail_silence();
        }
        const size_t max_bytes = std::min<size_t>(frame.bits.size(),
                                                  std::numeric_limits<uint32_t>::max() / 8u);
        const uint32_t storage_bits = static_cast<uint32_t>(max_bytes * 8u);
        const uint32_t bit_limit = std::min({frame.bits_gathered, frame.length, storage_bits});
        if (!frame.complete || bit_limit <= kFrameLengthBits) {
            return fail_silence();
        }

        // Work on a copy of channel state so a malformed frame never corrupts persistent state.
        std::array<ChannelState, kMaxChannels> next_state = channels_;

        BitReader br(frame.bits.data(), kFrameLengthBits, bit_limit);

        // 1. Optional XMA2 skip-raw-samples prefix.
        uint32_t has_skip = 0u;
        uint32_t start_skip = 0u;
        uint32_t end_skip = 0u;
        if (!br.read_bit(has_skip)) {
            return fail_silence();
        }
        if (has_skip != 0u) {
            uint32_t has_start = 0u;
            uint32_t has_end = 0u;
            if (!br.read_bit(has_start)) {
                return fail_silence();
            }
            if (has_start != 0u && !br.read_bits(10u, start_skip)) {
                return fail_silence();
            }
            if (!br.read_bit(has_end)) {
                return fail_silence();
            }
            if (has_end != 0u && !br.read_bits(10u, end_skip)) {
                return fail_silence();
            }
            if (has_start == 0u && has_end == 0u) {
                return fail_silence();
            }
            if (start_skip > kFrameSamples || end_skip > kFrameSamples ||
                start_skip + end_skip > kFrameSamples) {
                return fail_silence();
            }
        }

        // 2. Per-channel subframe layout parsing (tile header).
        struct SubframeSpec {
            uint16_t offset = 0u;
            uint16_t len = 0u;
        };
        std::array<std::array<SubframeSpec, kMaxSubframes>, kMaxChannels> layout{};
        std::array<uint32_t, kMaxChannels> num_subframes{};

        uint32_t fixed_len_flag = 0u;
        if (!br.read_bit(fixed_len_flag)) {
            return fail_silence();
        }
        if (fixed_len_flag != 0u) {
            uint32_t len_code = 0u;
            if (!br.read_bits(2u, len_code)) {
                return fail_silence();
            }
            const uint32_t sf_len = kMaxSubframeLen >> len_code;
            const uint32_t count = kFrameSamples / sf_len;
            for (uint32_t c = 0u; c < channels; ++c) {
                num_subframes[c] = count;
                for (uint32_t i = 0u; i < count; ++i) {
                    layout[c][i] = SubframeSpec{static_cast<uint16_t>(i * sf_len),
                                                static_cast<uint16_t>(sf_len)};
                }
            }
        } else {
            uint32_t tile_len_same = 1u;
            if (channels > 1u) {
                if (!br.read_bit(tile_len_same)) {
                    return fail_silence();
                }
            }
            std::array<uint32_t, kMaxChannels> filled{};
            for (;;) {
                uint32_t min_pos = kFrameSamples;
                for (uint32_t c = 0u; c < channels; ++c) {
                    min_pos = std::min(min_pos, filled[c]);
                }
                if (min_pos == kFrameSamples) {
                    break;
                }
                std::array<bool, kMaxChannels> selected{};
                uint32_t num_selected = 0u;
                if (tile_len_same != 0u) {
                    for (uint32_t c = 0u; c < channels; ++c) {
                        if (filled[c] == min_pos) {
                            selected[c] = true;
                            ++num_selected;
                        }
                    }
                } else {
                    uint32_t at_min = 0u;
                    for (uint32_t c = 0u; c < channels; ++c) {
                        if (filled[c] == min_pos) {
                            ++at_min;
                        }
                    }
                    if (at_min == 1u) {
                        for (uint32_t c = 0u; c < channels; ++c) {
                            if (filled[c] == min_pos) {
                                selected[c] = true;
                                num_selected = 1u;
                            }
                        }
                    } else {
                        for (uint32_t c = 0u; c < channels; ++c) {
                            if (filled[c] == min_pos) {
                                uint32_t bit = 0u;
                                if (!br.read_bit(bit)) {
                                    return fail_silence();
                                }
                                if (bit != 0u) {
                                    selected[c] = true;
                                    ++num_selected;
                                }
                            }
                        }
                    }
                }
                if (num_selected == 0u) {
                    return fail_silence();
                }
                uint32_t len_code = 0u;
                if (!br.read_bits(2u, len_code)) {
                    return fail_silence();
                }
                const uint32_t sf_len = kMaxSubframeLen >> len_code;
                if (min_pos + sf_len > kFrameSamples) {
                    return fail_silence();
                }
                for (uint32_t c = 0u; c < channels; ++c) {
                    if (selected[c]) {
                        if (num_subframes[c] >= kMaxSubframes) {
                            return fail_silence();
                        }
                        layout[c][num_subframes[c]++] = SubframeSpec{static_cast<uint16_t>(min_pos),
                                                                     static_cast<uint16_t>(sf_len)};
                        filled[c] += sf_len;
                    }
                }
            }
            for (uint32_t c = 0u; c < channels; ++c) {
                if (filled[c] != kFrameSamples || num_subframes[c] == 0u) {
                    return fail_silence();
                }
            }
        }

        // 3. Decode subframes in chronological offset order.
        std::array<uint32_t, kMaxChannels> sf_idx{};
        for (;;) {
            uint32_t cur_offset = kFrameSamples;
            uint32_t cur_len = 0u;
            for (uint32_t c = 0u; c < channels; ++c) {
                if (sf_idx[c] < num_subframes[c]) {
                    const auto& sp = layout[c][sf_idx[c]];
                    if (sp.offset < cur_offset || (sp.offset == cur_offset && cur_len == 0u)) {
                        cur_offset = sp.offset;
                        cur_len = sp.len;
                    }
                }
            }
            if (cur_offset == kFrameSamples) {
                break;
            }

            // Collect channels sharing (cur_offset, cur_len).
            std::array<uint32_t, kMaxChannels> group{};
            uint32_t group_count = 0u;
            for (uint32_t c = 0u; c < channels; ++c) {
                if (sf_idx[c] < num_subframes[c]) {
                    const auto& sp = layout[c][sf_idx[c]];
                    if (sp.offset == cur_offset && sp.len == cur_len) {
                        group[group_count++] = c;
                        ++sf_idx[c];
                    }
                }
            }

            // Coded-channel bitmask for this subframe group.
            std::array<bool, kMaxChannels> coded{};
            uint32_t coded_count = 0u;
            for (uint32_t g = 0u; g < group_count; ++g) {
                uint32_t bit = 0u;
                if (!br.read_bit(bit)) {
                    return fail_silence();
                }
                coded[g] = (bit != 0u);
                if (coded[g]) {
                    ++coded_count;
                }
            }

            // Optional stereo channel decorrelation header (when 2 channels are both coded).
            bool use_decorrelation = false;
            bool use_default_ms = true;
            bool givens_negate = false;
            float givens_cos = 0.7071067811865475f;
            float givens_sin = 0.7071067811865475f;
            if (group_count == 2u && coded_count == 2u) {
                uint32_t decorr = 0u;
                if (!br.read_bit(decorr)) {
                    return fail_silence();
                }
                use_decorrelation = (decorr != 0u);
                if (use_decorrelation) {
                    uint32_t def_ms = 0u;
                    if (!br.read_bit(def_ms)) {
                        return fail_silence();
                    }
                    use_default_ms = (def_ms != 0u);
                    if (!use_default_ms) {
                        uint32_t neg = 0u;
                        uint32_t angle_idx = 0u;
                        if (!br.read_bit(neg) || !br.read_bits(6u, angle_idx)) {
                            return fail_silence();
                        }
                        givens_negate = (neg != 0u);
                        const double theta = (static_cast<double>(angle_idx) + 0.5) * (kPi / 64.0);
                        givens_cos = static_cast<float>(std::cos(theta));
                        givens_sin = static_cast<float>(std::sin(theta));
                    }
                }
            }

            // Quantization step decode.
            int32_t base_q = 0;
            std::array<int32_t, kMaxChannels> ch_q{};
            if (coded_count > 0u) {
                if (!decode_quant_step(br, base_q)) {
                    return fail_silence();
                }
                for (uint32_t g = 0u; g < group_count; ++g) {
                    ch_q[g] = base_q;
                    if (coded[g] && coded_count > 1u) {
                        uint32_t has_mod = 0u;
                        if (!br.read_bit(has_mod)) {
                            return fail_silence();
                        }
                        if (has_mod != 0u) {
                            uint32_t sign = 0u;
                            uint32_t mag = 0u;
                            if (!br.read_bit(sign) || !br.read_bits(4u, mag)) {
                                return fail_silence();
                            }
                            ch_q[g] += sign != 0u ? -static_cast<int32_t>(mag)
                                                  : static_cast<int32_t>(mag);
                        }
                    }
                }
            }

            const ScaleFactorBandTable& sfb = get_sfb_table(sample_rate, cur_len);
            std::array<std::array<float, kMaxSubframeLen>, kMaxChannels> spectrum{};

            // Decode scale factors and spectral coefficients for each coded channel.
            for (uint32_t g = 0u; g < group_count; ++g) {
                spectrum[g].fill(0.0f);
                if (!coded[g]) {
                    continue;
                }
                ChannelState& cs = next_state[group[g]];
                std::array<int32_t, kMaxSfBands> sf{};

                uint32_t transmit_sf = 0u;
                if (!br.read_bit(transmit_sf)) {
                    return fail_silence();
                }
                if (transmit_sf != 0u) {
                    uint32_t delta_from_prev = 0u;
                    if (!br.read_bit(delta_from_prev)) {
                        return fail_silence();
                    }
                    if (delta_from_prev == 0u) {
                        uint32_t sf0 = 0u;
                        if (!br.read_bits(7u, sf0)) {
                            return fail_silence();
                        }
                        sf[0] = static_cast<int32_t>(sf0);
                        for (uint32_t b = 1u; b < sfb.num_bands; ++b) {
                            int32_t d = 0;
                            if (!decode_sf_delta(br, d)) {
                                return fail_silence();
                            }
                            sf[b] = std::clamp(sf[b - 1u] + d, -64, 191);
                        }
                    } else {
                        if (!cs.has_saved_sf) {
                            return fail_silence();
                        }
                        std::array<int32_t, kMaxSfBands> resampled{};
                        resample_scale_factors(cs, sample_rate, cur_len, sfb, resampled);
                        for (uint32_t b = 0u; b < sfb.num_bands; ++b) {
                            int32_t d = 0;
                            if (!decode_sf_delta(br, d)) {
                                return fail_silence();
                            }
                            sf[b] = std::clamp(resampled[b] + d, -64, 191);
                        }
                    }
                    cs.saved_sf = sf;
                    cs.saved_sf_len = cur_len;
                    cs.saved_sf_rate = sample_rate;
                    cs.has_saved_sf = true;
                } else {
                    if (!cs.has_saved_sf) {
                        return fail_silence();
                    }
                    resample_scale_factors(cs, sample_rate, cur_len, sfb, sf);
                }

                // Decode quantized spectral coefficients.
                int32_t q_coeffs[kMaxSubframeLen];
                if (!decode_spectral_coeffs(br, q_coeffs, cur_len)) {
                    return fail_silence();
                }

                // Inverse quantization: val * 10^((quant_step - sf_band) / 20) with exponent clamp.
                for (uint32_t b = 0u; b < sfb.num_bands; ++b) {
                    const float exp_val = std::clamp(
                        static_cast<float>(ch_q[g] - sf[b]) * 0.05f, -12.0f, 12.0f);
                    const float scale = std::pow(10.0f, exp_val);
                    const uint32_t start = sfb.offsets[b];
                    const uint32_t end = sfb.offsets[b + 1u];
                    for (uint32_t k = start; k < end; ++k) {
                        spectrum[g][k] = static_cast<float>(q_coeffs[k]) * scale;
                    }
                }
            }

            // Inverse channel decorrelation.
            if (use_decorrelation && group_count == 2u && coded_count == 2u) {
                if (use_default_ms) {
                    constexpr float kInvSqrt2 = 0.7071067811865475244f;
                    for (uint32_t k = 0u; k < cur_len; ++k) {
                        const float m = spectrum[0][k];
                        const float s = spectrum[1][k];
                        spectrum[0][k] = (m + s) * kInvSqrt2;
                        spectrum[1][k] = (m - s) * kInvSqrt2;
                    }
                } else {
                    for (uint32_t k = 0u; k < cur_len; ++k) {
                        const float u = spectrum[0][k];
                        const float v = givens_negate ? -spectrum[1][k] : spectrum[1][k];
                        spectrum[0][k] = givens_cos * u - givens_sin * v;
                        spectrum[1][k] = givens_sin * u + givens_cos * v;
                    }
                }
            }

            // Windowed IMDCT and overlap-add into each channel's persistent ring buffer.
            for (uint32_t g = 0u; g < group_count; ++g) {
                float imdct_out[2u * kMaxSubframeLen];
                if (coded[g]) {
                    imdct_fast(spectrum[g].data(), imdct_out, cur_len);
                } else {
                    std::fill(imdct_out, imdct_out + 2u * cur_len, 0.0f);
                }
                overlap_add(next_state[group[g]], cur_offset, cur_len, imdct_out);
            }
        }

        // 4. Trailing end-of-frame marker (1 bit = 1) and zero-padding check.
        uint32_t eof_marker = 0u;
        if (!br.read_bit(eof_marker) || eof_marker != 1u) {
            return fail_silence();
        }
        const uint32_t trailing = br.bits_left();
        if (trailing > 7u) {
            return fail_silence();
        }
        if (trailing > 0u) {
            uint32_t pad = 0u;
            if (!br.read_bits(trailing, pad) || pad != 0u) {
                return fail_silence();
            }
        }

        // 5. Commit channel state, emit 512 samples/channel, and shift overlap buffer.
        channels_ = next_state;
        for (uint32_t c = 0u; c < channels; ++c) {
            auto& buf = channels_[c].overlap_buf;
            for (uint32_t i = 0u; i < kFrameSamples; ++i) {
                const bool skipped = (i < start_skip) || (i >= kFrameSamples - end_skip);
                pcm[i * channels + c] = skipped ? int16_t{0} : float_to_int16_safe(buf[i]);
            }
            // Shift pending tail [512..1023] -> [0..511] and clear upper half.
            std::copy(buf.begin() + kFrameSamples, buf.end(), buf.begin());
            std::fill(buf.begin() + kFrameSamples, buf.end(), 0.0f);
        }
        return true;
    }

private:
    struct ChannelState {
        std::array<float, 2u * kFrameSamples> overlap_buf{};  // [0..511] current frame, [512..1023] next tail
        std::array<float, kMaxSubframeLen> prev_right{};      // unwindowed right half of previous subframe
        uint32_t prev_len = 0u;
        std::array<int32_t, kMaxSfBands> saved_sf{};
        uint32_t saved_sf_len = 0u;
        uint32_t saved_sf_rate = 0u;
        bool has_saved_sf = false;
    };

    static void resample_scale_factors(const ChannelState& cs, uint32_t sample_rate, uint32_t cur_len,
                                       const ScaleFactorBandTable& dst_sfb,
                                       std::array<int32_t, kMaxSfBands>& out_sf) {
        if (cs.saved_sf_len == cur_len && cs.saved_sf_rate == sample_rate) {
            out_sf = cs.saved_sf;
            return;
        }
        const uint32_t src_rate = cs.saved_sf_rate != 0u ? cs.saved_sf_rate : sample_rate;
        const uint32_t src_len = cs.saved_sf_len != 0u ? cs.saved_sf_len : cur_len;
        const ScaleFactorBandTable& src_sfb = get_sfb_table(src_rate, src_len);
        for (uint32_t b = 0u; b < dst_sfb.num_bands; ++b) {
            // Map band midpoint from destination bin space into source bin space.
            const uint32_t dst_mid = (uint32_t{dst_sfb.offsets[b]} + dst_sfb.offsets[b + 1u]) / 2u;
            const uint32_t src_bin = static_cast<uint32_t>(
                (uint64_t{dst_mid} * src_len) / cur_len);
            uint32_t sb = 0u;
            while (sb + 1u < src_sfb.num_bands && src_sfb.offsets[sb + 1u] <= src_bin) {
                ++sb;
            }
            out_sf[b] = cs.saved_sf[sb];
        }
    }

    static void overlap_add(ChannelState& cs, uint32_t cur_offset, uint32_t cur_len,
                            const float* imdct_out) {
        // Boundary B = 256 + cur_offset (lies in [256, 767]).
        const uint32_t center = (kFrameSamples / 2u) + cur_offset;
        const uint32_t prev_len = cs.prev_len != 0u ? cs.prev_len : cur_len;
        const uint32_t w = std::min(prev_len, cur_len);
        const float* win = get_sine_window(w);

        // 1. Window and add previous subframe's right half (if any) centered at `center`.
        if (cs.prev_len != 0u) {
            const uint32_t prev_base = center - (prev_len / 2u);
            const uint32_t flat = (prev_len - w) / 2u;
            for (uint32_t i = 0u; i < flat; ++i) {
                cs.overlap_buf[prev_base + i] += cs.prev_right[i];
            }
            for (uint32_t j = 0u; j < w; ++j) {
                cs.overlap_buf[prev_base + flat + j] += cs.prev_right[flat + j] * win[w - 1u - j];
            }
        }

        // 2. Window and add current subframe's left half centered at `center`.
        const uint32_t cur_base = center - (cur_len / 2u);
        const uint32_t lead_zero = (cur_len - w) / 2u;
        for (uint32_t j = 0u; j < w; ++j) {
            cs.overlap_buf[cur_base + lead_zero + j] += imdct_out[lead_zero + j] * win[j];
        }
        for (uint32_t i = lead_zero + w; i < cur_len; ++i) {
            cs.overlap_buf[cur_base + i] += imdct_out[i];
        }

        // 3. Save unwindowed right half for the next subframe boundary.
        std::copy(imdct_out + cur_len, imdct_out + 2u * cur_len, cs.prev_right.begin());
        cs.prev_len = cur_len;
    }

    std::array<ChannelState, kMaxChannels> channels_{};
};

}  // namespace

PacketHeader parse_packet_header(const uint8_t* packet) {
    const uint32_t v = load_be32(packet);
    PacketHeader h;
    h.frame_count = v >> 26;
    h.first_frame_bit = (v >> 11) & kNoFrame;
    h.metadata = (v >> 8) & 7u;
    h.skip_packets = v & 0xFFu;
    return h;
}

void write_packet_header(uint8_t* packet, const PacketHeader& h) {
    const uint32_t v = ((h.frame_count & 0x3Fu) << 26) | ((h.first_frame_bit & kNoFrame) << 11) |
                       ((h.metadata & 7u) << 8) | (h.skip_packets & 0xFFu);
    packet[0] = static_cast<uint8_t>(v >> 24);
    packet[1] = static_cast<uint8_t>(v >> 16);
    packet[2] = static_cast<uint8_t>(v >> 8);
    packet[3] = static_cast<uint8_t>(v);
}

uint32_t Context::word(unsigned i) const {
    return std::atomic_ref<uint32_t>(w_[i]).load(std::memory_order_acquire);
}

void Context::set_word(unsigned i, uint32_t v) {
    std::atomic_ref<uint32_t>(w_[i]).store(v, std::memory_order_release);
}

uint32_t Context::field(unsigned w, unsigned shift, unsigned bits) const {
    const uint32_t mask = bits >= 32 ? ~0u : ((1u << bits) - 1u);
    return (word(w) >> shift) & mask;
}

void Context::set_field(unsigned w, unsigned shift, unsigned bits, uint32_t v) {
    const uint32_t mask = (bits >= 32 ? ~0u : ((1u << bits) - 1u)) << shift;
    const uint32_t masked_v = (v << shift) & mask;
    std::atomic_ref<uint32_t> ref(w_[w]);
    uint32_t cur = ref.load(std::memory_order_acquire);
    for (;;) {
        const uint32_t next = (cur & ~mask) | masked_v;
        if (ref.compare_exchange_weak(cur, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
            break;
        }
    }
}

uint32_t Context::ring_blocks() const {
    const uint32_t n = field(0, 22, 5);
    return n != 0 ? n : 32u;
}

uint32_t ring_step(uint32_t block, uint32_t advance, uint32_t ring) {
    if (ring == 0) {
        return 0;
    }
    return (block % ring + advance % ring) % ring;
}

uint32_t ring_room(uint32_t write, uint32_t read, uint32_t ring, bool writable) {
    if (!writable || ring == 0) {
        return 0;
    }
    const uint32_t gap = (read % ring + ring - write % ring) % ring;
    return gap == 0 ? ring : gap;
}

std::optional<uint32_t> first_frame_from(const uint8_t* data, uint32_t packets, uint32_t packet) {
    if (data == nullptr || packets == 0u || packets > kMaxAddressablePackets) {
        return std::nullopt;
    }
    uint64_t p = packet;
    while (p < packets) {
        const PacketHeader h = parse_packet_header(data + size_t{p} * kPacketBytes);
        if (h.first_frame_bit < kPayloadBits) {
            const uint64_t pos = p * kPacketBits + kPacketHeaderBits + h.first_frame_bit;
            if (pos > std::numeric_limits<uint32_t>::max()) {
                return std::nullopt;
            }
            return static_cast<uint32_t>(pos);
        }
        p += 1u + uint64_t{h.skip_packets};
    }
    return std::nullopt;
}

std::optional<Frame> read_frame(const uint8_t* data, uint32_t packets, uint32_t at) {
    if (data == nullptr || packets == 0u || packets > kMaxAddressablePackets) {
        return std::nullopt;
    }
    if (at < kPacketHeaderBits || (at % kPacketBits) < kPacketHeaderBits) {
        const uint32_t pkt = at / kPacketBits;
        const auto first = first_frame_from(data, packets, pkt);
        if (!first) {
            return std::nullopt;
        }
        at = *first;
    }
    // A position on padding moves to the next packet's first frame (at most once per packet).
    for (uint64_t guard = 0u; guard <= packets; ++guard) {
        PacketWalker w(data, packets, at);
        if (!w.has_bits() || !w.position_fits_u32()) {
            return std::nullopt;
        }
        if (w.left_in_packet() <= kFrameLengthBits || w.peek(kFrameLengthBits) == kNoFrame) {
            const PacketHeader h = parse_packet_header(data + size_t{w.packet()} * kPacketBytes);
            const uint64_t next_pkt = uint64_t{w.packet()} + 1u + uint64_t{h.skip_packets};
            if (next_pkt >= packets) {
                return std::nullopt;
            }
            const auto next = first_frame_from(data, packets, static_cast<uint32_t>(next_pkt));
            if (!next || *next <= at) {
                return std::nullopt;
            }
            at = *next;
            continue;
        }
        Frame f;
        f.start = w.position();
        f.length = w.peek(kFrameLengthBits);
        if (f.length <= kFrameLengthBits) {
            return std::nullopt;
        }
        f.bits.assign((f.length + 7u) / 8u, 0u);
        while (f.bits_gathered < f.length && w.has_bits()) {
            put_bit(f, w.bit());
        }
        f.end = w.position_fits_u32() ? w.position() : std::numeric_limits<uint32_t>::max();
        if (f.bits_gathered < f.length) {
            f.complete = false;
            return f;
        }
        // The next frame follows in this packet unless the rest of it is padding.
        if (w.left_in_packet() > kFrameLengthBits && w.peek(kFrameLengthBits) != kNoFrame) {
            if (!w.position_fits_u32()) {
                return std::nullopt;
            }
            f.next = w.position();
            f.has_next = true;
            return f;
        }
        uint64_t following = w.packet();
        if (w.left_in_packet() > 0u) {
            following += 1u + uint64_t{parse_packet_header(data + size_t{w.packet()} * kPacketBytes).skip_packets};
        } else if (w.settle()) {
            following = w.packet();
        } else {
            following = packets;
        }
        if (following < packets) {
            const auto next = first_frame_from(data, packets, static_cast<uint32_t>(following));
            f.has_next = next.has_value();
            f.next = next.value_or(0u);
        } else {
            f.has_next = false;
            f.next = 0u;
        }
        return f;
    }
    return std::nullopt;
}

bool continue_frame(const uint8_t* data, uint32_t packets, uint32_t count, Frame& frame) {
    if (data == nullptr || packets == 0u || packets > kMaxAddressablePackets) {
        return false;
    }
    if (frame.bits_gathered > frame.length || count > frame.length - frame.bits_gathered) {
        return false;
    }
    const size_t required_bytes = (uint64_t{frame.length} + 7u) / 8u;
    if (frame.bits.size() < required_bytes) {
        return false;
    }
    PacketWalker w(data, packets, kPacketHeaderBits);
    for (uint32_t i = 0u; i < count; ++i) {
        if (frame.bits_gathered >= frame.length || !w.has_bits()) {
            return false;
        }
        put_bit(frame, w.bit());
    }
    frame.complete = frame.bits_gathered == frame.length;
    return true;
}

std::unique_ptr<FrameDecoder> make_silence_decoder() {
    return std::make_unique<SilenceDecoder>();
}

std::unique_ptr<FrameDecoder> make_wmapro_decoder() {
    return std::make_unique<WmaProDecoder>();
}

namespace {

const uint8_t* map_input(Context ctx, unsigned buffer, const ApuMap& map, uint32_t& packets) {
    packets = ctx.input_packets(buffer);
    if (packets == 0) {
        return nullptr;
    }
    return map(ctx.input_address(buffer) & ~kPacketAlignMask, size_t{packets} * kPacketBytes);
}

// Decodes `frame` into the voice's PCM queue. `cut` (samples per channel, 0 = all) shortens it.
void emit(Context ctx, Voice& v, const Frame& frame, uint32_t cut) {
    const uint32_t channels = ctx.channels();
    if (channels < 1u || channels > kMaxChannels) {
        return;
    }
    int16_t pcm[kFrameSamples * kMaxChannels];
    const size_t total_samples = size_t{kFrameSamples} * channels;
    ++v.stats.frames;
    if (v.decoder != nullptr && v.decoder->decode(frame, channels, ctx.sample_rate(), pcm)) {
        ++v.stats.frames_decoded;
    } else {
        std::fill(pcm, pcm + total_samples, int16_t{0});
    }
    const uint32_t keep = cut != 0 ? std::min(cut, kFrameSamples) : kFrameSamples;
    const uint32_t drop = std::min(v.skip, keep);
    v.skip -= drop;
    v.pcm.insert(v.pcm.end(), pcm + size_t{drop} * channels, pcm + size_t{keep} * channels);
}

// One step towards more PCM. True: a frame was emitted or the input state moved (call again);
// false: no input to decode now.
bool step(Context ctx, Voice& v, const ApuMap& map) {
    if (v.ended) {
        if (!ctx.input_valid(0) && !ctx.input_valid(1)) {
            return false;
        }
        v.ended = false;  // the title queued more input
    }
    uint32_t cur = ctx.current_input();
    if (!ctx.input_valid(cur)) {
        if (!ctx.input_valid(1 - cur)) {
            v.ended = true;
            v.partial.reset();
            ++v.stats.exhausted;
            ctx.set_error(kErrorInputExhausted);
            return false;
        }
        cur = 1 - cur;
        ctx.set_current_input(cur);
        ctx.set_read_bit(0);
        ++v.stats.input_switches;
    }
    uint32_t packets = 0;
    const uint8_t* data = map_input(ctx, cur, map, packets);
    if (data == nullptr) {
        ctx.set_input_valid(cur, false);
        ++v.stats.bad_frames;
        return true;
    }
    if (v.partial && ctx.read_bit() >= kPacketHeaderBits) {
        // The guest explicitly sought into a new bit offset: drop any stale partial frame.
        v.partial.reset();
    }
    if (v.partial && ctx.read_bit() < kPacketHeaderBits) {
        // The tail of a frame that began in the other buffer opens this one.
        Frame f = std::move(*v.partial);
        v.partial.reset();
        const uint32_t before_gathered = f.bits_gathered;
        const bool continued = continue_frame(data, packets, f.length - f.bits_gathered, f);
        if (continued && f.complete) {
            const auto first = first_frame_from(data, packets, 0);
            const uint32_t loops = ctx.loop_count();
            const uint32_t loop_end = ctx.loop_end_bit();
            const uint32_t tail_limit = first ? *first : (packets * kPacketBits);
            const bool tail_at_loop_end =
                loops > 0 && loop_end >= kPacketHeaderBits && loop_end < tail_limit;
            if (tail_at_loop_end) {
                emit(ctx, v, f, (ctx.loop_end_subframe() + 1) * kSubframeSamples);
                if (loops != kLoopForever) {
                    ctx.set_field(0, 12, 8, loops - 1);
                }
                if (v.decoder != nullptr) {
                    v.decoder->reset();
                }
                v.skip = ctx.loop_skip_subframes() * kSubframeSamples;
                ctx.set_read_bit(ctx.loop_start_bit());
                ++v.stats.loops;
                return true;
            }
            emit(ctx, v, f, 0);
            if (first) {
                ctx.set_read_bit(*first);
            } else {
                ctx.set_input_valid(cur, false);
            }
        } else if (!continued && f.bits_gathered > before_gathered && f.bits_gathered < f.length) {
            // Buffer was too short to complete the spanning frame: preserve partial state into the next buffer.
            v.partial = std::move(f);
            ctx.set_input_valid(cur, false);
            ctx.set_read_bit(0);
        } else {
            ++v.stats.bad_frames;
            const auto first = first_frame_from(data, packets, 0);
            if (first) {
                ctx.set_read_bit(*first);
            } else {
                ctx.set_input_valid(cur, false);
            }
        }
        return true;
    }
    std::optional<Frame> frame = read_frame(data, packets, ctx.read_bit());
    if (!frame) {
        ctx.set_input_valid(cur, false);
        ctx.set_read_bit(0);
        return true;
    }
    if (!frame->complete) {
        v.partial = std::move(frame);
        ctx.set_input_valid(cur, false);
        ctx.set_read_bit(0);
        return true;
    }
    // The loop end lies in this frame: cut it after the loop's end subframe and go back.
    // Bounded by frame->end when !frame->has_next so a loop_end in the other buffer never triggers
    // a premature loop on the last frame of the current buffer.
    const uint32_t loops = ctx.loop_count();
    const uint32_t loop_end = ctx.loop_end_bit();
    const uint32_t frame_end = frame->has_next ? frame->next : frame->end;
    const bool at_loop_end = loops > 0 && loop_end != 0 && loop_end >= frame->start && loop_end < frame_end;
    if (at_loop_end) {
        emit(ctx, v, *frame, (ctx.loop_end_subframe() + 1) * kSubframeSamples);
        if (loops != kLoopForever) {
            ctx.set_field(0, 12, 8, loops - 1);
        }
        if (v.decoder != nullptr) {
            v.decoder->reset();
        }
        v.skip = ctx.loop_skip_subframes() * kSubframeSamples;
        ctx.set_read_bit(ctx.loop_start_bit());
        ++v.stats.loops;
        return true;
    }
    emit(ctx, v, *frame, 0);
    if (frame->has_next) {
        ctx.set_read_bit(frame->next);
    } else {
        ctx.set_input_valid(cur, false);
        ctx.set_read_bit(0);
    }
    return true;
}

}  // namespace

void enable(Context ctx, Voice& voice, std::unique_ptr<FrameDecoder> decoder) {
    ctx.set_field(2, 26, 6, 0);
    voice.decoder = decoder != nullptr ? std::move(decoder) : make_wmapro_decoder();
    if (voice.decoder != nullptr) {
        voice.decoder->reset();
    }
    voice.pcm.clear();
    voice.partial.reset();
    voice.ended = false;
    voice.skip = ctx.read_bit() < kPacketHeaderBits ? ctx.start_skip_subframes() * kSubframeSamples : 0u;
}

void advance_read(Context ctx, uint32_t blocks) {
    ctx.set_read_block(ring_step(ctx.read_block(), blocks, ctx.ring_blocks()));
    if (blocks > 0) {
        ctx.set_output_writable(true);
    }
}

uint32_t pump(Context ctx, Voice& v, const ApuMap& map) {
    uint32_t written = 0;
    const uint32_t ring = ctx.ring_blocks();
    // Bounded: a ring of 32 blocks, a few steps per block.
    for (uint32_t guard = 0; guard < 2 * 32 && ctx.output_writable(); ++guard) {
        for (uint32_t steps = 0; v.pcm.size() < kBlockSamples && steps < 64; ++steps) {
            if (!step(ctx, v, map)) {
                break;
            }
        }
        if (v.pcm.empty() || (v.pcm.size() < kBlockSamples && !v.ended)) {
            break;  // wait for more input (a short tail is written once the stream has ended)
        }
        const uint32_t write = ctx.write_block();
        const uint64_t out_addr = uint64_t{ctx.output_address()} + uint64_t{write} * kBlockBytes;
        if (out_addr > 0xFFFFFFFFull || out_addr + kBlockBytes > 0x100000000ull) {
            break;
        }
        uint8_t* out = map(static_cast<uint32_t>(out_addr), kBlockBytes);
        if (out == nullptr) {
            break;
        }
        for (uint32_t i = 0; i < kBlockSamples; ++i) {
            int16_t s = 0;
            if (!v.pcm.empty()) {
                s = v.pcm.front();
                v.pcm.pop_front();
            }
            const int16_t mag = s == INT16_MIN ? INT16_MAX : static_cast<int16_t>(s < 0 ? -s : s);
            v.stats.peak = std::max(v.stats.peak, mag);
            out[2 * i] = static_cast<uint8_t>(static_cast<uint16_t>(s));
            out[2 * i + 1] = static_cast<uint8_t>(static_cast<uint16_t>(s) >> 8);
        }
        const uint32_t next = (write + 1) % ring;
        ctx.set_write_block(next);
        ++written;
        ++v.stats.blocks_written;
        if (next == ctx.read_block()) {
            ctx.set_output_writable(false);  // full until the title takes blocks
        }
    }
    return written;
}

}  // namespace playgta6::os::xma
