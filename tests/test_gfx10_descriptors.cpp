// gfx10 descriptor words (runtime/gpu/gfx10_descriptors.h): field positions against the public
// driver definitions, round trips, slot classification, format and sampler mappings; plus the
// STORAGE_BINDING usage of UAV textures whose descriptors those formats feed (G60). Synthetic
// values only.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu/cmd_encoder.h"
#include "gpu/d3d12_guest_objects.h"
#include "gpu/d3d12_guest_resources.h"
#include "gpu/descriptor_words.h"
#include "gpu/gfx10_descriptors.h"
#include "gpu/pso_cache.h"
#include "gpu/state_bridge.h"
#include "guest/com_bridge.h"
#include "os/vmem.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_gfx10_descriptors.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu::gfx10;
namespace d3d = playgta6::gpu::d3d12_state;

// Field positions: one field at a time, compared with the shifts of gfx10_sq_ko_reg.h.
void test_buffer_field_positions() {
    uint32_t w[4];
    BufferResource b;
    b.dst_sel = {0, 0, 0, 0};
    b.resource_level = false;
    encode(b, w);
    CHECK(w[0] == 0 && w[1] == 0 && w[2] == 0 && w[3] == 0);

    BufferResource s = b;
    s.base = 0x0000123456789abcull;
    encode(s, w);
    CHECK(w[0] == 0x56789abcu && w[1] == 0x1234u);  // base_address: 48 bits from dword 0 bit 0
    s = b;
    s.stride = 1;
    encode(s, w);
    CHECK(w[1] == (1u << 16));  // SqBufRsrcTWord1StrideShift = 16
    s = b;
    s.cache_swizzle = true;
    encode(s, w);
    CHECK(w[1] == (1u << 30));  // Gfx10SqBufRsrcTWord1CacheSwizzleShift = 30
    s = b;
    s.swizzle_enable = true;
    encode(s, w);
    CHECK(w[1] == (1u << 31));  // Gfx10SqBufRsrcTWord1SwizzleEnableShift = 31
    s = b;
    s.num_records = 0xdeadbeef;
    encode(s, w);
    CHECK(w[2] == 0xdeadbeefu);
    s = b;
    s.dst_sel = {1, 0, 0, 0};
    encode(s, w);
    CHECK(w[3] == 1u);
    s.dst_sel = {0, 1, 0, 0};
    encode(s, w);
    CHECK(w[3] == (1u << 3));
    s.dst_sel = {0, 0, 1, 0};
    encode(s, w);
    CHECK(w[3] == (1u << 6));
    s.dst_sel = {0, 0, 0, 1};
    encode(s, w);
    CHECK(w[3] == (1u << 9));
    s = b;
    s.format = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 12));  // Gfx10SqBufRsrcTWord3FormatShift = 12
    s = b;
    s.index_stride = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 21));
    s = b;
    s.add_tid_enable = true;
    encode(s, w);
    CHECK(w[3] == (1u << 23));
    s = b;
    s.resource_level = true;
    encode(s, w);
    CHECK(w[3] == (1u << 24));  // Gfx10SqBufRsrcTWord3ResourceLevelShift = 24
    s = b;
    s.oob_select = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 28));
    s = b;
    s.type = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 30));
}

void test_image_field_positions() {
    uint32_t w[8];
    ImageResource z;
    z.width = 1;
    z.height = 1;
    z.depth = 1;
    z.dst_sel = {0, 0, 0, 0};
    z.type = 0;
    z.resource_level = false;
    encode(z, w);
    for (uint32_t i = 0; i < 8; ++i) {
        CHECK(w[i] == 0);
    }
    ImageResource s = z;
    s.base = 0x0012'3456'789a'bc00ull;  // stored >> 8 in 40 bits
    encode(s, w);
    CHECK(w[0] == 0x56789abcu && (w[1] & 0xffu) == 0x34u);
    s = z;
    s.min_lod = 1;
    encode(s, w);
    CHECK(w[1] == (1u << 8));  // Gfx10SqImgRsrcTWord1MinLodShift = 8
    s = z;
    s.format = 1;
    encode(s, w);
    CHECK(w[1] == (1u << 20));  // Gfx10SqImgRsrcTWord1FormatShift = 20
    s = z;
    s.width = 2;
    encode(s, w);
    CHECK(w[1] == (1u << 30) && w[2] == 0);  // Gfx10SqImgRsrcTWord1WidthShift = 30
    s.width = 5;
    encode(s, w);
    CHECK(w[1] == 0 && w[2] == 1u);  // width - 1 = 4: bit 2 of the width lands in dword 2 bit 0
    s = z;
    s.height = 2;
    encode(s, w);
    CHECK(w[2] == (1u << 14));  // Gfx10SqImgRsrcTWord2HeightShift = 14
    s = z;
    s.resource_level = true;
    encode(s, w);
    CHECK(w[2] == (1u << 31));
    s = z;
    s.base_level = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 12));
    s = z;
    s.last_level = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 16));
    s = z;
    s.sw_mode = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 20));  // SqImgRsrcTWord3SwModeShift = 20
    s = z;
    s.bc_swizzle = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 25));
    s = z;
    s.type = ImageType::Tex2D;
    encode(s, w);
    CHECK(w[3] == (9u << 28));  // SqImgRsrcTWord3TypeShift = 28, SQ_RSRC_IMG_2D = 9
    s = z;
    s.depth = 2;
    encode(s, w);
    CHECK(w[4] == 1u);
    s = z;
    s.base_array = 1;
    encode(s, w);
    CHECK(w[4] == (1u << 16));  // Gfx10SqImgRsrcTWord4BaseArrayShift = 16
    s = z;
    s.max_mip = 1;
    encode(s, w);
    CHECK(w[5] == (1u << 4));  // Gfx10SqImgRsrcTWord5MaxMipShift = 4
    s = z;
    s.compression_en = true;
    encode(s, w);
    CHECK(w[6] == (1u << 21));  // SqImgRsrcTWord6CompressionEnShift = 21
    s = z;
    s.meta_data_address = 0x0012'3456'789a'bc00ull;
    encode(s, w);
    CHECK(w[6] == (0xbcu << 24) && w[7] == 0x3456789au);  // SqImgRsrcTWord6MetaDataAddressShift = 24
}

void test_sampler_field_positions() {
    uint32_t w[4];
    SamplerState z;
    encode(z, w);
    CHECK(w[0] == 0 && w[1] == 0 && w[2] == 0 && w[3] == 0);
    SamplerState s = z;
    s.clamp_y = 1;
    encode(s, w);
    CHECK(w[0] == (1u << 3));
    s = z;
    s.depth_compare_func = 1;
    encode(s, w);
    CHECK(w[0] == (1u << 12));  // SqImgSampTWord0DepthCompareFuncShift = 12
    s = z;
    s.filter_mode = 1;
    encode(s, w);
    CHECK(w[0] == (1u << 29));
    s = z;
    s.max_lod = 1;
    encode(s, w);
    CHECK(w[1] == (1u << 12));
    s = z;
    s.xy_min_filter = 1;
    encode(s, w);
    CHECK(w[2] == (1u << 22));  // SqImgSampTWord2XyMinFilterShift = 22
    s = z;
    s.mip_filter = 1;
    encode(s, w);
    CHECK(w[2] == (1u << 26));
    s = z;
    s.border_color_type = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 30));  // SqImgSampTWord3BorderColorTypeShift = 30
    s = z;
    s.border_color_ptr = 1;
    encode(s, w);
    CHECK(w[3] == 1u);  // Gfx10SqImgSampTWord3BorderColorPtrShift = 0
}

void test_round_trips() {
    BufferResource b;
    b.base = 0x0000'7e00'1234'5670ull;
    b.stride = 48;
    b.num_records = 4096;
    b.dst_sel = {Sel::Z, Sel::Y, Sel::X, Sel::One};
    b.format = 56;
    b.index_stride = 3;
    b.add_tid_enable = true;
    b.oob_select = Oob::Complete;
    uint32_t w[8] = {};
    encode(b, w);
    const BufferResource b2 = decode_buffer(w);
    CHECK(b2.base == b.base && b2.stride == b.stride && b2.num_records == b.num_records && b2.dst_sel == b.dst_sel &&
          b2.format == b.format && b2.index_stride == 3 && b2.add_tid_enable && b2.resource_level &&
          b2.oob_select == Oob::Complete && b2.type == 0);
    CHECK(plausible_buffer(w));
    CHECK(classify_slot(w) == SlotContent::Buffer);

    ImageResource i;
    i.base = 0x0000'00a1'b2c3'd400ull;
    i.min_lod = 0x123;
    i.format = 181;
    i.width = 3840;
    i.height = 2160;
    i.dst_sel = {Sel::X, Sel::Y, Sel::Z, Sel::One};
    i.base_level = 2;
    i.last_level = 11;
    i.sw_mode = 27;
    i.bc_swizzle = 5;
    i.type = ImageType::Tex2DArray;
    i.depth = 6;
    i.base_array = 2;
    i.array_pitch = 3;
    i.max_mip = 11;
    i.min_lod_warn = 0x456;
    i.perf_mod = 4;
    i.corner_samples = true;
    i.lod_hdw_cnt_en = true;
    i.prt_default = true;
    i.big_page = true;
    i.counter_bank_id = 0x5a;
    i.iterate_256 = true;
    i.max_uncompressed_block_size = 2;
    i.max_compressed_block_size = 1;
    i.meta_pipe_aligned = true;
    i.write_compress_enable = true;
    i.compression_en = true;
    i.alpha_is_on_msb = true;
    i.color_transform = true;
    i.meta_data_address = 0x0000'00f0'0000'1200ull;
    encode(i, w);
    const ImageResource i2 = decode_image(w);
    CHECK(i2.base == i.base && i2.min_lod == i.min_lod && i2.format == i.format && i2.width == 3840 &&
          i2.height == 2160 && i2.dst_sel == i.dst_sel && i2.base_level == 2 && i2.last_level == 11 &&
          i2.sw_mode == 27 && i2.bc_swizzle == 5 && i2.type == ImageType::Tex2DArray && i2.depth == 6 &&
          i2.base_array == 2 && i2.array_pitch == 3 && i2.max_mip == 11 && i2.min_lod_warn == 0x456 &&
          i2.perf_mod == 4 && i2.corner_samples && i2.lod_hdw_cnt_en && i2.prt_default && i2.big_page &&
          i2.counter_bank_id == 0x5a && i2.iterate_256 && i2.max_uncompressed_block_size == 2 &&
          i2.max_compressed_block_size == 1 && i2.meta_pipe_aligned && i2.write_compress_enable &&
          i2.compression_en && i2.alpha_is_on_msb && i2.color_transform &&
          i2.meta_data_address == i.meta_data_address && i2.resource_level);
    CHECK(plausible_image(w));
    CHECK(classify_slot(w) == SlotContent::Image);

    SamplerState s;
    s.clamp_x = 6;
    s.clamp_y = 2;
    s.clamp_z = 1;
    s.max_aniso_ratio = 4;
    s.depth_compare_func = 3;
    s.force_unnormalized = true;
    s.aniso_threshold = 5;
    s.aniso_bias = 33;
    s.filter_mode = 2;
    s.min_lod = 0x10;
    s.max_lod = 0xfff;
    s.perf_mip = 9;
    s.perf_z = 3;
    s.lod_bias = 0x3f00;
    s.lod_bias_sec = 7;
    s.xy_mag_filter = 3;
    s.xy_min_filter = 2;
    s.z_filter = 2;
    s.mip_filter = 2;
    s.mip_point_preclamp = true;
    s.aniso_override = true;
    s.blend_prt = true;
    s.deriv_adjust_en = true;
    s.border_color_ptr = 0x7ff;
    s.border_color_type = 3;
    uint32_t sw[4];
    encode(s, sw);
    const SamplerState s2 = decode_sampler(sw);
    uint32_t sw2[4];
    encode(s2, sw2);
    CHECK(std::memcmp(sw, sw2, sizeof(sw)) == 0);
    CHECK(s2.clamp_x == 6 && s2.max_aniso_ratio == 4 && s2.filter_mode == 2 && s2.lod_bias == 0x3f00 &&
          s2.border_color_ptr == 0x7ff && s2.border_color_type == 3 && s2.deriv_adjust_en);
}

void test_classify_slot() {
    uint32_t zero[8] = {};
    CHECK(classify_slot(zero) == SlotContent::Zero);
    // A free-list link (next index in the first dword) and plain float data are not descriptors.
    uint32_t link[8] = {42, 0, 0, 0, 0, 0, 0, 0};
    CHECK(classify_slot(link) == SlotContent::Unknown);
    uint32_t floats[8] = {0x3f800000, 0xbf800000, 0, 0xffffffff, 0x3f800000, 0x3f800000, 0, 0};
    CHECK(classify_slot(floats) == SlotContent::Unknown);
    // Buffer descriptors: resource_level is not required (constant-buffer words with dword 3 =
    // 0x0002403C: selects X, W, 0, 0, format 36, resource_level 0); a reserved format or a non-zero
    // type is rejected.
    BufferResource b;
    b.format = Fmt::F32;
    uint32_t w[8] = {};
    encode(b, w);
    CHECK(classify_slot(w) == SlotContent::Buffer);
    const uint32_t cbv[8] = {0x1000, 0x00100000, 6, 0x0002403Cu, 0, 0, 0, 0};
    CHECK(classify_slot(cbv) == SlotContent::Buffer);
    CHECK(decode_buffer(cbv).stride == 16 && decode_buffer(cbv).num_records == 6 && !decode_buffer(cbv).resource_level);
    b.type = 1;
    encode(b, w);
    CHECK(classify_slot(w) == SlotContent::Unknown);
    b.type = 0;
    b.format = 100;
    encode(b, w);
    CHECK(classify_slot(w) == SlotContent::Unknown);
    // Image with base_level > last_level or a reserved format is rejected.
    ImageResource i;
    i.format = 56;
    encode(i, w);
    CHECK(classify_slot(w) == SlotContent::Image);
    i.base_level = 3;
    i.last_level = 1;
    encode(i, w);
    CHECK(classify_slot(w) == SlotContent::Unknown);
    i.base_level = 0;
    i.format = 155;
    encode(i, w);
    CHECK(classify_slot(w) == SlotContent::Unknown);
}

// G60: typeless, depth/stencil-plane and Xbox DXGI formats map to gfx10 image formats; reverse
// lookups of the shared hardware formats keep their canonical DXGI format.
void test_g60_formats() {
    struct Case {
        uint32_t dxgi;
        uint32_t hw;
        Swizzle sel;
    };
    const Swizzle x001 = {Sel::X, Sel::Zero, Sel::Zero, Sel::One};
    const Swizzle xyz1 = {Sel::X, Sel::Y, Sel::Z, Sel::One};
    const Swizzle zyxw = {Sel::Z, Sel::Y, Sel::X, Sel::W};
    const Swizzle zyx1 = {Sel::Z, Sel::Y, Sel::X, Sel::One};
    const Case cases[] = {
        {115, 136, zyxw},              // B4G4R4A4_UNORM -> 4_4_4_4_UNORM
        {118, 7, x001},                // D16_UNORM_S8_UINT -> 16_UNORM (depth plane)
        {119, 7, x001},                // R16_UNORM_X8_TYPELESS
        {120, 5, x001},                // X16_TYPELESS_G8_UINT -> 8_UINT (stencil plane)
        {19, 22, x001},                // R32G8X24_TYPELESS -> 32_FLOAT
        {22, 5, x001},                 // X32_TYPELESS_G8X24_UINT -> 8_UINT
        {44, 141, x001},               // R24G8_TYPELESS -> 8_24_UNORM
        {47, 5, x001},                 // X24_TYPELESS_G8_UINT -> 8_UINT
        {90, 56, zyxw},                // B8G8R8A8_TYPELESS -> 8_8_8_8_UNORM
        {92, 56, zyx1},                // B8G8R8X8_TYPELESS
        {5, 72, xyz1},                 // R32G32B32_TYPELESS -> 32_32_32_UINT
        {89, 50, kIdentitySwizzle},    // R10G10B10_XR_BIAS_A2_UNORM -> 2_10_10_10_UNORM
    };
    for (const Case& c : cases) {
        HardwareFormat f;
        CHECK(hardware_format(c.dxgi, &f) && f.format == c.hw && f.swizzle == c.sel);
        // T# encode/decode keeps the format and selects, and the slot classifies as an image
        // (valid_image_format accepts every new hardware format).
        ImageResource img;
        img.base = 0x0000000100040000ull;
        img.format = f.format;
        img.width = 64;
        img.height = 32;
        img.dst_sel = compose_component_mapping(f.swizzle, 0x1688u);
        img.type = ImageType::Tex2D;
        uint32_t words[kImageDwords] = {};
        encode(img, words);
        CHECK(plausible_image(words));
        CHECK(classify_slot(words) == SlotContent::Image);
        const ImageResource dec = decode_image(words);
        CHECK(dec.format == c.hw && dec.dst_sel == c.sel && dec.width == 64 && dec.height == 32);
        // The reverse lookup lands on a DXGI format with the same hardware format and selects.
        bool exact = false;
        const uint32_t back = dxgi_format(dec.format, dec.dst_sel, &exact);
        HardwareFormat bf;
        CHECK(exact && hardware_format(back, &bf) && bf.format == c.hw && bf.swizzle == c.sel);
    }
    // The new hardware format reverses to its only DXGI format; aliases do not take over the
    // canonical reverse mappings of shared hardware formats.
    bool exact = false;
    CHECK(dxgi_format(136, zyxw, &exact) == 115 && exact);
    CHECK(dxgi_format(7, x001, &exact) == 56 && exact);    // R16_UNORM, not D16 / Xbox D16S8
    CHECK(dxgi_format(5, x001, &exact) == 62 && exact);    // R8_UINT, not a stencil plane
    CHECK(dxgi_format(22, x001, &exact) == 41 && exact);   // R32_FLOAT
    CHECK(dxgi_format(141, x001, &exact) == 45 && exact);  // D24_UNORM_S8_UINT
    CHECK(dxgi_format(72, xyz1, &exact) == 7 && exact);    // R32G32B32_UINT
    CHECK(dxgi_format(50, kIdentitySwizzle, &exact) == 24 && exact);  // R10G10B10A2_UNORM
    CHECK(dxgi_format(56, zyxw, &exact) == 87 && exact);   // B8G8R8A8_UNORM
    CHECK(dxgi_format(56, zyx1, &exact) == 88 && exact);   // B8G8R8X8_UNORM
    CHECK(dxgi_format(56, kIdentitySwizzle, &exact) == 28 && exact);  // R8G8B8A8_UNORM
    CHECK(dxgi_format(136, kIdentitySwizzle, &exact) == 115 && !exact);
    // 4_4_4_4 is image-only: not a buffer format.
    CHECK(buffer_format_bytes(136) == 0);
}

void test_formats() {
    HardwareFormat f;
    CHECK(hardware_format(28, &f) && f.format == 56 && f.swizzle == kIdentitySwizzle);  // R8G8B8A8_UNORM
    CHECK(hardware_format(87, &f) && f.format == 56 && f.swizzle == (Swizzle{Sel::Z, Sel::Y, Sel::X, Sel::W}));
    CHECK(hardware_format(29, &f) && f.format == 130);  // 8_8_8_8_SRGB
    CHECK(hardware_format(24, &f) && f.format == 50);   // R10G10B10A2_UNORM -> 2_10_10_10_UNORM
    CHECK(hardware_format(26, &f) && f.format == 36);   // R11G11B10_FLOAT -> 10_11_11_FLOAT
    CHECK(hardware_format(98, &f) && f.format == 181);  // BC7_UNORM
    CHECK(hardware_format(2, &f) && f.format == 77);    // R32G32B32A32_FLOAT
    CHECK(hardware_format(61, &f) && f.format == 1 &&
          f.swizzle == (Swizzle{Sel::X, Sel::Zero, Sel::Zero, Sel::One}));  // R8_UNORM -> 8_UNORM (X,0,0,1)
    CHECK(hardware_format(65, &f) && f.format == 1 &&
          f.swizzle == (Swizzle{Sel::Zero, Sel::Zero, Sel::Zero, Sel::X}));  // A8_UNORM -> 8_UNORM (0,0,0,X)
    CHECK(!hardware_format(100, &f));                   // video format
    bool exact = false;
    CHECK(dxgi_format(56, Swizzle{Sel::Z, Sel::Y, Sel::X, Sel::W}, &exact) == 87 && exact);
    CHECK(dxgi_format(56, kIdentitySwizzle, &exact) == 28 && exact);
    CHECK(dxgi_format(56, Swizzle{Sel::W, Sel::Z, Sel::Y, Sel::X}, &exact) == 28 && !exact);
    CHECK(dxgi_format(1, Swizzle{Sel::X, Sel::Zero, Sel::Zero, Sel::One}, &exact) == 61 && exact);
    CHECK(dxgi_format(1, Swizzle{Sel::Zero, Sel::Zero, Sel::Zero, Sel::X}, &exact) == 65 && exact);
    CHECK(dxgi_format(1, kIdentitySwizzle, &exact) == 61 && !exact);
    CHECK(dxgi_format(200, kIdentitySwizzle, &exact) == 0 && !exact);
    // A8_UNORM (65) T# image descriptor encode/decode round-trip preserves distinct DXGI format 65.
    {
        HardwareFormat a8;
        CHECK(hardware_format(65, &a8));
        ImageResource img;
        img.base = 0x0000'0001'0002'0000ull;
        img.format = a8.format;
        img.width = 256;
        img.height = 128;
        img.dst_sel = compose_component_mapping(a8.swizzle, 0x1688u);
        img.type = ImageType::Tex2D;
        uint32_t words[kImageDwords] = {};
        encode(img, words);
        CHECK(plausible_image(words));
        CHECK(classify_slot(words) == SlotContent::Image);
        const ImageResource dec = decode_image(words);
        CHECK(dec.format == 1u && dec.dst_sel == (Swizzle{Sel::Zero, Sel::Zero, Sel::Zero, Sel::X}));
        bool a8_exact = false;
        CHECK(dxgi_format(dec.format, dec.dst_sel, &a8_exact) == 65u && a8_exact);
    }
    // Every DXGI format of the table maps to a hardware format that maps back to a DXGI format with
    // the same hardware format.
    for (uint32_t dxgi = 1; dxgi <= 120; ++dxgi) {
        HardwareFormat h;
        if (!hardware_format(dxgi, &h)) {
            continue;
        }
        const uint32_t back = dxgi_format(h.format, h.swizzle);
        HardwareFormat h2;
        CHECK(back != 0 && hardware_format(back, &h2) && h2.format == h.format);
    }
    test_g60_formats();
    CHECK(buffer_format_bytes(Fmt::F32) == 4 && buffer_format_bytes(77) == 16 && buffer_format_bytes(74) == 12 &&
          buffer_format_bytes(14) == 2 && buffer_format_bytes(1) == 1 && buffer_format_bytes(0) == 0);
}

void test_component_mapping() {
    // D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING = 0x1688 (R, G, B, A).
    CHECK(compose_component_mapping(kIdentitySwizzle, 0x1688) == kIdentitySwizzle);
    const Swizzle bgra = {Sel::Z, Sel::Y, Sel::X, Sel::W};
    CHECK(compose_component_mapping(bgra, 0x1688) == bgra);
    // (A, 0, 1, R): 3 | 4 << 3 | 5 << 6 | 0 << 9.
    const uint32_t m = 3u | (4u << 3) | (5u << 6) | (1u << 12);
    CHECK(compose_component_mapping(kIdentitySwizzle, m) == (Swizzle{Sel::W, Sel::Zero, Sel::One, Sel::X}));
    uint32_t back = 0;
    CHECK(component_mapping_from(kIdentitySwizzle, Swizzle{Sel::W, Sel::Zero, Sel::One, Sel::X}, &back) && back == m);
    CHECK(component_mapping_from(bgra, bgra, &back) && back == 0x1688);
    // A select the format does not provide cannot be expressed.
    CHECK(!component_mapping_from(Swizzle{Sel::X, Sel::Zero, Sel::Zero, Sel::One},
                                  Swizzle{Sel::Y, Sel::Zero, Sel::Zero, Sel::One}, &back));
}

void test_samplers() {
    CHECK(lod_to_u4_8(0.0f) == 0 && lod_to_u4_8(1.0f) == 256 && lod_to_u4_8(1000.0f) == 0xfff &&
          lod_to_u4_8(-1.0f) == 0);
    CHECK(u4_8_to_lod(256) == 1.0f);
    CHECK(bias_to_s6_8(-1.0f) == 0x3f00 && s6_8_to_bias(0x3f00) == -1.0f && s6_8_to_bias(bias_to_s6_8(2.5f)) == 2.5f);

    d3d::SamplerDesc d;  // MIN_MAG_MIP_LINEAR, wrap
    d.max_lod = 13.0f;
    bool exact = false;
    SamplerState s = sampler_from_d3d12(d, &exact);
    CHECK(exact && s.xy_mag_filter == 1 && s.xy_min_filter == 1 && s.mip_filter == 2 && s.clamp_x == 0 &&
          s.max_lod == 13u * 256u && s.border_color_type == 0);
    d3d::SamplerDesc back = d3d12_from_sampler(s);
    CHECK(back.filter == 0x15 && back.address_u == 1 && back.max_lod == 13.0f && back.min_lod == 0.0f);

    d3d::SamplerDesc c;
    c.filter = 0x80 | 0x15;  // COMPARISON_MIN_MAG_MIP_LINEAR
    c.comparison_func = 4;   // LESS_EQUAL
    c.address_u = 4;         // BORDER
    c.address_v = 3;         // CLAMP
    c.address_w = 5;         // MIRROR_ONCE
    c.border_color[0] = c.border_color[1] = c.border_color[2] = c.border_color[3] = 1.0f;
    c.mip_lod_bias = -0.5f;
    s = sampler_from_d3d12(c, &exact);
    CHECK(exact && s.depth_compare_func == 3 && s.clamp_x == 6 && s.clamp_y == 2 && s.clamp_z == 3 &&
          s.border_color_type == 2 && s.filter_mode == 0);
    back = d3d12_from_sampler(s);
    CHECK(back.filter == (0x80u | 0x15u) && back.comparison_func == 4 && back.address_u == 4 && back.address_v == 3 &&
          back.address_w == 5 && back.border_color[0] == 1.0f && back.mip_lod_bias == -0.5f);

    d3d::SamplerDesc a;
    a.filter = 0x55;  // ANISOTROPIC
    a.max_anisotropy = 8;
    s = sampler_from_d3d12(a, &exact);
    CHECK(s.xy_mag_filter == 3 && s.max_aniso_ratio == 3);
    back = d3d12_from_sampler(s);
    CHECK(back.filter == 0x55 && back.max_anisotropy == 8);

    d3d::SamplerDesc mx;
    mx.filter = 0x180 | 0x15;  // MAXIMUM_MIN_MAG_MIP_LINEAR
    s = sampler_from_d3d12(mx, &exact);
    CHECK(s.filter_mode == 2 && d3d12_from_sampler(s).filter == (0x180u | 0x15u));

    d3d::SamplerDesc odd;
    odd.border_color[0] = 0.5f;
    sampler_from_d3d12(odd, &exact);
    CHECK(!exact);
}

// G60: a BGRA8 UAV texture gets STORAGE_BINDING only when the WebGPU device has
// "bgra8unorm-storage"; without it the texture keeps its other usages and the device notes why.
struct GpuPacket {
    playgta6::GpuOpcode opcode{};
    std::vector<uint8_t> payload;
};

std::vector<GpuPacket> split_packets(const std::vector<std::vector<uint8_t>>& captured) {
    std::vector<uint8_t> all;
    for (const auto& c : captured) {
        all.insert(all.end(), c.begin(), c.end());
    }
    std::vector<GpuPacket> out;
    size_t off = 0;
    while (off + sizeof(playgta6::GpuPacketHeader) <= all.size()) {
        playgta6::GpuPacketHeader h{};
        std::memcpy(&h, all.data() + off, sizeof(h));
        off += sizeof(h);
        if (off + h.payload_bytes > all.size()) {
            break;
        }
        GpuPacket p;
        p.opcode = static_cast<playgta6::GpuOpcode>(h.opcode);
        p.payload.assign(all.begin() + static_cast<std::ptrdiff_t>(off),
                         all.begin() + static_cast<std::ptrdiff_t>(off + h.payload_bytes));
        off += h.payload_bytes;
        out.push_back(std::move(p));
    }
    CHECK(off == all.size());
    return out;
}

void test_bgra8_uav_storage_usage() {
    namespace com = playgta6::guest::com;
    namespace dg = playgta6::gpu::d3d12_guest;
    namespace pso = playgta6::gpu::pso;
    using playgta6::GpuCreateTexture2Payload;
    using playgta6::GpuOpcode;
    namespace GpuTextureUsage = playgta6::GpuTextureUsage;

    playgta6::os::VirtualMemoryArena arena;
    com::reset_com_bridge();
    com::set_guest_memory(&arena);
    for (const bool enabled : {true, false}) {
        std::vector<std::vector<uint8_t>> captured;
        dg::DeviceOptions options;
        options.sink = [&captured](const uint8_t* data, size_t size) { captured.emplace_back(data, data + size); };
        options.bridge.features = enabled ? pso::WgpuFeature::Bgra8UnormStorage : 0u;
        options.defer_gpu_objects = false;  // the create packet is checked at creation
        com::Ref<dg::Device> device = com::create_object<dg::Device>(com::implements(dg::kID3D12Device), options);
        CHECK(device);
        if (!device) {
            continue;
        }
        dg::GuestResourceDesc desc{};
        desc.Dimension = 3;  // TEXTURE2D
        desc.Width = 64;
        desc.Height = 64;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = 87;  // B8G8R8A8_UNORM
        desc.SampleCount = 1;
        desc.Flags = dg::ResourceFlags::AllowUnorderedAccess;
        dg::Resource::Placement placement;
        placement.heap.Type = dg::HeapType::Default;
        com::Ref<dg::Resource> resource =
            com::create_object<dg::Resource>(com::implements(dg::kID3D12Resource), dg::device_ref(device.get()), desc,
                                             std::move(placement), uint32_t{0});
        CHECK(resource && resource->initialize() && resource->has_gpu_texture());

        const std::vector<GpuPacket> ps = split_packets(captured);
        const GpuPacket* create = nullptr;
        size_t creates = 0;
        for (const GpuPacket& p : ps) {
            if (p.opcode == GpuOpcode::CreateTexture2) {
                create = &p;
                ++creates;
            }
        }
        CHECK(creates == 1 && create != nullptr && create->payload.size() >= sizeof(GpuCreateTexture2Payload));
        if (create != nullptr && create->payload.size() >= sizeof(GpuCreateTexture2Payload)) {
            GpuCreateTexture2Payload t{};
            std::memcpy(&t, create->payload.data(), sizeof(t));
            CHECK(t.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Bgra8Unorm));
            CHECK((t.usage_flags & GpuTextureUsage::TextureBinding) != 0);
            CHECK(((t.usage_flags & GpuTextureUsage::StorageBinding) != 0) == enabled);
        }
        bool warned = false;
        for (const std::string& d : device->diagnostics()) {
            warned = warned || d.find("has no WebGPU storage binding") != std::string::npos;
        }
        CHECK(warned == !enabled);
        resource = {};
        device = {};
    }
    com::reset_com_bridge();
    com::set_guest_memory(nullptr);
}

// G83 (#1748 / #651): full 3D view dimension round-trips (1D, 1DArray, 2D, 2DArray, 2DMS,
// 2DMSArray, 3D, Cube, CubeArray) through view_words / image_words_at_address / decode_view_words.
void test_g83_3d_view_dimensions() {
    namespace bridge = playgta6::gpu::bridge;
    namespace pso = playgta6::gpu::pso;
    using playgta6::GpuTextureDimension;

    d3d::DescriptorHeapRegistry heaps;
    bridge::StateBridgeDevice device(heaps);

    auto reg_tex = [&](uint32_t id, uint64_t va, GpuTextureDimension dim, uint32_t w, uint32_t h, uint32_t layers,
                       uint32_t mips, uint32_t samples, pso::DxgiFormat fmt) {
        bridge::TextureInfo ti;
        ti.format = fmt;
        ti.width = w;
        ti.height = h;
        ti.depth_or_array_size = layers;
        ti.mip_levels = mips;
        ti.sample_count = samples;
        ti.dimension = dim;
        ti.gpu_va = va;
        device.register_texture(id, ti);
    };

    reg_tex(10, 0x0000'1000'0001'0000ull, GpuTextureDimension::Dim1D, 512, 1, 8, 5, 1, pso::DxgiFormat::R8G8B8A8_UNORM);
    reg_tex(11, 0x0000'1000'0002'0000ull, GpuTextureDimension::Dim2D, 1024, 512, 12, 6, 1, pso::DxgiFormat::R16G16B16A16_FLOAT);
    reg_tex(12, 0x0000'1000'0003'0000ull, GpuTextureDimension::Dim2D, 256, 256, 6, 1, 4, pso::DxgiFormat::R8G8B8A8_UNORM);
    reg_tex(13, 0x0000'1000'0004'0000ull, GpuTextureDimension::Dim3D, 64, 32, 16, 4, 1, pso::DxgiFormat::R11G11B10_FLOAT);
    reg_tex(14, 0x0000'1000'0005'0000ull, GpuTextureDimension::Dim2D, 128, 128, 18, 4, 1, pso::DxgiFormat::BC7_UNORM);
    reg_tex(15, 0x0000'1000'0006'0000ull, GpuTextureDimension::Dim2D, 128, 128, 6, 4, 1, pso::DxgiFormat::R9G9B9E5_SHAREDEXP);

    uint32_t words[8] = {};
    d3d::ViewDescriptor back{};

    // 1D & 1DArray
    {
        d3d::ViewDescriptor v{};
        v.kind = d3d::DescriptorKind::Srv;
        v.resource_id = 10;
        v.dimension = d3d::ViewDimension::Texture1D;
        v.component_mapping = d3d::kDefaultShader4ComponentMapping;
        v.first_mip = 1;
        v.mip_levels = 3;
        CHECK(bridge::view_words(device, v, words) == bridge::WordsStatus::Ok);
        CHECK(plausible_image(words) && decode_image(words).type == ImageType::Tex1D);
        CHECK(bridge::decode_view_words(device, words, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.dimension == d3d::ViewDimension::Texture1D && back.first_mip == 1 && back.mip_levels == 3 &&
              back.first_array_slice == 0 && back.array_size == 1);

        v.dimension = d3d::ViewDimension::Texture1DArray;
        v.first_array_slice = 2;
        v.array_size = 5;
        CHECK(bridge::view_words(device, v, words) == bridge::WordsStatus::Ok);
        CHECK(decode_image(words).type == ImageType::Tex1DArray);
        CHECK(bridge::decode_view_words(device, words, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.dimension == d3d::ViewDimension::Texture1DArray && back.first_array_slice == 2 && back.array_size == 5);
    }

    // 2D, 2DArray, min_lod_clamp, and UAV single-mip round-trip
    {
        d3d::ViewDescriptor v{};
        v.kind = d3d::DescriptorKind::Srv;
        v.resource_id = 11;
        v.dimension = d3d::ViewDimension::Texture2DArray;
        v.component_mapping = d3d::kDefaultShader4ComponentMapping;
        v.first_mip = 2;
        v.mip_levels = 4;
        v.first_array_slice = 3;
        v.array_size = 6;
        v.min_lod_clamp = 1.5f;
        CHECK(bridge::view_words(device, v, words) == bridge::WordsStatus::Ok);
        const ImageResource dec = decode_image(words);
        CHECK(dec.type == ImageType::Tex2DArray && dec.base_level == 2 && dec.last_level == 5 && dec.max_mip == 5 &&
              dec.base_array == 3 && dec.depth == 9 && dec.min_lod == lod_to_u4_8(1.5f));
        CHECK(bridge::decode_view_words(device, words, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.dimension == d3d::ViewDimension::Texture2DArray && back.format == 10 && back.first_mip == 2 &&
              back.mip_levels == 4 && back.first_array_slice == 3 && back.array_size == 6 && back.min_lod_clamp == 1.5f);

        d3d::ViewDescriptor uav{};
        uav.kind = d3d::DescriptorKind::Uav;
        uav.resource_id = 11;
        uav.dimension = d3d::ViewDimension::Texture2DArray;
        uav.first_mip = 3;
        uav.first_array_slice = 4;
        uav.array_size = 2;
        CHECK(bridge::view_words(device, uav, words) == bridge::WordsStatus::Ok);
        CHECK(bridge::decode_view_words(device, words, d3d::DescriptorRangeType::Uav, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.kind == d3d::DescriptorKind::Uav && back.dimension == d3d::ViewDimension::Texture2DArray &&
              back.first_mip == 3 && back.mip_levels == 1 && back.first_array_slice == 4 && back.array_size == 2);
    }

    // 2DMS & 2DMSArray (4x MSAA -> last_level = log2(4) = 2)
    {
        d3d::ViewDescriptor v{};
        v.kind = d3d::DescriptorKind::Srv;
        v.resource_id = 12;
        v.dimension = d3d::ViewDimension::Texture2DMs;
        v.component_mapping = d3d::kDefaultShader4ComponentMapping;
        CHECK(bridge::view_words(device, v, words) == bridge::WordsStatus::Ok);
        const ImageResource ms = decode_image(words);
        CHECK(ms.type == ImageType::Tex2DMsaa && ms.base_level == 0 && ms.last_level == 2 && ms.max_mip == 2);
        CHECK(bridge::decode_view_words(device, words, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.dimension == d3d::ViewDimension::Texture2DMs && back.first_mip == 0 && back.mip_levels == 1);

        v.dimension = d3d::ViewDimension::Texture2DMsArray;
        v.first_array_slice = 1;
        v.array_size = 4;
        CHECK(bridge::view_words(device, v, words) == bridge::WordsStatus::Ok);
        const ImageResource msa = decode_image(words);
        CHECK(msa.type == ImageType::Tex2DMsaaArray && msa.base_array == 1 && msa.depth == 5 && msa.last_level == 2);
        CHECK(bridge::decode_view_words(device, words, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.dimension == d3d::ViewDimension::Texture2DMsArray && back.first_array_slice == 1 &&
              back.array_size == 4 && back.first_mip == 0 && back.mip_levels == 1);
    }

    // 3D texture
    {
        d3d::ViewDescriptor v{};
        v.kind = d3d::DescriptorKind::Srv;
        v.resource_id = 13;
        v.dimension = d3d::ViewDimension::Texture3D;
        v.component_mapping = d3d::kDefaultShader4ComponentMapping;
        v.first_mip = 1;
        v.mip_levels = 3;
        CHECK(bridge::view_words(device, v, words) == bridge::WordsStatus::Ok);
        const ImageResource d3 = decode_image(words);
        CHECK(d3.type == ImageType::Tex3D && d3.width == 64 && d3.height == 32 && d3.depth == 16 &&
              d3.base_level == 1 && d3.last_level == 3 && d3.format == 36);
        CHECK(bridge::decode_view_words(device, words, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.dimension == d3d::ViewDimension::Texture3D && back.format == 26 && back.first_mip == 1 &&
              back.mip_levels == 3 && back.array_size == 16);
    }

    // Cube (6 faces) and CubeArray (First2DArrayFace == 0 & > 0, NumCubes == 1, 2, unbounded)
    {
        d3d::ViewDescriptor cube{};
        cube.kind = d3d::DescriptorKind::Srv;
        cube.resource_id = 15;
        cube.dimension = d3d::ViewDimension::TextureCube;
        cube.component_mapping = d3d::kDefaultShader4ComponentMapping;
        cube.first_mip = 0;
        cube.mip_levels = 4;
        CHECK(bridge::view_words(device, cube, words) == bridge::WordsStatus::Ok);
        const ImageResource ci = decode_image(words);
        CHECK(ci.type == ImageType::Cube && ci.base_array == 0 && ci.depth == 6 && ci.format == 132);
        CHECK(bridge::decode_view_words(device, words, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.dimension == d3d::ViewDimension::TextureCube && back.format == 67 && back.first_array_slice == 0 &&
              back.array_size == 1 && back.mip_levels == 4);

        // CubeArray with First2DArrayFace = 6, NumCubes = 1 (must stay TextureCubeArray, not collapse to TextureCube!)
        d3d::ViewDescriptor ca{};
        ca.kind = d3d::DescriptorKind::Srv;
        ca.resource_id = 14;
        ca.dimension = d3d::ViewDimension::TextureCubeArray;
        ca.component_mapping = d3d::kDefaultShader4ComponentMapping;
        ca.first_mip = 1;
        ca.mip_levels = 3;
        ca.first_array_slice = 6;
        ca.array_size = 1;
        CHECK(bridge::view_words(device, ca, words) == bridge::WordsStatus::Ok);
        const ImageResource ca1 = decode_image(words);
        CHECK(ca1.type == ImageType::Cube && ca1.base_array == 6 && ca1.depth == 12);
        CHECK(bridge::decode_view_words(device, words, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.dimension == d3d::ViewDimension::TextureCubeArray && back.format == 98 &&
              back.first_array_slice == 6 && back.array_size == 1 && back.first_mip == 1 && back.mip_levels == 3);

        // CubeArray with First2DArrayFace = 6, NumCubes = 2 (12 faces -> depth = 18)
        ca.array_size = 2;
        CHECK(bridge::view_words(device, ca, words) == bridge::WordsStatus::Ok);
        CHECK(decode_image(words).base_array == 6 && decode_image(words).depth == 18);
        CHECK(bridge::decode_view_words(device, words, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.dimension == d3d::ViewDimension::TextureCubeArray && back.first_array_slice == 6 && back.array_size == 2);

        // CubeArray with First2DArrayFace = 0, NumCubes = 0xffffffff (all 3 cubes of the 18-layer texture)
        ca.first_array_slice = 0;
        ca.array_size = 0xffffffffu;
        CHECK(bridge::view_words(device, ca, words) == bridge::WordsStatus::Ok);
        CHECK(decode_image(words).base_array == 0 && decode_image(words).depth == 18);
        CHECK(bridge::decode_view_words(device, words, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.dimension == d3d::ViewDimension::TextureCubeArray && back.first_array_slice == 0 && back.array_size == 3);
    }

    // image_words_at_address: 3D, CubeArray, error statuses
    {
        bridge::AddressTexture at;
        at.dimension = 4;  // 3D
        at.width = 32;
        at.height = 32;
        at.depth_or_array_size = 8;
        at.mip_levels = 3;
        at.format = 28;
        d3d::ViewDescriptor v{};
        v.kind = d3d::DescriptorKind::Srv;
        v.dimension = d3d::ViewDimension::Texture3D;
        v.component_mapping = d3d::kDefaultShader4ComponentMapping;
        v.mip_levels = 0xffffu;
        CHECK(bridge::image_words_at_address(0x0000'2000'0000'0100ull, at, v, words) == bridge::WordsStatus::Ok);
        const ImageResource dec = decode_image(words);
        CHECK(dec.base == 0x0000'2000'0000'0100ull && dec.type == ImageType::Tex3D && dec.depth == 8 && dec.last_level == 2);
        CHECK(bridge::image_words_at_address(0, at, v, words) == bridge::WordsStatus::NoAddress);
        CHECK(bridge::image_words_at_address(0x10080ull, at, v, words) == bridge::WordsStatus::MisalignedAddress);
        at.format = 100;  // unsupported video format
        CHECK(bridge::image_words_at_address(0x10000ull, at, v, words) == bridge::WordsStatus::UnsupportedFormat);
    }
}

// G83 (#1748 / #651): complete DXGI format encode/decode coverage across sRGB/typeless, UNORM/SNORM,
// FLOAT/HDR, UINT/SINT, BC1-7, Depth/Stencil planes (including packed GFX10 IMG_FMT 142..146), and
// custom SRV component mappings on BGRA / R8 / sRGB views.
void test_g83_formats_depth_srgb_and_swizzles() {
    namespace bridge = playgta6::gpu::bridge;
    namespace pso = playgta6::gpu::pso;

    // 1. Every concrete 3D shader format class round-trips through T# encode/decode.
    const uint32_t kAllFormats[] = {
        // FLOAT & HDR
        2, 6, 10, 16, 26, 34, 41, 54, 67,
        // UNORM / SNORM / sRGB / packed
        11, 13, 24, 28, 29, 31, 35, 37, 49, 51, 56, 58, 61, 63, 65, 85, 86, 87, 88, 91, 93, 115,
        // UINT / SINT (8/16/32-bit, 1/2/3/4-comp, R10G10B10A2_UINT)
        3, 4, 7, 8, 12, 14, 17, 18, 25, 30, 32, 36, 38, 42, 43, 50, 52, 57, 59, 62, 64,
        // BC1-BC7 (linear, sRGB, SNORM, HDR UF16/SF16)
        71, 72, 74, 75, 77, 78, 80, 81, 83, 84, 95, 96, 98, 99,
        // Depth / Stencil + Xbox D16S8
        20, 21, 22, 40, 45, 46, 47, 55, 118, 119, 120,
    };
    for (const uint32_t dxgi : kAllFormats) {
        HardwareFormat hf;
        CHECK(hardware_format(dxgi, &hf));
        ImageResource img;
        img.base = 0x0000'0040'0001'0000ull;
        img.format = hf.format;
        img.dst_sel = hf.swizzle;
        img.width = 16;
        img.height = 16;
        uint32_t w[8] = {};
        encode(img, w);
        CHECK(plausible_image(w) && classify_slot(w) == SlotContent::Image);
        const ImageResource dec = decode_image(w);
        bool exact = false;
        const uint32_t back = dxgi_format(dec.format, dec.dst_sel, &exact);
        CHECK(exact && back != 0);
    }

    // 2. Packed GFX10 depth/stencil IMG_FMTs 142..146 reverse-lookup cleanly.
    // Note: IMG_FMT_24_8_UNORM (143) is MSB-to-LSB, so 24-bit depth lives in Y [31:8].
    bool exact = false;
    CHECK(dxgi_format(142, Swizzle{Sel::Y, Sel::Zero, Sel::Zero, Sel::One}, &exact) == 47 && exact);
    CHECK(dxgi_format(143, Swizzle{Sel::Y, Sel::Zero, Sel::Zero, Sel::One}, &exact) == 46 && exact);
    CHECK(dxgi_format(144, Swizzle{Sel::X, Sel::Zero, Sel::Zero, Sel::One}, &exact) == 47 && exact);
    CHECK(dxgi_format(145, Swizzle{Sel::Y, Sel::Zero, Sel::Zero, Sel::One}, &exact) == 22 && exact);
    CHECK(dxgi_format(146, Swizzle{Sel::X, Sel::Zero, Sel::Zero, Sel::One}, &exact) == 21 && exact);

    // 3. View-word decode over typeless sRGB textures and depth-stencil textures preserves exact view formats.
    d3d::DescriptorHeapRegistry heaps;
    bridge::StateBridgeDevice device(heaps);
    auto reg2d = [&](uint32_t id, uint64_t va, pso::DxgiFormat fmt) {
        bridge::TextureInfo ti;
        ti.format = fmt;
        ti.width = 64;
        ti.height = 64;
        ti.mip_levels = 1;
        ti.gpu_va = va;
        device.register_texture(id, ti);
    };
    reg2d(20, 0x0000'3000'0001'0000ull, pso::DxgiFormat::R8G8B8A8_TYPELESS);
    reg2d(21, 0x0000'3000'0002'0000ull, pso::DxgiFormat::B8G8R8A8_TYPELESS);
    reg2d(22, 0x0000'3000'0003'0000ull, pso::DxgiFormat::BC7_TYPELESS);
    reg2d(23, 0x0000'3000'0004'0000ull, pso::DxgiFormat::R24G8_TYPELESS);
    reg2d(24, 0x0000'3000'0005'0000ull, pso::DxgiFormat::D32_FLOAT_S8X24_UINT);
    reg2d(25, 0x0000'3000'0006'0000ull, pso::DxgiFormat::D16_UNORM_S8_UINT);
    reg2d(26, 0x0000'3000'0007'0000ull, pso::DxgiFormat::B8G8R8A8_UNORM);
    reg2d(27, 0x0000'3000'0008'0000ull, pso::DxgiFormat::R8_UNORM);

    uint32_t w[8] = {};
    d3d::ViewDescriptor back{};
    auto check_view_fmt = [&](uint32_t id, uint32_t req_fmt, uint32_t expected_fmt,
                              uint32_t mapping = d3d::kDefaultShader4ComponentMapping) {
        d3d::ViewDescriptor v{};
        v.kind = d3d::DescriptorKind::Srv;
        v.resource_id = id;
        v.dimension = d3d::ViewDimension::Texture2D;
        v.format = req_fmt;
        v.component_mapping = mapping;
        CHECK(bridge::view_words(device, v, w) == bridge::WordsStatus::Ok);
        CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.format == expected_fmt && back.component_mapping == mapping);
    };
    // Typeless sRGB & linear views
    check_view_fmt(20, 29, 29);  // R8G8B8A8_TYPELESS -> R8G8B8A8_UNORM_SRGB
    check_view_fmt(20, 28, 28);  // R8G8B8A8_TYPELESS -> R8G8B8A8_UNORM
    check_view_fmt(21, 91, 91);  // B8G8R8A8_TYPELESS -> B8G8R8A8_UNORM_SRGB
    check_view_fmt(21, 87, 87);  // B8G8R8A8_TYPELESS -> B8G8R8A8_UNORM
    check_view_fmt(22, 99, 99);  // BC7_TYPELESS -> BC7_UNORM_SRGB
    check_view_fmt(22, 98, 98);  // BC7_TYPELESS -> BC7_UNORM
    // Depth & stencil plane views
    check_view_fmt(23, 46, 46);   // R24G8_TYPELESS -> R24_UNORM_X8_TYPELESS (depth)
    check_view_fmt(23, 47, 47);   // R24G8_TYPELESS -> X24_TYPELESS_G8_UINT (stencil)
    check_view_fmt(24, 21, 21);   // D32_FLOAT_S8X24_UINT -> R32_FLOAT_X8X24_TYPELESS (depth)
    check_view_fmt(24, 22, 22);   // D32_FLOAT_S8X24_UINT -> X32_TYPELESS_G8X24_UINT (stencil)
    check_view_fmt(25, 119, 119); // D16_UNORM_S8_UINT -> R16_UNORM_X8_TYPELESS (depth)
    check_view_fmt(25, 120, 120); // D16_UNORM_S8_UINT -> X16_TYPELESS_G8_UINT (stencil)

    // Custom SRV swizzles that collide with another DXGI format entry still round-trip accurately!
    const uint32_t kRgb1 = 0u | (1u << 3) | (2u << 6) | (5u << 9) | (1u << 12);  // (R, G, B, 1)
    const uint32_t kBgra = 2u | (1u << 3) | (0u << 6) | (3u << 9) | (1u << 12);  // (B, G, R, A)
    const uint32_t k000R = 4u | (4u << 3) | (4u << 6) | (0u << 9) | (1u << 12);  // (0, 0, 0, R)
    check_view_fmt(26, 87, 87, kRgb1);  // B8G8R8A8_UNORM with (R, G, B, 1) (collides with BGRX8 swizzle)
    check_view_fmt(26, 91, 91, kRgb1);  // B8G8R8A8_UNORM_SRGB with (R, G, B, 1)
    check_view_fmt(26, 87, 87, kBgra);  // B8G8R8A8_UNORM with (B, G, R, A) (collides with RGBA8 swizzle)
    check_view_fmt(27, 61, 61, k000R);  // R8_UNORM with (0, 0, 0, R) (collides with A8_UNORM swizzle)

    // Raw packed IMG_FMT 142 (8_24_UINT stencil) & 146 (X24_8_32_FLOAT depth) decode cleanly via decode_view_words.
    ImageResource packed_stencil;
    packed_stencil.base = 0x0000'3000'0004'0000ull;  // texture 23 (R24G8_TYPELESS)
    packed_stencil.format = 142;
    packed_stencil.dst_sel = Swizzle{Sel::Y, Sel::Zero, Sel::Zero, Sel::One};
    encode(packed_stencil, w);
    CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
    CHECK(back.resource_id == 23 && back.format == 47 && back.component_mapping == d3d::kDefaultShader4ComponentMapping);

    ImageResource packed_depth;
    packed_depth.base = 0x0000'3000'0005'0000ull;  // texture 24 (D32_FLOAT_S8X24_UINT)
    packed_depth.format = 146;
    packed_depth.dst_sel = Swizzle{Sel::X, Sel::Zero, Sel::Zero, Sel::One};
    encode(packed_depth, w);
    CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
    CHECK(back.resource_id == 24 && back.format == 21 && back.component_mapping == d3d::kDefaultShader4ComponentMapping);
}

// G83 (#1748 / #651): complete V# (CBV, raw, structured, typed, 48-bit VAs, OOB modes) and
// S# (plausible_sampler, comparison funcs, aniso point/linear mip, border colors, LODs) coverage.
void test_g83_buffers_and_samplers() {
    namespace bridge = playgta6::gpu::bridge;
    d3d::DescriptorHeapRegistry heaps;
    bridge::StateBridgeDevice device(heaps);

    const uint64_t kHighVa = 0x0000'7abc'1234'0000ull;  // 48-bit GPU VA
    device.register_buffer(100, kHighVa, 4096);

    uint32_t w[8] = {};
    d3d::ViewDescriptor back{};

    // CBV with 48-bit VA
    {
        d3d::ViewDescriptor cbv{};
        cbv.kind = d3d::DescriptorKind::Cbv;
        cbv.first_element_or_address = kHighVa + 0x100;
        cbv.num_elements_or_size = 240;  // 15 * 16
        CHECK(bridge::view_words(device, cbv, w) == bridge::WordsStatus::Ok);
        CHECK(plausible_buffer(w) && classify_slot(w) == SlotContent::Buffer);
        CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Cbv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.first_element_or_address == kHighVa + 0x100 && back.num_elements_or_size == 240);
    }

    // Raw buffer SRV & UAV (Oob::Complete)
    {
        d3d::ViewDescriptor raw{};
        raw.kind = d3d::DescriptorKind::Srv;
        raw.resource_id = 100;
        raw.dimension = d3d::ViewDimension::Buffer;
        raw.flags = d3d::BufferViewFlags::Raw;
        raw.first_element_or_address = 16;  // 64 bytes
        raw.num_elements_or_size = 32;      // 128 bytes
        CHECK(bridge::view_words(device, raw, w) == bridge::WordsStatus::Ok);
        const BufferResource b = decode_buffer(w);
        CHECK(b.base == kHighVa + 64 && b.stride == 0 && b.num_records == 128 && b.oob_select == Oob::Complete);
        CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.resource_id == 100 && back.first_element_or_address == 64 && back.num_elements_or_size == 128);
    }

    // Structured buffer UAV (Oob::IndexAndOffset)
    {
        d3d::ViewDescriptor st{};
        st.kind = d3d::DescriptorKind::Uav;
        st.resource_id = 100;
        st.dimension = d3d::ViewDimension::Buffer;
        st.structure_byte_stride = 48;
        st.first_element_or_address = 5;   // 240 bytes
        st.num_elements_or_size = 10;      // 480 bytes
        CHECK(bridge::view_words(device, st, w) == bridge::WordsStatus::Ok);
        const BufferResource b = decode_buffer(w);
        CHECK(b.base == kHighVa + 240 && b.stride == 48 && b.num_records == 10 && b.oob_select == Oob::IndexAndOffset);
        CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Uav, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.kind == d3d::DescriptorKind::Uav && back.first_element_or_address == 240 &&
              back.num_elements_or_size == 480);
    }

    // Typed buffer SRV (R16G16B16A16_FLOAT = 8 bytes/elem) + unsupported sRGB typed buffer rejection
    {
        d3d::ViewDescriptor tb{};
        tb.kind = d3d::DescriptorKind::Srv;
        tb.resource_id = 100;
        tb.dimension = d3d::ViewDimension::Buffer;
        tb.format = 10;  // R16G16B16A16_FLOAT
        tb.component_mapping = d3d::kDefaultShader4ComponentMapping;
        tb.first_element_or_address = 8;   // 64 bytes
        tb.num_elements_or_size = 16;      // 128 bytes
        CHECK(bridge::view_words(device, tb, w) == bridge::WordsStatus::Ok);
        const BufferResource b = decode_buffer(w);
        CHECK(b.base == kHighVa + 64 && b.stride == 8 && b.num_records == 16 && b.format == 71);
        CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.first_element_or_address == 64 && back.num_elements_or_size == 128);

        tb.format = 29;  // R8G8B8A8_UNORM_SRGB: image-only, not a typed buffer format
        CHECK(bridge::buffer_words_at_address(kHighVa, tb, w) == bridge::WordsStatus::UnsupportedFormat);
    }

    // Oob::IndexOnly & Oob::NumRecords0 decode byte window as num_records * stride; Oob::Complete uses num_records bytes.
    for (const uint32_t oob : {Oob::IndexAndOffset, Oob::IndexOnly, Oob::NumRecords0, Oob::Complete}) {
        BufferResource b;
        b.base = kHighVa + 128;
        b.stride = 16;
        b.num_records = 32;
        b.format = Fmt::F32;
        b.oob_select = oob;
        std::memset(w, 0, sizeof(w));
        encode(b, w);
        CHECK(plausible_buffer(w));
        CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        const uint32_t expected_bytes = oob == Oob::Complete ? 32u : 32u * 16u;
        CHECK(back.first_element_or_address == 128 && back.num_elements_or_size == expected_bytes);
    }

    // Buffer plausibility rejection: reserved bits 19..20, 25..27, invalid sel 2.
    {
        BufferResource b;
        b.format = Fmt::F32;
        encode(b, w);
        CHECK(plausible_buffer(w));
        w[3] |= 1u << 19;
        CHECK(!plausible_buffer(w));
        encode(b, w);
        w[3] |= 1u << 25;
        CHECK(!plausible_buffer(w));
        b.dst_sel[0] = 2;  // not in SQ_SEL_XYZW01
        encode(b, w);
        CHECK(!plausible_buffer(w));
    }

    // Sampler round-trips: comparison funcs 2..8, anisotropic point/linear mip, border colors, plausibility.
    for (uint32_t cmp = 2; cmp <= 8; ++cmp) {
        d3d::SamplerDesc sd;
        sd.filter = 0x80u | 0x15u;
        sd.comparison_func = cmp;
        sd.address_u = 2;  // MIRROR
        sd.address_v = 3;  // CLAMP
        sd.address_w = 4;  // BORDER
        sd.border_color[3] = 1.0f;  // opaque black
        sd.min_lod = 1.25f;
        sd.max_lod = 10.5f;
        sd.mip_lod_bias = -2.25f;
        uint32_t sw[4] = {};
        bridge::sampler_words(sd, sw);
        CHECK(plausible_sampler(sw));
        d3d::SamplerDesc dec;
        CHECK(bridge::decode_sampler_words(sw, &dec) == bridge::DecodeStatus::Ok);
        CHECK(dec.filter == sd.filter && dec.comparison_func == cmp && dec.address_u == 2 && dec.address_v == 3 &&
              dec.address_w == 4 && dec.border_color[3] == 1.0f && dec.min_lod == 1.25f && dec.max_lod == 10.5f &&
              dec.mip_lod_bias == -2.25f);
    }
    // Anisotropic with linear mip (0x55), point mip (0x54), and comparison aniso (0xd5), max_anisotropy 1..16.
    for (const uint32_t aniso_filter : {0x54u, 0x55u, 0xd5u, 0x155u, 0x1d5u}) {
        for (const uint32_t max_a : {1u, 2u, 4u, 8u, 16u}) {
            d3d::SamplerDesc sd;
            sd.filter = aniso_filter;
            sd.max_anisotropy = max_a;
            sd.comparison_func = (aniso_filter & 0x180u) == 0x80u ? 2u : 0u;
            uint32_t sw[4] = {};
            bridge::sampler_words(sd, sw);
            CHECK(plausible_sampler(sw));
            d3d::SamplerDesc dec;
            CHECK(bridge::decode_sampler_words(sw, &dec) == bridge::DecodeStatus::Ok);
            CHECK(dec.filter == aniso_filter && dec.max_anisotropy == max_a);
        }
    }
    // Sampler plausibility failures & Zero status.
    {
        uint32_t zero[4] = {};
        d3d::SamplerDesc dec;
        CHECK(bridge::decode_sampler_words(zero, &dec) == bridge::DecodeStatus::Zero);
        uint32_t bad[4] = {0, 0, 0, 0x1000u};  // reserved bit 12 in dword 3
        CHECK(!plausible_sampler(bad) && bridge::decode_sampler_words(bad, &dec) == bridge::DecodeStatus::NotDescriptor);
        SamplerState s;
        s.max_aniso_ratio = 5;  // out of range (> 4)
        encode(s, bad);
        CHECK(!plausible_sampler(bad) && bridge::decode_sampler_words(bad, &dec) == bridge::DecodeStatus::NotDescriptor);
        s = SamplerState{};
        s.filter_mode = 3;  // out of range (> 2)
        encode(s, bad);
        CHECK(!plausible_sampler(bad) && bridge::decode_sampler_words(bad, &dec) == bridge::DecodeStatus::NotDescriptor);
        s = SamplerState{};
        s.z_filter = 3;  // out of range (> 2)
        encode(s, bad);
        CHECK(!plausible_sampler(bad) && bridge::decode_sampler_words(bad, &dec) == bridge::DecodeStatus::NotDescriptor);
    }
}

// Console color, HDR/10-bit format, swizzle, and GFX10.3 V#/T#/S# fidelity tests.
void test_console_color_and_gfx103_fidelity() {
    // 1. GFX10.3 V# LLC_NOALLOC (dw3[27:26]) is accepted by plausible_buffer() / classify_slot()
    // and round-trips cleanly, while reserved bit 25 is still rejected.
    {
        for (uint32_t llc = 0; llc <= 3; ++llc) {
            BufferResource b;
            b.base = 0x0000'0010'0002'0000ull;
            b.stride = 32;
            b.num_records = 64;
            b.format = Fmt::F32;
            b.llc_noalloc = llc;
            uint32_t w[8] = {};
            encode(b, w);
            CHECK(((w[3] >> 26) & 3u) == llc);
            CHECK(plausible_buffer(w));
            CHECK(classify_slot(w) == SlotContent::Buffer);
            const BufferResource dec = decode_buffer(w);
            CHECK(dec.llc_noalloc == llc && dec.base == b.base && dec.stride == 32);
        }
    }

    // 2. GFX10.3 T# LLC_NOALLOC (dw6[9:8]) and Xbox HDR 10-bit float formats:
    //    116 = R10G10B10_7E3_A2_FLOAT (IMG_FMT 285), 117 = R10G10B10_6E4_A2_FLOAT (IMG_FMT 131),
    //    plus Xbox 189 (2_10_10_10_SNORM = 51) and 190 (4_4_UNORM = 137), and reverse-only 128/129/140.
    {
        struct HdrCase {
            uint32_t dxgi;
            uint32_t hw;
            Swizzle sel;
        };
        const HdrCase cases[] = {
            {116, 285, kIdentitySwizzle},
            {117, 131, kIdentitySwizzle},
            {189, 51, kIdentitySwizzle},
            {190, 137, Swizzle{Sel::X, Sel::Y, Sel::Zero, Sel::One}},
        };
        for (const HdrCase& c : cases) {
            HardwareFormat hf;
            CHECK(hardware_format(c.dxgi, &hf) && hf.format == c.hw && hf.swizzle == c.sel);
            ImageResource img;
            img.base = 0x0000'0020'0004'0000ull;
            img.format = hf.format;
            img.dst_sel = hf.swizzle;
            img.width = 128;
            img.height = 64;
            img.llc_noalloc = 2;
            uint32_t w[8] = {};
            encode(img, w);
            CHECK(plausible_image(w) && classify_slot(w) == SlotContent::Image);
            const ImageResource dec = decode_image(w);
            CHECK(dec.format == c.hw && dec.llc_noalloc == 2 && dec.dst_sel == c.sel);
            bool exact = false;
            CHECK(dxgi_format(dec.format, dec.dst_sel, &exact) == c.dxgi && exact);
        }

        // GFX10.3 formats 286..293 accepted by plausible_image(); 294 rejected.
        ImageResource img;
        img.base = 0x10000ull;
        img.format = 293;
        uint32_t w[8] = {};
        encode(img, w);
        CHECK(plausible_image(w));
        img.format = 294;
        encode(img, w);
        CHECK(!plausible_image(w));

        // Secondary image-only GFX10 formats 128 (8_SRGB), 129 (8_8_SRGB), 140 (32_FLOAT_CLAMP)
        // decode to their closest DXGI equivalents without hijacking canonical forward encode.
        bool exact = false;
        CHECK(dxgi_format(128, Swizzle{Sel::X, Sel::Zero, Sel::Zero, Sel::One}, &exact) == 61 && exact);
        CHECK(dxgi_format(129, Swizzle{Sel::X, Sel::Y, Sel::Zero, Sel::One}, &exact) == 49 && exact);
        CHECK(dxgi_format(140, Swizzle{Sel::X, Sel::Zero, Sel::Zero, Sel::One}, &exact) == 41 && exact);
        HardwareFormat canon;
        CHECK(hardware_format(61, &canon) && canon.format == 1);
        CHECK(hardware_format(49, &canon) && canon.format == 14);
        CHECK(hardware_format(41, &canon) && canon.format == 22);
    }

    // 3. BGRA non-canonical alpha swizzle preserves BGR order (no red/blue swap) and 111R preserves A8.
    {
        bool exact = true;
        CHECK(dxgi_format(56, Swizzle{Sel::Z, Sel::Y, Sel::X, Sel::Zero}, &exact) == 87 && !exact);
        CHECK(dxgi_format(130, Swizzle{Sel::Z, Sel::Y, Sel::X, Sel::Zero}, &exact) == 91 && !exact);
        CHECK(dxgi_format(1, Swizzle{Sel::One, Sel::One, Sel::One, Sel::X}, &exact) == 65 && !exact);
        // Non-BGR custom swizzle still falls back to R8G8B8A8_UNORM.
        CHECK(dxgi_format(56, Swizzle{Sel::W, Sel::Z, Sel::Y, Sel::X}, &exact) == 28 && !exact);
    }

    // 4. component_mapping_from() handles identity shortcuts and implicit missing hardware channels
    //    (Y=0, Z=0, W=1) in their canonical slots on 1/2/3-channel formats.
    {
        uint32_t m = 0;
        const Swizzle kX001 = {Sel::X, Sel::Zero, Sel::Zero, Sel::One};
        const Swizzle kXY01 = {Sel::X, Sel::Y, Sel::Zero, Sel::One};
        const Swizzle kXYZ1 = {Sel::X, Sel::Y, Sel::Z, Sel::One};
        // Exact format swizzle returns D3D12 default mapping (0x1688).
        CHECK(component_mapping_from(kX001, kX001, &m) && m == 0x1688u);
        // R8/BC4 with {X, X, X, W} (implicit W=1): maps to (R, R, R, A).
        const uint32_t kRRRA = 0u | (0u << 3) | (0u << 6) | (3u << 9) | (1u << 12);
        CHECK(component_mapping_from(kX001, Swizzle{Sel::X, Sel::X, Sel::X, Sel::W}, &m) && m == kRRRA);
        // BC5/R8G8 with {Y, X, Z, W} (implicit Z=0, W=1): maps to (G, R, B, A).
        const uint32_t kGRBA = 1u | (0u << 3) | (2u << 6) | (3u << 9) | (1u << 12);
        CHECK(component_mapping_from(kXY01, Swizzle{Sel::Y, Sel::X, Sel::Z, Sel::W}, &m) && m == kGRBA);
        // R11G11B10/BC6H with {Z, Y, X, W} (implicit W=1): maps to (B, G, R, A).
        const uint32_t kBGRA = 2u | (1u << 3) | (0u << 6) | (3u << 9) | (1u << 12);
        CHECK(component_mapping_from(kXYZ1, Swizzle{Sel::Z, Sel::Y, Sel::X, Sel::W}, &m) && m == kBGRA);
        // Invalid cross-slot missing channel {Y, 0, 0, 1} on R8 still fails.
        CHECK(!component_mapping_from(kX001, Swizzle{Sel::Y, Sel::Zero, Sel::Zero, Sel::One}, &m));
    }

    // 5. S# sampler fidelity: MIP_FILTER_NONE locks LOD to [0, 0], ANISO_POINT & asymmetric mag/min
    //    filters round-trip, degamma overrides are detected, and comparison-aware decode preserves NEVER.
    {
        SamplerState none_mip;
        none_mip.xy_mag_filter = 1;
        none_mip.xy_min_filter = 1;
        none_mip.mip_filter = 0;  // SQ_TEX_MIP_FILTER_NONE
        none_mip.min_lod = lod_to_u4_8(2.0f);
        none_mip.max_lod = lod_to_u4_8(12.0f);
        bool exact = false;
        const d3d::SamplerDesc d_none = d3d12_from_sampler(none_mip, &exact);
        CHECK(exact && d_none.min_lod == 0.0f && d_none.max_lod == 0.0f && d_none.filter == 0x14u);

        // ANISO_POINT (mag=2, min=2) and asymmetric point-mag / aniso-linear-min (mag=0, min=3).
        SamplerState aniso_pt;
        aniso_pt.xy_mag_filter = 2;  // ANISO_POINT
        aniso_pt.xy_min_filter = 2;  // ANISO_POINT
        aniso_pt.mip_filter = 1;
        aniso_pt.max_aniso_ratio = 2;  // 4x
        const d3d::SamplerDesc d_ap = d3d12_from_sampler(aniso_pt, &exact);
        CHECK(exact && d_ap.filter == 0x40u && d_ap.max_anisotropy == 4u);
        const SamplerState ap_back = sampler_from_d3d12(d_ap);
        CHECK(ap_back.xy_mag_filter == 2 && ap_back.xy_min_filter == 2 && ap_back.max_aniso_ratio == 2);

        SamplerState asym;
        asym.xy_mag_filter = 0;  // POINT mag
        asym.xy_min_filter = 3;  // ANISO_BILINEAR min
        asym.mip_filter = 2;
        asym.max_aniso_ratio = 3;  // 8x
        const d3d::SamplerDesc d_asym = d3d12_from_sampler(asym);
        CHECK(d_asym.filter == 0x51u && d_asym.max_anisotropy == 8u);

        // Degamma & unnormalized & register border exactness reporting.
        SamplerState dg;
        dg.mip_filter = 1;
        CHECK(sampler_degamma_override(dg) == DegammaOverride::None);
        dg.force_degamma = true;
        CHECK(sampler_degamma_override(dg) == DegammaOverride::ForceDegamma);
        (void)d3d12_from_sampler(dg, &exact);
        CHECK(!exact);
        dg.skip_degamma = true;
        CHECK(sampler_degamma_override(dg) == DegammaOverride::SkipDegamma);

        SamplerState unnorm;
        unnorm.mip_filter = 1;
        unnorm.force_unnormalized = true;
        (void)d3d12_from_sampler(unnorm, &exact);
        CHECK(!exact);

        SamplerState reg_border;
        reg_border.mip_filter = 1;
        reg_border.border_color_type = 3;  // REGISTER
        (void)d3d12_from_sampler(reg_border, &exact);
        CHECK(!exact);

        // Comparison-aware decode: SQ_TEX_DEPTH_COMPARE_NEVER (0) -> D3D12_COMPARISON_FUNC_NEVER (1)
        // when shader uses image_sample_c*, and stripped when shader uses regular image_sample*.
        SamplerState cmp_never;
        cmp_never.xy_mag_filter = 1;
        cmp_never.xy_min_filter = 1;
        cmp_never.mip_filter = 1;
        cmp_never.depth_compare_func = 0;  // SQ_TEX_DEPTH_COMPARE_NEVER
        const d3d::SamplerDesc d_cmp = d3d12_from_sampler_for_comparison(cmp_never, true, &exact);
        CHECK(exact && (d_cmp.filter & 0x180u) == 0x80u && d_cmp.comparison_func == 1u);

        SamplerState color_with_cmp;
        color_with_cmp.xy_mag_filter = 1;
        color_with_cmp.xy_min_filter = 1;
        color_with_cmp.mip_filter = 1;
        color_with_cmp.depth_compare_func = 3;  // stale LESS_EQUAL on non-comparison sampler
        const d3d::SamplerDesc d_color = d3d12_from_sampler_for_comparison(color_with_cmp, false, &exact);
        CHECK(exact && (d_color.filter & 0x180u) == 0u && d_color.comparison_func == 0u);
    }

    // 6. GFX10.3 linear 1D/2D T# pitch vs effective depth.
    {
        ImageResource lin2d;
        lin2d.type = ImageType::Tex2D;
        lin2d.sw_mode = 0;  // SW_LINEAR
        lin2d.width = 1920;
        lin2d.height = 1080;
        lin2d.depth = 2048;  // word4 stores pitch - 1 = 2047 -> decoded depth = 2048
        CHECK(image_effective_depth(lin2d) == 1u);
        CHECK(image_linear_pitch_texels(lin2d) == 2048u);

        ImageResource arr2d = lin2d;
        arr2d.type = ImageType::Tex2DArray;
        arr2d.depth = 6;
        CHECK(image_effective_depth(arr2d) == 6u);
        CHECK(image_linear_pitch_texels(arr2d) == 1920u);

        ImageResource tiled2d = lin2d;
        tiled2d.sw_mode = 27;  // 64KB_S_X
        tiled2d.depth = 1;
        CHECK(image_effective_depth(tiled2d) == 1u);
        CHECK(image_linear_pitch_texels(tiled2d) == 1920u);
    }
}

}  // namespace

int main() {
    test_buffer_field_positions();
    test_image_field_positions();
    test_sampler_field_positions();
    test_round_trips();
    test_classify_slot();
    test_formats();
    test_component_mapping();
    test_samplers();
    test_bgra8_uav_storage_usage();
    test_g83_3d_view_dimensions();
    test_g83_formats_depth_srgb_and_swizzles();
    test_g83_buffers_and_samplers();
    test_console_color_and_gfx103_fidelity();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_gfx10_descriptors failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All gfx10_descriptors_tests passed.\n");
    return 0;
}
