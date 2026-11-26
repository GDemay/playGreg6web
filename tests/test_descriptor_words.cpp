// Unit tests for runtime/gpu/descriptor_words.{h,cpp}: D3D12 Shader4ComponentMapping -> GFX10 T#/V#
// word 3 DST_SEL_X/Y/Z/W swizzle encoding, zero-mapping normalization, single-channel R8/A8 fonts
// & masks, BGRA<->RGBA swizzles, and typeless concrete-view decode round-trips.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpu/descriptor_words.h"
#include "gpu/state_bridge.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_descriptor_words.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl((expr), #expr, __LINE__)

constexpr uint32_t encode_mapping(uint32_t r, uint32_t g, uint32_t b, uint32_t a) noexcept {
    return (r & 7u) | ((g & 7u) << 3u) | ((b & 7u) << 6u) | ((a & 7u) << 9u) | (1u << 12u);
}

constexpr uint32_t extract_dst_sel(uint32_t word3, uint32_t lane) noexcept {
    return (word3 >> (3u * lane)) & 7u;
}

void check_word3_dst_sel(const uint32_t words[8], uint8_t sx, uint8_t sy, uint8_t sz, uint8_t sw) {
    CHECK(extract_dst_sel(words[3], 0) == sx);
    CHECK(extract_dst_sel(words[3], 1) == sy);
    CHECK(extract_dst_sel(words[3], 2) == sz);
    CHECK(extract_dst_sel(words[3], 3) == sw);
}

void test_normalize_and_srv_swizzle_helper() {
    namespace bridge = playgta6::gpu::bridge;
    namespace d3d = playgta6::gpu::d3d12_state;
    namespace hw = playgta6::gpu::gfx10;

    // 0 (zero-initialized desc) normalizes to D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING (0x1688).
    CHECK(bridge::normalize_component_mapping(0u) == d3d::kDefaultShader4ComponentMapping);
    CHECK(bridge::normalize_component_mapping(d3d::kDefaultShader4ComponentMapping) ==
          d3d::kDefaultShader4ComponentMapping);
    // Explicit (R, R, R, R) has bit 12 set (0x1000) and must NOT be confused with 0.
    constexpr uint32_t kRRRR = encode_mapping(0, 0, 0, 0);
    CHECK(kRRRR == 0x1000u);
    CHECK(bridge::normalize_component_mapping(kRRRR) == kRRRR);

    hw::Swizzle s{};
    // R8_UNORM (61) canonical swizzle is (X, 0, 0, 1).
    CHECK(bridge::srv_component_swizzle(61, 0u, &s));
    CHECK(s == (hw::Swizzle{hw::Sel::X, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::One}));

    constexpr uint32_t kRRR1 = encode_mapping(0, 0, 0, 5);
    CHECK(bridge::srv_component_swizzle(61, kRRR1, &s));
    CHECK(s == (hw::Swizzle{hw::Sel::X, hw::Sel::X, hw::Sel::X, hw::Sel::One}));

    constexpr uint32_t k000R = encode_mapping(4, 4, 4, 0);
    CHECK(bridge::srv_component_swizzle(61, k000R, &s));
    CHECK(s == (hw::Swizzle{hw::Sel::Zero, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::X}));

    constexpr uint32_t k111R = encode_mapping(5, 5, 5, 0);
    CHECK(bridge::srv_component_swizzle(61, k111R, &s));
    CHECK(s == (hw::Swizzle{hw::Sel::One, hw::Sel::One, hw::Sel::One, hw::Sel::X}));

    // A8_UNORM (65) canonical swizzle is (0, 0, 0, X); zero mapping must NOT collapse to (0,0,0,0)!
    CHECK(bridge::srv_component_swizzle(65, 0u, &s));
    CHECK(s == (hw::Swizzle{hw::Sel::Zero, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::X}));

    constexpr uint32_t kAAAA = encode_mapping(3, 3, 3, 3);
    CHECK(bridge::srv_component_swizzle(65, kAAAA, &s));
    CHECK(s == (hw::Swizzle{hw::Sel::X, hw::Sel::X, hw::Sel::X, hw::Sel::X}));

    constexpr uint32_t k111A = encode_mapping(5, 5, 5, 3);
    CHECK(bridge::srv_component_swizzle(65, k111A, &s));
    CHECK(s == (hw::Swizzle{hw::Sel::One, hw::Sel::One, hw::Sel::One, hw::Sel::X}));

    // B8G8R8A8_UNORM (87) canonical swizzle is (Z, Y, X, W).
    CHECK(bridge::srv_component_swizzle(87, 0u, &s));
    CHECK(s == (hw::Swizzle{hw::Sel::Z, hw::Sel::Y, hw::Sel::X, hw::Sel::W}));
    constexpr uint32_t kBGRA = encode_mapping(2, 1, 0, 3);
    CHECK(bridge::srv_component_swizzle(87, kBGRA, &s));
    CHECK(s == (hw::Swizzle{hw::Sel::X, hw::Sel::Y, hw::Sel::Z, hw::Sel::W}));

    // Unsupported format returns false.
    CHECK(!bridge::srv_component_swizzle(0u, d3d::kDefaultShader4ComponentMapping, &s));
}

void test_texture_srv_word3_swizzles_and_roundtrips() {
    namespace bridge = playgta6::gpu::bridge;
    namespace d3d = playgta6::gpu::d3d12_state;
    namespace hw = playgta6::gpu::gfx10;
    namespace pso = playgta6::gpu::pso;

    d3d::DescriptorHeapRegistry heaps;
    bridge::StateBridgeDevice device(heaps);

    auto reg2d = [&](uint32_t id, uint64_t va, pso::DxgiFormat fmt) {
        bridge::TextureInfo ti{};
        ti.format = fmt;
        ti.width = 128;
        ti.height = 128;
        ti.mip_levels = 1;
        ti.gpu_va = va;
        device.register_texture(id, ti);
    };
    reg2d(1, 0x0000'2000'0001'0000ull, pso::DxgiFormat::R8_UNORM);
    reg2d(2, 0x0000'2000'0002'0000ull, pso::DxgiFormat::A8_UNORM);
    reg2d(3, 0x0000'2000'0003'0000ull, pso::DxgiFormat::B8G8R8A8_UNORM);
    reg2d(4, 0x0000'2000'0004'0000ull, pso::DxgiFormat::R8G8B8A8_UNORM);
    reg2d(5, 0x0000'2000'0005'0000ull, pso::DxgiFormat::R8_TYPELESS);
    reg2d(6, 0x0000'2000'0006'0000ull, pso::DxgiFormat::R16G16B16A16_TYPELESS);
    reg2d(7, 0x0000'2000'0007'0000ull, pso::DxgiFormat::R8G8B8A8_TYPELESS);

    uint32_t w[8] = {};
    d3d::ViewDescriptor back{};

    auto encode_and_check = [&](uint32_t id, uint32_t view_fmt, uint32_t mapping, uint8_t sx, uint8_t sy, uint8_t sz,
                                uint8_t sw, uint32_t expected_decoded_fmt, uint32_t expected_decoded_mapping) {
        d3d::ViewDescriptor v{};
        v.kind = d3d::DescriptorKind::Srv;
        v.resource_id = id;
        v.dimension = d3d::ViewDimension::Texture2D;
        v.format = view_fmt;
        v.component_mapping = mapping;
        CHECK(bridge::view_words(device, v, w) == bridge::WordsStatus::Ok);
        check_word3_dst_sel(w, sx, sy, sz, sw);
        CHECK(bridge::decode_view_words(device, w, d3d::DescriptorRangeType::Srv, &back) == bridge::DecodeStatus::Ok);
        CHECK(back.resource_id == id);
        CHECK(back.format == expected_decoded_fmt);
        CHECK(back.component_mapping == expected_decoded_mapping);
    };

    constexpr uint32_t kDefault = d3d::kDefaultShader4ComponentMapping;
    constexpr uint32_t kRRR1 = encode_mapping(0, 0, 0, 5);
    constexpr uint32_t k000R = encode_mapping(4, 4, 4, 0);
    constexpr uint32_t k111R = encode_mapping(5, 5, 5, 0);
    constexpr uint32_t kRRRR = encode_mapping(0, 0, 0, 0);
    constexpr uint32_t kAAAA = encode_mapping(3, 3, 3, 3);
    constexpr uint32_t k111A = encode_mapping(5, 5, 5, 3);
    constexpr uint32_t kBGRA = encode_mapping(2, 1, 0, 3);
    constexpr uint32_t kBGR1 = encode_mapping(2, 1, 0, 5);
    constexpr uint32_t kABGR = encode_mapping(3, 2, 1, 0);

    // R8_UNORM (61): fonts / UI masks / icons.
    encode_and_check(1, 61, 0u, hw::Sel::X, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::One, 61, kDefault);
    encode_and_check(1, 61, kDefault, hw::Sel::X, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::One, 61, kDefault);
    encode_and_check(1, 61, kRRR1, hw::Sel::X, hw::Sel::X, hw::Sel::X, hw::Sel::One, 61, kRRR1);
    encode_and_check(1, 61, k000R, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::X, 61, k000R);
    encode_and_check(1, 61, k111R, hw::Sel::One, hw::Sel::One, hw::Sel::One, hw::Sel::X, 61, k111R);
    encode_and_check(1, 61, kRRRR, hw::Sel::X, hw::Sel::X, hw::Sel::X, hw::Sel::X, 61, kRRRR);

    // A8_UNORM (65): zero mapping must preserve (0, 0, 0, X) rather than collapsing to (0, 0, 0, 0).
    encode_and_check(2, 65, 0u, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::X, 65, kDefault);
    encode_and_check(2, 65, kDefault, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::X, 65, kDefault);
    encode_and_check(2, 65, kAAAA, hw::Sel::X, hw::Sel::X, hw::Sel::X, hw::Sel::X, 65, kAAAA);
    encode_and_check(2, 65, k111A, hw::Sel::One, hw::Sel::One, hw::Sel::One, hw::Sel::X, 65, k111A);

    // B8G8R8A8_UNORM (87) & R8G8B8A8_UNORM (28): default, zero-init, and BGRA<->RGBA channel swaps.
    encode_and_check(3, 87, 0u, hw::Sel::Z, hw::Sel::Y, hw::Sel::X, hw::Sel::W, 87, kDefault);
    encode_and_check(3, 87, kBGRA, hw::Sel::X, hw::Sel::Y, hw::Sel::Z, hw::Sel::W, 87, kBGRA);
    encode_and_check(4, 28, 0u, hw::Sel::X, hw::Sel::Y, hw::Sel::Z, hw::Sel::W, 28, kDefault);
    encode_and_check(4, 28, kBGRA, hw::Sel::Z, hw::Sel::Y, hw::Sel::X, hw::Sel::W, 28, kBGRA);
    encode_and_check(4, 28, kRRR1, hw::Sel::X, hw::Sel::X, hw::Sel::X, hw::Sel::One, 28, kRRR1);

    // Non-sRGB typeless textures viewed with concrete format AND custom swizzles.
    encode_and_check(5, 61, k000R, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::X, 61, k000R);
    encode_and_check(5, 61, kRRR1, hw::Sel::X, hw::Sel::X, hw::Sel::X, hw::Sel::One, 61, kRRR1);
    encode_and_check(6, 10, kBGR1, hw::Sel::Z, hw::Sel::Y, hw::Sel::X, hw::Sel::One, 10, kBGR1);
    encode_and_check(7, 30, kABGR, hw::Sel::W, hw::Sel::Z, hw::Sel::Y, hw::Sel::X, 30, kABGR);
}

void test_address_image_and_typed_buffer_swizzles() {
    namespace bridge = playgta6::gpu::bridge;
    namespace d3d = playgta6::gpu::d3d12_state;
    namespace hw = playgta6::gpu::gfx10;

    uint32_t w[8] = {};
    // Slot-70 address texture (R8_UNORM): zero mapping defaults to (X,0,0,1); 111R encodes (1,1,1,X).
    bridge::AddressTexture at{};
    at.dimension = 3;
    at.width = 64;
    at.height = 64;
    at.depth_or_array_size = 1;
    at.mip_levels = 1;
    at.format = 61;  // R8_UNORM

    d3d::ViewDescriptor v{};
    v.kind = d3d::DescriptorKind::Srv;
    v.dimension = d3d::ViewDimension::Texture2D;
    v.format = 61;
    v.component_mapping = 0u;
    CHECK(bridge::image_words_at_address(0x0000'3000'0010'0000ull, at, v, w) == bridge::WordsStatus::Ok);
    check_word3_dst_sel(w, hw::Sel::X, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::One);

    v.component_mapping = encode_mapping(5, 5, 5, 0);  // 111R
    CHECK(bridge::image_words_at_address(0x0000'3000'0010'0000ull, at, v, w) == bridge::WordsStatus::Ok);
    check_word3_dst_sel(w, hw::Sel::One, hw::Sel::One, hw::Sel::One, hw::Sel::X);

    // Typed buffer SRV (R8_UNORM): zero mapping defaults to (X,0,0,1); RRR1 encodes (X,X,X,1).
    d3d::ViewDescriptor tb{};
    tb.kind = d3d::DescriptorKind::Srv;
    tb.dimension = d3d::ViewDimension::Buffer;
    tb.format = 61;
    tb.num_elements_or_size = 256;
    tb.component_mapping = 0u;
    CHECK(bridge::buffer_words_at_address(0x0000'4000'0000'0000ull, tb, w) == bridge::WordsStatus::Ok);
    check_word3_dst_sel(w, hw::Sel::X, hw::Sel::Zero, hw::Sel::Zero, hw::Sel::One);

    tb.component_mapping = encode_mapping(0, 0, 0, 5);  // RRR1
    CHECK(bridge::buffer_words_at_address(0x0000'4000'0000'0000ull, tb, w) == bridge::WordsStatus::Ok);
    check_word3_dst_sel(w, hw::Sel::X, hw::Sel::X, hw::Sel::X, hw::Sel::One);
}

}  // namespace

int main() {
    test_normalize_and_srv_swizzle_helper();
    test_texture_srv_word3_swizzles_and_roundtrips();
    test_address_image_and_typed_buffer_swizzles();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d test failure(s) in test_descriptor_words\n", g_failures);
        return 1;
    }
    std::printf("test_descriptor_words: all checks passed\n");
    return 0;
}
