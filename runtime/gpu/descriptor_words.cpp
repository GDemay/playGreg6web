#include "gpu/descriptor_words.h"

#include <algorithm>
#include <cstring>

#include "gpu/state_bridge.h"

namespace playgta6::gpu::bridge {

namespace d3d = playgta6::gpu::d3d12_state;
namespace hw = playgta6::gpu::gfx10;

namespace {

struct TextureShape {
    uint32_t dimension = 2;  // 1, 2, 3
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth_or_layers = 1;
    uint32_t mips = 1;
    uint32_t samples = 1;
    uint32_t format = 0;  // DXGI
};

uint32_t log2_u32(uint32_t v) noexcept {
    uint32_t r = 0;
    while (v > 1) {
        v >>= 1;
        ++r;
    }
    return r;
}

WordsStatus image_words(uint64_t address, const TextureShape& t, const d3d::ViewDescriptor& v, uint32_t out[8]) {
    if (address == 0) {
        return WordsStatus::NoAddress;
    }
    if (address % 256 != 0) {
        return WordsStatus::MisalignedAddress;
    }
    const uint32_t dxgi = v.format != 0 ? v.format : t.format;
    hw::HardwareFormat f;
    if (!hw::hardware_format(dxgi, &f)) {
        return WordsStatus::UnsupportedFormat;
    }
    hw::ImageResource img;
    img.base = address;
    img.format = f.format;
    img.dst_sel = v.kind == d3d::DescriptorKind::Srv
                      ? hw::compose_component_mapping(f.swizzle, normalize_component_mapping(v.component_mapping))
                      : f.swizzle;
    img.width = std::max(1u, t.width);
    img.height = std::max(1u, t.height);
    const uint32_t mips = std::max(1u, t.mips);
    img.max_mip = mips - 1;
    img.min_lod = v.kind == d3d::DescriptorKind::Srv ? hw::lod_to_u4_8(v.min_lod_clamp) : 0u;

    // Mip range: SRVs carry (first, count; 0xffff = the rest), UAVs one mip.
    const uint32_t first_mip = std::min<uint32_t>(v.first_mip, mips - 1);
    uint32_t last_mip = first_mip;
    if (v.kind == d3d::DescriptorKind::Srv) {
        const uint32_t count = v.mip_levels == 0xffffu || v.mip_levels == 0 ? mips - first_mip : v.mip_levels;
        last_mip = std::min(mips - 1, first_mip + count - 1);
    }
    img.base_level = first_mip;
    img.last_level = last_mip;

    uint32_t dimension = v.dimension;
    const uint32_t layers = std::max(1u, t.depth_or_layers);
    if (dimension == d3d::ViewDimension::Unknown) {  // default view of the whole resource
        if (t.dimension == 1) {
            dimension = layers > 1 ? d3d::ViewDimension::Texture1DArray : d3d::ViewDimension::Texture1D;
        } else if (t.dimension == 3) {
            dimension = d3d::ViewDimension::Texture3D;
        } else if (t.samples > 1) {
            dimension = layers > 1 ? d3d::ViewDimension::Texture2DMsArray : d3d::ViewDimension::Texture2DMs;
        } else {
            dimension = layers > 1 ? d3d::ViewDimension::Texture2DArray : d3d::ViewDimension::Texture2D;
        }
    }
    const uint32_t first_slice = std::min(v.first_array_slice, layers - 1);
    const uint32_t slice_count =
        v.array_size == 0xffffffffu || v.array_size == 0 ? layers - first_slice : std::min(v.array_size, layers - first_slice);
    switch (dimension) {
    case d3d::ViewDimension::Texture1D:
        img.type = hw::ImageType::Tex1D;
        break;
    case d3d::ViewDimension::Texture1DArray:
        img.type = hw::ImageType::Tex1DArray;
        img.base_array = first_slice;
        img.depth = first_slice + slice_count;
        break;
    case d3d::ViewDimension::Texture2D:
        img.type = hw::ImageType::Tex2D;
        break;
    case d3d::ViewDimension::Texture2DArray:
        img.type = hw::ImageType::Tex2DArray;
        img.base_array = first_slice;
        img.depth = first_slice + slice_count;
        break;
    case d3d::ViewDimension::Texture2DMs:
    case d3d::ViewDimension::Texture2DMsArray:
        img.type = dimension == d3d::ViewDimension::Texture2DMs ? hw::ImageType::Tex2DMsaa : hw::ImageType::Tex2DMsaaArray;
        img.base_level = 0;
        img.last_level = log2_u32(std::max(1u, t.samples));  // MSAA: last_level holds log2(samples)
        img.max_mip = img.last_level;
        if (dimension == d3d::ViewDimension::Texture2DMsArray) {
            img.base_array = first_slice;
            img.depth = first_slice + slice_count;
        }
        break;
    case d3d::ViewDimension::Texture3D:
        img.type = hw::ImageType::Tex3D;
        img.depth = layers;
        break;
    case d3d::ViewDimension::TextureCube:
        img.type = hw::ImageType::Cube;
        img.base_array = 0;
        img.depth = 6;
        break;
    case d3d::ViewDimension::TextureCubeArray: {
        img.type = hw::ImageType::Cube;
        // SRV cube arrays: First2DArrayFace / NumCubes in the array fields.
        const uint32_t max_cubes = layers > first_slice ? (layers - first_slice) / 6u : 0u;
        const uint32_t cubes = v.array_size == 0xffffffffu || v.array_size == 0
                                   ? max_cubes
                                   : (max_cubes > 0 ? std::min(v.array_size, max_cubes) : v.array_size);
        img.base_array = first_slice;
        img.depth = first_slice + 6u * std::max(1u, cubes);
        break;
    }
    default:
        return WordsStatus::UnsupportedView;
    }
    hw::encode(img, out);
    return WordsStatus::Ok;
}

}  // namespace

void constant_buffer_words(uint64_t address, uint32_t size, uint32_t out[4]) noexcept {
    out[0] = static_cast<uint32_t>(address);
    out[1] = static_cast<uint32_t>(address >> 32) & 0xffffu;
    out[1] |= kConsoleCbvStride << 16;
    out[2] = static_cast<uint32_t>((uint64_t{size} + kConsoleCbvStride - 1) / kConsoleCbvStride);
    out[3] = kConsoleCbvWord3;
}

bool srv_component_swizzle(uint32_t dxgi_format, uint32_t shader_4_component_mapping,
                           hw::Swizzle* out) noexcept {
    hw::HardwareFormat f;
    if (!hw::hardware_format(dxgi_format, &f)) {
        return false;
    }
    if (out != nullptr) {
        *out = hw::compose_component_mapping(f.swizzle, normalize_component_mapping(shader_4_component_mapping));
    }
    return true;
}

const char* words_status_name(WordsStatus status) noexcept {
    switch (status) {
    case WordsStatus::Ok: return "ok";
    case WordsStatus::Null: return "null descriptor";
    case WordsStatus::UnknownResource: return "unknown resource";
    case WordsStatus::NoAddress: return "texture without a descriptor address";
    case WordsStatus::MisalignedAddress: return "texture address not 256-byte aligned";
    case WordsStatus::UnsupportedFormat: return "format without a hardware format";
    case WordsStatus::UnsupportedView: return "view without descriptor words";
    }
    return "?";
}

WordsStatus view_words(const StateBridgeDevice& device, const d3d::ViewDescriptor& v, uint32_t out[8]) {
    std::memset(out, 0, kViewSlotBytes);
    switch (v.kind) {
    case d3d::DescriptorKind::Cbv:
        constant_buffer_words(v.first_element_or_address, v.num_elements_or_size, out);
        return v.first_element_or_address == 0 ? WordsStatus::Null : WordsStatus::Ok;
    case d3d::DescriptorKind::Srv:
    case d3d::DescriptorKind::Uav:
        break;
    case d3d::DescriptorKind::Empty:
        return WordsStatus::Null;
    default:
        return WordsStatus::UnsupportedView;
    }
    if (v.resource_id == 0) {
        return WordsStatus::Null;
    }
    TextureInfo tex;
    if (device.texture_info(v.resource_id, &tex)) {
        if (v.dimension == d3d::ViewDimension::Buffer) {
            return WordsStatus::UnsupportedView;
        }
        TextureShape t;
        t.dimension = tex.dimension == GpuTextureDimension::Dim1D ? 1u : (tex.dimension == GpuTextureDimension::Dim3D ? 3u : 2u);
        t.width = tex.width;
        t.height = tex.height;
        t.depth_or_layers = tex.depth_or_array_size;
        t.mips = tex.mip_levels;
        t.samples = tex.sample_count;
        t.format = static_cast<uint32_t>(tex.format);
        return image_words(tex.gpu_va, t, v, out);
    }
    const uint64_t va = device.buffer_gpu_va(v.resource_id);
    if (va == 0) {
        return WordsStatus::UnknownResource;
    }
    if (v.dimension != d3d::ViewDimension::Buffer && v.dimension != d3d::ViewDimension::Unknown) {
        return WordsStatus::UnsupportedView;  // e.g. acceleration structures
    }
    return buffer_words_at_address(va, v, out);
}

WordsStatus buffer_words_at_address(uint64_t va, const d3d::ViewDescriptor& v, uint32_t out[8]) {
    std::memset(out, 0, kViewSlotBytes);
    hw::BufferResource b;
    const bool raw = (v.flags & d3d::BufferViewFlags::Raw) != 0;
    if (v.structure_byte_stride != 0) {
        b.base = va + v.first_element_or_address * v.structure_byte_stride;
        b.stride = v.structure_byte_stride;
        b.num_records = v.num_elements_or_size;
        b.format = hw::Fmt::F32;
        b.oob_select = hw::Oob::IndexAndOffset;
    } else if (raw || v.format == 0) {
        b.base = va + v.first_element_or_address * 4u;
        b.num_records = v.num_elements_or_size * 4u;
        b.format = hw::Fmt::F32;
        b.oob_select = hw::Oob::Complete;
    } else {
        hw::HardwareFormat f;
        const uint32_t bytes = hw::hardware_format(v.format, &f) ? hw::buffer_format_bytes(f.format) : 0u;
        if (bytes == 0) {
            return WordsStatus::UnsupportedFormat;
        }
        b.base = va + v.first_element_or_address * bytes;
        b.stride = bytes;
        b.num_records = v.num_elements_or_size;
        b.format = f.format;
        b.dst_sel = v.kind == d3d::DescriptorKind::Srv
                        ? hw::compose_component_mapping(f.swizzle, normalize_component_mapping(v.component_mapping))
                        : f.swizzle;
        b.oob_select = hw::Oob::IndexAndOffset;
    }
    hw::encode(b, out);
    return WordsStatus::Ok;
}

const char* decode_status_name(DecodeStatus status) noexcept {
    switch (status) {
    case DecodeStatus::Ok: return "ok";
    case DecodeStatus::Zero: return "zero words (null descriptor)";
    case DecodeStatus::NotDescriptor: return "no descriptor words";
    case DecodeStatus::KindMismatch: return "descriptor of another kind";
    case DecodeStatus::UnresolvedAddress: return "buffer address in no known buffer";
    case DecodeStatus::UnknownTexture: return "image address of no known texture";
    case DecodeStatus::UnsupportedFormat: return "image format without a DXGI format";
    case DecodeStatus::Count: break;
    }
    return "?";
}

namespace {

d3d::DescriptorKind kind_of(d3d::DescriptorRangeType range) noexcept {
    switch (range) {
    case d3d::DescriptorRangeType::Cbv: return d3d::DescriptorKind::Cbv;
    case d3d::DescriptorRangeType::Uav: return d3d::DescriptorKind::Uav;
    default: return d3d::DescriptorKind::Srv;
    }
}

// Byte size of a buffer descriptor's window: records of `stride` bytes when the stride is set and
// out-of-bounds checks count records, else num_records bytes.
uint64_t buffer_bytes(const hw::BufferResource& b) noexcept {
    if (b.stride != 0 && b.oob_select != hw::Oob::Complete) {
        return uint64_t{b.num_records} * b.stride;
    }
    return b.num_records;
}

// Resolves the concrete DXGI view format WebGPU may use when viewing a texture of format `own`
// through a T# whose hardware format is `hw_fmt` (and default reverse lookup is `decoded_dxgi`).
// Returns 0 when the view should inherit the texture's own format.
uint32_t resolve_view_format(uint32_t own, uint32_t hw_fmt, uint32_t decoded_dxgi) noexcept {
    if (own == 0) {
        return 0;
    }
    // Depth-stencil textures: preserve the plane-specific view format so WebGPU selects the
    // depth-only or stencil-only aspect (aspect "all" is invalid for sampling depth-stencil).
    if (own == 44 || own == 45) {  // R24G8_TYPELESS / D24_UNORM_S8_UINT
        if (hw_fmt == 141 || hw_fmt == 143) {
            return 46;  // R24_UNORM_X8_TYPELESS (depth plane)
        }
        if (hw_fmt == 5 || hw_fmt == 142 || hw_fmt == 144) {
            return 47;  // X24_TYPELESS_G8_UINT (stencil plane)
        }
        return 0;
    }
    if (own == 19 || own == 20) {  // R32G8X24_TYPELESS / D32_FLOAT_S8X24_UINT
        if (hw_fmt == 22 || hw_fmt == 146) {
            return 21;  // R32_FLOAT_X8X24_TYPELESS (depth plane)
        }
        if (hw_fmt == 5 || hw_fmt == 145) {
            return 22;  // X32_TYPELESS_G8X24_UINT (stencil plane)
        }
        return 0;
    }
    if (own == 118) {  // D16_UNORM_S8_UINT (Xbox)
        if (hw_fmt == 7) {
            return 119;  // R16_UNORM_X8_TYPELESS (depth plane)
        }
        if (hw_fmt == 5) {
            return 120;  // X16_TYPELESS_G8_UINT (stencil plane)
        }
        return 0;
    }
    // sRGB-capable families (typeless, linear UNORM, sRGB): match by hardware format so typeless
    // textures and custom-swizzled sRGB views keep their linear vs sRGB view format.
    struct SrgbFamily {
        uint32_t typeless;
        uint32_t linear;
        uint32_t srgb;
        uint32_t linear_hw;
        uint32_t srgb_hw;
    };
    static constexpr SrgbFamily kSrgbFamilies[] = {
        {27, 28, 29, 56, 130},    // R8G8B8A8
        {90, 87, 91, 56, 130},    // B8G8R8A8
        {92, 88, 93, 56, 130},    // B8G8R8X8
        {70, 71, 72, 169, 170},   // BC1
        {73, 74, 75, 171, 172},   // BC2
        {76, 77, 78, 173, 174},   // BC3
        {97, 98, 99, 181, 182},   // BC7
    };
    for (const SrgbFamily& fam : kSrgbFamilies) {
        if (own == fam.typeless || own == fam.linear || own == fam.srgb) {
            if (hw_fmt == fam.srgb_hw) {
                return fam.srgb;
            }
            if (hw_fmt == fam.linear_hw) {
                return fam.linear;
            }
            if (own == 27) {  // R8G8B8A8_TYPELESS viewed as integer / snorm
                if (hw_fmt == 60) return 30;  // R8G8B8A8_UINT
                if (hw_fmt == 57) return 31;  // R8G8B8A8_SNORM
                if (hw_fmt == 61) return 32;  // R8G8B8A8_SINT
            }
            return 0;
        }
    }
    // Non-sRGB typeless texture families: resolve concrete view DXGI format from hw_fmt so custom
    // swizzles (such as R8_TYPELESS viewed as R8_UNORM with 000R, which shares hw_fmt 1 and
    // swizzle {0,0,0,X} with A8_UNORM) preserve both the concrete format and component mapping.
    switch (own) {
    case 1:  // R32G32B32A32_TYPELESS
        if (hw_fmt == 77) return 2;   // FLOAT
        if (hw_fmt == 75) return 3;   // UINT
        if (hw_fmt == 76) return 4;   // SINT
        break;
    case 5:  // R32G32B32_TYPELESS
        if (hw_fmt == 74) return 6;   // FLOAT
        if (hw_fmt == 72) return 7;   // UINT
        if (hw_fmt == 73) return 8;   // SINT
        break;
    case 9:  // R16G16B16A16_TYPELESS
        if (hw_fmt == 71) return 10;  // FLOAT
        if (hw_fmt == 65) return 11;  // UNORM
        if (hw_fmt == 69) return 12;  // UINT
        if (hw_fmt == 66) return 13;  // SNORM
        if (hw_fmt == 70) return 14;  // SINT
        break;
    case 15:  // R32G32_TYPELESS
        if (hw_fmt == 64) return 16;  // FLOAT
        if (hw_fmt == 62) return 17;  // UINT
        if (hw_fmt == 63) return 18;  // SINT
        break;
    case 23:  // R10G10B10A2_TYPELESS
        if (hw_fmt == 50) return 24;  // UNORM
        if (hw_fmt == 54) return 25;  // UINT
        break;
    case 33:  // R16G16_TYPELESS
        if (hw_fmt == 29) return 34;  // FLOAT
        if (hw_fmt == 23) return 35;  // UNORM
        if (hw_fmt == 27) return 36;  // UINT
        if (hw_fmt == 24) return 37;  // SNORM
        if (hw_fmt == 28) return 38;  // SINT
        break;
    case 39:  // R32_TYPELESS
        if (hw_fmt == 22) return 41;  // FLOAT
        if (hw_fmt == 20) return 42;  // UINT
        if (hw_fmt == 21) return 43;  // SINT
        break;
    case 48:  // R8G8_TYPELESS
        if (hw_fmt == 14) return 49;  // UNORM
        if (hw_fmt == 18) return 50;  // UINT
        if (hw_fmt == 15) return 51;  // SNORM
        if (hw_fmt == 19) return 52;  // SINT
        break;
    case 53:  // R16_TYPELESS
        if (hw_fmt == 13) return 54;  // FLOAT
        if (hw_fmt == 7)  return 56;  // UNORM
        if (hw_fmt == 11) return 57;  // UINT
        if (hw_fmt == 8)  return 58;  // SNORM
        if (hw_fmt == 12) return 59;  // SINT
        break;
    case 60:  // R8_TYPELESS (never A8_UNORM in D3D12)
        if (hw_fmt == 1) return 61;   // R8_UNORM
        if (hw_fmt == 5) return 62;   // R8_UINT
        if (hw_fmt == 2) return 63;   // R8_SNORM
        if (hw_fmt == 6) return 64;   // R8_SINT
        break;
    case 79:  // BC4_TYPELESS
        if (hw_fmt == 175) return 80; // UNORM
        if (hw_fmt == 176) return 81; // SNORM
        break;
    case 82:  // BC5_TYPELESS
        if (hw_fmt == 177) return 83; // UNORM
        if (hw_fmt == 178) return 84; // SNORM
        break;
    case 94:  // BC6H_TYPELESS
        if (hw_fmt == 179) return 95; // UF16
        if (hw_fmt == 180) return 96; // SF16
        break;
    default:
        break;
    }
    if (decoded_dxgi == own) {
        return decoded_dxgi;
    }
    // Same hardware format as the texture's own concrete format (e.g. R8_UNORM viewed with a custom
    // swizzle that collides with A8_UNORM's canonical swizzle, or D32_FLOAT viewed as R32_FLOAT).
    hw::HardwareFormat own_hw;
    if (hw::hardware_format(own, &own_hw) && own_hw.format == hw_fmt) {
        return own;
    }
    return 0u;
}

uint32_t view_dimension(const hw::ImageResource& img) noexcept {
    switch (img.type) {
    case hw::ImageType::Tex1D: return d3d::ViewDimension::Texture1D;
    case hw::ImageType::Tex1DArray: return d3d::ViewDimension::Texture1DArray;
    case hw::ImageType::Tex2D: return d3d::ViewDimension::Texture2D;
    case hw::ImageType::Tex2DArray: return d3d::ViewDimension::Texture2DArray;
    case hw::ImageType::Tex2DMsaa: return d3d::ViewDimension::Texture2DMs;
    case hw::ImageType::Tex2DMsaaArray: return d3d::ViewDimension::Texture2DMsArray;
    case hw::ImageType::Tex3D: return d3d::ViewDimension::Texture3D;
    case hw::ImageType::Cube:
        return (img.base_array != 0 || img.depth > img.base_array + 6) ? d3d::ViewDimension::TextureCubeArray
                                                                       : d3d::ViewDimension::TextureCube;
    default: return d3d::ViewDimension::Unknown;
    }
}

}  // namespace

DecodeStatus decode_view_words(const StateBridgeDevice& device, const uint32_t w[8], d3d::DescriptorRangeType range,
                               d3d::ViewDescriptor* out) {
    *out = d3d::ViewDescriptor{};
    out->kind = kind_of(range);
    const hw::SlotContent content = hw::classify_slot(w);
    if (content == hw::SlotContent::Zero) {
        return DecodeStatus::Zero;
    }
    if (content == hw::SlotContent::Unknown || range == d3d::DescriptorRangeType::Sampler) {
        return content == hw::SlotContent::Unknown ? DecodeStatus::NotDescriptor : DecodeStatus::KindMismatch;
    }
    if (content == hw::SlotContent::Buffer) {
        const hw::BufferResource b = hw::decode_buffer(w);
        const uint64_t bytes = buffer_bytes(b);
        if (range == d3d::DescriptorRangeType::Cbv) {
            out->first_element_or_address = b.base;
            out->num_elements_or_size = static_cast<uint32_t>(std::min<uint64_t>(bytes, 0xffffffffu));
            d3d::GpuVaResolution res;
            return device.resolve_gpu_va(b.base, &res) ? DecodeStatus::Ok : DecodeStatus::UnresolvedAddress;
        }
        d3d::GpuVaResolution res;
        if (!device.resolve_gpu_va(b.base, &res)) {
            return DecodeStatus::UnresolvedAddress;
        }
        out->dimension = static_cast<uint8_t>(d3d::ViewDimension::Buffer);
        out->resource_id = res.resource_id;
        out->structure_byte_stride = 1;
        out->first_element_or_address = res.offset;
        out->num_elements_or_size = static_cast<uint32_t>(
            std::min<uint64_t>(bytes, res.resource_size > res.offset ? res.resource_size - res.offset : 0));
        return DecodeStatus::Ok;
    }
    // Image words.
    if (range == d3d::DescriptorRangeType::Cbv) {
        return DecodeStatus::KindMismatch;
    }
    const hw::ImageResource img = hw::decode_image(w);
    const uint32_t id = device.texture_at(img.base);
    if (id == 0) {
        return DecodeStatus::UnknownTexture;
    }
    bool exact = false;
    const uint32_t dxgi = hw::dxgi_format(img.format, img.dst_sel, &exact);
    if (dxgi == 0) {
        return DecodeStatus::UnsupportedFormat;
    }
    out->resource_id = id;
    TextureInfo tex;
    const uint32_t own = device.texture_info(id, &tex) ? static_cast<uint32_t>(tex.format) : 0u;
    out->format = resolve_view_format(own, img.format, dxgi);
    out->dimension = static_cast<uint8_t>(view_dimension(img));
    if (out->kind == d3d::DescriptorKind::Srv) {
        hw::HardwareFormat f;
        uint32_t mapping = d3d::kDefaultShader4ComponentMapping;
        const uint32_t effective_dxgi = out->format != 0 ? out->format : (own != 0 ? own : dxgi);
        if (hw::hardware_format(effective_dxgi, &f) && f.format == img.format) {
            if (f.swizzle != img.dst_sel) {
                hw::component_mapping_from(f.swizzle, img.dst_sel, &mapping);
            }
        } else if (hw::hardware_format(dxgi, &f) && f.format == img.format && f.swizzle != img.dst_sel) {
            hw::component_mapping_from(f.swizzle, img.dst_sel, &mapping);
        }
        out->component_mapping = mapping;
        out->min_lod_clamp = hw::u4_8_to_lod(img.min_lod);
    }
    const bool msaa = img.type == hw::ImageType::Tex2DMsaa || img.type == hw::ImageType::Tex2DMsaaArray;
    out->first_mip = static_cast<uint16_t>(msaa ? 0u : img.base_level);
    out->mip_levels = static_cast<uint16_t>(msaa || out->kind == d3d::DescriptorKind::Uav
                                                ? 1u
                                                : img.last_level - img.base_level + 1u);
    out->first_array_slice = img.base_array;
    out->array_size = img.depth > img.base_array ? img.depth - img.base_array : 1u;
    if (out->dimension == d3d::ViewDimension::TextureCube || out->dimension == d3d::ViewDimension::TextureCubeArray) {
        out->array_size = std::max(1u, out->array_size / 6u);  // cube arrays: NumCubes
    }
    return DecodeStatus::Ok;
}

DecodeStatus decode_sampler_words(const uint32_t w[4], d3d::SamplerDesc* out) {
    if (w[0] == 0 && w[1] == 0 && w[2] == 0 && w[3] == 0) {
        return DecodeStatus::Zero;
    }
    if (!hw::plausible_sampler(w)) {
        return DecodeStatus::NotDescriptor;
    }
    *out = hw::d3d12_from_sampler(hw::decode_sampler(w));
    return DecodeStatus::Ok;
}

void sampler_words(const d3d::SamplerDesc& desc, uint32_t out[4]) {
    hw::encode(hw::sampler_from_d3d12(desc), out);
}

WordsStatus image_words_at_address(uint64_t address, const AddressTexture& texture, const d3d::ViewDescriptor& view,
                                   uint32_t out[8]) {
    std::memset(out, 0, kViewSlotBytes);
    TextureShape t;
    t.dimension = texture.dimension == 2 ? 1u : (texture.dimension == 4 ? 3u : 2u);
    t.width = static_cast<uint32_t>(std::min<uint64_t>(texture.width, 0xffffffffull));
    t.height = texture.height;
    t.depth_or_layers = texture.depth_or_array_size;
    t.mips = texture.mip_levels;
    t.samples = texture.sample_count;
    t.format = texture.format;
    return image_words(address, t, view, out);
}

}  // namespace playgta6::gpu::bridge
