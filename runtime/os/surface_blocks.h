#pragma once

#include <cstdint>

// Surface sizes in the GPU's 64 KiB-block layout (the console GPU generation's "64 KiB" swizzle
// modes), for the cases this file covers: a single-sample 2D surface, one mip level, one array
// slice, of a format with 1x1 elements (no block compression) of 1, 2, 4, 8 or 16 bytes.
//
// Rule: the surface is padded to whole 64 KiB blocks in both dimensions. A block holds
// 65,536 / bytes-per-element elements and is as square as possible, the width taking the extra
// factor of two:
//   1 byte -> 256 x 256, 2 -> 256 x 128, 4 -> 128 x 128, 8 -> 128 x 64, 16 -> 64 x 64.
// The surface is 64 KiB aligned and its size is a whole number of blocks.
//
// Not covered (callers keep their own sizing for these): surfaces smaller than one block in
// either dimension (the hardware then picks smaller block modes, whose choice is not modelled),
// MSAA, mip chains, arrays, 3D, depth/stencil planes and block-compressed formats.

namespace playgta6::os::surface {

inline constexpr uint64_t kBlockBytes = 64u * 1024u;

struct BlockDims {
    uint32_t width = 0;   // elements
    uint32_t height = 0;  // elements
};

// Block dimensions for an element size of 1, 2, 4, 8 or 16 bytes; false for any other size.
inline constexpr bool block64k_dims(uint32_t bytes_per_element, BlockDims* out) {
    BlockDims d{};
    switch (bytes_per_element) {
        case 1: d = {256, 256}; break;
        case 2: d = {256, 128}; break;
        case 4: d = {128, 128}; break;
        case 8: d = {128, 64}; break;
        case 16: d = {64, 64}; break;
        default: return false;
    }
    if (out != nullptr) {
        *out = d;
    }
    return true;
}

struct Block64kLayout {
    uint32_t padded_width = 0;   // elements, a multiple of the block width
    uint32_t padded_height = 0;  // elements, a multiple of the block height
    uint64_t row_pitch_bytes = 0;
    uint64_t size_bytes = 0;     // a multiple of kBlockBytes
    uint64_t alignment = kBlockBytes;
};

// The layout of a covered surface (see above); false when the surface is not covered.
inline constexpr bool block64k_layout_2d(uint32_t width, uint32_t height, uint32_t bytes_per_element,
                                         Block64kLayout* out) {
    BlockDims d{};
    if (!block64k_dims(bytes_per_element, &d) || width < d.width || height < d.height) {
        return false;
    }
    Block64kLayout l{};
    l.padded_width = (width + d.width - 1) / d.width * d.width;
    l.padded_height = (height + d.height - 1) / d.height * d.height;
    l.row_pitch_bytes = static_cast<uint64_t>(l.padded_width) * bytes_per_element;
    l.size_bytes = l.row_pitch_bytes * l.padded_height;
    if (out != nullptr) {
        *out = l;
    }
    return true;
}

inline constexpr uint64_t kBlock4kBytes = 4u * 1024u;

enum class BcFormat : uint32_t {
    BC1 = 1,
    BC2 = 2,
    BC3 = 3,
    BC4 = 4,
    BC5 = 5,
    BC6H = 6,
    BC7 = 7,
};

// Bytes per 4x4 block for BC1..BC7 (8 for BC1/BC4, 16 for BC2/BC3/BC5/BC6H/BC7), 0 otherwise.
inline constexpr uint32_t bc_block_bytes(uint32_t bc) {
    switch (bc) {
        case 1:
        case 4:
            return 8u;
        case 2:
        case 3:
        case 5:
        case 6:
        case 7:
            return 16u;
        default:
            return 0u;
    }
}

// Rounds pixel dimensions up to whole 4x4 BC blocks safely without 32-bit overflow.
// Returns false if either pixel dimension is zero.
inline constexpr bool bc_surface_grid(uint32_t pixel_width, uint32_t pixel_height, uint32_t* out_blocks_x,
                                      uint32_t* out_blocks_y) {
    if (pixel_width == 0 || pixel_height == 0) {
        return false;
    }
    const uint32_t bx = ((pixel_width - 1u) >> 2) + 1u;
    const uint32_t by = ((pixel_height - 1u) >> 2) + 1u;
    if (out_blocks_x != nullptr) {
        *out_blocks_x = bx;
    }
    if (out_blocks_y != nullptr) {
        *out_blocks_y = by;
    }
    return true;
}

// Exact linear byte size of packed BC1..BC7 4x4 blocks for a 2D surface.
// Returns false on invalid format, zero dimensions, or 64-bit byte-size overflow.
inline constexpr bool bc_linear_size_bytes(uint32_t pixel_width, uint32_t pixel_height, uint32_t bc,
                                           uint64_t* out_bytes) {
    const uint32_t bpb = bc_block_bytes(bc);
    uint32_t bx = 0;
    uint32_t by = 0;
    if (bpb == 0 || !bc_surface_grid(pixel_width, pixel_height, &bx, &by)) {
        return false;
    }
    const uint64_t blocks = static_cast<uint64_t>(bx) * static_cast<uint64_t>(by);
    if (blocks > (~uint64_t{0}) / bpb) {
        return false;
    }
    if (out_bytes != nullptr) {
        *out_bytes = blocks * bpb;
    }
    return true;
}

// 4 KiB tile dimensions in elements for 1, 2, 4, 8 or 16 bytes per element; false otherwise.
inline constexpr bool block4k_dims(uint32_t bytes_per_element, BlockDims* out) {
    BlockDims d{};
    switch (bytes_per_element) {
        case 1: d = {64, 64}; break;
        case 2: d = {64, 32}; break;
        case 4: d = {32, 32}; break;
        case 8: d = {32, 16}; break;
        case 16: d = {16, 16}; break;
        default: return false;
    }
    if (out != nullptr) {
        *out = d;
    }
    return true;
}

namespace detail {

inline constexpr bool padded_tile_layout_2d(uint32_t width, uint32_t height, uint32_t bytes_per_element,
                                            BlockDims tile, uint64_t alignment, Block64kLayout* out) {
    if (width == 0 || height == 0 || bytes_per_element == 0 || tile.width == 0 || tile.height == 0) {
        return false;
    }
    const uint64_t pw = ((static_cast<uint64_t>(width) + tile.width - 1u) / tile.width) * tile.width;
    const uint64_t ph = ((static_cast<uint64_t>(height) + tile.height - 1u) / tile.height) * tile.height;
    if (pw > 0xFFFFFFFFull || ph > 0xFFFFFFFFull) {
        return false;
    }
    const uint64_t row_pitch = pw * bytes_per_element;
    if (row_pitch > (~uint64_t{0}) / ph) {
        return false;
    }
    Block64kLayout l{};
    l.padded_width = static_cast<uint32_t>(pw);
    l.padded_height = static_cast<uint32_t>(ph);
    l.row_pitch_bytes = row_pitch;
    l.size_bytes = row_pitch * ph;
    l.alignment = alignment;
    if (out != nullptr) {
        *out = l;
    }
    return true;
}

}  // namespace detail

// 4 KiB-block layout for any non-zero 2D surface (padded up to whole 4 KiB tiles).
inline constexpr bool block4k_layout_2d(uint32_t width, uint32_t height, uint32_t bytes_per_element,
                                        Block64kLayout* out) {
    BlockDims d{};
    if (!block4k_dims(bytes_per_element, &d)) {
        return false;
    }
    return detail::padded_tile_layout_2d(width, height, bytes_per_element, d, kBlock4kBytes, out);
}

// 64 KiB tile dimensions in 4x4 BC blocks (8 B -> 128x64 blocks, 16 B -> 64x64 blocks).
inline constexpr bool block64k_bc_dims(uint32_t bc, BlockDims* out) {
    const uint32_t bpb = bc_block_bytes(bc);
    return bpb != 0 && block64k_dims(bpb, out);
}

// 64 KiB-block layout for a BC1..BC7 2D surface (padded_width/padded_height in 4x4 BC blocks).
inline constexpr bool block64k_bc_layout_2d(uint32_t pixel_width, uint32_t pixel_height, uint32_t bc,
                                            Block64kLayout* out) {
    uint32_t bx = 0;
    uint32_t by = 0;
    const uint32_t bpb = bc_block_bytes(bc);
    BlockDims d{};
    if (bpb == 0 || !bc_surface_grid(pixel_width, pixel_height, &bx, &by) || !block64k_dims(bpb, &d)) {
        return false;
    }
    return detail::padded_tile_layout_2d(bx, by, bpb, d, kBlockBytes, out);
}

// 4 KiB tile dimensions in 4x4 BC blocks (8 B -> 32x16 blocks, 16 B -> 16x16 blocks).
inline constexpr bool block4k_bc_dims(uint32_t bc, BlockDims* out) {
    const uint32_t bpb = bc_block_bytes(bc);
    return bpb != 0 && block4k_dims(bpb, out);
}

// 4 KiB-block layout for a BC1..BC7 2D surface (padded_width/padded_height in 4x4 BC blocks).
inline constexpr bool block4k_bc_layout_2d(uint32_t pixel_width, uint32_t pixel_height, uint32_t bc,
                                           Block64kLayout* out) {
    uint32_t bx = 0;
    uint32_t by = 0;
    const uint32_t bpb = bc_block_bytes(bc);
    if (bpb == 0 || !bc_surface_grid(pixel_width, pixel_height, &bx, &by)) {
        return false;
    }
    return block4k_layout_2d(bx, by, bpb, out);
}

// Mip dimension max(1u, base >> mip), safe for mip >= 32.
inline constexpr uint32_t mip_extent(uint32_t base, uint32_t mip) {
    if (mip >= 32u) {
        return 1u;
    }
    const uint32_t shifted = base >> mip;
    return shifted > 1u ? shifted : 1u;
}

}  // namespace playgta6::os::surface
