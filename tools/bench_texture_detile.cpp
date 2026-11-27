// Synthetic native production detiler benchmark (no game assets or GPU required).
// c++ -std=c++20 -O3 -DNDEBUG -Iruntime tools/bench_texture_detile.cpp runtime/gpu/gfx10_detile.cpp -o work/bench-detile
// bench-detile WIDTH HEIGHT DXGI_FORMAT SWIZZLE_MODE ITERATIONS [DEPTH]
// Modes are GFX10 enum values, e.g. 9=64KB_S, 10=64KB_D, 25=64KB_S_X.
#include "gpu/gfx10_detile.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static double process_cpu_ms() {
#if defined(_WIN32)
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) return -1;
    const uint64_t k = (uint64_t{kernel.dwHighDateTime} << 32) | kernel.dwLowDateTime;
    const uint64_t u = (uint64_t{user.dwHighDateTime} << 32) | user.dwLowDateTime;
    return static_cast<double>(k + u) / 10000.0;
#else
    return 1000.0 * std::clock() / CLOCKS_PER_SEC;
#endif
}

namespace g = playgta6::gpu::gfx10;
int main(int argc, char** argv) {
    if (argc < 6 || argc > 7) return 2;
    g::TextureLayoutDesc desc{};
    desc.width = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 0));
    desc.height = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 0));
    desc.dxgi_format = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 0));
    desc.d3d12_layout = 0x100u | static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 0));
    const size_t iterations = std::strtoull(argv[5], nullptr, 0);
    desc.depth = argc == 7 ? static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 0)) : 1;
    desc.mip_levels = 1;
    desc.array_size = 1;
    desc.pipe_bank_xor = 0xA5;
    const auto geom = g::format_geometry(desc.dxgi_format);
    const size_t size = g::surface_bytes(desc);
    if (!size || !iterations || !geom.supported) return 2;
    std::vector<uint8_t> src(size);
    uint32_t rng = 0x531fd021;
    for (auto& b : src) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; b = static_cast<uint8_t>(rng); }
    const uint32_t bx = (desc.width + geom.block_width - 1) / geom.block_width;
    const uint32_t by = (desc.height + geom.block_height - 1) / geom.block_height;
    std::vector<uint8_t> expected(size_t{bx} * by * desc.depth * geom.bytes_per_block);
    for (uint32_t z = 0; z < desc.depth; ++z)
        for (uint32_t y = 0; y < by; ++y)
            for (uint32_t x = 0; x < bx; ++x) {
                const auto off = g::surface_element_offset(desc, 0, x, y, z);
                if (off > src.size() || geom.bytes_per_block > src.size() - off) return 1;
                std::memcpy(expected.data() + ((size_t{z} * by + y) * bx + x) * geom.bytes_per_block,
                            src.data() + off, geom.bytes_per_block);
            }
    std::vector<uint8_t> out;
    if (!g::detile_surface_subresource(desc, 0, 0, src.data(), src.size(), out).ok || out != expected) return 1;
    const double cpu = process_cpu_ms();
    const auto begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i)
        if (!g::detile_surface_subresource(desc, 0, 0, src.data(), src.size(), out).ok) return 1;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    const double cpu_ms = process_cpu_ms() - cpu;
    if (out != expected) return 1;
    uint64_t digest = 0xcbf29ce484222325ull;
    for (uint8_t b : out) { digest ^= b; digest *= 0x100000001b3ull; }
    std::printf("{\"width\":%u,\"height\":%u,\"depth\":%u,\"dxgi_format\":%u,\"layout\":%u,\"iterations\":%zu,\"output_bytes\":%zu,\"elapsed_ms\":%.6f,\"ms_per_update\":%.6f,\"cpu_ms\":%.6f,\"digest\":\"%016llx\",\"correct\":true}\n",
                desc.width, desc.height, desc.depth, desc.dxgi_format, desc.d3d12_layout, iterations, out.size(), ms,
                ms / iterations, cpu_ms, static_cast<unsigned long long>(digest));
}
