// Synthetic actual DescriptorHeapRegistry calls; no GPU or game inputs.
// c++ -std=c++20 -O3 -DNDEBUG -Iruntime tools/bench_descriptor_ranges.cpp runtime/gpu/descriptor_heap.cpp -o bench-descriptor-ranges
#include "gpu/descriptor_heap.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>

namespace d = playgta6::gpu::d3d12_state;
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const uint32_t count = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 0));
    const uint32_t iterations = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 0));
    const uint32_t mode = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 0));
    if (!count || count > 16384 || !iterations || mode > 7 || ((mode == 5 || mode == 7) && count < 5)) return 2;
    d::DescriptorHeapRegistry registry;
    d::DescriptorHeapDesc desc;
    desc.type = d::DescriptorHeapType::CbvSrvUav;
    desc.num_descriptors = count * 6;
    d::DescriptorHeap *heap = nullptr, *other = nullptr, *third = nullptr;
    desc.flags = mode == 1 || mode == 4 ? d::DescriptorHeapFlags::ShaderVisible : d::DescriptorHeapFlags::None;
    if (registry.create_heap(desc, &heap) != d::Status::Ok) return 1;
    desc.flags = d::DescriptorHeapFlags::None;
    if (registry.create_heap(desc, &other) != d::Status::Ok ||
        registry.create_heap(desc, &third) != d::Status::Ok) return 1;
    const auto handle = [](d::DescriptorHeap* h, uint32_t index) {
        return d::CpuDescriptorHandle{h->cpu_start().ptr + uint64_t{index} * h->increment()};
    };
    std::vector<d::CpuDescriptorHandle> source[2], destination;
    d::ViewDesc view; view.format = 28; view.dimension = d::ViewDimension::Texture2D;
    for (uint32_t i = 0; i < count; ++i) {
        // Modes: interleaved disjoint / different heaps / disjoint bounds / true overlap /
        // multiple source heaps, all distinct from the destination heap.
        const uint32_t dst = mode == 2 ? count * 4 + i : (mode == 3 ? 6 * i : 6 * i + 1);
        destination.push_back(handle(heap, dst));
        for (uint32_t variant = 0; variant < 2; ++variant) {
            d::DescriptorHeap* src_heap = mode == 1 ? other : heap;
            if (mode == 4) src_heap = i % 2 ? other : third;
            const uint32_t index = 6 * i + variant * 2;
            source[variant].push_back(handle(src_heap, index));
            if (registry.create_shader_resource_view(2 * i + variant + 1, &view,
                    source[variant].back()) != d::Status::Ok) return 1;
        }
    }
    // Disjoint-bounds mode uses a compact source region so the bounding-box shortcut applies.
    if (mode == 2) {
        for (uint32_t i = 0; i < count; ++i) for (uint32_t v = 0; v < 2; ++v) {
            source[v][i] = handle(heap, i + v * count);
            if (registry.create_shader_resource_view(2 * i + v + 1, &view, source[v][i]) != d::Status::Ok) return 1;
        }
    }
    if (mode == 6 || mode == 7) {
        uint32_t rng = 0x814c09e3;
        for (uint32_t i = count - 1; i > 0; --i) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            const uint32_t j = rng % (i + 1);
            std::swap(source[0][i], source[0][j]);
            std::swap(source[1][i], source[1][j]);
        }
    }
    if (mode == 5 || mode == 7) destination[4] = source[0][4];
    const bool alias_control = mode == 3 || mode == 5 || mode == 7;
    const auto copy = [&](uint32_t v) {
        return registry.copy_descriptors(count, destination.data(), nullptr, count,
                                        source[v].data(), nullptr, desc.type) == d::Status::Ok;
    };
    if (!copy(0)) return 1;
    const auto begin = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) if (!copy(alias_control ? 0 : (i + 1) % 2)) return 1;
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    const uint32_t last = alias_control ? 0 : iterations % 2;
    uint64_t hash = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < count; ++i) {
        d::ResolvedDescriptor dst{}, src{};
        if (registry.resolve_cpu(destination[i], &dst) != d::Status::Ok ||
            registry.resolve_cpu(source[last][i], &src) != d::Status::Ok ||
            std::memcmp(dst.heap->view(dst.index), src.heap->view(src.index), sizeof(d::ViewDescriptor))) return 1;
        const auto* bytes = reinterpret_cast<const uint8_t*>(dst.heap->view(dst.index));
        for (size_t b = 0; b < sizeof(d::ViewDescriptor); ++b) hash = (hash ^ bytes[b]) * 0x100000001b3ull;
    }
    std::printf("{\"ranges\":%u,\"iterations\":%u,\"mode\":%u,\"destination_shader_visible\":%s,\"wall_ms\":%.6f,\"ms_per_call\":%.6f,\"digest\":\"%016llx\",\"correct\":true}\n",
                count, iterations, mode, mode == 1 || mode == 4 ? "true" : "false", elapsed,
                elapsed / iterations, static_cast<unsigned long long>(hash));
}
