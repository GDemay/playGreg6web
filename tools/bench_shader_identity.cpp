// Synthetic worker metadata scans, comparing the old lazy stage cache with the proposed eager
// stage cache. It predates #2167 and excludes its binding-access scan, Dawn,
// disk I/O, and browser/game execution; it is not a complete worker benchmark.
#include "gpu/native_exec/layout_model.h"
#include "gpu/native_exec/fragment_outputs_model.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace playgta6::gpu::native_exec;
int main(int argc, char** argv) {
    const size_t count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 12;
    const size_t bytes = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 3 * 1024 * 1024;
    const char* stages[] = {"vertex", "fragment", "compute"};
    const char* entries[] = {
        "@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(); }\n",
        "@fragment fn fs_main() -> @location(0) vec4f { return vec4f(); }\n",
        "@compute @workgroup_size(1) fn cs_main() {}\n"};
    uint64_t total_ns = 0, check = 0;
    size_t capacity_before = 0, capacity_after = 0;
    for (size_t i = 0; i < count; ++i) {
        const size_t stage = i % 3;
        std::string code = "@group(0) @binding(0) var<uniform> u: vec4f;\n";
        for (size_t j = 0; code.size() + 100 < bytes; ++j)
            code += "fn helper_" + std::to_string(j) + "(x: u32) -> u32 { return x * 3u + 7u; }\n";
        code += entries[stage];
        code.shrink_to_fit(); // WgslPack::get reads into a precisely sized string.
        const auto explicit_ep = extract_wgsl_entry_point(code, stages[stage]);
        capacity_before += code.capacity();
        const auto t = std::chrono::steady_clock::now();
#ifdef PROPOSED
        auto eps = resolve_all_shader_entry_points(code, explicit_ep);
        const auto& ep = eps[stage];
#else
        auto ep = resolve_shader_entry_point(code, explicit_ep, stages[stage]);
#endif
        auto group0 = parse_wgsl_group0_bindings(code);
        auto groups = parse_wgsl_group_bindings(code);
        auto declares = wgsl_declares_group0(code);
#ifdef PROPOSED
        auto mask = wgsl_fragment_output_mask(code, eps[1]);
#else
        auto mask = stage == 1 ? wgsl_fragment_output_mask(code, ep) : std::optional<uint32_t>{};
#endif
        std::string().swap(code);
        total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count();
        capacity_after += code.capacity();
        check += ep.size() + group0.size() + groups.size() + declares + (stage == 1 ? mask.value_or(0) : 0);
    }
    std::printf("count=%zu target_bytes=%zu scan_ns=%llu before_capacity=%zu after_capacity=%zu checksum=%llu\n",
                count, bytes, static_cast<unsigned long long>(total_ns), capacity_before, capacity_after,
                static_cast<unsigned long long>(check));
}
