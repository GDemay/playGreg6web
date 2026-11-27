// Synthetic native reserve-only workload; no physical page commit or game data.
// clang++ -std=c++20 -O3 -DNDEBUG -pthread -Iruntime tests/bench_vmem_reserve.cpp runtime/os/vmem.cpp runtime/os/vmem_backing.cpp -o bench_vmem
#include "os/vmem.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
using namespace playgta6::os;
int main(int argc, char** argv) {
    const size_t count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 16384;
    const bool hinted = argc > 2;
    VirtualMemoryArena arena(kDefaultArenaBase, (count + 1) * kAllocationGranularity);
    if (!arena.valid()) return 1;
    std::vector<uint64_t> addresses(count);
    const auto cpu_start = std::clock();
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < count; ++i) {
        addresses[i] = arena.reserve(hinted ? kDefaultArenaBase + i * kAllocationGranularity : 0, kPageSize);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const double cpu_ms = 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
    for (size_t i = 0; i < count; ++i)
        if (addresses[i] != kDefaultArenaBase + i * kAllocationGranularity) return 2;
    if (arena.reserved_bytes() != count * kPageSize) return 3;
    std::printf("{\"count\":%zu,\"hinted\":%s,\"milliseconds\":%.6f,\"cpu_ms\":%.6f,\"correct\":true}\n", count, hinted ? "true" : "false", ms, cpu_ms);
}
