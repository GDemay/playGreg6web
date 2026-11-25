// Contention stress test for XMem write tracking (O33): several threads each own a few 64 KiB
// physical pages mapped in a shared (window) reservation and loop over
//   write (faults on an armed view) -> take_written_pages on their own pages (re-arms them)
//   -> now and then unmap + remap one page (view tracking),
// while one "render" thread keeps taking the written pages of the whole reservation (as the GPU
// upload path does). Every write must be reported to its owner's next take, so the test doubles
// as a correctness check of the tracker under concurrency.
//
//   bench_xmem_contention [--full] [--threads N]
//
// Without --full (CTest) it runs a short smoke pass; --full prints iterations per second.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xbox_scarlett.h"

namespace {

using namespace playgta6::os;
using namespace playgta6::os::scarlett;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kWindowType = 0x20402000u;  // MEM_RESERVE | MEM_64K_PAGES | MEM_PHYSICAL
constexpr uint32_t kCpuProtect = 0x4u;         // PAGE_READWRITE
constexpr size_t kPagesPerThread = 4;

std::atomic<bool> g_stop{false};
std::atomic<int> g_failures{0};

struct Result {
    uint64_t iterations = 0;
};

void worker(uint64_t base_va, const uint64_t* pfns, int seed, Result* out) {
    auto* arena = win32::get_vmem_arena();
    uint8_t* p = arena->translate(base_va);
    const uint64_t len = kPagesPerThread * kXMemPageSize64K;
    WriteCursor cursor;
    std::vector<uint64_t> pages;
    take_written_pages(p, len, &cursor, &pages);  // (initial: everything new)
    uint64_t it = 0;
    while (!g_stop.load(std::memory_order_relaxed)) {
        const size_t i = static_cast<size_t>(it % kPagesPerThread);
        if (it % 64 == 63) {
            // The map path: unmap and remap the page (contents kept).
            const uint64_t va = base_va + i * kXMemPageSize64K;
            if (pg6_xbox_XMemMapPhysicalPagesScatter(&va, 1, nullptr) != 1 ||
                pg6_xbox_XMemMapPhysicalPagesScatter(&va, 1, &pfns[i]) != 1) {
                std::fprintf(stderr, "FAIL: remap (thread %d)\n", seed);
                g_failures.fetch_add(1);
                return;
            }
            take_written_pages(p, len, &cursor, &pages);  // (a new mapping counts as written)
        }
        const uint64_t off = i * kXMemPageSize64K + (it * 4096 + 17) % kXMemPageSize64K;
        p[off] = static_cast<uint8_t>(seed + it);
        pages.clear();
        if (!take_written_pages(p, len, &cursor, &pages) ||
            std::find(pages.begin(), pages.end(), off & ~uint64_t{kWrittenPageSize - 1}) == pages.end()) {
            std::fprintf(stderr, "FAIL: write at +0x%llx not reported (thread %d)\n",
                         static_cast<unsigned long long>(off), seed);
            g_failures.fetch_add(1);
            return;
        }
        if (p[off] != static_cast<uint8_t>(seed + it)) {
            std::fprintf(stderr, "FAIL: contents (thread %d)\n", seed);
            g_failures.fetch_add(1);
            return;
        }
        ++it;
    }
    out->iterations = it;
}

}  // namespace

int main(int argc, char** argv) {
    bool full = false;
    int threads = 4;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--full") == 0) {
            full = true;
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::max(1, std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr, "usage: bench_xmem_contention [--full] [--threads N]\n");
            return 64;
        }
    }
    xmem_reset_for_testing();
    auto* arena = win32::get_vmem_arena();
    if (!arena->has_shared_zone()) {
        std::printf("bench_xmem_contention: no shared zone on this host, skipped\n");
        return 0;
    }
    const size_t total = kPagesPerThread * static_cast<size_t>(threads);
    void* res = pg6_xbox_XMemVirtualAlloc(nullptr, uint64_t{total} * kXMemPageSize64K, kWindowType, 0x23u, kCpuProtect);
    if (res == nullptr || !arena->is_shared(arena->guest_address(res))) {
        std::fprintf(stderr, "FAIL: shared reservation\n");
        return 1;
    }
    const uint64_t base = arena->guest_address(res);
    std::vector<uint64_t> pfns(total);
    size_t n = total;
    if (pg6_xbox_XMemAllocatePhysicalPages(0x380u, &n, pfns.data()) != 1 || n != total) {
        std::fprintf(stderr, "FAIL: physical pages\n");
        return 1;
    }
    std::vector<uint64_t> vas(total);
    for (size_t i = 0; i < total; ++i) {
        vas[i] = base + i * kXMemPageSize64K;
    }
    if (pg6_xbox_XMemMapPhysicalPagesScatter(vas.data(), total, pfns.data()) != 1) {
        std::fprintf(stderr, "FAIL: map\n");
        return 1;
    }

    const uint64_t faults0 = shared_view_write_faults();
    std::vector<Result> results(static_cast<size_t>(threads));
    std::atomic<uint64_t> render_takes{0};
    std::vector<std::thread> pool;
    for (int w = 0; w < threads; ++w) {
        pool.emplace_back(worker, base + uint64_t{kPagesPerThread} * w * kXMemPageSize64K,
                          &pfns[kPagesPerThread * static_cast<size_t>(w)], w * 31 + 7, &results[static_cast<size_t>(w)]);
    }
    std::thread render([&] {
        WriteCursor cursor;
        std::vector<uint64_t> pages;
        const uint8_t* p = arena->translate(base);
        while (!g_stop.load(std::memory_order_relaxed)) {
            pages.clear();
            take_written_pages(p, uint64_t{total} * kXMemPageSize64K, &cursor, &pages);
            render_takes.fetch_add(1, std::memory_order_relaxed);
        }
    });
    const auto t0 = Clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(full ? 3000 : 300));
    g_stop.store(true);
    for (auto& t : pool) {
        t.join();
    }
    render.join();
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    uint64_t iters = 0;
    for (const Result& r : results) {
        iters += r.iterations;
    }
    std::printf("%d thread(s) + render taker, %.2f s: %.0f write/take iterations/s (%.0f per thread), "
                "%.0f render takes/s, %llu faults\n",
                threads, secs, iters / secs, iters / secs / threads, render_takes.load() / secs,
                static_cast<unsigned long long>(shared_view_write_faults() - faults0));
    pg6_xbox_XMemMapPhysicalPagesScatter(vas.data(), total, nullptr);
    pg6_xbox_XMemFreePhysicalPages(total, pfns.data());
    xmem_virtual_free(res);
    if (g_failures.load() != 0 || iters == 0) {
        std::fprintf(stderr, "bench_xmem_contention: FAILED\n");
        return 1;
    }
    return 0;
}
