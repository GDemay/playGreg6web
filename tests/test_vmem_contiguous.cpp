// VirtualMemoryArena on one contiguous guest address space (runtime/os/vmem.h): translation,
// contiguity across page runs and reservations, zero-fill on (re)commit, protections, the image
// base hint, exhaustion, many reservations and concurrent use.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_vmem_contiguous.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os;

constexpr uint64_t kImageBase = 0x140000000ULL;

bool all_zero(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (p[i] != 0) {
            return false;
        }
    }
    return true;
}

void test_translation_and_contiguity() {
    VirtualMemoryArena arena;
    CHECK(arena.valid());
    CHECK(arena.space_base() == playgta6::kGuestAddressSpaceBase);
    CHECK(arena.space_size() == kDefaultGuestSpaceSize);

    // Two back-to-back reservations, committed in separate runs with a different protection.
    const uint64_t a = arena.reserve(0, 64 * 1024, PageProtect::ReadWrite);
    const uint64_t b = arena.reserve(0, 64 * 1024, PageProtect::ReadWrite);
    CHECK(a == arena.space_base());
    CHECK(b == a + 64 * 1024);
    CHECK(arena.commit(a, 2 * kPageSize, PageProtect::ReadWrite));
    CHECK(arena.commit(a + 2 * kPageSize, 14 * kPageSize, PageProtect::ExecuteReadWrite));
    CHECK(arena.commit(b, 64 * 1024, PageProtect::ReadWrite));

    uint8_t* p = arena.host_ptr(a);
    CHECK(p != nullptr);
    CHECK(p == arena.translate(a));
    CHECK(reinterpret_cast<uintptr_t>(p) == static_cast<uintptr_t>(arena.guest_base() + a));
    CHECK(arena.guest_address(p) == a);
    for (uint64_t off : {uint64_t{1}, kPageSize - 1, kPageSize, 2 * kPageSize + 17,
                         uint64_t{64 * 1024}, uint64_t{128 * 1024 - 8}}) {
        CHECK(arena.host_ptr(a + off) == p + off);
        CHECK(arena.guest_address(p + off) == a + off);
    }
    // One host pointer spans both commits and both reservations.
    uint8_t* span = arena.host_ptr(a + kPageSize, 128 * 1024 - kPageSize);
    CHECK(span == p + kPageSize);
    if (span != nullptr) {
        for (size_t i = 0; i < 128 * 1024 - kPageSize; ++i) {
            span[i] = static_cast<uint8_t>(i * 7);
        }
        std::vector<uint8_t> back(128 * 1024 - kPageSize);
        CHECK(arena.read(a + kPageSize, back.data(), back.size()));
        bool same = true;
        for (size_t i = 0; i < back.size(); ++i) {
            same &= back[i] == static_cast<uint8_t>(i * 7);
        }
        CHECK(same);
    }
    // Writes through read()/write() and through the raw translation see the same bytes.
    const uint64_t value = 0x1122334455667788ULL;
    CHECK(arena.write(b + 64 * 1024 - 8, &value, 8));
    uint64_t raw = 0;
    std::memcpy(&raw, arena.translate(b + 64 * 1024 - 8), 8);
    CHECK(raw == value);

    // A gap (uncommitted page) or a free range breaks the span.
    CHECK(arena.decommit(a + kPageSize, kPageSize));
    CHECK(arena.host_ptr(a, 3 * kPageSize) == nullptr);
    CHECK(arena.host_ptr(b + 60 * 1024, 8 * 1024) == nullptr);  // runs past the reservation
    CHECK(arena.release(a));
    CHECK(arena.release(b));
}

void test_zero_fill() {
    VirtualMemoryArena arena;
    const uint64_t a = arena.reserve(0, 256 * 1024);
    CHECK(a != 0);
    CHECK(arena.commit(a, 256 * 1024, PageProtect::ReadWrite));
    std::vector<uint8_t> buf(256 * 1024, 0xAB);
    CHECK(arena.read(a, buf.data(), buf.size()));
    CHECK(all_zero(buf.data(), buf.size()));

    std::vector<uint8_t> junk(256 * 1024, 0xCD);
    CHECK(arena.write(a, junk.data(), junk.size()));
    // Decommit / recommit a middle run: zero again, neighbours untouched.
    CHECK(arena.decommit(a + 16 * kPageSize, 8 * kPageSize));
    CHECK(arena.committed_bytes() == 256 * 1024 - 8 * kPageSize);
    // O36: commit reports how much of the range was fresh (zero-filled) versus kept.
    uint64_t newly = 123;
    CHECK(arena.commit(a + 16 * kPageSize, 8 * kPageSize, PageProtect::ReadWrite, &newly));
    CHECK(newly == 8 * kPageSize);
    CHECK(arena.read(a, buf.data(), buf.size()));
    CHECK(all_zero(buf.data() + 16 * kPageSize, 8 * kPageSize));
    CHECK(buf[16 * kPageSize - 1] == 0xCD && buf[24 * kPageSize] == 0xCD);
    CHECK(arena.commit(a + 12 * kPageSize, 8 * kPageSize, PageProtect::ReadWrite, &newly));
    CHECK(newly == 0);  // all of it was committed already: contents kept
    CHECK(arena.decommit(a + 20 * kPageSize, 2 * kPageSize));
    CHECK(arena.commit(a + 16 * kPageSize, 8 * kPageSize, PageProtect::ReadWrite, &newly));
    CHECK(newly == 2 * kPageSize);
    CHECK(!arena.commit(a + 16 * kPageSize, 512 * 1024, PageProtect::ReadWrite, &newly));
    CHECK(newly == 0);

    // Release + reserve the same VA again: fresh, zeroed pages.
    CHECK(arena.release(a));
    CHECK(arena.reserve(a, 256 * 1024) == a);
    CHECK(arena.commit(a, 256 * 1024, PageProtect::ReadWrite));
    CHECK(arena.read(a, buf.data(), buf.size()));
    CHECK(all_zero(buf.data(), buf.size()));
    CHECK(arena.release(a));

    // A second arena placed where the first one was gets zeroed pages too.
    {
        VirtualMemoryArena first;
        const uint64_t va = first.reserve(0, 64 * 1024);
        CHECK(first.commit(va, 64 * 1024, PageProtect::ReadWrite));
        CHECK(first.write(va, junk.data(), 64 * 1024));
    }
    VirtualMemoryArena second;
    const uint64_t va = second.reserve(0, 64 * 1024);
    CHECK(second.commit(va, 64 * 1024, PageProtect::ReadWrite));
    CHECK(second.read(va, buf.data(), 64 * 1024));
    CHECK(all_zero(buf.data(), 64 * 1024));
}

void test_protection_bookkeeping() {
    VirtualMemoryArena arena;
    const uint64_t a = arena.reserve(0, 64 * 1024, PageProtect::ReadWrite);
    CHECK(arena.commit(a, 4 * kPageSize, PageProtect::ReadWrite));
    const uint64_t v = 42;
    CHECK(arena.write(a + kPageSize, &v, 8));

    PageProtect old = PageProtect::NoAccess;
    CHECK(arena.protect(a + kPageSize, kPageSize, PageProtect::ReadOnly, &old));
    CHECK(old == PageProtect::ReadWrite);
    MemoryBasicInfo info = arena.query(a + kPageSize);
    CHECK(info.base_address == a + kPageSize && info.region_size == kPageSize);
    CHECK(info.state == MemState::Commit && info.protect == PageProtect::ReadOnly);
    CHECK(info.allocation_base == a && info.allocation_protect == PageProtect::ReadWrite);
    uint64_t r = 0;
    CHECK(arena.read(a + kPageSize, &r, 8) && r == 42);
    CHECK(!arena.write(a + kPageSize, &v, 8));
    CHECK(arena.host_ptr(a + kPageSize) == nullptr);
    CHECK(arena.host_ptr_const(a + kPageSize) == arena.translate(a + kPageSize));

    CHECK(arena.protect(a + kPageSize, kPageSize, PageProtect::NoAccess, &old));
    CHECK(old == PageProtect::ReadOnly);
    CHECK(!arena.read(a + kPageSize, &r, 8));
    CHECK(arena.protect(a + kPageSize, kPageSize, PageProtect::ReadWrite, &old));
    CHECK(arena.read(a + kPageSize, &r, 8) && r == 42);  // data survives protection changes

    // Free space reports up to the next reservation / the end of the space.
    const MemoryBasicInfo free_info = arena.query(a + 64 * 1024);
    CHECK(free_info.state == MemState::Free);
    CHECK(free_info.region_size == arena.space_base() + arena.space_size() - (a + 64 * 1024));
    CHECK(arena.release(a));

#if defined(__EMSCRIPTEN__)
    CHECK(!VirtualMemoryArena::protections_enforced());
#else
    CHECK(VirtualMemoryArena::protections_enforced());
#endif
}

// Native hosts: protections and the inaccessible low range really fault. Probed in a child
// process (POSIX) or under SEH (MSVC); nothing to probe on wasm.
#if defined(_MSC_VER)
bool faults_on_write(volatile uint8_t* p) {
    __try {
        *p = 1;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER
                                                                  : EXCEPTION_CONTINUE_SEARCH) {
        return true;
    }
    return false;
}
bool faults_on_read(volatile uint8_t* p) {
    __try {
        (void)*p;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER
                                                                  : EXCEPTION_CONTINUE_SEARCH) {
        return true;
    }
    return false;
}
#elif !defined(__EMSCRIPTEN__) && !defined(_WIN32)
bool child_faults(void (*probe)(volatile uint8_t*), volatile uint8_t* p) {
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        probe(p);
        _exit(0);
    }
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) {
        return false;
    }
    return !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
void write_probe(volatile uint8_t* p) {
    *p = 1;
}
void read_probe(volatile uint8_t* p) {
    (void)*p;
}
bool faults_on_write(volatile uint8_t* p) {
    return child_faults(write_probe, p);
}
bool faults_on_read(volatile uint8_t* p) {
    return child_faults(read_probe, p);
}
#endif

void test_protection_enforced() {
#if defined(_MSC_VER) || (!defined(__EMSCRIPTEN__) && !defined(_WIN32))
    VirtualMemoryArena arena;
    const uint64_t a = arena.reserve(0, 64 * 1024);
    CHECK(arena.commit(a, 3 * kPageSize, PageProtect::ReadWrite));
    CHECK(arena.protect(a + kPageSize, kPageSize, PageProtect::ReadOnly));
    CHECK(arena.protect(a + 2 * kPageSize, kPageSize, PageProtect::NoAccess));
    volatile uint8_t* p = arena.translate(a);
    CHECK(!faults_on_write(p));
    CHECK(!faults_on_read(p + kPageSize));
    CHECK(faults_on_write(p + kPageSize));
    CHECK(faults_on_read(p + 2 * kPageSize));
    CHECK(faults_on_read(p + 3 * kPageSize));  // reserved, never committed
    CHECK(arena.decommit(a, kPageSize));
    CHECK(faults_on_read(p));  // decommitted
    if (!arena.is_identity()) {
        CHECK(faults_on_read(arena.translate(0)));       // null guest pointer
        CHECK(faults_on_read(arena.translate(0x1000)));  // small guest pointer
    }
    CHECK(arena.release(a));
#endif
}

void test_image_base_hint() {
    VirtualMemoryArena arena;
    // Allocations without a hint do not prevent the preferred image base from being used later.
    const uint64_t heap = arena.virtual_alloc(0, 1 << 20, kMemReserve | kMemCommit,
                                              PageProtect::ReadWrite);
    CHECK(heap == arena.space_base());
    const uint64_t image = arena.reserve(kImageBase, 0x0B06D000ULL, PageProtect::NoAccess);
    CHECK(image == kImageBase);
    CHECK(arena.commit(kImageBase + 0x1000, 0x10000, PageProtect::ReadWrite));
    CHECK(arena.host_ptr(kImageBase + 0x1000) == arena.translate(kImageBase + 0x1000));
    CHECK(arena.reserve(kImageBase + 0x20000, 0x1000) == 0);  // inside the image
    CHECK(arena.release(image));
    CHECK(arena.release(heap));
}

// Placeholders: reserve, split (MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), replace
// (MEM_REPLACE_PLACEHOLDER), back to a placeholder, coalesce, release.
void test_placeholders() {
    VirtualMemoryArena arena;
    constexpr uint64_t kG = 0x10000;
    // Only reserve + NoAccess + 64 KiB granularity.
    CHECK(arena.virtual_alloc(0, 8 * kG, kMemReserve | kMemCommit | kMemReservePlaceholder, PageProtect::NoAccess) == 0);
    CHECK(arena.virtual_alloc(0, 8 * kG, kMemReserve | kMemReservePlaceholder, PageProtect::ReadWrite) == 0);
    CHECK(arena.virtual_alloc(0, 8 * kG + 0x1000, kMemReserve | kMemReservePlaceholder, PageProtect::NoAccess) == 0);
    const uint64_t base = arena.virtual_alloc(0, 8 * kG, kMemReserve | kMemReservePlaceholder, PageProtect::NoAccess);
    CHECK(base != 0);
    const uint64_t reserved = arena.reserved_bytes();
    CHECK(!arena.commit(base, kG, PageProtect::ReadWrite));  // never committable
    CHECK(!arena.virtual_free(base, kG, kMemDecommit));

    // Split [base, +2G) | [+2G, +5G) | [+5G, +8G) out of the placeholder.
    CHECK(arena.virtual_free(base, 2 * kG, kMemRelease | kMemPreservePlaceholder));
    CHECK(arena.virtual_free(base + 2 * kG, 3 * kG, kMemRelease | kMemPreservePlaceholder));
    CHECK(arena.reservation_count() == 3);
    CHECK(arena.reserved_bytes() == reserved);
    CHECK(!arena.virtual_free(base + kG / 2, kG, kMemRelease | kMemPreservePlaceholder));  // not 64 KiB aligned
    CHECK(arena.query(base + 3 * kG).allocation_base == base + 2 * kG);

    // Replace the middle one exactly, committed: usable memory.
    CHECK(arena.virtual_alloc(base + 2 * kG, 2 * kG, kMemReserve | kMemReplacePlaceholder, PageProtect::ReadWrite) ==
          0);  // wrong size
    CHECK(arena.virtual_alloc(base + 2 * kG, 3 * kG, kMemReserve | kMemCommit | kMemReplacePlaceholder,
                              PageProtect::ReadWrite) == base + 2 * kG);
    const uint32_t value = 0xC0FFEE;
    CHECK(arena.write(base + 2 * kG + 8, &value, sizeof(value)));
    CHECK(arena.virtual_alloc(base + 2 * kG, 3 * kG, kMemReserve | kMemReplacePlaceholder, PageProtect::ReadWrite) ==
          0);  // no longer a placeholder
    CHECK(!arena.commit(base, kG, PageProtect::ReadWrite));  // the others still are
    // Coalescing needs placeholders only.
    CHECK(!arena.virtual_free(base, 8 * kG, kMemRelease | kMemCoalescePlaceholders));

    // Back to a placeholder (whole range only), then coalesce all three and release.
    CHECK(!arena.virtual_free(base + 2 * kG, kG, kMemRelease | kMemPreservePlaceholder));
    CHECK(arena.virtual_free(base + 2 * kG, 3 * kG, kMemRelease | kMemPreservePlaceholder));
    CHECK(arena.committed_bytes() == 0);
    CHECK(!arena.write(base + 2 * kG + 8, &value, sizeof(value)));
    CHECK(arena.virtual_free(base, 8 * kG, kMemRelease | kMemCoalescePlaceholders));
    CHECK(arena.reservation_count() == 1);
    CHECK(arena.virtual_free(base, 0, kMemRelease));
    CHECK(arena.reservation_count() == 0 && arena.reserved_bytes() == 0);
}

void test_exhaustion() {
    // A 1 MiB space: 16 granules, then clean failure.
    VirtualMemoryArena small(kDefaultArenaBase, 1 << 20);
    CHECK(small.valid());
    std::vector<uint64_t> got;
    for (int i = 0; i < 32; ++i) {
        const uint64_t va = small.reserve(0, 4096);
        if (va == 0) {
            break;
        }
        got.push_back(va);
    }
    CHECK(got.size() == 16);
    CHECK(small.reserve(0, 4096) == 0);
    CHECK(small.reserve(kDefaultArenaBase + (1 << 20), 4096) == 0);   // just past the end
    CHECK(small.reserve(kDefaultArenaBase - 0x10000, 4096) == 0);     // below the space
    CHECK(small.reserve(kDefaultArenaBase + 0xF0000, 0x20000) == 0);  // straddles the end
    CHECK(small.virtual_alloc(0, 2 << 20, kMemReserve, PageProtect::ReadWrite) == 0);
    CHECK(small.release(got[3]));
    CHECK(small.reserve(0, 4096) == got[3]);  // the hole is reused
    CHECK(small.query(kDefaultArenaBase + (1 << 20)).region_size == 0);

    // A space nobody can back: invalid, every reservation fails, nothing crashes.
    VirtualMemoryArena impossible(kDefaultArenaBase, uint64_t{1} << 62);
    CHECK(!impossible.valid());
    CHECK(impossible.reserve(0, 4096) == 0);
    CHECK(impossible.reserve(kImageBase, 4096) == 0);
    CHECK(impossible.virtual_alloc(0, 4096, kMemReserve | kMemCommit, PageProtect::ReadWrite) == 0);
    CHECK(impossible.query(kImageBase).state == MemState::Free);
    CHECK(impossible.host_ptr(kImageBase) == nullptr);
}

void test_many_reservations() {
    VirtualMemoryArena arena;
    constexpr int kCycles = 100000;
    const auto start = std::chrono::steady_clock::now();
    bool ok = true;
    for (int i = 0; i < kCycles; ++i) {
        const uint64_t va = arena.reserve(0, 8 * 1024, PageProtect::ReadWrite);
        ok &= va != 0;
        ok &= arena.commit(va, kPageSize, PageProtect::ReadWrite);
        const uint32_t v = static_cast<uint32_t>(i);
        ok &= arena.write(va + 16, &v, 4);
        ok &= arena.release(va);
        if (!ok) {
            break;
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    CHECK(ok);
    CHECK(arena.reservation_count() == 0 && arena.committed_bytes() == 0);
    std::printf("  %d reserve/commit/write/release cycles: %.1f ms\n", kCycles, ms);

    // Many live reservations, then release in a different order.
    std::vector<uint64_t> live;
    for (int i = 0; i < 2000; ++i) {
        live.push_back(arena.reserve(0, 4096));
    }
    CHECK(live.back() == arena.space_base() + 1999 * kAllocationGranularity);
    for (size_t i = 0; i < live.size(); i += 2) {
        CHECK(arena.release(live[i]));
    }
    for (size_t i = 1; i < live.size(); i += 2) {
        CHECK(arena.release(live[i]));
    }
    CHECK(arena.reservation_count() == 0);
}

void test_concurrent_reservations() {
    VirtualMemoryArena arena;
    constexpr int kThreads = 8;
    constexpr int kIterations = 500;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&arena, &failures, t] {
            for (int i = 0; i < kIterations; ++i) {
                const uint64_t va = arena.virtual_alloc(0, 3 * kPageSize, kMemReserve | kMemCommit,
                                                        PageProtect::ReadWrite);
                if (va == 0) {
                    failures.fetch_add(1);
                    continue;
                }
                uint64_t* p = reinterpret_cast<uint64_t*>(arena.host_ptr(va, 3 * kPageSize));
                if (p == nullptr || p[0] != 0) {
                    failures.fetch_add(1);
                }
                const uint64_t tag = (static_cast<uint64_t>(t) << 32) | static_cast<uint64_t>(i);
                if (p != nullptr) {
                    p[0] = tag;
                    p[(3 * kPageSize) / 8 - 1] = tag;
                }
                uint64_t back = 0;
                if (!arena.read(va + 3 * kPageSize - 8, &back, 8) || back != tag) {
                    failures.fetch_add(1);
                }
                if (!arena.virtual_free(va, 0, kMemRelease)) {
                    failures.fetch_add(1);
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    CHECK(failures.load() == 0);
    CHECK(arena.reservation_count() == 0 && arena.committed_bytes() == 0);
}

void test_process_arena() {
    using namespace playgta6::os::win32;
    VirtualMemoryArena& process = default_vmem_arena();
    CHECK(process.valid());
    CHECK(process.space_base() == playgta6::kGuestAddressSpaceBase);
    CHECK(process.space_size() == kProcessGuestSpaceSize);
    CHECK(get_vmem_arena() == &process);
    CHECK(__playgta6_guest_base == process.guest_base());
#if defined(__EMSCRIPTEN__)
    CHECK(process.is_identity());  // guest VA == linear-memory address in the browser build
#endif
    std::printf("  process arena: guest_base=0x%llx (%s), %llu GiB\n",
                static_cast<unsigned long long>(process.guest_base()),
                process.is_identity() ? "identity" : "offset",
                static_cast<unsigned long long>(process.space_size() >> 30));

    // The preferred image base is available in the process space.
    const uint64_t image = process.reserve(kImageBase, 0x10000, PageProtect::ReadWrite);
    CHECK(image == kImageBase);
    CHECK(process.commit(image, 0x10000, PageProtect::ReadWrite));
    CHECK(process.host_ptr(image) ==
          reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(__playgta6_guest_base + image)));
    CHECK(process.release(image));

    // Installing another arena republishes the translation base, uninstalling restores it.
    VirtualMemoryArena other;
    set_vmem_arena(&other);
    CHECK(__playgta6_guest_base == other.guest_base());
    set_vmem_arena(nullptr);
    CHECK(__playgta6_guest_base == process.guest_base());
}

// take_written_pages: pages written (or recommitted, i.e. zero-filled again) since the previous
// call. Windows tracks writes (MEM_WRITE_WATCH); other hosts report "unknown" (false).
void test_written_pages() {
    set_write_tracking(true);
    VirtualMemoryArena arena;
    set_write_tracking(false);
    const uint64_t a = arena.reserve(0, 64 * 1024);
    CHECK(a != 0);
    CHECK(arena.commit(a, 64 * 1024, PageProtect::ReadWrite));
    uint8_t* p = arena.host_ptr(a, 64 * 1024);
    CHECK(p != nullptr);
    if (p == nullptr) {
        return;
    }
    // A fresh caller (seq 0) may be further behind than the commit log remembers: the first call
    // can answer "unknown", and it brings `seq` up to date either way.
    uint64_t seq = 0;
    std::vector<uint64_t> pages;
    take_written_pages(p, 64 * 1024, &seq, &pages);
    pages.clear();
    const bool tracked = take_written_pages(p, 64 * 1024, &seq, &pages);
#if defined(_WIN32)
    CHECK(tracked);
#endif
    if (!tracked) {
        return;
    }
    // Primed: nothing new until something is written.
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &seq, &pages) && pages.empty());
    p[5 * kWrittenPageSize + 3] = 1;  // a plain store (lifted code writes this way)
    const uint64_t v = 7;
    CHECK(arena.write(a + 9 * kWrittenPageSize, &v, sizeof(v)));
    CHECK(take_written_pages(p, 64 * 1024, &seq, &pages));
    CHECK(pages == (std::vector<uint64_t>{5 * kWrittenPageSize, 9 * kWrittenPageSize}));
    // Reported once.
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &seq, &pages) && pages.empty());
    // An unaligned range reports offsets from its page-aligned start; pages outside it are not
    // consumed for other callers.
    p[2 * kWrittenPageSize] = 2;
    p[12 * kWrittenPageSize] = 3;
    CHECK(take_written_pages(p + 2 * kWrittenPageSize + 100, 100, &seq, &pages));
    CHECK(pages == (std::vector<uint64_t>{0}));
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &seq, &pages));
    CHECK(pages == (std::vector<uint64_t>{12 * kWrittenPageSize}));
    // Decommit + recommit zero-fills without a write: still reported.
    pages.clear();
    CHECK(arena.decommit(a + 4 * kPageSize, kPageSize));
    CHECK(arena.commit(a + 4 * kPageSize, kPageSize, PageProtect::ReadWrite));
    CHECK(take_written_pages(p, 64 * 1024, &seq, &pages));
    bool recommitted = !pages.empty();
    for (uint64_t off : pages) {
        recommitted &= off >= 4 * kPageSize && off < 5 * kPageSize;
    }
    CHECK(recommitted);
    // A caller further behind than the remembered commits gets "unknown".
    uint64_t stale = seq + 1;
    pages.clear();
    CHECK(!take_written_pages(p, 64 * 1024, &stale, &pages));
    CHECK(arena.release(a));
}

// O36: the OS write bits are consumed once and fanned out: every WriteCursor is told each page
// written since its own last take, whichever reader consumed the bits, and no write is lost.
void test_written_pages_fanout() {
    set_write_tracking(true);
    VirtualMemoryArena arena;
    set_write_tracking(false);
    const uint64_t a = arena.reserve(0, 64 * 1024);
    CHECK(a != 0);
    CHECK(arena.commit(a, 64 * 1024, PageProtect::ReadWrite));
    uint8_t* p = arena.host_ptr(a, 64 * 1024);
    CHECK(p != nullptr);
    if (p == nullptr) {
        return;
    }
    WriteCursor x;
    WriteCursor y;
    std::vector<uint64_t> pages;
    // Prime both (first takes may report everything or answer "unknown").
    take_written_pages(p, 64 * 1024, &x, &pages);
    take_written_pages(p, 64 * 1024, &y, &pages);
    pages.clear();
    const bool tracked = take_written_pages(p, 64 * 1024, &x, &pages);
    if (!tracked) {
        return;  // not a write-watched host
    }
    CHECK(take_written_pages(p, 64 * 1024, &y, &pages));
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &x, &pages) && pages.empty());
    CHECK(take_written_pages(p, 64 * 1024, &y, &pages) && pages.empty());
    // A write consumed by x is still reported to y, and each only once.
    p[3 * kWrittenPageSize] = 1;
    CHECK(take_written_pages(p, 64 * 1024, &x, &pages));
    CHECK(pages == (std::vector<uint64_t>{3 * kWrittenPageSize}));
    pages.clear();
    p[7 * kWrittenPageSize + 5] = 1;
    CHECK(take_written_pages(p, 64 * 1024, &y, &pages));
    CHECK(pages == (std::vector<uint64_t>{3 * kWrittenPageSize, 7 * kWrittenPageSize}));
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &y, &pages) && pages.empty());
    CHECK(take_written_pages(p, 64 * 1024, &x, &pages));
    CHECK(pages == (std::vector<uint64_t>{7 * kWrittenPageSize}));
    pages.clear();
    // A narrow take consumes the OS bits of its pages only for the stamps: a reader of a wider
    // range afterwards still sees them; a reader of a disjoint range does not.
    p[1 * kWrittenPageSize] = 2;
    p[10 * kWrittenPageSize] = 2;
    WriteCursor narrow;
    take_written_pages(p, 64 * 1024, &narrow, &pages);
    pages.clear();
    take_written_pages(p, 64 * 1024, &narrow, &pages);
    pages.clear();
    p[2 * kWrittenPageSize] = 3;
    CHECK(take_written_pages(p, 4 * kWrittenPageSize, &narrow, &pages));
    CHECK(pages == (std::vector<uint64_t>{2 * kWrittenPageSize}));
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &x, &pages));
    CHECK(pages == (std::vector<uint64_t>{1 * kWrittenPageSize, 2 * kWrittenPageSize, 10 * kWrittenPageSize}));
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &y, &pages));
    CHECK(pages == (std::vector<uint64_t>{1 * kWrittenPageSize, 2 * kWrittenPageSize, 10 * kWrittenPageSize}));
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &x, &pages) && pages.empty());
    // Many writers across 2 MiB-chunk-sized spans and a legacy (commit_seq) reader in between:
    // nothing is lost for the cursor readers.
    uint64_t seq = 0;
    take_written_pages(p, 64 * 1024, &seq, &pages);
    pages.clear();
    for (uint64_t pg = 0; pg < 16; pg += 2) {
        p[pg * kWrittenPageSize] = 9;
    }
    take_written_pages(p, 64 * 1024, &seq, &pages);  // consumes the OS bits
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &x, &pages));
    CHECK(pages.size() == 8);
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &y, &pages));
    CHECK(pages.size() == 8);
    // Peek readers never reset the OS bits: an uncollected write is reported (every time until a
    // collect stamps it), and a write is never missed next to buffer-flush style takes.
    WriteCursor peeker;
    peeker.peek_watch = true;
    take_written_pages(p, 64 * 1024, &peeker, &pages);
    pages.clear();
    take_written_pages(p, 64 * 1024, &peeker, &pages);
    take_written_pages(p, 64 * 1024, &peeker, &pages);
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &peeker, &pages));
    CHECK(pages.empty());
    p[4 * kWrittenPageSize] = 5;
    CHECK(take_written_pages(p, 64 * 1024, &peeker, &pages));
    CHECK(!pages.empty() && pages[0] == 4 * kWrittenPageSize);
    pages.clear();
    p[6 * kWrittenPageSize] = 5;
    CHECK(take_written_pages(p, 64 * 1024, &x, &pages));  // a non-peek reader consumes the bits
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &peeker, &pages));
    bool saw4 = false;
    bool saw6 = false;
    for (uint64_t off : pages) {
        saw4 |= off == 4 * kWrittenPageSize;
        saw6 |= off == 6 * kWrittenPageSize;
    }
    CHECK(saw4 && saw6);  // stamped by x's take, still new for the peeker
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &peeker, &pages));
    CHECK(pages.empty() || pages.size() <= 2);  // at most a repeat of the previous stamps
    CHECK(arena.release(a));
}

}  // namespace

// A WriteBatch collects the OS write bits of its ranges once: takes inside it report the writes made
// before the batch (to every cursor), and a write after the batch shows up in the next take once
// the batch is gone.
void test_write_batch() {
    set_write_tracking(true);
    VirtualMemoryArena arena;
    set_write_tracking(false);
    const uint64_t a = arena.reserve(0, 64 * 1024);
    CHECK(a != 0);
    CHECK(arena.commit(a, 64 * 1024, PageProtect::ReadWrite));
    uint8_t* p = arena.host_ptr(a, 64 * 1024);
    CHECK(p != nullptr);
    if (p == nullptr) {
        return;
    }
    WriteCursor x;
    WriteCursor y;
    std::vector<uint64_t> pages;
    take_written_pages(p, 64 * 1024, &x, &pages);
    take_written_pages(p, 64 * 1024, &y, &pages);
    pages.clear();
    if (!take_written_pages(p, 64 * 1024, &x, &pages)) {
        return;  // not a write-watched host
    }
    take_written_pages(p, 64 * 1024, &y, &pages);
    pages.clear();
    p[2 * kWrittenPageSize] = 1;
    {
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges{
            {reinterpret_cast<uintptr_t>(p), reinterpret_cast<uintptr_t>(p) + 64 * 1024}};
        const WriteBatch batch(&ranges);
        CHECK(take_written_pages(p, 64 * 1024, &x, &pages));
        CHECK(pages == (std::vector<uint64_t>{2 * kWrittenPageSize}));
        pages.clear();
        CHECK(take_written_pages(p, 64 * 1024, &y, &pages));
        CHECK(pages == (std::vector<uint64_t>{2 * kWrittenPageSize}));
        pages.clear();
        p[5 * kWrittenPageSize] = 1;  // after the batch's collect: not reported inside it
        CHECK(take_written_pages(p, 64 * 1024, &x, &pages) && pages.empty());
    }
    CHECK(take_written_pages(p, 64 * 1024, &x, &pages));
    CHECK(pages == (std::vector<uint64_t>{5 * kWrittenPageSize}));

    // A ranges take inside a batch (served in one pass) reports the same pages per range, and the
    // other cursor still sees them afterwards.
    const uintptr_t base = reinterpret_cast<uintptr_t>(p);
    p[1 * kWrittenPageSize] = 1;
    p[9 * kWrittenPageSize] = 1;
    p[12 * kWrittenPageSize] = 1;  // between the two ranges: not reported to this take
    {
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges{{base, base + 4 * kWrittenPageSize},
                                                            {base + 8 * kWrittenPageSize, base + 11 * kWrittenPageSize}};
        std::vector<std::pair<uintptr_t, uintptr_t>> collect = ranges;
        const WriteBatch batch(&collect);
        std::vector<uintptr_t> taken;
        std::vector<uint8_t> ok;
        take_written_pages_ranges(ranges, &x, &taken, &ok);
        CHECK(ok == (std::vector<uint8_t>{1, 1}));
        std::sort(taken.begin(), taken.end());
        CHECK(taken == (std::vector<uintptr_t>{base + 1 * kWrittenPageSize, base + 9 * kWrittenPageSize}));
        taken.clear();
        take_written_pages_ranges(ranges, &x, &taken, &ok);
        CHECK(ok == (std::vector<uint8_t>{1, 1}) && taken.empty());
    }
    pages.clear();
    CHECK(take_written_pages(p, 64 * 1024, &y, &pages));
    for (uint64_t off : {uint64_t{1}, uint64_t{9}, uint64_t{12}}) {
        CHECK(std::find(pages.begin(), pages.end(), off * kWrittenPageSize) != pages.end());
    }
}

void test_mem_reset_and_reclaim_telemetry() {
    VirtualMemoryArena arena;
    CHECK(arena.valid());
    CHECK(arena.peak_committed_bytes() == 0);
    CHECK(arena.decommitted_bytes_total() == 0);
    CHECK(arena.reclaimed_pages_reused() == 0);

    const uint64_t base = arena.reserve(0, 256 * 1024, PageProtect::ReadWrite);
    CHECK(base != 0);
    // MEM_RESET / MEM_RESET_UNDO fail cleanly on uncommitted pages, free addresses, and combined flags.
    CHECK(arena.virtual_alloc(base, 4 * kPageSize, kMemReset, PageProtect::ReadWrite) == 0);
    CHECK(arena.virtual_alloc(base, 4 * kPageSize, kMemResetUndo, PageProtect::ReadWrite) == 0);
    CHECK(arena.virtual_alloc(0, 4 * kPageSize, kMemReset, PageProtect::ReadWrite) == 0);
    CHECK(arena.virtual_alloc(base + 512 * 1024, kPageSize, kMemReset, PageProtect::ReadWrite) == 0);

    uint64_t newly = 0;
    CHECK(arena.commit(base, 32 * kPageSize, PageProtect::ReadWrite, &newly));
    CHECK(newly == 32 * kPageSize);
    CHECK(arena.committed_bytes() == 32 * kPageSize);
    CHECK(arena.peak_committed_bytes() == 32 * kPageSize);
    CHECK(arena.reclaimed_pages_reused() == 0);

    // Combined flags with MEM_RESET or MEM_RESET_UNDO are rejected.
    CHECK(arena.virtual_alloc(base, 4 * kPageSize, kMemReset | kMemCommit, PageProtect::ReadWrite) == 0);
    CHECK(arena.virtual_alloc(base, 4 * kPageSize, kMemReset | kMemResetUndo, PageProtect::ReadWrite) == 0);
    // Spanning committed + uncommitted fails cleanly.
    CHECK(arena.virtual_alloc(base + 30 * kPageSize, 4 * kPageSize, kMemReset, PageProtect::ReadWrite) == 0);

    std::vector<uint8_t> data(32 * kPageSize, 0x5A);
    CHECK(arena.write(base, data.data(), data.size()));

    // MEM_RESET on committed pages preserves committed state, protection, and committed_bytes.
    CHECK(arena.virtual_alloc(base + 4 * kPageSize, 8 * kPageSize, kMemReset, PageProtect::ReadWrite) ==
          base + 4 * kPageSize);
    const MemoryBasicInfo reset_info = arena.query(base + 4 * kPageSize);
    CHECK(reset_info.state == MemState::Commit);
    CHECK(reset_info.protect == PageProtect::ReadWrite);
    CHECK(arena.committed_bytes() == 32 * kPageSize);
    // Pages remain writable after MEM_RESET (and MEM_RESET_UNDO may succeed or fail cleanly if OS discarded).
    const uint64_t undo_ret =
        arena.virtual_alloc(base + 4 * kPageSize, 8 * kPageSize, kMemResetUndo, PageProtect::ReadWrite);
    CHECK(undo_ret == 0 || undo_ret == base + 4 * kPageSize);
    const uint64_t probe = 0xDEADBEEFCAFEBABEULL;
    CHECK(arena.write(base + 4 * kPageSize, &probe, sizeof(probe)));
    uint64_t back = 0;
    CHECK(arena.read(base + 4 * kPageSize, &back, sizeof(back)) && back == probe);

    // MEM_RESET fails cleanly on placeholders.
    const uint64_t ph =
        arena.virtual_alloc(0, 2 * kAllocationGranularity, kMemReserve | kMemReservePlaceholder, PageProtect::NoAccess);
    CHECK(ph != 0);
    CHECK(arena.virtual_alloc(ph, kPageSize, kMemReset, PageProtect::NoAccess) == 0);
    CHECK(arena.virtual_free(ph, 0, kMemRelease));

    // MEM_DECOMMIT + recommit zero-guarantee and reuse telemetry.
    CHECK(arena.decommit(base + 8 * kPageSize, 12 * kPageSize));
    CHECK(arena.committed_bytes() == 20 * kPageSize);
    CHECK(arena.peak_committed_bytes() == 32 * kPageSize);
    CHECK(arena.decommitted_bytes_total() == 12 * kPageSize);
    CHECK(arena.reclaimed_pages_reused() == 0);

    CHECK(arena.commit(base + 8 * kPageSize, 12 * kPageSize, PageProtect::ReadWrite, &newly));
    CHECK(newly == 12 * kPageSize);
    CHECK(arena.reclaimed_pages_reused() == 12);
    std::vector<uint8_t> zeros(12 * kPageSize, 0xFF);
    CHECK(arena.read(base + 8 * kPageSize, zeros.data(), zeros.size()));
    CHECK(all_zero(zeros.data(), zeros.size()));

    // Release + re-reserve same VA + commit: counts reused pages across reservation lifetime too.
    CHECK(arena.release(base));
    CHECK(arena.committed_bytes() == 0);
    CHECK(arena.decommitted_bytes_total() == (12 + 32) * kPageSize);
    CHECK(arena.reserve(base, 256 * 1024, PageProtect::ReadWrite) == base);
    CHECK(arena.commit(base, 16 * kPageSize, PageProtect::ReadWrite, &newly));
    CHECK(newly == 16 * kPageSize);
    CHECK(arena.reclaimed_pages_reused() == 12 + 16);
    CHECK(arena.read(base, zeros.data(), 12 * kPageSize));
    CHECK(all_zero(zeros.data(), 12 * kPageSize));
    CHECK(arena.release(base));
}

void test_lookup_cache_consistency() {
    VirtualMemoryArena arena;
    CHECK(arena.valid());

    // Reserve > 64 regions across > 64 MiB to exercise all cache slots and collisions.
    constexpr size_t kRegions = 96;
    std::vector<uint64_t> vas;
    vas.reserve(kRegions);
    for (size_t i = 0; i < kRegions; ++i) {
        const uint64_t va = arena.reserve(0, 1 << 20, PageProtect::ReadWrite);
        CHECK(va != 0);
        CHECK(arena.commit(va, 2 * kPageSize, PageProtect::ReadWrite));
        const uint64_t tag = 0xA500000000000000ULL | static_cast<uint64_t>(i);
        CHECK(arena.write(va + kPageSize, &tag, sizeof(tag)));
        vas.push_back(va);
    }

    // Hot random-access reads & queries across all reservations.
    for (int round = 0; round < 4; ++round) {
        for (size_t i = 0; i < kRegions; ++i) {
            const size_t idx = (i * 29 + static_cast<size_t>(round) * 7) % kRegions;
            uint64_t got = 0;
            CHECK(arena.read(vas[idx] + kPageSize, &got, sizeof(got)));
            CHECK(got == (0xA500000000000000ULL | static_cast<uint64_t>(idx)));
            CHECK(arena.query(vas[idx] + kPageSize).allocation_base == vas[idx]);
        }
    }

    // Release odd reservations: lookup cache must immediately invalidate them.
    for (size_t i = 1; i < kRegions; i += 2) {
        CHECK(arena.release(vas[i]));
        CHECK(arena.host_ptr(vas[i] + kPageSize) == nullptr);
        CHECK(arena.query(vas[i] + kPageSize).state == MemState::Free);
    }
    // Even reservations remain intact and readable.
    for (size_t i = 0; i < kRegions; i += 2) {
        uint64_t got = 0;
        CHECK(arena.read(vas[i] + kPageSize, &got, sizeof(got)));
        CHECK(got == (0xA500000000000000ULL | static_cast<uint64_t>(i)));
        CHECK(arena.release(vas[i]));
    }

    // Placeholder split/replace/revert/coalesce must keep lookup cache consistent.
    constexpr uint64_t kG = kAllocationGranularity;
    const uint64_t ph =
        arena.virtual_alloc(0, 6 * kG, kMemReserve | kMemReservePlaceholder, PageProtect::NoAccess);
    CHECK(ph != 0);
    CHECK(arena.query(ph + 3 * kG).allocation_base == ph);  // prime cache with whole placeholder
    CHECK(arena.virtual_free(ph + 2 * kG, 2 * kG, kMemRelease | kMemPreservePlaceholder));
    CHECK(arena.query(ph + 3 * kG).allocation_base == ph + 2 * kG);  // cache must see split piece
    CHECK(arena.virtual_alloc(ph + 2 * kG, 2 * kG, kMemReserve | kMemCommit | kMemReplacePlaceholder,
                              PageProtect::ReadWrite) == ph + 2 * kG);
    const uint32_t marker = 0x12345678u;
    CHECK(arena.write(ph + 3 * kG, &marker, sizeof(marker)));
    CHECK(arena.virtual_free(ph + 2 * kG, 2 * kG, kMemRelease | kMemPreservePlaceholder));
    CHECK(arena.host_ptr(ph + 3 * kG) == nullptr);
    CHECK(arena.virtual_free(ph, 6 * kG, kMemRelease | kMemCoalescePlaceholders));
    CHECK(arena.query(ph + 3 * kG).allocation_base == ph);
    CHECK(arena.virtual_free(ph, 0, kMemRelease));
    CHECK(arena.query(ph + 3 * kG).state == MemState::Free);
}

// A pointer from host_ptr_const_held stays valid while held: a concurrent decommit of its pages (another
// guest thread freeing memory the GPU flush is hashing in place) waits for the release instead of
// pulling the pages out from under the reader. A plain host_ptr_const gives no such guarantee.
void test_held_pointer_blocks_decommit() {
    VirtualMemoryArena arena;
    CHECK(arena.valid());
    const uint64_t a = arena.reserve(0, 64 * 1024, PageProtect::ReadWrite);
    CHECK(arena.commit(a, 4 * kPageSize, PageProtect::ReadWrite));
    std::unique_lock<std::mutex> hold;
    CHECK(arena.host_ptr_const_held(a, 4 * kPageSize, hold) == arena.translate(a));
    CHECK(hold.owns_lock());
    std::atomic<bool> decommitted{false};
    std::thread freer([&] {
        arena.decommit(a, 4 * kPageSize);
        decommitted = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(!decommitted.load());
    CHECK(arena.translate(a)[0] == 0);  // still readable while held
    hold.unlock();
    freer.join();
    CHECK(decommitted.load());
    // A range that is not committed fails and leaves nothing held.
    std::unique_lock<std::mutex> hold2;
    CHECK(arena.host_ptr_const_held(a, kPageSize, hold2) == nullptr);
    CHECK(!hold2.owns_lock());
    CHECK(!arena.host_ptr_const_held(0, 1, hold2));
}

// #1077: Lazy commit (commit-on-first-touch on Windows) + NoAccess commit deferral + write-watch
// compatibility + HostWriteScope materialization for kernel writes.
void test_lazy_commit_savings_and_first_touch() {
    VirtualMemoryArena arena;
    CHECK(arena.valid());

    // Deferred NoAccess commits cost 0 host commit on every platform.
    constexpr uint64_t k4M = 4ull << 20;
    const uint64_t na = arena.reserve(0, k4M, PageProtect::NoAccess);
    CHECK(na != 0);
    CHECK(arena.commit(na, k4M, PageProtect::NoAccess));
    CHECK(arena.committed_bytes() == k4M);
    CHECK(arena.host_committed_bytes() == 0);
    CHECK(arena.lazy_deferred_bytes() == k4M);
    CHECK(arena.query(na).state == MemState::Commit);
    CHECK(arena.query(na).protect == PageProtect::NoAccess);
    CHECK(arena.protect(na, k4M, PageProtect::ReadWrite));
    uint64_t zero_val = 0xDEADBEEFull;
    CHECK(arena.read(na + kPageSize, &zero_val, sizeof(zero_val)) && zero_val == 0);
    CHECK(arena.release(na));
    CHECK(arena.committed_bytes() == 0 && arena.host_committed_bytes() == 0 && arena.lazy_deferred_bytes() == 0);

#if defined(_WIN32)
    constexpr uint64_t k64M = 64ull << 20;
    constexpr uint64_t k64K = 64ull << 10;
    constexpr uint64_t k1M = 1ull << 20;
    const bool prev_lazy = lazy_commit_enabled();
    set_lazy_commit(true);
    CHECK(lazy_commit_enabled());

    const uint64_t va = arena.virtual_alloc(0, k64M, kMemReserve | kMemCommit, PageProtect::ReadWrite);
    CHECK(va != 0);
    CHECK(arena.committed_bytes() == k64M);
    CHECK(arena.host_committed_bytes() == 0);
    CHECK(arena.lazy_deferred_bytes() == k64M);
    CHECK(arena.query(va + 17 * k1M).state == MemState::Commit);
    CHECK(arena.query(va + 17 * k1M).protect == PageProtect::ReadWrite);

    // OS VirtualQuery confirms the untouched pages are still MEM_RESERVE (0 OS private commit).
    MEMORY_BASIC_INFORMATION mbi{};
    CHECK(VirtualQuery(arena.translate(va), &mbi, sizeof(mbi)) == sizeof(mbi));
    CHECK(mbi.State == MEM_RESERVE);

    // arena.read() and zero fill_committed() return zeros without materializing any host pages.
    std::vector<uint8_t> buf(256 * 1024, 0xCC);
    CHECK(arena.read(va + 5 * k1M, buf.data(), buf.size()));
    CHECK(all_zero(buf.data(), buf.size()));
    CHECK(arena.fill_committed(va + 6 * k1M, nullptr, k1M));
    CHECK(arena.host_committed_bytes() == 0);
    CHECK(arena.lazy_deferred_bytes() == k64M);

    // First raw guest store faults in exactly one 64 KiB granule (99.9% untouched commit saved).
    const uint64_t faults0 = lazy_commit_faults();
    uint8_t* raw = arena.translate(va);
    raw[17 * k1M + 123] = 0x7E;
    CHECK(lazy_commit_faults() == faults0 + 1);
    CHECK(arena.host_committed_bytes() == k64K);
    CHECK(arena.lazy_deferred_bytes() == k64M - k64K);
    CHECK(raw[17 * k1M + 0] == 0 && raw[17 * k1M + 123] == 0x7E && raw[17 * k1M + k64K - 1] == 0);

    // First raw guest read faults in one 64 KiB granule, deterministically zero.
    const volatile uint8_t* vraw = raw;
    CHECK(vraw[40 * k1M + 42] == 0);
    CHECK(lazy_commit_faults() == faults0 + 2);
    CHECK(arena.host_committed_bytes() == 2 * k64K);

    // HostWriteScope eagerly materializes before a Windows kernel write (fread).
    {
        const char* path = "test_vmem_lazy_io.tmp";
        if (std::FILE* f = std::fopen(path, "wb")) {
            std::vector<uint8_t> payload(128 * 1024, 0x3D);
            std::fwrite(payload.data(), 1, payload.size(), f);
            std::fclose(f);
        }
        std::FILE* f = std::fopen(path, "rb");
        CHECK(f != nullptr);
        if (f != nullptr) {
            std::setvbuf(f, nullptr, _IONBF, 0);
            {
                const HostWriteScope host_write(raw + 2 * k1M, 128 * 1024);
                CHECK(arena.host_committed_bytes() == 4 * k64K);
                CHECK(std::fread(raw + 2 * k1M, 1, 128 * 1024, f) == 128 * 1024);
            }
            std::fclose(f);
            CHECK(raw[2 * k1M] == 0x3D && raw[2 * k1M + 128 * 1024 - 1] == 0x3D);
        }
        std::remove(path);
    }

    // Protection changes on unmaterialized sub-ranges do not materialize, and remain enforced.
    const uint64_t hc_before_prot = arena.host_committed_bytes();
    CHECK(arena.protect(va + 32 * k1M, k1M, PageProtect::ReadOnly));
    CHECK(arena.protect(va + 33 * k1M, k1M, PageProtect::NoAccess));
    CHECK(arena.host_committed_bytes() == hc_before_prot);
#if defined(_MSC_VER)
    CHECK(!faults_on_read(raw + 32 * k1M + 16));
    CHECK(raw[32 * k1M + 16] == 0);
    CHECK(faults_on_write(raw + 32 * k1M + 16));
    CHECK(faults_on_read(raw + 33 * k1M + 16));
    CHECK(faults_on_write(raw + 33 * k1M + 16));
#endif

    // MEM_RESET succeeds cleanly across mixed materialized + unmaterialized lazy ranges.
    const uint64_t hc_before_reset = arena.host_committed_bytes();
    CHECK(arena.virtual_alloc(va, k64M, kMemReset, PageProtect::ReadWrite) == va);
    CHECK(arena.host_committed_bytes() == hc_before_reset);

    // Decommit reclaims both materialized and deferred bytes.
    CHECK(arena.decommit(va, k64M));
    CHECK(arena.committed_bytes() == 0);
    CHECK(arena.host_committed_bytes() == 0);
    CHECK(arena.lazy_deferred_bytes() == 0);
#if defined(_MSC_VER)
    CHECK(faults_on_read(raw + 17 * k1M + 123));
#endif
    CHECK(arena.release(va));

    // Toggle off -> eager commit materializes immediately.
    set_lazy_commit(false);
    const uint64_t eager = arena.virtual_alloc(0, 256 * 1024, kMemReserve | kMemCommit, PageProtect::ReadWrite);
    CHECK(eager != 0);
    CHECK(arena.host_committed_bytes() == 256 * 1024);
    CHECK(arena.lazy_deferred_bytes() == 0);
    CHECK(arena.release(eager));
    set_lazy_commit(true);

    // MEM_WRITE_WATCH tracks writes on lazy-committed ranges without reporting untouched pages.
    set_write_tracking(true);
    VirtualMemoryArena watched;
    set_write_tracking(false);
    const uint64_t wva = watched.virtual_alloc(0, 256 * 1024, kMemReserve | kMemCommit, PageProtect::ReadWrite);
    CHECK(wva != 0 && watched.host_committed_bytes() == 0);
    uint8_t* wp = watched.host_ptr(wva, 256 * 1024);
    CHECK(wp != nullptr && watched.host_committed_bytes() == 0);
    WriteCursor cur;
    std::vector<uint64_t> dirty;
    take_written_pages(wp, 256 * 1024, &cur, &dirty);
    dirty.clear();
    CHECK(take_written_pages(wp, 256 * 1024, &cur, &dirty) && dirty.empty());
    CHECK(watched.host_committed_bytes() == 0);
    wp[20 * kWrittenPageSize + 7] = 0x99;
    CHECK(watched.host_committed_bytes() == k64K);
    CHECK(take_written_pages(wp, 256 * 1024, &cur, &dirty));
    CHECK(dirty == (std::vector<uint64_t>{20 * kWrittenPageSize}));
    CHECK(watched.release(wva));

    set_lazy_commit(prev_lazy);
#endif
}

// #1077 + #1913/#1928: Concurrent first-touch VEH faults (including while holding
// VirtualMemoryArena::mutex_ via host_ptr_const_held) racing with decommit/recommit/protect/reset.
void test_lazy_commit_concurrent_stress() {
    VirtualMemoryArena arena;
    CHECK(arena.valid());
    constexpr uint64_t kGranules = 32;  // 32 x 64 KiB = 2 MiB
    constexpr uint64_t k64K = 64ull << 10;
    constexpr uint64_t kTotal = kGranules * k64K;
    const uint64_t va = arena.virtual_alloc(0, kTotal, kMemReserve | kMemCommit, PageProtect::ReadWrite);
    CHECK(va != 0);

    std::atomic<int> failures{0};
    std::atomic<bool> stop{false};

    // 4 reader threads using host_ptr_const_held (holds arena::mutex_ while faulting on lazy pages!)
    // plus arena.read().
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&, r] {
            uint64_t s = 0x123456789ull + static_cast<uint64_t>(r) * 97;
            std::vector<uint8_t> tmp(kPageSize);
            while (!stop.load(std::memory_order_relaxed)) {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                const uint64_t g = s % kGranules;
                const uint64_t addr = va + g * k64K;
                if ((s & 1) == 0) {
                    std::unique_lock<std::mutex> hold;
                    const uint8_t* p = arena.host_ptr_const_held(addr, k64K, hold);
                    if (p != nullptr) {
                        const uint8_t v0 = p[0];
                        const uint8_t v1 = p[k64K - 1];
                        if (v0 != 0 && v0 != 0xA5 && v0 != 0x5A) {
                            failures.fetch_add(1);
                        }
                        if (v1 != 0 && v1 != 0xA5 && v1 != 0x5A) {
                            failures.fetch_add(1);
                        }
                    }
                } else {
                    if (arena.read(addr, tmp.data(), tmp.size())) {
                        const uint8_t v = tmp[0];
                        if (v != 0 && v != 0xA5 && v != 0x5A) {
                            failures.fetch_add(1);
                        }
                    }
                }
            }
        });
    }

    // 4 mutator threads cycling lazy commits, decommits, protects, resets, writes, and zero-fills.
    std::vector<std::thread> mutators;
    for (int m = 0; m < 4; ++m) {
        mutators.emplace_back([&, m] {
            uint64_t s = 0xCAFEBABE1ull + static_cast<uint64_t>(m) * 131;
            std::vector<uint8_t> fill(k64K);
            for (int i = 0; i < 600; ++i) {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                const uint64_t g = (static_cast<uint64_t>(m) * (kGranules / 4)) + (s % (kGranules / 4));
                const uint64_t addr = va + g * k64K;
                switch (s % 6) {
                    case 0:
                        arena.decommit(addr, k64K);
                        break;
                    case 1:
                        arena.commit(addr, k64K, PageProtect::ReadWrite);
                        break;
                    case 2:
                        arena.virtual_alloc(addr, k64K, kMemReset, PageProtect::ReadWrite);
                        break;
                    case 3:
                        arena.fill_committed(addr, nullptr, k64K);
                        break;
                    case 4:
                        arena.protect(addr, k64K, (s & 16) ? PageProtect::ReadOnly : PageProtect::ReadWrite);
                        break;
                    default: {
                        const uint8_t tag = (s & 8) ? 0xA5 : 0x5A;
                        std::memset(fill.data(), tag, fill.size());
                        arena.write(addr, fill.data(), fill.size());
                        break;
                    }
                }
            }
        });
    }

    for (auto& t : mutators) {
        t.join();
    }
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : readers) {
        t.join();
    }
    CHECK(failures.load() == 0);
    CHECK(arena.release(va));
    CHECK(arena.committed_bytes() == 0 && arena.host_committed_bytes() == 0 && arena.lazy_deferred_bytes() == 0);
}

// Differential first-fit model: variable sizes, exact hints, holes and exhaustion.
void test_first_fit_prefix() {
    constexpr uint64_t gran = kAllocationGranularity;
    {
        VirtualMemoryArena holes(kDefaultArenaBase, 16 * gran);
        CHECK(holes.reserve(kDefaultArenaBase + gran, gran) == kDefaultArenaBase + gran);
        CHECK(holes.reserve(0, 2 * gran) == kDefaultArenaBase + 2 * gran);
        CHECK(holes.reserve(0, gran) == kDefaultArenaBase);
        CHECK(holes.release(kDefaultArenaBase + gran));
        CHECK(holes.reserve(0, gran) == kDefaultArenaBase + gran);
    }
    {
        VirtualMemoryArena placeholders(kDefaultArenaBase, 16 * gran);
        const auto ph = kMemReserve | kMemReservePlaceholder;
        const auto base = placeholders.virtual_alloc(0, 2 * gran, ph, PageProtect::NoAccess);
        CHECK(base == kDefaultArenaBase);
        CHECK(placeholders.virtual_alloc(0, 2 * gran, ph, PageProtect::NoAccess) == base + 2 * gran);
        // Coalesce over the lazy cursor; the predecessor now contains it.
        CHECK(placeholders.virtual_free(base, 4 * gran, kMemRelease | kMemCoalescePlaceholders));
        CHECK(placeholders.reserve(0, gran) == base + 4 * gran);
        CHECK(placeholders.virtual_free(base, gran, kMemRelease | kMemPreservePlaceholder));
        CHECK(placeholders.virtual_free(base + gran, gran, kMemRelease | kMemPreservePlaceholder));
        CHECK(placeholders.virtual_free(base + gran, 0, kMemRelease));
        CHECK(placeholders.reserve(0, gran) == base + gran);
    }
    constexpr size_t slots = 128;
    VirtualMemoryArena arena(kDefaultArenaBase, slots * gran);
    std::vector<size_t> occupied(slots, 0);
    struct Live { size_t start; size_t count; uint64_t va; };
    std::vector<Live> live;
    uint32_t seed = 0x195eac31;
    auto next = [&] { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; };
    for (size_t step = 0; step < 6000; ++step) {
        if (!live.empty() && next() % 3 == 0) {
            const size_t i = next() % live.size();
            const auto r = live[i];
            CHECK(arena.release(r.va));
            for (size_t k = r.start; k < r.start + r.count; ++k) occupied[k] = 0;
            live.erase(live.begin() + i);
            continue;
        }
        const size_t count = 1 + next() % 8;
        const bool hinted = next() % 4 == 0;
        const size_t hint = next() % slots;
        size_t expected = slots;
        for (size_t i = hinted ? hint : 0; i + count <= slots; ++i) {
            bool free = true;
            for (size_t k = i; k < i + count; ++k) free &= occupied[k] == 0;
            if (free) { expected = i; break; }
            if (hinted) break;
        }
        const uint64_t va = arena.reserve(hinted ? kDefaultArenaBase + hint * gran : 0,
                                          count * gran - kPageSize);
        CHECK(va == (expected == slots ? 0 : kDefaultArenaBase + expected * gran));
        if (va != 0 && expected != slots) {
            for (size_t k = expected; k < expected + count; ++k) occupied[k] = 1;
            live.push_back({expected, count, va});
        }
    }
    for (const auto& r : live) CHECK(arena.release(r.va));
    CHECK(arena.reserve(0, slots * gran) == kDefaultArenaBase);
}

int main() {
    std::printf("test_vmem_contiguous\n");
    test_translation_and_contiguity();
    test_zero_fill();
    test_protection_bookkeeping();
    test_protection_enforced();
    test_image_base_hint();
    test_placeholders();
    test_exhaustion();
    test_many_reservations();
    test_first_fit_prefix();
    test_concurrent_reservations();
    test_process_arena();
    test_written_pages();
    test_written_pages_fanout();
    test_write_batch();
    test_mem_reset_and_reclaim_telemetry();
    test_lookup_cache_consistency();
    test_held_pointer_blocks_decommit();
    test_lazy_commit_savings_and_first_touch();
    test_lazy_commit_concurrent_stress();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_vmem_contiguous: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_vmem_contiguous: all tests passed\n");
    return 0;
}
