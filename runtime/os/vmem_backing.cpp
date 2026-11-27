// Host backing of VirtualMemoryArena (see vmem.h): the only file in runtime/os/ that talks to the
// OS memory manager. One section per platform; nothing here leaks into headers.
#include "os/vmem_backing.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <utility>
#elif defined(__EMSCRIPTEN__)
#include <emscripten/heap.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#endif

namespace playgta6::os::backing {

#if defined(_WIN32)

namespace {

std::atomic<bool> g_write_watch{false};

// O33: the reservations made with MEM_WRITE_WATCH (base -> end). A take asks GetWriteWatch only
// inside one of them: anywhere else (zone placeholders, untracked arenas) the call fails anyway,
// and a failing call costs as much as a successful one. A leaf lock.
std::shared_mutex g_watched_mutex;
std::map<uintptr_t, uintptr_t> g_watched;

void note_watched(const void* base, uint64_t size) {
    std::unique_lock<std::shared_mutex> lock(g_watched_mutex);
    g_watched[reinterpret_cast<uintptr_t>(base)] = reinterpret_cast<uintptr_t>(base) + static_cast<uintptr_t>(size);
}

bool watched_covers(uintptr_t lo, uintptr_t hi, uintptr_t* res_lo = nullptr, uintptr_t* res_hi = nullptr) {
    std::shared_lock<std::shared_mutex> lock(g_watched_mutex);
    auto it = g_watched.upper_bound(lo);
    if (it == g_watched.begin() || hi > std::prev(it)->second) {
        return false;
    }
    if (res_lo != nullptr) {
        *res_lo = std::prev(it)->first;
        *res_hi = std::prev(it)->second;
    }
    return true;
}

DWORD win_protect(PageProtect protect) {
    if (is_writable(protect)) {
        return PAGE_READWRITE;
    }
    return is_readable(protect) ? PAGE_READONLY : PAGE_NOACCESS;
}

}  // namespace

uint8_t* reserve(uint64_t size, uint64_t preferred) {
    // MEM_WRITE_WATCH: the memory manager records written pages (take_written_pages), which lets
    // the GPU layer find changed CPU-visible buffer pages without comparing whole buffers.
    const DWORD type = MEM_RESERVE | (g_write_watch.load(std::memory_order_relaxed) ? MEM_WRITE_WATCH : 0);
    void* p = VirtualAlloc(reinterpret_cast<void*>(static_cast<uintptr_t>(preferred)),
                           static_cast<SIZE_T>(size), type, PAGE_NOACCESS);
    if (p != nullptr && preferred != 0 && reinterpret_cast<uintptr_t>(p) != preferred) {
        VirtualFree(p, 0, MEM_RELEASE);
        return nullptr;
    }
    if (p != nullptr && (type & MEM_WRITE_WATCH) != 0) {
        note_watched(p, size);
    }
    return static_cast<uint8_t*>(p);
}

// O36: shared zones (placeholders for shared page views). VirtualAlloc2, MapViewOfFile3 and
// UnmapViewOfFile2 are looked up at run time (Windows 10 1803+), so older hosts keep working
// without shared pages.
namespace {

using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void*, ULONG);
using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void*, ULONG);
using UnmapViewOfFile2Fn = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);

struct PlaceholderApi {
    VirtualAlloc2Fn virtual_alloc2 = nullptr;
    MapViewOfFile3Fn map_view3 = nullptr;
    UnmapViewOfFile2Fn unmap_view2 = nullptr;
    bool ok() const { return virtual_alloc2 != nullptr && map_view3 != nullptr && unmap_view2 != nullptr; }
};

const PlaceholderApi& placeholder_api() {
    static const PlaceholderApi api = [] {
        PlaceholderApi a;
        if (HMODULE k = GetModuleHandleW(L"kernelbase.dll")) {
            a.virtual_alloc2 = reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(k, "VirtualAlloc2"));
            a.map_view3 = reinterpret_cast<MapViewOfFile3Fn>(GetProcAddress(k, "MapViewOfFile3"));
            a.unmap_view2 = reinterpret_cast<UnmapViewOfFile2Fn>(GetProcAddress(k, "UnmapViewOfFile2"));
        }
        return a;
    }();
    return api;
}

#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif

// O33: the zone table is read on every fault that is not a tracked view: a reader/writer lock.
std::shared_mutex g_zone_mutex;
std::map<uintptr_t, std::pair<uintptr_t, uintptr_t>> g_zones;  // reservation base -> [zone start, end)

bool in_zone(uintptr_t addr) {
    std::shared_lock<std::shared_mutex> lock(g_zone_mutex);
    for (const auto& [base, zone] : g_zones) {
        if (addr >= zone.first && addr < zone.second) {
            return true;
        }
    }
    return false;
}

// R40: write tracking of shared page views. A view of a file mapping cannot carry
// MEM_WRITE_WATCH, so take_written_pages tracks them itself: every view of a page (one memory
// object) is made read-only ("armed") when a caller takes its written pages, and the first write
// through any of them faults into track_fault, which marks every view of the page written and
// makes them writable again (one fault per page and taking interval, none for pages no caller
// tracks). A write is stamped on every view of the page (a write through one changes what the
// others show) with a clock each caller compares with its own (WriteCursor::view_clock), so callers
// whose ranges overlap all see it.
//
// O33: two lock levels instead of one global mutex, so the write-fault handler, takes (flush and
// re-arm), host-write pins and maps of other pages no longer serialise on each other:
//   g_track_mutex (reader/writer): the view and page tables. Exclusive only to add or remove a
//     view (map/unmap); faults, takes, pins and guest protections hold it shared.
//   TrackedPage::mutex: one page's state (armed, pins, untracked) and its views' stamps.
// Lock order: the arena's mutex, then g_track_mutex, then one TrackedPage::mutex at a time (leaves:
// nothing here writes guest memory while holding them). The clock is atomic; see
// take_written_pages for why a take reports against the clock read before it starts.
struct TrackedPage;
struct TrackedView {
    uint64_t size = 0;
    void* page = nullptr;
    TrackedPage* tracked = nullptr;  // g_tracked_pages[page] (map nodes are stable)
    uint64_t stamp = 0;  // g_track_clock at the last write seen (or mapping: new contents); page mutex
};
struct TrackedPage {
    std::mutex mutex;
    std::vector<uintptr_t> views;  // (changed with g_track_mutex held exclusively)
    bool armed = false;      // its views are read-only: the next write faults
    bool untracked = false;  // a view got a guest protection: always reported written, never armed
    uint32_t pins = 0;       // host writes in flight (pin_views): never armed, reported written
};
std::shared_mutex g_track_mutex;
std::map<uintptr_t, TrackedView> g_views;  // view base -> view, guarded by g_track_mutex
std::map<void*, TrackedPage> g_tracked_pages;
std::atomic<uint64_t> g_track_clock{1};
std::once_flag g_track_handler_once;
std::atomic<uint64_t> g_track_faults{0};

// The `*_page_locked` helpers: g_track_mutex held (either way) and the page's mutex held.
void protect_views_page_locked(const TrackedPage& page, DWORD protect) {
    for (uintptr_t v : page.views) {
        DWORD old = 0;
        VirtualProtect(reinterpret_cast<void*>(v), static_cast<SIZE_T>(g_views.find(v)->second.size), protect, &old);
    }
}

// The page's views become writable, each stamped written.
void disarm_page_locked(TrackedPage& page) {
    const uint64_t stamp = g_track_clock.fetch_add(1) + 1;
    for (uintptr_t v : page.views) {
        g_views.find(v)->second.stamp = stamp;
    }
    if (page.armed) {
        page.armed = false;
        protect_views_page_locked(page, PAGE_READWRITE);
    }
}

void arm_page_locked(TrackedPage& page) {
    if (page.pins != 0) {
        return;  // a host write is in flight: the kernel cannot fault into track_fault
    }
    for (uintptr_t v : page.views) {
        DWORD old = 0;
        if (!VirtualProtect(reinterpret_cast<void*>(v), static_cast<SIZE_T>(g_views.find(v)->second.size),
                            PAGE_READONLY, &old)) {
            page.armed = true;  // (so disarm restores the views armed so far)
            disarm_page_locked(page);
            page.untracked = true;
            return;
        }
    }
    page.armed = true;
}

LONG CALLBACK track_fault(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD* rec = info->ExceptionRecord;
    if (rec->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || rec->NumberParameters < 2 ||
        rec->ExceptionInformation[0] != 1) {  // a write
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const uintptr_t addr = static_cast<uintptr_t>(rec->ExceptionInformation[1]);
    bool view = false;
    {
        std::shared_lock<std::shared_mutex> lock(g_track_mutex);
        auto it = g_views.upper_bound(addr);
        if (it != g_views.begin() && addr - std::prev(it)->first < std::prev(it)->second.size) {
            view = true;
            TrackedPage& page = *std::prev(it)->second.tracked;
            std::lock_guard<std::mutex> page_lock(page.mutex);
            if (page.armed) {
                disarm_page_locked(page);
                g_track_faults.fetch_add(1, std::memory_order_relaxed);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            if (!page.untracked) {
                // O33 fast path: another thread disarmed the page first, so its views are writable
                // (a take re-arming them meanwhile only faults the write in again).
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
    }
    // A view with a guest protection, or a view unmapped after it was disarmed: the write goes
    // through when the page is writable now. Anything else is a real fault.
    MEMORY_BASIC_INFORMATION mbi{};
    if ((view || in_zone(addr)) && VirtualQuery(reinterpret_cast<void*>(addr), &mbi, sizeof(mbi)) != 0 &&
        mbi.State == MEM_COMMIT && mbi.Protect == PAGE_READWRITE) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

// g_track_mutex held exclusively.
void forget_view_locked(uintptr_t at) {
    auto it = g_views.find(at);
    if (it == g_views.end()) {
        return;
    }
    if (auto pit = g_tracked_pages.find(it->second.page); pit != g_tracked_pages.end()) {
        TrackedPage& page = pit->second;
        bool empty = false;
        {
            std::lock_guard<std::mutex> page_lock(page.mutex);
            disarm_page_locked(page);  // a write racing with the unmap reaches the page's other views
            page.views.erase(std::remove(page.views.begin(), page.views.end(), at), page.views.end());
            empty = page.views.empty();
        }
        if (empty) {
            g_tracked_pages.erase(pit);
        }
    }
    g_views.erase(it);
}

void track_view(uint8_t* at, void* page, uint64_t size) {
    const uintptr_t v = reinterpret_cast<uintptr_t>(at);
    std::unique_lock<std::shared_mutex> lock(g_track_mutex);
    forget_view_locked(v);
    TrackedPage& tp = g_tracked_pages[page];
    std::lock_guard<std::mutex> page_lock(tp.mutex);
    tp.views.push_back(v);
    g_views[v] = TrackedView{size, page, &tp, 0};
    disarm_page_locked(tp);  // the new view is writable: so are the others until the next take
}

void untrack_view(uint8_t* at) {
    std::unique_lock<std::shared_mutex> lock(g_track_mutex);
    forget_view_locked(reinterpret_cast<uintptr_t>(at));
}

void untrack_range(uintptr_t lo, uintptr_t hi) {
    std::unique_lock<std::shared_mutex> lock(g_track_mutex);
    for (auto it = g_views.lower_bound(lo); it != g_views.end() && it->first < hi;) {
        const uintptr_t v = it->first;
        ++it;
        forget_view_locked(v);
    }
}

// A guest protection of [p, p + len): views it touches become writable and their pages
// untracked unless it is plain read/write.
void track_protect(uint8_t* p, uint64_t len, bool read_write) {
    const uintptr_t lo = reinterpret_cast<uintptr_t>(p);
    const uintptr_t hi = lo + static_cast<uintptr_t>(len);
    std::shared_lock<std::shared_mutex> lock(g_track_mutex);
    auto it = g_views.upper_bound(lo);
    if (it != g_views.begin()) {
        --it;
    }
    for (; it != g_views.end() && it->first < hi; ++it) {
        if (it->first + it->second.size <= lo) {
            continue;
        }
        TrackedPage& page = *it->second.tracked;
        std::lock_guard<std::mutex> page_lock(page.mutex);
        disarm_page_locked(page);
        page.untracked = page.untracked || !read_write;
    }
}

void all_written(uintptr_t lo, uintptr_t hi, uintptr_t origin, std::vector<uint64_t>* offsets) {
    const uintptr_t first = lo & ~(uintptr_t{kWrittenPageSize} - 1);
    if (first >= hi) {
        return;
    }
    const size_t at = offsets->size();
    offsets->resize(at + static_cast<size_t>((hi - first + kWrittenPageSize - 1) / kWrittenPageSize));
    uint64_t* out = offsets->data() + at;
    for (uintptr_t page = first; page < hi; page += kWrittenPageSize) {
        *out++ = static_cast<uint64_t>(page - origin);
    }
}

}  // namespace

uint64_t tracked_view_faults() {
    return g_track_faults.load(std::memory_order_relaxed);
}

namespace {

// #1077: Windows lazy commit (commit-on-first-touch via leaf-locked VEH).
// Large MEM_COMMIT requests (>= kLazyCommitThreshold) register [lo, hi) here without calling
// VirtualAlloc(..., MEM_COMMIT), avoiding OS PrivateUsage commit charge for untouched zero pages.
// On first read or write fault, lazy_commit_fault commits a 64 KiB aligned granule clamped to the
// contiguous unmaterialized sub-run and resumes execution.
//
// Lock order: VirtualMemoryArena::mutex_ -> g_lazy_mutex -> LazyRange::mutex (leaves: never calls
// back into VirtualMemoryArena, never touches guest memory while holding locks, and never nests
// with g_track_mutex / g_stamp_mutex). Thus a GuestRangeView / host_ptr_const_held reader holding
// VirtualMemoryArena::mutex_ can safely fault into lazy_commit_fault without self-deadlock.
constexpr uint64_t kLazyCommitThreshold = 64u << 10;  // 64 KiB
constexpr uint64_t kLazyFaultGranule = 64u << 10;     // 64 KiB

struct LazyRange {
    uintptr_t lo = 0;
    uintptr_t hi = 0;
    PageProtect protect = PageProtect::NoAccess;
    std::mutex mutex;
    size_t total_pages = 0;
    size_t materialized_pages = 0;
    std::vector<uint64_t> committed_bits;  // 1 bit per 4 KiB page; cleared when fully_materialized
    bool fully_materialized = false;
};

std::atomic<bool> g_lazy_commit{true};
std::atomic<uint64_t> g_lazy_deferred_bytes{0};
std::atomic<uint64_t> g_lazy_faults{0};
std::atomic<size_t> g_lazy_range_count{0};
std::shared_mutex g_lazy_mutex;
std::map<uintptr_t, std::unique_ptr<LazyRange>> g_lazy_ranges;
std::once_flag g_lazy_handler_once;

inline bool lazy_bit_get(const LazyRange& r, size_t idx) {
    return r.fully_materialized || ((r.committed_bits[idx >> 6] & (uint64_t{1} << (idx & 63u))) != 0);
}

inline void lazy_bit_set_run(LazyRange& r, size_t lo_idx, size_t hi_idx) {
    for (size_t i = lo_idx; i < hi_idx; ++i) {
        r.committed_bits[i >> 6] |= (uint64_t{1} << (i & 63u));
    }
}

size_t lazy_count_unmaterialized(const LazyRange& r, size_t lo_idx, size_t hi_idx) {
    if (r.fully_materialized || lo_idx >= hi_idx) {
        return 0;
    }
    size_t unmat = 0;
    for (size_t i = lo_idx; i < hi_idx; ++i) {
        if ((r.committed_bits[i >> 6] & (uint64_t{1} << (i & 63u))) == 0) {
            ++unmat;
        }
    }
    return unmat;
}

std::unique_ptr<LazyRange> make_lazy_range(uintptr_t lo, uintptr_t hi, PageProtect prot) {
    auto r = std::make_unique<LazyRange>();
    r->lo = lo;
    r->hi = hi;
    r->protect = prot;
    r->total_pages = static_cast<size_t>((hi - lo) / kPageSize);
    r->committed_bits.assign((r->total_pages + 63u) / 64u, 0);
    return r;
}

// Slices [sub_lo, sub_hi) out of `src` with protection `prot`. Returns nullptr if empty or if
// every page in [sub_lo, sub_hi) is already materialized (fully materialized ranges need no entry).
std::unique_ptr<LazyRange> slice_lazy_range(const LazyRange& src, uintptr_t sub_lo, uintptr_t sub_hi,
                                            PageProtect prot) {
    if (sub_lo >= sub_hi || src.fully_materialized) {
        return nullptr;
    }
    const size_t src_off = static_cast<size_t>((sub_lo - src.lo) / kPageSize);
    const size_t pages = static_cast<size_t>((sub_hi - sub_lo) / kPageSize);
    const size_t unmat = lazy_count_unmaterialized(src, src_off, src_off + pages);
    if (unmat == 0) {
        return nullptr;
    }
    auto dst = make_lazy_range(sub_lo, sub_hi, prot);
    dst->materialized_pages = pages - unmat;
    if (dst->materialized_pages != 0) {
        for (size_t i = 0; i < pages; ++i) {
            if (lazy_bit_get(src, src_off + i)) {
                dst->committed_bits[i >> 6] |= (uint64_t{1} << (i & 63u));
            }
        }
    }
    return dst;
}

// Commits all unmaterialized pages of `r` overlapping [req_lo, req_hi). Caller holds shared or
// exclusive g_lazy_mutex AND r.mutex.
bool materialize_subrange_locked(LazyRange& r, uintptr_t req_lo, uintptr_t req_hi) {
    if (r.fully_materialized || r.protect == PageProtect::NoAccess || req_lo >= req_hi) {
        return true;
    }
    const uintptr_t aligned_lo = std::max<uintptr_t>(r.lo, req_lo & ~(uintptr_t{kPageSize} - 1));
    const uintptr_t aligned_hi =
        std::min<uintptr_t>(r.hi, (req_hi + kPageSize - 1) & ~(uintptr_t{kPageSize} - 1));
    if (aligned_lo >= aligned_hi) {
        return true;
    }
    const size_t first = static_cast<size_t>((aligned_lo - r.lo) / kPageSize);
    const size_t last = static_cast<size_t>((aligned_hi - r.lo) / kPageSize);
    const DWORD prot = win_protect(r.protect);
    for (size_t i = first; i < last;) {
        if (lazy_bit_get(r, i)) {
            ++i;
            continue;
        }
        const size_t run_lo = i;
        while (i < last && !lazy_bit_get(r, i)) {
            ++i;
        }
        const uintptr_t run_addr = r.lo + run_lo * kPageSize;
        const uint64_t run_bytes = static_cast<uint64_t>(i - run_lo) * kPageSize;
        if (VirtualAlloc(reinterpret_cast<void*>(run_addr), static_cast<SIZE_T>(run_bytes), MEM_COMMIT, prot) ==
            nullptr) {
            return false;
        }
        lazy_bit_set_run(r, run_lo, i);
        r.materialized_pages += (i - run_lo);
        g_lazy_deferred_bytes.fetch_sub(run_bytes, std::memory_order_relaxed);
    }
    if (r.materialized_pages == r.total_pages) {
        r.fully_materialized = true;
        r.committed_bits.clear();
        r.committed_bits.shrink_to_fit();
    }
    return true;
}

bool materialize_lazy_range(uintptr_t lo, uintptr_t hi) {
    if (lo >= hi || g_lazy_range_count.load(std::memory_order_acquire) == 0) {
        return true;
    }
    std::shared_lock<std::shared_mutex> lock(g_lazy_mutex);
    auto it = g_lazy_ranges.upper_bound(lo);
    if (it != g_lazy_ranges.begin() && std::prev(it)->second->hi > lo) {
        --it;
    }
    for (; it != g_lazy_ranges.end() && it->first < hi; ++it) {
        LazyRange& r = *it->second;
        const uintptr_t ov_lo = std::max<uintptr_t>(lo, r.lo);
        const uintptr_t ov_hi = std::min<uintptr_t>(hi, r.hi);
        if (ov_lo < ov_hi) {
            std::lock_guard<std::mutex> range_lock(r.mutex);
            if (!materialize_subrange_locked(r, ov_lo, ov_hi)) {
                return false;
            }
        }
    }
    return true;
}

// Erases [lo, hi) from g_lazy_ranges (splitting boundary ranges as needed) and updates
// g_lazy_deferred_bytes. Caller holds exclusive g_lazy_mutex.
void erase_lazy_range_locked(uintptr_t lo, uintptr_t hi,
                             std::vector<std::pair<uintptr_t, uintptr_t>>* materialized_runs) {
    auto it = g_lazy_ranges.upper_bound(lo);
    if (it != g_lazy_ranges.begin() && std::prev(it)->second->hi > lo) {
        --it;
    }
    std::vector<std::unique_ptr<LazyRange>> keep;
    while (it != g_lazy_ranges.end() && it->first < hi) {
        std::unique_ptr<LazyRange> r = std::move(it->second);
        it = g_lazy_ranges.erase(it);
        const uintptr_t ov_lo = std::max<uintptr_t>(lo, r->lo);
        const uintptr_t ov_hi = std::min<uintptr_t>(hi, r->hi);
        const size_t ov_first = static_cast<size_t>((ov_lo - r->lo) / kPageSize);
        const size_t ov_last = static_cast<size_t>((ov_hi - r->lo) / kPageSize);
        const size_t unmat = lazy_count_unmaterialized(*r, ov_first, ov_last);
        if (unmat != 0) {
            g_lazy_deferred_bytes.fetch_sub(static_cast<uint64_t>(unmat) * kPageSize, std::memory_order_relaxed);
        }
        if (materialized_runs != nullptr && unmat < (ov_last - ov_first)) {
            if (r->fully_materialized) {
                materialized_runs->emplace_back(ov_lo, ov_hi);
            } else {
                for (size_t i = ov_first; i < ov_last;) {
                    if (!lazy_bit_get(*r, i)) {
                        ++i;
                        continue;
                    }
                    const size_t run_lo = i;
                    while (i < ov_last && lazy_bit_get(*r, i)) {
                        ++i;
                    }
                    materialized_runs->emplace_back(r->lo + run_lo * kPageSize, r->lo + i * kPageSize);
                }
            }
        }
        if (auto pre = slice_lazy_range(*r, r->lo, ov_lo, r->protect)) {
            keep.push_back(std::move(pre));
        }
        if (auto suf = slice_lazy_range(*r, ov_hi, r->hi, r->protect)) {
            keep.push_back(std::move(suf));
        }
    }
    for (auto& piece : keep) {
        const uintptr_t key = piece->lo;
        g_lazy_ranges.emplace(key, std::move(piece));
    }
    g_lazy_range_count.store(g_lazy_ranges.size(), std::memory_order_release);
}

LONG CALLBACK lazy_commit_fault(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD* rec = info->ExceptionRecord;
    if (rec->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || rec->NumberParameters < 2) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const ULONG_PTR access = rec->ExceptionInformation[0];
    if (access != 0 && access != 1) {  // read (0) or write (1) only; never DEP (8)
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if (g_lazy_range_count.load(std::memory_order_acquire) == 0) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const uintptr_t addr = static_cast<uintptr_t>(rec->ExceptionInformation[1]);
    std::shared_lock<std::shared_mutex> table_lock(g_lazy_mutex);
    auto it = g_lazy_ranges.upper_bound(addr);
    if (it == g_lazy_ranges.begin()) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    --it;
    LazyRange& r = *it->second;
    if (addr < r.lo || addr >= r.hi) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // Enforce guest protection before committing: a write to ReadOnly or any access to NoAccess
    // must remain a genuine access violation.
    if (access == 1 ? !is_writable(r.protect) : !is_readable(r.protect)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    std::lock_guard<std::mutex> range_lock(r.mutex);
    const size_t page_idx = static_cast<size_t>((addr - r.lo) / kPageSize);
    if (lazy_bit_get(r, page_idx)) {
        // Another thread concurrently materialized this page first; retry the faulting access.
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    const uintptr_t page_addr = addr & ~(uintptr_t{kPageSize} - 1);
    const uintptr_t gran_base = page_addr & ~(uintptr_t{kLazyFaultGranule} - 1);
    const uintptr_t gran_lo = std::max<uintptr_t>(r.lo, gran_base);
    const uintptr_t gran_hi = std::min<uintptr_t>(r.hi, gran_base + kLazyFaultGranule);
    const size_t bound_lo = static_cast<size_t>((gran_lo - r.lo) / kPageSize);
    const size_t bound_hi = static_cast<size_t>((gran_hi - r.lo) / kPageSize);
    size_t run_lo = page_idx;
    while (run_lo > bound_lo && !lazy_bit_get(r, run_lo - 1)) {
        --run_lo;
    }
    size_t run_hi = page_idx + 1;
    while (run_hi < bound_hi && !lazy_bit_get(r, run_hi)) {
        ++run_hi;
    }
    const uintptr_t commit_lo = r.lo + run_lo * kPageSize;
    const uint64_t commit_bytes = static_cast<uint64_t>(run_hi - run_lo) * kPageSize;
    if (VirtualAlloc(reinterpret_cast<void*>(commit_lo), static_cast<SIZE_T>(commit_bytes), MEM_COMMIT,
                     win_protect(r.protect)) == nullptr) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    lazy_bit_set_run(r, run_lo, run_hi);
    r.materialized_pages += (run_hi - run_lo);
    if (r.materialized_pages == r.total_pages) {
        r.fully_materialized = true;
        r.committed_bits.clear();
        r.committed_bits.shrink_to_fit();
    }
    g_lazy_deferred_bytes.fetch_sub(commit_bytes, std::memory_order_relaxed);
    g_lazy_faults.fetch_add(1, std::memory_order_relaxed);
    return EXCEPTION_CONTINUE_EXECUTION;
}

}  // namespace

void pin_views(uintptr_t lo, uintptr_t hi, bool pin) {
    // #1077: Kernel writes (ReadFile/fread/DStorage/Httpfs) cannot trigger user-mode VEH, so
    // materialize any lazy-committed pages overlapping [lo, hi) before the kernel writes to them.
    if (pin) {
        materialize_lazy_range(lo, hi);
    }
    std::shared_lock<std::shared_mutex> lock(g_track_mutex);
    if (g_views.empty()) {
        return;
    }
    auto it = g_views.upper_bound(lo);
    if (it != g_views.begin()) {
        --it;
    }
    void* last = nullptr;  // (consecutive views of one page: pinned once)
    for (; it != g_views.end() && it->first < hi; ++it) {
        if (it->first + it->second.size <= lo || it->second.page == last) {
            continue;
        }
        last = it->second.page;
        TrackedPage& page = *it->second.tracked;
        std::lock_guard<std::mutex> page_lock(page.mutex);
        if (pin) {
            ++page.pins;
        } else if (page.pins != 0) {
            --page.pins;
        }
        // Writable for the kernel before the write, stamped written again after it (a take
        // between the two saw the page pinned, i.e. written, and left it unarmed).
        disarm_page_locked(page);
    }
}

namespace {
void drop_stamps(uintptr_t lo, uintptr_t hi);  // O36 (below)
void drop_stamps_covered(uintptr_t lo, uintptr_t hi);  // #1077 (below)
}  // namespace

void release(uint8_t* base, uint64_t size) {
    if (base != nullptr && size != 0 && g_lazy_range_count.load(std::memory_order_acquire) != 0) {
        std::unique_lock<std::shared_mutex> lock(g_lazy_mutex);
        const uintptr_t lo = reinterpret_cast<uintptr_t>(base);
        erase_lazy_range_locked(lo, lo + static_cast<uintptr_t>(size), nullptr);
    }
    {
        uintptr_t watched_end = 0;
        {
            std::unique_lock<std::shared_mutex> lock(g_watched_mutex);
            if (auto it = g_watched.find(reinterpret_cast<uintptr_t>(base)); it != g_watched.end()) {
                watched_end = it->second;
                g_watched.erase(it);
            }
        }
        if (watched_end > reinterpret_cast<uintptr_t>(base)) {
            drop_stamps(reinterpret_cast<uintptr_t>(base), watched_end);
        }
    }
    std::pair<uintptr_t, uintptr_t> zone{0, 0};
    {
        std::unique_lock<std::shared_mutex> lock(g_zone_mutex);
        if (auto it = g_zones.find(reinterpret_cast<uintptr_t>(base)); it != g_zones.end()) {
            zone = it->second;
            g_zones.erase(it);
        }
    }
    if (zone.second != 0) {
        untrack_range(zone.first, zone.second);
    }
    if (zone.first != 0 && zone.first == reinterpret_cast<uintptr_t>(base)) {
        zone.first = 0;  // (no lower part)
    }
    if (zone.second != 0) {
        // The zone: views (unmapped, then their placeholders) and placeholders, piece by piece.
        uintptr_t at = zone.first != 0 ? zone.first : reinterpret_cast<uintptr_t>(base);
        while (at < zone.second) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (VirtualQuery(reinterpret_cast<void*>(at), &mbi, sizeof(mbi)) == 0 || mbi.RegionSize == 0) {
                break;
            }
            if (mbi.Type == MEM_MAPPED) {
                UnmapViewOfFile(mbi.BaseAddress);
            } else if (mbi.State != MEM_FREE) {
                VirtualFree(mbi.AllocationBase, 0, MEM_RELEASE);
            }
            at = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
        }
        if (zone.first == 0) {
            return;
        }
    }
    VirtualFree(base, 0, MEM_RELEASE);
}

bool shared_pages_supported() {
    return placeholder_api().ok();
}

uint8_t* reserve_with_shared_zone(uint64_t size, uint64_t preferred, uint64_t zone_offset) {
    const PlaceholderApi& api = placeholder_api();
    if (!api.ok() || zone_offset >= size || zone_offset % (64u << 10) != 0) {
        return reserve(size, preferred);
    }
    void* p = api.virtual_alloc2(GetCurrentProcess(), reinterpret_cast<void*>(static_cast<uintptr_t>(preferred)),
                                 static_cast<SIZE_T>(size), MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
                                 nullptr, 0);
    if (p == nullptr) {
        return nullptr;
    }
    if (preferred != 0 && reinterpret_cast<uintptr_t>(p) != preferred) {
        VirtualFree(p, 0, MEM_RELEASE);
        return nullptr;
    }
    uint8_t* base = static_cast<uint8_t*>(p);
    if (zone_offset != 0) {
        // Split off the lower part and make it an ordinary reservation (with write tracking when
        // asked for and the host allows it on a replaced placeholder).
        if (!VirtualFree(base, static_cast<SIZE_T>(zone_offset), MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            VirtualFree(base, 0, MEM_RELEASE);
            return nullptr;
        }
        const ULONG ww = g_write_watch.load(std::memory_order_relaxed) ? MEM_WRITE_WATCH : 0;
        void* lower = api.virtual_alloc2(GetCurrentProcess(), base, static_cast<SIZE_T>(zone_offset),
                                         MEM_RESERVE | MEM_REPLACE_PLACEHOLDER | ww, PAGE_NOACCESS, nullptr, 0);
        if (lower != nullptr && ww != 0) {
            note_watched(base, zone_offset);
        } else if (lower == nullptr && ww != 0) {
            lower = api.virtual_alloc2(GetCurrentProcess(), base, static_cast<SIZE_T>(zone_offset),
                                       MEM_RESERVE | MEM_REPLACE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
        }
        if (lower == nullptr) {
            VirtualFree(base, 0, MEM_RELEASE);
            VirtualFree(base + zone_offset, 0, MEM_RELEASE);
            return nullptr;
        }
    }
    std::unique_lock<std::shared_mutex> lock(g_zone_mutex);
    g_zones[reinterpret_cast<uintptr_t>(base)] = {reinterpret_cast<uintptr_t>(base + zone_offset),
                                                  reinterpret_cast<uintptr_t>(base + size)};
    return base;
}

void* create_shared_page(uint64_t size) {
    HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, static_cast<DWORD>(size >> 32),
                                  static_cast<DWORD>(size & 0xFFFFFFFFu), nullptr);
    return h;
}

void release_shared_page(void* page) {
    if (page != nullptr) {
        CloseHandle(page);
    }
}

bool map_shared_page(uint8_t* at, void* page, uint64_t size) {
    const PlaceholderApi& api = placeholder_api();
    if (!api.ok() || page == nullptr) {
        return false;
    }
    // [at, at + size) must be a placeholder of its own: split it out of a larger one. Other
    // threads split the same placeholder concurrently (R40: no lock around this), so a split that
    // fails is retried against what the range has become.
    for (int attempt = 0;; ++attempt) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(at, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_RESERVE) {
            return false;
        }
        if (mbi.AllocationBase == at && mbi.RegionSize == size) {  // (BaseAddress is always `at`)
            break;
        }
        if (VirtualFree(at, static_cast<SIZE_T>(size), MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            break;
        }
        if (attempt == 16) {
            return false;
        }
    }
    if (api.map_view3(page, GetCurrentProcess(), at, 0, static_cast<SIZE_T>(size), MEM_REPLACE_PLACEHOLDER,
                      PAGE_READWRITE, nullptr, 0) != at) {
        return false;
    }
    track_view(at, page, size);
    return true;
}

bool unmap_shared_page(uint8_t* at, uint64_t /*size*/) {
    const PlaceholderApi& api = placeholder_api();
    if (!api.ok()) {
        return false;
    }
    untrack_view(at);
    return api.unmap_view2(GetCurrentProcess(), at, MEM_PRESERVE_PLACEHOLDER) != 0;
}

uint8_t* view_shared_page(void* page, uint64_t size) {
    return static_cast<uint8_t*>(MapViewOfFile(page, FILE_MAP_ALL_ACCESS, 0, 0, static_cast<SIZE_T>(size)));
}

void unview_shared_page(uint8_t* view, uint64_t /*size*/) {
    if (view != nullptr) {
        UnmapViewOfFile(view);
    }
}

bool commit(uint8_t* p, uint64_t len, PageProtect protect) {
    if (p == nullptr || len == 0) {
        return false;
    }
    const uintptr_t lo = reinterpret_cast<uintptr_t>(p);
    const uintptr_t hi = lo + static_cast<uintptr_t>(len);
    if (g_lazy_commit.load(std::memory_order_relaxed) && len >= kLazyCommitThreshold &&
        protect != PageProtect::NoAccess) {
        std::call_once(g_lazy_handler_once, [] { AddVectoredExceptionHandler(1, &lazy_commit_fault); });
        auto r = make_lazy_range(lo, hi, protect);
        std::unique_lock<std::shared_mutex> lock(g_lazy_mutex);
        erase_lazy_range_locked(lo, hi, nullptr);
        g_lazy_ranges.emplace(lo, std::move(r));
        g_lazy_range_count.store(g_lazy_ranges.size(), std::memory_order_release);
        g_lazy_deferred_bytes.fetch_add(len, std::memory_order_relaxed);
        return true;
    }
    if (g_lazy_range_count.load(std::memory_order_acquire) != 0) {
        std::unique_lock<std::shared_mutex> lock(g_lazy_mutex);
        erase_lazy_range_locked(lo, hi, nullptr);
    }
    // MEM_COMMIT of reserved (or decommitted) pages yields zero-filled pages.
    return VirtualAlloc(p, static_cast<SIZE_T>(len), MEM_COMMIT, win_protect(protect)) != nullptr;
}

bool decommit(uint8_t* p, uint64_t len) {
    if (p == nullptr || len == 0) {
        return false;
    }
    const uintptr_t lo = reinterpret_cast<uintptr_t>(p);
    const uintptr_t hi = lo + static_cast<uintptr_t>(len);
    bool ok = true;
    if (g_lazy_range_count.load(std::memory_order_acquire) == 0) {
        ok = VirtualFree(p, static_cast<SIZE_T>(len), MEM_DECOMMIT) != 0;
    } else {
        std::unique_lock<std::shared_mutex> lock(g_lazy_mutex);
        std::vector<std::pair<uintptr_t, uintptr_t>> to_free;
        // Collect non-lazy gaps before erasing lazy ranges (which records materialized sub-runs).
        uintptr_t cursor = lo;
        auto it = g_lazy_ranges.upper_bound(lo);
        if (it != g_lazy_ranges.begin() && std::prev(it)->second->hi > lo) {
            --it;
        }
        for (; it != g_lazy_ranges.end() && it->first < hi; ++it) {
            const uintptr_t ov_lo = std::max<uintptr_t>(cursor, it->second->lo);
            const uintptr_t ov_hi = std::min<uintptr_t>(hi, it->second->hi);
            if (cursor < ov_lo) {
                to_free.emplace_back(cursor, ov_lo);
            }
            if (ov_hi > cursor) {
                cursor = ov_hi;
            }
        }
        if (cursor < hi) {
            to_free.emplace_back(cursor, hi);
        }
        erase_lazy_range_locked(lo, hi, &to_free);
        for (const auto& [run_lo, run_hi] : to_free) {
            if (run_hi > run_lo &&
                VirtualFree(reinterpret_cast<void*>(run_lo), static_cast<SIZE_T>(run_hi - run_lo), MEM_DECOMMIT) == 0) {
                ok = false;
            }
        }
    }
    drop_stamps_covered(lo, hi);
    return ok;
}

#ifndef MEM_RESET
#define MEM_RESET 0x00080000
#endif
#ifndef MEM_RESET_UNDO
#define MEM_RESET_UNDO 0x01000000
#endif

namespace {

bool reset_os_run(uintptr_t lo, uintptr_t hi, bool undo) {
    if (hi <= lo) {
        return true;
    }
    void* ptr = reinterpret_cast<void*>(lo);
    const SIZE_T bytes = static_cast<SIZE_T>(hi - lo);
    if (VirtualAlloc(ptr, bytes, undo ? MEM_RESET_UNDO : MEM_RESET, PAGE_NOACCESS) == nullptr) {
        return false;
    }
    if (!undo) {
        // #1077: Evict reset pages from the process working set immediately (VirtualUnlock on
        // unlocked pages trims WorkingSetSize while preserving MEM_COMMIT and MEM_RESET_UNDO).
        (void)VirtualUnlock(ptr, bytes);
    }
    return true;
}

bool reset_pages_impl(uint8_t* p, uint64_t len, bool undo) {
    if (p == nullptr || len == 0) {
        return false;
    }
    const uintptr_t lo = reinterpret_cast<uintptr_t>(p);
    const uintptr_t hi = lo + static_cast<uintptr_t>(len);
    if (g_lazy_range_count.load(std::memory_order_acquire) == 0) {
        return reset_os_run(lo, hi, undo);
    }
    std::unique_lock<std::shared_mutex> lock(g_lazy_mutex);
    uintptr_t cursor = lo;
    auto it = g_lazy_ranges.upper_bound(lo);
    if (it != g_lazy_ranges.begin() && std::prev(it)->second->hi > lo) {
        --it;
    }
    for (; it != g_lazy_ranges.end() && it->first < hi; ++it) {
        const LazyRange& r = *it->second;
        const uintptr_t ov_lo = std::max<uintptr_t>(cursor, r.lo);
        const uintptr_t ov_hi = std::min<uintptr_t>(hi, r.hi);
        if (cursor < ov_lo && !reset_os_run(cursor, ov_lo, undo)) {
            return false;
        }
        if (ov_lo < ov_hi) {
            if (r.fully_materialized) {
                if (!reset_os_run(ov_lo, ov_hi, undo)) {
                    return false;
                }
            } else {
                const size_t first = static_cast<size_t>((ov_lo - r.lo) / kPageSize);
                const size_t last = static_cast<size_t>((ov_hi - r.lo) / kPageSize);
                for (size_t i = first; i < last;) {
                    if (!lazy_bit_get(r, i)) {
                        ++i;
                        continue;
                    }
                    const size_t run_lo = i;
                    while (i < last && lazy_bit_get(r, i)) {
                        ++i;
                    }
                    if (!reset_os_run(r.lo + run_lo * kPageSize, r.lo + i * kPageSize, undo)) {
                        return false;
                    }
                }
            }
            cursor = ov_hi;
        }
    }
    return cursor < hi ? reset_os_run(cursor, hi, undo) : true;
}

}  // namespace

bool reset_pages(uint8_t* p, uint64_t len) {
    return reset_pages_impl(p, len, false);
}

bool reset_undo_pages(uint8_t* p, uint64_t len) {
    return reset_pages_impl(p, len, true);
}

bool protect(uint8_t* p, uint64_t len, PageProtect protect) {
    if (p == nullptr || len == 0) {
        return false;
    }
    const DWORD win_prot = win_protect(protect);
    track_protect(p, len, win_prot == PAGE_READWRITE);
    if (g_lazy_range_count.load(std::memory_order_acquire) == 0) {
        DWORD old = 0;
        return VirtualProtect(p, static_cast<SIZE_T>(len), win_prot, &old) != 0;
    }
    const uintptr_t lo = reinterpret_cast<uintptr_t>(p);
    const uintptr_t hi = lo + static_cast<uintptr_t>(len);
    std::unique_lock<std::shared_mutex> lock(g_lazy_mutex);
    uintptr_t cursor = lo;
    auto it = g_lazy_ranges.upper_bound(lo);
    if (it != g_lazy_ranges.begin() && std::prev(it)->second->hi > lo) {
        --it;
    }
    std::vector<std::unique_ptr<LazyRange>> keep;
    bool ok = true;
    while (it != g_lazy_ranges.end() && it->first < hi) {
        std::unique_ptr<LazyRange> r = std::move(it->second);
        it = g_lazy_ranges.erase(it);
        const uintptr_t ov_lo = std::max<uintptr_t>(cursor, r->lo);
        const uintptr_t ov_hi = std::min<uintptr_t>(hi, r->hi);
        if (cursor < ov_lo) {
            DWORD old = 0;
            if (VirtualProtect(reinterpret_cast<void*>(cursor), static_cast<SIZE_T>(ov_lo - cursor), win_prot, &old) ==
                0) {
                ok = false;
            }
        }
        if (ov_lo < ov_hi) {
            bool range_ok = true;
            if (r->fully_materialized) {
                DWORD old = 0;
                if (VirtualProtect(reinterpret_cast<void*>(ov_lo), static_cast<SIZE_T>(ov_hi - ov_lo), win_prot,
                                   &old) == 0) {
                    range_ok = false;
                }
            } else {
                const size_t first = static_cast<size_t>((ov_lo - r->lo) / kPageSize);
                const size_t last = static_cast<size_t>((ov_hi - r->lo) / kPageSize);
                for (size_t i = first; i < last;) {
                    if (!lazy_bit_get(*r, i)) {
                        ++i;
                        continue;
                    }
                    const size_t run_lo = i;
                    while (i < last && lazy_bit_get(*r, i)) {
                        ++i;
                    }
                    DWORD old = 0;
                    if (VirtualProtect(reinterpret_cast<void*>(r->lo + run_lo * kPageSize),
                                       static_cast<SIZE_T>((i - run_lo) * kPageSize), win_prot, &old) == 0) {
                        range_ok = false;
                    }
                }
            }
            if (!range_ok) {
                ok = false;
                // Keep r->protect intact so lazy_commit_fault never disagrees with OS protection.
                keep.push_back(std::move(r));
            } else {
                if (auto pre = slice_lazy_range(*r, r->lo, ov_lo, r->protect)) {
                    keep.push_back(std::move(pre));
                }
                if (auto mid = slice_lazy_range(*r, ov_lo, ov_hi, protect)) {
                    keep.push_back(std::move(mid));
                }
                if (auto suf = slice_lazy_range(*r, ov_hi, r->hi, r->protect)) {
                    keep.push_back(std::move(suf));
                }
            }
            cursor = ov_hi;
        } else {
            // No overlap (should not happen given loop condition); preserve whole range.
            keep.push_back(std::move(r));
        }
    }
    for (auto& piece : keep) {
        const uintptr_t key = piece->lo;
        g_lazy_ranges.emplace(key, std::move(piece));
    }
    g_lazy_range_count.store(g_lazy_ranges.size(), std::memory_order_release);
    if (cursor < hi) {
        DWORD old = 0;
        if (VirtualProtect(reinterpret_cast<void*>(cursor), static_cast<SIZE_T>(hi - cursor), win_prot, &old) == 0) {
            ok = false;
        }
    }
    return ok;
}

bool enforces_protection() {
    return true;
}

void set_lazy_commit(bool on) {
    g_lazy_commit.store(on, std::memory_order_relaxed);
}

bool lazy_commit_enabled() {
    return g_lazy_commit.load(std::memory_order_relaxed);
}

uint64_t lazy_deferred_bytes() {
    return g_lazy_deferred_bytes.load(std::memory_order_relaxed);
}

uint64_t lazy_deferred_bytes_in_range(const uint8_t* base, uint64_t size) {
    if (base == nullptr || size == 0 || g_lazy_range_count.load(std::memory_order_acquire) == 0) {
        return 0;
    }
    const uintptr_t lo = reinterpret_cast<uintptr_t>(base);
    const uintptr_t hi = lo + static_cast<uintptr_t>(size);
    uint64_t deferred = 0;
    std::shared_lock<std::shared_mutex> lock(g_lazy_mutex);
    auto it = g_lazy_ranges.upper_bound(lo);
    if (it != g_lazy_ranges.begin() && std::prev(it)->second->hi > lo) {
        --it;
    }
    for (; it != g_lazy_ranges.end() && it->first < hi; ++it) {
        LazyRange& r = *it->second;
        std::lock_guard<std::mutex> range_lock(r.mutex);
        if (r.fully_materialized) {
            continue;
        }
        const uintptr_t ov_lo = std::max<uintptr_t>(lo, r.lo);
        const uintptr_t ov_hi = std::min<uintptr_t>(hi, r.hi);
        if (ov_lo < ov_hi) {
            if (ov_lo == r.lo && ov_hi == r.hi) {
                deferred += static_cast<uint64_t>(r.total_pages - r.materialized_pages) * kPageSize;
            } else {
                const size_t first = static_cast<size_t>((ov_lo - r.lo) / kPageSize);
                const size_t last = static_cast<size_t>((ov_hi - r.lo) / kPageSize);
                deferred += static_cast<uint64_t>(lazy_count_unmaterialized(r, first, last)) * kPageSize;
            }
        }
    }
    return deferred;
}

uint64_t lazy_commit_faults() {
    return g_lazy_faults.load(std::memory_order_relaxed);
}

namespace {

struct BackingSegment {
    uintptr_t addr = 0;
    size_t len = 0;
    bool materialized = true;
};

template <typename Lock>
bool try_lock_bounded_lazy(Lock& lock) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
    do {
        if (lock.try_lock()) {
            return true;
        }
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

// Snapshots [lo, hi) into materialized vs unmaterialized segments while holding g_lazy_mutex
// (shared) and per-range mutexes, then releases all locks BEFORE any guest/caller memory touch
// (prevents VEH re-entrancy / SRWLOCK deadlock if dst or src faults into lazy_commit_fault or
// track_fault).
bool collect_backing_segments(uintptr_t lo, uintptr_t hi, bool try_lock_only,
                              std::vector<BackingSegment>* segs) {
    std::shared_lock<std::shared_mutex> lock(g_lazy_mutex, std::defer_lock);
    if (try_lock_only) {
        if (!try_lock_bounded_lazy(lock)) {
            return false;
        }
    } else {
        lock.lock();
    }
    uintptr_t cursor = lo;
    auto it = g_lazy_ranges.upper_bound(lo);
    if (it != g_lazy_ranges.begin() && std::prev(it)->second->hi > lo) {
        --it;
    }
    for (; it != g_lazy_ranges.end() && it->first < hi; ++it) {
        LazyRange& r = *it->second;
        const uintptr_t ov_lo = std::max<uintptr_t>(cursor, r.lo);
        const uintptr_t ov_hi = std::min<uintptr_t>(hi, r.hi);
        if (cursor < ov_lo) {
            segs->push_back({cursor, static_cast<size_t>(ov_lo - cursor), true});
        }
        if (ov_lo < ov_hi) {
            std::unique_lock<std::mutex> range_lock(r.mutex, std::defer_lock);
            if (try_lock_only) {
                if (!try_lock_bounded_lazy(range_lock)) {
                    return false;
                }
            } else {
                range_lock.lock();
            }
            if (r.fully_materialized) {
                segs->push_back({ov_lo, static_cast<size_t>(ov_hi - ov_lo), true});
            } else {
                uintptr_t p = ov_lo;
                const size_t end_idx = static_cast<size_t>((ov_hi - 1 - r.lo) / kPageSize) + 1;
                while (p < ov_hi) {
                    const size_t idx = static_cast<size_t>((p - r.lo) / kPageSize);
                    const bool mat = lazy_bit_get(r, idx);
                    size_t next_idx = idx + 1;
                    while (next_idx < end_idx && lazy_bit_get(r, next_idx) == mat) {
                        ++next_idx;
                    }
                    const uintptr_t run_end = std::min<uintptr_t>(ov_hi, r.lo + next_idx * kPageSize);
                    segs->push_back({p, static_cast<size_t>(run_end - p), mat});
                    p = run_end;
                }
            }
            cursor = ov_hi;
        }
    }
    if (cursor < hi) {
        segs->push_back({cursor, static_cast<size_t>(hi - cursor), true});
    }
    return true;
}

}  // namespace

bool read_backing(const uint8_t* src, void* dst, size_t len, bool try_lock_only) {
    if (len == 0) {
        return true;
    }
    if (g_lazy_range_count.load(std::memory_order_acquire) == 0) {
        std::memcpy(dst, src, len);
        return true;
    }
    const uintptr_t lo = reinterpret_cast<uintptr_t>(src);
    const uintptr_t hi = lo + len;
    std::vector<BackingSegment> segs;
    if (!collect_backing_segments(lo, hi, try_lock_only, &segs)) {
        return false;
    }
    // All lazy locks are released before touching guest or caller memory.
    uint8_t* out = static_cast<uint8_t*>(dst);
    for (const BackingSegment& s : segs) {
        uint8_t* dst_seg = out + (s.addr - lo);
        if (s.materialized) {
            std::memcpy(dst_seg, reinterpret_cast<const void*>(s.addr), s.len);
        } else {
            std::memset(dst_seg, 0, s.len);
        }
    }
    return true;
}

bool materialize_for_write(uint8_t* dst, size_t len) {
    if (len == 0) {
        return true;
    }
    const uintptr_t lo = reinterpret_cast<uintptr_t>(dst);
    return materialize_lazy_range(lo, lo + len);
}

void zero_backing(uint8_t* dst, size_t len) {
    if (len == 0) {
        return;
    }
    if (g_lazy_range_count.load(std::memory_order_acquire) == 0) {
        std::memset(dst, 0, len);
        return;
    }
    const uintptr_t lo = reinterpret_cast<uintptr_t>(dst);
    const uintptr_t hi = lo + len;
    std::vector<BackingSegment> segs;
    (void)collect_backing_segments(lo, hi, false, &segs);
    // All lazy locks are released before touching guest memory (safe if dst hits an armed shared view).
    for (const BackingSegment& s : segs) {
        if (s.materialized) {
            std::memset(reinterpret_cast<void*>(s.addr), 0, s.len);
        }
    }
}

uint64_t shared_view_size(const uint8_t* at) {
    if (at == nullptr) {
        return 0;
    }
    std::shared_lock<std::shared_mutex> lock(g_track_mutex);
    auto it = g_views.find(reinterpret_cast<uintptr_t>(at));
    return it != g_views.end() ? it->second.size : 0;
}

void set_write_watch(bool on) {
    g_write_watch.store(on, std::memory_order_relaxed);
}

namespace {

// O36: GetWriteWatch(RESET) is destructive, so the pages it reports are fanned out here: every
// take stamps the pages it consumed with a new epoch, and each reader keeps the epoch it has seen
// (WriteCursor::watch_epoch) and is told the pages stamped after it. Readers of the same pages
// (buffer flush, texture sync) therefore never steal a write from each other. Stamps live in
// 2 MiB chunks allocated on the first write (512 x u32 plus the newest stamp of the chunk); a
// released reservation drops its chunks. Stamping, the GetWriteWatch call itself and the reader's
// scan run under one mutex, so a reader's new epoch covers exactly what it was shown.
constexpr uintptr_t kStampChunkShift = 21;
constexpr size_t kStampChunkPages = size_t{1} << (kStampChunkShift - 12);
struct StampChunk {
    uint32_t stamp[kStampChunkPages] = {};
    uint32_t newest = 0;
};
std::mutex g_stamp_mutex;
std::map<uintptr_t, std::unique_ptr<StampChunk>> g_stamp_chunks;  // address >> kStampChunkShift
uint64_t g_watch_epoch = 0;                                       // newest epoch handed out

void drop_stamps(uintptr_t lo, uintptr_t hi) {
    std::lock_guard<std::mutex> lock(g_stamp_mutex);
    g_stamp_chunks.erase(g_stamp_chunks.lower_bound(lo >> kStampChunkShift),
                         g_stamp_chunks.lower_bound(((hi - 1) >> kStampChunkShift) + 1));
}

// #1077: Reclaim StampChunks wholly covered by a decommitted range (leaving partially covered
// chunks intact so live sibling buffers sharing a 2 MiB chunk never lose their write stamps).
void drop_stamps_covered(uintptr_t lo, uintptr_t hi) {
    constexpr uintptr_t kChunkSize = uintptr_t{1} << kStampChunkShift;
    const uintptr_t aligned_lo = (lo + kChunkSize - 1) & ~(kChunkSize - 1);
    const uintptr_t aligned_hi = hi & ~(kChunkSize - 1);
    if (aligned_lo >= aligned_hi) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_stamp_mutex);
    if (g_stamp_chunks.empty()) {
        return;
    }
    g_stamp_chunks.erase(g_stamp_chunks.lower_bound(aligned_lo >> kStampChunkShift),
                         g_stamp_chunks.lower_bound(aligned_hi >> kStampChunkShift));
}

bool written_pages_by_watch(uintptr_t start, uintptr_t end, uintptr_t origin, uint64_t* epoch, bool peek,
                            std::vector<uint64_t>* offsets);

// Ranges this thread collected in one go (precollect_begin): a stamped take inside one needs no
// GetWriteWatch call of its own (a write after the collect is reported by the next one).
thread_local std::vector<std::pair<uintptr_t, uintptr_t>> t_precollected;

// (Sorted and disjoint: precollect_begin builds them in address order.)
bool precollected_covers(uintptr_t lo, uintptr_t hi) {
    auto it = std::upper_bound(t_precollected.begin(), t_precollected.end(), lo,
                               [](uintptr_t v, const std::pair<uintptr_t, uintptr_t>& r) { return v < r.first; });
    return it != t_precollected.begin() && hi <= std::prev(it)->second;
}

// Peek mode: GetWriteWatch without RESET over the range (no page protection change, no TLB
// flush) reports the bits nobody has collected yet; those count as written. When there are some and
// the window around the range was last collected more than kPeekCollectMs ago, the window is
// collected (one reset for all readers' pages, stamped like any take) and the range peeked again.
// Result: the pages stamped after *epoch plus the pages still uncollected; never fewer than written.
// The reader's new epoch is the one read before it looked at anything: a collect running
// meanwhile stamps later values, which the next call reports again (twice, never missed).
constexpr int64_t kPeekCollectMs = 8;
std::map<uintptr_t, int64_t> g_last_collect_ms;  // window start -> last collect, under g_stamp_mutex

int64_t now_ms() {
    return static_cast<int64_t>(GetTickCount64());
}

bool written_pages_peek(uintptr_t start, uintptr_t end, uintptr_t origin, uintptr_t res_lo, uintptr_t res_hi,
                        uint64_t* epoch, std::vector<uint64_t>* offsets) {
    thread_local std::vector<PVOID> bits;
    const uint64_t pages = (end - start + kWrittenPageSize - 1) / kWrittenPageSize;
    const size_t cap = static_cast<size_t>(std::min<uint64_t>(pages, 4096));
    if (bits.size() < cap) {
        bits.resize(cap);
    }
    uint64_t e0 = 0;
    {
        std::lock_guard<std::mutex> lock(g_stamp_mutex);
        e0 = g_watch_epoch;
    }
    auto peek_bits = [&](size_t* n) {
        ULONG_PTR count = cap;
        DWORD granularity = 0;
        if (GetWriteWatch(0, reinterpret_cast<PVOID>(start), static_cast<SIZE_T>(end - start), bits.data(), &count,
                          &granularity) != 0 ||
            granularity != kWrittenPageSize) {
            return false;
        }
        *n = static_cast<size_t>(count);
        return !(*n == cap && cap < pages);  // more bits than the buffer: unknown (caller: all written)
    };
    size_t n = 0;
    if (!peek_bits(&n)) {
        return false;
    }
    if (n > 0) {
        // Collect a 64 MiB aligned window around the range (clipped to the reservation), at most
        // every kPeekCollectMs per window: one reset for every reader's pages in it.
        constexpr uintptr_t kWindow = uintptr_t{64} << 20;
        const uintptr_t wlo = std::max<uintptr_t>(res_lo, start & ~(kWindow - 1));
        const uintptr_t whi = std::min<uintptr_t>(res_hi, (end + kWindow - 1) & ~(kWindow - 1));
        bool due = false;
        {
            std::lock_guard<std::mutex> lock(g_stamp_mutex);
            int64_t& last = g_last_collect_ms[wlo];
            if (now_ms() - last >= kPeekCollectMs) {
                last = now_ms();
                due = true;
            }
        }
        if (due) {
            std::vector<uint64_t> none;
            if (written_pages_by_watch(wlo, whi, wlo, nullptr, false, &none)) {
                if (!peek_bits(&n)) {
                    return false;
                }
            }
        }
    }
    const uint64_t seen = *epoch;
    for (size_t i = 0; i < n; ++i) {
        offsets->push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bits[i]) - origin));
    }
    std::lock_guard<std::mutex> lock(g_stamp_mutex);
    for (auto it = g_stamp_chunks.lower_bound(start >> kStampChunkShift);
         it != g_stamp_chunks.end() && (it->first << kStampChunkShift) < end; ++it) {
        const StampChunk& chunk = *it->second;
        if (chunk.newest <= seen) {
            continue;
        }
        const uintptr_t chunk_base = it->first << kStampChunkShift;
        const uintptr_t lo = std::max<uintptr_t>(chunk_base, start);
        const uintptr_t hi = std::min<uintptr_t>(chunk_base + (uintptr_t{1} << kStampChunkShift), end);
        for (uintptr_t a = lo; a < hi; a += kWrittenPageSize) {
            if (chunk.stamp[(a >> 12) & (kStampChunkPages - 1)] > seen) {
                offsets->push_back(static_cast<uint64_t>(a - origin));
            }
        }
    }
    *epoch = e0;
    return true;
}

// MEM_WRITE_WATCH pages of [start, end) (one reservation), as offsets from `origin`.
// O33: GetWriteWatch's output buffer is kept per thread (it was up to 512 KiB allocated and zeroed
// on every take), and the pages it reports are appended in one go.
// `epoch` null: the pages this call consumed. Otherwise: the pages stamped after *epoch, by this
// call or by any other reader's, and *epoch moves to the newest stamp. False: not watched, or the
// call failed (the caller then treats the range as written; stamps already made stay valid).
bool written_pages_by_watch(uintptr_t start, uintptr_t end, uintptr_t origin, uint64_t* epoch, bool peek,
                            std::vector<uint64_t>* offsets) {
    uintptr_t res_lo = 0;
    uintptr_t res_hi = 0;
    if (!watched_covers(start, end, &res_lo, &res_hi)) {
        return false;
    }
    if (peek && epoch != nullptr) {
        return written_pages_peek(start, end, origin, res_lo, res_hi, epoch, offsets);
    }
    thread_local std::vector<PVOID> written;
    const uint64_t pages = (end - start + kWrittenPageSize - 1) / kWrittenPageSize;
    const size_t cap = static_cast<size_t>(std::min<uint64_t>(pages, 65536));
    if (written.size() < cap) {
        written.resize(cap);
    }
    const size_t before = offsets->size();
    std::lock_guard<std::mutex> lock(g_stamp_mutex);
    const uint64_t seen = epoch != nullptr ? *epoch : 0;
    // Stamps are 32-bit; once the epochs run out every take reports "unknown" (conservative).
    if (g_watch_epoch >= 0xFFFFFFF0u) {
        return false;
    }
    const uint32_t mine = static_cast<uint32_t>(g_watch_epoch + 1);
    bool consumed_any = false;
    bool ok = true;
    const bool collected_already = epoch != nullptr && precollected_covers(start, end);
    for (; !collected_already;) {
        ULONG_PTR count = cap;
        DWORD granularity = 0;
        // Each call resets what it reports (WRITE_WATCH_FLAG_RESET, no separate ResetWriteWatch);
        // a full buffer may leave more pages, so call again.
        if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, reinterpret_cast<PVOID>(start), static_cast<SIZE_T>(end - start),
                          written.data(), &count, &granularity) != 0 ||
            granularity != kWrittenPageSize) {
            ok = false;
            break;
        }
        if (count > 0) {
            consumed_any = true;
        }
        const size_t at = offsets->size();
        if (epoch == nullptr) {
            offsets->resize(at + static_cast<size_t>(count));
        }
        uint64_t* out = epoch == nullptr ? offsets->data() + at : nullptr;
        StampChunk* chunk = nullptr;
        uintptr_t chunk_key = ~uintptr_t{0};
        for (ULONG_PTR i = 0; i < count; ++i) {
            const uintptr_t a = reinterpret_cast<uintptr_t>(written[i]);
            if (out != nullptr) {
                out[i] = static_cast<uint64_t>(a - origin);
            }
            if ((a >> kStampChunkShift) != chunk_key) {
                chunk_key = a >> kStampChunkShift;
                auto& slot = g_stamp_chunks[chunk_key];
                if (!slot) {
                    slot = std::make_unique<StampChunk>();
                }
                chunk = slot.get();
            }
            chunk->stamp[(a >> 12) & (kStampChunkPages - 1)] = mine;
            chunk->newest = mine;
        }
        if (count < cap) {
            break;
        }
    }
    if (consumed_any) {
        g_watch_epoch = mine;  // published with the stamps, still under the lock
    }
    if (!ok) {
        offsets->resize(before);
        return false;
    }
    if (epoch == nullptr) {
        return true;
    }
    // Everything in the range stamped after what this reader has seen (a chunk with no newer
    // stamp is skipped whole).
    for (auto it = g_stamp_chunks.lower_bound(start >> kStampChunkShift);
         it != g_stamp_chunks.end() && (it->first << kStampChunkShift) < end; ++it) {
        const StampChunk& chunk = *it->second;
        if (chunk.newest <= seen) {
            continue;
        }
        const uintptr_t chunk_base = it->first << kStampChunkShift;
        const uintptr_t lo = std::max<uintptr_t>(chunk_base, start);
        const uintptr_t hi = std::min<uintptr_t>(chunk_base + (uintptr_t{1} << kStampChunkShift), end);
        for (uintptr_t a = lo; a < hi; a += kWrittenPageSize) {
            if (chunk.stamp[(a >> 12) & (kStampChunkPages - 1)] > seen) {
                offsets->push_back(static_cast<uint64_t>(a - origin));
            }
        }
    }
    *epoch = g_watch_epoch;
    return true;
}

}  // namespace

void precollect_begin(std::vector<std::pair<uintptr_t, uintptr_t>>* ranges) {
    t_precollected.clear();
    if (ranges == nullptr || ranges->empty()) {
        return;
    }
    constexpr uintptr_t kMergeGap = uintptr_t{8} << 20;
    std::sort(ranges->begin(), ranges->end());
    std::vector<std::pair<uintptr_t, uintptr_t>> hulls;
    for (const auto& r : *ranges) {
        const uintptr_t lo = r.first & ~(uintptr_t{kWrittenPageSize} - 1);
        const uintptr_t hi = (r.second + kWrittenPageSize - 1) & ~(uintptr_t{kWrittenPageSize} - 1);
        if (!hulls.empty() && lo <= hulls.back().second + kMergeGap) {
            hulls.back().second = std::max(hulls.back().second, hi);
        } else {
            hulls.emplace_back(lo, hi);
        }
    }
    std::vector<uint64_t> none;
    for (const auto& h : hulls) {
        if (h.second <= h.first || !watched_covers(h.first, h.second)) {
            continue;
        }
        none.clear();
        if (written_pages_by_watch(h.first, h.second, h.first, nullptr, false, &none)) {
            t_precollected.push_back(h);
        }
    }
}

void precollect_end() {
    t_precollected.clear();
}

void take_precollected_pages(const std::vector<std::pair<uintptr_t, uintptr_t>>& ranges, uint64_t watch_epoch,
                             std::vector<uint8_t>* handled, std::vector<uintptr_t>* pages) {
    handled->assign(ranges.size(), 0);
    if (t_precollected.empty() || ranges.empty()) {
        return;
    }
    // Pass 1: the ranges a take would serve from the precollected bits alone (no shared page view
    // in them, inside one watched reservation and one collected hull). The same checks a single
    // take makes, each lock taken once for the batch instead of once per range.
    size_t eligible = 0;
    {
        std::shared_lock<std::shared_mutex> lock(g_track_mutex);
        for (size_t i = 0; i < ranges.size(); ++i) {
            const uintptr_t start = ranges[i].first & ~(uintptr_t{kWrittenPageSize} - 1);
            const uintptr_t end = ranges[i].second;
            if (end <= start || !precollected_covers(start, end)) {
                continue;
            }
            auto it = g_views.upper_bound(start);
            if (it != g_views.begin() && std::prev(it)->first + std::prev(it)->second.size > start) {
                continue;
            }
            if (it != g_views.end() && it->first < end) {
                continue;
            }
            (*handled)[i] = 1;
            ++eligible;
        }
    }
    if (eligible == 0) {
        return;
    }
    {
        std::shared_lock<std::shared_mutex> lock(g_watched_mutex);
        for (size_t i = 0; i < ranges.size(); ++i) {
            if ((*handled)[i] == 0) {
                continue;
            }
            const uintptr_t start = ranges[i].first & ~(uintptr_t{kWrittenPageSize} - 1);
            auto it = g_watched.upper_bound(start);
            if (it == g_watched.begin() || ranges[i].second > std::prev(it)->second) {
                (*handled)[i] = 0;
            }
        }
    }
    // Pass 2: the pages stamped after `watch_epoch` in each, under one stamp lock (what
    // written_pages_by_watch reports for a range it need not collect again).
    std::lock_guard<std::mutex> lock(g_stamp_mutex);
    if (g_watch_epoch >= 0xFFFFFFF0u) {
        handled->assign(ranges.size(), 0);  // out of epochs: the single takes report "unknown"
        return;
    }
    const uint64_t seen = watch_epoch;
    for (size_t i = 0; i < ranges.size(); ++i) {
        if ((*handled)[i] == 0) {
            continue;
        }
        const uintptr_t start = ranges[i].first & ~(uintptr_t{kWrittenPageSize} - 1);
        const uintptr_t end = ranges[i].second;
        for (auto it = g_stamp_chunks.lower_bound(start >> kStampChunkShift);
             it != g_stamp_chunks.end() && (it->first << kStampChunkShift) < end; ++it) {
            const StampChunk& chunk = *it->second;
            if (chunk.newest <= seen) {
                continue;
            }
            const uintptr_t chunk_base = it->first << kStampChunkShift;
            const uintptr_t lo = std::max<uintptr_t>(chunk_base, start);
            const uintptr_t hi = std::min<uintptr_t>(chunk_base + (uintptr_t{1} << kStampChunkShift), end);
            for (uintptr_t a = lo; a < hi; a += kWrittenPageSize) {
                if (chunk.stamp[(a >> 12) & (kStampChunkPages - 1)] > seen) {
                    pages->push_back(a);
                }
            }
        }
    }
}

namespace {

// views_only: report only through shared page views (never MEM_WRITE_WATCH, whose take is
// destructive for other readers of the same pages); false when any part of the range is not
// covered by tracked views. On false nothing is consumed (the caller's clock stays).
bool take_pages_impl(const uint8_t* p, uint64_t len, uint64_t* view_clock, uint64_t* watch_epoch, bool peek,
                     std::vector<uint64_t>* offsets, bool views_only) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(p) & ~(uintptr_t{kWrittenPageSize} - 1);
    const uintptr_t end = reinterpret_cast<uintptr_t>(p) + static_cast<uintptr_t>(len);
    {
        // R40: shared page views in the range (see TrackedView): those stamped after the caller's
        // previous call, and the pages they show armed for the next interval. Pieces of the range
        // between views (unmapped zone slots, or arena memory) are taken from MEM_WRITE_WATCH
        // where it covers them, else reported written.
        // O33: the tables shared (faults and other takes run meanwhile), each page under its own
        // lock. The caller's new position is the clock read before any page is looked at: a write
        // that disarms a page after this take armed it stamps a later value, so the next take
        // reports it (a write stamped meanwhile may be reported twice, never missed).
        const uint64_t clock = g_track_clock.load();
        std::shared_lock<std::shared_mutex> lock(g_track_mutex);
        auto it = g_views.upper_bound(start);
        if (it != g_views.begin() && std::prev(it)->first + std::prev(it)->second.size > start) {
            --it;
        }
        if (it != g_views.end() && it->first < end) {
            // Installed before anything is armed, and called first, ahead of handlers installed
            // earlier (the harness' fault reporter).
            std::call_once(g_track_handler_once, [] { AddVectoredExceptionHandler(1, &track_fault); });
            uintptr_t at = start;
            for (; it != g_views.end() && it->first < end; ++it) {
                const uintptr_t lo = std::max<uintptr_t>(it->first, start);
                const uintptr_t hi = std::min<uintptr_t>(it->first + static_cast<uintptr_t>(it->second.size), end);
                if (at < lo) {
                    if (views_only) {
                        return false;
                    }
                    if (!written_pages_by_watch(at, lo, start, watch_epoch, peek, offsets)) {
                        all_written(at, lo, start, offsets);
                    }
                }
                TrackedPage& page = *it->second.tracked;
                std::lock_guard<std::mutex> page_lock(page.mutex);
                const bool untracked = page.untracked || page.pins != 0;
                if (view_clock == nullptr || it->second.stamp > *view_clock || untracked) {
                    all_written(lo, hi, start, offsets);
                }
                if (!untracked && !page.armed) {
                    arm_page_locked(page);
                }
                at = hi;
            }
            if (at < end) {
                if (views_only) {
                    return false;
                }
                if (!written_pages_by_watch(at, end, start, watch_epoch, peek, offsets)) {
                    all_written(at, end, start, offsets);
                }
            }
            if (view_clock != nullptr) {
                *view_clock = clock;
            }
            return true;
        }
    }
    if (views_only) {
        return false;
    }
    return written_pages_by_watch(start, end, start, watch_epoch, peek, offsets);
}

}  // namespace

bool take_written_pages(const uint8_t* p, uint64_t len, uint64_t* view_clock, uint64_t* watch_epoch, bool peek,
                        std::vector<uint64_t>* offsets) {
    return take_pages_impl(p, len, view_clock, watch_epoch, peek, offsets, false);
}

bool take_written_view_pages(const uint8_t* p, uint64_t len, uint64_t* view_clock, std::vector<uint64_t>* offsets) {
    return view_clock != nullptr && take_pages_impl(p, len, view_clock, nullptr, false, offsets, true);
}

void write_positions(uint64_t* view_clock, uint64_t* watch_epoch) {
    *view_clock = g_track_clock.load();
    std::lock_guard<std::mutex> lock(g_stamp_mutex);
    *watch_epoch = g_watch_epoch;
}

#elif defined(__EMSCRIPTEN__)

// Linear memory [0, kGuestAddressSpaceBase) is the runtime's; [kGuestAddressSpaceBase, max) is the
// guest pool handed out to arenas. Uncommitted pool pages are zero (fresh from memory.grow, or
// zeroed when an "ever committed" page is committed again), so commits are zero-filled without
// touching pages that were never used.

namespace {

constexpr uint64_t kPoolStart = kGuestAddressSpaceBase;
constexpr uint64_t kPoolLimit = uint64_t{16} << 30;  // browser maximum for memory64
constexpr uint64_t kPoolPages = (kPoolLimit - kPoolStart) / kPageSize;

std::atomic<uint64_t> g_dirty[kPoolPages / 64];  // bit set: page was committed at least once
std::mutex g_pool_mutex;
std::map<uint64_t, uint64_t> g_regions;  // start -> size, guarded by g_pool_mutex
std::atomic_flag g_sbrk_lock = ATOMIC_FLAG_INIT;
std::atomic<bool> g_sbrk_fenced{false};  // set once __wrap_sbrk has run

uint64_t pool_end() {
    const uint64_t max = static_cast<uint64_t>(emscripten_get_heap_max());
    return align_down(max < kPoolLimit ? max : kPoolLimit, kAllocationGranularity);
}

}  // namespace

}  // namespace playgta6::os::backing

// The runtime heap fence: linked with -Wl,--wrap=sbrk, every sbrk (dlmalloc, emmalloc, mmap
// emulation) lands here and may not move the break into the guest pool.
extern "C" void* __real_sbrk(intptr_t increment);
extern "C" void* __wrap_sbrk(intptr_t increment) {
    using playgta6::os::backing::g_sbrk_lock;
    playgta6::os::backing::g_sbrk_fenced.store(true, std::memory_order_relaxed);
    while (g_sbrk_lock.test_and_set(std::memory_order_acquire)) {
    }
    void* result = reinterpret_cast<void*>(intptr_t{-1});
    const uintptr_t brk = reinterpret_cast<uintptr_t>(__real_sbrk(0));
    if (increment <= 0 ||
        static_cast<uint64_t>(increment) + 16 <= playgta6::kGuestAddressSpaceBase - brk) {
        result = __real_sbrk(increment);
    } else {
        errno = ENOMEM;
    }
    g_sbrk_lock.clear(std::memory_order_release);
    return result;
}

namespace playgta6::os::backing {

uint8_t* reserve(uint64_t size, uint64_t preferred) {
    // Without the sbrk fence (-Wl,--wrap=sbrk) the runtime heap could grow into the pool: refuse
    // to hand it out. sbrk(0) reaches __wrap_sbrk exactly when the fence is linked in.
    sbrk(0);
    if (!g_sbrk_fenced.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    const uint64_t end_of_pool = pool_end();
    if (size == 0 || end_of_pool <= kPoolStart || size > end_of_pool - kPoolStart) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    uint64_t start = 0;
    if (preferred != 0) {
        if (preferred < kPoolStart || preferred > end_of_pool - size) {
            return nullptr;
        }
        auto next = g_regions.lower_bound(preferred);
        if (next != g_regions.end() && next->first < preferred + size) {
            return nullptr;
        }
        if (next != g_regions.begin() && std::prev(next)->first + std::prev(next)->second > preferred) {
            return nullptr;
        }
        start = preferred;
    } else {
        // Highest gap that fits, so the low end of the pool stays free for the identity arena.
        uint64_t gap_end = end_of_pool;
        for (auto it = g_regions.rbegin(); it != g_regions.rend(); ++it) {
            const uint64_t used_end = it->first + it->second;
            if (gap_end - used_end >= size) {
                start = gap_end - size;
                break;
            }
            gap_end = it->first;
        }
        if (start == 0 && gap_end - kPoolStart >= size) {
            start = gap_end - size;
        }
        if (start == 0) {
            return nullptr;
        }
    }
    g_regions.emplace(start, size);
    return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(start));
}

void release(uint8_t* base, uint64_t /*size*/) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_regions.erase(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)));
}

bool commit(uint8_t* p, uint64_t len, PageProtect /*protect*/) {
    const uint64_t start = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
    if (start < kPoolStart || len > kPoolLimit - start) {
        return false;
    }
    const uint64_t end = start + len;
    // Grow linear memory to cover the run (it never shrinks). With threads another thread may
    // grow it concurrently; emscripten_resize_heap then reports failure, so re-check the size.
    while (static_cast<uint64_t>(emscripten_get_heap_size()) < end) {
        if (!emscripten_resize_heap(static_cast<size_t>(end)) &&
            static_cast<uint64_t>(emscripten_get_heap_size()) < end) {
            return false;
        }
    }
    // O36 zero-skip: never-committed pages stay zero from memory.grow and are NOT touched.
    // Previously-committed (dirty) pages are coalesced into contiguous runs and zeroed on recommit.
    const uint64_t first_page = (start - kPoolStart) / kPageSize;
    const uint64_t last_page = (end - kPoolStart) / kPageSize;
    uint64_t zero_run_start = last_page;
    auto flush_zero_run = [&](uint64_t run_end) {
        if (zero_run_start < run_end) {
            std::memset(reinterpret_cast<void*>(
                            static_cast<uintptr_t>(kPoolStart + zero_run_start * kPageSize)),
                        0, static_cast<size_t>((run_end - zero_run_start) * kPageSize));
            zero_run_start = last_page;
        }
    };
    for (uint64_t page = first_page; page < last_page; ++page) {
        const uint64_t bit = uint64_t{1} << (page % 64);
        if ((g_dirty[page / 64].fetch_or(bit, std::memory_order_relaxed) & bit) != 0) {
            if (zero_run_start == last_page) {
                zero_run_start = page;
            }
        } else {
            flush_zero_run(page);
        }
    }
    flush_zero_run(last_page);
    return true;
}

bool decommit(uint8_t* /*p*/, uint64_t /*len*/) {
    return true;  // linear memory cannot shrink; the pages are zeroed when committed again
}

bool reset_pages(uint8_t* p, uint64_t len) {
    const uint64_t start = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
    if (len == 0 || start < kPoolStart || len > kPoolLimit - start) {
        return false;
    }
    // Linear memory cannot discard physical pages; committed state and g_dirty stay intact.
    return start + len <= static_cast<uint64_t>(emscripten_get_heap_size());
}

bool reset_undo_pages(uint8_t* p, uint64_t len) {
    return reset_pages(p, len);
}

bool protect(uint8_t* /*p*/, uint64_t /*len*/, PageProtect /*protect*/) {
    return true;  // wasm has no page protection: tracked by the arena, not enforced
}

bool enforces_protection() {
    return false;
}

void set_lazy_commit(bool) {}
bool lazy_commit_enabled() { return false; }
uint64_t lazy_deferred_bytes() { return 0; }
uint64_t lazy_deferred_bytes_in_range(const uint8_t*, uint64_t) { return 0; }
uint64_t lazy_commit_faults() { return 0; }
bool read_backing(const uint8_t* src, void* dst, size_t len, bool) {
    std::memcpy(dst, src, len);
    return true;
}
bool materialize_for_write(uint8_t*, size_t) { return true; }
void zero_backing(uint8_t* dst, size_t len) { std::memset(dst, 0, len); }
uint64_t shared_view_size(const uint8_t*) { return 0; }

void precollect_begin(std::vector<std::pair<uintptr_t, uintptr_t>>*) {}
void precollect_end() {}
void take_precollected_pages(const std::vector<std::pair<uintptr_t, uintptr_t>>& ranges, uint64_t, std::vector<uint8_t>* handled,
                             std::vector<uintptr_t>*) {
    handled->assign(ranges.size(), 0);
}

bool take_written_pages(const uint8_t*, uint64_t, uint64_t*, uint64_t*, bool, std::vector<uint64_t>*) {
    return false;  // no write tracking in linear memory
}

bool take_written_view_pages(const uint8_t*, uint64_t, uint64_t*, std::vector<uint64_t>*) {
    return false;
}

void write_positions(uint64_t*, uint64_t*) {}

uint64_t tracked_view_faults() {
    return 0;
}

void pin_views(uintptr_t, uintptr_t, bool) {}

void set_write_watch(bool) {}

// One linear memory: no second view of anything (the translated window aliases instead).
bool shared_pages_supported() {
    return false;
}
uint8_t* reserve_with_shared_zone(uint64_t size, uint64_t preferred, uint64_t) {
    return reserve(size, preferred);
}
void* create_shared_page(uint64_t) {
    return nullptr;
}
void release_shared_page(void*) {}
bool map_shared_page(uint8_t*, void*, uint64_t) {
    return false;
}
bool unmap_shared_page(uint8_t*, uint64_t) {
    return false;
}
uint8_t* view_shared_page(void*, uint64_t) {
    return nullptr;
}
void unview_shared_page(uint8_t*, uint64_t) {}

#else  // POSIX

namespace {

constexpr int kReserveFlags = MAP_PRIVATE | MAP_ANONYMOUS
#ifdef MAP_NORESERVE
                              | MAP_NORESERVE
#endif
    ;

int posix_protect(PageProtect protect) {
    if (is_writable(protect)) {
        return PROT_READ | PROT_WRITE;
    }
    return is_readable(protect) ? PROT_READ : PROT_NONE;
}

}  // namespace

uint8_t* reserve(uint64_t size, uint64_t preferred) {
    void* p = mmap(reinterpret_cast<void*>(static_cast<uintptr_t>(preferred)),
                   static_cast<size_t>(size), PROT_NONE, kReserveFlags, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    if (preferred != 0 && reinterpret_cast<uintptr_t>(p) != preferred) {
        munmap(p, static_cast<size_t>(size));  // the hint was not honoured
        return nullptr;
    }
    return static_cast<uint8_t*>(p);
}

void release(uint8_t* base, uint64_t size) {
    munmap(base, static_cast<size_t>(size));
}

bool commit(uint8_t* p, uint64_t len, PageProtect protect) {
    // Reserved and decommitted pages are fresh anonymous mappings: zero on first touch.
    return mprotect(p, static_cast<size_t>(len), posix_protect(protect)) == 0;
}

bool decommit(uint8_t* p, uint64_t len) {
    // Replace the run with fresh inaccessible pages: frees the memory, contents read as zero
    // after the next commit (portable, unlike MADV_DONTNEED / MADV_FREE).
    return mmap(p, static_cast<size_t>(len), PROT_NONE, kReserveFlags | MAP_FIXED, -1, 0) == p;
}

bool reset_pages(uint8_t* p, uint64_t len) {
    if (p == nullptr || len == 0) {
        return false;
    }
    return madvise(p, static_cast<size_t>(len), MADV_DONTNEED) == 0;
}

bool reset_undo_pages(uint8_t* p, uint64_t len) {
    if (p == nullptr || len == 0) {
        return false;
    }
#ifdef MADV_WILLNEED
    return madvise(p, static_cast<size_t>(len), MADV_WILLNEED) == 0;
#else
    return true;
#endif
}

bool protect(uint8_t* p, uint64_t len, PageProtect protect) {
    return mprotect(p, static_cast<size_t>(len), posix_protect(protect)) == 0;
}

bool enforces_protection() {
    return true;
}

void set_lazy_commit(bool) {}
bool lazy_commit_enabled() { return false; }
uint64_t lazy_deferred_bytes() { return 0; }
uint64_t lazy_deferred_bytes_in_range(const uint8_t*, uint64_t) { return 0; }
uint64_t lazy_commit_faults() { return 0; }
bool read_backing(const uint8_t* src, void* dst, size_t len, bool) {
    std::memcpy(dst, src, len);
    return true;
}
bool materialize_for_write(uint8_t*, size_t) { return true; }
void zero_backing(uint8_t* dst, size_t len) { std::memset(dst, 0, len); }
uint64_t shared_view_size(const uint8_t*) { return 0; }

void precollect_begin(std::vector<std::pair<uintptr_t, uintptr_t>>*) {}
void precollect_end() {}
void take_precollected_pages(const std::vector<std::pair<uintptr_t, uintptr_t>>& ranges, uint64_t, std::vector<uint8_t>* handled,
                             std::vector<uintptr_t>*) {
    handled->assign(ranges.size(), 0);
}

bool take_written_pages(const uint8_t*, uint64_t, uint64_t*, uint64_t*, bool, std::vector<uint64_t>*) {
    return false;  // not tracked on POSIX hosts
}

bool take_written_view_pages(const uint8_t*, uint64_t, uint64_t*, std::vector<uint64_t>*) {
    return false;
}

void write_positions(uint64_t*, uint64_t*) {}

uint64_t tracked_view_faults() {
    return 0;
}

void pin_views(uintptr_t, uintptr_t, bool) {}

void set_write_watch(bool) {}

// O36: shared pages are memfd / POSIX shared memory objects mapped MAP_SHARED | MAP_FIXED over
// reserved pages; any reserved range can hold them.
bool shared_pages_supported() {
    return true;
}

uint8_t* reserve_with_shared_zone(uint64_t size, uint64_t preferred, uint64_t) {
    return reserve(size, preferred);
}

void* create_shared_page(uint64_t size) {
    int fd = -1;
#if defined(__linux__)
    fd = static_cast<int>(syscall(SYS_memfd_create, "pg6-page", 0u));
#endif
    if (fd < 0) {
        char name[64];
        static std::atomic<unsigned> counter{0};
        std::snprintf(name, sizeof(name), "/pg6-page-%d-%u", static_cast<int>(getpid()), counter.fetch_add(1));
        fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) {
            shm_unlink(name);
        }
    }
    if (fd < 0) {
        return nullptr;
    }
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        close(fd);
        return nullptr;
    }
    return reinterpret_cast<void*>(static_cast<intptr_t>(fd) + 1);
}

void release_shared_page(void* page) {
    if (page != nullptr) {
        close(static_cast<int>(reinterpret_cast<intptr_t>(page) - 1));
    }
}

bool map_shared_page(uint8_t* at, void* page, uint64_t size) {
    if (page == nullptr) {
        return false;
    }
    const int fd = static_cast<int>(reinterpret_cast<intptr_t>(page) - 1);
    return mmap(at, static_cast<size_t>(size), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == at;
}

bool unmap_shared_page(uint8_t* at, uint64_t size) {
    return mmap(at, static_cast<size_t>(size), PROT_NONE, kReserveFlags | MAP_FIXED, -1, 0) == at;
}

uint8_t* view_shared_page(void* page, uint64_t size) {
    if (page == nullptr) {
        return nullptr;
    }
    const int fd = static_cast<int>(reinterpret_cast<intptr_t>(page) - 1);
    void* p = mmap(nullptr, static_cast<size_t>(size), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    return p == MAP_FAILED ? nullptr : static_cast<uint8_t*>(p);
}

void unview_shared_page(uint8_t* view, uint64_t size) {
    if (view != nullptr) {
        munmap(view, static_cast<size_t>(size));
    }
}

#endif

}  // namespace playgta6::os::backing
