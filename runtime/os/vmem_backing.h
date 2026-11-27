#pragma once

#include <cstdint>
#include <vector>

#include "os/vmem.h"

// Internal to VirtualMemoryArena (vmem.cpp): the per-platform host backing implemented in
// vmem_backing.cpp. All ranges are page aligned and lie inside one reservation.
namespace playgta6::os::backing {

// Reserves `size` bytes of inaccessible address space; with `preferred` != 0 only exactly at
// that host address. nullptr on failure.
uint8_t* reserve(uint64_t size, uint64_t preferred);
void release(uint8_t* base, uint64_t size);
// Makes reserved (never or no longer committed) pages usable; they read as zero.
bool commit(uint8_t* p, uint64_t len, PageProtect protect);
// Gives committed pages back (where the platform can); the range may include uncommitted pages.
bool decommit(uint8_t* p, uint64_t len);
// Hints the OS that committed pages' contents are no longer needed (MEM_RESET / MADV_DONTNEED)
// while preserving committed state and protection.
bool reset_pages(uint8_t* p, uint64_t len);
// Reclaims pages previously marked with reset_pages if still resident (MEM_RESET_UNDO).
bool reset_undo_pages(uint8_t* p, uint64_t len);
// Applies a protection to committed pages (no-op where protections cannot be enforced).
bool protect(uint8_t* p, uint64_t len, PageProtect protect);
bool enforces_protection();

// #1077: Windows lazy commit (commit-on-first-touch via leaf-locked VEH). When enabled (default
// on Windows, false elsewhere), commit() calls >= 64 KiB defer OS MEM_COMMIT until first touch or
// explicit materialization, avoiding private commit charge for untouched zero pages.
void set_lazy_commit(bool on);
bool lazy_commit_enabled();
uint64_t lazy_deferred_bytes();
uint64_t lazy_deferred_bytes_in_range(const uint8_t* base, uint64_t size);
uint64_t lazy_commit_faults();
// Copies [src, src + len) into dst without materializing unmaterialized lazy pages (fills zeros).
// When `try_lock_only` is true (DiagnosticReads, #1916), uses bounded try-locks and returns false
// on contention instead of blocking.
bool read_backing(const uint8_t* src, void* dst, size_t len, bool try_lock_only = false);
// Materializes any lazy pages in [dst, dst + len) before a bulk write (avoids per-chunk faults).
bool materialize_for_write(uint8_t* dst, size_t len);
// Zeroes [dst, dst + len), skipping unmaterialized lazy pages (already logically zero).
void zero_backing(uint8_t* dst, size_t len);
// Returns the mapped shared-view size starting at `at`, or 0 if not a tracked view head.
uint64_t shared_view_size(const uint8_t* at);
// See playgta6::os::take_written_pages (vmem.h). Windows reservations carry MEM_WRITE_WATCH and
// shared page views are tracked by write protection (R40); other platforms return false.
// `view_clock`: the caller's position for shared page views (null: report them all written).
// `watch_epoch`: the caller's position in the MEM_WRITE_WATCH fan-out (it is told every page
// written since, whoever consumed the OS bits); null: only the pages this call consumed.
bool take_written_pages(const uint8_t* p, uint64_t len, uint64_t* view_clock, uint64_t* watch_epoch,
                        bool peek, std::vector<uint64_t>* offsets);
// Like take_written_pages but only for ranges wholly covered by tracked shared page views: it never
// reads MEM_WRITE_WATCH (a destructive take), and returns false (consuming nothing) otherwise.
bool take_written_view_pages(const uint8_t* p, uint64_t len, uint64_t* view_clock, std::vector<uint64_t>* offsets);
// #1963: the current shared-view clock and MEM_WRITE_WATCH epoch: a cursor set to them before a batch
// of takes is told, by its next takes, every write stamped after (unchanged where not tracked).
void write_positions(uint64_t* view_clock, uint64_t* watch_epoch);
// Batch take (flush): collects the MEM_WRITE_WATCH bits of the merged `ranges` ([lo, hi) host
// addresses) once, stamping them for every reader, and lets this thread's takes inside a collected
// range skip their own GetWriteWatch call until precollect_end. Takes of other ranges are unchanged.
void precollect_begin(std::vector<std::pair<uintptr_t, uintptr_t>>* ranges);
void precollect_end();
// Inside a precollect: serves the takes of the sorted `ranges` that need no GetWriteWatch of their
// own (no shared page view, wholly inside a collected range) in one pass, under each lock once.
// handled[i] = 1: range i's pages stamped after `watch_epoch` were appended to `pages` (host
// addresses), exactly what take_written_pages would report for it; 0: take it as usual.
void take_precollected_pages(const std::vector<std::pair<uintptr_t, uintptr_t>>& ranges, uint64_t watch_epoch,
                             std::vector<uint8_t>* handled, std::vector<uintptr_t>* pages);
// Whether later reserve() calls ask for write tracking (where the platform has it).
void set_write_watch(bool on);
// R40: writes caught on armed shared page views (Windows; 0 elsewhere).
uint64_t tracked_view_faults();
// R40: pins (or unpins) the shared page views overlapping [lo, hi) for a host write: a pinned
// page is never armed and is reported written (see os::HostWriteScope).
void pin_views(uintptr_t lo, uintptr_t hi, bool pin);

// O36: shared pages - memory objects one process can see at several addresses at once (the
// console's physical pages mapped at several virtual addresses). Native hosts only.
bool shared_pages_supported();
// Like reserve(), with the part [zone_offset, size) able to hold shared page views (Windows: kept
// as placeholders; POSIX: any reserved range can). zone_offset == size: no zone.
uint8_t* reserve_with_shared_zone(uint64_t size, uint64_t preferred, uint64_t zone_offset);
// A zero-filled shared page of `size` bytes (a multiple of 64 KiB); nullptr on failure.
void* create_shared_page(uint64_t size);
// Drops the handle; views keep the memory alive until they are unmapped.
void release_shared_page(void* page);
// Shows `page` read/write at `at` (64 KiB aligned, inside a shared zone), replacing its reserved
// pages; unmap_shared_page makes them reserved again.
bool map_shared_page(uint8_t* at, void* page, uint64_t size);
bool unmap_shared_page(uint8_t* at, uint64_t size);
// A view of `page` at an address the OS picks (the runtime's own access to the page's bytes).
uint8_t* view_shared_page(void* page, uint64_t size);
void unview_shared_page(uint8_t* view, uint64_t size);

}  // namespace playgta6::os::backing
