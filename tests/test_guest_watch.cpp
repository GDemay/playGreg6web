#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "guest/guest_watch.h"
#include "os/vmem.h"

namespace {
int g_failures = 0;
#define CHECK(expr)                                                                        \
    do {                                                                                   \
        if (!(expr)) {                                                                     \
            std::fprintf(stderr, "FAIL [test_guest_watch.cpp:%d]: %s\n", __LINE__, #expr); \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

using namespace playgta6::guest;
namespace os = playgta6::os;

void test_parse() {
    WatchRange r;
    CHECK(parse_watch_spec("1477ae348", &r) && r.va == 0x1477ae348ull && r.len == 8);
    CHECK(parse_watch_spec("0x1477AE310:0x40", &r) && r.va == 0x1477ae310ull && r.len == 0x40);
    CHECK(parse_watch_spec("0x1000:16", &r) && r.va == 0x1000 && r.len == 16);
    CHECK(!parse_watch_spec("", &r));
    CHECK(!parse_watch_spec("0", &r));
    CHECK(!parse_watch_spec("0x1000:0", &r));
    CHECK(!parse_watch_spec("0x1000:4097", &r));
    CHECK(!parse_watch_spec("xyz", &r));
    CHECK(!parse_watch_spec("0x1000:", &r));
    CHECK(!parse_watch_spec("ffffffffffffffff:8", &r));
    CHECK(!parse_watch_spec("12345678901234567", &r));
    CHECK(parse_watch_spec("*0x149844898+0x608:16", &r) && r.indirect && r.va == 0x149844898ull &&
          r.offset == 0x608 && r.len == 16);
    CHECK(parse_watch_spec("*1000", &r) && r.indirect && r.offset == 0 && r.len == 8);
    CHECK(!parse_watch_spec("*0x1000+", &r) && !parse_watch_spec("0x1000+8", &r));
}

void test_overlap_and_snapshot() {
    GuestWatch w;
    CHECK(!w.active() && !w.overlaps(0x1000, 0x1000));
    w.configure({{0x2008, 8}, {0x1000, 4}});
    CHECK(w.active() && w.ranges().size() == 2 && w.ranges()[0].va == 0x1000);
    CHECK(w.overlaps(0x1003, 1) && !w.overlaps(0x1004, 0x1000));
    CHECK(w.overlaps(0x2000, 0x9) && !w.overlaps(0x2000, 0x8) && !w.overlaps(0x2010, 0x100));
    CHECK(w.overlaps(0x0, 0x10000) && w.overlaps(0x200f, 0));  // len 0: one byte
    CHECK(w.overlaps(0xfff, 2) && !w.overlaps(0xffff'ffff'ffff'fff0ull, 0x100));

    const std::string s = w.snapshot([](uint64_t va, void* dst, size_t len) {
        if (va == 0x1000) return false;
        uint8_t b[8] = {1, 0, 0, 0, 0, 0, 0, 0x80};
        std::memcpy(dst, b, len);
        return true;
    });
    CHECK(s == " [0x1000:4]=unreadable [0x2008:8]=8000000000000001");

    // Indirect: the pointer at 0x3000 is followed at snapshot time; never matched by overlaps().
    w.configure({{0x3000, 8, true, 0x8}});
    CHECK(!w.overlaps(0x3000, 8));
    const std::string ind = w.snapshot([](uint64_t va, void* dst, size_t len) {
        uint64_t v = va == 0x3000 ? 0x5000 : (va == 0x5008 ? 0x2a : 0);
        if (va != 0x3000 && va != 0x5008) return false;
        std::memcpy(dst, &v, len < 8 ? len : 8);
        return true;
    });
    CHECK(ind == " [*0x3000+0x8:8]@0x5008=000000000000002a");
    const std::string null_ptr = w.snapshot([](uint64_t, void* dst, size_t len) {
        std::memset(dst, 0, len);
        return true;
    });
    CHECK(null_ptr == " [*0x3000+0x8:8]=null");

    w.configure({{0x10, 10}});
    const std::string t = w.snapshot([](uint64_t, void* dst, size_t len) {
        std::memset(dst, 0xab, len);
        return true;
    });
    CHECK(t == " [0x10:10]=abababababababab,abab");

    w.configure({{0x10, 8}}, 2);
    CHECK(w.take_line() && w.take_line() && !w.take_line() && w.lines() == 3);

    int first = 0;
    for (uint64_t k = 0; k < GuestWatch::kEntriesPerFunction + 5; ++k) {
        first += w.take_entry(0x140001000) ? 1 : 0;
    }
    CHECK(first == static_cast<int>(GuestWatch::kEntriesPerFunction));
    CHECK(w.take_entry(0x140002000) && !w.take_entry(0));
    w.configure({{0x10, 8}});  // a new configuration starts the counts afresh
    CHECK(w.take_entry(0x140001000));
}

// The arena reports what a watch needs from the runtime side: MEM_RESET and checked writes, next
// to commit / decommit / release.
struct Seen {
    os::ArenaEvent event;
    uint64_t va;
    uint64_t size;
    bool ok;
};
std::vector<Seen> g_seen;

void record(const os::VirtualMemoryArena&, os::ArenaEvent e, uint64_t va, uint64_t size, os::PageProtect, bool ok) {
    g_seen.push_back(Seen{e, va, size, ok});
}

bool saw(os::ArenaEvent e, uint64_t va, uint64_t size, bool ok) {
    for (const Seen& s : g_seen) {
        if (s.event == e && s.va == va && s.size == size && s.ok == ok) return true;
    }
    return false;
}

void test_arena_events() {
    os::VirtualMemoryArena arena;
    if (!arena.valid()) {
        std::fprintf(stderr, "skip arena events: no backing\n");
        return;
    }
    const os::ArenaObserver previous = os::set_arena_observer(&record);
    const uint64_t base = arena.virtual_alloc(0, 0x10000, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
    CHECK(base != 0);
    const uint32_t v = 0x12345678;
    CHECK(arena.write(base + 0x10, &v, sizeof(v)));
    CHECK(saw(os::ArenaEvent::Write, base + 0x10, 4, true));
    CHECK(arena.virtual_alloc(base + 0x1000, 0x1000, os::kMemReset, os::PageProtect::ReadWrite) == base + 0x1000);
    CHECK(saw(os::ArenaEvent::Reset, base + 0x1000, 0x1000, true));
    CHECK(arena.virtual_free(base + 0x2000, 0x1000, os::kMemDecommit));
    CHECK(saw(os::ArenaEvent::Decommit, base + 0x2000, 0x1000, true));
    CHECK(!arena.write(base + 0x2000, &v, sizeof(v)));  // decommitted: refused and reported
    CHECK(saw(os::ArenaEvent::Write, base + 0x2000, 4, false));
    CHECK(arena.virtual_free(base, 0, os::kMemRelease));
    CHECK(saw(os::ArenaEvent::Release, base, 0, true));
    os::set_arena_observer(previous);
}

}  // namespace

int main() {
    test_parse();
    test_overlap_and_snapshot();
    test_arena_events();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_guest_watch: ok\n");
    return 0;
}
