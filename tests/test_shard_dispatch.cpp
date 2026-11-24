#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/shard_dispatch.h"

namespace {

// ---- ABI layout static checks ----
static_assert(kPg6ShardDispatchVersion == 1u);
static_assert(std::is_same_v<Pg6GuestState, playgta6::GuestState>);
static_assert(sizeof(Pg6ShardFunctionEntry) == 16);
static_assert(alignof(Pg6ShardFunctionEntry) == 8);
static_assert(offsetof(Pg6ShardFunctionEntry, guest_va) == 0);
static_assert(offsetof(Pg6ShardFunctionEntry, shard_idx) == 8);
static_assert(offsetof(Pg6ShardFunctionEntry, local_idx) == 12);
static_assert(sizeof(Pg6ShardDispatchTable) == 16 + 2 * sizeof(void*));
static_assert(sizeof(void*) != 8 || sizeof(Pg6ShardDispatchTable) == 32);
static_assert(offsetof(Pg6ShardDispatchTable, version) == 0);
static_assert(offsetof(Pg6ShardDispatchTable, shard_count) == 4);
static_assert(offsetof(Pg6ShardDispatchTable, entry_count) == 8);
static_assert(offsetof(Pg6ShardDispatchTable, shards) == 16);
static_assert(offsetof(Pg6ShardDispatchTable, entries) == 16 + sizeof(void*));

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_shard_dispatch.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace GuestReg = playgta6::GuestReg;
using playgta6::GuestState;
using playgta6::guest::ShardDispatchRegistry;

// Encodes (module_id, shard_id, local_idx) into state->gpr[Rax] so tests can verify
// exact dispatcher invocation across modules and concurrent threads without shared mutable state.
constexpr uint64_t encode_tag(uint32_t module_id, uint32_t shard_id, uint32_t local_idx) {
    return (static_cast<uint64_t>(module_id) << 48) |
           (static_cast<uint64_t>(shard_id) << 32) |
           static_cast<uint64_t>(local_idx);
}

void mod0_shard0_dispatch(uint32_t local_idx, Pg6GuestState* state) {
    if (state != nullptr) {
        state->gpr[GuestReg::Rax] = encode_tag(0, 0, local_idx);
    }
}

void mod0_shard1_dispatch(uint32_t local_idx, Pg6GuestState* state) {
    if (state != nullptr) {
        state->gpr[GuestReg::Rax] = encode_tag(0, 1, local_idx);
    }
}

void mod1_shard0_dispatch(uint32_t local_idx, Pg6GuestState* state) {
    if (state != nullptr) {
        state->gpr[GuestReg::Rax] = encode_tag(1, 0, local_idx);
    }
}

void mod1_shard1_dispatch(uint32_t local_idx, Pg6GuestState* state) {
    if (state != nullptr) {
        state->gpr[GuestReg::Rax] = encode_tag(1, 1, local_idx);
    }
}

void test_single_module_and_unsorted_entries() {
    ShardDispatchRegistry registry;
    CHECK(registry.entry_count() == 0);
    CHECK(registry.shard_count() == 0);

    const Pg6ShardDispatchFn shards[] = {
        &mod0_shard0_dispatch,
        &mod0_shard1_dispatch,
    };
    // Intentionally unsorted guest_va order.
    const Pg6ShardFunctionEntry entries[] = {
        {0x140003000ULL, 1u, 42u},
        {0x140001000ULL, 0u, 7u},
        {0x140002000ULL, 1u, 0u},
        {0x140001500ULL, 0u, 19u},
    };
    const Pg6ShardDispatchTable table{
        kPg6ShardDispatchVersion,
        2u,
        4u,
        shards,
        entries,
    };

    std::string err = "stale error";
    CHECK(registry.add_table(table, &err));
    CHECK(err.empty());
    CHECK(registry.shard_count() == 2);
    CHECK(registry.entry_count() == 4);

    ShardDispatchRegistry::ResolvedShardTarget target{};
    CHECK(registry.lookup(0x140001000ULL, target));
    CHECK(target.dispatcher == &mod0_shard0_dispatch);
    CHECK(target.local_idx == 7u);

    CHECK(registry.lookup(0x140001500ULL, target));
    CHECK(target.dispatcher == &mod0_shard0_dispatch);
    CHECK(target.local_idx == 19u);

    CHECK(registry.lookup(0x140002000ULL, target));
    CHECK(target.dispatcher == &mod0_shard1_dispatch);
    CHECK(target.local_idx == 0u);

    CHECK(registry.lookup(0x140003000ULL, target));
    CHECK(target.dispatcher == &mod0_shard1_dispatch);
    CHECK(target.local_idx == 42u);

    GuestState state{};
    CHECK(registry.dispatch(0x140003000ULL, &state));
    CHECK(state.gpr[GuestReg::Rax] == encode_tag(0, 1, 42u));

    CHECK(registry.dispatch(0x140001000ULL, &state));
    CHECK(state.gpr[GuestReg::Rax] == encode_tag(0, 0, 7u));

    // Missing VAs (below min, between entries, above max) return false and clear `target`.
    for (uint64_t missing_va : {0x0ULL, 0x140000FFFULL, 0x140001234ULL, 0x140004000ULL}) {
        target.dispatcher = &mod0_shard0_dispatch;
        target.local_idx = 999u;
        CHECK(!registry.lookup(missing_va, target));
        CHECK(target.dispatcher == nullptr);
        CHECK(target.local_idx == 0u);

        state.gpr[GuestReg::Rax] = 0xDEADBEEFULL;
        CHECK(!registry.dispatch(missing_va, &state));
        CHECK(state.gpr[GuestReg::Rax] == 0xDEADBEEFULL);
    }
}

void test_multi_module_shard_idx_remapping() {
    ShardDispatchRegistry registry;

    // Module 0 (e.g. game.exe): local shard_idx 0 and 1.
    const Pg6ShardDispatchFn mod0_shards[] = {
        &mod0_shard0_dispatch,
        &mod0_shard1_dispatch,
    };
    const Pg6ShardFunctionEntry mod0_entries[] = {
        {0x140002000ULL, 0u, 10u},
        {0x140005000ULL, 1u, 20u},
    };
    const Pg6ShardDispatchTable mod0_table{
        kPg6ShardDispatchVersion,
        2u,
        2u,
        mod0_shards,
        mod0_entries,
    };

    // Module 1 (e.g. vcruntime140.dll): overlapping local shard_idx 0 and 1,
    // with interleaved VAs below, between, and above Module 0's VAs.
    const Pg6ShardDispatchFn mod1_shards[] = {
        &mod1_shard0_dispatch,
        &mod1_shard1_dispatch,
    };
    const Pg6ShardFunctionEntry mod1_entries[] = {
        {0x180001000ULL, 1u, 300u},
        {0x140003000ULL, 0u, 100u},
        {0x130001000ULL, 0u, 50u},
    };
    const Pg6ShardDispatchTable mod1_table{
        kPg6ShardDispatchVersion,
        2u,
        3u,
        mod1_shards,
        mod1_entries,
    };

    std::string err;
    CHECK(registry.add_table(mod0_table, &err));
    CHECK(registry.add_table(mod1_table, &err));
    CHECK(registry.shard_count() == 4);
    CHECK(registry.entry_count() == 5);

    struct Expect {
        uint64_t va;
        Pg6ShardDispatchFn fn;
        uint32_t module_id;
        uint32_t shard_id;
        uint32_t local_idx;
    };
    const Expect expected[] = {
        {0x130001000ULL, &mod1_shard0_dispatch, 1, 0, 50u},
        {0x140002000ULL, &mod0_shard0_dispatch, 0, 0, 10u},
        {0x140003000ULL, &mod1_shard0_dispatch, 1, 0, 100u},
        {0x140005000ULL, &mod0_shard1_dispatch, 0, 1, 20u},
        {0x180001000ULL, &mod1_shard1_dispatch, 1, 1, 300u},
    };

    for (const Expect& e : expected) {
        ShardDispatchRegistry::ResolvedShardTarget target{};
        CHECK(registry.lookup(e.va, target));
        CHECK(target.dispatcher == e.fn);
        CHECK(target.local_idx == e.local_idx);

        GuestState state{};
        CHECK(registry.dispatch(e.va, &state));
        CHECK(state.gpr[GuestReg::Rax] == encode_tag(e.module_id, e.shard_id, e.local_idx));
    }

    // Empty table is a valid no-op.
    const Pg6ShardDispatchTable empty_table{kPg6ShardDispatchVersion, 0u, 0u, nullptr, nullptr};
    CHECK(registry.add_table(empty_table, &err));
    CHECK(registry.shard_count() == 4);
    CHECK(registry.entry_count() == 5);
}

void test_validation_rejects_malformed_without_corruption() {
    ShardDispatchRegistry registry;

    const Pg6ShardDispatchFn base_shards[] = {&mod0_shard0_dispatch};
    const Pg6ShardFunctionEntry base_entries[] = {{0x140010000ULL, 0u, 3u}};
    const Pg6ShardDispatchTable base_table{
        kPg6ShardDispatchVersion,
        1u,
        1u,
        base_shards,
        base_entries,
    };
    CHECK(registry.add_table(base_table));

    auto verify_intact = [&]() {
        CHECK(registry.shard_count() == 1);
        CHECK(registry.entry_count() == 1);
        GuestState state{};
        CHECK(registry.dispatch(0x140010000ULL, &state));
        CHECK(state.gpr[GuestReg::Rax] == encode_tag(0, 0, 3u));
    };

    // 1. Bad version.
    {
        std::string err;
        Pg6ShardDispatchTable bad = base_table;
        bad.version = kPg6ShardDispatchVersion + 1u;
        CHECK(!registry.add_table(bad, &err));
        CHECK(!err.empty());
        verify_intact();
    }

    // 2. Null shards with shard_count > 0.
    {
        std::string err;
        const Pg6ShardFunctionEntry entries[] = {{0x140020000ULL, 0u, 1u}};
        const Pg6ShardDispatchTable bad{kPg6ShardDispatchVersion, 1u, 1u, nullptr, entries};
        CHECK(!registry.add_table(bad, &err));
        CHECK(!err.empty());
        verify_intact();
    }

    // 3. Null entries with entry_count > 0.
    {
        std::string err;
        const Pg6ShardDispatchTable bad{kPg6ShardDispatchVersion, 1u, 1u, base_shards, nullptr};
        CHECK(!registry.add_table(bad, &err));
        CHECK(!err.empty());
        verify_intact();
    }

    // 4. Out-of-range shard_idx.
    {
        std::string err;
        const Pg6ShardFunctionEntry entries[] = {{0x140020000ULL, 1u, 0u}};
        const Pg6ShardDispatchTable bad{kPg6ShardDispatchVersion, 1u, 1u, base_shards, entries};
        CHECK(!registry.add_table(bad, &err));
        CHECK(!err.empty());
        verify_intact();
    }

    // 5. Entries present when shard_count == 0.
    {
        std::string err;
        const Pg6ShardFunctionEntry entries[] = {{0x140020000ULL, 0u, 0u}};
        const Pg6ShardDispatchTable bad{kPg6ShardDispatchVersion, 0u, 1u, nullptr, entries};
        CHECK(!registry.add_table(bad, &err));
        CHECK(!err.empty());
        verify_intact();
    }

    // 6. Null dispatcher in shards array.
    {
        std::string err;
        const Pg6ShardDispatchFn null_shards[] = {&mod1_shard0_dispatch, nullptr};
        const Pg6ShardFunctionEntry entries[] = {{0x140020000ULL, 1u, 0u}};
        const Pg6ShardDispatchTable bad{kPg6ShardDispatchVersion, 2u, 1u, null_shards, entries};
        CHECK(!registry.add_table(bad, &err));
        CHECK(!err.empty());
        verify_intact();
    }

    // 7. Duplicate guest_va within one incoming table.
    {
        std::string err;
        const Pg6ShardFunctionEntry dup_entries[] = {
            {0x140030000ULL, 0u, 1u},
            {0x140030000ULL, 0u, 2u},
        };
        const Pg6ShardDispatchTable bad{kPg6ShardDispatchVersion, 1u, 2u, base_shards, dup_entries};
        CHECK(!registry.add_table(bad, &err));
        CHECK(!err.empty());
        verify_intact();
    }

    // 8. Duplicate guest_va colliding with an already registered entry.
    {
        std::string err;
        const Pg6ShardFunctionEntry dup_entries[] = {
            {0x140020000ULL, 0u, 1u},
            {0x140010000ULL, 0u, 9u},
        };
        const Pg6ShardDispatchTable bad{kPg6ShardDispatchVersion, 1u, 2u, base_shards, dup_entries};
        CHECK(!registry.add_table(bad, &err));
        CHECK(!err.empty());
        verify_intact();
        ShardDispatchRegistry::ResolvedShardTarget target{};
        CHECK(!registry.lookup(0x140020000ULL, target));
    }

    // 9. clear() resets cleanly and allows re-registration of the same VA.
    registry.clear();
    CHECK(registry.shard_count() == 0);
    CHECK(registry.entry_count() == 0);
    ShardDispatchRegistry::ResolvedShardTarget target{};
    CHECK(!registry.lookup(0x140010000ULL, target));
    CHECK(registry.add_table(base_table));
    verify_intact();
}

void test_concurrent_readonly_dispatch() {
    ShardDispatchRegistry registry;
    const Pg6ShardDispatchFn shards[] = {&mod0_shard0_dispatch, &mod0_shard1_dispatch};
    std::vector<Pg6ShardFunctionEntry> entries;
    constexpr uint32_t kCount = 256;
    entries.reserve(kCount);
    for (uint32_t i = 0; i < kCount; ++i) {
        entries.push_back({0x140100000ULL + static_cast<uint64_t>(i) * 16u, i & 1u, i});
    }
    const Pg6ShardDispatchTable table{
        kPg6ShardDispatchVersion,
        2u,
        entries.size(),
        shards,
        entries.data(),
    };
    CHECK(registry.add_table(table));

    std::atomic<bool> ok{true};
    std::vector<std::thread> threads;
    constexpr int kThreads = 8;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            GuestState state{};
            for (int iter = 0; iter < 50; ++iter) {
                for (uint32_t i = 0; i < kCount; ++i) {
                    const uint32_t idx = (i + static_cast<uint32_t>(t * 17)) % kCount;
                    const uint64_t va = 0x140100000ULL + static_cast<uint64_t>(idx) * 16u;
                    if (!registry.dispatch(va, &state) ||
                        state.gpr[GuestReg::Rax] != encode_tag(0, idx & 1u, idx)) {
                        ok.store(false, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    CHECK(ok.load());
}

}  // namespace

int main() {
    test_single_module_and_unsorted_entries();
    test_multi_module_shard_idx_remapping();
    test_validation_rejects_malformed_without_corruption();
    test_concurrent_readonly_dispatch();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_shard_dispatch: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_shard_dispatch: all checks passed\n");
    return 0;
}
