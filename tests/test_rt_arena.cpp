// Unit tests for the paged RT address arena (gpu/rt_arena.h) and zero-alloc indirect/mesh
// clamp batch planner (gpu/indirect_clamp_batch.h).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu/cmd_encoder.h"
#include "gpu/indirect_clamp_batch.h"
#include "gpu/rt_arena.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_rt_arena.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
using namespace playgta6::gpu::rt_arena;

void test_overflow_and_edge_hardening() {
    Arena a(1, 8);
    // 40-bit overflow checks without uint64_t wraparound.
    CHECK(a.ensure_range(0xFFFFFFFF00ull, 0x200ull) == Ensure::OutOfRange);
    CHECK(a.ensure_range(0, UINT64_MAX) == Ensure::OutOfRange);
    CHECK(a.ensure_range(kAddressMask, 2) == Ensure::OutOfRange);
    CHECK(a.ensure_range(kAddressMask, 1) == Ensure::Uploaded);
    CHECK(a.stats().out_of_range == 3);

    // pin_structure rejects 40-bit overflow without disturbing existing pins.
    CHECK(a.pin_structure(1, 0, 2 * kPageBytes) == Ensure::Uploaded);
    CHECK(a.stats().pinned == 2);
    CHECK(a.pin_structure(2, 0xFFFFF00000ull, 0x200000ull) == Ensure::OutOfRange);
    CHECK(a.pin_structure(2, 0, UINT64_MAX) == Ensure::OutOfRange);
    CHECK(a.stats().pinned == 2);

    // mark_cpu_remapped and mark_gpu_written_range clamp safely at kAddressMask.
    a.mark_gpu_written_range(kAddressMask - 16, 1024);
    CHECK(a.stats().gpu_owned >= 1);
    a.mark_cpu_remapped(kAddressMask - 16, 1024);
    CHECK(a.ensure_page(page_of(kAddressMask)) == Ensure::Uploaded);

    // mark_gpu_written clears stale dirty bit; subsequent mark_cpu_dirty still increments skipped_gpu_owned.
    Arena b(1, 4);
    CHECK(b.ensure_page(10) == Ensure::Uploaded);
    b.take_uploads();
    b.mark_cpu_dirty(10);
    b.mark_gpu_written(10);
    CHECK(b.ensure_page(10) == Ensure::Resident);
    CHECK(b.stats().skipped_gpu_owned == 0);
    b.mark_cpu_dirty(10);
    CHECK(b.ensure_page(10) == Ensure::Resident);
    CHECK(b.stats().skipped_gpu_owned == 1);
}

void test_scratch_recycling_and_staging() {
    Arena a(2, 8);  // 16 pages total
    a.set_scratch_budget_pages(4);
    CHECK(a.scratch_budget_pages() == 4);

    ScratchAllocation s1{};
    CHECK(a.acquire_scratch(2 * kPageBytes, &s1) == Ensure::Uploaded);
    CHECK(s1.valid() && s1.page_count == 2 && s1.gpu_addr == kDefaultScratchBase);
    CHECK(a.stats().scratch_live_pages == 2 && a.stats().scratch_recycled_pages == 0);
    CHECK(a.stats().pinned == 2 && a.stats().gpu_owned == 2);

    // Exceeding budget fails cleanly.
    ScratchAllocation too_big{};
    CHECK(a.acquire_scratch(3 * kPageBytes, &too_big) == Ensure::ArenaFull);
    CHECK(!too_big.valid() && a.stats().scratch_budget_exceeded == 1);

    // Release and re-acquire: must recycle exact same pages with Ensure::Resident (zero uploads).
    a.take_uploads();
    a.release_scratch(s1);
    CHECK(a.stats().scratch_live_pages == 0 && a.stats().scratch_recycled_pages == 2);
    CHECK(a.stats().pinned == 0);

    ScratchAllocation s2{};
    CHECK(a.acquire_scratch(kPageBytes + 128, &s2) == Ensure::Resident);
    CHECK(s2.valid() && s2.first_page == s1.first_page && s2.gpu_addr == s1.gpu_addr);
    CHECK(a.take_uploads().empty());
    CHECK(a.stats().scratch_reuses == 1);

    // Release s2, then allocate a larger 3-page scratch block: must evict recycled s2 to stay in budget.
    a.release_scratch(s2);
    ScratchAllocation s3{};
    CHECK(a.acquire_scratch(3 * kPageBytes, &s3) == Ensure::Uploaded);
    CHECK(s3.valid() && s3.page_count == 3);
    CHECK(a.stats().scratch_evictions == 1);
    CHECK(a.stats().scratch_live_pages + a.stats().scratch_recycled_pages <= a.scratch_budget_pages());
    a.release_scratch(s3);

    // Zero-alloc take_uploads_into preserves caller vector capacity.
    a.take_uploads();
    std::vector<Upload> uploads;
    uploads.reserve(16);
    const Upload* up_ptr = uploads.data();
    a.reserve_upload_capacity(16);
    CHECK(a.ensure_page(3) == Ensure::Uploaded);
    a.take_uploads_into(uploads);
    CHECK(uploads.size() == 1 && uploads.data() == up_ptr);

    // stage_structure across a page boundary + simulated pg6_load32 readback.
    Arena st(2, 4);
    std::vector<uint32_t> backing[kArenaCount];
    const uint32_t* ptrs[kArenaCount] = {};
    for (uint32_t i = 0; i < 2; ++i) {
        backing[i].assign(4u * (kPageBytes / 4u), 0u);
        ptrs[i] = backing[i].data();
    }
    constexpr uint64_t kGpuAddr = 0x10000ull - 256ull;  // straddles page 0 -> page 1
    std::vector<uint32_t> payload(256, 0u);             // 1024 bytes
    for (uint32_t i = 0; i < payload.size(); ++i) payload[i] = 0xCA000000u | i;
    std::vector<StagedWrite> writes;
    writes.reserve(8);
    CHECK(st.stage_structure(77, kGpuAddr, payload.size() * 4u, true, &writes) == Ensure::Uploaded);
    CHECK(writes.size() == 2);
    CHECK(writes[0].byte_size == 256u && writes[1].byte_size == 768u);
    const auto* src = reinterpret_cast<const uint8_t*>(payload.data());
    for (const auto& w : writes) {
        std::memcpy(reinterpret_cast<uint8_t*>(backing[w.arena].data()) + w.arena_byte_offset,
                    src + w.src_byte_offset, w.byte_size);
    }
    uint64_t faults = 0;
    for (uint32_t i = 0; i < payload.size(); ++i) {
        CHECK(st.load32_simulated(kGpuAddr + 4ull * i, ptrs, &faults) == payload[i]);
    }
    CHECK(faults == 0);
    // Unmapped address increments fault counter and returns 0.
    CHECK(st.load32_simulated(0x55550000ull, ptrs, &faults) == 0u && faults == 1u);

    // JSON includes both legacy keys and new scratch/staging metrics.
    const std::string j = a.json();
    CHECK(j.find("\"scratch_reuses\": 1") != std::string::npos);
    CHECK(j.find("\"scratch_evictions\": 1") != std::string::npos);
    CHECK(j.find("\"scratch_budget_exceeded\": 1") != std::string::npos);
    CHECK(st.json().find("\"staged_structures\": 1") != std::string::npos);
}

void test_pack_instance_3x4() {
    Instance3x4 inst[2]{};
    inst[0].transform[0][0] = 2.0f;
    inst[0].transform[1][1] = 4.0f;
    inst[0].transform[2][2] = 0.5f;
    inst[0].transform[0][3] = 6.0f;
    inst[0].transform[1][3] = -8.0f;
    inst[0].transform[2][3] = 1.0f;
    inst[0].instance_id = 0x123456u;
    inst[0].instance_mask = 0xAAu;
    inst[0].hit_group_contribution = 0x654321u;
    inst[0].flags = 0x05u;
    inst[0].blas_address = 0x4000000000ull;  // 40-bit, 256-byte aligned
    inst[0].blas_root_node = 0x0000000Du;

    inst[1] = inst[0];
    inst[1].transform_is_inverse = true;
    inst[1].instance_id = 0x777u;

    uint32_t out[kInstance3x4Dwords * 2]{};
    CHECK(pack_instances_3x4(inst, 2, out, kInstance3x4Dwords * 2) == 2);

    float inv00 = 0.0f, inv03 = 0.0f, inv11 = 0.0f, inv13 = 0.0f, inv22 = 0.0f, inv23 = 0.0f;
    std::memcpy(&inv00, &out[0], 4);
    std::memcpy(&inv03, &out[3], 4);
    std::memcpy(&inv11, &out[5], 4);
    std::memcpy(&inv13, &out[7], 4);
    std::memcpy(&inv22, &out[10], 4);
    std::memcpy(&inv23, &out[11], 4);
    CHECK(std::fabs(inv00 - 0.5f) < 1e-5f && std::fabs(inv03 - (-3.0f)) < 1e-5f);
    CHECK(std::fabs(inv11 - 0.25f) < 1e-5f && std::fabs(inv13 - 2.0f) < 1e-5f);
    CHECK(std::fabs(inv22 - 2.0f) < 1e-5f && std::fabs(inv23 - (-2.0f)) < 1e-5f);
    CHECK(out[12] == (0x123456u | (0xAAu << 24)));
    CHECK(out[13] == (0x654321u | (0x05u << 24)));
    CHECK(out[14] == static_cast<uint32_t>(0x4000000000ull >> 8));
    CHECK(out[15] == 0x0000000Du);

    // Invalid address or singular matrix fails cleanly.
    Instance3x4 bad = inst[0];
    bad.blas_address = 0x10010ull;  // not 256-byte aligned
    CHECK(!pack_instance_3x4(bad, out));
    bad = inst[0];
    bad.transform[0][0] = 0.0f;
    CHECK(!pack_instance_3x4(bad, out));
}

void test_indirect_clamp_batch_zero_alloc_and_mesh() {
    using playgta6::GpuOpcode;
    using playgta6::gpu::CommandEncoder;
    using playgta6::gpu::bridge::IndirectBatchKind;
    using playgta6::gpu::bridge::IndirectClampBatch;
    using playgta6::gpu::bridge::kIndirectClampBatchSize;

    // 1. Multi-draw mixing Draw and DrawIndexed, rejecting mismatched kinds.
    IndirectClampBatch b;
    CommandEncoder enc;
    b.begin(0, 120, 0, 0, IndirectBatchKind::Draw);
    enc.draw_indexed(6, 1, 0, 0, 0);
    enc.indirect(GpuOpcode::DrawIndirect, 50, 0);
    enc.indirect(GpuOpcode::DrawIndexedIndirect, 50, 16);
    b.scan(enc.data(), enc.size());
    CHECK(b.open());
    CHECK(b.try_join(10, 0, 256, IndirectBatchKind::DrawIndexed) == 0);
    CHECK(b.try_join(11, 0, 512, IndirectBatchKind::Draw) == 120);
    CHECK(b.try_join(12, 0, 768, IndirectBatchKind::DispatchMesh) == IndirectClampBatch::kNoSlot);
    CHECK(b.stats().rejected_kind == 1 && b.open());

    // DispatchIndirect in a plain Draw run must close the batch.
    enc.indirect(GpuOpcode::DispatchIndirect, 50, 0);
    b.scan(enc.data(), enc.size());
    CHECK(!b.open() && b.stats().closed_barrier_opcode == 1);

    // 2. DispatchMesh (type-105) clamp batching: DispatchIndirect reading from clamp_ring keeps run open,
    // marks mesh_output_buffer written, and closes on unknown DispatchIndirect buffer.
    constexpr uint32_t kClampRing = 700;
    constexpr uint32_t kMeshOut = 701;
    CommandEncoder menc;
    b.reset_stats();
    b.begin(0, 128, 0, 0, IndirectBatchKind::DispatchMesh, kClampRing, kMeshOut);
    menc.set_compute_pipeline(9);
    menc.indirect(GpuOpcode::DispatchIndirect, kClampRing, 0);
    menc.indirect(GpuOpcode::DrawIndirect, kClampRing, 12);
    b.scan(menc.data(), menc.size());
    CHECK(b.open() && b.stats().mesh_dispatches_scanned == 1);
    CHECK(b.is_written(kMeshOut));
    // A clamp reading the mesh output buffer as args or count is rejected; independent args join!
    CHECK(b.try_join(kMeshOut, 0, 256, IndirectBatchKind::DispatchMesh) == IndirectClampBatch::kNoSlot);
    CHECK(b.try_join(800, kMeshOut, 256, IndirectBatchKind::DispatchMesh) == IndirectClampBatch::kNoSlot);
    CHECK(b.stats().rejected_written == 2 && b.open());
    CHECK(b.try_join(800, 801, 256, IndirectBatchKind::DispatchMesh) == 0);
    CHECK(b.try_join(802, 0, 512, IndirectBatchKind::DispatchMesh) == 128);
    CHECK(b.joined() == 2 && b.remaining_slots() == (kIndirectClampBatchSize - 1u - 2u));

    // DispatchIndirect from an unknown buffer closes the mesh run.
    menc.indirect(GpuOpcode::DispatchIndirect, 999, 0);
    b.scan(menc.data(), menc.size());
    CHECK(!b.open() && b.stats().closed_barrier_opcode == 1);

    // 3. Inline zero-alloc written table handles buffer id 0, max capacity, and graceful overflow -> close.
    b.reset_stats();
    b.begin(0, 120, 0, 0);
    CommandEncoder wenc;
    for (uint32_t i = 0; i < IndirectClampBatch::kWrittenMaxEntries; ++i) {
        wenc.write_buffer(i, 0, "abcd", 4);  // includes buffer id 0!
    }
    b.scan(wenc.data(), wenc.size());
    CHECK(b.open() && b.written_count() == IndirectClampBatch::kWrittenMaxEntries);
    CHECK(b.is_written(0u));
    CHECK(b.is_written(IndirectClampBatch::kWrittenMaxEntries - 1u));
    CHECK(!b.is_written(IndirectClampBatch::kWrittenMaxEntries));
    // Duplicate writes do not overflow.
    wenc.write_buffer(0u, 0, "efgh", 4);
    b.scan(wenc.data(), wenc.size());
    CHECK(b.open());
    // One more distinct buffer triggers graceful overflow -> close.
    wenc.write_buffer(IndirectClampBatch::kWrittenMaxEntries, 0, "ijkl", 4);
    b.scan(wenc.data(), wenc.size());
    CHECK(!b.open() && b.stats().closed_table_overflow == 1);

    // 4. Barrier opcodes (ReadbackBuffer, DestroyResource, CreateBuffer, ClearColor, Present) all close run.
    const GpuOpcode barrier_ops[] = {
        GpuOpcode::BeginFrame, GpuOpcode::ClearColor, GpuOpcode::CreateBuffer,
        GpuOpcode::Present, GpuOpcode::DestroyResource, GpuOpcode::DispatchCompute,
        GpuOpcode::ReadbackBuffer, GpuOpcode::ReadbackTexture,
    };
    for (GpuOpcode op : barrier_ops) {
        CommandEncoder oenc;
        const uint32_t dummy[4] = {1, 2, 3, 4};
        oenc.emit_packet(op, dummy, sizeof(dummy));
        b.begin(0, 120, 0, 0);
        b.scan(oenc.data(), oenc.size());
        CHECK(!b.open());
    }

    // 5. Explicit note_buffer_written & note_barrier + repeated run epoch cycling.
    b.begin(0, 120, 0, 0);
    CHECK(b.note_buffer_written(55));
    CHECK(b.try_join(55, 0, 128) == IndirectClampBatch::kNoSlot);
    b.note_barrier();
    CHECK(!b.open());
    for (uint32_t r = 0; r < 512; ++r) {
        b.begin(0, 64, 0, 0);
        CHECK(!b.is_written(55));
        CHECK(b.note_buffer_written(r));
        CHECK(b.is_written(r));
        CHECK(b.try_join(r + 1000u, 0, 64u) == 0u);
    }
}

}  // namespace

int main() {
    // Address -> page mapping and entry packing.
    CHECK(page_of(0x12345678ull) == 0x1234);
    CHECK(page_of(0xFFFFFFFFFFull) == 0xFFFFFFull);
    CHECK(page_of(0x1FFFFFFFFFFull) == 0xFFFFFFull);  // bits above 40 ignored
    CHECK(top_index(0xFFFFFFull) == 4095 && l2_index(0xFFFFFFull) == 4095);
    CHECK(top_index(0x1000) == 1 && l2_index(0x1000) == 0);
    CHECK(kPagesPerArena == 8192);
    for (uint32_t a = 0; a < 4; ++a) {
        const uint32_t e = pack_entry(a, 8191);
        CHECK(entry_valid(e) && entry_arena(e) == a && entry_slot(e) == 8191);
    }
    CHECK(pack_entry(0, 0) != 0);
    CHECK(kBindArena0 + 3 < kBindDirectory && kBindStats == 21);

    // Residency, directory packing, mapping.
    Arena a(2, 4);  // 8 pages
    CHECK(a.capacity_pages() == 8);
    CHECK(a.lookup_entry(0x500000) == 0);
    CHECK(a.ensure_page(0x50) == Ensure::Uploaded);
    CHECK(a.ensure_page(0x50) == Ensure::Resident);
    const uint32_t e = a.lookup_entry((0x50ull << 16) | 0x123);
    CHECK(entry_valid(e) && entry_arena(e) == 0 && entry_slot(e) == 0);
    CHECK(a.directory()[0] == 1);                   // top[0] -> table 1
    CHECK(a.directory()[kTopEntries + 0x50] == e);  // L2 word
    uint32_t f = 0, l = 0;
    CHECK(a.take_dir_dirty(&f, &l) && f == 0 && l == kTopEntries + 0x50);
    CHECK(!a.take_dir_dirty(&f, &l));
    CHECK(a.ensure_page(0xABCDEF) == Ensure::Uploaded);  // second table
    CHECK(a.directory().size() == kTopEntries + 2u * kL2Entries);
    CHECK(a.take_uploads().size() == 2);
    CHECK(a.take_uploads().empty());

    // Range, LRU eviction, epoch protection.
    Arena b(1, 4);
    b.begin_dispatch();
    CHECK(b.ensure_range(0, 4 * kPageBytes) == Ensure::Uploaded);
    CHECK(b.ensure_page(100) == Ensure::ArenaFull);  // all four protected by this dispatch
    b.begin_dispatch();
    b.begin_frame();
    CHECK(b.ensure_page(1) == Ensure::Resident);  // refresh page 1
    b.begin_dispatch();
    CHECK(b.ensure_page(100) == Ensure::Uploaded);  // evicts LRU victim
    CHECK(b.is_resident(1) && b.is_resident(100));
    CHECK(b.stats().evictions == 1);
    CHECK(b.lookup_entry(100 * kPageBytes) != 0);

    // Pinning keeps structure pages; arena full when pins fill it; release frees them.
    Arena c(1, 4);
    CHECK(c.pin_structure(7, 0, 3 * kPageBytes) == Ensure::Uploaded);
    c.begin_dispatch();
    CHECK(c.ensure_page(50) == Ensure::Uploaded);
    c.begin_dispatch();
    CHECK(c.ensure_page(51) == Ensure::Uploaded);  // evicts 50, not pinned 0..2
    CHECK(c.is_resident(0) && c.is_resident(2) && !c.is_resident(50));
    CHECK(c.pin_structure(8, 200 * kPageBytes, 2 * kPageBytes) == Ensure::ArenaFull);
    CHECK(c.stats().pinned == 3);  // failed pin rolled back
    c.release_structure(7);
    c.begin_dispatch();
    CHECK(c.ensure_page(60) == Ensure::Uploaded);
    CHECK(c.stats().pinned == 0);

    // GPU ownership: dirty marks do not re-upload until remapped.
    Arena d(1, 4);
    d.ensure_page(5);
    d.take_uploads();
    d.mark_gpu_written(5);
    d.mark_cpu_dirty(5);
    CHECK(d.ensure_page(5) == Ensure::Resident);
    CHECK(d.take_uploads().empty() && d.stats().skipped_gpu_owned == 1);
    d.mark_cpu_remapped(5 * kPageBytes, kPageBytes);
    CHECK(d.ensure_page(5) == Ensure::Uploaded && d.take_uploads().size() == 1);
    d.mark_cpu_dirty(5);
    CHECK(d.ensure_page(5) == Ensure::Uploaded);

    // Limits and counters.
    Arena g(4, 8192);
    CHECK(g.capacity_bytes() == (2048ull << 20));
    CHECK(g.ensure_page(1ull << 24) == Ensure::OutOfRange);
    Arena t(1, 100000);
    uint32_t ok = 0;
    for (uint32_t i = 0; i < kMaxL2Tables + 2; ++i) ok += t.ensure_page(static_cast<uint64_t>(i) << 12) == Ensure::Uploaded;
    CHECK(ok == kMaxL2Tables && t.stats().table_full == 2);
    CHECK((kTopEntries + kMaxL2Tables * kL2Entries) * 4ull < (1ull << 20));
    a.note_skip(SkipReason::RtOff);
    a.set_fault_count(9);
    const std::string j = a.json();
    CHECK(j.find("\"rt_off\": 1") != std::string::npos && j.find("\"rt_fault\": 9") != std::string::npos);

    test_overflow_and_edge_hardening();
    test_scratch_recycling_and_staging();
    test_pack_instance_3x4();
    test_indirect_clamp_batch_zero_alloc_and_mesh();

    if (g_failures == 0) std::printf("test_rt_arena: OK\n");
    return g_failures ? 1 : 0;
}
