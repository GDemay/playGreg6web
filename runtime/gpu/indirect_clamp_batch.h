// Batching of ExecuteIndirect clamp passes (backlog 41, G80).
//
// Every ExecuteIndirect needs its argument records clamped by a tiny compute pass before the draws
// or mesh dispatches read them. Running that pass right before each draw alternates compute and
// render work, which costs a pass break and a GPU pipeline bubble per draw. The bridge instead
// reserves, at the point where the first clamp of a run is recorded, a fixed number of patchable
// slots right behind it; the clamps of later executions in the same list are written into those
// slots (earlier in the stream) instead of ahead of their own draws. A clamp may move earlier only
// when nothing recorded in between can have changed the buffers it reads, so the clamp still sees
// exactly the bytes it would have seen right before its draw and every result stays identical.
//
// This class is the pure planning half: it tracks one open run, scans the packets recorded since
// the run's reservation and decides whether a new execution may join. It uses a zero-allocation
// fixed-capacity generation-stamped open-addressed table for written buffer tracking (with graceful
// overflow -> close fallback) and supports multi-draw (Draw / DrawIndexed) as well as mesh-dispatch
// (type-105 / DispatchMesh / DispatchIndirect) clamp batching.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "abi/playgta6_abi.h"

namespace playgta6::gpu::bridge {

// Clamps per run, the inline one included (the reference's default run length).
inline constexpr uint32_t kIndirectClampBatchSize = 7;

// Execution kind tracked by a clamp batch run (mirrors IndirectKind without pulling state_bridge.h).
enum class IndirectBatchKind : uint8_t {
    Draw = 0,
    DrawIndexed = 1,
    Dispatch = 2,
    DispatchMesh = 3,  // Xbox type-105 / DispatchMesh (DispatchIndirect + DrawIndirect pair)
};

struct IndirectClampBatchStats {
    uint64_t runs_opened = 0;
    uint64_t slots_joined = 0;
    uint64_t rejected_written = 0;
    uint64_t rejected_full = 0;
    uint64_t rejected_kind = 0;
    uint64_t closed_window_wrap = 0;
    uint64_t closed_barrier_opcode = 0;
    uint64_t closed_table_overflow = 0;
    uint64_t mesh_dispatches_scanned = 0;
};

class IndirectClampBatch {
public:
    static constexpr size_t kNoSlot = ~size_t{0};
    // Fixed inline open-addressed hash table sizing (power of two, <= 75% load factor).
    static constexpr size_t kWrittenTableCapacity = 128;
    static constexpr uint32_t kWrittenMaxEntries = 96;

    bool open() const { return open_; }

    // Legacy 4-argument begin(): starts a render multi-draw run with kIndirectClampBatchSize - 1
    // patchable slots. Strictly zero heap allocations.
    void begin(size_t reserve_offset, size_t slot_bytes, size_t scanned_to, uint32_t window) {
        begin(reserve_offset, slot_bytes, scanned_to, window, IndirectBatchKind::Draw, 0u, 0u,
              kIndirectClampBatchSize - 1u);
    }

    // Extended begin(): supports multi-draw and DispatchMesh (type-105) runs.
    // For DispatchMesh runs, `clamp_ring_buffer` (when non-zero) identifies the ring buffer read by
    // the mesh stage's DispatchIndirect packets, and `mesh_output_buffer` (when non-zero) is marked
    // written whenever a mesh DispatchIndirect is scanned so no later clamp may read it earlier.
    void begin(size_t reserve_offset, size_t slot_bytes, size_t scanned_to, uint32_t window,
               IndirectBatchKind kind, uint32_t clamp_ring_buffer = 0u,
               uint32_t mesh_output_buffer = 0u,
               uint32_t max_join_slots = kIndirectClampBatchSize - 1u) {
        open_ = true;
        reserve_ = reserve_offset;
        slot_bytes_ = slot_bytes;
        scanned_ = scanned_to;
        used_ = 0u;
        max_join_slots_ = max_join_slots;
        last_window_ = window;
        kind_ = kind;
        clamp_ring_buffer_ = clamp_ring_buffer;
        mesh_output_buffer_ = mesh_output_buffer;
        clear_written();
        ++stats_.runs_opened;
    }

    void close() {
        open_ = false;
        clear_written();
    }

    uint32_t joined() const { return used_; }
    uint32_t remaining_slots() const { return open_ && used_ < max_join_slots_ ? (max_join_slots_ - used_) : 0u; }
    IndirectBatchKind kind() const { return kind_; }
    uint32_t written_count() const { return written_count_; }
    const IndirectClampBatchStats& stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }

    bool is_written(uint32_t id) const {
        if (written_count_ == 0u) return false;
        size_t idx = hash_u32(id) & (kWrittenTableCapacity - 1u);
        for (size_t probe = 0; probe < kWrittenTableCapacity; ++probe) {
            if (written_epochs_[idx] != written_epoch_) {
                return false;
            }
            if (written_keys_[idx] == id) {
                return true;
            }
            idx = (idx + 1u) & (kWrittenTableCapacity - 1u);
        }
        return false;
    }

    // Records an out-of-band buffer write (e.g., copy/clear/UAV transition). Closes gracefully on
    // inline table overflow so correctness is always preserved with zero heap allocations.
    bool note_buffer_written(uint32_t id) {
        if (!open_) return false;
        if (!insert_written(id)) {
            ++stats_.closed_table_overflow;
            close();
            return false;
        }
        return true;
    }

    // Explicit memory/UAV ordering barrier closes any open run.
    void note_barrier() {
        if (open_) {
            ++stats_.closed_barrier_opcode;
            close();
        }
    }

    // Walks the packets recorded since the last scan. Any packet that can write a buffer other than
    // by a plain WriteBuffer (or a recognized mesh-stage DispatchIndirect in a DispatchMesh run), or
    // that we do not recognise, closes the run.
    void scan(const uint8_t* data, size_t size) {
        if (!open_) {
            return;
        }
        if (data == nullptr || size < scanned_) {  // the encoder was reset under us
            close();
            return;
        }
        while (open_) {
            const size_t remaining = size - scanned_;
            if (remaining < sizeof(GpuPacketHeader)) {
                break;
            }
            GpuPacketHeader h{};
            std::memcpy(&h, data + scanned_, sizeof(h));
            if (h.payload_bytes > remaining - sizeof(GpuPacketHeader)) {
                break;
            }
            const size_t end = scanned_ + sizeof(h) + static_cast<size_t>(h.payload_bytes);
            note_packet(static_cast<GpuOpcode>(h.opcode), data + scanned_ + sizeof(h), h.payload_bytes);
            scanned_ = end;
        }
    }

    // May a clamp reading `args` / `count` (0 = none) writing `window` move into the next slot?
    // Returns the slot's encoder offset and claims it, or kNoSlot (the run stays open unless the
    // window wrapped, which also ends it). Call scan() first.
    size_t try_join(uint32_t args, uint32_t count, uint32_t window) {
        return try_join(args, count, window, kind_);
    }

    // Kind-aware try_join() verifying multi-draw (Draw/DrawIndexed) or DispatchMesh compatibility.
    size_t try_join(uint32_t args, uint32_t count, uint32_t window, IndirectBatchKind join_kind) {
        if (!open_) {
            return kNoSlot;
        }
        if (used_ >= max_join_slots_) {
            ++stats_.rejected_full;
            return kNoSlot;
        }
        if (!kinds_compatible(kind_, join_kind)) {
            ++stats_.rejected_kind;
            return kNoSlot;
        }
        // A reused ring window would be overwritten ahead of draws that still read it.
        if (window <= last_window_) {
            ++stats_.closed_window_wrap;
            close();
            return kNoSlot;
        }
        if (is_written(args) || (count != 0u && is_written(count))) {
            ++stats_.rejected_written;
            return kNoSlot;
        }
        last_window_ = window;
        ++stats_.slots_joined;
        return reserve_ + static_cast<size_t>(used_++) * slot_bytes_;
    }

private:
    static constexpr bool kinds_compatible(IndirectBatchKind run_kind, IndirectBatchKind join_kind) {
        if (run_kind == join_kind) return true;
        // Multi-draw runs can freely mix Draw and DrawIndexed clamps in the same batch.
        const bool run_draw = run_kind == IndirectBatchKind::Draw || run_kind == IndirectBatchKind::DrawIndexed;
        const bool join_draw = join_kind == IndirectBatchKind::Draw || join_kind == IndirectBatchKind::DrawIndexed;
        return run_draw && join_draw;
    }

    static constexpr uint32_t hash_u32(uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    void clear_written() {
        written_count_ = 0u;
        ++written_epoch_;
        if (written_epoch_ == 0u) {
            written_epochs_.fill(0u);
            written_epoch_ = 1u;
        }
    }

    bool insert_written(uint32_t id) {
        size_t idx = hash_u32(id) & (kWrittenTableCapacity - 1u);
        for (size_t probe = 0; probe < kWrittenTableCapacity; ++probe) {
            if (written_epochs_[idx] != written_epoch_) {
                if (written_count_ >= kWrittenMaxEntries) {
                    return false;
                }
                written_epochs_[idx] = written_epoch_;
                written_keys_[idx] = id;
                ++written_count_;
                return true;
            }
            if (written_keys_[idx] == id) {
                return true;
            }
            idx = (idx + 1u) & (kWrittenTableCapacity - 1u);
        }
        return false;
    }

    void note_packet(GpuOpcode op, const uint8_t* payload, uint32_t bytes) {
        using O = GpuOpcode;
        switch (op) {
            case O::WriteBuffer:
                if (bytes >= sizeof(uint32_t)) {
                    uint32_t id = 0;
                    std::memcpy(&id, payload, sizeof(id));
                    if (!insert_written(id)) {
                        ++stats_.closed_table_overflow;
                        close();
                    }
                } else {
                    ++stats_.closed_barrier_opcode;
                    close();
                }
                return;
            case O::DispatchIndirect:
                // In a DispatchMesh (type-105) run, each mesh command emits a DispatchIndirect
                // reading from the clamp ring (and writing only its dedicated mesh output buffer)
                // followed by a DrawIndirect. General DispatchIndirect outside a mesh run can write
                // arbitrary storage buffers and closes the batch.
                if (kind_ == IndirectBatchKind::DispatchMesh && bytes >= sizeof(GpuIndirectPayload)) {
                    GpuIndirectPayload p{};
                    std::memcpy(&p, payload, sizeof(p));
                    if (clamp_ring_buffer_ == 0u || p.buffer_id == clamp_ring_buffer_) {
                        ++stats_.mesh_dispatches_scanned;
                        if (mesh_output_buffer_ != 0u && !insert_written(mesh_output_buffer_)) {
                            ++stats_.closed_table_overflow;
                            close();
                        }
                        return;
                    }
                }
                ++stats_.closed_barrier_opcode;
                close();
                return;
            // Packets that cannot change buffer contents (draws only read their arguments; a pixel
            // shader writing an argument buffer is not supported, as WebGPU gives fragment storage
            // writes no ordering against indirect reads in the same pass either).
            case O::Nop:
            case O::CreateShaderModule:
            case O::Draw:
            case O::CreateRenderPipeline:
            case O::SetPipeline:
            case O::SetVertexBuffer:
            case O::CreateTexture:
            case O::WriteTexture:
            case O::CreateSampler:
            case O::CreateBindGroup:
            case O::SetBindGroup:
            case O::SetIndexBuffer:
            case O::DrawIndexed:
            case O::CreateComputePipeline:
            case O::SetComputePipeline:
            case O::CreateBindGroupLayout:
            case O::CreatePipelineLayout:
            case O::CreateRenderPipeline2:
            case O::CreateComputePipeline2:
            case O::BeginRenderPass:
            case O::EndRenderPass:
            case O::SetViewport:
            case O::SetScissorRect:
            case O::SetBlendConstant:
            case O::SetStencilReference:
            case O::CreateBindGroup2:
            case O::SetBindGroup2:
            case O::CreateSampler2:
            case O::CreateTexture2:
            case O::DrawIndirect:
            case O::DrawIndexedIndirect:
                return;
            // Explicit ordering / buffer-clobbering opcodes in playgta6_abi.h:
            // BeginFrame, ClearColor, CreateBuffer, Present, DestroyResource, DispatchCompute,
            // ReadbackBuffer, ReadbackTexture, and any future unknown opcode.
            case O::BeginFrame:
            case O::ClearColor:
            case O::CreateBuffer:
            case O::Present:
            case O::DestroyResource:
            case O::DispatchCompute:
            case O::ReadbackBuffer:
            case O::ReadbackTexture:
            default:
                ++stats_.closed_barrier_opcode;
                close();
                return;
        }
    }

    bool open_ = false;
    IndirectBatchKind kind_ = IndirectBatchKind::Draw;
    size_t reserve_ = 0;
    size_t slot_bytes_ = 0;
    size_t scanned_ = 0;
    uint32_t used_ = 0;
    uint32_t max_join_slots_ = kIndirectClampBatchSize - 1u;
    uint32_t last_window_ = 0;
    uint32_t clamp_ring_buffer_ = 0;
    uint32_t mesh_output_buffer_ = 0;
    uint32_t written_count_ = 0;
    uint32_t written_epoch_ = 1;
    std::array<uint32_t, kWrittenTableCapacity> written_keys_{};
    std::array<uint32_t, kWrittenTableCapacity> written_epochs_{};
    IndirectClampBatchStats stats_{};
};

}  // namespace playgta6::gpu::bridge
