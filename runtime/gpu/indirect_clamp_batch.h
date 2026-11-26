// Batching of ExecuteIndirect clamp passes (backlog 41).
//
// Every ExecuteIndirect needs its argument records clamped by a tiny compute pass before the draws
// read them. Running that pass right before each draw alternates compute and render work, which
// costs a pass break and a GPU pipeline bubble per draw. The bridge instead reserves, at the point
// where the first clamp of a run is recorded, a fixed number of patchable slots right behind it; the
// clamps of later executions in the same list are written into those slots (earlier in the stream)
// instead of ahead of their own draws. A clamp may move earlier only when nothing recorded in
// between can have changed the buffers it reads, so the clamp still sees exactly the bytes it
// would have seen right before its draw and every result stays identical.
//
// This class is the pure planning half: it tracks one open run, scans the packets recorded since
// the run's reservation and decides whether a new execution may join. It does not touch the
// encoder.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_set>

#include "abi/playgta6_abi.h"

namespace playgta6::gpu::bridge {

// Clamps per run, the inline one included (the reference's default run length).
inline constexpr uint32_t kIndirectClampBatchSize = 7;

class IndirectClampBatch {
public:
    static constexpr size_t kNoSlot = ~size_t{0};

    bool open() const { return open_; }

    // A run starts: `reserve_offset` is where the first patchable slot lives, `slot_bytes` the size
    // of each of the `kIndirectClampBatchSize - 1` slots, `scanned_to` the encoder size after the
    // reservation, `window` the output window of the run's first clamp.
    void begin(size_t reserve_offset, size_t slot_bytes, size_t scanned_to, uint32_t window) {
        open_ = true;
        reserve_ = reserve_offset;
        slot_bytes_ = slot_bytes;
        scanned_ = scanned_to;
        used_ = 0;
        last_window_ = window;
        written_.clear();
    }

    void close() {
        open_ = false;
        written_.clear();
    }

    uint32_t joined() const { return used_; }

    // Walks the packets recorded since the last scan. Any packet that can write a buffer other than
    // by a plain WriteBuffer, or that we do not recognise, closes the run.
    void scan(const uint8_t* data, size_t size) {
        if (!open_) {
            return;
        }
        if (size < scanned_) {  // the encoder was reset under us
            close();
            return;
        }
        while (open_ && scanned_ + sizeof(GpuPacketHeader) <= size) {
            GpuPacketHeader h{};
            std::memcpy(&h, data + scanned_, sizeof(h));
            const size_t end = scanned_ + sizeof(h) + h.payload_bytes;
            if (end > size) {
                break;
            }
            note_packet(static_cast<GpuOpcode>(h.opcode), data + scanned_ + sizeof(h), h.payload_bytes);
            scanned_ = end;
        }
    }

    // May a clamp reading `args` / `count` (0 = none) writing `window` move into the next slot?
    // Returns the slot's encoder offset and claims it, or kNoSlot (the run stays open unless the
    // window wrapped, which also ends it). Call scan() first.
    size_t try_join(uint32_t args, uint32_t count, uint32_t window) {
        if (!open_ || used_ + 1 >= kIndirectClampBatchSize) {
            return kNoSlot;
        }
        // A reused ring window would be overwritten ahead of draws that still read it.
        if (window <= last_window_) {
            close();
            return kNoSlot;
        }
        if (written_.count(args) != 0 || (count != 0 && written_.count(count) != 0)) {
            return kNoSlot;
        }
        last_window_ = window;
        return reserve_ + static_cast<size_t>(used_++) * slot_bytes_;
    }

private:
    void note_packet(GpuOpcode op, const uint8_t* payload, uint32_t bytes) {
        using O = GpuOpcode;
        switch (op) {
            case O::WriteBuffer:
                if (bytes >= sizeof(uint32_t)) {
                    uint32_t id = 0;
                    std::memcpy(&id, payload, sizeof(id));
                    written_.insert(id);
                } else {
                    close();
                }
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
            // Compute dispatches write storage buffers; frames, buffer (re)creation, destruction and
            // read-backs are ordering points. Anything else is unknown: stay safe.
            default:
                close();
                return;
        }
    }

    bool open_ = false;
    size_t reserve_ = 0;
    size_t slot_bytes_ = 0;
    size_t scanned_ = 0;
    uint32_t used_ = 0;
    uint32_t last_window_ = 0;
    std::unordered_set<uint32_t> written_;
};

}  // namespace playgta6::gpu::bridge
