// Xbox-specific D3D12 slots (G27): frame events and pacing, PresentX planes, pipeline-state streams
// and overrides, views at GPU addresses, command-list staging and the recorded-only Xbox slots.
// Evidence, contracts and confidence: d3d12_xs_xbox.h.

#include "gpu/d3d12_xs_xbox.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

#include "gpu/d3d12_guest_resources.h"
#include "gpu/descriptor_words.h"
#include "gpu/present_pacing.h"
#include "guest/boot.h"
#include "os/os_shim.h"
#include "os/win32_kernel.h"
#include "os/xmem_windows.h"

namespace playgta6::gpu::d3d12_guest {

namespace {

constexpr uint32_t kInfinite = 0xFFFFFFFFu;
// Longest a PRESENT wait (WaitFrameEventX type 2) waits for its token's PresentX before it falls
// back to the pacer tick (a token the caller itself would present later never arrives).
constexpr uint32_t kPresentWaitCapMs = 500;
constexpr uint64_t kMaxStreamBytes = 64u << 10;
constexpr uint32_t kMaxOverrides = 16;
constexpr uint32_t kMaxPlanes = 8;
constexpr uint32_t kMaxAllocationInfos = 4096;

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%" PRIx64, v);
    return b;
}

constexpr size_t align8(size_t v) {
    return (v + 7) & ~size_t{7};
}

constexpr uint64_t align_up(uint64_t v, uint64_t a) {
    return a == 0 ? v : (v + a - 1) / a * a;
}

}  // namespace

namespace xbox {

// ---------------------------------------------------------------------------------------------
// Console-native pipelines (slot 57)
// ---------------------------------------------------------------------------------------------

bool parse_native_pipeline(const GuestNativePipelineDesc& desc, NativePipeline* out, std::string* error) {
    constexpr uint64_t kMaxRecordBytes = 64u << 10;  // header record and pipeline metadata
    auto range = [&](NativeRange r) { return desc.ranges[static_cast<size_t>(r)]; };
    auto fail = [&](const std::string& message) {
        *error = message;
        return false;
    };
    for (size_t i = 0; i < kNativeRangeCount; ++i) {
        const GuestNativeRange& r = desc.ranges[i];
        if ((r.address == 0) != (r.size == 0)) {
            return fail("range " + std::to_string(i) + " has an address or a size but not both");
        }
        if (r.size > kMaxNativeRangeBytes) {
            return fail("range " + std::to_string(i) + " is too large (" + std::to_string(r.size) + " bytes)");
        }
    }
    NativePipeline p;
    p.root_signature = desc.pRootSignature;
    p.header = range(NativeRange::Header);
    p.metadata = range(NativeRange::PipelineMetadata);
    if (p.header.size == 0 || p.metadata.size == 0) {
        return fail("no header record or pipeline metadata (ranges 0/1)");
    }
    if (p.header.size > kMaxRecordBytes || p.metadata.size > kMaxRecordBytes) {
        return fail("header record or pipeline metadata larger than 64 KiB");
    }
    auto stage = [&](NativeRange code, NativeRange metadata, const char* name, NativeStage* s) {
        s->code = range(code);
        s->metadata = range(metadata);
        if ((s->code.size == 0) != (s->metadata.size == 0)) {
            return fail(std::string(name) + " stage has code without metadata or metadata without code");
        }
        if (s->code.size != 0 && s->code.address % kNativeCodeAlignment != 0) {
            return fail(std::string(name) + " stage code is not 256-byte aligned");
        }
        return true;
    };
    NativeStage a, b;
    if (!stage(NativeRange::PixelCode, NativeRange::PixelMetadata, "pixel", &p.pixel) ||
        !stage(NativeRange::VertexCode, NativeRange::VertexMetadata, "vertex", &a) ||
        !stage(NativeRange::VertexCodeAlt, NativeRange::VertexMetadataAlt, "vertex", &b)) {
        return false;
    }
    if (a.present() == b.present()) {
        return fail(a.present() ? "two vertex stages (ranges 4/5 and 6/7)" : "no vertex stage (ranges 4/5 or 6/7)");
    }
    p.vertex = a.present() ? a : b;
    p.vertex_alt = b.present();
    *out = p;
    return true;
}

bool parse_native_compute_pipeline(const GuestNativeComputePipelineDesc& desc, NativeComputePipeline* out,
                                   std::string* error) {
    constexpr uint64_t kMaxRecordBytes = 64u << 10;
    auto fail = [&](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };
    for (size_t i = 0; i < kNativeComputeRangeCount; ++i) {
        const GuestNativeRange& r = desc.ranges[i];
        if ((r.address == 0) != (r.size == 0)) {
            return fail("range " + std::to_string(i) + " has an address or a size but not both");
        }
        if (r.size > kMaxNativeRangeBytes) {
            return fail("range " + std::to_string(i) + " is too large (" + std::to_string(r.size) + " bytes)");
        }
    }
    NativeComputePipeline p;
    p.root_signature = desc.pRootSignature;
    p.header = desc.ranges[0];
    if (p.header.size == 0) {
        return fail("no header record (range 0)");
    }
    if (p.header.size > kMaxRecordBytes) {
        return fail("header record larger than 64 KiB");
    }
    if (desc.ranges[2].size != 0 && desc.ranges[3].size != 0) {
        // 4-range layout (matching slot 57): header, pipeline metadata, compute code, compute metadata.
        p.metadata = desc.ranges[1];
        if (p.metadata.size > kMaxRecordBytes) {
            return fail("pipeline metadata larger than 64 KiB");
        }
        p.compute.code = desc.ranges[2];
        p.compute.metadata = desc.ranges[3];
    } else if (desc.ranges[1].size != 0 && desc.ranges[3].size == 0) {
        // 3-range compact layout: header, compute code, optional compute metadata.
        p.compute.code = desc.ranges[1];
        p.compute.metadata = desc.ranges[2];
    } else {
        return fail("no compute stage code range");
    }
    if (p.compute.code.address % kNativeCodeAlignment != 0) {
        return fail("compute stage code is not 256-byte aligned");
    }
    *out = p;
    return true;
}

// ---------------------------------------------------------------------------------------------
// Slot 70/71 address-based texture view metadata registry
// ---------------------------------------------------------------------------------------------

namespace {

struct AddressTextureViewRegistry {
    std::mutex mutex;
    std::unordered_map<uint64_t, AddressTextureViewInfo> by_va;
};

AddressTextureViewRegistry& address_texture_registry() {
    static AddressTextureViewRegistry reg;
    return reg;
}

}  // namespace

void record_address_texture_view(const AddressTextureViewInfo& info) {
    if (info.gpu_va == 0) {
        return;
    }
    AddressTextureViewRegistry& reg = address_texture_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.by_va[info.gpu_va] = info;
}

bool lookup_address_texture_view(uint64_t gpu_va, AddressTextureViewInfo* out) {
    if (gpu_va == 0) {
        return false;
    }
    AddressTextureViewRegistry& reg = address_texture_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    const auto it = reg.by_va.find(gpu_va);
    if (it == reg.by_va.end()) {
        return false;
    }
    if (out != nullptr) {
        *out = it->second;
    }
    return true;
}

size_t address_texture_view_count() {
    AddressTextureViewRegistry& reg = address_texture_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    return reg.by_va.size();
}

void clear_address_texture_views() {
    AddressTextureViewRegistry& reg = address_texture_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.by_va.clear();
}

// ---------------------------------------------------------------------------------------------
// Pipeline-state streams
// ---------------------------------------------------------------------------------------------

namespace {

// D3D12 defaults for a stream without the corresponding subobject (CD3DX12 defaults).
void set_stream_defaults(GuestGraphicsPipelineStateDesc* g) {
    g->SampleMask = 0xFFFFFFFFu;
    g->SampleCount = 1;
    g->PrimitiveTopologyType = 3;  // TRIANGLE
    for (GuestRenderTargetBlendDesc& rt : g->BlendState.RenderTarget) {
        rt.SrcBlend = 2;        // ONE
        rt.DestBlend = 1;       // ZERO
        rt.BlendOp = 1;         // ADD
        rt.SrcBlendAlpha = 2;
        rt.DestBlendAlpha = 1;
        rt.BlendOpAlpha = 1;
        rt.LogicOp = 4;         // NOOP
        rt.RenderTargetWriteMask = 0xF;
    }
    g->RasterizerState.FillMode = 3;  // SOLID
    g->RasterizerState.CullMode = 3;  // BACK
    g->RasterizerState.DepthClipEnable = 1;
    GuestDepthStencilDesc& ds = g->DepthStencilState;
    ds.DepthEnable = 1;
    ds.DepthWriteMask = 1;  // ALL
    ds.DepthFunc = 2;       // LESS
    ds.StencilReadMask = 0xFF;
    ds.StencilWriteMask = 0xFF;
    for (GuestDepthStencilOpDesc* op : {&ds.FrontFace, &ds.BackFace}) {
        op->StencilFailOp = op->StencilDepthFailOp = op->StencilPassOp = 1;  // KEEP
        op->StencilFunc = 8;                                                  // ALWAYS
    }
}

struct SubobjectLayout {
    size_t data_offset;  // from the subobject start
    size_t size;
};

bool layout_of(uint32_t type, SubobjectLayout* out) {
    switch (type) {
    case 0:   // ROOT_SIGNATURE
        *out = {8, 8};
        return true;
    case 1: case 2: case 3: case 4: case 5: case 6:  // VS PS DS HS GS CS
    case 23: case 24:                                // AS MS
    case 19:                                         // CACHED_PSO
    case 28:                                         // SERIALIZED_ROOT_SIGNATURE
        *out = {8, 16};
        return true;
    case 12:  // INPUT_LAYOUT
        *out = {8, 16};
        return true;
    case 7:   // STREAM_OUTPUT
        *out = {8, 32};
        return true;
    case 22:  // VIEW_INSTANCING
        *out = {8, 24};
        return true;
    case 8:   // BLEND
        *out = {4, sizeof(GuestBlendDesc)};
        return true;
    case 9: case 13: case 14: case 16: case 18: case 20:  // SAMPLE_MASK IB_STRIP_CUT TOPOLOGY DSV_FORMAT NODE_MASK FLAGS
        *out = {4, 4};
        return true;
    case 10: case 26:  // RASTERIZER, RASTERIZER1
        *out = {4, 44};
        return true;
    case 27:  // RASTERIZER2
        *out = {4, 40};
        return true;
    case 11:  // DEPTH_STENCIL
        *out = {4, 52};
        return true;
    case 21:  // DEPTH_STENCIL1
        *out = {4, 56};
        return true;
    case 25:  // DEPTH_STENCIL2
        *out = {4, 60};
        return true;
    case 15:  // RENDER_TARGET_FORMATS
        *out = {4, 36};
        return true;
    case 17:  // SAMPLE_DESC
        *out = {4, 8};
        return true;
    default:
        return false;
    }
}

template <typename T>
T load(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

}  // namespace

bool parse_pipeline_stream(const uint8_t* bytes, size_t size, StreamPipeline* out, std::string* error) {
    StreamPipeline p;
    set_stream_defaults(&p.graphics);
    bool has_cs = false, has_mesh = false;
    size_t off = 0;
    while (off < size) {
        if (size - off < 4) {
            *error = "truncated subobject at offset " + std::to_string(off);
            return false;
        }
        const uint32_t type = load<uint32_t>(bytes + off);
        SubobjectLayout l{};
        if (!layout_of(type, &l)) {
            *error = "unknown subobject type " + std::to_string(type) + " at offset " + std::to_string(off);
            return false;
        }
        if (off + l.data_offset + l.size > size) {
            *error = "subobject type " + std::to_string(type) + " exceeds the stream";
            return false;
        }
        const uint8_t* d = bytes + off + l.data_offset;
        GuestGraphicsPipelineStateDesc& g = p.graphics;
        switch (type) {
        case 0:
            g.pRootSignature = p.compute.pRootSignature = load<uint64_t>(d);
            break;
        case 1: std::memcpy(&g.VS, d, 16); break;
        case 2: std::memcpy(&g.PS, d, 16); break;
        case 3: std::memcpy(&g.DS, d, 16); break;
        case 4: std::memcpy(&g.HS, d, 16); break;
        case 5: std::memcpy(&g.GS, d, 16); break;
        case 6:
            std::memcpy(&p.compute.CS, d, 16);
            has_cs = true;
            break;
        case 23:
            p.amplification = true;
            has_mesh = true;
            break;
        case 24:
            std::memcpy(&p.mesh, d, 16);
            has_mesh = true;
            break;
        case 7: std::memcpy(&g.StreamOutput, d, 32); break;
        case 8: std::memcpy(&g.BlendState, d, sizeof(GuestBlendDesc)); break;
        case 9: g.SampleMask = load<uint32_t>(d); break;
        case 10: std::memcpy(&g.RasterizerState, d, 44); break;
        case 26:
        case 27: {  // float depth bias (RASTERIZER1/2): converted to the integer field
            GuestRasterizerDesc r{};
            r.FillMode = load<uint32_t>(d);
            r.CullMode = load<uint32_t>(d + 4);
            r.FrontCounterClockwise = load<int32_t>(d + 8);
            r.DepthBias = static_cast<int32_t>(load<float>(d + 12));
            r.DepthBiasClamp = load<float>(d + 16);
            r.SlopeScaledDepthBias = load<float>(d + 20);
            r.DepthClipEnable = load<int32_t>(d + 24);
            if (type == 26) {
                r.MultisampleEnable = load<int32_t>(d + 28);
                r.AntialiasedLineEnable = load<int32_t>(d + 32);
                r.ForcedSampleCount = load<uint32_t>(d + 36);
                r.ConservativeRaster = load<uint32_t>(d + 40);
            } else {
                r.ForcedSampleCount = load<uint32_t>(d + 32);
                r.ConservativeRaster = load<uint32_t>(d + 36);
            }
            g.RasterizerState = r;
            break;
        }
        case 11:
        case 21:  // DEPTH_STENCIL1 = DEPTH_STENCIL + DepthBoundsTestEnable (not supported, ignored)
            std::memcpy(&g.DepthStencilState, d, 52);
            break;
        case 25: {  // DEPTH_STENCIL2: per-face stencil masks (front face masks are used)
            GuestDepthStencilDesc ds{};
            ds.DepthEnable = load<int32_t>(d);
            ds.DepthWriteMask = load<uint32_t>(d + 4);
            ds.DepthFunc = load<uint32_t>(d + 8);
            ds.StencilEnable = load<int32_t>(d + 12);
            std::memcpy(&ds.FrontFace, d + 16, 16);
            ds.StencilReadMask = d[32];
            ds.StencilWriteMask = d[33];
            std::memcpy(&ds.BackFace, d + 36, 16);
            g.DepthStencilState = ds;
            break;
        }
        case 12: std::memcpy(&g.InputLayout, d, 16); break;
        case 13: g.IBStripCutValue = load<uint32_t>(d); break;
        case 14: g.PrimitiveTopologyType = load<uint32_t>(d); break;
        case 15:
            std::memcpy(g.RTVFormats, d, 32);
            g.NumRenderTargets = load<uint32_t>(d + 32);
            break;
        case 16: g.DSVFormat = load<uint32_t>(d); break;
        case 17:
            g.SampleCount = load<uint32_t>(d);
            g.SampleQuality = load<uint32_t>(d + 4);
            break;
        case 18: g.NodeMask = p.compute.NodeMask = load<uint32_t>(d); break;
        case 19: std::memcpy(&g.CachedPSO, d, 16); break;
        case 20: g.Flags = p.compute.Flags = load<uint32_t>(d); break;
        default:  // VIEW_INSTANCING, SERIALIZED_ROOT_SIGNATURE
            ++p.ignored_subobjects;
            break;
        }
        ++p.subobjects;
        off = align8(off + l.data_offset + l.size);
    }
    p.kind = has_mesh ? StreamPipelineKind::Mesh : (has_cs ? StreamPipelineKind::Compute : StreamPipelineKind::Graphics);
    *out = p;
    return true;
}

// ---------------------------------------------------------------------------------------------
// Frame pacing
// ---------------------------------------------------------------------------------------------

FramePacer::FramePacer() : origin_(Clock::now()) {}

uint64_t FramePacer::frame_locked(Clock::time_point now) const {
    if (now < origin_) {
        return base_frame_;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - origin_).count();
    return base_frame_ + static_cast<uint64_t>(elapsed) / interval_us_;
}

FramePacer::Clock::time_point FramePacer::start_of_locked(uint64_t frame) const {
    if (frame <= base_frame_) {
        return origin_;
    }
    const uint64_t delta = frame - base_frame_;
    const uint64_t max_frames = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / interval_us_;
    const uint64_t clamped = std::min(delta, max_frames);
    return origin_ + std::chrono::microseconds(static_cast<int64_t>(clamped * interval_us_));
}

void FramePacer::set_interval(uint32_t interval_us, uint32_t period) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = Clock::now();
    const uint64_t current = frame_locked(now);
    // Keep the tick count continuous: the current tick starts now at the new interval.
    base_frame_ = current;
    origin_ = now;
    interval_us_ = std::clamp(interval_us == 0 ? kDefaultIntervalUs : interval_us, kMinIntervalUs, kMaxIntervalUs);
    period_ = period;
    cv_.notify_all();
}

uint32_t FramePacer::interval_us() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return interval_us_;
}

uint32_t FramePacer::period() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return period_;
}

uint64_t FramePacer::current_frame() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return frame_locked(Clock::now());
}

FramePacer::WaitResult FramePacer::wait_after(uint64_t after, uint32_t timeout_ms, uint64_t* frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    return wait_locked(lock, after, timeout_ms, frame);
}

FramePacer::WaitResult FramePacer::wait_event(uint32_t kind, uint32_t timeout_ms, uint64_t* frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    uint64_t& last = last_tick_[kind % kKinds];
    const uint64_t current = frame_locked(Clock::now());
    if (!shutdown_ && last != 0 && current > last + 1) {
        // More than one tick behind: end at once on the overdue tick.
        last = last + 1;
        if (frame != nullptr) {
            *frame = last;
        }
        return WaitResult::Ok;
    }
    // Otherwise the next tick after both now and the kind's last tick.
    uint64_t reached = 0;
    const WaitResult r = wait_locked(lock, std::max(current, last), timeout_ms, &reached);
    if (frame != nullptr) {
        *frame = reached;
    }
    if (r == WaitResult::Ok) {
        last = std::max(last, reached);
    }
    return r;
}

uint64_t FramePacer::last_event_tick(uint32_t kind) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_tick_[kind % kKinds];
}

void FramePacer::reset_event_ticks() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (uint64_t& tick : last_tick_) {
        tick = 0;
    }
}

FramePacer::WaitResult FramePacer::wait_locked(std::unique_lock<std::mutex>& lock, uint64_t after,
                                               uint32_t timeout_ms, uint64_t* frame) {
    const auto start = Clock::now();
    const bool bounded = timeout_ms != kInfinite;
    const auto deadline = start + std::chrono::milliseconds(bounded ? timeout_ms : 0);
    // Bound any future target to at most one interval past the tick at wait entry, so an
    // ahead-of-clock or out-of-range token never blocks indefinitely or overflows wake math.
    const uint64_t initial = frame_locked(start);
    const uint64_t target_after = std::min(after, initial);
    for (;;) {
        const auto now = Clock::now();
        const uint64_t current = frame_locked(now);
        if (shutdown_) {
            if (frame != nullptr) {
                *frame = std::max<uint64_t>(current, 1);
            }
            return WaitResult::Shutdown;
        }
        if (current > target_after) {
            if (frame != nullptr) {
                *frame = std::max<uint64_t>(current, 1);
            }
            return WaitResult::Ok;
        }
        if (bounded && now >= deadline) {
            if (frame != nullptr) {
                *frame = std::max<uint64_t>(current, 1);
            }
            return WaitResult::Timeout;
        }
        auto wake = start_of_locked(target_after + 1);
        if (bounded && deadline < wake) {
            wake = deadline;
        }
        // Woken by the tick, a new interval or shutdown; spurious wake-ups loop.
        cv_.wait_until(lock, wake);
    }
}

void FramePacer::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
    cv_.notify_all();
}

bool FramePacer::is_shut_down() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_;
}

// ---------------------------------------------------------------------------------------------
// Xbox Scarlett 3D helpers (G84): fast clears, predication, async compute sync, VRS/depth bounds
// ---------------------------------------------------------------------------------------------

void FastClearRegistry::register_surface_meta(uint64_t surface_key, CompressionMetaKind kind, uint64_t meta_va,
                                              uint64_t meta_bytes) {
    if (surface_key == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    SurfaceCompressionMeta& entry = surfaces_[surface_key];
    entry.surface_key = surface_key;
    entry.meta_kind = kind;
    entry.meta_va = meta_va;
    entry.meta_bytes = meta_bytes;
}

void FastClearRegistry::record_color_clear(uint64_t surface_key, uint32_t dxgi_format, const float rgba[4]) {
    if (surface_key == 0 || rgba == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    SurfaceCompressionMeta& entry = surfaces_[surface_key];
    entry.surface_key = surface_key;
    entry.clear_value.kind = FastClearKind::Color;
    entry.clear_value.dxgi_format = dxgi_format;
    for (int i = 0; i < 4; ++i) {
        entry.clear_value.rgba[i] = std::isfinite(rgba[i]) ? rgba[i] : 0.0f;
    }
    entry.clear_value.valid = true;
    entry.fast_clear_active = true;
    ++entry.clear_generation;
    ++color_clears_;
}

void FastClearRegistry::record_depth_stencil_clear(uint64_t surface_key, uint32_t dxgi_format, uint32_t flags,
                                                   float depth, uint8_t stencil) {
    if (surface_key == 0 || (flags & 3u) == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    SurfaceCompressionMeta& entry = surfaces_[surface_key];
    entry.surface_key = surface_key;
    entry.clear_value.kind = FastClearKind::DepthStencil;
    entry.clear_value.dxgi_format = dxgi_format;
    entry.clear_value.clear_flags = flags & 3u;
    if ((flags & 1u) != 0) {
        entry.clear_value.depth = std::isfinite(depth) ? std::clamp(depth, 0.0f, 1.0f) : 1.0f;
    }
    if ((flags & 2u) != 0) {
        entry.clear_value.stencil = stencil;
    }
    entry.clear_value.valid = true;
    entry.fast_clear_active = true;
    ++entry.clear_generation;
    ++depth_stencil_clears_;
}

void FastClearRegistry::record_uav_clear_float(uint64_t surface_key, const float rgba[4]) {
    if (surface_key == 0 || rgba == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    SurfaceCompressionMeta& entry = surfaces_[surface_key];
    entry.surface_key = surface_key;
    entry.clear_value.kind = FastClearKind::UnorderedAccessFloat;
    for (int i = 0; i < 4; ++i) {
        entry.clear_value.rgba[i] = rgba[i];
    }
    entry.clear_value.valid = true;
    entry.fast_clear_active = true;
    ++entry.clear_generation;
    ++uav_clears_;
}

void FastClearRegistry::record_uav_clear_uint(uint64_t surface_key, const uint32_t rgba[4]) {
    if (surface_key == 0 || rgba == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    SurfaceCompressionMeta& entry = surfaces_[surface_key];
    entry.surface_key = surface_key;
    entry.clear_value.kind = FastClearKind::UnorderedAccessUint;
    for (int i = 0; i < 4; ++i) {
        entry.clear_value.uint_rgba[i] = rgba[i];
    }
    entry.clear_value.valid = true;
    entry.fast_clear_active = true;
    ++entry.clear_generation;
    ++uav_clears_;
}

bool FastClearRegistry::mark_decompressed(uint64_t surface_key) {
    if (surface_key == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = surfaces_.find(surface_key);
    if (it == surfaces_.end()) {
        return false;
    }
    const bool was_active = it->second.fast_clear_active;
    it->second.fast_clear_active = false;
    ++it->second.decompress_count;
    ++decompresses_;
    return was_active;
}

bool FastClearRegistry::lookup(uint64_t surface_key, SurfaceCompressionMeta* out) const {
    if (surface_key == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = surfaces_.find(surface_key);
    if (it == surfaces_.end()) {
        return false;
    }
    if (out != nullptr) {
        *out = it->second;
    }
    return true;
}

bool FastClearRegistry::erase(uint64_t surface_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    return surfaces_.erase(surface_key) != 0;
}

void FastClearRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    surfaces_.clear();
    color_clears_ = 0;
    depth_stencil_clears_ = 0;
    uav_clears_ = 0;
    decompresses_ = 0;
}

size_t FastClearRegistry::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return surfaces_.size();
}

FastClearRegistry::Stats FastClearRegistry::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats s{};
    s.color_clears = color_clears_;
    s.depth_stencil_clears = depth_stencil_clears_;
    s.uav_clears = uav_clears_;
    s.decompresses = decompresses_;
    for (const auto& [key, meta] : surfaces_) {
        if (meta.fast_clear_active) {
            ++s.active_fast_clears;
        }
        switch (meta.meta_kind) {
        case CompressionMetaKind::Dcc: ++s.dcc_surfaces; break;
        case CompressionMetaKind::Cmask: ++s.cmask_surfaces; break;
        case CompressionMetaKind::Htile: ++s.htile_surfaces; break;
        case CompressionMetaKind::None: break;
        }
    }
    return s;
}

FastClearRegistry& fast_clear_registry() {
    static FastClearRegistry reg;
    return reg;
}

bool parse_guest_clear_color(uint64_t color_va, float out_rgba[4]) {
    if (color_va == 0 || out_rgba == nullptr) {
        return false;
    }
    float tmp[4] = {};
    if (!com::read_guest(color_va, tmp, sizeof(tmp))) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        out_rgba[i] = tmp[i];
    }
    return true;
}

bool parse_guest_uav_clear_uint(uint64_t values_va, uint32_t out_rgba[4]) {
    if (values_va == 0 || out_rgba == nullptr) {
        return false;
    }
    uint32_t tmp[4] = {};
    if (!com::read_guest(values_va, tmp, sizeof(tmp))) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        out_rgba[i] = tmp[i];
    }
    return true;
}

bool PredicationTracker::set(uint64_t buffer_va, uint64_t aligned_offset, uint32_t raw_op, std::string* error) {
    if (buffer_va == 0) {
        state_ = PredicationState{};
        ++stats_.clear_calls;
        return true;
    }
    if ((aligned_offset & 7u) != 0) {
        ++stats_.invalid_calls;
        if (error != nullptr) {
            *error = "SetPredication: offset must be 8-byte aligned";
        }
        return false;
    }
    if (raw_op > static_cast<uint32_t>(PredicationOp::NotEqualZero)) {
        ++stats_.invalid_calls;
        if (error != nullptr) {
            *error = "SetPredication: invalid predication op " + std::to_string(raw_op);
        }
        return false;
    }
    state_.active = true;
    state_.buffer_va = buffer_va;
    state_.aligned_offset = aligned_offset;
    state_.op = static_cast<PredicationOp>(raw_op);
    state_.last_evaluated = false;
    state_.last_skip_work = false;
    state_.last_value = 0;
    ++stats_.set_calls;
    return true;
}

void PredicationTracker::reset() {
    state_ = PredicationState{};
    scopes_.clear();
}

bool PredicationTracker::push_scope() {
    if (scopes_.size() >= kMaxScopeDepth) {
        return false;
    }
    scopes_.push_back(state_);
    return true;
}

bool PredicationTracker::pop_scope() {
    if (scopes_.empty()) {
        return false;
    }
    state_ = scopes_.back();
    scopes_.pop_back();
    return true;
}

bool PredicationTracker::should_skip_work(const std::function<bool(uint64_t va, uint64_t* value)>& reader) {
    if (!state_.active) {
        ++stats_.work_passed;
        return false;
    }
    ++stats_.evaluations;
    uint64_t val = 0;
    const uint64_t addr = state_.buffer_va + state_.aligned_offset;
    if (!reader || !reader(addr, &val)) {
        // Fail safe: unreadable predicate memory must never silently drop rendering work.
        ++stats_.unreadable_reads;
        ++stats_.work_passed;
        state_.last_evaluated = false;
        state_.last_skip_work = false;
        return false;
    }
    const bool skip = (state_.op == PredicationOp::EqualZero) ? (val == 0) : (val != 0);
    state_.last_evaluated = true;
    state_.last_value = val;
    state_.last_skip_work = skip;
    if (skip) {
        ++stats_.work_skipped;
    } else {
        ++stats_.work_passed;
    }
    return skip;
}

bool PredicationTracker::should_skip_work_guest(const Device* device) {
    return should_skip_work([device](uint64_t va, uint64_t* out_val) -> bool {
        const uint64_t cpu_va = device != nullptr ? device->cpu_address_for_gpu(va) : va;
        uint64_t v = 0;
        if (!com::read_guest(cpu_va, v)) {
            return false;
        }
        *out_val = v;
        return true;
    });
}

size_t AsyncComputeSync::record_signal(uint64_t queue_id, QueueKind kind, uint64_t fence_id, uint64_t value) {
    size_t newly_satisfied = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        switch (kind) {
        case QueueKind::Direct: ++stats_.signals_direct; break;
        case QueueKind::Compute: ++stats_.signals_compute; break;
        case QueueKind::Copy: ++stats_.signals_copy; break;
        }
        FenceState& f = fences_[fence_id];
        if (!f.ever_signalled || value > f.value) {
            f.value = value;
        }
        f.last_signaller_queue = queue_id;
        f.last_signaller_kind = kind;
        f.ever_signalled = true;
        ++next_seq_;

        auto it = pending_waits_.begin();
        while (it != pending_waits_.end()) {
            if (it->fence_id == fence_id && it->value <= f.value) {
                ++newly_satisfied;
                ++stats_.waits_satisfied;
                if (it->queue_kind != kind) {
                    ++stats_.cross_queue_syncs;
                }
                it = pending_waits_.erase(it);
            } else {
                ++it;
            }
        }
    }
    cv_.notify_all();
    return newly_satisfied;
}

bool AsyncComputeSync::record_wait(uint64_t queue_id, QueueKind kind, uint64_t fence_id, uint64_t value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = fences_.find(fence_id);
    if (it != fences_.end() && it->second.ever_signalled && it->second.value >= value) {
        ++stats_.waits_immediate;
        ++stats_.waits_satisfied;
        if (it->second.last_signaller_kind != kind) {
            ++stats_.cross_queue_syncs;
        }
        return true;
    }
    if (it != fences_.end() && it->second.ever_signalled && it->second.last_signaller_queue == queue_id &&
        it->second.value < value) {
        ++stats_.self_waits_detected;
    }
    ++stats_.waits_deferred;
    pending_waits_.push_back(PendingWait{next_seq_++, queue_id, kind, fence_id, value});
    return false;
}

bool AsyncComputeSync::wait_until_satisfied(uint64_t fence_id, uint64_t value, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto reached = [&] {
        if (shutdown_) {
            return true;
        }
        const auto it = fences_.find(fence_id);
        return it != fences_.end() && it->second.ever_signalled && it->second.value >= value;
    };
    const bool woke = cv_.wait_for(lock, timeout.count() < 0 ? std::chrono::milliseconds(0) : timeout, reached);
    const auto it = fences_.find(fence_id);
    const bool satisfied = it != fences_.end() && it->second.ever_signalled && it->second.value >= value;
    if (!satisfied && !woke) {
        ++stats_.wait_timeouts;
    }
    return satisfied;
}

bool AsyncComputeSync::is_potential_self_wait(uint64_t queue_id, uint64_t fence_id, uint64_t value) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = fences_.find(fence_id);
    if (it == fences_.end() || !it->second.ever_signalled) {
        return false;
    }
    return it->second.value < value && it->second.last_signaller_queue == queue_id;
}

uint64_t AsyncComputeSync::fence_value(uint64_t fence_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = fences_.find(fence_id);
    return (it != fences_.end() && it->second.ever_signalled) ? it->second.value : 0u;
}

size_t AsyncComputeSync::pending_wait_count(uint64_t fence_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fence_id == 0) {
        return pending_waits_.size();
    }
    size_t n = 0;
    for (const PendingWait& w : pending_waits_) {
        if (w.fence_id == fence_id) {
            ++n;
        }
    }
    return n;
}

void AsyncComputeSync::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
}

void AsyncComputeSync::reset() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = false;
        next_seq_ = 1;
        fences_.clear();
        pending_waits_.clear();
        stats_ = Stats{};
    }
    cv_.notify_all();
}

AsyncComputeSync::Stats AsyncComputeSync::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

namespace {

struct CommandListXboxStateTable {
    std::mutex mutex;
    std::unordered_map<uint64_t, DepthBoundsState> depth_bounds;
    std::unordered_map<uint64_t, ShadingRateState> shading_rate;
};

CommandListXboxStateTable& command_list_xbox_state() {
    static CommandListXboxStateTable table;
    return table;
}

}  // namespace

bool is_valid_shading_rate(uint32_t rate) {
    switch (rate) {
    case 0x0:  // D3D12_SHADING_RATE_1X1
    case 0x1:  // D3D12_SHADING_RATE_1X2
    case 0x4:  // D3D12_SHADING_RATE_2X1
    case 0x5:  // D3D12_SHADING_RATE_2X2
    case 0x6:  // D3D12_SHADING_RATE_2X4
    case 0x9:  // D3D12_SHADING_RATE_4X2
    case 0xA:  // D3D12_SHADING_RATE_4X4
        return true;
    default:
        return false;
    }
}

bool is_valid_shading_rate_combiner(uint32_t combiner) {
    return combiner <= 4u;  // PASSTHROUGH(0)..SUM(4)
}

bool lookup_command_list_depth_bounds(uint64_t list_identity, DepthBoundsState* out) {
    if (list_identity == 0) {
        return false;
    }
    CommandListXboxStateTable& t = command_list_xbox_state();
    std::lock_guard<std::mutex> lock(t.mutex);
    const auto it = t.depth_bounds.find(list_identity);
    if (it == t.depth_bounds.end()) {
        return false;
    }
    if (out != nullptr) {
        *out = it->second;
    }
    return true;
}

bool lookup_command_list_shading_rate(uint64_t list_identity, ShadingRateState* out) {
    if (list_identity == 0) {
        return false;
    }
    CommandListXboxStateTable& t = command_list_xbox_state();
    std::lock_guard<std::mutex> lock(t.mutex);
    const auto it = t.shading_rate.find(list_identity);
    if (it == t.shading_rate.end()) {
        return false;
    }
    if (out != nullptr) {
        *out = it->second;
    }
    return true;
}

void clear_command_list_xbox_state() {
    CommandListXboxStateTable& t = command_list_xbox_state();
    std::lock_guard<std::mutex> lock(t.mutex);
    t.depth_bounds.clear();
    t.shading_rate.clear();
}

}  // namespace xbox

// ---------------------------------------------------------------------------------------------
// Device: frame events
// ---------------------------------------------------------------------------------------------

namespace {
std::atomic<bool> g_present_token_waits{true};
std::atomic<uint32_t> g_present_max_interval{0};
std::atomic<uint64_t> g_frame_waits{0};
std::atomic<uint64_t> g_frame_wait_us{0};
std::atomic<uint64_t> g_frame_polls{0};
std::atomic<uint64_t> g_frame_timeouts{0};
}  // namespace

void xbox::set_present_token_waits(bool on) { g_present_token_waits.store(on, std::memory_order_relaxed); }

void xbox::set_present_max_interval(uint32_t max_interval) {
    g_present_max_interval.store(max_interval, std::memory_order_relaxed);
}

uint32_t xbox::present_max_interval() {
    return g_present_max_interval.load(std::memory_order_relaxed);
}

xbox::FrameWaitTotals xbox::frame_wait_totals() {
    FrameWaitTotals t;
    t.waits = g_frame_waits.load(std::memory_order_relaxed);
    t.wait_us = g_frame_wait_us.load(std::memory_order_relaxed);
    t.polls = g_frame_polls.load(std::memory_order_relaxed);
    t.timeouts = g_frame_timeouts.load(std::memory_order_relaxed);
    return t;
}

HResult Device::SetFrameIntervalX(uint64_t, uint32_t interval_us, uint32_t period, uint32_t) {
    os::log("d3d12_xs: frame interval set to " + std::to_string(interval_us) + " us, period " +
            std::to_string(period));
    pacer_.set_interval(interval_us, period);
    // The title drives the frame interval: no extra 60 Hz present pacing on top.
    present_pacer().set_title_driven(true);
    count(xbox::Counter::FrameIntervalChanges);
    return kSOk;
}

HResult Device::ScheduleFrameEventX(uint32_t type, uint32_t offset_us, uint64_t event_handle, uint32_t) {
    // Events are ticks of the pacer; the offset shifts the event inside the frame. WaitFrameEventX
    // waits on the pacer directly; a title that passes an event handle gets it signalled at every
    // tick (`offset_us` in) by the frame-event thread, every scheduled type on each tick.
    count(xbox::Counter::FrameEventsScheduled);
    if (event_handle == 0) {
        return kSOk;
    }
    std::lock_guard<std::mutex> lock(frame_event_mutex_);
    if (frame_event_stop_) {
        return kSOk;
    }
    frame_events_[type] = ScheduledFrameEvent{std::min<uint32_t>(offset_us, xbox::FramePacer::kMaxIntervalUs),
                                              event_handle};
    if (!frame_event_thread_.joinable()) {
        frame_event_thread_ = std::thread([this] { run_frame_events(); });
    }
    return kSOk;
}

void Device::run_frame_events() {
    for (;;) {
        uint64_t frame = 0;
        if (pacer_.wait_after(pacer_.current_frame(), kInfinite, &frame) == xbox::FramePacer::WaitResult::Shutdown) {
            return;
        }
        const auto tick = std::chrono::steady_clock::now();
        std::vector<ScheduledFrameEvent> due;
        {
            std::lock_guard<std::mutex> lock(frame_event_mutex_);
            if (frame_event_stop_) {
                return;
            }
            for (const auto& [type, event] : frame_events_) {
                due.push_back(event);
            }
        }
        std::sort(due.begin(), due.end(),
                  [](const ScheduledFrameEvent& a, const ScheduledFrameEvent& b) { return a.offset_us < b.offset_us; });
        for (const ScheduledFrameEvent& event : due) {
            std::unique_lock<std::mutex> lock(frame_event_mutex_);
            if (frame_event_cv_.wait_until(lock, tick + std::chrono::microseconds(event.offset_us),
                                           [this] { return frame_event_stop_; })) {
                return;
            }
            lock.unlock();
            pg6_win32_set_event(event.handle);
        }
    }
}

void Device::stop_frame_event_thread() {
    {
        std::lock_guard<std::mutex> lock(frame_event_mutex_);
        frame_event_stop_ = true;
    }
    frame_event_cv_.notify_all();
    if (frame_event_thread_.joinable() && frame_event_thread_.get_id() != std::this_thread::get_id()) {
        frame_event_thread_.join();
    }
}

HResult Device::WaitFrameEventX(uint32_t type, uint32_t timeout_ms, uint64_t, uint32_t flags, GuestPtr<uint64_t> token) {
    if (token.is_null()) {
        note_bad_argument("WaitFrameEventX: null token pointer");
        return kEInvalidArg;
    }
    uint64_t in = 0;
    token.read(in);
    count(xbox::Counter::FrameWaits);
    uint64_t frame = 0;
    if (timeout_ms == 0) {
        // Poll: never fails and never blocks (the executable traps if this fails).
        g_frame_polls.fetch_add(1, std::memory_order_relaxed);
        frame = pacer_.current_frame();
    } else {
        // Type 2 (PRESENT) for a token: completes once PresentX has presented that token, as on
        // the console, where the event means the frame's GPU work is done. Titles reuse that
        // frame's CPU-written memory (upload rings) on it, and our side reads that memory at
        // ExecuteCommandLists, before the frame's PresentX: completing earlier hands the title a
        // region the frame's submissions have not read yet (G66, #691). A token nobody presents
        // within kPresentWaitCapMs (a wait on the caller's own future present) falls back to the
        // pacer tick below and is counted. Other waits (ORIGIN, ...) take at most one interval.
        const auto started = std::chrono::steady_clock::now();
        uint64_t after = pacer_.current_frame();
        if (type == 2 && in != 0 && g_present_token_waits.load(std::memory_order_relaxed)) {
            const uint64_t completed = last_present_token.load(std::memory_order_acquire);
            if (completed != 0 && in <= completed) {
                after = 0;
            } else if (in > last_frame_token_.load(std::memory_order_acquire)) {
                // A token never handed out (ahead of every frame): nothing will present it.
                after = std::min(in, pacer_.current_frame());
            } else {
                const uint32_t bound = timeout_ms == kInfinite
                                           ? kPresentWaitCapMs
                                           : std::min<uint32_t>(timeout_ms, kPresentWaitCapMs);
                if (wait_present_token(in, bound)) {
                    after = 0;
                    count(xbox::Counter::PresentWaitsSatisfied);
                } else {
                    after = std::min(in, pacer_.current_frame());
                    count(xbox::Counter::PresentWaitsUnpresented);
                }
            }
        }
        // A satisfied PRESENT token (after == 0) ends at once; every other wait ends on a tick, once per
        // tick per event kind (FramePacer::wait_event).
        const auto r = after == 0 ? pacer_.wait_after(0, timeout_ms, &frame)
                                  : pacer_.wait_event(type, timeout_ms, &frame);
        g_frame_waits.fetch_add(1, std::memory_order_relaxed);
        g_frame_wait_us.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                            std::chrono::steady_clock::now() - started)
                                                            .count()),
                                  std::memory_order_relaxed);
        if (r == xbox::FramePacer::WaitResult::Timeout) {
            g_frame_timeouts.fetch_add(1, std::memory_order_relaxed);
            count(xbox::Counter::FrameWaitTimeouts);
            // Only the origin wait with flag 2 gets a fresh token on a timeout; other waits leave it.
            if (type == 0 && (flags & 2) != 0) {
                token.write(std::max<uint64_t>(std::max<uint64_t>(frame, 1),
                                               last_frame_token_.load(std::memory_order_acquire)));
            }
            // A timed-out wait is not a failure: S_FALSE, as a WaitForSingleObject-style timeout
            // (titles trap on a failed HRESULT from this call).
            return kSFalse;
        }
    }
    // Always hand out a strictly monotonic, non-zero D3D12XBOX_FRAME_PIPELINE_TOKEN.
    const uint64_t floor_tok = std::max<uint64_t>(frame, 1);
    uint64_t prev = last_frame_token_.load(std::memory_order_relaxed);
    uint64_t out_tok = floor_tok;
    for (;;) {
        out_tok = std::max<uint64_t>(floor_tok, prev + 1);
        if (last_frame_token_.compare_exchange_weak(prev, out_tok, std::memory_order_acq_rel,
                                                    std::memory_order_relaxed)) {
            break;
        }
    }
    token.write(out_tok);
    return kSOk;
}

void Device::note_present_token(uint64_t token) {
    {
        std::lock_guard<std::mutex> lock(present_mutex_);
        uint64_t prev = last_present_token.load(std::memory_order_relaxed);
        if (token > prev) {
            last_present_token.store(token, std::memory_order_release);
        }
    }
    present_cv_.notify_all();
}

bool Device::wait_present_token(uint64_t token, uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(present_mutex_);
    auto presented = [&] {
        return present_shutdown_ || last_present_token.load(std::memory_order_acquire) >= token;
    };
    if (timeout_ms == kInfiniteFrameWait) {
        present_cv_.wait(lock, presented);
    } else if (!present_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), presented)) {
        return false;
    }
    return last_present_token.load(std::memory_order_acquire) >= token;
}

HResult Device::QueryFrameStatistics_unverified(uint64_t, uint32_t, GuestPtr<uint32_t> count_out, uint64_t) {
    // No statistics: the caller reads entries only when the count is non-zero.
    if (!count_out.is_null()) {
        count_out.write(0u);
    }
    return kSOk;
}

// ---------------------------------------------------------------------------------------------
// Device: no-op and refused slots
// ---------------------------------------------------------------------------------------------

HResult Device::Slot49_SetFeatureToggle_unverified(uint32_t, uint32_t) {
    count(xbox::Counter::FeatureToggles);
    return kSOk;
}

HResult Device::Slot50_RegisterRemovalCallbacks_unverified(uint64_t, uint64_t, uint64_t) {
    count(xbox::Counter::XboxNoOps);
    return kSOk;
}

HResult Device::Slot61_unverified(uint64_t, uint64_t) {
    count(xbox::Counter::XboxNoOps);
    return kSOk;
}

HResult Device::Slot119_unverified(uint64_t, uint32_t, uint32_t) {
    count(xbox::Counter::XboxNoOps);
    return kSOk;
}

HResult Device::Slot132_unverified(uint32_t) {
    count(xbox::Counter::XboxNoOps);
    return kSOk;
}

HResult Device::RegisterPagePool_unverified(uint64_t, uint32_t, GuestPtr<uint64_t> handle) {
    const uint64_t h = 0x9000 + counter(xbox::Counter::PagePools) + 1;
    count(xbox::Counter::PagePools);
    if (!handle.is_null()) {
        handle.write(h);
    }
    return kSOk;
}

HResult Device::CreateComputePipelineStateX(com::CallContext& ctx) {
    os::win32::HostStallScope stall;  // pipeline builds can block the guest thread
    // Slot 62: (desc, a, b, riid, out); desc is D3D12_COMPUTE_PIPELINE_STATE_DESC (DXIL compute
    // shader). a and b were always 0; other values are recorded and otherwise ignored.
    const uint64_t desc_va = ctx.int_arg(1);
    const uint64_t a = ctx.int_arg(2), b = ctx.int_arg(3);
    OutPtr out(ctx.int_arg(5));
    Guid riid{};
    GuestComputePipelineStateDesc g{};
    if (desc_va == 0 || !com::read_guest(desc_va, g) || !com::read_guest(ctx.int_arg(4), riid)) {
        note_bad_argument("CreateComputePipelineState (Xbox slot 62): unreadable descriptor or interface id");
        if (out) {
            out.write(0);
        }
        return kEInvalidArg;
    }
    if (a != 0 || b != 0) {
        note_bad_argument("CreateComputePipelineState (Xbox slot 62): unknown arguments " + hex(a) + ", " + hex(b) +
                          " ignored");
    }
    const HResult hr = create_compute_pso(g, riid, out);
    if (hr >= 0) {
        count(xbox::Counter::ComputePipelineStatesX);
    } else if (out) {
        out.write(0);  // the caller frees the bytecode and takes its failure path
    }
    return hr;
}

HResult Device::CreatePipelineStateFromBlob_unverified(com::CallContext& ctx) {
    // Slot 67: (&desc, riid, out), a console-native compute pipeline. When a valid native compute
    // descriptor is present, index its stage for the DXIL join and dump it when dump_dir is active,
    // then refuse with E_NOTIMPL and a cleared out pointer so the caller falls back to slot 62.
    count(xbox::Counter::BlobPipelineStatesRefused);
    const uint64_t desc_va = ctx.int_arg(1);
    const uint64_t out_va = ctx.int_arg(3);
    if (out_va != 0) {
        const uint64_t zero = 0;
        com::write_guest(out_va, &zero, sizeof(zero));
    }
    if (desc_va != 0) {
        xbox::GuestNativeComputePipelineDesc d{};
        bool readable = com::read_guest(desc_va, d);
        if (!readable) {
            // Compact 3-range form (0x38 bytes) near a page boundary.
            readable = com::read_guest(desc_va, &d, sizeof(d) - sizeof(xbox::GuestNativeRange));
            d.ranges[3] = {};
        }
        xbox::NativeComputePipeline p;
        std::string error;
        if (readable && xbox::parse_native_compute_pipeline(d, &p, &error)) {
            std::vector<uint8_t> code(static_cast<size_t>(p.compute.code.size));
            std::vector<uint8_t> meta(static_cast<size_t>(p.compute.metadata.size));
            const bool code_ok = com::read_guest(p.compute.code.address, code.data(), code.size());
            const bool meta_ok = meta.empty() || com::read_guest(p.compute.metadata.address, meta.data(), meta.size());
            if (code_ok && meta_ok) {
                const uint64_t code_id = shader_identity(code.data(), code.size());
                const uint64_t stage_id = native_stage_identity(code.data(), code.size(), meta.data(), meta.size());
                note_shader_size(stage_id, code.size());
                if (dump_) {
                    dump_->native_stage(code_id, stage_id, PipelineDump::NativeStageKind::Compute, /*layout_b=*/false,
                                        code.data(), code.size(), meta.data(), meta.size());
                }
                std::lock_guard<std::mutex> lock(shader_mutex_);
                if (indexed_native_.insert(stage_id).second) {
                    static const DxilDigest kZero{};
                    NativeStageRecord rec{};
                    rec.identity = stage_id;
                    rec.code_identity = code_id;
                    rec.code_bytes = static_cast<uint32_t>(code.size());
                    rec.metadata_bytes = static_cast<uint32_t>(meta.size());
                    for (size_t o = xbox::kNativeDigestFirst; o + 16 <= meta.size(); o += xbox::kNativeDigestStride) {
                        DxilDigest dig;
                        std::memcpy(dig.data(), meta.data() + o, 16);
                        if (dig != kZero) {
                            native_by_digest_.emplace(dig, rec);
                        }
                    }
                    if (meta.size() >= 28) {
                        DxilDigest dig;
                        std::memcpy(dig.data(), meta.data() + meta.size() - 28, 16);
                        if (dig != kZero) {
                            native_by_digest_.emplace(dig, rec);
                        }
                    }
                }
            }
        }
    }
    return kENotImpl;
}

HResult Device::CreateStateObject(uint64_t, const Guid&, OutPtr out) {
    count(xbox::Counter::StateObjectsRefused);
    if (out) {
        out.write(0);
    }
    return kENotImpl;
}

// ---------------------------------------------------------------------------------------------
// Device: pipeline states
// ---------------------------------------------------------------------------------------------

HResult Device::CreatePipelineStateFromStream(uint64_t stream_desc, const Guid& riid, OutPtr out) {
    xbox::GuestPipelineStateStreamDesc sd{};
    if (stream_desc == 0 || !com::read_guest(stream_desc, sd) || sd.SizeInBytes == 0 ||
        sd.SizeInBytes > kMaxStreamBytes || sd.pPipelineStateSubobjectStream == 0) {
        note_bad_argument("CreatePipelineState (stream): unreadable stream descriptor");
        if (out) {
            out.write(0);
        }
        return kEInvalidArg;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(sd.SizeInBytes));
    if (!com::read_guest(sd.pPipelineStateSubobjectStream, bytes.data(), bytes.size())) {
        note_bad_argument("CreatePipelineState (stream): unreadable stream at " + hex(sd.pPipelineStateSubobjectStream));
        if (out) {
            out.write(0);
        }
        return kEInvalidArg;
    }
    xbox::StreamPipeline p;
    std::string error;
    if (!xbox::parse_pipeline_stream(bytes.data(), bytes.size(), &p, &error)) {
        note_bad_argument("CreatePipelineState (stream): " + error);
        if (out) {
            out.write(0);
        }
        return kEInvalidArg;
    }
    count(xbox::Counter::StreamPipelineStates);
    switch (p.kind) {
    case xbox::StreamPipelineKind::Compute:
        return create_compute_pso(p.compute, riid, out);
    case xbox::StreamPipelineKind::Mesh:
        return create_mesh_pso(p.graphics, p.mesh, p.amplification, riid, out);
    case xbox::StreamPipelineKind::Graphics:
        break;
    }
    return create_graphics_pso(p.graphics, riid, out);
}

HResult Device::CreateNativePipelineState(uint64_t desc_va, const Guid& riid, OutPtr out) {
    xbox::GuestNativePipelineDesc d{};
    xbox::NativePipeline p;
    std::string error = "unreadable descriptor";
    if (desc_va == 0 || !com::read_guest(desc_va, d) || !xbox::parse_native_pipeline(d, &p, &error)) {
        count(xbox::Counter::NativePipelineStatesRejected);
        note_bad_argument("CreatePipelineState (native, Xbox slot 57): " + error);
        if (out) {
            out.write(0);
        }
        return kEInvalidArg;
    }
    if (options_.refuse_native_pipelines) {
        // Index the stages for the DXIL join, then fail: the title creates the pipeline through
        // slot 86 from DXIL when it is requested (see "Native pipelines and their DXIL").
        NativeStageRecord vs, ps;
        const bool ok = read_native_stage(p.vertex, true, &vs, false, p.vertex_alt) &&
                        read_native_stage(p.pixel, true, &ps, true, p.vertex_alt);
        count(ok ? xbox::Counter::NativePipelineStatesRefused : xbox::Counter::NativePipelineStatesRejected);
        if (out) {
            out.write(0);
        }
        return ok ? kENotImpl : kEInvalidArg;
    }
    const HResult hr = create_native_pso(p, riid, out);
    if (hr >= 0) {
        count(xbox::Counter::NativePipelineStates);
    } else {
        count(xbox::Counter::NativePipelineStatesRejected);
        if (out) {
            out.write(0);
        }
    }
    return hr;
}

HResult Device::CreatePipelineStateWithOverrides(InterfaceArg<IPipelineStateHost> base, uint32_t n, uint64_t overrides,
                                                 const Guid& riid, OutPtr out) {
    if (!base || n > kMaxOverrides || (n != 0 && overrides == 0)) {
        note_bad_argument("CreatePipelineStateWithOverrides: invalid base pipeline state or override array");
        if (out) {
            out.write(0);
        }
        return kEInvalidArg;
    }
    auto* state = static_cast<PipelineState*>(base.get());
    const GraphicsPsoSource* source = state->graphics_source();
    if (state->native() != nullptr) {
        // Console-native base: its state is an undecoded register image, so the overrides cannot
        // be applied; the base (same stages) is handed out and the entries are counted.
        count(xbox::Counter::NativeOverridesIgnored, n);
        return hand_out(com::Ref<PipelineState>::share(state), riid, out);
    }
    if (source == nullptr) {
        // Compute or placeholder base: nothing to override, hand out the base itself.
        note_bad_argument("CreatePipelineStateWithOverrides: base is not a graphics pipeline (base returned)");
        return hand_out(com::Ref<PipelineState>::share(state), riid, out);
    }
    GraphicsPsoSource derived = *source;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t entry = overrides + uint64_t{xbox::kPsoOverrideStride} * i;
        uint32_t kind = 0;
        if (!com::read_guest(entry, &kind, sizeof(kind))) {
            note_bad_argument("CreatePipelineStateWithOverrides: unreadable override " + std::to_string(i));
            if (out) {
                out.write(0);
            }
            return kEInvalidArg;
        }
        bool ok = true;
        switch (static_cast<xbox::PsoOverrideKind>(kind)) {
        case xbox::PsoOverrideKind::DepthStencil: {
            GuestDepthStencilDesc g{};
            ok = com::read_guest(entry + 4, g);
            convert_depth_stencil_desc(g, &derived.desc.DepthStencilState);
            break;
        }
        case xbox::PsoOverrideKind::Rasterizer: {
            GuestRasterizerDesc g{};
            ok = com::read_guest(entry + 4, g);
            convert_rasterizer_desc(g, &derived.desc.RasterizerState);
            break;
        }
        case xbox::PsoOverrideKind::Blend: {
            GuestBlendDesc g{};
            ok = com::read_guest(entry + 4, g);
            convert_blend_desc(g, &derived.desc.BlendState);
            break;
        }
        default:
            count(xbox::Counter::UnknownOverrideKinds);  // kind 5 (and anything else): ignored
            break;
        }
        if (!ok) {
            note_bad_argument("CreatePipelineStateWithOverrides: unreadable override " + std::to_string(i));
            if (out) {
                out.write(0);
            }
            return kEInvalidArg;
        }
    }
    count(xbox::Counter::OverridePipelineStates);
    return create_graphics_pso_from(std::move(derived), riid, out);
}

// ---------------------------------------------------------------------------------------------
// Device: views at GPU addresses, allocation info, alias windows
// ---------------------------------------------------------------------------------------------

namespace {

// Element size of a buffer view (structured: stride; raw: 4; typed: the format's size).
uint32_t buffer_element_size(uint32_t format, uint32_t stride, bool raw) {
    if (stride != 0) {
        return stride;
    }
    if (raw) {
        return 4;
    }
    const DxgiFormatLayout l = dxgi_format_layout(format);
    return l.known && l.block_width == 1 ? l.block_bytes : 0;
}

}  // namespace

namespace {

// The registry record a view description produces, for the descriptor words (resource 0).
d3d::ViewDescriptor words_record(d3d::DescriptorKind kind, const d3d::ViewDesc& vd) {
    d3d::ViewDescriptor v{};
    v.kind = kind;
    v.format = vd.format;
    v.dimension = static_cast<uint8_t>(vd.dimension);
    v.flags = static_cast<uint16_t>(vd.flags);
    v.first_element_or_address = vd.first_element;
    v.num_elements_or_size = vd.num_elements;
    v.structure_byte_stride = vd.structure_byte_stride;
    v.component_mapping = kind == d3d::DescriptorKind::Srv ? vd.component_mapping : 0u;
    v.first_mip = static_cast<uint16_t>(std::min<uint32_t>(vd.first_mip, 0xffffu));
    v.mip_levels = kind == d3d::DescriptorKind::Srv ? static_cast<uint16_t>(std::min<uint32_t>(vd.mip_levels, 0xffffu))
                                                    : uint16_t{1};
    v.first_array_slice = vd.first_array_slice;
    v.array_size = vd.array_size;
    v.plane_slice = vd.plane_slice;
    v.min_lod_clamp = kind == d3d::DescriptorKind::Srv ? vd.min_lod_clamp : 0.0f;
    return v;
}

}  // namespace

void Device::view_at_address(bool uav, uint64_t resource_desc, const d3d::ViewDesc& desc, uint64_t va,
                             uint64_t dest) {
    const char* what = uav ? "CreateUnorderedAccessView (Xbox slot 71)" : "CreateShaderResourceView (Xbox slot 70)";
    const d3d::DescriptorKind kind = uav ? d3d::DescriptorKind::Uav : d3d::DescriptorKind::Srv;
    uint32_t words[bridge::kViewSlotBytes / 4] = {};
    d3d::ViewDesc vd = desc;
    uint32_t id = 0;
    bool has_texture_view_info = false;
    xbox::AddressTextureViewInfo tex_view_info{};
    if (vd.dimension == d3d::ViewDimension::Buffer) {
        bridge::buffer_words_at_address(va, words_record(kind, desc), words);
        const bool raw = (vd.flags & d3d::BufferViewFlags::Raw) != 0;
        const uint32_t elem = buffer_element_size(vd.format, vd.structure_byte_stride, raw);
        d3d::GpuVaResolution res;
        const bool resolved = state_bridge().resolve_gpu_va(va, &res);
        if (resolved && elem != 0 && res.offset % elem == 0) {
            id = res.resource_id;
            vd.first_element += res.offset / elem;
        } else {
            const uint32_t safe_elem = elem != 0 ? elem : 1u;
            const uint64_t view_va = va + vd.first_element * uint64_t{safe_elem};
            uint64_t raw_bytes = vd.num_elements != 0
                                     ? uint64_t{vd.num_elements} * uint64_t{safe_elem}
                                     : (uint64_t{64u} << 10);
            if (vd.num_elements == 0 && resolved && res.resource_size > res.offset) {
                raw_bytes = std::min<uint64_t>(raw_bytes, res.resource_size - res.offset);
            }
            const uint64_t byte_size = std::clamp<uint64_t>(raw_bytes, uint64_t{4u}, uint64_t{16u} << 20);
            const bool is_typed = vd.structure_byte_stride == 0 && !raw;
            const bool require_zero_offset = is_typed || elem == 0 || (uint64_t{256u} % elem != 0);
            uint32_t reloc_id = 0;
            uint64_t reloc_offset = 0;
            if (relocate_storage_buffer(view_va, byte_size, &reloc_id, &reloc_offset, require_zero_offset) &&
                (elem == 0 || reloc_offset % elem == 0)) {
                id = reloc_id;
                vd.first_element = elem != 0 ? reloc_offset / elem : uint64_t{0};
            } else if (resolved) {
                note_bad_argument(std::string(what) + ": address " + hex(va) +
                                  " is not element-aligned inside its buffer and could not be relocated (null descriptor written)");
            } else {
                note_bad_argument(std::string(what) + ": unknown GPU address " + hex(va) + " (null descriptor written)");
            }
        }
        count(xbox::Counter::ViewsAtAddress);
    } else {
        // A texture described only by its RESOURCE_DESC and address: no resource object exists.
        GuestResourceDesc rd{};
        if (resource_desc == 0 || !com::read_guest(resource_desc, rd)) {
            note_bad_argument(std::string(what) + ": unreadable resource description for a texture view");
            return;
        }
        bridge::AddressTexture t;
        t.dimension = rd.Dimension;
        t.width = rd.Width;
        t.height = rd.Height;
        t.depth_or_array_size = rd.DepthOrArraySize;
        uint32_t full = 1;
        for (uint64_t extent = std::max<uint64_t>({rd.Width, rd.Height, rd.Dimension == 4 ? rd.DepthOrArraySize : 1u});
             extent > 1; extent >>= 1) {
            ++full;
        }
        t.mip_levels = rd.MipLevels == 0 ? full : rd.MipLevels;
        t.format = rd.Format;
        t.sample_count = rd.SampleCount;
        const bridge::WordsStatus ws = bridge::image_words_at_address(va, t, words_record(kind, desc), words);
        tex_view_info.gpu_va = va;
        tex_view_info.dxgi_format = desc.format != 0 ? desc.format : rd.Format;
        tex_view_info.width = static_cast<uint32_t>(std::min<uint64_t>(rd.Width, uint64_t{0xFFFFFFFFu}));
        tex_view_info.height = rd.Height;
        tex_view_info.depth = rd.DepthOrArraySize == 0 ? 1u : static_cast<uint32_t>(rd.DepthOrArraySize);
        tex_view_info.mip_levels = t.mip_levels;
        tex_view_info.d3d12_layout = rd.Layout;
        tex_view_info.is_uav = uav;
        has_texture_view_info = true;
        if (rd.Dimension != ResourceDimension::Buffer && rd.SampleCount <= 1) {
            const GpuTextureDimension tex_dim =
                rd.Dimension == ResourceDimension::Texture1D
                    ? GpuTextureDimension::Dim1D
                    : (rd.Dimension == ResourceDimension::Texture3D ? GpuTextureDimension::Dim3D
                                                                    : GpuTextureDimension::Dim2D);
            id = ensure_address_texture(va, tex_view_info.dxgi_format, tex_view_info.width, tex_view_info.height,
                                        tex_view_info.depth, tex_view_info.mip_levels, tex_view_info.d3d12_layout,
                                        tex_dim, uav);
        }
        if (count_texture_view_at_address() == 1) {
            os::log(std::string("d3d12_xs: ") + what + ": first texture view at an address (" + hex(va) + ", format " +
                    std::to_string(rd.Format) + ", " + std::to_string(rd.Width) + "x" + std::to_string(rd.Height) +
                    ", layout " + std::to_string(rd.Layout) + "): image words written (" + bridge::words_status_name(ws) +
                    (id != 0 ? "), GPU texture id " + std::to_string(id) : "), no GPU texture"));
        }
    }
    const d3d::Status s = uav ? heaps_.create_unordered_access_view(id, 0, &vd, d3d::CpuDescriptorHandle{dest})
                              : heaps_.create_shader_resource_view(id, &vd, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string(what) + ": " + d3d::status_name(s));
        return;
    }
    if (has_texture_view_info) {
        xbox::record_address_texture_view(tex_view_info);
    }
    store_descriptor_words(dest, words, bridge::kViewSlotBytes);
}

uint64_t Device::count_texture_view_at_address() {
    count(xbox::Counter::TextureViewsAtAddress);
    return counter(xbox::Counter::TextureViewsAtAddress);
}

void Device::CreateShaderResourceViewAtAddress(uint64_t resource_desc, uint64_t view_desc, uint64_t dest) {
    GuestShaderResourceViewDesc g{};
    uint64_t va = 0;
    if (view_desc == 0 || !com::read_guest(view_desc, g) ||
        !com::read_guest(view_desc + xbox::kViewAtAddressOffset, &va, sizeof(va))) {
        note_bad_argument("CreateShaderResourceView (Xbox slot 70): unreadable descriptor");
        return;
    }
    d3d::ViewDesc vd;
    if (!parse_srv_desc(g, &vd)) {
        note_bad_argument("CreateShaderResourceView (Xbox slot 70): invalid view dimension " +
                          std::to_string(g.ViewDimension));
        return;
    }
    view_at_address(false, resource_desc, vd, va, dest);
}

void Device::CreateUnorderedAccessViewAtAddress(uint64_t resource_desc, uint64_t view_desc, uint64_t dest) {
    GuestUnorderedAccessViewDesc g{};
    uint64_t va = 0;
    if (view_desc == 0 || !com::read_guest(view_desc, g) ||
        !com::read_guest(view_desc + xbox::kViewAtAddressOffset, &va, sizeof(va))) {
        note_bad_argument("CreateUnorderedAccessView (Xbox slot 71): unreadable descriptor");
        return;
    }
    d3d::ViewDesc vd;
    if (!parse_uav_desc(g, &vd)) {
        note_bad_argument("CreateUnorderedAccessView (Xbox slot 71): invalid view dimension " +
                          std::to_string(g.ViewDimension));
        return;
    }
    view_at_address(true, resource_desc, vd, va, dest);
}

GuestResourceAllocationInfo Device::GetResourceAllocationInfo1(uint32_t visible_mask, uint32_t n, uint64_t descs,
                                                               uint64_t infos) {
    const GuestResourceAllocationInfo total = GetResourceAllocationInfo(visible_mask, n, descs);
    if (infos == 0 || total.SizeInBytes == UINT64_MAX || n > kMaxAllocationInfos) {
        return total;
    }
    uint64_t offset = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const GuestResourceAllocationInfo one =
            GetResourceAllocationInfo(visible_mask, 1, descs + sizeof(GuestResourceDesc) * i);
        offset = align_up(offset, one.Alignment);
        const xbox::GuestResourceAllocationInfo1 info{offset, one.Alignment, one.SizeInBytes};
        if (!com::write_guest(infos + sizeof(info) * i, info)) {
            note_bad_argument("GetResourceAllocationInfo1: unwritable info array");
            break;
        }
        offset += one.SizeInBytes;
    }
    return total;
}

void Device::register_gpu_alias(uint64_t gpu_base, uint64_t cpu_base, uint64_t size) {
    std::lock_guard<std::mutex> lock(alias_mutex_);
    for (const AliasWindow& w : aliases_) {
        if (w.gpu_base == gpu_base && w.cpu_base == cpu_base && w.size == size) {
            return;
        }
    }
    aliases_.push_back(AliasWindow{gpu_base, cpu_base, size});
}

uint64_t Device::cpu_address_for_gpu(uint64_t gpu_va) const {
    std::lock_guard<std::mutex> lock(alias_mutex_);
    for (const AliasWindow& w : aliases_) {
        if (gpu_va >= w.gpu_base && gpu_va - w.gpu_base < w.size) {
            return w.cpu_base + (gpu_va - w.gpu_base);
        }
    }
    return gpu_va;
}

// ---------------------------------------------------------------------------------------------
// Allocator and queue
// ---------------------------------------------------------------------------------------------

HResult CommandAllocator::SetAllocationCallback_unverified(uint64_t callback, uint64_t) {
    callback_.store(callback);
    return kSOk;
}

HResult CommandQueue::SuspendX(uint32_t) {
    suspended_.store(true);
    device_->count(xbox::Counter::Suspends);
    return kSOk;
}

HResult CommandQueue::ResumeX() {
    suspended_.store(false);
    device_->count(xbox::Counter::Resumes);
    return kSOk;
}

HResult CommandQueue::Slot31_UpdatePageMappings_unverified(com::CallContext& ctx) {
    // (range count, ranges, page-pool address, pool pages, 0); range = {u64 GPU-window address,
    // u32 entry count, u64 entries}, entry = {u32 kind, u32 pool page, u32 page count}
    // (d3d12_xs_xbox.h). Kinds: 0 consecutive pool pages, 1 one pool page repeated, 2 unmap,
    // 3 leave as is. Applied to the XMem GPU page table (os/xmem_windows.h).
    const uint64_t count = static_cast<uint32_t>(ctx.int_arg(1));
    const uint64_t ranges_va = ctx.int_arg(2);
    const uint64_t pool_va = ctx.int_arg(3);
    const uint64_t pool_pages = static_cast<uint32_t>(ctx.int_arg(4));
    constexpr uint64_t kMaxRanges = 1u << 16, kMaxEntries = 1u << 20;
    if (count > kMaxRanges || (count != 0 && ranges_va == 0)) {
        device_->note_bad_argument("page-mapping update (queue slot 31): bad range array");
        return kEInvalidArg;
    }
    std::vector<os::xmem_windows::PageMappingRange> ranges(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        struct {
            uint64_t gpu_va;
            uint32_t entries;
            uint32_t pad;
            uint64_t entries_va;
        } r{};
        static_assert(sizeof(r) == 0x18);
        if (!com::read_guest(ranges_va + sizeof(r) * i, &r, sizeof(r)) || r.entries > kMaxEntries ||
            (r.entries != 0 && r.entries_va == 0)) {
            device_->note_bad_argument("page-mapping update (queue slot 31): unreadable range " + std::to_string(i));
            return kEInvalidArg;
        }
        ranges[i].gpu_va = r.gpu_va;
        ranges[i].entries.resize(r.entries);
        for (uint32_t e = 0; e < r.entries; ++e) {
            uint32_t words[3] = {};
            if (!com::read_guest(r.entries_va + sizeof(words) * e, words, sizeof(words))) {
                device_->note_bad_argument("page-mapping update (queue slot 31): unreadable entry");
                return kEInvalidArg;
            }
            ranges[i].entries[e] = {words[0], words[1], words[2]};
        }
    }
    const auto result = os::xmem_windows::apply_gpu_page_mappings(ranges, pool_va, pool_pages);
    device_->count(xbox::Counter::PageMappingsApplied);
    if (result.outside_window != 0 || result.skipped_entries != 0) {
        device_->note_bad_argument("page-mapping update (queue slot 31): " + std::to_string(result.outside_window) +
                                   " pages outside a GPU window, " + std::to_string(result.skipped_entries) +
                                   " unknown entries");
    }
    return kSOk;
}

HResult CommandQueue::PresentX(uint32_t plane_count, uint64_t planes, uint64_t parameters) {
    if (plane_count == 0) {
        device_->count(xbox::Counter::PresentsWithoutPlanes);
        return kSOk;
    }
    xbox::GuestPresentPlaneX plane{};
    if (plane_count > kMaxPlanes || planes == 0 || !com::read_guest(planes, plane)) {
        device_->note_bad_argument("PresentX: unreadable plane array (" + std::to_string(plane_count) + " planes)");
        return kEInvalidArg;
    }
    // Every plane's texture is drawn into the canvas at the Present (Device::present), every
    // frame: titles alternate back buffers whose render-target views predate the present, and
    // compose the frame from several planes (e.g. a 4K scene plane under a 1080p overlay plane).
    // Each plane contributes its first resource, full screen.
    std::array<uint32_t, kMaxPlanes> textures{};
    uint64_t max_token = plane.Token;
    for (uint32_t i = 0; i < plane_count; ++i) {
        xbox::GuestPresentPlaneX p{};
        uint64_t resource = 0;
        if (!com::read_guest(planes + uint64_t{i} * sizeof(p), p)) {
            device_->note_bad_argument("PresentX: plane " + std::to_string(i) + " descriptor is unreadable");
            continue;
        }
        if (p.Token > max_token) {
            max_token = p.Token;
        }
        if (p.ResourceCount == 0 || p.ppResources == 0 ||
            !com::read_guest(p.ppResources, &resource, sizeof(resource)) || resource == 0) {
            device_->note_bad_argument("PresentX: plane " + std::to_string(i) + " has no readable resource");
            continue;
        }
        com::detail::Resolved r;
        if (!com::detail::lookup_interface(resource, &r) || r.iface != &kID3D12Resource.info) {
            device_->note_bad_argument("PresentX: plane " + std::to_string(i) + " resource " + hex(resource) +
                                       " is not a resource");
            continue;
        }
        auto* res = static_cast<Resource*>(static_cast<IResourceHost*>(r.host));
        if (res->desc().Dimension != ResourceDimension::Texture2D || !res->has_gpu_texture()) {
            device_->note_bad_argument("PresentX: plane " + std::to_string(i) + " resource " + hex(resource) +
                                       " is not a 2D texture with a GPU object");
            continue;
        }
        textures[i] = res->id();
    }
    device_->count(xbox::Counter::PresentsWithPlanes);
    if (plane_count > 1) {
        device_->count(xbox::Counter::MultiPlanePresents);
    }
    const uint32_t drawn = device_->present(std::span<const uint32_t>(textures.data(), plane_count));
    // The frame is done on our side once its Present has been handed off (its submissions and their
    // CPU-visible buffer snapshots came before): PRESENT waits for this token may complete now.
    if (max_token != 0) {
        device_->note_present_token(max_token);
    }
    device_->on_frame_presented();  // fence publication epoch (DeviceOptions::fence_publish_lag)
    device_->count(xbox::Counter::PresentPlanesDrawn, drawn);
    device_->count(xbox::Counter::PresentPlanesNotDrawn, plane_count - drawn);
    // O36: the flip waits for the title's sync interval on the 60 Hz display (gpu/present_pacing.h):
    // no wait when the present parameters ask for an immediate flip (their first member, a BOOL),
    // otherwise the frame interval the title set (slot 78) in vblanks, at least one.
    uint32_t immediate = 0;
    if (parameters != 0 && !com::read_guest(parameters, &immediate, sizeof(immediate))) {
        immediate = 0;
    }
    uint32_t interval = immediate != 0 ? 0u : device_->present_interval_vblanks();
    const uint32_t max_interval = g_present_max_interval.load(std::memory_order_relaxed);
    if (max_interval > 0) {
        interval = std::min(interval, max_interval);
    }
    present_pacer().present(interval);
    return kSOk;
}

// ---------------------------------------------------------------------------------------------
// Command list
// ---------------------------------------------------------------------------------------------

void CommandList::initialize_guest_payload() {
    std::vector<uint8_t> zeros(static_cast<size_t>(kComPayloadSize), 0);
    com::write_guest(payload_address(), zeros.data(), zeros.size());
}

void CommandList::SetPipelineStateX(com::CallContext& ctx) {
    if (!usable()) {
        return;
    }
    // The executable stages the PSO's 32-byte payload at list + 0x544 and passes only `this`.
    uint64_t staged[2] = {};
    const uint64_t staged_va = identity() + xbox::kListStagedPipelineOffset;
    uint64_t pointer = 0;
    if (com::read_guest(staged_va, staged, sizeof(staged)) && staged[0] == xbox::kPipelineStateTag) {
        pointer = staged[1];
        const uint64_t zero = 0;
        com::write_guest(staged_va, &zero, sizeof(zero));  // consumed: a later explicit call uses rdx
    } else {
        pointer = ctx.int_arg(1);
    }
    auto state = InterfaceArg<IPipelineStateHost>::resolve(pointer);
    if (!state) {
        note("SetPipelineState: no staged pipeline state and invalid argument " + hex(pointer));
        return;
    }
    bind_pipeline(static_cast<PipelineState*>(state.get()));
}

void CommandList::ExecuteIndirect(com::CallContext& ctx) {
    // (signature, max command count, argument buffer, argument offset, count buffer, count offset)
    if (!usable()) {
        return;
    }
    const auto skip = [&](xbox::Counter counter) {
        device_->count(counter);
        count_work(Work::IndirectExecutionsSkipped);
        device_->note_skipped_draw();
    };
    auto signature_arg = InterfaceArg<ICommandSignatureHost>::resolve(ctx.int_arg(1));
    const auto* signature = signature_arg ? static_cast<const CommandSignature*>(signature_arg.get()) : nullptr;
    if (signature == nullptr) {
        skip(xbox::Counter::IndirectSkipped);
        return;
    }
    bridge::IndirectExecution e;
    switch (signature->work_type()) {
    case CommandSignature::kIndirectArgDraw: e.kind = bridge::IndirectKind::Draw; break;
    case CommandSignature::kIndirectArgDrawIndexed: e.kind = bridge::IndirectKind::DrawIndexed; break;
    case CommandSignature::kIndirectArgDispatch: e.kind = bridge::IndirectKind::Dispatch; break;
    case CommandSignature::kIndirectArgDispatchMesh:
        // G66: the bound graphics pipeline must be an emulated mesh pipeline.
        if (graphics_pso_ == nullptr || graphics_pso_->mesh() == nullptr || !graphics_pso_->resolved()) {
            skip(xbox::Counter::IndirectMeshSkipped);
            return;
        }
        e.kind = bridge::IndirectKind::DispatchMesh;
        break;
    case CommandSignature::kIndirectArgXbox105:
        // Xbox-only kind (meaning not established; read as a single-pass dispatch with a 12-byte
        // group-count record): a mesh dispatch when an emulated mesh pipeline is bound, otherwise a
        // compute dispatch.
        if (graphics_pso_ != nullptr && graphics_pso_->mesh() != nullptr && graphics_pso_->resolved()) {
            e.kind = bridge::IndirectKind::DispatchMesh;
        } else if (compute_pso_ != nullptr) {
            e.kind = bridge::IndirectKind::Dispatch;
        } else {
            skip(xbox::Counter::IndirectType105Skipped);
            return;
        }
        break;
    default:
        skip(xbox::Counter::IndirectSkipped);
        return;
    }
    Resource* args = resource(InterfaceArg<IResourceHost>::resolve(ctx.int_arg(3)));
    const uint64_t count_pointer = ctx.int_arg(5);
    Resource* count = count_pointer != 0 ? resource(InterfaceArg<IResourceHost>::resolve(count_pointer)) : nullptr;
    const bool graphics = e.kind != bridge::IndirectKind::Dispatch;
    PipelineState* pso = graphics ? graphics_pso_ : compute_pso_;
    // Only single-argument signatures (no root argument / view changes per command) are expressed.
    const uint32_t min_stride =
        e.kind == bridge::IndirectKind::DrawIndexed
            ? 20u
            : (e.kind == bridge::IndirectKind::Draw ? 16u : 12u);
    const uint64_t args_offset = ctx.int_arg(4);
    const uint64_t count_offset = count != nullptr ? ctx.int_arg(6) : 0u;
    if (signature->argument_count() != 1 || signature->byte_stride() < min_stride || args == nullptr ||
        !args->is_buffer() || (args_offset & 3u) != 0 ||
        (count_pointer != 0 && (count == nullptr || !count->is_buffer() || (count_offset & 3u) != 0)) ||
        (pso != nullptr && !pso->resolved()) || (graphics && rect_list_)) {
        skip(xbox::Counter::IndirectSkipped);
        return;
    }
    e.max_count = static_cast<uint32_t>(ctx.int_arg(2));
    if (e.max_count == 0) {
        device_->count(xbox::Counter::IndirectZeroCount);
        return;
    }
    e.stride = signature->byte_stride();
    e.args_buffer = args->id();
    e.args_offset = args_offset;
    e.count_buffer = count != nullptr ? count->id() : 0u;
    e.count_offset = count_offset;
    pinned_.push_back(com::Ref<com::ComObject>::share(args));
    if (count != nullptr) {
        pinned_.push_back(com::Ref<com::ComObject>::share(count));
    }
    if (!bridge_.ExecuteIndirect(e)) {
        skip(e.kind == bridge::IndirectKind::DispatchMesh ? xbox::Counter::IndirectMeshSkipped
             : signature->work_type() == CommandSignature::kIndirectArgXbox105 ? xbox::Counter::IndirectType105Skipped
                                                                              : xbox::Counter::IndirectSkipped);
        return;
    }
    device_->count(xbox::Counter::IndirectExecuted);
    count_work(Work::IndirectExecutionsEmitted);
}

void CommandList::OMSetDepthBounds(float min_depth, float max_depth) {
    if (!usable()) {
        return;
    }
    device_->count(xbox::Counter::DepthBoundsIgnored);  // no depth-bounds test in WebGPU
    if (!std::isfinite(min_depth) || !std::isfinite(max_depth) || min_depth > max_depth) {
        note("OMSetDepthBounds: invalid depth range");
        return;
    }
    xbox::CommandListXboxStateTable& t = xbox::command_list_xbox_state();
    std::lock_guard<std::mutex> lock(t.mutex);
    t.depth_bounds[identity()] = xbox::DepthBoundsState{min_depth, max_depth, true};
}

void CommandList::RSSetShadingRate(uint32_t rate, uint64_t combiners) {
    if (!usable()) {
        return;
    }
    device_->count(xbox::Counter::ShadingRateIgnored);  // full-rate shading
    if (!xbox::is_valid_shading_rate(rate)) {
        note("RSSetShadingRate: invalid shading rate " + hex(rate));
        return;
    }
    uint32_t comb[2] = {0, 0};
    if (combiners != 0) {
        if (!com::read_guest(combiners, comb, sizeof(comb))) {
            note("RSSetShadingRate: unreadable combiner array at " + hex(combiners));
            return;
        }
        if (!xbox::is_valid_shading_rate_combiner(comb[0]) || !xbox::is_valid_shading_rate_combiner(comb[1])) {
            note("RSSetShadingRate: invalid combiner values (" + std::to_string(comb[0]) + ", " +
                 std::to_string(comb[1]) + ")");
            return;
        }
    }
    xbox::CommandListXboxStateTable& t = xbox::command_list_xbox_state();
    std::lock_guard<std::mutex> lock(t.mutex);
    xbox::ShadingRateState& s = t.shading_rate[identity()];
    s.base_rate = rate;
    s.combiners[0] = comb[0];
    s.combiners[1] = comb[1];
    s.rate_set = true;
}

void CommandList::RSSetShadingRateImage(InterfaceArg<IResourceHost> image) {
    if (!usable()) {
        return;
    }
    device_->count(xbox::Counter::ShadingRateIgnored);
    if (image.va() != 0 && !image) {
        note("RSSetShadingRateImage: invalid resource " + hex(image.va()));
        return;
    }
    xbox::CommandListXboxStateTable& t = xbox::command_list_xbox_state();
    std::lock_guard<std::mutex> lock(t.mutex);
    xbox::ShadingRateState& s = t.shading_rate[identity()];
    s.image_va = image.va();
    s.image_set = true;
}

void CommandList::note_xbox_slot(uint32_t slot) {
    ++xbox_calls_[slot];
    device_->count(xbox::Counter::XboxNoOps);
}

std::vector<std::pair<uint32_t, uint64_t>> CommandList::xbox_slot_calls() const {
    return {xbox_calls_.begin(), xbox_calls_.end()};
}

}  // namespace playgta6::gpu::d3d12_guest
