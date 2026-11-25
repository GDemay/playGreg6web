#include "gpu/frame_capture.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include "os/os_shim.h"

namespace playgta6::gpu {

namespace {

struct FrameRecord {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t packets = 0;
};

struct PresentSample {
    double wall_ms = 0;
    FrameCaptureSample counters;
};

struct Omitted {
    uint64_t packet = 0;
    uint16_t opcode = 0;
    uint32_t length = 0;
    uint64_t hash = 0;
};

// A write held during a deferred capture's prefix.
struct HeldTextureWrite {
    uint64_t seq = 0;
    GpuWriteTexturePayload box{};
    std::vector<uint8_t> packet;  // the whole packet (header + payload + data)
};

struct HeldBuffer {
    uint64_t seq = 0;  // of the newest write
    uint64_t lo = UINT64_MAX;
    uint64_t hi = 0;
    std::vector<uint8_t> image;  // bytes [0, hi); never-written bytes are 0 like a fresh buffer
};

struct CaptureState {
    std::mutex mutex;
    bool prefix = false;                     // deferred and not started yet
    std::atomic<bool> begin_requested{false};
    uint64_t first_present = 0;
    uint64_t held_seq = 0;
    std::map<uint32_t, HeldBuffer> held_buffers;
    std::map<uint32_t, std::vector<HeldTextureWrite>> held_textures;
    bool open = false;    // a capture runs (until finish)
    bool active = false;  // still writing packets
    FrameCaptureConfig config;
    std::FILE* file = nullptr;
    uint64_t bytes = 0;
    uint64_t packets = 0;
    FrameRecord current;
    std::vector<FrameRecord> frames;
    std::vector<PresentSample> presents;
    std::vector<Omitted> omitted;
    uint64_t omitted_bytes = 0;
    uint64_t thinned_packets = 0;
    uint64_t thinned_bytes = 0;
    std::map<uint64_t, uint64_t> big_writes;  // (opcode, id) -> large writes seen
    std::chrono::steady_clock::time_point start;
    std::atomic<FrameCaptureCounters> counters{nullptr};
};

CaptureState& state() {
    static CaptureState s;
    return s;
}

uint64_t fnv1a64(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// Offset of the u32 length of the trailing data in the payload, or -1.
int data_length_offset(uint16_t opcode) {
    switch (static_cast<GpuOpcode>(opcode)) {
        case GpuOpcode::WriteBuffer:
            return static_cast<int>(offsetof(GpuWriteBufferPayload, data_len));
        case GpuOpcode::CreateShaderModule:
            return static_cast<int>(offsetof(GpuCreateShaderModulePayload, code_len));
        case GpuOpcode::WriteTexture:
            return static_cast<int>(offsetof(GpuWriteTexturePayload, data_len));
        default:
            return -1;
    }
}

size_t fixed_payload_size(uint16_t opcode) {
    switch (static_cast<GpuOpcode>(opcode)) {
        case GpuOpcode::WriteBuffer:
            return sizeof(GpuWriteBufferPayload);
        case GpuOpcode::CreateShaderModule:
            return sizeof(GpuCreateShaderModulePayload);
        case GpuOpcode::WriteTexture:
            return sizeof(GpuWriteTexturePayload);
        default:
            return 0;
    }
}

void write_bytes(CaptureState& s, const void* p, size_t n) {
    if (n != 0 && std::fwrite(p, 1, n, s.file) != n) {
        s.active = false;  // disk error: stop writing, the sidecar still reports what was written
    }
    s.bytes += n;
    s.current.bytes += n;
}

void sample_present(CaptureState& s) {
    PresentSample p;
    p.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s.start).count();
    if (FrameCaptureCounters c = s.counters.load(std::memory_order_acquire)) {
        c(&p.counters);
    }
    s.presents.push_back(p);
}

void write_packet(CaptureState& s, const GpuPacketHeader& h, const uint8_t* packet) {
    const uint8_t* payload = packet + sizeof(h);
    const int len_off = data_length_offset(h.opcode);
    const size_t fixed = fixed_payload_size(h.opcode);
    if (!s.config.keep_payloads && len_off >= 0 && h.payload_bytes > fixed) {
        uint32_t data_len = 0;
        std::memcpy(&data_len, payload + len_off, sizeof(data_len));
        const size_t tail = h.payload_bytes - fixed;
        s.omitted.push_back(Omitted{s.packets, h.opcode, data_len, fnv1a64(payload + fixed, tail)});
        s.omitted_bytes += tail;
        GpuPacketHeader out = h;
        out.payload_bytes = static_cast<uint32_t>(fixed);
        std::vector<uint8_t> head(payload, payload + fixed);
        const uint32_t zero = 0;
        std::memcpy(head.data() + len_off, &zero, sizeof(zero));
        write_bytes(s, &out, sizeof(out));
        write_bytes(s, head.data(), head.size());
    } else if (s.config.thin_uploads > 1 && len_off >= 0 && h.payload_bytes > fixed &&
               (h.opcode == static_cast<uint16_t>(GpuOpcode::WriteBuffer) ||
                h.opcode == static_cast<uint16_t>(GpuOpcode::WriteTexture)) &&
               h.payload_bytes - fixed > s.config.thin_min_bytes) {
        uint32_t id = 0;  // id / texture_id is the first field of both payloads
        std::memcpy(&id, payload, sizeof(id));
        const uint64_t n = s.big_writes[(static_cast<uint64_t>(h.opcode) << 32) | id]++;
        if (n % s.config.thin_uploads == 0) {
            write_bytes(s, packet, sizeof(h) + h.payload_bytes);
        } else {
            s.thinned_packets++;
            s.thinned_bytes += h.payload_bytes - fixed;
            GpuPacketHeader out = h;
            out.payload_bytes = static_cast<uint32_t>(fixed);
            std::vector<uint8_t> head(payload, payload + fixed);
            const uint32_t zero = 0;
            std::memcpy(head.data() + len_off, &zero, sizeof(zero));
            std::memcpy(head.data() + fixed - sizeof(uint32_t), &kThinUploadMarker, sizeof(uint32_t));  // reserved
            write_bytes(s, &out, sizeof(out));
            write_bytes(s, head.data(), head.size());
        }
    } else {
        write_bytes(s, packet, sizeof(h) + h.payload_bytes);
    }
    ++s.packets;
    ++s.current.packets;
}

bool box_contains(const GpuWriteTexturePayload& outer, const GpuWriteTexturePayload& inner) {
    return outer.mip_level == inner.mip_level && outer.origin_x <= inner.origin_x && outer.origin_y <= inner.origin_y &&
           outer.origin_z <= inner.origin_z &&
           uint64_t(outer.origin_x) + outer.width >= uint64_t(inner.origin_x) + inner.width &&
           uint64_t(outer.origin_y) + outer.height >= uint64_t(inner.origin_y) + inner.height &&
           uint64_t(outer.origin_z) + outer.depth_or_layers >= uint64_t(inner.origin_z) + inner.depth_or_layers;
}

// A packet of the deferred prefix: creation/destruction written, writes held, commands dropped.
void prefix_packet(CaptureState& s, const GpuPacketHeader& h, const uint8_t* packet) {
    const uint8_t* payload = packet + sizeof(h);
    switch (static_cast<GpuOpcode>(h.opcode)) {
        case GpuOpcode::WriteBuffer: {
            GpuWriteBufferPayload w{};
            if (h.payload_bytes < sizeof(w)) {
                return;
            }
            std::memcpy(&w, payload, sizeof(w));
            const size_t n = std::min<size_t>(w.data_len, h.payload_bytes - sizeof(w));
            HeldBuffer& b = s.held_buffers[w.id];
            const uint64_t end = uint64_t(w.dst_offset) + n;
            if (b.image.size() < end) {
                b.image.resize(end);
            }
            std::memcpy(b.image.data() + w.dst_offset, payload + sizeof(w), n);
            b.lo = std::min<uint64_t>(b.lo, w.dst_offset);
            b.hi = std::max<uint64_t>(b.hi, end);
            b.seq = ++s.held_seq;
            return;
        }
        case GpuOpcode::WriteTexture: {
            HeldTextureWrite held;
            if (h.payload_bytes < sizeof(held.box)) {
                return;
            }
            std::memcpy(&held.box, payload, sizeof(held.box));
            std::vector<HeldTextureWrite>& list = s.held_textures[held.box.texture_id];
            list.erase(std::remove_if(list.begin(), list.end(),
                                      [&](const HeldTextureWrite& old) { return box_contains(held.box, old.box); }),
                       list.end());
            held.seq = ++s.held_seq;
            held.packet.assign(packet, packet + sizeof(h) + h.payload_bytes);
            list.push_back(std::move(held));
            return;
        }
        case GpuOpcode::DestroyResource: {
            GpuDestroyResourcePayload d{};
            if (h.payload_bytes >= sizeof(d)) {
                std::memcpy(&d, payload, sizeof(d));
                s.held_buffers.erase(d.id);
                s.held_textures.erase(d.id);
            }
            write_packet(s, h, packet);
            return;
        }
        case GpuOpcode::CreateBuffer:
        case GpuOpcode::CreateShaderModule:
        case GpuOpcode::CreateRenderPipeline:
        case GpuOpcode::CreateTexture:
        case GpuOpcode::CreateSampler:
        case GpuOpcode::CreateBindGroup:
        case GpuOpcode::CreateComputePipeline:
        case GpuOpcode::CreateBindGroupLayout:
        case GpuOpcode::CreatePipelineLayout:
        case GpuOpcode::CreateRenderPipeline2:
        case GpuOpcode::CreateComputePipeline2:
        case GpuOpcode::CreateBindGroup2:
        case GpuOpcode::CreateSampler2:
        case GpuOpcode::CreateTexture2:
            write_packet(s, h, packet);
            return;
        default:
            return;  // render / compute commands, Present: dropped before the start
    }
}

// Ends the prefix: writes the held buffer images and texture writes in the order of their newest
// write; the following packets are frame 0's commands.
void flush_prefix(CaptureState& s) {
    struct Item {
        uint64_t seq;
        const HeldBuffer* buffer;
        uint32_t id;
        const HeldTextureWrite* texture;
    };
    std::vector<Item> items;
    for (const auto& [id, b] : s.held_buffers) {
        if (b.hi > b.lo) {
            items.push_back(Item{b.seq, &b, id, nullptr});
        }
    }
    for (const auto& [id, list] : s.held_textures) {
        for (const HeldTextureWrite& w : list) {
            items.push_back(Item{w.seq, nullptr, id, &w});
        }
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.seq < b.seq; });
    std::vector<uint8_t> packet;
    for (const Item& it : items) {
        if (it.texture != nullptr) {
            GpuPacketHeader h{};
            std::memcpy(&h, it.texture->packet.data(), sizeof(h));
            write_packet(s, h, it.texture->packet.data());
            continue;
        }
        const HeldBuffer& b = *it.buffer;
        const uint64_t n = b.hi - b.lo;
        GpuWriteBufferPayload w{};
        w.id = it.id;
        w.dst_offset = static_cast<uint32_t>(b.lo);
        w.data_len = static_cast<uint32_t>(n);
        GpuPacketHeader h{};
        h.opcode = static_cast<uint16_t>(GpuOpcode::WriteBuffer);
        h.payload_bytes = static_cast<uint32_t>(sizeof(w) + n);
        packet.resize(sizeof(h) + sizeof(w) + n);
        std::memcpy(packet.data(), &h, sizeof(h));
        std::memcpy(packet.data() + sizeof(h), &w, sizeof(w));
        std::memcpy(packet.data() + sizeof(h) + sizeof(w), b.image.data() + b.lo, n);
        write_packet(s, h, packet.data());
    }
    s.held_buffers.clear();
    s.held_textures.clear();
    s.prefix = false;
}

bool start_due(CaptureState& s) {
    if (s.begin_requested.load(std::memory_order_acquire)) {
        return true;
    }
    return s.config.start_ms != 0 &&
           std::chrono::steady_clock::now() - s.start >= std::chrono::milliseconds(s.config.start_ms);
}

void observe_locked(CaptureState& s, const uint8_t* data, size_t len) {
    size_t at = 0;
    while (at + sizeof(GpuPacketHeader) <= len) {
        GpuPacketHeader h{};
        std::memcpy(&h, data + at, sizeof(h));
        if (h.payload_bytes > len - at - sizeof(h)) {
            break;  // a truncated packet: never produced by the encoder; ignore the rest
        }
        if (s.prefix) {
            prefix_packet(s, h, data + at);
        } else if (s.active) {
            write_packet(s, h, data + at);
        }
        at += sizeof(h) + h.payload_bytes;
        if (h.opcode != static_cast<uint16_t>(GpuOpcode::Present)) {
            continue;
        }
        sample_present(s);
        if (s.prefix) {
            if (start_due(s)) {
                flush_prefix(s);
                s.first_present = s.presents.size();  // the next Present ends frame 0
            }
            continue;
        }
        if (s.active) {
            s.frames.push_back(s.current);
            s.current = FrameRecord{};
            s.current.offset = s.bytes;
            if (s.frames.size() >= s.config.frames) {
                s.active = false;
                std::fflush(s.file);
            }
        }
    }
}

void observer(const uint8_t* data, size_t len) {
    frame_capture_observe(data, len);
}

std::string json_escape(const std::string& in) {
    std::string out;
    for (char c : in) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        if (static_cast<unsigned char>(c) < 0x20) {
            continue;
        }
        out += c;
    }
    return out;
}

// The per-Present samples as a JSON array (the sidecar's "presents" value and the boot report's).
std::string presents_array_json(const CaptureState& s) {
    std::string out = "[";
    char buf[320];
    for (size_t i = 0; i < s.presents.size(); ++i) {
        const PresentSample& p = s.presents[i];
        std::snprintf(buf, sizeof(buf),
                      "%s\n    {\"index\": %zu, \"wall_ms\": %.3f, \"dispatches\": %" PRIu64 ", \"import_calls\": %" PRIu64
                      ", \"reserved_bytes\": %" PRIu64 ", \"committed_bytes\": %" PRIu64 ", \"counters\": {",
                      i == 0 ? "" : ",", i, p.wall_ms, p.counters.dispatches, p.counters.import_calls,
                      p.counters.reserved_bytes, p.counters.committed_bytes);
        out += buf;
        for (size_t k = 0; k < p.counters.counters.size(); ++k) {
            out += k == 0 ? "\"" : ", \"";
            out += json_escape(p.counters.counters[k].first);
            std::snprintf(buf, sizeof(buf), "\": %" PRIu64, p.counters.counters[k].second);
            out += buf;
        }
        out += "}}";
    }
    out += s.presents.empty() ? "]" : "\n  ]";
    return out;
}

}  // namespace

bool frame_capture_start(const FrameCaptureConfig& config) {
    CaptureState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.open || config.path.empty() || config.frames == 0) {
        return false;
    }
    s.file = std::fopen(config.path.c_str(), "wb");
    if (s.file == nullptr) {
        return false;
    }
    s.config = config;
    s.open = true;
    s.active = true;
    s.bytes = s.packets = s.omitted_bytes = 0;
    s.thinned_packets = s.thinned_bytes = 0;
    s.big_writes.clear();
    s.current = FrameRecord{};
    s.frames.clear();
    s.presents.clear();
    s.omitted.clear();
    s.prefix = config.deferred;
    s.begin_requested.store(false);
    s.first_present = 0;
    s.held_seq = 0;
    s.held_buffers.clear();
    s.held_textures.clear();
    s.start = std::chrono::steady_clock::now();
    if (config.attach) {
        os::set_wgpu_submit_observer(&observer);
    }
    return true;
}

void frame_capture_observe(const uint8_t* data, size_t len) {
    CaptureState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.open) {
        observe_locked(s, data, len);
    }
}

void frame_capture_finish() {
    CaptureState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.open) {
        return;
    }
    if (s.config.attach) {
        os::set_wgpu_submit_observer(nullptr);
    }
    std::fclose(s.file);
    s.file = nullptr;
    s.open = false;
    s.active = false;
    std::FILE* side = std::fopen((s.config.path + ".json").c_str(), "wb");
    if (side == nullptr) {
        return;
    }
    std::string name = s.config.path;
    const size_t slash = name.find_last_of("/\\");
    if (slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    std::fprintf(side,
                 "{\n  \"schema\": \"playgta6-frame-capture\",\n  \"schema_version\": 1,\n  \"trace\": \"%s\",\n"
                 "  \"payloads\": \"%s\",\n  \"frames_requested\": %u,\n  \"frames_complete\": %zu,\n"
                 "  \"presents_seen\": %zu,\n  \"packets\": %" PRIu64 ",\n  \"bytes\": %" PRIu64 ",\n"
                 "  \"omitted_bytes\": %" PRIu64 ",\n  \"thinned_packets\": %" PRIu64 ",\n  \"thinned_bytes\": %" PRIu64
                 ",\n  \"first_present\": %" PRIu64 ",\n  \"frames\": [",
                 json_escape(name).c_str(), s.config.keep_payloads ? "kept" : "omitted", s.config.frames,
                 s.frames.size(), s.presents.size(), s.packets, s.bytes, s.omitted_bytes, s.thinned_packets, s.thinned_bytes,
                 s.prefix ? uint64_t(s.presents.size()) : s.first_present);
    for (size_t i = 0; i < s.frames.size(); ++i) {
        const FrameRecord& f = s.frames[i];
        std::fprintf(side,
                     "%s\n    {\"index\": %zu, \"offset\": %" PRIu64 ", \"bytes\": %" PRIu64 ", \"packets\": %" PRIu64 "}",
                     i == 0 ? "" : ",", i, f.offset, f.bytes, f.packets);
    }
    std::fprintf(side, "%s],\n  \"presents\": %s,\n  \"omitted\": [", s.frames.empty() ? "" : "\n  ",
                 presents_array_json(s).c_str());
    for (size_t i = 0; i < s.omitted.size(); ++i) {
        const Omitted& o = s.omitted[i];
        std::fprintf(side,
                     "%s\n    {\"packet\": %" PRIu64 ", \"opcode\": %u, \"length\": %u, \"fnv1a64\": \"0x%016" PRIx64 "\"}",
                     i == 0 ? "" : ",", o.packet, static_cast<unsigned>(o.opcode), o.length, o.hash);
    }
    std::fprintf(side, "%s]\n}\n", s.omitted.empty() ? "" : "\n  ");
    std::fclose(side);
}

std::string frame_capture_presents_json() {
    CaptureState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return presents_array_json(s);
}

void frame_capture_set_counters(FrameCaptureCounters counters) {
    state().counters.store(counters, std::memory_order_release);
}

void frame_capture_begin() {
    state().begin_requested.store(true, std::memory_order_release);
}

uint32_t frame_capture_frames() {
    CaptureState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return static_cast<uint32_t>(s.frames.size());
}

}  // namespace playgta6::gpu
