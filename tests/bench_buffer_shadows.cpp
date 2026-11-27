// Actual Device::flush_cpu_buffers shadow allocation benchmark. No GPU or game data. Windows tracking required.
// cmake -S runtime -B work/build -DPLAYGTA6_OPUS=OFF -DPLAYGTA6_PERFORMANCE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
// cmake --build work/build --config Release --target bench_buffer_shadows
// bench_buffer_shadows BUFFERS FLUSHES DIRTY_BUFFERS PATTERN
// PATTERN: sparse-first, sparse-resident (first page warm), sparse-fully-resident,
// dense-first (FLUSHES=1), dense-resident
#include "gpu/d3d12_guest_resources.h"
#include "guest/com_bridge.h"
#include "os/vmem.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static double process_cpu_ms() {
#if defined(_WIN32)
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) return -1;
    const uint64_t k = (uint64_t{kernel.dwHighDateTime} << 32) | kernel.dwLowDateTime;
    const uint64_t u = (uint64_t{user.dwHighDateTime} << 32) | user.dwLowDateTime;
    return static_cast<double>(k + u) / 10000.0;
#else
    return 1000.0 * std::clock() / CLOCKS_PER_SEC;
#endif
}

using playgta6::GpuPacketHeader;
using playgta6::GpuWriteBufferPayload;
using playgta6::GpuOpcode;
namespace gpu = playgta6::gpu;
namespace dg = gpu::d3d12_guest;
namespace com = playgta6::guest::com;
namespace os = playgta6::os;
int main(int argc, char** argv) {
    if (argc != 5) return 2;
    const size_t count = std::strtoull(argv[1], nullptr, 0);
    const size_t frames = std::strtoull(argv[2], nullptr, 0);
    const size_t dirty = std::strtoull(argv[3], nullptr, 0);
    const bool dense = std::strncmp(argv[4], "dense-", 6) == 0;
    const bool fully_resident_sparse = std::strcmp(argv[4], "sparse-fully-resident") == 0;
    const bool resident = fully_resident_sparse || std::strcmp(argv[4], "sparse-resident") == 0 || std::strcmp(argv[4], "dense-resident") == 0;
    const bool warm_all_pages = dense || fully_resident_sparse;
    const bool known = resident || std::strcmp(argv[4], "sparse-first") == 0 || std::strcmp(argv[4], "dense-first") == 0;
    if (!known || !count || !frames || dirty > count || (dense && !resident && frames != 1)) return 2;
    try {
        os::set_write_tracking(true);
        dg::set_flush_mode(dg::FlushMode::Index);
        os::VirtualMemoryArena arena;
        com::reset_com_bridge();
        com::set_guest_memory(&arena);
        constexpr uint64_t gran = 65536, bytes = 49152, page_bytes = 4096;
        // A 64 KiB hole every 20 buffers gives ~205 take hulls at 4096 buffers, matching
        // the published flush profile's ~210 ranges. Logical CPU-visible data: 192 MiB.
        const uint64_t mem = com::guest_alloc((count + count / 20 + 1) * gran);
        if (!mem) throw std::runtime_error("allocation failed");
        dg::DeviceOptions options;
        options.sink = [](const uint8_t*, size_t) {};
        auto device = com::create_object<dg::Device>(com::implements(dg::kID3D12Device), options);
        if (!device) throw std::runtime_error("device failed");
        device->initialize_guest_payload();
        dg::GuestResourceDesc desc{};
        desc.Dimension = 1; desc.Width = bytes; desc.Height = 1; desc.DepthOrArraySize = 1;
        desc.MipLevels = 1; desc.SampleCount = 1; desc.Layout = 1;
        std::vector<com::Ref<dg::Resource>> resources;
        std::vector<uint64_t> addresses;
        std::unordered_map<uint32_t, size_t> ids;
        std::vector<uint8_t> zero(bytes, 0);
        if (resident) {
            for (size_t page = 0; page < (warm_all_pages ? 12u : 1u); ++page) zero[page * page_bytes] = 0xA5;
        }
        for (size_t i = 0; i < count; ++i) {
            const uint64_t va = mem + (i + i / 20) * gran;
            if (!com::write_guest(va, zero.data(), zero.size())) throw std::runtime_error("initial write failed");
            dg::Resource::Placement placement{};
            placement.address = va;
            auto r = com::create_object<dg::Resource>(com::implements(dg::kID3D12Resource), dg::device_ref(device.get()), desc, std::move(placement), 0);
            if (!r || !r->initialize()) throw std::runtime_error("resource failed");
            ids.emplace(r->id(), i);
            addresses.push_back(va);
            resources.push_back(std::move(r));
        }
        auto flush = [&](gpu::CommandEncoder& enc) {
            device->begin_submission(); device->flush_cpu_buffers(enc); device->end_submission();
        };
        for (int i = 0; i < 4; ++i) { gpu::CommandEncoder enc; flush(enc); }
        uintptr_t lo = 0, hi = 0;
        const bool tracked = resources[0]->write_tracked_host_range(&lo, &hi);
        if (!tracked || !resources[0]->flush_primed) {
            std::fprintf(stderr, "index benchmark requires active native OS write tracking (Windows); refusing unsupported lane\n");
            return 77;
        }
        const auto visits0 = dg::work_total(dg::Work::FlushBuffersVisited);
        const auto skips0 = dg::work_total(dg::Work::FlushBuffersSkipped);
        const auto compared0 = dg::work_total(dg::Work::ShadowComparedBytes);
        const auto takes0 = dg::work_total(dg::Work::FlushIndexTakes);
        uint64_t resident_before = 0;
        for (const auto& r : resources) resident_before += r->shadow_resident_bytes();
        std::vector<std::array<uint32_t, 12>> expected(count);
        if (resident) for (auto& words : expected)
            for (size_t page = 0; page < (warm_all_pages ? 12u : 1u); ++page) words[page] = 0xA5;
        std::vector<size_t> selected(count, SIZE_MAX);
        uint64_t digest = 0xcbf29ce484222325ull, writes = 0;
        double wall_ms = 0, cpu_ms = 0;
        for (size_t frame = 0; frame < frames; ++frame) {
            std::fill(selected.begin(), selected.end(), SIZE_MAX);
            const size_t page = frame % 12;
            const uint32_t value = static_cast<uint32_t>(frame + 1);
            for (size_t j = 0; j < dirty; ++j) {
                const size_t i = (j * count / dirty + frame * 37) % count;
                for (size_t q = 0; q < (dense ? 12u : 1u); ++q) {
                    const size_t written_page = dense ? q : page;
                    if (!com::write_guest(addresses[i] + written_page * page_bytes, value)) throw std::runtime_error("dirty write failed");
                    expected[i][written_page] = value;
                }
                selected[i] = dense ? 0u : page;
            }
            gpu::CommandEncoder enc;
            const double cpu = process_cpu_ms();
            const auto begin = std::chrono::steady_clock::now();
            flush(enc);
            wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            cpu_ms += process_cpu_ms() - cpu;
            // Validate packet bytes/order outside timing; the digest must match baseline exactly.
            size_t off = 0, got = 0, previous = 0;
            while (off < enc.size()) {
                GpuPacketHeader h{};
                if (enc.size() - off < sizeof(h)) throw std::runtime_error("short packet");
                std::memcpy(&h, enc.data() + off, sizeof(h)); off += sizeof(h);
                if (h.payload_bytes > enc.size() - off || static_cast<GpuOpcode>(h.opcode) != GpuOpcode::WriteBuffer || h.payload_bytes < sizeof(GpuWriteBufferPayload)) throw std::runtime_error("unexpected packet");
                GpuWriteBufferPayload w{}; std::memcpy(&w, enc.data() + off, sizeof(w));
                const auto it = ids.find(w.id);
                const size_t expected_bytes = dense ? bytes : page_bytes;
                const size_t expected_page = dense ? 0u : page;
                if (it == ids.end() || w.data_len != expected_bytes || w.dst_offset != expected_page * page_bytes || selected[it->second] != expected_page ||
                    (got && it->second <= previous) || h.payload_bytes != sizeof(w) + w.data_len) throw std::runtime_error("wrong packet/order");
                for (size_t k = 0; k < expected_bytes; k += 4) {
                    uint32_t actual; std::memcpy(&actual, enc.data() + off + sizeof(w) + k, 4);
                    const size_t written_page = dense ? k / page_bytes : page;
                    if (actual != (k % page_bytes == 0 ? expected[it->second][written_page] : 0)) throw std::runtime_error("wrong packet bytes");
                }
                previous = it->second; ++got; off += h.payload_bytes;
            }
            if (got != dirty) throw std::runtime_error("wrong upload count");
            writes += got;
            for (size_t k = 0; k < enc.size(); ++k) { digest ^= enc.data()[k]; digest *= 0x100000001b3ull; }
        }
        const auto visits = dg::work_total(dg::Work::FlushBuffersVisited) - visits0;
        const auto skips = dg::work_total(dg::Work::FlushBuffersSkipped) - skips0;
        const auto compared = dg::work_total(dg::Work::ShadowComparedBytes) - compared0;
        if (visits != frames * dirty || skips != frames * (count - dirty) || compared != frames * dirty * (dense ? bytes : page_bytes)) throw std::runtime_error("wrong visit/skip/cursor comparison counters");
        uint64_t resident_after = 0;
        for (const auto& r : resources) resident_after += r->shadow_resident_bytes();
        std::printf("{\"buffers\":%zu,\"logical_bytes\":%llu,\"flushes\":%zu,\"dirty_buffers_per_flush\":%zu,\"pattern\":\"%s\",\"shadow_resident_before_bytes\":%llu,\"shadow_resident_after_bytes\":%llu,\"tracking\":%s,\"wall_ms\":%.6f,\"ms_per_flush\":%.6f,\"cpu_ms\":%.6f,\"visited\":%llu,\"skipped\":%llu,\"compared_bytes\":%llu,\"take_ranges\":%llu,\"uploads\":%llu,\"packet_digest\":\"%016llx\",\"correct\":true}\n",
                    count, static_cast<unsigned long long>(count * bytes), frames, dirty, argv[4], static_cast<unsigned long long>(resident_before), static_cast<unsigned long long>(resident_after), tracked ? "true" : "false", wall_ms, wall_ms / frames, cpu_ms,
                    static_cast<unsigned long long>(visits), static_cast<unsigned long long>(skips), static_cast<unsigned long long>(compared),
                    static_cast<unsigned long long>(dg::work_total(dg::Work::FlushIndexTakes) - takes0), static_cast<unsigned long long>(writes), static_cast<unsigned long long>(digest));
        resources.clear(); device.reset(); com::guest_free(mem); com::reset_com_bridge(); com::set_guest_memory(nullptr);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "flush benchmark: %s\n", e.what()); return 1; }
}
