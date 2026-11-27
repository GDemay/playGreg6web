// Native synthetic benchmark of actual Dawn BlobCache::store filesystem operations.
// Build: c++ -O3 -DNDEBUG -std=c++20 -pthread -Iruntime tools/bench_blob_cache.cpp -o work/bench-blob
// Run: work/bench-blob ENTRIES STORES VALUE_BYTES CAPPED [PARENT_DIRECTORY]
// Each invocation creates/removes its own directory. Setup and correctness checks are untimed.
// Run separately built baseline/patch binaries alternately, with no concurrent builds or tests.
#include "gpu/native_exec/blob_cache.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using playgta6::gpu::native_exec::BlobCache;
namespace fs = std::filesystem;

std::string key(uint64_t i) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(i));
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::fprintf(stderr, "usage: bench-blob ENTRIES STORES VALUE_BYTES CAPPED [PARENT_DIRECTORY]\n");
        return 2;
    }
    fs::path dir;
    try {
        const size_t entries = std::stoull(argv[1]);
        const size_t stores = std::stoull(argv[2]);
        const size_t value_bytes = std::stoull(argv[3]);
        const bool capped = std::stoul(argv[4]) != 0;
        if (!entries || !stores || !value_bytes || stores > entries) throw std::runtime_error("invalid workload");
        const uint64_t file_bytes = 32 + 16 + value_bytes;
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        dir = (argc == 6 ? fs::path(argv[5]) : fs::temp_directory_path()) /
              ("pg6-blob-bench-" + std::to_string(nonce));
        BlobCache cache;
        if (!cache.open(dir.string(), capped ? file_bytes * entries : 0)) throw std::runtime_error("open failed");
        std::vector<uint8_t> value(value_bytes);
        for (size_t i = 0; i < value.size(); ++i) value[i] = static_cast<uint8_t>(i * 37 + 11);
        for (size_t i = 0; i < entries; ++i) {
            const auto k = key(i);
            cache.store(k.data(), k.size(), value.data(), value.size());
        }
        const auto initial = cache.stats();
        if (initial.entries != entries || initial.store_rejected || initial.evictions) throw std::runtime_error("setup failed");
        // Prepare keys before timing so reported time includes only the production store calls.
        std::vector<std::string> keys;
        for (size_t i = 0; i < stores; ++i) keys.push_back(key(entries + i));
        const auto cpu_begin = std::clock();
        const auto begin = std::chrono::steady_clock::now();
        for (const auto& k : keys) cache.store(k.data(), k.size(), value.data(), value.size());
        const auto end = std::chrono::steady_clock::now();
        const double cpu_ms = 1000.0 * (std::clock() - cpu_begin) / CLOCKS_PER_SEC;
        const auto result = cache.stats();
        if (result.stores != entries + stores || result.store_rejected ||
            result.evictions != (capped ? stores : 0) ||
            result.entries != (capped ? entries : entries + stores) ||
            result.total_bytes != result.entries * file_bytes) throw std::runtime_error("incorrect store/eviction counts");
        std::vector<uint8_t> actual(value_bytes);
        // Check every expected survivor's checksum/value and each expected evicted key, untimed.
        for (size_t i = 0; i < entries + stores; ++i) {
            const auto k = key(i);
            const size_t n = cache.load(k.data(), k.size(), actual.data(), actual.size());
            if (capped && i < stores) {
                if (n != 0) throw std::runtime_error("wrong LRU victim");
            } else if (n != value.size() || actual != value) throw std::runtime_error("wrong surviving blob");
        }
        const double ms = std::chrono::duration<double, std::milli>(end - begin).count();
        std::printf("{\"entries\":%zu,\"stores\":%zu,\"value_bytes\":%zu,\"capped\":%s,\"elapsed_ms\":%.6f,\"us_per_store\":%.6f,\"cpu_ms\":%.6f,\"correct\":true}\n",
                    entries, stores, value_bytes, capped ? "true" : "false", ms, ms * 1000 / stores, cpu_ms);
        fs::remove_all(dir);
        return 0;
    } catch (const std::exception& e) {
        if (!dir.empty()) { std::error_code ec; fs::remove_all(dir, ec); }
        std::fprintf(stderr, "benchmark failed: %s\n", e.what());
        return 1;
    }
}
