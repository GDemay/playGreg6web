#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "wasm_boot.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_wasm_boot.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using playgta6::boot::WasmBootConfig;
using playgta6::boot::WasmBootResult;

struct FakeFile {
    std::vector<uint8_t> bytes;
};

struct FakeFs {
    std::map<std::string, FakeFile> files;
    std::map<int32_t, std::string> handles;
    int32_t next_handle = 1;
    int opens = 0;
    int reads = 0;
    int closes = 0;
    int fail_read_on = -1;  // 1-based read count to fail
    int64_t force_read_rc = 0;
    bool force_read = false;
};

FakeFs* g_fs = nullptr;
std::vector<std::string> g_logs;

int32_t fake_open(std::string_view rel_path) {
    ++g_fs->opens;
    const std::string key(rel_path);
    if (g_fs->files.find(key) == g_fs->files.end()) {
        return -2;
    }
    const int32_t h = g_fs->next_handle++;
    g_fs->handles[h] = key;
    return h;
}

int64_t fake_read(int32_t handle, uint64_t offset, void* dst, size_t len) {
    ++g_fs->reads;
    if (g_fs->fail_read_on > 0 && g_fs->reads == g_fs->fail_read_on) {
        return -9;
    }
    if (g_fs->force_read) {
        return g_fs->force_read_rc;
    }
    auto it = g_fs->handles.find(handle);
    if (it == g_fs->handles.end()) {
        return -9;
    }
    const FakeFile& file = g_fs->files[it->second];
    if (offset > file.bytes.size()) {
        return -9;
    }
    const size_t avail = file.bytes.size() - static_cast<size_t>(offset);
    const size_t n = avail < len ? avail : len;
    if (n > 0 && dst != nullptr) {
        std::memcpy(dst, file.bytes.data() + static_cast<size_t>(offset), n);
    }
    return static_cast<int64_t>(n);
}

void fake_close(int32_t handle) {
    ++g_fs->closes;
    g_fs->handles.erase(handle);
}

void fake_log(std::string_view message) {
    g_logs.emplace_back(message);
}

void put_file(FakeFs& fs, const std::string& path, std::string_view text) {
    FakeFile f;
    f.bytes.assign(text.begin(), text.end());
    fs.files[path] = std::move(f);
}

void put_bytes(FakeFs& fs, const std::string& path, std::vector<uint8_t> bytes) {
    FakeFile f;
    f.bytes = std::move(bytes);
    fs.files[path] = std::move(f);
}

std::vector<uint8_t> pgim_stub(size_t extra = 12) {
    std::vector<uint8_t> bytes(4 + extra, 0);
    bytes[0] = 'P';
    bytes[1] = 'G';
    bytes[2] = 'I';
    bytes[3] = 'M';
    for (size_t i = 0; i < extra; ++i) {
        bytes[4 + i] = static_cast<uint8_t>(i + 1);
    }
    return bytes;
}

WasmBootConfig make_cfg(FakeFs& fs) {
    g_fs = &fs;
    g_logs.clear();
    WasmBootConfig cfg;
    cfg.open_fn = fake_open;
    cfg.read_fn = fake_read;
    cfg.close_fn = fake_close;
    cfg.log_fn = fake_log;
    cfg.read_chunk_bytes = 4;
    return cfg;
}

void test_chunked_httpfs_reads() {
    FakeFs fs;
    const std::vector<uint8_t> payload = pgim_stub(12);  // 16 bytes, 4-byte chunks
    put_bytes(fs, "boot/guest.pgim", payload);
    WasmBootConfig cfg = make_cfg(fs);

    std::vector<uint8_t> out;
    std::string err;
    CHECK(playgta6::boot::load_file_via_httpfs(cfg, "boot/guest.pgim", out, err));
    CHECK(err.empty());
    CHECK(out == payload);
    CHECK(fs.opens == 1);
    CHECK(fs.closes == 1);
    CHECK(fs.handles.empty());
    CHECK(fs.reads >= 4);  // 16 bytes / 4 + possible short/eof
}

void test_close_on_read_error() {
    FakeFs fs;
    put_bytes(fs, "boot/guest.pgim", pgim_stub(32));
    WasmBootConfig cfg = make_cfg(fs);
    fs.fail_read_on = 2;

    std::vector<uint8_t> out;
    std::string err;
    CHECK(!playgta6::boot::load_file_via_httpfs(cfg, "boot/guest.pgim", out, err));
    CHECK(err.find("httpfs_read") != std::string::npos);
    CHECK(fs.opens == 1);
    CHECK(fs.closes == 1);
    CHECK(fs.handles.empty());
}

void test_max_size_guard() {
    FakeFs fs;
    put_bytes(fs, "boot/guest.pgim", pgim_stub(64));
    WasmBootConfig cfg = make_cfg(fs);
    cfg.max_file_bytes = 8;

    std::vector<uint8_t> out;
    std::string err;
    CHECK(!playgta6::boot::load_file_via_httpfs(cfg, "boot/guest.pgim", out, err));
    CHECK(err.find("max_file_bytes") != std::string::npos);
    CHECK(fs.closes == 1);
    CHECK(fs.handles.empty());
    CHECK(out.size() <= 8);
}

void test_negative_read_guard() {
    FakeFs fs;
    put_bytes(fs, "boot/guest.pgim", pgim_stub());
    WasmBootConfig cfg = make_cfg(fs);
    fs.force_read = true;
    fs.force_read_rc = -3;

    std::vector<uint8_t> out;
    std::string err;
    CHECK(!playgta6::boot::load_file_via_httpfs(cfg, "boot/guest.pgim", out, err));
    CHECK(fs.closes == 1);
}

void test_manifest_pgim_extraction() {
    const char* json = R"({
      "schema": "playgta6-recomp-program",
      "schema_version": 1,
      "modules": [
        {"module": "vcruntime140", "file": "vcruntime140.dll"},
        {"module": "game"}
      ],
      "files": [
        {"path": "vcruntime140\\image\\guest.pgim", "size": 10},
        {"path": "game/image/guest.pgim", "size": 20},
        {"path": "game/image/guest.pgim.json", "size": 4},
        {"path": "game/image/guest.pgim", "size": 20}
      ]
    })";
    const std::vector<std::string> paths = playgta6::boot::extract_pgim_paths_from_manifest(json);
    CHECK(paths.size() == 2);
    CHECK(paths[0] == "vcruntime140/image/guest.pgim");
    CHECK(paths[1] == "game/image/guest.pgim");
}

void test_escaped_pgim_path() {
    const char* json = R"({"files":[{"path":"mod\\image\\guest.pgim"}]})";
    const std::vector<std::string> paths = playgta6::boot::extract_pgim_paths_from_manifest(json);
    CHECK(paths.size() == 1);
    CHECK(paths[0] == "mod/image/guest.pgim");
}

void test_milestone_log_format() {
    FakeFs fs;
    WasmBootConfig cfg = make_cfg(fs);
    playgta6::boot::emit_boot_milestone(cfg, "PGIM_LOAD");
    playgta6::boot::emit_boot_milestone(cfg, "PGIM_LOAD", "loaded boot/guest.pgim (16 bytes)");
    playgta6::boot::emit_boot_milestone(cfg, "TLS_INIT", "smoke");
    playgta6::boot::emit_boot_milestone(cfg, "ENTRY_CALL", "");
    CHECK(g_logs.size() == 4);
    CHECK(g_logs[0] == "[boot] PGIM_LOAD");
    CHECK(g_logs[1] == "[boot] PGIM_LOAD: loaded boot/guest.pgim (16 bytes)");
    CHECK(g_logs[2] == "[boot] TLS_INIT: smoke");
    CHECK(g_logs[3] == "[boot] ENTRY_CALL");
}

void test_fallback_pgim_load() {
    FakeFs fs;
    put_bytes(fs, "boot/guest.pgim", pgim_stub(8));
    WasmBootConfig cfg = make_cfg(fs);
    cfg.manifest_path = "boot/program-manifest.json";
    cfg.fallback_pgim_path = "boot/guest.pgim";

    const WasmBootResult r = playgta6::boot::run_wasm_guest_boot(cfg);
    CHECK(r.ok);
    CHECK(!r.linked_guest_available);
    CHECK(r.modules_loaded == 1);
    CHECK(r.total_bytes_loaded == 12);
    CHECK(r.stage == "PGIM_LOAD");
    CHECK(r.exit_code == 0);
    bool saw_pgim = false;
    for (const std::string& line : g_logs) {
        if (line.find("[boot] PGIM_LOAD") == 0) {
            saw_pgim = true;
        }
    }
    CHECK(saw_pgim);
}

void test_manifest_driven_load() {
    FakeFs fs;
    put_file(fs, "boot/program-manifest.json",
             R"({"files":[{"path":"mod/image/guest.pgim"}]})");
    put_bytes(fs, "mod/image/guest.pgim", pgim_stub(4));
    WasmBootConfig cfg = make_cfg(fs);

    const WasmBootResult r = playgta6::boot::run_wasm_guest_boot(cfg);
    CHECK(r.ok);
    CHECK(r.modules_loaded == 1);
    CHECK(r.total_bytes_loaded == 8);
    CHECK(fs.files.count("boot/guest.pgim") == 0);
}

void test_invalid_pgim_magic() {
    FakeFs fs;
    std::vector<uint8_t> junk = {'N', 'O', 'P', 'E', 1, 2, 3, 4};
    put_bytes(fs, "boot/guest.pgim", junk);
    WasmBootConfig cfg = make_cfg(fs);

    const WasmBootResult r = playgta6::boot::run_wasm_guest_boot(cfg);
    CHECK(!r.ok);
    CHECK(r.stage == "pgim_magic");
    CHECK(r.error.find("invalid PGIM magic") != std::string::npos);
    CHECK(r.modules_loaded == 0);
}

void test_missing_everything() {
    FakeFs fs;
    WasmBootConfig cfg = make_cfg(fs);
    const WasmBootResult r = playgta6::boot::run_wasm_guest_boot(cfg);
    CHECK(!r.ok);
    CHECK(r.stage == "httpfs");
}

void test_has_pgim_magic() {
    const std::vector<uint8_t> ok = pgim_stub();
    CHECK(playgta6::boot::has_pgim_magic(ok));
    const uint8_t short_buf[] = {'P', 'G', 'I'};
    CHECK(!playgta6::boot::has_pgim_magic(short_buf));
    const uint8_t bad[] = {'P', 'G', 'I', 'X'};
    CHECK(!playgta6::boot::has_pgim_magic(bad));
}

}  // namespace

int main() {
    test_chunked_httpfs_reads();
    test_close_on_read_error();
    test_max_size_guard();
    test_negative_read_guard();
    test_manifest_pgim_extraction();
    test_escaped_pgim_path();
    test_milestone_log_format();
    test_fallback_pgim_load();
    test_manifest_driven_load();
    test_invalid_pgim_magic();
    test_missing_everything();
    test_has_pgim_magic();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d test(s) failed\n", g_failures);
        return 1;
    }
    std::printf("test_wasm_boot: ok\n");
    return 0;
}
