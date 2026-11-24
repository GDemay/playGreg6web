#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "gpu/d3d12_guest_objects.h"
#include "gpu/d3d12_guest_resources.h"
#include "gpu/d3d12_xs_exports.h"
#include "guest/boot.h"
#include "guest/com_bridge.h"
#include "guest/guest_runtime.h"
#include "guest/program_loader.h"
#include "os/httpfs_mailbox.h"
#include "os/image_loader.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
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

void test_resolve_module_pgim_path() {
    const std::vector<std::string> manifest_paths = {
        "boot/vcruntime140_1/image/guest.pgim",
        "vcruntime140/image/guest.pgim",
        "boot/msvcp140/image/guest.pgim",
        "BOOT/Main/image/guest.pgim",
    };

    CHECK(playgta6::boot::resolve_module_pgim_path("main", manifest_paths) ==
          "BOOT/Main/image/guest.pgim");
    CHECK(playgta6::boot::resolve_module_pgim_path("MSVCP140", manifest_paths) ==
          "boot/msvcp140/image/guest.pgim");
    CHECK(playgta6::boot::resolve_module_pgim_path("vcruntime140", manifest_paths) ==
          "vcruntime140/image/guest.pgim");
    CHECK(playgta6::boot::resolve_module_pgim_path("vcruntime140_1", manifest_paths) ==
          "boot/vcruntime140_1/image/guest.pgim");

    // Prefix collision guard: "vcruntime140" must not match "vcruntime140_1".
    const std::vector<std::string> only_vcr1 = {"boot/vcruntime140_1/image/guest.pgim"};
    CHECK(playgta6::boot::resolve_module_pgim_path("vcruntime140", only_vcr1) ==
          "boot/vcruntime140/image/guest.pgim");

    // Fallback when manifest has no paths or no match.
    const std::vector<std::string> empty_paths;
    CHECK(playgta6::boot::resolve_module_pgim_path("msvcp140", empty_paths) ==
          "boot/msvcp140/image/guest.pgim");
    CHECK(playgta6::boot::resolve_module_pgim_path("ucrtbase", manifest_paths) ==
          "boot/ucrtbase/image/guest.pgim");
    CHECK(playgta6::boot::resolve_module_pgim_path("", manifest_paths).empty());
}

constexpr uint32_t kTestImageSize = 0x4000;
constexpr uint32_t kTestTextRva = 0x2000;
constexpr uint64_t kTestDllBase = 0x1'4000'0000ULL;
constexpr uint64_t kTestExeBase = 0x1'4001'0000ULL;

void put_bytes_at(std::vector<uint8_t>& buf, size_t at, const void* src, size_t n) {
    if (buf.size() < at + n) {
        buf.resize(at + n, 0);
    }
    std::memcpy(buf.data() + at, src, n);
}

void put_u16(std::vector<uint8_t>& buf, size_t at, uint16_t v) { put_bytes_at(buf, at, &v, 2); }
void put_u32(std::vector<uint8_t>& buf, size_t at, uint32_t v) { put_bytes_at(buf, at, &v, 4); }
void put_u64(std::vector<uint8_t>& buf, size_t at, uint64_t v) { put_bytes_at(buf, at, &v, 8); }

void append_le(std::vector<uint8_t>& out, uint64_t v, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        out.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
}

std::vector<uint8_t> make_valid_pgim(uint64_t base, uint32_t entry_rva) {
    namespace pgim = playgta6::os::pgim;

    std::vector<uint8_t> pe_hdr(0x1000, 0);
    constexpr size_t kOpt = 0x80 + 24;
    put_u16(pe_hdr, 0, 0x5A4D);
    put_u32(pe_hdr, 0x3C, 0x80);
    put_u32(pe_hdr, 0x80, 0x00004550);
    put_u16(pe_hdr, 0x84, 0x8664);
    put_u16(pe_hdr, kOpt, 0x020B);
    put_u64(pe_hdr, kOpt + 24, base);
    put_u32(pe_hdr, kOpt + 56, kTestImageSize);
    put_u32(pe_hdr, kOpt + 108, 16);

    struct Sec {
        const char* name;
        uint32_t rva;
        uint32_t prot;
        uint32_t flags;
        std::vector<uint8_t> bytes;
    };
    std::vector<Sec> secs = {
        {"<headers>", 0, pgim::kProtRead, pgim::kSectionFlagHeaders, std::move(pe_hdr)},
        {".text", kTestTextRva, pgim::kProtRead | pgim::kProtExecute, 0, {}},
    };

    const uint64_t tables_end = 64 + 64 * secs.size();
    std::vector<uint64_t> offsets;
    uint64_t cursor = tables_end;
    for (const Sec& s : secs) {
        if (s.bytes.empty()) {
            offsets.push_back(0);
            continue;
        }
        const uint64_t off = (cursor + 4095) / 4096 * 4096;
        offsets.push_back(off);
        cursor = off + s.bytes.size();
    }

    std::vector<uint8_t> out = {'P', 'G', 'I', 'M'};
    append_le(out, 1, 4);
    append_le(out, 64, 4);
    append_le(out, 64, 4);
    append_le(out, pgim::kFlagHeadersMapped, 4);
    append_le(out, 4096, 4);
    append_le(out, base, 8);
    append_le(out, kTestImageSize, 4);
    append_le(out, 0x1000, 4);
    append_le(out, entry_rva, 4);
    append_le(out, secs.size(), 4);
    append_le(out, 0, 4);
    append_le(out, 0, 4);
    append_le(out, cursor, 8);

    for (size_t i = 0; i < secs.size(); ++i) {
        std::string name = secs[i].name;
        name.resize(16, '\0');
        out.insert(out.end(), name.begin(), name.end());
        append_le(out, secs[i].rva, 4);
        append_le(out, 0x1000, 4);
        append_le(out, secs[i].prot, 4);
        append_le(out, secs[i].flags, 4);
        append_le(out, offsets[i], 8);
        append_le(out, secs[i].bytes.size(), 8);
        append_le(out, pgim::fnv1a_64(secs[i].bytes.data(), secs[i].bytes.size()), 8);
        append_le(out, 0, 4);
        append_le(out, 0, 4);
    }
    for (size_t i = 0; i < secs.size(); ++i) {
        if (!secs[i].bytes.empty()) {
            out.resize(static_cast<size_t>(offsets[i]), 0);
            out.insert(out.end(), secs[i].bytes.begin(), secs[i].bytes.end());
        }
    }
    return out;
}

bool g_test_dllmain_ran = false;
bool g_test_entry_ran = false;
std::function<void()> g_test_entry_hook;

void fake_dll_main(playgta6::GuestState* s) {
    g_test_dllmain_ran = true;
    s->gpr[playgta6::GuestReg::Rax] = 1;  // TRUE
    s->gpr[playgta6::GuestReg::Rsp] += 8;
}

void fake_exe_entry(playgta6::GuestState* s) {
    g_test_entry_ran = true;
    if (g_test_entry_hook) {
        g_test_entry_hook();
    }
    s->gpr[playgta6::GuestReg::Rax] = 42;
    s->gpr[playgta6::GuestReg::Rsp] += 8;
}

const Pg6LiftedFunctionEntry kTestDllTable[] = {{kTestDllBase + kTestTextRva, &fake_dll_main}};
const uint64_t kTestDllTableCount = 1;
const Pg6LiftedFunctionEntry kTestExeTable[] = {{kTestExeBase + kTestTextRva, &fake_exe_entry}};
const uint64_t kTestExeTableCount = 1;

void test_program_boot_invalid_descriptor() {
    FakeFs fs;
    WasmBootConfig cfg = make_cfg(fs);

    const Pg6ProgramModule mods[] = {
        {"helper.dll", "helper", kTestDllBase, kTestImageSize, 0, kPg6ProgramModuleDll, 0, nullptr,
         nullptr, nullptr, nullptr},
        {"game.exe", "main", kTestExeBase, kTestImageSize, kTestExeBase + kTestTextRva,
         kPg6ProgramModuleMain, 0, nullptr, nullptr, nullptr, nullptr},
    };

    // Wrong version.
    const Pg6Program bad_ver{99u, 2, mods, nullptr, nullptr};
    WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, bad_ver);
    CHECK(!r.ok);
    CHECK(r.linked_guest_available);
    CHECK(r.stage == "program_prepare");
    CHECK(!r.error.empty());
    CHECK(r.modules_loaded == 0);
    CHECK(fs.opens == 0);

    // No modules.
    const Pg6Program empty_prog{kPg6ProgramVersion, 0, nullptr, nullptr, nullptr};
    r = playgta6::boot::run_wasm_program_boot(cfg, empty_prog);
    CHECK(!r.ok);
    CHECK(r.stage == "program_prepare");
    CHECK(r.error.find("no modules") != std::string::npos);

    // Main module not last.
    const Pg6ProgramModule wrong_order[] = {mods[1], mods[0]};
    const Pg6Program bad_order{kPg6ProgramVersion, 2, wrong_order, nullptr, nullptr};
    r = playgta6::boot::run_wasm_program_boot(cfg, bad_order);
    CHECK(!r.ok);
    CHECK(r.stage == "program_prepare");

    // Empty build module name.
    Pg6ProgramModule empty_name[] = {mods[0], mods[1]};
    empty_name[0].module = "";
    const Pg6Program bad_name{kPg6ProgramVersion, 2, empty_name, nullptr, nullptr};
    r = playgta6::boot::run_wasm_program_boot(cfg, bad_name);
    CHECK(!r.ok);
    CHECK(r.stage == "program_prepare");
}

void test_program_boot_missing_module_pgim() {
    FakeFs fs;
    put_file(fs, "boot/program-manifest.json",
             R"({"files":[{"path":"helper/image/guest.pgim"},{"path":"main/image/guest.pgim"}]})");
    put_bytes(fs, "helper/image/guest.pgim", pgim_stub(8));
    WasmBootConfig cfg = make_cfg(fs);

    const Pg6ProgramModule mods[] = {
        {"helper.dll", "helper", kTestDllBase, kTestImageSize, 0, kPg6ProgramModuleDll, 0, nullptr,
         nullptr, nullptr, nullptr},
        {"game.exe", "main", kTestExeBase, kTestImageSize, kTestExeBase + kTestTextRva,
         kPg6ProgramModuleMain, 0, nullptr, nullptr, nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 2, mods, nullptr, nullptr};

    const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
    CHECK(!r.ok);
    CHECK(r.linked_guest_available);
    CHECK(r.stage == "httpfs");
    CHECK(r.modules_loaded == 1);
    CHECK(r.total_bytes_loaded == 12);
    CHECK(r.error.find("httpfs_open failed") != std::string::npos);
}

void test_program_boot_bad_magic() {
    FakeFs fs;
    put_bytes(fs, "boot/helper/image/guest.pgim", pgim_stub(8));
    put_bytes(fs, "boot/main/image/guest.pgim", std::vector<uint8_t>{'B', 'A', 'D', '!', 1, 2});
    WasmBootConfig cfg = make_cfg(fs);

    const Pg6ProgramModule mods[] = {
        {"helper.dll", "helper", kTestDllBase, kTestImageSize, 0, kPg6ProgramModuleDll, 0, nullptr,
         nullptr, nullptr, nullptr},
        {"game.exe", "main", kTestExeBase, kTestImageSize, kTestExeBase + kTestTextRva,
         kPg6ProgramModuleMain, 0, nullptr, nullptr, nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 2, mods, nullptr, nullptr};

    const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
    CHECK(!r.ok);
    CHECK(r.stage == "pgim_magic");
    CHECK(r.modules_loaded == 1);
    CHECK(r.error.find("invalid PGIM magic") != std::string::npos);
}

void test_program_boot_corrupt_pgim_error_propagation() {
    FakeFs fs;
    // One module without leading "boot/", one with leading "boot/", and no manifest:
    // exercises fallback path resolution and prefix-stripping retry seamlessly.
    put_bytes(fs, "helper/image/guest.pgim", pgim_stub(12));
    put_bytes(fs, "boot/main/image/guest.pgim", pgim_stub(16));
    WasmBootConfig cfg = make_cfg(fs);

    const Pg6ProgramModule mods[] = {
        {"helper.dll", "helper", kTestDllBase, kTestImageSize, 0, kPg6ProgramModuleDll, 0, nullptr,
         nullptr, nullptr, nullptr},
        {"game.exe", "main", kTestExeBase, kTestImageSize, kTestExeBase + kTestTextRva,
         kPg6ProgramModuleMain, 0, nullptr, nullptr, nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 2, mods, nullptr, nullptr};

    playgta6::os::VirtualMemoryArena arena;
    playgta6::os::win32::set_vmem_arena(&arena);
    const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
    playgta6::guest::reset_boot();
    playgta6::os::win32::set_vmem_arena(nullptr);

    CHECK(!r.ok);
    CHECK(r.linked_guest_available);
    CHECK(r.modules_loaded == 2);
    CHECK(r.total_bytes_loaded == 16 + 20);
    CHECK(r.stage == "load_failed");
    CHECK(!r.error.empty());
}

void test_program_boot_full_synthetic_execution() {
    FakeFs fs;
    const std::vector<uint8_t> dll_pgim = make_valid_pgim(kTestDllBase, kTestTextRva);
    const std::vector<uint8_t> exe_pgim = make_valid_pgim(kTestExeBase, kTestTextRva);
    put_file(fs, "boot/program-manifest.json",
             R"({"files":[{"path":"helper/image/guest.pgim"},{"path":"main/image/guest.pgim"}]})");
    // helper is stored under "boot/..." while manifest omits "boot/"; main matches manifest as-is.
    put_bytes(fs, "boot/helper/image/guest.pgim", dll_pgim);
    put_bytes(fs, "main/image/guest.pgim", exe_pgim);

    WasmBootConfig cfg = make_cfg(fs);
    cfg.read_chunk_bytes = 4096;

    const Pg6ProgramModule mods[] = {
        {"helper.dll", "helper", kTestDllBase, kTestImageSize, kTestDllBase + kTestTextRva,
         kPg6ProgramModuleDll, 0, kTestDllTable, &kTestDllTableCount, nullptr, nullptr},
        {"game_scarlett_master.exe", "main", kTestExeBase, kTestImageSize,
         kTestExeBase + kTestTextRva, kPg6ProgramModuleMain, 0, kTestExeTable, &kTestExeTableCount,
         nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 2, mods, nullptr, nullptr};

    g_test_dllmain_ran = false;
    g_test_entry_ran = false;

    playgta6::os::VirtualMemoryArena arena;
    playgta6::os::win32::set_vmem_arena(&arena);
    const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
    const bool dll_registered = playgta6::os::win32::GetModuleHandleA("helper.dll") == kTestDllBase;
    playgta6::guest::reset_boot();
    playgta6::os::win32::set_vmem_arena(nullptr);

    CHECK(r.ok);
    CHECK(r.linked_guest_available);
    CHECK(r.stage == "ENTRY_CALL");
    CHECK(r.exit_code == 42);
    CHECK(r.error.empty());
    CHECK(r.modules_loaded == 2);
    CHECK(r.total_bytes_loaded == dll_pgim.size() + exe_pgim.size());
    CHECK(g_test_dllmain_ran);
    CHECK(g_test_entry_ran);
    CHECK(dll_registered);

    int pgim_milestones = 0;
    for (const std::string& line : g_logs) {
        if (line.find("[boot] PGIM_LOAD: ") == 0) {
            ++pgim_milestones;
        }
    }
    CHECK(pgim_milestones == 2);
}

std::vector<uint8_t> make_synthetic_wgpk(const std::vector<std::pair<uint64_t, uint8_t>>& entries) {
    std::vector<uint8_t> out(16 + entries.size() * 20, 0);
    std::memcpy(out.data(), "WGPK", 4);
    const uint32_t version = 1;
    const uint32_t count = static_cast<uint32_t>(entries.size());
    const uint32_t flags = 0;
    std::memcpy(out.data() + 4, &version, 4);
    std::memcpy(out.data() + 8, &count, 4);
    std::memcpy(out.data() + 12, &flags, 4);
    for (size_t i = 0; i < entries.size(); ++i) {
        const std::string code = "// stage " + std::to_string(entries[i].second) + "\n";
        const uint32_t offset = static_cast<uint32_t>(out.size());
        const uint32_t length = static_cast<uint32_t>(code.size());
        uint8_t* entry = out.data() + 16 + i * 20;
        std::memcpy(entry, &entries[i].first, 8);
        entry[8] = entries[i].second;
        std::memcpy(entry + 12, &offset, 4);
        std::memcpy(entry + 16, &length, 4);
        out.insert(out.end(), code.begin(), code.end());
    }
    return out;
}

void setup_synthetic_program_fs(FakeFs& fs) {
    put_file(fs, "boot/program-manifest.json",
             R"({"files":[{"path":"main/image/guest.pgim"}]})");
    put_bytes(fs, "main/image/guest.pgim", make_valid_pgim(kTestExeBase, kTestTextRva));
}

void test_program_boot_scoped_httpfs_bridge() {
    FakeFs fs;
    setup_synthetic_program_fs(fs);
    // "game/<rel>" takes precedence over "<rel>" for unprefixed paths.
    put_file(fs, "game/data/config.bin", "GAME_CONFIG_DATA");
    put_file(fs, "data/config.bin", "ROOT_CONFIG_DATA");
    // Unprefixed path missing under "game/" falls back to "<rel>".
    put_file(fs, "loose/fallback.txt", "FALLBACK_PAYLOAD");
    // Already-prefixed "game/" and "boot/" paths must not be double-prefixed.
    put_file(fs, "game/exact.dat", "EXACT_GAME");
    put_file(fs, "boot/meta.txt", "BOOT_META");

    WasmBootConfig cfg = make_cfg(fs);
    cfg.read_chunk_bytes = 4096;

    const Pg6ProgramModule mods[] = {
        {"game_scarlett_master.exe", "main", kTestExeBase, kTestImageSize,
         kTestExeBase + kTestTextRva, kPg6ProgramModuleMain, 0, kTestExeTable, &kTestExeTableCount,
         nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 1, mods, nullptr, nullptr};

    CHECK(playgta6::os::win32::get_httpfs_client() == nullptr);

    bool hook_ran = false;
    bool client_installed = false;
    std::string read_game_pref;
    std::string read_fallback;
    std::string read_exact_game;
    std::string read_boot_meta;
    std::string read_win32;
    bool missing_failed = false;
    bool empty_failed = false;
    bool close_ok = false;

    g_test_entry_hook = [&]() {
        hook_ran = true;
        playgta6::os::HttpfsMailboxClient* client = playgta6::os::win32::get_httpfs_client();
        client_installed = (client != nullptr);
        if (client == nullptr) {
            return;
        }

        auto read_all_via_client = [&](std::string_view path, bool& closed_out) -> std::string {
            const int32_t h = client->open(path);
            if (h < 0) {
                return {};
            }
            char buf[64]{};
            const int64_t n1 = client->read(h, 0, buf, 5);
            const int64_t n2 = client->read(h, 5, buf + 5, sizeof(buf) - 5);
            closed_out = client->close(h);
            if (n1 < 0 || n2 < 0) {
                return {};
            }
            return std::string(buf, static_cast<size_t>(n1 + n2));
        };

        bool c1 = false;
        bool c2 = false;
        bool c3 = false;
        bool c4 = false;
        read_game_pref = read_all_via_client("\\./data\\config.bin", c1);
        read_fallback = read_all_via_client("/loose/fallback.txt", c2);
        read_exact_game = read_all_via_client("/game/exact.dat", c3);
        read_boot_meta = read_all_via_client("./boot/meta.txt", c4);
        close_ok = c1 && c2 && c3 && c4;
        missing_failed = (client->open("does/not/exist.bin") < 0);
        empty_failed = (client->open("///./") < 0);

        // End-to-end Win32 CreateFileA / ReadFile / CloseHandle via the scoped bridge.
        const playgta6::os::win32::Handle fh = playgta6::os::win32::CreateFileA(
            "G:\\data\\config.bin",
            playgta6::os::win32::GENERIC_READ,
            playgta6::os::win32::FILE_SHARE_READ,
            nullptr,
            playgta6::os::win32::OPEN_EXISTING,
            playgta6::os::win32::FILE_ATTRIBUTE_NORMAL);
        if (fh != playgta6::os::win32::INVALID_HANDLE_VALUE) {
            char wbuf[64]{};
            uint32_t got = 0;
            if (playgta6::os::win32::ReadFile(fh, wbuf, sizeof(wbuf), &got, nullptr)) {
                read_win32.assign(wbuf, got);
            }
            playgta6::os::win32::CloseHandle(fh);
        }
    };

    playgta6::os::VirtualMemoryArena arena;
    playgta6::os::win32::set_vmem_arena(&arena);
    const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
    g_test_entry_hook = nullptr;
    playgta6::guest::reset_boot();
    playgta6::os::win32::set_vmem_arena(nullptr);

    CHECK(r.ok);
    CHECK(r.exit_code == 42);
    CHECK(hook_ran);
    CHECK(client_installed);
    CHECK(read_game_pref == "GAME_CONFIG_DATA");
    CHECK(read_fallback == "FALLBACK_PAYLOAD");
    CHECK(read_exact_game == "EXACT_GAME");
    CHECK(read_boot_meta == "BOOT_META");
    CHECK(read_win32 == "GAME_CONFIG_DATA");
    CHECK(missing_failed);
    CHECK(empty_failed);
    CHECK(close_ok);
    CHECK(fs.handles.empty());
    // Scoped bridge must restore nullptr on exit.
    CHECK(playgta6::os::win32::get_httpfs_client() == nullptr);
}

void test_program_boot_preserves_existing_httpfs_client() {
    FakeFs fs;
    setup_synthetic_program_fs(fs);
    WasmBootConfig cfg = make_cfg(fs);
    cfg.read_chunk_bytes = 4096;

    const Pg6ProgramModule mods[] = {
        {"game_scarlett_master.exe", "main", kTestExeBase, kTestImageSize,
         kTestExeBase + kTestTextRva, kPg6ProgramModuleMain, 0, kTestExeTable, &kTestExeTableCount,
         nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 1, mods, nullptr, nullptr};

    auto mb = std::make_unique<playgta6::os::HttpfsMailbox>();
    playgta6::os::HttpfsMailboxClient::init_mailbox(*mb);
    playgta6::os::HttpfsMailboxClient existing(mb.get());
    playgta6::os::win32::set_httpfs_client(&existing);

    playgta6::os::HttpfsMailboxClient* seen_inside = nullptr;
    g_test_entry_hook = [&]() {
        seen_inside = playgta6::os::win32::get_httpfs_client();
    };

    playgta6::os::VirtualMemoryArena arena;
    playgta6::os::win32::set_vmem_arena(&arena);
    const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
    g_test_entry_hook = nullptr;
    playgta6::guest::reset_boot();
    playgta6::os::win32::set_vmem_arena(nullptr);

    CHECK(r.ok);
    CHECK(seen_inside == &existing);
    CHECK(playgta6::os::win32::get_httpfs_client() == &existing);
    playgta6::os::win32::set_httpfs_client(nullptr);
}

void test_program_boot_shader_pack_loading() {
    namespace dg = playgta6::gpu::d3d12_guest;
    namespace dx = playgta6::gpu::d3d12_xs_exports;

    const Pg6ProgramModule mods[] = {
        {"game_scarlett_master.exe", "main", kTestExeBase, kTestImageSize,
         kTestExeBase + kTestTextRva, kPg6ProgramModuleMain, 0, kTestExeTable, &kTestExeTableCount,
         nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 1, mods, nullptr, nullptr};

    auto run_and_probe = [&](FakeFs& fs,
                             const WasmBootConfig& cfg,
                             uint64_t vs_id,
                             uint64_t ps_id,
                             bool& resolved_out,
                             uint64_t& vs_key_out,
                             uint64_t& ps_key_out,
                             bool& unknown_resolved_out) -> WasmBootResult {
        (void)fs;
        resolved_out = false;
        vs_key_out = 0;
        ps_key_out = 0;
        unknown_resolved_out = true;

        g_test_entry_hook = [&]() {
            uint64_t dev_ptr = 0;
            const int32_t hr = pg6_d3d12_d3d12_xbox_create_device(
                nullptr, nullptr, &dg::IID_ID3D12Device, &dev_ptr);
            CHECK(hr == dg::kSOk);
            CHECK(dev_ptr != 0);
            auto dev = dx::current_device();
            CHECK(static_cast<bool>(dev));
            if (!dev) {
                return;
            }
            dg::GraphicsPsoSource src;
            src.desc.vs = vs_id;
            src.desc.ps = ps_id;
            resolved_out = dev->resolve_with_pack(src, &vs_key_out, &ps_key_out);

            dg::GraphicsPsoSource unknown_src;
            unknown_src.desc.vs = 0xDEADBEEF12345678ULL;
            unknown_src.desc.ps = 0xCAFEBABE87654321ULL;
            uint64_t u_vs = 0;
            uint64_t u_ps = 0;
            unknown_resolved_out = dev->resolve_with_pack(unknown_src, &u_vs, &u_ps);
        };

        playgta6::os::VirtualMemoryArena arena;
        playgta6::os::win32::set_vmem_arena(&arena);
        dx::reset_for_tests();
        const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
        g_test_entry_hook = nullptr;
        dx::reset_for_tests();
        playgta6::guest::com::reset_com_bridge();
        playgta6::guest::reset_boot();
        playgta6::os::win32::set_vmem_arena(nullptr);
        return r;
    };

    // 1. Preferred shader pack (`shaders/splash.wgsl.pack`) resolves parsed keys.
    {
        FakeFs fs;
        setup_synthetic_program_fs(fs);
        put_bytes(fs, "shaders/splash.wgsl.pack",
                  make_synthetic_wgpk({{0x1111ULL, 1}, {0x2222ULL, 2}}));
        WasmBootConfig cfg = make_cfg(fs);
        cfg.read_chunk_bytes = 4096;

        bool resolved = false;
        uint64_t vs_key = 0;
        uint64_t ps_key = 0;
        bool unknown_resolved = true;
        const WasmBootResult r = run_and_probe(
            fs, cfg, 0x1111ULL, 0x2222ULL, resolved, vs_key, ps_key, unknown_resolved);
        CHECK(r.ok);
        CHECK(r.exit_code == 42);
        CHECK(resolved);
        CHECK(vs_key == 0x1111ULL);
        CHECK(ps_key == 0x2222ULL);
        CHECK(!unknown_resolved);
        bool saw_pack_log = false;
        bool saw_gpu_stats = false;
        for (const std::string& line : g_logs) {
            if (line == "[boot] SHADER_PACK: shaders/splash.wgsl.pack loaded (2 key(s), primary)") {
                saw_pack_log = true;
            }
            if (line.find("[boot] GPU_STATS: ") == 0) {
                saw_gpu_stats = true;
            }
        }
        CHECK(saw_pack_log);
        CHECK(saw_gpu_stats);
    }

    // 2. Missing preferred pack falls back to `shaders/fallback.wgsl.pack`.
    {
        FakeFs fs;
        setup_synthetic_program_fs(fs);
        put_bytes(fs, "shaders/fallback.wgsl.pack",
                  make_synthetic_wgpk({{0x3333ULL, 1}, {0x4444ULL, 2}}));
        WasmBootConfig cfg = make_cfg(fs);
        cfg.read_chunk_bytes = 4096;

        bool resolved = false;
        uint64_t vs_key = 0;
        uint64_t ps_key = 0;
        bool unknown_resolved = true;
        const WasmBootResult r = run_and_probe(
            fs, cfg, 0x3333ULL, 0x4444ULL, resolved, vs_key, ps_key, unknown_resolved);
        CHECK(r.ok);
        CHECK(r.exit_code == 42);
        CHECK(resolved);
        CHECK(vs_key == 0x3333ULL);
        CHECK(ps_key == 0x4444ULL);
        CHECK(!unknown_resolved);
        bool saw_fallback_notice = false;
        bool saw_pack_log = false;
        for (const std::string& line : g_logs) {
            if (line ==
                "[boot] SHADER_PACK: primary shaders/splash.wgsl.pack unavailable; falling back to "
                "shaders/fallback.wgsl.pack") {
                saw_fallback_notice = true;
            }
            if (line ==
                "[boot] SHADER_PACK: shaders/fallback.wgsl.pack loaded (2 key(s), fallback)") {
                saw_pack_log = true;
            }
        }
        CHECK(saw_fallback_notice);
        CHECK(saw_pack_log);
    }

    // 3. Missing shader packs degrade cleanly without failing boot (and previous pack is not leaked).
    {
        FakeFs fs;
        setup_synthetic_program_fs(fs);
        WasmBootConfig cfg = make_cfg(fs);
        cfg.read_chunk_bytes = 4096;

        bool resolved = true;
        uint64_t vs_key = 0;
        uint64_t ps_key = 0;
        bool unknown_resolved = true;
        const WasmBootResult r = run_and_probe(
            fs, cfg, 0x1111ULL, 0x2222ULL, resolved, vs_key, ps_key, unknown_resolved);
        CHECK(r.ok);
        CHECK(r.exit_code == 42);
        CHECK(!resolved);
        CHECK(!unknown_resolved);
        bool saw_none_log = false;
        for (const std::string& line : g_logs) {
            if (line == "[boot] SHADER_PACK: none loaded (draws will use unresolved PSOs)") {
                saw_none_log = true;
            }
        }
        CHECK(saw_none_log);
    }

    // 4. Malformed shader pack degrades cleanly without failing boot.
    {
        FakeFs fs;
        setup_synthetic_program_fs(fs);
        put_bytes(fs, "shaders/splash.wgsl.pack", {'N', 'O', 'P', 'E', 0, 1, 2, 3});
        WasmBootConfig cfg = make_cfg(fs);
        cfg.read_chunk_bytes = 4096;

        bool resolved = true;
        uint64_t vs_key = 0;
        uint64_t ps_key = 0;
        bool unknown_resolved = true;
        const WasmBootResult r = run_and_probe(
            fs, cfg, 0x1111ULL, 0x2222ULL, resolved, vs_key, ps_key, unknown_resolved);
        CHECK(r.ok);
        CHECK(r.exit_code == 42);
        CHECK(!resolved);
    }
}

struct ScopedEnvVar {
    std::string name;
    bool had_prev = false;
    std::string prev_value;

    explicit ScopedEnvVar(const char* var_name) : name(var_name) {
        if (const char* cur = std::getenv(var_name); cur != nullptr) {
            had_prev = true;
            prev_value = cur;
        }
    }

    void set(const std::string& val) const {
#ifdef _WIN32
        _putenv_s(name.c_str(), val.c_str());
#else
        setenv(name.c_str(), val.c_str(), 1);
#endif
    }

    void unset() const {
#ifdef _WIN32
        _putenv_s(name.c_str(), "");
#else
        unsetenv(name.c_str());
#endif
    }

    ~ScopedEnvVar() {
        if (had_prev) {
            set(prev_value);
        } else {
            unset();
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;
};

void test_program_boot_pipeline_dump_triggers() {
    namespace dg = playgta6::gpu::d3d12_guest;
    namespace dx = playgta6::gpu::d3d12_xs_exports;

    // Verify trailing-field aggregate initialization compatibility (without dump_pipelines_dir).
    {
        WasmBootConfig legacy_init{"boot/program-manifest.json",
                                   "boot/guest.pgim",
                                   "shaders/splash.wgsl.pack",
                                   "shaders/fallback.wgsl.pack",
                                   4096,
                                   1024 * 1024,
                                   {},
                                   {},
                                   {},
                                   {}};
        CHECK(legacy_init.dump_pipelines_dir.empty());
    }

    const Pg6ProgramModule mods[] = {
        {"game_scarlett_master.exe", "main", kTestExeBase, kTestImageSize,
         kTestExeBase + kTestTextRva, kPg6ProgramModuleMain, 0, kTestExeTable, &kTestExeTableCount,
         nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 1, mods, nullptr, nullptr};

    auto run_boot_with_hook = [&](const WasmBootConfig& cfg,
                                  std::function<void()> hook) -> WasmBootResult {
        g_test_entry_hook = std::move(hook);
        playgta6::os::VirtualMemoryArena arena;
        playgta6::os::win32::set_vmem_arena(&arena);
        dx::reset_for_tests();
        const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
        g_test_entry_hook = nullptr;
        dx::reset_for_tests();
        playgta6::guest::com::reset_com_bridge();
        playgta6::guest::reset_boot();
        playgta6::os::win32::set_vmem_arena(nullptr);
        return r;
    };

    ScopedEnvVar env_guard("PG6_DUMP_PIPELINES");

    // 1. Default WasmBootConfig does NOT enable pipeline dumping when flag & env are absent.
    {
        env_guard.unset();
        FakeFs fs;
        setup_synthetic_program_fs(fs);
        WasmBootConfig cfg = make_cfg(fs);
        cfg.read_chunk_bytes = 4096;

        bool device_created = false;
        bool dump_active = true;
        const WasmBootResult r = run_boot_with_hook(cfg, [&]() {
            uint64_t dev_ptr = 0;
            const int32_t hr = pg6_d3d12_d3d12_xbox_create_device(
                nullptr, nullptr, &dg::IID_ID3D12Device, &dev_ptr);
            CHECK(hr == dg::kSOk);
            auto dev = dx::current_device();
            device_created = static_cast<bool>(dev);
            if (dev) {
                dump_active = (dev->pipeline_dump() != nullptr);
            }
        });
        CHECK(r.ok);
        CHECK(device_created);
        CHECK(!dump_active);
        CHECK(fs.handles.empty());
        bool saw_dump_log = false;
        for (const std::string& line : g_logs) {
            if (line.find("[boot] PIPELINE_DUMP") == 0) {
                saw_dump_log = true;
            }
        }
        CHECK(!saw_dump_log);
    }

    // 2. Explicit cfg.dump_pipelines_dir enables dumping and emits PIPELINE_DUMP milestone.
    {
        env_guard.unset();
        const std::filesystem::path temp_dir =
            std::filesystem::temp_directory_path() / "pg6_test_wasm_boot_explicit_dump";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);

        FakeFs fs;
        setup_synthetic_program_fs(fs);
        WasmBootConfig cfg = make_cfg(fs);
        cfg.read_chunk_bytes = 4096;
        cfg.dump_pipelines_dir = temp_dir.string();

        bool dump_active = false;
        std::string observed_dir;
        const WasmBootResult r = run_boot_with_hook(cfg, [&]() {
            uint64_t dev_ptr = 0;
            const int32_t hr = pg6_d3d12_d3d12_xbox_create_device(
                nullptr, nullptr, &dg::IID_ID3D12Device, &dev_ptr);
            CHECK(hr == dg::kSOk);
            auto dev = dx::current_device();
            if (dev && dev->pipeline_dump() != nullptr) {
                dump_active = true;
                observed_dir = dev->pipeline_dump()->dir();
            }
        });
        CHECK(r.ok);
        CHECK(dump_active);
        CHECK(observed_dir == cfg.dump_pipelines_dir);
        CHECK(std::filesystem::exists(temp_dir / "manifest.json", ec));

        bool saw_dump_log = false;
        const std::string expected_line = "[boot] PIPELINE_DUMP: " + cfg.dump_pipelines_dir;
        for (const std::string& line : g_logs) {
            if (line == expected_line) {
                saw_dump_log = true;
            }
        }
        CHECK(saw_dump_log);

        std::filesystem::remove_all(temp_dir, ec);
    }

    // 3. PG6_DUMP_PIPELINES environment variable activates dumping when config dir is empty.
    {
        const std::filesystem::path temp_dir =
            std::filesystem::temp_directory_path() / "pg6_test_wasm_boot_env_dump";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        env_guard.set(temp_dir.string());

        FakeFs fs;
        setup_synthetic_program_fs(fs);
        WasmBootConfig cfg = make_cfg(fs);
        cfg.read_chunk_bytes = 4096;

        bool dump_active = false;
        std::string observed_dir;
        const WasmBootResult r = run_boot_with_hook(cfg, [&]() {
            uint64_t dev_ptr = 0;
            const int32_t hr = pg6_d3d12_d3d12_xbox_create_device(
                nullptr, nullptr, &dg::IID_ID3D12Device, &dev_ptr);
            CHECK(hr == dg::kSOk);
            auto dev = dx::current_device();
            if (dev && dev->pipeline_dump() != nullptr) {
                dump_active = true;
                observed_dir = dev->pipeline_dump()->dir();
            }
        });
        CHECK(r.ok);
        CHECK(dump_active);
        CHECK(observed_dir == temp_dir.string());

        bool saw_dump_log = false;
        const std::string expected_line = "[boot] PIPELINE_DUMP: " + temp_dir.string();
        for (const std::string& line : g_logs) {
            if (line == expected_line) {
                saw_dump_log = true;
            }
        }
        CHECK(saw_dump_log);

        env_guard.unset();
        std::filesystem::remove_all(temp_dir, ec);
    }

    // 4. HTTPFS `shaders/dump_pipelines.flag` triggers `PIPELINE_DUMP: /pg6_dump` and
    //    immediately closes the probed handle.
    {
        env_guard.unset();
        FakeFs fs;
        setup_synthetic_program_fs(fs);
        put_file(fs, "shaders/dump_pipelines.flag", "1");
        WasmBootConfig cfg = make_cfg(fs);
        cfg.read_chunk_bytes = 4096;

        bool handles_empty_at_entry = false;
        int closes_at_entry = 0;
        const WasmBootResult r = run_boot_with_hook(cfg, [&]() {
            handles_empty_at_entry = fs.handles.empty();
            closes_at_entry = fs.closes;
        });
        CHECK(r.ok);
        CHECK(handles_empty_at_entry);
        CHECK(closes_at_entry >= 3);  // manifest + main pgim + dump_pipelines.flag
        CHECK(fs.handles.empty());

        bool saw_dump_log = false;
        for (const std::string& line : g_logs) {
            if (line == "[boot] PIPELINE_DUMP: /pg6_dump") {
                saw_dump_log = true;
            }
        }
        CHECK(saw_dump_log);
    }
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
    test_resolve_module_pgim_path();
    test_program_boot_invalid_descriptor();
    test_program_boot_missing_module_pgim();
    test_program_boot_bad_magic();
    test_program_boot_corrupt_pgim_error_propagation();
    test_program_boot_full_synthetic_execution();
    test_program_boot_scoped_httpfs_bridge();
    test_program_boot_preserves_existing_httpfs_client();
    test_program_boot_shader_pack_loading();
    test_program_boot_pipeline_dump_triggers();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d test(s) failed\n", g_failures);
        return 1;
    }
    std::printf("test_wasm_boot: ok\n");
    return 0;
}
