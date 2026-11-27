// Windows only: runs the real playgta6_boot (no game) through its watchdog path. A tiny synthetic
// image is booted with --test-stall (a test seam: the boot thread hangs once the boot is active),
// then --watchdog-ms fires: all-thread stacks on stderr, an all-thread minidump, the boot report
// with fault_record.minidump, exit 10 -- all within a time bound.
// Usage: test_boot_watchdog_e2e <playgta6_boot.exe> [minidump_stack.py]
#ifdef _WIN32
#include "os/image_loader.h"

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace playgta6::os::pgim;

static uint64_t fnv(const uint8_t* d, size_t n) {  // image_loader.cpp is not linked: same FNV-1a
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) { h ^= d[i]; h *= 0x100000001b3ULL; }
    return h;
}

static void put32(std::vector<uint8_t>& o, uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back(uint8_t(v >> (8 * i))); }
static void put64(std::vector<uint8_t>& o, uint64_t v) { for (int i = 0; i < 8; ++i) o.push_back(uint8_t(v >> (8 * i))); }
static void set16(std::vector<uint8_t>& b, size_t at, uint16_t v) { b[at] = uint8_t(v); b[at + 1] = uint8_t(v >> 8); }
static void set32(std::vector<uint8_t>& b, size_t at, uint32_t v) { for (int i = 0; i < 4; ++i) b[at + i] = uint8_t(v >> (8 * i)); }
static void set64(std::vector<uint8_t>& b, size_t at, uint64_t v) { for (int i = 0; i < 8; ++i) b[at + i] = uint8_t(v >> (8 * i)); }

struct Sec { const char* name; uint32_t rva, vsize, prot, flags; std::vector<uint8_t> data; };

// Minimal x64 image: PE headers + one int3-filled .text, no TLS, no imports.
static std::vector<uint8_t> build_image() {
    constexpr uint64_t base = 0x140000000ULL;
    constexpr uint32_t lfanew = 0x80, size_of_image = 0x2000, entry = 0x1000;
    std::vector<uint8_t> headers(0x400, 0);
    set16(headers, 0, 0x5A4D);
    set32(headers, 0x3C, lfanew);
    set32(headers, lfanew, 0x00004550);
    set16(headers, lfanew + 4, 0x8664);
    set16(headers, lfanew + 20, 240);
    const size_t opt = lfanew + 24;
    set16(headers, opt, 0x20B);
    set32(headers, opt + 16, entry);
    set64(headers, opt + 24, base);
    set16(headers, opt + 68, 2);
    set64(headers, opt + 72, 0x40000);
    set32(headers, opt + 108, 16);
    set32(headers, opt + 56, size_of_image);
    std::vector<Sec> secs = {
        {"<headers>", 0, 0x400, kProtRead, kSectionFlagHeaders, headers},
        {".text", 0x1000, 0x400, kProtRead | kProtExecute, 0, std::vector<uint8_t>(0x400, 0xCC)},
    };
    const uint64_t tables = 64 + 64 * secs.size();
    std::vector<uint64_t> offsets;
    uint64_t cursor = tables;
    for (const Sec& s : secs) {
        const uint64_t off = (cursor + 4095) / 4096 * 4096;
        offsets.push_back(off);
        cursor = off + s.data.size();
    }
    std::vector<uint8_t> out = {'P', 'G', 'I', 'M'};
    put32(out, 1); put32(out, 64); put32(out, 64); put32(out, kFlagHeadersMapped); put32(out, 4096);
    put64(out, base); put32(out, size_of_image); put32(out, 0x1000); put32(out, entry);
    put32(out, uint32_t(secs.size())); put32(out, 0); put32(out, 0); put64(out, cursor);
    for (size_t i = 0; i < secs.size(); ++i) {
        std::string name = secs[i].name;
        name.resize(16, '\0');
        out.insert(out.end(), name.begin(), name.end());
        put32(out, secs[i].rva); put32(out, secs[i].vsize); put32(out, secs[i].prot); put32(out, secs[i].flags);
        put64(out, offsets[i]); put64(out, secs[i].data.size());
        put64(out, fnv(secs[i].data.data(), secs[i].data.size()));
        put32(out, 0); put32(out, 0);
    }
    for (size_t i = 0; i < secs.size(); ++i) {
        out.resize(size_t(offsets[i]), 0);
        out.insert(out.end(), secs[i].data.begin(), secs[i].data.end());
    }
    return out;
}

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static bool has(const std::string& s, const char* t) { return s.find(t) != std::string::npos; }
static int fail(const std::string& what, const std::string& detail) {
    std::fprintf(stderr, "FAIL: %s\n%s\n", what.c_str(), detail.c_str());
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const char* script = argc > 2 ? argv[2] : nullptr;
    char tmp[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp);
    const std::string dir = std::string(tmp) + "pg6_bootwd_" + std::to_string(GetCurrentProcessId()) + "\\";
    CreateDirectoryA(dir.c_str(), nullptr);
    const std::string pgim = dir + "stall.pgim", report = dir + "report.json", err_path = dir + "stderr.txt";
    {
        const std::vector<uint8_t> img = build_image();
        std::ofstream(pgim, std::ios::binary).write(reinterpret_cast<const char*>(img.data()), std::streamsize(img.size()));
    }
    constexpr DWORD kWatchdogMs = 3000, kMaxMs = 60000;
    std::string cmd = std::string("\"") + argv[1] + "\" --pgim \"" + pgim + "\" --report \"" + report +
                      "\" --watchdog-ms " + std::to_string(kWatchdogMs) + " --quiet --test-stall";
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE err = CreateFileA(err_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = err;
    si.hStdError = err;
    PROCESS_INFORMATION pi{};
    const DWORD t0 = GetTickCount();
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi))
        return fail("CreateProcess", cmd);
    CloseHandle(err);
    const bool exited = WaitForSingleObject(pi.hProcess, kMaxMs) == WAIT_OBJECT_0;
    const DWORD elapsed = GetTickCount() - t0;
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    if (!exited) TerminateProcess(pi.hProcess, 99);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    const std::string child_err = slurp(err_path);
    if (!exited) return fail("playgta6_boot still running after the bound", child_err);
    if (code != 10 || elapsed < kWatchdogMs)
        return fail("exit " + std::to_string(code) + " after " + std::to_string(elapsed) + " ms, want 10 after >= watchdog",
                    child_err);
    if (!has(child_err, "[harness] watchdog:") || !has(child_err, "[hang-stack] thread ") ||
        !has(child_err, "all-thread dump written to"))
        return fail("watchdog stderr text missing", child_err);
    const std::string body = slurp(report);
    if (!has(body, "host_watchdog_timeout") || !has(body, "\"fault_record\": {\"minidump\": \""))
        return fail("report lacks the watchdog fault_record / minidump", body + "\n" + child_err);
    std::string dump;  // the report names the dump (JSON-escaped path)
    {
        const size_t at = body.find("\"minidump\": \"");
        const size_t b = at + 13, e = body.find('"', b);
        for (size_t i = b; i < e; ++i) {
            if (body[i] == '\\' && i + 1 < e) ++i;
            dump += body[i];
        }
    }
    std::ifstream df(dump, std::ios::binary | std::ios::ate);
    const long long size = df ? static_cast<long long>(df.tellg()) : -1;
    df.close();
    if (size <= 1024) return fail("dump missing or empty", dump + " size " + std::to_string(size));
    if (script != nullptr) {
        const std::string out = dump + ".txt";
        const std::string py =
            std::string("python -I \"") + script + "\" \"" + report + "\" --all-threads > \"" + out + "\"";
        if (std::system(py.c_str()) != 0) return fail("minidump_stack.py failed", slurp(out));
        const std::string text = slurp(out);
        if (!has(text, "thread ") || !has(text, "playgta6_boot.exe+0x")) return fail("parsed dump lacks stacks", text);
    }
    std::printf("boot watchdog e2e OK (%lu ms, dump %lld bytes)\n", static_cast<unsigned long>(elapsed), size);
    return 0;
}
#else
int main() { return 0; }
#endif
