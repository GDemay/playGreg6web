// #2002 lazy commit (commit-on-first-touch, Windows): guest commits of >= 64 KiB are only
// committed by the OS when first touched. A user-mode host touch faults into the commit handler,
// but a kernel write or read of a page not yet committed fails instead (ReadFile -> ERROR_NOACCESS).
// One test per class of host path that hands guest memory to the kernel or reads it from the host:
//   1. guest ReadFile (sync and OVERLAPPED) whose backend writes the buffer with a real ReadFile
//   2. DirectStorage request completion (queue + status array, and the direct read_at path)
//   3. CRT fread / fwrite on a host FILE (the stream hands large blocks to ReadFile / WriteFile)
//   4. host reads of untouched lazy pages (raw read, arena read), with a crash-reporting
//      vectored handler installed like the boot harness's, and a host probe that catches its own
//      faults (profiler unwinder) not being taken for a crash.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "guest/host_probe.h"
#include "os/httpfs_mailbox.h"
#include "os/ucrt_bridge.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xbox_scarlett.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_lazy_host_access.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

#if defined(_WIN32)

using namespace playgta6::os;
namespace w32 = playgta6::os::win32;

constexpr uint32_t kSize = 192 * 1024;  // spans three 64 KiB fault granules
constexpr uint32_t kGenericRead = 0x80000000u;
constexpr uint32_t kShareRead = 0x1u;
constexpr uint32_t kOpenExisting = 3u;
constexpr uint32_t kAttrNormal = 0x80u;

uint8_t pattern(uint64_t i) {
    return static_cast<uint8_t>((i * 131u) ^ (i >> 9) ^ 0x5Au);
}

bool matches(const uint8_t* p, uint64_t file_offset, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        if (p[i] != pattern(file_offset + i)) {
            return false;
        }
    }
    return true;
}

std::string make_host_file(const char* name, size_t len) {
    std::vector<uint8_t> bytes(len);
    for (size_t i = 0; i < len; ++i) {
        bytes[i] = pattern(i);
    }
    std::FILE* f = std::fopen(name, "wb");
    CHECK(f != nullptr);
    if (f != nullptr) {
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
    }
    return name;
}

// A fresh guest VirtualAlloc buffer, still uncommitted at the OS level (lazy).
uint8_t* fresh_lazy_buffer(uint64_t* va_out) {
    const uint64_t va = w32::VirtualAlloc(0, kSize, kMemReserve | kMemCommit,
                                          static_cast<uint32_t>(PageProtect::ReadWrite));
    CHECK(va != 0);
    *va_out = va;
    if (va == 0) {
        return nullptr;
    }
    uint8_t* p = w32::get_vmem_arena()->translate(va);
    MEMORY_BASIC_INFORMATION mbi{};
    CHECK(::VirtualQuery(p, &mbi, sizeof(mbi)) == sizeof(mbi));
    CHECK(mbi.State == MEM_RESERVE);  // the hazard is real: nothing committed yet
    return p;
}

// Positioned read from a host file with the real kernel ReadFile (writes `dst` from the kernel).
int64_t kernel_read(HANDLE h, uint64_t offset, void* dst, size_t len) {
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(offset);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD got = 0;
    if (!::ReadFile(h, dst, static_cast<DWORD>(len), &got, &ov)) {
        return ::GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    }
    return static_cast<int64_t>(got);
}

// Counts access violations it sees outside a host probe, like the boot harness's crash handler
// (which reports them as host faults).
volatile long g_crash_handler_avs = 0;
LONG CALLBACK crash_like_handler(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION &&
        !playgta6::guest::host_probe_active()) {
        InterlockedIncrement(&g_crash_handler_avs);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

void test_readfile_into_lazy_buffer(HANDLE host) {
    // Negative control: a kernel write straight into the untouched lazy buffer fails.
    {
        uint64_t va = 0;
        uint8_t* p = fresh_lazy_buffer(&va);
        if (p != nullptr) {
            CHECK(kernel_read(host, 0, p, kSize) < 0);
            CHECK(w32::VirtualFree(va, 0, kMemRelease));
        }
    }

    HttpfsDirectBackend backend;
    backend.max_chunk_bytes = 1u << 20;  // one kernel ReadFile spans the whole destination
    backend.open = [](std::string_view) -> int32_t { return 1; };
    backend.read = [host](int32_t, uint64_t offset, void* dst, size_t len) { return kernel_read(host, offset, dst, len); };
    backend.close = [](int32_t) { return true; };
    HttpfsMailboxClient client(backend);
    w32::set_httpfs_client(&client);

    const w32::Handle h = w32::CreateFileA("G:\\lazy\\stream.bin", kGenericRead, kShareRead, nullptr, kOpenExisting,
                                           kAttrNormal);
    CHECK(h != ~uint64_t{0});  // win32::INVALID_HANDLE_VALUE (shadowed by <windows.h>)

    // Synchronous ReadFile at the file pointer.
    uint64_t va = 0;
    uint8_t* p = fresh_lazy_buffer(&va);
    if (p != nullptr) {
        uint32_t got = 0;
        CHECK(w32::ReadFile(h, p, kSize, &got, nullptr));
        CHECK(got == kSize);
        CHECK(matches(p, 0, kSize));
        CHECK(w32::VirtualFree(va, 0, kMemRelease));
    }

    // OVERLAPPED ReadFile at an offset (completes before the call returns).
    p = fresh_lazy_buffer(&va);
    if (p != nullptr) {
        struct {
            uint64_t internal = 0, internal_high = 0;
            uint32_t offset = 0, offset_high = 0;
            uint64_t event = 0;
        } ov;
        ov.offset = 4096;
        uint32_t got = 0;
        CHECK(w32::ReadFile(h, p + 100, kSize - 4096 - 100, &got, &ov));
        CHECK(got == kSize - 4096 - 100);
        CHECK(matches(p + 100, 4096, got));
        CHECK(p[0] == 0);
        CHECK(w32::VirtualFree(va, 0, kMemRelease));
    }
    CHECK(w32::CloseHandle(h));
    w32::set_httpfs_client(nullptr);
}

void test_dstorage_into_lazy_buffer(HANDLE host) {
    using namespace playgta6::os::scarlett;
    reset_dstorage_for_testing();
    set_dstorage_host_callbacks([](std::string_view) -> int64_t { return kSize; },
                                [host](std::string_view, uint64_t offset, void* dst, size_t len) {
                                    return kernel_read(host, offset, dst, len);
                                });
    Pg6IDStorageFactory* factory = nullptr;
    CHECK(pg6_xbox_DStorageGetFactory(&kIidIDStorageFactory, reinterpret_cast<void**>(&factory)) == gdk::S_OK_HR);
    if (factory == nullptr) {
        return;
    }
    Pg6IDStorageFile* file = nullptr;
    CHECK(factory->OpenFileA("lazy/stream.bin", kIidIDStorageFile, reinterpret_cast<void**>(&file)) == gdk::S_OK_HR);
    Pg6DStorageQueueDesc qdesc{};
    qdesc.capacity = 16;
    qdesc.name = "lazy_queue";
    Pg6IDStorageQueue* queue = nullptr;
    CHECK(factory->CreateQueue(&qdesc, kIidIDStorageQueue, reinterpret_cast<void**>(&queue)) == gdk::S_OK_HR);
    Pg6IDStorageStatusArray* status = nullptr;
    CHECK(factory->CreateStatusArray(2, "lazy_status", kIidIDStorageStatusArray, reinterpret_cast<void**>(&status)) ==
          gdk::S_OK_HR);
    if (file != nullptr && queue != nullptr && status != nullptr) {
        // Queue request -> completion into a status slot.
        uint64_t va = 0;
        uint8_t* p = fresh_lazy_buffer(&va);
        if (p != nullptr) {
            queue->EnqueueRead(file, 8192, p, kSize - 8192);
            queue->EnqueueStatus(status, 0);
            queue->Submit();
            CHECK(status->IsComplete(0));
            CHECK(status->GetHResult(0) == gdk::S_OK_HR);
            CHECK(matches(p, 8192, kSize - 8192));
            CHECK(w32::VirtualFree(va, 0, kMemRelease));
        }
        // Direct read_at path.
        p = fresh_lazy_buffer(&va);
        if (p != nullptr) {
            CHECK(dstorage_file_read_at(file, 0, p, kSize) == gdk::S_OK_HR);
            CHECK(matches(p, 0, kSize));
            CHECK(w32::VirtualFree(va, 0, kMemRelease));
        }
    }
    if (status != nullptr) status->Release();
    if (queue != nullptr) queue->Release();
    if (file != nullptr) file->Release();
    factory->Release();
    reset_dstorage_for_testing();
}

void test_crt_host_stream(const std::string& path) {
    // fread: an unbuffered host stream reads straight into the guest buffer.
    uint64_t va = 0;
    uint8_t* p = fresh_lazy_buffer(&va);
    if (p != nullptr) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        CHECK(f != nullptr);
        if (f != nullptr) {
            std::setvbuf(f, nullptr, _IONBF, 0);
            CHECK(pg6_crt_fread(p, 1, kSize, f) == kSize);
            CHECK(matches(p, 0, kSize));
            std::fclose(f);
        }
        CHECK(w32::VirtualFree(va, 0, kMemRelease));
    }

    // fwrite: a kernel read of a buffer the guest never touched (zeros) must not fail.
    const char* out_path = "test_lazy_host_access_out.tmp";
    p = fresh_lazy_buffer(&va);
    if (p != nullptr) {
        // Negative control: the kernel cannot read the untouched pages.
        HANDLE raw = ::CreateFileA(out_path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        CHECK(raw != INVALID_HANDLE_VALUE);
        if (raw != INVALID_HANDLE_VALUE) {
            DWORD put = 0;
            CHECK(!::WriteFile(raw, p, kSize, &put, nullptr));
            ::CloseHandle(raw);
        }
        std::FILE* f = std::fopen(out_path, "wb");
        CHECK(f != nullptr);
        if (f != nullptr) {
            std::setvbuf(f, nullptr, _IONBF, 0);
            CHECK(pg6_crt_fwrite(p, 1, kSize, f) == kSize);
            std::fclose(f);
        }
        std::FILE* rf = std::fopen(out_path, "rb");
        CHECK(rf != nullptr);
        if (rf != nullptr) {
            std::vector<uint8_t> back(kSize, 0xEE);
            CHECK(std::fread(back.data(), 1, back.size(), rf) == kSize);
            bool zero = true;
            for (uint8_t b : back) zero = zero && b == 0;
            CHECK(zero);
            std::fclose(rf);
        }
        CHECK(w32::VirtualFree(va, 0, kMemRelease));
    }
    std::remove(out_path);
}

#if defined(_MSC_VER)
// Reads `p` like the profiler's unwinder: inside a host probe, under its own __try.
bool probe_faults(const void* p) {
    bool caught = false;
    playgta6::guest::host_probe_enter();
    __try {
        (void)*static_cast<volatile const uint64_t*>(p);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        caught = true;
    }
    playgta6::guest::host_probe_leave();
    return caught;
}
#endif

void test_host_reads(LONG avs_before) {
    // Raw host reads of untouched lazy pages fault into the commit handler (zeros), never into a
    // crash handler at the end of the vectored chain.
    uint64_t va = 0;
    uint8_t* p = fresh_lazy_buffer(&va);
    if (p != nullptr) {
        const volatile uint8_t* vp = p;
        CHECK(vp[0] == 0 && vp[kSize - 1] == 0 && vp[70000] == 0);
        std::vector<uint8_t> buf(4096, 0xCC);
        CHECK(w32::get_vmem_arena()->read(va + 131072, buf.data(), buf.size()));
        CHECK(buf[0] == 0 && buf[4095] == 0);
        CHECK(w32::VirtualFree(va, 0, kMemRelease));
    }
    CHECK(g_crash_handler_avs == avs_before);

#if defined(_MSC_VER)
    // A host probe (profiler unwinder) reading an unmapped address under its own __try: the
    // fault reaches its __except and a crash handler first in the chain lets it pass.
    void* first = ::AddVectoredExceptionHandler(1, &crash_like_handler);
    void* hole = ::VirtualAlloc(nullptr, 65536, MEM_RESERVE, PAGE_NOACCESS);
    CHECK(hole != nullptr);
    CHECK(probe_faults(static_cast<uint8_t*>(hole) + 0x588));
    CHECK(g_crash_handler_avs == avs_before);
    CHECK(!playgta6::guest::host_probe_active());
    ::VirtualFree(hole, 0, MEM_RELEASE);
    ::RemoveVectoredExceptionHandler(first);
#endif
}

#endif  // _WIN32

}  // namespace

int main() {
#if defined(_WIN32)
    set_lazy_commit(true);
    // Installed before any lazy commit registers the commit handler, at the end of the chain
    // (like the boot harness's crash handler): it must never see a lazy-commit fault.
    void* last = ::AddVectoredExceptionHandler(0, &crash_like_handler);
    const std::string path = make_host_file("test_lazy_host_access.tmp", kSize);
    HANDLE host = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(host != INVALID_HANDLE_VALUE);
    if (host != INVALID_HANDLE_VALUE) {
        test_readfile_into_lazy_buffer(host);
        test_dstorage_into_lazy_buffer(host);
        ::CloseHandle(host);
    }
    test_crt_host_stream(path);
    test_host_reads(0);
    CHECK(g_crash_handler_avs == 0);
    ::RemoveVectoredExceptionHandler(last);
    std::remove(path.c_str());
#else
    std::printf("test_lazy_host_access: lazy commit is Windows-only; skipped\n");
#endif
    if (g_failures != 0) {
        std::fprintf(stderr, "test_lazy_host_access: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_lazy_host_access: all checks passed\n");
    return 0;
}
