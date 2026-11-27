// Windows only: the exit backstop (#663), shared by playgta6_boot and tests/test_minidump_e2e.cpp.
// The report path of a fault or watchdog stop runs arbitrary runtime code (report building,
// device shutdown, flushes) while other threads may hold any lock; a hang there used to keep the
// process alive until the runner's timeout with no report at all. A thread created at startup (no
// thread creation, heap or CRT lock needed when it fires) waits the delay after arming, prints
// every thread's stack and writes an all-thread dump through the minidump helper, then terminates
// with the armed exit code.
#pragma once
#ifdef _WIN32
#include "guest/minidump_writer.h"

#include <atomic>
#include <cstring>

namespace playgta6::guest::backstop {

inline std::atomic<int> g_code{-1};
inline HANDLE g_event = nullptr;
inline DWORD g_delay_ms = 30000;
inline int g_default_code = 10;
inline char g_dump_path[MAX_PATH * 2] = {};  // empty: stacks only

// stderr without the CRT stream lock (a stuck thread may hold it).
inline void raw_stderr(const char* text) {
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), text, static_cast<DWORD>(std::strlen(text)), &written, nullptr);
}

inline DWORD WINAPI thread_main(void*) {
    WaitForSingleObject(g_event, INFINITE);
    Sleep(g_delay_ms);
    raw_stderr("[harness] exit backstop: the fault/stop report path did not finish in time; thread stacks follow\n");
    const bool ok = minidump::Writer::dump_threads(g_dump_path[0] != 0 ? g_dump_path : nullptr, "report path stuck", 20000);
    if (ok && g_dump_path[0] != 0) {
        raw_stderr("[harness] exit backstop: all-thread dump written to ");
        raw_stderr(g_dump_path);
        raw_stderr("\n");
    }
    raw_stderr("[harness] exit backstop: terminating\n");
    const int code = g_code.load();
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code >= 0 ? code : g_default_code));
    return 0;
}

// dump_path may be null (stacks only). Call after Writer::init().
inline void start(const char* dump_path, int default_code, DWORD delay_ms = 30000) {
    g_delay_ms = delay_ms;
    g_default_code = default_code;
    if (dump_path != nullptr) std::strncpy(g_dump_path, dump_path, sizeof(g_dump_path) - 1);
    g_event = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    if (g_event == nullptr) return;
    HANDLE t = CreateThread(nullptr, 64 * 1024, &thread_main, nullptr, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (t != nullptr) CloseHandle(t);
}

inline void arm(int code) {
    int expected = -1;
    if (g_code.compare_exchange_strong(expected, code) && g_event != nullptr) SetEvent(g_event);
}

}  // namespace playgta6::guest::backstop
#endif
