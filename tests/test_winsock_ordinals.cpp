#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "abi/playgta6_abi.h"
#include "os/guest_heap.h"
#include "os/host_marshallers.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/winsock_ordinals.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_winsock_ordinals.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace ws = playgta6::os::winsock;
namespace win32 = playgta6::os::win32;
using win32::proc_address;

const char* make_int_resource(uint32_t ordinal) {
    return reinterpret_cast<const char*>(static_cast<uintptr_t>(ordinal));
}

struct ExpectedOrdinal {
    uint32_t ordinal;
    const char* name;
    Pg6ProcAddress address;
};

const ExpectedOrdinal kExpected[] = {
    {1, "accept", proc_address(&pg6_ws2_accept)},
    {2, "bind", proc_address(&pg6_ws2_bind)},
    {3, "closesocket", proc_address(&pg6_ws2_closesocket)},
    {4, "connect", proc_address(&pg6_ws2_connect)},
    {6, "getsockname", proc_address(&pg6_ws2_getsockname)},
    {7, "getsockopt", proc_address(&pg6_ws2_getsockopt)},
    {9, "htons", proc_address(&pg6_ws2_htons)},
    {10, "inet_addr", proc_address(&pg6_ws2_inet_addr)},
    {11, "inet_ntoa", proc_address(&pg6_ws2_inet_ntoa)},
    {13, "listen", proc_address(&pg6_ws2_listen)},
    {16, "recv", proc_address(&pg6_ws2_recv)},
    {17, "recvfrom", proc_address(&pg6_ws2_recvfrom)},
    {18, "select", proc_address(&pg6_ws2_select)},
    {19, "send", proc_address(&pg6_ws2_send)},
    {20, "sendto", proc_address(&pg6_ws2_sendto)},
    {21, "setsockopt", proc_address(&pg6_ws2_setsockopt)},
    {22, "shutdown", proc_address(&pg6_ws2_shutdown)},
    {23, "socket", proc_address(&pg6_ws2_socket)},
    {52, "gethostbyname", proc_address(&pg6_ws2_gethostbyname)},
    {57, "gethostname", proc_address(&pg6_ws2_gethostname)},
    {111, "WSAGetLastError", proc_address(&pg6_ws2_WSAGetLastError)},
    {115, "WSAStartup", proc_address(&pg6_ws2_WSAStartup)},
    {151, "__WSAFDIsSet", proc_address(&pg6_ws2___WSAFDIsSet)},
};

int32_t last_error() {
    return pg6_ws2_WSAGetLastError();
}

template <typename Fn>
Fn as_fn(Pg6ProcAddress address) {
    return reinterpret_cast<Fn>(static_cast<uintptr_t>(address));
}

void test_registration() {
    CHECK(sizeof(kExpected) / sizeof(kExpected[0]) == ws::kOrdinalExportCount);
    CHECK(pg6_winsock_ordinals_register_all() == 25u);
    CHECK(pg6_winsock_ordinals_register_all() == ws::kRegisteredExportCount);

    const Pg6Handle module = pg6_win32_get_module_handle_a("WS2_32.dll");
    CHECK(module != 0);
    CHECK(pg6_win32_get_module_handle_a("ws2_32") == module);

    for (const auto& e : kExpected) {
        CHECK(e.address != 0);
        CHECK(win32::GetProcAddress(module, make_int_resource(e.ordinal)) == e.address);
        CHECK(pg6_win32_get_proc_address(module, make_int_resource(e.ordinal)) == e.address);
        CHECK(pg6_loader_get_proc_address_by_ordinal(module, e.ordinal) == e.address);
        CHECK(win32::resolve_import_ordinal("ws2_32.dll", e.ordinal) == e.address);
        CHECK(pg6_loader_get_proc_address_by_name(module, e.name) == e.address);
        CHECK(win32::resolve_import("WS2_32", e.name) == e.address);
    }
    CHECK(win32::resolve_import("WS2_32.dll", "getaddrinfo") == proc_address(&pg6_ws2_getaddrinfo));
    CHECK(win32::resolve_import("WS2_32.dll", "freeaddrinfo") ==
          proc_address(&pg6_ws2_freeaddrinfo));

    for (uint32_t ordinal : {5u, 8u, 12u, 14u, 15u, 24u, 51u, 116u, 500u}) {
        win32::SetLastError(0);
        CHECK(pg6_loader_get_proc_address_by_ordinal(module, ordinal) == 0);
        CHECK(win32::GetLastError() == win32::ERROR_PROC_NOT_FOUND);
    }
    CHECK(pg6_loader_get_proc_address_by_name(module, "HTONS") == 0);

    auto htons_fn = as_fn<uint16_t (*)(uint16_t)>(
        win32::GetProcAddress(module, make_int_resource(ws::kOrdHtons)));
    CHECK(htons_fn(0x1234u) == 0x3412u);
    auto inet_addr_fn = as_fn<uint32_t (*)(const char*)>(
        pg6_loader_get_proc_address_by_ordinal(module, ws::kOrdInetAddr));
    CHECK(inet_addr_fn("1.2.3.4") == 0x04030201u);
}

void test_htons() {
    CHECK(pg6_ws2_htons(0x0000u) == 0x0000u);
    CHECK(pg6_ws2_htons(0x1234u) == 0x3412u);
    CHECK(pg6_ws2_htons(0x00FFu) == 0xFF00u);
    CHECK(pg6_ws2_htons(0xFFFFu) == 0xFFFFu);
    CHECK(pg6_ws2_htons(80u) == 0x5000u);
    CHECK(pg6_ws2_htons(pg6_ws2_htons(0xBEEFu)) == 0xBEEFu);

    const uint16_t port = pg6_ws2_htons(0x1F90u);
    uint8_t bytes[2];
    std::memcpy(bytes, &port, sizeof(bytes));
    CHECK(bytes[0] == 0x1Fu && bytes[1] == 0x90u);
}

bool inet_bytes_are(const char* text, uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    const uint32_t value = pg6_ws2_inet_addr(text);
    uint8_t bytes[4];
    std::memcpy(bytes, &value, sizeof(bytes));
    return bytes[0] == a && bytes[1] == b && bytes[2] == c && bytes[3] == d;
}

void test_inet_addr() {
    CHECK(inet_bytes_are("127.0.0.1", 127, 0, 0, 1));
    CHECK(inet_bytes_are("192.168.1.20", 192, 168, 1, 20));
    CHECK(inet_bytes_are("0.0.0.0", 0, 0, 0, 0));
    CHECK(pg6_ws2_inet_addr("0.0.0.0") == 0u);
    CHECK(pg6_ws2_inet_addr("1.2.3.4") == 0x04030201u);
    CHECK(pg6_ws2_inet_addr("10.0.0.1") == 0x0100000Au);
    CHECK(pg6_ws2_inet_addr("255.255.255.255") == ws::kInetAddrNone);

    CHECK(inet_bytes_are("127.1", 127, 0, 0, 1));
    CHECK(inet_bytes_are("10.1.258", 10, 1, 1, 2));
    CHECK(inet_bytes_are("16909060", 1, 2, 3, 4));
    CHECK(inet_bytes_are("0x7f.0.0.1", 127, 0, 0, 1));
    CHECK(inet_bytes_are("0X7F000001", 127, 0, 0, 1));
    CHECK(inet_bytes_are("010.0.0.1", 8, 0, 0, 1));
    CHECK(inet_bytes_are("1.2.3.4 trailing", 1, 2, 3, 4));

    const char* invalid[] = {"",          "abc",        "1.2.3.256", "1.2.3.4.5",   "1..2",
                             "1.2.3.",    ".1.2.3",     "08.0.0.1",  "1.2.3.4x",    "4294967296",
                             "256.1.1.1", "1.16777216", "1.2.65536", "0x",          "-1.2.3.4",
                             " 1.2.3.4",  "0xg.1.1.1"};
    for (const char* text : invalid) {
        if (pg6_ws2_inet_addr(text) != ws::kInetAddrNone) {
            std::fprintf(stderr, "inet_addr accepted \"%s\"\n", text);
            CHECK(false);
        }
    }
    CHECK(pg6_ws2_inet_addr(nullptr) == ws::kInetAddrNone);
}

void test_inet_ntoa() {
    playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
    char* p_loopback = pg6_ws2_inet_ntoa(Pg6InAddr{0x0100007Fu});
    CHECK(p_loopback != nullptr && std::strcmp(p_loopback, "127.0.0.1") == 0);
    // Direct shim must return a guest-accessible buffer (never a raw host TLS pointer).
    const uint64_t loopback_va = heap.to_guest(p_loopback);
    CHECK(loopback_va != 0 && heap.owns(loopback_va));

    char* p_second = pg6_ws2_inet_ntoa(Pg6InAddr{0x04030201u});
    CHECK(p_second == p_loopback && std::strcmp(p_second, "1.2.3.4") == 0);
    CHECK(std::strcmp(pg6_ws2_inet_ntoa(Pg6InAddr{0u}), "0.0.0.0") == 0);
    CHECK(std::strcmp(pg6_ws2_inet_ntoa(Pg6InAddr{ws::kInetAddrNone}), "255.255.255.255") == 0);

    const uint32_t parsed = pg6_ws2_inet_addr("192.168.10.20");
    CHECK(std::strcmp(pg6_ws2_inet_ntoa(Pg6InAddr{parsed}), "192.168.10.20") == 0);

    // WS2_32!#11 guest thunk (pg6_gthunk_inet_ntoa): Win64 by-value in_addr in ECX, returns guest VA.
    playgta6::GuestState state{};
    const uint64_t stack = heap.alloc(64);
    state.gpr[playgta6::GuestReg::Rsp] = stack;
    state.gpr[playgta6::GuestReg::Rcx] = 0xCAFEBABE00000000ull | parsed;
    pg6_gthunk_inet_ntoa(&state);
    CHECK(state.gpr[playgta6::GuestReg::Rsp] == stack + 8);
    const uint64_t gva = state.gpr[playgta6::GuestReg::Rax];
    CHECK(gva != 0);
    if (gva != 0) {
        const auto* host_str = reinterpret_cast<const char*>(heap.to_host(gva));
        CHECK(host_str != nullptr && std::strcmp(host_str, "192.168.10.20") == 0);
    }

    // Per-thread isolation for both pg6_ws2_inet_ntoa and pg6_gthunk_inet_ntoa.
    uint64_t worker_shim_va = 0;
    uint64_t worker_gthunk_va = 0;
    std::string worker_shim_text;
    std::string worker_gthunk_text;
    std::thread worker([&] {
        char* wp = pg6_ws2_inet_ntoa(Pg6InAddr{0x08080808u});
        if (wp != nullptr) {
            worker_shim_va = heap.to_guest(wp);
            worker_shim_text = wp;
        }
        playgta6::GuestState ws_state{};
        const uint64_t wstack = heap.alloc(64);
        ws_state.gpr[playgta6::GuestReg::Rsp] = wstack;
        ws_state.gpr[playgta6::GuestReg::Rcx] = 0x01010101u;
        pg6_gthunk_inet_ntoa(&ws_state);
        worker_gthunk_va = ws_state.gpr[playgta6::GuestReg::Rax];
        if (worker_gthunk_va != 0) {
            worker_gthunk_text = reinterpret_cast<const char*>(heap.to_host(worker_gthunk_va));
        }
    });
    worker.join();
    CHECK(worker_shim_va != 0 && worker_shim_va != loopback_va && worker_shim_text == "8.8.8.8");
    CHECK(worker_gthunk_va != 0 && worker_gthunk_va != gva && worker_gthunk_text == "1.1.1.1");
    CHECK(std::strcmp(p_loopback, "192.168.10.20") == 0);
}

void test_gethostname() {
    char buffer[64];
    std::memset(buffer, 'x', sizeof(buffer));
    CHECK(pg6_ws2_gethostname(buffer, sizeof(buffer)) == 0);
    CHECK(std::strcmp(buffer, "xbox-offline") == 0);

    char exact[13];
    CHECK(pg6_ws2_gethostname(exact, sizeof(exact)) == 0);
    CHECK(std::strcmp(exact, ws::kHostName) == 0);

    char small[12];
    pg6_ws2_WSASetLastError(0);
    CHECK(pg6_ws2_gethostname(small, sizeof(small)) == -1);
    CHECK(last_error() == ws::kWsaEFault);
    pg6_ws2_WSASetLastError(0);
    CHECK(pg6_ws2_gethostname(nullptr, 64) == -1);
    CHECK(last_error() == ws::kWsaEFault);
    pg6_ws2_WSASetLastError(0);
    CHECK(pg6_ws2_gethostname(buffer, -1) == -1);
    CHECK(last_error() == ws::kWsaEFault);
}

void test_wsa_startup() {
    CHECK(pg6_ws2_WSAStartup(0x0202u, nullptr) == ws::kWsaEFault);
    CHECK(last_error() == ws::kWsaEFault);

    Pg6WsaData data;
    std::memset(&data, 0xCC, sizeof(data));
    CHECK(pg6_ws2_WSAStartup(0x0202u, &data) == 0);
    CHECK(data.version == 0x0202u);
    CHECK(data.high_version == 0x0202u);
    CHECK(data.vendor_info == 0u);
    CHECK(std::strlen(data.description) > 0 && std::strlen(data.description) < 257);
    CHECK(std::strlen(data.system_status) > 0 && std::strlen(data.system_status) < 129);
    CHECK(last_error() == 0);

    Pg6WsaData old_data;
    CHECK(pg6_ws2_WSAStartup(0x0101u, &old_data) == 0);
    CHECK(old_data.version == 0x0101u);

    Pg6WsaData newer;
    CHECK(pg6_ws2_WSAStartup(0x0303u, &newer) == 0);
    CHECK(newer.version == 0x0202u);
}

void test_offline_network() {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const Pg6Socket bogus = 0x1234u;
    char buffer[16] = {};
    int32_t length = sizeof(buffer);

    CHECK(pg6_ws2_socket(2, 1, 6) == ws::kInvalidSocket);
    CHECK(last_error() == ws::kWsaENetUnreach);

    CHECK(pg6_ws2_connect(bogus, buffer, 16) == -1);
    CHECK(last_error() == ws::kWsaENetUnreach);
    CHECK(pg6_ws2_send(bogus, buffer, 4, 0) == -1);
    CHECK(last_error() == ws::kWsaENetUnreach);
    CHECK(pg6_ws2_recv(bogus, buffer, 4, 0) == -1);
    CHECK(last_error() == ws::kWsaENetUnreach);
    CHECK(pg6_ws2_sendto(bogus, buffer, 4, 0, buffer, 16) == -1);
    CHECK(last_error() == ws::kWsaENetUnreach);
    CHECK(pg6_ws2_recvfrom(bogus, buffer, 4, 0, buffer, &length) == -1);
    CHECK(last_error() == ws::kWsaENetUnreach);

    CHECK(pg6_ws2_bind(bogus, buffer, 16) == -1);
    CHECK(last_error() == ws::kWsaENotSock);
    CHECK(pg6_ws2_listen(bogus, 5) == -1);
    CHECK(last_error() == ws::kWsaENotSock);
    CHECK(pg6_ws2_accept(bogus, buffer, &length) == ws::kInvalidSocket);
    CHECK(last_error() == ws::kWsaENotSock);
    CHECK(pg6_ws2_getsockname(bogus, buffer, &length) == -1);
    CHECK(last_error() == ws::kWsaENotSock);
    CHECK(pg6_ws2_getsockopt(bogus, 0xFFFF, 0x1008, buffer, &length) == -1);
    CHECK(last_error() == ws::kWsaENotSock);
    CHECK(pg6_ws2_setsockopt(bogus, 0xFFFF, 0x0004, buffer, 4) == -1);
    CHECK(last_error() == ws::kWsaENotSock);
    CHECK(pg6_ws2_shutdown(bogus, 2) == -1);
    CHECK(last_error() == ws::kWsaENotSock);
    CHECK(pg6_ws2_closesocket(bogus) == -1);
    CHECK(last_error() == ws::kWsaENotSock);

    CHECK(pg6_ws2_gethostbyname("example.com") == nullptr);
    CHECK(last_error() == ws::kWsaENetUnreach);
    CHECK(pg6_ws2_gethostbyname(nullptr) == nullptr);
    CHECK(last_error() == ws::kWsaENetUnreach);

    Pg6AddrInfo* info = reinterpret_cast<Pg6AddrInfo*>(static_cast<uintptr_t>(0xBAADF00Du));
    CHECK(pg6_ws2_getaddrinfo("example.com", "443", nullptr, &info) == ws::kWsaENetUnreach);
    CHECK(info == nullptr);
    CHECK(last_error() == ws::kWsaENetUnreach);
    pg6_ws2_freeaddrinfo(nullptr);

    CHECK(pg6_ws2_select(0, nullptr, nullptr, nullptr, nullptr) == -1);
    CHECK(last_error() == ws::kWsaEInval);
    Pg6FdSet empty_read;
    Pg6FdSet empty_write;
    CHECK(pg6_ws2_select(0, &empty_read, &empty_write, nullptr, nullptr) == -1);
    CHECK(last_error() == ws::kWsaEInval);
    Pg6FdSet read_set;
    read_set.fd_count = 1;
    read_set.fd_array[0] = bogus;
    CHECK(pg6_ws2_select(0, &read_set, nullptr, nullptr, nullptr) == -1);
    CHECK(last_error() == ws::kWsaENotSock);
    CHECK(read_set.fd_count == 1 && read_set.fd_array[0] == bogus);

    CHECK(Clock::now() - start < std::chrono::seconds(2));
}

void test_fd_is_set() {
    Pg6FdSet set;
    CHECK(pg6_ws2___WSAFDIsSet(5, &set) == 0);
    CHECK(pg6_ws2___WSAFDIsSet(0, &set) == 0);
    CHECK(pg6_ws2___WSAFDIsSet(5, nullptr) == 0);

    set.fd_count = 3;
    set.fd_array[0] = 5;
    set.fd_array[1] = 9;
    set.fd_array[2] = 0x7FF812345678ull;
    set.fd_array[3] = 77;
    CHECK(pg6_ws2___WSAFDIsSet(5, &set) != 0);
    CHECK(pg6_ws2___WSAFDIsSet(9, &set) != 0);
    CHECK(pg6_ws2___WSAFDIsSet(0x7FF812345678ull, &set) != 0);
    CHECK(pg6_ws2___WSAFDIsSet(0x12345678ull, &set) == 0);
    CHECK(pg6_ws2___WSAFDIsSet(77, &set) == 0);
    CHECK(pg6_ws2___WSAFDIsSet(6, &set) == 0);

    Pg6FdSet full;
    for (uint32_t i = 0; i < kPg6FdSetSize; ++i) {
        full.fd_array[i] = 1000u + i;
    }
    full.fd_count = 0xFFFFFFFFu;
    CHECK(pg6_ws2___WSAFDIsSet(1000u + kPg6FdSetSize - 1, &full) != 0);
    CHECK(pg6_ws2___WSAFDIsSet(1000u + kPg6FdSetSize, &full) == 0);
    CHECK(offsetof(Pg6FdSet, fd_array) == 8u);
}

void test_thread_local_last_error() {
    CHECK(pg6_ws2_connect(1, nullptr, 0) == -1);
    CHECK(last_error() == ws::kWsaENetUnreach);

    int32_t fresh_error = -1;
    int32_t worker_after_socket = -1;
    int32_t worker_after_lookup = -1;
    std::thread worker([&] {
        fresh_error = last_error();
        pg6_ws2_socket(2, 2, 17);
        worker_after_socket = last_error();
        pg6_ws2_gethostbyname("localhost");
        worker_after_lookup = last_error();
    });
    worker.join();
    CHECK(fresh_error == 0);
    CHECK(worker_after_socket == ws::kWsaENetUnreach);
    CHECK(worker_after_lookup == ws::kWsaENetUnreach);
    CHECK(last_error() == ws::kWsaENetUnreach);

    char name[32];
    CHECK(pg6_ws2_gethostname(name, sizeof(name)) == 0);
    CHECK(pg6_ws2_htons(1) == 0x0100u);
    CHECK(last_error() == ws::kWsaENetUnreach);

    pg6_ws2_WSASetLastError(ws::kWsaEInval);
    CHECK(last_error() == ws::kWsaEInval);
    pg6_ws2_WSASetLastError(0);
    CHECK(last_error() == 0);
}

}  // namespace

int main() {
    test_registration();
    test_htons();
    test_inet_addr();
    test_inet_ntoa();
    test_gethostname();
    test_wsa_startup();
    test_offline_network();
    test_fd_is_set();
    test_thread_local_last_error();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("test_winsock_ordinals: all checks passed\n");
    return 0;
}
