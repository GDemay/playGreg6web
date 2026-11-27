#include "os/winsock_ordinals.h"

#include <cstdio>
#include <cstring>
#include <mutex>

#include "os/guest_heap.h"
#include "os/win32_loader.h"

namespace ws = playgta6::os::winsock;
namespace win32 = playgta6::os::win32;

namespace {

thread_local int32_t t_wsa_last_error = 0;

void set_wsa_error(int32_t error) {
    t_wsa_last_error = error;
}

int32_t fail_socket_error(int32_t error) {
    set_wsa_error(error);
    return ws::kSocketError;
}

Pg6Socket fail_invalid_socket(int32_t error) {
    set_wsa_error(error);
    return ws::kInvalidSocket;
}

uint32_t fd_count(const Pg6FdSet* set) {
    if (set == nullptr) {
        return 0;
    }
    return set->fd_count < kPg6FdSetSize ? set->fd_count : kPg6FdSetSize;
}

constexpr uint32_t byte_swap32(uint32_t v) {
    return (v >> 24) | ((v >> 8) & 0x0000FF00u) | ((v << 8) & 0x00FF0000u) | (v << 24);
}

constexpr bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

int digit_value(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

// BSD inet_addr grammar: "a.b.c.d", "a.b.c" (c = 16 bits), "a.b" (b = 24 bits) or "a"
// (32 bits); each part decimal, octal (leading 0) or hex (0x). Host-order result.
bool parse_ipv4(const char* cp, uint32_t& host_order) {
    if (cp == nullptr || *cp == '\0') {
        return false;
    }
    uint32_t parts[4]{};
    int count = 0;
    const char* p = cp;
    for (;;) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        uint32_t base = 10;
        if (*p == '0') {
            ++p;
            if (*p == 'x' || *p == 'X') {
                base = 16;
                ++p;
                if (digit_value(*p) < 0) {
                    return false;
                }
            } else {
                base = 8;
            }
        }
        uint64_t value = 0;
        for (int d = digit_value(*p); d >= 0; d = digit_value(*p)) {
            if (static_cast<uint32_t>(d) >= base) {
                return false;
            }
            value = value * base + static_cast<uint32_t>(d);
            if (value > 0xFFFFFFFFull) {
                return false;
            }
            ++p;
        }
        if (count == 4) {
            return false;
        }
        parts[count++] = static_cast<uint32_t>(value);
        if (*p != '.') {
            break;
        }
        ++p;
    }
    if (*p != '\0' && !is_space(*p)) {
        return false;
    }
    switch (count) {
        case 1:
            host_order = parts[0];
            return true;
        case 2:
            if (parts[0] > 0xFFu || parts[1] > 0xFFFFFFu) {
                return false;
            }
            host_order = (parts[0] << 24) | parts[1];
            return true;
        case 3:
            if (parts[0] > 0xFFu || parts[1] > 0xFFu || parts[2] > 0xFFFFu) {
                return false;
            }
            host_order = (parts[0] << 24) | (parts[1] << 16) | parts[2];
            return true;
        default:
            if (parts[0] > 0xFFu || parts[1] > 0xFFu || parts[2] > 0xFFu || parts[3] > 0xFFu) {
                return false;
            }
            host_order = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3];
            return true;
    }
}

void format_ipv4(uint32_t net_order, char* out, size_t cap) {
    const uint8_t a = static_cast<uint8_t>(net_order & 0xFFu);
    const uint8_t b = static_cast<uint8_t>((net_order >> 8) & 0xFFu);
    const uint8_t c = static_cast<uint8_t>((net_order >> 16) & 0xFFu);
    const uint8_t d = static_cast<uint8_t>((net_order >> 24) & 0xFFu);
    std::snprintf(out, cap, "%u.%u.%u.%u", a, b, c, d);
}

}  // namespace

extern "C" {

Pg6Socket pg6_ws2_accept(Pg6Socket, void*, int32_t*) {
    return fail_invalid_socket(ws::kWsaENotSock);
}

int32_t pg6_ws2_bind(Pg6Socket, const void*, int32_t) {
    return fail_socket_error(ws::kWsaENotSock);
}

int32_t pg6_ws2_closesocket(Pg6Socket) {
    return fail_socket_error(ws::kWsaENotSock);
}

int32_t pg6_ws2_connect(Pg6Socket, const void*, int32_t) {
    return fail_socket_error(ws::kWsaENetUnreach);
}

int32_t pg6_ws2_getsockname(Pg6Socket, void*, int32_t*) {
    return fail_socket_error(ws::kWsaENotSock);
}

int32_t pg6_ws2_getsockopt(Pg6Socket, int32_t, int32_t, char*, int32_t*) {
    return fail_socket_error(ws::kWsaENotSock);
}

uint16_t pg6_ws2_htons(uint16_t host_short) {
    return static_cast<uint16_t>((host_short << 8) | (host_short >> 8));
}

uint32_t pg6_ws2_inet_addr(const char* cp) {
    uint32_t host_order = 0;
    if (!parse_ipv4(cp, host_order)) {
        return ws::kInetAddrNone;
    }
    return byte_swap32(host_order);
}

char* pg6_ws2_inet_ntoa(Pg6InAddr in) {
    // Format into a per-thread 16-byte buffer in guest-accessible heap memory (re-allocating
    // transparently if the guest arena/heap resets) so callers never observe a raw host TLS
    // pointer outside guest memory. Guest imports bind pg6_gthunk_inet_ntoa (os/host_marshallers.h)
    // for the Win64 by-value struct in_addr ABI.
    thread_local playgta6::os::GuestHeap* t_heap = nullptr;
    thread_local uint64_t t_guest_va = 0;
    thread_local char t_fallback[16];

    playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
    if (t_heap != &heap || t_guest_va == 0 || !heap.owns(t_guest_va)) {
        t_heap = &heap;
        t_guest_va = heap.alloc(16, 16, playgta6::os::HeapFamily::Crt,
                                playgta6::os::kHeapZeroMemory);
    }
    char* out = (t_guest_va != 0) ? reinterpret_cast<char*>(heap.to_host(t_guest_va)) : t_fallback;
    format_ipv4(in.s_addr, out, 16);
    return out;
}

int32_t pg6_ws2_listen(Pg6Socket, int32_t) {
    return fail_socket_error(ws::kWsaENotSock);
}

int32_t pg6_ws2_recv(Pg6Socket, char*, int32_t, int32_t) {
    return fail_socket_error(ws::kWsaENetUnreach);
}

int32_t pg6_ws2_recvfrom(Pg6Socket, char*, int32_t, int32_t, void*, int32_t*) {
    return fail_socket_error(ws::kWsaENetUnreach);
}

int32_t pg6_ws2_select(int32_t, Pg6FdSet* read_fds, Pg6FdSet* write_fds, Pg6FdSet* except_fds,
                       const void*) {
    const uint32_t total = fd_count(read_fds) + fd_count(write_fds) + fd_count(except_fds);
    return fail_socket_error(total == 0 ? ws::kWsaEInval : ws::kWsaENotSock);
}

int32_t pg6_ws2_send(Pg6Socket, const char*, int32_t, int32_t) {
    return fail_socket_error(ws::kWsaENetUnreach);
}

int32_t pg6_ws2_sendto(Pg6Socket, const char*, int32_t, int32_t, const void*, int32_t) {
    return fail_socket_error(ws::kWsaENetUnreach);
}

int32_t pg6_ws2_setsockopt(Pg6Socket, int32_t, int32_t, const char*, int32_t) {
    return fail_socket_error(ws::kWsaENotSock);
}

int32_t pg6_ws2_shutdown(Pg6Socket, int32_t) {
    return fail_socket_error(ws::kWsaENotSock);
}

Pg6Socket pg6_ws2_socket(int32_t, int32_t, int32_t) {
    return fail_invalid_socket(ws::kWsaENetUnreach);
}

Pg6HostEnt* pg6_ws2_gethostbyname(const char*) {
    set_wsa_error(ws::kWsaENetUnreach);
    return nullptr;
}

int32_t pg6_ws2_gethostname(char* name, int32_t name_length) {
    constexpr size_t kLength = sizeof(ws::kHostName);
    if (name == nullptr || name_length < 0 || static_cast<size_t>(name_length) < kLength) {
        return fail_socket_error(ws::kWsaEFault);
    }
    std::memcpy(name, ws::kHostName, kLength);
    return 0;
}

int32_t pg6_ws2_WSAGetLastError(void) {
    return t_wsa_last_error;
}

void pg6_ws2_WSASetLastError(int32_t error) {
    set_wsa_error(error);
}

int32_t pg6_ws2_WSAStartup(uint16_t version_requested, Pg6WsaData* wsa_data) {
    if (wsa_data == nullptr) {
        set_wsa_error(ws::kWsaEFault);
        return ws::kWsaEFault;
    }
    *wsa_data = Pg6WsaData{};
    wsa_data->high_version = ws::kWinsockVersion22;
    if ((version_requested & 0xFFu) == 0) {
        wsa_data->version = ws::kWinsockVersion22;
        set_wsa_error(ws::kWsaEInval);
        return ws::kWsaEInval;
    }
    const bool newer = (version_requested & 0xFFu) > 2u ||
                       ((version_requested & 0xFFu) == 2u && (version_requested >> 8) > 2u);
    wsa_data->version = newer ? ws::kWinsockVersion22 : version_requested;
    wsa_data->max_sockets = 0;
    wsa_data->max_udp_dg = 0;
    std::snprintf(wsa_data->description, sizeof(wsa_data->description), "%s",
                  "PlayGTA6Web offline WinSock 2.2");
    std::snprintf(wsa_data->system_status, sizeof(wsa_data->system_status), "%s",
                  "Offline (network unreachable)");
    set_wsa_error(0);
    return 0;
}

int32_t pg6_ws2___WSAFDIsSet(Pg6Socket fd, Pg6FdSet* set) {
    const uint32_t count = fd_count(set);
    for (uint32_t i = 0; i < count; ++i) {
        if (set->fd_array[i] == fd) {
            return 1;
        }
    }
    return 0;
}

int32_t pg6_ws2_getaddrinfo(const char*, const char*, const void*, Pg6AddrInfo** result) {
    if (result != nullptr) {
        *result = nullptr;
    }
    set_wsa_error(ws::kWsaENetUnreach);
    return ws::kWsaENetUnreach;
}

void pg6_ws2_freeaddrinfo(Pg6AddrInfo*) {}

uint32_t pg6_winsock_ordinals_register_all(void) {
    using win32::export_entry;
    static std::once_flag once;
    static uint32_t registered = 0;

    std::call_once(once, [] {
        const Pg6ExportEntry entries[] = {
            export_entry("accept", &pg6_ws2_accept, 1u),
            export_entry("bind", &pg6_ws2_bind, 2u),
            export_entry("closesocket", &pg6_ws2_closesocket, 3u),
            export_entry("connect", &pg6_ws2_connect, 4u),
            export_entry("getsockname", &pg6_ws2_getsockname, 6u),
            export_entry("getsockopt", &pg6_ws2_getsockopt, 7u),
            export_entry("htons", &pg6_ws2_htons, 9u),
            export_entry("inet_addr", &pg6_ws2_inet_addr, 10u),
            export_entry("inet_ntoa", &pg6_ws2_inet_ntoa, 11u),
            export_entry("listen", &pg6_ws2_listen, 13u),
            export_entry("recv", &pg6_ws2_recv, 16u),
            export_entry("recvfrom", &pg6_ws2_recvfrom, 17u),
            export_entry("select", &pg6_ws2_select, 18u),
            export_entry("send", &pg6_ws2_send, 19u),
            export_entry("sendto", &pg6_ws2_sendto, 20u),
            export_entry("setsockopt", &pg6_ws2_setsockopt, 21u),
            export_entry("shutdown", &pg6_ws2_shutdown, 22u),
            export_entry("socket", &pg6_ws2_socket, 23u),
            export_entry("gethostbyname", &pg6_ws2_gethostbyname, 52u),
            export_entry("gethostname", &pg6_ws2_gethostname, 57u),
            export_entry("WSAGetLastError", &pg6_ws2_WSAGetLastError, 111u),
            export_entry("WSAStartup", &pg6_ws2_WSAStartup, 115u),
            export_entry("__WSAFDIsSet", &pg6_ws2___WSAFDIsSet, 151u),
            export_entry("getaddrinfo", &pg6_ws2_getaddrinfo),
            export_entry("freeaddrinfo", &pg6_ws2_freeaddrinfo),
        };
        static_assert(sizeof(entries) / sizeof(entries[0]) == ws::kRegisteredExportCount);
        registered = static_cast<uint32_t>(
            win32::register_exports("WS2_32.dll", entries, sizeof(entries) / sizeof(entries[0])));
    });
    return registered;
}

}  // extern "C"
