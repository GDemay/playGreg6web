#include "os/winsock_ordinals.h"

#include <cstring>

#include "os/win32_kernel.h"
#include "os/win32_loader.h"

namespace ws = playgta6::os::winsock;
namespace win32 = playgta6::os::win32;

namespace {

constexpr int32_t kSocketError = -1;
constexpr Pg6Socket kInvalidSocket = ~uint64_t{0};

void set_wsa_error(int32_t error) {
    win32::SetLastError(static_cast<uint32_t>(error));
}

// The xbox_offline stub already ran and set WSANOTINITIALISED (before WSAStartup) or its generic
// offline error; refine the latter to the error the specific network operation reports offline.
template <typename T>
T refine_offline_error(T result, int32_t offline_error) {
    if (static_cast<int32_t>(win32::GetLastError()) != ws::kWsaNotInitialised) {
        set_wsa_error(offline_error);
    }
    return result;
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

// BSD inet_addr / inet_aton grammar, as accepted by ws2_32: "a.b.c.d", "a.b.c" (c = 16 bits),
// "a.b" (b = 24 bits) or "a" (32 bits); each part decimal, octal (leading 0) or hex (0x).
// Parsing stops at the terminator or the first whitespace. Returns the address in host order.
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
                return false;  // '8'/'9' in octal, or a hex letter in a decimal/octal part.
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

}  // namespace

extern "C" {

Pg6Socket pg6_ws2_ord_accept(Pg6Socket s, void* address, int32_t* address_length) {
    return pg6_win32_accept(s, address, address_length);
}

int32_t pg6_ws2_ord_bind(Pg6Socket s, const void* name, int32_t name_length) {
    return pg6_win32_bind(s, name, name_length);
}

int32_t pg6_ws2_ord_closesocket(Pg6Socket s) {
    return pg6_win32_closesocket(s);
}

int32_t pg6_ws2_ord_connect(Pg6Socket s, const void* name, int32_t name_length) {
    return refine_offline_error(pg6_win32_connect(s, name, name_length), ws::kWsaEHostUnreach);
}

int32_t pg6_ws2_ord_getsockname(Pg6Socket, void*, int32_t*) {
    // No socket can exist offline (the WSAStartup state is private to xbox_offline).
    set_wsa_error(ws::kWsaENotSock);
    return kSocketError;
}

int32_t pg6_ws2_ord_getsockopt(Pg6Socket s,
                               int32_t level,
                               int32_t option_name,
                               char* option_value,
                               int32_t* option_length) {
    return pg6_win32_getsockopt(s, level, option_name, option_value, option_length);
}

uint16_t pg6_ws2_ord_htons(uint16_t host_short) {
    // The guest is little-endian x86-64, so network order is always the byte-swapped value.
    return static_cast<uint16_t>((host_short << 8) | (host_short >> 8));
}

int32_t pg6_ws2_ord_ioctlsocket(Pg6Socket s, int32_t command, uint32_t* argument) {
    return pg6_win32_ioctlsocket(s, command, argument);
}

uint32_t pg6_ws2_ord_inet_addr(const char* cp) {
    uint32_t host_order = 0;
    if (!parse_ipv4(cp, host_order)) {
        return ws::kInetAddrNone;
    }
    // Network byte order as the little-endian guest reads it from the returned u_long.
    return byte_swap32(host_order);
}

int32_t pg6_ws2_ord_listen(Pg6Socket s, int32_t backlog) {
    return pg6_win32_listen(s, backlog);
}

int32_t pg6_ws2_ord_recv(Pg6Socket s, char* buffer, int32_t length, int32_t flags) {
    return refine_offline_error(pg6_win32_recv(s, buffer, length, flags), ws::kWsaENetDown);
}

int32_t pg6_ws2_ord_recvfrom(Pg6Socket s,
                             char* buffer,
                             int32_t length,
                             int32_t flags,
                             void* from,
                             int32_t* from_length) {
    return refine_offline_error(pg6_win32_recvfrom(s, buffer, length, flags, from, from_length),
                                ws::kWsaENetDown);
}

int32_t pg6_ws2_ord_select(int32_t nfds,
                           Pg6FdSet* read_fds,
                           Pg6FdSet* write_fds,
                           Pg6FdSet* except_fds,
                           const void* timeout) {
    // Never waits, whatever the timeout: no descriptor can become ready offline. Like ws2_32,
    // a select with no sockets at all is WSAEINVAL; any listed SOCKET is not a socket.
    const int32_t result = pg6_win32_select(nfds, read_fds, write_fds, except_fds, timeout);
    const uint32_t total = fd_count(read_fds) + fd_count(write_fds) + fd_count(except_fds);
    return refine_offline_error(result, total == 0 ? ws::kWsaEInval : ws::kWsaENotSock);
}

int32_t pg6_ws2_ord_send(Pg6Socket s, const char* buffer, int32_t length, int32_t flags) {
    return refine_offline_error(pg6_win32_send(s, buffer, length, flags), ws::kWsaENetDown);
}

int32_t pg6_ws2_ord_sendto(Pg6Socket s,
                           const char* buffer,
                           int32_t length,
                           int32_t flags,
                           const void* to,
                           int32_t to_length) {
    return refine_offline_error(pg6_win32_sendto(s, buffer, length, flags, to, to_length),
                                ws::kWsaENetDown);
}

int32_t pg6_ws2_ord_setsockopt(Pg6Socket s,
                               int32_t level,
                               int32_t option_name,
                               const char* option_value,
                               int32_t option_length) {
    return pg6_win32_setsockopt(s, level, option_name, option_value, option_length);
}

int32_t pg6_ws2_ord_shutdown(Pg6Socket s, int32_t how) {
    return pg6_win32_shutdown(s, how);
}

Pg6Socket pg6_ws2_ord_socket(int32_t af, int32_t type, int32_t protocol) {
    // xbox_offline reports WSAENETDOWN (or WSANOTINITIALISED) and never opens a socket.
    pg6_win32_socket(af, type, protocol);
    return kInvalidSocket;
}

Pg6HostEnt* pg6_ws2_ord_gethostbyname(const char*) {
    set_wsa_error(ws::kWsaHostNotFound);
    return nullptr;
}

int32_t pg6_ws2_ord_gethostname(char* name, int32_t name_length) {
    constexpr size_t kLength = sizeof(ws::kHostName);  // Including the terminator.
    if (name == nullptr || name_length < 0 || static_cast<size_t>(name_length) < kLength) {
        set_wsa_error(ws::kWsaEFault);
        return kSocketError;
    }
    std::memcpy(name, ws::kHostName, kLength);
    return 0;
}

int32_t pg6_ws2_ord_wsa_get_last_error(void) {
    return pg6_win32_wsa_get_last_error();
}

int32_t pg6_ws2_ord_wsa_startup(uint16_t version_requested, Pg6WsaData* wsa_data) {
    return pg6_win32_wsa_startup(version_requested, wsa_data);
}

int32_t pg6_ws2_ord_wsa_fd_is_set(Pg6Socket fd, Pg6FdSet* set) {
    const uint32_t count = fd_count(set);
    for (uint32_t i = 0; i < count; ++i) {
        if (set->fd_array[i] == fd) {
            return 1;
        }
    }
    return 0;
}

int32_t pg6_ws2_getaddrinfo(const char* node_name,
                            const char* service_name,
                            const void* hints,
                            Pg6AddrInfo** result) {
    return pg6_win32_getaddrinfo(node_name, service_name, hints, result);
}

void pg6_ws2_freeaddrinfo(Pg6AddrInfo* info) {
    pg6_win32_freeaddrinfo(info);
}

uint32_t pg6_winsock_ordinals_register_all(void) {
    using win32::export_entry;
    const Pg6ExportEntry entries[] = {
        export_entry("accept", &pg6_ws2_ord_accept, ws::kOrdAccept),
        export_entry("bind", &pg6_ws2_ord_bind, ws::kOrdBind),
        export_entry("closesocket", &pg6_ws2_ord_closesocket, ws::kOrdCloseSocket),
        export_entry("connect", &pg6_ws2_ord_connect, ws::kOrdConnect),
        export_entry("getsockname", &pg6_ws2_ord_getsockname, ws::kOrdGetSockName),
        export_entry("getsockopt", &pg6_ws2_ord_getsockopt, ws::kOrdGetSockOpt),
        export_entry("htons", &pg6_ws2_ord_htons, ws::kOrdHtons),
        export_entry("ioctlsocket", &pg6_ws2_ord_ioctlsocket, ws::kOrdIoctlSocket),
        export_entry("inet_addr", &pg6_ws2_ord_inet_addr, ws::kOrdInetAddr),
        export_entry("listen", &pg6_ws2_ord_listen, ws::kOrdListen),
        export_entry("recv", &pg6_ws2_ord_recv, ws::kOrdRecv),
        export_entry("recvfrom", &pg6_ws2_ord_recvfrom, ws::kOrdRecvFrom),
        export_entry("select", &pg6_ws2_ord_select, ws::kOrdSelect),
        export_entry("send", &pg6_ws2_ord_send, ws::kOrdSend),
        export_entry("sendto", &pg6_ws2_ord_sendto, ws::kOrdSendTo),
        export_entry("setsockopt", &pg6_ws2_ord_setsockopt, ws::kOrdSetSockOpt),
        export_entry("shutdown", &pg6_ws2_ord_shutdown, ws::kOrdShutdown),
        export_entry("socket", &pg6_ws2_ord_socket, ws::kOrdSocket),
        export_entry("gethostbyname", &pg6_ws2_ord_gethostbyname, ws::kOrdGetHostByName),
        export_entry("gethostname", &pg6_ws2_ord_gethostname, ws::kOrdGetHostName),
        export_entry("WSAGetLastError", &pg6_ws2_ord_wsa_get_last_error, ws::kOrdWsaGetLastError),
        export_entry("WSAStartup", &pg6_ws2_ord_wsa_startup, ws::kOrdWsaStartup),
        export_entry("__WSAFDIsSet", &pg6_ws2_ord_wsa_fd_is_set, ws::kOrdWsaFdIsSet),
        export_entry("getaddrinfo", &pg6_ws2_getaddrinfo),
        export_entry("freeaddrinfo", &pg6_ws2_freeaddrinfo),
    };
    static_assert(sizeof(entries) / sizeof(entries[0]) == ws::kRegisteredExportCount);
    return static_cast<uint32_t>(
        win32::register_exports(ws::kModuleName, entries, sizeof(entries) / sizeof(entries[0])));
}

}  // extern "C"
