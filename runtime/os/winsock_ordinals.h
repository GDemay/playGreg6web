#pragma once

#include <cstddef>
#include <cstdint>

#include "os/xbox_offline.h"

// WS2_32.dll ordinal-imported exports from game_scarlett_master.exe (docs/imports_required.md):
// 23 ordinals plus getaddrinfo / freeaddrinfo by name. Every ordinal is published in
// Win32ModuleRegistry under both its fixed ordinal and its canonical export name.
//
// Identifiers avoid <winsock2.h> / POSIX spellings (htons, fd_set, INADDR_NONE, WSAE* macros).
// No call performs network I/O or blocks. Pure helpers (byte-order, IPv4 parse/format,
// gethostname, __WSAFDIsSet, WSAStartup, thread-local last-error) succeed locally; socket I/O
// fails with WSAENETUNREACH / WSAENOTSOCK.

struct Pg6HostEnt;  // HOSTENT (never produced offline).

struct Pg6InAddr {
    uint32_t s_addr = 0;  // IPv4 address in network byte order.
};
static_assert(sizeof(Pg6InAddr) == 4, "Pg6InAddr must match struct in_addr");

// Win64 fd_set: u_int fd_count, 4 bytes padding, SOCKET fd_array[FD_SETSIZE].
inline constexpr uint32_t kPg6FdSetSize = 64u;
struct Pg6FdSet {
    uint32_t fd_count = 0;
    uint32_t reserved = 0;
    Pg6Socket fd_array[kPg6FdSetSize]{};
};
static_assert(sizeof(Pg6FdSet) == 520, "Pg6FdSet must match the 64-bit fd_set layout");

namespace playgta6::os::winsock {

inline constexpr char kModuleName[] = "WS2_32.dll";
inline constexpr char kHostName[] = "xbox-offline";

inline constexpr uint32_t kOrdinalExportCount = 23u;
inline constexpr uint32_t kRegisteredExportCount = 25u;

inline constexpr Pg6Socket kInvalidSocket = ~uint64_t{0};
inline constexpr int32_t kSocketError = -1;
inline constexpr uint32_t kInetAddrNone = 0xFFFFFFFFu;
inline constexpr uint16_t kWinsockVersion22 = 0x0202u;

inline constexpr int32_t kWsaEFault = 10014;
inline constexpr int32_t kWsaEInval = 10022;
inline constexpr int32_t kWsaENotSock = 10038;
inline constexpr int32_t kWsaENetUnreach = 10051;
inline constexpr int32_t kWsaNotInitialised = 10093;

// Fixed WS2_32.dll ordinals (Windows Sockets 1.1).
inline constexpr uint32_t kOrdAccept = 1;
inline constexpr uint32_t kOrdBind = 2;
inline constexpr uint32_t kOrdCloseSocket = 3;
inline constexpr uint32_t kOrdConnect = 4;
inline constexpr uint32_t kOrdGetSockName = 6;
inline constexpr uint32_t kOrdGetSockOpt = 7;
inline constexpr uint32_t kOrdHtons = 9;
inline constexpr uint32_t kOrdInetAddr = 10;
inline constexpr uint32_t kOrdInetNtoa = 11;
inline constexpr uint32_t kOrdListen = 13;
inline constexpr uint32_t kOrdRecv = 16;
inline constexpr uint32_t kOrdRecvFrom = 17;
inline constexpr uint32_t kOrdSelect = 18;
inline constexpr uint32_t kOrdSend = 19;
inline constexpr uint32_t kOrdSendTo = 20;
inline constexpr uint32_t kOrdSetSockOpt = 21;
inline constexpr uint32_t kOrdShutdown = 22;
inline constexpr uint32_t kOrdSocket = 23;
inline constexpr uint32_t kOrdGetHostByName = 52;
inline constexpr uint32_t kOrdGetHostName = 57;
inline constexpr uint32_t kOrdWsaGetLastError = 111;
inline constexpr uint32_t kOrdWsaStartup = 115;
inline constexpr uint32_t kOrdWsaFdIsSet = 151;

}  // namespace playgta6::os::winsock

extern "C" {

Pg6Socket pg6_ws2_accept(Pg6Socket s, void* address, int32_t* address_length);
int32_t pg6_ws2_bind(Pg6Socket s, const void* name, int32_t name_length);
int32_t pg6_ws2_closesocket(Pg6Socket s);
int32_t pg6_ws2_connect(Pg6Socket s, const void* name, int32_t name_length);
int32_t pg6_ws2_getsockname(Pg6Socket s, void* name, int32_t* name_length);
int32_t pg6_ws2_getsockopt(Pg6Socket s,
                           int32_t level,
                           int32_t option_name,
                           char* option_value,
                           int32_t* option_length);
uint16_t pg6_ws2_htons(uint16_t host_short);
uint32_t pg6_ws2_inet_addr(const char* cp);
char* pg6_ws2_inet_ntoa(Pg6InAddr in);
int32_t pg6_ws2_listen(Pg6Socket s, int32_t backlog);
int32_t pg6_ws2_recv(Pg6Socket s, char* buffer, int32_t length, int32_t flags);
int32_t pg6_ws2_recvfrom(Pg6Socket s,
                         char* buffer,
                         int32_t length,
                         int32_t flags,
                         void* from,
                         int32_t* from_length);
int32_t pg6_ws2_select(int32_t nfds,
                       Pg6FdSet* read_fds,
                       Pg6FdSet* write_fds,
                       Pg6FdSet* except_fds,
                       const void* timeout);
int32_t pg6_ws2_send(Pg6Socket s, const char* buffer, int32_t length, int32_t flags);
int32_t pg6_ws2_sendto(Pg6Socket s,
                       const char* buffer,
                       int32_t length,
                       int32_t flags,
                       const void* to,
                       int32_t to_length);
int32_t pg6_ws2_setsockopt(Pg6Socket s,
                           int32_t level,
                           int32_t option_name,
                           const char* option_value,
                           int32_t option_length);
int32_t pg6_ws2_shutdown(Pg6Socket s, int32_t how);
Pg6Socket pg6_ws2_socket(int32_t af, int32_t type, int32_t protocol);
Pg6HostEnt* pg6_ws2_gethostbyname(const char* name);
int32_t pg6_ws2_gethostname(char* name, int32_t name_length);
int32_t pg6_ws2_WSAGetLastError(void);
void pg6_ws2_WSASetLastError(int32_t error);
int32_t pg6_ws2_WSAStartup(uint16_t version_requested, Pg6WsaData* wsa_data);
int32_t pg6_ws2___WSAFDIsSet(Pg6Socket fd, Pg6FdSet* set);

int32_t pg6_ws2_getaddrinfo(const char* node_name,
                            const char* service_name,
                            const void* hints,
                            Pg6AddrInfo** result);
void pg6_ws2_freeaddrinfo(Pg6AddrInfo* info);

// Registers all 23 ordinals (name + ordinal) plus getaddrinfo/freeaddrinfo under WS2_32.dll.
// Idempotent; returns 25 on success.
uint32_t pg6_winsock_ordinals_register_all(void);

}  // extern "C"
