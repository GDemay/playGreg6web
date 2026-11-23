#pragma once

#include <cstddef>
#include <cstdint>

#include "os/xbox_offline.h"

// WS2_32.dll ordinal-imported exports (docs/imports_required.md: 23 ordinals plus getaddrinfo /
// freeaddrinfo by name) and their Win32ModuleRegistry (O7) bindings.
//
// The guest imports most of WinSock by ordinal, so these shims publish every symbol both under its
// fixed WS2_32.dll ordinal and under its canonical export name. Socket state, WSAStartup reference
// counting and the WSA last-error slot are shared with the offline stubs in xbox_offline.h (O6):
// WSAStartup/WSACleanup pair across both files, and WSAGetLastError reads the same thread-local
// Win32 GetLastError slot.
//
// Offline model (no call ever performs network I/O or blocks):
//   * pure helpers work for real: htons (#9), inet_addr (#11), __WSAFDIsSet (#151),
//     gethostname (#57, "playgta6web"), WSAStartup (#115), WSAGetLastError (#111);
//   * socket (#23) fails with WSAENETDOWN, connect (#4) with WSAEHOSTUNREACH, data transfer
//     (send/recv/sendto/recvfrom) with WSAENETDOWN, name resolution (gethostbyname,
//     getaddrinfo) with WSAHOST_NOT_FOUND;
//   * every other call on a SOCKET fails with WSAENOTSOCK (no socket can exist offline);
//   * calls that require WSAStartup report WSANOTINITIALISED before it, like ws2_32.
//
// Identifiers deliberately avoid the <winsock2.h> / <sys/socket.h> / <netinet/in.h> spellings
// (htons, fd_set, WSAE*, INADDR_NONE, ... are macros on some platforms).

struct Pg6HostEnt;  // HOSTENT (never produced offline).

// fd_set (64-bit layout: u_int fd_count, 4 bytes padding, SOCKET fd_array[FD_SETSIZE]).
inline constexpr uint32_t kPg6FdSetSize = 64u;
struct Pg6FdSet {
    uint32_t fd_count = 0;
    uint32_t reserved = 0;
    Pg6Socket fd_array[kPg6FdSetSize]{};
};
static_assert(sizeof(Pg6FdSet) == 520, "Pg6FdSet must match the 64-bit fd_set layout");

namespace playgta6::os::winsock {

inline constexpr char kModuleName[] = "WS2_32.dll";
inline constexpr char kHostName[] = "playgta6web";

// Number of exports pg6_winsock_ordinals_register_all() publishes (23 ordinals + 2 names).
inline constexpr uint32_t kOrdinalExportCount = 23u;
inline constexpr uint32_t kRegisteredExportCount = 25u;

// inet_addr failure value (INADDR_NONE).
inline constexpr uint32_t kInetAddrNone = 0xFFFFFFFFu;

// WSA error codes used here (the shared ones live in playgta6::os::offline).
inline constexpr int32_t kWsaEFault = 10014;
inline constexpr int32_t kWsaEInval = 10022;
inline constexpr int32_t kWsaENotSock = 10038;
inline constexpr int32_t kWsaENetDown = 10050;
inline constexpr int32_t kWsaEHostUnreach = 10065;
inline constexpr int32_t kWsaNotInitialised = 10093;
inline constexpr int32_t kWsaHostNotFound = 11001;

// WS2_32.dll ordinals (fixed since Windows Sockets 1.1).
inline constexpr uint32_t kOrdAccept = 1;
inline constexpr uint32_t kOrdBind = 2;
inline constexpr uint32_t kOrdCloseSocket = 3;
inline constexpr uint32_t kOrdConnect = 4;
inline constexpr uint32_t kOrdGetSockName = 6;
inline constexpr uint32_t kOrdGetSockOpt = 7;
inline constexpr uint32_t kOrdHtons = 9;
inline constexpr uint32_t kOrdIoctlSocket = 10;
inline constexpr uint32_t kOrdInetAddr = 11;
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

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints (Win64 WinSock signatures; long/u_long are 32-bit).
// ---------------------------------------------------------------------------

extern "C" {

Pg6Socket pg6_ws2_ord_accept(Pg6Socket s, void* address, int32_t* address_length);         // #1
int32_t pg6_ws2_ord_bind(Pg6Socket s, const void* name, int32_t name_length);              // #2
int32_t pg6_ws2_ord_closesocket(Pg6Socket s);                                              // #3
int32_t pg6_ws2_ord_connect(Pg6Socket s, const void* name, int32_t name_length);           // #4
int32_t pg6_ws2_ord_getsockname(Pg6Socket s, void* name, int32_t* name_length);            // #6
int32_t pg6_ws2_ord_getsockopt(Pg6Socket s,                                                // #7
                               int32_t level,
                               int32_t option_name,
                               char* option_value,
                               int32_t* option_length);
uint16_t pg6_ws2_ord_htons(uint16_t host_short);                                           // #9
int32_t pg6_ws2_ord_ioctlsocket(Pg6Socket s, int32_t command, uint32_t* argument);         // #10
uint32_t pg6_ws2_ord_inet_addr(const char* cp);                                            // #11
int32_t pg6_ws2_ord_listen(Pg6Socket s, int32_t backlog);                                  // #13
int32_t pg6_ws2_ord_recv(Pg6Socket s, char* buffer, int32_t length, int32_t flags);        // #16
int32_t pg6_ws2_ord_recvfrom(Pg6Socket s,                                                  // #17
                             char* buffer,
                             int32_t length,
                             int32_t flags,
                             void* from,
                             int32_t* from_length);
int32_t pg6_ws2_ord_select(int32_t nfds,                                                   // #18
                           Pg6FdSet* read_fds,
                           Pg6FdSet* write_fds,
                           Pg6FdSet* except_fds,
                           const void* timeout);
int32_t pg6_ws2_ord_send(Pg6Socket s, const char* buffer, int32_t length, int32_t flags);  // #19
int32_t pg6_ws2_ord_sendto(Pg6Socket s,                                                    // #20
                           const char* buffer,
                           int32_t length,
                           int32_t flags,
                           const void* to,
                           int32_t to_length);
int32_t pg6_ws2_ord_setsockopt(Pg6Socket s,                                                // #21
                               int32_t level,
                               int32_t option_name,
                               const char* option_value,
                               int32_t option_length);
int32_t pg6_ws2_ord_shutdown(Pg6Socket s, int32_t how);                                    // #22
Pg6Socket pg6_ws2_ord_socket(int32_t af, int32_t type, int32_t protocol);                  // #23
Pg6HostEnt* pg6_ws2_ord_gethostbyname(const char* name);                                   // #52
int32_t pg6_ws2_ord_gethostname(char* name, int32_t name_length);                          // #57
int32_t pg6_ws2_ord_wsa_get_last_error(void);                                              // #111
int32_t pg6_ws2_ord_wsa_startup(uint16_t version_requested, Pg6WsaData* wsa_data);         // #115
int32_t pg6_ws2_ord_wsa_fd_is_set(Pg6Socket fd, Pg6FdSet* set);                            // #151

int32_t pg6_ws2_getaddrinfo(const char* node_name,
                            const char* service_name,
                            const void* hints,
                            Pg6AddrInfo** result);
void pg6_ws2_freeaddrinfo(Pg6AddrInfo* info);

// Publishes all 23 ordinals (by ordinal and by name) plus getaddrinfo/freeaddrinfo (by name)
// under WS2_32.dll. Replaces earlier name-only bindings (shim_exports). Idempotent; returns the
// number of exports registered (25 on success).
uint32_t pg6_winsock_ordinals_register_all(void);

}  // extern "C"
