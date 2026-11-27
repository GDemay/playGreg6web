#pragma once

// Host code that reads memory which may be unmapped and catches the fault itself (an SEH __try
// around a profiler stack walk, a stack hint read): vectored handlers run before any SEH frame,
// so a crash-reporting vectored handler would take such a fault for a host crash. It asks
// host_probe_active() first and passes the fault on to the probe's own __except.

namespace playgta6::guest {

inline thread_local int g_host_probe_depth = 0;

class HostProbeScope {
public:
    HostProbeScope() { ++g_host_probe_depth; }
    ~HostProbeScope() { --g_host_probe_depth; }
    HostProbeScope(const HostProbeScope&) = delete;
    HostProbeScope& operator=(const HostProbeScope&) = delete;
};

// The same without an object, for a function holding the __try itself (MSVC rejects __try in a
// function with objects to unwind): enter before the __try, leave after it on every path.
inline void host_probe_enter() {
    ++g_host_probe_depth;
}
inline void host_probe_leave() {
    --g_host_probe_depth;
}

// True on a thread inside a HostProbeScope.
inline bool host_probe_active() {
    return g_host_probe_depth > 0;
}

}  // namespace playgta6::guest
