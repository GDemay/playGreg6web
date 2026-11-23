#include "os/os_shim.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>

extern "C" {
EM_IMPORT(env) void host_log(const char* ptr, size_t len);
EM_IMPORT(env) int32_t host_httpfs_open(const char* ptr, size_t len);
EM_IMPORT(env) int64_t host_httpfs_read(int32_t handle, uint64_t offset, void* dst, size_t len);
EM_IMPORT(env) void host_httpfs_close(int32_t handle);
EM_IMPORT(env) void host_wgpu_submit(const uint8_t* ptr, size_t len);
}
#else
#include <cstdio>
#endif

namespace playgta6::os {

void log(std::string_view message) {
#ifdef __EMSCRIPTEN__
    host_log(message.data(), message.size());
#else
    std::fwrite(message.data(), 1, message.size(), stderr);
    std::fputc('\n', stderr);
#endif
}

int32_t httpfs_open(std::string_view rel_path) {
#ifdef __EMSCRIPTEN__
    return host_httpfs_open(rel_path.data(), rel_path.size());
#else
    (void)rel_path;
    return -1;
#endif
}

int64_t httpfs_read(int32_t handle, uint64_t offset, void* dst, size_t len) {
#ifdef __EMSCRIPTEN__
    return host_httpfs_read(handle, offset, dst, len);
#else
    (void)handle;
    (void)offset;
    (void)dst;
    (void)len;
    return -1;
#endif
}

void httpfs_close(int32_t handle) {
#ifdef __EMSCRIPTEN__
    host_httpfs_close(handle);
#else
    (void)handle;
#endif
}

void wgpu_submit(const uint8_t* data, size_t len) {
#ifdef __EMSCRIPTEN__
    host_wgpu_submit(data, len);
#else
    (void)data;
    (void)len;
#endif
}

}  // namespace playgta6::os
