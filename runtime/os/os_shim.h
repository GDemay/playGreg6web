#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace playgta6::os {

void log(std::string_view message);
int32_t httpfs_open(std::string_view rel_path);
int64_t httpfs_read(int32_t handle, uint64_t offset, void* dst, size_t len);
void httpfs_close(int32_t handle);
void wgpu_submit(const uint8_t* data, size_t len);

}  // namespace playgta6::os
