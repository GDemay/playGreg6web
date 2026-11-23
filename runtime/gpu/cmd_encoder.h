#pragma once

#include <cstdint>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::gpu {

class CommandEncoder {
public:
    void clear();
    void begin_frame();
    void clear_color(float r, float g, float b, float a);
    void present();

    const uint8_t* data() const { return buf_.data(); }
    size_t size() const { return buf_.size(); }

private:
    void write_packet(GpuOpcode opcode, const void* payload, uint32_t payload_bytes);
    std::vector<uint8_t> buf_;
};

}  // namespace playgta6::gpu
