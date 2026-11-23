#include "gpu/cmd_encoder.h"

#include <cstring>

namespace playgta6::gpu {

void CommandEncoder::clear() {
    buf_.clear();
}

void CommandEncoder::write_packet(GpuOpcode opcode, const void* payload, uint32_t payload_bytes) {
    GpuPacketHeader header{
        static_cast<uint16_t>(opcode),
        0,
        payload_bytes,
    };
    const size_t offset = buf_.size();
    buf_.resize(offset + sizeof(GpuPacketHeader) + payload_bytes);
    std::memcpy(buf_.data() + offset, &header, sizeof(GpuPacketHeader));
    if (payload_bytes > 0 && payload != nullptr) {
        std::memcpy(buf_.data() + offset + sizeof(GpuPacketHeader), payload, payload_bytes);
    }
}

void CommandEncoder::begin_frame() {
    write_packet(GpuOpcode::BeginFrame, nullptr, 0);
}

void CommandEncoder::clear_color(float r, float g, float b, float a) {
    const float rgba[4] = {r, g, b, a};
    write_packet(GpuOpcode::ClearColor, rgba, sizeof(rgba));
}

void CommandEncoder::present() {
    write_packet(GpuOpcode::Present, nullptr, 0);
}

}  // namespace playgta6::gpu
