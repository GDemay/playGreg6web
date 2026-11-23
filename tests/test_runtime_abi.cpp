#include <cassert>
#include <cstdint>
#include <cstring>

#include "abi/playgta6_abi.h"
#include "gpu/cmd_encoder.h"

int main() {
    playgta6::gpu::CommandEncoder enc;
    enc.begin_frame();
    enc.clear_color(0.1f, 0.2f, 0.3f, 1.0f);
    enc.present();

    // 8 (BeginFrame) + (8 + 16) (ClearColor) + 8 (Present) = 40 bytes
    assert(enc.size() == 40);

    playgta6::GpuPacketHeader h0{};
    std::memcpy(&h0, enc.data(), sizeof(h0));
    assert(h0.opcode == static_cast<uint16_t>(playgta6::GpuOpcode::BeginFrame));
    assert(h0.payload_bytes == 0);

    playgta6::GpuPacketHeader h1{};
    std::memcpy(&h1, enc.data() + 8, sizeof(h1));
    assert(h1.opcode == static_cast<uint16_t>(playgta6::GpuOpcode::ClearColor));
    assert(h1.payload_bytes == 16);

    playgta6::GpuPacketHeader h2{};
    std::memcpy(&h2, enc.data() + 32, sizeof(h2));
    assert(h2.opcode == static_cast<uint16_t>(playgta6::GpuOpcode::Present));
    assert(h2.payload_bytes == 0);

    return 0;
}
