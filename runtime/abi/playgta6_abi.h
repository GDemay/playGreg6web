#pragma once

#include <cstdint>

namespace playgta6 {

inline constexpr uint32_t kAbiVersion = 1;

// Shared input snapshot written by web/input/input_bridge.js into a SharedArrayBuffer.
struct InputState {
    uint32_t seq;
    uint32_t buttons;
    int32_t mouse_dx;
    int32_t mouse_dy;
    float left_stick_x;
    float left_stick_y;
    float right_stick_x;
    float right_stick_y;
    float left_trigger;
    float right_trigger;
};
static_assert(sizeof(InputState) == 40, "InputState layout must stay 40 bytes");

// Opcodes encoded into the GPU command buffer (runtime/gpu -> web/workers/wgpu_worker.js).
enum class GpuOpcode : uint16_t {
    Nop = 0,
    BeginFrame = 1,
    ClearColor = 2,
    CreateBuffer = 3,
    WriteBuffer = 4,
    CreateShaderModule = 5,
    Draw = 6,
    Present = 7,
};

struct GpuPacketHeader {
    uint16_t opcode;
    uint16_t flags;
    uint32_t payload_bytes;
};
static_assert(sizeof(GpuPacketHeader) == 8, "GpuPacketHeader layout must stay 8 bytes");

}  // namespace playgta6
