#pragma once

#include <cstddef>
#include <cstdint>

#include "abi/playgta6_abi.h"

namespace playgta6::os {

// Button bit definitions matching web/input/input_bridge.js (bits 0..5, 8..22).
namespace input_button {
inline constexpr uint32_t kMoveForward = 1u << 0;   // KeyW
inline constexpr uint32_t kMoveBackward = 1u << 1;  // KeyS
inline constexpr uint32_t kMoveLeft = 1u << 2;      // KeyA
inline constexpr uint32_t kMoveRight = 1u << 3;     // KeyD
inline constexpr uint32_t kJump = 1u << 4;          // Space / A
inline constexpr uint32_t kSprint = 1u << 5;        // ShiftLeft / L3
inline constexpr uint32_t kGamepadA = 1u << 8;
inline constexpr uint32_t kGamepadB = 1u << 9;
inline constexpr uint32_t kGamepadX = 1u << 10;
inline constexpr uint32_t kGamepadY = 1u << 11;
inline constexpr uint32_t kLeftBumper = 1u << 12;
inline constexpr uint32_t kRightBumper = 1u << 13;
inline constexpr uint32_t kMenu = 1u << 14;
inline constexpr uint32_t kView = 1u << 15;
// Extended gamepad bits 16..22 (web/input/input_bridge.js INPUT_BUTTONS).
inline constexpr uint32_t kLeftStickPress = 1u << 16;   // L3
inline constexpr uint32_t kRightStickPress = 1u << 17;  // R3
inline constexpr uint32_t kDPadUp = 1u << 18;
inline constexpr uint32_t kDPadDown = 1u << 19;
inline constexpr uint32_t kDPadLeft = 1u << 20;
inline constexpr uint32_t kDPadRight = 1u << 21;
inline constexpr uint32_t kGuide = 1u << 22;
}  // namespace input_button

enum class SeqlockMode : uint8_t {
    // Matches docs/wasm_abi.md and web/input/input_bridge.js: seq is incremented after each write.
    // Reader retries whenever seq changes between start and end of snapshot.
    MonotonicSeq = 0,
    // Strict even/odd seqlock: writer bumps seq to odd before writing and to even after writing.
    // Reader retries whenever seq is odd or changes during snapshot.
    EvenOddSeqlock = 1,
};

enum class StickResponseCurve : uint8_t {
    Linear = 0,
    Cubic = 1,
    Expo = 2,
};

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct MouseDelta {
    int32_t dx = 0;
    int32_t dy = 0;
};

struct DeadzoneConfig {
    float stick_inner = 0.15f;
    float stick_outer = 0.98f;
    float trigger_inner = 0.05f;
    float trigger_outer = 0.98f;
};

// Pointer Lock mouse-delta-to-camera-look configuration matching web/input/input_profiles.js (W374).
struct MouseLookConfig {
    float sensitivity_x = 1.0f;
    float sensitivity_y = 1.0f;
    float pixels_for_full_deflection = 40.0f;  // Raw pixels producing magnitude 1.0 at sensitivity 1.0
    float deadzone_pixels = 0.0f;              // Radial micro-jitter deadzone in pixels (W374 default: 0)
    int32_t max_delta_pixels = 500;            // Spike clamp per poll matching W374 DEFAULT_MAX_MOUSE_DELTA
    bool invert_y = false;                     // false: mouse up (dy < 0) -> look up (+y); true: inverted
    StickResponseCurve curve = StickResponseCurve::Linear;
    float expo_exponent = 2.0f;
};

// Cumulative + per-poll raw mouse state snapshot.
struct MouseStateSnapshot {
    uint32_t buttons = 0;  // GameInputMouseButtons bitmask (Left=1, Right=2, Middle=4, B4=8, B5=16)
    int64_t position_x = 0;
    int64_t position_y = 0;
    int64_t wheel_x = 0;
    int64_t wheel_y = 0;
    int32_t delta_x = 0;
    int32_t delta_y = 0;
};

// Decoded keyboard scan-code entry matching GameInputKeyState fields.
struct KeyboardScanEntry {
    uint32_t scan_code = 0;
    uint32_t code_point = 0;
    uint8_t virtual_key = 0;
    uint8_t is_dead_key = 0;
};

// Rumble & haptic telemetry recorded by the virtual GameInput gamepad device.
struct RumbleTelemetry {
    uint64_t set_calls = 0;
    uint64_t nonzero_updates = 0;
    uint64_t stop_calls = 0;
    uint64_t haptic_calls = 0;
    uint64_t last_timestamp_us = 0;
    float last_low_frequency = 0.0f;
    float last_high_frequency = 0.0f;
    float last_left_trigger = 0.0f;
    float last_right_trigger = 0.0f;
    float peak_low_frequency = 0.0f;
    float peak_high_frequency = 0.0f;
    float peak_left_trigger = 0.0f;
    float peak_right_trigger = 0.0f;
};

struct GameInputFrame {
    uint32_t seq = 0;
    uint32_t buttons = 0;
    uint32_t buttons_pressed = 0;   // Rising edges since previous poll
    uint32_t buttons_released = 0;  // Falling edges since previous poll
    MouseDelta mouse_delta{};
    Vec2 left_stick{};
    Vec2 right_stick{};
    float left_trigger = 0.0f;
    float right_trigger = 0.0f;
    bool consistent = false;
    uint64_t timestamp_us = 0;      // Monotonic strictly-increasing poll timestamp (us)
};

// Button bitmask helpers.
inline constexpr bool is_button_down(uint32_t buttons, uint32_t mask) {
    return (buttons & mask) == mask && mask != 0;
}

inline constexpr bool is_any_button_down(uint32_t buttons, uint32_t mask) {
    return (buttons & mask) != 0;
}

inline constexpr uint32_t compute_buttons_pressed(uint32_t prev, uint32_t curr) {
    return (~prev) & curr;
}

inline constexpr uint32_t compute_buttons_released(uint32_t prev, uint32_t curr) {
    return prev & (~curr);
}

// Radial stick deadzone normalization: maps magnitude in [inner, outer] -> [0, 1] while preserving angle.
Vec2 normalize_stick(float x,
                     float y,
                     float inner_deadzone = 0.15f,
                     float outer_deadzone = 0.98f);

// Linear trigger deadzone normalization: maps [inner, outer] -> [0, 1].
float normalize_trigger(float value,
                        float inner_deadzone = 0.05f,
                        float outer_deadzone = 0.98f);

// Applies a radial stick response curve (Linear/Cubic/Expo) and optional sensitivity multiplier
// while preserving angle and clamping onto the unit disk (matching web/input/input_profiles.js).
Vec2 apply_stick_response_curve(Vec2 stick,
                                StickResponseCurve curve = StickResponseCurve::Linear,
                                float exponent = 2.0f,
                                float sensitivity = 1.0f);

// Converts raw Pointer Lock mouse deltas into a normalized Y-up camera look vector in [-1, 1]
// clamped onto the unit disk (matching web/input/input_bridge.js / W374 camera feel).
Vec2 mouse_delta_to_look(MouseDelta delta, const MouseLookConfig& config = {});

// Decodes pressed SAB keyboard bits (0..5: W, A, S, D, Space, Left Shift) in deterministic order
// without heap allocations. Returns the number of entries written (up to `capacity`).
size_t decode_keyboard_scan_entries(uint32_t buttons, KeyboardScanEntry* out, size_t capacity);

// Rumble/haptics telemetry from the guest GameInput.dll virtual gamepad (xbox_scarlett_guest_input.cpp).
RumbleTelemetry game_input_rumble_telemetry();
void reset_game_input_rumble_telemetry();

// Host/test entry point to query GameInputApi::GetTemporalReading against a live guest IGameInput
// object without disturbing slot 7's stub-probe coverage in test_xbox_scarlett_guest.cpp.
bool game_input_guest_temporal_reading(uint64_t game_input_va,
                                       uint64_t timestamp_us,
                                       uint64_t device_va,
                                       uint64_t* out_reading_va,
                                       int32_t* out_hr = nullptr);

// Lock-free snapshot reader over the 40-byte playgta6::InputState SharedArrayBuffer.
class GameInputReader {
public:
    explicit GameInputReader(playgta6::InputState* shared_state = nullptr,
                             SeqlockMode mode = SeqlockMode::MonotonicSeq,
                             DeadzoneConfig deadzone = {});

    void attach(playgta6::InputState* shared_state,
                SeqlockMode mode = SeqlockMode::MonotonicSeq);
    bool is_attached() const { return state_ != nullptr; }

    void set_deadzone(const DeadzoneConfig& config) { deadzone_ = config; }
    const DeadzoneConfig& deadzone() const { return deadzone_; }

    void set_seqlock_mode(SeqlockMode mode) { mode_ = mode; }
    SeqlockMode seqlock_mode() const { return mode_; }

    // Lock-free consistent read of all 40 bytes of playgta6::InputState.
    // Retries up to max_retries times if a concurrent write tears the snapshot.
    // Returns true if a consistent (untorn) snapshot was captured; false on timeout/unattached.
    bool read_snapshot(playgta6::InputState& out,
                       uint32_t max_retries = 64,
                       uint32_t* retries_used = nullptr) const;

    // Atomically reads and zeroes accumulated mouse_dx / mouse_dy from the shared buffer
    // (pairing with Atomics.add in web/input/input_bridge.js).
    MouseDelta consume_mouse_delta();

    // Cumulative mouse state accumulated across consumed deltas since attach().
    const MouseStateSnapshot& mouse_state() const { return mouse_state_; }

    // Monotonic timestamp (us) of the most recent consistent poll().
    uint64_t last_timestamp_us() const { return last_timestamp_us_; }

    // Reads the current button bitmask atomically.
    uint32_t buttons() const;

    // Captures a consistent snapshot, computes pressed/released button transitions since the
    // previous poll, optionally consumes accumulated mouse deltas, and normalizes sticks/triggers.
    bool poll(GameInputFrame& out, bool consume_mouse = true, uint32_t max_retries = 64);

    // Atomic writer helpers matching both MonotonicSeq and EvenOddSeqlock protocols (useful for
    // host integration and multi-threaded unit tests without data races).
    static void write_monotonic(playgta6::InputState& dst,
                                const playgta6::InputState& values,
                                bool accumulate_mouse = false);
    static void write_seqlock(playgta6::InputState& dst,
                              const playgta6::InputState& values,
                              bool accumulate_mouse = false);

private:
    playgta6::InputState* state_ = nullptr;
    SeqlockMode mode_ = SeqlockMode::MonotonicSeq;
    DeadzoneConfig deadzone_{};
    uint32_t prev_buttons_ = 0;
    uint64_t last_timestamp_us_ = 0;
    MouseStateSnapshot mouse_state_{};
};

}  // namespace playgta6::os
