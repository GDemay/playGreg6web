#include "os/game_input.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>

namespace playgta6::os {

static_assert(sizeof(playgta6::InputState) == 40, "InputState must remain 40 bytes");
static_assert(alignof(playgta6::InputState) >= std::atomic_ref<uint32_t>::required_alignment,
              "InputState alignment must satisfy atomic_ref<uint32_t>");
static_assert(alignof(playgta6::InputState) >= std::atomic_ref<int32_t>::required_alignment,
              "InputState alignment must satisfy atomic_ref<int32_t>");
static_assert(alignof(playgta6::InputState) >= std::atomic_ref<float>::required_alignment,
              "InputState alignment must satisfy atomic_ref<float>");

namespace {

inline float normalize_zero(float v) {
    return v == 0.0f ? 0.0f : v;
}

inline uint64_t steady_now_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

void sanitize_deadzone_bounds(float inner_raw,
                              float outer_raw,
                              float default_inner,
                              float default_outer,
                              float& inner_out,
                              float& outer_out) {
    const float safe_inner = std::isfinite(inner_raw) ? inner_raw : default_inner;
    const float safe_outer = std::isfinite(outer_raw) ? outer_raw : default_outer;
    inner_out = std::clamp(safe_inner, 0.0f, 0.99f);
    const float min_outer = std::min(1.0f, inner_out + 1e-4f);
    outer_out = std::clamp(safe_outer, min_outer, 1.0f);
}

// Clamps a finite 2D vector onto the unit disk (|v| <= 1) while preserving angle.
Vec2 clamp_unit_disk(float x, float y) {
    if (!std::isfinite(x) || !std::isfinite(y)) {
        return Vec2{0.0f, 0.0f};
    }
    const float mag = std::hypot(x, y);
    if (!std::isfinite(mag)) {
        const float max_abs = std::max(std::fabs(x), std::fabs(y));
        if (max_abs <= 0.0f) {
            return Vec2{0.0f, 0.0f};
        }
        const float sx = x / max_abs;
        const float sy = y / max_abs;
        const float smag = std::hypot(sx, sy);
        if (smag <= 0.0f || !std::isfinite(smag)) {
            return Vec2{0.0f, 0.0f};
        }
        return Vec2{normalize_zero(std::clamp(sx / smag, -1.0f, 1.0f)),
                    normalize_zero(std::clamp(sy / smag, -1.0f, 1.0f))};
    }
    if (mag <= 0.0f) {
        return Vec2{0.0f, 0.0f};
    }
    if (mag > 1.0f) {
        x /= mag;
        y /= mag;
    }
    return Vec2{normalize_zero(std::clamp(x, -1.0f, 1.0f)),
                normalize_zero(std::clamp(y, -1.0f, 1.0f))};
}

struct SabKeyMapEntry {
    uint32_t bit;
    uint32_t scan_code;
    uint32_t code_point;
    uint8_t virtual_key;
};

constexpr SabKeyMapEntry kSabKeyMap[] = {
    {input_button::kMoveForward, 0x11, 'w', 0x57},   // W
    {input_button::kMoveLeft, 0x1E, 'a', 0x41},      // A
    {input_button::kMoveBackward, 0x1F, 's', 0x53},  // S
    {input_button::kMoveRight, 0x20, 'd', 0x44},     // D
    {input_button::kJump, 0x39, ' ', 0x20},          // Space
    {input_button::kSprint, 0x2A, 0, 0xA0},          // Left Shift
};

}  // namespace

Vec2 normalize_stick(float x, float y, float inner_deadzone, float outer_deadzone) {
    if (!std::isfinite(x) || !std::isfinite(y)) {
        return Vec2{0.0f, 0.0f};
    }

    float inner = 0.0f;
    float outer = 0.0f;
    sanitize_deadzone_bounds(inner_deadzone, outer_deadzone, 0.15f, 0.98f, inner, outer);

    const float mag = std::hypot(x, y);
    if (!std::isfinite(mag)) {
        // Huge finite inputs whose Euclidean norm overflows float: clamp direction onto unit disk.
        return clamp_unit_disk(x, y);
    }
    if (mag <= inner || mag <= 0.0f) {
        return Vec2{0.0f, 0.0f};
    }

    const float clamped_mag = std::min(mag, outer);
    const float normalized_mag = std::clamp((clamped_mag - inner) / (outer - inner), 0.0f, 1.0f);
    const float scale = normalized_mag / mag;
    return clamp_unit_disk(x * scale, y * scale);
}

float normalize_trigger(float value, float inner_deadzone, float outer_deadzone) {
    if (!std::isfinite(value)) {
        return 0.0f;
    }

    float inner = 0.0f;
    float outer = 0.0f;
    sanitize_deadzone_bounds(inner_deadzone, outer_deadzone, 0.05f, 0.98f, inner, outer);

    if (value <= inner) {
        return 0.0f;
    }
    if (value >= outer) {
        return 1.0f;
    }
    return normalize_zero(std::clamp((value - inner) / (outer - inner), 0.0f, 1.0f));
}

Vec2 apply_stick_response_curve(Vec2 stick,
                                StickResponseCurve curve,
                                float exponent,
                                float sensitivity) {
    if (!std::isfinite(stick.x) || !std::isfinite(stick.y)) {
        return Vec2{0.0f, 0.0f};
    }
    const Vec2 clamped = clamp_unit_disk(stick.x, stick.y);
    const float mag = std::hypot(clamped.x, clamped.y);
    if (mag <= 0.0f || !std::isfinite(mag)) {
        return Vec2{0.0f, 0.0f};
    }
    const float r = std::min(mag, 1.0f);
    float curved = r;
    switch (curve) {
        case StickResponseCurve::Linear:
            curved = r;
            break;
        case StickResponseCurve::Cubic:
            curved = r * r * r;
            break;
        case StickResponseCurve::Expo: {
            const float exp = std::isfinite(exponent) ? std::clamp(exponent, 0.1f, 10.0f) : 2.0f;
            curved = std::pow(r, exp);
            break;
        }
    }
    const float sens = (std::isfinite(sensitivity) && sensitivity >= 0.0f) ? sensitivity : 1.0f;
    const float final_mag = std::clamp(curved * sens, 0.0f, 1.0f);
    const float scale = final_mag / mag;
    return clamp_unit_disk(clamped.x * scale, clamped.y * scale);
}

Vec2 mouse_delta_to_look(MouseDelta delta, const MouseLookConfig& config) {
    int32_t raw_dx = delta.dx;
    int32_t raw_dy = delta.dy;
    if (config.max_delta_pixels > 0) {
        const int32_t cap = config.max_delta_pixels;
        raw_dx = std::clamp(raw_dx, -cap, cap);
        raw_dy = std::clamp(raw_dy, -cap, cap);
    }
    if (raw_dx == 0 && raw_dy == 0) {
        return Vec2{0.0f, 0.0f};
    }

    const float dx = static_cast<float>(raw_dx);
    const float dy = static_cast<float>(raw_dy);
    if (std::isfinite(config.deadzone_pixels) && config.deadzone_pixels > 0.0f) {
        if (dx * dx + dy * dy <= config.deadzone_pixels * config.deadzone_pixels) {
            return Vec2{0.0f, 0.0f};
        }
    }

    const float sens_x =
        (std::isfinite(config.sensitivity_x) && config.sensitivity_x > 0.0f) ? config.sensitivity_x : 1.0f;
    const float sens_y =
        (std::isfinite(config.sensitivity_y) && config.sensitivity_y > 0.0f) ? config.sensitivity_y : 1.0f;
    const float scale_px =
        (std::isfinite(config.pixels_for_full_deflection) && config.pixels_for_full_deflection > 0.0f)
            ? config.pixels_for_full_deflection
            : 40.0f;

    const float nx = (dx * sens_x) / scale_px;
    // DOM Pointer Lock reports movementY < 0 when moving mouse up; map to Xbox Y-up (+y = look up)
    // unless invert_y is enabled.
    const float ny = ((config.invert_y ? dy : -dy) * sens_y) / scale_px;
    const Vec2 disk = clamp_unit_disk(nx, ny);
    if (config.curve == StickResponseCurve::Linear) {
        return disk;
    }
    return apply_stick_response_curve(disk, config.curve, config.expo_exponent, 1.0f);
}

size_t decode_keyboard_scan_entries(uint32_t buttons, KeyboardScanEntry* out, size_t capacity) {
    if (out == nullptr || capacity == 0) {
        return 0;
    }
    size_t written = 0;
    for (const SabKeyMapEntry& entry : kSabKeyMap) {
        if ((buttons & entry.bit) == 0u) {
            continue;
        }
        if (written >= capacity) {
            break;
        }
        out[written++] = KeyboardScanEntry{entry.scan_code, entry.code_point, entry.virtual_key, 0u};
    }
    return written;
}

GameInputReader::GameInputReader(playgta6::InputState* shared_state,
                                 SeqlockMode mode,
                                 DeadzoneConfig deadzone)
    : mode_(mode), deadzone_(deadzone) {
    attach(shared_state, mode);
}

void GameInputReader::attach(playgta6::InputState* shared_state, SeqlockMode mode) {
    mode_ = mode;
    if (shared_state != nullptr &&
        (reinterpret_cast<uintptr_t>(shared_state) % alignof(playgta6::InputState)) == 0) {
        state_ = shared_state;
    } else {
        state_ = nullptr;
    }
    prev_buttons_ = 0;
    last_timestamp_us_ = 0;
    mouse_state_ = MouseStateSnapshot{};
}

bool GameInputReader::read_snapshot(playgta6::InputState& out,
                                    uint32_t max_retries,
                                    uint32_t* retries_used) const {
    if (state_ == nullptr) {
        if (retries_used != nullptr) {
            *retries_used = 0;
        }
        return false;
    }

    const uint32_t attempts = max_retries == 0 ? 1u : max_retries;
    std::atomic_ref<uint32_t> seq_ref(state_->seq);
    std::atomic_ref<uint32_t> buttons_ref(state_->buttons);
    std::atomic_ref<int32_t> mouse_dx_ref(state_->mouse_dx);
    std::atomic_ref<int32_t> mouse_dy_ref(state_->mouse_dy);
    std::atomic_ref<float> lx_ref(state_->left_stick_x);
    std::atomic_ref<float> ly_ref(state_->left_stick_y);
    std::atomic_ref<float> rx_ref(state_->right_stick_x);
    std::atomic_ref<float> ry_ref(state_->right_stick_y);
    std::atomic_ref<float> lt_ref(state_->left_trigger);
    std::atomic_ref<float> rt_ref(state_->right_trigger);

    for (uint32_t attempt = 0; attempt < attempts; ++attempt) {
        const uint32_t seq_before = seq_ref.load(std::memory_order_acquire);
        if (mode_ == SeqlockMode::EvenOddSeqlock && (seq_before & 1u) != 0u) {
            continue;
        }

        playgta6::InputState snap{};
        snap.seq = seq_before;
        snap.buttons = buttons_ref.load(std::memory_order_relaxed);
        snap.mouse_dx = mouse_dx_ref.load(std::memory_order_relaxed);
        snap.mouse_dy = mouse_dy_ref.load(std::memory_order_relaxed);
        snap.left_stick_x = lx_ref.load(std::memory_order_relaxed);
        snap.left_stick_y = ly_ref.load(std::memory_order_relaxed);
        snap.right_stick_x = rx_ref.load(std::memory_order_relaxed);
        snap.right_stick_y = ry_ref.load(std::memory_order_relaxed);
        snap.left_trigger = lt_ref.load(std::memory_order_relaxed);
        snap.right_trigger = rt_ref.load(std::memory_order_relaxed);

        std::atomic_thread_fence(std::memory_order_acquire);
        const uint32_t seq_after = seq_ref.load(std::memory_order_relaxed);

        if (seq_before == seq_after) {
            out = snap;
            if (retries_used != nullptr) {
                *retries_used = attempt;
            }
            return true;
        }
    }

    if (retries_used != nullptr) {
        *retries_used = attempts;
    }
    return false;
}

MouseDelta GameInputReader::consume_mouse_delta() {
    if (state_ == nullptr) {
        return MouseDelta{0, 0};
    }
    std::atomic_ref<int32_t> dx_ref(state_->mouse_dx);
    std::atomic_ref<int32_t> dy_ref(state_->mouse_dy);
    MouseDelta delta;
    delta.dx = dx_ref.exchange(0, std::memory_order_acq_rel);
    delta.dy = dy_ref.exchange(0, std::memory_order_acq_rel);
    mouse_state_.position_x += delta.dx;
    mouse_state_.position_y += delta.dy;
    mouse_state_.delta_x = delta.dx;
    mouse_state_.delta_y = delta.dy;
    return delta;
}

uint32_t GameInputReader::buttons() const {
    if (state_ == nullptr) {
        return 0;
    }
    return std::atomic_ref<uint32_t>(state_->buttons).load(std::memory_order_acquire);
}

bool GameInputReader::poll(GameInputFrame& out, bool consume_mouse, uint32_t max_retries) {
    playgta6::InputState snap{};
    if (!read_snapshot(snap, max_retries)) {
        out = GameInputFrame{};
        out.consistent = false;
        return false;
    }

    out.seq = snap.seq;
    out.buttons = snap.buttons;
    out.buttons_pressed = compute_buttons_pressed(prev_buttons_, snap.buttons);
    out.buttons_released = compute_buttons_released(prev_buttons_, snap.buttons);
    prev_buttons_ = snap.buttons;

    if (consume_mouse) {
        out.mouse_delta = consume_mouse_delta();
    } else {
        out.mouse_delta = MouseDelta{snap.mouse_dx, snap.mouse_dy};
        mouse_state_.delta_x = snap.mouse_dx;
        mouse_state_.delta_y = snap.mouse_dy;
    }

    out.left_stick = normalize_stick(
        snap.left_stick_x, snap.left_stick_y, deadzone_.stick_inner, deadzone_.stick_outer);
    out.right_stick = normalize_stick(
        snap.right_stick_x, snap.right_stick_y, deadzone_.stick_inner, deadzone_.stick_outer);
    out.left_trigger = normalize_trigger(
        snap.left_trigger, deadzone_.trigger_inner, deadzone_.trigger_outer);
    out.right_trigger = normalize_trigger(
        snap.right_trigger, deadzone_.trigger_inner, deadzone_.trigger_outer);

    uint64_t ts = steady_now_us();
    if (ts <= last_timestamp_us_) {
        ts = last_timestamp_us_ + 1;
    }
    last_timestamp_us_ = ts;
    out.timestamp_us = ts;
    out.consistent = true;
    return true;
}

void GameInputReader::write_monotonic(playgta6::InputState& dst,
                                      const playgta6::InputState& values,
                                      bool accumulate_mouse) {
    std::atomic_ref<uint32_t>(dst.buttons).store(values.buttons, std::memory_order_relaxed);
    if (accumulate_mouse) {
        std::atomic_ref<int32_t>(dst.mouse_dx).fetch_add(values.mouse_dx, std::memory_order_relaxed);
        std::atomic_ref<int32_t>(dst.mouse_dy).fetch_add(values.mouse_dy, std::memory_order_relaxed);
    } else {
        std::atomic_ref<int32_t>(dst.mouse_dx).store(values.mouse_dx, std::memory_order_relaxed);
        std::atomic_ref<int32_t>(dst.mouse_dy).store(values.mouse_dy, std::memory_order_relaxed);
    }
    std::atomic_ref<float>(dst.left_stick_x).store(values.left_stick_x, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.left_stick_y).store(values.left_stick_y, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_stick_x).store(values.right_stick_x, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_stick_y).store(values.right_stick_y, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.left_trigger).store(values.left_trigger, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_trigger).store(values.right_trigger, std::memory_order_relaxed);

    std::atomic_ref<uint32_t>(dst.seq).fetch_add(1u, std::memory_order_release);
}

void GameInputReader::write_seqlock(playgta6::InputState& dst,
                                    const playgta6::InputState& values,
                                    bool accumulate_mouse) {
    std::atomic_ref<uint32_t> seq_ref(dst.seq);
    const uint32_t cur = seq_ref.load(std::memory_order_relaxed);
    const uint32_t odd_seq = (cur & 1u) != 0u ? cur : (cur + 1u);
    seq_ref.store(odd_seq, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);

    std::atomic_ref<uint32_t>(dst.buttons).store(values.buttons, std::memory_order_relaxed);
    if (accumulate_mouse) {
        std::atomic_ref<int32_t>(dst.mouse_dx).fetch_add(values.mouse_dx, std::memory_order_relaxed);
        std::atomic_ref<int32_t>(dst.mouse_dy).fetch_add(values.mouse_dy, std::memory_order_relaxed);
    } else {
        std::atomic_ref<int32_t>(dst.mouse_dx).store(values.mouse_dx, std::memory_order_relaxed);
        std::atomic_ref<int32_t>(dst.mouse_dy).store(values.mouse_dy, std::memory_order_relaxed);
    }
    std::atomic_ref<float>(dst.left_stick_x).store(values.left_stick_x, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.left_stick_y).store(values.left_stick_y, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_stick_x).store(values.right_stick_x, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_stick_y).store(values.right_stick_y, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.left_trigger).store(values.left_trigger, std::memory_order_relaxed);
    std::atomic_ref<float>(dst.right_trigger).store(values.right_trigger, std::memory_order_relaxed);

    seq_ref.store(odd_seq + 1u, std::memory_order_release);
}

}  // namespace playgta6::os
