// Host input mapping for the native boot harness (runtime/guest/host_input.h).
#include <cmath>
#include <cstdio>

#include "guest/host_input.h"
#include "os/game_input.h"

namespace {
int g_failures = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)
bool near(float a, float b) { return std::fabs(a - b) < 1e-4f; }
namespace hi = playgta6::guest::host_input;
namespace button = playgta6::os::input_button;
}  // namespace

int main() {
    hi::XPadSample p;
    p.buttons = hi::kXA | hi::kXStart | hi::kXDpadLeft | hi::kXRightThumb;
    p.left_trigger = 255;
    p.right_trigger = 0;
    p.thumb_lx = 32767;
    p.thumb_ly = -32768;
    p.thumb_rx = 1000;  // small value passes through: the adapter owns the deadzone
    p.thumb_ry = 0;
    const auto s = hi::convert_pad(p);
    CHECK(s.buttons == (button::kGamepadA | button::kMenu | button::kDPadLeft | button::kRightStickPress));
    CHECK(near(s.left_stick_x, 1.0f));
    CHECK(near(s.left_stick_y, -1.0f));
    CHECK(near(s.right_stick_x, 1000.0f / 32767.0f));
    CHECK(near(s.left_trigger, 1.0f) && near(s.right_trigger, 0.0f));

    const auto k = hi::convert_keyboard(hi::kKeyW | hi::kKeyD | hi::kKeySpace | hi::kKeyC | hi::kKeyTab);
    CHECK(near(k.left_stick_x, 1.0f) && near(k.left_stick_y, 1.0f));
    CHECK(k.buttons == (button::kGamepadA | button::kView));
    CHECK(near(k.right_trigger, 1.0f) && near(k.left_trigger, 0.0f));
    const auto c = hi::convert_keyboard(hi::kKeyA | hi::kKeyD);
    CHECK(near(c.left_stick_x, 0.0f));

    const auto m = hi::merge(s, k);
    CHECK(m.buttons == (s.buttons | k.buttons));
    CHECK(near(m.left_stick_x, 1.0f));
    CHECK(near(m.left_stick_y, -1.0f));
    CHECK(near(m.right_trigger, 1.0f) && near(m.left_trigger, 1.0f));

    playgta6::InputState state{};
    playgta6::os::GameInputReader::write_monotonic(state, m);
    CHECK(state.buttons == m.buttons);

    if (g_failures == 0) {
        std::puts("host input: ok");
    }
    return g_failures == 0 ? 0 : 1;
}
