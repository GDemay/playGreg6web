#include "guest/input_script.h"
#include "os/thread_name.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

#include "os/game_input.h"

namespace playgta6::guest {

namespace {

bool parse_number(const std::string& text, uint64_t& out) {
    if (text.empty()) {
        return false;
    }
    // Reject leading whitespace or signs (+/-) so negative values never wrap to UINT64_MAX.
    if (!std::isdigit(static_cast<unsigned char>(text[0]))) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(text.c_str(), &end, 0);
    if (errno == ERANGE || end == nullptr || *end != '\0') {
        return false;
    }
    out = static_cast<uint64_t>(v);
    return true;
}

bool button_bit(const std::string& name, uint32_t& bit) {
    namespace b = playgta6::os::input_button;
    struct Name {
        const char* text;
        uint32_t bit;
    };
    static constexpr Name kNames[] = {
        {"A", b::kGamepadA},       {"B", b::kGamepadB},     {"X", b::kGamepadX},   {"Y", b::kGamepadY},
        {"LB", b::kLeftBumper},    {"RB", b::kRightBumper}, {"Menu", b::kMenu},    {"View", b::kView},
        // Left stick pushed fully (the bridge's W/A/S/D bits, which the GameInput layer turns
        // into the left thumbstick): menu navigation.
        {"Up", b::kMoveForward},   {"Down", b::kMoveBackward}, {"Left", b::kMoveLeft}, {"Right", b::kMoveRight},
        // D-pad and stick clicks (gamepad buttons only).
        {"DUp", b::kDPadUp},       {"DDown", b::kDPadDown},    {"DLeft", b::kDPadLeft}, {"DRight", b::kDPadRight},
        {"LS", b::kLeftStickPress}, {"RS", b::kRightStickPress},
    };
    for (const Name& n : kNames) {
        if (name == n.text) {
            bit = n.bit;
            return true;
        }
    }
    return false;
}

// Analog names (left/right stick, triggers): full deflection while held. False when `name` is not one.
bool analog_name(const std::string& name, InputScriptEvent& e) {
    if (name == "LUp") e.left_stick_y = 1.0f;
    else if (name == "LDown") e.left_stick_y = -1.0f;
    else if (name == "LLeft") e.left_stick_x = -1.0f;
    else if (name == "LRight") e.left_stick_x = 1.0f;
    else if (name == "RUp") e.right_stick_y = 1.0f;
    else if (name == "RDown") e.right_stick_y = -1.0f;
    else if (name == "RLeft") e.right_stick_x = -1.0f;
    else if (name == "RRight") e.right_stick_x = 1.0f;
    else if (name == "LT") e.left_trigger = 1.0f;
    else if (name == "RT") e.right_trigger = 1.0f;
    else return false;
    return true;
}

bool parse_buttons(const std::string& text, InputScriptEvent& e) {
    uint32_t& out = e.buttons;
    uint64_t number = 0;
    if (parse_number(text, number)) {
        if (number == 0 || number > 0xffffffffull) {
            return false;
        }
        out = static_cast<uint32_t>(number);
        return true;
    }
    out = 0;
    bool any_analog = false;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t plus = text.find('+', start);
        const std::string part = text.substr(start, plus == std::string::npos ? std::string::npos : plus - start);
        uint32_t bit = 0;
        if (analog_name(part, e)) {
            any_analog = true;
        } else if (button_bit(part, bit)) {
            out |= bit;
        } else {
            return false;
        }
        if (plus == std::string::npos) {
            break;
        }
        start = plus + 1;
    }
    return out != 0 || any_analog;
}

// `500ms`, `2s`, `1.5s` or plain milliseconds.
bool parse_duration(const std::string& text, uint64_t& out) {
    if (text.empty()) {
        return false;
    }
    std::string num = text;
    double scale = 1.0;
    if (num.size() > 2 && num.compare(num.size() - 2, 2, "ms") == 0) {
        num.resize(num.size() - 2);
    } else if (num.size() > 1 && num.back() == 's') {
        num.pop_back();
        scale = 1000.0;
    }
    if (num.empty() || num.find_first_not_of("0123456789.") != std::string::npos) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(num.c_str(), &end);
    if (errno == ERANGE || end == nullptr || *end != '\0' || !std::isfinite(v) || v < 0.0) {
        return false;
    }
    const double scaled = v * scale;
    // Cap well below UINT64_MAX (1e18 ms > 31 million years) to avoid float-to-uint64 UB.
    if (!std::isfinite(scaled) || scaled > 1.0e18) {
        return false;
    }
    out = static_cast<uint64_t>(scaled + 0.5);
    return true;
}

bool parse_condition(const std::string& text, InputCondition& c) {
    struct OpName {
        const char* text;
        InputCondition::Op op;
    };
    static const OpName kOps[] = {{"==", InputCondition::Op::Eq}, {"!=", InputCondition::Op::Ne},
                                  {">=", InputCondition::Op::Ge}, {"<=", InputCondition::Op::Le},
                                  {">", InputCondition::Op::Gt},  {"<", InputCondition::Op::Lt}};
    // The first operator character in the token; two-character operators win over one.
    const size_t at = text.find_first_of("=!<>");
    if (at == std::string::npos || at == 0) {
        return false;
    }
    size_t len = 0;
    for (const OpName& o : kOps) {
        const size_t n = std::strlen(o.text);
        if (text.compare(at, n, o.text) == 0 && n > len) {
            len = n;
            c.op = o.op;
        }
    }
    if (len == 0) {
        return false;
    }
    const std::string metric = text.substr(0, at);
    if (metric == "draws_per_frame" || metric == "draws") {
        c.metric = InputCondition::Metric::DrawsPerFrame;
    } else if (metric == "presents") {
        c.metric = InputCondition::Metric::Presents;
    } else {
        return false;
    }
    c.values.clear();
    const std::string rest = text.substr(at + len);
    size_t start = 0;
    while (true) {
        const size_t bar = rest.find('|', start);
        const std::string part = rest.substr(start, bar == std::string::npos ? std::string::npos : bar - start);
        uint64_t v = 0;
        if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos || !parse_number(part, v)) {
            return false;
        }
        c.values.push_back(v);
        if (bar == std::string::npos) {
            break;
        }
        start = bar + 1;
    }
    return c.values.size() == 1 || c.op == InputCondition::Op::Eq || c.op == InputCondition::Op::Ne;
}

// `<cond> [for <dur>] [timeout <dur>]` from `fields`; false with `error` set.
bool parse_wait_tail(std::istringstream& fields, InputScriptEvent& e, std::string& error) {
    std::string cond;
    if (!(fields >> cond) || !parse_condition(cond, e.cond)) {
        error = "bad condition '" + cond + "'";
        return false;
    }
    std::string key;
    while (fields >> key) {
        std::string value;
        uint64_t* slot = key == "for" ? &e.for_ms : key == "timeout" ? &e.timeout_ms : nullptr;
        if (slot == nullptr) {
            error = "unexpected '" + key + "'";
            return false;
        }
        if (!(fields >> value) || !parse_duration(value, *slot)) {
            error = "bad duration '" + value + "' after " + key;
            return false;
        }
    }
    return true;
}

void sort_presses(std::vector<InputScriptEvent>& out, size_t from) {
    std::stable_sort(out.begin() + static_cast<std::ptrdiff_t>(from), out.end(),
                     [](const InputScriptEvent& a, const InputScriptEvent& b) { return a.time_ms < b.time_ms; });
}

std::string hex(uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof(b), "0x%x", v);
    return b;
}

}  // namespace

bool condition_holds(const InputCondition& cond, const InputStatus& status) {
    const uint64_t v = cond.metric == InputCondition::Metric::Presents ? status.presents : status.draws_per_frame;
    const bool any_eq = std::find(cond.values.begin(), cond.values.end(), v) != cond.values.end();
    const uint64_t x = cond.values.empty() ? 0 : cond.values[0];
    switch (cond.op) {
        case InputCondition::Op::Eq: return any_eq;
        case InputCondition::Op::Ne: return !any_eq;
        case InputCondition::Op::Ge: return v >= x;
        case InputCondition::Op::Le: return v <= x;
        case InputCondition::Op::Gt: return v > x;
        case InputCondition::Op::Lt: return v < x;
    }
    return false;
}

std::string condition_text(const InputCondition& cond) {
    static const char* const kOps[] = {"==", "!=", ">=", "<=", ">", "<"};
    std::string s = cond.metric == InputCondition::Metric::Presents ? "presents" : "draws_per_frame";
    s += kOps[static_cast<int>(cond.op)];
    for (size_t i = 0; i < cond.values.size(); ++i) {
        s += (i ? "|" : "") + std::to_string(cond.values[i]);
    }
    return s;
}

bool ConditionTracker::update(uint64_t now_ms, const InputStatus& status) {
    const bool holds = condition_holds(*cond_, status);
    if (holds && !holds_) {
        since_ms_ = now_ms;
    } else if (holds && now_ms < since_ms_) {
        // Guard against unsigned underflow if a caller feeds a non-monotonic timestamp.
        since_ms_ = now_ms;
    }
    holds_ = holds;
    if (holds && now_ms >= since_ms_ && (now_ms - since_ms_) >= for_ms_) {
        met_ = true;
    }
    return met_;
}

bool parse_input_script(const std::string& text, std::vector<InputScriptEvent>& out, std::string& error) {
    out.clear();
    std::istringstream lines(text);
    std::string line;
    size_t number = 0;
    size_t run_start = 0;           // first press of the current run (sorted when it ends)
    size_t open_repeat = SIZE_MAX;  // index of an unclosed `repeat`
    while (std::getline(lines, line)) {
        ++number;
        if (const size_t hash = line.find('#'); hash != std::string::npos) {
            line.resize(hash);
        }
        std::istringstream fields(line);
        std::string time_text;
        if (!(fields >> time_text)) {
            continue;  // blank or comment
        }
        const std::string where = "line " + std::to_string(number) + ": ";
        InputScriptEvent e;
        if (time_text == "wait" || time_text == "repeat" || time_text == "end" || time_text == "capture" || time_text == "state") {
            sort_presses(out, run_start);
            if (time_text == "wait") {
                e.kind = InputStepKind::Wait;
                if (!parse_wait_tail(fields, e, error)) {
                    error = where + error;
                    return false;
                }
            } else if (time_text == "repeat") {
                std::string every, period, until;
                fields >> every >> period >> until;
                if (every != "every" || !parse_duration(period, e.every_ms) || e.every_ms == 0 || until != "until") {
                    error = where + "expected 'repeat every <dur> until <cond> ...'";
                    return false;
                }
                if (open_repeat != SIZE_MAX) {
                    error = where + "nested repeat";
                    return false;
                }
                e.kind = InputStepKind::RepeatBegin;
                if (!parse_wait_tail(fields, e, error)) {
                    error = where + error;
                    return false;
                }
                open_repeat = out.size();
            } else if (time_text == "state") {
                e.kind = InputStepKind::State;
                std::string tok[8];
                for (std::string& t : tok) {
                    fields >> t;
                }
                float* axes[] = {&e.left_stick_x, &e.left_stick_y, &e.right_stick_x, &e.right_stick_y, &e.left_trigger,
                                 &e.right_trigger};
                uint64_t buttons = 0;
                bool ok = parse_number(tok[0], e.present) && parse_number(tok[1], buttons) && buttons <= 0xffffffffull;
                for (int k = 0; ok && k < 6; ++k) {
                    char* end = nullptr;
                    errno = 0;
                    const float val = std::strtof(tok[2 + k].c_str(), &end);
                    *axes[k] = val;
                    ok = !tok[2 + k].empty() && errno != ERANGE && end != nullptr && *end == '\0' && std::isfinite(val);
                }
                std::string trailing;
                if (!ok || (fields >> trailing)) {
                    error = where + "expected 'state <present> <buttons> <lx> <ly> <rx> <ry> <lt> <rt>'";
                    return false;
                }
                if (open_repeat != SIZE_MAX) {
                    error = where + "'state' inside repeat";
                    return false;
                }
                e.buttons = static_cast<uint32_t>(buttons);
            } else {
                std::string extra;
                if (fields >> extra) {
                    error = where + "unexpected '" + extra + "'";
                    return false;
                }
                if (time_text == "end") {
                    if (open_repeat == SIZE_MAX) {
                        error = where + "'end' without 'repeat'";
                        return false;
                    }
                    e.kind = InputStepKind::RepeatEnd;
                    out[open_repeat].end_index = out.size();
                    open_repeat = SIZE_MAX;
                } else {
                    if (open_repeat != SIZE_MAX) {
                        error = where + "'capture' inside repeat";
                        return false;
                    }
                    e.kind = InputStepKind::Capture;
                }
            }
            out.push_back(std::move(e));
            run_start = out.size();
            continue;
        }
        std::string buttons_text;
        std::string hold_text;
        std::string extra;
        fields >> buttons_text >> hold_text >> extra;
        if (!parse_number(time_text, e.time_ms)) {
            error = where + "bad time '" + time_text + "'";
            return false;
        }
        if (!parse_buttons(buttons_text, e)) {
            error = where + "bad buttons '" + buttons_text + "'";
            return false;
        }
        if (!hold_text.empty() && (!parse_number(hold_text, e.hold_ms) || e.hold_ms == 0)) {
            error = where + "bad hold time '" + hold_text + "'";
            return false;
        }
        if (!extra.empty()) {
            error = where + "unexpected '" + extra + "'";
            return false;
        }
        out.push_back(std::move(e));
    }
    if (open_repeat != SIZE_MAX) {
        error = "line " + std::to_string(number) + ": 'repeat' without 'end'";
        return false;
    }
    sort_presses(out, run_start);
    return true;
}

namespace {

void append_menu_cycle(std::ostringstream& s, uint64_t base_ms) {
    // Cursor-resetting Legal Notices cycle (DUp x6, DDown x2, A): always lands on "Next >"
    // (Row 2) without opening Row 0 ("Copyright") and advances earlier setup screens via A.
    for (uint64_t i = 0; i < 6; ++i) {
        s << (base_ms + i * 400) << " DUp 200\n";
    }
    s << (base_ms + 3000) << " DDown 200\n";
    s << (base_ms + 4500) << " DDown 200\n";
    s << (base_ms + 6500) << " A 220\n";
}

void append_legal_terms_cycle(std::ostringstream& s, uint64_t base_ms) {
    // Rockstar Games Legal Terms & Policies 2-checkbox + Confirm > cycle (16 s):
    // DUp x6 -> Right-Stick down scroll (900 ms) -> A (Privacy Policy) ->
    // DDown -> Right-Stick down scroll (900 ms) -> A (Terms of Service) ->
    // DDown -> A (Confirm >) -> A (follow-up confirm). Never presses B.
    for (uint64_t i = 0; i < 6; ++i) {
        s << (base_ms + i * 450) << " DUp 200\n";
    }
    s << (base_ms + 2700) << " RDown 900\n";
    s << (base_ms + 4000) << " A 220\n";
    s << (base_ms + 5200) << " DDown 200\n";
    s << (base_ms + 5800) << " RDown 900\n";
    s << (base_ms + 7100) << " A 220\n";
    s << (base_ms + 8300) << " DDown 200\n";
    s << (base_ms + 9500) << " A 220\n";
    s << (base_ms + 12000) << " A 220\n";
}

std::string build_chrome_to_ingame_preset() {
    std::ostringstream s;
    s << "# Hardened chrome-to-ingame preset (warm + cold boot resilient, >=200ms holds, zero B)\n";
    // Phase 1 (5s..74s): Start Screen (~45-55s warm) -> Setup (Language / Subtitles / Brightness /
    // Navigation Experience ~60-75s) with spaced 220ms A presses. Stops before ~80s so plain A
    // never opens Row 0 (Copyright) on Legal Notices.
    for (uint64_t t = 5000; t <= 74000; t += 3000) {
        s << t << " A 220\n";
    }
    // Phase 2 (80s..206s): 9 rounds of 14s cursor-resetting Legal Notices cycle covering both
    // warm (~80-110s) and cold (~150-198s) boots safely.
    for (uint64_t round = 0; round < 9; ++round) {
        append_menu_cycle(s, 80000 + round * 14000);
    }
    // Phase 3 (210s..274s): 4 rounds of 16s Legal Terms scroll + 2-checkbox + Confirm > cycle.
    for (uint64_t round = 0; round < 4; ++round) {
        append_legal_terms_cycle(s, 210000 + round * 16000);
    }
    // Phase 4a (280s..340s): Main Menu -> Play / Story Mode / New Game confirms.
    for (uint64_t t = 280000; t <= 340000; t += 4000) {
        s << t << " A 220\n";
    }
    // Phase 4b (346s..418s): Gentle in-game movement & camera look sweeps + A confirms.
    for (uint64_t block = 0; block < 4; ++block) {
        const uint64_t base = 346000 + block * 18000;
        s << base << " A 220\n";
        s << (base + 2000) << " LUp 2000\n";
        s << (base + 5000) << " RRight 1500\n";
        s << (base + 7500) << " RLeft 1500\n";
        s << (base + 10000) << (block % 2 == 0 ? " LUp+LRight+RRight 2000\n" : " LUp+LLeft+RLeft 2000\n");
        s << (base + 14000) << " A 220\n";
    }
    s << "418000 A 220\n";
    return s.str();
}

std::string build_menu_to_ingame_preset() {
    std::ostringstream s;
    s << "# Hardened screen-synced menu-to-ingame preset (>=200ms holds, Legal Terms scroll, zero B)\n";
    // Phase 1: Start Screen & initial setup screens (Language / UI Text Size / Subtitles /
    // Brightness / Navigation Experience / Finish Setup) until Legal Notices (37) or Legal Terms (33|50|51).
    s << "repeat every 2500ms until draws_per_frame==37|33|50|51 for 2s timeout 180s\n";
    s << "  0 A 220\n";
    s << "end\n";
    // Phase 2: Legal Notices (DUp x6, DDown x2, A -> Next >) until Legal Terms (33|50|51).
    s << "repeat every 14s until draws_per_frame==33|50|51 for 2s timeout 180s\n";
    append_menu_cycle(s, 0);
    s << "end\n";
    // Phase 3: Rockstar Games Legal Terms & Policies (scroll + 2 checkboxes + Confirm >) until passed.
    s << "repeat every 16s until draws_per_frame!=33|50|51 for 2s timeout 180s\n";
    append_legal_terms_cycle(s, 0);
    s << "end\n";
    // Phase 4: Main Menu -> Play / Story Mode / New Game selection.
    s << "repeat every 3s until draws_per_frame>55 for 2s timeout 120s\n";
    s << "  0 A 220\n";
    s << "end\n";
    // Phase 5: Brief in-game movement, camera look, and trigger probes.
    s << "1000 LUp 1500\n";
    s << "3000 RRight 1200\n";
    s << "4700 RLeft 1200\n";
    s << "6500 LUp+RT 1000\n";
    s << "8000 RRight+LT 1000\n";
    return s.str();
}

std::string build_chrome_menu_flow_6_preset() {
    std::ostringstream s;
    s << "# Hardened warm-boot chrome-menu-flow-6 preset (>=200ms holds, Legal Terms scroll, zero B)\n";
    // Start Screen (~45-55s warm, or immediate after FIRST_VISIBLE_FRAME) + Setup (~60-75s).
    for (uint64_t t : {2000ull, 4000ull, 6000ull}) {
        s << t << " A 200\n";
    }
    // 3x 15s Legal Notices / Setup cursor-resetting cycles (35s, 50s, 65s).
    for (uint64_t round = 0; round < 3; ++round) {
        append_menu_cycle(s, 35000 + round * 15000);
    }
    // Legal Terms scroll + 2-checkbox + Confirm > followed by Main Menu confirms.
    for (uint64_t round = 0; round < 4; ++round) {
        append_legal_terms_cycle(s, 85000 + round * 16000);
    }
    for (uint64_t t = 152000; t <= 200000; t += 4000) {
        s << t << " A 220\n";
    }
    return s.str();
}

std::string build_calibration_skip_preset() {
    std::ostringstream s;
    s << "# Hardened calibration-skip preset (>=200ms holds, zero B)\n";
    for (uint64_t t = 5000; t <= 120000; t += 2500) {
        s << t << " A 200\n";
    }
    return s.str();
}

std::string build_legal_notices_preset() {
    std::ostringstream s;
    s << "# Hardened legal-notices preset (>=200ms holds, cursor-reset Next > + Legal Terms, zero B)\n";
    for (uint64_t t = 20000; t <= 62000; t += 2500) {
        s << t << " A 200\n";
    }
    for (uint64_t round = 0; round < 4; ++round) {
        append_menu_cycle(s, 66000 + round * 14000);
    }
    for (uint64_t round = 0; round < 3; ++round) {
        append_legal_terms_cycle(s, 126000 + round * 16000);
    }
    return s.str();
}

std::string describe_press_event(const InputScriptEvent& e) {
    if (e.buttons != 0) {
        return hex(e.buttons);
    }
    std::string out;
    auto add = [&](const char* tag) {
        if (!out.empty()) out += "+";
        out += tag;
    };
    if (e.left_stick_y > 0.0f) add("LUp");
    if (e.left_stick_y < 0.0f) add("LDown");
    if (e.left_stick_x < 0.0f) add("LLeft");
    if (e.left_stick_x > 0.0f) add("LRight");
    if (e.right_stick_y > 0.0f) add("RUp");
    if (e.right_stick_y < 0.0f) add("RDown");
    if (e.right_stick_x < 0.0f) add("RLeft");
    if (e.right_stick_x > 0.0f) add("RRight");
    if (e.left_trigger > 0.0f) add("LT");
    if (e.right_trigger > 0.0f) add("RT");
    return out.empty() ? "0x0" : out;
}

bool event_has_active_input(const InputScriptEvent& e) {
    return e.buttons != 0 || e.left_stick_x != 0.0f || e.left_stick_y != 0.0f || e.right_stick_x != 0.0f ||
           e.right_stick_y != 0.0f || e.left_trigger != 0.0f || e.right_trigger != 0.0f;
}

}  // namespace

std::vector<std::string> builtin_input_script_preset_names() {
    return {"chrome-to-ingame", "menu-to-ingame", "chrome-menu-flow-6", "calibration-skip", "legal-notices"};
}

bool builtin_input_script_preset(const std::string& name, std::string& out_script) {
    if (name == "chrome-to-ingame") {
        out_script = build_chrome_to_ingame_preset();
        return true;
    }
    if (name == "menu-to-ingame") {
        out_script = build_menu_to_ingame_preset();
        return true;
    }
    if (name == "chrome-menu-flow-6") {
        out_script = build_chrome_menu_flow_6_preset();
        return true;
    }
    if (name == "calibration-skip") {
        out_script = build_calibration_skip_preset();
        return true;
    }
    if (name == "legal-notices") {
        out_script = build_legal_notices_preset();
        return true;
    }
    return false;
}

bool parse_input_script_or_preset(const std::string& name_or_text,
                                  std::vector<InputScriptEvent>& out,
                                  std::string& error) {
    std::string preset_script;
    if (builtin_input_script_preset(name_or_text, preset_script)) {
        return parse_input_script(preset_script, out, error);
    }
    return parse_input_script(name_or_text, out, error);
}

InputScriptPlayer::InputScriptPlayer(std::vector<InputScriptEvent> events, playgta6::InputState* state)
    : events_(std::move(events)), state_(state) {}

InputScriptPlayer::~InputScriptPlayer() { stop(); }

void InputScriptPlayer::start() {
    if (state_ == nullptr || thread_.joinable()) {
        return;
    }
    stop_.store(false);
    finished_.store(false);
    yielded_.store(false);
    writes_.store(0);
    waits_done_.store(0);
    waits_timed_out_.store(0);
    thread_ = std::thread([this] { run(); });
}

void InputScriptPlayer::stop() {
    stop_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void InputScriptPlayer::write(const InputScriptEvent* held) {
    playgta6::InputState values{};
    if (held != nullptr) {
        values.buttons = held->buttons;
        values.left_stick_x = held->left_stick_x;
        values.left_stick_y = held->left_stick_y;
        values.right_stick_x = held->right_stick_x;
        values.right_stick_y = held->right_stick_y;
        values.left_trigger = held->left_trigger;
        values.right_trigger = held->right_trigger;
    }
    playgta6::os::GameInputReader::write_monotonic(*state_, values);
    writes_.fetch_add(1);
}

void InputScriptPlayer::run() {
    playgta6::os::name_this_thread(L"pg6-input-script");
    using clock = std::chrono::steady_clock;
    const clock::time_point t0 = clock::now();
    auto ms_since_start = [&] {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count());
    };
    auto log = [&](const std::string& what) {
        if (log_) {
            std::fprintf(stderr, "[input] %llu ms: %s\n", static_cast<unsigned long long>(ms_since_start()), what.c_str());
        }
    };
    enum class Until { Reached, Met, Stopped, TimedOut };
    bool state_latched = false;
    auto check_yield = [&]() -> bool {
        if (live_yield_probe_ && live_yield_probe_()) {
            yielded_.store(true);
            stop_.store(true);
            log("yielded to live user input");
            return true;
        }
        return false;
    };
    auto release_if_state_latched = [&]() {
        if (state_latched) {
            write(nullptr);
            state_latched = false;
        }
    };
    // Sleeps until `at` in short steps so stop() and live yield take effect promptly; with a
    // tracker, also samples the live status every step and returns Met as soon as its condition is met.
    auto wait_until = [&](clock::time_point at, ConditionTracker* tracker, int tracker_step_ms = 10) {
        while (!stop_.load()) {
            if (check_yield()) {
                return Until::Stopped;
            }
            if (tracker != nullptr && tracker->update(ms_since_start(), status())) {
                return Until::Met;
            }
            const clock::time_point now = clock::now();
            if (now >= at) {
                return Until::Reached;
            }
            const auto step = std::chrono::milliseconds(tracker != nullptr ? tracker_step_ms : 20);
            std::this_thread::sleep_for(std::min<clock::duration>(at - now, step));
        }
        return Until::Stopped;
    };
    // Presses `e` at base + e.time_ms (skipped while `tracker`'s condition holds, and bounded by
    // `limit` when executing inside a timed repeat). Holds for the full effective hold_ms measured
    // from actual press time so late wakeups or back-to-back presses never collapse to 0 ms.
    auto press = [&](const InputScriptEvent& e, clock::time_point base, ConditionTracker* tracker,
                     clock::time_point limit) {
        const clock::time_point at = base + std::chrono::milliseconds(e.time_ms);
        const bool has_limit = limit != clock::time_point::max();
        const clock::time_point wake_at = has_limit ? std::min(at, limit) : at;
        const Until u = wait_until(wake_at, tracker);
        if (u != Until::Reached) {
            return u;
        }
        if (has_limit && clock::now() >= limit) {
            return Until::TimedOut;
        }
        if (tracker != nullptr && tracker->holds_now()) {
            return Until::Reached;  // already on the target screen: no press
        }
        const uint64_t effective_hold = std::max(e.hold_ms, min_hold_ms_);
        if (log_) {
            log("press " + describe_press_event(e) + " " + std::to_string(effective_hold));
        }
        const clock::time_point pressed_at = clock::now();
        write(&e);
        const clock::time_point hold_end = pressed_at + std::chrono::milliseconds(effective_hold);
        const Until held = wait_until(has_limit ? std::min(hold_end, limit) : hold_end, nullptr);
        write(nullptr);
        if (held == Until::Stopped) {
            return Until::Stopped;
        }
        if (has_limit && clock::now() >= limit) {
            return Until::TimedOut;
        }
        return held;
    };
    auto finish_wait = [&](const InputScriptEvent& e, bool met, const std::string& what) {
        const InputStatus s = status();
        log(what + " " + condition_text(e.cond) + (met ? " met" : " timed out") + " (presents " +
            std::to_string(s.presents) + ", draws_per_frame " + std::to_string(s.draws_per_frame) + ")");
        if (!met) {
            waits_timed_out_.fetch_add(1);
        }
        waits_done_.fetch_add(1);
    };
    clock::time_point base = t0;
    for (size_t i = 0; i < events_.size(); ++i) {
        const InputScriptEvent& e = events_[i];
        switch (e.kind) {
            case InputStepKind::Press:
                state_latched = false;
                if (press(e, base, nullptr, clock::time_point::max()) == Until::Stopped) {
                    return;
                }
                break;
            case InputStepKind::Wait: {
                ConditionTracker tracker(e.cond, e.for_ms);
                const clock::time_point limit =
                    e.timeout_ms ? clock::now() + std::chrono::milliseconds(e.timeout_ms) : clock::time_point::max();
                const Until u = wait_until(limit, &tracker);
                if (u == Until::Stopped) {
                    release_if_state_latched();
                    return;
                }
                finish_wait(e, u == Until::Met, "wait");
                base = clock::now();
                break;
            }
            case InputStepKind::RepeatBegin: {
                ConditionTracker tracker(e.cond, e.for_ms);
                const clock::time_point start = clock::now();
                const clock::time_point limit =
                    e.timeout_ms ? start + std::chrono::milliseconds(e.timeout_ms) : clock::time_point::max();
                Until u = Until::Reached;
                uint64_t rounds = 0;
                for (clock::time_point round = start; u == Until::Reached && round < limit;
                     round += std::chrono::milliseconds(e.every_ms)) {
                    ++rounds;
                    for (size_t k = i + 1; k < e.end_index && u == Until::Reached; ++k) {
                        u = press(events_[k], round, &tracker, limit);
                    }
                    if (u == Until::Reached) {
                        u = wait_until(std::min(limit, round + std::chrono::milliseconds(e.every_ms)), &tracker);
                    }
                }
                if (u == Until::Stopped) {
                    release_if_state_latched();
                    return;
                }
                finish_wait(e, u == Until::Met, "repeat (" + std::to_string(rounds) + " round(s))");
                base = clock::now();
                i = e.end_index;
                break;
            }
            case InputStepKind::State: {
                InputCondition reached;
                reached.metric = InputCondition::Metric::Presents;
                reached.op = InputCondition::Op::Ge;
                reached.values = {e.present};
                ConditionTracker tracker(reached, 0);
                if (wait_until(clock::time_point::max(), &tracker, 1) == Until::Stopped) {
                    release_if_state_latched();
                    return;
                }
                if (log_) {
                    log("state " + hex(e.buttons) + " at present " + std::to_string(e.present));
                }
                write(&e);
                state_latched = event_has_active_input(e);
                break;
            }
            case InputStepKind::Capture:
                log("capture");
                if (on_capture_) {
                    on_capture_();
                }
                break;
            case InputStepKind::RepeatEnd:
                break;
        }
    }
    finished_.store(true);
}

}  // namespace playgta6::guest
