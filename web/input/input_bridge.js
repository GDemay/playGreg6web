// Writes playgta6::InputState (40 bytes, runtime/abi/playgta6_abi.h & docs/wasm_abi.md)
// into a SharedArrayBuffer using atomic writes + even/odd seqlock framing compatible with
// both SeqlockMode::MonotonicSeq and SeqlockMode::EvenOddSeqlock in runtime/os/game_input.cpp.

import {
  INPUT_BUTTONS,
  DEFAULT_KEY_BINDINGS,
  applyRadialDeadzone,
} from './input_profiles.js';

export { INPUT_BUTTONS, DEFAULT_KEY_BINDINGS, applyRadialDeadzone };

export const INPUT_ABI_VERSION = 1;
export const INPUT_STATE_BYTES = 40;
export const INPUT_STATE_WORDS = 10;

export const INPUT_OFFSETS = Object.freeze({
  SEQ: 0,            // uint32_t (u32[0])
  BUTTONS: 4,        // uint32_t (u32[1])
  MOUSE_DX: 8,       // int32_t  (i32[2])
  MOUSE_DY: 12,      // int32_t  (i32[3])
  LEFT_STICK_X: 16,  // float32  (f32[4])
  LEFT_STICK_Y: 20,  // float32  (f32[5])
  RIGHT_STICK_X: 24, // float32  (f32[6])
  RIGHT_STICK_Y: 28, // float32  (f32[7])
  LEFT_TRIGGER: 32,  // float32  (f32[8])
  RIGHT_TRIGGER: 36, // float32  (f32[9])
});

export const INPUT_WORD_INDICES = Object.freeze({
  SEQ: 0,
  BUTTONS: 1,
  MOUSE_DX: 2,
  MOUSE_DY: 3,
  LEFT_STICK_X: 4,
  LEFT_STICK_Y: 5,
  RIGHT_STICK_X: 6,
  RIGHT_STICK_Y: 7,
  LEFT_TRIGGER: 8,
  RIGHT_TRIGGER: 9,
});

export const INPUT_ABI_LAYOUT = Object.freeze({
  version: INPUT_ABI_VERSION,
  byteLength: INPUT_STATE_BYTES,
  wordCount: INPUT_STATE_WORDS,
  fields: Object.freeze([
    Object.freeze({ name: 'seq', key: 'SEQ', type: 'uint32', offset: 0, wordIndex: 0, size: 4 }),
    Object.freeze({ name: 'buttons', key: 'BUTTONS', type: 'uint32', offset: 4, wordIndex: 1, size: 4 }),
    Object.freeze({ name: 'mouse_dx', key: 'MOUSE_DX', type: 'int32', offset: 8, wordIndex: 2, size: 4 }),
    Object.freeze({ name: 'mouse_dy', key: 'MOUSE_DY', type: 'int32', offset: 12, wordIndex: 3, size: 4 }),
    Object.freeze({ name: 'left_stick_x', key: 'LEFT_STICK_X', type: 'float32', offset: 16, wordIndex: 4, size: 4 }),
    Object.freeze({ name: 'left_stick_y', key: 'LEFT_STICK_Y', type: 'float32', offset: 20, wordIndex: 5, size: 4 }),
    Object.freeze({ name: 'right_stick_x', key: 'RIGHT_STICK_X', type: 'float32', offset: 24, wordIndex: 6, size: 4 }),
    Object.freeze({ name: 'right_stick_y', key: 'RIGHT_STICK_Y', type: 'float32', offset: 28, wordIndex: 7, size: 4 }),
    Object.freeze({ name: 'left_trigger', key: 'LEFT_TRIGGER', type: 'float32', offset: 32, wordIndex: 8, size: 4 }),
    Object.freeze({ name: 'right_trigger', key: 'RIGHT_TRIGGER', type: 'float32', offset: 36, wordIndex: 9, size: 4 }),
  ]),
});

// W3C Standard Gamepad button index -> INPUT_BUTTONS bitmask (indices 6 & 7 are analog triggers).
export const STANDARD_GAMEPAD_BUTTON_MAP = Object.freeze([
  INPUT_BUTTONS.A,          // 0: Bottom face button (A)
  INPUT_BUTTONS.B,          // 1: Right face button (B)
  INPUT_BUTTONS.X,          // 2: Left face button (X)
  INPUT_BUTTONS.Y,          // 3: Top face button (Y)
  INPUT_BUTTONS.LB,         // 4: Left bumper (LB)
  INPUT_BUTTONS.RB,         // 5: Right bumper (RB)
  0,                        // 6: Left trigger (analog -> lt)
  0,                        // 7: Right trigger (analog -> rt)
  INPUT_BUTTONS.VIEW,       // 8: Back / View
  INPUT_BUTTONS.MENU,       // 9: Start / Menu
  INPUT_BUTTONS.LS,         // 10: Left stick press (L3)
  INPUT_BUTTONS.RS,         // 11: Right stick press (R3)
  INPUT_BUTTONS.DPAD_UP,    // 12: D-Pad Up
  INPUT_BUTTONS.DPAD_DOWN,  // 13: D-Pad Down
  INPUT_BUTTONS.DPAD_LEFT,  // 14: D-Pad Left
  INPUT_BUTTONS.DPAD_RIGHT, // 15: D-Pad Right
  INPUT_BUTTONS.GUIDE,      // 16: Home / Guide
]);

const PREVENT_DEFAULT_KEYS = new Set([
  'Space',
  'Enter',
  'NumpadEnter',
  'Backspace',
  'Tab',
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
]);

const EMPTY_ARRAY = Object.freeze([]);

const EDITABLE_TAG_NAMES = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

function isEditableElement(el) {
  if (!el || typeof el !== 'object') return false;
  if (el.isContentEditable) return true;
  const tag = typeof el.tagName === 'string' ? el.tagName.toUpperCase() : '';
  return EDITABLE_TAG_NAMES.has(tag);
}

const _f32ScratchBuf = new ArrayBuffer(4);
const _f32Scratch = new Float32Array(_f32ScratchBuf);
const _i32Scratch = new Int32Array(_f32ScratchBuf);

function f32ToI32(value) {
  _f32Scratch[0] = value;
  return _i32Scratch[0];
}

function i32ToF32(bits) {
  _i32Scratch[0] = bits;
  return _f32Scratch[0];
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return 0;
  return value < min ? min : value > max ? max : value;
}

/**
 * Radial 2D stick deadzone normalization matching playgta6::os::normalize_stick.
 * Maps stick magnitude in [innerDeadzone, outerDeadzone] -> [0, 1] while preserving direction.
 * Pass a pre-allocated `out` object `{ x, y }` for zero-allocation polling.
 */
export function applyDeadzone(x, y, innerDeadzone = 0.15, outerDeadzone = 0.98, out = null) {
  const target = out || { x: 0, y: 0 };
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    target.x = 0;
    target.y = 0;
    return target;
  }
  const inner = clamp(innerDeadzone, 0, 0.99);
  const outer = clamp(outerDeadzone, inner + 1e-4, 1);

  const mag = Math.hypot(x, y);
  if (mag <= inner || mag <= 0) {
    target.x = 0;
    target.y = 0;
    return target;
  }

  const clampedMag = Math.min(mag, outer);
  const normalizedMag = clamp((clampedMag - inner) / (outer - inner), 0, 1);
  const scale = normalizedMag / mag;
  target.x = clamp(x * scale, -1, 1);
  target.y = clamp(y * scale, -1, 1);
  return target;
}

/**
 * Axial 1D deadzone normalization with rescaling into [-1, 1].
 */
export function applyAxialDeadzone(value, innerDeadzone = 0.15, outerDeadzone = 0.98) {
  if (!Number.isFinite(value)) return 0;
  const inner = clamp(innerDeadzone, 0, 0.99);
  const outer = clamp(outerDeadzone, inner + 1e-4, 1);
  const abs = Math.abs(value);
  if (abs <= inner) return 0;
  const normalized = clamp((Math.min(abs, outer) - inner) / (outer - inner), 0, 1);
  return value < 0 ? -normalized : normalized;
}

/**
 * Linear trigger deadzone normalization matching playgta6::os::normalize_trigger.
 * Maps trigger value in [innerDeadzone, outerDeadzone] -> [0, 1].
 */
export function applyTriggerDeadzone(value, innerDeadzone = 0.05, outerDeadzone = 0.98) {
  if (!Number.isFinite(value)) return 0;
  const inner = clamp(innerDeadzone, 0, 0.99);
  const outer = clamp(outerDeadzone, inner + 1e-4, 1);
  if (value <= inner) return 0;
  if (value >= outer) return 1;
  return clamp((value - inner) / (outer - inner), 0, 1);
}

function isGamepadButtonPressed(button) {
  if (button == null) return false;
  if (typeof button === 'number') return button > 0.5;
  return Boolean(button.pressed || (typeof button.value === 'number' && button.value > 0.5));
}

function readGamepadTrigger(button) {
  if (button == null) return 0;
  if (typeof button === 'number') return clamp(button, 0, 1);
  if (typeof button.value === 'number' && Number.isFinite(button.value)) {
    return clamp(button.value, 0, 1);
  }
  return button.pressed ? 1 : 0;
}

export const MIN_SCRIPT_HOLD_MS = 180;
export const CHROME_MENU_FLOW_4_HOLD_MS = 200;
export const IDLE_GAMEPAD_POLL_INTERVAL_MS = 500;
export const DEFAULT_VISIBLE_FRAME_FALLBACK_MS = 2000;
export const FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS = 500;
export const FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS = 220;

export const FIRST_VISIBLE_FRAME_ANCHOR_PRESETS = Object.freeze([
  'menu-to-ingame',
  'chrome-to-ingame',
  'chrome-menu-flow-5',
  'chrome-menu-flow-4',
  'chrome-menu-flow-2',
  'chrome-menu-flow',
]);
const FIRST_VISIBLE_FRAME_ANCHOR_PRESET_SET = new Set(FIRST_VISIBLE_FRAME_ANCHOR_PRESETS);
const FIRST_VISIBLE_FRAME_MIN_GAP_MS = 500;

export const FIRST_VISIBLE_FRAME_INITIAL_A_EVENTS = Object.freeze([
  Object.freeze({
    atMs: FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS,
    holdMs: FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS,
    button: INPUT_BUTTONS.A,
  }),
  Object.freeze({
    atMs: 2500,
    holdMs: FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS,
    button: INPUT_BUTTONS.A,
  }),
  Object.freeze({
    atMs: 4500,
    holdMs: FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS,
    button: INPUT_BUTTONS.A,
  }),
]);

export const INPUT_SCRIPT_START_TRIGGERS = Object.freeze({
  IMMEDIATE: 'immediate',
  FIRST_PRESENT: 'FIRST_PRESENT',
  FIRST_VISIBLE_FRAME: 'FIRST_VISIBLE_FRAME',
});

function buildPeriodicButtonEvents(startMs, endMs, periodMs, holdMs, button) {
  const events = [];
  for (let atMs = startMs; atMs <= endMs; atMs += periodMs) {
    events.push(Object.freeze({ atMs, holdMs, button }));
  }
  return Object.freeze(events);
}

// Mirrors tools/run_native_boot.py::_menu_cycle (PR #1483):
// Up x6 (cursor reset to top), Down, Down, A — selects "Next >" on LEGAL NOTICES
// (and advances earlier screens via A) with >= 180 ms holds for 15-30 FPS browser boots.
const CHROME_MENU_FLOW_4_CYCLE_OFFSETS = Object.freeze([
  Object.freeze({ offsetMs: 0, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 400, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 800, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 1200, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 1600, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 2000, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 3000, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_DOWN }),
  Object.freeze({ offsetMs: 4500, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_DOWN }),
  Object.freeze({ offsetMs: 6500, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.A }),
]);

// W362 (Refs #663): 16 s ROCKSTAR GAMES LEGAL TERMS & POLICIES checkbox-accept cycle.
// Live Chrome t90 layout has 3 rows:
//   Row 0: [ ] Privacy Policy (SCROLL [R] / SELECT [A])
//   Row 1: [ ] Terms of Service (SCROLL [R] / SELECT [A])
//   Row 2: Confirm > ("Confirm setup and start the game." — disabled until Row 0 & Row 1 checked)
// Right-stick scroll uses ry: -0.95 (GameInput/XInput Y-up ABI where -1.0 is down).
// All button holds are >= 200 ms (>= MIN_SCRIPT_HOLD_MS) and B is never pressed.
export const CHROME_LEGAL_TERMS_ACCEPT_OFFSETS = Object.freeze([
  Object.freeze({ offsetMs: 0, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 450, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 900, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 1350, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 1800, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 2250, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP }),
  Object.freeze({ offsetMs: 2700, holdMs: 900, sticks: Object.freeze({ ry: -0.95 }) }),
  Object.freeze({ offsetMs: 4000, holdMs: 220, button: INPUT_BUTTONS.A }),
  Object.freeze({ offsetMs: 5200, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_DOWN }),
  Object.freeze({ offsetMs: 5800, holdMs: 900, sticks: Object.freeze({ ry: -0.95 }) }),
  Object.freeze({ offsetMs: 7100, holdMs: 220, button: INPUT_BUTTONS.A }),
  Object.freeze({ offsetMs: 8300, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_DOWN }),
  Object.freeze({ offsetMs: 9500, holdMs: 220, button: INPUT_BUTTONS.A }),
  Object.freeze({ offsetMs: 12000, holdMs: 220, button: INPUT_BUTTONS.A }),
]);

function expandOffsetCycle(events, baseMs, offsets) {
  for (let i = 0; i < offsets.length; i++) {
    const step = offsets[i];
    const ev = {
      atMs: baseMs + step.offsetMs,
      holdMs: step.holdMs,
    };
    if (step.button !== undefined) ev.button = step.button;
    if (step.sticks !== undefined) ev.sticks = step.sticks;
    if (step.triggers !== undefined) ev.triggers = step.triggers;
    events.push(Object.freeze(ev));
  }
}

function buildChromeMenuFlow4Events() {
  const events = [];
  // Phase 1 (5s..75s): Early A confirmations every 2500 ms (holdMs: 200 >= 180 ms, 29 presses).
  // Starts at 5000 ms (anchored to FIRST_PRESENT, not gated on FIRST_VISIBLE_FRAME) so a dark
  // Start Screen never stalls waiting 110 s like chrome-menu-flow-2, then continues across
  // 6-star -> Bink2 -> Rockstar -> UI TEXT SIZE -> FINISH SETUP?.
  for (let atMs = 5000; atMs <= 75000; atMs += 2500) {
    events.push(
      Object.freeze({ atMs, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.A })
    );
  }
  // Phase 2 (80s..170.5s): 7 rounds of Timon's 14 s _menu_cycle (Up x6, Down, Down, A)
  // matching tools/run_native_boot.py::_to_list_screen (PR #1483) to advance any remaining
  // setup screens and deterministically select "Next >" on LEGAL NOTICES.
  for (let round = 0; round < 7; round++) {
    expandOffsetCycle(events, 80000 + round * 14000, CHROME_MENU_FLOW_4_CYCLE_OFFSETS);
  }
  // Phase 3 (178s..180s): Post-Legal List Screen top-item select matching native menu-flow-4
  // (PR #1483: DUp to put cursor in 50-draw top state, then +2000 ms A to trigger the ~3 s
  // transition into the 34-draw screen / Main Menu, followed by a ~10 s settle window).
  events.push(
    Object.freeze({ atMs: 178000, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.DPAD_UP })
  );
  events.push(
    Object.freeze({ atMs: 180000, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.A })
  );
  // Phase 4 (184s..360s, W361/W362): Extended long-Bink2 continuation so Wasm64 browser runs where
  // 4K Bink2 takes 30-160 s never run out of scripted inputs after UI TEXT SIZE.
  // Phase 4a (184s..212s): 10 spaced A confirmations every 3000 ms (holdMs: 200, never B) to
  // advance through UI TEXT SIZE -> BRIGHTNESS / HDR / SUBTITLES -> FINISH SETUP? -> LEGAL NOTICES.
  for (let i = 0; i < 10; i++) {
    events.push(
      Object.freeze({
        atMs: 184000 + i * 3000,
        holdMs: CHROME_MENU_FLOW_4_HOLD_MS,
        button: INPUT_BUTTONS.A,
      })
    );
  }
  // Phase 4b (218s..316s): 7 rounds of Timon's 14 s _menu_cycle (Up x6, Down, Down, A) every
  // 14000 ms to deterministically highlight & confirm "Next >" on LEGAL NOTICES (and advance any
  // remaining setup screen).
  for (let round = 0; round < 7; round++) {
    expandOffsetCycle(events, 218000 + round * 14000, CHROME_MENU_FLOW_4_CYCLE_OFFSETS);
  }
  // Phase 4c (320s..360s, W362): 2 rounds of 16 s ROCKSTAR GAMES LEGAL TERMS & POLICIES
  // checkbox-accept sequence (320s..352s: Up x6 -> right-stick scroll -> A [Privacy Policy] ->
  // Down -> right-stick scroll -> A [Terms of Service] -> Down -> A [Confirm >] -> A) followed
  // by an A confirmation at 358s for Main Menu / Story entry (never presses B).
  for (let round = 0; round < 2; round++) {
    expandOffsetCycle(events, 320000 + round * 16000, CHROME_LEGAL_TERMS_ACCEPT_OFFSETS);
  }
  events.push(
    Object.freeze({ atMs: 358000, holdMs: 220, button: INPUT_BUTTONS.A })
  );
  return Object.freeze(events);
}

// W364 (Refs #663): Fast 4K Bink2 (~22-26 s after FIRST_VISIBLE_FRAME with G73+G75+R63)
// setup flow matching #1581/#1592/#1614:
// - Phase 1 (0s..10s): Start Screen A presses at 2s, 4s, 6s to start 4K Bink2 intro.
// - Phase 2 (35s..80s): 3x 15 s-spaced _menu_cycle rounds (Up x6, Down, Down, A at 35s, 50s, 65s)
//   to pass LEGAL NOTICES right after Bink2 settles and reach ROCKSTAR GAMES LEGAL TERMS & POLICIES.
// - Phase 3 (85s+, looping every 16 s): CHROME_LEGAL_TERMS_ACCEPT_OFFSETS to check Privacy Policy
//   + Terms of Service, click Confirm >, and enter Main Menu / Story (never presses B).
function buildChromeMenuFlow5Events() {
  const events = [];
  for (const atMs of [2000, 4000, 6000]) {
    events.push(
      Object.freeze({ atMs, holdMs: CHROME_MENU_FLOW_4_HOLD_MS, button: INPUT_BUTTONS.A })
    );
  }
  for (let round = 0; round < 3; round++) {
    expandOffsetCycle(events, 35000 + round * 15000, CHROME_MENU_FLOW_4_CYCLE_OFFSETS);
  }
  return Object.freeze(events);
}

function buildChromeToIngameEvents() {
  const events = [];
  // Phase 1 (5s..145s): Spaced A every 3000 ms (holdMs: 220, 47 presses: 5000..143000 ms)
  // to advance Start Screen -> 4K Bink2 -> R* -> UI TEXT SIZE -> BRIGHTNESS -> FINISH SETUP?
  // -> LEGAL NOTICES.
  for (let atMs = 5000; atMs <= 145000; atMs += 3000) {
    events.push(Object.freeze({ atMs, holdMs: 220, button: INPUT_BUTTONS.A }));
  }
  // Phase 2 (150s..206s): 4 rounds of 14 s LEGAL NOTICES _menu_cycle (Up x6, Down, Down, A)
  // to click "Next >" into ROCKSTAR GAMES LEGAL TERMS & POLICIES (harmless if already there).
  for (let round = 0; round < 4; round++) {
    expandOffsetCycle(events, 150000 + round * 14000, CHROME_MENU_FLOW_4_CYCLE_OFFSETS);
  }
  // Phase 3 (210s..274s): 4 rounds of 16 s CHROME_LEGAL_TERMS_ACCEPT_OFFSETS to check BOTH
  // Privacy Policy & Terms of Service checkboxes and click "Confirm >" ("Confirm setup and
  // start the game.").
  for (let round = 0; round < 4; round++) {
    expandOffsetCycle(events, 210000 + round * 16000, CHROME_LEGAL_TERMS_ACCEPT_OFFSETS);
  }
  // Phase 4a (280s..340s): Main Menu -> Story Mode / New Game selection (spaced A every 4000 ms,
  // holdMs: 220, 16 presses: 280000..340000 ms, never pressing B).
  for (let atMs = 280000; atMs <= 340000; atMs += 4000) {
    events.push(Object.freeze({ atMs, holdMs: 220, button: INPUT_BUTTONS.A }));
  }
  // Phase 4b (346s..420s): Gentle in-game movement & camera look sweeps interspersed with brief
  // A confirms (completes cleanly at 418220 ms <= 420000 ms with all inputs released; never B).
  for (let block = 0; block < 4; block++) {
    const base = 346000 + block * 18000;
    const strafeSign = block % 2 === 0 ? 1 : -1;
    events.push(Object.freeze({ atMs: base, holdMs: 220, button: INPUT_BUTTONS.A }));
    events.push(
      Object.freeze({
        atMs: base + 2000,
        holdMs: 2000,
        sticks: Object.freeze({ lx: 0, ly: 0.85 }),
      })
    );
    events.push(
      Object.freeze({
        atMs: base + 5000,
        holdMs: 1500,
        sticks: Object.freeze({ rx: 0.7, ry: 0.15 }),
      })
    );
    events.push(
      Object.freeze({
        atMs: base + 7500,
        holdMs: 1500,
        sticks: Object.freeze({ rx: -0.7, ry: -0.15 }),
      })
    );
    events.push(
      Object.freeze({
        atMs: base + 10000,
        holdMs: 2000,
        sticks: Object.freeze({ lx: 0.5 * strafeSign, ly: 0.75, rx: 0.35 * strafeSign, ry: 0 }),
      })
    );
    events.push(Object.freeze({ atMs: base + 14000, holdMs: 220, button: INPUT_BUTTONS.A }));
  }
  events.push(Object.freeze({ atMs: 418000, holdMs: 220, button: INPUT_BUTTONS.A }));
  return Object.freeze(events);
}

function buildChromeMenuFlow2Events() {
  const events = [];
  // Phase 1: Human-paced A confirmations every 2500 ms from 110000 ms through 165000 ms
  // (Start Screen -> CONTINUE -> Calibration -> UI TEXT SIZE in Chrome Wasm64 boots).
  for (let atMs = 110000; atMs <= 165000; atMs += 2500) {
    events.push(Object.freeze({ atMs, holdMs: 100, button: INPUT_BUTTONS.A }));
  }
  // Phase 2: 3 human-paced DPAD_DOWN, DPAD_DOWN, A cycles every 12000 ms from 169000 ms
  // to highlight and select "Next >" on LEGAL NOTICES (mirrors PR #1333).
  for (let cycle = 0; cycle < 3; cycle++) {
    const base = 169000 + cycle * 12000;
    events.push(Object.freeze({ atMs: base, holdMs: 150, button: INPUT_BUTTONS.DPAD_DOWN }));
    events.push(Object.freeze({ atMs: base + 2000, holdMs: 150, button: INPUT_BUTTONS.DPAD_DOWN }));
    events.push(Object.freeze({ atMs: base + 4500, holdMs: 100, button: INPUT_BUTTONS.A }));
  }
  // Phase 3: Single A at 206500 ms (+9 s after Phase 2) to leave the post-legal list screen.
  events.push(Object.freeze({ atMs: 206500, holdMs: 100, button: INPUT_BUTTONS.A }));
  // Phase 4: Cycling DPAD_DOWN x N + A (N = 1..5, spaced 25000 ms apart from 230000 ms)
  // to advance through post-legal menu screens (never presses B or DPAD_UP).
  for (let n = 1; n <= 5; n++) {
    const groupStart = 230000 + (n - 1) * 25000;
    for (let i = 0; i < n; i++) {
      events.push(
        Object.freeze({ atMs: groupStart + i * 1500, holdMs: 150, button: INPUT_BUTTONS.DPAD_DOWN })
      );
    }
    events.push(
      Object.freeze({ atMs: groupStart + n * 1500 + 1500, holdMs: 100, button: INPUT_BUTTONS.A })
    );
  }
  return Object.freeze(events);
}

// Named deterministic scripted-input presets.
// - 'start-screen': matches tools/run_native_boot.py (A held 100 ms every 1000 ms starting at 5000 ms).
// - 'menu-flow': periodic A presses on Start Screen, clean pauses across CONTINUE / UI TEXT SIZE,
//   spaced A confirmations through calibration screens, then gentle periodic D-pad Down/Up + A
//   navigation probes in the Main Menu (never presses B).
// - 'calibration-skip': steady A confirmations spaced safely every 2500 ms starting at 5000 ms
//   to advance cleanly through Start Screen -> CONTINUE -> UI TEXT SIZE -> calibration screens
//   without double-tapping or backing out (B).
// - 'intro-watch': waits 18 s (past the 15 s+ 4K Bink2 intro video) before issuing spaced A presses
//   so automated captures record the full intro video before entering menus.
// - 'legal-notices': mirrors PR #948's native boot sequence (A every 2500 ms from 20000 ms to
//   62000 ms through Start Screen / loading / UI TEXT SIZE, then looping DPAD_DOWN, DPAD_DOWN, A
//   every 12000 ms from 64000 ms to select "Next >" on LEGAL NOTICES with interspersed A every
//   3000 ms; never presses B or DPAD_UP).
// - 'chrome-menu-flow': Chrome-timed equivalent for real Wasm64 browser boot (waits through the
//   4K Bink2 intro & shader init with spaced A every 2500 ms from 165000 ms through 215000 ms,
//   then loops DPAD_DOWN, DPAD_DOWN, A, A, A every 12000 ms from 218000 ms; never presses B or DPAD_UP).
// - 'chrome-full-boot': Real Wasm64 headless Chrome preset that lets the entire 4K Bink2 intro
//   video & Rockstar logo play unskipped before spaced A every 2500 ms from 235000 ms through
//   295000 ms, then loops DPAD_DOWN, DPAD_DOWN, A, A, A every 12000 ms from 298000 ms to advance
//   LEGAL NOTICES ("Next >") and subsequent setup screens (never presses B or DPAD_UP).
// - 'menu-to-ingame': Bounded, deterministic 5-phase Chrome automation preset driving
//   Start Screen A confirm -> Calibration/Legal accepts (DPAD_DOWN x2 + A for "Next >") ->
//   Intro cinematic skip (hold A / MENU) -> Main Menu Story/Play selection -> brief in-game
//   left stick, right stick, and LT/RT trigger movement (runtime #1139), completing cleanly
//   at 59300 ms with all inputs released (never presses B or DPAD_UP).
// - 'chrome-menu-flow-2': Mirrors PR #1333 (menu-flow-2) for Chrome Wasm64 boots:
//   human-paced A every 2500 ms from 110000 ms through 165000 ms (Start Screen / Calibration /
//   UI TEXT SIZE), 3 cycles of DPAD_DOWN, DPAD_DOWN, A every 12000 ms from 169000 ms for
//   LEGAL NOTICES ("Next >"), A at 206500 ms to leave the post-legal list screen, cycling
//   DPAD_DOWN x N + A (N = 1..5, spaced 25000 ms apart from 230000 ms), then looping A every
//   9000 ms from 360000 ms into Main Menu / In-Game (never presses B or DPAD_UP).
// - 'chrome-menu-flow-4': Mirrors native menu-flow-4 (tools/run_native_boot.py, PR #1483)
//   for Chrome Wasm64 boots with >= 180 ms (200 ms) holds for 15-30 FPS & #1501 press latch:
//   Phase 1 early A every 2500 ms from 5000 ms through 75000 ms (starts right after FIRST_PRESENT
//   so a dark Start Screen never stalls while splash waits for FIRST_VISIBLE_FRAME, continuing
//   through 6-star -> Bink2 -> Rockstar -> UI TEXT SIZE -> FINISH SETUP?), Phase 2 7 rounds of
//   14 s cursor-resetting _menu_cycle (DPAD_UP x6, DPAD_DOWN, DPAD_DOWN, A) from 80000 ms for
//   LEGAL NOTICES ("Next >"), Phase 3 Post-Legal List Screen DPAD_UP (178000 ms) + A (180000 ms),
//   Phase 4 (W361/W362, 184000..358000 ms) extended long-Bink2 continuation (4a: 10 spaced A every
//   3000 ms from 184000 ms through 211000 ms; 4b: 7 rounds of 14 s _menu_cycle from 218000 ms;
//   4c: 2 rounds of 16 s CHROME_LEGAL_TERMS_ACCEPT_OFFSETS from 320000 ms to check Privacy Policy
//   + Terms of Service and click "Confirm >", followed by A at 358000 ms), then Phase 5 looping
//   CHROME_LEGAL_TERMS_ACCEPT_OFFSETS every 16000 ms from 368000 ms (never presses B).
// - 'chrome-to-ingame': Clean, fast FIRST_VISIBLE_FRAME-anchored Chrome Wasm64 preset driving
//   Start Screen -> 4K Bink2 -> UI TEXT SIZE / BRIGHTNESS / FINISH SETUP? (Phase 1: spaced A every
//   3000 ms from 5000 ms through 145000 ms) -> LEGAL NOTICES "Next >" (Phase 2: 4 rounds of 14 s
//   _menu_cycle from 150000 ms through 206000 ms) -> ROCKSTAR GAMES LEGAL TERMS & POLICIES
//   2-checkbox + "Confirm >" accept (Phase 3: 4 rounds of 16 s CHROME_LEGAL_TERMS_ACCEPT_OFFSETS
//   from 210000 ms through 274000 ms) -> Main Menu / Story Mode selection & gentle in-game
//   movement/camera probes (Phase 4: spaced A every 4000 ms through 340000 ms, then A + left-stick
//   forward / right-stick look sweeps through 420000 ms; bounded, never presses B).
export const INPUT_SCRIPT_PRESETS = Object.freeze({
  'start-screen': Object.freeze({
    name: 'start-screen',
    button: INPUT_BUTTONS.A,
    startMs: 5000,
    holdMs: 100,
    periodMs: 1000,
  }),
  'menu-flow': Object.freeze({
    name: 'menu-flow',
    events: Object.freeze([
      // Phase 1: Start Screen ("PRESS A TO START") — periodic A presses.
      Object.freeze({ atMs: 5000, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 6500, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 8000, holdMs: 100, button: INPUT_BUTTONS.A }),
      // Phase 2: Clean pause across CONTINUE and UI TEXT SIZE transitions.
      Object.freeze({ atMs: 11500, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 14500, holdMs: 100, button: INPUT_BUTTONS.A }),
      // Phase 3: Accept calibration screens with safe spacing (no double-tap, no B).
      Object.freeze({ atMs: 17500, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 20500, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 23500, holdMs: 100, button: INPUT_BUTTONS.A }),
    ]),
    // Phase 4: Main Menu gentle periodic D-pad Down -> Up -> A navigation probes.
    loop: Object.freeze({
      startMs: 27000,
      periodMs: 8000,
      events: Object.freeze([
        Object.freeze({ offsetMs: 0, holdMs: 100, button: INPUT_BUTTONS.DPAD_DOWN }),
        Object.freeze({ offsetMs: 2000, holdMs: 100, button: INPUT_BUTTONS.DPAD_UP }),
        Object.freeze({ offsetMs: 4000, holdMs: 100, button: INPUT_BUTTONS.A }),
      ]),
    }),
  }),
  'calibration-skip': Object.freeze({
    name: 'calibration-skip',
    button: INPUT_BUTTONS.A,
    startMs: 5000,
    holdMs: 100,
    periodMs: 2500,
  }),
  'intro-watch': Object.freeze({
    name: 'intro-watch',
    button: INPUT_BUTTONS.A,
    startMs: 18000,
    holdMs: 100,
    periodMs: 2500,
  }),
  'legal-notices': Object.freeze({
    name: 'legal-notices',
    // Phase 1: After the intro video, press A every 2500 ms from 20000 ms to 62000 ms
    // through Start Screen -> loading -> NAVIGATION EXPERIENCE -> UI TEXT SIZE -> pool/swan.
    events: buildPeriodicButtonEvents(20000, 62000, 2500, 100, INPUT_BUTTONS.A),
    // Phase 2: Loop DPAD_DOWN, DPAD_DOWN, A every 12000 ms from 64000 ms to highlight and
    // confirm "Next >" on LEGAL NOTICES, with interspersed A every 3000 ms (never B or DPAD_UP).
    loop: Object.freeze({
      startMs: 64000,
      periodMs: 12000,
      events: Object.freeze([
        Object.freeze({ offsetMs: 0, holdMs: 100, button: INPUT_BUTTONS.DPAD_DOWN }),
        Object.freeze({ offsetMs: 500, holdMs: 100, button: INPUT_BUTTONS.DPAD_DOWN }),
        Object.freeze({ offsetMs: 1000, holdMs: 100, button: INPUT_BUTTONS.A }),
        Object.freeze({ offsetMs: 4000, holdMs: 100, button: INPUT_BUTTONS.A }),
        Object.freeze({ offsetMs: 7000, holdMs: 100, button: INPUT_BUTTONS.A }),
        Object.freeze({ offsetMs: 10000, holdMs: 100, button: INPUT_BUTTONS.A }),
      ]),
    }),
  }),
  'chrome-menu-flow': Object.freeze({
    name: 'chrome-menu-flow',
    // Phase 1: Wait through real Wasm64 browser 4K Bink2 intro & shader init, then press A
    // every 2500 ms from 165000 ms through 215000 ms.
    events: buildPeriodicButtonEvents(165000, 215000, 2500, 100, INPUT_BUTTONS.A),
    // Phase 2: Loop DPAD_DOWN, DPAD_DOWN, A, A, A every 12000 ms from 218000 ms to advance
    // LEGAL NOTICES ("Next >") and subsequent list screens (never B or DPAD_UP).
    loop: Object.freeze({
      startMs: 218000,
      periodMs: 12000,
      events: Object.freeze([
        Object.freeze({ offsetMs: 0, holdMs: 100, button: INPUT_BUTTONS.DPAD_DOWN }),
        Object.freeze({ offsetMs: 500, holdMs: 100, button: INPUT_BUTTONS.DPAD_DOWN }),
        Object.freeze({ offsetMs: 1200, holdMs: 100, button: INPUT_BUTTONS.A }),
        Object.freeze({ offsetMs: 4500, holdMs: 100, button: INPUT_BUTTONS.A }),
        Object.freeze({ offsetMs: 8000, holdMs: 100, button: INPUT_BUTTONS.A }),
      ]),
    }),
  }),
  'chrome-full-boot': Object.freeze({
    name: 'chrome-full-boot',
    // Phase 1: Wait through the entire 4K Bink2 intro video & Rockstar logo in real Wasm64
    // headless Chrome without skipping it, then press A every 2500 ms from 235000 ms
    // through 295000 ms.
    events: buildPeriodicButtonEvents(235000, 295000, 2500, 100, INPUT_BUTTONS.A),
    // Phase 2: Loop DPAD_DOWN, DPAD_DOWN, A, A, A every 12000 ms from 298000 ms to advance
    // LEGAL NOTICES ("Next >") and subsequent setup screens (never B or DPAD_UP).
    loop: Object.freeze({
      startMs: 298000,
      periodMs: 12000,
      events: Object.freeze([
        Object.freeze({ offsetMs: 0, holdMs: 100, button: INPUT_BUTTONS.DPAD_DOWN }),
        Object.freeze({ offsetMs: 500, holdMs: 100, button: INPUT_BUTTONS.DPAD_DOWN }),
        Object.freeze({ offsetMs: 1200, holdMs: 100, button: INPUT_BUTTONS.A }),
        Object.freeze({ offsetMs: 4500, holdMs: 100, button: INPUT_BUTTONS.A }),
        Object.freeze({ offsetMs: 8000, holdMs: 100, button: INPUT_BUTTONS.A }),
      ]),
    }),
  }),
  'menu-to-ingame': Object.freeze({
    name: 'menu-to-ingame',
    events: Object.freeze([
      // Phase 1: Start Screen ("PRESS A TO START") confirmations.
      Object.freeze({ atMs: 5000, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 7000, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 9000, holdMs: 100, button: INPUT_BUTTONS.A }),
      // Phase 2: Calibration screens (CONTINUE / UI TEXT SIZE / Display) & Legal Notices ("Next >").
      Object.freeze({ atMs: 12000, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 14500, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 17000, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 20000, holdMs: 100, button: INPUT_BUTTONS.DPAD_DOWN }),
      Object.freeze({ atMs: 20500, holdMs: 100, button: INPUT_BUTTONS.DPAD_DOWN }),
      Object.freeze({ atMs: 21200, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 24000, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 26500, holdMs: 100, button: INPUT_BUTTONS.A }),
      // Phase 3: Intro cinematic skip (hold-A skip, MENU skip fallback, confirm A).
      Object.freeze({ atMs: 30000, holdMs: 600, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 32500, holdMs: 150, button: INPUT_BUTTONS.MENU }),
      Object.freeze({ atMs: 34500, holdMs: 100, button: INPUT_BUTTONS.A }),
      // Phase 4: Main Menu Story/Play selection & New Game / Continue confirmations.
      Object.freeze({ atMs: 38000, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 40500, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 43000, holdMs: 100, button: INPUT_BUTTONS.A }),
      Object.freeze({ atMs: 45500, holdMs: 100, button: INPUT_BUTTONS.A }),
      // Phase 5: Brief in-game camera & movement (left stick, right stick, and LT/RT triggers).
      Object.freeze({
        atMs: 50000,
        holdMs: 1500,
        sticks: Object.freeze({ lx: 0, ly: 0.85 }),
      }),
      Object.freeze({
        atMs: 52000,
        holdMs: 1200,
        sticks: Object.freeze({ rx: 0.75, ry: 0.25 }),
      }),
      Object.freeze({
        atMs: 53700,
        holdMs: 1200,
        sticks: Object.freeze({ lx: 0.6, ly: 0.4, rx: -0.75, ry: -0.2 }),
      }),
      Object.freeze({
        atMs: 55500,
        holdMs: 1000,
        sticks: Object.freeze({ rx: 0.35, ry: 0 }),
        triggers: Object.freeze({ lt: 0.85 }),
      }),
      Object.freeze({
        atMs: 57000,
        holdMs: 1000,
        sticks: Object.freeze({ lx: 0, ly: 0.9 }),
        triggers: Object.freeze({ rt: 0.9 }),
      }),
      Object.freeze({
        atMs: 58500,
        holdMs: 800,
        sticks: Object.freeze({ lx: -0.5, ly: 0.5, rx: 0.5, ry: 0 }),
        triggers: Object.freeze({ lt: 0.5, rt: 0.75 }),
      }),
    ]),
  }),
  'chrome-menu-flow-2': Object.freeze({
    name: 'chrome-menu-flow-2',
    events: buildChromeMenuFlow2Events(),
    // Phase 5: Periodic A every 9000 ms from 360000 ms to advance cleanly into Main Menu /
    // In-Game (mirrors PR #1333 menu-flow-2; never presses B or DPAD_UP).
    loop: Object.freeze({
      startMs: 360000,
      periodMs: 9000,
      events: Object.freeze([
        Object.freeze({ offsetMs: 0, holdMs: 100, button: INPUT_BUTTONS.A }),
      ]),
    }),
  }),
  'chrome-menu-flow-4': Object.freeze({
    name: 'chrome-menu-flow-4',
    events: buildChromeMenuFlow4Events(),
    // Phase 5 (W362): Looping 16 s CHROME_LEGAL_TERMS_ACCEPT_OFFSETS (Up x6, scroll, A, Down,
    // scroll, A, Down, A, A; >= 200 ms holds) from 368000 ms after Phase 4 settles (~10 s after
    // 358000 ms; never presses B).
    loop: Object.freeze({
      startMs: 368000,
      periodMs: 16000,
      events: CHROME_LEGAL_TERMS_ACCEPT_OFFSETS,
    }),
  }),
  'chrome-menu-flow-5': Object.freeze({
    name: 'chrome-menu-flow-5',
    events: buildChromeMenuFlow5Events(),
    // Phase 3 (W364): Looping 16 s CHROME_LEGAL_TERMS_ACCEPT_OFFSETS from 85000 ms
    // (Up x6, right-stick scroll, A [Privacy Policy], Down, right-stick scroll,
    // A [Terms of Service], Down, A [Confirm >], A; >= 200 ms holds, never presses B).
    loop: Object.freeze({
      startMs: 85000,
      periodMs: 16000,
      events: CHROME_LEGAL_TERMS_ACCEPT_OFFSETS,
    }),
  }),
  'chrome-to-ingame': Object.freeze({
    name: 'chrome-to-ingame',
    events: buildChromeToIngameEvents(),
  }),
});

export const INPUT_SCRIPT_PRESET_NAMES = Object.freeze(Object.keys(INPUT_SCRIPT_PRESETS));

const INPUT_BUTTON_ENTRIES = Object.freeze(
  Object.entries(INPUT_BUTTONS).map(([name, bit]) => Object.freeze([name, bit]))
);

function formatScriptButtonMask(mask) {
  const normalized = mask >>> 0;
  if (normalized === 0) return null;
  const names = [];
  let matchedBits = 0;
  for (let i = 0; i < INPUT_BUTTON_ENTRIES.length; i++) {
    const [name, bit] = INPUT_BUTTON_ENTRIES[i];
    if ((normalized & bit) !== 0) {
      names.push(name);
      matchedBits = (matchedBits | bit) >>> 0;
    }
  }
  const unknownBits = (normalized & ~matchedBits) >>> 0;
  if (unknownBits !== 0) {
    names.push(`0x${unknownBits.toString(16)}`);
  }
  return names.length > 0 ? names.join('+') : null;
}

const SCRIPT_BUTTON_ALIASES = Object.freeze({
  START: INPUT_BUTTONS.MENU,
  BACK: INPUT_BUTTONS.VIEW,
  SELECT: INPUT_BUTTONS.VIEW,
});

function resolveSingleScriptButton(value) {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new RangeError(`Scripted input button mask must be a finite number, got ${value}`);
    }
    return value >>> 0;
  }
  if (typeof value === 'string') {
    const key = value.trim().toUpperCase().replace(/^BUTTON_/, '');
    if (Object.prototype.hasOwnProperty.call(INPUT_BUTTONS, key)) {
      return INPUT_BUTTONS[key];
    }
    if (Object.prototype.hasOwnProperty.call(SCRIPT_BUTTON_ALIASES, key)) {
      return SCRIPT_BUTTON_ALIASES[key];
    }
    throw new RangeError(`Unknown scripted input button: ${value}`);
  }
  if (Array.isArray(value)) {
    let mask = 0;
    for (let i = 0; i < value.length; i++) {
      mask = (mask | resolveSingleScriptButton(value[i])) >>> 0;
    }
    return mask;
  }
  throw new TypeError(`Invalid scripted input button specifier: ${typeof value}`);
}

function resolveScriptButtonMask(button, buttonsMask) {
  let mask = 0;
  if (buttonsMask !== undefined && buttonsMask !== null) {
    if (typeof buttonsMask !== 'number' || !Number.isFinite(buttonsMask)) {
      throw new RangeError(`buttonsMask must be a finite number, got ${buttonsMask}`);
    }
    mask = (mask | (buttonsMask >>> 0)) >>> 0;
  }
  if (button !== undefined && button !== null) {
    mask = (mask | resolveSingleScriptButton(button)) >>> 0;
  }
  return mask >>> 0;
}

function normalizeScriptSticks(sticks) {
  if (sticks === undefined || sticks === null) return null;
  if (typeof sticks !== 'object') {
    throw new TypeError('Scripted input sticks must be an object');
  }
  const rawLx = sticks.lx !== undefined ? sticks.lx : sticks.leftX;
  const rawLy = sticks.ly !== undefined ? sticks.ly : sticks.leftY;
  const rawRx = sticks.rx !== undefined ? sticks.rx : sticks.rightX;
  const rawRy = sticks.ry !== undefined ? sticks.ry : sticks.rightY;

  const lx = rawLx !== undefined && rawLx !== null ? clamp(rawLx, -1, 1) : null;
  const ly = rawLy !== undefined && rawLy !== null ? clamp(rawLy, -1, 1) : null;
  const rx = rawRx !== undefined && rawRx !== null ? clamp(rawRx, -1, 1) : null;
  const ry = rawRy !== undefined && rawRy !== null ? clamp(rawRy, -1, 1) : null;

  if (lx === null && ly === null && rx === null && ry === null) return null;
  return Object.freeze({ lx, ly, rx, ry });
}

function scriptSticksEqual(a, b) {
  if (a === b) return true;
  const alx = a ? a.lx : 0;
  const aly = a ? a.ly : 0;
  const arx = a ? a.rx : 0;
  const ary = a ? a.ry : 0;
  const blx = b ? b.lx : 0;
  const bly = b ? b.ly : 0;
  const brx = b ? b.rx : 0;
  const bry = b ? b.ry : 0;
  return alx === blx && aly === bly && arx === brx && ary === bry;
}

function normalizeScriptTriggers(triggers) {
  if (triggers === undefined || triggers === null) return null;
  if (typeof triggers !== 'object') {
    throw new TypeError('Scripted input triggers must be an object');
  }
  const rawLt =
    triggers.lt !== undefined
      ? triggers.lt
      : triggers.left !== undefined
        ? triggers.left
        : triggers.leftTrigger;
  const rawRt =
    triggers.rt !== undefined
      ? triggers.rt
      : triggers.right !== undefined
        ? triggers.right
        : triggers.rightTrigger;

  const lt = rawLt !== undefined && rawLt !== null ? clamp(rawLt, 0, 1) : null;
  const rt = rawRt !== undefined && rawRt !== null ? clamp(rawRt, 0, 1) : null;

  if (lt === null && rt === null) return null;
  return Object.freeze({ lt, rt });
}

function scriptTriggersEqual(a, b) {
  if (a === b) return true;
  const alt = a ? a.lt : 0;
  const art = a ? a.rt : 0;
  const blt = b ? b.lt : 0;
  const brt = b ? b.rt : 0;
  return alt === blt && art === brt;
}

function defaultNowMs() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function validateNonNegativeFinite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new RangeError(`${name} must be a finite number >= 0, got ${value}`);
  }
  return value;
}

function validatePositiveFinite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be a finite number > 0, got ${value}`);
  }
  return value;
}

function validateOptionalEndMs(value) {
  if (value === undefined || value === null) return null;
  return validateNonNegativeFinite(value, 'endMs');
}

function validateOptionalStartOffsetMs(value) {
  if (value === undefined || value === null) return 0;
  return validateNonNegativeFinite(value, 'startOffsetMs');
}

function validateOptionalMinHoldMs(value) {
  if (value === undefined || value === null) return 0;
  return validateNonNegativeFinite(value, 'minHoldMs');
}

function validateOptionalTimeoutMs(value, name) {
  if (value === undefined || value === null) return null;
  return validateNonNegativeFinite(value, name);
}

function normalizeStartTrigger(startOn, waitForFirstVisibleFrame) {
  if (startOn === undefined || startOn === null || startOn === '') {
    return waitForFirstVisibleFrame === true
      ? INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME
      : INPUT_SCRIPT_START_TRIGGERS.IMMEDIATE;
  }
  if (typeof startOn !== 'string') {
    throw new RangeError(`startOn must be a string trigger name, got ${typeof startOn}`);
  }
  const normalized = startOn.trim().toLowerCase().replace(/[\s-]+/g, '_');
  if (normalized === 'immediate' || normalized === 'now') {
    return INPUT_SCRIPT_START_TRIGGERS.IMMEDIATE;
  }
  if (normalized === 'first_present' || normalized === 'firstpresent') {
    return INPUT_SCRIPT_START_TRIGGERS.FIRST_PRESENT;
  }
  if (normalized === 'first_visible_frame' || normalized === 'firstvisibleframe') {
    return INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME;
  }
  throw new RangeError(`Unknown scripted input startOn trigger: ${startOn}`);
}

function validateOptionalMaxPresses(value) {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`maxPresses must be an integer >= 0, got ${value}`);
  }
  return value;
}

/**
 * Deterministic scripted input scheduler supporting named presets (e.g. 'start-screen'),
 * repeating press schedules, and explicit sorted event arrays.
 *
 * Provides pure time evaluation via `sampleAt(elapsedMs)` plus a timer-driven
 * `start()` / `stop()` loop with injectable clock & timer hooks for deterministic testing.
 */
const EMPTY_SCRIPT_SAMPLE = Object.freeze({ buttonsMask: 0, activeEventCount: 0 });

export class ScriptedInputPlayer {
  constructor(presetOrSpec, options = {}) {
    const opts = options || {};

    let spec = presetOrSpec;
    let presetName = typeof opts.preset === 'string' ? opts.preset : null;

    if (typeof spec === 'string') {
      if (!Object.prototype.hasOwnProperty.call(INPUT_SCRIPT_PRESETS, spec)) {
        throw new RangeError(`Unknown input script preset: ${spec}`);
      }
      presetName = spec;
      spec = INPUT_SCRIPT_PRESETS[spec];
    } else if (
      spec &&
      typeof spec === 'object' &&
      !Array.isArray(spec) &&
      typeof spec.preset === 'string' &&
      !Array.isArray(spec.events) &&
      spec.loop === undefined &&
      spec.holdMs === undefined &&
      spec.periodMs === undefined
    ) {
      if (!Object.prototype.hasOwnProperty.call(INPUT_SCRIPT_PRESETS, spec.preset)) {
        throw new RangeError(`Unknown input script preset: ${spec.preset}`);
      }
      presetName = spec.preset;
      spec = {
        ...INPUT_SCRIPT_PRESETS[spec.preset],
        ...(spec.endMs !== undefined ? { endMs: spec.endMs } : null),
        ...(spec.maxPresses !== undefined ? { maxPresses: spec.maxPresses } : null),
        ...(spec.startOffsetMs !== undefined ? { startOffsetMs: spec.startOffsetMs } : null),
        ...(spec.minHoldMs !== undefined ? { minHoldMs: spec.minHoldMs } : null),
      };
    }

    if (
      Array.isArray(spec) ||
      (spec &&
        typeof spec === 'object' &&
        (Array.isArray(spec.events) || (spec.loop !== undefined && spec.loop !== null)))
    ) {
      const rawEvents = Array.isArray(spec)
        ? spec
        : Array.isArray(spec.events)
          ? spec.events
          : EMPTY_ARRAY;
      const specObj = Array.isArray(spec) ? null : spec;
      const startOffsetMs = validateOptionalStartOffsetMs(
        opts.startOffsetMs !== undefined ? opts.startOffsetMs : specObj?.startOffsetMs
      );
      const minHoldMs = validateOptionalMinHoldMs(
        opts.minHoldMs !== undefined ? opts.minHoldMs : specObj?.minHoldMs
      );
      const endMs = validateOptionalEndMs(
        opts.endMs !== undefined ? opts.endMs : specObj?.endMs
      );
      const maxPresses = validateOptionalMaxPresses(
        opts.maxPresses !== undefined ? opts.maxPresses : specObj?.maxPresses
      );

      if (!presetName && specObj) {
        if (typeof specObj.preset === 'string') presetName = specObj.preset;
        else if (typeof specObj.name === 'string') presetName = specObj.name;
      }

      const indexed = rawEvents.map((ev, index) => {
        if (!ev || typeof ev !== 'object') {
          throw new TypeError(`Scripted input event at index ${index} must be an object`);
        }
        const atMs =
          validateNonNegativeFinite(
            ev.atMs !== undefined ? ev.atMs : ev.startMs,
            `events[${index}].atMs`
          ) + startOffsetMs;
        const rawHoldMs = validatePositiveFinite(ev.holdMs, `events[${index}].holdMs`);
        const holdMs = minHoldMs > 0 ? Math.max(rawHoldMs, minHoldMs) : rawHoldMs;
        const buttonsMask = resolveScriptButtonMask(ev.button, ev.buttonsMask);
        const sticks = normalizeScriptSticks(ev.sticks);
        const triggers = normalizeScriptTriggers(ev.triggers);
        return { index, atMs, holdMs, buttonsMask, sticks, triggers };
      });

      indexed.sort((a, b) => (a.atMs !== b.atMs ? a.atMs - b.atMs : a.index - b.index));

      const capped = maxPresses !== null ? indexed.slice(0, maxPresses) : indexed;
      const normalized = [];
      for (let i = 0; i < capped.length; i++) {
        const ev = capped[i];
        if (endMs !== null && ev.atMs >= endMs) continue;
        const evEnd = endMs !== null ? Math.min(ev.atMs + ev.holdMs, endMs) : ev.atMs + ev.holdMs;
        if (evEnd <= ev.atMs) continue;
        normalized.push(
          Object.freeze({
            atMs: ev.atMs,
            holdMs: evEnd - ev.atMs,
            endMs: evEnd,
            buttonsMask: ev.buttonsMask,
            sticks: ev.sticks,
            triggers: ev.triggers,
          })
        );
      }

      let normalizedLoop = null;
      if (specObj && specObj.loop !== undefined && specObj.loop !== null) {
        const rawLoop = specObj.loop;
        if (typeof rawLoop !== 'object' || Array.isArray(rawLoop)) {
          throw new TypeError('Scripted input loop must be an object');
        }
        const loopStartMs =
          validateNonNegativeFinite(rawLoop.startMs ?? 0, 'loop.startMs') + startOffsetMs;
        const loopPeriodMs = validatePositiveFinite(rawLoop.periodMs, 'loop.periodMs');
        if (!Array.isArray(rawLoop.events)) {
          throw new TypeError('Scripted input loop.events must be an array');
        }
        const loopIndexed = rawLoop.events.map((ev, index) => {
          if (!ev || typeof ev !== 'object') {
            throw new TypeError(`Scripted input loop event at index ${index} must be an object`);
          }
          const offsetMs = validateNonNegativeFinite(
            ev.offsetMs !== undefined ? ev.offsetMs : 0,
            `loop.events[${index}].offsetMs`
          );
          if (offsetMs >= loopPeriodMs) {
            throw new RangeError(
              `loop.events[${index}].offsetMs (${offsetMs}) must be < loop.periodMs (${loopPeriodMs})`
            );
          }
          const rawHoldMs = validatePositiveFinite(ev.holdMs, `loop.events[${index}].holdMs`);
          const holdMs = minHoldMs > 0 ? Math.max(rawHoldMs, minHoldMs) : rawHoldMs;
          const buttonsMask = resolveScriptButtonMask(ev.button, ev.buttonsMask);
          const sticks = normalizeScriptSticks(ev.sticks);
          const triggers = normalizeScriptTriggers(ev.triggers);
          return { index, offsetMs, holdMs, buttonsMask, sticks, triggers };
        });

        loopIndexed.sort((a, b) =>
          a.offsetMs !== b.offsetMs ? a.offsetMs - b.offsetMs : a.index - b.index
        );

        const maxLoopPresses =
          maxPresses !== null ? Math.max(0, maxPresses - capped.length) : null;

        if (
          loopIndexed.length > 0 &&
          maxLoopPresses !== 0 &&
          (endMs === null || loopStartMs < endMs)
        ) {
          normalizedLoop = Object.freeze({
            startMs: loopStartMs,
            periodMs: loopPeriodMs,
            maxLoopPresses,
            events: Object.freeze(
              loopIndexed.map((ev) =>
                Object.freeze({
                  offsetMs: ev.offsetMs,
                  holdMs: ev.holdMs,
                  buttonsMask: ev.buttonsMask,
                  sticks: ev.sticks,
                  triggers: ev.triggers,
                })
              )
            ),
          });
        }
      }

      this._mode = 'events';
      this._rule = null;
      this._events = Object.freeze(normalized);
      this._loop = normalizedLoop;
      this._endMs = endMs;
      this._maxPresses = maxPresses;
      this.startOffsetMs = startOffsetMs;
      this.minHoldMs = minHoldMs;
    } else if (spec && typeof spec === 'object') {
      const startOffsetMs = validateOptionalStartOffsetMs(
        opts.startOffsetMs !== undefined ? opts.startOffsetMs : spec.startOffsetMs
      );
      const minHoldMs = validateOptionalMinHoldMs(
        opts.minHoldMs !== undefined ? opts.minHoldMs : spec.minHoldMs
      );
      const startMs = validateNonNegativeFinite(spec.startMs ?? 0, 'startMs') + startOffsetMs;
      const rawHoldMs = validatePositiveFinite(spec.holdMs, 'holdMs');
      const holdMs = minHoldMs > 0 ? Math.max(rawHoldMs, minHoldMs) : rawHoldMs;
      const periodMs = validatePositiveFinite(spec.periodMs, 'periodMs');
      const endMs = validateOptionalEndMs(opts.endMs !== undefined ? opts.endMs : spec.endMs);
      const maxPresses = validateOptionalMaxPresses(
        opts.maxPresses !== undefined ? opts.maxPresses : spec.maxPresses
      );
      const buttonsMask = resolveScriptButtonMask(spec.button, spec.buttonsMask);
      const sticks = normalizeScriptSticks(spec.sticks);
      const triggers = normalizeScriptTriggers(spec.triggers);

      if (!presetName) {
        if (typeof spec.preset === 'string') presetName = spec.preset;
        else if (typeof spec.name === 'string') presetName = spec.name;
      }

      this._mode = 'repeating';
      this._rule = Object.freeze({
        startMs,
        holdMs,
        periodMs,
        endMs,
        maxPresses,
        buttonsMask,
        sticks,
        triggers,
      });
      this._events = EMPTY_ARRAY;
      this._loop = null;
      this._endMs = endMs;
      this._maxPresses = maxPresses;
      this.startOffsetMs = startOffsetMs;
      this.minHoldMs = minHoldMs;
    } else {
      throw new TypeError(
        'Input script spec must be a preset name, repeating schedule object, or event array'
      );
    }

    this.preset = presetName;

    this._nowFn = typeof opts.nowFn === 'function' ? opts.nowFn : defaultNowMs;
    this._setTimeoutFn =
      opts.setTimeoutFn !== undefined
        ? opts.setTimeoutFn
        : typeof globalThis !== 'undefined' && typeof globalThis.setTimeout === 'function'
          ? globalThis.setTimeout.bind(globalThis)
          : null;
    this._clearTimeoutFn =
      opts.clearTimeoutFn !== undefined
        ? opts.clearTimeoutFn
        : typeof globalThis !== 'undefined' && typeof globalThis.clearTimeout === 'function'
          ? globalThis.clearTimeout.bind(globalThis)
          : null;
    this._setIntervalFn =
      opts.setIntervalFn !== undefined
        ? opts.setIntervalFn
        : typeof globalThis !== 'undefined' && typeof globalThis.setInterval === 'function'
          ? globalThis.setInterval.bind(globalThis)
          : null;
    this._clearIntervalFn =
      opts.clearIntervalFn !== undefined
        ? opts.clearIntervalFn
        : typeof globalThis !== 'undefined' && typeof globalThis.clearInterval === 'function'
          ? globalThis.clearInterval.bind(globalThis)
          : null;

    const explicitInterval =
      opts.setIntervalFn !== undefined && opts.setTimeoutFn === undefined;
    const hasTickMs = typeof opts.tickMs === 'number' && Number.isFinite(opts.tickMs) && opts.tickMs > 0;
    this._useInterval = Boolean(hasTickMs || explicitInterval);
    this._tickMs = hasTickMs ? opts.tickMs : 16;

    this._onSample = typeof opts.onSample === 'function' ? opts.onSample : null;
    this._onComplete = typeof opts.onComplete === 'function' ? opts.onComplete : null;
    this._bridgeSink =
      typeof opts._bridgeSink === 'function'
        ? opts._bridgeSink
        : opts.bridge && typeof opts.bridge._applyScriptSample === 'function'
          ? (sample) => opts.bridge._applyScriptSample(sample)
          : null;

    this._active = false;
    this._startedAt = 0;
    this._stoppedElapsedMs = 0;
    this._pressCount = 0;
    this._scriptedPressesSent = 0;
    this._lastScriptedButton = null;
    this._currentMask = 0;
    this._currentSticks = null;
    this._currentTriggers = null;
    this._timerId = null;
    this._intervalId = null;
    this._runToken = 0;
    this._reanchoredInitialEventsApplied = false;
  }

  _setBridgeSink(sinkFn) {
    this._bridgeSink = typeof sinkFn === 'function' ? sinkFn : null;
  }

  /**
   * Pure deterministic evaluation at `elapsedMs` (ms since playback start).
   * Returns `{ buttonsMask, activeEventCount }` (plus `sticks` / `triggers` when set by an active event).
   */
  sampleAt(elapsedMs, reuseIdle = false) {
    if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
      return reuseIdle ? EMPTY_SCRIPT_SAMPLE : { buttonsMask: 0, activeEventCount: 0 };
    }

    if (this._mode === 'repeating') {
      const { startMs, holdMs, periodMs, endMs, maxPresses, buttonsMask, sticks, triggers } =
        this._rule;
      if (maxPresses === 0 || elapsedMs < startMs) {
        return reuseIdle ? EMPTY_SCRIPT_SAMPLE : { buttonsMask: 0, activeEventCount: 0 };
      }
      if (endMs !== null && (endMs <= startMs || elapsedMs >= endMs)) {
        return reuseIdle ? EMPTY_SCRIPT_SAMPLE : { buttonsMask: 0, activeEventCount: 0 };
      }

      let maxK = Math.floor((elapsedMs - startMs) / periodMs);
      if (maxPresses !== null) {
        maxK = Math.min(maxK, maxPresses - 1);
      }
      if (endMs !== null) {
        maxK = Math.min(maxK, Math.ceil((endMs - startMs) / periodMs) - 1);
      }
      if (maxK < 0) {
        return reuseIdle ? EMPTY_SCRIPT_SAMPLE : { buttonsMask: 0, activeEventCount: 0 };
      }

      const latestEnd = startMs + maxK * periodMs + holdMs;
      if (elapsedMs >= latestEnd) {
        return reuseIdle ? EMPTY_SCRIPT_SAMPLE : { buttonsMask: 0, activeEventCount: 0 };
      }

      let activeEventCount = 1;
      if (holdMs > periodMs) {
        let lowK = Math.max(0, Math.floor((elapsedMs - startMs - holdMs) / periodMs) + 1);
        while (lowK <= maxK && startMs + lowK * periodMs + holdMs <= elapsedMs) {
          lowK += 1;
        }
        while (lowK > 0 && startMs + (lowK - 1) * periodMs + holdMs > elapsedMs) {
          lowK -= 1;
        }
        activeEventCount = Math.max(1, maxK - lowK + 1);
      }

      const result = { buttonsMask, activeEventCount };
      if (sticks) {
        result.sticks = {
          lx: sticks.lx ?? 0,
          ly: sticks.ly ?? 0,
          rx: sticks.rx ?? 0,
          ry: sticks.ry ?? 0,
        };
      }
      if (triggers) {
        result.triggers = {
          lt: triggers.lt ?? 0,
          rt: triggers.rt ?? 0,
        };
      }
      return result;
    }

    let buttonsMask = 0;
    let activeEventCount = 0;
    let hasSticks = false;
    let lx = 0;
    let ly = 0;
    let rx = 0;
    let ry = 0;
    let hasTriggers = false;
    let lt = 0;
    let rt = 0;

    for (let i = 0; i < this._events.length; i++) {
      const ev = this._events[i];
      if (ev.atMs > elapsedMs) break;
      if (elapsedMs < ev.endMs) {
        buttonsMask = (buttonsMask | ev.buttonsMask) >>> 0;
        activeEventCount += 1;
        if (ev.sticks) {
          hasSticks = true;
          if (ev.sticks.lx !== null) lx = ev.sticks.lx;
          if (ev.sticks.ly !== null) ly = ev.sticks.ly;
          if (ev.sticks.rx !== null) rx = ev.sticks.rx;
          if (ev.sticks.ry !== null) ry = ev.sticks.ry;
        }
        if (ev.triggers) {
          hasTriggers = true;
          if (ev.triggers.lt !== null) lt = ev.triggers.lt;
          if (ev.triggers.rt !== null) rt = ev.triggers.rt;
        }
      }
    }

    if (
      this._loop &&
      elapsedMs >= this._loop.startMs &&
      (this._endMs === null || elapsedMs < this._endMs)
    ) {
      const { startMs: loopStartMs, periodMs, events: loopEvents, maxLoopPresses } = this._loop;
      const loopLen = loopEvents.length;
      for (let j = 0; j < loopLen; j++) {
        const ev = loopEvents[j];
        if (maxLoopPresses !== null && j >= maxLoopPresses) continue;
        const firstAt = loopStartMs + ev.offsetMs;
        if (elapsedMs < firstAt) continue;
        if (this._endMs !== null && firstAt >= this._endMs) continue;

        let maxK = Math.floor((elapsedMs - firstAt) / periodMs);
        if (maxLoopPresses !== null) {
          maxK = Math.min(maxK, Math.floor((maxLoopPresses - 1 - j) / loopLen));
        }
        if (this._endMs !== null) {
          maxK = Math.min(maxK, Math.ceil((this._endMs - firstAt) / periodMs) - 1);
        }
        if (maxK < 0) continue;

        const latestEnd = firstAt + maxK * periodMs + ev.holdMs;
        if (elapsedMs >= latestEnd) continue;

        let countForEv = 1;
        if (ev.holdMs > periodMs) {
          let lowK = Math.max(0, Math.floor((elapsedMs - firstAt - ev.holdMs) / periodMs) + 1);
          while (lowK <= maxK && firstAt + lowK * periodMs + ev.holdMs <= elapsedMs) {
            lowK += 1;
          }
          while (lowK > 0 && firstAt + (lowK - 1) * periodMs + ev.holdMs > elapsedMs) {
            lowK -= 1;
          }
          countForEv = Math.max(1, maxK - lowK + 1);
        }

        buttonsMask = (buttonsMask | ev.buttonsMask) >>> 0;
        activeEventCount += countForEv;
        if (ev.sticks) {
          hasSticks = true;
          if (ev.sticks.lx !== null) lx = ev.sticks.lx;
          if (ev.sticks.ly !== null) ly = ev.sticks.ly;
          if (ev.sticks.rx !== null) rx = ev.sticks.rx;
          if (ev.sticks.ry !== null) ry = ev.sticks.ry;
        }
        if (ev.triggers) {
          hasTriggers = true;
          if (ev.triggers.lt !== null) lt = ev.triggers.lt;
          if (ev.triggers.rt !== null) rt = ev.triggers.rt;
        }
      }
    }

    if (reuseIdle && buttonsMask === 0 && activeEventCount === 0 && !hasSticks && !hasTriggers) {
      return EMPTY_SCRIPT_SAMPLE;
    }

    const result = { buttonsMask, activeEventCount };
    if (hasSticks) {
      result.sticks = { lx, ly, rx, ry };
    }
    if (hasTriggers) {
      result.triggers = { lt, rt };
    }
    return result;
  }

  /**
   * Pure count of scheduled presses that have started at or before `elapsedMs`.
   */
  pressCountAt(elapsedMs) {
    if (!Number.isFinite(elapsedMs) || elapsedMs < 0) return 0;

    if (this._mode === 'repeating') {
      const { startMs, periodMs, endMs, maxPresses } = this._rule;
      if (maxPresses === 0 || elapsedMs < startMs) return 0;
      if (endMs !== null && endMs <= startMs) return 0;

      let maxK = Math.floor((elapsedMs - startMs) / periodMs);
      if (maxPresses !== null) {
        maxK = Math.min(maxK, maxPresses - 1);
      }
      if (endMs !== null) {
        maxK = Math.min(maxK, Math.ceil((endMs - startMs) / periodMs) - 1);
      }
      return maxK >= 0 ? maxK + 1 : 0;
    }

    let count = 0;
    for (let i = 0; i < this._events.length; i++) {
      if (this._events[i].atMs <= elapsedMs) {
        count += 1;
      } else {
        break;
      }
    }

    if (this._loop && elapsedMs >= this._loop.startMs) {
      const { startMs: loopStartMs, periodMs, events: loopEvents, maxLoopPresses } = this._loop;
      const loopLen = loopEvents.length;
      for (let j = 0; j < loopLen; j++) {
        const ev = loopEvents[j];
        if (maxLoopPresses !== null && j >= maxLoopPresses) continue;
        const firstAt = loopStartMs + ev.offsetMs;
        if (elapsedMs < firstAt) continue;
        if (this._endMs !== null && firstAt >= this._endMs) continue;

        let maxK = Math.floor((elapsedMs - firstAt) / periodMs);
        if (maxLoopPresses !== null) {
          maxK = Math.min(maxK, Math.floor((maxLoopPresses - 1 - j) / loopLen));
        }
        if (this._endMs !== null) {
          maxK = Math.min(maxK, Math.ceil((this._endMs - firstAt) / periodMs) - 1);
        }
        if (maxK >= 0) {
          count += maxK + 1;
        }
      }
    }

    return count;
  }

  /**
   * Pure next transition timestamp strictly greater than `elapsedMs`, or `null` if none remain.
   */
  nextTransitionMs(elapsedMs) {
    const t = Number.isFinite(elapsedMs) && elapsedMs >= 0 ? elapsedMs : 0;

    if (this._mode === 'repeating') {
      const { startMs, holdMs, periodMs, endMs, maxPresses } = this._rule;
      if (maxPresses === 0) return null;
      if (endMs !== null && (endMs <= startMs || t >= endMs)) return null;
      if (t < startMs) return startMs;

      let maxK = Infinity;
      if (maxPresses !== null) {
        maxK = Math.min(maxK, maxPresses - 1);
      }
      if (endMs !== null) {
        maxK = Math.min(maxK, Math.ceil((endMs - startMs) / periodMs) - 1);
      }
      if (maxK < 0) return null;

      const k = Math.min(Math.floor((t - startMs) / periodMs), maxK);
      if (holdMs < periodMs) {
        const rawEnd = startMs + k * periodMs + holdMs;
        const pressEnd = endMs !== null ? Math.min(rawEnd, endMs) : rawEnd;
        if (t < pressEnd) return pressEnd;
        if (k + 1 <= maxK) return startMs + (k + 1) * periodMs;
        return null;
      }

      if (!Number.isFinite(maxK)) {
        return endMs !== null && t < endMs ? endMs : null;
      }
      const rawFinalEnd = startMs + maxK * periodMs + holdMs;
      const finalEnd = endMs !== null ? Math.min(rawFinalEnd, endMs) : rawFinalEnd;
      return t < finalEnd ? finalEnd : null;
    }

    let nextMs = Infinity;
    for (let i = 0; i < this._events.length; i++) {
      const ev = this._events[i];
      if (ev.atMs > t && ev.atMs < nextMs) {
        nextMs = ev.atMs;
      }
      if (ev.endMs > t && ev.endMs < nextMs) {
        nextMs = ev.endMs;
      }
    }

    if (this._loop && (this._endMs === null || t < this._endMs)) {
      const { startMs: loopStartMs, periodMs, events: loopEvents, maxLoopPresses } = this._loop;
      const loopLen = loopEvents.length;
      for (let j = 0; j < loopLen; j++) {
        const ev = loopEvents[j];
        if (maxLoopPresses !== null && j >= maxLoopPresses) continue;
        const firstAt = loopStartMs + ev.offsetMs;
        if (this._endMs !== null && firstAt >= this._endMs) continue;

        let maxK = Infinity;
        if (maxLoopPresses !== null) {
          maxK = Math.min(maxK, Math.floor((maxLoopPresses - 1 - j) / loopLen));
        }
        if (this._endMs !== null) {
          maxK = Math.min(maxK, Math.ceil((this._endMs - firstAt) / periodMs) - 1);
        }
        if (maxK < 0) continue;

        if (t < firstAt) {
          if (firstAt < nextMs) nextMs = firstAt;
          continue;
        }

        const k = Math.min(Math.floor((t - firstAt) / periodMs), maxK);
        if (ev.holdMs < periodMs) {
          const rawEnd = firstAt + k * periodMs + ev.holdMs;
          const evEnd = this._endMs !== null ? Math.min(rawEnd, this._endMs) : rawEnd;
          if (t < evEnd && evEnd < nextMs) {
            nextMs = evEnd;
          }
          if (k + 1 <= maxK) {
            const nextStart = firstAt + (k + 1) * periodMs;
            if (nextStart > t && nextStart < nextMs) {
              nextMs = nextStart;
            }
          }
        } else {
          if (!Number.isFinite(maxK)) {
            if (this._endMs !== null && t < this._endMs && this._endMs < nextMs) {
              nextMs = this._endMs;
            }
          } else {
            const rawFinalEnd = firstAt + maxK * periodMs + ev.holdMs;
            const finalEnd =
              this._endMs !== null ? Math.min(rawFinalEnd, this._endMs) : rawFinalEnd;
            if (t < finalEnd && finalEnd < nextMs) {
              nextMs = finalEnd;
            }
          }
        }
      }
    }

    return Number.isFinite(nextMs) ? nextMs : null;
  }

  _clearTimers() {
    if (this._timerId !== null && typeof this._clearTimeoutFn === 'function') {
      this._clearTimeoutFn(this._timerId);
    }
    this._timerId = null;

    if (this._intervalId !== null && typeof this._clearIntervalFn === 'function') {
      this._clearIntervalFn(this._intervalId);
    }
    this._intervalId = null;
  }

  _emitSample(sample) {
    if (this._bridgeSink) {
      this._bridgeSink(sample);
    }
    if (this._onSample) {
      this._onSample(sample, this.getState());
    }
  }

  _tick(expectedToken, elapsedOverride) {
    if (!this._active || expectedToken !== this._runToken) return;

    const elapsedMs =
      elapsedOverride !== undefined
        ? Math.max(0, elapsedOverride)
        : Math.max(0, this._nowFn() - this._startedAt);

    const prevMask = this._currentMask;
    const sample = this.sampleAt(elapsedMs, true);
    const nextMask = sample.buttonsMask >>> 0;
    const nextSticks = sample.sticks || null;
    const nextTriggers = sample.triggers || null;
    const changed =
      nextMask !== prevMask ||
      !scriptSticksEqual(nextSticks, this._currentSticks) ||
      !scriptTriggersEqual(nextTriggers, this._currentTriggers);

    if (changed) {
      const prevPressCount = this._pressCount;
      const newPressCount = Math.max(prevPressCount, this.pressCountAt(elapsedMs));
      this._pressCount = newPressCount;
      this._currentMask = nextMask;
      this._currentSticks = nextSticks;
      this._currentTriggers = nextTriggers;
      if (nextMask !== 0) {
        const newlyPressed = (nextMask & ~prevMask) >>> 0;
        if (newlyPressed !== 0) {
          const addedPresses = Math.max(1, newPressCount - prevPressCount);
          this._scriptedPressesSent += addedPresses;
          this._lastScriptedButton = formatScriptButtonMask(newlyPressed);
        } else if (newPressCount > prevPressCount) {
          this._scriptedPressesSent += newPressCount - prevPressCount;
          this._lastScriptedButton = formatScriptButtonMask(nextMask);
        }
      }
      this._emitSample(sample);
      if (!this._active || expectedToken !== this._runToken) return;
    }

    const nextMs = this.nextTransitionMs(elapsedMs);
    if (nextMs === null && sample.activeEventCount === 0) {
      this._pressCount = Math.max(this._pressCount, this.pressCountAt(elapsedMs));
      this._clearTimers();
      this._active = false;
      this._stoppedElapsedMs = elapsedMs;
      if (this._onComplete) {
        this._onComplete(this.getState());
      }
      return;
    }

    if (!this._useInterval && nextMs !== null && typeof this._setTimeoutFn === 'function') {
      if (this._timerId !== null && typeof this._clearTimeoutFn === 'function') {
        this._clearTimeoutFn(this._timerId);
        this._timerId = null;
      }
      const delay = Math.max(0, Math.ceil(nextMs - elapsedMs));
      this._timerId = this._setTimeoutFn(() => {
        this._timerId = null;
        this._tick(expectedToken);
      }, delay);
    }
  }

  /**
   * Starts timer-driven playback from `elapsedMs = 0`.
   */
  start() {
    const hadHeldState =
      this._currentMask !== 0 ||
      this._currentSticks !== null ||
      this._currentTriggers !== null;
    this._clearTimers();
    this._runToken += 1;
    const token = this._runToken;

    this._startedAt = this._nowFn();
    this._stoppedElapsedMs = 0;
    this._pressCount = 0;
    this._scriptedPressesSent = 0;
    this._lastScriptedButton = null;
    this._currentMask = 0;
    this._currentSticks = null;
    this._currentTriggers = null;
    this._active = true;

    this._tick(token, 0);

    if (
      hadHeldState &&
      this._currentMask === 0 &&
      this._currentSticks === null &&
      this._currentTriggers === null &&
      token === this._runToken
    ) {
      this._emitSample({ buttonsMask: 0, activeEventCount: 0 });
    }

    if (
      this._active &&
      token === this._runToken &&
      this._useInterval &&
      typeof this._setIntervalFn === 'function'
    ) {
      this._intervalId = this._setIntervalFn(() => {
        this._tick(token);
      }, this._tickMs);
    }

    return this;
  }

  /**
   * Re-anchors (resets) playback to `elapsedMs = 0` at the current clock time (`nowFn()`),
   * e.g. when `FIRST_VISIBLE_FRAME` arrives after a slow black asset-loading screen.
   * Optionally prepends non-overlapping `initialEvents` (such as an immediate 220 ms `A`
   * press within 500 ms so the Start Screen 3.2 s idle watchdog never fires) while keeping
   * subsequent script phases synchronized from the new anchor.
   */
  reanchor(options = {}) {
    const opts = options || {};
    const initialEvents = Array.isArray(opts.initialEvents) ? opts.initialEvents : null;
    const preservePressCount = opts.preservePressCount !== false;

    if (
      this._mode === 'events' &&
      initialEvents &&
      initialEvents.length > 0 &&
      !this._reanchoredInitialEventsApplied &&
      this._maxPresses !== 0
    ) {
      const merged = this._events.slice();
      const added = [];
      for (let i = 0; i < initialEvents.length; i++) {
        const raw = initialEvents[i];
        if (!raw || typeof raw !== 'object') continue;
        const atMs = validateNonNegativeFinite(
          raw.atMs !== undefined ? raw.atMs : raw.startMs,
          `initialEvents[${i}].atMs`
        );
        if (this._endMs !== null && atMs >= this._endMs) continue;
        const rawHoldMs = validatePositiveFinite(raw.holdMs, `initialEvents[${i}].holdMs`);
        const holdMs = this.minHoldMs > 0 ? Math.max(rawHoldMs, this.minHoldMs) : rawHoldMs;
        const evEnd = this._endMs !== null ? Math.min(atMs + holdMs, this._endMs) : atMs + holdMs;
        if (evEnd <= atMs) continue;

        let overlaps = false;
        for (let j = 0; j < merged.length; j++) {
          const existing = merged[j];
          if (
            atMs < existing.endMs + FIRST_VISIBLE_FRAME_MIN_GAP_MS &&
            evEnd + FIRST_VISIBLE_FRAME_MIN_GAP_MS > existing.atMs
          ) {
            overlaps = true;
            break;
          }
        }
        if (overlaps) continue;

        const normalizedEv = Object.freeze({
          atMs,
          holdMs: evEnd - atMs,
          endMs: evEnd,
          buttonsMask: resolveScriptButtonMask(raw.button, raw.buttonsMask),
          sticks: normalizeScriptSticks(raw.sticks),
          triggers: normalizeScriptTriggers(raw.triggers),
        });
        merged.push(normalizedEv);
        added.push(normalizedEv);
      }
      if (added.length > 0) {
        merged.sort((a, b) => a.atMs - b.atMs);
        const finalEvents =
          this._maxPresses !== null ? merged.slice(0, this._maxPresses) : merged;
        this._events = Object.freeze(finalEvents);
      }
      this._reanchoredInitialEventsApplied = true;
    }

    const hadHeldState =
      this._currentMask !== 0 ||
      this._currentSticks !== null ||
      this._currentTriggers !== null;
    this._clearTimers();
    this._runToken += 1;
    const token = this._runToken;

    this._startedAt = this._nowFn();
    this._stoppedElapsedMs = 0;
    this._pressCount = 0;
    if (!preservePressCount) {
      this._scriptedPressesSent = 0;
      this._lastScriptedButton = null;
    }
    this._currentMask = 0;
    this._currentSticks = null;
    this._currentTriggers = null;
    this._active = true;

    this._tick(token, 0);

    if (
      hadHeldState &&
      this._currentMask === 0 &&
      this._currentSticks === null &&
      this._currentTriggers === null &&
      token === this._runToken
    ) {
      this._emitSample({ buttonsMask: 0, activeEventCount: 0 });
    }

    if (
      this._active &&
      token === this._runToken &&
      this._useInterval &&
      typeof this._setIntervalFn === 'function'
    ) {
      this._intervalId = this._setIntervalFn(() => {
        this._tick(token);
      }, this._tickMs);
    }

    return this;
  }

  /**
   * Manually advances playback to `elapsedMs` (or `nowFn() - startedAt` when omitted).
   */
  step(elapsedMs) {
    if (!this._active) return this.sampleAt(elapsedMs ?? this._stoppedElapsedMs);
    this._tick(this._runToken, elapsedMs);
    return {
      buttonsMask: this._currentMask,
      activeEventCount: this.sampleAt(
        elapsedMs !== undefined ? elapsedMs : Math.max(0, this._nowFn() - this._startedAt)
      ).activeEventCount,
      ...(this._currentSticks ? { sticks: { ...this._currentSticks } } : null),
      ...(this._currentTriggers ? { triggers: { ...this._currentTriggers } } : null),
    };
  }

  _stopWithoutSink() {
    this._runToken += 1;
    this._clearTimers();
    if (this._active) {
      this._stoppedElapsedMs = Math.max(0, this._nowFn() - this._startedAt);
      this._active = false;
    }
    this._currentMask = 0;
    this._currentSticks = null;
    this._currentTriggers = null;
  }

  /**
   * Stops playback, cancels timers, and clears any held scripted buttons immediately.
   */
  stop() {
    const hadHeldState =
      this._currentMask !== 0 ||
      this._currentSticks !== null ||
      this._currentTriggers !== null;
    this._stopWithoutSink();
    if (hadHeldState) {
      this._emitSample({ buttonsMask: 0, activeEventCount: 0 });
    }
    return this;
  }

  getState() {
    const elapsedMs = this._active
      ? Math.max(0, this._nowFn() - this._startedAt)
      : this._stoppedElapsedMs;
    const pressCount = this._active
      ? Math.max(this._pressCount, this.pressCountAt(elapsedMs))
      : this._pressCount;
    return {
      active: this._active,
      preset: this.preset,
      activePreset: this._active ? this.preset : null,
      elapsedMs,
      elapsedScriptMs: elapsedMs,
      pressCount,
      scriptedPressesSent: this._scriptedPressesSent,
      lastScriptedButton: this._lastScriptedButton,
      currentMask: this._currentMask,
    };
  }
}

export class InputBridge {
  constructor(canvasOrOptions = null, options = {}) {
    let canvas = canvasOrOptions;
    let opts = options;

    // Support both `new InputBridge(canvas, options)` and `new InputBridge({ canvas, ... })`.
    if (
      canvasOrOptions !== null &&
      typeof canvasOrOptions === 'object' &&
      typeof canvasOrOptions.addEventListener !== 'function' &&
      arguments.length === 1
    ) {
      opts = canvasOrOptions;
      canvas = opts.canvas ?? null;
    }

    opts = opts || {};

    this._canvas = canvas || null;
    this._window =
      opts.windowObj !== undefined
        ? opts.windowObj
        : typeof window !== 'undefined'
          ? window
          : null;
    this._document =
      opts.documentObj !== undefined
        ? opts.documentObj
        : typeof document !== 'undefined'
          ? document
          : null;
    this._navigator =
      opts.navigatorObj !== undefined
        ? opts.navigatorObj
        : typeof navigator !== 'undefined'
          ? navigator
          : null;

    const SabCtor =
      opts.sharedArrayBufferCtor !== undefined
        ? opts.sharedArrayBufferCtor
        : typeof SharedArrayBuffer === 'function'
          ? SharedArrayBuffer
          : null;

    if (opts.sab) {
      const isSab = Boolean(SabCtor && opts.sab instanceof SabCtor);
      const isAb = opts.sab instanceof ArrayBuffer;
      if ((!isSab && !isAb) || opts.sab.byteLength < INPUT_STATE_BYTES) {
        throw new RangeError(
          `sab must be a SharedArrayBuffer or ArrayBuffer of at least ${INPUT_STATE_BYTES} bytes`
        );
      }
      this.sab = opts.sab;
      this.isSharedBuffer = isSab;
    } else if (SabCtor) {
      try {
        this.sab = new SabCtor(INPUT_STATE_BYTES);
        this.isSharedBuffer = true;
      } catch {
        this.sab = new ArrayBuffer(INPUT_STATE_BYTES);
        this.isSharedBuffer = false;
      }
    } else {
      this.sab = new ArrayBuffer(INPUT_STATE_BYTES);
      this.isSharedBuffer = false;
    }

    this.i32 = new Int32Array(this.sab, 0, INPUT_STATE_WORDS);
    this._sabI32 = this.i32;
    this.u32 = new Uint32Array(this.sab, 0, INPUT_STATE_WORDS);
    this.f32 = new Float32Array(this.sab, 0, INPUT_STATE_WORDS);
    this.view = new DataView(this.sab, 0, INPUT_STATE_BYTES);

    let atomicsImpl =
      opts.atomicsObj !== undefined
        ? opts.atomicsObj
        : typeof Atomics === 'object' && Atomics !== null
          ? Atomics
          : null;
    if (atomicsImpl) {
      if (
        typeof atomicsImpl.load !== 'function' ||
        typeof atomicsImpl.store !== 'function' ||
        typeof atomicsImpl.add !== 'function'
      ) {
        atomicsImpl = null;
      } else if (!this.isSharedBuffer) {
        // Probe once: older engines throw TypeError for Atomics on non-shared ArrayBuffers.
        try {
          atomicsImpl.load(this.u32, INPUT_WORD_INDICES.SEQ);
        } catch {
          atomicsImpl = null;
        }
      }
    }
    this._atomics = atomicsImpl;
    this.usesAtomics = Boolean(this._atomics);

    this._keyBindings = opts.keyBindings
      ? { ...DEFAULT_KEY_BINDINGS, ...opts.keyBindings }
      : { ...DEFAULT_KEY_BINDINGS };

    this._stickDeadzone = opts.stickDeadzone ?? 0.15;
    this._leftStickDeadzone = opts.leftStickDeadzone ?? this._stickDeadzone;
    this._rightStickDeadzone = opts.rightStickDeadzone ?? this._stickDeadzone;
    this._stickOuterDeadzone = opts.stickOuterDeadzone ?? 0.98;
    this._triggerDeadzone = opts.triggerDeadzone ?? 0.05;
    this._triggerOuterDeadzone = opts.triggerOuterDeadzone ?? 0.98;
    this._invertGamepadY = opts.invertGamepadY ?? true;
    this._gamepadIndex = opts.gamepadIndex ?? null;
    this._allowUnlockedMouseButtons = Boolean(opts.allowUnlockedMouseButtons);
    this._autoStartOnGamepad = opts.autoStart !== false;
    this._onPointerLockChangeCallback =
      typeof opts.onPointerLockChange === 'function' ? opts.onPointerLockChange : null;

    // Input hardening, hot-plug & zero-alloc snapshot telemetry
    this._blurResets = 0;
    this._stuckResetCount = 0;
    this._pollsCount = 0;
    this._writesCount = 0;
    this._notifyCount = 0;
    this._scriptedEventsDispatched = 0;
    this._gamepadConnects = 0;
    this._gamepadDisconnects = 0;
    this._activeGamepadIndex = null;
    this._hasConnectedGamepad = false;
    this._hasNonZeroGpState = false;
    this._hasCustomNowFn = typeof opts.nowFn === 'function';
    this._nowFn = this._hasCustomNowFn ? opts.nowFn : defaultNowMs;
    this._idleGamepadPollIntervalMs =
      typeof opts.idleGamepadPollIntervalMs === 'number' &&
      Number.isFinite(opts.idleGamepadPollIntervalMs) &&
      opts.idleGamepadPollIntervalMs >= 0
        ? opts.idleGamepadPollIntervalMs
        : IDLE_GAMEPAD_POLL_INTERVAL_MS;
    this._lastIdleGamepadPollMs = null;
    this._idleGamepadPollsSkipped = 0;

    // Input source state
    this._pressedKeys = new Set();
    this._keyButtons = 0;
    this._keyLx = 0;
    this._keyLy = 0;

    this._mouseButtonsMask = 0; // bitmask of raw DOM mouse buttons currently held
    this._mouseButtonBits = 0;  // mapped INPUT_BUTTONS bits from mouse
    this._mouseLt = 0;
    this._mouseRt = 0;
    this._wheelDelta = 0;

    this._gpButtons = 0;
    this._gpLx = 0;
    this._gpLy = 0;
    this._gpRx = 0;
    this._gpRy = 0;
    this._gpLt = 0;
    this._gpRt = 0;

    this._scriptPlayer = null;
    this._pendingScript = null;
    this._scriptStartedOn = null;
    this._scriptManuallyStopped = false;
    this._scriptReanchorOnFirstVisibleFrame = false;
    this._scriptReanchoredCount = 0;
    this._firstPresentNotified = false;
    this._firstVisibleFrameNotified = false;
    this._scriptButtons = 0;
    this._scriptLx = 0;
    this._scriptLy = 0;
    this._scriptRx = 0;
    this._scriptRy = 0;
    this._scriptLt = 0;
    this._scriptRt = 0;

    // Pre-allocated scratch objects for zero-allocation 60/120Hz polling, snapshots & telemetry.
    this._leftStickScratch = { x: 0, y: 0 };
    this._rightStickScratch = { x: 0, y: 0 };
    this._selectedGamepadScratch = { pad: null, index: -1 };
    this._snapshotScratch = {
      seq: 0,
      buttons: 0,
      mouse_dx: 0,
      mouse_dy: 0,
      lx: 0,
      ly: 0,
      rx: 0,
      ry: 0,
      lt: 0,
      rt: 0,
      pointerLocked: false,
      wheelDelta: 0,
    };
    this._telemetryScratch = {
      pollsCount: 0,
      writesCount: 0,
      stuckResetCount: 0,
      activePreset: null,
      scriptedEventsDispatched: 0,
      seq: 0,
      buttons: 0,
      pointerLocked: false,
      activeGamepadIndex: null,
    };

    // Last committed merged state (stored as IEEE-754 int32 bits for fast change detection).
    this.buttons = 0;
    this._lastLx = 0;
    this._lastLy = 0;
    this._lastRx = 0;
    this._lastRy = 0;
    this._lastLt = 0;
    this._lastRt = 0;
    this._lastLxBits = 0;
    this._lastLyBits = 0;
    this._lastRxBits = 0;
    this._lastRyBits = 0;
    this._lastLtBits = 0;
    this._lastRtBits = 0;

    this.pointerLocked = Boolean(
      this._canvas && this._document && this._document.pointerLockElement === this._canvas
    );
    this._rafId = null;
    this._manualStart = false;
    this.running = false;
    this.disposed = false;

    this._boundTick = (rafTimestampMs) => {
      if (!this.running || this.disposed) return;
      if (!this._hasConnectedGamepad) {
        const lastIdle = this._lastIdleGamepadPollMs;
        const intervalMs = this._idleGamepadPollIntervalMs;
        const nowMs =
          !this._hasCustomNowFn &&
          typeof rafTimestampMs === 'number' &&
          Number.isFinite(rafTimestampMs)
            ? rafTimestampMs
            : this._nowFn();
        if (lastIdle !== null && intervalMs > 0) {
          const elapsed = nowMs - lastIdle;
          if (elapsed >= 0 && elapsed < intervalMs) {
            this._idleGamepadPollsSkipped += 1;
            const win = this._window;
            if (win && typeof win.requestAnimationFrame === 'function') {
              this._rafId = win.requestAnimationFrame(this._boundTick);
            }
            return;
          }
        }
        this.pollGamepads();
        this._lastIdleGamepadPollMs = this._hasConnectedGamepad ? null : nowMs;
      } else {
        this.pollGamepads();
        if (!this._hasConnectedGamepad) {
          this._lastIdleGamepadPollMs =
            !this._hasCustomNowFn &&
            typeof rafTimestampMs === 'number' &&
            Number.isFinite(rafTimestampMs)
              ? rafTimestampMs
              : this._nowFn();
        }
      }
      const win = this._window;
      if (this.running && !this.disposed && win && typeof win.requestAnimationFrame === 'function') {
        this._rafId = win.requestAnimationFrame(this._boundTick);
      }
    };

    this._listeners = [];
    this._bindEvents();

    if (opts.autoStart === true) {
      this.start();
    }
  }

  _atomicAddU32(index, value) {
    if (this._atomics) {
      return this._atomics.add(this.u32, index, value);
    }
    const prev = this.u32[index];
    this.u32[index] = (prev + value) >>> 0;
    return prev;
  }

  _atomicStoreU32(index, value) {
    if (this._atomics) {
      return this._atomics.store(this.u32, index, value);
    }
    const next = value >>> 0;
    this.u32[index] = next;
    return next;
  }

  _atomicLoadU32(index) {
    if (this._atomics) {
      return this._atomics.load(this.u32, index);
    }
    return this.u32[index];
  }

  _atomicAddI32(index, value) {
    if (this._atomics) {
      return this._atomics.add(this.i32, index, value);
    }
    const prev = this.i32[index];
    this.i32[index] = (prev + value) | 0;
    return prev;
  }

  _atomicStoreI32(index, value) {
    if (this._atomics) {
      return this._atomics.store(this.i32, index, value);
    }
    const next = value | 0;
    this.i32[index] = next;
    return next;
  }

  _atomicLoadI32(index) {
    if (this._atomics) {
      return this._atomics.load(this.i32, index);
    }
    return this.i32[index];
  }

  _notifySeq() {
    if (this._atomics && this.isSharedBuffer && typeof this._atomics.notify === 'function') {
      try {
        this._atomics.notify(this._sabI32, INPUT_WORD_INDICES.SEQ, 1);
        this._notifyCount += 1;
      } catch {
        // Ignore host environments where Atomics.notify is unavailable/restricted.
      }
    }
  }

  _shouldIgnoreKeydown(e, code) {
    if (this.isPointerLocked()) return false;
    // Do not hijack browser shortcuts (Ctrl+R, Ctrl+W, Alt+Tab, Cmd+*, etc.) when unlocked.
    if (e.metaKey || e.altKey) return true;
    if (e.ctrlKey && code !== 'ControlLeft' && code !== 'ControlRight') return true;
    // Do not intercept typing in form controls or contentEditable surfaces when unlocked.
    if (isEditableElement(e.target) || isEditableElement(this._document?.activeElement)) {
      return true;
    }
    // Do not intercept keys while the document itself is unfocused.
    if (this._document && typeof this._document.hasFocus === 'function') {
      try {
        if (!this._document.hasFocus()) return true;
      } catch {
        // Ignore host errors from non-standard document mocks.
      }
    }
    return false;
  }

  _addListener(target, event, handler, options) {
    if (!target || typeof target.addEventListener !== 'function') return;
    target.addEventListener(event, handler, options);
    this._listeners.push({ target, event, handler, options });
  }

  _bindEvents() {
    this._addListener(this._canvas, 'click', () => {
      if (this.disposed || !this._canvas) return;
      if (!this.isPointerLocked() && typeof this._canvas.requestPointerLock === 'function') {
        try {
          const result = this._canvas.requestPointerLock();
          if (result && typeof result.catch === 'function') {
            result.catch(() => {});
          }
        } catch {
          // Pointer lock may reject when invoked outside a user gesture.
        }
      }
    });

    this._addListener(this._canvas, 'contextmenu', (e) => {
      if (this.isPointerLocked() && typeof e?.preventDefault === 'function') {
        e.preventDefault();
      }
    });

    this._addListener(this._document, 'pointerlockchange', () => {
      const nowLocked = Boolean(
        this._canvas && this._document && this._document.pointerLockElement === this._canvas
      );
      const changed = nowLocked !== this.pointerLocked;
      this.pointerLocked = nowLocked;
      if (!nowLocked && this._mouseButtonsMask !== 0) {
        this._clearMouseButtons();
        this._commitMergedState(true);
      }
      if (changed && this._onPointerLockChangeCallback) {
        this._onPointerLockChangeCallback(nowLocked);
      }
    });

    this._addListener(this._window, 'mousemove', (e) => {
      if (this.disposed || !this.isPointerLocked()) return;
      const dx = (e?.movementX ?? 0) | 0;
      const dy = (e?.movementY ?? 0) | 0;
      if (dx === 0 && dy === 0) return;

      // Even/odd seqlock framing around atomic mouse delta accumulation.
      // Matches Atomics.exchange(0) in runtime/os/game_input.cpp::consume_mouse_delta().
      this._atomicAddU32(INPUT_WORD_INDICES.SEQ, 1);
      if (dx !== 0) this._atomicAddI32(INPUT_WORD_INDICES.MOUSE_DX, dx);
      if (dy !== 0) this._atomicAddI32(INPUT_WORD_INDICES.MOUSE_DY, dy);
      this._atomicAddU32(INPUT_WORD_INDICES.SEQ, 1);
      this._notifySeq();
      this._writesCount += 1;
    });

    this._addListener(this._window, 'mousedown', (e) => {
      if (this.disposed) return;
      if (!this.isPointerLocked() && !this._allowUnlockedMouseButtons) return;
      const btn = e?.button;
      if (typeof btn !== 'number' || btn < 0 || btn > 4) return;
      const bit = 1 << btn;
      if ((this._mouseButtonsMask & bit) !== 0) return;
      this._mouseButtonsMask |= bit;
      this._recomputeMouseButtons();
      this._commitMergedState(false);
    });

    this._addListener(this._window, 'mouseup', (e) => {
      if (this.disposed) return;
      const btn = e?.button;
      if (typeof btn !== 'number' || btn < 0 || btn > 4) return;
      const bit = 1 << btn;
      if ((this._mouseButtonsMask & bit) === 0) return;
      this._mouseButtonsMask &= ~bit;
      this._recomputeMouseButtons();
      this._commitMergedState(false);
    });

    this._addListener(this._window, 'wheel', (e) => {
      if (this.disposed || !this.isPointerLocked()) return;
      const deltaY = Number.isFinite(e?.deltaY) ? e.deltaY : 0;
      if (deltaY !== 0) {
        this._wheelDelta += deltaY;
      }
    });

    this._addListener(this._window, 'keydown', (e) => {
      if (this.disposed || !e || !e.code) return;
      const code = e.code;
      const isWasd = code === 'KeyW' || code === 'KeyS' || code === 'KeyA' || code === 'KeyD';
      const binding = this._keyBindings[code] || 0;
      if (!binding && !isWasd) return;
      if (this._shouldIgnoreKeydown(e, code)) return;

      if (
        this.isPointerLocked() &&
        PREVENT_DEFAULT_KEYS.has(code) &&
        typeof e.preventDefault === 'function'
      ) {
        e.preventDefault();
      }

      if (e.repeat && this._pressedKeys.has(code)) return;
      if (this._pressedKeys.has(code)) return;

      this._pressedKeys.add(code);
      this._recomputeKeyboardState();
      this._commitMergedState(false);
    });

    this._addListener(this._window, 'keyup', (e) => {
      if (this.disposed || !e || !e.code) return;
      if (!this._pressedKeys.delete(e.code)) return;
      this._recomputeKeyboardState();
      this._commitMergedState(false);
    });

    this._addListener(this._window, 'blur', () => {
      this._clearTransientInputs();
    });

    this._addListener(this._window, 'pagehide', () => {
      this._clearTransientInputs();
    });

    this._addListener(this._document, 'visibilitychange', () => {
      const hidden =
        Boolean(this._document?.hidden) || this._document?.visibilityState === 'hidden';
      if (hidden) {
        this._clearTransientInputs();
      }
    });

    this._addListener(this._window, 'gamepadconnected', (e) => {
      if (this.disposed) return;
      this._gamepadConnects += 1;
      this._hasConnectedGamepad = true;
      this._lastIdleGamepadPollMs = null;
      const eventPad = e?.gamepad && e.gamepad.connected !== false ? e.gamepad : null;
      const eventPadIndex =
        eventPad && typeof eventPad.index === 'number' ? eventPad.index : null;
      const eventPadAllowed =
        eventPad !== null &&
        (this._gamepadIndex === null ||
          (eventPadIndex !== null && this._gamepadIndex === eventPadIndex));

      if (eventPadAllowed && eventPadIndex !== null) {
        this._activeGamepadIndex = eventPadIndex;
      }

      let selected = this._selectGamepadEntry();
      if (!selected && eventPadAllowed) {
        this._selectedGamepadScratch.pad = eventPad;
        this._selectedGamepadScratch.index = eventPadIndex ?? 0;
        selected = this._selectedGamepadScratch;
      }
      if (selected || eventPad !== null) {
        this._hasConnectedGamepad = true;
        this._lastIdleGamepadPollMs = null;
      }
      this._pollSelectedGamepadEntry(selected, false);
      this._selectedGamepadScratch.pad = null;
      if (this._autoStartOnGamepad && !this.running) {
        this._startLoop(false);
      }
    });

    this._addListener(this._window, 'gamepaddisconnected', (e) => {
      if (this.disposed) return;
      this._gamepadDisconnects += 1;
      const disconnectedIndex =
        typeof e?.gamepad?.index === 'number' ? e.gamepad.index : null;
      if (
        disconnectedIndex === null ||
        this._activeGamepadIndex === disconnectedIndex
      ) {
        this._activeGamepadIndex = null;
      }
      this._clearGamepadState();

      const remaining = this._selectGamepadEntry(disconnectedIndex);
      if (remaining) {
        this._hasConnectedGamepad = true;
        this._lastIdleGamepadPollMs = null;
        this._pollSelectedGamepadEntry(remaining, true);
        this._selectedGamepadScratch.pad = null;
      } else {
        this._hasConnectedGamepad = false;
        this._lastIdleGamepadPollMs = this._nowFn();
        this._activeGamepadIndex = null;
        this._commitMergedState(true);
        if (!this._manualStart) {
          this.stop();
        }
      }
    });
  }

  isPointerLocked() {
    if (this._canvas && this._document && 'pointerLockElement' in this._document) {
      this.pointerLocked = this._document.pointerLockElement === this._canvas;
    }
    return this.pointerLocked;
  }

  _recomputeKeyboardState() {
    let mask = 0;
    for (const code of this._pressedKeys) {
      mask |= this._keyBindings[code] || 0;
    }
    this._keyButtons = mask >>> 0;

    let kx =
      (this._pressedKeys.has('KeyD') ? 1 : 0) - (this._pressedKeys.has('KeyA') ? 1 : 0);
    let ky =
      (this._pressedKeys.has('KeyW') ? 1 : 0) - (this._pressedKeys.has('KeyS') ? 1 : 0);

    if (kx !== 0 && ky !== 0) {
      kx *= Math.SQRT1_2;
      ky *= Math.SQRT1_2;
    }
    this._keyLx = kx;
    this._keyLy = ky;
  }

  _recomputeMouseButtons() {
    const m = this._mouseButtonsMask;
    this._mouseRt = (m & (1 << 0)) !== 0 ? 1 : 0; // Left click -> Right Trigger (Fire)
    this._mouseLt = (m & (1 << 2)) !== 0 ? 1 : 0; // Right click -> Left Trigger (Aim)

    let bits = 0;
    if ((m & (1 << 1)) !== 0) bits |= INPUT_BUTTONS.RS; // Middle click -> RS
    if ((m & (1 << 3)) !== 0) bits |= INPUT_BUTTONS.LB; // Side Back -> LB
    if ((m & (1 << 4)) !== 0) bits |= INPUT_BUTTONS.RB; // Side Forward -> RB
    this._mouseButtonBits = bits >>> 0;
  }

  _clearMouseButtons() {
    this._mouseButtonsMask = 0;
    this._mouseButtonBits = 0;
    this._mouseLt = 0;
    this._mouseRt = 0;
  }

  _clearGamepadState() {
    this._gpButtons = 0;
    this._gpLx = 0;
    this._gpLy = 0;
    this._gpRx = 0;
    this._gpRy = 0;
    this._gpLt = 0;
    this._gpRt = 0;
    this._hasNonZeroGpState = false;
  }

  _clearTransientInputs() {
    if (this.disposed) return;
    this._blurResets += 1;
    this._stuckResetCount += 1;

    this._pressedKeys.clear();
    this._recomputeKeyboardState();
    this._clearMouseButtons();
    this._clearGamepadState();
    this._commitMergedState(true, true);
  }

  get hasConnectedGamepad() {
    return this._hasConnectedGamepad;
  }

  _selectGamepadEntry(excludeIndex = null) {
    this._selectedGamepadScratch.pad = null;
    this._selectedGamepadScratch.index = -1;

    if (!this._navigator || typeof this._navigator.getGamepads !== 'function') {
      this._hasConnectedGamepad = false;
      return null;
    }
    let pads;
    try {
      pads = this._navigator.getGamepads();
    } catch {
      this._hasConnectedGamepad = false;
      return null;
    }
    if (!pads) {
      this._hasConnectedGamepad = false;
      return null;
    }

    if (typeof this._gamepadIndex === 'number') {
      const target = pads[this._gamepadIndex];
      const targetConnected = Boolean(target && target.connected !== false);
      if (targetConnected) {
        const idx = typeof target.index === 'number' ? target.index : this._gamepadIndex;
        if (excludeIndex === null || (this._gamepadIndex !== excludeIndex && idx !== excludeIndex)) {
          this._hasConnectedGamepad = true;
          this._selectedGamepadScratch.pad = target;
          this._selectedGamepadScratch.index = idx;
          return this._selectedGamepadScratch;
        }
      }
      let anyConnected = false;
      for (let i = 0; i < pads.length; i++) {
        const p = pads[i];
        if (!p || p.connected === false) continue;
        const idx = typeof p.index === 'number' ? p.index : i;
        if (excludeIndex !== null && idx === excludeIndex) continue;
        anyConnected = true;
        break;
      }
      this._hasConnectedGamepad = anyConnected;
      return null;
    }

    let anyConnected = false;
    let activePad = null;
    let activeIdx = -1;
    let firstStandardPad = null;
    let firstStandardIdx = -1;
    let fallbackPad = null;
    let fallbackIdx = -1;

    for (let i = 0; i < pads.length; i++) {
      const pad = pads[i];
      if (!pad || pad.connected === false) continue;
      const idx = typeof pad.index === 'number' ? pad.index : i;
      if (excludeIndex !== null && idx === excludeIndex) continue;
      anyConnected = true;

      if (this._activeGamepadIndex !== null && idx === this._activeGamepadIndex) {
        activePad = pad;
        activeIdx = idx;
      }
      if (pad.mapping === 'standard' && firstStandardPad === null) {
        firstStandardPad = pad;
        firstStandardIdx = idx;
      }
      if (fallbackPad === null) {
        fallbackPad = pad;
        fallbackIdx = idx;
      }
    }

    this._hasConnectedGamepad = anyConnected;

    if (activePad !== null && (activePad.mapping === 'standard' || firstStandardPad === null)) {
      this._selectedGamepadScratch.pad = activePad;
      this._selectedGamepadScratch.index = activeIdx;
      return this._selectedGamepadScratch;
    }
    if (firstStandardPad !== null) {
      this._selectedGamepadScratch.pad = firstStandardPad;
      this._selectedGamepadScratch.index = firstStandardIdx;
      return this._selectedGamepadScratch;
    }
    if (fallbackPad !== null) {
      this._selectedGamepadScratch.pad = fallbackPad;
      this._selectedGamepadScratch.index = fallbackIdx;
      return this._selectedGamepadScratch;
    }
    return null;
  }

  _selectGamepad(excludeIndex = null) {
    const entry = this._selectGamepadEntry(excludeIndex);
    const pad = entry ? entry.pad : null;
    this._selectedGamepadScratch.pad = null;
    return pad;
  }

  _pollSelectedGamepadEntry(entry, forceCommit = false) {
    if (!entry || !entry.pad) {
      if (!forceCommit && this._activeGamepadIndex === null && !this._hasNonZeroGpState) {
        return false;
      }
      this._activeGamepadIndex = null;
      this._clearGamepadState();
      return this._commitMergedState(forceCommit);
    }

    const { pad, index } = entry;
    this._activeGamepadIndex = index;

    const buttons = pad.buttons || EMPTY_ARRAY;
    let gpMask = 0;
    const maxButtons = Math.min(buttons.length, STANDARD_GAMEPAD_BUTTON_MAP.length);
    for (let i = 0; i < maxButtons; i++) {
      const bit = STANDARD_GAMEPAD_BUTTON_MAP[i];
      if (bit !== 0 && isGamepadButtonPressed(buttons[i])) {
        gpMask |= bit;
      }
    }
    const nextGpButtons = gpMask >>> 0;

    const rawLt = readGamepadTrigger(buttons[6]);
    const rawRt = readGamepadTrigger(buttons[7]);
    const nextGpLt = applyTriggerDeadzone(rawLt, this._triggerDeadzone, this._triggerOuterDeadzone);
    const nextGpRt = applyTriggerDeadzone(rawRt, this._triggerDeadzone, this._triggerOuterDeadzone);

    const axes = pad.axes || EMPTY_ARRAY;
    const ySign = this._invertGamepadY ? -1 : 1;
    const rawLx = Number.isFinite(axes[0]) ? axes[0] : 0;
    const rawLy = Number.isFinite(axes[1]) ? axes[1] * ySign : 0;
    const rawRx = Number.isFinite(axes[2]) ? axes[2] : 0;
    const rawRy = Number.isFinite(axes[3]) ? axes[3] * ySign : 0;

    const left = applyRadialDeadzone(
      rawLx,
      rawLy,
      this._leftStickDeadzone,
      this._stickOuterDeadzone,
      this._leftStickScratch
    );
    const right = applyRadialDeadzone(
      rawRx,
      rawRy,
      this._rightStickDeadzone,
      this._stickOuterDeadzone,
      this._rightStickScratch
    );

    if (
      !forceCommit &&
      nextGpButtons === this._gpButtons &&
      nextGpLt === this._gpLt &&
      nextGpRt === this._gpRt &&
      left.x === this._gpLx &&
      left.y === this._gpLy &&
      right.x === this._gpRx &&
      right.y === this._gpRy
    ) {
      return false;
    }

    this._gpButtons = nextGpButtons;
    this._gpLt = nextGpLt;
    this._gpRt = nextGpRt;
    this._gpLx = left.x;
    this._gpLy = left.y;
    this._gpRx = right.x;
    this._gpRy = right.y;
    this._hasNonZeroGpState =
      nextGpButtons !== 0 ||
      nextGpLt !== 0 ||
      nextGpRt !== 0 ||
      left.x !== 0 ||
      left.y !== 0 ||
      right.x !== 0 ||
      right.y !== 0;

    return this._commitMergedState(forceCommit);
  }

  /**
   * Polls navigator.getGamepads(), maps buttons/sticks/triggers with deadzones,
   * merges with keyboard + mouse state, and commits to the SharedArrayBuffer if changed.
   * Returns true if a new snapshot was committed.
   */
  pollGamepads() {
    if (this.disposed) return false;
    this._pollsCount += 1;
    const committed = this._pollSelectedGamepadEntry(this._selectGamepadEntry(), false);
    this._selectedGamepadScratch.pad = null;
    return committed;
  }

  _commitMergedState(force = false, clearMouseDelta = false) {
    const mergedButtons =
      (this._keyButtons | this._mouseButtonBits | this._gpButtons | this._scriptButtons) >>> 0;

    // Priority: keyboard WASD -> gamepad sticks -> scripted sticks.
    const lx = clamp(
      this._keyLx !== 0 ? this._keyLx : this._gpLx !== 0 ? this._gpLx : this._scriptLx,
      -1,
      1
    );
    const ly = clamp(
      this._keyLy !== 0 ? this._keyLy : this._gpLy !== 0 ? this._gpLy : this._scriptLy,
      -1,
      1
    );
    const rx = clamp(this._gpRx !== 0 ? this._gpRx : this._scriptRx, -1, 1);
    const ry = clamp(this._gpRy !== 0 ? this._gpRy : this._scriptRy, -1, 1);

    const lt = clamp(Math.max(this._mouseLt, this._gpLt, this._scriptLt), 0, 1);
    const rt = clamp(Math.max(this._mouseRt, this._gpRt, this._scriptRt), 0, 1);

    // Compare IEEE-754 float32 representations so sub-f32 noise does not cause spurious writes.
    const lxBits = f32ToI32(lx);
    const lyBits = f32ToI32(ly);
    const rxBits = f32ToI32(rx);
    const ryBits = f32ToI32(ry);
    const ltBits = f32ToI32(lt);
    const rtBits = f32ToI32(rt);

    if (
      !force &&
      mergedButtons === this.buttons &&
      lxBits === this._lastLxBits &&
      lyBits === this._lastLyBits &&
      rxBits === this._lastRxBits &&
      ryBits === this._lastRyBits &&
      ltBits === this._lastLtBits &&
      rtBits === this._lastRtBits
    ) {
      return false;
    }

    this.buttons = mergedButtons;
    this._lastLxBits = lxBits;
    this._lastLyBits = lyBits;
    this._lastRxBits = rxBits;
    this._lastRyBits = ryBits;
    this._lastLtBits = ltBits;
    this._lastRtBits = rtBits;
    this._lastLx = i32ToF32(lxBits);
    this._lastLy = i32ToF32(lyBits);
    this._lastRx = i32ToF32(rxBits);
    this._lastRy = i32ToF32(ryBits);
    this._lastLt = i32ToF32(ltBits);
    this._lastRt = i32ToF32(rtBits);

    // Even/odd seqlock update:
    // 1. Bump seq to odd before writing fields.
    // 2. Write buttons and all 6 float32 channels atomically (preserving mouse_dx / mouse_dy
    //    unless explicitly clearing transient state on blur/visibilitychange).
    // 3. Bump seq to even after writing fields.
    this._atomicAddU32(INPUT_WORD_INDICES.SEQ, 1);
    this._atomicStoreU32(INPUT_WORD_INDICES.BUTTONS, mergedButtons);
    if (clearMouseDelta) {
      this._atomicStoreI32(INPUT_WORD_INDICES.MOUSE_DX, 0);
      this._atomicStoreI32(INPUT_WORD_INDICES.MOUSE_DY, 0);
    }
    this._atomicStoreI32(INPUT_WORD_INDICES.LEFT_STICK_X, lxBits);
    this._atomicStoreI32(INPUT_WORD_INDICES.LEFT_STICK_Y, lyBits);
    this._atomicStoreI32(INPUT_WORD_INDICES.RIGHT_STICK_X, rxBits);
    this._atomicStoreI32(INPUT_WORD_INDICES.RIGHT_STICK_Y, ryBits);
    this._atomicStoreI32(INPUT_WORD_INDICES.LEFT_TRIGGER, ltBits);
    this._atomicStoreI32(INPUT_WORD_INDICES.RIGHT_TRIGGER, rtBits);
    this._atomicAddU32(INPUT_WORD_INDICES.SEQ, 1);
    this._notifySeq();
    this._writesCount += 1;

    return true;
  }

  _startLoop(isManual) {
    if (this.disposed) return;
    if (isManual) this._manualStart = true;
    if (this.running) return;
    if (!this._window || typeof this._window.requestAnimationFrame !== 'function') return;

    // Always poll immediately on tick 1 after starting so pre-connected/mock gamepads are detected right away.
    this._lastIdleGamepadPollMs = null;
    this.running = true;
    this._rafId = this._window.requestAnimationFrame(this._boundTick);
  }

  start() {
    this._startLoop(true);
  }

  stop() {
    this._manualStart = false;
    if (!this.running) return;
    this.running = false;
    if (
      this._rafId !== null &&
      this._window &&
      typeof this._window.cancelAnimationFrame === 'function'
    ) {
      this._window.cancelAnimationFrame(this._rafId);
    }
    this._rafId = null;
  }

  _applyScriptSample(sample) {
    if (this.disposed) return false;
    const nextButtons =
      sample && typeof sample.buttonsMask === 'number' ? sample.buttonsMask >>> 0 : 0;
    const sticks = sample && sample.sticks ? sample.sticks : null;
    const triggers = sample && sample.triggers ? sample.triggers : null;
    const nextLx = sticks && Number.isFinite(sticks.lx) ? clamp(sticks.lx, -1, 1) : 0;
    const nextLy = sticks && Number.isFinite(sticks.ly) ? clamp(sticks.ly, -1, 1) : 0;
    const nextRx = sticks && Number.isFinite(sticks.rx) ? clamp(sticks.rx, -1, 1) : 0;
    const nextRy = sticks && Number.isFinite(sticks.ry) ? clamp(sticks.ry, -1, 1) : 0;
    const nextLt = triggers && Number.isFinite(triggers.lt) ? clamp(triggers.lt, 0, 1) : 0;
    const nextRt = triggers && Number.isFinite(triggers.rt) ? clamp(triggers.rt, 0, 1) : 0;

    if (
      sample &&
      (sample.activeEventCount > 0 ||
        nextButtons !== 0 ||
        nextLx !== 0 ||
        nextLy !== 0 ||
        nextRx !== 0 ||
        nextRy !== 0 ||
        nextLt !== 0 ||
        nextRt !== 0)
    ) {
      this._scriptedEventsDispatched += 1;
    }

    this._scriptButtons = nextButtons;
    this._scriptLx = nextLx;
    this._scriptLy = nextLy;
    this._scriptRx = nextRx;
    this._scriptRy = nextRy;
    this._scriptLt = nextLt;
    this._scriptRt = nextRt;
    return this._commitMergedState(false);
  }

  _clearPendingScript() {
    const pending = this._pendingScript;
    if (!pending) return;
    if (pending.timerId !== null && typeof pending.clearTimeoutFn === 'function') {
      pending.clearTimeoutFn(pending.timerId);
    }
    pending.timerId = null;
    this._pendingScript = null;
  }

  _activatePendingScript(reason) {
    const pending = this._pendingScript;
    if (!pending || this.disposed) return false;
    this._clearPendingScript();
    this._scriptPlayer = pending.player;
    this._scriptStartedOn = reason;
    this._scriptManuallyStopped = false;
    if (
      reason === INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME &&
      pending.reanchorOnFirstVisibleFrame
    ) {
      pending.player.reanchor({
        initialEvents: FIRST_VISIBLE_FRAME_INITIAL_A_EVENTS,
        preservePressCount: false,
      });
    } else {
      pending.player.start();
    }
    return true;
  }

  _armPendingTimer(pending, delayMs, reason) {
    if (pending.timerId !== null && typeof pending.clearTimeoutFn === 'function') {
      pending.clearTimeoutFn(pending.timerId);
      pending.timerId = null;
    }
    if (delayMs === 0) {
      return this._activatePendingScript(reason);
    }
    if (typeof pending.setTimeoutFn !== 'function') {
      return this._activatePendingScript(reason);
    }
    pending.timerId = pending.setTimeoutFn(() => {
      if (this._pendingScript !== pending || this.disposed) return;
      pending.timerId = null;
      this._activatePendingScript(reason);
    }, delayMs);
    return false;
  }

  /**
   * Notifies the bridge that the guest runtime has reached `FIRST_PRESENT`.
   * Starts any scripted preset armed with `startOn: 'FIRST_PRESENT'` immediately,
   * and arms (or immediately triggers when `visibleFrameFallbackMs === 0`) the bounded
   * non-stalling fallback for presets armed with `startOn: 'FIRST_VISIBLE_FRAME'` so a
   * dark Start Screen holding the splash overlay can never stall scripted input forever.
   * Returns `true` if a pending script started synchronously.
   */
  notifyFirstPresent() {
    if (this.disposed) return false;
    this._firstPresentNotified = true;
    const pending = this._pendingScript;
    if (!pending) return false;
    if (pending.trigger === INPUT_SCRIPT_START_TRIGGERS.FIRST_PRESENT) {
      return this._activatePendingScript(INPUT_SCRIPT_START_TRIGGERS.FIRST_PRESENT);
    }
    if (pending.trigger === INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME) {
      return this._armPendingTimer(
        pending,
        pending.visibleFrameFallbackMs,
        'FIRST_PRESENT_FALLBACK'
      );
    }
    return false;
  }

  /**
   * Notifies the bridge that the guest runtime has produced its first non-black visible
   * frame (`FIRST_VISIBLE_FRAME`, which also implies `FIRST_PRESENT`).
   * Starts any pending milestone-armed script immediately and cancels any fallback timer.
   * When an anchorable menu preset (`menu-to-ingame`, `chrome-to-ingame`, `chrome-menu-flow-4`,
   * `chrome-menu-flow-2`, `chrome-menu-flow`) already started earlier (e.g. via
   * `FIRST_PRESENT`, `FIRST_PRESENT_FALLBACK`, or `immediate`) during a slow black
   * asset-loading screen, cleanly re-anchors the script clock to `FIRST_VISIBLE_FRAME`
   * and schedules an initial 220 ms `A` press within 500 ms so the Start Screen 3.2 s
   * watchdog never fires.
   * Returns `true` if a script started or re-anchored synchronously.
   */
  notifyFirstVisibleFrame() {
    if (this.disposed) return false;
    const alreadyVisible = this._firstVisibleFrameNotified;
    this._firstPresentNotified = true;
    this._firstVisibleFrameNotified = true;
    if (alreadyVisible) return false;
    if (this._pendingScript) {
      return this._activatePendingScript(INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME);
    }
    if (
      this._scriptPlayer &&
      !this._scriptManuallyStopped &&
      this._scriptReanchorOnFirstVisibleFrame &&
      this._scriptStartedOn !== INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME
    ) {
      this._scriptStartedOn = INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME;
      this._scriptReanchoredCount += 1;
      this._scriptPlayer.reanchor({
        initialEvents: FIRST_VISIBLE_FRAME_INITIAL_A_EVENTS,
        preservePressCount: true,
      });
      return true;
    }
    return false;
  }

  /**
   * Convenience milestone dispatcher for `'FIRST_PRESENT'` and `'FIRST_VISIBLE_FRAME'`.
   */
  notifyMilestone(milestone) {
    if (this.disposed || typeof milestone !== 'string') return false;
    const normalized = milestone.trim().toLowerCase().replace(/[\s-]+/g, '_');
    if (normalized === 'first_present' || normalized === 'firstpresent') {
      return this.notifyFirstPresent();
    }
    if (normalized === 'first_visible_frame' || normalized === 'firstvisibleframe') {
      return this.notifyFirstVisibleFrame();
    }
    return false;
  }

  /**
   * Starts deterministic scripted input playback (preset name, repeating schedule,
   * explicit event array, or ScriptedInputPlayer instance). Merges scripted buttons
   * alongside live keyboard/mouse/gamepad input without clobbering live state on release.
   *
   * Supports optional `startOn`:
   * - `'immediate'` (default): starts playback immediately.
   * - `'FIRST_PRESENT'`: starts immediately if `FIRST_PRESENT` already fired, otherwise
   *   arms until `notifyFirstPresent()` (or `notifyFirstVisibleFrame()`, or optional
   *   `firstPresentTimeoutMs`).
   * - `'FIRST_VISIBLE_FRAME'` (or `waitForFirstVisibleFrame: true`): starts on
   *   `notifyFirstVisibleFrame()`, and automatically falls back cleanly after
   *   `visibleFrameFallbackMs` (default 2000 ms) once `notifyFirstPresent()` fires so a
   *   dark Start Screen can never stall scripted input forever.
   */
  startScript(presetOrSpec, options = {}) {
    if (this.disposed) return null;
    const opts = options || {};
    const startTrigger = normalizeStartTrigger(opts.startOn, opts.waitForFirstVisibleFrame);
    const visibleFrameFallbackMs =
      opts.visibleFrameFallbackMs !== undefined && opts.visibleFrameFallbackMs !== null
        ? validateNonNegativeFinite(opts.visibleFrameFallbackMs, 'visibleFrameFallbackMs')
        : DEFAULT_VISIBLE_FRAME_FALLBACK_MS;
    const firstPresentTimeoutMs = validateOptionalTimeoutMs(
      opts.firstPresentTimeoutMs,
      'firstPresentTimeoutMs'
    );

    this.stopScript();

    let player;
    if (presetOrSpec instanceof ScriptedInputPlayer) {
      player = presetOrSpec;
      player._setBridgeSink((sample) => {
        this._applyScriptSample(sample);
      });
    } else {
      player = new ScriptedInputPlayer(presetOrSpec, {
        ...opts,
        _bridgeSink: (sample) => {
          this._applyScriptSample(sample);
        },
      });
    }

    const isDefaultAnchorPreset =
      typeof player.preset === 'string' &&
      FIRST_VISIBLE_FRAME_ANCHOR_PRESET_SET.has(player.preset);
    const reanchorOnFirstVisibleFrame =
      typeof opts.reanchorOnFirstVisibleFrame === 'boolean'
        ? opts.reanchorOnFirstVisibleFrame
        : isDefaultAnchorPreset;

    this._scriptPlayer = player;
    this._scriptStartedOn = null;
    this._scriptManuallyStopped = false;
    this._scriptReanchorOnFirstVisibleFrame = reanchorOnFirstVisibleFrame;

    if (startTrigger === INPUT_SCRIPT_START_TRIGGERS.IMMEDIATE) {
      this._scriptStartedOn = INPUT_SCRIPT_START_TRIGGERS.IMMEDIATE;
      player.start();
      return player;
    }

    if (
      startTrigger === INPUT_SCRIPT_START_TRIGGERS.FIRST_PRESENT &&
      (this._firstPresentNotified || this._firstVisibleFrameNotified)
    ) {
      this._scriptStartedOn = INPUT_SCRIPT_START_TRIGGERS.FIRST_PRESENT;
      player.start();
      return player;
    }

    if (
      startTrigger === INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME &&
      this._firstVisibleFrameNotified
    ) {
      this._scriptStartedOn = INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME;
      if (reanchorOnFirstVisibleFrame) {
        player.reanchor({
          initialEvents: FIRST_VISIBLE_FRAME_INITIAL_A_EVENTS,
          preservePressCount: false,
        });
      } else {
        player.start();
      }
      return player;
    }

    const setTimeoutFn =
      opts.setTimeoutFn !== undefined
        ? opts.setTimeoutFn
        : typeof globalThis !== 'undefined' && typeof globalThis.setTimeout === 'function'
          ? globalThis.setTimeout.bind(globalThis)
          : null;
    const clearTimeoutFn =
      opts.clearTimeoutFn !== undefined
        ? opts.clearTimeoutFn
        : typeof globalThis !== 'undefined' && typeof globalThis.clearTimeout === 'function'
          ? globalThis.clearTimeout.bind(globalThis)
          : null;

    const pending = {
      player,
      trigger: startTrigger,
      visibleFrameFallbackMs,
      firstPresentTimeoutMs,
      reanchorOnFirstVisibleFrame,
      setTimeoutFn,
      clearTimeoutFn,
      timerId: null,
    };
    this._pendingScript = pending;

    if (
      startTrigger === INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME &&
      this._firstPresentNotified
    ) {
      this._armPendingTimer(pending, visibleFrameFallbackMs, 'FIRST_PRESENT_FALLBACK');
    } else if (firstPresentTimeoutMs !== null) {
      this._armPendingTimer(pending, firstPresentTimeoutMs, 'FIRST_PRESENT_TIMEOUT');
    }

    return player;
  }

  /**
   * Stops any active or pending scripted input playback, cleans up timers, and clears
   * scripted button bits immediately from the SharedArrayBuffer while preserving live inputs.
   */
  stopScript() {
    this._scriptManuallyStopped = true;
    this._clearPendingScript();
    if (this._scriptPlayer) {
      this._scriptPlayer.stop();
    }
    const hadScriptState =
      this._scriptButtons !== 0 ||
      this._scriptLx !== 0 ||
      this._scriptLy !== 0 ||
      this._scriptRx !== 0 ||
      this._scriptRy !== 0 ||
      this._scriptLt !== 0 ||
      this._scriptRt !== 0;
    this._scriptButtons = 0;
    this._scriptLx = 0;
    this._scriptLy = 0;
    this._scriptRx = 0;
    this._scriptRy = 0;
    this._scriptLt = 0;
    this._scriptRt = 0;
    if (hadScriptState && !this.disposed) {
      this._commitMergedState(false);
    }
  }

  /**
   * Returns current scripted playback status and active-preset telemetry:
   * `{ active, preset, activePreset, elapsedMs, elapsedScriptMs, pressCount,
   *    scriptedPressesSent, lastScriptedButton, currentMask }`.
   */
  getScriptState() {
    if (!this._scriptPlayer) {
      return {
        active: false,
        preset: null,
        activePreset: null,
        elapsedMs: 0,
        elapsedScriptMs: 0,
        pressCount: 0,
        scriptedPressesSent: 0,
        lastScriptedButton: null,
        currentMask: 0,
      };
    }
    return this._scriptPlayer.getState();
  }

  /**
   * Returns a sanitized bridge & scripted-input telemetry snapshot for HUD/diagnostics.
   */
  getStats() {
    const script = this.getScriptState();
    return {
      running: this.running,
      disposed: this.disposed,
      pointerLocked: this.isPointerLocked(),
      isSharedBuffer: this.isSharedBuffer,
      usesAtomics: this.usesAtomics,
      buttons: this.buttons,
      scriptActive: script.active,
      scriptPending: Boolean(this._pendingScript),
      scriptStartedOn: this._scriptStartedOn,
      scriptReanchoredCount: this._scriptReanchoredCount,
      firstPresentNotified: this._firstPresentNotified,
      firstVisibleFrameNotified: this._firstVisibleFrameNotified,
      activePreset: script.activePreset,
      preset: script.preset,
      scriptedPressesSent: script.scriptedPressesSent,
      lastScriptedButton: script.lastScriptedButton,
      elapsedScriptMs: script.elapsedScriptMs,
      currentScriptMask: script.currentMask,
      availablePresets: INPUT_SCRIPT_PRESET_NAMES,
      blurResets: this._blurResets,
      stuckResetCount: this._stuckResetCount,
      pollsCount: this._pollsCount,
      writesCount: this._writesCount,
      notifyCount: this._notifyCount,
      scriptedEventsDispatched: this._scriptedEventsDispatched,
      gamepadConnects: this._gamepadConnects,
      gamepadDisconnects: this._gamepadDisconnects,
      activeGamepadIndex: this._activeGamepadIndex,
      hasConnectedGamepad: this._hasConnectedGamepad,
      idleGamepadPollsSkipped: this._idleGamepadPollsSkipped,
    };
  }

  /**
   * Alias for `getStats()`, returning a sanitized bridge & scripted-input status snapshot.
   */
  getStatus() {
    return this.getStats();
  }

  /**
   * Returns hot-path input-state telemetry (`pollsCount`, `writesCount`, `stuckResetCount`,
   * `activePreset`, `scriptedEventsDispatched`, `seq`, `buttons`, `pointerLocked`,
   * `activeGamepadIndex`).
   * Pass `{ reuse: true }` or a pre-allocated target object (`out`) for zero-allocation reads.
   */
  getInputTelemetry(optionsOrOut = null) {
    let target = null;
    if (optionsOrOut && typeof optionsOrOut === 'object') {
      if (typeof optionsOrOut.reuse === 'boolean' || 'out' in optionsOrOut) {
        if (optionsOrOut.out && typeof optionsOrOut.out === 'object') {
          target = optionsOrOut.out;
        } else if (optionsOrOut.reuse === true) {
          target = this._telemetryScratch;
        }
      } else {
        target = optionsOrOut;
      }
    }
    const out = target || {
      pollsCount: 0,
      writesCount: 0,
      stuckResetCount: 0,
      activePreset: null,
      scriptedEventsDispatched: 0,
      seq: 0,
      buttons: 0,
      pointerLocked: false,
      activeGamepadIndex: null,
    };
    out.pollsCount = this._pollsCount;
    out.writesCount = this._writesCount;
    out.stuckResetCount = this._stuckResetCount;
    out.activePreset =
      this._scriptPlayer && this._scriptPlayer._active ? this._scriptPlayer.preset : null;
    out.scriptedEventsDispatched = this._scriptedEventsDispatched;
    out.seq = this._atomicLoadU32(INPUT_WORD_INDICES.SEQ);
    out.buttons = this.buttons;
    out.pointerLocked = this.isPointerLocked();
    out.activeGamepadIndex = this._activeGamepadIndex;
    return out;
  }

  consumeWheelDelta() {
    const delta = this._wheelDelta;
    this._wheelDelta = 0;
    return delta;
  }

  reset() {
    this._scriptManuallyStopped = true;
    this._clearPendingScript();
    if (this._scriptPlayer) {
      this._scriptPlayer._stopWithoutSink();
    }
    this._scriptButtons = 0;
    this._scriptLx = 0;
    this._scriptLy = 0;
    this._scriptRx = 0;
    this._scriptRy = 0;
    this._scriptLt = 0;
    this._scriptRt = 0;

    this._pressedKeys.clear();
    this._recomputeKeyboardState();
    this._clearMouseButtons();
    this._wheelDelta = 0;

    this._clearGamepadState();
    this._activeGamepadIndex = null;
    this._hasConnectedGamepad = false;
    this._lastIdleGamepadPollMs = null;

    this.buttons = 0;
    this._lastLx = 0;
    this._lastLy = 0;
    this._lastRx = 0;
    this._lastRy = 0;
    this._lastLt = 0;
    this._lastRt = 0;
    this._lastLxBits = 0;
    this._lastLyBits = 0;
    this._lastRxBits = 0;
    this._lastRyBits = 0;
    this._lastLtBits = 0;
    this._lastRtBits = 0;

    this._atomicAddU32(INPUT_WORD_INDICES.SEQ, 1);
    this._atomicStoreU32(INPUT_WORD_INDICES.BUTTONS, 0);
    this._atomicStoreI32(INPUT_WORD_INDICES.MOUSE_DX, 0);
    this._atomicStoreI32(INPUT_WORD_INDICES.MOUSE_DY, 0);
    this._atomicStoreI32(INPUT_WORD_INDICES.LEFT_STICK_X, 0);
    this._atomicStoreI32(INPUT_WORD_INDICES.LEFT_STICK_Y, 0);
    this._atomicStoreI32(INPUT_WORD_INDICES.RIGHT_STICK_X, 0);
    this._atomicStoreI32(INPUT_WORD_INDICES.RIGHT_STICK_Y, 0);
    this._atomicStoreI32(INPUT_WORD_INDICES.LEFT_TRIGGER, 0);
    this._atomicStoreI32(INPUT_WORD_INDICES.RIGHT_TRIGGER, 0);
    this._atomicAddU32(INPUT_WORD_INDICES.SEQ, 1);
    this._notifySeq();
    this._writesCount += 1;
  }

  /**
   * Captures a consistent snapshot from the 40-byte SharedArrayBuffer using the same
   * even/odd seqlock protocol as playgta6::os::GameInputReader::read_snapshot().
   * Pass `{ reuse: true }` (zero-alloc internal scratch) or a pre-allocated `out` object
   * to avoid per-frame allocations.
   */
  getSnapshot(maxRetriesOrOptions = 64, out = null) {
    let maxRetries = 64;
    let target = out;

    if (maxRetriesOrOptions && typeof maxRetriesOrOptions === 'object') {
      if (typeof maxRetriesOrOptions.maxRetries === 'number') {
        maxRetries = maxRetriesOrOptions.maxRetries;
      }
      if (maxRetriesOrOptions.out && typeof maxRetriesOrOptions.out === 'object') {
        target = maxRetriesOrOptions.out;
      } else if (maxRetriesOrOptions.reuse === true) {
        target = this._snapshotScratch;
      }
    } else if (typeof maxRetriesOrOptions === 'number') {
      maxRetries = maxRetriesOrOptions;
    }

    const attempts = maxRetries > 0 ? maxRetries : 1;
    for (let i = 0; i < attempts; i++) {
      const seqBefore = this._atomicLoadU32(INPUT_WORD_INDICES.SEQ);
      if ((seqBefore & 1) !== 0) continue;

      const buttons = this._atomicLoadU32(INPUT_WORD_INDICES.BUTTONS);
      const mouse_dx = this._atomicLoadI32(INPUT_WORD_INDICES.MOUSE_DX);
      const mouse_dy = this._atomicLoadI32(INPUT_WORD_INDICES.MOUSE_DY);
      const lx = i32ToF32(this._atomicLoadI32(INPUT_WORD_INDICES.LEFT_STICK_X));
      const ly = i32ToF32(this._atomicLoadI32(INPUT_WORD_INDICES.LEFT_STICK_Y));
      const rx = i32ToF32(this._atomicLoadI32(INPUT_WORD_INDICES.RIGHT_STICK_X));
      const ry = i32ToF32(this._atomicLoadI32(INPUT_WORD_INDICES.RIGHT_STICK_Y));
      const lt = i32ToF32(this._atomicLoadI32(INPUT_WORD_INDICES.LEFT_TRIGGER));
      const rt = i32ToF32(this._atomicLoadI32(INPUT_WORD_INDICES.RIGHT_TRIGGER));

      const seqAfter = this._atomicLoadU32(INPUT_WORD_INDICES.SEQ);
      if (seqBefore === seqAfter) {
        const snap = target || {
          seq: 0,
          buttons: 0,
          mouse_dx: 0,
          mouse_dy: 0,
          lx: 0,
          ly: 0,
          rx: 0,
          ry: 0,
          lt: 0,
          rt: 0,
          pointerLocked: false,
          wheelDelta: 0,
        };
        snap.seq = seqBefore;
        snap.buttons = buttons;
        snap.mouse_dx = mouse_dx;
        snap.mouse_dy = mouse_dy;
        snap.lx = lx;
        snap.ly = ly;
        snap.rx = rx;
        snap.ry = ry;
        snap.lt = lt;
        snap.rt = rt;
        snap.pointerLocked = this.isPointerLocked();
        snap.wheelDelta = this._wheelDelta;
        return snap;
      }
    }

    return null;
  }

  destroy() {
    if (this.disposed) return;
    this.stopScript();
    this.stop();
    for (const { target, event, handler, options } of this._listeners) {
      if (target && typeof target.removeEventListener === 'function') {
        target.removeEventListener(event, handler, options);
      }
    }
    this._listeners.length = 0;
    this.reset();
    this.disposed = true;
  }

  dispose() {
    this.destroy();
  }

  detach() {
    this.destroy();
  }
}
