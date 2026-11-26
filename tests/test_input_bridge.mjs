import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  INPUT_ABI_VERSION,
  INPUT_STATE_BYTES,
  INPUT_STATE_WORDS,
  INPUT_OFFSETS,
  INPUT_WORD_INDICES,
  INPUT_ABI_LAYOUT,
  INPUT_BUTTONS,
  DEFAULT_KEY_BINDINGS,
  STANDARD_GAMEPAD_BUTTON_MAP,
  IDLE_GAMEPAD_POLL_INTERVAL_MS,
  MIN_SCRIPT_HOLD_MS,
  FIRST_VISIBLE_FRAME_ANCHOR_PRESETS,
  INPUT_SCRIPT_START_TRIGGERS,
  CHROME_LEGAL_TERMS_ACCEPT_OFFSETS,
  applyDeadzone,
  applyRadialDeadzone as bridgeApplyRadialDeadzone,
  applyAxialDeadzone,
  applyTriggerDeadzone,
  INPUT_SCRIPT_PRESETS,
  INPUT_SCRIPT_PRESET_NAMES,
  ScriptedInputPlayer,
  InputBridge,
} from '../web/input/input_bridge.js';
import { applyRadialDeadzone } from '../web/input/input_profiles.js';
import { sanitizeString } from '../web/telemetry/crash_report.js';
import { HudController, HIDDEN_HUD_MIN_RENDER_INTERVAL_MS } from '../web/ui/hud_controller.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');

class MockEventTarget {
  constructor() {
    this._handlers = new Map();
  }

  addEventListener(type, handler) {
    if (!this._handlers.has(type)) {
      this._handlers.set(type, new Set());
    }
    this._handlers.get(type).add(handler);
  }

  removeEventListener(type, handler) {
    const set = this._handlers.get(type);
    if (!set) return;
    set.delete(handler);
    if (set.size === 0) this._handlers.delete(type);
  }

  dispatch(type, event = {}) {
    const set = this._handlers.get(type);
    if (!set) return;
    for (const fn of [...set]) {
      fn(event);
    }
  }

  listenerCount(type) {
    if (type !== undefined) {
      return this._handlers.get(type)?.size ?? 0;
    }
    let total = 0;
    for (const set of this._handlers.values()) {
      total += set.size;
    }
    return total;
  }
}

function createMockEnv({ gamepads = [], withRaf = false } = {}) {
  const canvas = new MockEventTarget();
  canvas.pointerLockRequests = 0;
  canvas.requestPointerLock = () => {
    canvas.pointerLockRequests += 1;
    return Promise.resolve();
  };

  const doc = new MockEventTarget();
  doc.pointerLockElement = null;
  doc.hidden = false;
  doc.visibilityState = 'visible';

  const win = new MockEventTarget();
  let nextRafId = 1;
  const rafQueue = new Map();
  if (withRaf) {
    win.requestAnimationFrame = (cb) => {
      const id = nextRafId++;
      rafQueue.set(id, cb);
      return id;
    };
    win.cancelAnimationFrame = (id) => {
      rafQueue.delete(id);
    };
  }

  let currentGamepads = gamepads;
  const nav = {
    getGamepads: () => currentGamepads,
  };

  return {
    canvas,
    doc,
    win,
    nav,
    setGamepads(pads) {
      currentGamepads = pads;
    },
    lockPointer() {
      doc.pointerLockElement = canvas;
      doc.dispatch('pointerlockchange');
    },
    unlockPointer() {
      doc.pointerLockElement = null;
      doc.dispatch('pointerlockchange');
    },
    flushRaf() {
      const entries = [...rafQueue.entries()];
      rafQueue.clear();
      for (const [, cb] of entries) {
        cb(16.67);
      }
      return entries.length;
    },
    pendingRafCount() {
      return rafQueue.size;
    },
  };
}

function makeStandardGamepad({
  buttons = {},
  axes = [0, 0, 0, 0],
  connected = true,
  mapping = 'standard',
  index = 0,
} = {}) {
  const btnArray = Array.from({ length: 17 }, (_, i) => {
    const spec = buttons[i];
    if (typeof spec === 'number') {
      return { pressed: spec > 0.5, touched: spec > 0, value: spec };
    }
    if (spec && typeof spec === 'object') {
      return {
        pressed: Boolean(spec.pressed),
        touched: Boolean(spec.touched ?? spec.pressed),
        value: spec.value ?? (spec.pressed ? 1 : 0),
      };
    }
    return { pressed: false, touched: false, value: 0 };
  });

  return {
    id: 'Xbox Wireless Controller (STANDARD GAMEPAD)',
    index,
    connected,
    mapping,
    buttons: btnArray,
    axes: axes.slice(),
  };
}

function approxEqual(actual, expected, eps = 1e-5) {
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `Expected ${actual} to be within ${eps} of ${expected}`
  );
}

test('ABI constants and button bitmasks match runtime/abi/playgta6_abi.h, runtime/os/game_input.h, and xbox_scarlett_guest_input.cpp byte-for-byte', () => {
  assert.equal(INPUT_ABI_VERSION, 1);
  assert.equal(INPUT_STATE_BYTES, 40);
  assert.equal(INPUT_STATE_WORDS, 10);
  assert.deepEqual(INPUT_OFFSETS, {
    SEQ: 0,
    BUTTONS: 4,
    MOUSE_DX: 8,
    MOUSE_DY: 12,
    LEFT_STICK_X: 16,
    LEFT_STICK_Y: 20,
    RIGHT_STICK_X: 24,
    RIGHT_STICK_Y: 28,
    LEFT_TRIGGER: 32,
    RIGHT_TRIGGER: 36,
  });
  assert.deepEqual(INPUT_WORD_INDICES, {
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
  assert.equal(INPUT_ABI_LAYOUT.version, INPUT_ABI_VERSION);
  assert.equal(INPUT_ABI_LAYOUT.byteLength, INPUT_STATE_BYTES);
  assert.equal(INPUT_ABI_LAYOUT.wordCount, INPUT_STATE_WORDS);
  assert.equal(INPUT_ABI_LAYOUT.fields.length, INPUT_STATE_WORDS);

  // Verify C++ headers in repo still agree on kAbiVersion, struct InputState field order/types/offsets,
  // sizeof(InputState) == 40, and bitmask shifts.
  const abiHeader = fs.readFileSync(
    path.join(REPO_ROOT, 'runtime/abi/playgta6_abi.h'),
    'utf8'
  );
  assert.match(abiHeader, /kAbiVersion\s*=\s*1\s*;/);
  assert.match(abiHeader, /static_assert\(sizeof\(InputState\)\s*==\s*40/);

  const structMatch = abiHeader.match(/struct\s+InputState\s*\{([\s\S]*?)\};/);
  assert.ok(structMatch, 'Expected struct InputState in runtime/abi/playgta6_abi.h');
  const cppFields = [];
  const cppTypeToJs = { uint32_t: 'uint32', int32_t: 'int32', float: 'float32' };
  for (const rawLine of structMatch[1].split('\n')) {
    const cleaned = rawLine.replace(/\/\/.*$/, '').trim();
    if (!cleaned) continue;
    const m = cleaned.match(/^(uint32_t|int32_t|float)\s+([a-z0-9_]+)\s*;$/);
    assert.ok(m, `Unexpected field declaration in struct InputState: ${cleaned}`);
    cppFields.push({ type: cppTypeToJs[m[1]], name: m[2] });
  }
  assert.equal(cppFields.length, INPUT_ABI_LAYOUT.fields.length);
  for (let i = 0; i < cppFields.length; i++) {
    const jsField = INPUT_ABI_LAYOUT.fields[i];
    const cppField = cppFields[i];
    assert.equal(jsField.name, cppField.name, `Field #${i} name mismatch`);
    assert.equal(jsField.type, cppField.type, `Field #${i} type mismatch`);
    assert.equal(jsField.offset, i * 4, `Field #${i} byte offset mismatch`);
    assert.equal(jsField.wordIndex, i, `Field #${i} word index mismatch`);
    assert.equal(jsField.size, 4, `Field #${i} byte size mismatch`);
    assert.equal(INPUT_OFFSETS[jsField.key], jsField.offset);
    assert.equal(INPUT_WORD_INDICES[jsField.key], jsField.wordIndex);
  }

  const gameInputHeader = fs.readFileSync(
    path.join(REPO_ROOT, 'runtime/os/game_input.h'),
    'utf8'
  );
  const expectedCppBits = [
    ['kMoveForward', INPUT_BUTTONS.MOVE_FORWARD, 0],
    ['kMoveBackward', INPUT_BUTTONS.MOVE_BACKWARD, 1],
    ['kMoveLeft', INPUT_BUTTONS.MOVE_LEFT, 2],
    ['kMoveRight', INPUT_BUTTONS.MOVE_RIGHT, 3],
    ['kJump', INPUT_BUTTONS.JUMP, 4],
    ['kSprint', INPUT_BUTTONS.SPRINT, 5],
    ['kGamepadA', INPUT_BUTTONS.A, 8],
    ['kGamepadB', INPUT_BUTTONS.B, 9],
    ['kGamepadX', INPUT_BUTTONS.X, 10],
    ['kGamepadY', INPUT_BUTTONS.Y, 11],
    ['kLeftBumper', INPUT_BUTTONS.LB, 12],
    ['kRightBumper', INPUT_BUTTONS.RB, 13],
    ['kMenu', INPUT_BUTTONS.MENU, 14],
    ['kView', INPUT_BUTTONS.VIEW, 15],
  ];

  for (const [cppName, jsValue, bit] of expectedCppBits) {
    assert.equal(jsValue, 1 << bit, `${cppName} bit mismatch`);
    const re = new RegExp(`constexpr\\s+uint32_t\\s+${cppName}\\s*=\\s*1u\\s*<<\\s*${bit}\\s*;`);
    assert.match(gameInputHeader, re, `Header mismatch for ${cppName}`);
  }

  // Verify Xbox GameInput guest shim references the exact same SAB keyboard bits (0..5).
  const guestInputCpp = fs.readFileSync(
    path.join(REPO_ROOT, 'runtime/os/xbox_scarlett_guest_input.cpp'),
    'utf8'
  );
  for (const sym of ['kMoveForward', 'kMoveLeft', 'kMoveBackward', 'kMoveRight', 'kJump', 'kSprint']) {
    assert.match(guestInputCpp, new RegExp(`os::input_button::${sym}\\b`));
  }

  // Verify W3C Standard Gamepad button map covers all 17 indices accurately.
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP.length, 17);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[0], INPUT_BUTTONS.A);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[1], INPUT_BUTTONS.B);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[2], INPUT_BUTTONS.X);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[3], INPUT_BUTTONS.Y);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[4], INPUT_BUTTONS.LB);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[5], INPUT_BUTTONS.RB);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[6], 0);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[7], 0);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[8], INPUT_BUTTONS.VIEW);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[9], INPUT_BUTTONS.MENU);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[10], INPUT_BUTTONS.LS);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[11], INPUT_BUTTONS.RS);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[12], INPUT_BUTTONS.DPAD_UP);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[13], INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[14], INPUT_BUTTONS.DPAD_LEFT);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[15], INPUT_BUTTONS.DPAD_RIGHT);
  assert.equal(STANDARD_GAMEPAD_BUTTON_MAP[16], INPUT_BUTTONS.GUIDE);

  // Ensure all defined INPUT_BUTTONS bits are distinct powers of two.
  const values = Object.values(INPUT_BUTTONS);
  const seen = new Set();
  for (const v of values) {
    assert.ok(v > 0 && (v & (v - 1)) === 0, `Expected single bit mask, got ${v}`);
    assert.ok(!seen.has(v), `Duplicate button bitmask ${v}`);
    seen.add(v);
  }
});

test('SharedArrayBuffer 40-byte binary layout matches DataView offsets and seqlock protocol', () => {
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  assert.ok(bridge.sab instanceof SharedArrayBuffer);
  assert.equal(bridge.sab.byteLength, 40);
  assert.equal(bridge.i32.length, 10);
  assert.equal(bridge.u32.length, 10);
  assert.equal(bridge.f32.length, 10);

  const initial = bridge.getSnapshot();
  assert.equal(initial.seq, 0);
  assert.equal(initial.buttons, 0);
  assert.equal(initial.mouse_dx, 0);
  assert.equal(initial.mouse_dy, 0);
  assert.equal(initial.lx, 0);
  assert.equal(initial.ly, 0);
  assert.equal(initial.rx, 0);
  assert.equal(initial.ry, 0);
  assert.equal(initial.lt, 0);
  assert.equal(initial.rt, 0);

  // Lock pointer and drive every field: mouse delta, buttons, left/right sticks, triggers.
  env.lockPointer();
  env.win.dispatch('mousemove', { movementX: 17, movementY: -23 });
  env.win.dispatch('keydown', { code: 'KeyW' });
  env.setGamepads([
    makeStandardGamepad({
      buttons: {
        0: 1, // A
        6: 0.5, // LT
        7: 0.98, // RT
      },
      axes: [0, 0, 0.5, -0.5], // RX = +0.5, RY = +0.5 (Y inverted)
    }),
  ]);
  assert.equal(bridge.pollGamepads(), true);

  // Verify raw little-endian bytes via DataView at exact C++ playgta6::InputState offsets.
  const dv = new DataView(bridge.sab);
  const seq = dv.getUint32(0, true);
  assert.ok(seq > 0 && seq % 2 === 0, `seq should be positive even, got ${seq}`);
  assert.equal(dv.getUint32(4, true), INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.A);
  assert.equal(dv.getInt32(8, true), 17);
  assert.equal(dv.getInt32(12, true), -23);
  approxEqual(dv.getFloat32(16, true), 0.0);
  approxEqual(dv.getFloat32(20, true), 1.0);

  const expectedRight = applyDeadzone(0.5, 0.5);
  approxEqual(dv.getFloat32(24, true), expectedRight.x);
  approxEqual(dv.getFloat32(28, true), expectedRight.y);
  approxEqual(dv.getFloat32(32, true), applyTriggerDeadzone(0.5));
  approxEqual(dv.getFloat32(36, true), 1.0);

  // Seqlock odd sequence (write in progress) causes getSnapshot() to retry and return null.
  Atomics.store(bridge.u32, 0, seq + 1);
  assert.equal(bridge.getSnapshot(4), null);
  Atomics.store(bridge.u32, 0, seq + 2);
  const recovered = bridge.getSnapshot();
  assert.equal(recovered.seq, seq + 2);

  bridge.destroy();
});

test('constructor supports options-only call, custom SAB injection, and Node without DOM globals', () => {
  const customSab = new SharedArrayBuffer(40);
  const bridge = new InputBridge({
    sab: customSab,
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  assert.equal(bridge.sab, customSab);
  assert.equal(bridge.pollGamepads(), false);
  assert.equal(bridge.getSnapshot().seq, 0);
  bridge.destroy();

  assert.throws(
    () => new InputBridge({ sab: new SharedArrayBuffer(16) }),
    RangeError
  );
});

test('keyboard WASD drives normalized left stick axes + digital bits and Xbox key bindings work cleanly', () => {
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  // W -> forward (+Y)
  env.win.dispatch('keydown', { code: 'KeyW' });
  let snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_FORWARD);
  approxEqual(snap.lx, 0);
  approxEqual(snap.ly, 1);
  const seqAfterW = snap.seq;

  // Repeat keydown should not bump seq.
  env.win.dispatch('keydown', { code: 'KeyW', repeat: true });
  assert.equal(bridge.getSnapshot().seq, seqAfterW);

  // Unbound key should not bump seq.
  env.win.dispatch('keydown', { code: 'KeyZ' });
  assert.equal(bridge.getSnapshot().seq, seqAfterW);

  // W + D -> diagonal unit vector (1/sqrt(2), 1/sqrt(2))
  env.win.dispatch('keydown', { code: 'KeyD' });
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.MOVE_RIGHT);
  approxEqual(snap.lx, Math.SQRT1_2);
  approxEqual(snap.ly, Math.SQRT1_2);
  approxEqual(Math.hypot(snap.lx, snap.ly), 1.0);

  // Opposing W + S cancels Y axis to 0 while D remains +1
  env.win.dispatch('keydown', { code: 'KeyS' });
  snap = bridge.getSnapshot();
  approxEqual(snap.lx, 1);
  approxEqual(snap.ly, 0);

  // Release W and D -> only S remains (0, -1)
  env.win.dispatch('keyup', { code: 'KeyW' });
  env.win.dispatch('keyup', { code: 'KeyD' });
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_BACKWARD);
  approxEqual(snap.lx, 0);
  approxEqual(snap.ly, -1);
  env.win.dispatch('keyup', { code: 'KeyS' });

  // Verify all key bindings (Space, Enter, NumpadEnter, Shift, Ctrl, Backspace, E, F, R, Q, Tab, Escape, Arrows)
  const keysToVerify = [
    ['Space', INPUT_BUTTONS.JUMP | INPUT_BUTTONS.A],
    ['Enter', INPUT_BUTTONS.A],
    ['NumpadEnter', INPUT_BUTTONS.A],
    ['ShiftLeft', INPUT_BUTTONS.SPRINT | INPUT_BUTTONS.LS],
    ['ControlLeft', INPUT_BUTTONS.B],
    ['Backspace', INPUT_BUTTONS.B],
    ['KeyR', INPUT_BUTTONS.X],
    ['KeyF', INPUT_BUTTONS.Y],
    ['KeyQ', INPUT_BUTTONS.LB],
    ['KeyE', INPUT_BUTTONS.RB],
    ['Tab', INPUT_BUTTONS.VIEW],
    ['Escape', INPUT_BUTTONS.B | INPUT_BUTTONS.MENU],
    ['KeyC', INPUT_BUTTONS.RS],
    ['ArrowUp', INPUT_BUTTONS.DPAD_UP],
    ['ArrowDown', INPUT_BUTTONS.DPAD_DOWN],
    ['ArrowLeft', INPUT_BUTTONS.DPAD_LEFT],
    ['ArrowRight', INPUT_BUTTONS.DPAD_RIGHT],
  ];

  for (const [code, expectedMask] of keysToVerify) {
    assert.equal(DEFAULT_KEY_BINDINGS[code], expectedMask);
    env.win.dispatch('keydown', { code });
    assert.equal(bridge.getSnapshot().buttons, expectedMask, `keydown ${code}`);
    env.win.dispatch('keyup', { code });
    assert.equal(bridge.getSnapshot().buttons, 0, `keyup ${code}`);
  }

  // Prevent default on navigation keys while pointer is locked.
  env.lockPointer();
  let prevented = false;
  env.win.dispatch('keydown', {
    code: 'Space',
    preventDefault: () => {
      prevented = true;
    },
  });
  assert.equal(prevented, true);

  bridge.destroy();
});

test('Pointer Lock click, mouse delta accumulation, C++ atomic consume, mouse buttons, and wheel', () => {
  const lockChanges = [];
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
    onPointerLockChange: (locked) => lockChanges.push(locked),
  });

  // Canvas click requests pointer lock when unlocked.
  env.canvas.dispatch('click');
  assert.equal(env.canvas.pointerLockRequests, 1);

  // Mousemove & mousedown ignored while unlocked.
  env.win.dispatch('mousemove', { movementX: 50, movementY: -20 });
  env.win.dispatch('mousedown', { button: 0 });
  let snap = bridge.getSnapshot();
  assert.equal(snap.mouse_dx, 0);
  assert.equal(snap.mouse_dy, 0);
  assert.equal(snap.rt, 0);
  assert.equal(snap.seq, 0);

  // Acquire pointer lock.
  env.lockPointer();
  assert.equal(bridge.isPointerLocked(), true);
  assert.deepEqual(lockChanges, [true]);

  // Clicking again while already locked should not re-request pointer lock.
  env.canvas.dispatch('click');
  assert.equal(env.canvas.pointerLockRequests, 1);

  // Accumulate multiple mousemove deltas.
  env.win.dispatch('mousemove', { movementX: 12, movementY: -5 });
  env.win.dispatch('mousemove', { movementX: 8, movementY: -3 });
  snap = bridge.getSnapshot();
  assert.equal(snap.mouse_dx, 20);
  assert.equal(snap.mouse_dy, -8);
  assert.equal(snap.seq, 4); // 2 updates x 2 seqlock bumps

  // Keyboard update MUST NOT clobber unconsumed mouse deltas.
  env.win.dispatch('keydown', { code: 'KeyW' });
  snap = bridge.getSnapshot();
  assert.equal(snap.mouse_dx, 20);
  assert.equal(snap.mouse_dy, -8);

  // Simulate C++ GameInputReader::consume_mouse_delta() atomic exchange(0).
  const consumedDx = Atomics.exchange(bridge.i32, 2, 0);
  const consumedDy = Atomics.exchange(bridge.i32, 3, 0);
  assert.equal(consumedDx, 20);
  assert.equal(consumedDy, -8);
  assert.equal(bridge.getSnapshot().mouse_dx, 0);
  assert.equal(bridge.getSnapshot().mouse_dy, 0);

  // Mouse buttons -> triggers & buttons (0: RT fire, 2: LT aim, 1: RS, 3: LB, 4: RB).
  env.win.dispatch('mousedown', { button: 0 });
  env.win.dispatch('mousedown', { button: 2 });
  env.win.dispatch('mousedown', { button: 1 });
  snap = bridge.getSnapshot();
  approxEqual(snap.rt, 1.0);
  approxEqual(snap.lt, 1.0);
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.RS);

  env.win.dispatch('mouseup', { button: 0 });
  snap = bridge.getSnapshot();
  approxEqual(snap.rt, 0.0);
  approxEqual(snap.lt, 1.0);

  // Scroll wheel tracking while pointer-locked.
  env.win.dispatch('wheel', { deltaY: 120 });
  env.win.dispatch('wheel', { deltaY: -40 });
  assert.equal(bridge.getSnapshot().wheelDelta, 80);
  assert.equal(bridge.consumeWheelDelta(), 80);
  assert.equal(bridge.consumeWheelDelta(), 0);

  // Losing pointer lock releases held mouse buttons/triggers automatically.
  env.unlockPointer();
  assert.equal(bridge.isPointerLocked(), false);
  assert.deepEqual(lockChanges, [true, false]);
  snap = bridge.getSnapshot();
  approxEqual(snap.lt, 0.0);
  approxEqual(snap.rt, 0.0);
  assert.equal(snap.buttons & INPUT_BUTTONS.RS, 0);

  bridge.destroy();
});

test('window blur and document visibilitychange clear stuck keys and mouse buttons', () => {
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  env.lockPointer();
  env.win.dispatch('keydown', { code: 'KeyW' });
  env.win.dispatch('keydown', { code: 'Space' });
  env.win.dispatch('mousedown', { button: 0 });

  let snap = bridge.getSnapshot();
  assert.ok(snap.buttons !== 0);
  assert.equal(snap.ly, 1);
  assert.equal(snap.rt, 1);

  // Window loses focus -> all keyboard keys & mouse triggers cleared.
  env.win.dispatch('blur');
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);
  assert.equal(snap.lt, 0);
  assert.equal(snap.rt, 0);

  // Hold keys again and hide document tab -> cleared via visibilitychange.
  env.win.dispatch('keydown', { code: 'KeyD' });
  env.win.dispatch('keydown', { code: 'KeyR' });
  assert.notEqual(bridge.getSnapshot().buttons, 0);

  env.doc.hidden = true;
  env.doc.visibilityState = 'hidden';
  env.doc.dispatch('visibilitychange');

  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);

  bridge.destroy();
});

test('deadzone math matches runtime/os/game_input.cpp normalization and clamps safely', () => {
  // Radial stick deadzone
  assert.deepEqual(applyDeadzone(0.1, -0.08, 0.15, 0.98), { x: 0, y: 0 });

  const diag = applyDeadzone(0.5, 0.5, 0.15, 0.98);
  approxEqual(diag.x, diag.y);
  const expectedMag = (Math.hypot(0.5, 0.5) - 0.15) / (0.98 - 0.15);
  approxEqual(Math.hypot(diag.x, diag.y), expectedMag);

  const clamped = applyDeadzone(2.0, 0.0, 0.15, 0.98);
  approxEqual(clamped.x, 1.0);
  approxEqual(clamped.y, 0.0);

  const clampedDiag = applyDeadzone(1.0, 1.0, 0.15, 0.98);
  approxEqual(Math.hypot(clampedDiag.x, clampedDiag.y), 1.0);
  approxEqual(clampedDiag.x, Math.SQRT1_2);
  approxEqual(clampedDiag.y, Math.SQRT1_2);

  assert.deepEqual(applyDeadzone(Number.NaN, 0.5), { x: 0, y: 0 });
  assert.deepEqual(applyDeadzone(0.5, Number.POSITIVE_INFINITY), { x: 0, y: 0 });

  // Axial deadzone
  assert.equal(applyAxialDeadzone(0.1, 0.15, 0.98), 0);
  approxEqual(applyAxialDeadzone(-0.565, 0.15, 0.98), -0.5);
  assert.equal(applyAxialDeadzone(1.5, 0.15, 0.98), 1);
  assert.equal(applyAxialDeadzone(Number.NaN), 0);

  // Trigger deadzone
  assert.equal(applyTriggerDeadzone(0.03, 0.05, 0.95), 0);
  approxEqual(applyTriggerDeadzone(0.5, 0.05, 0.95), 0.5);
  assert.equal(applyTriggerDeadzone(1.2, 0.05, 0.95), 1);
  assert.equal(applyTriggerDeadzone(Number.NaN), 0);
});

test('Standard Gamepad API maps all 17 buttons, analog triggers, sticks, and merges with keyboard/mouse', () => {
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  // Verify every W3C Standard Gamepad digital button index (0..5, 8..16).
  const digitalButtons = [
    [0, INPUT_BUTTONS.A],
    [1, INPUT_BUTTONS.B],
    [2, INPUT_BUTTONS.X],
    [3, INPUT_BUTTONS.Y],
    [4, INPUT_BUTTONS.LB],
    [5, INPUT_BUTTONS.RB],
    [8, INPUT_BUTTONS.VIEW],
    [9, INPUT_BUTTONS.MENU],
    [10, INPUT_BUTTONS.LS],
    [11, INPUT_BUTTONS.RS],
    [12, INPUT_BUTTONS.DPAD_UP],
    [13, INPUT_BUTTONS.DPAD_DOWN],
    [14, INPUT_BUTTONS.DPAD_LEFT],
    [15, INPUT_BUTTONS.DPAD_RIGHT],
    [16, INPUT_BUTTONS.GUIDE],
  ];

  for (const [idx, expectedBit] of digitalButtons) {
    env.setGamepads([makeStandardGamepad({ buttons: { [idx]: { pressed: true, value: 1 } } })]);
    assert.equal(bridge.pollGamepads(), true, `button ${idx} press should commit`);
    assert.equal(bridge.getSnapshot().buttons, expectedBit, `button ${idx} bitmask`);
  }

  // Test sticks (with W3C Y-down -> Xbox Y-up inversion) and analog triggers.
  env.setGamepads([
    makeStandardGamepad({
      buttons: {
        0: { pressed: true, value: 1 },
        6: { pressed: true, value: 0.515 },
        7: { pressed: true, value: 0.98 },
      },
      // Left stick: right & up (axes[1] = -0.98 -> ly = +1 after inversion & deadzone)
      // Right stick: inside deadzone -> (0, 0)
      axes: [0.0, -0.98, 0.05, -0.05],
    }),
  ]);

  assert.equal(bridge.pollGamepads(), true);
  let snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.A);
  approxEqual(snap.lx, 0);
  approxEqual(snap.ly, 1.0);
  approxEqual(snap.rx, 0);
  approxEqual(snap.ry, 0);
  approxEqual(snap.lt, applyTriggerDeadzone(0.515));
  approxEqual(snap.rt, 1.0);

  // Polling again with identical gamepad state must NOT bump seq.
  const seqSteady = snap.seq;
  assert.equal(bridge.pollGamepads(), false);
  assert.equal(bridge.getSnapshot().seq, seqSteady);

  // Merge check: keyboard buttons OR with gamepad buttons; active keyboard WASD overrides gamepad stick,
  // while zero keyboard axes yield to gamepad axes; triggers take max(mouse, gamepad).
  env.lockPointer();
  env.win.dispatch('keydown', { code: 'KeyE' }); // RB
  env.win.dispatch('keydown', { code: 'KeyA' }); // MOVE_LEFT -> keyLx = -1, keyLy = 0
  env.win.dispatch('mousedown', { button: 2 });  // Right click -> mouseLt = 1.0

  snap = bridge.getSnapshot();
  assert.equal(
    snap.buttons,
    INPUT_BUTTONS.A | INPUT_BUTTONS.RB | INPUT_BUTTONS.MOVE_LEFT
  );
  approxEqual(snap.lx, -1.0); // Keyboard A overrides zero gamepad X
  approxEqual(snap.ly, 1.0);  // Keyboard Y is 0 -> gamepad Y (+1.0) overrides zero keyboard Y
  approxEqual(snap.lt, 1.0);  // max(mouse 1.0, gamepad 0.5)
  approxEqual(snap.rt, 1.0);  // max(mouse 0.0, gamepad 1.0)

  // Release keyboard & mouse, then disconnect gamepad -> returns cleanly to zero.
  env.win.dispatch('keyup', { code: 'KeyE' });
  env.win.dispatch('keyup', { code: 'KeyA' });
  env.win.dispatch('mouseup', { button: 2 });
  env.setGamepads([null]);
  assert.equal(bridge.pollGamepads(), true);
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);
  assert.equal(snap.lt, 0);
  assert.equal(snap.rt, 0);

  bridge.destroy();
});

test('requestAnimationFrame polling loop start()/stop() and gamepadconnected auto-start', () => {
  const env = createMockEnv({ withRaf: true });
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  assert.equal(bridge.running, false);
  assert.equal(env.pendingRafCount(), 0);

  // Explicit start()/stop()
  bridge.start();
  assert.equal(bridge.running, true);
  assert.equal(env.pendingRafCount(), 1);

  env.setGamepads([makeStandardGamepad({ buttons: { 3: 1 } })]); // Y
  env.flushRaf();
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.Y);
  assert.equal(env.pendingRafCount(), 1); // Rescheduled next frame

  bridge.stop();
  assert.equal(bridge.running, false);
  assert.equal(env.pendingRafCount(), 0);

  // Auto-start on gamepadconnected and auto-stop on gamepaddisconnected
  env.setGamepads([makeStandardGamepad({ buttons: { 2: 1 } })]); // X
  env.win.dispatch('gamepadconnected');
  assert.equal(bridge.running, true);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.X);

  env.setGamepads([null]);
  env.win.dispatch('gamepaddisconnected');
  assert.equal(bridge.running, false);
  assert.equal(bridge.getSnapshot().buttons, 0);

  bridge.destroy();
});

test('destroy() / dispose() cleans up all event listeners, stops rAF loop, and zeros SAB', () => {
  const env = createMockEnv({ withRaf: true });
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
    autoStart: true,
  });

  assert.ok(env.canvas.listenerCount() > 0);
  assert.ok(env.doc.listenerCount() > 0);
  assert.ok(env.win.listenerCount() > 0);
  assert.equal(bridge.running, true);

  env.lockPointer();
  env.win.dispatch('keydown', { code: 'KeyW' });
  env.win.dispatch('mousemove', { movementX: 15, movementY: 9 });
  assert.notEqual(bridge.getSnapshot().buttons, 0);

  bridge.destroy();

  assert.equal(bridge.disposed, true);
  assert.equal(bridge.running, false);
  assert.equal(env.pendingRafCount(), 0);
  assert.equal(env.canvas.listenerCount(), 0);
  assert.equal(env.doc.listenerCount(), 0);
  assert.equal(env.win.listenerCount(), 0);

  const zeroed = bridge.getSnapshot();
  assert.equal(zeroed.buttons, 0);
  assert.equal(zeroed.mouse_dx, 0);
  assert.equal(zeroed.mouse_dy, 0);
  assert.equal(zeroed.lx, 0);
  assert.equal(zeroed.ly, 0);
  const seqAfterDestroy = zeroed.seq;

  // Subsequent events or polls are ignored and dispose() is idempotent.
  env.win.dispatch('keydown', { code: 'Space' });
  env.win.dispatch('mousemove', { movementX: 100, movementY: 100 });
  assert.equal(bridge.pollGamepads(), false);
  bridge.dispose();
  assert.equal(bridge.getSnapshot().seq, seqAfterDestroy);
  assert.equal(bridge.getSnapshot().buttons, 0);
});

test('menu bindings (Enter/Space -> A, Escape/Backspace -> B, Tab -> View) respect browser shortcuts, editable targets, and focus', () => {
  const env = createMockEnv();
  let docFocused = true;
  env.doc.hasFocus = () => docFocused;
  env.doc.activeElement = null;

  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  // When unlocked, browser shortcuts (Ctrl+R, Ctrl+W, Alt+Tab, Meta+R) are NOT hijacked.
  let prevented = false;
  const preventDefault = () => {
    prevented = true;
  };
  env.win.dispatch('keydown', { code: 'KeyR', ctrlKey: true, preventDefault });
  env.win.dispatch('keydown', { code: 'KeyW', ctrlKey: true, preventDefault });
  env.win.dispatch('keydown', { code: 'Tab', altKey: true, preventDefault });
  env.win.dispatch('keydown', { code: 'Enter', metaKey: true, preventDefault });
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getSnapshot().ly, 0);
  assert.equal(prevented, false);

  // Typing inside an <input>/<textarea>/contentEditable does NOT trigger game inputs when unlocked.
  env.win.dispatch('keydown', { code: 'Enter', target: { tagName: 'INPUT' }, preventDefault });
  env.win.dispatch('keydown', { code: 'Space', target: { tagName: 'TEXTAREA' }, preventDefault });
  env.win.dispatch('keydown', { code: 'Backspace', target: { isContentEditable: true }, preventDefault });
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(prevented, false);

  // Unfocused document ignores keydown until focus returns.
  docFocused = false;
  env.win.dispatch('keydown', { code: 'Enter', preventDefault });
  env.win.dispatch('keydown', { code: 'Escape', preventDefault });
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(prevented, false);

  // Focused unlocked page accepts Enter/Space -> A, Escape/Backspace -> B, Tab -> View without calling preventDefault.
  docFocused = true;
  env.win.dispatch('keydown', { code: 'Enter', preventDefault });
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(prevented, false, 'should not fight browser defaults when pointer is unlocked');
  env.win.dispatch('keyup', { code: 'Enter' });

  env.win.dispatch('keydown', { code: 'Escape', preventDefault });
  assert.equal(bridge.getSnapshot().buttons & INPUT_BUTTONS.B, INPUT_BUTTONS.B);
  assert.equal(bridge.getSnapshot().buttons & INPUT_BUTTONS.MENU, INPUT_BUTTONS.MENU);
  env.win.dispatch('keyup', { code: 'Escape' });

  env.win.dispatch('keydown', { code: 'Backspace', preventDefault });
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.B);
  env.win.dispatch('keyup', { code: 'Backspace' });

  // When pointer-locked, crouch-walk (ControlLeft + KeyW) and preventDefault on Enter/Backspace work.
  env.lockPointer();
  env.win.dispatch('keydown', { code: 'ControlLeft', ctrlKey: true });
  env.win.dispatch('keydown', { code: 'KeyW', ctrlKey: true });
  let snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.B | INPUT_BUTTONS.MOVE_FORWARD);
  approxEqual(snap.ly, 1.0);

  env.win.dispatch('keydown', { code: 'Enter', preventDefault });
  assert.equal(prevented, true);

  // pagehide clears all stuck keys cleanly as well.
  env.win.dispatch('pagehide');
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.ly, 0);

  bridge.destroy();
});

test('headless safety: never throws when SharedArrayBuffer, Atomics, or navigator.getGamepads is missing or throws', () => {
  // 1. Missing SharedArrayBuffer and missing Atomics -> falls back to 40-byte ArrayBuffer cleanly.
  const env = createMockEnv();
  const throwingNav = {
    getGamepads() {
      throw new Error('Permissions policy blocked Gamepad API');
    },
  };
  const bridgeNoSab = new InputBridge({
    canvas: env.canvas,
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: throwingNav,
    sharedArrayBufferCtor: null,
    atomicsObj: null,
  });

  assert.equal(bridgeNoSab.isSharedBuffer, false);
  assert.equal(bridgeNoSab.usesAtomics, false);
  assert.ok(bridgeNoSab.sab instanceof ArrayBuffer);
  assert.equal(bridgeNoSab.sab.byteLength, INPUT_STATE_BYTES);

  // Keyboard, mouse, gamepad polling, reset, and snapshot all work without throwing.
  env.lockPointer();
  env.win.dispatch('keydown', { code: 'Enter' });
  env.win.dispatch('keydown', { code: 'KeyW' });
  env.win.dispatch('mousemove', { movementX: 6, movementY: -4 });
  assert.equal(bridgeNoSab.pollGamepads(), false);

  const snap = bridgeNoSab.getSnapshot();
  assert.ok(snap !== null);
  assert.ok(snap.seq > 0 && (snap.seq & 1) === 0);
  assert.equal(snap.buttons, INPUT_BUTTONS.A | INPUT_BUTTONS.MOVE_FORWARD);
  assert.equal(snap.mouse_dx, 6);
  assert.equal(snap.mouse_dy, -4);
  approxEqual(snap.ly, 1.0);

  // Verify raw 40-byte DataView layout matches playgta6::InputState even in fallback mode.
  const dv = new DataView(bridgeNoSab.sab);
  assert.equal(dv.getUint32(INPUT_OFFSETS.SEQ, true), snap.seq);
  assert.equal(dv.getUint32(INPUT_OFFSETS.BUTTONS, true), INPUT_BUTTONS.A | INPUT_BUTTONS.MOVE_FORWARD);
  assert.equal(dv.getInt32(INPUT_OFFSETS.MOUSE_DX, true), 6);
  assert.equal(dv.getInt32(INPUT_OFFSETS.MOUSE_DY, true), -4);
  approxEqual(dv.getFloat32(INPUT_OFFSETS.LEFT_STICK_Y, true), 1.0);

  // blur clears stuck keys even without SharedArrayBuffer/Atomics.
  env.win.dispatch('blur');
  assert.equal(bridgeNoSab.getSnapshot().buttons, 0);
  assert.equal(bridgeNoSab.getSnapshot().ly, 0);
  bridgeNoSab.destroy();

  // 2. SharedArrayBuffer constructor throws (e.g., SecurityError) and navigator has no getGamepads.
  class ThrowingSab {
    constructor() {
      throw new Error('SharedArrayBuffer unavailable without COOP/COEP');
    }
  }
  const bridgeThrowingSab = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: {},
    sharedArrayBufferCtor: ThrowingSab,
  });
  assert.equal(bridgeThrowingSab.isSharedBuffer, false);
  assert.equal(bridgeThrowingSab.sab.byteLength, INPUT_STATE_BYTES);
  assert.equal(bridgeThrowingSab.pollGamepads(), false);
  assert.equal(bridgeThrowingSab.getSnapshot().seq, 0);
  bridgeThrowingSab.destroy();

  // 3. Undersized ArrayBuffer or non-buffer opts.sab still throws RangeError.
  assert.throws(() => new InputBridge({ sab: new ArrayBuffer(20) }), RangeError);
  assert.throws(() => new InputBridge({ sab: {} }), RangeError);
});

test('zero-alloc deadzone and snapshot reuse + steady-state gamepad polling avoids spurious commits', () => {
  const scratchStick = { x: 99, y: 99 };
  const returned = applyDeadzone(0.5, 0.5, 0.15, 0.98, scratchStick);
  assert.equal(returned, scratchStick, 'applyDeadzone with out param must reuse object in-place');
  assert.ok(scratchStick.x > 0 && scratchStick.y > 0);

  const zeroed = applyDeadzone(0.05, -0.05, 0.15, 0.98, scratchStick);
  assert.equal(zeroed, scratchStick);
  assert.equal(scratchStick.x, 0);
  assert.equal(scratchStick.y, 0);

  const env = createMockEnv({
    gamepads: [
      makeStandardGamepad({
        buttons: { 0: 1, 6: 0.4, 7: 0.8 },
        axes: [0.6, -0.4, -0.5, 0.5],
      }),
    ],
  });
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  assert.equal(bridge.pollGamepads(), true);
  const reusableSnap = {};
  const snap1 = bridge.getSnapshot(64, reusableSnap);
  assert.equal(snap1, reusableSnap, 'getSnapshot with out param must reuse object in-place');
  assert.equal(snap1.buttons, INPUT_BUTTONS.A);
  const seq1 = snap1.seq;

  // 100 consecutive identical polls must not bump seq or fail.
  for (let i = 0; i < 100; i++) {
    assert.equal(bridge.pollGamepads(), false);
  }
  const snap2 = bridge.getSnapshot(64, reusableSnap);
  assert.equal(snap2, reusableSnap);
  assert.equal(snap2.seq, seq1);

  bridge.destroy();
});

function createFakeClockAndTimers(initialNow = 0) {
  let now = initialNow;
  let nextId = 1;
  const timeouts = new Map();
  const intervals = new Map();

  function setTimeoutFn(cb, delay = 0) {
    const id = nextId++;
    timeouts.set(id, { id, cb, due: now + Math.max(0, delay) });
    return id;
  }

  function clearTimeoutFn(id) {
    timeouts.delete(id);
  }

  function setIntervalFn(cb, period = 16) {
    const id = nextId++;
    const step = Math.max(1, period);
    intervals.set(id, { id, cb, period: step, due: now + step });
    return id;
  }

  function clearIntervalFn(id) {
    intervals.delete(id);
  }

  function advanceTo(targetMs) {
    assert.ok(targetMs >= now, `Cannot move fake clock backwards (${now} -> ${targetMs})`);
    while (true) {
      let nextDue = Infinity;
      let kind = null;
      let entry = null;

      for (const item of timeouts.values()) {
        if (item.due <= targetMs && item.due < nextDue) {
          nextDue = item.due;
          kind = 'timeout';
          entry = item;
        }
      }
      for (const item of intervals.values()) {
        if (item.due <= targetMs && item.due < nextDue) {
          nextDue = item.due;
          kind = 'interval';
          entry = item;
        }
      }

      if (!entry) break;
      now = nextDue;
      if (kind === 'timeout') {
        timeouts.delete(entry.id);
        entry.cb();
      } else {
        entry.due += entry.period;
        entry.cb();
      }
    }
    now = targetMs;
  }

  return {
    nowFn: () => now,
    setTimeoutFn,
    clearTimeoutFn,
    setIntervalFn,
    clearIntervalFn,
    advanceTo,
    advanceBy(deltaMs) {
      advanceTo(now + deltaMs);
    },
    pendingTimeouts: () => timeouts.size,
    pendingIntervals: () => intervals.size,
  };
}

test('INPUT_SCRIPT_PRESETS["start-screen"] matches tools/run_native_boot.py and exact sampleAt() boundaries', () => {
  const preset = INPUT_SCRIPT_PRESETS['start-screen'];
  assert.ok(preset);
  assert.equal(preset.button, INPUT_BUTTONS.A);
  assert.equal(preset.startMs, 5000);
  assert.equal(preset.holdMs, 100);
  assert.equal(preset.periodMs, 1000);

  // Verify parity with tools/run_native_boot.py constants.
  const nativeBootPy = fs.readFileSync(
    path.join(REPO_ROOT, 'tools/run_native_boot.py'),
    'utf8'
  );
  assert.match(nativeBootPy, /INPUT_START_MS\s*=\s*5_000/);
  assert.match(nativeBootPy, /INPUT_PERIOD_MS\s*=\s*1_000/);
  assert.match(nativeBootPy, /INPUT_HOLD_MS\s*=\s*100/);

  const player = new ScriptedInputPlayer('start-screen');
  assert.equal(player.preset, 'start-screen');

  // Exact boundary table:
  // t = 0..4999 -> 0
  // t = 5000..5099 -> BUTTON_A
  // t = 5100..5999 -> 0
  // t = 6000..6099 -> BUTTON_A
  // t = 6100..6999 -> 0
  const cases = [
    [-10, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [4999, 0, 0, 0],
    [4999.9, 0, 0, 0],
    [5000, INPUT_BUTTONS.A, 1, 1],
    [5050, INPUT_BUTTONS.A, 1, 1],
    [5099, INPUT_BUTTONS.A, 1, 1],
    [5099.9, INPUT_BUTTONS.A, 1, 1],
    [5100, 0, 0, 1],
    [5500, 0, 0, 1],
    [5999, 0, 0, 1],
    [5999.9, 0, 0, 1],
    [6000, INPUT_BUTTONS.A, 1, 2],
    [6050, INPUT_BUTTONS.A, 1, 2],
    [6099, INPUT_BUTTONS.A, 1, 2],
    [6100, 0, 0, 2],
    [120000, INPUT_BUTTONS.A, 1, 116],
    [120100, 0, 0, 116],
  ];

  for (const [t, expectedMask, expectedActiveCount, expectedPressCount] of cases) {
    assert.deepEqual(
      player.sampleAt(t),
      { buttonsMask: expectedMask, activeEventCount: expectedActiveCount },
      `sampleAt(${t})`
    );
    assert.equal(player.pressCountAt(t), expectedPressCount, `pressCountAt(${t})`);
  }

  // Optional maxPresses and endMs caps on 'start-screen'.
  const cappedByCount = new ScriptedInputPlayer('start-screen', { maxPresses: 2 });
  assert.equal(cappedByCount.sampleAt(5000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(cappedByCount.sampleAt(6000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(cappedByCount.sampleAt(6100).buttonsMask, 0);
  assert.equal(cappedByCount.sampleAt(7000).buttonsMask, 0);
  assert.equal(cappedByCount.pressCountAt(9000), 2);
  assert.equal(cappedByCount.nextTransitionMs(6100), null);

  const cappedByEnd = new ScriptedInputPlayer('start-screen', { endMs: 6050 });
  assert.equal(cappedByEnd.sampleAt(5000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(cappedByEnd.sampleAt(6049).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(cappedByEnd.sampleAt(6050).buttonsMask, 0);
  assert.equal(cappedByEnd.sampleAt(7000).buttonsMask, 0);
  assert.equal(cappedByEnd.nextTransitionMs(6000), 6050);
  assert.equal(cappedByEnd.nextTransitionMs(6050), null);
});

test('ScriptedInputPlayer supports custom multi-event schedules, simultaneous buttons, sticks, maxPresses, and validates inputs', () => {
  // Unsorted multi-event schedule with simultaneous overlapping presses, string button names,
  // bitmasks, and analog sticks.
  const events = [
    { atMs: 250, holdMs: 100, button: ['B', 'MENU'] },
    { atMs: 100, holdMs: 200, buttonsMask: INPUT_BUTTONS.A | INPUT_BUTTONS.RB, sticks: { lx: 0.75, ly: -0.5 } },
    { atMs: 400, holdMs: 50, button: 'BUTTON_X' },
  ];
  const player = new ScriptedInputPlayer(events);
  assert.equal(player.preset, null);

  assert.deepEqual(player.sampleAt(0), { buttonsMask: 0, activeEventCount: 0 });
  assert.deepEqual(player.sampleAt(99), { buttonsMask: 0, activeEventCount: 0 });

  assert.deepEqual(player.sampleAt(100), {
    buttonsMask: INPUT_BUTTONS.A | INPUT_BUTTONS.RB,
    activeEventCount: 1,
    sticks: { lx: 0.75, ly: -0.5, rx: 0, ry: 0 },
  });

  // Overlap window [250, 300): both events active simultaneously -> OR'd buttons, activeEventCount = 2.
  assert.deepEqual(player.sampleAt(250), {
    buttonsMask: INPUT_BUTTONS.A | INPUT_BUTTONS.RB | INPUT_BUTTONS.B | INPUT_BUTTONS.MENU,
    activeEventCount: 2,
    sticks: { lx: 0.75, ly: -0.5, rx: 0, ry: 0 },
  });
  assert.deepEqual(player.sampleAt(299), {
    buttonsMask: INPUT_BUTTONS.A | INPUT_BUTTONS.RB | INPUT_BUTTONS.B | INPUT_BUTTONS.MENU,
    activeEventCount: 2,
    sticks: { lx: 0.75, ly: -0.5, rx: 0, ry: 0 },
  });

  // [300, 350): first event released, second event still held.
  assert.deepEqual(player.sampleAt(300), {
    buttonsMask: INPUT_BUTTONS.B | INPUT_BUTTONS.MENU,
    activeEventCount: 1,
  });
  assert.deepEqual(player.sampleAt(350), { buttonsMask: 0, activeEventCount: 0 });

  // [400, 450): third event active.
  assert.deepEqual(player.sampleAt(400), { buttonsMask: INPUT_BUTTONS.X, activeEventCount: 1 });
  assert.deepEqual(player.sampleAt(450), { buttonsMask: 0, activeEventCount: 0 });
  assert.equal(player.pressCountAt(500), 3);

  // maxPresses caps explicit event schedules after sorting by atMs.
  const capped = new ScriptedInputPlayer({ events, maxPresses: 2 });
  assert.equal(capped.sampleAt(100).buttonsMask, INPUT_BUTTONS.A | INPUT_BUTTONS.RB);
  assert.equal(capped.sampleAt(300).buttonsMask, INPUT_BUTTONS.B | INPUT_BUTTONS.MENU);
  assert.equal(capped.sampleAt(400).buttonsMask, 0);
  assert.equal(capped.pressCountAt(500), 2);

  // Repeating rule with overlapping holdMs > periodMs and maxPresses.
  const overlappingRepeat = new ScriptedInputPlayer({
    button: 'Y',
    startMs: 100,
    holdMs: 250,
    periodMs: 100,
    maxPresses: 3,
  });
  assert.deepEqual(overlappingRepeat.sampleAt(100), { buttonsMask: INPUT_BUTTONS.Y, activeEventCount: 1 });
  assert.deepEqual(overlappingRepeat.sampleAt(200), { buttonsMask: INPUT_BUTTONS.Y, activeEventCount: 2 });
  assert.deepEqual(overlappingRepeat.sampleAt(300), { buttonsMask: INPUT_BUTTONS.Y, activeEventCount: 3 });
  assert.deepEqual(overlappingRepeat.sampleAt(350), { buttonsMask: INPUT_BUTTONS.Y, activeEventCount: 2 });
  assert.deepEqual(overlappingRepeat.sampleAt(549), { buttonsMask: INPUT_BUTTONS.Y, activeEventCount: 1 });
  assert.deepEqual(overlappingRepeat.sampleAt(550), { buttonsMask: 0, activeEventCount: 0 });

  // Validation errors.
  assert.throws(() => new ScriptedInputPlayer('does-not-exist'), RangeError);
  assert.throws(() => new ScriptedInputPlayer({ button: 'INVALID', startMs: 0, holdMs: 50, periodMs: 100 }), RangeError);
  assert.throws(() => new ScriptedInputPlayer({ button: INPUT_BUTTONS.A, startMs: -1, holdMs: 50, periodMs: 100 }), RangeError);
  assert.throws(() => new ScriptedInputPlayer({ button: INPUT_BUTTONS.A, startMs: 0, holdMs: 0, periodMs: 100 }), RangeError);
  assert.throws(() => new ScriptedInputPlayer({ button: INPUT_BUTTONS.A, startMs: 0, holdMs: 50, periodMs: 0 }), RangeError);
  assert.throws(() => new ScriptedInputPlayer(null), TypeError);
});

test('InputBridge scripted playback writes to 40-byte SAB and merges cleanly with live keyboard/gamepad inputs', () => {
  const env = createMockEnv();
  const clock = createFakeClockAndTimers(1000);
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  // Before starting any script.
  assert.deepEqual(bridge.getScriptState(), {
    active: false,
    preset: null,
    activePreset: null,
    elapsedMs: 0,
    elapsedScriptMs: 0,
    pressCount: 0,
    scriptedPressesSent: 0,
    lastScriptedButton: null,
    currentMask: 0,
  });

  const player = bridge.startScript('start-screen', {
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });
  assert.ok(player instanceof ScriptedInputPlayer);
  assert.equal(clock.pendingTimeouts(), 1);

  // t = 0..4999: no scripted button yet, seq untouched.
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getSnapshot().seq, 0);
  assert.deepEqual(bridge.getScriptState(), {
    active: true,
    preset: 'start-screen',
    activePreset: 'start-screen',
    elapsedMs: 0,
    elapsedScriptMs: 0,
    pressCount: 0,
    scriptedPressesSent: 0,
    lastScriptedButton: null,
    currentMask: 0,
  });

  clock.advanceBy(4999);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getSnapshot().seq, 0);

  // t = 5000: first scripted A press fires into 40-byte SAB with valid even seqlock.
  clock.advanceBy(1);
  let snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.A);
  assert.equal(snap.seq, 2);
  const dv = new DataView(bridge.sab);
  assert.equal(dv.getUint32(INPUT_OFFSETS.BUTTONS, true), INPUT_BUTTONS.A);
  assert.deepEqual(bridge.getScriptState(), {
    active: true,
    preset: 'start-screen',
    activePreset: 'start-screen',
    elapsedMs: 5000,
    elapsedScriptMs: 5000,
    pressCount: 1,
    scriptedPressesSent: 1,
    lastScriptedButton: 'A',
    currentMask: INPUT_BUTTONS.A,
  });

  // Concurrent live keyboard (KeyW -> MOVE_FORWARD) + gamepad (B) merge via OR without clobbering.
  env.win.dispatch('keydown', { code: 'KeyW' });
  env.setGamepads([makeStandardGamepad({ buttons: { 1: 1 } })]); // B
  bridge.pollGamepads();
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.A | INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.B);
  approxEqual(snap.ly, 1.0);

  // t = 5100: scripted A releases, live keyboard + gamepad buttons remain intact!
  clock.advanceBy(100);
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.B);
  approxEqual(snap.ly, 1.0);
  assert.deepEqual(bridge.getScriptState(), {
    active: true,
    preset: 'start-screen',
    activePreset: 'start-screen',
    elapsedMs: 5100,
    elapsedScriptMs: 5100,
    pressCount: 1,
    scriptedPressesSent: 1,
    lastScriptedButton: 'A',
    currentMask: 0,
  });

  // Live Enter (also maps to INPUT_BUTTONS.A) stays held even when scripted A releases at t = 6100.
  env.win.dispatch('keydown', { code: 'Enter' });
  clock.advanceBy(900); // t = 6000 (scripted A pressed)
  assert.equal(bridge.getScriptState().pressCount, 2);
  assert.equal(bridge.getScriptState().currentMask, INPUT_BUTTONS.A);
  assert.equal(
    bridge.getSnapshot().buttons,
    INPUT_BUTTONS.A | INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.B
  );

  clock.advanceBy(100); // t = 6100 (scripted A released, Enter still held)
  assert.equal(bridge.getScriptState().currentMask, 0);
  assert.equal(
    bridge.getSnapshot().buttons,
    INPUT_BUTTONS.A | INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.B
  );

  // Release live Enter and gamepad; keep KeyW held.
  env.win.dispatch('keyup', { code: 'Enter' });
  env.setGamepads([null]);
  bridge.pollGamepads();
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.MOVE_FORWARD);

  // Advance to t = 7050 (mid-press of 3rd scripted A) and call stopScript():
  // clears scripted A immediately from SAB, cancels timer, preserves live KeyW!
  clock.advanceBy(950); // t = 7050
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.A);
  assert.equal(bridge.getScriptState().pressCount, 3);

  bridge.stopScript();
  assert.equal(clock.pendingTimeouts(), 0);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.MOVE_FORWARD);
  assert.deepEqual(bridge.getScriptState(), {
    active: false,
    preset: 'start-screen',
    activePreset: null,
    elapsedMs: 7050,
    elapsedScriptMs: 7050,
    pressCount: 3,
    scriptedPressesSent: 3,
    lastScriptedButton: 'A',
    currentMask: 0,
  });

  // Advancing clock further must NOT re-trigger stopped script.
  clock.advanceBy(5000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.MOVE_FORWARD);

  bridge.destroy();
});

test('InputBridge scripted playback supports interval mode, scripted sticks, auto-completion, blur resilience, and destroy()/detach() cleanup', () => {
  const env = createMockEnv();
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  // 1. Interval-driven playback with finite maxPresses auto-stops & cleans up interval.
  let completedState = null;
  bridge.startScript(
    {
      button: INPUT_BUTTONS.A,
      startMs: 100,
      holdMs: 50,
      periodMs: 200,
      maxPresses: 2,
      sticks: { lx: 0.5, ly: -0.25 },
    },
    {
      nowFn: clock.nowFn,
      setIntervalFn: clock.setIntervalFn,
      clearIntervalFn: clock.clearIntervalFn,
      tickMs: 25,
      onComplete: (state) => {
        completedState = state;
      },
    }
  );
  assert.equal(clock.pendingIntervals(), 1);

  clock.advanceTo(100);
  let snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.A);
  approxEqual(snap.lx, 0.5);
  approxEqual(snap.ly, -0.25);

  // Live WASD overrides scripted sticks while held, then yields back to scripted sticks on release.
  env.win.dispatch('keydown', { code: 'KeyA' }); // keyLx = -1
  snap = bridge.getSnapshot();
  approxEqual(snap.lx, -1.0);
  approxEqual(snap.ly, -0.25);

  // Window blur clears stuck human keyboard key without stopping active scripted playback.
  env.win.dispatch('blur');
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.A);
  approxEqual(snap.lx, 0.5);
  approxEqual(snap.ly, -0.25);

  // Release after press 1 (t = 150), fire press 2 (t = 300), finish at t = 350 -> auto-stops.
  clock.advanceTo(150);
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  approxEqual(snap.lx, 0);
  approxEqual(snap.ly, 0);

  clock.advanceTo(300);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  clock.advanceTo(350);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(clock.pendingIntervals(), 0);
  assert.equal(bridge.getScriptState().active, false);
  assert.equal(bridge.getScriptState().pressCount, 2);
  assert.ok(completedState !== null && completedState.active === false && completedState.pressCount === 2);

  // 2. destroy() while a script is actively holding a button cancels timers and zeros SAB.
  bridge.startScript(
    [{ atMs: 0, holdMs: 500, button: INPUT_BUTTONS.MENU }],
    {
      nowFn: clock.nowFn,
      setTimeoutFn: clock.setTimeoutFn,
      clearTimeoutFn: clock.clearTimeoutFn,
    }
  );
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.MENU);
  assert.equal(clock.pendingTimeouts(), 1);

  bridge.destroy();
  assert.equal(clock.pendingTimeouts(), 0);
  assert.equal(bridge.getScriptState().active, false);
  assert.equal(bridge.getScriptState().currentMask, 0);
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Starting a script after destroy() is safely ignored.
  assert.equal(bridge.startScript('start-screen'), null);
  assert.equal(clock.pendingTimeouts(), 0);

  // 3. detach() alias also stops active scripts and cleans up timers in headless mode without SAB/timers.
  const headlessClock = createFakeClockAndTimers(0);
  const headlessBridge = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
    sharedArrayBufferCtor: null,
    atomicsObj: null,
  });
  headlessBridge.startScript('start-screen', {
    nowFn: headlessClock.nowFn,
    setTimeoutFn: headlessClock.setTimeoutFn,
    clearTimeoutFn: headlessClock.clearTimeoutFn,
  });
  headlessClock.advanceTo(5000);
  assert.equal(headlessBridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  headlessBridge.detach();
  assert.equal(headlessClock.pendingTimeouts(), 0);
  assert.equal(headlessBridge.disposed, true);
  assert.equal(headlessBridge.getScriptState().active, false);
  assert.equal(headlessBridge.getSnapshot().buttons, 0);
});

test('W174 INPUT_SCRIPT_PRESETS and INPUT_SCRIPT_PRESET_NAMES expose frozen built-in presets cleanly', () => {
  assert.equal(Object.isFrozen(INPUT_SCRIPT_PRESETS), true);
  assert.equal(Object.isFrozen(INPUT_SCRIPT_PRESET_NAMES), true);
  assert.deepEqual(INPUT_SCRIPT_PRESET_NAMES, [
    'start-screen',
    'menu-flow',
    'calibration-skip',
    'intro-watch',
    'legal-notices',
    'chrome-menu-flow',
    'chrome-full-boot',
    'menu-to-ingame',
    'chrome-menu-flow-2',
    'chrome-menu-flow-4',
    'chrome-menu-flow-5',
    'chrome-to-ingame',
  ]);
  assert.deepEqual(Object.keys(INPUT_SCRIPT_PRESETS), [...INPUT_SCRIPT_PRESET_NAMES]);

  for (const name of INPUT_SCRIPT_PRESET_NAMES) {
    const spec = INPUT_SCRIPT_PRESETS[name];
    assert.ok(spec && typeof spec === 'object', `Preset ${name} must exist`);
    assert.equal(Object.isFrozen(spec), true, `Preset ${name} must be frozen`);
    assert.equal(spec.name, name);
    assert.match(name, /^[a-z0-9][a-z0-9_-]{0,63}$/);
    if (Array.isArray(spec.events)) {
      assert.equal(Object.isFrozen(spec.events), true, `Preset ${name} events must be frozen`);
      for (const ev of spec.events) {
        assert.equal(Object.isFrozen(ev), true, `Preset ${name} event must be frozen`);
        if (ev.sticks) {
          assert.equal(Object.isFrozen(ev.sticks), true, `Preset ${name} event.sticks must be frozen`);
        }
        if (ev.triggers) {
          assert.equal(Object.isFrozen(ev.triggers), true, `Preset ${name} event.triggers must be frozen`);
        }
      }
    }
    if (spec.loop) {
      assert.equal(Object.isFrozen(spec.loop), true, `Preset ${name} loop must be frozen`);
      assert.equal(Object.isFrozen(spec.loop.events), true, `Preset ${name} loop.events must be frozen`);
      for (const ev of spec.loop.events) {
        assert.equal(Object.isFrozen(ev), true, `Preset ${name} loop event must be frozen`);
      }
    }
    const player = new ScriptedInputPlayer(name);
    assert.equal(player.preset, name);
  }
});

test('W174 menu-flow preset advances Start Screen -> CONTINUE -> UI TEXT SIZE -> calibration -> periodic Main Menu probes without pressing B', () => {
  const player = new ScriptedInputPlayer('menu-flow');
  assert.equal(player.preset, 'menu-flow');

  // 0..4999 ms: silent during initial boot settle.
  for (const t of [0, 1000, 4999]) {
    assert.deepEqual(player.sampleAt(t), { buttonsMask: 0, activeEventCount: 0 }, `idle at ${t}`);
    assert.equal(player.pressCountAt(t), 0);
  }

  // Phase 1: Start Screen periodic A presses (5000, 6500, 8000).
  for (const [idx, atMs] of [5000, 6500, 8000].entries()) {
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `Start Screen A at ${atMs}`);
    assert.equal(player.sampleAt(atMs + 99).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(atMs + 100).buttonsMask, 0);
    assert.equal(player.pressCountAt(atMs), idx + 1);
  }

  // Clean pause across Start Screen -> CONTINUE (8100..11499 -> 3.4 s quiet window).
  for (let t = 8100; t < 11500; t += 100) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `pause before CONTINUE at ${t}`);
  }
  assert.equal(player.sampleAt(11500).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(11600).buttonsMask, 0);
  assert.equal(player.pressCountAt(11500), 4);

  // Clean pause across CONTINUE -> UI TEXT SIZE (11600..14499 -> 2.9 s quiet window).
  for (let t = 11600; t < 14500; t += 100) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `pause before UI TEXT SIZE at ${t}`);
  }
  assert.equal(player.sampleAt(14500).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(14600).buttonsMask, 0);
  assert.equal(player.pressCountAt(14500), 5);

  // Phase 3: Spaced calibration screen acceptances (17500, 20500, 23500).
  for (const [idx, atMs] of [17500, 20500, 23500].entries()) {
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0);
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `calibration A at ${atMs}`);
    assert.equal(player.sampleAt(atMs + 100).buttonsMask, 0);
    assert.equal(player.pressCountAt(atMs), 6 + idx);
  }

  // Clean settle pause before Main Menu navigation probes (23600..26999).
  for (let t = 23600; t < 27000; t += 100) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `pause before Main Menu at ${t}`);
  }

  // Phase 4: Periodic Main Menu probes (DPAD_DOWN -> DPAD_UP -> A every 8000 ms starting at 27000).
  for (let cycle = 0; cycle < 4; cycle++) {
    const base = 27000 + cycle * 8000;
    assert.equal(player.sampleAt(base).buttonsMask, INPUT_BUTTONS.DPAD_DOWN, `cycle ${cycle} down`);
    assert.equal(player.sampleAt(base + 100).buttonsMask, 0);
    assert.equal(player.pressCountAt(base), 8 + cycle * 3 + 1);

    assert.equal(player.sampleAt(base + 2000).buttonsMask, INPUT_BUTTONS.DPAD_UP, `cycle ${cycle} up`);
    assert.equal(player.sampleAt(base + 2100).buttonsMask, 0);
    assert.equal(player.pressCountAt(base + 2000), 8 + cycle * 3 + 2);

    assert.equal(player.sampleAt(base + 4000).buttonsMask, INPUT_BUTTONS.A, `cycle ${cycle} A`);
    assert.equal(player.sampleAt(base + 4100).buttonsMask, 0);
    assert.equal(player.pressCountAt(base + 4000), 8 + cycle * 3 + 3);
  }

  // Invariant: menu-flow NEVER presses B anywhere in 120 seconds of sampling.
  for (let t = 0; t <= 120000; t += 50) {
    const { buttonsMask } = player.sampleAt(t);
    assert.equal(buttonsMask & INPUT_BUTTONS.B, 0, `menu-flow must never press B at t=${t}`);
  }

  // Exact nextTransitionMs across fixed -> loop boundary and repeated cycles.
  assert.equal(player.nextTransitionMs(0), 5000);
  assert.equal(player.nextTransitionMs(23500), 23600);
  assert.equal(player.nextTransitionMs(23600), 27000);
  assert.equal(player.nextTransitionMs(27000), 27100);
  assert.equal(player.nextTransitionMs(27100), 29000);
  assert.equal(player.nextTransitionMs(31100), 35000);

  // maxPresses caps both fixed-only and loop-entering counts cleanly.
  const cappedFixed = new ScriptedInputPlayer({ preset: 'menu-flow', maxPresses: 5 });
  assert.equal(cappedFixed.preset, 'menu-flow');
  assert.equal(cappedFixed.sampleAt(14500).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(cappedFixed.sampleAt(17500).buttonsMask, 0);
  assert.equal(cappedFixed.sampleAt(27000).buttonsMask, 0);
  assert.equal(cappedFixed.pressCountAt(60000), 5);
  assert.equal(cappedFixed.nextTransitionMs(14600), null);

  const cappedLoop = new ScriptedInputPlayer('menu-flow', { maxPresses: 10 });
  assert.equal(cappedLoop.sampleAt(27000).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(cappedLoop.sampleAt(29000).buttonsMask, INPUT_BUTTONS.DPAD_UP);
  assert.equal(cappedLoop.sampleAt(31000).buttonsMask, 0);
  assert.equal(cappedLoop.pressCountAt(100000), 10);
  assert.equal(cappedLoop.nextTransitionMs(29100), null);

  // endMs truncates mid-loop press cleanly.
  const cappedEnd = new ScriptedInputPlayer('menu-flow', { endMs: 29050 });
  assert.equal(cappedEnd.sampleAt(29000).buttonsMask, INPUT_BUTTONS.DPAD_UP);
  assert.equal(cappedEnd.sampleAt(29049).buttonsMask, INPUT_BUTTONS.DPAD_UP);
  assert.equal(cappedEnd.sampleAt(29050).buttonsMask, 0);
  assert.equal(cappedEnd.nextTransitionMs(29000), 29050);
  assert.equal(cappedEnd.nextTransitionMs(29050), null);
});

test('W174 calibration-skip preset sends steady 2.5 s spaced A confirmations without double-tap or B', () => {
  const spec = INPUT_SCRIPT_PRESETS['calibration-skip'];
  assert.equal(spec.button, INPUT_BUTTONS.A);
  assert.equal(spec.startMs, 5000);
  assert.equal(spec.holdMs, 100);
  assert.equal(spec.periodMs, 2500);

  const player = new ScriptedInputPlayer('calibration-skip');
  assert.equal(player.preset, 'calibration-skip');

  assert.equal(player.sampleAt(0).buttonsMask, 0);
  assert.equal(player.sampleAt(4999).buttonsMask, 0);

  const expectedPressTimes = [5000, 7500, 10000, 12500, 15000, 17500];
  for (let i = 0; i < expectedPressTimes.length; i++) {
    const t = expectedPressTimes[i];
    assert.equal(player.sampleAt(t - 1).buttonsMask, 0);
    assert.equal(player.sampleAt(t).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(t + 99).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(t + 100).buttonsMask, 0);
    // Safe 2400 ms cooldown between presses — no double-taps.
    assert.equal(player.sampleAt(t + 1000).buttonsMask, 0);
    assert.equal(player.sampleAt(t + 2499).buttonsMask, 0);
    assert.equal(player.pressCountAt(t), i + 1);
  }

  // Never presses B or any button other than A.
  for (let t = 0; t <= 30000; t += 50) {
    const mask = player.sampleAt(t).buttonsMask;
    assert.ok(mask === 0 || mask === INPUT_BUTTONS.A, `unexpected mask ${mask} at t=${t}`);
    assert.equal(mask & INPUT_BUTTONS.B, 0);
  }

  // Capped with maxPresses works cleanly.
  const capped = new ScriptedInputPlayer('calibration-skip', { maxPresses: 3 });
  assert.equal(capped.sampleAt(10000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(capped.sampleAt(12500).buttonsMask, 0);
  assert.equal(capped.pressCountAt(30000), 3);
  assert.equal(capped.nextTransitionMs(10100), null);
});

test('W174 intro-watch preset waits past the 15 s+ Bink2 intro video before issuing first A press', () => {
  const spec = INPUT_SCRIPT_PRESETS['intro-watch'];
  assert.equal(spec.button, INPUT_BUTTONS.A);
  assert.ok(spec.startMs >= 15000, `intro-watch startMs (${spec.startMs}) must exceed 15 s`);
  assert.equal(spec.startMs, 18000);
  assert.equal(spec.holdMs, 100);
  assert.equal(spec.periodMs, 2500);

  const player = new ScriptedInputPlayer('intro-watch');
  assert.equal(player.preset, 'intro-watch');

  // Completely silent throughout the entire 15 s+ intro video up to 17999 ms.
  for (const t of [0, 5000, 10000, 15000, 16500, 17999]) {
    assert.deepEqual(player.sampleAt(t), { buttonsMask: 0, activeEventCount: 0 }, `silent at ${t}`);
    assert.equal(player.pressCountAt(t), 0);
  }

  // First A press fires at 18000 ms, then repeats safely every 2500 ms.
  assert.equal(player.nextTransitionMs(0), 18000);
  assert.equal(player.sampleAt(18000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.pressCountAt(18000), 1);
  assert.equal(player.sampleAt(18100).buttonsMask, 0);
  assert.equal(player.sampleAt(20500).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.pressCountAt(20500), 2);
  assert.equal(player.sampleAt(20600).buttonsMask, 0);
});

test('W174 InputBridge exposes active-preset telemetry via getStats(), getStatus(), and getScriptState() end-to-end', () => {
  const env = createMockEnv();
  const clock = createFakeClockAndTimers(500);
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  // Idle telemetry before any script starts.
  const idleStats = bridge.getStats();
  assert.deepEqual(bridge.getStatus(), idleStats);
  assert.equal(idleStats.scriptActive, false);
  assert.equal(idleStats.activePreset, null);
  assert.equal(idleStats.preset, null);
  assert.equal(idleStats.scriptedPressesSent, 0);
  assert.equal(idleStats.lastScriptedButton, null);
  assert.equal(idleStats.elapsedScriptMs, 0);
  assert.equal(idleStats.currentScriptMask, 0);
  assert.deepEqual(idleStats.availablePresets, INPUT_SCRIPT_PRESET_NAMES);

  // Start menu-flow and drive clock through Start Screen -> calibration -> Main Menu probes.
  bridge.startScript('menu-flow', {
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });

  let stats = bridge.getStats();
  assert.equal(stats.scriptActive, true);
  assert.equal(stats.activePreset, 'menu-flow');
  assert.equal(stats.scriptedPressesSent, 0);
  assert.equal(stats.lastScriptedButton, null);
  assert.equal(stats.elapsedScriptMs, 0);

  // Advance to first Start Screen A press (t = 5000 ms).
  clock.advanceBy(5000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  stats = bridge.getStats();
  assert.equal(stats.activePreset, 'menu-flow');
  assert.equal(stats.scriptedPressesSent, 1);
  assert.equal(stats.lastScriptedButton, 'A');
  assert.equal(stats.elapsedScriptMs, 5000);
  assert.equal(stats.currentScriptMask, INPUT_BUTTONS.A);

  // Advance into quiet pause before CONTINUE (t = 10000 ms): 3 Start Screen A presses sent.
  clock.advanceBy(5000);
  assert.equal(bridge.getSnapshot().buttons, 0);
  stats = bridge.getStats();
  assert.equal(stats.scriptedPressesSent, 3);
  assert.equal(stats.lastScriptedButton, 'A');
  assert.equal(stats.elapsedScriptMs, 10000);

  // Advance to Main Menu DPAD_DOWN probe (t = 27000 ms -> 8 fixed A + 1 DPAD_DOWN = 9 presses).
  clock.advanceBy(17000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  stats = bridge.getStats();
  assert.equal(stats.activePreset, 'menu-flow');
  assert.equal(stats.scriptedPressesSent, 9);
  assert.equal(stats.lastScriptedButton, 'DPAD_DOWN');
  assert.equal(stats.elapsedScriptMs, 27000);

  // Advance to Main Menu DPAD_UP probe (t = 29000 ms -> 10 presses).
  clock.advanceBy(2000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  stats = bridge.getStats();
  assert.equal(stats.scriptedPressesSent, 10);
  assert.equal(stats.lastScriptedButton, 'DPAD_UP');
  assert.equal(stats.elapsedScriptMs, 29000);

  // Advance to Main Menu A probe (t = 31000 ms -> 11 presses).
  clock.advanceBy(2000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  stats = bridge.getStats();
  assert.equal(stats.scriptedPressesSent, 11);
  assert.equal(stats.lastScriptedButton, 'A');
  assert.equal(stats.elapsedScriptMs, 31000);

  // Advance into second Main Menu loop cycle DPAD_DOWN (t = 35000 ms -> 12 presses).
  clock.advanceBy(4000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  stats = bridge.getStats();
  assert.equal(stats.scriptedPressesSent, 12);
  assert.equal(stats.lastScriptedButton, 'DPAD_DOWN');
  assert.equal(stats.elapsedScriptMs, 35000);

  // Stopping clears activePreset to null and releases SAB button while preserving final counts.
  bridge.stopScript();
  assert.equal(bridge.getSnapshot().buttons, 0);
  stats = bridge.getStats();
  assert.equal(stats.scriptActive, false);
  assert.equal(stats.activePreset, null);
  assert.equal(stats.preset, 'menu-flow');
  assert.equal(stats.scriptedPressesSent, 12);
  assert.equal(stats.lastScriptedButton, 'DPAD_DOWN');
  assert.equal(stats.elapsedScriptMs, 35000);

  // Switching to intro-watch resets telemetry cleanly and stays silent during intro video.
  bridge.startScript('intro-watch', {
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });
  stats = bridge.getStats();
  assert.equal(stats.activePreset, 'intro-watch');
  assert.equal(stats.scriptedPressesSent, 0);
  assert.equal(stats.lastScriptedButton, null);
  assert.equal(stats.elapsedScriptMs, 0);

  clock.advanceBy(16000);
  stats = bridge.getStats();
  assert.equal(stats.activePreset, 'intro-watch');
  assert.equal(stats.scriptedPressesSent, 0);
  assert.equal(stats.lastScriptedButton, null);
  assert.equal(stats.elapsedScriptMs, 16000);
  assert.equal(bridge.getSnapshot().buttons, 0);

  clock.advanceBy(2000); // t = 18000 ms
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  stats = bridge.getStats();
  assert.equal(stats.scriptedPressesSent, 1);
  assert.equal(stats.lastScriptedButton, 'A');
  assert.equal(stats.elapsedScriptMs, 18000);

  // Simultaneous multi-button press formats lastScriptedButton cleanly and auto-complete clears activePreset.
  bridge.startScript(
    [
      { atMs: 100, holdMs: 200, button: ['A', 'RB'] },
      { atMs: 200, holdMs: 150, button: ['B', 'MENU'] },
    ],
    {
      nowFn: clock.nowFn,
      setTimeoutFn: clock.setTimeoutFn,
      clearTimeoutFn: clock.clearTimeoutFn,
    }
  );
  clock.advanceBy(100);
  assert.equal(bridge.getStats().lastScriptedButton, 'A+RB');
  assert.equal(bridge.getStats().scriptedPressesSent, 1);
  clock.advanceBy(100);
  assert.equal(bridge.getStats().lastScriptedButton, 'B+MENU');
  assert.equal(bridge.getStats().scriptedPressesSent, 2);
  clock.advanceBy(150);
  stats = bridge.getStats();
  assert.equal(stats.scriptActive, false);
  assert.equal(stats.activePreset, null);
  assert.equal(stats.scriptedPressesSent, 2);
  assert.equal(stats.lastScriptedButton, 'B+MENU');

  // Loop validation errors.
  assert.throws(
    () => new ScriptedInputPlayer({ events: [], loop: 'bad' }),
    TypeError
  );
  assert.throws(
    () => new ScriptedInputPlayer({ events: [], loop: { startMs: 0, periodMs: 1000, events: 'bad' } }),
    TypeError
  );
  assert.throws(
    () =>
      new ScriptedInputPlayer({
        events: [],
        loop: {
          startMs: 0,
          periodMs: 1000,
          events: [{ offsetMs: 1000, holdMs: 100, button: 'A' }],
        },
      }),
    RangeError
  );

  bridge.destroy();
});

test('W199 legal-notices preset mirrors #948 timings, DPAD_DOWN -> DPAD_DOWN -> A loop with 3 s A spacing, and zero B/DPAD_UP', () => {
  const spec = INPUT_SCRIPT_PRESETS['legal-notices'];
  assert.ok(spec);
  assert.equal(spec.name, 'legal-notices');
  assert.equal(spec.events.length, 17); // 20000..60000 step 2500
  assert.equal(spec.events[0].atMs, 20000);
  assert.equal(spec.events[spec.events.length - 1].atMs, 60000);
  assert.equal(spec.loop.startMs, 64000);
  assert.equal(spec.loop.periodMs, 12000);
  assert.equal(spec.loop.events.length, 6);

  const player = new ScriptedInputPlayer('legal-notices');
  assert.equal(player.preset, 'legal-notices');

  // Silent during 4K Bink2 intro video (0..19999 ms).
  for (const t of [0, 5000, 15000, 19999]) {
    assert.deepEqual(player.sampleAt(t), { buttonsMask: 0, activeEventCount: 0 }, `silent at ${t}`);
    assert.equal(player.pressCountAt(t), 0);
  }

  // Phase 1: A every 2500 ms from 20000 ms through 60000 ms (<= 62000 ms).
  let fixedCount = 0;
  for (let atMs = 20000; atMs <= 62000; atMs += 2500) {
    fixedCount += 1;
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0, `idle before A at ${atMs}`);
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `A pressed at ${atMs}`);
    assert.equal(player.sampleAt(atMs + 99).buttonsMask, INPUT_BUTTONS.A, `A held at ${atMs + 99}`);
    assert.equal(player.sampleAt(atMs + 100).buttonsMask, 0, `A released at ${atMs + 100}`);
    assert.equal(player.pressCountAt(atMs), fixedCount);
  }
  assert.equal(fixedCount, 17);

  // Quiet window between last fixed A (60100 ms) and LEGAL NOTICES loop start (64000 ms).
  for (let t = 60100; t < 64000; t += 100) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `quiet pause at ${t}`);
  }

  // Phase 2: Every 12000 ms from 64000 ms: DPAD_DOWN (+0), DPAD_DOWN (+500), A (+1000),
  // plus interspersed A every 3000 ms (+4000, +7000, +10000).
  const expectedLoopOffsets = [
    [0, INPUT_BUTTONS.DPAD_DOWN],
    [500, INPUT_BUTTONS.DPAD_DOWN],
    [1000, INPUT_BUTTONS.A],
    [4000, INPUT_BUTTONS.A],
    [7000, INPUT_BUTTONS.A],
    [10000, INPUT_BUTTONS.A],
  ];

  const loopATimestamps = [];
  for (let cycle = 0; cycle < 4; cycle++) {
    const base = 64000 + cycle * 12000;
    for (let j = 0; j < expectedLoopOffsets.length; j++) {
      const [offsetMs, expectedBtn] = expectedLoopOffsets[j];
      const t = base + offsetMs;
      assert.equal(player.sampleAt(t - 1).buttonsMask, 0, `cycle ${cycle} idle before +${offsetMs}`);
      assert.equal(player.sampleAt(t).buttonsMask, expectedBtn, `cycle ${cycle} press at +${offsetMs}`);
      assert.equal(player.sampleAt(t + 99).buttonsMask, expectedBtn, `cycle ${cycle} hold at +${offsetMs}`);
      assert.equal(player.sampleAt(t + 100).buttonsMask, 0, `cycle ${cycle} release at +${offsetMs}`);
      assert.equal(player.pressCountAt(t), fixedCount + cycle * 6 + j + 1);
      if (expectedBtn === INPUT_BUTTONS.A) {
        loopATimestamps.push(t);
      }
    }
  }

  // Every A in the loop is spaced by exactly 3000 ms (both within and across 12 s cycles).
  for (let i = 1; i < loopATimestamps.length; i++) {
    assert.equal(
      loopATimestamps[i] - loopATimestamps[i - 1],
      3000,
      `expected 3000 ms between loop A presses (${loopATimestamps[i - 1]} -> ${loopATimestamps[i]})`
    );
  }

  // Invariant: NEVER presses B or DPAD_UP anywhere in 240 seconds; only A or DPAD_DOWN.
  for (let t = 0; t <= 240000; t += 50) {
    const mask = player.sampleAt(t).buttonsMask;
    assert.equal(mask & INPUT_BUTTONS.B, 0, `legal-notices must never press B at t=${t}`);
    assert.equal(mask & INPUT_BUTTONS.DPAD_UP, 0, `legal-notices must never press DPAD_UP at t=${t}`);
    assert.ok(
      mask === 0 || mask === INPUT_BUTTONS.A || mask === INPUT_BUTTONS.DPAD_DOWN,
      `unexpected button mask ${mask} at t=${t}`
    );
  }

  // Exact nextTransitionMs across fixed -> loop boundary and within/across cycles.
  assert.equal(player.nextTransitionMs(0), 20000);
  assert.equal(player.nextTransitionMs(60000), 60100);
  assert.equal(player.nextTransitionMs(60100), 64000);
  assert.equal(player.nextTransitionMs(64000), 64100);
  assert.equal(player.nextTransitionMs(64100), 64500);
  assert.equal(player.nextTransitionMs(64600), 65000);
  assert.equal(player.nextTransitionMs(65100), 68000);
  assert.equal(player.nextTransitionMs(74100), 76000);

  // maxPresses & endMs caps work cleanly on legal-notices.
  const capped = new ScriptedInputPlayer({ preset: 'legal-notices', maxPresses: 20 });
  assert.equal(capped.preset, 'legal-notices');
  assert.equal(capped.sampleAt(64000).buttonsMask, INPUT_BUTTONS.DPAD_DOWN); // #18
  assert.equal(capped.sampleAt(64500).buttonsMask, INPUT_BUTTONS.DPAD_DOWN); // #19
  assert.equal(capped.sampleAt(65000).buttonsMask, INPUT_BUTTONS.A);         // #20
  assert.equal(capped.sampleAt(68000).buttonsMask, 0);
  assert.equal(capped.pressCountAt(120000), 20);
  assert.equal(capped.nextTransitionMs(65100), null);

  const cappedEnd = new ScriptedInputPlayer('legal-notices', { endMs: 64550 });
  assert.equal(cappedEnd.sampleAt(64500).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(cappedEnd.sampleAt(64549).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(cappedEnd.sampleAt(64550).buttonsMask, 0);
  assert.equal(cappedEnd.nextTransitionMs(64500), 64550);
  assert.equal(cappedEnd.nextTransitionMs(64550), null);
});

test('W199 chrome-menu-flow preset waits through Wasm64 intro/shader init, presses A 165s..215s, loops DPAD_DOWN -> DPAD_DOWN -> A x3 from 218s, and zero B/DPAD_UP', () => {
  const spec = INPUT_SCRIPT_PRESETS['chrome-menu-flow'];
  assert.ok(spec);
  assert.equal(spec.name, 'chrome-menu-flow');
  assert.equal(spec.events.length, 21); // 165000..215000 inclusive step 2500
  assert.equal(spec.events[0].atMs, 165000);
  assert.equal(spec.events[spec.events.length - 1].atMs, 215000);
  assert.equal(spec.loop.startMs, 218000);
  assert.equal(spec.loop.periodMs, 12000);
  assert.equal(spec.loop.events.length, 5);

  const player = new ScriptedInputPlayer('chrome-menu-flow');
  assert.equal(player.preset, 'chrome-menu-flow');

  // Completely silent during 4K Bink2 intro & shader init (0..164999 ms).
  for (const t of [0, 20000, 64000, 120000, 160000, 164999]) {
    assert.deepEqual(player.sampleAt(t), { buttonsMask: 0, activeEventCount: 0 }, `silent at ${t}`);
    assert.equal(player.pressCountAt(t), 0);
  }

  // Phase 1: Spaced A every 2500 ms from 165000 ms through 215000 ms (21 presses).
  let fixedCount = 0;
  for (let atMs = 165000; atMs <= 215000; atMs += 2500) {
    fixedCount += 1;
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0, `idle before A at ${atMs}`);
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `A pressed at ${atMs}`);
    assert.equal(player.sampleAt(atMs + 99).buttonsMask, INPUT_BUTTONS.A, `A held at ${atMs + 99}`);
    assert.equal(player.sampleAt(atMs + 100).buttonsMask, 0, `A released at ${atMs + 100}`);
    assert.equal(player.pressCountAt(atMs), fixedCount);
  }
  assert.equal(fixedCount, 21);

  // Quiet transition window (215100..217999 ms) before loop start at 218000 ms.
  for (let t = 215100; t < 218000; t += 100) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `quiet pause at ${t}`);
  }

  // Phase 2: Loop DPAD_DOWN (+0), DPAD_DOWN (+500), A (+1200), A (+4500), A (+8000) every 12000 ms from 218000 ms.
  const expectedLoopOffsets = [
    [0, INPUT_BUTTONS.DPAD_DOWN],
    [500, INPUT_BUTTONS.DPAD_DOWN],
    [1200, INPUT_BUTTONS.A],
    [4500, INPUT_BUTTONS.A],
    [8000, INPUT_BUTTONS.A],
  ];

  for (let cycle = 0; cycle < 4; cycle++) {
    const base = 218000 + cycle * 12000;
    for (let j = 0; j < expectedLoopOffsets.length; j++) {
      const [offsetMs, expectedBtn] = expectedLoopOffsets[j];
      const t = base + offsetMs;
      assert.equal(player.sampleAt(t - 1).buttonsMask, 0, `cycle ${cycle} idle before +${offsetMs}`);
      assert.equal(player.sampleAt(t).buttonsMask, expectedBtn, `cycle ${cycle} press at +${offsetMs}`);
      assert.equal(player.sampleAt(t + 99).buttonsMask, expectedBtn, `cycle ${cycle} hold at +${offsetMs}`);
      assert.equal(player.sampleAt(t + 100).buttonsMask, 0, `cycle ${cycle} release at +${offsetMs}`);
      assert.equal(player.pressCountAt(t), fixedCount + cycle * 5 + j + 1);
    }
  }

  // Invariant: NEVER presses B or DPAD_UP anywhere in 320 seconds; only A or DPAD_DOWN.
  for (let t = 0; t <= 320000; t += 50) {
    const mask = player.sampleAt(t).buttonsMask;
    assert.equal(mask & INPUT_BUTTONS.B, 0, `chrome-menu-flow must never press B at t=${t}`);
    assert.equal(mask & INPUT_BUTTONS.DPAD_UP, 0, `chrome-menu-flow must never press DPAD_UP at t=${t}`);
    assert.ok(
      mask === 0 || mask === INPUT_BUTTONS.A || mask === INPUT_BUTTONS.DPAD_DOWN,
      `unexpected button mask ${mask} at t=${t}`
    );
  }

  // Exact nextTransitionMs across intro -> fixed -> loop boundaries.
  assert.equal(player.nextTransitionMs(0), 165000);
  assert.equal(player.nextTransitionMs(215000), 215100);
  assert.equal(player.nextTransitionMs(215100), 218000);
  assert.equal(player.nextTransitionMs(218000), 218100);
  assert.equal(player.nextTransitionMs(218100), 218500);
  assert.equal(player.nextTransitionMs(218600), 219200);
  assert.equal(player.nextTransitionMs(219300), 222500);
  assert.equal(player.nextTransitionMs(222600), 226000);
  assert.equal(player.nextTransitionMs(226100), 230000);

  // maxPresses & endMs caps work cleanly on chrome-menu-flow.
  const capped = new ScriptedInputPlayer({ preset: 'chrome-menu-flow', maxPresses: 24 });
  assert.equal(capped.preset, 'chrome-menu-flow');
  assert.equal(capped.sampleAt(218000).buttonsMask, INPUT_BUTTONS.DPAD_DOWN); // #22
  assert.equal(capped.sampleAt(218500).buttonsMask, INPUT_BUTTONS.DPAD_DOWN); // #23
  assert.equal(capped.sampleAt(219200).buttonsMask, INPUT_BUTTONS.A);         // #24
  assert.equal(capped.sampleAt(222500).buttonsMask, 0);
  assert.equal(capped.pressCountAt(300000), 24);
  assert.equal(capped.nextTransitionMs(219300), null);

  const cappedEnd = new ScriptedInputPlayer('chrome-menu-flow', { endMs: 219250 });
  assert.equal(cappedEnd.sampleAt(219200).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(cappedEnd.sampleAt(219249).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(cappedEnd.sampleAt(219250).buttonsMask, 0);
  assert.equal(cappedEnd.nextTransitionMs(219200), 219250);
  assert.equal(cappedEnd.nextTransitionMs(219250), null);
});

test('W199 InputBridge drives legal-notices and chrome-menu-flow into SAB and formats HUD telemetry cleanly', () => {
  const env = createMockEnv();
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  assert.ok(bridge.getStats().availablePresets.includes('legal-notices'));
  assert.ok(bridge.getStats().availablePresets.includes('chrome-menu-flow'));

  // 1. legal-notices end-to-end playback through DPAD_DOWN -> DPAD_DOWN -> A -> interspersed A.
  bridge.startScript('legal-notices', {
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });

  let stats = bridge.getStats();
  assert.equal(stats.scriptActive, true);
  assert.equal(stats.activePreset, 'legal-notices');
  assert.equal(stats.scriptedPressesSent, 0);
  assert.equal(stats.lastScriptedButton, null);

  // First A press at 20000 ms.
  clock.advanceBy(20000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  stats = bridge.getStats();
  assert.equal(stats.scriptedPressesSent, 1);
  assert.equal(stats.lastScriptedButton, 'A');
  assert.equal(stats.elapsedScriptMs, 20000);

  // Advance to LEGAL NOTICES loop start (64000 ms -> 17 fixed A + 1st DPAD_DOWN = 18 presses).
  clock.advanceBy(44000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  stats = bridge.getStats();
  assert.equal(stats.activePreset, 'legal-notices');
  assert.equal(stats.scriptedPressesSent, 18);
  assert.equal(stats.lastScriptedButton, 'DPAD_DOWN');

  // 2nd DPAD_DOWN at 64500 ms -> 19 presses.
  clock.advanceBy(500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  stats = bridge.getStats();
  assert.equal(stats.scriptedPressesSent, 19);
  assert.equal(stats.lastScriptedButton, 'DPAD_DOWN');

  // Confirm "Next >" with A at 65000 ms -> 20 presses.
  clock.advanceBy(500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  stats = bridge.getStats();
  assert.equal(stats.scriptedPressesSent, 20);
  assert.equal(stats.lastScriptedButton, 'A');

  // Interspersed A at 68000 ms -> 21 presses.
  clock.advanceBy(3000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 21);

  // 2. Switch to chrome-menu-flow: resets telemetry cleanly and waits through 165 s intro/shader init.
  bridge.startScript('chrome-menu-flow', {
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });
  stats = bridge.getStats();
  assert.equal(stats.scriptActive, true);
  assert.equal(stats.activePreset, 'chrome-menu-flow');
  assert.equal(stats.scriptedPressesSent, 0);
  assert.equal(stats.lastScriptedButton, null);
  assert.equal(stats.elapsedScriptMs, 0);
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Still silent at 164999 ms.
  clock.advanceBy(164999);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getStats().scriptedPressesSent, 0);

  // First A at 165000 ms.
  clock.advanceBy(1);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 1);
  assert.equal(bridge.getStats().lastScriptedButton, 'A');

  // Advance to loop start (218000 ms -> 21 fixed A + 1st DPAD_DOWN = 22 presses).
  clock.advanceBy(53000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  stats = bridge.getStats();
  assert.equal(stats.scriptedPressesSent, 22);
  assert.equal(stats.lastScriptedButton, 'DPAD_DOWN');

  // 2nd DPAD_DOWN at 218500 ms -> 23 presses.
  clock.advanceBy(500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(bridge.getStats().scriptedPressesSent, 23);

  // A at 219200 ms -> 24 presses; A at 222500 ms -> 25 presses; A at 226000 ms -> 26 presses.
  clock.advanceBy(700);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 24);
  assert.equal(bridge.getStats().lastScriptedButton, 'A');

  clock.advanceBy(3300);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 25);

  clock.advanceBy(3500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 26);

  // Stopping clears activePreset & SAB buttons while preserving telemetry counters.
  bridge.stopScript();
  assert.equal(bridge.getSnapshot().buttons, 0);
  stats = bridge.getStatus();
  assert.equal(stats.scriptActive, false);
  assert.equal(stats.activePreset, null);
  assert.equal(stats.preset, 'chrome-menu-flow');
  assert.equal(stats.scriptedPressesSent, 26);
  assert.equal(stats.lastScriptedButton, 'A');

  bridge.destroy();
});

test('W229 chrome-full-boot preset plays full 4K Bink2 intro unskipped, presses A 235s..295s, loops DPAD_DOWN -> DPAD_DOWN -> A x3 from 298s, and zero B/DPAD_UP', () => {
  assert.ok(INPUT_SCRIPT_PRESET_NAMES.includes('chrome-full-boot'));
  const spec = INPUT_SCRIPT_PRESETS['chrome-full-boot'];
  assert.ok(spec);
  assert.equal(spec.name, 'chrome-full-boot');
  assert.equal(spec.events.length, 25); // 235000..295000 inclusive step 2500
  assert.equal(spec.events[0].atMs, 235000);
  assert.equal(spec.events[spec.events.length - 1].atMs, 295000);
  assert.equal(spec.loop.startMs, 298000);
  assert.equal(spec.loop.periodMs, 12000);
  assert.equal(spec.loop.events.length, 5);

  const player = new ScriptedInputPlayer('chrome-full-boot');
  assert.equal(player.preset, 'chrome-full-boot');

  // Zero button presses before 235000 ms so the 4K Bink2 intro video & Rockstar logo play unskipped.
  for (const t of [0, 20000, 64000, 120000, 165000, 218000, 234999]) {
    assert.deepEqual(player.sampleAt(t), { buttonsMask: 0, activeEventCount: 0 }, `silent at ${t}`);
    assert.equal(player.pressCountAt(t), 0);
  }

  // Phase 1: Spaced A every 2500 ms (holdMs: 100) from 235000 ms through 295000 ms (25 presses).
  let fixedCount = 0;
  for (let atMs = 235000; atMs <= 295000; atMs += 2500) {
    fixedCount += 1;
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0, `idle before A at ${atMs}`);
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `A pressed at ${atMs}`);
    assert.equal(player.sampleAt(atMs + 99).buttonsMask, INPUT_BUTTONS.A, `A held at ${atMs + 99}`);
    assert.equal(player.sampleAt(atMs + 100).buttonsMask, 0, `A released at ${atMs + 100}`);
    assert.equal(player.pressCountAt(atMs), fixedCount);
  }
  assert.equal(fixedCount, 25);

  // Quiet transition window (295100..297999 ms) before loop start at 298000 ms.
  for (let t = 295100; t < 298000; t += 100) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `quiet pause at ${t}`);
  }

  // Phase 2: Loop DPAD_DOWN (+0), DPAD_DOWN (+500), A (+1200), A (+4500), A (+8000) every 12000 ms from 298000 ms.
  const expectedLoopOffsets = [
    [0, INPUT_BUTTONS.DPAD_DOWN],
    [500, INPUT_BUTTONS.DPAD_DOWN],
    [1200, INPUT_BUTTONS.A],
    [4500, INPUT_BUTTONS.A],
    [8000, INPUT_BUTTONS.A],
  ];

  for (let cycle = 0; cycle < 4; cycle++) {
    const base = 298000 + cycle * 12000;
    for (let j = 0; j < expectedLoopOffsets.length; j++) {
      const [offsetMs, expectedBtn] = expectedLoopOffsets[j];
      const t = base + offsetMs;
      assert.equal(player.sampleAt(t - 1).buttonsMask, 0, `cycle ${cycle} idle before +${offsetMs}`);
      assert.equal(player.sampleAt(t).buttonsMask, expectedBtn, `cycle ${cycle} press at +${offsetMs}`);
      assert.equal(player.sampleAt(t + 99).buttonsMask, expectedBtn, `cycle ${cycle} hold at +${offsetMs}`);
      assert.equal(player.sampleAt(t + 100).buttonsMask, 0, `cycle ${cycle} release at +${offsetMs}`);
      assert.equal(player.pressCountAt(t), fixedCount + cycle * 5 + j + 1);
    }
  }

  // Invariant: Zero presses before 235000 ms, and NEVER presses B or DPAD_UP anywhere in 420 seconds; only A or DPAD_DOWN.
  for (let t = 0; t <= 420000; t += 50) {
    const mask = player.sampleAt(t).buttonsMask;
    if (t < 235000) {
      assert.equal(mask, 0, `chrome-full-boot must emit zero presses before 235000 ms (got ${mask} at t=${t})`);
    }
    assert.equal(mask & INPUT_BUTTONS.B, 0, `chrome-full-boot must never press B at t=${t}`);
    assert.equal(mask & INPUT_BUTTONS.DPAD_UP, 0, `chrome-full-boot must never press DPAD_UP at t=${t}`);
    assert.ok(
      mask === 0 || mask === INPUT_BUTTONS.A || mask === INPUT_BUTTONS.DPAD_DOWN,
      `unexpected button mask ${mask} at t=${t}`
    );
  }

  // Exact nextTransitionMs across intro -> Phase 1 -> Phase 2 boundaries.
  assert.equal(player.nextTransitionMs(0), 235000);
  assert.equal(player.nextTransitionMs(295000), 295100);
  assert.equal(player.nextTransitionMs(295100), 298000);
  assert.equal(player.nextTransitionMs(298000), 298100);
  assert.equal(player.nextTransitionMs(298100), 298500);
  assert.equal(player.nextTransitionMs(298600), 299200);
  assert.equal(player.nextTransitionMs(299300), 302500);
  assert.equal(player.nextTransitionMs(302600), 306000);
  assert.equal(player.nextTransitionMs(306100), 310000);

  // maxPresses & endMs caps work cleanly on chrome-full-boot.
  const capped = new ScriptedInputPlayer({ preset: 'chrome-full-boot', maxPresses: 28 });
  assert.equal(capped.preset, 'chrome-full-boot');
  assert.equal(capped.sampleAt(298000).buttonsMask, INPUT_BUTTONS.DPAD_DOWN); // #26
  assert.equal(capped.sampleAt(298500).buttonsMask, INPUT_BUTTONS.DPAD_DOWN); // #27
  assert.equal(capped.sampleAt(299200).buttonsMask, INPUT_BUTTONS.A);         // #28
  assert.equal(capped.sampleAt(302500).buttonsMask, 0);
  assert.equal(capped.pressCountAt(400000), 28);
  assert.equal(capped.nextTransitionMs(299300), null);

  const cappedEnd = new ScriptedInputPlayer('chrome-full-boot', { endMs: 299250 });
  assert.equal(cappedEnd.sampleAt(299200).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(cappedEnd.sampleAt(299249).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(cappedEnd.sampleAt(299250).buttonsMask, 0);
  assert.equal(cappedEnd.nextTransitionMs(299200), 299250);
  assert.equal(cappedEnd.nextTransitionMs(299250), null);
});

test('W229 InputBridge drives chrome-full-boot into SAB and formats HUD telemetry cleanly', () => {
  const env = createMockEnv();
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  assert.ok(bridge.getStats().availablePresets.includes('chrome-full-boot'));

  bridge.startScript('chrome-full-boot', {
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });

  let stats = bridge.getStats();
  assert.equal(stats.scriptActive, true);
  assert.equal(stats.activePreset, 'chrome-full-boot');
  assert.equal(stats.scriptedPressesSent, 0);
  assert.equal(stats.lastScriptedButton, null);
  assert.equal(stats.elapsedScriptMs, 0);
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Silent throughout the entire 4K Bink2 intro & Rockstar logo (0..234999 ms).
  clock.advanceBy(234999);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getStats().scriptedPressesSent, 0);

  // First A press at 235000 ms.
  clock.advanceBy(1);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 1);
  assert.equal(bridge.getStats().lastScriptedButton, 'A');

  // Advance to loop start (298000 ms -> 25 fixed A + 1st DPAD_DOWN = 26 presses).
  clock.advanceBy(63000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  stats = bridge.getStats();
  assert.equal(stats.scriptedPressesSent, 26);
  assert.equal(stats.lastScriptedButton, 'DPAD_DOWN');

  // 2nd DPAD_DOWN at 298500 ms -> 27 presses.
  clock.advanceBy(500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(bridge.getStats().scriptedPressesSent, 27);

  // A at 299200 ms -> 28 presses; A at 302500 ms -> 29 presses; A at 306000 ms -> 30 presses.
  clock.advanceBy(700);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 28);
  assert.equal(bridge.getStats().lastScriptedButton, 'A');

  clock.advanceBy(3300);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 29);

  clock.advanceBy(3500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 30);

  // Stopping clears activePreset & SAB buttons while preserving telemetry counters.
  bridge.stopScript();
  assert.equal(bridge.getSnapshot().buttons, 0);
  stats = bridge.getStatus();
  assert.equal(stats.scriptActive, false);
  assert.equal(stats.activePreset, null);
  assert.equal(stats.preset, 'chrome-full-boot');
  assert.equal(stats.scriptedPressesSent, 30);
  assert.equal(stats.lastScriptedButton, 'A');

  bridge.destroy();
});

test('W271 applyRadialDeadzone smoothly rescales [inner, outer] -> [0, 1], clamps non-finite inputs, and integrates with gamepad polling', () => {
  assert.equal(bridgeApplyRadialDeadzone, applyRadialDeadzone);

  // Inside deadzone -> exact +0
  const inside = applyRadialDeadzone(0.1, -0.08, 0.15, 0.95);
  assert.deepEqual(inside, { x: 0, y: 0 });
  assert.ok(Object.is(inside.x, 0));
  assert.ok(Object.is(inside.y, 0));

  // Boundary at innerDeadzone -> 0; just outside -> smooth ramp from 0
  assert.deepEqual(applyRadialDeadzone(0.15, 0, 0.15, 0.95), { x: 0, y: 0 });
  const midX = applyRadialDeadzone(0.55, 0, 0.15, 0.95);
  approxEqual(midX.x, (0.55 - 0.15) / (0.95 - 0.15)); // 0.5
  assert.equal(midX.y, 0);

  // Diagonal direction preservation & smooth rescale from [0.2, 0.9]
  const diag = applyRadialDeadzone(0.5, 0.5, 0.2, 0.9);
  approxEqual(diag.x, diag.y);
  approxEqual(Math.hypot(diag.x, diag.y), (Math.hypot(0.5, 0.5) - 0.2) / (0.9 - 0.2));

  // Outer deadzone saturation & unit-disk clamping on diagonal corner
  const satX = applyRadialDeadzone(0.95, 0, 0.15, 0.95);
  approxEqual(satX.x, 1.0);
  assert.equal(satX.y, 0);

  const corner = applyRadialDeadzone(1.0, 1.0, 0.15, 0.95);
  approxEqual(Math.hypot(corner.x, corner.y), 1.0);
  approxEqual(corner.x, Math.SQRT1_2);
  approxEqual(corner.y, Math.SQRT1_2);

  // Non-finite inputs clamp cleanly to { x: 0, y: 0 }
  assert.deepEqual(applyRadialDeadzone(Number.NaN, 0.5, 0.15, 0.95), { x: 0, y: 0 });
  assert.deepEqual(applyRadialDeadzone(0.5, Number.POSITIVE_INFINITY, 0.15, 0.95), { x: 0, y: 0 });
  assert.deepEqual(applyRadialDeadzone(Number.NEGATIVE_INFINITY, Number.NaN, 0.15, 0.95), { x: 0, y: 0 });
  assert.deepEqual(applyRadialDeadzone(undefined, 0.5, 0.15, 0.95), { x: 0, y: 0 });

  // Zero-alloc scratch object reuse
  const scratch = { x: 42, y: -42 };
  const reused = applyRadialDeadzone(0.55, 0, 0.15, 0.95, scratch);
  assert.equal(reused, scratch);
  approxEqual(scratch.x, 0.5);
  assert.equal(scratch.y, 0);

  // End-to-end integration inside InputBridge.pollGamepads() with per-stick deadzones
  const env = createMockEnv({
    gamepads: [
      makeStandardGamepad({
        index: 0,
        axes: [0.55, 0.0, 0.2, -0.2],
      }),
    ],
  });
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
    leftStickDeadzone: 0.15,
    rightStickDeadzone: 0.3,
    stickOuterDeadzone: 0.95,
  });

  assert.equal(bridge.pollGamepads(), true);
  const snap = bridge.getSnapshot();
  approxEqual(snap.lx, 0.5);
  approxEqual(snap.ly, 0);
  // Right stick hypot(0.2, 0.2) ~= 0.2828 < rightStickDeadzone 0.3 -> zeroed
  assert.equal(snap.rx, 0);
  assert.equal(snap.ry, 0);
  assert.equal(bridge.getStats().activeGamepadIndex, 0);

  bridge.destroy();
});

test('W271 window.blur and document.visibilitychange clear stuck keyboard, mouse, and gamepad inputs and track blurResets', () => {
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  assert.equal(bridge.getStats().blurResets, 0);

  // Hold keyboard, mouse buttons/delta, and gamepad buttons/sticks/triggers simultaneously
  env.lockPointer();
  env.win.dispatch('keydown', { code: 'KeyW' });
  env.win.dispatch('mousedown', { button: 0 });
  env.win.dispatch('mousemove', { movementX: 14, movementY: -9 });
  env.setGamepads([
    makeStandardGamepad({
      index: 0,
      buttons: { 0: 1, 6: 0.7, 7: 0.8 }, // A, LT, RT
      axes: [0.7, -0.5, -0.6, 0.6],
    }),
  ]);
  assert.equal(bridge.pollGamepads(), true);

  let snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.A);
  assert.equal(snap.mouse_dx, 14);
  assert.equal(snap.mouse_dy, -9);
  assert.ok(snap.lx !== 0 && snap.ly !== 0 && snap.rx !== 0 && snap.ry !== 0);
  assert.ok(snap.lt > 0 && snap.rt > 0);

  // 1. window.blur immediately writes a clean zeroed frame to the SAB and increments blurResets
  const seqBeforeBlur = snap.seq;
  env.win.dispatch('blur');
  snap = bridge.getSnapshot();
  assert.ok(snap.seq > seqBeforeBlur && (snap.seq & 1) === 0);
  assert.equal(snap.buttons, 0);
  assert.equal(snap.mouse_dx, 0);
  assert.equal(snap.mouse_dy, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);
  assert.equal(snap.rx, 0);
  assert.equal(snap.ry, 0);
  assert.equal(snap.lt, 0);
  assert.equal(snap.rt, 0);
  assert.equal(bridge.getStats().blurResets, 1);

  // 2. Re-engage gamepad + keyboard; visibilitychange while visible (hidden === false) must NOT clear
  env.win.dispatch('keydown', { code: 'KeyD' });
  assert.equal(bridge.pollGamepads(), true);
  assert.notEqual(bridge.getSnapshot().buttons, 0);

  env.doc.hidden = false;
  env.doc.visibilityState = 'visible';
  env.doc.dispatch('visibilitychange');
  assert.notEqual(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getStats().blurResets, 1);

  // 3. visibilitychange with document.hidden === true clears keyboard, mouse, and gamepad cleanly
  env.doc.hidden = true;
  env.doc.visibilityState = 'hidden';
  env.doc.dispatch('visibilitychange');

  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.mouse_dx, 0);
  assert.equal(snap.mouse_dy, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);
  assert.equal(snap.rx, 0);
  assert.equal(snap.ry, 0);
  assert.equal(snap.lt, 0);
  assert.equal(snap.rt, 0);
  assert.equal(bridge.getStats().blurResets, 2);

  bridge.destroy();
});

test('W271 gamepad hot-plug connect/disconnect/failover updates stats cleanly and destroy() removes all listeners', () => {
  const env = createMockEnv({ withRaf: true });
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  // Initial stats
  let stats = bridge.getStats();
  assert.equal(stats.blurResets, 0);
  assert.equal(stats.gamepadConnects, 0);
  assert.equal(stats.gamepadDisconnects, 0);
  assert.equal(stats.activeGamepadIndex, null);

  // 1. Connect Xbox controller at index 0 with held A + left stick + right stick + triggers
  const xboxPad = makeStandardGamepad({
    index: 0,
    buttons: { 0: 1, 6: 0.6, 7: 0.9 },
    axes: [0.8, -0.6, -0.5, 0.5],
  });
  env.setGamepads([xboxPad]);
  env.win.dispatch('gamepadconnected', { gamepad: xboxPad });

  stats = bridge.getStats();
  assert.equal(bridge.running, true);
  assert.equal(stats.gamepadConnects, 1);
  assert.equal(stats.gamepadDisconnects, 0);
  assert.equal(stats.activeGamepadIndex, 0);

  let snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.A);
  assert.ok(snap.lx > 0 && snap.ly > 0 && snap.rx < 0 && snap.ry < 0);
  assert.ok(snap.lt > 0 && snap.rt > 0);

  // 2. Disconnect pad 0 even before navigator.getGamepads() removes it -> immediate zeroed SAB frame
  env.win.dispatch('gamepaddisconnected', { gamepad: { index: 0 } });
  stats = bridge.getStats();
  assert.equal(stats.gamepadDisconnects, 1);
  assert.equal(stats.activeGamepadIndex, null);
  assert.equal(bridge.running, false);
  assert.equal(env.pendingRafCount(), 0);

  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);
  assert.equal(snap.rx, 0);
  assert.equal(snap.ry, 0);
  assert.equal(snap.lt, 0);
  assert.equal(snap.rt, 0);

  // 3. Reconnect pad 0 AND plug in DualSense at index 1; disconnect pad 0 -> seamless failover to pad 1
  const dualSensePad = {
    ...makeStandardGamepad({ index: 1, buttons: { 3: 1 }, axes: [0, 0, 0.6, 0] }),
    id: 'DualSense Wireless Controller (STANDARD GAMEPAD)',
  };
  env.setGamepads([xboxPad, dualSensePad]);
  env.win.dispatch('gamepadconnected', { gamepad: xboxPad });
  assert.equal(bridge.getStats().gamepadConnects, 2);
  assert.equal(bridge.getStats().activeGamepadIndex, 0);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  env.setGamepads([null, dualSensePad]);
  env.win.dispatch('gamepaddisconnected', { gamepad: { index: 0 } });
  stats = bridge.getStats();
  assert.equal(stats.gamepadDisconnects, 2);
  assert.equal(stats.activeGamepadIndex, 1);
  assert.equal(bridge.running, true);
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.Y);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);
  assert.ok(snap.rx > 0);

  // 4. destroy() removes all listeners cleanly (blur, visibilitychange, gamepad*, key*, mouse*),
  // cancels rAF, zeros SAB, clears activeGamepadIndex, and ignores subsequent events.
  bridge.destroy();
  assert.equal(bridge.disposed, true);
  assert.equal(bridge.running, false);
  assert.equal(env.pendingRafCount(), 0);
  assert.equal(env.canvas.listenerCount(), 0);
  assert.equal(env.doc.listenerCount(), 0);
  assert.equal(env.win.listenerCount(), 0);

  stats = bridge.getStats();
  assert.equal(stats.activeGamepadIndex, null);
  const seqAfterDestroy = bridge.getSnapshot().seq;
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Post-destroy events must not mutate stats or SAB
  env.win.dispatch('blur');
  env.doc.hidden = true;
  env.doc.dispatch('visibilitychange');
  env.win.dispatch('gamepadconnected', { gamepad: xboxPad });
  env.win.dispatch('gamepaddisconnected', { gamepad: dualSensePad });
  assert.equal(bridge.getStats().blurResets, stats.blurResets);
  assert.equal(bridge.getStats().gamepadConnects, stats.gamepadConnects);
  assert.equal(bridge.getStats().gamepadDisconnects, stats.gamepadDisconnects);
  assert.equal(bridge.getSnapshot().seq, seqAfterDestroy);
});

test('W297 menu-to-ingame preset progresses deterministically from Start Screen through Calibration/Legal, Intro skip, Main Menu, and in-game sticks/triggers, and completes cleanly bounded', () => {
  assert.ok(INPUT_SCRIPT_PRESET_NAMES.includes('menu-to-ingame'));
  const spec = INPUT_SCRIPT_PRESETS['menu-to-ingame'];
  assert.ok(spec);
  assert.equal(spec.name, 'menu-to-ingame');
  assert.equal(spec.loop, undefined, 'menu-to-ingame must be cleanly bounded without an infinite loop');
  assert.equal(spec.events.length, 24);
  assert.equal(spec.events[0].atMs, 5000);
  assert.equal(spec.events[spec.events.length - 1].atMs, 58500);
  assert.equal(spec.events[spec.events.length - 1].holdMs, 800);

  const player = new ScriptedInputPlayer('menu-to-ingame');
  assert.equal(player.preset, 'menu-to-ingame');

  // Initial quiet boot settle window (0..4999 ms).
  for (const t of [0, 1000, 4999]) {
    assert.deepEqual(player.sampleAt(t), { buttonsMask: 0, activeEventCount: 0 }, `silent at ${t}`);
    assert.equal(player.pressCountAt(t), 0);
  }

  // Phase 1: Start Screen A confirmations (5000, 7000, 9000).
  for (const [idx, atMs] of [5000, 7000, 9000].entries()) {
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0);
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `Start Screen A at ${atMs}`);
    assert.equal(player.sampleAt(atMs + 99).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(atMs + 100).buttonsMask, 0);
    assert.equal(player.pressCountAt(atMs), idx + 1);
  }

  // Phase 2: Calibration & Legal Notices accepts (12000..26500) including DPAD_DOWN x2 + A for "Next >".
  const phase2 = [
    [12000, INPUT_BUTTONS.A],
    [14500, INPUT_BUTTONS.A],
    [17000, INPUT_BUTTONS.A],
    [20000, INPUT_BUTTONS.DPAD_DOWN],
    [20500, INPUT_BUTTONS.DPAD_DOWN],
    [21200, INPUT_BUTTONS.A],
    [24000, INPUT_BUTTONS.A],
    [26500, INPUT_BUTTONS.A],
  ];
  for (const [idx, [atMs, btn]] of phase2.entries()) {
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0, `idle before phase2 at ${atMs}`);
    assert.equal(player.sampleAt(atMs).buttonsMask, btn, `phase2 btn at ${atMs}`);
    assert.equal(player.sampleAt(atMs + 99).buttonsMask, btn);
    assert.equal(player.sampleAt(atMs + 100).buttonsMask, 0);
    assert.equal(player.pressCountAt(atMs), 3 + idx + 1);
  }

  // Phase 3: Intro cinematic skip (hold A 600ms, MENU 150ms, confirm A 100ms).
  assert.equal(player.sampleAt(30000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(30599).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(30600).buttonsMask, 0);
  assert.equal(player.sampleAt(32500).buttonsMask, INPUT_BUTTONS.MENU);
  assert.equal(player.sampleAt(32649).buttonsMask, INPUT_BUTTONS.MENU);
  assert.equal(player.sampleAt(32650).buttonsMask, 0);
  assert.equal(player.sampleAt(34500).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(34600).buttonsMask, 0);
  assert.equal(player.pressCountAt(34500), 14);

  // Phase 4: Main Menu Story/Play selection confirmations (38000, 40500, 43000, 45500).
  for (const [idx, atMs] of [38000, 40500, 43000, 45500].entries()) {
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0);
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `Main Menu A at ${atMs}`);
    assert.equal(player.sampleAt(atMs + 100).buttonsMask, 0);
    assert.equal(player.pressCountAt(atMs), 14 + idx + 1);
  }

  // Phase 5: In-game left stick, right stick, and LT/RT trigger probes (50000..59300 ms).
  assert.deepEqual(player.sampleAt(50000), {
    buttonsMask: 0,
    activeEventCount: 1,
    sticks: { lx: 0, ly: 0.85, rx: 0, ry: 0 },
  });
  assert.deepEqual(player.sampleAt(51500), { buttonsMask: 0, activeEventCount: 0 });

  assert.deepEqual(player.sampleAt(52000), {
    buttonsMask: 0,
    activeEventCount: 1,
    sticks: { lx: 0, ly: 0, rx: 0.75, ry: 0.25 },
  });
  assert.deepEqual(player.sampleAt(53200), { buttonsMask: 0, activeEventCount: 0 });

  assert.deepEqual(player.sampleAt(53700), {
    buttonsMask: 0,
    activeEventCount: 1,
    sticks: { lx: 0.6, ly: 0.4, rx: -0.75, ry: -0.2 },
  });
  assert.deepEqual(player.sampleAt(54900), { buttonsMask: 0, activeEventCount: 0 });

  assert.deepEqual(player.sampleAt(55500), {
    buttonsMask: 0,
    activeEventCount: 1,
    sticks: { lx: 0, ly: 0, rx: 0.35, ry: 0 },
    triggers: { lt: 0.85, rt: 0 },
  });
  assert.deepEqual(player.sampleAt(56500), { buttonsMask: 0, activeEventCount: 0 });

  assert.deepEqual(player.sampleAt(57000), {
    buttonsMask: 0,
    activeEventCount: 1,
    sticks: { lx: 0, ly: 0.9, rx: 0, ry: 0 },
    triggers: { lt: 0, rt: 0.9 },
  });
  assert.deepEqual(player.sampleAt(58000), { buttonsMask: 0, activeEventCount: 0 });

  assert.deepEqual(player.sampleAt(58500), {
    buttonsMask: 0,
    activeEventCount: 1,
    sticks: { lx: -0.5, ly: 0.5, rx: 0.5, ry: 0 },
    triggers: { lt: 0.5, rt: 0.75 },
  });

  // Bounded completion at 59300 ms: all axes/buttons released, no further transitions.
  assert.deepEqual(player.sampleAt(59300), { buttonsMask: 0, activeEventCount: 0 });
  assert.equal(player.nextTransitionMs(58500), 59300);
  assert.equal(player.nextTransitionMs(59300), null);
  assert.equal(player.pressCountAt(120000), 24);

  // Invariants over 120 seconds: NEVER presses B or DPAD_UP, axes bounded, idle after 59300 ms.
  for (let t = 0; t <= 120000; t += 50) {
    const sample = player.sampleAt(t);
    assert.equal(sample.buttonsMask & INPUT_BUTTONS.B, 0, `must never press B at t=${t}`);
    assert.equal(sample.buttonsMask & INPUT_BUTTONS.DPAD_UP, 0, `must never press DPAD_UP at t=${t}`);
    if (sample.sticks) {
      for (const axis of ['lx', 'ly', 'rx', 'ry']) {
        assert.ok(sample.sticks[axis] >= -1 && sample.sticks[axis] <= 1, `${axis} out of range at t=${t}`);
      }
    }
    if (sample.triggers) {
      for (const trig of ['lt', 'rt']) {
        assert.ok(sample.triggers[trig] >= 0 && sample.triggers[trig] <= 1, `${trig} out of range at t=${t}`);
      }
    }
    if (t >= 59300) {
      assert.deepEqual(sample, { buttonsMask: 0, activeEventCount: 0 }, `must stay idle after 59300 ms at t=${t}`);
    }
  }

  // maxPresses and endMs caps work cleanly on menu-to-ingame.
  const capped = new ScriptedInputPlayer({ preset: 'menu-to-ingame', maxPresses: 20 });
  assert.equal(capped.preset, 'menu-to-ingame');
  assert.deepEqual(capped.sampleAt(52000).sticks, { lx: 0, ly: 0, rx: 0.75, ry: 0.25 }); // #20
  assert.deepEqual(capped.sampleAt(53700), { buttonsMask: 0, activeEventCount: 0 });
  assert.equal(capped.pressCountAt(120000), 20);
  assert.equal(capped.nextTransitionMs(53200), null);

  const cappedEnd = new ScriptedInputPlayer('menu-to-ingame', { endMs: 56000 });
  assert.deepEqual(cappedEnd.sampleAt(55999).triggers, { lt: 0.85, rt: 0 });
  assert.deepEqual(cappedEnd.sampleAt(56000), { buttonsMask: 0, activeEventCount: 0 });
  assert.equal(cappedEnd.nextTransitionMs(55500), 56000);
  assert.equal(cappedEnd.nextTransitionMs(56000), null);
});

test('W297 InputBridge drives menu-to-ingame (buttons, sticks, triggers) into 40-byte SAB, merges with live inputs, and auto-completes cleanly', () => {
  const env = createMockEnv();
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });
  const dv = new DataView(bridge.sab);

  assert.ok(bridge.getStats().availablePresets.includes('menu-to-ingame'));

  let completedState = null;
  bridge.startScript('menu-to-ingame', {
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
    onComplete: (state) => {
      completedState = state;
    },
  });

  assert.equal(bridge.getInputTelemetry().activePreset, 'menu-to-ingame');
  assert.equal(bridge.getInputTelemetry().scriptedEventsDispatched, 0);

  // Phase 1 Start Screen A at 5000 ms.
  clock.advanceTo(5000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getInputTelemetry().scriptedEventsDispatched, 1);

  // Phase 2 Legal Notices DPAD_DOWN at 20000 ms.
  clock.advanceTo(20000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);

  // Phase 3 Intro skip MENU at 32500 ms.
  clock.advanceTo(32500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.MENU);

  // Phase 4 Main Menu A at 45500 ms.
  clock.advanceTo(45500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getInputTelemetry().scriptedEventsDispatched, 18);

  // Phase 5 In-game movement: left stick forward (50000 ms).
  clock.advanceTo(50000);
  let snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  approxEqual(snap.lx, 0);
  approxEqual(snap.ly, 0.85);
  approxEqual(dv.getFloat32(INPUT_OFFSETS.LEFT_STICK_Y, true), 0.85);

  // Phase 5 In-game camera look: right stick (52000 ms).
  clock.advanceTo(52000);
  snap = bridge.getSnapshot();
  approxEqual(snap.lx, 0);
  approxEqual(snap.ly, 0);
  approxEqual(snap.rx, 0.75);
  approxEqual(snap.ry, 0.25);
  approxEqual(dv.getFloat32(INPUT_OFFSETS.RIGHT_STICK_X, true), 0.75);
  approxEqual(dv.getFloat32(INPUT_OFFSETS.RIGHT_STICK_Y, true), 0.25);

  // Phase 5 In-game aim probe: right stick + LT trigger (55500 ms), merged with live mouse RT.
  clock.advanceTo(55500);
  env.lockPointer();
  env.win.dispatch('mousedown', { button: 0 }); // live mouse RT = 1.0
  snap = bridge.getSnapshot();
  approxEqual(snap.rx, 0.35);
  approxEqual(snap.lt, 0.85);
  approxEqual(snap.rt, 1.0);
  approxEqual(dv.getFloat32(INPUT_OFFSETS.LEFT_TRIGGER, true), 0.85);
  approxEqual(dv.getFloat32(INPUT_OFFSETS.RIGHT_TRIGGER, true), 1.0);

  // Window blur clears stuck human mouse RT while preserving active scripted LT & right stick!
  env.win.dispatch('blur');
  snap = bridge.getSnapshot();
  approxEqual(snap.rx, 0.35);
  approxEqual(snap.lt, 0.85);
  approxEqual(snap.rt, 0);
  assert.equal(bridge.getInputTelemetry().stuckResetCount, 1);

  // Phase 5 In-game accelerate probe: left stick + RT trigger (57000 ms).
  clock.advanceTo(57000);
  snap = bridge.getSnapshot();
  approxEqual(snap.ly, 0.9);
  approxEqual(snap.lt, 0);
  approxEqual(snap.rt, 0.9);
  approxEqual(dv.getFloat32(INPUT_OFFSETS.RIGHT_TRIGGER, true), 0.9);

  // Phase 5 Combined dual stick + dual triggers (58500 ms).
  clock.advanceTo(58500);
  snap = bridge.getSnapshot();
  approxEqual(snap.lx, -0.5);
  approxEqual(snap.ly, 0.5);
  approxEqual(snap.rx, 0.5);
  approxEqual(snap.ry, 0);
  approxEqual(snap.lt, 0.5);
  approxEqual(snap.rt, 0.75);

  // Auto-completion at 59300 ms: every channel zeroed, timers cleared, activePreset cleared.
  clock.advanceTo(59300);
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);
  assert.equal(snap.rx, 0);
  assert.equal(snap.ry, 0);
  assert.equal(snap.lt, 0);
  assert.equal(snap.rt, 0);
  assert.equal(clock.pendingTimeouts(), 0);
  assert.ok(completedState !== null && completedState.active === false);
  assert.equal(bridge.getScriptState().active, false);
  assert.equal(bridge.getInputTelemetry().activePreset, null);
  assert.equal(bridge.getInputTelemetry().scriptedEventsDispatched, 24);

  // Custom repeating trigger spec + stopScript() immediately clears held scripted triggers.
  bridge.startScript(
    {
      startMs: 0,
      holdMs: 500,
      periodMs: 1000,
      triggers: { left: 0.6, rightTrigger: 0.8 },
    },
    {
      nowFn: clock.nowFn,
      setTimeoutFn: clock.setTimeoutFn,
      clearTimeoutFn: clock.clearTimeoutFn,
    }
  );
  snap = bridge.getSnapshot();
  approxEqual(snap.lt, 0.6);
  approxEqual(snap.rt, 0.8);
  assert.equal(bridge.getInputTelemetry().scriptedEventsDispatched, 25);

  bridge.stopScript();
  snap = bridge.getSnapshot();
  assert.equal(snap.lt, 0);
  assert.equal(snap.rt, 0);
  assert.equal(bridge.getInputTelemetry().scriptedEventsDispatched, 25, 'stopScript must not increment dispatched count');

  // Invalid triggers specifier throws TypeError.
  assert.throws(
    () => new ScriptedInputPlayer([{ atMs: 0, holdMs: 100, triggers: 'invalid' }]),
    TypeError
  );

  bridge.destroy();
});

test('W297 zero-alloc getSnapshot({ reuse: true }) and getInputTelemetry() track pollsCount, writesCount, stuckResetCount, activePreset, and scriptedEventsDispatched', () => {
  const env = createMockEnv();
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  // 1. Initial telemetry is clean zeroed state.
  const initialTel = bridge.getInputTelemetry();
  assert.equal(initialTel.pollsCount, 0);
  assert.equal(initialTel.writesCount, 0);
  assert.equal(initialTel.stuckResetCount, 0);
  assert.equal(initialTel.activePreset, null);
  assert.equal(initialTel.scriptedEventsDispatched, 0);
  assert.equal(initialTel.seq, 0);
  assert.equal(initialTel.buttons, 0);
  assert.equal(initialTel.pointerLocked, false);
  assert.equal(initialTel.activeGamepadIndex, null);

  // 2. Zero-alloc getSnapshot({ reuse: true }) reuses the exact same internal object in-place.
  const reusedSnap1 = bridge.getSnapshot({ reuse: true });
  const reusedSnap2 = bridge.getSnapshot({ reuse: true });
  assert.equal(reusedSnap1, reusedSnap2, 'getSnapshot({ reuse: true }) must reuse internal scratch object');
  assert.equal(reusedSnap1.seq, 0);

  // Default getSnapshot() preserves backwards-compatible fresh-object behavior.
  const freshSnap1 = bridge.getSnapshot();
  const freshSnap2 = bridge.getSnapshot();
  assert.notEqual(freshSnap1, freshSnap2);
  assert.notEqual(freshSnap1, reusedSnap1);

  // getSnapshot({ out }) reuses caller-provided object.
  const callerSnap = {};
  assert.equal(bridge.getSnapshot({ out: callerSnap, maxRetries: 8 }), callerSnap);

  // 3. Zero-alloc getInputTelemetry({ reuse: true }) and getInputTelemetry(out) reuse in-place.
  const reusedTel1 = bridge.getInputTelemetry({ reuse: true });
  const reusedTel2 = bridge.getInputTelemetry({ reuse: true });
  assert.equal(reusedTel1, reusedTel2, 'getInputTelemetry({ reuse: true }) must reuse internal scratch object');
  const callerTel = {};
  assert.equal(bridge.getInputTelemetry(callerTel), callerTel);
  assert.equal(bridge.getInputTelemetry({ out: callerTel }), callerTel);
  assert.notEqual(bridge.getInputTelemetry(), reusedTel1);

  // 4. Gamepad poll/write path: steady-state polling increments pollsCount without spurious writes
  //    and reuses internal _selectedGamepadScratch (zero per-frame allocations).
  const pad = makeStandardGamepad({
    index: 0,
    buttons: { 0: 1, 6: 0.4, 7: 0.8 },
    axes: [0.6, -0.4, -0.5, 0.5],
  });
  env.setGamepads([pad]);

  const entry1 = bridge._selectGamepadEntry();
  const entry2 = bridge._selectGamepadEntry();
  assert.equal(entry1, entry2, '_selectGamepadEntry must reuse pre-allocated scratch entry');
  assert.equal(entry1, bridge._selectedGamepadScratch);

  assert.equal(bridge.pollGamepads(), true);
  assert.equal(bridge._selectedGamepadScratch.pad, null, 'pollGamepads must clear scratch pad ref');
  bridge.getInputTelemetry({ reuse: true });
  assert.equal(reusedTel1.pollsCount, 1);
  assert.equal(reusedTel1.writesCount, 1);
  assert.equal(reusedTel1.activeGamepadIndex, 0);
  assert.equal(reusedTel1.buttons, INPUT_BUTTONS.A);

  // 100 identical polls -> pollsCount += 100, writesCount unchanged.
  for (let i = 0; i < 100; i++) {
    assert.equal(bridge.pollGamepads(), false);
  }
  bridge.getInputTelemetry({ reuse: true });
  assert.equal(reusedTel1.pollsCount, 101);
  assert.equal(reusedTel1.writesCount, 1);

  // 5. Mousemove + keyboard writes increment writesCount and update reused snapshot in-place.
  env.lockPointer();
  env.win.dispatch('mousemove', { movementX: 11, movementY: -7 });
  env.win.dispatch('keydown', { code: 'KeyW' });
  bridge.getSnapshot({ reuse: true });
  assert.equal(reusedSnap1.mouse_dx, 11);
  assert.equal(reusedSnap1.mouse_dy, -7);
  assert.equal(reusedSnap1.buttons, INPUT_BUTTONS.A | INPUT_BUTTONS.MOVE_FORWARD);
  assert.equal(reusedSnap1.pointerLocked, true);
  bridge.getInputTelemetry({ reuse: true });
  assert.equal(reusedTel1.writesCount, 3);

  // 6. Seqlock odd sequence respected by getSnapshot({ reuse: true, maxRetries: 4 }).
  const savedSeq = reusedSnap1.seq;
  Atomics.store(bridge.u32, INPUT_WORD_INDICES.SEQ, savedSeq + 1);
  assert.equal(bridge.getSnapshot({ reuse: true, maxRetries: 4 }), null);
  Atomics.store(bridge.u32, INPUT_WORD_INDICES.SEQ, savedSeq + 2);
  assert.equal(bridge.getSnapshot({ reuse: true, maxRetries: 4 }), reusedSnap1);
  assert.equal(reusedSnap1.seq, savedSeq + 2);

  // 7. Blur / visibilitychange (hidden) / pagehide increment stuckResetCount (and writesCount).
  env.win.dispatch('blur');
  env.doc.hidden = false;
  env.doc.visibilityState = 'visible';
  env.doc.dispatch('visibilitychange'); // visible -> must NOT increment stuckResetCount
  env.doc.hidden = true;
  env.doc.visibilityState = 'hidden';
  env.doc.dispatch('visibilitychange');
  env.win.dispatch('pagehide');

  bridge.getInputTelemetry({ reuse: true });
  assert.equal(reusedTel1.stuckResetCount, 3);
  assert.equal(reusedTel1.writesCount, 6);
  assert.equal(bridge.getStats().stuckResetCount, 3);
  assert.equal(bridge.getStats().blurResets, 3);
  assert.equal(bridge.getStats().pollsCount, 101);
  assert.equal(bridge.getStats().writesCount, 6);

  // 8. Scripted playback updates activePreset and scriptedEventsDispatched in hot telemetry.
  bridge.startScript('menu-to-ingame', {
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });
  bridge.getInputTelemetry({ reuse: true });
  assert.equal(reusedTel1.activePreset, 'menu-to-ingame');
  assert.equal(reusedTel1.scriptedEventsDispatched, 0);

  clock.advanceTo(5000);
  bridge.getInputTelemetry({ reuse: true });
  assert.equal(reusedTel1.scriptedEventsDispatched, 1);
  assert.equal(reusedTel1.writesCount, 7);

  // Release at 5100 ms writes zeroed state to SAB without incrementing scriptedEventsDispatched.
  clock.advanceTo(5100);
  bridge.getInputTelemetry({ reuse: true });
  assert.equal(reusedTel1.scriptedEventsDispatched, 1);
  assert.equal(reusedTel1.writesCount, 8);
  assert.equal(bridge.getStats().scriptedEventsDispatched, 1);

  bridge.stopScript();
  bridge.getInputTelemetry({ reuse: true });
  assert.equal(reusedTel1.activePreset, null);
  assert.equal(reusedTel1.scriptedEventsDispatched, 1);

  bridge.destroy();
});

test('W339 chrome-menu-flow-2 preset mirrors PR #1333 (Start Screen A -> LEGAL NOTICES -> list screen A -> Down x1..5 + A -> 9s A loop) with zero B or DPAD_UP', () => {
  assert.ok(INPUT_SCRIPT_PRESET_NAMES.includes('chrome-menu-flow-2'));
  const spec = INPUT_SCRIPT_PRESETS['chrome-menu-flow-2'];
  assert.ok(spec);
  assert.equal(spec.name, 'chrome-menu-flow-2');
  assert.equal(spec.events.length, 53); // 23 Phase 1 + 9 Phase 2 + 1 Phase 3 + 20 Phase 4
  assert.equal(spec.events[0].atMs, 110000);
  assert.equal(spec.events[spec.events.length - 1].atMs, 339000);
  assert.equal(spec.loop.startMs, 360000);
  assert.equal(spec.loop.periodMs, 9000);
  assert.equal(spec.loop.events.length, 1);

  // Fixed event timestamps must be strictly increasing.
  for (let i = 1; i < spec.events.length; i++) {
    assert.ok(
      spec.events[i].atMs > spec.events[i - 1].atMs,
      `event ${i} (${spec.events[i].atMs}) must be strictly after event ${i - 1} (${spec.events[i - 1].atMs})`
    );
  }

  const player = new ScriptedInputPlayer('chrome-menu-flow-2');
  assert.equal(player.preset, 'chrome-menu-flow-2');
  assert.equal(player.startOffsetMs, 0);

  // Silent during Chrome Wasm64 intro & boot before 110000 ms.
  for (const t of [0, 20000, 60000, 105000, 109999]) {
    assert.deepEqual(player.sampleAt(t), { buttonsMask: 0, activeEventCount: 0 }, `silent at ${t}`);
    assert.equal(player.pressCountAt(t), 0);
  }

  // Phase 1: Human-paced A confirmations every 2500 ms from 110000 ms through 165000 ms (23 presses).
  let expectedCount = 0;
  for (let atMs = 110000; atMs <= 165000; atMs += 2500) {
    expectedCount += 1;
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0, `idle before Phase 1 A at ${atMs}`);
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `Phase 1 A at ${atMs}`);
    assert.equal(player.sampleAt(atMs + 99).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(atMs + 100).buttonsMask, 0);
    assert.equal(player.pressCountAt(atMs), expectedCount);
  }
  assert.equal(expectedCount, 23);

  // Quiet transition pause before LEGAL NOTICES (165100..168999 ms).
  for (let t = 165100; t < 169000; t += 100) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `quiet pause at ${t}`);
  }

  // Phase 2: 3 human-paced DPAD_DOWN (+0, 150ms), DPAD_DOWN (+2000, 150ms), A (+4500, 100ms)
  // cycles every 12000 ms from 169000 ms to select "Next >" on LEGAL NOTICES.
  for (let cycle = 0; cycle < 3; cycle++) {
    const base = 169000 + cycle * 12000;
    const cycleEvents = [
      [base, 150, INPUT_BUTTONS.DPAD_DOWN],
      [base + 2000, 150, INPUT_BUTTONS.DPAD_DOWN],
      [base + 4500, 100, INPUT_BUTTONS.A],
    ];
    for (const [atMs, holdMs, btn] of cycleEvents) {
      expectedCount += 1;
      assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0, `idle before Phase 2 at ${atMs}`);
      assert.equal(player.sampleAt(atMs).buttonsMask, btn, `Phase 2 press at ${atMs}`);
      assert.equal(player.sampleAt(atMs + holdMs - 1).buttonsMask, btn);
      assert.equal(player.sampleAt(atMs + holdMs).buttonsMask, 0);
      assert.equal(player.pressCountAt(atMs), expectedCount);
    }
  }
  assert.equal(expectedCount, 32);

  // Phase 3: Single A at 206500 ms (+9 s after last Phase 2 A at 197500 ms) leaves the list screen.
  for (let t = 197600; t < 206500; t += 500) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `quiet settle before list-screen A at ${t}`);
  }
  expectedCount += 1;
  assert.equal(player.sampleAt(206500).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(206599).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(206600).buttonsMask, 0);
  assert.equal(player.pressCountAt(206500), expectedCount);
  assert.equal(expectedCount, 33);

  // Phase 4: Cycling DPAD_DOWN x N + A (N = 1..5, spaced 25000 ms apart from 230000 ms).
  let phase4DDownCount = 0;
  for (let n = 1; n <= 5; n++) {
    const groupStart = 230000 + (n - 1) * 25000;
    for (let i = 0; i < n; i++) {
      expectedCount += 1;
      phase4DDownCount += 1;
      const atMs = groupStart + i * 1500;
      assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0);
      assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.DPAD_DOWN, `N=${n} DDown #${i + 1}`);
      assert.equal(player.sampleAt(atMs + 149).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
      assert.equal(player.sampleAt(atMs + 150).buttonsMask, 0);
      assert.equal(player.pressCountAt(atMs), expectedCount);
    }
    expectedCount += 1;
    const confirmAtMs = groupStart + n * 1500 + 1500;
    assert.equal(player.sampleAt(confirmAtMs - 1).buttonsMask, 0);
    assert.equal(player.sampleAt(confirmAtMs).buttonsMask, INPUT_BUTTONS.A, `N=${n} confirm A`);
    assert.equal(player.sampleAt(confirmAtMs + 99).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(confirmAtMs + 100).buttonsMask, 0);
    assert.equal(player.pressCountAt(confirmAtMs), expectedCount);
  }
  assert.equal(phase4DDownCount, 15); // 1 + 2 + 3 + 4 + 5
  assert.equal(expectedCount, 53);

  // Phase 5: Periodic A every 9000 ms starting at 360000 ms.
  for (let cycle = 0; cycle < 5; cycle++) {
    expectedCount += 1;
    const atMs = 360000 + cycle * 9000;
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0);
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `Phase 5 A cycle ${cycle}`);
    assert.equal(player.sampleAt(atMs + 99).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(atMs + 100).buttonsMask, 0);
    assert.equal(player.pressCountAt(atMs), expectedCount);
  }

  // Full 480-second invariant: NEVER presses B or DPAD_UP; only A or DPAD_DOWN.
  for (let t = 0; t <= 480000; t += 50) {
    const mask = player.sampleAt(t).buttonsMask;
    assert.equal(mask & INPUT_BUTTONS.B, 0, `chrome-menu-flow-2 must never press B at t=${t}`);
    assert.equal(mask & INPUT_BUTTONS.DPAD_UP, 0, `chrome-menu-flow-2 must never press DPAD_UP at t=${t}`);
    assert.ok(
      mask === 0 || mask === INPUT_BUTTONS.A || mask === INPUT_BUTTONS.DPAD_DOWN,
      `unexpected button mask ${mask} at t=${t}`
    );
  }

  // Exact nextTransitionMs across phase boundaries.
  assert.equal(player.nextTransitionMs(0), 110000);
  assert.equal(player.nextTransitionMs(165100), 169000);
  assert.equal(player.nextTransitionMs(197600), 206500);
  assert.equal(player.nextTransitionMs(206600), 230000);
  assert.equal(player.nextTransitionMs(339100), 360000);
  assert.equal(player.nextTransitionMs(360100), 369000);

  // maxPresses and endMs caps work cleanly on chrome-menu-flow-2.
  const capped = new ScriptedInputPlayer({ preset: 'chrome-menu-flow-2', maxPresses: 54 });
  assert.equal(capped.sampleAt(360000).buttonsMask, INPUT_BUTTONS.A); // #54 (1st loop A)
  assert.equal(capped.sampleAt(369000).buttonsMask, 0);
  assert.equal(capped.pressCountAt(450000), 54);
  assert.equal(capped.nextTransitionMs(360100), null);

  const cappedEnd = new ScriptedInputPlayer('chrome-menu-flow-2', { endMs: 206550 });
  assert.equal(cappedEnd.sampleAt(206500).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(cappedEnd.sampleAt(206549).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(cappedEnd.sampleAt(206550).buttonsMask, 0);
  assert.equal(cappedEnd.nextTransitionMs(206550), null);
});

test('W339 InputBridge drives chrome-menu-flow-2 into SAB through all 5 phases with accurate telemetry', () => {
  const env = createMockEnv();
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  assert.ok(bridge.getStats().availablePresets.includes('chrome-menu-flow-2'));

  bridge.startScript('chrome-menu-flow-2', {
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });

  assert.equal(bridge.getStats().activePreset, 'chrome-menu-flow-2');
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Phase 1 first A at 110000 ms.
  clock.advanceTo(110000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 1);
  assert.equal(bridge.getStats().lastScriptedButton, 'A');

  // Phase 2 LEGAL NOTICES DPAD_DOWN at 169000 ms (23 Phase 1 + 1 = 24).
  clock.advanceTo(169000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(bridge.getStats().scriptedPressesSent, 24);
  assert.equal(bridge.getStats().lastScriptedButton, 'DPAD_DOWN');

  // Phase 3 post-legal list-screen A at 206500 ms (32 + 1 = 33).
  clock.advanceTo(206500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 33);
  assert.equal(bridge.getStats().lastScriptedButton, 'A');

  // Phase 4 N=5 confirm A at 339000 ms (53rd fixed press).
  clock.advanceTo(339000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 53);

  // Phase 5 looping A at 360000 ms (54th) and 369000 ms (55th).
  clock.advanceTo(360000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 54);

  clock.advanceTo(369000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 55);

  bridge.stopScript();
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getStats().activePreset, null);
  assert.equal(bridge.getStats().preset, 'chrome-menu-flow-2');

  bridge.destroy();
});

test('W339 startOffsetMs shifts preset and custom timelines cleanly while keeping all existing presets 100% backwards compatible', () => {
  // 1. Anchoring menu-to-ingame to firstVisibleFrame at 105000 ms shifts its entire 5..59.3s timeline.
  const baseMenuToIngame = new ScriptedInputPlayer('menu-to-ingame');
  const shifted = new ScriptedInputPlayer('menu-to-ingame', { startOffsetMs: 105000 });
  assert.equal(baseMenuToIngame.startOffsetMs, 0);
  assert.equal(shifted.startOffsetMs, 105000);
  assert.equal(shifted.preset, 'menu-to-ingame');

  // Completely silent at unshifted menu-to-ingame timestamps (5000..59300 ms) and up to 109999 ms.
  for (const t of [0, 5000, 20000, 50000, 58500, 105000, 109999]) {
    assert.deepEqual(shifted.sampleAt(t), { buttonsMask: 0, activeEventCount: 0 }, `silent before offset at ${t}`);
    assert.equal(shifted.pressCountAt(t), 0);
  }
  assert.equal(shifted.nextTransitionMs(0), 110000);

  // Every shifted sample matches base sample at (t - 105000) across all 5 phases through completion.
  for (const baseT of [5000, 20000, 21200, 30000, 32500, 45500, 50000, 55500, 58500, 59300]) {
    assert.deepEqual(
      shifted.sampleAt(baseT + 105000),
      baseMenuToIngame.sampleAt(baseT),
      `shifted sample matches base at baseT=${baseT}`
    );
    assert.equal(
      shifted.pressCountAt(baseT + 105000),
      baseMenuToIngame.pressCountAt(baseT),
      `shifted pressCount matches base at baseT=${baseT}`
    );
  }
  assert.equal(shifted.nextTransitionMs(163500), 164300);
  assert.equal(shifted.nextTransitionMs(164300), null);

  // 2. Spec-object form `{ preset, startOffsetMs }` and options override precedence.
  const fromSpec = new ScriptedInputPlayer({ preset: 'menu-to-ingame', startOffsetMs: 50000 });
  assert.equal(fromSpec.startOffsetMs, 50000);
  assert.equal(fromSpec.nextTransitionMs(0), 55000);
  assert.equal(fromSpec.sampleAt(55000).buttonsMask, INPUT_BUTTONS.A);

  const overridden = new ScriptedInputPlayer(
    { preset: 'menu-to-ingame', startOffsetMs: 50000 },
    { startOffsetMs: 20000 }
  );
  assert.equal(overridden.startOffsetMs, 20000);
  assert.equal(overridden.nextTransitionMs(0), 25000);

  // 3. startOffsetMs shifts events + loop presets (chrome-menu-flow-2, legal-notices, chrome-menu-flow).
  const shiftedFlow2 = new ScriptedInputPlayer('chrome-menu-flow-2', { startOffsetMs: 15000 });
  assert.equal(shiftedFlow2.sampleAt(110000).buttonsMask, 0);
  assert.equal(shiftedFlow2.sampleAt(125000).buttonsMask, INPUT_BUTTONS.A); // 110000 + 15000
  assert.equal(shiftedFlow2.sampleAt(184000).buttonsMask, INPUT_BUTTONS.DPAD_DOWN); // 169000 + 15000
  assert.equal(shiftedFlow2.sampleAt(221500).buttonsMask, INPUT_BUTTONS.A); // 206500 + 15000
  assert.equal(shiftedFlow2.sampleAt(375000).buttonsMask, INPUT_BUTTONS.A); // loop 360000 + 15000
  assert.equal(shiftedFlow2.nextTransitionMs(0), 125000);

  const shiftedLegal = new ScriptedInputPlayer('legal-notices', { startOffsetMs: 10000 });
  assert.equal(shiftedLegal.sampleAt(20000).buttonsMask, 0);
  assert.equal(shiftedLegal.sampleAt(30000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(shiftedLegal.sampleAt(74000).buttonsMask, INPUT_BUTTONS.DPAD_DOWN); // loop 64000 + 10000

  // 4. startOffsetMs shifts repeating presets (start-screen) and custom event arrays cleanly.
  const shiftedStart = new ScriptedInputPlayer('start-screen', { startOffsetMs: 10000 });
  assert.equal(shiftedStart.sampleAt(5000).buttonsMask, 0);
  assert.equal(shiftedStart.sampleAt(14999).buttonsMask, 0);
  assert.equal(shiftedStart.sampleAt(15000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(shiftedStart.nextTransitionMs(0), 15000);

  const shiftedArray = new ScriptedInputPlayer(
    [{ atMs: 1000, holdMs: 200, button: INPUT_BUTTONS.X }],
    { startOffsetMs: 4000 }
  );
  assert.equal(shiftedArray.sampleAt(1000).buttonsMask, 0);
  assert.equal(shiftedArray.sampleAt(5000).buttonsMask, INPUT_BUTTONS.X);
  assert.equal(shiftedArray.sampleAt(5200).buttonsMask, 0);

  // 5. startOffsetMs composes cleanly with endMs and maxPresses caps.
  const shiftedCappedEnd = new ScriptedInputPlayer('menu-to-ingame', {
    startOffsetMs: 100000,
    endMs: 120050,
  });
  assert.equal(shiftedCappedEnd.sampleAt(120000).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(shiftedCappedEnd.sampleAt(120049).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(shiftedCappedEnd.sampleAt(120050).buttonsMask, 0);
  assert.equal(shiftedCappedEnd.nextTransitionMs(120050), null);

  // 6. InputBridge.startScript forwards startOffsetMs cleanly and auto-completes at shifted end.
  const env = createMockEnv();
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });
  let completed = false;
  const livePlayer = bridge.startScript('menu-to-ingame', {
    startOffsetMs: 105000,
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
    onComplete: () => {
      completed = true;
    },
  });
  assert.equal(livePlayer.startOffsetMs, 105000);
  clock.advanceTo(5000);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getStats().scriptedPressesSent, 0);

  clock.advanceTo(110000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 1);

  clock.advanceTo(164300);
  assert.equal(completed, true);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getScriptState().active, false);
  bridge.destroy();

  // 7. Validation: invalid startOffsetMs throws RangeError.
  for (const bad of [-1, -100, Number.NaN, Number.POSITIVE_INFINITY, '100']) {
    assert.throws(
      () => new ScriptedInputPlayer('menu-to-ingame', { startOffsetMs: bad }),
      RangeError,
      `expected RangeError for startOffsetMs=${String(bad)}`
    );
    assert.throws(
      () => new ScriptedInputPlayer('start-screen', { startOffsetMs: bad }),
      RangeError,
      `expected RangeError for repeating startOffsetMs=${String(bad)}`
    );
  }

  // 8. Backwards compatibility: all existing presets default to startOffsetMs=0 with exact original first transitions.
  const expectedFirstTransitions = [
    ['start-screen', 5000],
    ['menu-flow', 5000],
    ['calibration-skip', 5000],
    ['intro-watch', 18000],
    ['legal-notices', 20000],
    ['chrome-menu-flow', 165000],
    ['chrome-full-boot', 235000],
    ['menu-to-ingame', 5000],
    ['chrome-menu-flow-2', 110000],
    ['chrome-menu-flow-4', 5000],
    ['chrome-menu-flow-5', 2000],
    ['chrome-to-ingame', 5000],
  ];
  for (const [name, expectedFirstMs] of expectedFirstTransitions) {
    const p = new ScriptedInputPlayer(name);
    assert.equal(p.startOffsetMs, 0, `${name} default startOffsetMs must be 0`);
    assert.equal(p.nextTransitionMs(0), expectedFirstMs, `${name} first transition mismatch`);
  }
});

test('W364 InputBridge throttles idle navigator.getGamepads() polling to 2 Hz and wakes immediately on gamepadconnected or first-tick detection', () => {
  assert.equal(IDLE_GAMEPAD_POLL_INTERVAL_MS, 500);

  let nowMs = 0;
  let getGamepadsCalls = 0;
  const env = createMockEnv({ withRaf: true });
  const baseGetGamepads = env.nav.getGamepads;
  env.nav.getGamepads = () => {
    getGamepadsCalls += 1;
    return baseGetGamepads();
  };

  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
    nowFn: () => nowMs,
  });

  // 1. First tick always polls immediately even when idle (detects pre-connected/mock pads on tick 1).
  bridge.start();
  env.flushRaf();
  assert.equal(getGamepadsCalls, 1, 'tick 1 must poll getGamepads() immediately');
  assert.equal(bridge.hasConnectedGamepad, false);

  // 2. Subsequent 60 Hz ticks within 500 ms skip navigator.getGamepads() completely.
  for (let i = 1; i < 30; i += 1) {
    nowMs = i * 16; // 16..464 ms (< 500 ms)
    env.flushRaf();
  }
  assert.equal(getGamepadsCalls, 1, 'idle rAF ticks < 500 ms must not call getGamepads()');
  assert.equal(bridge.getStats().idleGamepadPollsSkipped, 29);

  // 3. At >= 500 ms, exactly one idle probe fires (2 Hz).
  nowMs = 500;
  env.flushRaf();
  assert.equal(getGamepadsCalls, 2, 'idle probe must fire once at 500 ms');
  nowMs = 750;
  env.flushRaf();
  assert.equal(getGamepadsCalls, 2, 'idle ticks between 500 ms and 1000 ms must remain throttled');

  // 4. gamepadconnected wakes up 60 Hz polling immediately with zero latency.
  const pad = makeStandardGamepad({ buttons: { 0: 1 } }); // A
  env.setGamepads([pad]);
  env.win.dispatch('gamepadconnected', { gamepad: pad });
  assert.equal(bridge.hasConnectedGamepad, true);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  const callsAfterConnect = getGamepadsCalls;

  for (let i = 1; i <= 5; i += 1) {
    nowMs += 16;
    env.flushRaf();
  }
  assert.equal(
    getGamepadsCalls,
    callsAfterConnect + 5,
    'connected gamepad must poll on every 60 Hz rAF tick'
  );

  // 5. gamepaddisconnected returns to 500 ms idle throttling.
  env.setGamepads([null]);
  env.win.dispatch('gamepaddisconnected', { gamepad: pad });
  assert.equal(bridge.hasConnectedGamepad, false);
  const callsAfterDisconnect = getGamepadsCalls;

  nowMs += 100;
  env.flushRaf();
  nowMs += 100;
  env.flushRaf();
  assert.equal(
    getGamepadsCalls,
    callsAfterDisconnect,
    'after disconnect, ticks within 500 ms must stay throttled'
  );

  // 6. Plugging in a gamepad without a gamepadconnected event is picked up on the next 500 ms probe
  //    and immediately flips hasConnectedGamepad = true for 60 Hz polling.
  const silentPad = makeStandardGamepad({ buttons: { 3: 1 } }); // Y
  env.setGamepads([silentPad]);
  nowMs += 300; // total 500 ms since disconnect
  env.flushRaf();
  assert.equal(bridge.hasConnectedGamepad, true);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.Y);
  const callsAfterSilentDetect = getGamepadsCalls;
  nowMs += 16;
  env.flushRaf();
  assert.equal(getGamepadsCalls, callsAfterSilentDetect + 1);

  bridge.destroy();

  // 7. Pre-connected mock gamepad is detected immediately on tick 1 after start().
  let preCalls = 0;
  const preEnv = createMockEnv({
    withRaf: true,
    gamepads: [makeStandardGamepad({ buttons: { 2: 1 } })], // X
  });
  const basePre = preEnv.nav.getGamepads;
  preEnv.nav.getGamepads = () => {
    preCalls += 1;
    return basePre();
  };
  const preBridge = new InputBridge(preEnv.canvas, {
    windowObj: preEnv.win,
    documentObj: preEnv.doc,
    navigatorObj: preEnv.nav,
  });
  preBridge.start();
  preEnv.flushRaf();
  assert.equal(preCalls, 1);
  assert.equal(preBridge.hasConnectedGamepad, true);
  assert.equal(preBridge.getSnapshot().buttons, INPUT_BUTTONS.X);
  preBridge.destroy();
});

test('W364 sanitizeString fast-path and clean cache preserve exact redaction on paths, secrets, tokens, and buffers', () => {
  // Clean telemetry strings stay unchanged across repeated calls (hitting clean cache).
  const cleanSamples = [
    'Real Wasm64',
    'Draws: 48 real (120 presents)',
    'GPU: nvidia / ada lovelace',
    'GET /api/status 200 OK',
    'guestVa=0x140045678 offset=+0x2c hash=0x123456789abcdef0',
    'WebGPU frame #120 ready? Yes.',
    '0 WebGPU validation errors',
  ];
  for (const sample of cleanSamples) {
    assert.equal(sanitizeString(sample), sample);
    assert.equal(sanitizeString(sample), sample);
  }

  // Host paths, file:// URLs, query strings, auth/secrets, bare tokens, and binary/hex/base64 buffers
  // must all still be redacted identically.
  assert.equal(
    sanitizeString('Failed C:\\Users\\alice\\GTA6\\game.exe at startup'),
    'Failed [REDACTED_PATH] at startup'
  );
  assert.equal(
    sanitizeString('Cannot open /home/bob/secret.bin or /Users/carol/data.rpf'),
    'Cannot open [REDACTED_PATH] or [REDACTED_PATH]'
  );
  assert.equal(
    sanitizeString('at boot (file:///C:/Users/alice/playgta6web/web/loader.js:42:10)'),
    'at boot ([REDACTED_FILE_URL]:42:10)'
  );
  assert.equal(
    sanitizeString('Mirror \\\\nas01\\share\\alice\\dump.bin and //nas02/share/bob/chunk.rpf'),
    'Mirror [REDACTED_PATH] and [REDACTED_PATH]'
  );
  assert.equal(
    sanitizeString('Config at ~/secrets/keys.json'),
    'Config at [REDACTED_PATH]'
  );
  assert.equal(
    sanitizeString('GET https://alice:pass@proxy.local/api?token=secret123'),
    'GET https://[REDACTED]@proxy.local/api?[REDACTED]'
  );
  assert.equal(sanitizeString('Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.sig12345678'), 'Authorization: Bearer [REDACTED]');
  assert.equal(sanitizeString('Basic dXNlcjpwYXNz'), 'Basic [REDACTED]');
  assert.equal(sanitizeString('token=supersecret123'), 'token=[REDACTED]');
  assert.equal(sanitizeString('ghp_1234567890abcdef1234567890abcdef'), '[REDACTED_TOKEN]');
  assert.equal(sanitizeString('AKIAIOSFODNN7EXAMPLE'), '[REDACTED_TOKEN]');
  assert.equal(
    sanitizeString('data:application/octet-stream;base64,QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo='),
    '[REDACTED_BUFFER]'
  );
  assert.equal(
    sanitizeString('00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10 11'),
    '[REDACTED_BUFFER]'
  );
  assert.equal(sanitizeString('a'.repeat(70)), '[REDACTED_BUFFER]');
  assert.equal(sanitizeString('\x00asm\x01\x00\x00\x00'), '[REDACTED_BUFFER]');
});

test('W364 HudController skips visual-only DOM churn while hidden, keeps core telemetry intact, and flushes on setVisible(true)', () => {
  assert.equal(HIDDEN_HUD_MIN_RENDER_INTERVAL_MS, 500);

  let nowMs = 1000;
  const makeEl = () => ({
    textContent: '',
    title: '',
    hidden: false,
    dataset: {},
    style: {},
    attrs: new Map(),
    setAttribute(k, v) {
      this.attrs.set(k, v);
    },
    getAttribute(k) {
      return this.attrs.has(k) ? this.attrs.get(k) : null;
    },
  });

  const hudRootEl = makeEl();
  const framesEl = makeEl();
  const fpsEl = makeEl();
  const pillFpsEl = makeEl();
  const bootStageEl = makeEl();
  const bootModeBadge = makeEl();

  const controller = new HudController({
    bootMode: 'real-wasm64',
    visible: false,
    now: () => nowMs,
    displayElements: {
      hudRootEl,
      framesEl,
      fpsEl,
      pillFpsEl,
      bootStageEl,
      bootModeBadge,
    },
  });

  // First telemetry update while hidden updates core telemetry elements (frames, fps)
  // while skipping visual-only elements (pillFpsEl).
  controller.updateTelemetry({
    gpu: { presentCount: 10, fps: 60, gpuAdapter: 'nvidia / ada lovelace' },
  });
  assert.equal(framesEl.textContent, '10');
  assert.equal(fpsEl.textContent, '60');
  assert.equal(pillFpsEl.textContent, '', 'visual-only pillFpsEl must be skipped while HUD hidden');

  // Rapid updates within 500 ms while hidden are throttled with zero additional DOM writes.
  const writesAfterFirst = controller.domWrites;
  nowMs = 1050;
  controller.updateTelemetry({ gpu: { presentCount: 11, fps: 59, gpuAdapter: 'nvidia / ada lovelace' } });
  nowMs = 1100;
  controller.updateTelemetry({ gpu: { presentCount: 12, fps: 58, gpuAdapter: 'nvidia / ada lovelace' } });
  assert.equal(controller.domWrites, writesAfterFirst);
  assert.equal(controller.getRenderStats().throttledRenders, 2);
  assert.equal(controller.getTelemetry().gpu.presentCount, 12);

  // After >= 500 ms while hidden, core telemetry updates cleanly.
  nowMs = 1550;
  controller.updateTelemetry({ gpu: { presentCount: 25, fps: 60, gpuAdapter: 'nvidia / ada lovelace' } });
  assert.equal(framesEl.textContent, '25');
  assert.equal(pillFpsEl.textContent, '', 'pillFpsEl still skipped while hidden');

  // Making HUD visible immediately flushes all deferred visual elements.
  controller.setVisible(true);
  assert.equal(hudRootEl.hidden, false);
  assert.equal(pillFpsEl.textContent, '60 fps');
  assert.equal(framesEl.textContent, '25');
  controller.destroy();
});

test('W364 chrome-menu-flow-5 preset schedule, FIRST_VISIBLE_FRAME anchor, and 16 s Legal Terms loop', () => {
  assert.ok(INPUT_SCRIPT_PRESET_NAMES.includes('chrome-menu-flow-5'));
  assert.ok(FIRST_VISIBLE_FRAME_ANCHOR_PRESETS.includes('chrome-menu-flow-5'));

  const spec = INPUT_SCRIPT_PRESETS['chrome-menu-flow-5'];
  assert.ok(spec);
  assert.equal(spec.name, 'chrome-menu-flow-5');
  assert.equal(spec.events.length, 30, '3 Start Screen A + 3x9 LEGAL NOTICES cycle events = 30');
  assert.ok(spec.loop);
  assert.equal(spec.loop.startMs, 85000);
  assert.equal(spec.loop.periodMs, 16000);
  assert.equal(spec.loop.events, CHROME_LEGAL_TERMS_ACCEPT_OFFSETS);

  // Every event hold must be >= MIN_SCRIPT_HOLD_MS (180 ms) and B must never be pressed.
  for (const ev of spec.events) {
    assert.ok(ev.holdMs >= MIN_SCRIPT_HOLD_MS, `holdMs ${ev.holdMs} < ${MIN_SCRIPT_HOLD_MS}`);
    assert.equal((ev.button || 0) & INPUT_BUTTONS.B, 0, 'chrome-menu-flow-5 must never press B');
  }
  for (const ev of spec.loop.events) {
    assert.ok(ev.holdMs >= MIN_SCRIPT_HOLD_MS);
    assert.equal((ev.button || 0) & INPUT_BUTTONS.B, 0);
  }

  const player = new ScriptedInputPlayer('chrome-menu-flow-5');

  // Phase 1 (0s..10s): Start Screen A at 2s, 4s, 6s.
  assert.equal(player.sampleAt(0).buttonsMask, 0);
  assert.equal(player.sampleAt(1999).buttonsMask, 0);
  for (const atMs of [2000, 4000, 6000]) {
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `Start Screen A at ${atMs}`);
    assert.equal(player.sampleAt(atMs + 199).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(atMs + 200).buttonsMask, 0);
  }

  // Quiet window during 4K Bink2 playback (6200..34999 ms).
  for (let t = 6200; t < 35000; t += 1000) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `quiet during Bink2 at ${t}`);
  }

  // Phase 2 (35s..80s): 3x (Up x6, Down, Down, A) rounds at 35s, 50s, 65s.
  for (const base of [35000, 50000, 65000]) {
    for (let u = 0; u < 6; u += 1) {
      assert.equal(player.sampleAt(base + u * 400).buttonsMask, INPUT_BUTTONS.DPAD_UP);
    }
    assert.equal(player.sampleAt(base + 3000).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
    assert.equal(player.sampleAt(base + 4500).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
    assert.equal(player.sampleAt(base + 6500).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(base + 6700).buttonsMask, 0);
  }

  // Quiet settle window before Legal Terms loop (71700..84999 ms).
  assert.equal(player.sampleAt(75000).buttonsMask, 0);
  assert.equal(player.sampleAt(84999).buttonsMask, 0);

  // Phase 3 (85s+, looping every 16 s): CHROME_LEGAL_TERMS_ACCEPT_OFFSETS.
  for (let cycle = 0; cycle < 3; cycle += 1) {
    const base = 85000 + cycle * 16000;
    assert.equal(player.sampleAt(base).buttonsMask, INPUT_BUTTONS.DPAD_UP);
    const scroll1 = player.sampleAt(base + 2700);
    assert.equal(scroll1.buttonsMask, 0);
    assert.ok(scroll1.sticks && scroll1.sticks.ry < -0.9, 'right-stick scroll down for Privacy Policy');
    assert.equal(player.sampleAt(base + 4000).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(base + 5200).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
    const scroll2 = player.sampleAt(base + 5800);
    assert.ok(scroll2.sticks && scroll2.sticks.ry < -0.9, 'right-stick scroll down for Terms of Service');
    assert.equal(player.sampleAt(base + 7100).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(base + 8300).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
    assert.equal(player.sampleAt(base + 9500).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(base + 12000).buttonsMask, INPUT_BUTTONS.A);
  }

  // Re-anchoring on FIRST_VISIBLE_FRAME prepends 500 ms A without overlapping 2s/4s/6s presses.
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  bridge.startScript('chrome-menu-flow-5', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME,
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });
  clock.advanceTo(30000);
  assert.equal(bridge.notifyFirstVisibleFrame(), true);
  assert.equal(bridge.getStats().scriptStartedOn, INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME);

  clock.advanceTo(30500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A, 'initial 500 ms A after FIRST_VISIBLE_FRAME');
  clock.advanceTo(30720);
  assert.equal(bridge.getSnapshot().buttons, 0);
  clock.advanceTo(32000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A, '2 s Start Screen A');
  clock.advanceTo(30000 + 85000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP, '85 s Legal Terms loop start');
  bridge.destroy();
});
