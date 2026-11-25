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
  applyDeadzone,
  applyAxialDeadzone,
  applyTriggerDeadzone,
  INPUT_SCRIPT_PRESETS,
  INPUT_SCRIPT_PRESET_NAMES,
  ScriptedInputPlayer,
  InputBridge,
} from '../web/input/input_bridge.js';

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
  ]);
  assert.deepEqual(Object.keys(INPUT_SCRIPT_PRESETS), [...INPUT_SCRIPT_PRESET_NAMES]);

  for (const name of INPUT_SCRIPT_PRESET_NAMES) {
    const spec = INPUT_SCRIPT_PRESETS[name];
    assert.ok(spec && typeof spec === 'object', `Preset ${name} must exist`);
    assert.equal(Object.isFrozen(spec), true, `Preset ${name} must be frozen`);
    assert.equal(spec.name, name);
    assert.match(name, /^[a-z0-9][a-z0-9_-]{0,63}$/);
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
