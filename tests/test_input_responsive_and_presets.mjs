import test from 'node:test';
import assert from 'node:assert/strict';

import {
  INPUT_BUTTONS,
  DEFAULT_MIN_PRESS_HOLD_MS,
  LIVE_TAP_LATCH_BUTTON_MASK,
  DEFAULT_RIGHT_STICK_KEYS,
  applyDeadzone,
  applyRadialDeadzone,
  ScriptedInputPlayer,
  InputBridge,
} from '../web/input/input_bridge.js';

function approxEqual(actual, expected, eps = 1e-5) {
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `Expected ${actual} to be within ${eps} of ${expected}`
  );
}

class MockEventTarget {
  constructor() {
    this._handlers = new Map();
  }
  addEventListener(type, handler) {
    if (!this._handlers.has(type)) this._handlers.set(type, new Set());
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
    for (const fn of [...set]) fn(event);
  }
}

function createFakeClockAndTimers(initialNow = 0) {
  let now = initialNow;
  let nextId = 1;
  const timeouts = new Map();
  function setTimeoutFn(cb, delay = 0) {
    const id = nextId++;
    timeouts.set(id, { id, cb, due: now + Math.max(0, delay) });
    return id;
  }
  function clearTimeoutFn(id) {
    timeouts.delete(id);
  }
  function advanceTo(targetMs) {
    assert.ok(targetMs >= now, `Cannot move fake clock backwards (${now} -> ${targetMs})`);
    while (true) {
      let nextDue = Infinity;
      let entry = null;
      for (const item of timeouts.values()) {
        if (item.due <= targetMs && item.due < nextDue) {
          nextDue = item.due;
          entry = item;
        }
      }
      if (!entry) break;
      now = nextDue;
      timeouts.delete(entry.id);
      entry.cb();
    }
    now = targetMs;
  }
  return {
    nowFn: () => now,
    setTimeoutFn,
    clearTimeoutFn,
    advanceTo,
    advanceBy(deltaMs) {
      advanceTo(now + deltaMs);
    },
    pendingTimeouts: () => timeouts.size,
  };
}

function createMockEnv({ gamepads = [], withRaf = false, focused = true } = {}) {
  const canvas = new MockEventTarget();
  canvas.requestPointerLock = () => Promise.resolve();
  const doc = new MockEventTarget();
  doc.pointerLockElement = null;
  doc.hidden = false;
  doc.visibilityState = 'visible';
  let isFocused = focused;
  doc.hasFocus = () => isFocused;
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
  let currentPads = gamepads;
  const nav = { getGamepads: () => currentPads };
  return {
    canvas,
    doc,
    win,
    nav,
    setFocused(val) {
      isFocused = val;
    },
    setGamepads(pads) {
      currentPads = pads;
    },
    lockPointer() {
      doc.pointerLockElement = canvas;
      doc.dispatch('pointerlockchange');
    },
    unlockPointer() {
      doc.pointerLockElement = null;
      doc.dispatch('pointerlockchange');
    },
    flushRaf(ts = 16.67) {
      const entries = [...rafQueue.entries()];
      rafQueue.clear();
      for (const [, cb] of entries) cb(ts);
      return entries.length;
    },
    pendingRafCount: () => rafQueue.size,
  };
}

function makeStandardGamepad({ buttons = {}, axes = [0, 0, 0, 0], index = 0, connected = true } = {}) {
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
    id: `Xbox Wireless Controller #${index}`,
    index,
    connected,
    mapping: 'standard',
    buttons: btnArray,
    axes: axes.slice(),
  };
}

test('live tap-hold latching (minPressHoldMs) preserves rapid keyboard/gamepad taps across low-FPS polls without making WASD floaty', () => {
  assert.ok(DEFAULT_MIN_PRESS_HOLD_MS >= 60);
  assert.equal(LIVE_TAP_LATCH_BUTTON_MASK & INPUT_BUTTONS.MOVE_FORWARD, 0);
  assert.equal(LIVE_TAP_LATCH_BUTTON_MASK & INPUT_BUTTONS.A, INPUT_BUTTONS.A);

  const env = createMockEnv();
  const clock = createFakeClockAndTimers(1000);
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
    minPressHoldMs: 100,
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });

  // 1. Quick 16 ms Enter tap stays asserted in SAB until t = 1100 ms (survives 10-15 FPS polls!).
  env.win.dispatch('keydown', { code: 'Enter' });
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceBy(16);
  env.win.dispatch('keyup', { code: 'Enter' });
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A, 'quick 16 ms tap must stay latched');
  assert.equal(bridge.getStats().latchedButtons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().tapLatchesTriggered, 1);

  // Guest polls at 15 FPS (67 ms after keydown) -> still sees A!
  clock.advanceTo(1067);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(1099);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // At t = 1100 ms latch expires cleanly with zero pending timers.
  clock.advanceTo(1100);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getStats().latchedButtons, 0);
  assert.equal(clock.pendingTimeouts(), 0);

  // 2. Movement keys (WASD) release immediately on keyup so 3D movement never feels floaty.
  env.win.dispatch('keydown', { code: 'KeyW' });
  clock.advanceBy(16);
  env.win.dispatch('keyup', { code: 'KeyW' });
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getSnapshot().ly, 0);
  assert.equal(clock.pendingTimeouts(), 0);

  // 3. Long hold (>= minPressHoldMs) releases immediately on keyup without extra delay.
  env.win.dispatch('keydown', { code: 'ArrowDown' });
  clock.advanceBy(120);
  env.win.dispatch('keyup', { code: 'ArrowDown' });
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(clock.pendingTimeouts(), 0);

  // 4. Rapid gamepad button tap is latched across low-FPS polls too.
  env.setGamepads([makeStandardGamepad({ buttons: { 0: 1 } })]);
  bridge.pollGamepads();
  clock.advanceBy(20);
  env.setGamepads([makeStandardGamepad({ buttons: {} })]);
  bridge.pollGamepads();
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceBy(80);
  assert.equal(bridge.getSnapshot().buttons, 0);

  // 5. Window blur and setMinPressHoldMs(0) cancel active tap latches immediately.
  env.win.dispatch('keydown', { code: 'Space' });
  clock.advanceBy(10);
  env.win.dispatch('keyup', { code: 'Space' });
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.JUMP | INPUT_BUTTONS.A);
  env.win.dispatch('blur');
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getStats().latchedButtons, 0);
  assert.equal(clock.pendingTimeouts(), 0);

  env.win.dispatch('keydown', { code: 'Tab' });
  clock.advanceBy(10);
  env.win.dispatch('keyup', { code: 'Tab' });
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.VIEW);
  assert.equal(bridge.setMinPressHoldMs(0), 0);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(clock.pendingTimeouts(), 0);

  bridge.destroy();
});

test('Pointer Lock unlock zeroes unconsumed SAB mouse deltas and wheelDelta even with zero held buttons/keys', () => {
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  env.lockPointer();
  env.win.dispatch('mousemove', { movementX: 42, movementY: -18 });
  env.win.dispatch('wheel', { deltaY: 240 });
  let snap = bridge.getSnapshot();
  assert.equal(snap.mouse_dx, 42);
  assert.equal(snap.mouse_dy, -18);
  assert.equal(snap.wheelDelta, 240);

  // No keyboard keys or mouse buttons are held, yet unlock MUST clear residual SAB deltas & wheel!
  env.unlockPointer();
  snap = bridge.getSnapshot();
  assert.equal(snap.mouse_dx, 0);
  assert.equal(snap.mouse_dy, 0);
  assert.equal(snap.wheelDelta, 0);

  // Blur also clears wheelDelta cleanly.
  env.lockPointer();
  env.win.dispatch('wheel', { deltaY: -120 });
  assert.equal(bridge.getSnapshot().wheelDelta, -120);
  env.win.dispatch('blur');
  assert.equal(bridge.getSnapshot().wheelDelta, 0);

  bridge.destroy();
});

test('multi-gamepad connect/disconnect preserves active controller, unfocused rAF skips polling, and swapped triggers respect triggerDeadzone', () => {
  const env = createMockEnv({ withRaf: true, focused: true });
  const pad0 = makeStandardGamepad({ index: 0, buttons: { 0: 1 } });
  const pad1 = makeStandardGamepad({ index: 1, buttons: { 3: 1 } });
  env.setGamepads([pad0]);

  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });
  bridge.start();
  env.flushRaf();
  assert.equal(bridge.getStats().activeGamepadIndex, 0);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // Connecting secondary Pad 1 must NOT hijack active Pad 0!
  env.setGamepads([pad0, pad1]);
  env.win.dispatch('gamepadconnected', { gamepad: pad1 });
  assert.equal(bridge.getStats().activeGamepadIndex, 0);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // Disconnecting inactive Pad 1 must NOT clear Pad 0 or bump writesCount spuriously!
  const writesBefore = bridge.getStats().writesCount;
  env.setGamepads([pad0, null]);
  env.win.dispatch('gamepaddisconnected', { gamepad: pad1 });
  assert.equal(bridge.getStats().activeGamepadIndex, 0);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().writesCount, writesBefore);

  // Unfocused document skips rAF gamepad polling while keeping rAF armed.
  env.win.dispatch('blur');
  env.setFocused(false);
  const pollsBeforeBlur = bridge.getStats().pollsCount;
  env.flushRaf();
  assert.equal(bridge.getStats().pollsCount, pollsBeforeBlur);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(env.pendingRafCount(), 1);

  // Refocusing resumes rAF polling immediately.
  env.setFocused(true);
  env.flushRaf();
  assert.ok(bridge.getStats().pollsCount > pollsBeforeBlur);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  bridge.destroy();

  // Swapped triggers-as-bumpers filters analog trigger noise below triggerDeadzone.
  const swapEnv = createMockEnv();
  const swapBridge = new InputBridge(swapEnv.canvas, {
    windowObj: swapEnv.win,
    documentObj: swapEnv.doc,
    navigatorObj: swapEnv.nav,
    swapTriggersAndBumpers: true,
    triggerDeadzone: 0.1,
  });
  swapEnv.setGamepads([
    makeStandardGamepad({
      buttons: {
        6: { pressed: true, value: 0.02 }, // noise below triggerDeadzone -> must NOT assert LB
        7: { pressed: true, value: 0.85 }, // real pull -> asserts RB
      },
    }),
  ]);
  swapBridge.pollGamepads();
  assert.equal(swapBridge.getSnapshot().buttons, INPUT_BUTTONS.RB);
  swapBridge.destroy();
});

test('keyboard right-stick keys (PageUp/PageDown) and opt-in mouseLookToRightStick/wheelToRightStickY drive RIGHT_STICK cleanly', () => {
  assert.equal(DEFAULT_RIGHT_STICK_KEYS.down, 'PageDown');
  const env = createMockEnv({ withRaf: true });
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
    mouseLookToRightStick: true,
    wheelToRightStickY: true,
    mouseLookPixelsForFullDeflection: 40,
  });

  // PageDown drives right stick down (ry = -1) for Legal Terms scrolling without a gamepad!
  env.win.dispatch('keydown', { code: 'PageDown' });
  approxEqual(bridge.getSnapshot().ry, -1.0);
  env.win.dispatch('keydown', { code: 'Numpad6' });
  approxEqual(bridge.getSnapshot().rx, Math.SQRT1_2);
  approxEqual(bridge.getSnapshot().ry, -Math.SQRT1_2);
  env.win.dispatch('keyup', { code: 'PageDown' });
  env.win.dispatch('keyup', { code: 'Numpad6' });
  assert.equal(bridge.getSnapshot().rx, 0);
  assert.equal(bridge.getSnapshot().ry, 0);

  // Pointer Lock mouseLookToRightStick maps deltas onto unit disk and decays to 0 over rAF ticks.
  env.lockPointer();
  env.win.dispatch('mousemove', { movementX: 20, movementY: -20 });
  approxEqual(bridge.getSnapshot().rx, 0.5);
  approxEqual(bridge.getSnapshot().ry, 0.5);

  // Large diagonal spike clamps cleanly onto unit disk (hypot <= 1.0).
  env.win.dispatch('mousemove', { movementX: 200, movementY: -200 });
  approxEqual(bridge.getSnapshot().rx, Math.SQRT1_2, 1e-4);
  approxEqual(bridge.getSnapshot().ry, Math.SQRT1_2, 1e-4);

  bridge.start();
  for (let i = 0; i < 20; i++) env.flushRaf(i * 16.67);
  assert.equal(bridge.getSnapshot().rx, 0);
  assert.equal(bridge.getSnapshot().ry, 0);

  // Wheel scroll down pulses ry = -0.95 and unlock clears it immediately.
  env.win.dispatch('wheel', { deltaY: 120 });
  approxEqual(bridge.getSnapshot().ry, -0.95);
  env.unlockPointer();
  assert.equal(bridge.getSnapshot().ry, 0);

  // applyDeadzone overflow & -0 parity with applyRadialDeadzone.
  const over1 = applyDeadzone(1e308, 1e308);
  const over2 = applyRadialDeadzone(1e308, 1e308);
  approxEqual(over1.x, over2.x, 1e-5);
  approxEqual(over1.y, over2.y, 1e-5);
  assert.ok(Object.is(applyDeadzone(-0.0, 0.5).x, 0));

  bridge.destroy();
});

test('hardened chrome-to-ingame avoids Copyright Row 0 trap on warm boots (~80-110s Legal Notices) while preserving cold-boot & in-game coverage', () => {
  const player = new ScriptedInputPlayer('chrome-to-ingame');
  // Warm boot: Start Screen ~45-55s, Setup ~60-75s -> Phase 1 A hits cleanly through 74s.
  assert.equal(player.sampleAt(50000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(74000).buttonsMask, INPUT_BUTTONS.A);
  // By 80s (warm LEGAL NOTICES), cursor-resetting DUp x6 -> DDown x2 -> A cycle is active!
  assert.equal(player.sampleAt(80000).buttonsMask, INPUT_BUTTONS.DPAD_UP);
  assert.equal(player.sampleAt(83000).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(player.sampleAt(84500).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(player.sampleAt(86500).buttonsMask, INPUT_BUTTONS.A);
  // Cold boot (~150s LEGAL NOTICES) still covered by Phase 2 round 5!
  assert.equal(player.sampleAt(150000).buttonsMask, INPUT_BUTTONS.DPAD_UP);
  assert.equal(player.sampleAt(156500).buttonsMask, INPUT_BUTTONS.A);
  // Legal Terms scroll + checkboxes + Confirm > (210s..274s), Main Menu (280s..340s), In-Game (346s..418s).
  assert.ok(player.sampleAt(212700).sticks.ry < -0.9);
  assert.equal(player.sampleAt(280000).buttonsMask, INPUT_BUTTONS.A);
  assert.ok(player.sampleAt(348500).sticks.ly > 0.8);
});
