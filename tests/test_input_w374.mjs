import test from 'node:test';
import assert from 'node:assert/strict';

import {
  INPUT_ABI_VERSION,
  INPUT_STATE_BYTES,
  INPUT_STATE_WORDS,
  INPUT_OFFSETS,
  INPUT_WORD_INDICES,
  INPUT_BUTTONS,
  DEFAULT_KEY_BINDINGS,
  DEFAULT_MOUSE_DEADZONE,
  DEFAULT_MAX_MOUSE_DELTA,
  MIN_SCRIPT_HOLD_MS,
  CHROME_MENU_FLOW_4_HOLD_MS,
  CHROME_LEGAL_TERMS_ACCEPT_OFFSETS,
  FIRST_VISIBLE_FRAME_ANCHOR_PRESETS,
  INTERACTIVE_YIELD_PRESETS,
  INPUT_SCRIPT_PRESETS,
  INPUT_SCRIPT_PRESET_NAMES,
  INPUT_SCRIPT_START_TRIGGERS,
  FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS,
  applyMouseDelta,
  ScriptedInputPlayer,
  InputBridge,
} from '../web/input/input_bridge.js';

import {
  RESPONSE_CURVES,
  BUILTIN_PROFILES,
  transformStickAxes,
  InputProfileManager,
} from '../web/input/input_profiles.js';

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
}

function createMockEnv({ gamepads = [] } = {}) {
  const canvas = new MockEventTarget();
  canvas.requestPointerLock = () => Promise.resolve();
  const doc = new MockEventTarget();
  doc.pointerLockElement = null;
  doc.hidden = false;
  doc.visibilityState = 'visible';
  const win = new MockEventTarget();
  let currentPads = gamepads;
  const nav = { getGamepads: () => currentPads };
  return {
    canvas,
    doc,
    win,
    nav,
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
  };
}

function makeStandardGamepad({ buttons = {}, axes = [0, 0, 0, 0], index = 0 } = {}) {
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
    connected: true,
    mapping: 'standard',
    buttons: btnArray,
    axes: axes.slice(),
  };
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

test('W374 applyMouseDelta & Pointer Lock smooth sensitivity, sub-pixel remainder accumulation, invertLookY, deadzone, and spike clamp', () => {
  assert.equal(DEFAULT_MOUSE_DEADZONE, 0);
  assert.equal(DEFAULT_MAX_MOUSE_DELTA, 500);

  // 1. Pure zero-alloc applyMouseDelta with fractional sensitivity (0.3) accumulates remainders smoothly.
  const out = { dx: 0, dy: 0, remX: 0, remY: 0 };
  const emittedX = [];
  for (let i = 0; i < 10; i++) {
    const res = applyMouseDelta(2, 0, 0.3, 0.3, false, 0, 500, out);
    assert.equal(res, out, 'applyMouseDelta must reuse caller scratch object');
    emittedX.push(res.dx);
  }
  // 10 * (2 * 0.3) = 6.0 exact pixels emitted across 10 steps instead of 0!
  assert.equal(
    emittedX.reduce((a, b) => a + b, 0),
    6
  );
  approxEqual(out.remX, 0, 1e-9);

  // Negative sub-pixel accumulation is symmetric.
  const negOut = { dx: 0, dy: 0, remX: 0, remY: 0 };
  let negSum = 0;
  for (let i = 0; i < 10; i++) {
    applyMouseDelta(-2, -1, 0.3, 0.4, false, 0, 500, negOut);
    negSum += negOut.dx;
  }
  assert.equal(negSum, -6);
  approxEqual(negOut.remX, 0, 1e-9);

  // 2. End-to-end InputBridge Pointer Lock mousemove with low sensitivity, invertLookY, and spike clamp.
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
    mouseSensitivityX: 0.3,
    mouseSensitivityY: 0.5,
    invertLookY: true,
    maxMouseDelta: 100,
    mouseDeadzone: 0,
  });

  env.lockPointer();

  // First 2px X -> 0.6 (sub-pixel remainder only -> NO SAB write or seq bump!)
  env.win.dispatch('mousemove', { movementX: 2, movementY: 0 });
  assert.equal(bridge.getSnapshot().mouse_dx, 0);
  assert.equal(bridge.getSnapshot().seq, 0);
  assert.equal(bridge.getStats().writesCount, 0);

  // Second 2px X -> 1.2 -> emits dx = 1, remainder = 0.2!
  env.win.dispatch('mousemove', { movementX: 2, movementY: 0 });
  assert.equal(bridge.getSnapshot().mouse_dx, 1);
  assert.equal(bridge.getSnapshot().seq, 2);
  assert.equal(bridge.getStats().writesCount, 1);

  // Third & fourth 2px X -> 0.8 (no emit), 1.4 -> emits dx = 1 (total 2)!
  env.win.dispatch('mousemove', { movementX: 2, movementY: 0 });
  assert.equal(bridge.getSnapshot().mouse_dx, 1);
  assert.equal(bridge.getStats().writesCount, 1);
  env.win.dispatch('mousemove', { movementX: 2, movementY: 0 });
  assert.equal(bridge.getSnapshot().mouse_dx, 2);
  assert.equal(bridge.getStats().writesCount, 2);

  // invertLookY negates Y: movementY = +4 * 0.5 * (-1) = -2
  env.win.dispatch('mousemove', { movementX: 0, movementY: 4 });
  assert.equal(bridge.getSnapshot().mouse_dy, -2);

  // Consume deltas like C++ runtime, then test OS/Chrome pointer-lock spike clamp (2000px -> clamped to 100px raw -> 30 X / -50 Y).
  Atomics.exchange(bridge.i32, INPUT_WORD_INDICES.MOUSE_DX, 0);
  Atomics.exchange(bridge.i32, INPUT_WORD_INDICES.MOUSE_DY, 0);
  bridge._mouseRemX = 0;
  bridge._mouseRemY = 0;
  env.win.dispatch('mousemove', { movementX: 2400, movementY: -1800 });
  assert.equal(bridge.getSnapshot().mouse_dx, 30);  // clamp(2400, 100) * 0.3 = 30
  assert.equal(bridge.getSnapshot().mouse_dy, 50);  // clamp(-1800, -100) * 0.5 * -1 = +50

  // 3. Optional micro-jitter deadzone ignores tiny noise without mutating remainders or bumping seq.
  const dzEnv = createMockEnv();
  const dzBridge = new InputBridge(dzEnv.canvas, {
    windowObj: dzEnv.win,
    documentObj: dzEnv.doc,
    navigatorObj: dzEnv.nav,
    mouseDeadzone: 2,
    maxMouseDelta: 0, // disabled
  });
  dzEnv.lockPointer();

  // Magnitude hypot(1, 1) = 1.414 <= 2 -> filtered out completely
  dzEnv.win.dispatch('mousemove', { movementX: 1, movementY: 1 });
  dzEnv.win.dispatch('mousemove', { movementX: -2, movementY: 0 });
  assert.equal(dzBridge.getSnapshot().mouse_dx, 0);
  assert.equal(dzBridge.getSnapshot().mouse_dy, 0);
  assert.equal(dzBridge.getSnapshot().seq, 0);

  // Above deadzone -> passes through; with maxMouseDelta: 0 unclamped large delta passes through.
  dzEnv.win.dispatch('mousemove', { movementX: 600, movementY: -3 });
  assert.equal(dzBridge.getSnapshot().mouse_dx, 600);
  assert.equal(dzBridge.getSnapshot().mouse_dy, -3);

  bridge.destroy();
  dzBridge.destroy();
});

test('W374 Pointer Lock suppresses contextmenu & wheel scroll and cleanly clears mouse buttons, remainders, and keys on unlock or blur', () => {
  const env = createMockEnv();
  const bridge = new InputBridge(env.canvas, {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
    mouseSensitivityX: 0.3,
    mouseSensitivityY: 0.3,
  });

  // Unlocked: contextmenu is NOT suppressed.
  let ctxPrevented = false;
  env.canvas.dispatch('contextmenu', {
    preventDefault() {
      ctxPrevented = true;
    },
  });
  assert.equal(ctxPrevented, false);

  // Locked: contextmenu & wheel ARE suppressed.
  env.lockPointer();
  env.canvas.dispatch('contextmenu', {
    preventDefault() {
      ctxPrevented = true;
    },
  });
  assert.equal(ctxPrevented, true);

  let wheelPrevented = false;
  env.win.dispatch('wheel', {
    deltaY: 120,
    preventDefault() {
      wheelPrevented = true;
    },
  });
  assert.equal(wheelPrevented, true);
  assert.equal(bridge.consumeWheelDelta(), 120);

  // Hold LMB (Fire/RT), RMB (Aim/LT), MMB (RS), WASD (KeyW), and accumulate sub-pixel remainder (0.9).
  env.win.dispatch('mousedown', { button: 0 });
  env.win.dispatch('mousedown', { button: 2 });
  env.win.dispatch('mousedown', { button: 1 });
  env.win.dispatch('keydown', { code: 'KeyW' });
  env.win.dispatch('mousemove', { movementX: 3, movementY: 0 }); // 0.9 remainder

  let snap = bridge.getSnapshot();
  approxEqual(snap.rt, 1.0);
  approxEqual(snap.lt, 1.0);
  approxEqual(snap.ly, 1.0);
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.RS);
  assert.equal(snap.mouse_dx, 0);

  // Releasing Pointer Lock MUST clear held mouse buttons, held keyboard keys, AND sub-pixel remainders!
  env.unlockPointer();
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.lt, 0);
  assert.equal(snap.rt, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.ly, 0);

  // Re-lock and move 2px (0.6): if stale 0.9 remainder had survived, 0.9 + 0.6 = 1.5 would emit dx=1!
  env.lockPointer();
  env.win.dispatch('mousemove', { movementX: 2, movementY: 0 });
  assert.equal(bridge.getSnapshot().mouse_dx, 0, 'sub-pixel remainder must be cleared on Pointer Lock release');

  // Window blur also clears remainders + keys + mouse buttons.
  env.win.dispatch('keydown', { code: 'KeyD' });
  env.win.dispatch('mousedown', { button: 0 });
  env.win.dispatch('blur');
  snap = bridge.getSnapshot();
  assert.equal(snap.buttons, 0);
  assert.equal(snap.lx, 0);
  assert.equal(snap.rt, 0);
  // After blur, another 2px (0.6) must NOT combine with pre-blur 0.6 remainder.
  env.win.dispatch('mousemove', { movementX: 2, movementY: 0 });
  assert.equal(bridge.getSnapshot().mouse_dx, 0, 'sub-pixel remainder must be cleared on blur');

  bridge.destroy();
});

test('W374 InputBridge wires all profile options (stickCurve, swapSticks, swapTriggersAndBumpers, invertLookY, sensitivity, IJKL/menu left-stick) and supports live applyProfile()', () => {
  const manager = new InputProfileManager({ storage: null });

  // 1. lefty-kb: IJKL drives both MOVE_* bits AND normalized left stick (lx, ly).
  const envLefty = createMockEnv();
  const leftyBridge = new InputBridge(
    envLefty.canvas,
    manager.toInputBridgeOptions('lefty-kb', {
      windowObj: envLefty.win,
      documentObj: envLefty.doc,
      navigatorObj: envLefty.nav,
    })
  );
  envLefty.win.dispatch('keydown', { code: 'KeyI' });
  envLefty.win.dispatch('keydown', { code: 'KeyL' });
  let snap = leftyBridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.MOVE_FORWARD | INPUT_BUTTONS.MOVE_RIGHT);
  approxEqual(snap.lx, Math.SQRT1_2);
  approxEqual(snap.ly, Math.SQRT1_2);
  leftyBridge.destroy();

  // 2. menu profile: WASD maps to DPAD_* ONLY and does NOT drive left-stick (lx, ly).
  const envMenu = createMockEnv();
  const menuBridge = new InputBridge(
    envMenu.canvas,
    manager.toInputBridgeOptions('menu', {
      windowObj: envMenu.win,
      documentObj: envMenu.doc,
      navigatorObj: envMenu.nav,
    })
  );
  envMenu.win.dispatch('keydown', { code: 'KeyW' });
  snap = menuBridge.getSnapshot();
  assert.equal(snap.buttons, INPUT_BUTTONS.DPAD_UP);
  assert.equal(snap.lx, 0, 'menu preset WASD must not leak into left stick X');
  assert.equal(snap.ly, 0, 'menu preset WASD must not leak into left stick Y');

  // Switch dynamically from menu -> on_foot -> driving via applyProfile()!
  envMenu.win.dispatch('keyup', { code: 'KeyW' });
  assert.equal(menuBridge.applyProfile('on_foot'), true);
  assert.equal(menuBridge.getStats().profileId, 'on_foot');
  assert.equal(menuBridge.getStats().mouseSensitivityX, BUILTIN_PROFILES.on_foot.mouseSensitivityX);
  assert.equal(menuBridge.getStats().stickCurve, RESPONSE_CURVES.EXPO);

  // On-foot: WASD drives left stick + KeyP -> MENU, KeyF -> Y, KeyR -> X, KeyQ -> LB, KeyE -> RB, KeyC -> RS
  envMenu.win.dispatch('keydown', { code: 'KeyW' });
  assert.equal(menuBridge.getSnapshot().buttons, INPUT_BUTTONS.MOVE_FORWARD);
  approxEqual(menuBridge.getSnapshot().ly, 1.0);
  envMenu.win.dispatch('keyup', { code: 'KeyW' });

  envMenu.win.dispatch('keydown', { code: 'KeyP' });
  assert.equal(menuBridge.getSnapshot().buttons, INPUT_BUTTONS.MENU);
  envMenu.win.dispatch('keyup', { code: 'KeyP' });

  // Switch to driving -> Space becomes handbrake (RB), KeyE becomes horn (LS)
  assert.equal(menuBridge.applyProfile('driving'), true);
  assert.equal(menuBridge.getStats().profileId, 'driving');
  assert.equal(menuBridge.getStats().stickCurve, RESPONSE_CURVES.CUBIC);
  envMenu.win.dispatch('keydown', { code: 'Space' });
  assert.equal(menuBridge.getSnapshot().buttons, INPUT_BUTTONS.RB);
  envMenu.win.dispatch('keyup', { code: 'Space' });
  menuBridge.destroy();

  // 3. Gamepad profile transforms: swapSticks + invertLookY + cubic stickCurve + swapTriggersAndBumpers
  const customProfile = {
    ...BUILTIN_PROFILES.southpaw,
    id: 'southpaw-pro',
    name: 'Southpaw Pro',
    invertLookY: true,
    stickCurve: RESPONSE_CURVES.CUBIC,
    swapTriggersAndBumpers: true,
  };
  manager.createProfile(customProfile);
  const envGp = createMockEnv({
    gamepads: [
      makeStandardGamepad({
        buttons: {
          4: { pressed: true, value: 1 },    // Physical LB -> swapped to LT = 1.0
          5: { pressed: false, value: 0 },   // Physical RB -> swapped to RT = 0.0
          6: { pressed: false, value: 0 },   // Physical LT -> swapped to LB = 0
          7: { pressed: true, value: 0.9 },  // Physical RT -> swapped to RB = 1
        },
        // W3C axes (Y-down): physical L=(0.0, -0.6) -> Y-up=(0.0, +0.6); physical R=(0.7, -0.2) -> Y-up=(0.7, +0.2)
        axes: [0.0, -0.6, 0.7, -0.2],
      }),
    ],
  });
  const gpBridge = new InputBridge(
    envGp.canvas,
    manager.toInputBridgeOptions('southpaw-pro', {
      windowObj: envGp.win,
      documentObj: envGp.doc,
      navigatorObj: envGp.nav,
    })
  );
  assert.equal(gpBridge.pollGamepads(), true);
  const gpSnap = gpBridge.getSnapshot();
  const expectedSticks = transformStickAxes(0.0, 0.6, 0.7, 0.2, customProfile);
  approxEqual(gpSnap.lx, expectedSticks.lx, 1e-4);
  approxEqual(gpSnap.ly, expectedSticks.ly, 1e-4);
  approxEqual(gpSnap.rx, expectedSticks.rx, 1e-4);
  approxEqual(gpSnap.ry, expectedSticks.ry, 1e-4);
  assert.ok(gpSnap.ry < 0, 'invertLookY must negate look stick Y');
  assert.equal(gpSnap.buttons, INPUT_BUTTONS.RB, 'physical RT should map to RB when swapTriggersAndBumpers=true');
  approxEqual(gpSnap.lt, 1.0, 1e-5);
  approxEqual(gpSnap.rt, 0.0, 1e-5);

  gpBridge.destroy();
});

test('W374 chrome-menu-flow-6 preset schedule, FIRST_VISIBLE_FRAME anchor, interactive-yield flag, and zero B invariant', () => {
  assert.ok(INPUT_SCRIPT_PRESET_NAMES.includes('chrome-menu-flow-6'));
  assert.ok(FIRST_VISIBLE_FRAME_ANCHOR_PRESETS.includes('chrome-menu-flow-6'));
  assert.ok(INTERACTIVE_YIELD_PRESETS.includes('chrome-menu-flow-6'));

  const spec = INPUT_SCRIPT_PRESETS['chrome-menu-flow-6'];
  assert.ok(spec);
  assert.equal(Object.isFrozen(spec), true);
  assert.equal(spec.name, 'chrome-menu-flow-6');
  assert.equal(spec.yieldOnUserInput, true);
  assert.equal(spec.events.length, 30);
  assert.equal(spec.loop.startMs, 85000);
  assert.equal(spec.loop.periodMs, 16000);
  assert.equal(spec.loop.events, CHROME_LEGAL_TERMS_ACCEPT_OFFSETS);

  for (const ev of spec.events) {
    assert.ok(ev.holdMs >= MIN_SCRIPT_HOLD_MS);
    assert.ok(ev.holdMs >= CHROME_MENU_FLOW_4_HOLD_MS);
    assert.equal((ev.button || 0) & INPUT_BUTTONS.B, 0);
  }
  for (const ev of spec.loop.events) {
    assert.ok(ev.holdMs >= MIN_SCRIPT_HOLD_MS);
    assert.equal((ev.button || 0) & INPUT_BUTTONS.B, 0);
  }

  const player = new ScriptedInputPlayer('chrome-menu-flow-6');
  assert.equal(player.sampleAt(2000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(35000).buttonsMask, INPUT_BUTTONS.DPAD_UP);
  assert.equal(player.sampleAt(85000).buttonsMask, INPUT_BUTTONS.DPAD_UP);
  assert.ok(player.sampleAt(87700).sticks.ry < -0.9);
  assert.equal(player.sampleAt(89000).buttonsMask, INPUT_BUTTONS.A);

  // Full 160 s sweep: never presses B.
  for (let t = 0; t <= 160000; t += 100) {
    assert.equal(player.sampleAt(t).buttonsMask & INPUT_BUTTONS.B, 0);
  }
});

test('W374 interactive scripted-input yield guard pauses on keyboard, Pointer Lock mouse, and gamepad without fighting player, and supports resumeScript()', () => {
  // 1. Keyboard yield during active scripted hold on chrome-menu-flow-6 (yield defaults to true).
  const env1 = createMockEnv();
  const clock1 = createFakeClockAndTimers(0);
  const yieldEvents = [];
  const bridge1 = new InputBridge(env1.canvas, {
    windowObj: env1.win,
    documentObj: env1.doc,
    navigatorObj: env1.nav,
    onScriptYield: (reason, state) => yieldEvents.push({ reason, active: state.active }),
  });

  bridge1.startScript('chrome-menu-flow-6', {
    nowFn: clock1.nowFn,
    setTimeoutFn: clock1.setTimeoutFn,
    clearTimeoutFn: clock1.clearTimeoutFn,
  });
  assert.equal(bridge1.isScriptYielded(), false);
  assert.equal(bridge1.getStats().yieldScriptOnUserInput, true);

  // Script presses A at t = 2000 ms.
  clock1.advanceTo(2000);
  assert.equal(bridge1.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge1.getStats().scriptActive, true);

  // User presses KeyW while script is holding A -> script yields IMMEDIATELY, A is cleared, only KeyW is active!
  env1.win.dispatch('keydown', { code: 'KeyW' });
  assert.equal(bridge1.isScriptYielded(), true);
  assert.equal(bridge1.getSnapshot().buttons, INPUT_BUTTONS.MOVE_FORWARD);
  approxEqual(bridge1.getSnapshot().ly, 1.0);
  assert.equal(bridge1.getStats().scriptActive, false);
  assert.equal(bridge1.getStats().scriptYieldedByUser, true);
  assert.equal(bridge1.getStats().yieldReason, 'keyboard');
  assert.equal(bridge1.getStats().userInputCount, 1);
  assert.equal(bridge1.scriptYieldedByUser, true);
  assert.equal(bridge1.yieldReason, 'keyboard');
  assert.equal(bridge1.userInputCount, 1);
  assert.equal(bridge1.getScriptState().active, false);
  assert.equal(clock1.pendingTimeouts(), 0, 'yielding must cancel pending script timers');
  assert.deepEqual(yieldEvents, [{ reason: 'keyboard', active: false }]);

  // Zero-alloc getInputTelemetry({ reuse: true }) reflects yield state cleanly.
  const tel = bridge1.getInputTelemetry({ reuse: true });
  assert.equal(tel.scriptYieldedByUser, true);
  assert.equal(tel.yieldReason, 'keyboard');
  assert.equal(tel.userInputCount, 1);
  assert.equal(bridge1.getInputTelemetry({ reuse: true }), tel);

  // Advance clock across future script events (4s, 6s, 35s) -> scripted inputs stay paused!
  env1.win.dispatch('keyup', { code: 'KeyW' });
  clock1.advanceTo(40000);
  assert.equal(bridge1.getSnapshot().buttons, 0);
  assert.equal(bridge1.getStats().scriptActive, false);

  // Late notifyFirstVisibleFrame() MUST NOT hijack a user-yielded script!
  assert.equal(bridge1.notifyFirstVisibleFrame(), false);
  assert.equal(bridge1.isScriptYielded(), true);
  assert.equal(bridge1.getStats().scriptActive, false);

  // resumeScript() resumes from where it paused (elapsed = 2000 ms, right on the 2000 ms A event).
  assert.equal(bridge1.resumeScript(), true);
  assert.equal(bridge1.isScriptYielded(), false);
  assert.equal(bridge1.getStats().scriptActive, true);
  assert.equal(bridge1.getStats().scriptYieldedByUser, false);
  assert.equal(bridge1.getStats().yieldReason, null);
  assert.equal(bridge1.getSnapshot().buttons, INPUT_BUTTONS.A);

  // 200 ms later it releases, and +2000 ms later fires the 4000 ms A press!
  clock1.advanceBy(200);
  assert.equal(bridge1.getSnapshot().buttons, 0);
  clock1.advanceBy(1800);
  assert.equal(bridge1.getSnapshot().buttons, INPUT_BUTTONS.A);
  bridge1.destroy();

  // 2. Pointer Lock mousemove yields scripted right-stick scroll cleanly (clearing ry immediately!).
  const env2 = createMockEnv();
  const clock2 = createFakeClockAndTimers(0);
  const bridge2 = new InputBridge(env2.canvas, {
    windowObj: env2.win,
    documentObj: env2.doc,
    navigatorObj: env2.nav,
  });
  bridge2.startScript('chrome-menu-flow-6', {
    nowFn: clock2.nowFn,
    setTimeoutFn: clock2.setTimeoutFn,
    clearTimeoutFn: clock2.clearTimeoutFn,
  });
  clock2.advanceTo(87700); // Legal Terms right-stick scroll ry = -0.95
  approxEqual(bridge2.getSnapshot().ry, -0.95);

  env2.lockPointer();
  env2.win.dispatch('mousemove', { movementX: 15, movementY: -6 });
  assert.equal(bridge2.isScriptYielded(), true);
  assert.equal(bridge2.getStats().yieldReason, 'mouse');
  assert.equal(bridge2.getSnapshot().ry, 0, 'scripted right-stick scroll must clear immediately on mouse yield');
  assert.equal(bridge2.getSnapshot().mouse_dx, 15);
  assert.equal(bridge2.getSnapshot().mouse_dy, -6);
  bridge2.destroy();

  // 3. Gamepad input yields any opt-in preset (e.g. chrome-menu-flow-5 with yieldScriptOnUserInput: true).
  const env3 = createMockEnv();
  const clock3 = createFakeClockAndTimers(0);
  const bridge3 = new InputBridge(env3.canvas, {
    windowObj: env3.win,
    documentObj: env3.doc,
    navigatorObj: env3.nav,
  });
  bridge3.startScript('chrome-menu-flow-5', {
    yieldScriptOnUserInput: true,
    nowFn: clock3.nowFn,
    setTimeoutFn: clock3.setTimeoutFn,
    clearTimeoutFn: clock3.clearTimeoutFn,
  });
  clock3.advanceTo(2000);
  assert.equal(bridge3.getSnapshot().buttons, INPUT_BUTTONS.A);

  env3.setGamepads([makeStandardGamepad({ buttons: { 3: 1 } })]); // Player presses Y on gamepad
  assert.equal(bridge3.pollGamepads(), true);
  assert.equal(bridge3.isScriptYielded(), true);
  assert.equal(bridge3.getStats().yieldReason, 'gamepad');
  assert.equal(bridge3.getSnapshot().buttons, INPUT_BUTTONS.Y, 'scripted A cleared, only live gamepad Y remains');
  bridge3.destroy();

  // 4. Pending milestone-armed chrome-menu-flow-6 yielded before FIRST_VISIBLE_FRAME stays yielded,
  //    and resumeScript() starts cleanly once resumed.
  const env4 = createMockEnv();
  const clock4 = createFakeClockAndTimers(0);
  const bridge4 = new InputBridge(env4.canvas, {
    windowObj: env4.win,
    documentObj: env4.doc,
    navigatorObj: env4.nav,
  });
  bridge4.startScript('chrome-menu-flow-6', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME,
    nowFn: clock4.nowFn,
    setTimeoutFn: clock4.setTimeoutFn,
    clearTimeoutFn: clock4.clearTimeoutFn,
  });
  assert.equal(bridge4.getStats().scriptPending, true);

  // User presses Enter before FIRST_VISIBLE_FRAME -> yields pending script cleanly!
  env4.win.dispatch('keydown', { code: 'Enter' });
  env4.win.dispatch('keyup', { code: 'Enter' });
  assert.equal(bridge4.isScriptYielded(), true);
  assert.equal(bridge4.getStats().scriptPending, false);
  assert.equal(bridge4.notifyFirstVisibleFrame(), false);
  assert.equal(bridge4.getStats().scriptActive, false);

  // Calling resumeScript() after FIRST_VISIBLE_FRAME starts anchored cleanly!
  assert.equal(bridge4.resumeScript(), true);
  assert.equal(bridge4.isScriptYielded(), false);
  assert.equal(bridge4.getStats().scriptActive, true);
  clock4.advanceBy(FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS);
  assert.equal(bridge4.getSnapshot().buttons, INPUT_BUTTONS.A);
  bridge4.destroy();

  // 5. Backwards compatibility: chrome-menu-flow-5 defaults to yieldScriptOnUserInput: false,
  //    and chrome-menu-flow-6 supports explicit opt-out { yieldScriptOnUserInput: false }.
  const env5 = createMockEnv();
  const clock5 = createFakeClockAndTimers(0);
  const bridge5 = new InputBridge(env5.canvas, {
    windowObj: env5.win,
    documentObj: env5.doc,
    navigatorObj: env5.nav,
  });
  bridge5.startScript('chrome-menu-flow-6', {
    yieldScriptOnUserInput: false,
    nowFn: clock5.nowFn,
    setTimeoutFn: clock5.setTimeoutFn,
    clearTimeoutFn: clock5.clearTimeoutFn,
  });
  clock5.advanceTo(2000);
  env5.win.dispatch('keydown', { code: 'KeyW' });
  assert.equal(bridge5.isScriptYielded(), false);
  assert.equal(bridge5.getSnapshot().buttons, INPUT_BUTTONS.A | INPUT_BUTTONS.MOVE_FORWARD);
  assert.equal(bridge5.resumeScript(), false, 'resumeScript() returns false when not yielded');
  bridge5.destroy();

  // 6. Verify 40-byte ABI constants remain 100% unchanged.
  assert.equal(INPUT_ABI_VERSION, 1);
  assert.equal(INPUT_STATE_BYTES, 40);
  assert.equal(INPUT_STATE_WORDS, 10);
  assert.equal(INPUT_OFFSETS.RIGHT_TRIGGER, 36);
  assert.equal(DEFAULT_KEY_BINDINGS.KeyF, INPUT_BUTTONS.Y);
  assert.equal(DEFAULT_KEY_BINDINGS.KeyR, INPUT_BUTTONS.X);
});
