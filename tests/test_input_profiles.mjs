import test from 'node:test';
import assert from 'node:assert/strict';

import {
  INPUT_BUTTONS,
  DEFAULT_KEY_BINDINGS,
  applyDeadzone,
  InputBridge,
} from '../web/input/input_bridge.js';

import {
  RESPONSE_CURVES,
  PROFILE_LIMITS,
  PROFILE_SCHEMA_VERSION,
  DEFAULT_STORAGE_KEY,
  applyResponseCurve,
  transformStickAxes,
  BUILTIN_PROFILES,
  InputProfileManager,
} from '../web/input/input_profiles.js';

function approxEqual(actual, expected, eps = 1e-6) {
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `Expected ${actual} to be within ${eps} of ${expected}`
  );
}

class MemoryStorage {
  constructor() {
    this._map = new Map();
  }

  getItem(key) {
    return this._map.has(key) ? this._map.get(key) : null;
  }

  setItem(key, value) {
    this._map.set(String(key), String(value));
  }

  removeItem(key) {
    this._map.delete(String(key));
  }

  clear() {
    this._map.clear();
  }
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

function createMockBridgeEnv(gamepads = []) {
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
  };
}

test('applyResponseCurve handles linear, cubic, expo, negative symmetry, clamping, and invalid inputs', () => {
  assert.deepEqual(Object.values(RESPONSE_CURVES).sort(), ['cubic', 'expo', 'linear']);

  // Linear
  assert.equal(applyResponseCurve(0, RESPONSE_CURVES.LINEAR), 0);
  assert.equal(applyResponseCurve(-0, RESPONSE_CURVES.LINEAR), 0);
  assert.ok(Object.is(applyResponseCurve(-0, RESPONSE_CURVES.LINEAR), 0));
  approxEqual(applyResponseCurve(0.5, RESPONSE_CURVES.LINEAR), 0.5);
  approxEqual(applyResponseCurve(-0.5, RESPONSE_CURVES.LINEAR), -0.5);
  assert.equal(applyResponseCurve(1, RESPONSE_CURVES.LINEAR), 1);
  assert.equal(applyResponseCurve(-1, RESPONSE_CURVES.LINEAR), -1);

  // Cubic
  approxEqual(applyResponseCurve(0.5, RESPONSE_CURVES.CUBIC), 0.125);
  approxEqual(applyResponseCurve(-0.5, RESPONSE_CURVES.CUBIC), -0.125);
  approxEqual(applyResponseCurve(0.8, RESPONSE_CURVES.CUBIC), 0.512);
  approxEqual(applyResponseCurve(-0.8, RESPONSE_CURVES.CUBIC), -0.512);
  assert.equal(applyResponseCurve(1, RESPONSE_CURVES.CUBIC), 1);
  assert.equal(applyResponseCurve(-1, RESPONSE_CURVES.CUBIC), -1);

  // Expo (default exponent = 2.0 and custom exponents)
  approxEqual(applyResponseCurve(0.5, RESPONSE_CURVES.EXPO), 0.25);
  approxEqual(applyResponseCurve(-0.5, RESPONSE_CURVES.EXPO), -0.25);
  approxEqual(applyResponseCurve(0.5, RESPONSE_CURVES.EXPO, 2.5), Math.pow(0.5, 2.5));
  approxEqual(applyResponseCurve(-0.5, RESPONSE_CURVES.EXPO, 2.5), -Math.pow(0.5, 2.5));
  approxEqual(applyResponseCurve(0.5, RESPONSE_CURVES.EXPO, 1.0), 0.5);
  approxEqual(applyResponseCurve(0.5, RESPONSE_CURVES.EXPO, 3.0), 0.125);

  // Odd symmetry across all curves
  for (const curve of Object.values(RESPONSE_CURVES)) {
    for (const v of [0.1, 0.25, 0.5, 0.75, 0.95, 1.0]) {
      approxEqual(applyResponseCurve(-v, curve), -applyResponseCurve(v, curve));
    }
  }

  // Clamping and non-finite inputs
  for (const curve of Object.values(RESPONSE_CURVES)) {
    assert.equal(applyResponseCurve(2.5, curve), 1);
    assert.equal(applyResponseCurve(-3.7, curve), -1);
    assert.equal(applyResponseCurve(Number.NaN, curve), 0);
    assert.equal(applyResponseCurve(Number.POSITIVE_INFINITY, curve), 0);
    assert.equal(applyResponseCurve(Number.NEGATIVE_INFINITY, curve), 0);
  }

  assert.throws(() => applyResponseCurve(0.5, 'invalid-curve'), RangeError);
});

test('BUILTIN_PROFILES defines default, southpaw, fps, and lefty-kb with frozen immutability', () => {
  assert.deepEqual(
    Object.keys(BUILTIN_PROFILES).sort(),
    ['default', 'fps', 'lefty-kb', 'southpaw']
  );

  // Default matches InputBridge DEFAULT_KEY_BINDINGS and deadzones 100%
  const def = BUILTIN_PROFILES.default;
  assert.equal(def.id, 'default');
  assert.deepEqual(def.keyBindings, DEFAULT_KEY_BINDINGS);
  assert.equal(def.mouseSensitivityX, 1.0);
  assert.equal(def.mouseSensitivityY, 1.0);
  assert.equal(def.invertLookY, false);
  assert.equal(def.leftStickDeadzone, 0.15);
  assert.equal(def.rightStickDeadzone, 0.15);
  assert.equal(def.triggerDeadzone, 0.05);
  assert.equal(def.stickCurve, RESPONSE_CURVES.LINEAR);
  assert.equal(def.swapSticks, false);
  assert.equal(def.swapTriggersAndBumpers, false);

  // Southpaw swaps sticks
  const southpaw = BUILTIN_PROFILES.southpaw;
  assert.equal(southpaw.id, 'southpaw');
  assert.equal(southpaw.swapSticks, true);

  // FPS has elevated sensitivity, tighter deadzones, and expo curve
  const fps = BUILTIN_PROFILES.fps;
  assert.equal(fps.id, 'fps');
  assert.ok(fps.mouseSensitivityX > def.mouseSensitivityX);
  assert.ok(fps.mouseSensitivityY > def.mouseSensitivityY);
  assert.ok(fps.rightStickDeadzone < def.rightStickDeadzone);
  assert.equal(fps.stickCurve, RESPONSE_CURVES.EXPO);

  // Lefty-kb uses IJKL movement
  const lefty = BUILTIN_PROFILES['lefty-kb'];
  assert.equal(lefty.id, 'lefty-kb');
  assert.equal(lefty.keyBindings.KeyI, INPUT_BUTTONS.MOVE_FORWARD);
  assert.equal(lefty.keyBindings.KeyK, INPUT_BUTTONS.MOVE_BACKWARD);
  assert.equal(lefty.keyBindings.KeyJ, INPUT_BUTTONS.MOVE_LEFT);
  assert.equal(lefty.keyBindings.KeyL, INPUT_BUTTONS.MOVE_RIGHT);
  assert.equal(lefty.keyBindings.KeyW, undefined);

  // Built-in presets are deeply frozen
  assert.equal(Object.isFrozen(BUILTIN_PROFILES), true);
  for (const preset of Object.values(BUILTIN_PROFILES)) {
    assert.equal(Object.isFrozen(preset), true);
    assert.equal(Object.isFrozen(preset.keyBindings), true);
    assert.throws(() => {
      preset.mouseSensitivityX = 9.9;
    }, TypeError);
    assert.throws(() => {
      preset.keyBindings.KeyZ = INPUT_BUTTONS.GUIDE;
    }, TypeError);
  }
});

test('rebindKey validates button names, detects conflicts, handles allowOverwrite, and preserves built-in immutability', () => {
  const manager = new InputProfileManager({ storage: null });

  // Invalid button name or profile ID throws
  assert.throws(() => manager.rebindKey('default', 'KeyT', 'NOT_A_BUTTON'), RangeError);
  assert.throws(() => manager.rebindKey('missing-profile', 'KeyT', 'X'), RangeError);
  assert.throws(() => manager.rebindKey('default', '', 'X'), TypeError);

  // Binding an unbound key to an unbound button succeeds with no conflict
  const cleanRes = manager.rebindKey('default', 'KeyG', 'GUIDE', { allowOverwrite: false });
  assert.deepEqual(cleanRes, { changed: true, conflictKey: null });
  assert.equal(manager.getProfile('default').keyBindings.KeyG, INPUT_BUTTONS.GUIDE);

  // Rebinding same key to same button is a no-op
  const noopRes = manager.rebindKey('default', 'KeyG', 'GUIDE', { allowOverwrite: false });
  assert.deepEqual(noopRes, { changed: false, conflictKey: null });

  // Conflict: button 'X' is already bound to 'KeyR'; allowOverwrite: false throws and leaves state unchanged
  assert.throws(
    () => manager.rebindKey('default', 'KeyT', 'X', { allowOverwrite: false }),
    /already bound to "KeyR"/
  );
  assert.equal(manager.getProfile('default').keyBindings.KeyR, INPUT_BUTTONS.X);
  assert.equal(manager.getProfile('default').keyBindings.KeyT, undefined);

  // Conflict: key 'KeyF' is already bound to 'Y'; allowOverwrite: false throws
  assert.throws(
    () => manager.rebindKey('default', 'KeyF', 'GUIDE', { allowOverwrite: false }),
    /Key "KeyF" is already bound/
  );
  assert.equal(manager.getProfile('default').keyBindings.KeyF, INPUT_BUTTONS.Y);

  // allowOverwrite: true moves 'X' from 'KeyR' to 'KeyT' and reports conflictKey: 'KeyR'
  const moveRes = manager.rebindKey('default', 'KeyT', 'X', { allowOverwrite: true });
  assert.deepEqual(moveRes, { changed: true, conflictKey: 'KeyR' });
  const updated = manager.getProfile('default');
  assert.equal(updated.keyBindings.KeyT, INPUT_BUTTONS.X);
  assert.equal(updated.keyBindings.KeyR, undefined);

  // Compound binding split: ShiftLeft/ShiftRight have SPRINT | LS; rebinding SPRINT preserves LS
  const sprintRes = manager.rebindKey('default', 'KeyV', 'SPRINT', { allowOverwrite: true });
  assert.equal(sprintRes.changed, true);
  assert.equal(sprintRes.conflictKey, 'ShiftLeft');
  const afterSprint = manager.getProfile('default');
  assert.equal(afterSprint.keyBindings.KeyV, INPUT_BUTTONS.SPRINT);
  assert.equal(afterSprint.keyBindings.ShiftLeft, INPUT_BUTTONS.LS);
  assert.equal(afterSprint.keyBindings.ShiftRight, INPUT_BUTTONS.LS);

  // Built-in preset MUST remain completely untouched
  assert.equal(BUILTIN_PROFILES.default.keyBindings.KeyR, INPUT_BUTTONS.X);
  assert.equal(BUILTIN_PROFILES.default.keyBindings.KeyT, undefined);
  assert.equal(
    BUILTIN_PROFILES.default.keyBindings.ShiftLeft,
    INPUT_BUTTONS.SPRINT | INPUT_BUTTONS.LS
  );

  // Mutating returned profile objects does not mutate manager internal state
  const leaked = manager.getActiveProfile();
  leaked.keyBindings.KeyT = 0;
  assert.equal(manager.getActiveProfile().keyBindings.KeyT, INPUT_BUTTONS.X);

  // resetProfile restores working copy back to built-in default
  const reset = manager.resetProfile('default');
  assert.deepEqual(reset.keyBindings, DEFAULT_KEY_BINDINGS);
  assert.deepEqual(manager.getProfile('default').keyBindings, DEFAULT_KEY_BINDINGS);
});

test('transformStickAxes applies swapSticks, deadzone, response curve, invertLookY, and clamping', () => {
  // Inside deadzone -> zeroed
  assert.deepEqual(transformStickAxes(0.05, -0.05, 0.1, 0.05, BUILTIN_PROFILES.default), {
    lx: 0,
    ly: 0,
    rx: 0,
    ry: 0,
  });

  // Default linear matches applyDeadzone directly
  const defOut = transformStickAxes(0.6, 0.0, 0.0, 0.5, BUILTIN_PROFILES.default);
  const expL = applyDeadzone(0.6, 0.0, 0.15);
  const expR = applyDeadzone(0.0, 0.5, 0.15);
  approxEqual(defOut.lx, expL.x);
  approxEqual(defOut.ly, expL.y);
  approxEqual(defOut.rx, expR.x);
  approxEqual(defOut.ry, expR.y);

  // Southpaw swaps physical left and right sticks
  const spOut = transformStickAxes(0.6, 0.0, 0.0, 0.5, BUILTIN_PROFILES.southpaw);
  approxEqual(spOut.lx, expR.x);
  approxEqual(spOut.ly, expR.y);
  approxEqual(spOut.rx, expL.x);
  approxEqual(spOut.ry, expL.y);

  // invertLookY negates ry only, leaving ly untouched
  const invertedProfile = {
    ...BUILTIN_PROFILES.default,
    invertLookY: true,
    stickCurve: RESPONSE_CURVES.CUBIC,
  };
  const invOut = transformStickAxes(0.0, 0.565, 0.0, 0.565, invertedProfile);
  const normVal = applyDeadzone(0.0, 0.565, 0.15).y; // ~0.5
  const curvedVal = applyResponseCurve(normVal, RESPONSE_CURVES.CUBIC); // ~0.125
  approxEqual(invOut.ly, curvedVal);
  approxEqual(invOut.ry, -curvedVal);

  // Combined swapSticks + invertLookY: physical left stick becomes look stick and gets inverted
  const southpawInverted = {
    ...BUILTIN_PROFILES.southpaw,
    invertLookY: true,
  };
  const spInvOut = transformStickAxes(0.0, 0.8, 0.0, 0.6, southpawInverted);
  assert.ok(spInvOut.ly > 0, 'movement stick Y should remain positive');
  assert.ok(spInvOut.ry < 0, 'swapped look stick Y should be inverted');

  // Extreme / non-finite inputs clamp safely to [-1, 1]
  const clamped = transformStickAxes(5.0, -5.0, Number.NaN, -10.0, BUILTIN_PROFILES.fps);
  assert.ok(clamped.lx >= -1 && clamped.lx <= 1);
  assert.ok(clamped.ly >= -1 && clamped.ly <= 1);
  assert.equal(clamped.rx, 0);
  assert.equal(clamped.ry, 0);

  // Manager method delegates to active profile or named profile ID
  const manager = new InputProfileManager({ storage: null });
  manager.setActiveProfile('southpaw');
  const mgrOut = manager.transformStickAxes(0.6, 0.0, 0.0, 0.5);
  assert.deepEqual(mgrOut, spOut);
});

test('toInputBridgeOptions integrates directly with real InputBridge for remapped keys and deadzones', () => {
  const manager = new InputProfileManager({ storage: null });
  manager.rebindKey('default', 'KeyT', 'X', { allowOverwrite: true });
  manager.updateProfile('default', {
    leftStickDeadzone: 0.25,
    triggerDeadzone: 0.2,
  });

  const env = createMockBridgeEnv([
    {
      id: 'Mock Xbox Controller',
      index: 0,
      connected: true,
      mapping: 'standard',
      buttons: Array.from({ length: 17 }, (_, idx) => {
        if (idx === 6) return { pressed: false, touched: true, value: 0.15 }; // below 0.2 trigger deadzone
        if (idx === 7) return { pressed: true, touched: true, value: 0.6 };    // above 0.2 trigger deadzone
        return { pressed: false, touched: false, value: 0 };
      }),
      axes: [0.2, 0.0, 0.0, 0.0], // 0.2 is below 0.25 stick deadzone
    },
  ]);

  const bridgeOptions = manager.toInputBridgeOptions('default', {
    windowObj: env.win,
    documentObj: env.doc,
    navigatorObj: env.nav,
  });

  const bridge = new InputBridge(env.canvas, bridgeOptions);

  // Old KeyR was rebound away -> must NOT trigger X in InputBridge
  env.win.dispatch('keydown', { code: 'KeyR' });
  assert.equal(bridge.getSnapshot().buttons, 0);
  env.win.dispatch('keyup', { code: 'KeyR' });

  // New KeyT MUST trigger X in InputBridge
  env.win.dispatch('keydown', { code: 'KeyT' });
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.X);
  env.win.dispatch('keyup', { code: 'KeyT' });
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Deadzones from profile take effect in real InputBridge.pollGamepads()
  assert.equal(bridge.pollGamepads(), true);
  const snap = bridge.getSnapshot();
  assert.equal(snap.lx, 0, 'stick deflection 0.2 should be inside 0.25 deadzone');
  assert.equal(snap.lt, 0, 'trigger deflection 0.15 should be inside 0.20 deadzone');
  assert.ok(snap.rt > 0, 'trigger deflection 0.60 should exceed 0.20 deadzone');

  bridge.destroy();

  // Verify lefty-kb preset also works directly with InputBridge
  const envLefty = createMockBridgeEnv();
  const leftyBridge = new InputBridge(
    envLefty.canvas,
    manager.toInputBridgeOptions('lefty-kb', {
      windowObj: envLefty.win,
      documentObj: envLefty.doc,
      navigatorObj: envLefty.nav,
    })
  );
  envLefty.win.dispatch('keydown', { code: 'KeyI' });
  assert.equal(leftyBridge.getSnapshot().buttons, INPUT_BUTTONS.MOVE_FORWARD);
  envLefty.win.dispatch('keyup', { code: 'KeyI' });
  envLefty.win.dispatch('keydown', { code: 'KeyH' });
  assert.equal(leftyBridge.getSnapshot().buttons, INPUT_BUTTONS.X);
  leftyBridge.destroy();
});

test('save, load, exportJson, and importJson persist state and strictly validate + clamp schemas', () => {
  const storage = new MemoryStorage();
  const writer = new InputProfileManager({ storage });

  writer.rebindKey('default', 'KeyT', 'X');
  writer.createProfile({
    ...BUILTIN_PROFILES.fps,
    id: 'custom-pro',
    name: 'Custom Pro',
    invertLookY: true,
    swapTriggersAndBumpers: true,
    mouseSensitivityX: 2.25,
    mouseSensitivityY: 1.75,
  });
  writer.setActiveProfile('custom-pro');

  assert.equal(writer.save(), true);
  assert.ok(storage.getItem(DEFAULT_STORAGE_KEY));

  const reader = new InputProfileManager({ storage });
  assert.equal(reader.getActiveProfile().id, 'default');
  assert.equal(reader.load(), true);

  assert.equal(reader.getActiveProfile().id, 'custom-pro');
  assert.equal(reader.getActiveProfile().invertLookY, true);
  assert.equal(reader.getActiveProfile().swapTriggersAndBumpers, true);
  assert.equal(reader.getActiveProfile().mouseSensitivityX, 2.25);
  assert.equal(reader.getProfile('default').keyBindings.KeyT, INPUT_BUTTONS.X);
  assert.equal(reader.listProfiles().length, 5);

  // Round-trip exportJson / importJson with out-of-range sensitivity/deadzone clamping
  const exported = JSON.parse(writer.exportJson());
  assert.equal(exported.schemaVersion, PROFILE_SCHEMA_VERSION);
  const customEntry = exported.profiles.find((p) => p.id === 'custom-pro');
  customEntry.mouseSensitivityX = 999;
  customEntry.mouseSensitivityY = -5;
  customEntry.leftStickDeadzone = 1.5;
  customEntry.rightStickDeadzone = -0.3;
  customEntry.triggerDeadzone = 2.0;

  const importer = new InputProfileManager({ storage: null });
  importer.importJson(JSON.stringify(exported));
  const clampedCustom = importer.getProfile('custom-pro');
  assert.equal(clampedCustom.mouseSensitivityX, PROFILE_LIMITS.MAX_SENSITIVITY);
  assert.equal(clampedCustom.mouseSensitivityY, PROFILE_LIMITS.MIN_SENSITIVITY);
  assert.equal(clampedCustom.leftStickDeadzone, PROFILE_LIMITS.MAX_DEADZONE);
  assert.equal(clampedCustom.rightStickDeadzone, PROFILE_LIMITS.MIN_DEADZONE);
  assert.equal(clampedCustom.triggerDeadzone, PROFILE_LIMITS.MAX_DEADZONE);

  // Strict schema validation rejects malformed JSON and keeps existing state untouched
  const beforeState = importer.exportJson();
  const invalidPayloads = [
    'not valid json {',
    '[]',
    JSON.stringify({ schemaVersion: 99, activeProfileId: 'default', profiles: [BUILTIN_PROFILES.default] }),
    JSON.stringify({ schemaVersion: 1, activeProfileId: 'default', profiles: [] }),
    JSON.stringify({ schemaVersion: 1, activeProfileId: 'nonexistent', profiles: [BUILTIN_PROFILES.default] }),
    JSON.stringify({
      schemaVersion: 1,
      activeProfileId: 'default',
      profiles: [{ ...BUILTIN_PROFILES.default, stickCurve: 'quartic' }],
    }),
    JSON.stringify({
      schemaVersion: 1,
      activeProfileId: 'default',
      profiles: [{ ...BUILTIN_PROFILES.default, mouseSensitivityX: 'fast' }],
    }),
    JSON.stringify({
      schemaVersion: 1,
      activeProfileId: 'default',
      profiles: [{ ...BUILTIN_PROFILES.default, invertLookY: 'yes' }],
    }),
    JSON.stringify({
      schemaVersion: 1,
      activeProfileId: 'default',
      profiles: [{ ...BUILTIN_PROFILES.default, keyBindings: { KeyW: 1 << 30 } }],
    }),
    JSON.stringify({
      schemaVersion: 1,
      activeProfileId: 'default',
      profiles: [BUILTIN_PROFILES.default, BUILTIN_PROFILES.default],
    }),
  ];

  for (const payload of invalidPayloads) {
    assert.throws(() => importer.importJson(payload));
  }
  assert.equal(importer.exportJson(), beforeState, 'failed imports must not mutate state');

  // Empty/null storage returns false gracefully
  const emptyMgr = new InputProfileManager({ storage: new MemoryStorage() });
  assert.equal(emptyMgr.load(), false);
  const noStorageMgr = new InputProfileManager({ storage: null });
  assert.equal(noStorageMgr.save(), false);
  assert.equal(noStorageMgr.load(), false);
});
