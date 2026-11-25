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
  PROFILE_ALLOWED_FIELDS,
  PROFILE_SCHEMA_VERSION,
  DEFAULT_STORAGE_KEY,
  GAMEPLAY_PROFILE_PRESETS,
  applyResponseCurve,
  applyRadialDeadzone,
  transformStickAxes,
  validateProfileOverrides,
  applyProfileOverrides,
  exportProfileJson,
  importProfileJson,
  detectBindingConflicts,
  BUILTIN_PROFILES,
  setActiveProfile,
  getActiveProfile,
  getActiveProfileId,
  resolveKeyBindingFast,
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

test('BUILTIN_PROFILES defines default, southpaw, fps, lefty-kb, menu-navigation, menu, on_foot, and driving with frozen immutability', () => {
  assert.deepEqual(
    Object.keys(BUILTIN_PROFILES).sort(),
    ['default', 'driving', 'fps', 'lefty-kb', 'menu', 'menu-navigation', 'on_foot', 'southpaw']
  );
  assert.equal(Object.isFrozen(GAMEPLAY_PROFILE_PRESETS), true);
  assert.deepEqual(GAMEPLAY_PROFILE_PRESETS, ['menu', 'on_foot', 'driving']);

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
  assert.equal(reader.listProfiles().length, Object.keys(BUILTIN_PROFILES).length + 1);

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

test('exportProfileJson and importProfileJson round-trip deterministically and enforce strict allowlisted schema', () => {
  const manager = new InputProfileManager({ storage: null });
  const cleanCustom = manager.createProfile({
    id: 'tourney-v1',
    name: 'Tournament Clean',
    keyBindings: {
      KeyD: INPUT_BUTTONS.MOVE_RIGHT,
      KeyA: INPUT_BUTTONS.MOVE_LEFT,
      KeyS: INPUT_BUTTONS.MOVE_BACKWARD,
      KeyW: INPUT_BUTTONS.MOVE_FORWARD,
      Space: INPUT_BUTTONS.JUMP,
      KeyR: INPUT_BUTTONS.X,
    },
    mouseSensitivityX: 2.0,
    mouseSensitivityY: 1.5,
    invertLookY: false,
    leftStickDeadzone: 0.1,
    rightStickDeadzone: 0.08,
    triggerDeadzone: 0.05,
    stickCurve: RESPONSE_CURVES.CUBIC,
    swapSticks: false,
    swapTriggersAndBumpers: true,
  });

  // Manager and standalone exports produce identical deterministic JSON with sorted keyBindings keys
  const exportedViaMgr = manager.exportProfileJson('tourney-v1');
  const exportedStandalone = exportProfileJson(cleanCustom);
  assert.equal(exportedViaMgr, exportedStandalone);

  const parsedEnvelope = JSON.parse(exportedViaMgr);
  assert.equal(parsedEnvelope.schemaVersion, PROFILE_SCHEMA_VERSION);
  assert.deepEqual(Object.keys(parsedEnvelope.profile).sort(), [...PROFILE_ALLOWED_FIELDS].sort());
  assert.deepEqual(Object.keys(parsedEnvelope.profile.keyBindings), [
    'KeyA',
    'KeyD',
    'KeyR',
    'KeyS',
    'KeyW',
    'Space',
  ]);

  // Standalone importProfileJson accepts envelope format and standalone bare profile JSON
  const importedStandalone = importProfileJson(exportedViaMgr);
  assert.deepEqual(importedStandalone, cleanCustom);
  const importedBare = importProfileJson(JSON.stringify(parsedEnvelope.profile));
  assert.deepEqual(importedBare, cleanCustom);

  // Manager importProfileJson inserts cleanly, supports allowOverwrite, activate, and rejectConflicts
  const receiver = new InputProfileManager({ storage: null });
  const imported = receiver.importProfileJson(exportedViaMgr, {
    allowOverwrite: false,
    activate: true,
    rejectConflicts: true,
  });
  assert.deepEqual(imported, cleanCustom);
  assert.equal(receiver.getActiveProfile().id, 'tourney-v1');

  // allowOverwrite: false rejects duplicate ID without mutating existing profile
  assert.throws(
    () => receiver.importProfileJson(exportedViaMgr, { allowOverwrite: false }),
    RangeError
  );

  // rejectConflicts: true rejects a profile with duplicate key-to-action bindings (e.g. default preset)
  const defaultExport = manager.exportProfileJson('default');
  assert.throws(
    () => receiver.importProfileJson(defaultExport, { rejectConflicts: true }),
    /binding conflicts/i
  );
  // Default (rejectConflicts: false) allows round-tripping built-in profiles
  const reimportedDefault = receiver.importProfileJson(defaultExport);
  assert.equal(reimportedDefault.id, 'default');
  assert.deepEqual(reimportedDefault.keyBindings, BUILTIN_PROFILES.default.keyBindings);

  // Unknown profileId on exportProfileJson throws RangeError
  assert.throws(() => manager.exportProfileJson('does-not-exist'), RangeError);
});

test('importProfileJson rejects prototype pollution, unknown fields, control chars, out-of-range values, oversized payloads, and invalid key codes', () => {
  const manager = new InputProfileManager({ storage: null });
  const validProfile = manager.getProfile('fps');
  const beforeSnapshot = manager.exportJson();

  const wrap = (profile) =>
    JSON.stringify({ schemaVersion: PROFILE_SCHEMA_VERSION, profile });

  // Build oversized keyBindings map (> MAX_KEY_BINDINGS)
  const tooManyBindings = {};
  for (let i = 0; i <= PROFILE_LIMITS.MAX_KEY_BINDINGS; i++) {
    tooManyBindings[`Key${String(i).padStart(2, '0')}`] = INPUT_BUTTONS.A;
  }

  const invalidCases = [
    // Non-string / malformed / wrong root
    [null, TypeError],
    [12345, TypeError],
    ['{not valid json', SyntaxError],
    ['[]', TypeError],
    ['"string"', TypeError],
    // Envelope schemaVersion & unknown fields
    [JSON.stringify({ schemaVersion: 99, profile: validProfile }), RangeError],
    [JSON.stringify({ profile: validProfile }), RangeError],
    [JSON.stringify({ schemaVersion: 1, profile: validProfile, extra: true }), RangeError],
    [wrap({ ...validProfile, unexpectedField: 'nope' }), RangeError],
    // Prototype-pollution keys at root, profile, keyBindings, and id
    ['{"schemaVersion":1,"__proto__":{"polluted":true},"profile":' + JSON.stringify(validProfile) + '}', RangeError],
    ['{"schemaVersion":1,"constructor":{"polluted":true},"profile":' + JSON.stringify(validProfile) + '}', RangeError],
    ['{"schemaVersion":1,"prototype":{"polluted":true},"profile":' + JSON.stringify(validProfile) + '}', RangeError],
    ['{"schemaVersion":1,"profile":{"__proto__":{"polluted":true},' + JSON.stringify(validProfile).slice(1) + '}', RangeError],
    ['{"schemaVersion":1,"profile":' + JSON.stringify({ ...validProfile, keyBindings: {} }).replace('{}', '{"__proto__":1}') + '}', RangeError],
    ['{"schemaVersion":1,"profile":' + JSON.stringify({ ...validProfile, keyBindings: {} }).replace('{}', '{"constructor":1}') + '}', RangeError],
    ['{"schemaVersion":1,"profile":' + JSON.stringify({ ...validProfile, keyBindings: {} }).replace('{}', '{"prototype":1}') + '}', RangeError],
    [wrap({ ...validProfile, id: '__proto__' }), RangeError],
    [wrap({ ...validProfile, id: 'constructor' }), RangeError],
    [wrap({ ...validProfile, id: 'prototype' }), RangeError],
    // Control characters in id, name, and keyBindings keys
    [wrap({ ...validProfile, id: 'bad\nid' }), RangeError],
    [wrap({ ...validProfile, id: 'bad\u0000id' }), RangeError],
    [wrap({ ...validProfile, name: 'Bad\tName' }), RangeError],
    [wrap({ ...validProfile, name: 'Bad\u001bName' }), RangeError],
    [wrap({ ...validProfile, name: 'Bad\u007fName' }), RangeError],
    [wrap({ ...validProfile, name: 'Bad\u0085Name' }), RangeError],
    [wrap({ ...validProfile, keyBindings: { 'KeyW\n': INPUT_BUTTONS.MOVE_FORWARD } }), RangeError],
    // Bounded lengths & entry counts
    [wrap({ ...validProfile, id: 'a'.repeat(PROFILE_LIMITS.MAX_ID_LENGTH + 1) }), RangeError],
    [wrap({ ...validProfile, name: 'N'.repeat(PROFILE_LIMITS.MAX_NAME_LENGTH + 1) }), RangeError],
    [wrap({ ...validProfile, keyBindings: { ['K' + 'a'.repeat(PROFILE_LIMITS.MAX_KEY_CODE_LENGTH)]: INPUT_BUTTONS.A } }), RangeError],
    [wrap({ ...validProfile, keyBindings: tooManyBindings }), RangeError],
    [' '.repeat(PROFILE_LIMITS.MAX_JSON_BYTES + 1), RangeError],
    // Invalid profile ID format
    [wrap({ ...validProfile, id: 'has space' }), RangeError],
    [wrap({ ...validProfile, id: '../escape' }), RangeError],
    // Strict finite sensitivity / deadzone ranges (no silent clamping in importProfileJson)
    [wrap({ ...validProfile, mouseSensitivityX: PROFILE_LIMITS.MIN_SENSITIVITY - 0.01 }), RangeError],
    [wrap({ ...validProfile, mouseSensitivityX: PROFILE_LIMITS.MAX_SENSITIVITY + 0.01 }), RangeError],
    [wrap({ ...validProfile, mouseSensitivityY: 0 }), RangeError],
    [wrap({ ...validProfile, leftStickDeadzone: -0.01 }), RangeError],
    [wrap({ ...validProfile, rightStickDeadzone: PROFILE_LIMITS.MAX_DEADZONE + 0.05 }), RangeError],
    [wrap({ ...validProfile, triggerDeadzone: 1.0 }), RangeError],
    [wrap({ ...validProfile, mouseSensitivityX: null }), TypeError],
    [wrap({ ...validProfile, triggerDeadzone: '0.1' }), TypeError],
    // Invalid key codes & bitmasks
    [wrap({ ...validProfile, keyBindings: { keyW: INPUT_BUTTONS.MOVE_FORWARD } }), RangeError],
    [wrap({ ...validProfile, keyBindings: { W: INPUT_BUTTONS.MOVE_FORWARD } }), RangeError],
    [wrap({ ...validProfile, keyBindings: { 'Key W': INPUT_BUTTONS.MOVE_FORWARD } }), RangeError],
    [wrap({ ...validProfile, keyBindings: { 'Key-W': INPUT_BUTTONS.MOVE_FORWARD } }), RangeError],
    [wrap({ ...validProfile, keyBindings: { '123Key': INPUT_BUTTONS.MOVE_FORWARD } }), RangeError],
    [wrap({ ...validProfile, keyBindings: { KeyW: -1 } }), RangeError],
    [wrap({ ...validProfile, keyBindings: { KeyW: 1.5 } }), RangeError],
    [wrap({ ...validProfile, keyBindings: { KeyW: 1 << 6 } }), RangeError],  // gap bit
    [wrap({ ...validProfile, keyBindings: { KeyW: 1 << 30 } }), RangeError], // unknown bit
  ];

  for (const [payload, expectedError] of invalidCases) {
    assert.throws(
      () => manager.importProfileJson(payload),
      expectedError,
      `Expected ${expectedError.name} for payload: ${String(payload).slice(0, 120)}`
    );
  }

  // Zero prototype pollution and zero state mutation after all rejected payloads
  assert.equal(({}).polluted, undefined);
  assert.equal(Object.prototype.polluted, undefined);
  assert.equal(manager.exportJson(), beforeSnapshot);

  // Existing importJson also rejects prototype-pollution keys and unknown root fields
  assert.throws(
    () => manager.importJson('{"schemaVersion":1,"activeProfileId":"default","__proto__":{"polluted":true},"profiles":[' + JSON.stringify(validProfile) + ']}'),
    RangeError
  );
  assert.throws(
    () => manager.importJson(JSON.stringify({ schemaVersion: 1, activeProfileId: 'default', extraRoot: 1, profiles: [validProfile] })),
    RangeError
  );
  assert.equal(({}).polluted, undefined);
  assert.equal(manager.exportJson(), beforeSnapshot);
});

test('detectBindingConflicts returns deterministic duplicate key-to-action conflicts and optional multi-action keys', () => {
  const manager = new InputProfileManager({ storage: null });

  // Clean 1-to-1 profile has zero conflicts
  const cleanProfile = {
    id: 'clean',
    name: 'Clean',
    keyBindings: {
      KeyW: INPUT_BUTTONS.MOVE_FORWARD,
      KeyS: INPUT_BUTTONS.MOVE_BACKWARD,
      KeyA: INPUT_BUTTONS.MOVE_LEFT,
      KeyD: INPUT_BUTTONS.MOVE_RIGHT,
      Space: INPUT_BUTTONS.JUMP,
      KeyR: INPUT_BUTTONS.X,
    },
  };
  assert.deepEqual(detectBindingConflicts(cleanProfile), []);
  assert.deepEqual(manager.detectBindingConflicts(cleanProfile), []);

  // Deterministic ordering regardless of key insertion order
  const dupOrderA = {
    keyBindings: {
      KeyT: INPUT_BUTTONS.X,
      KeyR: INPUT_BUTTONS.X,
      KeyI: INPUT_BUTTONS.MOVE_FORWARD,
      KeyW: INPUT_BUTTONS.MOVE_FORWARD,
    },
  };
  const dupOrderB = {
    keyBindings: {
      KeyW: INPUT_BUTTONS.MOVE_FORWARD,
      KeyR: INPUT_BUTTONS.X,
      KeyI: INPUT_BUTTONS.MOVE_FORWARD,
      KeyT: INPUT_BUTTONS.X,
    },
  };
  const expectedDups = [
    {
      type: 'duplicate-action',
      action: 'MOVE_FORWARD',
      mask: INPUT_BUTTONS.MOVE_FORWARD,
      keys: ['KeyI', 'KeyW'],
    },
    {
      type: 'duplicate-action',
      action: 'X',
      mask: INPUT_BUTTONS.X,
      keys: ['KeyR', 'KeyT'],
    },
  ];
  assert.deepEqual(detectBindingConflicts(dupOrderA), expectedDups);
  assert.deepEqual(detectBindingConflicts(dupOrderB), expectedDups);

  // Built-in default preset conflicts match DEFAULT_KEY_BINDINGS aliases deterministically
  assert.deepEqual(manager.detectBindingConflicts('default'), [
    {
      type: 'duplicate-action',
      action: 'A',
      mask: INPUT_BUTTONS.A,
      keys: ['Enter', 'NumpadEnter', 'Space'],
    },
    {
      type: 'duplicate-action',
      action: 'B',
      mask: INPUT_BUTTONS.B,
      keys: ['Backspace', 'ControlLeft', 'ControlRight', 'Escape'],
    },
    {
      type: 'duplicate-action',
      action: 'LS',
      mask: INPUT_BUTTONS.LS,
      keys: ['ShiftLeft', 'ShiftRight'],
    },
    {
      type: 'duplicate-action',
      action: 'SPRINT',
      mask: INPUT_BUTTONS.SPRINT,
      keys: ['ShiftLeft', 'ShiftRight'],
    },
  ]);

  // Built-in lefty-kb preset duplicate conflicts + optional includeMultiActionKeys
  assert.deepEqual(manager.detectBindingConflicts('lefty-kb'), [
    {
      type: 'duplicate-action',
      action: 'LS',
      mask: INPUT_BUTTONS.LS,
      keys: ['ShiftLeft', 'ShiftRight'],
    },
    {
      type: 'duplicate-action',
      action: 'SPRINT',
      mask: INPUT_BUTTONS.SPRINT,
      keys: ['ShiftLeft', 'ShiftRight'],
    },
  ]);

  const withMultiAction = manager.detectBindingConflicts('lefty-kb', {
    includeMultiActionKeys: true,
  });
  assert.deepEqual(
    withMultiAction.filter((c) => c.type === 'multi-action-key'),
    [
      {
        type: 'multi-action-key',
        key: 'ShiftLeft',
        mask: INPUT_BUTTONS.SPRINT | INPUT_BUTTONS.LS,
        actions: ['LS', 'SPRINT'],
      },
      {
        type: 'multi-action-key',
        key: 'ShiftRight',
        mask: INPUT_BUTTONS.SPRINT | INPUT_BUTTONS.LS,
        actions: ['LS', 'SPRINT'],
      },
      {
        type: 'multi-action-key',
        key: 'Space',
        mask: INPUT_BUTTONS.JUMP | INPUT_BUTTONS.A,
        actions: ['A', 'JUMP'],
      },
    ]
  );

  // Default argument inspects active profile
  manager.setActiveProfile('lefty-kb');
  assert.deepEqual(manager.detectBindingConflicts(), manager.detectBindingConflicts('lefty-kb'));

  // Invalid inputs throw cleanly
  assert.throws(() => manager.detectBindingConflicts('unknown-id'), RangeError);
  assert.throws(() => manager.detectBindingConflicts(42), TypeError);
  assert.throws(() => detectBindingConflicts({ keyBindings: { 'bad key': INPUT_BUTTONS.A } }), RangeError);
});

test('safe fallback reset recovers cleanly from corrupted storage, throwing storage, and custom profile resets', () => {
  // 1. resetToDefaults restores all built-ins, drops custom profiles, sets active to 'default'
  const storage = new MemoryStorage();
  const manager = new InputProfileManager({ storage });
  manager.rebindKey('default', 'KeyT', 'X');
  manager.createProfile({
    ...BUILTIN_PROFILES.fps,
    id: 'custom-broken',
    name: 'Custom Broken',
    mouseSensitivityX: 5.5,
    invertLookY: true,
  });
  manager.setActiveProfile('custom-broken');

  // Backwards compatibility: resetProfile on custom profile without fallbackToDefault still throws
  assert.throws(() => manager.resetProfile('custom-broken'), RangeError);

  // Safe fallback reset on a custom profile restores default bindings/tuning while keeping id/name
  const resetCustom = manager.resetProfile('custom-broken', { fallbackToDefault: true });
  assert.equal(resetCustom.id, 'custom-broken');
  assert.equal(resetCustom.name, 'Custom Broken');
  assert.equal(resetCustom.mouseSensitivityX, BUILTIN_PROFILES.default.mouseSensitivityX);
  assert.equal(resetCustom.invertLookY, false);
  assert.deepEqual(resetCustom.keyBindings, BUILTIN_PROFILES.default.keyBindings);

  // Safe fallback reset on a missing profile ID falls back to built-in default
  const fallbackUnknown = manager.resetProfile('nonexistent', { fallbackToDefault: true });
  assert.equal(fallbackUnknown.id, 'default');
  assert.equal(manager.getActiveProfile().id, 'default');
  assert.deepEqual(fallbackUnknown.keyBindings, BUILTIN_PROFILES.default.keyBindings);

  // Full resetToDefaults drops custom profiles and persists clean defaults
  manager.setActiveProfile('custom-broken');
  const defaults = manager.resetToDefaults({ persist: true });
  assert.equal(defaults.id, 'default');
  assert.equal(manager.getActiveProfile().id, 'default');
  assert.equal(manager.getProfile('custom-broken'), null);
  assert.equal(manager.listProfiles().length, Object.keys(BUILTIN_PROFILES).length);
  assert.ok(storage.getItem(DEFAULT_STORAGE_KEY));

  // 2. resetToDefaults({ persist: true }) never throws even if storage.setItem throws QuotaExceededError
  const failingWriteStorage = {
    getItem: () => null,
    setItem: () => {
      throw new Error('QuotaExceededError');
    },
  };
  const resilientMgr = new InputProfileManager({ storage: failingWriteStorage });
  resilientMgr.setActiveProfile('fps');
  const recovered = resilientMgr.resetToDefaults({ persist: true });
  assert.equal(recovered.id, 'default');
  assert.equal(resilientMgr.getActiveProfile().id, 'default');

  // 3. load({ fallbackToDefault: true, persistFallback: true }) recovers from corrupt / polluted localStorage
  const corruptPayloads = [
    '{corrupt json',
    '{"schemaVersion":1,"activeProfileId":"default","__proto__":{"polluted":true},"profiles":[]}',
    JSON.stringify({ schemaVersion: 999, activeProfileId: 'default', profiles: [BUILTIN_PROFILES.default] }),
  ];

  for (const corrupt of corruptPayloads) {
    const corruptStorage = new MemoryStorage();
    corruptStorage.setItem(DEFAULT_STORAGE_KEY, corrupt);

    // Backwards-compatible default load() still throws on corrupt data
    const strictLoader = new InputProfileManager({ storage: corruptStorage });
    assert.throws(() => strictLoader.load());

    // Safe fallback load() recovers cleanly to defaults and repairs persisted storage
    const safeLoader = new InputProfileManager({ storage: corruptStorage });
    safeLoader.setActiveProfile('fps');
    assert.equal(
      safeLoader.load({ fallbackToDefault: true, persistFallback: true }),
      false
    );
    assert.equal(safeLoader.getActiveProfile().id, 'default');
    assert.deepEqual(safeLoader.getActiveProfile().keyBindings, DEFAULT_KEY_BINDINGS);

    // Repaired storage now loads cleanly on a fresh manager
    const repairedReader = new InputProfileManager({ storage: corruptStorage });
    assert.equal(repairedReader.load(), true);
    assert.equal(repairedReader.getActiveProfile().id, 'default');
  }

  // 4. Constructor autoLoad + fallbackToDefault recovers from throwing storage.getItem
  const throwingReadStorage = {
    getItem: () => {
      throw new Error('SecurityError');
    },
    setItem: () => {
      throw new Error('SecurityError');
    },
  };
  const autoRecovered = new InputProfileManager({
    storage: throwingReadStorage,
    autoLoad: true,
    fallbackToDefault: true,
    persistFallback: true,
  });
  assert.equal(autoRecovered.getActiveProfile().id, 'default');
  assert.equal(autoRecovered.listProfiles().length, Object.keys(BUILTIN_PROFILES).length);
});

test('applyRadialDeadzone enforces circular threshold, rotational/reflectional symmetry, curves, and unit-disk clamping', () => {
  // Zero / inside circular deadzone -> exact +0 outputs
  const zeroOut = applyRadialDeadzone(-0, -0, 0.15, RESPONSE_CURVES.LINEAR);
  assert.ok(Object.is(zeroOut.x, 0));
  assert.ok(Object.is(zeroOut.y, 0));
  assert.deepEqual(applyRadialDeadzone(0.15, 0, 0.15, RESPONSE_CURVES.LINEAR), { x: 0, y: 0 });
  assert.deepEqual(applyRadialDeadzone(0.1, 0.1, 0.15, RESPONSE_CURVES.LINEAR), { x: 0, y: 0 });

  // True radial (circular) threshold: (0.12, 0.12) has hypot ~0.1697 > 0.15 even though each axis < 0.15
  const diagOutside = applyRadialDeadzone(0.12, 0.12, 0.15, RESPONSE_CURVES.LINEAR);
  assert.ok(diagOutside.x > 0 && diagOutside.y > 0);
  approxEqual(diagOutside.x, diagOutside.y);

  // Zero deadzone lets tiny non-zero deflections through
  const tinyOut = applyRadialDeadzone(0.01, 0, 0, RESPONSE_CURVES.LINEAR);
  assert.ok(tinyOut.x > 0);
  assert.equal(tinyOut.y, 0);

  // Odd, axis-reflection, swap, and rotational symmetry across all response curves
  const angles = [0, Math.PI / 6, Math.PI / 4, Math.PI / 3, Math.PI / 2, (3 * Math.PI) / 4, Math.PI, -Math.PI / 3];
  for (const curve of Object.values(RESPONSE_CURVES)) {
    const base = applyRadialDeadzone(0.45, 0.35, 0.15, curve);
    const negBoth = applyRadialDeadzone(-0.45, -0.35, 0.15, curve);
    const negX = applyRadialDeadzone(-0.45, 0.35, 0.15, curve);
    const negY = applyRadialDeadzone(0.45, -0.35, 0.15, curve);
    const swapped = applyRadialDeadzone(0.35, 0.45, 0.15, curve);

    approxEqual(negBoth.x, -base.x);
    approxEqual(negBoth.y, -base.y);
    approxEqual(negX.x, -base.x);
    approxEqual(negX.y, base.y);
    approxEqual(negY.x, base.x);
    approxEqual(negY.y, -base.y);
    approxEqual(swapped.x, base.y);
    approxEqual(swapped.y, base.x);

    // Rotational invariance: output magnitude depends only on input radius, not angle
    const radius = 0.6;
    const expectedMag = Math.hypot(
      applyRadialDeadzone(radius, 0, 0.15, curve).x,
      applyRadialDeadzone(radius, 0, 0.15, curve).y
    );
    assert.ok(expectedMag > 0 && expectedMag < 1);
    for (const theta of angles) {
      const sample = applyRadialDeadzone(
        radius * Math.cos(theta),
        radius * Math.sin(theta),
        0.15,
        curve
      );
      approxEqual(Math.hypot(sample.x, sample.y), expectedMag, 1e-6);
    }
  }

  // Response curve ordering at mid-deflection: cubic < expo(2) < linear
  const midLinear = applyRadialDeadzone(0.55, 0, 0.15, RESPONSE_CURVES.LINEAR).x;
  const midExpo = applyRadialDeadzone(0.55, 0, 0.15, RESPONSE_CURVES.EXPO).x;
  const midCubic = applyRadialDeadzone(0.55, 0, 0.15, RESPONSE_CURVES.CUBIC).x;
  assert.ok(midCubic > 0 && midCubic < midExpo && midExpo < midLinear && midLinear < 1);

  // Custom exponent and outerDeadzone options
  const customExpo = applyRadialDeadzone(0.6, 0, 0.2, RESPONSE_CURVES.EXPO, {
    exponent: 2.5,
    outerDeadzone: 1.0,
  });
  approxEqual(customExpo.x, Math.pow((0.6 - 0.2) / (1.0 - 0.2), 2.5));
  assert.equal(customExpo.y, 0);

  // Unit-disk clamping for diagonal corners, oversized inputs, and finite overflow
  for (const curve of Object.values(RESPONSE_CURVES)) {
    const corner = applyRadialDeadzone(1.0, 1.0, 0.15, curve);
    approxEqual(corner.x, Math.SQRT1_2);
    approxEqual(corner.y, Math.SQRT1_2);
    approxEqual(Math.hypot(corner.x, corner.y), 1.0);

    const huge = applyRadialDeadzone(5.0, -5.0, 0.15, curve);
    approxEqual(huge.x, Math.SQRT1_2);
    approxEqual(huge.y, -Math.SQRT1_2);
    assert.ok(Math.hypot(huge.x, huge.y) <= 1 + 1e-12);

    const overflow = applyRadialDeadzone(1e308, 1e308, 0.15, curve);
    approxEqual(overflow.x, Math.SQRT1_2);
    approxEqual(overflow.y, Math.SQRT1_2);
  }

  // Non-finite inputs return zero; out-of-range deadzone clamps safely; unknown curve throws
  assert.deepEqual(applyRadialDeadzone(Number.NaN, 0.5, 0.15, RESPONSE_CURVES.LINEAR), { x: 0, y: 0 });
  assert.deepEqual(applyRadialDeadzone(0.5, Number.POSITIVE_INFINITY, 0.15, RESPONSE_CURVES.LINEAR), { x: 0, y: 0 });
  assert.deepEqual(applyRadialDeadzone(0.5, 0, 1.5, RESPONSE_CURVES.LINEAR), { x: 0, y: 0 }); // clamps dz to 0.9
  assert.ok(applyRadialDeadzone(0.95, 0, 1.5, RESPONSE_CURVES.LINEAR).x > 0);
  assert.ok(applyRadialDeadzone(0.05, 0, -0.5, RESPONSE_CURVES.LINEAR).x > 0); // clamps dz to 0
  assert.throws(() => applyRadialDeadzone(0.5, 0.5, 0.15, 'quartic'), RangeError);
});

test('menu-navigation built-in profile provides complete Xbox Series menu bindings and integrates with InputBridge', () => {
  const menuProfile = BUILTIN_PROFILES['menu-navigation'];
  assert.ok(menuProfile, 'menu-navigation preset must exist');
  assert.equal(menuProfile.id, 'menu-navigation');
  assert.equal(Object.isFrozen(menuProfile), true);
  assert.equal(Object.isFrozen(menuProfile.keyBindings), true);

  // Confirm (A) & Cancel/Back (B)
  assert.equal(menuProfile.keyBindings.Enter, INPUT_BUTTONS.A);
  assert.equal(menuProfile.keyBindings.NumpadEnter, INPUT_BUTTONS.A);
  assert.equal(menuProfile.keyBindings.Space, INPUT_BUTTONS.A);
  assert.equal(menuProfile.keyBindings.Escape, INPUT_BUTTONS.B);
  assert.equal(menuProfile.keyBindings.Backspace, INPUT_BUTTONS.B);

  // D-pad navigation via arrow keys
  assert.equal(menuProfile.keyBindings.ArrowUp, INPUT_BUTTONS.DPAD_UP);
  assert.equal(menuProfile.keyBindings.ArrowDown, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(menuProfile.keyBindings.ArrowLeft, INPUT_BUTTONS.DPAD_LEFT);
  assert.equal(menuProfile.keyBindings.ArrowRight, INPUT_BUTTONS.DPAD_RIGHT);

  // Y/X shortcuts plus tab/system navigation (LB, RB, VIEW, MENU)
  assert.equal(menuProfile.keyBindings.KeyX, INPUT_BUTTONS.X);
  assert.equal(menuProfile.keyBindings.KeyY, INPUT_BUTTONS.Y);
  assert.equal(menuProfile.keyBindings.KeyQ, INPUT_BUTTONS.LB);
  assert.equal(menuProfile.keyBindings.KeyE, INPUT_BUTTONS.RB);
  assert.equal(menuProfile.keyBindings.Tab, INPUT_BUTTONS.VIEW);
  assert.equal(menuProfile.keyBindings.KeyM, INPUT_BUTTONS.MENU);

  // Drift-resistant calibration deadzones & cubic response curve
  assert.ok(menuProfile.leftStickDeadzone >= 0.2);
  assert.ok(menuProfile.rightStickDeadzone >= 0.2);
  assert.ok(menuProfile.triggerDeadzone >= 0.1);
  assert.equal(menuProfile.stickCurve, RESPONSE_CURVES.CUBIC);

  // Manager lifecycle & strict JSON round-trip
  const manager = new InputProfileManager({ storage: null });
  assert.throws(() => manager.deleteProfile('menu-navigation'), RangeError);
  const exported = manager.exportProfileJson('menu-navigation');
  const reimported = importProfileJson(exported);
  assert.deepEqual(reimported, manager.getProfile('menu-navigation'));

  // Direct InputBridge integration: every menu shortcut maps cleanly and gameplay-only WASD stays off
  const env = createMockBridgeEnv();
  const bridge = new InputBridge(
    env.canvas,
    manager.toInputBridgeOptions('menu-navigation', {
      windowObj: env.win,
      documentObj: env.doc,
      navigatorObj: env.nav,
    })
  );

  const menuChecks = [
    ['Enter', INPUT_BUTTONS.A],
    ['Space', INPUT_BUTTONS.A],
    ['Escape', INPUT_BUTTONS.B],
    ['Backspace', INPUT_BUTTONS.B],
    ['ArrowUp', INPUT_BUTTONS.DPAD_UP],
    ['ArrowDown', INPUT_BUTTONS.DPAD_DOWN],
    ['ArrowLeft', INPUT_BUTTONS.DPAD_LEFT],
    ['ArrowRight', INPUT_BUTTONS.DPAD_RIGHT],
    ['KeyX', INPUT_BUTTONS.X],
    ['KeyY', INPUT_BUTTONS.Y],
    ['KeyQ', INPUT_BUTTONS.LB],
    ['KeyE', INPUT_BUTTONS.RB],
    ['Tab', INPUT_BUTTONS.VIEW],
    ['KeyM', INPUT_BUTTONS.MENU],
    ['KeyW', 0],
  ];

  for (const [code, expectedMask] of menuChecks) {
    env.win.dispatch('keydown', { code });
    assert.equal(
      bridge.getSnapshot().buttons,
      expectedMask,
      `Expected ${code} to produce button mask 0x${expectedMask.toString(16)}`
    );
    env.win.dispatch('keyup', { code });
    assert.equal(bridge.getSnapshot().buttons, 0);
  }

  bridge.destroy();
});

test('custom profile override validation accepts safe overrides and strictly rejects invalid overrides', () => {
  // Valid partial overrides via validateProfileOverrides & applyProfileOverrides
  const validPatch = validateProfileOverrides({
    leftStickDeadzone: 0.22,
    rightStickDeadzone: 0.18,
    stickCurve: RESPONSE_CURVES.EXPO,
    invertLookY: true,
    keyBindings: {
      KeyP: INPUT_BUTTONS.MENU,
      Space: 0,
    },
  });
  assert.equal(validPatch.leftStickDeadzone, 0.22);
  assert.equal(validPatch.rightStickDeadzone, 0.18);
  assert.equal(validPatch.stickCurve, RESPONSE_CURVES.EXPO);
  assert.equal(validPatch.invertLookY, true);
  assert.deepEqual(validPatch.keyBindings, { KeyP: INPUT_BUTTONS.MENU });

  const mergedMenu = applyProfileOverrides('menu-navigation', {
    id: 'custom-menu',
    name: 'Custom Menu',
    leftStickDeadzone: 0.25,
    keyBindings: {
      Space: 0,
      KeyP: INPUT_BUTTONS.MENU,
    },
  });
  assert.equal(mergedMenu.id, 'custom-menu');
  assert.equal(mergedMenu.leftStickDeadzone, 0.25);
  assert.equal(mergedMenu.keyBindings.Space, undefined, 'mask 0 unbinds key when merging');
  assert.equal(mergedMenu.keyBindings.Enter, INPUT_BUTTONS.A, 'unrelated base bindings preserved');
  assert.equal(mergedMenu.keyBindings.KeyP, INPUT_BUTTONS.MENU);
  assert.equal(BUILTIN_PROFILES['menu-navigation'].keyBindings.Space, INPUT_BUTTONS.A);

  const replacedMenu = applyProfileOverrides(
    BUILTIN_PROFILES['menu-navigation'],
    { keyBindings: { Enter: INPUT_BUTTONS.A } },
    { mergeKeyBindings: false }
  );
  assert.deepEqual(replacedMenu.keyBindings, { Enter: INPUT_BUTTONS.A });

  // Build prototype-pollution objects safely via Object.defineProperty / JSON.parse
  const protoPolluteOverride = JSON.parse('{"__proto__":{"polluted":true},"leftStickDeadzone":0.2}');
  const protoPolluteBindings = { keyBindings: JSON.parse('{"__proto__":1}') };
  const tooManyBindings = {};
  for (let i = 0; i <= PROFILE_LIMITS.MAX_KEY_BINDINGS; i++) {
    tooManyBindings[`Key${String(i).padStart(2, '0')}`] = INPUT_BUTTONS.A;
  }

  const invalidOverrides = [
    [null, TypeError],
    [undefined, TypeError],
    [[], TypeError],
    ['invalid', TypeError],
    [new Date(), TypeError],
    // Prototype pollution & unknown fields
    [protoPolluteOverride, RangeError],
    [{ constructor: { polluted: true } }, RangeError],
    [{ prototype: { polluted: true } }, RangeError],
    [protoPolluteBindings, RangeError],
    [{ unknownField: 123 }, RangeError],
    // Out-of-range numeric settings
    [{ leftStickDeadzone: -0.01 }, RangeError],
    [{ leftStickDeadzone: PROFILE_LIMITS.MAX_DEADZONE + 0.05 }, RangeError],
    [{ rightStickDeadzone: 1.0 }, RangeError],
    [{ triggerDeadzone: -0.1 }, RangeError],
    [{ triggerDeadzone: 0.95 }, RangeError],
    [{ mouseSensitivityX: PROFILE_LIMITS.MIN_SENSITIVITY - 0.05 }, RangeError],
    [{ mouseSensitivityY: PROFILE_LIMITS.MAX_SENSITIVITY + 1 }, RangeError],
    // Wrong numeric/boolean/curve types
    [{ leftStickDeadzone: Number.NaN }, TypeError],
    [{ rightStickDeadzone: Number.POSITIVE_INFINITY }, TypeError],
    [{ mouseSensitivityX: '1.5' }, TypeError],
    [{ invertLookY: 'true' }, TypeError],
    [{ swapSticks: 1 }, TypeError],
    [{ swapTriggersAndBumpers: null }, TypeError],
    [{ stickCurve: 'quartic' }, RangeError],
    [{ stickCurve: 42 }, RangeError],
    // Invalid id / name / keyBindings
    [{ id: '../escape' }, RangeError],
    [{ id: 'bad\nid' }, RangeError],
    [{ name: '   ' }, TypeError],
    [{ name: 'Bad\u0000Name' }, RangeError],
    [{ keyBindings: null }, TypeError],
    [{ keyBindings: [] }, TypeError],
    [{ keyBindings: { badKey: INPUT_BUTTONS.A } }, RangeError],
    [{ keyBindings: { KeyW: -1 } }, RangeError],
    [{ keyBindings: { KeyW: 1.5 } }, RangeError],
    [{ keyBindings: { KeyW: 1 << 6 } }, RangeError],
    [{ keyBindings: { KeyW: 1 << 30 } }, RangeError],
    [{ keyBindings: tooManyBindings }, RangeError],
  ];

  const manager = new InputProfileManager({ storage: null });
  const snapshotBefore = manager.exportJson();

  for (const [badOverride, expectedErr] of invalidOverrides) {
    assert.throws(() => validateProfileOverrides(badOverride), expectedErr);
    assert.throws(() => applyProfileOverrides('menu-navigation', badOverride), expectedErr);
    assert.throws(() => manager.updateProfile('menu-navigation', badOverride), expectedErr);
  }

  // Attempting to change profile id via updateProfile is rejected
  assert.throws(() => manager.updateProfile('menu-navigation', { id: 'renamed-id' }), RangeError);
  // Unknown base profile ID on applyProfileOverrides throws
  assert.throws(() => applyProfileOverrides('does-not-exist', {}), RangeError);
  // createProfile also strictly rejects unknown fields and out-of-range deadzones
  assert.throws(
    () => manager.createProfile({ ...BUILTIN_PROFILES['menu-navigation'], id: 'bad-custom', unknownField: true }),
    RangeError
  );
  assert.throws(
    () => manager.createProfile({ ...BUILTIN_PROFILES['menu-navigation'], id: 'bad-dz', leftStickDeadzone: 1.2 }),
    RangeError
  );
  // toInputBridgeOptions rejects non-object or prototype-polluting overrides
  assert.throws(() => manager.toInputBridgeOptions('menu-navigation', null), TypeError);
  assert.throws(
    () => manager.toInputBridgeOptions('menu-navigation', JSON.parse('{"__proto__":{"polluted":true}}')),
    RangeError
  );

  // Zero prototype pollution and zero manager state mutation after all rejected calls
  assert.equal(({}).polluted, undefined);
  assert.equal(Object.prototype.polluted, undefined);
  assert.equal(manager.exportJson(), snapshotBefore);
});

test('W329 menu, on_foot, and driving presets layer cleanly over INPUT_BUTTONS and DEFAULT_KEY_BINDINGS and integrate with InputBridge', () => {
  const manager = new InputProfileManager({ storage: null });

  // 1. menu preset: Arrow + WASD -> DPAD (no MOVE_* bits), confirm/back, drift-resistant deadzones, cubic curve
  const menu = BUILTIN_PROFILES.menu;
  assert.equal(menu.id, 'menu');
  assert.equal(Object.isFrozen(menu), true);
  assert.equal(Object.isFrozen(menu.keyBindings), true);
  assert.equal(menu.keyBindings.Enter, INPUT_BUTTONS.A);
  assert.equal(menu.keyBindings.NumpadEnter, INPUT_BUTTONS.A);
  assert.equal(menu.keyBindings.Space, INPUT_BUTTONS.A);
  assert.equal(menu.keyBindings.Escape, INPUT_BUTTONS.B);
  assert.equal(menu.keyBindings.Backspace, INPUT_BUTTONS.B);
  assert.equal(menu.keyBindings.ArrowUp, INPUT_BUTTONS.DPAD_UP);
  assert.equal(menu.keyBindings.ArrowDown, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(menu.keyBindings.ArrowLeft, INPUT_BUTTONS.DPAD_LEFT);
  assert.equal(menu.keyBindings.ArrowRight, INPUT_BUTTONS.DPAD_RIGHT);
  assert.equal(menu.keyBindings.KeyW, INPUT_BUTTONS.DPAD_UP);
  assert.equal(menu.keyBindings.KeyS, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(menu.keyBindings.KeyA, INPUT_BUTTONS.DPAD_LEFT);
  assert.equal(menu.keyBindings.KeyD, INPUT_BUTTONS.DPAD_RIGHT);
  assert.equal(menu.keyBindings.KeyQ, INPUT_BUTTONS.LB);
  assert.equal(menu.keyBindings.KeyE, INPUT_BUTTONS.RB);
  assert.equal(menu.keyBindings.Tab, INPUT_BUTTONS.VIEW);
  assert.equal(menu.keyBindings.KeyM, INPUT_BUTTONS.MENU);
  assert.ok(menu.leftStickDeadzone >= 0.2);
  assert.ok(menu.rightStickDeadzone >= 0.2);
  assert.ok(menu.triggerDeadzone >= 0.1);
  assert.equal(menu.stickCurve, RESPONSE_CURVES.CUBIC);
  // Ensure no gameplay MOVE_* bits exist in menu preset
  const moveBits =
    INPUT_BUTTONS.MOVE_FORWARD |
    INPUT_BUTTONS.MOVE_BACKWARD |
    INPUT_BUTTONS.MOVE_LEFT |
    INPUT_BUTTONS.MOVE_RIGHT;
  for (const mask of Object.values(menu.keyBindings)) {
    assert.equal(mask & moveBits, 0, 'menu preset must not emit gameplay movement bits');
  }

  // 2. on_foot preset: preserves 100% of DEFAULT_KEY_BINDINGS + KeyP (MENU) + KeyV (VIEW)
  const onFoot = BUILTIN_PROFILES.on_foot;
  assert.equal(onFoot.id, 'on_foot');
  assert.equal(Object.isFrozen(onFoot), true);
  assert.equal(Object.isFrozen(onFoot.keyBindings), true);
  for (const [keyCode, expectedMask] of Object.entries(DEFAULT_KEY_BINDINGS)) {
    assert.equal(
      onFoot.keyBindings[keyCode],
      expectedMask,
      `on_foot must preserve DEFAULT_KEY_BINDINGS.${keyCode}`
    );
  }
  assert.equal(onFoot.keyBindings.KeyP, INPUT_BUTTONS.MENU);
  assert.equal(onFoot.keyBindings.KeyV, INPUT_BUTTONS.VIEW);
  assert.equal(onFoot.stickCurve, RESPONSE_CURVES.EXPO);
  assert.ok(onFoot.mouseSensitivityX > 1.0 && onFoot.mouseSensitivityY > 1.0);
  assert.ok(onFoot.rightStickDeadzone < BUILTIN_PROFILES.default.rightStickDeadzone);

  // 3. driving preset: vehicle-tuned bindings & cubic steering curve
  const driving = BUILTIN_PROFILES.driving;
  assert.equal(driving.id, 'driving');
  assert.equal(Object.isFrozen(driving), true);
  assert.equal(Object.isFrozen(driving.keyBindings), true);
  assert.equal(driving.keyBindings.KeyW, INPUT_BUTTONS.MOVE_FORWARD);
  assert.equal(driving.keyBindings.KeyS, INPUT_BUTTONS.MOVE_BACKWARD);
  assert.equal(driving.keyBindings.KeyA, INPUT_BUTTONS.MOVE_LEFT);
  assert.equal(driving.keyBindings.KeyD, INPUT_BUTTONS.MOVE_RIGHT);
  assert.equal(driving.keyBindings.Space, INPUT_BUTTONS.RB);       // Handbrake
  assert.equal(driving.keyBindings.KeyE, INPUT_BUTTONS.LS);        // Horn
  assert.equal(driving.keyBindings.KeyF, INPUT_BUTTONS.Y);         // Enter/Exit vehicle
  assert.equal(driving.keyBindings.KeyQ, INPUT_BUTTONS.LB);        // Drive-by weapon
  assert.equal(driving.keyBindings.KeyC, INPUT_BUTTONS.RS);        // Look behind
  assert.equal(driving.keyBindings.KeyH, INPUT_BUTTONS.DPAD_RIGHT); // Headlights
  assert.equal(driving.keyBindings.KeyX, INPUT_BUTTONS.A);         // Duck
  assert.equal(driving.keyBindings.KeyR, INPUT_BUTTONS.B);         // Cinematic camera
  assert.equal(driving.keyBindings.KeyV, INPUT_BUTTONS.VIEW);      // Camera cycle
  assert.equal(driving.keyBindings.KeyP, INPUT_BUTTONS.MENU);      // Pause
  assert.equal(driving.keyBindings.Escape, INPUT_BUTTONS.MENU);    // Pause
  assert.equal(driving.stickCurve, RESPONSE_CURVES.CUBIC);
  assert.ok(driving.leftStickDeadzone < BUILTIN_PROFILES.default.leftStickDeadzone);
  assert.ok(driving.triggerDeadzone < BUILTIN_PROFILES.default.triggerDeadzone);

  // Strict JSON round-trip & deletion protection for all 3 gameplay presets
  for (const presetId of GAMEPLAY_PROFILE_PRESETS) {
    assert.throws(() => manager.deleteProfile(presetId), RangeError);
    const exported = manager.exportProfileJson(presetId);
    const reimported = importProfileJson(exported);
    assert.deepEqual(reimported, manager.getProfile(presetId));
  }

  // Real InputBridge integration for menu, on_foot, and driving
  const bridgeCases = [
    [
      'menu',
      [
        ['KeyW', INPUT_BUTTONS.DPAD_UP],
        ['ArrowDown', INPUT_BUTTONS.DPAD_DOWN],
        ['Enter', INPUT_BUTTONS.A],
        ['Escape', INPUT_BUTTONS.B],
        ['KeyR', 0], // gameplay-only key zeroed out in menu
      ],
    ],
    [
      'on_foot',
      [
        ['KeyW', INPUT_BUTTONS.MOVE_FORWARD],
        ['Space', INPUT_BUTTONS.JUMP | INPUT_BUTTONS.A],
        ['KeyR', INPUT_BUTTONS.X],
        ['KeyP', INPUT_BUTTONS.MENU],
        ['KeyV', INPUT_BUTTONS.VIEW],
      ],
    ],
    [
      'driving',
      [
        ['KeyW', INPUT_BUTTONS.MOVE_FORWARD],
        ['Space', INPUT_BUTTONS.RB],
        ['KeyE', INPUT_BUTTONS.LS],
        ['KeyH', INPUT_BUTTONS.DPAD_RIGHT],
        ['KeyX', INPUT_BUTTONS.A],
        ['KeyR', INPUT_BUTTONS.B],
        ['Escape', INPUT_BUTTONS.MENU],
      ],
    ],
  ];

  for (const [presetId, checks] of bridgeCases) {
    const env = createMockBridgeEnv();
    const bridge = new InputBridge(
      env.canvas,
      manager.toInputBridgeOptions(presetId, {
        windowObj: env.win,
        documentObj: env.doc,
        navigatorObj: env.nav,
      })
    );
    for (const [code, expectedMask] of checks) {
      env.win.dispatch('keydown', { code });
      assert.equal(
        bridge.getSnapshot().buttons,
        expectedMask,
        `${presetId}: keydown ${code} expected 0x${expectedMask.toString(16)}`
      );
      env.win.dispatch('keyup', { code });
      assert.equal(bridge.getSnapshot().buttons, 0, `${presetId}: keyup ${code} must clear`);
    }
    bridge.destroy();
  }
});

test('W329 setActiveProfile and zero-alloc O(1) resolveKeyBindingFast work at module level and on InputProfileManager', () => {
  try {
    // Module-level defaults to 'default'
    setActiveProfile('default');
    assert.equal(getActiveProfileId(), 'default');
    assert.equal(getActiveProfile(), BUILTIN_PROFILES.default);
    assert.equal(resolveKeyBindingFast('KeyW'), INPUT_BUTTONS.MOVE_FORWARD);
    assert.equal(resolveKeyBindingFast('Space'), INPUT_BUTTONS.JUMP | INPUT_BUTTONS.A);
    assert.equal(resolveKeyBindingFast('KeyP'), 0);

    // Switch through menu -> on_foot -> driving and verify immediate O(1) resolution
    assert.equal(setActiveProfile('menu'), BUILTIN_PROFILES.menu);
    assert.equal(getActiveProfileId(), 'menu');
    assert.equal(resolveKeyBindingFast('KeyW'), INPUT_BUTTONS.DPAD_UP);
    assert.equal(resolveKeyBindingFast('Space'), INPUT_BUTTONS.A);
    assert.equal(resolveKeyBindingFast('KeyE'), INPUT_BUTTONS.RB);
    assert.equal(resolveKeyBindingFast('ShiftLeft'), 0);

    assert.equal(setActiveProfile('on_foot'), BUILTIN_PROFILES.on_foot);
    assert.equal(getActiveProfileId(), 'on_foot');
    assert.equal(resolveKeyBindingFast('KeyW'), INPUT_BUTTONS.MOVE_FORWARD);
    assert.equal(resolveKeyBindingFast('Space'), INPUT_BUTTONS.JUMP | INPUT_BUTTONS.A);
    assert.equal(resolveKeyBindingFast('KeyE'), INPUT_BUTTONS.RB);
    assert.equal(resolveKeyBindingFast('KeyP'), INPUT_BUTTONS.MENU);
    assert.equal(resolveKeyBindingFast('KeyV'), INPUT_BUTTONS.VIEW);

    assert.equal(setActiveProfile('driving'), BUILTIN_PROFILES.driving);
    assert.equal(getActiveProfileId(), 'driving');
    assert.equal(resolveKeyBindingFast('KeyW'), INPUT_BUTTONS.MOVE_FORWARD);
    assert.equal(resolveKeyBindingFast('Space'), INPUT_BUTTONS.RB);
    assert.equal(resolveKeyBindingFast('KeyE'), INPUT_BUTTONS.LS);
    assert.equal(resolveKeyBindingFast('KeyH'), INPUT_BUTTONS.DPAD_RIGHT);
    assert.equal(resolveKeyBindingFast('KeyX'), INPUT_BUTTONS.A);
    assert.equal(resolveKeyBindingFast('KeyR'), INPUT_BUTTONS.B);

    // Custom profile object supported & frozen; external mutation cannot corrupt lookup table
    const customDriving = applyProfileOverrides('driving', {
      id: 'driving-custom',
      name: 'Driving Custom',
      keyBindings: { KeyG: INPUT_BUTTONS.GUIDE },
    });
    const activeCustom = setActiveProfile(customDriving);
    assert.equal(Object.isFrozen(activeCustom), true);
    assert.equal(getActiveProfileId(), 'driving-custom');
    assert.equal(resolveKeyBindingFast('KeyG'), INPUT_BUTTONS.GUIDE);
    customDriving.keyBindings.KeyG = 0;
    assert.equal(resolveKeyBindingFast('KeyG'), INPUT_BUTTONS.GUIDE);

    // Invalid profile names / non-objects throw without mutating active state
    assert.throws(() => setActiveProfile('does-not-exist'), RangeError);
    assert.throws(() => setActiveProfile('__proto__'), RangeError);
    assert.throws(() => setActiveProfile(null), TypeError);
    assert.throws(() => setActiveProfile(123), TypeError);
    assert.throws(() => setActiveProfile([]), TypeError);
    assert.equal(getActiveProfileId(), 'driving-custom');
    assert.equal(resolveKeyBindingFast('KeyG'), INPUT_BUTTONS.GUIDE);

    // Prototype keys, unbound keys, and non-string inputs safely return 0
    for (const unsafeKey of [
      '__proto__',
      'constructor',
      'prototype',
      'toString',
      'valueOf',
      'hasOwnProperty',
      '',
      'KeyZ',
      null,
      undefined,
      0,
      42,
      {},
      [],
    ]) {
      assert.equal(resolveKeyBindingFast(unsafeKey), 0);
    }

    // Verify 100% parity with every key of every built-in preset & hot-loop stability
    const sampleKeys = [
      'KeyW',
      'KeyA',
      'KeyS',
      'KeyD',
      'Space',
      'KeyE',
      'KeyR',
      'KeyF',
      'KeyH',
      'KeyX',
      'KeyP',
      'KeyV',
      'Escape',
      'ArrowUp',
      'KeyZ',
    ];
    for (const [id, preset] of Object.entries(BUILTIN_PROFILES)) {
      setActiveProfile(id);
      for (const code of Object.keys(preset.keyBindings)) {
        assert.equal(resolveKeyBindingFast(code), preset.keyBindings[code]);
      }
      let expectedSum = 0;
      for (const code of sampleKeys) {
        expectedSum += preset.keyBindings[code] || 0;
      }
      let hotSum = 0;
      const iterations = 5000;
      for (let i = 0; i < iterations; i++) {
        for (let k = 0; k < sampleKeys.length; k++) {
          hotSum += resolveKeyBindingFast(sampleKeys[k]);
        }
      }
      assert.equal(hotSum, expectedSum * iterations);
    }
  } finally {
    setActiveProfile('default');
  }

  // InputProfileManager instance-level resolveKeyBindingFast tracks every lifecycle mutation
  const manager = new InputProfileManager({ storage: null });
  assert.equal(manager.getActiveProfileId(), 'default');
  assert.equal(manager.resolveKeyBindingFast('Space'), INPUT_BUTTONS.JUMP | INPUT_BUTTONS.A);
  assert.equal(manager.resolveKeyBindingFast('__proto__'), 0);
  assert.equal(manager.resolveKeyBindingFast(null), 0);

  // Switch active profile -> lookup updates immediately
  manager.setActiveProfile('driving');
  assert.equal(manager.getActiveProfileId(), 'driving');
  assert.equal(manager.resolveKeyBindingFast('Space'), INPUT_BUTTONS.RB);
  assert.equal(manager.resolveKeyBindingFast('KeyE'), INPUT_BUTTONS.LS);

  // rebindKey on active profile updates fast lookup immediately
  manager.rebindKey('driving', 'KeyU', 'GUIDE', { allowOverwrite: false });
  assert.equal(manager.resolveKeyBindingFast('KeyU'), INPUT_BUTTONS.GUIDE);

  // Failed conflict rebind leaves fast lookup untouched
  assert.throws(
    () => manager.rebindKey('driving', 'KeyT', 'GUIDE', { allowOverwrite: false }),
    /already bound to "KeyU"/
  );
  assert.equal(manager.resolveKeyBindingFast('KeyU'), INPUT_BUTTONS.GUIDE);
  assert.equal(manager.resolveKeyBindingFast('KeyT'), 0);

  // Overwrite rebind moves binding in fast lookup cleanly
  const moved = manager.rebindKey('driving', 'KeyT', 'GUIDE', { allowOverwrite: true });
  assert.deepEqual(moved, { changed: true, conflictKey: 'KeyU' });
  assert.equal(manager.resolveKeyBindingFast('KeyU'), 0);
  assert.equal(manager.resolveKeyBindingFast('KeyT'), INPUT_BUTTONS.GUIDE);

  // unbindKey removes entry from fast lookup
  assert.equal(manager.unbindKey('driving', 'KeyT'), true);
  assert.equal(manager.resolveKeyBindingFast('KeyT'), 0);

  // updateProfile on active profile updates fast lookup
  manager.updateProfile('driving', { keyBindings: { KeyW: INPUT_BUTTONS.MOVE_FORWARD, KeyG: INPUT_BUTTONS.RB } });
  assert.equal(manager.resolveKeyBindingFast('KeyG'), INPUT_BUTTONS.RB);
  assert.equal(manager.resolveKeyBindingFast('Space'), 0);

  // resetProfile restores built-in driving bindings in fast lookup
  manager.resetProfile('driving');
  assert.equal(manager.resolveKeyBindingFast('Space'), INPUT_BUTTONS.RB);
  assert.equal(manager.resolveKeyBindingFast('KeyG'), 0);

  // Custom active profile delete falls back to default and updates fast lookup
  manager.createProfile({
    ...BUILTIN_PROFILES.menu,
    id: 'temp-menu',
    name: 'Temp Menu',
  });
  manager.setActiveProfile('temp-menu');
  assert.equal(manager.resolveKeyBindingFast('KeyW'), INPUT_BUTTONS.DPAD_UP);
  manager.deleteProfile('temp-menu');
  assert.equal(manager.getActiveProfileId(), 'default');
  assert.equal(manager.resolveKeyBindingFast('KeyW'), INPUT_BUTTONS.MOVE_FORWARD);

  // importProfileJson with activate: true and overwriting active profile sync fast lookup
  const onFootJson = manager.exportProfileJson('on_foot');
  manager.importProfileJson(onFootJson, { allowOverwrite: true, activate: true });
  assert.equal(manager.getActiveProfileId(), 'on_foot');
  assert.equal(manager.resolveKeyBindingFast('KeyP'), INPUT_BUTTONS.MENU);

  // resetToDefaults restores default fast lookup
  manager.resetToDefaults();
  assert.equal(manager.getActiveProfileId(), 'default');
  assert.equal(manager.resolveKeyBindingFast('KeyP'), 0);
});

test('W329 key conflict safety for menu, on_foot, and driving presets', () => {
  const manager = new InputProfileManager({ storage: null });

  // menu preset has zero multi-action keys (every key maps to a single unambiguous action)
  const menuAll = manager.detectBindingConflicts('menu', { includeMultiActionKeys: true });
  assert.deepEqual(
    menuAll.filter((c) => c.type === 'multi-action-key'),
    []
  );

  // driving preset separates handbrake (Space -> RB), horn (KeyE -> LS), and pause (Escape -> MENU) cleanly
  assert.equal(BUILTIN_PROFILES.driving.keyBindings.Space, INPUT_BUTTONS.RB);
  assert.equal(BUILTIN_PROFILES.driving.keyBindings.KeyE, INPUT_BUTTONS.LS);
  assert.equal(BUILTIN_PROFILES.driving.keyBindings.Escape, INPUT_BUTTONS.MENU);
  const drivingDups = manager.detectBindingConflicts('driving');
  // RB (handbrake) must have zero duplicate key conflicts in driving
  assert.equal(
    drivingDups.some((c) => c.action === 'RB'),
    false,
    'driving handbrake (RB) should be uniquely bound to Space'
  );

  // Clean 1-to-1 derived override from on_foot passes importProfileJson({ rejectConflicts: true })
  const conflictFreeOnFoot = applyProfileOverrides(
    'on_foot',
    {
      id: 'on_foot_strict',
      name: 'On Foot Strict',
      keyBindings: {
        KeyW: INPUT_BUTTONS.MOVE_FORWARD,
        KeyS: INPUT_BUTTONS.MOVE_BACKWARD,
        KeyA: INPUT_BUTTONS.MOVE_LEFT,
        KeyD: INPUT_BUTTONS.MOVE_RIGHT,
        Space: INPUT_BUTTONS.JUMP,
        ShiftLeft: INPUT_BUTTONS.SPRINT,
        KeyR: INPUT_BUTTONS.X,
        KeyF: INPUT_BUTTONS.Y,
        KeyP: INPUT_BUTTONS.MENU,
      },
    },
    { mergeKeyBindings: false }
  );
  assert.deepEqual(detectBindingConflicts(conflictFreeOnFoot, { includeMultiActionKeys: true }), []);
  const roundTripped = importProfileJson(exportProfileJson(conflictFreeOnFoot), {
    rejectConflicts: true,
  });
  assert.deepEqual(roundTripped, conflictFreeOnFoot);
});

test('W271 applyRadialDeadzone regression: zero-alloc out-object reuse and InputBridge applyDeadzone parity', () => {
  const out = { x: 99, y: 99 };

  // Inside deadzone reuses `out` in-place and sets +0
  const inside = applyRadialDeadzone(0.08, -0.09, 0.15, 0.98, out);
  assert.equal(inside, out);
  assert.ok(Object.is(out.x, 0));
  assert.ok(Object.is(out.y, 0));

  // Non-finite inputs reuse `out` in-place and zero safely
  const nanOut = applyRadialDeadzone(Number.NaN, 0.5, 0.15, 0.98, out);
  assert.equal(nanOut, out);
  assert.equal(out.x, 0);
  assert.equal(out.y, 0);

  // Exact numerical parity with InputBridge applyDeadzone across cardinal, diagonal, and clamped inputs
  const samples = [
    [0, 0],
    [0.14, 0],
    [0.5, 0],
    [0, -0.75],
    [0.5, 0.5],
    [-0.6, 0.3],
    [0.98, 0],
    [1.0, 1.0],
    [2.5, -1.2],
  ];
  for (const [sx, sy] of samples) {
    const expected = applyDeadzone(sx, sy, 0.15, 0.98);
    const actual = applyRadialDeadzone(sx, sy, 0.15, 0.98, out);
    assert.equal(actual, out, 'must reuse pre-allocated out object');
    approxEqual(actual.x, expected.x);
    approxEqual(actual.y, expected.y);
    assert.ok(Math.hypot(actual.x, actual.y) <= 1 + 1e-12);
  }

  // Curve signature with `{ out }` option also reuses `out` in-place
  const curved = applyRadialDeadzone(0.565, 0, 0.15, RESPONSE_CURVES.EXPO, {
    out,
    outerDeadzone: 0.98,
    exponent: 2.0,
  });
  assert.equal(curved, out);
  approxEqual(out.x, 0.25, 1e-5);
  assert.equal(out.y, 0);
});
