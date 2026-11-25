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
  applyResponseCurve,
  transformStickAxes,
  exportProfileJson,
  importProfileJson,
  detectBindingConflicts,
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
