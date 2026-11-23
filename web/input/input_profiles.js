// Input profile management, stick response curves, key remapping, and JSON/storage
// persistence designed to integrate directly with InputBridge (web/input/input_bridge.js).

import {
  INPUT_BUTTONS,
  DEFAULT_KEY_BINDINGS,
  applyDeadzone,
} from './input_bridge.js';

export const RESPONSE_CURVES = Object.freeze({
  LINEAR: 'linear',
  CUBIC: 'cubic',
  EXPO: 'expo',
});

export const PROFILE_SCHEMA_VERSION = 1;
export const DEFAULT_STORAGE_KEY = 'playgta6.inputProfiles.v1';

export const PROFILE_LIMITS = Object.freeze({
  MIN_SENSITIVITY: 0.1,
  MAX_SENSITIVITY: 10.0,
  MIN_DEADZONE: 0.0,
  MAX_DEADZONE: 0.9,
  DEFAULT_EXPO_EXPONENT: 2.0,
  MIN_EXPO_EXPONENT: 0.1,
  MAX_EXPO_EXPONENT: 10.0,
});

const VALID_CURVES = new Set(Object.values(RESPONSE_CURVES));

const ALL_BUTTON_BITS = Object.values(INPUT_BUTTONS).reduce(
  (acc, bit) => (acc | bit) >>> 0,
  0
);

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return 0;
  return value < min ? min : value > max ? max : value;
}

function normalizeZero(value) {
  return value === 0 ? 0 : value;
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function deepFreezeProfile(profile) {
  Object.freeze(profile.keyBindings);
  return Object.freeze(profile);
}

function cloneProfile(profile) {
  return {
    id: profile.id,
    name: profile.name,
    keyBindings: { ...profile.keyBindings },
    mouseSensitivityX: profile.mouseSensitivityX,
    mouseSensitivityY: profile.mouseSensitivityY,
    invertLookY: profile.invertLookY,
    leftStickDeadzone: profile.leftStickDeadzone,
    rightStickDeadzone: profile.rightStickDeadzone,
    triggerDeadzone: profile.triggerDeadzone,
    stickCurve: profile.stickCurve,
    swapSticks: profile.swapSticks,
    swapTriggersAndBumpers: profile.swapTriggersAndBumpers,
  };
}

function resolveDefaultStorage() {
  try {
    const ls = globalThis?.localStorage;
    if (ls && typeof ls.getItem === 'function' && typeof ls.setItem === 'function') {
      return ls;
    }
  } catch {
    // Accessing localStorage can throw SecurityError in restricted browser contexts.
  }
  return null;
}

function validateAndNormalizeKeyBindings(rawBindings, contextLabel = 'keyBindings') {
  if (!isPlainObject(rawBindings)) {
    throw new TypeError(`${contextLabel} must be a plain object`);
  }

  const normalized = {};
  for (const [keyCode, rawMask] of Object.entries(rawBindings)) {
    if (typeof keyCode !== 'string' || keyCode.trim().length === 0) {
      throw new TypeError(`${contextLabel} keys must be non-empty strings`);
    }
    if (!Number.isInteger(rawMask) || rawMask < 0 || rawMask > 0xffffffff) {
      throw new RangeError(
        `${contextLabel}["${keyCode}"] must be a valid INPUT_BUTTONS integer bitmask`
      );
    }
    const mask = rawMask >>> 0;
    if ((mask & ~ALL_BUTTON_BITS) !== 0) {
      throw new RangeError(
        `${contextLabel}["${keyCode}"] contains unknown button bits (0x${mask.toString(16)})`
      );
    }
    if (mask !== 0) {
      normalized[keyCode] = mask;
    }
  }

  return normalized;
}

function validateAndNormalizeProfile(rawProfile, contextLabel = 'profile') {
  if (!isPlainObject(rawProfile)) {
    throw new TypeError(`${contextLabel} must be a plain object`);
  }

  if (typeof rawProfile.id !== 'string' || rawProfile.id.trim().length === 0) {
    throw new TypeError(`${contextLabel}.id must be a non-empty string`);
  }
  if (typeof rawProfile.name !== 'string' || rawProfile.name.trim().length === 0) {
    throw new TypeError(`${contextLabel}.name must be a non-empty string`);
  }

  const numericFields = [
    ['mouseSensitivityX', PROFILE_LIMITS.MIN_SENSITIVITY, PROFILE_LIMITS.MAX_SENSITIVITY],
    ['mouseSensitivityY', PROFILE_LIMITS.MIN_SENSITIVITY, PROFILE_LIMITS.MAX_SENSITIVITY],
    ['leftStickDeadzone', PROFILE_LIMITS.MIN_DEADZONE, PROFILE_LIMITS.MAX_DEADZONE],
    ['rightStickDeadzone', PROFILE_LIMITS.MIN_DEADZONE, PROFILE_LIMITS.MAX_DEADZONE],
    ['triggerDeadzone', PROFILE_LIMITS.MIN_DEADZONE, PROFILE_LIMITS.MAX_DEADZONE],
  ];

  const clampedNumbers = {};
  for (const [field, min, max] of numericFields) {
    const val = rawProfile[field];
    if (typeof val !== 'number' || !Number.isFinite(val)) {
      throw new TypeError(`${contextLabel}.${field} must be a finite number`);
    }
    clampedNumbers[field] = clamp(val, min, max);
  }

  const booleanFields = ['invertLookY', 'swapSticks', 'swapTriggersAndBumpers'];
  for (const field of booleanFields) {
    if (typeof rawProfile[field] !== 'boolean') {
      throw new TypeError(`${contextLabel}.${field} must be a boolean`);
    }
  }

  if (typeof rawProfile.stickCurve !== 'string' || !VALID_CURVES.has(rawProfile.stickCurve)) {
    throw new RangeError(
      `${contextLabel}.stickCurve must be one of: ${[...VALID_CURVES].join(', ')}`
    );
  }

  const keyBindings = validateAndNormalizeKeyBindings(
    rawProfile.keyBindings,
    `${contextLabel}.keyBindings`
  );

  return {
    id: rawProfile.id.trim(),
    name: rawProfile.name.trim(),
    keyBindings,
    mouseSensitivityX: clampedNumbers.mouseSensitivityX,
    mouseSensitivityY: clampedNumbers.mouseSensitivityY,
    invertLookY: rawProfile.invertLookY,
    leftStickDeadzone: clampedNumbers.leftStickDeadzone,
    rightStickDeadzone: clampedNumbers.rightStickDeadzone,
    triggerDeadzone: clampedNumbers.triggerDeadzone,
    stickCurve: rawProfile.stickCurve,
    swapSticks: rawProfile.swapSticks,
    swapTriggersAndBumpers: rawProfile.swapTriggersAndBumpers,
  };
}

/**
 * Applies a symmetric stick response curve to a value in [-1, 1].
 * Supports 'linear', 'cubic', and 'expo' (power curve with configurable exponent).
 */
export function applyResponseCurve(value, curve = RESPONSE_CURVES.LINEAR, exponent) {
  if (!VALID_CURVES.has(curve)) {
    throw new RangeError(`Unknown response curve: ${String(curve)}`);
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 0;
  }

  const clamped = clamp(value, -1, 1);
  if (clamped === 0) return 0;

  const sign = clamped < 0 ? -1 : 1;
  const abs = Math.abs(clamped);

  let curved;
  switch (curve) {
    case RESPONSE_CURVES.LINEAR:
      curved = abs;
      break;
    case RESPONSE_CURVES.CUBIC:
      curved = abs * abs * abs;
      break;
    case RESPONSE_CURVES.EXPO: {
      const exp =
        exponent === undefined
          ? PROFILE_LIMITS.DEFAULT_EXPO_EXPONENT
          : Number.isFinite(exponent)
            ? clamp(exponent, PROFILE_LIMITS.MIN_EXPO_EXPONENT, PROFILE_LIMITS.MAX_EXPO_EXPONENT)
            : PROFILE_LIMITS.DEFAULT_EXPO_EXPONENT;
      curved = Math.pow(abs, exp);
      break;
    }
    default:
      curved = abs;
  }

  return normalizeZero(clamp(sign * curved, -1, 1));
}

function applyRadialStickCurve(x, y, curve, exponent) {
  const mag = Math.hypot(x, y);
  if (mag <= 0 || !Number.isFinite(mag)) {
    return { x: 0, y: 0 };
  }
  const clampedMag = Math.min(mag, 1);
  const curvedMag = applyResponseCurve(clampedMag, curve, exponent);
  const scale = curvedMag / mag;
  return {
    x: normalizeZero(clamp(x * scale, -1, 1)),
    y: normalizeZero(clamp(y * scale, -1, 1)),
  };
}

const LEFTY_KB_BINDINGS = {
  KeyI: INPUT_BUTTONS.MOVE_FORWARD,
  KeyK: INPUT_BUTTONS.MOVE_BACKWARD,
  KeyJ: INPUT_BUTTONS.MOVE_LEFT,
  KeyL: INPUT_BUTTONS.MOVE_RIGHT,
  Space: INPUT_BUTTONS.JUMP | INPUT_BUTTONS.A,
  ShiftRight: INPUT_BUTTONS.SPRINT | INPUT_BUTTONS.LS,
  ShiftLeft: INPUT_BUTTONS.SPRINT | INPUT_BUTTONS.LS,
  ControlRight: INPUT_BUTTONS.B,
  Slash: INPUT_BUTTONS.RS,
  KeyH: INPUT_BUTTONS.X,
  KeyP: INPUT_BUTTONS.Y,
  KeyU: INPUT_BUTTONS.LB,
  KeyO: INPUT_BUTTONS.RB,
  Tab: INPUT_BUTTONS.VIEW,
  Escape: INPUT_BUTTONS.MENU,
  ArrowUp: INPUT_BUTTONS.DPAD_UP,
  ArrowDown: INPUT_BUTTONS.DPAD_DOWN,
  ArrowLeft: INPUT_BUTTONS.DPAD_LEFT,
  ArrowRight: INPUT_BUTTONS.DPAD_RIGHT,
};

export const BUILTIN_PROFILES = Object.freeze({
  default: deepFreezeProfile({
    id: 'default',
    name: 'Default',
    keyBindings: { ...DEFAULT_KEY_BINDINGS },
    mouseSensitivityX: 1.0,
    mouseSensitivityY: 1.0,
    invertLookY: false,
    leftStickDeadzone: 0.15,
    rightStickDeadzone: 0.15,
    triggerDeadzone: 0.05,
    stickCurve: RESPONSE_CURVES.LINEAR,
    swapSticks: false,
    swapTriggersAndBumpers: false,
  }),
  southpaw: deepFreezeProfile({
    id: 'southpaw',
    name: 'Southpaw',
    keyBindings: { ...DEFAULT_KEY_BINDINGS },
    mouseSensitivityX: 1.0,
    mouseSensitivityY: 1.0,
    invertLookY: false,
    leftStickDeadzone: 0.15,
    rightStickDeadzone: 0.15,
    triggerDeadzone: 0.05,
    stickCurve: RESPONSE_CURVES.LINEAR,
    swapSticks: true,
    swapTriggersAndBumpers: false,
  }),
  fps: deepFreezeProfile({
    id: 'fps',
    name: 'FPS Precision',
    keyBindings: { ...DEFAULT_KEY_BINDINGS },
    mouseSensitivityX: 1.5,
    mouseSensitivityY: 1.5,
    invertLookY: false,
    leftStickDeadzone: 0.12,
    rightStickDeadzone: 0.1,
    triggerDeadzone: 0.03,
    stickCurve: RESPONSE_CURVES.EXPO,
    swapSticks: false,
    swapTriggersAndBumpers: false,
  }),
  'lefty-kb': deepFreezeProfile({
    id: 'lefty-kb',
    name: 'Left-Handed Keyboard (IJKL)',
    keyBindings: { ...LEFTY_KB_BINDINGS },
    mouseSensitivityX: 1.0,
    mouseSensitivityY: 1.0,
    invertLookY: false,
    leftStickDeadzone: 0.15,
    rightStickDeadzone: 0.15,
    triggerDeadzone: 0.05,
    stickCurve: RESPONSE_CURVES.LINEAR,
    swapSticks: false,
    swapTriggersAndBumpers: false,
  }),
});

/**
 * Transforms raw stick axes according to a profile:
 * 1. Optional stick swap (southpaw)
 * 2. Radial deadzone normalization matching InputBridge
 * 3. Radial response curve ('linear' | 'cubic' | 'expo')
 * 4. Optional look-Y inversion on the right (look) stick
 * All outputs are cleanly clamped to [-1, 1].
 */
export function transformStickAxes(lx, ly, rx, ry, profile = BUILTIN_PROFILES.default) {
  const cfg = isPlainObject(profile) ? profile : BUILTIN_PROFILES.default;

  const swap = Boolean(cfg.swapSticks);
  const inLx = swap ? rx : lx;
  const inLy = swap ? ry : ly;
  const inRx = swap ? lx : rx;
  const inRy = swap ? ly : ry;

  const leftDz = clamp(
    Number.isFinite(cfg.leftStickDeadzone) ? cfg.leftStickDeadzone : 0.15,
    PROFILE_LIMITS.MIN_DEADZONE,
    PROFILE_LIMITS.MAX_DEADZONE
  );
  const rightDz = clamp(
    Number.isFinite(cfg.rightStickDeadzone) ? cfg.rightStickDeadzone : 0.15,
    PROFILE_LIMITS.MIN_DEADZONE,
    PROFILE_LIMITS.MAX_DEADZONE
  );

  const curve = VALID_CURVES.has(cfg.stickCurve) ? cfg.stickCurve : RESPONSE_CURVES.LINEAR;

  const leftNorm = applyDeadzone(inLx, inLy, leftDz);
  const rightNorm = applyDeadzone(inRx, inRy, rightDz);

  const leftCurved = applyRadialStickCurve(leftNorm.x, leftNorm.y, curve);
  const rightCurved = applyRadialStickCurve(rightNorm.x, rightNorm.y, curve);

  const outRy = cfg.invertLookY ? normalizeZero(-rightCurved.y) : rightCurved.y;

  return {
    lx: normalizeZero(clamp(leftCurved.x, -1, 1)),
    ly: normalizeZero(clamp(leftCurved.y, -1, 1)),
    rx: normalizeZero(clamp(rightCurved.x, -1, 1)),
    ry: normalizeZero(clamp(outRy, -1, 1)),
  };
}

export class InputProfileManager {
  constructor(options = {}) {
    const opts = isPlainObject(options) ? options : {};
    this._storage = opts.storage !== undefined ? opts.storage : resolveDefaultStorage();
    this._storageKey =
      typeof opts.storageKey === 'string' && opts.storageKey.trim().length > 0
        ? opts.storageKey.trim()
        : DEFAULT_STORAGE_KEY;

    this._profiles = new Map();
    this._seedBuiltins();
    this._activeProfileId = 'default';

    if (opts.autoLoad === true) {
      this.load();
    }
  }

  _seedBuiltins() {
    this._profiles.clear();
    for (const preset of Object.values(BUILTIN_PROFILES)) {
      this._profiles.set(preset.id, cloneProfile(preset));
    }
  }

  _requireProfile(profileId) {
    if (typeof profileId !== 'string' || !this._profiles.has(profileId)) {
      throw new RangeError(`Unknown input profile: ${String(profileId)}`);
    }
    return this._profiles.get(profileId);
  }

  listProfiles() {
    return Array.from(this._profiles.values(), (profile) => cloneProfile(profile));
  }

  getProfile(profileId) {
    const profile = this._profiles.get(profileId);
    return profile ? cloneProfile(profile) : null;
  }

  getActiveProfile() {
    return cloneProfile(this._profiles.get(this._activeProfileId));
  }

  setActiveProfile(profileId) {
    const profile = this._requireProfile(profileId);
    this._activeProfileId = profile.id;
    return cloneProfile(profile);
  }

  createProfile(rawProfile) {
    const normalized = validateAndNormalizeProfile(rawProfile, 'profile');
    if (this._profiles.has(normalized.id)) {
      throw new RangeError(`Input profile "${normalized.id}" already exists`);
    }
    this._profiles.set(normalized.id, normalized);
    return cloneProfile(normalized);
  }

  updateProfile(profileId, patch) {
    const existing = this._requireProfile(profileId);
    if (!isPlainObject(patch)) {
      throw new TypeError('patch must be a plain object');
    }
    const merged = {
      ...cloneProfile(existing),
      ...patch,
      id: existing.id,
      keyBindings:
        patch.keyBindings !== undefined ? patch.keyBindings : { ...existing.keyBindings },
    };
    const validated = validateAndNormalizeProfile(merged, `profile("${profileId}")`);
    this._profiles.set(existing.id, validated);
    return cloneProfile(validated);
  }

  deleteProfile(profileId) {
    this._requireProfile(profileId);
    if (Object.prototype.hasOwnProperty.call(BUILTIN_PROFILES, profileId)) {
      throw new RangeError(`Cannot delete built-in input profile "${profileId}"`);
    }
    this._profiles.delete(profileId);
    if (this._activeProfileId === profileId) {
      this._activeProfileId = 'default';
    }
    return true;
  }

  resetProfile(profileId = this._activeProfileId) {
    if (!Object.prototype.hasOwnProperty.call(BUILTIN_PROFILES, profileId)) {
      throw new RangeError(`Cannot reset non-built-in input profile: ${String(profileId)}`);
    }
    const restored = cloneProfile(BUILTIN_PROFILES[profileId]);
    this._profiles.set(profileId, restored);
    return cloneProfile(restored);
  }

  /**
   * Rebinds a DOM KeyboardEvent.code to a named INPUT_BUTTONS action on a profile.
   * Validates buttonName against INPUT_BUTTONS, detects conflicts on either the target key
   * or another key already bound to the same button, and never mutates BUILTIN_PROFILES.
   */
  rebindKey(profileId, keyCode, buttonName, { allowOverwrite = true } = {}) {
    const profile = this._requireProfile(profileId);

    if (typeof keyCode !== 'string' || keyCode.trim().length === 0) {
      throw new TypeError('keyCode must be a non-empty string');
    }
    if (
      typeof buttonName !== 'string' ||
      !Object.prototype.hasOwnProperty.call(INPUT_BUTTONS, buttonName)
    ) {
      throw new RangeError(`Unknown button name: ${String(buttonName)}`);
    }

    const mask = INPUT_BUTTONS[buttonName];
    const bindings = profile.keyBindings;
    const currentMaskOnKey = bindings[keyCode] || 0;

    // Identify any other keys currently holding this button bit.
    const otherConflictKeys = [];
    for (const [existingCode, existingMask] of Object.entries(bindings)) {
      if (existingCode !== keyCode && (existingMask & mask) !== 0) {
        otherConflictKeys.push(existingCode);
      }
    }

    const keyHasDifferentBinding = currentMaskOnKey !== 0 && currentMaskOnKey !== mask;

    // No-op if the key is already bound solely to this button and no other key holds it.
    if (currentMaskOnKey === mask && otherConflictKeys.length === 0) {
      return { changed: false, conflictKey: null };
    }

    const conflictKey = keyHasDifferentBinding
      ? keyCode
      : otherConflictKeys.length > 0
        ? otherConflictKeys[0]
        : null;

    if (conflictKey !== null && !allowOverwrite) {
      if (keyHasDifferentBinding) {
        throw new Error(
          `Key "${keyCode}" is already bound in profile "${profileId}" (allowOverwrite is false)`
        );
      }
      throw new Error(
        `Button "${buttonName}" is already bound to "${conflictKey}" in profile "${profileId}" (allowOverwrite is false)`
      );
    }

    for (const otherCode of otherConflictKeys) {
      const remaining = (bindings[otherCode] & ~mask) >>> 0;
      if (remaining === 0) {
        delete bindings[otherCode];
      } else {
        bindings[otherCode] = remaining;
      }
    }

    bindings[keyCode] = mask;
    return { changed: true, conflictKey };
  }

  unbindKey(profileId, keyCode) {
    const profile = this._requireProfile(profileId);
    if (typeof keyCode !== 'string' || keyCode.trim().length === 0) {
      throw new TypeError('keyCode must be a non-empty string');
    }
    if (!Object.prototype.hasOwnProperty.call(profile.keyBindings, keyCode)) {
      return false;
    }
    delete profile.keyBindings[keyCode];
    return true;
  }

  transformStickAxes(lx, ly, rx, ry, profileOrId) {
    let profile;
    if (profileOrId === undefined || profileOrId === null) {
      profile = this._profiles.get(this._activeProfileId);
    } else if (typeof profileOrId === 'string') {
      profile = this._requireProfile(profileOrId);
    } else if (isPlainObject(profileOrId)) {
      profile = profileOrId;
    } else {
      throw new TypeError('profileOrId must be a profile ID string or profile object');
    }
    return transformStickAxes(lx, ly, rx, ry, profile);
  }

  /**
   * Produces an options object ready to pass directly into `new InputBridge(...)`.
   * Explicitly zeros out any DEFAULT_KEY_BINDINGS keys omitted by this profile so
   * InputBridge's default-merge does not leave removed defaults active.
   */
  toInputBridgeOptions(profileId = this._activeProfileId, overrides = {}) {
    const profile = this._requireProfile(profileId);
    const keyBindings = {};
    for (const defaultKey of Object.keys(DEFAULT_KEY_BINDINGS)) {
      keyBindings[defaultKey] = 0;
    }
    Object.assign(keyBindings, profile.keyBindings);

    return {
      ...(isPlainObject(overrides) ? overrides : {}),
      keyBindings,
      stickDeadzone: profile.leftStickDeadzone,
      leftStickDeadzone: profile.leftStickDeadzone,
      rightStickDeadzone: profile.rightStickDeadzone,
      triggerDeadzone: profile.triggerDeadzone,
      invertGamepadY: true,
      invertLookY: profile.invertLookY,
      mouseSensitivityX: profile.mouseSensitivityX,
      mouseSensitivityY: profile.mouseSensitivityY,
      stickCurve: profile.stickCurve,
      swapSticks: profile.swapSticks,
      swapTriggersAndBumpers: profile.swapTriggersAndBumpers,
      profileId: profile.id,
    };
  }

  exportJson() {
    const payload = {
      schemaVersion: PROFILE_SCHEMA_VERSION,
      activeProfileId: this._activeProfileId,
      profiles: this.listProfiles(),
    };
    return JSON.stringify(payload, null, 2);
  }

  importJson(jsonStr) {
    if (typeof jsonStr !== 'string') {
      throw new TypeError('importJson expects a JSON string');
    }

    const parsed = JSON.parse(jsonStr);
    if (!isPlainObject(parsed)) {
      throw new TypeError('Input profiles JSON root must be a plain object');
    }

    if (
      parsed.schemaVersion !== undefined &&
      parsed.schemaVersion !== PROFILE_SCHEMA_VERSION
    ) {
      throw new RangeError(
        `Unsupported input profiles schemaVersion: ${String(parsed.schemaVersion)}`
      );
    }

    if (!Array.isArray(parsed.profiles) || parsed.profiles.length === 0) {
      throw new TypeError('Input profiles JSON must include a non-empty profiles array');
    }

    if (typeof parsed.activeProfileId !== 'string' || parsed.activeProfileId.trim().length === 0) {
      throw new TypeError('Input profiles JSON must include a non-empty activeProfileId string');
    }

    const seenIds = new Set();
    const validatedProfiles = [];
    for (let i = 0; i < parsed.profiles.length; i++) {
      const normalized = validateAndNormalizeProfile(parsed.profiles[i], `profiles[${i}]`);
      if (seenIds.has(normalized.id)) {
        throw new RangeError(`Duplicate profile id "${normalized.id}" in imported JSON`);
      }
      seenIds.add(normalized.id);
      validatedProfiles.push(normalized);
    }

    const nextMap = new Map();
    for (const preset of Object.values(BUILTIN_PROFILES)) {
      nextMap.set(preset.id, cloneProfile(preset));
    }
    for (const profile of validatedProfiles) {
      nextMap.set(profile.id, profile);
    }

    const nextActiveId = parsed.activeProfileId.trim();
    if (!nextMap.has(nextActiveId)) {
      throw new RangeError(`activeProfileId "${nextActiveId}" does not match any profile`);
    }

    // Commit atomically only after full validation passes.
    this._profiles = nextMap;
    this._activeProfileId = nextActiveId;
    return this.listProfiles();
  }

  save() {
    if (!this._storage || typeof this._storage.setItem !== 'function') {
      return false;
    }
    this._storage.setItem(this._storageKey, this.exportJson());
    return true;
  }

  load() {
    if (!this._storage || typeof this._storage.getItem !== 'function') {
      return false;
    }
    const raw = this._storage.getItem(this._storageKey);
    if (raw === null || raw === undefined || raw === '') {
      return false;
    }
    this.importJson(raw);
    return true;
  }
}
