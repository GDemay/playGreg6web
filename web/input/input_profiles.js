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
  MAX_ID_LENGTH: 64,
  MAX_NAME_LENGTH: 80,
  MAX_KEY_CODE_LENGTH: 32,
  MAX_KEY_BINDINGS: 64,
  MAX_PROFILES: 64,
  MAX_JSON_BYTES: 65536,
});

export const PROFILE_ALLOWED_FIELDS = Object.freeze([
  'id',
  'name',
  'keyBindings',
  'mouseSensitivityX',
  'mouseSensitivityY',
  'invertLookY',
  'leftStickDeadzone',
  'rightStickDeadzone',
  'triggerDeadzone',
  'stickCurve',
  'swapSticks',
  'swapTriggersAndBumpers',
]);

const VALID_CURVES = new Set(Object.values(RESPONSE_CURVES));
const ALLOWED_PROFILE_FIELD_SET = new Set(PROFILE_ALLOWED_FIELDS);
const ALLOWED_PROFILE_ENVELOPE_FIELD_SET = new Set(['schemaVersion', 'profile']);
const ALLOWED_BUNDLE_FIELD_SET = new Set(['schemaVersion', 'activeProfileId', 'profiles']);
const FORBIDDEN_OBJECT_KEYS = new Set(['__proto__', 'constructor', 'prototype']);
const CONTROL_CHAR_RE = /[\u0000-\u001F\u007F-\u009F]/;
const VALID_PROFILE_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const VALID_KEY_CODE_RE = /^[A-Z][A-Za-z0-9]{1,31}$/;

function compareStrings(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

const SORTED_BUTTON_ENTRIES = Object.freeze(
  Object.entries(INPUT_BUTTONS)
    .sort(([a], [b]) => compareStrings(a, b))
    .map(([name, mask]) => Object.freeze([name, mask]))
);

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
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function getUtf8ByteLength(str) {
  if (typeof TextEncoder !== 'undefined') {
    return new TextEncoder().encode(str).byteLength;
  }
  return str.length;
}

function assertSafePlainObject(value, contextLabel, allowedKeys = null) {
  if (!isPlainObject(value)) {
    throw new TypeError(`${contextLabel} must be a plain object`);
  }
  const ownKeys = Object.getOwnPropertyNames(value);
  for (const key of ownKeys) {
    if (FORBIDDEN_OBJECT_KEYS.has(key)) {
      throw new RangeError(`${contextLabel} contains forbidden key "${key}"`);
    }
    if (CONTROL_CHAR_RE.test(key)) {
      throw new RangeError(`${contextLabel} contains control characters in key`);
    }
    if (allowedKeys !== null && !allowedKeys.has(key)) {
      throw new RangeError(`${contextLabel} contains unknown field "${key}"`);
    }
  }
}

function parseSafeJson(rawJson, contextLabel = 'JSON') {
  if (typeof rawJson !== 'string') {
    throw new TypeError(`${contextLabel} must be a JSON string`);
  }
  if (
    rawJson.length > PROFILE_LIMITS.MAX_JSON_BYTES ||
    getUtf8ByteLength(rawJson) > PROFILE_LIMITS.MAX_JSON_BYTES
  ) {
    throw new RangeError(
      `${contextLabel} exceeds max size of ${PROFILE_LIMITS.MAX_JSON_BYTES} bytes`
    );
  }
  return JSON.parse(rawJson, (key, value) => {
    if (key !== '' && FORBIDDEN_OBJECT_KEYS.has(key)) {
      throw new RangeError(`${contextLabel} contains forbidden key "${key}"`);
    }
    return value;
  });
}

function validateKeyCode(keyCode, contextLabel = 'keyCode') {
  if (typeof keyCode !== 'string' || keyCode.trim().length === 0) {
    throw new TypeError(`${contextLabel} must be a non-empty string`);
  }
  if (FORBIDDEN_OBJECT_KEYS.has(keyCode)) {
    throw new RangeError(`${contextLabel} contains forbidden key "${keyCode}"`);
  }
  if (CONTROL_CHAR_RE.test(keyCode)) {
    throw new RangeError(`${contextLabel} must not contain control characters`);
  }
  if (keyCode.length > PROFILE_LIMITS.MAX_KEY_CODE_LENGTH) {
    throw new RangeError(
      `${contextLabel} exceeds max length of ${PROFILE_LIMITS.MAX_KEY_CODE_LENGTH}`
    );
  }
  if (!VALID_KEY_CODE_RE.test(keyCode)) {
    throw new RangeError(`${contextLabel} "${keyCode}" is not a valid DOM KeyboardEvent.code`);
  }
  return keyCode;
}

function validateProfileId(rawId, contextLabel = 'profile.id') {
  if (typeof rawId !== 'string' || rawId.trim().length === 0) {
    throw new TypeError(`${contextLabel} must be a non-empty string`);
  }
  if (CONTROL_CHAR_RE.test(rawId)) {
    throw new RangeError(`${contextLabel} must not contain control characters`);
  }
  const trimmed = rawId.trim();
  if (FORBIDDEN_OBJECT_KEYS.has(trimmed)) {
    throw new RangeError(`${contextLabel} contains forbidden key "${trimmed}"`);
  }
  if (trimmed.length > PROFILE_LIMITS.MAX_ID_LENGTH) {
    throw new RangeError(
      `${contextLabel} exceeds max length of ${PROFILE_LIMITS.MAX_ID_LENGTH}`
    );
  }
  if (!VALID_PROFILE_ID_RE.test(trimmed)) {
    throw new RangeError(
      `${contextLabel} must match ${VALID_PROFILE_ID_RE.source}`
    );
  }
  return trimmed;
}

function validateProfileName(rawName, contextLabel = 'profile.name') {
  if (typeof rawName !== 'string' || rawName.trim().length === 0) {
    throw new TypeError(`${contextLabel} must be a non-empty string`);
  }
  if (CONTROL_CHAR_RE.test(rawName)) {
    throw new RangeError(`${contextLabel} must not contain control characters`);
  }
  const trimmed = rawName.trim();
  if (trimmed.length > PROFILE_LIMITS.MAX_NAME_LENGTH) {
    throw new RangeError(
      `${contextLabel} exceeds max length of ${PROFILE_LIMITS.MAX_NAME_LENGTH}`
    );
  }
  return trimmed;
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
  assertSafePlainObject(rawBindings, contextLabel);

  const ownKeys = Object.getOwnPropertyNames(rawBindings);
  if (ownKeys.length > PROFILE_LIMITS.MAX_KEY_BINDINGS) {
    throw new RangeError(
      `${contextLabel} exceeds max entry count of ${PROFILE_LIMITS.MAX_KEY_BINDINGS}`
    );
  }

  const normalized = {};
  for (const keyCode of ownKeys) {
    validateKeyCode(keyCode, `${contextLabel} key`);
    const rawMask = rawBindings[keyCode];
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

function validateAndNormalizeProfile(
  rawProfile,
  contextLabel = 'profile',
  { strict = false } = {}
) {
  assertSafePlainObject(rawProfile, contextLabel, strict ? ALLOWED_PROFILE_FIELD_SET : null);

  const id = validateProfileId(rawProfile.id, `${contextLabel}.id`);
  const name = validateProfileName(rawProfile.name, `${contextLabel}.name`);

  const numericFields = [
    ['mouseSensitivityX', PROFILE_LIMITS.MIN_SENSITIVITY, PROFILE_LIMITS.MAX_SENSITIVITY],
    ['mouseSensitivityY', PROFILE_LIMITS.MIN_SENSITIVITY, PROFILE_LIMITS.MAX_SENSITIVITY],
    ['leftStickDeadzone', PROFILE_LIMITS.MIN_DEADZONE, PROFILE_LIMITS.MAX_DEADZONE],
    ['rightStickDeadzone', PROFILE_LIMITS.MIN_DEADZONE, PROFILE_LIMITS.MAX_DEADZONE],
    ['triggerDeadzone', PROFILE_LIMITS.MIN_DEADZONE, PROFILE_LIMITS.MAX_DEADZONE],
  ];

  const validatedNumbers = {};
  for (const [field, min, max] of numericFields) {
    const val = rawProfile[field];
    if (typeof val !== 'number' || !Number.isFinite(val)) {
      throw new TypeError(`${contextLabel}.${field} must be a finite number`);
    }
    if (strict && (val < min || val > max)) {
      throw new RangeError(
        `${contextLabel}.${field} must be within [${min}, ${max}] (got ${val})`
      );
    }
    validatedNumbers[field] = normalizeZero(clamp(val, min, max));
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
    id,
    name,
    keyBindings,
    mouseSensitivityX: validatedNumbers.mouseSensitivityX,
    mouseSensitivityY: validatedNumbers.mouseSensitivityY,
    invertLookY: rawProfile.invertLookY,
    leftStickDeadzone: validatedNumbers.leftStickDeadzone,
    rightStickDeadzone: validatedNumbers.rightStickDeadzone,
    triggerDeadzone: validatedNumbers.triggerDeadzone,
    stickCurve: rawProfile.stickCurve,
    swapSticks: rawProfile.swapSticks,
    swapTriggersAndBumpers: rawProfile.swapTriggersAndBumpers,
  };
}

/**
 * Serializes a single validated input profile to a canonical JSON string
 * using the strict single-profile envelope `{ schemaVersion, profile }`.
 */
export function exportProfileJson(profile) {
  const normalized = validateAndNormalizeProfile(profile, 'profile', { strict: true });
  const sortedBindings = {};
  for (const key of Object.keys(normalized.keyBindings).sort(compareStrings)) {
    sortedBindings[key] = normalized.keyBindings[key];
  }
  const payload = {
    schemaVersion: PROFILE_SCHEMA_VERSION,
    profile: {
      ...normalized,
      keyBindings: sortedBindings,
    },
  };
  return JSON.stringify(payload, null, 2);
}

/**
 * Parses and strictly validates a single-profile JSON string (either the
 * `{ schemaVersion, profile }` envelope produced by `exportProfileJson` or a
 * standalone allowlisted profile object). Rejects unknown fields, prototype
 * pollution keys, control characters, out-of-range numeric settings, and
 * invalid DOM KeyboardEvent.code bindings.
 */
export function importProfileJson(rawJson, options = {}) {
  const opts = isPlainObject(options) ? options : {};
  const parsed = parseSafeJson(rawJson, 'importProfileJson');
  assertSafePlainObject(parsed, 'importProfileJson root');

  let rawProfile;
  const hasEnvelopeKeys =
    Object.prototype.hasOwnProperty.call(parsed, 'schemaVersion') ||
    Object.prototype.hasOwnProperty.call(parsed, 'profile');

  if (hasEnvelopeKeys) {
    assertSafePlainObject(parsed, 'profileEnvelope', ALLOWED_PROFILE_ENVELOPE_FIELD_SET);
    if (parsed.schemaVersion !== PROFILE_SCHEMA_VERSION) {
      throw new RangeError(
        `Unsupported input profile schemaVersion: ${String(parsed.schemaVersion)}`
      );
    }
    rawProfile = parsed.profile;
  } else {
    rawProfile = parsed;
  }

  const normalized = validateAndNormalizeProfile(rawProfile, 'profile', { strict: true });

  if (opts.rejectConflicts === true) {
    const conflicts = detectBindingConflicts(normalized);
    if (conflicts.length > 0) {
      const summary = conflicts
        .map((c) => `${c.action}(${c.keys.join(',')})`)
        .join('; ');
      throw new RangeError(`Imported profile contains binding conflicts: ${summary}`);
    }
  }

  return cloneProfile(normalized);
}

/**
 * Inspects a profile's `keyBindings` map and returns a deterministic list of
 * duplicate key-to-action conflicts (two or more keys mapped to the same
 * INPUT_BUTTONS action bit). Pass `{ includeMultiActionKeys: true }` to also
 * report single keys mapped to multiple distinct actions.
 */
export function detectBindingConflicts(profile, options = {}) {
  const opts = isPlainObject(options) ? options : {};
  const includeMultiActionKeys = opts.includeMultiActionKeys === true;

  assertSafePlainObject(profile, 'profile');
  if (!Object.prototype.hasOwnProperty.call(profile, 'keyBindings')) {
    throw new TypeError('profile.keyBindings must be a plain object');
  }

  const bindings = validateAndNormalizeKeyBindings(profile.keyBindings, 'profile.keyBindings');
  const sortedKeyEntries = Object.entries(bindings).sort(([a], [b]) => compareStrings(a, b));

  const conflicts = [];

  for (const [actionName, buttonBit] of SORTED_BUTTON_ENTRIES) {
    const boundKeys = [];
    for (const [keyCode, mask] of sortedKeyEntries) {
      if ((mask & buttonBit) !== 0) {
        boundKeys.push(keyCode);
      }
    }
    if (boundKeys.length > 1) {
      conflicts.push({
        type: 'duplicate-action',
        action: actionName,
        mask: buttonBit,
        keys: boundKeys,
      });
    }
  }

  if (includeMultiActionKeys) {
    for (const [keyCode, mask] of sortedKeyEntries) {
      const boundActions = [];
      for (const [actionName, buttonBit] of SORTED_BUTTON_ENTRIES) {
        if ((mask & buttonBit) !== 0) {
          boundActions.push(actionName);
        }
      }
      if (boundActions.length > 1) {
        conflicts.push({
          type: 'multi-action-key',
          key: keyCode,
          mask,
          actions: boundActions,
        });
      }
    }
  }

  return conflicts;
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
      this.load({
        fallbackToDefault: opts.fallbackToDefault === true,
        persistFallback: opts.persistFallback === true,
      });
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
    assertSafePlainObject(patch, 'patch');
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

  resetProfile(profileId = this._activeProfileId, { fallbackToDefault = false } = {}) {
    if (
      typeof profileId === 'string' &&
      Object.prototype.hasOwnProperty.call(BUILTIN_PROFILES, profileId)
    ) {
      const restored = cloneProfile(BUILTIN_PROFILES[profileId]);
      this._profiles.set(profileId, restored);
      return cloneProfile(restored);
    }

    if (fallbackToDefault) {
      if (typeof profileId === 'string' && this._profiles.has(profileId)) {
        const existing = this._profiles.get(profileId);
        const restoredCustom = {
          ...cloneProfile(BUILTIN_PROFILES.default),
          id: existing.id,
          name: existing.name,
        };
        this._profiles.set(profileId, restoredCustom);
        return cloneProfile(restoredCustom);
      }
      const restoredDefault = cloneProfile(BUILTIN_PROFILES.default);
      this._profiles.set('default', restoredDefault);
      this._activeProfileId = 'default';
      return cloneProfile(restoredDefault);
    }

    throw new RangeError(`Cannot reset non-built-in input profile: ${String(profileId)}`);
  }

  /**
   * Safe fallback reset: restores all built-in profiles cleanly from immutable
   * presets, drops custom/corrupt entries, sets active profile to 'default',
   * and optionally persists without throwing if storage is unavailable or full.
   */
  resetToDefaults({ persist = false } = {}) {
    this._seedBuiltins();
    this._activeProfileId = 'default';
    if (persist) {
      try {
        this.save();
      } catch {
        // Ignore storage write errors during safe fallback recovery.
      }
    }
    return this.getActiveProfile();
  }

  /**
   * Rebinds a DOM KeyboardEvent.code to a named INPUT_BUTTONS action on a profile.
   * Validates buttonName against INPUT_BUTTONS, detects conflicts on either the target key
   * or another key already bound to the same button, and never mutates BUILTIN_PROFILES.
   */
  rebindKey(profileId, keyCode, buttonName, { allowOverwrite = true } = {}) {
    const profile = this._requireProfile(profileId);

    validateKeyCode(keyCode, 'keyCode');
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
    validateKeyCode(keyCode, 'keyCode');
    if (!Object.prototype.hasOwnProperty.call(profile.keyBindings, keyCode)) {
      return false;
    }
    delete profile.keyBindings[keyCode];
    return true;
  }

  detectBindingConflicts(profileOrId = this._activeProfileId, options = {}) {
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
    return detectBindingConflicts(profile, options);
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

  exportProfileJson(profileId = this._activeProfileId) {
    const profile = this._requireProfile(profileId);
    return exportProfileJson(profile);
  }

  importProfileJson(rawJson, options = {}) {
    const opts = isPlainObject(options) ? options : {};
    const allowOverwrite = opts.allowOverwrite !== false;
    const activate = opts.activate === true;

    const normalized = importProfileJson(rawJson, {
      rejectConflicts: opts.rejectConflicts === true,
    });

    if (this._profiles.has(normalized.id) && !allowOverwrite) {
      throw new RangeError(`Input profile "${normalized.id}" already exists`);
    }

    this._profiles.set(normalized.id, normalized);
    if (activate) {
      this._activeProfileId = normalized.id;
    }
    return cloneProfile(normalized);
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

    const parsed = parseSafeJson(jsonStr, 'importJson');
    assertSafePlainObject(parsed, 'Input profiles JSON root', ALLOWED_BUNDLE_FIELD_SET);

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
    if (parsed.profiles.length > PROFILE_LIMITS.MAX_PROFILES) {
      throw new RangeError(
        `Input profiles JSON exceeds max profile count of ${PROFILE_LIMITS.MAX_PROFILES}`
      );
    }

    const nextActiveId = validateProfileId(parsed.activeProfileId, 'activeProfileId');

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

  load({ fallbackToDefault = false, persistFallback = false } = {}) {
    if (!this._storage || typeof this._storage.getItem !== 'function') {
      return false;
    }
    let raw;
    try {
      raw = this._storage.getItem(this._storageKey);
    } catch (err) {
      if (fallbackToDefault) {
        this.resetToDefaults({ persist: persistFallback });
        return false;
      }
      throw err;
    }
    if (raw === null || raw === undefined || raw === '') {
      return false;
    }
    try {
      this.importJson(raw);
      return true;
    } catch (err) {
      if (fallbackToDefault) {
        this.resetToDefaults({ persist: persistFallback });
        return false;
      }
      throw err;
    }
  }
}
