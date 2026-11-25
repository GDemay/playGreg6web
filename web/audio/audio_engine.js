// Main-thread WebAudio engine for PlayGta6Web.
// Manages AudioContext lifecycle, AudioWorklet registration & wiring, SharedArrayBuffer
// ring allocation/attachment, user-gesture unlock, volume/mute controls, and live telemetry.

export const PROCESSOR_NAME = 'playgta6-audio-processor';
export const DEFAULT_SAMPLE_RATE = 48000;
export const DEFAULT_CHANNELS = 2;
export const DEFAULT_CAPACITY_FRAMES = 4096;
export const DEFAULT_WORKLET_URL = new URL(
  '../workers/audio_worklet_processor.js',
  import.meta.url
).href;

// Atomic Int32 header layout at byte offset 0..32 of the SharedArrayBuffer.
// Matches web/workers/audio_worker.js and web/workers/audio_worklet_processor.js byte-for-byte.
export const HDR_WRITE_CURSOR = 0;
export const HDR_READ_CURSOR = 1;
export const HDR_CAPACITY_FRAMES = 2;
export const HDR_CHANNELS = 3;
export const HDR_UNDERRUNS = 4;
export const HDR_UNDERRUN_FRAMES = 5;
export const HDR_OVERFLOWS = 6;
export const HDR_DROPPED_FRAMES = 7;

export const RING_HEADER_INT32_COUNT = 8;
export const RING_HEADER_BYTES = RING_HEADER_INT32_COUNT * Int32Array.BYTES_PER_ELEMENT;

export const DEFAULT_MAX_RESUME_RETRIES = 5;
export const MAX_ERROR_MESSAGE_LENGTH = 256;
export const DEFAULT_GAIN_RAMP_MS = 15;
export const MAX_GAIN_RAMP_MS = 5000;
export const DEFAULT_UNLOCK_FADE_IN_MS = 25;
export const MAX_GUEST_PCM_CHANNELS = 8;
export const MIN_TARGET_LATENCY_MS = 10;
export const MAX_TARGET_LATENCY_MS = 120;
export const DEFAULT_TARGET_LATENCY_MS = 30;
export const AUDIO_PREFS_STORAGE_KEY = 'playgta6:audio:prefs:v1';
export const DEFAULT_GESTURE_EVENTS = Object.freeze([
  'pointerdown',
  'keydown',
  'touchstart',
  'click',
]);

const KNOWN_OUTPUT_STATES = new Set([
  'idle',
  'initializing',
  'suspended',
  'running',
  'interrupted',
  'closed',
  'unsupported',
  'error',
]);

function isSharedArrayBufferValue(buf) {
  return typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer;
}

function isDetachedArrayBuffer(buf) {
  if (!(buf instanceof ArrayBuffer)) {
    return false;
  }
  if (buf.detached === true) {
    return true;
  }
  try {
    void buf.byteLength;
    return false;
  } catch {
    return true;
  }
}

function extractInterleavedFloat32(samples) {
  if (samples === null || samples === undefined) {
    return null;
  }
  if (samples instanceof Float32Array) {
    try {
      const buf = samples.buffer;
      if (!buf || (!isSharedArrayBufferValue(buf) && isDetachedArrayBuffer(buf))) {
        return null;
      }
      return samples;
    } catch {
      return null;
    }
  }
  if (isSharedArrayBufferValue(samples)) {
    const count = Math.floor(samples.byteLength / Float32Array.BYTES_PER_ELEMENT);
    return new Float32Array(samples, 0, count);
  }
  if (samples instanceof ArrayBuffer) {
    if (isDetachedArrayBuffer(samples)) {
      return null;
    }
    const count = Math.floor(samples.byteLength / Float32Array.BYTES_PER_ELEMENT);
    return new Float32Array(samples, 0, count);
  }
  if (ArrayBuffer.isView(samples)) {
    try {
      const buf = samples.buffer;
      if (!buf || (!isSharedArrayBufferValue(buf) && isDetachedArrayBuffer(buf))) {
        return null;
      }
      const byteLen = samples.byteLength;
      const count = Math.floor(byteLen / Float32Array.BYTES_PER_ELEMENT);
      if (count <= 0) {
        return new Float32Array(0);
      }
      if (samples.byteOffset % Float32Array.BYTES_PER_ELEMENT === 0) {
        return new Float32Array(buf, samples.byteOffset, count);
      }
      const bytes = new Uint8Array(buf, samples.byteOffset, count * Float32Array.BYTES_PER_ELEMENT);
      const aligned = new Uint8Array(bytes.byteLength);
      aligned.set(bytes);
      return new Float32Array(aligned.buffer);
    } catch {
      return null;
    }
  }
  if (Array.isArray(samples)) {
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      out[i] = Number(samples[i]);
    }
    return out;
  }
  return null;
}

const CONTROL_CHARS_RE = /[\u0000-\u001f\u007f]+/g;
const FILE_URL_RE = /file:\/\/[^\s'")]+/gi;
const URL_USERINFO_RE = /\b([a-z][a-z0-9+.-]*:\/\/)[^/\s@]+@/gi;
const URL_QUERY_RE = /(\b(?:https?|wss?):\/\/[^\s?#]+)\?[^\s#'")]*/gi;
const AUTH_SCHEME_RE = /\b(Bearer|Basic)\s+[^\s'"]+/gi;
const SECRET_KV_RE =
  /\b(token|api[_-]?key|password|passwd|secret|authorization)(\s*[:=]\s*)[^\s,;'"&)]+/gi;
const WIN_DRIVE_PATH_RE = /\b[A-Za-z]:\\[^\s'")]+/g;
const WIN_UNC_PATH_RE = /(^|[\s('"])\\\\[^\s'")]+/g;
const POSIX_HOST_PATH_RE =
  /(^|[\s('"])\/(?:Users|home|var|etc|tmp|private|root|opt)\/[^\s'")]+/g;

function sanitizeFiniteNonNegInt(value, fallback = 0) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return fallback;
  }
  const floored = Math.floor(value);
  return floored === 0 ? 0 : floored;
}

function sanitizePositiveFiniteNumber(value, fallback) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return fallback;
  }
  return value;
}

function sanitizeRampMs(value, fallback = 0) {
  const safeFallback =
    typeof fallback === 'number' && Number.isFinite(fallback)
      ? fallback <= 0
        ? 0
        : fallback >= MAX_GAIN_RAMP_MS
          ? MAX_GAIN_RAMP_MS
          : fallback
      : 0;
  if (value === undefined || value === null) {
    return safeFallback;
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return safeFallback;
  }
  if (value <= 0) {
    return 0;
  }
  if (value >= MAX_GAIN_RAMP_MS) {
    return MAX_GAIN_RAMP_MS;
  }
  return value;
}

function clampPeakLevel(value, fallback = 0) {
  const safeFallback =
    typeof fallback === 'number' && Number.isFinite(fallback)
      ? fallback <= 0
        ? 0
        : fallback >= 1
          ? 1
          : fallback
      : 0;
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return safeFallback;
  }
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function isViewOnDetachedBuffer(view) {
  if (!ArrayBuffer.isView(view)) {
    return false;
  }
  try {
    const buf = view.buffer;
    if (!buf) {
      return true;
    }
    return !isSharedArrayBufferValue(buf) && isDetachedArrayBuffer(buf);
  } catch {
    return true;
  }
}

function normalizeGuestPcmRingDescriptor(ringDescriptor, fallbackSampleRate, fallbackChannels) {
  if (!ringDescriptor || typeof ringDescriptor !== 'object') {
    return { ok: false, reason: 'invalid-descriptor', ring: null };
  }

  let sab = null;
  if (ringDescriptor.sab !== undefined && ringDescriptor.sab !== null) {
    if (!isSharedArrayBufferValue(ringDescriptor.sab)) {
      return { ok: false, reason: 'invalid-buffer', ring: null };
    }
    sab = ringDescriptor.sab;
  }

  let buffer = null;
  if (ringDescriptor.buffer !== undefined && ringDescriptor.buffer !== null) {
    if (isSharedArrayBufferValue(ringDescriptor.buffer)) {
      buffer = ringDescriptor.buffer;
    } else if (ringDescriptor.buffer instanceof ArrayBuffer) {
      if (isDetachedArrayBuffer(ringDescriptor.buffer)) {
        return { ok: false, reason: 'detached-buffer', ring: null };
      }
      buffer = ringDescriptor.buffer;
    } else {
      return { ok: false, reason: 'invalid-buffer', ring: null };
    }
  }

  let memory = null;
  if (ringDescriptor.memory !== undefined && ringDescriptor.memory !== null) {
    if (typeof ringDescriptor.memory !== 'object') {
      return { ok: false, reason: 'invalid-buffer', ring: null };
    }
    try {
      const memBuf = ringDescriptor.memory.buffer;
      if (memBuf instanceof ArrayBuffer && isDetachedArrayBuffer(memBuf)) {
        return { ok: false, reason: 'detached-buffer', ring: null };
      }
      if (!memBuf || (!isSharedArrayBufferValue(memBuf) && !(memBuf instanceof ArrayBuffer))) {
        return { ok: false, reason: 'invalid-buffer', ring: null };
      }
      memory = ringDescriptor.memory;
    } catch {
      return { ok: false, reason: 'detached-buffer', ring: null };
    }
  }

  let samples = null;
  if (ringDescriptor.samples !== undefined && ringDescriptor.samples !== null) {
    if (!ArrayBuffer.isView(ringDescriptor.samples)) {
      return { ok: false, reason: 'invalid-buffer', ring: null };
    }
    if (isViewOnDetachedBuffer(ringDescriptor.samples)) {
      return { ok: false, reason: 'detached-buffer', ring: null };
    }
    samples = ringDescriptor.samples;
  }

  let data = null;
  if (ringDescriptor.data !== undefined && ringDescriptor.data !== null) {
    if (!ArrayBuffer.isView(ringDescriptor.data)) {
      return { ok: false, reason: 'invalid-buffer', ring: null };
    }
    if (isViewOnDetachedBuffer(ringDescriptor.data)) {
      return { ok: false, reason: 'detached-buffer', ring: null };
    }
    data = ringDescriptor.data;
  }

  const rawBase =
    ringDescriptor.base !== undefined
      ? ringDescriptor.base
      : ringDescriptor.dataPtr !== undefined
        ? ringDescriptor.dataPtr
        : ringDescriptor.dataOffset !== undefined
          ? ringDescriptor.dataOffset
          : ringDescriptor.byteOffset;
  let base = 0;
  let hasExplicitBase = false;
  if (rawBase !== undefined && rawBase !== null) {
    if (typeof rawBase !== 'number' || !Number.isInteger(rawBase) || rawBase < 0) {
      return { ok: false, reason: 'invalid-base', ring: null };
    }
    base = rawBase === 0 ? 0 : rawBase;
    hasExplicitBase = true;
  }

  if (!sab && !buffer && !memory && !samples && !data && !hasExplicitBase) {
    return { ok: false, reason: 'missing-buffer', ring: null };
  }

  let channels = fallbackChannels;
  if (ringDescriptor.channels !== undefined && ringDescriptor.channels !== null) {
    if (
      typeof ringDescriptor.channels !== 'number' ||
      !Number.isInteger(ringDescriptor.channels) ||
      ringDescriptor.channels < 1 ||
      ringDescriptor.channels > MAX_GUEST_PCM_CHANNELS
    ) {
      return { ok: false, reason: 'invalid-channels', ring: null };
    }
    channels = ringDescriptor.channels;
  }

  const rawSampleRate =
    ringDescriptor.sampleRate !== undefined
      ? ringDescriptor.sampleRate
      : ringDescriptor.sample_rate;
  let sampleRate = fallbackSampleRate;
  if (rawSampleRate !== undefined && rawSampleRate !== null) {
    if (typeof rawSampleRate !== 'number' || !Number.isFinite(rawSampleRate) || rawSampleRate <= 0) {
      return { ok: false, reason: 'invalid-sample-rate', ring: null };
    }
    sampleRate = rawSampleRate;
  }

  const rawCapacitySamples =
    ringDescriptor.capacity !== undefined
      ? ringDescriptor.capacity
      : ringDescriptor.capacitySamples;
  const rawCapacityFrames = ringDescriptor.capacityFrames;

  let capacity = 0;
  let capacityFrames = 0;
  if (rawCapacitySamples !== undefined && rawCapacitySamples !== null) {
    if (
      typeof rawCapacitySamples !== 'number' ||
      !Number.isInteger(rawCapacitySamples) ||
      rawCapacitySamples <= 0
    ) {
      return { ok: false, reason: 'invalid-capacity', ring: null };
    }
    capacity = rawCapacitySamples;
    capacityFrames = Math.floor(capacity / channels);
  } else if (rawCapacityFrames !== undefined && rawCapacityFrames !== null) {
    if (
      typeof rawCapacityFrames !== 'number' ||
      !Number.isInteger(rawCapacityFrames) ||
      rawCapacityFrames <= 0
    ) {
      return { ok: false, reason: 'invalid-capacity', ring: null };
    }
    capacityFrames = rawCapacityFrames;
    capacity = rawCapacityFrames * channels;
  } else if (samples && typeof samples.length === 'number' && samples.length > 0) {
    capacity = samples.length;
    capacityFrames = Math.floor(capacity / channels);
  } else if (data && typeof data.length === 'number' && data.length > 0) {
    capacity = data.length;
    capacityFrames = Math.floor(capacity / channels);
  } else {
    return { ok: false, reason: 'invalid-capacity', ring: null };
  }

  if (capacityFrames <= 0) {
    return { ok: false, reason: 'invalid-capacity', ring: null };
  }

  const rawReadIndexPtr =
    ringDescriptor.readIndexPtr !== undefined
      ? ringDescriptor.readIndexPtr
      : ringDescriptor.read_index_ptr !== undefined
        ? ringDescriptor.read_index_ptr
        : ringDescriptor.readIndexOffset;
  let readIndexPtr;
  if (rawReadIndexPtr !== undefined && rawReadIndexPtr !== null) {
    if (
      typeof rawReadIndexPtr !== 'number' ||
      !Number.isInteger(rawReadIndexPtr) ||
      rawReadIndexPtr < 0
    ) {
      return { ok: false, reason: 'invalid-index-ptr', ring: null };
    }
    readIndexPtr = rawReadIndexPtr === 0 ? 0 : rawReadIndexPtr;
  }

  const rawWriteIndexPtr =
    ringDescriptor.writeIndexPtr !== undefined
      ? ringDescriptor.writeIndexPtr
      : ringDescriptor.write_index_ptr !== undefined
        ? ringDescriptor.write_index_ptr
        : ringDescriptor.writeIndexOffset;
  let writeIndexPtr;
  if (rawWriteIndexPtr !== undefined && rawWriteIndexPtr !== null) {
    if (
      typeof rawWriteIndexPtr !== 'number' ||
      !Number.isInteger(rawWriteIndexPtr) ||
      rawWriteIndexPtr < 0
    ) {
      return { ok: false, reason: 'invalid-index-ptr', ring: null };
    }
    writeIndexPtr = rawWriteIndexPtr === 0 ? 0 : rawWriteIndexPtr;
  }

  let readIndex = null;
  if (ringDescriptor.readIndex !== undefined && ringDescriptor.readIndex !== null) {
    if (!ArrayBuffer.isView(ringDescriptor.readIndex) || isViewOnDetachedBuffer(ringDescriptor.readIndex)) {
      return { ok: false, reason: 'invalid-index-view', ring: null };
    }
    readIndex = ringDescriptor.readIndex;
  }

  let writeIndex = null;
  if (ringDescriptor.writeIndex !== undefined && ringDescriptor.writeIndex !== null) {
    if (!ArrayBuffer.isView(ringDescriptor.writeIndex) || isViewOnDetachedBuffer(ringDescriptor.writeIndex)) {
      return { ok: false, reason: 'invalid-index-view', ring: null };
    }
    writeIndex = ringDescriptor.writeIndex;
  }

  const normalized = {
    base,
    capacity,
    capacitySamples: capacity,
    capacityFrames,
    channels,
    sampleRate,
    sample_rate: sampleRate,
  };
  if (sab) normalized.sab = sab;
  if (buffer) normalized.buffer = buffer;
  if (memory) normalized.memory = memory;
  if (samples) normalized.samples = samples;
  if (data) normalized.data = data;
  if (readIndexPtr !== undefined) {
    normalized.readIndexPtr = readIndexPtr;
    normalized.read_index_ptr = readIndexPtr;
  }
  if (writeIndexPtr !== undefined) {
    normalized.writeIndexPtr = writeIndexPtr;
    normalized.write_index_ptr = writeIndexPtr;
  }
  if (readIndex) normalized.readIndex = readIndex;
  if (writeIndex) normalized.writeIndex = writeIndex;

  return { ok: true, reason: null, ring: normalized };
}

function sanitizeOutputState(state) {
  if (typeof state !== 'string') {
    return 'unknown';
  }
  const normalized = state.trim().toLowerCase();
  if (KNOWN_OUTPUT_STATES.has(normalized)) {
    return normalized;
  }
  return 'unknown';
}

/**
 * Sanitizes an error or string for safe telemetry snapshots:
 * strips control characters, redacts host paths, file:// URLs, URL credentials/queries,
 * and auth tokens, and bounds length to MAX_ERROR_MESSAGE_LENGTH.
 *
 * @param {unknown} err
 * @returns {string|null}
 */
export function sanitizeResumeError(err) {
  if (err === null || err === undefined) {
    return null;
  }

  let raw = '';
  if (typeof err === 'string') {
    raw = err;
  } else if (typeof err === 'object') {
    let name = '';
    let message = '';
    try {
      if (typeof err.name === 'string') {
        name = err.name.trim();
      }
    } catch {
      // Ignore hostile property getters.
    }
    try {
      if (typeof err.message === 'string') {
        message = err.message.trim();
      }
    } catch {
      // Ignore hostile property getters.
    }

    if (name && message && name !== 'Error' && !message.startsWith(name)) {
      raw = `${name}: ${message}`;
    } else if (message) {
      raw = message;
    } else if (name) {
      raw = name;
    } else {
      try {
        raw = String(err);
      } catch {
        raw = '';
      }
    }
  } else {
    try {
      raw = String(err);
    } catch {
      raw = '';
    }
  }

  let out = raw
    .replace(CONTROL_CHARS_RE, ' ')
    .replace(FILE_URL_RE, '[REDACTED_FILE_URL]')
    .replace(URL_USERINFO_RE, '$1[REDACTED]@')
    .replace(URL_QUERY_RE, '$1?[REDACTED]')
    .replace(AUTH_SCHEME_RE, '$1 [REDACTED]')
    .replace(SECRET_KV_RE, '$1$2[REDACTED]')
    .replace(WIN_DRIVE_PATH_RE, '[REDACTED_PATH]')
    .replace(WIN_UNC_PATH_RE, '$1[REDACTED_PATH]')
    .replace(POSIX_HOST_PATH_RE, '$1[REDACTED_PATH]')
    .replace(/\s+/g, ' ')
    .trim();

  if (out.length === 0) {
    return 'AudioContext resume failed';
  }
  if (out.length > MAX_ERROR_MESSAGE_LENGTH) {
    return `${out.slice(0, MAX_ERROR_MESSAGE_LENGTH - 3)}...`;
  }
  return out;
}

/**
 * Clamps a volume/gain input to a finite number in [0, 1].
 * Non-number and non-finite inputs (NaN, ±Infinity) resolve to the sanitized fallback.
 *
 * @param {unknown} value
 * @param {number} [fallback=1.0]
 * @returns {number}
 */
export function clampVolume(value, fallback = 1.0) {
  const safeFallback =
    typeof fallback === 'number' && Number.isFinite(fallback)
      ? fallback <= 0.0
        ? 0.0
        : fallback >= 1.0
          ? 1.0
          : fallback
      : 1.0;

  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return safeFallback;
  }
  if (value <= 0.0) return 0.0;
  if (value >= 1.0) return 1.0;
  return value;
}

export const clampGain = clampVolume;

/**
 * Clamps a jitter-buffer latency target in milliseconds to [MIN_TARGET_LATENCY_MS, MAX_TARGET_LATENCY_MS]
 * ([10, 120], default 30 ms). Non-number and non-finite inputs resolve to the sanitized fallback.
 *
 * @param {unknown} value
 * @param {number} [fallback=DEFAULT_TARGET_LATENCY_MS]
 * @returns {number}
 */
export function clampTargetLatencyMs(value, fallback = DEFAULT_TARGET_LATENCY_MS) {
  const safeFallback =
    typeof fallback === 'number' && Number.isFinite(fallback)
      ? fallback <= MIN_TARGET_LATENCY_MS
        ? MIN_TARGET_LATENCY_MS
        : fallback >= MAX_TARGET_LATENCY_MS
          ? MAX_TARGET_LATENCY_MS
          : fallback
      : DEFAULT_TARGET_LATENCY_MS;

  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return safeFallback;
  }
  if (value <= MIN_TARGET_LATENCY_MS) {
    return MIN_TARGET_LATENCY_MS;
  }
  if (value >= MAX_TARGET_LATENCY_MS) {
    return MAX_TARGET_LATENCY_MS;
  }
  return value;
}

function resolveStorageCandidate(storage) {
  if (storage !== undefined) {
    return storage && typeof storage === 'object' ? storage : null;
  }
  try {
    if (typeof globalThis !== 'undefined' && globalThis.localStorage) {
      return globalThis.localStorage;
    }
  } catch {
    // Accessing localStorage can throw SecurityError in sandboxed/opaque contexts.
  }
  return null;
}

/**
 * Persists strictly sanitized `{ muted, volume }` preferences under
 * `AUDIO_PREFS_STORAGE_KEY` (`'playgta6:audio:prefs:v1'`), catching
 * `SecurityError`, `QuotaExceededError`, and unavailable storage gracefully.
 *
 * @param {{ muted?: boolean, volume?: number }} prefs
 * @param {Storage|null} [storage]
 * @returns {{ ok: boolean, saved: boolean, prefs: { muted: boolean, volume: number }|null, key: string, reason: string|null, error?: string|null }}
 */
export function saveAudioPreferences(prefs, storage = undefined) {
  const targetStorage = resolveStorageCandidate(storage);
  if (!targetStorage || typeof targetStorage.setItem !== 'function') {
    return {
      ok: false,
      saved: false,
      prefs: null,
      key: AUDIO_PREFS_STORAGE_KEY,
      reason: 'storage-unavailable',
    };
  }

  let rawMuted = false;
  let rawVolume = 1.0;
  if (prefs && typeof prefs === 'object') {
    try {
      rawMuted = prefs.muted;
    } catch {
      rawMuted = false;
    }
    try {
      rawVolume = prefs.volume;
    } catch {
      rawVolume = 1.0;
    }
  }

  const sanitized = {
    muted: Boolean(rawMuted),
    volume: clampVolume(rawVolume, 1.0),
  };

  try {
    targetStorage.setItem(AUDIO_PREFS_STORAGE_KEY, JSON.stringify(sanitized));
    return {
      ok: true,
      saved: true,
      prefs: { muted: sanitized.muted, volume: sanitized.volume },
      key: AUDIO_PREFS_STORAGE_KEY,
      reason: null,
    };
  } catch (err) {
    return {
      ok: false,
      saved: false,
      prefs: null,
      key: AUDIO_PREFS_STORAGE_KEY,
      reason: 'storage-error',
      error: sanitizeResumeError(err),
    };
  }
}

/**
 * Loads and sanitizes `{ muted, volume }` preferences stored under
 * `AUDIO_PREFS_STORAGE_KEY` (`'playgta6:audio:prefs:v1'`), catching
 * `SecurityError`, corrupted JSON, and unavailable storage gracefully.
 *
 * @param {Storage|null} [storage]
 * @returns {{ ok: boolean, loaded: boolean, prefs: { muted: boolean, volume: number }|null, key: string, reason: string|null, error?: string|null }}
 */
export function loadAudioPreferences(storage = undefined) {
  const targetStorage = resolveStorageCandidate(storage);
  if (!targetStorage || typeof targetStorage.getItem !== 'function') {
    return {
      ok: false,
      loaded: false,
      prefs: null,
      key: AUDIO_PREFS_STORAGE_KEY,
      reason: 'storage-unavailable',
    };
  }

  let raw;
  try {
    raw = targetStorage.getItem(AUDIO_PREFS_STORAGE_KEY);
  } catch (err) {
    return {
      ok: false,
      loaded: false,
      prefs: null,
      key: AUDIO_PREFS_STORAGE_KEY,
      reason: 'storage-error',
      error: sanitizeResumeError(err),
    };
  }

  if (raw === null || raw === undefined) {
    return {
      ok: true,
      loaded: false,
      prefs: null,
      key: AUDIO_PREFS_STORAGE_KEY,
      reason: 'not-found',
    };
  }

  if (typeof raw !== 'string' || raw.trim().length === 0) {
    return {
      ok: false,
      loaded: false,
      prefs: null,
      key: AUDIO_PREFS_STORAGE_KEY,
      reason: 'invalid-data',
    };
  }

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return {
      ok: false,
      loaded: false,
      prefs: null,
      key: AUDIO_PREFS_STORAGE_KEY,
      reason: 'invalid-data',
    };
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return {
      ok: false,
      loaded: false,
      prefs: null,
      key: AUDIO_PREFS_STORAGE_KEY,
      reason: 'invalid-data',
    };
  }

  const hasMuted = typeof parsed.muted === 'boolean';
  const hasVolume = typeof parsed.volume === 'number' && Number.isFinite(parsed.volume);
  if (!hasMuted && !hasVolume) {
    return {
      ok: false,
      loaded: false,
      prefs: null,
      key: AUDIO_PREFS_STORAGE_KEY,
      reason: 'invalid-data',
    };
  }

  return {
    ok: true,
    loaded: true,
    prefs: {
      muted: hasMuted ? parsed.muted : false,
      volume: hasVolume ? clampVolume(parsed.volume, 1.0) : 1.0,
    },
    hasMuted,
    hasVolume,
    key: AUDIO_PREFS_STORAGE_KEY,
    reason: null,
  };
}

export function audioRingByteLength(
  capacityFrames = DEFAULT_CAPACITY_FRAMES,
  channels = DEFAULT_CHANNELS
) {
  if (!Number.isInteger(capacityFrames) || capacityFrames <= 0) {
    throw new RangeError(`capacityFrames must be a positive integer, got ${capacityFrames}`);
  }
  if (!Number.isInteger(channels) || channels <= 0) {
    throw new RangeError(`channels must be a positive integer, got ${channels}`);
  }
  return RING_HEADER_BYTES + capacityFrames * channels * Float32Array.BYTES_PER_ELEMENT;
}

/**
 * Allocates and atomically initializes a SharedArrayBuffer ring matching
 * AudioRingBuffer in web/workers/audio_worker.js.
 */
export function createAudioRingSab(
  capacityFrames = DEFAULT_CAPACITY_FRAMES,
  channels = DEFAULT_CHANNELS
) {
  if (typeof SharedArrayBuffer === 'undefined') {
    throw new Error(
      'SharedArrayBuffer is unavailable (requires Cross-Origin Isolation with COOP/COEP headers)'
    );
  }
  const sab = new SharedArrayBuffer(audioRingByteLength(capacityFrames, channels));
  const ctrl = new Int32Array(sab, 0, RING_HEADER_INT32_COUNT);
  Atomics.store(ctrl, HDR_WRITE_CURSOR, 0);
  Atomics.store(ctrl, HDR_READ_CURSOR, 0);
  Atomics.store(ctrl, HDR_CAPACITY_FRAMES, capacityFrames);
  Atomics.store(ctrl, HDR_CHANNELS, channels);
  Atomics.store(ctrl, HDR_UNDERRUNS, 0);
  Atomics.store(ctrl, HDR_UNDERRUN_FRAMES, 0);
  Atomics.store(ctrl, HDR_OVERFLOWS, 0);
  Atomics.store(ctrl, HDR_DROPPED_FRAMES, 0);
  return sab;
}

function resolveDefaultAudioContextCtor() {
  if (typeof globalThis === 'undefined') {
    return null;
  }
  return globalThis.AudioContext || globalThis.webkitAudioContext || null;
}

function resolveDefaultAudioWorkletNodeCtor() {
  if (typeof globalThis === 'undefined') {
    return null;
  }
  return globalThis.AudioWorkletNode || null;
}

export class AudioEngine {
  /**
   * @param {object} [options]
   * @param {number} [options.sampleRate=48000]
   * @param {number} [options.channels=2]
   * @param {number} [options.capacityFrames=4096]
   * @param {SharedArrayBuffer} [options.sab]
   * @param {{ sab: SharedArrayBuffer }} [options.ringBuffer]
   * @param {string} [options.workletUrl]
   * @param {string} [options.processorName='playgta6-audio-processor']
   * @param {number} [options.volume=1.0]
   * @param {number} [options.gain]
   * @param {boolean} [options.muted=false]
   * @param {number} [options.smoothingFrames]
   * @param {string|number} [options.latencyHint='interactive']
   * @param {number} [options.maxResumeRetries=5]
   * @param {Function} [options.AudioContextCtor]
   * @param {Function} [options.AudioWorkletNodeCtor]
   * @param {(err: Error) => void} [options.onError]
   * @param {(state: string) => void} [options.onStateChange]
   * @param {(data: any) => void} [options.onWorkletMessage]
   */
  constructor(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};

    this._requestedSampleRate =
      typeof opts.sampleRate === 'number' && Number.isFinite(opts.sampleRate) && opts.sampleRate > 0
        ? opts.sampleRate
        : DEFAULT_SAMPLE_RATE;
    this._requestedChannels =
      Number.isInteger(opts.channels) && opts.channels > 0
        ? opts.channels
        : DEFAULT_CHANNELS;
    this._requestedCapacityFrames =
      Number.isInteger(opts.capacityFrames) && opts.capacityFrames > 0
        ? opts.capacityFrames
        : DEFAULT_CAPACITY_FRAMES;
    this._requestedMaxQueuedFrames =
      Number.isInteger(opts.maxQueuedFrames) && opts.maxQueuedFrames > 0
        ? opts.maxQueuedFrames
        : null;

    this._workletUrl =
      typeof opts.workletUrl === 'string' && opts.workletUrl.length > 0
        ? opts.workletUrl
        : DEFAULT_WORKLET_URL;
    this._processorName =
      typeof opts.processorName === 'string' && opts.processorName.length > 0
        ? opts.processorName
        : PROCESSOR_NAME;
    this._latencyHint = opts.latencyHint !== undefined ? opts.latencyHint : 'interactive';
    this._smoothingFrames = opts.smoothingFrames;
    this._maxResumeRetries = sanitizeFiniteNonNegInt(
      opts.maxResumeRetries,
      DEFAULT_MAX_RESUME_RETRIES
    );
    this._autoResumeOnGesture = Boolean(opts.autoResumeOnGesture);
    this._gestureTarget = opts.gestureTarget;

    this._AudioContextCtor =
      opts.AudioContextCtor !== undefined
        ? opts.AudioContextCtor
        : resolveDefaultAudioContextCtor();
    this._AudioWorkletNodeCtor =
      opts.AudioWorkletNodeCtor !== undefined
        ? opts.AudioWorkletNodeCtor
        : resolveDefaultAudioWorkletNodeCtor();

    this._onError = typeof opts.onError === 'function' ? opts.onError : null;
    this._onStateChange = typeof opts.onStateChange === 'function' ? opts.onStateChange : null;
    this._onWorkletMessage =
      typeof opts.onWorkletMessage === 'function' ? opts.onWorkletMessage : null;

    const rawInitialVolume =
      opts.volume !== undefined
        ? opts.volume
        : opts.gain !== undefined
          ? opts.gain
          : opts.masterGain !== undefined
            ? opts.masterGain
            : 1.0;
    this._volume = clampVolume(rawInitialVolume, 1.0);
    this._muted = Boolean(opts.muted);
    this._audioWorker = opts.audioWorker || opts.worker || opts.bridge || null;
    this._unlockFadeInMs = sanitizeRampMs(opts.unlockFadeInMs, DEFAULT_UNLOCK_FADE_IN_MS);
    this._targetLatencyMs = clampTargetLatencyMs(
      opts.targetLatencyMs,
      DEFAULT_TARGET_LATENCY_MS
    );
    this._autoRecoverOnInterruption = opts.autoRecoverOnInterruption !== false;

    this._sab = null;
    this._ctrl = null;
    this._samples = null;
    this._fallbackCtrl = null;
    this._fallbackSamples = null;
    this._capacityFrames = this._requestedCapacityFrames;
    this._maxQueuedFrames =
      this._requestedMaxQueuedFrames !== null
        ? Math.min(this._requestedMaxQueuedFrames, this._requestedCapacityFrames)
        : this._requestedCapacityFrames;
    this._channels = this._requestedChannels;
    this._modulus = this._requestedCapacityFrames * 2;

    this._context = null;
    this._node = null;
    this._masterGainNode = null;
    this._masterGainRampMs = 0;
    this._masterGainHeldSilent = false;
    this._unlockedOnGesture = false;
    this._wasPlaying = false;
    this._explicitlySuspended = false;
    this._pendingAutoRecovery = false;
    this._inAutoRecovery = false;
    this._inManualResume = false;
    this._status = 'idle';
    this._error = null;
    this._destroyed = false;
    this._startPromise = null;
    this._resumePromise = null;
    this._gestureCleanup = null;
    this._lastGestureTarget = opts.gestureTarget || null;
    this._lastGestureOptions = null;
    this._boundContextStateChange = null;

    this._guestPcmRing = null;
    this._guestRingAttached = false;
    this._guestFramesDrained = 0;
    this._guestUnderruns = 0;
    this._guestOverruns = 0;
    this._guestPeakLevel = 0;

    this._resumeAttempts = 0;
    this._resumeSuccesses = 0;
    this._resumeFailures = 0;
    this._interruptionCount = 0;
    this._autoRecoveryCount = 0;
    this._stateTransitions = 0;
    this._lastResumeError = null;
    this._lastTrackedState = 'idle';
    this._gainClamps = 0;
    this._invalidGainInputs = 0;
    this._submittedFrames = 0;
    this._submittedBatches = 0;
    this._consumedFrames = 0;
    this._sanitizedSamples = 0;

    const initialSab = opts.sab || (opts.ringBuffer && opts.ringBuffer.sab) || null;
    if (initialSab) {
      this.attachSab(initialSab);
    } else if (typeof SharedArrayBuffer !== 'undefined') {
      this.attachSab(
        createAudioRingSab(this._requestedCapacityFrames, this._requestedChannels)
      );
    } else {
      this._initFallbackRing(this._requestedCapacityFrames, this._requestedChannels);
    }
  }

  get sab() {
    return this._sab;
  }

  get context() {
    return this._context;
  }

  get node() {
    return this._node;
  }

  get volume() {
    return this._volume;
  }

  get gain() {
    return this._volume;
  }

  get masterGain() {
    return this._volume;
  }

  get masterGainNode() {
    return this._masterGainNode;
  }

  get audioWorker() {
    return this._audioWorker;
  }

  get guestRingAttached() {
    return this._guestRingAttached;
  }

  get guestPcmRing() {
    return this._guestPcmRing ? { ...this._guestPcmRing } : null;
  }

  get unlockedOnGesture() {
    return this._unlockedOnGesture;
  }

  get muted() {
    return this._muted;
  }

  get targetLatencyMs() {
    return this._targetLatencyMs;
  }

  get interruptionCount() {
    return this._interruptionCount;
  }

  get autoRecoveryCount() {
    return this._autoRecoveryCount;
  }

  get pendingAutoRecovery() {
    return this._pendingAutoRecovery;
  }

  get error() {
    return this._error;
  }

  /**
   * Attaches or replaces the host audio worker / bridge used to forward guest
   * PCM ring attachment, detachment, and master gain/mute control messages.
   *
   * @param {object|null} workerOrBridge
   * @returns {object|null}
   */
  setAudioWorker(workerOrBridge) {
    this._audioWorker = workerOrBridge && typeof workerOrBridge === 'object' ? workerOrBridge : null;
    return this._audioWorker;
  }

  _readContextState(ctx = this._context) {
    if (!ctx) {
      return null;
    }
    try {
      const raw = ctx.state;
      return typeof raw === 'string' && raw.length > 0 ? raw : null;
    } catch {
      return null;
    }
  }

  _initFallbackRing(capacityFrames, channels) {
    const cap =
      Number.isInteger(capacityFrames) && capacityFrames > 0
        ? capacityFrames
        : DEFAULT_CAPACITY_FRAMES;
    const ch = Number.isInteger(channels) && channels > 0 ? channels : DEFAULT_CHANNELS;
    const ctrl = new Int32Array(RING_HEADER_INT32_COUNT);
    ctrl[HDR_WRITE_CURSOR] = 0;
    ctrl[HDR_READ_CURSOR] = 0;
    ctrl[HDR_CAPACITY_FRAMES] = cap;
    ctrl[HDR_CHANNELS] = ch;
    ctrl[HDR_UNDERRUNS] = 0;
    ctrl[HDR_UNDERRUN_FRAMES] = 0;
    ctrl[HDR_OVERFLOWS] = 0;
    ctrl[HDR_DROPPED_FRAMES] = 0;
    this._fallbackCtrl = ctrl;
    this._fallbackSamples = new Float32Array(cap * ch);
    this._capacityFrames = cap;
    this._channels = ch;
    this._modulus = cap * 2;
    this._maxQueuedFrames =
      this._requestedMaxQueuedFrames !== null
        ? Math.min(this._requestedMaxQueuedFrames, cap)
        : cap;
  }

  _activeCtrl() {
    return this._ctrl || this._fallbackCtrl;
  }

  _activeSamples() {
    return this._samples || this._fallbackSamples;
  }

  _loadHdr(index) {
    if (this._ctrl) {
      return Atomics.load(this._ctrl, index);
    }
    if (this._fallbackCtrl) {
      return this._fallbackCtrl[index];
    }
    return 0;
  }

  _storeHdr(index, value) {
    if (this._ctrl) {
      Atomics.store(this._ctrl, index, value);
    } else if (this._fallbackCtrl) {
      this._fallbackCtrl[index] = value;
    }
  }

  _addHdr(index, delta) {
    if (this._ctrl) {
      return Atomics.add(this._ctrl, index, delta);
    }
    if (this._fallbackCtrl) {
      const prev = this._fallbackCtrl[index];
      this._fallbackCtrl[index] = prev + delta;
      return prev;
    }
    return 0;
  }

  _computeBufferedFrames() {
    const ctrl = this._activeCtrl();
    const cap = this._capacityFrames;
    if (!ctrl || cap <= 0) {
      return 0;
    }
    const write = this._loadHdr(HDR_WRITE_CURSOR);
    const read = this._loadHdr(HDR_READ_CURSOR);
    let used = write - read;
    if (used < 0) {
      used += cap * 2;
    }
    if (!Number.isFinite(used) || used < 0) {
      return 0;
    }
    if (used > cap) {
      return cap;
    }
    return Math.floor(used);
  }

  get state() {
    if (this._destroyed) {
      return 'closed';
    }
    if (this._status === 'unsupported' || this._status === 'error') {
      return this._status;
    }
    const ctxState = this._readContextState();
    if (ctxState !== null) {
      return ctxState;
    }
    return this._status;
  }

  get outputState() {
    return sanitizeOutputState(this.state);
  }

  get contextState() {
    return sanitizeOutputState(this.state);
  }

  get sampleRate() {
    if (this._context) {
      try {
        const rate = this._context.sampleRate;
        if (typeof rate === 'number' && Number.isFinite(rate) && rate > 0) {
          return rate;
        }
      } catch {
        // Ignore hostile getter.
      }
    }
    return this._requestedSampleRate;
  }

  /**
   * Attaches an existing SharedArrayBuffer (e.g. from AudioRingBuffer) to the engine
   * and forwards it to the active AudioWorkletNode if already running.
   */
  attachSab(sab) {
    if (typeof SharedArrayBuffer === 'undefined' || !(sab instanceof SharedArrayBuffer)) {
      throw new TypeError('AudioEngine.attachSab requires a SharedArrayBuffer');
    }
    if (sab.byteLength < RING_HEADER_BYTES) {
      throw new RangeError(
        `SharedArrayBuffer too small (${sab.byteLength} bytes; need > ${RING_HEADER_BYTES})`
      );
    }

    const ctrl = new Int32Array(sab, 0, RING_HEADER_INT32_COUNT);
    let cap = Atomics.load(ctrl, HDR_CAPACITY_FRAMES);
    let ch = Atomics.load(ctrl, HDR_CHANNELS);

    if (cap === 0) {
      ch = ch > 0 ? ch : this._requestedChannels;
      const maxFrames = Math.floor(
        (sab.byteLength - RING_HEADER_BYTES) / (ch * Float32Array.BYTES_PER_ELEMENT)
      );
      if (maxFrames <= 0) {
        throw new RangeError('SharedArrayBuffer has zero frame capacity');
      }
      cap = maxFrames;
      Atomics.store(ctrl, HDR_WRITE_CURSOR, 0);
      Atomics.store(ctrl, HDR_READ_CURSOR, 0);
      Atomics.store(ctrl, HDR_CAPACITY_FRAMES, cap);
      Atomics.store(ctrl, HDR_CHANNELS, ch);
      Atomics.store(ctrl, HDR_UNDERRUNS, 0);
      Atomics.store(ctrl, HDR_UNDERRUN_FRAMES, 0);
      Atomics.store(ctrl, HDR_OVERFLOWS, 0);
      Atomics.store(ctrl, HDR_DROPPED_FRAMES, 0);
    } else {
      ch = ch > 0 ? ch : this._requestedChannels;
      const requiredBytes = audioRingByteLength(cap, ch);
      if (sab.byteLength < requiredBytes) {
        throw new RangeError(
          `SharedArrayBuffer (${sab.byteLength} bytes) is smaller than header requires (${requiredBytes} bytes)`
        );
      }
    }

    this._sab = sab;
    this._ctrl = ctrl;
    this._samples = new Float32Array(sab, RING_HEADER_BYTES, cap * ch);
    this._fallbackCtrl = null;
    this._fallbackSamples = null;
    this._capacityFrames = cap;
    this._channels = ch;
    this._modulus = cap * 2;
    this._maxQueuedFrames =
      this._requestedMaxQueuedFrames !== null
        ? Math.min(this._requestedMaxQueuedFrames, cap)
        : cap;

    if (this._node && this._node.port && typeof this._node.port.postMessage === 'function') {
      try {
        this._node.port.postMessage({ type: 'attach', sab });
      } catch {
        // Ignore postMessage errors on closed/detached port.
      }
    }

    return sab;
  }

  _fail(status, message, cause = null) {
    const err = cause instanceof Error ? cause : new Error(message);
    if (cause instanceof Error && message && err.message !== message) {
      err.diagnostic = message;
    }
    this._status = status;
    this._error = err;
    if (this._onError) {
      try {
        this._onError(err);
      } catch {
        // Never allow user callbacks to break graceful audio error reporting.
      }
    }
    this._emitStateChange();
    return {
      ok: false,
      state: this.state,
      sampleRate: this.sampleRate,
      sab: this._sab,
      error: err,
    };
  }

  _recordStateTransition(nextState) {
    const normalized = sanitizeOutputState(nextState);
    if (normalized !== this._lastTrackedState) {
      this._lastTrackedState = normalized;
      this._stateTransitions += 1;
    }
  }

  _emitStateChange() {
    const currentState = this.state;
    this._recordStateTransition(currentState);
    if (this._onStateChange) {
      try {
        this._onStateChange(currentState);
      } catch {
        // Ignore callback exceptions.
      }
    }
  }

  async _safeCloseContext(ctx) {
    if (!ctx || typeof ctx.close !== 'function') {
      return;
    }
    if (ctx.state === 'closed') {
      return;
    }
    try {
      await ctx.close();
    } catch {
      // Ignore close errors during teardown / fallback.
    }
  }

  /**
   * Initializes the AudioContext, loads the AudioWorklet module, and connects
   * the AudioWorkletNode to `context.destination`.
   *
   * Never throws unhandled exceptions when WebAudio / AudioWorklet / SAB is
   * unsupported; returns `{ ok: false, state, error }` instead.
   */
  async start() {
    if (this._destroyed) {
      return {
        ok: false,
        state: 'closed',
        sampleRate: this.sampleRate,
        sab: this._sab,
        error: new Error('AudioEngine has been destroyed'),
      };
    }

    if (this._node && this._context) {
      return {
        ok: true,
        state: this.state,
        sampleRate: this.sampleRate,
        sab: this._sab,
        error: null,
      };
    }

    if (this._startPromise) {
      return this._startPromise;
    }

    this._startPromise = this._startInternal().finally(() => {
      this._startPromise = null;
    });
    return this._startPromise;
  }

  async _startInternal() {
    this._status = 'initializing';
    this._error = null;

    if (!this._sab) {
      if (typeof SharedArrayBuffer === 'undefined') {
        return this._fail(
          'unsupported',
          'SharedArrayBuffer is unavailable; Cross-Origin Isolation (COOP/COEP) is required for low-latency audio'
        );
      }
      try {
        this.attachSab(
          createAudioRingSab(this._requestedCapacityFrames, this._requestedChannels)
        );
      } catch (err) {
        return this._fail('error', 'Failed to allocate audio SharedArrayBuffer', err);
      }
    }

    const AudioContextCtor = this._AudioContextCtor;
    if (typeof AudioContextCtor !== 'function') {
      return this._fail(
        'unsupported',
        'WebAudio AudioContext is not supported in this browser/environment'
      );
    }

    const AudioWorkletNodeCtor = this._AudioWorkletNodeCtor;
    if (typeof AudioWorkletNodeCtor !== 'function') {
      return this._fail(
        'unsupported',
        'AudioWorkletNode is not supported in this browser/environment (requires a secure context)'
      );
    }

    let ctx;
    try {
      ctx = new AudioContextCtor({
        sampleRate: this._requestedSampleRate,
        latencyHint: this._latencyHint,
      });
    } catch (err) {
      return this._fail('error', 'Failed to construct AudioContext', err);
    }

    if (!ctx.audioWorklet || typeof ctx.audioWorklet.addModule !== 'function') {
      await this._safeCloseContext(ctx);
      return this._fail(
        'unsupported',
        'AudioContext.audioWorklet is unavailable in this browser (requires HTTPS or localhost secure context)'
      );
    }

    try {
      await ctx.audioWorklet.addModule(this._workletUrl);
    } catch (err) {
      await this._safeCloseContext(ctx);
      return this._fail(
        'error',
        `Failed to load AudioWorklet module from ${this._workletUrl}`,
        err
      );
    }

    if (this._destroyed) {
      await this._safeCloseContext(ctx);
      return {
        ok: false,
        state: 'closed',
        sampleRate: this._requestedSampleRate,
        sab: this._sab,
        error: new Error('AudioEngine was destroyed during initialization'),
      };
    }

    let masterGainNode = null;
    if (typeof ctx.createGain === 'function') {
      try {
        const candidate = ctx.createGain();
        if (candidate && candidate.gain && typeof candidate.connect === 'function') {
          masterGainNode = candidate;
        }
      } catch {
        masterGainNode = null;
      }
    }

    const initialCtxState = this._readContextState(ctx);
    const holdSilentUntilResume =
      masterGainNode !== null &&
      (initialCtxState === 'suspended' || initialCtxState === 'interrupted');

    const processorOptions = {
      sab: this._sab,
      // Keep the worklet at unity gain when a WebAudio master GainNode owns
      // gain/mute ramping so output is never double-attenuated.
      gain: masterGainNode ? 1.0 : this._volume,
      muted: masterGainNode ? false : this._muted,
    };
    if (this._smoothingFrames !== undefined) {
      processorOptions.smoothingFrames = this._smoothingFrames;
    }

    let node;
    try {
      node = new AudioWorkletNodeCtor(ctx, this._processorName, {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [this._channels],
        processorOptions,
      });
    } catch (err) {
      await this._safeCloseContext(ctx);
      return this._fail(
        'error',
        `Failed to instantiate AudioWorkletNode "${this._processorName}"`,
        err
      );
    }

    if (node.port) {
      node.port.onmessage = (event) => {
        const data = event ? event.data : null;
        if (data && data.type === 'error') {
          const workletErr = new Error(data.error || 'AudioWorklet processor error');
          this._error = workletErr;
          if (this._onError) {
            try {
              this._onError(workletErr);
            } catch {
              // Ignore callback errors.
            }
          }
        } else if (
          data &&
          (data.type === 'guest-audio-telemetry' || data.type === 'audio-stats')
        ) {
          this.updateGuestAudioTelemetry(data.stats || data);
        }
        if (this._onWorkletMessage) {
          try {
            this._onWorkletMessage(data);
          } catch {
            // Ignore callback errors.
          }
        }
      };
    }

    try {
      if (masterGainNode && typeof node.connect === 'function' && ctx.destination) {
        const initGain = holdSilentUntilResume ? 0.0 : this._muted ? 0.0 : this._volume;
        const now =
          typeof ctx.currentTime === 'number' && Number.isFinite(ctx.currentTime)
            ? ctx.currentTime
            : 0;
        try {
          if (typeof masterGainNode.gain.setValueAtTime === 'function') {
            masterGainNode.gain.setValueAtTime(initGain, now);
          } else {
            masterGainNode.gain.value = initGain;
          }
        } catch {
          // Ignore AudioParam init errors on restricted mocks.
        }
        node.connect(masterGainNode);
        masterGainNode.connect(ctx.destination);
      } else if (typeof node.connect === 'function' && ctx.destination) {
        node.connect(ctx.destination);
      }
    } catch (err) {
      try {
        if (typeof node.disconnect === 'function') node.disconnect();
      } catch {
        // Ignore.
      }
      try {
        if (masterGainNode && typeof masterGainNode.disconnect === 'function') {
          masterGainNode.disconnect();
        }
      } catch {
        // Ignore.
      }
      await this._safeCloseContext(ctx);
      return this._fail('error', 'Failed to connect AudioWorkletNode to destination', err);
    }

    this._context = ctx;
    this._node = node;
    this._masterGainNode = masterGainNode;
    this._masterGainHeldSilent = holdSilentUntilResume;
    this._status = initialCtxState || 'running';
    this._wasPlaying = this._status === 'running';
    this._explicitlySuspended = false;
    this._pendingAutoRecovery = false;

    this._boundContextStateChange = () => {
      if (this._destroyed || !this._context) {
        return;
      }
      const prevState = this._lastTrackedState;
      const liveState = this._readContextState(this._context);
      if (liveState !== null) {
        this._status = liveState;
      }
      const nextState = sanitizeOutputState(this.state);

      if (nextState === 'running') {
        this._wasPlaying = true;
        this._explicitlySuspended = false;
        if (this._pendingAutoRecovery) {
          if (this._inManualResume && !this._inAutoRecovery) {
            this._pendingAutoRecovery = false;
            this._clearGestureListeners();
          } else {
            this._markInterruptionRecovered();
          }
          if (
            this._masterGainNode &&
            this._masterGainHeldSilent &&
            !this._inUnlockGesture &&
            !this._inAutoRecovery
          ) {
            this._masterGainHeldSilent = false;
            this._scheduleMasterGainRamp(
              this._muted ? 0.0 : this._volume,
              this._unlockFadeInMs,
              0.0
            );
          }
        }
      } else if (
        (nextState === 'interrupted' || nextState === 'suspended') &&
        !this._explicitlySuspended &&
        !this._pendingAutoRecovery &&
        (prevState === 'running' || this._wasPlaying || this._unlockedOnGesture)
      ) {
        this._interruptionCount += 1;
        this._pendingAutoRecovery = true;
        if (this._masterGainNode) {
          this._masterGainHeldSilent = true;
        }
        this._forwardToAudioWorker(
          'setOutputState',
          {
            type: 'audio-context-state',
            setOutputState: true,
            outputState: 'suspended',
            interrupted: nextState === 'interrupted',
          },
          ['suspended']
        );
        if (this._autoRecoverOnInterruption) {
          this._armInterruptionRecovery();
        }
      } else if (nextState === 'closed') {
        this._pendingAutoRecovery = false;
        this._clearGestureListeners();
      }

      this._emitStateChange();
    };
    try {
      if (typeof ctx.addEventListener === 'function') {
        ctx.addEventListener('statechange', this._boundContextStateChange);
      } else if ('onstatechange' in ctx) {
        ctx.onstatechange = this._boundContextStateChange;
      }
    } catch {
      // Ignore event registration failures in restricted/headless contexts.
    }

    this._emitStateChange();

    if (
      this._autoResumeOnGesture &&
      (this.outputState === 'suspended' || this.outputState === 'interrupted')
    ) {
      this.resumeOnUserGesture(this._gestureTarget);
    }

    return {
      ok: true,
      state: this.state,
      sampleRate: this.sampleRate,
      sab: this._sab,
      error: null,
    };
  }

  _clearGestureListeners() {
    if (this._gestureCleanup) {
      const fn = this._gestureCleanup;
      this._gestureCleanup = null;
      try {
        fn();
      } catch {
        // Ignore teardown errors on gesture cleanup.
      }
    }
  }

  _markInterruptionRecovered() {
    if (!this._pendingAutoRecovery) {
      return false;
    }
    this._pendingAutoRecovery = false;
    this._autoRecoveryCount += 1;
    this._clearGestureListeners();
    this._forwardToAudioWorker(
      'setOutputState',
      {
        type: 'audio-context-state',
        setOutputState: true,
        outputState: 'active',
        recoveredFromInterruption: true,
      },
      ['active']
    );
    return true;
  }

  _armInterruptionRecovery(target = undefined, options = undefined) {
    if (this._destroyed || !this._context) {
      return () => {};
    }
    const recoveryTarget =
      target !== undefined ? target : this._lastGestureTarget || this._gestureTarget;
    const recoveryOptions =
      options !== undefined ? options : this._lastGestureOptions || {};
    try {
      return this.resumeOnUserGesture(recoveryTarget, recoveryOptions);
    } catch {
      return () => {};
    }
  }

  /**
   * Resumes a suspended AudioContext (typically triggered by a user gesture).
   * Optionally retries failed attempts when `options.maxRetries > 0`.
   *
   * @param {object} [options]
   * @param {number} [options.maxRetries=0] Additional retry attempts after an initial failure.
   * @param {number} [options.retryDelayMs=0] Delay in ms between retry attempts.
   * @returns {Promise<boolean>} True if context is running after resume.
   */
  resume(options = {}) {
    if (this._destroyed || !this._context) {
      return Promise.resolve(false);
    }
    const currentState = this._readContextState();
    if (currentState === 'running') {
      return Promise.resolve(true);
    }
    if (currentState === 'closed') {
      return Promise.resolve(false);
    }
    if (this._resumePromise) {
      return this._resumePromise;
    }

    const opts = options && typeof options === 'object' ? options : {};
    const maxRetries = sanitizeFiniteNonNegInt(opts.maxRetries, 0);
    const retryDelayMs = sanitizeFiniteNonNegInt(opts.retryDelayMs, 0);

    const isManual = !this._inAutoRecovery;
    if (isManual) {
      this._inManualResume = true;
    }
    const promise = this._resumeInternal(maxRetries, retryDelayMs).finally(() => {
      if (isManual) {
        this._inManualResume = false;
      }
      if (this._resumePromise === promise) {
        this._resumePromise = null;
      }
    });
    this._resumePromise = promise;
    return promise;
  }

  async _resumeInternal(maxRetries, retryDelayMs) {
    const totalAttempts = 1 + maxRetries;

    for (let attempt = 0; attempt < totalAttempts; attempt++) {
      if (this._destroyed || !this._context) {
        return false;
      }
      const preState = this._readContextState();
      if (preState === 'running') {
        return true;
      }
      if (preState === 'closed') {
        return false;
      }

      this._resumeAttempts += 1;

      if (typeof this._context.resume !== 'function') {
        const msg = 'AudioContext.resume is unavailable';
        this._resumeFailures += 1;
        this._lastResumeError = sanitizeResumeError(msg);
        this._error = new Error(msg);
        if (this._onError) {
          try {
            this._onError(this._error);
          } catch {
            // Ignore.
          }
        }
        return false;
      }

      try {
        await this._context.resume();
        if (this._destroyed || !this._context) {
          this._resumeFailures += 1;
          this._lastResumeError = sanitizeResumeError('AudioEngine was destroyed during resume');
          return false;
        }

        const postRawState = this._readContextState();
        const nextState = sanitizeOutputState(postRawState || 'running');
        this._status = nextState;
        this._emitStateChange();

        if (postRawState === 'running') {
          this._resumeSuccesses += 1;
          this._lastResumeError = null;
          this._error = null;
          this._wasPlaying = true;
          this._explicitlySuspended = false;
          if (this._pendingAutoRecovery) {
            if (this._inManualResume && !this._inAutoRecovery) {
              this._pendingAutoRecovery = false;
              this._clearGestureListeners();
            } else {
              this._markInterruptionRecovered();
            }
          }
          if (
            this._masterGainNode &&
            this._masterGainHeldSilent &&
            !this._inUnlockGesture &&
            !this._inAutoRecovery
          ) {
            this._masterGainHeldSilent = false;
            this._scheduleMasterGainRamp(
              this._muted ? 0.0 : this._volume,
              this._unlockFadeInMs,
              0.0
            );
          }
          return true;
        }

        this._resumeFailures += 1;
        this._lastResumeError = sanitizeResumeError(
          `AudioContext remained in state "${nextState}" after resume`
        );
      } catch (err) {
        this._resumeFailures += 1;
        const sanitized = sanitizeResumeError(err);
        this._lastResumeError = sanitized;
        this._error =
          err instanceof Error ? err : new Error(sanitized || 'AudioContext resume failed');
        if (this._onError) {
          try {
            this._onError(this._error);
          } catch {
            // Ignore.
          }
        }
      }

      if (attempt + 1 < totalAttempts && !this._destroyed && this._context) {
        if (retryDelayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
        }
      }
    }

    return false;
  }

  /**
   * Explicitly recovers an interrupted or unexpectedly suspended `AudioContext`,
   * updating `autoRecoveryCount`, fading master gain back in cleanly without
   * clicks, and arming gesture fallback if autoplay still blocks immediate resume.
   * Never throws or produces unhandled promise rejections.
   *
   * @param {object} [options]
   * @param {number} [options.maxRetries=0]
   * @param {number} [options.retryDelayMs=0]
   * @param {number} [options.fadeInMs]
   * @param {EventTarget} [options.gestureTarget]
   * @param {boolean} [options.armGestureFallback=true]
   * @returns {Promise<{ ok: boolean, recovered: boolean, alreadyRunning: boolean, state: string, interruptionCount: number, autoRecoveryCount: number, reason: string|null, error?: string|null }>}
   */
  async recoverFromInterruption(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};

    if (this._destroyed) {
      return {
        ok: false,
        recovered: false,
        alreadyRunning: false,
        state: 'closed',
        interruptionCount: this._interruptionCount,
        autoRecoveryCount: this._autoRecoveryCount,
        reason: 'closed',
      };
    }

    if (!this._context) {
      return {
        ok: false,
        recovered: false,
        alreadyRunning: false,
        state: this.state,
        interruptionCount: this._interruptionCount,
        autoRecoveryCount: this._autoRecoveryCount,
        reason: 'not-started',
      };
    }

    const currentState = this._readContextState();
    if (currentState === 'closed') {
      return {
        ok: false,
        recovered: false,
        alreadyRunning: false,
        state: 'closed',
        interruptionCount: this._interruptionCount,
        autoRecoveryCount: this._autoRecoveryCount,
        reason: 'closed',
      };
    }

    if (currentState === 'running') {
      const recoveredPending = this._markInterruptionRecovered();
      return {
        ok: true,
        recovered: recoveredPending,
        alreadyRunning: true,
        state: 'running',
        interruptionCount: this._interruptionCount,
        autoRecoveryCount: this._autoRecoveryCount,
        reason: null,
      };
    }

    if (currentState === 'interrupted' || currentState === 'suspended') {
      if (!this._pendingAutoRecovery) {
        if (currentState === 'interrupted') {
          this._interruptionCount += 1;
        }
        this._pendingAutoRecovery = true;
      }
      this._explicitlySuspended = false;
    }

    const fadeInMs = sanitizeRampMs(
      opts.fadeInMs !== undefined ? opts.fadeInMs : opts.rampMs,
      this._unlockFadeInMs
    );
    if (this._masterGainNode) {
      this._scheduleMasterGainRamp(0.0, 0, 0.0);
      this._masterGainHeldSilent = true;
    }

    const prevInAutoRecovery = this._inAutoRecovery;
    this._inAutoRecovery = true;
    let resumed = false;
    try {
      resumed = await this.resume({
        maxRetries: opts.maxRetries,
        retryDelayMs: opts.retryDelayMs,
      });
    } catch {
      resumed = false;
    } finally {
      this._inAutoRecovery = prevInAutoRecovery;
    }

    if (!resumed || this._destroyed || !this._context) {
      if (!this._destroyed && this._context && opts.armGestureFallback !== false) {
        this._armInterruptionRecovery(opts.gestureTarget);
      }
      return {
        ok: false,
        recovered: false,
        alreadyRunning: false,
        state: this.state,
        interruptionCount: this._interruptionCount,
        autoRecoveryCount: this._autoRecoveryCount,
        reason: this._destroyed ? 'closed' : 'resume-failed',
        error: this._lastResumeError,
      };
    }

    this._markInterruptionRecovered();
    if (this._masterGainNode && this._masterGainHeldSilent) {
      this._masterGainHeldSilent = false;
      this._scheduleMasterGainRamp(this._muted ? 0.0 : this._volume, fadeInMs, 0.0);
    }

    return {
      ok: true,
      recovered: true,
      alreadyRunning: false,
      state: 'running',
      interruptionCount: this._interruptionCount,
      autoRecoveryCount: this._autoRecoveryCount,
      reason: null,
    };
  }

  /**
   * Safely suspends a running AudioContext without throwing in headless,
   * unsupported, or closed states.
   *
   * @returns {Promise<boolean>} True if context is suspended after call.
   */
  async suspend() {
    if (this._destroyed || !this._context) {
      return false;
    }
    const currentState = this._readContextState();
    if (currentState === 'suspended') {
      this._explicitlySuspended = true;
      return true;
    }
    if (currentState === 'closed' || typeof this._context.suspend !== 'function') {
      return false;
    }
    this._explicitlySuspended = true;
    try {
      await this._context.suspend();
      if (this._destroyed || !this._context) {
        return false;
      }
      const nextState = sanitizeOutputState(this._readContextState() || 'suspended');
      this._status = nextState;
      if (nextState !== 'suspended') {
        this._explicitlySuspended = false;
      }
      this._emitStateChange();
      return nextState === 'suspended';
    } catch (err) {
      this._explicitlySuspended = false;
      this._error =
        err instanceof Error
          ? err
          : new Error(sanitizeResumeError(err) || 'AudioContext suspend failed');
      if (this._onError) {
        try {
          this._onError(this._error);
        } catch {
          // Ignore.
        }
      }
      return false;
    }
  }

  /**
   * Attaches user-gesture listeners (`pointerdown`, `keydown`, `touchstart`, `click`
   * by default) to `target` to unlock a suspended AudioContext on user gesture.
   * Safe in headless/autoplay-restricted browsers: never throws when `window` or
   * `target` is absent or when `resume()` fails. Cleans up all listeners once
   * running (or when `destroy()`/`close()` is called). If a gesture resume attempt
   * fails, listeners automatically re-arm up to `options.maxRetries` times.
   *
   * @param {EventTarget} [target] Defaults to `window` / `document` / `globalThis` when available.
   * @param {object} [options]
   * @param {number} [options.maxRetries] Maximum gesture retry re-arms after an initial failure.
   * @param {string[]} [options.events] Custom gesture event names.
   * @returns {() => void} Manual cleanup function.
   */
  resumeOnUserGesture(target, options = {}) {
    let resolvedTarget = target || null;
    if (!resolvedTarget) {
      try {
        if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
          resolvedTarget = window;
        } else if (
          typeof document !== 'undefined' &&
          typeof document.addEventListener === 'function'
        ) {
          resolvedTarget = document;
        } else if (
          typeof globalThis !== 'undefined' &&
          typeof globalThis.addEventListener === 'function'
        ) {
          resolvedTarget = globalThis;
        }
      } catch {
        resolvedTarget = null;
      }
    }

    if (
      this._destroyed ||
      this._status === 'unsupported' ||
      !resolvedTarget ||
      typeof resolvedTarget.addEventListener !== 'function'
    ) {
      return () => {};
    }

    const currentCtxState = this._readContextState();
    if (this._context && (currentCtxState === 'running' || currentCtxState === 'closed')) {
      return () => {};
    }

    // Replace any previously armed unlock listeners to avoid leaks.
    if (this._gestureCleanup) {
      this._gestureCleanup();
      this._gestureCleanup = null;
    }

    const opts = options && typeof options === 'object' ? options : {};
    let retriesRemaining = sanitizeFiniteNonNegInt(opts.maxRetries, this._maxResumeRetries);
    let disposed = false;
    let activeCleanup = null;
    const rawEvents =
      Array.isArray(opts.events) && opts.events.length > 0 ? opts.events : DEFAULT_GESTURE_EVENTS;
    const events = [];
    for (let i = 0; i < rawEvents.length; i++) {
      const evt = rawEvents[i];
      if (typeof evt === 'string' && evt.length > 0 && !events.includes(evt)) {
        events.push(evt);
      }
    }
    if (events.length === 0) {
      return () => {};
    }

    this._lastGestureTarget = resolvedTarget;
    this._lastGestureOptions = {
      events: events.slice(),
      maxRetries: retriesRemaining,
    };

    const removeListeners = () => {
      if (activeCleanup) {
        const fn = activeCleanup;
        activeCleanup = null;
        fn();
      }
    };

    const cleanup = () => {
      if (disposed) {
        return;
      }
      disposed = true;
      removeListeners();
      if (this._gestureCleanup === cleanup) {
        this._gestureCleanup = null;
      }
    };

    const arm = () => {
      if (disposed || this._destroyed) {
        return;
      }

      const onGesture = () => {
        removeListeners();
        if (disposed || this._destroyed || this._status === 'unsupported') {
          cleanup();
          return;
        }
        if (!this._context) {
          if (this._status === 'idle' || this._status === 'initializing') {
            arm();
          } else {
            cleanup();
          }
          return;
        }
        const recovering = this._pendingAutoRecovery;
        const prevInAutoRecovery = this._inAutoRecovery;
        if (recovering) {
          this._inAutoRecovery = true;
        }
        let resumePromise;
        try {
          resumePromise = Promise.resolve(this.resume());
        } catch {
          resumePromise = Promise.resolve(false);
        }
        resumePromise
          .catch(() => false)
          .then((resumed) => {
            if (recovering) {
              this._inAutoRecovery = prevInAutoRecovery;
            }
            const postState = this._readContextState();
            if (resumed) {
              this._unlockedOnGesture = true;
              const markedRecovery = recovering ? this._markInterruptionRecovered() : false;
              if (this._masterGainNode && this._masterGainHeldSilent) {
                this._masterGainHeldSilent = false;
                this._scheduleMasterGainRamp(
                  this._muted ? 0.0 : this._volume,
                  this._unlockFadeInMs,
                  0.0
                );
              }
              if (!markedRecovery) {
                this._forwardToAudioWorker(
                  'setOutputState',
                  {
                    type: 'audio-context-state',
                    setOutputState: true,
                    outputState: 'active',
                    unlockedOnGesture: true,
                    ...(recovering ? { recoveredFromInterruption: true } : {}),
                  },
                  ['active']
                );
              }
            }
            if (
              !resumed &&
              !disposed &&
              !this._destroyed &&
              this._context &&
              postState !== 'running' &&
              postState !== 'closed' &&
              retriesRemaining > 0
            ) {
              retriesRemaining -= 1;
              arm();
            } else if (
              resumed ||
              this._destroyed ||
              postState === 'closed' ||
              retriesRemaining <= 0
            ) {
              cleanup();
            }
          });
      };

      const registered = [];
      for (const evt of events) {
        try {
          resolvedTarget.addEventListener(evt, onGesture, { once: true, passive: true });
          registered.push(evt);
        } catch {
          // Ignore unsupported/blocked event listener registrations in restricted hosts.
        }
      }

      activeCleanup = () => {
        if (typeof resolvedTarget.removeEventListener === 'function') {
          for (const evt of registered) {
            try {
              resolvedTarget.removeEventListener(evt, onGesture);
            } catch {
              // Ignore listener teardown failures.
            }
          }
        }
      };
    };

    arm();
    this._gestureCleanup = cleanup;
    return cleanup;
  }

  _scheduleMasterGainRamp(targetGain, rampMs = 0, fromGain = undefined) {
    const safeTarget = clampVolume(targetGain, 0.0);
    const safeRampMs = sanitizeRampMs(rampMs, 0);
    this._masterGainRampMs = safeRampMs;

    const gainNode = this._masterGainNode;
    if (!gainNode || !gainNode.gain) {
      return false;
    }

    const param = gainNode.gain;
    const now =
      this._context &&
      typeof this._context.currentTime === 'number' &&
      Number.isFinite(this._context.currentTime)
        ? this._context.currentTime
        : 0;
    const anchorGain =
      fromGain !== undefined
        ? clampVolume(fromGain, 0.0)
        : typeof param.value === 'number' && Number.isFinite(param.value)
          ? clampVolume(param.value, safeTarget)
          : safeTarget;

    try {
      if (typeof param.cancelScheduledValues === 'function') {
        param.cancelScheduledValues(now);
      }
      if (safeRampMs > 0 && typeof param.linearRampToValueAtTime === 'function') {
        if (typeof param.setValueAtTime === 'function') {
          param.setValueAtTime(anchorGain, now);
        } else {
          param.value = anchorGain;
        }
        param.linearRampToValueAtTime(safeTarget, now + safeRampMs / 1000);
      } else if (typeof param.setValueAtTime === 'function') {
        param.setValueAtTime(safeTarget, now);
      } else {
        param.value = safeTarget;
      }
      return true;
    } catch {
      return false;
    }
  }

  _forwardToAudioWorker(methodName, messagePayload, methodArgs = []) {
    const worker = this._audioWorker;
    if (!worker || typeof worker !== 'object') {
      return false;
    }
    try {
      if (methodName && typeof worker[methodName] === 'function') {
        worker[methodName](...methodArgs);
        return true;
      }
      if (typeof worker.postMessage === 'function') {
        worker.postMessage(messagePayload);
        return true;
      }
      return false;
    } catch (err) {
      if (this._onError) {
        try {
          this._onError(
            err instanceof Error
              ? err
              : new Error(sanitizeResumeError(err) || 'Audio worker forwarding failed')
          );
        } catch {
          // Ignore callback exceptions.
        }
      }
      return false;
    }
  }

  /**
   * Attaches a #1027 guest stereo PCM ring descriptor (`Playgta6AudioRing` /
   * `SharedArrayBuffer` / wasm linear memory ring), tracks attachment state
   * locally, and forwards cleanly to the audio worker / bridge when present.
   *
   * Never throws on invalid descriptors or destroyed engines.
   *
   * @param {object} ringDescriptor
   * @returns {{ ok: boolean, attached: boolean, forwarded: boolean, reason: string|null, ring: object|null }}
   */
  attachGuestPcmRing(ringDescriptor) {
    if (this._destroyed) {
      return {
        ok: false,
        attached: false,
        forwarded: false,
        reason: 'closed',
        ring: null,
      };
    }

    const parsed = normalizeGuestPcmRingDescriptor(
      ringDescriptor,
      this.sampleRate,
      this._channels || DEFAULT_CHANNELS
    );
    if (!parsed.ok || !parsed.ring) {
      return {
        ok: false,
        attached: this._guestRingAttached,
        forwarded: false,
        reason: parsed.reason || 'invalid-descriptor',
        ring: null,
      };
    }

    this._guestPcmRing = parsed.ring;
    this._guestRingAttached = true;

    const forwarded = this._forwardToAudioWorker(
      'attachGuestPcmRing',
      {
        type: 'attachGuestPcmRing',
        attachGuestPcmRing: true,
        ring: parsed.ring,
      },
      [parsed.ring]
    );

    return {
      ok: true,
      attached: true,
      forwarded,
      reason: null,
      ring: { ...parsed.ring },
    };
  }

  /**
   * Detaches the active guest stereo PCM ring locally and forwards detachment
   * cleanly to the audio worker / bridge when present. Idempotent and safe
   * to call when no ring is attached or after destroy.
   *
   * @returns {{ ok: boolean, attached: boolean, wasAttached: boolean, forwarded: boolean }}
   */
  detachGuestPcmRing() {
    const wasAttached = this._guestRingAttached;
    this._guestPcmRing = null;
    this._guestRingAttached = false;

    const forwarded = this._forwardToAudioWorker(
      'detachGuestPcmRing',
      {
        type: 'detachGuestPcmRing',
        detachGuestPcmRing: true,
      },
      []
    );

    return {
      ok: true,
      attached: false,
      wasAttached,
      forwarded,
    };
  }

  /**
   * Updates sanitized guest stereo PCM ring telemetry reported by the host
   * audio worker / bridge (`guestRingAttached`, `guestFramesDrained`,
   * `guestUnderruns`, `guestOverruns`, `guestPeakLevel`).
   * Non-finite, negative, or hostile inputs are safely ignored/clamped.
   *
   * @param {object} stats
   * @returns {{ guestRingAttached: boolean, guestFramesDrained: number, guestUnderruns: number, guestOverruns: number, guestPeakLevel: number, unlockedOnGesture: boolean, muted: boolean, masterGain: number }}
   */
  updateGuestAudioTelemetry(stats) {
    if (stats && typeof stats === 'object') {
      const src =
        stats.stats && typeof stats.stats === 'object' ? stats.stats : stats;

      try {
        if (typeof src.guestRingAttached === 'boolean') {
          this._guestRingAttached = src.guestRingAttached;
          if (!src.guestRingAttached) {
            this._guestPcmRing = null;
          }
        }
      } catch {
        // Ignore hostile getters.
      }

      try {
        if (src.guestFramesDrained !== undefined) {
          this._guestFramesDrained = sanitizeFiniteNonNegInt(
            src.guestFramesDrained,
            this._guestFramesDrained
          );
        }
      } catch {
        // Ignore hostile getters.
      }

      try {
        if (src.guestUnderruns !== undefined) {
          this._guestUnderruns = sanitizeFiniteNonNegInt(
            src.guestUnderruns,
            this._guestUnderruns
          );
        }
      } catch {
        // Ignore hostile getters.
      }

      try {
        const rawOverruns =
          src.guestOverruns !== undefined ? src.guestOverruns : src.guestOverflows;
        if (rawOverruns !== undefined) {
          this._guestOverruns = sanitizeFiniteNonNegInt(rawOverruns, this._guestOverruns);
        }
      } catch {
        // Ignore hostile getters.
      }

      try {
        if (src.guestPeakLevel !== undefined) {
          this._guestPeakLevel = clampPeakLevel(src.guestPeakLevel, this._guestPeakLevel);
        }
      } catch {
        // Ignore hostile getters.
      }
    }

    return {
      guestRingAttached: Boolean(this._guestRingAttached),
      guestFramesDrained: sanitizeFiniteNonNegInt(this._guestFramesDrained, 0),
      guestUnderruns: sanitizeFiniteNonNegInt(this._guestUnderruns, 0),
      guestOverruns: sanitizeFiniteNonNegInt(this._guestOverruns, 0),
      guestPeakLevel: clampPeakLevel(this._guestPeakLevel, 0),
      unlockedOnGesture: Boolean(this._unlockedOnGesture),
      muted: Boolean(this._muted),
      masterGain: clampVolume(this._volume, 1.0),
    };
  }

  /**
   * Unlocks a suspended/interrupted `AudioContext` from a user gesture and
   * smoothly fades master gain in from 0 to the active effective gain over
   * `rampMs` / `fadeInMs` so playback resumes without clicks or pops.
   *
   * Safe when not started, already running, muted, or closed.
   *
   * @param {number|object} [options] Ramp duration in ms or options `{ rampMs, fadeInMs, maxRetries, retryDelayMs }`.
   * @returns {Promise<{ ok: boolean, unlocked: boolean, alreadyRunning: boolean, state: string, fadeInMs: number, effectiveGain: number, reason?: string }>}
   */
  async unlockOnGesture(options = {}) {
    const opts =
      typeof options === 'number'
        ? { fadeInMs: options }
        : options && typeof options === 'object'
          ? options
          : {};
    const rawFadeMs =
      opts.fadeInMs !== undefined
        ? opts.fadeInMs
        : opts.rampMs !== undefined
          ? opts.rampMs
          : this._unlockFadeInMs;
    const fadeInMs = sanitizeRampMs(rawFadeMs, this._unlockFadeInMs);

    if (this._destroyed) {
      return {
        ok: false,
        unlocked: false,
        alreadyRunning: false,
        state: 'closed',
        fadeInMs,
        effectiveGain: 0,
        reason: 'closed',
      };
    }

    if (!this._context) {
      return {
        ok: false,
        unlocked: false,
        alreadyRunning: false,
        state: this.state,
        fadeInMs,
        effectiveGain: this._muted ? 0 : this._volume,
        reason: 'not-started',
      };
    }

    const currentState = this._readContextState();
    if (currentState === 'closed') {
      return {
        ok: false,
        unlocked: false,
        alreadyRunning: false,
        state: 'closed',
        fadeInMs,
        effectiveGain: 0,
        reason: 'closed',
      };
    }

    const effectiveTarget = this._muted ? 0.0 : this._volume;
    if (currentState === 'running') {
      this._unlockedOnGesture = true;
      if (this._masterGainNode && this._masterGainHeldSilent) {
        this._masterGainHeldSilent = false;
        this._scheduleMasterGainRamp(effectiveTarget, fadeInMs, 0.0);
      }
      if (this._gestureCleanup) {
        try {
          this._gestureCleanup();
        } catch {
          // Ignore cleanup errors.
        }
        this._gestureCleanup = null;
      }
      return {
        ok: true,
        unlocked: true,
        alreadyRunning: true,
        state: 'running',
        fadeInMs: 0,
        effectiveGain: effectiveTarget,
      };
    }

    // Anchor output at 0 before resuming so the audio graph never pops on unlock.
    if (this._masterGainNode) {
      this._scheduleMasterGainRamp(0.0, 0, 0.0);
      this._masterGainHeldSilent = true;
    } else if (
      !this._muted &&
      this._volume > 0 &&
      this._node &&
      this._node.port &&
      typeof this._node.port.postMessage === 'function'
    ) {
      try {
        this._node.port.postMessage({ type: 'setGain', gain: 0.0, immediate: true });
      } catch {
        // Ignore postMessage errors.
      }
    }

    this._inUnlockGesture = true;
    let resumed = false;
    try {
      resumed = await this.resume({
        maxRetries: opts.maxRetries,
        retryDelayMs: opts.retryDelayMs,
      });
    } finally {
      this._inUnlockGesture = false;
    }

    if (!resumed || this._destroyed || !this._context) {
      return {
        ok: false,
        unlocked: false,
        alreadyRunning: false,
        state: this.state,
        fadeInMs,
        effectiveGain: 0,
        reason: this._destroyed ? 'closed' : 'resume-failed',
      };
    }

    this._unlockedOnGesture = true;
    const postEffective = this._muted ? 0.0 : this._volume;
    if (this._masterGainNode) {
      this._masterGainHeldSilent = false;
      this._scheduleMasterGainRamp(postEffective, fadeInMs, 0.0);
    } else if (
      !this._muted &&
      this._volume > 0 &&
      this._node &&
      this._node.port &&
      typeof this._node.port.postMessage === 'function'
    ) {
      try {
        const msg = { type: 'setGain', gain: this._volume, rampMs: fadeInMs };
        if (fadeInMs === 0) {
          msg.immediate = true;
        } else {
          msg.rampFrames = Math.max(1, Math.round((fadeInMs / 1000) * this.sampleRate));
        }
        this._node.port.postMessage(msg);
      } catch {
        // Ignore postMessage errors.
      }
    }

    if (this._gestureCleanup) {
      try {
        this._gestureCleanup();
      } catch {
        // Ignore cleanup errors.
      }
      this._gestureCleanup = null;
    }

    this._forwardToAudioWorker(
      'setOutputState',
      {
        type: 'audio-context-state',
        setOutputState: true,
        outputState: 'active',
        unlockedOnGesture: true,
      },
      ['active']
    );

    return {
      ok: true,
      unlocked: true,
      alreadyRunning: false,
      state: 'running',
      fadeInMs,
      effectiveGain: postEffective,
    };
  }

  /**
   * Sets master output volume clamped to [0, 1] and forwards to the worklet.
   * Rejects non-finite (`NaN`, `±Infinity`) and non-number inputs without
   * mutating the active gain.
   *
   * @param {number} volume
   * @returns {number} Active sanitized volume in [0, 1].
   */
  setVolume(volume) {
    if (typeof volume !== 'number' || !Number.isFinite(volume)) {
      this._invalidGainInputs += 1;
      return this._volume;
    }

    const clamped = clampVolume(volume, this._volume);
    if (clamped !== volume || Object.is(volume, -0)) {
      this._gainClamps += 1;
    }

    this._volume = clamped;
    if (this._masterGainNode) {
      this._masterGainHeldSilent = false;
      this._scheduleMasterGainRamp(this._muted ? 0.0 : this._volume, 0);
    } else if (this._node && this._node.port && typeof this._node.port.postMessage === 'function') {
      try {
        this._node.port.postMessage({ type: 'setGain', gain: this._volume });
      } catch {
        // Ignore postMessage errors on closed/detached port.
      }
    }
    this._forwardToAudioWorker(
      'setMasterGain',
      {
        type: 'setMasterGain',
        setVolume: true,
        gain: this._volume,
        volume: this._volume,
      },
      [this._volume, 0]
    );
    return this._volume;
  }

  /**
   * Sets master gain clamped to [0, 1] with a smooth ramp (`rampMs`, default
   * `DEFAULT_GAIN_RAMP_MS`) via the WebAudio master `GainNode` when present,
   * or forwards ramped gain metadata to the worklet and audio worker.
   * Rejects non-finite (`NaN`, `±Infinity`) and non-number gain inputs without
   * mutating the active gain.
   *
   * @param {number} gain
   * @param {number} [rampMs=DEFAULT_GAIN_RAMP_MS]
   * @returns {number} Active sanitized master gain in [0, 1].
   */
  setMasterGain(gain, rampMs = DEFAULT_GAIN_RAMP_MS) {
    if (typeof gain !== 'number' || !Number.isFinite(gain)) {
      this._invalidGainInputs += 1;
      return this._volume;
    }

    const clamped = clampVolume(gain, this._volume);
    if (clamped !== gain || Object.is(gain, -0)) {
      this._gainClamps += 1;
    }

    const safeRampMs = sanitizeRampMs(rampMs, DEFAULT_GAIN_RAMP_MS);
    this._volume = clamped;
    this._masterGainRampMs = safeRampMs;

    if (this._masterGainNode) {
      this._masterGainHeldSilent = false;
      this._scheduleMasterGainRamp(this._muted ? 0.0 : this._volume, safeRampMs);
    } else if (this._node && this._node.port && typeof this._node.port.postMessage === 'function') {
      try {
        const msg = { type: 'setGain', gain: this._volume, rampMs: safeRampMs };
        if (safeRampMs === 0) {
          msg.immediate = true;
        } else {
          msg.rampFrames = Math.max(1, Math.round((safeRampMs / 1000) * this.sampleRate));
        }
        this._node.port.postMessage(msg);
      } catch {
        // Ignore postMessage errors on closed/detached port.
      }
    }

    this._forwardToAudioWorker(
      'setMasterGain',
      {
        type: 'setMasterGain',
        setVolume: true,
        gain: this._volume,
        volume: this._volume,
        rampMs: safeRampMs,
      },
      [this._volume, safeRampMs]
    );

    return this._volume;
  }

  /**
   * Alias for `setVolume(gain)` with identical `[0, 1]` finite clamping guardrails.
   *
   * @param {number} gain
   * @returns {number} Active sanitized gain in [0, 1].
   */
  setGain(gain) {
    return this.setVolume(gain);
  }

  /**
   * Sets mute state (strictly coerced to boolean) and forwards to the master
   * GainNode / worklet / audio worker. When `rampMs` is provided, schedules a
   * smooth linear ramp to avoid mute/unmute clicks.
   *
   * @param {boolean} muted
   * @param {number} [rampMs]
   * @returns {boolean}
   */
  setMuted(muted, rampMs = undefined) {
    this._muted = Boolean(muted);
    const hasExplicitRamp = rampMs !== undefined;
    const safeRampMs = hasExplicitRamp ? sanitizeRampMs(rampMs, DEFAULT_GAIN_RAMP_MS) : 0;

    if (this._masterGainNode) {
      this._masterGainHeldSilent = false;
      this._scheduleMasterGainRamp(
        this._muted ? 0.0 : this._volume,
        hasExplicitRamp ? safeRampMs : DEFAULT_GAIN_RAMP_MS
      );
    } else if (this._node && this._node.port && typeof this._node.port.postMessage === 'function') {
      try {
        if (!hasExplicitRamp) {
          this._node.port.postMessage({ type: 'setMuted', muted: this._muted });
        } else {
          const msg = { type: 'setMuted', muted: this._muted, rampMs: safeRampMs };
          if (safeRampMs === 0) {
            msg.immediate = true;
          } else {
            msg.rampFrames = Math.max(1, Math.round((safeRampMs / 1000) * this.sampleRate));
          }
          this._node.port.postMessage(msg);
        }
      } catch {
        // Ignore postMessage errors on closed/detached port.
      }
    }

    this._forwardToAudioWorker(
      'setMuted',
      hasExplicitRamp
        ? { type: 'setMuted', setMute: true, muted: this._muted, rampMs: safeRampMs }
        : { type: 'setMuted', setMute: true, muted: this._muted },
      hasExplicitRamp ? [this._muted, safeRampMs] : [this._muted]
    );

    return this._muted;
  }

  /**
   * Convenience helper to mute audio output.
   * @returns {boolean} Always `true`.
   */
  mute() {
    return this.setMuted(true);
  }

  /**
   * Convenience helper to unmute audio output.
   * @returns {boolean} Always `false`.
   */
  unmute() {
    return this.setMuted(false);
  }

  /**
   * Toggles mute state and returns the updated muted flag.
   * @returns {boolean}
   */
  toggleMute() {
    return this.setMuted(!this._muted);
  }

  /**
   * Sets the bounded low-latency jitter-buffer target in milliseconds,
   * clamped to `[MIN_TARGET_LATENCY_MS, MAX_TARGET_LATENCY_MS]` (`[10, 120]`,
   * default `30`), and forwards a control message to `audioWorker` when attached.
   * Non-finite / non-number inputs are ignored without mutating the active target.
   *
   * @param {number} ms
   * @returns {number} Active sanitized target latency in ms.
   */
  setTargetLatencyMs(ms) {
    if (typeof ms !== 'number' || !Number.isFinite(ms)) {
      return this._targetLatencyMs;
    }

    const clamped = clampTargetLatencyMs(ms, this._targetLatencyMs);
    this._targetLatencyMs = clamped;
    const sampleRate = sanitizePositiveFiniteNumber(
      this.sampleRate,
      this._requestedSampleRate
    );
    const targetLatencyFrames = Math.max(1, Math.round((clamped / 1000) * sampleRate));

    this._forwardToAudioWorker(
      'setTargetLatencyMs',
      {
        type: 'setTargetLatencyMs',
        setTargetLatencyMs: true,
        targetLatencyMs: clamped,
        targetLatencyFrames,
        sampleRate,
      },
      [clamped]
    );

    return this._targetLatencyMs;
  }

  /**
   * Persists sanitized `{ muted, volume }` preferences for this engine under
   * `'playgta6:audio:prefs:v1'`, gracefully catching `SecurityError` or
   * unavailable storage.
   *
   * @param {Storage|null} [storage]
   * @returns {{ ok: boolean, saved: boolean, prefs: { muted: boolean, volume: number }|null, key: string, reason: string|null, error?: string|null }}
   */
  saveAudioPreferences(storage = undefined) {
    return saveAudioPreferences(
      { muted: this._muted, volume: this._volume },
      storage
    );
  }

  /**
   * Loads sanitized `{ muted, volume }` preferences from `'playgta6:audio:prefs:v1'`
   * and applies them to this engine unless `options.apply === false`.
   * Gracefully catches `SecurityError`, corrupted JSON, and unavailable storage.
   *
   * @param {Storage|null} [storage]
   * @param {{ apply?: boolean }} [options]
   * @returns {{ ok: boolean, loaded: boolean, applied: boolean, prefs: { muted: boolean, volume: number }|null, key: string, reason: string|null, error?: string|null }}
   */
  loadAudioPreferences(storage = undefined, options = {}) {
    const res = loadAudioPreferences(storage);
    if (!res.loaded || !res.prefs) {
      const out = {
        ok: res.ok,
        loaded: false,
        applied: false,
        prefs: null,
        key: AUDIO_PREFS_STORAGE_KEY,
        reason: res.reason,
      };
      if (res.error !== undefined) {
        out.error = res.error;
      }
      return out;
    }

    const opts = options && typeof options === 'object' ? options : {};
    const shouldApply = opts.apply !== false;
    const nextPrefs = {
      muted: res.hasMuted ? res.prefs.muted : this._muted,
      volume: res.hasVolume ? res.prefs.volume : this._volume,
    };

    if (shouldApply) {
      if (res.hasVolume) {
        this.setVolume(nextPrefs.volume);
      }
      if (res.hasMuted) {
        this.setMuted(nextPrefs.muted);
      }
    }

    return {
      ok: true,
      loaded: true,
      applied: shouldApply,
      prefs: nextPrefs,
      key: AUDIO_PREFS_STORAGE_KEY,
      reason: null,
    };
  }

  /**
   * Returns the number of currently queued PCM frames in the bounded ring buffer.
   * @returns {number}
   */
  availableFrames() {
    return this._computeBufferedFrames();
  }

  /**
   * Returns the number of free frame slots remaining in the bounded ring buffer.
   * @returns {number}
   */
  freeFrames() {
    const limit = Math.min(this._maxQueuedFrames, this._capacityFrames);
    if (limit <= 0) {
      return 0;
    }
    return Math.max(0, limit - this._computeBufferedFrames());
  }

  /**
   * Submits interleaved Float32 PCM frames into the bounded audio ring buffer
   * with strict overflow/overrun guards and non-finite sample sanitization.
   * Never throws on invalid inputs or closed engines.
   *
   * @param {Float32Array|ArrayBuffer|SharedArrayBuffer|ArrayBufferView|number[]} samples
   * @param {object} [options]
   * @param {number} [options.frameCount] Optional frame count cap within `samples`.
   * @param {'drop-newest'|'drop-oldest'} [options.overflowPolicy='drop-newest']
   * @returns {{ ok: boolean, acceptedFrames: number, submittedFrames: number, droppedFrames: number, overrun: boolean, bufferedFrames: number, capacityFrames: number, freeFrames: number, reason?: string }}
   */
  submitPcm(samples, options = {}) {
    const cap = this._capacityFrames;
    const channels = this._channels;
    const limit = Math.min(this._maxQueuedFrames, cap);

    if (this._destroyed) {
      return {
        ok: false,
        acceptedFrames: 0,
        submittedFrames: 0,
        droppedFrames: 0,
        overrun: false,
        bufferedFrames: this._computeBufferedFrames(),
        capacityFrames: cap,
        freeFrames: this.freeFrames(),
        reason: 'closed',
      };
    }

    const ctrl = this._activeCtrl();
    const ringSamples = this._activeSamples();
    if (!ctrl || !ringSamples || cap <= 0 || channels <= 0) {
      return {
        ok: false,
        acceptedFrames: 0,
        submittedFrames: 0,
        droppedFrames: 0,
        overrun: false,
        bufferedFrames: 0,
        capacityFrames: cap,
        freeFrames: 0,
        reason: 'uninitialized',
      };
    }

    const pcm = extractInterleavedFloat32(samples);
    if (!pcm) {
      return {
        ok: false,
        acceptedFrames: 0,
        submittedFrames: 0,
        droppedFrames: 0,
        overrun: false,
        bufferedFrames: this._computeBufferedFrames(),
        capacityFrames: cap,
        freeFrames: this.freeFrames(),
        reason: 'invalid-input',
      };
    }

    const opts = options && typeof options === 'object' ? options : {};
    const availableInputFrames = Math.floor(pcm.length / channels);
    const requestedFrameCount =
      Number.isInteger(opts.frameCount) && opts.frameCount >= 0
        ? Math.min(opts.frameCount, availableInputFrames)
        : availableInputFrames;

    if (requestedFrameCount <= 0) {
      const buffered = this._computeBufferedFrames();
      return {
        ok: true,
        acceptedFrames: 0,
        submittedFrames: 0,
        droppedFrames: 0,
        overrun: false,
        bufferedFrames: buffered,
        capacityFrames: cap,
        freeFrames: Math.max(0, limit - buffered),
      };
    }

    const modulus = cap * 2;
    const write = this._loadHdr(HDR_WRITE_CURSOR);
    let read = this._loadHdr(HDR_READ_CURSOR);
    let used = write - read;
    if (used < 0) {
      used += modulus;
    }
    if (!Number.isFinite(used) || used < 0) {
      used = 0;
    } else if (used > cap) {
      used = cap;
    }

    const policy = opts.overflowPolicy === 'drop-oldest' ? 'drop-oldest' : 'drop-newest';
    let free = Math.max(0, limit - used);
    let srcFrameOffset = 0;
    let framesToWrite = requestedFrameCount;
    let dropped = 0;

    if (policy === 'drop-oldest' && requestedFrameCount > free) {
      const overflow = requestedFrameCount - free;
      const discardQueued = Math.min(used, overflow);
      if (discardQueued > 0) {
        read = (read + discardQueued) % modulus;
        this._storeHdr(HDR_READ_CURSOR, read);
        used -= discardQueued;
        free = Math.max(0, limit - used);
      }
      const skipIncoming = Math.max(0, requestedFrameCount - free);
      srcFrameOffset = skipIncoming;
      framesToWrite = requestedFrameCount - skipIncoming;
      dropped = discardQueued + skipIncoming;
    } else {
      framesToWrite = requestedFrameCount <= free ? requestedFrameCount : free;
      dropped = requestedFrameCount - framesToWrite;
    }

    if (framesToWrite > 0) {
      const writeSlot = write >= cap ? write - cap : write;
      const firstFrames = Math.min(framesToWrite, cap - writeSlot);
      const secondFrames = framesToWrite - firstFrames;
      const srcBase = srcFrameOffset * channels;
      const firstSampleCount = firstFrames * channels;
      const dstFirstBase = writeSlot * channels;

      let sanitized = 0;
      for (let i = 0; i < firstSampleCount; i++) {
        let s = pcm[srcBase + i];
        if (!Number.isFinite(s)) {
          s = 0.0;
          sanitized += 1;
        }
        ringSamples[dstFirstBase + i] = s;
      }

      if (secondFrames > 0) {
        const secondSampleCount = secondFrames * channels;
        const srcSecondBase = srcBase + firstSampleCount;
        for (let i = 0; i < secondSampleCount; i++) {
          let s = pcm[srcSecondBase + i];
          if (!Number.isFinite(s)) {
            s = 0.0;
            sanitized += 1;
          }
          ringSamples[i] = s;
        }
      }

      if (sanitized > 0) {
        this._sanitizedSamples += sanitized;
      }

      let nextWrite = write + framesToWrite;
      if (nextWrite >= modulus) {
        nextWrite -= modulus;
      }
      this._storeHdr(HDR_WRITE_CURSOR, nextWrite);
      this._submittedFrames += framesToWrite;
      this._submittedBatches += 1;
    }

    if (dropped > 0) {
      this._addHdr(HDR_OVERFLOWS, 1);
      this._addHdr(HDR_DROPPED_FRAMES, dropped);
    }

    const nextBuffered = Math.min(cap, used + framesToWrite);
    return {
      ok: true,
      acceptedFrames: framesToWrite,
      submittedFrames: framesToWrite,
      droppedFrames: dropped,
      overrun: dropped > 0,
      bufferedFrames: nextBuffered,
      capacityFrames: cap,
      freeFrames: Math.max(0, limit - nextBuffered),
    };
  }

  /**
   * Pulls interleaved PCM frames from the bounded ring buffer with underrun
   * zero-fill guard and atomic underrun telemetry accounting.
   *
   * Accepts either a destination `Float32Array` (filled in place, zero-padded on
   * underrun) or a non-negative integer frame count (drains without allocating).
   *
   * @param {Float32Array|number} outputOrFrames
   * @param {object} [options]
   * @param {boolean} [options.applyGain=true]
   * @returns {{ ok: boolean, framesRead: number, requestedFrames: number, underrun: boolean, underrunFrames: number, bufferedFrames: number }}
   */
  pullPcm(outputOrFrames, options = {}) {
    const cap = this._capacityFrames;
    const channels = this._channels;
    const ctrl = this._activeCtrl();
    const ringSamples = this._activeSamples();

    let output = null;
    let requestedFrames = 0;

    if (outputOrFrames instanceof Float32Array) {
      if (
        !outputOrFrames.buffer ||
        (!isSharedArrayBufferValue(outputOrFrames.buffer) &&
          isDetachedArrayBuffer(outputOrFrames.buffer))
      ) {
        return {
          ok: false,
          framesRead: 0,
          requestedFrames: 0,
          underrun: false,
          underrunFrames: 0,
          bufferedFrames: this._computeBufferedFrames(),
        };
      }
      output = outputOrFrames;
      requestedFrames = channels > 0 ? Math.floor(output.length / channels) : 0;
    } else if (typeof outputOrFrames === 'number' && Number.isFinite(outputOrFrames)) {
      requestedFrames = Math.max(0, Math.floor(outputOrFrames));
    } else {
      return {
        ok: false,
        framesRead: 0,
        requestedFrames: 0,
        underrun: false,
        underrunFrames: 0,
        bufferedFrames: this._computeBufferedFrames(),
      };
    }

    if (!ctrl || !ringSamples || cap <= 0 || channels <= 0 || requestedFrames <= 0) {
      if (output && requestedFrames > 0) {
        output.fill(0, 0, requestedFrames * channels);
      }
      return {
        ok: Boolean(ctrl && ringSamples && cap > 0 && channels > 0),
        framesRead: 0,
        requestedFrames,
        underrun: false,
        underrunFrames: 0,
        bufferedFrames: this._computeBufferedFrames(),
      };
    }

    const opts = options && typeof options === 'object' ? options : {};
    const applyGain = opts.applyGain !== false;
    const effectiveGain = applyGain ? (this._muted ? 0.0 : clampVolume(this._volume, 1.0)) : 1.0;

    const modulus = cap * 2;
    const read = this._loadHdr(HDR_READ_CURSOR);
    const write = this._loadHdr(HDR_WRITE_CURSOR);
    let used = write - read;
    if (used < 0) {
      used += modulus;
    }
    if (!Number.isFinite(used) || used < 0) {
      used = 0;
    } else if (used > cap) {
      used = cap;
    }

    const framesToRead = requestedFrames <= used ? requestedFrames : used;
    const missingFrames = requestedFrames - framesToRead;

    if (framesToRead > 0) {
      if (output) {
        const readSlot = read >= cap ? read - cap : read;
        const firstFrames = Math.min(framesToRead, cap - readSlot);
        const secondFrames = framesToRead - firstFrames;
        const firstSampleCount = firstFrames * channels;
        const srcFirstBase = readSlot * channels;

        if (effectiveGain === 0.0) {
          output.fill(0, 0, framesToRead * channels);
        } else {
          let sanitized = 0;
          for (let i = 0; i < firstSampleCount; i++) {
            let s = ringSamples[srcFirstBase + i];
            if (!Number.isFinite(s)) {
              s = 0.0;
              sanitized += 1;
            }
            output[i] = effectiveGain === 1.0 ? s : s * effectiveGain;
          }
          if (secondFrames > 0) {
            const secondSampleCount = secondFrames * channels;
            for (let i = 0; i < secondSampleCount; i++) {
              let s = ringSamples[i];
              if (!Number.isFinite(s)) {
                s = 0.0;
                sanitized += 1;
              }
              output[firstSampleCount + i] = effectiveGain === 1.0 ? s : s * effectiveGain;
            }
          }
          if (sanitized > 0) {
            this._sanitizedSamples += sanitized;
          }
        }
      }

      let nextRead = read + framesToRead;
      if (nextRead >= modulus) {
        nextRead -= modulus;
      }
      this._storeHdr(HDR_READ_CURSOR, nextRead);
      this._consumedFrames += framesToRead;
    }

    if (missingFrames > 0) {
      if (output) {
        output.fill(0, framesToRead * channels, requestedFrames * channels);
      }
      this._addHdr(HDR_UNDERRUNS, 1);
      this._addHdr(HDR_UNDERRUN_FRAMES, missingFrames);
    }

    return {
      ok: true,
      framesRead: framesToRead,
      requestedFrames,
      underrun: missingFrames > 0,
      underrunFrames: missingFrames,
      bufferedFrames: Math.max(0, used - framesToRead),
    };
  }

  /**
   * Alias for `pullPcm(outputOrFrames, options)`.
   */
  consumePcm(outputOrFrames, options = {}) {
    return this.pullPcm(outputOrFrames, options);
  }

  /**
   * Resets queued PCM frames, ring cursors, and underrun/overrun/submit telemetry
   * counters cleanly without closing the AudioContext.
   *
   * @param {object} [options]
   * @param {boolean} [options.preserveStats=false] Keep telemetry counters while only flushing the ring.
   * @param {boolean} [options.resetGain=false] Reset volume to 1.0 and muted to false.
   * @param {boolean} [options.clearGestureListeners=false] Remove armed user-gesture unlock listeners.
   * @returns {ReturnType<AudioEngine['getAudioStats']>}
   */
  reset(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};

    const ctrl = this._activeCtrl();
    if (ctrl) {
      this._storeHdr(HDR_WRITE_CURSOR, 0);
      this._storeHdr(HDR_READ_CURSOR, 0);
      if (!opts.preserveStats) {
        this._storeHdr(HDR_UNDERRUNS, 0);
        this._storeHdr(HDR_UNDERRUN_FRAMES, 0);
        this._storeHdr(HDR_OVERFLOWS, 0);
        this._storeHdr(HDR_DROPPED_FRAMES, 0);
      }
    }

    const samples = this._activeSamples();
    if (samples) {
      samples.fill(0);
    }

    if (!opts.preserveStats) {
      this._submittedFrames = 0;
      this._submittedBatches = 0;
      this._consumedFrames = 0;
      this._sanitizedSamples = 0;
      this._resumeAttempts = 0;
      this._resumeSuccesses = 0;
      this._resumeFailures = 0;
      this._interruptionCount = 0;
      this._autoRecoveryCount = 0;
      this._lastResumeError = null;
      this._error = null;
      this._gainClamps = 0;
      this._invalidGainInputs = 0;
      this._guestFramesDrained = 0;
      this._guestUnderruns = 0;
      this._guestOverruns = 0;
      this._guestPeakLevel = 0;
    }

    if (opts.detachGuestRing) {
      this.detachGuestPcmRing();
    }

    if (opts.resetGain) {
      this.setVolume(1.0);
      this.setMuted(false);
    }

    if (opts.clearGestureListeners && this._gestureCleanup) {
      this._gestureCleanup();
      this._gestureCleanup = null;
    }

    return this.getAudioStats();
  }

  /**
   * Reads live WebAudio + SAB ring buffer telemetry without blocking or
   * round-tripping to the audio thread.
   */
  getTelemetry() {
    let bufferedFrames = 0;
    let underruns = 0;
    let underrunFrames = 0;
    let overflows = 0;
    let droppedFrames = 0;

    const ctrl = this._activeCtrl();
    if (ctrl && this._capacityFrames > 0) {
      bufferedFrames = this._computeBufferedFrames();
      underruns = sanitizeFiniteNonNegInt(this._loadHdr(HDR_UNDERRUNS), 0);
      underrunFrames = sanitizeFiniteNonNegInt(this._loadHdr(HDR_UNDERRUN_FRAMES), 0);
      overflows = sanitizeFiniteNonNegInt(this._loadHdr(HDR_OVERFLOWS), 0);
      droppedFrames = sanitizeFiniteNonNegInt(this._loadHdr(HDR_DROPPED_FRAMES), 0);
    }

    const capacityFrames = sanitizeFiniteNonNegInt(
      this._capacityFrames,
      this._requestedCapacityFrames
    );
    const limit = Math.min(this._maxQueuedFrames, capacityFrames);
    const freeFrames = Math.max(0, limit - bufferedFrames);
    const outputState = sanitizeOutputState(this.state);
    const gain = clampVolume(this._volume, 1.0);
    const muted = Boolean(this._muted);
    const effectiveGain = muted ? 0.0 : gain;
    const guestRingAttached = Boolean(this._guestRingAttached);
    const guestFramesDrained = sanitizeFiniteNonNegInt(this._guestFramesDrained, 0);
    const guestUnderruns = sanitizeFiniteNonNegInt(this._guestUnderruns, 0);
    const guestOverruns = sanitizeFiniteNonNegInt(this._guestOverruns, 0);
    const guestPeakLevel = clampPeakLevel(this._guestPeakLevel, 0);
    const unlockedOnGesture = Boolean(this._unlockedOnGesture);
    const targetLatencyMs = clampTargetLatencyMs(
      this._targetLatencyMs,
      DEFAULT_TARGET_LATENCY_MS
    );
    const interruptionCount = sanitizeFiniteNonNegInt(this._interruptionCount, 0);
    const autoRecoveryCount = sanitizeFiniteNonNegInt(this._autoRecoveryCount, 0);
    const pendingAutoRecovery = Boolean(this._pendingAutoRecovery);

    return {
      sampleRate: sanitizePositiveFiniteNumber(this.sampleRate, this._requestedSampleRate),
      state: this.state,
      outputState,
      contextState: outputState,
      channels: sanitizeFiniteNonNegInt(this._channels, this._requestedChannels),
      capacityFrames,
      bufferedFrames,
      freeFrames,
      submittedFrames: sanitizeFiniteNonNegInt(this._submittedFrames, 0),
      consumedFrames: sanitizeFiniteNonNegInt(this._consumedFrames, 0),
      sanitizedSamples: sanitizeFiniteNonNegInt(this._sanitizedSamples, 0),
      underruns,
      underrunFrames,
      overflows,
      overruns: overflows,
      droppedFrames,
      volume: gain,
      gain,
      masterGain: gain,
      muted,
      effectiveGain,
      targetLatencyMs,
      gainClamps: sanitizeFiniteNonNegInt(this._gainClamps, 0),
      invalidGainInputs: sanitizeFiniteNonNegInt(this._invalidGainInputs, 0),
      resumeAttempts: sanitizeFiniteNonNegInt(this._resumeAttempts, 0),
      resumeSuccesses: sanitizeFiniteNonNegInt(this._resumeSuccesses, 0),
      resumeFailures: sanitizeFiniteNonNegInt(this._resumeFailures, 0),
      interruptionCount,
      autoRecoveryCount,
      pendingAutoRecovery,
      stateTransitions: sanitizeFiniteNonNegInt(this._stateTransitions, 0),
      lastResumeError: this._lastResumeError,
      error: this._error ? sanitizeResumeError(this._error) : null,
      guestRingAttached,
      guestFramesDrained,
      guestUnderruns,
      guestOverruns,
      guestPeakLevel,
      unlockedOnGesture,
    };
  }

  /**
   * Returns a sanitized, finite diagnostic snapshot of AudioContext resume/state
   * telemetry, clamped gain/mute controls, live SAB ring counters, and #1027
   * guest stereo PCM ring telemetry.
   */
  getStats() {
    const telemetry = this.getTelemetry();
    return {
      outputState: telemetry.outputState,
      contextState: telemetry.contextState,
      state: telemetry.outputState,
      sampleRate: telemetry.sampleRate,
      channels: telemetry.channels,
      capacityFrames: telemetry.capacityFrames,
      bufferedFrames: telemetry.bufferedFrames,
      freeFrames: telemetry.freeFrames,
      submittedFrames: telemetry.submittedFrames,
      consumedFrames: telemetry.consumedFrames,
      sanitizedSamples: telemetry.sanitizedSamples,
      underruns: telemetry.underruns,
      underrunFrames: telemetry.underrunFrames,
      overflows: telemetry.overflows,
      overruns: telemetry.overruns,
      droppedFrames: telemetry.droppedFrames,
      gain: telemetry.gain,
      volume: telemetry.volume,
      masterGain: telemetry.masterGain,
      muted: telemetry.muted,
      effectiveGain: telemetry.effectiveGain,
      targetLatencyMs: telemetry.targetLatencyMs,
      gainClamps: telemetry.gainClamps,
      invalidGainInputs: telemetry.invalidGainInputs,
      resumeAttempts: telemetry.resumeAttempts,
      resumeSuccesses: telemetry.resumeSuccesses,
      resumeFailures: telemetry.resumeFailures,
      interruptionCount: telemetry.interruptionCount,
      autoRecoveryCount: telemetry.autoRecoveryCount,
      pendingAutoRecovery: telemetry.pendingAutoRecovery,
      stateTransitions: telemetry.stateTransitions,
      lastResumeError: telemetry.lastResumeError,
      error: telemetry.error,
      guestRingAttached: telemetry.guestRingAttached,
      guestFramesDrained: telemetry.guestFramesDrained,
      guestUnderruns: telemetry.guestUnderruns,
      guestOverruns: telemetry.guestOverruns,
      guestPeakLevel: telemetry.guestPeakLevel,
      unlockedOnGesture: telemetry.unlockedOnGesture,
    };
  }

  /**
   * Returns a compact, sanitized PCM & AudioContext telemetry snapshot for
   * runtime dashboards and diagnostics:
   * `{ submittedFrames, droppedFrames, underruns, overruns, contextState, sampleRate, muted, ... }`.
   */
  getAudioStats() {
    const telemetry = this.getTelemetry();
    return {
      submittedFrames: telemetry.submittedFrames,
      droppedFrames: telemetry.droppedFrames,
      underruns: telemetry.underruns,
      underrunFrames: telemetry.underrunFrames,
      overruns: telemetry.overruns,
      contextState: telemetry.contextState,
      sampleRate: telemetry.sampleRate,
      muted: telemetry.muted,
      volume: telemetry.volume,
      masterGain: telemetry.masterGain,
      effectiveGain: telemetry.effectiveGain,
      targetLatencyMs: telemetry.targetLatencyMs,
      channels: telemetry.channels,
      capacityFrames: telemetry.capacityFrames,
      bufferedFrames: telemetry.bufferedFrames,
      freeFrames: telemetry.freeFrames,
      consumedFrames: telemetry.consumedFrames,
      sanitizedSamples: telemetry.sanitizedSamples,
      resumeAttempts: telemetry.resumeAttempts,
      resumeSuccesses: telemetry.resumeSuccesses,
      resumeFailures: telemetry.resumeFailures,
      interruptionCount: telemetry.interruptionCount,
      autoRecoveryCount: telemetry.autoRecoveryCount,
      pendingAutoRecovery: telemetry.pendingAutoRecovery,
      lastResumeError: telemetry.lastResumeError,
      guestRingAttached: telemetry.guestRingAttached,
      guestFramesDrained: telemetry.guestFramesDrained,
      guestUnderruns: telemetry.guestUnderruns,
      guestOverruns: telemetry.guestOverruns,
      guestPeakLevel: telemetry.guestPeakLevel,
      unlockedOnGesture: telemetry.unlockedOnGesture,
    };
  }

  /**
   * Alias for `destroy()`: idempotently stops the worklet, removes gesture
   * listeners, disconnects nodes, and closes the AudioContext.
   */
  async close() {
    return this.destroy();
  }

  /**
   * Alias for `destroy()`: idempotently stops the worklet, removes `statechange`
   * and gesture listeners, disconnects nodes, and closes the AudioContext.
   */
  async dispose() {
    return this.destroy();
  }

  /**
   * Stops the AudioWorkletProcessor, removes any pending user-gesture unlock
   * listeners, disconnects the node, and closes the AudioContext.
   */
  async destroy() {
    if (this._destroyed) {
      return;
    }

    if (this._guestRingAttached) {
      this.detachGuestPcmRing();
    } else {
      this._guestPcmRing = null;
    }

    this._destroyed = true;
    this._status = 'closed';
    this._resumePromise = null;
    this._masterGainHeldSilent = false;
    this._pendingAutoRecovery = false;
    this._wasPlaying = false;
    this._explicitlySuspended = false;

    this._clearGestureListeners();

    const node = this._node;
    this._node = null;
    if (node) {
      try {
        if (node.port) {
          if (typeof node.port.postMessage === 'function') {
            node.port.postMessage({ type: 'stop' });
          }
          node.port.onmessage = null;
          if (typeof node.port.close === 'function') {
            node.port.close();
          }
        }
      } catch {
        // Ignore teardown errors on port.
      }
      try {
        if (typeof node.disconnect === 'function') {
          node.disconnect();
        }
      } catch {
        // Ignore teardown errors on disconnect.
      }
    }

    const masterGainNode = this._masterGainNode;
    this._masterGainNode = null;
    if (masterGainNode) {
      try {
        if (typeof masterGainNode.disconnect === 'function') {
          masterGainNode.disconnect();
        }
      } catch {
        // Ignore teardown errors on master gain disconnect.
      }
    }

    const ctx = this._context;
    this._context = null;
    if (ctx) {
      if (this._boundContextStateChange) {
        try {
          if (typeof ctx.removeEventListener === 'function') {
            ctx.removeEventListener('statechange', this._boundContextStateChange);
          } else if (ctx.onstatechange === this._boundContextStateChange) {
            ctx.onstatechange = null;
          }
        } catch {
          // Ignore teardown errors on context listener removal.
        }
        this._boundContextStateChange = null;
      }
      await this._safeCloseContext(ctx);
    }

    this._emitStateChange();
  }
}
