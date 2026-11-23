import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import PerfMonitorDefault, {
  DEFAULT_WINDOW_SIZE,
  DEFAULT_JANK_THRESHOLD_MS,
  DEFAULT_HUD_INTERVAL_MS,
  BYTES_PER_MB,
  PerfMonitor,
} from '../web/telemetry/perf_monitor.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');

function createDeterministicClock(startMs = 0) {
  let currentMs = startMs;
  return {
    now: () => currentMs,
    advance(deltaMs) {
      currentMs += deltaMs;
      return currentMs;
    },
    set(valueMs) {
      currentMs = valueMs;
      return currentMs;
    },
  };
}

function createFakeTimers() {
  let nextId = 1;
  const active = new Map();
  const clearedIds = [];

  return {
    setInterval(fn, ms) {
      const id = nextId++;
      active.set(id, { fn, ms });
      return id;
    },
    clearInterval(id) {
      clearedIds.push(id);
      active.delete(id);
    },
    tick(id) {
      const entry = active.get(id);
      if (entry) {
        entry.fn();
      }
    },
    tickAll() {
      for (const entry of [...active.values()]) {
        entry.fn();
      }
    },
    activeCount() {
      return active.size;
    },
    getIntervalMs(id) {
      return active.get(id)?.ms ?? null;
    },
    clearedIds,
  };
}

test('default export and constructor defaults / validation', () => {
  assert.equal(PerfMonitorDefault, PerfMonitor);
  assert.equal(DEFAULT_WINDOW_SIZE, 120);
  assert.equal(DEFAULT_JANK_THRESHOLD_MS, 33.34);
  assert.equal(DEFAULT_HUD_INTERVAL_MS, 250);
  assert.equal(BYTES_PER_MB, 1024 * 1024);

  const clock = createDeterministicClock(100);
  const monitor = new PerfMonitor({ now: clock.now });
  assert.equal(monitor.windowSize, 120);
  assert.equal(monitor.jankThresholdMs, 33.34);
  assert.equal(monitor.isHudAttached, false);
  assert.ok(monitor._frameDeltas instanceof Float64Array);
  assert.equal(monitor._frameDeltas.length, 120);

  const snap = monitor.getSnapshot();
  assert.equal(snap.timestampMs, 100);
  assert.equal(snap.frames.fps, 0);
  assert.equal(snap.frames.avgFrameTimeMs, 0);
  assert.equal(snap.frames.minFrameTimeMs, 0);
  assert.equal(snap.frames.p95FrameTimeMs, 0);
  assert.equal(snap.frames.p99FrameTimeMs, 0);
  assert.equal(snap.frames.maxFrameTimeMs, 0);
  assert.equal(snap.frames.totalFrames, 0);
  assert.equal(snap.frames.jankFrames, 0);
  assert.equal(snap.frames.windowJankFrames, 0);
  assert.equal(snap.frames.sampleCount, 0);
  assert.equal(snap.gpu.gpuPacketsPerSec, 0);
  assert.equal(snap.gpu.drawCallsPerFrame, 0);
  assert.equal(snap.gpu.totalGpuPackets, 0);
  assert.equal(snap.io.cacheHitRate, 0);
  assert.equal(snap.io.ioThroughputMBps, 0);
  assert.equal(snap.io.totalBytesRead, 0);
  assert.equal(snap.io.cacheHits, 0);
  assert.equal(snap.io.opfsCacheHits, 0);
  assert.equal(snap.io.cacheMisses, 0);
  assert.equal(snap.audio.bufferedFrames, 0);
  assert.equal(snap.audio.underruns, 0);
  assert.equal(snap.audio.underrunFrames, 0);
  assert.equal(snap.audio.activeVoices, 0);

  assert.throws(() => new PerfMonitor(null), TypeError);
  assert.throws(() => new PerfMonitor({ windowSize: 0 }), RangeError);
  assert.throws(() => new PerfMonitor({ windowSize: 2.5 }), RangeError);
  assert.throws(() => new PerfMonitor({ jankThresholdMs: 0 }), RangeError);
  assert.throws(() => new PerfMonitor({ jankThresholdMs: -5 }), RangeError);
  assert.throws(() => new PerfMonitor({ jankThresholdMs: Number.NaN }), RangeError);
  assert.throws(() => new PerfMonitor({ gpuWindowSize: 0 }), RangeError);
  assert.throws(() => new PerfMonitor({ ioWindowSize: -1 }), RangeError);
  assert.throws(() => new PerfMonitor({ now: 'clock' }), TypeError);
  assert.throws(() => new PerfMonitor({ setInterval: null }), TypeError);
  assert.throws(() => new PerfMonitor({ clearInterval: 123 }), TypeError);
});

test('recordFrame with deterministic timestamps computes exact FPS, avg, min, p95, p99, max, and jank', () => {
  const clock = createDeterministicClock(0);
  const monitor = new PerfMonitor({ windowSize: 120, now: clock.now });

  // Baseline frame sets timestamp without producing a delta yet.
  assert.equal(monitor.recordFrame(1000), null);
  let snap = monitor.getSnapshot();
  assert.equal(snap.frames.totalFrames, 1);
  assert.equal(snap.frames.sampleCount, 0);
  assert.equal(snap.frames.fps, 0);

  // Build 100 deterministic deltas:
  //   94 frames @ 16ms
  //    4 frames @ 25ms
  //    1 frame  @ 40ms (jank > 33.34ms)
  //    1 frame  @ 80ms (jank > 33.34ms)
  const deltas = [
    ...Array(94).fill(16),
    ...Array(4).fill(25),
    40,
    80,
  ];
  let ts = 1000;
  for (const dt of deltas) {
    ts += dt;
    assert.equal(monitor.recordFrame(ts), dt);
  }

  snap = monitor.getSnapshot();
  const expectedSum = 94 * 16 + 4 * 25 + 40 + 80; // 1724
  const expectedAvg = expectedSum / 100; // 17.24
  assert.equal(snap.frames.totalFrames, 101);
  assert.equal(snap.frames.sampleCount, 100);
  assert.ok(Math.abs(snap.frames.avgFrameTimeMs - expectedAvg) < 1e-9);
  assert.ok(Math.abs(snap.frames.fps - 1000 / expectedAvg) < 1e-9);
  assert.equal(snap.frames.minFrameTimeMs, 16);
  // Sorted indices (0-based): 0..93 => 16, 94..97 => 25, 98 => 40, 99 => 80
  // Nearest-rank p95 => ceil(0.95 * 100) - 1 = 94 => 25
  // Nearest-rank p99 => ceil(0.99 * 100) - 1 = 98 => 40
  assert.equal(snap.frames.p95FrameTimeMs, 25);
  assert.equal(snap.frames.p99FrameTimeMs, 40);
  assert.equal(snap.frames.maxFrameTimeMs, 80);
  assert.equal(snap.frames.jankFrames, 2);
  assert.equal(snap.frames.windowJankFrames, 2);
});

test('recordFrameTimeMs computes exact percentiles and handles jank threshold boundaries', () => {
  const monitor = new PerfMonitor({ windowSize: 100, jankThresholdMs: 33.34 });

  // Deltas 1..100ms
  for (let i = 1; i <= 100; i += 1) {
    assert.equal(monitor.recordFrameTimeMs(i), i);
  }

  const snap = monitor.getSnapshot();
  assert.equal(snap.frames.totalFrames, 100);
  assert.equal(snap.frames.sampleCount, 100);
  assert.equal(snap.frames.minFrameTimeMs, 1);
  assert.equal(snap.frames.avgFrameTimeMs, 50.5);
  assert.ok(Math.abs(snap.frames.fps - 1000 / 50.5) < 1e-9);
  assert.equal(snap.frames.p95FrameTimeMs, 95);
  assert.equal(snap.frames.p99FrameTimeMs, 99);
  assert.equal(snap.frames.maxFrameTimeMs, 100);
  // 34..100 => 67 jank frames (> 33.34ms)
  assert.equal(snap.frames.jankFrames, 67);
  assert.equal(snap.frames.windowJankFrames, 67);

  // Boundary check: exactly equal to jankThresholdMs is NOT jank.
  const exactMonitor = new PerfMonitor({ windowSize: 10, jankThresholdMs: 33.34 });
  exactMonitor.recordFrameTimeMs(33.34);
  exactMonitor.recordFrameTimeMs(33.35);
  const exactSnap = exactMonitor.getSnapshot();
  assert.equal(exactSnap.frames.jankFrames, 1);
  assert.equal(exactSnap.frames.windowJankFrames, 1);

  assert.throws(() => exactMonitor.recordFrameTimeMs(-0.1), RangeError);
  assert.throws(() => exactMonitor.recordFrameTimeMs(Number.NaN), RangeError);
  assert.throws(() => exactMonitor.recordFrameTimeMs(Number.POSITIVE_INFINITY), RangeError);
});

test('ring buffer wraps cleanly beyond windowSize and evicts stale spikes from rolling stats', () => {
  const monitor = new PerfMonitor({ windowSize: 8, jankThresholdMs: 33.34 });

  // Push 4 huge jank spikes first.
  monitor.recordFrameTimeMs(120);
  monitor.recordFrameTimeMs(150);
  monitor.recordFrameTimeMs(90);
  monitor.recordFrameTimeMs(80);

  let snap = monitor.getSnapshot();
  assert.equal(snap.frames.sampleCount, 4);
  assert.equal(snap.frames.maxFrameTimeMs, 150);
  assert.equal(snap.frames.jankFrames, 4);
  assert.equal(snap.frames.windowJankFrames, 4);

  // Push 12 smooth 16ms frames (well past windowSize = 8).
  for (let i = 0; i < 12; i += 1) {
    monitor.recordFrameTimeMs(16);
  }

  snap = monitor.getSnapshot();
  assert.equal(snap.frames.totalFrames, 16);
  assert.equal(snap.frames.sampleCount, 8);
  assert.equal(snap.frames.minFrameTimeMs, 16);
  assert.equal(snap.frames.avgFrameTimeMs, 16);
  assert.equal(snap.frames.p95FrameTimeMs, 16);
  assert.equal(snap.frames.p99FrameTimeMs, 16);
  assert.equal(snap.frames.maxFrameTimeMs, 16);
  assert.ok(Math.abs(snap.frames.fps - 62.5) < 1e-9);
  // Cumulative jank count remains 4, while rolling window now has 0 jank frames.
  assert.equal(snap.frames.jankFrames, 4);
  assert.equal(snap.frames.windowJankFrames, 0);
});

test('recordFrame clamps backward clock skew safely to zero delta', () => {
  const monitor = new PerfMonitor({ windowSize: 4 });
  monitor.recordFrame(500);
  assert.equal(monitor.recordFrame(450), 0);
  assert.equal(monitor.recordFrame(470), 20);

  const snap = monitor.getSnapshot();
  assert.equal(snap.frames.totalFrames, 3);
  assert.equal(snap.frames.sampleCount, 2);
  assert.equal(snap.frames.minFrameTimeMs, 0);
  assert.equal(snap.frames.maxFrameTimeMs, 20);
  assert.equal(snap.frames.avgFrameTimeMs, 10);
});

test('GPU batch aggregation computes rolling gpuPacketsPerSec, drawCallsPerFrame, and cumulative totals', () => {
  const clock = createDeterministicClock(0);
  const monitor = new PerfMonitor({ windowSize: 4, gpuWindowSize: 4, now: clock.now });

  // 4 batches spaced 250ms apart (1000ms total), 50 packets + 15 draw calls per batch, 1 frame per batch.
  for (let i = 0; i < 4; i += 1) {
    clock.advance(250);
    monitor.recordGpuBatch({
      packets: 50,
      drawCalls: 15,
      computeDispatches: 3,
      bytes: 2048,
    });
    monitor.recordFrameTimeMs(16);
  }

  let snap = monitor.getSnapshot();
  assert.equal(snap.gpu.totalBatches, 4);
  assert.equal(snap.gpu.totalGpuPackets, 200);
  assert.equal(snap.gpu.totalDrawCalls, 60);
  assert.equal(snap.gpu.totalComputeDispatches, 12);
  assert.equal(snap.gpu.totalGpuBytes, 8192);
  assert.ok(Math.abs(snap.gpu.gpuPacketsPerSec - 200) < 1e-9);
  assert.equal(snap.gpu.drawCallsPerFrame, 15);

  // Wrap GPU ring with faster rate: 4 batches spaced 100ms apart (400ms total), 100 packets & 40 draw calls each.
  for (let i = 0; i < 4; i += 1) {
    clock.advance(100);
    monitor.recordGpuBatch({
      packets: 100,
      drawCalls: 40,
      computeDispatches: 1,
      bytes: 4096,
    });
    monitor.recordFrameTimeMs(16);
  }

  snap = monitor.getSnapshot();
  assert.equal(snap.gpu.totalBatches, 8);
  assert.equal(snap.gpu.totalGpuPackets, 600);
  assert.equal(snap.gpu.totalDrawCalls, 220);
  assert.equal(snap.gpu.totalComputeDispatches, 16);
  assert.equal(snap.gpu.totalGpuBytes, 24576);
  // 400 packets over 400ms => 1000 packets/sec
  assert.ok(Math.abs(snap.gpu.gpuPacketsPerSec - 1000) < 1e-9);
  assert.equal(snap.gpu.drawCallsPerFrame, 40);
});

test('GPU drawCallsPerFrame handles frame-before-batch order and explicit durationMs', () => {
  const monitor = new PerfMonitor({ windowSize: 2, gpuWindowSize: 2 });

  monitor.recordFrame(0);
  monitor.recordGpuBatch({ packets: 60, drawCalls: 10, durationMs: 500 });
  assert.equal(monitor.getSnapshot().gpu.drawCallsPerFrame, 10);
  assert.ok(Math.abs(monitor.getSnapshot().gpu.gpuPacketsPerSec - 120) < 1e-9);

  monitor.recordFrame(16);
  monitor.recordGpuBatch({ packets: 120, drawCalls: 20, durationMs: 500 });
  assert.equal(monitor.getSnapshot().gpu.drawCallsPerFrame, 15);
  assert.ok(Math.abs(monitor.getSnapshot().gpu.gpuPacketsPerSec - 180) < 1e-9);

  monitor.recordFrame(32);
  monitor.recordGpuBatch({ packets: 200, drawCalls: 40, durationMs: 500 });
  // Rolling 2-frame window now reflects 20 and 40 => 30 draw calls/frame.
  assert.equal(monitor.getSnapshot().gpu.drawCallsPerFrame, 30);
  // Rolling 2-batch window reflects 120 + 200 packets over 1000ms => 320 packets/sec.
  assert.ok(Math.abs(monitor.getSnapshot().gpu.gpuPacketsPerSec - 320) < 1e-9);
});

test('I/O read tracking computes cacheHitRate, opfsCacheHits, cacheMisses, and rolling ioThroughputMBps', () => {
  const monitor = new PerfMonitor({ ioWindowSize: 2 });

  // 1 MiB miss in 100ms => 10 MB/s
  monitor.recordIoRead({
    bytes: 1 * BYTES_PER_MB,
    cacheHit: false,
    opfsHit: false,
    durationMs: 100,
  });
  // 2 MiB OPFS hit in 200ms => combined 3 MiB in 300ms => 10 MB/s
  monitor.recordIoRead({
    bytes: 2 * BYTES_PER_MB,
    cacheHit: true,
    opfsHit: true,
    durationMs: 200,
  });

  let snap = monitor.getSnapshot();
  assert.equal(snap.io.totalBytesRead, 3 * BYTES_PER_MB);
  assert.equal(snap.io.cacheHits, 1);
  assert.equal(snap.io.opfsCacheHits, 1);
  assert.equal(snap.io.cacheMisses, 1);
  assert.equal(snap.io.totalReads, 2);
  assert.equal(snap.io.cacheHitRate, 0.5);
  assert.ok(Math.abs(snap.io.ioThroughputMBps - 10) < 1e-9);

  // Memory-only cache hit (opfsHit omitted/false): 4 MiB in 50ms
  monitor.recordIoRead({
    bytes: 4 * BYTES_PER_MB,
    cacheHit: true,
    durationMs: 50,
  });
  // OPFS hit with cacheHit omitted should also count as a cache hit: 6 MiB in 150ms
  monitor.recordIoRead({
    bytes: 6 * BYTES_PER_MB,
    opfsHit: true,
    durationMs: 150,
  });

  snap = monitor.getSnapshot();
  assert.equal(snap.io.totalBytesRead, 13 * BYTES_PER_MB);
  assert.equal(snap.io.cacheHits, 3);
  assert.equal(snap.io.opfsCacheHits, 2);
  assert.equal(snap.io.cacheMisses, 1);
  assert.equal(snap.io.totalReads, 4);
  assert.equal(snap.io.cacheHitRate, 0.75);
  // Rolling ioWindowSize = 2 holds last two reads: (4 + 6) MiB / (50 + 150)ms = 10 MiB / 0.2s = 50 MB/s
  assert.ok(Math.abs(snap.io.ioThroughputMBps - 50) < 1e-9);
});

test('audio telemetry pass-through supports full and partial updates', () => {
  const monitor = new PerfMonitor();

  monitor.updateAudioTelemetry({
    bufferedFrames: 2048,
    underruns: 2,
    underrunFrames: 256,
    activeVoices: 14,
  });

  let snap = monitor.getSnapshot();
  assert.deepEqual(snap.audio, {
    bufferedFrames: 2048,
    underruns: 2,
    underrunFrames: 256,
    activeVoices: 14,
  });

  // Partial update preserves untouched fields; negative/NaN values are safely ignored.
  monitor.updateAudioTelemetry({
    bufferedFrames: 1024,
    underruns: -1,
    activeVoices: Number.NaN,
  });

  snap = monitor.getSnapshot();
  assert.deepEqual(snap.audio, {
    bufferedFrames: 1024,
    underruns: 2,
    underrunFrames: 256,
    activeVoices: 14,
  });
});

test('toHudMetrics matches web/index.html playgta6Hud.updateMetrics contract without clobbering status/input', () => {
  const indexHtml = fs.readFileSync(path.join(REPO_ROOT, 'web', 'index.html'), 'utf8');
  assert.match(indexHtml, /function updateMetrics\(metrics = \{\}\)/);
  assert.match(indexHtml, /metrics\.fps/);
  assert.match(indexHtml, /metrics\.gpuPackets/);
  assert.match(indexHtml, /metrics\.opfsCacheHits/);
  assert.match(indexHtml, /metrics\.cacheHits/);

  const clock = createDeterministicClock(0);
  const monitor = new PerfMonitor({ now: clock.now });

  monitor.recordFrame(0);
  monitor.recordFrame(16);
  monitor.recordFrame(32);
  clock.advance(500);
  monitor.recordGpuBatch({ packets: 42, drawCalls: 12, computeDispatches: 2, bytes: 1024 });
  monitor.recordIoRead({ bytes: 2 * BYTES_PER_MB, cacheHit: true, opfsHit: true, durationMs: 100 });
  monitor.recordIoRead({ bytes: 1 * BYTES_PER_MB, cacheHit: true, opfsHit: false, durationMs: 50 });
  monitor.updateAudioTelemetry({
    bufferedFrames: 1536,
    underruns: 1,
    underrunFrames: 128,
    activeVoices: 8,
  });

  const hudMetrics = monitor.toHudMetrics();
  assert.ok(Math.abs(hudMetrics.fps - 62.5) < 1e-9);
  assert.equal(hudMetrics.gpuPackets, 42);
  assert.equal(hudMetrics.opfsCacheHits, 1);
  assert.equal(hudMetrics.cacheHits, 2);
  assert.equal(hudMetrics.avgFrameTimeMs, 16);
  assert.equal(hudMetrics.totalFrames, 3);
  assert.equal(hudMetrics.jankFrames, 0);
  assert.equal(hudMetrics.audioBufferedFrames, 1536);
  assert.equal(hudMetrics.audioUnderruns, 1);
  assert.equal(hudMetrics.audioUnderrunFrames, 128);
  assert.equal(hudMetrics.audioActiveVoices, 8);

  // Must not clobber unrelated HUD fields.
  assert.equal('pointerLock' in hudMetrics, false);
  assert.equal('gamepad' in hudMetrics, false);
  assert.equal('status' in hudMetrics, false);

  // Simulate exact web/index.html updateMetrics rendering logic.
  const domState = {
    fps: '--',
    fpsState: '',
    gpuPackets: '0',
    opfsCacheHits: '0',
    opfsState: '',
    pointerLock: 'Locked',
    gamepad: 'Xbox Wireless Controller',
    status: 'Running',
  };
  const mockHud = {
    updateMetrics(metrics = {}) {
      if (metrics.fps !== undefined && metrics.fps !== null) {
        domState.fps =
          typeof metrics.fps === 'number' && Number.isFinite(metrics.fps)
            ? metrics.fps >= 10
              ? Math.round(metrics.fps).toString()
              : metrics.fps.toFixed(1)
            : String(metrics.fps);
        domState.fpsState = typeof metrics.fps === 'number' && metrics.fps >= 30 ? 'ok' : 'info';
      }
      const packets = metrics.gpuPackets ?? metrics.packets;
      if (packets !== undefined && packets !== null) {
        domState.gpuPackets = String(packets);
      }
      const opfsHits = metrics.opfsCacheHits ?? metrics.cacheHits;
      if (opfsHits !== undefined && opfsHits !== null) {
        domState.opfsCacheHits = String(opfsHits);
        domState.opfsState = Number(opfsHits) > 0 ? 'ok' : '';
      }
      if (metrics.pointerLock !== undefined) domState.pointerLock = String(metrics.pointerLock);
      if (metrics.gamepad !== undefined) domState.gamepad = String(metrics.gamepad);
      if (metrics.status !== undefined) domState.status = String(metrics.status);
    },
  };

  mockHud.updateMetrics(hudMetrics);
  assert.equal(domState.fps, '63');
  assert.equal(domState.fpsState, 'ok');
  assert.equal(domState.gpuPackets, '42');
  assert.equal(domState.opfsCacheHits, '1');
  assert.equal(domState.opfsState, 'ok');
  assert.equal(domState.pointerLock, 'Locked');
  assert.equal(domState.gamepad, 'Xbox Wireless Controller');
  assert.equal(domState.status, 'Running');
});

test('attachHud / detachHud periodic pusher with injectable timers', () => {
  const clock = createDeterministicClock(0);
  const timers = createFakeTimers();
  const monitor = new PerfMonitor({
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  const received = [];
  const hud = {
    updateMetrics(metrics) {
      received.push(metrics);
    },
  };

  assert.throws(() => monitor.attachHud(null), TypeError);
  assert.throws(() => monitor.attachHud({}), TypeError);
  assert.throws(() => monitor.attachHud(hud, 0), RangeError);
  assert.throws(() => monitor.attachHud(hud, -10), RangeError);
  assert.equal(monitor.pushHudMetrics(), null);

  const detach = monitor.attachHud(hud, 250);
  assert.equal(monitor.isHudAttached, true);
  assert.equal(timers.activeCount(), 1);
  assert.equal(timers.getIntervalMs(1), 250);
  assert.equal(received.length, 0);

  monitor.recordFrameTimeMs(20);
  monitor.recordGpuBatch({ packets: 12, drawCalls: 4, durationMs: 20 });
  timers.tickAll();
  assert.equal(received.length, 1);
  assert.equal(received[0].fps, 50);
  assert.equal(received[0].gpuPackets, 12);

  // Re-attaching clears the prior timer so no duplicate intervals leak.
  const received2 = [];
  const hud2 = { updateMetrics: (m) => received2.push(m) };
  monitor.attachHud(hud2, 100);
  assert.equal(timers.activeCount(), 1);
  assert.deepEqual(timers.clearedIds, [1]);
  assert.equal(timers.getIntervalMs(2), 100);

  timers.tickAll();
  assert.equal(received.length, 1);
  assert.equal(received2.length, 1);

  // Detaching via returned function or detachHud() is idempotent.
  detach();
  assert.equal(monitor.isHudAttached, false);
  assert.equal(timers.activeCount(), 0);
  monitor.detachHud();
  assert.equal(monitor.isHudAttached, false);

  timers.tickAll();
  assert.equal(received2.length, 1);
});

test('reset clears all rings and counters while keeping HUD attached', () => {
  const clock = createDeterministicClock(0);
  const timers = createFakeTimers();
  const monitor = new PerfMonitor({
    windowSize: 16,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  const pushed = [];
  monitor.attachHud({ updateMetrics: (m) => pushed.push(m) }, 250);

  monitor.recordFrameTimeMs(16);
  monitor.recordFrameTimeMs(50);
  clock.advance(500);
  monitor.recordGpuBatch({ packets: 80, drawCalls: 20, computeDispatches: 4, bytes: 4096 });
  monitor.recordIoRead({ bytes: 2 * BYTES_PER_MB, cacheHit: true, opfsHit: true, durationMs: 100 });
  monitor.updateAudioTelemetry({
    bufferedFrames: 2048,
    underruns: 3,
    underrunFrames: 384,
    activeVoices: 6,
  });

  monitor.reset();

  assert.equal(monitor.isHudAttached, true);
  const snap = monitor.getSnapshot();
  assert.equal(snap.frames.fps, 0);
  assert.equal(snap.frames.avgFrameTimeMs, 0);
  assert.equal(snap.frames.maxFrameTimeMs, 0);
  assert.equal(snap.frames.totalFrames, 0);
  assert.equal(snap.frames.jankFrames, 0);
  assert.equal(snap.frames.sampleCount, 0);
  assert.equal(snap.gpu.totalGpuPackets, 0);
  assert.equal(snap.gpu.totalDrawCalls, 0);
  assert.equal(snap.gpu.totalComputeDispatches, 0);
  assert.equal(snap.gpu.totalGpuBytes, 0);
  assert.equal(snap.gpu.gpuPacketsPerSec, 0);
  assert.equal(snap.gpu.drawCallsPerFrame, 0);
  assert.equal(snap.io.totalBytesRead, 0);
  assert.equal(snap.io.cacheHits, 0);
  assert.equal(snap.io.opfsCacheHits, 0);
  assert.equal(snap.io.cacheMisses, 0);
  assert.equal(snap.io.cacheHitRate, 0);
  assert.equal(snap.io.ioThroughputMBps, 0);
  assert.deepEqual(snap.audio, {
    bufferedFrames: 0,
    underruns: 0,
    underrunFrames: 0,
    activeVoices: 0,
  });

  // HUD remains attached and receives fresh zeroed metrics on next tick.
  timers.tickAll();
  assert.equal(pushed.length, 1);
  assert.equal(pushed[0].fps, 0);
  assert.equal(pushed[0].gpuPackets, 0);
  assert.equal(pushed[0].opfsCacheHits, 0);
});
