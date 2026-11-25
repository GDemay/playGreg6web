import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import PerfMonitorDefault, {
  DEFAULT_WINDOW_SIZE,
  DEFAULT_JANK_THRESHOLD_MS,
  DEFAULT_TARGET_FPS,
  DEFAULT_FRAME_BUDGET_MS,
  DEFAULT_HUD_INTERVAL_MS,
  BYTES_PER_MB,
  PERF_PHASES,
  FRAME_BOTTLENECKS,
  classifyFrameBottleneck,
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
  assert.equal(pushed[0].onePercentLowFps, 0);
  assert.equal(pushed[0].fps1PctLow, 0);
  assert.equal(pushed[0].p50FrameMs, 0);
  assert.equal(pushed[0].p95FrameMs, 0);
  assert.equal(pushed[0].p99FrameMs, 0);
  assert.equal(pushed[0].jankFrameCount, 0);
  assert.equal(pushed[0].gpuPackets, 0);
  assert.equal(pushed[0].opfsCacheHits, 0);
});

test('W149: empty state returns zeroed P50/P95/P99, 1% low FPS, and jank counters across snapshot, summary, and HUD', () => {
  const clock = createDeterministicClock(250);
  const monitor = new PerfMonitor({ windowSize: 300, jankThresholdMs: 50, now: clock.now });

  assert.equal(monitor.windowSize, 300);
  assert.equal(monitor.jankThresholdMs, 50);
  assert.ok(monitor._frameDeltas instanceof Float64Array);
  assert.equal(monitor._frameDeltas.length, 300);

  const snap = monitor.getSnapshot();
  const summary = monitor.getSummary();
  const hud = monitor.toHudMetrics();

  for (const view of [snap.frames, summary, hud]) {
    assert.equal(view.fps, 0);
    assert.equal(view.onePercentLowFps, 0);
    assert.equal(view.fps1PctLow, 0);
    assert.equal(view.p50FrameTimeMs, 0);
    assert.equal(view.p95FrameTimeMs, 0);
    assert.equal(view.p99FrameTimeMs, 0);
    assert.equal(view.p50FrameMs, 0);
    assert.equal(view.p95FrameMs, 0);
    assert.equal(view.p99FrameMs, 0);
    assert.equal(view.p50Ms, 0);
    assert.equal(view.p95Ms, 0);
    assert.equal(view.p99Ms, 0);
    assert.equal(view.jankFrames, 0);
    assert.equal(view.jankFrameCount, 0);
    assert.equal(view.windowJankFrames, 0);
  }

  assert.equal(summary.timestampMs, 250);
  assert.equal(summary.windowSize, 300);
  assert.equal(summary.jankThresholdMs, 50);
  assert.equal(summary.sampleCount, 0);
  assert.equal(summary.totalFrames, 0);

  // Baseline timestamp frame (no delta recorded yet) must keep percentiles and 1% low FPS at 0.
  assert.equal(monitor.recordFrame(1000), null);
  const afterBaseline = monitor.getSummary();
  assert.equal(afterBaseline.totalFrames, 1);
  assert.equal(afterBaseline.sampleCount, 0);
  assert.equal(afterBaseline.p50FrameMs, 0);
  assert.equal(afterBaseline.p95FrameMs, 0);
  assert.equal(afterBaseline.p99FrameMs, 0);
  assert.equal(afterBaseline.onePercentLowFps, 0);
  assert.equal(afterBaseline.jankFrameCount, 0);
});

test('W149: single sample computes exact P50/P95/P99, 1% low FPS, jank threshold, and zero-delta guard', () => {
  const smooth = new PerfMonitor({ windowSize: 300, jankThresholdMs: 50 });
  smooth.recordFrameTimeMs(16);

  const smoothSummary = smooth.getSummary();
  assert.equal(smoothSummary.sampleCount, 1);
  assert.equal(smoothSummary.p50FrameMs, 16);
  assert.equal(smoothSummary.p95FrameMs, 16);
  assert.equal(smoothSummary.p99FrameMs, 16);
  assert.equal(smoothSummary.p50Ms, 16);
  assert.equal(smoothSummary.p95Ms, 16);
  assert.equal(smoothSummary.p99Ms, 16);
  assert.ok(Math.abs(smoothSummary.fps - 62.5) < 1e-9);
  assert.ok(Math.abs(smoothSummary.onePercentLowFps - 62.5) < 1e-9);
  assert.ok(Math.abs(smoothSummary.fps1PctLow - 62.5) < 1e-9);
  assert.equal(smoothSummary.jankFrames, 0);
  assert.equal(smoothSummary.jankFrameCount, 0);
  assert.equal(smoothSummary.windowJankFrames, 0);

  // Single jank sample above 50ms threshold.
  const janky = new PerfMonitor({ windowSize: 300, jankThresholdMs: 50 });
  janky.recordFrameTimeMs(64);
  const jankySummary = janky.getSummary();
  assert.equal(jankySummary.p50FrameMs, 64);
  assert.equal(jankySummary.p95FrameMs, 64);
  assert.equal(jankySummary.p99FrameMs, 64);
  assert.ok(Math.abs(jankySummary.fps - 15.625) < 1e-9);
  assert.ok(Math.abs(jankySummary.onePercentLowFps - 15.625) < 1e-9);
  assert.equal(jankySummary.jankFrames, 1);
  assert.equal(jankySummary.jankFrameCount, 1);
  assert.equal(jankySummary.windowJankFrames, 1);

  // Boundary: exactly equal to 50ms is NOT jank.
  const boundary = new PerfMonitor({ windowSize: 300, jankThresholdMs: 50 });
  boundary.recordFrameTimeMs(50);
  assert.equal(boundary.getSummary().jankFrameCount, 0);
  boundary.recordFrameTimeMs(50.001);
  assert.equal(boundary.getSummary().jankFrameCount, 1);

  // Single 0ms delta must not divide by zero into Infinity or NaN.
  const zeroDt = new PerfMonitor({ windowSize: 10 });
  zeroDt.recordFrameTimeMs(0);
  const zeroSummary = zeroDt.getSummary();
  assert.equal(zeroSummary.p50FrameMs, 0);
  assert.equal(zeroSummary.p95FrameMs, 0);
  assert.equal(zeroSummary.p99FrameMs, 0);
  assert.equal(zeroSummary.fps, 0);
  assert.equal(zeroSummary.onePercentLowFps, 0);
  assert.equal(zeroSummary.fps1PctLow, 0);
  assert.ok(Number.isFinite(zeroSummary.onePercentLowFps));
});

test('W149: 300-sample Float64Array ring computes exact P50/P95/P99, 1% low FPS, and jank without reallocating buffers', () => {
  const monitor = new PerfMonitor({ windowSize: 300, jankThresholdMs: 50 });
  const initialRingRef = monitor._frameDeltas;
  const initialScratchRef = monitor._scratchSorted;

  // 300 samples in unsorted order:
  //   250 @ 16ms, 35 @ 20ms, 10 @ 35ms, 3 @ 60ms (jank), 2 @ 100ms (jank)
  const samples = [
    ...Array(2).fill(100),
    ...Array(35).fill(20),
    ...Array(3).fill(60),
    ...Array(250).fill(16),
    ...Array(10).fill(35),
  ];
  assert.equal(samples.length, 300);

  for (const dt of samples) {
    monitor.recordFrameTimeMs(dt);
  }

  // Zero-allocation check: typed arrays remain the exact pre-allocated instances.
  assert.equal(monitor._frameDeltas, initialRingRef);
  assert.equal(monitor._scratchSorted, initialScratchRef);

  // Sorted indices (0..299):
  //   0..249 => 16, 250..284 => 20, 285..294 => 35, 295..297 => 60, 298..299 => 100
  // Nearest-rank indices:
  //   p50 => ceil(0.50 * 300) - 1 = 149 => 16
  //   p95 => ceil(0.95 * 300) - 1 = 284 => 20
  //   p99 => ceil(0.99 * 300) - 1 = 296 => 60
  const expectedAvg = (250 * 16 + 35 * 20 + 10 * 35 + 3 * 60 + 2 * 100) / 300;
  const expectedFps = 1000 / expectedAvg;
  const expected1PctLow = 1000 / 60;

  const snap = monitor.getSnapshot();
  const summary = monitor.getSummary();
  const hud = monitor.toHudMetrics();

  for (const view of [snap.frames, summary, hud]) {
    assert.ok(Math.abs(view.avgFrameTimeMs - expectedAvg) < 1e-9);
    assert.ok(Math.abs(view.fps - expectedFps) < 1e-9);
    assert.equal(view.p50FrameTimeMs, 16);
    assert.equal(view.p95FrameTimeMs, 20);
    assert.equal(view.p99FrameTimeMs, 60);
    assert.equal(view.p50FrameMs, 16);
    assert.equal(view.p95FrameMs, 20);
    assert.equal(view.p99FrameMs, 60);
    assert.equal(view.p50Ms, 16);
    assert.equal(view.p95Ms, 20);
    assert.equal(view.p99Ms, 60);
    assert.ok(Math.abs(view.onePercentLowFps - expected1PctLow) < 1e-9);
    assert.ok(Math.abs(view.fps1PctLow - expected1PctLow) < 1e-9);
    assert.equal(view.jankFrames, 5);
    assert.equal(view.jankFrameCount, 5);
    assert.equal(view.windowJankFrames, 5);
  }
  assert.equal(snap.frames.minFrameTimeMs, 16);
  assert.equal(snap.frames.maxFrameTimeMs, 100);
  assert.equal(snap.frames.sampleCount, 300);
});

test('W149: wrap-around ring buffer updates P50/P95/P99, 1% low FPS, and rolling vs cumulative jank cleanly', () => {
  const monitor = new PerfMonitor({ windowSize: 10, jankThresholdMs: 50 });

  // Fill ring with 10 slow 100ms frames.
  for (let i = 0; i < 10; i += 1) {
    monitor.recordFrameTimeMs(100);
  }
  let summary = monitor.getSummary();
  assert.equal(summary.sampleCount, 10);
  assert.equal(summary.p50FrameMs, 100);
  assert.equal(summary.p95FrameMs, 100);
  assert.equal(summary.p99FrameMs, 100);
  assert.ok(Math.abs(summary.onePercentLowFps - 10) < 1e-9);
  assert.equal(summary.jankFrameCount, 10);
  assert.equal(summary.windowJankFrames, 10);

  // Partial wrap: overwrite 5 entries with 16ms frames -> ring holds [16 x5, 100 x5].
  for (let i = 0; i < 5; i += 1) {
    monitor.recordFrameTimeMs(16);
  }
  summary = monitor.getSummary();
  assert.equal(summary.totalFrames, 15);
  assert.equal(summary.sampleCount, 10);
  // Sorted indices 0..4 => 16, 5..9 => 100. p50 => idx 4 (16), p95/p99 => idx 9 (100).
  assert.equal(summary.p50FrameMs, 16);
  assert.equal(summary.p95FrameMs, 100);
  assert.equal(summary.p99FrameMs, 100);
  assert.ok(Math.abs(summary.onePercentLowFps - 10) < 1e-9);
  assert.equal(summary.jankFrameCount, 10);
  assert.equal(summary.windowJankFrames, 5);

  // Full wrap: overwrite remaining 5 entries with 16ms frames -> ring holds [16 x10].
  for (let i = 0; i < 5; i += 1) {
    monitor.recordFrameTimeMs(16);
  }
  summary = monitor.getSummary();
  assert.equal(summary.totalFrames, 20);
  assert.equal(summary.sampleCount, 10);
  assert.equal(summary.p50FrameMs, 16);
  assert.equal(summary.p95FrameMs, 16);
  assert.equal(summary.p99FrameMs, 16);
  assert.ok(Math.abs(summary.onePercentLowFps - 62.5) < 1e-9);
  assert.equal(summary.jankFrameCount, 10);
  assert.equal(summary.windowJankFrames, 0);

  // Reset clears everything cleanly and subsequent samples work from scratch.
  monitor.reset();
  const resetSummary = monitor.getSummary();
  assert.equal(resetSummary.totalFrames, 0);
  assert.equal(resetSummary.sampleCount, 0);
  assert.equal(resetSummary.p50FrameMs, 0);
  assert.equal(resetSummary.p95FrameMs, 0);
  assert.equal(resetSummary.p99FrameMs, 0);
  assert.equal(resetSummary.onePercentLowFps, 0);
  assert.equal(resetSummary.fps1PctLow, 0);
  assert.equal(resetSummary.jankFrames, 0);
  assert.equal(resetSummary.jankFrameCount, 0);
  assert.equal(resetSummary.windowJankFrames, 0);

  monitor.recordFrameTimeMs(25);
  const postReset = monitor.getSummary();
  assert.equal(postReset.totalFrames, 1);
  assert.equal(postReset.sampleCount, 1);
  assert.equal(postReset.p50FrameMs, 25);
  assert.equal(postReset.p95FrameMs, 25);
  assert.equal(postReset.p99FrameMs, 25);
  assert.ok(Math.abs(postReset.onePercentLowFps - 40) < 1e-9);
  assert.equal(postReset.jankFrameCount, 0);
});

test('W149: guards against NaN, Infinity, negative, and extreme frame times', () => {
  const monitor = new PerfMonitor({ windowSize: 8, jankThresholdMs: 50 });

  // Extreme timestamp delta overflow in recordFrame clamps safely to 0.
  monitor.recordFrame(-Number.MAX_VALUE);
  assert.equal(monitor.recordFrame(Number.MAX_VALUE), 0);

  // Subnormal tiny frame delta must not produce Infinity FPS / 1% low FPS.
  monitor.recordFrameTimeMs(1e-320);

  // Even if internal ring buffer slots are externally corrupted, snapshot/summary stay sanitized.
  monitor._frameDeltas[0] = Number.NaN;
  monitor._frameDeltas[1] = Number.POSITIVE_INFINITY;
  monitor._frameDeltas[2] = -42;
  monitor._frameCount = 4;
  monitor._frameDeltas[3] = 20;

  const summary = monitor.getSummary();
  for (const key of [
    'fps',
    'onePercentLowFps',
    'fps1PctLow',
    'avgFrameTimeMs',
    'minFrameTimeMs',
    'p50FrameMs',
    'p95FrameMs',
    'p99FrameMs',
    'p50Ms',
    'p95Ms',
    'p99Ms',
    'maxFrameTimeMs',
  ]) {
    assert.ok(Number.isFinite(summary[key]), `${key} must be finite, got ${summary[key]}`);
    assert.ok(summary[key] >= 0, `${key} must be non-negative, got ${summary[key]}`);
  }
  assert.equal(summary.p50Ms, 0);
  assert.equal(summary.p95Ms, 20);
  assert.equal(summary.p99Ms, 20);
  assert.ok(Math.abs(summary.onePercentLowFps - 50) < 1e-9);
});

test('W242: PERF_PHASES exports canonical frozen phases and default snapshot phase state', () => {
  assert.deepEqual(PERF_PHASES, {
    BOOT: 'boot',
    LOADING_ASSETS: 'loading_assets',
    GAMEPLAY: 'gameplay',
    MENU: 'menu',
  });
  assert.equal(Object.isFrozen(PERF_PHASES), true);

  const monitor = new PerfMonitor({ windowSize: 60 });
  assert.equal(monitor.currentPhase, PERF_PHASES.BOOT);

  const snap = monitor.getSnapshot();
  assert.equal(snap.currentPhase, PERF_PHASES.BOOT);
  assert.deepEqual(Object.keys(snap.phases), ['boot', 'loading_assets', 'gameplay', 'menu']);

  for (const phase of Object.values(PERF_PHASES)) {
    assert.deepEqual(snap.phases[phase], {
      frameCount: 0,
      avgFps: 0,
      onePercentLowFps: 0,
      avgFrameMs: 0,
      p95FrameMs: 0,
      p99FrameMs: 0,
      maxFrameMs: 0,
    });
  }

  // Optional initialPhase constructor option is sanitized.
  const gameplayStart = new PerfMonitor({ initialPhase: PERF_PHASES.GAMEPLAY });
  assert.equal(gameplayStart.currentPhase, PERF_PHASES.GAMEPLAY);
  assert.equal(gameplayStart.getSnapshot().currentPhase, PERF_PHASES.GAMEPLAY);

  const invalidStart = new PerfMonitor({ initialPhase: '__proto__' });
  assert.equal(invalidStart.currentPhase, PERF_PHASES.BOOT);
  assert.equal(invalidStart.getSnapshot().currentPhase, PERF_PHASES.BOOT);
});

test('W242: setPhase sanitizes inputs and prevents prototype pollution or arbitrary keys', () => {
  const monitor = new PerfMonitor({ windowSize: 16 });

  assert.equal(monitor.setPhase('GAMEPLAY'), PERF_PHASES.GAMEPLAY);
  assert.equal(monitor.currentPhase, 'gameplay');

  assert.equal(monitor.setPhase('  loading-assets  '), PERF_PHASES.LOADING_ASSETS);
  assert.equal(monitor.currentPhase, 'loading_assets');

  assert.equal(monitor.setPhase('Loading Assets'), PERF_PHASES.LOADING_ASSETS);
  assert.equal(monitor.currentPhase, 'loading_assets');

  assert.equal(monitor.setPhase('Menu'), PERF_PHASES.MENU);
  assert.equal(monitor.currentPhase, 'menu');

  for (const invalid of [
    undefined,
    null,
    '',
    '   ',
    42,
    {},
    [],
    'unknown_phase',
    '__proto__',
    'constructor',
    'toString',
  ]) {
    assert.equal(monitor.setPhase(invalid), PERF_PHASES.BOOT);
    assert.equal(monitor.currentPhase, PERF_PHASES.BOOT);
  }

  // Even if internal _currentPhase is externally corrupted, snapshot and recording stay sanitized.
  monitor._currentPhase = '__proto__';
  monitor.recordFrameTimeMs(20);
  const snap = monitor.getSnapshot();
  assert.equal(snap.currentPhase, PERF_PHASES.BOOT);
  assert.equal(snap.phases.boot.frameCount, 1);
  assert.deepEqual(Object.keys(snap.phases), ['boot', 'loading_assets', 'gameplay', 'menu']);
});

test('W242: phase-segmented histograms isolate gameplay FPS & 1%-low from boot/loading stalls', () => {
  const monitor = new PerfMonitor({ windowSize: 120, jankThresholdMs: 33.34 });

  // Phase 1: boot stalls (5 frames @ 200ms)
  assert.equal(monitor.currentPhase, PERF_PHASES.BOOT);
  for (let i = 0; i < 5; i += 1) {
    monitor.recordFrameTimeMs(200);
  }

  // Phase 2: loading_assets heavy RPF/Wasm stalls (10 frames: 8 @ 100ms, 1 @ 250ms, 1 @ 500ms)
  monitor.setPhase(PERF_PHASES.LOADING_ASSETS);
  for (let i = 0; i < 8; i += 1) {
    monitor.recordFrameTimeMs(100);
  }
  monitor.recordFrameTimeMs(250);
  monitor.recordFrameTimeMs(500);

  // Phase 3: smooth gameplay (100 frames: 95 @ 16ms, 4 @ 20ms, 1 @ 25ms)
  monitor.setPhase(PERF_PHASES.GAMEPLAY);
  for (let i = 0; i < 95; i += 1) {
    monitor.recordFrameTimeMs(16);
  }
  for (let i = 0; i < 4; i += 1) {
    monitor.recordFrameTimeMs(20);
  }
  monitor.recordFrameTimeMs(25);

  // Phase 4: menu (4 frames @ 32ms)
  monitor.setPhase(PERF_PHASES.MENU);
  for (let i = 0; i < 4; i += 1) {
    monitor.recordFrameTimeMs(32);
  }

  const snap = monitor.getSnapshot();
  assert.equal(snap.currentPhase, PERF_PHASES.MENU);

  // Global stats still reflect ALL 119 frames (including boot/loading stalls).
  assert.equal(snap.frames.totalFrames, 119);
  assert.equal(snap.frames.sampleCount, 119);
  assert.equal(snap.frames.maxFrameTimeMs, 500);
  assert.ok(snap.frames.onePercentLowFps < 10, 'global 1%-low is skewed by loading stalls');

  // Boot bucket: 5 frames @ 200ms => 5 FPS avg & 1%-low
  assert.equal(snap.phases.boot.frameCount, 5);
  assert.equal(snap.phases.boot.avgFrameMs, 200);
  assert.equal(snap.phases.boot.p95FrameMs, 200);
  assert.equal(snap.phases.boot.p99FrameMs, 200);
  assert.equal(snap.phases.boot.maxFrameMs, 200);
  assert.ok(Math.abs(snap.phases.boot.avgFps - 5) < 1e-9);
  assert.ok(Math.abs(snap.phases.boot.onePercentLowFps - 5) < 1e-9);

  // Loading assets bucket: [100 x8, 250, 500] => avg 155ms, p95=500 (ceil(9.5)-1=9), p99=500, max=500
  assert.equal(snap.phases.loading_assets.frameCount, 10);
  assert.equal(snap.phases.loading_assets.avgFrameMs, 155);
  assert.equal(snap.phases.loading_assets.p95FrameMs, 500);
  assert.equal(snap.phases.loading_assets.p99FrameMs, 500);
  assert.equal(snap.phases.loading_assets.maxFrameMs, 500);
  assert.ok(Math.abs(snap.phases.loading_assets.avgFps - 1000 / 155) < 1e-9);
  assert.ok(Math.abs(snap.phases.loading_assets.onePercentLowFps - 2) < 1e-9);

  // Gameplay bucket is completely unskewed by boot/loading stalls!
  // Sorted indices (0..99): 0..94 => 16, 95..98 => 20, 99 => 25
  // p95 => ceil(0.95 * 100) - 1 = 94 => 16
  // p99 => ceil(0.99 * 100) - 1 = 98 => 20
  const expectedGameplayAvg = (95 * 16 + 4 * 20 + 25) / 100; // 16.25
  assert.equal(snap.phases.gameplay.frameCount, 100);
  assert.ok(Math.abs(snap.phases.gameplay.avgFrameMs - expectedGameplayAvg) < 1e-9);
  assert.ok(Math.abs(snap.phases.gameplay.avgFps - 1000 / expectedGameplayAvg) < 1e-9);
  assert.equal(snap.phases.gameplay.p95FrameMs, 16);
  assert.equal(snap.phases.gameplay.p99FrameMs, 20);
  assert.equal(snap.phases.gameplay.maxFrameMs, 25);
  assert.ok(Math.abs(snap.phases.gameplay.onePercentLowFps - 50) < 1e-9);

  // Menu bucket: 4 frames @ 32ms => 31.25 FPS
  assert.equal(snap.phases.menu.frameCount, 4);
  assert.equal(snap.phases.menu.avgFrameMs, 32);
  assert.equal(snap.phases.menu.p95FrameMs, 32);
  assert.equal(snap.phases.menu.p99FrameMs, 32);
  assert.equal(snap.phases.menu.maxFrameMs, 32);
  assert.ok(Math.abs(snap.phases.menu.avgFps - 31.25) < 1e-9);
  assert.ok(Math.abs(snap.phases.menu.onePercentLowFps - 31.25) < 1e-9);
});

test('W242: recordFrame timestamp deltas accumulate into active phase and wrap cleanly with zero reallocations', () => {
  const monitor = new PerfMonitor({ windowSize: 4 });
  const bootBufRef = monitor._phaseBuckets[PERF_PHASES.BOOT].deltas;
  const gameplayBufRef = monitor._phaseBuckets[PERF_PHASES.GAMEPLAY].deltas;

  // Baseline frame sets timestamp without recording a delta sample.
  assert.equal(monitor.recordFrame(1000), null);
  assert.equal(monitor.getSnapshot().phases.boot.frameCount, 0);

  // Record 2 boot deltas: 100ms, 50ms
  assert.equal(monitor.recordFrame(1100), 100);
  assert.equal(monitor.recordFrame(1150), 50);

  // Switch to gameplay and record 6 frames (exceeding windowSize = 4):
  // First 2 are 80ms spikes, followed by 4 smooth 16ms frames which evict the spikes.
  monitor.setPhase(PERF_PHASES.GAMEPLAY);
  assert.equal(monitor.recordFrame(1230), 80);
  assert.equal(monitor.recordFrame(1310), 80);
  assert.equal(monitor.recordFrame(1326), 16);
  assert.equal(monitor.recordFrame(1342), 16);
  assert.equal(monitor.recordFrame(1358), 16);
  assert.equal(monitor.recordFrame(1374), 16);

  // Pre-allocated Float64Array buffers must never be reallocated.
  assert.equal(monitor._phaseBuckets[PERF_PHASES.BOOT].deltas, bootBufRef);
  assert.equal(monitor._phaseBuckets[PERF_PHASES.GAMEPLAY].deltas, gameplayBufRef);

  const snap = monitor.getSnapshot();
  assert.equal(snap.phases.boot.frameCount, 2);
  assert.equal(snap.phases.boot.avgFrameMs, 75);
  assert.equal(snap.phases.boot.maxFrameMs, 100);

  // Gameplay wrapped past windowSize=4: frameCount tracks all 6 samples, rolling stats reflect last 4 @ 16ms.
  assert.equal(snap.phases.gameplay.frameCount, 6);
  assert.equal(snap.phases.gameplay.avgFrameMs, 16);
  assert.equal(snap.phases.gameplay.p95FrameMs, 16);
  assert.equal(snap.phases.gameplay.p99FrameMs, 16);
  assert.equal(snap.phases.gameplay.maxFrameMs, 16);
  assert.ok(Math.abs(snap.phases.gameplay.avgFps - 62.5) < 1e-9);
  assert.ok(Math.abs(snap.phases.gameplay.onePercentLowFps - 62.5) < 1e-9);
});

test('W242: reset clears all phase histograms, restores initial phase, and guards corrupted/zero values', () => {
  const monitor = new PerfMonitor({ windowSize: 8 });

  monitor.setPhase(PERF_PHASES.LOADING_ASSETS);
  monitor.recordFrameTimeMs(120);
  monitor.setPhase(PERF_PHASES.GAMEPLAY);
  monitor.recordFrameTimeMs(16);
  monitor.recordFrameTimeMs(0);

  let snap = monitor.getSnapshot();
  assert.equal(snap.currentPhase, PERF_PHASES.GAMEPLAY);
  assert.equal(snap.phases.loading_assets.frameCount, 1);
  assert.equal(snap.phases.gameplay.frameCount, 2);

  monitor.reset();

  snap = monitor.getSnapshot();
  assert.equal(monitor.currentPhase, PERF_PHASES.BOOT);
  assert.equal(snap.currentPhase, PERF_PHASES.BOOT);
  for (const phase of Object.values(PERF_PHASES)) {
    assert.deepEqual(snap.phases[phase], {
      frameCount: 0,
      avgFps: 0,
      onePercentLowFps: 0,
      avgFrameMs: 0,
      p95FrameMs: 0,
      p99FrameMs: 0,
      maxFrameMs: 0,
    });
  }

  // Zero-delta and corrupted buffer slots stay finite and non-negative in phase stats.
  monitor.setPhase(PERF_PHASES.GAMEPLAY);
  monitor.recordFrameTimeMs(0);
  monitor.recordFrameTimeMs(1e-320);
  const bucket = monitor._phaseBuckets[PERF_PHASES.GAMEPLAY];
  bucket.deltas[0] = Number.NaN;
  bucket.deltas[1] = Number.POSITIVE_INFINITY;
  bucket.deltas[2] = -50;
  bucket.deltas[3] = 25;
  bucket.sampleCount = 4;
  bucket.frameCount = 4;

  const guarded = monitor.getSnapshot().phases.gameplay;
  for (const key of [
    'frameCount',
    'avgFps',
    'onePercentLowFps',
    'avgFrameMs',
    'p95FrameMs',
    'p99FrameMs',
    'maxFrameMs',
  ]) {
    assert.ok(Number.isFinite(guarded[key]), `${key} must be finite, got ${guarded[key]}`);
    assert.ok(guarded[key] >= 0, `${key} must be non-negative, got ${guarded[key]}`);
  }
  assert.equal(guarded.p95FrameMs, 25);
  assert.equal(guarded.p99FrameMs, 25);
  assert.equal(guarded.maxFrameMs, 25);
  assert.ok(Math.abs(guarded.onePercentLowFps - 40) < 1e-9);
});

test('W287: classifyFrameBottleneck exports canonical constants, classifies all 5 buckets, breaks ties deterministically, and sanitizes NaN/negative inputs', () => {
  assert.equal(DEFAULT_TARGET_FPS, 30);
  assert.equal(DEFAULT_FRAME_BUDGET_MS, 33.333);
  assert.deepEqual(FRAME_BOTTLENECKS, {
    WITHIN_BUDGET: 'within_budget',
    IO_BOUND: 'io_bound',
    SHADER_COMPILE_BOUND: 'shader_compile_bound',
    GPU_BOUND: 'gpu_bound',
    CPU_BOUND: 'cpu_bound',
  });
  assert.equal(Object.isFrozen(FRAME_BOTTLENECKS), true);

  // Within budget (including exact 33.333 ms boundary and 0 ms).
  assert.equal(classifyFrameBottleneck({ frameMs: 0 }), 'within_budget');
  assert.equal(classifyFrameBottleneck({ frameMs: 16.6, gpuMs: 12, cpuMs: 4.6 }), 'within_budget');
  assert.equal(
    classifyFrameBottleneck({ frameMs: 33.333, ioWaitMs: 25, gpuMs: 8 }),
    'within_budget'
  );

  // Over budget: each dominant bottleneck category.
  assert.equal(
    classifyFrameBottleneck({
      frameMs: 48,
      ioWaitMs: 22,
      shaderCompileMs: 5,
      gpuMs: 12,
      cpuMs: 9,
    }),
    'io_bound'
  );
  assert.equal(
    classifyFrameBottleneck({
      frameMs: 55,
      ioWaitMs: 4,
      shaderCompileMs: 31,
      gpuMs: 10,
      cpuMs: 10,
    }),
    'shader_compile_bound'
  );
  assert.equal(
    classifyFrameBottleneck({
      frameMs: 42,
      ioWaitMs: 2,
      shaderCompileMs: 1,
      gpuMs: 27,
      cpuMs: 12,
    }),
    'gpu_bound'
  );
  assert.equal(
    classifyFrameBottleneck({
      frameMs: 40,
      ioWaitMs: 1,
      shaderCompileMs: 2,
      gpuMs: 11,
      cpuMs: 26,
    }),
    'cpu_bound'
  );

  // Over budget with no stage breakdown (or all zeros) defaults deterministically to cpu_bound.
  assert.equal(classifyFrameBottleneck({ frameMs: 33.334 }), 'cpu_bound');
  assert.equal(
    classifyFrameBottleneck({
      frameMs: 50,
      ioWaitMs: 0,
      shaderCompileMs: 0,
      gpuMs: 0,
      cpuMs: 0,
    }),
    'cpu_bound'
  );

  // Deterministic tie-breaking priority: io_bound > shader_compile_bound > gpu_bound > cpu_bound.
  assert.equal(
    classifyFrameBottleneck({
      frameMs: 60,
      ioWaitMs: 15,
      shaderCompileMs: 15,
      gpuMs: 15,
      cpuMs: 15,
    }),
    'io_bound'
  );
  assert.equal(
    classifyFrameBottleneck({
      frameMs: 60,
      ioWaitMs: 5,
      shaderCompileMs: 20,
      gpuMs: 20,
      cpuMs: 20,
    }),
    'shader_compile_bound'
  );
  assert.equal(
    classifyFrameBottleneck({
      frameMs: 60,
      ioWaitMs: 5,
      shaderCompileMs: 5,
      gpuMs: 25,
      cpuMs: 25,
    }),
    'gpu_bound'
  );

  // Custom budgetMs support (e.g., 60 FPS = 16.667 ms).
  assert.equal(classifyFrameBottleneck({ frameMs: 20, gpuMs: 14, cpuMs: 6 }, 16.667), 'gpu_bound');
  assert.equal(
    classifyFrameBottleneck({ frameMs: 16.667, gpuMs: 14, cpuMs: 2.6 }, 16.667),
    'within_budget'
  );

  // Omitted frameMs derives total from sanitized stage components.
  assert.equal(classifyFrameBottleneck({ ioWaitMs: 30, cpuMs: 10 }), 'io_bound');
  assert.equal(classifyFrameBottleneck({ gpuMs: 10, cpuMs: 10 }), 'within_budget');

  // NaN, negative, Infinity, null/undefined, and invalid budgetMs sanitization.
  assert.equal(classifyFrameBottleneck(null), 'within_budget');
  assert.equal(classifyFrameBottleneck(undefined), 'within_budget');
  assert.equal(classifyFrameBottleneck({}), 'within_budget');
  assert.equal(classifyFrameBottleneck({ frameMs: Number.NaN, ioWaitMs: 50 }), 'within_budget');
  assert.equal(classifyFrameBottleneck({ frameMs: -10, gpuMs: 50 }), 'within_budget');
  assert.equal(
    classifyFrameBottleneck({ frameMs: Number.POSITIVE_INFINITY, cpuMs: 40 }),
    'within_budget'
  );
  assert.equal(
    classifyFrameBottleneck({
      frameMs: 45,
      ioWaitMs: Number.NaN,
      shaderCompileMs: -99,
      gpuMs: Number.POSITIVE_INFINITY,
      cpuMs: 12,
    }),
    'cpu_bound'
  );
  assert.equal(
    classifyFrameBottleneck({ frameMs: 45, gpuMs: 30 }, Number.NaN),
    'gpu_bound'
  );
  assert.equal(
    classifyFrameBottleneck({ frameMs: 20, gpuMs: 15 }, -10),
    'within_budget'
  );
  assert.equal(
    classifyFrameBottleneck({ frameMs: 40, ioWaitMs: 25 }, 0),
    'io_bound'
  );
});

test('W287: PerfMonitor tracks 30 FPS budget compliance, bottleneck counts, and dominantBottleneck via recordDetailedFrame and recordFrame', () => {
  const clock = createDeterministicClock(100);
  const monitor = new PerfMonitor({ windowSize: 32, now: clock.now });

  // Constructor defaults and validation.
  assert.equal(monitor.targetFps, 30);
  assert.equal(monitor.budgetMs, 33.333);
  assert.throws(() => new PerfMonitor({ targetFps: 0 }), RangeError);
  assert.throws(() => new PerfMonitor({ targetFps: -30 }), RangeError);
  assert.throws(() => new PerfMonitor({ targetFps: Number.NaN }), RangeError);
  assert.throws(() => new PerfMonitor({ budgetMs: 0 }), RangeError);
  assert.throws(() => new PerfMonitor({ budgetMs: -16 }), RangeError);
  assert.throws(() => new PerfMonitor({ budgetMs: Number.POSITIVE_INFINITY }), RangeError);

  const custom60 = new PerfMonitor({ targetFps: 60 });
  assert.equal(custom60.targetFps, 60);
  assert.equal(custom60.budgetMs, 16.667);

  // Empty initial state across getStats(), getSnapshot(), getSummary(), and toHudMetrics().
  const initStats = monitor.getStats();
  const initSnap = monitor.getSnapshot();
  assert.deepEqual(initStats, initSnap);
  assert.equal(initStats.targetFps, 30);
  assert.equal(initStats.budgetMs, 33.333);
  assert.equal(initStats.framesWithin30FpsBudget, 0);
  assert.equal(initStats.framesOver30FpsBudget, 0);
  assert.equal(initStats.fps30ComplianceRate, 0);
  assert.equal(initStats.dominantBottleneck, 'within_budget');
  assert.deepEqual(initStats.bottleneckCounts, {
    within_budget: 0,
    io_bound: 0,
    shader_compile_bound: 0,
    gpu_bound: 0,
    cpu_bound: 0,
  });

  // Record 5 within-budget frames and 5 over-budget frames across all 4 bottlenecks:
  //   5 within_budget, 1 io_bound, 2 shader_compile_bound, 1 gpu_bound, 1 cpu_bound.
  assert.equal(
    monitor.recordDetailedFrame({ frameMs: 16.5, gpuMs: 10, cpuMs: 6.5 }),
    'within_budget'
  );
  assert.equal(
    monitor.recordDetailedFrame({ frameMs: 28, ioWaitMs: 5, gpuMs: 15, cpuMs: 8 }),
    'within_budget'
  );
  assert.equal(
    monitor.recordDetailedFrame({ frameMs: 33.333, gpuMs: 20, cpuMs: 13.333 }),
    'within_budget'
  );
  assert.equal(
    monitor.recordDetailedFrame({ frameMs: 65, ioWaitMs: 42, gpuMs: 13, cpuMs: 10 }),
    'io_bound'
  );
  assert.equal(
    monitor.recordDetailedFrame({ frameMs: 90, shaderCompileMs: 60, gpuMs: 18, cpuMs: 12 }),
    'shader_compile_bound'
  );
  assert.equal(
    monitor.recordDetailedFrame({ frameMs: 75, shaderCompileMs: 48, gpuMs: 15, cpuMs: 12 }),
    'shader_compile_bound'
  );
  assert.equal(
    monitor.recordDetailedFrame({ frameMs: 44, ioWaitMs: 2, gpuMs: 31, cpuMs: 11 }),
    'gpu_bound'
  );
  assert.equal(
    monitor.recordDetailedFrame({ frameMs: 41, ioWaitMs: 1, gpuMs: 10, cpuMs: 30 }),
    'cpu_bound'
  );

  // Also test recordFrame timestamp delta wiring (baseline + 2 within-budget frames).
  monitor.recordFrame(1000); // baseline resets timestamp delta chain only when null; here _lastFrameTimestamp is null
  assert.equal(monitor.recordFrame(1020), 20);
  assert.equal(monitor.recordFrame(1045, { gpuMs: 12 }), 25);

  const stats = monitor.getStats();
  const snap = monitor.getSnapshot();
  const summary = monitor.getSummary();
  const hud = monitor.toHudMetrics();

  assert.equal(stats.framesWithin30FpsBudget, 5);
  assert.equal(stats.framesOver30FpsBudget, 5);
  assert.equal(stats.fps30ComplianceRate, 0.5);
  assert.equal(stats.dominantBottleneck, 'shader_compile_bound');
  assert.deepEqual(stats.bottleneckCounts, {
    within_budget: 5,
    io_bound: 1,
    shader_compile_bound: 2,
    gpu_bound: 1,
    cpu_bound: 1,
  });

  // Consistent across snapshot (top-level, budget, frames), summary, and HUD metrics.
  for (const view of [snap, snap.budget, snap.frames, summary, hud]) {
    assert.equal(view.targetFps, 30);
    assert.equal(view.budgetMs, 33.333);
    assert.equal(view.framesWithin30FpsBudget, 5);
    assert.equal(view.framesOver30FpsBudget, 5);
    assert.equal(view.fps30ComplianceRate, 0.5);
    assert.equal(view.dominantBottleneck, 'shader_compile_bound');
    assert.deepEqual(view.bottleneckCounts, {
      within_budget: 5,
      io_bound: 1,
      shader_compile_bound: 2,
      gpu_bound: 1,
      cpu_bound: 1,
    });
  }

  // Returned bottleneckCounts is a defensive copy (mutation cannot corrupt internal state).
  stats.bottleneckCounts.gpu_bound = 9999;
  assert.equal(monitor.getStats().bottleneckCounts.gpu_bound, 1);
  assert.equal(monitor.getStats().dominantBottleneck, 'shader_compile_bound');
});

test('W287: recordFrame timestamp mode, recordDetailedFrame timestamp/invalid inputs, tie-breaking, and reset()', () => {
  const monitor = new PerfMonitor({ windowSize: 16 });
  const initialRingRef = monitor._frameDeltas;
  const initialCountsRef = monitor._bottleneckCounts;

  // Non-object sample to recordDetailedFrame is a safe no-op returning null.
  assert.equal(monitor.recordDetailedFrame(null), null);
  assert.equal(monitor.recordDetailedFrame(undefined), null);
  assert.equal(monitor.getStats().frames.totalFrames, 0);

  // recordDetailedFrame using timestampMs when frameMs is omitted.
  assert.equal(monitor.recordDetailedFrame({ timestampMs: 500, gpuMs: 25 }), null);
  assert.equal(monitor.recordDetailedFrame({ timestampMs: 545, gpuMs: 30, cpuMs: 15 }), 'gpu_bound');
  assert.equal(monitor.recordDetailedFrame({ timestampMs: 590, ioWaitMs: 35, cpuMs: 10 }), 'io_bound');

  // Equal counts between io_bound (1) and gpu_bound (1) break tie to io_bound.
  let stats = monitor.getStats();
  assert.equal(stats.framesWithin30FpsBudget, 0);
  assert.equal(stats.framesOver30FpsBudget, 2);
  assert.equal(stats.fps30ComplianceRate, 0);
  assert.equal(stats.dominantBottleneck, 'io_bound');
  assert.equal(stats.budget.lastBottleneck, 'io_bound');

  // recordFrame without breakdown attributes over-budget deltas to cpu_bound.
  assert.equal(monitor.recordFrame(640), 50);
  assert.equal(monitor.recordFrame(690), 50);
  stats = monitor.getStats();
  assert.equal(stats.framesOver30FpsBudget, 4);
  assert.equal(stats.bottleneckCounts.cpu_bound, 2);
  assert.equal(stats.dominantBottleneck, 'cpu_bound');

  // Sanitized NaN/negative frameMs in recordDetailedFrame counts as 0ms (within_budget).
  assert.equal(
    monitor.recordDetailedFrame({ frameMs: Number.NaN, ioWaitMs: -12, gpuMs: Number.POSITIVE_INFINITY }),
    'within_budget'
  );
  stats = monitor.getStats();
  assert.equal(stats.framesWithin30FpsBudget, 1);
  assert.equal(stats.framesOver30FpsBudget, 4);
  assert.ok(Math.abs(stats.fps30ComplianceRate - 0.2) < 1e-9);
  assert.ok(stats.fps30ComplianceRate >= 0 && stats.fps30ComplianceRate <= 1);

  // Zero-allocation invariant: pre-allocated structures stay the same instances.
  assert.equal(monitor._frameDeltas, initialRingRef);
  assert.equal(monitor._bottleneckCounts, initialCountsRef);

  // reset() zeroes all budget & bottleneck counters cleanly without reallocating.
  monitor.reset();
  assert.equal(monitor._bottleneckCounts, initialCountsRef);

  const resetStats = monitor.getStats();
  assert.equal(resetStats.targetFps, 30);
  assert.equal(resetStats.budgetMs, 33.333);
  assert.equal(resetStats.framesWithin30FpsBudget, 0);
  assert.equal(resetStats.framesOver30FpsBudget, 0);
  assert.equal(resetStats.fps30ComplianceRate, 0);
  assert.equal(resetStats.dominantBottleneck, 'within_budget');
  assert.equal(resetStats.budget.lastBottleneck, 'within_budget');
  assert.deepEqual(resetStats.bottleneckCounts, {
    within_budget: 0,
    io_bound: 0,
    shader_compile_bound: 0,
    gpu_bound: 0,
    cpu_bound: 0,
  });

  // Post-reset smooth frames yield 100% compliance (1.0) and within_budget dominantBottleneck.
  monitor.recordDetailedFrame({ frameMs: 16.6, gpuMs: 10, cpuMs: 6.6 });
  monitor.recordDetailedFrame({ frameMs: 30, gpuMs: 18, cpuMs: 12 });
  const postReset = monitor.getStats();
  assert.equal(postReset.framesWithin30FpsBudget, 2);
  assert.equal(postReset.framesOver30FpsBudget, 0);
  assert.equal(postReset.fps30ComplianceRate, 1);
  assert.equal(postReset.dominantBottleneck, 'within_budget');
});
