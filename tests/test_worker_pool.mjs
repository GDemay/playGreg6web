import test from 'node:test';
import assert from 'node:assert/strict';

import {
  WorkerPool,
  PRIORITY,
  DEFAULT_MAX_RESPAWNS_PER_SLOT,
  DEFAULT_RESPAWN_BACKOFF_BASE_MS,
  DEFAULT_RESPAWN_BACKOFF_MAX_MS,
  WorkerPoolError,
  PoolTerminatedError,
  QueueFullError,
  JobAbortedError,
  JobTimeoutError,
  WorkerCrashError,
  CircuitOpenError,
  serveWorkerPool,
  withTransfer,
  serializeError,
  deserializeError,
  defaultPoolSize,
} from '../web/workers/worker_pool.js';

const flush = () => new Promise((resolve) => setImmediate(resolve));

/** Manual clock providing injectable now/setTimeout/clearTimeout. */
function createClock() {
  let time = 0;
  let nextId = 1;
  const timers = new Map();
  return {
    now: () => time,
    setTimeout(fn, ms) {
      const id = nextId++;
      timers.set(id, { at: time + ms, fn });
      return id;
    },
    clearTimeout(id) {
      timers.delete(id);
    },
    advance(ms) {
      const target = time + ms;
      for (;;) {
        let best = null;
        for (const [id, t] of timers) {
          if (t.at <= target && (!best || t.at < best.t.at)) best = { id, t };
        }
        if (!best) break;
        timers.delete(best.id);
        time = Math.max(time, best.t.at);
        best.t.fn();
      }
      time = target;
    },
    get pending() {
      return timers.size;
    },
  };
}

/** Worker double with scriptable replies, errors and termination. */
class FakeWorker {
  constructor() {
    this.listeners = { message: [], error: [], messageerror: [], exit: [] };
    this.received = [];
    this.transfers = [];
    this.terminated = false;
    this.autoReply = null;
  }
  addEventListener(type, fn) {
    this.listeners[type].push(fn);
  }
  postMessage(msg, transfer = []) {
    if (this.terminated) throw new Error('postMessage on terminated worker');
    this.received.push(msg);
    this.transfers.push(transfer);
    if (this.autoReply && msg.type) {
      queueMicrotask(() => this.autoReply(msg, this));
    }
  }
  terminate() {
    this.terminated = true;
  }
  emit(type, ev) {
    if (this.terminated) return;
    for (const fn of this.listeners[type]) fn(ev);
  }
  reply(msg) {
    this.emit('message', { data: msg });
  }
  jobs() {
    return this.received.filter((m) => m.type);
  }
  cancels() {
    return this.received.filter((m) => m.cancel);
  }
  last() {
    return this.jobs().at(-1);
  }
}

function setup(options = {}) {
  const clock = createClock();
  const workers = [];
  const pool = new WorkerPool({
    workerFactory: () => {
      const w = new FakeWorker();
      workers.push(w);
      return w;
    },
    maxWorkers: 2,
    now: clock.now,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    ...options,
  });
  return { pool, workers, clock };
}

function settleState(promise) {
  const state = { status: 'pending', value: undefined, count: 0 };
  promise.then(
    (v) => {
      state.status = 'resolved';
      state.value = v;
      state.count++;
    },
    (e) => {
      state.status = 'rejected';
      state.value = e;
      state.count++;
    }
  );
  return state;
}

test('constructor validates options', () => {
  assert.throws(() => new WorkerPool({}), TypeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), maxWorkers: 0 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), maxWorkers: 2, minWorkers: 3 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), overflow: 'drop' }), RangeError);
});

test('defaultPoolSize clamps and reserves headroom', () => {
  assert.equal(defaultPoolSize({ hardwareConcurrency: 16 }), 8);
  assert.equal(defaultPoolSize({ hardwareConcurrency: 6 }), 4);
  assert.equal(defaultPoolSize({ hardwareConcurrency: 1 }), 1);
  assert.equal(defaultPoolSize({ hardwareConcurrency: 16, reserve: 4, max: 6 }), 6);
  assert.equal(defaultPoolSize({ hardwareConcurrency: NaN }), 2);
});

test('lazy spawn, bounded concurrency and result delivery', async () => {
  const { pool, workers, clock } = setup();
  assert.equal(workers.length, 0);
  const a = pool.submit('t', 1);
  const b = pool.submit('t', 2);
  const c = pool.submit('t', 3);
  assert.equal(workers.length, 2);
  assert.equal(pool.stats().busy, 2);
  assert.equal(pool.stats().queued, 1);
  assert.deepEqual(workers[0].last(), { id: 1, type: 't', payload: 1 });

  workers[0].reply({ id: 1, ok: true, result: 'A' });
  assert.equal(await a, 'A');
  // third job moved onto the freed worker, never exceeding two workers
  assert.equal(workers.length, 2);
  assert.equal(workers[0].last().payload, 3);
  workers[1].reply({ id: 2, ok: true, result: 'B' });
  workers[0].reply({ id: 3, ok: true, result: 'C' });
  assert.deepEqual(await Promise.all([b, c]), ['B', 'C']);
  const s = pool.stats();
  assert.equal(s.completed, 3);
  assert.equal(s.workers, 2);
  assert.equal(s.busy, 0);
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('idle workers are shrunk down to minWorkers', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 3, minWorkers: 1, idleTimeoutMs: 100 });
  const jobs = [pool.submit('t', 1), pool.submit('t', 2), pool.submit('t', 3)];
  workers.forEach((w, i) => w.reply({ id: i + 1, ok: true, result: i }));
  await Promise.all(jobs);
  assert.equal(pool.stats().workers, 3);
  clock.advance(99);
  assert.equal(pool.stats().workers, 3);
  clock.advance(1);
  assert.equal(pool.stats().workers, 1);
  assert.equal(workers.filter((w) => w.terminated).length, 2);
  assert.equal(clock.pending, 0);
  await pool.terminate();
});

test('a busy worker is not shrunk and its idle timer is cancelled', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, idleTimeoutMs: 100 });
  const a = pool.submit('t', 1);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await a;
  clock.advance(50);
  const b = pool.submit('t', 2);
  clock.advance(500);
  assert.equal(workers[0].terminated, false);
  workers[0].reply({ id: 2, ok: true, result: 2 });
  await b;
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('FIFO within a priority level and priority ordering across levels', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const blocker = pool.submit('t', 'blocker');
  const order = [];
  const track = (label, p) => p.then(() => order.push(label));
  const all = [
    track('low1', pool.submit('t', 'low1', { priority: 'low' })),
    track('norm1', pool.submit('t', 'norm1')),
    track('high1', pool.submit('t', 'high1', { priority: PRIORITY.HIGH })),
    track('norm2', pool.submit('t', 'norm2', { priority: 'normal' })),
    track('high2', pool.submit('t', 'high2', { priority: 'high' })),
    track('low2', pool.submit('t', 'low2', { priority: 'low' })),
  ];
  const started = [];
  workers[0].reply({ id: workers[0].last().id, ok: true, result: null });
  await blocker;
  for (let i = 0; i < 6; i++) {
    started.push(workers[0].last().payload);
    workers[0].reply({ id: workers[0].last().id, ok: true, result: null });
    await flush();
  }
  await Promise.all(all);
  assert.deepEqual(started, ['high1', 'high2', 'norm1', 'norm2', 'low1', 'low2']);
  assert.deepEqual(order, started);
  await pool.terminate();
});

test('invalid priority and type reject without throwing synchronously', async () => {
  const { pool } = setup();
  await assert.rejects(pool.submit('t', 1, { priority: 'urgent' }), RangeError);
  await assert.rejects(pool.submit('', 1), TypeError);
  assert.equal(pool.stats().submitted, 0);
  await pool.terminate();
});

test('transfer list is forwarded to postMessage', async () => {
  const { pool, workers } = setup();
  const buf = new ArrayBuffer(16);
  const p = pool.submit('inflate', { buf }, { transfer: [buf] });
  assert.equal(workers[0].transfers[0][0], buf);
  workers[0].reply({ id: 1, ok: true, result: 'x' });
  await p;
  await pool.terminate();
});

test('real postMessage transfer detaches the caller buffer (structured clone)', async () => {
  const { pool, workers } = setup();
  const buf = new ArrayBuffer(32);
  const p = pool.submit('t', buf, { transfer: [buf] });
  // The fake does not detach; emulate what a real postMessage does with structuredClone.
  const clone = structuredClone(buf, { transfer: workers[0].transfers[0] });
  assert.equal(buf.byteLength, 0);
  assert.equal(clone.byteLength, 32);
  workers[0].reply({ id: 1, ok: true, result: clone });
  assert.equal(await p, clone);
  await pool.terminate();
});

test('postMessage failure rejects the job only; worker stays usable', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const warm = pool.submit('t', 0);
  workers[0].reply({ id: 1, ok: true, result: 0 });
  await warm;
  const orig = workers[0].postMessage.bind(workers[0]);
  workers[0].postMessage = () => {
    throw new DOMException('cannot clone', 'DataCloneError');
  };
  await assert.rejects(pool.submit('t', () => {}), /cannot clone/);
  workers[0].postMessage = orig;
  const ok = pool.submit('t', 2);
  workers[0].reply({ id: 3, ok: true, result: 'fine' });
  assert.equal(await ok, 'fine');
  assert.equal(pool.stats().crashed, 0);
  await pool.terminate();
});

test('queue limit with reject policy', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, maxQueue: 1 });
  const a = pool.submit('t', 'a');
  const b = pool.submit('t', 'b');
  await assert.rejects(pool.submit('t', 'c'), (e) => e instanceof QueueFullError && e.limit === 1);
  assert.equal(pool.stats().queued, 1);
  assert.equal(pool.stats().rejected, 1);
  workers[0].reply({ id: 1, ok: true, result: 'A' });
  await a;
  workers[0].reply({ id: 2, ok: true, result: 'B' });
  await b;
  await pool.terminate();
});

test('queue limit with wait policy applies backpressure in order', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, maxQueue: 1, overflow: 'wait' });
  const a = pool.submit('t', 'a');
  const b = pool.submit('t', 'b');
  const c = pool.submit('t', 'c');
  const d = pool.submit('t', 'd');
  assert.equal(pool.stats().queued, 1);
  assert.equal(pool.stats().blocked, 2);
  const seen = [];
  for (let i = 1; i <= 4; i++) {
    seen.push(workers[0].last().payload);
    workers[0].reply({ id: i, ok: true, result: i });
    await flush();
  }
  assert.deepEqual(await Promise.all([a, b, c, d]), [1, 2, 3, 4]);
  assert.deepEqual(seen, ['a', 'b', 'c', 'd']);
  await pool.terminate();
});

test('maxQueue 0 only accepts work that can start immediately', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, maxQueue: 0 });
  const a = pool.submit('t', 'a');
  await assert.rejects(pool.submit('t', 'b'), QueueFullError);
  workers[0].reply({ id: 1, ok: true, result: 'A' });
  await a;
  await pool.terminate();
});

test('aborting a blocked job removes it', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, maxQueue: 0, overflow: 'wait' });
  const a = pool.submit('t', 'a');
  const ac = new AbortController();
  const b = pool.submit('t', 'b', { signal: ac.signal });
  assert.equal(pool.stats().blocked, 1);
  ac.abort();
  await assert.rejects(b, JobAbortedError);
  assert.equal(pool.stats().blocked, 0);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await a;
  await pool.terminate();
});

test('per-job timeout rejects, requests cancel, then hard-kills after grace', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, cancelGraceMs: 50 });
  const p = pool.submit('slow', 1, { timeoutMs: 100 });
  const state = settleState(p);
  clock.advance(99);
  await flush();
  assert.equal(state.status, 'pending');
  clock.advance(1);
  await flush();
  assert.equal(state.status, 'rejected');
  assert.ok(state.value instanceof JobTimeoutError);
  assert.equal(state.value.timeoutMs, 100);
  assert.deepEqual(workers[0].cancels(), [{ id: 1, cancel: true }]);
  clock.advance(50);
  assert.equal(workers[0].terminated, true);
  assert.equal(pool.stats().hardKills, 1);
  assert.equal(pool.stats().workers, 0);
  assert.equal(clock.pending, 0);
  await pool.terminate();
});

test('defaultTimeoutMs applies and timeout timer is cleared on success', async () => {
  const { pool, workers, clock } = setup({ defaultTimeoutMs: 500 });
  const p = pool.submit('t', 1);
  assert.equal(clock.pending, 1);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await p;
  assert.equal(clock.pending, 0);
  await pool.terminate();
});

test('abort while queued rejects immediately and never reaches a worker', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 'a');
  const ac = new AbortController();
  const b = pool.submit('t', 'b', { signal: ac.signal });
  ac.abort(new Error('nope'));
  await assert.rejects(b, (e) => e.name === 'AbortError' && e instanceof JobAbortedError && e.reason.message === 'nope');
  assert.equal(pool.stats().queued, 0);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await a;
  assert.equal(workers[0].jobs().length, 1);
  assert.equal(pool.stats().cancelled, 1);
  await pool.terminate();
});

test('already-aborted signal rejects at submit', async () => {
  const { pool, workers } = setup();
  const ac = new AbortController();
  ac.abort();
  await assert.rejects(pool.submit('t', 1, { signal: ac.signal }), JobAbortedError);
  assert.equal(workers.length, 0);
  await pool.terminate();
});

test('abort while running: cooperative ack keeps the worker alive', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, cancelGraceMs: 100 });
  const ac = new AbortController();
  const p = pool.submit('t', 1, { signal: ac.signal });
  const queued = pool.submit('t', 2);
  const state = settleState(p);
  ac.abort();
  await flush();
  assert.equal(state.status, 'rejected');
  assert.equal(state.count, 1);
  assert.deepEqual(workers[0].cancels(), [{ id: 1, cancel: true }]);
  // worker is occupied until it acknowledges; queued job waits
  assert.equal(workers[0].jobs().length, 1);
  workers[0].reply({ id: 1, cancelled: true });
  assert.equal(workers[0].jobs().length, 2);
  assert.equal(workers[0].terminated, false);
  clock.advance(1000); // grace timer must have been cleared
  assert.equal(workers[0].terminated, false);
  workers[0].reply({ id: 2, ok: true, result: 'q' });
  assert.equal(await queued, 'q');
  assert.equal(state.count, 1);
  assert.equal(pool.stats().hardKills, 0);
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('abort while running: a late result also counts as acknowledgement and is dropped', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const ac = new AbortController();
  const p = pool.submit('t', 1, { signal: ac.signal });
  ac.abort();
  await assert.rejects(p, JobAbortedError);
  workers[0].reply({ id: 1, ok: true, result: 'too late' });
  assert.equal(pool.stats().busy, 0);
  assert.equal(pool.stats().completed, 0);
  await pool.terminate();
});

test('abort while running: no ack within grace hard-kills and replaces the worker', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, cancelGraceMs: 100 });
  const ac = new AbortController();
  const p = pool.submit('t', 1, { signal: ac.signal });
  const next = pool.submit('t', 2);
  const state = settleState(p);
  ac.abort();
  await flush();
  assert.equal(state.status, 'rejected');
  clock.advance(100);
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);
  assert.equal(workers[1].last().payload, 2);
  workers[1].reply({ id: 2, ok: true, result: 'next' });
  assert.equal(await next, 'next');
  assert.equal(state.count, 1);
  assert.equal(pool.stats().hardKills, 1);
  assert.equal(pool.stats().crashed, 0);
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('cancelGraceMs 0 kills the worker immediately', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, cancelGraceMs: 0 });
  const ac = new AbortController();
  const p = pool.submit('t', 1, { signal: ac.signal });
  ac.abort();
  await assert.rejects(p, JobAbortedError);
  assert.equal(workers[0].terminated, true);
  await pool.terminate();
});

test('worker error mid-job rejects with WorkerCrashError, replaces worker, queue continues', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 'a');
  const b = pool.submit('t', 'b');
  workers[0].emit('error', { message: 'boom' });
  await assert.rejects(a, (e) => e instanceof WorkerCrashError && e.reason === 'boom' && e instanceof WorkerPoolError);
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);
  assert.equal(workers[1].last().payload, 'b');
  workers[1].reply({ id: 2, ok: true, result: 'B' });
  assert.equal(await b, 'B');
  const s = pool.stats();
  assert.equal(s.crashed, 1);
  assert.equal(s.failed, 1);
  assert.equal(s.completed, 1);
  await pool.terminate();
});

test('messageerror and exit events are treated as crashes', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 'a');
  workers[0].emit('messageerror', {});
  await assert.rejects(a, (e) => e instanceof WorkerCrashError && /messageerror/.test(e.reason));
  const b = pool.submit('t', 'b');
  workers[1].emit('exit', { code: 1 });
  await assert.rejects(b, (e) => e instanceof WorkerCrashError && /exited/.test(e.reason));
  assert.equal(pool.stats().crashed, 2);
  await pool.terminate();
});

test('idle worker crash is replaced lazily and not attributed to any job', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 'a');
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await a;
  workers[0].emit('error', { message: 'idle boom' });
  assert.equal(pool.stats().workers, 0);
  const b = pool.submit('t', 'b');
  assert.equal(workers.length, 2);
  workers[1].reply({ id: 2, ok: true, result: 2 });
  assert.equal(await b, 2);
  await pool.terminate();
});

test('retry policy retries crashes but never job-level errors', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, retries: 2, breaker: false });
  const p = pool.submit('t', 'x');
  workers[0].emit('error', { message: 'c1' });
  assert.equal(workers.length, 2);
  workers[1].emit('error', { message: 'c2' });
  assert.equal(workers.length, 3);
  workers[2].reply({ id: 1, ok: true, result: 'third time' });
  assert.equal(await p, 'third time');
  assert.equal(pool.stats().retried, 2);

  const q = pool.submit('t', 'y');
  workers[2].emit('error', { message: 'c3' });
  workers[3].emit('error', { message: 'c4' });
  workers[4].emit('error', { message: 'c5' });
  await assert.rejects(q, WorkerCrashError);

  const r = pool.submit('t', 'z');
  const spawned = workers.length;
  const w = workers.at(-1);
  w.reply({ id: w.last().id, ok: false, error: { name: 'Error', message: 'job failed' } });
  await assert.rejects(r, /job failed/);
  assert.equal(workers.length, spawned); // a job-level error neither retries nor respawns
  assert.equal(pool.stats().retried, 4);
  await pool.terminate();
});

test('jobs with a transfer list are not retried after a crash', async () => {
  const { pool, workers } = setup({ maxWorkers: 1, retries: 3 });
  const buf = new ArrayBuffer(8);
  const p = pool.submit('t', buf, { transfer: [buf] });
  workers[0].emit('error', { message: 'boom' });
  await assert.rejects(p, WorkerCrashError);
  assert.equal(pool.stats().retried, 0);
  await pool.terminate();
});

test('circuit breaker opens after K crashes in the window and stops respawning', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, breaker: { maxCrashes: 3, windowMs: 1000 } });
  const jobs = [pool.submit('t', 1), pool.submit('t', 2), pool.submit('t', 3), pool.submit('t', 4)];
  const settled = jobs.map(settleState);
  workers[0].emit('error', { message: 'a' });
  workers[1].emit('error', { message: 'b' });
  assert.equal(pool.stats().circuitOpen, false);
  workers[2].emit('error', { message: 'c' });
  await flush();
  assert.equal(pool.stats().circuitOpen, true);
  assert.equal(workers.length, 3); // no respawn
  assert.ok(settled[0].value instanceof WorkerCrashError);
  assert.ok(settled[1].value instanceof WorkerCrashError);
  assert.ok(settled[2].value instanceof WorkerCrashError);
  // queued job 4 was rejected by the breaker, not left hanging
  assert.ok(settled[3].value instanceof CircuitOpenError);
  assert.equal(settled[3].value.crashes, 3);
  await assert.rejects(pool.submit('t', 5), CircuitOpenError);
  assert.equal(workers.length, 3);
  pool.resetCircuit();
  const ok = pool.submit('t', 6);
  assert.equal(workers.length, 4);
  workers[3].reply({ id: 5, ok: true, result: 'recovered' });
  assert.equal(await ok, 'recovered');
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('crashes outside the breaker window do not accumulate', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, breaker: { maxCrashes: 2, windowMs: 100 } });
  const a = pool.submit('t', 1);
  workers[0].emit('error', { message: 'a' });
  await assert.rejects(a, WorkerCrashError);
  clock.advance(200);
  const b = pool.submit('t', 2);
  workers[1].emit('error', { message: 'b' });
  await assert.rejects(b, WorkerCrashError);
  assert.equal(pool.stats().circuitOpen, false);
  await pool.terminate();
});

test('worker factory failure cannot spin forever', async () => {
  let calls = 0;
  const pool = new WorkerPool({
    workerFactory: () => {
      calls++;
      throw new Error('script missing');
    },
    maxWorkers: 2,
    breaker: { maxCrashes: 3, windowMs: 1000 },
  });
  const results = await Promise.allSettled([pool.submit('t', 1), pool.submit('t', 2), pool.submit('t', 3), pool.submit('t', 4)]);
  assert.ok(calls <= 4);
  assert.ok(results.every((r) => r.status === 'rejected'));
  assert.ok(results[0].reason instanceof WorkerCrashError);
  assert.match(results[0].reason.reason, /script missing/);
  await pool.terminate();
  // and with the breaker disabled a failing factory still terminates the pump loop
  const pool2 = new WorkerPool({
    workerFactory: () => {
      throw new Error('nope');
    },
    breaker: false,
  });
  await assert.rejects(Promise.all([pool2.submit('t', 1), pool2.submit('t', 2)]), WorkerCrashError);
  await pool2.terminate();
});

test('error envelope round-trips into a real Error', async () => {
  const { pool, workers } = setup();
  const p = pool.submit('t', 1);
  const err = new RangeError('bad range');
  err.code = 'E_RANGE';
  workers[0].reply({ id: 1, ok: false, error: serializeError(err) });
  await assert.rejects(p, (e) => {
    assert.ok(e instanceof RangeError);
    assert.equal(e.message, 'bad range');
    assert.equal(e.code, 'E_RANGE');
    assert.equal(e.remote, true);
    assert.equal(e.stack, err.stack);
    return true;
  });
  const custom = deserializeError({ name: 'InflateError', message: 'corrupt', stack: 'S' });
  assert.ok(custom instanceof Error);
  assert.equal(custom.name, 'InflateError');
  assert.deepEqual(serializeError('str'), { name: 'Error', message: 'str' });
  assert.equal(pool.stats().crashed, 0);
  await pool.terminate();
});

test('progress messages reach onProgress; a throwing callback is contained', async () => {
  const { pool, workers } = setup();
  const seen = [];
  const p = pool.submit('t', 1, {
    onProgress: (v) => {
      seen.push(v);
      throw new Error('callback bug');
    },
  });
  workers[0].reply({ id: 1, progress: 0.25 });
  workers[0].reply({ id: 1, progress: 0.5 });
  workers[0].reply({ id: 1, ok: true, result: 'done' });
  workers[0].reply({ id: 1, progress: 1 }); // after settle: ignored
  assert.equal(await p, 'done');
  assert.deepEqual(seen, [0.25, 0.5]);
  await pool.terminate();
});

test('unknown, late, duplicate and malformed messages are ignored', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const p = pool.submit('t', 1);
  workers[0].reply({ id: 999, ok: true, result: 'stray' });
  workers[0].reply(null);
  workers[0].reply('junk');
  workers[0].reply({ ok: true, result: 'no id' });
  assert.equal(pool.stats().completed, 0);
  workers[0].reply({ id: 1, ok: true, result: 'real' });
  workers[0].reply({ id: 1, ok: true, result: 'duplicate' });
  workers[0].reply({ id: 1, ok: false, error: { message: 'dup err' } });
  assert.equal(await p, 'real');
  const s = pool.stats();
  assert.equal(s.completed, 1);
  assert.equal(s.failed, 0);
  await pool.terminate();
});

test('events from a replaced worker are ignored', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 'a');
  const b = pool.submit('t', 'b');
  const old = workers[0];
  old.emit('error', { message: 'x' });
  await assert.rejects(a, WorkerCrashError);
  // terminated worker's emit is a no-op in the fake; invoke listeners directly to prove the pool guards
  for (const fn of old.listeners.message) fn({ data: { id: 2, ok: true, result: 'stale' } });
  for (const fn of old.listeners.error) fn({ message: 'again' });
  assert.equal(pool.stats().crashed, 1);
  workers[1].reply({ id: 2, ok: true, result: 'B' });
  assert.equal(await b, 'B');
  await pool.terminate();
});

test('drain resolves when idle', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  await pool.drain();
  const a = pool.submit('t', 1);
  const b = pool.submit('t', 2);
  let drained = false;
  const d = pool.drain().then(() => {
    drained = true;
  });
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await flush();
  assert.equal(drained, false);
  workers[0].reply({ id: 2, ok: true, result: 2 });
  await d;
  await Promise.all([a, b]);
  assert.equal(drained, true);
  await pool.terminate();
});

test('graceful terminate finishes outstanding work, then rejects new submits', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 1);
  const b = pool.submit('t', 2);
  let done = false;
  const t = pool.terminate({ graceful: true }).then(() => {
    done = true;
  });
  await assert.rejects(pool.submit('t', 3), PoolTerminatedError);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await flush();
  assert.equal(done, false);
  workers[0].reply({ id: 2, ok: true, result: 2 });
  await t;
  assert.deepEqual(await Promise.all([a, b]), [1, 2]);
  assert.equal(workers[0].terminated, true);
  assert.equal(pool.stats().workers, 0);
  assert.equal(clock.pending, 0);
});

test('graceful terminate on an idle pool resolves immediately', async () => {
  const { pool, workers } = setup();
  const a = pool.submit('t', 1);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await a;
  await pool.terminate({ graceful: true });
  assert.equal(workers[0].terminated, true);
});

test('immediate terminate rejects queued and running jobs and kills workers', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, maxQueue: 1, overflow: 'wait', defaultTimeoutMs: 1000 });
  const ac = new AbortController();
  const running = settleState(pool.submit('t', 1, { signal: ac.signal }));
  const queued = settleState(pool.submit('t', 2));
  const blocked = settleState(pool.submit('t', 3));
  const dr = pool.drain();
  await pool.terminate();
  await dr;
  await flush();
  for (const s of [running, queued, blocked]) {
    assert.equal(s.status, 'rejected');
    assert.ok(s.value instanceof PoolTerminatedError);
    assert.equal(s.count, 1);
  }
  assert.equal(workers[0].terminated, true);
  assert.equal(clock.pending, 0);
  ac.abort(); // listener was removed; must be a no-op
  await assert.rejects(pool.submit('t', 4), PoolTerminatedError);
});

test('terminate is idempotent and can upgrade a graceful shutdown', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const a = settleState(pool.submit('t', 1));
  const t1 = pool.terminate({ graceful: true });
  const t2 = pool.terminate();
  assert.equal(t1, t2);
  await t2;
  await flush();
  assert.equal(a.status, 'rejected');
  assert.equal(workers[0].terminated, true);
  await pool.terminate();
  await pool.terminate({ graceful: true });
});

test('terminate cancels a worker that was mid hard-cancel without leaking timers', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, cancelGraceMs: 500 });
  const ac = new AbortController();
  const p = pool.submit('t', 1, { signal: ac.signal });
  ac.abort();
  await assert.rejects(p, JobAbortedError);
  assert.equal(clock.pending, 1);
  await pool.terminate();
  assert.equal(workers[0].terminated, true);
  assert.equal(clock.pending, 0);
});

test('stats report average wait and run time from the injected clock', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1 });
  const a = pool.submit('t', 1);
  const b = pool.submit('t', 2);
  clock.advance(10);
  workers[0].reply({ id: 1, ok: true, result: 1 }); // a: wait 0, run 10; b starts: wait 10
  await a;
  clock.advance(30);
  workers[0].reply({ id: 2, ok: true, result: 2 }); // b: run 30
  await b;
  const s = pool.stats();
  assert.equal(s.avgWaitMs, 5);
  assert.equal(s.avgRunMs, 20);
  assert.equal(s.submitted, 2);
  assert.equal(s.spawned, 1);
  await pool.terminate();
});

test('every job settles exactly once across mixed outcomes', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 2, retries: 1, cancelGraceMs: 10 });
  const ac = new AbortController();
  const states = [
    settleState(pool.submit('t', 'ok')),
    settleState(pool.submit('t', 'abort', { signal: ac.signal })),
    settleState(pool.submit('t', 'queued-abort', { signal: ac.signal })),
    settleState(pool.submit('t', 'tmo', { timeoutMs: 5 })),
  ];
  workers[0].reply({ id: 1, ok: true, result: 1 });
  await flush();
  ac.abort();
  clock.advance(5);
  clock.advance(100);
  workers[0].reply({ id: 1, ok: true, result: 'dup' });
  workers[1].reply({ id: 2, cancelled: true });
  await pool.terminate();
  await flush();
  for (const s of states) assert.equal(s.count, 1);
  assert.equal(clock.pending, 0);
});

// ---- end-to-end against serveWorkerPool through in-memory ports -----------

/** Build a pair of connected message scopes with microtask delivery. */
function createPortPair() {
  const make = () => ({ listeners: new Set(), peer: null, closed: false });
  const a = make();
  const b = make();
  a.peer = b;
  b.peer = a;
  const wrap = (side) => ({
    addEventListener(type, fn) {
      if (type === 'message') side.listeners.add(fn);
    },
    removeEventListener(type, fn) {
      side.listeners.delete(fn);
    },
    postMessage(msg, transfer = []) {
      if (side.closed) return;
      const data = structuredClone(msg, { transfer });
      setImmediate(() => {
        if (side.peer.closed) return;
        for (const fn of [...side.peer.listeners]) fn({ data });
      });
    },
  });
  return { pool: wrap(a), worker: wrap(b), sideA: a, sideB: b };
}

/** FakeWorker-like adapter whose far end is a real serveWorkerPool scope. */
function createLoopbackFactory(handlers, created = []) {
  return () => {
    const { pool: poolPort, worker: workerPort, sideB } = createPortPair();
    serveWorkerPool(handlers, { scope: workerPort });
    const listeners = { message: [], error: [], messageerror: [], exit: [] };
    poolPort.addEventListener('message', (ev) => listeners.message.forEach((fn) => fn(ev)));
    const w = {
      terminated: false,
      addEventListener(type, fn) {
        listeners[type].push(fn);
      },
      postMessage: (msg, transfer) => poolPort.postMessage(msg, transfer),
      terminate() {
        w.terminated = true;
        sideB.closed = true;
      },
    };
    created.push(w);
    return w;
  };
}

test('e2e: serveWorkerPool handles jobs, transfers, progress and errors', async () => {
  const handlers = {
    sum: async (payload) => payload.reduce((a, b) => a + b, 0),
    inflate: async (payload, ctx) => {
      const out = new Uint8Array(payload.buf.byteLength * 2);
      ctx.progress(0.5);
      ctx.transfer(out.buffer);
      return { size: out.length, buf: out.buffer };
    },
    wrapped: async () => withTransfer({ tag: 'w' }, []),
    plain: async () => ({ result: 'unwrapped', transfer: [] }),
    boom: async () => {
      const e = new TypeError('bad input');
      e.code = 'E_BAD';
      throw e;
    },
  };
  const created = [];
  const pool = new WorkerPool({ workerFactory: createLoopbackFactory(handlers, created), maxWorkers: 2 });
  assert.equal(await pool.submit('sum', [1, 2, 3]), 6);

  const input = new Uint8Array([1, 2, 3, 4]).buffer;
  const progress = [];
  const res = await pool.submit('inflate', { buf: input }, { transfer: [input], onProgress: (v) => progress.push(v) });
  assert.equal(input.byteLength, 0); // detached by transfer
  assert.equal(res.size, 8);
  assert.equal(res.buf.byteLength, 8);
  assert.deepEqual(progress, [0.5]);

  assert.deepEqual(await pool.submit('wrapped', null), { tag: 'w' });
  assert.equal(await pool.submit('plain', null), 'unwrapped');

  await assert.rejects(pool.submit('boom', null), (e) => e instanceof TypeError && e.code === 'E_BAD' && e.remote === true);
  await assert.rejects(pool.submit('nope', null), (e) => e instanceof TypeError && /Unknown job type/.test(e.message));
  assert.equal(pool.stats().crashed, 0);
  await pool.terminate();
});

test('e2e: cooperative cancellation through ctx.signal keeps the worker', async () => {
  let observed = false;
  const handlers = {
    wait: (_p, ctx) =>
      new Promise((resolve) => {
        ctx.signal.addEventListener('abort', () => {
          observed = true;
          resolve('ignored');
        });
      }),
    quick: async () => 'quick',
  };
  const created = [];
  const pool = new WorkerPool({
    workerFactory: createLoopbackFactory(handlers, created),
    maxWorkers: 1,
    cancelGraceMs: 10000,
  });
  const ac = new AbortController();
  const p = pool.submit('wait', null, { signal: ac.signal });
  const q = pool.submit('quick', null);
  await flush();
  ac.abort();
  await assert.rejects(p, JobAbortedError);
  assert.equal(await q, 'quick');
  assert.equal(observed, true);
  assert.equal(created.length, 1);
  assert.equal(created[0].terminated, false);
  assert.equal(pool.stats().hardKills, 0);
  await pool.terminate();
});

test('e2e: a handler that ignores cancellation is hard-killed after the grace period', async () => {
  const clock = createClock();
  const handlers = { hang: () => new Promise(() => {}), quick: async () => 'quick' };
  const created = [];
  const pool = new WorkerPool({
    workerFactory: createLoopbackFactory(handlers, created),
    maxWorkers: 1,
    cancelGraceMs: 50,
    now: clock.now,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
  });
  const p = pool.submit('hang', null, { timeoutMs: 20 });
  const q = pool.submit('quick', null);
  await flush();
  clock.advance(20);
  await assert.rejects(p, JobTimeoutError);
  await flush();
  clock.advance(50);
  assert.equal(created[0].terminated, true);
  assert.equal(await q, 'quick');
  assert.equal(created.length, 2);
  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('serveWorkerPool ignores duplicate ids, malformed messages and unknown cancels; unsubscribe works', async () => {
  const sent = [];
  const listeners = new Set();
  const scope = {
    addEventListener: (t, fn) => listeners.add(fn),
    removeEventListener: (t, fn) => listeners.delete(fn),
    postMessage: (m) => sent.push(m),
  };
  let runs = 0;
  let release;
  const stop = serveWorkerPool(
    {
      slow: () => {
        runs++;
        return new Promise((r) => (release = r));
      },
    },
    { scope }
  );
  const send = (data) => listeners.forEach((fn) => fn({ data }));
  send(null);
  send({ type: 'slow' }); // no id
  send({ id: 1, type: 'slow', payload: 0 });
  send({ id: 1, type: 'slow', payload: 0 }); // duplicate while active
  send({ id: 77, cancel: true }); // unknown id: acked, harmless
  assert.deepEqual(sent, [{ id: 77, cancelled: true }]);
  release('v');
  await flush();
  assert.equal(runs, 1);
  assert.deepEqual(sent.at(-1), { id: 1, ok: true, result: 'v' });
  stop();
  assert.equal(listeners.size, 0);
});

test('serveWorkerPool reports a result that cannot be cloned as a job error', async () => {
  const sent = [];
  const listeners = new Set();
  const scope = {
    addEventListener: (t, fn) => listeners.add(fn),
    postMessage: (m) => {
      if (m.ok === true) throw new DOMException('no clone', 'DataCloneError');
      sent.push(m);
    },
  };
  serveWorkerPool({ bad: async () => () => {} }, { scope });
  listeners.forEach((fn) => fn({ data: { id: 5, type: 'bad', payload: null } }));
  await flush();
  assert.equal(sent[0].id, 5);
  assert.equal(sent[0].ok, false);
  assert.equal(sent[0].error.name, 'DataCloneError');
});

// ---- W140: taskTimeoutMs watchdog, respawn cap & saturation telemetry -----

test('W140: constructor validates taskTimeoutMs and maxRespawnsPerSlot', () => {
  assert.equal(DEFAULT_MAX_RESPAWNS_PER_SLOT, 16);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), taskTimeoutMs: -1 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), taskTimeoutMs: NaN }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), maxRespawnsPerSlot: -1 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), maxRespawnsPerSlot: 1.5 }), RangeError);
});

test('W140: hung task times out with TASK_TIMEOUT, respawns worker, and runs queued task', async () => {
  const { pool, workers, clock } = setup({
    maxWorkers: 1,
    taskTimeoutMs: 50,
    cancelGraceMs: 0,
  });

  const hung = pool.submit('hang', 'job-1');
  const next = pool.submit('work', 'job-2');
  assert.equal(workers.length, 1);
  assert.equal(pool.getStats().peakQueueDepth, 1);

  clock.advance(50);
  await assert.rejects(
    hung,
    (err) => err instanceof JobTimeoutError && err.code === 'TASK_TIMEOUT' && err.timeoutMs === 50
  );

  // Wedged worker was terminated and replacement worker took the queued job.
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);
  assert.equal(workers[1].last().payload, 'job-2');

  workers[1].reply({ id: 2, ok: true, result: 'ok-2' });
  assert.equal(await next, 'ok-2');

  const stats = pool.getStats();
  assert.deepEqual(stats, pool.stats());
  assert.equal(stats.timedOutTasks, 1);
  assert.equal(stats.respawnedWorkers, 1);
  assert.equal(stats.peakQueueDepth, 1);
  assert.equal(stats.hardKills, 1);
  assert.equal(stats.completed, 1);
  assert.equal(clock.pending, 0);

  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('W140: per-task options.timeoutMs overrides or disables pool taskTimeoutMs', async () => {
  const { pool, workers, clock } = setup({
    maxWorkers: 1,
    taskTimeoutMs: 100,
    cancelGraceMs: 0,
  });

  // Shorter per-task override fires at 25ms instead of 100ms.
  const shortTask = pool.submit('slow', 1, { timeoutMs: 25 });
  clock.advance(25);
  await assert.rejects(
    shortTask,
    (err) => err instanceof JobTimeoutError && err.code === 'TASK_TIMEOUT' && err.timeoutMs === 25
  );
  assert.equal(pool.getStats().timedOutTasks, 1);

  // timeoutMs: 0 disables timeout even when pool taskTimeoutMs > 0.
  const noTimeoutTask = pool.submit('long', 2, { timeoutMs: 0 });
  assert.equal(workers.length, 2);
  assert.equal(pool.getStats().respawnedWorkers, 1);
  clock.advance(500);
  assert.equal(clock.pending, 0);
  workers[1].reply({ id: 2, ok: true, result: 'finished-late' });
  assert.equal(await noTimeoutTask, 'finished-late');
  assert.equal(pool.getStats().timedOutTasks, 1);

  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('W140: worker error triggers clean respawn, drains queued work, and honors maxRespawnsPerSlot', async () => {
  const { pool, workers, clock } = setup({
    maxWorkers: 1,
    maxRespawnsPerSlot: 2,
    breaker: false,
  });

  const t1 = pool.submit('t', 1);
  const t2 = pool.submit('t', 2);
  const t3 = pool.submit('t', 3);
  const t4 = pool.submit('t', 4);
  assert.equal(pool.getStats().peakQueueDepth, 3);

  // Crash #1 -> respawn #1 picks up t2.
  workers[0].emit('error', { message: 'crash-1' });
  await assert.rejects(t1, WorkerCrashError);
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);
  assert.equal(workers[1].last().payload, 2);
  assert.equal(pool.getStats().respawnedWorkers, 1);

  // Crash #2 -> respawn #2 picks up t3.
  workers[1].emit('error', { message: 'crash-2' });
  await assert.rejects(t2, WorkerCrashError);
  assert.equal(workers.length, 3);
  assert.equal(workers[2].last().payload, 3);
  assert.equal(pool.getStats().respawnedWorkers, 2);

  // Crash #3 -> slot exhausted maxRespawnsPerSlot (2); no 4th worker spawned, t4 fails cleanly.
  workers[2].emit('error', { message: 'crash-3' });
  await assert.rejects(t3, WorkerCrashError);
  await assert.rejects(t4, (err) => err instanceof WorkerCrashError && /respawn limit/.test(err.reason));
  assert.equal(workers.length, 3);
  assert.equal(pool.getStats().respawnedWorkers, 2);
  assert.equal(pool.getStats().workers, 0);

  // resetCircuit() restores respawn budget.
  pool.resetCircuit();
  const t5 = pool.submit('t', 5);
  assert.equal(workers.length, 4);
  workers[3].reply({ id: 5, ok: true, result: 'recovered' });
  assert.equal(await t5, 'recovered');

  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('W140: worker error with minWorkers > 0 eagerly respawns and idle shrink does not count as respawn', async () => {
  const { pool, workers, clock } = setup({
    maxWorkers: 2,
    minWorkers: 1,
    idleTimeoutMs: 100,
  });

  // Spawn 2 workers, complete both, then let surplus worker shrink via idle timeout.
  const a = pool.submit('t', 1);
  const b = pool.submit('t', 2);
  workers[0].reply({ id: 1, ok: true, result: 1 });
  workers[1].reply({ id: 2, ok: true, result: 2 });
  await Promise.all([a, b]);
  clock.advance(100);
  assert.equal(pool.getStats().workers, 1);
  assert.equal(pool.getStats().respawnedWorkers, 0);

  // Re-expanding after normal idle shrink does NOT count as a crash/timeout respawn.
  const c = pool.submit('t', 3);
  const d = pool.submit('t', 4);
  assert.equal(pool.getStats().workers, 2);
  assert.equal(pool.getStats().respawnedWorkers, 0);
  const alive = workers.filter((w) => !w.terminated);
  alive[0].reply({ id: alive[0].last().id, ok: true, result: 'c' });
  alive[1].reply({ id: alive[1].last().id, ok: true, result: 'd' });
  await Promise.all([c, d]);

  // Shrink back down to minWorkers (1), then crash that sole remaining worker.
  clock.advance(100);
  assert.equal(pool.getStats().workers, 1);
  const sole = workers.find((w) => !w.terminated);
  sole.emit('error', { message: 'unexpected crash' });
  // Eagerly respawned back to minWorkers === 1.
  assert.equal(pool.getStats().workers, 1);
  assert.equal(pool.getStats().respawnedWorkers, 1);

  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('W140: peakQueueDepth tracks max pending queue length across normal, reject, and wait policies', async () => {
  const { pool, workers } = setup({ maxWorkers: 2 });
  // First 2 jobs dispatch immediately without waiting in queue.
  const j1 = pool.submit('t', 1);
  const j2 = pool.submit('t', 2);
  assert.equal(pool.getStats().peakQueueDepth, 0);

  // Next 3 jobs queue up.
  const j3 = pool.submit('t', 3);
  const j4 = pool.submit('t', 4);
  const j5 = pool.submit('t', 5);
  assert.equal(pool.getStats().queued, 3);
  assert.equal(pool.getStats().peakQueueDepth, 3);

  // Complete all jobs; peakQueueDepth remains 3.
  workers[0].reply({ id: 1, ok: true, result: 1 });
  workers[1].reply({ id: 2, ok: true, result: 2 });
  workers[0].reply({ id: 3, ok: true, result: 3 });
  workers[1].reply({ id: 4, ok: true, result: 4 });
  workers[0].reply({ id: 5, ok: true, result: 5 });
  await Promise.all([j1, j2, j3, j4, j5]);
  assert.equal(pool.getStats().queued, 0);
  assert.equal(pool.getStats().peakQueueDepth, 3);
  await pool.terminate();

  // Rejected overflow jobs must not inflate peakQueueDepth.
  const { pool: rejectPool, workers: rWorkers } = setup({ maxWorkers: 1, maxQueue: 1, overflow: 'reject' });
  const r1 = rejectPool.submit('t', 1);
  const r2 = rejectPool.submit('t', 2);
  await assert.rejects(rejectPool.submit('t', 3), QueueFullError);
  assert.equal(rejectPool.getStats().peakQueueDepth, 1);
  rWorkers[0].reply({ id: 1, ok: true, result: 1 });
  rWorkers[0].reply({ id: 2, ok: true, result: 2 });
  await Promise.all([r1, r2]);
  await rejectPool.terminate();

  // Wait overflow policy includes blocked pending tasks in peakQueueDepth.
  const { pool: waitPool, workers: wWorkers } = setup({ maxWorkers: 1, maxQueue: 1, overflow: 'wait' });
  const w1 = waitPool.submit('t', 1);
  const w2 = waitPool.submit('t', 2);
  const w3 = waitPool.submit('t', 3);
  const w4 = waitPool.submit('t', 4);
  assert.equal(waitPool.getStats().peakQueueDepth, 3);
  for (let i = 1; i <= 4; i++) {
    wWorkers[0].reply({ id: i, ok: true, result: i });
    await flush();
  }
  await Promise.all([w1, w2, w3, w4]);
  assert.equal(waitPool.getStats().peakQueueDepth, 3);
  await waitPool.terminate();
});

test('W140: terminate() and destroy() clear all pending watchdog timers with zero leaked handles', async () => {
  // Manual clock: verify all timers (timeout, grace, idle) are cleared.
  const { pool, workers, clock } = setup({
    maxWorkers: 2,
    taskTimeoutMs: 5000,
    cancelGraceMs: 5000,
    idleTimeoutMs: 5000,
  });
  const running = settleState(pool.submit('t', 1));
  const doneJob = pool.submit('t', 2);
  workers[1].reply({ id: 2, ok: true, result: 'done' });
  await doneJob;
  // Worker 0 has task timeout timer, worker 1 has idle timer.
  assert.equal(clock.pending, 2);
  await pool.destroy();
  await flush();
  assert.equal(running.status, 'rejected');
  assert.ok(running.value instanceof PoolTerminatedError);
  assert.equal(clock.pending, 0);
  assert.equal(workers[0].terminated, true);
  assert.equal(workers[1].terminated, true);

  // Real Node timers: verify destroy/terminate clear long-lived setTimeouts cleanly.
  let activeTimers = 0;
  const realTimers = new Set();
  const realPool = new WorkerPool({
    workerFactory: () => new FakeWorker(),
    maxWorkers: 2,
    taskTimeoutMs: 60000,
    cancelGraceMs: 60000,
    idleTimeoutMs: 60000,
    setTimeout: (fn, ms) => {
      activeTimers++;
      const handle = globalThis.setTimeout(() => {
        realTimers.delete(handle);
        activeTimers--;
        fn();
      }, ms);
      realTimers.add(handle);
      return handle;
    },
    clearTimeout: (handle) => {
      if (realTimers.delete(handle)) activeTimers--;
      globalThis.clearTimeout(handle);
    },
  });
  const p1 = settleState(realPool.submit('hang', 1));
  const p2 = settleState(realPool.submit('hang', 2));
  assert.equal(activeTimers, 2);
  await realPool.terminate();
  await flush();
  assert.equal(p1.status, 'rejected');
  assert.equal(p2.status, 'rejected');
  assert.equal(activeTimers, 0);
  assert.equal(realTimers.size, 0);
});

// ---- W218: numeric priority scheduling, warm reuse & queue-depth telemetry --

test('W218: numeric priority orders higher priority first with strict FIFO tie-breaking', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const blocker = pool.submit('t', 'blocker');

  const submissions = [
    { label: 'p-15-a', opts: { priority: -15 } },
    { label: 'low-a', opts: { priority: 'low' } },
    { label: 'default-a', opts: {} },
    { label: 'p5-a', opts: { priority: 5 } },
    { label: 'p0-b', opts: { priority: 0 } },
    { label: 'p50-a', opts: { priority: 50 } },
    { label: 'high-a', opts: { priority: 'high' } },
    { label: 'p5-b', opts: { priority: 5 } },
    { label: 'p10-b', opts: { priority: 10 } },
    { label: 'p2.5', opts: { priority: 2.5 } },
    { label: 'norm-c', opts: { priority: 'normal' } },
    { label: 'p50-b', opts: { priority: 50 } },
    { label: 'p-15-b', opts: { priority: -15 } },
  ];

  const promises = submissions.map((s) => pool.submit('t', s.label, s.opts));
  assert.equal(pool.getStats().queuedTasks, submissions.length);
  assert.equal(pool.getStats().peakQueuedTasks, submissions.length);

  workers[0].reply({ id: workers[0].last().id, ok: true, result: 'done' });
  await blocker;

  const started = [];
  for (let i = 0; i < submissions.length; i++) {
    const job = workers[0].last();
    started.push(job.payload);
    workers[0].reply({ id: job.id, ok: true, result: job.payload });
    await flush();
  }
  await Promise.all(promises);

  assert.deepEqual(started, [
    'p50-a',
    'p50-b',
    'high-a',
    'p10-b',
    'p5-a',
    'p5-b',
    'p2.5',
    'default-a',
    'p0-b',
    'norm-c',
    'low-a',
    'p-15-a',
    'p-15-b',
  ]);
  await pool.terminate();
});

test('W218: rejects non-finite priorities and invalid per-task timeoutMs without throwing synchronously', async () => {
  const { pool } = setup();
  await assert.rejects(pool.submit('t', 1, { priority: NaN }), RangeError);
  await assert.rejects(pool.submit('t', 1, { priority: Infinity }), RangeError);
  await assert.rejects(pool.submit('t', 1, { priority: -Infinity }), RangeError);
  await assert.rejects(pool.submit('t', 1, { priority: {} }), RangeError);
  await assert.rejects(pool.submit('t', 1, { timeoutMs: -5 }), RangeError);
  await assert.rejects(pool.submit('t', 1, { timeoutMs: NaN }), RangeError);
  assert.equal(pool.getStats().submitted, 0);
  assert.equal(pool.getStats().queuedTasks, 0);
  await pool.terminate();
});

test('W218: cancelling queued tasks across heap positions preserves priority and FIFO ordering', async () => {
  const { pool, workers } = setup({ maxWorkers: 1 });
  const blocker = pool.submit('t', 'blocker');

  const acRoot = new AbortController();
  const acMid = new AbortController();
  const acLeaf = new AbortController();

  const p1 = pool.submit('t', 'p10-first', { priority: 10, signal: acRoot.signal });
  const p2 = pool.submit('t', 'p10-second', { priority: 10 });
  const p3 = pool.submit('t', 'p5-first', { priority: 5 });
  const p4 = pool.submit('t', 'p5-second', { priority: 5, signal: acMid.signal });
  const p5 = pool.submit('t', 'p5-third', { priority: 5 });
  const p6 = pool.submit('t', 'p1-first', { priority: 1 });
  const p7 = pool.submit('t', 'p1-second', { priority: 1, signal: acLeaf.signal });
  const p8 = pool.submit('t', 'p20-top', { priority: 20 });

  assert.equal(pool.getStats().queuedTasks, 8);
  assert.equal(pool.getStats().peakQueuedTasks, 8);

  acRoot.abort('drop-root-ish');
  acMid.abort('drop-mid');
  acLeaf.abort('drop-leaf');

  await assert.rejects(p1, JobAbortedError);
  await assert.rejects(p4, JobAbortedError);
  await assert.rejects(p7, JobAbortedError);
  assert.equal(pool.getStats().queuedTasks, 5);
  assert.equal(pool.getStats().cancelled, 3);

  workers[0].reply({ id: workers[0].last().id, ok: true, result: 'done' });
  await blocker;

  const started = [];
  for (let i = 0; i < 5; i++) {
    const job = workers[0].last();
    started.push(job.payload);
    workers[0].reply({ id: job.id, ok: true, result: job.payload });
    await flush();
  }
  await Promise.all([p2, p3, p5, p6, p8]);

  assert.deepEqual(started, ['p20-top', 'p10-second', 'p5-first', 'p5-third', 'p1-first']);
  assert.equal(pool.getStats().peakQueuedTasks, 8);
  assert.equal(pool.getStats().queuedTasks, 0);
  await pool.terminate();
});

test('W218: warm worker reuse prefers most recently active idle worker and supports warmUp()', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 3, minWorkers: 1, idleTimeoutMs: 100 });

  assert.throws(() => pool.warmUp(-1), RangeError);
  assert.throws(() => pool.warmUp(1.5), RangeError);

  assert.equal(pool.warmUp(2), 2);
  assert.equal(workers.length, 2);
  assert.equal(pool.getStats().activeWorkers, 0);
  assert.equal(pool.getStats().idleWorkers, 2);

  // Dispatch 3 concurrent jobs: 2 use pre-warmed workers, 1 spawns lazily up to maxWorkers.
  const j1 = pool.submit('t', 'j1');
  const j2 = pool.submit('t', 'j2');
  const j3 = pool.submit('t', 'j3');
  assert.equal(workers.length, 3);
  assert.equal(pool.getStats().activeWorkers, 3);
  assert.equal(pool.getStats().idleWorkers, 0);

  // Finish workers 0, 1, 2 at t=10, t=20, t=30 so worker 2 is most recently warm.
  clock.advance(10);
  workers[0].reply({ id: workers[0].last().id, ok: true, result: 1 });
  clock.advance(10);
  workers[1].reply({ id: workers[1].last().id, ok: true, result: 2 });
  clock.advance(10);
  workers[2].reply({ id: workers[2].last().id, ok: true, result: 3 });
  await Promise.all([j1, j2, j3]);
  assert.equal(pool.getStats().idleWorkers, 3);
  assert.equal(pool.getStats().warmReuses, 0);

  // At t=80, submit a single job: MRU warm worker (workers[2]) is reused, no new worker spawned.
  clock.advance(50);
  const j4 = pool.submit('t', 'j4');
  assert.equal(workers.length, 3);
  assert.equal(workers[2].last().payload, 'j4');
  assert.equal(pool.getStats().activeWorkers, 1);
  assert.equal(pool.getStats().idleWorkers, 2);
  assert.equal(pool.getStats().warmReuses, 1);

  // Advance 50ms (t=130): older idle workers[0] (t=110) and workers[1] (t=120) shrink away,
  // while busy warm worker[2] stays alive.
  clock.advance(50);
  assert.equal(workers[0].terminated, true);
  assert.equal(workers[1].terminated, true);
  assert.equal(workers[2].terminated, false);
  assert.equal(pool.getStats().workers, 1);

  workers[2].reply({ id: workers[2].last().id, ok: true, result: 4 });
  assert.equal(await j4, 4);
  assert.equal(pool.getStats().idleWorkers, 1);
  assert.equal(pool.getStats().activeWorkers, 0);

  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('W218: error, crash, and timeout recovery preserves full pool capacity and priority order', async () => {
  const { pool, workers, clock } = setup({
    maxWorkers: 2,
    retries: 0,
    taskTimeoutMs: 50,
    cancelGraceMs: 20,
    breaker: false,
  });

  // Fill both workers and queue 4 prioritized tasks.
  const runA = pool.submit('t', 'runA');
  const runB = pool.submit('t', 'runB');
  const qLow = pool.submit('t', 'qLow', { priority: 1 });
  const qUrgent = pool.submit('t', 'qUrgent', { priority: 100 });
  const qMid = pool.submit('t', 'qMid', { priority: 20 });
  const qHigh = pool.submit('t', 'qHigh', { priority: 50 });

  assert.equal(pool.getStats().activeWorkers, 2);
  assert.equal(pool.getStats().idleWorkers, 0);
  assert.equal(pool.getStats().queuedTasks, 4);
  assert.equal(pool.getStats().peakQueuedTasks, 4);

  // 1) Worker 0 returns a job-level error -> stays alive, reuses warm slot for qUrgent (prio 100).
  clock.advance(10);
  workers[0].reply({ id: workers[0].last().id, ok: false, error: { name: 'Error', message: 'bad-task' } });
  await assert.rejects(runA, /bad-task/);
  assert.equal(workers.length, 2);
  assert.equal(workers[0].last().payload, 'qUrgent');
  assert.equal(pool.getStats().activeWorkers, 2);
  assert.equal(pool.getStats().warmReuses, 1);

  // 2) Worker 1 crashes -> replaced cleanly without losing capacity; picks up qHigh (prio 50).
  clock.advance(10);
  workers[1].emit('error', { message: 'segfault' });
  await assert.rejects(runB, WorkerCrashError);
  assert.equal(workers[1].terminated, true);
  assert.equal(workers.length, 3);
  assert.equal(workers[2].last().payload, 'qHigh');
  assert.equal(pool.getStats().activeWorkers, 2);
  assert.equal(pool.getStats().respawnedWorkers, 1);

  // 3) qUrgent (started at t=10) times out at t=60, ignores cancel, hard-kills at t=80.
  //    qHigh (started at t=20) completes at t=60 (run time = 40ms) -> worker[2] picks up qMid (prio 20).
  clock.advance(40);
  await assert.rejects(qUrgent, JobTimeoutError);
  workers[2].reply({ id: workers[2].last().id, ok: true, result: 'HIGH' });
  assert.equal(await qHigh, 'HIGH');
  assert.equal(workers[2].last().payload, 'qMid');

  // Hard-kill grace expires for workers[0] at t=80 -> worker[3] spawns and picks up qLow (prio 1).
  clock.advance(20);
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 4);
  assert.equal(workers[3].last().payload, 'qLow');
  assert.equal(pool.getStats().activeWorkers, 2);

  // Complete remaining two jobs at t=100 on full 2-worker capacity (qMid ran 40ms, qLow ran 20ms).
  clock.advance(20);
  workers[2].reply({ id: workers[2].last().id, ok: true, result: 'MID' });
  workers[3].reply({ id: workers[3].last().id, ok: true, result: 'LOW' });
  assert.deepEqual(await Promise.all([qMid, qLow]), ['MID', 'LOW']);

  const stats = pool.getStats();
  assert.deepEqual(stats, pool.stats());
  assert.equal(stats.workers, 2);
  assert.equal(stats.activeWorkers, 0);
  assert.equal(stats.idleWorkers, 2);
  assert.equal(stats.queuedTasks, 0);
  assert.equal(stats.peakQueuedTasks, 4);
  assert.equal(stats.completedTasks, 3);
  assert.equal(stats.failedTasks, 3);
  assert.equal(stats.timedOutTasks, 1);
  assert.equal(stats.crashed, 1);
  assert.equal(stats.respawnedWorkers, 2);
  assert.ok(stats.avgTaskMs > 0);
  assert.equal(stats.avgTaskMs, stats.avgRunMs);

  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('W218: cancel postMessage failure hard-kills broken worker immediately without wedging slot', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 1, cancelGraceMs: 5000 });
  const ac = new AbortController();
  const p1 = pool.submit('t', 'first', { signal: ac.signal });
  const p2 = pool.submit('t', 'second');

  // Simulate worker throwing when asked to cancel.
  workers[0].postMessage = () => {
    throw new Error('channel closed');
  };
  ac.abort();
  await assert.rejects(p1, JobAbortedError);

  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);
  assert.equal(workers[1].last().payload, 'second');
  workers[1].reply({ id: workers[1].last().id, ok: true, result: 'recovered' });
  assert.equal(await p2, 'recovered');
  assert.equal(clock.pending, 0);

  await pool.terminate();
});

test('W218: telemetry reports exact activeWorkers, idleWorkers, queuedTasks, peakQueuedTasks, completedTasks, failedTasks, avgTaskMs', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 2 });

  // Initial empty state.
  let s = pool.getStats();
  assert.equal(s.activeWorkers, 0);
  assert.equal(s.idleWorkers, 0);
  assert.equal(s.queuedTasks, 0);
  assert.equal(s.peakQueuedTasks, 0);
  assert.equal(s.completedTasks, 0);
  assert.equal(s.failedTasks, 0);
  assert.equal(s.avgTaskMs, 0);

  // Submit 4 tasks: 2 active, 2 queued.
  const t1 = pool.submit('t', 1, { priority: 1 });
  const t2 = pool.submit('t', 2, { priority: 2 });
  const t3 = pool.submit('t', 3, { priority: 10 });
  const t4 = pool.submit('t', 4, { priority: 5 });

  s = pool.getStats();
  assert.equal(s.activeWorkers, 2);
  assert.equal(s.idleWorkers, 0);
  assert.equal(s.queuedTasks, 2);
  assert.equal(s.peakQueuedTasks, 2);

  // t1 completes at t=20 (20ms), t3 (prio 10) dispatches onto worker 0.
  clock.advance(20);
  workers[0].reply({ id: 1, ok: true, result: 'r1' });
  assert.equal(await t1, 'r1');
  assert.equal(workers[0].last().id, 3);

  // t2 fails at t=40 (40ms), t4 (prio 5) dispatches onto worker 1.
  clock.advance(20);
  workers[1].reply({ id: 2, ok: false, error: { name: 'Error', message: 'boom' } });
  await assert.rejects(t2, /boom/);
  assert.equal(workers[1].last().id, 4);

  // t3 finishes at t=60 (run 40ms), t4 finishes at t=100 (run 60ms).
  clock.advance(20);
  workers[0].reply({ id: 3, ok: true, result: 'r3' });
  assert.equal(await t3, 'r3');
  s = pool.getStats();
  assert.equal(s.activeWorkers, 1);
  assert.equal(s.idleWorkers, 1);
  assert.equal(s.queuedTasks, 0);

  clock.advance(40);
  workers[1].reply({ id: 4, ok: true, result: 'r4' });
  assert.equal(await t4, 'r4');

  // Run durations: 20 + 40 + 40 + 60 = 160ms across 4 tasks => avgTaskMs === 40.
  s = pool.getStats();
  assert.equal(s.activeWorkers, 0);
  assert.equal(s.idleWorkers, 2);
  assert.equal(s.queuedTasks, 0);
  assert.equal(s.peakQueuedTasks, 2);
  assert.equal(s.completedTasks, 3);
  assert.equal(s.failedTasks, 1);
  assert.equal(s.avgTaskMs, 40);
  assert.equal(s.warmReuses, 2);

  await pool.terminate();
});

// ---- W274: idle-worker trimming, crash-loop circuit breaker & backoff -----

test('W274: constructor validates maxConsecutiveCrashes, crashWindowMs, and respawn backoff options', () => {
  assert.equal(DEFAULT_RESPAWN_BACKOFF_BASE_MS, 0);
  assert.equal(DEFAULT_RESPAWN_BACKOFF_MAX_MS, 5000);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), maxConsecutiveCrashes: -1 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), maxConsecutiveCrashes: 1.5 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), crashWindowMs: -1 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), crashWindowMs: NaN }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), crashWindowMs: Infinity }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), respawnBackoffBaseMs: -1 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), respawnBackoffBaseMs: NaN }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), respawnBackoffBaseMs: Infinity }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), respawnBackoffMaxMs: -1 }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), respawnBackoffMaxMs: NaN }), RangeError);
  assert.throws(() => new WorkerPool({ workerFactory: () => ({}), respawnBackoffMaxMs: Infinity }), RangeError);
});

test('W274: trimIdleWorkers(minWarm) terminates excess LRU idle workers without disrupting active tasks', async () => {
  const { pool, workers, clock } = setup({ maxWorkers: 4, minWorkers: 2, idleTimeoutMs: 500 });

  assert.throws(() => pool.trimIdleWorkers(-1), RangeError);
  assert.throws(() => pool.trimIdleWorkers(1.5), RangeError);
  assert.throws(() => pool.trimIdleWorkers('1'), RangeError);

  // Spawn 4 workers and finish in deterministic order (0, 1, 2, 3) so worker 3 is warmest.
  const jobs = [pool.submit('t', 0), pool.submit('t', 1), pool.submit('t', 2), pool.submit('t', 3)];
  for (let i = 0; i < 4; i++) {
    clock.advance(10);
    workers[i].reply({ id: workers[i].last().id, ok: true, result: i });
  }
  await Promise.all(jobs);
  assert.equal(pool.getStats().idleWorkers, 4);
  assert.equal(clock.pending, 4);

  // Occupy the warmest worker (workers[3]) with an active job; 3 workers remain idle.
  const activeJob = pool.submit('t', 'active');
  assert.equal(workers[3].last().payload, 'active');
  assert.equal(pool.getStats().activeWorkers, 1);
  assert.equal(pool.getStats().idleWorkers, 3);
  assert.equal(clock.pending, 3);

  // Trim down to 1 warm idle worker: oldest 2 (workers[0], workers[1]) terminate; workers[2] and active workers[3] stay.
  const res1 = pool.trimIdleWorkers(1);
  assert.deepEqual(res1, { trimmed: 2, remainingIdle: 1, active: 1 });
  assert.equal(workers[0].terminated, true);
  assert.equal(workers[1].terminated, true);
  assert.equal(workers[2].terminated, false);
  assert.equal(workers[3].terminated, false);
  assert.equal(pool.getStats().trimmedWorkers, 2);
  assert.equal(pool.getStats().respawnCount, 0);
  assert.equal(clock.pending, 1);

  // Default minWarm = 0 sheds the last idle worker even below minWorkers without touching the active job.
  const res2 = pool.trimIdleWorkers();
  assert.deepEqual(res2, { trimmed: 1, remainingIdle: 0, active: 1 });
  assert.equal(workers[2].terminated, true);
  assert.equal(workers[3].terminated, false);
  assert.equal(pool.getStats().trimmedWorkers, 3);
  assert.equal(clock.pending, 0);

  // Active task finishes cleanly.
  workers[3].reply({ id: workers[3].last().id, ok: true, result: 'active-ok' });
  assert.equal(await activeJob, 'active-ok');
  assert.equal(pool.getStats().activeWorkers, 0);
  assert.equal(pool.getStats().idleWorkers, 1);

  // Asking to keep more warm workers than currently idle is a no-op.
  assert.deepEqual(pool.trimIdleWorkers(5), { trimmed: 0, remainingIdle: 1, active: 0 });
  assert.equal(pool.getStats().trimmedWorkers, 3);

  // Re-expanding after trim does not inflate respawnCount.
  const re1 = pool.submit('t', 'r1');
  const re2 = pool.submit('t', 'r2');
  const re3 = pool.submit('t', 'r3');
  assert.equal(pool.getStats().workers, 3);
  assert.equal(pool.getStats().respawnCount, 0);
  for (const w of workers.filter((x) => !x.terminated)) {
    w.reply({ id: w.last().id, ok: true, result: 'ok' });
  }
  await Promise.all([re1, re2, re3]);

  // Automatic idleTimeoutMs shrinking also increments trimmedWorkers (3 -> minWorkers=2).
  clock.advance(500);
  assert.equal(pool.getStats().workers, 2);
  assert.equal(pool.getStats().trimmedWorkers, 4);

  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('W274: crash-rate circuit breaker (maxConsecutiveCrashes, crashWindowMs) trips, counts circuitBreakerTrips, and resets via resetCircuitBreaker()', async () => {
  const { pool, workers, clock } = setup({
    maxWorkers: 1,
    maxConsecutiveCrashes: 3,
    crashWindowMs: 1000,
  });

  const j1 = pool.submit('t', 1);
  const j2 = pool.submit('t', 2);
  const j3 = pool.submit('t', 3);
  const j4 = pool.submit('t', 4);
  const s4 = settleState(j4);

  workers[0].emit('error', { message: 'err-1' });
  await assert.rejects(j1, WorkerCrashError);
  assert.equal(pool.getStats().circuitOpen, false);
  assert.equal(pool.getStats().circuitBreakerTrips, 0);

  clock.advance(100);
  workers[1].emit('messageerror', {});
  await assert.rejects(j2, WorkerCrashError);
  assert.equal(pool.getStats().circuitOpen, false);
  assert.equal(pool.getStats().circuitBreakerTrips, 0);

  clock.advance(100);
  workers[2].emit('error', { message: 'err-3' });
  await assert.rejects(j3, WorkerCrashError);
  await flush();

  // Circuit tripped: queued job j4 rejected with CircuitOpenError and no 4th worker spawned.
  assert.equal(s4.status, 'rejected');
  assert.ok(s4.value instanceof CircuitOpenError);
  assert.equal(workers.length, 3);
  assert.equal(pool.getStats().circuitOpen, true);
  assert.equal(pool.getStats().circuitBreakerTrips, 1);

  // Subsequent submits fail fast without spawning.
  await assert.rejects(pool.submit('t', 5), CircuitOpenError);
  assert.equal(workers.length, 3);

  // Manual recovery via resetCircuitBreaker() re-enables the pool while keeping cumulative trip count.
  pool.resetCircuitBreaker();
  assert.equal(pool.getStats().circuitOpen, false);
  assert.equal(pool.getStats().circuitBreakerTrips, 1);

  const ok = pool.submit('t', 6);
  assert.equal(workers.length, 4);
  workers[3].reply({ id: workers[3].last().id, ok: true, result: 'healthy' });
  assert.equal(await ok, 'healthy');

  // Crashes spaced past crashWindowMs do not trip.
  const sp1 = pool.submit('t', 7);
  workers[3].emit('error', { message: 'spaced-1' });
  await assert.rejects(sp1, WorkerCrashError);
  clock.advance(1500);
  const sp2 = pool.submit('t', 8);
  workers[4].emit('error', { message: 'spaced-2' });
  await assert.rejects(sp2, WorkerCrashError);
  assert.equal(pool.getStats().circuitOpen, false);
  assert.equal(pool.getStats().circuitBreakerTrips, 1);

  // Trip a second time within window -> circuitBreakerTrips increments to 2.
  const sp3 = pool.submit('t', 9);
  workers[5].emit('error', { message: 'burst-2' });
  await assert.rejects(sp3, WorkerCrashError);
  const sp4 = pool.submit('t', 10);
  workers[6].emit('error', { message: 'burst-3' });
  await assert.rejects(sp4, WorkerCrashError);
  assert.equal(pool.getStats().circuitOpen, true);
  assert.equal(pool.getStats().circuitBreakerTrips, 2);

  await pool.terminate();
  assert.equal(clock.pending, 0);

  // maxConsecutiveCrashes: 0 disables circuit tripping.
  const { pool: noBreakPool, workers: nbWorkers } = setup({
    maxWorkers: 1,
    maxConsecutiveCrashes: 0,
  });
  for (let i = 0; i < 6; i++) {
    const p = noBreakPool.submit('t', i);
    nbWorkers.at(-1).emit('error', { message: `c-${i}` });
    await assert.rejects(p, WorkerCrashError);
  }
  assert.equal(noBreakPool.getStats().circuitOpen, false);
  assert.equal(noBreakPool.getStats().circuitBreakerTrips, 0);
  await noBreakPool.terminate();
});

test('W274: bounded exponential-backoff respawn caps delay at respawnBackoffMaxMs and resets streak on success', async () => {
  const { pool, workers, clock } = setup({
    maxWorkers: 1,
    maxConsecutiveCrashes: 0,
    maxRespawnsPerSlot: 10,
    respawnBackoffBaseMs: 50,
    respawnBackoffMaxMs: 150,
  });

  const t1 = pool.submit('t', 1);
  const t2 = pool.submit('t', 2);
  const t3 = pool.submit('t', 3);
  const t4 = pool.submit('t', 4);
  const t5 = pool.submit('t', 5);
  assert.equal(workers.length, 1);

  // Crash #1 -> 50ms backoff before respawn #1.
  workers[0].emit('error', { message: 'crash-1' });
  await assert.rejects(t1, WorkerCrashError);
  assert.equal(workers.length, 1);
  clock.advance(49);
  assert.equal(workers.length, 1);
  clock.advance(1);
  assert.equal(workers.length, 2);
  assert.equal(workers[1].last().payload, 2);
  assert.equal(pool.getStats().respawnCount, 1);

  // Crash #2 -> 100ms backoff before respawn #2.
  workers[1].emit('error', { message: 'crash-2' });
  await assert.rejects(t2, WorkerCrashError);
  assert.equal(workers.length, 2);
  clock.advance(99);
  assert.equal(workers.length, 2);
  clock.advance(1);
  assert.equal(workers.length, 3);
  assert.equal(workers[2].last().payload, 3);
  assert.equal(pool.getStats().respawnCount, 2);

  // Crash #3 -> 200ms capped at respawnBackoffMaxMs (150ms).
  workers[2].emit('error', { message: 'crash-3' });
  await assert.rejects(t3, WorkerCrashError);
  clock.advance(149);
  assert.equal(workers.length, 3);
  clock.advance(1);
  assert.equal(workers.length, 4);
  assert.equal(workers[3].last().payload, 4);
  assert.equal(pool.getStats().respawnCount, 3);

  // Crash #4 -> stays bounded at 150ms.
  workers[3].emit('error', { message: 'crash-4' });
  await assert.rejects(t4, WorkerCrashError);
  clock.advance(149);
  assert.equal(workers.length, 4);
  clock.advance(1);
  assert.equal(workers.length, 5);
  assert.equal(workers[4].last().payload, 5);
  assert.equal(pool.getStats().respawnCount, 4);

  // Completing a job successfully resets the slot's crash streak back to 0.
  workers[4].reply({ id: workers[4].last().id, ok: true, result: 'ok-5' });
  assert.equal(await t5, 'ok-5');

  // Crash #5 after recovery uses base backoff (50ms), not the 150ms cap.
  const t6 = pool.submit('t', 6);
  const t7 = pool.submit('t', 7);
  workers[4].emit('error', { message: 'crash-after-recovery' });
  await assert.rejects(t6, WorkerCrashError);
  clock.advance(49);
  assert.equal(workers.length, 5);
  clock.advance(1);
  assert.equal(workers.length, 6);
  assert.equal(workers[5].last().payload, 7);
  workers[5].reply({ id: workers[5].last().id, ok: true, result: 'ok-7' });
  assert.equal(await t7, 'ok-7');
  assert.equal(pool.getStats().respawnCount, 5);

  await pool.terminate();
  assert.equal(clock.pending, 0);
});

test('W274: backoff respawn handles graceful/immediate terminate, aborts, circuit trip, and minWorkers without timer leaks', async () => {
  // 1) Graceful terminate finishes queued work once backoff expires.
  {
    const { pool, workers, clock } = setup({
      maxWorkers: 1,
      maxConsecutiveCrashes: 0,
      respawnBackoffBaseMs: 50,
    });
    const j1 = pool.submit('t', 1);
    const j2 = pool.submit('t', 2);
    workers[0].emit('error', { message: 'boom' });
    await assert.rejects(j1, WorkerCrashError);
    assert.equal(workers.length, 1);

    let terminated = false;
    const termP = pool.terminate({ graceful: true }).then(() => {
      terminated = true;
    });
    clock.advance(50);
    assert.equal(workers.length, 2);
    assert.equal(terminated, false);
    workers[1].reply({ id: workers[1].last().id, ok: true, result: 'finished' });
    assert.equal(await j2, 'finished');
    await termP;
    assert.equal(terminated, true);
    assert.equal(workers[1].terminated, true);
    assert.equal(clock.pending, 0);
  }

  // 2) Immediate terminate clears pending backoff timers cleanly.
  {
    const { pool, workers, clock } = setup({
      maxWorkers: 1,
      maxConsecutiveCrashes: 0,
      respawnBackoffBaseMs: 100,
    });
    const j1 = pool.submit('t', 1);
    const j2 = settleState(pool.submit('t', 2));
    workers[0].emit('error', { message: 'boom' });
    await assert.rejects(j1, WorkerCrashError);
    assert.equal(clock.pending, 1);
    await pool.terminate();
    await flush();
    assert.equal(j2.status, 'rejected');
    assert.ok(j2.value instanceof PoolTerminatedError);
    assert.equal(clock.pending, 0);
  }

  // 3) Aborting the only queued job during backoff clears the unnecessary backoff timer.
  {
    const { pool, workers, clock } = setup({
      maxWorkers: 1,
      maxConsecutiveCrashes: 0,
      respawnBackoffBaseMs: 100,
    });
    const ac = new AbortController();
    const j1 = pool.submit('t', 1);
    const j2 = pool.submit('t', 2, { signal: ac.signal });
    workers[0].emit('error', { message: 'boom' });
    await assert.rejects(j1, WorkerCrashError);
    assert.equal(clock.pending, 1);
    ac.abort();
    await assert.rejects(j2, JobAbortedError);
    assert.equal(clock.pending, 0);
    // Submitting before the 100ms window expires still respects the remaining backoff.
    clock.advance(40);
    const j3 = pool.submit('t', 3);
    assert.equal(workers.length, 1);
    clock.advance(59);
    assert.equal(workers.length, 1);
    clock.advance(1);
    assert.equal(workers.length, 2);
    workers[1].reply({ id: workers[1].last().id, ok: true, result: 'ok-3' });
    assert.equal(await j3, 'ok-3');
    await pool.terminate();
    assert.equal(clock.pending, 0);
  }

  // 4) minWorkers > 0 respawns idle crashed worker after backoff expires.
  {
    const { pool, workers, clock } = setup({
      maxWorkers: 1,
      minWorkers: 1,
      maxConsecutiveCrashes: 0,
      respawnBackoffBaseMs: 60,
    });
    assert.equal(pool.warmUp(1), 1);
    workers[0].emit('error', { message: 'idle-crash' });
    assert.equal(pool.getStats().workers, 0);
    clock.advance(59);
    assert.equal(pool.getStats().workers, 0);
    clock.advance(1);
    assert.equal(pool.getStats().workers, 1);
    assert.equal(pool.getStats().respawnCount, 1);
    await pool.terminate();
    assert.equal(clock.pending, 0);
  }
});

test('W274: getStats() surfaces respawnCount, trimmedWorkers, circuitBreakerTrips, circuitOpen, and watchdogTimeouts', async () => {
  const { pool, workers, clock } = setup({
    maxWorkers: 2,
    taskTimeoutMs: 50,
    cancelGraceMs: 0,
    maxConsecutiveCrashes: 2,
    crashWindowMs: 1000,
  });

  // Initial W274 telemetry fields are zeroed.
  let s = pool.getStats();
  assert.deepEqual(s, pool.stats());
  assert.equal(s.respawnCount, 0);
  assert.equal(s.trimmedWorkers, 0);
  assert.equal(s.circuitBreakerTrips, 0);
  assert.equal(s.circuitOpen, false);
  assert.equal(s.watchdogTimeouts, 0);

  // Warm 2 workers, trim 1.
  pool.warmUp(2);
  assert.deepEqual(pool.trimIdleWorkers(1), { trimmed: 1, remainingIdle: 1, active: 0 });
  assert.equal(pool.getStats().trimmedWorkers, 1);

  // Trigger 1 watchdog timeout -> hard-kill, then next job respawns the killed slot.
  const slow = pool.submit('hang', 1);
  clock.advance(50);
  await assert.rejects(slow, JobTimeoutError);
  const follow = pool.submit('work', 2);
  const liveAfterTimeout = workers.find((w) => !w.terminated && w.last()?.payload === 2);
  assert.ok(liveAfterTimeout);
  liveAfterTimeout.reply({ id: liveAfterTimeout.last().id, ok: true, result: 'done-2' });
  assert.equal(await follow, 'done-2');

  s = pool.getStats();
  assert.equal(s.watchdogTimeouts, 1);
  assert.equal(s.timedOutTasks, 1);
  assert.equal(s.respawnCount, s.respawnedWorkers);
  assert.ok(s.respawnCount >= 1);

  // Trigger 2 crashes to trip the circuit breaker.
  const c1 = pool.submit('t', 3);
  const wA = workers.find((w) => !w.terminated && w.last()?.payload === 3);
  wA.emit('error', { message: 'crash-a' });
  await assert.rejects(c1, WorkerCrashError);

  const c2 = pool.submit('t', 4);
  const wB = workers.find((w) => !w.terminated && w.last()?.payload === 4);
  wB.emit('error', { message: 'crash-b' });
  await assert.rejects(c2, WorkerCrashError);

  s = pool.getStats();
  assert.deepEqual(s, pool.stats());
  assert.equal(s.trimmedWorkers, 1);
  assert.equal(s.watchdogTimeouts, 1);
  assert.equal(s.timedOutTasks, 1);
  assert.equal(s.respawnCount, s.respawnedWorkers);
  assert.equal(s.circuitBreakerTrips, 1);
  assert.equal(s.circuitOpen, true);

  await pool.terminate();
  assert.equal(clock.pending, 0);
});
