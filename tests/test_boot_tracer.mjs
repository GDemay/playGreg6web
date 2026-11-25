import test from 'node:test';
import assert from 'node:assert/strict';

import BootTracerDefault, {
  BOOT_MILESTONES,
  TRAP_KIND,
  DEFAULT_MAX_TRAPS,
  DEFAULT_SLOW_THRESHOLD_MS,
  CHROME_TRACE_CATEGORY,
  DEFAULT_TRACE_PID,
  DEFAULT_TRACE_TID,
  STAGE_NOT_STARTED,
  BootTracer,
  parseLogLine,
  extractGuestVa,
  extractImportRef,
  normalizeGuestVa,
} from '../web/telemetry/boot_tracer.js';
import { CrashReporter } from '../web/telemetry/crash_report.js';

function createClock(startMs = 1000) {
  let current = startMs;
  return {
    now: () => current,
    advance(ms) {
      current += ms;
      return current;
    },
    set(ms) {
      current = ms;
      return current;
    },
  };
}

function createHud() {
  const calls = [];
  return {
    calls,
    setStatus(text, level) {
      calls.push({ text, level });
    },
  };
}

test('exports ordered milestones, trap kinds and defaults', () => {
  assert.equal(BootTracerDefault, BootTracer);
  assert.deepEqual(BOOT_MILESTONES, [
    'PGIM_LOAD',
    'TLS_INIT',
    'CRT_INITTERM',
    'ENTRY_CALL',
    'XMEM_INIT',
    'GDK_INIT',
    'D3D12_CREATE_DEVICE',
    'BINK_OPEN',
    'FIRST_PRESENT',
  ]);
  assert.ok(Object.isFrozen(BOOT_MILESTONES));
  assert.ok(Object.isFrozen(TRAP_KIND));
  assert.deepEqual(Object.keys(TRAP_KIND), [
    'UNIMPLEMENTED_OPCODE',
    'UNRESOLVED_IMPORT',
    'INDIRECT_CALL_MISS',
    'DIVIDE_ERROR',
    'GUEST_TRAP',
    'WASM_TRAP',
  ]);
  for (const [key, value] of Object.entries(TRAP_KIND)) {
    assert.equal(key, value);
  }
  assert.equal(DEFAULT_MAX_TRAPS, 256);
  assert.equal(new BootTracer().maxTraps, 256);
});

test('constructor validates options', () => {
  assert.throws(() => new BootTracer(null), TypeError);
  assert.throws(() => new BootTracer({ maxTraps: 0 }), RangeError);
  assert.throws(() => new BootTracer({ maxTraps: 1.5 }), RangeError);
  assert.throws(() => new BootTracer({ now: 5 }), TypeError);
});

test('empty tracer summary starts at 0% and NOT_STARTED', () => {
  const clock = createClock();
  const tracer = new BootTracer({ now: clock.now });
  const summary = tracer.getSummary();
  assert.equal(summary.stage, STAGE_NOT_STARTED);
  assert.equal(summary.progressPct, 0);
  assert.equal(summary.readyForSplash, false);
  assert.equal(summary.elapsedMs, 0);
  assert.equal(summary.milestonesReached, 0);
  assert.equal(summary.milestones.length, BOOT_MILESTONES.length);
  assert.ok(summary.milestones.every((m) => m.reached === false && m.elapsedMs === null && m.count === 0));
  assert.equal(summary.trapCount, 0);
  assert.equal(summary.uniqueTrapCount, 0);
  assert.equal(summary.lastTrap, null);
  assert.deepEqual(summary.unresolvedImports, []);
  assert.deepEqual(summary.recentTraps, []);
});

test('full milestone progression goes from 0% to 100% and readyForSplash', () => {
  const clock = createClock(500);
  const tracer = new BootTracer({ now: clock.now });
  const seen = [];
  BOOT_MILESTONES.forEach((name, idx) => {
    clock.advance(10);
    const entry = tracer.markMilestone(name, { step: idx });
    assert.equal(entry.firstReach, true);
    assert.equal(entry.index, idx);
    assert.equal(entry.elapsedMs, (idx + 1) * 10);
    const summary = tracer.getSummary();
    assert.equal(summary.stage, name);
    seen.push(summary.progressPct);
    assert.equal(summary.readyForSplash, name === 'FIRST_PRESENT');
  });
  assert.deepEqual(seen, [11, 22, 33, 44, 56, 67, 78, 89, 100]);
  const summary = tracer.getSummary();
  assert.equal(summary.progressPct, 100);
  assert.equal(summary.readyForSplash, true);
  assert.equal(summary.milestonesReached, 9);
  assert.equal(summary.elapsedMs, 90);
  assert.deepEqual(summary.milestones[3].detail, { step: 3 });
});

test('markMilestone keeps first timestamp, counts repeats and is case-insensitive', () => {
  const clock = createClock();
  const tracer = new BootTracer({ now: clock.now });
  clock.advance(5);
  tracer.markMilestone('tls_init', 'first');
  clock.advance(20);
  const again = tracer.markMilestone('TLS_INIT', 'second');
  assert.equal(again.firstReach, false);
  assert.equal(again.count, 2);
  assert.equal(again.elapsedMs, 5);
  assert.equal(again.lastElapsedMs, 25);
  assert.equal(again.detail, 'first');
  assert.throws(() => tracer.markMilestone('NOT_A_MILESTONE'), RangeError);
  assert.throws(() => tracer.markMilestone(undefined), RangeError);
});

test('stage and progress follow the furthest milestone even when out of order', () => {
  const tracer = new BootTracer({ now: createClock().now });
  tracer.markMilestone('GDK_INIT');
  tracer.markMilestone('PGIM_LOAD');
  const summary = tracer.getSummary();
  assert.equal(summary.stage, 'GDK_INIT');
  assert.equal(summary.progressPct, 67);
  assert.equal(summary.milestonesReached, 2);
});

test('elapsedMs is monotonic even if the clock goes backwards', () => {
  const clock = createClock(100);
  const tracer = new BootTracer({ now: clock.now });
  clock.advance(50);
  tracer.markMilestone('PGIM_LOAD');
  clock.set(120);
  const entry = tracer.markMilestone('TLS_INIT');
  assert.equal(entry.elapsedMs, 50);
  clock.set(Number.NaN);
  assert.equal(tracer.getSummary().elapsedMs, 50);
});

test('explicit startMs option anchors elapsed time', () => {
  const clock = createClock(1000);
  const tracer = new BootTracer({ now: clock.now, startMs: 900 });
  assert.equal(tracer.markMilestone('PGIM_LOAD').elapsedMs, 100);
});

test('ingestLogLine maps [boot]/[wasm] lines to milestones', () => {
  const clock = createClock();
  const tracer = new BootTracer({ now: clock.now });
  const cases = [
    ['[boot] loaded pgim image (3 sections)', 'PGIM_LOAD'],
    ['[boot] TLS callbacks executed', 'TLS_INIT'],
    ['[boot] running _initterm_e table', 'CRT_INITTERM'],
    ['[boot] calling guest entry point 0x140001000', 'ENTRY_CALL'],
    ['[wasm] XMemSetAllocationHooks installed', 'XMEM_INIT'],
    ['[wasm] XGameRuntimeInitialize -> S_OK', 'GDK_INIT'],
    ['[wasm] D3D12XboxCreateDevice returned device 0x7', 'D3D12_CREATE_DEVICE'],
    ['[boot] BinkOpen movies/rockstar_logo.bk2', 'BINK_OPEN'],
    ['[wasm] first present submitted to swapchain', 'FIRST_PRESENT'],
  ];
  for (const [line, expected] of cases) {
    clock.advance(1);
    const result = tracer.ingestLogLine(line);
    assert.equal(result?.type, 'milestone', line);
    assert.deepEqual(
      result.milestones.map((m) => m.milestone),
      [expected],
      line
    );
    assert.equal(result.milestones[0].detail.line.length > 0, true);
  }
  const summary = tracer.getSummary();
  assert.equal(summary.progressPct, 100);
  assert.equal(summary.readyForSplash, true);
  assert.equal(summary.milestones[0].detail.source, 'boot');
});

test('ingestLogLine accepts explicit milestone tokens and prefixed lines', () => {
  const tracer = new BootTracer({ now: createClock().now });
  const res = tracer.ingestLogLine('12:00:01.123 LOG: [boot] milestone=crt_initterm');
  assert.deepEqual(
    res.milestones.map((m) => m.milestone),
    ['CRT_INITTERM']
  );
  const multi = tracer.ingestLogLine('[boot] TLS_INIT done, PGIM_LOAD done');
  assert.deepEqual(
    multi.milestones.map((m) => m.milestone),
    ['PGIM_LOAD', 'TLS_INIT']
  );
});

test('ingestLogLine ignores failures, unknown tags and noise', () => {
  const tracer = new BootTracer({ now: createClock().now });
  assert.equal(tracer.ingestLogLine('[boot] D3D12XboxCreateDevice failed hr=0x887a0005'), null);
  assert.equal(tracer.ingestLogLine('[wasm] module instantiated'), null);
  assert.equal(tracer.ingestLogLine('[gpu] first present'), null);
  assert.equal(tracer.ingestLogLine('no tag here'), null);
  assert.equal(tracer.ingestLogLine(''), null);
  assert.equal(tracer.ingestLogLine(undefined), null);
  assert.equal(tracer.ingestLogLine(42), null);
  assert.equal(tracer.ingestLogLine('[import] bound d3d12_xs.dll!D3D12XboxCreateDevice'), null);
  assert.equal(tracer.ingestLogLine('[import] scanning import directory'), null);
  const summary = tracer.getSummary();
  assert.equal(summary.milestonesReached, 0);
  assert.equal(summary.trapCount, 0);
});

test('unresolved import lines never mark milestones', () => {
  const tracer = new BootTracer({ now: createClock().now });
  const res = tracer.ingestLogLine('[import] unresolved import d3d12_xs.dll!D3D12XboxCreateDevice (slot 0x3200)');
  assert.equal(res.type, 'trap');
  assert.equal(tracer.getSummary().milestonesReached, 0);
});

test('[unimpl] lines become fatal UNIMPLEMENTED_OPCODE traps with guest VA', () => {
  const tracer = new BootTracer({ now: createClock().now });
  const res = tracer.ingestLogLine('[unimpl] vpdpbusd at 0x140001234');
  assert.equal(res.type, 'trap');
  assert.equal(res.trap.kind, TRAP_KIND.UNIMPLEMENTED_OPCODE);
  assert.equal(res.trap.guestVa, '0x140001234');
  assert.equal(res.trap.fatal, true);
  assert.equal(res.trap.message, 'vpdpbusd at 0x140001234');
});

test('[import] lines parse DLL!Symbol and ordinal imports, skipping slot offsets', () => {
  const tracer = new BootTracer({ now: createClock().now });
  const named = tracer.ingestLogLine(
    '[import] unresolved import d3d12_xs.dll!D3D12XboxCreateDevice (slot 0x3200): no runtime shim family'
  );
  assert.equal(named.trap.kind, TRAP_KIND.UNRESOLVED_IMPORT);
  assert.equal(named.trap.dll, 'd3d12_xs.dll');
  assert.equal(named.trap.symbol, 'D3D12XboxCreateDevice');
  assert.equal(named.trap.import, 'd3d12_xs.dll!D3D12XboxCreateDevice');
  assert.equal(named.trap.guestVa, null);
  assert.equal(named.trap.fatal, false);

  const ordinal = tracer.ingestLogLine('[import] missing WS2_32.dll!#115 called from 0x140abc000');
  assert.equal(ordinal.trap.dll, 'WS2_32.dll');
  assert.equal(ordinal.trap.symbol, '#115');
  assert.equal(ordinal.trap.guestVa, '0x140abc000');

  const imports = tracer.getSummary().unresolvedImports;
  const ws2 = imports.find((i) => i.symbol === '#115');
  assert.equal(ws2.ordinal, 115);
  assert.equal(ws2.name, 'WS2_32.dll!#115');
  assert.equal(ws2.guestVa, '0x140abc000');
});

test('[trap] lines are classified into trap kinds', () => {
  const tracer = new BootTracer({ now: createClock().now });
  const cases = [
    ['[trap] unimplemented instruction at 0x140001234', TRAP_KIND.UNIMPLEMENTED_OPCODE, '0x140001234'],
    ['[trap] unresolved import KERNEL32.dll!#42 (slot 0x20c8)', TRAP_KIND.UNRESOLVED_IMPORT, null],
    ['[trap] indirect call to unknown target 0x14000dead', TRAP_KIND.INDIRECT_CALL_MISS, '0x14000dead'],
    ['[trap] divide error rip=0x140002000', TRAP_KIND.DIVIDE_ERROR, '0x140002000'],
    ['[trap] int3 at 0x140003000', TRAP_KIND.GUEST_TRAP, '0x140003000'],
    ['[trap] RuntimeError: unreachable', TRAP_KIND.WASM_TRAP, null],
    ['[trap] kind=DIVIDE_ERROR va=0x0000000140004000', TRAP_KIND.DIVIDE_ERROR, '0x140004000'],
  ];
  for (const [line, kind, va] of cases) {
    const res = tracer.ingestLogLine(line);
    assert.equal(res.type, 'trap', line);
    assert.equal(res.trap.kind, kind, line);
    assert.equal(res.trap.guestVa, va, line);
    assert.equal(res.trap.fatal, true, line);
  }
  assert.equal(tracer.getSummary().trapCount, cases.length);
});

test('[wasm] runtime errors become WASM_TRAP records', () => {
  const tracer = new BootTracer({ now: createClock().now });
  const res = tracer.ingestLogLine('[wasm] RuntimeError: memory access out of bounds');
  assert.equal(res.type, 'trap');
  assert.equal(res.trap.kind, TRAP_KIND.WASM_TRAP);
  assert.equal(res.trap.fatal, true);
});

test('parseLogLine is pure and exposes parse helpers', () => {
  const parsed = parseLogLine('[unimpl] opcode 0f 38 at 0x140001234');
  assert.equal(parsed.type, 'trap');
  assert.equal(parsed.tag, 'unimpl');
  assert.equal(parsed.trap.guestVa, '0x140001234');
  assert.equal(parseLogLine('[boot] nothing interesting'), null);

  assert.equal(extractGuestVa('slot 0x20c8 then 0x140001234'), '0x140001234');
  assert.equal(extractGuestVa('rva=0x1000'), null);
  assert.equal(extractGuestVa(''), null);
  assert.deepEqual(extractImportRef('x WS2_32.dll!#115 y'), { dll: 'WS2_32.dll', symbol: '#115', ordinal: 115 });
  assert.equal(extractImportRef('no import'), null);
  assert.equal(normalizeGuestVa(0x140001234), '0x140001234');
  assert.equal(normalizeGuestVa(0x140001234n), '0x140001234');
  assert.equal(normalizeGuestVa('0X00140001234'), '0x140001234');
  assert.equal(normalizeGuestVa(-1), null);
  assert.equal(normalizeGuestVa('nope'), null);
});

test('unresolved imports are deduplicated and sorted by hit count', () => {
  const clock = createClock();
  const tracer = new BootTracer({ now: clock.now });
  const hits = [
    ['xgameruntime.dll', 'XGameSaveInitialize'],
    ['d3d12_xs.dll', 'D3D12XboxCreateDevice'],
    ['D3D12_XS.DLL', 'D3D12XboxCreateDevice'],
    ['WS2_32.dll', '#115'],
    ['d3d12_xs.dll', 'D3D12XboxCreateDevice'],
    ['WS2_32.dll', '#115'],
  ];
  for (const [dll, symbol] of hits) {
    clock.advance(1);
    tracer.recordTrap({ kind: TRAP_KIND.UNRESOLVED_IMPORT, dll, symbol });
  }
  const summary = tracer.getSummary();
  assert.deepEqual(
    summary.unresolvedImports.map((i) => [i.name, i.count]),
    [
      ['d3d12_xs.dll!D3D12XboxCreateDevice', 3],
      ['WS2_32.dll!#115', 2],
      ['xgameruntime.dll!XGameSaveInitialize', 1],
    ]
  );
  const top = summary.unresolvedImports[0];
  assert.equal(top.firstElapsedMs, 2);
  assert.equal(top.lastElapsedMs, 5);
  assert.equal(summary.trapCount, 6);
  assert.equal(summary.uniqueTrapCount, 3);
  assert.equal(summary.lastTrap.count, 2);
});

test('ties in unresolved import counts keep first-seen order', () => {
  const clock = createClock();
  const tracer = new BootTracer({ now: clock.now });
  for (const symbol of ['Zeta', 'Alpha', 'Mid']) {
    clock.advance(1);
    tracer.recordTrap({ kind: 'unresolved_import', dll: 'a.dll', symbol });
  }
  assert.deepEqual(
    tracer.getSummary().unresolvedImports.map((i) => i.symbol),
    ['Zeta', 'Alpha', 'Mid']
  );
});

test('recordTrap validates kind, defaults, and recovers fields from the message', () => {
  const tracer = new BootTracer({ now: createClock().now });
  assert.throws(() => tracer.recordTrap(null), TypeError);
  assert.throws(() => tracer.recordTrap({ kind: 'BOGUS' }), RangeError);

  const generic = tracer.recordTrap({ message: 'guest raised #UD' });
  assert.equal(generic.kind, TRAP_KIND.GUEST_TRAP);
  assert.equal(generic.fatal, true);
  assert.equal(generic.guestVa, null);
  assert.equal(generic.stage, STAGE_NOT_STARTED);

  const recovered = tracer.recordTrap({
    kind: TRAP_KIND.UNRESOLVED_IMPORT,
    message: 'unresolved import xmem.dll!XMemAllocDefault at 0x140005000',
  });
  assert.equal(recovered.dll, 'xmem.dll');
  assert.equal(recovered.symbol, 'XMemAllocDefault');
  assert.equal(recovered.guestVa, '0x140005000');

  const numeric = tracer.recordTrap({ kind: TRAP_KIND.UNRESOLVED_IMPORT, dll: 'WS2_32.dll', symbol: 23, guestVa: 0x1400n });
  assert.equal(numeric.symbol, '#23');
  assert.equal(numeric.guestVa, '0x1400');

  const forced = tracer.recordTrap({ kind: TRAP_KIND.UNRESOLVED_IMPORT, dll: 'a.dll', symbol: 'B', fatal: true });
  assert.equal(forced.fatal, true);

  tracer.markMilestone('ENTRY_CALL');
  const staged = tracer.recordTrap({ kind: TRAP_KIND.DIVIDE_ERROR, guestVa: '0x140006000' });
  assert.equal(staged.stage, 'ENTRY_CALL');
  assert.equal(tracer.getSummary().lastTrap.seq, staged.seq);
});

test('trap ring is bounded and evicts the oldest records', () => {
  const tracer = new BootTracer({ now: createClock().now, maxTraps: 4 });
  for (let i = 0; i < 10; i += 1) {
    tracer.recordTrap({ kind: TRAP_KIND.GUEST_TRAP, guestVa: 0x140000000 + i });
  }
  const summary = tracer.getSummary();
  assert.equal(summary.trapCount, 10);
  assert.equal(summary.uniqueTrapCount, 10);
  assert.equal(summary.recentTraps.length, 4);
  assert.deepEqual(
    summary.recentTraps.map((t) => t.seq),
    [7, 8, 9, 10]
  );
  assert.equal(summary.recentTraps[0].guestVa, '0x140000006');
  assert.equal(summary.lastTrap.seq, 10);
  assert.equal(tracer.trapCount, 10);
});

test('default ring holds exactly 256 traps', () => {
  const tracer = new BootTracer({ now: createClock().now });
  for (let i = 0; i < 300; i += 1) {
    tracer.recordTrap({ kind: TRAP_KIND.UNIMPLEMENTED_OPCODE, guestVa: i });
  }
  const summary = tracer.getSummary();
  assert.equal(summary.recentTraps.length, 256);
  assert.equal(summary.recentTraps[0].seq, 45);
  assert.equal(summary.trapCount, 300);
});

test('fatal traps are forwarded to CrashReporter as errors, non-fatal as warn logs', () => {
  const clock = createClock(1700000000000);
  const reporter = new CrashReporter({ now: clock.now });
  const tracer = new BootTracer({ now: clock.now, crashReporter: reporter });

  tracer.markMilestone('PGIM_LOAD');
  tracer.ingestLogLine('[import] unresolved import d3d12_xs.dll!D3D12XboxCreateDevice (slot 0x3200)');
  tracer.ingestLogLine('[unimpl] vpdpbusd at 0x140001234');

  const errors = reporter.getErrors();
  assert.equal(errors.length, 1);
  assert.equal(errors[0].category, 'wasm_trap');
  assert.equal(errors[0].name, 'GuestTrap');
  assert.match(errors[0].message, /UNIMPLEMENTED_OPCODE @ 0x140001234/);
  assert.equal(errors[0].context.kind, TRAP_KIND.UNIMPLEMENTED_OPCODE);
  assert.equal(errors[0].context.guestVa, '0x140001234');
  assert.equal(errors[0].context.stage, 'PGIM_LOAD');

  const logs = reporter.getLogs();
  assert.equal(logs.length, 2);
  assert.equal(logs[0].level, 'info');
  assert.match(logs[0].message, /milestone PGIM_LOAD reached/);
  assert.equal(logs[1].level, 'warn');
  assert.equal(logs[1].source, 'boot_tracer');
  assert.match(logs[1].message, /UNRESOLVED_IMPORT d3d12_xs\.dll!D3D12XboxCreateDevice/);

  const report = reporter.buildReport({ extra: { boot: tracer.getSummary() } });
  assert.equal(report.extra.boot.stage, 'PGIM_LOAD');
});

test('HUD receives milestone and trap status updates', () => {
  const hud = createHud();
  const tracer = new BootTracer({ now: createClock().now, hud });
  tracer.markMilestone('PGIM_LOAD');
  tracer.markMilestone('PGIM_LOAD');
  tracer.markMilestone('FIRST_PRESENT');
  tracer.recordTrap({ kind: TRAP_KIND.UNRESOLVED_IMPORT, dll: 'a.dll', symbol: 'B' });
  tracer.recordTrap({ kind: TRAP_KIND.DIVIDE_ERROR, guestVa: '0x140001000' });
  assert.deepEqual(hud.calls, [
    { text: 'Boot 11%: PGIM_LOAD', level: 'info' },
    { text: 'Boot 100%: FIRST_PRESENT', level: 'ok' },
    { text: 'Trap UNRESOLVED_IMPORT a.dll!B', level: 'warn' },
    { text: 'Trap DIVIDE_ERROR @ 0x140001000', level: 'error' },
  ]);
});

test('HUD falls back to updateMetrics and telemetry sinks never throw', () => {
  const metrics = [];
  const tracer = new BootTracer({
    now: createClock().now,
    hud: { updateMetrics: (m) => metrics.push(m) },
  });
  tracer.markMilestone('TLS_INIT');
  assert.deepEqual(metrics, [{ status: 'Boot 22%: TLS_INIT', level: 'info' }]);

  const throwing = new BootTracer({
    now: createClock().now,
    hud: {
      setStatus() {
        throw new Error('hud broke');
      },
    },
    crashReporter: {
      recordError() {
        throw new Error('reporter broke');
      },
      recordLog() {
        throw new Error('reporter broke');
      },
    },
  });
  assert.doesNotThrow(() => {
    throwing.markMilestone('PGIM_LOAD');
    throwing.recordTrap({ kind: TRAP_KIND.GUEST_TRAP });
    throwing.recordTrap({ kind: TRAP_KIND.UNRESOLVED_IMPORT, dll: 'a.dll', symbol: 'B' });
  });
  assert.equal(throwing.getSummary().trapCount, 2);
});

test('summaries and JSON never leak host paths or credentials', () => {
  const tracer = new BootTracer({ now: createClock().now });
  tracer.ingestLogLine('[boot] loaded pgim from C:\\Users\\alice\\Games\\GTA6\\game.pgim');
  tracer.ingestLogLine('[trap] unimplemented at 0x140001234 in /home/bob/work/lifted.wasm token=abc123');
  tracer.ingestLogLine('[import] unresolved import d3d12_xs.dll!Foo from file:///Users/carol/x.pgim');
  tracer.markMilestone('TLS_INIT', { path: '/Users/dave/local-data/tls.bin', apiKey: 'secret-value' });
  tracer.recordTrap({
    kind: TRAP_KIND.GUEST_TRAP,
    dll: 'C:\\Windows\\System32\\evil.dll',
    message: 'crash near \\\\server\\share\\dump.bin Bearer abc.def.ghi',
  });

  const json = tracer.toJson();
  for (const leak of ['alice', 'bob', 'carol', 'dave', 'abc123', 'secret-value', 'server\\\\share', 'abc.def.ghi', 'System32']) {
    assert.equal(json.includes(leak), false, `leaked ${leak}`);
  }
  assert.match(json, /REDACTED/);
  assert.match(json, /0x140001234/);
  assert.match(json, /d3d12_xs\.dll!Foo/);

  const summary = JSON.parse(json);
  assert.equal(summary.milestones[0].reached, true);
  assert.equal(summary.milestones[1].detail.apiKey, '[REDACTED]');
  assert.deepEqual(JSON.parse(tracer.toJson(0)), summary);
});

test('long trap messages are truncated', () => {
  const tracer = new BootTracer({ now: createClock().now });
  const record = tracer.recordTrap({ kind: TRAP_KIND.GUEST_TRAP, message: 'x'.repeat(5000) });
  assert.equal(record.message.length, 512);
  assert.ok(record.message.endsWith('...'));
});

test('reset clears state, restarts the clock, and keeps sinks attached', () => {
  const clock = createClock(0);
  const hud = createHud();
  const tracer = new BootTracer({ now: clock.now, hud, maxTraps: 2 });
  clock.advance(100);
  tracer.markMilestone('BINK_OPEN');
  tracer.recordTrap({ kind: TRAP_KIND.UNRESOLVED_IMPORT, dll: 'a.dll', symbol: 'B' });
  tracer.recordTrap({ kind: TRAP_KIND.GUEST_TRAP });
  tracer.recordTrap({ kind: TRAP_KIND.GUEST_TRAP });

  tracer.reset();
  const summary = tracer.getSummary();
  assert.equal(summary.stage, STAGE_NOT_STARTED);
  assert.equal(summary.progressPct, 0);
  assert.equal(summary.elapsedMs, 0);
  assert.equal(summary.trapCount, 0);
  assert.equal(summary.uniqueTrapCount, 0);
  assert.equal(summary.lastTrap, null);
  assert.deepEqual(summary.unresolvedImports, []);
  assert.deepEqual(summary.recentTraps, []);
  assert.equal(tracer.maxTraps, 2);

  clock.advance(7);
  const entry = tracer.markMilestone('BINK_OPEN');
  assert.equal(entry.firstReach, true);
  assert.equal(entry.elapsedMs, 7);
  const seq = tracer.recordTrap({ kind: TRAP_KIND.GUEST_TRAP }).seq;
  assert.equal(seq, 1);
  assert.equal(hud.calls.at(-2).text, 'Boot 89%: BINK_OPEN');
});

test('exports Chrome trace and slow-threshold defaults and validates slowThresholdMs', () => {
  assert.equal(DEFAULT_SLOW_THRESHOLD_MS, null);
  assert.equal(CHROME_TRACE_CATEGORY, 'playgta6.boot');
  assert.equal(DEFAULT_TRACE_PID, 1);
  assert.equal(DEFAULT_TRACE_TID, 1);
  assert.equal(new BootTracer().slowThresholdMs, null);
  assert.equal(new BootTracer({ slowThresholdMs: 0 }).slowThresholdMs, 0);
  assert.equal(new BootTracer({ slowThresholdMs: 250.5 }).slowThresholdMs, 250.5);

  assert.throws(() => new BootTracer({ slowThresholdMs: -1 }), RangeError);
  assert.throws(() => new BootTracer({ slowThresholdMs: Number.NaN }), RangeError);
  assert.throws(() => new BootTracer({ slowThresholdMs: Number.POSITIVE_INFINITY }), RangeError);
  assert.throws(() => new BootTracer({ slowThresholdMs: '100' }), RangeError);
});

test('getMilestoneDeltas computes inter-milestone deltas and cumulative timings from boot start', () => {
  const clock = createClock(1000);
  const tracer = new BootTracer({ now: clock.now });

  assert.deepEqual(tracer.getMilestoneDeltas(), []);
  assert.deepEqual(tracer.getSummary().slowMilestones, []);
  assert.ok(tracer.getSummary().milestones.every((m) => m.deltaFromPrevMs === null && m.slow === false));

  clock.advance(12.5);
  const m1 = tracer.markMilestone('PGIM_LOAD');
  assert.equal(m1.deltaFromPrevMs, 12.5);

  clock.advance(27.5);
  const m2 = tracer.markMilestone('TLS_INIT');
  assert.equal(m2.deltaFromPrevMs, 27.5);

  // Zero-delta milestone reached at the exact same timestamp as previous.
  const m3 = tracer.markMilestone('CRT_INITTERM');
  assert.equal(m3.deltaFromPrevMs, 0);

  clock.advance(60);
  const m4 = tracer.markMilestone('ENTRY_CALL');
  assert.equal(m4.deltaFromPrevMs, 60);

  const deltas = tracer.getMilestoneDeltas();
  assert.deepEqual(
    deltas.map(({ name, atMs, deltaMs, cumulativeMs }) => ({ name, atMs, deltaMs, cumulativeMs })),
    [
      { name: 'PGIM_LOAD', atMs: 12.5, deltaMs: 12.5, cumulativeMs: 12.5 },
      { name: 'TLS_INIT', atMs: 40, deltaMs: 27.5, cumulativeMs: 40 },
      { name: 'CRT_INITTERM', atMs: 40, deltaMs: 0, cumulativeMs: 40 },
      { name: 'ENTRY_CALL', atMs: 100, deltaMs: 60, cumulativeMs: 100 },
    ]
  );

  // Invariant: sum of deltas equals final cumulativeMs.
  const totalDelta = deltas.reduce((sum, d) => sum + d.deltaMs, 0);
  assert.equal(totalDelta, deltas.at(-1).cumulativeMs);
  assert.ok(deltas.every((d) => d.deltaFromPrevMs === d.deltaMs && d.slow === false));

  const summary = tracer.getSummary();
  assert.equal(summary.milestones[0].deltaFromPrevMs, 12.5);
  assert.equal(summary.milestones[1].deltaFromPrevMs, 27.5);
  assert.equal(summary.milestones[2].deltaFromPrevMs, 0);
  assert.equal(summary.milestones[3].deltaFromPrevMs, 60);
  assert.equal(summary.milestones[4].deltaFromPrevMs, null);
});

test('slowThresholdMs flags milestones exceeding threshold in summary and deltas', () => {
  const clock = createClock(0);
  const tracer = new BootTracer({ now: clock.now, slowThresholdMs: 50 });

  clock.advance(50); // exact boundary: NOT slow
  const exact = tracer.markMilestone('PGIM_LOAD');
  assert.equal(exact.slow, false);

  clock.advance(50.1); // exceeds 50ms: slow
  const slow1 = tracer.markMilestone('TLS_INIT');
  assert.equal(slow1.slow, true);

  clock.advance(10); // fast
  const fast = tracer.markMilestone('CRT_INITTERM');
  assert.equal(fast.slow, false);

  clock.advance(120); // slow
  const slow2 = tracer.markMilestone('ENTRY_CALL');
  assert.equal(slow2.slow, true);

  const summary = tracer.getSummary();
  assert.equal(summary.slowThresholdMs, 50);
  assert.deepEqual(
    summary.slowMilestones.map((m) => m.name),
    ['TLS_INIT', 'ENTRY_CALL']
  );
  assert.equal(summary.milestones[0].slow, false);
  assert.equal(summary.milestones[1].slow, true);
  assert.equal(summary.milestones[2].slow, false);
  assert.equal(summary.milestones[3].slow, true);
  assert.equal(summary.milestones[4].slow, false);

  // Per-call override does not mutate the tracer's configured slowThresholdMs.
  const stricter = tracer.getMilestoneDeltas({ slowThresholdMs: 25 });
  assert.deepEqual(
    stricter.filter((d) => d.slow).map((d) => d.name),
    ['PGIM_LOAD', 'TLS_INIT', 'ENTRY_CALL']
  );
  assert.equal(tracer.slowThresholdMs, 50);
  assert.deepEqual(
    tracer.getMilestoneDeltas({ slowThresholdMs: null }).filter((d) => d.slow),
    []
  );

  assert.throws(() => tracer.getMilestoneDeltas(null), TypeError);
  assert.throws(() => tracer.getMilestoneDeltas({ slowThresholdMs: -5 }), RangeError);
});

test('getMilestoneDeltas handles out-of-order, duplicate, and backwards-clock milestones', () => {
  const clock = createClock(500);
  const tracer = new BootTracer({ now: clock.now });

  // Out of canonical order: GDK_INIT first, then duplicate GDK_INIT, then PGIM_LOAD, then backwards clock TLS_INIT.
  clock.advance(20);
  tracer.markMilestone('GDK_INIT');

  // Duplicate later at +15ms must not duplicate entry or shift subsequent deltas.
  clock.advance(15);
  const dup = tracer.markMilestone('GDK_INIT');
  assert.equal(dup.firstReach, false);
  assert.equal(dup.count, 2);
  assert.equal(dup.elapsedMs, 20);
  assert.equal(dup.lastElapsedMs, 35);
  assert.equal(dup.deltaFromPrevMs, 20);

  // Next distinct milestone at +30ms (total elapsed 65ms): delta from previous distinct milestone (20ms) is 45ms.
  clock.advance(30);
  tracer.markMilestone('PGIM_LOAD');

  // Clock jumps backwards: monotonic clamp keeps elapsedMs at 65ms and deltaMs at 0ms.
  clock.set(510);
  tracer.markMilestone('TLS_INIT');

  const deltas = tracer.getMilestoneDeltas();
  assert.deepEqual(
    deltas.map(({ name, atMs, deltaMs, cumulativeMs }) => ({ name, atMs, deltaMs, cumulativeMs })),
    [
      { name: 'GDK_INIT', atMs: 20, deltaMs: 20, cumulativeMs: 20 },
      { name: 'PGIM_LOAD', atMs: 65, deltaMs: 45, cumulativeMs: 65 },
      { name: 'TLS_INIT', atMs: 65, deltaMs: 0, cumulativeMs: 65 },
    ]
  );
  assert.equal(
    deltas.reduce((sum, d) => sum + d.deltaMs, 0),
    deltas.at(-1).cumulativeMs
  );
});

test('toChromeTraceEvents emits valid Chromium/Perfetto JSON schema with phase spans and instant markers', () => {
  const clock = createClock(2000);
  const tracer = new BootTracer({ now: clock.now, slowThresholdMs: 30 });

  clock.advance(15.25);
  tracer.markMilestone('PGIM_LOAD', { segments: 4 });

  clock.advance(20);
  tracer.recordTrap({
    kind: TRAP_KIND.UNRESOLVED_IMPORT,
    dll: 'd3d12_xs.dll',
    symbol: 'D3D12XboxCreateDevice',
    message: 'missing shim',
  });

  clock.advance(25.5);
  tracer.markMilestone('TLS_INIT');

  // Duplicate milestone should not add extra trace events.
  clock.advance(5);
  tracer.markMilestone('PGIM_LOAD');

  const trace = tracer.toChromeTraceEvents();
  assert.equal(trace.displayTimeUnit, 'ms');
  assert.ok(Array.isArray(trace.traceEvents));
  assert.equal(trace.metadata.source, 'playgta6web.boot_tracer');
  assert.equal(trace.metadata.stage, 'TLS_INIT');
  assert.equal(trace.metadata.milestonesReached, 2);
  assert.equal(trace.metadata.trapCount, 1);
  assert.equal(trace.metadata.slowThresholdMs, 30);

  // JSON round-trip must succeed cleanly.
  const parsed = JSON.parse(JSON.stringify(trace));
  assert.deepEqual(parsed, trace);

  // Verify schema of every event and non-decreasing timestamp ordering.
  let prevTs = 0;
  for (const ev of trace.traceEvents) {
    assert.equal(typeof ev.name, 'string');
    assert.ok(ev.name.length > 0);
    assert.equal(ev.cat, CHROME_TRACE_CATEGORY);
    assert.ok(ev.ph === 'X' || ev.ph === 'i');
    assert.equal(typeof ev.ts, 'number');
    assert.ok(Number.isInteger(ev.ts) && ev.ts >= 0);
    assert.ok(ev.ts >= prevTs, `expected non-decreasing ts (${ev.ts} >= ${prevTs})`);
    prevTs = ev.ts;
    assert.equal(ev.pid, DEFAULT_TRACE_PID);
    assert.equal(ev.tid, DEFAULT_TRACE_TID);
    assert.equal(typeof ev.args, 'object');
    if (ev.ph === 'X') {
      assert.equal(typeof ev.dur, 'number');
      assert.ok(Number.isInteger(ev.dur) && ev.dur >= 0);
      assert.equal('s' in ev, false);
    } else {
      assert.equal(ev.s, 't');
      assert.equal('dur' in ev, false);
    }
  }

  const spans = trace.traceEvents.filter((e) => e.ph === 'X');
  assert.equal(spans.length, 2);
  assert.deepEqual(
    spans.map((s) => ({ name: s.name, ts: s.ts, dur: s.dur, slow: s.args.slow })),
    [
      { name: 'phase:PGIM_LOAD', ts: 0, dur: 15250, slow: false },
      { name: 'phase:TLS_INIT', ts: 15250, dur: 45500, slow: true },
    ]
  );

  // Spans are contiguous: second span starts exactly where first ends.
  assert.equal(spans[0].ts + spans[0].dur, spans[1].ts);

  const instants = trace.traceEvents.filter((e) => e.ph === 'i');
  assert.equal(instants.length, 3);
  assert.deepEqual(
    instants.map((i) => ({ name: i.name, ts: i.ts })),
    [
      { name: 'PGIM_LOAD', ts: 15250 },
      { name: 'trap:UNRESOLVED_IMPORT', ts: 35250 },
      { name: 'TLS_INIT', ts: 60750 },
    ]
  );
  assert.equal(instants[0].args.count, 2);
  assert.deepEqual(instants[0].args.detail, { segments: 4 });
  assert.equal(instants[1].args.import, 'd3d12_xs.dll!D3D12XboxCreateDevice');

  // Option controls: exclude traps, include metadata, custom pid/tid/category.
  const custom = tracer.toChromeTraceEvents({
    pid: 42,
    tid: 7,
    category: 'custom.boot',
    includeTraps: false,
    includeMetadata: true,
    slowThresholdMs: 100,
  });
  const meta = custom.traceEvents.filter((e) => e.ph === 'M');
  assert.equal(meta.length, 2);
  assert.equal(meta[0].name, 'process_name');
  assert.equal(meta[0].pid, 42);
  assert.equal(meta[1].name, 'thread_name');
  assert.equal(meta[1].tid, 7);
  assert.equal(custom.traceEvents.filter((e) => e.name.startsWith('trap:')).length, 0);
  assert.ok(custom.traceEvents.filter((e) => e.ph !== 'M').every((e) => e.cat === 'custom.boot'));
  assert.ok(custom.traceEvents.filter((e) => e.ph === 'X').every((e) => e.args.slow === false));

  assert.throws(() => tracer.toChromeTraceEvents(null), TypeError);
  assert.throws(() => tracer.toChromeTraceEvents({ pid: -1 }), RangeError);
  assert.throws(() => tracer.toChromeTraceEvents({ tid: 1.5 }), RangeError);
  assert.throws(() => tracer.toChromeTraceEvents({ slowThresholdMs: -1 }), RangeError);
});

test('toChromeTraceEvents strictly sanitizes host paths, credentials, and binary buffers', () => {
  const clock = createClock(100);
  const tracer = new BootTracer({ now: clock.now });

  clock.advance(10);
  tracer.ingestLogLine('[boot] loaded pgim from C:\\Users\\alice\\Games\\GTA6\\boot.pgim');
  clock.advance(15);
  tracer.markMilestone('TLS_INIT', {
    path: '/home/bob/local-data/tls.bin',
    url: 'file:///Users/carol/secret.pgim',
    apiKey: 'super-secret-token',
    rawBytes: new Uint8Array([0xde, 0xad, 0xbe, 0xef]),
    rawBuf: new ArrayBuffer(16),
  });
  clock.advance(5);
  tracer.recordTrap({
    kind: TRAP_KIND.GUEST_TRAP,
    message: 'fault in \\\\fileserver\\share\\dump.dmp Bearer tok.123.456',
  });

  const trace = tracer.toChromeTraceEvents({
    category: 'playgta6.boot C:\\Users\\dave\\cat token=xyz999',
  });
  const json = JSON.stringify(trace);

  for (const leak of [
    'alice',
    'bob',
    'carol',
    'dave',
    'super-secret-token',
    'fileserver',
    'tok.123.456',
    'xyz999',
    '222', // 0xde raw byte must not appear as array dump
  ]) {
    assert.equal(json.includes(leak), false, `Chrome trace leaked ${leak}`);
  }
  assert.match(json, /REDACTED/);

  const tlsInstant = trace.traceEvents.find((e) => e.ph === 'i' && e.name === 'TLS_INIT');
  assert.deepEqual(tlsInstant.args.detail.rawBytes, { type: 'Uint8Array', byteLength: 4 });
  assert.deepEqual(tlsInstant.args.detail.rawBuf, { type: 'ArrayBuffer', byteLength: 16 });
  assert.equal(tlsInstant.args.detail.apiKey, '[REDACTED]');
});

test('reset clears milestone deltas, slowMilestones, and Chrome traceEvents while keeping slowThresholdMs', () => {
  const clock = createClock(100);
  const tracer = new BootTracer({ now: clock.now, slowThresholdMs: 25 });

  clock.advance(40);
  tracer.markMilestone('PGIM_LOAD');
  clock.advance(10);
  tracer.recordTrap({ kind: TRAP_KIND.GUEST_TRAP, guestVa: '0x140001000' });

  assert.equal(tracer.getMilestoneDeltas().length, 1);
  assert.equal(tracer.getSummary().slowMilestones.length, 1);
  assert.equal(tracer.toChromeTraceEvents().traceEvents.length, 3);

  tracer.reset();
  assert.equal(tracer.slowThresholdMs, 25);
  assert.deepEqual(tracer.getMilestoneDeltas(), []);
  assert.deepEqual(tracer.getSummary().slowMilestones, []);
  assert.deepEqual(tracer.toChromeTraceEvents().traceEvents, []);

  // Post-reset deltas anchor to the new boot start clock.
  clock.advance(12);
  tracer.markMilestone('TLS_INIT');
  clock.advance(30);
  tracer.markMilestone('CRT_INITTERM');

  assert.deepEqual(
    tracer.getMilestoneDeltas().map(({ name, atMs, deltaMs, cumulativeMs, slow }) => ({
      name,
      atMs,
      deltaMs,
      cumulativeMs,
      slow,
    })),
    [
      { name: 'TLS_INIT', atMs: 12, deltaMs: 12, cumulativeMs: 12, slow: false },
      { name: 'CRT_INITTERM', atMs: 42, deltaMs: 30, cumulativeMs: 42, slow: true },
    ]
  );

  const postResetTrace = tracer.toChromeTraceEvents();
  const spans = postResetTrace.traceEvents.filter((e) => e.ph === 'X');
  assert.deepEqual(
    spans.map((s) => ({ name: s.name, ts: s.ts, dur: s.dur })),
    [
      { name: 'phase:TLS_INIT', ts: 0, dur: 12000 },
      { name: 'phase:CRT_INITTERM', ts: 12000, dur: 30000 },
    ]
  );
});
