import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  INPUT_BUTTONS,
  INPUT_OFFSETS,
  INPUT_WORD_INDICES,
  INPUT_SCRIPT_PRESETS,
  INPUT_SCRIPT_PRESET_NAMES,
  MIN_SCRIPT_HOLD_MS,
  CHROME_MENU_FLOW_4_HOLD_MS,
  DEFAULT_VISIBLE_FRAME_FALLBACK_MS,
  INPUT_SCRIPT_START_TRIGGERS,
  ScriptedInputPlayer,
  InputBridge,
} from '../web/input/input_bridge.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');

function createFakeClockAndTimers(initialNow = 0) {
  let now = initialNow;
  let nextId = 1;
  const timeouts = new Map();

  function setTimeoutFn(cb, delay = 0) {
    const id = nextId++;
    timeouts.set(id, { id, cb, due: now + Math.max(0, delay) });
    return id;
  }

  function clearTimeoutFn(id) {
    timeouts.delete(id);
  }

  function advanceTo(targetMs) {
    assert.ok(targetMs >= now, `Cannot move fake clock backwards (${now} -> ${targetMs})`);
    while (true) {
      let nextDue = Infinity;
      let entry = null;
      for (const item of timeouts.values()) {
        if (item.due <= targetMs && item.due < nextDue) {
          nextDue = item.due;
          entry = item;
        }
      }
      if (!entry) break;
      now = nextDue;
      timeouts.delete(entry.id);
      entry.cb();
    }
    now = targetMs;
  }

  return {
    nowFn: () => now,
    setTimeoutFn,
    clearTimeoutFn,
    advanceTo,
    advanceBy(deltaMs) {
      advanceTo(now + deltaMs);
    },
    pendingTimeouts: () => timeouts.size,
  };
}

test('W355 chrome-menu-flow-4 matches native menu-flow-4 sequence (tools/run_native_boot.py #1483) with >= 180 ms holds and zero B presses', () => {
  assert.ok(INPUT_SCRIPT_PRESET_NAMES.includes('chrome-menu-flow-4'));
  assert.equal(MIN_SCRIPT_HOLD_MS, 180);
  assert.ok(
    CHROME_MENU_FLOW_4_HOLD_MS >= MIN_SCRIPT_HOLD_MS,
    `CHROME_MENU_FLOW_4_HOLD_MS (${CHROME_MENU_FLOW_4_HOLD_MS}) must be >= ${MIN_SCRIPT_HOLD_MS}`
  );

  // Verify parity with tools/run_native_boot.py menu-flow-4 constants & structure.
  const nativeBootPy = fs.readFileSync(
    path.join(REPO_ROOT, 'tools/run_native_boot.py'),
    'utf8'
  );
  assert.match(nativeBootPy, /def\s+_menu_cycle\(/);
  assert.match(nativeBootPy, /def\s+menu_flow_4_script_text\(/);
  assert.match(nativeBootPy, /"DUp",\s*100.*for\s+i\s+in\s+range\(6\)/);
  assert.match(nativeBootPy, /3_000,\s*"DDown"/);
  assert.match(nativeBootPy, /4_500,\s*"DDown"/);
  assert.match(nativeBootPy, /6_500,\s*"A"/);
  assert.match(nativeBootPy, /14_000/);
  assert.match(nativeBootPy, /"1000 DUp 150",\s*"3000 A 100"/);

  const spec = INPUT_SCRIPT_PRESETS['chrome-menu-flow-4'];
  assert.ok(spec);
  assert.equal(Object.isFrozen(spec), true);
  assert.equal(spec.name, 'chrome-menu-flow-4');
  assert.equal(spec.events.length, 94); // 29 Phase 1 + 63 Phase 2 (7 x 9) + 2 Phase 3
  assert.equal(spec.events[0].atMs, 5000);
  assert.equal(spec.events[spec.events.length - 1].atMs, 180000);
  assert.equal(spec.loop.startMs, 190000);
  assert.equal(spec.loop.periodMs, 14000);
  assert.equal(spec.loop.events.length, 9);

  // Every fixed and loop event must have holdMs >= 180 ms, strictly increasing timestamps,
  // zero overlap, and >= 200 ms released gap between consecutive presses.
  for (let i = 0; i < spec.events.length; i++) {
    const ev = spec.events[i];
    assert.ok(ev.holdMs >= 180, `fixed event ${i} holdMs (${ev.holdMs}) must be >= 180 ms`);
    assert.equal(ev.holdMs, CHROME_MENU_FLOW_4_HOLD_MS);
    if (i > 0) {
      const prev = spec.events[i - 1];
      assert.ok(
        ev.atMs >= prev.atMs + prev.holdMs + 200,
        `fixed event ${i} (${ev.atMs}) must leave >= 200 ms release gap after event ${i - 1} end (${prev.atMs + prev.holdMs})`
      );
    }
  }
  for (let i = 0; i < spec.loop.events.length; i++) {
    const ev = spec.loop.events[i];
    assert.ok(ev.holdMs >= 180, `loop event ${i} holdMs (${ev.holdMs}) must be >= 180 ms`);
    assert.equal(ev.holdMs, CHROME_MENU_FLOW_4_HOLD_MS);
    if (i > 0) {
      const prev = spec.loop.events[i - 1];
      assert.ok(
        ev.offsetMs >= prev.offsetMs + prev.holdMs + 200,
        `loop event ${i} (+${ev.offsetMs}) must leave >= 200 ms release gap after loop event ${i - 1} end (+${prev.offsetMs + prev.holdMs})`
      );
    }
  }

  const player = new ScriptedInputPlayer('chrome-menu-flow-4');
  assert.equal(player.preset, 'chrome-menu-flow-4');
  assert.equal(player.startOffsetMs, 0);

  // Brief initial quiet settle before Start Screen A at 5000 ms (does NOT wait 110 s like chrome-menu-flow-2).
  for (const t of [0, 1000, 4999]) {
    assert.deepEqual(player.sampleAt(t), { buttonsMask: 0, activeEventCount: 0 }, `silent at ${t}`);
    assert.equal(player.pressCountAt(t), 0);
  }

  // Phase 1 (5s..75s): Early A every 2500 ms (29 presses, 200 ms hold) driving Start Screen ->
  // 6-star -> Bink2 -> Rockstar -> UI TEXT SIZE -> FINISH SETUP?.
  let expectedCount = 0;
  for (let atMs = 5000; atMs <= 75000; atMs += 2500) {
    expectedCount += 1;
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0, `idle before Phase 1 A at ${atMs}`);
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `Phase 1 A press at ${atMs}`);
    assert.equal(
      player.sampleAt(atMs + CHROME_MENU_FLOW_4_HOLD_MS - 1).buttonsMask,
      INPUT_BUTTONS.A,
      `Phase 1 A held >= 180 ms at ${atMs}`
    );
    assert.equal(
      player.sampleAt(atMs + CHROME_MENU_FLOW_4_HOLD_MS).buttonsMask,
      0,
      `Phase 1 A released at ${atMs + CHROME_MENU_FLOW_4_HOLD_MS}`
    );
    assert.equal(player.pressCountAt(atMs), expectedCount);
  }
  assert.equal(expectedCount, 29);

  // Quiet transition pause (75200..79999 ms) before Phase 2 LEGAL NOTICES rounds.
  for (let t = 75200; t < 80000; t += 200) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `quiet pause at ${t}`);
  }

  // Phase 2 (80s..170.5s): 7 rounds of Timon's 14 s _menu_cycle (Up x6, Down, Down, A).
  const expectedCycle = [
    [0, INPUT_BUTTONS.DPAD_UP],
    [400, INPUT_BUTTONS.DPAD_UP],
    [800, INPUT_BUTTONS.DPAD_UP],
    [1200, INPUT_BUTTONS.DPAD_UP],
    [1600, INPUT_BUTTONS.DPAD_UP],
    [2000, INPUT_BUTTONS.DPAD_UP],
    [3000, INPUT_BUTTONS.DPAD_DOWN],
    [4500, INPUT_BUTTONS.DPAD_DOWN],
    [6500, INPUT_BUTTONS.A],
  ];

  for (let round = 0; round < 7; round++) {
    const base = 80000 + round * 14000;
    for (const [offsetMs, expectedBtn] of expectedCycle) {
      expectedCount += 1;
      const t = base + offsetMs;
      assert.equal(player.sampleAt(t - 1).buttonsMask, 0, `round ${round} idle before +${offsetMs}`);
      assert.equal(player.sampleAt(t).buttonsMask, expectedBtn, `round ${round} press at +${offsetMs}`);
      assert.equal(
        player.sampleAt(t + CHROME_MENU_FLOW_4_HOLD_MS - 1).buttonsMask,
        expectedBtn,
        `round ${round} held 200 ms at +${offsetMs}`
      );
      assert.equal(player.sampleAt(t + CHROME_MENU_FLOW_4_HOLD_MS).buttonsMask, 0);
      assert.equal(player.pressCountAt(t), expectedCount);
    }
  }
  assert.equal(expectedCount, 92);

  // Quiet settle window before Post-Legal List Screen Up + A (170700..177999 ms).
  for (let t = 170700; t < 178000; t += 300) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `quiet settle before Phase 3 at ${t}`);
  }

  // Phase 3 (178s..180s): Post-Legal List Screen DUp (top cursor, 50 draws/frame) + 2000 ms later A.
  expectedCount += 1;
  assert.equal(player.sampleAt(177999).buttonsMask, 0);
  assert.equal(player.sampleAt(178000).buttonsMask, INPUT_BUTTONS.DPAD_UP);
  assert.equal(player.sampleAt(178199).buttonsMask, INPUT_BUTTONS.DPAD_UP);
  assert.equal(player.sampleAt(178200).buttonsMask, 0);
  assert.equal(player.pressCountAt(178000), expectedCount);

  expectedCount += 1;
  assert.equal(player.sampleAt(179999).buttonsMask, 0);
  assert.equal(player.sampleAt(180000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(180199).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(180200).buttonsMask, 0);
  assert.equal(player.pressCountAt(180000), expectedCount);
  assert.equal(expectedCount, 94);

  // Quiet ~10 s settle window (180200..189999 ms) for the 34-draw screen / Main Menu transition.
  for (let t = 180200; t < 190000; t += 500) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `transition settle at ${t}`);
  }

  // Phase 4 (190s+): Looping 14 s _menu_cycle (Up x6, Down, Down, A; 200 ms holds).
  for (let cycle = 0; cycle < 3; cycle++) {
    const base = 190000 + cycle * 14000;
    for (const [offsetMs, expectedBtn] of expectedCycle) {
      expectedCount += 1;
      const t = base + offsetMs;
      assert.equal(player.sampleAt(t - 1).buttonsMask, 0, `loop ${cycle} idle before +${offsetMs}`);
      assert.equal(player.sampleAt(t).buttonsMask, expectedBtn, `loop ${cycle} press at +${offsetMs}`);
      assert.equal(player.sampleAt(t + CHROME_MENU_FLOW_4_HOLD_MS - 1).buttonsMask, expectedBtn);
      assert.equal(player.sampleAt(t + CHROME_MENU_FLOW_4_HOLD_MS).buttonsMask, 0);
      assert.equal(player.pressCountAt(t), expectedCount);
    }
  }

  // Full 300-second invariant: NEVER presses B; only A, DPAD_UP, or DPAD_DOWN.
  const allowed = new Set([0, INPUT_BUTTONS.A, INPUT_BUTTONS.DPAD_UP, INPUT_BUTTONS.DPAD_DOWN]);
  for (let t = 0; t <= 300000; t += 50) {
    const mask = player.sampleAt(t).buttonsMask;
    assert.equal(mask & INPUT_BUTTONS.B, 0, `chrome-menu-flow-4 must never press B at t=${t}`);
    assert.ok(allowed.has(mask), `unexpected button mask 0x${mask.toString(16)} at t=${t}`);
  }

  // Exact nextTransitionMs across phase boundaries.
  assert.equal(player.nextTransitionMs(0), 5000);
  assert.equal(player.nextTransitionMs(5000), 5200);
  assert.equal(player.nextTransitionMs(75200), 80000);
  assert.equal(player.nextTransitionMs(170700), 178000);
  assert.equal(player.nextTransitionMs(178200), 180000);
  assert.equal(player.nextTransitionMs(180200), 190000);
  assert.equal(player.nextTransitionMs(196700), 204000);

  // Verify 15 FPS (67 ms frame cadence) and 30 FPS (33 ms frame cadence) sampling catches every press.
  for (const frameMs of [33, 67]) {
    for (const ev of spec.events) {
      let seenFrames = 0;
      const firstFrame = Math.ceil(ev.atMs / frameMs) * frameMs;
      for (let f = firstFrame; f < ev.atMs + ev.holdMs; f += frameMs) {
        if (player.sampleAt(f).buttonsMask === ev.button) {
          seenFrames += 1;
        }
      }
      assert.ok(
        seenFrames >= 2,
        `event at ${ev.atMs} (hold ${ev.holdMs} ms) must span >= 2 frames at ${frameMs} ms/frame (got ${seenFrames})`
      );
    }
  }

  // Composes cleanly with maxPresses, endMs, and startOffsetMs.
  const capped = new ScriptedInputPlayer({ preset: 'chrome-menu-flow-4', maxPresses: 95 });
  assert.equal(capped.sampleAt(190000).buttonsMask, INPUT_BUTTONS.DPAD_UP); // #95 (1st loop DUp)
  assert.equal(capped.sampleAt(190400).buttonsMask, 0);
  assert.equal(capped.pressCountAt(300000), 95);
  assert.equal(capped.nextTransitionMs(190200), null);

  const shifted = new ScriptedInputPlayer('chrome-menu-flow-4', { startOffsetMs: 12000 });
  assert.equal(shifted.sampleAt(5000).buttonsMask, 0);
  assert.equal(shifted.sampleAt(17000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(shifted.sampleAt(190000).buttonsMask, INPUT_BUTTONS.DPAD_UP); // 178000 + 12000
  assert.equal(shifted.sampleAt(192000).buttonsMask, INPUT_BUTTONS.A);       // 180000 + 12000
  assert.equal(shifted.sampleAt(202000).buttonsMask, INPUT_BUTTONS.DPAD_UP); // loop 190000 + 12000
});

test('W355 minHoldMs upgrades legacy presets (chrome-menu-flow, chrome-menu-flow-2, start-screen) to >= 180 ms without breaking defaults', () => {
  // Default legacy presets remain 100% backwards-compatible.
  const legacyFlow2 = new ScriptedInputPlayer('chrome-menu-flow-2');
  assert.equal(legacyFlow2.minHoldMs, 0);
  assert.equal(legacyFlow2.sampleAt(110099).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(legacyFlow2.sampleAt(110100).buttonsMask, 0);

  // minHoldMs >= 180 extends fixed events, loop events, and repeating presets cleanly.
  const upgradedFlow2 = new ScriptedInputPlayer('chrome-menu-flow-2', { minHoldMs: 180 });
  assert.equal(upgradedFlow2.minHoldMs, 180);
  assert.equal(upgradedFlow2.sampleAt(110000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(upgradedFlow2.sampleAt(110179).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(upgradedFlow2.sampleAt(110180).buttonsMask, 0);
  // Phase 2 DPAD_DOWN (150 ms -> 180 ms) & Phase 5 loop A (100 ms -> 180 ms).
  assert.equal(upgradedFlow2.sampleAt(169179).buttonsMask, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(upgradedFlow2.sampleAt(169180).buttonsMask, 0);
  assert.equal(upgradedFlow2.sampleAt(360179).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(upgradedFlow2.sampleAt(360180).buttonsMask, 0);

  // Spec-object form `{ preset, minHoldMs }` on repeating preset.
  const upgradedStart = new ScriptedInputPlayer({ preset: 'start-screen', minHoldMs: 200 });
  assert.equal(upgradedStart.minHoldMs, 200);
  assert.equal(upgradedStart.sampleAt(5199).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(upgradedStart.sampleAt(5200).buttonsMask, 0);

  // Validation rejects negative / non-finite / non-number minHoldMs.
  for (const bad of [-1, Number.NaN, Number.POSITIVE_INFINITY, '180']) {
    assert.throws(
      () => new ScriptedInputPlayer('chrome-menu-flow-2', { minHoldMs: bad }),
      RangeError
    );
    assert.throws(
      () => new ScriptedInputPlayer('start-screen', { minHoldMs: bad }),
      RangeError
    );
  }
});

test('W355 InputBridge starts chrome-menu-flow-4 cleanly on FIRST_PRESENT and never stalls when FIRST_VISIBLE_FRAME waits on a dark Start Screen', () => {
  const clock = createFakeClockAndTimers(0);
  const notifyCalls = [];
  const spyAtomics = {
    load: (arr, idx) => Atomics.load(arr, idx),
    store: (arr, idx, val) => Atomics.store(arr, idx, val),
    add: (arr, idx, val) => Atomics.add(arr, idx, val),
    notify: (arr, idx, count) => {
      notifyCalls.push({ idx, count, seq: Atomics.load(arr, idx) });
      return Atomics.notify(arr, idx, count);
    },
  };

  // 1. startOn: 'FIRST_PRESENT' arms cleanly before FIRST_PRESENT and starts immediately upon
  //    notifyFirstPresent() WITHOUT waiting for FIRST_VISIBLE_FRAME!
  const bridge1 = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
    atomicsObj: spyAtomics,
  });

  bridge1.startScript('chrome-menu-flow-4', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_PRESENT,
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });

  assert.equal(bridge1.getScriptState().active, false);
  assert.equal(bridge1.getScriptState().preset, 'chrome-menu-flow-4');
  assert.equal(bridge1.getStats().scriptPending, true);
  assert.equal(bridge1.getStats().firstPresentNotified, false);
  assert.equal(bridge1.getStats().firstVisibleFrameNotified, false);

  // Advance 25 s before FIRST_PRESENT: script has not started yet.
  clock.advanceBy(25000);
  assert.equal(bridge1.getSnapshot().buttons, 0);
  assert.equal(bridge1.getStats().scriptedPressesSent, 0);

  // FIRST_PRESENT fires while Start Screen is still dark (FIRST_VISIBLE_FRAME NEVER fires).
  assert.equal(bridge1.notifyFirstPresent(), true);
  assert.equal(bridge1.getStats().scriptPending, false);
  assert.equal(bridge1.getStats().scriptActive, true);
  assert.equal(bridge1.getStats().scriptStartedOn, 'FIRST_PRESENT');
  assert.equal(bridge1.getStats().firstPresentNotified, true);
  assert.equal(bridge1.getStats().firstVisibleFrameNotified, false);

  // 5000 ms after FIRST_PRESENT -> first Start Screen A press lands in SAB with Atomics.notify!
  const notifiesBefore = notifyCalls.length;
  clock.advanceBy(5000);
  assert.equal(bridge1.getSnapshot().buttons, INPUT_BUTTONS.A);
  const dv = new DataView(bridge1.sab);
  assert.equal(dv.getUint32(INPUT_OFFSETS.BUTTONS, true), INPUT_BUTTONS.A);
  assert.equal(bridge1.getStats().scriptedPressesSent, 1);
  assert.equal(bridge1.getStats().lastScriptedButton, 'A');
  assert.ok(notifyCalls.length > notifiesBefore, 'SAB commit must wake loader via Atomics.notify');
  assert.equal(notifyCalls[notifyCalls.length - 1].idx, INPUT_WORD_INDICES.SEQ);
  assert.equal(notifyCalls[notifyCalls.length - 1].seq % 2, 0);

  // Still held at +199 ms, released at +200 ms.
  clock.advanceBy(199);
  assert.equal(bridge1.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceBy(1);
  assert.equal(bridge1.getSnapshot().buttons, 0);

  // Starting another FIRST_PRESENT script after FIRST_PRESENT already fired starts immediately.
  bridge1.startScript('chrome-menu-flow-4', {
    startOn: 'first-present',
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });
  assert.equal(bridge1.getStats().scriptPending, false);
  assert.equal(bridge1.getStats().scriptActive, true);
  assert.equal(bridge1.getStats().scriptStartedOn, 'FIRST_PRESENT');

  bridge1.destroy();
  assert.equal(clock.pendingTimeouts(), 0);

  // 2. startOn: 'FIRST_VISIBLE_FRAME' (or waitForFirstVisibleFrame: true) MUST NOT stall forever
  //    if FIRST_VISIBLE_FRAME stays waiting on a dark Start Screen: once FIRST_PRESENT fires,
  //    the bounded visibleFrameFallbackMs (default 2000 ms) starts playback automatically!
  const clock2 = createFakeClockAndTimers(0);
  const bridge2 = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });

  bridge2.startScript('chrome-menu-flow-4', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME,
    nowFn: clock2.nowFn,
    setTimeoutFn: clock2.setTimeoutFn,
    clearTimeoutFn: clock2.clearTimeoutFn,
  });
  assert.equal(bridge2.getStats().scriptPending, true);
  assert.equal(bridge2.getStats().scriptActive, false);

  // FIRST_PRESENT fires, but FIRST_VISIBLE_FRAME never arrives (dark Start Screen).
  assert.equal(bridge2.notifyMilestone('FIRST_PRESENT'), false);
  assert.equal(bridge2.getStats().scriptPending, true);
  assert.equal(bridge2.getStats().firstPresentNotified, true);

  // Right before fallback deadline (1999 ms): still pending; at 2000 ms: starts cleanly!
  clock2.advanceBy(DEFAULT_VISIBLE_FRAME_FALLBACK_MS - 1);
  assert.equal(bridge2.getStats().scriptActive, false);
  clock2.advanceBy(1);
  assert.equal(bridge2.getStats().scriptPending, false);
  assert.equal(bridge2.getStats().scriptActive, true);
  assert.equal(bridge2.getStats().scriptStartedOn, 'FIRST_PRESENT_FALLBACK');

  // 5000 ms later -> Start Screen A fires into SAB!
  clock2.advanceBy(5000);
  assert.equal(bridge2.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge2.getStats().scriptedPressesSent, 1);

  // Late notifyFirstVisibleFrame() after fallback already started is a safe no-op (no restart).
  assert.equal(bridge2.notifyFirstVisibleFrame(), false);
  assert.equal(bridge2.getStats().scriptedPressesSent, 1);
  assert.equal(bridge2.getStats().firstVisibleFrameNotified, true);

  bridge2.destroy();
  assert.equal(clock2.pendingTimeouts(), 0);

  // 3. When FIRST_VISIBLE_FRAME DOES arrive before visibleFrameFallbackMs, script starts
  //    immediately on FIRST_VISIBLE_FRAME and cancels the fallback timer (no double start).
  const clock3 = createFakeClockAndTimers(0);
  const bridge3 = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  bridge3.startScript('chrome-menu-flow-4', {
    waitForFirstVisibleFrame: true,
    visibleFrameFallbackMs: 5000,
    nowFn: clock3.nowFn,
    setTimeoutFn: clock3.setTimeoutFn,
    clearTimeoutFn: clock3.clearTimeoutFn,
  });
  bridge3.notifyFirstPresent();
  clock3.advanceBy(1000);
  assert.equal(bridge3.getStats().scriptPending, true);
  assert.equal(bridge3.notifyMilestone('first-visible-frame'), true);
  assert.equal(bridge3.getStats().scriptActive, true);
  assert.equal(bridge3.getStats().scriptPending, false);
  assert.equal(bridge3.getStats().scriptStartedOn, 'FIRST_VISIBLE_FRAME');

  // Passing the original 5000 ms fallback mark does not reset elapsedScriptMs.
  clock3.advanceBy(5000);
  assert.equal(bridge3.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge3.getStats().elapsedScriptMs, 5000);

  bridge3.destroy();
  assert.equal(clock3.pendingTimeouts(), 0);

  // 4. visibleFrameFallbackMs: 0 starts synchronously on notifyFirstPresent(), and
  //    firstPresentTimeoutMs prevents stalling even if notifyFirstPresent() never fires.
  const clock4 = createFakeClockAndTimers(0);
  const bridge4 = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  bridge4.startScript('chrome-menu-flow-4', {
    startOn: 'FIRST_VISIBLE_FRAME',
    visibleFrameFallbackMs: 0,
    nowFn: clock4.nowFn,
    setTimeoutFn: clock4.setTimeoutFn,
    clearTimeoutFn: clock4.clearTimeoutFn,
  });
  assert.equal(bridge4.notifyFirstPresent(), true);
  assert.equal(bridge4.getStats().scriptStartedOn, 'FIRST_PRESENT_FALLBACK');
  bridge4.stopScript();

  const bridge5 = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  bridge5.startScript('chrome-menu-flow-4', {
    startOn: 'FIRST_PRESENT',
    firstPresentTimeoutMs: 4000,
    nowFn: clock4.nowFn,
    setTimeoutFn: clock4.setTimeoutFn,
    clearTimeoutFn: clock4.clearTimeoutFn,
  });
  clock4.advanceBy(4000);
  assert.equal(bridge5.getStats().scriptActive, true);
  assert.equal(bridge5.getStats().scriptStartedOn, 'FIRST_PRESENT_TIMEOUT');

  // stopScript() cancels pending timers cleanly with zero leaks.
  const bridge6 = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  bridge6.startScript('chrome-menu-flow-4', {
    startOn: 'FIRST_PRESENT',
    firstPresentTimeoutMs: 10000,
    nowFn: clock4.nowFn,
    setTimeoutFn: clock4.setTimeoutFn,
    clearTimeoutFn: clock4.clearTimeoutFn,
  });
  bridge6.stopScript();
  assert.equal(bridge6.getStats().scriptPending, false);
  assert.equal(bridge6.notifyFirstPresent(), false);

  bridge4.destroy();
  bridge5.destroy();
  bridge6.destroy();
  assert.equal(clock4.pendingTimeouts(), 0);

  // 5. Validation rejects unknown startOn and negative/non-finite fallback timeouts.
  const bridgeVal = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  assert.throws(() => bridgeVal.startScript('chrome-menu-flow-4', { startOn: 'invalid' }), RangeError);
  assert.throws(
    () => bridgeVal.startScript('chrome-menu-flow-4', { startOn: 'FIRST_VISIBLE_FRAME', visibleFrameFallbackMs: Infinity }),
    RangeError
  );
  assert.throws(
    () => bridgeVal.startScript('chrome-menu-flow-4', { startOn: 'FIRST_VISIBLE_FRAME', visibleFrameFallbackMs: -1 }),
    RangeError
  );
  assert.throws(
    () => bridgeVal.startScript('chrome-menu-flow-4', { startOn: 'FIRST_PRESENT', firstPresentTimeoutMs: -5 }),
    RangeError
  );
  bridgeVal.destroy();
});

test('W355 InputBridge drives chrome-menu-flow-4 end-to-end through Start Screen, LEGAL NOTICES, Post-Legal Up+A, and Main Menu loop', () => {
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });

  bridge.startScript('chrome-menu-flow-4', {
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });
  assert.equal(bridge.getStats().activePreset, 'chrome-menu-flow-4');
  assert.equal(bridge.getStats().scriptStartedOn, 'immediate');

  // Phase 1: Start Screen A at 5000 ms (press #1).
  clock.advanceTo(5000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 1);
  assert.equal(bridge.getStats().lastScriptedButton, 'A');

  // Phase 2: Round 0 cursor-reset DPAD_UP at 80000 ms (29 + 1 = #30), DPAD_DOWN at 83000 ms (#36), A at 86500 ms (#38).
  clock.advanceTo(80000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  assert.equal(bridge.getStats().scriptedPressesSent, 30);
  assert.equal(bridge.getStats().lastScriptedButton, 'DPAD_UP');

  clock.advanceTo(83000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(bridge.getStats().scriptedPressesSent, 36);
  assert.equal(bridge.getStats().lastScriptedButton, 'DPAD_DOWN');

  clock.advanceTo(86500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 38);

  // Phase 3: Post-Legal List Screen DPAD_UP at 178000 ms (#93), A at 180000 ms (#94).
  clock.advanceTo(178000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  assert.equal(bridge.getStats().scriptedPressesSent, 93);

  clock.advanceTo(180000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 94);

  // Phase 4: Looping _menu_cycle starts at 190000 ms (#95 DPAD_UP) -> 196500 ms (#103 A).
  clock.advanceTo(190000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  assert.equal(bridge.getStats().scriptedPressesSent, 95);

  clock.advanceTo(196500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 103);

  bridge.destroy();
  assert.equal(clock.pendingTimeouts(), 0);
});
