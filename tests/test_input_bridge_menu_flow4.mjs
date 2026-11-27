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
  CHROME_LEGAL_TERMS_ACCEPT_OFFSETS,
  DEFAULT_VISIBLE_FRAME_FALLBACK_MS,
  FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS,
  FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS,
  FIRST_VISIBLE_FRAME_ANCHOR_PRESETS,
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
  assert.equal(spec.events.length, 196); // 29 Phase 1 + 63 Phase 2 + 2 Phase 3 + 10 Phase 4a + 63 Phase 4b + 28 Phase 4c legal-terms + 1 Phase 4c A
  assert.equal(spec.events[0].atMs, 5000);
  assert.equal(spec.events[93].atMs, 180000); // Phase 3 end preserved intact
  assert.equal(spec.events[spec.events.length - 1].atMs, 358000);
  assert.equal(spec.loop.startMs, 368000);
  assert.equal(spec.loop.periodMs, 16000);
  assert.equal(spec.loop.events.length, 14);

  // Every fixed and loop event must have holdMs >= 180 ms (>= CHROME_MENU_FLOW_4_HOLD_MS),
  // strictly increasing timestamps, zero overlap, and >= 200 ms released gap between presses.
  for (let i = 0; i < spec.events.length; i++) {
    const ev = spec.events[i];
    assert.ok(ev.holdMs >= 180, `fixed event ${i} holdMs (${ev.holdMs}) must be >= 180 ms`);
    assert.ok(
      ev.holdMs >= CHROME_MENU_FLOW_4_HOLD_MS,
      `fixed event ${i} holdMs (${ev.holdMs}) must be >= ${CHROME_MENU_FLOW_4_HOLD_MS}`
    );
    if (i < 167) {
      assert.equal(ev.holdMs, CHROME_MENU_FLOW_4_HOLD_MS);
    }
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
    assert.ok(
      ev.holdMs >= CHROME_MENU_FLOW_4_HOLD_MS,
      `loop event ${i} holdMs (${ev.holdMs}) must be >= ${CHROME_MENU_FLOW_4_HOLD_MS}`
    );
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

  // Quiet transition pause (180200..183999 ms) before Phase 4a (W361).
  for (let t = 180200; t < 184000; t += 200) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `transition settle before Phase 4a at ${t}`);
  }

  // Phase 4a (184s..212s, W361): 10 spaced A confirmations every 3000 ms (holdMs: 200, never B).
  for (let i = 0; i < 10; i++) {
    expectedCount += 1;
    const atMs = 184000 + i * 3000;
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0, `Phase 4a idle before ${atMs}`);
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A, `Phase 4a A press at ${atMs}`);
    assert.equal(
      player.sampleAt(atMs + CHROME_MENU_FLOW_4_HOLD_MS - 1).buttonsMask,
      INPUT_BUTTONS.A,
      `Phase 4a A held 200 ms at ${atMs}`
    );
    assert.equal(player.sampleAt(atMs + CHROME_MENU_FLOW_4_HOLD_MS).buttonsMask, 0);
    assert.equal(player.pressCountAt(atMs), expectedCount);
  }
  assert.equal(expectedCount, 104);

  // Quiet transition pause (211200..217999 ms) before Phase 4b LEGAL NOTICES rounds.
  for (let t = 211200; t < 218000; t += 200) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `quiet pause before Phase 4b at ${t}`);
  }

  // Phase 4b (218s..316s, W361): 7 rounds of Timon's 14 s _menu_cycle (Up x6, Down, Down, A).
  for (let round = 0; round < 7; round++) {
    const base = 218000 + round * 14000;
    for (const [offsetMs, expectedBtn] of expectedCycle) {
      expectedCount += 1;
      const t = base + offsetMs;
      assert.equal(player.sampleAt(t - 1).buttonsMask, 0, `Phase 4b round ${round} idle before +${offsetMs}`);
      assert.equal(player.sampleAt(t).buttonsMask, expectedBtn, `Phase 4b round ${round} press at +${offsetMs}`);
      assert.equal(player.sampleAt(t + CHROME_MENU_FLOW_4_HOLD_MS - 1).buttonsMask, expectedBtn);
      assert.equal(player.sampleAt(t + CHROME_MENU_FLOW_4_HOLD_MS).buttonsMask, 0);
      assert.equal(player.pressCountAt(t), expectedCount);
    }
  }
  assert.equal(expectedCount, 167);

  // Quiet settle window (308700..319999 ms) before Phase 4c Post-Legal -> Main Menu probes.
  for (let t = 308700; t < 320000; t += 300) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `quiet settle before Phase 4c at ${t}`);
  }

  // Phase 4c (320s..360s, W362): 2 rounds of 16 s CHROME_LEGAL_TERMS_ACCEPT_OFFSETS
  // (Up x6 -> right-stick scroll -> A [Privacy Policy] -> Down -> right-stick scroll ->
  // A [Terms of Service] -> Down -> A [Confirm >] -> A) followed by A at 358s.
  for (let round = 0; round < 2; round++) {
    const base = 320000 + round * 16000;
    for (const step of CHROME_LEGAL_TERMS_ACCEPT_OFFSETS) {
      expectedCount += 1;
      const t = base + step.offsetMs;
      const before = player.sampleAt(t - 1);
      assert.equal(before.buttonsMask, 0, `Phase 4c round ${round} idle before +${step.offsetMs}`);
      assert.equal(before.sticks, undefined, `Phase 4c round ${round} sticks idle before +${step.offsetMs}`);
      const duringStart = player.sampleAt(t);
      const duringEnd = player.sampleAt(t + step.holdMs - 1);
      if (step.button !== undefined) {
        assert.equal(duringStart.buttonsMask, step.button, `Phase 4c round ${round} press at +${step.offsetMs}`);
        assert.equal(duringEnd.buttonsMask, step.button);
      } else {
        assert.equal(duringStart.buttonsMask, 0);
        assert.equal(duringStart.sticks?.ry, step.sticks.ry, `Phase 4c round ${round} scroll at +${step.offsetMs}`);
        assert.equal(duringEnd.sticks?.ry, step.sticks.ry);
      }
      const after = player.sampleAt(t + step.holdMs);
      assert.equal(after.buttonsMask, 0);
      assert.equal(after.sticks, undefined);
      assert.equal(player.pressCountAt(t), expectedCount);
    }
  }
  expectedCount += 1;
  assert.equal(player.sampleAt(357999).buttonsMask, 0);
  assert.equal(player.sampleAt(358000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(358219).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(358220).buttonsMask, 0);
  assert.equal(player.pressCountAt(358000), expectedCount);
  assert.equal(expectedCount, 196);

  // Quiet ~10 s settle window (358220..367999 ms) before Phase 5 fallback loop.
  for (let t = 358500; t < 368000; t += 500) {
    assert.equal(player.sampleAt(t).buttonsMask, 0, `transition settle before Phase 5 loop at ${t}`);
  }

  // Phase 5 (368s+): Looping 16 s CHROME_LEGAL_TERMS_ACCEPT_OFFSETS.
  for (let cycle = 0; cycle < 3; cycle++) {
    const base = 368000 + cycle * 16000;
    for (const step of CHROME_LEGAL_TERMS_ACCEPT_OFFSETS) {
      expectedCount += 1;
      const t = base + step.offsetMs;
      assert.equal(player.sampleAt(t - 1).buttonsMask, 0, `loop ${cycle} idle before +${step.offsetMs}`);
      const s = player.sampleAt(t);
      if (step.button !== undefined) {
        assert.equal(s.buttonsMask, step.button, `loop ${cycle} press at +${step.offsetMs}`);
        assert.equal(player.sampleAt(t + step.holdMs - 1).buttonsMask, step.button);
      } else {
        assert.equal(s.buttonsMask, 0);
        assert.equal(s.sticks?.ry, step.sticks.ry, `loop ${cycle} scroll at +${step.offsetMs}`);
        assert.equal(player.sampleAt(t + step.holdMs - 1).sticks?.ry, step.sticks.ry);
      }
      assert.equal(player.sampleAt(t + step.holdMs).buttonsMask, 0);
      assert.equal(player.sampleAt(t + step.holdMs).sticks, undefined);
      assert.equal(player.pressCountAt(t), expectedCount);
    }
  }

  // Full 420-second invariant: NEVER presses B; only A, DPAD_UP, or DPAD_DOWN.
  const allowed = new Set([0, INPUT_BUTTONS.A, INPUT_BUTTONS.DPAD_UP, INPUT_BUTTONS.DPAD_DOWN]);
  for (let t = 0; t <= 420000; t += 50) {
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
  assert.equal(player.nextTransitionMs(180200), 184000);
  assert.equal(player.nextTransitionMs(211200), 218000);
  assert.equal(player.nextTransitionMs(308700), 320000);
  assert.equal(player.nextTransitionMs(320200), 320450);
  assert.equal(player.nextTransitionMs(322450), 322700);
  assert.equal(player.nextTransitionMs(323600), 324000);
  assert.equal(player.nextTransitionMs(358220), 368000);
  assert.equal(player.nextTransitionMs(380220), 384000);

  // Verify 15 FPS (67 ms frame cadence) and 30 FPS (33 ms frame cadence) sampling catches every press & scroll.
  for (const frameMs of [33, 67]) {
    for (const ev of spec.events) {
      let seenFrames = 0;
      const firstFrame = Math.ceil(ev.atMs / frameMs) * frameMs;
      for (let f = firstFrame; f < ev.atMs + ev.holdMs; f += frameMs) {
        const s = player.sampleAt(f);
        if (ev.button !== undefined && s.buttonsMask === ev.button) {
          seenFrames += 1;
        } else if (ev.sticks !== undefined && s.sticks?.ry === ev.sticks.ry) {
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
  const capped = new ScriptedInputPlayer({ preset: 'chrome-menu-flow-4', maxPresses: 197 });
  assert.equal(capped.sampleAt(368000).buttonsMask, INPUT_BUTTONS.DPAD_UP); // #197 (1st loop DUp)
  assert.equal(capped.sampleAt(368450).buttonsMask, 0);
  assert.equal(capped.pressCountAt(450000), 197);
  assert.equal(capped.nextTransitionMs(368200), null);

  const shifted = new ScriptedInputPlayer('chrome-menu-flow-4', { startOffsetMs: 12000 });
  assert.equal(shifted.sampleAt(5000).buttonsMask, 0);
  assert.equal(shifted.sampleAt(17000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(shifted.sampleAt(190000).buttonsMask, INPUT_BUTTONS.DPAD_UP); // 178000 + 12000
  assert.equal(shifted.sampleAt(192000).buttonsMask, INPUT_BUTTONS.A);       // 180000 + 12000
  assert.equal(shifted.sampleAt(196000).buttonsMask, INPUT_BUTTONS.A);       // 184000 + 12000
  assert.equal(shifted.sampleAt(230000).buttonsMask, INPUT_BUTTONS.DPAD_UP); // 218000 + 12000
  assert.equal(shifted.sampleAt(370000).buttonsMask, INPUT_BUTTONS.A);       // 358000 + 12000
  assert.equal(shifted.sampleAt(380000).buttonsMask, INPUT_BUTTONS.DPAD_UP); // loop 368000 + 12000
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

  // W357: Late notifyFirstVisibleFrame() after fallback already started re-anchors
  // chrome-menu-flow-4 cleanly to FIRST_VISIBLE_FRAME, releases any held fallback button,
  // and fires a crisp 220 ms A press at +500 ms so slow-loading boots never miss Start Screen.
  assert.equal(bridge2.notifyFirstVisibleFrame(), true);
  assert.equal(bridge2.getStats().firstVisibleFrameNotified, true);
  assert.equal(bridge2.getStats().scriptStartedOn, 'FIRST_VISIBLE_FRAME');
  assert.equal(bridge2.getStats().scriptReanchoredCount, 1);
  assert.equal(bridge2.getStats().elapsedScriptMs, 0);
  assert.equal(bridge2.getSnapshot().buttons, 0);

  // Duplicate notifyFirstVisibleFrame() is an idempotent no-op (no double restart).
  assert.equal(bridge2.notifyFirstVisibleFrame(), false);
  assert.equal(bridge2.getStats().scriptReanchoredCount, 1);

  // +500 ms after FIRST_VISIBLE_FRAME -> crisp 220 ms A press lands in SAB!
  clock2.advanceBy(FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS);
  assert.equal(bridge2.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge2.getStats().scriptedPressesSent, 2);
  clock2.advanceBy(FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS - 1);
  assert.equal(bridge2.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock2.advanceBy(1);
  assert.equal(bridge2.getSnapshot().buttons, 0);

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

  // Phase 4a (W361): 10 spaced A confirmations from 184000 ms (#95) through 211000 ms (#104).
  clock.advanceTo(184000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 95);

  clock.advanceTo(211000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 104);

  // Phase 4b (W361): 7 rounds of 14 s _menu_cycle from 218000 ms (#105 DPAD_UP -> #113 A).
  clock.advanceTo(218000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  assert.equal(bridge.getStats().scriptedPressesSent, 105);

  clock.advanceTo(224500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 113);

  // Phase 4c (W362): 2 rounds of ROCKSTAR GAMES LEGAL TERMS & POLICIES checkbox-accept
  // (320s DPAD_UP #168, 322.7s ry scroll, 324s A Privacy Policy #174, 325.2s DPAD_DOWN #175,
  // 325.8s ry scroll, 327.1s A Terms of Service #176, 328.3s DPAD_DOWN #177, 329.5s A Confirm > #178)
  // followed by A at 358s (#192).
  clock.advanceTo(320000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  assert.equal(bridge.getStats().scriptedPressesSent, 168);

  clock.advanceTo(322700);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.ok(Math.abs(bridge.getSnapshot().ry - -0.95) < 1e-5);

  clock.advanceTo(324000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getSnapshot().ry, 0);
  assert.equal(bridge.getStats().scriptedPressesSent, 174);

  clock.advanceTo(325200);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(bridge.getStats().scriptedPressesSent, 175);

  clock.advanceTo(327100);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 176);

  clock.advanceTo(328300);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  assert.equal(bridge.getStats().scriptedPressesSent, 177);

  clock.advanceTo(329500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 178);

  clock.advanceTo(358000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 192);

  // Phase 5: Looping 16 s legal-terms-accept starts at 368000 ms (#193 DPAD_UP) -> 377500 ms (#203 A Confirm >).
  clock.advanceTo(368000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  assert.equal(bridge.getStats().scriptedPressesSent, 193);

  clock.advanceTo(377500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().scriptedPressesSent, 203);

  bridge.destroy();
  assert.equal(clock.pendingTimeouts(), 0);
});

test('W357 chrome-menu-flow-4 re-anchors to FIRST_VISIBLE_FRAME 90 s after FIRST_PRESENT, fires 220 ms A within 500 ms, and runs Phases 1->4 cleanly', () => {
  assert.deepEqual(FIRST_VISIBLE_FRAME_ANCHOR_PRESETS, [
    'menu-to-ingame',
    'chrome-to-ingame',
    'chrome-menu-flow-6',
    'chrome-menu-flow-5',
    'chrome-menu-flow-4',
    'chrome-menu-flow-2',
    'chrome-menu-flow',
  ]);
  assert.ok(FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS <= 500);
  assert.ok(FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS >= 200);

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

  const bridge = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
    atomicsObj: spyAtomics,
  });

  bridge.startScript('chrome-menu-flow-4', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME,
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });

  // Simulate live Chrome t86: FIRST_PRESENT fires at 45.1 s, black asset loading lasts 90 s.
  clock.advanceTo(45100);
  assert.equal(bridge.notifyFirstPresent(), false);

  // Fallback starts at FIRST_PRESENT + 2000 ms (W355 compatibility if visible frame never fires).
  clock.advanceBy(DEFAULT_VISIBLE_FRAME_FALLBACK_MS);
  assert.equal(bridge.getStats().scriptStartedOn, 'FIRST_PRESENT_FALLBACK');
  assert.equal(bridge.getStats().scriptActive, true);

  // Advance 90 s after FIRST_PRESENT (t = 135100 ms): Phase 1 (5s..75s) already finished!
  // Fallback is now deep in Phase 2 (88 s elapsed since fallback).
  clock.advanceTo(45100 + 90000);
  assert.equal(bridge.getStats().elapsedScriptMs, 88000);
  const pressesBeforeVisible = bridge.getStats().scriptedPressesSent;
  assert.ok(pressesBeforeVisible >= 29, 'Phase 1 already elapsed before FIRST_VISIBLE_FRAME');

  // FIRST_VISIBLE_FRAME finally arrives at FIRST_PRESENT + 90 s (Start Screen OFFLINEPLAYER [A] visible!).
  assert.equal(bridge.notifyFirstVisibleFrame(), true);
  assert.equal(bridge.getStats().scriptStartedOn, 'FIRST_VISIBLE_FRAME');
  assert.equal(bridge.getStats().scriptReanchoredCount, 1);
  assert.equal(bridge.getStats().elapsedScriptMs, 0);
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Crisp >= 200 ms (220 ms) A press fires within <= 600 ms (at +500 ms) of FIRST_VISIBLE_FRAME!
  const notifiesBefore = notifyCalls.length;
  clock.advanceBy(FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS - 1);
  assert.equal(bridge.getSnapshot().buttons, 0);
  clock.advanceBy(1);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  const dv = new DataView(bridge.sab);
  assert.equal(dv.getUint32(INPUT_OFFSETS.BUTTONS, true), INPUT_BUTTONS.A);
  assert.equal(bridge.getStats().lastScriptedButton, 'A');
  assert.equal(bridge.getStats().scriptedPressesSent, pressesBeforeVisible + 1);
  assert.ok(notifyCalls.length > notifiesBefore, 'Re-anchored Start Screen A must wake SAB via Atomics.notify');

  // Held for the full 220 ms (>= 200 ms), then cleanly released.
  clock.advanceBy(FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS - 1);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceBy(1);
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Second Start Screen safety A press at +2500 ms (still before 3.2 s 0x144DCBC9D watchdog!).
  const visibleAt = 45100 + 90000;
  clock.advanceTo(visibleAt + 2500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 2500 + FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS - 1);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 2500 + FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS);
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Phases 1 -> 2 -> 3 -> 4 all run cleanly synchronized from FIRST_VISIBLE_FRAME!
  // Phase 1: +5000 ms A (200 ms hold)
  clock.advanceTo(visibleAt + 5000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 5200);
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Phase 2: +80000 ms Round 0 cursor-reset DPAD_UP, +83000 ms DPAD_DOWN, +86500 ms A
  clock.advanceTo(visibleAt + 80000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  clock.advanceTo(visibleAt + 83000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  clock.advanceTo(visibleAt + 86500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // Phase 3: +178000 ms Post-Legal List Screen DPAD_UP, +180000 ms A
  clock.advanceTo(visibleAt + 178000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  clock.advanceTo(visibleAt + 180000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // Phase 4 (W361/W362): +184000 ms Phase 4a A -> +218000 ms Phase 4b _menu_cycle -> +320000..+358000 ms Phase 4c checkbox accept
  clock.advanceTo(visibleAt + 184000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 218000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  clock.advanceTo(visibleAt + 224500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 320000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  clock.advanceTo(visibleAt + 324000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 327100);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 329500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 358000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // Phase 5: +368000 ms Looping legal-terms-accept DPAD_UP -> +377500 ms Confirm > A
  clock.advanceTo(visibleAt + 368000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  clock.advanceTo(visibleAt + 377500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // Never presses B at any point in the re-anchored player.
  for (let t = 0; t <= 400000; t += 100) {
    assert.equal(
      bridge._scriptPlayer.sampleAt(t).buttonsMask & INPUT_BUTTONS.B,
      0,
      `re-anchored chrome-menu-flow-4 must never press B at t=${t}`
    );
  }

  bridge.destroy();
  assert.equal(clock.pendingTimeouts(), 0);
});

test('W357 menu-to-ingame re-anchors & reactivates cleanly even if it already auto-completed 90 s after FIRST_PRESENT', () => {
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });

  let completionCount = 0;
  bridge.startScript('menu-to-ingame', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_PRESENT,
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
    onComplete: () => {
      completionCount += 1;
    },
  });

  // FIRST_PRESENT at 45.1 s.
  clock.advanceTo(45100);
  assert.equal(bridge.notifyFirstPresent(), true);
  assert.equal(bridge.getStats().scriptActive, true);

  // Advance 90 s (t = 135.1 s): menu-to-ingame (59.3 s total) already auto-completed on the black loading screen!
  const visibleAt = 45100 + 90000;
  clock.advanceTo(visibleAt);
  assert.equal(bridge.getStats().scriptActive, false);
  assert.equal(completionCount, 1);
  assert.equal(bridge.getSnapshot().buttons, 0);

  // FIRST_VISIBLE_FRAME arrives at +90 s -> re-anchors & reactivates cleanly!
  assert.equal(bridge.notifyFirstVisibleFrame(), true);
  assert.equal(bridge.getStats().scriptActive, true);
  assert.equal(bridge.getStats().scriptStartedOn, 'FIRST_VISIBLE_FRAME');
  assert.equal(bridge.getStats().scriptReanchoredCount, 1);
  assert.equal(bridge.getStats().elapsedScriptMs, 0);

  // Crisp 220 ms A press fires at +500 ms (<= 600 ms) of FIRST_VISIBLE_FRAME!
  clock.advanceTo(visibleAt + FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS + FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS - 1);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS + FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS);
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Second Start Screen safety A press at +2500 ms (< 3.2 s watchdog).
  clock.advanceTo(visibleAt + 2500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // Phases 1 -> 5 run cleanly from the FIRST_VISIBLE_FRAME anchor:
  clock.advanceTo(visibleAt + 5000); // Phase 1 Start Screen A
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  clock.advanceTo(visibleAt + 20000); // Phase 2 Legal Notices DPAD_DOWN
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);

  clock.advanceTo(visibleAt + 32500); // Phase 3 Intro skip MENU
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.MENU);

  clock.advanceTo(visibleAt + 45500); // Phase 4 Main Menu A
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  clock.advanceTo(visibleAt + 50000); // Phase 5 In-game left stick forward
  assert.ok(Math.abs(bridge.getSnapshot().ly - 0.85) < 1e-5);

  // Completes cleanly at +59300 ms from FIRST_VISIBLE_FRAME.
  clock.advanceTo(visibleAt + 59300);
  assert.equal(bridge.getStats().scriptActive, false);
  assert.equal(completionCount, 2);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.equal(bridge.getSnapshot().ly, 0);

  bridge.destroy();
  assert.equal(clock.pendingTimeouts(), 0);
});

test('W357 chrome-menu-flow-2, chrome-menu-flow, immediate-started scripts, mid-hold release, opt-out, and manual stop behave cleanly', () => {
  // 1. chrome-menu-flow-2 and chrome-menu-flow also re-anchor and schedule +0.5s, +2.5s, +4.5s A presses.
  for (const [preset, phase1AtMs] of [
    ['chrome-menu-flow-2', 110000],
    ['chrome-menu-flow', 165000],
  ]) {
    const clock = createFakeClockAndTimers(0);
    const bridge = new InputBridge({
      windowObj: null,
      documentObj: null,
      navigatorObj: null,
    });
    bridge.startScript(preset, {
      startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_PRESENT,
      nowFn: clock.nowFn,
      setTimeoutFn: clock.setTimeoutFn,
      clearTimeoutFn: clock.clearTimeoutFn,
    });
    clock.advanceTo(30000);
    bridge.notifyFirstPresent();
    clock.advanceBy(90000);
    const visibleAt = clock.nowFn();
    assert.equal(bridge.notifyFirstVisibleFrame(), true, `${preset} must re-anchor on FIRST_VISIBLE_FRAME`);
    assert.equal(bridge.getStats().scriptStartedOn, 'FIRST_VISIBLE_FRAME');

    for (const offset of [500, 2500, 4500]) {
      clock.advanceTo(visibleAt + offset);
      assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A, `${preset} initial A at +${offset} ms`);
      clock.advanceTo(visibleAt + offset + FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS - 1);
      assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
      clock.advanceTo(visibleAt + offset + FIRST_VISIBLE_FRAME_INITIAL_A_HOLD_MS);
      assert.equal(bridge.getSnapshot().buttons, 0);
    }

    // Original Phase 1 runs synchronized from FIRST_VISIBLE_FRAME.
    clock.advanceTo(visibleAt + phase1AtMs);
    assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A, `${preset} Phase 1 A at +${phase1AtMs} ms`);
    bridge.destroy();
    assert.equal(clock.pendingTimeouts(), 0);
  }

  // 2. Immediate-started chrome-menu-flow-4 (e.g. app_bootstrap.js default) releases a mid-hold
  //    button cleanly and re-anchors when FIRST_VISIBLE_FRAME fires 90 s after FIRST_PRESENT.
  const clockImm = createFakeClockAndTimers(0);
  const bridgeImm = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  bridgeImm.startScript('chrome-menu-flow-4', {
    nowFn: clockImm.nowFn,
    setTimeoutFn: clockImm.setTimeoutFn,
    clearTimeoutFn: clockImm.clearTimeoutFn,
  });
  clockImm.advanceTo(10000);
  bridgeImm.notifyFirstPresent();
  // Advance directly into a held DPAD_UP in Phase 2 (t = 80050 ms).
  clockImm.advanceTo(80050);
  assert.equal(bridgeImm.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  // FIRST_VISIBLE_FRAME fires mid-hold -> immediately clears DPAD_UP to 0 and schedules +500 ms A!
  assert.equal(bridgeImm.notifyFirstVisibleFrame(), true);
  assert.equal(bridgeImm.getSnapshot().buttons, 0);
  clockImm.advanceBy(FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS);
  assert.equal(bridgeImm.getSnapshot().buttons, INPUT_BUTTONS.A);
  bridgeImm.destroy();
  assert.equal(clockImm.pendingTimeouts(), 0);

  // 3. Non-anchor preset ('start-screen'), explicit opt-out (reanchorOnFirstVisibleFrame: false),
  //    and manually stopped scripts do NOT re-anchor or revive on late notifyFirstVisibleFrame().
  const clockNoop = createFakeClockAndTimers(0);
  const bridgeStartScreen = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  bridgeStartScreen.startScript('start-screen', {
    startOn: 'FIRST_PRESENT',
    nowFn: clockNoop.nowFn,
    setTimeoutFn: clockNoop.setTimeoutFn,
    clearTimeoutFn: clockNoop.clearTimeoutFn,
  });
  bridgeStartScreen.notifyFirstPresent();
  clockNoop.advanceBy(5000);
  assert.equal(bridgeStartScreen.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridgeStartScreen.notifyFirstVisibleFrame(), false);
  assert.equal(bridgeStartScreen.getStats().scriptStartedOn, 'FIRST_PRESENT');
  assert.equal(bridgeStartScreen.getStats().scriptReanchoredCount, 0);
  bridgeStartScreen.destroy();

  const bridgeOptOut = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  bridgeOptOut.startScript('chrome-menu-flow-4', {
    startOn: 'FIRST_PRESENT',
    reanchorOnFirstVisibleFrame: false,
    nowFn: clockNoop.nowFn,
    setTimeoutFn: clockNoop.setTimeoutFn,
    clearTimeoutFn: clockNoop.clearTimeoutFn,
  });
  bridgeOptOut.notifyFirstPresent();
  clockNoop.advanceBy(10000);
  assert.equal(bridgeOptOut.notifyFirstVisibleFrame(), false);
  assert.equal(bridgeOptOut.getStats().scriptStartedOn, 'FIRST_PRESENT');
  assert.equal(bridgeOptOut.getStats().scriptReanchoredCount, 0);
  bridgeOptOut.destroy();

  const bridgeStopped = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });
  bridgeStopped.startScript('chrome-menu-flow-4', {
    startOn: 'FIRST_PRESENT',
    nowFn: clockNoop.nowFn,
    setTimeoutFn: clockNoop.setTimeoutFn,
    clearTimeoutFn: clockNoop.clearTimeoutFn,
  });
  bridgeStopped.notifyFirstPresent();
  bridgeStopped.stopScript();
  assert.equal(bridgeStopped.notifyFirstVisibleFrame(), false);
  assert.equal(bridgeStopped.getStats().scriptActive, false);
  assert.equal(bridgeStopped.getStats().scriptReanchoredCount, 0);
  bridgeStopped.destroy();
  assert.equal(clockNoop.pendingTimeouts(), 0);
});

test('W361 chrome-menu-flow-4 Phase 4 drives long-Bink2 runs (t89: UI TEXT SIZE at +175.2 s) through LEGAL NOTICES and Main Menu to 358 s', () => {
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });

  bridge.startScript('chrome-menu-flow-4', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME,
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });

  // Simulate live Chrome t89: FIRST_VISIBLE_FRAME fires at 176.8 s after page load.
  const visibleAt = 176800;
  clock.advanceTo(45000);
  bridge.notifyFirstPresent();
  clock.advanceTo(visibleAt);
  assert.equal(bridge.notifyFirstVisibleFrame(), true);
  assert.equal(bridge.getStats().scriptStartedOn, 'FIRST_VISIBLE_FRAME');

  // +500 ms initial A press clears Start Screen into 6-star -> 4K Bink2 -> Rockstar -> UI TEXT SIZE.
  clock.advanceTo(visibleAt + FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // Suppose 4K Bink2 finishes late so UI TEXT SIZE appears at +175.2 s (352 s wall clock):
  // Phases 1-3 remain intact through +180 s, and Phase 4a seamlessly continues with 10 spaced A
  // confirmations (184s..211s every 3000 ms, 200 ms hold) through UI TEXT SIZE -> FINISH SETUP?.
  for (let i = 0; i < 10; i++) {
    const atMs = 184000 + i * 3000;
    clock.advanceTo(visibleAt + atMs);
    assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A, `Phase 4a A #${i} at +${atMs} ms`);
    clock.advanceTo(visibleAt + atMs + CHROME_MENU_FLOW_4_HOLD_MS - 1);
    assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A, `Phase 4a A #${i} held >= 180 ms`);
    clock.advanceTo(visibleAt + atMs + CHROME_MENU_FLOW_4_HOLD_MS);
    assert.equal(bridge.getSnapshot().buttons, 0, `Phase 4a A #${i} released cleanly`);
  }

  // Phase 4b (218s..316s): 7 rounds of Up x6, Down, Down, A every 14000 ms for LEGAL NOTICES ("Next >").
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
    const base = 218000 + round * 14000;
    for (const [offsetMs, btn] of expectedCycle) {
      clock.advanceTo(visibleAt + base + offsetMs);
      assert.equal(bridge.getSnapshot().buttons, btn, `Phase 4b round ${round} +${offsetMs} ms`);
      clock.advanceTo(visibleAt + base + offsetMs + CHROME_MENU_FLOW_4_HOLD_MS - 1);
      assert.equal(bridge.getSnapshot().buttons, btn);
      clock.advanceTo(visibleAt + base + offsetMs + CHROME_MENU_FLOW_4_HOLD_MS);
      assert.equal(bridge.getSnapshot().buttons, 0);
    }
  }

  // Phase 4c (320s..360s, W362): 2 rounds of ROCKSTAR GAMES LEGAL TERMS & POLICIES 2-checkbox +
  // Confirm > sequence (320s..352s: Up x6 -> scroll -> A -> Down -> scroll -> A -> Down -> A -> A)
  // followed by A at 358s.
  for (let round = 0; round < 2; round++) {
    const base = 320000 + round * 16000;
    for (const step of CHROME_LEGAL_TERMS_ACCEPT_OFFSETS) {
      const atMs = base + step.offsetMs;
      clock.advanceTo(visibleAt + atMs);
      if (step.button !== undefined) {
        assert.equal(bridge.getSnapshot().buttons, step.button, `Phase 4c round ${round} press at +${atMs} ms`);
      } else {
        assert.equal(bridge.getSnapshot().buttons, 0);
        assert.ok(Math.abs(bridge.getSnapshot().ry - step.sticks.ry) < 1e-5, `Phase 4c round ${round} scroll at +${atMs} ms`);
      }
      clock.advanceTo(visibleAt + atMs + step.holdMs - 1);
      if (step.button !== undefined) {
        assert.equal(bridge.getSnapshot().buttons, step.button);
      } else {
        assert.ok(Math.abs(bridge.getSnapshot().ry - step.sticks.ry) < 1e-5);
      }
      clock.advanceTo(visibleAt + atMs + step.holdMs);
      assert.equal(bridge.getSnapshot().buttons, 0);
      assert.equal(bridge.getSnapshot().ry, 0);
    }
  }
  clock.advanceTo(visibleAt + 358000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 358219);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 358220);
  assert.equal(bridge.getSnapshot().buttons, 0);

  // Zero B presses across the entire 0..380s re-anchored schedule.
  for (let t = 0; t <= 380000; t += 100) {
    assert.equal(
      bridge._scriptPlayer.sampleAt(t).buttonsMask & INPUT_BUTTONS.B,
      0,
      `W361/W362 schedule must never press B at t=${t}`
    );
  }

  bridge.destroy();
  assert.equal(clock.pendingTimeouts(), 0);
});

test('W362 CHROME_LEGAL_TERMS_ACCEPT_OFFSETS executes Up x6 -> scroll -> A -> Down -> scroll -> A -> Down -> A -> A with >= 200 ms holds and zero B', () => {
  assert.equal(Object.isFrozen(CHROME_LEGAL_TERMS_ACCEPT_OFFSETS), true);
  assert.equal(CHROME_LEGAL_TERMS_ACCEPT_OFFSETS.length, 14);

  const expectedSequence = [
    { offsetMs: 0, holdMs: 200, button: INPUT_BUTTONS.DPAD_UP },
    { offsetMs: 450, holdMs: 200, button: INPUT_BUTTONS.DPAD_UP },
    { offsetMs: 900, holdMs: 200, button: INPUT_BUTTONS.DPAD_UP },
    { offsetMs: 1350, holdMs: 200, button: INPUT_BUTTONS.DPAD_UP },
    { offsetMs: 1800, holdMs: 200, button: INPUT_BUTTONS.DPAD_UP },
    { offsetMs: 2250, holdMs: 200, button: INPUT_BUTTONS.DPAD_UP },
    { offsetMs: 2700, holdMs: 900, ry: -0.95 },
    { offsetMs: 4000, holdMs: 220, button: INPUT_BUTTONS.A },
    { offsetMs: 5200, holdMs: 200, button: INPUT_BUTTONS.DPAD_DOWN },
    { offsetMs: 5800, holdMs: 900, ry: -0.95 },
    { offsetMs: 7100, holdMs: 220, button: INPUT_BUTTONS.A },
    { offsetMs: 8300, holdMs: 200, button: INPUT_BUTTONS.DPAD_DOWN },
    { offsetMs: 9500, holdMs: 220, button: INPUT_BUTTONS.A },
    { offsetMs: 12000, holdMs: 220, button: INPUT_BUTTONS.A },
  ];

  for (let i = 0; i < expectedSequence.length; i++) {
    const actual = CHROME_LEGAL_TERMS_ACCEPT_OFFSETS[i];
    const exp = expectedSequence[i];
    assert.equal(Object.isFrozen(actual), true);
    assert.equal(actual.offsetMs, exp.offsetMs);
    assert.equal(actual.holdMs, exp.holdMs);
    assert.ok(actual.holdMs >= CHROME_MENU_FLOW_4_HOLD_MS);
    if (exp.button !== undefined) {
      assert.equal(actual.button, exp.button);
      assert.equal(actual.button & INPUT_BUTTONS.B, 0);
    } else {
      assert.equal(actual.button, undefined);
      assert.equal(Object.isFrozen(actual.sticks), true);
      assert.equal(actual.sticks.ry, exp.ry);
    }
    if (i > 0) {
      const prev = CHROME_LEGAL_TERMS_ACCEPT_OFFSETS[i - 1];
      assert.ok(actual.offsetMs >= prev.offsetMs + prev.holdMs + 200);
    }
  }
  const last = CHROME_LEGAL_TERMS_ACCEPT_OFFSETS[CHROME_LEGAL_TERMS_ACCEPT_OFFSETS.length - 1];
  assert.ok(last.offsetMs + last.holdMs < 16000, 'cycle must fit comfortably within 16000 ms');
});

test('W362 chrome-to-ingame preset drives Start Screen -> LEGAL NOTICES -> Legal Terms 2-checkbox + Confirm > -> Main Menu -> in-game probes bounded by 420 s with zero B', () => {
  assert.ok(INPUT_SCRIPT_PRESET_NAMES.includes('chrome-to-ingame'));
  assert.ok(FIRST_VISIBLE_FRAME_ANCHOR_PRESETS.includes('chrome-to-ingame'));

  const spec = INPUT_SCRIPT_PRESETS['chrome-to-ingame'];
  assert.ok(spec);
  assert.equal(Object.isFrozen(spec), true);
  assert.equal(spec.name, 'chrome-to-ingame');
  assert.equal(spec.loop, undefined);
  // 24 Phase 1 (5s..74s) + 81 Phase 2 (9x9: 80s..206s warm+cold) + 56 Phase 3 (4x14) + 16 Phase 4a + 25 Phase 4b (4x6 + 1) = 202 events
  assert.equal(spec.events.length, 202);
  assert.equal(spec.events[0].atMs, 5000);
  assert.equal(spec.events[spec.events.length - 1].atMs, 418000);

  // Every event is frozen, >= 180 ms hold, strictly increasing, >= 200 ms gap, and finishes <= 420000 ms.
  for (let i = 0; i < spec.events.length; i++) {
    const ev = spec.events[i];
    assert.equal(Object.isFrozen(ev), true);
    assert.ok(ev.holdMs >= MIN_SCRIPT_HOLD_MS, `event ${i} holdMs (${ev.holdMs}) must be >= 180 ms`);
    assert.ok(ev.atMs + ev.holdMs <= 420000, `event ${i} must complete within 420 s`);
    if (ev.sticks) {
      assert.equal(Object.isFrozen(ev.sticks), true);
    }
    if (i > 0) {
      const prev = spec.events[i - 1];
      assert.ok(
        ev.atMs >= prev.atMs + prev.holdMs + 200,
        `event ${i} (${ev.atMs}) must leave >= 200 ms gap after event ${i - 1} end (${prev.atMs + prev.holdMs})`
      );
    }
  }

  const player = new ScriptedInputPlayer('chrome-to-ingame');
  assert.equal(player.preset, 'chrome-to-ingame');
  assert.equal(player.nextTransitionMs(0), 5000);

  // Phase 1 (5s..74s): Spaced A every 3000 ms (holdMs: 220, 24 presses: 5000..74000 ms) — stops
  // before warm LEGAL NOTICES (~80s) so plain A never opens Row 0 ("Copyright").
  let expectedCount = 0;
  for (let atMs = 5000; atMs <= 74000; atMs += 3000) {
    expectedCount += 1;
    assert.equal(player.sampleAt(atMs - 1).buttonsMask, 0);
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(atMs + 219).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(atMs + 220).buttonsMask, 0);
    assert.equal(player.pressCountAt(atMs), expectedCount);
  }
  assert.equal(expectedCount, 24);

  // Phase 2 (80s..206s): 9 rounds of 14 s LEGAL NOTICES _menu_cycle (Up x6, Down, Down, A)
  // covering both warm (~80..110s) and cold (~150..198s) boots cleanly.
  const expectedMenuCycle = [
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
  for (let round = 0; round < 9; round++) {
    const base = 80000 + round * 14000;
    for (const [offsetMs, btn] of expectedMenuCycle) {
      expectedCount += 1;
      const t = base + offsetMs;
      assert.equal(player.sampleAt(t - 1).buttonsMask, 0);
      assert.equal(player.sampleAt(t).buttonsMask, btn);
      assert.equal(player.sampleAt(t + 199).buttonsMask, btn);
      assert.equal(player.sampleAt(t + 200).buttonsMask, 0);
      assert.equal(player.pressCountAt(t), expectedCount);
    }
  }
  assert.equal(expectedCount, 105);

  // Phase 3 (210s..274s): 4 rounds of 16 s CHROME_LEGAL_TERMS_ACCEPT_OFFSETS (Up x6 -> A -> Down -> A -> Down -> A).
  for (let round = 0; round < 4; round++) {
    const base = 210000 + round * 16000;
    for (const step of CHROME_LEGAL_TERMS_ACCEPT_OFFSETS) {
      expectedCount += 1;
      const t = base + step.offsetMs;
      const s = player.sampleAt(t);
      if (step.button !== undefined) {
        assert.equal(s.buttonsMask, step.button, `Phase 3 round ${round} +${step.offsetMs}`);
        assert.equal(player.sampleAt(t + step.holdMs - 1).buttonsMask, step.button);
      } else {
        assert.equal(s.buttonsMask, 0);
        assert.equal(s.sticks?.ry, step.sticks.ry, `Phase 3 round ${round} scroll +${step.offsetMs}`);
        assert.equal(player.sampleAt(t + step.holdMs - 1).sticks?.ry, step.sticks.ry);
      }
      assert.equal(player.sampleAt(t + step.holdMs).buttonsMask, 0);
      assert.equal(player.sampleAt(t + step.holdMs).sticks, undefined);
      assert.equal(player.pressCountAt(t), expectedCount);
    }
  }
  assert.equal(expectedCount, 161);

  // Phase 4a (280s..340s): Main Menu -> Story Mode spaced A every 4000 ms (16 presses).
  for (let atMs = 280000; atMs <= 340000; atMs += 4000) {
    expectedCount += 1;
    assert.equal(player.sampleAt(atMs).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(atMs + 219).buttonsMask, INPUT_BUTTONS.A);
    assert.equal(player.sampleAt(atMs + 220).buttonsMask, 0);
    assert.equal(player.pressCountAt(atMs), expectedCount);
  }
  assert.equal(expectedCount, 177);

  // Phase 4b (346s..420s): Gentle in-game left-stick forward + right-stick look sweeps + A confirms.
  assert.equal(player.sampleAt(346000).buttonsMask, INPUT_BUTTONS.A);
  assert.deepEqual(player.sampleAt(348500).sticks, { lx: 0, ly: 0.85, rx: 0, ry: 0 });
  assert.deepEqual(player.sampleAt(351500).sticks, { lx: 0, ly: 0, rx: 0.7, ry: 0.15 });
  assert.deepEqual(player.sampleAt(354000).sticks, { lx: 0, ly: 0, rx: -0.7, ry: -0.15 });
  assert.deepEqual(player.sampleAt(356500).sticks, { lx: 0.5, ly: 0.75, rx: 0.35, ry: 0 });
  assert.equal(player.sampleAt(360000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(418000).buttonsMask, INPUT_BUTTONS.A);
  assert.equal(player.sampleAt(418220).buttonsMask, 0);
  assert.equal(player.nextTransitionMs(418220), null);
  assert.equal(player.pressCountAt(450000), 202);

  // Zero B presses across entire 425 s window; only A, DPAD_UP, DPAD_DOWN buttons ever used.
  const allowed = new Set([0, INPUT_BUTTONS.A, INPUT_BUTTONS.DPAD_UP, INPUT_BUTTONS.DPAD_DOWN]);
  for (let t = 0; t <= 425000; t += 50) {
    const mask = player.sampleAt(t).buttonsMask;
    assert.equal(mask & INPUT_BUTTONS.B, 0, `chrome-to-ingame must never press B at t=${t}`);
    assert.ok(allowed.has(mask), `unexpected mask 0x${mask.toString(16)} at t=${t}`);
  }

  // Every event spans >= 2 frames at both 15 FPS (67 ms) and 30 FPS (33 ms).
  for (const frameMs of [33, 67]) {
    for (const ev of spec.events) {
      let seenFrames = 0;
      const firstFrame = Math.ceil(ev.atMs / frameMs) * frameMs;
      for (let f = firstFrame; f < ev.atMs + ev.holdMs; f += frameMs) {
        const s = player.sampleAt(f);
        if (ev.button !== undefined && s.buttonsMask === ev.button) {
          seenFrames += 1;
        } else if (ev.sticks !== undefined && s.sticks) {
          seenFrames += 1;
        }
      }
      assert.ok(seenFrames >= 2, `chrome-to-ingame event at ${ev.atMs} must span >= 2 frames at ${frameMs} ms/frame`);
    }
  }
});

test('W362 InputBridge re-anchors chrome-to-ingame to FIRST_VISIBLE_FRAME, writes Legal Terms checkboxes & in-game sticks to SAB, and completes cleanly', () => {
  const clock = createFakeClockAndTimers(0);
  const bridge = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
  });

  let completed = 0;
  bridge.startScript('chrome-to-ingame', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME,
    nowFn: clock.nowFn,
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
    onComplete: () => {
      completed += 1;
    },
  });

  // Simulate 90 s black asset loading between FIRST_PRESENT and FIRST_VISIBLE_FRAME.
  clock.advanceTo(45000);
  bridge.notifyFirstPresent();
  const visibleAt = 135000;
  clock.advanceTo(visibleAt);
  assert.equal(bridge.notifyFirstVisibleFrame(), true);
  assert.equal(bridge.getStats().scriptStartedOn, 'FIRST_VISIBLE_FRAME');
  assert.equal(bridge.getStats().scriptReanchoredCount, 1);

  // Initial +500 ms Start Screen A lands cleanly in SAB.
  clock.advanceTo(visibleAt + FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // Phase 2 LEGAL NOTICES Next > cycle at +150s..+156.5s.
  clock.advanceTo(visibleAt + 150000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  clock.advanceTo(visibleAt + 156500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // Phase 3 ROCKSTAR GAMES LEGAL TERMS & POLICIES checkbox sequence at +210s:
  // Up x6 -> scroll ry:-0.95 -> A (Privacy Policy) -> Down -> scroll ry:-0.95 -> A (ToS) -> Down -> A (Confirm >)
  clock.advanceTo(visibleAt + 210000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_UP);
  clock.advanceTo(visibleAt + 212700);
  assert.equal(bridge.getSnapshot().buttons, 0);
  assert.ok(Math.abs(bridge.getSnapshot().ry - -0.95) < 1e-5);
  clock.advanceTo(visibleAt + 214000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge.getSnapshot().ry, 0);
  clock.advanceTo(visibleAt + 215200);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  clock.advanceTo(visibleAt + 215800);
  assert.ok(Math.abs(bridge.getSnapshot().ry - -0.95) < 1e-5);
  clock.advanceTo(visibleAt + 217100);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 218300);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.DPAD_DOWN);
  clock.advanceTo(visibleAt + 219500);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);

  // Phase 4 Main Menu A -> in-game left-stick forward & right-stick look sweeps.
  clock.advanceTo(visibleAt + 280000);
  assert.equal(bridge.getSnapshot().buttons, INPUT_BUTTONS.A);
  clock.advanceTo(visibleAt + 348000);
  assert.ok(Math.abs(bridge.getSnapshot().ly - 0.85) < 1e-5);
  clock.advanceTo(visibleAt + 351000);
  assert.ok(Math.abs(bridge.getSnapshot().rx - 0.7) < 1e-5);
  assert.ok(Math.abs(bridge.getSnapshot().ry - 0.15) < 1e-5);

  // Completes cleanly at +418220 ms with all buttons and sticks released.
  clock.advanceTo(visibleAt + 420000);
  assert.equal(bridge.getStats().scriptActive, false);
  assert.equal(completed, 1);
  const finalSnap = bridge.getSnapshot();
  assert.equal(finalSnap.buttons, 0);
  assert.equal(finalSnap.lx, 0);
  assert.equal(finalSnap.ly, 0);
  assert.equal(finalSnap.rx, 0);
  assert.equal(finalSnap.ry, 0);

  bridge.destroy();
  assert.equal(clock.pendingTimeouts(), 0);
});

test('Orch-32 P1 #4b: swapchainProbeSupported=true prevents 2s FIRST_PRESENT_FALLBACK from burning presses during black boot', async () => {
  const { PROBE_SUPPORTED_VISIBLE_FRAME_FALLBACK_MS } = await import('../web/input/input_bridge.js');
  assert.equal(PROBE_SUPPORTED_VISIBLE_FRAME_FALLBACK_MS, 180000);

  // 1. With swapchainProbeSupported: true, FIRST_PRESENT does NOT fire the 2s fallback during a
  //    90s black pre-BINK_OPEN boot; zero presses are burned until real FIRST_VISIBLE_FRAME arrives.
  const clock1 = createFakeClockAndTimers(0);
  const bridge1 = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
    swapchainProbeSupported: true,
  });
  bridge1.startScript('chrome-menu-flow-4', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME,
    nowFn: clock1.nowFn,
    setTimeoutFn: clock1.setTimeoutFn,
    clearTimeoutFn: clock1.clearTimeoutFn,
  });
  assert.equal(bridge1.notifyFirstPresent(), false);

  // Advance 90s (45x past old 2s fallback): MUST still be pending with 0 presses burned!
  clock1.advanceBy(90000);
  assert.equal(bridge1.getStats().scriptPending, true);
  assert.equal(bridge1.getStats().scriptActive, false);
  assert.equal(bridge1.getStats().scriptedPressesSent, 0);
  assert.equal(bridge1.getStats().swapchainProbeSupported, true);

  // Real FIRST_VISIBLE_FRAME arrives at +90s -> starts cleanly and lands crisp +500ms A press!
  assert.equal(bridge1.notifyFirstVisibleFrame(), true);
  assert.equal(bridge1.getStats().scriptPending, false);
  assert.equal(bridge1.getStats().scriptActive, true);
  assert.equal(bridge1.getStats().scriptStartedOn, 'FIRST_VISIBLE_FRAME');
  clock1.advanceBy(FIRST_VISIBLE_FRAME_INITIAL_A_DELAY_MS);
  assert.equal(bridge1.getSnapshot().buttons, INPUT_BUTTONS.A);
  assert.equal(bridge1.getStats().scriptedPressesSent, 1);
  bridge1.destroy();
  assert.equal(clock1.pendingTimeouts(), 0);

  // 2. Late setSwapchainProbeSupported(true) arriving after notifyFirstPresent() (before 2s)
  //    re-arms from the 2s fallback to the 180s safety fallback cleanly.
  const clock2 = createFakeClockAndTimers(0);
  const bridge2 = new InputBridge({ windowObj: null, documentObj: null, navigatorObj: null });
  bridge2.startScript('chrome-menu-flow-4', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME,
    nowFn: clock2.nowFn,
    setTimeoutFn: clock2.setTimeoutFn,
    clearTimeoutFn: clock2.clearTimeoutFn,
  });
  bridge2.notifyFirstPresent();
  clock2.advanceBy(500);
  bridge2.setSwapchainProbeSupported(true);
  clock2.advanceBy(80000);
  assert.equal(bridge2.getStats().scriptPending, true);
  assert.equal(bridge2.getStats().scriptedPressesSent, 0);
  // At 180s safety deadline (100s more), safety fallback fires so it can never hang forever.
  clock2.advanceBy(100000);
  assert.equal(bridge2.getStats().scriptActive, true);
  assert.equal(bridge2.getStats().scriptStartedOn, 'FIRST_PRESENT_FALLBACK');
  bridge2.destroy();
  assert.equal(clock2.pendingTimeouts(), 0);

  // 3. Explicit caller visibleFrameFallbackMs override is still honored even with swapchainProbeSupported=true,
  //    and swapchainProbeSupported=false keeps the 2000ms fallback unchanged.
  const clock3 = createFakeClockAndTimers(0);
  const bridge3 = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
    swapchainProbeSupported: true,
  });
  bridge3.startScript('chrome-menu-flow-4', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME,
    visibleFrameFallbackMs: 3500,
    nowFn: clock3.nowFn,
    setTimeoutFn: clock3.setTimeoutFn,
    clearTimeoutFn: clock3.clearTimeoutFn,
  });
  bridge3.notifyFirstPresent();
  clock3.advanceBy(3499);
  assert.equal(bridge3.getStats().scriptPending, true);
  clock3.advanceBy(1);
  assert.equal(bridge3.getStats().scriptStartedOn, 'FIRST_PRESENT_FALLBACK');
  bridge3.destroy();

  const clock4 = createFakeClockAndTimers(0);
  const bridge4 = new InputBridge({
    windowObj: null,
    documentObj: null,
    navigatorObj: null,
    swapchainProbeSupported: false,
  });
  bridge4.startScript('chrome-menu-flow-4', {
    startOn: INPUT_SCRIPT_START_TRIGGERS.FIRST_VISIBLE_FRAME,
    nowFn: clock4.nowFn,
    setTimeoutFn: clock4.setTimeoutFn,
    clearTimeoutFn: clock4.clearTimeoutFn,
  });
  bridge4.notifyFirstPresent();
  clock4.advanceBy(DEFAULT_VISIBLE_FRAME_FALLBACK_MS);
  assert.equal(bridge4.getStats().scriptStartedOn, 'FIRST_PRESENT_FALLBACK');
  bridge4.destroy();
});
