import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { compactWgslForMetadata } = require('../web/workers/wgpu_worker.js');
const { parseWgslGroup0Bindings, findWgslStageEntryPoints } = require('../web/workers/render_state_packets.js');

function sourceFor(names, body) {
  const declarations = names.map((name, i) =>
    `@group(0) @binding(${i}) var<uniform> ${name}: vec4<f32>;`).join('\n');
  return `${declarations}\n@fragment fn fs_main() {${body}}`;
}

function bindingsInStub(source) {
  return [...compactWgslForMetadata(source).matchAll(/@binding\((\d+)\)/g)].map((match) => Number(match[1]));
}

test('WGSL compact metadata preserves identifier boundaries, comments, duplicate names, and stage metadata', () => {
  const names = ['a', 'aa', 'a0', '_a', 'b', 'c', 'unused', 'a'];
  const source = sourceFor(names, `${' '.repeat(17000)} a+aa; a0; _a; object.b; abc; unused2; /* c */ // unused\n`);
  assert.deepEqual(bindingsInStub(source), [0, 1, 2, 3, 7]);
  const stub = compactWgslForMetadata(source);
  assert.deepEqual(parseWgslGroup0Bindings(stub), parseWgslGroup0Bindings(source));
  assert.deepEqual(findWgslStageEntryPoints(stub, 'fragment'), ['fs_main']);
});

test('WGSL compact metadata handles references at the 4096-character prefix boundary', () => {
  const names = ['u0', 'u1', 'u2', 'u3', 'u4'];
  // `u0` ends exactly at the prefix boundary, but continues as an identifier.
  const falseHit = sourceFor(names, `${' '.repeat(4094)}u0suffix; u1; u2; u3; u4;${' '.repeat(17000)}`);
  assert.deepEqual(bindingsInStub(falseHit), [1, 2, 3, 4]);
  // A token at the same boundary followed by punctuation is a real reference.
  const trueHit = sourceFor(names, `${' '.repeat(4094)}u0; u1; u2; u3; u4;${' '.repeat(17000)}`);
  assert.deepEqual(bindingsInStub(trueHit), [0, 1, 2, 3, 4]);
});

test('WGSL compact metadata handles empty, early, late, and over-limit binding sets', () => {
  assert.deepEqual(bindingsInStub('@fragment fn fs_main() {}'), []);
  const names = Array.from({ length: 65 }, (_, i) => `item_${i}`);
  const early = sourceFor(names, `${names.map((name) => ` ${name};`).join('')}${' '.repeat(17000)}`);
  const late = sourceFor(names, `${' '.repeat(17000)}${names.map((name) => ` ${name};`).join('')}`);
  assert.deepEqual(bindingsInStub(early), names.map((_, i) => i));
  assert.deepEqual(bindingsInStub(late), names.map((_, i) => i));
  const longNames = Array.from({ length: 5 }, (_, i) => `v${i}_${'x'.repeat(900)}`);
  const longSource = sourceFor(longNames, `${' '.repeat(17000)} ${longNames.join('; ')};`);
  assert.deepEqual(bindingsInStub(longSource), [0, 1, 2, 3, 4]);
});
