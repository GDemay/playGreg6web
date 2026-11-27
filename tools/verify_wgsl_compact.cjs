"use strict";
// Differential test of metadata strings from two worker implementations.
const path = require("node:path");
const [baseline, patch] = process.argv.slice(2);
if (!baseline || !patch) throw Error("expected BASELINE_JS PATCH_JS");
const base = require(path.resolve(baseline)).compactWgslForMetadata;
const next = require(path.resolve(patch)).compactWgslForMetadata;
let state = 1289;
function rand(n) {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
  return state % n;
}
const names = [
  "a",
  "aa",
  "aaa",
  "a0",
  "_a",
  "x",
  "xx",
  "xxx",
  "q0",
  "q00",
  "q0_",
  "x1",
  "x12",
  "x_",
  "_",
  "__",
  "ab",
  "abc",
  "ab1",
  "B1",
  "b1",
];
for (let t = 0; t < 5000; t++) {
  const count = rand(25),
    vars = [];
  for (let i = 0; i < count; i++) vars.push(names[rand(names.length)]);
  const declarations = vars
    .map((v, i) => `@group(0) @binding(${i}) var<uniform> ${v}: vec4<f32>;`)
    .join("\n");
  const parts = [];
  for (let j = 0; j < rand(200); j++) {
    const v = names[rand(names.length)];
    const boundaries = ["", " ", ".", "x", ";", "+", "(", "]", "_", "\n"];
    const pre = boundaries[rand(10)],
      post = boundaries[rand(10)],
      kind = rand(6);
    parts.push(
      kind === 0 ? `// ${v}\n` : kind === 1 ? `/* ${v} */` : pre + v + post,
    );
  }
  const source =
    declarations +
    "\n@fragment fn f() {\n" +
    " ".repeat(17000) +
    parts.join("") +
    "\n}";
  if (base(source) !== next(source))
    throw Error(`metadata mismatch at case ${t}`);
}
console.log(JSON.stringify({ passed: 5000, seed: 1289 }));
