"use strict";
const fs = require("node:fs"),
  http = require("node:http"),
  os = require("node:os"),
  path = require("node:path"),
  { spawn } = require("node:child_process");
const [basePath, patchPath, chrome, output, samplesArg = "9", callsArg = "20"] =
  process.argv.slice(2);
if (!basePath || !patchPath || !chrome || !output)
  throw new Error(
    "Provide BASELINE_JS PATCH_JS CHROME OUTPUT [SAMPLES] [CALLS]",
  );
const samples = Number(samplesArg),
  calls = Number(callsArg);
if (![samples, calls].every((v) => Number.isInteger(v) && v > 0))
  throw new Error("sample and call counts must be positive integers");
const files = { base: path.resolve(basePath), patch: path.resolve(patchPath) };
const page = `<!doctype html><script src="/source.js"></script><script>
function make(n, target, last, variant) {let d=Array.from({length:n},(_,i)=>\`@group(0) @binding(\${i}) var<storage, read> cb_\${i}: array<f32>;\`).join('\\n');let refs=variant==='unused'?'':Array.from({length:n},(_,i)=>\`acc = acc + cb_\${i}[0];\`).join('\\n');let statement=variant==='prefix'?'acc = acc + cb_0_suffix;\\n':'acc = acc + 1.0;\\n';let fill=statement.repeat(Math.max(1,Math.floor((target-d.length-refs.length-100)/statement.length)));return \`\${d}\n@fragment fn fs_main() -> @location(0) vec4<f32> {\nvar acc = 0.0;\n\${last?fill+refs:refs+fill}\nreturn vec4<f32>(acc);\n}\`;}
function runCase(n, target, last, iterations, variant) {let source=make(n,target,last,variant),fn=PlayGta6Wgpu.compactWgslForMetadata, result;for(let i=0;i<5;i++) result=fn(source);let before=performance.now();for(let i=0;i<iterations;i++) result=fn(source);return {durationMs:performance.now()-before,sourceLength:source.length,stub:result};}
</script>`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  let server = http.createServer((req, res) => {
    let u = new URL(req.url, "http://localhost");
    res.setHeader("Cache-Control", "no-store");
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    if (u.pathname === "/source.js") {
      res.setHeader("Content-Type", "text/javascript");
      res.end(fs.readFileSync(files[u.searchParams.get("v")]));
    } else {
      res.setHeader("Content-Type", "text/html");
      res.end(
        page.replace("/source.js", "/source.js?v=" + u.searchParams.get("v")),
      );
    }
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  let dir = fs.mkdtempSync(path.join(os.tmpdir(), "pg6-jsbench-"));
  let browser = spawn(
    chrome,
    [
      "--headless=new",
      "--no-first-run",
      "--disable-extensions",
      "--disable-background-networking",
      "--disable-background-timer-throttling",
      "--disable-renderer-backgrounding",
      "--remote-debugging-port=0",
      "--user-data-dir=" + dir,
      "about:blank",
    ],
    { stdio: "ignore" },
  );
  let ws;
  try {
    let port;
    for (let i = 0; i < 100; i++) {
      if (browser.exitCode !== null) throw Error("browser exited");
      try {
        port = fs
          .readFileSync(path.join(dir, "DevToolsActivePort"), "utf8")
          .split("\n")[0];
      } catch {}
      if (port) break;
      await sleep(100);
    }
    if (!port) throw Error("no port");
    let ver = await (
      await fetch(`http://127.0.0.1:${port}/json/version`)
    ).json();
    ws = new WebSocket(ver.webSocketDebuggerUrl);
    await new Promise((r, j) => {
      ws.onopen = r;
      ws.onerror = j;
    });
    let id = 0,
      pending = new Map();
    ws.onmessage = (e) => {
      let m = JSON.parse(e.data),
        p = pending.get(m.id);
      if (p) {
        pending.delete(m.id);
        m.error ? p[1](Error(JSON.stringify(m.error))) : p[0](m.result);
      }
    };
    let send = (method, params = {}, sessionId) =>
      new Promise((resolve, reject) => {
        let key = ++id;
        pending.set(key, [resolve, reject]);
        ws.send(
          JSON.stringify({
            id: key,
            method,
            params,
            ...(sessionId ? { sessionId } : {}),
          }),
        );
      });
    let evaluate = async (s, expression) => {
      let r = await send(
        "Runtime.evaluate",
        { expression, awaitPromise: true, returnByValue: true },
        s,
      );
      if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
      return r.result.value;
    };
    let pages = {};
    for (let v of ["base", "patch"]) {
      let { targetId } = await send("Target.createTarget", {
        url: `http://127.0.0.1:${server.address().port}/?v=${v}`,
      });
      let { sessionId } = await send("Target.attachToTarget", {
        targetId,
        flatten: true,
      });
      await send("Runtime.enable", {}, sessionId);
      for (let i = 0; i < 100; i++) {
        if ((await evaluate(sessionId, "typeof runCase")) === "function") break;
        await sleep(100);
      }
      pages[v] = sessionId;
    }
    let result = {
      browser: ver.Browser,
      loadStart: os.loadavg(),
      sampling: `${samples} alternating variant pairs, ${calls} calls/sample, five warmups per call, Chrome V8 compact-only; no GPU`,
      cases: [],
    };
    for (let [n, size, last, variant] of [
      [0, 230000, true],
      [4, 230000, true],
      [16, 230000, true],
      [32, 230000, true],
      [16, 230000, false],
      [16, 5000, true],
      [16, 230000, true, "prefix"],
      [16, 230000, true, "unused"],
      [65, 230000, true],
    ]) {
      let records = [];
      for (let i = 0; i < samples; i++) {
        let record = {};
        for (let v of i % 2 ? ["patch", "base"] : ["base", "patch"]) {
          await send("Page.bringToFront", {}, pages[v]);
          record[v] = await evaluate(
            pages[v],
            `runCase(${n},${size},${last},${calls},${JSON.stringify(variant || "")})`,
          );
        }
        if (record.base.stub !== record.patch.stub)
          throw Error("Output mismatch");
        records.push({
          base: record.base.durationMs,
          patch: record.patch.durationMs,
          sourceLength: record.base.sourceLength,
          stubLength: record.base.stub.length,
        });
      }
      result.cases.push({
        n,
        size,
        last,
        variant: variant || "normal",
        samples: records,
      });
    }
    result.loadEnd = os.loadavg();
    fs.writeFileSync(output, JSON.stringify(result, null, 2));
    console.log(
      JSON.stringify({
        browser: result.browser,
        cases: result.cases.length,
        loadStart: result.loadStart,
        loadEnd: result.loadEnd,
      }),
    );
  } finally {
    if (ws) ws.close();
    browser.kill();
    await Promise.race([
      new Promise((r) => browser.once("exit", r)),
      sleep(3000),
    ]);
    server.close();
    fs.rmSync(dir, {
      recursive: true,
      force: true,
      maxRetries: 5,
      retryDelay: 100,
    });
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
