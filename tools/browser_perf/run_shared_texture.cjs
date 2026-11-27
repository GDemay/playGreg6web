'use strict';
// Node 24+, Chrome/Chrome Headless Shell with bundled SwiftShader, no npm packages.
// node tools/browser_perf/run_shared_texture.cjs BASELINE PATCH CHROME OUTPUT [ROUNDS SAMPLES FRAMES]
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const [baselineArg, patchArg, chrome, output, roundsArg = '7', samplesArg = '5', framesArg = '8'] = process.argv.slice(2);
if (!baselineArg || !patchArg || !chrome || !output) throw new Error('Provide BASELINE PATCH CHROME OUTPUT [ROUNDS SAMPLES FRAMES]');
const roots = { base: path.resolve(baselineArg), patch: path.resolve(patchArg) };
const rounds = Number(roundsArg), samples = Number(samplesArg), frames = Number(framesArg);
if (![rounds, samples, frames].every((n) => Number.isInteger(n) && n > 0)) throw new Error('Counts must be positive integers');
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
(async () => {
  const server = http.createServer((request, response) => {
    try {
      const parts = decodeURIComponent(new URL(request.url, 'http://localhost').pathname).split('/').filter(Boolean);
      const root = roots[parts.shift()];
      if (!root) throw new Error('Unknown variant');
      const relative = parts.join('/');
      const sourceRoot = relative.startsWith('tools/browser_perf/') ? roots.patch : root;
      const target = path.resolve(sourceRoot, relative);
      if (!target.startsWith(sourceRoot + path.sep)) throw new Error('Invalid path');
      response.writeHead(200, {
        'Content-Type': target.endsWith('.html') ? 'text/html' : 'text/javascript',
        'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cache-Control': 'no-store',
      });
      response.end(fs.readFileSync(target));
    } catch { response.writeHead(404); response.end('Not found'); }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'pg6-shared-texture-'));
  const flags = ['--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-extensions',
    '--disable-background-networking', '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows', '--remote-debugging-port=0', '--user-data-dir=' + profile,
    '--enable-unsafe-webgpu', '--no-sandbox', '--disable-dev-shm-usage', '--use-angle=swiftshader',
    '--use-vulkan=swiftshader', '--enable-unsafe-swiftshader', 'about:blank'];
  const browser = spawn(chrome, flags, { stdio: 'ignore' });
  let startupError, ws;
  browser.on('error', (error) => { startupError = error; });
  try {
    let port;
    for (let i = 0; i < 120; i++) {
      if (startupError) throw startupError;
      if (browser.exitCode !== null) throw new Error('Chrome exited before opening DevTools');
      try { port = fs.readFileSync(path.join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0]; } catch {}
      if (port) break;
      await sleep(250);
    }
    if (!port) throw new Error('Chrome startup timed out');
    const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
    ws = new WebSocket(version.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
    let nextId = 0;
    const pending = new Map();
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      const pair = pending.get(message.id);
      if (!pair) return;
      pending.delete(message.id);
      message.error ? pair[1](new Error(JSON.stringify(message.error))) : pair[0](message.result);
    };
    const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
      const id = ++nextId; pending.set(id, [resolve, reject]);
      ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
    const evaluate = async (sessionId, expression) => {
      const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, sessionId);
      if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    const pages = {};
    for (const variant of ['base', 'patch']) {
      const url = `http://127.0.0.1:${server.address().port}/${variant}/tools/browser_perf/texture_shared.html`;
      const { targetId } = await send('Target.createTarget', { url });
      const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
      await send('Runtime.enable', {}, sessionId); await send('Page.enable', {}, sessionId);
      for (let i = 0; i < 100; i++) {
        if (await evaluate(sessionId, 'typeof runSharedTextureBenchmark') === 'function') break;
        if (i === 99) throw new Error('Harness did not load');
        await sleep(100);
      }
      pages[variant] = sessionId;
    }
    const result = {
      benchmark: 'shared', environment: { browser: version.Browser, platform: process.platform, arch: process.arch,
        gpuLane: 'Linux Chrome SwiftShader SOFTWARE GPU; no hardware-GPU timing claim', hostLoadStart: os.loadavg(),
        scheduling: 'background throttling disabled; each target brought to front before its sample' },
      sampling: 'alternating baseline/patch page order; same software GPU/browser instance, identical harness source; browser encoding wall and queue-completion wall kept separate',
      rounds, samplesPerInvocation: samples, framesPerSample: frames, before: [], after: [],
    };
    for (let round = 0; round < rounds; round++) {
      for (const variant of round % 2 ? ['patch', 'base'] : ['base', 'patch']) {
        await send('Page.bringToFront', {}, pages[variant]); await sleep(250);
        const value = await evaluate(pages[variant], `runSharedTextureBenchmark(${JSON.stringify({ samples, warmups: 3, framesPerSample: frames })})`);
        if (value.adapterInfo.architecture !== 'swiftshader') throw new Error('Expected software SwiftShader adapter');
        result[variant === 'base' ? 'before' : 'after'].push(value);
        fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
      }
    }
    result.environment.hostLoadEnd = os.loadavg();
    fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
    await send('Browser.close');
  } finally {
    if (ws) ws.close(); browser.kill('SIGTERM'); server.close();
    fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
