import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  GpuBootPanelController,
  buildMissingShaderExport,
  sanitizeSwapchainColorMode,
  formatMemoryBytes,
} = require('../web/ui/gpu_boot_panel.js');

function createMockElement(tagName = 'div') {
  const attrs = new Map();
  const listeners = new Map();
  const children = [];
  const el = {
    tagName: String(tagName).toUpperCase(),
    className: '',
    textContent: '',
    hidden: false,
    children,
    parentNode: null,
    style: { display: '', width: '' },
    setAttribute(k, v) {
      attrs.set(k, String(v));
    },
    getAttribute(k) {
      return attrs.has(k) ? attrs.get(k) : null;
    },
    appendChild(child) {
      children.push(child);
      child.parentNode = el;
      return child;
    },
    removeChild(child) {
      const idx = children.indexOf(child);
      if (idx >= 0) {
        children.splice(idx, 1);
        child.parentNode = null;
      }
      return child;
    },
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      if (listeners.has(type)) listeners.get(type).delete(fn);
    },
    click() {
      const set = listeners.get('click');
      if (set) {
        for (const fn of Array.from(set)) fn();
      }
    },
    querySelector(selector) {
      const matchClass = selector.startsWith('.') ? selector.slice(1) : null;
      const stack = [...children];
      while (stack.length > 0) {
        const node = stack.shift();
        if (matchClass && String(node.className || '').split(/\s+/).includes(matchClass)) {
          return node;
        }
        if (Array.isArray(node.children)) {
          stack.push(...node.children);
        }
      }
      return null;
    },
  };
  return el;
}

function createMockDocument() {
  const head = createMockElement('head');
  const body = createMockElement('body');
  const docEl = createMockElement('html');
  docEl.appendChild(head);
  docEl.appendChild(body);
  return {
    head,
    body,
    documentElement: docEl,
    createElement(tag) {
      return createMockElement(tag);
    },
    querySelector(sel) {
      return docEl.querySelector(sel);
    },
  };
}

test('GpuBootPanel tracks #1077 GPU memory, Wasm linear memory, shader streaming, and swapchainColorMode across DOM, state, reset, and export', () => {
  assert.equal(sanitizeSwapchainColorMode('sdr-srgb'), 'sdr-srgb');
  assert.equal(
    sanitizeSwapchainColorMode(null, { toneMapActive: true, toneMapMode: 'aces' }),
    'hdr-tonemap-aces'
  );
  assert.equal(sanitizeSwapchainColorMode('../../../etc/passwd'), null);
  assert.equal(formatMemoryBytes(null), '—');
  assert.equal(formatMemoryBytes(0), '0 B');
  assert.equal(formatMemoryBytes(128 * 1024 * 1024), '128.0 MiB');
  assert.equal(formatMemoryBytes(2 * 1024 * 1024 * 1024), '2.00 GiB');

  const doc = createMockDocument();
  const panel = new GpuBootPanelController({ documentObj: doc });
  const root = panel.getState().root;

  panel.update({
    swapchainStats: {
      swapchainFormat: 'rgba16float',
      toneMapMode: 'aces',
      isHdrStream: true,
      toneMapActive: true,
      applySrgbEncode: true,
    },
    gpuMemory: {
      liveBufferBytes: 40 * 1024 * 1024,
      liveTextureBytes: 88 * 1024 * 1024,
      peakBufferBytes: 64 * 1024 * 1024,
      peakTextureBytes: 128 * 1024 * 1024,
      stagingPoolBytes: 8 * 1024 * 1024,
    },
    wasmLinearMemoryBytes: 512 * 1024 * 1024,
    shaderPackStats: {
      rangedShaderReady: 72,
      rangedShaderFetches: 19,
      rangedShaderErrors: 0,
      asyncShaderPendingCount: 2,
      rangedPackActive: true,
    },
  });

  const state = panel.getState();
  assert.equal(state.swapchainColorMode, 'hdr-tonemap-aces');
  assert.equal(state.liveBufferBytes, 40 * 1024 * 1024);
  assert.equal(state.liveTextureBytes, 88 * 1024 * 1024);
  assert.equal(state.totalLiveGpuBytes, 128 * 1024 * 1024);
  assert.equal(state.peakBufferBytes, 64 * 1024 * 1024);
  assert.equal(state.peakTextureBytes, 128 * 1024 * 1024);
  assert.equal(state.wasmLinearMemoryBytes, 512 * 1024 * 1024);
  assert.equal(state.rangedShaderReady, 72);
  assert.equal(state.rangedShaderFetches, 19);

  assert.equal(root.querySelector('.pg6-gpu-panel__swapchain-color-mode').textContent, 'hdr-tonemap-aces');
  assert.equal(root.querySelector('.pg6-gpu-panel__gpu-vram-live').textContent, '128.0 MiB');
  assert.equal(root.querySelector('.pg6-gpu-panel__gpu-vram-peak').textContent, '192.0 MiB');
  assert.equal(root.querySelector('.pg6-gpu-panel__wasm-linear-memory').textContent, '512.0 MiB');
  assert.equal(root.querySelector('.pg6-gpu-panel__shader-streaming').textContent, '72 ready · 19 fetches (2 pending)');

  const exported = JSON.parse(panel.exportMissingShadersJson());
  assert.equal(exported.gpu.swapchainColorMode, 'hdr-tonemap-aces');
  assert.equal(exported.gpu.toneMapMode, 'aces');
  assert.equal(exported.gpu.toneMapActive, true);
  assert.equal(exported.gpuMemory.totalLiveGpuBytes, 128 * 1024 * 1024);
  assert.equal(exported.gpuMemory.wasmLinearMemoryBytes, 512 * 1024 * 1024);
  assert.equal(exported.shaderStreaming.rangedShaderReady, 72);
  assert.equal(exported.shaderStreaming.rangedShaderFetches, 19);

  // Direct buildMissingShaderExport sanitizes hostile swapchainColorMode & BigInt bytes
  const direct = buildMissingShaderExport({
    swapchainColorMode: 'javascript:alert(1)',
    swapchainFormat: 'bgra8unorm-srgb',
    liveBufferBytes: 1024n,
    wasmLinearMemoryBytes: 2048n,
    rangedShaderReady: 5n,
  });
  assert.equal(direct.gpu.swapchainColorMode, 'sdr-srgb');
  assert.equal(direct.gpuMemory.liveBufferBytes, 1024);
  assert.equal(direct.gpuMemory.wasmLinearMemoryBytes, 2048);
  assert.equal(direct.shaderStreaming.rangedShaderReady, 5);

  // reset() clears all new telemetry cleanly
  panel.reset();
  assert.equal(panel.getState().swapchainColorMode, null);
  assert.equal(panel.getState().totalLiveGpuBytes, null);
  assert.equal(panel.getState().rangedShaderReady, null);
  assert.equal(root.querySelector('.pg6-gpu-panel__gpu-vram-live').textContent, '—');

  panel.destroy();
});

test('GpuBootPanel Export JSON button triggers browser download fallback when no onExportMissingShaders callback is configured', () => {
  const doc = createMockDocument();
  const clickedAnchors = [];
  const origCreate = doc.createElement.bind(doc);
  doc.createElement = (tag) => {
    const node = origCreate(tag);
    if (String(tag).toLowerCase() === 'a') {
      node.click = () => {
        clickedAnchors.push({ href: node.href, download: node.download });
      };
    }
    return node;
  };

  const origCreateObjectURL = URL.createObjectURL;
  const origRevokeObjectURL = URL.revokeObjectURL;
  const createdUrls = [];
  const revokedUrls = [];
  URL.createObjectURL = () => {
    const url = 'blob:mock-gpu-panel-export';
    createdUrls.push(url);
    return url;
  };
  URL.revokeObjectURL = (u) => {
    revokedUrls.push(u);
  };

  try {
    const panel = new GpuBootPanelController({ documentObj: doc });
    const exportBtn = panel.getState().root.querySelector('.pg6-gpu-panel__export-btn');
    exportBtn.click();

    assert.equal(clickedAnchors.length, 1);
    assert.equal(clickedAnchors[0].download, 'playgta6-missing-shaders.json');
    assert.equal(clickedAnchors[0].href, 'blob:mock-gpu-panel-export');
    assert.deepEqual(revokedUrls, ['blob:mock-gpu-panel-export']);
    assert.equal(panel.getState().exportCount, 1);

    // Programmatic exportMissingShadersJson() still does NOT trigger a browser download
    panel.exportMissingShadersJson();
    assert.equal(clickedAnchors.length, 1);

    panel.destroy();
  } finally {
    URL.createObjectURL = origCreateObjectURL;
    URL.revokeObjectURL = origRevokeObjectURL;
  }
});
