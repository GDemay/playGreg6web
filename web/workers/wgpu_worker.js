'use strict';

// Dedicated OffscreenCanvas WebGPU worker consuming binary GpuPacketHeader streams.
// Opcode contract: docs/gpu_commands.md & runtime/abi/playgta6_abi.h.
const OP_NOP = 0;
const OP_BEGIN_FRAME = 1;
const OP_CLEAR_COLOR = 2;
const OP_PRESENT = 7;

let canvas = null;
let ctx = null;
let device = null;
let format = 'bgra8unorm';
let frameCount = 0;

async function initGpu(offscreen, width, height) {
  canvas = offscreen;
  canvas.width = width;
  canvas.height = height;

  if (!navigator.gpu) {
    self.postMessage({ error: 'navigator.gpu unavailable in worker' });
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    self.postMessage({ error: 'no WebGPU adapter available' });
    return;
  }

  device = await adapter.requestDevice();
  format = navigator.gpu.getPreferredCanvasFormat();
  ctx = canvas.getContext('webgpu');
  ctx.configure({ device, format, alphaMode: 'opaque' });

  const info = adapter.info ? `${adapter.info.vendor || 'gpu'} ${adapter.info.architecture || ''}`.trim() : 'WebGPU';
  self.postMessage({ ready: true, adapter: info });
}

function executeCommandBuffer(arrayBuffer) {
  if (!device || !ctx) return;
  const view = new DataView(arrayBuffer);
  let offset = 0;
  let clearColor = { r: 0.05, g: 0.07, b: 0.11, a: 1.0 };
  let shouldPresent = false;

  while (offset + 8 <= view.byteLength) {
    const opcode = view.getUint16(offset, true);
    const payloadBytes = view.getUint32(offset + 4, true);
    const payloadOffset = offset + 8;
    if (payloadOffset + payloadBytes > view.byteLength) break;

    if (opcode === OP_CLEAR_COLOR && payloadBytes >= 16) {
      clearColor = {
        r: view.getFloat32(payloadOffset, true),
        g: view.getFloat32(payloadOffset + 4, true),
        b: view.getFloat32(payloadOffset + 8, true),
        a: view.getFloat32(payloadOffset + 12, true),
      };
    } else if (opcode === OP_PRESENT) {
      shouldPresent = true;
    } else if (opcode === OP_NOP || opcode === OP_BEGIN_FRAME) {
      // Valid framing opcodes.
    }

    offset = payloadOffset + payloadBytes;
  }

  if (shouldPresent) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: ctx.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: clearColor,
        },
      ],
    });
    pass.end();
    device.queue.submit([encoder.finish()]);
    frameCount += 1;
    self.postMessage({ frame: frameCount });
  }
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  try {
    if (msg.init && msg.canvas) {
      await initGpu(msg.canvas, msg.width || 1280, msg.height || 720);
    } else if (msg.resize && canvas) {
      canvas.width = Math.max(1, msg.width | 0);
      canvas.height = Math.max(1, msg.height | 0);
    } else if (msg.cmdBuffer) {
      executeCommandBuffer(msg.cmdBuffer);
    }
  } catch (err) {
    self.postMessage({ error: String(err) });
  }
};
