'use strict';
// Load this identical harness against baseline and patch checkouts. The runner alternates
// variants externally. No game data, movie decoding, drawing, pacing or FPS is measured.
window.runSharedTextureBenchmark = async function ({ samples = 9, warmups = 3, framesPerSample = 4 } = {}) {
  const assert = (ok, message) => { if (!ok) throw new Error(message); };
  assert(crossOriginIsolated && typeof SharedArrayBuffer === 'function', 'COOP/COEP required');
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter, 'WebGPU adapter unavailable');
  const features = adapter.features.has('texture-compression-bc') ? ['texture-compression-bc'] : [];
  const real = await adapter.requestDevice({ requiredFeatures: features });
  const errors = [];
  real.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
  real.pushErrorScope('validation');
  let counts;
  const resetCounts = () => { counts = { writeTexture: 0, writeTextureBytes: 0, writeBuffer: 0, writeBufferBytes: 0, copyBufferToTexture: 0, submits: 0, createBuffer: 0 }; };
  resetCounts();
  const queue = new Proxy(real.queue, { get(target, key) {
    if (key === 'writeTexture') return (...args) => { counts.writeTexture++; counts.writeTextureBytes += args[1].byteLength; return target.writeTexture(...args); };
    if (key === 'writeBuffer') return (...args) => { counts.writeBuffer++; counts.writeBufferBytes += args[1] === undefined ? 0 : args[2].byteLength; return target.writeBuffer(...args); };
    if (key === 'submit') return (...args) => { counts.submits++; return target.submit(...args); };
    const value = Reflect.get(target, key, target); return typeof value === 'function' ? value.bind(target) : value;
  } });
  const device = new Proxy(real, { get(target, key) {
    if (key === 'queue') return queue;
    if (key === 'createBuffer') return (...args) => { counts.createBuffer++; return target.createBuffer(...args); };
    if (key === 'createCommandEncoder') return (...args) => {
      const encoder = target.createCommandEncoder(...args);
      return new Proxy(encoder, { get(enc, method) {
        if (method === 'copyBufferToTexture') return (...copyArgs) => { counts.copyBufferToTexture++; return enc.copyBufferToTexture(...copyArgs); };
        const value = Reflect.get(enc, method, enc); return typeof value === 'function' ? value.bind(enc) : value;
      } });
    };
    const value = Reflect.get(target, key, target); return typeof value === 'function' ? value.bind(target) : value;
  } });
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device: real, format });
  const staging = new PlayGta6StagingPool.GpuStagingPool({ device, chunkSize: 4 * 1024 * 1024, maxPoolBytes: 64 * 1024 * 1024 });
  const processor = new PlayGta6Wgpu.WgpuCommandProcessor(device, ctx, { stagingPool: staging, format, gpuStallWarnMs: 0 });
  const handler = PlayGta6TexturePackets.installTexturePackets(processor);
  function packet(opcode, payloadBytes, fill, shared = false) {
    const buffer = shared ? new SharedArrayBuffer(8 + payloadBytes) : new ArrayBuffer(8 + payloadBytes);
    const view = new DataView(buffer); view.setUint16(0, opcode, true); view.setUint32(4, payloadBytes, true); fill(view, new Uint8Array(buffer)); return new Uint8Array(buffer);
  }
  function create(id, width, height, { code = 1, layers = 1, dimension = 1, mips = 1 } = {}) {
    processor.executeCommandBuffer(packet(35, 40, (v) => {
      for (const [at, n] of [[0,id],[4,width],[8,height],[12,layers],[16,mips],[20,1],[24,7]]) v.setUint32(8 + at, n, true);
      v.setUint8(36, dimension); v.setUint8(37, code);
    }));
  }
  function write(id, width, height, data, { bytesPerRow = width, rowsPerImage = height, layers = 1, x = 0, y = 0, z = 0, shared = true } = {}) {
    return packet(13, 48 + data.length, (v, bytes) => {
      for (const [at,n] of [[0,id],[8,x],[12,y],[16,z],[20,width],[24,height],[28,layers],[32,bytesPerRow],[36,rowsPerImage],[40,data.length]]) v.setUint32(8 + at,n,true);
      bytes.set(data,56);
    }, shared);
  }
  async function read(id, width, height, layers = 1, blockBytes = 1, blockWidth = 1, blockHeight = 1) {
    processor._flushPendingUploads(true);
    const rowBytes = Math.ceil(width / blockWidth) * blockBytes;
    const rows = Math.ceil(height / blockHeight);
    const pitch = Math.ceil(rowBytes / 256) * 256;
    const buffer = real.createBuffer({ size: pitch * rows * layers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = real.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: processor.textures.get(id) }, { buffer, bytesPerRow: pitch, rowsPerImage: rows }, { width, height, depthOrArrayLayers: layers });
    real.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
    const source = new Uint8Array(buffer.getMappedRange()); const tight = new Uint8Array(rowBytes * rows * layers);
    for (let row = 0; row < rows * layers; row++) tight.set(source.subarray(row * pitch, row * pitch + rowBytes), row * rowBytes);
    buffer.unmap(); buffer.destroy(); return tight;
  }
  const equal = (a,b,label) => { assert(a.length === b.length, label + ' length'); for (let i=0;i<a.length;i++) if (a[i]!==b[i]) throw new Error(label + ' byte ' + i); };
  // Array/SAB/tight-last-row correctness, including a staged write followed by a direct
  // write to the same texture. Validation occurs outside all timing and operation counts.
  create(90, 17, 5, { layers: 2 });
  const first = new Uint8Array(17*5*2).fill(11);
  processor.executeCommandBuffer(write(90,17,5,first,{layers:2}));
  const expected = first.slice();
  const partial = new Uint8Array(3*2).fill(97);
  processor.executeCommandBuffer(write(90,3,2,partial,{x:4,y:1,z:1}));
  for (let row=1;row<3;row++) expected.fill(97,17*5+row*17+4,17*5+row*17+7);
  equal(await read(90,17,5,2),expected,'array partial order');
  // Explicitly stage an earlier write, then overwrite it directly (same texture).
  create(91,17,5);
  const zero = new Uint8Array(17*5).fill(3);
  const encoder = processor._ensurePendingEncoder();
  staging.stageTextureUpload(encoder,processor.textures.get(91),zero,{width:17,height:5,bytesPerRow:17,rowsPerImage:5});
  processor._hasPendingUploadCommands = true; handler._pendingStagedTextures.add(processor.textures.get(91));
  const later = new Uint8Array(17*5).fill(83);
  processor.executeCommandBuffer(write(91,17,5,later));
  equal(await read(91,17,5),later,'staged then direct');
  const correctness = ['SAB 2D-array full+partial write exact readback','staged then direct ordering exact readback'];
  if (features.length) {
    create(92,8,8,{code:62,layers:2});
    const bc = new Uint8Array(16*2*2*2); for(let i=0;i<bc.length;i++) bc[i]=(i*19+5)&255;
    processor.executeCommandBuffer(write(92,8,8,bc,{bytesPerRow:32,rowsPerImage:2,layers:2}));
    equal(await read(92,8,8,2,16,4,4),bc,'BC7 array'); correctness.push('BC7 array encoded blocks exact readback');
  }
  create(93,256,2);
  const original = new Uint8Array(512).fill(39);
  const mutablePacket = write(93,256,2,original);
  processor.executeCommandBuffer(mutablePacket);
  new Uint8Array(mutablePacket.buffer,mutablePacket.byteOffset+56,512).fill(107);
  equal(await read(93,256,2),original,'SAB content timeline snapshot before source mutation');
  correctness.push('source SAB overwritten immediately after actual writeTexture call; original GPU pixels retained');
  const planes = [[1,3840,2160]];
  const packets = planes.map(([id,width,height]) => {
    create(id,width,height);
    const data = new Uint8Array(width*height); for(let i=0;i<data.length;i++) data[i]=(i*37+id*17)&255;
    return { id,width,height,data,packet:write(id,width,height,data),controlPacket:write(id,width,height,data,{shared:false}) };
  });
  await real.queue.onSubmittedWorkDone();
  let frame = 0;
  async function sample(uploads = packets) {
    resetCounts(); const sabCopiesBefore=handler.getStats().sabCopyCount; const encodeStart = performance.now();
    for(let i=0;i<framesPerSample;i++,frame++) {
      staging.beginFrame(frame);
      for(const p of uploads) processor.executeCommandBuffer(p.packet);
      processor._flushPendingUploads(true);
    }
    const encodeWallMs = performance.now()-encodeStart;
    const waitStart = performance.now(); await real.queue.onSubmittedWorkDone();
    return { encodeWallMs, completionWaitWallMs:performance.now()-waitStart, frames:framesPerSample, appSabSnapshots:handler.getStats().sabCopyCount-sabCopiesBefore, queueCounts:{...counts} };
  }
  for(let i=0;i<warmups;i++) await sample();
  const rawSamples = []; for(let i=0;i<samples;i++) rawSamples.push(await sample());
  // Nonshared aligned Y uploads are unchanged; shared Y is the changed workload.
  const controlUploads = [{packet:packets[0].controlPacket}];
  for(let i=0;i<warmups;i++) await sample(controlUploads);
  const controlSamples = []; for(let i=0;i<samples;i++) controlSamples.push(await sample(controlUploads));
  for(const p of packets) equal(await read(p.id,p.width,p.height),p.data,'video plane '+p.id);
  correctness.push('all 8,294,400 Y-plane bytes exact final readback');
  const scopedError = await real.popErrorScope(); if(scopedError) errors.push(scopedError.message);
  assert(errors.length===0,errors.join('\n'));
  const result = { benchmark:'actual production nonchunked shared-source WriteTexture on real browser WebGPU device; synthetic 3840x2160 R8 Y plane', userAgent:navigator.userAgent, adapterInfo:{ vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description }, samples,warmups,framesPerSample,rawSamples,controlWorkload:'nonshared aligned 3840x2160 Y upload (unchanged path)',controlSamples,correctness,validationErrors:errors,unmeasured:'game/movie decoding, actual runtime producer, drawing, gameplay, FPS; completion wait is wall-clock queue fence latency, not GPU timestamp execution time' };
  processor.destroy(); real.destroy(); document.getElementById('result').textContent=JSON.stringify(result,null,2); return result;
};
