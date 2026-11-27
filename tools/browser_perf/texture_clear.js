'use strict';
// Consumer-contract benchmark: the C++ boot test separately proves activation.
// Before: small pooled textures receive CPU zero WriteTextures, large textures are recreated.
// After: both use the existing ClearTexture consumer packet. Does not time C++ packet generation.
window.runTextureClearBenchmark = async function ({ mode = 'baseline', samples = 9, warmups = 3, operationsPerSample = 4, sizes = [128, 1024] } = {}) {
  const check = (ok,message) => { if(!ok) throw new Error(message); };
  check(mode==='baseline'||mode==='gpu-clear'||mode==='consume-only','mode must be baseline, gpu-clear or consume-only');
  check(crossOriginIsolated && typeof SharedArrayBuffer==='function','COOP/COEP required for Wasm-like SAB packets');
  const adapter = await navigator.gpu.requestAdapter(); check(adapter,'WebGPU unavailable');
  const requiredFeatures = adapter.features.has('texture-compression-bc') ? ['texture-compression-bc'] : [];
  const real = await adapter.requestDevice({requiredFeatures});
  const errors=[]; real.addEventListener('uncapturederror',e=>errors.push(e.error.message)); real.pushErrorScope('validation');
  let counts; const reset=()=>{counts={createTexture:0,writeTexture:0,writeTextureBytes:0,copyBufferToTexture:0,submit:0,consumerCopies:0,consumerSubmits:0,consumerBytes:0,createBuffer:0};}; reset();
  const queue=new Proxy(real.queue,{get(target,key){
    if(key==='writeTexture') return(...args)=>{counts.writeTexture++;counts.writeTextureBytes+=args[1].byteLength;return target.writeTexture(...args);};
    if(key==='submit') return(...args)=>{counts.submit++;return target.submit(...args);};
    const v=Reflect.get(target,key,target);return typeof v==='function'?v.bind(target):v;
  }});
  const device=new Proxy(real,{get(target,key){
    if(key==='queue')return queue;
    if(key==='createTexture'||key==='createBuffer')return(...args)=>{counts[key]++;return target[key](...args);};
    if(key==='createCommandEncoder')return(...args)=>{
      const encoder=target.createCommandEncoder(...args);return new Proxy(encoder,{get(enc,k){
        if(k==='copyBufferToTexture')return(...a)=>{counts.copyBufferToTexture++;return enc.copyBufferToTexture(...a);};
        const v=Reflect.get(enc,k,enc);return typeof v==='function'?v.bind(enc):v;
      }});
    };
    const v=Reflect.get(target,key,target);return typeof v==='function'?v.bind(target):v;
  }});
  const ctx=document.getElementById('canvas').getContext('webgpu');const format=navigator.gpu.getPreferredCanvasFormat();ctx.configure({device:real,format});
  const processor=new PlayGta6Wgpu.WgpuCommandProcessor(device,ctx,{format,gpuStallWarnMs:0});
  PlayGta6TexturePackets.installTexturePackets(processor);
  function packet(opcode,size,fill){const b=new Uint8Array(new SharedArrayBuffer(8+size));const v=new DataView(b.buffer);v.setUint16(0,opcode,true);v.setUint32(4,size,true);fill(v,b);return b;}
  function create(id,w,h,{d=1,mips=1,dimension=1,code=22}={}){
    processor.executeCommandBuffer(packet(35,40,v=>{for(const[o,n]of[[0,id],[4,w],[8,h],[12,d],[16,mips],[20,1],[24,7]])v.setUint32(8+o,n,true);v.setUint8(36,dimension);v.setUint8(37,code);}));
    return processor.textures.get(id);
  }
  const clear=id=>processor.executeCommandBuffer(packet(63,8,v=>v.setUint32(8,id,true)));
  function zeroPacket(id,w,h){return packet(13,48+w*h*4,(v)=>{for(const[o,n]of[[0,id],[20,w],[24,h],[28,1],[32,w*4],[36,h],[40,w*h*4]])v.setUint32(8+o,n,true);});}
  async function read(texture,mip,w,h,d,block=1,bpp=4){
    const rows=Math.ceil(h/block);const rowBytes=Math.ceil(w/block)*bpp;const pitch=Math.ceil(rowBytes/256)*256;
    const buffer=real.createBuffer({size:pitch*rows*d,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const enc=real.createCommandEncoder();
    enc.copyTextureToBuffer({texture,mipLevel:mip},{buffer,bytesPerRow:pitch,rowsPerImage:rows},{width:w,height:h,depthOrArrayLayers:d});real.queue.submit([enc.finish()]);await buffer.mapAsync(GPUMapMode.READ);
    const src=new Uint8Array(buffer.getMappedRange());const out=new Uint8Array(rowBytes*rows*d);for(let r=0;r<rows*d;r++)out.set(src.subarray(r*pitch,r*pitch+rowBytes),r*rowBytes);
    buffer.unmap();buffer.destroy();return out;
  }
  const correctness=[];
  const shapes=[{name:'RGBA8 array/mips',w:32,h:16,d:6,mips:3},{name:'RGBA8 3D/mips',w:32,h:16,d:4,mips:3,dimension:2},{name:'R8 signed nonrenderable',w:17,h:5,d:1,mips:1,code:2,bpp:1}];
  if(requiredFeatures.length)shapes.push({name:'BC7 array/mip tails',w:16,h:8,d:2,mips:4,code:62,block:4,bpp:16});
  let id=100;
  for(const shape of shapes){
    const {w,h,d,mips}=shape;const texture=create(id,w,h,shape);const block=shape.block||1,bpp=shape.bpp||4;
    for(let m=0;m<mips;m++){
      const mw=Math.max(block,w>>m),mh=Math.max(block,h>>m),md=shape.dimension===2?Math.max(1,d>>m):d;
      const rows=Math.ceil(mh/block),pitch=Math.ceil(mw/block)*bpp;const dirty=new Uint8Array(pitch*rows*md).fill(0x55);
      real.queue.writeTexture({texture,mipLevel:m},dirty,{bytesPerRow:pitch,rowsPerImage:rows},{width:mw,height:mh,depthOrArrayLayers:md});
    }
    clear(id);
    for(let m=0;m<mips;m++){
      const mw=Math.max(block,w>>m),mh=Math.max(block,h>>m),md=shape.dimension===2?Math.max(1,d>>m):d;
      const got=await read(texture,m,mw,mh,md,block,bpp);for(let i=0;i<got.length;i++)if(got[i]!==0)throw new Error(shape.name+' stale byte '+i);
    }
    correctness.push(shape.name+' every stored block/texel zero');id++;
  }
  // Earlier uploads are wiped and later partial writes survive the clear (queue order).
  const orderTexture=create(110,16,16);real.queue.writeTexture({texture:orderTexture},new Uint8Array(16*16*4).fill(11),{bytesPerRow:64},{width:16,height:16});
  clear(110);real.queue.writeTexture({texture:orderTexture,origin:{x:3,y:4}},new Uint8Array(2*3*4).fill(79),{bytesPerRow:8},{width:2,height:3});
  const got=await read(orderTexture,0,16,16,1);for(let y=0;y<16;y++)for(let x=0;x<16;x++)for(let c=0;c<4;c++){
    const expected=x>=3&&x<5&&y>=4&&y<7?79:0;if(got[(y*16+x)*4+c]!==expected)throw new Error('clear/partial ordering');
  }
  correctness.push('write -> ClearTexture -> partial write exact readback');
  const results=[];
  for(const size of sizes){
    check(size===128||size===1024,'size must be 128 or 1024');
    const textureId=size===128?1:2;create(textureId,size,size);const zeros=zeroPacket(textureId,size,size);
    // Force every created/cleared texture to be observed by identical GPU work.
    // Otherwise fresh baseline textures could remain lazily uninitialized forever.
    const sink=real.createBuffer({size:size*size*4,usage:GPUBufferUsage.COPY_DST});
    // Initial resources / pooled reusable zero buffer are primed outside samples.
    clear(textureId);await real.queue.onSubmittedWorkDone();
    async function sample(){
      reset();const t=performance.now();
      for(let i=0;i<operationsPerSample;i++){
        if(mode==='consume-only') { /* unchanged texture, identical GPU sink consumption */ }
        else if(mode==='gpu-clear')clear(textureId);
        else if(size===128)processor.executeCommandBuffer(zeros);
        else{processor.textures.get(textureId).destroy();create(textureId,size,size);}
        const consumer=real.createCommandEncoder();
        consumer.copyTextureToBuffer({texture:processor.textures.get(textureId)},
          {buffer:sink,bytesPerRow:size*4,rowsPerImage:size},
          {width:size,height:size,depthOrArrayLayers:1});
        real.queue.submit([consumer.finish()]);
        counts.consumerCopies++;counts.consumerSubmits++;counts.consumerBytes+=size*size*4;
      }
      const encodeWallMs=performance.now()-t;const startWait=performance.now();await real.queue.onSubmittedWorkDone();
      return{encodeWallMs,completionWaitWallMs:performance.now()-startWait,operations:operationsPerSample,queueCounts:{...counts}};
    }
    for(let i=0;i<warmups;i++)await sample();const rawSamples=[];for(let i=0;i<samples;i++)rawSamples.push(await sample());
    const pixels=await read(processor.textures.get(textureId),0,size,size,1);for(let i=0;i<pixels.length;i++)if(pixels[i]!==0)throw new Error('benchmark final nonzero pixel');
    sink.destroy();
    results.push({width:size,height:size,bytes:size*size*4,consumer:'each operation copies the complete texture to one preallocated GPU sink, same commands in both modes',baselineContract:size===128?'reused <=64KiB texture receives CPU zero WriteTexture':'texture >64KiB recreated (large pool disabled)',rawSamples});
  }
  const scopedError=await real.popErrorScope();if(scopedError)errors.push(scopedError.message);check(errors.length===0,errors.join('\n'));
  const result={benchmark:'existing worker ClearTexture consumer contract on actual browser GPU; synthetic sampled-only COPY_DST textures',mode,userAgent:navigator.userAgent,adapterInfo:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},samples,warmups,operationsPerSample,results,correctness,validationErrors:errors,unmeasured:'actual Wasm producer, C++ allocation/copy/packet generation, real pooling hit distributions, gameplay or FPS; fence wait wall is not GPU timestamp execution time'};
  processor.destroy();real.destroy();document.getElementById('result').textContent=JSON.stringify(result,null,2);return result;
};
