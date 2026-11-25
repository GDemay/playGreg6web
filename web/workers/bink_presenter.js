'use strict';

// WebGPU Bink2 / splash video frame presenter.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6BinkPresenter)
// and Node.js (module.exports). Renders decoded RGBA8, YUV420P, YUVA420P, and NV12 frames onto
// a WebGPU swapchain target with aspect-ratio-preserving letterboxing, reusable plane textures,
// 256-byte-aligned uploads (direct with reusable scratch buffer or via GpuStagingPool),
// selectable BT.709/BT.601 limited/full-range conversion, and zero per-frame pipeline or
// bind-group churn.

(function (root) {
  const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256;
  const UNIFORM_BUFFER_SIZE = 96; // 24 x 4-byte lanes, 16-byte aligned WGSL struct
  const DEFAULT_PRESENTATION_FORMAT = 'bgra8unorm';
  const DEFAULT_TARGET_FPS = 30;
  const DOUBLE_BUFFER_SLOTS = 2;
  const UNIFORM_FLAG_FULL_RANGE = 1;

  const BINK_PIXEL_FORMAT = Object.freeze({
    RGBA8: 'rgba8',
    YUV420P: 'yuv420p',
    YUVA420P: 'yuva420p',
    NV12: 'nv12',
  });

  const YUV_COLOR_SPACE = Object.freeze({
    BT709: 'bt709',
    BT601: 'bt601',
  });

  const YUV_COLOR_RANGE = Object.freeze({
    LIMITED: 'limited',
    FULL: 'full',
  });

  const PIXEL_FORMAT_MODE = Object.freeze({
    [BINK_PIXEL_FORMAT.RGBA8]: 0,
    [BINK_PIXEL_FORMAT.YUV420P]: 1,
    [BINK_PIXEL_FORMAT.YUVA420P]: 2,
    [BINK_PIXEL_FORMAT.NV12]: 3,
  });

  const COLOR_SPACE_ID = Object.freeze({
    [YUV_COLOR_SPACE.BT709]: 0,
    [YUV_COLOR_SPACE.BT601]: 1,
  });

  // Limited-range ITU-R YCbCr -> RGB conversion matrices (Y in [16..235], U/V in [16..240]).
  const YUV_LIMITED_OFFSET = Object.freeze([16 / 255, 128 / 255, 128 / 255, 0]);
  // Full-range ITU-R YCbCr -> RGB conversion offset (Y in [0..255], U/V in [0..255]).
  const YUV_FULL_OFFSET = Object.freeze([0, 128 / 255, 128 / 255, 0]);

  const YUV_MATRICES = Object.freeze({
    [YUV_COLOR_SPACE.BT709]: Object.freeze({
      row0: Object.freeze([1.16438356, 0.0, 1.79274107, 0.0]),
      row1: Object.freeze([1.16438356, -0.21324861, -0.53290933, 0.0]),
      row2: Object.freeze([1.16438356, 2.11240179, 0.0, 0.0]),
      offset: YUV_LIMITED_OFFSET,
    }),
    [YUV_COLOR_SPACE.BT601]: Object.freeze({
      row0: Object.freeze([1.16438356, 0.0, 1.59602679, 0.0]),
      row1: Object.freeze([1.16438356, -0.39176229, -0.81296765, 0.0]),
      row2: Object.freeze([1.16438356, 2.01723214, 0.0, 0.0]),
      offset: YUV_LIMITED_OFFSET,
    }),
  });

  const YUV_FULL_RANGE_MATRICES = Object.freeze({
    [YUV_COLOR_SPACE.BT709]: Object.freeze({
      row0: Object.freeze([1.0, 0.0, 1.5748, 0.0]),
      row1: Object.freeze([1.0, -0.18732427, -0.46812427, 0.0]),
      row2: Object.freeze([1.0, 1.8556, 0.0, 0.0]),
      offset: YUV_FULL_OFFSET,
    }),
    [YUV_COLOR_SPACE.BT601]: Object.freeze({
      row0: Object.freeze([1.0, 0.0, 1.402, 0.0]),
      row1: Object.freeze([1.0, -0.34413629, -0.71413629, 0.0]),
      row2: Object.freeze([1.0, 1.772, 0.0, 0.0]),
      offset: YUV_FULL_OFFSET,
    }),
  });

  const YUV_RANGE_MATRICES = Object.freeze({
    [YUV_COLOR_RANGE.LIMITED]: YUV_MATRICES,
    [YUV_COLOR_RANGE.FULL]: YUV_FULL_RANGE_MATRICES,
  });

  // WebGPU bitmask constants with headless/Node fallbacks.
  const TEXTURE_USAGE = Object.freeze({
    COPY_DST:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_DST === 'number'
        ? GPUTextureUsage.COPY_DST
        : 0x02,
    TEXTURE_BINDING:
      typeof GPUTextureUsage !== 'undefined' &&
      typeof GPUTextureUsage.TEXTURE_BINDING === 'number'
        ? GPUTextureUsage.TEXTURE_BINDING
        : 0x04,
  });

  const BUFFER_USAGE = Object.freeze({
    COPY_DST:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : 0x0008,
    UNIFORM:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.UNIFORM === 'number'
        ? GPUBufferUsage.UNIFORM
        : 0x0040,
  });

  const SHADER_STAGE = Object.freeze({
    VERTEX:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.VERTEX === 'number'
        ? GPUShaderStage.VERTEX
        : 0x1,
    FRAGMENT:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.FRAGMENT === 'number'
        ? GPUShaderStage.FRAGMENT
        : 0x2,
  });

  const DEFAULT_CLEAR_COLOR = Object.freeze({ r: 0, g: 0, b: 0, a: 1 });
  const DEFAULT_UV_TRANSFORM = Object.freeze([1, 1, 0, 0]);

  const BINK_PRESENTER_WGSL = /* wgsl */ `
struct PresenterUniforms {
  mode: u32,
  color_space: u32,
  opacity: f32,
  flags: u32,
  uv_transform: vec4<f32>,
  yuv_row0: vec4<f32>,
  yuv_row1: vec4<f32>,
  yuv_row2: vec4<f32>,
  yuv_offset: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: PresenterUniforms;
@group(0) @binding(1) var video_sampler: sampler;
@group(0) @binding(2) var plane_primary: texture_2d<f32>;
@group(0) @binding(3) var plane_u: texture_2d<f32>;
@group(0) @binding(4) var plane_v: texture_2d<f32>;
@group(0) @binding(5) var plane_a: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  var uvs = array<vec2<f32>, 3>(
    vec2<f32>(0.0,  1.0),
    vec2<f32>(2.0,  1.0),
    vec2<f32>(0.0, -1.0)
  );
  var out: VertexOutput;
  out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
  out.uv = uvs[vertex_index];
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let uv = in.uv * uniforms.uv_transform.xy + uniforms.uv_transform.zw;
  let clamped_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));

  let primary = textureSample(plane_primary, video_sampler, clamped_uv);
  let uv_sample = textureSample(plane_u, video_sampler, clamped_uv);
  let v_sample = textureSample(plane_v, video_sampler, clamped_uv).r;
  let a_sample = textureSample(plane_a, video_sampler, clamped_uv).r;

  let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
  if (!in_bounds) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  if (uniforms.mode == 0u) {
    return vec4<f32>(primary.rgb, primary.a * uniforms.opacity);
  }

  let chroma_u = uv_sample.r;
  let chroma_v = select(v_sample, uv_sample.g, uniforms.mode == 3u);
  let yuv = vec3<f32>(primary.r, chroma_u, chroma_v) - uniforms.yuv_offset.xyz;
  let rgb = clamp(
    vec3<f32>(
      dot(uniforms.yuv_row0.xyz, yuv),
      dot(uniforms.yuv_row1.xyz, yuv),
      dot(uniforms.yuv_row2.xyz, yuv)
    ),
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );
  let alpha = select(1.0, a_sample, uniforms.mode == 2u) * uniforms.opacity;
  return vec4<f32>(rgb, alpha);
}
`.trim();

  function assertPositiveInteger(value, name) {
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive integer, got ${value}`);
    }
  }

  function assertPositiveFinite(value, name) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      throw new RangeError(`${name} must be a positive finite number, got ${value}`);
    }
  }

  function assertNonNegativeInteger(value, name) {
    if (!Number.isInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative integer, got ${value}`);
    }
  }

  function validateColorSpace(colorSpace) {
    if (colorSpace !== YUV_COLOR_SPACE.BT709 && colorSpace !== YUV_COLOR_SPACE.BT601) {
      throw new RangeError(
        `Unsupported colorSpace "${colorSpace}"; expected "${YUV_COLOR_SPACE.BT709}" or "${YUV_COLOR_SPACE.BT601}"`
      );
    }
  }

  function validateColorRange(colorRange) {
    if (colorRange !== YUV_COLOR_RANGE.LIMITED && colorRange !== YUV_COLOR_RANGE.FULL) {
      throw new RangeError(
        `Unsupported colorRange "${colorRange}"; expected "${YUV_COLOR_RANGE.LIMITED}" or "${YUV_COLOR_RANGE.FULL}"`
      );
    }
  }

  function validatePixelFormat(format) {
    if (
      format !== BINK_PIXEL_FORMAT.RGBA8 &&
      format !== BINK_PIXEL_FORMAT.YUV420P &&
      format !== BINK_PIXEL_FORMAT.YUVA420P &&
      format !== BINK_PIXEL_FORMAT.NV12
    ) {
      throw new RangeError(`Unsupported pixel format "${format}"`);
    }
  }

  function getYuvMatrix(
    colorSpace = YUV_COLOR_SPACE.BT709,
    colorRange = YUV_COLOR_RANGE.LIMITED
  ) {
    validateColorSpace(colorSpace);
    validateColorRange(colorRange);
    return YUV_RANGE_MATRICES[colorRange][colorSpace];
  }

  function alignUp(value, alignment) {
    assertNonNegativeInteger(value, 'value');
    assertPositiveInteger(alignment, 'alignment');
    const remainder = value % alignment;
    return remainder === 0 ? value : value + (alignment - remainder);
  }

  function toUint8Array(input, label) {
    if (input instanceof Uint8Array) {
      return input;
    }
    if (ArrayBuffer.isView(input)) {
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    if (input instanceof ArrayBuffer) {
      return new Uint8Array(input);
    }
    if (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer) {
      return new Uint8Array(input);
    }
    throw new TypeError(
      `${label} expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer`
    );
  }

  function normalizeClearColor(clearColor) {
    if (!clearColor) {
      return { ...DEFAULT_CLEAR_COLOR };
    }
    if (Array.isArray(clearColor) && clearColor.length >= 4) {
      return {
        r: clearColor[0],
        g: clearColor[1],
        b: clearColor[2],
        a: clearColor[3],
      };
    }
    return {
      r: typeof clearColor.r === 'number' ? clearColor.r : DEFAULT_CLEAR_COLOR.r,
      g: typeof clearColor.g === 'number' ? clearColor.g : DEFAULT_CLEAR_COLOR.g,
      b: typeof clearColor.b === 'number' ? clearColor.b : DEFAULT_CLEAR_COLOR.b,
      a: typeof clearColor.a === 'number' ? clearColor.a : DEFAULT_CLEAR_COLOR.a,
    };
  }

  /**
   * Compute a centered letterbox/pillarbox viewport inside `(dstWidth, dstHeight)` that
   * preserves the exact aspect ratio of `(srcWidth, srcHeight)`.
   *
   * @param {number} srcWidth
   * @param {number} srcHeight
   * @param {number} dstWidth
   * @param {number} dstHeight
   * @returns {{ x: number, y: number, width: number, height: number, scaleX: number, scaleY: number }}
   */
  function computeLetterboxViewport(srcWidth, srcHeight, dstWidth, dstHeight) {
    assertPositiveFinite(srcWidth, 'srcWidth');
    assertPositiveFinite(srcHeight, 'srcHeight');
    assertPositiveFinite(dstWidth, 'dstWidth');
    assertPositiveFinite(dstHeight, 'dstHeight');

    const srcAspect = srcWidth / srcHeight;
    const dstAspect = dstWidth / dstHeight;

    let x = 0;
    let y = 0;
    let width = dstWidth;
    let height = dstHeight;

    if (Math.abs(srcAspect - dstAspect) > 1e-9) {
      if (srcAspect > dstAspect) {
        // Source is wider than destination: fit width, letterbox top & bottom.
        width = dstWidth;
        height = Math.min(dstHeight, dstWidth / srcAspect);
        x = 0;
        y = (dstHeight - height) / 2;
      } else {
        // Source is taller/narrower than destination: fit height, pillarbox left & right.
        height = dstHeight;
        width = Math.min(dstWidth, dstHeight * srcAspect);
        x = (dstWidth - width) / 2;
        y = 0;
      }
    }

    return {
      x,
      y,
      width,
      height,
      scaleX: width / dstWidth,
      scaleY: height / dstHeight,
    };
  }

  /**
   * Pack the 96-byte `PresenterUniforms` struct into a `Uint8Array` matching the exact
   * WGSL layout in `BINK_PRESENTER_WGSL`.
   *
   * @param {{
   *   pixelFormat?: string,
   *   colorSpace?: string,
   *   colorRange?: string,
   *   opacity?: number,
   *   flags?: number,
   *   uvTransform?: number[],
   * }} [params]
   * @returns {Uint8Array}
   */
  function packPresenterUniforms(params = {}) {
    const pixelFormat = params.pixelFormat ?? BINK_PIXEL_FORMAT.RGBA8;
    const colorSpace = params.colorSpace ?? YUV_COLOR_SPACE.BT709;
    const colorRange = params.colorRange ?? YUV_COLOR_RANGE.LIMITED;
    const opacity = params.opacity ?? 1.0;
    const flags = params.flags ?? 0;
    const uvTransform = params.uvTransform ?? DEFAULT_UV_TRANSFORM;

    validatePixelFormat(pixelFormat);
    validateColorSpace(colorSpace);
    validateColorRange(colorRange);

    const matrix = YUV_RANGE_MATRICES[colorRange][colorSpace];
    const buffer = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
    const u32 = new Uint32Array(buffer);
    const f32 = new Float32Array(buffer);

    const packedFlags =
      ((flags >>> 0) | (colorRange === YUV_COLOR_RANGE.FULL ? UNIFORM_FLAG_FULL_RANGE : 0)) >>> 0;

    // Header (bytes 0..15)
    u32[0] = PIXEL_FORMAT_MODE[pixelFormat];
    u32[1] = COLOR_SPACE_ID[colorSpace];
    f32[2] = opacity;
    u32[3] = packedFlags;

    // uv_transform (bytes 16..31)
    f32[4] = uvTransform[0] ?? 1.0;
    f32[5] = uvTransform[1] ?? 1.0;
    f32[6] = uvTransform[2] ?? 0.0;
    f32[7] = uvTransform[3] ?? 0.0;

    // yuv_row0 (bytes 32..47)
    f32[8] = matrix.row0[0];
    f32[9] = matrix.row0[1];
    f32[10] = matrix.row0[2];
    f32[11] = matrix.row0[3];

    // yuv_row1 (bytes 48..63)
    f32[12] = matrix.row1[0];
    f32[13] = matrix.row1[1];
    f32[14] = matrix.row1[2];
    f32[15] = matrix.row1[3];

    // yuv_row2 (bytes 64..79)
    f32[16] = matrix.row2[0];
    f32[17] = matrix.row2[1];
    f32[18] = matrix.row2[2];
    f32[19] = matrix.row2[3];

    // yuv_offset (bytes 80..95)
    f32[20] = matrix.offset[0];
    f32[21] = matrix.offset[1];
    f32[22] = matrix.offset[2];
    f32[23] = matrix.offset[3];

    return new Uint8Array(buffer);
  }

  function validatePlaneBuffer(src, tightRowBytes, height, srcStride, label) {
    assertPositiveInteger(tightRowBytes, `${label} tightRowBytes`);
    assertPositiveInteger(height, `${label} height`);
    assertPositiveInteger(srcStride, `${label} stride`);

    if (srcStride < tightRowBytes) {
      throw new RangeError(
        `${label} stride (${srcStride}) must be >= tight row size (${tightRowBytes})`
      );
    }

    const minRequiredBytes = srcStride * (height - 1) + tightRowBytes;
    if (src.byteLength < minRequiredBytes) {
      throw new RangeError(
        `${label} buffer too small (${src.byteLength} bytes; need at least ${minRequiredBytes} bytes)`
      );
    }
  }

  /**
   * Prepare a plane's pixel rows with 256-byte `bytesPerRow` alignment, optionally reusing a
   * caller-supplied scratch `Uint8Array` to avoid per-frame allocations.
   *
   * @param {Uint8Array} src
   * @param {number} tightRowBytes
   * @param {number} height
   * @param {number} srcStride
   * @param {string} label
   * @param {Uint8Array | null} [scratchBuffer]
   * @returns {{
   *   data: Uint8Array,
   *   bytesPerRow: number,
   *   totalBytes: number,
   *   padded: boolean,
   *   usesScratch: boolean,
   * }}
   */
  function prepareAlignedPlaneData(
    src,
    tightRowBytes,
    height,
    srcStride,
    label,
    scratchBuffer = null
  ) {
    validatePlaneBuffer(src, tightRowBytes, height, srcStride, label);

    const alignedBytesPerRow = alignUp(tightRowBytes, WEBGPU_BYTES_PER_ROW_ALIGNMENT);
    const totalBytes = alignedBytesPerRow * height;

    if (srcStride === alignedBytesPerRow && src.byteLength >= totalBytes) {
      const slice = src.byteLength === totalBytes ? src : src.subarray(0, totalBytes);
      return {
        data: slice,
        bytesPerRow: alignedBytesPerRow,
        totalBytes,
        padded: false,
        usesScratch: false,
      };
    }

    const canReuseScratch =
      scratchBuffer instanceof Uint8Array && scratchBuffer.byteLength >= totalBytes;
    const dst = canReuseScratch
      ? scratchBuffer.subarray(0, totalBytes)
      : new Uint8Array(totalBytes);
    const hasPaddingTail = alignedBytesPerRow > tightRowBytes;

    for (let row = 0; row < height; row += 1) {
      const srcOffset = row * srcStride;
      const dstOffset = row * alignedBytesPerRow;
      dst.set(src.subarray(srcOffset, srcOffset + tightRowBytes), dstOffset);
      if (canReuseScratch && hasPaddingTail) {
        dst.fill(0, dstOffset + tightRowBytes, dstOffset + alignedBytesPerRow);
      }
    }

    return {
      data: dst,
      bytesPerRow: alignedBytesPerRow,
      totalBytes,
      padded: true,
      usesScratch: canReuseScratch,
    };
  }

  function createEmptySlot() {
    return {
      format: null,
      width: 0,
      height: 0,
      uvWidth: 0,
      uvHeight: 0,
      rgba: null,
      y: null,
      u: null,
      v: null,
      uv: null,
      a: null,
      bindGroup: null,
    };
  }

  function destroyPlaneResource(plane) {
    if (plane && plane.texture && typeof plane.texture.destroy === 'function') {
      plane.texture.destroy();
    }
  }

  function destroySlotResources(slot) {
    destroyPlaneResource(slot.rgba);
    destroyPlaneResource(slot.y);
    destroyPlaneResource(slot.u);
    destroyPlaneResource(slot.v);
    destroyPlaneResource(slot.uv);
    destroyPlaneResource(slot.a);
    slot.rgba = null;
    slot.y = null;
    slot.u = null;
    slot.v = null;
    slot.uv = null;
    slot.a = null;
    slot.bindGroup = null;
    slot.format = null;
    slot.width = 0;
    slot.height = 0;
    slot.uvWidth = 0;
    slot.uvHeight = 0;
  }

  /**
   * WebGPU video & splash frame presenter for Bink2 (`YUV420P`, `YUVA420P`, `NV12`) and `RGBA8` frames.
   */
  class BinkVideoPresenter {
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('BinkVideoPresenter constructor expects an options object');
      }

      const {
        device,
        presentationFormat = DEFAULT_PRESENTATION_FORMAT,
        stagingPool = null,
        colorSpace = YUV_COLOR_SPACE.BT709,
        colorRange = YUV_COLOR_RANGE.LIMITED,
        targetFps = DEFAULT_TARGET_FPS,
        doubleBuffer = true,
        slotCount,
      } = options;

      if (!device || typeof device.createTexture !== 'function') {
        throw new TypeError('BinkVideoPresenter requires a WebGPU device with createTexture()');
      }
      if (typeof presentationFormat !== 'string' || presentationFormat.length === 0) {
        throw new TypeError('presentationFormat must be a non-empty string');
      }
      if (
        stagingPool !== null &&
        stagingPool !== undefined &&
        (typeof stagingPool !== 'object' || typeof stagingPool.stageTextureUpload !== 'function')
      ) {
        throw new TypeError('stagingPool must be null or provide stageTextureUpload()');
      }
      validateColorSpace(colorSpace);
      validateColorRange(colorRange);
      assertPositiveFinite(targetFps, 'targetFps');

      const resolvedSlotCount =
        slotCount !== undefined ? slotCount : doubleBuffer ? DOUBLE_BUFFER_SLOTS : 1;
      assertPositiveInteger(resolvedSlotCount, 'slotCount');

      this._device = device;
      this._presentationFormat = presentationFormat;
      this._stagingPool = stagingPool || null;
      this._colorSpace = colorSpace;
      this._colorRange = colorRange;
      this._targetFps = targetFps;

      // Reusable plane texture slots (double-buffered by default).
      this._slots = [];
      for (let i = 0; i < resolvedSlotCount; i += 1) {
        this._slots.push(createEmptySlot());
      }
      this._activeSlotIndex = 0;
      this._hasFrame = false;
      this._unpresentedFrame = false;

      // Current video stream state.
      this._pixelFormat = null;
      this._frameWidth = 0;
      this._frameHeight = 0;
      this._uvTransform = [...DEFAULT_UV_TRANSFORM];

      // Deferred staging uploads queued when upload*Frame is called without a commandEncoder.
      this._pendingUploads = [];

      // Reusable scratch buffer for 256-byte row-pitch alignment.
      this._uploadScratch = null;
      this._scratchAllocations = 0;

      // Lazily constructed GPU pipeline resources.
      this._shaderModule = null;
      this._bindGroupLayout = null;
      this._pipelineLayout = null;
      this._pipeline = null;
      this._sampler = null;
      this._uniformBuffer = null;
      this._dummyTexture = null;
      this._dummyTextureView = null;
      this._uniformsDirty = true;
      this._lastUniformOpacity = null;

      // Playback pacing & control state.
      this._paused = false;
      this._looping = false;
      this._skipped = false;
      this._lastAdvanceMs = null;

      // Telemetry.
      this._uploadedFrames = 0;
      this._presentedFrames = 0;
      this._droppedFrames = 0;
      this._resizeCount = 0;
      this._planeReallocations = 0;
      this._bytesUploaded = 0;
      this._destroyed = false;
    }

    get presentationFormat() {
      return this._presentationFormat;
    }

    get colorSpace() {
      return this._colorSpace;
    }

    get colorRange() {
      return this._colorRange;
    }

    get targetFps() {
      return this._targetFps;
    }

    get pixelFormat() {
      return this._pixelFormat;
    }

    get frameWidth() {
      return this._frameWidth;
    }

    get frameHeight() {
      return this._frameHeight;
    }

    get isPaused() {
      return this._paused;
    }

    get isLooping() {
      return this._looping;
    }

    get isSkipped() {
      return this._skipped;
    }

    get isDestroyed() {
      return this._destroyed;
    }

    _assertNotDestroyed() {
      if (this._destroyed) {
        throw new Error('BinkVideoPresenter has been destroyed');
      }
    }

    _createPlaneTexture(width, height, format, label) {
      const texture = this._device.createTexture({
        label,
        size: { width, height, depthOrArrayLayers: 1 },
        format,
        usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
      });
      const view =
        texture && typeof texture.createView === 'function' ? texture.createView() : null;
      return { texture, view, width, height, format };
    }

    _destroyAllSlots() {
      for (let i = 0; i < this._slots.length; i += 1) {
        destroySlotResources(this._slots[i]);
      }
      this._pendingUploads = [];
    }

    _selectWriteSlot(format, width, height) {
      const formatOrSizeChanged =
        this._hasFrame &&
        (this._pixelFormat !== format ||
          this._frameWidth !== width ||
          this._frameHeight !== height);

      if (formatOrSizeChanged) {
        // Destroy stale textures across all slots immediately on resolution/format change.
        this._destroyAllSlots();
        this._resizeCount += 1;
        this._planeReallocations += 1;
        this._uniformsDirty = true;
        return 0;
      }

      if (!this._hasFrame) {
        this._planeReallocations += 1;
        return 0;
      }

      return (this._activeSlotIndex + 1) % this._slots.length;
    }

    _ensureSlotTextures(slotIndex, format, width, height) {
      const slot = this._slots[slotIndex];
      const uvWidth = Math.ceil(width / 2);
      const uvHeight = Math.ceil(height / 2);

      const matches =
        slot.format === format &&
        slot.width === width &&
        slot.height === height &&
        (format === BINK_PIXEL_FORMAT.RGBA8
          ? slot.rgba !== null
          : format === BINK_PIXEL_FORMAT.NV12
            ? slot.y !== null && slot.uv !== null
            : slot.y !== null &&
              slot.u !== null &&
              slot.v !== null &&
              (format !== BINK_PIXEL_FORMAT.YUVA420P || slot.a !== null));

      if (matches) {
        return slot;
      }

      destroySlotResources(slot);

      if (format === BINK_PIXEL_FORMAT.RGBA8) {
        slot.rgba = this._createPlaneTexture(
          width,
          height,
          'rgba8unorm',
          `playgta6-bink-rgba-slot${slotIndex}`
        );
      } else if (format === BINK_PIXEL_FORMAT.NV12) {
        slot.y = this._createPlaneTexture(
          width,
          height,
          'r8unorm',
          `playgta6-bink-y-slot${slotIndex}`
        );
        slot.uv = this._createPlaneTexture(
          uvWidth,
          uvHeight,
          'rg8unorm',
          `playgta6-bink-uv-slot${slotIndex}`
        );
      } else {
        slot.y = this._createPlaneTexture(
          width,
          height,
          'r8unorm',
          `playgta6-bink-y-slot${slotIndex}`
        );
        slot.u = this._createPlaneTexture(
          uvWidth,
          uvHeight,
          'r8unorm',
          `playgta6-bink-u-slot${slotIndex}`
        );
        slot.v = this._createPlaneTexture(
          uvWidth,
          uvHeight,
          'r8unorm',
          `playgta6-bink-v-slot${slotIndex}`
        );
        if (format === BINK_PIXEL_FORMAT.YUVA420P) {
          slot.a = this._createPlaneTexture(
            width,
            height,
            'r8unorm',
            `playgta6-bink-a-slot${slotIndex}`
          );
        }
      }

      slot.format = format;
      slot.width = width;
      slot.height = height;
      slot.uvWidth = uvWidth;
      slot.uvHeight = uvHeight;
      slot.bindGroup = null;
      return slot;
    }

    _preparePlaneForUpload(src, tightRowBytes, height, srcStride, label) {
      validatePlaneBuffer(src, tightRowBytes, height, srcStride, label);

      const alignedBytesPerRow = alignUp(tightRowBytes, WEBGPU_BYTES_PER_ROW_ALIGNMENT);
      const totalBytes = alignedBytesPerRow * height;

      if (srcStride === alignedBytesPerRow && src.byteLength >= totalBytes) {
        return prepareAlignedPlaneData(src, tightRowBytes, height, srcStride, label);
      }

      if (!this._uploadScratch || this._uploadScratch.byteLength < totalBytes) {
        this._uploadScratch = new Uint8Array(totalBytes);
        this._scratchAllocations += 1;
      }

      return prepareAlignedPlaneData(
        src,
        tightRowBytes,
        height,
        srcStride,
        label,
        this._uploadScratch
      );
    }

    _uploadPlane(slotIndex, dstTexture, prepared, width, height, commandEncoder) {
      if (this._stagingPool) {
        const layout = {
          width,
          height,
          bytesPerRow: prepared.bytesPerRow,
          rowsPerImage: height,
        };
        if (commandEncoder) {
          this._stagingPool.stageTextureUpload(commandEncoder, dstTexture, prepared.data, layout);
        } else {
          // Clone non-owned or scratch-backed data so subsequent plane uploads cannot overwrite it.
          const ownedData =
            prepared.padded && !prepared.usesScratch ? prepared.data : new Uint8Array(prepared.data);
          this._pendingUploads.push({
            slotIndex,
            texture: dstTexture,
            data: ownedData,
            layout,
          });
        }
      } else {
        if (!this._device.queue || typeof this._device.queue.writeTexture !== 'function') {
          throw new TypeError(
            'BinkVideoPresenter requires device.queue.writeTexture() when no stagingPool is supplied'
          );
        }
        this._device.queue.writeTexture(
          { texture: dstTexture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
          prepared.data,
          { offset: 0, bytesPerRow: prepared.bytesPerRow, rowsPerImage: height },
          { width, height, depthOrArrayLayers: 1 }
        );
      }

      this._bytesUploaded += prepared.totalBytes;
    }

    _dropPendingUploadsForSlot(slotIndex) {
      if (this._pendingUploads.length === 0) {
        return;
      }
      this._pendingUploads = this._pendingUploads.filter((item) => item.slotIndex !== slotIndex);
    }

    /**
     * Flush any queued stagingPool texture uploads onto `commandEncoder`.
     * Called automatically by `presentFrame()` before beginning the render pass.
     *
     * @param {GPUCommandEncoder} commandEncoder
     * @returns {number} Number of plane uploads flushed.
     */
    flushPendingUploads(commandEncoder) {
      this._assertNotDestroyed();
      if (!this._stagingPool || this._pendingUploads.length === 0) {
        return 0;
      }
      if (!commandEncoder || typeof commandEncoder.copyBufferToTexture !== 'function') {
        throw new TypeError(
          'flushPendingUploads requires a GPUCommandEncoder with copyBufferToTexture()'
        );
      }

      const pending = this._pendingUploads;
      this._pendingUploads = [];
      for (let i = 0; i < pending.length; i += 1) {
        const item = pending[i];
        this._stagingPool.stageTextureUpload(
          commandEncoder,
          item.texture,
          item.data,
          item.layout
        );
      }
      return pending.length;
    }

    /**
     * Upload an RGBA8 video or splash frame into the next double-buffered texture slot.
     *
     * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} rgbaBytes
     * @param {number} width
     * @param {number} height
     * @param {{ strideBytes?: number, commandEncoder?: GPUCommandEncoder, encoder?: GPUCommandEncoder }} [options]
     */
    uploadRgbaFrame(rgbaBytes, width, height, options = {}) {
      this._assertNotDestroyed();
      assertPositiveInteger(width, 'width');
      assertPositiveInteger(height, 'height');
      if (options !== undefined && options !== null && typeof options !== 'object') {
        throw new TypeError('uploadRgbaFrame options must be an object');
      }

      const src = toUint8Array(rgbaBytes, 'uploadRgbaFrame');
      const tightRowBytes = width * 4;
      const strideBytes = options && options.strideBytes !== undefined ? options.strideBytes : tightRowBytes;
      const commandEncoder = options ? options.commandEncoder || options.encoder || null : null;

      validatePlaneBuffer(src, tightRowBytes, height, strideBytes, 'RGBA8');

      if (this._unpresentedFrame) {
        this._droppedFrames += 1;
      }

      const writeSlotIndex = this._selectWriteSlot(BINK_PIXEL_FORMAT.RGBA8, width, height);
      const slot = this._ensureSlotTextures(writeSlotIndex, BINK_PIXEL_FORMAT.RGBA8, width, height);
      this._dropPendingUploadsForSlot(writeSlotIndex);

      const prepared = this._preparePlaneForUpload(src, tightRowBytes, height, strideBytes, 'RGBA8');
      this._uploadPlane(writeSlotIndex, slot.rgba.texture, prepared, width, height, commandEncoder);

      if (this._pixelFormat !== BINK_PIXEL_FORMAT.RGBA8) {
        this._uniformsDirty = true;
      }
      this._pixelFormat = BINK_PIXEL_FORMAT.RGBA8;
      this._frameWidth = width;
      this._frameHeight = height;
      this._activeSlotIndex = writeSlotIndex;
      this._hasFrame = true;
      this._unpresentedFrame = true;
      this._uploadedFrames += 1;

      return {
        format: BINK_PIXEL_FORMAT.RGBA8,
        width,
        height,
        bytesPerRow: prepared.bytesPerRow,
        bytesUploaded: prepared.totalBytes,
        slotIndex: writeSlotIndex,
        uploadedFrames: this._uploadedFrames,
      };
    }

    /**
     * Upload a planar YUV420P or YUVA420P (with optional alpha plane) video frame into
     * the next reusable plane texture slot.
     *
     * @param {{
     *   y: Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer,
     *   u: Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer,
     *   v: Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer,
     *   a?: Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer | null,
     *   width: number,
     *   height: number,
     *   yStride?: number,
     *   uvStride?: number,
     *   aStride?: number,
     *   colorSpace?: string,
     *   colorRange?: string,
     *   commandEncoder?: GPUCommandEncoder,
     *   encoder?: GPUCommandEncoder,
     * }} frame
     */
    uploadYuv420Frame(frame) {
      this._assertNotDestroyed();
      if (!frame || typeof frame !== 'object') {
        throw new TypeError('uploadYuv420Frame expects a frame descriptor object');
      }

      const {
        y,
        u,
        v,
        a = null,
        width,
        height,
        yStride,
        uvStride,
        aStride,
        colorSpace,
        colorRange,
      } = frame;
      const commandEncoder = frame.commandEncoder || frame.encoder || null;

      assertPositiveInteger(width, 'width');
      assertPositiveInteger(height, 'height');

      if (colorSpace !== undefined && colorSpace !== null) {
        validateColorSpace(colorSpace);
      }
      if (colorRange !== undefined && colorRange !== null) {
        validateColorRange(colorRange);
      }

      const uvWidth = Math.ceil(width / 2);
      const uvHeight = Math.ceil(height / 2);
      const hasAlpha = a !== null && a !== undefined;
      const format = hasAlpha ? BINK_PIXEL_FORMAT.YUVA420P : BINK_PIXEL_FORMAT.YUV420P;

      const ySrc = toUint8Array(y, 'uploadYuv420Frame.y');
      const uSrc = toUint8Array(u, 'uploadYuv420Frame.u');
      const vSrc = toUint8Array(v, 'uploadYuv420Frame.v');
      const aSrc = hasAlpha ? toUint8Array(a, 'uploadYuv420Frame.a') : null;

      const resolvedYStride = yStride !== undefined ? yStride : width;
      const resolvedUvStride = uvStride !== undefined ? uvStride : uvWidth;
      const resolvedAStride = aStride !== undefined ? aStride : width;

      validatePlaneBuffer(ySrc, width, height, resolvedYStride, 'Y plane');
      validatePlaneBuffer(uSrc, uvWidth, uvHeight, resolvedUvStride, 'U plane');
      validatePlaneBuffer(vSrc, uvWidth, uvHeight, resolvedUvStride, 'V plane');
      if (hasAlpha && aSrc) {
        validatePlaneBuffer(aSrc, width, height, resolvedAStride, 'A plane');
      }

      if (this._unpresentedFrame) {
        this._droppedFrames += 1;
      }

      const writeSlotIndex = this._selectWriteSlot(format, width, height);
      const slot = this._ensureSlotTextures(writeSlotIndex, format, width, height);
      this._dropPendingUploadsForSlot(writeSlotIndex);

      const preparedY = this._preparePlaneForUpload(
        ySrc,
        width,
        height,
        resolvedYStride,
        'Y plane'
      );
      this._uploadPlane(writeSlotIndex, slot.y.texture, preparedY, width, height, commandEncoder);

      const preparedU = this._preparePlaneForUpload(
        uSrc,
        uvWidth,
        uvHeight,
        resolvedUvStride,
        'U plane'
      );
      this._uploadPlane(
        writeSlotIndex,
        slot.u.texture,
        preparedU,
        uvWidth,
        uvHeight,
        commandEncoder
      );

      const preparedV = this._preparePlaneForUpload(
        vSrc,
        uvWidth,
        uvHeight,
        resolvedUvStride,
        'V plane'
      );
      this._uploadPlane(
        writeSlotIndex,
        slot.v.texture,
        preparedV,
        uvWidth,
        uvHeight,
        commandEncoder
      );

      let preparedA = null;
      if (hasAlpha && aSrc) {
        preparedA = this._preparePlaneForUpload(
          aSrc,
          width,
          height,
          resolvedAStride,
          'A plane'
        );
        this._uploadPlane(writeSlotIndex, slot.a.texture, preparedA, width, height, commandEncoder);
      }

      if (colorSpace !== undefined && colorSpace !== null && colorSpace !== this._colorSpace) {
        this._colorSpace = colorSpace;
        this._uniformsDirty = true;
      }
      if (colorRange !== undefined && colorRange !== null && colorRange !== this._colorRange) {
        this._colorRange = colorRange;
        this._uniformsDirty = true;
      }
      if (this._pixelFormat !== format) {
        this._uniformsDirty = true;
      }

      this._pixelFormat = format;
      this._frameWidth = width;
      this._frameHeight = height;
      this._activeSlotIndex = writeSlotIndex;
      this._hasFrame = true;
      this._unpresentedFrame = true;
      this._uploadedFrames += 1;

      const bytesUploaded =
        preparedY.totalBytes +
        preparedU.totalBytes +
        preparedV.totalBytes +
        (preparedA ? preparedA.totalBytes : 0);

      return {
        format,
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        width,
        height,
        uvWidth,
        uvHeight,
        yBytesPerRow: preparedY.bytesPerRow,
        uvBytesPerRow: preparedU.bytesPerRow,
        aBytesPerRow: preparedA ? preparedA.bytesPerRow : null,
        bytesUploaded,
        slotIndex: writeSlotIndex,
        uploadedFrames: this._uploadedFrames,
      };
    }

    /**
     * Upload a semi-planar NV12 (`r8unorm` Y plane + interleaved `rg8unorm` UV plane) video frame
     * into the next reusable plane texture slot.
     *
     * @param {{
     *   y: Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer,
     *   uv: Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer,
     *   width: number,
     *   height: number,
     *   yStride?: number,
     *   uvStride?: number,
     *   colorSpace?: string,
     *   colorRange?: string,
     *   commandEncoder?: GPUCommandEncoder,
     *   encoder?: GPUCommandEncoder,
     * }} frame
     */
    uploadNv12Frame(frame) {
      this._assertNotDestroyed();
      if (!frame || typeof frame !== 'object') {
        throw new TypeError('uploadNv12Frame expects a frame descriptor object');
      }

      const {
        y,
        uv,
        width,
        height,
        yStride,
        uvStride,
        colorSpace,
        colorRange,
      } = frame;
      const commandEncoder = frame.commandEncoder || frame.encoder || null;

      assertPositiveInteger(width, 'width');
      assertPositiveInteger(height, 'height');

      if (colorSpace !== undefined && colorSpace !== null) {
        validateColorSpace(colorSpace);
      }
      if (colorRange !== undefined && colorRange !== null) {
        validateColorRange(colorRange);
      }

      const uvWidth = Math.ceil(width / 2);
      const uvHeight = Math.ceil(height / 2);
      const uvTightRowBytes = uvWidth * 2;
      const resolvedYStride = yStride !== undefined ? yStride : width;
      const resolvedUvStride = uvStride !== undefined ? uvStride : uvTightRowBytes;

      const ySrc = toUint8Array(y, 'uploadNv12Frame.y');
      const uvSrc = toUint8Array(uv, 'uploadNv12Frame.uv');

      validatePlaneBuffer(ySrc, width, height, resolvedYStride, 'Y plane');
      validatePlaneBuffer(uvSrc, uvTightRowBytes, uvHeight, resolvedUvStride, 'UV plane');

      if (this._unpresentedFrame) {
        this._droppedFrames += 1;
      }

      const writeSlotIndex = this._selectWriteSlot(BINK_PIXEL_FORMAT.NV12, width, height);
      const slot = this._ensureSlotTextures(writeSlotIndex, BINK_PIXEL_FORMAT.NV12, width, height);
      this._dropPendingUploadsForSlot(writeSlotIndex);

      const preparedY = this._preparePlaneForUpload(
        ySrc,
        width,
        height,
        resolvedYStride,
        'Y plane'
      );
      this._uploadPlane(writeSlotIndex, slot.y.texture, preparedY, width, height, commandEncoder);

      const preparedUv = this._preparePlaneForUpload(
        uvSrc,
        uvTightRowBytes,
        uvHeight,
        resolvedUvStride,
        'UV plane'
      );
      this._uploadPlane(
        writeSlotIndex,
        slot.uv.texture,
        preparedUv,
        uvWidth,
        uvHeight,
        commandEncoder
      );

      if (colorSpace !== undefined && colorSpace !== null && colorSpace !== this._colorSpace) {
        this._colorSpace = colorSpace;
        this._uniformsDirty = true;
      }
      if (colorRange !== undefined && colorRange !== null && colorRange !== this._colorRange) {
        this._colorRange = colorRange;
        this._uniformsDirty = true;
      }
      if (this._pixelFormat !== BINK_PIXEL_FORMAT.NV12) {
        this._uniformsDirty = true;
      }

      this._pixelFormat = BINK_PIXEL_FORMAT.NV12;
      this._frameWidth = width;
      this._frameHeight = height;
      this._activeSlotIndex = writeSlotIndex;
      this._hasFrame = true;
      this._unpresentedFrame = true;
      this._uploadedFrames += 1;

      const bytesUploaded = preparedY.totalBytes + preparedUv.totalBytes;

      return {
        format: BINK_PIXEL_FORMAT.NV12,
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        width,
        height,
        uvWidth,
        uvHeight,
        yBytesPerRow: preparedY.bytesPerRow,
        uvBytesPerRow: preparedUv.bytesPerRow,
        bytesUploaded,
        slotIndex: writeSlotIndex,
        uploadedFrames: this._uploadedFrames,
      };
    }

    /**
     * Dispatch a frame descriptor to `uploadRgbaFrame`, `uploadYuv420Frame`, or `uploadNv12Frame`.
     *
     * @param {object} frame
     */
    uploadFrame(frame) {
      this._assertNotDestroyed();
      if (!frame || typeof frame !== 'object') {
        throw new TypeError('uploadFrame expects a frame descriptor object');
      }
      const format = frame.format;
      if (format === BINK_PIXEL_FORMAT.NV12 || (!format && frame.y && frame.uv && !frame.u && !frame.v)) {
        return this.uploadNv12Frame(frame);
      }
      if (
        format === BINK_PIXEL_FORMAT.YUV420P ||
        format === BINK_PIXEL_FORMAT.YUVA420P ||
        (!format && frame.y && frame.u && frame.v)
      ) {
        return this.uploadYuv420Frame(frame);
      }
      if (format === BINK_PIXEL_FORMAT.RGBA8 || (!format && (frame.rgba || frame.data))) {
        return this.uploadRgbaFrame(frame.rgba || frame.data, frame.width, frame.height, {
          strideBytes: frame.strideBytes,
          commandEncoder: frame.commandEncoder || frame.encoder || null,
        });
      }
      validatePixelFormat(format);
      throw new TypeError('uploadFrame could not determine pixel format from frame descriptor');
    }

    _ensurePipelineResources() {
      if (this._pipeline !== null) {
        return;
      }

      this._shaderModule = this._device.createShaderModule({
        label: 'playgta6-bink-presenter-shader',
        code: BINK_PRESENTER_WGSL,
      });

      this._bindGroupLayout = this._device.createBindGroupLayout({
        label: 'playgta6-bink-presenter-bgl',
        entries: [
          {
            binding: 0,
            visibility: SHADER_STAGE.VERTEX | SHADER_STAGE.FRAGMENT,
            buffer: { type: 'uniform' },
          },
          {
            binding: 1,
            visibility: SHADER_STAGE.FRAGMENT,
            sampler: { type: 'filtering' },
          },
          {
            binding: 2,
            visibility: SHADER_STAGE.FRAGMENT,
            texture: { sampleType: 'float', viewDimension: '2d' },
          },
          {
            binding: 3,
            visibility: SHADER_STAGE.FRAGMENT,
            texture: { sampleType: 'float', viewDimension: '2d' },
          },
          {
            binding: 4,
            visibility: SHADER_STAGE.FRAGMENT,
            texture: { sampleType: 'float', viewDimension: '2d' },
          },
          {
            binding: 5,
            visibility: SHADER_STAGE.FRAGMENT,
            texture: { sampleType: 'float', viewDimension: '2d' },
          },
        ],
      });

      this._pipelineLayout =
        typeof this._device.createPipelineLayout === 'function'
          ? this._device.createPipelineLayout({
              label: 'playgta6-bink-presenter-layout',
              bindGroupLayouts: [this._bindGroupLayout],
            })
          : 'auto';

      this._pipeline = this._device.createRenderPipeline({
        label: 'playgta6-bink-presenter-pipeline',
        layout: this._pipelineLayout,
        vertex: {
          module: this._shaderModule,
          entryPoint: 'vs_main',
        },
        fragment: {
          module: this._shaderModule,
          entryPoint: 'fs_main',
          targets: [
            {
              format: this._presentationFormat,
              blend: {
                color: {
                  srcFactor: 'src-alpha',
                  dstFactor: 'one-minus-src-alpha',
                  operation: 'add',
                },
                alpha: {
                  srcFactor: 'one',
                  dstFactor: 'one-minus-src-alpha',
                  operation: 'add',
                },
              },
            },
          ],
        },
        primitive: {
          topology: 'triangle-list',
          cullMode: 'none',
        },
      });

      this._sampler = this._device.createSampler({
        label: 'playgta6-bink-presenter-sampler',
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });

      this._uniformBuffer = this._device.createBuffer({
        label: 'playgta6-bink-presenter-uniforms',
        size: UNIFORM_BUFFER_SIZE,
        usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
      });

      // 1x1 neutral fallback texture so all 4 texture bindings remain valid in every mode.
      this._dummyTexture = this._device.createTexture({
        label: 'playgta6-bink-presenter-dummy',
        size: { width: 1, height: 1, depthOrArrayLayers: 1 },
        format: 'r8unorm',
        usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
      });
      this._dummyTextureView =
        this._dummyTexture && typeof this._dummyTexture.createView === 'function'
          ? this._dummyTexture.createView()
          : null;

      if (this._device.queue && typeof this._device.queue.writeTexture === 'function') {
        const dummyBytes = new Uint8Array(WEBGPU_BYTES_PER_ROW_ALIGNMENT);
        dummyBytes[0] = 0xff;
        this._device.queue.writeTexture(
          { texture: this._dummyTexture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
          dummyBytes,
          { offset: 0, bytesPerRow: WEBGPU_BYTES_PER_ROW_ALIGNMENT, rowsPerImage: 1 },
          { width: 1, height: 1, depthOrArrayLayers: 1 }
        );
      }

      this._uniformsDirty = true;
    }

    _ensureSlotBindGroup(slotIndex) {
      const slot = this._slots[slotIndex];
      if (slot.bindGroup) {
        return slot.bindGroup;
      }

      const primaryView =
        slot.format === BINK_PIXEL_FORMAT.RGBA8 ? slot.rgba.view : slot.y.view;
      const uView =
        slot.format === BINK_PIXEL_FORMAT.RGBA8
          ? this._dummyTextureView
          : slot.format === BINK_PIXEL_FORMAT.NV12
            ? slot.uv.view
            : slot.u.view;
      const vView =
        slot.format === BINK_PIXEL_FORMAT.YUV420P ||
        slot.format === BINK_PIXEL_FORMAT.YUVA420P
          ? slot.v.view
          : this._dummyTextureView;
      const aView =
        slot.format === BINK_PIXEL_FORMAT.YUVA420P && slot.a
          ? slot.a.view
          : this._dummyTextureView;

      slot.bindGroup = this._device.createBindGroup({
        label: `playgta6-bink-presenter-bg-slot${slotIndex}`,
        layout: this._bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this._uniformBuffer } },
          { binding: 1, resource: this._sampler },
          { binding: 2, resource: primaryView },
          { binding: 3, resource: uView },
          { binding: 4, resource: vView },
          { binding: 5, resource: aView },
        ],
      });

      return slot.bindGroup;
    }

    _syncUniformBuffer(opacity) {
      if (!this._uniformsDirty && this._lastUniformOpacity === opacity) {
        return;
      }
      const packed = packPresenterUniforms({
        pixelFormat: this._pixelFormat || BINK_PIXEL_FORMAT.RGBA8,
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        opacity,
        uvTransform: this._uvTransform,
      });
      if (this._device.queue && typeof this._device.queue.writeBuffer === 'function') {
        this._device.queue.writeBuffer(this._uniformBuffer, 0, packed);
      }
      this._uniformsDirty = false;
      this._lastUniformOpacity = opacity;
    }

    /**
     * Encode a WebGPU render pass that blits the most recently uploaded frame onto
     * `targetView` with aspect-ratio-preserving letterboxing (`setViewport`) and a
     * single full-screen triangle `draw(3, 1, 0, 0)`.
     *
     * @param {GPUCommandEncoder} commandEncoder
     * @param {GPUTextureView} targetView
     * @param {number} dstWidth
     * @param {number} dstHeight
     * @param {{
     *   clearColor?: { r: number, g: number, b: number, a: number } | number[],
     *   loadOp?: 'clear' | 'load',
     *   storeOp?: 'store' | 'discard',
     *   opacity?: number,
     * }} [options]
     */
    presentFrame(commandEncoder, targetView, dstWidth, dstHeight, options = {}) {
      this._assertNotDestroyed();
      if (!commandEncoder || typeof commandEncoder.beginRenderPass !== 'function') {
        throw new TypeError('presentFrame requires a GPUCommandEncoder with beginRenderPass()');
      }
      if (!targetView || typeof targetView !== 'object') {
        throw new TypeError('presentFrame requires a target GPUTextureView');
      }
      assertPositiveFinite(dstWidth, 'dstWidth');
      assertPositiveFinite(dstHeight, 'dstHeight');

      const {
        clearColor = DEFAULT_CLEAR_COLOR,
        loadOp = 'clear',
        storeOp = 'store',
        opacity = 1.0,
      } = options || {};

      // Flush any deferred staging uploads before starting the render pass.
      if (this._stagingPool && this._pendingUploads.length > 0) {
        this.flushPendingUploads(commandEncoder);
      }

      this._ensurePipelineResources();

      const colorAttachment = {
        view: targetView,
        loadOp,
        storeOp,
      };
      if (loadOp === 'clear') {
        colorAttachment.clearValue = normalizeClearColor(clearColor);
      }

      const pass = commandEncoder.beginRenderPass({
        label: 'playgta6-bink-present-pass',
        colorAttachments: [colorAttachment],
      });

      if (!this._hasFrame) {
        pass.end();
        return {
          presented: false,
          viewport: null,
          format: this._pixelFormat,
          colorSpace: this._colorSpace,
          colorRange: this._colorRange,
          slotIndex: this._activeSlotIndex,
          presentedFrames: this._presentedFrames,
          framesPresented: this._presentedFrames,
        };
      }

      const viewport = computeLetterboxViewport(
        this._frameWidth,
        this._frameHeight,
        dstWidth,
        dstHeight
      );

      this._syncUniformBuffer(opacity);
      const bindGroup = this._ensureSlotBindGroup(this._activeSlotIndex);

      pass.setPipeline(this._pipeline);
      pass.setViewport(viewport.x, viewport.y, viewport.width, viewport.height, 0, 1);
      if (typeof pass.setScissorRect === 'function') {
        const sx = Math.max(0, Math.floor(viewport.x));
        const sy = Math.max(0, Math.floor(viewport.y));
        const sw = Math.max(1, Math.min(Math.floor(dstWidth) - sx, Math.ceil(viewport.width)));
        const sh = Math.max(1, Math.min(Math.floor(dstHeight) - sy, Math.ceil(viewport.height)));
        pass.setScissorRect(sx, sy, sw, sh);
      }
      pass.setBindGroup(0, bindGroup);
      pass.draw(3, 1, 0, 0);
      pass.end();

      this._presentedFrames += 1;
      this._unpresentedFrame = false;

      return {
        presented: true,
        viewport,
        format: this._pixelFormat,
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        slotIndex: this._activeSlotIndex,
        presentedFrames: this._presentedFrames,
        framesPresented: this._presentedFrames,
      };
    }

    /**
     * Set UV crop/scale transform `[scaleX, scaleY, offsetX, offsetY]` applied in the shader.
     */
    setUvTransform(scaleX = 1, scaleY = 1, offsetX = 0, offsetY = 0) {
      this._assertNotDestroyed();
      this._uvTransform = [scaleX, scaleY, offsetX, offsetY];
      this._uniformsDirty = true;
    }

    /**
     * Switch active YUV color space (`bt709` or `bt601`) and optionally color range
     * (`limited` or `full`).
     */
    setColorSpace(colorSpace, colorRange) {
      this._assertNotDestroyed();
      validateColorSpace(colorSpace);
      if (colorRange !== undefined && colorRange !== null) {
        validateColorRange(colorRange);
      }
      if (colorSpace !== this._colorSpace) {
        this._colorSpace = colorSpace;
        this._uniformsDirty = true;
      }
      if (colorRange !== undefined && colorRange !== null && colorRange !== this._colorRange) {
        this._colorRange = colorRange;
        this._uniformsDirty = true;
      }
    }

    /**
     * Switch active YUV color range (`limited` or `full`).
     */
    setColorRange(colorRange) {
      this._assertNotDestroyed();
      validateColorRange(colorRange);
      if (colorRange !== this._colorRange) {
        this._colorRange = colorRange;
        this._uniformsDirty = true;
      }
    }

    /**
     * Record dropped frames detected by an external playback driver.
     */
    recordDroppedFrame(count = 1) {
      this._assertNotDestroyed();
      assertPositiveInteger(count, 'count');
      this._droppedFrames += count;
    }

    /**
     * Update playback target FPS for `shouldAdvance(nowMs)`.
     */
    setTargetFps(fps) {
      this._assertNotDestroyed();
      assertPositiveFinite(fps, 'targetFps');
      this._targetFps = fps;
    }

    /**
     * Pause or resume frame advancement.
     */
    setPaused(paused) {
      this._assertNotDestroyed();
      const next = Boolean(paused);
      if (this._paused && !next) {
        // Reset timing anchor on resume so the next tick advances cleanly without burst catch-up.
        this._lastAdvanceMs = null;
      }
      this._paused = next;
    }

    /**
     * Enable or disable looping state metadata for the playback driver.
     */
    setLooping(looping) {
      this._assertNotDestroyed();
      this._looping = Boolean(looping);
    }

    /**
     * Mark the active video/splash sequence as skipped (stops `shouldAdvance`).
     */
    setSkipped(skipped) {
      this._assertNotDestroyed();
      this._skipped = Boolean(skipped);
    }

    /**
     * Convenience helper to skip the active intro/splash sequence.
     */
    skip() {
      this.setSkipped(true);
    }

    /**
     * Determine whether the video decoder should advance to the next frame at timestamp `nowMs`
     * according to `targetFps`, pause state, and skip state.
     *
     * @param {number} nowMs Monotonic timestamp in milliseconds.
     * @returns {boolean}
     */
    shouldAdvance(nowMs) {
      if (this._destroyed || this._paused || this._skipped) {
        return false;
      }
      if (typeof nowMs !== 'number' || !Number.isFinite(nowMs)) {
        throw new RangeError(`nowMs must be a finite number, got ${nowMs}`);
      }

      const intervalMs = 1000 / this._targetFps;

      if (this._lastAdvanceMs === null) {
        this._lastAdvanceMs = nowMs;
        return true;
      }

      if (nowMs < this._lastAdvanceMs) {
        this._lastAdvanceMs = nowMs;
        return false;
      }

      const elapsed = nowMs - this._lastAdvanceMs;
      if (elapsed + 1e-6 >= intervalMs) {
        if (elapsed >= intervalMs * 2) {
          const missedIntervals = Math.floor((elapsed + 1e-6) / intervalMs) - 1;
          if (missedIntervals > 0) {
            this._droppedFrames += missedIntervals;
          }
          this._lastAdvanceMs = nowMs;
        } else {
          this._lastAdvanceMs += intervalMs;
        }
        return true;
      }

      return false;
    }

    /**
     * Return presenter state and frame/upload telemetry.
     */
    getStats() {
      return {
        pixelFormat: this._pixelFormat,
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        presentationFormat: this._presentationFormat,
        frameWidth: this._frameWidth,
        frameHeight: this._frameHeight,
        lastFrameWidth: this._frameWidth,
        lastFrameHeight: this._frameHeight,
        targetFps: this._targetFps,
        frameIntervalMs: 1000 / this._targetFps,
        uploadedFrames: this._uploadedFrames,
        presentedFrames: this._presentedFrames,
        framesPresented: this._presentedFrames,
        droppedFrames: this._droppedFrames,
        resizeCount: this._resizeCount,
        planeReallocations: this._planeReallocations,
        scratchAllocations: this._scratchAllocations,
        bytesUploaded: this._bytesUploaded,
        slotCount: this._slots.length,
        activeSlotIndex: this._activeSlotIndex,
        hasFrame: this._hasFrame,
        paused: this._paused,
        looping: this._looping,
        skipped: this._skipped,
        pipelineReady: this._pipeline !== null,
        usesStagingPool: this._stagingPool !== null,
        pendingUploads: this._pendingUploads.length,
        destroyed: this._destroyed,
      };
    }

    /**
     * Destroy all allocated GPU textures and uniform buffers idempotently.
     */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;

      this._destroyAllSlots();

      if (this._dummyTexture && typeof this._dummyTexture.destroy === 'function') {
        this._dummyTexture.destroy();
      }
      this._dummyTexture = null;
      this._dummyTextureView = null;

      if (this._uniformBuffer && typeof this._uniformBuffer.destroy === 'function') {
        this._uniformBuffer.destroy();
      }
      this._uniformBuffer = null;

      this._uploadScratch = null;
      this._shaderModule = null;
      this._bindGroupLayout = null;
      this._pipelineLayout = null;
      this._pipeline = null;
      this._sampler = null;
      this._hasFrame = false;
      this._unpresentedFrame = false;
      this._lastAdvanceMs = null;
      this._device = null;
      this._stagingPool = null;
    }

    /**
     * Alias for `destroy()` for callers expecting a `dispose()` lifecycle hook.
     */
    dispose() {
      this.destroy();
    }
  }

  const exportedApi = Object.freeze({
    WEBGPU_BYTES_PER_ROW_ALIGNMENT,
    UNIFORM_BUFFER_SIZE,
    UNIFORM_FLAG_FULL_RANGE,
    DEFAULT_PRESENTATION_FORMAT,
    DEFAULT_TARGET_FPS,
    BINK_PIXEL_FORMAT,
    YUV_COLOR_SPACE,
    YUV_COLOR_RANGE,
    YUV_MATRICES,
    YUV_FULL_RANGE_MATRICES,
    YUV_RANGE_MATRICES,
    TEXTURE_USAGE,
    BUFFER_USAGE,
    SHADER_STAGE,
    BINK_PRESENTER_WGSL,
    alignUp,
    getYuvMatrix,
    prepareAlignedPlaneData,
    computeLetterboxViewport,
    packPresenterUniforms,
    BinkVideoPresenter,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6BinkPresenter = exportedApi;
  } else if (root) {
    root.PlayGta6BinkPresenter = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
