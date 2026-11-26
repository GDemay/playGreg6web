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
  const UNIFORM_FLOAT_COUNT = UNIFORM_BUFFER_SIZE / 4;
  const DEFAULT_PRESENTATION_FORMAT = 'bgra8unorm';
  const DEFAULT_TARGET_FPS = 30;
  const TARGET_FPS_30_INTERVAL_MS = 1000 / 30;
  const DEFAULT_BIND_GROUP_CACHE_CAPACITY = 16;
  const DOUBLE_BUFFER_SLOTS = 2;
  const TRIPLE_BUFFER_SLOTS = 3;
  const UNIFORM_FLAG_FULL_RANGE = 1;
  const UNIFORM_FLAG_SWAP_CHROMA = 2;
  const UNIFORM_FLAG_OUTPUT_LINEAR = 4;

  const BINK_PIXEL_FORMAT = Object.freeze({
    RGBA8: 'rgba8',
    BGRA8: 'bgra8',
    YUV420P: 'yuv420p',
    YUVA420P: 'yuva420p',
    NV12: 'nv12',
  });

  const BINK_CHROMA_ORDER = Object.freeze({
    YUV: 'yuv',
    YVU: 'yvu',
  });

  const BINK_LOAD_OP = Object.freeze({
    CLEAR: 'clear',
    LOAD: 'load',
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
    [BINK_PIXEL_FORMAT.BGRA8]: 0,
    [BINK_PIXEL_FORMAT.YUV420P]: 1,
    [BINK_PIXEL_FORMAT.YUVA420P]: 2,
    [BINK_PIXEL_FORMAT.NV12]: 3,
  });

  function isLinearPresentationFormat(format) {
    if (typeof format !== 'string') {
      return false;
    }
    const normalized = format.trim().toLowerCase();
    return normalized.endsWith('-srgb') || normalized === 'rgba16float';
  }

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
    RENDER_ATTACHMENT:
      typeof GPUTextureUsage !== 'undefined' &&
      typeof GPUTextureUsage.RENDER_ATTACHMENT === 'number'
        ? GPUTextureUsage.RENDER_ATTACHMENT
        : 0x10,
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
  const EMPTY_OBJECT = Object.freeze({});

  function defaultNowMs() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function sanitizeNonNegativeNumber(value) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0;
  }

  function sanitizeNonNegativeInteger(value) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0
      ? Math.floor(value)
      : 0;
  }

  function clampUnit(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return value >= 1 ? 1 : value;
  }

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

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
  let clamped = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  let lo = clamped / 12.92;
  let hi = pow((clamped + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(hi, lo, clamped <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let uv = in.uv * uniforms.uv_transform.xy + uniforms.uv_transform.zw;
  let uv_corner0 = uniforms.uv_transform.zw;
  let uv_corner1 = uniforms.uv_transform.zw + uniforms.uv_transform.xy;
  let uv_lo = clamp(min(uv_corner0, uv_corner1), vec2<f32>(0.0), vec2<f32>(1.0));
  let uv_hi = clamp(max(uv_corner0, uv_corner1), vec2<f32>(0.0), vec2<f32>(1.0));

  let luma_dim = vec2<f32>(textureDimensions(plane_primary));
  let luma_half = vec2<f32>(0.5) / max(luma_dim, vec2<f32>(1.0));
  let luma_lo = min(uv_lo + luma_half, uv_hi);
  let luma_hi = max(uv_hi - luma_half, luma_lo);
  let clamped_luma_uv = clamp(uv, luma_lo, luma_hi);

  let chroma_dim = vec2<f32>(textureDimensions(plane_u));
  let chroma_scale = luma_dim / max(chroma_dim * 2.0, vec2<f32>(1.0));
  let chroma_half = vec2<f32>(0.5) / max(chroma_dim, vec2<f32>(1.0));
  let chroma_lo = min(uv_lo * chroma_scale + chroma_half, uv_hi * chroma_scale);
  let chroma_hi = max(uv_hi * chroma_scale - chroma_half, chroma_lo);
  let clamped_chroma_uv = select(
    clamped_luma_uv,
    clamp(uv * chroma_scale, chroma_lo, chroma_hi),
    uniforms.mode != 0u
  );

  let primary = textureSample(plane_primary, video_sampler, clamped_luma_uv);
  let uv_sample = textureSample(plane_u, video_sampler, clamped_chroma_uv);
  let v_sample = textureSample(plane_v, video_sampler, clamped_chroma_uv).r;
  let a_sample = textureSample(plane_a, video_sampler, clamped_luma_uv).r;

  let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
  if (!in_bounds) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  let output_linear = (uniforms.flags & 4u) != 0u;

  if (uniforms.mode == 0u) {
    let out_rgb = select(primary.rgb, srgb_to_linear(primary.rgb), output_linear);
    return vec4<f32>(out_rgb, primary.a * uniforms.opacity);
  }

  let raw_u = uv_sample.r;
  let raw_v = select(v_sample, uv_sample.g, uniforms.mode == 3u);
  let swap_chroma = (uniforms.flags & 2u) != 0u;
  let chroma_u = select(raw_u, raw_v, swap_chroma);
  let chroma_v = select(raw_v, raw_u, swap_chroma);
  let yuv = vec3<f32>(primary.r, chroma_u, chroma_v) - uniforms.yuv_offset.xyz;
  let srgb = clamp(
    vec3<f32>(
      dot(uniforms.yuv_row0.xyz, yuv),
      dot(uniforms.yuv_row1.xyz, yuv),
      dot(uniforms.yuv_row2.xyz, yuv)
    ),
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );
  let rgb = select(srgb, srgb_to_linear(srgb), output_linear);
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

  function validateChromaOrder(chromaOrder) {
    if (chromaOrder !== BINK_CHROMA_ORDER.YUV && chromaOrder !== BINK_CHROMA_ORDER.YVU) {
      throw new RangeError(
        `Unsupported chromaOrder "${chromaOrder}"; expected "${BINK_CHROMA_ORDER.YUV}" or "${BINK_CHROMA_ORDER.YVU}"`
      );
    }
  }

  function validateLoadOp(loadOp) {
    if (loadOp !== BINK_LOAD_OP.CLEAR && loadOp !== BINK_LOAD_OP.LOAD) {
      throw new RangeError(
        `Unsupported loadOp "${loadOp}"; expected "${BINK_LOAD_OP.CLEAR}" or "${BINK_LOAD_OP.LOAD}"`
      );
    }
  }

  function validateStoreOp(storeOp) {
    if (storeOp !== 'store' && storeOp !== 'discard') {
      throw new RangeError(`Unsupported storeOp "${storeOp}"; expected "store" or "discard"`);
    }
  }

  function validateOpacity(opacity) {
    if (typeof opacity !== 'number' || !Number.isFinite(opacity) || opacity < 0 || opacity > 1) {
      throw new RangeError(`opacity must be a finite number in [0, 1], got ${opacity}`);
    }
  }

  function validatePixelFormat(format) {
    if (
      format !== BINK_PIXEL_FORMAT.RGBA8 &&
      format !== BINK_PIXEL_FORMAT.BGRA8 &&
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

  function writeClearColor(clearColor, out) {
    if (!clearColor) {
      out.r = DEFAULT_CLEAR_COLOR.r;
      out.g = DEFAULT_CLEAR_COLOR.g;
      out.b = DEFAULT_CLEAR_COLOR.b;
      out.a = DEFAULT_CLEAR_COLOR.a;
      return out;
    }
    if (Array.isArray(clearColor) && clearColor.length >= 4) {
      out.r = clearColor[0];
      out.g = clearColor[1];
      out.b = clearColor[2];
      out.a = clearColor[3];
      return out;
    }
    out.r = typeof clearColor.r === 'number' ? clearColor.r : DEFAULT_CLEAR_COLOR.r;
    out.g = typeof clearColor.g === 'number' ? clearColor.g : DEFAULT_CLEAR_COLOR.g;
    out.b = typeof clearColor.b === 'number' ? clearColor.b : DEFAULT_CLEAR_COLOR.b;
    out.a = typeof clearColor.a === 'number' ? clearColor.a : DEFAULT_CLEAR_COLOR.a;
    return out;
  }

  /**
   * Compute a centered letterbox/pillarbox viewport inside `(dstWidth, dstHeight)` that
   * preserves the exact aspect ratio of `(srcWidth, srcHeight)`, writing into `out`.
   */
  function writeLetterboxViewport(srcWidth, srcHeight, dstWidth, dstHeight, out) {
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

    out.x = x;
    out.y = y;
    out.width = width;
    out.height = height;
    out.scaleX = width / dstWidth;
    out.scaleY = height / dstHeight;
    return out;
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
    return writeLetterboxViewport(srcWidth, srcHeight, dstWidth, dstHeight, {
      x: 0,
      y: 0,
      width: 0,
      height: 0,
      scaleX: 0,
      scaleY: 0,
    });
  }

  /**
   * Pack the 96-byte `PresenterUniforms` struct into a `Uint8Array` matching the exact
   * WGSL layout in `BINK_PRESENTER_WGSL`.
   *
   * @param {{
   *   pixelFormat?: string,
   *   colorSpace?: string,
   *   colorRange?: string,
   *   chromaOrder?: string,
   *   opacity?: number,
   *   flags?: number,
   *   uvTransform?: number[],
   * }} [params]
   * @returns {Uint8Array}
   */
  function writePresenterUniforms(params, f32, u32) {
    const pixelFormat = params.pixelFormat ?? BINK_PIXEL_FORMAT.RGBA8;
    const colorSpace = params.colorSpace ?? YUV_COLOR_SPACE.BT709;
    const colorRange = params.colorRange ?? YUV_COLOR_RANGE.LIMITED;
    const chromaOrder = params.chromaOrder ?? BINK_CHROMA_ORDER.YUV;
    const opacity = params.opacity ?? 1.0;
    const flags = params.flags ?? 0;
    const uvTransform = params.uvTransform ?? DEFAULT_UV_TRANSFORM;

    validatePixelFormat(pixelFormat);
    validateColorSpace(colorSpace);
    validateColorRange(colorRange);
    validateChromaOrder(chromaOrder);
    validateOpacity(opacity);

    const matrix = YUV_RANGE_MATRICES[colorRange][colorSpace];
    const outputLinear =
      params.outputLinear !== undefined
        ? Boolean(params.outputLinear)
        : isLinearPresentationFormat(params.presentationFormat);
    const packedFlags =
      ((flags >>> 0) |
        (colorRange === YUV_COLOR_RANGE.FULL ? UNIFORM_FLAG_FULL_RANGE : 0) |
        (chromaOrder === BINK_CHROMA_ORDER.YVU ? UNIFORM_FLAG_SWAP_CHROMA : 0) |
        (outputLinear ? UNIFORM_FLAG_OUTPUT_LINEAR : 0)) >>> 0;

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
  }

  function packPresenterUniforms(params = {}) {
    const buffer = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
    const u32 = new Uint32Array(buffer);
    const f32 = new Float32Array(buffer);
    writePresenterUniforms(params, f32, u32);
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

  function isSharedArrayBufferView(bytes) {
    return Boolean(
      typeof SharedArrayBuffer !== 'undefined' &&
        bytes &&
        bytes.buffer &&
        (bytes.buffer instanceof SharedArrayBuffer ||
          Object.prototype.toString.call(bytes.buffer) === '[object SharedArrayBuffer]')
    );
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
    scratchBuffer = null,
    out = null
  ) {
    validatePlaneBuffer(src, tightRowBytes, height, srcStride, label);

    const isSrcStrideAligned =
      srcStride % WEBGPU_BYTES_PER_ROW_ALIGNMENT === 0 && srcStride >= tightRowBytes;
    const directTotalBytes = srcStride * height;

    if (isSrcStrideAligned && src.byteLength >= directTotalBytes) {
      const slice = src.byteLength === directTotalBytes ? src : src.subarray(0, directTotalBytes);
      if (out) {
        out.data = slice;
        out.bytesPerRow = srcStride;
        out.totalBytes = directTotalBytes;
        out.padded = false;
        out.usesScratch = false;
        return out;
      }
      return {
        data: slice,
        bytesPerRow: srcStride,
        totalBytes: directTotalBytes,
        padded: false,
        usesScratch: false,
      };
    }

    const alignedBytesPerRow = alignUp(tightRowBytes, WEBGPU_BYTES_PER_ROW_ALIGNMENT);
    const totalBytes = alignedBytesPerRow * height;

    const canReuseScratch =
      scratchBuffer instanceof Uint8Array && scratchBuffer.byteLength >= totalBytes;
    const dst = canReuseScratch
      ? scratchBuffer.byteLength === totalBytes
        ? scratchBuffer
        : scratchBuffer.subarray(0, totalBytes)
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

    if (out) {
      out.data = dst;
      out.bytesPerRow = alignedBytesPerRow;
      out.totalBytes = totalBytes;
      out.padded = true;
      out.usesScratch = canReuseScratch;
      return out;
    }

    return {
      data: dst,
      bytesPerRow: alignedBytesPerRow,
      totalBytes,
      padded: true,
      usesScratch: canReuseScratch,
    };
  }

  function createEmptySlot(slotIndex = 0) {
    return {
      format: null,
      width: 0,
      height: 0,
      uvWidth: 0,
      uvHeight: 0,
      generation: 0,
      rgba: null,
      y: null,
      u: null,
      v: null,
      uv: null,
      a: null,
      bindGroup: null,
      bindGroupGeneration: 0,
      bindGroupFormat: null,
      bindGroupWidth: 0,
      bindGroupHeight: 0,
      bindGroupChromaOrder: null,
      bindGroupLabel: `playgta6-bink-presenter-bg-slot${slotIndex}`,
    };
  }

  function destroyPlaneResource(plane) {
    if (
      plane &&
      !plane.external &&
      plane.texture &&
      typeof plane.texture.destroy === 'function'
    ) {
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
    slot.bindGroupGeneration = 0;
    slot.bindGroupFormat = null;
    slot.bindGroupWidth = 0;
    slot.bindGroupHeight = 0;
    slot.bindGroupChromaOrder = null;
    slot.generation = 0;
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
        chromaOrder = BINK_CHROMA_ORDER.YUV,
        loadOp = BINK_LOAD_OP.CLEAR,
        targetFps = DEFAULT_TARGET_FPS,
        doubleBuffer = true,
        enableTripleBuffering = false,
        allowUnalignedDirectUpload = false,
        planeRingDepth,
        slotCount,
        bindGroupCacheCapacity = DEFAULT_BIND_GROUP_CACHE_CAPACITY,
        now = defaultNowMs,
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
      if (typeof now !== 'function') {
        throw new TypeError('now must be a function returning a finite timestamp in milliseconds');
      }
      validateColorSpace(colorSpace);
      validateColorRange(colorRange);
      validateChromaOrder(chromaOrder);
      validateLoadOp(loadOp);
      assertPositiveFinite(targetFps, 'targetFps');

      if (planeRingDepth !== undefined) {
        assertPositiveInteger(planeRingDepth, 'planeRingDepth');
      }
      if (slotCount !== undefined) {
        assertPositiveInteger(slotCount, 'slotCount');
      }
      if (
        planeRingDepth !== undefined &&
        slotCount !== undefined &&
        planeRingDepth !== slotCount
      ) {
        throw new RangeError(
          `Conflicting planeRingDepth (${planeRingDepth}) and slotCount (${slotCount})`
        );
      }

      const resolvedSlotCount =
        planeRingDepth !== undefined
          ? planeRingDepth
          : enableTripleBuffering
            ? TRIPLE_BUFFER_SLOTS
            : slotCount !== undefined
              ? slotCount
              : doubleBuffer
                ? DOUBLE_BUFFER_SLOTS
                : 1;
      assertPositiveInteger(resolvedSlotCount, 'slotCount');
      assertPositiveInteger(bindGroupCacheCapacity, 'bindGroupCacheCapacity');

      this._device = device;
      this._presentationFormat = presentationFormat;
      this._stagingPool = stagingPool || null;
      this._allowUnalignedDirectUpload = Boolean(allowUnalignedDirectUpload);
      this._colorSpace = colorSpace;
      this._colorRange = colorRange;
      this._chromaOrder = chromaOrder;
      this._loadOp = loadOp;
      this._targetFps = targetFps;
      this._now = now;

      // Reusable plane texture ring slots (double-buffered by default; opt-in triple-buffering
      // via enableTripleBuffering: true or planeRingDepth: 3).
      this._planeRingDepth = resolvedSlotCount;
      this._slots = [];
      for (let i = 0; i < resolvedSlotCount; i += 1) {
        this._slots.push(createEmptySlot(i));
      }
      this._activeSlotIndex = 0;
      this._activePlaneSlot = 0;
      this._hasFrame = false;
      this._unpresentedFrame = false;

      // Current video stream state.
      this._pixelFormat = null;
      this._frameWidth = 0;
      this._frameHeight = 0;
      this._uvTransform = [...DEFAULT_UV_TRANSFORM];

      // Deferred staging uploads queued when upload*Frame is called without a commandEncoder.
      this._pendingUploads = [];
      this._pendingUploadsDrain = [];

      // Reusable scratch buffer for 256-byte row-pitch alignment.
      this._uploadScratch = null;
      this._scratchAllocations = 0;

      // Bounded LRU GPUBindGroup cache keyed by (yView, uView, vView, aView, sampler, uniformBuffer)
      // object identity + slot generation/layout/device/chromaOrder.
      this._bindGroupCacheCapacity = bindGroupCacheCapacity;
      this._bindGroupCache = [];
      this._bindGroupCacheDevice = device;
      this._pipelineDevice = null;
      this._bindGroupUniformBufferRef = null;
      this._bindGroupUniformResource = null;
      this._bindGroupEntriesScratch = [
        { binding: 0, resource: null },
        { binding: 1, resource: null },
        { binding: 2, resource: null },
        { binding: 3, resource: null },
        { binding: 4, resource: null },
        { binding: 5, resource: null },
      ];
      this._bindGroupDescScratch = {
        label: 'playgta6-bink-presenter-bg',
        layout: null,
        entries: this._bindGroupEntriesScratch,
      };

      // Pre-allocated upload & present descriptor/result scratch objects for zero-alloc hot paths.
      this._rgbaUploadOptionsScratch = {
        strideBytes: undefined,
        commandEncoder: null,
      };
      this._rgbaUploadResultScratch = {
        format: BINK_PIXEL_FORMAT.RGBA8,
        width: 0,
        height: 0,
        bytesPerRow: 0,
        bytesUploaded: 0,
        zeroCopy: false,
        slotIndex: 0,
        uploadedFrames: 0,
        uploadMs: 0,
      };
      this._yuvUploadResultScratch = {
        format: BINK_PIXEL_FORMAT.YUV420P,
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        chromaOrder: this._chromaOrder,
        width: 0,
        height: 0,
        uvWidth: 0,
        uvHeight: 0,
        yBytesPerRow: 0,
        uvBytesPerRow: 0,
        aBytesPerRow: null,
        yZeroCopy: false,
        chromaZeroCopy: false,
        bytesUploaded: 0,
        slotIndex: 0,
        uploadedFrames: 0,
        uploadMs: 0,
      };
      this._nv12UploadResultScratch = {
        format: BINK_PIXEL_FORMAT.NV12,
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        chromaOrder: this._chromaOrder,
        width: 0,
        height: 0,
        uvWidth: 0,
        uvHeight: 0,
        yBytesPerRow: 0,
        uvBytesPerRow: 0,
        yZeroCopy: false,
        chromaZeroCopy: false,
        bytesUploaded: 0,
        slotIndex: 0,
        uploadedFrames: 0,
        uploadMs: 0,
      };
      this._preparedPlaneScratch = {
        data: null,
        bytesPerRow: 0,
        totalBytes: 0,
        padded: false,
        usesScratch: false,
      };
      this._uploadOriginScratch = { x: 0, y: 0, z: 0 };
      this._uploadDestinationScratch = {
        texture: null,
        mipLevel: 0,
        origin: this._uploadOriginScratch,
      };
      this._uploadDataLayoutScratch = {
        offset: 0,
        bytesPerRow: 0,
        rowsPerImage: 0,
      };
      this._uploadSizeScratch = {
        width: 0,
        height: 0,
        depthOrArrayLayers: 1,
      };
      this._stagingLayoutScratch = {
        width: 0,
        height: 0,
        bytesPerRow: 0,
        rowsPerImage: 0,
      };
      this._uniformParamsScratch = {
        pixelFormat: BINK_PIXEL_FORMAT.RGBA8,
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        chromaOrder: this._chromaOrder,
        opacity: 1.0,
        uvTransform: this._uvTransform,
      };
      this._clearColorValueScratch = {
        r: DEFAULT_CLEAR_COLOR.r,
        g: DEFAULT_CLEAR_COLOR.g,
        b: DEFAULT_CLEAR_COLOR.b,
        a: DEFAULT_CLEAR_COLOR.a,
      };
      this._clearColorAttachmentScratch = {
        view: null,
        loadOp: BINK_LOAD_OP.CLEAR,
        storeOp: 'store',
        clearValue: this._clearColorValueScratch,
      };
      this._clearRenderPassDescScratch = {
        label: 'playgta6-bink-present-pass',
        colorAttachments: [this._clearColorAttachmentScratch],
      };
      this._loadColorAttachmentScratch = {
        view: null,
        loadOp: BINK_LOAD_OP.LOAD,
        storeOp: 'store',
      };
      this._loadRenderPassDescScratch = {
        label: 'playgta6-bink-present-pass',
        colorAttachments: [this._loadColorAttachmentScratch],
      };
      this._viewportScratch = {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
        scaleX: 0,
        scaleY: 0,
      };
      this._presentResultScratch = {
        presented: false,
        viewport: null,
        format: null,
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        chromaOrder: this._chromaOrder,
        loadOp: this._loadOp,
        slotIndex: 0,
        presentedFrames: 0,
        framesPresented: 0,
      };

      // Lazily constructed GPU pipeline resources.
      this._shaderModule = null;
      this._bindGroupLayout = null;
      this._pipelineLayout = null;
      this._pipeline = null;
      this._sampler = null;
      this._uniformBuffer = null;
      this._dummyTexture = null;
      this._dummyTextureView = null;
      this._textureGeneration = 0;

      // Pre-allocated uniform scratch buffer (96 bytes = 24 x f32/u32 lanes) reused across frames.
      this._uniformScratchF32 = new Float32Array(UNIFORM_FLOAT_COUNT);
      this._uniformScratchU32 = new Uint32Array(
        this._uniformScratchF32.buffer,
        this._uniformScratchF32.byteOffset,
        UNIFORM_FLOAT_COUNT
      );
      this._uniformLastUploadedU32 = new Uint32Array(UNIFORM_FLOAT_COUNT);
      this._hasUploadedUniforms = false;
      this._uniformsDirty = true;
      this._lastUniformOpacity = null;

      // Playback pacing & control state.
      this._paused = false;
      this._looping = false;
      this._skipped = false;
      this._lastAdvanceMs = null;
      this._lastPresentMs = null;

      // Telemetry.
      this._uploadedFrames = 0;
      this._presentedFrames = 0;
      this._droppedFrames = 0;
      this._resizeCount = 0;
      this._planeReallocations = 0;
      this._bytesUploaded = 0;
      this._directZeroCopyUploads = 0;
      this._chromaRepackUploads = 0;
      this._bindGroupCacheHits = 0;
      this._bindGroupCacheMisses = 0;
      this._uniformWritesSkipped = 0;
      this._uploadTimingCount = 0;
      this._totalUploadMs = 0;
      this._maxUploadMs = 0;
      this._lastUploadMs = 0;
      this._lastFrameIntervalMs = 0;
      this._measuredFrameIntervals = 0;
      this._targetFps30Hits = 0;
      this._planeSlotRotations = 0;
      this._reusedUploadDescriptors = 0;
      this._framesOver30FpsBudget = 0;
      this._peakPresentIntervalMs = 0;
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

    get chromaOrder() {
      return this._chromaOrder;
    }

    get loadOp() {
      return this._loadOp;
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

    get framesPresented() {
      return this._presentedFrames;
    }

    get presentedFrames() {
      return this._presentedFrames;
    }

    get droppedFrames() {
      return this._droppedFrames;
    }

    get avgUploadMs() {
      return this._uploadTimingCount > 0
        ? sanitizeNonNegativeNumber(this._totalUploadMs / this._uploadTimingCount)
        : 0;
    }

    get maxUploadMs() {
      return sanitizeNonNegativeNumber(this._maxUploadMs);
    }

    get lastFrameIntervalMs() {
      return sanitizeNonNegativeNumber(this._lastFrameIntervalMs);
    }

    get bindGroupCacheHits() {
      return this._bindGroupCacheHits;
    }

    get bindGroupCacheMisses() {
      return this._bindGroupCacheMisses;
    }

    get bindGroupHitRate() {
      const total = this._bindGroupCacheHits + this._bindGroupCacheMisses;
      return total > 0 ? clampUnit(this._bindGroupCacheHits / total) : 0;
    }

    get bindGroupCacheSize() {
      return this._bindGroupCache.length;
    }

    get bindGroupCacheCapacity() {
      return this._bindGroupCacheCapacity;
    }

    get targetFpsHitRate30() {
      return this._measuredFrameIntervals > 0
        ? clampUnit(this._targetFps30Hits / this._measuredFrameIntervals)
        : 0;
    }

    get uniformWritesSkipped() {
      return this._uniformWritesSkipped;
    }

    get planeRingDepth() {
      return this._planeRingDepth;
    }

    get activePlaneSlot() {
      return this._activePlaneSlot;
    }

    get planeSlotRotations() {
      return sanitizeNonNegativeInteger(this._planeSlotRotations);
    }

    get reusedUploadDescriptors() {
      return sanitizeNonNegativeInteger(this._reusedUploadDescriptors);
    }

    get framesOver30FpsBudget() {
      return sanitizeNonNegativeInteger(this._framesOver30FpsBudget);
    }

    get peakPresentIntervalMs() {
      return sanitizeNonNegativeNumber(this._peakPresentIntervalMs);
    }

    get allowUnalignedDirectUpload() {
      return this._allowUnalignedDirectUpload;
    }

    _recordUploadDuration(startMs, endMs) {
      const raw = endMs - startMs;
      const durationMs = Number.isFinite(raw) && raw > 0 ? raw : 0;
      this._uploadTimingCount += 1;
      this._totalUploadMs += durationMs;
      this._lastUploadMs = durationMs;
      if (durationMs > this._maxUploadMs) {
        this._maxUploadMs = durationMs;
      }
      return durationMs;
    }

    _assertNotDestroyed() {
      if (this._destroyed) {
        throw new Error('BinkVideoPresenter has been destroyed');
      }
    }

    _createPlaneTexture(width, height, format, label, extraUsage = 0) {
      const usage = TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST | extraUsage;
      const texture = this._device.createTexture({
        label,
        size: { width, height, depthOrArrayLayers: 1 },
        format,
        usage,
      });
      const view =
        texture && typeof texture.createView === 'function' ? texture.createView() : null;
      return { texture, view, width, height, format, usage };
    }

    _destroyAllSlots() {
      for (let i = 0; i < this._slots.length; i += 1) {
        destroySlotResources(this._slots[i]);
      }
      this._pendingUploads.length = 0;
      this._pendingUploadsDrain.length = 0;
      this._uploadDestinationScratch.texture = null;
      this._invalidateBindGroupCache();
    }

    _invalidateBindGroupCache() {
      this._bindGroupCache.length = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        const slot = this._slots[i];
        slot.bindGroup = null;
        slot.bindGroupGeneration = 0;
        slot.bindGroupFormat = null;
        slot.bindGroupWidth = 0;
        slot.bindGroupHeight = 0;
        slot.bindGroupChromaOrder = null;
      }
    }

    _releasePipelineResources() {
      if (this._dummyTexture && typeof this._dummyTexture.destroy === 'function') {
        this._dummyTexture.destroy();
      }
      this._dummyTexture = null;
      this._dummyTextureView = null;

      if (this._uniformBuffer && typeof this._uniformBuffer.destroy === 'function') {
        this._uniformBuffer.destroy();
      }
      this._uniformBuffer = null;
      this._bindGroupUniformBufferRef = null;
      this._bindGroupUniformResource = null;
      this._bindGroupDescScratch.layout = null;
      for (let i = 0; i < this._bindGroupEntriesScratch.length; i += 1) {
        this._bindGroupEntriesScratch[i].resource = null;
      }
      this._hasUploadedUniforms = false;
      this._uniformsDirty = true;
      this._lastUniformOpacity = null;

      this._shaderModule = null;
      this._bindGroupLayout = null;
      this._pipelineLayout = null;
      this._pipeline = null;
      this._sampler = null;
      this._pipelineDevice = null;
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
        this._activeSlotIndex = 0;
        this._activePlaneSlot = 0;
        this._resizeCount += 1;
        this._planeReallocations += 1;
        if (this._pixelFormat !== format) {
          this._uniformsDirty = true;
        }
        return 0;
      }

      if (!this._hasFrame) {
        this._planeReallocations += 1;
        return 0;
      }

      return (this._activePlaneSlot + 1) % this._planeRingDepth;
    }

    _ensureSlotTextures(slotIndex, format, width, height, requiredExtraUsage = 0) {
      if (this._bindGroupCacheDevice !== this._device) {
        this._destroyAllSlots();
        this._releasePipelineResources();
        this._bindGroupCacheDevice = this._device;
      }
      const slot = this._slots[slotIndex];
      const uvWidth = Math.ceil(width / 2);
      const uvHeight = Math.ceil(height / 2);
      const isPackedRgb =
        format === BINK_PIXEL_FORMAT.RGBA8 || format === BINK_PIXEL_FORMAT.BGRA8;
      const rgbaHasRequiredUsage =
        slot.rgba !== null &&
        !slot.rgba.external &&
        ((slot.rgba.usage || 0) & requiredExtraUsage) === requiredExtraUsage;

      const matches =
        slot.format === format &&
        slot.width === width &&
        slot.height === height &&
        (isPackedRgb
          ? rgbaHasRequiredUsage
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

      if (isPackedRgb) {
        const texFormat = format === BINK_PIXEL_FORMAT.BGRA8 ? 'bgra8unorm' : 'rgba8unorm';
        slot.rgba = this._createPlaneTexture(
          width,
          height,
          texFormat,
          `playgta6-bink-${format}-slot${slotIndex}`,
          requiredExtraUsage
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

      this._textureGeneration += 1;
      slot.generation = this._textureGeneration;
      slot.format = format;
      slot.width = width;
      slot.height = height;
      slot.uvWidth = uvWidth;
      slot.uvHeight = uvHeight;
      slot.bindGroup = null;
      slot.bindGroupGeneration = 0;
      slot.bindGroupFormat = null;
      slot.bindGroupWidth = 0;
      slot.bindGroupHeight = 0;
      slot.bindGroupChromaOrder = null;
      return slot;
    }

    _ensureUploadScratch(totalBytes) {
      if (!this._uploadScratch || this._uploadScratch.byteLength < totalBytes) {
        this._uploadScratch = new Uint8Array(totalBytes);
        this._scratchAllocations += 1;
      }
      return this._uploadScratch.byteLength === totalBytes
        ? this._uploadScratch
        : this._uploadScratch.subarray(0, totalBytes);
    }

    _preparePlaneForUpload(src, tightRowBytes, height, srcStride, label) {
      validatePlaneBuffer(src, tightRowBytes, height, srcStride, label);

      const isSharedSrc = isSharedArrayBufferView(src);

      // WebGPU `GPUQueue.writeTexture` does NOT require 256-byte `bytesPerRow` alignment —
      // only `copyBufferToTexture` (stagingPool) does. When `allowUnalignedDirectUpload` is
      // enabled and we are uploading via `queue.writeTexture`, skip the per-row CPU repack
      // completely (zero-copy for plain ArrayBuffers, single bulk `TypedArray.set()` into the
      // persistent scratch buffer for SharedArrayBuffers).
      if (!this._stagingPool && this._allowUnalignedDirectUpload) {
        const minRequiredBytes =
          height > 1 ? srcStride * (height - 1) + tightRowBytes : tightRowBytes;
        const fullStrideBytes = srcStride * height;
        const directTotalBytes =
          src.byteLength >= fullStrideBytes ? fullStrideBytes : minRequiredBytes;
        const srcSlice =
          src.byteLength === directTotalBytes ? src : src.subarray(0, directTotalBytes);
        const out = this._preparedPlaneScratch;
        if (isSharedSrc) {
          const dst = this._ensureUploadScratch(directTotalBytes);
          dst.set(srcSlice, 0);
          out.data = dst;
          out.bytesPerRow = srcStride;
          out.totalBytes = directTotalBytes;
          out.padded = false;
          out.usesScratch = true;
          return out;
        }
        out.data = srcSlice;
        out.bytesPerRow = srcStride;
        out.totalBytes = directTotalBytes;
        out.padded = false;
        out.usesScratch = false;
        return out;
      }

      const isSrcStrideAligned =
        srcStride % WEBGPU_BYTES_PER_ROW_ALIGNMENT === 0 && srcStride >= tightRowBytes;
      const directTotalBytes = srcStride * height;

      if (isSrcStrideAligned && src.byteLength >= directTotalBytes) {
        if (!this._stagingPool && isSharedSrc) {
          const srcSlice =
            src.byteLength === directTotalBytes ? src : src.subarray(0, directTotalBytes);
          const dst = this._ensureUploadScratch(directTotalBytes);
          dst.set(srcSlice, 0);
          const out = this._preparedPlaneScratch;
          out.data = dst;
          out.bytesPerRow = srcStride;
          out.totalBytes = directTotalBytes;
          out.padded = false;
          out.usesScratch = true;
          return out;
        }
        return prepareAlignedPlaneData(
          src,
          tightRowBytes,
          height,
          srcStride,
          label,
          null,
          this._preparedPlaneScratch
        );
      }

      const alignedBytesPerRow = alignUp(tightRowBytes, WEBGPU_BYTES_PER_ROW_ALIGNMENT);
      const totalBytes = alignedBytesPerRow * height;
      this._ensureUploadScratch(totalBytes);

      return prepareAlignedPlaneData(
        src,
        tightRowBytes,
        height,
        srcStride,
        label,
        this._uploadScratch,
        this._preparedPlaneScratch
      );
    }

    _writeTextureWithScratch(dstTexture, data, bytesPerRow, rowsPerImage, width, height) {
      this._uploadDestinationScratch.texture = dstTexture;
      this._uploadDestinationScratch.mipLevel = 0;
      this._uploadOriginScratch.x = 0;
      this._uploadOriginScratch.y = 0;
      this._uploadOriginScratch.z = 0;

      this._uploadDataLayoutScratch.offset = 0;
      this._uploadDataLayoutScratch.bytesPerRow = bytesPerRow;
      this._uploadDataLayoutScratch.rowsPerImage = rowsPerImage;

      this._uploadSizeScratch.width = width;
      this._uploadSizeScratch.height = height;
      this._uploadSizeScratch.depthOrArrayLayers = 1;

      this._device.queue.writeTexture(
        this._uploadDestinationScratch,
        data,
        this._uploadDataLayoutScratch,
        this._uploadSizeScratch
      );
    }

    _uploadPlane(slotIndex, dstTexture, prepared, width, height, commandEncoder, isChroma = false) {
      if (this._stagingPool) {
        if (commandEncoder) {
          this._stagingLayoutScratch.width = width;
          this._stagingLayoutScratch.height = height;
          this._stagingLayoutScratch.bytesPerRow = prepared.bytesPerRow;
          this._stagingLayoutScratch.rowsPerImage = height;
          this._stagingPool.stageTextureUpload(
            commandEncoder,
            dstTexture,
            prepared.data,
            this._stagingLayoutScratch
          );
          this._reusedUploadDescriptors += 1;
        } else {
          const layout = {
            width,
            height,
            bytesPerRow: prepared.bytesPerRow,
            rowsPerImage: height,
          };
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
        this._writeTextureWithScratch(
          dstTexture,
          prepared.data,
          prepared.bytesPerRow,
          height,
          width,
          height
        );
        this._reusedUploadDescriptors += 1;
      }

      this._bytesUploaded += prepared.totalBytes;
      if (!prepared.padded) {
        this._directZeroCopyUploads += 1;
      } else if (isChroma) {
        this._chromaRepackUploads += 1;
      }
    }

    _dropPendingUploadsForSlot(slotIndex) {
      const pending = this._pendingUploads;
      if (pending.length === 0) {
        return;
      }
      let writeIdx = 0;
      for (let i = 0; i < pending.length; i += 1) {
        if (pending[i].slotIndex !== slotIndex) {
          pending[writeIdx] = pending[i];
          writeIdx += 1;
        }
      }
      pending.length = writeIdx;
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

      const uploadStartMs = this._now();
      const pending = this._pendingUploads;
      this._pendingUploads = this._pendingUploadsDrain;
      this._pendingUploads.length = 0;
      this._pendingUploadsDrain = pending;
      const count = pending.length;
      for (let i = 0; i < count; i += 1) {
        const item = pending[i];
        this._stagingPool.stageTextureUpload(
          commandEncoder,
          item.texture,
          item.data,
          item.layout
        );
      }
      pending.length = 0;
      const flushDurationRaw = this._now() - uploadStartMs;
      const flushMs =
        Number.isFinite(flushDurationRaw) && flushDurationRaw > 0 ? flushDurationRaw : 0;
      if (flushMs > 0 && this._uploadTimingCount > 0) {
        this._totalUploadMs += flushMs;
        this._lastUploadMs += flushMs;
        if (this._lastUploadMs > this._maxUploadMs) {
          this._maxUploadMs = this._lastUploadMs;
        }
      }
      return count;
    }

    /**
     * Upload an RGBA8 video or splash frame into the next double-buffered texture slot.
     *
     * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} rgbaBytes
     * @param {number} width
     * @param {number} height
     * @param {{ strideBytes?: number, commandEncoder?: GPUCommandEncoder, encoder?: GPUCommandEncoder }} [options]
     */
    uploadRgbaFrame(rgbaBytes, width, height, options = null) {
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

      const pixelFormat =
        options && options.format === BINK_PIXEL_FORMAT.BGRA8
          ? BINK_PIXEL_FORMAT.BGRA8
          : BINK_PIXEL_FORMAT.RGBA8;
      const label = pixelFormat === BINK_PIXEL_FORMAT.BGRA8 ? 'BGRA8' : 'RGBA8';
      validatePlaneBuffer(src, tightRowBytes, height, strideBytes, label);

      const uploadStartMs = this._now();

      if (this._unpresentedFrame) {
        this._droppedFrames += 1;
      }

      const writeSlotIndex = this._selectWriteSlot(pixelFormat, width, height);
      const slot = this._ensureSlotTextures(writeSlotIndex, pixelFormat, width, height);
      this._dropPendingUploadsForSlot(writeSlotIndex);

      const prepared = this._preparePlaneForUpload(src, tightRowBytes, height, strideBytes, label);
      const bytesPerRow = prepared.bytesPerRow;
      const totalBytes = prepared.totalBytes;
      const padded = prepared.padded;
      this._uploadPlane(writeSlotIndex, slot.rgba.texture, prepared, width, height, commandEncoder);

      const uploadMs = this._recordUploadDuration(uploadStartMs, this._now());

      const rotatedSlot = this._hasFrame && writeSlotIndex !== this._activePlaneSlot;
      if (this._pixelFormat !== pixelFormat) {
        this._uniformsDirty = true;
      }
      this._pixelFormat = pixelFormat;
      this._frameWidth = width;
      this._frameHeight = height;
      this._activeSlotIndex = writeSlotIndex;
      this._activePlaneSlot = writeSlotIndex;
      if (rotatedSlot) {
        this._planeSlotRotations += 1;
      }
      this._hasFrame = true;
      this._unpresentedFrame = true;
      this._uploadedFrames += 1;

      const res = this._rgbaUploadResultScratch;
      res.format = pixelFormat;
      res.width = width;
      res.height = height;
      res.bytesPerRow = bytesPerRow;
      res.bytesUploaded = totalBytes;
      res.zeroCopy = !padded;
      res.slotIndex = writeSlotIndex;
      res.uploadedFrames = this._uploadedFrames;
      res.uploadMs = uploadMs;
      return res;
    }

    /**
     * Upload a BGRA8 video or splash frame without swapping R/B channels (uses native `bgra8unorm`).
     */
    uploadBgraFrame(bgraBytes, width, height, options = null) {
      const opts = options && typeof options === 'object' ? { ...options, format: BINK_PIXEL_FORMAT.BGRA8 } : { format: BINK_PIXEL_FORMAT.BGRA8 };
      return this.uploadRgbaFrame(bgraBytes, width, height, opts);
    }

    /**
     * Bind a caller-owned external RGBA/BGRA GPUTexture (e.g. destination of
     * `queue.copyExternalImageToTexture`) directly into the active slot without taking
     * destroy ownership, so GPU-only external images render real pixels without CPU readback.
     */
    bindExternalTexture(texture, width, height, format = BINK_PIXEL_FORMAT.RGBA8) {
      this._assertNotDestroyed();
      if (!texture || typeof texture !== 'object') {
        throw new TypeError('bindExternalTexture expects a GPUTexture object');
      }
      assertPositiveInteger(width, 'width');
      assertPositiveInteger(height, 'height');
      const pixelFormat =
        format === BINK_PIXEL_FORMAT.BGRA8 ? BINK_PIXEL_FORMAT.BGRA8 : BINK_PIXEL_FORMAT.RGBA8;

      const uploadStartMs = this._now();
      if (this._unpresentedFrame) {
        this._droppedFrames += 1;
      }

      const writeSlotIndex = this._selectWriteSlot(pixelFormat, width, height);
      const slot = this._slots[writeSlotIndex];
      destroySlotResources(slot);
      this._dropPendingUploadsForSlot(writeSlotIndex);

      const view =
        typeof texture.createView === 'function' ? texture.createView() : null;
      slot.rgba = {
        texture,
        view,
        width,
        height,
        format: pixelFormat === BINK_PIXEL_FORMAT.BGRA8 ? 'bgra8unorm' : 'rgba8unorm',
        usage: texture.usage || 0,
        external: true,
      };
      this._textureGeneration += 1;
      slot.generation = this._textureGeneration;
      slot.format = pixelFormat;
      slot.width = width;
      slot.height = height;
      slot.uvWidth = Math.ceil(width / 2);
      slot.uvHeight = Math.ceil(height / 2);
      slot.bindGroup = null;
      slot.bindGroupGeneration = 0;

      this._bytesUploaded += width * height * 4;
      this._directZeroCopyUploads += 1;

      const uploadMs = this._recordUploadDuration(uploadStartMs, this._now());
      const rotatedSlot = this._hasFrame && writeSlotIndex !== this._activePlaneSlot;
      if (this._pixelFormat !== pixelFormat) {
        this._uniformsDirty = true;
      }
      this._pixelFormat = pixelFormat;
      this._frameWidth = width;
      this._frameHeight = height;
      this._activeSlotIndex = writeSlotIndex;
      this._activePlaneSlot = writeSlotIndex;
      if (rotatedSlot) {
        this._planeSlotRotations += 1;
      }
      this._hasFrame = true;
      this._unpresentedFrame = true;
      this._uploadedFrames += 1;

      const res = this._rgbaUploadResultScratch;
      res.format = pixelFormat;
      res.width = width;
      res.height = height;
      res.bytesPerRow = width * 4;
      res.bytesUploaded = width * height * 4;
      res.zeroCopy = true;
      res.slotIndex = writeSlotIndex;
      res.uploadedFrames = this._uploadedFrames;
      res.uploadMs = uploadMs;
      return res;
    }

    /**
     * Upload an external GPU-copyable image source (`ImageBitmap`, `OffscreenCanvas`, `HTMLVideoElement`)
     * directly into the active RGBA slot via `queue.copyExternalImageToTexture` (with `RENDER_ATTACHMENT`
     * usage as required by WebGPU).
     */
    uploadExternalImage(source, width, height) {
      this._assertNotDestroyed();
      if (!source || typeof source !== 'object') {
        throw new TypeError('uploadExternalImage expects an external image source object');
      }
      assertPositiveInteger(width, 'width');
      assertPositiveInteger(height, 'height');
      if (
        !this._device.queue ||
        typeof this._device.queue.copyExternalImageToTexture !== 'function'
      ) {
        throw new TypeError(
          'uploadExternalImage requires device.queue.copyExternalImageToTexture()'
        );
      }

      const uploadStartMs = this._now();
      if (this._unpresentedFrame) {
        this._droppedFrames += 1;
      }

      const writeSlotIndex = this._selectWriteSlot(BINK_PIXEL_FORMAT.RGBA8, width, height);
      const slot = this._ensureSlotTextures(
        writeSlotIndex,
        BINK_PIXEL_FORMAT.RGBA8,
        width,
        height,
        TEXTURE_USAGE.RENDER_ATTACHMENT
      );
      this._dropPendingUploadsForSlot(writeSlotIndex);

      this._device.queue.copyExternalImageToTexture(
        { source },
        { texture: slot.rgba.texture },
        { width, height, depthOrArrayLayers: 1 }
      );
      this._bytesUploaded += width * height * 4;
      this._directZeroCopyUploads += 1;

      const uploadMs = this._recordUploadDuration(uploadStartMs, this._now());
      const rotatedSlot = this._hasFrame && writeSlotIndex !== this._activePlaneSlot;
      if (this._pixelFormat !== BINK_PIXEL_FORMAT.RGBA8) {
        this._uniformsDirty = true;
      }
      this._pixelFormat = BINK_PIXEL_FORMAT.RGBA8;
      this._frameWidth = width;
      this._frameHeight = height;
      this._activeSlotIndex = writeSlotIndex;
      this._activePlaneSlot = writeSlotIndex;
      if (rotatedSlot) {
        this._planeSlotRotations += 1;
      }
      this._hasFrame = true;
      this._unpresentedFrame = true;
      this._uploadedFrames += 1;

      const res = this._rgbaUploadResultScratch;
      res.format = BINK_PIXEL_FORMAT.RGBA8;
      res.width = width;
      res.height = height;
      res.bytesPerRow = width * 4;
      res.bytesUploaded = width * height * 4;
      res.zeroCopy = true;
      res.slotIndex = writeSlotIndex;
      res.uploadedFrames = this._uploadedFrames;
      res.uploadMs = uploadMs;
      return res;
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
     *   uStride?: number,
     *   vStride?: number,
     *   aStride?: number,
     *   colorSpace?: string,
     *   colorRange?: string,
     *   chromaOrder?: string,
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
        uStride,
        vStride,
        aStride,
        colorSpace,
        colorRange,
        chromaOrder,
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
      if (chromaOrder !== undefined && chromaOrder !== null) {
        validateChromaOrder(chromaOrder);
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
      const resolvedUStride =
        uStride !== undefined ? uStride : uvStride !== undefined ? uvStride : uvWidth;
      const resolvedVStride =
        vStride !== undefined ? vStride : uvStride !== undefined ? uvStride : uvWidth;
      const resolvedAStride = aStride !== undefined ? aStride : width;

      validatePlaneBuffer(ySrc, width, height, resolvedYStride, 'Y plane');
      validatePlaneBuffer(uSrc, uvWidth, uvHeight, resolvedUStride, 'U plane');
      validatePlaneBuffer(vSrc, uvWidth, uvHeight, resolvedVStride, 'V plane');
      if (hasAlpha && aSrc) {
        validatePlaneBuffer(aSrc, width, height, resolvedAStride, 'A plane');
      }

      const uploadStartMs = this._now();

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
      const yBytesPerRow = preparedY.bytesPerRow;
      const yTotalBytes = preparedY.totalBytes;
      const yPadded = preparedY.padded;
      this._uploadPlane(
        writeSlotIndex,
        slot.y.texture,
        preparedY,
        width,
        height,
        commandEncoder,
        false
      );

      const preparedU = this._preparePlaneForUpload(
        uSrc,
        uvWidth,
        uvHeight,
        resolvedUStride,
        'U plane'
      );
      const uBytesPerRow = preparedU.bytesPerRow;
      const uTotalBytes = preparedU.totalBytes;
      const uPadded = preparedU.padded;
      this._uploadPlane(
        writeSlotIndex,
        slot.u.texture,
        preparedU,
        uvWidth,
        uvHeight,
        commandEncoder,
        true
      );

      const preparedV = this._preparePlaneForUpload(
        vSrc,
        uvWidth,
        uvHeight,
        resolvedVStride,
        'V plane'
      );
      const vTotalBytes = preparedV.totalBytes;
      const vPadded = preparedV.padded;
      this._uploadPlane(
        writeSlotIndex,
        slot.v.texture,
        preparedV,
        uvWidth,
        uvHeight,
        commandEncoder,
        true
      );

      let aBytesPerRow = null;
      let aTotalBytes = 0;
      if (hasAlpha && aSrc) {
        const preparedA = this._preparePlaneForUpload(
          aSrc,
          width,
          height,
          resolvedAStride,
          'A plane'
        );
        aBytesPerRow = preparedA.bytesPerRow;
        aTotalBytes = preparedA.totalBytes;
        this._uploadPlane(
          writeSlotIndex,
          slot.a.texture,
          preparedA,
          width,
          height,
          commandEncoder,
          false
        );
      }

      const uploadMs = this._recordUploadDuration(uploadStartMs, this._now());

      if (colorSpace !== undefined && colorSpace !== null && colorSpace !== this._colorSpace) {
        this._colorSpace = colorSpace;
        this._uniformsDirty = true;
      }
      if (colorRange !== undefined && colorRange !== null && colorRange !== this._colorRange) {
        this._colorRange = colorRange;
        this._uniformsDirty = true;
      }
      if (chromaOrder !== undefined && chromaOrder !== null && chromaOrder !== this._chromaOrder) {
        this._chromaOrder = chromaOrder;
        this._invalidateBindGroupCache();
        this._uniformsDirty = true;
      }
      if (this._pixelFormat !== format) {
        this._uniformsDirty = true;
      }

      const rotatedSlot = this._hasFrame && writeSlotIndex !== this._activePlaneSlot;
      this._pixelFormat = format;
      this._frameWidth = width;
      this._frameHeight = height;
      this._activeSlotIndex = writeSlotIndex;
      this._activePlaneSlot = writeSlotIndex;
      if (rotatedSlot) {
        this._planeSlotRotations += 1;
      }
      this._hasFrame = true;
      this._unpresentedFrame = true;
      this._uploadedFrames += 1;

      const bytesUploaded = yTotalBytes + uTotalBytes + vTotalBytes + aTotalBytes;

      const res = this._yuvUploadResultScratch;
      res.format = format;
      res.colorSpace = this._colorSpace;
      res.colorRange = this._colorRange;
      res.chromaOrder = this._chromaOrder;
      res.width = width;
      res.height = height;
      res.uvWidth = uvWidth;
      res.uvHeight = uvHeight;
      res.yBytesPerRow = yBytesPerRow;
      res.uvBytesPerRow = uBytesPerRow;
      res.aBytesPerRow = aBytesPerRow;
      res.yZeroCopy = !yPadded;
      res.chromaZeroCopy = !uPadded && !vPadded;
      res.bytesUploaded = bytesUploaded;
      res.slotIndex = writeSlotIndex;
      res.uploadedFrames = this._uploadedFrames;
      res.uploadMs = uploadMs;
      return res;
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
     *   chromaOrder?: string,
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
        chromaOrder,
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
      if (chromaOrder !== undefined && chromaOrder !== null) {
        validateChromaOrder(chromaOrder);
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

      const uploadStartMs = this._now();

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
      const yBytesPerRow = preparedY.bytesPerRow;
      const yTotalBytes = preparedY.totalBytes;
      const yPadded = preparedY.padded;
      this._uploadPlane(
        writeSlotIndex,
        slot.y.texture,
        preparedY,
        width,
        height,
        commandEncoder,
        false
      );

      const preparedUv = this._preparePlaneForUpload(
        uvSrc,
        uvTightRowBytes,
        uvHeight,
        resolvedUvStride,
        'UV plane'
      );
      const uvBytesPerRow = preparedUv.bytesPerRow;
      const uvTotalBytes = preparedUv.totalBytes;
      const uvPadded = preparedUv.padded;
      this._uploadPlane(
        writeSlotIndex,
        slot.uv.texture,
        preparedUv,
        uvWidth,
        uvHeight,
        commandEncoder,
        true
      );

      const uploadMs = this._recordUploadDuration(uploadStartMs, this._now());

      if (colorSpace !== undefined && colorSpace !== null && colorSpace !== this._colorSpace) {
        this._colorSpace = colorSpace;
        this._uniformsDirty = true;
      }
      if (colorRange !== undefined && colorRange !== null && colorRange !== this._colorRange) {
        this._colorRange = colorRange;
        this._uniformsDirty = true;
      }
      if (chromaOrder !== undefined && chromaOrder !== null && chromaOrder !== this._chromaOrder) {
        this._chromaOrder = chromaOrder;
        this._invalidateBindGroupCache();
        this._uniformsDirty = true;
      }
      if (this._pixelFormat !== BINK_PIXEL_FORMAT.NV12) {
        this._uniformsDirty = true;
      }

      const rotatedSlot = this._hasFrame && writeSlotIndex !== this._activePlaneSlot;
      this._pixelFormat = BINK_PIXEL_FORMAT.NV12;
      this._frameWidth = width;
      this._frameHeight = height;
      this._activeSlotIndex = writeSlotIndex;
      this._activePlaneSlot = writeSlotIndex;
      if (rotatedSlot) {
        this._planeSlotRotations += 1;
      }
      this._hasFrame = true;
      this._unpresentedFrame = true;
      this._uploadedFrames += 1;

      const bytesUploaded = yTotalBytes + uvTotalBytes;

      const res = this._nv12UploadResultScratch;
      res.format = BINK_PIXEL_FORMAT.NV12;
      res.colorSpace = this._colorSpace;
      res.colorRange = this._colorRange;
      res.chromaOrder = this._chromaOrder;
      res.width = width;
      res.height = height;
      res.uvWidth = uvWidth;
      res.uvHeight = uvHeight;
      res.yBytesPerRow = yBytesPerRow;
      res.uvBytesPerRow = uvBytesPerRow;
      res.yZeroCopy = !yPadded;
      res.chromaZeroCopy = !uvPadded;
      res.bytesUploaded = bytesUploaded;
      res.slotIndex = writeSlotIndex;
      res.uploadedFrames = this._uploadedFrames;
      res.uploadMs = uploadMs;
      return res;
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
      if (
        format === BINK_PIXEL_FORMAT.BGRA8 ||
        format === BINK_PIXEL_FORMAT.RGBA8 ||
        (!format && (frame.rgba || frame.bgra || frame.data))
      ) {
        const opts = this._rgbaUploadOptionsScratch;
        opts.strideBytes = frame.strideBytes;
        opts.commandEncoder = frame.commandEncoder || frame.encoder || null;
        opts.format =
          format === BINK_PIXEL_FORMAT.BGRA8
            ? BINK_PIXEL_FORMAT.BGRA8
            : BINK_PIXEL_FORMAT.RGBA8;
        return this.uploadRgbaFrame(
          frame.bgra || frame.rgba || frame.data,
          frame.width,
          frame.height,
          opts
        );
      }
      validatePixelFormat(format);
      throw new TypeError('uploadFrame could not determine pixel format from frame descriptor');
    }

    _ensurePipelineResources() {
      if (this._pipeline !== null && this._pipelineDevice === this._device) {
        return;
      }
      if (this._pipelineDevice !== null && this._pipelineDevice !== this._device) {
        this._releasePipelineResources();
        this._invalidateBindGroupCache();
      }

      if (this._shaderModule === null) {
        this._shaderModule = this._device.createShaderModule({
          label: 'playgta6-bink-presenter-shader',
          code: BINK_PRESENTER_WGSL,
        });
      }

      if (this._bindGroupLayout === null) {
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
      }

      if (this._pipelineLayout === null) {
        this._pipelineLayout =
          typeof this._device.createPipelineLayout === 'function'
            ? this._device.createPipelineLayout({
                label: 'playgta6-bink-presenter-layout',
                bindGroupLayouts: [this._bindGroupLayout],
              })
            : 'auto';
      }

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

      if (this._sampler === null) {
        this._sampler = this._device.createSampler({
          label: 'playgta6-bink-presenter-sampler',
          magFilter: 'linear',
          minFilter: 'linear',
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge',
        });
      }

      if (this._uniformBuffer === null) {
        this._uniformBuffer = this._device.createBuffer({
          label: 'playgta6-bink-presenter-uniforms',
          size: UNIFORM_BUFFER_SIZE,
          usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
        });
      }

      if (this._dummyTexture === null) {
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
          this._writeTextureWithScratch(
            this._dummyTexture,
            dummyBytes,
            WEBGPU_BYTES_PER_ROW_ALIGNMENT,
            1,
            1,
            1
          );
        }
      }

      this._pipelineDevice = this._device;
      this._bindGroupCacheDevice = this._device;
      this._uniformsDirty = true;
    }

    _getOrCreateBindGroup(
      yView,
      uView,
      vView,
      aView,
      sampler,
      uniformBuffer,
      generation,
      label
    ) {
      const device = this._device;
      const layout = this._bindGroupLayout;
      const chromaOrder = this._chromaOrder;

      if (this._bindGroupCacheDevice !== device) {
        this._invalidateBindGroupCache();
        this._bindGroupCacheDevice = device;
      }

      const cache = this._bindGroupCache;
      const len = cache.length;
      for (let i = 0; i < len; i += 1) {
        const entry = cache[i];
        if (
          entry.yView === yView &&
          entry.uView === uView &&
          entry.vView === vView &&
          entry.aView === aView &&
          entry.sampler === sampler &&
          entry.uniformBuffer === uniformBuffer &&
          entry.layout === layout &&
          entry.device === device &&
          entry.generation === generation &&
          entry.chromaOrder === chromaOrder
        ) {
          if (i > 0) {
            for (let j = i; j > 0; j -= 1) {
              cache[j] = cache[j - 1];
            }
            cache[0] = entry;
          }
          this._bindGroupCacheHits += 1;
          return entry.bindGroup;
        }
      }

      this._bindGroupCacheMisses += 1;

      if (
        this._bindGroupUniformBufferRef !== uniformBuffer ||
        this._bindGroupUniformResource === null
      ) {
        this._bindGroupUniformBufferRef = uniformBuffer;
        this._bindGroupUniformResource = { buffer: uniformBuffer };
      }

      const entries = this._bindGroupEntriesScratch;
      entries[0].resource = this._bindGroupUniformResource;
      entries[1].resource = sampler;
      entries[2].resource = yView;
      entries[3].resource = uView;
      entries[4].resource = vView;
      entries[5].resource = aView;

      const desc = this._bindGroupDescScratch;
      desc.label = label;
      desc.layout = layout;

      const bindGroup = device.createBindGroup(desc);

      let entry;
      if (len < this._bindGroupCacheCapacity) {
        entry = {
          yView,
          uView,
          vView,
          aView,
          sampler,
          uniformBuffer,
          layout,
          device,
          generation,
          chromaOrder,
          bindGroup,
        };
        cache.push(entry);
        for (let j = len; j > 0; j -= 1) {
          cache[j] = cache[j - 1];
        }
        cache[0] = entry;
      } else {
        // Evict LRU entry (at index len - 1) and reuse its entry shell in-place.
        entry = cache[len - 1];
        entry.yView = yView;
        entry.uView = uView;
        entry.vView = vView;
        entry.aView = aView;
        entry.sampler = sampler;
        entry.uniformBuffer = uniformBuffer;
        entry.layout = layout;
        entry.device = device;
        entry.generation = generation;
        entry.chromaOrder = chromaOrder;
        entry.bindGroup = bindGroup;
        for (let j = len - 1; j > 0; j -= 1) {
          cache[j] = cache[j - 1];
        }
        cache[0] = entry;
      }

      return bindGroup;
    }

    _ensureSlotBindGroup(slotIndex) {
      const slot = this._slots[slotIndex];
      const isPackedRgb =
        slot.format === BINK_PIXEL_FORMAT.RGBA8 || slot.format === BINK_PIXEL_FORMAT.BGRA8;
      const primaryView = isPackedRgb ? slot.rgba.view : slot.y.view;
      const uView = isPackedRgb
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

      const bindGroup = this._getOrCreateBindGroup(
        primaryView,
        uView,
        vView,
        aView,
        this._sampler,
        this._uniformBuffer,
        slot.generation,
        slot.bindGroupLabel
      );

      slot.bindGroup = bindGroup;
      slot.bindGroupGeneration = slot.generation;
      slot.bindGroupFormat = slot.format;
      slot.bindGroupWidth = slot.width;
      slot.bindGroupHeight = slot.height;
      slot.bindGroupChromaOrder = this._chromaOrder;

      return bindGroup;
    }

    _syncUniformBuffer(opacity) {
      if (this._hasUploadedUniforms && !this._uniformsDirty && this._lastUniformOpacity === opacity) {
        this._uniformWritesSkipped += 1;
        return false;
      }

      const uniformParams = this._uniformParamsScratch;
      uniformParams.pixelFormat = this._pixelFormat || BINK_PIXEL_FORMAT.RGBA8;
      uniformParams.colorSpace = this._colorSpace;
      uniformParams.colorRange = this._colorRange;
      uniformParams.chromaOrder = this._chromaOrder;
      uniformParams.presentationFormat = this._presentationFormat;
      uniformParams.opacity = opacity;
      uniformParams.uvTransform = this._uvTransform;

      writePresenterUniforms(
        uniformParams,
        this._uniformScratchF32,
        this._uniformScratchU32
      );

      if (this._hasUploadedUniforms) {
        let identical = true;
        for (let i = 0; i < UNIFORM_FLOAT_COUNT; i += 1) {
          if (this._uniformScratchU32[i] !== this._uniformLastUploadedU32[i]) {
            identical = false;
            break;
          }
        }
        if (identical) {
          this._uniformsDirty = false;
          this._lastUniformOpacity = opacity;
          this._uniformWritesSkipped += 1;
          return false;
        }
      }

      if (this._device.queue && typeof this._device.queue.writeBuffer === 'function') {
        this._device.queue.writeBuffer(this._uniformBuffer, 0, this._uniformScratchF32);
      }
      this._uniformLastUploadedU32.set(this._uniformScratchU32);
      this._hasUploadedUniforms = true;
      this._uniformsDirty = false;
      this._lastUniformOpacity = opacity;
      return true;
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
     *   nowMs?: number,
     * }} [options]
     */
    presentFrame(commandEncoder, targetView, dstWidth, dstHeight, options = null) {
      this._assertNotDestroyed();
      if (!commandEncoder || typeof commandEncoder.beginRenderPass !== 'function') {
        throw new TypeError('presentFrame requires a GPUCommandEncoder with beginRenderPass()');
      }
      if (!targetView || typeof targetView !== 'object') {
        throw new TypeError('presentFrame requires a target GPUTextureView');
      }
      assertPositiveFinite(dstWidth, 'dstWidth');
      assertPositiveFinite(dstHeight, 'dstHeight');
      if (options !== undefined && options !== null && typeof options !== 'object') {
        throw new TypeError('presentFrame options must be an object');
      }

      const opts = options || EMPTY_OBJECT;
      const clearColor = opts.clearColor !== undefined ? opts.clearColor : DEFAULT_CLEAR_COLOR;
      const loadOp = opts.loadOp !== undefined ? opts.loadOp : this._loadOp;
      const storeOp = opts.storeOp !== undefined ? opts.storeOp : 'store';
      const opacity = opts.opacity !== undefined ? opts.opacity : 1.0;
      const explicitNowMs = opts.nowMs;

      validateLoadOp(loadOp);
      validateStoreOp(storeOp);
      validateOpacity(opacity);
      if (explicitNowMs !== undefined && (typeof explicitNowMs !== 'number' || !Number.isFinite(explicitNowMs))) {
        throw new RangeError(`nowMs must be a finite number, got ${explicitNowMs}`);
      }

      // Flush any deferred staging uploads before starting the render pass.
      if (this._stagingPool && this._pendingUploads.length > 0) {
        this.flushPendingUploads(commandEncoder);
      }

      this._ensurePipelineResources();

      let renderPassDesc;
      if (loadOp === BINK_LOAD_OP.CLEAR) {
        writeClearColor(clearColor, this._clearColorValueScratch);
        this._clearColorAttachmentScratch.view = targetView;
        this._clearColorAttachmentScratch.storeOp = storeOp;
        renderPassDesc = this._clearRenderPassDescScratch;
      } else {
        this._loadColorAttachmentScratch.view = targetView;
        this._loadColorAttachmentScratch.storeOp = storeOp;
        renderPassDesc = this._loadRenderPassDescScratch;
      }

      const pass = commandEncoder.beginRenderPass(renderPassDesc);

      if (!this._hasFrame) {
        pass.end();
        const res = this._presentResultScratch;
        res.presented = false;
        res.viewport = null;
        res.format = this._pixelFormat;
        res.colorSpace = this._colorSpace;
        res.colorRange = this._colorRange;
        res.chromaOrder = this._chromaOrder;
        res.loadOp = loadOp;
        res.slotIndex = this._activeSlotIndex;
        res.presentedFrames = this._presentedFrames;
        res.framesPresented = this._presentedFrames;
        return res;
      }

      const viewport = writeLetterboxViewport(
        this._frameWidth,
        this._frameHeight,
        dstWidth,
        dstHeight,
        this._viewportScratch
      );

      this._syncUniformBuffer(opacity);
      const bindGroup = this._ensureSlotBindGroup(this._activeSlotIndex);

      pass.setPipeline(this._pipeline);
      pass.setViewport(viewport.x, viewport.y, viewport.width, viewport.height, 0, 1);
      if (typeof pass.setScissorRect === 'function') {
        const sx = Math.max(0, Math.floor(viewport.x));
        const sy = Math.max(0, Math.floor(viewport.y));
        const sw = Math.max(
          1,
          Math.min(Math.floor(dstWidth) - sx, Math.ceil(viewport.x + viewport.width) - sx)
        );
        const sh = Math.max(
          1,
          Math.min(Math.floor(dstHeight) - sy, Math.ceil(viewport.y + viewport.height) - sy)
        );
        pass.setScissorRect(sx, sy, sw, sh);
      }
      pass.setBindGroup(0, bindGroup);
      pass.draw(3, 1, 0, 0);
      pass.end();

      const presentNowMs = explicitNowMs !== undefined ? explicitNowMs : this._now();
      if (this._lastPresentMs !== null) {
        const deltaMs = presentNowMs - this._lastPresentMs;
        if (Number.isFinite(deltaMs) && deltaMs > 0) {
          this._lastFrameIntervalMs = deltaMs;
          this._measuredFrameIntervals += 1;
          if (deltaMs > this._peakPresentIntervalMs) {
            this._peakPresentIntervalMs = deltaMs;
          }
          if (deltaMs <= TARGET_FPS_30_INTERVAL_MS) {
            this._targetFps30Hits += 1;
          } else {
            this._framesOver30FpsBudget += 1;
          }
        } else {
          this._lastFrameIntervalMs = 0;
        }
      }
      this._lastPresentMs = Number.isFinite(presentNowMs) ? presentNowMs : this._lastPresentMs;
      this._presentedFrames += 1;
      this._unpresentedFrame = false;

      const res = this._presentResultScratch;
      res.presented = true;
      res.viewport = viewport;
      res.format = this._pixelFormat;
      res.colorSpace = this._colorSpace;
      res.colorRange = this._colorRange;
      res.chromaOrder = this._chromaOrder;
      res.loadOp = loadOp;
      res.slotIndex = this._activeSlotIndex;
      res.presentedFrames = this._presentedFrames;
      res.framesPresented = this._presentedFrames;
      return res;
    }

    /**
     * Alias for `presentFrame(...)`.
     */
    present(commandEncoder, targetView, dstWidth, dstHeight, options = null) {
      return this.presentFrame(commandEncoder, targetView, dstWidth, dstHeight, options);
    }

    /**
     * Invalidate cached bind groups on output/surface resize without blanking the active frame.
     * Optional `width` / `height` arguments are validated when provided.
     */
    resize(width, height) {
      this._assertNotDestroyed();
      if (width !== undefined || height !== undefined) {
        assertPositiveInteger(width, 'width');
        assertPositiveInteger(height, 'height');
      }
      this._invalidateBindGroupCache();
    }

    /**
     * Switch to a new WebGPU device (e.g. after device loss / recovery), releasing all GPU
     * textures, pipeline resources, and cached bind groups belonging to the previous device.
     */
    setDevice(newDevice) {
      this._assertNotDestroyed();
      if (!newDevice || typeof newDevice.createTexture !== 'function') {
        throw new TypeError('setDevice requires a WebGPU device with createTexture()');
      }
      if (newDevice === this._device) {
        return;
      }
      this._destroyAllSlots();
      this._releasePipelineResources();
      this._device = newDevice;
      this._bindGroupCacheDevice = newDevice;
      this._pixelFormat = null;
      this._frameWidth = 0;
      this._frameHeight = 0;
      this._activeSlotIndex = 0;
      this._activePlaneSlot = 0;
      this._hasFrame = false;
      this._unpresentedFrame = false;
      this._lastAdvanceMs = null;
      this._lastPresentMs = null;
    }

    /**
     * Reset active plane slots, cached bind groups, and playback timing anchors so a new video
     * stream or recovered GPU device can start cleanly. Optionally binds `newDevice`.
     */
    reset(newDevice = null) {
      this._assertNotDestroyed();
      if (newDevice !== null && newDevice !== undefined) {
        if (typeof newDevice.createTexture !== 'function') {
          throw new TypeError('reset(newDevice) requires a WebGPU device with createTexture()');
        }
        if (newDevice !== this._device) {
          this._destroyAllSlots();
          this._releasePipelineResources();
          this._device = newDevice;
          this._bindGroupCacheDevice = newDevice;
        } else {
          this._destroyAllSlots();
        }
      } else {
        this._destroyAllSlots();
      }
      this._pixelFormat = null;
      this._frameWidth = 0;
      this._frameHeight = 0;
      this._activeSlotIndex = 0;
      this._activePlaneSlot = 0;
      this._hasFrame = false;
      this._unpresentedFrame = false;
      this._lastAdvanceMs = null;
      this._lastPresentMs = null;
      this._uniformsDirty = true;
    }

    /**
     * Set UV crop/scale transform `[scaleX, scaleY, offsetX, offsetY]` applied in the shader.
     */
    setUvTransform(scaleX = 1, scaleY = 1, offsetX = 0, offsetY = 0) {
      this._assertNotDestroyed();
      if (
        this._uvTransform[0] === scaleX &&
        this._uvTransform[1] === scaleY &&
        this._uvTransform[2] === offsetX &&
        this._uvTransform[3] === offsetY
      ) {
        return;
      }
      this._uvTransform[0] = scaleX;
      this._uvTransform[1] = scaleY;
      this._uvTransform[2] = offsetX;
      this._uvTransform[3] = offsetY;
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
     * Switch active chroma plane ordering (`yuv` for Y/Cb/Cr or `yvu` for Bink2 native Y/Cr/Cb).
     */
    setChromaOrder(chromaOrder) {
      this._assertNotDestroyed();
      validateChromaOrder(chromaOrder);
      if (chromaOrder !== this._chromaOrder) {
        this._chromaOrder = chromaOrder;
        this._invalidateBindGroupCache();
        this._uniformsDirty = true;
      }
    }

    /**
     * Switch presentation swapchain target format (e.g. `bgra8unorm` <-> `bgra8unorm-srgb` /
     * `rgba16float`), lazily rebuilding only the render pipeline and updating linear-output
     * sRGB conversion uniforms.
     */
    setPresentationFormat(format) {
      this._assertNotDestroyed();
      if (typeof format !== 'string' || format.trim().length === 0) {
        throw new TypeError('presentationFormat must be a non-empty string');
      }
      const normalized = format.trim();
      if (normalized !== this._presentationFormat) {
        this._presentationFormat = normalized;
        this._pipeline = null;
        this._uniformsDirty = true;
      }
    }

    /**
     * Switch default render-pass `loadOp` (`clear` or `load` for menu/UI underlay composition).
     */
    setLoadOp(loadOp) {
      this._assertNotDestroyed();
      validateLoadOp(loadOp);
      this._loadOp = loadOp;
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
     * Return sanitized frame-pacing and upload timing telemetry.
     */
    getPacingStats() {
      const targetFps = sanitizeNonNegativeNumber(this._targetFps);
      const targetFrameIntervalMs = targetFps > 0 ? 1000 / targetFps : 0;
      const presentedFrames = sanitizeNonNegativeInteger(this._presentedFrames);
      const droppedFrames = sanitizeNonNegativeInteger(this._droppedFrames);
      const totalFrames = presentedFrames + droppedFrames;
      const dropRate = totalFrames > 0 ? droppedFrames / totalFrames : 0;
      const uploadSamples = sanitizeNonNegativeInteger(this._uploadTimingCount);
      const avgUploadMs =
        uploadSamples > 0
          ? sanitizeNonNegativeNumber(this._totalUploadMs / uploadSamples)
          : 0;
      const maxUploadMs = sanitizeNonNegativeNumber(this._maxUploadMs);
      const lastUploadMs = sanitizeNonNegativeNumber(this._lastUploadMs);
      const lastFrameIntervalMs = sanitizeNonNegativeNumber(this._lastFrameIntervalMs);

      return {
        targetFps,
        targetFrameIntervalMs,
        presentedFrames,
        droppedFrames,
        dropRate,
        uploadSamples,
        avgUploadMs,
        maxUploadMs,
        lastUploadMs,
        lastFrameIntervalMs,
      };
    }

    /**
     * Reset telemetry counters and pacing timing accumulators without disrupting
     * allocated GPU textures, bind-group caches, or playback configuration.
     */
    resetStats() {
      this._assertNotDestroyed();
      this._uploadedFrames = 0;
      this._presentedFrames = 0;
      this._droppedFrames = 0;
      this._resizeCount = 0;
      this._planeReallocations = 0;
      this._scratchAllocations = 0;
      this._bytesUploaded = 0;
      this._directZeroCopyUploads = 0;
      this._chromaRepackUploads = 0;
      this._bindGroupCacheHits = 0;
      this._bindGroupCacheMisses = 0;
      this._uniformWritesSkipped = 0;
      this._uploadTimingCount = 0;
      this._totalUploadMs = 0;
      this._maxUploadMs = 0;
      this._lastUploadMs = 0;
      this._lastFrameIntervalMs = 0;
      this._measuredFrameIntervals = 0;
      this._targetFps30Hits = 0;
      this._planeSlotRotations = 0;
      this._reusedUploadDescriptors = 0;
      this._framesOver30FpsBudget = 0;
      this._peakPresentIntervalMs = 0;
      this._lastPresentMs = null;
    }

    /**
     * Return presenter state and frame/upload telemetry.
     */
    getStats() {
      const pacing = this.getPacingStats();
      return {
        pixelFormat: this._pixelFormat,
        colorSpace: this._colorSpace,
        colorRange: this._colorRange,
        chromaOrder: this._chromaOrder,
        loadOp: this._loadOp,
        presentationFormat: this._presentationFormat,
        frameWidth: this._frameWidth,
        frameHeight: this._frameHeight,
        lastFrameWidth: this._frameWidth,
        lastFrameHeight: this._frameHeight,
        targetFps: pacing.targetFps,
        frameIntervalMs: pacing.targetFrameIntervalMs,
        targetFrameIntervalMs: pacing.targetFrameIntervalMs,
        uploadedFrames: this._uploadedFrames,
        presentedFrames: pacing.presentedFrames,
        framesPresented: pacing.presentedFrames,
        droppedFrames: pacing.droppedFrames,
        dropRate: pacing.dropRate,
        uploadSamples: pacing.uploadSamples,
        avgUploadMs: pacing.avgUploadMs,
        maxUploadMs: pacing.maxUploadMs,
        lastUploadMs: pacing.lastUploadMs,
        lastFrameIntervalMs: pacing.lastFrameIntervalMs,
        targetFpsHitRate30: this.targetFpsHitRate30,
        framesOver30FpsBudget: sanitizeNonNegativeInteger(this._framesOver30FpsBudget),
        peakPresentIntervalMs: sanitizeNonNegativeNumber(this._peakPresentIntervalMs),
        resizeCount: this._resizeCount,
        planeReallocations: this._planeReallocations,
        scratchAllocations: this._scratchAllocations,
        directZeroCopyUploads: this._directZeroCopyUploads,
        chromaRepackUploads: this._chromaRepackUploads,
        reusedUploadDescriptors: sanitizeNonNegativeInteger(this._reusedUploadDescriptors),
        bytesUploaded: this._bytesUploaded,
        bindGroupCacheHits: this._bindGroupCacheHits,
        bindGroupCacheMisses: this._bindGroupCacheMisses,
        bindGroupHitRate: this.bindGroupHitRate,
        bindGroupCacheSize: this._bindGroupCache.length,
        bindGroupCacheCapacity: this._bindGroupCacheCapacity,
        uniformWritesSkipped: this._uniformWritesSkipped,
        slotCount: this._slots.length,
        planeRingDepth: sanitizeNonNegativeInteger(this._planeRingDepth),
        planeSlotRotations: sanitizeNonNegativeInteger(this._planeSlotRotations),
        activeSlotIndex: this._activeSlotIndex,
        hasFrame: this._hasFrame,
        paused: this._paused,
        looping: this._looping,
        skipped: this._skipped,
        pipelineReady: this._pipeline !== null,
        usesStagingPool: this._stagingPool !== null,
        allowUnalignedDirectUpload: this._allowUnalignedDirectUpload,
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
      this._releasePipelineResources();

      this._uploadScratch = null;
      this._preparedPlaneScratch.data = null;
      this._uploadDestinationScratch.texture = null;
      this._clearColorAttachmentScratch.view = null;
      this._loadColorAttachmentScratch.view = null;
      this._presentResultScratch.viewport = null;
      this._rgbaUploadOptionsScratch.commandEncoder = null;
      this._activeSlotIndex = 0;
      this._activePlaneSlot = 0;
      this._hasFrame = false;
      this._unpresentedFrame = false;
      this._lastAdvanceMs = null;
      this._lastPresentMs = null;
      this._bindGroupCacheDevice = null;
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
    UNIFORM_FLAG_SWAP_CHROMA,
    UNIFORM_FLAG_OUTPUT_LINEAR,
    DEFAULT_PRESENTATION_FORMAT,
    DEFAULT_TARGET_FPS,
    TARGET_FPS_30_INTERVAL_MS,
    DEFAULT_BIND_GROUP_CACHE_CAPACITY,
    TRIPLE_BUFFER_SLOTS,
    BINK_PIXEL_FORMAT,
    BINK_CHROMA_ORDER,
    BINK_LOAD_OP,
    YUV_COLOR_SPACE,
    YUV_COLOR_RANGE,
    YUV_MATRICES,
    YUV_FULL_RANGE_MATRICES,
    YUV_RANGE_MATRICES,
    TEXTURE_USAGE,
    BUFFER_USAGE,
    SHADER_STAGE,
    BINK_PRESENTER_WGSL,
    isLinearPresentationFormat,
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
