import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const texMod = require('../web/workers/texture_formats.js');

const {
  DXGI_FORMAT,
  DXGI_FORMAT_NAMES,
  DXGI_FORMAT_MAX_ID,
  DXGI_FORMAT_TABLE,
  WEBGPU_FORMAT_INFO,
  WEBGPU_BYTES_PER_ROW_ALIGNMENT,
  TEXTURE_COMPRESSION_BC_FEATURE,
  TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE,
  TEXTURE_FORMATS_TIER1_FEATURE,
  CHROMIUM_UNORM16_FEATURE,
  DEPTH32FLOAT_STENCIL8_FEATURE,
  BGRA8UNORM_STORAGE_FEATURE,
  RG11B10UFLOAT_RENDERABLE_FEATURE,
  FLOAT32_FILTERABLE_FEATURE,
  getWebGpuFormatInfo,
  describeTextureFormatCapabilities,
  validateTextureDimensionAndFormat,
  lookupDxgiFormat,
  mapDxgiFormatToWebGpu,
  getSrgbSiblingFormat,
  classifyTextureFormatSupport,
  resolveStorageCompatibleFormat,
  calcMipSize,
  calcMipLayout,
  computeMipLayout,
  computeMipByteSize,
  calcSubresourceLayout,
  calcTextureMemoryBytes,
  padTextureRowsForWriteTexture,
  decodeBc1ToRgba8,
  decodeBc3ToRgba8,
  decodeBc4ToR8,
} = texMod;

/** Pack 16 2-bit BC1/BC3 color indices into a little-endian uint32. */
function pack2BitIndices(indices16) {
  assert.equal(indices16.length, 16);
  let bits = 0;
  for (let i = 0; i < 16; i += 1) {
    bits = (bits | ((indices16[i] & 0x3) << (i * 2))) >>> 0;
  }
  return bits;
}

/** Build an 8-byte BC1 block from RGB565 endpoints and 16 2-bit indices. */
function encodeBc1Block(color0, color1, indices16) {
  const out = new Uint8Array(8);
  const view = new DataView(out.buffer);
  view.setUint16(0, color0 & 0xffff, true);
  view.setUint16(2, color1 & 0xffff, true);
  view.setUint32(4, pack2BitIndices(indices16), true);
  return out;
}

/** Build an 8-byte BC4 / BC3-alpha block from 8-bit endpoints and 16 3-bit indices. */
function encodeBc4Block(endpoint0, endpoint1, indices16) {
  assert.equal(indices16.length, 16);
  const out = new Uint8Array(8);
  out[0] = endpoint0 & 0xff;
  out[1] = endpoint1 & 0xff;

  let lo = 0;
  for (let i = 0; i < 8; i += 1) {
    lo |= (indices16[i] & 0x7) << (i * 3);
  }
  let hi = 0;
  for (let i = 0; i < 8; i += 1) {
    hi |= (indices16[i + 8] & 0x7) << (i * 3);
  }

  out[2] = lo & 0xff;
  out[3] = (lo >>> 8) & 0xff;
  out[4] = (lo >>> 16) & 0xff;
  out[5] = hi & 0xff;
  out[6] = (hi >>> 8) & 0xff;
  out[7] = (hi >>> 16) & 0xff;
  return out;
}

/** Build a 16-byte BC3 block from alpha + RGB565 endpoints and indices. */
function encodeBc3Block(alpha0, alpha1, alphaIndices16, color0, color1, colorIndices16) {
  const out = new Uint8Array(16);
  out.set(encodeBc4Block(alpha0, alpha1, alphaIndices16), 0);
  out.set(encodeBc1Block(color0, color1, colorIndices16), 8);
  return out;
}

test('UMD export attaches to globalThis.PlayGta6TextureFormats and module.exports', () => {
  assert.equal(globalThis.PlayGta6TextureFormats, texMod);
  assert.equal(WEBGPU_BYTES_PER_ROW_ALIGNMENT, 256);
  assert.equal(TEXTURE_COMPRESSION_BC_FEATURE, 'texture-compression-bc');
});

test('DXGI_FORMAT numeric values match official DirectX dxgiformat.h constants', () => {
  assert.equal(DXGI_FORMAT.R16G16B16A16_FLOAT, 10);
  assert.equal(DXGI_FORMAT.R8G8B8A8_UNORM, 28);
  assert.equal(DXGI_FORMAT.R8G8B8A8_UNORM_SRGB, 29);
  assert.equal(DXGI_FORMAT.D32_FLOAT, 40);
  assert.equal(DXGI_FORMAT.R32_FLOAT, 41);
  assert.equal(DXGI_FORMAT.D24_UNORM_S8_UINT, 45);
  assert.equal(DXGI_FORMAT.R16_FLOAT, 54);
  assert.equal(DXGI_FORMAT.R8_UNORM, 61);
  assert.equal(DXGI_FORMAT.BC1_UNORM, 71);
  assert.equal(DXGI_FORMAT.BC1_UNORM_SRGB, 72);
  assert.equal(DXGI_FORMAT.BC2_UNORM, 74);
  assert.equal(DXGI_FORMAT.BC2_UNORM_SRGB, 75);
  assert.equal(DXGI_FORMAT.BC3_UNORM, 77);
  assert.equal(DXGI_FORMAT.BC3_UNORM_SRGB, 78);
  assert.equal(DXGI_FORMAT.BC4_UNORM, 80);
  assert.equal(DXGI_FORMAT.BC4_SNORM, 81);
  assert.equal(DXGI_FORMAT.BC5_UNORM, 83);
  assert.equal(DXGI_FORMAT.BC5_SNORM, 84);
  assert.equal(DXGI_FORMAT.B8G8R8A8_UNORM, 87);
  assert.equal(DXGI_FORMAT.B8G8R8A8_UNORM_SRGB, 91);
  assert.equal(DXGI_FORMAT.BC6H_UF16, 95);
  assert.equal(DXGI_FORMAT.BC6H_SF16, 96);
  assert.equal(DXGI_FORMAT.BC7_UNORM, 98);
  assert.equal(DXGI_FORMAT.BC7_UNORM_SRGB, 99);
});

test('mapDxgiFormatToWebGpu maps uncompressed and depth formats accurately', () => {
  const cases = [
    [DXGI_FORMAT.R8G8B8A8_UNORM, 'rgba8unorm', 4, false],
    [DXGI_FORMAT.R8G8B8A8_UNORM_SRGB, 'rgba8unorm-srgb', 4, false],
    [DXGI_FORMAT.B8G8R8A8_UNORM, 'bgra8unorm', 4, false],
    [DXGI_FORMAT.B8G8R8A8_UNORM_SRGB, 'bgra8unorm-srgb', 4, false],
    [DXGI_FORMAT.R16G16B16A16_FLOAT, 'rgba16float', 8, false],
    [DXGI_FORMAT.R32_FLOAT, 'r32float', 4, false],
    [DXGI_FORMAT.R8_UNORM, 'r8unorm', 1, false],
    [DXGI_FORMAT.R16_FLOAT, 'r16float', 2, false],
    [DXGI_FORMAT.D24_UNORM_S8_UINT, 'depth24plus-stencil8', 4, true],
    [DXGI_FORMAT.D32_FLOAT, 'depth32float', 4, true],
  ];

  for (const [dxgi, expectedFormat, bytesPerBlock, isDepth] of cases) {
    const info = mapDxgiFormatToWebGpu(dxgi);
    assert.equal(info.webgpuFormat, expectedFormat);
    assert.equal(info.blockWidth, 1);
    assert.equal(info.blockHeight, 1);
    assert.equal(info.bytesPerBlock, bytesPerBlock);
    assert.equal(info.isCompressed, false);
    assert.equal(info.isDepth, isDepth);
    assert.equal(info.requiresFeature, null);
    assert.equal(info.needsSoftwareDecode, false);
  }

  // String lookup also works with or without DXGI_FORMAT_ prefix
  assert.equal(mapDxgiFormatToWebGpu('R8G8B8A8_UNORM').webgpuFormat, 'rgba8unorm');
  assert.equal(mapDxgiFormatToWebGpu('DXGI_FORMAT_D32_FLOAT').webgpuFormat, 'depth32float');

  assert.throws(() => mapDxgiFormatToWebGpu(DXGI_FORMAT.UNKNOWN), RangeError);
  assert.throws(() => mapDxgiFormatToWebGpu(9999), RangeError);
  assert.throws(() => mapDxgiFormatToWebGpu('NOT_A_FORMAT'), RangeError);
  assert.throws(() => mapDxgiFormatToWebGpu(null), TypeError);
});

test('mapDxgiFormatToWebGpu maps BC1..BC7 with and without texture-compression-bc feature', () => {
  const bcCases = [
    [DXGI_FORMAT.BC1_UNORM, 'bc1-rgba-unorm', 'rgba8unorm', 8],
    [DXGI_FORMAT.BC1_UNORM_SRGB, 'bc1-rgba-unorm-srgb', 'rgba8unorm-srgb', 8],
    [DXGI_FORMAT.BC2_UNORM, 'bc2-rgba-unorm', 'rgba8unorm', 16],
    [DXGI_FORMAT.BC2_UNORM_SRGB, 'bc2-rgba-unorm-srgb', 'rgba8unorm-srgb', 16],
    [DXGI_FORMAT.BC3_UNORM, 'bc3-rgba-unorm', 'rgba8unorm', 16],
    [DXGI_FORMAT.BC3_UNORM_SRGB, 'bc3-rgba-unorm-srgb', 'rgba8unorm-srgb', 16],
    [DXGI_FORMAT.BC4_UNORM, 'bc4-r-unorm', 'rgba8unorm', 8],
    [DXGI_FORMAT.BC4_SNORM, 'bc4-r-snorm', 'rgba8unorm', 8],
    [DXGI_FORMAT.BC5_UNORM, 'bc5-rg-unorm', 'rgba8unorm', 16],
    [DXGI_FORMAT.BC5_SNORM, 'bc5-rg-snorm', 'rgba8unorm', 16],
    [DXGI_FORMAT.BC6H_UF16, 'bc6h-rgb-ufloat', 'rgba8unorm', 16],
    [DXGI_FORMAT.BC6H_SF16, 'bc6h-rgb-float', 'rgba8unorm', 16],
    [DXGI_FORMAT.BC7_UNORM, 'bc7-rgba-unorm', 'rgba8unorm', 16],
    [DXGI_FORMAT.BC7_UNORM_SRGB, 'bc7-rgba-unorm-srgb', 'rgba8unorm-srgb', 16],
  ];

  const withBcSet = new Set([TEXTURE_COMPRESSION_BC_FEATURE]);
  const withBcArray = [TEXTURE_COMPRESSION_BC_FEATURE];
  const withoutBcSet = new Set(['timestamp-query']);

  for (const [dxgi, nativeFmt, fallbackFmt, bytesPerBlock] of bcCases) {
    const hw = mapDxgiFormatToWebGpu(dxgi, withBcSet);
    assert.equal(hw.webgpuFormat, nativeFmt);
    assert.equal(hw.nativeWebgpuFormat, nativeFmt);
    assert.equal(hw.blockWidth, 4);
    assert.equal(hw.blockHeight, 4);
    assert.equal(hw.bytesPerBlock, bytesPerBlock);
    assert.equal(hw.isCompressed, true);
    assert.equal(hw.isDepth, false);
    assert.equal(hw.requiresFeature, TEXTURE_COMPRESSION_BC_FEATURE);
    assert.equal(hw.needsSoftwareDecode, false);

    // Array feature list also works
    assert.equal(mapDxgiFormatToWebGpu(dxgi, withBcArray).webgpuFormat, nativeFmt);

    // Without 'texture-compression-bc' -> fallback rgba8unorm / rgba8unorm-srgb + needsSoftwareDecode
    const sw = mapDxgiFormatToWebGpu(dxgi, withoutBcSet);
    assert.equal(sw.webgpuFormat, fallbackFmt);
    assert.equal(sw.nativeWebgpuFormat, nativeFmt);
    assert.equal(sw.blockWidth, 4);
    assert.equal(sw.blockHeight, 4);
    assert.equal(sw.bytesPerBlock, bytesPerBlock);
    assert.equal(sw.isCompressed, true);
    assert.equal(sw.requiresFeature, TEXTURE_COMPRESSION_BC_FEATURE);
    assert.equal(sw.needsSoftwareDecode, true);

    // Omitting supportedFeatures defaults safely to software decode fallback
    const swDefault = mapDxgiFormatToWebGpu(dxgi);
    assert.equal(swDefault.webgpuFormat, fallbackFmt);
    assert.equal(swDefault.needsSoftwareDecode, true);
  }
});

test('calcMipSize shrinks dimensions down to 1x1 cleanly including large mip levels', () => {
  assert.deepEqual(calcMipSize(1024, 512, 1, 0), { width: 1024, height: 512, depthOrLayers: 1 });
  assert.deepEqual(calcMipSize(1024, 512, 1, 1), { width: 512, height: 256, depthOrLayers: 1 });
  assert.deepEqual(calcMipSize(1024, 512, 1, 9), { width: 2, height: 1, depthOrLayers: 1 });
  assert.deepEqual(calcMipSize(1024, 512, 1, 10), { width: 1, height: 1, depthOrLayers: 1 });
  assert.deepEqual(calcMipSize(1024, 512, 1, 12), { width: 1, height: 1, depthOrLayers: 1 });
  // Must not suffer from JS 32-bit shift wraparound (mipLevel === 32)
  assert.deepEqual(calcMipSize(1024, 512, 1, 32), { width: 1, height: 1, depthOrLayers: 1 });

  // Non-power-of-two floor halving
  assert.deepEqual(calcMipSize(15, 7, 6, 1), { width: 7, height: 3, depthOrLayers: 6 });
  assert.deepEqual(calcMipSize(15, 7, 6, 2), { width: 3, height: 1, depthOrLayers: 6 });
  assert.deepEqual(calcMipSize(15, 7, 6, 3), { width: 1, height: 1, depthOrLayers: 6 });

  // 3D texture depth halving when is3D=true
  assert.deepEqual(calcMipSize(64, 32, 16, 2, true), { width: 16, height: 8, depthOrLayers: 4 });
  assert.deepEqual(calcMipSize(64, 32, 16, 5, true), { width: 2, height: 1, depthOrLayers: 1 });
});

test('calcMipLayout computes 256-byte aligned bytesPerRow for uncompressed and BCn textures', () => {
  const rgba8 = mapDxgiFormatToWebGpu(DXGI_FORMAT.R8G8B8A8_UNORM);
  const bc1 = mapDxgiFormatToWebGpu(DXGI_FORMAT.BC1_UNORM, [TEXTURE_COMPRESSION_BC_FEATURE]);
  const bc3 = mapDxgiFormatToWebGpu(DXGI_FORMAT.BC3_UNORM, [TEXTURE_COMPRESSION_BC_FEATURE]);

  // Uncompressed non-aligned row (10px * 4B = 40B -> aligned to 256B)
  assert.deepEqual(calcMipLayout(rgba8, 10, 6), {
    width: 10,
    height: 6,
    depthOrLayers: 1,
    blocksX: 10,
    blocksY: 6,
    unalignedBytesPerRow: 40,
    bytesPerRow: 256,
    rowsPerImage: 6,
    totalBytes: 256 * 6,
  });

  // Uncompressed exact 256-byte row (64px * 4B = 256B -> stays 256B)
  const exactRgba = calcMipLayout(rgba8, 64, 32, 2);
  assert.equal(exactRgba.unalignedBytesPerRow, 256);
  assert.equal(exactRgba.bytesPerRow, 256);
  assert.equal(exactRgba.rowsPerImage, 32);
  assert.equal(exactRgba.totalBytes, 256 * 32 * 2);

  // Uncompressed slightly above 256B (65px * 4B = 260B -> 512B)
  assert.equal(calcMipLayout(rgba8, 65, 4).bytesPerRow, 512);

  // BC1 4x4 compressed non-multiple-of-4 (10x6 -> 3x2 blocks, 3*8 = 24B -> 256B)
  assert.deepEqual(calcMipLayout(bc1, 10, 6), {
    width: 10,
    height: 6,
    depthOrLayers: 1,
    blocksX: 3,
    blocksY: 2,
    unalignedBytesPerRow: 24,
    bytesPerRow: 256,
    rowsPerImage: 2,
    totalBytes: 512,
  });

  // BC1 1x1 tail mip -> 1x1 block (8B -> 256B)
  assert.deepEqual(calcMipLayout(bc1, 1, 1), {
    width: 1,
    height: 1,
    depthOrLayers: 1,
    blocksX: 1,
    blocksY: 1,
    unalignedBytesPerRow: 8,
    bytesPerRow: 256,
    rowsPerImage: 1,
    totalBytes: 256,
  });

  // BC3 256x128 -> 64x32 blocks, 64*16 = 1024B (already multiple of 256B)
  assert.deepEqual(calcMipLayout(bc3, 256, 128, 6), {
    width: 256,
    height: 128,
    depthOrLayers: 6,
    blocksX: 64,
    blocksY: 32,
    unalignedBytesPerRow: 1024,
    bytesPerRow: 1024,
    rowsPerImage: 32,
    totalBytes: 1024 * 32 * 6,
  });

  // Custom tight alignment (alignBytesPerRow = 1)
  const tight = calcMipLayout(bc3, 13, 9, 1, 1);
  assert.equal(tight.blocksX, 4);
  assert.equal(tight.blocksY, 3);
  assert.equal(tight.unalignedBytesPerRow, 64);
  assert.equal(tight.bytesPerRow, 64);
  assert.equal(tight.totalBytes, 192);
});

test('padTextureRowsForWriteTexture pads rows to 256-byte stride with zero padding and handles tight fast path', () => {
  const unaligned = 12;
  const aligned = 256;
  const rows = 3;
  const src = new Uint8Array(unaligned * rows);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < unaligned; c += 1) {
      src[r * unaligned + c] = (r + 1) * 16 + c;
    }
  }

  const padded = padTextureRowsForWriteTexture(src, unaligned, aligned, rows);
  assert.equal(padded.byteLength, aligned * rows);

  for (let r = 0; r < rows; r += 1) {
    const rowStart = r * aligned;
    for (let c = 0; c < unaligned; c += 1) {
      assert.equal(padded[rowStart + c], (r + 1) * 16 + c);
    }
    for (let c = unaligned; c < aligned; c += 1) {
      assert.equal(padded[rowStart + c], 0);
    }
  }

  // Tight path (aligned === unaligned) returns an independent copy of exact length
  const tight = padTextureRowsForWriteTexture(src, unaligned, unaligned, rows);
  assert.equal(tight.byteLength, src.byteLength);
  assert.deepEqual(tight, src);
  tight[0] = 0xff;
  assert.notEqual(src[0], 0xff);

  // Validation errors
  assert.throws(() => padTextureRowsForWriteTexture(src.subarray(0, 10), 12, 256, 3), RangeError);
  assert.throws(() => padTextureRowsForWriteTexture(src, 12, 8, 3), RangeError);
});

test('decodeBc1ToRgba8 decodes opaque 4-color mode and 3-color + 1-bit punchthrough alpha exactly', () => {
  // 1) Opaque 4-color mode: color0 > color1
  // Pure red (R5=31, G6=0, B5=0 -> 0xF800) > pure blue (R5=0, G6=0, B5=31 -> 0x001F)
  // Expected palette:
  //   0: (255, 0, 0, 255)
  //   1: (0, 0, 255, 255)
  //   2: (170, 0, 85, 255)
  //   3: (85, 0, 170, 255)
  const opaqueIndices = [
    0, 1, 2, 3,
    3, 2, 1, 0,
    0, 2, 1, 3,
    1, 3, 0, 2,
  ];
  const opaqueBlock = encodeBc1Block(0xf800, 0x001f, opaqueIndices);
  const opaqueRgba = decodeBc1ToRgba8(opaqueBlock, 4, 4);
  assert.equal(opaqueRgba.byteLength, 4 * 4 * 4);

  const expectedOpaquePalette = [
    [255, 0, 0, 255],
    [0, 0, 255, 255],
    [170, 0, 85, 255],
    [85, 0, 170, 255],
  ];
  for (let i = 0; i < 16; i += 1) {
    const exp = expectedOpaquePalette[opaqueIndices[i]];
    assert.deepEqual(Array.from(opaqueRgba.subarray(i * 4, i * 4 + 4)), exp);
  }

  // 2) Punchthrough 3-color + transparent mode: color0 <= color1
  // color0 = pure blue (0x001F) <= color1 = pure white (0xFFFF)
  // Expected palette:
  //   0: (0, 0, 255, 255)
  //   1: (255, 255, 255, 255)
  //   2: (127, 127, 255, 255)  [(0+255)>>1, (0+255)>>1, (255+255)>>1]
  //   3: (0, 0, 0, 0)          [transparent black punchthrough]
  const alphaIndices = [
    0, 1, 2, 3,
    3, 3, 3, 3,
    2, 2, 1, 0,
    3, 0, 1, 2,
  ];
  const alphaBlock = encodeBc1Block(0x001f, 0xffff, alphaIndices);
  const alphaRgba = decodeBc1ToRgba8(alphaBlock, 4, 4);

  const expectedAlphaPalette = [
    [0, 0, 255, 255],
    [255, 255, 255, 255],
    [127, 127, 255, 255],
    [0, 0, 0, 0],
  ];
  for (let i = 0; i < 16; i += 1) {
    const exp = expectedAlphaPalette[alphaIndices[i]];
    assert.deepEqual(Array.from(alphaRgba.subarray(i * 4, i * 4 + 4)), exp);
  }

  // 3) Non-multiple-of-4 clamping (3x2 image from 1 BC1 block)
  const clipped = decodeBc1ToRgba8(opaqueBlock, 3, 2);
  assert.equal(clipped.byteLength, 3 * 2 * 4);
  // Row 0: indices [0, 1, 2]
  assert.deepEqual(Array.from(clipped.subarray(0, 4)), [255, 0, 0, 255]);
  assert.deepEqual(Array.from(clipped.subarray(4, 8)), [0, 0, 255, 255]);
  assert.deepEqual(Array.from(clipped.subarray(8, 12)), [170, 0, 85, 255]);
  // Row 1: indices [3, 2, 1]
  assert.deepEqual(Array.from(clipped.subarray(12, 16)), [85, 0, 170, 255]);
  assert.deepEqual(Array.from(clipped.subarray(16, 20)), [170, 0, 85, 255]);
  assert.deepEqual(Array.from(clipped.subarray(20, 24)), [0, 0, 255, 255]);

  // Truncated input throws RangeError
  assert.throws(() => decodeBc1ToRgba8(new Uint8Array(7), 4, 4), RangeError);
});

test('decodeBc3ToRgba8 decodes 8-level and 6-level interpolated alpha plus 4-color RGB565', () => {
  // Block 0 (top-left): 8-level alpha (alpha0 = 210 > alpha1 = 0 -> step of 30)
  //   alphaTable = [210, 0, 180, 150, 120, 90, 60, 30]
  //   color0 = 0x0000 <= color1 = 0xFFFF -> BC3 ALWAYS uses 4-color interpolation (NO punchthrough!)
  //   RGB palette = [0, 255, 85, 170] across R, G, B
  const alphaIdx8 = [
    0, 1, 2, 3,
    4, 5, 6, 7,
    7, 6, 5, 4,
    3, 2, 1, 0,
  ];
  const colorIdx = [
    0, 1, 2, 3,
    0, 1, 2, 3,
    3, 2, 1, 0,
    3, 2, 1, 0,
  ];
  const block0 = encodeBc3Block(210, 0, alphaIdx8, 0x0000, 0xffff, colorIdx);
  const rgba0 = decodeBc3ToRgba8(block0, 4, 4);
  assert.equal(rgba0.byteLength, 64);

  const expectedAlpha8 = [210, 0, 180, 150, 120, 90, 60, 30];
  const expectedGray = [0, 255, 85, 170];
  for (let i = 0; i < 16; i += 1) {
    const g = expectedGray[colorIdx[i]];
    const a = expectedAlpha8[alphaIdx8[i]];
    assert.deepEqual(Array.from(rgba0.subarray(i * 4, i * 4 + 4)), [g, g, g, a]);
  }

  // Block 1: 6-level alpha + explicit 0 & 255 (alpha0 = 50 <= alpha1 = 200 -> step of 30)
  //   alphaTable = [50, 200, 80, 110, 140, 170, 0, 255]
  //   color0 = 0x07E0 (pure green: 0,255,0), color1 = 0x07E0 (pure green)
  const alphaIdx6 = [
    0, 1, 2, 3,
    4, 5, 6, 7,
    6, 7, 0, 1,
    2, 3, 4, 5,
  ];
  const greenIndices = new Array(16).fill(0);
  const block1 = encodeBc3Block(50, 200, alphaIdx6, 0x07e0, 0x07e0, greenIndices);
  const rgba1 = decodeBc3ToRgba8(block1, 4, 4);

  const expectedAlpha6 = [50, 200, 80, 110, 140, 170, 0, 255];
  for (let i = 0; i < 16; i += 1) {
    assert.deepEqual(Array.from(rgba1.subarray(i * 4, i * 4 + 4)), [
      0,
      255,
      0,
      expectedAlpha6[alphaIdx6[i]],
    ]);
  }

  assert.throws(() => decodeBc3ToRgba8(new Uint8Array(15), 4, 4), RangeError);
});

test('decodeBc4ToR8 decodes 8-level/6-level red channels and clamps non-power-of-two dimensions', () => {
  // Create a 5x3 (non-power-of-two, 2x1 blocks) BC4 texture:
  //   Block (0, 0): 8-level mode red0=210 > red1=0 -> [210, 0, 180, 150, 120, 90, 60, 30]
  //   Block (1, 0): 6-level mode red0=50 <= red1=200 -> [50, 200, 80, 110, 140, 170, 0, 255]
  // Only x=0..4 and y=0..2 should appear in the 5x3 output (15 bytes).
  const b0Indices = [
    0, 1, 2, 3,
    4, 5, 6, 7,
    2, 4, 6, 0,
    7, 7, 7, 7, // row 3 clipped (y=3 >= height=3)
  ];
  const b1Indices = [
    6, 1, 2, 3, // only px=0 (x=4) survives -> index 6 -> 0
    7, 0, 0, 0, // only px=0 (x=4) survives -> index 7 -> 255
    3, 5, 5, 5, // only px=0 (x=4) survives -> index 3 -> 110
    1, 1, 1, 1, // row 3 clipped
  ];

  const blocks = new Uint8Array(16);
  blocks.set(encodeBc4Block(210, 0, b0Indices), 0);
  blocks.set(encodeBc4Block(50, 200, b1Indices), 8);

  // Also test reading from a subarray with non-zero byteOffset
  const outer = new Uint8Array(32);
  outer.set(blocks, 7);
  const sub = outer.subarray(7, 23);

  const decoded = decodeBc4ToR8(sub, 5, 3);
  assert.equal(decoded.byteLength, 15);
  assert.deepEqual(Array.from(decoded), [
    // y = 0: block0 [0,1,2,3] + block1 [6]
    210, 0, 180, 150, 0,
    // y = 1: block0 [4,5,6,7] + block1 [7]
    120, 90, 60, 30, 255,
    // y = 2: block0 [2,4,6,0] + block1 [3]
    180, 120, 60, 210, 110,
  ]);

  // 1x1 edge case
  const single = decodeBc4ToR8(encodeBc4Block(210, 0, b0Indices), 1, 1);
  assert.deepEqual(Array.from(single), [210]);

  // Truncated input throws RangeError
  assert.throws(() => decodeBc4ToR8(new Uint8Array(15), 5, 3), RangeError);
});

test('getSrgbSiblingFormat maps linear <-> sRGB pairs bidirectionally and rejects non-sRGB/invalid inputs', () => {
  const pairs = [
    ['rgba8unorm', 'rgba8unorm-srgb'],
    ['bgra8unorm', 'bgra8unorm-srgb'],
    ['bc1-rgba-unorm', 'bc1-rgba-unorm-srgb'],
    ['bc2-rgba-unorm', 'bc2-rgba-unorm-srgb'],
    ['bc3-rgba-unorm', 'bc3-rgba-unorm-srgb'],
    ['bc7-rgba-unorm', 'bc7-rgba-unorm-srgb'],
  ];

  for (const [linear, srgb] of pairs) {
    assert.equal(getSrgbSiblingFormat(linear), srgb);
    assert.equal(getSrgbSiblingFormat(srgb), linear);
    assert.equal(getSrgbSiblingFormat(getSrgbSiblingFormat(linear)), linear);
  }

  // Case/whitespace tolerance
  assert.equal(getSrgbSiblingFormat(' RGBA8UNORM '), 'rgba8unorm-srgb');
  assert.equal(getSrgbSiblingFormat('BC7-RGBA-UNORM-SRGB'), 'bc7-rgba-unorm');

  // Formats without an sRGB sibling return null
  const noSibling = [
    'rgba16float',
    'rgb10a2unorm',
    'r8unorm',
    'rg8unorm',
    'rgba8snorm',
    'rgba8uint',
    'r16unorm',
    'bc4-r-unorm',
    'bc4-r-snorm',
    'bc5-rg-unorm',
    'bc5-rg-snorm',
    'bc6h-rgb-ufloat',
    'bc6h-rgb-float',
    'depth24plus',
    'depth32float-stencil8',
    '',
    '   ',
    'toString',
    'constructor',
    '__proto__',
    'not-a-format',
  ];
  for (const fmt of noSibling) {
    assert.equal(getSrgbSiblingFormat(fmt), null, `expected null for ${JSON.stringify(fmt)}`);
  }

  // Non-string inputs return null
  for (const bad of [null, undefined, 0, 28, {}, [], true]) {
    assert.equal(getSrgbSiblingFormat(bad), null);
  }
});

test('classifyTextureFormatSupport handles core WebGPU formats unconditionally', () => {
  const coreFormats = [
    'rgba8unorm',
    'rgba8unorm-srgb',
    'bgra8unorm',
    'bgra8unorm-srgb',
    'rgba16float',
    'rgba32float',
    'rgb10a2unorm',
    'rg11b10ufloat',
    'r8unorm',
    'rg8unorm',
    'r16float',
    'rg16float',
    'r32float',
    'depth16unorm',
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
    'stencil8',
  ];

  for (const fmt of coreFormats) {
    assert.deepEqual(classifyTextureFormatSupport(fmt), {
      supported: true,
      requiredFeature: null,
      fallbackFormat: null,
      reason: null,
    });
    assert.deepEqual(
      classifyTextureFormatSupport(fmt, { dimension: '3d', deviceFeatures: new Set() }),
      {
        supported: true,
        requiredFeature: null,
        fallbackFormat: null,
        reason: null,
      }
    );
  }

  // Invalid inputs return unsupported with invalid-format reason
  for (const bad of [null, undefined, '', '   ', 42, {}]) {
    assert.deepEqual(classifyTextureFormatSupport(bad), {
      supported: false,
      requiredFeature: null,
      fallbackFormat: null,
      reason: 'invalid-format',
    });
  }
});

test('classifyTextureFormatSupport handles 2D/1D BC formats across Set, GPUSupportedFeatures-like, and Array feature collections', () => {
  const bcFormats = [
    'bc1-rgba-unorm',
    'bc1-rgba-unorm-srgb',
    'bc2-rgba-unorm',
    'bc2-rgba-unorm-srgb',
    'bc3-rgba-unorm',
    'bc3-rgba-unorm-srgb',
    'bc4-r-unorm',
    'bc4-r-snorm',
    'bc5-rg-unorm',
    'bc5-rg-snorm',
    'bc6h-rgb-ufloat',
    'bc6h-rgb-float',
    'bc7-rgba-unorm',
    'bc7-rgba-unorm-srgb',
  ];

  const gpuSupportedFeaturesMock = {
    has(name) {
      return name === TEXTURE_COMPRESSION_BC_FEATURE;
    },
  };

  for (const fmt of bcFormats) {
    // Missing features (omitted options, null, empty Set, empty Array)
    for (const opts of [
      undefined,
      {},
      { dimension: '2d', deviceFeatures: null },
      { dimension: '1d', deviceFeatures: new Set() },
      { dimension: '2d', deviceFeatures: [] },
    ]) {
      assert.deepEqual(classifyTextureFormatSupport(fmt, opts), {
        supported: false,
        requiredFeature: TEXTURE_COMPRESSION_BC_FEATURE,
        fallbackFormat: 'rgba8unorm',
        reason: 'missing-bc',
      });
    }

    // Feature present via Set, GPUSupportedFeatures-like object, and Array
    for (const features of [
      new Set([TEXTURE_COMPRESSION_BC_FEATURE]),
      gpuSupportedFeaturesMock,
      [TEXTURE_COMPRESSION_BC_FEATURE],
    ]) {
      assert.deepEqual(classifyTextureFormatSupport(fmt, { dimension: '2d', deviceFeatures: features }), {
        supported: true,
        requiredFeature: TEXTURE_COMPRESSION_BC_FEATURE,
        fallbackFormat: null,
        reason: null,
      });
      assert.deepEqual(classifyTextureFormatSupport(fmt, { dimension: '1d', deviceFeatures: features }), {
        supported: true,
        requiredFeature: TEXTURE_COMPRESSION_BC_FEATURE,
        fallbackFormat: null,
        reason: null,
      });
    }
  }
});

test('classifyTextureFormatSupport distinguishes 3D BC sliced feature from 2D BC', () => {
  const bcFormats = [
    'bc1-rgba-unorm',
    'bc3-rgba-unorm-srgb',
    'bc4-r-unorm',
    'bc5-rg-snorm',
    'bc6h-rgb-ufloat',
    'bc7-rgba-unorm',
  ];

  // Having only 2D BC is NOT sufficient for 3D BC
  const only2dBc = new Set([TEXTURE_COMPRESSION_BC_FEATURE]);
  const with3dBcSet = new Set([TEXTURE_COMPRESSION_BC_FEATURE, TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE]);
  const with3dBcArray = [TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE];
  const with3dBcHas = {
    has(name) {
      return name === TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE;
    },
  };

  for (const fmt of bcFormats) {
    assert.deepEqual(
      classifyTextureFormatSupport(fmt, { dimension: '3d', deviceFeatures: only2dBc }),
      {
        supported: false,
        requiredFeature: TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE,
        fallbackFormat: 'rgba8unorm',
        reason: 'missing-bc-sliced-3d',
      }
    );

    assert.deepEqual(
      classifyTextureFormatSupport(fmt, { dimension: '3d', deviceFeatures: null }),
      {
        supported: false,
        requiredFeature: TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE,
        fallbackFormat: 'rgba8unorm',
        reason: 'missing-bc-sliced-3d',
      }
    );

    for (const features of [with3dBcSet, with3dBcArray, with3dBcHas]) {
      assert.deepEqual(
        classifyTextureFormatSupport(fmt, { dimension: '3d', deviceFeatures: features }),
        {
          supported: true,
          requiredFeature: TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE,
          fallbackFormat: null,
          reason: null,
        }
      );
    }
  }
});

test('classifyTextureFormatSupport handles 16-bit unorm/snorm with tier1 or chromium experimental feature and float16 fallbacks', () => {
  const cases = [
    ['r16unorm', 'r16float'],
    ['r16snorm', 'r16float'],
    ['rg16unorm', 'rg16float'],
    ['rg16snorm', 'rg16float'],
    ['rgba16unorm', 'rgba16float'],
    ['rgba16snorm', 'rgba16float'],
  ];

  const tier1Set = new Set([TEXTURE_FORMATS_TIER1_FEATURE]);
  const chromiumArr = ['chromium-experimental-unorm16-texture-formats'];
  const bothSet = new Set([TEXTURE_FORMATS_TIER1_FEATURE, 'chromium-experimental-unorm16-texture-formats']);

  for (const [fmt, expectedFallback] of cases) {
    // Absent -> float16 fallback matching channel count
    assert.deepEqual(classifyTextureFormatSupport(fmt, { deviceFeatures: new Set() }), {
      supported: false,
      requiredFeature: TEXTURE_FORMATS_TIER1_FEATURE,
      fallbackFormat: expectedFallback,
      reason: 'missing-unorm16',
    });

    // Supported via standard texture-formats-tier1
    assert.deepEqual(classifyTextureFormatSupport(fmt, { deviceFeatures: tier1Set }), {
      supported: true,
      requiredFeature: TEXTURE_FORMATS_TIER1_FEATURE,
      fallbackFormat: null,
      reason: null,
    });

    // Supported via chromium-experimental-unorm16-texture-formats
    assert.deepEqual(classifyTextureFormatSupport(fmt, { deviceFeatures: chromiumArr }), {
      supported: true,
      requiredFeature: 'chromium-experimental-unorm16-texture-formats',
      fallbackFormat: null,
      reason: null,
    });

    // Prefers canonical texture-formats-tier1 when both are present
    assert.deepEqual(classifyTextureFormatSupport(fmt, { deviceFeatures: bothSet }), {
      supported: true,
      requiredFeature: TEXTURE_FORMATS_TIER1_FEATURE,
      fallbackFormat: null,
      reason: null,
    });
  }
});

test('classifyTextureFormatSupport handles depth32float-stencil8 with depth24plus-stencil8 fallback', () => {
  assert.deepEqual(classifyTextureFormatSupport('depth32float-stencil8'), {
    supported: false,
    requiredFeature: DEPTH32FLOAT_STENCIL8_FEATURE,
    fallbackFormat: 'depth24plus-stencil8',
    reason: 'missing-depth32float-stencil8',
  });

  assert.deepEqual(
    classifyTextureFormatSupport('depth32float-stencil8', { deviceFeatures: new Set(['texture-compression-bc']) }),
    {
      supported: false,
      requiredFeature: DEPTH32FLOAT_STENCIL8_FEATURE,
      fallbackFormat: 'depth24plus-stencil8',
      reason: 'missing-depth32float-stencil8',
    }
  );

  for (const features of [
    new Set([DEPTH32FLOAT_STENCIL8_FEATURE]),
    [DEPTH32FLOAT_STENCIL8_FEATURE],
    { has: (f) => f === DEPTH32FLOAT_STENCIL8_FEATURE },
  ]) {
    assert.deepEqual(
      classifyTextureFormatSupport('depth32float-stencil8', { deviceFeatures: features }),
      {
        supported: true,
        requiredFeature: DEPTH32FLOAT_STENCIL8_FEATURE,
        fallbackFormat: null,
        reason: null,
      }
    );
  }
});

test('W163 DXGI_FORMAT includes Xbox Series UI, font, LUT, and video formats with official numeric codes', () => {
  const expectedCodes = [
    ['R32G32B32A32_UINT', 3],
    ['R16G16B16A16_UNORM', 11],
    ['R16G16B16A16_UINT', 12],
    ['R16G16B16A16_SNORM', 13],
    ['R32G32_FLOAT', 16],
    ['R32G32_UINT', 17],
    ['R16G16_UNORM', 35],
    ['R16G16_UINT', 36],
    ['R8G8_UINT', 50],
    ['R16_UNORM', 56],
    ['R16_UINT', 57],
    ['R8_UINT', 62],
    ['R8_SNORM', 63],
    ['A8_UNORM', 65],
  ];

  for (const [name, code] of expectedCodes) {
    assert.equal(DXGI_FORMAT[name], code, `DXGI_FORMAT.${name}`);
    assert.equal(DXGI_FORMAT_NAMES[code], name, `DXGI_FORMAT_NAMES[${code}]`);
  }
});

test('W163 mapDxgiFormatToWebGpu maps A8_UNORM font/mask atlas format with alpha-only swizzle metadata', () => {
  const byCode = mapDxgiFormatToWebGpu(DXGI_FORMAT.A8_UNORM);
  assert.equal(byCode.dxgiFormat, 65);
  assert.equal(byCode.name, 'A8_UNORM');
  assert.equal(byCode.webgpuFormat, 'r8unorm');
  assert.equal(byCode.nativeWebgpuFormat, 'r8unorm');
  assert.equal(byCode.blockWidth, 1);
  assert.equal(byCode.blockHeight, 1);
  assert.equal(byCode.bytesPerBlock, 1);
  assert.equal(byCode.isCompressed, false);
  assert.equal(byCode.isDepth, false);
  assert.equal(byCode.isSrgb, false);
  assert.equal(byCode.isAlphaOnly, true);
  assert.equal(byCode.swizzle, '000r');
  assert.equal(byCode.requiresFeature, null);
  assert.equal(byCode.needsSoftwareDecode, false);

  assert.deepEqual(mapDxgiFormatToWebGpu('A8_UNORM'), byCode);
  assert.deepEqual(mapDxgiFormatToWebGpu('DXGI_FORMAT_A8_UNORM'), byCode);

  // Non-alpha single-channel R8_UNORM keeps isAlphaOnly=false and swizzle=null
  const r8 = mapDxgiFormatToWebGpu(DXGI_FORMAT.R8_UNORM);
  assert.equal(r8.isAlphaOnly, false);
  assert.equal(r8.swizzle, null);

  // 256-byte row alignment for 100x32 A8 font atlas glyph strip
  assert.deepEqual(calcMipLayout(DXGI_FORMAT.A8_UNORM, 100, 32), {
    width: 100,
    height: 32,
    depthOrLayers: 1,
    blocksX: 100,
    blocksY: 32,
    unalignedBytesPerRow: 100,
    bytesPerRow: 256,
    rowsPerImage: 32,
    totalBytes: 256 * 32,
  });
});

test('W163 mapDxgiFormatToWebGpu maps core integer, snorm, and 64/128-bit UI/LUT/video formats', () => {
  const coreCases = [
    [DXGI_FORMAT.R8_SNORM, 'r8snorm', 1],
    [DXGI_FORMAT.R8_UINT, 'r8uint', 1],
    [DXGI_FORMAT.R8G8_UINT, 'rg8uint', 2],
    [DXGI_FORMAT.R16_UINT, 'r16uint', 2],
    [DXGI_FORMAT.R16G16_UINT, 'rg16uint', 4],
    [DXGI_FORMAT.R16G16B16A16_UINT, 'rgba16uint', 8],
    [DXGI_FORMAT.R32G32_FLOAT, 'rg32float', 8],
    [DXGI_FORMAT.R32G32_UINT, 'rg32uint', 8],
    [DXGI_FORMAT.R32G32B32A32_UINT, 'rgba32uint', 16],
  ];

  for (const [dxgi, expectedFormat, bytesPerBlock] of coreCases) {
    const info = mapDxgiFormatToWebGpu(dxgi);
    assert.equal(info.webgpuFormat, expectedFormat);
    assert.equal(info.nativeWebgpuFormat, expectedFormat);
    assert.equal(info.blockWidth, 1);
    assert.equal(info.blockHeight, 1);
    assert.equal(info.bytesPerBlock, bytesPerBlock);
    assert.equal(info.isCompressed, false);
    assert.equal(info.isDepth, false);
    assert.equal(info.isSrgb, false);
    assert.equal(info.isAlphaOnly, false);
    assert.equal(info.swizzle, null);
    assert.equal(info.requiresFeature, null);
    assert.equal(info.needsSoftwareDecode, false);

    // Mip layout calculation works directly from DXGI code
    const layout = calcMipLayout(dxgi, 17, 5);
    assert.equal(layout.unalignedBytesPerRow, 17 * bytesPerBlock);
    assert.equal(layout.bytesPerRow % WEBGPU_BYTES_PER_ROW_ALIGNMENT, 0);
    assert.ok(layout.bytesPerRow >= layout.unalignedBytesPerRow);
    assert.equal(layout.totalBytes, layout.bytesPerRow * 5);
  }
});

test('W163 mapDxgiFormatToWebGpu maps 16-bit unorm/snorm formats with tier1/chromium feature and float16 fallback', () => {
  const norm16Cases = [
    [DXGI_FORMAT.R16_UNORM, 'r16unorm', 'r16float', 2],
    [DXGI_FORMAT.R16G16_UNORM, 'rg16unorm', 'rg16float', 4],
    [DXGI_FORMAT.R16G16B16A16_UNORM, 'rgba16unorm', 'rgba16float', 8],
    [DXGI_FORMAT.R16G16B16A16_SNORM, 'rgba16snorm', 'rgba16float', 8],
  ];

  const tier1 = new Set([TEXTURE_FORMATS_TIER1_FEATURE]);
  const chromium = [CHROMIUM_UNORM16_FEATURE];
  const empty = new Set();

  for (const [dxgi, nativeFmt, fallbackFmt, bytesPerBlock] of norm16Cases) {
    const withTier1 = mapDxgiFormatToWebGpu(dxgi, tier1);
    assert.equal(withTier1.webgpuFormat, nativeFmt);
    assert.equal(withTier1.nativeWebgpuFormat, nativeFmt);
    assert.equal(withTier1.bytesPerBlock, bytesPerBlock);
    assert.equal(withTier1.requiresFeature, TEXTURE_FORMATS_TIER1_FEATURE);
    assert.equal(withTier1.needsSoftwareDecode, false);

    const withChromium = mapDxgiFormatToWebGpu(dxgi, chromium);
    assert.equal(withChromium.webgpuFormat, nativeFmt);
    assert.equal(withChromium.nativeWebgpuFormat, nativeFmt);

    const withoutFeature = mapDxgiFormatToWebGpu(dxgi, empty);
    assert.equal(withoutFeature.webgpuFormat, fallbackFmt);
    assert.equal(withoutFeature.nativeWebgpuFormat, nativeFmt);
    assert.equal(withoutFeature.bytesPerBlock, bytesPerBlock);
    assert.equal(withoutFeature.requiresFeature, TEXTURE_FORMATS_TIER1_FEATURE);
    assert.equal(withoutFeature.needsSoftwareDecode, false);

    // Omitting supportedFeatures falls back cleanly too
    const defaultMapped = mapDxgiFormatToWebGpu(dxgi);
    assert.equal(defaultMapped.webgpuFormat, fallbackFmt);
    assert.equal(defaultMapped.nativeWebgpuFormat, nativeFmt);

    // Matches classifyTextureFormatSupport fallback
    assert.equal(
      classifyTextureFormatSupport(nativeFmt, { deviceFeatures: empty }).fallbackFormat,
      fallbackFmt
    );
  }
});

test('W163 resolveStorageCompatibleFormat handles core WebGPU storage formats directly', () => {
  const coreStorageFormats = [
    'rgba8unorm',
    'rgba8snorm',
    'rgba8uint',
    'rgba8sint',
    'rgba16uint',
    'rgba16sint',
    'rgba16float',
    'r32uint',
    'r32sint',
    'r32float',
    'rg32uint',
    'rg32sint',
    'rg32float',
    'rgba32uint',
    'rgba32sint',
    'rgba32float',
  ];

  for (const fmt of coreStorageFormats) {
    assert.deepEqual(resolveStorageCompatibleFormat(fmt), {
      storageFormat: fmt,
      supportsStorageBinding: true,
      needsLinearSiblingView: false,
      reason: null,
    });
  }

  // Case and whitespace normalization
  assert.deepEqual(resolveStorageCompatibleFormat('  RGBA16FLOAT '), {
    storageFormat: 'rgba16float',
    supportsStorageBinding: true,
    needsLinearSiblingView: false,
    reason: null,
  });
});

test('W163 resolveStorageCompatibleFormat maps sRGB UAV formats to linear siblings or rejects safely', () => {
  // rgba8unorm-srgb -> linear sibling rgba8unorm view
  assert.deepEqual(resolveStorageCompatibleFormat('rgba8unorm-srgb'), {
    storageFormat: 'rgba8unorm',
    supportsStorageBinding: true,
    needsLinearSiblingView: true,
    reason: 'srgb-requires-linear-view',
  });
  assert.deepEqual(resolveStorageCompatibleFormat(' RGBA8UNORM-SRGB '), {
    storageFormat: 'rgba8unorm',
    supportsStorageBinding: true,
    needsLinearSiblingView: true,
    reason: 'srgb-requires-linear-view',
  });

  // bgra8unorm-srgb and bgra8unorm require 'bgra8unorm-storage'
  assert.equal(BGRA8UNORM_STORAGE_FEATURE, 'bgra8unorm-storage');
  assert.deepEqual(resolveStorageCompatibleFormat('bgra8unorm-srgb'), {
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'missing-bgra8unorm-storage',
  });
  assert.deepEqual(resolveStorageCompatibleFormat('bgra8unorm'), {
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'missing-bgra8unorm-storage',
  });

  // When 'bgra8unorm-storage' is enabled, both bgra8unorm and bgra8unorm-srgb resolve cleanly
  const bgraFeatureVariants = [
    new Set([BGRA8UNORM_STORAGE_FEATURE]),
    [BGRA8UNORM_STORAGE_FEATURE],
    { has: (f) => f === BGRA8UNORM_STORAGE_FEATURE },
    { deviceFeatures: new Set([BGRA8UNORM_STORAGE_FEATURE]) },
    { supportedFeatures: [BGRA8UNORM_STORAGE_FEATURE] },
  ];
  for (const features of bgraFeatureVariants) {
    assert.deepEqual(resolveStorageCompatibleFormat('bgra8unorm', features), {
      storageFormat: 'bgra8unorm',
      supportsStorageBinding: true,
      needsLinearSiblingView: false,
      reason: null,
    });
    assert.deepEqual(resolveStorageCompatibleFormat('bgra8unorm-srgb', features), {
      storageFormat: 'bgra8unorm',
      supportsStorageBinding: true,
      needsLinearSiblingView: true,
      reason: 'srgb-requires-linear-view',
    });
  }

  // Unknown non-storage sRGB format
  assert.deepEqual(resolveStorageCompatibleFormat('r8unorm-srgb'), {
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'srgb-not-storage-compatible',
  });
});

test('W163 resolveStorageCompatibleFormat rejects depth/stencil, compressed, optional tier1 without feature, and invalid formats', () => {
  const depthStencilFormats = [
    'depth16unorm',
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
    'depth32float-stencil8',
    'stencil8',
  ];
  for (const fmt of depthStencilFormats) {
    assert.deepEqual(resolveStorageCompatibleFormat(fmt), {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'depth-stencil-not-storage-compatible',
    });
  }

  const compressedFormats = [
    'bc1-rgba-unorm',
    'bc1-rgba-unorm-srgb',
    'bc2-rgba-unorm',
    'bc3-rgba-unorm-srgb',
    'bc4-r-unorm',
    'bc5-rg-unorm',
    'bc6h-rgb-ufloat',
    'bc7-rgba-unorm',
    'bc7-rgba-unorm-srgb',
    'etc2-rgb8unorm-srgb',
    'eac-r11unorm',
    'astc-4x4-unorm-srgb',
  ];
  for (const fmt of compressedFormats) {
    assert.deepEqual(resolveStorageCompatibleFormat(fmt), {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'compressed-not-storage-compatible',
    });
  }

  // Tier-1 storage formats require 'texture-formats-tier1'
  const tier1Formats = [
    'r8unorm',
    'r8snorm',
    'r8uint',
    'rg8unorm',
    'rg8uint',
    'r16unorm',
    'r16uint',
    'r16float',
    'rg16unorm',
    'rg16uint',
    'rg16float',
    'rgba16unorm',
    'rgba16snorm',
    'rgb10a2unorm',
    'rg11b10ufloat',
  ];
  for (const fmt of tier1Formats) {
    assert.deepEqual(resolveStorageCompatibleFormat(fmt), {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'missing-texture-formats-tier1',
    });
    assert.deepEqual(
      resolveStorageCompatibleFormat(fmt, { deviceFeatures: new Set([TEXTURE_FORMATS_TIER1_FEATURE]) }),
      {
        storageFormat: fmt,
        supportsStorageBinding: true,
        needsLinearSiblingView: false,
        reason: null,
      }
    );
  }

  // Unsupported / non-storage core formats
  for (const fmt of ['rgb9e5ufloat', 'not-a-format', 'toString', '__proto__']) {
    assert.deepEqual(resolveStorageCompatibleFormat(fmt), {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'unsupported-storage-format',
    });
  }

  // Invalid inputs
  for (const bad of [null, undefined, '', '   ', 0, 28, {}, []]) {
    assert.deepEqual(resolveStorageCompatibleFormat(bad), {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'invalid-format',
    });
  }
});

test('W301 DXGI_FORMAT_TABLE is a frozen dense O(1) table for DXGI ids 0..132 with zero-alloc lookups', () => {
  assert.equal(DXGI_FORMAT_MAX_ID, 132);
  assert.ok(Array.isArray(DXGI_FORMAT_TABLE));
  assert.equal(DXGI_FORMAT_TABLE.length, 133);
  assert.equal(Object.isFrozen(DXGI_FORMAT_TABLE), true);
  assert.equal(DXGI_FORMAT_TABLE[0], null);

  for (let i = 0; i <= DXGI_FORMAT_MAX_ID; i += 1) {
    assert.equal(Object.prototype.hasOwnProperty.call(DXGI_FORMAT_TABLE, i), true, `dense slot ${i}`);
    const entry = DXGI_FORMAT_TABLE[i];
    if (entry !== null) {
      assert.equal(Object.isFrozen(entry), true, `frozen entry at ${i}`);
      assert.equal(entry.dxgiFormat, i);
      assert.equal(entry.name, DXGI_FORMAT_NAMES[i]);
      assert.ok(typeof entry.webgpuFormat === 'string' && entry.webgpuFormat.length > 0);
      assert.ok(entry.aspect === 'all' || entry.aspect === 'depth-only' || entry.aspect === 'stencil-only');
    }
  }

  // Zero-alloc identity: repeated mapDxgiFormatToWebGpu / lookupDxgiFormat calls return the exact same frozen object
  const withAllFeatures = new Set([
    TEXTURE_COMPRESSION_BC_FEATURE,
    DEPTH32FLOAT_STENCIL8_FEATURE,
    TEXTURE_FORMATS_TIER1_FEATURE,
  ]);
  const noFeatures = new Set();

  for (let code = 1; code <= DXGI_FORMAT_MAX_ID; code += 1) {
    if (DXGI_FORMAT_TABLE[code] === null) {
      assert.equal(lookupDxgiFormat(code), null);
      continue;
    }
    const a = mapDxgiFormatToWebGpu(code, withAllFeatures);
    const b = mapDxgiFormatToWebGpu(code, withAllFeatures);
    assert.equal(a, b, `native lookup for code ${code} must not allocate`);
    assert.equal(a, DXGI_FORMAT_TABLE[code], `native lookup for code ${code} returns dense table entry`);
    assert.equal(Object.isFrozen(a), true);

    const f1 = mapDxgiFormatToWebGpu(code, noFeatures);
    const f2 = mapDxgiFormatToWebGpu(code, noFeatures);
    assert.equal(f1, f2, `fallback lookup for code ${code} must not allocate`);
    assert.equal(Object.isFrozen(f1), true);

    assert.equal(lookupDxgiFormat(code, withAllFeatures), a);
    assert.equal(lookupDxgiFormat(code, noFeatures), f1);
    assert.equal(lookupDxgiFormat(a.name, withAllFeatures), a);
    assert.equal(lookupDxgiFormat(`DXGI_FORMAT_${a.name}`, withAllFeatures), a);
  }

  // Non-throwing lookupDxgiFormat returns null for invalid/unsupported inputs
  for (const bad of [0, -1, 133, 999, 28.5, NaN, Infinity, '', 'UNKNOWN', 'NO_SUCH_FORMAT', null, undefined, {}]) {
    assert.equal(lookupDxgiFormat(bad), null, `expected null for ${String(bad)}`);
  }

  // classifyTextureFormatSupport & resolveStorageCompatibleFormat also reuse frozen records
  assert.equal(classifyTextureFormatSupport('rgba16float'), classifyTextureFormatSupport('rgba16float'));
  assert.equal(
    classifyTextureFormatSupport('bc7-rgba-unorm', { deviceFeatures: withAllFeatures }),
    classifyTextureFormatSupport('bc7-rgba-unorm', { deviceFeatures: withAllFeatures })
  );
  assert.equal(
    resolveStorageCompatibleFormat('rgba16float'),
    resolveStorageCompatibleFormat('rgba16float')
  );
});

test('W301 complete HDR10, render-target, BC1-BC7, and depth/stencil metadata (aspect, sRGB, HDR, feature)', () => {
  const withAll = new Set([
    TEXTURE_COMPRESSION_BC_FEATURE,
    DEPTH32FLOAT_STENCIL8_FEATURE,
    TEXTURE_FORMATS_TIER1_FEATURE,
  ]);

  // HDR10 & render-target formats (including shorthand string aliases)
  const hdrCases = [
    [DXGI_FORMAT.R10G10B10A2_UNORM, 'R10G10B10A2_UNORM', 'rgb10a2unorm', 4],
    [DXGI_FORMAT.R11G11B10_FLOAT, 'R11G11B10_FLOAT', 'rg11b10ufloat', 4],
    [DXGI_FORMAT.R16G16B16A16_FLOAT, 'RGBA16_FLOAT', 'rgba16float', 8],
    [DXGI_FORMAT.R16G16_FLOAT, 'RG16_FLOAT', 'rg16float', 4],
    [DXGI_FORMAT.R16_FLOAT, 'R16_FLOAT', 'r16float', 2],
    [DXGI_FORMAT.R32G32B32A32_FLOAT, 'RGBA32_FLOAT', 'rgba32float', 16],
    [DXGI_FORMAT.R9G9B9E5_SHAREDEXP, 'R9G9B9E5_SHAREDEXP', 'rgb9e5ufloat', 4],
  ];

  for (const [code, alias, webgpuFormat, bytesPerBlock] of hdrCases) {
    const desc = mapDxgiFormatToWebGpu(code);
    assert.equal(desc.webgpuFormat, webgpuFormat);
    assert.equal(desc.blockWidth, 1);
    assert.equal(desc.blockHeight, 1);
    assert.equal(desc.bytesPerBlock, bytesPerBlock);
    assert.equal(desc.aspect, 'all');
    assert.equal(desc.isCompressed, false);
    assert.equal(desc.isDepth, false);
    assert.equal(desc.hasStencil, false);
    assert.equal(desc.isSrgb, false);
    assert.equal(desc.isHdr, true);
    assert.equal(desc.requiresFeature, null);
    assert.equal(desc.needsSoftwareDecode, false);
    assert.equal(mapDxgiFormatToWebGpu(alias), desc);
    assert.equal(mapDxgiFormatToWebGpu(`DXGI_FORMAT_${alias}`), desc);
    assert.equal(WEBGPU_FORMAT_INFO[webgpuFormat].isHdr, true);
    assert.equal(WEBGPU_FORMAT_INFO[webgpuFormat].aspect, 'all');
  }

  // BC1..BC7 complete metadata (unorm, sRGB, snorm, BC6H ufloat/float)
  const bcMetaCases = [
    [DXGI_FORMAT.BC1_UNORM, 'bc1-rgba-unorm', 8, false, false],
    [DXGI_FORMAT.BC1_UNORM_SRGB, 'bc1-rgba-unorm-srgb', 8, true, false],
    [DXGI_FORMAT.BC2_UNORM, 'bc2-rgba-unorm', 16, false, false],
    [DXGI_FORMAT.BC2_UNORM_SRGB, 'bc2-rgba-unorm-srgb', 16, true, false],
    [DXGI_FORMAT.BC3_UNORM, 'bc3-rgba-unorm', 16, false, false],
    [DXGI_FORMAT.BC3_UNORM_SRGB, 'bc3-rgba-unorm-srgb', 16, true, false],
    [DXGI_FORMAT.BC4_UNORM, 'bc4-r-unorm', 8, false, false],
    [DXGI_FORMAT.BC4_SNORM, 'bc4-r-snorm', 8, false, false],
    [DXGI_FORMAT.BC5_UNORM, 'bc5-rg-unorm', 16, false, false],
    [DXGI_FORMAT.BC5_SNORM, 'bc5-rg-snorm', 16, false, false],
    [DXGI_FORMAT.BC6H_UF16, 'bc6h-rgb-ufloat', 16, false, true],
    [DXGI_FORMAT.BC6H_SF16, 'bc6h-rgb-float', 16, false, true],
    [DXGI_FORMAT.BC7_UNORM, 'bc7-rgba-unorm', 16, false, false],
    [DXGI_FORMAT.BC7_UNORM_SRGB, 'bc7-rgba-unorm-srgb', 16, true, false],
  ];

  for (const [code, webgpuFormat, bytesPerBlock, isSrgb, isHdr] of bcMetaCases) {
    const desc = mapDxgiFormatToWebGpu(code, withAll);
    assert.equal(desc.webgpuFormat, webgpuFormat);
    assert.equal(desc.blockWidth, 4);
    assert.equal(desc.blockHeight, 4);
    assert.equal(desc.bytesPerBlock, bytesPerBlock);
    assert.equal(desc.aspect, 'all');
    assert.equal(desc.isCompressed, true);
    assert.equal(desc.isDepth, false);
    assert.equal(desc.hasStencil, false);
    assert.equal(desc.isSrgb, isSrgb);
    assert.equal(desc.isHdr, isHdr);
    assert.equal(desc.requiresFeature, TEXTURE_COMPRESSION_BC_FEATURE);
    assert.equal(WEBGPU_FORMAT_INFO[webgpuFormat].requiresFeature, TEXTURE_COMPRESSION_BC_FEATURE);
    assert.equal(WEBGPU_FORMAT_INFO[webgpuFormat].isHdr, isHdr);
  }

  // Depth & depth/stencil metadata (aspect, hasStencil, required feature & fallback)
  const depthCases = [
    [DXGI_FORMAT.D32_FLOAT, 'depth32float', 4, 'depth-only', false, null],
    [DXGI_FORMAT.D16_UNORM, 'depth16unorm', 2, 'depth-only', false, null],
    [DXGI_FORMAT.D24_UNORM_S8_UINT, 'depth24plus-stencil8', 4, 'all', true, null],
    [DXGI_FORMAT.D32_FLOAT_S8X24_UINT, 'depth32float-stencil8', 8, 'all', true, DEPTH32FLOAT_STENCIL8_FEATURE],
    [DXGI_FORMAT.R32_FLOAT_X8X24_TYPELESS, 'depth32float-stencil8', 8, 'depth-only', true, DEPTH32FLOAT_STENCIL8_FEATURE],
    [DXGI_FORMAT.X32_TYPELESS_G8X24_UINT, 'depth32float-stencil8', 8, 'stencil-only', true, DEPTH32FLOAT_STENCIL8_FEATURE],
    [DXGI_FORMAT.R24_UNORM_X8_TYPELESS, 'depth24plus-stencil8', 4, 'depth-only', true, null],
    [DXGI_FORMAT.X24_TYPELESS_G8_UINT, 'depth24plus-stencil8', 4, 'stencil-only', true, null],
  ];

  for (const [code, webgpuFormat, bytesPerBlock, aspect, hasStencil, reqFeature] of depthCases) {
    const desc = mapDxgiFormatToWebGpu(code, withAll);
    assert.equal(desc.webgpuFormat, webgpuFormat);
    assert.equal(desc.bytesPerBlock, bytesPerBlock);
    assert.equal(desc.aspect, aspect);
    assert.equal(desc.isDepth, true);
    assert.equal(desc.hasStencil, hasStencil);
    assert.equal(desc.isSrgb, false);
    assert.equal(desc.requiresFeature, reqFeature);
  }

  // D32_FLOAT_S8X24_UINT falls back cleanly to depth24plus-stencil8 when feature is absent
  const d32s8Fallback = mapDxgiFormatToWebGpu(DXGI_FORMAT.D32_FLOAT_S8X24_UINT, new Set());
  assert.equal(d32s8Fallback.webgpuFormat, 'depth24plus-stencil8');
  assert.equal(d32s8Fallback.nativeWebgpuFormat, 'depth32float-stencil8');
  assert.equal(d32s8Fallback.aspect, 'all');
  assert.equal(d32s8Fallback.hasStencil, true);
  assert.equal(d32s8Fallback.requiresFeature, DEPTH32FLOAT_STENCIL8_FEATURE);

  // WEBGPU_FORMAT_INFO depth/stencil aspects
  assert.equal(getWebGpuFormatInfo('depth32float').aspect, 'depth-only');
  assert.equal(getWebGpuFormatInfo('depth16unorm').aspect, 'depth-only');
  assert.equal(getWebGpuFormatInfo('depth24plus-stencil8').aspect, 'all');
  assert.equal(getWebGpuFormatInfo('depth32float-stencil8').aspect, 'all');
  assert.equal(getWebGpuFormatInfo('stencil8').aspect, 'stencil-only');
});

test('W301 computeMipLayout and computeMipByteSize handle BC 4x4 rounding, 256B alignment, out reuse, and degenerate dims safely', () => {
  // BC7 256x128 mip 0: 64x32 blocks, 64*16 = 1024B/row (exact 256B multiple)
  const bc7M0 = computeMipLayout(DXGI_FORMAT.BC7_UNORM_SRGB, 256, 128, 0);
  assert.deepEqual(bc7M0, {
    width: 256,
    height: 128,
    mipLevel: 0,
    copyWidth: 256,
    copyHeight: 128,
    blocksX: 64,
    blocksY: 32,
    blockWidth: 4,
    blockHeight: 4,
    bytesPerBlock: 16,
    unalignedBytesPerRow: 1024,
    bytesPerRow: 1024,
    rowsPerImage: 32,
    tightBytes: 1024 * 32,
    totalBytes: 1024 * 32,
    byteSize: 1024 * 32,
  });
  assert.equal(computeMipByteSize(DXGI_FORMAT.BC7_UNORM_SRGB, 256, 128, 0), 1024 * 32);

  // BC7 tail mip 7 on 256x128 -> 2x1 pixels -> rounded to 1x1 block (4x4 copy extent), 16B -> 256B/row
  const bc7Tail = computeMipLayout('bc7-rgba-unorm-srgb', 256, 128, 7);
  assert.equal(bc7Tail.width, 2);
  assert.equal(bc7Tail.height, 1);
  assert.equal(bc7Tail.copyWidth, 4);
  assert.equal(bc7Tail.copyHeight, 4);
  assert.equal(bc7Tail.blocksX, 1);
  assert.equal(bc7Tail.blocksY, 1);
  assert.equal(bc7Tail.unalignedBytesPerRow, 16);
  assert.equal(bc7Tail.bytesPerRow, 256);
  assert.equal(bc7Tail.tightBytes, 16);
  assert.equal(bc7Tail.totalBytes, 256);
  assert.equal(computeMipByteSize('bc7-rgba-unorm-srgb', 256, 128, 7), 256);
  assert.equal(computeMipByteSize('bc7-rgba-unorm-srgb', 256, 128, 7, 1, false), 16);

  // BC1 non-multiple-of-4 mip: 40x24 mip 2 -> 10x6 -> 3x2 blocks (12x8 copy), 24B -> 256B/row, 512B total
  const bc1M2 = computeMipLayout(DXGI_FORMAT.BC1_UNORM, 40, 24, 2);
  assert.equal(bc1M2.width, 10);
  assert.equal(bc1M2.height, 6);
  assert.equal(bc1M2.copyWidth, 12);
  assert.equal(bc1M2.copyHeight, 8);
  assert.equal(bc1M2.blocksX, 3);
  assert.equal(bc1M2.blocksY, 2);
  assert.equal(bc1M2.unalignedBytesPerRow, 24);
  assert.equal(bc1M2.bytesPerRow, 256);
  assert.equal(bc1M2.totalBytes, 512);
  assert.equal(computeMipByteSize(DXGI_FORMAT.BC1_UNORM, 40, 24, 2, 6), 512 * 6);

  // HDR10 / FP16 render targets via shorthand alias and DXGI id
  const hdrM1 = computeMipLayout('RGBA16_FLOAT', 1920, 1080, 1);
  assert.equal(hdrM1.width, 960);
  assert.equal(hdrM1.height, 540);
  assert.equal(hdrM1.unalignedBytesPerRow, 960 * 8);
  assert.equal(hdrM1.bytesPerRow, 7680); // 7680 % 256 === 0
  assert.equal(hdrM1.totalBytes, 7680 * 540);

  // Large mipLevel (32, Infinity) clamps cleanly to 1x1 without 32-bit shift wraparound
  const hugeMip = computeMipLayout(DXGI_FORMAT.R10G10B10A2_UNORM, 1024, 1024, 32);
  assert.equal(hugeMip.width, 1);
  assert.equal(hugeMip.height, 1);
  assert.equal(hugeMip.bytesPerRow, 256);
  assert.equal(hugeMip.totalBytes, 256);

  // Zero-alloc reusable `out` object returns same reference and overwrites cleanly
  const scratch = {};
  const r1 = computeMipLayout(DXGI_FORMAT.BC6H_UF16, 64, 64, 0, scratch);
  assert.equal(r1, scratch);
  assert.equal(scratch.width, 64);
  assert.equal(scratch.bytesPerRow, 256);
  assert.equal(scratch.totalBytes, 256 * 16);

  const r2 = computeMipLayout(DXGI_FORMAT.BC6H_UF16, 64, 64, 4, scratch);
  assert.equal(r2, scratch);
  assert.equal(scratch.width, 4);
  assert.equal(scratch.blocksX, 1);
  assert.equal(scratch.totalBytes, 256);

  // Degenerate 0 / negative / NaN / Infinity / invalid format NEVER throw and return zeroed layout
  const degenerateCases = [
    [DXGI_FORMAT.BC7_UNORM, 0, 64, 0],
    [DXGI_FORMAT.BC7_UNORM, 64, 0, 0],
    [DXGI_FORMAT.BC7_UNORM, NaN, 64, 0],
    [DXGI_FORMAT.BC7_UNORM, 64, NaN, 0],
    [DXGI_FORMAT.BC7_UNORM, -16, 64, 0],
    [DXGI_FORMAT.BC7_UNORM, Infinity, 64, 0],
    [DXGI_FORMAT.R8G8B8A8_UNORM, undefined, null, NaN],
    [DXGI_FORMAT.UNKNOWN, 64, 64, 0],
    ['not-a-format', 64, 64, 0],
    [null, 64, 64, 0],
  ];

  const firstZero = computeMipLayout(DXGI_FORMAT.BC7_UNORM, 0, 0, 0);
  assert.equal(Object.isFrozen(firstZero), true);
  for (const [fmt, w, h, mip] of degenerateCases) {
    const res = computeMipLayout(fmt, w, h, mip);
    assert.equal(res, firstZero, `degenerate (${String(fmt)}, ${w}, ${h}) reuses frozen zero singleton`);
    assert.equal(res.width, 0);
    assert.equal(res.height, 0);
    assert.equal(res.bytesPerRow, 0);
    assert.equal(res.totalBytes, 0);
    assert.equal(res.byteSize, 0);
    assert.equal(computeMipByteSize(fmt, w, h, mip), 0);

    // Passing `out` zeroes `out` in-place (no stale data retained)
    const outRes = computeMipLayout(fmt, w, h, mip, scratch);
    assert.equal(outRes, scratch);
    assert.equal(scratch.width, 0);
    assert.equal(scratch.height, 0);
    assert.equal(scratch.bytesPerRow, 0);
    assert.equal(scratch.totalBytes, 0);
  }

  // Degenerate mipLevel (negative / NaN) safely clamps to mip 0 without throwing
  const negMip = computeMipLayout(DXGI_FORMAT.R8G8B8A8_UNORM, 64, 32, -3);
  assert.equal(negMip.mipLevel, 0);
  assert.equal(negMip.width, 64);
  assert.equal(negMip.height, 32);
  assert.equal(negMip.bytesPerRow, 256);
  const nanMip = computeMipLayout(DXGI_FORMAT.R8G8B8A8_UNORM, 64, 32, NaN);
  assert.equal(nanMip.mipLevel, 0);
  assert.equal(nanMip.width, 64);
});

// ---------------------------------------------------------------------------
// W380 (#1763): Full 3D / Cube / 2D-Array / BC Format Capabilities & Layout
// ---------------------------------------------------------------------------

test('W380 exposes per-format capability flags on WEBGPU_FORMAT_INFO and feature-aware describeTextureFormatCapabilities()', () => {
  assert.equal(RG11B10UFLOAT_RENDERABLE_FEATURE, 'rg11b10ufloat-renderable');
  assert.equal(FLOAT32_FILTERABLE_FEATURE, 'float32-filterable');

  // rgba16float: 4ch float, renderable, filterable, multisampleable, storageCompatible, supports3d/cube/2d-array
  const rgba16f = getWebGpuFormatInfo('rgba16float');
  assert.equal(rgba16f.channels, 4);
  assert.equal(rgba16f.componentType, 'float');
  assert.equal(rgba16f.renderable, true);
  assert.equal(rgba16f.isRenderTarget, true);
  assert.equal(rgba16f.filterable, true);
  assert.equal(rgba16f.multisampleable, true);
  assert.equal(rgba16f.storageCompatible, true);
  assert.equal(rgba16f.supports3d, true);
  assert.equal(rgba16f.supportsCube, true);
  assert.equal(rgba16f.supports2dArray, true);
  assert.equal(rgba16f.supportsBcSliced3d, false);

  // depth32float: 1ch depth, renderable, NOT filterable, multisampleable, NOT storage, NOT supports3d
  const d32f = getWebGpuFormatInfo('depth32float');
  assert.equal(d32f.channels, 1);
  assert.equal(d32f.componentType, 'depth');
  assert.equal(d32f.renderable, true);
  assert.equal(d32f.filterable, false);
  assert.equal(d32f.multisampleable, true);
  assert.equal(d32f.storageCompatible, false);
  assert.equal(d32f.supports3d, false);
  assert.equal(d32f.supportsCube, true);
  assert.equal(d32f.supports2dArray, true);

  // depth24plus-stencil8: 2ch depth-stencil
  const d24s8 = getWebGpuFormatInfo('depth24plus-stencil8');
  assert.equal(d24s8.channels, 2);
  assert.equal(d24s8.componentType, 'depth-stencil');
  assert.equal(d24s8.supports3d, false);

  // bc7-rgba-unorm-srgb: 4ch unorm, NOT renderable, filterable, NOT multisampleable, supportsBcSliced3d
  const bc7 = getWebGpuFormatInfo('bc7-rgba-unorm-srgb');
  assert.equal(bc7.channels, 4);
  assert.equal(bc7.componentType, 'unorm');
  assert.equal(bc7.renderable, false);
  assert.equal(bc7.filterable, true);
  assert.equal(bc7.multisampleable, false);
  assert.equal(bc7.storageCompatible, false);
  assert.equal(bc7.supports3d, false);
  assert.equal(bc7.supportsBcSliced3d, true);

  // Feature-aware describeTextureFormatCapabilities()
  const rg11NoFeat = describeTextureFormatCapabilities('rg11b10ufloat', { deviceFeatures: new Set() });
  assert.equal(rg11NoFeat.channels, 3);
  assert.equal(rg11NoFeat.componentType, 'ufloat');
  assert.equal(rg11NoFeat.renderable, false);
  assert.equal(rg11NoFeat.multisampleable, false);
  assert.equal(rg11NoFeat.filterable, true);

  const rg11WithFeat = describeTextureFormatCapabilities('rg11b10ufloat', {
    deviceFeatures: new Set([RG11B10UFLOAT_RENDERABLE_FEATURE, TEXTURE_FORMATS_TIER1_FEATURE]),
  });
  assert.equal(rg11WithFeat.renderable, true);
  assert.equal(rg11WithFeat.isRenderTarget, true);
  assert.equal(rg11WithFeat.multisampleable, true);
  assert.equal(rg11WithFeat.storageCompatible, true);

  const r32fCore = describeTextureFormatCapabilities('r32float', { deviceFeatures: new Set() });
  assert.equal(r32fCore.filterable, false);
  assert.equal(r32fCore.multisampleable, false);
  assert.equal(r32fCore.renderable, true);
  const r32fFilt = describeTextureFormatCapabilities('r32float', {
    deviceFeatures: new Set([FLOAT32_FILTERABLE_FEATURE]),
  });
  assert.equal(r32fFilt.filterable, true);

  const bc3dWithFeat = describeTextureFormatCapabilities('bc3-rgba-unorm', {
    dimension: '3d',
    deviceFeatures: new Set([TEXTURE_COMPRESSION_BC_FEATURE, TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE]),
  });
  assert.equal(bc3dWithFeat.supports3d, true);
  assert.equal(bc3dWithFeat.supported, true);

  // Invalid format returns null cleanly.
  assert.equal(describeTextureFormatCapabilities('unknown-format'), null);
});

test('W380 validateTextureDimensionAndFormat enforces 3D, cube, cube-array, 2d-array, BC-sliced-3D, depth/stencil, and MSAA rules', () => {
  const bc2dOnly = new Set([TEXTURE_COMPRESSION_BC_FEATURE]);
  const bc3dFeatures = new Set([
    TEXTURE_COMPRESSION_BC_FEATURE,
    TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE,
  ]);

  // 1. Depth/stencil formats are disallowed on 3D and 1D textures, allowed on 2D / 2d-array / cube.
  assert.equal(
    validateTextureDimensionAndFormat({ format: 'depth32float', dimension: '3d', width: 64, height: 64, depthOrArrayLayers: 16 }).reason,
    'depth-stencil-unsupported-on-3d'
  );
  assert.equal(
    validateTextureDimensionAndFormat({ format: 'depth24plus-stencil8', dimension: '1d', width: 64 }).reason,
    'depth-stencil-unsupported-on-1d'
  );
  assert.equal(
    validateTextureDimensionAndFormat({ format: 'depth32float', dimension: '2d', viewDimension: '2d-array', width: 1024, height: 1024, depthOrArrayLayers: 4 }).valid,
    true
  );
  assert.equal(
    validateTextureDimensionAndFormat({ format: 'depth32float', dimension: 'cube', width: 512, height: 512, depthOrArrayLayers: 6 }).valid,
    true
  );

  // 2. BC formats on 3D require texture-compression-bc-sliced-3d; disallowed on 1D.
  const bc3dMissing = validateTextureDimensionAndFormat(
    { format: 'bc7-rgba-unorm', dimension: '3d', width: 64, height: 64, depthOrArrayLayers: 16 },
    { deviceFeatures: bc2dOnly }
  );
  assert.equal(bc3dMissing.valid, false);
  assert.equal(bc3dMissing.reason, 'missing-bc-sliced-3d');
  assert.equal(bc3dMissing.requiredFeature, TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE);

  const bc3dOk = validateTextureDimensionAndFormat(
    { format: 'bc7-rgba-unorm', dimension: '3d', width: 64, height: 64, depthOrArrayLayers: 16 },
    { deviceFeatures: bc3dFeatures }
  );
  assert.equal(bc3dOk.valid, true);

  assert.equal(
    validateTextureDimensionAndFormat(
      { format: 'bc1-rgba-unorm', dimension: '1d', width: 64 },
      { deviceFeatures: bc3dFeatures }
    ).reason,
    'compressed-unsupported-on-1d'
  );

  // 3. Cube and cube-array rules: square dimensions & 6 / multiple-of-6 layers.
  assert.equal(
    validateTextureDimensionAndFormat({ format: 'rgba16float', dimension: 'cube', width: 256, height: 128, depthOrArrayLayers: 6 }).reason,
    'cube-requires-square-dimensions'
  );
  assert.equal(
    validateTextureDimensionAndFormat({ format: 'rgba16float', dimension: 'cube', width: 256, height: 256, depthOrArrayLayers: 4 }).reason,
    'cube-requires-6-layers'
  );
  assert.equal(
    validateTextureDimensionAndFormat({ format: 'rgba16float', dimension: 'cube-array', width: 256, height: 256, depthOrArrayLayers: 10 }).reason,
    'cube-array-requires-multiple-of-6-layers'
  );
  assert.equal(
    validateTextureDimensionAndFormat({ format: 'rgba16float', dimension: 'cube-array', width: 256, height: 256, depthOrArrayLayers: 18 }).valid,
    true
  );

  // 4. MSAA 4x validation.
  assert.equal(
    validateTextureDimensionAndFormat({ format: 'rgba16float', dimension: '2d', width: 1280, height: 720, sampleCount: 4 }).valid,
    true
  );
  assert.equal(
    validateTextureDimensionAndFormat({ format: 'rgba32float', dimension: '2d', width: 1280, height: 720, sampleCount: 4 }).reason,
    'multisample-unsupported'
  );
  assert.equal(
    validateTextureDimensionAndFormat({ format: 'rgba16float', dimension: '3d', width: 32, height: 32, depthOrArrayLayers: 16, sampleCount: 4 }).reason,
    'multisample-unsupported'
  );
});

test('W380 calcSubresourceLayout and calcTextureMemoryBytes compute exact 3D, cube, 2D-array, and BC mip-chain sizes with zero-alloc out reuse', () => {
  // 1. 3D volume texture halves depth per mip (32x16x8 rgba16float -> mip 2 is 8x4x2).
  const scratch = {};
  const volMip2 = calcSubresourceLayout(
    'rgba16float',
    { width: 32, height: 16, depthOrArrayLayers: 8, mipLevel: 2, dimension: '3d' },
    scratch
  );
  assert.strictEqual(volMip2, scratch);
  assert.equal(volMip2.width, 8);
  assert.equal(volMip2.height, 4);
  assert.equal(volMip2.depthOrArrayLayers, 2);
  assert.equal(volMip2.unalignedBytesPerRow, 64); // 8 * 8B
  assert.equal(volMip2.bytesPerRow, 256);
  assert.equal(volMip2.sliceTightBytes, 256); // 64 * 4
  assert.equal(volMip2.sliceAlignedBytes, 1024); // 256 * 4
  assert.equal(volMip2.tightBytes, 512); // 256 * 2 slices
  assert.equal(volMip2.totalBytes, 2048); // 1024 * 2 slices

  // 2. Cube texture keeps 6 faces constant across mips (BC7 64x64x6 -> mip 5 is 2x2x6 -> 1x1 block per face).
  const cubeMip5 = calcSubresourceLayout(
    'bc7-rgba-unorm',
    { width: 64, height: 64, depthOrArrayLayers: 6, mipLevel: 5, dimension: 'cube' },
    scratch
  );
  assert.strictEqual(cubeMip5, scratch);
  assert.equal(cubeMip5.width, 2);
  assert.equal(cubeMip5.height, 2);
  assert.equal(cubeMip5.depthOrArrayLayers, 6);
  assert.equal(cubeMip5.copyWidth, 4);
  assert.equal(cubeMip5.copyHeight, 4);
  assert.equal(cubeMip5.blocksX, 1);
  assert.equal(cubeMip5.blocksY, 1);
  assert.equal(cubeMip5.tightBytes, 16 * 6);
  assert.equal(cubeMip5.totalBytes, 256 * 6);

  // 3. calcTextureMemoryBytes across full mip chains:
  // 3D 4x4x4 rgba8unorm with 3 mips (4x4x4 + 2x2x2 + 1x1x1 = 73 texels * 4 = 292 B tight).
  assert.equal(
    calcTextureMemoryBytes('rgba8unorm', {
      width: 4,
      height: 4,
      depthOrArrayLayers: 4,
      mipLevelCount: 3,
      dimension: '3d',
    }),
    292
  );

  // Cube 8x8 BC1 with 4 mips:
  // mip0: 8x8 -> 2x2 blocks * 8B * 6 = 192
  // mip1: 4x4 -> 1x1 block * 8B * 6 = 48
  // mip2: 2x2 -> 1x1 block * 8B * 6 = 48
  // mip3: 1x1 -> 1x1 block * 8B * 6 = 48
  // Total tight = 336 B; aligned (256B/row): (256*2 + 256*1 + 256*1 + 256*1) * 6 = 7680 B.
  assert.equal(
    calcTextureMemoryBytes('bc1-rgba-unorm', {
      width: 8,
      height: 8,
      mipLevelCount: 4,
      dimension: 'cube',
    }),
    336
  );
  assert.equal(
    calcTextureMemoryBytes('bc1-rgba-unorm', {
      width: 8,
      height: 8,
      mipLevelCount: 4,
      dimension: 'cube',
      aligned: true,
    }),
    7680
  );

  // 2D-array 4-cascade shadow map 1024x1024x4 depth32float with 2 mips: (1024^2 + 512^2) * 4B * 4 layers
  assert.equal(
    calcTextureMemoryBytes('depth32float', {
      width: 1024,
      height: 1024,
      depthOrArrayLayers: 4,
      mipLevelCount: 2,
      dimension: '2d-array',
    }),
    (1024 * 1024 + 512 * 512) * 4 * 4
  );

  // Degenerate dimensions & unknown format return 0 without throwing.
  assert.equal(calcTextureMemoryBytes('rgba8unorm', { width: 0, height: 64 }), 0);
  assert.equal(calcTextureMemoryBytes('invalid-fmt', { width: 64, height: 64 }), 0);
  assert.equal(calcSubresourceLayout('invalid-fmt', { width: 64, height: 64 }, scratch).totalBytes, 0);
});
