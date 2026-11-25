import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const texMod = require('../web/workers/texture_formats.js');

const {
  DXGI_FORMAT,
  WEBGPU_BYTES_PER_ROW_ALIGNMENT,
  TEXTURE_COMPRESSION_BC_FEATURE,
  TEXTURE_COMPRESSION_BC_SLICED_3D_FEATURE,
  TEXTURE_FORMATS_TIER1_FEATURE,
  DEPTH32FLOAT_STENCIL8_FEATURE,
  mapDxgiFormatToWebGpu,
  getSrgbSiblingFormat,
  classifyTextureFormatSupport,
  calcMipSize,
  calcMipLayout,
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
