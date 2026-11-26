// Native executor: swapchain blit model (web/workers/wgpu_swapchain.js `blitToCanvasIfNeeded`,
// `configureForStreamFormat`). Pure logic, no GPU: which stream formats need the intermediate
// texture + fullscreen-triangle blit, the blit shader (the worker's, verbatim), and the letterbox
// viewport used when the stream size differs from the presenting canvas / window size.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "gpu/native_exec/window_layout.h"  // PresentRect / letterbox_rect (the window's present rect)

namespace playgta6::gpu::native_exec {

// The worker's BLIT_FULLSCREEN_WGSL (raw passthrough blit). Use ONLY for pre-encoded SDR sources
// (e.g. rgba8unorm / bgra8unorm -> non-sRGB SDR canvas), linear-to-linear HDR blits (rgba16float
// canvas), or when writing into a hardware `-srgb` canvas target view.
//
// WARNING: Do NOT use `kBlitWgsl` when sampling an `*-srgb` intermediate (`rgba8unorm-srgb`,
// `bgra8unorm-srgb`) or linear HDR intermediate (`rgba16float`, `rgb10a2unorm`, `rg11b10ufloat`,
// etc.) onto a non-sRGB SDR canvas (`rgba8unorm`, `bgra8unorm`). Sampling `*-srgb` hardware-decodes
// sRGB -> linear (`~x^2.2`), and writing linear values raw to a non-sRGB unorm canvas skips sRGB
// re-encoding, crushing midtones (`~x^4.84`) and clipping dark UI gradients to pure black. Use
// `kBlitSrgbEncodeWgsl` / `kBlitToneMapWgsl` (or target the canvas `-srgb` sibling view) instead.
inline constexpr const char* kBlitWgsl = R"(struct BlitVsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> BlitVsOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
  );
  var out: BlitVsOut;
  out.pos = vec4<f32>(pos[vid], 0.0, 1.0);
  out.uv = uv[vid];
  return out;
}

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

@fragment
fn fs_main(in: BlitVsOut) -> @location(0) vec4<f32> {
  return textureSample(blit_tex, blit_sampler, in.uv);
}
)";

// The worker's BLIT_SRGB_ENCODE_WGSL: 2-binding zero-uniform fullscreen blit applying exact
// IEC 61966-2-1 linear -> sRGB encoding (alpha preserved linearly) when presenting an `*-srgb`
// or linear HDR intermediate texture onto a non-sRGB SDR canvas (`rgba8unorm` / `bgra8unorm`).
inline constexpr const char* kBlitSrgbEncodeWgsl = R"(struct BlitVsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> BlitVsOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
  );
  var out: BlitVsOut;
  out.pos = vec4<f32>(pos[vid], 0.0, 1.0);
  out.uv = uv[vid];
  return out;
}

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

fn linear_to_srgb_channel(v: f32) -> f32 {
  let c = clamp(v, 0.0, 1.0);
  if (c <= 0.0031308) {
    return c * 12.92;
  }
  return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(rgb: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    linear_to_srgb_channel(rgb.r),
    linear_to_srgb_channel(rgb.g),
    linear_to_srgb_channel(rgb.b)
  );
}

@fragment
fn fs_main(in: BlitVsOut) -> @location(0) vec4<f32> {
  let sampled = textureSample(blit_tex, blit_sampler, in.uv);
  return vec4<f32>(linear_to_srgb(sampled.rgb), sampled.a);
}
)";

// The worker's BLIT_TONEMAP_WGSL: exposure + optional Reinhard/ACES tonemapping + exact
// IEC 61966-2-1 linear -> sRGB encoding controlled by `ToneMapParams`.
inline constexpr const char* kBlitToneMapWgsl = R"(struct BlitVsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

struct ToneMapParams {
  mode: u32,
  exposure: f32,
  apply_srgb: u32,
  _pad: u32,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> BlitVsOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
  );
  var out: BlitVsOut;
  out.pos = vec4<f32>(pos[vid], 0.0, 1.0);
  out.uv = uv[vid];
  return out;
}

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;
@group(0) @binding(2) var<uniform> tonemap_params: ToneMapParams;

fn tonemap_reinhard(c: vec3<f32>) -> vec3<f32> {
  let x = max(c, vec3<f32>(0.0));
  return x / (vec3<f32>(1.0) + x);
}

fn tonemap_aces(c: vec3<f32>) -> vec3<f32> {
  let x = max(c, vec3<f32>(0.0));
  let a = 2.51;
  let b = 0.03;
  let d = 2.43;
  let e = 0.59;
  let f = 0.14;
  return clamp((x * (a * x + vec3<f32>(b))) / (x * (d * x + vec3<f32>(e)) + vec3<f32>(f)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb_channel(v: f32) -> f32 {
  let c = clamp(v, 0.0, 1.0);
  if (c <= 0.0031308) {
    return c * 12.92;
  }
  return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(rgb: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    linear_to_srgb_channel(rgb.r),
    linear_to_srgb_channel(rgb.g),
    linear_to_srgb_channel(rgb.b)
  );
}

@fragment
fn fs_main(in: BlitVsOut) -> @location(0) vec4<f32> {
  let sampled = textureSample(blit_tex, blit_sampler, in.uv);
  var rgb = sampled.rgb * tonemap_params.exposure;
  if (tonemap_params.mode == 1u) {
    rgb = tonemap_reinhard(rgb);
  } else if (tonemap_params.mode == 2u) {
    rgb = tonemap_aces(rgb);
  }
  if (tonemap_params.apply_srgb != 0u) {
    rgb = linear_to_srgb(rgb);
  }
  return vec4<f32>(rgb, sampled.a);
}
)";

// Format codes (docs/gpu_commands.md "WebGPU enum codes"): the canvas formats (isValidCanvasFormat).
inline bool is_canvas_format_code(uint8_t code) { return code == 22 || code == 27 || code == 40; }

// Non-sRGB SDR canvas formats (`rgba8unorm` = 22, `bgra8unorm` = 27) matching
// `SDR_NON_SRGB_CANVAS_FORMAT_SET` in web/workers/wgpu_swapchain.js.
inline bool is_sdr_non_srgb_canvas_format_code(uint8_t code) { return code == 22 || code == 27; }

// sRGB swapchain stream formats (`rgba8unorm-srgb` = 23, `bgra8unorm-srgb` = 28).
inline bool is_srgb_swapchain_format_code(uint8_t code) { return code == 23 || code == 28; }

// All sRGB WebGPU format codes (swapchain + BC1/2/3/7 sRGB), matching `isSrgbFormat` / `format_is_srgb`.
inline bool is_srgb_format_code(uint8_t code) {
    return code == 23 || code == 28 || code == 51 || code == 53 || code == 55 || code == 63;
}

// Linear HDR stream formats matching `HDR_STREAM_FORMATS` in web/workers/wgpu_swapchain.js:
// r16float(9), r32float(16), rg16float(21), rgb9e5ufloat(29), rgb10a2unorm(31),
// rg11b10ufloat(32), rg32float(35), rgba16float(40), rgba32float(43).
inline bool is_hdr_stream_format_code(uint8_t code) {
    return code == 9 || code == 16 || code == 21 || code == 29 || code == 31 || code == 32 || code == 35 ||
           code == 40 || code == 43;
}

// Compatible linear <-> sRGB sibling view format code (`COMPATIBLE_VIEW_FORMATS` in wgpu_swapchain.js).
// Returns 0 when no sibling view format exists (e.g. rgba16float).
inline uint8_t compatible_view_format_code(uint8_t code) {
    switch (code) {
        case 22: return 23;  // rgba8unorm -> rgba8unorm-srgb
        case 23: return 22;  // rgba8unorm-srgb -> rgba8unorm
        case 27: return 28;  // bgra8unorm -> bgra8unorm-srgb
        case 28: return 27;  // bgra8unorm-srgb -> bgra8unorm
        default: return 0;
    }
}

// A stream swapchain format that is not a canvas format but can be rendered into: colour formats
// (codes 1..43: r8unorm .. rgba32float, sRGB and packed ones included). The worker accepts any
// non-canvas name and lets WebGPU validation reject what cannot work (uint formats cannot be
// sampled as f32, for one); the executor does the same. Depth/stencil (44..49) and block
// compressed (50+) codes are not swapchain formats.
inline bool is_blit_stream_format_code(uint8_t code) { return code >= 1 && code <= 43 && !is_canvas_format_code(code); }

// True when blitting `source_format_code` onto `canvas_format_code` must apply exact IEC 61966-2-1
// linear -> sRGB encoding in the blit shader (matching WgpuSwapchain::_resolveEffectiveApplySrgbEncode):
// - Never double-encode when the target render-pass view is already an `-srgb` hardware view.
// - Never sRGB-encode onto an HDR linear canvas (e.g. `rgba16float`).
// - Always encode once when sampling an `*-srgb` texture (hardware-decoded to linear) or a linear
//   HDR texture onto a non-sRGB SDR canvas (`rgba8unorm` / `bgra8unorm`).
// - Never re-encode pre-encoded SDR (`rgba8unorm` / `bgra8unorm`) passthrough sources.
inline bool should_apply_srgb_encode_on_blit(uint8_t source_format_code, uint8_t canvas_format_code,
                                             bool target_view_is_srgb = false) {
    if (target_view_is_srgb) {
        return false;
    }
    if (!is_sdr_non_srgb_canvas_format_code(canvas_format_code)) {
        return false;
    }
    return is_hdr_stream_format_code(source_format_code) || is_srgb_format_code(source_format_code);
}

// Preferred canvas target view format code for a blit pass: uses the hardware `-srgb` sibling view
// when supported and the source outputs linear values onto an SDR unorm canvas; otherwise keeps
// `canvas_format_code` (in which case `should_apply_srgb_encode_on_blit(..., false)` selects shader
// sRGB encoding).
inline uint8_t blit_target_view_format_code(uint8_t source_format_code, uint8_t canvas_format_code,
                                            bool canvas_supports_srgb_view) {
    if (canvas_supports_srgb_view && is_sdr_non_srgb_canvas_format_code(canvas_format_code) &&
        (is_srgb_format_code(source_format_code) || is_hdr_stream_format_code(source_format_code))) {
        const uint8_t srgb_view = compatible_view_format_code(canvas_format_code);
        if (srgb_view != 0) {
            return srgb_view;
        }
    }
    return canvas_format_code;
}

// Select the matching zero-uniform blit WGSL shader for `(source_format_code, canvas_format_code)`.
inline const char* blit_shader_wgsl(uint8_t source_format_code, uint8_t canvas_format_code,
                                    bool target_view_is_srgb = false) {
    return should_apply_srgb_encode_on_blit(source_format_code, canvas_format_code, target_view_is_srgb)
               ? kBlitSrgbEncodeWgsl
               : kBlitWgsl;
}

// True when presenting needs a blit at all: a non-canvas stream format, or a stream size that
// differs from the canvas size.
inline bool blit_needed(bool stream_format_is_canvas_format, uint32_t src_w, uint32_t src_h, uint32_t dst_w,
                        uint32_t dst_h) {
    return !stream_format_is_canvas_format || src_w != dst_w || src_h != dst_h;
}

}  // namespace playgta6::gpu::native_exec
