// Runtime-owned RT BVH WGSL prelude helper (version 1).
// Pure ES module with zero external dependencies — runs in Web Workers and Node.js.
// Keep RT_BVH_PRELUDE_VERSION and RT_BVH_PRELUDE_WGSL byte-identical to
// runtime/gpu/rt_bvh.h::kPreludeVersion and runtime/gpu/rt_bvh.cpp::prelude_wgsl().

/** Prelude interface version matching runtime/gpu/rt_bvh.h::kPreludeVersion. */
export const RT_BVH_PRELUDE_VERSION = 1;

/**
 * Runtime-owned RT WGSL prelude matching runtime/gpu/rt_bvh.cpp::prelude_wgsl() verbatim.
 * Declares group 3 bindings 16-21 (pg6_arena0..3, pg6_arena_dir, pg6_rt_stats),
 * pg6_load32, and pg6_bvh_intersect.
 */
export const RT_BVH_PRELUDE_WGSL = `// pg6 runtime prelude, version 1 (rt_bvh.h kPreludeVersion). Group 3, bindings 16-21: contract C3.
@group(3) @binding(16) var<storage, read_write> pg6_arena0: array<u32>;
@group(3) @binding(17) var<storage, read_write> pg6_arena1: array<u32>;
@group(3) @binding(18) var<storage, read_write> pg6_arena2: array<u32>;
@group(3) @binding(19) var<storage, read_write> pg6_arena3: array<u32>;
@group(3) @binding(20) var<storage, read> pg6_arena_dir: array<u32>;
@group(3) @binding(21) var<storage, read_write> pg6_rt_stats: array<atomic<u32>>;

// A non-resident access: bumps rt_fault (word 0) and logs the page for the runtime to make resident
// (word 1 = log attempts, words 2..31 = faulting page + 1, first 30 only). rt_residency.h FaultTracker.
fn pg6_fault(page: u32) {
    atomicAdd(&pg6_rt_stats[0], 1u);
    let i = atomicAdd(&pg6_rt_stats[1], 1u);
    if (i < 30u) {
        atomicStore(&pg6_rt_stats[2u + i], page + 1u);
    }
}

// Resolves a 40-bit address through the two-level page directory once. Returns 0u on fault.
fn pg6_resolve(addr: vec2<u32>) -> u32 {
    let page = (addr.x >> 16u) | ((addr.y & 0xffu) << 16u);
    let t = pg6_arena_dir[page >> 12u];
    if (t == 0u) {
        pg6_fault(page);
        return 0u;
    }
    let e = pg6_arena_dir[4096u + (t - 1u) * 4096u + (page & 0xfffu)];
    if ((e & 0x80000000u) == 0u) {
        pg6_fault(page);
        return 0u;
    }
    return e;
}

fn pg6_read_resolved(e: u32, page_byte_off: u32) -> u32 {
    let w = (e & 0x1fffffffu) * 16384u + ((page_byte_off & 0xffffu) >> 2u);
    switch ((e >> 29u) & 3u) {
        case 0u: { return pg6_arena0[w]; }
        case 1u: { return pg6_arena1[w]; }
        case 2u: { return pg6_arena2[w]; }
        default: { return pg6_arena3[w]; }
    }
}

// Dword at a 40-bit address (x = low 32 bits, y = bits 39:32). Unmapped: counts a fault, reads 0.
fn pg6_load32(addr: vec2<u32>) -> u32 {
    let e = pg6_resolve(addr);
    if (e == 0u) {
        return 0u;
    }
    return pg6_read_resolved(e, addr.x);
}

fn pg6_add40(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    var carry = 0u;
    if (lo < a.x) { carry = 1u; }
    return vec2<u32>(lo, (a.y + b.y + carry) & 0xffu);
}

fn pg6_addr_off(a: vec2<u32>, off: u32) -> vec2<u32> {
    return pg6_add40(a, vec2<u32>(off, 0u));
}

fn pg6_f(a: vec2<u32>, off: u32) -> f32 {
    return bitcast<f32>(pg6_load32(pg6_addr_off(a, off)));
}

fn pg6_f_resolved(e: u32, page_off: u32, off: u32) -> f32 {
    return bitcast<f32>(pg6_read_resolved(e, page_off + off));
}

// Bit-exact IEEE-754 NaN test (immune to fast-math folding \`x != x\` to false on Metal/HLSL).
fn pg6_isnan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

// Slab test of one box: .x = 1.0 when entered, .y = the entry distance.
fn pg6_slab(mn: vec3<f32>, mx: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, extent: f32) -> vec2<f32> {
    var lo = 0.0;
    var hi = extent;
    for (var a = 0u; a < 3u; a = a + 1u) {
        if (pg6_isnan(mn[a]) || pg6_isnan(mx[a]) || mn[a] > mx[a]) {
            return vec2<f32>(0.0, 0.0);
        }
        let t0 = (mn[a] - o[a]) * inv[a];
        let t1 = (mx[a] - o[a]) * inv[a];
        if (pg6_isnan(t0) || pg6_isnan(t1)) {
            continue;
        }
        var a0 = t1;
        var a1 = t0;
        if (t0 < t1) {
            a0 = t0;
            a1 = t1;
        }
        if (a0 > lo) { lo = a0; }
        if (a1 < hi) { hi = a1; }
    }
    if (lo <= hi) {
        return vec2<f32>(1.0, lo);
    }
    return vec2<f32>(0.0, 0.0);
}

fn pg6_tri_miss() -> vec4<u32> {
    return vec4<u32>(0x7f800000u, 0x3f800000u, 0u, 0u);
}

fn pg6_vec_nan(v: vec3<f32>) -> bool {
    return pg6_isnan(v.x) || pg6_isnan(v.y) || pg6_isnan(v.z);
}

fn pg6_triangle(e: u32, page_off: u32, kind: u32, o: vec3<f32>, d: vec3<f32>) -> vec4<u32> {
    var vi = vec3<u32>(0u, 1u, 2u);
    if (kind == 1u) { vi = vec3<u32>(1u, 3u, 2u); }
    if (kind == 2u) { vi = vec3<u32>(2u, 3u, 4u); }
    if (kind == 3u) { vi = vec3<u32>(2u, 4u, 0u); }
    var v: array<vec3<f32>, 3>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let b = vi[i] * 12u;
        v[i] = vec3<f32>(pg6_f_resolved(e, page_off, b), pg6_f_resolved(e, page_off, b + 4u),
                         pg6_f_resolved(e, page_off, b + 8u)) - o;
    }
    var kz = 0u;
    var az = abs(d.x);
    if (abs(d.y) > az) { kz = 1u; az = abs(d.y); }
    if (abs(d.z) > az) { kz = 2u; }
    let dz = d[kz];
    if (dz == 0.0) {
        return pg6_tri_miss();
    }
    var kx = (kz + 1u) % 3u;
    var ky = (kx + 1u) % 3u;
    if (dz < 0.0) {
        let t = kx;
        kx = ky;
        ky = t;
    }
    let sx = d[kx] / dz;
    let sy = d[ky] / dz;
    let sz = 1.0 / dz;
    var px: array<f32, 3>;
    var py: array<f32, 3>;
    var pz: array<f32, 3>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        px[i] = v[i][kx] - sx * v[i][kz];
        py[i] = v[i][ky] - sy * v[i][kz];
        pz[i] = sz * v[i][kz];
    }
    let u = px[2] * py[1] - py[2] * px[1];
    let vv = px[0] * py[2] - py[0] * px[2];
    let w = px[1] * py[0] - py[1] * px[0];
    if ((u < 0.0 || vv < 0.0 || w < 0.0) && (u > 0.0 || vv > 0.0 || w > 0.0)) {
        return pg6_tri_miss();
    }
    let det = u + vv + w;
    let t_num = u * pz[0] + vv * pz[1] + w * pz[2];
    if (det == 0.0 || pg6_isnan(det) || pg6_isnan(t_num) || pg6_isnan(u) || pg6_isnan(vv) || pg6_isnan(w)) {
        return pg6_tri_miss();
    }
    let sel = (pg6_read_resolved(e, page_off + 60u) >> (8u * kind)) & 0xffu;
    var wt = array<f32, 4>(u, vv, w, 0.0);
    return vec4<u32>(bitcast<u32>(t_num), bitcast<u32>(det), bitcast<u32>(wt[sel & 3u]),
                     bitcast<u32>(wt[(sel >> 2u) & 3u]));
}

// C4: the console's image_bvh(64)_intersect_ray node test. desc = the 128-bit BVH descriptor,
// node = the node pointer (x low, y high; high 0 for the 32-bit form).
fn pg6_bvh_intersect(desc: vec4<u32>, node: vec2<u32>, extent: f32, origin: vec3<f32>, dir: vec3<f32>,
                     inv_dir: vec3<f32>) -> vec4<u32> {
    let kind = node.x & 7u;
    if (kind >= 6u) {
        return pg6_tri_miss();
    }
    let nan_ray = pg6_isnan(extent) || pg6_vec_nan(origin) || pg6_vec_nan(dir) || pg6_vec_nan(inv_dir);
    // address = (base256 << 8) + ((ptr >> 3) << 6), 40 bits (guaranteed 64-byte aligned)
    let base = vec2<u32>(desc.x << 8u, ((desc.y & 0xffu) << 8u) | (desc.x >> 24u));
    let idx = vec2<u32>((node.x >> 3u) | (node.y << 29u), node.y >> 3u);
    let off = vec2<u32>(idx.x << 6u, (idx.y << 6u) | (idx.x >> 26u));
    let addr = pg6_add40(base, off);
    let page_off = addr.x & 0xffffu;
    if (kind < 4u) {
        if (nan_ray) {
            return pg6_tri_miss();
        }
        let e0 = pg6_resolve(addr);
        if (e0 == 0u) {
            return pg6_tri_miss();
        }
        return pg6_triangle(e0, page_off, kind, origin, dir);
    }
    if (nan_ray) {
        return vec4<u32>(0xffffffffu);
    }
    let e0 = pg6_resolve(addr);
    if (e0 == 0u) {
        return vec4<u32>(0xffffffffu);
    }
    let single_page = kind == 4u || page_off <= 0xff80u;
    var ptr: array<u32, 4>;
    var key: array<f32, 4>;
    var hit: array<bool, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        ptr[i] = pg6_read_resolved(e0, page_off + 4u * i);
        key[i] = 0.0;
        hit[i] = false;
        if (ptr[i] == 0xffffffffu) {
            continue;
        }
        var mn: vec3<f32>;
        var mx: vec3<f32>;
        if (kind == 5u) {
            let b = 16u + 24u * i;
            if (single_page) {
                mn = vec3<f32>(pg6_f_resolved(e0, page_off, b), pg6_f_resolved(e0, page_off, b + 4u),
                               pg6_f_resolved(e0, page_off, b + 8u));
                mx = vec3<f32>(pg6_f_resolved(e0, page_off, b + 12u), pg6_f_resolved(e0, page_off, b + 16u),
                               pg6_f_resolved(e0, page_off, b + 20u));
            } else {
                mn = vec3<f32>(pg6_f(addr, b), pg6_f(addr, b + 4u), pg6_f(addr, b + 8u));
                mx = vec3<f32>(pg6_f(addr, b + 12u), pg6_f(addr, b + 16u), pg6_f(addr, b + 20u));
            }
        } else {
            let b = 16u + 12u * i;
            let a = unpack2x16float(pg6_read_resolved(e0, page_off + b));
            let bb = unpack2x16float(pg6_read_resolved(e0, page_off + b + 4u));
            let c = unpack2x16float(pg6_read_resolved(e0, page_off + b + 8u));
            mn = vec3<f32>(a.x, a.y, bb.x);
            mx = vec3<f32>(bb.y, c.x, c.y);
        }
        let s = pg6_slab(mn, mx, origin, inv_dir, extent);
        hit[i] = s.x > 0.5;
        key[i] = s.y;
    }
    var out = vec4<u32>(0xffffffffu);
    let sort = (desc.y >> 31u) != 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (!hit[i]) {
            continue;
        }
        var rank = i;
        if (sort) {
            rank = 0u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                if (hit[j] && (key[j] < key[i] || (key[j] == key[i] && j < i))) {
                    rank = rank + 1u;
                }
            }
        }
        out[rank] = ptr[i];
    }
    return out;
}
`;

const RT_HELPER_CALL_RE = /\b(?:pg6_bvh_intersect|pg6_load32)\s*\(/;
const RT_HELPER_DEF_RE = /\bfn\s+(?:pg6_load32|pg6_bvh_intersect)\s*\(/;

function isWhitespaceCode(ch) {
  return (
    ch === 0x20 || // space
    ch === 0x09 || // \t
    ch === 0x0a || // \n
    ch === 0x0d || // \r
    ch === 0x0b || // \v
    ch === 0x0c || // \f
    ch === 0xfeff  // UTF-8 BOM
  );
}

function skipCommentAt(wgsl, pos) {
  const len = wgsl.length;
  if (pos + 1 >= len || wgsl.charCodeAt(pos) !== 0x2f) {
    return pos;
  }
  const next = wgsl.charCodeAt(pos + 1);
  if (next === 0x2f) {
    // Line comment: consume up to (not including) newline.
    let i = pos + 2;
    while (i < len && wgsl.charCodeAt(i) !== 0x0a) {
      i += 1;
    }
    return i;
  }
  if (next === 0x2a) {
    // Nested WGSL block comment /* ... */.
    let i = pos + 2;
    let depth = 1;
    while (i < len && depth > 0) {
      const c = wgsl.charCodeAt(i);
      if (c === 0x2f && i + 1 < len && wgsl.charCodeAt(i + 1) === 0x2a) {
        depth += 1;
        i += 2;
      } else if (c === 0x2a && i + 1 < len && wgsl.charCodeAt(i + 1) === 0x2f) {
        depth -= 1;
        i += 2;
      } else {
        i += 1;
      }
    }
    return i;
  }
  return pos;
}

function skipLeadingTrivia(wgsl, start) {
  const len = wgsl.length;
  let i = start;
  while (i < len) {
    const ch = wgsl.charCodeAt(i);
    if (isWhitespaceCode(ch)) {
      i += 1;
      continue;
    }
    const afterComment = skipCommentAt(wgsl, i);
    if (afterComment !== i) {
      i = afterComment;
      continue;
    }
    break;
  }
  return i;
}

function matchDirectiveKeyword(wgsl, pos) {
  const len = wgsl.length;
  if (wgsl.startsWith('enable', pos)) {
    const after = pos + 6;
    if (after < len) {
      const ch = wgsl.charCodeAt(after);
      if (isWhitespaceCode(ch) || ch === 0x2f) {
        return after;
      }
    }
  } else if (wgsl.startsWith('requires', pos)) {
    const after = pos + 8;
    if (after < len) {
      const ch = wgsl.charCodeAt(after);
      if (isWhitespaceCode(ch) || ch === 0x2f) {
        return after;
      }
    }
  } else if (wgsl.startsWith('diagnostic', pos)) {
    const after = pos + 10;
    if (after < len) {
      const ch = wgsl.charCodeAt(after);
      if (isWhitespaceCode(ch) || ch === 0x2f || ch === 0x28 /* '(' */) {
        return after;
      }
    }
  }
  return -1;
}

/**
 * Find the byte/character offset immediately after any leading WGSL global directives
 * (`enable ...;`, `requires ...;`, `diagnostic(...);`). Returns `0` when the shader
 * has no top-of-file directives.
 *
 * @param {string} wgsl
 * @returns {number}
 */
function findGlobalDirectivesEnd(wgsl) {
  const len = wgsl.length;
  let pos = 0;
  let lastDirectiveEnd = 0;

  while (pos < len) {
    const kwStart = skipLeadingTrivia(wgsl, pos);
    if (kwStart >= len) {
      break;
    }
    const afterKw = matchDirectiveKeyword(wgsl, kwStart);
    if (afterKw < 0) {
      break;
    }

    // Scan to the directive's terminating ';', skipping comments and tracking parentheses.
    let i = afterKw;
    let parenDepth = 0;
    let foundSemi = false;
    while (i < len) {
      const afterComment = skipCommentAt(wgsl, i);
      if (afterComment !== i) {
        i = afterComment;
        continue;
      }
      const ch = wgsl.charCodeAt(i);
      if (ch === 0x28 /* '(' */) {
        parenDepth += 1;
        i += 1;
      } else if (ch === 0x29 /* ')' */) {
        if (parenDepth > 0) {
          parenDepth -= 1;
        }
        i += 1;
      } else if (ch === 0x3b /* ';' */ && parenDepth === 0) {
        i += 1;
        foundSemi = true;
        break;
      } else {
        i += 1;
      }
    }

    if (!foundSemi) {
      break;
    }

    // Keep any same-line trailing spaces, inline '//' comment, and line break with the directive.
    let tail = i;
    while (tail < len && (wgsl.charCodeAt(tail) === 0x20 || wgsl.charCodeAt(tail) === 0x09)) {
      tail += 1;
    }
    if (
      tail + 1 < len &&
      wgsl.charCodeAt(tail) === 0x2f &&
      wgsl.charCodeAt(tail + 1) === 0x2f
    ) {
      tail = skipCommentAt(wgsl, tail);
    }
    if (tail < len && wgsl.charCodeAt(tail) === 0x0d) {
      tail += 1;
      if (tail < len && wgsl.charCodeAt(tail) === 0x0a) {
        tail += 1;
      }
      i = tail;
    } else if (tail < len && wgsl.charCodeAt(tail) === 0x0a) {
      i = tail + 1;
    }

    lastDirectiveEnd = i;
    pos = i;
  }

  return lastDirectiveEnd;
}

function stripWgslComments(wgsl) {
  const len = wgsl.length;
  let out = '';
  let chunkStart = 0;
  let i = 0;
  while (i < len) {
    if (wgsl.charCodeAt(i) === 0x2f && i + 1 < len) {
      const next = wgsl.charCodeAt(i + 1);
      if (next === 0x2f || next === 0x2a) {
        if (i > chunkStart) {
          out += wgsl.slice(chunkStart, i);
        }
        out += ' ';
        i = skipCommentAt(wgsl, i);
        chunkStart = i;
        continue;
      }
    }
    i += 1;
  }
  if (chunkStart === 0) {
    return wgsl;
  }
  if (chunkStart < len) {
    out += wgsl.slice(chunkStart);
  }
  return out;
}

/**
 * Return `true` iff `wgsl` calls `pg6_bvh_intersect(` or `pg6_load32(` outside comments and does
 * not already define `fn pg6_load32(` or `fn pg6_bvh_intersect(`.
 *
 * @param {string} wgsl
 * @returns {boolean}
 */
export function needsRtBvhPrelude(wgsl) {
  if (typeof wgsl !== 'string' || wgsl.length === 0) {
    return false;
  }
  // Fast pre-check before comment stripping.
  if (!RT_HELPER_CALL_RE.test(wgsl)) {
    return false;
  }
  const code = stripWgslComments(wgsl);
  return RT_HELPER_CALL_RE.test(code) && !RT_HELPER_DEF_RE.test(code);
}

/**
 * Prepend `RT_BVH_PRELUDE_WGSL` when `wgsl` requires the runtime RT helpers, placing
 * the prelude after any leading WGSL `enable` / `requires` / `diagnostic` directives.
 * Strips a leading UTF-8 BOM so it is never embedded mid-shader.
 * Returns `wgsl` unchanged if the prelude is not needed or is already present.
 *
 * @param {string} wgsl
 * @returns {string}
 */
export function applyRtBvhPrelude(wgsl) {
  if (!needsRtBvhPrelude(wgsl)) {
    return wgsl;
  }
  const source = wgsl.charCodeAt(0) === 0xfeff ? wgsl.slice(1) : wgsl;
  const insertAt = findGlobalDirectivesEnd(source);
  if (insertAt === 0) {
    return `${RT_BVH_PRELUDE_WGSL}\n${source}`;
  }
  const prefix = source.slice(0, insertAt);
  const rest = source.slice(insertAt);
  const sepBefore = prefix.endsWith('\n') ? '' : '\n';
  const sepAfter = rest.startsWith('\n') || rest.startsWith('\r\n') ? '' : '\n';
  return `${prefix}${sepBefore}${RT_BVH_PRELUDE_WGSL}${sepAfter}${rest}`;
}
