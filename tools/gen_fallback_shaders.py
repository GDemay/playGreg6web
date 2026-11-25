#!/usr/bin/env python3
"""Fallback WGPK shader pack + S6 reflection-manifest generator (Track 4 / S9).

Emits a deterministic `WGPK` v1 `.wgsl.pack` of original, WebGPU-compliant WGSL
fallback shaders (splash/UI blit, solid/debug-grid 3D mesh, unlit textured mesh,
and a compute clear/indirect-cull stub), plus matching `shader-index.json` and
S6 `reflection-manifest.json`.

Binary pack layout matches `shaders/src/pack.rs` and `tools/inspect_wgsl_pack.py`
byte-for-byte. Reflection schema matches `shaders/src/reflect.rs` and
`web/workers/reflection_loader.js` (`schema_version` 1).

Security invariants:
  - Refuses to write into tracked repository source trees unless the output
    resolves under gitignored `local-data/`, `work/`, or a directory outside the
    repository (e.g. temp dirs used by unit tests).
  - Never commits `.wgsl.pack` binaries; default output is `local-data/shaders/`.
  - Pure Python 3.12 standard library only.

CLI:
    python tools/gen_fallback_shaders.py \\
        [--out-dir local-data/shaders] \\
        [--pack-name fallback.wgsl.pack] \\
        [--json]
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import inspect_wgsl_pack as wgpk  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = _REPO_ROOT
DEFAULT_OUT_DIR = REPO_ROOT / "local-data" / "shaders"
DEFAULT_PACK_NAME = "fallback.wgsl.pack"
SHADER_INDEX_FILE_NAME = "shader-index.json"
REFLECTION_MANIFEST_FILE_NAME = "reflection-manifest.json"
SHADER_INDEX_ABI_VERSION = 1
LAYOUT_MANIFEST_SCHEMA_VERSION = 1

ALLOWED_REPO_OUTPUT_DIR_NAMES: tuple[str, ...] = ("local-data", "work")

VISIBILITY_VERTEX = 0x1
VISIBILITY_FRAGMENT = 0x2
VISIBILITY_COMPUTE = 0x4

STAGE_VISIBILITY: dict[str, int] = {
    "vertex": VISIBILITY_VERTEX,
    "fragment": VISIBILITY_FRAGMENT,
    "compute": VISIBILITY_COMPUTE,
}

STAGE_ORDER: dict[str, int] = {"vertex": 0, "fragment": 1, "compute": 2}


class FallbackShaderError(ValueError):
    """Raised when fallback shader generation fails validation or safety checks."""

    def __init__(self, message: str, *, code: str = "fallback_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class UnsafeOutputPathError(FallbackShaderError):
    """Raised when the requested output directory is a tracked repository path."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="unsafe_output_path")


@dataclass(frozen=True)
class WgslHygieneIssue:
    """Single sanitized WGSL syntax hygiene violation (never contains raw source)."""

    code: str
    line: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "line": self.line,
            "message": sanitize_text(self.message),
        }


class WgslHygieneError(FallbackShaderError):
    """Raised when a fallback WGSL source fails syntax hygiene validation."""

    def __init__(
        self,
        message: str,
        *,
        issues: Sequence[WgslHygieneIssue] = (),
        shader_key: str | None = None,
    ) -> None:
        super().__init__(sanitize_text(message), code="wgsl_hygiene_error")
        self.issues: tuple[WgslHygieneIssue, ...] = tuple(issues)
        self.shader_key: str | None = sanitize_text(shader_key) if shader_key is not None else None


# ---------------------------------------------------------------------------
# Sanitization helpers (zero host paths or usernames in JSON / diagnostics)
# ---------------------------------------------------------------------------

_FILE_URL_RE = re.compile(r"\bfile://[^\s\"'<>|,;()\[\]]+", re.IGNORECASE)
_UNC_BS_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_FS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_:])//[^\s\"'<>|,;()\[\]]+")
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_POSIX_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./])/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes|proc|run|srv)/[^\s\"'<>|,;()\[\]]*"
)


def _current_usernames(extra_usernames: Sequence[str] = ()) -> list[str]:
    """Return active OS username candidates (length >= 3) for redaction."""
    candidates: list[str] = [
        os.environ.get("USERNAME", ""),
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
        *extra_usernames,
    ]
    try:
        candidates.append(getpass.getuser())
    except (OSError, KeyError, RuntimeError):
        pass
    try:
        candidates.append(Path.home().name)
    except (OSError, RuntimeError):
        pass

    names: list[str] = []
    for candidate in candidates:
        cleaned = str(candidate).strip()
        if len(cleaned) >= 3 and cleaned not in names:
            names.append(cleaned)
    return names


def _home_path_variants() -> list[str]:
    """Return concrete home directory path strings (native + POSIX) for pre-redaction."""
    variants: list[str] = []
    try:
        home = Path.home()
        for raw in (str(home), home.as_posix()):
            cleaned = raw.rstrip("/\\")
            if len(cleaned) >= 4 and cleaned not in variants:
                variants.append(cleaned)
    except (OSError, RuntimeError):
        pass
    return variants


def sanitize_text(raw: str | None, *, extra_usernames: Sequence[str] = ()) -> str:
    """Strip host paths, drive letters, UNC paths, POSIX roots, and active usernames."""
    if raw is None:
        return ""
    text = str(raw)
    text = _FILE_URL_RE.sub("<path>", text)
    for home_str in _home_path_variants():
        text = re.sub(re.escape(home_str), "<path>", text, flags=re.IGNORECASE)
    text = _UNC_BS_PATH_RE.sub("<path>", text)
    text = _WINDOWS_PATH_RE.sub("<path>", text)
    text = _UNC_FS_PATH_RE.sub("<path>", text)
    text = _POSIX_PATH_RE.sub("<path>", text)
    for username in _current_usernames(extra_usernames):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        text = pattern.sub("<user>", text)
    return text


def sanitize_source_label(source: str | Path | None) -> str:
    """Return a strictly sanitized basename-only label (never a host path)."""
    if source is None:
        return "<unknown>"
    raw = str(source).strip()
    if not raw:
        return "<unknown>"
    if raw.startswith("<") and raw.endswith(">"):
        return sanitize_text(raw)
    stripped = raw.rstrip("/\\")
    if not stripped:
        return "<unknown>"
    parts = re.split(r"[/\\]+", stripped)
    base = parts[-1]
    if not base or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    return sanitize_text(base)


def sanitize_value(value: Any, *, extra_usernames: Sequence[str] = ()) -> Any:
    """Recursively sanitize strings/paths and redact raw bytes in JSON metadata."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return sanitize_text(value, extra_usernames=extra_usernames)
    if isinstance(value, Path):
        return sanitize_source_label(value)
    if isinstance(value, Mapping):
        return {
            sanitize_text(str(k), extra_usernames=extra_usernames): sanitize_value(
                v, extra_usernames=extra_usernames
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_value(v, extra_usernames=extra_usernames) for v in value]
    return value


# ---------------------------------------------------------------------------
# WGSL syntax hygiene validation
# ---------------------------------------------------------------------------

_WGSL_HYGIENE_TOKEN_RE = re.compile(r"/\*|\*/|//|\n|\{|\}|\blet\b")


def check_wgsl_hygiene(source: str) -> tuple[WgslHygieneIssue, ...]:
    """Validate WGSL syntax hygiene and return any deterministic issues found.

    Checks enforced:
      - Non-empty WGSL source.
      - LF newlines only (rejects CRLF `\\r\\n` and bare CR `\\r`).
      - Closed and matched nested `/* ... */` block comments (respecting `//` line comments).
      - Balanced `{}` braces outside comments.
      - No module-scope `let` declarations (WGSL requires `const` or `override` at module scope).
    """
    if not isinstance(source, str):
        raise TypeError(f"expected WGSL source str, got {type(source).__name__}")

    issues: list[WgslHygieneIssue] = []
    if not source.strip():
        issues.append(
            WgslHygieneIssue(
                code="empty_source",
                line=1,
                message="WGSL source must not be empty",
            )
        )
        return tuple(issues)

    # 1. Enforce LF-only newlines and build an LF-normalized view for line-accurate scanning.
    line_no = 1
    idx = 0
    length = len(source)
    while idx < length:
        ch = source[idx]
        if ch == "\r":
            issues.append(
                WgslHygieneIssue(
                    code="non_lf_newline",
                    line=line_no,
                    message="WGSL source must use LF newlines only (found CR)",
                )
            )
            if idx + 1 < length and source[idx + 1] == "\n":
                idx += 2
            else:
                idx += 1
            line_no += 1
            continue
        if ch == "\n":
            line_no += 1
        idx += 1

    normalized = source.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Single-pass lexical hygiene scan for comments, braces, and module-scope `let`.
    scan_line = 1
    in_line_comment = False
    open_comment_lines: list[int] = []
    open_brace_lines: list[int] = []

    for match in _WGSL_HYGIENE_TOKEN_RE.finditer(normalized):
        tok = match.group(0)
        if tok == "\n":
            scan_line += 1
            in_line_comment = False
            continue
        if in_line_comment:
            continue
        if tok == "/*":
            open_comment_lines.append(scan_line)
            continue
        if tok == "*/":
            if open_comment_lines:
                open_comment_lines.pop()
            else:
                issues.append(
                    WgslHygieneIssue(
                        code="unmatched_block_comment_close",
                        line=scan_line,
                        message="unexpected closing block comment '*/'",
                    )
                )
            continue
        if open_comment_lines:
            continue
        if tok == "//":
            in_line_comment = True
            continue
        if tok == "{":
            open_brace_lines.append(scan_line)
            continue
        if tok == "}":
            if open_brace_lines:
                open_brace_lines.pop()
            else:
                issues.append(
                    WgslHygieneIssue(
                        code="unbalanced_braces",
                        line=scan_line,
                        message="unexpected closing brace '}'",
                    )
                )
            continue
        if tok == "let" and not open_brace_lines:
            issues.append(
                WgslHygieneIssue(
                    code="module_scope_let",
                    line=scan_line,
                    message="module-scope 'let' declaration is invalid in WGSL (use 'const' or 'override')",
                )
            )

    for open_line in open_comment_lines:
        issues.append(
            WgslHygieneIssue(
                code="unclosed_block_comment",
                line=open_line,
                message="unclosed block comment '/*'",
            )
        )
    for open_line in open_brace_lines:
        issues.append(
            WgslHygieneIssue(
                code="unbalanced_braces",
                line=open_line,
                message="unclosed opening brace '{'",
            )
        )

    issues.sort(key=lambda item: (item.line, item.code, item.message))
    return tuple(issues)


def validate_wgsl_hygiene(source: str, *, shader_key: str | None = None) -> None:
    """Raise `WgslHygieneError` if `source` fails any WGSL syntax hygiene check."""
    issues = check_wgsl_hygiene(source)
    if not issues:
        return
    label = f"shader '{sanitize_text(shader_key)}'" if shader_key else "WGSL shader"
    details = "; ".join(f"line {issue.line} [{issue.code}]: {issue.message}" for issue in issues)
    raise WgslHygieneError(
        f"{label} failed WGSL hygiene validation ({len(issues)} issue(s)): {details}",
        issues=issues,
        shader_key=shader_key,
    )


# ---------------------------------------------------------------------------
# Built-in original WGSL fallback shaders (zero copyrighted game code)
# ---------------------------------------------------------------------------

SPLASH_VS_WGSL = """\
struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VsOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}
"""

SPLASH_FS_WGSL = """\
struct SplashUniforms {
    tint: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: SplashUniforms;
@group(0) @binding(1) var splash_tex: texture_2d<f32>;
@group(0) @binding(2) var splash_sampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let sample = textureSample(splash_tex, splash_sampler, uv);
    return sample * uniforms.tint;
}
"""

DEBUG_MESH_VS_WGSL = """\
struct Camera {
    mvp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
}

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.position = camera.mvp * vec4<f32>(position, 1.0);
    out.world_pos = position;
    return out;
}
"""

DEBUG_MESH_FS_WGSL = """\
struct Camera {
    mvp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

@fragment
fn fs_main(@location(0) world_pos: vec3<f32>) -> @location(0) vec4<f32> {
    let grid = abs(fract(world_pos.xz * 0.5) - vec2<f32>(0.5, 0.5));
    let line = step(vec2<f32>(0.48, 0.48), grid);
    let intensity = max(line.x, line.y);
    let base = vec3<f32>(0.12, 0.14, 0.18);
    let accent = vec3<f32>(0.20, 0.85, 0.45);
    let color = mix(base, accent, intensity);
    // Keep camera in the live binding set so group 0 matches the vertex stage.
    let _keep = camera.mvp[0][0] * 0.0;
    return vec4<f32>(color + vec3<f32>(_keep, _keep, _keep), 1.0);
}
"""

UNLIT_MESH_VS_WGSL = """\
struct MeshUniforms {
    mvp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: MeshUniforms;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
) -> VsOut {
    var out: VsOut;
    out.position = uniforms.mvp * vec4<f32>(position, 1.0);
    out.uv = uv;
    return out;
}
"""

UNLIT_MESH_FS_WGSL = """\
struct MeshUniforms {
    mvp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: MeshUniforms;
@group(0) @binding(1) var albedo_tex: texture_2d<f32>;
@group(0) @binding(2) var albedo_sampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let sample = textureSample(albedo_tex, albedo_sampler, uv);
    let _keep = uniforms.mvp[0][0] * 0.0;
    return sample + vec4<f32>(_keep, _keep, _keep, 0.0);
}
"""

COMPUTE_CLEAR_WGSL = """\
struct ClearParams {
    value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: ClearParams;
@group(0) @binding(1) var<storage, read_write> buffer: array<u32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < arrayLength(&buffer)) {
        buffer[gid.x] = params.value;
    }
}
"""


# ---------------------------------------------------------------------------
# Reflection helpers (matching shaders/src/reflect.rs content hashes)
# ---------------------------------------------------------------------------


def format_shader_hash(shader_hash: int) -> str:
    return wgpk.format_shader_hash(shader_hash)


def format_wgsl_filename(shader_hash: int) -> str:
    return f"{wgpk.normalize_shader_hash(shader_hash):016x}.wgsl"


def _layout_member_json(entry: Mapping[str, Any]) -> str:
    if "buffer" in entry:
        buf = entry["buffer"]
        return (
            f'"buffer":{{"type":{json.dumps(buf["type"])},'
            f'"hasDynamicOffset":{str(bool(buf.get("hasDynamicOffset", False))).lower()},'
            f'"minBindingSize":{int(buf.get("minBindingSize", 0))}}}'
        )
    if "sampler" in entry:
        return f'"sampler":{{"type":{json.dumps(entry["sampler"]["type"])}}}'
    if "texture" in entry:
        tex = entry["texture"]
        return (
            f'"texture":{{"sampleType":{json.dumps(tex["sampleType"])},'
            f'"viewDimension":{json.dumps(tex["viewDimension"])},'
            f'"multisampled":{str(bool(tex.get("multisampled", False))).lower()}}}'
        )
    if "storageTexture" in entry:
        st = entry["storageTexture"]
        return (
            f'"storageTexture":{{"access":{json.dumps(st["access"])},'
            f'"format":{json.dumps(st["format"])},'
            f'"viewDimension":{json.dumps(st["viewDimension"])}}}'
        )
    if "externalTexture" in entry:
        return '"externalTexture":{}'
    raise FallbackShaderError(f"layout entry missing resource member: {entry!r}")


def serialize_layout_entry_json(entry: Mapping[str, Any]) -> str:
    return (
        f'{{"binding":{int(entry["binding"])},'
        f'"visibility":{int(entry["visibility"])},'
        f"{_layout_member_json(entry)}}}"
    )


def bind_group_layout_content_hash(entries: Sequence[Mapping[str, Any]]) -> int:
    ordered = sorted(entries, key=lambda item: int(item["binding"]))
    canonical = "[" + ",".join(serialize_layout_entry_json(e) for e in ordered) + "]"
    return wgpk.fnv1a_64(f"bgl:{canonical}")


def pipeline_layout_content_hash(bgl_ids: Sequence[int]) -> int:
    formatted = ",".join(format_shader_hash(i) for i in bgl_ids)
    return wgpk.fnv1a_64(f"pl:[{formatted}]")


def _buffer_entry(
    binding: int,
    visibility: int,
    *,
    buffer_type: str = "uniform",
    min_binding_size: int,
) -> dict[str, Any]:
    return {
        "binding": binding,
        "visibility": visibility,
        "buffer": {
            "type": buffer_type,
            "hasDynamicOffset": False,
            "minBindingSize": min_binding_size,
        },
    }


def _sampler_entry(binding: int, visibility: int, *, sampler_type: str = "filtering") -> dict[str, Any]:
    return {
        "binding": binding,
        "visibility": visibility,
        "sampler": {"type": sampler_type},
    }


def _texture_entry(
    binding: int,
    visibility: int,
    *,
    sample_type: str = "float",
    view_dimension: str = "2d",
    multisampled: bool = False,
) -> dict[str, Any]:
    return {
        "binding": binding,
        "visibility": visibility,
        "texture": {
            "sampleType": sample_type,
            "viewDimension": view_dimension,
            "multisampled": multisampled,
        },
    }


def _binding_record(
    *,
    group: int,
    binding: int,
    name: str,
    resource: str,
    register_class: str,
) -> dict[str, Any]:
    return {
        "group": group,
        "binding": binding,
        "name": name,
        "resource": resource,
        "in_layout": True,
        "register": {"class": register_class, "index": binding, "space": group},
        "spirv": {"set": group, "binding": binding},
        "root_constants": False,
        "binding_array": None,
    }


def _clone_layout_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "binding": int(entry["binding"]),
        "visibility": int(entry["visibility"]),
    }
    if "buffer" in entry:
        buf = entry["buffer"]
        out["buffer"] = {
            "type": buf["type"],
            "hasDynamicOffset": bool(buf.get("hasDynamicOffset", False)),
            "minBindingSize": int(buf.get("minBindingSize", 0)),
        }
    elif "sampler" in entry:
        out["sampler"] = {"type": entry["sampler"]["type"]}
    elif "texture" in entry:
        tex = entry["texture"]
        out["texture"] = {
            "sampleType": tex["sampleType"],
            "viewDimension": tex["viewDimension"],
            "multisampled": bool(tex.get("multisampled", False)),
        }
    elif "storageTexture" in entry:
        st = entry["storageTexture"]
        out["storageTexture"] = {
            "access": st["access"],
            "format": st["format"],
            "viewDimension": st["viewDimension"],
        }
    elif "externalTexture" in entry:
        out["externalTexture"] = {}
    return out


def _layout_member_kind(entry: Mapping[str, Any]) -> str:
    for key in ("buffer", "sampler", "texture", "storageTexture", "externalTexture"):
        if key in entry:
            return key
    raise FallbackShaderError(f"unknown layout member in {entry!r}")


def merge_layout_entries(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two GPUBindGroupLayoutEntry dicts (shaders/src/reflect.rs rules)."""
    existing_kind = _layout_member_kind(existing)
    incoming_kind = _layout_member_kind(incoming)
    if existing_kind != incoming_kind:
        raise FallbackShaderError(
            f"resource kind mismatch: {existing_kind} vs {incoming_kind}",
            code="binding_conflict",
        )

    visibility = int(existing["visibility"]) | int(incoming["visibility"])
    binding = int(existing["binding"])

    if existing_kind == "buffer":
        a = existing["buffer"]
        b = incoming["buffer"]
        if a["type"] != b["type"]:
            raise FallbackShaderError(
                f"buffer type mismatch: {a['type']} vs {b['type']}",
                code="binding_conflict",
            )
        if bool(a.get("hasDynamicOffset", False)) != bool(b.get("hasDynamicOffset", False)):
            raise FallbackShaderError("buffer hasDynamicOffset mismatch", code="binding_conflict")
        return {
            "binding": binding,
            "visibility": visibility,
            "buffer": {
                "type": a["type"],
                "hasDynamicOffset": bool(a.get("hasDynamicOffset", False)),
                "minBindingSize": max(int(a.get("minBindingSize", 0)), int(b.get("minBindingSize", 0))),
            },
        }

    if existing_kind == "sampler":
        if existing["sampler"]["type"] != incoming["sampler"]["type"]:
            raise FallbackShaderError("sampler type mismatch", code="binding_conflict")
        return {
            "binding": binding,
            "visibility": visibility,
            "sampler": {"type": existing["sampler"]["type"]},
        }

    if existing_kind == "texture":
        a = existing["texture"]
        b = incoming["texture"]
        if a["viewDimension"] != b["viewDimension"]:
            raise FallbackShaderError("texture viewDimension mismatch", code="binding_conflict")
        if bool(a.get("multisampled", False)) != bool(b.get("multisampled", False)):
            raise FallbackShaderError("texture multisampled mismatch", code="binding_conflict")
        sample_type = a["sampleType"]
        if a["sampleType"] != b["sampleType"]:
            float_pair = {a["sampleType"], b["sampleType"]} == {"float", "unfilterable-float"}
            if not float_pair:
                raise FallbackShaderError("texture sampleType mismatch", code="binding_conflict")
            sample_type = "float"
        return {
            "binding": binding,
            "visibility": visibility,
            "texture": {
                "sampleType": sample_type,
                "viewDimension": a["viewDimension"],
                "multisampled": bool(a.get("multisampled", False)),
            },
        }

    if existing_kind == "storageTexture":
        a = existing["storageTexture"]
        b = incoming["storageTexture"]
        if (
            a["access"] != b["access"]
            or a["format"] != b["format"]
            or a["viewDimension"] != b["viewDimension"]
        ):
            raise FallbackShaderError("storageTexture mismatch", code="binding_conflict")
        return {
            "binding": binding,
            "visibility": visibility,
            "storageTexture": {
                "access": a["access"],
                "format": a["format"],
                "viewDimension": a["viewDimension"],
            },
        }

    return {"binding": binding, "visibility": visibility, "externalTexture": {}}


def merge_group_layouts(stage_layouts: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Merge per-stage group-0..N layouts into dense pipeline bind-group layouts."""
    groups: dict[int, dict[int, dict[str, Any]]] = {}
    max_group = -1
    for layout_groups in stage_layouts:
        for group_idx, entries in enumerate(layout_groups):
            max_group = max(max_group, group_idx)
            slot = groups.setdefault(group_idx, {})
            for entry in entries:
                binding = int(entry["binding"])
                if binding not in slot:
                    slot[binding] = _clone_layout_entry(entry)
                else:
                    slot[binding] = merge_layout_entries(slot[binding], entry)

    if max_group < 0:
        return []

    merged: list[dict[str, Any]] = []
    for group_idx in range(max_group + 1):
        entries = list(groups.get(group_idx, {}).values())
        entries.sort(key=lambda item: int(item["binding"]))
        merged.append({"entries": entries})
    return merged


# ---------------------------------------------------------------------------
# Fallback shader catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FallbackShader:
    key: str
    stage: str
    entry_point: str
    wgsl: str
    layout_entries: tuple[dict[str, Any], ...]
    bindings: tuple[dict[str, Any], ...]
    vertex_inputs: tuple[dict[str, Any], ...] = ()
    fragment_outputs: tuple[dict[str, Any], ...] = ()
    workgroup_size: tuple[int, int, int] | None = None
    pipeline_key: str | None = None

    @property
    def hash(self) -> int:
        return wgpk.fnv1a_64(self.wgsl)

    @property
    def hash_hex(self) -> str:
        return format_shader_hash(self.hash)


def builtin_fallback_shaders() -> tuple[FallbackShader, ...]:
    """Return the fixed, deterministic catalog of fallback shaders."""
    splash_vs = FallbackShader(
        key="splash_vs",
        stage="vertex",
        entry_point="vs_main",
        wgsl=SPLASH_VS_WGSL,
        layout_entries=(),
        bindings=(),
        vertex_inputs=(),
        pipeline_key="splash",
    )
    splash_fs = FallbackShader(
        key="splash_fs",
        stage="fragment",
        entry_point="fs_main",
        wgsl=SPLASH_FS_WGSL,
        layout_entries=(
            _buffer_entry(0, VISIBILITY_FRAGMENT, min_binding_size=16),
            _texture_entry(1, VISIBILITY_FRAGMENT),
            _sampler_entry(2, VISIBILITY_FRAGMENT),
        ),
        bindings=(
            _binding_record(group=0, binding=0, name="uniforms", resource="buffer", register_class="b"),
            _binding_record(group=0, binding=1, name="splash_tex", resource="texture", register_class="t"),
            _binding_record(group=0, binding=2, name="splash_sampler", resource="sampler", register_class="s"),
        ),
        fragment_outputs=({"location": 0, "type": "vec4<f32>"},),
        pipeline_key="splash",
    )
    debug_vs = FallbackShader(
        key="debug_mesh_vs",
        stage="vertex",
        entry_point="vs_main",
        wgsl=DEBUG_MESH_VS_WGSL,
        layout_entries=(_buffer_entry(0, VISIBILITY_VERTEX, min_binding_size=64),),
        bindings=(
            _binding_record(group=0, binding=0, name="camera", resource="buffer", register_class="b"),
        ),
        vertex_inputs=({"location": 0, "type": "vec3<f32>", "format": "float32x3"},),
        pipeline_key="debug_mesh",
    )
    debug_fs = FallbackShader(
        key="debug_mesh_fs",
        stage="fragment",
        entry_point="fs_main",
        wgsl=DEBUG_MESH_FS_WGSL,
        layout_entries=(_buffer_entry(0, VISIBILITY_FRAGMENT, min_binding_size=64),),
        bindings=(
            _binding_record(group=0, binding=0, name="camera", resource="buffer", register_class="b"),
        ),
        fragment_outputs=({"location": 0, "type": "vec4<f32>"},),
        pipeline_key="debug_mesh",
    )
    unlit_vs = FallbackShader(
        key="unlit_mesh_vs",
        stage="vertex",
        entry_point="vs_main",
        wgsl=UNLIT_MESH_VS_WGSL,
        layout_entries=(_buffer_entry(0, VISIBILITY_VERTEX, min_binding_size=64),),
        bindings=(
            _binding_record(group=0, binding=0, name="uniforms", resource="buffer", register_class="b"),
        ),
        vertex_inputs=(
            {"location": 0, "type": "vec3<f32>", "format": "float32x3"},
            {"location": 1, "type": "vec2<f32>", "format": "float32x2"},
        ),
        pipeline_key="unlit_mesh",
    )
    unlit_fs = FallbackShader(
        key="unlit_mesh_fs",
        stage="fragment",
        entry_point="fs_main",
        wgsl=UNLIT_MESH_FS_WGSL,
        layout_entries=(
            _buffer_entry(0, VISIBILITY_FRAGMENT, min_binding_size=64),
            _texture_entry(1, VISIBILITY_FRAGMENT),
            _sampler_entry(2, VISIBILITY_FRAGMENT),
        ),
        bindings=(
            _binding_record(group=0, binding=0, name="uniforms", resource="buffer", register_class="b"),
            _binding_record(group=0, binding=1, name="albedo_tex", resource="texture", register_class="t"),
            _binding_record(group=0, binding=2, name="albedo_sampler", resource="sampler", register_class="s"),
        ),
        fragment_outputs=({"location": 0, "type": "vec4<f32>"},),
        pipeline_key="unlit_mesh",
    )
    compute = FallbackShader(
        key="compute_clear",
        stage="compute",
        entry_point="cs_main",
        wgsl=COMPUTE_CLEAR_WGSL,
        layout_entries=(
            _buffer_entry(0, VISIBILITY_COMPUTE, min_binding_size=16),
            _buffer_entry(1, VISIBILITY_COMPUTE, buffer_type="storage", min_binding_size=4),
        ),
        bindings=(
            _binding_record(group=0, binding=0, name="params", resource="buffer", register_class="b"),
            _binding_record(group=0, binding=1, name="buffer", resource="buffer", register_class="u"),
        ),
        workgroup_size=(64, 1, 1),
        pipeline_key=None,
    )
    return canonicalize_shader_catalog(
        (
            splash_vs,
            splash_fs,
            debug_vs,
            debug_fs,
            unlit_vs,
            unlit_fs,
            compute,
        )
    )


def is_canonical_shader_order(shaders: Sequence[FallbackShader]) -> bool:
    """Return True iff `shaders` are in strictly ascending 64-bit hash order."""
    previous_hash: int | None = None
    for shader in shaders:
        current = shader.hash
        if previous_hash is not None and current <= previous_hash:
            return False
        previous_hash = current
    return True


def canonicalize_shader_catalog(
    shaders: Sequence[FallbackShader],
) -> tuple[FallbackShader, ...]:
    """Validate WGSL hygiene, stage names, and hash uniqueness, then sort canonically.

    Canonical generation order matches `shaders/src/pack.rs` (`is_canonical_hash_order`):
    strictly ascending unsigned 64-bit FNV-1a shader hash.
    """
    validated: list[FallbackShader] = []
    seen_hashes: dict[int, str] = {}

    for shader in shaders:
        if shader.stage not in STAGE_ORDER:
            raise FallbackShaderError(
                f"invalid shader stage for '{sanitize_text(shader.key)}': {sanitize_text(shader.stage)!r}",
                code="invalid_shader_stage",
            )
        validate_wgsl_hygiene(shader.wgsl, shader_key=shader.key)
        shader_hash = shader.hash
        if shader_hash in seen_hashes:
            raise FallbackShaderError(
                f"duplicate shader hash {format_shader_hash(shader_hash)} "
                f"for '{sanitize_text(seen_hashes[shader_hash])}' and '{sanitize_text(shader.key)}'",
                code="duplicate_shader_hash",
            )
        seen_hashes[shader_hash] = shader.key
        validated.append(shader)

    validated.sort(key=lambda item: (item.hash, STAGE_ORDER[item.stage], item.key))
    return tuple(validated)


# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------


def _is_under(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def is_safe_output_dir(path: Path | str, *, repo_root: Path | str = REPO_ROOT) -> bool:
    """Return True iff `path` may receive generated fallback shader artifacts.

    Allowed destinations:
      - `<repo>/local-data/**` or `<repo>/work/**` (gitignored)
      - any path outside the repository (temp dirs used by tests)
    """
    target = Path(path).expanduser().resolve()
    root = Path(repo_root).resolve()

    if not _is_under(target, root):
        return True

    for name in ALLOWED_REPO_OUTPUT_DIR_NAMES:
        if _is_under(target, root / name):
            return True
    return False


def validate_output_dir(path: Path | str, *, repo_root: Path | str = REPO_ROOT) -> Path:
    """Resolve and validate an output directory, raising on tracked source paths."""
    target = Path(path).expanduser().resolve()
    if not is_safe_output_dir(target, repo_root=repo_root):
        raise UnsafeOutputPathError(
            "Refusing to write generated shader artifacts into a tracked repository "
            "source directory. Point --out-dir at gitignored local-data/, work/, or a temp directory."
        )
    return target


# ---------------------------------------------------------------------------
# Pack / index / reflection builders
# ---------------------------------------------------------------------------


@dataclass
class GeneratedFallbackArtifacts:
    pack_name: str
    pack_bytes: bytes
    shader_index: dict[str, Any]
    reflection_manifest: dict[str, Any]
    shaders: tuple[FallbackShader, ...]
    pack_ranges: tuple[tuple[int, int], ...] = ()
    shader_index_text: str = ""
    reflection_manifest_text: str = ""

    def summary(self) -> dict[str, Any]:
        safe_pack_name = sanitize_source_label(self.pack_name)
        return sanitize_value(
            {
                "ok": True,
                "packName": safe_pack_name,
                "packBytes": len(self.pack_bytes),
                "entryCount": len(self.shaders),
                "shaderHashes": [shader.hash_hex for shader in self.shaders],
                "canonicalOrder": is_canonical_shader_order(self.shaders),
                "wgslHygieneChecked": True,
                "stageCounts": {
                    "vertex": sum(1 for s in self.shaders if s.stage == "vertex"),
                    "fragment": sum(1 for s in self.shaders if s.stage == "fragment"),
                    "compute": sum(1 for s in self.shaders if s.stage == "compute"),
                },
                "outputs": [
                    safe_pack_name,
                    SHADER_INDEX_FILE_NAME,
                    REFLECTION_MANIFEST_FILE_NAME,
                ],
            }
        )


def _stage_group_layouts(shader: FallbackShader) -> list[list[dict[str, Any]]]:
    if not shader.layout_entries:
        return []
    return [[_clone_layout_entry(entry) for entry in shader.layout_entries]]


def _intern_pipeline_layout(
    bind_group_layouts: dict[str, dict[str, Any]],
    pipeline_layouts: dict[str, dict[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    bgl_ids: list[str] = []
    for group in groups:
        entries = [_clone_layout_entry(e) for e in group.get("entries", [])]
        entries.sort(key=lambda item: int(item["binding"]))
        bgl_id = format_shader_hash(bind_group_layout_content_hash(entries))
        bind_group_layouts.setdefault(bgl_id, {"entries": entries})
        bgl_ids.append(bgl_id)
    pl_id = format_shader_hash(
        pipeline_layout_content_hash([wgpk.normalize_shader_hash(i) for i in bgl_ids])
    )
    pipeline_layouts.setdefault(pl_id, {"bindGroupLayouts": list(bgl_ids)})
    return pl_id, bgl_ids


def build_reflection_manifest(shaders: Sequence[FallbackShader]) -> dict[str, Any]:
    """Build a canonical, sanitized S6 schema_version=1 reflection manifest."""
    ordered_shaders = sorted(
        shaders,
        key=lambda item: (item.hash, STAGE_ORDER.get(item.stage, 99), item.key),
    )
    bind_group_layouts: dict[str, dict[str, Any]] = {}
    pipeline_layouts: dict[str, dict[str, Any]] = {}
    shader_records: dict[str, dict[str, Any]] = {}

    for shader in ordered_shaders:
        groups = [{"entries": entries} for entries in _stage_group_layouts(shader)]
        # Dense group list: no groups => empty pipeline layout (no bind groups).
        pl_id, bgl_ids = _intern_pipeline_layout(bind_group_layouts, pipeline_layouts, groups)
        sorted_bindings = sorted(
            (dict(b) for b in shader.bindings),
            key=lambda item: (
                int(item.get("group", 0)),
                int(item.get("binding", 0)),
                str(item.get("name", "")),
            ),
        )
        sorted_vertex_inputs = sorted(
            (dict(v) for v in shader.vertex_inputs),
            key=lambda item: int(item.get("location", 0)),
        )
        sorted_fragment_outputs = sorted(
            (dict(f) for f in shader.fragment_outputs),
            key=lambda item: int(item.get("location", 0)),
        )
        record: dict[str, Any] = {
            "stage": shader.stage,
            "entry_point": shader.entry_point,
            "pipeline_layout": pl_id,
            "bind_group_layouts": bgl_ids,
            "layout_complete": True,
            "workgroup_size": list(shader.workgroup_size) if shader.workgroup_size else None,
            "bindings": sorted_bindings,
            "vertex_inputs": sorted_vertex_inputs,
            "fragment_outputs": sorted_fragment_outputs,
            "requires_lowering": [],
            "limit_violations": [],
        }
        shader_records[shader.hash_hex] = record

    # Pre-merged graphics pipelines for each vs+fs pair.
    pipelines: dict[str, dict[str, Any]] = {}
    by_pipeline: dict[str, list[FallbackShader]] = {}
    for shader in ordered_shaders:
        if shader.pipeline_key:
            by_pipeline.setdefault(shader.pipeline_key, []).append(shader)

    for pipeline_name in sorted(by_pipeline):
        members = by_pipeline[pipeline_name]
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda s: (STAGE_ORDER[s.stage], s.hash))
        stage_layouts = [_stage_group_layouts(s) for s in ordered]
        merged_groups = merge_group_layouts(stage_layouts)
        pl_id, bgl_ids = _intern_pipeline_layout(
            bind_group_layouts,
            pipeline_layouts,
            merged_groups,
        )
        stages = {s.stage: s.hash_hex for s in ordered}
        pipeline_key = "+".join(s.hash_hex for s in ordered)
        pipelines[pipeline_key] = {
            "stages": stages,
            "pipeline_layout": pl_id,
            "bind_group_layouts": bgl_ids,
            "layout_complete": True,
            "limit_violations": [],
        }

    # Deterministic key ordering via sorted insertion into plain dicts for json.dumps.
    manifest = {
        "schema_version": LAYOUT_MANIFEST_SCHEMA_VERSION,
        "register_convention": {
            "source": "dxc-spirv",
            "shifts": {"b": 0, "t": 0, "u": 0, "s": 0},
        },
        "bind_group_layouts": {
            key: bind_group_layouts[key] for key in sorted(bind_group_layouts)
        },
        "pipeline_layouts": {
            key: pipeline_layouts[key] for key in sorted(pipeline_layouts)
        },
        "shaders": {
            key: shader_records[key] for key in sorted(shader_records)
        },
        "pipelines": {
            key: pipelines[key] for key in sorted(pipelines)
        },
    }
    return sanitize_value(manifest)


def render_shader_index_json(
    pack_name: str,
    shaders: Sequence[FallbackShader],
    ranges: Sequence[tuple[int, int]],
) -> str:
    """Render sanitized `shader-index.json` matching `shaders/src/pack.rs` / docs/gpu_commands.md."""
    safe_pack_name = sanitize_source_label(pack_name)
    entries = sorted(
        zip(shaders, ranges, strict=True),
        key=lambda item: item[0].hash,
    )

    lines: list[str] = [
        "{",
        f'  "abi_version": {SHADER_INDEX_ABI_VERSION},',
        f'  "pack": {json.dumps(safe_pack_name)},',
    ]
    if not entries:
        lines.append('  "shaders": {}')
    else:
        lines.append('  "shaders": {')
        for idx, (shader, (offset, length)) in enumerate(entries):
            key = shader.hash_hex
            wgsl_path = f"shaders/{format_wgsl_filename(shader.hash)}"
            block = [
                f"    {json.dumps(key)}: {{",
                f'      "stage": {json.dumps(sanitize_text(shader.stage))},',
                f'      "wgsl": {json.dumps(wgsl_path)},',
                f'      "entry_point": {json.dumps(sanitize_text(shader.entry_point))},',
                f'      "pack_offset": {offset},',
                f'      "pack_length": {length}',
                "    }",
            ]
            chunk = "\n".join(block)
            if idx + 1 < len(entries):
                chunk += ","
            lines.append(chunk)
        lines.append("  }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def render_reflection_manifest_json(manifest: Mapping[str, Any]) -> str:
    """Pretty-print the sanitized reflection manifest with stable 2-space indentation."""
    return json.dumps(sanitize_value(manifest), indent=2, sort_keys=False) + "\n"


def build_fallback_artifacts(
    *,
    pack_name: str = DEFAULT_PACK_NAME,
    shaders: Sequence[FallbackShader] | None = None,
) -> GeneratedFallbackArtifacts:
    """Build in-memory pack bytes + index + reflection manifest (no disk I/O)."""
    if not pack_name or "/" in pack_name or "\\" in pack_name:
        raise FallbackShaderError(
            "pack-name must be a plain file name (no directory separators)",
            code="invalid_pack_name",
        )
    if not pack_name.endswith(".wgsl.pack"):
        raise FallbackShaderError(
            "pack-name must end with .wgsl.pack",
            code="invalid_pack_name",
        )

    raw_catalog = shaders if shaders is not None else builtin_fallback_shaders()
    catalog = canonicalize_shader_catalog(raw_catalog)

    pack_entries = [
        {"hash": shader.hash, "stage": shader.stage, "wgsl": shader.wgsl}
        for shader in catalog
    ]
    pack_bytes = wgpk.build_wgpk(pack_entries, flags=0)
    bundle = wgpk.parse_wgpk(pack_bytes)
    ranges = tuple((entry.offset, entry.length) for entry in bundle.entries)

    shader_index_text = render_shader_index_json(pack_name, catalog, ranges)
    shader_index = json.loads(shader_index_text)
    reflection = build_reflection_manifest(catalog)
    reflection_text = render_reflection_manifest_json(reflection)

    return GeneratedFallbackArtifacts(
        pack_name=pack_name,
        pack_bytes=pack_bytes,
        shader_index=shader_index,
        reflection_manifest=reflection,
        shaders=catalog,
        pack_ranges=ranges,
        shader_index_text=shader_index_text,
        reflection_manifest_text=reflection_text,
    )


def write_fallback_artifacts(
    out_dir: Path | str,
    *,
    pack_name: str = DEFAULT_PACK_NAME,
    repo_root: Path | str = REPO_ROOT,
    shaders: Sequence[FallbackShader] | None = None,
) -> GeneratedFallbackArtifacts:
    """Validate safety, build artifacts, and write them under `out_dir`."""
    destination = validate_output_dir(out_dir, repo_root=repo_root)
    artifacts = build_fallback_artifacts(pack_name=pack_name, shaders=shaders)
    destination.mkdir(parents=True, exist_ok=True)

    pack_path = destination / artifacts.pack_name
    index_path = destination / SHADER_INDEX_FILE_NAME
    reflection_path = destination / REFLECTION_MANIFEST_FILE_NAME

    pack_path.write_bytes(artifacts.pack_bytes)
    index_path.write_text(artifacts.shader_index_text, encoding="utf-8", newline="\n")
    reflection_path.write_text(
        artifacts.reflection_manifest_text, encoding="utf-8", newline="\n"
    )
    return artifacts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen_fallback_shaders.py",
        description=(
            "Generate a deterministic WGPK v1 fallback.wgsl.pack plus matching "
            "shader-index.json and S6 reflection-manifest.json into a gitignored "
            "output directory (default: local-data/shaders)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: local-data/shaders). Must be gitignored or outside the repo.",
    )
    parser.add_argument(
        "--pack-name",
        default=DEFAULT_PACK_NAME,
        help=f"Output pack file name (default: {DEFAULT_PACK_NAME}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit a machine-readable summary (basenames only; no host paths).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        artifacts = write_fallback_artifacts(
            args.out_dir,
            pack_name=args.pack_name,
            repo_root=args.repo_root,
        )
    except UnsafeOutputPathError as exc:
        error_payload = sanitize_value(
            {
                "ok": False,
                "errorCode": exc.code,
                "error": exc.message,
            }
        )
        if args.as_json:
            print(json.dumps(error_payload, indent=2))
        else:
            print(f"Error: {error_payload['error']}", file=sys.stderr)
        return 1
    except FallbackShaderError as exc:
        error_payload = sanitize_value(
            {
                "ok": False,
                "errorCode": exc.code,
                "error": exc.message,
            }
        )
        if args.as_json:
            print(json.dumps(error_payload, indent=2))
        else:
            print(f"Error: {error_payload['error']}", file=sys.stderr)
        return 1
    except OSError as exc:
        message = sanitize_text(
            f"failed to write fallback shader artifacts: {exc.strerror or type(exc).__name__}"
        )
        if args.as_json:
            print(
                json.dumps(
                    sanitize_value(
                        {
                            "ok": False,
                            "errorCode": "io_error",
                            "error": message,
                        }
                    ),
                    indent=2,
                )
            )
        else:
            print(f"Error: {message}", file=sys.stderr)
        return 1

    summary = artifacts.summary()
    if args.as_json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            sanitize_text(
                f"Wrote {summary['entryCount']} fallback shaders -> {summary['packName']} "
                f"({summary['packBytes']} bytes), {SHADER_INDEX_FILE_NAME}, "
                f"{REFLECTION_MANIFEST_FILE_NAME}"
            )
        )
        print(
            "  stages: "
            f"vertex={summary['stageCounts']['vertex']}, "
            f"fragment={summary['stageCounts']['fragment']}, "
            f"compute={summary['stageCounts']['compute']}"
        )
        for shader_hash in summary["shaderHashes"]:
            print(f"  - {shader_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
