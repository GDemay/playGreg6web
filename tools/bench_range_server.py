#!/usr/bin/env python3
"""Local HTTP 206 Range and caching streaming benchmark CLI for playgta6web (Track 5 / W25 / T63).

Benchmarks `server/serve_local.py` (or any target HTTP/HTTPS URL) with:
- Concurrent HTTP `206 Partial Content` byte-range reads across mixed strided and
  deterministic pseudo-random offsets, verifying status `206`, exact returned byte
  counts, `Content-Range`, and Cross-Origin Isolation headers (COOP/COEP/CORP +
  `Accept-Ranges: bytes`).
- Simulated 8-lane HTTP/1.1 Keep-Alive boot-burst mode (`--8lane-boot-profile`)
  issuing concurrent `2 MiB` DirectStorage slabs + `64 KiB` Bink2/`shaders.rpf`
  probes with strong `ETag` / `If-Match` validation and TCP socket reuse accounting.
- Conditional `If-None-Match` (`304 Not Modified`) revalidation latency probe.

Security & privacy invariant:
- Machine-readable `--json` output NEVER includes raw host filesystem paths or usernames.
- Strictly uses Python 3.12 standard library modules only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import http.client
from http import HTTPStatus
from http.server import ThreadingHTTPServer
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE_BYTES = 2 * 1024 * 1024  # 2 MiB
DEFAULT_CHUNK_BYTES = 256 * 1024  # 256 KiB
DEFAULT_REQUESTS = 64
DEFAULT_CONCURRENCY = 4
DEFAULT_SEED = 1337
DEFAULT_TIMEOUT_SEC = 10.0

DEFAULT_BOOT_LANES = 8
DEFAULT_BOOT_SLAB_BYTES = 2 * 1024 * 1024  # 2 MiB DirectStorage slabs
DEFAULT_BOOT_PROBE_BYTES = 64 * 1024  # 64 KiB Bink2 / shaders.rpf probes
DEFAULT_BOOT_FILE_BYTES = 8 * 1024 * 1024  # 8 MiB synthetic archive for self-test
BOOT_PROFILE_NAME = "8lane-boot-profile"
MULTI_LANE_PROFILE_NAME = "multi-lane-range"

CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)

REQUIRED_ISOLATION_HEADERS: dict[str, str] = {
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-embedder-policy": "require-corp",
    "cross-origin-resource-policy": "same-origin",
    "accept-ranges": "bytes",
}


@dataclass(frozen=True)
class BenchmarkResult:
    """Sanitized benchmark metrics for HTTP 206 Range & 304 revalidation workloads."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_bytes_transferred: int
    elapsed_sec: float
    throughput_mbps: float
    requests_per_sec: float
    latency_ms: dict[str, float]
    not_modified_latency_ms: float | None
    coop_coep_valid: bool
    etag_observed: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to camelCase JSON-safe dict with zero host paths."""
        return {
            "totalRequests": self.total_requests,
            "successfulRequests": self.successful_requests,
            "failedRequests": self.failed_requests,
            "totalBytesTransferred": self.total_bytes_transferred,
            "elapsedSec": round(self.elapsed_sec, 6),
            "throughputMBps": round(self.throughput_mbps, 4),
            "requestsPerSec": round(self.requests_per_sec, 4),
            "latencyMs": {k: round(v, 4) for k, v in self.latency_ms.items()},
            "notModifiedLatencyMs": (
                round(self.not_modified_latency_ms, 4)
                if self.not_modified_latency_ms is not None
                else None
            ),
            "coopCoepValid": self.coop_coep_valid,
            "etagObserved": self.etag_observed,
        }


@dataclass(frozen=True)
class BootRangeRequest:
    """One scheduled byte-range read belonging to a specific parallel HTTP/1.1 lane."""

    lane: int
    kind: str  # "slab" or "probe"
    start: int
    end: int


@dataclass(frozen=True)
class BootProfileResult:
    """Sanitized benchmark metrics for multi-lane HTTP/1.1 Keep-Alive Range bursts."""

    profile: str
    lanes: int
    keep_alive: bool
    verify_etag: bool
    slab_bytes: int
    probe_bytes: int
    total_requests: int
    slab_requests: int
    probe_requests: int
    successful_requests: int
    error_count: int
    total_bytes_transferred: int
    elapsed_sec: float
    mb_per_sec: float
    requests_per_sec: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    latency_ms: dict[str, float]
    slab_latency_ms: dict[str, float]
    probe_latency_ms: dict[str, float]
    connections_opened: int
    requests_per_connection: float
    connection_reuse_ratio: float
    etag_validated_requests: int
    if_match_412_verified: bool | None
    not_modified_latency_ms: float | None
    coop_coep_valid: bool
    etag_observed: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize multi-lane boot-burst metrics to a JSON-safe dict with zero host paths."""
        return {
            "profile": self.profile,
            "lanes": self.lanes,
            "keep_alive": self.keep_alive,
            "verify_etag": self.verify_etag,
            "slab_bytes": self.slab_bytes,
            "probe_bytes": self.probe_bytes,
            "total_requests": self.total_requests,
            "slab_requests": self.slab_requests,
            "probe_requests": self.probe_requests,
            "successful_requests": self.successful_requests,
            "error_count": self.error_count,
            "total_bytes_transferred": self.total_bytes_transferred,
            "elapsed_sec": round(self.elapsed_sec, 6),
            "mb_per_sec": round(self.mb_per_sec, 4),
            "requests_per_sec": round(self.requests_per_sec, 4),
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "latency_ms": {k: round(v, 4) for k, v in self.latency_ms.items()},
            "slab_latency_ms": {k: round(v, 4) for k, v in self.slab_latency_ms.items()},
            "probe_latency_ms": {k: round(v, 4) for k, v in self.probe_latency_ms.items()},
            "connections_opened": self.connections_opened,
            "requests_per_connection": round(self.requests_per_connection, 4),
            "connection_reuse_ratio": round(self.connection_reuse_ratio, 4),
            "etag_validated_requests": self.etag_validated_requests,
            "if_match_412_verified": self.if_match_412_verified,
            "not_modified_latency_ms": (
                round(self.not_modified_latency_ms, 4)
                if self.not_modified_latency_ms is not None
                else None
            ),
            "coop_coep_valid": self.coop_coep_valid,
            "etag_observed": self.etag_observed,
        }


def _is_strong_etag(etag: str | None) -> bool:
    """Return True if `etag` is a syntactically valid RFC 9110 strong entity-tag."""
    if not etag:
        return False
    cleaned = etag.strip()
    if cleaned.startswith(("W/", "w/")):
        return False
    return len(cleaned) >= 2 and cleaned.startswith('"') and cleaned.endswith('"')


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Compute a linear-interpolated percentile on a non-empty sorted sequence."""
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_values[lo])
    weight = rank - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * weight)


def compute_percentiles(samples_ms: Sequence[float]) -> dict[str, float]:
    """Return min, avg, p50, p95, p99, and max latency summary in milliseconds."""
    if not samples_ms:
        return {
            "min": 0.0,
            "avg": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    ordered = sorted(float(x) for x in samples_ms)
    return {
        "min": ordered[0],
        "avg": float(statistics.fmean(ordered)),
        "p50": _percentile(ordered, 50.0),
        "p95": _percentile(ordered, 95.0),
        "p99": _percentile(ordered, 99.0),
        "max": ordered[-1],
    }


def validate_isolation_headers(headers: Mapping[str, str]) -> bool:
    """Verify COOP, COEP, CORP, and Accept-Ranges headers (case-insensitive names)."""
    normalized = {k.lower(): v.strip().lower() for k, v in headers.items()}
    for name, expected in REQUIRED_ISOLATION_HEADERS.items():
        if normalized.get(name) != expected:
            return False
    return True


def plan_range_offsets(
    file_size: int,
    chunk_bytes: int,
    num_requests: int,
    seed: int = DEFAULT_SEED,
) -> list[tuple[int, int]]:
    """Generate deterministic mixed strided and random inclusive `(start, end)` ranges."""
    if file_size <= 0:
        raise ValueError("file_size must be positive")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if num_requests <= 0:
        raise ValueError("num_requests must be positive")

    effective_chunk = min(chunk_bytes, file_size)
    max_start = file_size - effective_chunk
    rng = random.Random(seed)
    ranges: list[tuple[int, int]] = []

    stride_index = 0
    for idx in range(num_requests):
        if max_start == 0:
            start = 0
        elif idx % 2 == 0:
            start = (stride_index * effective_chunk) % (max_start + 1)
            stride_index += 1
        else:
            start = rng.randint(0, max_start)
        end = start + effective_chunk - 1
        ranges.append((start, end))

    return ranges


def plan_boot_burst_workload(
    file_size: int,
    num_requests: int,
    *,
    lanes: int = DEFAULT_BOOT_LANES,
    slab_bytes: int = DEFAULT_BOOT_SLAB_BYTES,
    probe_bytes: int = DEFAULT_BOOT_PROBE_BYTES,
    seed: int = DEFAULT_SEED,
) -> list[list[BootRangeRequest]]:
    """Generate deterministic per-lane Range plans matching GTA VI's 8-lane boot burst.

    Even lanes model DirectStorage bulk streaming (75% `slab_bytes` + 25% `probe_bytes`),
    while odd lanes model Bink2 / `shaders.rpf` / ReadFile probes (75% `probe_bytes` +
    25% `slab_bytes`). Offsets are deterministically mixed across strided, hot working-set,
    and pseudo-random locations within `[0, file_size)`.
    """
    if file_size <= 0:
        raise ValueError("file_size must be positive")
    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    if slab_bytes <= 0:
        raise ValueError("slab_bytes must be positive")
    if probe_bytes <= 0:
        raise ValueError("probe_bytes must be positive")

    effective_slab = min(slab_bytes, file_size)
    effective_probe = min(probe_bytes, file_size)
    max_slab_start = file_size - effective_slab
    max_probe_start = file_size - effective_probe

    active_lanes = min(lanes, num_requests)
    base_count, remainder = divmod(num_requests, active_lanes)
    rng = random.Random(seed)

    # Pre-seed a small hot working set of 64 KiB-aligned probe offsets (shaders.rpf / Bink2).
    hot_probe_count = max(1, min(8, (file_size + effective_probe - 1) // effective_probe))
    hot_probe_offsets = [
        min(idx * effective_probe, max_probe_start) for idx in range(hot_probe_count)
    ]

    lane_plans: list[list[BootRangeRequest]] = []
    slab_stride_idx = 0
    probe_hot_idx = 0

    for lane_idx in range(active_lanes):
        lane_req_count = base_count + (1 if lane_idx < remainder else 0)
        plan: list[BootRangeRequest] = []
        is_bulk_lane = (lane_idx % 2) == 0

        for seq in range(lane_req_count):
            if is_bulk_lane:
                # Seq 0, 2, 3 -> slab; seq 1 -> probe (so even 1-lane >=2 reqs tests both).
                is_slab = (seq % 4) != 1
            else:
                # Seq 0, 1, 2 -> probe; seq 3 -> slab (75% probes, 25% slabs).
                is_slab = (seq % 4) == 3

            if is_slab:
                if max_slab_start == 0:
                    start = 0
                elif (slab_stride_idx % 2) == 0:
                    start = (slab_stride_idx * effective_slab) % (max_slab_start + 1)
                else:
                    start = rng.randint(0, max_slab_start)
                slab_stride_idx += 1
                end = start + effective_slab - 1
                plan.append(BootRangeRequest(lane=lane_idx, kind="slab", start=start, end=end))
            else:
                if max_probe_start == 0:
                    start = 0
                elif (probe_hot_idx % 2) == 0:
                    start = hot_probe_offsets[probe_hot_idx % len(hot_probe_offsets)]
                else:
                    start = rng.randint(0, max_probe_start)
                probe_hot_idx += 1
                end = start + effective_probe - 1
                plan.append(BootRangeRequest(lane=lane_idx, kind="probe", start=start, end=end))

        lane_plans.append(plan)

    return lane_plans


def load_serve_local_module(repo_root: Path = DEFAULT_REPO_ROOT) -> Any:
    """Dynamically load `server/serve_local.py` without mutating `sys.path`."""
    server_file = repo_root / "server" / "serve_local.py"
    if not server_file.is_file():
        raise FileNotFoundError("server/serve_local.py not found")

    spec = importlib.util.spec_from_file_location("_playgta6_bench_serve_local", server_file)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create module spec for server/serve_local.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_deterministic_payload(target: Path, file_bytes: int, seed: int) -> None:
    """Write a deterministic synthetic binary file of exact size `file_bytes`."""
    rng = random.Random(seed)
    block_size = 1024 * 1024
    remaining = file_bytes
    with target.open("wb") as fh:
        while remaining > 0:
            size = min(block_size, remaining)
            fh.write(rng.randbytes(size))
            remaining -= size


@dataclass(frozen=True)
class _ParsedTarget:
    scheme: str
    host: str
    port: int
    request_path: str


def _parse_target_url(url: str) -> _ParsedTarget:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r} (expected http or https)")
    if not parsed.hostname:
        raise ValueError("Target URL must include a hostname")
    port = parsed.port or (443 if scheme == "https" else 80)
    request_path = parsed.path or "/"
    if parsed.query:
        request_path = f"{request_path}?{parsed.query}"
    return _ParsedTarget(
        scheme=scheme,
        host=parsed.hostname,
        port=port,
        request_path=request_path,
    )


def _open_connection(target: _ParsedTarget, timeout: float) -> http.client.HTTPConnection:
    if target.scheme == "https":
        return http.client.HTTPSConnection(target.host, target.port, timeout=timeout)
    return http.client.HTTPConnection(target.host, target.port, timeout=timeout)


class _ConnectionTracker:
    """Thread-safe counter of actual TCP `connect()` calls made across worker lanes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.connections_opened = 0

    def record_connect(self) -> None:
        with self._lock:
            self.connections_opened += 1


class _TrackingHTTPConnection(http.client.HTTPConnection):
    """HTTP/1.1 connection that records every underlying TCP socket `connect()`."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: float,
        tracker: _ConnectionTracker,
    ) -> None:
        super().__init__(host, port, timeout=timeout)
        self._tracker = tracker

    def connect(self) -> None:
        super().connect()
        self._tracker.record_connect()


class _TrackingHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS/1.1 connection that records every underlying TCP/TLS socket `connect()`."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: float,
        tracker: _ConnectionTracker,
    ) -> None:
        super().__init__(host, port, timeout=timeout)
        self._tracker = tracker

    def connect(self) -> None:
        super().connect()
        self._tracker.record_connect()


def _open_tracking_connection(
    target: _ParsedTarget,
    timeout: float,
    tracker: _ConnectionTracker,
) -> http.client.HTTPConnection:
    if target.scheme == "https":
        return _TrackingHTTPSConnection(target.host, target.port, timeout=timeout, tracker=tracker)
    return _TrackingHTTPConnection(target.host, target.port, timeout=timeout, tracker=tracker)


class _ThreadLocalConnections:
    """Per-worker HTTP keep-alive connection manager with deterministic teardown."""

    def __init__(self, target: _ParsedTarget, timeout: float) -> None:
        self._target = target
        self._timeout = timeout
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all: list[http.client.HTTPConnection] = []

    def get(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = _open_connection(self._target, self._timeout)
            self._local.conn = conn
            with self._lock:
                self._all.append(conn)
        return conn

    def reset(self) -> http.client.HTTPConnection:
        old = getattr(self._local, "conn", None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
            self._local.conn = None
        return self.get()

    def close_all(self) -> None:
        with self._lock:
            conns = list(self._all)
            self._all.clear()
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass


def _probe_target(
    target: _ParsedTarget,
    timeout: float,
) -> tuple[int, str | None, bool]:
    """Probe target resource size, ETag, and isolation headers via HEAD (fallback Range 0-0)."""
    conn = _open_connection(target, timeout)
    try:
        conn.request("HEAD", target.request_path, headers={"Connection": "close"})
        resp = conn.getresponse()
        resp.read()
        headers = dict(resp.getheaders())
        headers_lower = {k.lower(): v for k, v in headers.items()}
        if resp.status == HTTPStatus.OK and "content-length" in headers_lower:
            file_size = int(headers_lower["content-length"])
            if file_size > 0:
                return (
                    file_size,
                    headers_lower.get("etag"),
                    validate_isolation_headers(headers),
                )
    finally:
        conn.close()

    # Fallback: single-byte Range probe for servers without HEAD Content-Length.
    conn = _open_connection(target, timeout)
    try:
        conn.request(
            "GET",
            target.request_path,
            headers={"Range": "bytes=0-0", "Connection": "close"},
        )
        resp = conn.getresponse()
        resp.read()
        headers = dict(resp.getheaders())
        headers_lower = {k.lower(): v for k, v in headers.items()}
        if resp.status == HTTPStatus.PARTIAL_CONTENT:
            content_range = headers_lower.get("content-range", "")
            match = CONTENT_RANGE_RE.match(content_range.strip())
            if match and match.group(3) != "*":
                file_size = int(match.group(3))
                if file_size > 0:
                    return (
                        file_size,
                        headers_lower.get("etag"),
                        validate_isolation_headers(headers),
                    )
        raise RuntimeError(
            f"Unable to determine target size via HEAD or Range probe (status={resp.status})"
        )
    finally:
        conn.close()


def _probe_not_modified(
    target: _ParsedTarget,
    etag: str | None,
    timeout: float,
) -> tuple[float | None, bool]:
    """Send conditional `If-None-Match` probe and return `(latency_ms_or_none, headers_ok)`."""
    if not etag:
        return None, True

    conn = _open_connection(target, timeout)
    try:
        start_t = time.perf_counter()
        conn.request(
            "GET",
            target.request_path,
            headers={"If-None-Match": etag, "Connection": "close"},
        )
        resp = conn.getresponse()
        body = resp.read()
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        headers = dict(resp.getheaders())
        headers_ok = validate_isolation_headers(headers)
        if resp.status == HTTPStatus.NOT_MODIFIED and len(body) == 0:
            return elapsed_ms, headers_ok
        return None, headers_ok
    except (http.client.HTTPException, OSError):
        return None, False
    finally:
        conn.close()


def _probe_if_match_mismatch(
    target: _ParsedTarget,
    etag: str | None,
    timeout: float,
) -> tuple[bool | None, bool]:
    """Verify server rejects a stale `If-Match` Range request with `412 Precondition Failed`."""
    if not _is_strong_etag(etag):
        return None, True

    stale_etag = (
        '"playgta6-stale-etag-probe-alt"'
        if etag == '"playgta6-stale-etag-probe"'
        else '"playgta6-stale-etag-probe"'
    )
    conn = _open_connection(target, timeout)
    try:
        conn.request(
            "GET",
            target.request_path,
            headers={
                "Range": "bytes=0-0",
                "If-Match": stale_etag,
                "Connection": "close",
            },
        )
        resp = conn.getresponse()
        body = resp.read()
        headers = dict(resp.getheaders())
        headers_ok = validate_isolation_headers(headers)
        matched_412 = resp.status == HTTPStatus.PRECONDITION_FAILED and len(body) == 0
        return matched_412, headers_ok
    except (http.client.HTTPException, OSError):
        return False, False
    finally:
        conn.close()


@dataclass(frozen=True)
class _RangeOutcome:
    ok: bool
    bytes_read: int
    latency_ms: float
    headers_ok: bool
    etag: str | None


def _execute_range_request(
    pool: _ThreadLocalConnections,
    target: _ParsedTarget,
    start: int,
    end: int,
    expected_file_size: int,
) -> _RangeOutcome:
    expected_len = end - start + 1
    req_headers = {"Range": f"bytes={start}-{end}"}
    expected_prefix = f"bytes {start}-{end}/"

    for attempt in range(2):
        conn = pool.get() if attempt == 0 else pool.reset()
        t0 = time.perf_counter()
        try:
            conn.request("GET", target.request_path, headers=req_headers)
            resp = conn.getresponse()
            body = resp.read()
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except (http.client.HTTPException, OSError):
            if attempt == 0:
                continue
            return _RangeOutcome(
                ok=False,
                bytes_read=0,
                latency_ms=0.0,
                headers_ok=False,
                etag=None,
            )

        headers = dict(resp.getheaders())
        headers_lower = {k.lower(): v for k, v in headers.items()}
        headers_ok = validate_isolation_headers(headers)
        etag = headers_lower.get("etag")
        content_range = headers_lower.get("content-range", "").strip()
        range_matches = (
            content_range == f"bytes {start}-{end}/{expected_file_size}"
            or content_range.startswith(expected_prefix)
        )

        if resp.status == HTTPStatus.PARTIAL_CONTENT and len(body) == expected_len and range_matches:
            return _RangeOutcome(
                ok=True,
                bytes_read=len(body),
                latency_ms=latency_ms,
                headers_ok=headers_ok,
                etag=etag,
            )
        return _RangeOutcome(
            ok=False,
            bytes_read=0,
            latency_ms=latency_ms,
            headers_ok=headers_ok,
            etag=etag,
        )

    return _RangeOutcome(ok=False, bytes_read=0, latency_ms=0.0, headers_ok=False, etag=None)


def run_benchmark(
    url: str,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    num_requests: int = DEFAULT_REQUESTS,
    concurrency: int = DEFAULT_CONCURRENCY,
    seed: int = DEFAULT_SEED,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> BenchmarkResult:
    """Run concurrent HTTP 206 Range and 304 If-None-Match benchmark against `url`."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    target = _parse_target_url(url)
    file_size, etag_observed, probe_headers_ok = _probe_target(target, timeout)
    planned_ranges = plan_range_offsets(
        file_size=file_size,
        chunk_bytes=chunk_bytes,
        num_requests=num_requests,
        seed=seed,
    )

    pool = _ThreadLocalConnections(target, timeout)
    outcomes: list[_RangeOutcome] = []

    batch_start = time.perf_counter()
    try:
        worker_count = min(concurrency, num_requests)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _execute_range_request,
                    pool,
                    target,
                    start,
                    end,
                    file_size,
                )
                for start, end in planned_ranges
            ]
            for fut in futures:
                outcomes.append(fut.result())
    finally:
        pool.close_all()
    elapsed_sec = max(time.perf_counter() - batch_start, 1e-9)

    successful = 0
    failed = 0
    total_bytes = 0
    latencies_ms: list[float] = []
    headers_all_valid = probe_headers_ok

    for outcome in outcomes:
        if outcome.etag and not etag_observed:
            etag_observed = outcome.etag
        if not outcome.headers_ok:
            headers_all_valid = False
        if outcome.ok:
            successful += 1
            total_bytes += outcome.bytes_read
            latencies_ms.append(outcome.latency_ms)
        else:
            failed += 1

    not_modified_latency_ms, not_modified_headers_ok = _probe_not_modified(
        target,
        etag_observed,
        timeout,
    )
    if not not_modified_headers_ok:
        headers_all_valid = False

    throughput_mbps = (total_bytes / (1024.0 * 1024.0)) / elapsed_sec
    requests_per_sec = successful / elapsed_sec

    return BenchmarkResult(
        total_requests=num_requests,
        successful_requests=successful,
        failed_requests=failed,
        total_bytes_transferred=total_bytes,
        elapsed_sec=elapsed_sec,
        throughput_mbps=throughput_mbps,
        requests_per_sec=requests_per_sec,
        latency_ms=compute_percentiles(latencies_ms),
        not_modified_latency_ms=not_modified_latency_ms,
        coop_coep_valid=bool(headers_all_valid and successful > 0),
        etag_observed=etag_observed,
    )


def run_self_test(
    *,
    file_bytes: int = DEFAULT_FILE_BYTES,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    num_requests: int = DEFAULT_REQUESTS,
    concurrency: int = DEFAULT_CONCURRENCY,
    seed: int = DEFAULT_SEED,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    repo_root: Path = DEFAULT_REPO_ROOT,
) -> BenchmarkResult:
    """Spin up `server/serve_local.py` on an ephemeral localhost port and benchmark it."""
    if file_bytes <= 0:
        raise ValueError("file_bytes must be positive")

    serve_local = load_serve_local_module(repo_root)
    handler_cls = getattr(serve_local, "PlayGta6Handler")

    class _QuietHandler(handler_cls):  # type: ignore[misc,valid-type]
        def log_message(self, format: str, *args: Any) -> None:
            return None

    with tempfile.TemporaryDirectory(prefix="playgta6_bench_") as tmp_name:
        base = Path(tmp_name)
        web_root = base / "web"
        data_root = base / "data"
        web_root.mkdir()
        data_root.mkdir()

        (web_root / "index.html").write_text("<!doctype html><title>bench</title>", encoding="utf-8")
        payload_path = data_root / "bench-payload.bin"
        _write_deterministic_payload(payload_path, file_bytes, seed)

        handler_factory = partial(_QuietHandler, web_root=web_root, data_root=data_root)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        server_thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/data/bench-payload.bin"
            return run_benchmark(
                url,
                chunk_bytes=chunk_bytes,
                num_requests=num_requests,
                concurrency=concurrency,
                seed=seed,
                timeout=timeout,
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=3.0)


@dataclass(frozen=True)
class _BootRangeOutcome:
    kind: str
    ok: bool
    bytes_read: int
    latency_ms: float
    headers_ok: bool
    etag: str | None
    etag_validated: bool


def _close_connection_quietly(conn: http.client.HTTPConnection | None) -> None:
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass


def _execute_lane_worker(
    lane_requests: Sequence[BootRangeRequest],
    target: _ParsedTarget,
    file_size: int,
    expected_etag: str | None,
    *,
    keep_alive: bool,
    verify_etag: bool,
    timeout: float,
    tracker: _ConnectionTracker,
    start_barrier: threading.Barrier | None,
) -> list[_BootRangeOutcome]:
    """Execute one lane's sequential Range requests over a persistent (or per-request) socket."""
    if start_barrier is not None:
        try:
            start_barrier.wait()
        except threading.BrokenBarrierError:
            pass

    outcomes: list[_BootRangeOutcome] = []
    conn: http.client.HTTPConnection | None = None
    strong_expected_etag = _is_strong_etag(expected_etag)

    try:
        for req in lane_requests:
            expected_len = req.end - req.start + 1
            expected_prefix = f"bytes {req.start}-{req.end}/"
            req_headers: dict[str, str] = {
                "Range": f"bytes={req.start}-{req.end}",
                "Connection": "keep-alive" if keep_alive else "close",
            }
            if verify_etag and expected_etag:
                req_headers["If-Match"] = expected_etag

            outcome: _BootRangeOutcome | None = None
            for attempt in range(2):
                if conn is None:
                    conn = _open_tracking_connection(target, timeout, tracker)
                    reused_socket = False
                else:
                    reused_socket = conn.sock is not None

                t0 = time.perf_counter()
                try:
                    conn.request("GET", target.request_path, headers=req_headers)
                    resp = conn.getresponse()
                    body = resp.read()
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                except (http.client.HTTPException, OSError):
                    _close_connection_quietly(conn)
                    conn = None
                    if attempt == 0 and (reused_socket or keep_alive):
                        continue
                    outcome = _BootRangeOutcome(
                        kind=req.kind,
                        ok=False,
                        bytes_read=0,
                        latency_ms=0.0,
                        headers_ok=False,
                        etag=None,
                        etag_validated=False,
                    )
                    break

                headers = dict(resp.getheaders())
                headers_lower = {k.lower(): v for k, v in headers.items()}
                headers_ok = validate_isolation_headers(headers)
                resp_etag = headers_lower.get("etag")
                content_range = headers_lower.get("content-range", "").strip()
                range_matches = (
                    content_range == f"bytes {req.start}-{req.end}/{file_size}"
                    or content_range.startswith(expected_prefix)
                )

                if not keep_alive or resp.will_close:
                    _close_connection_quietly(conn)
                    conn = None

                range_ok = (
                    resp.status == HTTPStatus.PARTIAL_CONTENT
                    and len(body) == expected_len
                    and range_matches
                )
                if verify_etag:
                    etag_ok = (
                        strong_expected_etag
                        and _is_strong_etag(resp_etag)
                        and resp_etag == expected_etag
                    )
                    ok = range_ok and etag_ok
                    etag_validated = ok
                else:
                    ok = range_ok
                    etag_validated = False

                outcome = _BootRangeOutcome(
                    kind=req.kind,
                    ok=ok,
                    bytes_read=len(body) if ok else 0,
                    latency_ms=latency_ms,
                    headers_ok=headers_ok,
                    etag=resp_etag,
                    etag_validated=etag_validated,
                )
                break

            assert outcome is not None
            outcomes.append(outcome)
    finally:
        _close_connection_quietly(conn)

    return outcomes


def run_boot_profile_benchmark(
    url: str,
    *,
    lanes: int = DEFAULT_BOOT_LANES,
    num_requests: int = DEFAULT_REQUESTS,
    slab_bytes: int = DEFAULT_BOOT_SLAB_BYTES,
    probe_bytes: int = DEFAULT_BOOT_PROBE_BYTES,
    keep_alive: bool = True,
    verify_etag: bool = True,
    seed: int = DEFAULT_SEED,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    profile: str = BOOT_PROFILE_NAME,
) -> BootProfileResult:
    """Run multi-lane HTTP/1.1 Keep-Alive Range burst benchmark against `url`."""
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if slab_bytes <= 0:
        raise ValueError("slab_bytes must be positive")
    if probe_bytes <= 0:
        raise ValueError("probe_bytes must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    target = _parse_target_url(url)
    file_size, etag_observed, probe_headers_ok = _probe_target(target, timeout)
    lane_plans = plan_boot_burst_workload(
        file_size=file_size,
        num_requests=num_requests,
        lanes=lanes,
        slab_bytes=slab_bytes,
        probe_bytes=probe_bytes,
        seed=seed,
    )

    active_lanes = len(lane_plans)
    tracker = _ConnectionTracker()
    start_barrier = (
        threading.Barrier(active_lanes, timeout=min(timeout, 5.0)) if active_lanes > 1 else None
    )

    outcomes: list[_BootRangeOutcome] = []
    batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=active_lanes) as executor:
        futures = [
            executor.submit(
                _execute_lane_worker,
                lane_plan,
                target,
                file_size,
                etag_observed,
                keep_alive=keep_alive,
                verify_etag=verify_etag,
                timeout=timeout,
                tracker=tracker,
                start_barrier=start_barrier,
            )
            for lane_plan in lane_plans
        ]
        for fut in futures:
            outcomes.extend(fut.result())
    elapsed_sec = max(time.perf_counter() - batch_start, 1e-9)

    successful = 0
    error_count = 0
    slab_requests = 0
    probe_requests = 0
    total_bytes = 0
    etag_validated = 0
    all_latencies_ms: list[float] = []
    slab_latencies_ms: list[float] = []
    probe_latencies_ms: list[float] = []
    headers_all_valid = probe_headers_ok

    for outcome in outcomes:
        if outcome.kind == "slab":
            slab_requests += 1
        else:
            probe_requests += 1
        if outcome.etag and not etag_observed:
            etag_observed = outcome.etag
        if not outcome.headers_ok:
            headers_all_valid = False
        if outcome.ok:
            successful += 1
            total_bytes += outcome.bytes_read
            all_latencies_ms.append(outcome.latency_ms)
            if outcome.kind == "slab":
                slab_latencies_ms.append(outcome.latency_ms)
            else:
                probe_latencies_ms.append(outcome.latency_ms)
            if outcome.etag_validated:
                etag_validated += 1
        else:
            error_count += 1

    not_modified_latency_ms, not_modified_headers_ok = _probe_not_modified(
        target,
        etag_observed,
        timeout,
    )
    if not not_modified_headers_ok:
        headers_all_valid = False

    if verify_etag:
        if_match_412_verified, mismatch_headers_ok = _probe_if_match_mismatch(
            target,
            etag_observed,
            timeout,
        )
        if not mismatch_headers_ok:
            headers_all_valid = False
    else:
        if_match_412_verified = None

    mb_per_sec = (total_bytes / (1024.0 * 1024.0)) / elapsed_sec
    requests_per_sec = successful / elapsed_sec
    latency_summary = compute_percentiles(all_latencies_ms)
    slab_latency_summary = compute_percentiles(slab_latencies_ms)
    probe_latency_summary = compute_percentiles(probe_latencies_ms)

    connections_opened = tracker.connections_opened
    requests_per_connection = (
        (successful / connections_opened) if connections_opened > 0 else 0.0
    )
    connection_reuse_ratio = (
        max(0.0, (successful - connections_opened) / successful) if successful > 0 else 0.0
    )

    return BootProfileResult(
        profile=profile,
        lanes=active_lanes,
        keep_alive=keep_alive,
        verify_etag=verify_etag,
        slab_bytes=min(slab_bytes, file_size),
        probe_bytes=min(probe_bytes, file_size),
        total_requests=num_requests,
        slab_requests=slab_requests,
        probe_requests=probe_requests,
        successful_requests=successful,
        error_count=error_count,
        total_bytes_transferred=total_bytes,
        elapsed_sec=elapsed_sec,
        mb_per_sec=mb_per_sec,
        requests_per_sec=requests_per_sec,
        p50_ms=latency_summary["p50"],
        p95_ms=latency_summary["p95"],
        p99_ms=latency_summary["p99"],
        latency_ms=latency_summary,
        slab_latency_ms=slab_latency_summary,
        probe_latency_ms=probe_latency_summary,
        connections_opened=connections_opened,
        requests_per_connection=requests_per_connection,
        connection_reuse_ratio=connection_reuse_ratio,
        etag_validated_requests=etag_validated,
        if_match_412_verified=if_match_412_verified,
        not_modified_latency_ms=not_modified_latency_ms,
        coop_coep_valid=bool(headers_all_valid and successful > 0),
        etag_observed=etag_observed,
    )


def run_boot_profile_self_test(
    *,
    file_bytes: int = DEFAULT_BOOT_FILE_BYTES,
    lanes: int = DEFAULT_BOOT_LANES,
    num_requests: int = DEFAULT_REQUESTS,
    slab_bytes: int = DEFAULT_BOOT_SLAB_BYTES,
    probe_bytes: int = DEFAULT_BOOT_PROBE_BYTES,
    keep_alive: bool = True,
    verify_etag: bool = True,
    seed: int = DEFAULT_SEED,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    profile: str = BOOT_PROFILE_NAME,
    repo_root: Path = DEFAULT_REPO_ROOT,
) -> BootProfileResult:
    """Spin up `server/serve_local.py` with a synthetic `.rpf` file and run the boot-burst benchmark."""
    if file_bytes <= 0:
        raise ValueError("file_bytes must be positive")

    serve_local = load_serve_local_module(repo_root)
    handler_cls = getattr(serve_local, "PlayGta6Handler")
    server_cls = getattr(serve_local, "QuietThreadingHTTPServer", ThreadingHTTPServer)

    class _QuietHandler(handler_cls):  # type: ignore[misc,valid-type]
        def log_message(self, format: str, *args: Any) -> None:
            return None

    class _BenchHTTPServer(server_cls):  # type: ignore[misc,valid-type]
        request_queue_size = max(64, lanes * 4)

    with tempfile.TemporaryDirectory(prefix="playgta6_bench_boot_") as tmp_name:
        base = Path(tmp_name)
        web_root = base / "web"
        data_root = base / "data"
        web_root.mkdir()
        data_root.mkdir()

        (web_root / "index.html").write_text("<!doctype html><title>bench</title>", encoding="utf-8")
        # Use a `.rpf` suffix so `serve_local.py` exercises its ReadOnlyFileHandleCache.
        payload_path = data_root / "boot-burst.rpf"
        _write_deterministic_payload(payload_path, file_bytes, seed)

        handler_factory = partial(_QuietHandler, web_root=web_root, data_root=data_root)
        httpd = _BenchHTTPServer(("127.0.0.1", 0), handler_factory)
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        server_thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/data/boot-burst.rpf"
            return run_boot_profile_benchmark(
                url,
                lanes=lanes,
                num_requests=num_requests,
                slab_bytes=slab_bytes,
                probe_bytes=probe_bytes,
                keep_alive=keep_alive,
                verify_etag=verify_etag,
                seed=seed,
                timeout=timeout,
                profile=profile,
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=3.0)


def format_text_report(result: BenchmarkResult, *, mode: str) -> str:
    """Format a concise human-readable benchmark report (zero host paths)."""
    lat = result.latency_ms
    nm_str = (
        f"{result.not_modified_latency_ms:.3f} ms (304 OK)"
        if result.not_modified_latency_ms is not None
        else "unavailable / non-304"
    )
    lines = [
        f"playgta6web HTTP 206 Range Benchmark ({mode})",
        f"  Requests:         {result.successful_requests}/{result.total_requests} ok ({result.failed_requests} failed)",
        f"  Transferred:      {result.total_bytes_transferred:,} bytes in {result.elapsed_sec:.4f} s",
        f"  Throughput:       {result.throughput_mbps:.2f} MiB/s ({result.requests_per_sec:.1f} req/s)",
        (
            "  Latency (ms):     "
            f"min={lat['min']:.3f} avg={lat['avg']:.3f} "
            f"p50={lat['p50']:.3f} p95={lat['p95']:.3f} "
            f"p99={lat['p99']:.3f} max={lat['max']:.3f}"
        ),
        f"  If-None-Match:    {nm_str}",
        f"  COOP/COEP/CORP:   {'PASS' if result.coop_coep_valid else 'FAIL'}",
        f"  ETag Observed:    {result.etag_observed or 'none'}",
    ]
    return "\n".join(lines)


def format_boot_profile_report(result: BootProfileResult, *, mode: str) -> str:
    """Format a concise human-readable multi-lane boot-burst benchmark report (zero host paths)."""
    lat = result.latency_ms
    slab_lat = result.slab_latency_ms
    probe_lat = result.probe_latency_ms
    nm_str = (
        f"{result.not_modified_latency_ms:.3f} ms (304 OK)"
        if result.not_modified_latency_ms is not None
        else "unavailable / non-304"
    )
    if result.if_match_412_verified is True:
        if_match_412_str = "PASS"
    elif result.if_match_412_verified is False:
        if_match_412_str = "FAIL"
    else:
        if_match_412_str = "SKIP"

    lines = [
        f"playgta6web HTTP/1.1 Keep-Alive Range Benchmark ({result.profile}, {mode})",
        (
            f"  Config:           lanes={result.lanes} "
            f"keep_alive={'ON' if result.keep_alive else 'OFF'} "
            f"verify_etag={'ON' if result.verify_etag else 'OFF'} "
            f"slab={result.slab_bytes:,}B probe={result.probe_bytes:,}B"
        ),
        (
            f"  Requests:         {result.successful_requests}/{result.total_requests} ok "
            f"(slabs={result.slab_requests}, probes={result.probe_requests}, "
            f"errors={result.error_count})"
        ),
        f"  Transferred:      {result.total_bytes_transferred:,} bytes in {result.elapsed_sec:.4f} s",
        f"  Throughput:       {result.mb_per_sec:.2f} MiB/s ({result.requests_per_sec:.1f} req/s)",
        (
            "  Latency (ms):     "
            f"min={lat['min']:.3f} avg={lat['avg']:.3f} "
            f"p50={result.p50_ms:.3f} p95={result.p95_ms:.3f} "
            f"p99={result.p99_ms:.3f} max={lat['max']:.3f}"
        ),
        (
            "  Slab / Probe ms:  "
            f"slab(p50={slab_lat['p50']:.3f}, p95={slab_lat['p95']:.3f}, p99={slab_lat['p99']:.3f}) | "
            f"probe(p50={probe_lat['p50']:.3f}, p95={probe_lat['p95']:.3f}, p99={probe_lat['p99']:.3f})"
        ),
        (
            "  Keep-Alive Reuse: "
            f"connections_opened={result.connections_opened} "
            f"requests_per_connection={result.requests_per_connection:.2f} "
            f"reuse_ratio={result.connection_reuse_ratio * 100.0:.1f}%"
        ),
        (
            "  ETag / If-Match:  "
            f"etag_validated={result.etag_validated_requests}/{result.total_requests} "
            f"412={if_match_412_str} etag={result.etag_observed or 'none'}"
        ),
        f"  If-None-Match:    {nm_str}",
        f"  COOP/COEP/CORP:   {'PASS' if result.coop_coep_valid else 'FAIL'}",
    ]
    return "\n".join(lines)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local HTTP 206 Range & If-None-Match streaming benchmark for playgta6web."
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--url",
        type=str,
        default=None,
        help="Target HTTP/HTTPS URL to benchmark (e.g. http://127.0.0.1:8080/data/file.bin).",
    )
    target_group.add_argument(
        "--self-test",
        action="store_true",
        help="Spawn in-process server/serve_local.py with a synthetic payload (default if --url is omitted).",
    )
    parser.add_argument(
        "--file-bytes",
        type=_positive_int,
        default=None,
        help=(
            f"Synthetic payload size in bytes for --self-test "
            f"(default: {DEFAULT_FILE_BYTES}, or {DEFAULT_BOOT_FILE_BYTES} with --8lane-boot-profile)."
        ),
    )
    parser.add_argument(
        "--chunk-bytes",
        type=_positive_int,
        default=DEFAULT_CHUNK_BYTES,
        help=f"Byte size per HTTP 206 Range read (default: {DEFAULT_CHUNK_BYTES}).",
    )
    parser.add_argument(
        "--requests",
        type=_positive_int,
        default=DEFAULT_REQUESTS,
        help=f"Total number of HTTP 206 Range requests (default: {DEFAULT_REQUESTS}).",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent worker threads (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--8lane-boot-profile",
        action="store_true",
        dest="eight_lane_boot_profile",
        help=(
            "Run simulated 8-lane HTTP/1.1 Keep-Alive Range boot-burst benchmark "
            "(2 MiB slabs + 64 KiB probes + If-Match/ETag validation)."
        ),
    )
    parser.add_argument(
        "--lanes",
        type=_positive_int,
        default=None,
        help=(
            f"Number of parallel HTTP/1.1 Keep-Alive worker lanes "
            f"(default: {DEFAULT_BOOT_LANES} with --8lane-boot-profile)."
        ),
    )
    parser.add_argument(
        "--slab-bytes",
        type=_positive_int,
        default=None,
        help=(
            f"Large slab Range read size in bytes for multi-lane boot-burst mode "
            f"(default: {DEFAULT_BOOT_SLAB_BYTES})."
        ),
    )
    parser.add_argument(
        "--probe-bytes",
        type=_positive_int,
        default=None,
        help=(
            f"Small probe Range read size in bytes for multi-lane boot-burst mode "
            f"(default: {DEFAULT_BOOT_PROBE_BYTES})."
        ),
    )
    etag_group = parser.add_mutually_exclusive_group()
    etag_group.add_argument(
        "--verify-etag",
        action="store_true",
        dest="verify_etag",
        default=None,
        help="Send If-Match and strictly validate strong ETag on every Range response.",
    )
    etag_group.add_argument(
        "--no-verify-etag",
        action="store_false",
        dest="verify_etag",
        help="Disable If-Match / ETag validation in multi-lane boot-burst mode.",
    )
    keepalive_group = parser.add_mutually_exclusive_group()
    keepalive_group.add_argument(
        "--keep-alive",
        action="store_true",
        dest="keep_alive",
        default=None,
        help="Reuse persistent HTTP/1.1 Keep-Alive sockets per lane (default in multi-lane mode).",
    )
    keepalive_group.add_argument(
        "--no-keep-alive",
        action="store_false",
        dest="keep_alive",
        help="Disable HTTP/1.1 Keep-Alive and send Connection: close on every Range request.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic PRNG seed for offsets and synthetic payload (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Per-request HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SEC}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON output with zero host paths.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    use_self_test = args.self_test or args.url is None
    use_boot_profile = bool(
        args.eight_lane_boot_profile
        or args.lanes is not None
        or args.slab_bytes is not None
        or args.probe_bytes is not None
        or args.verify_etag is not None
        or args.keep_alive is not None
    )

    if use_boot_profile:
        profile = BOOT_PROFILE_NAME if args.eight_lane_boot_profile else MULTI_LANE_PROFILE_NAME
        if args.lanes is not None:
            lanes = args.lanes
        elif args.eight_lane_boot_profile:
            lanes = DEFAULT_BOOT_LANES
        else:
            lanes = args.concurrency
        slab_bytes = args.slab_bytes if args.slab_bytes is not None else DEFAULT_BOOT_SLAB_BYTES
        probe_bytes = args.probe_bytes if args.probe_bytes is not None else DEFAULT_BOOT_PROBE_BYTES
        keep_alive = True if args.keep_alive is None else args.keep_alive
        verify_etag = (
            args.eight_lane_boot_profile if args.verify_etag is None else args.verify_etag
        )
        file_bytes = args.file_bytes if args.file_bytes is not None else DEFAULT_BOOT_FILE_BYTES

        if use_self_test:
            mode = "self-test"
            boot_result = run_boot_profile_self_test(
                file_bytes=file_bytes,
                lanes=lanes,
                num_requests=args.requests,
                slab_bytes=slab_bytes,
                probe_bytes=probe_bytes,
                keep_alive=keep_alive,
                verify_etag=verify_etag,
                seed=args.seed,
                timeout=args.timeout,
                profile=profile,
            )
        else:
            mode = "url"
            boot_result = run_boot_profile_benchmark(
                args.url,
                lanes=lanes,
                num_requests=args.requests,
                slab_bytes=slab_bytes,
                probe_bytes=probe_bytes,
                keep_alive=keep_alive,
                verify_etag=verify_etag,
                seed=args.seed,
                timeout=args.timeout,
                profile=profile,
            )

        if args.json_output:
            print(json.dumps(boot_result.to_dict(), indent=2))
        else:
            print(format_boot_profile_report(boot_result, mode=mode))

        ok = (
            boot_result.error_count == 0
            and boot_result.successful_requests == boot_result.total_requests
            and boot_result.coop_coep_valid
            and (
                not boot_result.verify_etag
                or boot_result.etag_validated_requests == boot_result.total_requests
            )
            and (
                boot_result.etag_observed is None
                or boot_result.not_modified_latency_ms is not None
            )
        )
        return 0 if ok else 1

    file_bytes = args.file_bytes if args.file_bytes is not None else DEFAULT_FILE_BYTES
    if use_self_test:
        mode = "self-test"
        result = run_self_test(
            file_bytes=file_bytes,
            chunk_bytes=args.chunk_bytes,
            num_requests=args.requests,
            concurrency=args.concurrency,
            seed=args.seed,
            timeout=args.timeout,
        )
    else:
        mode = "url"
        result = run_benchmark(
            args.url,
            chunk_bytes=args.chunk_bytes,
            num_requests=args.requests,
            concurrency=args.concurrency,
            seed=args.seed,
            timeout=args.timeout,
        )

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_text_report(result, mode=mode))

    ok = (
        result.failed_requests == 0
        and result.successful_requests == result.total_requests
        and result.coop_coep_valid
        and (result.etag_observed is None or result.not_modified_latency_ms is not None)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
