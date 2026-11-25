import contextlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import bench_range_server  # noqa: E402


class TestBenchRangeServerMath(unittest.TestCase):
    def test_compute_percentiles_empty_and_single(self) -> None:
        empty = bench_range_server.compute_percentiles([])
        self.assertEqual(
            empty,
            {"min": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0},
        )

        single = bench_range_server.compute_percentiles([4.25])
        for key in ("min", "avg", "p50", "p95", "p99", "max"):
            self.assertAlmostEqual(single[key], 4.25, places=6)

    def test_compute_percentiles_interpolation_and_ordering(self) -> None:
        samples = [50.0, 10.0, 30.0, 20.0, 40.0]
        summary = bench_range_server.compute_percentiles(samples)
        self.assertAlmostEqual(summary["min"], 10.0, places=6)
        self.assertAlmostEqual(summary["avg"], 30.0, places=6)
        self.assertAlmostEqual(summary["p50"], 30.0, places=6)
        self.assertAlmostEqual(summary["p95"], 48.0, places=6)
        self.assertAlmostEqual(summary["p99"], 49.6, places=6)
        self.assertAlmostEqual(summary["max"], 50.0, places=6)

        hundred = [float(i) for i in range(1, 101)]
        summary_100 = bench_range_server.compute_percentiles(hundred)
        self.assertAlmostEqual(summary_100["min"], 1.0, places=6)
        self.assertAlmostEqual(summary_100["avg"], 50.5, places=6)
        self.assertAlmostEqual(summary_100["p50"], 50.5, places=6)
        self.assertAlmostEqual(summary_100["p95"], 95.05, places=6)
        self.assertAlmostEqual(summary_100["p99"], 99.01, places=6)
        self.assertAlmostEqual(summary_100["max"], 100.0, places=6)

    def test_plan_range_offsets_bounds_clamping_and_determinism(self) -> None:
        ranges_a = bench_range_server.plan_range_offsets(
            file_size=65536,
            chunk_bytes=8192,
            num_requests=16,
            seed=99,
        )
        ranges_b = bench_range_server.plan_range_offsets(
            file_size=65536,
            chunk_bytes=8192,
            num_requests=16,
            seed=99,
        )
        self.assertEqual(ranges_a, ranges_b)
        self.assertEqual(len(ranges_a), 16)
        for start, end in ranges_a:
            self.assertGreaterEqual(start, 0)
            self.assertLess(end, 65536)
            self.assertEqual(end - start + 1, 8192)

        # Chunk larger than file clamps cleanly to full file length.
        clamped = bench_range_server.plan_range_offsets(
            file_size=1024,
            chunk_bytes=4096,
            num_requests=4,
            seed=7,
        )
        self.assertEqual(clamped, [(0, 1023)] * 4)

    def test_validate_isolation_headers(self) -> None:
        good = {
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Resource-Policy": "same-origin",
            "Accept-Ranges": "bytes",
        }
        self.assertTrue(bench_range_server.validate_isolation_headers(good))

        # Case-insensitive header names should also pass.
        good_lower = {k.lower(): v for k, v in good.items()}
        self.assertTrue(bench_range_server.validate_isolation_headers(good_lower))

        for missing_key in list(good.keys()):
            broken = dict(good)
            del broken[missing_key]
            with self.subTest(missing=missing_key):
                self.assertFalse(bench_range_server.validate_isolation_headers(broken))

        wrong_coep = dict(good)
        wrong_coep["Cross-Origin-Embedder-Policy"] = "unsafe-none"
        self.assertFalse(bench_range_server.validate_isolation_headers(wrong_coep))

    def test_plan_boot_burst_workload_determinism_mix_and_clamping(self) -> None:
        file_size = 8 * 1024 * 1024  # 8 MiB
        slab_bytes = 2 * 1024 * 1024  # 2 MiB
        probe_bytes = 64 * 1024  # 64 KiB

        plans_a = bench_range_server.plan_boot_burst_workload(
            file_size=file_size,
            num_requests=64,
            lanes=8,
            slab_bytes=slab_bytes,
            probe_bytes=probe_bytes,
            seed=2026,
        )
        plans_b = bench_range_server.plan_boot_burst_workload(
            file_size=file_size,
            num_requests=64,
            lanes=8,
            slab_bytes=slab_bytes,
            probe_bytes=probe_bytes,
            seed=2026,
        )
        self.assertEqual(plans_a, plans_b)
        self.assertEqual(len(plans_a), 8)

        slab_count = 0
        probe_count = 0
        for lane_idx, lane_plan in enumerate(plans_a):
            self.assertEqual(len(lane_plan), 8)
            for req in lane_plan:
                self.assertEqual(req.lane, lane_idx)
                self.assertGreaterEqual(req.start, 0)
                self.assertLess(req.end, file_size)
                length = req.end - req.start + 1
                if req.kind == "slab":
                    self.assertEqual(length, slab_bytes)
                    slab_count += 1
                elif req.kind == "probe":
                    self.assertEqual(length, probe_bytes)
                    probe_count += 1
                else:
                    self.fail(f"Unexpected request kind: {req.kind!r}")

        self.assertEqual(slab_count, 32)
        self.assertEqual(probe_count, 32)

        # Single-lane workload with >= 2 requests exercises both slabs and probes.
        single_lane = bench_range_server.plan_boot_burst_workload(
            file_size=4 * 1024 * 1024,
            num_requests=4,
            lanes=1,
            slab_bytes=slab_bytes,
            probe_bytes=probe_bytes,
            seed=42,
        )
        self.assertEqual(len(single_lane), 1)
        kinds = {req.kind for req in single_lane[0]}
        self.assertEqual(kinds, {"slab", "probe"})

        # Clamping when file is smaller than slab_bytes and num_requests < lanes.
        clamped = bench_range_server.plan_boot_burst_workload(
            file_size=128 * 1024,
            num_requests=3,
            lanes=8,
            slab_bytes=slab_bytes,
            probe_bytes=probe_bytes,
            seed=7,
        )
        self.assertEqual(len(clamped), 3)
        for lane_plan in clamped:
            for req in lane_plan:
                self.assertGreaterEqual(req.start, 0)
                self.assertLess(req.end, 128 * 1024)
                expected_size = 128 * 1024 if req.kind == "slab" else probe_bytes
                self.assertEqual(req.end - req.start + 1, expected_size)

        with self.assertRaises(ValueError):
            bench_range_server.plan_boot_burst_workload(file_size=0, num_requests=8, lanes=8)
        with self.assertRaises(ValueError):
            bench_range_server.plan_boot_burst_workload(file_size=1024, num_requests=0, lanes=8)
        with self.assertRaises(ValueError):
            bench_range_server.plan_boot_burst_workload(file_size=1024, num_requests=8, lanes=0)


class TestBenchRangeServerIntegration(unittest.TestCase):
    def test_self_test_small_payload_end_to_end(self) -> None:
        result = bench_range_server.run_self_test(
            file_bytes=256 * 1024,
            chunk_bytes=16 * 1024,
            num_requests=16,
            concurrency=2,
            seed=123,
        )

        self.assertEqual(result.total_requests, 16)
        self.assertEqual(result.successful_requests, 16)
        self.assertEqual(result.failed_requests, 0)
        self.assertEqual(result.total_bytes_transferred, 16 * 16 * 1024)
        self.assertGreater(result.elapsed_sec, 0.0)
        self.assertGreater(result.throughput_mbps, 0.0)
        self.assertGreater(result.requests_per_sec, 0.0)

        expected_mbps = (result.total_bytes_transferred / (1024.0 * 1024.0)) / result.elapsed_sec
        self.assertAlmostEqual(result.throughput_mbps, expected_mbps, places=4)
        expected_rps = result.successful_requests / result.elapsed_sec
        self.assertAlmostEqual(result.requests_per_sec, expected_rps, places=4)

        lat = result.latency_ms
        self.assertEqual(set(lat.keys()), {"min", "avg", "p50", "p95", "p99", "max"})
        self.assertGreaterEqual(lat["min"], 0.0)
        self.assertLessEqual(lat["min"], lat["p50"])
        self.assertLessEqual(lat["p50"], lat["p95"])
        self.assertLessEqual(lat["p95"], lat["p99"])
        self.assertLessEqual(lat["p99"], lat["max"])
        self.assertLessEqual(lat["min"], lat["avg"])
        self.assertLessEqual(lat["avg"], lat["max"])

        self.assertIsNotNone(result.not_modified_latency_ms)
        assert result.not_modified_latency_ms is not None
        self.assertGreaterEqual(result.not_modified_latency_ms, 0.0)
        self.assertTrue(result.coop_coep_valid)
        self.assertIsNotNone(result.etag_observed)
        assert result.etag_observed is not None
        self.assertTrue(result.etag_observed.startswith('"') and result.etag_observed.endswith('"'))

    def test_detects_missing_coop_coep_and_non_304(self) -> None:
        payload = bytes(range(256)) * 64  # 16 KiB
        etag = '"mock-etag-1"'

        class _NoIsolationHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args) -> None:
                return None

            def do_HEAD(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("ETag", etag)
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

            def do_GET(self) -> None:
                range_hdr = self.headers.get("Range")
                if range_hdr and range_hdr.startswith("bytes="):
                    start_s, end_s = range_hdr[6:].split("-", 1)
                    start, end = int(start_s), int(end_s)
                    chunk = payload[start : end + 1]
                    self.send_response(HTTPStatus.PARTIAL_CONTENT)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.send_header("ETag", etag)
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(chunk)
                    return

                # Deliberately ignore If-None-Match and return 200 instead of 304.
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("ETag", etag)
                self.end_headers()
                self.wfile.write(payload)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _NoIsolationHandler)
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        thread.start()

        try:
            result = bench_range_server.run_benchmark(
                f"http://127.0.0.1:{port}/data/mock.bin",
                chunk_bytes=4096,
                num_requests=8,
                concurrency=2,
            )
            self.assertEqual(result.successful_requests, 8)
            self.assertEqual(result.failed_requests, 0)
            self.assertFalse(result.coop_coep_valid)
            self.assertIsNone(result.not_modified_latency_ms)
            self.assertEqual(result.etag_observed, etag)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=3.0)

    def test_cli_self_test_json_schema_and_zero_host_paths(self) -> None:
        script_path = REPO_ROOT / "tools" / "bench_range_server.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--self-test",
                "--file-bytes",
                str(128 * 1024),
                "--chunk-bytes",
                str(16 * 1024),
                "--requests",
                "12",
                "--concurrency",
                "2",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertEqual(proc.stderr.strip(), "")

        raw_json = proc.stdout.strip()
        data = json.loads(raw_json)
        expected_keys = {
            "totalRequests",
            "successfulRequests",
            "failedRequests",
            "totalBytesTransferred",
            "elapsedSec",
            "throughputMBps",
            "requestsPerSec",
            "latencyMs",
            "notModifiedLatencyMs",
            "coopCoepValid",
            "etagObserved",
        }
        self.assertEqual(set(data.keys()), expected_keys)
        self.assertEqual(data["totalRequests"], 12)
        self.assertEqual(data["successfulRequests"], 12)
        self.assertEqual(data["failedRequests"], 0)
        self.assertEqual(data["totalBytesTransferred"], 12 * 16 * 1024)
        self.assertTrue(data["coopCoepValid"])
        self.assertIsInstance(data["notModifiedLatencyMs"], (int, float))
        self.assertIsInstance(data["etagObserved"], str)
        self.assertEqual(
            set(data["latencyMs"].keys()),
            {"min", "avg", "p50", "p95", "p99", "max"},
        )

        # Ensure zero host filesystem paths, temp dirs, or usernames leak into JSON output.
        sensitive_tokens = {
            str(REPO_ROOT),
            str(REPO_ROOT).replace("\\", "/"),
            tempfile.gettempdir(),
            tempfile.gettempdir().replace("\\", "/"),
            str(Path.home()),
            str(Path.home()).replace("\\", "/"),
            os.environ.get("USERNAME", ""),
            os.environ.get("USER", ""),
        }
        lowered = raw_json.lower()
        for token in sensitive_tokens:
            if token and len(token) > 2:
                self.assertNotIn(token.lower(), lowered)

    def test_cli_text_report_and_default_self_test(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = bench_range_server.main(
                [
                    "--file-bytes",
                    str(64 * 1024),
                    "--chunk-bytes",
                    str(8 * 1024),
                    "--requests",
                    "8",
                    "--concurrency",
                    "2",
                ]
            )
        self.assertEqual(rc, 0)
        output = buf.getvalue()
        self.assertIn("self-test", output)
        self.assertIn("8/8 ok", output)
        self.assertIn("Throughput:", output)
        self.assertIn("p50=", output)
        self.assertIn("p95=", output)
        self.assertIn("p99=", output)
        self.assertIn("304 OK", output)
        self.assertIn("PASS", output)


class TestBootProfileBenchmarkIntegration(unittest.TestCase):
    def test_8lane_boot_profile_self_test_keep_alive_and_etag(self) -> None:
        slab_bytes = 2 * 1024 * 1024  # 2 MiB
        probe_bytes = 64 * 1024  # 64 KiB
        result = bench_range_server.run_boot_profile_self_test(
            file_bytes=4 * 1024 * 1024,  # 4 MiB synthetic .rpf
            lanes=8,
            num_requests=32,  # 4 requests/lane: 16 x 2 MiB slabs + 16 x 64 KiB probes
            slab_bytes=slab_bytes,
            probe_bytes=probe_bytes,
            keep_alive=True,
            verify_etag=True,
            seed=2026,
        )

        self.assertEqual(result.profile, "8lane-boot-profile")
        self.assertEqual(result.lanes, 8)
        self.assertTrue(result.keep_alive)
        self.assertTrue(result.verify_etag)
        self.assertEqual(result.total_requests, 32)
        self.assertEqual(result.slab_requests, 16)
        self.assertEqual(result.probe_requests, 16)
        self.assertEqual(result.successful_requests, 32)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(
            result.total_bytes_transferred,
            16 * slab_bytes + 16 * probe_bytes,
        )
        self.assertGreater(result.elapsed_sec, 0.0)
        self.assertGreater(result.mb_per_sec, 0.0)
        self.assertGreater(result.requests_per_sec, 0.0)

        expected_mbps = (result.total_bytes_transferred / (1024.0 * 1024.0)) / result.elapsed_sec
        self.assertAlmostEqual(result.mb_per_sec, expected_mbps, places=4)

        self.assertGreaterEqual(result.p50_ms, 0.0)
        self.assertLessEqual(result.p50_ms, result.p95_ms)
        self.assertLessEqual(result.p95_ms, result.p99_ms)
        self.assertAlmostEqual(result.p50_ms, result.latency_ms["p50"], places=6)
        self.assertAlmostEqual(result.p95_ms, result.latency_ms["p95"], places=6)
        self.assertAlmostEqual(result.p99_ms, result.latency_ms["p99"], places=6)
        self.assertGreaterEqual(result.slab_latency_ms["p50"], 0.0)
        self.assertGreaterEqual(result.probe_latency_ms["p50"], 0.0)

        # Persistent HTTP/1.1 Keep-Alive: 8 sockets reused across 32 requests (4.0 req/conn).
        self.assertEqual(result.connections_opened, 8)
        self.assertAlmostEqual(result.requests_per_connection, 4.0, places=4)
        self.assertAlmostEqual(result.connection_reuse_ratio, 0.75, places=4)

        # Strong ETag / If-Match validated on all 32 requests + 412 mismatch check.
        self.assertEqual(result.etag_validated_requests, 32)
        self.assertTrue(result.if_match_412_verified)
        self.assertIsNotNone(result.not_modified_latency_ms)
        self.assertTrue(result.coop_coep_valid)
        self.assertIsNotNone(result.etag_observed)

    def test_keep_alive_disabled_opens_connection_per_request(self) -> None:
        result = bench_range_server.run_boot_profile_self_test(
            file_bytes=512 * 1024,
            lanes=4,
            num_requests=16,
            slab_bytes=128 * 1024,
            probe_bytes=16 * 1024,
            keep_alive=False,
            verify_etag=True,
            seed=77,
        )

        self.assertEqual(result.successful_requests, 16)
        self.assertEqual(result.error_count, 0)
        self.assertFalse(result.keep_alive)
        self.assertEqual(result.connections_opened, 16)
        self.assertAlmostEqual(result.requests_per_connection, 1.0, places=4)
        self.assertAlmostEqual(result.connection_reuse_ratio, 0.0, places=4)
        self.assertEqual(result.etag_validated_requests, 16)
        self.assertTrue(result.if_match_412_verified)
        self.assertTrue(result.coop_coep_valid)

    def test_detects_etag_mismatch_and_server_forced_connection_close(self) -> None:
        payload = bytes(range(256)) * 256  # 64 KiB
        head_etag = '"mock-strong-etag-v1"'
        get_etag = '"mock-strong-etag-v2"'  # Deliberate drift vs HEAD ETag

        class _ForcedCloseAndDriftHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            drift_etag = False

            def log_message(self, format: str, *args) -> None:
                return None

            def _send_common_headers(self, etag_val: str) -> None:
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", etag_val)
                self.send_header("Connection", "close")

            def do_HEAD(self) -> None:
                self.close_connection = True
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", str(len(payload)))
                self._send_common_headers(head_etag)
                self.end_headers()

            def do_GET(self) -> None:
                self.close_connection = True
                current_etag = get_etag if self.drift_etag else head_etag
                if_none_match = self.headers.get("If-None-Match")
                range_hdr = self.headers.get("Range")
                if not range_hdr and if_none_match == head_etag:
                    self.send_response(HTTPStatus.NOT_MODIFIED)
                    self._send_common_headers(head_etag)
                    self.end_headers()
                    return

                if_match = self.headers.get("If-Match")
                if if_match and if_match != current_etag:
                    self.send_response(HTTPStatus.PRECONDITION_FAILED)
                    self.send_header("Content-Length", "0")
                    self._send_common_headers(current_etag)
                    self.end_headers()
                    return

                if range_hdr and range_hdr.startswith("bytes="):
                    start_s, end_s = range_hdr[6:].split("-", 1)
                    start, end = int(start_s), int(end_s)
                    chunk = payload[start : end + 1]
                    self.send_response(HTTPStatus.PARTIAL_CONTENT)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self._send_common_headers(current_etag)
                    self.end_headers()
                    self.wfile.write(chunk)
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", str(len(payload)))
                self._send_common_headers(current_etag)
                self.end_headers()
                self.wfile.write(payload)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _ForcedCloseAndDriftHandler)
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        thread.start()

        try:
            url = f"http://127.0.0.1:{port}/data/mock.rpf"

            # 1) Server forces Connection: close despite keep_alive=True -> reuse must report 1.0 req/conn.
            _ForcedCloseAndDriftHandler.drift_etag = False
            no_reuse = bench_range_server.run_boot_profile_benchmark(
                url,
                lanes=4,
                num_requests=12,
                slab_bytes=16 * 1024,
                probe_bytes=4 * 1024,
                keep_alive=True,
                verify_etag=True,
            )
            self.assertEqual(no_reuse.successful_requests, 12)
            self.assertEqual(no_reuse.error_count, 0)
            self.assertEqual(no_reuse.connections_opened, 12)
            self.assertAlmostEqual(no_reuse.requests_per_connection, 1.0, places=4)
            self.assertAlmostEqual(no_reuse.connection_reuse_ratio, 0.0, places=4)
            self.assertEqual(no_reuse.etag_validated_requests, 12)
            self.assertTrue(no_reuse.if_match_412_verified)

            # 2) Server ETag drifts -> If-Match fails with 412 and error_count catches every request.
            _ForcedCloseAndDriftHandler.drift_etag = True
            drifted = bench_range_server.run_boot_profile_benchmark(
                url,
                lanes=2,
                num_requests=6,
                slab_bytes=16 * 1024,
                probe_bytes=4 * 1024,
                keep_alive=True,
                verify_etag=True,
            )
            self.assertEqual(drifted.successful_requests, 0)
            self.assertEqual(drifted.error_count, 6)
            self.assertEqual(drifted.etag_validated_requests, 0)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=3.0)

    def test_cli_8lane_boot_profile_json_schema_and_zero_host_paths(self) -> None:
        script_path = REPO_ROOT / "tools" / "bench_range_server.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--8lane-boot-profile",
                "--file-bytes",
                str(4 * 1024 * 1024),
                "--lanes",
                "8",
                "--requests",
                "24",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertEqual(proc.stderr.strip(), "")

        raw_json = proc.stdout.strip()
        data = json.loads(raw_json)
        required_keys = {
            "profile",
            "lanes",
            "keep_alive",
            "verify_etag",
            "slab_bytes",
            "probe_bytes",
            "total_requests",
            "slab_requests",
            "probe_requests",
            "successful_requests",
            "error_count",
            "total_bytes_transferred",
            "elapsed_sec",
            "mb_per_sec",
            "requests_per_sec",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "latency_ms",
            "slab_latency_ms",
            "probe_latency_ms",
            "connections_opened",
            "requests_per_connection",
            "connection_reuse_ratio",
            "etag_validated_requests",
            "if_match_412_verified",
            "not_modified_latency_ms",
            "coop_coep_valid",
            "etag_observed",
        }
        self.assertEqual(set(data.keys()), required_keys)
        self.assertEqual(data["profile"], "8lane-boot-profile")
        self.assertEqual(data["lanes"], 8)
        self.assertTrue(data["keep_alive"])
        self.assertTrue(data["verify_etag"])
        self.assertEqual(data["slab_bytes"], 2 * 1024 * 1024)
        self.assertEqual(data["probe_bytes"], 64 * 1024)
        self.assertEqual(data["total_requests"], 24)
        self.assertEqual(data["successful_requests"], 24)
        self.assertEqual(data["error_count"], 0)
        self.assertGreater(data["mb_per_sec"], 0.0)
        self.assertLessEqual(data["p50_ms"], data["p95_ms"])
        self.assertLessEqual(data["p95_ms"], data["p99_ms"])
        self.assertEqual(data["connections_opened"], 8)
        self.assertAlmostEqual(data["requests_per_connection"], 3.0, places=4)
        self.assertEqual(data["etag_validated_requests"], 24)
        self.assertTrue(data["if_match_412_verified"])
        self.assertTrue(data["coop_coep_valid"])

        sensitive_tokens = {
            str(REPO_ROOT),
            str(REPO_ROOT).replace("\\", "/"),
            tempfile.gettempdir(),
            tempfile.gettempdir().replace("\\", "/"),
            str(Path.home()),
            str(Path.home()).replace("\\", "/"),
            os.environ.get("USERNAME", ""),
            os.environ.get("USER", ""),
        }
        lowered = raw_json.lower()
        for token in sensitive_tokens:
            if token and len(token) > 2:
                self.assertNotIn(token.lower(), lowered)

    def test_cli_8lane_text_report_and_composable_flags(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = bench_range_server.main(
                [
                    "--8lane-boot-profile",
                    "--file-bytes",
                    str(512 * 1024),
                    "--slab-bytes",
                    str(128 * 1024),
                    "--probe-bytes",
                    str(16 * 1024),
                    "--lanes",
                    "4",
                    "--requests",
                    "16",
                ]
            )
        self.assertEqual(rc, 0)
        output = buf.getvalue()
        self.assertIn("8lane-boot-profile", output)
        self.assertIn("16/16 ok", output)
        self.assertIn("errors=0", output)
        self.assertIn("MiB/s", output)
        self.assertIn("p50=", output)
        self.assertIn("p95=", output)
        self.assertIn("p99=", output)
        self.assertIn("connections_opened=4", output)
        self.assertIn("requests_per_connection=4.00", output)
        self.assertIn("etag_validated=16/16", output)
        self.assertIn("412=PASS", output)
        self.assertIn("304 OK", output)
        self.assertIn("PASS", output)

        # Composable flags (--lanes + --verify-etag + --no-keep-alive) without --8lane-boot-profile.
        json_buf = io.StringIO()
        with contextlib.redirect_stdout(json_buf):
            rc2 = bench_range_server.main(
                [
                    "--file-bytes",
                    str(256 * 1024),
                    "--slab-bytes",
                    str(64 * 1024),
                    "--probe-bytes",
                    str(16 * 1024),
                    "--lanes",
                    "4",
                    "--requests",
                    "8",
                    "--verify-etag",
                    "--no-keep-alive",
                    "--json",
                ]
            )
        self.assertEqual(rc2, 0)
        parsed = json.loads(json_buf.getvalue())
        self.assertEqual(parsed["profile"], "multi-lane-range")
        self.assertFalse(parsed["keep_alive"])
        self.assertTrue(parsed["verify_etag"])
        self.assertEqual(parsed["connections_opened"], 8)
        self.assertAlmostEqual(parsed["requests_per_connection"], 1.0, places=4)
        self.assertEqual(parsed["etag_validated_requests"], 8)
        self.assertEqual(parsed["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
