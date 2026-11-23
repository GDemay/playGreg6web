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


if __name__ == "__main__":
    unittest.main()
