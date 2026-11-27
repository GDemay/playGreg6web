import contextlib
import http.client
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

import serve_local  # noqa: E402


class TestServeLocal(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.web_root = base / "web"
        self.data_root = base / "data"
        self.web_root.mkdir()
        self.data_root.mkdir()

        (self.web_root / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")
        (self.web_root / "demo.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
        (self.data_root / "sample.bin").write_bytes(b"0123456789abcdef")

        self.httpd = serve_local.create_server("127.0.0.1", 0, self.web_root, self.data_root)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def _request(self, method: str, path: str, headers: dict[str, str] | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(method, path, headers=headers or {})
            resp = conn.getresponse()
            body = resp.read()
            return resp.status, dict(resp.getheaders()), body
        finally:
            conn.close()

    def _assert_isolation_headers(self, headers: dict[str, str]) -> None:
        self.assertEqual(headers.get("Cross-Origin-Opener-Policy"), "same-origin")
        self.assertEqual(headers.get("Cross-Origin-Embedder-Policy"), "require-corp")
        self.assertEqual(headers.get("Cross-Origin-Resource-Policy"), "same-origin")
        self.assertEqual(headers.get("Accept-Ranges"), "bytes")

    def test_coop_coep_and_wasm_mime(self) -> None:
        status, headers, body = self._request("GET", "/demo.wasm")
        self.assertEqual(status, 200)
        self._assert_isolation_headers(headers)
        self.assertEqual(headers.get("Content-Type"), "application/wasm")
        self.assertTrue(headers.get("ETag"))
        self.assertTrue(headers.get("Last-Modified"))
        self.assertEqual(body, b"\x00asm\x01\x00\x00\x00")

    def test_http_range_partial_content(self) -> None:
        status, headers, body = self._request("GET", "/data/sample.bin", {"Range": "bytes=4-9"})
        self.assertEqual(status, 206)
        self._assert_isolation_headers(headers)
        self.assertEqual(headers.get("Content-Range"), "bytes 4-9/16")
        self.assertEqual(headers.get("Content-Length"), "6")
        self.assertTrue(headers.get("ETag"))
        self.assertEqual(body, b"456789")

    def test_suffix_and_open_ended_ranges(self) -> None:
        status, headers, body = self._request("GET", "/data/sample.bin", {"Range": "bytes=-16"})
        self.assertEqual(status, 206)
        self._assert_isolation_headers(headers)
        self.assertEqual(headers.get("Content-Range"), "bytes 0-15/16")
        self.assertEqual(headers.get("Content-Length"), "16")
        self.assertEqual(body, b"0123456789abcdef")

        status, headers, body = self._request("GET", "/data/sample.bin", {"Range": "bytes=-4"})
        self.assertEqual(status, 206)
        self.assertEqual(headers.get("Content-Range"), "bytes 12-15/16")
        self.assertEqual(headers.get("Content-Length"), "4")
        self.assertEqual(body, b"cdef")

        status, headers, body = self._request("GET", "/data/sample.bin", {"Range": "bytes=10-"})
        self.assertEqual(status, 206)
        self._assert_isolation_headers(headers)
        self.assertEqual(headers.get("Content-Range"), "bytes 10-15/16")
        self.assertEqual(headers.get("Content-Length"), "6")
        self.assertEqual(body, b"abcdef")

    def test_invalid_range_returns_416(self) -> None:
        for bad_range in ("bytes=100-200", "bytes=10-5", "bytes=-0", "bytes=-", "bytes=abc-def"):
            with self.subTest(bad_range=bad_range):
                status, headers, body = self._request("GET", "/data/sample.bin", {"Range": bad_range})
                self.assertEqual(status, 416)
                self._assert_isolation_headers(headers)
                self.assertEqual(headers.get("Content-Range"), "bytes */16")
                self.assertEqual(body, b"")

    def test_head_request_headers_without_body(self) -> None:
        status, headers, body = self._request("HEAD", "/data/sample.bin")
        self.assertEqual(status, 200)
        self._assert_isolation_headers(headers)
        self.assertEqual(headers.get("Content-Length"), "16")
        self.assertTrue(headers.get("ETag"))
        self.assertTrue(headers.get("Last-Modified"))
        self.assertEqual(body, b"")

        # HEAD on /api/status should also work with empty body and isolation headers.
        st_status, st_headers, st_body = self._request("HEAD", "/api/status")
        self.assertEqual(st_status, 200)
        self._assert_isolation_headers(st_headers)
        self.assertEqual(st_headers.get("Content-Type"), "application/json")
        self.assertEqual(st_headers.get("Cache-Control"), "no-store")
        self.assertGreater(int(st_headers.get("Content-Length", "0")), 0)
        self.assertEqual(st_body, b"")

    def test_if_none_match_returns_304(self) -> None:
        status, headers, _ = self._request("GET", "/data/sample.bin")
        self.assertEqual(status, 200)
        etag = headers["ETag"]

        # Exact ETag match -> 304 Not Modified
        status304, headers304, body304 = self._request(
            "GET", "/data/sample.bin", {"If-None-Match": etag}
        )
        self.assertEqual(status304, 304)
        self._assert_isolation_headers(headers304)
        self.assertEqual(headers304.get("ETag"), etag)
        self.assertEqual(body304, b"")

        # Weak comparison and wildcard should also yield 304 per RFC 9110.
        status_weak, _, body_weak = self._request(
            "GET", "/data/sample.bin", {"If-None-Match": f"W/{etag}"}
        )
        self.assertEqual(status_weak, 304)
        self.assertEqual(body_weak, b"")

        status_wild, _, body_wild = self._request(
            "GET", "/data/sample.bin", {"If-None-Match": "*"}
        )
        self.assertEqual(status_wild, 304)
        self.assertEqual(body_wild, b"")

        # Mismatched ETag -> 200 OK with full body.
        status200, _, body200 = self._request(
            "GET", "/data/sample.bin", {"If-None-Match": '"stale-etag"'}
        )
        self.assertEqual(status200, 200)
        self.assertEqual(body200, b"0123456789abcdef")

    def test_if_range_matching_and_mismatched_etag(self) -> None:
        _, headers, _ = self._request("HEAD", "/data/sample.bin")
        etag = headers["ETag"]
        last_modified = headers["Last-Modified"]

        # Matching strong ETag -> 206 Partial Content
        status, resp_headers, body = self._request(
            "GET",
            "/data/sample.bin",
            {"Range": "bytes=2-5", "If-Range": etag},
        )
        self.assertEqual(status, 206)
        self._assert_isolation_headers(resp_headers)
        self.assertEqual(resp_headers.get("Content-Range"), "bytes 2-5/16")
        self.assertEqual(body, b"2345")

        # Matching Last-Modified HTTP-date -> 206 Partial Content
        status_lm, _, body_lm = self._request(
            "GET",
            "/data/sample.bin",
            {"Range": "bytes=2-5", "If-Range": last_modified},
        )
        self.assertEqual(status_lm, 206)
        self.assertEqual(body_lm, b"2345")

        # Mismatched ETag -> 200 OK full representation (RFC 9110)
        status_miss, resp_miss, body_miss = self._request(
            "GET",
            "/data/sample.bin",
            {"Range": "bytes=2-5", "If-Range": '"0-0"'},
        )
        self.assertEqual(status_miss, 200)
        self._assert_isolation_headers(resp_miss)
        self.assertIsNone(resp_miss.get("Content-Range"))
        self.assertEqual(resp_miss.get("Content-Length"), "16")
        self.assertEqual(body_miss, b"0123456789abcdef")

        # Weak ETag in If-Range must NOT match -> fall back to 200 OK full file
        status_weak, _, body_weak = self._request(
            "GET",
            "/data/sample.bin",
            {"Range": "bytes=2-5", "If-Range": f"W/{etag}"},
        )
        self.assertEqual(status_weak, 200)
        self.assertEqual(body_weak, b"0123456789abcdef")

    def test_api_status_schema_and_no_host_path_leak(self) -> None:
        status, headers, body = self._request("GET", "/api/status")
        self.assertEqual(status, 200)
        self._assert_isolation_headers(headers)
        self.assertEqual(headers.get("Content-Type"), "application/json")
        self.assertEqual(headers.get("Cache-Control"), "no-store")

        raw_text = body.decode("utf-8")
        data = json.loads(raw_text)
        expected_keys = {
            "ok",
            "coopCoep",
            "dataDirConfigured",
            "hasManifest",
            "hasBootset",
            "hasWasmBinary",
            "sideModuleCount",
            "sideModuleUrls",
            "shaderPackCount",
            "splashImage",
            "shaderPackUrl",
            "preferredShaderPackUrl",
            "reflectionManifestUrl",
            "pgimUrl",
            "programManifestUrl",
            "gameBase",
        }
        self.assertEqual(set(data.keys()), expected_keys)
        self.assertTrue(data["ok"])
        self.assertTrue(data["coopCoep"])
        self.assertTrue(data["dataDirConfigured"])
        self.assertFalse(data["hasManifest"])
        self.assertFalse(data["hasBootset"])
        self.assertFalse(data["hasWasmBinary"])
        self.assertEqual(data["sideModuleCount"], 0)
        self.assertEqual(data["sideModuleUrls"], [])
        self.assertEqual(data["shaderPackCount"], 0)
        self.assertIsNone(data["splashImage"])
        self.assertIsNone(data["shaderPackUrl"])
        self.assertIsNone(data["preferredShaderPackUrl"])
        self.assertIsNone(data["reflectionManifestUrl"])
        self.assertIsNone(data["pgimUrl"])
        self.assertIsNone(data["programManifestUrl"])
        self.assertIsNone(data["gameBase"])

        # Populate artifacts and verify dynamic detection (runtime.wasm excluded from sideModuleUrls).
        (self.data_root / "data-manifest.json").write_text("{}", encoding="utf-8")
        (self.data_root / "bootset.json").write_text("[]", encoding="utf-8")
        dist_dir = self.web_root / "dist"
        dist_dir.mkdir()
        (dist_dir / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
        (dist_dir / "runtime.js").write_text("export const ready = true;\n", encoding="utf-8")
        shaders_dir = self.data_root / "shaders"
        shaders_dir.mkdir()
        (shaders_dir / "base.wgsl.pack").write_bytes(b"WGPK")
        (self.data_root / "extra.wgsl.pack").write_bytes(b"WGPK")

        status2, _, body2 = self._request("GET", "/api/status")
        self.assertEqual(status2, 200)
        raw_text2 = body2.decode("utf-8")
        data2 = json.loads(raw_text2)
        self.assertTrue(data2["hasManifest"])
        self.assertTrue(data2["hasBootset"])
        self.assertTrue(data2["hasWasmBinary"])
        self.assertEqual(data2["sideModuleCount"], 0)
        self.assertEqual(data2["sideModuleUrls"], [])
        self.assertEqual(data2["shaderPackCount"], 2)
        self.assertEqual(data2["shaderPackUrl"], "/shaders/base.wgsl.pack")
        self.assertEqual(data2["preferredShaderPackUrl"], "/shaders/base.wgsl.pack")

        # Stage side-module .wasm chunks and verify natural-sorted discovery.
        for name in ("guest_part_10.wasm", "guest_part_2.wasm", "guest_part_0.wasm", "guest_part_1.wasm"):
            (dist_dir / name).write_bytes(b"\x00asm\x01\x00\x00\x00")

        status3, _, body3 = self._request("GET", "/api/status")
        self.assertEqual(status3, 200)
        raw_text3 = body3.decode("utf-8")
        data3 = json.loads(raw_text3)
        self.assertEqual(data3["sideModuleCount"], 4)
        self.assertEqual(
            data3["sideModuleUrls"],
            [
                "/dist/guest_part_0.wasm",
                "/dist/guest_part_1.wasm",
                "/dist/guest_part_2.wasm",
                "/dist/guest_part_10.wasm",
            ],
        )

        # Ensure no absolute host paths, temp paths, or usernames leak into JSON.
        for sensitive in (
            str(self.web_root),
            str(self.data_root),
            str(Path(self.tmp.name)),
            os.environ.get("USERNAME", ""),
            os.environ.get("USER", ""),
        ):
            if sensitive and len(sensitive) > 2:
                self.assertNotIn(sensitive, raw_text)
                self.assertNotIn(sensitive, raw_text2)
                self.assertNotIn(sensitive, raw_text3)

    def test_dist_js_and_wasm_etag_no_cache_and_relink_invalidation(self) -> None:
        dist_dir = self.web_root / "dist"
        dist_dir.mkdir()
        runtime_js = dist_dir / "runtime.js"
        runtime_wasm = dist_dir / "runtime.wasm"
        side_wasm = dist_dir / "guest_part_0.wasm"

        cases = (
            ("/dist/runtime.js", runtime_js, b"console.log('v1');\n", b"console.log('v2-relinked');\n", b"console.log('v3-relinked');\n", "text/javascript"),
            ("/dist/runtime.wasm", runtime_wasm, b"\x00asm\x01\x00\x00\x00v1", b"\x00asm\x01\x00\x00\x00v2-bigger", b"\x00asm\x01\x00\x00\x00v3-bigger", "application/wasm"),
            ("/dist/guest_part_0.wasm", side_wasm, b"\x00asm\x01\x00\x00\x00p0", b"\x00asm\x01\x00\x00\x00p0-new!", b"\x00asm\x01\x00\x00\x00p0-fin!", "application/wasm"),
        )

        for url_path, file_path, v1, v2_diff_size, v3_same_size, expected_mime in cases:
            with self.subTest(path=url_path):
                file_path.write_bytes(v1)
                expected_etag_v1 = serve_local.compute_etag(file_path.stat())

                # Initial GET -> 200 OK with Cache-Control: no-cache, strong ETag, COOP/COEP.
                st, hdr, body = self._request("GET", url_path)
                self.assertEqual(st, 200)
                self._assert_isolation_headers(hdr)
                self.assertEqual(hdr.get("Content-Type"), expected_mime)
                self.assertEqual(hdr.get("Cache-Control"), "no-cache")
                self.assertEqual(hdr.get("ETag"), expected_etag_v1)
                self.assertTrue(hdr.get("Last-Modified"))
                self.assertEqual(body, v1)

                # HEAD -> 200 OK with matching ETag & Cache-Control: no-cache, empty body.
                st_h, hdr_h, body_h = self._request("HEAD", url_path)
                self.assertEqual(st_h, 200)
                self._assert_isolation_headers(hdr_h)
                self.assertEqual(hdr_h.get("Cache-Control"), "no-cache")
                self.assertEqual(hdr_h.get("ETag"), expected_etag_v1)
                self.assertEqual(body_h, b"")

                # Revalidation with matching strong, weak, and list If-None-Match -> 304 + COOP/COEP.
                for inm in (
                    expected_etag_v1,
                    f"W/{expected_etag_v1}",
                    f'"other-tag", {expected_etag_v1}',
                ):
                    st304, hdr304, body304 = self._request("GET", url_path, {"If-None-Match": inm})
                    self.assertEqual(st304, 304)
                    self._assert_isolation_headers(hdr304)
                    self.assertEqual(hdr304.get("Cache-Control"), "no-cache")
                    self.assertEqual(hdr304.get("ETag"), expected_etag_v1)
                    self.assertTrue(hdr304.get("Last-Modified"))
                    self.assertEqual(body304, b"")

                # HEAD with matching If-None-Match -> 304 + COOP/COEP.
                st304_h, hdr304_h, body304_h = self._request(
                    "HEAD", url_path, {"If-None-Match": expected_etag_v1}
                )
                self.assertEqual(st304_h, 304)
                self._assert_isolation_headers(hdr304_h)
                self.assertEqual(hdr304_h.get("Cache-Control"), "no-cache")
                self.assertEqual(body304_h, b"")

                # HTTP 206 Range request preserves Cache-Control: no-cache and ETag.
                st206, hdr206, body206 = self._request("GET", url_path, {"Range": "bytes=0-3"})
                self.assertEqual(st206, 206)
                self._assert_isolation_headers(hdr206)
                self.assertEqual(hdr206.get("Cache-Control"), "no-cache")
                self.assertEqual(hdr206.get("ETag"), expected_etag_v1)
                self.assertEqual(body206, v1[:4])

                # Relink with different size -> stale If-None-Match MUST return 200 with fresh ETag.
                file_path.write_bytes(v2_diff_size)
                expected_etag_v2 = serve_local.compute_etag(file_path.stat())
                self.assertNotEqual(expected_etag_v2, expected_etag_v1)

                st_v2, hdr_v2, body_v2 = self._request(
                    "GET", url_path, {"If-None-Match": expected_etag_v1}
                )
                self.assertEqual(st_v2, 200)
                self._assert_isolation_headers(hdr_v2)
                self.assertEqual(hdr_v2.get("Cache-Control"), "no-cache")
                self.assertEqual(hdr_v2.get("ETag"), expected_etag_v2)
                self.assertEqual(body_v2, v2_diff_size)

                # Relink with identical byte size but bumped mtime_ns -> ETag MUST change -> 200 OK.
                self.assertEqual(len(v3_same_size), len(v2_diff_size))
                prev_stat = file_path.stat()
                file_path.write_bytes(v3_same_size)
                os.utime(
                    file_path,
                    ns=(prev_stat.st_atime_ns, prev_stat.st_mtime_ns + 5_000_000_000),
                )
                expected_etag_v3 = serve_local.compute_etag(file_path.stat())
                self.assertNotEqual(expected_etag_v3, expected_etag_v2)

                st_v3, hdr_v3, body_v3 = self._request(
                    "GET", url_path, {"If-None-Match": expected_etag_v2}
                )
                self.assertEqual(st_v3, 200)
                self._assert_isolation_headers(hdr_v3)
                self.assertEqual(hdr_v3.get("Cache-Control"), "no-cache")
                self.assertEqual(hdr_v3.get("ETag"), expected_etag_v3)
                self.assertEqual(body_v3, v3_same_size)

                # Subsequent revalidation with the latest ETag hits 304 again.
                st_v3_304, hdr_v3_304, body_v3_304 = self._request(
                    "GET", url_path, {"If-None-Match": expected_etag_v3}
                )
                self.assertEqual(st_v3_304, 304)
                self._assert_isolation_headers(hdr_v3_304)
                self.assertEqual(hdr_v3_304.get("ETag"), expected_etag_v3)
                self.assertEqual(body_v3_304, b"")

    def test_path_traversal_blocked(self) -> None:
        status, headers, _ = self._request("GET", "/data/../../README.md")
        self.assertEqual(status, 404)
        self._assert_isolation_headers(headers)

    def test_http11_keep_alive_reuses_single_socket_across_all_response_types(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            def send(method: str, path: str, headers: dict[str, str] | None = None):
                conn.request(method, path, headers=headers or {})
                resp = conn.getresponse()
                body = resp.read()
                return resp, dict(resp.getheaders()), body

            # 1. GET / -> 200 OK
            resp1, hdr1, body1 = send("GET", "/")
            self.assertEqual(resp1.version, 11)
            self.assertEqual(resp1.status, 200)
            self.assertFalse(resp1.will_close)
            self.assertEqual(hdr1.get("Connection", "").lower(), "keep-alive")
            self.assertEqual(hdr1.get("Content-Length"), str(len(body1)))
            self._assert_isolation_headers(hdr1)
            self.assertEqual(body1, b"<h1>ok</h1>")
            self.assertIsNotNone(conn.sock)
            first_sock = conn.sock
            first_peer = first_sock.getsockname()

            # 2. HEAD /data/sample.bin -> 200 OK with exact Content-Length, empty body
            resp2, hdr2, body2 = send("HEAD", "/data/sample.bin")
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(conn.sock.getsockname(), first_peer)
            self.assertEqual(resp2.version, 11)
            self.assertEqual(resp2.status, 200)
            self.assertFalse(resp2.will_close)
            self.assertEqual(hdr2.get("Content-Length"), "16")
            self._assert_isolation_headers(hdr2)
            self.assertEqual(body2, b"")
            etag = hdr2["ETag"]

            # 3. GET /data/sample.bin with Range -> 206 Partial Content
            resp3, hdr3, body3 = send("GET", "/data/sample.bin", {"Range": "bytes=4-9"})
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp3.version, 11)
            self.assertEqual(resp3.status, 206)
            self.assertFalse(resp3.will_close)
            self.assertEqual(hdr3.get("Content-Range"), "bytes 4-9/16")
            self.assertEqual(hdr3.get("Content-Length"), "6")
            self._assert_isolation_headers(hdr3)
            self.assertEqual(body3, b"456789")

            # 4. GET /data/sample.bin with If-None-Match -> 304 Not Modified (must not hang)
            resp4, hdr4, body4 = send("GET", "/data/sample.bin", {"If-None-Match": etag})
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp4.version, 11)
            self.assertEqual(resp4.status, 304)
            self.assertFalse(resp4.will_close)
            self._assert_isolation_headers(hdr4)
            self.assertEqual(body4, b"")

            # 5. GET /api/status -> 200 OK JSON with exact Content-Length
            resp5, hdr5, body5 = send("GET", "/api/status")
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp5.version, 11)
            self.assertEqual(resp5.status, 200)
            self.assertFalse(resp5.will_close)
            self.assertEqual(hdr5.get("Content-Length"), str(len(body5)))
            self._assert_isolation_headers(hdr5)
            self.assertTrue(json.loads(body5.decode("utf-8"))["ok"])

            # 6. HEAD /api/status -> 200 OK with Content-Length > 0 and empty body
            resp6, hdr6, body6 = send("HEAD", "/api/status")
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp6.version, 11)
            self.assertEqual(resp6.status, 200)
            self.assertFalse(resp6.will_close)
            self.assertGreater(int(hdr6.get("Content-Length", "0")), 0)
            self._assert_isolation_headers(hdr6)
            self.assertEqual(body6, b"")

            # 7. OPTIONS / -> 204 No Content with Allow header
            resp7, hdr7, body7 = send("OPTIONS", "/")
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp7.version, 11)
            self.assertEqual(resp7.status, 204)
            self.assertFalse(resp7.will_close)
            self.assertIn("GET", hdr7.get("Allow", ""))
            self.assertIn("HEAD", hdr7.get("Allow", ""))
            self.assertIn("OPTIONS", hdr7.get("Allow", ""))
            self._assert_isolation_headers(hdr7)
            self.assertEqual(body7, b"")

            # 8. GET /nonexistent -> 404 Not Found with Content-Length: 0
            resp8, hdr8, body8 = send("GET", "/nonexistent-asset.bin")
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp8.version, 11)
            self.assertEqual(resp8.status, 404)
            self.assertFalse(resp8.will_close)
            self.assertEqual(hdr8.get("Content-Length"), "0")
            self._assert_isolation_headers(hdr8)
            self.assertEqual(body8, b"")

            # 9. GET /data/sample.bin with invalid Range -> 416 with Content-Length: 0
            resp9, hdr9, body9 = send("GET", "/data/sample.bin", {"Range": "bytes=100-200"})
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp9.version, 11)
            self.assertEqual(resp9.status, 416)
            self.assertFalse(resp9.will_close)
            self.assertEqual(hdr9.get("Content-Length"), "0")
            self.assertEqual(hdr9.get("Content-Range"), "bytes */16")
            self._assert_isolation_headers(hdr9)
            self.assertEqual(body9, b"")

            # 10. Final GET /demo.wasm proves socket stayed synchronized and reusable
            resp10, hdr10, body10 = send("GET", "/demo.wasm")
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(conn.sock.getsockname(), first_peer)
            self.assertEqual(resp10.version, 11)
            self.assertEqual(resp10.status, 200)
            self.assertFalse(resp10.will_close)
            self.assertEqual(hdr10.get("Content-Length"), "8")
            self._assert_isolation_headers(hdr10)
            self.assertEqual(body10, b"\x00asm\x01\x00\x00\x00")
        finally:
            conn.close()

    def test_http11_connection_close_closes_socket_cleanly(self) -> None:
        for conn_header in ("close", "keep-alive, close"):
            with self.subTest(connection_header=conn_header):
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                try:
                    conn.request("GET", "/", headers={"Connection": conn_header})
                    resp = conn.getresponse()
                    body = resp.read()
                    headers = dict(resp.getheaders())
                    self.assertEqual(resp.version, 11)
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(headers.get("Connection", "").lower(), "close")
                    self.assertEqual(headers.get("Content-Length"), str(len(body)))
                    self.assertTrue(resp.will_close)
                    self.assertIsNone(conn.sock)
                    self.assertEqual(body, b"<h1>ok</h1>")
                finally:
                    conn.close()

    def test_unsupported_method_returns_405_with_content_length_and_closes(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("POST", "/api/status", body=b'{"x":1}')
            resp = conn.getresponse()
            body = resp.read()
            headers = dict(resp.getheaders())
            self.assertEqual(resp.version, 11)
            self.assertEqual(resp.status, 405)
            self.assertEqual(headers.get("Content-Length"), "0")
            self.assertEqual(headers.get("Connection", "").lower(), "close")
            self.assertIn("GET", headers.get("Allow", ""))
            self._assert_isolation_headers(headers)
            self.assertTrue(resp.will_close)
            self.assertEqual(body, b"")
        finally:
            conn.close()

    def test_malformed_request_returns_400_with_http11_and_content_length(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(b"BADREQUEST\r\n\r\n")
            chunks: list[bytes] = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
        raw = b"".join(chunks)
        header_block, _, body = raw.partition(b"\r\n\r\n")
        header_text = header_block.decode("iso-8859-1")
        self.assertTrue(header_text.startswith("HTTP/1.1 400 "))
        self.assertIn("Connection: close", header_text)
        self.assertIn(f"Content-Length: {len(body)}", header_text)

    def test_large_range_streaming_and_1mib_chunk_size(self) -> None:
        self.assertGreaterEqual(serve_local.STREAM_CHUNK_SIZE, 512 * 1024)
        self.assertEqual(serve_local.STREAM_CHUNK_SIZE, 1024 * 1024)

        # Build a 2.5 MiB deterministic payload and stream full + 2 MiB Range on one socket.
        pattern = bytes(range(256))
        total_size = 2_621_440  # 2.5 MiB
        payload = pattern * (total_size // len(pattern))
        (self.data_root / "large.pgim").write_bytes(payload)

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            range_start = 262_144
            range_len = 2 * 1024 * 1024
            range_end = range_start + range_len - 1
            conn.request(
                "GET",
                "/data/large.pgim",
                headers={"Range": f"bytes={range_start}-{range_end}"},
            )
            resp1 = conn.getresponse()
            body1 = resp1.read()
            hdr1 = dict(resp1.getheaders())
            first_sock = conn.sock

            self.assertEqual(resp1.version, 11)
            self.assertEqual(resp1.status, 206)
            self.assertFalse(resp1.will_close)
            self.assertEqual(hdr1.get("Content-Length"), str(range_len))
            self.assertEqual(hdr1.get("Content-Range"), f"bytes {range_start}-{range_end}/{total_size}")
            self.assertEqual(body1, payload[range_start : range_end + 1])

            conn.request("GET", "/data/large.pgim")
            resp2 = conn.getresponse()
            body2 = resp2.read()
            hdr2 = dict(resp2.getheaders())
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp2.status, 200)
            self.assertEqual(hdr2.get("Content-Length"), str(total_size))
            self.assertEqual(body2, payload)
        finally:
            conn.close()

    def test_keep_alive_idle_timeout_closes_quietly(self) -> None:
        httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            keep_alive_timeout=0.15,
        )
        port = httpd.server_address[1]
        stderr_buf = io.StringIO()
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with contextlib.redirect_stderr(stderr_buf):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                try:
                    conn.request("GET", "/")
                    resp = conn.getresponse()
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(resp.read(), b"<h1>ok</h1>")
                    self.assertIsNotNone(conn.sock)
                    # Wait past the 0.15s idle keep-alive timeout; server should close silently.
                    time.sleep(0.35)
                    conn.sock.settimeout(2.0)
                    self.assertEqual(conn.sock.recv(16), b"")
                finally:
                    conn.close()
            stderr_text = stderr_buf.getvalue()
            self.assertNotIn("Traceback", stderr_text)
            self.assertNotIn("Request timed out", stderr_text)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

    def test_mid_body_truncation_marks_close_connection(self) -> None:
        sample = self.data_root / "shrinking.bin"
        sample.write_bytes(b"0123456789abcdef")

        class _ShrinkingReader:
            def __init__(self) -> None:
                self._sent = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def seek(self, _offset: int) -> None:
                return None

            def read(self, _n: int) -> bytes:
                if not self._sent:
                    self._sent = True
                    return b"0123"
                return b""

            def close(self) -> None:
                return None

        written: list[bytes] = []

        class _CollectingWriter:
            def write(self, data: bytes) -> int:
                written.append(data)
                return len(data)

        handler = object.__new__(serve_local.PlayGta6Handler)
        handler.web_root = self.web_root
        handler.data_root = self.data_root
        handler.game_root = None
        handler.shaders_root = self.data_root / "shaders"
        handler.boot_root = None
        handler.headers = {}
        handler.close_connection = False
        handler.wfile = _CollectingWriter()
        handler.send_response = lambda *a, **kw: None
        handler.send_header = lambda *a, **kw: None
        handler.end_headers = lambda *a, **kw: None

        orig_resolve = handler._resolve_target
        fake_path = type(
            "_FakePath",
            (),
            {
                "is_file": lambda self: True,
                "stat": lambda self: sample.stat(),
                "open": lambda self, _mode="rb": _ShrinkingReader(),
                "__str__": lambda self: str(sample),
            },
        )()
        handler._resolve_target = lambda _url: fake_path
        try:
            handler._serve_file("/data/shrinking.bin", send_body=True)
        finally:
            handler._resolve_target = orig_resolve

        self.assertEqual(b"".join(written), b"0123")
        self.assertTrue(handler.close_connection)


class TestServeLocalDumpEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.web_root = self.base / "web"
        self.data_root = self.base / "data"
        self.dump_root = self.base / "dumps"
        self.outside_root = self.base / "outside"
        self.web_root.mkdir()
        self.data_root.mkdir()
        self.dump_root.mkdir()
        self.outside_root.mkdir()
        (self.web_root / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @contextlib.contextmanager
    def _running_server(
        self,
        *,
        dump_root: Path | None = None,
        max_dump_file_bytes: int = serve_local.DEFAULT_MAX_DUMP_FILE_BYTES,
    ):
        httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            dump_root=dump_root,
            max_dump_file_bytes=max_dump_file_bytes,
        )
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield port
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

    def _request(
        self,
        port: int,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            resp_body = resp.read()
            return resp.status, dict(resp.getheaders()), resp_body
        finally:
            conn.close()

    def _assert_isolation_headers(self, headers: dict[str, str]) -> None:
        self.assertEqual(headers.get("Cross-Origin-Opener-Policy"), "same-origin")
        self.assertEqual(headers.get("Cross-Origin-Embedder-Policy"), "require-corp")
        self.assertEqual(headers.get("Cross-Origin-Resource-Policy"), "same-origin")
        self.assertEqual(headers.get("Accept-Ranges"), "bytes")

    def _assert_no_host_path_leak(self, raw_text: str) -> None:
        for sensitive in (
            str(self.base),
            str(self.dump_root),
            str(self.outside_root),
            str(self.web_root),
            str(self.data_root),
            os.environ.get("USERNAME", ""),
            os.environ.get("USER", ""),
        ):
            if sensitive and len(sensitive) > 2:
                self.assertNotIn(sensitive, raw_text)

    def test_disabled_by_default_rejects_dump_post(self) -> None:
        with self._running_server(dump_root=None) as port:
            status, headers, body = self._request(
                port,
                "POST",
                "/__pg6_dump/dxil/shader_001.bin",
                body=b"DXBC\x01\x02\x03\x04",
            )
            self.assertIn(status, (403, 404))
            self.assertEqual(status, 403)
            self._assert_isolation_headers(headers)
            self.assertEqual(headers.get("Content-Type"), "application/json")
            self.assertEqual(headers.get("Cache-Control"), "no-store")
            text = body.decode("utf-8")
            self._assert_no_host_path_leak(text)
            payload = json.loads(text)
            self.assertFalse(payload["ok"])
            self.assertEqual(list(self.dump_root.iterdir()), [])

            # Non-dump POST endpoints remain 405 Method Not Allowed.
            st405, hdr405, body405 = self._request(
                port, "POST", "/api/status", body=b'{"x":1}'
            )
            self.assertEqual(st405, 405)
            self._assert_isolation_headers(hdr405)
            self.assertEqual(body405, b"")

    def test_valid_atomic_write_and_overwrite(self) -> None:
        with self._running_server(dump_root=self.dump_root) as port:
            data1 = b"DXBC\x00\x01\x02\x03menu_intro_pipeline"
            status, headers, body = self._request(
                port,
                "POST",
                "/__pg6_dump/menu/intro/pso_deadbeef.dxil",
                body=data1,
            )
            self.assertEqual(status, 200)
            self._assert_isolation_headers(headers)
            self.assertEqual(headers.get("Content-Type"), "application/json")
            self.assertEqual(headers.get("Cache-Control"), "no-store")
            raw_text = body.decode("utf-8")
            self._assert_no_host_path_leak(raw_text)
            resp_json = json.loads(raw_text)
            self.assertEqual(
                resp_json,
                {
                    "ok": True,
                    "path": "menu/intro/pso_deadbeef.dxil",
                    "bytes_written": len(data1),
                    "total_bytes": len(data1),
                },
            )
            written_file = self.dump_root / "menu" / "intro" / "pso_deadbeef.dxil"
            self.assertEqual(written_file.read_bytes(), data1)

            # Overwrite without ?append=1 replaces contents atomically.
            data2 = b"SHORTER"
            st2, hdr2, body2 = self._request(
                port,
                "POST",
                "/__pg6_dump/menu/intro/pso_deadbeef.dxil",
                body=data2,
            )
            self.assertEqual(st2, 200)
            self._assert_isolation_headers(hdr2)
            self.assertEqual(
                json.loads(body2.decode("utf-8")),
                {
                    "ok": True,
                    "path": "menu/intro/pso_deadbeef.dxil",
                    "bytes_written": len(data2),
                    "total_bytes": len(data2),
                },
            )
            self.assertEqual(written_file.read_bytes(), data2)

            # Zero-byte write is valid and leaves no temporary files behind.
            st3, hdr3, body3 = self._request(
                port, "POST", "/__pg6_dump/empty.bin", body=b""
            )
            self.assertEqual(st3, 200)
            self._assert_isolation_headers(hdr3)
            self.assertEqual(
                json.loads(body3.decode("utf-8")),
                {
                    "ok": True,
                    "path": "empty.bin",
                    "bytes_written": 0,
                    "total_bytes": 0,
                },
            )
            self.assertEqual((self.dump_root / "empty.bin").read_bytes(), b"")
            leftover_tmp = list(self.dump_root.rglob(".pg6_dump_*.tmp"))
            self.assertEqual(leftover_tmp, [])

            # Non-dump POST still returns 405 even when dump endpoint is enabled.
            st405, hdr405, body405 = self._request(
                port, "POST", "/api/status", body=b"{}"
            )
            self.assertEqual(st405, 405)
            self._assert_isolation_headers(hdr405)
            self.assertEqual(body405, b"")

            # OPTIONS on dump endpoint advertises POST while / stays GET, HEAD, OPTIONS.
            st_opt, hdr_opt, _ = self._request(port, "OPTIONS", "/__pg6_dump/a.bin")
            self.assertEqual(st_opt, 204)
            self._assert_isolation_headers(hdr_opt)
            self.assertIn("POST", hdr_opt.get("Allow", ""))

    def test_append_chunked_streaming_write_and_keep_alive(self) -> None:
        with self._running_server(dump_root=self.dump_root) as port:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                chunks = [b"CHUNK0:", b"DXIL_BODY_", b"FINAL_TAIL!"]
                expected_total = 0
                first_sock = None
                for idx, chunk in enumerate(chunks):
                    path = "/__pg6_dump/stream/pipeline.dxil?append=1"
                    conn.request("POST", path, body=chunk)
                    resp = conn.getresponse()
                    raw = resp.read()
                    hdrs = dict(resp.getheaders())
                    self.assertEqual(resp.status, 200)
                    self._assert_isolation_headers(hdrs)
                    self.assertFalse(resp.will_close)
                    if idx == 0:
                        first_sock = conn.sock
                    else:
                        self.assertIs(conn.sock, first_sock)
                    expected_total += len(chunk)
                    data = json.loads(raw.decode("utf-8"))
                    self.assertEqual(
                        data,
                        {
                            "ok": True,
                            "path": "stream/pipeline.dxil",
                            "bytes_written": len(chunk),
                            "total_bytes": expected_total,
                        },
                    )
                    self._assert_no_host_path_leak(raw.decode("utf-8"))

                out_file = self.dump_root / "stream" / "pipeline.dxil"
                self.assertEqual(out_file.read_bytes(), b"".join(chunks))

                # ?append=0 overwrites instead of appending.
                conn.request("POST", "/__pg6_dump/stream/pipeline.dxil?append=0", body=b"RESET")
                resp_reset = conn.getresponse()
                raw_reset = resp_reset.read()
                self.assertEqual(resp_reset.status, 200)
                self.assertEqual(
                    json.loads(raw_reset.decode("utf-8")),
                    {
                        "ok": True,
                        "path": "stream/pipeline.dxil",
                        "bytes_written": 5,
                        "total_bytes": 5,
                    },
                )
                self.assertEqual(out_file.read_bytes(), b"RESET")
            finally:
                conn.close()

    def test_traversal_and_invalid_relpath_rejected(self) -> None:
        (self.dump_root / "existing_dir").mkdir()
        (self.dump_root / "regular_file.bin").write_bytes(b"abc")

        bad_paths = [
            "/__pg6_dump/",
            "/__pg6_dump/?append=1",
            "/__pg6_dump/.",
            "/__pg6_dump/..",
            "/__pg6_dump/../escape.bin",
            "/__pg6_dump/sub/../../escape.bin",
            "/__pg6_dump/sub/./file.bin",
            "/__pg6_dump/%2e%2e/escape.bin",
            "/__pg6_dump/%2E%2E/escape.bin",
            "/__pg6_dump/sub/%2e%2e/%2e%2e/escape.bin",
            "/__pg6_dump/..%2fescape.bin",
            "/__pg6_dump//etc/passwd",
            "/__pg6_dump/%2fetc/passwd",
            "/__pg6_dump/C:/Windows/win.ini",
            "/__pg6_dump/C%3A/foo.bin",
            "/__pg6_dump/c:foo.bin",
            "/__pg6_dump/file.bin:ads",
            "/__pg6_dump/..%5cescape.bin",
            "/__pg6_dump/%5cetc%5cpasswd",
            "/__pg6_dump/sub%5cfile.bin",
            "/__pg6_dump/foo%00bar.bin",
            "/__pg6_dump/foo%0abar.bin",
            "/__pg6_dump/existing_dir/",
            "/__pg6_dump/existing_dir",
            "/__pg6_dump/regular_file.bin/child.bin",
        ]

        with self._running_server(dump_root=self.dump_root) as port:
            for url_path in bad_paths:
                with self.subTest(url_path=url_path):
                    status, headers, body = self._request(
                        port, "POST", url_path, body=b"bad"
                    )
                    self.assertIn(status, (400, 403))
                    self._assert_isolation_headers(headers)
                    self.assertEqual(headers.get("Content-Type"), "application/json")
                    text = body.decode("utf-8")
                    self._assert_no_host_path_leak(text)
                    self.assertFalse(json.loads(text)["ok"])

        self.assertFalse((self.base / "escape.bin").exists())
        self.assertEqual(list(self.outside_root.iterdir()), [])

    def test_symlink_escape_rejected(self) -> None:
        dir_link = self.dump_root / "escape_dir"
        file_link = self.dump_root / "escape_file.bin"
        dangling_link = self.dump_root / "dangling_escape.bin"
        outside_file = self.outside_root / "target.bin"
        outside_file.write_bytes(b"ORIGINAL")

        try:
            os.symlink(self.outside_root, dir_link, target_is_directory=True)
            os.symlink(outside_file, file_link, target_is_directory=False)
            os.symlink(
                self.outside_root / "nonexistent.bin",
                dangling_link,
                target_is_directory=False,
            )
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable in this environment: {exc}")

        with self._running_server(dump_root=self.dump_root) as port:
            for url_path in (
                "/__pg6_dump/escape_dir/pwn.bin",
                "/__pg6_dump/escape_dir/nested/pwn.bin",
                "/__pg6_dump/escape_file.bin",
                "/__pg6_dump/escape_file.bin?append=1",
                "/__pg6_dump/dangling_escape.bin",
            ):
                with self.subTest(url_path=url_path):
                    status, headers, body = self._request(
                        port, "POST", url_path, body=b"OVERWRITTEN"
                    )
                    self.assertEqual(status, 403)
                    self._assert_isolation_headers(headers)
                    text = body.decode("utf-8")
                    self._assert_no_host_path_leak(text)
                    self.assertFalse(json.loads(text)["ok"])

        self.assertEqual(outside_file.read_bytes(), b"ORIGINAL")
        self.assertFalse((self.outside_root / "pwn.bin").exists())
        self.assertFalse((self.outside_root / "nonexistent.bin").exists())

    def test_oversized_payload_returns_413_for_single_and_append(self) -> None:
        cap = 32
        with self._running_server(
            dump_root=self.dump_root, max_dump_file_bytes=cap
        ) as port:
            # Exact boundary succeeds.
            st_ok, hdr_ok, body_ok = self._request(
                port, "POST", "/__pg6_dump/exact.bin", body=b"A" * cap
            )
            self.assertEqual(st_ok, 200)
            self._assert_isolation_headers(hdr_ok)
            self.assertEqual(json.loads(body_ok.decode("utf-8"))["total_bytes"], cap)

            # Single payload exceeding cap -> 413 Payload Too Large, file not created.
            st_big, hdr_big, body_big = self._request(
                port, "POST", "/__pg6_dump/too_big.bin", body=b"B" * (cap + 1)
            )
            self.assertEqual(st_big, 413)
            self._assert_isolation_headers(hdr_big)
            self.assertEqual(hdr_big.get("Connection", "").lower(), "close")
            self._assert_no_host_path_leak(body_big.decode("utf-8"))
            self.assertFalse(json.loads(body_big.decode("utf-8"))["ok"])
            self.assertFalse((self.dump_root / "too_big.bin").exists())

            # Append mode checks resulting total size <= cap.
            st1, _, _ = self._request(
                port, "POST", "/__pg6_dump/chunked.bin", body=b"1" * 20
            )
            self.assertEqual(st1, 200)
            st2, _, body2 = self._request(
                port, "POST", "/__pg6_dump/chunked.bin?append=1", body=b"2" * 12
            )
            self.assertEqual(st2, 200)
            self.assertEqual(json.loads(body2.decode("utf-8"))["total_bytes"], 32)

            st3, hdr3, body3 = self._request(
                port, "POST", "/__pg6_dump/chunked.bin?append=1", body=b"X"
            )
            self.assertEqual(st3, 413)
            self._assert_isolation_headers(hdr3)
            self._assert_no_host_path_leak(body3.decode("utf-8"))
            self.assertFalse(json.loads(body3.decode("utf-8"))["ok"])
            # Existing file contents remain intact after rejected append.
            self.assertEqual(
                (self.dump_root / "chunked.bin").read_bytes(),
                (b"1" * 20) + (b"2" * 12),
            )

    def test_missing_content_length_returns_411(self) -> None:
        with self._running_server(dump_root=self.dump_root) as port:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                sock.sendall(
                    b"POST /__pg6_dump/nolen.bin HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
                )
                chunks: list[bytes] = []
                while True:
                    data = sock.recv(4096)
                    if not data:
                        break
                    chunks.append(data)
            raw = b"".join(chunks)
            header_block, _, body = raw.partition(b"\r\n\r\n")
            header_text = header_block.decode("iso-8859-1")
            self.assertTrue(header_text.startswith("HTTP/1.1 411 "))
            self.assertIn("Cross-Origin-Opener-Policy: same-origin", header_text)
            self.assertIn("Cross-Origin-Embedder-Policy: require-corp", header_text)
            self.assertFalse(json.loads(body.decode("utf-8"))["ok"])
            self.assertFalse((self.dump_root / "nolen.bin").exists())

    def test_resolve_dump_target_unit_and_negative_cap(self) -> None:
        res, status, err = serve_local.resolve_dump_target(
            self.dump_root, "dxil/frame_01/pso.bin"
        )
        self.assertIsNotNone(res)
        self.assertIsNone(status)
        self.assertIsNone(err)
        target, rel = res
        self.assertEqual(rel, "dxil/frame_01/pso.bin")
        self.assertEqual(target, self.dump_root.resolve() / "dxil" / "frame_01" / "pso.bin")

        for bad in ("", "/abs.bin", "../up.bin", "a\\b.bin", "C:/x.bin", "a\x00b.bin", "a//b.bin"):
            with self.subTest(bad=bad):
                res_bad, st_bad, err_bad = serve_local.resolve_dump_target(self.dump_root, bad)
                self.assertIsNone(res_bad)
                self.assertIn(st_bad, (400, 403))
                self.assertTrue(err_bad)

        with self.assertRaises(ValueError):
            serve_local.create_server(
                "127.0.0.1",
                0,
                self.web_root,
                self.data_root,
                dump_root=self.dump_root,
                max_dump_file_bytes=-1,
            )

    def test_default_max_dump_file_bytes_is_256mib_and_keepalive_reuses_socket(self) -> None:
        self.assertEqual(serve_local.DEFAULT_MAX_DUMP_FILE_BYTES, 256 * 1024 * 1024)
        with self._running_server(dump_root=self.dump_root) as port:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                sock_ids: list[int] = []
                for idx in range(5):
                    payload = f"shader-{idx}".encode("utf-8")
                    conn.request(
                        "POST",
                        f"/__pg6_dump/dxil/s_{idx:04d}.dxil",
                        body=payload,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "Connection": "keep-alive",
                        },
                    )
                    resp = conn.getresponse()
                    body = resp.read()
                    self.assertEqual(resp.status, 200)
                    self.assertNotEqual(resp.getheader("Connection", "").lower(), "close")
                    self.assertTrue(json.loads(body.decode("utf-8"))["ok"])
                    self.assertIsNotNone(conn.sock)
                    sock_ids.append(conn.sock.fileno())
                self.assertEqual(len(set(sock_ids)), 1, "expected HTTP/1.1 keep-alive socket reuse")
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
