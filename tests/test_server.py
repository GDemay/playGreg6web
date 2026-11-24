import http.client
import json
import os
import sys
import tempfile
import threading
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


if __name__ == "__main__":
    unittest.main()
