import concurrent.futures
from functools import partial
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

import serve_local  # noqa: E402


def _make_deterministic_bytes(size: int) -> bytes:
    """Create a deterministic multi-MiB byte pattern with unique 4-byte page indices."""
    block = bytearray(size)
    page = 4096
    for offset in range(0, size, page):
        end = min(offset + page, size)
        tag = (offset // page).to_bytes(4, "little")
        chunk_len = end - offset
        repeats = (chunk_len // 4) + 1
        block[offset:end] = (tag * repeats)[:chunk_len]
    return bytes(block)


class TestServeLocalRangeFastpath(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.web_root = base / "web"
        self.data_root = base / "data"
        self.dump_root = base / "dumps"
        self.web_root.mkdir()
        self.data_root.mkdir()
        self.dump_root.mkdir()

        (self.web_root / "index.html").write_text("<h1>playgta6web</h1>", encoding="utf-8")
        (self.web_root / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")

        # 3.5 MiB deterministic payload spanning >3x 1 MiB STREAM_CHUNK_BYTES buffers.
        self.large_size = 3 * 1024 * 1024 + 512 * 1024
        self.large_payload = _make_deterministic_bytes(self.large_size)
        self.large_file = self.data_root / "splash.wgsl.pack"
        self.large_file.write_bytes(self.large_payload)

        self.httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def _request(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        *,
        port: int | None = None,
        body: bytes | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", port or self.port, timeout=10)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, dict(resp.getheaders()), data
        finally:
            conn.close()

    def _assert_coop_coep(self, headers: dict[str, str]) -> None:
        self.assertEqual(headers.get("Cross-Origin-Opener-Policy"), "same-origin")
        self.assertEqual(headers.get("Cross-Origin-Embedder-Policy"), "require-corp")
        self.assertEqual(headers.get("Cross-Origin-Resource-Policy"), "same-origin")
        self.assertEqual(headers.get("Accept-Ranges"), "bytes")

    def test_stream_chunk_bytes_constant_is_1mib_and_validated(self) -> None:
        self.assertEqual(serve_local.STREAM_CHUNK_BYTES, 1024 * 1024)
        self.assertEqual(serve_local.STREAM_CHUNK_SIZE, serve_local.STREAM_CHUNK_BYTES)
        self.assertGreaterEqual(serve_local.SOCKET_SNDBUF_BYTES, 1024 * 1024)
        self.assertTrue(serve_local.PlayGta6Handler.disable_nagle_algorithm)

        with self.assertRaises(ValueError):
            serve_local.create_server(
                "127.0.0.1",
                0,
                self.web_root,
                self.data_root,
                stream_chunk_bytes=0,
            )
        with self.assertRaises(ValueError):
            serve_local.create_server(
                "127.0.0.1",
                0,
                self.web_root,
                self.data_root,
                stream_chunk_bytes=-4096,
            )

    def test_client_socket_sets_tcp_nodelay(self) -> None:
        observed: list[tuple[int, int]] = []

        class _SocketProbeHandler(serve_local.PlayGta6Handler):
            def log_message(self, format: str, *args) -> None:
                return None

            def do_GET(self) -> None:
                nodelay = self.connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
                sndbuf = self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                observed.append((nodelay, sndbuf))
                super().do_GET()

        handler = partial(
            _SocketProbeHandler,
            web_root=self.web_root,
            data_root=self.data_root,
        )
        httpd = serve_local.QuietThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            status, headers, body = self._request("GET", "/runtime.wasm", port=port)
            self.assertEqual(status, 200)
            self._assert_coop_coep(headers)
            self.assertEqual(body, b"\x00asm\x01\x00\x00\x00")
            self.assertEqual(len(observed), 1)
            nodelay, sndbuf = observed[0]
            self.assertNotEqual(nodelay, 0, "TCP_NODELAY must be enabled on client sockets")
            self.assertGreater(sndbuf, 0)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

    def test_etag_if_none_match_and_if_modified_since_304_preserve_coop_coep(self) -> None:
        st, hdr, body = self._request("GET", "/runtime.wasm")
        self.assertEqual(st, 200)
        self._assert_coop_coep(hdr)
        self.assertEqual(hdr.get("Cache-Control"), "no-cache")
        etag = hdr.get("ETag")
        last_modified = hdr.get("Last-Modified")
        self.assertTrue(etag)
        self.assertTrue(last_modified)
        self.assertEqual(body, b"\x00asm\x01\x00\x00\x00")

        # GET and HEAD with exact, weak-prefixed, list, and wildcard If-None-Match -> 304 + COOP/COEP
        for method in ("GET", "HEAD"):
            for inm in (etag, f"W/{etag}", f'"other", {etag}', "*"):
                with self.subTest(method=method, if_none_match=inm):
                    st304, hdr304, body304 = self._request(
                        method, "/runtime.wasm", {"If-None-Match": inm}
                    )
                    self.assertEqual(st304, 304)
                    self._assert_coop_coep(hdr304)
                    self.assertEqual(hdr304.get("ETag"), etag)
                    self.assertEqual(hdr304.get("Last-Modified"), last_modified)
                    self.assertEqual(hdr304.get("Cache-Control"), "no-cache")
                    self.assertEqual(body304, b"")

        # If-Modified-Since without If-None-Match -> 304 with COOP/COEP
        for method in ("GET", "HEAD"):
            with self.subTest(method=method, conditional="If-Modified-Since"):
                st_ims, hdr_ims, body_ims = self._request(
                    method, "/runtime.wasm", {"If-Modified-Since": last_modified}
                )
                self.assertEqual(st_ims, 304)
                self._assert_coop_coep(hdr_ims)
                self.assertEqual(hdr_ims.get("ETag"), etag)
                self.assertEqual(hdr_ims.get("Cache-Control"), "no-cache")
                self.assertEqual(body_ims, b"")

        # Per RFC 9110, If-None-Match takes precedence over If-Modified-Since:
        # stale If-None-Match must return 200 even if If-Modified-Since matches.
        st_prec, hdr_prec, body_prec = self._request(
            "GET",
            "/runtime.wasm",
            {"If-None-Match": '"stale-etag"', "If-Modified-Since": last_modified},
        )
        self.assertEqual(st_prec, 200)
        self._assert_coop_coep(hdr_prec)
        self.assertEqual(body_prec, b"\x00asm\x01\x00\x00\x00")

        # Same byte length with bumped mtime_ns invalidates ETag -> 200 OK.
        wasm_path = self.web_root / "runtime.wasm"
        prev_stat = wasm_path.stat()
        updated_bytes = b"\x00asm\x01\x00\x00\x01"
        wasm_path.write_bytes(updated_bytes)
        os.utime(wasm_path, ns=(prev_stat.st_atime_ns, prev_stat.st_mtime_ns + 2_000_000_000))
        st_new, hdr_new, body_new = self._request(
            "GET", "/runtime.wasm", {"If-None-Match": etag}
        )
        self.assertEqual(st_new, 200)
        self._assert_coop_coep(hdr_new)
        self.assertNotEqual(hdr_new.get("ETag"), etag)
        self.assertEqual(body_new, updated_bytes)

    def test_multi_mib_206_partial_content_streaming_fidelity_and_keep_alive(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            # 1. Multi-MiB interior Range spanning >2x STREAM_CHUNK_BYTES (2.5 MiB)
            start = 131_072
            length = 2 * 1024 * 1024 + 512 * 1024
            end = start + length - 1
            conn.request("GET", "/data/splash.wgsl.pack", headers={"Range": f"bytes={start}-{end}"})
            resp1 = conn.getresponse()
            body1 = resp1.read()
            hdr1 = dict(resp1.getheaders())
            first_sock = conn.sock

            self.assertEqual(resp1.status, 206)
            self.assertFalse(resp1.will_close)
            self._assert_coop_coep(hdr1)
            self.assertEqual(hdr1.get("Cache-Control"), "no-cache")
            self.assertTrue(hdr1.get("ETag"))
            self.assertEqual(hdr1.get("Content-Length"), str(length))
            self.assertEqual(hdr1.get("Content-Range"), f"bytes {start}-{end}/{self.large_size}")
            self.assertEqual(
                hashlib.sha256(body1).hexdigest(),
                hashlib.sha256(self.large_payload[start : end + 1]).hexdigest(),
            )
            self.assertEqual(body1, self.large_payload[start : end + 1])

            # 2. Suffix Range spanning >1 MiB (1.5 MiB) on same keep-alive socket
            suffix_len = 1 * 1024 * 1024 + 512 * 1024
            conn.request("GET", "/data/splash.wgsl.pack", headers={"Range": f"bytes=-{suffix_len}"})
            resp2 = conn.getresponse()
            body2 = resp2.read()
            hdr2 = dict(resp2.getheaders())
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp2.status, 206)
            self._assert_coop_coep(hdr2)
            self.assertEqual(
                hdr2.get("Content-Range"),
                f"bytes {self.large_size - suffix_len}-{self.large_size - 1}/{self.large_size}",
            )
            self.assertEqual(body2, self.large_payload[-suffix_len:])

            # 3. Open-ended Range spanning >2 MiB on same keep-alive socket
            open_start = 1024 * 1024
            conn.request("GET", "/data/splash.wgsl.pack", headers={"Range": f"bytes={open_start}-"})
            resp3 = conn.getresponse()
            body3 = resp3.read()
            hdr3 = dict(resp3.getheaders())
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp3.status, 206)
            self._assert_coop_coep(hdr3)
            self.assertEqual(
                hdr3.get("Content-Range"),
                f"bytes {open_start}-{self.large_size - 1}/{self.large_size}",
            )
            self.assertEqual(body3, self.large_payload[open_start:])

            # 4. Full 3.5 MiB 200 OK GET on the same socket
            conn.request("GET", "/data/splash.wgsl.pack")
            resp4 = conn.getresponse()
            body4 = resp4.read()
            hdr4 = dict(resp4.getheaders())
            self.assertIs(conn.sock, first_sock)
            self.assertEqual(resp4.status, 200)
            self._assert_coop_coep(hdr4)
            self.assertEqual(hdr4.get("Content-Length"), str(self.large_size))
            self.assertEqual(
                hashlib.sha256(body4).hexdigest(),
                hashlib.sha256(self.large_payload).hexdigest(),
            )
        finally:
            conn.close()

    def test_concurrent_multi_mib_range_reads_thread_local_buffer_isolation(self) -> None:
        ranges = [
            (0, 1_572_863),
            (524_288, 2_097_151),
            (1_048_576, 2_621_439),
            (1_572_864, 3_145_727),
            (2_097_152, self.large_size - 1),
            (65_536, 2_228_223),
        ]

        def fetch_range(bounds: tuple[int, int]) -> tuple[int, int, str, str]:
            lo, hi = bounds
            st, hdr, body = self._request(
                "GET", "/data/splash.wgsl.pack", {"Range": f"bytes={lo}-{hi}"}
            )
            self.assertEqual(st, 206)
            self._assert_coop_coep(hdr)
            self.assertEqual(hdr.get("Content-Range"), f"bytes {lo}-{hi}/{self.large_size}")
            return (
                lo,
                hi,
                hashlib.sha256(body).hexdigest(),
                hashlib.sha256(self.large_payload[lo : hi + 1]).hexdigest(),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(fetch_range, ranges * 2))

        for lo, hi, actual_sha, expected_sha in results:
            self.assertEqual(actual_sha, expected_sha, msg=f"Range {lo}-{hi} hash mismatch")

    def test_custom_stream_chunk_bytes_and_fastpath_truncation(self) -> None:
        httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            stream_chunk_bytes=64 * 1024,
        )
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            lo, hi = 10_000, 300_000
            st, hdr, body = self._request(
                "GET",
                "/data/splash.wgsl.pack",
                {"Range": f"bytes={lo}-{hi}"},
                port=port,
            )
            self.assertEqual(st, 206)
            self._assert_coop_coep(hdr)
            self.assertEqual(body, self.large_payload[lo : hi + 1])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

        # Verify readinto fastpath marks close_connection on mid-stream truncation.
        class _TruncatingReadintoFile:
            def __init__(self) -> None:
                self._calls = 0

            def readinto(self, mv: memoryview) -> int:
                self._calls += 1
                if self._calls == 1:
                    data = b"ABCD"
                    mv[: len(data)] = data
                    return len(data)
                return 0

            def close(self) -> None:
                return None

        sent = bytearray()

        class _FakeSocketWriter:
            _sock = object()

            def write(self, data) -> int:
                sent.extend(data)
                return len(data)

        handler = object.__new__(serve_local.PlayGta6Handler)
        handler._stream_chunk_bytes = 16
        handler.close_connection = False
        handler.wfile = _FakeSocketWriter()
        handler._stream_file_range(_TruncatingReadintoFile(), length=32)
        self.assertEqual(bytes(sent), b"ABCD")
        self.assertTrue(handler.close_connection)

    def test_traversal_and_dump_endpoint_preserved(self) -> None:
        st_trav, hdr_trav, _ = self._request("GET", "/data/../../README.md")
        self.assertEqual(st_trav, 404)
        self._assert_coop_coep(hdr_trav)

        # Disabled dump endpoint still returns 403 JSON with COOP/COEP intact.
        st_dis, hdr_dis, body_dis = self._request(
            "POST", "/__pg6_dump/test.bin", body=b"abc"
        )
        self.assertEqual(st_dis, 403)
        self._assert_coop_coep(hdr_dis)
        self.assertFalse(json.loads(body_dis.decode("utf-8"))["ok"])

        # Enabled dump endpoint writes cleanly with COOP/COEP intact.
        httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            dump_root=self.dump_root,
        )
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            payload = b"DXBC_FASTPATH_CHECK"
            st_ok, hdr_ok, body_ok = self._request(
                "POST", "/__pg6_dump/sub/pso.bin", port=port, body=payload
            )
            self.assertEqual(st_ok, 200)
            self._assert_coop_coep(hdr_ok)
            self.assertEqual(json.loads(body_ok.decode("utf-8"))["bytes_written"], len(payload))
            self.assertEqual((self.dump_root / "sub" / "pso.bin").read_bytes(), payload)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

    def test_range_request_with_matching_if_none_match_or_ims_returns_206_not_304(self) -> None:
        st0, hdr0, _ = self._request("HEAD", "/data/splash.wgsl.pack")
        self.assertEqual(st0, 200)
        etag = hdr0.get("ETag")
        last_modified = hdr0.get("Last-Modified")
        self.assertTrue(etag)
        self.assertTrue(last_modified)

        start, end = 4096, 8191
        expected = self.large_payload[start : end + 1]
        for extra_headers in (
            {"Range": f"bytes={start}-{end}", "If-None-Match": etag},
            {"Range": f"bytes={start}-{end}", "If-Modified-Since": last_modified},
            {
                "Range": f"bytes={start}-{end}",
                "If-None-Match": etag,
                "If-Modified-Since": last_modified,
            },
        ):
            with self.subTest(headers=sorted(extra_headers.keys())):
                st_r, hdr_r, body_r = self._request("GET", "/data/splash.wgsl.pack", extra_headers)
                self.assertEqual(st_r, 206)
                self._assert_coop_coep(hdr_r)
                self.assertEqual(hdr_r.get("Content-Range"), f"bytes {start}-{end}/{self.large_size}")
                self.assertEqual(body_r, expected)

    def test_http11_keep_alive_sequential_range_get_head_404_status_and_conditionals(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            def send(method: str, path: str, headers: dict[str, str] | None = None):
                conn.request(method, path, headers=headers or {})
                resp = conn.getresponse()
                data = resp.read()
                return resp, dict(resp.getheaders()), data

            # 1. Range GET (206)
            resp1, hdr1, body1 = send("GET", "/data/splash.wgsl.pack", {"Range": "bytes=0-4095"})
            self.assertEqual(resp1.version, 11)
            self.assertEqual(resp1.status, 206)
            self.assertFalse(resp1.will_close)
            self._assert_coop_coep(hdr1)
            self.assertEqual(hdr1.get("Content-Length"), "4096")
            self.assertEqual(body1, self.large_payload[:4096])
            sock = conn.sock
            self.assertIsNotNone(sock)
            etag = hdr1["ETag"]

            # 2. Full GET (200) on the same socket
            resp2, hdr2, body2 = send("GET", "/index.html")
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp2.version, 11)
            self.assertEqual(resp2.status, 200)
            self.assertFalse(resp2.will_close)
            self._assert_coop_coep(hdr2)
            self.assertEqual(hdr2.get("Content-Length"), str(len(body2)))
            self.assertEqual(body2, b"<h1>playgta6web</h1>")

            # 3. HEAD (200, 0 body bytes) on the same socket
            resp3, hdr3, body3 = send("HEAD", "/runtime.wasm")
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp3.version, 11)
            self.assertEqual(resp3.status, 200)
            self.assertFalse(resp3.will_close)
            self._assert_coop_coep(hdr3)
            self.assertEqual(hdr3.get("Content-Length"), "8")
            self.assertEqual(body3, b"")

            # 4. 404 Not Found (Content-Length: 0) on the same socket
            resp4, hdr4, body4 = send("GET", "/data/missing-archive.rpf")
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp4.version, 11)
            self.assertEqual(resp4.status, 404)
            self.assertFalse(resp4.will_close)
            self._assert_coop_coep(hdr4)
            self.assertEqual(hdr4.get("Content-Length"), "0")
            self.assertEqual(body4, b"")

            # 5. /api/status JSON (200) on the same socket
            resp5, hdr5, body5 = send("GET", "/api/status")
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp5.version, 11)
            self.assertEqual(resp5.status, 200)
            self.assertFalse(resp5.will_close)
            self._assert_coop_coep(hdr5)
            self.assertEqual(hdr5.get("Content-Length"), str(len(body5)))
            self.assertTrue(json.loads(body5.decode("utf-8"))["ok"])

            # 6. 412 Precondition Failed + 304 Not Modified + 416 + OPTIONS still keep socket alive
            resp6, hdr6, body6 = send(
                "GET",
                "/data/splash.wgsl.pack",
                {"Range": "bytes=0-15", "If-Match": '"stale-etag"'},
            )
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp6.status, 412)
            self.assertFalse(resp6.will_close)
            self._assert_coop_coep(hdr6)
            self.assertEqual(hdr6.get("Content-Length"), "0")
            self.assertEqual(hdr6.get("ETag"), etag)
            self.assertEqual(body6, b"")

            resp7, hdr7, body7 = send(
                "GET",
                "/data/splash.wgsl.pack",
                {"If-None-Match": etag},
            )
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp7.status, 304)
            self.assertFalse(resp7.will_close)
            self._assert_coop_coep(hdr7)
            self.assertEqual(body7, b"")

            resp8, hdr8, body8 = send(
                "GET",
                "/data/splash.wgsl.pack",
                {"Range": f"bytes={self.large_size + 100}-"},
            )
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp8.status, 416)
            self.assertFalse(resp8.will_close)
            self.assertEqual(hdr8.get("Content-Length"), "0")
            self.assertEqual(body8, b"")

            resp9, hdr9, body9 = send("OPTIONS", "/data/splash.wgsl.pack")
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp9.status, 204)
            self.assertFalse(resp9.will_close)
            self._assert_coop_coep(hdr9)
            self.assertEqual(body9, b"")

            # Final Range GET proves keep-alive framing stayed synchronized throughout.
            resp10, hdr10, body10 = send(
                "GET",
                "/data/splash.wgsl.pack",
                {"Range": "bytes=4096-8191", "If-Match": etag},
            )
            self.assertIs(conn.sock, sock)
            self.assertEqual(resp10.status, 206)
            self.assertFalse(resp10.will_close)
            self.assertEqual(body10, self.large_payload[4096:8192])
        finally:
            conn.close()

    def test_etag_if_match_and_if_range_validation(self) -> None:
        pack_path = self.data_root / "tiny.wgsl.pack"
        v1 = b"WGPK0123456789abcdef"
        pack_path.write_bytes(v1)
        expected_etag_v1 = serve_local.compute_etag(pack_path.stat())

        st_head, hdr_head, body_head = self._request("HEAD", "/data/tiny.wgsl.pack")
        self.assertEqual(st_head, 200)
        self._assert_coop_coep(hdr_head)
        self.assertEqual(hdr_head.get("ETag"), expected_etag_v1)
        self.assertEqual(body_head, b"")

        # If-Match matching ETag / wildcard / list -> 206 Partial Content
        for if_match in (expected_etag_v1, "*", f'"other-tag", {expected_etag_v1}'):
            with self.subTest(if_match=if_match):
                st, hdr, body = self._request(
                    "GET",
                    "/data/tiny.wgsl.pack",
                    {"Range": "bytes=4-11", "If-Match": if_match},
                )
                self.assertEqual(st, 206)
                self._assert_coop_coep(hdr)
                self.assertEqual(hdr.get("ETag"), expected_etag_v1)
                self.assertEqual(hdr.get("Content-Range"), f"bytes 4-11/{len(v1)}")
                self.assertEqual(body, v1[4:12])

        # If-Match mismatch or weak ETag -> 412 Precondition Failed with Content-Length: 0
        for method in ("GET", "HEAD"):
            for bad_if_match in ('"wrong-etag"', f"W/{expected_etag_v1}", ""):
                with self.subTest(method=method, bad_if_match=bad_if_match):
                    st, hdr, body = self._request(
                        method,
                        "/data/tiny.wgsl.pack",
                        {"Range": "bytes=4-11", "If-Match": bad_if_match},
                    )
                    self.assertEqual(st, 412)
                    self._assert_coop_coep(hdr)
                    self.assertEqual(hdr.get("Content-Length"), "0")
                    self.assertEqual(hdr.get("ETag"), expected_etag_v1)
                    self.assertEqual(body, b"")

        # If-Range matching strong ETag -> 206; mismatched or weak ETag -> 200 full file
        st_ir_ok, hdr_ir_ok, body_ir_ok = self._request(
            "GET",
            "/data/tiny.wgsl.pack",
            {"Range": "bytes=4-11", "If-Range": expected_etag_v1},
        )
        self.assertEqual(st_ir_ok, 206)
        self.assertEqual(hdr_ir_ok.get("Content-Range"), f"bytes 4-11/{len(v1)}")
        self.assertEqual(body_ir_ok, v1[4:12])

        for stale_if_range in ('"stale-etag"', f"W/{expected_etag_v1}"):
            with self.subTest(stale_if_range=stale_if_range):
                st_ir_miss, hdr_ir_miss, body_ir_miss = self._request(
                    "GET",
                    "/data/tiny.wgsl.pack",
                    {"Range": "bytes=4-11", "If-Range": stale_if_range},
                )
                self.assertEqual(st_ir_miss, 200)
                self._assert_coop_coep(hdr_ir_miss)
                self.assertIsNone(hdr_ir_miss.get("Content-Range"))
                self.assertEqual(hdr_ir_miss.get("Content-Length"), str(len(v1)))
                self.assertEqual(body_ir_miss, v1)

        # Updating file on disk immediately turns previous ETag stale for both If-Match and If-Range.
        v2 = b"WGPK-UPDATED-PAYLOAD-0123456789"
        pack_path.write_bytes(v2)
        expected_etag_v2 = serve_local.compute_etag(pack_path.stat())
        self.assertNotEqual(expected_etag_v2, expected_etag_v1)

        st_412, hdr_412, body_412 = self._request(
            "GET",
            "/data/tiny.wgsl.pack",
            {"Range": "bytes=0-7", "If-Match": expected_etag_v1},
        )
        self.assertEqual(st_412, 412)
        self.assertEqual(hdr_412.get("ETag"), expected_etag_v2)
        self.assertEqual(hdr_412.get("Content-Length"), "0")
        self.assertEqual(body_412, b"")

        st_200, hdr_200, body_200 = self._request(
            "GET",
            "/data/tiny.wgsl.pack",
            {"Range": "bytes=0-7", "If-Range": expected_etag_v1},
        )
        self.assertEqual(st_200, 200)
        self.assertEqual(hdr_200.get("ETag"), expected_etag_v2)
        self.assertEqual(body_200, v2)

    def test_concurrent_8_thread_rpf_range_reads_and_disk_update_invalidation(self) -> None:
        rpf_path = self.data_root / "common.rpf"
        payload_v1 = _make_deterministic_bytes(256 * 1024)
        rpf_path.write_bytes(payload_v1)

        slices = [(i * 4096, i * 4096 + 8191) for i in range(32)]

        def fetch_slice(bounds: tuple[int, int]) -> tuple[int, int, bytes, str]:
            lo, hi = bounds
            st, hdr, body = self._request(
                "GET",
                "/data/common.rpf",
                {"Range": f"bytes={lo}-{hi}"},
            )
            self.assertEqual(st, 206)
            self._assert_coop_coep(hdr)
            return lo, hi, body, hdr["ETag"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results_v1 = list(pool.map(fetch_slice, slices))

        etag_v1 = serve_local.compute_etag(rpf_path.stat())
        for lo, hi, body, etag in results_v1:
            self.assertEqual(etag, etag_v1)
            self.assertEqual(body, payload_v1[lo : hi + 1])

        cache = self.httpd.file_handle_cache
        self.assertIsNotNone(cache)
        assert cache is not None
        stats_v1 = cache.stats()
        self.assertGreater(stats_v1["hits"], 0)
        self.assertGreaterEqual(stats_v1["entries"], 1)

        # Update file on disk (different size) -> cached handle invalidates cleanly.
        payload_v2 = bytes(((b ^ 0x5A) for b in _make_deterministic_bytes(320 * 1024)))
        rpf_path.write_bytes(payload_v2)
        etag_v2 = serve_local.compute_etag(rpf_path.stat())
        self.assertNotEqual(etag_v2, etag_v1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results_v2 = list(pool.map(fetch_slice, slices))

        for lo, hi, body, etag in results_v2:
            self.assertEqual(etag, etag_v2)
            self.assertEqual(body, payload_v2[lo : hi + 1])

        stats_v2 = cache.stats()
        self.assertGreaterEqual(stats_v2["invalidations"], 1)

        # Same-size overwrite with bumped mtime_ns also invalidates cached handle.
        prev_st = rpf_path.stat()
        payload_v3 = bytes(((b ^ 0xA5) for b in payload_v2))
        self.assertEqual(len(payload_v3), len(payload_v2))
        rpf_path.write_bytes(payload_v3)
        os.utime(rpf_path, ns=(prev_st.st_atime_ns, prev_st.st_mtime_ns + 5_000_000_000))
        etag_v3 = serve_local.compute_etag(rpf_path.stat())
        self.assertNotEqual(etag_v3, etag_v2)

        st_v3, hdr_v3, body_v3 = self._request(
            "GET",
            "/data/common.rpf",
            {"Range": "bytes=0-16383", "If-Match": etag_v3},
        )
        self.assertEqual(st_v3, 206)
        self.assertEqual(hdr_v3.get("ETag"), etag_v3)
        self.assertEqual(body_v3, payload_v3[:16384])
        self.assertGreaterEqual(cache.stats()["invalidations"], stats_v2["invalidations"] + 1)

        # Deleting the cached file works cleanly (including on Windows) and yields 404.
        rpf_path.unlink()
        st_404, _, body_404 = self._request("GET", "/data/common.rpf")
        self.assertEqual(st_404, 404)
        self.assertEqual(body_404, b"")

    def test_file_handle_cache_bounded_lru_eviction_and_shutdown_cleanup(self) -> None:
        with self.assertRaises(ValueError):
            serve_local.create_server(
                "127.0.0.1",
                0,
                self.web_root,
                self.data_root,
                file_handle_cache_size=-1,
            )

        files = {
            "a.rpf": b"RPF-AAAA-0123456789",
            "b.pack": b"PACK-BBBB-0123456789",
            "c.pgim": b"PGIM-CCCC-0123456789",
            "d.wasm": b"\x00asm\x01\x00\x00\x00DDDD",
        }
        for name, data in files.items():
            (self.data_root / name).write_bytes(data)

        httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            file_handle_cache_size=2,
        )
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        cache = httpd.file_handle_cache
        self.assertIsNotNone(cache)
        assert cache is not None
        try:
            # Non-cacheable .html file should never enter the FD cache.
            st_html, _, body_html = self._request("GET", "/index.html", port=port)
            self.assertEqual(st_html, 200)
            self.assertEqual(body_html, b"<h1>playgta6web</h1>")
            self.assertEqual(len(cache), 0)
            self.assertEqual(cache.stats()["open_handles"], 0)

            for name, expected in files.items():
                st, _, body = self._request(
                    "GET",
                    f"/data/{name}",
                    {"Range": "bytes=0-7"},
                    port=port,
                )
                self.assertEqual(st, 206)
                self.assertEqual(body, expected[:8])
                self.assertLessEqual(len(cache), 2)
                self.assertLessEqual(cache.stats()["open_handles"], 2)

            stats = cache.stats()
            self.assertEqual(stats["entries"], 2)
            self.assertEqual(stats["open_handles"], 2)
            self.assertGreaterEqual(stats["evictions"], 2)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

        # Shutdown / server_close must close every cached handle with zero FD leaks.
        final_stats = cache.stats()
        self.assertEqual(final_stats["entries"], 0)
        self.assertEqual(final_stats["open_handles"], 0)


if __name__ == "__main__":
    unittest.main()
