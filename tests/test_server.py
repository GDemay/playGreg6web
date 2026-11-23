import http.client
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

    def test_coop_coep_and_wasm_mime(self) -> None:
        status, headers, body = self._request("GET", "/demo.wasm")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Cross-Origin-Opener-Policy"), "same-origin")
        self.assertEqual(headers.get("Cross-Origin-Embedder-Policy"), "require-corp")
        self.assertEqual(headers.get("Content-Type"), "application/wasm")
        self.assertEqual(body, b"\x00asm\x01\x00\x00\x00")

    def test_http_range_partial_content(self) -> None:
        status, headers, body = self._request("GET", "/data/sample.bin", {"Range": "bytes=4-9"})
        self.assertEqual(status, 206)
        self.assertEqual(headers.get("Content-Range"), "bytes 4-9/16")
        self.assertEqual(headers.get("Content-Length"), "6")
        self.assertEqual(body, b"456789")

    def test_invalid_range_returns_416(self) -> None:
        status, headers, _ = self._request("GET", "/data/sample.bin", {"Range": "bytes=100-200"})
        self.assertEqual(status, 416)
        self.assertEqual(headers.get("Content-Range"), "bytes */16")

    def test_path_traversal_blocked(self) -> None:
        status, _, _ = self._request("GET", "/data/../../README.md")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
