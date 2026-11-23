"""Tests for W43 local splash / shader / PGIM boot-bundle endpoints on serve_local.py."""

from __future__ import annotations

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

SYNTHETIC_CONFIG = """<?xml version="1.0" encoding="utf-8"?>
<Game configVersion="1">
  <Identity Name="Synthetic.BootTitle" Publisher="CN=Synthetic" Version="1.0.0.0" />
  <TitleId>AABBCCDD</TitleId>
  <ExecutableList>
    <Executable Name="Game.exe" Id="Game" TargetDeviceFamily="Scarlett" />
  </ExecutableList>
  <ShellVisuals DefaultDisplayName="Synthetic Boot"
                SplashScreenImage="Assets\\xbs_splash.png"
                BackgroundColor="#000000" />
</Game>
"""

XXE_CONFIG = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<Game configVersion="1">
  <ShellVisuals SplashScreenImage="Assets\\xbs_splash.png" />
</Game>
"""


class TestServeLocalBoot(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.web_root = base / "web"
        self.data_root = base / "data"
        self.game_root = base / "game"
        self.shaders_root = self.data_root / "shaders"
        self.boot_root = base / "work" / "recomp_build"

        self.web_root.mkdir()
        self.data_root.mkdir()
        self.game_root.mkdir()
        self.shaders_root.mkdir(parents=True)
        self.boot_root.mkdir(parents=True)

        (self.web_root / "index.html").write_text("<h1>boot</h1>", encoding="utf-8")
        (self.data_root / "data-manifest.json").write_text("{}", encoding="utf-8")
        (self.data_root / "bootset.json").write_text("[]", encoding="utf-8")

        (self.game_root / "MicrosoftGame.Config").write_text(SYNTHETIC_CONFIG, encoding="utf-8")
        assets = self.game_root / "Assets"
        assets.mkdir()
        self.splash_bytes = b"\x89PNG\r\n\x1a\n" + b"splash-bytes-0123456789"
        (assets / "xbs_splash.png").write_bytes(self.splash_bytes)

        self.pack_bytes = b"WGPK" + b"pack-payload-0123456789abcdef"
        (self.shaders_root / "other.wgsl.pack").write_bytes(b"WGPK" + b"other")
        (self.shaders_root / "fallback.wgsl.pack").write_bytes(self.pack_bytes)
        (self.shaders_root / "reflection-manifest.json").write_text(
            '{"schema_version":6}', encoding="utf-8"
        )

        self.pgim_bytes = b"PGIM" + b"guest-image-0123456789abcdef"
        (self.boot_root / "guest.pgim").write_bytes(self.pgim_bytes)
        (self.boot_root / "meta.json").write_text('{"ok":true}', encoding="utf-8")
        (self.boot_root / "helper.js").write_text("export default 1;\n", encoding="utf-8")
        (self.boot_root / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
        (self.boot_root / "evil.exe").write_bytes(b"MZ-disallowed")
        (self.boot_root / "lifted.ll").write_text("; disallowed\n", encoding="utf-8")
        (self.boot_root / ".env").write_text("SECRET=1\n", encoding="utf-8")

        # Outside roots used for traversal / symlink escape attempts.
        self.secret_outside = base / "outside_secret.bin"
        self.secret_outside.write_bytes(b"TOP-SECRET-BYTES")

        self.httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            self.web_root,
            self.data_root,
            game_root=self.game_root,
            shaders_root=self.shaders_root,
            boot_root=self.boot_root,
        )
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

    def _assert_no_host_leak(self, raw_text: str) -> None:
        sensitive = [
            str(self.web_root),
            str(self.data_root),
            str(self.game_root),
            str(self.boot_root),
            str(Path(self.tmp.name)),
            str(REPO_ROOT),
            os.environ.get("USERNAME", ""),
            os.environ.get("USER", ""),
            os.environ.get("LOGNAME", ""),
        ]
        for token in sensitive:
            if token and len(token) > 2:
                self.assertNotIn(token, raw_text)
                self.assertNotIn(token.replace("\\", "/"), raw_text)
        self.assertNotRegex(raw_text, r"[A-Za-z]:\\")
        self.assertNotRegex(raw_text, r"/(?:Users|home|tmp|var|opt|root)/")

    def test_api_status_discovers_boot_urls_without_host_paths(self) -> None:
        status, headers, body = self._request("GET", "/api/status")
        self.assertEqual(status, 200)
        self._assert_isolation_headers(headers)
        self.assertEqual(headers.get("Content-Type"), "application/json")
        self.assertEqual(headers.get("Cache-Control"), "no-store")

        raw_text = body.decode("utf-8")
        data = json.loads(raw_text)

        # Legacy fields remain intact.
        for key in (
            "ok",
            "coopCoep",
            "dataDirConfigured",
            "hasManifest",
            "hasBootset",
            "hasWasmBinary",
            "shaderPackCount",
        ):
            self.assertIn(key, data)
        self.assertTrue(data["ok"])
        self.assertTrue(data["coopCoep"])
        self.assertTrue(data["dataDirConfigured"])
        self.assertTrue(data["hasManifest"])
        self.assertTrue(data["hasBootset"])
        self.assertGreaterEqual(data["shaderPackCount"], 1)

        # New boot-discovery fields (relative URLs only).
        self.assertEqual(data["splashImage"], "/game/Assets/xbs_splash.png")
        self.assertEqual(data["shaderPackUrl"], "/shaders/fallback.wgsl.pack")
        self.assertEqual(data["reflectionManifestUrl"], "/shaders/reflection-manifest.json")
        self.assertEqual(data["pgimUrl"], "/boot/guest.pgim")

        self._assert_no_host_leak(raw_text)

    def test_api_status_null_when_assets_missing(self) -> None:
        # Empty synthetic dirs -> null discovery fields, legacy keys still present.
        empty = Path(self.tmp.name) / "empty"
        web = empty / "web"
        data = empty / "data"
        shaders = data / "shaders"
        boot = empty / "boot"
        game = empty / "game"
        for path in (web, data, shaders, boot, game):
            path.mkdir(parents=True)
        (web / "index.html").write_text("x", encoding="utf-8")

        httpd = serve_local.create_server(
            "127.0.0.1",
            0,
            web,
            data,
            game_root=game,
            shaders_root=shaders,
            boot_root=boot,
        )
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                conn.request("GET", "/api/status")
                resp = conn.getresponse()
                raw = resp.read().decode("utf-8")
                self.assertEqual(resp.status, 200)
            finally:
                conn.close()
            data = json.loads(raw)
            self.assertIsNone(data["splashImage"])
            self.assertIsNone(data["shaderPackUrl"])
            self.assertIsNone(data["reflectionManifestUrl"])
            self.assertIsNone(data["pgimUrl"])
            self._assert_no_host_leak(raw)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

    def test_xxe_config_does_not_set_splash_image(self) -> None:
        (self.game_root / "MicrosoftGame.Config").write_text(XXE_CONFIG, encoding="utf-8")
        status, _, body = self._request("GET", "/api/status")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIsNone(data["splashImage"])

    def test_shaders_and_boot_serve_200_206_etag_coop_coep(self) -> None:
        for path, expected in (
            ("/shaders/fallback.wgsl.pack", self.pack_bytes),
            ("/boot/guest.pgim", self.pgim_bytes),
            ("/game/Assets/xbs_splash.png", self.splash_bytes),
        ):
            with self.subTest(path=path, mode="full"):
                status, headers, body = self._request("GET", path)
                self.assertEqual(status, 200)
                self._assert_isolation_headers(headers)
                self.assertTrue(headers.get("ETag"))
                self.assertTrue(headers.get("Last-Modified"))
                self.assertEqual(body, expected)

            with self.subTest(path=path, mode="range"):
                status, headers, body = self._request("GET", path, {"Range": "bytes=2-7"})
                self.assertEqual(status, 206)
                self._assert_isolation_headers(headers)
                self.assertTrue(headers.get("ETag"))
                self.assertEqual(headers.get("Content-Range"), f"bytes 2-7/{len(expected)}")
                self.assertEqual(body, expected[2:8])

            with self.subTest(path=path, mode="head"):
                status, headers, body = self._request("HEAD", path)
                self.assertEqual(status, 200)
                self._assert_isolation_headers(headers)
                self.assertTrue(headers.get("ETag"))
                self.assertEqual(body, b"")

    def test_boot_blocks_disallowed_extensions(self) -> None:
        for path in ("/boot/evil.exe", "/boot/lifted.ll", "/boot/.env"):
            with self.subTest(path=path):
                status, headers, body = self._request("GET", path)
                self.assertIn(status, (403, 404))
                self._assert_isolation_headers(headers)
                self.assertEqual(body, b"")
                # Disallowed extension that resolves inside boot root must be 403.
                if path in ("/boot/evil.exe", "/boot/lifted.ll"):
                    self.assertEqual(status, 403)

    def test_boot_and_shaders_block_path_traversal(self) -> None:
        outside_name = self.secret_outside.name
        parent_attempts = [
            f"/boot/../{outside_name}",
            "/boot/../../README.md",
            "/boot/%2e%2e/" + outside_name,
            f"/shaders/../{outside_name}",
            "/shaders/../../README.md",
            f"/game/../{outside_name}",
            "/data/../../README.md",
        ]
        for path in parent_attempts:
            with self.subTest(path=path):
                status, headers, body = self._request("GET", path)
                self.assertEqual(status, 404)
                self._assert_isolation_headers(headers)
                self.assertEqual(body, b"")
                self.assertNotIn(b"TOP-SECRET", body)

    def test_boot_allows_safe_extensions(self) -> None:
        for path in ("/boot/meta.json", "/boot/helper.js", "/boot/runtime.wasm"):
            with self.subTest(path=path):
                status, headers, body = self._request("GET", path)
                self.assertEqual(status, 200)
                self._assert_isolation_headers(headers)
                self.assertTrue(headers.get("ETag"))
                self.assertGreater(len(body), 0)

    def test_legacy_routes_still_work(self) -> None:
        status, headers, body = self._request("GET", "/")
        self.assertEqual(status, 200)
        self._assert_isolation_headers(headers)
        self.assertIn(b"boot", body)

        status, headers, body = self._request("GET", "/data/data-manifest.json")
        self.assertEqual(status, 200)
        self._assert_isolation_headers(headers)
        self.assertEqual(body, b"{}")


class TestServeLocalBootSymlink(unittest.TestCase):
    def test_boot_rejects_external_symlink(self) -> None:
        if os.name == "nt":
            # Creating symlinks on Windows often requires elevated privileges.
            self.skipTest("symlink escape coverage is POSIX-oriented")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web = base / "web"
            data = base / "data"
            boot = base / "boot"
            web.mkdir()
            data.mkdir()
            boot.mkdir()
            (web / "index.html").write_text("x", encoding="utf-8")

            outside = base / "outside.env"
            outside.write_text("SECRET=1\n", encoding="utf-8")
            link = boot / "linked.env"
            try:
                link.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"unable to create symlink: {exc}")

            httpd = serve_local.create_server(
                "127.0.0.1",
                0,
                web,
                data,
                boot_root=boot,
            )
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                try:
                    conn.request("GET", "/boot/linked.env")
                    resp = conn.getresponse()
                    body = resp.read()
                    status = resp.status
                finally:
                    conn.close()
                self.assertIn(status, (403, 404))
                self.assertNotIn(b"SECRET", body)
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
