#!/usr/bin/env python3
"""Local dev & asset-streaming HTTP server for playgta6web.

- Serves `web/` at `/` and `--data` (default `local-data/`) at `/data/`.
- Sends Cross-Origin Isolation headers (COOP/COEP) required for SharedArrayBuffer.
- Supports HTTP `Range: bytes=start-end` (`206 Partial Content`) for streaming large files.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import posixpath
import re
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/json", ".json")


def safe_resolve(root: Path, rel_url_path: str) -> Path | None:
    """Resolve a URL path strictly inside `root`, rejecting traversal."""
    normalized = posixpath.normpath("/" + unquote(rel_url_path)).lstrip("/")
    if not normalized or normalized.startswith(".."):
        return None
    candidate = (root / normalized).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


def parse_byte_range(header: str, file_size: int) -> tuple[int, int] | None:
    """Parse a single HTTP byte range against `file_size`."""
    match = RANGE_RE.match(header.strip())
    if not match or file_size <= 0:
        return None
    start_s, end_s = match.groups()
    if not start_s and not end_s:
        return None
    if start_s and end_s:
        start, end = int(start_s), min(int(end_s), file_size - 1)
    elif start_s:
        start, end = int(start_s), file_size - 1
    else:
        suffix = int(end_s)
        if suffix <= 0:
            return None
        start, end = max(file_size - suffix, 0), file_size - 1
    if start > end or start >= file_size:
        return None
    return start, end


class PlayGta6Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, web_root: Path, data_root: Path, **kwargs):
        self.web_root = web_root
        self.data_root = data_root
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _resolve_target(self) -> Path | None:
        url_path = urlsplit(self.path).path
        if url_path == "/":
            return (self.web_root / "index.html").resolve()
        if url_path.startswith("/data/"):
            return safe_resolve(self.data_root, url_path[len("/data/"):])
        return safe_resolve(self.web_root, url_path.lstrip("/"))

    def _serve_file(self, send_body: bool) -> None:
        target = self._resolve_target()
        if target is None or not target.is_file():
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        file_size = target.stat().st_size
        content_type, _ = mimetypes.guess_type(str(target))
        content_type = content_type or "application/octet-stream"

        range_header = self.headers.get("Range")
        start, end = 0, max(file_size - 1, 0)
        status = HTTPStatus.OK

        if range_header:
            parsed = parse_byte_range(range_header, file_size)
            if parsed is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = parsed
            status = HTTPStatus.PARTIAL_CONTENT

        length = (end - start + 1) if file_size > 0 else 0
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        if not send_body or length == 0:
            return

        with target.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:
        self._serve_file(send_body=True)

    def do_HEAD(self) -> None:
        self._serve_file(send_body=False)


def create_server(host: str, port: int, web_root: Path, data_root: Path) -> ThreadingHTTPServer:
    handler = partial(PlayGta6Handler, web_root=web_root, data_root=data_root)
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Local COOP/COEP + HTTP Range server for playgta6web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--web", type=Path, default=repo_root / "web")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("GTA6_DATA_DIR", repo_root / "local-data")),
    )
    args = parser.parse_args()

    args.data.mkdir(parents=True, exist_ok=True)
    httpd = create_server(args.host, args.port, args.web, args.data)
    print(f"Serving web={args.web} and /data={args.data} at http://{args.host}:{args.port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
