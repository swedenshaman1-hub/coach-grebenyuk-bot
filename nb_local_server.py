"""Authenticated localhost NotebookLM gateway for optional proxy mode."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from notebook_registry import load_registry


SECRET = os.environ.get("NOTEBOOKLM_LOCAL_SECRET", "").strip()
HOST = os.environ.get("NOTEBOOKLM_LOCAL_HOST", "127.0.0.1").strip()
MAX_BODY_BYTES = int(os.environ.get("NOTEBOOKLM_PROXY_MAX_BODY", "65536"))
RATE_LIMIT_PER_MINUTE = int(os.environ.get("NOTEBOOKLM_PROXY_RATE_LIMIT", "60"))
_slots = threading.BoundedSemaphore(2)
_source_cache: dict[str, list[dict]] = {}
_calls: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = threading.Lock()


def _allowed_notebooks() -> set[str]:
    registry = load_registry()
    return {
        notebook.uuid
        for collection in registry.collections.values()
        for notebook in collection.notebooks
    }


ALLOWED_NOTEBOOKS = _allowed_notebooks()


def _source_catalog(client, notebook_id: str) -> list[dict]:
    raw_notebook = client.get_notebook(notebook_id)
    notebook = (
        raw_notebook[0]
        if raw_notebook and isinstance(raw_notebook[0], list)
        else raw_notebook
    )
    raw_sources = (
        notebook[1]
        if notebook and len(notebook) > 1 and isinstance(notebook[1], list)
        else []
    )
    sources: list[dict] = []
    for source in raw_sources:
        if not isinstance(source, list) or not source:
            continue
        wrapper = source[0] if isinstance(source[0], list) else []
        source_id = wrapper[0] if wrapper and isinstance(wrapper[0], str) else ""
        title = source[1] if len(source) > 1 and isinstance(source[1], str) else ""
        metadata = source[2] if len(source) > 2 and isinstance(source[2], list) else []
        url = None
        if len(metadata) > 7 and isinstance(metadata[7], list) and metadata[7]:
            url = metadata[7][0]
        if source_id and title:
            sources.append({"id": source_id, "title": title, "url": url})
    return sources


def _rate_limited(ip: str) -> bool:
    now = time.monotonic()
    with _rate_lock:
        calls = _calls[ip]
        while calls and now - calls[0] > 60:
            calls.popleft()
        if len(calls) >= RATE_LIMIT_PER_MINUTE:
            return True
        calls.append(now)
        return False


class Handler(BaseHTTPRequestHandler):
    server_version = "NotebookLMStrictGateway/3"

    def do_POST(self):
        if self.path != "/ask":
            self.send_error(404)
            return
        if self.headers.get("X-Secret") != SECRET:
            self.send_error(403)
            return
        if _rate_limited(self.client_address[0]):
            self.send_error(429)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self.send_error(400)
            return
        if length < 1 or length > MAX_BODY_BYTES:
            self.send_error(413)
            return
        try:
            body = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400)
            return

        notebook_id = str(body.get("notebook_id") or "").strip()
        query = str(body.get("query") or "").strip()
        conversation_id = str(body.get("conversation_id") or "").strip() or None
        sources_only = bool(body.get("sources_only"))
        if notebook_id not in ALLOWED_NOTEBOOKS:
            self._json({"ok": False, "error": "notebook is not allowed"})
            return
        if not query and not sources_only:
            self._json({"ok": False, "error": "empty query"})
            return

        try:
            from notebooklm_mcp_2026 import server
            from notebooklm_mcp_2026.tools.query import query_notebook

            with _slots:
                sources = _source_cache.get(notebook_id) or _source_catalog(
                    server.get_client(), notebook_id
                )
                if not sources:
                    self._json({"ok": False, "error": "NotebookLM source list is empty"})
                    return
                _source_cache[notebook_id] = sources
                if sources_only:
                    result = {"status": "success", "sources": sources}
                else:
                    result = query_notebook(
                        notebook_id=notebook_id,
                        query=query,
                        conversation_id=conversation_id,
                    )
                    result["sources"] = sources
            self._json({"ok": result.get("status") == "success", "result": result})
        except Exception as exc:
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _json(self, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Never log request bodies, auth values or user voice/text content.
        print(f"[NB-Server] {self.client_address[0]} {fmt % args}", flush=True)


if __name__ == "__main__":
    if not SECRET:
        raise SystemExit("NOTEBOOKLM_LOCAL_SECRET is required")
    port = int(os.environ.get("PORT", "8766"))
    print(f"Strict NotebookLM proxy listening on {HOST}:{port}", flush=True)
    ThreadingHTTPServer((HOST, port), Handler).serve_forever()
