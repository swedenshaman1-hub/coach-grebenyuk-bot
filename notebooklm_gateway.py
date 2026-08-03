"""Single strict gateway to NotebookLM.

No Telegram handler imports ``notebooklm-mcp-2026`` directly.  This module owns
authentication restoration, source discovery, error classification, bounded
retries and the optional authenticated localhost proxy.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from platformdirs import user_data_dir

from strict_contract import ErrorType, SourceInfo


logger = logging.getLogger(__name__)


@dataclass
class GatewayResponse:
    ok: bool
    raw_answer: str = ""
    conversation_id: str = ""
    sources: list[SourceInfo] = field(default_factory=list)
    source_fingerprint: str = ""
    attempts: int = 0
    error_type: ErrorType = ErrorType.NONE
    error: str = ""
    elapsed_seconds: float = 0.0


def classify_error(error: str) -> ErrorType:
    value = str(error).lower()
    if any(marker in value for marker in (
        "401", "403", "not authenticated", "authentication expired",
        "cookies expired", "rpc error 16", "accounts.google.com",
    )):
        return ErrorType.AUTH
    if "429" in value or "too many requests" in value or "rate limit" in value:
        return ErrorType.RATE_LIMIT
    if "timeout" in value or "timed out" in value:
        return ErrorType.TIMEOUT
    if any(marker in value for marker in ("500", "502", "503", "504")):
        return ErrorType.SERVER
    if any(marker in value for marker in (
        "connection reset", "connection refused", "temporary failure",
        "name resolution", "urlopen error", "network",
    )):
        return ErrorType.NETWORK
    if "not configured" in value or "source list is empty" in value:
        return ErrorType.CONFIGURATION
    return ErrorType.UNKNOWN


class NotebookLMGateway:
    def __init__(self):
        self.local_url = os.getenv("NOTEBOOKLM_LOCAL_URL", "").strip().rstrip("/")
        self.local_secret = os.getenv("NOTEBOOKLM_LOCAL_SECRET", "").strip()
        # Match notebooklm-mcp-2026 locally; Railway overrides this path.
        self.data_dir = os.getenv("NOTEBOOKLM_MCP_DATA_DIR", "").strip() or str(
            Path(user_data_dir("notebooklm-mcp-2026"))
        )
        self.base_url = os.getenv(
            "NOTEBOOKLM_BASE_URL", "https://notebook.google.com"
        ).strip().rstrip("/")
        self.auth_data = self._load_auth_data()
        self._source_cache: dict[str, list[SourceInfo]] = {}
        self._slots = threading.BoundedSemaphore(
            max(1, int(os.getenv("NOTEBOOKLM_MAX_PARALLEL", "2")))
        )
        self.deadline_seconds = min(
            # 254-source notebooks regularly need 70-100 seconds.  Never let a
            # stale Railway value reintroduce the old 85-second false timeout.
            150, max(110, int(os.getenv("NOTEBOOKLM_REQUEST_DEADLINE", "110")))
        )
        self.max_attempts = min(
            3, max(1, int(os.getenv("NOTEBOOKLM_MAX_ATTEMPTS", "2")))
        )
        if self.local_url and not self.local_secret:
            raise RuntimeError("NOTEBOOKLM_LOCAL_SECRET is required for proxy mode")
        if self.auth_data:
            self._refresh_and_persist_auth()

    def _load_auth_data(self) -> dict:
        raw = os.getenv("NOTEBOOKLM_AUTH_JSON", "").strip()
        encoded = os.getenv("NOTEBOOKLM_AUTH_JSON_B64", "").strip()
        if encoded:
            try:
                raw = base64.b64decode(encoded).decode("utf-8")
            except Exception as exc:
                logger.error("NotebookLM auth base64 is invalid: %s", type(exc).__name__)
                return {}
        if raw:
            try:
                data = json.loads(raw)
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                logger.error("NotebookLM auth JSON is invalid")
                return {}
        if self.data_dir:
            path = os.path.join(self.data_dir, "auth.json")
            try:
                with open(path, "r", encoding="utf-8") as source:
                    data = json.load(source)
                return data if isinstance(data, dict) else {}
            except (OSError, json.JSONDecodeError):
                pass
        return {}

    def _refresh_and_persist_auth(self) -> None:
        """Refresh CSRF/build label without logging any secret material."""
        try:
            import httpx

            jar = httpx.Cookies()
            notebook_host = urlparse(self.base_url).hostname or "notebook.google.com"
            for key, value in (self.auth_data.get("cookies") or {}).items():
                domain = notebook_host if key in {"OSID", "__Secure-OSID"} else ".google.com"
                jar.set(key, value, domain=domain)
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            with httpx.Client(
                cookies=jar, headers=headers, follow_redirects=True, timeout=20.0
            ) as client:
                page = client.get(f"{self.base_url}/")
            if page.status_code == 200 and "accounts.google.com" not in str(page.url):
                csrf = re.search(r'"SNlM0e":"([^"]+)"', page.text)
                session = re.search(r'"FdrFJe":"(\d+)"', page.text)
                build = re.search(r"boq_labs-tailwind-frontend_[\w.]+", page.text)
                if csrf:
                    self.auth_data["csrf_token"] = csrf.group(1)
                if session:
                    self.auth_data["session_id"] = session.group(1)
                if build:
                    os.environ["NOTEBOOKLM_BL"] = build.group(0).rstrip(".")
                logger.info("NotebookLM startup authentication refresh succeeded")
            else:
                logger.warning("NotebookLM startup authentication requires manual renewal")
        except Exception as exc:
            logger.warning("NotebookLM startup auth refresh failed: %s", type(exc).__name__)

        if self.data_dir:
            try:
                os.makedirs(self.data_dir, exist_ok=True)
                path = os.path.join(self.data_dir, "auth.json")
                with open(path, "w", encoding="utf-8") as target:
                    json.dump(self.auth_data, target)
            except OSError as exc:
                logger.warning("Could not persist NotebookLM auth: %s", type(exc).__name__)

    @property
    def configured(self) -> bool:
        return bool(self.local_url or (self.auth_data.get("cookies") or {}))

    @staticmethod
    def _source_fingerprint(sources: list[SourceInfo]) -> str:
        payload = "\n".join(
            sorted(f"{item.id}\t{item.title}" for item in sources)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_sources(items: list[dict]) -> list[SourceInfo]:
        result: list[SourceInfo] = []
        for item in items or []:
            source_id = str(item.get("id") or "").strip()
            title = str(item.get("title") or "").strip()
            if source_id and title:
                result.append(
                    SourceInfo(
                        id=source_id,
                        title=title,
                        url=str(item.get("url") or "").strip() or None,
                    )
                )
        return result

    def _direct_once(
        self,
        notebook_id: str,
        query: str,
        conversation_id: str | None,
        sources_only: bool,
        known_sources: list[SourceInfo] | None = None,
    ) -> dict:
        script = r'''
import json
import os
import sys
import time

payload = json.load(sys.stdin)
if payload.get("build_label"):
    os.environ["NOTEBOOKLM_BL"] = payload["build_label"]

# notebooklm-mcp-2026 0.2.1 hardcodes the retired host.  Patch the
# package configuration before importing its client/protocol modules so both
# local and Railway installations use Google's current NotebookLM host.
from notebooklm_mcp_2026 import config as package_config
base_url = payload.get("base_url") or "https://notebook.google.com"
package_config.BASE_URL = base_url
package_config.BATCHEXECUTE_URL = f"{base_url}/_/LabsTailwindUi/data/batchexecute"
package_config.DEFAULT_HEADERS["Origin"] = base_url
package_config.DEFAULT_HEADERS["Referer"] = f"{base_url}/"

from notebooklm_mcp_2026 import server
from notebooklm_mcp_2026.client import NotebookLMClient
from notebooklm_mcp_2026.tools.query import query_notebook

auth = payload.get("auth") or {}
server._client = NotebookLMClient(
    cookies=auth.get("cookies", {}),
    csrf_token=auth.get("csrf_token", ""),
    session_id=auth.get("session_id", ""),
)

started = time.monotonic()
sources = payload.get("known_sources") or []
if payload.get("sources_only") or not sources:
    raw_notebook = server._client.get_notebook(payload["notebook_id"])
    notebook = raw_notebook[0] if raw_notebook and isinstance(raw_notebook[0], list) else raw_notebook
    raw_sources = notebook[1] if notebook and len(notebook) > 1 and isinstance(notebook[1], list) else []
    sources = []
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

if not sources:
    print(json.dumps({"status": "error", "error": "NotebookLM source list is empty"}))
    raise SystemExit(0)

if payload.get("sources_only"):
    print(json.dumps({
        "status": "success",
        "sources": sources,
        "elapsed": round(time.monotonic() - started, 3),
    }, ensure_ascii=False))
    raise SystemExit(0)

result = query_notebook(
    notebook_id=payload["notebook_id"],
    query=payload["query"],
    conversation_id=payload.get("conversation_id") or None,
)
result["sources"] = sources
result["elapsed"] = round(time.monotonic() - started, 3)
print(json.dumps(result, ensure_ascii=False))
'''
        payload = {
            "notebook_id": notebook_id,
            "query": query,
            "conversation_id": conversation_id,
            "sources_only": sources_only,
            "auth": self.auth_data,
            "build_label": os.getenv("NOTEBOOKLM_BL", ""),
            "base_url": self.base_url,
            "known_sources": [
                {"id": item.id, "title": item.title, "url": item.url}
                for item in (known_sources or [])
            ],
        }
        try:
            process = subprocess.run(
                [sys.executable, "-c", script],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.deadline_seconds,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "NotebookLM timeout"}
        if process.returncode != 0:
            return {
                "status": "error",
                "error": (process.stderr or process.stdout or "subprocess failed")[-2000:],
            }
        output = process.stdout.strip()
        if not output:
            return {"status": "error", "error": "NotebookLM returned empty output"}
        try:
            return json.loads(output.splitlines()[-1])
        except json.JSONDecodeError as exc:
            return {"status": "error", "error": f"NotebookLM JSON error: {exc}"}

    def _proxy_once(
        self,
        notebook_id: str,
        query: str,
        conversation_id: str | None,
        sources_only: bool,
    ) -> dict:
        body = json.dumps(
            {
                "notebook_id": notebook_id,
                "query": query,
                "conversation_id": conversation_id,
                "sources_only": sources_only,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.local_url}/ask",
            data=body,
            headers={"Content-Type": "application/json", "X-Secret": self.local_secret},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.deadline_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("ok"):
                return payload.get("result") or {"status": "error", "error": "empty proxy result"}
            return {"status": "error", "error": str(payload.get("error") or "proxy error")}
        except urllib.error.HTTPError as exc:
            return {"status": "error", "error": f"proxy HTTP {exc.code}"}
        except Exception as exc:
            return {"status": "error", "error": f"proxy {type(exc).__name__}: {exc}"}

    def _call_once(
        self,
        notebook_id: str,
        query: str,
        conversation_id: str | None,
        sources_only: bool,
        known_sources: list[SourceInfo] | None = None,
    ) -> dict:
        if self.local_url:
            return self._proxy_once(notebook_id, query, conversation_id, sources_only)
        return self._direct_once(
            notebook_id, query, conversation_id, sources_only, known_sources
        )

    def ask(
        self,
        notebook_id: str,
        query: str,
        conversation_id: str | None = None,
    ) -> GatewayResponse:
        if not self.configured:
            return GatewayResponse(
                ok=False,
                error_type=ErrorType.CONFIGURATION,
                error="NotebookLM gateway is not configured",
            )
        started = time.monotonic()
        last_error = ""
        last_type = ErrorType.UNKNOWN
        with self._slots:
            sources = self._source_cache.get(notebook_id) or []
            if not sources:
                catalog = self._call_once(notebook_id, "", None, True)
                sources = self._parse_sources(catalog.get("sources") or [])
                if catalog.get("status") != "success" or not sources:
                    last_error = str(
                        catalog.get("error") or "NotebookLM source list is empty"
                    )
                    return GatewayResponse(
                        ok=False,
                        attempts=1,
                        error_type=classify_error(last_error),
                        error=last_error,
                        elapsed_seconds=time.monotonic() - started,
                    )
                self._source_cache[notebook_id] = sources
            for attempt in range(1, self.max_attempts + 1):
                result = self._call_once(
                    notebook_id, query, conversation_id, False, sources
                )
                response_sources = self._parse_sources(result.get("sources") or [])
                answer = str(result.get("answer") or "").strip()
                if result.get("status") == "success" and answer and response_sources:
                    return GatewayResponse(
                        ok=True,
                        raw_answer=answer,
                        conversation_id=str(result.get("conversation_id") or ""),
                        sources=response_sources,
                        source_fingerprint=self._source_fingerprint(response_sources),
                        attempts=attempt,
                        elapsed_seconds=time.monotonic() - started,
                    )
                last_error = str(result.get("error") or "NotebookLM returned no verified payload")
                last_type = classify_error(last_error)
                retryable = last_type in {
                    ErrorType.RATE_LIMIT,
                    ErrorType.SERVER,
                    ErrorType.NETWORK,
                }
                if attempt >= self.max_attempts or not retryable:
                    break
                delay = 0.8 * (2 ** (attempt - 1)) + random.uniform(0.05, 0.35)
                time.sleep(delay)
        return GatewayResponse(
            ok=False,
            attempts=attempt,
            error_type=last_type,
            error=last_error,
            elapsed_seconds=time.monotonic() - started,
        )

    def health(self, notebook_id: str) -> GatewayResponse:
        if not self.configured:
            return GatewayResponse(
                ok=False,
                error_type=ErrorType.CONFIGURATION,
                error="NotebookLM gateway is not configured",
            )
        started = time.monotonic()
        with self._slots:
            result = self._call_once(notebook_id, "", None, True)
        sources = self._parse_sources(result.get("sources") or [])
        if result.get("status") == "success" and sources:
            self._source_cache[notebook_id] = sources
            return GatewayResponse(
                ok=True,
                sources=sources,
                source_fingerprint=self._source_fingerprint(sources),
                attempts=1,
                elapsed_seconds=time.monotonic() - started,
            )
        error = str(result.get("error") or "NotebookLM health check failed")
        return GatewayResponse(
            ok=False,
            attempts=1,
            error_type=classify_error(error),
            error=error,
            elapsed_seconds=time.monotonic() - started,
        )
