"""Stable, source-grounded knowledge access backed by Gemini File Search."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass

from google import genai
from google.genai import types


logger = logging.getLogger(__name__)

STORE_NAME = os.getenv("GEMINI_FILE_SEARCH_STORE", "").strip()
MODEL_NAME = os.getenv("GEMINI_FILE_SEARCH_MODEL", "gemini-2.5-flash").strip()
_REQUEST_TIMEOUT_MS = int(os.getenv("KNOWLEDGE_TIMEOUT_MS", "45000"))
_MAX_PARALLEL_REQUESTS = int(os.getenv("KNOWLEDGE_MAX_PARALLEL", "2"))
_request_slots = threading.BoundedSemaphore(max(1, _MAX_PARALLEL_REQUESTS))


class KnowledgeStoreError(RuntimeError):
    """Raised when a grounded answer cannot be produced safely."""


@dataclass(frozen=True)
class KnowledgeAnswer:
    text: str
    grounding_chunks: int
    elapsed_seconds: float


def is_configured() -> bool:
    return bool(STORE_NAME)


def _client(api_key: str) -> genai.Client:
    if not api_key:
        raise KnowledgeStoreError("GEMINI_API_KEY is not configured")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
    )


def _grounding_chunk_count(response) -> int:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return 0
    metadata = getattr(candidates[0], "grounding_metadata", None)
    chunks = getattr(metadata, "grounding_chunks", None) if metadata else None
    return len(chunks or [])


def ask(
    api_key: str,
    prompt: str,
    system_instruction: str = "",
) -> KnowledgeAnswer:
    """Return an answer only when File Search supplied grounding evidence."""
    if not is_configured():
        raise KnowledgeStoreError("GEMINI_FILE_SEARCH_STORE is not configured")

    started_at = time.monotonic()
    with _request_slots:
        client = _client(api_key)
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                grounding_instruction = (
                    "Используй только сведения, найденные инструментом File Search. "
                    "Не дополняй ответ внешними знаниями и не упоминай источники, "
                    "документы, поиск или внутреннее устройство системы. Если в "
                    "документах недостаточно сведений, ответь ровно: "
                    "НЕДОСТАТОЧНО_ДАННЫХ"
                )
                if system_instruction:
                    grounding_instruction += f"\n\n{system_instruction}"
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=grounding_instruction,
                        tools=[
                            types.Tool(
                                file_search=types.FileSearch(
                                    file_search_store_names=[STORE_NAME],
                                    top_k=12,
                                )
                            )
                        ],
                        max_output_tokens=768,
                    ),
                )
                text = (response.text or "").strip()
                chunks = _grounding_chunk_count(response)
                normalized_text = re.sub(r"[\s-]+", "_", text.upper())
                if not text or "НЕДОСТАТОЧНО_ДАННЫХ" in normalized_text:
                    raise KnowledgeStoreError("knowledge store returned no usable answer")
                if chunks < 1:
                    raise KnowledgeStoreError("answer was not grounded in indexed documents")
                if len(text.split()) < 40 and attempt == 0:
                    raise KnowledgeStoreError("grounded answer was unexpectedly too short")
                elapsed = time.monotonic() - started_at
                logger.info(
                    "File Search answer ready in %.2fs with %s grounding chunks",
                    elapsed,
                    chunks,
                )
                return KnowledgeAnswer(text=text, grounding_chunks=chunks, elapsed_seconds=elapsed)
            except Exception as exc:
                last_error = exc
                value = str(exc).lower()
                transient = any(
                    marker in value
                    for marker in (
                        "429",
                        "500",
                        "502",
                        "503",
                        "504",
                        "timeout",
                        "timed out",
                        "temporarily unavailable",
                        "connection reset",
                        "not grounded",
                        "no usable answer",
                        "too short",
                    )
                )
                if attempt < 2 and transient:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                break

    raise KnowledgeStoreError(
        f"File Search query failed: {type(last_error).__name__}: {last_error}"
    )


def health_check(api_key: str) -> tuple[bool, str]:
    """Verify that the store exists, has documents, and can ground a real answer."""
    if not is_configured():
        return False, "GEMINI_FILE_SEARCH_STORE is not configured"
    try:
        client = _client(api_key)
        store = client.file_search_stores.get(name=STORE_NAME)
        documents = client.file_search_stores.documents.list(parent=STORE_NAME)
        first_document = next(iter(documents), None)
        if first_document is None:
            return False, "File Search store has no indexed documents"
        result = ask(
            api_key,
            "Объясни одним коротким абзацем метод пяти единичек в бизнесе.",
        )
        display_name = getattr(store, "display_name", "") or STORE_NAME
        return True, (
            f"{display_name}: retrieval OK, grounding chunks={result.grounding_chunks}, "
            f"elapsed={result.elapsed_seconds:.2f}s"
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
