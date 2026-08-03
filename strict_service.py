"""Strict orchestration: NotebookLM -> evidence validation -> verified storage."""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from notebook_registry import NotebookRegistry
from notebooklm_gateway import NotebookLMGateway
from strict_contract import (
    ContractResult,
    ErrorType,
    ResultStatus,
    build_strict_prompt,
    parse_and_validate,
    render_verified_answer,
)
from verified_repository import KnowledgeCard, VerifiedRepository


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceAnswer:
    status: ResultStatus
    text: str = ""
    source_kind: str = ""
    request_id: str = ""
    verified_at: int | None = None
    error_type: ErrorType = ErrorType.NONE
    attempts: int = 0
    validation_errors: tuple[str, ...] = ()


class StrictKnowledgeService:
    def __init__(
        self,
        registry: NotebookRegistry,
        gateway: NotebookLMGateway,
        repository: VerifiedRepository,
        collection_id: str = "grebenyuk",
    ):
        self.registry = registry
        self.gateway = gateway
        self.repository = repository
        self.collection = registry.collection(collection_id)
        self.card_ttl_seconds = int(os.getenv("VERIFIED_CARD_TTL", str(30 * 86400)))
        self.require_fresh = os.getenv("REQUIRE_FRESH_VERIFICATION", "false").lower() in {
            "1", "true", "yes", "on",
        }

    def init(self) -> None:
        self.repository.init_db()

    def _log(
        self,
        request_id: str,
        chat_id: int,
        source_kind: str,
        started: float,
        attempts: int,
        status: ResultStatus,
        error_type: ErrorType,
    ) -> None:
        self.repository.log_request(
            request_id=request_id,
            chat_id=chat_id,
            collection=self.collection.id,
            notebook_ids=[item.uuid for item in self.collection.notebooks],
            source_kind=source_kind,
            duration_ms=int((time.monotonic() - started) * 1000),
            attempts=attempts,
            validation_status=status.value,
            error_type=error_type.value,
            renderer_used=False,
        )

    @staticmethod
    def _stale_text(card: KnowledgeCard) -> str:
        verified = datetime.fromtimestamp(card.verified_at, timezone.utc).astimezone()
        stamp = verified.strftime("%d.%m.%Y")
        return (
            f"⚠️ Основная база сейчас недоступна. Ниже ответ, подтверждённый {stamp}; "
            "повторно проверить его сейчас не удалось.\n\n"
            f"{card.answer}"
        )

    def answer(
        self,
        question: str,
        history: list[dict[str, str]],
        chat_id: int,
        force_fresh: bool = False,
    ) -> ServiceAnswer:
        request_id = str(uuid.uuid4())
        started = time.monotonic()
        cache_key = self.repository.cache_key(
            self.collection.id, question, history
        )
        stale_card = self.repository.get_verified(cache_key, include_expired=True)
        conversations = self.repository.get_session(
            chat_id,
            self.collection.id,
            self.collection.notebook_set_hash,
        )
        prompt = build_strict_prompt(question, history)
        verified_results: list[ContractResult] = []
        raw_parts: list[str] = []
        fingerprints: list[str] = []
        total_attempts = 0
        last_error_type = ErrorType.NONE

        for notebook in self.collection.notebooks:
            gateway_result = self.gateway.ask(
                notebook.uuid,
                prompt,
                conversations.get(notebook.uuid),
            )
            total_attempts += gateway_result.attempts
            if not gateway_result.ok:
                last_error_type = gateway_result.error_type
                logger.warning(
                    "Strict NotebookLM request %s failed: notebook=%s type=%s",
                    request_id,
                    notebook.id,
                    gateway_result.error_type.value,
                )
                status = (
                    ResultStatus.AUTH_REQUIRED
                    if gateway_result.error_type is ErrorType.AUTH
                    else ResultStatus.UNAVAILABLE
                )
                if stale_card and not (force_fresh or self.require_fresh):
                    self._log(
                        request_id, chat_id, "verified_cache", started,
                        total_attempts, status, gateway_result.error_type,
                    )
                    return ServiceAnswer(
                        status=status,
                        text=self._stale_text(stale_card),
                        source_kind="verified_cache",
                        request_id=request_id,
                        verified_at=stale_card.verified_at,
                        error_type=gateway_result.error_type,
                        attempts=total_attempts,
                    )
                self._log(
                    request_id, chat_id, "none", started,
                    total_attempts, status, gateway_result.error_type,
                )
                return ServiceAnswer(
                    status=status,
                    request_id=request_id,
                    error_type=gateway_result.error_type,
                    attempts=total_attempts,
                )

            raw_parts.append(gateway_result.raw_answer)
            fingerprints.append(gateway_result.source_fingerprint)
            if gateway_result.conversation_id:
                conversations[notebook.uuid] = gateway_result.conversation_id
            parsed = parse_and_validate(gateway_result.raw_answer, gateway_result.sources)
            self.repository.store_raw(request_id + ":" + notebook.id, gateway_result.raw_answer)

            if parsed.status is ResultStatus.INSUFFICIENT:
                self.repository.set_session(
                    chat_id,
                    self.collection.id,
                    self.collection.notebook_set_hash,
                    conversations,
                )
                self._log(
                    request_id, chat_id, "notebooklm_fresh", started,
                    total_attempts, ResultStatus.INSUFFICIENT, ErrorType.NONE,
                )
                return ServiceAnswer(
                    status=ResultStatus.INSUFFICIENT,
                    request_id=request_id,
                    source_kind="notebooklm_fresh",
                    attempts=total_attempts,
                )
            if not parsed.is_verified:
                logger.warning(
                    "Strict validation rejected request %s: %s",
                    request_id,
                    "; ".join(parsed.validation_errors)[:500],
                )
                self._log(
                    request_id, chat_id, "notebooklm_rejected", started,
                    total_attempts, ResultStatus.PARTIAL, ErrorType.VALIDATION,
                )
                return ServiceAnswer(
                    status=ResultStatus.PARTIAL,
                    request_id=request_id,
                    source_kind="notebooklm_rejected",
                    error_type=ErrorType.VALIDATION,
                    attempts=total_attempts,
                    validation_errors=tuple(parsed.validation_errors),
                )
            verified_results.append(parsed)

        self.repository.set_session(
            chat_id,
            self.collection.id,
            self.collection.notebook_set_hash,
            conversations,
        )
        all_claims = [claim for result in verified_results for claim in result.claims]
        combined = ContractResult(
            status=ResultStatus.VERIFIED,
            answer="\n\n".join(result.answer for result in verified_results if result.answer),
            claims=all_claims,
            missing_information=[],
            confidence=(
                "high"
                if verified_results and all(item.confidence == "high" for item in verified_results)
                else "medium"
            ),
            raw_answer="\n\n--- NOTEBOOK ---\n\n".join(raw_parts),
        )
        public_answer = render_verified_answer(combined)
        combined.answer = public_answer
        source_fingerprint = hashlib.sha256(
            "\n".join(fingerprints).encode("utf-8")
        ).hexdigest()
        card = self.repository.store_verified(
            cache_key=cache_key,
            collection=self.collection.id,
            question=question,
            result=combined,
            notebook_ids=[item.uuid for item in self.collection.notebooks],
            source_fingerprint=source_fingerprint,
            ttl_seconds=self.card_ttl_seconds,
        )
        self._log(
            request_id, chat_id, "notebooklm_fresh", started,
            total_attempts, ResultStatus.VERIFIED, last_error_type,
        )
        logger.info(
            "Strict request ready: request=%s claims=%s elapsed=%.2fs",
            request_id,
            len(all_claims),
            time.monotonic() - started,
        )
        return ServiceAnswer(
            status=ResultStatus.VERIFIED,
            text=public_answer,
            source_kind="notebooklm_fresh",
            request_id=request_id,
            verified_at=card.verified_at,
            attempts=total_attempts,
        )

    def reset_session(self, chat_id: int) -> None:
        self.repository.clear_session(chat_id)

    def health(self) -> tuple[bool, str]:
        total_sources = 0
        elapsed = 0.0
        for notebook in self.collection.notebooks:
            result = self.gateway.health(notebook.uuid)
            elapsed += result.elapsed_seconds
            if not result.ok:
                detail = f"{notebook.id}: {result.error_type.value}"
                self.repository.set_health("notebooklm_gateway", False, detail)
                return False, detail
            total_sources += len(result.sources)
        detail = (
            f"collection={self.collection.id}, notebooks={len(self.collection.notebooks)}, "
            f"sources={total_sources}, elapsed={elapsed:.2f}s"
        )
        self.repository.set_health("notebooklm_gateway", True, detail)
        return True, detail

    def source_summary(self) -> str:
        lines = [
            f"Коллекция: {self.collection.id}",
            f"Режим: {self.collection.mode}",
        ]
        for notebook in self.collection.notebooks:
            lines.append(f"{notebook.id}: {notebook.uuid} — {notebook.role}")
        return "\n".join(lines)

    def cache_info(
        self,
        question: str,
        history: list[dict[str, str]],
    ) -> KnowledgeCard | None:
        key = self.repository.cache_key(self.collection.id, question, history)
        return self.repository.get_verified(key, include_expired=True)
