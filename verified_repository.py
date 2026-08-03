"""Persistent verified-knowledge storage.

Local development uses SQLite. Railway uses PostgreSQL whenever ``DATABASE_URL``
is present. Only evidence-validated cards are accepted by ``store_verified``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from strict_contract import ContractResult


DEFAULT_SQLITE_PATH = Path(__file__).with_name("data") / "verified_knowledge.db"


@dataclass(frozen=True)
class KnowledgeCard:
    id: str
    cache_key: str
    collection: str
    question: str
    answer: str
    claims: list[dict[str, str]]
    notebook_ids: list[str]
    source_refs: list[dict[str, str]]
    source_fingerprint: str
    confidence: str
    created_at: int
    verified_at: int
    expires_at: int
    raw_response_hash: str


class VerifiedRepository:
    def __init__(self, database_url: str | None = None):
        self.database_url = (database_url or os.getenv("DATABASE_URL", "")).strip()
        self.is_postgres = self.database_url.startswith(("postgres://", "postgresql://"))
        self.sqlite_path = Path(
            os.getenv("VERIFIED_DB_PATH", "") or DEFAULT_SQLITE_PATH
        ).resolve()

    @property
    def backend_name(self) -> str:
        return "PostgreSQL" if self.is_postgres else f"SQLite:{self.sqlite_path}"

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.is_postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError("psycopg is required when DATABASE_URL is configured") from exc
            with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
                yield connection
            return

        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.sqlite_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 15000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _sql(self, query: str) -> str:
        return query.replace("?", "%s") if self.is_postgres else query

    def init_db(self) -> None:
        integer = "BIGINT" if self.is_postgres else "INTEGER"
        statements = [
            f"""
            CREATE TABLE IF NOT EXISTS knowledge_cards (
                id TEXT PRIMARY KEY,
                cache_key TEXT NOT NULL UNIQUE,
                collection TEXT NOT NULL,
                question TEXT NOT NULL,
                normalized_question TEXT NOT NULL,
                answer TEXT NOT NULL,
                claims_json TEXT NOT NULL,
                notebook_ids_json TEXT NOT NULL,
                source_refs_json TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence TEXT NOT NULL,
                created_at {integer} NOT NULL,
                verified_at {integer} NOT NULL,
                expires_at {integer} NOT NULL,
                raw_response_hash TEXT NOT NULL
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS notebook_sessions (
                chat_id {integer} PRIMARY KEY,
                collection TEXT NOT NULL,
                notebook_set_hash TEXT NOT NULL,
                conversation_ids_json TEXT NOT NULL,
                updated_at {integer} NOT NULL
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS request_audit (
                request_id TEXT PRIMARY KEY,
                chat_hash TEXT NOT NULL,
                collection TEXT NOT NULL,
                notebook_ids_json TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                duration_ms {integer} NOT NULL,
                attempts INTEGER NOT NULL,
                validation_status TEXT NOT NULL,
                error_type TEXT NOT NULL,
                renderer_used INTEGER NOT NULL,
                created_at {integer} NOT NULL
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS raw_responses (
                request_id TEXT PRIMARY KEY,
                raw_response TEXT NOT NULL,
                response_hash TEXT NOT NULL,
                created_at {integer} NOT NULL,
                expires_at {integer} NOT NULL
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS health_state (
                component TEXT PRIMARY KEY,
                ok INTEGER NOT NULL,
                detail TEXT NOT NULL,
                checked_at {integer} NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_cards_collection_expiry ON knowledge_cards(collection, expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_audit_created_at ON request_audit(created_at)",
        ]
        with self._connect() as connection:
            if not self.is_postgres:
                connection.execute("PRAGMA journal_mode = WAL")
            for statement in statements:
                connection.execute(statement)

    @staticmethod
    def normalize_question(question: str) -> str:
        return " ".join(question.lower().split())

    @staticmethod
    def cache_key(collection: str, question: str, history: list[dict[str, str]]) -> str:
        context = [
            {
                "role": item.get("role", ""),
                "text": " ".join(str(item.get("text") or "").lower().split())[:800],
            }
            for item in history[-4:]
        ]
        payload = json.dumps(
            {
                "v": 3,
                "collection": collection,
                "question": VerifiedRepository.normalize_question(question),
                "history": context,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_chat_id(chat_id: int) -> str:
        salt = os.getenv("LOG_HASH_SALT", "growth-architect-v3")
        return hashlib.sha256(f"{salt}:{chat_id}".encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _row_to_card(row: Any) -> KnowledgeCard | None:
        if row is None:
            return None
        return KnowledgeCard(
            id=str(row["id"]),
            cache_key=str(row["cache_key"]),
            collection=str(row["collection"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            claims=json.loads(row["claims_json"]),
            notebook_ids=json.loads(row["notebook_ids_json"]),
            source_refs=json.loads(row["source_refs_json"]),
            source_fingerprint=str(row["source_fingerprint"]),
            confidence=str(row["confidence"]),
            created_at=int(row["created_at"]),
            verified_at=int(row["verified_at"]),
            expires_at=int(row["expires_at"]),
            raw_response_hash=str(row["raw_response_hash"]),
        )

    def get_verified(self, cache_key: str, include_expired: bool = False) -> KnowledgeCard | None:
        query = "SELECT * FROM knowledge_cards WHERE cache_key = ? AND status = 'verified'"
        params: list[Any] = [cache_key]
        if not include_expired:
            query += " AND expires_at > ?"
            params.append(int(time.time()))
        with self._connect() as connection:
            row = connection.execute(self._sql(query), params).fetchone()
        return self._row_to_card(row)

    def store_verified(
        self,
        cache_key: str,
        collection: str,
        question: str,
        result: ContractResult,
        notebook_ids: list[str],
        source_fingerprint: str,
        ttl_seconds: int,
    ) -> KnowledgeCard:
        if not result.is_verified:
            raise ValueError("Unverified NotebookLM results cannot enter knowledge_cards")
        now = int(time.time())
        card_id = str(uuid.uuid4())
        raw_hash = hashlib.sha256(result.raw_answer.encode("utf-8")).hexdigest()
        claims = [claim.as_dict() for claim in result.claims]
        source_refs = [
            {
                "source_id": claim.source_id,
                "source": claim.source,
                "evidence": claim.evidence,
                "citation": claim.citation,
            }
            for claim in result.claims
        ]
        values = (
            card_id,
            cache_key,
            collection,
            question,
            self.normalize_question(question),
            result.answer,
            json.dumps(claims, ensure_ascii=False),
            json.dumps(notebook_ids, ensure_ascii=False),
            json.dumps(source_refs, ensure_ascii=False),
            source_fingerprint,
            "verified",
            result.confidence,
            now,
            now,
            now + ttl_seconds,
            raw_hash,
        )
        query = """
            INSERT INTO knowledge_cards(
                id, cache_key, collection, question, normalized_question, answer,
                claims_json, notebook_ids_json, source_refs_json,
                source_fingerprint, status, confidence, created_at, verified_at,
                expires_at, raw_response_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                id = excluded.id,
                question = excluded.question,
                normalized_question = excluded.normalized_question,
                answer = excluded.answer,
                claims_json = excluded.claims_json,
                notebook_ids_json = excluded.notebook_ids_json,
                source_refs_json = excluded.source_refs_json,
                source_fingerprint = excluded.source_fingerprint,
                status = excluded.status,
                confidence = excluded.confidence,
                created_at = excluded.created_at,
                verified_at = excluded.verified_at,
                expires_at = excluded.expires_at,
                raw_response_hash = excluded.raw_response_hash
        """
        with self._connect() as connection:
            connection.execute(self._sql(query), values)
        card = self.get_verified(cache_key)
        if card is None:
            raise RuntimeError("Verified card was not persisted")
        return card

    def store_raw(self, request_id: str, raw_response: str, ttl_seconds: int = 7 * 86400) -> None:
        now = int(time.time())
        response_hash = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
        query = """
            INSERT INTO raw_responses(request_id, raw_response, response_hash, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                raw_response = excluded.raw_response,
                response_hash = excluded.response_hash,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at
        """
        with self._connect() as connection:
            connection.execute(
                self._sql(query),
                (request_id, raw_response, response_hash, now, now + ttl_seconds),
            )
            connection.execute(
                self._sql("DELETE FROM raw_responses WHERE expires_at <= ?"),
                (now,),
            )

    def get_session(self, chat_id: int, collection: str, notebook_set_hash: str) -> dict[str, str]:
        query = """
            SELECT conversation_ids_json, notebook_set_hash
            FROM notebook_sessions WHERE chat_id = ? AND collection = ?
        """
        with self._connect() as connection:
            row = connection.execute(self._sql(query), (chat_id, collection)).fetchone()
        if row is None or str(row["notebook_set_hash"]) != notebook_set_hash:
            return {}
        try:
            value = json.loads(row["conversation_ids_json"])
            return {str(key): str(item) for key, item in value.items() if item}
        except (TypeError, json.JSONDecodeError):
            return {}

    def set_session(
        self,
        chat_id: int,
        collection: str,
        notebook_set_hash: str,
        conversation_ids: dict[str, str],
    ) -> None:
        now = int(time.time())
        query = """
            INSERT INTO notebook_sessions(
                chat_id, collection, notebook_set_hash, conversation_ids_json, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                collection = excluded.collection,
                notebook_set_hash = excluded.notebook_set_hash,
                conversation_ids_json = excluded.conversation_ids_json,
                updated_at = excluded.updated_at
        """
        with self._connect() as connection:
            connection.execute(
                self._sql(query),
                (
                    chat_id,
                    collection,
                    notebook_set_hash,
                    json.dumps(conversation_ids, ensure_ascii=False),
                    now,
                ),
            )

    def clear_session(self, chat_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                self._sql("DELETE FROM notebook_sessions WHERE chat_id = ?"),
                (chat_id,),
            )

    def log_request(
        self,
        request_id: str,
        chat_id: int,
        collection: str,
        notebook_ids: list[str],
        source_kind: str,
        duration_ms: int,
        attempts: int,
        validation_status: str,
        error_type: str,
        renderer_used: bool,
    ) -> None:
        query = """
            INSERT INTO request_audit(
                request_id, chat_hash, collection, notebook_ids_json, source_kind,
                duration_ms, attempts, validation_status, error_type,
                renderer_used, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO NOTHING
        """
        with self._connect() as connection:
            connection.execute(
                self._sql(query),
                (
                    request_id,
                    self.hash_chat_id(chat_id),
                    collection,
                    json.dumps(notebook_ids),
                    source_kind,
                    duration_ms,
                    attempts,
                    validation_status,
                    error_type,
                    1 if renderer_used else 0,
                    int(time.time()),
                ),
            )

    def set_health(self, component: str, ok: bool, detail: str) -> None:
        query = """
            INSERT INTO health_state(component, ok, detail, checked_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(component) DO UPDATE SET
                ok = excluded.ok,
                detail = excluded.detail,
                checked_at = excluded.checked_at
        """
        with self._connect() as connection:
            connection.execute(
                self._sql(query),
                (component, 1 if ok else 0, detail[:1000], int(time.time())),
            )

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            cards = connection.execute(
                "SELECT COUNT(*) AS value FROM knowledge_cards WHERE status = 'verified'"
            ).fetchone()
            audits = connection.execute(
                "SELECT COUNT(*) AS value FROM request_audit"
            ).fetchone()
        return {"verified_cards": int(cards["value"]), "requests": int(audits["value"])}
