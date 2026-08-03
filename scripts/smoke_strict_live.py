"""One real NotebookLM smoke test without printing secrets or source text."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from notebook_registry import load_registry
from notebooklm_gateway import NotebookLMGateway
from strict_service import StrictKnowledgeService
from verified_repository import VerifiedRepository


def main() -> int:
    registry = load_registry()
    repository = VerifiedRepository(database_url="")
    repository.sqlite_path = Path(tempfile.gettempdir()) / "grebenyuk-strict-smoke.db"
    if repository.sqlite_path.exists():
        repository.sqlite_path.unlink()
    service = StrictKnowledgeService(registry, NotebookLMGateway(), repository)
    service.init()
    health_ok, health_detail = service.health()
    print(f"health_ok={health_ok} detail={health_detail}")
    if not health_ok:
        return 2
    question = "Что такое метод пяти единичек и из каких элементов он состоит?"
    result = service.answer(question, [], 1288155468, force_fresh=True)
    print(
        f"status={result.status.value} source={result.source_kind} "
        f"attempts={result.attempts} chars={len(result.text)}"
    )
    card = service.cache_info(question, [])
    print(
        f"card={bool(card)} claims={len(card.claims) if card else 0} "
        f"source_refs={len(card.source_refs) if card else 0}"
    )
    return 0 if result.status.value == "verified" and card and card.claims else 3


if __name__ == "__main__":
    raise SystemExit(main())
