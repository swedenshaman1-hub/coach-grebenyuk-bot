"""Production smoke test for a broad request to begin business coaching."""

from __future__ import annotations

from notebook_registry import load_registry
from notebooklm_gateway import NotebookLMGateway
from strict_contract import ResultStatus
from strict_service import StrictKnowledgeService
from verified_repository import VerifiedRepository


def main() -> int:
    service = StrictKnowledgeService(
        load_registry(),
        NotebookLMGateway(),
        VerifiedRepository(),
    )
    service.init()
    question = "Давай начнем с развития моего бизнеса."
    result = service.answer(question, [], -900000001, force_fresh=True)
    actionable = bool(result.text.strip()) and "?" in result.text
    print(
        f"status={result.status.value} source={result.source_kind} "
        f"attempts={result.attempts} chars={len(result.text)} actionable={actionable}"
    )
    if result.status is ResultStatus.VERIFIED:
        return 0 if actionable and result.source_kind == "notebooklm_fresh" else 3
    if result.status is ResultStatus.INSUFFICIENT:
        safe_clarification = actionable and "диагностик" in result.text.lower()
        return 0 if safe_clarification else 4
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
