import json
import tempfile
import unittest
from pathlib import Path

from notebook_registry import load_registry
from notebooklm_gateway import GatewayResponse
from strict_contract import ErrorType, ResultStatus, SourceInfo
from strict_service import StrictKnowledgeService
from verified_repository import VerifiedRepository


VALID_RAW = json.dumps(
    {
        "status": "verified",
        "answer": "Метод требует выбрать одну целевую аудиторию.",
        "claims": [
            {
                "text": "Метод требует выбрать одну целевую аудиторию",
                "evidence": "Нужно выбрать одну целевую аудиторию и не распыляться",
                "source": "Метод пяти единичек",
                "citation": "[1]",
            }
        ],
        "missing_information": [],
        "confidence": "high",
    },
    ensure_ascii=False,
)


class FakeGateway:
    configured = True

    def __init__(self, response):
        self.response = response

    def ask(self, notebook_id, query, conversation_id=None):
        return self.response

    def health(self, notebook_id):
        return self.response


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        repo = VerifiedRepository(database_url="")
        repo.sqlite_path = Path(self.temp.name) / "verified.db"
        success = GatewayResponse(
            ok=True,
            raw_answer=VALID_RAW,
            conversation_id="conversation",
            sources=[SourceInfo(id="src-1", title="Метод пяти единичек")],
            source_fingerprint="fingerprint",
            attempts=1,
        )
        self.repo = repo
        self.service = StrictKnowledgeService(load_registry(), FakeGateway(success), repo)
        self.service.init()

    def tearDown(self):
        self.temp.cleanup()

    def test_fresh_verified_answer_is_saved(self):
        result = self.service.answer("Что такое метод?", [], 10)
        self.assertEqual(result.status, ResultStatus.VERIFIED)
        self.assertIn("целевую аудиторию", result.text)
        self.assertEqual(self.repo.stats()["verified_cards"], 1)

    def test_unavailable_uses_only_marked_verified_card(self):
        self.service.answer("Что такое метод?", [], 10)
        self.service.gateway = FakeGateway(
            GatewayResponse(
                ok=False,
                error_type=ErrorType.TIMEOUT,
                error="timeout",
                attempts=2,
            )
        )
        result = self.service.answer("Что такое метод?", [], 10)
        self.assertEqual(result.source_kind, "verified_cache")
        self.assertTrue(result.text.startswith("⚠️"))

    def test_unavailable_without_card_never_improvises(self):
        self.service.gateway = FakeGateway(
            GatewayResponse(
                ok=False,
                error_type=ErrorType.AUTH,
                error="401",
                attempts=1,
            )
        )
        result = self.service.answer("Новый вопрос", [], 10)
        self.assertEqual(result.status, ResultStatus.AUTH_REQUIRED)
        self.assertEqual(result.text, "")

    def test_insufficient_becomes_actionable_clarification(self):
        insufficient_raw = json.dumps(
            {
                "status": "insufficient",
                "answer": "",
                "claims": [],
                "missing_information": [
                    "Чем занимается бизнес",
                    "Какой текущий результат",
                ],
                "confidence": "insufficient",
            },
            ensure_ascii=False,
        )
        self.service.gateway = FakeGateway(
            GatewayResponse(
                ok=True,
                raw_answer=insufficient_raw,
                conversation_id="conversation",
                sources=[SourceInfo(id="src-1", title="Метод пяти единичек")],
                source_fingerprint="fingerprint",
                attempts=1,
            )
        )
        result = self.service.answer(
            "Давай начнем с моего бизнеса",
            [],
            10,
            force_fresh=True,
        )
        self.assertEqual(result.status, ResultStatus.INSUFFICIENT)
        self.assertEqual(result.source_kind, "notebooklm_clarification")
        self.assertIn("короткой диагностики", result.text)
        self.assertIn("Чем занимается бизнес?", result.text)

    def test_verified_coaching_start_always_asks_diagnostic_questions(self):
        result = self.service.answer(
            "Давай начнем с развития моего бизнеса",
            [],
            10,
            force_fresh=True,
        )
        self.assertEqual(result.status, ResultStatus.VERIFIED)
        self.assertGreaterEqual(result.text.count("?"), 4)
        self.assertIn("главным ограничением роста", result.text)

    def test_session_opening_is_immediate_and_does_not_call_knowledge(self):
        self.service.gateway = FakeGateway(
            GatewayResponse(
                ok=False,
                error_type=ErrorType.UNKNOWN,
                error="must not be called",
                attempts=1,
            )
        )
        result = self.service.answer(
            "Давай начнем с развития моего бизнеса",
            [],
            10,
        )
        self.assertEqual(result.status, ResultStatus.INSUFFICIENT)
        self.assertEqual(result.source_kind, "local_clarification")
        self.assertEqual(result.attempts, 0)
        self.assertGreaterEqual(result.text.count("?"), 4)


if __name__ == "__main__":
    unittest.main()
