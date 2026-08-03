import tempfile
import unittest
from pathlib import Path

from strict_contract import Claim, ContractResult, ResultStatus
from verified_repository import VerifiedRepository


def verified_result():
    return ContractResult(
        status=ResultStatus.VERIFIED,
        answer="Подтверждённый ответ.",
        claims=[
            Claim(
                text="Подтверждённый ответ",
                evidence="Подтверждённый фрагмент ответа",
                source="Источник",
                source_id="source-1",
            )
        ],
        confidence="high",
        raw_answer='{"status":"verified","citation":"[1]"}',
    )


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "verified.db"
        self.repo = VerifiedRepository(database_url="")
        self.repo.sqlite_path = self.path
        self.repo.init_db()

    def tearDown(self):
        self.temp.cleanup()

    def test_only_verified_cards_can_be_saved(self):
        result = ContractResult(status=ResultStatus.PARTIAL)
        with self.assertRaises(ValueError):
            self.repo.store_verified("key", "grebenyuk", "q", result, ["nb"], "fp", 60)

    def test_card_and_session_survive_repository_restart(self):
        self.repo.store_verified(
            "key", "grebenyuk", "Вопрос", verified_result(), ["nb"], "fp", 3600
        )
        self.repo.set_session(12, "grebenyuk", "set-hash", {"nb": "conversation"})
        restarted = VerifiedRepository(database_url="")
        restarted.sqlite_path = self.path
        card = restarted.get_verified("key")
        self.assertIsNotNone(card)
        self.assertEqual(
            restarted.get_session(12, "grebenyuk", "set-hash")["nb"],
            "conversation",
        )

    def test_reset_session_does_not_delete_verified_cards(self):
        self.repo.store_verified(
            "key", "grebenyuk", "Вопрос", verified_result(), ["nb"], "fp", 3600
        )
        self.repo.set_session(12, "grebenyuk", "set-hash", {"nb": "conversation"})
        self.repo.clear_session(12)
        self.assertEqual(self.repo.get_session(12, "grebenyuk", "set-hash"), {})
        self.assertIsNotNone(self.repo.get_verified("key"))


if __name__ == "__main__":
    unittest.main()
