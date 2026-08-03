import unittest
from pathlib import Path

from notebook_registry import load_registry


ROOT = Path(__file__).resolve().parents[1]


class StaticPolicyTests(unittest.TestCase):
    def test_uuid_is_not_duplicated_in_python_code(self):
        uuid = load_registry().collection("grebenyuk").notebooks[0].uuid
        offenders = [
            path.name
            for path in ROOT.glob("*.py")
            if uuid in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])

    def test_production_bot_has_no_file_search_or_emergency_answer(self):
        bot = (ROOT / "bot.py").read_text(encoding="utf-8")
        self.assertNotIn("knowledge_store", bot)
        self.assertNotIn("_emergency_coach_answer", bot)
        self.assertNotIn("GEMINI_FILE_SEARCH_STORE", bot)
        self.assertNotIn("COACH_SYSTEM_PROMPT", bot)

    def test_no_automatic_internet_search_provider(self):
        production = "\n".join(
            path.read_text(encoding="utf-8")
            for path in ROOT.glob("*.py")
        ).lower()
        self.assertNotIn("google_search=", production)
        self.assertNotIn("web_search", production)

    def test_telegram_profile_mutation_is_opt_in(self):
        bot = (ROOT / "bot.py").read_text(encoding="utf-8")
        self.assertIn('os.getenv("SYNC_TELEGRAM_PROFILE", "false")', bot)


if __name__ == "__main__":
    unittest.main()
