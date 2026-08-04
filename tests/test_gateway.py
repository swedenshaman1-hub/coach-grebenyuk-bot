import json
import threading
import unittest

from notebooklm_gateway import NotebookLMGateway, classify_error
from strict_contract import ErrorType, SourceInfo


class FakeGateway(NotebookLMGateway):
    def __init__(self, results):
        self.results = list(results)
        self.conversation_ids = []
        self.local_url = "fake"
        self.local_secret = "secret"
        self.auth_data = {}
        self._slots = threading.BoundedSemaphore(1)
        self.deadline_seconds = 10
        self.max_attempts = 2
        self._source_cache = {"nb": [SourceInfo(id="s1", title="Source 1")]}

    def _call_once(
        self, notebook_id, query, conversation_id, sources_only, known_sources=None
    ):
        self.conversation_ids.append(conversation_id)
        return self.results.pop(0)


class GatewayTests(unittest.TestCase):
    def test_401_is_auth_required_and_not_retried(self):
        gateway = FakeGateway([{"status": "error", "error": "HTTP 401"}])
        result = gateway.ask("nb", "q")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, ErrorType.AUTH)
        self.assertEqual(result.attempts, 1)

    def test_timeout_retries_once_on_a_clean_conversation(self):
        gateway = FakeGateway([
            {"status": "error", "error": "timeout"},
            {
                "status": "success",
                "answer": "verified",
                "conversation_id": "new-conversation",
                "sources": [{"id": "s1", "title": "Source 1"}],
            },
        ])
        result = gateway.ask("nb", "q", conversation_id="stalled-conversation")
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(gateway.conversation_ids, ["stalled-conversation", None])

    def test_second_timeout_returns_a_bounded_failure(self):
        gateway = FakeGateway([
            {"status": "error", "error": "timeout"},
            {"status": "error", "error": "timeout"},
        ])
        result = gateway.ask("nb", "q")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, ErrorType.TIMEOUT)
        self.assertEqual(result.attempts, 2)

    def test_error_classification(self):
        self.assertEqual(classify_error("429 rate limit"), ErrorType.RATE_LIMIT)
        self.assertEqual(classify_error("503 unavailable"), ErrorType.SERVER)


if __name__ == "__main__":
    unittest.main()
