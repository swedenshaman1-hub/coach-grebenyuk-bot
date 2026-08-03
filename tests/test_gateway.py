import json
import threading
import unittest

from notebooklm_gateway import NotebookLMGateway, classify_error
from strict_contract import ErrorType, SourceInfo


class FakeGateway(NotebookLMGateway):
    def __init__(self, results):
        self.results = list(results)
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
        return self.results.pop(0)


class GatewayTests(unittest.TestCase):
    def test_401_is_auth_required_and_not_retried(self):
        gateway = FakeGateway([{"status": "error", "error": "HTTP 401"}])
        result = gateway.ask("nb", "q")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, ErrorType.AUTH)
        self.assertEqual(result.attempts, 1)

    def test_timeout_is_not_duplicated_after_full_deadline(self):
        gateway = FakeGateway([{"status": "error", "error": "timeout"}])
        result = gateway.ask("nb", "q")
        self.assertEqual(result.error_type, ErrorType.TIMEOUT)
        self.assertEqual(result.attempts, 1)

    def test_error_classification(self):
        self.assertEqual(classify_error("429 rate limit"), ErrorType.RATE_LIMIT)
        self.assertEqual(classify_error("503 unavailable"), ErrorType.SERVER)


if __name__ == "__main__":
    unittest.main()
