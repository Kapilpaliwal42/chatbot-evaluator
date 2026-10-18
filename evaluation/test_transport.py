"""Failure modes at the model boundary must never silently produce passes."""

import copy
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from evaluation.core import OllamaJudge


CASE = {"id": "transport", "question": "What is the return window?",
        "answer": "14 days.", "references": [{"id": "policy", "text": "Returns are accepted for 14 days."}],
        "requirements": [], "expected_verdict": "pass"}
JUDGMENT = {"scores": {"groundedness": 4, "instruction_following": 4, "completeness": 4},
            "rationale": "The answer accurately states the supplied policy.", "findings": [],
            "needs_human_review": False, "human_review_reasons": []}


class TransportTests(unittest.TestCase):
    def test_structured_request_and_held_out_label(self):
        case = copy.deepcopy(CASE)
        judge = OllamaJudge()
        response = {"response": json.dumps(JUDGMENT), "done": True, "done_reason": "stop"}
        with patch.object(judge, "_request", return_value=response) as request:
            self.assertEqual(judge.judge(case)["verdict"], "pass")
        route, payload = request.call_args.args
        self.assertEqual(route, "/api/generate")
        self.assertEqual(payload["format"]["type"], "object")
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertNotIn("expected_verdict", payload["prompt"])
        self.assertEqual(case["raw_response"], response["response"])

    def test_complete_json_at_token_limit_is_still_rejected(self):
        case = copy.deepcopy(CASE)
        judge = OllamaJudge()
        response = {"response": json.dumps(JUDGMENT), "done": True, "done_reason": "length"}
        with patch.object(judge, "_request", return_value=response):
            with self.assertRaisesRegex(ValueError, "output-token limit"):
                judge.judge(case)
        self.assertEqual(case["raw_response"], response["response"])

    def test_oversized_documents_are_not_silently_truncated(self):
        case = copy.deepcopy(CASE)
        case["references"][0]["text"] = "x" * 20000
        judge = OllamaJudge()
        with patch.object(judge, "_request") as request:
            with self.assertRaisesRegex(ValueError, "context budget"):
                judge.judge(case)
        request.assert_not_called()

    def test_network_failure_has_a_clear_error(self):
        judge = OllamaJudge()
        with patch("evaluation.core.urlopen", side_effect=URLError("connection refused")):
            with self.assertRaisesRegex(RuntimeError, "Cannot reach Ollama"):
                judge.doctor()

    def test_invalid_server_origin_is_rejected(self):
        for url in ("file:///secret", "http://user:password@localhost:11434", "http://localhost/api", "http://localhost?x=1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OllamaJudge(base_url=url)


if __name__ == "__main__":
    unittest.main()
