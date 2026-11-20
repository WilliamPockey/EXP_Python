import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app import app, create_app
from verifier import ModelError


class VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture_path = Path(__file__).resolve().parents[1] / "examples" / "assignment.json"
        cls.fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        self.request = deepcopy(self.fixture["request"])
        self.payload = deepcopy(self.fixture["response"])
        self.model = Mock()
        self.model.verify.return_value = self.payload
        self.client = self.enterContext(TestClient(create_app(self.model)))

    def assert_invalid_model_output(self) -> None:
        response = self.client.post("/verify", json=self.request)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "Invalid verification model output"})

    def test_assignment_example(self) -> None:
        with TestClient(app) as demo_client:
            response = demo_client.post("/verify", json=self.request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.payload)
        self.assertEqual(
            [claim["verdict"] for claim in response.json()["claims"]],
            ["CONTRADICTED", "SUPPORTED", "SUPPORTED", "UNSUPPORTED"],
        )

    def test_rejects_invalid_passage_indices(self) -> None:
        citation = self.payload["claims"][0]["citations"][0]
        for index in (-1, len(self.request["context_passages"]), True, "0"):
            with self.subTest(index=index):
                citation["index"] = index
                self.assert_invalid_model_output()

    def test_rejects_a_real_quote_from_the_wrong_passage(self) -> None:
        self.payload["claims"][0]["citations"][0]["index"] = 1
        self.assert_invalid_model_output()

    def test_rejects_non_verbatim_or_blank_quotes(self) -> None:
        citation = self.payload["claims"][0]["citations"][0]
        original_quote = citation["quote"]
        for quote in ("Invented evidence", original_quote.lower(), original_quote + " ", " "):
            with self.subTest(quote=quote):
                citation["quote"] = quote
                self.assert_invalid_model_output()

    def test_accepts_a_verbatim_fragment(self) -> None:
        citation = self.payload["claims"][0]["citations"][0]
        citation["quote"] = "WAL does not work over a network filesystem"
        response = self.client.post("/verify", json=self.request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.payload)

    def test_supported_and_contradicted_require_citations(self) -> None:
        for claim in self.payload["claims"][:2]:
            with self.subTest(verdict=claim["verdict"]):
                original_citations = claim["citations"]
                claim["citations"] = []
                self.assert_invalid_model_output()
                claim["citations"] = original_citations

    def test_unsupported_rejects_citations(self) -> None:
        self.payload["claims"][-1]["citations"] = self.payload["claims"][0]["citations"]
        self.assert_invalid_model_output()

    def test_rejects_malformed_model_output(self) -> None:
        for payload in (None, {}, "not JSON", {"claims": ["not a claim"]}):
            with self.subTest(payload=payload):
                self.model.verify.return_value = payload
                self.assert_invalid_model_output()

    def test_rejects_unknown_verdict(self) -> None:
        self.payload["claims"][0]["verdict"] = "UNKNOWN"
        self.assert_invalid_model_output()

    def test_blank_answer_does_not_call_the_model(self) -> None:
        self.request["generated_answer"] = " \t\n"
        response = self.client.post("/verify", json=self.request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"claims": []})
        self.model.verify.assert_not_called()

    def test_non_factual_demo_answer_has_no_claims(self) -> None:
        self.request["generated_answer"] = "Please check the documentation."
        with TestClient(app) as demo_client:
            response = demo_client.post("/verify", json=self.request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"claims": []})

    def test_empty_context_demo_retains_all_claims(self) -> None:
        self.request["context_passages"] = []
        with TestClient(app) as demo_client:
            response = demo_client.post("/verify", json=self.request)

        self.assertEqual(response.status_code, 200)
        claims = response.json()["claims"]
        self.assertEqual(
            [claim["claim_text"] for claim in claims],
            [claim["claim_text"] for claim in self.payload["claims"]],
        )
        self.assertTrue(all(claim["verdict"] == "UNSUPPORTED" for claim in claims))
        self.assertTrue(all(claim["citations"] == [] for claim in claims))

    def test_demo_reports_unavailable_for_unrecognized_answer(self) -> None:
        self.request["generated_answer"] = "SQLite is a database engine."
        with TestClient(app) as demo_client:
            response = demo_client.post("/verify", json=self.request)

        self.assertEqual(response.status_code, 503)

    def test_invalid_request_does_not_call_the_model(self) -> None:
        self.request["context_passages"] = [42]
        response = self.client.post("/verify", json=self.request)
        self.assertEqual(response.status_code, 422)
        self.model.verify.assert_not_called()

    def test_upstream_errors_return_http_errors(self) -> None:
        for error, status in ((ModelError("upstream failed"), 502), (TimeoutError(), 504)):
            with self.subTest(error=error):
                self.model.verify.side_effect = error
                response = self.client.post("/verify", json=self.request)
                self.assertEqual(response.status_code, status)
