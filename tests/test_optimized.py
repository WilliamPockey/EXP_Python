import io
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import create_app
from ollama_client import OllamaClient
from optimized import OptimizedVerifier, split_coordinated_predicates
from verifier import InvalidModelOutput, VerifyRequest, validate_report


def model_reply(payload: object) -> io.BytesIO:
    envelope = {"message": {"content": json.dumps(payload)}, "prompt_eval_count": 10, "eval_count": 5}
    return io.BytesIO(json.dumps(envelope).encode())


def supported(index: int, quote: str) -> dict:
    return {"verdict": "SUPPORTED", "reasoning": "The matching source states the fact.", "citations": [
        {"index": index, "quote": quote},
    ]}


class OptimizedVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = OptimizedVerifier(OllamaClient("test", "http://localhost:11434", 5))
        self.texts = ["Slate creates a manifest.", "Slate retains snapshots for seven days."]
        self.request = VerifyRequest(
            context_passages=self.texts.copy(),
            generated_answer="Slate creates a manifest and retains snapshots for seven days.",
        )

    def test_pre_split_keeps_the_shared_subject_and_leaves_conditions_intact(self) -> None:
        self.assertEqual(split_coordinated_predicates(self.request.generated_answer), self.texts)
        conditioned = "While active, Slate creates a manifest and retains snapshots."
        self.assertEqual(split_coordinated_predicates(conditioned), [conditioned])
        coordinated_objects = "Slate stores logs and snapshots."
        self.assertEqual(split_coordinated_predicates(coordinated_objects), [coordinated_objects])

    def test_batches_claims_without_forwarding_sources_to_extraction(self) -> None:
        replies = [
            model_reply({"0": [self.texts[0]], "1": [self.texts[1], "  SLATE creates a manifest.  "]}),
            model_reply({"1": supported(1, self.texts[1]), "0": supported(0, self.texts[0])}),
        ]
        with patch("ollama_client.urlopen", side_effect=replies) as send:
            payload = self.model.verify(self.request)

        report = validate_report(payload, self.request.context_passages)
        self.assertEqual([claim.claim_text for claim in report.claims], self.texts)
        self.assertEqual(send.call_count, 2)
        first = json.loads(send.call_args_list[0].args[0].data)["messages"][0]["content"]
        self.assertEqual(json.loads(first.split("Input JSON:\n", 1)[1]), dict(zip(("0", "1"), self.texts)))
        self.assertNotIn("context_passages", first)
        batch = json.loads(send.call_args_list[1].args[0].data)["messages"][0]["content"]
        self.assertEqual(json.loads(batch.split("Input JSON:\n", 1)[1])["claims"], dict(zip(("0", "1"), self.texts)))
        self.assertEqual(self.model.last_run.usage, {"prompt_tokens": 20, "completion_tokens": 10})

    def test_repairs_only_the_invalid_claim_once_and_preserves_other_claims(self) -> None:
        replies = [
            model_reply({"0": self.texts[:1], "1": self.texts[1:]}),
            model_reply({"0": supported(0, self.texts[0]), "1": supported(0, self.texts[1])}),
            model_reply({"1": {"verdict": "SUPPORTED", "reasoning": "The retention period is stated.", "evidence_ids": [1]}}),
        ]
        with patch("ollama_client.urlopen", side_effect=replies) as send:
            payload = self.model.verify(self.request)

        report = validate_report(payload, self.request.context_passages)
        self.assertEqual([claim.citations[0].index for claim in report.claims], [0, 1])
        self.assertEqual(send.call_count, 3)
        message = json.loads(send.call_args.args[0].data)["messages"][0]["content"]
        repair_input = json.loads(message.split("Input JSON:\n", 1)[1])
        self.assertEqual(repair_input["claims"], {"1": self.texts[1]})
        self.assertIn("verbatim substring", repair_input["citation_errors"]["1"])
        self.assertEqual(repair_input["evidence_catalog"]["1"], {"index": 1, "quote": self.texts[1]})
        self.assertEqual(self.model.last_run.raw_quotes, (1, 2))
        self.assertTrue(self.model.last_run.repair_succeeded)

    def test_rejects_a_still_invalid_repair_at_the_http_endpoint(self) -> None:
        invalid = {"0": supported(99, self.texts[0])}
        bad_repair = {"0": {"verdict": "SUPPORTED", "reasoning": "The source states the fact.", "evidence_ids": [99]}}
        replies = [model_reply({"0": self.texts[:1], "1": []}), model_reply(invalid), model_reply(bad_repair)]
        with patch("ollama_client.urlopen", side_effect=replies) as send, TestClient(create_app(self.model)) as client:
            response = client.post("/verify", json=self.request.model_dump())

        self.assertEqual(response.status_code, 502)
        self.assertEqual(send.call_count, 3)

    def test_does_not_retry_a_batch_with_missing_claim_ids(self) -> None:
        replies = [model_reply({"0": self.texts[:1], "1": self.texts[1:]}), model_reply({"0": supported(0, self.texts[0])})]
        with patch("ollama_client.urlopen", side_effect=replies) as send:
            with self.assertRaisesRegex(InvalidModelOutput, "exactly the requested claim IDs"):
                self.model.verify(self.request)
        self.assertEqual(send.call_count, 2)
        self.assertFalse(self.model.last_run.repair_attempted)

    def test_empty_context_preserves_claims_and_skips_verification(self) -> None:
        request = self.request.model_copy(update={"context_passages": []})
        with patch("ollama_client.urlopen", return_value=model_reply({"0": self.texts[:1], "1": self.texts[1:]})) as send:
            report = validate_report(self.model.verify(request), [])
        self.assertEqual([claim.verdict for claim in report.claims], ["UNSUPPORTED", "UNSUPPORTED"])
        send.assert_called_once()

    def test_blank_answer_and_nonfactual_extraction_skip_the_batch(self) -> None:
        blank = self.request.model_copy(update={"generated_answer": " \n"})
        advice = self.request.model_copy(update={"generated_answer": "Please review the logs."})
        ambiguous = self.request.model_copy(update={"generated_answer": "Consider an alternative."})
        with patch("ollama_client.urlopen", return_value=model_reply({"0": []})) as send:
            self.assertEqual(self.model.verify(blank), {"claims": []})
            self.assertEqual(self.model.verify(advice), {"claims": []})
            send.assert_not_called()
            self.assertEqual(self.model.verify(ambiguous), {"claims": []})
            send.assert_called_once()

    def test_repair_can_combine_evidence_from_different_passages(self) -> None:
        text = "The Lab exporter uses nightly compaction."
        request = VerifyRequest(
            context_passages=["The Lab exporter uses the Elm profile.", "The Elm profile compacts nightly."],
            generated_answer=text,
        )
        first = {"0": {"verdict": "SUPPORTED", "reasoning": "The two passages establish the schedule.", "citations": []}}
        repair = {"0": {"verdict": "SUPPORTED", "reasoning": "Lab uses Elm, which compacts nightly.", "evidence_ids": [0, 1]}}
        replies = [model_reply({"0": [text]}), model_reply(first), model_reply(repair)]
        with patch("ollama_client.urlopen", side_effect=replies):
            report = validate_report(self.model.verify(request), request.context_passages)
        self.assertEqual([citation.index for citation in report.claims[0].citations], [0, 1])
