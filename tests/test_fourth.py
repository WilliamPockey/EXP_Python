import io
import json
import unittest
from math import sqrt
from unittest.mock import Mock, patch

from evaluate_ablation import PairedOllamaClient
from fourth import CONFIGURATIONS, FEATURES, FourthVerifier
from ollama_client import OllamaClient
from optimized import OptimizedVerifier
from retrieval import SentenceRetriever
from verifier import Citation, VerifyRequest, validate_report


def reply(payload: dict) -> io.BytesIO:
    return io.BytesIO(json.dumps({
        "message": {"content": json.dumps(payload)}, "prompt_eval_count": 6, "eval_count": 4,
    }).encode())


def supported(index: int, quote: str) -> dict:
    return {"verdict": "SUPPORTED", "reasoning": "The sentence states the claim.",
            "citations": [{"index": index, "quote": quote}]}


def call_input(call) -> dict:
    prompt = json.loads(call.args[0].data)["messages"][0]["content"]
    return json.loads(prompt.split("Input JSON:\n", 1)[1])


class FourthVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OllamaClient("test", "http://localhost:11434", 5)
        self.retriever = Mock(spec=SentenceRetriever)
        self.texts = ["Cypress packs files.", "Cypress signs the manifest."]
        self.request = VerifyRequest(context_passages=self.texts, generated_answer="Cypress packs files and signs the manifest.")

    def test_matrix_contains_exactly_the_two_anchors_and_eight_single_change_runs(self) -> None:
        self.assertEqual(len(CONFIGURATIONS), 10)
        self.assertEqual(len(set(CONFIGURATIONS.values())), 10)
        for feature in FEATURES:
            self.assertEqual(CONFIGURATIONS[f"add_{feature}"] - CONFIGURATIONS["baseline3"], {feature})
            self.assertEqual(CONFIGURATIONS["baseline4"] - CONFIGURATIONS[f"drop_{feature}"], {feature})

    def test_disabled_switches_send_the_original_model_requests(self) -> None:
        single = self.request.model_copy(update={"generated_answer": self.texts[0]})
        generation_requests = []
        models = [OptimizedVerifier(self.client), FourthVerifier(self.client, self.retriever, frozenset())]
        for model in models:
            with patch("ollama_client.urlopen", side_effect=[
                reply({"0": self.texts[:1]}), reply({"0": supported(0, self.texts[0])}),
            ]) as send:
                model.verify(single)
                generation_requests.append([call.args[0].data for call in send.call_args_list])
        self.assertEqual(*generation_requests)
        self.retriever.retrieve.assert_not_called()

    def test_full_pipeline_splits_then_extracts_then_verifies_claims_individually(self) -> None:
        self.retriever.retrieve.return_value = {
            str(index): [Citation(index=index, quote=text)] for index, text in enumerate(self.texts)
        }
        responses = [
            reply({"sentences": self.texts}), reply({"0": self.texts[:1], "1": self.texts[1:]}),
            reply({"0": supported(0, self.texts[0])}), reply({"1": supported(1, self.texts[1])}),
        ]
        with patch("ollama_client.urlopen", side_effect=responses) as send:
            payload = FourthVerifier(self.client, self.retriever).verify(self.request)
        self.assertEqual([claim.claim_text for claim in validate_report(payload, self.texts).claims], self.texts)
        self.assertEqual(send.call_count, 4)
        self.assertEqual(call_input(send.call_args_list[0]), {"answer": self.request.generated_answer})
        self.assertIn(b"Delta relay", send.call_args_list[1].args[0].data)
        for index, call in enumerate(send.call_args_list[2:]):
            key = str(index)
            self.assertEqual(call_input(call)["claims"], {key: self.texts[index]})
            self.assertEqual(call_input(call)["evidence_sentences"], {
                key: [{"index": index, "quote": self.texts[index]}],
            })

    def test_out_of_retrieval_citation_gets_one_repair_using_the_original_catalog(self) -> None:
        self.retriever.retrieve.return_value = {"0": [Citation(index=0, quote=self.texts[0])]}
        single = self.request.model_copy(update={"generated_answer": self.texts[0]})
        responses = [reply({"0": self.texts[:1]}), reply({"0": supported(1, self.texts[1])}), reply({
            "0": {"verdict": "SUPPORTED", "reasoning": "The first sentence states it.", "evidence_ids": [0]},
        })]
        with patch("ollama_client.urlopen", side_effect=responses) as send:
            payload = FourthVerifier(self.client, self.retriever, frozenset({"rag"})).verify(single)
        self.assertEqual(validate_report(payload, self.texts).claims[0].citations[0].index, 0)
        self.assertEqual(send.call_count, 3)
        repair = call_input(send.call_args)
        self.assertEqual(len(repair["evidence_catalog"]), 2)
        self.assertIn("outside the retrieved evidence", repair["citation_errors"]["0"])

    def test_dense_shortlist_is_reranked_and_returns_three_verbatim_sentences(self) -> None:
        chunks = ["Primer.", "Notes.", "Manual.", "Queue holds four jobs.", "Summary.",
                  "Intro.", "Queue holds four jobs in memory."]
        cosines = [1, 0.95, 0.9, 0.75, 0.85, 0.8, 0.1]
        vectors = [(score, sqrt(1 - score * score)) for score in cosines] + [(1, 0)]
        embeddings = Mock(encode=Mock(return_value=vectors))
        result = SentenceRetriever(embeddings).retrieve({"0": chunks[3]}, chunks)["0"]
        self.assertEqual([item.index for item in result], [3, 0, 1])
        self.assertEqual([item.quote for item in result], [chunks[3], chunks[0], chunks[1]])

    def test_paired_completions_reuse_output_and_keep_logical_calls(self) -> None:
        client = PairedOllamaClient("test", "http://localhost:11434", 5)
        with patch("ollama_client.urlopen", return_value=reply({"sentences": []})) as send:
            first = client.start_run()
            second = client.start_run()
            self.assertEqual(client.generate("prompt", {}, first), client.generate("prompt", {}, second))
        send.assert_called_once()
        self.assertEqual((client.live_requests, client.cache_hits), (1, 1))
        for run in (first, second):
            self.assertEqual(run.calls, 1)
            self.assertEqual(run.usage, {"prompt_tokens": 6, "completion_tokens": 4})
