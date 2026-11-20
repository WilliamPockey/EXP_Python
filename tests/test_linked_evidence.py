import io
import json
import unittest
from unittest.mock import Mock, patch

from linked_evidence import LinkedEvidenceVerifier, LinkedRetriever, linked_bundles
from ollama_client import OllamaClient
from retrieval import SentenceRetriever
from verifier import Citation, VerifyRequest, validate_report


def reply(payload: dict) -> io.BytesIO:
    return io.BytesIO(json.dumps({"message": {"content": json.dumps(payload)}}).encode())


class LinkedEvidenceTests(unittest.TestCase):
    def test_expansion_recovers_a_second_hop_outside_the_dense_shortlist(self) -> None:
        passages = [
            "Rill uses the Amber profile.",
            "The Amber profile stores output in the Gold store.",
            "The Gold store resides in the western cluster.",
            *[f"A different installation {number} stores output in the western cluster." for number in range(9)],
        ]
        dense = Mock(spec=SentenceRetriever)
        dense.retrieve.return_value = {"0": [Citation(index=i, quote=passages[i]) for i in range(3, 9)]}

        result = LinkedRetriever(dense).retrieve({"0": "Rill stores output in the western cluster."}, passages)["0"]

        self.assertTrue({0, 1, 2}.issubset(citation.index for citation in result))
        self.assertLessEqual(len(result), 8)
        self.assertTrue(all(citation.quote == passages[citation.index] for citation in result))
        dense.retrieve.assert_called_once()

    def test_bundles_contain_all_hops_and_keep_an_unrelated_passage_out(self) -> None:
        candidates = {
            "A": Citation(index=0, quote="Rill uses the Amber profile."),
            "B": Citation(index=1, quote="The Amber profile stores output in the Gold store."),
            "C": Citation(index=2, quote="The Gold store resides in the western cluster."),
            "D": Citation(index=3, quote="The Silver store resides in the western cluster."),
        }
        bundles = linked_bundles("Rill stores output in the western cluster.", candidates)
        self.assertEqual(bundles["J1"], ["A", "B", "C"])
        self.assertTrue(all("D" not in bundle for bundle in bundles.values()))

    def test_model_bundle_choice_restores_every_original_quote(self) -> None:
        verifier = LinkedEvidenceVerifier(OllamaClient("test", "http://localhost:11434", 5), Mock())
        passages = ["Rill uses the Amber profile.", "The Amber profile rotates output each hour."]
        answer = "Rill rotates output each hour."
        candidates = {"0": [Citation(index=i, quote=quote) for i, quote in enumerate(passages)]}
        replies = [reply({"0": [answer]}), reply({
            "relation": "supports", "evidence_ids": ["J1"], "reasoning": "Rill uses Amber, which rotates hourly.",
        })]
        with patch.object(verifier.linked, "retrieve", return_value=candidates):
            with patch("ollama_client.urlopen", side_effect=replies) as send:
                report = verifier.verify(VerifyRequest(context_passages=passages, generated_answer=answer))

        validated = validate_report(report, passages)
        self.assertEqual(validated.claims[0].citations, candidates["0"])
        self.assertEqual(verifier.last_run.retrieved_evidence, candidates)
        self.assertFalse(verifier.last_run.repair_attempted)
        self.assertEqual(send.call_count, 2)

    def test_unknown_evidence_id_requests_only_the_existing_single_repair(self) -> None:
        verifier = LinkedEvidenceVerifier(OllamaClient("test", "http://localhost:11434", 5), Mock())
        answer = "Rill rotates output each hour."
        candidates = {"0": [Citation(index=0, quote=answer)]}
        replies = [reply({"0": [answer]}), reply({
            "relation": "supports", "evidence_ids": ["missing"], "reasoning": "A citation is required.",
        }), reply({"0": {"verdict": "SUPPORTED", "reasoning": "The source states this.", "evidence_ids": [0]}})]
        with patch.object(verifier.linked, "retrieve", return_value=candidates):
            with patch("ollama_client.urlopen", side_effect=replies) as send:
                report = verifier.verify(VerifyRequest(context_passages=[answer], generated_answer=answer))

        self.assertEqual(validate_report(report, [answer]).claims[0].citations, candidates["0"])
        self.assertTrue(verifier.last_run.repair_succeeded)
        self.assertEqual(send.call_count, 3)
