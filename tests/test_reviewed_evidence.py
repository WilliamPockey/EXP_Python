import io
import json
import unittest
from unittest.mock import Mock, patch

from ollama_client import OllamaClient
from reviewed_evidence import ReviewedEvidenceVerifier, apply_conflict_check, atomic_units, filter_extraction
from verifier import Citation, VerifyRequest, validate_report


def reply(payload: dict) -> io.BytesIO:
    return io.BytesIO(json.dumps({"message": {"content": json.dumps(payload)}}).encode())


class ReviewedEvidenceTests(unittest.TestCase):
    def test_split_carries_the_action_into_each_qualified_object(self) -> None:
        answer = "Quill nodes emit JSON in audit mode and CBOR in batch mode."
        self.assertEqual(atomic_units(answer), [
            "Quill nodes emit JSON in audit mode.", "Quill nodes emit CBOR in batch mode.",
        ])
        ambiguous = "While in safe mode, Quill does not read and write files."
        self.assertEqual(atomic_units(ambiguous), [ambiguous])

    def test_timeout_coordination_restores_the_object_and_preserves_the_unit(self) -> None:
        answer = "Moss sends heartbeats over TCP and times them out after ten seconds."
        self.assertEqual(atomic_units(answer), [
            "Moss sends heartbeats over TCP.", "Moss times heartbeats out after ten seconds.",
        ])

    def test_filter_splits_echoes_and_rejects_unanchored_claims_and_quantities(self) -> None:
        answer = "Moss sends heartbeats over TCP and times them out after ten seconds."
        payload = {"0": [answer, "", "Moss archives corrupted snapshots in a remote directory.",
                         "Moss times heartbeats out after fifty seconds."]}
        self.assertEqual(filter_extraction(payload, {"0": answer}), {
            "0": "Moss sends heartbeats over TCP.", "1": "Moss times heartbeats out after ten seconds.",
        })

    def test_filter_keeps_distinct_numbers_but_removes_a_marked_restatement(self) -> None:
        units = {
            "0": "Pine processes eight jobs in memory.",
            "1": "In other words, Pine processes eight jobs in RAM.",
            "2": "Pine processes nine jobs on disk.",
        }
        payload = {key: [value] for key, value in units.items()}
        self.assertEqual(list(filter_extraction(payload, units).values()), [units["0"], units["2"]])

    def test_source_first_decision_preserves_the_complete_refuting_chain(self) -> None:
        model = ReviewedEvidenceVerifier(OllamaClient("test", "http://localhost:11434", 5), Mock())
        passages = ["Plover uses the Amber profile.", "The Amber profile never keeps warnings."]
        claim = "Plover keeps warnings."
        candidates = {"0": [Citation(index=i, quote=text) for i, text in enumerate(passages)]}
        extraction = reply({"0": [claim]})
        decision = reply({"evidence_ids": ["J1"], "reasoning": "Plover uses Amber, which never keeps warnings.",
                          "comparison": "opposite"})
        with patch.object(model.linked, "retrieve", return_value=candidates):
            with patch("ollama_client.urlopen", side_effect=[extraction, decision]) as send:
                report = validate_report(model.verify(VerifyRequest(context_passages=passages, generated_answer=claim)), passages)
        self.assertEqual(report.claims[0].verdict, "CONTRADICTED")
        self.assertEqual(report.claims[0].citations, candidates["0"])
        schema = json.loads(send.call_args.args[0].data)["format"]
        self.assertEqual(list(schema["properties"]), ["evidence_ids", "reasoning", "comparison"])

    def test_explicit_conflicts_correct_only_exact_scope_and_preserve_citations(self) -> None:
        cases = [
            ("A Maple channel accepts a ninth simultaneous subscriber.",
             "Each Maple channel accepts at most eight simultaneous subscribers.", "CONTRADICTED"),
            ("Grove writes traces even when the cache is disabled.",
             "When the cache is disabled, Grove never writes traces.", "CONTRADICTED"),
            ("Grove writes traces when the cache is enabled.",
             "When the cache is disabled, Grove never writes traces.", "SUPPORTED"),
            ("Quill nodes emit JSON in audit mode.",
             "In audit mode, Quill nodes emit CBOR only.", "CONTRADICTED"),
            ("Quill nodes emit JSON in batch mode.",
             "In audit mode, Quill nodes emit CBOR only.", "SUPPORTED"),
        ]
        for claim, quote, expected in cases:
            with self.subTest(claim=claim, quote=quote):
                original = {"claims": [{"claim_text": claim, "verdict": "SUPPORTED", "reasoning": "Model output.",
                                        "citations": [{"index": 0, "quote": quote}]}]}
                result = apply_conflict_check(original, [quote])
                validated = validate_report(result, [quote]).claims[0]
                self.assertEqual(validated.verdict, expected)
                self.assertEqual(validated.citations, [Citation(index=0, quote=quote)])
                self.assertEqual(original["claims"][0]["verdict"], "SUPPORTED")

    def test_conflicting_source_delegates_the_verdict_to_the_model(self) -> None:
        claim = "Grove writes traces in safe mode."
        negative = "Grove never writes traces in safe mode."
        original = {"claims": [{"claim_text": claim, "verdict": "SUPPORTED", "reasoning": "Model output.",
                                "citations": [{"index": 0, "quote": negative}]}]}
        self.assertEqual(apply_conflict_check(original, [negative, claim]), original)
