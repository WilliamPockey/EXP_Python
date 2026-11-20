import unittest
from unittest.mock import Mock, patch

from ollama_client import OllamaClient
from reviewed_evidence import ReviewedEvidenceVerifier
from scoped_evidence import ScopedEvidenceVerifier, filter_claims, review_scope
from retrieval import source_sentences
from verifier import VerifyRequest, validate_report


def prediction(text: str, quotes: list[dict], verdict: str = "SUPPORTED") -> dict:
    return {"claim_text": text, "verdict": verdict, "reasoning": "The model picked these passages.", "citations": quotes}


class ScopedEvidenceTests(unittest.TestCase):
    def test_sibling_profile_property_proves_neither_verdict(self) -> None:
        passages = [
            "Larch uses the Gold transport profile for health probes.",
            "The Gold transport profile uses UDP for health probes.",
            "Health probes under the Silver profile time out after four seconds.",
        ]
        sources = source_sentences(passages)
        text = "Larch times health probes out after four seconds."
        result, change = review_scope(prediction(text, [sources[0].model_dump(), sources[1].model_dump()]), sources)
        self.assertEqual((result["verdict"], result["citations"], change), ("UNSUPPORTED", [], "property_only_in_other_scope"))

        supporting = "Under the Gold profile, Larch health probes time out after four seconds."
        scoped_sources = source_sentences([*passages, supporting])
        supported = prediction(text, [scoped_sources[0].model_dump(), scoped_sources[3].model_dump()])
        self.assertEqual(review_scope(supported, scoped_sources), (supported, None))

    def test_binding_applies_only_to_the_named_object(self) -> None:
        sources = source_sentences([
            "Larch uses the Gold profile for health probes.",
            "Under the Silver profile, backup requests time out after four seconds.",
        ])
        claim = prediction("Larch times backup requests out after four seconds.", [sources[1].model_dump()])
        self.assertEqual(review_scope(claim, sources), (claim, None))

    def test_bound_opposition_cites_the_binding_and_correct_policy(self) -> None:
        passages = [
            "The Gale importer is configured to use the Spruce parser.",
            "The Spruce parser always preserves custom attributes in its parsed output.",
            "The Pine parser discards custom attributes.",
        ]
        sources = source_sentences(passages)
        draft = prediction("The parser used by the Gale importer discards custom attributes.", [s.model_dump() for s in sources])
        result, change = review_scope(draft, sources)
        self.assertEqual((result["verdict"], change), ("CONTRADICTED", "opposite_action_in_bound_scope"))
        report = validate_report({"claims": [result]}, passages)
        self.assertEqual(report.claims[0].citations, sources[:2])
        no_binding = {**draft, "citations": draft["citations"][1:]}
        self.assertEqual(review_scope(no_binding, sources), (no_binding, None))

    def test_incompatible_exclusive_values_require_the_same_scope_and_slot(self) -> None:
        sources = source_sentences(["In audit mode, Cobalt emits JSON only.", "In audit mode, Cobalt emits CSV only."])
        draft = prediction("Cobalt emits JSON in audit mode.", [sources[0].model_dump()])
        result, change = review_scope(draft, sources)
        self.assertEqual((result["verdict"], change), ("UNSUPPORTED", "conflicting_same_scope"))
        separated = source_sentences([sources[0].quote, "In batch mode, Cobalt emits CSV only."])
        self.assertEqual(review_scope(draft, separated), (draft, None))

    def test_exclusive_time_matches_possessive_form_without_flattening_roles(self) -> None:
        sources = source_sentences([
            "The only backup of the Yew service is scheduled for 09:30 UTC.",
            "The only backup of the Yew service is scheduled for 11:00 UTC.",
        ])
        text = "The Yew service's only backup is scheduled for 09:30 UTC."
        result, reason = review_scope(prediction(text, [sources[0].model_dump()]), sources)
        self.assertEqual((result["verdict"], reason), ("UNSUPPORTED", "conflicting_same_scope"))

    def test_output_filter_keeps_asserted_conditions_and_distinct_facts(self) -> None:
        answer = "Grove writes logs when cache is disabled. Grove writes traces when cache is enabled."
        drafts = [prediction(text, [], "UNSUPPORTED") for text in (
            "Grove writes logs.", "Grove writes logs when cache is disabled.",
            "When cache is disabled, Grove writes logs.", "Grove writes traces when cache is enabled.",
            "It is unclear whether Grove writes traces.",
        )]
        kept, events = filter_claims(drafts, answer)
        self.assertEqual([index for index, _ in kept], [1, 3])
        self.assertEqual([event["removed"] for event in events],
                         ["lost_original_condition", "duplicate_claim", "unasserted_meta"])

    def test_filter_keeps_the_only_prediction_about_a_conditional_claim(self) -> None:
        draft = prediction("Grove writes logs.", [], "UNSUPPORTED")
        kept, events = filter_claims([draft], "Grove writes logs when cache is disabled.")
        self.assertEqual((kept, events), ([(0, draft)], []))

    def test_filter_reindexes_retrieval_telemetry(self) -> None:
        client = OllamaClient("test", "http://localhost:11434", 5)
        model = ScopedEvidenceVerifier(client, Mock(), check_scope=False)
        records = source_sentences(["Yew adopts a policy.", "Yew stores events."])
        request = VerifyRequest(context_passages=[p.quote for p in records], generated_answer="Yew stores events.")
        def recorded(_request):
            run = client.start_run()
            run.retrieved_evidence = {str(i): [item] for i, item in enumerate(records)}
            return {"claims": [prediction("It is unclear whether Yew adopts a policy.", [], "UNSUPPORTED"),
                               prediction("Yew stores events.", [records[1].model_dump()])]}
        with patch.object(ReviewedEvidenceVerifier, "verify", side_effect=recorded):
            result = validate_report(model.verify(request), request.context_passages)
        self.assertEqual(len(result.claims), 1)
        self.assertEqual(model.last_run.retrieved_evidence, {"0": [records[1]]})
