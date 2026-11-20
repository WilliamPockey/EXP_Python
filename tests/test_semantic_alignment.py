import unittest
from pathlib import Path

from evaluate import evaluate_case, rescore_alignment
from scoring import MATCH_THRESHOLD
from semantic_alignment import semantic_candidates
from validate_dataset import read_cases
from verifier import VerificationReport


class SemanticAlignmentTests(unittest.TestCase):
    def align(self, gold: str, prediction: str, answer: str | None = None) -> bool:
        return bool(semantic_candidates([gold], [prediction], answer or prediction, MATCH_THRESHOLD))

    def test_copy_paraphrase_preserves_actor_action_and_condition(self):
        gold = "The Rowan migrator performs the copy when the target object already exists."
        self.assertTrue(self.align(gold, "If the target exists, Rowan copies an object."))
        for prediction in (
            "Rowan copies an object if the target is missing.",
            "Rowan skips the copy if the target already exists.",
            "Cedar copies an object if the target already exists.",
            "Rowan copies a log file if the target already exists.",
            "Rowan copies an object.",
            "Rowan copies an object only if the target already exists.",
            "Rowan copies an object if the target exists and skips the copy if it is missing.",
        ):
            with self.subTest(prediction=prediction):
                self.assertFalse(self.align(gold, prediction))

    def test_explicit_scope_is_required_even_at_high_word_overlap(self):
        gold = "In audit mode, the local Rowan renderer writes compiled templates to disk."
        self.assertTrue(self.align(gold, "The local Rowan renderer writes compiled templates to disk while in audit mode."))
        for prediction in (
            "In batch mode, the local Rowan renderer writes compiled templates to disk.",
            "The local Rowan renderer writes compiled templates to disk.",
            "The local Rowan renderer writes compiled templates to disk in audit mode and in batch mode.",
            "Under the Cedar profile, the local Rowan renderer writes compiled templates to disk.",
        ):
            self.assertFalse(self.align(gold, prediction))
        self.assertFalse(self.align(
            "Under the Cedar transport profile, the Rowan service sends status pings over HTTP/2.",
            "Under the Birch transport profile, the Rowan service sends status pings over HTTP/2.",
        ))

    def test_metric_reference_active_passive_and_local_pronoun(self):
        gold = "A Rowan lease's age is measured from its most recent renewal."
        pronoun = "Its age is measured from the last renewal."
        answer = f"A Rowan lease expires at an age of exactly 45 minutes. {pronoun} While the lease is active, Rowan logs events."
        self.assertTrue(self.align(gold, pronoun, answer))
        self.assertTrue(self.align(gold, "Rowan measures the lease age from the last renewal.", answer))
        self.assertTrue(self.align(gold, "The age of a Rowan lease is measured from the last renewal.", answer))
        self.assertFalse(self.align(gold, pronoun, f"A Cedar lease expires. {pronoun}"))
        self.assertFalse(self.align(gold, pronoun, f"A Rowan lease records a Cedar lease. {pronoun}"))
        self.assertFalse(self.align(gold, pronoun, f"A Rowan lease expires. {pronoun} A Cedar lease expires. {pronoun}"))
        self.assertFalse(self.align(gold, "Rowan measures the lease age from the first renewal.", answer))
        self.assertFalse(self.align(gold, "Rowan measures the lease age from the last renewal.",
                                    f"A Cedar lease expires. {answer}"))

    def test_pronoun_recovery_requires_a_predicate_in_the_answer(self):
        gold = "The Rowan report's duration is measured from the last checkpoint."
        prediction = "Its duration is measured from the last checkpoint."
        self.assertTrue(self.align(gold, prediction, "The Rowan report starts and its duration is measured from the last checkpoint."))
        self.assertFalse(self.align(gold, prediction, "The Rowan report starts. Its duration is measured from the first checkpoint."))
        self.assertFalse(self.align(gold, prediction, "The Rowan report stores Cedar. Its duration is measured from the last checkpoint."))

    def test_temporal_subordinate_clause_does_not_supply_a_competing_main_subject(self):
        gold = "The Rowan index retains metadata for one hour."
        prediction = "It retains metadata for one hour."
        answer = "The Rowan index warms up before the first update is accepted. It retains metadata for one hour."
        self.assertTrue(self.align(gold, prediction, answer))
        self.assertFalse(self.align(gold, prediction, answer.replace("the first update", "Cedar")))

    def test_completed_event_must_match_on_both_sides(self):
        gold = "The Rowan report cannot be edited after archiving."
        self.assertTrue(self.align(gold, "The Rowan report cannot be edited once archived."))
        self.assertFalse(self.align(gold, "The Rowan report cannot be edited before archiving."))
        self.assertFalse(self.align(gold, "The Rowan report cannot be edited once published."))

    def test_loss_event_nominalization_keeps_the_participant_and_timing(self):
        gold = "The Rowan relay restarts after the loss of the primary connection."
        self.assertTrue(self.align(gold, "The Rowan relay restarts if the primary connection is lost."))
        self.assertFalse(self.align(gold, "The Rowan relay restarts if the secondary connection is lost."))
        self.assertFalse(self.align(gold, "The Rowan relay restarts before the loss of the primary connection."))
        self.assertFalse(self.align(gold, "The Rowan relay restarts if the primary connection is not lost."))

    def test_time_quantity_field_and_numeric_equivalence(self):
        gold = "A Rowan import attempt has a two-minute timeout."
        self.assertTrue(self.align(gold, "Each Rowan import attempt times out after 120 seconds."))
        self.assertFalse(self.align(gold, "Each Rowan import attempt times out after two seconds."))
        self.assertFalse(self.align(gold, "Each Rowan import attempt times out after two minutes in audit mode."))

    def test_rescoring_preserves_one_to_one_extraction_and_evidence_validation(self):
        cases = Path(__file__).resolve().parents[1] / "data" / "dev.jsonl"
        case = next(case for case in read_cases(cases) if case.id == "dev-03")
        claims = [{**gold.model_dump(exclude={"evidence_sets"}), "citations": [
            citation.model_dump() for citation in gold.evidence_sets[0]
        ]} for gold in case.expected_claims]
        claims[0]["claim_text"] = "Each Mica import attempt times out after 120 seconds."

        class Recorded:
            def verify(self, _request):
                return VerificationReport.model_validate({"claims": claims}).model_dump()

        before = evaluate_case(case, Recorded())
        after = rescore_alignment(case, before)
        self.assertFalse(before["case_correct"])
        self.assertTrue(after["case_correct"])
        self.assertEqual({key for key in before if before[key] != after[key]}, {"matches", "case_correct"})
        claims.append(claims[0])
        after = rescore_alignment(case, evaluate_case(case, Recorded()))
        self.assertFalse(after["case_correct"])
        self.assertEqual(len(after["matches"]), 2)
        claims.pop()
        claims[0]["citations"] = claims[1]["citations"]
        after = rescore_alignment(case, evaluate_case(case, Recorded()))
        self.assertFalse(after["case_correct"])
        self.assertFalse(after["matches"][0]["evidence_covered"])
