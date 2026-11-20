import unittest

from fact_evidence import check_facts, parse_fact
from verifier import VerifyRequest, validate_report


class FactEvidenceTests(unittest.TestCase):
    def review(self, text, passages, *, verdict="SUPPORTED", answer=None, citations=None, check_gaps=True):
        request = VerifyRequest(context_passages=passages, generated_answer=answer or text)
        if citations is None:
            citations = [{"index": 0, "quote": passages[0]}] if verdict != "UNSUPPORTED" else []
        old = {"claim_text": text, "verdict": verdict, "reasoning": "Model judgment.", "citations": citations}
        report, events = check_facts({"claims": [old]}, request, check_gaps=check_gaps)
        validate_report(report, passages)
        return report["claims"][0], events

    def test_only_daily_run_matches_across_possession_phrasings(self):
        text = "The Hearth production job's only daily run is scheduled for 02:00 UTC."
        passages = [
            "The only daily run of the Hearth production job is scheduled for 02:00 UTC.",
            "The only daily run of the Hearth production job is scheduled for 03:00 UTC.",
        ]
        actual, events = self.review(text, passages, verdict="CONTRADICTED")
        self.assertEqual((actual["verdict"], actual["citations"]), ("UNSUPPORTED", []))
        self.assertEqual(events[0]["fact_change"], "conflicting_field")

    def test_exclusive_slot_keeps_owner_field_zone_and_condition_separate(self):
        text = "In audit mode, the Larch job's only daily run is scheduled for 01:00 UTC."
        agreement = [
            "In audit mode, the Larch job's only daily run starts at 01:00 UTC.",
            "The only daily run of the Larch job is set for 01:00 UTC in audit mode.",
        ]
        other_slots = [
            "The only daily run of the Larch job is scheduled for 02:00 UTC in batch mode.",
            "The only daily run of the Cedar job is scheduled for 02:00 UTC in audit mode.",
            "The only weekly run of the Larch job is scheduled for 02:00 UTC in audit mode.",
            "The only daily run of the Larch job is scheduled for 02:00 CET in audit mode.",
            "The Larch job's daily run is scheduled for 02:00 UTC in audit mode.",
        ]
        self.assertEqual(self.review(text, agreement + other_slots)[1], [])
        conflict = "The single daily run of the Larch job starts at 02:00 UTC in audit mode."
        self.assertEqual(self.review(text, [*agreement, conflict])[0]["verdict"], "UNSUPPORTED")

    def test_copy_action_requires_same_target_condition_and_actor(self):
        exists = "If the target object already exists, the Larch migrator skips the copy."
        missing = "If the target object is missing, the Larch migrator performs the copy."
        for claim, index in (
            ("Larch copies an object when the target exists.", 0),
            ("Larch skips the copy if the target does not exist.", 1),
        ):
            with self.subTest(claim=claim):
                actual, events = self.review(claim, [exists, missing])
                self.assertEqual(actual["verdict"], "CONTRADICTED")
                self.assertEqual(actual["citations"], [{"index": index, "quote": [exists, missing][index]}])
                self.assertEqual(events[0]["fact_change"], "opposite_action")
        for claim in (
            "Larch copies an object when the target is missing.",
            "Cedar copies an object when the target already exists.",
            "Larch copies a backup when the target already exists.",
            "Larch copies an object when the target exists and the retry succeeds.",
        ):
            with self.subTest(claim=claim):
                self.assertEqual(self.review(claim, [missing, exists])[1], [])

    def test_copy_action_with_both_source_policies_is_unresolved(self):
        text = "If the target object is missing, Cedar copies an object."
        sources = [text, "Cedar skips the copy when the target is missing."]
        self.assertEqual(self.review(text, sources)[0]["verdict"], "UNSUPPORTED")

    def test_timeout_converts_equal_units_and_keeps_field_identity(self):
        source = "A Larch import attempt has a 120-second timeout."
        actual, events = self.review("Each Larch import attempt times out after two minutes.", [source], verdict="CONTRADICTED")
        self.assertEqual(actual["verdict"], "SUPPORTED")
        self.assertEqual(events[0]["fact_change"], "same_fact")
        q = parse_fact("A Larch import attempt has a 250-millisecond timeout.")
        self.assertEqual(q, parse_fact("A Larch import attempt times out after 0.25 sec."))
        self.assertEqual(parse_fact("Larch sends a keepalive every thirty-five seconds.").value, 35)
        self.assertIsNone(parse_fact("Larch sends a keepalive every thirty sixth seconds."))
        for claim in ("Larch sends a keepalive every two minutes.", "A Cedar import attempt times out after two minutes."):
            self.assertEqual(self.review(claim, [source])[1], [])

    def test_keepalive_restores_only_an_unambiguous_original_scope(self):
        source = "While a lease is active, Larch sends a keepalive every five seconds."
        claim = "Larch sends a keepalive every five minutes."
        scoped = "While the lease is active, Larch sends a keepalive every five minutes."
        other_scope = "While the lease is paused, Larch sends a keepalive every five minutes."
        actual, events = self.review(claim, [source], answer=scoped)
        self.assertEqual((actual["verdict"], actual["claim_text"]), ("CONTRADICTED", scoped))
        self.assertEqual(events[0]["fact_change"], "different_duration")
        for answer in (claim, other_scope, f"{scoped} {other_scope}", f"{scoped} {claim}"):
            self.assertEqual(self.review(claim, [source], answer=answer)[1], [])

    def test_strict_threshold_excludes_boundary_after_conversion(self):
        source = "A Larch lease expires only when its age exceeds 30 minutes."
        actual, events = self.review("A Larch lease expires at an age of exactly 1800 seconds.", [source])
        self.assertEqual(actual["verdict"], "CONTRADICTED")
        self.assertEqual(events[0]["fact_change"], "duration_threshold")
        actual, events = self.review("A Larch lease expires at an age of exactly 31 minutes.", [source], verdict="UNSUPPORTED")
        self.assertEqual((actual["verdict"], events), ("UNSUPPORTED", []))

    def test_missing_evidence_requires_a_gap_across_all_passages(self):
        for claim, related, explicit in (
            ("It never measures wall-clock time.", "The Cedar report lists CPU time.", "Cedar measures elapsed duration."),
            ("An MD5 digest is 128 bits long.", "Cedar hashes the result with MD5.", "MD5 has a 16-byte digest."),
            ("Cedar wraps the migration in a transaction.", "Cedar snapshots the database before each migration.",
             "Cedar runs the migration atomically."),
        ):
            with self.subTest(claim=claim):
                actual, events = self.review(claim, [related], verdict="CONTRADICTED")
                self.assertEqual((actual["verdict"], actual["citations"]), ("UNSUPPORTED", []))
                self.assertEqual(events[0]["fact_change"], "missing_evidence")
                self.assertEqual(self.review(claim, [related, explicit])[1], [])
                self.assertEqual(self.review(claim, [related], check_gaps=False)[1], [])

    def test_other_cross_passage_quotes_are_preserved(self):
        claim = "The active configuration uses HTTP/2."
        passages = ["Larch uses the Cedar profile.", "The Cedar profile uses HTTP/2."]
        references = [{"index": index, "quote": quote} for index, quote in enumerate(passages)]
        actual, events = self.review(claim, passages, citations=references)
        self.assertEqual((actual["citations"], events), (references, []))
