import io
import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from baselines import INPUT_SEPARATOR, LexicalBaseline, OllamaBaseline
from evaluate import evaluate_case
from scoring import summarize
from validate_dataset import read_cases
from verifier import VerifyRequest, validate_report


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "dev.jsonl"
        cls.cases = {case.id: case for case in read_cases(path)}

    def test_lexical_checks_exact_number_and_negation_conflicts(self) -> None:
        request = VerifyRequest(
            context_passages=["Queue capacity is 5 jobs. The queue never persists jobs."],
            generated_answer="Queue capacity is 6 jobs. The queue persists jobs.",
        )
        payload = LexicalBaseline().verify(request)
        report = validate_report(payload, request.context_passages)
        self.assertEqual([claim.verdict for claim in report.claims], ["CONTRADICTED"] * 2)

    def test_ollama_sends_one_request_with_input_only_and_records_usage(self) -> None:
        request = self.cases["dev-11"].request
        model = OllamaBaseline("test", "http://localhost:11434", timeout=5)
        reply = {
            "message": {"content": '{"claims": []}'},
            "prompt_eval_count": 25,
            "eval_count": 7,
        }
        with patch("ollama_client.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as send:
            self.assertEqual(model.verify(request), {"claims": []})

        send.assert_called_once()
        outgoing = send.call_args.args[0]
        self.assertEqual(outgoing.full_url, "http://localhost:11434/api/chat")
        messages = json.loads(outgoing.data)["messages"]
        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertEqual(json.loads(messages[0]["content"].split(INPUT_SEPARATOR, 1)[1]), request.model_dump())
        self.assertEqual(model.last_run.usage, {"prompt_tokens": 25, "completion_tokens": 7})
        self.assertEqual(model.last_run.calls, 1)

    def test_evidence_accepts_alternative_groups_and_requires_all_premises(self) -> None:
        case = self.cases["dev-09"]
        gold = case.expected_claims[0]
        citations = [item.model_dump() for item in gold.evidence_sets[1]]
        prediction = {**gold.model_dump(exclude={"evidence_sets"}), "citations": citations}
        model = Mock(verify=Mock(return_value={"claims": [prediction]}), last_usage={})
        passed = evaluate_case(case, model)
        self.assertTrue(passed["case_correct"])
        self.assertEqual(summarize([passed])["evidence_coverage"]["matched"], 1)

        prediction["citations"] = citations[1:]
        missing_premise = evaluate_case(case, model)
        self.assertFalse(missing_premise["case_correct"])
        self.assertEqual(missing_premise["status"], "ok")
        self.assertFalse(missing_premise["matches"][0]["evidence_covered"])

    def test_duplicate_and_changed_number_do_not_match_extra_gold_claims(self) -> None:
        case = self.cases["dev-03"]
        gold = case.expected_claims[0]
        prediction = {
            **gold.model_dump(exclude={"evidence_sets"}),
            "citations": [item.model_dump() for item in gold.evidence_sets[0]],
        }
        changed_number = {**prediction, "claim_text": gold.claim_text.replace("two-minute", "three-minute")}
        model = Mock(verify=Mock(return_value={"claims": [prediction, deepcopy(prediction), changed_number]}), last_usage={})
        result = evaluate_case(case, model)
        metrics = summarize([result])["claim_extraction"]
        self.assertEqual((metrics["matched"], metrics["predicted"], metrics["expected"]), (1, 3, 2))
        self.assertFalse(result["case_correct"])

    def test_invalid_report_is_scored_as_missed_claims_and_counts_quotes(self) -> None:
        case = self.cases["dev-09"]
        gold = case.expected_claims[0]
        prediction = {
            **gold.model_dump(exclude={"evidence_sets"}),
            "citations": [gold.evidence_sets[0][0].model_dump(), {"index": 1, "quote": "invented"}],
        }
        model = Mock(verify=Mock(return_value={"claims": [prediction]}), last_usage={})
        result = evaluate_case(case, model)
        metrics = summarize([result])
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["raw_quotes"], {"valid": 1, "total": 2})
        self.assertEqual(metrics["claim_extraction"]["expected"], 1)
        self.assertEqual(metrics["claim_extraction"]["matched"], 0)
        self.assertFalse(result["case_correct"])
