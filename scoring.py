from collections import Counter
from math import ceil

from text_features import content_words, is_negated, numbers, token_overlap
from validate_dataset import VERDICTS, GoldClaim
from verifier import Claim


MATCH_THRESHOLD = 0.65


def align_claims(gold: list[GoldClaim], predicted: list[Claim]) -> list[tuple[int, int, float]]:
    gold_words = [content_words(claim.claim_text) for claim in gold]
    predicted_words = [content_words(claim.claim_text) for claim in predicted]
    candidates = []
    for gold_index, expected in enumerate(gold_words):
        for pred_index, actual in enumerate(predicted_words):
            if numbers(expected) != numbers(actual) or is_negated(expected) != is_negated(actual):
                continue
            overlap = token_overlap(expected, actual)
            if overlap >= MATCH_THRESHOLD:
                candidates.append((gold_index, pred_index, overlap))

    used_gold, used_predictions, matches = set(), set(), []
    for gold_index, pred_index, overlap in sorted(candidates, key=lambda item: (-item[2], *item[:2])):
        if gold_index not in used_gold and pred_index not in used_predictions:
            used_gold.add(gold_index)
            used_predictions.add(pred_index)
            matches.append((gold_index, pred_index, overlap))
    return sorted(matches)


def covers_evidence(gold: GoldClaim, predicted: Claim) -> bool:
    return predicted.verdict == gold.verdict and any(
        all(
            any(actual.index == expected.index and expected.quote in actual.quote for actual in predicted.citations)
            for expected in group
        )
        for group in gold.evidence_sets
    )


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(results: list[dict]) -> dict:
    status_counts = Counter(case["status"] for case in results)
    gold_total = sum(len(case["expected_claims"]) for case in results)
    pred_total = sum(len(case["prediction"]["claims"]) for case in results if case["prediction"] is not None)
    matches = [match for case in results for match in case["matches"]]
    gold_verdicts = Counter(match["expected_verdict"] for match in matches)
    predicted_verdicts = Counter(match["predicted_verdict"] for match in matches)
    correct_by_verdict = Counter(
        match["expected_verdict"] for match in matches if match["verdict_correct"]
    )
    by_verdict = {
        verdict: ratio(2 * correct_by_verdict[verdict], gold_verdicts[verdict] + predicted_verdicts[verdict])
        for verdict in VERDICTS
    }
    gold_with_evidence = sum(
        bool(claim["evidence_sets"]) for case in results for claim in case["expected_claims"]
    )
    evidence_covered = sum(match["evidence_covered"] is True for match in matches)
    quotes_valid = sum(case["raw_quotes"]["valid"] for case in results)
    quotes_total = sum(case["raw_quotes"]["total"] for case in results)
    latency = sorted(case["latency_ms"] for case in results)
    repair_attempts = sum(case["repair_attempted"] for case in results)
    repairs_succeeded = sum(case["repair_succeeded"] for case in results)
    usage = Counter()
    for case in results:
        usage.update(case["usage"])

    return {
        "cases": len(results),
        "statuses": dict(status_counts),
        "claim_extraction": {
            "matched": len(matches), "predicted": pred_total, "expected": gold_total,
            "precision": ratio(len(matches), pred_total),
            "recall": ratio(len(matches), gold_total),
            "f1": ratio(2 * len(matches), pred_total + gold_total),
        },
        "verdict_macro_f1_on_matches": sum(by_verdict.values()) / len(VERDICTS),
        "verdict_f1_on_matches": by_verdict,
        "evidence_coverage": {
            "matched": evidence_covered, "expected": gold_with_evidence,
            "rate": ratio(evidence_covered, gold_with_evidence),
        },
        "raw_verbatim_citations": {
            "valid": quotes_valid, "total": quotes_total,
            "rate": quotes_valid / quotes_total if quotes_total else None,
        },
        "end_to_end": {
            "correct_cases": sum(case["case_correct"] for case in results),
            "accuracy": ratio(sum(case["case_correct"] for case in results), len(results)),
        },
        "latency_ms": {
            "p50": latency[ceil(0.50 * len(latency)) - 1],
            "p95": latency[ceil(0.95 * len(latency)) - 1],
        },
        "model_calls": sum(case["model_calls"] for case in results),
        "citation_repair": {
            "attempted": repair_attempts,
            "succeeded": repairs_succeeded,
            "rate": repairs_succeeded / repair_attempts if repair_attempts else None,
        },
        "token_usage": dict(usage),
        "cases_with_token_usage": sum(bool(case["usage"]) for case in results),
    }
