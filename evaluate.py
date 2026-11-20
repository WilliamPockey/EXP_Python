import argparse
import json
import math
from pathlib import Path
from time import perf_counter

from baselines import LexicalBaseline, OllamaBaseline
from ollama_client import DEFAULT_OLLAMA_URL, DEFAULT_TIMEOUT, ModelRun, OllamaClient
from optimized import OptimizedVerifier
from scoring import align_claims, covers_evidence, summarize
from validate_dataset import EvaluationCase, read_cases
from verifier import InvalidModelOutput, ModelError, VerificationModel, count_raw_quotes, validate_report


def evaluate_case(case: EvaluationCase, model: VerificationModel) -> dict:
    start = perf_counter()
    report = None
    status, error = "ok", None
    quotes_valid = quotes_total = 0
    try:
        payload = model.verify(case.request)
        quotes_valid, quotes_total = count_raw_quotes(payload, case.request.context_passages)
        report = validate_report(payload, case.request.context_passages)
    except InvalidModelOutput as failure:
        status, error = "invalid_output", str(failure)
    except TimeoutError as failure:
        status, error = "timeout", str(failure)
    except ModelError as failure:
        status, error = "upstream_error", str(failure)
    latency_ms = (perf_counter() - start) * 1000
    run = getattr(model, "last_run", None)
    if not isinstance(run, ModelRun):
        run = ModelRun()
    if run.raw_quotes is not None:
        quotes_valid, quotes_total = run.raw_quotes

    gold = case.expected_claims
    predicted = report.claims if report is not None else []
    matches = []
    for gold_index, pred_index, similarity in align_claims(gold, predicted):
        expected, actual = gold[gold_index], predicted[pred_index]
        matches.append({
            "expected_index": gold_index,
            "predicted_index": pred_index,
            "text_overlap": similarity,
            "expected_verdict": expected.verdict,
            "predicted_verdict": actual.verdict,
            "verdict_correct": expected.verdict == actual.verdict,
            "evidence_covered": covers_evidence(expected, actual) if expected.evidence_sets else None,
        })
    case_correct = (
        status == "ok"
        and len(matches) == len(gold) == len(predicted)
        and all(match["verdict_correct"] and match["evidence_covered"] is not False for match in matches)
    )

    return {
        "id": case.id,
        "category": case.category,
        "status": status,
        "error": error,
        "latency_ms": round(latency_ms, 3),
        "model_calls": run.calls,
        "usage": run.usage,
        "repair_attempted": run.repair_attempted,
        "repair_succeeded": run.repair_succeeded,
        "raw_quotes": {"valid": quotes_valid, "total": quotes_total},
        "expected_claims": [claim.model_dump() for claim in gold],
        "prediction": report.model_dump() if report is not None else None,
        "matches": matches,
        "case_correct": case_correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare grounding verifiers on the labeled dataset.")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument(
        "--method", choices=("lexical", "llm", "optimized", "baselines", "all"), default="lexical"
    )
    parser.add_argument("--model", help="Ollama model tag, required for methods that call Ollama")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Ollama socket timeout in seconds")
    parser.add_argument("--output", type=Path, help="Write JSON summaries and per-case diagnostics")
    args = parser.parse_args()
    if args.method != "lexical" and not args.model:
        parser.error("--model is required for methods that call Ollama")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")

    cases = list(read_cases(Path(__file__).with_name("data") / f"{args.split}.jsonl"))
    if not cases:
        raise ValueError(f"{args.split} dataset has no cases")
    methods: dict[str, VerificationModel] = {}
    if args.method in {"lexical", "baselines", "all"}:
        methods["lexical"] = LexicalBaseline()
    if args.method in {"llm", "baselines", "all"}:
        methods["llm"] = OllamaBaseline(args.model, args.ollama_url, args.timeout)
    if args.method in {"optimized", "all"}:
        client = OllamaClient(args.model, args.ollama_url, args.timeout)
        methods["optimized"] = OptimizedVerifier(client)

    evaluation = {"split": args.split, "llm_model": args.model if args.method != "lexical" else None, "methods": {}}
    for name, model in methods.items():
        rows = [evaluate_case(case, model) for case in cases]
        metrics = summarize(rows)
        evaluation["methods"][name] = {"metrics": metrics, "cases": rows}
        print(f"{name} ({args.split})\n{json.dumps(metrics, indent=2)}", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")

    for name in methods.keys() - {"lexical"}:
        statuses = evaluation["methods"][name]["metrics"]["statuses"]
        if not (statuses.get("ok") or statuses.get("invalid_output")):
            raise SystemExit(f"{name}: Ollama returned no evaluable responses. Check the server and model.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
