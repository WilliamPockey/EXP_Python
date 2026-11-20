import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from baselines import LexicalBaseline, OllamaBaseline
from evaluate import evaluate_case, rescore_alignment
from fourth import CONFIGURATIONS, FULL_FEATURES, FourthVerifier
from fact_evidence import FactEvidenceVerifier
from linked_evidence import LinkedEvidenceVerifier
from ollama_client import DEFAULT_OLLAMA_URL, DEFAULT_TIMEOUT, ModelRun, OllamaClient
from retrieval import DEFAULT_EMBED_MODEL, OllamaEmbeddings, SentenceRetriever
from reviewed_evidence import ReviewedEvidenceVerifier, apply_conflict_check
from scoped_evidence import ScopedEvidenceVerifier, apply_scope_checks
from scoring import summarize
from validate_dataset import EvaluationCase, read_cases
from verifier import Citation, InvalidModelOutput, VerificationModel, VerifyRequest


class PairedOllamaClient(OllamaClient):
    """Share identical completion requests within one ablation run."""

    def __init__(self, model: str, base_url: str, timeout: float) -> None:
        super().__init__(model, base_url, timeout)
        self.cache: dict[tuple[str, str], tuple[object, str | None, dict[str, int]]] = {}
        self.lock = Lock()
        self.cache_hits = 0
        self.live_requests = 0

    def generate(self, prompt: str, schema: dict, run: ModelRun) -> object:
        key = (prompt, json.dumps(schema))
        with self.lock:
            cached = self.cache.get(key)
            if cached is None:
                self.live_requests += 1
            else:
                self.cache_hits += 1
        if cached is not None:
            payload, error, usage = cached
            run.calls += 1
            for field, tokens in usage.items():
                run.usage[field] = run.usage.get(field, 0) + tokens
            if error is not None:
                raise InvalidModelOutput(error)
            return payload

        before = dict(run.usage)
        payload, error = None, None
        try:
            payload = super().generate(prompt, schema, run)
            return payload
        except InvalidModelOutput as failure:
            error = str(failure)
            raise
        finally:
            if payload is not None or error is not None:
                usage = {field: tokens - before.get(field, 0) for field, tokens in run.usage.items()}
                with self.lock:
                    self.cache[key] = (payload, error, usage)


def suite_methods(
    suite: str, client: OllamaClient, retriever: SentenceRetriever,
) -> dict[str, tuple[VerificationModel, frozenset[str]]]:
    methods = {}
    if suite in {"final-baselines", "submission"}:
        methods = {
            "baseline1": (LexicalBaseline(), frozenset()),
            "baseline2": (OllamaBaseline(client.model, client.url.removesuffix("/api/chat"),
                                         client.timeout, client=client), frozenset()),
        }
    if suite in {"ablation", "final-baselines"}:
        return {**methods, **{
            name: (FourthVerifier(client, retriever, features), features)
            for name, features in CONFIGURATIONS.items()
        }}
    no_split = FULL_FEATURES - {"split"}
    if suite == "submission":
        return {**methods, "checked_facts": (FactEvidenceVerifier(client, retriever), no_split)}
    if suite == "alignment":
        return {"checked_facts": (FactEvidenceVerifier(client, retriever), no_split)}
    if suite in {"facts-dev", "facts"}:
        methods = {"scope_checked": (ScopedEvidenceVerifier(client, retriever), no_split)}
        if suite == "facts-dev":
            methods["checked_values"] = (FactEvidenceVerifier(client, retriever, check_gaps=False), no_split)
        methods["checked_facts"] = (FactEvidenceVerifier(client, retriever), no_split)
        return methods
    if suite in {"scope-dev", "scope"}:
        methods = {"guarded_evidence": (ReviewedEvidenceVerifier(client, retriever, check_conflicts=True), no_split)}
        if suite == "scope-dev":
            methods["filtered_claims"] = (ScopedEvidenceVerifier(client, retriever, check_scope=False), no_split)
        methods["scope_checked"] = (ScopedEvidenceVerifier(client, retriever), no_split)
        return methods
    if suite in {"decision-dev", "decision"}:
        methods = {"linked_evidence": (LinkedEvidenceVerifier(client, retriever), no_split)}
        if suite == "decision-dev":
            methods["answer_grounded"] = (ReviewedEvidenceVerifier(client, retriever, review=False), no_split)
            methods["reviewed_evidence"] = (ReviewedEvidenceVerifier(client, retriever), no_split)
        else:
            methods["guarded_evidence"] = (ReviewedEvidenceVerifier(client, retriever, check_conflicts=True), no_split)
        return methods
    if suite == "evidence-dev":
        return {
            "drop_split": (FourthVerifier(client, retriever, no_split), no_split),
            "expanded_sentence_ids": (LinkedEvidenceVerifier(client, retriever, use_bundles=False), no_split),
            "linked_evidence": (LinkedEvidenceVerifier(client, retriever), no_split),
        }
    return {
        "baseline3": (FourthVerifier(client, retriever, frozenset()), frozenset()),
        "baseline4": (FourthVerifier(client, retriever), FULL_FEATURES),
        "linked_evidence": (LinkedEvidenceVerifier(client, retriever), no_split),
    }


def compare_case(case: EvaluationCase, methods: dict[str, tuple[VerificationModel, frozenset[str]]], suite: str) -> dict:
    rows = {name: evaluate_case(case, model) for name, (model, _) in methods.items()}
    if suite == "alignment":
        rows["semantic_alignment"] = rescore_alignment(case, rows["checked_facts"])
    elif suite in {"final-baselines", "submission"}:
        for name, row in rows.items():
            rows[name] = {**rescore_alignment(case, row),
                          "lexical_scoring": {key: row[key] for key in ("matches", "case_correct")}}
    return rows


@dataclass
class RecordedVerifier:
    prediction: dict
    last_run: ModelRun

    def verify(self, request: VerifyRequest) -> object:
        return self.prediction


def recorded_run(row: dict) -> ModelRun:
    retrieval = row["retrieved_evidence"]
    return ModelRun(
        calls=row["model_calls"], usage=row["usage"],
        raw_quotes=(row["raw_quotes"]["valid"], row["raw_quotes"]["total"]),
        repair_attempted=row["repair_attempted"], repair_succeeded=row["repair_succeeded"],
        retrieved_evidence={key: [Citation.model_validate(citation) for citation in sources]
                            for key, sources in retrieval.items()} if retrieval is not None else None,
    )


def replay_conflict_checks(path: Path, model: str, embed_model: str) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("suite"), report.get("llm_model"), report.get("embed_model")) != ("decision-dev", model, embed_model):
        raise ValueError("conflict replay requires this model's completed decision-dev report")
    methods = report["splits"]["dev"]
    baseline = methods["reviewed_evidence"]
    cases = {case.id: case for case in read_cases(Path(__file__).with_name("data") / "dev.jsonl")}
    if [row["id"] for row in baseline["cases"]] != list(cases):
        raise ValueError("recorded report has a different set of dev cases")
    revised = []
    for row in baseline["cases"]:
        if row["status"] != "ok":
            revised.append(row)
            continue
        case = cases[row["id"]]
        prediction = apply_conflict_check(row["prediction"], case.request.context_passages)
        scored = evaluate_case(case, RecordedVerifier(prediction, recorded_run(row)))
        scored["latency_ms"] = row["latency_ms"]
        revised.append(scored)
    methods["guarded_evidence"] = {
        "features": baseline["features"], "run_mode": "deterministic replay of reviewed_evidence",
        "metrics": summarize(revised), "cases": revised,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Scored deterministic conflict checks on {len(revised)} recorded dev cases in {path}")


def replay_scope_checks(path: Path, model: str, embed_model: str) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("suite"), report.get("llm_model"), report.get("embed_model")) != ("scope-dev", model, embed_model):
        raise ValueError("scope replay requires this model's completed scope-dev report")
    methods = report["splits"]["dev"]
    baseline = methods["guarded_evidence"]
    cases = {case.id: case for case in read_cases(Path(__file__).with_name("data") / "dev.jsonl")}
    if [row["id"] for row in baseline["cases"]] != list(cases):
        raise ValueError("recorded report has a different set of dev cases")
    for name, check_scope in (("filtered_claims", False), ("scope_checked", True)):
        revised = []
        for row in baseline["cases"]:
            if row["status"] != "ok":
                revised.append(row)
                continue
            case = cases[row["id"]]
            run = recorded_run(row)
            prediction = apply_scope_checks(row["prediction"], case.request, run, check_scope=check_scope)
            scored = evaluate_case(case, RecordedVerifier(prediction, run))
            scored["latency_ms"] = row["latency_ms"]
            revised.append(scored)
        methods[name] = {"features": baseline["features"], "run_mode": "deterministic replay of guarded_evidence",
                         "metrics": summarize(revised), "cases": revised}
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Scored scope and claim checks on {len(cases)} recorded dev cases in {path}")


def replay_alignment(source: Path, output: Path, split: str, model: str, embed_model: str) -> None:
    report = json.loads(source.read_text(encoding="utf-8"))
    if (report.get("suite"), report.get("llm_model"), report.get("embed_model")) != ("alignment", model, embed_model):
        raise ValueError("alignment replay requires a matching alignment report")
    if split not in report["splits"]:
        raise ValueError("alignment replay requires the requested split in the source report")
    methods = report["splits"][split]
    before = methods["checked_facts"]
    cases = list(read_cases(Path(__file__).with_name("data") / f"{split}.jsonl"))
    if [row["id"] for row in before["cases"]] != [case.id for case in cases]:
        raise ValueError("recorded report has a different set of cases")
    rows = [rescore_alignment(case, row) for case, row in zip(cases, before["cases"], strict=True)]
    methods["semantic_alignment"] = {"features": before["features"], "metrics": summarize(rows), "cases": rows}
    report["splits"] = {split: methods}
    report["replayed_from"] = str(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Rescored {len(rows)} recorded {split} cases in {output}")


def compare_split(
    split: str, configurations: dict[str, tuple[VerificationModel, frozenset[str]]], workers: int, suite: str,
) -> dict:
    data = Path(__file__).with_name("data") / f"{split}.jsonl"
    cases = list(read_cases(data))
    if not cases:
        raise ValueError(f"{split} dataset has no cases")
    completed = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(compare_case, case, configurations, suite): case.id for case in cases}
        for future in as_completed(futures):
            case_id = futures[future]
            completed[case_id] = future.result()
            print(f"{split}: {len(completed)}/{len(cases)} cases complete ({case_id})", flush=True)

    methods = {}
    features_by_method = {name: features for name, (_, features) in configurations.items()}
    if suite == "alignment":
        features_by_method["semantic_alignment"] = features_by_method["checked_facts"]
    for name, features in features_by_method.items():
        rows = [completed[case.id][name] for case in cases]
        metrics = summarize(rows)
        methods[name] = {"features": sorted(features), "metrics": metrics, "cases": rows}
        if suite in {"final-baselines", "submission"}:
            methods[name]["lexical_metrics"] = summarize([{**row, **row["lexical_scoring"]} for row in rows])
        correct = metrics["end_to_end"]["correct_cases"]
        evidence = metrics["evidence_coverage"]
        cross = metrics["cross_passage_evidence"]
        print(
            f"{split} {name}: {correct}/{len(cases)} correct, evidence {evidence['matched']}/{evidence['expected']}, "
            f"cross-passage {cross['cited_with_correct_verdict']}/{cross['expected']}, {metrics['statuses']}",
            flush=True,
        )
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired verification comparisons and ablations.")
    parser.add_argument(
        "--suite", default="ablation",
        choices=("ablation", "evidence-dev", "evidence", "decision-dev", "decision", "scope-dev", "scope",
                 "facts-dev", "facts", "alignment", "final-baselines", "submission"),
    )
    parser.add_argument("--split", choices=("dev", "holdout", "both"), default="both")
    parser.add_argument("--model", required=True)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--workers", type=int, default=1)
    replays = parser.add_mutually_exclusive_group()
    replays.add_argument("--replay-conflicts", action="store_true", help="Rescore a completed decision-dev report without inference")
    replays.add_argument("--replay-scope", action="store_true", help="Rescore a completed scope-dev report without inference")
    replays.add_argument("--alignment-from", type=Path, help="Rescore an existing alignment report without inference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--workers and --timeout must be positive, and timeout must be finite")
    if args.suite.endswith("-dev") and args.split != "dev":
        parser.error(f"{args.suite} is a development-only comparison, use --split dev")
    if args.replay_conflicts:
        if args.suite != "decision-dev":
            parser.error("--replay-conflicts requires --suite decision-dev --split dev")
        replay_conflict_checks(args.output, args.model, args.embed_model)
        return
    if args.replay_scope:
        if args.suite != "scope-dev":
            parser.error("--replay-scope requires --suite scope-dev --split dev")
        replay_scope_checks(args.output, args.model, args.embed_model)
        return
    if args.alignment_from:
        if args.suite != "alignment" or args.split == "both":
            parser.error("--alignment-from requires --suite alignment and one split")
        replay_alignment(args.alignment_from, args.output, args.split, args.model, args.embed_model)
        return

    client = PairedOllamaClient(args.model, args.ollama_url, args.timeout)
    retriever = SentenceRetriever(OllamaEmbeddings(args.embed_model, args.ollama_url, args.timeout))
    configurations = suite_methods(args.suite, client, retriever)
    splits = ("dev", "holdout") if args.split == "both" else (args.split,)
    result = {"suite": args.suite, "llm_model": args.model, "embed_model": args.embed_model, "splits": {}}
    if args.suite == "alignment":
        result["alignment"] = {"checked_facts": "lexical", "semantic_alignment": "semantic"}
    elif args.suite in {"final-baselines", "submission"}:
        result["alignment"] = {"metrics": "semantic", "lexical_metrics": "lexical"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for split in splits:
        result["splits"][split] = compare_split(split, configurations, args.workers, args.suite)
        result["generation_cache"] = {"hits": client.cache_hits, "live_requests": client.live_requests}
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", flush=True)

    for split, methods in result["splits"].items():
        for name, method in methods.items():
            statuses = method["metrics"]["statuses"]
            if not (statuses.get("ok") or statuses.get("invalid_output")):
                raise SystemExit(f"{split} {name}: Ollama returned no evaluable responses")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
