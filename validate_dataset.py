from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from verifier import Citation, StrictModel, VerifyRequest, validate_report


Category = Literal["fully_supported", "distortion", "extrinsic", "multi_passage", "boundary"]
EvidenceSet = Annotated[list[Citation], Field(min_length=1)]
VERDICTS = ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED")
SPLIT_COVERAGE = {
    "dev": Counter(fully_supported=3, distortion=3, extrinsic=2, multi_passage=2, boundary=2),
    "holdout": Counter(fully_supported=4, distortion=4, extrinsic=5, multi_passage=5, boundary=2),
}


class GoldClaim(StrictModel):
    claim_text: str
    verdict: str
    reasoning: str
    evidence_sets: list[EvidenceSet]


class EvaluationCase(StrictModel):
    id: str = Field(pattern=r"^(dev|holdout)-[0-9]{2}$")
    category: Category
    request: VerifyRequest
    expected_claims: list[GoldClaim]


def validate_case(case: EvaluationCase) -> None:
    claims = case.expected_claims
    verdicts = {claim.verdict for claim in claims}
    passages = case.request.context_passages

    if not case.request.generated_answer.strip() and claims:
        raise ValueError("a blank answer must have no claims")
    if not passages and verdicts - {"UNSUPPORTED"}:
        raise ValueError("claims without source passages must be UNSUPPORTED")

    for claim in claims:
        fields = claim.model_dump(exclude={"evidence_sets"})
        for evidence_set in claim.evidence_sets or [[]]:
            citations = [citation.model_dump() for citation in evidence_set]
            validate_report({"claims": [{**fields, "citations": citations}]}, passages)

    if case.category == "fully_supported" and verdicts != {"SUPPORTED"}:
        raise ValueError("fully_supported requires at least one claim and all claims SUPPORTED")
    if case.category == "distortion" and "CONTRADICTED" not in verdicts:
        raise ValueError("distortion requires a CONTRADICTED claim")
    if case.category == "extrinsic" and "UNSUPPORTED" not in verdicts:
        raise ValueError("extrinsic requires an UNSUPPORTED claim")
    if case.category == "multi_passage" and not any(
        claim.evidence_sets
        and all(len({citation.index for citation in group}) >= 2 for group in claim.evidence_sets)
        for claim in claims
    ):
        raise ValueError("multi_passage requires a claim with all evidence sets spanning passages")


def read_cases(path: Path) -> Iterator[EvaluationCase]:
    with path.open(encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            try:
                case = EvaluationCase.model_validate_json(line)
                validate_case(case)
            except ValueError as error:
                raise ValueError(f"{path.name}:{line_number}: {error}") from error
            yield case


def main() -> None:
    data_dir = Path(__file__).with_name("data")
    seen_ids: set[str] = set()
    seen_requests: set[tuple[tuple[str, ...], str]] = set()
    total_claims = 0

    for split, expected_coverage in SPLIT_COVERAGE.items():
        path = data_dir / f"{split}.jsonl"
        coverage: Counter[str] = Counter()
        verdict_counts: Counter[str] = Counter()

        for line_number, case in enumerate(read_cases(path), start=1):
            if not case.id.startswith(f"{split}-") or case.id in seen_ids:
                raise ValueError(f"{path.name}:{line_number}: incorrect split or duplicate id: {case.id}")
            request_key = (tuple(case.request.context_passages), case.request.generated_answer)
            if request_key in seen_requests:
                raise ValueError(f"{path.name}:{line_number}: duplicate request: {case.id}")

            seen_ids.add(case.id)
            seen_requests.add(request_key)
            coverage[case.category] += 1
            verdict_counts.update(claim.verdict for claim in case.expected_claims)

        if coverage != expected_coverage:
            raise ValueError(
                f"{path.name}: expected category counts {dict(expected_coverage)}, "
                f"got {dict(coverage)}"
            )
        if set(verdict_counts) != set(VERDICTS):
            raise ValueError(f"{path.name}: each verdict must appear at least once")

        total_claims += verdict_counts.total()
        summary = ", ".join(f"{verdict}={verdict_counts[verdict]}" for verdict in VERDICTS)
        print(f"{split}: {coverage.total()} cases, {verdict_counts.total()} claims ({summary})")

    print(f"Validated {len(seen_ids)} cases and {total_claims} claims.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
