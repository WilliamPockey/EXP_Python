from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


Verdict = Literal["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class VerifyRequest(StrictModel):
    context_passages: list[str]
    generated_answer: str


class Citation(StrictModel):
    index: int = Field(ge=0)
    quote: str

    @field_validator("quote")
    @classmethod
    def require_quote_text(cls, quote: str) -> str:
        if not quote.strip():
            raise ValueError("quote must contain text")
        return quote


class Claim(StrictModel):
    claim_text: str
    verdict: Verdict
    reasoning: str
    citations: list[Citation]

    @field_validator("claim_text", "reasoning")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must contain text")
        return value


class VerificationReport(StrictModel):
    claims: list[Claim]


class InvalidModelOutput(ValueError):
    pass


class ModelError(RuntimeError):
    """An upstream model call failed."""


class ModelUnavailable(ModelError):
    """The configured model cannot handle this request."""


class VerificationModel(Protocol):
    def verify(self, request: VerifyRequest) -> object:
        """Return untrusted, decoded JSON. Raise ModelError or TimeoutError on failure."""
        ...


def validate_citation(citation: Citation, context_passages: list[str]) -> None:
    if not 0 <= citation.index < len(context_passages):
        raise InvalidModelOutput("passage index is out of bounds")
    if citation.quote not in context_passages[citation.index]:
        raise InvalidModelOutput("quote is not a verbatim substring")


def count_raw_quotes(payload: object, passages: list[str]) -> tuple[int, int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        return 0, 0
    valid = total = 0
    for claim in payload["claims"]:
        if not isinstance(claim, dict) or not isinstance(claim.get("citations"), list):
            continue
        for entry in claim["citations"]:
            total += 1
            try:
                validate_citation(Citation.model_validate(entry), passages)
            except (ValidationError, InvalidModelOutput):
                continue
            valid += 1
    return valid, total


def validate_report(payload: object, context_passages: list[str]) -> VerificationReport:
    try:
        report = VerificationReport.model_validate(payload)
    except ValidationError as error:
        raise InvalidModelOutput("report does not match the response schema") from error

    for claim_index, claim in enumerate(report.claims):
        if claim.verdict == "UNSUPPORTED":
            if claim.citations:
                raise InvalidModelOutput(f"claim {claim_index}: UNSUPPORTED forbids citations")
        elif not claim.citations:
            raise InvalidModelOutput(f"claim {claim_index}: {claim.verdict} requires citations")

        for citation_index, citation in enumerate(claim.citations):
            try:
                validate_citation(citation, context_passages)
            except InvalidModelOutput as error:
                raise InvalidModelOutput(f"claim {claim_index}, citation {citation_index}: {error}") from error

    return report
