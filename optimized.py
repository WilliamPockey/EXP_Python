import json
import re

from pydantic import TypeAdapter, ValidationError

from ollama_client import ModelRun, OllamaClient
from text_features import content_words, sentences, token_overlap
from verifier import (
    Citation,
    Claim,
    InvalidModelOutput,
    StrictModel,
    Verdict,
    VerifyRequest,
    count_raw_quotes,
    validate_report,
)


EXTRACT_PROMPT = (
    "Extract atomic factual claims for each numbered answer sentence. Return an object with the same "
    "sentence IDs and arrays of standalone facts. Split each independent predicate into its own claim. "
    "Resolve pronouns from the sentence list and preserve quantities, scope, negation and conditions. "
    "Use an empty array for questions, advice and acknowledgements. Include uncertain assertions. "
    'Example input: {"0":"Oak loads four files and skips archived files."} '
    'Example output: {"0":["Oak loads four files.","Oak skips archived files."]}'
)
VERIFY_PROMPT = (
    "Check every supplied claim ID against the numbered source passages. Keep exactly those IDs. "
    "SUPPORTED requires all parts of the claim to follow from the passages. CONTRADICTED requires "
    "an explicit conflict under the same conditions. UNSUPPORTED means missing or unresolved evidence. "
    "Check numbers, units, negation and conditions carefully. When a conclusion needs multiple passages, "
    "cite all the premises. Use the actual zero-based passage index and copy exact contiguous source text "
    "as quote. SUPPORTED and CONTRADICTED require citations. UNSUPPORTED uses an empty citations array."
)
REPAIR_PROMPT = (
    "Repair each supplied claim's citations. Return a verdict, concise reasoning, and evidence_ids "
    "for the same claim IDs. Choose evidence_ids from the numbered evidence catalog. "
    "Include every necessary fragment for multi-passage reasoning. Use SUPPORTED for proof and "
    "CONTRADICTED for explicit refutation. Use UNSUPPORTED with an empty evidence_ids list "
    "when the available fragments do not establish either verdict."
)
NONFACTUAL = re.compile(
    r"^(?:please (?:read|review|check|see|consult|refer to)\b|(?:thanks|thank you)[.!]*$)",
    re.IGNORECASE,
)
CLAIM_LIST = TypeAdapter(list[str])
PREDICATE = (
    r"(?:is|are|has|have|can|may|must|will|reads?|writes?|saves?|loads?|stores?|keeps?|retains?|"
    r"uses?|returns?|includes?|serves?|records?|creates?|rotates?|wraps?|removes?|skips?|sends?|"
    r"accepts?|deletes?|adds?|groups?|expires?|starts?|stops?|copies?|receives?)"
)
FINITE_VERB = re.compile(rf"\b{PREDICATE}\b", re.IGNORECASE)
SECOND_SUBJECT = re.compile(
    rf"^(?:(?:it|they|we|he|she)\s+|(?:the|an?|its|their|this|that)\s+(?:[\w-]+\s+){{1,5}}?){PREDICATE}\b",
    re.IGNORECASE,
)
SCOPE_MARKER = re.compile(
    r"^(?:if|unless|when|while|for|before|after)\b|\b(?:not|never|neither|either|whether|only)\b",
    re.IGNORECASE,
)


class RepairAssessment(StrictModel):
    verdict: Verdict
    reasoning: str
    evidence_ids: list[int]


class AssessmentHeader(StrictModel):
    verdict: Verdict
    reasoning: str
    citations: object


class Assessment(AssessmentHeader):
    citations: list[Citation]


def split_coordinated_predicates(sentence: str) -> list[str]:
    if SCOPE_MARKER.search(sentence):
        return [sentence]
    for connector in re.finditer(r",?\s+and\s+", sentence, flags=re.IGNORECASE):
        left = sentence[:connector.start()].strip()
        right = sentence[connector.end():].strip()
        verb = FINITE_VERB.search(left)
        if not verb or not left[:verb.start()].strip() or "," in left[:verb.start()]:
            continue
        if SECOND_SUBJECT.match(right):
            return [left + ".", right]
        if FINITE_VERB.match(right) and len(right.split()) > 1:
            return [left + ".", left[:verb.start()] + right]
    return [sentence]


def indexed_schema(ids: list[str], entry_schema: dict) -> dict:
    entry = entry_schema.copy()
    definitions = {**entry.pop("$defs", {}), "Entry": entry}
    return {
        "$defs": definitions,
        "type": "object",
        "properties": {entry_id: {"$ref": "#/$defs/Entry"} for entry_id in ids},
        "required": ids,
        "additionalProperties": False,
    }


def repair_entry_schema() -> dict:
    schema = RepairAssessment.model_json_schema()
    properties = schema.pop("properties")
    variants = []
    for verdicts, constraint in (
        (["SUPPORTED", "CONTRADICTED"], {"minItems": 1}),
        (["UNSUPPORTED"], {"maxItems": 0}),
    ):
        variants.append({
            **schema,
            "properties": {
                "verdict": {"type": "string", "enum": verdicts},
                "evidence_ids": {**properties["evidence_ids"], **constraint},
                "reasoning": properties["reasoning"],
            },
        })
    return {"anyOf": variants}


def extract_claims(payload: object, sentence_ids: list[str]) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != set(sentence_ids):
        raise InvalidModelOutput("claim extraction must contain exactly the requested sentence IDs")
    try:
        extracted = [CLAIM_LIST.validate_python(payload[sentence_id], strict=True) for sentence_id in sentence_ids]
    except ValidationError as error:
        raise InvalidModelOutput("claim extraction does not match the response schema") from error

    claims: dict[str, str] = {}
    seen: set[str] = set()
    for group in extracted:
        for text in group:
            if not text.strip():
                raise InvalidModelOutput("claim extraction contains a blank claim")
            for statement in sentences(text):
                normalized = " ".join(statement.casefold().rstrip(".?!").split())
                if normalized not in seen:
                    claims[str(len(claims))] = statement
                    seen.add(normalized)
    return claims


def check_assessments(
    payload: object, claims: dict[str, str], passages: list[str]
) -> tuple[dict[str, Claim], dict[str, str]]:
    if not isinstance(payload, dict):
        raise InvalidModelOutput("the assessment batch must be a JSON object")
    if set(payload) != set(claims):
        raise InvalidModelOutput("the assessment batch must contain exactly the requested claim IDs")

    accepted: dict[str, Claim] = {}
    rejected: dict[str, str] = {}
    for claim_id, text in claims.items():
        item = payload[claim_id]
        if not isinstance(item, dict):
            raise InvalidModelOutput(f"claim {claim_id}: expected an assessment object")
        try:
            header = AssessmentHeader.model_validate({"citations": None, **item})
        except ValidationError as error:
            raise InvalidModelOutput(f"claim {claim_id}: invalid assessment fields") from error
        if not header.reasoning.strip():
            raise InvalidModelOutput(f"claim {claim_id}: reasoning must contain text")
        try:
            report = validate_report({"claims": [{"claim_text": text, **item}]}, passages)
        except InvalidModelOutput as error:
            rejected[claim_id] = str(error).removeprefix("claim 0: ").removeprefix("claim 0, ")
        else:
            accepted[claim_id] = report.claims[0]
    return accepted, rejected


class OptimizedVerifier:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client
        self.extraction_schema = CLAIM_LIST.json_schema()

    @property
    def last_run(self) -> ModelRun | None:
        return self.client.last_run

    def verify(self, request: VerifyRequest) -> object:
        run = self.client.start_run()
        answer_sentences = {
            str(index): sentence
            for index, sentence in enumerate(
                piece
                for original in sentences(request.generated_answer)
                if not original.endswith("?") and not NONFACTUAL.match(original)
                for piece in split_coordinated_predicates(original)
            )
        }
        if not answer_sentences:
            return {"claims": []}

        answer = json.dumps(answer_sentences, ensure_ascii=False)
        sentence_ids = list(answer_sentences)
        extraction = self.client.generate(
            f"{EXTRACT_PROMPT}\n\nInput JSON:\n{answer}", indexed_schema(sentence_ids, self.extraction_schema), run
        )
        claims = extract_claims(extraction, sentence_ids)
        if not claims:
            return {"claims": []}
        passages = request.context_passages
        if not passages:
            return {"claims": [
                {"claim_text": text, "verdict": "UNSUPPORTED", "reasoning": "No source passages were provided.", "citations": []}
                for text in claims.values()
            ]}

        sources = dict(enumerate(passages))
        source_words = [(index, content_words(passage)) for index, passage in sources.items()]
        hints = {}
        for claim_id, claim_text in claims.items():
            words = content_words(claim_text)
            ranked = sorted(source_words, key=lambda source: -token_overlap(words, source[1]))
            hints[claim_id] = [index for index, _ in ranked[:2]]
        batch = {"claims": claims, "likely_relevant_passage_ids": hints, "passages": sources}
        prompt = VERIFY_PROMPT + "\n\nInput JSON:\n" + json.dumps(batch, ensure_ascii=False)
        payload = self.client.generate(prompt, indexed_schema(list(claims), Assessment.model_json_schema()), run)
        if isinstance(payload, dict):
            run.raw_quotes = count_raw_quotes({"claims": list(payload.values())}, passages)
        accepted, rejected = check_assessments(payload, claims, passages)

        if rejected:
            run.repair_attempted = True
            pending = {claim_id: claims[claim_id] for claim_id in rejected}
            accepted.update(self.repair(pending, passages, rejected, run))
            run.repair_succeeded = True

        return {"claims": [accepted[claim_id].model_dump() for claim_id in claims]}

    def repair(
        self, claims: dict[str, str], passages: list[str], issues: dict[str, str], run: ModelRun
    ) -> dict[str, Claim]:
        catalog = [
            {"index": index, "quote": sentence}
            for index, passage in enumerate(passages)
            for sentence in sentences(passage)
        ]
        repair = {"claims": claims, "evidence_catalog": dict(enumerate(catalog)), "citation_errors": issues}
        prompt = REPAIR_PROMPT + "\n\nInput JSON:\n" + json.dumps(repair, ensure_ascii=False)
        payload = self.client.generate(
            prompt, indexed_schema(list(claims), repair_entry_schema()), run
        )
        if not isinstance(payload, dict) or set(payload) != set(claims):
            raise InvalidModelOutput("citation repair must contain exactly the requested claim IDs")
        rebuilt = {}
        for claim_id, item in payload.items():
            try:
                assessment = RepairAssessment.model_validate(item)
            except ValidationError as error:
                raise InvalidModelOutput(f"claim {claim_id}: invalid citation repair fields") from error
            if any(evidence_id < 0 or evidence_id >= len(catalog) for evidence_id in assessment.evidence_ids):
                raise InvalidModelOutput(f"claim {claim_id}: repair evidence ID is out of bounds")
            rebuilt[claim_id] = {
                "verdict": assessment.verdict,
                "reasoning": assessment.reasoning,
                "citations": [catalog[evidence_id] for evidence_id in dict.fromkeys(assessment.evidence_ids)],
            }
        fixed, still_rejected = check_assessments(rebuilt, claims, passages)
        if still_rejected:
            details = "; ".join(f"claim {claim_id}: {reason}" for claim_id, reason in still_rejected.items())
            raise InvalidModelOutput(f"citation repair failed: {details}")
        return fixed
