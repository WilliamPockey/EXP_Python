import re
from dataclasses import dataclass
from fractions import Fraction

from explicit_conflict import split_scope
from ollama_client import OllamaClient
from retrieval import SentenceRetriever, source_sentences
from reviewed_evidence import WRITTEN_QUANTITIES, atomic_units
from scoped_evidence import ScopedEvidenceVerifier
from text_features import WORD_PATTERN, sentences
from verifier import Citation, VerifyRequest


FLAGS = re.IGNORECASE
DETERMINERS = frozenset({"a", "an", "the", "each"})
SCHEDULE_SLOT = r"(?:only|single|exactly one)\s+(?P<slot>[\w-]+(?:\s+[\w-]+)*?)"
CLOCK_VALUE = (
    r"\s+(?:is scheduled for|is set for|is at|starts? at|occurs? at|runs? at)"
    r"\s+(?P<clock>(?:[01]\d|2[0-3]):[0-5]\d)(?:\s+(?P<zone>UTC|GMT|[ECMP][SD]T|CET|CEST))?"
)
SCHEDULES = (
    re.compile(rf"(?P<owner>.+?)[’']s\s+{SCHEDULE_SLOT}{CLOCK_VALUE}", FLAGS),
    re.compile(rf"(?:the\s+)?{SCHEDULE_SLOT}\s+of\s+(?P<owner>.+?){CLOCK_VALUE}", FLAGS),
)
TARGET_STATE = re.compile(r"target(?: object)?\s+(?P<state>already exist|exist|is missing|does not exist)", FLAGS)
COPY_ACTION = re.compile(
    r"(?P<actor>.+?)\s+(?P<action>skips? the copy|performs? the copy|cop(?:y|ies) (?:a|an|the) object)", FLAGS
)
UNIT_SECONDS = {
    "ms": Fraction(1, 1000), "millisecond": Fraction(1, 1000),
    "s": 1, "sec": 1, "second": 1, "min": 60, "minute": 60, "h": 3600, "hr": 3600, "hour": 3600,
}
CARDINALS = {word: int(value) for word, value in WRITTEN_QUANTITIES.items()
             if value.isdecimal() and not word.endswith(("st", "rd", "th"))}
DURATION = (
    r"(?P<number>\d+(?:\.\d+)?|[a-z]+(?:[- ][a-z]+)?)"
    r"[ -](?P<unit>milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|hours?|hrs?|h)"
)
DURATION_PATTERNS = (
    ("timeout", "equal", re.compile(rf"(?P<entity>.+?)\s+times? out after {DURATION}", FLAGS)),
    ("timeout", "equal", re.compile(rf"(?P<entity>.+?)\s+has (?:a|an) {DURATION} timeout", FLAGS)),
    ("keepalive", "equal", re.compile(rf"(?P<entity>.+?)\s+sends? (?:a |an )?keepalives? every {DURATION}", FLAGS)),
    ("lease_expiration_age", "equal", re.compile(rf"(?P<entity>.+? lease)\s+expires? at an age of exactly {DURATION}", FLAGS)),
)
EXPIRATION_THRESHOLD = re.compile(rf"(?P<entity>.+? lease)\s+expires? only when its age exceeds {DURATION}", FLAGS)
UNRECOGNIZED_ENTITY = re.compile(r"[,;]|\b(?:not|never|sometimes|may|might|could|can|and|or|except|unless)\b", FLAGS)
WALL_CLOCK = re.compile(r"\b(?:wall[- ]clock(?: (?:time|duration))?|wall time|elapsed (?:time|duration))\b", FLAGS)
CPU_TIME = re.compile(r"\b(?:CPU|processor) time\b", FLAGS)
DIGEST_LENGTH = re.compile(r"\b(?:bits?|bytes?|bit[- ]length|length|size)\b", FLAGS)
DIGEST = re.compile(r"\b(?:sha[- ]?(?:1|2(?:24|56)|384|512)|md5)\b", FLAGS)
TRANSACTION = re.compile(r"\b(?:transactions?|transactional(?:ly)?)\b", FLAGS)
MIGRATION = re.compile(r"\bmigrations?\b", FLAGS)
TRANSACTION_EVIDENCE = re.compile(r"\b(?:transactions?|transactional(?:ly)?|atomic(?:ally)?|commit\w*|roll[- ]?back|roll\w* back)\b", FLAGS)


def entity_words(text: str) -> tuple[str, ...]:
    return tuple(word for word in WORD_PATTERN.findall(text.casefold()) if word not in DETERMINERS)


@dataclass(frozen=True)
class Fact:
    field: str
    entity: tuple[str, ...]
    scope: tuple[str, ...]
    value: str | bool | Fraction
    qualifier: tuple[str, ...] = ()
    relation: str = "equal"

    @property
    def key(self) -> tuple:
        return self.field, self.entity, self.scope, self.qualifier


def parse_schedule(body: str, scope: tuple[str, ...]) -> Fact | None:
    for pattern in SCHEDULES:
        match = pattern.fullmatch(body)
        if match and not UNRECOGNIZED_ENTITY.search(match["owner"]):
            # Keep the field and owner in separate positions across the two phrasings.
            slot = entity_words(match["slot"])
            if slot and not UNRECOGNIZED_ENTITY.search(match["slot"]):
                zone = (match["zone"] or "").upper()
                return Fact("exclusive_schedule", entity_words(match["owner"]), scope, match["clock"], (*slot, zone))
    return None


def parse_copy(body: str, scope: tuple[str, ...]) -> Fact | None:
    # The generic “the copy” refers to the object copy only under this explicit target state.
    state = TARGET_STATE.fullmatch(" ".join(scope))
    match = COPY_ACTION.fullmatch(body)
    if not state or not match or UNRECOGNIZED_ENTITY.search(match["actor"]):
        return None
    actor = entity_words(match["actor"])
    if len(actor) > 1 and actor[-1] == "migrator":
        actor = actor[:-1]
    if actor in {("it",), ("they",)}:
        return None
    target_exists = state["state"] in {"already exist", "exist"}
    # A copy of a named or different object does not match this grammar.
    return Fact("target_copy", actor, ("target_exists" if target_exists else "target_missing",),
                not match["action"].casefold().startswith("skip"))


def duration_seconds(match: re.Match) -> Fraction | None:
    raw = match["number"].casefold()
    if raw[0].isdigit():
        number = Fraction(raw)
    else:
        words = raw.replace("-", " ").split()
        if not all(word in CARDINALS for word in words):
            return None
        amounts = [CARDINALS[word] for word in words]
        if len(amounts) == 2 and not (amounts[0] in range(20, 100, 10) and 1 <= amounts[1] <= 9):
            return None
        number = Fraction(sum(amounts))
    unit = match["unit"].casefold()
    if unit not in UNIT_SECONDS:
        unit = unit.removesuffix("s")
    return number * UNIT_SECONDS[unit]


def duration_fact(field: str, scope: tuple[str, ...], relation: str, match: re.Match) -> Fact | None:
    entity = entity_words(match["entity"])
    value = duration_seconds(match)
    if not entity or entity in {("it",), ("they",)} or UNRECOGNIZED_ENTITY.search(match["entity"]) or value is None:
        return None
    return Fact(field, entity, scope, value, relation=relation)


def parse_fact(text: str) -> Fact | None:
    if threshold := EXPIRATION_THRESHOLD.fullmatch(text.strip().rstrip(".")):
        return duration_fact("lease_expiration_age", (), "greater", threshold)
    body, scope = split_scope(text)
    if fact := parse_schedule(body, scope) or parse_copy(body, scope):
        return fact
    for field, relation, pattern in DURATION_PATTERNS:
        if match := pattern.fullmatch(body):
            return duration_fact(field, scope, relation, match)
    return None


def restore_scope(fact: Fact, answer_facts: list[tuple[str, Fact]]) -> tuple[str, Fact] | None:
    if fact.scope or fact.field not in {"keepalive", "timeout"}:
        return None
    matches = {
        other.scope: (text, other)
        for text, other in answer_facts
        if (other.field, other.entity, other.value, other.relation) == (fact.field, fact.entity, fact.value, fact.relation)
    }
    # A unique scoped answer atom is required. An unscoped original is also an ambiguity.
    if len(matches) == 1 and () not in matches:
        return next(iter(matches.values()))
    return None


def proof_gap(text: str, source: str) -> str | None:
    if WALL_CLOCK.search(text) and CPU_TIME.search(source) and not WALL_CLOCK.search(source):
        return "The CPU time in the source leaves wall-clock measurement unspecified."
    if (digest := DIGEST.search(text)) and DIGEST_LENGTH.search(text):
        same_algorithm = digest[0].casefold().replace("-", "")
        if (any(match[0].casefold().replace("-", "") == same_algorithm for match in DIGEST.finditer(source))
                and not DIGEST_LENGTH.search(source)):
            return "The source names the algorithm and leaves digest length unspecified."
    if TRANSACTION.search(text) and MIGRATION.search(source) and not TRANSACTION_EVIDENCE.search(source):
        return "The migration description leaves transaction boundaries unspecified."
    return None


def compare_fact(fact: Fact, facts: dict[Fact, Citation]) -> tuple[str, str, Citation | None, str] | None:
    candidates = [(other, citation) for other, citation in facts.items() if other.key == fact.key]
    if not candidates or fact.relation != "equal":
        return None
    equal_values = {other.value: citation for other, citation in candidates if other.relation == "equal"}
    excludes_value = [citation for other, citation in candidates
                      if other.relation == "greater" and Fraction(fact.value) <= Fraction(other.value)]
    if len(equal_values) > 1 or (fact.value in equal_values and excludes_value):
        return "UNSUPPORTED", "The source assigns incompatible values to the same field in the same scope.", None, "conflicting_field"
    if fact.field == "exclusive_schedule":
        # This check handles unresolved competing schedules. Leave single-schedule verification to the previous stages.
        return None
    if fact in facts:
        return "SUPPORTED", "The original sentence states the same scoped fact.", facts[fact], "same_fact"
    if excludes_value:
        return "CONTRADICTED", "The source requires a strictly greater age than the asserted expiration age.", excludes_value[0], "duration_threshold"
    if equal_values:
        citation = next(iter(equal_values.values()))
        if fact.field == "target_copy":
            return "CONTRADICTED", "The source states the opposite copy action under the same target condition.", citation, "opposite_action"
        return "CONTRADICTED", "The asserted duration differs from the source duration after unit conversion.", citation, "different_duration"
    return None


def check_facts(report: dict, request: VerifyRequest, *, check_gaps: bool) -> tuple[dict, list[dict]]:
    sources = source_sentences(request.context_passages)
    facts: dict[Fact, Citation] = {}
    for source in sources:
        if fact := parse_fact(source.quote):
            facts.setdefault(fact, source)
    answer_facts = [(atom, fact) for sentence in sentences(request.generated_answer) for atom in atomic_units(sentence)
                    if (fact := parse_fact(atom))]
    full_source = "\n".join(request.context_passages)
    claims, events = [], []
    parsed_claims = [parse_fact(claim["claim_text"]) for claim in report["claims"]]
    for claim, fact in zip(report["claims"], parsed_claims, strict=True):
        original = text = claim["claim_text"]
        result = None
        if fact:
            restored = restore_scope(fact, answer_facts)
            if restored and restored[1] not in parsed_claims:
                text, fact = restored
            result = compare_fact(fact, facts)
        if result is None and check_gaps and (reason := proof_gap(text, full_source)):
            result = "UNSUPPORTED", reason, None, "missing_evidence"
        if result:
            verdict, reason, citation, change = result
            citations = [citation.model_dump()] if citation else []
            # Retain an existing verified citation when it already contains the decisive sentence.
            existing_citations = claim["citations"]
            if verdict != claim["verdict"] or text != original or (citations and citations[0] not in existing_citations):
                change_event = {"claim_text": original, "fact_change": change, "before": claim["verdict"], "after": verdict}
                if text != original:
                    change_event["restored_claim_text"] = text
                events.append(change_event)
                claim = {**claim, "claim_text": text, "verdict": verdict, "reasoning": reason, "citations": citations}
        claims.append(claim)
    return {"claims": claims}, events


class FactEvidenceVerifier(ScopedEvidenceVerifier):
    def __init__(self, client: OllamaClient, retriever: SentenceRetriever, *, check_gaps: bool = True) -> None:
        super().__init__(client, retriever)
        self.check_gaps = check_gaps

    def verify(self, request: VerifyRequest) -> object:
        report = super().verify(request)
        revised, events = check_facts(report, request, check_gaps=self.check_gaps)
        run = self.last_run
        assert run is not None
        run.postprocess_events.extend(events)
        return revised
