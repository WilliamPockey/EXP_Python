import json
import re
from typing import Literal

from pydantic import Field

from explicit_conflict import correct_supported_conflicts
from linked_evidence import EvidenceDecision, LinkedEvidenceVerifier
from ollama_client import ModelRun, OllamaClient
from optimized import FINITE_VERB, NONFACTUAL, SCOPE_MARKER, SECOND_SUBJECT, extract_claims, indexed_schema, split_coordinated_predicates
from retrieval import SentenceRetriever
from text_features import NUMBER_WORDS, WORD_PATTERN, content_words, is_negated, sentences, token_overlap
from verifier import StrictModel, VerifyRequest


COORDINATION = re.compile(r",?\s+and\s+", re.IGNORECASE)
SCOPED_OBJECTS = re.compile(
    r"^(?P<subject>[\w'-]+(?:\s+[\w'-]+){0,7})\s+"
    r"(?P<verb>emits?|uses?|returns?|writes?|accepts?|selects?|sends?|produces?)\s+"
    r"(?P<left>[^,.;]+?\s+(?:in|under|for|when)\s+[^,.;]+?),?\s+and\s+"
    r"(?P<right>[^,.;]+?\s+(?:in|under|for|when)\s+[^,.;]+)[.]?$",
    re.IGNORECASE,
)
TIME_OUT = re.compile(r"\btimes?\s+(?:(?P<object>them|it)\s+)?out\b", re.IGNORECASE)
SENT_OBJECT = re.compile(r"\b(?:sends?|queues?|buffers?|stores?)\s+([\w -]+?)\s+(?:over|via|through|to|using)\b", re.IGNORECASE)
RESTATEMENT = re.compile(r"^(?:in other words|put simply|that is),\s+", re.IGNORECASE)
EXTRACTION_REVIEW = (
    "\nFor each sentence ID, preserve only facts actually asserted in that sentence. Other answer sentences "
    "may resolve an omitted subject or object. Copy wording where possible. Treat repetitions as one claim. "
    "Do not supply related facts, implications, or explanations of the answer. Split coordinated assertions "
    "even if the second shares a subject, object, or verb. Carry the condition of each assertion with it."
    '\nExample: {"0":"Nova sends heartbeats over UDP and times them out after ten seconds."}'
    '\nOutput: {"0":["Nova sends heartbeats over UDP.","Nova times heartbeats out after ten seconds."]}'
)
EVIDENCE_REVIEW = (
    "Read the claim and the supplied source sentences. Select all necessary sentence or bundle IDs. "
    "A bundle expands to all of its member sentences. Derive what the sources actually say about the "
    "subject, joining named entities across sentences when needed. In `reasoning`, state the source fact "
    "first in your own words. Compare its action, polarity, numeric bounds, units, and scope against the claim. "
    "Then choose `same` if the sources entail the entire claim, `opposite` if they establish an incompatible "
    "fact in the same scope, or `undetermined` if a premise is missing or the sources conflict. "
    "Source facts about a different entity or condition do not decide the claim. Use no IDs for undetermined."
    "\nClaim: Beacon allows 45 simultaneous uploads.\nSources: A: Beacon limits parallel uploads to 32."
    '\nAnswer: {"evidence_ids":["A"],"reasoning":"Beacon caps uploads at 32, so 45 simultaneous uploads violate the cap.","comparison":"opposite"}'
    "\nClaim: Yarrow's selected recipe retains warnings.\nSources: A: Yarrow uses the Ruby recipe. "
    "B: Ruby recipes retain warnings. C: Onyx recipes discard warnings. Bundle J1: A, B."
    '\nAnswer: {"evidence_ids":["J1"],"reasoning":"Yarrow uses Ruby and Ruby retains warnings.","comparison":"same"}'
    "\nClaim: Willow audits writes in preview mode.\nSources: A: Willow audits writes in full mode."
    '\nAnswer: {"evidence_ids":[],"reasoning":"The only audit rule given is for full mode, so preview mode is undecided.","comparison":"undetermined"}'
    "\nClaim: Iris forwards rejected receipts.\nSources: A: Iris never forwards rejected receipts."
    '\nAnswer: {"evidence_ids":["A"],"reasoning":"The source says Iris never forwards rejected receipts, directly opposing the claim.","comparison":"opposite"}'
)
RELATION_BY_COMPARISON = {"same": "supports", "opposite": "refutes", "undetermined": "insufficient"}
WRITTEN_QUANTITIES = {
    **NUMBER_WORDS,
    **{word: str(value) for value, word in enumerate(
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), start=13
    )},
    **{word: str(value * 10) for value, word in enumerate(
        "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2
    )},
    **{word: str(value) for value, word in enumerate(
        "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split(), start=1
    ) if word != "second"},
    "hundred": "hundred", "thousand": "thousand", "million": "million", "billion": "billion",
}


class ReviewedDecision(StrictModel):
    evidence_ids: list[str]
    reasoning: str = Field(min_length=1)
    comparison: Literal["same", "opposite", "undetermined"]


def atomic_units(sentence: str) -> list[str]:
    if SCOPE_MARKER.search(sentence):
        return [sentence]
    scoped = SCOPED_OBJECTS.fullmatch(sentence)
    if scoped and not COORDINATION.search(scoped["left"]) and not COORDINATION.search(scoped["right"]):
        prefix = f'{scoped["subject"]} {scoped["verb"]}'
        return [f'{prefix} {scoped["left"]}.', f'{prefix} {scoped["right"].rstrip(".")}.']

    for connector in COORDINATION.finditer(sentence):
        left = sentence[:connector.start()].strip()
        right = sentence[connector.end():].strip()
        timeout = TIME_OUT.match(right)
        left_verb = FINITE_VERB.search(left) or TIME_OUT.search(left)
        if not left_verb or left_verb.start() == 0 or "," in left[:left_verb.start()]:
            continue
        if timeout:
            if timeout["object"]:
                antecedent = SENT_OBJECT.search(left)
                if not antecedent:
                    continue
                start, end = timeout.span("object")
                right = right[:start] + antecedent[1] + right[end:]
            return [left + ".", left[:left_verb.start()] + right]
        if SECOND_SUBJECT.match(right):
            return [left + ".", right]
    return split_coordinated_predicates(sentence)


def lexical_forms(text: str) -> set[str]:
    words = set(content_words(text))
    for word in tuple(words):
        if len(word) > 4 and word.endswith("ies"):
            words.add(word[:-3] + "y")
        elif len(word) > 3 and word.endswith("s"):
            words.add(word[:-1])
        if len(word) > 5 and word.endswith(("ed", "ing")):
            base = word[:-2] if word.endswith("ed") else word[:-3]
            words.update((base, base + "e"))
    return words


def quantities_in(text: str) -> tuple[str, ...]:
    words = WORD_PATTERN.findall(text.casefold())
    quantities = []
    index = 0
    while index < len(words):
        word = words[index]
        value = word if word[0].isdecimal() else WRITTEN_QUANTITIES.get(word)
        if value and value.isdecimal() and int(value) in range(20, 100, 10) and index + 1 < len(words):
            unit = WRITTEN_QUANTITIES.get(words[index + 1], "")
            if unit.isdecimal() and 1 <= int(unit) <= 9:
                value = str(int(value) + int(unit))
                index += 1
        if value is not None:
            quantities.append(value)
        index += 1
    return tuple(quantities)


def filter_extraction(payload: object, answer_units: dict[str, str]) -> dict[str, str]:
    ids = list(answer_units)
    if not isinstance(payload, dict) or set(payload) != set(ids) or any(
        not isinstance(group, list) or any(not isinstance(item, str) for item in group) for group in payload.values()
    ):
        return extract_claims(payload, ids)

    original = " ".join(answer_units.values())
    vocabulary = lexical_forms(original)
    quantities = set(quantities_in(original))
    filtered: dict[str, list[str]] = {key: [] for key in ids}
    earlier: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for key, source in answer_units.items():
        for proposal in payload[key]:
            for sentence in sentences(proposal):
                if sentence.endswith("?") or NONFACTUAL.match(sentence):
                    continue
                for atom in atomic_units(sentence):
                    words = content_words(atom)
                    proposed_quantities = quantities_in(atom)
                    novelty = sum(not (lexical_forms(word) & vocabulary) for word in words)
                    if not words or novelty > max(1, len(words) * 0.40) or not set(proposed_quantities) <= quantities:
                        continue
                    if RESTATEMENT.match(source) and any(
                        proposed_quantities == old_quantities and is_negated(words) == is_negated(old)
                        and token_overlap(words, old) >= 0.65 for old, old_quantities in earlier
                    ):
                        continue
                    filtered[key].append(atom)
                    earlier.append((words, proposed_quantities))
    return extract_claims(filtered, ids)


class ReviewedEvidenceVerifier(LinkedEvidenceVerifier):
    def __init__(
        self, client: OllamaClient, dense: SentenceRetriever, *, review: bool = True, check_conflicts: bool = False,
    ) -> None:
        super().__init__(client, dense)
        self.review = review
        self.check_conflicts = check_conflicts
        self.review_schema = ReviewedDecision.model_json_schema()

    def verify(self, request: VerifyRequest) -> object:
        payload = super().verify(request)
        return apply_conflict_check(payload, request.context_passages) if self.check_conflicts else payload

    def segment(self, answer: str, run: ModelRun) -> dict[str, str]:
        units = []
        for sentence in sentences(answer):
            if not sentence.endswith("?") and not NONFACTUAL.match(sentence):
                units.extend(atomic_units(sentence))
        return {str(index): text for index, text in enumerate(units)}

    def extract(self, answer_sentences: dict[str, str], run: ModelRun) -> dict[str, str]:
        prompt = self.extraction_prompt + EXTRACTION_REVIEW + "\n\nInput JSON:\n" + json.dumps(answer_sentences, ensure_ascii=False)
        raw = self.client.generate(prompt, indexed_schema(list(answer_sentences), self.extraction_schema), run)
        return filter_extraction(raw, answer_sentences)

    def decide(self, data: dict, run: ModelRun) -> EvidenceDecision:
        if not self.review:
            return super().decide(data, run)
        prompt = EVIDENCE_REVIEW + "\n\nInput JSON:\n" + json.dumps(data, ensure_ascii=False)
        decision = ReviewedDecision.model_validate(self.client.generate(prompt, self.review_schema, run))
        return EvidenceDecision(
            evidence_ids=decision.evidence_ids,
            reasoning=decision.reasoning,
            relation=RELATION_BY_COMPARISON[decision.comparison],
        )


def apply_conflict_check(payload: dict, passages: list[str]) -> dict:
    return correct_supported_conflicts(payload, passages, WRITTEN_QUANTITIES)
