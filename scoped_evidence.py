import re
from collections import defaultdict

from explicit_conflict import ASSERTED_VALUE, ONLY_VALUE, split_scope
from linked_evidence import name_tokens
from retrieval import SentenceRetriever, source_sentences
from reviewed_evidence import ReviewedEvidenceVerifier, atomic_units
from ollama_client import ModelRun, OllamaClient
from text_features import content_words, sentences
from verifier import Citation, VerifyRequest


REFERENCE = re.compile(
    r"\b(?P<name>[\w-]+)\s+(?:transport\s+|export\s+|runtime\s+|input\s+|output\s+)?"
    r"(?P<kind>profile|parser|mode|preset|policy)\b", re.I,
)
GENERIC = frozenset("a an the new same other second different selected current active default transport export runtime input output".split())
BINDING = re.compile(r"\b(?:uses?|is configured to use|is assigned to|is set to|runs in)(?:\s+the)?\s*$", re.I)
TOPICS = {
    "timeout": re.compile(r"\b(?:time(?:s|d)?\s+(?:(?:them|it)\s+|[\w-]+\s+[\w-]+\s+)?out|timeouts?)\b", re.I),
    "format": re.compile(r"\b(?:emits?|formats?|encod(?:e|es|ed|ing))\b", re.I),
    "retention": re.compile(r"\b(?:keep|keeps|preserv\w+|retain\w+|discard\w+|delet\w+|remov\w+)\b", re.I),
    "compression": re.compile(r"\bcompress\w*\b", re.I),
    "encryption": re.compile(r"\bencrypt\w*\b", re.I),
}
RETENTION = re.compile(r"\b(?P<verb>keep|keeps|preserve|preserves|retain|retains|discard|discards|delete|deletes|remove|removes)\s+(?P<object>[^,.]+)", re.I)
NEUTRAL_OUTPUT = re.compile(r"\s+in\s+(?:its|the)\s+(?:parsed\s+)?output$", re.I)
TIMESTAMP = re.compile(r"\b(?:[01]\d|2[0-3]):[0-5]\d\b")
EXCLUSIVE = re.compile(r"\b(?:only|single|exactly one)\b", re.I)
TIME_POSSESSIVE = re.compile(r"^(?:the\s+)?(.+?)[’']s\s+((?:only|single)\s+.+?)(\s+(?:is|was|starts?|runs?|occurs?)\b.*)$", re.I)
TIME_OF_OWNER = re.compile(r"^(?:the\s+)?((?:only|single)\s+.+?)\s+of\s+(.+?)(\s+(?:is|was|starts?|runs?|occurs?)\b.*)$", re.I)
META = re.compile(r"^(?:it is (?:unclear|uncertain|unknown) whether|the (?:answer|response) (?:claims|says|states|asserts|mentions))\b", re.I)


def references(text: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for match in REFERENCE.finditer(text):
        name = match["name"].casefold()
        if name not in GENERIC:
            result[match["kind"].casefold()].add(name)
    return dict(result)


def context_bindings(claim: str, sources: list[Citation]) -> tuple[dict[str, set[str]], dict[tuple[str, str], Citation]]:
    bindings: dict[tuple[str, str], Citation] = {}
    active = references(claim)
    claim_names = name_tokens(claim)
    claim_words = set(content_words(claim))
    _, claim_condition = split_scope(claim)
    implied: dict[str, set[str]] = defaultdict(set)
    for citation in sources:
        statement, condition = split_scope(citation.quote)
        if condition and condition != claim_condition:
            continue
        for match in REFERENCE.finditer(statement):
            prefix = statement[:match.start()]
            kind, name = match["kind"].casefold(), match["name"].casefold()
            tail = statement[match.end():].strip()
            if tail.casefold().startswith("for ") and not set(content_words(tail[4:])) <= claim_words:
                continue
            if name not in GENERIC and BINDING.search(prefix) and name_tokens(prefix) & claim_names:
                implied[kind].add(name)
                bindings.setdefault((kind, name), citation)
    for kind, names in implied.items():
        if len(names) == 1:
            active.setdefault(kind, names)
    return active, bindings


def out_of_scope(text: str, active: dict[str, set[str]]) -> bool:
    return any(kind in active and len(active[kind]) == 1 and names.isdisjoint(active[kind])
               for kind, names in references(text).items())


def claim_key(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    body, condition = split_scope(text)
    return content_words(body), condition


def filter_claims(claims: list[dict], answer: str) -> tuple[list[tuple[int, dict]], list[dict]]:
    originals = {claim_key(atom) for sentence in sentences(answer) for atom in atomic_units(sentence)}
    surviving_keys = {claim_key(claim["claim_text"]) for claim in claims}
    scoped_bodies = {body for body, condition in originals & surviving_keys if condition}
    source_has_meta = any(META.match(sentence) for sentence in sentences(answer))
    seen = set()
    kept, events = [], []
    for index, claim in enumerate(claims):
        text = claim["claim_text"]
        body, condition = claim_key(text)
        key = (body, condition)
        reason = None
        if META.match(text) and not source_has_meta:
            reason = "unasserted_meta"
        elif key in seen:
            reason = "duplicate_claim"
        elif not condition and key not in originals and body in scoped_bodies:
            reason = "lost_original_condition"
        if reason:
            events.append({"claim_text": text, "removed": reason})
        else:
            seen.add(key)
            kept.append((index, claim))
    return kept, events


def exclusive_slot(text: str, *, source: bool) -> tuple[tuple, str] | None:
    statement, scope = split_scope(text)
    timestamp = TIMESTAMP.search(statement)
    if timestamp and EXCLUSIVE.search(statement):
        scaffold = statement[:timestamp.start()] + " " + statement[timestamp.end():]
        if owner := TIME_POSSESSIVE.match(scaffold):
            key = (content_words(owner[1]), content_words(owner[2]), content_words(owner[3]))
        elif owner := TIME_OF_OWNER.match(scaffold):
            key = (content_words(owner[2]), content_words(owner[1]), content_words(owner[3]))
        else:
            key = (content_words(scaffold),)
        return ("time", key, scope), timestamp[0]
    value = ONLY_VALUE.search(statement) or (None if source else ASSERTED_VALUE.search(statement))
    if value and scope:
        return ("value", content_words(statement[:value.start()]), value[1].casefold(), scope), value[2].casefold()
    return None


def has_conflicting_slot(text: str, sources: list[Citation]) -> bool:
    slot = exclusive_slot(text, source=False)
    if not slot:
        return False
    key, value = slot
    observed = {other[1] for item in sources if (other := exclusive_slot(item.quote, source=True)) and other[0] == key}
    return value in observed and len(observed) > 1


def missing_scope_property(text: str, active: dict[str, set[str]], sources: list[Citation]) -> str | None:
    if not active:
        return None
    topics = [(name, pattern) for name, pattern in TOPICS.items() if pattern.search(text)]
    if len(topics) != 1:
        return None
    topic, pattern = topics[0]
    facts = [item.quote for item in sources if pattern.search(item.quote)]
    if facts and all(out_of_scope(fact, active) for fact in facts):
        return topic
    return None


def retention_action(text: str) -> tuple[bool, tuple[str, ...]] | None:
    match = RETENTION.search(text)
    if not match or re.search(r"\b(?:never|not|may|can|and|or)\b", text, re.I):
        return None
    keeping = match["verb"].casefold().startswith(("keep", "preserv", "retain"))
    return keeping, content_words(NEUTRAL_OUTPUT.sub("", match["object"].strip().rstrip(".")))


def opposing_scope_pair(
    claim: dict, active: dict[str, set[str]], bindings: dict[tuple[str, str], Citation], sources: list[Citation],
) -> list[dict] | None:
    text = claim["claim_text"]
    action = retention_action(text)
    if action is None or not action[1] or split_scope(text)[1]:
        return None
    cited = {(item["index"], item["quote"]) for item in claim["citations"]}
    for (kind, name), binding in bindings.items():
        if active.get(kind) != {name} or (binding.index, binding.quote) not in cited:
            continue
        policies = [item for item in sources if references(item.quote).get(kind) == {name}
                    and not split_scope(item.quote)[1]]
        if any(retention_action(item.quote) == action for item in policies):
            continue
        for item in policies:
            if ((item.index, item.quote) in cited and re.search(r"\balways\b", item.quote, re.I)
                    and retention_action(item.quote) == (not action[0], action[1])):
                return [binding.model_dump(), item.model_dump()]
    return None


def review_scope(claim: dict, sources: list[Citation]) -> tuple[dict, str | None]:
    if claim["verdict"] == "UNSUPPORTED":
        return claim, None
    text = claim["claim_text"]
    if has_conflicting_slot(text, sources):
        return {**claim, "verdict": "UNSUPPORTED", "citations": [],
                "reasoning": "The sources assign different exclusive values to the same subject and condition."}, "conflicting_same_scope"
    active, bindings = context_bindings(text, sources)
    if topic := missing_scope_property(text, active, sources):
        return {**claim, "verdict": "UNSUPPORTED", "citations": [],
                "reasoning": f"The sources describe {topic} only under a different profile or mode than the claim's."}, "property_only_in_other_scope"
    if claim["verdict"] == "SUPPORTED" and (pair := opposing_scope_pair(claim, active, bindings, sources)):
        return {**claim, "verdict": "CONTRADICTED", "citations": pair,
                "reasoning": "The first cited sentence identifies the active configuration. The second states the opposite action for the same object."}, "opposite_action_in_bound_scope"
    return claim, None


def apply_scope_checks(report: dict, request: VerifyRequest, run: ModelRun, *, check_scope: bool) -> dict:
    kept, events = filter_claims(report["claims"], request.generated_answer)
    claims = []
    sources = source_sentences(request.context_passages) if check_scope else []
    for _, claim in kept:
        revised, reason = review_scope(claim, sources) if check_scope else (claim, None)
        if reason:
            events.append({"claim_text": claim["claim_text"], "scope_change": reason,
                           "before": claim["verdict"], "after": revised["verdict"]})
        claims.append(revised)
    run.postprocess_events = events
    if run.retrieved_evidence is not None:
        run.retrieved_evidence = {str(index): run.retrieved_evidence[str(old)] for index, (old, _) in enumerate(kept)}
    return {"claims": claims}


class ScopedEvidenceVerifier(ReviewedEvidenceVerifier):
    def __init__(self, client: OllamaClient, retriever: SentenceRetriever, *, check_scope: bool = True) -> None:
        super().__init__(client, retriever, check_conflicts=True)
        self.check_scope = check_scope

    def verify(self, request: VerifyRequest) -> object:
        report = super().verify(request)
        run = self.last_run
        assert run is not None
        return apply_scope_checks(report, request, run, check_scope=self.check_scope)
