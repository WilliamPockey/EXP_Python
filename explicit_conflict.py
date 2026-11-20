import re

from text_features import WORD_PATTERN, sentences


BOUND = re.compile(r"\b(?P<kind>at most|at least|no more than|no fewer than|up to)\s+(?P<number>[\w.]+)\b", re.I)
PREFIX_SCOPE = re.compile(r"^\s*(?:even\s+)?(?:if|when|while|in)\s+([^,]+),\s*(.+)$", re.I)
SUFFIX_SCOPE = re.compile(r"\s+(?:even\s+)?(?:if|when|while)\s+(.+)$", re.I)
MODE_SCOPE = re.compile(r"\s+in\s+([\w-]+(?:\s+[\w-]+){0,2}\s+mode)$", re.I)
NEGATIVE = re.compile(r"\b(?:never|does not|do not)\b", re.I)
AMBIGUOUS = re.compile(r"\b(?:and|or|neither|unless|except|only if|not only)\b", re.I)
MODAL = re.compile(r"\b(?:may|might|can|could|possibly|sometimes)\b", re.I)
ONLY_VALUE = re.compile(r"\b(emit|use|return|produce|select)s?\s+([\w/-]+)\s+only$", re.I)
ASSERTED_VALUE = re.compile(r"\b(emit|use|return|produce|select)s?\s+([\w/-]+)$", re.I)
DEFAULT_NOISE = frozenset("a an the each every in".split())
ALIASES = {"includes": "contain", "include": "contain", "contains": "contain"}


def normalized(text: str) -> tuple[str, ...]:
    words = []
    for word in WORD_PATTERN.findall(text.casefold()):
        if word in DEFAULT_NOISE:
            continue
        singular = word[:-1] if len(word) > 4 and word.endswith("s") else word
        words.append(ALIASES.get(word, singular))
    return tuple(words)


def split_scope(text: str) -> tuple[str, tuple[str, ...]]:
    statement = text.strip().rstrip(".")
    prefix = PREFIX_SCOPE.match(statement)
    if prefix:
        return prefix[2], normalized(prefix[1])
    suffix = SUFFIX_SCOPE.search(statement) or MODE_SCOPE.search(statement)
    if suffix:
        return statement[:suffix.start()], normalized(suffix[1])
    return statement, ()


def explicit_conflict(claim: str, quote: str, number_words: dict[str, str]) -> str | None:
    statement, claim_scope = split_scope(claim)
    source, source_scope = split_scope(quote)
    if claim_scope != source_scope or AMBIGUOUS.search(statement) or AMBIGUOUS.search(source):
        return None

    claim_negated, source_negated = bool(NEGATIVE.search(statement)), bool(NEGATIVE.search(source))
    if (claim_negated != source_negated and not MODAL.search(statement) and not MODAL.search(source)
            and normalized(NEGATIVE.sub("", statement)) == normalized(NEGATIVE.sub("", source))):
        return f"The cited source states the opposite for the same subject and condition: {quote}"
    if claim_negated or source_negated or re.search(r"\b(?:not|no)\b", statement, re.I):
        return None

    bound = BOUND.search(source)
    if bound:
        limit = number_words.get(bound["number"].casefold(), bound["number"])
        if not limit.isdecimal():
            return None
        for token in WORD_PATTERN.finditer(statement):
            value = number_words.get(token[0].casefold(), token[0])
            if not value.isdecimal() or not normalized(statement[token.end():]):
                continue
            same_measure = (
                normalized(statement[:token.start()]) == normalized(source[:bound.start()])
                and normalized(statement[token.end():]) == normalized(source[bound.end():])
            )
            upper_bound = bound["kind"].casefold() in {"at most", "no more than", "up to"}
            out_of_range = int(value) > int(limit) if upper_bound else int(value) < int(limit)
            if same_measure and out_of_range:
                return f"The cited limit is {bound[0]}, which excludes the asserted quantity {value}."

    asserted = ASSERTED_VALUE.search(statement)
    exclusive = ONLY_VALUE.search(source)
    if asserted and exclusive and claim_scope and (
        normalized(statement[:asserted.start()]) == normalized(source[:exclusive.start()])
        and asserted[1].casefold() == exclusive[1].casefold()
        and asserted[2].casefold() != exclusive[2].casefold()
    ):
        return f"For the same subject and condition, the cited source allows only {exclusive[2]}, not {asserted[2]}."
    return None


def correct_supported_conflicts(report: dict, passages: list[str], number_words: dict[str, str]) -> dict:
    source_facts = {
        (normalized(statement), scope)
        for passage in passages
        for chunk in sentences(passage)
        for statement, scope in [split_scope(chunk)]
    }
    corrected = []
    for claim in report["claims"]:
        statement, scope = split_scope(claim["claim_text"])
        if claim["verdict"] == "SUPPORTED" and (normalized(statement), scope) not in source_facts:
            for citation in claim["citations"]:
                reason = explicit_conflict(claim["claim_text"], citation["quote"], number_words)
                if reason:
                    claim = {**claim, "verdict": "CONTRADICTED", "reasoning": reason}
                    break
        corrected.append(claim)
    return {"claims": corrected}
