import re
from dataclasses import dataclass

from explicit_conflict import normalized, split_scope
from fact_evidence import Fact, entity_words, parse_fact
from optimized import PREDICATE
from reviewed_evidence import atomic_units
from text_features import content_words, is_negated, numbers, token_overlap


CLAUSE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+|,\s+(?:and|but)\s+|\s+(?:and|but)\s+(?=it(?:s)?\s)", re.I)
SUBJECT_VERB = re.compile(rf"\b(?:{PREDICATE}|warms? up|reports|reported|measures|measured)\b", re.I)
HEAD_PRONOUN = re.compile(r"^(it|its)\b", re.I)
AMBIGUOUS_SUBJECT = re.compile(r"\b(?:and|or|it|its|they|this|that)\b|,", re.I)
SCOPED_START = re.compile(r"^(?:if|when|while|unless|in|under|before|after|during)\b", re.I)
OTHER_REFERENT = re.compile(r"\b(?:the|an?)\s+(?!age\s+of\b)", re.I)
NAMED_OBJECT = re.compile(r"\b[A-Z][a-z]+\b")
METRIC = r"(?:age|duration|latency)"
MEASURED_FROM = re.compile(rf"(?P<owner>.+?)[’']s\s+(?P<metric>{METRIC}) is measured from (?P<origin>.+)", re.I)
MEASUREMENT_OF = re.compile(rf"(?:the\s+)?(?P<metric>{METRIC}) of (?P<owner>.+?) is measured from (?P<origin>.+)", re.I)
ACTIVE_MEASURE = re.compile(
    rf"(?P<actor>.+?) measures? (?:the )?(?P<object>[\w-]+) (?P<metric>{METRIC}) from (?P<origin>.+)", re.I,
)
CONFIGURATION = r"(?:in|under)\s+(?P<condition>(?:[\w-]+\s+){1,3}(?:mode|profile|preset|policy))"
CONFIG_SCOPE = re.compile(rf"\b{CONFIGURATION}\b", re.I)
SCOPE_WORDS = re.compile(rf"\b(?:if|when|while|unless|except|before|after|once|during|{CONFIGURATION})\b", re.I)
WHILE_CONFIG = re.compile(rf"\bwhile\s+(?={CONFIGURATION}\b)", re.I)
TIME_SCOPE = re.compile(r"\s+(?P<marker>before|after|once|during|unless|except)\s+(?P<condition>.+)$", re.I)
COMPLETION = {"after": re.compile(r"([a-z]{3,})ing", re.I), "once": re.compile(r"([a-z]{3,})ed", re.I)}
LOSS_EVENT = re.compile(
    r"(?:\b(?:if|when)\s+(?P<passive>[^,]+?)\s+(?:is|are) lost|\bafter\s+(?:the\s+)?loss of (?P<nominal>[^,]+))(?=,|$)",
    re.I,
)
LIMITERS = re.compile(r"\b(?:only if|only when|only while|only|at most|at least|exactly|no more than|no fewer than)\b", re.I)
NEGATIVE = re.compile(r"\b(?:not|no|never|cannot|without|doesn't|don't|can't)\b", re.I)
UNIT = re.compile(r"\b(?:milliseconds?|seconds?|minutes?|hours?|days?|bytes?|bits?|percent)\b", re.I)


def named_subject(clause: str) -> str | None:
    if SCOPED_START.match(clause):
        return None
    verb = SUBJECT_VERB.search(clause)
    if not verb:
        return None
    subject = clause[:verb.start()].strip()
    if subject and any(word[:1].isupper() for word in subject.split() if word.casefold() not in {"a", "an", "the"}):
        if not AMBIGUOUS_SUBJECT.search(subject):
            return subject
    return None


def resolve_pronoun(text: str, clauses: list[str]) -> str | None:
    pronoun = HEAD_PRONOUN.match(text)
    if not pronoun:
        return text
    tail = content_words(text[pronoun.end():])
    candidates = set()
    for previous, clause in zip(clauses, clauses[1:]):
        head = HEAD_PRONOUN.match(clause)
        if not head or head[0].casefold() != pronoun[0].casefold():
            continue
        # The answer must actually contain this pronoun and its predicate.
        if content_words(clause[head.end():]) != tail:
            continue
        main_clause, _ = scope_parts(previous)
        subject = named_subject(main_clause)
        if not subject:
            return None
        predicate = main_clause[len(subject):]
        if OTHER_REFERENT.search(predicate) or NAMED_OBJECT.search(previous.replace(subject, "", 1)):
            return None
        candidates.add(subject)
    if len(candidates) != 1:
        return None
    subject = candidates.pop()
    possessive = "'s" if pronoun[0].casefold() == "its" else ""
    return subject + possessive + text[pronoun.end():]


def measurement_key(body: str, entities: set[tuple[str, ...]]) -> tuple | None:
    if match := MEASURED_FROM.fullmatch(body) or MEASUREMENT_OF.fullmatch(body):
        entity = entity_words(match["owner"])
    elif match := ACTIVE_MEASURE.fullmatch(body):
        candidates = {entity for entity in entities if entity[-1:] == (match["object"].casefold(),)}
        entity = (*entity_words(match["actor"]), match["object"].casefold())
        if candidates != {entity}:
            return None
    else:
        return None
    origin = re.sub(r"^(?:(?:its|the)\s+)?(?:last|most recent)\s+", "most recent ", match["origin"], flags=re.I)
    return entity, match["metric"].casefold(), normalized(origin)


def scope_parts(text: str) -> tuple[str, tuple[str, ...]]:
    text = WHILE_CONFIG.sub("", text)
    body, scope = split_scope(text)
    # A loss event can be stated as either an event condition or a completed event.
    if loss := LOSS_EVENT.search(text.rstrip(".")):
        if loss.start() == 0 or loss.end() == len(text.rstrip(".")):
            remaining = (text[:loss.start()] + " " + text[loss.end():]).strip(" ,.")
            return remaining, ("loss_event", *normalized(loss["passive"] or loss["nominal"]))
    if scope:
        return body, scope
    if config := CONFIG_SCOPE.search(body):
        remaining = (body[:config.start()] + " " + body[config.end():]).strip(" ,")
        return remaining, normalized(config["condition"])
    if timing := TIME_SCOPE.search(body):
        # A duration following a timeout stays part of the quantity field.
        if timing["marker"].casefold() == "after" and re.match(r"(?:\d|[a-z]+\s+(?:seconds?|minutes?|hours?|days?)\b)", timing["condition"], re.I):
            return body, ()
        marker = timing["marker"].casefold()
        condition = timing["condition"]
        completion = COMPLETION.get(marker)
        if completion and (event := completion.fullmatch(condition)):
            scope = ("after_completion", event[1].casefold())
        else:
            scope = (marker, *normalized(condition))
        return body[:timing.start()], scope
    return body, ()


@dataclass(frozen=True)
class Signature:
    words: tuple[str, ...]
    scope: tuple[str, ...]
    limiters: tuple[str, ...]
    negatives: bool
    units: tuple[str, ...]
    clauses: int
    fact: Fact | None
    measurement: tuple | None


def signature(text: str, clauses: list[str], entities: set[tuple[str, ...]]) -> Signature | None:
    resolved = resolve_pronoun(text, clauses)
    if resolved is None:
        return None
    body, scope = scope_parts(resolved)
    scope_count = len(SCOPE_WORDS.findall(WHILE_CONFIG.sub("", resolved)))
    # Multiple conditions can belong to distinct predicates within a single prediction.
    clause_count = max(len(atomic_units(body)), scope_count)
    if clause_count > 1:
        return None
    return Signature(
        content_words(body if scope[:1] in {("after_completion",), ("loss_event",)} else resolved),
        scope, tuple(word.casefold() for word in LIMITERS.findall(resolved)),
        bool(NEGATIVE.search(body)), tuple(unit.casefold().rstrip("s") for unit in UNIT.findall(resolved)),
        scope_count, parse_fact(resolved), measurement_key(body, entities),
    )


def semantic_candidates(
    gold: list[str], predicted: list[str], answer: str, threshold: float,
) -> list[tuple[int, int, float]]:
    clauses = CLAUSE_BOUNDARY.split(answer.strip())
    entities = {entity_words(subject) for clause in clauses if (subject := named_subject(clause))}
    mapped_gold = [signature(text, clauses, entities) for text in gold]
    mapped_predictions = [signature(text, clauses, entities) for text in predicted]
    candidates = []
    for gold_index, expected in enumerate(mapped_gold):
        for predicted_index, actual in enumerate(mapped_predictions):
            if gold[gold_index].strip().casefold().rstrip(".") == predicted[predicted_index].strip().casefold().rstrip("."):
                candidates.append((gold_index, predicted_index, 1.0))
                continue
            if expected is None or actual is None or (expected.negatives, expected.limiters) != (actual.negatives, actual.limiters):
                continue
            if expected.fact is not None and actual.fact is not None:
                if expected.fact == actual.fact:
                    candidates.append((gold_index, predicted_index, 1.0))
                continue
            if expected.scope != actual.scope or expected.clauses != actual.clauses:
                continue
            if expected.measurement is not None and actual.measurement is not None:
                if expected.measurement == actual.measurement:
                    candidates.append((gold_index, predicted_index, 1.0))
                continue
            if numbers(expected.words) != numbers(actual.words) or expected.units != actual.units:
                continue
            # An unresolved measurement construction is not safe to align by shared words.
            if expected.measurement is not None or actual.measurement is not None:
                continue
            overlap = token_overlap(expected.words, actual.words)
            if overlap >= threshold and is_negated(expected.words) == is_negated(actual.words):
                candidates.append((gold_index, predicted_index, overlap))
    return candidates
