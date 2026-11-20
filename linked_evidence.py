import json
import re
from typing import Literal

from pydantic import Field, ValidationError

from fourth import FULL_FEATURES, FourthVerifier
from ollama_client import ModelRun, OllamaClient
from optimized import check_assessments
from retrieval import SentenceRetriever, source_sentences
from text_features import content_words
from verifier import Citation, Claim, StrictModel


MAX_EVIDENCE = 8
MAX_BUNDLES = 4
LINK_TOKEN = re.compile(r"\b[A-Z][A-Za-z\d_-]+\b")
LINK_PHRASE = re.compile(r"\b(?:[a-z][\w-]+ (?:profile|mode|queue|parser|policy|cluster|store|preset|family)|tier \d+)\b", re.IGNORECASE)
COMMON_CAPITALIZED = frozenset("the an and when where if in for every all only under after before with from while however no".split())
RELATIONS = {"supports": "SUPPORTED", "refutes": "CONTRADICTED", "insufficient": "UNSUPPORTED"}
DECISION_PROMPT = (
    "Judge the single claim using only the evidence. supports: the evidence proves the full statement. "
    "refutes: it proves the opposite in the same scope. insufficient: proof is missing or sources conflict. "
    "Follow a shared name or configuration across sentences. A similar statement about another "
    "entity is irrelevant. Give a short explanation and IDs of every sentence used. "
    "An insufficient verdict uses no IDs."
)
CHAIN_PROMPT = (
    "\nThe optional bundles are candidate joins, whose listed evidence IDs travel together. "
    "Select a bundle ID only when every part is needed for your finding. You may instead select the "
    "necessary individual sentence IDs. Apply a rule about the linked configuration to the entity "
    "that uses it. Missing evidence means insufficient. Opposite behavior in the linked "
    "configuration means refutes, even if a different configuration agrees with the claim."
    "\nExample claim: The Spruce dashboard rerenders hourly."
    "\nSentences: A: Spruce uses the Copper display preset. B: Copper display presets rerender every hour."
    '\nDecision: {"relation":"supports","evidence_ids":["A","B"],"reasoning":"Spruce uses Copper, which rerenders hourly."}'
    "\nExample claim: North tasks are retained."
    "\nSentences: A: North uses family Vega for tasks. B: Vega task settings never retain tasks. C: Family Luna retains tasks."
    '\nDecision: {"relation":"refutes","evidence_ids":["A","B"],"reasoning":"North uses Vega, which explicitly never retains tasks."}'
)


class EvidenceDecision(StrictModel):
    relation: Literal["supports", "refutes", "insufficient"]
    evidence_ids: list[str]
    reasoning: str = Field(min_length=1)


def name_tokens(text: str) -> set[str]:
    return {name.casefold() for name in LINK_TOKEN.findall(text)} - COMMON_CAPITALIZED


def anchors(text: str) -> set[str]:
    return name_tokens(text) | {phrase.casefold() for phrase in LINK_PHRASE.findall(text)}


def neighbors(position: int, chunks: list[Citation], tags: list[set[str]], query: set[str]) -> list[int]:
    return [
        other for other, tag in enumerate(tags)
        if chunks[other].index != chunks[position].index and (tags[position] & tag) - query
    ]


class LinkedRetriever:
    def __init__(self, dense: SentenceRetriever) -> None:
        self.dense = dense

    def retrieve(self, claims: dict[str, str], passages: list[str]) -> dict[str, list[Citation]]:
        chunks = source_sentences(passages)
        tags = [anchors(citation.quote) for citation in chunks]
        positions = {(citation.index, citation.quote): index for index, citation in enumerate(chunks)}
        ranked = self.dense.retrieve(claims, passages, top_k=6, dense_limit=8)
        result = {}
        for key, text in claims.items():
            query = anchors(text)
            ranked_ids = [positions[citation.index, citation.quote] for citation in ranked[key]]
            rank = {position: index for index, position in enumerate(ranked_ids)}
            source_matches = sorted(
                (index for index, tag in enumerate(tags) if name_tokens(text) & tag),
                key=lambda index: (rank.get(index, len(chunks)), index),
            )[:2]
            selected = dict.fromkeys(ranked_ids[:3] + source_matches)
            frontier = source_matches or ranked_ids[:3]
            visited = set(frontier)
            for _ in range(2):
                expanded = {index for position in frontier for index in neighbors(position, chunks, tags, query)} - visited
                frontier = sorted(expanded, key=lambda index: (rank.get(index, len(chunks)), index))
                for index in frontier:
                    selected[index] = None
                    if len(selected) == MAX_EVIDENCE:
                        break
                if len(selected) >= MAX_EVIDENCE:
                    break
                visited.update(frontier)
            for index in ranked_ids:
                selected.setdefault(index, None)
            result[key] = [chunks[index] for index in sorted(list(selected)[:MAX_EVIDENCE])]
        return result


def linked_bundles(text: str, candidates: dict[str, Citation]) -> dict[str, list[str]]:
    keys = list(candidates)
    chunks = list(candidates.values())
    tags = [anchors(citation.quote) for citation in chunks]
    query = anchors(text)
    starts = [index for index, tag in enumerate(tags) if name_tokens(text) & tag]
    paths: set[tuple[int, ...]] = set()
    for start in starts:
        for second in neighbors(start, chunks, tags, query):
            paths.add(tuple(sorted((start, second))))
            for third in neighbors(second, chunks, tags, query):
                if chunks[third].index != chunks[start].index:
                    paths.add(tuple(sorted((start, second, third))))
    terms = set(content_words(text))
    word_sets = [set(content_words(citation.quote)) for citation in chunks]
    ranked = sorted(paths, key=lambda path: (-len(terms & set.union(*(word_sets[i] for i in path))), len(path), path))
    return {f"J{index + 1}": [keys[i] for i in path] for index, path in enumerate(ranked[:MAX_BUNDLES])}


def restore_citations(
    decision: EvidenceDecision, candidates: dict[str, Citation], bundles: dict[str, list[str]],
) -> list[dict]:
    ids = dict.fromkeys(member for key in decision.evidence_ids for member in bundles.get(key, [key]))
    if ids.keys() - candidates.keys():
        raise ValueError("evidence ID is outside the retrieved sentence or bundle catalog")
    return [candidates[key].model_dump() for key in ids]


class LinkedEvidenceVerifier(FourthVerifier):
    def __init__(self, client: OllamaClient, dense: SentenceRetriever, *, use_bundles: bool = True) -> None:
        super().__init__(client, dense, FULL_FEATURES - {"split"})
        self.linked = LinkedRetriever(dense)
        self.use_bundles = use_bundles
        self.decision_schema = EvidenceDecision.model_json_schema()

    def decide(self, data: dict, run: ModelRun) -> EvidenceDecision:
        instructions = DECISION_PROMPT + (CHAIN_PROMPT if self.use_bundles else "")
        prompt = instructions + "\n\nInput JSON:\n" + json.dumps(data, ensure_ascii=False)
        return EvidenceDecision.model_validate(self.client.generate(prompt, self.decision_schema, run))

    def assess(
        self, claims: dict[str, str], passages: list[str], run: ModelRun,
    ) -> tuple[dict[str, Claim], dict[str, str]]:
        run.retrieved_evidence = self.linked.retrieve(claims, passages)
        run.raw_quotes = (0, 0)
        payload, rejected = {}, {}
        for key, text in claims.items():
            candidates = {chr(ord("A") + index): citation for index, citation in enumerate(run.retrieved_evidence[key])}
            bundles = linked_bundles(text, candidates) if self.use_bundles else {}
            data = {"claim": text, "sentences": {ident: item.quote for ident, item in candidates.items()}}
            if bundles:
                data["bundles"] = bundles
            try:
                decision = self.decide(data, run)
                payload[key] = {
                    "verdict": RELATIONS[decision.relation],
                    "reasoning": decision.reasoning.strip(),
                    "citations": restore_citations(decision, candidates, bundles),
                }
                if not payload[key]["reasoning"]:
                    del payload[key]
                    rejected[key] = "the evidence decision has a blank explanation"
            except ValidationError:
                rejected[key] = "the evidence decision does not match the response schema"
            except ValueError as error:
                rejected[key] = str(error)
        accepted, failed_checks = check_assessments(payload, {key: claims[key] for key in payload}, passages)
        return accepted, {**rejected, **failed_checks}
