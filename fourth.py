import json

from pydantic import ValidationError

from ollama_client import ModelRun, OllamaClient
from optimized import (
    EXTRACT_PROMPT,
    VERIFY_PROMPT,
    Assessment,
    OptimizedVerifier,
    check_assessments,
    indexed_schema,
)
from retrieval import SentenceRetriever
from verifier import Claim, InvalidModelOutput, StrictModel, count_raw_quotes


FEATURES = ("split", "examples", "rag", "per_claim")
FULL_FEATURES = frozenset(FEATURES)
CONFIGURATIONS = {
    "baseline3": frozenset(),
    "baseline4": FULL_FEATURES,
    **{f"add_{feature}": frozenset({feature}) for feature in FEATURES},
    **{f"drop_{feature}": FULL_FEATURES - {feature} for feature in FEATURES},
}

SPLIT_PROMPT = (
    "Split the answer into short independent sentence units in order. Split coordinated predicates "
    "if they express separate facts, repeating the shared subject and condition as needed. Keep a "
    "coordination of objects together. Preserve names, quantities, units, negation, conditional scope "
    "and temporal order. Use only the answer's information. Omit standalone advice, questions and thanks. "
    'Example answer: "In safe mode, Pine reads gzip and zip files and never deletes backups. Thanks!" '
    'Output: {"sentences":["In safe mode, Pine reads gzip and zip files.",'
    '"In safe mode, Pine never deletes backups."]}'
)
EXTRA_EXAMPLES = (
    '\nExample input: {"0":"If the request is signed, Basin accepts it, but it never saves its signature.",'
    '"1":"It saves the request body for 90 seconds."}'
    '\nExample output: {"0":["Basin accepts a request if it is signed.",'
    '"Basin never saves a signed request\'s signature."],'
    '"1":["Basin saves the request body for 90 seconds."]}'
    '\nExample input: {"0":"The Delta relay uses TCP in compact mode and UDP in bulk mode.",'
    '"1":"So UDP is used by the relay in bulk mode.","2":"Please check the monitor."}'
    '\nExample output: {"0":["The Delta relay uses TCP in compact mode.",'
    '"The Delta relay uses UDP in bulk mode."],"1":[],"2":[]}'
    '\nExample input: {"0":"Before releasing the lock, Reed records the batch ID and the timestamp.",'
    '"1":"It may discard entries older than 120 minutes."}'
    '\nExample output: {"0":["Reed records the batch ID before releasing the lock.",'
    '"Reed records the timestamp before releasing the lock."],'
    '"1":["Reed may discard entries older than 120 minutes."]}'
)
RAG_VERIFY_PROMPT = (
    "Check each claim against its retrieved evidence sentences. Keep exactly the supplied claim IDs. "
    "Use SUPPORTED if the sentences entail the whole claim, CONTRADICTED for explicit refutation under "
    "the same conditions, and UNSUPPORTED if evidence is missing or conflicting. Use all necessary "
    "premises when sentences together prove a claim. Include concise reasoning. For support or refutation, "
    "use the original passage index shown in the evidence and copy the quote verbatim. Use empty citations "
    "for UNSUPPORTED."
)


class SentenceSplit(StrictModel):
    sentences: list[str]


class FourthVerifier(OptimizedVerifier):
    def __init__(
        self, client: OllamaClient, retriever: SentenceRetriever, features: frozenset[str] = FULL_FEATURES
    ) -> None:
        super().__init__(client)
        self.retriever = retriever
        self.features = features
        if "examples" in features:
            self.extraction_prompt = EXTRACT_PROMPT + EXTRA_EXAMPLES

    def segment(self, answer: str, run: ModelRun) -> dict[str, str]:
        if "split" not in self.features:
            return super().segment(answer, run)
        if not answer.strip():
            return {}
        prompt = SPLIT_PROMPT + "\n\nInput JSON:\n" + json.dumps({"answer": answer}, ensure_ascii=False)
        raw = self.client.generate(prompt, SentenceSplit.model_json_schema(), run)
        try:
            result = SentenceSplit.model_validate(raw)
        except ValidationError as error:
            raise InvalidModelOutput("LLM segmentation does not match the response schema") from error
        if any(not sentence.strip() for sentence in result.sentences):
            raise InvalidModelOutput("LLM segmentation contains a blank sentence")
        return {str(index): sentence.strip() for index, sentence in enumerate(result.sentences)}

    def assess(
        self, claims: dict[str, str], passages: list[str], run: ModelRun
    ) -> tuple[dict[str, Claim], dict[str, str]]:
        use_rag = "rag" in self.features
        per_claim = "per_claim" in self.features
        if not use_rag and not per_claim:
            return super().assess(claims, passages, run)

        evidence = self.retriever.retrieve(claims, passages) if use_rag else {}
        if use_rag:
            run.retrieved_evidence = evidence
        groups = [{key: text} for key, text in claims.items()] if per_claim else [claims]
        combined: dict[str, object] = {}
        for group in groups:
            if use_rag:
                stage_input = {
                    "claims": group,
                    "evidence_sentences": {key: [cite.model_dump() for cite in evidence[key]] for key in group},
                }
                instructions = RAG_VERIFY_PROMPT
            else:
                stage_input = self.passage_inputs(group, passages)
                instructions = VERIFY_PROMPT
            prompt = instructions + "\n\nInput JSON:\n" + json.dumps(stage_input, ensure_ascii=False)
            payload = self.client.generate(prompt, indexed_schema(list(group), Assessment.model_json_schema()), run)
            if isinstance(payload, dict):
                valid, total = count_raw_quotes({"claims": list(payload.values())}, passages)
                old_valid, old_total = run.raw_quotes or (0, 0)
                run.raw_quotes = (old_valid + valid, old_total + total)
            if not isinstance(payload, dict) or set(payload) != set(group):
                raise InvalidModelOutput("the assessment must contain exactly the requested claim IDs")
            combined.update(payload)

        accepted, rejected = check_assessments(combined, claims, passages)
        if use_rag:
            for key, claim in list(accepted.items()):
                if any(not any(cite.index == item.index and cite.quote in item.quote for item in evidence[key])
                       for cite in claim.citations):
                    rejected[key] = "citation is outside the retrieved evidence sentences"
                    del accepted[key]
        return accepted, rejected
