import re

from ollama_client import ModelRun, OllamaClient
from text_features import NEGATIONS, content_words, is_negated, numbers, sentences, token_overlap
from verifier import VerificationReport, VerifyRequest


NONFACTUAL_START = re.compile(r"^(?:please|thanks|thank you)\b", re.IGNORECASE)
RANGE_MARKERS = re.compile(
    r"\b(?:at most|at least|up to|more than|less than|over|under|between|approximately|about|after|before)\b",
    re.IGNORECASE,
)
SUPPORT_THRESHOLD = 0.72
MINIMAL_PROMPT = (
    "Extract atomic facts from generated_answer. Use context_passages as evidence. "
    "Return a claims array with claim_text, verdict, reasoning, and citations for each fact. "
    "Use SUPPORTED if the passages entail the fact, CONTRADICTED if they refute it, and "
    "UNSUPPORTED if they give insufficient evidence. For SUPPORTED or CONTRADICTED, "
    "include a citation with the exact quote and its zero-based passage index. "
    "Use empty citations for UNSUPPORTED. Return an empty claims array if the answer has no facts."
)
INPUT_SEPARATOR = "\n\nInput JSON:\n"


def has_conflict(
    claim: str, claim_words: tuple[str, ...], quote: str, quote_words: tuple[str, ...]
) -> bool:
    claim_values, quote_values = numbers(claim_words), numbers(quote_words)
    claim_core = tuple(word for word in claim_words if word not in claim_values and word not in NEGATIONS)
    quote_core = tuple(word for word in quote_words if word not in quote_values and word not in NEGATIONS)
    if not claim_core or claim_core != quote_core:
        return False
    if is_negated(claim_words) != is_negated(quote_words):
        return claim_values == quote_values
    return (
        len(claim_values) == len(quote_values) == 1
        and claim_values != quote_values
        and not RANGE_MARKERS.search(claim)
        and not RANGE_MARKERS.search(quote)
    )


class LexicalBaseline:
    def verify(self, request: VerifyRequest) -> object:
        candidates = [
            (index, sentence, content_words(sentence))
            for index, passage in enumerate(request.context_passages)
            for sentence in sentences(passage)
        ]
        claims = []
        seen = set()
        for sentence in sentences(request.generated_answer):
            words = content_words(sentence)
            if not words or words in seen or sentence.endswith("?") or NONFACTUAL_START.match(sentence):
                continue
            seen.add(words)
            candidate = max(candidates, key=lambda item: token_overlap(words, item[2]), default=None)
            verdict = "UNSUPPORTED"
            reasoning = "No source sentence passed the lexical checks."
            citations = []
            if candidate is not None:
                index, quote, quote_words = candidate
                if has_conflict(sentence, words, quote, quote_words):
                    verdict = "CONTRADICTED"
                    reasoning = "The closest source sentence differs in negation or an exact number."
                elif (
                    token_overlap(words, quote_words) >= SUPPORT_THRESHOLD
                    and is_negated(words) == is_negated(quote_words)
                    and set(numbers(words)).issubset(numbers(quote_words))
                ):
                    verdict = "SUPPORTED"
                    reasoning = "The closest source sentence passed the lexical overlap checks."
                if verdict != "UNSUPPORTED":
                    citations.append({"index": index, "quote": quote})

            claims.append({
                "claim_text": sentence, "verdict": verdict, "reasoning": reasoning, "citations": citations,
            })
        return {"claims": claims}


class OllamaBaseline:
    def __init__(self, model: str, base_url: str, timeout: float, *, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient(model, base_url, timeout)
        self.schema = VerificationReport.model_json_schema()

    @property
    def last_run(self) -> ModelRun | None:
        return self.client.last_run

    def verify(self, request: VerifyRequest) -> object:
        run = self.client.start_run()
        prompt = MINIMAL_PROMPT + INPUT_SEPARATOR + request.model_dump_json()
        return self.client.generate(prompt, self.schema, run)
