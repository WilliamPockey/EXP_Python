from collections import Counter
from math import isfinite, log1p, sqrt

from ollama_client import post_json
from text_features import content_words, sentences
from verifier import Citation, InvalidModelOutput


DEFAULT_EMBED_MODEL = "qwen3-embedding:4b"
DENSE_CANDIDATES = 6
TOP_K = 3
QUERY_INSTRUCTION = "Instruct: Retrieve evidence that supports or refutes the given factual claim.\nQuery: "


def source_sentences(passages: list[str]) -> list[Citation]:
    return [
        Citation(index=index, quote=sentence)
        for index, passage in enumerate(passages)
        for sentence in sentences(passage)
    ]


class OllamaEmbeddings:
    def __init__(self, model: str, base_url: str, timeout: float) -> None:
        self.model = model
        self.url = base_url.rstrip("/") + "/api/embed"
        self.timeout = timeout
        self.vectors: dict[str, tuple[float, ...]] = {}

    def encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self.vectors))
        if missing:
            response = post_json(self.url, {"model": self.model, "input": missing}, self.timeout)
            embeddings = response.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(missing):
                raise InvalidModelOutput("Ollama returned the wrong number of embeddings")
            vectors = {}
            for text, vector in zip(missing, embeddings, strict=True):
                if not isinstance(vector, list) or not vector or any(
                    type(value) not in (int, float) or not isfinite(value) for value in vector
                ):
                    raise InvalidModelOutput("Ollama returned an invalid embedding vector")
                norm = sqrt(sum(value * value for value in vector))
                if norm == 0 or not isfinite(norm):
                    raise InvalidModelOutput("Ollama returned an invalid embedding norm")
                vectors[text] = tuple(value / norm for value in vector)
            self.vectors.update(vectors)
        vectors = [self.vectors[text] for text in texts]
        if len({len(vector) for vector in vectors}) > 1:
            raise InvalidModelOutput("Ollama returned inconsistent embedding dimensions")
        return vectors


class SentenceRetriever:
    def __init__(self, embeddings: OllamaEmbeddings) -> None:
        self.embeddings = embeddings

    def retrieve(
        self, claims: dict[str, str], passages: list[str], *, top_k: int = TOP_K,
        dense_limit: int = DENSE_CANDIDATES,
    ) -> dict[str, list[Citation]]:
        chunks = source_sentences(passages)
        if not chunks:
            return {claim_id: [] for claim_id in claims}
        vectors = self.embeddings.encode(
            [chunk.quote for chunk in chunks] + [QUERY_INSTRUCTION + text for text in claims.values()]
        )
        index, queries = vectors[:len(chunks)], vectors[len(chunks):]
        documents = [Counter(content_words(chunk.quote)) for chunk in chunks]
        lengths = [sum(document.values()) for document in documents]
        average_length = sum(lengths) / len(documents) or 1
        frequencies = Counter(word for document in documents for word in document)

        results = {}
        for (claim_id, text), query in zip(claims.items(), queries, strict=True):
            similarity = [sum(a * b for a, b in zip(vector, query, strict=True)) for vector in index]
            candidates = sorted(range(len(index)), key=lambda pos: (-similarity[pos], pos))[:dense_limit]
            terms = set(content_words(text))
            scores = {}
            for pos in candidates:
                length_factor = 1.2 * (0.25 + 0.75 * lengths[pos] / average_length)
                scores[pos] = sum(
                    log1p((len(documents) - frequencies[word] + 0.5) / (frequencies[word] + 0.5))
                    * documents[pos][word] * 2.2 / (documents[pos][word] + length_factor)
                    for word in sorted(terms & documents[pos].keys())
                )
            reranked = sorted(candidates, key=lambda pos: (-scores[pos], -similarity[pos], pos))
            results[claim_id] = [chunks[pos] for pos in reranked[:top_k]]
        return results
