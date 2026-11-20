import os

from fastapi import FastAPI, HTTPException

from demo_model import AssignmentDemoModel
from fact_evidence import FactEvidenceVerifier
from ollama_client import DEFAULT_OLLAMA_URL, DEFAULT_TIMEOUT, OllamaClient
from optimized import OptimizedVerifier
from retrieval import DEFAULT_EMBED_MODEL, OllamaEmbeddings, SentenceRetriever
from verifier import (
    InvalidModelOutput,
    ModelError,
    ModelUnavailable,
    VerificationModel,
    VerificationReport,
    VerifyRequest,
    validate_report,
)


def create_app(model: VerificationModel) -> FastAPI:
    api = FastAPI(title="Citation Grounding Verifier")

    @api.post("/verify", response_model=VerificationReport)
    def verify(request: VerifyRequest) -> VerificationReport:
        if not request.generated_answer.strip():
            return VerificationReport(claims=[])

        try:
            payload = model.verify(request)
            return validate_report(payload, request.context_passages)
        except InvalidModelOutput as error:
            raise HTTPException(
                status_code=502, detail="Invalid verification model output"
            ) from error
        except ModelUnavailable as error:
            raise HTTPException(
                status_code=503, detail="Verification model is unavailable for this request"
            ) from error
        except ModelError as error:
            raise HTTPException(status_code=502, detail="Verification model request failed") from error
        except TimeoutError as error:
            raise HTTPException(status_code=504, detail="Verification model timed out") from error

    return api


def configured_client() -> OllamaClient:
    model = os.environ.get("GROUNDING_MODEL")
    if not model:
        raise RuntimeError("Set GROUNDING_MODEL to an installed Ollama model tag")
    base_url = os.environ.get("GROUNDING_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    return OllamaClient(model, base_url, DEFAULT_TIMEOUT)


def create_optimized_app() -> FastAPI:
    return create_app(OptimizedVerifier(configured_client()))


def create_submission_app() -> FastAPI:
    client = configured_client()
    base_url = os.environ.get("GROUNDING_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    embed_model = os.environ.get("GROUNDING_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    retriever = SentenceRetriever(OllamaEmbeddings(embed_model, base_url, client.timeout))
    return create_app(FactEvidenceVerifier(client, retriever))


app = create_app(AssignmentDemoModel())
