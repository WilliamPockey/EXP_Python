import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from verifier import InvalidModelOutput, ModelError


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 60


@dataclass
class ModelRun:
    calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    raw_quotes: tuple[int, int] | None = None
    repair_attempted: bool = False
    repair_succeeded: bool = False


class OllamaClient:
    def __init__(self, model: str, base_url: str, timeout: float) -> None:
        self.model = model
        self.url = base_url.rstrip("/") + "/api/chat"
        self.timeout = timeout
        self._run: ContextVar[ModelRun | None] = ContextVar("ollama_run", default=None)

    @property
    def last_run(self) -> ModelRun | None:
        return self._run.get()

    def start_run(self) -> ModelRun:
        run = ModelRun()
        self._run.set(run)
        return run

    def generate(self, prompt: str, schema: dict, run: ModelRun) -> object:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0, "seed": 0, "num_predict": 2048},
        }
        request = Request(
            self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        run.calls += 1
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            error.close()
            raise ModelError(f"Ollama returned HTTP {error.code}") from error
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TimeoutError("Ollama request timed out") from error
            raise ModelError(f"Ollama connection failed: {error.reason}") from error
        except TimeoutError:
            raise
        except OSError as error:
            raise ModelError(f"Ollama transport failed: {error}") from error
        except (UnicodeError, ValueError) as error:
            raise InvalidModelOutput("Ollama response is not valid JSON") from error

        if not isinstance(result, dict):
            raise InvalidModelOutput("Ollama response must be a JSON object")
        for field, api_field in (("prompt_tokens", "prompt_eval_count"), ("completion_tokens", "eval_count")):
            value = result.get(api_field)
            if type(value) is int and value >= 0:
                run.usage[field] = run.usage.get(field, 0) + value
        if result.get("done_reason") == "length":
            raise InvalidModelOutput("Ollama output reached the token limit")
        try:
            return json.loads(result["message"]["content"])
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidModelOutput("Ollama message content is not a JSON report") from error
