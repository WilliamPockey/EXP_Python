import json
from copy import deepcopy
from pathlib import Path

from verifier import ModelUnavailable, VerifyRequest


class AssignmentDemoModel:
    """Return fixture data for the assignment example and known edge inputs."""

    def __init__(self) -> None:
        fixture_path = Path(__file__).parent / "examples" / "assignment.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self._request = VerifyRequest.model_validate(fixture["request"])
        self._response = fixture["response"]

    def verify(self, request: VerifyRequest) -> object:
        if request == self._request:
            return deepcopy(self._response)

        if request.generated_answer == "Please check the documentation.":
            return {"claims": []}

        if (
            request.generated_answer == self._request.generated_answer
            and not request.context_passages
        ):
            return {
                "claims": [
                    {
                        "claim_text": claim["claim_text"],
                        "verdict": "UNSUPPORTED",
                        "reasoning": "No source passages were provided.",
                        "citations": [],
                    }
                    for claim in self._response["claims"]
                ]
            }

        raise ModelUnavailable("The demo model only handles its documented sample inputs.")
