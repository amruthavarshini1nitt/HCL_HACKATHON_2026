"""Step B9: evidence-grounded structured answer generation (schema-validated)."""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from generation.llm_client import ChatClient, LenientModel, get_llm
from generation.prompt_templates import ANSWER_SYSTEM, format_evidence
from retrieval.types import EvidenceRecord

logger = logging.getLogger(__name__)


class Claim(LenientModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)
    kind: Literal["reported", "interpretation", "suggestion"] = "reported"

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, v: object) -> str:
        """Map variants like 'Reported finding' / 'AI suggestion' onto the three allowed kinds."""
        s = str(v or "reported").lower()
        if "sugg" in s:
            return "suggestion"
        if "interp" in s or "infer" in s:
            return "interpretation"
        return "reported"


class AnswerOutput(LenientModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_sufficient: bool = True


def generate_answer(query: str, evidence: list[EvidenceRecord], *, intent: str = "factual",
                    notes: list[str] | None = None, client: ChatClient | None = None) -> AnswerOutput:
    """Call the LLM with delimited evidence and parse the structured output.

    notes: extra constraints (missing evidence, validator feedback for a retry).
    Raises LLMUnavailable on service failure (caller decides how to degrade).
    """
    client = client or get_llm()
    user = (f"<question intent=\"{intent}\">{query}</question>\n\n<evidence_set>\n"
            f"{format_evidence(evidence)}\n</evidence_set>")
    if notes:
        user += "\n\nAdditional constraints:\n" + "\n".join(f"- {n}" for n in notes)
    return client.complete_model(
        AnswerOutput, [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=1800, temperature=0.1)
