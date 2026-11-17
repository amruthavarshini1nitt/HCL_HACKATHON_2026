"""Step B10: output guardrails. Citation checks, claim-to-evidence support, sensitive-data redaction."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from config import settings
from generation.answer_generator import AnswerOutput
from generation.llm_client import ChatClient, LenientModel, LLMUnavailable
from generation.prompt_templates import VERIFY_SYSTEM
from guardrails.citation_validator import CitationReport, validate_citations
from retrieval.keyword_search import tokenize
from retrieval.types import EvidenceRecord

logger = logging.getLogger(__name__)

SIG_NUMBER = re.compile(r"\d+\.\d+|\d{3,}|\d+(?=\s*%)")
ID_TOKEN = re.compile(r"\[[^\]]+\]")
SENSITIVE = [
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "[redacted key]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "[redacted key]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[redacted key]"),
    (re.compile(r"\b[A-Z][A-Z0-9_]*(?:API_KEY|SECRET|TOKEN)\b"), "[redacted variable]"),
    (re.compile(r"[A-Za-z]:\\(?:Users|Windows|Program Files)[^\s\"'<>]*"), "[path removed]"),
    (re.compile(r"(?<!\w)/(?:home|Users|etc|var|root)/[^\s\"'<>]*"), "[path removed]"),
    (re.compile(r"You are ResearchIQ, an evidence-grounded assistant", re.I), "[removed]"),
]


class _Verdict(BaseModel):
    index: int
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str = ""


class _Verdicts(LenientModel):
    verdicts: list[_Verdict]


@dataclass
class ClaimCheck:
    index: int
    status: Literal["supported", "partial", "unsupported", "unchecked"]
    reasons: list[str] = field(default_factory=list)


@dataclass
class OutputValidation:
    citation_report: CitationReport
    claim_checks: list[ClaimCheck]
    verifier_used: bool
    redactions: int = 0
    answer: str = ""

    @property
    def unsupported(self) -> list[int]:
        return sorted({c.index for c in self.claim_checks if c.status == "unsupported"}
                      | self.citation_report.invalid_claims)

    @property
    def ok(self) -> bool:
        return not self.unsupported and not any(
            i.kind in ("unknown_paper", "unknown_figure", "unknown_page", "unknown_source")
            and i.claim_index is None for i in self.citation_report.issues)


def redact_sensitive(text: str) -> tuple[str, int]:
    """Remove secrets, env-var names, local paths and system-prompt text from user-facing output."""
    n = 0
    for rx, repl in SENSITIVE:
        text, k = rx.subn(repl, text)
        n += k
    for secret in (settings.openai_api_key, settings.llm_api_key, settings.vlm_api_key, settings.hf_token,
                   settings.qdrant_api_key):
        if secret and len(secret) > 8 and secret in text:
            text = text.replace(secret, "[redacted key]")
            n += 1
    return text, n


def _evidence_text(ev: list[EvidenceRecord]) -> str:
    return " ".join(f"{e.content} {e.caption or ''} {e.parent_content or ''}" for e in ev)


def deterministic_support(claim_text: str, cited: list[EvidenceRecord], *, check_overlap: bool = True
                          ) -> tuple[bool, list[str]]:
    """Numbers in the claim must appear in the cited evidence; wording must overlap with it."""
    reasons: list[str] = []
    ev_text = _evidence_text(cited)
    body = ID_TOKEN.sub(" ", claim_text)
    flat = ev_text.replace(",", "")
    for num in SIG_NUMBER.findall(body.replace(",", "")):
        if num not in flat:
            reasons.append(f"number {num} not found in cited evidence")
    if check_overlap:
        ctoks = set(tokenize(body))
        if ctoks:
            overlap = len(ctoks & set(tokenize(ev_text))) / len(ctoks)
            if overlap < settings.claim_overlap_min:
                reasons.append(f"low wording overlap with cited evidence ({overlap:.0%})")
    return not reasons, reasons


def _verify_with_llm(items: list[tuple[int, str, list[EvidenceRecord]]], client: ChatClient
                     ) -> dict[int, _Verdict]:
    parts = []
    for idx, text, cited in items:
        ev = "\n".join(f"[{e.source_id}] {(e.content or '')[:900]} {(e.parent_content or '')[:400]}"
                       for e in cited)
        parts.append(f"<claim index=\"{idx}\">{text}</claim>\n<cited_evidence>\n{ev}\n</cited_evidence>")
    out = client.complete_model(_Verdicts, [{"role": "system", "content": VERIFY_SYSTEM},
                                            {"role": "user", "content": "\n\n".join(parts)}], max_tokens=900)
    return {v.index: v for v in out.verdicts}


def validate_output(out: AnswerOutput, evidence: list[EvidenceRecord], verifier: ChatClient | None = None
                    ) -> OutputValidation:
    """Run all post-generation checks. A citation that exists is not assumed to support its claim."""
    by_id = {e.source_id: e for e in evidence}
    claims = [c.model_dump() for c in out.claims]
    citation = validate_citations(out.answer, claims, evidence)
    checks: list[ClaimCheck] = []
    to_verify: list[tuple[int, str, list[EvidenceRecord]]] = []
    for i, c in enumerate(out.claims):
        if c.kind == "suggestion":
            checks.append(ClaimCheck(i, "unchecked", ["AI-generated suggestion; not a reported finding"]))
            continue
        cited = [by_id[s] for s in c.source_ids if s in by_id]
        if not cited:
            checks.append(ClaimCheck(i, "unsupported", ["no valid cited evidence"]))
            continue
        ok, reasons = deterministic_support(c.text, cited, check_overlap=c.kind == "reported")
        checks.append(ClaimCheck(i, "supported" if ok else "unsupported", reasons))
        if ok:
            to_verify.append((i, c.text, cited))

    verifier_used = False
    if verifier is not None and to_verify:
        try:
            verdicts = _verify_with_llm(to_verify, verifier)
            verifier_used = True
            for idx, _, _ in to_verify:
                v = verdicts.get(idx)
                if v is None:
                    continue
                chk = checks[idx]
                chk.status = v.verdict
                if v.verdict != "supported":
                    chk.reasons.append(f"verifier: {v.reason or v.verdict}")
        except LLMUnavailable:
            logger.warning("claim verifier unavailable; using deterministic checks only")
    answer, n = redact_sensitive(out.answer)
    return OutputValidation(citation, checks, verifier_used, n, answer)
