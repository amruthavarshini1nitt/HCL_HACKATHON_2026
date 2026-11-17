"""Step B2: input guardrails for a user question (validation, rate limit, prompt injection)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from guardrails.injection_detector import InjectionResult, PromptGuard, detect_injection
from guardrails.input_validator import RateLimiter, validate_query

logger = logging.getLogger(__name__)
_limiter = RateLimiter()


@dataclass
class GuardDecision:
    allowed: bool
    query: str = ""
    reason: str = ""             # user-safe explanation when blocked
    injection: InjectionResult | None = None


def check_query(query: object, session_key: str = "default", *, guard: PromptGuard | None = None,
                limiter: RateLimiter | None = None, use_model: bool = True) -> GuardDecision:
    """Run deterministic checks, then the injection classifier. Scope is checked by the refiner."""
    v = validate_query(query)
    if not v.ok:
        return GuardDecision(False, reason=v.reason)
    if not (limiter or _limiter).allow(session_key):
        return GuardDecision(False, v.cleaned, "Too many requests. Please wait a moment and try again.")
    inj = detect_injection(v.cleaned, guard=guard, use_model=use_model)
    if inj.flagged:
        logger.warning("blocked query (method=%s score=%.2f)", inj.method, inj.score)
        return GuardDecision(False, v.cleaned,
                             "This request looks like an attempt to override the assistant's instructions, "
                             "so it was not processed. Please ask a question about the papers.", inj)
    return GuardDecision(True, v.cleaned, injection=inj)
