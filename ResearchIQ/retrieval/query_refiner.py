"""Step B3: LLM-based intent classification and query rewriting (never answers the question)."""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from generation.llm_client import ChatClient, LenientModel, LLMUnavailable, get_llm
from generation.prompt_templates import QUERY_REFINE_SYSTEM
from retrieval.query_classifier import classify_intent, default_content_types
from retrieval.types import ContentType, Intent, RefinedQuery

logger = logging.getLogger(__name__)


class _LLMRefinement(LenientModel):
    refined_query: str
    intent: Intent = "factual"
    keywords: list[str] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list)
    content_types: list[ContentType] = Field(default_factory=lambda: ["text"])
    in_scope: bool = True


def fallback_refinement(query: str) -> RefinedQuery:
    """Heuristic refinement used when the LLM is unavailable: original query, rule-based intent."""
    intent = classify_intent(query)
    return RefinedQuery(original_query=query, refined_query=query, intent=intent,
                        content_types=default_content_types(query, intent), refiner_used_llm=False)


def refine_query(query: str, client: ChatClient | None = None) -> RefinedQuery:
    """Return a validated RefinedQuery. On any LLM failure, degrade gracefully to heuristics."""
    client = client or get_llm()
    try:
        out = client.complete_model(
            _LLMRefinement,
            [{"role": "system", "content": QUERY_REFINE_SYSTEM},
             {"role": "user", "content": f"<question>{query}</question>"}],
            max_tokens=500)
    except LLMUnavailable as e:
        logger.warning("query refiner fell back to heuristics: %s", e)
        return fallback_refinement(query)
    refined = out.refined_query.strip() or query
    subs = [s.strip() for s in out.subqueries if s and s.strip()][:3]
    if out.intent in ("factual", "other") and len(query.split()) < 15:
        subs = []  # keep simple questions simple
    types = list(dict.fromkeys(out.content_types)) or ["text"]
    return RefinedQuery(original_query=query, refined_query=refined[:500], intent=out.intent,
                        keywords=[k.strip() for k in out.keywords if k.strip()][:12], subqueries=subs,
                        content_types=types, in_scope=out.in_scope)
