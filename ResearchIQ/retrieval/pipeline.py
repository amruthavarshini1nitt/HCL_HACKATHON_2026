"""Pipeline B (retrieval half): guard -> refine -> hybrid search -> rerank -> expand evidence."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from config import settings
from database.metadata_store import MetadataStore
from database.qdrant_store import QdrantStore
from generation.llm_client import ChatClient
from guardrails.injection_detector import PromptGuard
from guardrails.input_validator import RateLimiter
from ingestion.embedding import Embedder
from retrieval.evidence_expander import expand_evidence
from retrieval.hybrid_search import HybridSearcher
from retrieval.keyword_search import BM25Searcher
from retrieval.query_guard import check_query
from retrieval.query_refiner import refine_query
from retrieval.reranker import Reranker, rerank
from retrieval.types import EvidenceRecord, RefinedQuery
from retrieval.vector_search import VectorSearcher

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    allowed: bool
    message: str = ""                     # user-facing reason when not allowed
    refined: RefinedQuery | None = None
    evidence: list[EvidenceRecord] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


class Retriever:
    """Wires the retrieval stages. All components are injectable for tests."""

    def __init__(self, store: MetadataStore, qdrant: QdrantStore, embedder: Embedder,
                 reranker: Reranker | None = None, llm: ChatClient | None = None,
                 guard: PromptGuard | None = None, limiter: RateLimiter | None = None,
                 use_prompt_guard: bool = True) -> None:
        self.store, self.llm, self.guard, self.limiter = store, llm, guard, limiter
        self.use_prompt_guard = use_prompt_guard
        self.reranker = reranker or Reranker()
        self.hybrid = HybridSearcher(VectorSearcher(qdrant, store, embedder), BM25Searcher(store))

    def retrieve(self, query: str, session_key: str = "default", paper_ids: list[str] | None = None,
                 top_n: int | None = None, top_k_dense: int | None = None,
                 top_k_bm25: int | None = None) -> RetrievalResult:
        timings: dict[str, float] = {}
        t = time.perf_counter()
        decision = check_query(query, session_key, guard=self.guard, limiter=self.limiter,
                               use_model=self.use_prompt_guard)
        timings["guard"] = time.perf_counter() - t
        if not decision.allowed:
            return RetrievalResult(False, decision.reason, timings=timings,
                                   debug={"blocked_by": decision.injection.method if decision.injection else "validation"})

        t = time.perf_counter()
        rq = refine_query(decision.query, self.llm)
        timings["refine"] = time.perf_counter() - t
        if not rq.in_scope:
            return RetrievalResult(False, "This question doesn't appear to be about the uploaded research "
                                   "papers or AI/ML research. Please ask a research question.", rq, timings=timings)

        fused, hinfo = self.hybrid.search(rq, paper_ids=paper_ids, top_k_dense=top_k_dense,
                                          top_k_bm25=top_k_bm25)
        timings["retrieve"] = hinfo["latency_s"]

        n = top_n or settings.rerank_top_n
        if rq.intent in ("comparison", "synthesis"):
            n = min(n + 2, 8)
        scope = paper_ids or sorted({c.paper_id for c in fused})
        # rerank against the ORIGINAL user query, as specified
        selected, rinfo = rerank(rq.original_query, fused, self.reranker, top_n=n,
                                 ensure_paper_ids=scope if rq.intent == "comparison" and len(scope) > 1 else None)
        timings["rerank"] = rinfo["latency_s"]

        t = time.perf_counter()
        evidence = expand_evidence(selected, rq.intent, self.store)
        timings["expand"] = time.perf_counter() - t
        return RetrievalResult(True, refined=rq, evidence=evidence, timings=timings,
                               debug={"hybrid": hinfo, "rerank": rinfo,
                                      "refiner_used_llm": rq.refiner_used_llm})
