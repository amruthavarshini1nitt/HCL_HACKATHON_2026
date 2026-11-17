"""Step B4: parallel dense + BM25 retrieval over query variants, fused with RRF."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from config import settings
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.keyword_search import BM25Searcher
from retrieval.types import Candidate, RefinedQuery
from retrieval.vector_search import VectorSearcher

logger = logging.getLogger(__name__)

TYPE_GROUPS = {"table": ["table"], "figure": ["figure", "chart"]}


def query_variants(rq: RefinedQuery) -> dict[str, str]:
    """Original and refined queries always; subqueries only for complex questions."""
    variants = {"original": rq.original_query}
    if rq.refined_query.strip() and rq.refined_query.strip().lower() != rq.original_query.lower():
        variants["refined"] = rq.refined_query
    complex_q = rq.intent in ("comparison", "synthesis") or len(rq.original_query.split()) >= 20
    if complex_q:
        for i, s in enumerate(rq.subqueries[:3], start=1):
            variants[f"sub{i}"] = s
    return variants


class HybridSearcher:
    def __init__(self, vector: VectorSearcher, keyword: BM25Searcher) -> None:
        self.vector, self.keyword = vector, keyword

    def search(self, rq: RefinedQuery, *, paper_ids: list[str] | None = None, page_number: int | None = None,
               section: str | None = None, top_k_dense: int | None = None, top_k_bm25: int | None = None,
               max_candidates: int = 40) -> tuple[list[Candidate], dict[str, Any]]:
        """Return (fused candidates, debug info with latency and the variants used)."""
        t0 = time.perf_counter()
        kd, kb = top_k_dense or settings.top_k_dense, top_k_bm25 or settings.top_k_bm25
        base = {"paper_ids": paper_ids, "page_number": page_number, "section": section}
        variants = query_variants(rq)
        names = list(variants)
        vectors = self.vector.embedder.embed_texts([variants[n] for n in names])  # one batched call

        tasks: dict[str, Any] = {}
        # typed passes so tables/figures are not crowded out by text when the question needs them
        typed = [t for t in rq.content_types if t in TYPE_GROUPS]
        refined_name = "refined" if "refined" in variants else "original"
        with ThreadPoolExecutor(max_workers=8) as ex:
            for n, vec in zip(names, vectors):
                tasks[f"dense:{n}"] = ex.submit(self.vector.search_vectors, [vec], kd, **base)
                tasks[f"bm25:{n}"] = ex.submit(self.keyword.search, variants[n], kb, **base)
            for t in typed:
                vec = vectors[names.index(refined_name)]
                et = TYPE_GROUPS[t]
                tasks[f"dense:{t}"] = ex.submit(self.vector.search_vectors, [vec], max(kd // 2, 5),
                                                element_types=et, **base)
                tasks[f"bm25:{t}"] = ex.submit(self.keyword.search, variants[refined_name],
                                               max(kb // 2, 5), element_types=et, **base)
            rank_lists: dict[str, list[Candidate]] = {}
            for name, fut in tasks.items():
                res = fut.result()
                rank_lists[name] = res[0] if name.startswith("dense") else res
        # original/refined carry full weight; sub-queries and typed passes slightly less
        weights = {n: (0.7 if n.endswith(("sub1", "sub2", "sub3")) else 1.0) for n in rank_lists}
        fused = reciprocal_rank_fusion(rank_lists, k=settings.rrf_k, weights=weights)[:max_candidates]
        info = {"variants": variants, "typed_passes": typed,
                "list_sizes": {n: len(v) for n, v in rank_lists.items()},
                "n_candidates": len(fused), "latency_s": time.perf_counter() - t0}
        return fused, info
