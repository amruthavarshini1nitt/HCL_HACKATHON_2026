"""Step B6: cross-encoder reranking with BAAI/bge-reranker-base (falls back to RRF order if unavailable)."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from config import settings
from retrieval.types import Candidate

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model: Any = None
        self._failed = False
        self._lock = threading.Lock()

    def _load(self) -> Any | None:
        with self._lock:
            if self._model is not None or self._failed:
                return self._model
            try:
                from sentence_transformers import CrossEncoder
                kwargs: dict[str, Any] = {"max_length": 512, "device": "cpu"}
                if settings.hf_token:
                    kwargs["token"] = settings.hf_token
                self._model = CrossEncoder(self.model_name, **kwargs)
            except Exception:  # noqa: BLE001
                self._failed = True
                logger.warning("Reranker %s unavailable; keeping RRF order", self.model_name, exc_info=True)
            return self._model

    @property
    def available(self) -> bool:
        return self._load() is not None

    def score(self, query: str, docs: list[str]) -> list[float] | None:
        model = self._load()
        if model is None or not docs:
            return None
        try:
            return [float(s) for s in model.predict([(query, d) for d in docs], batch_size=16,
                                                    show_progress_bar=False)]
        except Exception:  # noqa: BLE001
            logger.warning("reranker inference failed", exc_info=True)
            return None


def rerank(query: str, candidates: list[Candidate], reranker: Reranker, *, top_n: int | None = None,
           ensure_paper_ids: list[str] | None = None, max_candidates: int = 30
           ) -> tuple[list[Candidate], dict[str, Any]]:
    """Score (original query, chunk) pairs; keep the top_n.

    ensure_paper_ids: for cross-paper comparisons, make sure the best chunk of each listed paper
    is kept (replacing the lowest-ranked extras) so one paper cannot crowd out the others.
    """
    t0 = time.perf_counter()
    n = top_n or settings.rerank_top_n
    pool = candidates[:max_candidates]
    docs = [f"{c.section + '. ' if c.section else ''}{c.content}"[:1800] for c in pool]
    scores = reranker.score(query, docs)
    used = scores is not None
    if used:
        for c, s in zip(pool, scores):
            c.rerank_score = s
        ranked = sorted(pool, key=lambda c: -(c.rerank_score or 0.0))
    else:
        ranked = list(pool)
    selected = ranked[:n]
    if ensure_paper_ids:
        have = {c.paper_id for c in selected}
        for pid in ensure_paper_ids:
            if pid in have:
                continue
            best = next((c for c in ranked if c.paper_id == pid), None)
            if best is None:
                continue
            # replace the lowest-ranked selected chunk from a paper that has more than one chunk
            counts: dict[str, int] = {}
            for c in selected:
                counts[c.paper_id] = counts.get(c.paper_id, 0) + 1
            victim = next((c for c in reversed(selected) if counts[c.paper_id] > 1), None)
            if victim is not None:
                selected.remove(victim)
            selected.append(best)
            have.add(pid)
    selected.sort(key=lambda c: -(c.rerank_score if c.rerank_score is not None else c.score))
    return selected, {"reranker_used": used, "model": reranker.model_name if used else None,
                      "n_scored": len(pool), "latency_s": time.perf_counter() - t0}
