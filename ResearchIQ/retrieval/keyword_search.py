"""Keyword retrieval with BM25 (rank_bm25), rebuilt automatically when the index changes."""
from __future__ import annotations

import logging
import re
import threading
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from config import settings
from database.metadata_store import MetadataStore
from retrieval.types import Candidate

logger = logging.getLogger(__name__)

_STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was "
                  "were which with what how does do did we our their than then these those can using used "
                  "paper papers study".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

_version = 0
_version_lock = threading.Lock()


def bump_index_version() -> None:
    """Call after ingestion or deletion so the BM25 index is rebuilt on next use."""
    global _version
    with _version_lock:
        _version += 1


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25Searcher:
    def __init__(self, store: MetadataStore) -> None:
        self.store = store
        self._sig: tuple[Any, ...] | None = None
        self._bm25: BM25Okapi | None = None
        self._rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        sig = (_version, self.store.stats()["searchable_chunks"])
        with self._lock:
            if sig == self._sig:
                return
            rows = self.store.list_searchable_chunks()
            titles = {p["paper_id"]: p["title"] or "" for p in self.store.list_papers()}
            corpus = [tokenize(f"{titles.get(r['paper_id'], '')} {r['section'] or ''} {r['content']}")
                      for r in rows]
            self._rows = rows
            self._bm25 = BM25Okapi(corpus) if corpus and any(corpus) else None
            self._sig = sig
            logger.info("BM25 index rebuilt: %d chunks", len(rows))

    def search(self, query: str, limit: int | None = None, *, paper_ids: list[str] | None = None,
               element_types: list[str] | None = None, page_number: int | None = None,
               section: str | None = None) -> list[Candidate]:
        limit = limit or settings.top_k_bm25
        self._ensure()
        toks = tokenize(query)
        if not self._bm25 or not toks:
            return []
        scores = np.asarray(self._bm25.get_scores(toks), dtype=float)
        mask = np.ones(len(self._rows), dtype=bool)
        for i, r in enumerate(self._rows):
            if (paper_ids and r["paper_id"] not in paper_ids) or \
               (element_types and r["element_type"] not in element_types) or \
               (page_number is not None and r["page_number"] != page_number) or \
               (section and r["section"] != section):
                mask[i] = False
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:limit]
        out: list[Candidate] = []
        for i in order:
            if scores[i] <= 0:
                break
            r = self._rows[i]
            out.append(Candidate(chunk_id=r["chunk_id"], paper_id=r["paper_id"], page_number=r["page_number"],
                                 element_type=r["element_type"], chunk_type=r["chunk_type"],
                                 section=r["section"], content=r["content"], score=float(scores[i])))
        return out
