"""Step B5: Reciprocal Rank Fusion with de-duplication and provenance."""
from __future__ import annotations

from retrieval.types import Candidate


def reciprocal_rank_fusion(rank_lists: dict[str, list[Candidate]], k: int = 60,
                           weights: dict[str, float] | None = None) -> list[Candidate]:
    """Fuse named ranked lists. `name` is "<method>:<variant>" (e.g. "dense:refined").

    score(d) = sum_over_lists weight * 1 / (k + rank). Duplicates are merged by chunk_id; each result
    keeps every list rank it appeared in (`ranks`) and the retrieval methods that found it.
    """
    merged: dict[str, Candidate] = {}
    for name, items in rank_lists.items():
        w = (weights or {}).get(name, 1.0)
        method = name.split(":", 1)[0]
        for rank, cand in enumerate(items, start=1):
            cur = merged.get(cand.chunk_id)
            if cur is None:
                cur = cand.model_copy(update={"score": 0.0, "ranks": {}, "methods": []})
                merged[cand.chunk_id] = cur
            if name in cur.ranks:  # same list containing a duplicate id: count the best rank only
                continue
            cur.ranks[name] = rank
            if method not in cur.methods:
                cur.methods.append(method)
            cur.score += w / (k + rank)
    return sorted(merged.values(), key=lambda c: (-c.score, c.chunk_id))
