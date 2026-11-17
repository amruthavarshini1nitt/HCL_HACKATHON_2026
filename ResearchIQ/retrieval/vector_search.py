"""Dense retrieval: embed the query with text-embedding-3-small and search Qdrant."""
from __future__ import annotations

from typing import Any, Protocol

from config import settings
from database.metadata_store import MetadataStore
from database.qdrant_store import QdrantStore
from retrieval.types import Candidate


class QueryEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def hits_to_candidates(hits: list[dict[str, Any]], store: MetadataStore) -> list[Candidate]:
    """Attach chunk text from SQLite to Qdrant hits (Qdrant stores only searchable metadata)."""
    rows = {r["chunk_id"]: r for r in store.get_chunks([h["chunk_id"] for h in hits])}
    out: list[Candidate] = []
    for h in hits:
        r = rows.get(h["chunk_id"])
        if not r:
            continue  # stale vector without a metadata row
        out.append(Candidate(chunk_id=r["chunk_id"], paper_id=r["paper_id"], page_number=r["page_number"],
                             element_type=r["element_type"], chunk_type=r["chunk_type"],
                             section=r["section"], content=r["content"], score=h["score"]))
    return out


class VectorSearcher:
    def __init__(self, qdrant: QdrantStore, store: MetadataStore, embedder: QueryEmbedder) -> None:
        self.qdrant, self.store, self.embedder = qdrant, store, embedder

    def search_vectors(self, vectors: list[list[float]], limit: int | None = None, **filters: Any
                       ) -> list[list[Candidate]]:
        limit = limit or settings.top_k_dense
        return [hits_to_candidates(self.qdrant.search(v, limit=limit, **filters), self.store)
                for v in vectors]

    def search(self, query: str, limit: int | None = None, **filters: Any) -> list[Candidate]:
        return self.search_vectors(self.embedder.embed_texts([query]), limit, **filters)[0]
