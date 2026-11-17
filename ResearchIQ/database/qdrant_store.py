"""Qdrant wrapper: one collection holding text, table and visual-description vectors."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from config import settings

logger = logging.getLogger(__name__)

PAYLOAD_INDEXES = {
    "paper_id": models.PayloadSchemaType.KEYWORD,
    "element_type": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.KEYWORD,
    "page_number": models.PayloadSchemaType.INTEGER,
}


def point_id(chunk_id: str) -> str:
    """Deterministic UUID per chunk id so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    """Dense vector store with payload filtering by paper, element type, page and section."""

    def __init__(self, client: QdrantClient | None = None, collection: str | None = None,
                 dim: int | None = None) -> None:
        self.collection = collection or settings.qdrant_collection
        self.dim = dim or settings.embedding_dim
        if client is not None:
            self.client = client
        elif settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))
        self.ensure_collection()

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )
            logger.info("created Qdrant collection %s (dim=%d)", self.collection, self.dim)
        for field, schema in PAYLOAD_INDEXES.items():
            try:  # local mode ignores indexes; server mode needs them for fast filters
                self.client.create_payload_index(self.collection, field, schema)
            except Exception:  # noqa: BLE001 - index may already exist
                pass

    def upsert(self, items: list[dict[str, Any]]) -> None:
        """items: [{chunk_id, vector, payload}]. Payload must include paper_id, element_type."""
        points = []
        for it in items:
            if len(it["vector"]) != self.dim:
                raise ValueError(f"vector dim {len(it['vector'])} != collection dim {self.dim}")
            points.append(models.PointStruct(
                id=point_id(it["chunk_id"]), vector=it["vector"],
                payload={**it["payload"], "chunk_id": it["chunk_id"]}))
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[i:i + 128])

    @staticmethod
    def build_filter(paper_ids: list[str] | None = None, element_types: list[str] | None = None,
                     page_number: int | None = None, section: str | None = None) -> models.Filter | None:
        must: list[models.Condition] = []
        if paper_ids:
            must.append(models.FieldCondition(key="paper_id", match=models.MatchAny(any=paper_ids)))
        if element_types:
            must.append(models.FieldCondition(key="element_type", match=models.MatchAny(any=element_types)))
        if page_number is not None:
            must.append(models.FieldCondition(key="page_number", match=models.MatchValue(value=page_number)))
        if section:
            must.append(models.FieldCondition(key="section", match=models.MatchValue(value=section)))
        return models.Filter(must=must) if must else None

    def search(self, vector: list[float], limit: int = 20, **filters: Any) -> list[dict[str, Any]]:
        res = self.client.query_points(
            self.collection, query=vector, limit=limit,
            query_filter=self.build_filter(**filters), with_payload=True).points
        return [{"chunk_id": p.payload["chunk_id"], "score": p.score, "payload": p.payload} for p in res]

    def delete_paper(self, paper_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=self.build_filter(paper_ids=[paper_id])))

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def close(self) -> None:
        self.client.close()
