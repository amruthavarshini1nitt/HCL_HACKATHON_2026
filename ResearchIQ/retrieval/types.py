"""Shared retrieval data types."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Intent = Literal["factual", "methodology", "comparison", "visual", "results", "limitations",
                 "synthesis", "other"]
ContentType = Literal["text", "table", "figure"]


class RefinedQuery(BaseModel):
    """Structured output of the query refiner (schema from the project brief + in_scope flag)."""
    original_query: str
    refined_query: str
    intent: Intent = "factual"
    keywords: list[str] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list, max_length=3)
    content_types: list[ContentType] = Field(default_factory=lambda: ["text"])
    in_scope: bool = True
    refiner_used_llm: bool = True


class Candidate(BaseModel):
    """A retrieved chunk with provenance of how it was found."""
    chunk_id: str
    paper_id: str
    page_number: int | None = None
    element_type: str
    chunk_type: str = "child"
    section: str | None = None
    content: str = ""
    score: float = 0.0                                   # current ranking score (RRF, then reranker)
    ranks: dict[str, int] = Field(default_factory=dict)  # "dense:original" -> 1-based rank
    methods: list[str] = Field(default_factory=list)
    rerank_score: float | None = None


class EvidenceRecord(BaseModel):
    """Evidence passed to answer generation and shown in the UI."""
    source_id: str                        # chunk_id (== element_id for tables/figures)
    paper_id: str
    paper_title: str | None = None
    page_number: int | None = None
    section: str | None = None
    element_type: str
    content: str
    parent_content: str | None = None
    label: str | None = None              # "Figure 2" / "Table 1" when applicable
    caption: str | None = None
    asset_path: str | None = None
    related_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    retrieval: dict[str, Any] = Field(default_factory=dict)
    derived_from_vlm: bool = False        # content is a model-generated visual description
    injection_flag: bool = False          # text has instruction-like phrases (treated as data only)
