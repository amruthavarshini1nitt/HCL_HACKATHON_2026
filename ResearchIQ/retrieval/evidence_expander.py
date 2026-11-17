"""Step B7: turn reranked chunks into evidence records with parent context, captions and figure assets."""
from __future__ import annotations

import logging
from typing import Any

from database.metadata_store import MetadataStore
from guardrails.injection_detector import strip_instruction_sentences
from retrieval.types import Candidate, EvidenceRecord, Intent

logger = logging.getLogger(__name__)

PARENT_INTENTS: set[str] = {"methodology", "comparison", "synthesis", "limitations"}
SHORT_CHUNK_TOKENS = 80


def _visual_record(el: dict[str, Any], title: str | None, score: float, how: str) -> EvidenceRecord:
    meta = el.get("metadata") or {}
    return EvidenceRecord(
        source_id=el["element_id"], paper_id=el["paper_id"], paper_title=title, page_number=el["page_number"],
        section=el["section"], element_type=el["element_type"], content=el["content"] or el["caption"] or "",
        parent_content=meta.get("context_text") or None, label=el["label"], caption=el["caption"],
        asset_path=el["asset_path"], score=score, retrieval={"how": how},
        derived_from_vlm=meta.get("vlm_status") == "ok")


def expand_evidence(candidates: list[Candidate], intent: Intent, store: MetadataStore,
                    *, max_extra: int = 3) -> list[EvidenceRecord]:
    """Build evidence records. Parent sections are added only when the intent or chunk size needs them."""
    titles = {p["paper_id"]: p["title"] for p in store.list_papers()}
    records: list[EvidenceRecord] = []
    seen: set[str] = {c.chunk_id for c in candidates}
    used_parents: set[str] = set()
    extras: list[EvidenceRecord] = []
    want_tables = intent in ("results", "comparison", "methodology")
    want_figs = intent in ("visual", "methodology")

    for c in candidates:
        score = c.rerank_score if c.rerank_score is not None else c.score
        retrieval = {"methods": c.methods, "ranks": c.ranks, "rrf_score": c.score,
                     "rerank_score": c.rerank_score}
        if c.element_type in ("table", "figure", "chart"):
            el = store.get_element(c.chunk_id)
            if el:
                rec = _visual_record(el, titles.get(c.paper_id), score, "retrieved")
                rec.retrieval = retrieval
                records.append(rec)
                continue
        chunk = store.get_chunk(c.chunk_id)
        parent_text = None
        if chunk and chunk["parent_id"] and chunk["parent_id"] not in used_parents:
            if intent in PARENT_INTENTS or (chunk["token_count"] or 0) < SHORT_CHUNK_TOKENS:
                parent = store.get_chunk(chunk["parent_id"])
                if parent and parent["content"] != c.content:
                    parent_text = parent["content"]
                    used_parents.add(chunk["parent_id"])
        records.append(EvidenceRecord(
            source_id=c.chunk_id, paper_id=c.paper_id, paper_title=titles.get(c.paper_id),
            page_number=c.page_number, section=c.section, element_type=c.element_type, content=c.content,
            parent_content=parent_text, score=score, retrieval=retrieval))
        # linked tables / figures mentioned by the paragraphs this chunk was built from
        rel_ids: list[str] = []
        for eid in (chunk or {}).get("source_element_ids") or []:
            for link in store.get_links(eid):
                other = link["dst_element_id"] if link["src_element_id"] == eid else link["src_element_id"]
                if other == eid:
                    continue
                el = store.get_element(other)
                if not el or el["element_type"] not in ("table", "figure", "chart"):
                    continue
                if other not in rel_ids:
                    rel_ids.append(other)
                is_table = el["element_type"] == "table"
                if other not in seen and len(extras) < max_extra and \
                        ((is_table and want_tables) or (not is_table and want_figs)):
                    seen.add(other)
                    extras.append(_visual_record(el, titles.get(el["paper_id"]), score * 0.5,
                                                 f"linked from {c.chunk_id}"))
        records[-1].related_ids = rel_ids
    final = records + extras
    for r in final:  # retrieved text is evidence, never instructions: strip, flag and log
        changed = False
        for attr in ("content", "parent_content", "caption"):
            val = getattr(r, attr)
            if val:
                clean, hit = strip_instruction_sentences(val)
                if hit:
                    setattr(r, attr, clean)
                    changed = True
        if changed:
            r.injection_flag = True
            logger.warning("instruction-like sentences removed from evidence %s", r.source_id)
    return final
