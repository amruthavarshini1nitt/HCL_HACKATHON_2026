"""Step A8: orchestrate ingestion of one paper and index it into SQLite + Qdrant (idempotent)."""
from __future__ import annotations

import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import fitz

from config import settings
from database.metadata_store import MetadataStore
from database.qdrant_store import QdrantStore
from generation.llm_client import ChatClient, get_vlm
from ingestion import pymupdf_utils as pu
from ingestion.chunker import chunk_document
from ingestion.embedding import Embedder, EmbeddingError
from ingestion.output_normalizer import NormalizedDocument, normalize, save_normalized
from ingestion.pdf_parser import parse_pdf
from ingestion.table_extractor import extract_tables
from ingestion.text_processor import extract_text_elements
from ingestion.visual_analyzer import describe_image, description_to_text, save_description
from ingestion.visual_extractor import extract_visuals
from retrieval.keyword_search import bump_index_version

logger = logging.getLogger(__name__)
Progress = Callable[[str, float], None]


@dataclass
class IngestReport:
    paper_id: str
    ok: bool
    message: str = ""
    skipped: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def embed_text_for(chunk: dict[str, Any], title: str | None) -> str:
    """Text actually embedded: content plus paper title and section for disambiguation."""
    head = " | ".join(x for x in (title, chunk.get("section")) if x)
    return f"{head}\n{chunk['content']}" if head else chunk["content"]


def remove_paper_index(paper_id: str, store: MetadataStore, qdrant: QdrantStore, *,
                       delete_files: bool = False) -> None:
    """Remove a paper's vectors, rows and (optionally) derived files. Keeps the catalogue row unless deleted."""
    qdrant.delete_paper(paper_id)
    store.clear_paper_content(paper_id)
    for d in (settings.figures_dir, settings.tables_dir):
        shutil.rmtree(d / paper_id, ignore_errors=True)
    (settings.normalized_dir / f"{paper_id}.json").unlink(missing_ok=True)
    (settings.chunks_dir / f"{paper_id}.json").unlink(missing_ok=True)
    bump_index_version()
    if delete_files:
        p = store.get_paper(paper_id)
        if p:
            Path(p["storage_path"]).unlink(missing_ok=True)


def ingest_paper(paper_id: str, store: MetadataStore, qdrant: QdrantStore, embedder: Embedder | None = None,
                 *, force: bool = False, vlm: ChatClient | None = None,
                 progress: Progress | None = None) -> IngestReport:
    """Run the full offline pipeline for one registered paper."""
    paper = store.get_paper(paper_id)
    if not paper:
        return IngestReport(paper_id, False, "Unknown paper.")
    if paper["status"] == "indexed" and not force:
        return IngestReport(paper_id, True, "Already indexed.", skipped=True)
    pdf = Path(paper["storage_path"])
    if not pdf.exists():
        store.update_paper(paper_id, status="failed", error="Stored PDF file is missing.")
        return IngestReport(paper_id, False, "Stored PDF file is missing.")

    def tick(msg: str, frac: float) -> None:
        logger.info("[%s] %s", paper_id, msg)
        if progress:
            progress(msg, frac)

    embedder = embedder or Embedder()
    vlm = vlm or get_vlm()
    timings: dict[str, float] = {}
    t_all = time.perf_counter()
    store.update_paper(paper_id, status="processing", error=None)
    try:
        remove_paper_index(paper_id, store, qdrant)

        t = time.perf_counter()
        tick("Parsing PDF (Docling / PyMuPDF / OCR)", 0.1)
        parsed = parse_pdf(pdf)
        timings["parse"] = time.perf_counter() - t

        t = time.perf_counter()
        tick("Normalizing structure", 0.3)
        with fitz.open(str(pdf)) as f:
            meta = {k: (v or "") for k, v in (f.metadata or {}).items()}
        first_text = pu.page_texts(pdf)[0] if parsed.page_count else ""
        doc: NormalizedDocument = normalize(
            parsed.items, paper_id, parsed.page_count, page_sources=parsed.page_sources,
            warnings=parsed.warnings, first_page_text=first_text, pdf_meta=meta,
            fallback_title=Path(paper["original_filename"]).stem, filename=paper["original_filename"])
        text_els = extract_text_elements(doc)
        links = extract_tables(doc, pdf, text_els)
        links += extract_visuals(doc, pdf, text_els)
        timings["normalize_extract"] = time.perf_counter() - t

        t = time.perf_counter()
        tick("Describing figures (VLM, cached)", 0.45)
        figs = [e for e in doc.elements if e.element_type in ("figure", "chart") and e.asset_path]
        vlm_calls = cached = failed = 0

        def work(el: Any) -> tuple[Any, Any, bool]:
            d, from_cache = describe_image(Path(el.asset_path), el.caption, el.metadata.get("context_text"), vlm)
            return el, d, from_cache

        with ThreadPoolExecutor(max_workers=3) as ex:
            for el, desc, from_cache in ex.map(work, figs):
                label = el.label or "Figure"
                if desc is None:
                    failed += 1
                    el.metadata["vlm_status"] = "unavailable"
                    el.content = el.caption or ""   # still searchable by caption
                    continue
                cached += from_cache
                vlm_calls += not from_cache
                el.content = description_to_text(desc, label, el.caption)
                el.metadata.update(vlm_status="ok", visual_type=desc.visual_type, uncertain=desc.uncertain)
                if desc.is_chart:
                    el.element_type = "chart"
                save_description(desc, el.element_id, paper_id)
        timings["visual"] = time.perf_counter() - t
        if failed:
            doc.warnings.append(f"{failed} figure(s) could not be described (vision model unavailable).")

        t = time.perf_counter()
        tick("Chunking", 0.65)
        others = [e for e in doc.elements if e.element_type in ("table", "figure", "chart")]
        chunks = chunk_document(paper_id, text_els, others)
        timings["chunk"] = time.perf_counter() - t

        t = time.perf_counter()
        tick("Embedding", 0.75)
        searchable = [c for c in chunks if c["chunk_type"] != "parent"]
        vectors = embedder.embed_texts([embed_text_for(c, doc.title) for c in searchable])
        timings["embed"] = time.perf_counter() - t

        t = time.perf_counter()
        tick("Indexing", 0.9)
        store.add_elements([{**e.model_dump(), } for e in doc.elements])
        store.add_links(links)
        store.add_chunks(chunks)
        qdrant.upsert([{
            "chunk_id": c["chunk_id"], "vector": v,
            "payload": {"paper_id": paper_id, "element_type": c["element_type"],
                        "chunk_type": c["chunk_type"], "page_number": c["page_number"],
                        "section": c.get("section"), "parent_id": c.get("parent_id")}}
            for c, v in zip(searchable, vectors)])
        save_normalized(doc, settings.normalized_dir)
        (settings.chunks_dir / f"{paper_id}.json").write_text(
            json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        timings["index"] = time.perf_counter() - t
        bump_index_version()
        timings["total"] = time.perf_counter() - t_all

        store.update_paper(paper_id, title=doc.title or paper["title"], authors=doc.authors,
                           year=doc.year, source=doc.source, identifier=doc.identifier,
                           page_count=doc.page_count, status="indexed", ingest_seconds=timings["total"])
        tick("Done", 1.0)
        counts = {"elements": len(doc.elements), "chunks": len(searchable),
                  "tables": sum(e.element_type == "table" for e in doc.elements),
                  "visuals": len(figs), "vlm_calls": vlm_calls, "vlm_cache_hits": cached}
        return IngestReport(paper_id, True, "Indexed.", timings=timings, counts=counts, warnings=doc.warnings)
    except EmbeddingError as e:
        store.update_paper(paper_id, status="failed", error=str(e))
        return IngestReport(paper_id, False, str(e))
    except Exception:  # noqa: BLE001 - never leak internals to the UI
        logger.exception("ingestion failed for %s", paper_id)
        store.update_paper(paper_id, status="failed", error="Processing failed. See server logs.")
        return IngestReport(paper_id, False, "Processing failed. See server logs.")
