"""Step A2: structured PDF parsing. Docling first, PyMuPDF verification/fallback, PaddleOCR for scans.

Per-page source selection avoids duplicated elements: each page's items come from exactly one parser.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings
from ingestion import pymupdf_utils as pu

logger = logging.getLogger(__name__)

_converter: Any = None


@dataclass
class ParseResult:
    items: list[dict[str, Any]]
    page_count: int
    page_sources: dict[int, str] = field(default_factory=dict)  # page -> docling|pymupdf|ocr
    warnings: list[str] = field(default_factory=list)


def _get_converter() -> Any:
    """Create the Docling converter once. OCR is disabled in Docling; PaddleOCR handles scans."""
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True
        _converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converter


def _parse_with_docling(pdf_path: Path) -> list[dict[str, Any]]:
    from docling_core.types.doc import DocItemLabel, PictureItem, TableItem

    result = _get_converter().convert(str(pdf_path))
    doc = result.document
    items: list[dict[str, Any]] = []
    skip = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.FOOTNOTE}
    # captions already attached to a table/picture must not be emitted again (avoids duplicates)
    used_caps = {(it.caption_text(doc) or "").strip() for it, _ in doc.iterate_items()
                 if isinstance(it, (TableItem, PictureItem))} - {""}
    for item, level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        p = prov[0]
        page_no = int(p.page_no)
        try:
            ph = doc.pages[page_no].size.height
            b = p.bbox.to_top_left_origin(page_height=ph)
            bbox = [float(b.l), float(b.t), float(b.r), float(b.b)]
        except Exception:  # noqa: BLE001
            bbox = [0.0, 0.0, 0.0, 0.0]
        label = getattr(item, "label", None)
        base = {"page": page_no, "bbox": bbox, "level": 0, "caption": None, "table": None,
                "source": "docling"}
        if isinstance(item, TableItem):
            headers: list[str] = []
            rows: list[list[str]] = []
            try:
                df = item.export_to_dataframe(doc)
                headers = [str(c) for c in df.columns]
                rows = [[str(v) for v in r] for r in df.values.tolist()]
            except Exception:  # noqa: BLE001
                logger.debug("table export failed", exc_info=True)
            if not headers and not rows:
                continue
            items.append({**base, "kind": "table", "text": "", "caption": item.caption_text(doc) or None,
                          "table": {"headers": headers, "rows": rows}})
        elif isinstance(item, PictureItem):
            items.append({**base, "kind": "picture", "text": "", "caption": item.caption_text(doc) or None})
        elif label in skip:
            continue
        else:
            text = (getattr(item, "text", "") or "").strip()
            if not text or (label == DocItemLabel.CAPTION and text in used_caps):
                continue
            if label == DocItemLabel.CAPTION:
                kind = "caption"
            elif label == DocItemLabel.TITLE:
                kind = "title"
            elif label == DocItemLabel.SECTION_HEADER:
                kind, base["level"] = "heading", int(getattr(item, "level", 1) or 1)
            elif label == DocItemLabel.FORMULA:
                kind = "formula"
            elif label == DocItemLabel.LIST_ITEM:
                kind = "list_item"
            else:
                kind = "text"
            items.append({**base, "kind": kind, "text": text})
    return items


def parse_pdf(pdf_path: Path | str) -> ParseResult:
    """Parse a PDF into raw items with per-page parser selection and OCR for scanned pages."""
    path = Path(pdf_path)
    texts = pu.page_texts(path)
    n = len(texts)
    warnings: list[str] = []
    docling_items: list[dict[str, Any]] = []
    try:
        docling_items = _parse_with_docling(path)
    except Exception:  # noqa: BLE001
        logger.warning("Docling failed on %s; using PyMuPDF fallback", path.name, exc_info=True)
        warnings.append("Docling failed; PyMuPDF structural fallback used.")

    by_page: dict[int, list[dict[str, Any]]] = {}
    for it in docling_items:
        by_page.setdefault(it["page"], []).append(it)

    low_text = set(pu.scanned_pages(texts))  # pages where even raw PyMuPDF text is thin
    items: list[dict[str, Any]] = []
    sources: dict[int, str] = {}
    fallback_pages: list[int] = []
    for pno in range(1, n + 1):
        d_items = by_page.get(pno, [])
        d_chars = sum(len(i["text"]) for i in d_items if i["kind"] not in ("table", "picture"))
        if pno in low_text and d_chars < settings.ocr_min_chars_per_page:
            ocr_items = pu.ocr_page(path, pno)
            if ocr_items:
                items.extend(ocr_items)
                sources[pno] = "ocr"
                items.extend(i for i in d_items if i["kind"] in ("table", "picture"))
                continue
            warnings.append(f"Page {pno} looks scanned but OCR produced no text.")
        # Compare Docling's prose with raw text OUTSIDE its table/figure regions: text drawn inside
        # figures and tables is not prose, and counting it wrongly flagged figure-heavy pages.
        visual_rects = [i["bbox"] for i in d_items if i["kind"] in ("table", "picture") and any(i["bbox"])]
        raw_prose = pu.text_chars_outside(path, pno, visual_rects) if visual_rects else len(texts[pno - 1].strip())
        if d_items and (d_chars >= 0.3 * raw_prose or pno in low_text):
            items.extend(d_items)
            sources[pno] = "docling"
        else:
            fallback_pages.append(pno)  # Docling missed or truncated this page's text
    if fallback_pages:
        # Keep Docling's tables/figures on fallback pages (PyMuPDF cannot detect vector figures);
        # take only the text from PyMuPDF, skipping text inside those regions.
        keep: dict[int, list[list[float]]] = {}
        for p in fallback_pages:
            vis = [i for i in by_page.get(p, []) if i["kind"] in ("table", "picture") and any(i["bbox"])]
            items.extend(vis)
            keep[p] = [i["bbox"] for i in vis]
        fb = pu.parse_pages_with_pymupdf(path, fallback_pages, exclude_rects={p: r for p, r in keep.items() if r})
        docling_caps = {(i.get("caption") or "").strip() for p in fallback_pages for i in by_page.get(p, [])} - {""}
        fb = [i for i in fb if not (i["kind"] == "caption" and i["text"].strip() in docling_caps)]
        items.extend(fb)
        for p in fallback_pages:
            sources[p] = "pymupdf"
        if docling_items:
            warnings.append(f"PyMuPDF fallback used for pages {fallback_pages}.")
    # Stable sort by page only: within a page, parser reading order (multi-column aware) is kept.
    items.sort(key=lambda i: i["page"])
    return ParseResult(items=items, page_count=n, page_sources=sources, warnings=warnings)
