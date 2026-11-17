"""Branch 2 (tables): persist structured tables, render table images and link explanatory text."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from config import settings
from ingestion import pymupdf_utils as pu
from ingestion.output_normalizer import Element, NormalizedDocument
from ingestion.text_processor import find_mentions

logger = logging.getLogger(__name__)


def extract_tables(doc: NormalizedDocument, pdf_path: Path, text_elements: list[Element],
                   tables_dir: Path | None = None) -> list[tuple[str, str, str]]:
    """Fill in table assets in place and return links (src, dst, relation).

    Writes `<tables_dir>/<element_id>.json` (headers/rows/caption/page) and a cropped PNG.
    """
    tables_dir = tables_dir or settings.tables_dir
    links: list[tuple[str, str, str]] = []
    for el in doc.elements:
        if el.element_type != "table":
            continue
        record = {"element_id": el.element_id, "paper_id": el.paper_id, "page_number": el.page_number,
                  "label": el.label, "caption": el.caption, "headers": el.metadata.get("headers", []),
                  "rows": el.metadata.get("rows", [])}
        (tables_dir / el.paper_id).mkdir(parents=True, exist_ok=True)
        (tables_dir / el.paper_id / f"{el.element_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        if el.bbox and any(el.bbox):
            try:
                img = pu.crop_region(pdf_path, el.page_number, el.bbox,
                                     tables_dir / el.paper_id / f"{el.element_id}.png")
                el.asset_path = str(img) if img else None
            except Exception:  # noqa: BLE001
                logger.warning("table crop failed for %s", el.element_id, exc_info=True)
        if el.label:
            for m in find_mentions(text_elements, el.label):
                links.append((m.element_id, el.element_id, "references"))
    return links
