"""Branch 3 (figures): crop original images, drop decorative ones, link nearby explanatory text."""
from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from ingestion import pymupdf_utils as pu
from ingestion.output_normalizer import Element, NormalizedDocument
from ingestion.text_processor import find_mentions

logger = logging.getLogger(__name__)


def _is_meaningful(el: Element, min_px: int) -> bool:
    """Keep figures with a caption/label, or large enough to be a real figure (not a logo/icon)."""
    if el.caption or el.label:
        return True
    if not el.bbox:
        return False
    w, h = el.bbox[2] - el.bbox[0], el.bbox[3] - el.bbox[1]
    return w >= min_px and h >= min_px


def extract_visuals(doc: NormalizedDocument, pdf_path: Path, text_elements: list[Element],
                    figures_dir: Path | None = None) -> list[tuple[str, str, str]]:
    """Crop each meaningful figure to PNG, attach context, remove non-meaningful ones from `doc`.

    Sets `asset_path` and `metadata['context_text']` / `['context_element_ids']`.
    Returns links (src paragraph, figure, 'explains').
    """
    figures_dir = figures_dir or settings.figures_dir
    links: list[tuple[str, str, str]] = []
    keep: list[Element] = []
    by_page: dict[int, list[Element]] = {}
    for t in text_elements:
        by_page.setdefault(t.page_number, []).append(t)
    for el in doc.elements:
        if el.element_type not in ("figure", "chart"):
            keep.append(el)
            continue
        if not _is_meaningful(el, settings.min_figure_px) or not el.bbox:
            logger.debug("dropping non-meaningful figure %s", el.element_id)
            continue
        out = figures_dir / el.paper_id / f"{el.element_id}.png"
        try:
            img = pu.crop_region(pdf_path, el.page_number, el.bbox, out)
        except Exception:  # noqa: BLE001
            logger.warning("figure crop failed for %s", el.element_id, exc_info=True)
            img = None
        if img is None:
            continue
        el.asset_path = str(img)
        ctx: list[Element] = find_mentions(text_elements, el.label)[:3] if el.label else []
        if not ctx:  # fall back to the closest paragraphs on the same page
            page_paras = [t for t in by_page.get(el.page_number, []) if t.element_type == "paragraph"]
            page_paras.sort(key=lambda t: abs(((t.bbox or [0, 0, 0, 0])[1]) - el.bbox[1]))
            ctx = page_paras[:2]
        el.metadata["context_element_ids"] = [c.element_id for c in ctx]
        el.metadata["context_text"] = "\n".join(c.content for c in ctx)[:2500]
        links.extend((c.element_id, el.element_id, "explains") for c in ctx)
        keep.append(el)
    doc.elements = keep
    return links
