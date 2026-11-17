"""PyMuPDF helpers: page text, rendering, cropping, a structural fallback parser, and OCR.

All parser output uses the "raw item" dict shape consumed by `output_normalizer`:
    {"kind": title|heading|text|caption|formula|table|picture|list_item,
     "text": str, "page": int (1-based), "bbox": [x0,y0,x1,y1] (top-left origin, PDF points),
     "level": int, "caption": str|None, "table": {"headers": [...], "rows": [[...]]} | None,
     "source": "docling"|"pymupdf"|"ocr"}
"""
from __future__ import annotations

import logging
import re
import statistics
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from config import settings

logger = logging.getLogger(__name__)

CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?|table|tab\.)\s*([A-Za-z]?\d+)", re.I)


# ---------------- basic page utilities ----------------
def page_texts(pdf_path: Path | str) -> list[str]:
    """Plain text per page (index 0 = page 1)."""
    with fitz.open(str(pdf_path)) as doc:
        return [p.get_text("text") for p in doc]


def scanned_pages(texts: list[str], min_chars: int | None = None) -> list[int]:
    """1-based page numbers whose extracted text is too short to be a real text page."""
    limit = settings.ocr_min_chars_per_page if min_chars is None else min_chars
    return [i + 1 for i, t in enumerate(texts) if len(t.strip()) < limit]


def render_page(pdf_path: Path | str, page_number: int, out_path: Path, dpi: int = 150) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(str(pdf_path)) as doc:
        pix = doc.load_page(page_number - 1).get_pixmap(dpi=dpi)
        pix.save(str(out_path))
    return out_path


def crop_region(pdf_path: Path | str, page_number: int, bbox: list[float], out_path: Path,
                dpi: int = 200, pad: float = 4.0) -> Path | None:
    """Render a bbox region of a page to PNG. Returns None if the region is degenerate."""
    with fitz.open(str(pdf_path)) as doc:
        page = doc.load_page(page_number - 1)
        r = fitz.Rect(*bbox) + (-pad, -pad, pad, pad)
        r = r & page.rect
        if r.is_empty or r.width < 8 or r.height < 8:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        page.get_pixmap(dpi=dpi, clip=r).save(str(out_path))
    return out_path


def page_size(pdf_path: Path | str, page_number: int) -> tuple[float, float]:
    with fitz.open(str(pdf_path)) as doc:
        r = doc.load_page(page_number - 1).rect
        return r.width, r.height


# ---------------- structural fallback parser ----------------
def _clean(t: str) -> str:
    return re.sub(r"[ \t]+", " ", t.replace("\u00ad", "")).strip()


def _center_in(r: fitz.Rect, rects: list[fitz.Rect]) -> bool:
    c = fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
    return any(x.contains(c) for x in rects)


def text_chars_outside(pdf_path: Path | str, page_number: int, rects: list[list[float]]) -> int:
    """Characters of raw page text whose word box lies outside the given (table/figure) regions."""
    with fitz.open(str(pdf_path)) as doc:
        page = doc.load_page(page_number - 1)
        ex = [fitz.Rect(*r) for r in rects]
        return sum(len(w[4]) for w in page.get_text("words") if not _center_in(fitz.Rect(w[:4]), ex))


def parse_pages_with_pymupdf(pdf_path: Path | str, pages: list[int] | None = None, *,
                             exclude_rects: dict[int, list[list[float]]] | None = None) -> list[dict[str, Any]]:
    """Heuristic structure extraction (headings by font size/weight, captions, tables, images).

    Used as fallback when Docling fails or misses a page. Quality is lower than Docling.
    exclude_rects: per-page regions already covered by another parser (e.g. Docling tables/figures);
    text inside them is skipped and no tables/images are emitted for those pages, avoiding duplicates.
    """
    items: list[dict[str, Any]] = []
    with fitz.open(str(pdf_path)) as doc:
        sizes: list[float] = []
        for p in doc:
            for b in p.get_text("dict")["blocks"]:
                for ln in b.get("lines", []):
                    for sp in ln["spans"]:
                        sizes.extend([sp["size"]] * max(1, len(sp["text"])))
        body = statistics.median(sizes) if sizes else 10.0
        for p in doc:
            pno = p.number + 1
            if pages and pno not in pages:
                continue
            ph = p.rect.height
            covered = [fitz.Rect(*r) for r in (exclude_rects or {}).get(pno, [])]
            visuals_elsewhere = bool(covered)
            # tables first so their text isn't duplicated as paragraphs
            table_rects: list[fitz.Rect] = list(covered)
            try:
                for tb in ([] if visuals_elsewhere else p.find_tables().tables):
                    rows = [[_clean(c or "") for c in r] for r in tb.extract()]
                    rows = [r for r in rows if any(r)]
                    if len(rows) >= 2 and len(rows[0]) >= 2:
                        table_rects.append(fitz.Rect(tb.bbox))
                        items.append({"kind": "table", "text": "", "page": pno, "bbox": list(tb.bbox),
                                      "level": 0, "caption": None,
                                      "table": {"headers": rows[0], "rows": rows[1:]}, "source": "pymupdf"})
            except Exception:  # noqa: BLE001
                logger.debug("find_tables failed on page %d", pno, exc_info=True)
            for blk in p.get_text("dict")["blocks"]:
                if blk.get("type") != 0:
                    continue
                r = fitz.Rect(blk["bbox"])
                if any(r.intersects(t) and (r & t).get_area() > 0.5 * r.get_area() for t in table_rects) \
                        or _center_in(r, covered):
                    continue
                if r.y1 < 30 or r.y0 > ph - 30:  # page header/footer margins
                    continue
                spans = [s for ln in blk["lines"] for s in ln["spans"]]
                text = _clean(" ".join(" ".join(s["text"] for s in ln["spans"]) for ln in blk["lines"]))
                if not text or not spans:
                    continue
                size = statistics.mean(s["size"] for s in spans)
                bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
                kind, level = "text", 0
                if CAPTION_RE.match(text):
                    kind = "caption"
                elif len(text) < 150 and "\n" not in text and (size >= body * 1.15 or (bold and len(text) < 90)):
                    kind, level = "heading", (1 if size >= body * 1.4 else 2)
                items.append({"kind": kind, "text": text, "page": pno, "bbox": list(r), "level": level,
                              "caption": None, "table": None, "source": "pymupdf"})
            for info in ([] if visuals_elsewhere else p.get_image_info()):
                r = fitz.Rect(info["bbox"])
                if r.width >= settings.min_figure_px * 0.5 and r.height >= settings.min_figure_px * 0.5:
                    items.append({"kind": "picture", "text": "", "page": pno, "bbox": list(r), "level": 0,
                                  "caption": None, "table": None, "source": "pymupdf"})
    return items


# ---------------- OCR (PaddleOCR, lazy) ----------------
_ocr_engine: Any = None
_ocr_failed = False


def _get_ocr() -> Any | None:
    global _ocr_engine, _ocr_failed
    if _ocr_engine is not None or _ocr_failed:
        return _ocr_engine
    try:
        from paddleocr import PaddleOCR  # heavy import, only when needed
        _ocr_engine = PaddleOCR(lang="en")
    except Exception:  # noqa: BLE001
        _ocr_failed = True
        logger.warning("PaddleOCR unavailable; scanned pages will have no OCR text", exc_info=True)
    return _ocr_engine


def _extract_ocr_lines(result: Any) -> list[tuple[str, list[float] | None]]:
    """Normalise PaddleOCR 2.x and 3.x outputs to [(text, [x0,y0,x1,y1] | None)]."""
    lines: list[tuple[str, list[float] | None]] = []
    if not result:
        return lines
    for page in result:
        if page is None:
            continue
        if isinstance(page, dict) or hasattr(page, "get"):  # 3.x
            texts = page.get("rec_texts") or []
            boxes = page.get("rec_boxes")
            for i, t in enumerate(texts):
                b = None
                if boxes is not None and i < len(boxes):
                    b = [float(x) for x in boxes[i]]
                lines.append((str(t), b))
        else:  # 2.x: [[box_points, (text, conf)], ...]
            for entry in page:
                try:
                    pts, (text, _conf) = entry
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    lines.append((str(text), [min(xs), min(ys), max(xs), max(ys)]))
                except (TypeError, ValueError):
                    continue
    return lines


def ocr_page(pdf_path: Path | str, page_number: int, dpi: int = 200) -> list[dict[str, Any]]:
    """OCR one page into paragraph-level raw items (grouped by vertical gaps)."""
    engine = _get_ocr()
    if engine is None:
        return []
    tmp = settings.cache_dir / f"ocr_{page_number}.png"
    render_page(pdf_path, page_number, tmp, dpi=dpi)
    try:
        res = engine.ocr(str(tmp)) if hasattr(engine, "ocr") else engine.predict(str(tmp))
        lines = _extract_ocr_lines(res)
    except Exception:  # noqa: BLE001
        logger.warning("OCR failed on page %d", page_number, exc_info=True)
        return []
    finally:
        tmp.unlink(missing_ok=True)
    scale = 72.0 / dpi
    items: list[dict[str, Any]] = []
    cur: list[str] = []
    cur_box: list[float] | None = None
    prev_bottom: float | None = None
    heights = [b[3] - b[1] for _, b in lines if b]
    gap = 1.2 * (statistics.median(heights) if heights else 0)

    def flush() -> None:
        nonlocal cur, cur_box
        if cur:
            bb = [v * scale for v in cur_box] if cur_box else [0, 0, 0, 0]
            items.append({"kind": "text", "text": _clean(" ".join(cur)), "page": page_number, "bbox": bb,
                          "level": 0, "caption": None, "table": None, "source": "ocr"})
        cur, cur_box = [], None

    for text, b in lines:
        if b and prev_bottom is not None and gap and b[1] - prev_bottom > gap:
            flush()
        cur.append(text)
        if b:
            cur_box = b[:] if cur_box is None else [min(cur_box[0], b[0]), min(cur_box[1], b[1]),
                                                    max(cur_box[2], b[2]), max(cur_box[3], b[3])]
            prev_bottom = b[3]
    flush()
    return items
