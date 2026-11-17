"""Step A3: convert raw parser items into the common normalized schema (pure, no I/O except save)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ElementType = Literal["title", "abstract", "heading", "paragraph", "equation", "table", "figure", "chart"]

LABEL_RE = re.compile(r"^\s*(fig(?:ure)?\.?|table|tab\.)\s*([A-Za-z]?\d+)", re.I)
ARXIV_RE = re.compile(r"arxiv[:\s]*(\d{4}\.\d{4,5})(v\d+)?", re.I)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
AFFIL_WORDS = re.compile(r"univers|institut|lab\b|laborator|school|department|research|center|centre|college|"
                         r"academy|google|microsoft|meta\b|openai|deepmind|corp|inc\b|ltd|@|https?:|\.edu|\.com|"
                         r"\b(china|usa|germany|france|india|japan|korea|uk|canada|singapore)\b", re.I)
MARKERS = re.compile(r"[\d\*†‡§¶♠♣♥♦]+")
ACL_ID_RE = re.compile(r"(\d{4}\.[a-z]+(?:-[a-z]+)*\.\d+)", re.I)
ARXIV_FILE_RE = re.compile(r"^(?:[0-9a-f]{16}_)?(\d{4}\.\d{4,5})(v\d+)?\.pdf$", re.I)


class Element(BaseModel):
    """One normalized document element. Schema matches the project brief."""
    paper_id: str
    element_id: str
    element_type: ElementType
    page_number: int
    section: str | None = None
    content: str = ""
    caption: str | None = None
    asset_path: str | None = None
    label: str | None = None                   # "Figure 2" / "Table 1"
    bbox: list[float] | None = None
    reading_order: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(BaseModel):
    paper_id: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    identifier: str | None = None
    source: str | None = None
    page_count: int = 0
    page_sources: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)


def normalize_label(raw: str) -> str | None:
    """'Fig. 3: ...' -> 'Figure 3', 'TABLE 2' -> 'Table 2'."""
    m = LABEL_RE.match(raw or "")
    if not m:
        return None
    kind = "Table" if m.group(1).lower().startswith("tab") else "Figure"
    return f"{kind} {m.group(2)}"


def table_to_text(headers: list[str], rows: list[list[str]], caption: str | None = None) -> str:
    """Textual representation of a table used for embedding and BM25."""
    parts = []
    if caption:
        parts.append(caption)
    if headers:
        parts.append("Columns: " + " | ".join(h for h in headers))
    for r in rows:
        cells = [f"{headers[i]}: {c}" if i < len(headers) and headers[i] else c for i, c in enumerate(r)]
        parts.append("Row: " + "; ".join(cells))
    return "\n".join(parts)


def _extract_authors(candidates: list[str], meta_author: str = "") -> list[str]:
    """Authors from PDF metadata if present, else name-like lines between title and abstract.

    Footnote markers (1,2,*,†) are stripped; any line mentioning an affiliation, address or e-mail is skipped.
    """
    if meta_author.strip():
        return [a.strip() for a in re.split(r";|,|\band\b", meta_author) if len(a.strip().split()) >= 2][:20]
    authors: list[str] = []
    for line in candidates:
        clean = MARKERS.sub(" ", line)
        if AFFIL_WORDS.search(clean):
            continue
        toks = [t.strip(" .") for t in re.split(r",|\band\b|;|\u2022|\u00b7", clean) if t.strip(" .")]
        names = [t for t in toks if 2 <= len(t.split()) <= 4
                 and all(w[0].isupper() for w in t.split() if w[0].isalpha())]
        if names and len(names) >= len(toks) / 2:
            authors.extend(n for n in (" ".join(x.split()) for x in names) if n not in authors)
    return authors[:20]


def identifier_from_filename(filename: str) -> tuple[str | None, str | None]:
    """ACL Anthology ('2025.acl-long.558') or arXiv ('2312.10997v5') ids encoded in the filename."""
    m = ARXIV_FILE_RE.match(filename or "")
    if m:
        return f"arXiv:{m.group(1)}", "arxiv.org"
    m = ACL_ID_RE.search(filename or "")
    if m:
        return f"ACL Anthology {m.group(1)}", "aclanthology.org"
    return None, None


def _extract_ids(first_page_text: str, pdf_meta: dict[str, str]) -> tuple[str | None, str | None, int | None]:
    """Return (identifier, source, year) from first-page text and PDF metadata."""
    ident = source = None
    year = None
    m = ARXIV_RE.search(first_page_text)
    if m:
        ident, source = f"arXiv:{m.group(1)}", "arxiv.org"
        yy, mm = int(m.group(1)[:2]), int(m.group(1)[2:4])
        if 1 <= mm <= 12:
            year = 2000 + yy
    else:
        d = DOI_RE.search(first_page_text) or DOI_RE.search(pdf_meta.get("subject", "") or "")
        if d:
            ident, source = d.group(1).rstrip(".,;)"), "doi.org"
    if year is None:
        cd = pdf_meta.get("creationDate", "")
        ym = re.match(r"D:(\d{4})", cd or "")
        years = [int(y) for y in YEAR_RE.findall(first_page_text)]
        if years:
            year = max(set(years), key=years.count) if len(set(years)) == 1 else min(max(years), 2100)
        elif ym:
            year = int(ym.group(1))
    return ident, source, year


def normalize(raw_items: list[dict[str, Any]], paper_id: str, page_count: int, *,
              page_sources: dict[int, str] | None = None, warnings: list[str] | None = None,
              first_page_text: str = "", pdf_meta: dict[str, str] | None = None,
              fallback_title: str | None = None, filename: str = "") -> NormalizedDocument:
    """Build the normalized document: sections, abstract, titles, caption pairing, labels."""
    pdf_meta = pdf_meta or {}
    doc = NormalizedDocument(paper_id=paper_id, page_count=page_count,
                             page_sources={str(k): v for k, v in (page_sources or {}).items()},
                             warnings=list(warnings or []))

    # pair fallback-style standalone captions with the nearest picture/table on the same page
    captions = [i for i in raw_items if i["kind"] == "caption"]
    used_caps: set[int] = set()
    for it in raw_items:
        if it["kind"] in ("picture", "table") and not it.get("caption"):
            want = "Table" if it["kind"] == "table" else "Figure"
            best, best_d = None, 1e9
            for c in captions:
                if id(c) in used_caps or c["page"] != it["page"]:
                    continue
                lab = normalize_label(c["text"])
                if not lab or not lab.startswith(want):
                    continue
                above = it["bbox"][1] - c["bbox"][3]
                below = c["bbox"][1] - it["bbox"][3]
                d = min((x for x in (above, below) if x >= -5), default=1e9)
                if d < best_d and d < 120:
                    best, best_d = c, d
            if best is not None:
                it["caption"] = best["text"]
                used_caps.add(id(best))

    title: str | None = None
    section: str | None = None
    section_stack: list[tuple[int, str]] = []
    in_abstract = False
    seen_abstract = False
    pre_abstract_text: list[str] = []
    order = 0

    def add(etype: str, page: int, content: str = "", **kw: Any) -> Element:
        nonlocal order
        order += 1
        el = Element(paper_id=paper_id, element_id=f"{paper_id}_E{order:03d}", element_type=etype,
                     page_number=page, section=section, content=content, reading_order=order, **kw)
        doc.elements.append(el)
        return el

    for it in raw_items:
        kind, page, text = it["kind"], it["page"], (it.get("text") or "").strip()
        bbox = it.get("bbox")
        src = {"parser": it.get("source", "")}
        if kind == "caption" and id(it) in used_caps:
            continue
        if kind == "title" or (title is None and kind == "heading" and page == 1 and it.get("level", 9) <= 1
                               and not seen_abstract and not re.match(r"abstract", text, re.I)):
            if title is None and text:
                title = text
                add("title", page, text, bbox=bbox, metadata=src)
                continue
        if kind == "heading":
            lvl = max(1, int(it.get("level") or 1))
            clean = re.sub(r"^\s*(\d+(\.\d+)*\.?|[IVX]+\.)\s+", "", text).strip() or text
            while section_stack and section_stack[-1][0] >= lvl:
                section_stack.pop()
            section_stack.append((lvl, clean))
            section = clean
            in_abstract = bool(re.fullmatch(r"abstract", clean, re.I))
            seen_abstract = seen_abstract or in_abstract
            add("heading", page, text, bbox=bbox, metadata={**src, "level": lvl,
                                                            "section_path": " > ".join(s for _, s in section_stack)})
            continue
        if kind in ("text", "list_item", "caption"):
            if not text:
                continue
            if not seen_abstract and re.match(r"abstract\b[\s\-:—.]*", text, re.I) and len(text) > 40:
                seen_abstract = True
                text = re.sub(r"^abstract\b[\s\-:—.]*", "", text, flags=re.I)
                add("abstract", page, text, bbox=bbox, metadata=src)
                continue
            if in_abstract:
                add("abstract", page, text, bbox=bbox, metadata=src)
                continue
            if not seen_abstract and page == 1:
                pre_abstract_text.append(text)
            lab = normalize_label(text) if kind == "caption" else None
            add("paragraph", page, text, bbox=bbox, label=lab,
                metadata={**src, "section_path": " > ".join(s for _, s in section_stack),
                          **({"is_caption": True} if kind == "caption" else {})})
        elif kind == "formula":
            add("equation", page, text, bbox=bbox, metadata=src)
        elif kind == "table":
            t = it.get("table") or {"headers": [], "rows": []}
            cap = it.get("caption")
            add("table", page, table_to_text(t["headers"], t["rows"], cap), caption=cap,
                label=normalize_label(cap or ""), bbox=bbox,
                metadata={**src, "headers": t["headers"], "rows": t["rows"]})
        elif kind == "picture":
            cap = it.get("caption")
            add("figure", page, "", caption=cap, label=normalize_label(cap or ""), bbox=bbox, metadata=src)

    meta_title = (pdf_meta.get("title") or "").strip()
    doc.title = title or (meta_title if len(meta_title) > 8 else None) or fallback_title
    doc.authors = _extract_authors(pre_abstract_text[:12], pdf_meta.get("author", ""))
    doc.identifier, doc.source, doc.year = _extract_ids(first_page_text, pdf_meta)
    if not doc.identifier:
        doc.identifier, doc.source = identifier_from_filename(filename)
    return doc


def save_normalized(doc: NormalizedDocument, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc.paper_id}.json"
    path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_normalized(path: Path) -> NormalizedDocument:
    return NormalizedDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))
