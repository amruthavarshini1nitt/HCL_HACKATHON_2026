"""Step A6: structure-aware parent-child chunking.

- Parents: consecutive paragraphs of one section, ~parent_chunk_tokens, never crossing a section.
- Children: ~child_chunk_tokens, packed from whole paragraphs; oversize paragraphs are split on
  sentence boundaries with ~overlap tokens. Equations are glued to the previous unit, never split.
- Tables and figures become standalone records whose chunk_id equals their element_id.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from config import settings
from ingestion.output_normalizer import Element

logger = logging.getLogger(__name__)

_enc: Any = None
_enc_failed = False
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def count_tokens(text: str) -> int:
    """Token count with tiktoken (cl100k, used by text-embedding-3-small); word-based estimate if unavailable."""
    global _enc, _enc_failed
    if _enc is None and not _enc_failed:
        try:
            import tiktoken
            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - offline or missing BPE file
            _enc_failed = True
            logger.warning("tiktoken unavailable; using word-count token estimate")
    if _enc is not None:
        return len(_enc.encode(text, disallowed_special=()))
    return int(len(text.split()) * 1.3) + 1


@dataclass
class _Unit:
    text: str
    element_ids: list[str]
    page: int
    atomic: bool = False


def _split_long(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split a long paragraph on sentence boundaries with token overlap."""
    sents = [s for s in SENT_RE.split(text) if s]
    pieces: list[str] = []
    cur: list[str] = []
    cur_t = 0
    for s in sents:
        st = count_tokens(s)
        if st > max_tokens:  # a single huge sentence: hard split by words
            words = s.split()
            step = max(20, int(max_tokens / 1.3))
            for i in range(0, len(words), step):
                pieces.append(" ".join(words[i:i + step]))
            continue
        if cur and cur_t + st > max_tokens:
            pieces.append(" ".join(cur))
            keep: list[str] = []
            kt = 0
            for prev in reversed(cur):  # carry trailing sentences as overlap
                pt = count_tokens(prev)
                if kt + pt > overlap:
                    break
                keep.insert(0, prev)
                kt += pt
            cur, cur_t = keep, kt
        cur.append(s)
        cur_t += st
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _make_children(units: list[_Unit], max_tokens: int, overlap: int) -> list[_Unit]:
    out: list[_Unit] = []
    buf: list[_Unit] = []
    buf_t = 0

    def flush() -> None:
        nonlocal buf, buf_t
        if buf:
            out.append(_Unit("\n".join(u.text for u in buf), [e for u in buf for e in u.element_ids],
                             buf[0].page))
        buf, buf_t = [], 0

    for u in units:
        t = count_tokens(u.text)
        if u.atomic and buf:  # equation: stays with the preceding text even if slightly over budget
            buf.append(u)
            buf_t += t
            continue
        if t > max_tokens:
            flush()
            for piece in _split_long(u.text, max_tokens, overlap):
                out.append(_Unit(piece, u.element_ids, u.page))
            continue
        if buf and buf_t + t > max_tokens:
            flush()
        buf.append(u)
        buf_t += t
    flush()
    return out


def chunk_document(paper_id: str, text_elements: list[Element], other_elements: list[Element]
                   ) -> list[dict[str, Any]]:
    """Return chunk records (parents, children, table and figure records) ready for SQLite."""
    cfg = settings
    chunks: list[dict[str, Any]] = []
    pn = cn = 0

    # group text into sections (a new heading starts a new group; abstract is its own group)
    groups: list[tuple[str | None, str, list[_Unit]]] = []
    for el in text_elements:
        if el.element_type == "heading":
            groups.append((el.section, "paragraph", []))
            continue
        etype = "abstract" if el.element_type == "abstract" else "paragraph"
        if not groups or groups[-1][1] != etype or (etype == "paragraph" and groups[-1][0] != el.section):
            groups.append((el.section, etype, []))
        groups[-1][2].append(_Unit(el.content, [el.element_id], el.page_number,
                                   atomic=el.element_type == "equation"))

    for section, etype, units in groups:
        if not units:
            continue
        # split group into parents at paragraph boundaries
        parents: list[list[_Unit]] = [[]]
        pt = 0
        for u in units:
            t = count_tokens(u.text)
            if parents[-1] and pt + t > cfg.parent_chunk_tokens:
                parents.append([])
                pt = 0
            parents[-1].append(u)
            pt += t
        for plist in parents:
            pn += 1
            pid = f"{paper_id}_P{pn:03d}"
            ptext = "\n".join(u.text for u in plist)
            eids = [e for u in plist for e in u.element_ids]
            chunks.append({"chunk_id": pid, "paper_id": paper_id, "parent_id": None, "chunk_type": "parent",
                           "element_type": etype, "page_number": plist[0].page, "section": section,
                           "content": ptext, "source_element_ids": eids, "token_count": count_tokens(ptext)})
            for child in _make_children(plist, cfg.child_chunk_tokens, cfg.chunk_overlap_tokens):
                if not child.text.strip():
                    continue
                cn += 1
                chunks.append({"chunk_id": f"{paper_id}_C{cn:04d}", "paper_id": paper_id, "parent_id": pid,
                               "chunk_type": "child", "element_type": etype, "page_number": child.page,
                               "section": section, "content": child.text,
                               "source_element_ids": child.element_ids,
                               "token_count": count_tokens(child.text)})

    for el in other_elements:
        if el.element_type == "table" and el.content.strip():
            chunks.append({"chunk_id": el.element_id, "paper_id": paper_id, "parent_id": None,
                           "chunk_type": "table", "element_type": "table", "page_number": el.page_number,
                           "section": el.section, "content": el.content, "source_element_ids": [el.element_id],
                           "token_count": count_tokens(el.content)})
        elif el.element_type in ("figure", "chart") and el.content.strip():
            chunks.append({"chunk_id": el.element_id, "paper_id": paper_id, "parent_id": None,
                           "chunk_type": "figure", "element_type": el.element_type,
                           "page_number": el.page_number, "section": el.section, "content": el.content,
                           "source_element_ids": [el.element_id], "token_count": count_tokens(el.content)})
    return chunks
