"""Branch 1 (text): clean text elements and expose them with section hierarchy and page refs."""
from __future__ import annotations

import re

from ingestion.output_normalizer import Element, NormalizedDocument

TEXT_TYPES = {"abstract", "heading", "paragraph", "equation"}


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)         # de-hyphenate line breaks
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text_elements(doc: NormalizedDocument) -> list[Element]:
    """Text-branch elements in reading order, cleaned. Equations are kept verbatim (not rewrapped)."""
    out: list[Element] = []
    for el in doc.elements:
        if el.element_type not in TEXT_TYPES:
            continue
        content = el.content if el.element_type == "equation" else clean_text(el.content)
        if not content:
            continue
        out.append(el.model_copy(update={"content": content}))
    return out


def find_mentions(text_elements: list[Element], label: str) -> list[Element]:
    """Paragraphs that explicitly mention a label such as 'Figure 2' or 'Table 1' (case-insensitive)."""
    kind, num = label.split()
    pat = re.compile(rf"\b{'fig(?:ure)?\\.?' if kind == 'Figure' else 'table'}\s*{re.escape(num)}\b", re.I)
    return [e for e in text_elements if e.element_type == "paragraph"
            and not e.metadata.get("is_caption") and pat.search(e.content)]
