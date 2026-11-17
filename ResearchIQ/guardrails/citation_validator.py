"""Citation existence and reference validation (does NOT judge whether a citation supports a claim)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from retrieval.types import EvidenceRecord

INLINE_ID = re.compile(r"\[([A-Za-z0-9_:#\-]+(?:\s*,\s*[A-Za-z0-9_:#\-]+)*)\]")
PAPER_ID = re.compile(r"\bP\d{3}\b")
FIG_TAB = re.compile(r"\b(fig(?:ure)?\.?|table)\s*([A-Za-z]?\d+)\b", re.I)
PAGE_REF = re.compile(r"\b(?:pages?|pp?\.)\s*(\d{1,4})(?:\s*[-–]\s*(\d{1,4}))?", re.I)


@dataclass
class CitationIssue:
    kind: str                  # unknown_source | unknown_paper | unknown_figure | unknown_page | no_citation
    detail: str
    claim_index: int | None = None


@dataclass
class CitationReport:
    issues: list[CitationIssue] = field(default_factory=list)
    invalid_claims: set[int] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not self.issues


def _norm(kind: str, num: str) -> str:
    return f"{'Table' if kind.lower().startswith('tab') else 'Figure'} {num}"


def validate_citations(answer: str, claims: list[dict], evidence: list[EvidenceRecord]) -> CitationReport:
    """Check every cited id, and every paper/figure/table/page named in the text, against the evidence set.

    claims: dicts with text, source_ids, kind.
    """
    rep = CitationReport()
    ids = {e.source_id for e in evidence}
    papers = {e.paper_id for e in evidence}
    labels = {e.label for e in evidence if e.label}
    for e in evidence:  # labels the evidence itself names (e.g. "as shown in Figure 3")
        labels.update(_norm(k, n) for k, n in FIG_TAB.findall(f"{e.content} {e.caption or ''}"))
    pages = {e.page_number for e in evidence if e.page_number is not None}

    for i, c in enumerate(claims):
        sids = c.get("source_ids") or []
        if c.get("kind") != "suggestion" and not sids:
            rep.issues.append(CitationIssue("no_citation", c.get("text", "")[:80], i))
            rep.invalid_claims.add(i)
        for s in sids:
            if s not in ids:
                rep.issues.append(CitationIssue("unknown_source", s, i))
                rep.invalid_claims.add(i)

    texts = [(None, answer)] + [(i, c.get("text", "")) for i, c in enumerate(claims)]
    for idx, text in texts:
        for grp in INLINE_ID.findall(text):
            for s in (x.strip() for x in grp.split(",")):
                if re.fullmatch(r"P\d{3}_[A-Z]\d+(?::\w+)?", s) and s not in ids:
                    rep.issues.append(CitationIssue("unknown_source", s, idx))
                    if idx is not None:
                        rep.invalid_claims.add(idx)
        for p in PAPER_ID.findall(text):
            if p not in papers:
                rep.issues.append(CitationIssue("unknown_paper", p, idx))
                if idx is not None:
                    rep.invalid_claims.add(idx)
        for kind, num in FIG_TAB.findall(text):
            lab = _norm(kind, num)
            if lab not in labels:
                rep.issues.append(CitationIssue("unknown_figure", lab, idx))
                if idx is not None:
                    rep.invalid_claims.add(idx)
        for a, b in PAGE_REF.findall(text):
            for n in (int(a), int(b)) if b else (int(a),):
                if n not in pages:
                    rep.issues.append(CitationIssue("unknown_page", str(n), idx))
                    if idx is not None:
                        rep.invalid_claims.add(idx)
    return rep
