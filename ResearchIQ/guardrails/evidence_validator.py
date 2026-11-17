"""Step B8: decide whether retrieved evidence is sufficient BEFORE generating an answer.

Not a chunk-count rule: checks relevance scores, query-term coverage, per-paper coverage for comparisons,
presence of requested figures/tables, and whether numbers exist when results are asked for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from config import settings
from retrieval.keyword_search import tokenize
from retrieval.types import EvidenceRecord, RefinedQuery

LABEL_REQ = re.compile(r"\b(fig(?:ure)?\.?|table)\s*([A-Za-z]?\d+)\b", re.I)
NUMBER_ASK = re.compile(r"\b(how (much|many)|accuracy|f1|score|bleu|rouge|recall|precision|auc|"
                        r"percent\w*|%|value|number|metric|latency|speed|improv\w+ by|outperform\w*)\b", re.I)
NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class SufficiencyReport:
    level: Literal["sufficient", "partial", "insufficient"]
    reasons: list[str] = field(default_factory=list)       # why not fully sufficient
    missing: list[str] = field(default_factory=list)       # what the user asked for but is absent
    checked: list[str] = field(default_factory=list)       # sources inspected, human readable
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def sufficient(self) -> bool:
        return self.level != "insufficient"


def _stem(tok: str) -> str:
    """Very light suffix stripping so fuse/fused/fusion-style variants match."""
    for suf in ("ations", "ation", "ings", "ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def _stems(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text)}


def _norm_label(kind: str, num: str) -> str:
    return f"{'Table' if kind.lower().startswith('tab') else 'Figure'} {num}"


def _relevance(rec: EvidenceRecord, query_terms: set[str]) -> float:
    """Reranker probability when available, else fraction of query terms found in the record."""
    rr = rec.retrieval.get("rerank_score")
    if rr is not None:
        return float(rr)
    toks = _stems(f"{rec.content} {rec.caption or ''}")
    return len(query_terms & toks) / len(query_terms) if query_terms else 0.0


def assess_evidence(rq: RefinedQuery, evidence: list[EvidenceRecord], selected_papers: list[str] | None = None
                    ) -> SufficiencyReport:
    checked = [f"{e.paper_title or e.paper_id}, p.{e.page_number}"
               + (f" ({e.label})" if e.label else "") for e in evidence]
    checked = list(dict.fromkeys(checked))
    if not evidence:
        return SufficiencyReport("insufficient", ["No evidence was retrieved."],
                                 ["Any passage relevant to the question"], checked)
    query_terms = _stems(rq.original_query) | {s for k in rq.keywords for s in _stems(k)}
    min_rel = settings.evidence_min_relevance
    scores = [_relevance(e, query_terms) for e in evidence]
    relevant = [e for e, s in zip(evidence, scores) if s >= min_rel]
    best = max(scores)
    all_text = " ".join(f"{e.content} {e.caption or ''} {e.parent_content or ''}" for e in evidence)
    covered = len(query_terms & _stems(all_text)) / len(query_terms) if query_terms else 1.0
    metrics = {"best_relevance": round(best, 4), "n_relevant": float(len(relevant)),
               "term_coverage": round(covered, 3)}
    reasons: list[str] = []
    missing: list[str] = []
    level: Literal["sufficient", "partial", "insufficient"] = "sufficient"

    if not relevant:
        return SufficiencyReport("insufficient", [f"No retrieved passage was relevant enough "
                                                  f"(best relevance {best:.2f} < {min_rel})."],
                                 ["Passages that address the question"], checked, metrics)
    if covered < 0.25:
        level = "partial"
        reasons.append(f"Only {covered:.0%} of the question's key terms appear in the evidence.")

    # explicit figure/table requests ("Figure 3", "Table 2")
    have = {e.label for e in evidence if e.label}
    for kind, num in LABEL_REQ.findall(rq.original_query):
        lab = _norm_label(kind, num)
        if lab not in have:
            missing.append(f"{lab} (requested but not found in retrieved sources)")
    if missing:
        level = "insufficient" if len(missing) and not have else "partial"
        reasons.append("A specifically requested figure/table was not retrieved.")

    if rq.intent == "visual":
        if not any(e.element_type in ("figure", "chart") for e in relevant):
            level = "insufficient"
            reasons.append("The question is about a figure/diagram but no relevant figure was retrieved.")
            missing.append("A relevant figure or chart")

    if rq.intent == "results" or NUMBER_ASK.search(rq.original_query):
        if not any(NUMBER.search(e.content) for e in relevant):
            level = "insufficient" if level == "insufficient" else "partial"
            reasons.append("The question asks for values but no relevant evidence contains numbers.")
            missing.append("Explicitly reported numeric values")

    if rq.intent == "comparison":
        scope = list(selected_papers or [])
        papers_with = {e.paper_id for e in relevant}
        if scope:
            lacking = [p for p in scope if p not in papers_with]
            if lacking:
                level = "insufficient" if len(papers_with) < 2 else "partial"
                reasons.append("Not every selected paper has relevant evidence.")
                missing.extend(f"Relevant evidence from {p}" for p in lacking)
        elif len(papers_with) < 2:
            level = "insufficient"
            reasons.append("A comparison needs evidence from at least two papers; only "
                           f"{len(papers_with)} had relevant passages.")
            missing.append("Relevant evidence from a second paper")
    return SufficiencyReport(level, reasons, missing, checked, metrics)


def abstention_message(report: SufficiencyReport) -> str:
    """User-facing explanation of what is missing and what was checked."""
    parts = ["I can't answer this reliably from the uploaded papers."]
    if report.reasons:
        parts.append("Why: " + " ".join(report.reasons))
    if report.missing:
        parts.append("Missing: " + "; ".join(dict.fromkeys(report.missing)) + ".")
    if report.checked:
        parts.append("Sources checked: " + "; ".join(report.checked[:8]) + ".")
    return "\n\n".join(parts)
