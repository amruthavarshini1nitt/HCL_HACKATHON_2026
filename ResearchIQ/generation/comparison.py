"""Multi-paper comparison: one evidence-grounded, validated answer per comparison aspect."""
from __future__ import annotations

import logging
from typing import Callable

from generation.assistant import AnswerResult, ResearchAssistant

logger = logging.getLogger(__name__)

CAVEAT = ("Reported numbers are only directly comparable when the datasets, splits, metrics and experimental "
          "settings match. Where they differ, the answer explains the difference instead of ranking the papers.")

ASPECTS: dict[str, str] = {
    "Methodology": "Compare the methodologies of the selected papers: the core approach, training or inference "
                   "procedure and key design choices.",
    "Architecture": "Compare the model or system architectures of the selected papers: their components and how "
                    "the components connect.",
    "Datasets": "Compare the datasets and benchmarks used in the selected papers, including data sources and "
                "evaluation splits.",
    "Evaluation metrics": "Compare the evaluation metrics and experimental settings (baselines, hardware, "
                          "hyperparameters) used by the selected papers.",
    "Reported results": "Compare the main results reported by the selected papers. For each result first state "
                        "the dataset, metric and experimental setting. Compare numbers only where these match; "
                        "otherwise explain why they are not directly comparable.",
    "Limitations": "Compare the limitations and weaknesses acknowledged by the authors of the selected papers "
                   "and point out any conflicting or missing evidence.",
}


def compare_papers(assistant: ResearchAssistant, paper_ids: list[str], aspects: list[str],
                   session_key: str = "compare",
                   on_progress: Callable[[str, float], None] | None = None) -> dict[str, AnswerResult]:
    """Run one grounded query per aspect, restricted to the selected papers."""
    if len(paper_ids) < 2:
        raise ValueError("Select at least two papers to compare.")
    out: dict[str, AnswerResult] = {}
    for i, name in enumerate(aspects):
        if on_progress:
            on_progress(name, i / max(len(aspects), 1))
        out[name] = assistant.ask(ASPECTS[name], f"{session_key}:cmp", paper_ids=paper_ids)
    if on_progress:
        on_progress("done", 1.0)
    return out
