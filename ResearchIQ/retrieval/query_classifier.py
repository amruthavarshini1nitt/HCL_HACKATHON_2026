"""Rule-based intent classification, used as a fallback when the LLM refiner is unavailable."""
from __future__ import annotations

import re

from retrieval.types import ContentType, Intent

_RULES: list[tuple[Intent, re.Pattern[str]]] = [
    ("visual", re.compile(r"\b(figure|fig\.|diagram|chart|plot|graph|architecture (diagram|figure)|"
                          r"illustrat\w+|visuali[sz]\w+|axis|axes|curve)\b", re.I)),
    ("comparison", re.compile(r"\b(compare|comparison|versus|vs\.?|differ\w*|contrast|better than|"
                              r"similarit\w+|between)\b", re.I)),
    ("limitations", re.compile(r"\b(limitation|weakness|drawback|shortcoming|fail\w*|future work|"
                               r"missing|gap|conflict\w*|contradict\w*)\b", re.I)),
    ("synthesis", re.compile(r"\b(synthes\w+|overview|survey|summar\w+|trends?|research directions?|"
                             r"open problems?|literature)\b", re.I)),
    ("results", re.compile(r"\b(result|accuracy|f1|score|perform\w+|benchmark|metric|outperform\w*|"
                           r"improv\w+|table|ablation|bleu|rouge)\b", re.I)),
    ("methodology", re.compile(r"\b(method\w*|approach|architecture|how does|how do|algorithm|"
                               r"pipeline|training|model design|loss|technique)\b", re.I)),
]


def classify_intent(query: str) -> Intent:
    for intent, rx in _RULES:
        if rx.search(query):
            return intent
    return "factual"


def default_content_types(query: str, intent: Intent) -> list[ContentType]:
    types: list[ContentType] = ["text"]
    if intent == "visual" or re.search(r"\b(figure|diagram|chart|plot|graph)\b", query, re.I):
        types.append("figure")
    if intent == "results" or re.search(r"\b(table|accuracy|benchmark|metric|dataset|ablation|score)\b",
                                        query, re.I):
        types.append("table")
    return types
