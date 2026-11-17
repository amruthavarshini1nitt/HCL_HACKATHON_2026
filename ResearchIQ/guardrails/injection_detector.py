"""Prompt-injection detection: Llama Prompt Guard 2 (22M) plus deterministic pattern checks.

Questions that merely *discuss* prompt injection (typical for research on LLM security) are not blocked:
a pattern match only counts when phrased as an instruction directed at the assistant.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# Imperative instructions aimed at the assistant.
_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|system)\b.{0,30}"
    r"\b(instructions?|prompts?|rules?|guidelines?|context)\b",
    r"\b(reveal|show|print|repeat|output|leak)\b.{0,30}\b(system prompt|hidden prompt|your instructions|"
    r"api[ _-]?keys?|environment variables?|secrets?|credentials?)\b",
    r"\byou are now\b.{0,40}\b(dan|unrestricted|jailbroken|developer mode|no longer)\b",
    r"\b(pretend|act)\b.{0,20}\b(you have no|without any)\b.{0,20}\b(restrictions|rules|filters)\b",
    r"\bdo anything now\b",
    r"\bnew instructions?\s*:",
    r"<\s*/?\s*(system|assistant)\s*>",
]
_RE = [re.compile(p, re.I | re.S) for p in _PATTERNS]
# Phrases indicating the user is asking ABOUT injection rather than performing it.
_DISCUSSION = re.compile(
    r"\b(paper|papers|study|studies|survey|attack|attacks|defen[cs]e|defen[cs]es|benchmark|detect|"
    r"mitigat\w*|robust\w*|how (do|does|can)|what (is|are)|explain|compare|author[s]?)\b", re.I)


@dataclass
class InjectionResult:
    flagged: bool
    score: float = 0.0
    method: str = "none"
    reason: str = ""


def pattern_check(text: str) -> InjectionResult:
    for rx in _RE:
        if rx.search(text):
            discussing = bool(_DISCUSSION.search(text)) and not re.match(
                r"\s*(please\s+)?(ignore|disregard|forget|reveal|print|show|you are now|new instructions?)",
                text, re.I)
            if discussing:
                return InjectionResult(False, 0.3, "pattern", "mentions injection-like phrase in a research context")
            return InjectionResult(True, 1.0, "pattern", "instruction-override pattern")
    return InjectionResult(False)


class PromptGuard:
    """Lazy wrapper around meta-llama/Llama-Prompt-Guard-2-22M (gated model, needs HF_TOKEN)."""

    def __init__(self) -> None:
        self._pipe: Any = None
        self._failed = False

    def _load(self) -> Any | None:
        if self._pipe is not None or self._failed:
            return self._pipe
        if not settings.prompt_guard_enabled:
            self._failed = True
            logger.info("Prompt Guard 2 disabled by PROMPT_GUARD_ENABLED=false; pattern checks only")
            return None
        try:
            from transformers import pipeline
            kwargs: dict[str, Any] = {"model": settings.prompt_guard_model, "device": -1}
            if settings.hf_token:
                kwargs["token"] = settings.hf_token
            self._pipe = pipeline("text-classification", **kwargs)
        except Exception:  # noqa: BLE001 - gated model, offline, etc.
            self._failed = True
            logger.warning("Prompt Guard 2 unavailable; using pattern checks only", exc_info=True)
        return self._pipe

    @property
    def available(self) -> bool:
        return self._load() is not None

    def score(self, text: str) -> float | None:
        """Probability that `text` is malicious (None if the model is unavailable)."""
        pipe = self._load()
        if pipe is None:
            return None
        try:
            out = pipe(text[:2000], truncation=True, max_length=512, top_k=None)
            if out and isinstance(out[0], list):
                out = out[0]
            for o in out:
                if str(o["label"]).upper() in ("LABEL_1", "MALICIOUS", "INJECTION", "JAILBREAK"):
                    return float(o["score"])
            return 0.0
        except Exception:  # noqa: BLE001
            logger.warning("Prompt Guard inference failed", exc_info=True)
            return None


_guard = PromptGuard()


def detect_injection(text: str, guard: PromptGuard | None = None, use_model: bool = True) -> InjectionResult:
    """Pattern check first (cheap, decisive); then the model for queries not clearly benign."""
    pat = pattern_check(text)
    if pat.flagged:
        return pat
    if use_model:
        g = guard or _guard
        s = g.score(text)
        if s is not None:
            # A research-context question that scores high is still allowed unless very confident.
            threshold = settings.injection_threshold
            if _DISCUSSION.search(text):
                threshold = max(threshold, 0.98)
            if s >= threshold:
                return InjectionResult(True, s, "prompt_guard_2", "classified as prompt injection")
            return InjectionResult(False, s, "prompt_guard_2")
    return pat


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
REMOVED_MARKER = "[instruction-like text removed]"


def strip_instruction_sentences(text: str) -> tuple[str, bool]:
    """Replace sentences that read as instructions to the assistant with a marker.

    Removal (not just flagging) matters: otherwise a claim that merely repeats the injected text would
    pass the evidence-support check, because the poisoned passage "supports" itself.
    Sentences that discuss injection in a research context are kept (see `pattern_check`).
    """
    parts = _SENT_SPLIT.split(text)
    kept: list[str] = []
    hit = False
    for s in parts:
        if pattern_check(s).flagged:
            hit = True
            if not kept or kept[-1] != REMOVED_MARKER:
                kept.append(REMOVED_MARKER)
        else:
            kept.append(s)
    return (" ".join(kept), hit)


def scan_document_text(text: str) -> InjectionResult:
    """Pattern scan for instructions embedded in retrieved PDF text (used to flag, never to obey)."""
    return pattern_check(text)
