"""Deterministic input guardrails: query validation and rate limiting."""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass

from config import settings

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


@dataclass
class ValidationResult:
    ok: bool
    cleaned: str = ""
    reason: str = ""


def validate_query(query: object, max_chars: int | None = None) -> ValidationResult:
    """Reject empty, non-string, oversize or garbage input; return a normalized query otherwise."""
    limit = max_chars or settings.max_query_chars
    if not isinstance(query, str):
        return ValidationResult(False, reason="The question must be text.")
    text = unicodedata.normalize("NFKC", query)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ValidationResult(False, reason="Please enter a question.")
    if len(text) > limit:
        return ValidationResult(False, reason=f"The question is too long (limit {limit} characters).")
    if not re.search(r"[A-Za-z0-9\u00c0-\uffff]", text):
        return ValidationResult(False, reason="The question contains no readable text.")
    letters = sum(c.isalnum() for c in text)
    if len(text) > 30 and letters / len(text) < 0.4:
        return ValidationResult(False, reason="The question looks malformed.")
    return ValidationResult(True, cleaned=text)


class RateLimiter:
    """Sliding-window limiter per key (e.g. Streamlit session id)."""

    def __init__(self, max_calls: int | None = None, window_s: float = 60.0) -> None:
        self.max_calls = max_calls if max_calls is not None else settings.rate_limit_per_min
        self.window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.max_calls:
                return False
            q.append(now)
            return True
