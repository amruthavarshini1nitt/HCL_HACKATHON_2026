"""Step A7: batched OpenAI embeddings with a persistent content-hash cache."""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from config import settings

logger = logging.getLogger(__name__)
MAX_CHARS = 24000  # ~6k tokens, safely under the 8191-token model limit


class EmbeddingError(Exception):
    """Embedding service failure with a user-safe message."""


class Embedder:
    def __init__(self, cache_path: Path | None = None, client: OpenAI | None = None) -> None:
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self._client = client
        path = cache_path or (settings.cache_dir / "embeddings.sqlite")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = path
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        self.api_calls = 0
        self.cache_hits = 0

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db, timeout=30)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}|{self.dim}|{text}".encode()).hexdigest()

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not settings.openai_api_key:
                raise EmbeddingError("The embedding service is not configured (missing OPENAI_API_KEY).")
            self._client = OpenAI(api_key=settings.openai_api_key, timeout=60, max_retries=0)
        return self._client

    def _request(self, batch: list[str]) -> list[list[float]]:
        kwargs: dict = {"model": self.model, "input": batch}
        if self.dim != 1536:
            kwargs["dimensions"] = self.dim
        last: Exception | None = None
        for attempt in range(4):
            try:
                resp = self._get_client().embeddings.create(**kwargs)
                self.api_calls += 1
                vecs = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
                if len(vecs) != len(batch) or any(len(v) != self.dim for v in vecs):
                    raise EmbeddingError("The embedding service returned an unexpected response.")
                return vecs
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last = e
            except APIStatusError as e:
                last = e
                if e.status_code < 500:
                    break
            time.sleep(min(2 ** attempt, 10))
        logger.error("embedding request failed: %s", type(last).__name__)
        raise EmbeddingError("The embedding service is currently unavailable.")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches; identical texts (any paper) are served from cache."""
        texts = [(t or " ")[:MAX_CHARS] for t in texts]
        keys = [self._key(t) for t in texts]
        out: list[list[float] | None] = [None] * len(texts)
        with self._conn() as c:
            for i, k in enumerate(keys):
                row = c.execute("SELECT v FROM emb WHERE k=?", (k,)).fetchone()
                if row:
                    out[i] = np.frombuffer(row[0], dtype=np.float32).tolist()
                    self.cache_hits += 1
        missing = [i for i, v in enumerate(out) if v is None]
        # de-duplicate identical missing texts inside this call
        uniq: dict[str, list[int]] = {}
        for i in missing:
            uniq.setdefault(keys[i], []).append(i)
        todo = list(uniq.items())
        bs = settings.embedding_batch_size
        for s in range(0, len(todo), bs):
            part = todo[s:s + bs]
            vecs = self._request([texts[idxs[0]] for _, idxs in part])
            with self._conn() as c:
                for (k, idxs), v in zip(part, vecs):
                    c.execute("INSERT OR REPLACE INTO emb VALUES (?,?)",
                              (k, np.asarray(v, dtype=np.float32).tobytes()))
                    for i in idxs:
                        out[i] = v
        return out  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
