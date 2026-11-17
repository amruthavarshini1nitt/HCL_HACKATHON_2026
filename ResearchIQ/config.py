"""Central configuration for ResearchIQ, loaded from environment variables / .env."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings. Secrets are never logged (see `safe_summary`)."""

    # --- paths ---
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", BASE_DIR / "data")))

    # --- embeddings ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim: int = _int("EMBEDDING_DIM", 1536)
    embedding_batch_size: int = _int("EMBEDDING_BATCH_SIZE", 64)

    # --- LLM (OpenAI-compatible endpoint, provider configurable) ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_api_key: str = os.getenv("LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_timeout_s: int = _int("LLM_TIMEOUT_S", 60)

    # --- VLM (Qwen2.5-VL-7B-Instruct via hosted inference) ---
    vlm_api_key: str = os.getenv("VLM_API_KEY", "")
    vlm_base_url: str = os.getenv("VLM_BASE_URL", "https://openrouter.ai/api/v1")
    vlm_model: str = os.getenv("VLM_MODEL", "qwen/qwen2.5-vl-72b-instruct")
    vlm_timeout_s: int = _int("VLM_TIMEOUT_S", 120)

    # --- Hugging Face models ---
    hf_token: str = os.getenv("HF_TOKEN", "")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    prompt_guard_model: str = os.getenv("PROMPT_GUARD_MODEL", "meta-llama/Llama-Prompt-Guard-2-22M")
    prompt_guard_enabled: bool = os.getenv("PROMPT_GUARD_ENABLED", "true").strip().lower() in ("1", "true", "yes")

    # --- Qdrant: set QDRANT_URL for a server, otherwise embedded local mode ---
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "researchiq")

    # --- upload limits ---
    max_file_mb: int = _int("MAX_FILE_MB", 50)
    max_pages: int = _int("MAX_PAGES", 80)

    # --- query limits ---
    max_query_chars: int = _int("MAX_QUERY_CHARS", 1000)
    rate_limit_per_min: int = _int("RATE_LIMIT_PER_MIN", 20)

    # --- retrieval ---
    top_k_dense: int = _int("TOP_K_DENSE", 20)
    top_k_bm25: int = _int("TOP_K_BM25", 20)
    rerank_top_n: int = _int("RERANK_TOP_N", 6)
    rrf_k: int = _int("RRF_K", 60)
    injection_threshold: float = _float("INJECTION_THRESHOLD", 0.9)

    # --- evidence / answer validation ---
    evidence_min_relevance: float = _float("EVIDENCE_MIN_RELEVANCE", 0.1)   # reranker probability
    claim_overlap_min: float = _float("CLAIM_OVERLAP_MIN", 0.3)             # lexical support floor
    max_answer_retries: int = _int("MAX_ANSWER_RETRIES", 1)

    # --- chunking (tokens) ---
    child_chunk_tokens: int = _int("CHILD_CHUNK_TOKENS", 300)
    parent_chunk_tokens: int = _int("PARENT_CHUNK_TOKENS", 1200)
    chunk_overlap_tokens: int = _int("CHUNK_OVERLAP_TOKENS", 50)

    # --- ingestion ---
    ocr_min_chars_per_page: int = _int("OCR_MIN_CHARS_PER_PAGE", 200)
    min_figure_px: int = _int("MIN_FIGURE_PX", 120)

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # ---- derived paths ----
    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def normalized_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def figures_dir(self) -> Path:
        return self.data_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.data_dir / "tables"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "db" / "metadata.sqlite"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "db" / "qdrant"

    def ensure_dirs(self) -> None:
        for p in (self.papers_dir, self.normalized_dir, self.chunks_dir, self.figures_dir,
                  self.tables_dir, self.pages_dir, self.cache_dir, self.sqlite_path.parent):
            p.mkdir(parents=True, exist_ok=True)

    def safe_summary(self) -> dict[str, object]:
        """Settings with secrets reduced to booleans, safe to log or show in debug UI."""
        return {
            "embedding_model": self.embedding_model,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "vlm_model": self.vlm_model,
            "reranker_model": self.reranker_model,
            "qdrant_mode": "server" if self.qdrant_url else "embedded",
            "has_openai_key": bool(self.openai_api_key),
            "has_llm_key": bool(self.llm_api_key),
            "has_vlm_key": bool(self.vlm_api_key),
            "has_hf_token": bool(self.hf_token),
        }


settings = Settings()


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
