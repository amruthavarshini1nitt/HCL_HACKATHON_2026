"""Shared, cached service objects for the Streamlit app (one set per server process)."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

import streamlit as st

from config import settings
from database.metadata_store import MetadataStore
from database.qdrant_store import QdrantStore
from generation.assistant import ResearchAssistant
from ingestion.embedding import Embedder
from retrieval.pipeline import Retriever

logger = logging.getLogger(__name__)


@dataclass
class Services:
    store: MetadataStore
    qdrant: QdrantStore
    embedder: Any
    assistant: Any


def build_services() -> Services:
    """Create stores and pipelines. Heavy models (reranker, Prompt Guard) load lazily on first use."""
    settings.ensure_dirs()
    store = MetadataStore(settings.sqlite_path)
    qdrant = QdrantStore()
    embedder = Embedder()
    retriever = Retriever(store, qdrant, embedder)
    return Services(store, qdrant, embedder, ResearchAssistant(retriever))


@st.cache_resource(show_spinner="Starting ResearchIQ...")
def get_services() -> Services:
    return build_services()


def session_key() -> str:
    """Stable per-browser-session id used for rate limiting."""
    if "session_key" not in st.session_state:
        st.session_state["session_key"] = uuid.uuid4().hex
    return st.session_state["session_key"]


def indexed_papers(store: MetadataStore) -> list[dict[str, Any]]:
    return store.list_papers(status="indexed")


def paper_label(p: dict[str, Any]) -> str:
    return f"{p['paper_id']} · {(p['title'] or p['original_filename'])[:70]}"
