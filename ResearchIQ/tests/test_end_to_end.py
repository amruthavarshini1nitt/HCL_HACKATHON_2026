"""End-to-end: PDF -> ingestion -> index -> retrieval -> answer, and the Streamlit UI smoke tests.

External services (Docling models, OpenAI, VLM, LLM) are replaced by deterministic fakes; everything
else (validation, parsing fallback, chunking, SQLite, Qdrant, BM25, RRF, evidence checks) is real.
"""
from pathlib import Path

import pytest
import streamlit as st
from qdrant_client import QdrantClient
from streamlit.testing.v1 import AppTest

from config import settings
from database.metadata_store import MetadataStore
from database.qdrant_store import QdrantStore
from generation.answer_generator import AnswerOutput, Claim
from generation.assistant import AnswerResult, CheckedClaim, ResearchAssistant
from generation.comparison import ASPECTS, compare_papers
from generation.llm_client import ChatClient
from generation.visual_qa import VisualAnswer
from ingestion import indexer, pymupdf_utils as pu
from ingestion.pdf_parser import ParseResult
from ingestion.upload_validator import register_upload
from ingestion.visual_analyzer import VisualDescription
from retrieval.pipeline import Retriever
from retrieval.query_refiner import _LLMRefinement
from tests.test_generation import ScriptedLLM
from tests.test_ingestion import make_pdf
from tests.test_retrieval import BowEmbedder, OverlapReranker


class FakeVLM(ChatClient):
    def __init__(self):
        self.available = True
        self.vqa_calls = 0

    def complete_model(self, schema, messages, **kw):
        if schema is VisualDescription:
            return VisualDescription(visual_type="architecture_diagram",
                                     summary="Hybrid architecture with an Encoder feeding a Retriever",
                                     components=["Encoder", "Retriever"], connections=["Encoder -> Retriever"])
        self.vqa_calls += 1
        return VisualAnswer(answer="The diagram shows an Encoder feeding a Retriever.", observations=["two boxes"])


@pytest.fixture()
def indexed(tmp_path):
    orig = settings.data_dir
    object.__setattr__(settings, "data_dir", tmp_path / "data")
    settings.ensure_dirs()
    store = MetadataStore(settings.sqlite_path)
    qd = QdrantStore(client=QdrantClient(":memory:"), collection="e2e", dim=settings.embedding_dim)
    data = make_pdf(tmp_path / "p.pdf").read_bytes()
    res = register_upload(data, "paper.pdf", store)
    assert res.ok
    orig_parse = indexer.parse_pdf
    indexer.parse_pdf = lambda p: ParseResult(items=pu.parse_pages_with_pymupdf(p), page_count=2,
                                              page_sources={1: "pymupdf", 2: "pymupdf"})
    vlm = FakeVLM()
    rep = indexer.ingest_paper(res.paper_id, store, qd, BowEmbedder(), vlm=vlm)
    indexer.parse_pdf = orig_parse
    assert rep.ok, rep.message
    yield store, qd, vlm, res.paper_id
    object.__setattr__(settings, "data_dir", orig)


def assistant_for(indexed, llm):
    store, qd, vlm, _ = indexed
    r = Retriever(store, qd, BowEmbedder(), reranker=OverlapReranker(), llm=llm, use_prompt_guard=False)
    return ResearchAssistant(r, llm=llm, vlm=vlm, use_verifier=False)


def test_text_question_end_to_end(indexed):
    _, _, vlm, pid = indexed
    src = "Retrieval systems combine dense and sparse signals."
    # find the real chunk id holding that sentence
    store = indexed[0]
    chunk = next(c for c in store.list_searchable_chunks() if "dense and sparse" in c["content"])
    llm = ScriptedLLM([AnswerOutput(
        answer=f"The paper says retrieval systems combine dense and sparse signals [{chunk['chunk_id']}].",
        claims=[Claim(text="Retrieval systems combine dense and sparse signals.", source_ids=[chunk["chunk_id"]])])],
        _LLMRefinement(refined_query="dense sparse signals retrieval systems", intent="factual"))
    res = assistant_for(indexed, llm).ask("What do retrieval systems combine, dense and sparse signals?")
    assert res.status == "answered", (res.status, res.answer)
    c = res.citations[0]
    assert c["page_number"] == chunk["page_number"] and c["paper_id"] == pid
    assert vlm.vqa_calls == 0                      # no VLM call for a text question
    assert {"retrieve", "rerank", "generate", "validate"} <= set(res.timings)


def test_visual_question_uses_original_figure(indexed):
    store, _, vlm, pid = indexed
    fig = next(e for e in store.list_elements(pid) if e["element_type"] in ("figure", "chart"))
    vqa_id = f"{fig['element_id']}:VQA"
    llm = ScriptedLLM([AnswerOutput(
        answer=f"The diagram shows an Encoder feeding a Retriever [{vqa_id}].",
        claims=[Claim(text="The diagram shows an Encoder feeding a Retriever.", source_ids=[vqa_id],
                      kind="interpretation")])],
        _LLMRefinement(refined_query="hybrid architecture diagram encoder retriever figure", intent="visual",
                       content_types=["text", "figure"]))
    res = assistant_for(indexed, llm).ask("Show the hybrid architecture diagram with encoder and retriever")
    assert res.status in ("answered", "partial"), (res.status, res.answer)
    assert vlm.vqa_calls == 1
    cite = res.citations[0]
    assert cite["label"] == "Figure 1" and cite["derived_from_vlm"] and Path(cite["asset_path"]).is_file()


def test_missing_evidence_abstains(indexed):
    llm = ScriptedLLM([], _LLMRefinement(refined_query="quantum chromodynamics lattice", intent="factual"))
    res = assistant_for(indexed, llm).ask("What lattice spacing does the quantum chromodynamics study use?")
    assert res.status == "abstained" and llm.gen_calls == 0


def test_comparison_requires_two_papers(indexed):
    with pytest.raises(ValueError):
        compare_papers(assistant_for(indexed, ScriptedLLM([])), ["P001"], ["Methodology"])


# ---------------- Streamlit UI (AppTest) ----------------
@pytest.fixture()
def ui_env(tmp_path, monkeypatch):
    orig = settings.data_dir
    object.__setattr__(settings, "data_dir", tmp_path / "data")
    settings.ensure_dirs()
    import ui.services as svc
    st.cache_resource.clear()
    box = {}

    def build():
        store = MetadataStore(settings.sqlite_path)
        qd = QdrantStore(client=QdrantClient(":memory:"), collection="ui", dim=settings.embedding_dim)

        class Asst:
            def ask(self, query, session_key="x", paper_ids=None):
                box.setdefault("asked", []).append((query, paper_ids))
                cl = CheckedClaim("Retrieval systems combine dense and sparse signals.", ["P001_C0001"],
                                  "reported", "supported")
                return AnswerResult("answered", "They combine dense and sparse signals [P001_C0001].", [cl],
                                    ["Only one paper checked."],
                                    [{"source_id": "P001_C0001", "paper_id": "P001", "paper_title": "Title One",
                                      "page_number": 3, "label": None, "element_type": "paragraph",
                                      "section": "Intro", "asset_path": None, "derived_from_vlm": False}])
        return svc.Services(store, qd, BowEmbedder(), Asst())

    monkeypatch.setattr(svc, "build_services", build)
    yield box
    st.cache_resource.clear()
    object.__setattr__(settings, "data_dir", orig)


def seed(n=2):
    store = MetadataStore(settings.sqlite_path)
    for i in range(1, n + 1):
        pid = store.register_paper(original_filename=f"p{i}.pdf", sha256=f"h{i}", storage_path="x",
                                   title=f"Title {i}", page_count=5, year=2024)
        store.update_paper(pid, status="indexed")


APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_ui_empty_state_has_no_errors(ui_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [m.label for m in at.metric][:4] == ["Papers uploaded", "Indexed chunks", "Visual assets processed",
                                                 "Tables extracted"]
    assert any("Upload and process papers" in i.value for i in at.info)


def test_ui_chat_shows_answer_and_citations(ui_env):
    seed()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.chat_input[0].set_value("What do retrieval systems combine?").run()
    assert not at.exception
    assert ui_env["asked"][0][0] == "What do retrieval systems combine?"
    md = " ".join(m.value for m in at.markdown)
    assert "**[1]**" in md and "Title One" in md and "page 3" in md   # numbered citation and source line
    assert any("Limitations" in m.value for m in at.markdown)


def test_ui_comparison_runs_selected_aspects(ui_env):
    seed()
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.multiselect(key="cmp_papers").set_value([o for o in at.multiselect(key="cmp_papers").options]).run()
    at.multiselect(key="cmp_aspects").set_value(["Methodology", "Reported results"]).run()
    assert not at.exception
    run_btn = [b for b in at.button if b.label == "Run comparison"][0]
    run_btn.click().run()
    assert not at.exception
    assert [a[0] for a in ui_env["asked"]] == [ASPECTS["Methodology"], ASPECTS["Reported results"]]
    assert all(a[1] == ["P001", "P002"] for a in ui_env["asked"])


def test_ui_evaluation_tab_handles_missing_results(ui_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
