import json
from pathlib import Path

import fitz
import pytest
from qdrant_client import QdrantClient

from config import settings
from database.metadata_store import MetadataStore
from database.qdrant_store import QdrantStore
from generation.llm_client import ChatClient
from ingestion import indexer, pymupdf_utils as pu
from ingestion.chunker import chunk_document, count_tokens
from ingestion.output_normalizer import normalize, normalize_label, table_to_text
from ingestion.pdf_parser import ParseResult
from ingestion.text_processor import extract_text_elements
from ingestion.upload_validator import UploadError, register_upload, safe_filename, validate_pdf_bytes
from ingestion.visual_analyzer import VisualDescription, describe_image, description_to_text

BODY = ("Retrieval systems combine dense and sparse signals. We evaluate on a benchmark and report accuracy. "
        "The results show consistent gains over baselines across all settings. ") * 4


def make_pdf(path: Path, *, encrypt: bool = False) -> Path:
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((72, 80), "Sparse Dense Fusion for Research QA", fontsize=20)
    p.insert_text((72, 110), "Alice Smith, Bob Jones", fontsize=11)
    p.insert_text((72, 130), "arXiv:2401.01234", fontsize=9)
    p.insert_text((72, 160), "Abstract", fontsize=13, fontname="hebo")
    p.insert_textbox(fitz.Rect(72, 170, 520, 260), "We propose a hybrid retrieval method. " * 6, fontsize=10)
    p.insert_text((72, 290), "1 Introduction", fontsize=14, fontname="hebo")
    p.insert_textbox(fitz.Rect(72, 300, 520, 420), BODY, fontsize=10)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 160), False)
    pix.set_rect(pix.irect, (200, 60, 60))
    p.insert_image(fitz.Rect(72, 430, 372, 590), pixmap=pix)
    p.insert_text((72, 610), "Figure 1: Overview of the hybrid architecture.", fontsize=10)
    p2 = doc.new_page()
    p2.insert_text((72, 80), "2 Method", fontsize=14, fontname="hebo")
    p2.insert_textbox(fitz.Rect(72, 90, 520, 220), BODY + " As shown in Figure 1, the pipeline fuses ranks.",
                      fontsize=10)
    x0, y0, cw, rh = 72, 250, 100, 22
    for r in range(3):
        for c in range(3):
            rect = fitz.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            p2.draw_rect(rect)
            p2.insert_text((rect.x0 + 4, rect.y1 - 6), [["Model", "Acc", "F1"], ["A", "81.2", "79.0"],
                                                         ["B", "85.5", "83.1"]][r][c], fontsize=9)
    p2.insert_text((72, 335), "Table 1: Accuracy of models on the benchmark.", fontsize=10)
    if encrypt:
        doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="y")
    else:
        doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def env(tmp_path, monkeypatch):
    orig = settings.data_dir
    object.__setattr__(settings, "data_dir", tmp_path / "data")
    settings.ensure_dirs()
    yield tmp_path
    object.__setattr__(settings, "data_dir", orig)


# ---------------- upload validation ----------------
def test_validate_rejects(env):
    good = make_pdf(env / "g.pdf").read_bytes()
    assert validate_pdf_bytes(good, "g.pdf").page_count == 2
    for data, name in [(b"", "a.pdf"), (b"hello", "a.pdf"), (good, "a.txt"), (b"%PDF-1.4 garbage", "a.pdf")]:
        with pytest.raises(UploadError):
            validate_pdf_bytes(data, name)
    enc = make_pdf(env / "e.pdf", encrypt=True).read_bytes()
    with pytest.raises(UploadError, match="Encrypted"):
        validate_pdf_bytes(enc, "e.pdf")


def test_size_and_page_limits(env, monkeypatch):
    good = make_pdf(env / "g.pdf").read_bytes()
    monkeypatch.setattr(type(settings), "max_pages", 1, raising=False) if False else None
    object.__setattr__(settings, "max_pages", 1)
    try:
        with pytest.raises(UploadError, match="pages"):
            validate_pdf_bytes(good, "g.pdf")
    finally:
        object.__setattr__(settings, "max_pages", 80)
    object.__setattr__(settings, "max_file_mb", 0)
    try:
        with pytest.raises(UploadError, match="limit"):
            validate_pdf_bytes(good, "g.pdf")
    finally:
        object.__setattr__(settings, "max_file_mb", 50)


def test_register_duplicate_and_filename(env):
    store = MetadataStore(settings.sqlite_path)
    data = make_pdf(env / "g.pdf").read_bytes()
    r1 = register_upload(data, "../../evil name.pdf", store)
    assert r1.ok and r1.paper_id == "P001" and ".." not in r1.filename
    r2 = register_upload(data, "copy.pdf", store)
    assert not r2.ok and r2.duplicate_of == "P001"
    assert not register_upload(b"nope", "x.pdf", store).ok
    assert safe_filename("a/b\\c d.pdf") == "c_d.pdf"


# ---------------- parsing / normalization ----------------
def test_pymupdf_fallback_and_normalize(env):
    pdf = make_pdf(env / "g.pdf")
    items = pu.parse_pages_with_pymupdf(pdf)
    kinds = {i["kind"] for i in items}
    assert {"heading", "text", "caption", "picture"} <= kinds
    doc = normalize(items, "P001", 2, first_page_text=pu.page_texts(pdf)[0])
    types = {e.element_type for e in doc.elements}
    assert "title" in types and "abstract" in types and "figure" in types
    assert doc.title and "Sparse Dense" in doc.title
    assert doc.identifier == "arXiv:2401.01234" and doc.year == 2024 and doc.source == "arxiv.org"
    fig = next(e for e in doc.elements if e.element_type == "figure")
    assert fig.label == "Figure 1" and "hybrid architecture" in (fig.caption or "")
    assert not any(e.metadata.get("is_caption") and e.label == "Figure 1" for e in doc.elements)  # no duplicate
    ids = [e.element_id for e in doc.elements]
    assert len(ids) == len(set(ids))
    sections = {e.section for e in doc.elements if e.element_type == "paragraph"}
    assert "Introduction" in sections


def test_label_and_table_text():
    assert normalize_label("Fig. 3: foo") == "Figure 3"
    assert normalize_label("TABLE 2 results") == "Table 2"
    assert normalize_label("random") is None
    txt = table_to_text(["Model", "Acc"], [["A", "81"]], "Table 1: x")
    assert "Model: A" in txt and "Acc: 81" in txt


# ---------------- chunking ----------------
def test_chunking_parent_child_and_equation(env):
    pdf = make_pdf(env / "g.pdf")
    doc = normalize(pu.parse_pages_with_pymupdf(pdf), "P001", 2)
    texts = extract_text_elements(doc)
    chunks = chunk_document("P001", texts, [e for e in doc.elements if e.element_type in ("table", "figure")])
    parents = {c["chunk_id"] for c in chunks if c["chunk_type"] == "parent"}
    children = [c for c in chunks if c["chunk_type"] == "child"]
    assert parents and children
    assert all(c["parent_id"] in parents for c in children)
    assert all(c["token_count"] <= settings.child_chunk_tokens + 150 for c in children)
    assert all(c["page_number"] and c["paper_id"] == "P001" for c in children)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_long_paragraph_split_with_overlap(env):
    from ingestion.output_normalizer import Element
    sent = "This sentence number {} talks about retrieval augmented generation systems. "
    long = " ".join(sent.format(i) for i in range(150))
    el = Element(paper_id="P9", element_id="P9_E001", element_type="paragraph", page_number=1,
                 section="S", content=long)
    ch = [c for c in chunk_document("P9", [el], []) if c["chunk_type"] == "child"]
    assert len(ch) > 2
    assert all(count_tokens(c["content"]) <= settings.child_chunk_tokens + 60 for c in ch)
    first_tail = ch[0]["content"].split(". ")[-1][:20]
    assert first_tail.split(" talks")[0][:15] in ch[1]["content"]  # overlap present


# ---------------- visual analysis cache ----------------
class FakeVLM(ChatClient):
    def __init__(self) -> None:
        self.calls = 0
        self.available = True

    def complete_model(self, schema, messages, **kw):
        self.calls += 1
        return VisualDescription(visual_type="architecture_diagram", summary="Boxes joined by arrows",
                                 components=["Encoder", "Retriever"], connections=["Encoder -> Retriever"])


def test_vlm_cache(env):
    img = env / "x.png"
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50), False)
    pix.set_rect(pix.irect, (1, 2, 3))
    pix.save(str(img))
    vlm = FakeVLM()
    d1, c1 = describe_image(img, "cap", "ctx", vlm)
    d2, c2 = describe_image(img, "cap", "ctx", vlm)
    assert vlm.calls == 1 and not c1 and c2 and d1 == d2
    assert "Encoder -> Retriever" in description_to_text(d1, "Figure 1", "cap")


# ---------------- full ingestion (Docling stubbed, fake embedder/VLM) ----------------
class FakeEmbedder:
    api_calls = 0

    def embed_texts(self, texts):
        out = []
        for t in texts:
            v = [0.0] * settings.embedding_dim
            v[hash(t) % settings.embedding_dim] = 1.0
            out.append(v)
        return out


def test_ingest_end_to_end_idempotent(env, monkeypatch):
    store = MetadataStore(settings.sqlite_path)
    qd = QdrantStore(client=QdrantClient(":memory:"), collection="t", dim=settings.embedding_dim)
    data = make_pdf(env / "g.pdf").read_bytes()
    pid = register_upload(data, "g.pdf", store).paper_id
    monkeypatch.setattr(indexer, "parse_pdf", lambda p: ParseResult(
        items=pu.parse_pages_with_pymupdf(p), page_count=2, page_sources={1: "pymupdf", 2: "pymupdf"}))
    vlm = FakeVLM()
    rep = indexer.ingest_paper(pid, store, qd, FakeEmbedder(), vlm=vlm)
    assert rep.ok, rep.message
    paper = store.get_paper(pid)
    assert paper["status"] == "indexed" and "Sparse Dense" in paper["title"]
    assert rep.counts["visuals"] == 1 and vlm.calls == 1
    n_points = qd.count()
    assert n_points == store.stats()["searchable_chunks"] > 0
    fig = store.list_elements(pid, "figure") + store.list_elements(pid, "chart")
    assert fig and Path(fig[0]["asset_path"]).exists() and "Encoder" in fig[0]["content"]
    assert (settings.normalized_dir / f"{pid}.json").exists()
    # skip when already indexed, and reprocessing does not duplicate points or call the VLM again (cache)
    assert indexer.ingest_paper(pid, store, qd, FakeEmbedder(), vlm=vlm).skipped
    rep2 = indexer.ingest_paper(pid, store, qd, FakeEmbedder(), force=True, vlm=vlm)
    assert rep2.ok and vlm.calls == 1 and qd.count() == n_points
    indexer.remove_paper_index(pid, store, qd, delete_files=True)
    assert qd.count() == 0 and not store.list_elements(pid)


def test_vlm_null_lists_are_accepted():
    """Qwen2.5-VL-72B returns null (not []) for empty list fields; this must validate."""
    d = VisualDescription.model_validate({"visual_type": "chart", "summary": "s", "legend": None,
                                          "components": None, "values": ["85", "70"], "uncertain": False})
    assert d.legend == [] and d.components == [] and d.values == ["85", "70"]


def test_author_and_identifier_extraction():
    from ingestion.output_normalizer import _extract_authors, identifier_from_filename
    lines = ["Xinke Jiang1,2*, Yue Fang1,2*, Rihong Qiu1,2*, Haoyu Zhang7",
             "1School of Computer Science, Peking University, Beijing, China",
             "6Peking University Information Technology Institute, Tianjin Binhai, China",
             "{xinkejiang, fangyue}@stu.pku.edu.cn"]
    assert _extract_authors(lines) == ["Xinke Jiang", "Yue Fang", "Rihong Qiu", "Haoyu Zhang"]
    assert _extract_authors(["Department of Operations, and Heilbronn Data Science Center"]) == []
    assert _extract_authors([], "Siran Li ; Linus Stenzel") == ["Siran Li", "Linus Stenzel"]
    assert identifier_from_filename("5e49fb2960a7bd44_2025.acl-long.558.pdf") == \
        ("ACL Anthology 2025.acl-long.558", "aclanthology.org")
    assert identifier_from_filename("2312.10997v5.pdf") == ("arXiv:2312.10997", "arxiv.org")
    assert identifier_from_filename("NeurIPS-2025-dynamicrag.pdf") == (None, None)
