import pytest
from qdrant_client import QdrantClient

from database.metadata_store import MetadataStore
from database.qdrant_store import QdrantStore


@pytest.fixture()
def store(tmp_path):
    return MetadataStore(tmp_path / "m.sqlite")


def test_register_and_duplicate(store):
    pid = store.register_paper(original_filename="a.pdf", sha256="h1", storage_path="x")
    assert pid == "P001"
    with pytest.raises(ValueError):
        store.register_paper(original_filename="b.pdf", sha256="h1", storage_path="y")
    assert store.register_paper(original_filename="c.pdf", sha256="h2", storage_path="z") == "P002"


def test_elements_chunks_and_cascade(store):
    pid = store.register_paper(original_filename="a.pdf", sha256="h", storage_path="x", authors=["A"])
    assert store.get_paper(pid)["authors"] == ["A"]
    store.add_elements([{"element_id": "P001_E001", "paper_id": pid, "element_type": "figure",
                         "page_number": 2, "content": "c", "bbox": [0, 0, 1, 1]}])
    store.add_chunks([
        {"chunk_id": "P001_C001", "paper_id": pid, "chunk_type": "parent", "element_type": "paragraph",
         "content": "parent"},
        {"chunk_id": "P001_C002", "paper_id": pid, "parent_id": "P001_C001", "chunk_type": "child",
         "element_type": "paragraph", "content": "child"}])
    assert store.get_element("P001_E001")["bbox"] == [0, 0, 1, 1]
    assert [c["chunk_id"] for c in store.list_searchable_chunks()] == ["P001_C002"]
    assert store.children_of("P001_C001")[0]["chunk_id"] == "P001_C002"
    assert store.stats()["visuals"] == 1
    store.delete_paper(pid)
    assert store.stats()["papers"] == 0 and store.get_chunk("P001_C002") is None


def test_qdrant_upsert_filter_delete():
    q = QdrantStore(client=QdrantClient(":memory:"), collection="t", dim=4)
    q.upsert([
        {"chunk_id": "a", "vector": [1, 0, 0, 0], "payload": {"paper_id": "P1", "element_type": "paragraph"}},
        {"chunk_id": "b", "vector": [0.9, 0.1, 0, 0], "payload": {"paper_id": "P2", "element_type": "table"}}])
    q.upsert([{"chunk_id": "a", "vector": [1, 0, 0, 0], "payload": {"paper_id": "P1", "element_type": "paragraph"}}])
    assert q.count() == 2  # idempotent
    assert q.search([1, 0, 0, 0], limit=2)[0]["chunk_id"] == "a"
    assert [r["chunk_id"] for r in q.search([1, 0, 0, 0], paper_ids=["P2"])] == ["b"]
    assert [r["chunk_id"] for r in q.search([1, 0, 0, 0], element_types=["table"])] == ["b"]
    with pytest.raises(ValueError):
        q.upsert([{"chunk_id": "z", "vector": [1], "payload": {}}])
    q.delete_paper("P1")
    assert q.count() == 1
