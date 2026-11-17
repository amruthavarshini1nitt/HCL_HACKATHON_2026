"""SQLite metadata store: paper catalogue, elements, chunks (parent/child) and references."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,               -- JSON list
    year INTEGER,
    source TEXT,
    identifier TEXT,            -- DOI / arXiv id
    original_filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    storage_path TEXT NOT NULL,
    page_count INTEGER,
    status TEXT NOT NULL DEFAULT 'uploaded',   -- uploaded|processing|indexed|failed
    error TEXT,
    ingest_seconds REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS elements (
    element_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    element_type TEXT NOT NULL,
    page_number INTEGER,
    section TEXT,
    content TEXT,
    caption TEXT,
    asset_path TEXT,
    label TEXT,                 -- e.g. "Figure 2", "Table 1"
    bbox TEXT,                  -- JSON [x0,y0,x1,y1]
    reading_order INTEGER,
    metadata TEXT               -- JSON
);
CREATE INDEX IF NOT EXISTS idx_elements_paper ON elements(paper_id, element_type, page_number);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    parent_id TEXT,             -- NULL for parent chunks
    chunk_type TEXT NOT NULL,   -- parent|child|table|figure
    element_type TEXT NOT NULL,
    page_number INTEGER,
    section TEXT,
    content TEXT NOT NULL,
    source_element_ids TEXT,    -- JSON list of element_ids
    token_count INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id, chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
CREATE TABLE IF NOT EXISTS element_links (
    src_element_id TEXT NOT NULL,
    dst_element_id TEXT NOT NULL,
    relation TEXT NOT NULL,     -- caption_of|explains|references
    PRIMARY KEY (src_element_id, dst_element_id, relation)
);
"""

_JSON_COLS = {"authors", "bbox", "metadata", "source_element_ids"}


class MetadataStore:
    """Thread-safe SQLite wrapper. One connection per call keeps Streamlit reruns safe."""

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        for k in _JSON_COLS & d.keys():
            if isinstance(d[k], str):
                try:
                    d[k] = json.loads(d[k])
                except json.JSONDecodeError:
                    pass
        return d

    # ---------------- papers ----------------
    def next_paper_id(self) -> str:
        with self._conn() as c:
            row = c.execute(
                "SELECT MAX(CAST(SUBSTR(paper_id, 2) AS INTEGER)) AS m FROM papers"
            ).fetchone()
        return f"P{(row['m'] or 0) + 1:03d}"

    def get_paper_by_hash(self, sha256: str) -> dict[str, Any] | None:
        with self._conn() as c:
            return self._row(c.execute("SELECT * FROM papers WHERE sha256=?", (sha256,)).fetchone())

    def register_paper(self, *, original_filename: str, sha256: str, storage_path: str,
                       page_count: int | None = None, title: str | None = None,
                       authors: list[str] | None = None, year: int | None = None,
                       source: str | None = None, identifier: str | None = None) -> str:
        """Insert a paper and return its id. Raises ValueError if the hash already exists."""
        with self._lock:
            if self.get_paper_by_hash(sha256):
                raise ValueError("duplicate paper")
            pid = self.next_paper_id()
            with self._conn() as c:
                c.execute(
                    "INSERT INTO papers (paper_id,title,authors,year,source,identifier,"
                    "original_filename,sha256,storage_path,page_count) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (pid, title, json.dumps(authors or []), year, source, identifier,
                     original_filename, sha256, storage_path, page_count),
                )
        return pid

    def update_paper(self, paper_id: str, **fields: Any) -> None:
        allowed = {"title", "authors", "year", "source", "identifier", "page_count", "status",
                   "error", "ingest_seconds", "storage_path"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"unknown paper fields: {sorted(bad)}")
        if "authors" in fields:
            fields["authors"] = json.dumps(fields["authors"] or [])
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE papers SET {sets} WHERE paper_id=?", (*fields.values(), paper_id))

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            return self._row(c.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone())

    def list_papers(self, status: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM papers", ()
        if status:
            q, args = q + " WHERE status=?", (status,)
        with self._conn() as c:
            return [self._row(r) for r in c.execute(q + " ORDER BY paper_id", args).fetchall()]  # type: ignore[misc]

    def delete_paper(self, paper_id: str) -> None:
        """Delete a paper and (via cascade) its elements and chunks."""
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT element_id FROM elements WHERE paper_id=?", (paper_id,))]
            c.executemany("DELETE FROM element_links WHERE src_element_id=? OR dst_element_id=?",
                          [(i, i) for i in ids])
            c.execute("DELETE FROM papers WHERE paper_id=?", (paper_id,))

    # ---------------- elements ----------------
    def clear_paper_content(self, paper_id: str) -> None:
        """Remove elements and chunks (used before reprocessing) but keep the catalogue row."""
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT element_id FROM elements WHERE paper_id=?", (paper_id,))]
            c.executemany("DELETE FROM element_links WHERE src_element_id=? OR dst_element_id=?",
                          [(i, i) for i in ids])
            c.execute("DELETE FROM elements WHERE paper_id=?", (paper_id,))
            c.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))

    def add_elements(self, elements: list[dict[str, Any]]) -> None:
        rows = [(e["element_id"], e["paper_id"], e["element_type"], e.get("page_number"),
                 e.get("section"), e.get("content"), e.get("caption"), e.get("asset_path"),
                 e.get("label"), json.dumps(e.get("bbox")), e.get("reading_order"),
                 json.dumps(e.get("metadata") or {})) for e in elements]
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO elements VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def get_element(self, element_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            return self._row(c.execute("SELECT * FROM elements WHERE element_id=?", (element_id,)).fetchone())

    def list_elements(self, paper_id: str, element_type: str | None = None,
                      page_number: int | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM elements WHERE paper_id=?", [paper_id]
        if element_type:
            q += " AND element_type=?"; args.append(element_type)
        if page_number is not None:
            q += " AND page_number=?"; args.append(page_number)
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY reading_order, element_id", args).fetchall()
        return [self._row(r) for r in rows]  # type: ignore[misc]

    def add_links(self, links: list[tuple[str, str, str]]) -> None:
        with self._conn() as c:
            c.executemany("INSERT OR IGNORE INTO element_links VALUES (?,?,?)", links)

    def get_links(self, element_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM element_links WHERE src_element_id=? OR dst_element_id=?",
                (element_id, element_id)).fetchall()
        return [dict(r) for r in rows]

    # ---------------- chunks ----------------
    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        rows = [(c["chunk_id"], c["paper_id"], c.get("parent_id"), c["chunk_type"], c["element_type"],
                 c.get("page_number"), c.get("section"), c["content"],
                 json.dumps(c.get("source_element_ids") or []), c.get("token_count"))
                for c in chunks]
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)", rows)

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            return self._row(c.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone())

    def get_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        with self._conn() as c:
            rows = c.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", chunk_ids).fetchall()
        return [self._row(r) for r in rows]  # type: ignore[misc]

    def list_searchable_chunks(self, paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """All retrievable chunks (everything except parents), for BM25 index building."""
        q, args = "SELECT * FROM chunks WHERE chunk_type != 'parent'", []
        if paper_ids:
            q += f" AND paper_id IN ({','.join('?' * len(paper_ids))})"; args = list(paper_ids)
        with self._conn() as c:
            return [self._row(r) for r in c.execute(q, args).fetchall()]  # type: ignore[misc]

    def children_of(self, parent_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM chunks WHERE parent_id=? ORDER BY chunk_id", (parent_id,)).fetchall()
        return [self._row(r) for r in rows]  # type: ignore[misc]

    # ---------------- stats ----------------
    def stats(self) -> dict[str, int]:
        with self._conn() as c:
            g = lambda q: c.execute(q).fetchone()[0]  # noqa: E731
            return {
                "papers": g("SELECT COUNT(*) FROM papers"),
                "indexed_papers": g("SELECT COUNT(*) FROM papers WHERE status='indexed'"),
                "searchable_chunks": g("SELECT COUNT(*) FROM chunks WHERE chunk_type!='parent'"),
                "tables": g("SELECT COUNT(*) FROM elements WHERE element_type='table'"),
                "visuals": g("SELECT COUNT(*) FROM elements WHERE element_type IN ('figure','chart')"),
            }
