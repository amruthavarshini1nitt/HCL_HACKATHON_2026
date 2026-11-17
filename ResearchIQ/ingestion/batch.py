"""Batch ingestion from the command line (same pipeline as the Streamlit upload).

Usage:
    python -m ingestion.batch <pdf> [<pdf> ...]
    python -m ingestion.batch --folder <dir>
Writes a JSON report with per-paper timings and counts to data/cache/last_batch_report.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from config import settings, setup_logging
from database.metadata_store import MetadataStore
from database.qdrant_store import QdrantStore
from ingestion.embedding import Embedder
from ingestion.indexer import ingest_paper
from ingestion.upload_validator import register_upload

logger = logging.getLogger("ingestion.batch")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--folder", type=Path)
    ap.add_argument("--force", action="store_true", help="reprocess papers that are already indexed")
    args = ap.parse_args(argv)
    files = list(args.files) + (sorted(args.folder.glob("*.pdf")) if args.folder else [])
    if not files:
        ap.error("no PDFs given")

    setup_logging()
    settings.ensure_dirs()
    store, qdrant, embedder = MetadataStore(settings.sqlite_path), QdrantStore(), Embedder()
    report = []
    t_all = time.perf_counter()
    for f in files:
        reg = register_upload(f.read_bytes(), f.name, store)
        pid = reg.paper_id
        entry = {"file": f.name, "paper_id": pid, "registered": reg.ok, "message": reg.message}
        if pid is None:
            print(f"[SKIP] {f.name}: {reg.message}")
            report.append(entry)
            continue
        print(f"[....] {f.name} -> {pid} ({'new' if reg.ok else 'duplicate'})", flush=True)
        rep = ingest_paper(pid, store, qdrant, embedder, force=args.force)
        entry.update(ok=rep.ok, skipped=rep.skipped, result=rep.message, timings=rep.timings,
                     counts=rep.counts, warnings=rep.warnings, title=(store.get_paper(pid) or {}).get("title"))
        status = "SKIP" if rep.skipped else ("OK  " if rep.ok else "FAIL")
        print(f"[{status}] {pid} {entry['title']!r}: {rep.message} {rep.counts} "
              f"total={rep.timings.get('total', 0):.0f}s", flush=True)
        for w in rep.warnings:
            print(f"       warning: {w}")
        report.append(entry)
    out = settings.cache_dir / "last_batch_report.json"
    out.write_text(json.dumps({"total_s": time.perf_counter() - t_all, "papers": report}, indent=2,
                              default=str), encoding="utf-8")
    print(f"\nReport: {out}")
    qdrant.close()
    return 0 if all(r.get("ok", False) or r.get("skipped") for r in report if r["paper_id"]) else 1


if __name__ == "__main__":
    sys.exit(main())
