"""Tab B: paper management (upload, processing status, metadata, reprocess/remove, assets)."""
from __future__ import annotations

import streamlit as st

from ingestion.indexer import IngestReport, ingest_paper, remove_paper_index
from ingestion.upload_validator import register_upload
from retrieval.keyword_search import bump_index_version
from ui.components import friendly_error, show_asset
from ui.services import Services

STATUS_BADGE = {"indexed": "🟢 indexed", "processing": "🟡 processing", "uploaded": "⚪ waiting",
                "failed": "🔴 failed"}


def _process(sv: Services, paper_id: str, label: str, force: bool = False) -> IngestReport:
    bar = st.progress(0.0, text=f"{label}: starting")

    def cb(msg: str, frac: float) -> None:
        bar.progress(min(frac, 1.0), text=f"{label}: {msg}")

    try:
        rep = ingest_paper(paper_id, sv.store, sv.qdrant, sv.embedder, force=force, progress=cb)
    except Exception as e:  # noqa: BLE001
        rep = IngestReport(paper_id, False, friendly_error(e))
    bar.empty()
    return rep


def _show_report(rep: IngestReport, label: str) -> None:
    if rep.ok and rep.skipped:
        st.info(f"{label}: already indexed.")
    elif rep.ok:
        c = rep.counts
        st.success(f"{label}: indexed {c.get('chunks', 0)} chunks, {c.get('tables', 0)} tables, "
                   f"{c.get('visuals', 0)} figures in {rep.timings.get('total', 0):.0f}s "
                   f"(VLM calls: {c.get('vlm_calls', 0)}, cached: {c.get('vlm_cache_hits', 0)}).")
        for w in rep.warnings:
            st.warning(w)
    else:
        st.error(f"{label}: {rep.message}")


def _upload_section(sv: Services) -> None:
    st.subheader("Upload papers")
    gen = st.session_state.setdefault("uploader_gen", 0)
    files = st.file_uploader("Research paper PDFs (open-access only)", type=["pdf"], accept_multiple_files=True,
                             key=f"uploader_{gen}")
    if st.button("Upload and process", type="primary", disabled=not files):
        new_ids: list[tuple[str, str]] = []
        for f in files:
            res = register_upload(f.getvalue(), f.name, sv.store)
            if res.ok:
                new_ids.append((res.paper_id, f.name))
                st.success(f"{f.name}: registered as {res.paper_id}.")
            elif res.duplicate_of:
                st.warning(f"{f.name}: duplicate. {res.message}")
            else:
                st.error(f"{f.name}: {res.message}")
        for pid, name in new_ids:
            _show_report(_process(sv, pid, name), name)
        st.session_state["uploader_gen"] = gen + 1
        if new_ids:
            st.button("Done: refresh list")  # triggers rerun and clears the uploader

    pending = [p for p in sv.store.list_papers() if p["status"] in ("uploaded", "failed")]
    if pending and st.button(f"Process {len(pending)} waiting/failed paper(s)"):
        for p in pending:
            _show_report(_process(sv, p["paper_id"], p["original_filename"], force=True), p["original_filename"])
        st.rerun()


def _metadata_form(sv: Services, p: dict) -> None:
    with st.form(f"meta_{p['paper_id']}"):
        title = st.text_input("Title", p["title"] or "")
        authors = st.text_input("Authors (comma separated)", ", ".join(p["authors"] or []))
        c1, c2, c3 = st.columns(3)
        year = c1.number_input("Year", 1950, 2100, int(p["year"] or 2024))
        source = c2.text_input("Source website", p["source"] or "")
        ident = c3.text_input("DOI / arXiv id", p["identifier"] or "")
        if st.form_submit_button("Save metadata"):
            sv.store.update_paper(p["paper_id"], title=title.strip() or None,
                                  authors=[a.strip() for a in authors.split(",") if a.strip()],
                                  year=int(year), source=source.strip() or None, identifier=ident.strip() or None)
            bump_index_version()
            st.success("Saved. Re-process the paper to refresh the title used in embeddings.")


def _assets(sv: Services, pid: str) -> None:
    figs = [e for e in sv.store.list_elements(pid) if e["element_type"] in ("figure", "chart")]
    tables = sv.store.list_elements(pid, "table")
    t1, t2 = st.tabs([f"Figures ({len(figs)})", f"Tables ({len(tables)})"])
    with t1:
        if not figs:
            st.caption("No figures extracted.")
        for e in figs:
            c1, c2 = st.columns([1, 1])
            with c1:
                show_asset(e["asset_path"], f"{e['label'] or e['element_id']} · page {e['page_number']}")
            with c2:
                st.markdown(f"**{e['label'] or e['element_id']}** ({e['element_type']}) · page {e['page_number']}")
                if e["caption"]:
                    st.caption(e["caption"])
                meta = e["metadata"] or {}
                if meta.get("vlm_status") == "unavailable":
                    st.warning("The vision model was unavailable; only the caption is searchable.")
                elif meta.get("uncertain"):
                    st.warning("The vision model marked this reading as uncertain.")
                st.markdown("**Vision model description**")
                st.text(e["content"] or "(none)")
    with t2:
        if not tables:
            st.caption("No tables extracted.")
        for e in tables:
            meta = e["metadata"] or {}
            st.markdown(f"**{e['label'] or e['element_id']}** · page {e['page_number']}")
            if e["caption"]:
                st.caption(e["caption"])
            headers, rows = meta.get("headers") or [], meta.get("rows") or []
            if rows:
                width = max(len(headers), max(len(r) for r in rows))
                cols = [h if h else f"col{i}" for i, h in enumerate(headers + [""] * (width - len(headers)))]
                seen: dict[str, int] = {}
                uniq = []
                for c in cols:
                    seen[c] = seen.get(c, 0) + 1
                    uniq.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
                st.dataframe([dict(zip(uniq, r + [""] * (width - len(r)))) for r in rows], width="stretch")
            show_asset(e["asset_path"], "Original table image")


def _paper_list(sv: Services) -> None:
    st.subheader("Papers")
    papers = sv.store.list_papers()
    if not papers:
        st.info("Nothing uploaded yet.")
        return
    for p in papers:
        pid = p["paper_id"]
        head = f"{pid} · {STATUS_BADGE.get(p['status'], p['status'])} · {(p['title'] or p['original_filename'])[:80]}"
        with st.expander(head):
            st.caption(f"{p['original_filename']} · {p['page_count']} pages · sha256 {p['sha256'][:12]}… · "
                       f"processing time {p['ingest_seconds'] or 0:.0f}s")
            if p["status"] == "failed":
                st.error(p["error"] or "Processing failed.")
            c1, c2, c3 = st.columns([1, 1, 2])
            if c1.button("Reprocess", key=f"re_{pid}"):
                _show_report(_process(sv, pid, pid, force=True), pid)
            confirm = c3.checkbox("Confirm removal", key=f"cf_{pid}")
            if c2.button("Remove", key=f"rm_{pid}", disabled=not confirm):
                remove_paper_index(pid, sv.store, sv.qdrant, delete_files=True)
                sv.store.delete_paper(pid)
                bump_index_version()
                st.rerun()
            _metadata_form(sv, p)
            if p["status"] == "indexed":
                _assets(sv, pid)


def render_papers(sv: Services) -> None:
    _upload_section(sv)
    st.divider()
    _paper_list(sv)
