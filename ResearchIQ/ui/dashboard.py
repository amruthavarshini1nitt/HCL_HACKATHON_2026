"""Tab A: dashboard."""
from __future__ import annotations

import streamlit as st

from config import settings
from ui.services import Services


def render_dashboard(sv: Services) -> None:
    st.title("ResearchIQ")
    st.caption("Multimodal research assistant: every answer is grounded in the uploaded papers, with page and "
               "figure citations, and AI interpretation is labelled separately from reported findings.")
    stats = sv.store.stats()
    papers = sv.store.list_papers()
    by_status: dict[str, int] = {}
    for p in papers:
        by_status[p["status"]] = by_status.get(p["status"], 0) + 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers uploaded", stats["papers"])
    c2.metric("Indexed chunks", stats["searchable_chunks"])
    c3.metric("Visual assets processed", stats["visuals"])
    c4.metric("Tables extracted", stats["tables"])
    if papers:
        st.markdown("**Processing status:** " + " · ".join(f"{k}: {v}" for k, v in sorted(by_status.items())))
        if stats["papers"] < 8:
            st.info(f"{stats['papers']} paper(s) uploaded. The project targets 8–10 papers on one topic.")
        elif stats["papers"] > 10:
            st.info("More than 10 papers uploaded; the project targets 8–10 papers on one topic.")

    cfg = settings.safe_summary()
    missing = [n for n, k in (("OPENAI_API_KEY (embeddings)", "has_openai_key"), ("LLM_API_KEY", "has_llm_key"),
                              ("VLM_API_KEY (figure understanding)", "has_vlm_key")) if not cfg[k]]
    if missing:
        st.warning("Missing configuration: " + ", ".join(missing) + ". Set these in `.env` and restart. "
                   "Features that need them will report a clear error.")
    if not cfg["has_hf_token"]:
        st.caption("HF_TOKEN not set: Prompt Guard 2 (gated model) cannot be downloaded, so only the "
                   "rule-based injection checks run.")

    st.subheader("Paper catalogue")
    if not papers:
        st.info("No papers yet. Upload PDFs in the **Paper Management** tab.")
    else:
        q = st.text_input("Search catalogue", placeholder="title, author, identifier, filename...",
                          key="catalogue_search").strip().lower()
        rows = []
        for p in papers:
            hay = " ".join([p["title"] or "", " ".join(p["authors"] or []), p["identifier"] or "",
                            p["original_filename"], str(p["year"] or "")]).lower()
            if q and q not in hay:
                continue
            rows.append({"ID": p["paper_id"], "Title": p["title"], "Authors": ", ".join(p["authors"] or []),
                         "Year": p["year"], "Source": p["source"], "Identifier": p["identifier"],
                         "Pages": p["page_count"], "Status": p["status"]})
        st.dataframe(rows, width="stretch", hide_index=True)
    with st.expander("Configuration (secrets hidden)"):
        st.json(cfg)
