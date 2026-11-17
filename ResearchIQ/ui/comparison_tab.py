"""Tab D: structured comparison of two or more papers."""
from __future__ import annotations

import streamlit as st

from generation.comparison import ASPECTS, CAVEAT, compare_papers
from ui.components import friendly_error, render_answer
from ui.services import Services, indexed_papers, paper_label, session_key


def render_comparison(sv: Services) -> None:
    st.subheader("Paper comparison")
    papers = indexed_papers(sv.store)
    if len(papers) < 2:
        st.info("At least two indexed papers are needed for a comparison.")
        return
    by_label = {paper_label(p): p for p in papers}
    chosen = st.multiselect("Papers to compare (2 or more)", list(by_label), key="cmp_papers")
    aspects = st.multiselect("Aspects", list(ASPECTS), default=list(ASPECTS), key="cmp_aspects")
    st.info(CAVEAT)
    if len(chosen) < 2 or not aspects:
        return
    st.dataframe([{"ID": by_label[c]["paper_id"], "Title": by_label[c]["title"], "Year": by_label[c]["year"],
                   "Identifier": by_label[c]["identifier"]} for c in chosen], hide_index=True, width="stretch")
    if st.button("Run comparison", type="primary"):
        ids = [by_label[c]["paper_id"] for c in chosen]
        bar = st.progress(0.0, text="Starting")
        try:
            results = compare_papers(sv.assistant, ids, aspects, session_key(),
                                     lambda name, f: bar.progress(f, text=f"Comparing: {name}"))
            st.session_state["cmp_results"] = results
        except Exception as e:  # noqa: BLE001
            st.error(friendly_error(e))
        bar.empty()
    results = st.session_state.get("cmp_results")
    if results:
        tabs = st.tabs(list(results))
        for tab, (name, res) in zip(tabs, results.items()):
            with tab:
                render_answer(res, key=f"cmp_{name}", show_debug=False)
