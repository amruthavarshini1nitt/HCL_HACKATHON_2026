"""Tab C: research chat with paper filters, citations and an evidence viewer."""
from __future__ import annotations

import streamlit as st

from generation.assistant import AnswerResult
from ui.components import friendly_error, render_answer
from ui.services import Services, indexed_papers, paper_label, session_key

EXAMPLES = [
    "What is the main contribution of each paper?",
    "Compare the methodologies used across the papers.",
    "What do the architecture figures show?",
    "What are the reported limitations and open problems?",
]


def render_chat(sv: Services) -> None:
    papers = indexed_papers(sv.store)
    if not papers:
        st.info("Upload and process papers in **Paper Management** to start asking questions.")
        return
    labels = {paper_label(p): p["paper_id"] for p in papers}
    top = st.columns([4, 1])
    chosen = top[0].multiselect("Search in (leave empty for all papers)", list(labels), key="chat_scope")
    show_debug = top[1].toggle("Debug", value=False, key="chat_debug",
                               help="Show the refined query, sufficiency check and latency")
    scope = [labels[c] for c in chosen] or None
    if top[1].button("Clear chat"):
        st.session_state["chat"] = []
        st.rerun()

    history: list[dict] = st.session_state.setdefault("chat", [])
    for i, m in enumerate(history):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_answer(m["result"], key=f"hist_{i}", show_debug=show_debug)

    if not history:
        st.caption("Try: " + " · ".join(f"_{e}_" for e in EXAMPLES))
    prompt = st.chat_input("Ask a research question about the papers")
    if not prompt:
        return
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and checking the answer..."):
            try:
                res: AnswerResult = sv.assistant.ask(prompt, session_key(), scope)
            except Exception as e:  # noqa: BLE001 - never show internals
                res = AnswerResult("error", friendly_error(e))
        render_answer(res, key=f"new_{len(history)}", show_debug=show_debug)
    history.append({"role": "assistant", "result": res})
