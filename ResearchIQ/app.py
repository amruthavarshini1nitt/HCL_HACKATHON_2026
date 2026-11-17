"""ResearchIQ Streamlit entry point. UI composition only; logic lives in the packages."""
from __future__ import annotations

import streamlit as st

from config import setup_logging

setup_logging()
st.set_page_config(page_title="ResearchIQ", page_icon="📚", layout="wide")

from ui.chat_tab import render_chat  # noqa: E402
from ui.comparison_tab import render_comparison  # noqa: E402
from ui.dashboard import render_dashboard  # noqa: E402
from ui.evaluation_tab import render_evaluation  # noqa: E402
from ui.papers_tab import render_papers  # noqa: E402
from ui.services import get_services  # noqa: E402


def main() -> None:
    sv = get_services()
    tabs = st.tabs(["Dashboard", "Paper Management", "Research Chat", "Paper Comparison", "Evaluation"])
    with tabs[0]:
        render_dashboard(sv)
    with tabs[1]:
        render_papers(sv)
    with tabs[2]:
        render_chat(sv)
    with tabs[3]:
        render_comparison(sv)
    with tabs[4]:
        render_evaluation()


main()
