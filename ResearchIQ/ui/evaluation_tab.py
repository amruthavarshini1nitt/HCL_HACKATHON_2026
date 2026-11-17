"""Tab E: evaluation dashboard. Shows only results written by real runs; never placeholder numbers.

Expected files (written by evaluation scripts):
  evaluation/questions.json                 list of evaluation questions
  evaluation/results/eval_results.json      {"run_at","k","summary":{...},"cases":[...]}
  evaluation/results/security_results.json  from evaluation.security_tests
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

EVAL_DIR = Path(__file__).resolve().parents[1] / "evaluation"


def _load(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        return f"{v:.1%}" if pct else f"{v:.3f}"
    return str(v)


def render_evaluation() -> None:
    st.subheader("Evaluation")
    questions = _load(EVAL_DIR / "questions.json")
    results = _load(EVAL_DIR / "results" / "eval_results.json")
    security = _load(EVAL_DIR / "results" / "security_results.json")

    n_q = len(questions) if isinstance(questions, list) else (len(questions.get("questions", []))
                                                              if isinstance(questions, dict) else 0)
    st.metric("Evaluation questions defined", n_q)
    if not results:
        st.info("No evaluation run recorded yet. Run `python -m evaluation.answer_evaluation` after indexing "
                "the papers; results will appear here.")
    else:
        s = results.get("summary", {})
        st.caption(f"Run at {results.get('run_at', 'unknown')} · K = {results.get('k', 'n/a')} · "
                   f"{s.get('n_questions', len(results.get('cases', [])))} questions")
        r = s.get("retrieval", {})
        st.markdown("**Retrieval metrics**")
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [("Recall@K", "recall@k"), ("Precision@K", "precision@k"),
                                            ("MRR", "mrr"), ("nDCG@K", "ndcg@k")]):
            col.metric(label, _fmt(r.get(key)))
        a = s.get("answer", {})
        st.markdown("**Answer quality**")
        cols = st.columns(3)
        cols[0].metric("Correctness", _fmt(a.get("correctness"), pct=True))
        cols[1].metric("Groundedness", _fmt(a.get("groundedness"), pct=True))
        cols[2].metric("Citation accuracy", _fmt(s.get("citation_accuracy"), pct=True))
        lat = s.get("latency", {})
        st.markdown("**Latency (seconds)**")
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [("Mean", "mean_s"), ("Median", "p50_s"), ("p95", "p95_s"),
                                            ("Max", "max_s")]):
            col.metric(label, _fmt(lat.get(key)))
        if lat.get("stages"):
            st.dataframe([{"stage": k, "mean_s": v} for k, v in lat["stages"].items()], hide_index=True)

        cases = results.get("cases", [])
        failed = [c for c in cases if not c.get("passed", True)]
        st.markdown(f"**Failed test cases ({len(failed)} of {len(cases)})**")
        for c in failed:
            with st.expander(f"{c.get('id', '?')} · {c.get('type', '')} · {str(c.get('question', ''))[:80]}"):
                st.write("Reason:", c.get("failure_reason") or "n/a")
                st.write("Expected:", c.get("expected_sources"))
                st.write("Retrieved:", c.get("retrieved_ids"))
                st.write("Answer:", c.get("answer"))
        with st.expander("All cases"):
            st.dataframe([{k: (str(v)[:120] if isinstance(v, (list, dict)) else v) for k, v in c.items()
                           if k not in ("answer", "retrieved_evidence")} for c in cases], hide_index=True)

    st.markdown("**Security tests**")
    if not security:
        st.info("No security run recorded. Run `python -m evaluation.security_tests`.")
    else:
        sm = security["summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Passed", f"{sm['passed']}/{sm['total']}")
        c2.metric("Failed", len(sm["failed"]))
        c3.metric("Prompt Guard 2 tested", "yes" if sm.get("prompt_guard_tested") else "no")
        for name in sm["failed"]:
            st.error(name)
        with st.expander("All security cases"):
            st.dataframe([{k: r[k] for k in ("category", "name", "passed", "detail")} for r in security["results"]],
                         hide_index=True)
