"""Streamlit UI.

The brief is explicit: "Intermediate reasoning steps must be visible -- we want
to see which documents the agent retrieved, how it ranked them, and how it
arrived at its conclusions. Do not show only the final output."

So the trace is the primary surface, not a debug panel. Every retrieval renders
its full score decomposition (dense rank, BM25 rank, fused rank, rerank score
and the reranker's stated reason), and every cited judgment is expandable to the
verbatim passage the agent actually read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lexi.agent import Agent  # noqa: E402
from lexi.config import settings  # noqa: E402
from lexi.schemas import DirectAnswer, PrecedentResearchReport  # noqa: E402
from lexi.trace import EventKind  # noqa: E402

st.set_page_config(page_title="Lexi Precedent Research", page_icon="⚖️", layout="wide")

CASE_BRIEF = """Client: Mrs. Lakshmi Devi
Matter: Motor accident claim - death of spouse

Mrs. Lakshmi Devi's husband was killed in a road accident involving a commercial \
truck. The truck driver was operating the vehicle without a valid driving licence at \
the time of the accident. The insurance company (National Insurance Co.) is denying \
the claim on the ground that the motor insurance policy is void due to the driver \
being unlicensed, and therefore they bear no liability to pay compensation.

Key facts:
- The deceased was 42 years old at the time of the accident
- Monthly income: Rs 35,000
- Dependents: wife and two minor children (ages 8 and 12)
- The truck was a commercial vehicle owned by a transport company
- The truck driver did not hold a valid driving licence
- The insurance company is contesting liability, arguing the policy is void

Research the corpus and advise: supporting precedents, adverse precedents, and strategy."""

EXAMPLES = {
    "— pick an example —": "",
    "Full case brief (deep research)": CASE_BRIEF,
    "Which judgments involve commercial vehicles?": "Which of these judgments involve commercial vehicles?",
    "Contributory negligence precedents": "Find precedents that support our argument on contributory negligence.",
    "Adverse research (insurer's side)": (
        "I act for the insurer. Find precedents I can use to defeat a claim where the "
        "driver had no valid licence, and assess how strong each one is."
    ),
    "Judgments citing Swaran Singh": "Which judgments cite National Insurance v. Swaran Singh, and what did each take from it?",
    "Quantum question": (
        "For a deceased aged 42 earning Rs 35,000/month with 3 dependants, what "
        "compensation is realistic, and which judgments in the corpus support that range?"
    ),
}

_KIND_ICON = {
    EventKind.PLAN: "💭",
    EventKind.TOOL_CALL: "🔧",
    EventKind.RETRIEVAL: "🔍",
    EventKind.FILTER: "🗂️",
    EventKind.SCREEN: "📖",
    EventKind.READ: "📄",
    EventKind.COMPUTE: "🧮",
    EventKind.BUDGET: "⚙️",
    EventKind.ANSWER: "✅",
    EventKind.ERROR: "❌",
}


# --- rendering ---------------------------------------------------------------


def render_event(ev, container) -> None:
    icon = _KIND_ICON.get(ev.kind, "•")
    with container.expander(f"{icon}  **{ev.label}** — {ev.detail[:150]}", expanded=False):
        st.caption(f"step {ev.seq} · {ev.kind.value} · {ev.elapsed_s}s")
        if ev.detail:
            st.text(ev.detail)

        if ev.docs and ev.kind == EventKind.RETRIEVAL:
            st.markdown("**Ranked results — full score decomposition**")
            st.dataframe(
                [
                    {
                        "rank": d.final_rank,
                        "doc_id": d.doc_id,
                        "title": (d.title or "")[:52],
                        "rerank /10": d.rerank_score,
                        "dense #": d.dense_rank,
                        "bm25 #": d.sparse_rank,
                        "fused #": d.fused_rank,
                        "why ranked here": d.why[:90],
                    }
                    for d in ev.docs
                ],
                use_container_width=True,
                hide_index=True,
            )
        elif ev.docs:
            st.dataframe(
                [{"doc_id": d.doc_id, "title": (d.title or "")[:70], "note": d.why[:100]}
                 for d in ev.docs],
                use_container_width=True,
                hide_index=True,
            )

        # --- what the agent actually READ ------------------------------------
        if ev.kind == EventKind.READ and ev.payload.get("passages"):
            pl = ev.payload
            st.markdown(f"**[{pl['doc_ids'][0]}] {pl.get('title','')}**")
            if pl.get("court"):
                st.caption(pl["court"])
            if pl.get("holding"):
                st.markdown(f"**Holding:** {pl['holding']}")
            if pl.get("ratio"):
                st.markdown(f"**Ratio:** {pl['ratio']}")
            st.markdown(f"**Passages the agent read** (selected for: _{pl.get('focus') or 'general'}_)")
            for i, psg in enumerate(pl["passages"], 1):
                st.markdown(
                    f"<div style='border-left:3px solid #1f4e79;padding:6px 12px;"
                    f"margin:8px 0;background:#f7f9fc'>"
                    f"<small><b>passage {i}</b> · section: {psg['section']} · "
                    f"relevance {psg['score']}</small><br>{psg['text']}</div>",
                    unsafe_allow_html=True,
                )

        if ev.kind == EventKind.COMPUTE and ev.payload.get("result"):
            r = ev.payload["result"]
            st.markdown("**Computation, step by step (each with its authority)**")
            st.dataframe(r["steps"], use_container_width=True, hide_index=True)


def render_report(rep: PrecedentResearchReport) -> None:
    st.subheader("Supporting precedents")
    if not rep.supporting:
        st.info("None identified.")
    for p in rep.supporting:
        with st.container(border=True):
            st.markdown(f"**[{p.doc_id}] {p.title}**  ·  `{p.strength}`")
            st.markdown(f"**Principle:** {p.principle}")
            st.markdown(f"**Fact alignment:** {p.fact_alignment}")
            st.markdown(f"**Why it matters:** {p.why_it_matters}")
            if p.quote:
                st.caption(f"“{p.quote}”")

    st.subheader("Adverse precedents")
    if not rep.adverse:
        st.warning(
            "No adverse precedents surfaced. In legal practice that is itself a "
            "warning sign — verify before relying on this."
        )
    for a in rep.adverse:
        colour = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(a.risk_level, "⚪")
        with st.container(border=True):
            st.markdown(f"**[{a.doc_id}] {a.title}**  ·  {colour} `{a.risk_level} risk`")
            st.markdown(f"**Principle:** {a.principle}")
            st.markdown(f"**Risk to client:** {a.risk_to_client}")
            st.markdown(f"**How to distinguish / counter:** {a.distinguishing_argument}")
            if a.quote:
                st.caption(f"“{a.quote}”")

    st.subheader("Strategy")
    s = rep.strategy
    with st.container(border=True):
        st.markdown("**Priority arguments**")
        for i, arg in enumerate(s.priority_arguments, 1):
            st.markdown(f"{i}. {arg}")
        st.markdown(f"**Realistic compensation range:** {s.compensation_range}")
        if s.compensation_reasoning:
            st.caption(s.compensation_reasoning)
        st.markdown("**Risks the client should know**")
        for r in s.risks:
            st.markdown(f"- {r}")
        if s.recommended_forum_or_relief:
            st.markdown(f"**Forum / relief:** {s.recommended_forum_or_relief}")

    if rep.caveats:
        st.subheader("Caveats")
        for c in rep.caveats:
            st.markdown(f"- {c}")


def render_answer(ans: DirectAnswer) -> None:
    st.subheader("Answer")
    st.markdown(ans.answer)
    if ans.cited_doc_ids:
        st.caption("Cited: " + ", ".join(ans.cited_doc_ids))
    st.caption(f"Confidence: {ans.confidence}")


# --- page --------------------------------------------------------------------

st.title("⚖️ Lexi — Legal Precedent Research Agent")
st.caption(
    f"Corpus: 56 Indian judgments · retrieval: Qwen3-Embedding-0.6B + BM25 → RRF → LLM rerank "
    f"· agent: LangGraph tool loop on {settings.chat_model}"
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
1. **Triage** sets a step *budget* — not a gate. The agent can escalate it
   mid-run if it is still finding new documents.
2. **The agent picks its own tools** each turn. There is no fixed
   retrieve→analyse→write pipeline.
3. **Retrieval** fuses dense + BM25 by reciprocal rank, then an LLM reranks.
4. **Output shape is a tool choice** — a full research report, or a direct
   answer.

Every step below is the real trace, not a replay.
        """
    )
    st.divider()
    st.caption("Tools available to the agent")
    st.code(
        "search_precedents   hybrid + rerank\n"
        "filter_judgments    exact, all 56\n"
        "screen_corpus       exhaustive LLM screen\n"
        "read_judgment       verbatim passages\n"
        "compute_quantum     deterministic maths",
        language="text",
    )

choice = st.selectbox("Examples", list(EXAMPLES.keys()))
question = st.text_area(
    "Ask anything about the corpus",
    value=EXAMPLES[choice],
    height=180,
    placeholder="e.g. Find precedents where an insurer was exonerated despite a third-party death.",
)

if st.button("Research", type="primary", disabled=not question.strip()):
    trace_box = st.container()
    trace_box.subheader("Agent trace")
    slot = trace_box.empty()
    events_area = trace_box.container()

    agent = Agent()
    n = 0
    try:
        with st.status("Agent working…", expanded=True) as status:
            for ev in agent.stream(question):
                n += 1
                status.update(label=f"Step {n}: {ev.label} — {ev.detail[:70]}")
                render_event(ev, events_area)
            status.update(label=f"Done — {n} steps", state="complete")
    except Exception as e:
        st.error(f"Agent failed: {e}")
        st.stop()

    result = getattr(agent, "last_result", None)
    trace = getattr(agent, "last_trace", None)

    st.divider()
    if isinstance(result, PrecedentResearchReport):
        render_report(result)
    elif isinstance(result, DirectAnswer):
        render_answer(result)
    else:
        st.warning("The agent stopped without emitting a terminal contract.")

    if trace:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps", len(trace.events))
        c2.metric("LLM calls", trace.llm_calls)
        c3.metric("Docs retrieved", len(trace.retrieved_doc_ids()))
        c4.metric("Tokens", f"{trace.in_tokens + trace.out_tokens:,}")
        with st.expander("Download raw trace (JSON)"):
            st.download_button(
                "trace.json", trace.to_json(), file_name="trace.json", mime="application/json"
            )
