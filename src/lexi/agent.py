"""The agent: a LangGraph tool-calling loop.

This is deliberately a CYCLE, not a pipeline. There is no fixed
retrieve -> analyse -> summarise sequence anywhere in this file. The graph is:

        agent  --(wants tools)-->  tools
          ^                          |
          +--------------------------+
          |
        (terminal contract called, or budget spent)
          |
          v
         END

The model decides, every turn, which tool to call next and when it has enough.
A general question may take one search; a precedent-research task may take a
dozen steps across screening, filtering, reading and quantum computation. Same
graph, no branching on query type.

Depth control is a BUDGET, not a gate:
  1. A triage call sets an initial step allowance from the question itself.
  2. The agent may exceed it, up to a ceiling, while it is still productive
     (still surfacing documents it has not seen). Spinning on repeat calls
     stops it.
  3. The shape of the output is itself a tool choice -- `submit_research_report`
     vs `submit_answer` -- so the three-part structure is guaranteed when
     relevant without any code path forcing it.
"""
from __future__ import annotations

import re
import time

from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .config import settings
from .llm import LLM, ContentFiltered, backoff_429, is_content_filter, throttle
from .schemas import DirectAnswer, PrecedentResearchReport
from .tools import ToolBox
from .trace import EventKind, Trace

# Recovers doc_ids from a prose answer so citations are not lost with it.
_DOC_ID_IN_TEXT = re.compile(r"\bdoc_\d{3}\b")

# Rule 3 in two variants so the change can be measured rather than assumed.
_ADVERSE_HUNT_SIDE_RELATIVE = """3. When the task is adversarial research, you MUST \
actively hunt for precedents the OTHER SIDE will use. First work out WHOSE SIDE YOU ARE \
ON, then search for the outcome that favours the OPPOSING side:
   - acting for a claimant or victim    -> `search_precedents(favours='insurer')`
   - acting for an insurer or defendant -> `search_precedents(favours='claimant')`
Search BOTH the opposing outcome and 'mixed' -- pay-and-recover orders concede a breach \
and the other side will cite them. Cover the pool; do not stop at the first two or three. \
Do not assume you act for the claimant: "I act for the insurer" flips which judgments are \
adverse, and searching the wrong side finds none at all."""

_ADVERSE_HUNT_FIXED = """3. When the task is adversarial research, you MUST actively hunt \
for precedents the OTHER SIDE will use -- try `search_precedents` with favours='insurer', \
or filter on the outcome going against the claimant. A brief that only finds favourable \
cases is malpractice."""

# Third variant. The side-relative rule above asked the agent to work out its own
# side every turn, and burial returned (0 -> 12): the conditional spent the
# agent's attention deciding sides instead of covering the pool. This one
# resolves the side ONCE, at triage, and hands the loop a constant with the same
# shape as the fixed rule that works. The agent never reasons about sides; the
# rule arrives pre-resolved. Fixes the failure the fixed rule cannot: with an
# insurer client it hunts the wrong pool, and in the last full run the agent
# buried all 11 outright claimant wins it had itself retrieved on q05.
_ADVERSE_HUNT_TRIAGE_RESOLVED = """3. When the task is adversarial research, you MUST \
actively hunt for precedents the OTHER SIDE will use. In this matter you act for the \
{client}; the opposing side is the {opponent}. Judgments whose outcome favours the \
{opponent} belong in your ADVERSE list; judgments favouring the {client} belong in \
SUPPORTING. Hunt the adverse pool deliberately: try `search_precedents` with \
favours='{opponent}', and also favours='mixed' -- pay-and-recover orders concede a \
breach and the other side will cite them. Cover the pool; do not stop at the first two \
or three. A brief that only finds favourable cases is malpractice."""


_ADVERSE_HUNT_TRIAGE_EXPLICIT = """3. When the task is adversarial research, you MUST \
actively hunt for precedents the OTHER SIDE will use. In this matter you act for the \
{client}; the opposing side is the {opponent}. Judgments whose outcome favours the \
{opponent} belong in your ADVERSE list; judgments favouring the {client} belong in \
SUPPORTING. Hunt the adverse pool deliberately, and START with the exhaustive call:
   `filter_judgments(outcome_favours='{opponent}')`   <- your ADVERSE pool
   `filter_judgments(outcome_favours='mixed')`        <- also adverse; pay-and-recover \
orders concede a breach and the {opponent} will cite them
   `filter_judgments(outcome_favours='{client}')`     <- your SUPPORTING pool
Run the {opponent} filter BEFORE you finish, every time. It scans all judgments and \
returns every match, so it is the only call that can tell you the size of the pool you \
are up against. `search_precedents(favours='{opponent}')` is a useful supplement but \
returns a ranked slice, not the pool. Do not filter on '{client}' and assume you have \
covered the adverse side -- that is your own side, and it is the most common way this \
goes wrong. Cover the pool; do not stop at the first two or three. A brief that only \
finds favourable cases is malpractice."""

SYSTEM_PROMPT = """You are a precedent research associate at an Indian law firm. You \
research a fixed corpus of {n_docs} court judgments and produce analysis a litigator \
can actually use.

TOOLS AND THEIR COSTS
- `filter_judgments` is exact and cheap. It scans every judgment's extracted metadata \
and returns ALL matches. Use it for enumerative questions ("which judgments involve X?") \
where completeness matters.
- `search_precedents` is semantic + keyword, reranked. Your default for "find precedents \
about X" questions.
- `screen_corpus` reads a summary of EVERY judgment. Slow and expensive, but exhaustive. \
Use it when recall genuinely matters, or to verify a search missed nothing.
- `read_judgment` opens one case for verbatim quotes. Do this before citing anything.
- `compute_quantum` does compensation arithmetic deterministically. Never estimate \
figures yourself.

HOW TO DECIDE DEPTH
Match effort to the question. A factual lookup deserves one or two tool calls and \
`submit_answer`. A research brief ("find precedents supporting X", "analyse this case") \
deserves several: search from multiple angles, deliberately hunt adverse authority, read \
the strongest cases, then `submit_research_report`.

NON-NEGOTIABLE RULES
1. Cite ONLY judgments from this corpus, by doc_id. You have background legal knowledge; \
do not present it as a corpus finding. If the corpus does not support a point, say so.
2. Before citing a case, `read_judgment` to confirm it says what you think.
{adverse_hunt_rule}
4. ADVERSE MEANS "THE OPPOSING SIDE WILL CITE THIS", NOT "THIS DEFEATS US". These are very \
different tests and the second one is far too narrow. A judgment that is distinguishable \
on the facts STILL BELONGS in the adverse list, with the distinction stated -- because \
opposing counsel will cite it anyway and your client's lawyer has to be ready for it. \
Writing "not adverse, different facts" in a caveat leaves them unprepared for an argument \
that is certainly coming. Put it in `adverse`, give it a risk level, and give the counter.
5. Risk level measures HOW HARD IT IS TO ANSWER, not whether you will win:
   high   = could defeat or gut the claim on its own
   medium = forces a real argument that could go either way
   low    = they will cite it, you have a clean answer, but you must have that answer ready
   "low" does NOT mean "leave it out". It means "answerable".
6. The only judgments that do not belong in `adverse` are ones the opposing side has no \
reason to cite at all -- a different area of law entirely. Those go in `caveats`.
7. Assess risk with differentiation. If everything is "medium" you have not assessed \
anything.
8. This corpus is MIXED -- it contains judgments from unrelated fields. Do not stretch an \
irrelevant case to fit.

Finish by calling exactly one terminal tool: `submit_research_report` for research tasks, \
`submit_answer` for everything else."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    budget: int
    seen_docs: list[str]
    # How many times the model has been pushed back for answering in prose
    # instead of calling a terminal tool. Bounded so the loop always terminates.
    nudges: int


class _Triage(BaseModel):
    """Sets the *starting* allowance. Never blocks a path -- see module docstring."""

    depth: str = Field(..., description="'simple' or 'research'")
    client_side: str = Field(
        "none",
        description="Whose position the requester holds: 'claimant', 'insurer', or "
                    "'none' when the question states or implies no side.",
    )
    reason: str


TRIAGE_PROMPT = """Classify how much research effort this request needs.

'simple'   -- a factual lookup, enumeration, or definition answerable from metadata or
              one search. Examples: "which judgments involve <some attribute>?",
              "what figure did <a specific document> apply?", "summarise <a document>".
'research' -- requires finding, comparing and weighing precedents, or producing strategy.
              Examples: "find precedents supporting <a legal argument>",
              "analyse this case brief", "what could the other side use against us?".

When genuinely ambiguous, choose 'research' -- under-researching is the costlier error.

Separately, state whose position the requester holds. "I act for the insurer", \
defending against a claim, or resisting compensation -> 'insurer'. A victim, dependant, \
or party seeking compensation -> 'claimant'. An enumerative or neutral question with no \
side -> 'none'. Do not guess a side that is not stated or clearly implied.

REQUEST: {question}"""


TRIAGE_PROMPT_ENUM_SIMPLE = """Classify how much research effort this request needs.

'simple'   -- a factual lookup, enumeration, or definition answerable from metadata or
              one search. Examples: "which judgments involve <some attribute>?",
              "what figure did <a specific document> apply?", "summarise <a document>".
              An enumeration is STILL simple when it asks for a detail per item --
              "which judgments cite X, and what did each take from it?", "which apply
              Y, and at what percentage?", "what multipliers were used, and what
              compensation resulted?" are all 'simple'. Reporting an attribute
              alongside each hit is still reporting; it is not weighing authority.
'research' -- requires finding, comparing and weighing precedents, or producing strategy.
              The test is whether the answer must take a POSITION -- argue for a party,
              assess risk, recommend a course of action, or say which authority is
              stronger. Examples: "find precedents supporting <a legal argument>",
              "analyse this case brief", "what could the other side use against us?".

When genuinely ambiguous, choose 'research' -- under-researching is the costlier error.

Separately, state whose position the requester holds. "I act for the insurer", \
defending against a claim, or resisting compensation -> 'insurer'. A victim, dependant, \
or party seeking compensation -> 'claimant'. An enumerative or neutral question with no \
side -> 'none'. Do not guess a side that is not stated or clearly implied.

REQUEST: {question}"""


def _triage(question: str, llm: LLM, trace: Trace) -> int:
    """Set the STARTING step allowance from the request itself.

    A table lookup, not a branch: both depths run the identical graph with the
    identical tools, and only the initial allowance differs -- which the agent
    can then escalate on its own (see `route`). Nothing about the workflow forks
    here, and an unrecognised label defaults to the deeper allowance.
    """
    budgets = {"simple": settings.budget_simple, "research": settings.budget_deep}
    try:
        prompt = (TRIAGE_PROMPT_ENUM_SIMPLE if settings.enable_enumeration_is_simple
                  else TRIAGE_PROMPT)
        res = llm.structured(prompt.format(question=question), _Triage)
        budget = budgets.get(res.depth.strip().lower(), settings.budget_deep)
        side = res.client_side.strip().lower()
        if side not in ("claimant", "insurer"):
            side = "none"
        detail = f"{res.depth} / side={side} -- {res.reason}"
    except Exception as e:  # triage must never block the run
        budget, side = settings.budget_deep, "none"
        detail = f"triage failed ({e}); defaulting to research"
    trace.add(EventKind.BUDGET, "triage", detail=detail,
              payload={"budget": budget, "client_side": side})
    return budget, side


def _hunt_rule(side: str) -> str:
    """Pick rule 3. The first branch keeps the measured-negative variant
    reproducible; the second is the triage-resolved fix under ablation; the
    fixed rule is the shipping default."""
    if settings.enable_side_relative_adverse:
        return _ADVERSE_HUNT_SIDE_RELATIVE
    if settings.enable_triage_side_resolution and side in ("claimant", "insurer"):
        opponent = "insurer" if side == "claimant" else "claimant"
        rule = (_ADVERSE_HUNT_TRIAGE_EXPLICIT if settings.enable_explicit_adverse_filter
                else _ADVERSE_HUNT_TRIAGE_RESOLVED)
        return rule.format(client=side, opponent=opponent)
    return _ADVERSE_HUNT_FIXED


def _compact_history(messages: list, keep_full: int, cap: int) -> list:
    """Shrink old ToolMessages so the history stops being resent in full.

    The agent loop hands the ENTIRE conversation to the model on every turn, so a
    tool result is transmitted once for every turn that follows it. Measured, that
    makes cost quadratic in tool calls (tokens ~= 600 * n^2) and a single
    read_judgment result -- 2,521 tokens of judgment text -- gets resent twenty
    times on a long run.

    Older results are replaced by their first line plus a marker. That line
    carries the identifying information (doc_id, title, holding, match counts);
    what is dropped is the raw passage text the agent has already reasoned over.
    The `keep_full` most recent are left untouched, because those are what the
    current turn is actually working with.

    If the agent later needs a verbatim quote from a compacted read, it can call
    `read_judgment` again -- one call against tens of thousands of tokens saved.
    And because the quote sanitiser always runs, a forgotten quote becomes a
    missing quote, never a fabricated one.
    """
    tool_idx = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if len(tool_idx) <= keep_full:
        return messages
    stale = set(tool_idx[:-keep_full])

    out = []
    for i, m in enumerate(messages):
        if i not in stale:
            out.append(m)
            continue
        text = str(getattr(m, "content", ""))
        if len(text) <= cap:
            out.append(m)
            continue
        head = text.split("\n", 1)[0][:cap]
        out.append(
            ToolMessage(
                content=f"{head}\n[... {len(text) - len(head):,} chars compacted; "
                        f"call the tool again if the full text is needed ...]",
                tool_call_id=m.tool_call_id,
            )
        )
    return out


def _digest_read(text: str, per_passage: int) -> str:
    """Digest one stale read_judgment result, keeping what the agent still uses.

    The reverted truncation variant (see _compact_history) taught the design
    constraint: the agent will pay whatever it costs to recover information it
    still needs. So this keeps ALL of the structured head -- holding, ratio,
    outcome, statutes, citations, the fields its reasoning actually runs on --
    and the opening of each passage for quotable material, dropping only deep
    passage text. And it never invites a re-read: the banner says the full text
    was already analysed, not "call the tool again".
    """
    head, sep, tail = text.partition("RELEVANT PASSAGES:\n")
    if not sep:
        return text  # unexpected shape; leave it alone
    outs = []
    for block in tail.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if len(block) > per_passage:
            cut = block[:per_passage]
            if " " in cut:
                cut = cut[: cut.rfind(" ")]
            block = cut + " ..."
        outs.append(block)
    return (
        head
        + "PASSAGE OPENINGS (digest of a full read earlier this session; "
        + "already analysed in full):\n"
        + "\n\n".join(outs)
    )


def _digest_stale_reads(messages: list, keep_full: int, per_passage: int) -> list:
    """Replace old read_judgment results with information-preserving digests.

    Only read results are touched -- they are ~70% of a long history and by far
    the largest messages -- and only once they age out of the most recent
    `keep_full` tool results. The transformation is applied to the copy sent to
    the model; graph state keeps the originals, so the recent window is always
    verbatim and nothing is lost permanently.
    """
    # The window counts READS, not all tool results. The v2 ablation failed
    # partly because any tool message consumed a window slot: a read followed
    # by two searches was digested two turns after the agent saw it, while it
    # was still quoting from it -- that is where the duplicate re-reads and
    # the starved runs came from. Searches are never digested, so they must
    # not age reads out either.
    read_idx = [
        i for i, m in enumerate(messages)
        if isinstance(m, ToolMessage)
        and str(getattr(m, "content", "")).startswith("[doc_")
        and "RELEVANT PASSAGES:" in str(getattr(m, "content", ""))
    ]
    if len(read_idx) <= keep_full:
        return messages
    stale = set(read_idx[:-keep_full])
    out = []
    for i, m in enumerate(messages):
        text = str(getattr(m, "content", ""))
        if i in stale:
            out.append(
                ToolMessage(content=_digest_read(text, per_passage),
                            tool_call_id=m.tool_call_id)
            )
        else:
            out.append(m)
    return out


def _invoke_retrying(model, msgs, attempts: int | None = None):
    """model.invoke with the same throttle/backoff policy as LLM.complete().

    The tool-bound client bypasses complete(), which used to mean the agent's
    own turns -- the heaviest requests in the system -- had no retries, no rate
    limiting and no 429 coordination. One timed-out request killed a run; then,
    with naive retries added, a 429 still did, because a 5s sleep does not
    clear a saturated per-minute bucket while the other workers keep spending
    it. Now: every attempt takes a limiter slot, 429s pause the whole process
    and obey the server's stated wait, transient faults back off exponentially.
    Content filters are never retried -- identical input gets an identical
    refusal, and agent_node already routes those -- and non-transient errors
    surface immediately (hiding one behind retries cost an afternoon once).
    """
    attempts = attempts or settings.max_retries
    last: Exception | None = None
    for attempt in range(attempts):
        throttle()
        try:
            return model.invoke(msgs)
        except Exception as e:  # noqa: BLE001 - provider raises wrapped types
            last = e
            msg, low = str(e), str(e).lower()
            if is_content_filter(e):
                raise
            if "429" in msg or "rate limit" in low:
                time.sleep(backoff_429(msg))
                continue
            transient = any(s in low for s in (
                "timeout", "connection", "reset", "temporarily",
            )) or any(s in msg for s in ("500", "502", "503", "504"))
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(min(5 * 2 ** attempt, 40))
    raise last


def build_graph(toolbox: ToolBox, llm: LLM):
    tools = toolbox.build()
    model = llm._client().bind_tools(tools)
    by_name = {t.name: t for t in tools}
    trace = toolbox.trace

    def agent_node(state: AgentState) -> dict:
        msgs = state["messages"]
        if settings.enable_history_compaction:
            msgs = _compact_history(
                msgs, settings.history_keep_full, settings.history_summary_chars
            )
        if settings.enable_digest_compaction:
            msgs = _digest_stale_reads(
                msgs, settings.digest_keep_full, settings.digest_passage_chars
            )
        try:
            resp = _invoke_retrying(model, msgs)
        except Exception as e:
            # The tool-bound client is invoked directly, so it never passes
            # through LLM.complete() and never receives that method's typed error
            # translation. The filter has to be detected here too: an earlier
            # version handled only ContentFiltered and 7 of 30 eval runs still
            # died on a raw provider error.
            if not is_content_filter(e):
                trace.add(EventKind.ERROR, "agent_turn_failed", detail=str(e)[:250])
                raise
            # One refused turn must not lose the whole run. Tell the model plainly
            # what happened so it can paraphrase, and let the loop continue with
            # what it has already gathered.
            trace.add(EventKind.ERROR, "content_filtered", detail=str(e)[:250])
            return {
                "messages": [
                    HumanMessage(
                        content=(
                            "Your previous turn was refused by the provider's content "
                            "filter, most likely because it reproduced graphic detail "
                            "from a judgment. Do not quote that passage. Paraphrase the "
                            "legal point neutrally, or proceed with what you have "
                            "already gathered, then call a terminal tool."
                        )
                    )
                ],
                "steps": state["steps"] + 1,
            }

        # The tool-bound client is invoked directly (bind_tools has no equivalent
        # on our wrapper), so usage has to be recorded here or the trace
        # under-reports cost by every agent turn.
        meta = getattr(resp, "usage_metadata", None) or {}
        llm.usage.add(meta.get("input_tokens", 0), meta.get("output_tokens", 0))
        if isinstance(resp, AIMessage) and resp.tool_calls:
            for tc in resp.tool_calls:
                trace.add(
                    EventKind.TOOL_CALL,
                    tc["name"],
                    detail=_brief_args(tc["args"]),
                    payload={"args": tc["args"]},
                )
        elif getattr(resp, "content", None):
            trace.add(EventKind.PLAN, "reasoning", detail=_first_line(resp.content))
        return {"messages": [resp], "steps": state["steps"] + 1}

    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        out: list[ToolMessage] = []
        for tc in getattr(last, "tool_calls", []):
            tool = by_name.get(tc["name"])
            if tool is None:
                out.append(ToolMessage(content=f"Unknown tool {tc['name']}", tool_call_id=tc["id"]))
                continue
            try:
                result = tool.invoke(tc["args"])
            except Exception as e:
                result = f"Tool error: {e}"
                trace.add(EventKind.ERROR, tc["name"], detail=str(e)[:300])
            out.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        seen = list(dict.fromkeys(state["seen_docs"] + trace.retrieved_doc_ids()))
        return {"messages": out, "seen_docs": seen}

    def route(state: AgentState) -> str:
        """Continue, escalate, or stop. The only control flow in the system."""
        if toolbox.result is not None:          # terminal contract emitted
            return END
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            # The model answered in prose instead of calling a terminal tool.
            #
            # Ending here was a structural hole, not a cosmetic one: EVERY
            # validation gate lives inside `submit_research_report` /
            # `submit_answer`, so a prose reply bypassed all of them. Measured
            # cost -- one held-out query did 14 tool calls of real research, then
            # answered in prose, and accounted for all 6 held-out burials because
            # nothing checked it and no citations were recorded.
            #
            # So the run is not allowed to end this way while there is any budget
            # left. Push back and require a terminal tool; the cap guarantees
            # termination, and `finalise` still salvages the text if the model
            # refuses to comply.
            if state["nudges"] < settings.max_terminal_nudges:
                state["nudges"] += 1
                trace.add(
                    EventKind.BUDGET,
                    "terminal_tool_required",
                    detail=f"answered in prose without a terminal contract "
                           f"(nudge {state['nudges']}/{settings.max_terminal_nudges})",
                )
                return "require_terminal"
            return END

        if state["steps"] < state["budget"]:
            return "tools"

        # At the limit: extend only if still productive and below the ceiling.
        newly_found = len(state["seen_docs"]) > len(set(state["seen_docs"][: -1] or []))
        if state["budget"] < settings.budget_ceiling and newly_found:
            state["budget"] = min(state["budget"] + 4, settings.budget_ceiling)
            trace.add(
                EventKind.BUDGET,
                "escalate",
                detail=f"still discovering documents at step {state['steps']}; "
                       f"budget -> {state['budget']}",
                payload={"budget": state["budget"]},
            )
            return "tools"

        trace.add(
            EventKind.BUDGET,
            "exhausted",
            detail=f"stopped at step {state['steps']} (budget {state['budget']})",
        )
        return "finalise"

    def require_terminal_node(state: AgentState) -> dict:
        """Refuse a prose answer and require a terminal contract.

        Naming the tools explicitly matters: a vague "please use a tool" gets a
        vague retry. This states which contract fits and why the prose reply was
        not accepted.
        """
        nudge = HumanMessage(
            content=(
                "You answered in prose without calling a terminal tool, so nothing was "
                "recorded and none of your research was captured.\n\n"
                "Call exactly one now:\n"
                "  `submit_research_report` -- for anything involving precedents for or "
                "against a position. It has the fields for supporting authority, adverse "
                "authority and strategy.\n"
                "  `submit_answer` -- for a factual or enumerative question.\n\n"
                "Use the judgments you already retrieved and cite them by doc_id. Do not "
                "start new research."
            )
        )
        return {"messages": [nudge], "steps": state["steps"] + 1}

    def finalise_node(state: AgentState) -> dict:
        """Budget ran out mid-run: force a terminal contract from what we have."""
        nudge = HumanMessage(
            content=(
                "You have run out of research budget. Call a terminal tool NOW "
                "(`submit_research_report` or `submit_answer`) using only what you have "
                "already retrieved. Note any gaps in `caveats`."
            )
        )
        resp = _invoke_retrying(model, state["messages"] + [nudge])
        return {"messages": [nudge, resp]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("finalise", finalise_node)
    g.add_node("require_terminal", require_terminal_node)
    g.set_entry_point("agent")
    g.add_conditional_edges(
        "agent", route,
        {"tools": "tools", "finalise": "finalise",
         "require_terminal": "require_terminal", END: END},
    )
    g.add_edge("require_terminal", "agent")
    g.add_edge("tools", "agent")            # <-- the cycle
    g.add_conditional_edges(
        "finalise", lambda s: "tools" if getattr(s["messages"][-1], "tool_calls", None) else END,
        {"tools": "tools", END: END},
    )
    return g.compile()


class Agent:
    """Public entry point. One instance per question."""

    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM(model=settings.chat_model)

    def run(self, question: str, n_docs: int = 56) -> tuple[Trace, object]:
        trace = Trace(question=question)
        toolbox = ToolBox(trace, self.llm)
        budget, side = _triage(question, self.llm, trace)
        graph = build_graph(toolbox, self.llm)

        state: AgentState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT.format(
                    n_docs=n_docs,
                    adverse_hunt_rule=_hunt_rule(side),
                )),
                HumanMessage(content=question),
            ],
            "steps": 0,
            "budget": budget,
            "seen_docs": [],
            "nudges": 0,
        }
        final = graph.invoke(state, {"recursion_limit": settings.budget_ceiling * 3})
        _capture_untooled_answer(toolbox, final, trace, question)

        trace.llm_calls = self.llm.usage.calls
        trace.in_tokens = self.llm.usage.in_tokens
        trace.out_tokens = self.llm.usage.out_tokens
        return trace, toolbox.result

    def stream(self, question: str, n_docs: int = 56):
        """Yield trace events as they happen, for the live UI."""
        trace = Trace(question=question)
        toolbox = ToolBox(trace, self.llm)
        budget, side = _triage(question, self.llm, trace)
        yield from trace.events

        graph = build_graph(toolbox, self.llm)
        state: AgentState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT.format(
                    n_docs=n_docs,
                    adverse_hunt_rule=_hunt_rule(side),
                )),
                HumanMessage(content=question),
            ],
            "steps": 0,
            "budget": budget,
            "seen_docs": [],
            "nudges": 0,
        }
        emitted = len(trace.events)
        for _ in graph.stream(state, {"recursion_limit": settings.budget_ceiling * 3}):
            while emitted < len(trace.events):
                yield trace.events[emitted]
                emitted += 1
        while emitted < len(trace.events):
            yield trace.events[emitted]
            emitted += 1

        trace.llm_calls = self.llm.usage.calls
        trace.in_tokens = self.llm.usage.in_tokens
        trace.out_tokens = self.llm.usage.out_tokens
        self.last_trace, self.last_result = trace, toolbox.result


def _capture_untooled_answer(
    toolbox: ToolBox, final_state, trace: Trace, question: str
) -> None:
    """Salvage an answer the model wrote without calling a terminal tool.

    The router ends the run when a turn arrives with no tool calls. On short
    lookups the model sometimes just writes the answer in prose -- correct
    content, wrong channel -- and the run returned None, discarding work the user
    could see happening in the trace. Rather than forcing a retry, wrap the text
    in the direct-answer contract and record that it arrived this way.
    """
    from .llm import _text_of

    if toolbox.result is not None:
        return
    messages = (final_state or {}).get("messages", []) if isinstance(final_state, dict) else []
    for msg in reversed(messages):
        if getattr(msg, "type", None) != "ai" or getattr(msg, "tool_calls", None):
            continue
        text = _text_of(msg.content).strip()
        if not text:
            continue
        cited = sorted(set(_DOC_ID_IN_TEXT.findall(text)))
        # `question` is required by the contract. Omitting it made this
        # fallback raise on every use -- so the salvage path crashed the exact
        # runs it existed to rescue.
        toolbox.result = DirectAnswer(
            question=question, answer=text, cited_doc_ids=cited, confidence="medium"
        )
        trace.add(
            EventKind.ANSWER,
            "answer_without_terminal_tool",
            detail="model answered in prose; captured as a direct answer",
            payload={"doc_ids": cited},
        )
        return


def _brief_args(args: dict) -> str:
    bits = []
    for k, v in args.items():
        s = str(v)
        bits.append(f"{k}={s[:70] + '...' if len(s) > 70 else s}")
    return " ".join(bits)[:300]


def _first_line(content) -> str:
    from .llm import _text_of

    return _text_of(content).strip().split("\n")[0][:300]
