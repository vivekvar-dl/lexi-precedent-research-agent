"""The agent's toolset.

Tool altitude is the whole design question here. Too low ("cosine_similarity")
and the model has to invent retrieval strategy in prose. Too high
("do_precedent_research") and the tool IS the hard-coded pipeline the brief
forbids. These sit in between: each does one retrieval or reasoning primitive
well, and the agent composes them however the question demands.

Every tool writes a structured event to the trace as a side effect, which is
what makes the reasoning visible in the UI and measurable in the evals.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .config import settings
from .index import get_tables
from .llm import LLM
from .quantum import compute_compensation
from .retrieve import (
    _in_clause,
    all_card_summaries,
    filter_cards,
    hybrid_search,
    passages_for,
)

# Used to spot a doc_id mentioned inside a free-text caveat, so "doc_014 is
# distinguishable because ..." counts as having addressed doc_014.
_DOC_ID = re.compile(r"\bdoc_\d{3}\b")

# An "adverse" entry that describes itself in these terms is not adverse. Matched
# against the agent's own risk statement, so this detects self-contradiction
# rather than second-guessing a legal judgement.
# Narrowed deliberately. Under the current definition an adverse entry is one the
# opposing side WILL CITE, so "distinguishable", "answerable" or "ultimately
# fails" are correct things to write there -- the earlier pattern treated those as
# padding and pushed genuinely-cited authority out of the list entirely.
#
# What remains is genuine self-contradiction: an entry claiming the opposing side
# has no reason to raise it at all. That judgment belongs in `caveats`.
_DISCLAIMER = re.compile(
    r"(?i)\b("
    r"(?:wholly |completely |entirely )?(?:irrelevant|unrelated)|"
    r"no bearing (?:on|whatsoever)|different area of law|"
    r"not (?:a )?motor[- ]accident|"
    r"(?:the )?(?:insurer|opposing side|other side) (?:would|will) not cite"
    r")\b"
)
from .schemas import DirectAnswer, PrecedentResearchReport, ScoredDoc
from .trace import EventKind, Trace

# =============================================================================
# Tool argument schemas
# =============================================================================


class SearchArgs(BaseModel):
    query: str = Field(..., description="Legal research query, in natural language")
    top_k: int = Field(8, ge=1, le=25)
    favours: str | None = Field(
        None,
        description=(
            "Optional outcome filter: 'claimant', 'insurer', 'mixed'. Use 'insurer' "
            "to hunt for precedents that went AGAINST a claimant."
        ),
    )


class FilterArgs(BaseModel):
    """Exact structured query. Every argument is optional; they AND together."""

    case_type_contains: str | None = None
    court_contains: str | None = None
    outcome_favours: str | None = Field(None, description="claimant | insurer | mixed | neutral")
    is_commercial_vehicle: bool | None = None
    has_licence_defect: bool | None = None
    is_death_claim: bool | None = None
    contributory_negligence: bool | None = None
    statute_contains: str | None = Field(None, description="e.g. 'section 149' or '163a'")
    precedent_contains: str | None = Field(None, description="e.g. 'swaran singh'")
    principle_contains: str | None = None
    min_multiplier: float | None = None
    max_multiplier: float | None = None


class ScreenArgs(BaseModel):
    criteria: str = Field(
        ..., description="What makes a judgment qualify. Be specific about the legal test."
    )


class ReadArgs(BaseModel):
    doc_id: str
    focus: str = Field("", description="What to look for; drives passage selection")


class QuantumArgs(BaseModel):
    monthly_income: float
    age: int
    dependents: int
    employment: str = Field("self_employed", description="'permanent' or 'self_employed'")
    award_year: int = 2026
    contributory_negligence_pct: float = 0.0


# =============================================================================
# Toolbox
# =============================================================================


class ToolBox:
    """Builds LangChain tools bound to one run's trace and LLM."""

    def __init__(self, trace: Trace, llm: LLM | None = None):
        self.trace = trace
        self.llm = llm or LLM(model=settings.chat_model)
        self.result: PrecedentResearchReport | DirectAnswer | None = None
        # Bounded revision rounds on the adverse analysis: enough to correct an
        # over-correction, capped so the agent can never be trapped in a loop.
        self._adverse_rounds = 0
        self._answer_check_done = False
        self._answer_synthesis_done = False

    # --- retrieval ------------------------------------------------------------

    def search_precedents(self, query: str, top_k: int = 8, favours: str | None = None) -> str:
        """Hybrid search: dense + BM25, fused by reciprocal rank, LLM-reranked.

        An automatic counter-search -- every query also returning judgments that
        favoured the opposing side -- was built here and then REVERTED after
        measurement. The hypothesis was that adverse coverage failed to generalise
        because it lived in output validation rather than retrieval. Held-out
        results refuted it: adverse recall fell 16.7% -> 11.7%, burial returned
        (0 -> 6), and every search cost twice the rerank calls.

        The diagnosis was simply wrong. Held-out retrieval recall is 100%: the
        agent already sees every damaging judgment. Adverse coverage fails
        downstream in synthesis, and supplying more documents diluted attention
        rather than focusing it. The code survives behind `counter_search_k`
        (default 0) so the negative result stays reproducible.
        """
        where = (
            f"outcome_favours = '{favours}'"
            if favours in {"claimant", "insurer", "mixed"}
            else None
        )
        docs = hybrid_search(query, k=top_k, where=where, rerank=True, llm=self.llm)
        self.trace.add(
            EventKind.RETRIEVAL,
            "search_precedents",
            detail=f"query={query!r} favours={favours} k={top_k}",
            payload={"query": query, "favours": favours},
            docs=docs,
        )
        if not docs:
            return "No judgments matched."

        main = "\n".join(
            f"[{d.doc_id}] {d.title}\n"
            f"   rerank={d.rerank_score}/10 (dense#{d.dense_rank} bm25#{d.sparse_rank} "
            f"fused#{d.fused_rank}) -- {d.why}"
            for d in docs
        )

        # Only when the caller has NOT already constrained the outcome -- an
        # explicit favours='insurer' search is already looking at one side.
        if favours is not None or settings.counter_search_k <= 0:
            return main
        seen = {d.doc_id for d in docs}
        counter = [
            d
            for d in hybrid_search(
                query,
                k=settings.counter_search_k,
                where="outcome_favours IN ('insurer', 'mixed')",
                rerank=True,
                llm=self.llm,
            )
            if d.doc_id not in seen
        ][: settings.counter_search_k]
        if not counter:
            return main

        self.trace.add(
            EventKind.RETRIEVAL,
            "search_precedents:opposing",
            detail=f"automatic counter-search -- {len(counter)} judgments favouring "
                   f"the opposing side",
            payload={"query": query, "automatic": True},
            docs=counter,
        )
        opposing = "\n".join(
            f"[{d.doc_id}] {d.title}\n   rerank={d.rerank_score}/10 -- {d.why}"
            for d in counter
        )
        return (
            f"{main}\n\n"
            f"--- ALSO RETRIEVED: authority favouring the OPPOSING side ---\n"
            f"(returned automatically on every search. Read these before deciding "
            f"whether they damage the client; if they do not, say so in `caveats`.)\n"
            f"{opposing}"
        )

    def filter_judgments(self, **kwargs) -> str:
        """Exhaustive structured filter -- scans all 56 cards, not a top-k sample."""
        clauses: list[str] = []
        f = {k: v for k, v in kwargs.items() if v is not None}

        for key, col in [
            ("case_type_contains", "case_type"),
            ("court_contains", "court"),
            ("statute_contains", "statutes_joined"),
            ("precedent_contains", "precedents_joined"),
            ("principle_contains", "principles_joined"),
        ]:
            if key in f:
                clauses.append(f"{col} LIKE '%{str(f[key]).lower().replace(chr(39), chr(39)*2)}%'")

        if "outcome_favours" in f:
            clauses.append(f"outcome_favours = '{f['outcome_favours']}'")
        for key, col in [
            ("is_commercial_vehicle", "is_commercial_vehicle"),
            ("has_licence_defect", "has_licence_defect"),
            ("is_death_claim", "is_death_claim"),
            ("contributory_negligence", "contributory_negligence"),
        ]:
            if key in f:
                clauses.append(f"{col} = {str(bool(f[key])).lower()}")
        if "min_multiplier" in f:
            clauses.append(f"multiplier >= {float(f['min_multiplier'])}")
        if "max_multiplier" in f:
            clauses.append(f"multiplier <= {float(f['max_multiplier'])}")

        where = " AND ".join(clauses) if clauses else "doc_id IS NOT NULL"
        rows = filter_cards(where)
        docs = [ScoredDoc(doc_id=r["doc_id"], title=r["title"]) for r in rows]
        self.trace.add(
            EventKind.FILTER,
            "filter_judgments",
            detail=f"WHERE {where}  ->  {len(rows)} of 56 judgments",
            payload={"where": where, "n_matched": len(rows),
                     "doc_ids": [r["doc_id"] for r in rows]},
            docs=docs,
        )
        if not rows:
            return f"No judgments matched: {where}"
        listing = "\n".join(
            f"[{r['doc_id']}] {r['title']} | {r['case_type']} | favours={r['outcome_favours']}"
            for r in rows
        )
        return f"{len(rows)} of 56 judgments matched ({where}):\n{listing}"

    def screen_corpus(self, criteria: str) -> str:
        """Show the model a compressed card for EVERY judgment and let it select.

        This is the recall backstop. It is affordable only because the corpus is
        56 documents; see the ADR for what replaces it at 5,000.
        """
        summaries = all_card_summaries()
        listing = "\n".join(
            f"[{s['doc_id']}] {s['title']} | {s['case_type']} | favours={s['favours']} | "
            f"{s['holding']}"
            for s in summaries
        )

        class _Pick(BaseModel):
            doc_id: str
            qualifies: bool
            reason: str

        class _Screen(BaseModel):
            picks: list[_Pick]

        res = self.llm.structured(
            f"Screen EVERY judgment below against this criterion. Do not skip any.\n\n"
            f"CRITERION: {criteria}\n\nJUDGMENTS ({len(summaries)} total):\n{listing}\n\n"
            f"Return one entry per doc_id with qualifies=true/false and a short reason.",
            _Screen,
        )
        hits = [p for p in res.picks if p.qualifies]
        docs = [ScoredDoc(doc_id=p.doc_id, why=p.reason) for p in hits]
        self.trace.add(
            EventKind.SCREEN,
            "screen_corpus",
            detail=f"criteria={criteria!r} -- screened {len(res.picks)}/{len(summaries)}, "
                   f"{len(hits)} qualified",
            payload={"criteria": criteria, "n_screened": len(res.picks),
                     "doc_ids": [p.doc_id for p in hits]},
            docs=docs,
        )
        return f"Screened all {len(summaries)} judgments. {len(hits)} qualify:\n" + "\n".join(
            f"[{p.doc_id}] {p.reason}" for p in hits
        )

    def read_judgment(self, doc_id: str, focus: str = "") -> str:
        cards, _ = get_tables()
        rows = cards.search().where(f"doc_id = '{doc_id}'").limit(1).to_list()
        if not rows:
            return f"No such judgment: {doc_id}"
        r = rows[0]

        passages = passages_for(
            focus or r["holding"] or r["title"], [doc_id], k=settings.read_passages_k
        ).get(doc_id, [])
        self.trace.add(
            EventKind.READ,
            "read_judgment",
            detail=f"{doc_id} focus={focus!r} -- {len(passages)} passages",
            # The passage TEXT is stored, not just a count. The brief asks to see
            # how the agent "arrived at its conclusions", and the answer is: from
            # these exact words. Without this the trace shows that a judgment was
            # opened but not what was read out of it, which is the part a lawyer
            # actually needs to audit a citation.
            payload={
                "doc_ids": [doc_id],
                "focus": focus,
                "title": r["title"],
                "court": r["court"],
                "holding": r["holding"],
                "ratio": r["ratio"],
                "passages": [
                    {"section": p["section"], "score": round(p["score"], 3),
                     "text": p["text"]}
                    for p in passages
                ],
            },
            docs=[ScoredDoc(doc_id=doc_id, title=r["title"])],
        )
        body = "\n\n".join(f"[{p['section']}] {p['text']}" for p in passages)
        return (
            f"[{doc_id}] {r['title']}\n"
            f"Court: {r['court']} | Decided: {r['decided_on']} | Type: {r['case_type']}\n"
            f"Outcome favours: {r['outcome_favours']}\n"
            f"HOLDING: {r['holding']}\nRATIO: {r['ratio']}\n"
            f"Statutes: {r['statutes_joined']}\nCites: {r['precedents_joined']}\n\n"
            f"RELEVANT PASSAGES:\n{body}"
        )

    def compute_quantum(self, **kwargs) -> str:
        args = QuantumArgs(**kwargs)
        res = compute_compensation(**args.model_dump())
        self.trace.add(
            EventKind.COMPUTE,
            "compute_quantum",
            detail=f"age={args.age} income={args.monthly_income} deps={args.dependents} "
                   f"-> Rs {res.total:,.0f}",
            payload={"inputs": args.model_dump(), "result": res.model_dump(mode="json")},
        )
        return res.summary()

    # --- terminal contracts ---------------------------------------------------

    def _retrieved_adverse_pool(self) -> list[dict]:
        """Every judgment this run retrieved whose outcome favours the opposing
        side, regardless of whether the report has dealt with it."""
        seen = set(self.trace.retrieved_doc_ids())
        if not seen:
            return []
        cards, _ = get_tables()
        rows = cards.search().where(_in_clause(sorted(seen))).limit(500).to_list()
        return [
            {"doc_id": r["doc_id"], "title": r["title"], "favours": r["outcome_favours"],
             "holding": (r["holding"] or "")[:180]}
            for r in rows
            if r["outcome_favours"] in ("insurer", "mixed")
        ]

    def _unaddressed_adverse(self, report: PrecedentResearchReport) -> list[dict]:
        """Judgments this run RETRIEVED that favour the opposing side and that the
        report neither cites nor explains away.

        Measured defect this closes: on a broad brief the agent surfaced 3 of 18
        available adverse authorities and stopped. It is not that it cannot find
        them -- it calls the outcome filter unprompted -- it satisfices, because
        three entries already *look* like a finished section. Since a partial
        adverse section is indistinguishable from a complete one, the reconciliation
        has to be enforced rather than requested.
        """
        addressed = (
            {a.doc_id for a in report.adverse}
            | {p.doc_id for p in report.supporting}
            | {d for c in report.caveats for d in _DOC_ID.findall(c)}
        )
        return [r for r in self._retrieved_adverse_pool() if r["doc_id"] not in addressed]

    def _report_problems(self, report: PrecedentResearchReport) -> list[dict]:
        """All adverse-analysis defects in one pass, so the fixes do not conflict."""
        problems: list[dict] = []

        # (1) Padding: entries whose own risk statement denies any risk.
        if settings.enable_adverse_gates and (bogus := self._self_negating_adverse(report)):
            problems.append({
                "summary": f"{len(bogus)} padded adverse entries",
                "doc_ids": bogus,
                "message": (
                    f"PADDED: {', '.join(bogus)} are listed as adverse, but their own "
                    "`risk_to_client` calls them irrelevant, unrelated, neutral or of no "
                    "bearing. A judgment that does no harm is not adverse authority — "
                    "move these to `caveats` with a one-line reason. Leave the genuinely "
                    "damaging ones where they are."
                ),
            })

        # (2) Coverage: adverse-pool judgments the report never engages.
        if settings.enable_adverse_gates and (gaps := self._unaddressed_adverse(report)):
            listing = "\n".join(
                f"  [{g['doc_id']}] favours={g['favours']} | {g['title'][:58]} | {g['holding']}"
                for g in gaps
            )
            problems.append({
                "summary": f"{len(gaps)} unaddressed adverse-pool judgments",
                "doc_ids": [g["doc_id"] for g in gaps],
                "message": (
                    f"UNADDRESSED: you retrieved these {len(gaps)} judgments whose outcome "
                    f"favours the opposing side, and the report neither cites nor explains "
                    f"them:\n{listing}\n"
                    "For each, either add it to `adverse` with a concrete statement of the "
                    "damage it does, add it to `supporting` if it actually helps, or name "
                    "its doc_id in `caveats` with why it does not apply."
                ),
            })

        # (3) Floor: an empty adverse section where damaging authority was available.
        #
        # Must be computed from the FULL retrieved adverse pool, not from what is
        # still unaddressed. Earlier version used the latter and was trivially
        # satisfiable: distinguish every adverse judgment in `caveats`, and the
        # unaddressed set empties, the pool reads as empty, and a report with zero
        # adverse entries sails through. Which is exactly what happened on the
        # main case brief.
        pool = self._retrieved_adverse_pool()
        if not report.adverse and pool:
            problems.append({
                "summary": "adverse section empty despite available adverse authority",
                "doc_ids": sorted(pool),
                "message": (
                    "EMPTY: the adverse section has no entries, but judgments favouring "
                    "the opposing side were retrieved. Reporting none is a stronger claim "
                    "than it looks — it tells the reader there is no exposure. If that is "
                    "genuinely true say so explicitly in `caveats`; otherwise name the "
                    "judgments that do damage."
                ),
            })

        # (4) Synthesis loss: judgments the reranker rated highly that the report
        # never mentions. Measured gap: retrieval recall 92.9% vs answer recall
        # 58.0% -- the agent finds the case, reads it, then drops it.
        #
        # Gated on the reranker's OWN high score rather than on everything
        # retrieved, which is what keeps this from wrecking precision: marginal
        # hits are not demanded, only judgments the system already called strong.
        # Requiring every retrieved document would trade one dimension for the
        # other; this asks the agent to reconcile with itself.
        if settings.enable_synthesis_check and (dropped := self._high_value_dropped(report)):
            problems.append({
                "summary": f"{len(dropped)} highly-ranked judgments cited nowhere",
                "doc_ids": [d for d, _ in dropped],
                "message": (
                    "DROPPED: your own reranker scored these judgments highly, but the "
                    "report does not cite them anywhere:\n"
                    + "\n".join(f"  [{d}] rerank {s:.0f}/10" for d, s in dropped)
                    + "\nFor each, either cite it (supporting or adverse), or name it in "
                    "`caveats` with why it does not belong. Do not cite one you have not "
                    "read -- use `read_judgment` first if needed."
                ),
            })

        # (5) Groundedness: a quote must exist in the judgment it is attributed to.
        if settings.enable_quote_verification and (fake := self._unverifiable_quotes(report)):
            problems.append({
                "summary": f"{len(fake)} quotes not found in the cited judgment",
                "doc_ids": [d for d, _ in fake],
                "message": (
                    "UNVERIFIABLE QUOTES: these do not appear in the judgment they are "
                    "attributed to:\n"
                    + "\n".join(f'  [{d}] "{q}"' for d, q in fake)
                    + "\nUse `read_judgment` and copy a line VERBATIM from the passages "
                    "it returns, or leave `quote` empty. Do not paraphrase into the "
                    "quote field -- a quotation that cannot be found is worse than none."
                ),
            })

        # (6) Calibration: every risk identical is not an assessment.
        levels = {a.risk_level for a in report.adverse}
        if len(report.adverse) >= 4 and len(levels) == 1:
            problems.append({
                "summary": f"all {len(report.adverse)} adverse entries rated "
                           f"'{levels.pop()}'",
                "doc_ids": [a.doc_id for a in report.adverse],
                "message": (
                    "UNCALIBRATED: every adverse entry carries the same risk level. Rank "
                    "them — which one could actually defeat the claim, and which merely "
                    "costs time to answer?"
                ),
            })

        return problems

    def _high_value_dropped(self, report) -> list[tuple[str, float]]:
        """Strongly-reranked judgments the report never mentions, worst first."""
        strong = self.trace.high_confidence_docs(settings.synthesis_threshold)
        if not strong:
            return []
        mentioned = self._mentioned_doc_ids(report)
        return sorted(
            ((d, s) for d, s in strong.items() if d not in mentioned),
            key=lambda x: -x[1],
        )[: settings.max_synthesis_prompts]

    @staticmethod
    def _mentioned_doc_ids(report) -> set[str]:
        """Every doc_id the report refers to, in any section."""
        ids: set[str] = set()
        for p in getattr(report, "supporting", []) or []:
            ids.add(p.doc_id)
        for a in getattr(report, "adverse", []) or []:
            ids.add(a.doc_id)
        ids |= set(getattr(report, "cited_doc_ids", []) or [])
        for c in getattr(report, "caveats", []) or []:
            ids |= set(_DOC_ID.findall(str(c)))
        if answer := getattr(report, "answer", ""):
            ids |= set(_DOC_ID.findall(answer))
        return ids

    def _unverifiable_quotes(self, report) -> list[tuple[str, str]]:
        """Quotes that do not actually appear in the judgment they are attributed to.

        This is the fix for the weakest measured criterion: `grounded_in_source`
        scored 0.44 -- the agent asserts things not traceable to the source.

        The check is deterministic. A quote either appears in the judgment or it
        does not; no judge, no rubric, no cost. That is the same principle as the
        deterministic ground truth in the eval set, and it is the most reliable
        thing in this project. Comparison is whitespace-normalised and
        case-insensitive so that reformatting is forgiven while fabrication is
        not.
        """
        entries = [
            (p.doc_id, p.quote)
            for p in list(getattr(report, "supporting", []) or [])
            + list(getattr(report, "adverse", []) or [])
            if getattr(p, "quote", "").strip()
        ]
        if not entries:
            return []

        _, chunks = get_tables()
        bad: list[tuple[str, str]] = []
        for doc_id, quote in entries:
            rows = chunks.search().where(f"doc_id = '{doc_id}'").limit(500).to_list()
            body = _norm(" ".join(r["text"] for r in rows))
            needle = _norm(quote)
            # Very short fragments match by accident; only check substantive ones.
            if len(needle) >= 25 and needle not in body:
                bad.append((doc_id, quote[:120]))
        return bad

    def _strip_unverifiable_quotes(self, report) -> list[str]:
        """Blank any quote not found in its judgment. Returns the doc_ids affected.

        Matches on the (doc_id, quote) PAIR, never on doc_id alone: two entries
        routinely cite the same judgment, and keying on the document removed a
        correctly-quoted entry alongside the fabricated one.
        """
        bad = {(d, _norm(q)) for d, q in self._unverifiable_quotes(report)}
        if not bad:
            return []
        hit: list[str] = []
        for entry in list(getattr(report, "supporting", []) or []) + list(
            getattr(report, "adverse", []) or []
        ):
            quote = getattr(entry, "quote", "") or ""
            if not quote.strip():
                continue
            # `_unverifiable_quotes` truncates its reported quote, so compare on
            # the stored prefix rather than requiring an exact match.
            if any(
                entry.doc_id == d and _norm(quote).startswith(q[:60])
                for d, q in bad
            ):
                entry.quote = ""
                hit.append(entry.doc_id)
        return hit

    def _self_negating_adverse(self, report: PrecedentResearchReport) -> list[str]:
        """Adverse entries whose own risk statement denies that they do harm.

        Second-order failure observed after the coverage gate landed: forced to
        account for 18 adverse-pool judgments, the agent listed 13 of them as
        `adverse` at uniform "low" risk with descriptions reading "irrelevant",
        "completely unrelated", "no bearing". It had moved the padding out of
        silence and into the adverse section -- option (a) is mechanically the
        cheapest way to satisfy the gate.

        Detecting it needs no judgement call: the entry contradicts itself. If the
        stated risk is that there is no risk, the judgment belongs in `caveats`.
        """
        return [
            a.doc_id
            for a in report.adverse
            if _DISCLAIMER.search(a.risk_to_client or "")
        ]

    def submit_research_report(self, **kwargs) -> str:
        report = PrecedentResearchReport(**kwargs)

        # ONE validation pass, not several. Separate gates pulled the agent in
        # opposite directions: a coverage gate alone produced 13 padded entries at
        # uniform "low" risk, and adding a padding gate on top emptied the section
        # completely. Both errors are the same mistake -- optimising one signal in
        # isolation -- so the checks are stated together and the agent is told
        # explicitly that both extremes are wrong.
        # Two rounds, not one. With a single round the agent learns that whatever
        # it submits second is accepted, and the cheapest way past a padding
        # complaint is to empty the section entirely -- which is the worse error.
        # Two rounds let it correct an over-correction; the cap still guarantees
        # termination.
        problems = self._report_problems(report)
        if problems and self._adverse_rounds < settings.max_adverse_revisions:
            self._adverse_rounds += 1
            self.trace.add(
                EventKind.BUDGET,
                "adverse_coverage_check",
                detail="; ".join(p["summary"] for p in problems),
                payload={"doc_ids": [d for p in problems for d in p["doc_ids"]]},
            )
            body = "\n\n".join(p["message"] for p in problems)
            return (
                "REPORT NOT ACCEPTED — revise the adverse analysis.\n\n"
                f"{body}\n\n"
                "Both extremes are wrong. An adverse section padded with harmless "
                "judgments misleads the reader, who reads its length as a measure of "
                "exposure. An empty one on a brief where damaging authority exists is "
                "worse. Report exactly the judgments that hurt the client, with "
                "differentiated risk levels, and dispose of the rest in `caveats`. "
                "Use `read_judgment` if you need to check one before deciding, then "
                "resubmit."
            )

        # Last line of defence on quotations. The revision gate asks the agent to
        # fix unverifiable quotes, but the budget is finite and after it is spent
        # the report is accepted -- which measurably let fabricated quotes through
        # (the gate fired 20 times across one evaluation and groundedness still did
        # not improve).
        #
        # So verification is not left to persuasion: any quote that cannot be
        # found in its judgment is REMOVED before the report ships. Losing a
        # quotation costs a little supporting detail; shipping an invented one
        # attributed to a named judgment is the single worst thing this system
        # could do.
        # The SANITISER always runs; only the revision GATE is switchable.
        #
        # As a gate this cost adverse recall 0.556 -> 0.056, because it consumed
        # revision rounds the adverse checks needed. As a sanitiser it consumes
        # nothing: an unverifiable quote is simply removed at acceptance. A live
        # run then still shipped a fabricated quote (1 of 10) because disabling
        # the gate had disabled the stripping too -- they are now independent.
        stripped = self._strip_unverifiable_quotes(report)
        if stripped:
            self.trace.add(
                EventKind.ERROR,
                "quotes_stripped",
                detail=f"removed {len(stripped)} quotes that could not be found in "
                       f"the cited judgment",
                payload={"doc_ids": stripped},
            )

        # Record whether anything adverse is still outstanding -- after the
        # revision budget is spent a report may ship with known gaps, and that
        # belongs in the trace rather than being quietly dropped.
        unreconciled = self._unaddressed_adverse(report)
        self.result = report
        self.trace.add(
            EventKind.ANSWER,
            "submit_research_report",
            detail=f"{len(report.supporting)} supporting, {len(report.adverse)} adverse"
                   + (f", {len(unreconciled)} still unaddressed" if unreconciled
                      else ", adverse reconciled"),
            payload={"doc_ids": [p.doc_id for p in report.supporting]
                                + [a.doc_id for a in report.adverse],
                     "unreconciled_adverse": [g["doc_id"] for g in unreconciled]},
        )
        return "Report submitted."

    def submit_answer(self, **kwargs) -> str:
        ans = DirectAnswer(**kwargs)

        # Close the escape hatch. Validation lived only on the research contract,
        # which made this one the cheapest way out: after two bounces on the
        # adverse analysis the agent switched to `submit_answer` and emitted a
        # brief with no adverse section at all. A gate that can be side-stepped by
        # changing the output shape is not a gate.
        #
        # The test is behavioural, not keyword-based: if this run retrieved
        # judgments favouring the opposing side and cited a substantial body of
        # authority, it was doing adversarial research whatever contract it now
        # reaches for.
        # Same synthesis check as the research contract. Applied here too because
        # any rule enforced on only one terminal tool is dodgeable by switching
        # tools -- which the agent has already been observed doing.
        dropped = self._high_value_dropped(ans) if settings.enable_synthesis_check else []
        if dropped and not self._answer_synthesis_done:
            self._answer_synthesis_done = True
            self.trace.add(
                EventKind.BUDGET,
                "synthesis_check",
                detail=f"{len(dropped)} highly-ranked judgments cited nowhere",
                payload={"doc_ids": [d for d, _ in dropped]},
            )
            return (
                "ANSWER NOT ACCEPTED — you dropped judgments your own search rated "
                "highly:\n"
                + "\n".join(f"  [{d}] rerank {s:.0f}/10" for d, s in dropped)
                + "\nCite them, or say explicitly why they do not answer the question. "
                "Then resubmit."
            )

        pool = self._retrieved_adverse_pool()
        if pool and len(ans.cited_doc_ids) >= 4 and not self._answer_check_done:
            self._answer_check_done = True
            self.trace.add(
                EventKind.BUDGET,
                "wrong_contract_check",
                detail=f"rejected: direct answer cites {len(ans.cited_doc_ids)} judgments "
                       f"with {len(pool)} adverse-pool judgments retrieved",
                payload={"doc_ids": [p["doc_id"] for p in pool]},
            )
            return (
                "ANSWER NOT ACCEPTED — wrong output contract.\n\n"
                f"You cited {len(ans.cited_doc_ids)} judgments and retrieved {len(pool)} "
                "whose outcome favours the opposing side. That is precedent research, not "
                "a factual lookup, and a direct answer has nowhere to report adverse "
                "authority — so using it here hides the client's exposure.\n\n"
                "Call `submit_research_report` instead, with supporting precedents, the "
                "authority that genuinely damages the client, and a strategy. Use "
                "`submit_answer` only for questions that do not involve weighing "
                "precedent for or against a position."
            )

        self.result = ans
        self.trace.add(
            EventKind.ANSWER,
            "submit_answer",
            detail=f"{len(ans.cited_doc_ids)} citations, confidence={ans.confidence}",
            payload={"doc_ids": ans.cited_doc_ids},
        )
        return "Answer submitted."

    # --- assembly -------------------------------------------------------------

    def build(self) -> list[StructuredTool]:
        def mk(fn: Callable, name: str, desc: str, schema: type[BaseModel]) -> StructuredTool:
            return StructuredTool.from_function(
                func=fn, name=name, description=desc, args_schema=schema
            )

        return [
            mk(
                self.search_precedents,
                "search_precedents",
                "Hybrid semantic + keyword search over the judgment corpus, LLM-reranked. "
                "Your default retrieval tool. Set favours='insurer' to hunt for adverse "
                "precedents that went against a claimant.",
                SearchArgs,
            ),
            mk(
                self.filter_judgments,
                "filter_judgments",
                "EXACT structured filter over all 56 judgments' extracted metadata. Returns "
                "every match, not a top-k sample. Use this for factual/enumerative questions "
                "('which judgments involve commercial vehicles?', 'which cite Section 149?') "
                "where completeness matters more than ranking.",
                FilterArgs,
            ),
            mk(
                self.screen_corpus,
                "screen_corpus",
                "Read a one-line summary of EVERY judgment in the corpus and select those "
                "meeting a criterion. Slower and more expensive than search, but exhaustive. "
                "Use when recall matters more than speed, or to double-check that a search "
                "did not miss anything.",
                ScreenArgs,
            ),
            mk(
                self.read_judgment,
                "read_judgment",
                "Open one judgment: metadata, holding, ratio, and the passages most relevant "
                "to `focus`. Use before citing a case, to get verbatim quotes and confirm it "
                "says what you think it says.",
                ReadArgs,
            ),
            mk(
                self.compute_quantum,
                "compute_quantum",
                "Deterministically compute motor-accident death compensation under the Sarla "
                "Verma / Pranay Sethi framework (multiplier, future prospects, dependency "
                "deduction, conventional heads). Use this instead of estimating figures "
                "yourself -- it returns the governing authority for each step.",
                QuantumArgs,
            ),
            mk(
                self.submit_research_report,
                "submit_research_report",
                "TERMINAL. Emit a full precedent-research report: supporting precedents, "
                "adverse precedents with honest risk assessment, and a strategy "
                "recommendation. Use for deep research tasks. Every doc_id must be one you "
                "actually retrieved and read.",
                PrecedentResearchReport,
            ),
            mk(
                self.submit_answer,
                "submit_answer",
                "TERMINAL. Emit a direct answer with citations. Use for general or factual "
                "questions that do not warrant a full research report.",
                DirectAnswer,
            ),
        ]


def tool_json(tools: list[StructuredTool]) -> str:
    return json.dumps([{"name": t.name, "description": t.description} for t in tools], indent=1)


def _norm(text: str) -> str:
    """Whitespace-normalised, lowercased -- forgives reformatting, not fabrication."""
    return " ".join(text.lower().split())
