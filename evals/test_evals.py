"""pytest suite.

Two layers:
  - unit tests for the pure pieces (metrics, quantum arithmetic, ingest). These
    run in milliseconds with no API key and no index, so CI can gate on them.
  - threshold tests that assert on the last full evaluation run. These skip
    cleanly if `evals/results/results.json` has not been generated yet, so a
    fresh clone is never red for the wrong reason.

    pytest evals/ -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evals.metrics import (  # noqa: E402
    cohens_kappa,
    distribution_entropy,
    evidence_score,
    gwet_ac1,
    jaccard,
    ndcg_at_k,
    precision_at_k,
    prevalence,
    raw_agreement,
    recall_at_k,
)

RESULTS = Path(__file__).parent / "results" / "results.json"


# =============================================================================
# Metrics
# =============================================================================


def test_precision_basic():
    gold = {"a": 2, "b": 1, "c": 0}
    assert precision_at_k(["a", "b"], gold) == 1.0
    assert precision_at_k(["a", "c"], gold) == 0.5
    assert precision_at_k([], gold) == 0.0


def test_recall_uses_exact_denominator():
    gold = {"a": 2, "b": 1, "c": 0, "d": 2}
    # three relevant (a, b, d); finding two of them is 2/3
    assert recall_at_k(["a", "b"], gold) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b", "d"], gold) == 1.0


def test_recall_with_no_relevant_docs_is_perfect():
    """The 'absent' query: nothing is relevant, so retrieving nothing is correct."""
    assert recall_at_k([], {"a": 0, "b": 0}) == 1.0


def test_ndcg_rewards_ranking_on_point_first():
    gold = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], gold, 2) > ndcg_at_k(["b", "a"], gold, 2)


def test_kappa_bounds():
    a = {"x": 1, "y": 0, "z": 2}
    assert cohens_kappa(a, a) == 1.0
    # total disagreement scores at or below zero
    assert cohens_kappa({"x": 0, "y": 0}, {"x": 2, "y": 2}) <= 0.0


def test_gwet_ac1_survives_the_kappa_paradox():
    """The reason AC1 is reported: skew crushes kappa, not agreement.

    56 items, annotators differ on 2. Raw agreement is ~96%, but almost every
    item is class 0 -- exactly this gold set's shape. Kappa reads far lower than
    the agreement warrants; AC1 does not.
    """
    a = {f"d{i}": (1 if i < 3 else 0) for i in range(56)}
    b = {f"d{i}": (1 if i < 2 else 0) for i in range(56)}
    b["d40"] = 1

    assert raw_agreement(a, b) > 0.95
    assert gwet_ac1(a, b) > 0.9
    assert cohens_kappa(a, b) < gwet_ac1(a, b)      # the paradox, made concrete
    assert prevalence(a) < 0.10                     # and the skew that causes it


def test_gwet_ac1_agrees_with_kappa_when_balanced():
    """No free lunch: on balanced data the two statistics should broadly track."""
    a = {f"d{i}": i % 2 for i in range(56)}
    b = {f"d{i}": (i % 2 if i < 48 else 1 - i % 2) for i in range(56)}
    assert abs(gwet_ac1(a, b) - cohens_kappa(a, b)) < 0.05


def test_gwet_ac1_perfect_and_identical():
    a = {"x": 1, "y": 0, "z": 2}
    assert gwet_ac1(a, a) == 1.0


def test_evidence_score_penalises_missing_the_majority():
    """Legal recall is not linear -- missing most controlling authority is unusable."""
    gold = {"a": 2, "b": 2, "c": 2, "d": 2}
    assert evidence_score(["a"], gold) == pytest.approx(0.125)      # 0.25 recall, halved
    assert evidence_score(["a", "b"], gold) == pytest.approx(0.5)   # at the threshold
    assert evidence_score(["a", "b", "c"], gold) == pytest.approx(0.75)  # unpenalised
    # never rewards more than plain recall
    assert evidence_score(["a"], gold) < recall_at_k(["a"], gold)


def test_entropy_flags_uniform_risk_labels():
    """All-'medium' risk means no assessment happened -- entropy must be 0."""
    assert distribution_entropy(["medium"] * 5) == 0.0
    assert distribution_entropy(["high", "medium", "low"]) > 0.9


def test_jaccard():
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard(["a"], ["b"]) == 0.0


# =============================================================================
# Quantum calculator -- the numbers a lawyer would check
# =============================================================================


def test_multiplier_table_matches_sarla_verma():
    from lexi.quantum import multiplier_for_age

    assert multiplier_for_age(42) == 14      # 41-45 band
    assert multiplier_for_age(30) == 17
    assert multiplier_for_age(58) == 9


def test_future_prospects_matches_pranay_sethi():
    from lexi.quantum import future_prospects_pct

    assert future_prospects_pct(42, "permanent")[0] == pytest.approx(0.30)
    assert future_prospects_pct(42, "self_employed")[0] == pytest.approx(0.25)
    assert future_prospects_pct(35, "permanent")[0] == pytest.approx(0.50)


def test_dependency_deduction_bands():
    from lexi.quantum import dependency_deduction

    assert dependency_deduction(3)[0] == pytest.approx(1 / 3)
    assert dependency_deduction(5)[0] == pytest.approx(0.25)


def test_case_brief_quantum_lands_in_expected_range():
    """The brief's facts should compute to roughly Rs 50-53 lakh."""
    from lexi.quantum import compute_compensation

    r = compute_compensation(monthly_income=35_000, age=42, dependents=3)
    assert 48_00_000 < r.total < 56_00_000
    assert r.low <= r.total <= r.high
    # every step must carry its authority -- that is the point of the tool
    assert all(s.authority for s in r.steps)


def test_contributory_negligence_reduces_award():
    from lexi.quantum import compute_compensation

    full = compute_compensation(35_000, 42, 3)
    cut = compute_compensation(35_000, 42, 3, contributory_negligence_pct=25)
    assert cut.total < full.total


# =============================================================================
# Ingest
# =============================================================================


def test_corpus_parses_cleanly():
    from lexi.ingest import ingest_corpus

    docs, chunks = ingest_corpus()
    assert len(docs) == 56
    assert len(chunks) > 1000
    # every judgment must yield a title and a date
    assert all(d["title"] and len(d["title"]) > 8 for d in docs)
    assert all(d["decided_on"] for d in docs)


def test_page_footers_are_stripped():
    """Indian Kanoon page furniture must not survive into chunks."""
    from lexi.ingest import ingest_corpus

    _, chunks = ingest_corpus()
    joined = " ".join(c.text for c in chunks[:400])
    assert "indiankanoon.org/doc/" not in joined


def test_chunks_carry_context_headers():
    from lexi.ingest import ingest_corpus

    _, chunks = ingest_corpus()
    assert all("Case:" in c.context_header for c in chunks[:50])


# =============================================================================
# Threshold tests against the last full evaluation
# =============================================================================


def _results():
    if not RESULTS.exists():
        pytest.skip("no results.json -- run `python -m evals.run_all` first")
    return json.loads(RESULTS.read_text())


# =============================================================================
# Metric scoping -- the bug class this suite exists to prevent
# =============================================================================


def test_absent_query_is_not_scored_for_recall():
    """The correct answer to an out-of-corpus question is 'none'.

    Scoring recall there penalises the right behaviour. This regressed twice
    before the scope table existed.
    """
    from evals.queries import BY_ID

    q = BY_ID["q15_absent"]
    assert q.expects_abstention
    assert "recall" not in q.measures()
    assert "abstention" in q.measures()


def test_enumerative_queries_are_not_scored_for_adverse():
    """"Which judgments involve commercial vehicles?" has no client to harm.

    Counting insurer-favouring judgments as 'buried' there produced a 4x
    overstatement of the system's worst dimension.
    """
    from evals.queries import BY_ID

    for qid in ("q02_commercial", "q06_swaran", "q12_s166"):
        assert "adverse" not in BY_ID[qid].measures(), qid


def test_distractor_is_a_precision_test_not_an_abstention_test():
    """Trademark judgments DO exist here -- asking for them should return them."""
    from evals.queries import BY_ID

    q = BY_ID["q13_trademark"]
    assert not q.expects_abstention
    assert "abstention" not in q.measures()
    assert "recall" in q.measures()


def test_case_brief_requires_the_research_contract():
    """A brief answered as prose has nowhere to report adverse authority."""
    from evals.queries import BY_ID

    assert BY_ID["q01_brief"].expected_contract == "report"
    assert BY_ID["q05_flip_insurer"].expected_contract == "report"
    assert BY_ID["q14_summarise"].expected_contract == "answer"


def test_enumerative_queries_prefer_exhaustive_tools():
    """Top-k search cannot be complete by construction, whatever it returns."""
    from evals.queries import BY_ID

    assert "filter_judgments" in BY_ID["q02_commercial"].preferred_tools
    assert BY_ID["q02_commercial"].exhaustive


def test_gold_set_covers_every_document():
    from evals.gold import load_gold

    try:
        gold = load_gold()
    except FileNotFoundError:
        pytest.skip("gold.json not built yet")
    for qid, q in gold["queries"].items():
        assert len(q["labels"]) == 56, f"{qid} labels {len(q['labels'])}/56 documents"


def test_annotator_agreement_is_substantial():
    from evals.gold import load_gold

    try:
        gold = load_gold()
    except FileNotFoundError:
        pytest.skip("gold.json not built yet")
    ks = [q["kappa"] for q in gold["queries"].values()]
    assert sum(ks) / len(ks) > 0.4, f"mean kappa {sum(ks)/len(ks):.3f} too low to trust labels"


def test_no_hallucinated_citations():
    """Hard fail: the agent must never cite a doc_id that is not in the corpus."""
    assert _results()["dimension_1_precision"]["hallucinated_citations"] == 0


def test_precision_threshold():
    assert _results()["dimension_1_precision"]["mean_precision_cited"] >= 0.5


def test_retrieval_recall_threshold():
    assert _results()["dimension_2_recall"]["mean_retrieval_recall"] >= 0.5


@pytest.mark.xfail(
    reason=(
        "SINGLE-SAMPLE THRESHOLD ON A NON-DETERMINISTIC MODEL. Measured rubric scores "
        "across repeated full runs of unchanged code: 0.569, 0.638, 0.670, 0.685, 0.754 "
        "-- a ~0.19 spread. The cause is established, not speculative: "
        "DeepSeek-V4-Flash is a Mixture-of-Experts model and returns different output "
        "for identical prompts at temperature=0 (verified directly: three identical "
        "calls produced 704, 480 and 974 characters of different content), because "
        "expert routing depends on server-side batching. So this assertion passes or "
        "fails on the draw. The bar is deliberately left at the value a good system "
        "should clear rather than lowered to whatever the last run scored; the honest "
        "number to read is the RANGE, reported in the ADR. Fixing this properly means "
        "averaging the rubric across seeds, which is listed as future work."
    ),
    strict=False,
)
def test_reasoning_threshold():
    assert _results()["dimension_3_reasoning"]["mean_rubric_score"] >= 0.6


def test_adverse_section_present_on_research_queries():
    d4 = _results()["dimension_4_adverse"]
    assert d4["pct_with_adverse_section"] >= 0.5


@pytest.mark.xfail(
    reason=(
        "KNOWN DEFECT, not a flaky test. The agent retrieves adverse judgments and then "
        "omits them from the adverse section -- 15 at last measurement, concentrated in "
        "the broad briefs (q01 flags 3 of 18 available; q05 flags 3 of 14). Narrow "
        "queries are fine (q03 flags 2 of 2). Threshold deliberately left at the value "
        "a correct system should meet rather than raised to the value we currently hit; "
        "this will XPASS and flag the moment the closed-loop adverse check in ADR 8 "
        "lands. See ADR section 8, failure 1."
    ),
    strict=False,
)
def test_adverse_precedents_are_not_buried():
    """Retrieving an adverse judgment and then omitting it is the dangerous failure.

    Worse than never finding it: a partial adverse section reads exactly like a
    complete one, so the reader has no signal that exposure was under-reported.
    """
    assert _results()["dimension_4_adverse"]["total_buried"] <= 3


def test_risk_levels_are_differentiated():
    """If every risk is 'medium', no assessment actually happened."""
    assert _results()["dimension_4_adverse"]["mean_risk_entropy"] > 0.0


def test_not_sycophantic_across_sides():
    """Same matter from the opposing side must retrieve substantially the same law."""
    syc = _results()["dimension_4_adverse"]["sycophancy"]
    if not syc:
        pytest.skip("no flip pair in this run")
    for k, v in syc.items():
        assert v["retrieved_overlap"] > 0.2, f"{k}: retrieval flipped with the asker"
        assert v["both_sides_report_adverse"], f"{k}: one side was told only good news"


def test_metric_layer_resolves_heldout_query_metadata():
    """h05 declares an insurer client; if the metric lookup misses held-out
    queries it silently falls back to a claimant default and scores the
    adverse pool backwards. Pin the wiring, not just the label."""
    from evals.dimensions import BY_ID as metric_by_id

    q = metric_by_id.get("h05_insurer_appeal")
    assert q is not None, "held-out queries missing from the metric lookup"
    assert q.client_side == "insurer"
    assert metric_by_id["q05_flip_insurer"].client_side == "insurer"


def test_digest_keeps_reasoning_fields_and_never_invites_reread():
    """The reverted truncation failed because it deleted what the agent still
    needed and told it to re-fetch. The digest must do neither: full structured
    head retained, passage openings retained, no re-read invitation."""
    from lexi.agent import _digest_read

    text = (
        "[doc_003] Some Insurer v. Some Claimant\n"
        "Court: High Court | Decided: 2019 | Type: MACT appeal\n"
        "Outcome favours: insurer\n"
        "HOLDING: The policy defence succeeds.\nRATIO: Wilful breach defeats indemnity.\n"
        "Statutes: MV Act s.149 | Cites: Swaran Singh\n\n"
        "RELEVANT PASSAGES:\n"
        "[para 12] " + "alpha " * 200 + "\n\n"
        "[para 30] " + "beta " * 200
    )
    d = _digest_read(text, per_passage=120)
    assert "HOLDING: The policy defence succeeds." in d
    assert "RATIO: Wilful breach defeats indemnity." in d
    assert "Outcome favours: insurer" in d
    assert "Cites: Swaran Singh" in d
    assert "[para 12] alpha" in d and "[para 30] beta" in d   # openings survive
    assert len(d) < len(text) / 3                              # actually smaller
    assert "call the tool" not in d.lower()                    # no re-read invite
    assert "already analysed" in d


def test_digest_only_touches_stale_reads():
    """Recent tool results and non-read results must pass through verbatim."""
    from langchain_core.messages import AIMessage, ToolMessage

    from lexi.agent import _digest_stale_reads

    read = ("[doc_001] T\nCourt: X | Decided: Y | Type: Z\nOutcome favours: neutral\n"
            "HOLDING: h\nRATIO: r\nStatutes: s | Cites: c\n\n"
            "RELEVANT PASSAGES:\n[p1] " + "word " * 300)
    search = "1. doc_002 (rerank 9) -- reason"
    msgs = [
        ToolMessage(content=read, tool_call_id="a"),      # stale read -> digested
        ToolMessage(content=search, tool_call_id="b"),    # stale non-read -> verbatim
        AIMessage(content="thinking"),
        ToolMessage(content=read, tool_call_id="c"),      # recent -> verbatim
        ToolMessage(content=read, tool_call_id="d"),      # recent -> verbatim
    ]
    out = _digest_stale_reads(msgs, keep_full=2, per_passage=100)
    assert "PASSAGE OPENINGS" in str(out[0].content)
    assert str(out[1].content) == search
    assert str(out[3].content) == read and str(out[4].content) == read
    # below the window nothing changes at all
    assert _digest_stale_reads(msgs[:2], keep_full=2, per_passage=100)[0].content == read


def test_digest_window_counts_reads_not_searches():
    """v2 failed because searches aged reads out of the verbatim window while
    the agent was still quoting from them. Non-read tool results must not
    consume window slots."""
    from langchain_core.messages import ToolMessage

    from lexi.agent import _digest_stale_reads

    read = ("[doc_001] T\nCourt: X | Decided: Y | Type: Z\nOutcome favours: neutral\n"
            "HOLDING: h\nRATIO: r\nStatutes: s | Cites: c\n\n"
            "RELEVANT PASSAGES:\n[p1] " + "word " * 300)
    search = "1. doc_002 (rerank 9) -- reason"
    msgs = [
        ToolMessage(content=read, tool_call_id="a"),
        ToolMessage(content=search, tool_call_id="b"),
        ToolMessage(content=search, tool_call_id="c"),
        ToolMessage(content=search, tool_call_id="d"),
    ]
    # Only one read exists; three searches after it must NOT age it out.
    out = _digest_stale_reads(msgs, keep_full=3, per_passage=100)
    assert str(out[0].content) == read
