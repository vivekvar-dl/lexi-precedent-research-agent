"""Oracle tests: score a KNOWN-correct and a KNOWN-broken agent.

The problem this solves: every measurement bug in this project was found by
noticing an odd number and digging. That is luck, not method, and it gives no
basis for believing the remaining metrics are right.

So instead of inspecting metrics, we feed them fabricated runs whose correct
score is known by construction:

  PERFECT   -- cites exactly the gold-relevant judgments, flags exactly the
               adverse ones, picks the right output contract, abstains where the
               corpus cannot answer. Every metric must be at or near its ceiling.
  BROKEN    -- cites only irrelevant judgments, hides adverse authority, uses the
               wrong contract, invents citations. Every metric must be at or near
               its floor.
  ADVERSARIAL -- specific pathologies that previously slipped through: padding
               the adverse section with harmless entries, burying adverse
               authority, answering an unanswerable question.

A metric that scores PERFECT below 1.0, or BROKEN above 0.0, is measuring the
wrong thing -- regardless of whether its output looks plausible on real data.
That is exactly how "buried=63" and "adverse_recall=23.8%" went unnoticed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evals.behaviour import dimension_5_behaviour  # noqa: E402
from evals.dimensions import dimension_2_recall, dimension_4_adverse  # noqa: E402
from evals.metrics import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402


# =============================================================================
# Synthetic fixtures
# =============================================================================

DOCS = [f"doc_{i:03d}" for i in range(1, 21)]


def _gold_query(relevant: dict[str, int], sides: dict[str, str], **kw) -> dict:
    labels = {d: relevant.get(d, 0) for d in DOCS}
    return {
        "labels": labels,
        "sides": {d: sides.get(d, "claimant") for d in DOCS},
        "kappa": 1.0,
        "n_relevant": sum(1 for v in labels.values() if v >= 1),
        "n_on_point": sum(1 for v in labels.values() if v == 2),
        **kw,
    }


def _run(qid: str, cited: list[str], retrieved: list[str], result: dict,
         result_type: str, tools: list[str] | None = None) -> dict:
    return {
        "qid": qid, "seed": 0, "ok": True,
        "cited": cited, "retrieved": retrieved,
        "result": result, "result_type": result_type,
        "tool_sequence": tools or ["search_precedents"],
        "llm_calls": 5, "tokens": 1000, "elapsed_s": 10.0,
    }


def _report(supporting: list[str], adverse: list[tuple[str, str, str]],
            caveats: list[str] | None = None) -> dict:
    return {
        "supporting": [{"doc_id": d, "title": d, "principle": "p",
                        "fact_alignment": "f", "why_it_matters": "w",
                        "strength": "strong"} for d in supporting],
        "adverse": [{"doc_id": d, "title": d, "principle": "p",
                     "risk_to_client": risk, "risk_level": lvl,
                     "distinguishing_argument": "d"} for d, lvl, risk in adverse],
        "strategy": {"priority_arguments": ["a"], "compensation_range": "x", "risks": ["r"]},
        "caveats": caveats or [],
    }


# The adversarial query fixture: 4 relevant, of which 2 favour the insurer.
GOLD_ADV = {
    "queries": {
        "q01_brief": _gold_query(
            relevant={"doc_001": 2, "doc_002": 2, "doc_003": 2, "doc_004": 1},
            sides={"doc_003": "insurer", "doc_004": "insurer"},
        )
    }
}


# =============================================================================
# A perfect agent must score at the ceiling
# =============================================================================


def test_perfect_agent_scores_ceiling_on_recall():
    runs = {"q01_brief": _run(
        "q01_brief",
        cited=["doc_001", "doc_002", "doc_003", "doc_004"],
        retrieved=["doc_001", "doc_002", "doc_003", "doc_004"],
        result=_report(["doc_001", "doc_002"],
                       [("doc_003", "high", "kills the claim"),
                        ("doc_004", "medium", "forces an argument")]),
        result_type="PrecedentResearchReport",
    )}
    d2 = dimension_2_recall(runs, GOLD_ADV)
    assert d2["mean_retrieval_recall"] == 1.0
    assert d2["mean_answer_recall"] == 1.0
    assert d2["mean_synthesis_loss"] == 0.0
    assert d2["mean_evidence_score"] == 1.0


def test_perfect_agent_scores_ceiling_on_adverse():
    runs = {"q01_brief": _run(
        "q01_brief",
        cited=["doc_001", "doc_002", "doc_003", "doc_004"],
        retrieved=["doc_001", "doc_002", "doc_003", "doc_004"],
        result=_report(["doc_001", "doc_002"],
                       [("doc_003", "high", "kills the claim"),
                        ("doc_004", "low", "answerable but costs time")]),
        result_type="PrecedentResearchReport",
    )}
    d4 = dimension_4_adverse(runs, GOLD_ADV)
    assert d4["mean_adverse_recall"] == 1.0
    assert d4["mean_adverse_recall_strict"] == 1.0
    assert d4["total_buried"] == 0
    assert d4["total_miscast"] == 0
    assert d4["mean_risk_entropy"] > 0.0          # differentiated, not uniform


# =============================================================================
# A broken agent must score at the floor
# =============================================================================


def test_broken_agent_scores_floor():
    """Cites only irrelevant judgments and hides every adverse authority."""
    runs = {"q01_brief": _run(
        "q01_brief",
        cited=["doc_010", "doc_011"],                     # both graded 0
        retrieved=["doc_010", "doc_011", "doc_003", "doc_004"],  # saw adverse, hid it
        result=_report(["doc_010", "doc_011"], []),
        result_type="PrecedentResearchReport",
    )}
    d2 = dimension_2_recall(runs, GOLD_ADV)
    d4 = dimension_4_adverse(runs, GOLD_ADV)

    assert d2["mean_answer_recall"] == 0.0
    assert d4["mean_adverse_recall"] == 0.0
    # Retrieved two adverse judgments and reported neither -- the dangerous failure.
    assert d4["total_buried"] == 2


def test_buried_counts_only_what_was_actually_retrieved():
    """An agent cannot bury what it never saw -- that is a recall miss, not concealment.

    These are different defects with different fixes (retrieval vs synthesis) and
    the metric must not conflate them.
    """
    runs = {"q01_brief": _run(
        "q01_brief",
        cited=["doc_001"],
        retrieved=["doc_001"],                       # never saw doc_003 / doc_004
        result=_report(["doc_001"], []),
        result_type="PrecedentResearchReport",
    )}
    d4 = dimension_4_adverse(runs, GOLD_ADV)
    assert d4["total_buried"] == 0
    assert d4["mean_adverse_recall"] == 0.0          # still a miss, just not burial


def test_miscast_counts_opposing_wins_only():
    """A pay-and-recover judgment cited as supporting is legal judgement, not
    miscasting. `mixed` sits in BOTH sides' adverse pools, so the old
    definition flagged the same document as miscast for a claimant client and
    an insurer client simultaneously -- a logical impossibility. Only an
    outright opposing-side win presented as supporting counts; the mixed count
    is reported separately so the narrowing hides nothing."""
    gold = {"queries": {"q01_brief": _gold_query(
        relevant={"doc_001": 2, "doc_002": 2, "doc_003": 2},
        sides={"doc_002": "mixed", "doc_003": "insurer"},
    )}}
    runs = {"q01_brief": _run(
        "q01_brief",
        cited=["doc_001", "doc_002", "doc_003"],
        retrieved=["doc_001", "doc_002", "doc_003"],
        # Cites the outright insurer win doc_003 as SUPPORTING a claimant.
        result=_report(["doc_001", "doc_002", "doc_003"], []),
        result_type="PrecedentResearchReport",
    )}
    d4 = dimension_4_adverse(runs, gold)
    assert d4["total_miscast"] == 1
    pq = d4["per_query"]["q01_brief"]
    assert pq["miscast_as_supporting"] == ["doc_003"]
    assert pq["mixed_cited_as_supporting"] == ["doc_002"]


# =============================================================================
# Pathologies that previously slipped through
# =============================================================================


def test_padded_adverse_section_does_not_earn_full_credit():
    """Listing harmless judgments as adverse must not inflate the score.

    The observed failure: forced to account for adverse authority, the agent
    listed 13 entries at uniform "low" risk, most reading "irrelevant" or
    "unrelated". Entropy must expose that even though the count looks healthy.
    """
    padded = [("doc_003", "low", "kills the claim")] + [
        (f"doc_{i:03d}", "low", "irrelevant, unrelated to this matter")
        for i in range(10, 16)
    ]
    runs = {"q01_brief": _run(
        "q01_brief",
        cited=["doc_001", "doc_003"],
        retrieved=["doc_001", "doc_002", "doc_003", "doc_004"],
        result=_report(["doc_001"], padded),
        result_type="PrecedentResearchReport",
    )}
    d4 = dimension_4_adverse(runs, GOLD_ADV)
    # Uniform risk levels carry no information -- entropy must be zero.
    assert d4["mean_risk_entropy"] == 0.0


def test_distinguishing_in_caveats_counts_as_addressed():
    """Naming an adverse judgment and explaining why it does not bite is correct
    lawyering, not burial. Only silence should be penalised."""
    runs = {"q01_brief": _run(
        "q01_brief",
        cited=["doc_001", "doc_003"],
        retrieved=["doc_001", "doc_003", "doc_004"],
        result=_report(["doc_001"], [("doc_003", "high", "kills the claim")],
                       caveats=["doc_004 turns on a different statutory provision"]),
        result_type="PrecedentResearchReport",
    )}
    d4 = dimension_4_adverse(runs, GOLD_ADV)
    assert d4["total_buried"] == 0


# =============================================================================
# Scope: metrics must not fire where they are meaningless
# =============================================================================


def test_enumerative_query_is_excluded_from_adverse_scoring():
    """The 4x overstatement bug: 'which judgments involve X?' has no client.

    q02_commercial is a structured query. Even with insurer-favouring judgments
    in its relevant set and none reported as adverse, Dimension 4 must not score
    it at all.
    """
    gold = {"queries": {"q02_commercial": _gold_query(
        relevant={"doc_001": 2, "doc_003": 2},
        sides={"doc_003": "insurer"},
    )}}
    runs = {"q02_commercial": _run(
        "q02_commercial", cited=["doc_001", "doc_003"],
        retrieved=["doc_001", "doc_003"],
        result={"answer": "two judgments", "cited_doc_ids": ["doc_001", "doc_003"],
                "confidence": "high"},
        result_type="DirectAnswer", tools=["filter_judgments"],
    )}
    d4 = dimension_4_adverse(runs, gold)
    assert "q02_commercial" in d4["skipped_non_adversarial"]
    assert "q02_commercial" not in d4["per_query"]


def test_abstention_is_scored_as_success_not_zero_recall():
    """On an unanswerable question, citing nothing is the CORRECT answer."""
    runs = {"q15_absent": _run(
        "q15_absent", cited=[], retrieved=["doc_001", "doc_002"],
        result={"answer": "no such judgments in this corpus", "cited_doc_ids": [],
                "confidence": "high"},
        result_type="DirectAnswer",
    )}
    d5 = dimension_5_behaviour(runs)
    assert d5["abstention_rate"] == 1.0
    assert d5["per_query"]["q15_absent"]["abstention_pass"] is True


def test_fabricating_authority_on_an_unanswerable_question_fails():
    runs = {"q15_absent": _run(
        "q15_absent", cited=["doc_001"], retrieved=["doc_001"],
        result={"answer": "here is one", "cited_doc_ids": ["doc_001"],
                "confidence": "high"},
        result_type="DirectAnswer",
    )}
    d5 = dimension_5_behaviour(runs)
    assert d5["abstention_rate"] == 0.0


def test_wrong_output_contract_is_a_high_severity_failure():
    """A brief answered as prose has nowhere to report adverse authority.

    This was a real defect -- the agent used it to escape an adverse-coverage
    gate -- and nothing scored it until Dimension 5 existed.
    """
    runs = {"q01_brief": _run(
        "q01_brief", cited=["doc_001"], retrieved=["doc_001"],
        result={"answer": "prose", "cited_doc_ids": ["doc_001"], "confidence": "high"},
        result_type="DirectAnswer",
    )}
    d5 = dimension_5_behaviour(runs)
    assert d5["contract_accuracy"] == 0.0
    assert "q01_brief" in d5["contract_failures_high_severity"]


def test_wrong_instrument_on_an_enumerative_query_is_flagged():
    """Top-k search cannot be exhaustive by construction, whatever it returns."""
    runs = {"q02_commercial": _run(
        "q02_commercial", cited=["doc_001"], retrieved=["doc_001"],
        result={"answer": "x", "cited_doc_ids": ["doc_001"], "confidence": "high"},
        result_type="DirectAnswer", tools=["search_precedents"],
    )}
    d5 = dimension_5_behaviour(runs)
    assert d5["trajectory_accuracy"] == 0.0
    assert "q02_commercial" in d5["trajectory_failures"]


# =============================================================================
# Metric primitives against hand-computed values
# =============================================================================


def test_primitives_against_hand_computed_values():
    gold = {"a": 2, "b": 1, "c": 0, "d": 2}
    assert precision_at_k(["a", "b", "c"], gold) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b"], gold) == pytest.approx(2 / 3)
    # perfect ranking beats reversed ranking
    assert ndcg_at_k(["a", "d", "b"], gold, 3) > ndcg_at_k(["b", "a", "d"], gold, 3)
    assert ndcg_at_k(["a", "d", "b"], gold, 3) == pytest.approx(1.0)
