"""Dimension 5: behavioural correctness.

The four graded dimensions score what the agent SAYS. This module scores what it
DOES, using the trace that the UI already renders. Standard agent-evaluation
practice separates trajectory, tool use and task completion from final-answer
quality, because an agent can produce a decent answer by an indefensible route --
and that route is what breaks on the next query.

Four measurements, each scoped to the queries where it means something:

  abstention  -- on a question the corpus cannot answer, the correct output cites
                 NOTHING. Silence is the right answer and must be scored as a
                 success, not as zero recall.
  contract    -- a case brief answered as prose has nowhere to report adverse
                 authority. Choosing the wrong output shape is a correctness
                 failure, not a formatting one. This was a real defect: the agent
                 escaped an adverse-coverage gate by switching contracts.
  trajectory  -- did it reach for the right instrument? Answering "which
                 judgments involve X?" from a top-k semantic search is wrong even
                 when the answer happens to be right, because it cannot be
                 complete by construction.
  cost        -- tool calls, LLM calls, tokens and wall-clock per query. The
                 adverse-coverage gates made broad briefs ~3x slower; without
                 this that tradeoff is invisible.
"""
from __future__ import annotations

from .metrics import mean
from .queries import BY_ID


def dimension_5_behaviour(runs: dict) -> dict:
    per_query: dict[str, dict] = {}

    for qid, run in runs.items():
        q = BY_ID.get(qid)
        if q is None:
            continue
        row: dict = {"ok": bool(run.get("ok")), "measures": sorted(q.measures())}

        if not run.get("ok"):
            per_query[qid] = row
            continue

        cited = run.get("cited") or []
        tools = run.get("tool_sequence") or []
        result_type = run.get("result_type")

        # --- abstention ------------------------------------------------------
        if "abstention" in q.measures():
            # Correct behaviour is to cite nothing at all.
            row["abstained"] = len(cited) == 0
            row["abstention_pass"] = len(cited) == 0

        # --- output contract -------------------------------------------------
        if "contract" in q.measures():
            expected = {"report": "PrecedentResearchReport", "answer": "DirectAnswer"}[
                q.expected_contract
            ]
            # A research report where a direct answer was expected is a lesser
            # error than the reverse: over-structuring is verbose, but answering a
            # brief as prose loses the adverse section entirely.
            row["expected_contract"] = expected
            row["actual_contract"] = result_type
            row["contract_ok"] = result_type == expected
            row["contract_severity"] = (
                "none" if result_type == expected
                else "high" if q.expected_contract == "report"
                else "low"
            )

        # --- trajectory ------------------------------------------------------
        if q.preferred_tools:
            used = set(tools)
            row["preferred_tools"] = sorted(q.preferred_tools)
            row["used_preferred_tool"] = bool(used & q.preferred_tools)
            row["tools_used"] = sorted(used)

        # --- cost ------------------------------------------------------------
        row |= {
            "n_tool_calls": len(tools),
            "llm_calls": run.get("llm_calls"),
            "tokens": run.get("tokens"),
            "elapsed_s": run.get("elapsed_s"),
        }
        per_query[qid] = row

    ok = [r for r in per_query.values() if r["ok"]]
    abst = [r for r in ok if "abstention_pass" in r]
    contract = [r for r in ok if "contract_ok" in r]
    traj = [r for r in ok if "used_preferred_tool" in r]

    return {
        "abstention_rate": mean(1.0 if r["abstention_pass"] else 0.0 for r in abst) if abst else None,
        "n_abstention_queries": len(abst),
        "contract_accuracy": mean(1.0 if r["contract_ok"] else 0.0 for r in contract)
        if contract else None,
        "contract_failures_high_severity": [
            q for q, r in per_query.items() if r.get("contract_severity") == "high"
        ],
        "trajectory_accuracy": mean(1.0 if r["used_preferred_tool"] else 0.0 for r in traj)
        if traj else None,
        "trajectory_failures": [
            q for q, r in per_query.items() if r.get("used_preferred_tool") is False
        ],
        "mean_tool_calls": mean(r["n_tool_calls"] for r in ok),
        "mean_tokens": mean(r["tokens"] for r in ok if r.get("tokens")),
        "mean_latency_s": mean(r["elapsed_s"] for r in ok if r.get("elapsed_s")),
        "max_latency_s": max((r["elapsed_s"] for r in ok if r.get("elapsed_s")), default=None),
        "run_success_rate": mean(1.0 if r["ok"] else 0.0 for r in per_query.values()),
        "per_query": per_query,
    }
