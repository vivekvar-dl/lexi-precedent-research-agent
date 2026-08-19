"""Ablation: measure each behavioural fix in isolation.

    python -m evals.ablate            # ~6 min per arm
    python -m evals.ablate --full     # all 15 queries instead of the subset

Why this exists. Four fixes were once changed together and evaluated in one
40-minute run. Several metrics moved and NONE of the movement could be
attributed -- full cost, zero signal. Worse, the held-out set then declined, and
there was no way to tell which change caused it.

So: one variable per arm, on a subset small enough to iterate on. The subset is
chosen to exercise the specific failure modes each fix targets, and it mixes
tuned and held-out queries so overfitting shows up in the same table:

  q01_brief        adversarial research -- adverse gates, synthesis, quotes
  q02_commercial   enumerative -- precision, over-citation
  q15_absent       abstention -- must cite nothing
  h03_owner_dilig. HELD-OUT adversarial -- does the fix generalise?
  h04_interest     HELD-OUT enumerative

Held-out arms are read-only evidence. Tuning against them would spend the only
unbiased measurement available.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .behaviour import dimension_5_behaviour
from .dimensions import dimension_1_precision, dimension_2_recall, dimension_4_adverse
from .gold import GOLD_PATH, load_gold
from .heldout import BY_ID as HELDOUT_BY_ID
from .queries import BY_ID as MAIN_BY_ID
from .runner import RUNS_DIR, run_agent

# Both sides of the flip are essential here: the change only bites when the
# client is the insurer, and q05 is the only tuned query where that is true.
# h05 is its held-out counterpart.
# Long, tool-heavy queries -- compaction only bites where the history grows, and
# these are the runs that cost 300-500K tokens. Two are held-out.
SUBSET = ["q01_brief", "q04_adverse_licence", "q11_fake_licence",
          "h03_owner_diligence", "h05_insurer_appeal"]
HELDOUT_GOLD = Path(__file__).parent / "gold_heldout.json"
OUT = Path(__file__).parent / "results" / "ablation.json"

# name -> the settings overrides that define this arm
ARMS: dict[str, dict] = {
    # Shipping configuration with the full message history resent every turn.
    "full_history": {"enable_history_compaction": False},
    # Identical, but old tool results are compacted. Cost should fall sharply;
    # quality must not move, or the change reverts like the other four.
    # RESULT: reverted. The agent re-read what compaction deleted (41 dup reads
    # on one query, tool calls x4, tokens UP, 2 recursion-limit crashes).
    "compacted":    {"enable_history_compaction": True},
    # The reverted configuration plus the reliability fixes the token ablation
    # exposed: agent-path retries and a 600s timeout (both in code, not
    # overrides). Quality must match full_history; only failures may move.
    "reliability":  {"enable_history_compaction": False},
    # The q05 failure isolated: with an insurer client the fixed rule hunts the
    # wrong pool -- every burial in the last full run was an outright claimant
    # win the agent had itself retrieved. side_on resolves the side once at
    # triage; side_off is its same-batch control. The subset carries both
    # sides, so a claimant-side burial regression (which killed the per-turn
    # variant) shows up in the same table.
    # RESULT: kept. adverse 0.379 -> 0.474, buried 4 -> 1 (claimant side 0),
    # h05 held-out 0.4 -> 0.9 broad / 1.0 strict, tokens -17%.
    "side_off": {"enable_triage_side_resolution": False},
    "side_on":  {"enable_triage_side_resolution": True},
    # The information-preserving successor to the reverted truncation: stale
    # reads keep their structured head and passage openings. Runs on the
    # token-heavy default subset; the verdict needs the run files too --
    # duplicate reads must stay at control levels or the re-read pathology is
    # back in a subtler form.
    # RESULT: reverted. tokens -21.6% (target 40-50), dup reads 2 -> 8,
    # heldout precision 0.844 -> 0.733. Better than truncation, below the bar.
    "digest_off": {"enable_digest_compaction": False},
    "digest_on":  {"enable_digest_compaction": True},
    # v3: window counts reads (code change), keeps the last 5 reads verbatim
    # and 600 chars per stale passage -- each parameter aimed at v2's measured
    # failure (reads digested while in active use). Same gate as v2.
    # RESULT: reverted -- quality finally held (the window diagnosis was
    # right), but tokens went UP 6.4%: the safe end of the dial saves nothing.
    # Three settings measured; the family is closed.
    "digest3_off": {"enable_digest_compaction": False},
    "digest3_on":  {"enable_digest_compaction": True,
                    "digest_keep_full": 5, "digest_passage_chars": 600},
    # Lever 2: trim reads at the SOURCE. Each read is ~2.2K tokens x resent on
    # every later request; the 4th-ranked passage is the marginal content.
    # Unlike compaction there is no removal trap -- the agent never had it.
    # Gate: tokens fall, recall and heldout rows hold.
    # STATUS: never run to verdict -- stopped mid-control at submission time.
    # read_passages_k defaults to 4, so shipping behaviour is unchanged.
    "readk_off": {"read_passages_k": 4},
    "readk3":    {"read_passages_k": 3},
    # Rule 3 names `filter_judgments(outcome_favours=<opponent>)` explicitly.
    # Diagnosed on q05: with an insurer client the agent filtered on
    # outcome_favours='insurer' -- its own supporting side -- and never queried
    # the claimant side at all. One correct call returns 16/16 of that pool, so
    # the gap is behavioural. Gate: strict adverse recall rises on the
    # insurer-side queries AND claimant-side burial stays at 0.
    # RESULT: reverted. The agent did start querying the right pools (verified
    # in traces) but burial went 2 -> 14 and tokens +60%; q01, a claimant-side
    # control, halved its strict recall. Retrieval was never the constraint.
    "advfilter_off": {"enable_explicit_adverse_filter": False},
    "advfilter_on":  {"enable_explicit_adverse_filter": True},
    # Triage treats "enumerate X, and give a detail per item" as simple. Gate:
    # contract accuracy rises AND recall on those queries does not fall -- the
    # risk is that a 4-step budget under-researches an enumerative question.
    # RESULT: reverted. Tool calls flat (17.8 -> 18.4), tokens UP 18%, and the
    # one contract that flipped is a query that flips unprompted between runs.
    # Triage controls the budget; the contract is a separate tool choice.
    "enum_off": {"enable_enumeration_is_simple": False},
    "enum_on":  {"enable_enumeration_is_simple": True},
}

# Each fix is measured on the queries it targets.
SUBSET_ADVERSE = ["q05_flip_insurer", "h05_insurer_appeal", "q01_brief",
                  "q04_adverse_licence", "h03_owner_diligence"]
SUBSET_ENUM = ["q06_swaran", "q09_multiplier", "q10_future_prospects",
               "q12_s166", "q02_commercial"]

# Side arms need the insurer-side queries; the default subset has only one.
SUBSET_SIDE = ["q01_brief", "q04_adverse_licence", "q05_flip_insurer",
               "h03_owner_diligence", "h05_insurer_appeal"]
ARM_SUBSETS = {
    "side_off": SUBSET_SIDE, "side_on": SUBSET_SIDE,
    "advfilter_off": SUBSET_ADVERSE, "advfilter_on": SUBSET_ADVERSE,
    "enum_off": SUBSET_ENUM, "enum_on": SUBSET_ENUM,
}


def _queries(full: bool):
    if full:
        return list(MAIN_BY_ID.values()) + list(HELDOUT_BY_ID.values())
    return [MAIN_BY_ID.get(q) or HELDOUT_BY_ID[q] for q in SUBSET]


def _merged_gold() -> dict:
    """Main + held-out labels in one dict, so a mixed subset can be scored."""
    gold = load_gold()
    if HELDOUT_GOLD.exists():
        gold["queries"] |= json.loads(HELDOUT_GOLD.read_text())["queries"]
    return gold


def run_arm(name: str, overrides: dict, queries) -> dict:
    """Run one configuration. Runs are cached per arm so a re-run is free."""
    from lexi.config import settings

    previous = {k: getattr(settings, k) for k in overrides}
    for k, v in overrides.items():
        setattr(settings, k, v)

    arm_dir = RUNS_DIR.parent / f"runs_ablate_{name.replace('+', 'plus_')}"
    original, runs = RUNS_DIR, {}
    try:
        import evals.runner as R

        R.RUNS_DIR = arm_dir
        # Queries run concurrently, same as the main runner. Looping them
        # sequentially here made an arm ~3x slower for no benefit -- almost all
        # of the time is spent waiting on the provider.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=settings.eval_workers) as pool:
            futures = {pool.submit(run_agent, q, 0): q for q in queries}
            for fut in as_completed(futures):
                q = futures[fut]
                rec = fut.result()
                runs[q.qid] = rec
                status = rec["result_type"] if rec.get("ok") else "FAILED"
                print(f"    {q.qid:<22} {status:<26} "
                      f"cited={len(rec.get('cited', []))}", flush=True)
    finally:
        import evals.runner as R

        R.RUNS_DIR = original
        for k, v in previous.items():
            setattr(settings, k, v)
    return runs


def score(runs: dict, gold: dict) -> dict:
    d1 = dimension_1_precision(runs, gold, check_faithfulness=False)
    d2 = dimension_2_recall(runs, gold)
    d4 = dimension_4_adverse(runs, gold)
    d5 = dimension_5_behaviour(runs)
    held = {q for q in runs if q.startswith("h")}
    return {
        "precision": d1["mean_precision_cited"],
        "ndcg": d1["mean_ndcg@10"],
        "retrieval_recall": d2["mean_retrieval_recall"],
        "answer_recall": d2["mean_answer_recall"],
        "synthesis_loss": d2["mean_synthesis_loss"],
        "adverse_recall": d4["mean_adverse_recall"],
        "adverse_strict": d4.get("mean_adverse_recall_strict"),
        "buried": d4["total_buried"],
        "contract": d5["contract_accuracy"],
        "abstention": d5["abstention_rate"],
        "tool_calls": d5["mean_tool_calls"],
        "tokens": d5["mean_tokens"],
        "latency_s": d5["mean_latency_s"],
        # Held-out only, so generalisation is visible in the same table.
        "heldout_answer_recall": _mean(
            d2["per_query"][q]["answer_recall"] for q in held if q in d2["per_query"]
        ),
        "heldout_precision": _mean(
            d1["per_query"][q]["precision_cited"] for q in held if q in d1["per_query"]
        ),
    }


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    full = "--full" in sys.argv
    only = [a for a in sys.argv[1:] if a in ARMS]
    arms = {k: v for k, v in ARMS.items() if not only or k in only}
    queries = _queries(full)
    gold = _merged_gold()

    print(f"ablation: {len(arms)} arms x {len(queries)} queries "
          f"({'full' if full else 'subset'})\n")
    results = {}
    for name, overrides in arms.items():
        qs = queries
        if not full and name in ARM_SUBSETS:
            qs = [MAIN_BY_ID.get(q) or HELDOUT_BY_ID[q] for q in ARM_SUBSETS[name]]
        print(f"  [{name}]  {overrides}  ({len(qs)} queries)", flush=True)
        results[name] = score(run_arm(name, overrides, qs), gold)
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))

    keys = ["tokens", "latency_s", "tool_calls", "precision", "ndcg",
            "answer_recall", "adverse_recall", "buried",
            "heldout_precision", "heldout_answer_recall"]
    width = max(len(k) for k in keys) + 2
    print("=" * (width + 12 * len(results)))
    print("ABLATION — each fix measured alone".center(width + 12 * len(results)))
    print("=" * (width + 12 * len(results)))
    print(f"{'metric':<{width}}" + "".join(f"{n:>12}" for n in results))
    print("-" * (width + 12 * len(results)))
    for k in keys:
        row = f"{k:<{width}}"
        for name in results:
            v = results[name].get(k)
            row += f"{v:>12.3f}" if isinstance(v, float) else f"{str(v):>12}"
        print(row)
    print(f"\nwrote {OUT}")
    print("\nRead each column against `baseline`. A fix that improves the tuned "
          "metrics\nbut not the heldout_* rows is fitted to the query set, not working.")


if __name__ == "__main__":
    main()
