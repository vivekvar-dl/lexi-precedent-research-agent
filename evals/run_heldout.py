"""Score the agent on the held-out set and compare against the tuned set.

    python -m evals.run_heldout

Prints a side-by-side. A large drop from the main set to held-out means the
tuning overfitted the queries it was tuned on. That result gets reported as-is:
tuning against this set would spend the only unbiased measurement available.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .behaviour import dimension_5_behaviour
from .dimensions import dimension_1_precision, dimension_2_recall, dimension_4_adverse
from .gold import GOLD_PATH, build_gold
from .heldout import HELDOUT
from .runner import load_runs, run_all

HELDOUT_GOLD = Path(__file__).parent / "gold_heldout.json"
OUT = Path(__file__).parent / "results" / "heldout.json"


def _build_heldout_gold() -> dict:
    """Label the held-out queries using the same three-tier protocol."""
    import evals.gold as gold_mod

    main_path = gold_mod.GOLD_PATH
    gold_mod.GOLD_PATH = HELDOUT_GOLD          # keep the two label sets separate
    try:
        return build_gold(queries=HELDOUT)
    finally:
        gold_mod.GOLD_PATH = main_path


def main() -> None:
    print("=== 1/3  held-out gold labels ===")
    gold = _build_heldout_gold()

    print("\n=== 2/3  agent runs on held-out queries ===")
    run_all(queries=HELDOUT, seed=0)
    runs = {q.qid: r for q, r in
            ((q, load_runs(0).get(q.qid)) for q in HELDOUT) if r}
    # load_runs only knows the main set; read held-out runs directly
    from .runner import run_path
    runs = {}
    for q in HELDOUT:
        p = run_path(q.qid, 0)
        if p.exists():
            runs[q.qid] = json.loads(p.read_text())

    print("\n=== 3/3  scoring ===")
    d1 = dimension_1_precision(runs, gold, check_faithfulness=False)
    d2 = dimension_2_recall(runs, gold)
    d4 = dimension_4_adverse(runs, gold)
    d5 = dimension_5_behaviour(runs)

    res = {
        "dimension_1_precision": d1,
        "dimension_2_recall": d2,
        "dimension_4_adverse": d4,
        "dimension_5_behaviour": d5,
        "gold": {k: {"n_relevant": v["n_relevant"], "kappa": v["kappa"],
                     "gwet_ac1": v.get("gwet_ac1")} for k, v in gold["queries"].items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    # --- side by side against the tuned set ---------------------------------
    main_res_path = Path(__file__).parent / "results" / "results.json"
    print("\n" + "=" * 70)
    print("GENERALISATION: tuned query set vs held-out")
    print("=" * 70)
    rows = [
        ("precision (cited)", "dimension_1_precision", "mean_precision_cited"),
        ("nDCG@10", "dimension_1_precision", "mean_ndcg@10"),
        ("retrieval recall", "dimension_2_recall", "mean_retrieval_recall"),
        ("answer recall", "dimension_2_recall", "mean_answer_recall"),
        ("evidence score", "dimension_2_recall", "mean_evidence_score"),
        ("adverse recall", "dimension_4_adverse", "mean_adverse_recall"),
        ("buried", "dimension_4_adverse", "total_buried"),
        ("contract accuracy", "dimension_5_behaviour", "contract_accuracy"),
        ("trajectory accuracy", "dimension_5_behaviour", "trajectory_accuracy"),
        ("abstention", "dimension_5_behaviour", "abstention_rate"),
    ]
    main = json.loads(main_res_path.read_text()) if main_res_path.exists() else {}
    print(f"{'metric':<22} {'tuned':>10} {'held-out':>10}  delta")
    print("-" * 70)
    for label, dim, key in rows:
        m = (main.get(dim) or {}).get(key)
        h = (res.get(dim) or {}).get(key)
        fm = f"{m:.3f}" if isinstance(m, float) else ("-" if m is None else str(m))
        fh = f"{h:.3f}" if isinstance(h, float) else ("-" if h is None else str(h))
        delta = ""
        if isinstance(m, float) and isinstance(h, float):
            d = h - m
            delta = f"{d:+.3f}" + ("  <-- drop" if d < -0.15 else "")
        print(f"{label:<22} {fm:>10} {fh:>10}  {delta}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
