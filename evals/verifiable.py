"""Deterministic ground truth -- used to score the ANNOTATORS, not the agent.

The gold set is LLM-labelled, which raises the obvious objection: if the answer
key itself is model output, why trust anything measured against it?

The standard answer in the literature is to validate judges against a
human-labelled sample. We have no lawyer on hand. But for a subset of queries we
can do something stronger than human labelling: COMPUTE the answer.

"Which judgments cite Swaran Singh?" is not a matter of opinion. Either the
string appears in the judgment or it does not. `grep` is the oracle, and it is
not an LLM.

So this module builds an independently-computed answer key over that subset,
then scores annotator A, annotator B and the adjudicated labels against it. The
resulting accuracy is a MEASURED property of the annotation process, reported
alongside the gold set. It does not prove the subjective labels are right, but it
bounds how much to trust them: annotators that score 0.95 on computable queries
have earned more credence on the contestable ones than annotators scoring 0.6.

Deliberately conservative: only queries whose relevance test is genuinely
mechanical appear here. Anything requiring legal judgement is excluded rather
than approximated.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexi.ingest import ingest_corpus  # noqa: E402

# qid -> (regex over full judgment text, grade to assign on a match)
#
# Each pattern is the literal mechanical test for that query. Where a query asks
# about a doctrine that has a canonical name, we match the name, not a paraphrase.
VERIFIABLE: dict[str, tuple[str, int]] = {
    # "Which judgments cite National Insurance v. Swaran Singh?"
    "q06_swaran": (r"(?i)swaran\s+sing", 2),
    # "Which judgments decide claims under Section 166 of the MV Act?"
    "q12_s166": (r"(?i)(section\s*166|s\.?\s*166|u/s\s*166)", 2),
    # "Are there any trademark or IP judgments in this corpus?"
    "q13_trademark": (r"(?i)(trade\s?marks?\s+act|passing\s+off|infring\w+\s+of\s+.{0,20}trade\s?mark)", 2),
    # "Which judgments apply future prospects under Pranay Sethi?"
    "q10_future_prospects": (r"(?i)(future\s+prospect|pranay\s+sethi)", 2),
    # "Find precedents about maritime salvage / admiralty" -- expected: none.
    "q15_absent": (r"(?i)(admiralty|maritime\s+salvage|shipping\s+collision)", 2),
    # --- scoring-only additions ------------------------------------------------
    # These three raise verifiable coverage from 6 queries to 9 (40% -> 60% of the
    # label set). They are deliberately NOT in DETERMINISTIC_OVERRIDE: the regex
    # finds judgments that MENTION the doctrine, which is a strong proxy but not
    # the same as judgments that TURN on it. Good enough to score an annotator
    # against; not good enough to overrule one.
    "q03_contrib": (r"(?i)contributory\s+negligen", 2),
    "q07_pay_recover": (r"(?i)(pay\s+and\s+recover|pay\s*&\s*recover)", 2),
    "q11_fake_licence": (
        r"(?i)(fake\s+licen[cs]e|forged\s+licen[cs]e|invalid\s+licen[cs]e|"
        r"expired\s+licen[cs]e|without\s+(a\s+)?valid\s+(driving\s+)?licen[cs]e)",
        2,
    ),
}

# Queries whose answer is fixed by construction rather than by matching.
CONSTRUCTED: dict[str, dict[str, int]] = {
    # "Summarise doc_003" -- only doc_003 can be relevant.
    "q14_summarise": {"doc_003": 2},
}


def compute_truth() -> dict[str, dict[str, int]]:
    """Build the deterministic key by scanning raw judgment text."""
    docs, _ = ingest_corpus()
    out: dict[str, dict[str, int]] = {}

    for qid, (pattern, grade) in VERIFIABLE.items():
        rx = re.compile(pattern)
        out[qid] = {d["doc_id"]: (grade if rx.search(d["text"]) else 0) for d in docs}

    for qid, fixed in CONSTRUCTED.items():
        out[qid] = {d["doc_id"]: fixed.get(d["doc_id"], 0) for d in docs}

    return out


def _score(pred: dict[str, int], truth: dict[str, int]) -> dict:
    """Binary agreement of one label set against the computed key."""
    keys = sorted(set(pred) & set(truth))
    if not keys:
        return {}
    tp = sum(1 for k in keys if pred[k] >= 1 and truth[k] >= 1)
    fp = sum(1 for k in keys if pred[k] >= 1 and truth[k] == 0)
    fn = sum(1 for k in keys if pred[k] == 0 and truth[k] >= 1)
    tn = sum(1 for k in keys if pred[k] == 0 and truth[k] == 0)
    prec = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
    rec = tp / (tp + fn) if tp + fn else 1.0
    return {
        "accuracy": (tp + tn) / len(keys),
        "precision": prec,
        "recall": rec,
        "f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_true_relevant": tp + fn,
        "false_positives": sorted(k for k in keys if pred[k] >= 1 and truth[k] == 0),
        "false_negatives": sorted(k for k in keys if pred[k] == 0 and truth[k] >= 1),
    }


def validate_annotators(gold: dict) -> dict:
    """Score annotator A, annotator B and the adjudicated labels against truth."""
    truth = compute_truth()
    per_query, agg = {}, {"annotator_a": [], "annotator_b": [], "adjudicated": []}

    for qid, t in truth.items():
        g = gold["queries"].get(qid)
        if not g:
            continue
        row = {
            "n_relevant_by_truth": sum(1 for v in t.values() if v >= 1),
            "annotator_a": _score(g["annotator_a"], t),
            "annotator_b": _score(g["annotator_b"], t),
            "adjudicated": _score(g["labels"], t),
            "kappa": g["kappa"],
        }
        per_query[qid] = row
        for k in agg:
            if row[k]:
                agg[k].append(row[k]["accuracy"])

    return {
        "per_query": per_query,
        "mean_accuracy": {k: (sum(v) / len(v) if v else None) for k, v in agg.items()},
        "n_queries_verified": len(per_query),
        "n_labels_verified": len(per_query) * 56,
    }


def main() -> None:
    from .gold import load_gold

    gold = load_gold()
    res = validate_annotators(gold)

    print("=" * 78)
    print("ANNOTATOR VALIDATION against deterministically-computed ground truth")
    print("=" * 78)
    print(f"{res['n_queries_verified']} queries x 56 judgments = "
          f"{res['n_labels_verified']} independently verifiable labels\n")

    print(f"{'query':<22} {'true rel':>8} {'A acc':>7} {'B acc':>7} {'final':>7} {'kappa':>7}")
    print("-" * 78)
    for qid, r in res["per_query"].items():
        print(
            f"{qid:<22} {r['n_relevant_by_truth']:>8} "
            f"{r['annotator_a'].get('accuracy', 0):>7.3f} "
            f"{r['annotator_b'].get('accuracy', 0):>7.3f} "
            f"{r['adjudicated'].get('accuracy', 0):>7.3f} "
            f"{r['kappa']:>7.3f}"
        )
    print("-" * 78)
    m = res["mean_accuracy"]
    print(f"{'MEAN':<22} {'':>8} {m['annotator_a'] or 0:>7.3f} "
          f"{m['annotator_b'] or 0:>7.3f} {m['adjudicated'] or 0:>7.3f}")

    print("\nWhere the adjudicated key disagrees with computed truth:")
    for qid, r in res["per_query"].items():
        adj = r["adjudicated"]
        if adj.get("false_positives") or adj.get("false_negatives"):
            print(f"  {qid}: FP={adj['false_positives'][:8]} FN={adj['false_negatives'][:8]}")


if __name__ == "__main__":
    main()


# =============================================================================
# Held-out verification
# =============================================================================
#
# The main gold set is anchored by 504 deterministically-computed labels. The
# held-out set had none -- and it is the set every generalisation claim rests on.
#
# That gap mattered: h02 was labelled as having ONE relevant judgment while a
# literal text search finds 34 discussing permits, and the two annotators agreed
# almost perfectly on that (AC1 0.964). Agreement is not accuracy, and a
# generalisation gap measured against an under-labelled key is partly a
# measurement artefact rather than an agent failure.

HELDOUT_VERIFIABLE: dict[str, tuple[str, int]] = {
    "h02_permit_breach": (
        r"(?i)(permit|route condition|fitness certificate|registration certificate)", 2
    ),
    "h04_interest_rate": (r"(?i)(interest\s+(?:@|at|of)\s*\d|per\s+annum|p\.a\.)", 2),
    "h06_absent_tax": (
        r"(?i)(goods and services tax|\bGST\b|input tax credit|VAT assessment)", 2
    ),
}


def compute_heldout_truth() -> dict[str, dict[str, int]]:
    docs, _ = ingest_corpus()
    out: dict[str, dict[str, int]] = {}
    for qid, (pattern, grade) in HELDOUT_VERIFIABLE.items():
        rx = re.compile(pattern)
        out[qid] = {d["doc_id"]: (grade if rx.search(d["text"]) else 0) for d in docs}
    return out


def validate_heldout() -> dict:
    """Score the held-out gold against computed truth, same protocol as the main set."""
    from pathlib import Path as _P

    path = _P(__file__).parent / "gold_heldout.json"
    if not path.exists():
        return {}
    gold = json.loads(path.read_text())["queries"]
    truth = compute_heldout_truth()
    out = {}
    for qid, t in truth.items():
        g = gold.get(qid)
        if not g:
            continue
        out[qid] = {
            "n_relevant_by_truth": sum(1 for v in t.values() if v >= 1),
            "n_relevant_by_gold": g["n_relevant"],
            "adjudicated": _score(g["labels"], t),
            "kappa": g["kappa"],
            "gwet_ac1": g.get("gwet_ac1"),
        }
    return out
