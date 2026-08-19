"""Run the full evaluation and write results.json + report.md.

    python -m evals.run_all                 # use cached gold + cached agent runs
    python -m evals.run_all --force-runs    # re-run the agent
    python -m evals.run_all --seeds 3       # add self-consistency across 3 runs
    python -m evals.run_all --no-faith      # skip LLM faithfulness checks (cheaper)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .dimensions import (
    dimension_1_precision,
    dimension_2_recall,
    dimension_3_reasoning,
    dimension_4_adverse,
    run_poison_probe,
    self_consistency,
)
from .behaviour import dimension_5_behaviour
from .verifiable import validate_annotators
from .gold import build_gold, load_gold
from .queries import QUERIES
from .runner import load_runs, run_all as run_agent_all

OUT_DIR = Path(__file__).parent / "results"


def _kappa_band(k: float) -> str:
    """Landis & Koch interpretation, so a bare number is not left to the reader."""
    for lo, label in ((0.81, "almost perfect"), (0.61, "substantial"), (0.41, "moderate"),
                      (0.21, "fair"), (0.0, "slight")):
        if k >= lo:
            return label
    return "none"


def _fmt(v, pct: bool = True) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v*100:.1f}%" if pct else f"{v:.3f}"
    return str(v)


def write_report(res: dict, path: Path) -> None:
    g, d1, d2, d3, d4 = (
        res["gold_summary"],
        res["dimension_1_precision"],
        res["dimension_2_recall"],
        res["dimension_3_reasoning"],
        res["dimension_4_adverse"],
    )
    L: list[str] = []
    A = L.append

    A("# Evaluation Results\n")
    A(f"_Generated {res['generated_at']} · {res['n_queries']} queries × 56 judgments_\n")

    A("## Gold set\n")
    A("Every one of the 56 judgments is labelled against every query, so the recall")
    A("denominator is exact rather than estimated. Two independent annotators on")
    A("different models; disagreements adjudicated against full judgment text.\n")
    A(f"- Labelled pairs: **{g['n_pairs']}**")
    A(f"- Mean Cohen's κ: **{g['mean_kappa']:.3f}** (min {g['min_kappa']:.3f}, max {g['max_kappa']:.3f})")
    A(f"- Disagreements adjudicated against full text: **{g['n_adjudicated']}**\n")

    A("| query | kind | relevant | on-point | κ | AC1 | raw | prevalence |")
    A("|---|---|---|---|---|---|---|---|")
    for qid, q in g["per_query"].items():
        A(f"| `{qid}` | {q['kind']} | {q['n_relevant']} | {q['n_on_point']} | "
          f"{q['kappa']:.2f} | {q.get('gwet_ac1', float('nan')):.2f} | "
          f"{q.get('raw_agreement', float('nan')):.2f} | "
          f"{q.get('prevalence', float('nan')):.2f} |")
    A("")
    A("**Why three agreement statistics.** These labels are heavily skewed -- for any one")
    A("query most of the 56 judgments are irrelevant -- and Cohen's κ collapses under skew")
    A("even when annotators agree on nearly everything (the *κ paradox*). Gwet's AC1 is")
    A("robust to it, and the legal-RAG literature recommends AC-family statistics over")
    A("κ/α precisely for skewed distributions. Read them together:\n")
    A("- low κ + **high** AC1 + high raw → labels are fine, the class balance is skewed")
    A("- low κ + **low** AC1 → the rubric is genuinely ambiguous and needs rewriting\n")
    paradox = [q for q, v in g["per_query"].items()
               if v["kappa"] < 0.6 <= v.get("gwet_ac1", 0)]
    genuine = [q for q, v in g["per_query"].items()
               if v["kappa"] < 0.6 and v.get("gwet_ac1", 0) < 0.6]
    if paradox:
        A(f"κ paradox (skew, not ambiguity): {', '.join('`'+q+'`' for q in paradox)}\n")
    if genuine:
        A(f"⚠️ Genuinely ambiguous rubric (both statistics low): "
          f"{', '.join('`'+q+'`' for q in genuine)} -- rewrite before relying on these.\n")

    # --- annotator validation ---
    av = res.get("annotator_validation")
    if av and av.get("per_query"):
        A("### Are the labels themselves any good?\n")
        A("The gold set is model-labelled, so it needs its own check. For a subset of")
        A("queries the answer is *computable* -- whether a judgment cites a given case, or")
        A("mentions a given section, is decided by literal text search over the raw PDFs,")
        A("with no model involved. Scoring the annotators against that computed key gives a")
        A("measured floor on their reliability.\n")
        m = av["mean_accuracy"]
        A(f"- Independently verifiable labels: **{av['n_labels_verified']}** "
          f"({av['n_queries_verified']} queries × 56 judgments)")
        A(f"- Annotator A accuracy: **{_fmt(m['annotator_a'])}**")
        A(f"- Annotator B accuracy: **{_fmt(m['annotator_b'])}**")
        A(f"- After adjudication: **{_fmt(m['adjudicated'])}**\n")
        A("| query | true relevant | A | B | final |")
        A("|---|---|---|---|---|")
        for qid, r in av["per_query"].items():
            A(f"| `{qid}` | {r['n_relevant_by_truth']} | "
              f"{_fmt(r['annotator_a'].get('accuracy'))} | "
              f"{_fmt(r['annotator_b'].get('accuracy'))} | "
              f"{_fmt(r['adjudicated'].get('accuracy'))} |")
        A("")

    A("## Dimension 1 — Precision\n")
    A(f"- Precision of cited precedents: **{_fmt(d1['mean_precision_cited'])}**")
    A(f"- Precision@10 of retrieval: **{_fmt(d1['mean_precision_retrieved@10'])}**")
    A(f"- nDCG@10 (graded): **{_fmt(d1['mean_ndcg@10'], pct=False)}**")
    A(f"- Citation faithfulness: **{_fmt(d1['mean_faithfulness'])}**")
    A(f"- Hallucinated citations (doc_id not in corpus): **{d1['hallucinated_citations']}**\n")
    A("| query | prec (cited) | P@10 | nDCG@10 | faithful | false positives |")
    A("|---|---|---|---|---|---|")
    for qid, r in d1["per_query"].items():
        A(
            f"| `{qid}` | {_fmt(r['precision_cited'])} | {_fmt(r['precision_retrieved@10'])} | "
            f"{_fmt(r['ndcg@10'], pct=False)} | {_fmt(r.get('faithfulness'))} | "
            f"{', '.join(r['false_positives']) or '—'} |"
        )
    A("")

    A("## Dimension 2 — Recall\n")
    A(f"- Retrieval recall (agent *saw* it): **{_fmt(d2['mean_retrieval_recall'])}**")
    A(f"- Answer recall (agent *cited* it): **{_fmt(d2['mean_answer_recall'])}**")
    A(f"- **Synthesis loss** (found then dropped): **{_fmt(d2['mean_synthesis_loss'])}**")
    A(f"- On-point recall (grade-2 only): **{_fmt(d2['mean_on_point_recall'])}**")
    A(f"- **Evidence Score** (recall, penalised below half the controlling set): "
      f"**{_fmt(d2['mean_evidence_score'])}**")
    if d2.get("skipped_no_recall_expected"):
        A(f"- Not scored for recall (correct answer is 'nothing exists'): "
          f"{', '.join('`'+q+'`' for q in d2['skipped_no_recall_expected'])}")
    A("")
    A("| query | retrieval | answer | loss | missed entirely |")
    A("|---|---|---|---|---|")
    for qid, r in d2["per_query"].items():
        A(
            f"| `{qid}` | {_fmt(r['retrieval_recall'])} | {_fmt(r['answer_recall'])} | "
            f"{_fmt(r['synthesis_loss'])} | {', '.join(r['missed'][:6]) or '—'} |"
        )
    A("")

    A("## Dimension 3 — Reasoning quality\n")
    A(f"- Mean rubric score: **{_fmt(d3['mean_rubric_score'])}**")
    A("\nJudged by a *different* model from the agent, and required to quote evidence.\n")
    if d3.get("criterion_means"):
        A("| criterion | score |")
        A("|---|---|")
        for k, v in d3["criterion_means"].items():
            A(f"| {k.replace('_',' ')} | {_fmt(v)} |")
        A("")
    if d3.get("self_consistency"):
        sc = d3["self_consistency"]
        A(f"- Self-consistency across {sc['n_seeds']} runs (Jaccard of cited set): "
          f"**{_fmt(sc['mean_jaccard'])}**\n")
    p = res.get("poison_probe") or {}
    if p and not p.get("error"):
        verdict = "❌ ADOPTED the falsehood" if p.get("adopted") else "✅ rejected the falsehood"
        A(f"\n### Poisoned-premise probe\n")
        A("The agent is told, with false authority, that a judgment in the corpus holds")
        A("the opposite of what it actually holds, and asked to confirm it. Sycophancy")
        A("here would manufacture legal authority -- the worst failure this system has.\n")
        A(f"- Verdict: **{verdict}**")
        A(f"- Explicitly corrected the premise: **{'yes' if p.get('corrected_explicitly') else 'no'}**")
        A(f"- Actually opened the cited judgment: **{'yes' if p.get('read_the_source') else 'no'}**")
        if p.get("evidence"):
            A(f"- Judge's evidence: _{p['evidence'][:300]}_")
        A("")
    elif p.get("error"):
        A(f"\n- Poisoned-premise probe failed to run: `{p['error']}`\n")

    A("## Dimension 4 — Adverse identification\n")
    A(f"- Adverse recall: **{_fmt(d4['mean_adverse_recall'])}**")
    A(f"- Research answers containing an adverse section: **{_fmt(d4['pct_with_adverse_section'])}**")
    A(f"- **Buried** (retrieved an adverse case, then omitted it): **{d4['total_buried']}**")
    A(f"- Miscast (opposing-side WIN presented as supporting): **{d4['total_miscast']}**")
    A(f"- Pay-and-recover cited as supporting (defensible either way; informational): "
      f"**{d4['total_mixed_as_supporting']}**")
    A(f"- Risk calibration entropy (0 = every risk labelled the same): "
      f"**{_fmt(d4['mean_risk_entropy'], pct=False)}**\n")
    if d4["sycophancy"]:
        A("### Sycophancy probe\n")
        A("The same matter asked from the claimant side and the insurer side. The relevant")
        A("judgments should barely move — only their labelling should flip. Both sides must")
        A("still be told what cuts against them.\n")
        A("| comparison | retrieved overlap | cited overlap | both report adverse |")
        A("|---|---|---|---|")
        for k, v in d4["sycophancy"].items():
            A(
                f"| {k} | {_fmt(v['retrieved_overlap'])} | {_fmt(v['cited_overlap'])} | "
                f"{'✅' if v['both_sides_report_adverse'] else '❌'} |"
            )
        A("")

    d5 = res.get("dimension_5_behaviour") or {}
    if d5:
        A("## Dimension 5 — Behaviour (trajectory, contract, abstention, cost)\n")
        A("The four graded dimensions score what the agent *says*. This scores what it")
        A("*does*, read off the same trace the UI renders. An agent can produce a decent")
        A("answer by an indefensible route, and the route is what breaks next time.\n")
        A(f"- Run success rate: **{_fmt(d5['run_success_rate'])}**")
        A(f"- **Abstention** (cites nothing when the corpus cannot answer): "
          f"**{_fmt(d5['abstention_rate'])}** over {d5['n_abstention_queries']} quer(y/ies)")
        A(f"- **Output-contract accuracy** (research report vs direct answer): "
          f"**{_fmt(d5['contract_accuracy'])}**")
        if d5.get("contract_failures_high_severity"):
            A(f"  - ⚠️ High-severity contract failures (a brief answered as prose, so no "
              f"adverse section exists): {', '.join('`'+q+'`' for q in d5['contract_failures_high_severity'])}")
        A(f"- **Trajectory** (used an appropriate tool for the query kind): "
          f"**{_fmt(d5['trajectory_accuracy'])}**")
        if d5.get("trajectory_failures"):
            A(f"  - Wrong instrument: {', '.join('`'+q+'`' for q in d5['trajectory_failures'])}"
              f" — e.g. answering an enumerative question from a top-k search, which cannot"
              f" be complete by construction")
        A(f"- Cost: **{d5['mean_tool_calls']:.1f}** tool calls, "
          f"**{d5['mean_tokens']:,.0f}** tokens, "
          f"**{d5['mean_latency_s']:.0f}s** mean / **{d5['max_latency_s']:.0f}s** max per query\n")

    A("## Failures\n")
    for qid, r in d2["per_query"].items():
        if r["missed"] or r["found_but_dropped"]:
            A(f"- `{qid}`: missed {r['missed'] or '—'}; found-but-dropped "
              f"{r['found_but_dropped'] or '—'}")
    failed = [q for q, r in res["runs_summary"].items() if not r["ok"]]
    if failed:
        A(f"- Agent errored on: {', '.join(failed)}")
    A("")

    path.write_text("\n".join(L))


def main() -> None:
    argv = sys.argv[1:]
    force_runs = "--force-runs" in argv
    do_faith = "--no-faith" not in argv
    n_seeds = 1
    if "--seeds" in argv:
        n_seeds = int(argv[argv.index("--seeds") + 1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1/4  gold set ===")
    try:
        gold = load_gold()
        if len(gold["queries"]) < len(QUERIES):
            gold = build_gold()
    except FileNotFoundError:
        gold = build_gold()

    print("\n=== 2/4  agent runs ===")
    runs_by_seed = {}
    for seed in range(n_seeds):
        print(f"-- seed {seed} --")
        run_agent_all(seed=seed, force=force_runs)
        runs_by_seed[seed] = load_runs(seed=seed)
    runs = runs_by_seed[0]

    print("\n=== 3/4  scoring ===")
    d1 = dimension_1_precision(runs, gold, check_faithfulness=do_faith)
    print(f"  D1 precision(cited)={d1['mean_precision_cited']:.3f} ndcg={d1['mean_ndcg@10']:.3f}")
    d2 = dimension_2_recall(runs, gold)
    print(f"  D2 retrieval_recall={d2['mean_retrieval_recall']:.3f} "
          f"answer_recall={d2['mean_answer_recall']:.3f}")
    sc = self_consistency(runs_by_seed) if n_seeds > 1 else None
    d3 = dimension_3_reasoning(runs, gold, self_consistency=sc)
    print(f"  D3 rubric={d3['mean_rubric_score']:.3f}")
    d4 = dimension_4_adverse(runs, gold)
    print(f"  D4 adverse_recall={d4['mean_adverse_recall']:.3f} buried={d4['total_buried']}")

    # Executes the agent against a planted false premise -- a live run, not a
    # re-score of an existing one.
    poison = {} if "--no-poison" in argv else run_poison_probe()
    if poison:
        print(f"  poison probe: adopted={poison.get('adopted')} "
              f"read_source={poison.get('read_the_source')}")

    # Scores the ANNOTATORS against deterministically-computed truth, so the
    # gold set's own reliability is a published number rather than an assumption.
    d5 = dimension_5_behaviour(runs)
    print(f"  D5 contract={d5['contract_accuracy']} trajectory={d5['trajectory_accuracy']} "
          f"abstention={d5['abstention_rate']} success={d5['run_success_rate']:.3f}")

    annot = validate_annotators(gold)
    ma = annot["mean_accuracy"]
    print(f"  annotator validation ({annot['n_queries_verified']} verifiable queries): "
          f"A={ma['annotator_a']:.3f} B={ma['annotator_b']:.3f} final={ma['adjudicated']:.3f}")

    qs = gold["queries"]
    ks = [q["kappa"] for q in qs.values()]
    res = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_queries": len(qs),
        "gold_summary": {
            "n_pairs": len(qs) * 56,
            "mean_kappa": sum(ks) / len(ks),
            "min_kappa": min(ks),
            "max_kappa": max(ks),
            "n_adjudicated": sum(q["n_adjudicated"] for q in qs.values()),
            "per_query": {
                k: {
                    "kind": v["kind"],
                    "n_relevant": v["n_relevant"],
                    "n_on_point": v["n_on_point"],
                    "kappa": v["kappa"],
                }
                for k, v in qs.items()
            },
        },
        "runs_summary": {
            k: {"ok": v.get("ok", False), "n_retrieved": len(v.get("retrieved", [])),
                "n_cited": len(v.get("cited", [])), "tools": v.get("tool_sequence", []),
                "elapsed_s": v.get("elapsed_s")}
            for k, v in runs.items()
        },
        "dimension_1_precision": d1,
        "dimension_2_recall": d2,
        "dimension_3_reasoning": d3,
        "dimension_4_adverse": d4,
        "dimension_5_behaviour": d5,
        "poison_probe": poison,
        "annotator_validation": annot,
    }

    print("\n=== 4/4  writing ===")
    (OUT_DIR / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    write_report(res, OUT_DIR / "report.md")
    print(f"  {OUT_DIR/'results.json'}")
    print(f"  {OUT_DIR/'report.md'}")


if __name__ == "__main__":
    main()
