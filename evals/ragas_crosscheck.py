"""Cross-check our custom metrics against RAGAS — triangulation, not replacement.

    # 1. export the stored agent runs into RAGAS's expected shape
    python -m evals.ragas_crosscheck export

    # 2. score them (needs a SEPARATE venv -- see below)
    <ragas-venv>/bin/python evals/ragas_crosscheck.py score

Why this exists
---------------
The evaluation in `dimensions.py` is bespoke, and bespoke metrics invite a fair
objection: *you wrote the ruler, so of course you measure well.* The defences
already in place are the deterministic 504-label subset, the held-out query set
and the oracle tests. This adds an independent one: score the same runs with an
off-the-shelf framework and see whether it agrees.

RAGAS is not used as the primary framework, for reasons in the ADR — it scores a
retrieve→generate pipeline, has no concept of adverse precedents (Dimension 4,
the brief's stated thesis), and estimates recall where we can label all 56
documents exactly. But its faithfulness and context-precision metrics overlap
enough with Dimensions 1 and 3 to be a useful second opinion.

Why a separate venv
-------------------
Installing `ragas` into the project venv downgraded `openai` (3.1.0 → 2.54.0)
and pulled in `langchain`, `langchain-community`, `datasets` and `instructor`.
RAGAS 0.4.3 then failed to import at all against the `langchain-community` it
had just installed. Rather than pin the whole project around an evaluation
side-tool, this runs in its own environment:

    python -m venv /tmp/ragas-venv
    /tmp/ragas-venv/bin/pip install "ragas==0.2.15" "langchain-community<0.4"

That keeps `requirements.txt` — the thing a reviewer installs, and the thing
that has to fit in a container next to the embedding model — free of it.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"
OUT = HERE / "results" / ("ragas_input_subset.jsonl" if os.environ.get("RAGAS_SUBSET") else "ragas_input.jsonl")
SCORES = HERE / "results" / "ragas_scores.json"


# =============================================================================
# Step 1 -- export (runs in the project venv; only reads JSON)
# =============================================================================


def _contexts(trace: dict) -> list[str]:
    """The passages the agent actually read — RAGAS's notion of 'contexts'.

    Deliberately the READ passages, not the retrieved doc list: RAGAS asks
    whether the answer is grounded in the supplied context, so the context has
    to be the text the agent actually had in front of it.
    """
    out: list[str] = []
    for ev in trace.get("events", []):
        if ev.get("kind") != "read":
            continue
        p = ev.get("payload") or {}
        head = f"[{(p.get('doc_ids') or ['?'])[0]}] {p.get('title') or ''}"
        for psg in p.get("passages", []):
            txt = (psg.get("text") or "").strip()
            if txt:
                out.append(f"{head} — {psg.get('section', '')}\n{txt}")
    return out


def _answer(run: dict) -> str:
    """Flatten our typed contract into the prose RAGAS expects."""
    r = run.get("result") or {}
    if run.get("result_type") == "DirectAnswer":
        return str(r.get("answer") or "")
    parts = []
    for p in r.get("supporting", []):
        parts.append(f"SUPPORTING [{p['doc_id']}] {p.get('principle','')} "
                     f"{p.get('fact_alignment','')} {p.get('why_it_matters','')}")
    for a in r.get("adverse", []):
        parts.append(f"ADVERSE [{a['doc_id']}] {a.get('principle','')} "
                     f"{a.get('risk_to_client','')} {a.get('distinguishing_argument','')}")
    s = r.get("strategy") or {}
    if s:
        parts.append("STRATEGY " + " ".join(s.get("priority_arguments", []))
                     + f" Range: {s.get('compensation_range','')}. "
                     + " ".join(s.get("risks", [])))
    return "\n".join(parts)


def export() -> Path:
    sys.path.insert(0, str(HERE.parent))
    from evals.gold import load_gold

    gold = load_gold()["queries"]
    # Held-out labels live in their own file. Without this merge every h0* query
    # exported reference="None.", and RAGAS then scored their context_precision
    # 0.000 -- correctly, against a reference that said nothing was relevant.
    ho = HERE / "gold_heldout.json"
    if ho.exists():
        gold |= json.loads(ho.read_text())["queries"]
    rows = []
    for f in sorted(RUNS.glob("*__0.json")):
        run = json.loads(f.read_text())
        if not run.get("ok"):
            continue
        qid = run["qid"]
        ctx = _contexts(run.get("trace") or {})
        ans = _answer(run)
        if not ctx or not ans:
            continue  # RAGAS scores neither empty contexts nor empty answers
        g = gold.get(qid, {})
        rel = sorted(d for d, v in (g.get("labels") or {}).items() if v >= 1)
        rows.append({
            "qid": qid,
            "user_input": run["question"],
            "retrieved_contexts": [c[:1200] for c in ctx[:8]],   # cap: keeps RAGAS within its timeout
            "response": ans,
            # A reference built from the gold set: the judgments that SHOULD be
            # cited. It is not a model-written ideal answer, so context_recall
            # against it is a coarse proxy -- reported, but our own exhaustive
            # recall is the number that counts.
            "reference": ("Relevant judgments: " + ", ".join(rel)) if rel else "None.",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"wrote {OUT}  ({len(rows)} runs, "
          f"{sum(len(r['retrieved_contexts']) for r in rows)} context passages)")
    return OUT


# =============================================================================
# Step 2 -- score (runs in the ragas venv)
# =============================================================================


def score() -> None:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import LLMContextPrecisionWithoutReference, faithfulness
    from ragas.run_config import RunConfig

    key = os.environ.get("LEXI_AZURE_API_KEY") or os.environ.get("AZURE_API_KEY")
    base = os.environ.get("LEXI_AZURE_BASE_URL",
                          "https://ai-service-strataos.services.ai.azure.com/models")
    api_version = os.environ.get("LEXI_AZURE_API_VERSION", "2024-05-01-preview")
    if not key:
        sys.exit("set LEXI_AZURE_API_KEY (see .env)")

    # RAGAS is judged by Kimi, NOT by the agent's own model. Same principle as
    # our rubric judge: a framework that graded DeepSeek with DeepSeek would be
    # no more independent than our own metrics.
    judge = ChatOpenAI(
        model=os.environ.get("LEXI_JUDGE_MODEL", "Kimi-K2.6"),
        api_key=key, base_url=base, temperature=0,
        default_query={"api-version": api_version},
    )

    rows = [json.loads(l) for l in Path(OUT).read_text().splitlines() if l.strip()]

    # answer_relevancy needs an embedding model; ours is local (sentence-
    # transformers) and Azure serves no embedding deployment here, so it is
    # skipped rather than silently scored with a different model than the
    # system actually uses.
    # WithoutReference, deliberately: the only reference we can build from the
    # gold set is a list of doc_ids, and "was this passage useful in producing
    # `Relevant judgments: doc_004, doc_012`?" is not a question a judge can
    # answer. The no-reference variant asks whether each retrieved context was
    # useful for the ANSWER, which is both answerable and the thing we care about.
    metrics = [faithfulness, LLMContextPrecisionWithoutReference()]
    # RAGAS's defaults (180s, 16 workers) time out on this data: a research answer
    # carries many passages of judgment text and faithfulness decomposes every
    # claim in it against every context. A first attempt lost 33 of 40 jobs to
    # timeouts. Scoring ONE query per evaluate() call, with a long timeout and low
    # concurrency, trades wall-clock for actually getting numbers -- and means a
    # single pathological query cannot empty the whole table.
    cfg = RunConfig(timeout=180, max_workers=6, max_retries=2)

    def _segments(resp: str, cap: int = 1400) -> list[str]:
        """Split a research answer on its own entry boundaries.

        Not truncation -- every segment is scored, so the whole answer is
        covered. This exists because Azure's content filter blocks the JUDGE's
        output when faithfulness decomposes a long answer containing
        fatal-accident detail (`MultiSeverity_ViolenceScore`); measured, a full
        9-11K-char answer is refused while ~1.5K segments pass. Same filter our
        own agent handles in `llm.py` -- RAGAS has no equivalent.
        """
        segs, cur = [], ""
        for line in resp.split("\n"):
            if cur and len(cur) + len(line) > cap:
                segs.append(cur)
                cur = line
            else:
                cur = f"{cur}\n{line}".strip()
        if cur:
            segs.append(cur)
        return segs or [resp]

    per_query: dict[str, dict] = {}
    for i, r in enumerate(rows, 1):
        base = {k: v for k, v in r.items() if k != "qid"}
        vals: dict[str, float | None] = {}

        # context precision: one call, the whole answer -- it does not decompose
        # claims, so it never trips the filter.
        cp = metrics[1]
        try:
            df = evaluate(Dataset.from_list([base]), metrics=[cp], llm=judge,
                          raise_exceptions=False, run_config=cfg).to_pandas()
            v = df.to_dict("records")[0].get(cp.name)
            vals[cp.name] = None if v != v else float(v)
        except Exception:
            vals[cp.name] = None

        # Faithfulness: all segments of this query in ONE evaluate() call.
        #
        # An earlier version called evaluate() per segment -- ~160 calls across
        # the set -- and drowned in APIConnectionError (147 of them): every call
        # builds its own async executor and connection pool, and the sockets
        # accumulate faster than they are reclaimed. Exactly the failure our own
        # `llm.py` fixes by caching the client. Batching segments per query cuts
        # this to ~40 calls and keeps each one small enough for the content
        # filter.
        segs = _segments(base["response"])
        got = []
        try:
            df = evaluate(Dataset.from_list([dict(base, response=s) for s in segs]),
                          metrics=[faithfulness], llm=judge, raise_exceptions=False,
                          run_config=cfg).to_pandas()
            got = [float(v) for v in df["faithfulness"].tolist()
                   if v == v and v is not None]
        except Exception as e:
            print(f"        faithfulness batch failed: {type(e).__name__}", flush=True)
        vals["faithfulness"] = (sum(got) / len(got)) if got else None
        vals["faithfulness_segments_scored"] = f"{len(got)}/{len(segs)}"
        time.sleep(2)   # let sockets close before the next query

        print(f"  [{i}/{len(rows)}] {r['qid']:<24} "
              f"faithfulness={'n/a' if vals['faithfulness'] is None else format(vals['faithfulness'], '.3f')}"
              f" ({vals['faithfulness_segments_scored']} segs)  "
              f"{cp.name}={'n/a' if vals[cp.name] is None else format(vals[cp.name], '.3f')}",
              flush=True)
        per_query[r["qid"]] = vals
        SCORES.write_text(json.dumps({"per_query": per_query}, indent=1))   # checkpoint

    def _mean(name):
        vs = [v[name] for v in per_query.values() if v[name] is not None]
        return sum(vs) / len(vs) if vs else None
    means = {m.name: _mean(m.name) for m in metrics}
    SCORES.write_text(json.dumps(
        {"means": means, "per_query": per_query, "n_runs": len(rows),
         "n_scored": {m.name: sum(1 for v in per_query.values()
                                  if v[m.name] is not None) for m in metrics}}, indent=1))

    print(f"\n{'qid':<24} " + " ".join(f"{m.name:>18}" for m in metrics))
    for qid, vals in per_query.items():
        print(f"{qid:<24} " + " ".join(
            f"{'n/a':>18}" if vals[m.name] is None else f"{vals[m.name]:>18.3f}"
            for m in metrics))
    print(f"\n{'MEAN':<24} " + " ".join(
        f"{'n/a':>18}" if means[m.name] is None else f"{means[m.name]:>18.3f}"
        for m in metrics))
    print(f"\nwrote {SCORES}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "export"
    {"export": export, "score": score}[cmd]()
