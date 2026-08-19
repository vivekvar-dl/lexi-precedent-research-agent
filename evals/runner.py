"""Runs the agent over the eval query set and caches the results.

Agent runs are the expensive part of evaluation, and all four dimensions score
the SAME runs. So they are executed once, cached to disk, and shared. Caching
also makes a scoring change re-runnable without paying for retrieval again --
and keeps the numbers in the ADR reproducible.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexi.agent import Agent  # noqa: E402
from lexi.config import settings  # noqa: E402
from lexi.schemas import DirectAnswer, PrecedentResearchReport  # noqa: E402

from .queries import QUERIES, EvalQuery  # noqa: E402

RUNS_DIR = Path(__file__).parent / "runs"


def run_path(qid: str, seed: int = 0) -> Path:
    return RUNS_DIR / f"{qid}__{seed}.json"


def run_agent(q: EvalQuery, seed: int = 0, force: bool = False) -> dict:
    """Execute one query, or return the cached run."""
    path = run_path(q.qid, seed)
    if path.exists() and not force:
        return json.loads(path.read_text())

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    record: dict = {"qid": q.qid, "seed": seed, "question": q.text, "kind": q.kind}

    try:
        agent = Agent()
        trace, result = agent.run(q.text)
        record |= {
            "ok": True,
            "trace": json.loads(trace.to_json()),
            "retrieved": trace.retrieved_doc_ids(),
            "tool_sequence": trace.tool_sequence(),
            "llm_calls": trace.llm_calls,
            "tokens": trace.in_tokens + trace.out_tokens,
            "result_type": type(result).__name__,
            "result": result.model_dump(mode="json") if result is not None else None,
            "cited": _cited_ids(result),
        }
    except Exception as e:
        record |= {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
            "retrieved": [],
            "cited": [],
            "result": None,
            "result_type": None,
        }

    record["elapsed_s"] = round(time.time() - t0, 1)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return record


def _cited_ids(result) -> list[str]:
    """Documents the agent actually put its name to, as distinct from ones it saw.

    The gap between `retrieved` and `cited` is the synthesis loss measured in
    Dimension 2.
    """
    if isinstance(result, PrecedentResearchReport):
        return list(
            dict.fromkeys([p.doc_id for p in result.supporting] + [a.doc_id for a in result.adverse])
        )
    if isinstance(result, DirectAnswer):
        return list(dict.fromkeys(result.cited_doc_ids))
    return []


def run_all(
    queries: list[EvalQuery] | None = None,
    seed: int = 0,
    force: bool = False,
    workers: int | None = None,
) -> dict:
    """Execute the query set, several queries at a time.

    Queries are wholly independent -- each gets its own agent, trace and output
    file -- so concurrency changes only wall-clock, never a result. Run
    sequentially this took ~3 minutes per query because almost all of that is
    waiting on the provider; the deployment allows 250 requests/minute and a
    single-threaded runner cannot get near it.

    Two shared resources are protected rather than duplicated: the embedding
    model is behind a lock (torch inference is not thread-safe), and the API rate
    limiter is process-wide, so the aggregate request rate is unchanged no matter
    how many workers run.
    """
    queries = queries or QUERIES
    workers = workers or settings.eval_workers
    out: dict[str, dict] = {}
    lock = Lock()
    done = 0

    def one(q: EvalQuery) -> tuple[str, dict]:
        return q.qid, run_agent(q, seed=seed, force=force)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, q): q for q in queries}
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                qid, rec = fut.result()
            except Exception as e:  # a crashed worker must not sink the suite
                qid, rec = q.qid, {"qid": q.qid, "seed": seed, "ok": False,
                                   "error": f"runner: {e}", "retrieved": [], "cited": [],
                                   "result": None, "result_type": None}
            with lock:
                out[qid] = rec
                done += 1
                if rec.get("ok"):
                    print(
                        f"[{done:>2}/{len(queries)}] {qid:<22} {rec['result_type']:<24} "
                        f"retrieved={len(rec['retrieved']):>2} cited={len(rec['cited']):>2} "
                        f"tools={len(rec['tool_sequence']):>2} {rec['elapsed_s']}s",
                        flush=True,
                    )
                else:
                    print(f"[{done:>2}/{len(queries)}] {qid:<22} FAILED: "
                          f"{rec['error'][:120]}", flush=True)
    return out


def load_runs(seed: int = 0) -> dict[str, dict]:
    out = {}
    for q in QUERIES:
        p = run_path(q.qid, seed)
        if p.exists():
            out[q.qid] = json.loads(p.read_text())
    return out


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_all(force=force)
