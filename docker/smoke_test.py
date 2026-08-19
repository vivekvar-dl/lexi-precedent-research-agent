"""Post-deployment verification, run INSIDE the container.

    az webapp ssh ...   # then: python /app/docker/smoke_test.py

Checks the things that actually differ between a laptop and a container, in
dependency order, so the first failure tells you which layer broke:

  1. index integrity     -- did the baked LanceDB store survive the image build
  2. offline model load  -- TRANSFORMERS_OFFLINE=1 means a cache-path mismatch
                            fails loudly here instead of silently downloading
                            1.1 GB on a user's first request
  3. retrieval           -- dense + BM25 + fusion against the real store
  4. LLM reachability    -- the app setting actually reached the process
  5. agent, simple path  -- end-to-end, cheap
  6. agent, research path-- end-to-end, expensive; the real load test

Exit code is the number of failed checks, so it is usable in CI.
"""
from __future__ import annotations

import os
import resource
import sys
import time

sys.path.insert(0, "/app/src")

FAILED = 0


def check(name: str):
    """Decorator-free timing/reporting helper; returns a closure to call."""
    def run(fn):
        global FAILED
        t0 = time.time()
        try:
            detail = fn()
            print(f"  PASS  {name:<34} {time.time()-t0:6.1f}s  {detail}", flush=True)
        except Exception as e:  # noqa: BLE001 - report every failure, never abort early
            FAILED += 1
            print(f"  FAIL  {name:<34} {time.time()-t0:6.1f}s  "
                  f"{type(e).__name__}: {str(e)[:130]}", flush=True)
    return run


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # linux: KB


print("=" * 78)
print("DEPLOYED CONTAINER SMOKE TEST")
print("=" * 78)
print(f"python {sys.version.split()[0]} | cwd {os.getcwd()} | "
      f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")
print()


@check("1. index integrity")
def _():
    from lexi.index import get_tables
    cards, chunks = get_tables()
    nc, nk = cards.count_rows(), chunks.count_rows()
    assert nc == 56, f"expected 56 cards, found {nc}"
    assert nk > 1500, f"expected ~1629 chunks, found {nk}"
    return f"{nc} cards, {nk} chunks"


@check("2. embedding model (offline)")
def _():
    from lexi.index import embed_query
    v = embed_query("driver had no valid licence")
    assert len(v) == 1024, f"unexpected dim {len(v)}"
    return f"dim {len(v)}, peak RSS {rss_mb():.0f} MB"


@check("3. hybrid retrieval")
def _():
    from lexi.retrieve import dense_search, rrf_fuse, sparse_search
    q = "insurer denies liability because driver had no valid licence"
    d, s = dense_search(q, 10), sparse_search(q, 10)
    fused = rrf_fuse(d, s)
    assert fused, "fusion returned nothing"
    return f"dense {len(d)}, bm25 {len(s)}, fused {len(fused)}, top={fused[0].doc_id}"


@check("4. LLM reachable")
def _():
    from lexi.llm import LLM
    out = LLM().complete("Reply with exactly: ok")
    assert out.strip(), "empty completion"
    return f"replied {out.strip()[:20]!r}"


@check("5. agent - simple query")
def _():
    from lexi.agent import Agent
    trace, result = Agent().run("Which of these judgments involve commercial vehicles?")
    n = sum(1 for e in trace.events if e.kind.value == "tool_call")
    assert type(result).__name__ == "DirectAnswer", f"got {type(result).__name__}"
    return (f"DirectAnswer, {n} tools, {len(result.cited_doc_ids)} cited, "
            f"{trace.in_tokens + trace.out_tokens:,} tok")


@check("6. agent - full research brief")
def _():
    from lexi.agent import Agent
    brief = (
        "Client: Mrs. Lakshmi Devi. Her husband was killed by a commercial truck whose "
        "driver held no valid licence. The insurer denies liability, arguing the policy "
        "is void. Deceased aged 42, income Rs 35,000/month, wife and two minor children. "
        "Research the corpus and advise: supporting precedents, adverse precedents, and "
        "strategy."
    )
    trace, result = Agent().run(brief)
    n = sum(1 for e in trace.events if e.kind.value == "tool_call")
    assert type(result).__name__ == "PrecedentResearchReport", f"got {type(result).__name__}"
    assert result.supporting, "no supporting precedents"
    assert result.adverse, "no adverse precedents - the brief's core requirement"
    return (f"{len(result.supporting)} supporting, {len(result.adverse)} adverse, "
            f"{n} tools, {trace.in_tokens + trace.out_tokens:,} tok, "
            f"peak RSS {rss_mb():.0f} MB")


print()
print("=" * 78)
print(f"{'ALL CHECKS PASSED' if not FAILED else f'{FAILED} CHECK(S) FAILED'} "
      f"| peak RSS {rss_mb():.0f} MB")
print("=" * 78)
sys.exit(FAILED)
