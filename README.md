# Lexi — Legal Precedent Research Agent

An agent that researches a corpus of 56 Indian court judgments and produces
litigator-usable analysis: supporting precedents, adverse precedents with honest
risk assessment, and a strategy recommendation.

It is not a pipeline. The agent selects its own tools each turn, so the same
system answers *"which of these judgments involve commercial vehicles?"* and
*"analyse this case brief and tell me what the other side will use against us"*
without any branching on query type.

**Hosted app:** <https://lexi-precedent-agent.azurewebsites.net>

**Read [`ADR.md`](ADR.md) for the design rationale, and [`RESULTS.md`](RESULTS.md)
for the evaluation methodology, results and failure analysis.**

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then add your AZURE_API_KEY
```

The LLM is **DeepSeek-V4-Flash on Azure AI Foundry**. Get the key from your
deployment's *Keys and Endpoint* page. To point at a different resource, override
`LEXI_AZURE_BASE_URL` and `LEXI_CHAT_MODEL`.

Any OpenAI-compatible endpoint works — provider access is isolated to
[`src/lexi/llm.py`](src/lexi/llm.py), so switching is a one-file change.

Build the index (one time, ~10 minutes — 56 LLM calls plus a 1.2 GB model download):

```bash
export PYTHONPATH=src
python -m lexi.ingest     # PDFs  -> cleaned text -> 1,629 structure-aware chunks
python -m lexi.enrich     # judgments -> 56 structured case cards  (resumable)
python -m lexi.index      # embeddings + BM25 -> LanceDB
```

Run the app:

```bash
streamlit run app.py
```

## Evaluation

```bash
export PYTHONPATH=src
python -m evals.gold        # build the gold label set (56 docs x 15 queries)
python -m evals.run_all     # run the agent, score all five dimensions
python -m evals.run_heldout # six held-out queries, never tuned against
python -m evals.ablate      # A/B one config change at a time
pytest evals/ -v            # unit tests + threshold assertions
```

Outputs land in `evals/results/report.md` and `evals/results/results.json`.

Useful flags:

| flag | effect |
|---|---|
| `--force-runs` | re-run the agent instead of using cached runs |
| `--seeds 3` | repeat each query 3× to measure self-consistency |
| `--no-faith` | skip LLM faithfulness checks (cheaper, fewer API calls) |

---

## Architecture

```
PDFs ──▶ ingest ──▶ enrich ──▶ index ──▶ retrieve ──▶ tools ──▶ agent ──▶ app
         (parse)    (LLM,      (LanceDB) (hybrid)    (5 tools)  (LangGraph
                     1×/doc)                                     tool loop)
                                                          │
                                                       trace ──▶ evals
```

The **trace** is the load-bearing shared object: Streamlit renders it live, and
the evaluation framework scores the very same events. One artifact, two
consumers.

### The four layers

**1. Offline enrichment.** One LLM pass per judgment produces a structured
`CaseCard` — court, issues, statutes, holding, *ratio*, disposition, factual
matrix, quantum, and crucially `outcome_favours` (claimant / insurer / mixed).
Run once, cached, committed. This turns *"semantic similarity over prose"* into
*"structured reasoning over a small knowledge base"*, and it is what makes
adverse retrieval possible at all — the agent can ask for judgments that went
*against* a claimant instead of hoping an embedding surfaces them.

**2. Hybrid retrieval.** Dense (`Qwen3-Embedding-0.6B`) + BM25, fused by
reciprocal rank, then LLM-reranked. Plus an exact metadata filter that scans all
56 cards, and a full-corpus screening mode.

Note what is matched: **candidate generation runs over the case cards**, each
judgment embedded as one synthesised document of its legal substance (holding,
*ratio*, issues, statutes, outcome). Chunks — structure-aware, each carrying a
contextual header — are searched only *within* an already-selected judgment, to
supply the verbatim passages `read_judgment` returns. Cards decide what is
relevant; chunks prove it. In law the citable unit is the judgment, not the
paragraph, and Indian judgments carry enough recital boilerplate that
chunk-level matching ranks furniture.

**3. The agent.** A LangGraph cycle — `agent ⇄ tools` — with five tools.
Depth is a *budget*, not a gate: triage sets a starting allowance, and the agent
escalates it mid-run while it is still surfacing new documents. The output shape
is itself a tool choice (`submit_research_report` vs `submit_answer`), so the
required three-part structure is guaranteed when relevant without any code path
forcing it.

**4. Evaluation.** All 56 judgments are labelled against every query by two
independent annotators on different models; disagreements are adjudicated
against full judgment text. That gives an *exact* recall denominator rather than
an estimated one.

### The tools

| tool | what it does | when the agent reaches for it |
|---|---|---|
| `search_precedents` | hybrid search + LLM rerank | default: "find precedents about X" |
| `filter_judgments` | exact filter over all 56 cards | enumerative questions where completeness matters |
| `screen_corpus` | LLM reads a summary of every judgment | recall backstop; verifying nothing was missed |
| `read_judgment` | opens one case, returns passages | before citing anything |
| `compute_quantum` | Sarla Verma / Pranay Sethi arithmetic | compensation figures — never estimated |

## Repository layout

```
src/lexi/
  config.py     all tunables; model choices with measured rationale
  schemas.py    typed contracts (CaseCard, ScoredDoc, output contracts)
  ingest.py     PDF -> clean text -> structure-aware chunks
  enrich.py     judgment -> CaseCard (one-time, resumable)
  index.py      LanceDB build: embeddings + BM25
  retrieve.py   dense + sparse -> RRF -> rerank; filters; screening
  quantum.py    deterministic compensation calculator
  tools.py      the five agent tools + terminal contracts
  agent.py      LangGraph tool loop, triage, budget escalation
  trace.py      typed events consumed by both the UI and the evals
  llm.py        provider wrapper (Azure AI Foundry / any OpenAI-compatible):
                structured output, rate limiting, retries, content-filter handling
evals/
  queries.py    15 eval queries across 6 query kinds
  heldout.py    6 held-out queries, written after the agent froze
  gold.py       dual-annotator gold set + kappa/AC1 + adjudication + override
  verifiable.py deterministic ground truth (504 labels settled by text search)
  metrics.py    precision, recall, nDCG, kappa, AC1, entropy, Jaccard
  dimensions.py graded dimensions 1-4
  behaviour.py  dimension 5: trajectory, contract, abstention, cost
  ablate.py     one-variable-at-a-time config ablation
  runner.py     cached agent runs
  run_all.py    orchestration -> results.json + report.md
  run_heldout.py generalisation run
  test_evals.py pytest: unit tests + threshold assertions
  test_oracle.py synthetic perfect/broken agents + mutation tests
app.py          Streamlit UI with live trace
```

## Deploying it

**What runs where.** `ingest` / `enrich` / `index` are one-time and belong on
your machine, not the server — they cost 56 LLM calls and produce the LanceDB
store. The server only needs to *read* that store and embed incoming queries.

**But query embedding cannot be precomputed.** Every dense search embeds the
user's question at request time, so `Qwen3-Embedding-0.6B` has to be resident in
the web process. Measured on CPU:

| | measured |
|---|---|
| model on disk | 1.1 GB |
| peak RSS with model loaded | **1,427 MB** |
| cold start (import + load + first embed) | 12.2 s |
| warm query embed | 0.159 s |
| LanceDB store (`index/lance`) | 9.6 MB |

So budget **≥ 2 GB RAM** for the instance. A 512 MB tier will OOM on model load
— that is the first thing to get right, and no amount of code tuning avoids it.

**Two deployment details that bite:**

1. `index/lance/` is git-ignored (rebuilt by `make index`), while
   `case_cards.json` and `chunks.json` *are* committed. So either commit the
   LanceDB directory too, or run `python -m lexi.index` once as a build step —
   it re-embeds from the committed chunks and needs no LLM calls.
2. A deep research query runs several minutes (§8: ~300 K tokens, ~5 min on a
   full brief). Streamlit holds this over a websocket so there is no HTTP
   request timeout, but any proxy idle-timeout in front of it must exceed that.

**If 2 GB is not available**, the honest options, in order of what they cost you:

- **A hosted embedding API** — drops server RAM to ~200 MB, adds a network hop
  per search, and changes the vectors, so the index must be rebuilt with the
  same model and the evals re-run before any number in the ADR still applies.
- **A smaller embedding model** — same caveat, and MLEB (§1.2) says you lose
  retrieval quality that was chosen on measurement.
- **BM25-only** (`hybrid_search(..., rerank=False)` with dense disabled) — zero
  model, zero RAM, and a large measured quality loss. Fine for a smoke test,
  not for a demo of the system the ADR describes.

## Notes on the corpus

Measured, not assumed:

- 56 judgments, 936 pages, 2,766,952 characters after cleaning (~692 K tokens)
- **All have real text layers** — no OCR needed
- Indian Kanoon exports; repeating page footers are stripped during ingest
- **The corpus is deliberately mixed.** Roughly 32 are motor-accident matters;
  the rest are trademark, excise, cheque dishonour, consumer, civil property and
  criminal judgments. These act as a planted precision test — an agent that
  drags *New Balance Athletics* into a motor-accident brief is over-retrieving.
- Outcome distribution: 28 favour the claimant, 16 mixed, 9 the insurer, 3 neutral

## Configuration

Everything tunable is in `src/lexi/config.py`, each default carrying the
measurement that set it (including the six changes that were built, measured and
**reverted** — those stay behind switches so the negative results reproduce).

Model roles are deliberately split: the agent runs on DeepSeek-V4-Flash, while
the reasoning judge and the second gold annotator run on Kimi-K2.6. That keeps
the evaluation from ever grading a model with itself. (An earlier build ran on
Gemini and split roles to multiply free-tier quota; the split survived the
migration to Azure for the independence reason, which was always the better
one.)
