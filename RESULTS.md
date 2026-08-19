# Evaluation Results and Analysis

Companion to [`ADR.md`](ADR.md), which records the design decisions. This
document covers the evaluation framework, the measurements behind each decision,
and the failure analysis.

Generated output: [`evals/results/report.md`](evals/results/report.md) ·
raw scores: `evals/results/*.json` · every table below is recomputable from the
stored runs in `evals/runs*/`.

---

## 0. The three facts that drove every decision

Before choosing anything I measured the corpus. Three findings did most of the
architectural work:

**The corpus is small — 56 documents, 936 pages, 2,766,952 characters (~692 K
tokens).** Small enough that an LLM can afford to look at *every* document, once
offline and again at query time. That single fact unlocks techniques that are
impossible at scale, and I have used them deliberately rather than pretending the
system is scale-agnostic. Section 6 is explicit about which ones die at 5,000.

**The corpus is deliberately mixed.** Roughly 32 judgments are motor-accident
matters. The remainder are trademark (*New Balance Athletics*, *Intel*), central
excise (*Monnet Ispat*), cheque dishonour (*Mandvi Co-op Bank*), consumer,
civil-property and criminal judgments. These are a planted **precision** test. A
naive top-k retriever will happily return an excise appeal for a motor-accident
brief because both contain "appeal", "liability" and "compensation".

**The brief's own example query is structured, not semantic.** *"Which of these
judgments involve commercial vehicles?"* has a definite, enumerable answer
And the honest answer is that *how many* depends on where you draw the line —
which is itself the argument. Measured three ways: the adjudicated gold set
(two annotators, disagreements resolved against full text) says **8**; the
enrichment layer's own `is_commercial_vehicle` flag fires on **17**; and **29**
judgments name a truck, lorry, bus, tempo or tractor. Those are not competing
estimates of one number, they are three different definitions of "commercial",
and a lawyer asking the question means a specific one.

That is exactly why this needs a metadata layer rather than a better embedding.
A vector search returns a *ranking* and cannot express "all of them" at any k —
top-5 caps below the true answer under every definition above, and gives the
reader no way to see which definition produced it. A structured predicate over
extracted metadata returns a complete set and makes the definition explicit and
arguable. That single example shaped the retrieval design more than anything
else in the brief.

---

## 1. Architecture

```
PDFs ──▶ ingest ──▶ enrich ──▶ index ──▶ retrieve ──▶ tools ──▶ agent ──▶ app
         (parse)    (LLM 1×/doc) (LanceDB) (hybrid)   (5 tools)  (LangGraph)
                                                        │
                                                     trace ──▶ evals
```

### 1.1 Offline enrichment is the centre of the design

One LLM pass per judgment produces a structured `CaseCard`: court, date, legal
issues, statutes cited, precedents cited, holding, *ratio*, disposition, a
factual matrix (vehicle type, licence defect, age, income, dependants), the
quantum method (multiplier, future-prospects %, award), and — the field that
matters most — `outcome_favours` ∈ {claimant, insurer, mixed, neutral}.

56 calls, once, cached and committed to the repo.

This converts the problem from *"semantic similarity over prose chunks"* into
*"structured reasoning over a small knowledge base"*, and buys three things that
plain RAG cannot:

1. **Exact enumeration.** "Which judgments involve commercial vehicles?" becomes
   a scan of 56 boolean fields, not a top-k gamble.
2. **Adverse retrieval that actually works.** Because `outcome_favours` is
   extracted at index time, the agent can *ask for* judgments that went against a
   claimant. Without it, surfacing adverse authority depends on an embedding
   happening to cooperate — which is exactly the failure mode that makes a legal
   research tool dangerous.
3. **Cheap full-corpus screening.** 56 compressed cards fit comfortably in one
   context window, so "read everything and pick" is a real option.

**Tradeoff accepted:** enrichment quality bounds everything downstream. A card
that misreads a holding poisons retrieval and reasoning together. I mitigated
this by keeping the deterministic parse authoritative for fields we can extract
reliably (title, date, source URL, page count) and letting the LLM fill only what
requires reading. It remains the system's single biggest dependency, and
§8 measures it rather than assuming it.

### 1.2 Retrieval: hybrid, fused, reranked, resolved to documents

- **Dense:** `Qwen3-Embedding-0.6B`.
- **Sparse:** BM25 over the same cards.
- **Fusion:** Reciprocal Rank Fusion.
- **Rerank:** an LLM scores the survivors 0–10 with a stated reason.
- **Exact filter:** a SQL-ish predicate over card metadata, returning *all*
  matches.
- **Screening:** an LLM reads a one-line summary of every judgment.

**Why this embedding model.** I picked it from MLEB (the Massive Legal Embedding
Benchmark, 10 expert-annotated legal datasets) rather than MTEB, because general
benchmarks disagree with legal ones. On MLEB, `Qwen3-Embedding-0.6B` scores
**77.13 nDCG@10** against **69.44** for BGE-M3 — my initial pick, and 7.7 points
worse on exactly this task. Ranking by the general leaderboard would have chosen
the weaker model. The three models above it are all proprietary APIs (Kanon 2
Embedder 86.0, Voyage 3 Large 85.7); Qwen3 is the best **open-source** option and
Apache-2.0.

**Why BM25 as well as vectors.** Legal queries carry exact tokens — "Section
149", "163A", "Swaran Singh" — that dense retrieval blurs. Embeddings understand
that negligence and rashness are related; they do not reliably distinguish
s.166 from s.163A, and that distinction decides cases.

**Why RRF rather than weighted score blending.** BM25 scores and cosine
similarities live on incomparable scales, so blending needs a normalisation
constant tuned per corpus — a hyperparameter that silently rots. RRF needs only
ranks.

**Why no vector database server.** LanceDB is embedded — it is just files. The
brief forbids submissions that require the reviewer to run local infrastructure,
which rules out Docker-hosted Qdrant/Weaviate/Milvus. At 1,629 chunks a served
ANN index would also be slower and *less* accurate than brute-force exact search,
since ANN is approximate. LanceDB additionally gives native full-text search, so
it replaced a separate `rank_bm25` dependency rather than adding one.

### 1.3 Chunking: structure-aware, contextual, document-resolved

The judgments are Indian Kanoon exports with a stable shape and a page footer
repeated on every page. Naive fixed-size chunking would embed that furniture into
every vector. So ingest strips footers, then segments on judgment structure
(preamble / judgment / order), then chunks within a section at ~2,200 characters
preferring paragraph then sentence boundaries.

Every chunk carries a **contextual header** — case name, court, date, section —
prepended before embedding. Without it, a passage reading *"the appeal is
allowed"* is meaningless in isolation.

**Where chunks are actually used — and where they are not.** Candidate
generation does *not* run over chunks. Dense and BM25 search both run over the
**case cards**: each judgment is embedded as one synthesised document (title,
court, issues, holding, ratio, principles, statutes, citations, outcome — see
`_card_document`). Chunks are queried only by `passages_for`, which is called
with explicit `doc_id`s after a judgment has already been selected, to pull the
verbatim passages `read_judgment` returns.

That is deliberate. In law the unit of precedential authority is the judgment,
not the paragraph, and matching a query against a judgment's *legal substance*
beats matching it against whichever paragraph happens to share vocabulary —
Indian judgments are full of recital boilerplate that chunk-level retrieval
ranks enthusiastically. **Cards decide what is relevant; chunks prove it.**

The cost of this choice is a real one and §6 is explicit about it: one vector
per judgment is lossy, and it is the first thing that breaks at scale.

**Tradeoff accepted:** I deliberately under-segment. A wrong boundary silently
truncates a *ratio* mid-sentence, which is worse than a slightly coarse chunk.

### 1.4 The agent: a cycle, not a pipeline

LangGraph, with exactly one cycle:

```
agent ──(wants tools)──▶ tools
  ▲                        │
  └────────────────────────┘
  │
  └──(terminal contract, or budget spent)──▶ END
```

There is no `retrieve → analyse → summarise` sequence anywhere in the codebase,
and no branching on query type. The model chooses among five tools every turn.

**Why LangGraph over LangChain or LlamaIndex.** The brief requires that
intermediate reasoning be visible; LangGraph streams state at every node, so the
requirement is nearly free. LangChain's `AgentExecutor` is legacy and opaque —
poor for a submission whose stated bar is *"you should be able to explain every
architectural choice"*. LlamaIndex is genuinely stronger at retrieval, but
retrieval is the *easy* part at 56 documents; its agent story is weaker, and
adopting both frameworks would double the surface area I have to defend.

**The trap I avoided:** LangGraph lets you build a fixed A→B→C graph, which would
be a hard-coded pipeline wearing a costume. The graph here has one decision
function (`route`) and one cycle.

### 1.5 Deterministic quantum

Compensation under the Motor Vehicles Act is a *formula* settled by binding
authority. An LLM that free-hands "approximately ₹45 lakhs" is useless to a
litigator. `compute_quantum` implements Sarla Verma (multiplier table, dependency
deduction) and Pranay Sethi (future prospects, conventional heads) in Python, and
returns the governing precedent for **each step**.

On the brief's facts (age 42, ₹35,000/month, 3 dependants) it computes
**₹50.99 lakh**, range **₹51.0–52.96 lakh**, plus interest.

**Is this hard-coding?** No — and the distinction matters. It is parameterised
over any claimant, the agent chooses whether to call it at all, and nothing about
Mrs. Lakshmi Devi appears in it. Hard-coding would be baking *her facts* into the
code; this is providing a domain *capability*, like a calculator.

---

## 2. How the agent decides simple vs. deep

The brief asks this specifically, and "the LLM decides" is not an answer. Three
mechanisms, none of them an `if`:

**1. Triage sets a starting allowance.** One cheap call classifies the request as
`simple` or `research`, setting a step budget (4 vs 14). When genuinely
ambiguous it is instructed to choose `research`, because under-researching is the
costlier error. If triage fails, the run defaults to `research` — a failure must
never silently make the agent shallower.

**2. The budget is escalatable, not a gate.** This is the important part. On
reaching its limit, the agent continues — up to a hard ceiling of 22 — *if it is
still discovering documents it has not seen*. A run that is still finding new law
gets more room; a run repeating itself is stopped. Depth is therefore governed by
observed productivity, not by a keyword match on the prompt.

**3. The output contract is itself a tool choice.** `submit_research_report`
(supporting / adverse / strategy) versus `submit_answer` (prose + citations). The
agent *elects* which shape to emit. So the mandated three-part structure is
guaranteed when relevant, without any code path forcing every query through it —
and a question like "summarise doc_003" is not inflated into a strategy memo.

---

## 3. Making reasoning visible

A typed event stream (`trace.py`): `plan`, `tool_call`, `retrieval`, `filter`,
`screen`, `read`, `compute`, `budget`, `answer`, `error`.

Retrieval events carry the **full score decomposition** — dense rank, BM25 rank,
fused rank, rerank score, and the reranker's stated reason per document — rather
than a final ordering. Streamlit renders this as a live table.

The design decision worth naming: **the evaluation framework consumes the same
trace objects.** Retrieval recall is computed from retrieval events; the
retrieved-vs-cited gap comes from comparing the trace against the terminal
contract. One artifact serves the demo and the measurement, so the thing that
makes the UI convincing is the thing that makes the metrics real.

---

## 4. Evaluation methodology

### 4.1 The gold set — how I know the agent isn't missing judgments

Standard RAG evaluation pools the top-k output of the system under test and
labels only that. Recall measured that way is circular: you cannot count what
nothing retrieved.

With 56 documents that compromise is unnecessary. **Every document is labelled
against every query** — the full 56 × 15 matrix, 840 pairs — so the recall
denominator is *exact*.

Three tiers, escalating cost only where it buys something:

| tier | what happens |
|---|---|
| 1 | Two **independent** annotators grade all 56 documents per query — different models, differently-worded prompts, so agreement measures label reliability rather than prompt determinism |
| 2 | **Cohen's κ** computed and published per query; low agreement is a disclosed property, not a hidden one |
| 3 | Every disagreement is re-judged against **full judgment text** — full text is spent exactly where the cheap signal was ambiguous |

Grades are graded, not binary: `0` irrelevant, `1` related, `2` directly on
point. Each label also carries the side it favours, which is what makes
Dimension 4 measurable at all.

### 4.2 Query set

15 queries across six kinds, chosen against the *measured* corpus: `structured`
(enumerative), `research` (multi-step), `adverse` (must surface damaging
authority), `flip` (same matter from the opposing side), `distractor` (answer
lives in the off-topic subset), `absent` (nothing in the corpus answers it — the
correct response is "none").

The `absent` query matters: a system that cannot say *"I don't have that"* will
manufacture authority, which in legal practice is the worst possible failure.

### 4.3 The four dimensions

| dimension | how it is measured | why this way |
|---|---|---|
| **1 Precision** | P@k over cited docs, nDCG@10 (graded), **citation faithfulness**, hallucinated-citation count | Relevance alone is insufficient — a fabricated or misstated citation is worse than a missing one, so faithfulness is scored as hard failure |
| **2 Recall** | **retrieval recall** and **answer recall**, reported separately; their difference is *synthesis loss* | Most evals conflate these and hide the commonest real failure: the system finds the right case, then drops it |
| **3 Reasoning** | rubric judge (5 criteria, must quote evidence) + **poisoned-context probe** + self-consistency across seeds | Attacked from three directions rather than trusted to one judge; the judge runs on a *different model* from the agent to avoid self-preference bias |
| **4 Adverse** | adverse recall, **buried** count, risk-calibration entropy, **sycophancy flip** | "Found an adverse case" ≠ "handled adverse authority honestly" |

Two probes are worth singling out:

**The sycophancy flip.** The same matter is posed from the claimant side and the
insurer side. The relevant *law* should barely move — only its labelling should
flip. A system whose analysis changes with who is asking is broken, and this is
the single most on-thesis check for a legal product.

**Buried adverse.** The agent *retrieved* a damaging judgment and then left it out
of the adverse section. That is materially more dangerous than never finding it,
because the output looks complete. It is counted separately.

---

## 5. Tradeoffs I made, and what they cost

| decision | gained | gave up |
|---|---|---|
| LLM enrichment as the index | exact structured queries, working adverse retrieval | quality ceiling set by one extraction pass; 56 API calls to rebuild |
| Small open embedding model | free, CPU-only, deployable, Apache-2.0 | ~9 points nDCG vs proprietary legal-specific embedders |
| LanceDB embedded | zero infrastructure, satisfies the brief's constraint | no horizontal scaling path without migration |
| Card-level annotation (tier 1) | complete 56 × 15 coverage affordably | cards mediate the labels; mitigated by tier-3 full-text adjudication |
| One framework (LangGraph) | small surface to defend, readable | hand-wrote retrieval LlamaIndex gives free |
| Provider isolated to one module | swapped Gemini → Azure DeepSeek without touching agent, tools, retrieval or evals | a thin indirection layer to maintain |
| Two model families (DeepSeek + Kimi) | independent judge and independent second annotator | two deployments to keep alive; global rate limit set by the slower one |

### A note on provider isolation

This system was built against Gemini and moved to Azure-hosted
**DeepSeek-V4-Flash** mid-build, after the Gemini free tier turned out to meter
**20 requests per day per model** and the project was then suspended outright.

The migration touched exactly one file — `llm.py` — plus model names in config.
The agent loop, tools, retrieval, trace and all four eval dimensions were
unchanged. That was not luck: everything depends on a narrow `LLM` interface
(`complete`, `structured`, `_client`) rather than on a provider SDK. Worth
recording because provider churn is the norm, not the exception.

DeepSeek-V4-Flash was chosen on three requirements: **native tool calling**
(non-negotiable — the agent is a tool loop), a **1M-token context** that holds
the largest judgment (229 K chars) whole, and quota that permits a real
evaluation run.

---

## 6. What changes at 5,000 documents

Measured baseline: **56 judgments → 1,629 chunks (29 per judgment), 1024-dim
vectors, 13 MB on disk.** Scaling 89× gives ~145,000 chunks and ~1 GB of index.
The useful surprise is that **index size is not what breaks** — 5,000 card
vectors is 20 MB and scans in milliseconds, and chunk search is always
pre-filtered to a handful of `doc_id`s. What breaks is the *retrieval model*,
two silent-truncation bugs, and the offline pipeline's shape.

### 6.1 Architecture: one vector per judgment stops being enough

This is the real change, and it is not about infrastructure.

Today `dense_search` matches a query against **one embedding per judgment**,
built from the card's holding, ratio, issues and statutes. At 56 documents that
is not merely adequate, it is *better* than chunk retrieval — it matches legal
substance instead of recital boilerplate, and the LLM reranker cleans up the
handful of near-misses.

At 5,000 it fails, for a specific reason: a judgment that decides five issues
across forty pages is compressed into a single point in embedding space. A query
about its *third* issue will not pull it, because the card vector is dominated by
the primary holding. At 56 documents that judgment still surfaces — the corpus is
small enough that it lands in the top 30 anyway. At 5,000 it is buried under
several hundred judgments whose *primary* holding matches the query better.

So candidate generation becomes **two-tier**:

1. **Chunk-level ANN recall** — search the 145,000 chunk vectors (IVF-PQ, ~100
   probes), take the top ~500 chunks, and roll them up to their parent
   judgments, scoring each judgment by its best-N chunks. This finds the
   third-issue case that the card vector misses.
2. **Card-level scoring of the survivors** — the existing dense + BM25 + RRF
   path, now running over ~200 candidate judgments rather than all 5,000, with
   the card vector acting as a *relevance prior* rather than the sole signal.
3. **Cascade rerank** — a cheap cross-encoder over the ~200, then the existing
   LLM reranker over the top ~20. Today's LLM rerank of 20 documents costs one
   call; running it over 200 would cost ten and dominate latency.

The card layer does not go away — it becomes the *filtering and reasoning*
substrate rather than the *recall* substrate. That inversion is the headline
architectural change.

### 6.2 Code: what I would change, file by file

- **`retrieve.py::all_card_summaries()` — delete the caller.** It loads every
  card into one prompt to feed `screen_corpus`. Measured: 56 cards = 17,579
  characters ≈ **4.4K tokens**; the same code at 5,000 judgments produces
  **≈392K tokens** — past any context window, and absurd per query even where
  it would fit. `screen_corpus` is replaced
  by *staged screening*: apply the structured filter first, then screen only the
  matching subset, in batches, with the agent told how many batches exist.
- **`retrieve.py::filter_cards(where, limit=200)` — the silent-truncation bug.**
  A predicate matching 2,000 judgments today returns 200 with no indication that
  anything was dropped. At 56 documents no filter can exceed the cap, so the bug
  is invisible; at 5,000 it makes `filter_judgments` quietly wrong on exactly the
  enumerative queries it exists to answer correctly. Fix: return a total count
  alongside the page, and make the tool surface *"2,041 matched, showing 200"* so
  the agent can narrow rather than confidently under-report. **This is the change
  I would make first** — it is a correctness bug, not a performance one.
- **`retrieve.py::_in_clause()` — string-built `IN (...)`.** Fine for 20 ids;
  at scale it produces multi-kilobyte predicates and invites injection through
  doc_ids. Replace with a parameterised filter or an `isin` pushdown.
- **`index.py::build_index()` — add an ANN index and stop rebuilding wholesale.**
  Today it is a full rebuild with `reset=True`, and brute-force search with no
  vector index (correct at this size — ANN would be slower *and* approximate).
  At 5,000: `create_index(metric="cosine", num_partitions≈380, num_sub_vectors=64)`
  on the chunk table, and incremental upsert keyed on a content hash so adding
  200 judgments does not re-embed 5,000.
- **`enrich.py` — the checkpoint strategy inverts at scale.** Enrichment
  already checkpoints after every document and resumes cleanly (built that way
  because a free-tier quota kept interrupting runs). The problem is *how*:
  `_save(cards)` rewrites the entire cards file after each document, so 5,000
  documents means 5,000 full rewrites of a growing file — O(n²) I/O that is
  invisible at 56. Append-only or per-document rows instead. Also missing:
  `card_schema_version` and `enriched_by_model` stamps, without which a
  re-enrichment cannot tell stale rows from current ones — and this corpus was
  in fact enriched under an earlier model than it now runs on (§8). Finally,
  `max_workers=3` is tuned to a quota that no longer applies, and failures are
  collected but never retried; at 5,000 that needs a dead-letter queue.
- **`agent.py` — budget becomes cost-aware.** The step budget counts *calls*, not
  tokens. At 5,000 documents a screening call is far more expensive than a read,
  so the allowance should be denominated in estimated tokens with a per-query
  ceiling, and `filter_judgments` results should page rather than dump.
- **`evals/gold.py` — exhaustive labelling dies.** 5,000 × 15 = 75,000 pairs.
  Replace with TREC-style depth-k pooling across several retrieval
  configurations, report *pooled* recall with its bias stated, and keep the
  deterministic subset (§4.1) as the anchor — the one part that gets *more*
  valuable at scale, since it is the only ground truth no model produced.

### 6.3 Pipeline: batch → incremental

The current pipeline is a straight line run by hand: `ingest → enrich → index`,
all-or-nothing, ~5 minutes of embedding on an M-series GPU. At 5,000 documents
that same path is ~7 hours of embedding alone, which makes "rebuild it" an
unaffordable answer to any mistake. Three changes:

1. **Content-hash idempotency.** Every stage keyed on a hash of the source PDF,
   so re-running processes only what changed. Today `build_index(reset=True)`
   discards everything.
2. **Enrichment becomes a queue, not a loop** — with dead-lettering for the
   documents that fail schema coercion (4 of 56 did, initially) rather than a
   best-effort skip.
3. **Corpus changes become versioned events.** At 56 the corpus is fixed. At
   5,000 it grows weekly, and the eval numbers stop being comparable across
   runs unless the gold set is pinned to a corpus version.

### 6.4 What becomes newly necessary

- **Negative-treatment detection.** The largest correctness gap. At 56 I can eyeball
  authority; at 5,000, citing an overruled judgment is a
  professional-negligence-grade error and nothing here detects it.
- **Citation-graph weighting.** Judgments cite each other; PageRank-style
  authority becomes a real ranking signal once the graph is dense enough.
- **Court hierarchy as a retrieval-time prior.** Supreme Court > High Court >
  Tribunal. At 56 the agent weighs this from the card; at 5,000 it must be in
  the ranking function.
- **Per-query cost as a first-class SLO.** Currently latency is nobody's problem
  (§8 measures ~300K tokens on a deep brief). At 5,000 it is the product.

### 6.5 What survives unchanged

The trace design, the tool interface, the agent loop, the four eval dimensions,
the retrieved-vs-cited separation, and the terminal-contract validation. **The
agent layer is scale-invariant; everything that breaks is in the retrieval and
data layers** — which is the right place for the damage to be, since those are
replaceable behind the same tool signatures.

---

## 7. What I would do with another week

Ordered by expected value, not by ease:

1. **Adverse *retrieval* breadth on insurer-side matters.** The reporting half
   is now fixed and shipped: triage-resolved side selection cut burial from 11
   to 4 on the insurer-side query and raised the held-out insurer query to 0.9
   broad / 1.0 strict (§8). What remains is coverage — on that query the agent
   retrieves only a fraction of the outright claimant wins that exist, so its
   strict recall is capped by what enters its context, not by how it
   categorises. Two levers, in order: raise `search_precedents` breadth when the
   adverse pool is known to be large, and improve the enrichment layer's
   `outcome_favours` precision, since that field is what any adverse query
   ultimately filters on.
2. **Sub-agent reads, then a prompt-caching check.** Cost is quadratic in
   tool calls and three history-compression variants have now measurably
   failed (§8) — the family is closed. The structural fix is delegation: a
   disposable reader context opens the judgment and returns only its extract,
   so full text never enters the loop's history and there is nothing to
   re-rent. Cheaper still would have been provider
   prompt caching -- DeepSeek's native API bills repeated prefixes at ~10% --
   but the check was made (identical 1.2K-token prefix sent twice;
   `prompt_tokens_details` null both times): Azure's serving does not expose
   it. Delegation stands alone as the successor.
3. **Decouple quote verification from the revision budget.** Verifying quotes is
   the right idea -- it converts a judged criterion into a deterministic one --
   but sharing bounded revision rounds with the adverse gates cost 0.5 adverse
   recall. Run it as a post-hoc sanitiser instead of a gate.
4. **Negative treatment detection.** The largest correctness gap in the system.
   Right now the agent will cite a judgment that a later court overruled, with
   full confidence and no warning.
5. **Human-validated judge.** Dimension 3 currently rests on an LLM judge that is
   itself unvalidated. I would hand-label ~50 reasoning samples with a lawyer,
   report judge-human correlation, and recalibrate. Without this the reasoning
   score is a number with unknown units.
6. **Span-level citations.** Every claim should link to the exact sentence in the
   judgment that supports it, checked by entailment. Faithfulness is currently
   measured at document level, which is too coarse.
7. **Multi-hop retrieval.** Judgments cite other judgments; several cite *Swaran
   Singh*. Following citations *inside* the corpus would materially improve
   recall on doctrinal questions.
8. **Adverse-coverage self-critique.** A dedicated pass whose only job is to ask
   "what did we not look for?" before the report is emitted.
9. **Quantum validated against decided cases.** 25 judgments state their own
   multiplier and award; I would back-test the calculator against them and report
   the error distribution. That converts a plausible tool into a measured one.
10. **Fine-tuned reranker** on legal relevance pairs harvested from the gold set.

---

## 8. Results, and the decisions the measurements settled

Full output in [`evals/results/report.md`](evals/results/report.md).
15 queries × 56 judgments = **840 labelled pairs**, mean Cohen's κ **0.748**,
74 disagreements adjudicated against full judgment text.

| dimension | metric | tuned | held-out |
|---|---|---|---|
| **1 Precision** | precision of cited precedents | 73.0% | 53.0% |
| | nDCG@10 (graded) | 0.740 | 0.576 |
| | citation faithfulness | 72.6% | — |
| | **hallucinated citations** | **0** | **0** |
| **2 Recall** | **retrieval recall** | **93.1%** | **100%** |
| | **answer recall** | **82.5%** | **81.7%** |
| | *synthesis loss* | 10.6% | — |
| | Evidence Score | 0.931 | 1.000 |
| **3 Reasoning** | rubric (independent judge) | 67.1% | — |
| | **poisoned-premise probe** | **passed** — rejected the falsehood, opened the judgment, corrected the premise | — |
| **4 Adverse** | **adverse recall** (broad / strict) | **51.2% / 47.9%** | 33.3% |
| | **buried** | **4** | **1** |
| | miscast (opposing-side wins presented as supporting) | **0** | — |
| | **risk-calibration entropy** | **0.904** | — |
| **5 Behaviour** | run success | **100%** (21/21, zero failures) | — |
| | trajectory / abstention | **100% / 100%** | — |
| | output-contract accuracy | 75.0% (all misses low-severity over-delivery) | — |
| | cost | 20.5 tools · 303K tokens · 286 s mean per query | — |
| **Gold set** | annotator accuracy (504 verifiable labels) | A 91.1% / B 90.1% | — |
| | key accuracy after computed override | **95.2%** | — |

Numbers are the final single-seed run on the settled configuration; run-to-run
MoE variance is bounded under failure mode 4 below, so treat single digits after
the decimal as weather. The held-out adverse figure is measured against the
*corrected* h05 key — the earlier 32.5% was partly scored against an inverted
pool and is not comparable.

### Decision: "adverse" means *the opposing side will cite this*

Adverse identification is the brief's stated thesis — *"a system that only finds
favorable cases is dangerous in legal practice"* — and it is the decision this
system's design turns on. Three candidate mechanisms were measured before the
definition itself turned out to be the lever:

| attempt | adverse recall | verdict |
|---|---|---|
| automatic counter-search in retrieval | 16.7% → 11.7% (held-out) | reverted |
| additional output-validation gates | 0.333 → 0.111 | reverted |
| **rewriting the definition of "adverse"** | **0.181 → 0.648** | **kept** |

The first two were machinery. The third was four lines of prompt.

*Provenance note: the gate row is re-derived from stored runs under the
corrected adverse metric. The redefinition row is quoted as originally measured
— those pre-fix full-run artefacts were overwritten by later evaluations, so it
cannot be re-derived, and under the corrected metric the same configuration now
measures **0.512** (§ headline table). The direction and magnitude of that
result are not in doubt — it is the only change that improved every held-out
metric simultaneously — but the exact pair of numbers belongs to the older
ruler, and I would rather say so than quietly restate them as current.*

The diagnosis that made it obvious came from the evals themselves: held-out
retrieval recall was already **100%**, so the agent was seeing every damaging
judgment. It simply was not reporting them -- because "adverse" had been defined
as *"this defeats us"*. Under that test, correctly distinguishing a judgment
("gratuitous passenger, not a third-party road user") disqualified it, and it was
filed under `caveats`. Legally sound, practically useless: opposing counsel cites
it regardless.

Redefining adverse as **"the opposing side will cite this"** -- with risk level
measuring *how hard it is to answer* rather than whether you win -- tripled the
score and raised risk-calibration entropy from 0.720 to 0.923, while burial stayed
at zero. Answer recall rose with it (71.3% → 87.0%): the same narrowness had been
suppressing citations generally.

**And it is the only one of the four that generalised.** On the held-out set --
six queries written after the agent stopped changing, never tuned against -- every
metric improved against the pre-fix run:

| held-out metric | before | after |
|---|---|---|
| adverse recall | 0.117 | **0.325** |
| precision | 0.474 | **0.560** |
| answer recall | 0.744 | **0.856** |
| buried | 6 | **2** |

The three mechanical fixes each made held-out *worse*. This one improved all of
it. That is the difference between a change that fits a query set and a change
that fixes a specification.

It is still only a partial fix: on the final full run adverse recall is 51.2%
tuned against 33.3% held-out, so roughly half the gain transfers and unseen
queries remain the weaker case. The direction is right and the magnitude is not
yet sufficient — the triage-side successor (next sections) moved the remaining
failure from categorisation to retrieval breadth.

**The lesson: I spent hours building mechanisms to enforce a behaviour I had
defined wrongly.** Retrieval recall on held-out was already 100% -- the agent was
seeing every damaging judgment throughout. I had misdiagnosed the layer, and
built machinery at the output to compensate for a definition that was wrong at
the input. Prompt semantics are part of the architecture, not a detail below it,
and the cheapest fix was the last one I tried.

### Decision: which output controls ship, and which do not

Four candidate behavioural controls were measured **in isolation** by ablation
(5 queries × 1 seed, one variable per arm). Only one earned a place in the
shipping configuration:

| arm | precision | nDCG | adverse recall | buried |
|---|---|---|---|---|
| **baseline** (all off) | **0.769** | **0.632** | **0.333** | 8 |
| + synthesis check | 0.705 | 0.602 | 0.306 | 8 |
| + quote verification | 0.689 | 0.550 | **0.111** | 7 |
| + adverse gates | 0.676 | 0.608 | 0.139 | **0** |
| all on | 0.565 | 0.344 | 0.333 | **0** |

*The adverse column here was **re-derived** from the stored run files after the
adverse metric itself was corrected (§"What the evaluation caught in itself" —
the original version scored adverse relative to a hardcoded side rather than to
whoever was asking). Precision, nDCG and burial are unchanged from the original
run; only the adverse figures moved, and they moved because the ruler was wrong,
not the runs. Every number above is recomputable from `evals/runs_ablate_*/`.*

Read plainly: **baseline beat almost every arm.** The synthesis check lowered
precision *and raised* synthesis loss, the exact opposite of its purpose. Quote
verification cut adverse recall by two-thirds (0.333 → 0.111), because it shares
a bounded revision budget with the adverse gates and starved them — two checks
competing for the same rounds, an interaction neither was designed against.

Only the adverse gates earned their place: they are the sole arm that reaches
**burial = 0**, for about nine points of precision. (Their own adverse-recall
figure falls under the corrected metric — the gates change *what gets reported*,
not what gets hunted, which is exactly the gap the side-resolution fix below
addresses.) That trade is worth taking,
because a reader can discount a marginal citation but cannot discount one they
were never shown.

Turning the two harmful fixes off produced the largest single improvement of the
project: **precision +20.6, nDCG +38.8, contract accuracy +25.0.**

**Why this was nearly missed.** All four were changed together and evaluated in
one run. Several metrics moved, and the gains from the adverse gates were
credited to the other two -- I reported "+7.9 precision, +12.0 recall, kept" for
changes that, measured alone, were harmful. Full cost, wrong conclusion. The
ablation harness (`evals/ablate.py`) exists so that cannot recur: one variable
per arm, tuned and held-out metrics in the same table, ~6 minutes instead of 40.

### Decision: resolve the client's side once, at triage

The fixed adverse rule hardcodes a hunt direction (`favours='insurer'`), which is
backwards when the client *is* the insurer. The full-set numbers isolate what
that asymmetry costs: on the insurer-side flip query,
**every one of the run's 11 burials is an outright claimant win the agent had
itself retrieved** — it saw the adverse authority, had no category for it under
the inverted rule, and silently dropped it. Claimant-side queries, by contrast,
run at burial 0.

The first fix rewrote rule 3 to make the agent work out its own side every
turn. Measured: **burial went 0 → 12 on the claimant side** — the queries that
were already working. The conditional appears to spend the agent's bounded
attention deciding which side it is on rather than covering the pool. Correct
diagnosis, wrong mechanism — the second time this project produced that pair
(quote verification was the first). Reverted behind
`enable_side_relative_adverse`.

The successor design moves the reasoning out of the loop entirely: triage
resolves the client's side once, and rule 3 arrives as a pre-resolved constant
with the same shape as the fixed rule that works ("you act for the INSURER;
judgments favouring the CLAIMANT are your adverse list"). The agent never
reasons about sides at runtime. That variant is `enable_triage_side_resolution`,
measured under the same two-arm protocol as everything else in this section —
and it is the one that survived. Same-batch control, both sides in the subset:
adverse recall 0.379 → 0.474, burial 4 → 1 with the claimant side holding at
zero, and the held-out insurer query went 0.4 → **0.9 broad / 1.0 strict** —
the strongest generalisation result in the project. Cost fell with it (tokens
−17%, tool calls 28.7 → 26.0): an agent hunting the right pool stops paying
for the wrong one.

### Decision: no mechanical history compaction — cost ships measured

Instrumenting the heaviest run exposed the system's real cost profile:
**tokens ≈ 600·n²** for *n* tool calls, consistent across 18 measured runs. The
loop resends the whole history every turn, so a judgment read on turn 5 is
retransmitted on all ~28 turns that follow; the full case brief cost 670K
tokens across 33 tool calls. The failure modes turned out to be linked: the
runs dying with `APITimeoutError` were precisely those whose requests had grown
to 60–75K tokens. Some of what I had filed as "provider flakiness" was
self-inflicted load.

The obvious fix — compact old tool results to one line plus "call the tool
again if the full text is needed" — was built, ablated, and produced the
sharpest negative result of the project:

| | full history | compacted |
|---|---|---|
| duplicate reads, worst run | 2 | **41** |
| mean tool calls | 26.5 | **101** |
| total tokens | 457K | **509K** |
| failure mode | 2 timeouts | 2 recursion-limit crashes |

The agent did exactly what the compaction message invited: it lost passages it
was still reasoning from and re-read them — one run re-fetched the same
judgments 41 times, two runs looped until LangGraph's recursion limit killed
them, and net tokens went **up**. The per-request saving was quadratic; the
extra turns it induced were too.

The design principle this bought: **an agent will pay whatever it costs to
recover information it still needs — you can only compress what it is finished
with, or compress losslessly for its purposes.**

The information-preserving version was then built and measured
(`enable_digest_compaction`): stale reads keep their full structured head —
holding, ratio, outcome, statutes, citations — plus each passage's opening, and
the banner never invites a re-read. It eliminated the catastrophic mode (zero
failures, worst-case duplicate reads 6 against the truncation variant's 41) and
cut tokens 21.6%, with −44% on the best query. **It still failed the
pre-registered gate**: duplicate reads rose 2 → 8 against the same-batch
control, two queries paid *more* net tokens than control once their extra tool
calls were counted, and held-out precision fell 0.844 → 0.733. Passage openings
capped at 240 characters evidently still starve some runs.

A third variant then tested that diagnosis directly: the verbatim window made
to count *reads* rather than all tool results (under v2, a read chased by two
searches was digested while the agent was still quoting from it), with gentler
600-character keeps. Quality recovered exactly as the diagnosis predicted —
precision, adverse recall and both held-out rows held — **and the savings
vanished**: tokens rose 6.4%, because one extra tool call at a ~28K-token
request outweighs everything a safe digest saves.

That completes the curve. Three settings of the aggressiveness dial: the
aggressive end is cheap and destructive, the safe end is harmless and free of
savings, and the midpoint fails both gates. **Mechanical history compression is
not a lever for this system** — a measured conclusion, not a hunch. All three
variants stay reproducible behind their switches; cost ships at ~600·n² with
the reliability fixes making it survivable rather than fatal, and the
structural successors — sub-agent reads that keep full judgment text out of
the loop's history entirely, and provider-side prompt caching — are specced in
§7.

What did ship from the investigation is the reliability half. The tool-bound
client bypasses `LLM.complete()` and its retry loop, so the agent's own turns
had **zero retries** — a single timed-out request killed the whole run,
discarding 20+ tool calls of gathered state. Agent-path calls now retry
transient failures with backoff (`_invoke_retrying`, content-filter errors
excluded), the timeout is 600s, and the agent path now takes rate-limiter
slots like every other call (a 429 in one worker pauses all of them -- the
per-minute bucket is shared, so one worker sleeping while two keep spending
clears nothing). In the confirmation arm both queries that had died at 300s
completed, and the final full pipeline then ran **21 of 21 queries without a
single failure** -- the first clean sweep of the project.

### Decision: routing stays budget-based, and adverse coverage is not a
retrieval problem

Two further candidates were measured against the two open gaps. Both were
diagnosed from the traces, both behaved exactly as designed at the layer they
targeted, and neither cleared its gate — which located the real constraint more
precisely than a success would have.

**Contract over-delivery.** `q06`, `q09` and `q10` each carry a second clause —
*"…and what proposition does each take from it"*, *"…and at what percentage"* —
and triage read that as comparison, classifying them `research`. `q12`, the same
shape without a second clause, was classified `simple`. Teaching triage that an
enumeration-with-detail is still an enumeration fixed the classification exactly
as intended (all four dropped to budget 4, verified directly). It did not fix the
contract: mean tool calls stayed flat at 17.8 → 18.4, tokens rose **18%** against
an intended 15% cut, and the single contract that flipped was `q12` — a query
observed flipping unprompted between runs, so the gain is unattributable.
**Triage sets the starting allowance; the output contract is a separate tool the
agent elects, and the agent escalates its own budget regardless.**

**Insurer-side adverse retrieval.** The trace showed the real defect plainly: on
`q05`, where the client *is* the insurer, the agent ran
`filter_judgments(outcome_favours='insurer')` — its own supporting side — and
never queried the claimant side at all. One correct call returns **16 of 16** of
that adverse pool, so the gap was behavioural rather than a retrieval limit.
Naming the exact call in rule 3 worked at the layer it targeted: the agent began
sweeping all three pools, confirmed in the traces. It still failed:

| | off | on |
|---|---|---|
| **buried** | **2** | **14** |
| adverse recall | 0.405 | 0.402 |
| tokens | 481K | **771K (+60%)** |
| `q01` strict recall *(claimant-side control)* | **0.667** | 0.333 |

On `q05` the agent surfaced 28 claimant judgments and flagged four, leaving
thirteen buried. **It found the damaging authority and did not report it** —
strictly worse than never retrieving it, and burial is the failure the adverse
gates exist to prevent.

**The pattern, stated once because it cost three experiments to learn:** every
one of these fixes was aimed at the layer where the *symptom* was visible rather
than the layer that *constrains* the outcome. Triage was not the reason a report
was over-delivered; retrieval was not the reason adverse authority went
unreported. Both are bounded by the terminal contract — how many adverse slots
the agent is willing to fill and which tool it elects to finish with. Widening
the funnel upstream simply increases what falls out of it. Correct diagnosis does
not identify the controlling layer, and I have now made that mistake three times
in one project: once with the per-turn side rule, once here with triage, once
here with retrieval.

### What generalises, and what does not

The held-out set (six queries written after the agent stopped changing, never
tuned against) splits the system cleanly:

| generalises | does not |
|---|---|
| **retrieval recall 93.1% → 100%** | precision 73.0% → 53.0% |
| **Evidence Score 0.931 → 1.000** | nDCG 0.740 → 0.576 |
| **answer recall 82.5% → 81.7%** | adverse recall 51.2% → 33.3% |

**Retrieval is genuinely solved** — perfect recall on unseen queries, from an
architecture never tuned against them, and it has generalised in every
configuration measured all project. It is the one component whose quality is
backed by unbiased evidence rather than by the queries it was developed
against. Answer recall now transfers with it.

**Precision and adverse coverage do not transfer**, and the precision gap has
held steady across every configuration tested — so it is a property of the
design, not of any particular fix: on a query it has not seen, the agent cites
roughly twice what the gold set counts as relevant.

### One structural bug the held-out set exposed

All six held-out burials came from a single query, and from one cause: the agent
did fourteen tool calls of genuine research, then answered **in prose without
calling a terminal tool**. Every validation gate lives inside
`submit_research_report` / `submit_answer`, so a prose reply bypassed all of them
-- and no citations were recorded, so everything it had retrieved counted as
buried.

The router now refuses to end a run that way while budget remains, and pushes
back for a terminal contract (bounded, so termination is guaranteed). Measured on
the same query afterwards: prose → `PrecedentResearchReport`, 0 citations → 6,
gates reached 0 → 2.

The general lesson, and the one worth carrying: **a check attached to a tool is
only enforced on runs that call that tool.** Validation belongs on the control
flow, not on the happy path.

### The headline: retrieval works, synthesis leaks

Retrieval recall of **93.6%** against an exact denominator says the hybrid layer
is doing its job — it surfaces nearly everything relevant. Answer recall is
**78.7%**. The 14.9-point gap is judgments the agent *read and then dropped*.

That is the single most useful number in this report, and it only exists because
retrieval and synthesis are scored separately. An eval that reported one blended
"recall: 78.7%" would have sent me to tune the retriever, which is the half that
is already working.

### Where it fails, in priority order

**1. Precision does not generalise.** 53.0% held-out against 73.0% tuned on the
final run, and the gap has held across every configuration tested -- including ones where tuned
precision moved by 20 points. It is a property of the design, not of any fix. On unseen queries the agent cites roughly twice what
the gold set counts as relevant. This is the largest real gap and it is a design
property, not a bug I introduced.

**2. Adverse coverage is real but partial, and the residue is precisely
located.** 51.2% tuned against 33.3% held-out on the final run, with burial 4
and 1. Strict adverse recall -- outright opposing-side wins, measured relative
to who is asking -- is 47.9%. What remains is concentrated on the insurer-side
flip query: the triage-side fix cut its burial from 11 to 4 and doubled what it
flags, but its strict recall is capped by *retrieval breadth* -- most of the 16
outright claimant wins never enter its context at all. The next lever is
retrieval, not categorisation.

**3. Reasoning is mediocre and groundedness is the weak criterion.** Rubric
67.1% on the final run, with `grounded_in_source` and `distinguishes_properly`
tied for weakest at **53.8%** — the agent asserts things not clearly traceable
to the judgment, and distinguishes adverse authority less sharply than a
litigator would. The obvious fix (verifying quotes) was built, measured,
and found to *destroy* adverse recall by consuming the shared revision budget. It
is switched off pending a redesign that does not compete for those rounds.

**4. Behavioural consistency is poor, and it is model-imposed.** Identical prompts
at `temperature=0` return different content -- verified directly: three calls
produced 704, 480 and 974 characters of unrelated text. DeepSeek-V4-Flash is a
Mixture-of-Experts model, and expert routing depends on server-side batching, so
determinism is not available at any temperature. Measured self-consistency across
seeds: Jaccard 0.60-0.80. The rubric score alone spans 0.569-0.754 across repeated
runs of unchanged code, which makes any single-run threshold a measure of luck.
The fix is majority voting across runs, at 3x cost -- a deliberate reliability/cost
decision rather than a bug.

**5. The adverse fix cost precision and reasoning.** Precision fell 77.1% →
67.5% and the rubric 65.4% → 60.0% when adverse recall tripled *(measured at the
time of that change; both have since recovered to 73.0% and 67.1% on the final
configuration, so the trade was real but not permanent)*. That is the expected
direction -- citing more judgments admits more marginal ones -- and it is the
trade I would make again in this domain, because a reader can discount a weak
citation but cannot discount authority they were never shown. It should be stated
as a choice rather than presented as a free win.

**6. Four of thirty runs failed in the pre-fix configuration**, split between
timeouts and content-filter refusals. The token investigation reframed the
timeout half: requests grown to 60–75K tokens under quadratic history growth
were brushing a 300s limit **with zero retries on the agent path** — the
tool-bound client bypasses `LLM.complete()` and its backoff, an integration gap
the eval had been reporting as provider flakiness. Fixed with agent-path
retries and a 600s ceiling; both previously-failing heavy queries now complete.
Content-filter refusals remain provider policy (Azure's `DefaultV2` trips on
judgments describing fatal collisions and counterfeiting); they are handled by
paraphrase-and-continue, and a relaxed filter would be the right production
answer.

### Cross-checking the evaluation against RAGAS

A bespoke evaluation invites a fair objection: *you wrote the ruler, so of course
you measure well.* Three defences were already in place — the 504
deterministically-computed labels, the held-out query set, and the oracle tests.
This adds a fourth: score the same stored runs with an off-the-shelf framework
and see whether it agrees. Code in
[`evals/ragas_crosscheck.py`](evals/ragas_crosscheck.py), results in
`evals/results/ragas_scores_subset4.json`.

| query | RAGAS faithfulness | RAGAS ctx-precision | our precision | our nDCG | our recall |
|---|---|---|---|---|---|
| `q01_brief` | 0.082 | **1.000** | 1.000 | 0.968 | 0.818 |
| `q04_adverse_licence` | 0.135 | **1.000** | 0.429 | **0.000** | 1.000 |
| `q12_s166` | **0.000** | **0.000** | 0.947 | **1.000** | 0.783 |
| `q14_summarise` | 0.622 | 0.639 | 1.000 | 1.000 | 1.000 |

*(Four queries, chosen to cover each behaviour the agent has — deep research,
adverse, enumerative, point-lookup. Two more from an earlier full-set attempt:
h01 0.197/1.000, h02 0.043/0.792. Scoring all 21 costs ~300 judge calls — 20×
this project's entire Dimension 3 — because RAGAS decomposes every claim against
every context; measured at 2–4 hours, which is not a good trade for a
cross-check.)*

**Verdict in one line: where both frameworks measure the same thing, they
agree; where they diverge, RAGAS's assumptions are what break.**

| | agree? | |
|---|---|---|
| retrieval quality | **yes** | RAGAS 1.000, ours 1.000 |
| point-lookup (`q14`) | **yes** | RAGAS 0.62/0.64, ours 1.000 — same direction |
| enumeration (`q12`) | no | RAGAS 0.000, ours 0.947 — it cannot score a 19-item id list |
| ranking vs. what-was-read (`q04`) | no | opposite directions, both correct (below) |
| groundedness | no | different question: sentence-level vs citation-level |

**Retrieval is independently corroborated.** `context_precision` is 1.000 on both
research queries: a framework with no knowledge of our gold set agrees the
passages the agent read were the right ones. That is the claim the cross-check
was run to test, and it holds.

**Two disagreements, in opposite directions, and both are informative.**

On **`q12_s166`** we measure 0.947 precision and nDCG 1.000; RAGAS scores 0.000
on both. The answer is a list of nineteen `doc_id`s. RAGAS decomposes an answer
into claims and verifies each against the retrieved passages — a `doc_id` list is
not a claim set, and four passages cannot substantiate statements about nineteen
documents. This is a **metric mismatch, not a measurement**: the framework is
built for short extractive answers, and an exhaustive enumeration is the one
output shape it cannot score.

On **`q04_adverse_licence`** the disagreement runs the other way and is more
interesting: our nDCG is **0.000** while RAGAS's context-precision is **1.000**.
Both are correct. Our nDCG scores the *search ranking* — nothing graded-relevant
reached the top ten. RAGAS scores the passages the agent *actually read*, which
were all useful. The agent recovered from a bad ranking using its other tools,
and answer recall on that query is 1.000. **That gap is the argument for the
agent loop over a fixed retrieve→generate pipeline, measured**: a pipeline would
have inherited the bad ranking with no way out.

**Faithfulness does not transfer to this task.** It averages 0.210, and the
reason is structural rather than a defect in the agent. A precedent report is
supporting analysis, *adverse argument*, and *strategy* — the last two are by
definition not present in the retrieved text, because inventing the opponent's
argument and recommending a course of action is the work. An answer scoring 1.0
on faithfulness would be one that only paraphrases its sources. Our own
groundedness measures are citation-level (72.6% faithfulness, 53.8%
`grounded_in_source`), which asks the answerable question: is each *citation*
traceable to the judgment it names.

**What it cost to run, which is itself a finding.** RAGAS could not be installed
alongside this project — it downgraded `openai` (3.1.0 → 2.54.0) and pulled in
`langchain`, `langchain-community`, `datasets` and `instructor`, then failed to
import against the `langchain-community` it had just installed. It runs in a
separate virtualenv on a pinned pair (`ragas==0.4.3` +
`langchain-community<0.4`; 0.2.15 is broken on Python 3.14). Three further
failures had to be engineered around, and two of the three were problems this
codebase had already solved for itself:

- **Azure's content filter blocked the judge's own output** on answers
  containing fatal-accident detail (`MultiSeverity_ViolenceScore`) — the same
  filter `llm.py` handles with paraphrase-and-continue. Worked around by scoring
  the answer in segments; all four queries reached 100% segment coverage.
- **147 `APIConnectionError`s from socket exhaustion**, because each
  `evaluate()` call builds its own executor and pool — the same failure
  `_build_client`'s `lru_cache` fixes for the agent. Worked around by batching.
- A 900s timeout set to stop timeouts then let one stalled call block for a
  quarter-hour; 180s with more workers was strictly better.

None of that is a reason to avoid RAGAS in general. It is the concrete reason it
was not the primary framework here: it measures a retrieve→generate pipeline
with short answers, and this system is a tool-calling agent emitting legal
briefs. Where the two frameworks measure the same thing, they agree.

### Validating the evaluation itself

An evaluation is a measuring instrument and needs its own calibration. Five
corrections were made to the harness during the build, each found by the oracle
tests, the deterministic subset, or a cross-check — and each would otherwise have
shipped as a confident number:

- **The poison probe was dead code.** Written, rendered in the report template,
  never called. Dimension 3 claimed a check that did not run.
- **The annotators were asked unanswerable questions.** The listing shown to them
  omitted `precedents_cited`, so "which judgments cite Swaran Singh?" was being
  answered by guesswork. They said 6; the text says 13.
- **`buried` was measured on queries with no client.** Enumerative questions have
  nothing to be adverse *to*; q08 literally asks for judgments where the insurer
  won, so those judgments are the requested output, not concealment. The metric
  read 63 buried where the true figure was 15 — a 4× overstatement of the
  system's worst dimension.
- **"Adverse" was not relative to who is asking — twice.** The strict pool
  hardcoded insurer-favouring judgments as adverse, so the insurer-side flip
  query was scored as burying its own supporting authority; and the one
  insurer-side *held-out* query (h05, "I act for the insurer") was labelled
  with the default claimant side, inverting its adverse pool in the held-out
  numbers too. The same conceptual bug, fixed in the metric and then found
  again in a label.
- **`miscast` flagged the same documents against both sides.** The identical
  eight documents were counted as "adverse presented as supporting" on the
  claimant-side brief *and* its insurer-side flip — a logical impossibility,
  since authority adverse to one side supports the other. All 23 counted
  miscasts were `mixed` (pay-and-recover) judgments, which sit in both sides'
  pools by construction; presenting one as supporting is a defensible
  characterisation, not an error. The metric now counts only outright
  opposing-side wins cited as supporting, and reports the mixed count
  separately so the narrowing hides nothing.

### Why the deterministic subset overrides the model labels

On `q06_swaran`, the two annotators reached **κ = 1.000 — perfect agreement — and
were 87.5% accurate.** Both missed seven judgments that verifiably cite the case.

Agreement is not accuracy. This is why the deterministic subset exists, and why
for queries whose relevance test is mechanical the computed key now *overrides*
the model labels (22 corrections, key accuracy 91.5% → 95.2%). Where `grep`
settles the question, `grep` wins. Where it cannot — `q09_multiplier` at
κ = 0.101, `q05_flip` at 0.271 — the report flags the rubric as ambiguous rather
than presenting the score as sound.

### Known limitations

Structural, and independent of any particular scoring:

- **Enrichment is a single point of failure.** Every downstream capability
  inherits its errors. During the build, 4 of 56 cards initially failed on schema
  coercion and 6 on API quota — recoverable, but it shows the dependency.
- **No negative-treatment awareness.** See §7.1. The system cannot tell you that
  a precedent it just recommended has been overruled.
- **The judge is unvalidated.** §7.2.
- **`outcome_favours` compresses a genuinely subtle judgment.** "Pay and recover"
  orders are coded `mixed`, which is right, but the claimant/insurer/mixed
  trichotomy will flatten cases where the result is favourable on liability and
  adverse on quantum.
- **Eight of twelve behavioural changes were reverted after measurement.** Kept:
  the adverse gates, the adverse redefinition, and triage-side resolution.
  Reverted, each behind a switch so the negative result stays reproducible:
  `enable_synthesis_check`, `enable_quote_verification`, `counter_search_k`,
  `enable_side_relative_adverse`, `enable_history_compaction`,
  `enable_digest_compaction`, `enable_enumeration_is_simple`,
  `enable_explicit_adverse_filter`. Defaults are set by the ablation tables, not
  by intuition.
- **Insurer-side adverse retrieval is the open problem.** Strict adverse recall
  (side-relative) is 47.9% overall, but on the insurer-side flip query the agent
  retrieves only a fraction of the outright claimant wins that exist. The
  triage-side fix repaired categorisation -- what it sees, it now reports -- but
  not coverage. Given the brief's thesis, this is the most important open
  problem, and it lives in retrieval.
- **Case cards were enriched under an earlier model** (Gemini flash-lite) before
  the provider migration. They were spot-checked against DeepSeek output on
  sample documents and agreed on holding, ratio, multiplier and citations, but a
  full re-enrichment on the current model would be the clean thing to do.
- **Annotator A still shares a model with the agent.** Annotator B and the judge
  are independent (Kimi-K2.6), which is what makes κ and the reasoning score
  meaningful. Annotator A remains DeepSeek, so a systematic DeepSeek blind spot
  could survive into the labels wherever both annotators happened to agree. The
  tier-3 adjudication pass only fires on *dis*agreements, so it does not catch
  this. A third annotator would.
