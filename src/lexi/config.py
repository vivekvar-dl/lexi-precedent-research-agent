"""Central configuration. Everything tunable lives here, nothing is hard-coded downstream."""
from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_prefix="LEXI_", extra="ignore"
    )

    # --- Paths ---------------------------------------------------------------
    corpus_dir: Path = ROOT / "lexi_research_take_home_assessment_docs"
    index_dir: Path = ROOT / "index"
    lance_dir: Path = ROOT / "index" / "lance"
    cards_path: Path = ROOT / "index" / "case_cards.json"
    chunks_path: Path = ROOT / "index" / "chunks.json"
    gold_path: Path = ROOT / "evals" / "gold.json"

    # --- Models --------------------------------------------------------------
    # Embeddings: best open-source model on MLEB (the legal-domain embedding
    # benchmark) at 77.13 nDCG@10 vs BGE-M3's 69.44. Apache-2.0, 0.6B params,
    # runs on CPU. See ADR "Retrieval" section.
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_dim: int = 1024
    # "auto" picks cuda > mps > cpu. On Apple Silicon, MPS is roughly an order of
    # magnitude faster than CPU for the index build; Streamlit Cloud falls back
    # to CPU automatically.
    embed_device: str = "auto"
    # Kept small on purpose. MPS allocates from the same unified memory as the
    # OS, so a large batch pushes the machine into swap and the build slows by
    # ~6x (measured: 3.3 s/batch -> 19.9 s/batch once swap filled). Small
    # batches cost a little throughput and buy predictability.
    embed_batch_size: int = 8
    # Chunks are ~2,200 chars (~600 tokens); this caps the attention window so a
    # stray long chunk cannot blow up activation memory.
    embed_max_seq_len: int = 1024

    # LLM: DeepSeek-V4-Flash on Azure AI Foundry. Chosen for native tool calling
    # (the agent is a tool loop, so this is non-negotiable), a 1M-token context
    # that swallows the largest judgment whole, and a real quota -- 250 RPM /
    # 250K TPM, versus the 20-requests-per-day free tier this project started on.
    #
    # Evaluation roles run on Kimi-K2.6 (Moonshot) -- a DIFFERENT model family
    # from the DeepSeek agent. This matters twice over:
    #   judge_model        the reasoning grade is not a model marking its own
    #                      homework (self-preference bias)
    #   annotator_b_model  Cohen's kappa on the gold set now measures genuine
    #                      inter-annotator agreement rather than one model's
    #                      sensitivity to prompt wording
    # Kimi is used only where JSON output is needed, not tool calling, so its
    # tool-use support is irrelevant to these roles.
    chat_model: str = "DeepSeek-V4-Flash"        # agent reasoning (needs tool calling)
    enrich_model: str = "DeepSeek-V4-Flash"      # bulk: 56 case cards
    judge_model: str = "Kimi-K2.6"               # independent reasoning judge
    annotator_a_model: str = "DeepSeek-V4-Flash"
    annotator_b_model: str = "Kimi-K2.6"         # independent second annotator

    # --- Chunking ------------------------------------------------------------
    chunk_chars: int = 2200
    chunk_overlap: int = 300

    # --- Retrieval -----------------------------------------------------------
    dense_k: int = 30          # candidates from vector search
    sparse_k: int = 30         # candidates from BM25 full-text search
    rrf_k: int = 60            # reciprocal-rank-fusion damping constant
    rerank_k: int = 12         # survivors handed to the LLM reranker

    # --- Agent budgets -------------------------------------------------------
    # A *budget*, not a gate: the agent may escalate mid-run (see agent.py).
    budget_simple: int = 4
    budget_deep: int = 14
    budget_ceiling: int = 22
    # How many times the terminal contract may bounce a report back for a better
    # adverse analysis. One round is gameable (the agent empties the section to
    # get past a padding complaint); two lets it recover from that.
    max_adverse_revisions: int = 2
    # How many times the agent is pushed back for answering in prose instead of
    # calling a terminal tool. Every validation gate lives inside those tools,
    # so a prose reply skipped all of them.
    max_terminal_nudges: int = 2

    # Synthesis check: a judgment the reranker scored at or above this is one
    # the system itself called strong, so dropping it silently is the agent
    # disagreeing with its own retrieval. Set at 7/10 deliberately -- low enough
    # to catch the measured 35-point gap between retrieval and answer recall,
    # high enough that marginal hits are never demanded, which is what stops
    # this from trading precision away for recall.
    # Every search also returns this many judgments favouring the opposing side.
    # Adverse coverage belongs in retrieval, not in output validation: held-out
    # testing showed retrieval generalising to unseen queries while the
    # validation gates did not.
    # REVERTED to 0 (disabled) after measurement. The automatic counter-search
    # was built on the hypothesis that adverse coverage failed to generalise
    # because it lived in output validation rather than retrieval. Held-out
    # testing refuted that: adverse recall went 16.7% -> 11.7% and burial
    # returned (0 -> 6), while costing 2x the rerank calls on every search.
    #
    # The diagnosis was wrong. Held-out retrieval recall is 100% -- the agent
    # already SEES every damaging judgment. Adverse coverage fails downstream,
    # in synthesis, and feeding it more documents diluted attention instead of
    # focusing it. Kept as a switch rather than deleted so the negative result
    # stays reproducible.
    counter_search_k: int = 0

    # --- Ablation switches ---------------------------------------------------
    # Each behavioural fix is individually switchable so its effect can be
    # measured in isolation. Four were once changed together and the resulting
    # numbers could not be attributed to any of them -- full cost, no signal.
    # Defaults set by ablation, not by intuition. Measured on 5 queries x 1 seed
    # (baseline -> arm):
    #
    #   synthesis check   precision 0.769 -> 0.705, held-out precision 0.861 ->
    #                     0.733, and synthesis loss ROSE 0.114 -> 0.164 -- the
    #                     opposite of its purpose. OFF.
    #   quote checking    adverse recall 0.556 -> 0.056. It shares the revision
    #                     budget with the adverse gates and starves them; two
    #                     checks competing for the same rounds. Also +60% latency.
    #                     OFF until decoupled from that budget.
    #   adverse gates     burial 8 -> 0, the only arm that reaches zero, for ~9
    #                     points of precision. Kept: concealing a damaging
    #                     judgment is the failure that matters most here, and a
    #                     reader can discount a marginal citation but cannot
    #                     discount one never shown to them.
    #
    # All three had looked like improvements when changed together, because the
    # adverse gates' gains were being credited to the other two.
    enable_synthesis_check: bool = False     # D1/D2: measured harmful
    enable_quote_verification: bool = False  # D3: starves the adverse gates
    enable_adverse_gates: bool = True        # D4: only fix that eliminates burial
    # Should the prompt work out which side the client is on before hunting
    # adverse authority? The instruction previously hardcoded
    # `favours='insurer'`, which is right for claimant's counsel and exactly
    # backwards when the client IS the insurer. Switchable so the change can be
    # attributed: on the run that introduced it, strict adverse recall rose
    # (0.125 -> 0.177) while burial rose too (0 -> 11), on one seed, and those
    # cannot be separated without an ablation arm.
    # REVERTED after ablation (2 arms x 5 queries, one variable):
    #   adverse recall     0.427 -> 0.226
    #   buried                 0 -> 12
    #   precision          0.771 -> 0.704
    #   held-out precision 0.850 -> 0.727
    #   latency             263s -> 405s
    #   strict adverse     0.389 -> 0.562   (the only gain)
    #
    # The diagnosis was right -- hardcoding favours='insurer' IS backwards when
    # the client is the insurer -- but the fix was not. Making rule 3 conditional
    # and longer appears to spend the agent's budget deciding which side it is on
    # rather than covering the adverse pool, and burial returned. A correct
    # diagnosis does not guarantee a correct fix.
    #
    # The right version of this is probably to resolve the side ONCE at triage and
    # hand the agent a single unambiguous instruction, rather than asking it to
    # reason about sides on every search. Kept as a switch so the negative result
    # stays reproducible.
    enable_side_relative_adverse: bool = False

    # Resolve the client's side ONCE at triage and hand the loop rule 3 as a
    # pre-resolved constant ("you act for the INSURER; claimant-favouring
    # judgments are your adverse list"). Targets the insurer-side asymmetry:
    # with an insurer client the fixed rule hunts the wrong pool, and the agent
    # buried all 11 outright claimant wins it had retrieved on q05. Distinct
    # from enable_side_relative_adverse, which asked the agent to reason about
    # sides every turn and measurably failed.
    #
    # KEPT after a same-batch two-arm ablation (joins the adverse gates and the
    # adverse redefinition; five other changes reverted). Adverse recall
    # 0.379 -> 0.474 with burial 4 -> 1, claimant-side burial stayed at zero
    # (the per-turn variant's failure mode), and the held-out insurer query
    # went 0.4 -> 0.9 broad / 1.0 strict. Cost fell alongside: tokens -17%,
    # latency -11%, tool calls 28.7 -> 26.0.
    enable_triage_side_resolution: bool = True

    # Rule 3 names the exact exhaustive call for the adverse pool
    # (`filter_judgments(outcome_favours=<opponent>)`) instead of only the ranked
    # `search_precedents`. Measured cause: on q05 the client IS the insurer, and
    # the agent ran `filter_judgments(outcome_favours='insurer')` -- its own
    # SUPPORTING side -- never once querying the claimant side where its adverse
    # authority lives. Verified that one correct call returns 16 of 16 strict
    # adverse judgments, so this is a behavioural gap, not a retrieval one.
    #
    # REVERTED -- the eighth of twelve changes to fail measurement, and the most
    # instructive. The instruction WORKED at the layer it targeted: the agent
    # swept all three pools (`claimant`, `insurer`, `mixed`) instead of one,
    # verified in the traces. It still failed the gate, because retrieving the
    # pool is not the constraint. Measured: burial 2 -> 14, tokens +60%
    # (481K -> 771K), adverse recall flat (0.405 -> 0.402), and q01 -- a
    # claimant-side control -- lost half its strict recall (0.667 -> 0.333).
    # On q05 the agent surfaced 28 claimant judgments and flagged 4, leaving 13
    # buried: it saw the damaging authority and did not report it, which is
    # worse than never retrieving it. The bottleneck is the terminal report's
    # bounded adverse capacity, not the search. Widening the funnel upstream
    # only increases what falls out of it.
    enable_explicit_adverse_filter: bool = False

    # Triage counts an enumeration as 'simple' even when it asks for a detail per
    # item. Measured cause: q06/q09/q10 each carry a second clause ("...and what
    # proposition does each take from it") which read as comparison, so all three
    # got a full research report where a lookup sufficed -- 3 of the 3 contract
    # misses, and ~15% of the token bill. q12, the same shape without a second
    # clause, was correctly classified simple.
    #
    # REVERTED after ablation -- the seventh of eleven changes to fail. The
    # diagnosis held (all four enumerations reclassified to budget 4, verified
    # directly) but the lever was wrong: triage sets the STARTING allowance,
    # while the output contract is a separate tool the agent elects, and the
    # agent escalates its own budget regardless. Measured: mean tool calls flat
    # (17.8 -> 18.4), tokens UP 18% (201K -> 239K) against an intended 15% cut,
    # and the single contract that flipped was q12 -- the same query observed
    # flipping between runs unprompted, so the gain is unattributable. A real
    # fix has to act on the terminal-tool choice, not the budget.
    enable_enumeration_is_simple: bool = False

    # Compact old tool results in the message history.
    #
    # Measured: token cost is QUADRATIC in tool calls -- tokens ~= 600 * n^2, flat
    # across 18 queries. Every turn resends the whole conversation, so a
    # read_judgment result (2,521 tokens) is retransmitted on every subsequent
    # turn. At 24 tool calls the early results are sent 20+ times.
    #
    # REVERTED after ablation -- the fifth of seven changes to fail measurement.
    # Truncating old results to one line + "call the tool again if needed"
    # destroyed information the agent still needed, and it paid whatever it took
    # to get it back: one query re-read the same judgments 41 times (vs <=2 dup
    # reads in the control), mean tool calls went 26.5 -> 101, two runs looped
    # into the graph recursion limit, and net tokens went UP (457K -> 509K).
    # The n^2 saving per request was eaten by 3-4x more turns. Any future cost
    # cut must PRESERVE what the agent still needs -- an information-keeping
    # digest (holding / ratio / quoted lines), not truncation. Kept as a switch
    # so the negative result stays reproducible.
    enable_history_compaction: bool = False
    history_keep_full: int = 3        # most recent tool results kept verbatim
    history_summary_chars: int = 320  # cap on each compacted result

    # The information-PRESERVING successor to the truncation above. Stale
    # read_judgment results keep their entire structured head -- holding,
    # ratio, outcome, statutes, citations, the fields the agent's reasoning
    # runs on -- plus the opening of each passage for quotable material; only
    # deep passage text is dropped, and the banner never invites a re-read.
    # MEASURED and left OFF. It fixed the catastrophic mode of the truncation
    # variant -- zero failures, worst-case duplicate reads 6 against 41 -- and
    # cut tokens 21.6% (best query -44%). It still failed the pre-registered
    # gate: duplicate reads rose 2 -> 8 against the same-batch control, two
    # queries paid MORE net tokens than control, and held-out precision fell
    # 0.844 -> 0.733. 240-char passage openings evidently still starve some
    # runs. Third variant (larger keeps, or digest-on-demand) is future work.
    #
    # v3 (window change, in code): the stale window now counts READS rather
    # than all tool results -- under v2 a read chased by two searches was
    # digested while still in active use, which is where the re-reads and the
    # starvation came from. v3 arms additionally override passage chars 240 ->
    # 600 and the window 3 -> 5.
    #
    # v3 RESULT: reverted, and the family is CLOSED. Quality held exactly as
    # the window diagnosis predicted -- but tokens went UP 6.4%: one extra
    # tool call (~28K-token request) outweighs everything a safe digest saves.
    # Three dial settings measured: aggressive breaks quality, safe saves
    # nothing, and v2's midpoint fails both gates. Mechanical history
    # compression is not a lever for this system; the structural fixes
    # (sub-agent reads, provider prompt caching) are the real successors.
    enable_digest_compaction: bool = False
    digest_keep_full: int = 3          # most recent tool results stay verbatim
    digest_passage_chars: int = 240    # kept from the head of each stale passage

    # Passages returned per read_judgment. Reads are 68% of all input tokens
    # (measured attribution over the final runs), and each passage is ~500
    # tokens that gets re-sent on every later request. Trimming at the SOURCE
    # avoids the removal trap that killed the compaction family: the agent
    # never misses what it never saw. k=3 under ablation; risk is recall and
    # groundedness losing whatever lived in the 4th-ranked passage.
    read_passages_k: int = 4

    synthesis_threshold: float = 7.0
    max_synthesis_prompts: int = 8

    # --- Evaluation ----------------------------------------------------------
    # Queries are independent, so they run concurrently. The process-wide rate
    # limiter still caps aggregate request rate, so this trades wall-clock for
    # nothing -- sequential runs sat idle waiting on the provider.
    # 4 workers timed out 4 of 30 runs against this deployment; 3 leaves the
    # provider headroom while still overlapping most of the waiting.
    # Dropped from 3 to 2: the automatic counter-search doubles rerank calls per
    # search, and this deployment caps at 250K tokens/minute. The binding limit
    # is tokens, not requests -- 3 workers tripped 429s once adverse retrieval
    # landed.
    eval_workers: int = 3

    # --- API rate limiting ---------------------------------------------------
    # DeepSeek allows 250 RPM, Kimi 100 RPM. The limiter is process-wide and
    # model-agnostic, so it is set below the LOWER of the two.
    rpm: int = 80
    max_retries: int = 5
    # Broad briefs with adverse-gate revisions run long: one measured at 623s
    # across 27 tool calls. 180s timed out 4 of 30 eval runs under concurrency,
    # and 300s still killed the 2 heaviest (60-75K-token requests brushing the
    # limit in prefill). A slow success beats a dead run; genuine hangs are
    # covered by retries in agent._invoke_retrying.
    request_timeout_s: float = 600.0

    # --- API -----------------------------------------------------------------
    azure_base_url: str = "https://ai-service-strataos.services.ai.azure.com/models"
    azure_api_version: str = "2024-05-01-preview"

    # Accept several spellings: Streamlit Cloud's secrets UI, CI, and local .env
    # files all tend to name this differently.
    azure_api_key: str = Field(
        "",
        validation_alias=AliasChoices(
            "AZURE_API_KEY", "AZURE_AI_API_KEY", "LEXI_AZURE_API_KEY", "OPENAI_API_KEY"
        ),
    )

    def require_key(self) -> str:
        import os

        key = self.azure_api_key or os.getenv("AZURE_API_KEY", "")
        if not key:
            raise RuntimeError(
                "No Azure AI API key. Copy .env.example to .env and set AZURE_API_KEY "
                "(Azure AI Foundry -> your deployment -> Keys and Endpoint)."
            )
        return key


settings = Settings()
