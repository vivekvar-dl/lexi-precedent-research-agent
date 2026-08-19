"""Hybrid retrieval: dense + BM25 -> RRF fusion -> LLM rerank, with exact
metadata filtering and a full-corpus screening mode.

Every stage keeps its own score. Nothing is collapsed into a single opaque
number, because the Streamlit trace renders the decomposition and the eval
framework reads it. `ScoredDoc` is the shared currency.
"""
from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field

from .config import settings
from .index import embed_query, get_tables
from .llm import LLM
from .schemas import ScoredDoc

# =============================================================================
# Stage 1 -- candidate generation
# =============================================================================


def dense_search(query: str, k: int, where: str | None = None) -> list[tuple[str, float]]:
    """Vector search over case cards. Returns (doc_id, similarity) by rank."""
    cards, _ = get_tables()
    q = cards.search(embed_query(query).tolist(), vector_column_name="vector").limit(k)
    if where:
        q = q.where(where, prefilter=True)
    rows = q.to_list()
    # LanceDB returns L2 `_distance`; vectors are normalised so sim = 1 - d/2.
    return [(r["doc_id"], 1.0 - r.get("_distance", 0.0) / 2.0) for r in rows]


def sparse_search(query: str, k: int, where: str | None = None) -> list[tuple[str, float]]:
    """BM25 full-text search over case cards (the concatenated `fts_text` field)."""
    cards, _ = get_tables()
    try:
        q = cards.search(query, query_type="fts", fts_columns="fts_text").limit(k)
        if where:
            q = q.where(where, prefilter=True)
        rows = q.to_list()
    except Exception:
        return []  # FTS chokes on some punctuation-heavy queries; dense carries it
    return [(r["doc_id"], float(r.get("_score", 0.0))) for r in rows]


def rrf_fuse(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    k: int = None,
) -> list[ScoredDoc]:
    """Reciprocal Rank Fusion.

    Chosen over score-weighted blending because BM25 and cosine live on
    incomparable scales; RRF only needs ranks, so it needs no normalisation
    constant to be tuned per corpus.
    """
    k = k or settings.rrf_k
    dr = {d: i + 1 for i, (d, _) in enumerate(dense)}
    sr = {d: i + 1 for i, (d, _) in enumerate(sparse)}
    ds = dict(dense)
    ss = dict(sparse)

    out = []
    for doc_id in set(dr) | set(sr):
        score = 0.0
        if doc_id in dr:
            score += 1.0 / (k + dr[doc_id])
        if doc_id in sr:
            score += 1.0 / (k + sr[doc_id])
        out.append(
            ScoredDoc(
                doc_id=doc_id,
                dense_score=ds.get(doc_id),
                dense_rank=dr.get(doc_id),
                sparse_score=ss.get(doc_id),
                sparse_rank=sr.get(doc_id),
                fused_score=score,
            )
        )
    out.sort(key=lambda d: d.fused_score or 0.0, reverse=True)
    for i, d in enumerate(out, 1):
        d.fused_rank = i
    return out


# =============================================================================
# Stage 2 -- LLM reranking
# =============================================================================


class _RerankItem(BaseModel):
    doc_id: str
    relevance: float = Field(..., ge=0.0, le=10.0)
    why: str


class _RerankResult(BaseModel):
    ranked: list[_RerankItem]


RERANK_PROMPT = """Rank these judgments by how relevant each is to the research question.

Relevance in law is NOT topical similarity. Score on:
  - Does the legal ISSUE match (not just the subject matter)?
  - Does the factual posture align?
  - Does the ratio actually bear on the question asked?

Score 0-10. Be decisive: an off-topic judgment scores 0-2 even if it shares
vocabulary. Do not inflate scores to be helpful -- a wrong precedent is worse
than no precedent.

QUESTION: {query}

CANDIDATES:
{candidates}

Return every doc_id you were given, with a score and a one-line reason."""


def llm_rerank(query: str, docs: list[ScoredDoc], llm: LLM | None = None) -> list[ScoredDoc]:
    if not docs:
        return docs
    llm = llm or LLM(model=settings.chat_model)
    cards, _ = get_tables()
    ids = [d.doc_id for d in docs]
    rows = {r["doc_id"]: r for r in cards.search().where(_in_clause(ids)).to_list()}

    blocks = []
    for d in docs:
        r = rows.get(d.doc_id, {})
        blocks.append(
            f"[{d.doc_id}] {r.get('title','')}\n"
            f"  type: {r.get('case_type','?')} | favours: {r.get('outcome_favours','?')}\n"
            f"  holding: {(r.get('holding') or '')[:300]}\n"
            f"  ratio: {(r.get('ratio') or '')[:300]}"
        )

    res = llm.structured(
        RERANK_PROMPT.format(query=query, candidates="\n\n".join(blocks)), _RerankResult
    )
    scores = {i.doc_id: (i.relevance, i.why) for i in res.ranked}

    for d in docs:
        s, why = scores.get(d.doc_id, (0.0, "not scored by reranker"))
        d.rerank_score, d.why = s, why
        d.title = rows.get(d.doc_id, {}).get("title", d.title)
    docs.sort(key=lambda d: (d.rerank_score or 0.0, d.fused_score or 0.0), reverse=True)
    for i, d in enumerate(docs, 1):
        d.final_rank = i
    return docs


# =============================================================================
# Public entry points
# =============================================================================


def hybrid_search(
    query: str,
    k: int | None = None,
    where: str | None = None,
    rerank: bool = True,
    llm: LLM | None = None,
) -> list[ScoredDoc]:
    """Dense + BM25 -> RRF -> optional LLM rerank. The default retrieval path."""
    k = k or settings.rerank_k
    dense = dense_search(query, settings.dense_k, where)
    sparse = sparse_search(query, settings.sparse_k, where)
    fused = rrf_fuse(dense, sparse)[: max(k, settings.rerank_k)]
    if not rerank:
        _attach_titles(fused)
        for i, d in enumerate(fused, 1):
            d.final_rank = i
        return fused[:k]
    return llm_rerank(query, fused, llm)[:k]


def filter_cards(where: str, limit: int = 200) -> list[dict]:
    """Exact structured query over the case cards.

    This is the path that answers 'which judgments involve commercial vehicles?'
    exhaustively. Vector top-k would return 5 and silently miss the rest.
    """
    cards, _ = get_tables()
    return cards.search().where(where).limit(limit).to_list()


def passages_for(query: str, doc_ids: Iterable[str], k: int = 4) -> dict[str, list[dict]]:
    """Best passages within specific judgments -- used for verbatim quotes."""
    _, chunks = get_tables()
    ids = list(doc_ids)
    if not ids:
        return {}
    rows = (
        chunks.search(embed_query(query).tolist(), vector_column_name="vector")
        .where(_in_clause(ids), prefilter=True)
        .limit(k * len(ids))
        .to_list()
    )
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["doc_id"], [])
        if len(out[r["doc_id"]]) < k:
            out[r["doc_id"]].append(
                {
                    "chunk_id": r["chunk_id"],
                    "section": r["section"],
                    "text": r["text"],
                    "score": 1.0 - r.get("_distance", 0.0) / 2.0,
                }
            )
    return out


def all_card_summaries() -> list[dict]:
    """Every card, compressed. Feeds the full-corpus screening mode.

    Only affordable because the corpus is 56 documents -- see ADR on what
    changes at 5,000.
    """
    cards, _ = get_tables()
    return [
        {
            "doc_id": r["doc_id"],
            "title": r["title"],
            "case_type": r["case_type"],
            "favours": r["outcome_favours"],
            "holding": (r["holding"] or "")[:220],
        }
        for r in sorted(cards.search().limit(10_000).to_list(), key=lambda r: r["doc_id"])
    ]


# =============================================================================
# helpers
# =============================================================================


def _in_clause(ids: list[str]) -> str:
    quoted = ", ".join(f"'{i}'" for i in ids)
    return f"doc_id IN ({quoted})"


def _attach_titles(docs: list[ScoredDoc]) -> None:
    if not docs:
        return
    cards, _ = get_tables()
    rows = {
        r["doc_id"]: r for r in cards.search().where(_in_clause([d.doc_id for d in docs])).to_list()
    }
    for d in docs:
        d.title = rows.get(d.doc_id, {}).get("title", d.title)
