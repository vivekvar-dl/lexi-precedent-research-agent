"""LanceDB index build: two tables, dense + full-text, no server.

Why two tables (see ADR):
  `cards`  -- one row per judgment. Answers "which CASES are about X" and carries
              every structured field, so metadata filters are exact and
              exhaustive rather than top-k guesses.
  `chunks` -- one row per passage. Answers "what exactly did it SAY about Y" and
              supplies verbatim quotes for citation.

Retrieval happens at chunk level for precision but always resolves to the
document, because in law the unit of precedential authority is the judgment,
not the paragraph.
"""
from __future__ import annotations

import json
import shutil
from functools import lru_cache
from threading import Lock

import lancedb
import numpy as np

from .config import settings
from .enrich import load_cards
from .ingest import ingest_corpus
from .schemas import CaseCard

# Qwen3-Embedding expects an instruction prefix on the QUERY side only;
# documents are embedded raw. Getting this asymmetry wrong costs real nDCG.
QUERY_INSTRUCTION = (
    "Instruct: Given a legal research question, retrieve Indian court judgments "
    "and passages that are relevant to answering it\nQuery: "
)


def _device() -> str:
    """Pick the fastest available backend.

    Apple Silicon exposes a GPU through MPS; defaulting to CPU there costs
    roughly an order of magnitude on the index build. Streamlit Cloud has
    neither, so this falls back to CPU cleanly.
    """
    if settings.embed_device != "auto":
        return settings.embed_device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _quieten_transformers() -> None:
    """Silence `transformers`' optional-dependency probing.

    On import it walks its full model registry, including image processors that
    need `torchvision`. Each miss logs a complete traceback -- 99 of them in one
    app start. Nothing is broken (text embedding is unaffected, verified), but a
    reviewer tailing deployment logs sees a wall of ModuleNotFoundError and
    reasonably assumes it is.

    Suppressed rather than fixed by installing torchvision: that is ~200 MB of
    image-model dependencies this system has no use for, and on a memory-capped
    host the weight is real.
    """
    import logging

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # never let log tuning break startup
        pass


@lru_cache(maxsize=1)
def get_encoder():
    _quieten_transformers()
    from sentence_transformers import SentenceTransformer

    dev = _device()
    print(f"embedding device: {dev}")
    enc = SentenceTransformer(settings.embed_model, device=dev)
    enc.max_seq_length = settings.embed_max_seq_len
    return enc


# One shared encoder, one lock. Torch inference is not reliably thread-safe --
# especially on MPS -- and the evaluation runs queries concurrently, so every
# call into the model is serialised. Encoding is milliseconds; the LLM round
# trips are what actually take time, and those stay parallel.
_encode_lock = Lock()


def embed_documents(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    with _encode_lock:
        return get_encoder().encode(
            texts,
            batch_size=batch_size or settings.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


def embed_query(text: str) -> np.ndarray:
    with _encode_lock:
        return get_encoder().encode(
            [QUERY_INSTRUCTION + text], normalize_embeddings=True, show_progress_bar=False
        )[0]


def _card_document(c: CaseCard) -> str:
    """The text we embed for a judgment. Dense, factual, no filler --
    this is what doc-level semantic search matches against."""
    parts = [
        c.title,
        f"Court: {c.court or 'unknown'}. Decided: {c.decided_on or 'unknown'}.",
        f"Type: {c.case_type or 'unknown'}.",
        f"Issues: {'; '.join(c.legal_issues)}" if c.legal_issues else "",
        f"Holding: {c.holding}" if c.holding else "",
        f"Ratio: {c.ratio}" if c.ratio else "",
        f"Principles: {'; '.join(c.key_principles)}" if c.key_principles else "",
        f"Statutes: {'; '.join(c.statutes_cited)}" if c.statutes_cited else "",
        f"Cites: {'; '.join(c.precedents_cited)}" if c.precedents_cited else "",
        f"Outcome favours: {c.outcome_favours.value}.",
    ]
    return "\n".join(p for p in parts if p)


def _card_row(c: CaseCard, vec: np.ndarray) -> dict:
    """Flatten a card into a filterable LanceDB row.

    List fields are stored twice: as a real list (for display) and joined into a
    lowercase string (so `LIKE '%section 149%'` works in a where-clause).
    """
    f, q = c.facts, c.quantum
    return {
        "doc_id": c.doc_id,
        "vector": vec.astype(np.float32),
        "title": c.title,
        "court": c.court or "",
        "decided_on": c.decided_on or "",
        "case_type": (c.case_type or "").lower(),
        "outcome_favours": c.outcome_favours.value,
        "holding": c.holding or "",
        "ratio": c.ratio or "",
        "text": _card_document(c),
        "statutes_joined": "; ".join(c.statutes_cited).lower(),
        "precedents_joined": "; ".join(c.precedents_cited).lower(),
        "principles_joined": "; ".join(c.key_principles).lower(),
        "issues_joined": "; ".join(c.legal_issues).lower(),
        # LanceDB's native FTS indexes ONE field at a time, so everything worth
        # matching lexically is concatenated here. This is the BM25 surface:
        # it must contain the exact tokens legal queries carry -- section
        # numbers, case names, doctrinal phrases.
        "fts_text": "\n".join(
            [
                c.title,
                _card_document(c),
                "; ".join(c.statutes_cited),
                "; ".join(c.precedents_cited),
                "; ".join(c.key_principles),
                "; ".join(c.legal_issues),
            ]
        ),
        # --- filterable domain facts (-1 / "" mean "not stated") ---
        "vehicle_type": (f.vehicle_type or "").lower(),
        "is_commercial_vehicle": bool(f.is_commercial_vehicle),
        "commercial_known": f.is_commercial_vehicle is not None,
        "driver_licence_defect": (f.driver_licence_defect or "").lower(),
        "has_licence_defect": bool(f.driver_licence_defect),
        "is_death_claim": bool(f.is_death_claim),
        "contributory_negligence": bool(f.contributory_negligence_found),
        "age": float(f.deceased_or_injured_age if f.deceased_or_injured_age is not None else -1),
        "monthly_income": float(f.monthly_income if f.monthly_income is not None else -1),
        "dependents": float(f.dependents_count if f.dependents_count is not None else -1),
        "multiplier": float(q.multiplier if q.multiplier is not None else -1),
        "total_awarded": float(q.total_awarded if q.total_awarded is not None else -1),
        "future_prospects_pct": float(
            q.future_prospects_pct if q.future_prospects_pct is not None else -1
        ),
        "n_pages": c.n_pages,
    }


def build_index(reset: bool = True) -> None:
    cards = load_cards()
    _, chunks = ingest_corpus()
    by_id = {c.doc_id: c for c in cards}

    if reset and settings.lance_dir.exists():
        shutil.rmtree(settings.lance_dir)
    settings.lance_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(settings.lance_dir)

    print(f"embedding {len(cards)} case cards ...")
    card_vecs = embed_documents([_card_document(c) for c in cards])
    cards_tbl = db.create_table("cards", [_card_row(c, v) for c, v in zip(cards, card_vecs)])

    print(f"embedding {len(chunks)} chunks ...")
    chunk_vecs = embed_documents([c.embed_text for c in chunks])
    chunk_rows = [
        {
            "chunk_id": ch.chunk_id,
            "doc_id": ch.doc_id,
            "vector": v.astype(np.float32),
            "section": ch.section,
            "para_start": ch.para_start or "",
            "text": ch.text,
            "context_header": ch.context_header,
            # denormalised so a chunk hit can be rendered without a second lookup
            "title": by_id[ch.doc_id].title if ch.doc_id in by_id else "",
            "outcome_favours": (
                by_id[ch.doc_id].outcome_favours.value if ch.doc_id in by_id else "neutral"
            ),
        }
        for ch, v in zip(chunks, chunk_vecs)
    ]
    chunks_tbl = db.create_table("chunks", chunk_rows)

    # BM25 full-text indexes. Legal queries carry exact tokens -- "Section 149",
    # "Swaran Singh", "163A" -- that dense vectors blur together.
    print("building full-text (BM25) indexes ...")
    cards_tbl.create_fts_index("fts_text", replace=True)
    chunks_tbl.create_fts_index("text", replace=True)

    print(f"\nindex ready at {settings.lance_dir}")
    print(f"  cards : {cards_tbl.count_rows()} rows")
    print(f"  chunks: {chunks_tbl.count_rows()} rows")


@lru_cache(maxsize=1)
def get_db():
    if not settings.lance_dir.exists():
        raise FileNotFoundError(
            f"{settings.lance_dir} missing -- run `python -m lexi.index` first"
        )
    return lancedb.connect(settings.lance_dir)


def get_tables():
    db = get_db()
    return db.open_table("cards"), db.open_table("chunks")


if __name__ == "__main__":
    build_index()
