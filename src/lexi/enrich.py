"""One-time offline pass: judgment text -> structured CaseCard.

This is the architectural centre of the system. Running an LLM once per document
(56 calls, cached and committed) converts the problem from "semantic similarity
over prose chunks" into "structured reasoning over a small knowledge base".

Two things become possible that plain vector RAG cannot do:
  1. Exhaustive structured queries. "Which judgments involve commercial
     vehicles?" is answered by scanning 56 boolean fields -- exactly, not
     probabilistically via top-k.
  2. Honest adverse retrieval. `outcome_favours` is extracted at index time, so
     the agent can ask for judgments that went AGAINST a claimant instead of
     hoping the embedding happens to surface them.

Cost: ~56 calls, one time. Re-run only when the corpus changes.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from .config import settings
from .ingest import ingest_corpus
from .llm import LLM
from .schemas import CaseCard

SYSTEM = (
    "You are a legal analyst building a structured index of Indian court judgments. "
    "Extract only what the judgment actually says. Never infer facts that are not "
    "stated. If a field is not addressed, return null or an empty list."
)

PROMPT = """Read this judgment and extract a structured record.

Guidance on the harder fields:

- `ratio`: the binding principle the court actually decided on. Not obiter, not a
  summary of the facts. One or two sentences.
- `holding`: what the court concluded on the specific dispute, in one sentence.
- `outcome_favours`: who the RESULT benefits.
    "claimant"  -> victim/dependant/insured won, or compensation was granted or increased
    "insurer"   -> insurer/defendant won, was exonerated, or compensation was reduced
    "mixed"     -> split result (e.g. liability upheld but quantum cut; pay-and-recover
                   orders where the insurer must pay but may recover from the owner)
    "neutral"   -> the judgment is not about a contested claim between these parties
  Judge by the OPERATIVE ORDER, not by sympathy in the language.
- `case_type`: short label, e.g. "motor accident compensation", "trademark
  infringement", "central excise", "criminal - rash and negligent driving".
  This corpus is mixed; classify honestly, do not force a motor-accident label.
- `statutes_cited`: section + act, e.g. "Section 149, Motor Vehicles Act 1988".
- `precedents_cited`: case names only, as written in the judgment.
- `facts` / `quantum`: fill only if the judgment addresses them. Leave null otherwise.

JUDGMENT METADATA
doc_id: {doc_id}
title: {title}
court: {court}
decided_on: {decided_on}

JUDGMENT TEXT
{text}
"""

# Judgments run to 229K chars. Gemini's context handles that, but the tail of a
# judgment (the ORDER) matters as much as the head, so when truncating we keep
# both ends rather than the first N characters.
_HEAD, _TAIL = 60_000, 30_000


def _fit(text: str) -> str:
    if len(text) <= _HEAD + _TAIL:
        return text
    return f"{text[:_HEAD]}\n\n[... middle omitted ...]\n\n{text[-_TAIL:]}"


def enrich_one(parsed: dict, llm: LLM) -> CaseCard:
    card = llm.structured(
        PROMPT.format(
            doc_id=parsed["doc_id"],
            title=parsed["title"],
            court=parsed["court"] or "unknown",
            decided_on=parsed["decided_on"] or "unknown",
            text=_fit(parsed["text"]),
        ),
        CaseCard,
        system=SYSTEM,
    )
    # Trust the deterministic parse over the model for fields we already know.
    card.doc_id = parsed["doc_id"]
    card.title = parsed["title"]
    card.source_url = parsed["source_url"]
    card.n_pages = parsed["n_pages"]
    card.n_chars = parsed["n_chars"]
    card.court = card.court or parsed["court"]
    card.decided_on = card.decided_on or parsed["decided_on"]
    return card


def _save(cards: dict[str, CaseCard]) -> None:
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.cards_path.write_text(
        json.dumps(
            [cards[k].model_dump(mode="json") for k in sorted(cards)], indent=1, ensure_ascii=False
        )
    )


def build_cards(max_workers: int = 3, resume: bool = True) -> list[CaseCard]:
    """Enrich every judgment, checkpointing after each one.

    Resumable on purpose: the free-tier quota is 20 req/min, so a full run can be
    interrupted. Re-running picks up only the documents still missing.
    """
    docs, _ = ingest_corpus()
    cards: dict[str, CaseCard] = {}
    if resume and settings.cards_path.exists():
        cards = {c.doc_id: c for c in load_cards()}
        print(f"resuming: {len(cards)} cards already built")

    todo = [d for d in docs if d["doc_id"] not in cards]
    if not todo:
        print("nothing to do -- all cards present")
        return [cards[k] for k in sorted(cards)]

    llm = LLM(model=settings.enrich_model)
    lock = Lock()
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(enrich_one, d, llm): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                card = fut.result()
                with lock:
                    cards[card.doc_id] = card
                    _save(cards)  # checkpoint every document
                print(
                    f"[{i:>2}/{len(todo)}] {card.doc_id} {card.outcome_favours.value:<8} "
                    f"{(card.case_type or '?')[:32]:<32} {card.title[:38]}",
                    flush=True,
                )
            except Exception as e:
                failed.append(d["doc_id"])
                print(f"[{i:>2}/{len(todo)}] {d['doc_id']} FAILED: {str(e)[:120]}",
                      file=sys.stderr, flush=True)

    print(
        f"\nusage: {llm.usage.calls} calls, {llm.usage.in_tokens:,} in / "
        f"{llm.usage.out_tokens:,} out tokens, {llm.usage.throttled} throttle waits"
    )
    if failed:
        print(f"FAILED ({len(failed)}): {failed} -- re-run to retry just these", file=sys.stderr)
    return [cards[k] for k in sorted(cards)]


def load_cards() -> list[CaseCard]:
    if not settings.cards_path.exists():
        raise FileNotFoundError(
            f"{settings.cards_path} missing -- run `python -m lexi.enrich` first"
        )
    return [CaseCard.model_validate(c) for c in json.loads(settings.cards_path.read_text())]


def main() -> None:
    cards = build_cards()  # checkpoints internally
    print(f"\nwrote {len(cards)} case cards -> {settings.cards_path}")

    from collections import Counter
    print("\nfavours:", dict(Counter(c.outcome_favours.value for c in cards)))
    print("types  :", dict(Counter((c.case_type or "?")[:28] for c in cards).most_common(10)))


if __name__ == "__main__":
    main()
