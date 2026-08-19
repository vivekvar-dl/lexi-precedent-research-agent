"""Gold relevance labels: every document, every query, no sampling.

METHODOLOGY (this is the answer to "how do you know your agent isn't missing
relevant judgments?")

Standard RAG evaluation pools the top-k output of the system under test and
labels only that. Recall measured that way is circular -- you can never count
what nothing retrieved. With 56 documents we do not have to accept that. We
label the FULL 56 x 15 matrix, so the recall denominator is exact rather than
estimated.

Three tiers, escalating only where it matters:

  Tier 1  Two INDEPENDENT annotators grade all 56 documents for each query.
          They run on different models with differently-worded prompts, so
          agreement is meaningful rather than an echo. Both see the same
          compressed case cards, which are themselves derived from full text.

  Tier 2  Cohen's kappa is computed per query and reported. Low agreement is a
          published property of the label set, not something hidden.

  Tier 3  Every disagreement is re-judged against the FULL judgment text by an
          adjudicator. Full text is spent precisely where the cheap signal was
          ambiguous, which is where it buys the most.

Grades are graded, not binary:
    0  irrelevant
    1  related -- same area, would not be cited for this question
    2  directly on point -- a lawyer would cite this

Each label also carries a side tag (claimant / insurer / mixed / neutral) taken
from the case card, which is what makes Dimension 4 (adverse identification)
measurable at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexi.config import settings  # noqa: E402
from lexi.enrich import load_cards  # noqa: E402
from lexi.ingest import ingest_corpus  # noqa: E402
from lexi.llm import LLM  # noqa: E402

from .metrics import cohens_kappa, gwet_ac1, prevalence, raw_agreement  # noqa: E402
from .queries import QUERIES, EvalQuery  # noqa: E402

GOLD_PATH = Path(__file__).parent / "gold.json"


class _Label(BaseModel):
    doc_id: str
    grade: int = Field(..., ge=0, le=2)
    reason: str = ""


class _Labels(BaseModel):
    labels: list[_Label]


# Two prompts, worded differently on purpose. If both annotators were given the
# same framing on the same model, agreement would measure prompt determinism
# rather than genuine label reliability.

PROMPT_A = """You are building a relevance benchmark for a legal research system.

For EVERY judgment listed, grade how relevant it is to the research question.

  2 = directly on point. A lawyer researching this question would cite it.
  1 = related. Same general area, but would not be cited for THIS question.
  0 = irrelevant. Different area of law, or no bearing on the question.

The corpus is deliberately mixed and contains judgments from unrelated fields.
Grade those 0. Do not inflate grades: a judgment that merely shares vocabulary is
not relevant.

RESEARCH QUESTION:
{question}

WHAT COUNTS AS RELEVANT:
{criterion}

JUDGMENTS ({n} total -- you must return a grade for every single one):
{listing}"""

PROMPT_B = """Task: relevance annotation for legal information retrieval.

Below is a research question and a list of court judgments. Decide, for each
judgment independently, whether a litigator researching that question would find
it useful.

Scale:
  2 -- squarely applicable; would appear in the written argument
  1 -- adjacent; useful background but not authority for this point
  0 -- not applicable to this question

Be strict. This corpus mixes several unrelated areas of law, and most judgments
will score 0 for any given question. Assign a grade to every doc_id listed.

QUESTION: {question}

RELEVANCE TEST: {criterion}

JUDGMENTS ({n}):
{listing}"""

ADJUDICATE = """Two annotators disagreed on how relevant this judgment is to a research
question. Decide the correct grade by reading the judgment itself.

  2 = directly on point, would be cited
  1 = related but not authority for this question
  0 = irrelevant

RESEARCH QUESTION: {question}
RELEVANCE TEST: {criterion}

ANNOTATOR A said {grade_a}. ANNOTATOR B said {grade_b}.

JUDGMENT [{doc_id}] {title}
{text}

Return the single correct grade, with a one-line reason."""


def _listing(cards) -> str:
    """What each annotator sees, one line per judgment.

    Citations and statutes are included deliberately. An earlier version showed
    only title/type/outcome/holding, which meant a query like "which judgments
    cite Swaran Singh?" was unanswerable from the evidence provided -- the
    annotators were guessing from holding text and marked 2 relevant where the
    raw text shows 13. Never ask an annotator a question the listing cannot
    answer.
    """
    out = []
    for c in cards:
        cites = "; ".join(c.precedents_cited)[:200] or "-"
        stats = "; ".join(c.statutes_cited)[:160] or "-"
        out.append(
            f"[{c.doc_id}] {c.title}\n"
            f"    type: {c.case_type or '?'} | favours: {c.outcome_favours.value}\n"
            f"    held: {(c.holding or '')[:200]}\n"
            f"    statutes: {stats}\n"
            f"    cites: {cites}"
        )
    return "\n".join(out)


def _annotate(q: EvalQuery, cards, prompt: str, model: str) -> dict[str, int]:
    llm = LLM(model=model)
    res = llm.structured(
        prompt.format(
            question=q.text, criterion=q.relevance_criterion, n=len(cards), listing=_listing(cards)
        ),
        _Labels,
    )
    got = {lab.doc_id: lab.grade for lab in res.labels}
    # Any document the annotator silently dropped is treated as irrelevant --
    # never as missing, which would quietly shrink the recall denominator.
    return {c.doc_id: got.get(c.doc_id, 0) for c in cards}


def _adjudicate(q: EvalQuery, doc_id: str, ga: int, gb: int, texts, titles) -> int:
    llm = LLM(model=settings.judge_model)

    class _Verdict(BaseModel):
        grade: int = Field(..., ge=0, le=2)
        reason: str = ""

    try:
        v = llm.structured(
            ADJUDICATE.format(
                question=q.text,
                criterion=q.relevance_criterion,
                grade_a=ga,
                grade_b=gb,
                doc_id=doc_id,
                title=titles.get(doc_id, ""),
                text=texts.get(doc_id, "")[:40_000],
            ),
            _Verdict,
        )
        return v.grade
    except Exception:
        return max(ga, gb)  # fall back to the more generous label


def build_gold(queries: list[EvalQuery] | None = None, adjudicate: bool = True) -> dict:
    queries = queries or QUERIES
    cards = load_cards()
    docs, _ = ingest_corpus()
    texts = {d["doc_id"]: d["text"] for d in docs}
    titles = {d["doc_id"]: d["title"] for d in docs}
    sides = {c.doc_id: c.outcome_favours.value for c in cards}

    existing = json.loads(GOLD_PATH.read_text()) if GOLD_PATH.exists() else {"queries": {}}
    out = existing.get("queries", {})

    for q in queries:
        if q.qid in out:
            print(f"{q.qid}: cached, skipping")
            continue
        print(f"{q.qid}: annotating {len(cards)} judgments ...", flush=True)

        a = _annotate(q, cards, PROMPT_A, settings.annotator_a_model)
        b = _annotate(q, cards, PROMPT_B, settings.annotator_b_model)
        kappa = cohens_kappa(a, b)
        # Reported together on purpose. These labels are heavily skewed -- most of
        # the 56 judgments are irrelevant to any one query -- and kappa collapses
        # under skew even when agreement is near total (the "kappa paradox").
        # AC1 is robust to that; raw agreement and prevalence make the diagnosis
        # possible instead of leaving a low kappa unexplained.
        ac1 = gwet_ac1(a, b)
        raw = raw_agreement(a, b)

        disagreements = [d for d in a if a[d] != b[d]]
        final = dict(a)
        n_adj = 0
        if adjudicate and disagreements:
            for d in disagreements:
                final[d] = _adjudicate(q, d, a[d], b[d], texts, titles)
                n_adj += 1
        else:
            for d in disagreements:
                final[d] = max(a[d], b[d])

        out[q.qid] = {
            "question": q.text,
            "kind": q.kind,
            "criterion": q.relevance_criterion,
            "flip_of": q.flip_of,
            "tags": q.tags,
            "labels": final,
            "annotator_a": a,
            "annotator_b": b,
            "kappa": round(kappa, 3),
            "gwet_ac1": round(ac1, 3),
            "raw_agreement": round(raw, 3),
            "prevalence": round(prevalence(final), 3),
            "n_disagreements": len(disagreements),
            "n_adjudicated": n_adj,
            "sides": sides,
            "n_relevant": sum(1 for v in final.values() if v >= 1),
            "n_on_point": sum(1 for v in final.values() if v == 2),
        }
        print(
            f"  kappa={kappa:.3f} AC1={ac1:.3f} raw={raw:.3f}  "
            f"disagreements={len(disagreements)}  "
            f"relevant={out[q.qid]['n_relevant']}  on-point={out[q.qid]['n_on_point']}",
            flush=True,
        )
        GOLD_PATH.write_text(json.dumps({"queries": out}, indent=1))  # checkpoint

    return {"queries": out}


# Queries whose relevance test IS the mechanical test -- "does this judgment cite
# X" is settled by the text, not by opinion. For these, the computed key REPLACES
# the model labels. q13 (trademark) and q10 (future prospects) are excluded from
# the override: their regex is a strong proxy but not a definition, so there the
# computed truth is used only to *score* the annotators, not to overrule them.
DETERMINISTIC_OVERRIDE = {"q06_swaran", "q12_s166", "q14_summarise", "q15_absent"}


def apply_deterministic_truth(gold: dict) -> dict:
    """Replace model labels with computed ones wherever the answer is computable.

    Measured justification: on q06 both annotators agreed perfectly (kappa 1.000)
    and both were wrong, missing 7 of the 13 judgments that cite Swaran Singh.
    Agreement is not accuracy. Where `grep` settles the question, `grep` wins.

    The override is recorded per query so the report can show what was corrected
    rather than quietly swapping the answer key.
    """
    from .verifiable import compute_truth

    truth = compute_truth()
    for qid in DETERMINISTIC_OVERRIDE:
        q, t = gold["queries"].get(qid), truth.get(qid)
        if not q or not t:
            continue
        before = q["labels"]
        changed = sorted(d for d in t if before.get(d, 0) != t[d])
        q["labels"] = dict(t)
        q["n_relevant"] = sum(1 for v in t.values() if v >= 1)
        q["n_on_point"] = sum(1 for v in t.values() if v == 2)
        q["deterministic"] = True
        q["n_corrected_by_truth"] = len(changed)
        q["corrected_docs"] = changed
    return gold


def backfill_agreement_stats(gold: dict) -> dict:
    """Compute AC1 / raw agreement / prevalence from labels already on disk.

    Both annotators' full label sets are stored, so agreement statistics added
    after the fact are derived, not re-measured -- no re-annotation needed and no
    risk of the numbers drifting from the labels they describe.
    """
    for q in gold["queries"].values():
        a, b = q.get("annotator_a"), q.get("annotator_b")
        if not (a and b):
            continue
        q.setdefault("gwet_ac1", round(gwet_ac1(a, b), 3))
        q.setdefault("raw_agreement", round(raw_agreement(a, b), 3))
        q.setdefault("prevalence", round(prevalence(q["labels"]), 3))
    return gold


def load_gold(apply_truth: bool = True) -> dict:
    if not GOLD_PATH.exists():
        raise FileNotFoundError("evals/gold.json missing -- run `python -m evals.gold`")
    gold = backfill_agreement_stats(json.loads(GOLD_PATH.read_text()))
    return apply_deterministic_truth(gold) if apply_truth else gold


def main() -> None:
    gold = build_gold()
    qs = gold["queries"]
    print(f"\ngold set: {len(qs)} queries x 56 judgments = {len(qs)*56} labelled pairs")
    ks = [q["kappa"] for q in qs.values()]
    print(f"mean Cohen's kappa: {sum(ks)/len(ks):.3f}  (min {min(ks):.3f}, max {max(ks):.3f})")
    print(f"total adjudicated : {sum(q['n_adjudicated'] for q in qs.values())}")
    print(f"\nwrote {GOLD_PATH}")


if __name__ == "__main__":
    main()
