"""Metric primitives. Pure functions, no I/O, unit-testable.

Implemented by hand rather than pulled from scikit-learn so the deployment stays
light and every formula is inspectable -- these numbers go in the ADR, so they
should be readable, not imported from a black box.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

# Graded relevance: 0 irrelevant, 1 related, 2 directly on point.
RELEVANT_THRESHOLD = 1


def precision_at_k(retrieved: Sequence[str], gold: dict[str, int], k: int | None = None) -> float:
    """Fraction of returned documents that are actually relevant."""
    items = list(retrieved)[: k or len(retrieved)]
    if not items:
        return 0.0
    hits = sum(1 for d in items if gold.get(d, 0) >= RELEVANT_THRESHOLD)
    return hits / len(items)


def recall_at_k(retrieved: Sequence[str], gold: dict[str, int], k: int | None = None) -> float:
    """Fraction of all relevant documents that were found.

    The denominator is exact, not estimated: every one of the 56 documents is
    labelled against every query, so we know precisely what should have been
    found. That is only possible because the corpus is small -- see the ADR.
    """
    relevant = {d for d, g in gold.items() if g >= RELEVANT_THRESHOLD}
    if not relevant:
        return 1.0  # nothing to find; not finding it is correct
    items = set(list(retrieved)[: k or len(retrieved)])
    return len(items & relevant) / len(relevant)


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: Sequence[str], gold: dict[str, int], k: int = 10) -> float:
    """Graded ranking quality. Rewards putting on-point (2) above merely related (1)."""
    gains = [gold.get(d, 0) for d in list(retrieved)[:k]]
    ideal = sorted(gold.values(), reverse=True)[:k]
    denom = dcg(ideal)
    return dcg(gains) / denom if denom else 0.0


def raw_agreement(a: dict[str, int], b: dict[str, int]) -> float:
    """Plain proportion of items the two annotators labelled identically."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return 0.0
    return sum(1 for k in keys if a[k] == b[k]) / len(keys)


def cohens_kappa(a: dict[str, int], b: dict[str, int]) -> float:
    """Inter-annotator agreement, chance-corrected.

    Interpret with `gwet_ac1` and `raw_agreement` beside it, never alone. On
    skewed label distributions kappa collapses toward zero even when annotators
    agree on almost everything -- the "kappa paradox" -- because the
    chance-agreement term is estimated from marginals that are themselves skewed.
    That is precisely this dataset: for any one query, most of the 56 judgments
    are irrelevant.
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        return 0.0
    pa = raw_agreement(a, b)
    ca, cb = Counter(a[k] for k in keys), Counter(b[k] for k in keys)
    pe = sum((ca[c] / len(keys)) * (cb[c] / len(keys)) for c in set(ca) | set(cb))
    return 1.0 if pe == 1 else (pa - pe) / (1 - pe)


def gwet_ac1(a: dict[str, int], b: dict[str, int]) -> float:
    """Gwet's AC1 -- agreement corrected for chance, robust to skew.

    Recommended over kappa/alpha for skewed distributions in the legal-RAG
    evaluation literature. It estimates chance agreement from the probability
    that a rater assigns a category *at random* rather than from the observed
    marginals, so a dataset that is overwhelmingly one class does not drive the
    statistic to zero.

    Read together with kappa: kappa low + AC1 high + raw agreement high means the
    labels are fine and the class balance is skewed. Both low means the rubric is
    genuinely ambiguous and needs rewriting.
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        return 0.0
    n = len(keys)
    categories = set(a[k] for k in keys) | set(b[k] for k in keys)
    if len(categories) < 2:
        return 1.0
    pa = raw_agreement(a, b)
    # pi_k: mean prevalence of category k across both raters
    pi = {
        c: (sum(1 for k in keys if a[k] == c) + sum(1 for k in keys if b[k] == c)) / (2 * n)
        for c in categories
    }
    q = len(categories)
    pe = sum(p * (1 - p) for p in pi.values()) / (q - 1)
    return 1.0 if pe >= 1 else (pa - pe) / (1 - pe)


def prevalence(labels: dict[str, int]) -> float:
    """Fraction of items judged relevant -- the skew that distorts kappa."""
    if not labels:
        return 0.0
    return sum(1 for v in labels.values() if v >= RELEVANT_THRESHOLD) / len(labels)


def evidence_score(retrieved: Sequence[str], gold: dict[str, int], k: int | None = None) -> float:
    """Recall with a penalty for missing the majority of controlling precedents.

    From the legal-IR literature: research that misses most of the governing
    authority is not "partially useful", it is unusable, so linear recall
    overstates it. Below half the relevant set the score is halved; a system that
    finds 40% of controlling precedent should not read as 40% good.
    """
    r = recall_at_k(retrieved, gold, k)
    return r if r >= 0.5 else r * 0.5


def distribution_entropy(labels: Sequence[str]) -> float:
    """Normalised entropy of a categorical distribution, in [0, 1].

    Used for risk calibration: if every adverse precedent is graded "medium",
    entropy is 0 and the agent is not really assessing risk at all.
    """
    if not labels:
        return 0.0
    counts = Counter(labels)
    n = len(labels)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    max_h = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return h / max_h if max_h else 0.0


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
