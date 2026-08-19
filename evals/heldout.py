"""Held-out queries: the generalisation test.

The adverse dimension was tuned across roughly six iterations against the main
query set. That is the textbook way to look good on your own benchmark and worse
on someone else's -- and the brief says explicitly that recall will be scored
against *their* internal benchmark, not mine.

So this file exists to be a fair test. Rules, kept honestly:

  1. These queries were written AFTER the agent stopped being modified.
  2. Nothing here has been looked at while changing a prompt, a gate or a metric.
  3. If the held-out scores track the main set, the tuning generalised. If they
     drop sharply, it overfitted -- and that is a finding to report, not a reason
     to tune again. Tuning against this set would destroy the only unbiased
     measurement available.

Coverage mirrors the main set's query kinds so the comparison is like-for-like,
and phrasing is deliberately different -- a litigator's wording, not the main
set's -- because robustness to phrasing is part of what is being tested.
"""
from __future__ import annotations

from .queries import EvalQuery

HELDOUT: list[EvalQuery] = [
    EvalQuery(
        "h01_minor_death",
        "Our client's 16-year-old son died as a pillion rider when the motorcycle he "
        "was on was hit by a bus. The bus operator says the boy contributed to the "
        "accident by not wearing a helmet. What does the corpus say about compensation "
        "for a deceased minor with no income, and about helmet-related contributory "
        "negligence?",
        "research",
        "Judgments on computing compensation where the deceased was a minor or had no "
        "established income (notional income, appropriate multiplier), and judgments "
        "addressing whether not wearing a helmet amounts to contributory negligence.\n"
        "  2 = decides either point.\n"
        "  1 = a motor-accident death claim sharing the setting but deciding neither.\n"
        "  0 = not a motor-accident matter.",
        adversarial=True,
        tags=["heldout", "quantum", "contributory"],
    ),
    EvalQuery(
        "h02_permit_breach",
        "Which judgments deal with a breach of permit or route conditions, rather than "
        "a driving licence defect?",
        "structured",
        "Judgments where the alleged policy breach concerns the vehicle's permit, route "
        "authorisation, fitness certificate or registration -- as distinct from the "
        "driver's licence.",
        tags=["heldout", "enumerative"],
    ),
    EvalQuery(
        "h03_owner_diligence",
        "Find precedents on what an owner must do to check a driver's licence before "
        "employing them, and whether failing to check defeats the insurer's defence.",
        "research",
        "Judgments addressing the owner's duty of care or due diligence in verifying a "
        "driver's licence, and the effect of that duty on whether an insurer can avoid "
        "liability for breach of a policy condition.",
        adversarial=True,
        tags=["heldout", "liability"],
    ),
    EvalQuery(
        "h04_interest_rate",
        "What rates of interest have been awarded on motor accident compensation, and "
        "from which date does interest run?",
        "structured",
        "Judgments that specify an interest rate on the compensation awarded, or decide "
        "the date from which interest runs.",
        tags=["heldout", "quantum"],
    ),
    EvalQuery(
        "h05_insurer_appeal",
        "I act for the insurer and want to appeal the quantum, not liability. Which "
        "judgments support reducing an award on appeal, and which cut against me?",
        "adverse",
        "Judgments where an appellate court reduced, or refused to reduce, a "
        "compensation award -- including the limits on an insurer's right to challenge "
        "quantum.\n"
        "  2 = decides whether an award should be reduced on appeal.\n"
        "  1 = an appeal against a motor-accident award deciding other points.\n"
        "  0 = not an appeal about compensation quantum.",
        adversarial=True,
        # "I act for the insurer" -- without this the adverse pool is scored
        # backwards, the same inversion fixed for q05 and missed here.
        client_side="insurer",
        tags=["heldout", "adverse", "flip"],
    ),
    EvalQuery(
        "h06_absent_tax",
        "Find judgments in this corpus about goods and services tax assessment disputes "
        "and input tax credit.",
        "absent",
        "Nothing in this corpus concerns GST assessment or input tax credit. The correct "
        "answer is that there are none.",
        tags=["heldout", "honesty"],
    ),
]

BY_ID = {q.qid: q for q in HELDOUT}
