"""Deterministic compensation calculator for Indian motor-accident death claims.

Compensation under the Motor Vehicles Act is a FORMULA, settled by binding
Supreme Court authority. An LLM that free-hands "approximately Rs 45 lakhs" is
useless to a lawyer. So this is arithmetic in Python, and every step reports the
precedent that governs it.

Authorities encoded here:
  Sarla Verma v. DTC (2009) 6 SCC 121          -- multiplier table, dependency deduction
  National Insurance v. Pranay Sethi (2017)    -- future prospects, conventional heads
    16 SCC 680 (Constitution Bench)
  Magma General v. Nanu Ram (2018) 18 SCC 130  -- consortium per dependent
  United India v. Satinder Kaur (2020)         -- consortium for each dependant
    11 SCC 419

This is a DOMAIN capability, not a hard-coded pipeline: it is parameterised over
any claimant, and the agent chooses whether to call it at all.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# Sarla Verma, as affirmed by the Constitution Bench in Pranay Sethi.
_MULTIPLIER: list[tuple[int, int, int]] = [
    (0, 15, 15), (16, 20, 18), (21, 25, 18), (26, 30, 17), (31, 35, 16),
    (36, 40, 15), (41, 45, 14), (46, 50, 13), (51, 55, 11), (56, 60, 9),
    (61, 65, 7), (66, 200, 5),
]

# Pranay Sethi, conventional heads as fixed in 2017, enhanced 10% every 3 years.
_BASE_YEAR = 2017
_LOSS_OF_ESTATE = 15_000
_FUNERAL = 15_000
_CONSORTIUM = 40_000


class QuantumStep(BaseModel):
    label: str
    value: str
    authority: str


class QuantumResult(BaseModel):
    total: float
    low: float = Field(..., description="Conservative end of a realistic range")
    high: float = Field(..., description="Optimistic end of a realistic range")
    steps: list[QuantumStep]
    notes: list[str] = Field(default_factory=list)

    def format_inr(self, v: float) -> str:
        return f"Rs {v:,.0f} (~Rs {v/100000:.2f} lakh)"

    def summary(self) -> str:
        lines = [f"{s.label}: {s.value}   [{s.authority}]" for s in self.steps]
        lines.append(f"TOTAL: {self.format_inr(self.total)}")
        lines.append(
            f"REALISTIC RANGE: {self.format_inr(self.low)} - {self.format_inr(self.high)}"
        )
        return "\n".join(lines + self.notes)


def multiplier_for_age(age: int) -> int:
    for lo, hi, m in _MULTIPLIER:
        if lo <= age <= hi:
            return m
    return 5


def dependency_deduction(dependents: int) -> tuple[float, str]:
    """Sarla Verma personal-expense deduction, by number of dependants."""
    if dependents <= 1:
        return 0.5, "1/2 (deceased was a bachelor / single dependant)"
    if dependents <= 3:
        return 1 / 3, "1/3 (2-3 dependants)"
    if dependents <= 6:
        return 0.25, "1/4 (4-6 dependants)"
    return 0.2, "1/5 (more than 6 dependants)"


def future_prospects_pct(age: int, employment: str) -> tuple[float, str]:
    """Pranay Sethi. `employment` is 'permanent' or 'self_employed'."""
    permanent = employment.lower().startswith("perm")
    if age < 40:
        pct = 0.50 if permanent else 0.40
    elif age <= 50:
        pct = 0.30 if permanent else 0.25
    elif age <= 60:
        pct = 0.15 if permanent else 0.10
    else:
        pct = 0.0
    kind = "permanent/salaried" if permanent else "self-employed/fixed wages"
    return pct, f"{int(pct*100)}% ({kind}, age {age})"


def _escalation(award_year: int) -> float:
    """Conventional heads rise 10% every three years from the 2017 baseline."""
    periods = max(0, (award_year - _BASE_YEAR) // 3)
    return 1.10**periods


def compute_compensation(
    monthly_income: float,
    age: int,
    dependents: int,
    employment: str = "self_employed",
    award_year: int = 2026,
    contributory_negligence_pct: float = 0.0,
) -> QuantumResult:
    """Full Sarla Verma / Pranay Sethi computation with per-step authority."""
    steps: list[QuantumStep] = []

    annual = monthly_income * 12
    steps.append(
        QuantumStep(
            label="Annual income",
            value=f"Rs {monthly_income:,.0f} x 12 = Rs {annual:,.0f}",
            authority="Sarla Verma (2009) 6 SCC 121",
        )
    )

    fp_pct, fp_desc = future_prospects_pct(age, employment)
    with_fp = annual * (1 + fp_pct)
    steps.append(
        QuantumStep(
            label="Add future prospects",
            value=f"+{fp_desc} -> Rs {with_fp:,.0f}",
            authority="Pranay Sethi (2017) 16 SCC 680 (Constitution Bench)",
        )
    )

    ded, ded_desc = dependency_deduction(dependents)
    after_ded = with_fp * (1 - ded)
    steps.append(
        QuantumStep(
            label="Deduct personal expenses",
            value=f"less {ded_desc} -> Rs {after_ded:,.0f}",
            authority="Sarla Verma (2009) 6 SCC 121",
        )
    )

    mult = multiplier_for_age(age)
    dependency_loss = after_ded * mult
    steps.append(
        QuantumStep(
            label="Apply multiplier",
            value=f"x {mult} (age {age}) -> Rs {dependency_loss:,.0f}",
            authority="Sarla Verma multiplier table, affirmed in Pranay Sethi",
        )
    )

    esc = _escalation(award_year)
    estate = _LOSS_OF_ESTATE * esc
    funeral = _FUNERAL * esc
    # Consortium is payable to each dependant (spousal + parental + filial).
    consortium = _CONSORTIUM * esc * max(dependents, 1)
    conventional = estate + funeral + consortium
    steps.append(
        QuantumStep(
            label="Conventional heads",
            value=(
                f"estate Rs {estate:,.0f} + funeral Rs {funeral:,.0f} + "
                f"consortium Rs {consortium:,.0f} ({dependents} dependants) "
                f"= Rs {conventional:,.0f}"
            ),
            authority="Pranay Sethi; Magma General v. Nanu Ram (2018) 18 SCC 130; "
            "United India v. Satinder Kaur (2020) 11 SCC 419",
        )
    )

    total = dependency_loss + conventional
    notes: list[str] = []

    if contributory_negligence_pct > 0:
        reduction = total * contributory_negligence_pct / 100
        total -= reduction
        steps.append(
            QuantumStep(
                label="Contributory negligence",
                value=f"less {contributory_negligence_pct:.0f}% = Rs {total:,.0f}",
                authority="apportionment of negligence; fact-specific",
            )
        )

    notes.append(
        "Interest is additionally payable, customarily 7.5-9% p.a. from the date "
        "of the claim petition until realisation."
    )
    notes.append(
        "Range reflects the two future-prospects bands available on these facts "
        "(self-employed vs permanent employment), which is the usual live dispute."
    )

    # The realistic band: swing future prospects between the two bands.
    alt_pct, _ = future_prospects_pct(age, "permanent" if employment != "permanent" else "self")
    alt_total = (
        annual * (1 + alt_pct) * (1 - ded) * mult + conventional
    ) * (1 - contributory_negligence_pct / 100)

    return QuantumResult(
        total=round(total),
        low=round(min(total, alt_total)),
        high=round(max(total, alt_total)),
        steps=steps,
        notes=notes,
    )
