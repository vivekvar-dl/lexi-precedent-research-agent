"""Typed contracts for the whole system.

Design note (see ADR): the CaseCard has a *general* core that applies to any
judgment corpus, plus an optional `facts` block whose fields are populated only
when the judgment happens to contain them. Nothing here is specific to the
particular client matter -- a brief is runtime input, never a schema field.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Corpus layer
# =============================================================================


class Favours(str, Enum):
    """Which side a judgment's outcome helps. Drives adverse-precedent surfacing."""

    CLAIMANT = "claimant"
    INSURER = "insurer"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class CaseFacts(BaseModel):
    """Domain-specific factual matrix. Every field optional -- absent when N/A."""

    vehicle_type: str | None = None
    is_commercial_vehicle: bool | None = None
    driver_licence_defect: str | None = Field(
        None, description="e.g. 'no licence', 'fake licence', 'expired', 'wrong class', or null"
    )
    owner_knowledge_alleged: bool | None = None
    deceased_or_injured_age: int | None = None
    monthly_income: float | None = None
    dependents_count: int | None = None
    is_death_claim: bool | None = None
    contributory_negligence_found: bool | None = None


class Quantum(BaseModel):
    """How the court computed compensation, when it did."""

    multiplier: float | None = None
    future_prospects_pct: float | None = None
    personal_expense_deduction: str | None = None
    total_awarded: float | None = None
    interest_rate_pct: float | None = None


class CaseCard(BaseModel):
    """One structured record per judgment. Built once, offline, then committed.

    This is what turns 'semantic similarity over prose' into 'structured
    reasoning over a small knowledge base' -- and it is what makes exhaustive
    structured queries ('which judgments involve commercial vehicles?')
    answerable exactly rather than probabilistically.
    """

    doc_id: str
    title: str
    court: str | None = None
    decided_on: str | None = None
    bench: str | None = None
    source_url: str | None = None

    # --- General legal core (corpus-agnostic) --------------------------------
    case_type: str | None = None
    legal_issues: list[str] = Field(default_factory=list)
    statutes_cited: list[str] = Field(default_factory=list)
    precedents_cited: list[str] = Field(default_factory=list)
    holding: str | None = None
    ratio: str | None = Field(None, description="The binding principle, not obiter")
    disposition: str | None = None
    outcome_favours: Favours = Favours.NEUTRAL
    key_principles: list[str] = Field(default_factory=list)

    # --- Optional domain blocks ----------------------------------------------
    facts: CaseFacts = Field(default_factory=CaseFacts)
    quantum: Quantum = Field(default_factory=Quantum)

    # --- Bookkeeping ---------------------------------------------------------
    n_pages: int = 0
    n_chars: int = 0

    # Extraction models are inconsistent about scalar-vs-list for these fields
    # (a two-judge bench naturally wants to be a list). Coerce rather than fail:
    # losing a whole case card over a formatting choice is the wrong trade.
    @field_validator("court", "decided_on", "bench", "case_type", "holding", "ratio",
                     "disposition", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v if x) or None
        return v

    @field_validator("legal_issues", "statutes_cited", "precedents_cited", "key_principles",
                     mode="before")
    @classmethod
    def _coerce_list(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("facts", "quantum", mode="before")
    @classmethod
    def _coerce_block(cls, v: Any) -> Any:
        # A model with nothing to report sometimes emits null, "", or [].
        # These fields are non-optional, so fall back to an empty block.
        if isinstance(v, dict) or isinstance(v, (CaseFacts, Quantum)):
            return v
        return {}

    @field_validator("outcome_favours", mode="before")
    @classmethod
    def _coerce_favours(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip().lower()
            return s if s in {f.value for f in Favours} else "neutral"
        return v

    def summary_line(self) -> str:
        """Compact one-liner used in full-corpus screening prompts."""
        bits = [f"[{self.doc_id}] {self.title}"]
        if self.court:
            bits.append(f"({self.court}, {self.decided_on or 'n.d.'})")
        if self.holding:
            bits.append(f"HELD: {self.holding}")
        bits.append(f"FAVOURS: {self.outcome_favours.value}")
        return " | ".join(bits)


class Chunk(BaseModel):
    """A retrievable passage, carrying enough context to stand alone."""

    chunk_id: str
    doc_id: str
    section: str
    para_start: str | None = None
    text: str
    context_header: str = Field(
        "", description="Prepended case/court/section context (contextual retrieval)"
    )

    @property
    def embed_text(self) -> str:
        return f"{self.context_header}\n\n{self.text}".strip()


# =============================================================================
# Retrieval layer
# =============================================================================


class ScoredDoc(BaseModel):
    """A retrieval hit with its score fully decomposed -- this is what the UI
    renders and what the eval framework reads. Never collapse to one number."""

    doc_id: str
    title: str = ""
    dense_score: float | None = None
    dense_rank: int | None = None
    sparse_score: float | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None
    final_rank: int | None = None
    why: str = Field("", description="Reranker's stated reason for the score")
    best_passage: str = ""


# =============================================================================
# Output contracts -- the agent elects one of these by calling a terminal tool
# =============================================================================


class PrecedentAnalysis(BaseModel):
    doc_id: str
    title: str
    principle: str = Field(..., description="Legal principle this judgment establishes")
    fact_alignment: str = Field(..., description="Which brief facts align, specifically")
    why_it_matters: str
    strength: Literal["strong", "moderate", "weak"]
    quote: str = Field("", description="Verbatim supporting line from the judgment")


class AdversePrecedent(BaseModel):
    """A judgment the OPPOSING side will cite.

    The admission test is "will they cite it", NOT "does it defeat us". An earlier
    version used the second test and measured badly: strict adverse recall fell to
    0.0, because the agent correctly distinguished the damaging judgments and then
    filed them under `caveats` as "not adverse, different facts". Legally sound,
    practically useless -- opposing counsel cites them regardless, and the reader
    is left unprepared for an argument that is certainly coming.

    So distinguishable authority belongs HERE, with the distinction stated. Only
    judgments the other side has no reason to raise at all belong in `caveats`.
    """

    doc_id: str
    title: str
    principle: str
    risk_to_client: str = Field(
        ...,
        description=(
            "What the opposing side will DO with this judgment -- the argument they "
            "will build from it. Include it even if that argument ultimately fails; "
            "say so, and say why. The reader needs to know the argument is coming."
        ),
    )
    risk_level: Literal["high", "medium", "low"] = Field(
        ...,
        description=(
            "How hard it is to ANSWER, not whether you win. "
            "high = could defeat or gut the claim on its own; "
            "medium = forces a real argument that could go either way; "
            "low = they will cite it, you have a clean answer, but you must have "
            "that answer ready. 'low' does NOT mean omit it. Differentiate: if "
            "everything is 'medium' you have not assessed anything."
        ),
    )
    distinguishing_argument: str = Field(
        ..., description="How to counter or distinguish it on the facts or the law"
    )
    quote: str = ""


class Strategy(BaseModel):
    priority_arguments: list[str]
    compensation_range: str
    compensation_reasoning: str = ""
    risks: list[str]
    recommended_forum_or_relief: str = ""


class PrecedentResearchReport(BaseModel):
    """Terminal contract for deep research tasks."""

    question: str
    supporting: list[PrecedentAnalysis]
    adverse: list[AdversePrecedent]
    strategy: Strategy
    caveats: list[str] = Field(default_factory=list)


class DirectAnswer(BaseModel):
    """Terminal contract for general/simple queries."""

    question: str
    answer: str
    cited_doc_ids: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
