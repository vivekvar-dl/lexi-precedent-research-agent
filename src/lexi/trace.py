"""Typed trace events.

Dual-purpose by design:
  1. Streamlit renders these live -- satisfying "intermediate reasoning steps
     must be visible; we want to see which documents the agent retrieved, how it
     ranked them, and how it arrived at its conclusions".
  2. The eval framework reads the SAME objects to compute retrieval precision,
     recall and citation faithfulness.

One artifact, two consumers. That is why retrieval events carry the full score
decomposition rather than a final ordering.
"""
from __future__ import annotations

import json
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .schemas import ScoredDoc


class EventKind(str, Enum):
    PLAN = "plan"                  # agent's stated intent for this step
    TOOL_CALL = "tool_call"        # a tool was invoked, with arguments
    RETRIEVAL = "retrieval"        # ranked documents + score decomposition
    FILTER = "filter"              # exact structured query + how many matched
    SCREEN = "screen"              # full-corpus LLM screening pass
    READ = "read"                  # a judgment was opened
    COMPUTE = "compute"            # deterministic calculation (e.g. quantum)
    BUDGET = "budget"              # budget set or escalated
    ANSWER = "answer"              # terminal contract emitted
    ERROR = "error"


class TraceEvent(BaseModel):
    seq: int
    kind: EventKind
    label: str
    detail: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    docs: list[ScoredDoc] = Field(default_factory=list)
    elapsed_s: float = 0.0


class Trace(BaseModel):
    """Ordered event log for one agent run."""

    question: str
    events: list[TraceEvent] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.time)
    llm_calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0

    def add(
        self,
        kind: EventKind,
        label: str,
        detail: str = "",
        payload: dict | None = None,
        docs: list[ScoredDoc] | None = None,
    ) -> TraceEvent:
        ev = TraceEvent(
            seq=len(self.events) + 1,
            kind=kind,
            label=label,
            detail=detail,
            payload=payload or {},
            docs=docs or [],
            elapsed_s=round(time.time() - self.started_at, 2),
        )
        self.events.append(ev)
        return ev

    # --- views used by the eval framework ------------------------------------

    def retrieved_doc_ids(self) -> list[str]:
        """Every doc the agent actually saw, in first-seen order.

        This is the denominator for retrieval recall -- distinct from what the
        agent finally chose to cite.
        """
        seen: list[str] = []
        for ev in self.events:
            for d in ev.docs:
                if d.doc_id not in seen:
                    seen.append(d.doc_id)
            for did in ev.payload.get("doc_ids", []):
                if did not in seen:
                    seen.append(did)
        return seen

    def final_ranking(self) -> list[ScoredDoc]:
        """The last ranked retrieval -- used for nDCG / P@k."""
        for ev in reversed(self.events):
            if ev.kind in (EventKind.RETRIEVAL, EventKind.SCREEN) and ev.docs:
                return ev.docs
        return []

    def high_confidence_docs(self, threshold: float) -> dict[str, float]:
        """Documents the RERANKER scored at or above `threshold`, best score kept.

        This is the system's own opinion of what matters, and it is what makes a
        synthesis check possible without a gold set: if the reranker rated a
        judgment 8/10 and the report never mentions it, that is the agent
        disagreeing with its own retrieval -- which it should have to justify.

        Measured motivation: retrieval recall 92.9%, answer recall 58.0%. The
        agent finds the right judgment, reads it, then drops it from the writeup
        roughly a third of the time.
        """
        best: dict[str, float] = {}
        for ev in self.events:
            for d in ev.docs:
                if d.rerank_score is None:
                    continue
                if d.rerank_score >= threshold:
                    best[d.doc_id] = max(best.get(d.doc_id, 0.0), d.rerank_score)
        return best

    def tool_sequence(self) -> list[str]:
        return [e.label for e in self.events if e.kind == EventKind.TOOL_CALL]

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=1, ensure_ascii=False)
