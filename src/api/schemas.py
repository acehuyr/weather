"""Request and response schemas for the HTTP API.

These are deliberately separate from the internal models in
`src.core.models`: the wire format should be free to stay stable while the
internals change, and DataFrames must never leak into a JSON response.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500,
                          examples=["What is the weather trend in Mumbai?"])


class PlanOut(BaseModel):
    intent: str
    location: str
    variables: list
    start_date: str
    end_date: str
    rationale: str


class DocumentOut(BaseModel):
    source: str
    title: str
    score: float
    excerpt: str


class TraceStep(BaseModel):
    agent: str
    ok: bool
    duration_ms: float
    note: str = ""
    error: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str
    key_findings: list
    recommendations: list
    confidence: str
    location: str = ""
    plan: Optional[PlanOut] = None
    sources: list = []
    trace: list = []
    # Chart-ready series so a client can plot without re-fetching anything.
    series: dict = {}


class HealthResponse(BaseModel):
    status: str
    llm_enabled: bool
    model: str
    indexed_chunks: int
    embedding_backend: str
    vector_backend: str


def series_from_bundle(bundle: Any) -> dict:
    """Extract plottable {label: {dates: [...], values: [...]}} series."""
    if bundle is None:
        return {}
    out: dict = {}
    for name in ("history", "forecast"):
        df = getattr(bundle, name, None)
        if df is None or df.empty:
            continue
        block: dict = {"dates": [d.date().isoformat() for d in df.index]}
        for col in df.columns:
            if col == "year":
                continue
            values = df[col].tolist()
            block[col] = [None if v != v else round(float(v), 2) for v in values]
        out[name] = block
    return out
