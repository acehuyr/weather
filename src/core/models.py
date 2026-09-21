"""Shared data contracts.

Two families live here on purpose:

* **Pydantic models** -> anything that crosses the LLM boundary. The planner
  asks Claude to emit a `QueryPlan`; structured outputs validate it for us.
* **Dataclasses** -> internal carriers that hold pandas objects, which
  Pydantic should not try to serialise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

Intent = Literal[
    "current",      # what is it like right now
    "forecast",     # what will happen in the next N days
    "trend",        # how has X moved over time
    "comparison",   # this period vs. a previous period
    "anomaly",      # is anything unusual / why is it unusual
    "general",      # conceptual question, answered mostly from the knowledge base
]

# Canonical variable names used across the whole pipeline.
VARIABLES = {
    "temperature": "temperature_2m",
    "humidity": "relative_humidity_2m",
    "rainfall": "precipitation",
    "wind_speed": "wind_speed_10m",
    "pressure": "surface_pressure",
}


# --------------------------------------------------------------------------
# LLM-facing schemas (Pydantic)
# --------------------------------------------------------------------------

class QueryPlan(BaseModel):
    """What the Planner agent decides before any data is fetched.

    Every field is required with a sentinel for "not specified" ("" or 0 or
    []). Strict JSON-schema validation is far more reliable that way than
    with optional/nullable fields.
    """

    intent: Intent = Field(description="The single best-fitting intent for the question.")
    location: str = Field(description="Place name, e.g. 'Mumbai'. Empty string if none given.")
    start_date: str = Field(description="YYYY-MM-DD, or '' to let the system choose.")
    end_date: str = Field(description="YYYY-MM-DD, or '' to let the system choose.")
    variables: list[str] = Field(
        description="Subset of: temperature, humidity, rainfall, wind_speed, pressure. "
                    "Empty list means 'pick sensible defaults'."
    )
    comparison_years: int = Field(
        description="How many previous years to compare against. 0 if no comparison is asked for."
    )
    needs_history: bool = Field(description="True if past observations are required.")
    needs_forecast: bool = Field(description="True if future predictions are required.")
    needs_knowledge: bool = Field(description="True if meteorological background would help.")
    rationale: str = Field(description="One sentence explaining the routing decision.")


class Insight(BaseModel):
    """The final synthesised answer."""

    answer: str = Field(description="The explanation, in plain language, 3-6 sentences.")
    key_findings: list[str] = Field(description="2-5 short bullet points, each a concrete fact.")
    recommendations: list[str] = Field(description="0-3 actionable suggestions. May be empty.")
    confidence: Literal["high", "medium", "low"] = Field(
        description="How well the retrieved data actually supports the answer."
    )


# --------------------------------------------------------------------------
# Internal carriers (dataclasses - they hold DataFrames)
# --------------------------------------------------------------------------

@dataclass
class Location:
    name: str
    latitude: float
    longitude: float
    country: str = ""
    admin1: str = ""
    timezone: str = "auto"

    @property
    def label(self) -> str:
        bits = [self.name, self.admin1, self.country]
        return ", ".join(b for b in bits if b)


@dataclass
class WeatherBundle:
    """Everything the Data agent managed to fetch for one question."""

    location: Location
    current: dict[str, Any] = field(default_factory=dict)
    forecast: Optional[pd.DataFrame] = None     # hourly or daily, future
    history: Optional[pd.DataFrame] = None      # daily, past
    baseline: Optional[pd.DataFrame] = None     # multi-year daily, for anomaly context
    sources: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [f"location={self.location.label}"]
        for name in ("forecast", "history", "baseline"):
            df = getattr(self, name)
            if df is not None and not df.empty:
                parts.append(f"{name}={len(df)} rows")
        return " ".join(parts)


@dataclass
class Document:
    """One retrievable chunk of the knowledge base."""

    id: str
    text: str
    source: str
    title: str = ""
    score: float = 0.0


@dataclass
class AgentResult:
    """Uniform envelope returned by every agent - powers the trace view."""

    agent: str
    ok: bool
    data: Any = None
    error: str = ""
    duration_ms: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "ok": self.ok,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
            "note": self.note,
        }


@dataclass
class PipelineResult:
    """What the orchestrator hands back to the API / UI."""

    question: str
    plan: Optional[QueryPlan] = None
    insight: Optional[Insight] = None
    # The conversational reply. Same grounded facts as `insight`, written as
    # prose instead of a structured report.
    reply: str = ""
    bundle: Optional[WeatherBundle] = None
    analysis: dict[str, Any] = field(default_factory=dict)
    anomalies: dict[str, Any] = field(default_factory=dict)
    documents: list[Document] = field(default_factory=list)
    trace: list[AgentResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def ok(self) -> bool:
        return self.insight is not None
