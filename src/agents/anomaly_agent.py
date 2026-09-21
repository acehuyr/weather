"""Anomaly agent: is anything here actually unusual?"""
from __future__ import annotations

from src.agents.base import Agent
from src.analysis import anomalies
from src.core.models import WeatherBundle


class AnomalyAgent(Agent):
    name = "anomaly"
    description = "Detects statistically unusual and threshold-breaching values"

    def execute(self, context: dict) -> dict:
        bundle: WeatherBundle = context["bundle"]
        if bundle is None:
            return {"status": "no_data", "anomalies": []}

        # Prefer observed history; fall back to the forecast so a
        # forward-looking question can still be screened for extremes.
        recent = bundle.history
        if recent is None or recent.empty:
            recent = bundle.forecast

        result = anomalies.detect(recent, bundle.baseline)
        result["threshold_flags"] = anomalies.flag_extremes(recent)

        # Be explicit about what the comparison was made against - "unusual
        # for this time of year" and "unusual for the last few weeks" are
        # very different claims.
        has_baseline = bundle.baseline is not None and not bundle.baseline.empty
        result["baseline_used"] = (
            "multi-year climatological normal" if has_baseline
            else "the observed window only (no multi-year baseline available)"
        )
        return result
