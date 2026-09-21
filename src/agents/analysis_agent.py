"""Analysis agent: statistics, trends and year-over-year comparison."""
from __future__ import annotations

from src.agents.base import Agent
from src.analysis import statistics, trends
from src.core.errors import AnalysisError
from src.core.models import QueryPlan, WeatherBundle


class AnalysisAgent(Agent):
    name = "analysis"
    description = "Computes statistics, trends and comparisons"

    def execute(self, context: dict) -> dict:
        bundle: WeatherBundle = context["bundle"]
        plan: QueryPlan = context["plan"]

        if bundle is None:
            raise AnalysisError("No weather data available to analyse.")

        result: dict = {}

        if bundle.history is not None and not bundle.history.empty:
            result["history_summary"] = statistics.summarize(bundle.history)
            result["rainfall"] = statistics.rainfall_profile(bundle.history)
            # A trend fit is only meaningful over a reasonable span; below a
            # month it is noise, and saying so is better than fitting it.
            if len(bundle.history) >= 30:
                result["trends"] = trends.analyse_trends(bundle.history)
            else:
                result["trends"] = {
                    "status": "skipped",
                    "reason": f"only {len(bundle.history)} days of data; "
                              "at least 30 are needed for a meaningful fit",
                }

        if bundle.forecast is not None and not bundle.forecast.empty:
            result["forecast_summary"] = statistics.summarize(bundle.forecast)
            result["forecast_rainfall"] = statistics.rainfall_profile(bundle.forecast)

        if plan.comparison_years > 0 and bundle.baseline is not None:
            result["year_over_year"] = trends.year_over_year(
                bundle.history, bundle.baseline
            )

        if not result:
            raise AnalysisError("No usable numeric data was returned for this query.")

        return result
