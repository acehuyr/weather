"""Data agent: turn a plan into actual observations.

Fetches only what the plan asks for. A "what is humidity" question triggers
zero network calls; an anomaly question pulls a decade of baseline data. That
selectivity is the point of planning first.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from config.settings import settings
from src.agents.base import Agent
from src.core.errors import LocationNotFound, ProviderError
from src.core.models import QueryPlan, WeatherBundle
from src.ingestion import open_meteo
from src.ingestion.geocoding import geocode


def _parse_date(value: str, fallback: date) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return fallback


class DataAgent(Agent):
    name = "data"
    description = "Fetches current, forecast and historical weather"

    def execute(self, context: dict) -> WeatherBundle:
        plan: QueryPlan = context["plan"]

        location = self._resolve(plan, context)
        bundle = WeatherBundle(location=location)

        # A conceptual question needs no observations at all.
        if plan.intent == "general" and not plan.needs_history:
            bundle.sources.append("none required")
            return bundle

        if plan.needs_forecast or plan.intent in {"current", "forecast"}:
            self._safe(bundle, "current conditions",
                       lambda: setattr(bundle, "current",
                                       open_meteo.fetch_current(location)))
            self._safe(bundle, "daily forecast",
                       lambda: setattr(bundle, "forecast",
                                       open_meteo.fetch_forecast(location)))

        if plan.needs_history:
            end = _parse_date(plan.end_date, date.today())
            start = _parse_date(plan.start_date, end - timedelta(days=30))
            self._safe(bundle, f"history {start} to {end}",
                       lambda: setattr(bundle, "history",
                                       open_meteo.fetch_history(location, start, end)))

            # Anomaly detection needs a climatological normal to compare
            # against; a few weeks of data cannot tell you what is normal.
            if plan.intent == "anomaly":
                base_start = end.replace(year=end.year - settings.baseline_years)
                self._safe(
                    bundle, f"{settings.baseline_years}-year baseline",
                    lambda: setattr(bundle, "baseline",
                                    open_meteo.fetch_history(location, base_start, end)),
                )

            if plan.comparison_years > 0:
                self._safe(
                    bundle, f"same window across {plan.comparison_years} years",
                    lambda: setattr(
                        bundle, "baseline",
                        open_meteo.fetch_same_window_across_years(
                            location, start, end, plan.comparison_years
                        ),
                    ),
                )

        self.log.info("fetched: %s", bundle.describe())
        return bundle

    def _resolve(self, plan: QueryPlan, context: dict):
        """Geocode the planned location, falling back to the selected one.

        The planner guesses at place names it has never seen, so "nagaur"
        reaches here unverified. If the geocoder cannot find it, using the
        user's selected location is better than failing - but the plan is
        corrected so the answer names the place it actually describes,
        rather than silently talking about somewhere else.
        """
        try:
            return geocode(plan.location)
        except LocationNotFound:
            fallback = context.get("default_location")
            if not fallback or fallback.lower() == plan.location.lower():
                raise
            self.log.warning("unknown place %r - falling back to %r",
                             plan.location, fallback)
            location = geocode(fallback)
            plan.location = location.name
            plan.rationale += (
                f" Could not find that place; used {location.name} instead."
            )
            return location

    def _safe(self, bundle: WeatherBundle, label: str, fn) -> None:
        """One failed fetch should not lose the fetches that succeeded."""
        try:
            fn()
            bundle.sources.append(label)
        except ProviderError as exc:
            self.log.warning("could not fetch %s: %s", label, exc)
            bundle.sources.append(f"{label} (FAILED: {exc})")
