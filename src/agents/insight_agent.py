"""Insight agent: the G in RAG.

Takes everything the pipeline produced and writes the answer. Two paths:

* LLM path - Claude receives the computed numbers and the retrieved
  reference text, and is instructed to use nothing else.
* Offline path - a deterministic template built from the same numbers. It is
  blunter, but it never invents anything, and it keeps the whole system
  demonstrable without an API key.
"""
from __future__ import annotations

from datetime import date

from src.agents.base import Agent
from src.core.errors import LLMError
from src.core.models import Insight, QueryPlan, WeatherBundle
from src.llm.client import get_llm
from src.llm.prompts import INSIGHT_SYSTEM, build_insight_prompt


class InsightAgent(Agent):
    name = "insight"
    description = "Generates the final grounded explanation"

    def summarise(self, context: dict) -> Insight:
        """Structured fields only, with no language model call.

        In conversational mode the Chat agent has already written the prose,
        and all that is still needed is the confidence rating and the bullet
        findings for the evidence panel. Running the full LLM insight there
        too meant a second round trip per question whose text was thrown
        away - it was costing more than every other agent combined.
        """
        return self._template(
            context["plan"],
            context.get("bundle"),
            context.get("analysis") or {},
            context.get("anomalies") or {},
            context.get("documents") or [],
        )

    def execute(self, context: dict) -> Insight:
        plan: QueryPlan = context["plan"]
        bundle: WeatherBundle = context.get("bundle")
        analysis: dict = context.get("analysis") or {}
        anomalies: dict = context.get("anomalies") or {}
        documents: list = context.get("documents") or []

        llm = get_llm()
        if llm.available:
            try:
                return self._generate(llm, context, plan, bundle, analysis,
                                      anomalies, documents)
            except LLMError as exc:
                self.log.warning("generation failed (%s) - using template", exc)

        return self._template(plan, bundle, analysis, anomalies, documents)

    # ------------------------------------------------------------------
    def _generate(self, llm, context, plan, bundle, analysis, anomalies,
                  documents) -> Insight:
        stat_keys = {"history_summary", "forecast_summary",
                     "rainfall", "forecast_rainfall"}
        prompt = build_insight_prompt(
            question=context["question"],
            today=date.today().isoformat(),
            location=bundle.location.label if bundle else plan.location,
            plan_summary=f"{plan.intent} ({plan.rationale})",
            current=(bundle.current if bundle else {}) or {},
            statistics={k: v for k, v in analysis.items() if k in stat_keys},
            trends=analysis.get("trends") or {},
            comparison=analysis.get("year_over_year") or {},
            anomalies={k: v for k, v in anomalies.items() if k != "threshold_flags"},
            extremes=anomalies.get("threshold_flags") or [],
            documents=documents,
        )
        return llm.parse(system=INSIGHT_SYSTEM, user=prompt, schema=Insight)

    # ------------------------------------------------------------------
    # Words that mark a finding as being about a given requested variable.
    VARIABLE_TERMS = {
        "temperature": ("temperature", "C,", "heat", "warm", "cold"),
        "rainfall": ("rain", "rainfall", "precipitation", "mm"),
        "humidity": ("humid", "humidity"),
        "wind_speed": ("wind", "gust"),
        "pressure": ("pressure", "hpa"),
    }

    @classmethod
    def _prioritise(cls, findings: list, variables: list) -> list:
        """Stable sort putting findings about the requested variables first."""
        if not variables:
            return findings
        terms: tuple = tuple(
            term
            for variable in variables
            for term in cls.VARIABLE_TERMS.get(variable, (variable,))
        )

        def relevant(text: str) -> int:
            lowered = text.lower()
            return 0 if any(term.lower() in lowered for term in terms) else 1

        return sorted(findings, key=lambda f: relevant(f))

    # ------------------------------------------------------------------
    def _template(self, plan, bundle, analysis, anomalies, documents) -> Insight:
        """Deterministic answer. Reports only what was actually measured."""
        place = bundle.location.label if bundle else plan.location
        lines: list = []
        findings: list = []
        recommendations: list = []

        current = (bundle.current if bundle else {}) or {}
        if current:
            temp = current.get("temperature_2m")
            hum = current.get("relative_humidity_2m")
            cond = str(current.get("condition", "")).lower()
            lines.append(
                f"Right now in {place} it is {temp} C with {hum} percent "
                f"relative humidity ({cond})."
            )
            feels = current.get("apparent_temperature")
            if feels is not None and temp is not None and abs(feels - temp) >= 2:
                findings.append(
                    f"It feels like {feels} C rather than {temp} C, because "
                    "humidity and wind change how the body loses heat."
                )

        # Distinguish observed history from forecast: describing predicted
        # days in the past tense is the fastest way to lose a reader's trust.
        historical = analysis.get("history_summary")
        summary = historical or analysis.get("forecast_summary")
        if summary:
            variables = summary.get("variables", {})
            start, end = summary.get("start"), summary.get("end")
            rows = summary.get("rows")
            if historical:
                lines.append(
                    f"Across the {rows} days observed in {place} "
                    f"({start} to {end}):"
                )
            else:
                lines.append(
                    f"The {rows}-day forecast for {place} "
                    f"({start} to {end}) shows:"
                )
            labels = [
                ("temperature_2m_max", "daily maximum temperature"),
                ("temperature_2m_min", "daily minimum temperature"),
            ]
            verb = "averaged" if historical else "averages"
            for key, label in labels:
                if key in variables:
                    v = variables[key]
                    findings.append(
                        f"The {label} {verb} {v['mean']} C, ranging from "
                        f"{v['min']} C to {v['max']} C."
                    )

        rain = analysis.get("rainfall") or analysis.get("forecast_rainfall")
        if rain:
            wet = rain["wet_days"]
            # "0 rainy days" alongside a non-zero total looks like a bug to a
            # reader, so name the threshold that produced the count.
            if wet:
                findings.append(
                    f"Rainfall totals {rain['total_mm']} mm over {wet} rainy "
                    f"day(s) of 2.5 mm or more; the wettest recorded "
                    f"{rain['heaviest_day_mm']} mm."
                )
            else:
                findings.append(
                    f"Rainfall totals {rain['total_mm']} mm, but no single day "
                    f"reaches the 2.5 mm rainy-day threshold "
                    f"(heaviest: {rain['heaviest_day_mm']} mm)."
                )

        for var, t in (analysis.get("trends") or {}).items():
            if not isinstance(t, dict) or t.get("status") != "ok":
                continue
            if t["direction"] == "stable":
                continue
            findings.append(
                f"{var} is {t['direction']} at about {t['slope_per_decade']} "
                f"units per decade (R-squared {t['r_squared']}, "
                f"{t['confidence']} confidence)."
            )

        for var, c in (analysis.get("year_over_year") or {}).items():
            if isinstance(c, dict) and "percent_change" in c:
                findings.append(
                    f"{var} is {c['verdict']}: {c['current']} against a "
                    f"historical mean of {c['historical_mean']} "
                    f"({c['percent_change']:+.1f} percent)."
                )

        total = anomalies.get("total", 0)
        if total:
            top = anomalies["anomalies"][0]
            baseline = anomalies.get("baseline_used", "the available baseline")
            lines.append(
                f"{total} unusual reading(s) were detected against {baseline}."
            )
            z = abs(top.get("z_score") or 0)
            findings.append(
                f"The most extreme was {top['variable']} on {top['date']} at "
                f"{top['value']}, about {z:.1f} standard deviations "
                f"{top['direction']} the seasonal normal."
            )
        elif anomalies.get("status") == "ok":
            lines.append("Nothing in this period was statistically unusual.")

        for flag in (anomalies.get("threshold_flags") or [])[:3]:
            findings.append(f"{flag['date']}: {flag['category']}.")
            if flag.get("type") == "rainfall":
                recommendations.append(
                    "Check local advisories before travelling on heavy-rain days."
                )
            elif flag.get("type") == "heat":
                recommendations.append(
                    "Limit outdoor activity in the afternoon and increase fluid intake."
                )

        if documents:
            sources = sorted({d.source for d in documents})
            lines.append(f"Background drawn from: {', '.join(sources)}.")

        if not lines and not findings:
            lines.append(
                f"No usable weather data could be retrieved for {place}, so the "
                "question could not be answered from measurements."
            )

        # The findings list is truncated to five, so what the user actually
        # asked about must sort to the front - otherwise a rainfall question
        # gets answered with temperature statistics.
        findings = self._prioritise(findings, plan.variables)

        confidence = "low"
        if findings and (analysis.get("history_summary") or current):
            confidence = "medium" if total or len(findings) < 3 else "high"

        # No "generated without a language model" footnote here. Whether an
        # LLM was used is a property of the system, not part of the answer -
        # it belongs in the UI status area, and pasting it into the prose
        # also corrupted the groundedness check by adding stray tokens.
        return Insight(
            answer=" ".join(lines),
            key_findings=findings[:5],
            # dict.fromkeys de-duplicates while preserving order.
            recommendations=list(dict.fromkeys(recommendations))[:3],
            confidence=confidence,
        )
