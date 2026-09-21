"""Planner agent: natural language -> structured execution plan.

This is the agentic core. Nothing else in the pipeline reads the raw
question; every downstream agent reads the QueryPlan instead. That means the
system's behaviour is inspectable: you can look at the plan and know exactly
what it is about to do before it does it.

Two implementations share one output contract:
  * LLM planning via structured outputs (accurate, handles paraphrase)
  * rule-based planning (deterministic, free, works offline)
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from config.settings import settings
from src.agents.base import Agent
from src.core.errors import LLMError
from src.core.models import QueryPlan
from src.ingestion.geocoding import _FALLBACK
from src.llm.client import get_llm
from src.llm.prompts import PLANNER_SYSTEM, build_planner_prompt

# Number words, so "the next ten days" is as matchable as "the next 10 days".
# Written-out counts are how people actually type, and the original
# `next \d+ days` silently missed every one of them.
_COUNT = (r"(?:\d+|a few|several|one|two|three|four|five|six|seven|eight|"
          r"nine|ten|eleven|twelve|fourteen|fifteen)")
_PERIOD = r"(?:days?|weeks?|weekends?|months?|fortnights?)"

# Checked in order - the first intent whose pattern matches wins, so the more
# specific intents must come first.
#
# Failure modes the evaluation harness caught, now handled here:
#
#   * A trailing \b silently kills stem matching. "anomal" never matched
#     "anomaly", because \b demands a non-word character after the stem.
#     Stems now carry explicit optional suffixes instead.
#   * Definitional questions ("what counts as a heat wave?") matched nothing
#     at all and landed on the "current" default. The unambiguous ones are
#     checked first; the weaker phrasing stays last so it cannot outrank a
#     real data request.
#   * The vocabulary was far too narrow. On the v1 held-out split only 4 of
#     12 questions reached the right intent, and every single miss collapsed
#     to "current" - the default - because no pattern matched at all. That
#     is the worst possible failure shape here: nothing downstream reads the
#     raw question, so a forecast question planned as "current" fetches the
#     wrong window and answers confidently about the wrong thing rather than
#     degrading visibly.
#
#     The fix deliberately adds whole synonym *families* per intent rather
#     than the specific phrases that failed. Patching in "outlook" alone
#     would move the score without moving the capability; the point is that
#     "prognosis" and "the days ahead" should work too, and neither appears
#     in any labelled case.
INTENT_PATTERNS = [
    # Phrasing that cannot be a request for measurements.
    ("general", r"\b(what is an?|what counts as|what does .+ mean|"
                r"what do .+ mean|meaning of|define|definition|"
                r"officially (?:called|classified|defined)|"
                r"(?:gets?|get |is |be )classified as|"
                r"explain how|how does .+ work|"
                r"how many years of data|enough to (?:prove|claim|conclude)|"
                r"at what point|difference between|line between|"
                r"talk me through|walk me through|"
                r"in simple terms|put simply|in plain english)\b"),
    ("anomaly", r"\b(unusual(?:ly)?|abnormal(?:ly)?|anomal(?:y|ies|ous)|"
                r"strange(?:ly)?|odd(?:ly)?|weird|freak(?:ish|y)?|bizarre|"
                r"why is|why are|why was|"
                r"extreme(?:ly)?|record|unprecedented|"
                r"out of (?:the )?(?:ordinary|line|character|whack)|"
                r"not normal|nothing normal|"
                r"(?:look|feel|seem)(?:s|ing|ed)? (?:off|wrong|strange|odd)|"
                r"(?:something|anything) (?:wrong|off|odd|strange)|"
                r"wrong with)\b"),
    ("comparison", r"\b(compare[ds]?|comparison|versus|vs\.?|"
                   r"last year|previous year|past years|earlier years|"
                   r"prior years|"
                   r"than before|than (?:it |they )?(?:was|were)|"
                   r"than (?:usual|normal|typical)|"
                   r"(?:stack|line|weigh|measure|set|put|pit)(?:s|ed|d)?"
                   r" .{0,40}?(?:up )?against|"
                   r"stack(?:s|ed)? up against|"
                   r"beside the|next to (?:what|the)|side by side|"
                   r"usually sees|typical year|"
                   rf"{_COUNT} (?:months?|years?) (?:ago|back)|"
                   r"same (?:time|period|month) (?:last|previous))\b"),
    ("trend", r"\b(trends?|"
              r"(?:over|across|during|through) the "
              r"(?:last |past )?(?:years|decades?|century)|"
              r"(?:last|past) (?:decade|century)|"
              r"changing|changed|long[- ]term|historic(?:al|ally)?|"
              r"pattern over|over time|"
              r"this year|so far this year|year to date|"
              r"since the (?:\d{4}s?|nineties|eighties|seventies|sixties)|"
              r"than it used to|used to be|"
              r"drift(?:s|ed|ing)?|"
              r"becom(?:e|es|ing) (?:hotter|colder|wetter|drier|rainier|"
              r"warmer|cooler)|"
              r"more .{0,25}? now than|"
              r"getting (?:hotter|colder|wetter|drier|warmer|cooler|"
              r"rainier))\b"),
    ("forecast", r"\b(forecast|outlook|prognosis|will|going to|"
                 r"expects?|expected|"
                 rf"next (?:{_COUNT}\s+)?{_PERIOD}|"
                 rf"(?:coming|upcoming|following) (?:{_COUNT}\s+)?{_PERIOD}|"
                 r"this (?:week|weekend|month)|over the (?:weekend|week)|"
                 rf"{_PERIOD} ahead|"
                 r"tomorrow|tonight|later today|coming days|"
                 r"upcoming|predict|due to|likely to|chance of)\b"),
    ("current", r"\b(right now|currently|current|at the moment|at present|"
                r"this (?:very )?(?:moment|second|minute)|"
                r"right this (?:minute|second)|"
                r"today'?s weather|how is the weather|what is the weather|"
                r"how'?s the weather|out there)\b"),
    # Weaker definitional phrasing - only after every data intent has missed.
    ("general", r"\b(what is|what are|what makes|what causes|"
                r"explain|how does|how do|why does|why do)\b"),
]


# Matched by the locative regex but not actually a place we can geocode to a
# useful point. "A heat wave in India" is a definition, not a city query.
NON_CITY = {
    "india", "asia", "europe", "africa", "america", "australia",
    "the world", "earth", "north india", "south india", "the country",
}

# Words that follow "in"/"for" without naming a place. Without this list the
# case-insensitive pass below turns "rain in the morning" into a hunt for a
# town called "the".
NOT_A_PLACE = NON_CITY | {
    "the", "this", "that", "these", "those", "last", "next", "past", "recent",
    "coming", "future", "general", "summer", "winter", "spring", "autumn",
    "monsoon", "morning", "evening", "afternoon", "night", "today",
    "tomorrow", "yesterday", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
    "my", "our", "your", "his", "her", "their", "a", "an", "celsius",
    "fahrenheit", "degrees", "percent", "days", "weeks", "months", "years",
    "my area", "my city", "my location", "here", "there",
}

LOCATIVE = re.compile(
    r"\b(?:in|at|for|around|near|over|across)\s+"
    r"([A-Za-z][a-zA-Z]+(?:[\s-][A-Za-z][a-zA-Z]+)?)"
)

VARIABLE_PATTERNS = {
    "temperature": r"\b(temperature|temp|hot|cold|heat|warm|cool|degrees)\b",
    "rainfall": r"\b(rain|rainfall|precipitation|monsoon|wet|drought|shower)\b",
    "humidity": r"\b(humid|humidity|muggy|moisture|damp)\b",
    "wind_speed": r"\b(wind|windy|gust|breeze|storm)\b",
    "pressure": r"\b(pressure|barometric|hpa)\b",
}

# Defaults per intent: (history_days, forecast, comparison_years, knowledge)
INTENT_DEFAULTS = {
    "current":    (0,    True,  0, False),
    "forecast":   (0,    True,  0, False),
    "trend":      (1095, False, 0, True),    # 3 years of daily observations
    "comparison": (30,   False, 5, True),
    "anomaly":    (30,   False, 0, True),
    "general":    (0,    False, 0, True),
}


class PlannerAgent(Agent):
    name = "planner"
    description = "Turns the question into a structured plan"

    def plan_raw(self, question: str, use_llm: bool = True) -> QueryPlan:
        """The planner's own output, before defaults are filled in.

        Exists for evaluation. `execute` applies `_apply_defaults`, which
        substitutes a fallback location when the question names none - so
        measuring the LLM through `execute` while measuring the rule-based
        planner through `_rule_based` scored one after defaults and the
        other before, and marked the LLM wrong on every definitional
        question for a substitution it never made.
        """
        if use_llm:
            llm = get_llm()
            if not llm.available:
                raise LLMError("LLM is not configured")
            return llm.parse(
                system=PLANNER_SYSTEM,
                user=build_planner_prompt(question, date.today().isoformat()),
                schema=QueryPlan,
            )
        return self._rule_based(question)

    def execute(self, context: dict) -> QueryPlan:
        question = context["question"]
        # Whatever place the user has selected in the UI. Used only when the
        # question itself names none - "will it rain this week?" should mean
        # where they are, not a hardcoded city.
        fallback = context.get("default_location") or "Mumbai"
        llm = get_llm()

        if llm.available:
            try:
                plan = llm.parse(
                    system=PLANNER_SYSTEM,
                    user=build_planner_prompt(question, date.today().isoformat()),
                    schema=QueryPlan,
                )
                self.log.info("LLM plan: intent=%s location=%r",
                              plan.intent, plan.location)
                return self._apply_defaults(plan, fallback)
            except LLMError as exc:
                # A planning failure should not end the request - the
                # rule-based planner is a genuine fallback, not a stub.
                self.log.warning("LLM planning failed (%s) - using rules", exc)

        return self._apply_defaults(self._rule_based(question), fallback)

    # ------------------------------------------------------------------
    def _rule_based(self, question: str) -> QueryPlan:
        q = question.lower()

        intent = "current"
        for candidate, pattern in INTENT_PATTERNS:
            if re.search(pattern, q):
                intent = candidate
                break

        variables = [v for v, pat in VARIABLE_PATTERNS.items() if re.search(pat, q)]

        years = 0
        if intent == "comparison":
            match = re.search(r"(\d+)\s*(?:previous\s+|past\s+|last\s+)?years?", q)
            years = int(match.group(1)) if match else 5

        return QueryPlan(
            intent=intent,
            location=self._extract_location(question),
            start_date="",
            end_date="",
            variables=variables,
            comparison_years=years,
            needs_history=intent in {"trend", "comparison", "anomaly"},
            needs_forecast=intent in {"current", "forecast"},
            needs_knowledge=True,
            rationale=f"Rule-based match on intent '{intent}'.",
        )

    @staticmethod
    def _extract_location(question: str) -> str:
        """Pull a place name out of the question.

        Three passes, most confident first. The last one accepts lowercase
        input, because people type "rain in nagaur" far more often than they
        capitalise. Anything it returns is only a *candidate* - the Data
        agent geocodes it, and falls back to the user's selected location if
        no such place exists.
        """
        lowered = question.lower()

        # 1. A place we already know - the most reliable signal offline.
        for key in sorted(_FALLBACK, key=len, reverse=True):
            if re.search(rf"\b{re.escape(key)}\b", lowered):
                return _FALLBACK[key].name

        # 2. A capitalised noun after a locative preposition.
        capitalised = re.search(
            r"\b(?:in|at|for|around|near|over)\s+"
            r"([A-Z][a-zA-Z]+(?:[\s-][A-Z][a-zA-Z]+)?)",
            question,
        )
        if capitalised:
            candidate = capitalised.group(1)
            if candidate.lower() not in NOT_A_PLACE:
                return candidate

        # 3. Same shape, ignoring case. Noisier, so the stopword filter does
        #    the work and geocoding makes the final call.
        for match in LOCATIVE.finditer(question):
            words = match.group(1).strip().split()
            # The two-word form exists for "New Delhi", but it also swallows
            # the next ordinary word: "in nagaur this year" captured
            # "nagaur this", which geocodes to nothing. Trim filler off the
            # end before deciding.
            while words and words[-1].lower() in NOT_A_PLACE:
                words.pop()
            if not words or words[0].lower() in NOT_A_PLACE:
                continue
            candidate = " ".join(words)
            if len(candidate) < 3:
                continue
            return candidate.title()

        return ""

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_defaults(plan: QueryPlan, fallback_location: str = "Mumbai") -> QueryPlan:
        """Fill in what the planner left blank, based on the chosen intent."""
        history_days, forecast, years, knowledge = INTENT_DEFAULTS.get(
            plan.intent, (0, True, 0, True)
        )

        if not plan.variables:
            plan.variables = (
                ["temperature", "rainfall"] if plan.intent != "current"
                else ["temperature", "humidity", "rainfall", "wind_speed"]
            )

        if plan.needs_history and not plan.start_date:
            days = history_days or 30
            plan.end_date = plan.end_date or date.today().isoformat()
            plan.start_date = (date.today() - timedelta(days=days)).isoformat()

        if plan.intent == "comparison" and plan.comparison_years == 0:
            plan.comparison_years = years
        # Cap it: each extra year is another archive request.
        plan.comparison_years = min(plan.comparison_years, settings.baseline_years)

        if not plan.location:
            plan.location = fallback_location
            plan.rationale += (
                f" No location in the question; used {fallback_location}."
            )

        plan.needs_forecast = plan.needs_forecast or forecast
        plan.needs_knowledge = plan.needs_knowledge or knowledge
        return plan
