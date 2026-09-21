"""Labelled evaluation set.

34 questions with hand-written ground truth. This is the measuring
instrument for the whole project, so a few rules were followed when writing
it:

* **Labels came before the code was tuned.** Writing the expected intent
  after watching what the planner does measures nothing.
* **Paraphrases are included on purpose.** "Is it going to pour tomorrow?"
  and "What is the rainfall forecast?" are the same intent in different
  clothes; a keyword matcher will get one and miss the other, and that gap is
  exactly what the LLM planner is supposed to close.
* **`gold_sources` is only filled in where the answer genuinely depends on
  the knowledge base.** Labelling a relevant document for "what is the
  temperature in Pune" would be inventing a ground truth that does not exist.

`variables` uses set semantics - order does not matter, and partial credit is
scored with F1 rather than exact match.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    question: str
    intent: str
    location: str = ""
    variables: tuple = ()
    # Knowledge-base files that a good retriever should surface. Empty means
    # "retrieval is not scored for this question".
    gold_sources: tuple = ()
    # Paraphrase cases stress the planner rather than plain keyword matching.
    paraphrase: bool = False
    # "dev" cases were used to debug the rule-based patterns, so any score on
    # them is fitted and reads high. "test" cases were written afterwards and
    # nothing was tuned against them - that split carries the honest number.
    split: str = "dev"


VARIABLES_FILE = "01_weather_variables.md"
CLIMATOLOGY_FILE = "02_climatology_and_anomalies.md"
MONSOON_FILE = "03_indian_monsoon.md"
EXTREMES_FILE = "04_extreme_weather_criteria.md"
TRENDS_FILE = "05_interpreting_trends.md"


DEV_CASES: tuple = (
    # ---------------- current (6) ----------------
    Case("What is the weather in Mumbai right now?", "current", "Mumbai",
         ("temperature",)),
    Case("weather in Pune right now", "current", "Pune"),
    Case("How is the weather in Chennai currently?", "current", "Chennai"),
    Case("What is the temperature in Delhi at the moment?", "current", "Delhi",
         ("temperature",)),
    Case("Tell me the current humidity in Kolkata", "current", "Kolkata",
         ("humidity",)),
    Case("How muggy is it in Chennai today?", "current", "Chennai",
         ("humidity",), (VARIABLES_FILE,), paraphrase=True),

    # ---------------- forecast (5) ----------------
    Case("Will rainfall increase this week in Pune?", "forecast", "Pune",
         ("rainfall",)),
    Case("Is it going to rain in Chennai tomorrow?", "forecast", "Chennai",
         ("rainfall",)),
    Case("What is the temperature forecast for Delhi?", "forecast", "Delhi",
         ("temperature",)),
    Case("Should I expect showers in Mumbai over the next few days?",
         "forecast", "Mumbai", ("rainfall",), paraphrase=True),
    Case("How windy will it get in Bengaluru this weekend?", "forecast",
         "Bengaluru", ("wind_speed",), paraphrase=True),

    # ---------------- trend (5) ----------------
    Case("What is the weather trend in Mumbai?", "trend", "Mumbai"),
    Case("How has the temperature in Delhi been changing?", "trend", "Delhi",
         ("temperature",), (TRENDS_FILE,)),
    Case("Show me the long-term rainfall pattern in Pune", "trend", "Pune",
         ("rainfall",), (TRENDS_FILE,)),
    Case("Has Kolkata been getting hotter over the years?", "trend", "Kolkata",
         ("temperature",), (TRENDS_FILE,), paraphrase=True),
    Case("What does the historical wind data for Chennai look like?", "trend",
         "Chennai", ("wind_speed",)),

    # ---------------- comparison (5) ----------------
    Case("How has the temperature in Delhi changed compared with previous years?",
         "comparison", "Delhi", ("temperature",), (CLIMATOLOGY_FILE,)),
    Case("How has rainfall in Pune compared with previous years?", "comparison",
         "Pune", ("rainfall",), (CLIMATOLOGY_FILE,)),
    Case("Is this month wetter in Mumbai than last year?", "comparison",
         "Mumbai", ("rainfall",), (CLIMATOLOGY_FILE,), paraphrase=True),
    Case("Compare the current temperature in Jaipur with past years",
         "comparison", "Jaipur", ("temperature",)),
    Case("How does this season in Chennai stack up against earlier years?",
         "comparison", "Chennai", (), (CLIMATOLOGY_FILE,), paraphrase=True),

    # ---------------- anomaly (6) ----------------
    Case("Why is the current weather in Chennai unusual?", "anomaly", "Chennai",
         (), (CLIMATOLOGY_FILE,)),
    Case("Is anything abnormal about the weather in Mumbai?", "anomaly",
         "Mumbai", (), (CLIMATOLOGY_FILE,)),
    Case("Why is it so unusually hot in Delhi?", "anomaly", "Delhi",
         ("temperature",), (EXTREMES_FILE, CLIMATOLOGY_FILE)),
    Case("Has there been any extreme rainfall in Mumbai recently?", "anomaly",
         "Mumbai", ("rainfall",), (EXTREMES_FILE,)),
    Case("Is the weather in Nagpur behaving strangely?", "anomaly", "Nagpur",
         (), (CLIMATOLOGY_FILE,), paraphrase=True),
    Case("Did Pune record anything out of the ordinary this month?", "anomaly",
         "Pune", (), (CLIMATOLOGY_FILE,), paraphrase=True),

    # ---------------- general / knowledge (7) ----------------
    Case("What is relative humidity?", "general", "", ("humidity",),
         (VARIABLES_FILE,)),
    Case("What counts as a heat wave in India?", "general", "",
         ("temperature",), (EXTREMES_FILE,)),
    Case("How much rain is officially called very heavy rainfall?", "general",
         "", ("rainfall",), (EXTREMES_FILE,)),
    Case("What is a climate normal?", "general", "", (), (CLIMATOLOGY_FILE,)),
    Case("Explain how the southwest monsoon works", "general", "",
         ("rainfall",), (MONSOON_FILE,)),
    Case("How many years of data do I need to claim a warming trend?",
         "general", "", (), (TRENDS_FILE,)),
    Case("What does falling atmospheric pressure mean?", "general", "",
         ("pressure",), (VARIABLES_FILE,)),
)


# --------------------------------------------------------------------------
# Held-out split
# --------------------------------------------------------------------------
# Written *after* the rule-based patterns were finalised, using phrasing
# deliberately absent from the keyword lists ("outlook", "at present", "out of
# line with the norm"). Nothing was tuned against these, so this is the split
# that says whether the planner generalises or has simply memorised its own
# regexes. Expect the rule-based planner to score materially lower here - that
# gap is the argument for the LLM planner, and it should be reported, not
# hidden.
TEST_CASES: tuple = (
    Case("Give me the outlook for Jaipur over the next ten days", "forecast",
         "Jaipur", split="test"),
    Case("Has Mumbai become rainier across the last decade?", "trend", "Mumbai",
         ("rainfall",), (TRENDS_FILE,), split="test"),
    Case("Is today's heat in Nagpur out of line with the norm?", "anomaly",
         "Nagpur", ("temperature",), (CLIMATOLOGY_FILE,), split="test"),
    Case("Set that against what Chennai usually sees in October", "comparison",
         "Chennai", (), (CLIMATOLOGY_FILE,), split="test"),
    Case("Right this minute, how warm is Pune?", "current", "Pune",
         ("temperature",), split="test"),
    Case("Put simply, what makes air feel sticky?", "general", "",
         ("humidity",), (VARIABLES_FILE,), split="test"),
    Case("What's the rain situation in Kolkata at present?", "current",
         "Kolkata", ("rainfall",), split="test"),
    Case("Does Delhi see more downpours now than it used to?", "trend", "Delhi",
         ("rainfall",), (TRENDS_FILE,), split="test"),
    Case("How do meteorologists decide something is a cold wave?", "general",
         "", ("temperature",), (EXTREMES_FILE,), split="test"),
    Case("Anything freakish in the Pune numbers lately?", "anomaly", "Pune",
         (), (CLIMATOLOGY_FILE,), split="test"),
    Case("Roughly what will Bengaluru look like on Friday?", "forecast",
         "Bengaluru", (), split="test"),
    Case("Line up this year's Mumbai monsoon beside the usual", "comparison",
         "Mumbai", ("rainfall",), (MONSOON_FILE,), split="test"),
)


CASES: tuple = DEV_CASES + TEST_CASES


def cases_in(split: str = None) -> tuple:
    if split is None:
        return CASES
    return tuple(c for c in CASES if c.split == split)


def retrieval_cases(split: str = None) -> tuple:
    """Only the cases where a gold document actually exists."""
    return tuple(c for c in cases_in(split) if c.gold_sources)


def stats() -> dict:
    intents: dict = {}
    for case in CASES:
        intents[case.intent] = intents.get(case.intent, 0) + 1
    return {
        "total": len(CASES),
        "dev": len(DEV_CASES),
        "test": len(TEST_CASES),
        "by_intent": dict(sorted(intents.items())),
        "with_gold_sources": len(retrieval_cases()),
        "paraphrases": sum(1 for c in CASES if c.paraphrase),
        "with_location": sum(1 for c in CASES if c.location),
    }
