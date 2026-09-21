"""Labelled evaluation set.

70 questions with hand-written ground truth. This is the measuring
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
    # "dev" cases are fair game for tuning, so any score on them is fitted
    # and reads high. "test" cases were written before the code they measure
    # and nothing has been tuned against them - that split carries the honest
    # number. A test case that gets used to debug a pattern must be moved to
    # dev in the same commit; see the retired-holdout block below.
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

    # ---------------- retired held-out split (12) ----------------
    # These were the v1 held-out set. On 2026-09-21 the rule-based intent
    # patterns were broadened using their failures, which spends their value
    # as a generalisation measurement - a case you tune against is a dev
    # case, whatever it used to be called. They stay in the set because they
    # are good questions and they guard the regression; they are just no
    # longer evidence that the planner generalises. TEST_CASES below was
    # written to replace them, before any pattern was touched.
    Case("Give me the outlook for Jaipur over the next ten days", "forecast",
         "Jaipur"),
    Case("Has Mumbai become rainier across the last decade?", "trend", "Mumbai",
         ("rainfall",), (TRENDS_FILE,), paraphrase=True),
    Case("Is today's heat in Nagpur out of line with the norm?", "anomaly",
         "Nagpur", ("temperature",), (CLIMATOLOGY_FILE,), paraphrase=True),
    Case("Set that against what Chennai usually sees in October", "comparison",
         "Chennai", (), (CLIMATOLOGY_FILE,), paraphrase=True),
    Case("Right this minute, how warm is Pune?", "current", "Pune",
         ("temperature",), paraphrase=True),
    Case("Put simply, what makes air feel sticky?", "general", "",
         ("humidity",), (VARIABLES_FILE,), paraphrase=True),
    Case("What's the rain situation in Kolkata at present?", "current",
         "Kolkata", ("rainfall",), paraphrase=True),
    Case("Does Delhi see more downpours now than it used to?", "trend", "Delhi",
         ("rainfall",), (TRENDS_FILE,), paraphrase=True),
    Case("How do meteorologists decide something is a cold wave?", "general",
         "", ("temperature",), (EXTREMES_FILE,), paraphrase=True),
    Case("Anything freakish in the Pune numbers lately?", "anomaly", "Pune",
         (), (CLIMATOLOGY_FILE,), paraphrase=True),
    Case("Roughly what will Bengaluru look like on Friday?", "forecast",
         "Bengaluru", (), paraphrase=True),
    Case("Line up this year's Mumbai monsoon beside the usual", "comparison",
         "Mumbai", ("rainfall",), (MONSOON_FILE,), paraphrase=True),
)


# --------------------------------------------------------------------------
# Held-out split (v2)
# --------------------------------------------------------------------------
# Written on 2026-09-21, *before* the intent patterns were broadened, and not
# consulted while broadening them. That ordering is the whole point: a split
# written after the fix, or edited once its failures were known, measures
# memorisation rather than generalisation.
#
# The v1 held-out set was spent doing exactly the fix this set now guards
# against, so it has been moved into DEV_CASES above rather than quietly
# left in place with a better-looking score.
#
# Phrasing is deliberately oblique ("losing its winters", "thinned out", "off
# the charts", "as things stand") and the cities lean on the less-used
# entries in the geocoder's offline table, so location extraction is stressed
# alongside intent. Expect the rule-based planner to score materially lower
# here than on dev; that gap is the argument for the LLM planner, and it
# should be reported, not hidden.
#
# Rule for maintainers: if you ever debug a pattern against one of these,
# move it to DEV_CASES in the same commit.
TEST_CASES: tuple = (
    # ---------------- current (4) ----------------
    Case("How's Hyderabad looking outside?", "current", "Hyderabad",
         paraphrase=True, split="test"),
    Case("Give me conditions in Ahmedabad as things stand", "current",
         "Ahmedabad", paraphrase=True, split="test"),
    Case("Temperature readout for Lucknow please", "current", "Lucknow",
         ("temperature",), split="test"),
    Case("Is it raining in Shimla this very moment?", "current", "Shimla",
         ("rainfall",), split="test"),

    # ---------------- forecast (4) ----------------
    Case("What should Hyderabad brace for over the weekend?", "forecast",
         "Hyderabad", paraphrase=True, split="test"),
    Case("Any chance of a soaking in Kolkata before Sunday?", "forecast",
         "Kolkata", ("rainfall",), paraphrase=True, split="test"),
    Case("Tell me how Jaipur shapes up over the coming fortnight", "forecast",
         "Jaipur", paraphrase=True, split="test"),
    Case("What kind of week is Bengaluru in for, heat-wise?", "forecast",
         "Bengaluru", ("temperature",), paraphrase=True, split="test"),

    # ---------------- trend (4) ----------------
    Case("Is Shimla losing its winters?", "trend", "Shimla", ("temperature",),
         (TRENDS_FILE,), paraphrase=True, split="test"),
    Case("Have Ahmedabad's summers drifted since the nineties?", "trend",
         "Ahmedabad", ("temperature",), (TRENDS_FILE,), paraphrase=True,
         split="test"),
    Case("Has the rain in Cherrapunji thinned out over time?", "trend",
         "Cherrapunji", ("rainfall",), (TRENDS_FILE,), paraphrase=True,
         split="test"),
    Case("What direction has Lucknow's climate been moving?", "trend",
         "Lucknow", (), (TRENDS_FILE,), paraphrase=True, split="test"),

    # ---------------- comparison (4) ----------------
    Case("How does Hyderabad this month measure against its usual September?",
         "comparison", "Hyderabad", (), (CLIMATOLOGY_FILE,), paraphrase=True,
         split="test"),
    Case("Put Nagpur's current spell next to what it saw twelve months ago",
         "comparison", "Nagpur", (), (CLIMATOLOGY_FILE,), paraphrase=True,
         split="test"),
    Case("Is Kolkata wetter or drier than it was three years back?",
         "comparison", "Kolkata", ("rainfall",), (CLIMATOLOGY_FILE,),
         paraphrase=True, split="test"),
    Case("Weigh this week's Jaipur heat against a typical year", "comparison",
         "Jaipur", ("temperature",), (CLIMATOLOGY_FILE,), paraphrase=True,
         split="test"),

    # ---------------- anomaly (4) ----------------
    Case("Does anything in the Ahmedabad data look off?", "anomaly",
         "Ahmedabad", (), (CLIMATOLOGY_FILE,), paraphrase=True, split="test"),
    Case("Has Shimla thrown up any surprises this week?", "anomaly", "Shimla",
         (), (CLIMATOLOGY_FILE,), paraphrase=True, split="test"),
    Case("Is Cherrapunji's rainfall off the charts right now?", "anomaly",
         "Cherrapunji", ("rainfall",), (EXTREMES_FILE,), paraphrase=True,
         split="test"),
    Case("Something looks wrong with Lucknow's temperatures lately", "anomaly",
         "Lucknow", ("temperature",), (CLIMATOLOGY_FILE,), paraphrase=True,
         split="test"),

    # ---------------- general / knowledge (4) ----------------
    Case("At what point does a storm get classified as a cyclone?", "general",
         "", ("wind_speed",), (EXTREMES_FILE,), split="test"),
    Case("Talk me through why the rains arrive when they do", "general", "",
         ("rainfall",), (MONSOON_FILE,), paraphrase=True, split="test"),
    Case("Is one scorching summer enough to prove anything?", "general", "",
         ("temperature",), (TRENDS_FILE,), paraphrase=True, split="test"),
    Case("Where's the line between weather and climate?", "general", "", (),
         (CLIMATOLOGY_FILE,), paraphrase=True, split="test"),
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
