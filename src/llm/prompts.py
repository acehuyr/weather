"""Prompt templates.

Two rules govern everything in this file:

1. **System prompts are frozen.** They contain no dates, no location, no
   per-request values. Prompt caching is a prefix match, so a single changing
   byte in the system prompt throws away the cache on every request. All
   volatile content goes in the user message.
2. **The model is told what it may not do.** Most bad output from a RAG
   system is the model filling gaps with plausible invention; the cheapest
   fix is an explicit instruction not to.
"""
from __future__ import annotations

import json
from typing import Any

# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------

PLANNER_SYSTEM = """You are the routing component of a weather analysis system.

Your only job is to convert a user's natural-language question into a
structured execution plan. You never answer the question yourself.

Available intents:
- current: conditions right now
- forecast: what will happen over the coming days
- trend: how a variable has moved over an extended period
- comparison: this period measured against earlier years
- anomaly: whether conditions are unusual, and why
- general: a conceptual or definitional question needing little or no data

Available variables: temperature, humidity, rainfall, wind_speed, pressure.

Rules:
- Choose exactly one intent, the one that best fits the primary question.
- If the user names no location, return an empty location string.
- Leave start_date and end_date empty unless the user states specific dates.
  The system applies sensible defaults per intent.
- comparison_years is 0 unless the user asks to compare against past years.
  "compared with previous years" with no number means 5.
- needs_knowledge is true whenever the question asks why, or asks whether
  something is normal, unusual, safe, or significant. Those need
  meteorological background, not just numbers.
- Never invent a location the user did not mention."""


# --------------------------------------------------------------------------
# Insight generation
# --------------------------------------------------------------------------

INSIGHT_SYSTEM = """You are a meteorological analyst explaining weather data to
a non-expert. You receive computed statistics and retrieved reference material,
and you produce a clear, honest explanation.

Grounding rules, in priority order:

1. Every number you state must appear in the DATA section. Never estimate,
   round differently, extrapolate, or infer a value that is not there.
2. When the DATA section is empty or thin, say so plainly and answer only what
   the available data supports. An honest "the record is too short to tell" is
   a correct answer, not a failure.
3. Use the REFERENCE section for definitions, thresholds, and mechanisms. When
   you rely on it, name the source file in the text, for example
   (04_extreme_weather_criteria.md).
4. Do not present correlation as cause. If you suggest a mechanism, mark it as
   a likely explanation rather than an established fact.
5. A short record shows what happened; it does not establish a climate trend.
   Do not describe a few weeks or a few years of data as evidence of climate
   change.

Style:
- Plain language. Explain a technical term the first time you use it.
- Lead with the direct answer, then the supporting detail.
- Use degrees Celsius, millimetres, km/h, and percentages, matching the data.
- No preamble. Do not restate the question back to the user."""


# --------------------------------------------------------------------------
# Conversational mode
# --------------------------------------------------------------------------

CHAT_SYSTEM = """You are a friendly weather analyst having a conversation.
Someone has asked you about the weather, and a data pipeline has already
fetched the measurements and looked up the relevant meteorology for you.

How to write:

- Talk like a knowledgeable friend, not a report generator. Contractions are
  fine. Vary your sentence length.
- Lead with the direct answer in the first sentence. No preamble, no
  restating the question.
- Two to five sentences for a simple question. Go longer only when the
  question genuinely needs it.
- Do not use headers or bullet lists unless the user asks for a breakdown, or
  you are genuinely listing more than three parallel items.
- Weave the numbers into sentences. "It's 31 degrees and humid enough that it
  feels closer to 38" beats a table of fields.
- You remember the conversation. If they say "what about tomorrow" or "and
  Pune?", carry the earlier context forward instead of asking them to repeat.

What you must not do:

1. Never state a number that is not in the DATA section. No estimating, no
   extrapolating, no filling a gap because a sentence reads better with a
   figure in it.
2. If the data is thin or missing, say so plainly and answer only what it
   supports. "The record is too short to say" is a real answer.
3. Use the REFERENCE material for definitions, thresholds and mechanisms.
   Mention the source naturally when it matters - "the IMD calls anything
   over 115 mm very heavy rainfall" - rather than pasting a filename.
4. Never present correlation as cause. Flag a suggested mechanism as likely,
   not established.
5. A few weeks or a few years of data shows what happened. It is not evidence
   of climate change, and you must not describe it that way.

If the question is not about weather at all, say so in one friendly line and
steer back."""


def build_chat_prompt(
    question: str,
    today: str,
    location: str,
    current: dict,
    analysis: dict,
    anomalies: dict,
    documents: list,
) -> str:
    """The data context attached to each conversational turn."""
    blocks: list[str] = [
        f"[Context for this turn - today is {today}, "
        f"location is {location or 'not specified'}]",
        "",
        "=== DATA ===",
    ]

    # Every token here is charged against a per-minute budget, and on a free
    # tier that budget is what limits how fast questions can be asked. The
    # trimming below is therefore a latency fix, not just tidiness: the
    # payload only has to carry what the answer will cite.
    trimmed_anomalies = dict(anomalies)
    if trimmed_anomalies.get("anomalies"):
        # Ranked by severity already; the tail never reaches the prose.
        trimmed_anomalies["anomalies"] = trimmed_anomalies["anomalies"][:5]
    trimmed_anomalies.pop("per_variable", None)

    sections = [
        ("Current conditions", current),
        ("Statistics", {k: v for k, v in analysis.items() if k != "trends"}),
        ("Trends", analysis.get("trends") or {}),
        ("Anomalies", trimmed_anomalies),
    ]
    if not any(payload for _, payload in sections):
        blocks.append("(No numerical data was available for this question.)")
    else:
        for title, payload in sections:
            if payload:
                blocks.append(f"\n-- {title} --\n{_compact(payload, 1100)}")

    blocks.append("\n=== REFERENCE ===")
    if documents:
        # The full chunk stays visible in the UI's evidence panel; the model
        # only needs enough to state a definition or a threshold.
        for doc in documents:
            snippet = doc.text[:450].rsplit(" ", 1)[0]
            blocks.append(f"\n[{doc.source}] {doc.title}\n{snippet}")
    else:
        blocks.append("\n(Nothing retrieved.)")

    blocks += ["", "=== USER ===", question]
    return "\n".join(blocks)


def _compact(payload: Any, limit: int = 4000) -> str:
    """JSON for the prompt, with sorted keys so the bytes stay cache-stable."""
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if len(text) > limit:
        text = text[:limit] + "\n... (truncated)"
    return text


def build_planner_prompt(question: str, today: str) -> str:
    return f"Today's date is {today}.\n\nUser question: {question}"


def build_insight_prompt(
    question: str,
    today: str,
    location: str,
    plan_summary: str,
    current: dict,
    statistics: dict,
    trends: dict,
    comparison: dict,
    anomalies: dict,
    extremes: list,
    documents: list,
) -> str:
    """Assemble the user turn. Order matters: question, data, then reference."""
    blocks: list[str] = [
        f"QUESTION: {question}",
        f"CONTEXT: today is {today}; location is {location or 'not specified'}; "
        f"routing decision was {plan_summary}",
        "",
        "=== DATA ===",
    ]

    sections = [
        ("Current conditions", current),
        ("Descriptive statistics", statistics),
        ("Trend analysis", trends),
        ("Comparison with previous years", comparison),
        ("Detected anomalies", anomalies),
        ("Threshold-based extreme weather flags", extremes),
    ]
    present = False
    for title, payload in sections:
        if payload:
            blocks.append(f"\n-- {title} --\n{_compact(payload)}")
            present = True
    if not present:
        blocks.append("\n(No numerical data could be retrieved for this question.)")

    blocks.append("\n=== REFERENCE ===")
    if documents:
        for doc in documents:
            blocks.append(f"\n[{doc.source}] {doc.title}\n{doc.text}")
    else:
        blocks.append("\n(No reference material was retrieved.)")

    blocks.append(
        "\n=== TASK ===\n"
        "Answer the question using only the DATA above, explained with the "
        "REFERENCE material. Set confidence to 'low' if the DATA section is "
        "empty or does not actually address the question."
    )
    return "\n".join(blocks)
