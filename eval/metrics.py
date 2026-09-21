"""Metric implementations.

Kept separate from the runner so each one can be unit-tested against hand-
worked examples. A metric you have not tested is a number you cannot defend.
"""
from __future__ import annotations

import re
from typing import Any


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def accuracy(predictions: list, truths: list) -> float:
    if not predictions:
        return 0.0
    hits = sum(1 for p, t in zip(predictions, truths) if p == t)
    return hits / len(predictions)


def set_f1(predicted: set, expected: set) -> float:
    """F1 over two sets. Both empty is a perfect score, not a divide-by-zero."""
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def confusion(predictions: list, truths: list) -> dict:
    """{true_intent: {predicted_intent: count}} - shows *how* it fails."""
    matrix: dict = {}
    for pred, truth in zip(predictions, truths):
        matrix.setdefault(truth, {})
        matrix[truth][pred] = matrix[truth].get(pred, 0) + 1
    return matrix


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def recall_at_k(retrieved: list, gold: set, k: int) -> float:
    """1.0 if any gold document appears in the top k, else 0.0."""
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def precision_at_1(retrieved: list, gold: set) -> float:
    return 1.0 if retrieved and retrieved[0] in gold else 0.0


def reciprocal_rank(retrieved: list, gold: set) -> float:
    """1/rank of the first gold hit; 0 if none was retrieved."""
    for position, source in enumerate(retrieved, start=1):
        if source in gold:
            return 1.0 / position
    return 0.0


def mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------
# Groundedness
# --------------------------------------------------------------------------

# Matches 12, 12.5, -3.4 but not the digits inside a token like 2m or MD5.
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")

# Dates must be removed before numbers are extracted, or a day-of-month gets
# counted as a measurement the answer never claimed. Two forms appear:
# "2026-08-21" from the template, and "23 Sept" / "Sept 23" once a language
# model writes the prose. Missing the second form cost six false violations
# in an evaluation run and looked exactly like hallucination.
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

_MONTHS = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
           r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|"
           r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
WORD_DATE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})"
    rf"|(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?)\b",
    re.IGNORECASE,
)

# Numbers that are part of the system's own vocabulary rather than claims
# about the data: IMD thresholds, the rainy-day cutoff, z-score bands.
ALLOWED_CONSTANTS = {
    0.0, 0.1, 1.0, 1.5, 2.0, 2.4, 2.5, 3.0, 4.5, 5.0, 6.4, 6.5, 7.0, 10.0,
    15.5, 15.6, 25.0, 30.0, 35.6, 37.0, 40.0, 45.0, 47.0, 64.4, 64.5,
    100.0, 115.5, 115.6, 204.4, 204.5,
}


def extract_numbers(text: str) -> set:
    cleaned = WORD_DATE.sub(" ", ISO_DATE.sub(" ", text or ""))
    return {round(float(m), 2) for m in NUMBER.findall(cleaned)}


def collect_known_numbers(payload: Any, seen: set = None) -> set:
    """Every number that actually appears anywhere in the computed data."""
    if seen is None:
        seen = set()
    if isinstance(payload, bool) or payload is None:
        return seen
    if isinstance(payload, (int, float)):
        value = float(payload)
        seen.add(round(value, 2))
        # Insight text rounds for readability, so accept both directions.
        seen.add(round(value, 1))
        seen.add(float(round(value)))
        seen.add(abs(round(value, 1)))
        return seen
    if isinstance(payload, str):
        seen |= extract_numbers(payload)
        return seen
    if isinstance(payload, dict):
        for key, item in payload.items():
            collect_known_numbers(key, seen)
            collect_known_numbers(item, seen)
        return seen
    if isinstance(payload, (list, tuple, set)):
        for item in payload:
            collect_known_numbers(item, seen)
        return seen
    return seen


def groundedness(answer_text: str, known: set) -> dict:
    """Fraction of numbers in the answer that trace back to computed data.

    This is the check that catches a hallucinated statistic: the model says
    "rainfall was 240 mm" when nothing in the pipeline produced 240.
    """
    claimed = extract_numbers(answer_text)
    # Year-like integers are dates, not measurements.
    claimed = {n for n in claimed if not (1900 <= n <= 2100 and n == int(n))}
    claimed -= ALLOWED_CONSTANTS

    if not claimed:
        return {"score": 1.0, "claimed": 0, "grounded": 0, "ungrounded": []}

    ungrounded = sorted(n for n in claimed if n not in known)
    grounded = len(claimed) - len(ungrounded)
    return {
        "score": grounded / len(claimed),
        "claimed": len(claimed),
        "grounded": grounded,
        "ungrounded": ungrounded,
    }
