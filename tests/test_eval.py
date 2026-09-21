"""Tests for the evaluation harness.

The metrics decide whether the project "works", so they are tested against
hand-worked examples. A metric bug silently invalidates every number in the
report.
"""
from __future__ import annotations

import pytest

from eval import metrics
from eval.dataset import CASES, retrieval_cases, stats


# ---------------------------------------------------------------- dataset
def test_dataset_is_well_formed():
    valid_intents = {"current", "forecast", "trend", "comparison",
                     "anomaly", "general"}
    valid_variables = {"temperature", "humidity", "rainfall",
                       "wind_speed", "pressure"}
    for case in CASES:
        assert case.intent in valid_intents, case.question
        assert set(case.variables) <= valid_variables, case.question
        assert case.question.strip()


def test_every_gold_source_file_exists():
    """A gold label pointing at a deleted file would silently score as a miss."""
    from config.settings import settings

    on_disk = {p.name for p in settings.kb_dir.glob("*.md")}
    for case in retrieval_cases():
        for source in case.gold_sources:
            assert source in on_disk, f"{source} missing for {case.question!r}"


def test_no_duplicate_questions():
    questions = [c.question for c in CASES]
    assert len(questions) == len(set(questions))


def test_dataset_covers_every_intent():
    assert set(stats()["by_intent"]) == {
        "current", "forecast", "trend", "comparison", "anomaly", "general"
    }
    # No intent should be so rare that its accuracy is meaningless.
    assert min(stats()["by_intent"].values()) >= 5


# ---------------------------------------------------------------- classification
def test_accuracy():
    assert metrics.accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert metrics.accuracy(["a", "x", "c"], ["a", "b", "c"]) == pytest.approx(2 / 3)
    assert metrics.accuracy([], []) == 0.0


def test_set_f1():
    assert metrics.set_f1(set(), set()) == 1.0          # both empty is correct
    assert metrics.set_f1({"a"}, set()) == 0.0
    assert metrics.set_f1({"a", "b"}, {"a", "b"}) == 1.0
    # predicted {a,b} vs expected {a}: precision 0.5, recall 1.0 -> F1 0.667
    assert metrics.set_f1({"a", "b"}, {"a"}) == pytest.approx(2 / 3)
    assert metrics.set_f1({"x"}, {"a"}) == 0.0


def test_confusion_records_the_direction_of_failure():
    matrix = metrics.confusion(["current", "current"], ["general", "current"])
    assert matrix["general"]["current"] == 1
    assert matrix["current"]["current"] == 1


# ---------------------------------------------------------------- retrieval
def test_recall_at_k():
    retrieved = ["a.md", "b.md", "c.md", "d.md"]
    assert metrics.recall_at_k(retrieved, {"c.md"}, 4) == 1.0
    assert metrics.recall_at_k(retrieved, {"c.md"}, 1) == 0.0
    assert metrics.recall_at_k(retrieved, {"a.md"}, 1) == 1.0
    assert metrics.recall_at_k(retrieved, {"z.md"}, 4) == 0.0


def test_precision_at_1():
    assert metrics.precision_at_1(["a.md", "b.md"], {"a.md"}) == 1.0
    assert metrics.precision_at_1(["b.md", "a.md"], {"a.md"}) == 0.0
    assert metrics.precision_at_1([], {"a.md"}) == 0.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert metrics.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert metrics.reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


# ---------------------------------------------------------------- groundedness
def test_extract_numbers_ignores_digits_inside_tokens():
    found = metrics.extract_numbers("temperature_2m was 31.5 C and MD5 ran")
    assert 31.5 in found
    assert 2.0 not in found          # the 2 in temperature_2m is not a claim
    assert 5.0 not in found          # nor the 5 in MD5


def test_extract_numbers_handles_negatives():
    assert -3.4 in metrics.extract_numbers("a departure of -3.4 C")


def test_extract_numbers_does_not_split_iso_dates():
    """2026-08-21 must not read as the claims 8 and 21."""
    found = metrics.extract_numbers("Across 2026-08-21 to 2026-09-15, 31.5 C")
    assert found == {31.5}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Rain arrives Thursday, 23 Sept, with 4 mm expected.", {4.0}),
        ("The heaviest falls on 21 September and 22 September.", set()),
        ("Expect showers Sept 23 and Sep 24.", set()),
        ("Wettest on 3rd August at 12.5 mm.", {12.5}),
        ("On 15 Jan it hit 8.2 C.", {8.2}),
    ],
)
def test_extract_numbers_ignores_written_dates(text, expected):
    """A day-of-month is not a measurement.

    Once a language model writes the prose it uses '23 Sept', not ISO form.
    Counting those as claims produced six false hallucination reports in an
    evaluation run.
    """
    assert metrics.extract_numbers(text) == expected


def test_groundedness_is_not_tripped_by_dates_in_the_answer():
    known = metrics.collect_known_numbers({"mean": 31.5})
    score = metrics.groundedness(
        "Between 2026-08-21 and 2026-09-15 the mean was 31.5 C.", known
    )
    assert score["score"] == 1.0
    assert score["ungrounded"] == []


def test_collect_known_numbers_walks_nested_structures():
    payload = {"summary": {"mean": 28.26, "values": [1.5, {"max": 35.9}]}}
    known = metrics.collect_known_numbers(payload)
    assert 28.26 in known
    assert 1.5 in known
    assert 35.9 in known


def test_groundedness_accepts_numbers_present_in_the_data():
    known = metrics.collect_known_numbers({"mean": 28.26, "total": 55.1})
    score = metrics.groundedness("The mean was 28.26 C and 55.1 mm fell.", known)
    assert score["score"] == 1.0
    assert score["ungrounded"] == []


def test_groundedness_catches_a_fabricated_number():
    """The check that would catch a hallucinated statistic."""
    known = metrics.collect_known_numbers({"total": 55.1})
    score = metrics.groundedness("Rainfall reached 240.7 mm this month.", known)
    assert score["score"] == 0.0
    assert 240.7 in score["ungrounded"]


def test_groundedness_ignores_years_and_published_thresholds():
    score = metrics.groundedness(
        "In 2024 the 64.5 mm heavy-rain threshold was crossed.", known=set()
    )
    # 2024 is a date and 64.5 is an IMD constant - neither is a data claim.
    assert score["score"] == 1.0
    assert score["claimed"] == 0


def test_groundedness_of_a_text_with_no_numbers_is_perfect():
    assert metrics.groundedness("Nothing unusual was detected.", set())["score"] == 1.0


def test_groundedness_tolerates_display_rounding():
    """The insight layer rounds for readability; that is not fabrication."""
    known = metrics.collect_known_numbers({"mean": 28.264})
    assert metrics.groundedness("about 28.3 C", known)["score"] == 1.0
