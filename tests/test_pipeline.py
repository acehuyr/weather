"""Tests that run without network access or an API key.

Anything touching the weather provider is marked `network` and skipped by
default, so `pytest` stays fast and deterministic:

    pytest                  # offline tests only
    pytest -m network       # include live provider calls
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis import anomalies, statistics, trends
from src.agents.planner import PlannerAgent
from src.core.models import Insight, QueryPlan
from src.rag.chunker import split_sections
from src.rag.embeddings import HashingEmbedder


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def warming_series() -> pd.DataFrame:
    """200 days with a deliberate +0.02 C/day trend and small noise."""
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.default_rng(42)
    values = 20 + 0.02 * np.arange(200) + rng.normal(0, 0.5, 200)
    return pd.DataFrame({"temperature_2m_mean": values}, index=idx)


@pytest.fixture
def flat_series() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {"temperature_2m_mean": 25 + rng.normal(0, 2, 120)}, index=idx
    )


# ---------------------------------------------------------------- analysis
def test_linear_trend_detects_known_slope(warming_series):
    result = trends.linear_trend(warming_series["temperature_2m_mean"])
    assert result["status"] == "ok"
    assert result["direction"] == "increasing"
    # The true slope is 0.02 C/day; allow for the injected noise.
    assert result["slope_per_day"] == pytest.approx(0.02, abs=0.005)


def test_flat_series_is_not_reported_as_a_trend(flat_series):
    result = trends.linear_trend(flat_series["temperature_2m_mean"])
    assert result["direction"] == "stable"
    assert result["confidence"] == "low"


def test_trend_refuses_tiny_samples():
    s = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
    assert trends.linear_trend(s)["status"] == "insufficient_data"


def test_summarize_is_json_safe(warming_series):
    out = statistics.summarize(warming_series)
    assert out["rows"] == 200
    var = out["variables"]["temperature_2m_mean"]
    assert isinstance(var["mean"], float)
    assert isinstance(out["start"], str)


def test_jsonable_converts_nan_to_none():
    assert statistics.jsonable(float("nan")) is None
    assert statistics.jsonable(np.float64(3.14159)) == 3.14


def test_rainfall_profile_uses_the_rainy_day_threshold():
    idx = pd.date_range("2024-06-01", periods=5, freq="D")
    df = pd.DataFrame({"precipitation_sum": [0.0, 1.0, 3.0, 80.0, 0.5]}, index=idx)
    out = statistics.rainfall_profile(df)
    assert out["total_mm"] == 84.5
    assert out["wet_days"] == 2          # only 3.0 and 80.0 reach 2.5 mm
    assert out["heaviest_day_mm"] == 80.0


# ---------------------------------------------------------------- anomalies
def test_rainfall_categories_match_imd_bands():
    assert anomalies.categorise_rainfall(0.0) == "trace / no rain"
    assert anomalies.categorise_rainfall(10.0) == "light rainfall"
    assert anomalies.categorise_rainfall(30.0) == "moderate rainfall"
    assert anomalies.categorise_rainfall(70.0) == "heavy rainfall"
    assert anomalies.categorise_rainfall(120.0) == "very heavy rainfall"
    assert anomalies.categorise_rainfall(250.0) == "extremely heavy rainfall"
    assert anomalies.categorise_rainfall(None) == "unknown"


def test_climatological_detection_flags_an_injected_spike():
    # Five years of baseline around 30 C, then one recent day at 40 C.
    base_idx = pd.date_range("2019-06-01", "2024-06-30", freq="D")
    rng = np.random.default_rng(1)
    baseline = pd.DataFrame(
        {"temperature_2m_max": 30 + rng.normal(0, 1, len(base_idx))}, index=base_idx
    )
    recent = pd.DataFrame(
        {"temperature_2m_max": [30.0, 40.0, 30.0]},
        index=pd.date_range("2025-06-10", periods=3, freq="D"),
    )
    out = anomalies.detect(recent, baseline)
    assert out["status"] == "ok"
    assert out["method"] == "climatological_zscore"
    assert out["total"] >= 1
    top = out["anomalies"][0]
    assert top["value"] == 40.0
    assert top["direction"] == "above"
    assert top["severity"] == "extreme"


def test_daily_rainfall_is_skipped_without_a_baseline():
    """Zero-inflated rainfall must not be run through IQR outlier detection."""
    idx = pd.date_range("2024-06-01", periods=40, freq="D")
    values = [0.0] * 30 + [5.0, 12.0, 0.0, 0.0, 8.0, 0.0, 0.0, 30.0, 0.0, 0.0]
    recent = pd.DataFrame({"precipitation_sum": values}, index=idx)
    out = anomalies.detect(recent, baseline=None)
    assert out["per_variable"]["precipitation_sum"]["method"] == "skipped"
    assert out["total"] == 0


def test_flag_extremes_catches_heavy_rain_and_heat():
    idx = pd.date_range("2024-05-01", periods=2, freq="D")
    df = pd.DataFrame(
        {"precipitation_sum": [150.0, 1.0], "temperature_2m_max": [35.0, 42.0]},
        index=idx,
    )
    flags = anomalies.flag_extremes(df)
    kinds = {f["type"] for f in flags}
    assert kinds == {"rainfall", "heat"}


def test_detect_handles_empty_input():
    assert anomalies.detect(None)["status"] == "no_data"


# ---------------------------------------------------------------- comparison
def test_year_over_year_computes_percent_change():
    current = pd.DataFrame(
        {"precipitation_sum": [10.0] * 10},
        index=pd.date_range("2025-07-01", periods=10, freq="D"),
    )
    past_frames = []
    for year, daily in ((2023, 5.0), (2024, 5.0)):
        frame = pd.DataFrame(
            {"precipitation_sum": [daily] * 10},
            index=pd.date_range(f"{year}-07-01", periods=10, freq="D"),
        )
        frame["year"] = year
        past_frames.append(frame)
    past = pd.concat(past_frames)

    out = trends.year_over_year(current, past)
    block = out["precipitation_sum"]
    assert block["aggregate"] == "sum"       # rain totals, not averages
    assert block["current"] == 100.0
    assert block["historical_mean"] == 50.0
    assert block["percent_change"] == 100.0
    assert block["verdict"] == "above normal"


# ---------------------------------------------------------------- planner
@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is the weather trend in Mumbai?", "trend"),
        ("Will rainfall increase this week?", "forecast"),
        ("How does this compare with previous years?", "comparison"),
        ("Why is the weather so unusual today?", "anomaly"),
        ("What is relative humidity?", "general"),
        ("weather in Pune right now", "current"),
    ],
)
def test_rule_based_intent_routing(question, expected):
    plan = PlannerAgent()._rule_based(question)
    assert plan.intent == expected


def test_planner_extracts_a_known_location():
    plan = PlannerAgent()._rule_based("How hot is it in Bengaluru?")
    assert plan.location == "Bengaluru"


@pytest.mark.parametrize(
    "question,expected",
    [
        # Lowercase and unknown to the bundled table - people rarely
        # capitalise, and the geocoder validates afterwards.
        ("rain in nagaur this year?", "Nagaur"),
        ("forecast for bhilwara next week", "Bhilwara"),
        ("weather in new delhi tomorrow", "New Delhi"),
        ("temperature in kochi last year", "Kochi"),
        # Not places.
        ("rain in the morning?", ""),
        ("will it rain this week?", ""),
        ("what counts as a heat wave in India?", ""),
    ],
)
def test_location_extraction_handles_case_and_filler(question, expected):
    assert PlannerAgent()._extract_location(question) == expected


def test_trailing_filler_is_trimmed_from_a_place_name():
    """'in nagaur this year' must not extract 'Nagaur This'."""
    assert "this" not in PlannerAgent()._extract_location(
        "rain in nagaur this year?"
    ).lower()


def test_plan_raw_does_not_apply_defaults():
    """The evaluation path must see the planner's own output.

    `execute` substitutes a fallback location when the question names none.
    Measuring one planner through `execute` and the other through
    `_rule_based` compared post-defaults output against pre-defaults output,
    and marked the LLM wrong on every definitional question.
    """
    raw = PlannerAgent().plan_raw("What is relative humidity?", use_llm=False)
    assert raw.location == ""          # no fallback substituted
    assert raw.start_date == ""        # no date window filled in


def test_plan_raw_matches_rule_based_exactly():
    agent = PlannerAgent()
    question = "How has rainfall in Pune compared with previous years?"
    assert agent.plan_raw(question, use_llm=False) == agent._rule_based(question)


def test_fallback_location_is_used_when_none_is_named():
    plan = PlannerAgent._apply_defaults(
        PlannerAgent()._rule_based("will it rain this week?"),
        fallback_location="Pune",
    )
    assert plan.location == "Pune"
    assert "Pune" in plan.rationale


def test_this_year_is_a_trend_not_current_conditions():
    assert PlannerAgent()._rule_based("rain in nagaur this year?").intent == "trend"


def test_planner_defaults_fill_in_a_date_window():
    plan = PlannerAgent._apply_defaults(
        PlannerAgent()._rule_based("What is the temperature trend?")
    )
    assert plan.start_date and plan.end_date
    assert plan.location == "Mumbai"          # documented demo default
    span = date.fromisoformat(plan.end_date) - date.fromisoformat(plan.start_date)
    assert span > timedelta(days=365)


def test_comparison_years_are_capped():
    plan = PlannerAgent._apply_defaults(
        QueryPlan(
            intent="comparison", location="Delhi", start_date="", end_date="",
            variables=[], comparison_years=50, needs_history=True,
            needs_forecast=False, needs_knowledge=True, rationale="test",
        )
    )
    assert plan.comparison_years <= 10


# ---------------------------------------------------------------- insight
def test_findings_about_the_asked_variable_sort_first():
    """A rainfall question must not be answered with temperature stats."""
    from src.agents.insight_agent import InsightAgent

    findings = [
        "The daily maximum temperature averaged 28.3 C.",
        "The daily minimum temperature averaged 22.2 C.",
        "temperature_2m_max is near normal.",
        "Rainfall totals 55.1 mm over 5 rainy days.",
        "precipitation_sum is below normal: -70.4 percent.",
    ]
    ordered = InsightAgent._prioritise(findings, ["rainfall"])
    assert "Rainfall" in ordered[0]
    assert "precipitation_sum" in ordered[1]
    # Everything is kept, only reordered.
    assert sorted(ordered) == sorted(findings)


def test_prioritise_is_a_no_op_without_requested_variables():
    from src.agents.insight_agent import InsightAgent

    findings = ["b", "a"]
    assert InsightAgent._prioritise(findings, []) == findings


# ---------------------------------------------------------------- RAG
def test_chunker_splits_on_headings():
    text = "# Title\n\nIntro text.\n\n## Section A\n\nBody A.\n\n## Section B\n\nBody B."
    sections = split_sections(text)
    headings = [h for h, _ in sections]
    assert "Section A" in headings and "Section B" in headings


def test_hashing_embedder_produces_unit_vectors():
    emb = HashingEmbedder(dim=256)
    vectors = emb.encode(["monsoon rainfall in Mumbai", "heat wave in Delhi"])
    assert vectors.shape == (2, 256)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_idf_makes_a_rare_term_outrank_a_common_one():
    corpus = ["temperature reading"] * 9 + ["temperature monsoon"]
    emb = HashingEmbedder(dim=4096).fit(corpus)
    rare = emb.idf[emb._index("monsoon")]
    common = emb.idf[emb._index("temperature")]
    assert rare > common


def test_similar_text_scores_higher_than_unrelated_text():
    emb = HashingEmbedder(dim=4096)
    vectors = emb.encode([
        "heavy rainfall warning for the coastal district",
        "rainfall warning issued for coastal areas",
        "stock market closed higher on tuesday",
    ])
    related = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert related > unrelated


def test_rrf_promotes_a_document_both_retrievers_agree_on():
    """The whole point of fusion: consensus beats one confident list."""
    from src.rag.fusion import reciprocal_rank_fusion

    lexical = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    semantic = [{"id": "x"}, {"id": "b"}, {"id": "y"}]
    fused = reciprocal_rank_fusion([lexical, semantic])

    # b is 2nd in both lists; a and x are 1st in one and absent from the
    # other, so neither accumulates a second contribution.
    assert fused[0]["id"] == "b"
    assert fused[0]["sources_agreeing"] == 2


def test_rrf_top_rank_still_carries_weight():
    """Consensus is not absolute: 1st + 3rd narrowly outranks 2nd + 2nd.

    With k = 60 the gaps between adjacent ranks are deliberately small, so a
    document one retriever is confident about is not buried by a document
    both retrievers rate as merely decent.
    """
    from src.rag.fusion import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([
        [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        [{"id": "c"}, {"id": "b"}, {"id": "d"}],
    ])
    # c: 1/63 + 1/61 = 0.032266 vs b: 1/62 + 1/62 = 0.032258
    assert [item["id"] for item in fused[:2]] == ["c", "b"]
    assert fused[0]["fusion_score"] > fused[1]["fusion_score"]


def test_rrf_keeps_documents_found_by_only_one_retriever():
    from src.rag.fusion import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([[{"id": "a"}], [{"id": "z"}]])
    assert {item["id"] for item in fused} == {"a", "z"}
    assert all(item["sources_agreeing"] == 1 for item in fused)


def test_rrf_is_unaffected_by_score_scale():
    """Ranks only - a backend with huge raw scores must not dominate."""
    from src.rag.fusion import reciprocal_rank_fusion

    small = [{"id": "a", "score": 0.01}, {"id": "b", "score": 0.009}]
    huge = [{"id": "b", "score": 900.0}, {"id": "a", "score": 800.0}]
    fused = reciprocal_rank_fusion([small, huge])
    # Perfectly symmetric input: both documents tie, neither list wins.
    assert {item["id"] for item in fused} == {"a", "b"}
    assert fused[0]["fusion_score"] == pytest.approx(fused[1]["fusion_score"])


def test_rrf_handles_an_empty_list():
    from src.rag.fusion import reciprocal_rank_fusion

    assert reciprocal_rank_fusion([[], []]) == []


# ---------------------------------------------------------------- contracts
def test_insight_rejects_an_invalid_confidence_value():
    with pytest.raises(Exception):
        Insight(answer="x", key_findings=[], recommendations=[],
                confidence="very sure")


# ---------------------------------------------------------------- network
@pytest.mark.network
def test_live_pipeline_end_to_end():
    from src.agents.orchestrator import get_orchestrator

    result = get_orchestrator().answer("What is the weather in Mumbai right now?")
    assert result.ok
    assert result.plan.intent == "current"
    assert result.bundle is not None
    assert result.insight.answer
