"""Trend detection and period-over-period comparison.

Deliberately dependency-light: a least-squares fit via numpy.polyfit plus a
hand-computed R-squared. No scipy needed, and every number is explainable in
a viva.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.analysis.statistics import jsonable


def linear_trend(series: pd.Series) -> dict[str, Any]:
    """Fit y = m*x + c over time and describe the slope in human units."""
    s = series.dropna()
    if len(s) < 3:
        return {"status": "insufficient_data", "points": int(len(s))}

    # x in days since the first observation, so the slope has real units.
    x = (s.index - s.index[0]).days.to_numpy(dtype=float)
    y = s.to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    span_days = float(x[-1] - x[0]) or 1.0
    change = slope * span_days

    # A weak fit means "no reliable trend" - say so rather than over-claiming.
    if r2 < 0.1 or abs(change) < 0.1:
        direction = "stable"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return {
        "status": "ok",
        "direction": direction,
        "slope_per_day": jsonable(slope),
        "slope_per_decade": jsonable(slope * 3652.5),
        "total_change_over_period": jsonable(change),
        "r_squared": jsonable(r2),
        "points": int(len(s)),
        "span_days": int(span_days),
        "confidence": "high" if r2 >= 0.5 else "medium" if r2 >= 0.2 else "low",
    }


def analyse_trends(df: Optional[pd.DataFrame]) -> dict[str, Any]:
    """Run linear_trend over every numeric column of a frame."""
    if df is None or df.empty:
        return {}
    out = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "year":
            continue
        out[col] = linear_trend(df[col])
    return out


def rolling(df: pd.DataFrame, column: str, window: int = 7) -> pd.Series:
    """Smoothed series for plotting - raw daily data is too noisy to read."""
    if column not in df.columns:
        return pd.Series(dtype=float)
    return df[column].rolling(window=window, min_periods=max(1, window // 2)).mean()


def year_over_year(
    current: Optional[pd.DataFrame], past: Optional[pd.DataFrame]
) -> dict[str, Any]:
    """Compare this year's window against the same window in earlier years.

    `past` must carry the `year` column produced by
    `open_meteo.fetch_same_window_across_years`.
    """
    if current is None or current.empty or past is None or past.empty:
        return {}
    if "year" not in past.columns:
        return {}

    numeric = [
        c for c in current.select_dtypes(include=[np.number]).columns if c != "year"
    ]
    result: dict[str, Any] = {"years_compared": sorted(int(y) for y in past["year"].unique())}

    for col in numeric:
        if col not in past.columns:
            continue
        # Rain is a total; everything else is an average.
        agg = "sum" if "precipitation" in col else "mean"
        now = getattr(current[col].dropna(), agg)()
        per_year = past.groupby("year")[col].agg(agg)
        if per_year.empty or pd.isna(now):
            continue
        baseline = float(per_year.mean())
        delta = float(now) - baseline
        pct = (delta / baseline * 100) if baseline else 0.0
        result[col] = {
            "aggregate": agg,
            "current": jsonable(now),
            "historical_mean": jsonable(baseline),
            "difference": jsonable(delta),
            "percent_change": jsonable(pct),
            "by_year": {str(int(y)): jsonable(v) for y, v in per_year.items()},
            "verdict": (
                "above normal" if pct > 10 else
                "below normal" if pct < -10 else
                "near normal"
            ),
        }
    return result
