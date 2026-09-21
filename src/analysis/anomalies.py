"""Anomaly detection.

"Unusual" only means something relative to a baseline, so the primary method
here compares each day against the *climatological normal* for that time of
year - the mean and spread of the same calendar window across several past
years. A plain z-score over the last 30 days would flag every winter day as
cold, which is not an anomaly, just January.

Two layers:
  1. statistical  - z-score vs. the day-of-year normal (falls back to IQR)
  2. rule-based   - IMD threshold categories, which give the LLM concrete
                    vocabulary ("very heavy rainfall") instead of raw numbers
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from config.settings import settings
from src.analysis.statistics import jsonable

DOY_WINDOW = 7  # +/- days pooled when computing the normal for a date

# India Meteorological Department 24-hour rainfall categories (mm),
# ordered high -> low so the first match wins.
RAINFALL_CATEGORIES = [
    (204.5, "extremely heavy rainfall"),
    (115.6, "very heavy rainfall"),
    (64.5, "heavy rainfall"),
    (15.6, "moderate rainfall"),
    (2.5, "light rainfall"),
    (0.1, "very light rainfall"),
]


def _doy_normals(baseline: pd.DataFrame, column: str) -> dict[int, tuple[float, float]]:
    """day-of-year -> (mean, std), pooled over a +/-DOY_WINDOW window."""
    if column not in baseline.columns:
        return {}
    s = baseline[column].dropna()
    if s.empty:
        return {}

    doy = s.index.dayofyear
    normals: dict[int, tuple[float, float]] = {}
    for target in range(1, 367):
        # Circular distance so 31 Dec and 1 Jan are neighbours.
        diff = np.abs(doy - target)
        diff = np.minimum(diff, 366 - diff)
        window = s[diff <= DOY_WINDOW]
        if len(window) >= 3:
            normals[target] = (float(window.mean()), float(window.std()))
    return normals


def _iqr_flags(series: pd.Series) -> list[dict[str, Any]]:
    """Fallback when there is no multi-year baseline to compare against."""
    s = series.dropna()
    if len(s) < 8:
        return []
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return []
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = []
    for ts, value in s[(s < lo) | (s > hi)].items():
        out.append({
            "date": jsonable(ts),
            "value": jsonable(value),
            "method": "iqr",
            "direction": "above" if value > hi else "below",
            "expected_range": [jsonable(lo), jsonable(hi)],
        })
    return out


def detect(
    recent: Optional[pd.DataFrame],
    baseline: Optional[pd.DataFrame] = None,
    z_threshold: float | None = None,
) -> dict[str, Any]:
    """Compare `recent` against `baseline`; return findings per variable."""
    z_threshold = z_threshold or settings.anomaly_z_threshold
    if recent is None or recent.empty:
        return {"status": "no_data", "anomalies": []}

    has_baseline = baseline is not None and not baseline.empty
    findings: list[dict[str, Any]] = []
    per_variable: dict[str, Any] = {}

    columns = [c for c in recent.select_dtypes(include=[np.number]).columns if c != "year"]

    for col in columns:
        if has_baseline:
            normals = _doy_normals(baseline, col)
            if normals:
                hits = []
                for ts, value in recent[col].dropna().items():
                    stats = normals.get(int(ts.dayofyear))
                    if not stats:
                        continue
                    mean, std = stats
                    if std <= 0:
                        continue
                    z = (float(value) - mean) / std
                    if abs(z) >= z_threshold:
                        hits.append({
                            "date": jsonable(ts),
                            "variable": col,
                            "value": jsonable(value),
                            "normal": jsonable(mean),
                            "departure": jsonable(float(value) - mean),
                            "z_score": jsonable(z),
                            "method": "climatological_zscore",
                            "direction": "above" if z > 0 else "below",
                            "severity": "extreme" if abs(z) >= 3 else "notable",
                        })
                per_variable[col] = {
                    "method": "climatological_zscore",
                    "count": len(hits),
                    "baseline_days": int(baseline[col].notna().sum()) if col in baseline else 0,
                }
                findings.extend(hits)
                continue

        # No usable baseline for this column. IQR on daily rainfall flags
        # essentially every wet day, because the distribution is
        # zero-inflated and most days sit at exactly 0 - so Q1 = Q3 = 0 and
        # any rain at all becomes an "outlier". Fall back to the published
        # category thresholds instead, which is what they exist for.
        if "precipitation" in col or "rain" in col:
            per_variable[col] = {
                "method": "skipped",
                "count": 0,
                "reason": "daily rainfall is too skewed for distribution-based "
                          "outlier detection without a climatological baseline; "
                          "see threshold_flags instead",
            }
            continue

        hits = [dict(h, variable=col) for h in _iqr_flags(recent[col])]
        per_variable[col] = {"method": "iqr", "count": len(hits)}
        findings.extend(hits)

    findings.sort(key=lambda h: abs(h.get("z_score") or 0), reverse=True)

    return {
        "status": "ok",
        "method": "climatological_zscore" if has_baseline else "iqr",
        "z_threshold": z_threshold,
        "total": len(findings),
        "anomalies": findings[:15],   # keep the LLM prompt bounded
        "per_variable": per_variable,
    }


def categorise_rainfall(mm: float | None) -> str:
    """Turn a 24-hour rainfall total into IMD vocabulary."""
    if mm is None or (isinstance(mm, float) and np.isnan(mm)):
        return "unknown"
    for threshold, label in RAINFALL_CATEGORIES:
        if mm >= threshold:
            return label
    return "trace / no rain"


def flag_extremes(recent: Optional[pd.DataFrame]) -> list[dict[str, Any]]:
    """Rule-based flags that do not need any baseline at all."""
    if recent is None or recent.empty:
        return []
    flags: list[dict[str, Any]] = []

    if "precipitation_sum" in recent.columns:
        for ts, mm in recent["precipitation_sum"].dropna().items():
            label = categorise_rainfall(float(mm))
            if "heavy" in label:
                flags.append({
                    "date": jsonable(ts), "type": "rainfall",
                    "category": label, "value_mm": jsonable(mm),
                })

    if "temperature_2m_max" in recent.columns:
        for ts, t in recent["temperature_2m_max"].dropna().items():
            if float(t) >= 40:
                flags.append({
                    "date": jsonable(ts), "type": "heat",
                    "category": "heatwave threshold (>=40 C in plains)",
                    "value_c": jsonable(t),
                })

    return flags
