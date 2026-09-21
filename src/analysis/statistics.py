"""Descriptive statistics.

Every function returns plain JSON-safe dicts - they are fed straight into
both the FastAPI response and the LLM prompt, so numpy scalars and NaN must
never escape this module.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def jsonable(value: Any) -> Any:
    """numpy/pandas scalar -> plain Python, NaN -> None."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else round(float(value), 2)
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def summarize(df: Optional[pd.DataFrame]) -> dict[str, Any]:
    """Min / max / mean / std / first / last for every numeric column."""
    if df is None or df.empty:
        return {}

    out: dict[str, Any] = {
        "rows": int(len(df)),
        "start": jsonable(df.index.min()),
        "end": jsonable(df.index.max()),
        "variables": {},
    }
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "year":
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        out["variables"][col] = {
            "mean": jsonable(s.mean()),
            "min": jsonable(s.min()),
            "max": jsonable(s.max()),
            "std": jsonable(s.std()),
            "latest": jsonable(s.iloc[-1]),
            "min_on": jsonable(s.idxmin()),
            "max_on": jsonable(s.idxmax()),
        }
    return out


def rainfall_profile(df: Optional[pd.DataFrame]) -> dict[str, Any]:
    """Rain-specific numbers that a mean alone hides."""
    if df is None or df.empty or "precipitation_sum" not in df.columns:
        return {}
    s = df["precipitation_sum"].dropna()
    if s.empty:
        return {}
    return {
        "total_mm": jsonable(s.sum()),
        "wet_days": int((s >= 2.5).sum()),      # IMD: >= 2.5 mm counts as a rainy day
        "dry_days": int((s < 2.5).sum()),
        "heaviest_day_mm": jsonable(s.max()),
        "heaviest_day": jsonable(s.idxmax()),
        "mean_on_wet_days": jsonable(s[s >= 2.5].mean()),
    }
