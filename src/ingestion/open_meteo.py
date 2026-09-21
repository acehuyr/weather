"""Open-Meteo provider.

Why Open-Meteo: no API key, generous free tier, and - crucially for the
"compared with previous years" questions - a reanalysis archive going back
to 1940 through the same request shape.

Two endpoints are used:
  * forecast  -> current conditions + next N days
  * archive   -> daily observations for any past range (about 5 days behind)
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from config.settings import settings
from src.core.errors import ProviderError
from src.core.logging import get_logger
from src.core.models import Location
from src.ingestion.cache import cached_get

log = get_logger("ingestion.open_meteo")

CURRENT_FIELDS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "weather_code", "wind_speed_10m", "surface_pressure",
]

DAILY_FIELDS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "wind_speed_10m_max",
]

# Presentation-only fields for the dashboard. Deliberately separate from
# DAILY_FIELDS: `weather_code` is a category, not a measurement, and feeding
# it to the trend fitter would produce a confident slope through icon
# numbers.
DASHBOARD_DAILY_FIELDS = DAILY_FIELDS + [
    "weather_code", "apparent_temperature_max", "apparent_temperature_min",
    "precipitation_probability_max", "uv_index_max", "sunrise", "sunset",
]

HOURLY_FIELDS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "precipitation_probability", "weather_code",
    "wind_speed_10m",
]

# The reanalysis archive lags real time. Asking for yesterday returns nulls.
ARCHIVE_LAG_DAYS = 5

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow",
    75: "Heavy snow", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 95: "Thunderstorm", 96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


def _to_frame(daily: dict) -> pd.DataFrame:
    """Open-Meteo returns parallel arrays; turn them into a tidy frame."""
    if not daily or "time" not in daily:
        return pd.DataFrame()
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "date"}).set_index("date").sort_index()
    # Columns the provider could not supply come back all-null - drop them so
    # downstream code can rely on `col in df.columns` meaning "usable".
    return df.dropna(axis=1, how="all")


def fetch_current(loc: Location) -> dict:
    """Current conditions as a flat dict."""
    payload = cached_get(
        settings.open_meteo_forecast_url,
        {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "current": ",".join(CURRENT_FIELDS),
            "timezone": "auto",
        },
        ttl=900,  # 15 min - "current" should not be older than that
    )
    current = dict(payload.get("current", {}))
    units = payload.get("current_units", {})
    code = current.get("weather_code")
    current["condition"] = WEATHER_CODES.get(code, "Unknown")
    current["units"] = units
    return current


def fetch_forecast(loc: Location, days: int | None = None) -> pd.DataFrame:
    """Daily forecast for the next `days` days (Open-Meteo caps at 16)."""
    days = min(days or settings.default_forecast_days, 16)
    payload = cached_get(
        settings.open_meteo_forecast_url,
        {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "daily": ",".join(DAILY_FIELDS),
            "forecast_days": days,
            "timezone": "auto",
        },
        ttl=3600,
    )
    df = _to_frame(payload.get("daily", {}))
    log.info("forecast: %d days for %s", len(df), loc.name)
    return df


HOURLY_FIELDS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "precipitation_probability", "weather_code",
    "wind_speed_10m",
]


def fetch_hourly(loc: Location, hours: int = 24) -> pd.DataFrame:
    """Hour-by-hour forecast, trimmed to start at the current hour.

    Open-Meteo returns whole days, so the first rows are usually in the past.
    An "hourly forecast" that opens at 00:00 this morning is confusing, so
    the frame is cut to now.
    """
    payload = cached_get(
        settings.open_meteo_forecast_url,
        {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "hourly": ",".join(HOURLY_FIELDS),
            "forecast_days": 3,
            "timezone": "auto",
        },
        ttl=1800,
    )

    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index().dropna(axis=1, how="all")

    # Local wall-clock "now" for this location, as the API reports it.
    now = pd.Timestamp.now(tz=payload.get("timezone", "UTC")).tz_localize(None)
    upcoming = df[df.index >= now.floor("h")]
    if upcoming.empty:
        upcoming = df

    df = upcoming.head(hours)
    df["condition"] = [
        WEATHER_CODES.get(code, "Unknown")
        for code in df.get("weather_code", pd.Series([None] * len(df)))
    ]
    log.info("hourly: %d hours for %s", len(df), loc.name)
    return df


def fetch_rich_forecast(loc: Location, days: int = 10) -> pd.DataFrame:
    """Daily forecast with the extra fields the dashboard displays.

    Kept apart from `fetch_forecast` so the analysis pipeline never sees
    categorical columns like `weather_code`.
    """
    days = min(days, 16)
    payload = cached_get(
        settings.open_meteo_forecast_url,
        {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "daily": ",".join(DASHBOARD_DAILY_FIELDS),
            "forecast_days": days,
            "timezone": "auto",
        },
        ttl=3600,
    )
    daily = payload.get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame()

    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "date"}).set_index("date").sort_index()
    # Unlike the analysis path we keep all-null columns here, because the
    # dashboard renders "-" for a missing field rather than assuming it.
    if "weather_code" in df.columns:
        df["condition"] = df["weather_code"].map(
            lambda c: WEATHER_CODES.get(int(c), "Unknown") if pd.notna(c) else "Unknown"
        )
    log.info("rich forecast: %d days for %s", len(df), loc.name)
    return df


def fetch_hourly(loc: Location, hours: int = 48) -> pd.DataFrame:
    """Hour-by-hour conditions, starting from the current hour."""
    payload = cached_get(
        settings.open_meteo_forecast_url,
        {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "hourly": ",".join(HOURLY_FIELDS),
            "forecast_days": max(2, (hours // 24) + 1),
            "timezone": "auto",
        },
        ttl=1800,
    )
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    if "weather_code" in df.columns:
        df["condition"] = df["weather_code"].map(
            lambda c: WEATHER_CODES.get(int(c), "Unknown") if pd.notna(c) else "Unknown"
        )

    # The API returns the whole day including hours already past; a forecast
    # strip should start now, not at midnight.
    now = pd.Timestamp.now().floor("h")
    upcoming = df[df.index >= now]
    if upcoming.empty:
        upcoming = df
    return upcoming.head(hours)


def fetch_history(loc: Location, start: date, end: date) -> pd.DataFrame:
    """Daily observations for a past range.

    The end date is clamped to the archive's availability so a request for
    "the last 30 days" does not come back half empty.
    """
    latest = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    end = min(end, latest)
    if start > end:
        start = end - timedelta(days=30)
    if start > end:
        raise ProviderError("Requested history range is not available yet.")

    payload = cached_get(
        settings.open_meteo_archive_url,
        {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(DAILY_FIELDS),
            "timezone": "auto",
        },
        ttl=60 * 60 * 24,  # the past does not change
    )
    df = _to_frame(payload.get("daily", {}))
    log.info("history: %d days (%s -> %s) for %s", len(df), start, end, loc.name)
    return df


def fetch_same_window_across_years(
    loc: Location, start: date, end: date, years: int
) -> pd.DataFrame:
    """The same calendar window in each of the previous `years` years.

    This is what makes "how does this compare with previous years?"
    answerable: one frame, one extra `year` column, ready to group by.
    """
    frames = []
    for offset in range(1, years + 1):
        try:
            s = start.replace(year=start.year - offset)
            e = end.replace(year=end.year - offset)
        except ValueError:  # 29 Feb in a non-leap year
            s = start.replace(year=start.year - offset, day=28)
            e = end.replace(year=end.year - offset, day=28)
        try:
            past = fetch_history(loc, s, e)
        except ProviderError as exc:
            log.warning("skipping year %d: %s", s.year, exc)
            continue
        if not past.empty:
            past = past.copy()
            past["year"] = s.year
            frames.append(past)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()
