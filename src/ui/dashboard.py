"""Dashboard rendering.

Layout follows the shape mainstream weather sites converge on, because it
works: a hero block with the current reading, a short day/night summary, a
column of secondary metrics, then hourly and multi-day views behind
sub-navigation. The wording, styling, metrics and charts here are this
project's own - no branded terms, no copied copy.

What this adds over a normal weather app is the last tab: the same data run
through the analysis layer, so you see whether today is actually unusual
rather than just what the number is.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.anomalies import categorise_rainfall
from src.ui.theme import (
    ACCENT,
    HOT,
    MUTED,
    RAIN,
    TEXT,
    departure_colour,
    style_figure,
    temp_colour,
)

# Weather code -> emoji. Open-Meteo uses WMO codes; these group them into the
# handful of visual states a reader actually distinguishes.
ICONS = {
    0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
    45: "🌫️", 48: "🌫️",
    51: "🌦️", 53: "🌦️", 55: "🌦️", 56: "🌦️", 57: "🌦️",
    61: "🌧️", 63: "🌧️", 65: "🌧️", 66: "🌨️", 67: "🌨️",
    71: "❄️", 73: "❄️", 75: "❄️", 77: "❄️",
    80: "🌧️", 81: "🌧️", 82: "⛈️",
    85: "🌨️", 86: "🌨️",
    95: "⛈️", 96: "⛈️", 99: "⛈️",
}

# How far the departure track runs, in degrees either side of normal.
DEPARTURE_SPAN = 6.0


def icon_for(code) -> str:
    try:
        return ICONS.get(int(code), "🌡️")
    except (TypeError, ValueError):
        return "🌡️"


def compute_departure(loc, years: int = 10) -> dict:
    """Today's max against the climatological normal for this date.

    Pools the same calendar window across `years` of archive data, exactly
    as the anomaly agent does, so the headline figure and the analysis tab
    can never disagree.
    """
    from datetime import date, timedelta

    from src.analysis.anomalies import _doy_normals
    from src.core.errors import WeatherIQError
    from src.ingestion.open_meteo import fetch_history, fetch_rich_forecast

    try:
        end = date.today() - timedelta(days=5)      # archive lag
        baseline = fetch_history(loc, end.replace(year=end.year - years), end)
        if baseline.empty:
            return {}

        normals = _doy_normals(baseline, "temperature_2m_max")
        stats = normals.get(date.today().timetuple().tm_yday)
        if not stats:
            return {}

        today = fetch_rich_forecast(loc, 1)
        if today.empty:
            return {}
        observed = today.iloc[0].get("temperature_2m_max")
        if observed is None or pd.isna(observed):
            return {}

        mean, sigma = stats
        return {
            "observed": float(observed),
            "normal": float(mean),
            "sigma": float(sigma),
            "departure": float(observed) - float(mean),
            "years": years,
        }
    except WeatherIQError:
        # The hero must still render without a baseline; the strip is simply
        # omitted rather than showing a number we cannot justify.
        return {}


def pretty_label(place) -> str:
    """`Location.label` without the repeats.

    A country-level match geocodes to name == admin1 == country, so the raw
    label renders as "India, India". Fixed here rather than on the model,
    because that label is also fed to the language model as prompt context
    and changing it would change what the model is asked.
    """
    seen, parts = set(), []
    for bit in (place.name, place.admin1, place.country):
        if bit and bit not in seen:
            seen.add(bit)
            parts.append(bit)
    return ", ".join(parts)


def _fmt(value, unit: str = "", digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------

def verdict_sentence(departure: dict) -> tuple[str, str]:
    """Today against its normal, said in words before it is said in numbers.

    The hero used to lead with `DEPARTURE FROM NORMAL` and
    `dashed band = ±1σ (1.1°)` - the single best idea in this project,
    written so that only a meteorologist could read it. The statistics have
    not gone anywhere; they are now the footnote rather than the headline.
    """
    if not departure:
        return "", ""

    delta = departure["departure"]
    sigma = departure.get("sigma") or 0
    years = departure.get("years", 10)
    size = abs(delta)

    if size < 0.75:
        headline = "About normal for the date."
    else:
        direction = "warmer" if delta > 0 else "cooler"
        # Two sigma is the conventional line between ordinary variation and
        # something worth remarking on, and it is the same threshold the
        # anomaly agent uses, so the two can never contradict each other.
        if sigma and size >= 2 * sigma:
            strength = "Much"
        elif size >= 2:
            strength = "Noticeably"
        else:
            strength = "Slightly"
        headline = f"{strength} {direction} than usual for the date."

    detail = (
        f"Today's {departure['observed']:.0f}° is {size:.1f}° "
        f"{'above' if delta >= 0 else 'below'} the "
        f"{departure['normal']:.1f}° average for this date across the last "
        f"{years} years."
    )
    return headline, detail


def _departure_markup(departure: dict) -> str:
    """The signature element: today's reading against its normal.

    A raw temperature means nothing without a baseline - 31 degrees is
    ordinary in Mumbai and alarming in Shimla. This track puts the ten-year
    normal for this calendar date at the centre, shades one standard
    deviation either side, and marks where today actually sits.

    Reading order is deliberate: the plain sentence, then the picture, then
    the method. A reader who stops after the first line has still got the
    answer.
    """
    if not departure:
        return ""

    delta = departure["departure"]
    sigma = departure.get("sigma") or 0

    def to_pct(degrees: float) -> float:
        return max(1.5, min(98.5, 50 + (degrees / DEPARTURE_SPAN) * 50))

    mark = to_pct(delta)
    band_left = to_pct(-sigma)
    band_width = to_pct(sigma) - band_left
    colour = departure_colour(delta)
    headline, detail = verdict_sentence(departure)

    return f"""
<div class="dep">
  <div class="verdict" style="color:{colour}">{headline}</div>
  <div class="verdict-sub">{detail}</div>
  <div class="dep-track" style="margin-top:14px">
    <div class="dep-band" style="left:{band_left:.1f}%;width:{band_width:.1f}%"></div>
    <div class="dep-zero"></div>
    <div class="dep-mark" style="left:{mark:.1f}%;background:{colour}"></div>
  </div>
  <div class="dep-scale">
    <span>−{DEPARTURE_SPAN:.0f}° cooler</span>
    <span>normal {departure['normal']:.1f}°</span>
    <span>+{DEPARTURE_SPAN:.0f}° warmer</span>
  </div>
  <div class="dep-note" style="margin-top:9px">
    Marker is today at <b style="color:{colour}">{delta:+.1f}°</b>. Baseline:
    {departure['years']} years of the same calendar window, ±7 days; the
    dashed band is one standard deviation (±{sigma:.1f}°).
  </div>
</div>"""


def render_hero(place, current: dict, daily: pd.DataFrame,
                departure: dict = None) -> None:
    """Current reading, location, and the departure scale beneath it."""
    temp = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    icon = icon_for(current.get("weather_code"))
    condition = current.get("condition", "")

    hi = lo = None
    if daily is not None and not daily.empty:
        row = daily.iloc[0]
        hi = row.get("temperature_2m_max")
        lo = row.get("temperature_2m_min")

    st.markdown(
        f"""
<div class="hero">
  <div class="hero-top">
    <div class="hero-read">
      <div class="hero-icon">{icon}</div>
      <div>
        <div class="hero-temp" style="color:{temp_colour(temp)}">
          {_fmt(temp, "", 1)}<span class="hero-unit">°C</span>
        </div>
        <div class="hero-cond">{condition} · feels like {_fmt(feels, "°", 0)}</div>
      </div>
    </div>
    <div class="hero-meta">
      <div class="hero-place">{pretty_label(place)}</div>
      <div class="hero-coord">{place.latitude:.3f}, {place.longitude:.3f}</div>
      <div class="hero-hilo">today
        <b class="num">{_fmt(hi, "°")}</b> / <b class="num">{_fmt(lo, "°")}</b>
      </div>
    </div>
  </div>
  {_departure_markup(departure)}
</div>
""",
        unsafe_allow_html=True,
    )


def render_today_summary(daily: pd.DataFrame) -> None:
    """Day and night in one sentence each, composed from the numbers.

    Open-Meteo returns measurements, not prose, so the narrative is built
    here rather than quoted from anywhere.
    """
    if daily is None or daily.empty:
        return
    row = daily.iloc[0]
    rain_chance = row.get("precipitation_probability_max")
    rain_mm = row.get("precipitation_sum")

    day_bits = [str(row.get("condition", "Unknown"))]
    if rain_chance is not None and not pd.isna(rain_chance) and rain_chance >= 20:
        day_bits.append(f"{int(rain_chance)}% chance of rain")
    if rain_mm and not pd.isna(rain_mm) and rain_mm >= 2.5:
        day_bits.append(categorise_rainfall(float(rain_mm)))

    st.markdown(
        f"""
<div class="card">
  <div class="card-head">Today <span class="card-date">
    {daily.index[0].strftime('%a, %d %b')}</span></div>
  <div class="summary-row">
    <span class="summary-icon">{icon_for(row.get('weather_code'))}</span>
    <span>{'; '.join(day_bits)}
      <b>High {_fmt(row.get('temperature_2m_max'), '°')}</b></span>
  </div>
  <div class="summary-row">
    <span class="summary-icon">🌙</span>
    <span>Overnight low
      <b>{_fmt(row.get('temperature_2m_min'), '°')}</b>, feeling like
      {_fmt(row.get('apparent_temperature_min'), '°')}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_current_details(current: dict, daily: pd.DataFrame) -> None:
    """Secondary metrics as labelled rows."""
    uv = None
    if daily is not None and not daily.empty:
        uv = daily.iloc[0].get("uv_index_max")

    rows = [
        ("Feels like", _fmt(current.get("apparent_temperature"), " °C")),
        ("Humidity", _fmt(current.get("relative_humidity_2m"), "%")),
        ("Wind", _fmt(current.get("wind_speed_10m"), " km/h", 1)),
        ("Rain now", _fmt(current.get("precipitation"), " mm", 1)),
        ("Pressure", _fmt(current.get("surface_pressure"), " hPa")),
        ("UV index today", _fmt(uv, "", 1)),
    ]
    body = "".join(
        f"<div class='metric-row'><span>{label}</span><b>{value}</b></div>"
        for label, value in rows
    )
    st.markdown(
        f"<div class='card'><div class='card-head'>Right now</div>{body}</div>",
        unsafe_allow_html=True,
    )


def render_outlook(daily: pd.DataFrame) -> None:
    """One forward-looking line: the next day that is worth knowing about."""
    if daily is None or len(daily) < 2:
        return

    note = None
    for when, row in daily.iloc[1:].iterrows():
        rain = row.get("precipitation_sum")
        if rain and not pd.isna(rain) and rain >= 15.6:
            note = (f"{when.strftime('%A')}: {categorise_rainfall(float(rain))} "
                    f"expected, around {rain:.0f} mm.")
            break
        hot = row.get("temperature_2m_max")
        if hot and not pd.isna(hot) and hot >= 38:
            note = f"{when.strftime('%A')} turns hot, near {hot:.0f}°."
            break

    if note is None:
        highs = daily["temperature_2m_max"].dropna()
        if len(highs) > 1:
            swing = highs.max() - highs.min()
            note = (f"Steady week ahead - highs vary by only {swing:.0f}°."
                    if swing < 3 else
                    f"Highs range from {highs.min():.0f}° to {highs.max():.0f}° "
                    "over the coming days.")

    if note:
        st.markdown(
            f"<div class='card'><div class='card-head'>Looking ahead</div>"
            f"<div class='outlook'>{note}</div></div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def hourly_chart(hourly: pd.DataFrame, key: str) -> None:
    """Temperature line with rain probability underneath."""
    if hourly is None or hourly.empty:
        st.info("Hourly data is unavailable for this location.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly.index, y=hourly["temperature_2m"], name="Temperature",
        mode="lines", line=dict(color=temp_colour(30), width=2.5),
        hovertemplate="%{x|%H:%M}<br>%{y:.1f} °C<extra></extra>",
    ))
    if "apparent_temperature" in hourly.columns:
        fig.add_trace(go.Scatter(
            x=hourly.index, y=hourly["apparent_temperature"], name="Feels like",
            mode="lines", line=dict(color=HOT, width=1.4, dash="dot"),
            hovertemplate="%{x|%H:%M}<br>feels %{y:.1f} °C<extra></extra>",
        ))
    if "precipitation_probability" in hourly.columns:
        fig.add_trace(go.Bar(
            x=hourly.index, y=hourly["precipitation_probability"],
            name="Rain chance", yaxis="y2",
            marker_color="rgba(91,155,213,.42)",
            hovertemplate="%{x|%H:%M}<br>%{y:.0f}%% chance<extra></extra>",
        ))

    # Plotly stacks a date under every tick by default, which at hourly
    # resolution produces an unreadable picket fence. One tick every three
    # hours, time only, with the date carried by the axis title instead.
    span_days = (hourly.index[-1] - hourly.index[0]).days
    fig.update_layout(
        yaxis=dict(title="°C"),
        yaxis2=dict(title="rain chance %", overlaying="y", side="right",
                    range=[0, 100], showgrid=False),
        hovermode="x unified",
        xaxis=dict(
            tickformat="%H:%M",
            dtick=3 * 3600_000 if span_days < 2 else 6 * 3600_000,
            tickangle=0,
            title=dict(
                text=hourly.index[0].strftime("from %a %d %b"),
                font=dict(size=11),
            ),
        ),
    )
    st.plotly_chart(style_figure(fig, 320), width="stretch", key=key)


def range_chart(daily: pd.DataFrame, key: str) -> None:
    """Daily high-low range as bars, with rainfall behind."""
    if daily is None or daily.empty:
        return

    labels = [d.strftime("%a %d") for d in daily.index]
    lows = daily["temperature_2m_min"]
    highs = daily["temperature_2m_max"]

    fig = go.Figure()
    if "precipitation_sum" in daily.columns:
        fig.add_trace(go.Bar(
            x=labels, y=daily["precipitation_sum"], name="Rain (mm)",
            marker_color="rgba(91,155,213,.28)", yaxis="y2",
            hovertemplate="%{y:.1f} mm<extra></extra>",
        ))

    # One bar per day spanning low to high - the range is the story, and a
    # pair of separate lines makes the reader do the subtraction.
    fig.add_trace(go.Bar(
        x=labels, y=(highs - lows), base=lows, name="Low to high",
        marker=dict(color=[temp_colour(v) for v in highs]),
        width=0.45,
        customdata=list(zip(lows, highs)),
        hovertemplate="low %{customdata[0]:.0f}° · high %{customdata[1]:.0f}°"
                      "<extra></extra>",
    ))

    # Plotly would otherwise start the axis at the lowest bar base, so every
    # bar appears to sit on the floor and the overnight lows vanish. Pad the
    # range so the bottom of each range bar is actually readable.
    floor = float(lows.min()) - 3 if lows.notna().any() else 0
    ceiling = float(highs.max()) + 2 if highs.notna().any() else 40

    fig.update_layout(
        barmode="overlay",
        yaxis=dict(title="°C", range=[floor, ceiling]),
        yaxis2=dict(title="rain mm", overlaying="y", side="right",
                    showgrid=False),
    )
    st.plotly_chart(style_figure(fig, 330), width="stretch", key=key)


def day_cards(daily: pd.DataFrame) -> None:
    """A compact strip of upcoming days."""
    if daily is None or daily.empty:
        return
    for chunk_start in range(0, len(daily), 5):
        chunk = daily.iloc[chunk_start:chunk_start + 5]
        for column, (when, row) in zip(st.columns(len(chunk)), chunk.iterrows()):
            chance = row.get("precipitation_probability_max")
            column.markdown(
                f"""
<div class="day-card">
  <div class="day-name">{when.strftime('%a')}</div>
  <div class="day-date">{when.strftime('%d %b')}</div>
  <div class="day-icon">{icon_for(row.get('weather_code'))}</div>
  <div class="day-hi" style="color:{temp_colour(row.get('temperature_2m_max'))}">
    {_fmt(row.get('temperature_2m_max'), '°')}</div>
  <div class="day-lo">{_fmt(row.get('temperature_2m_min'), '°')}</div>
  <div class="day-rain">{'💧 ' + str(int(chance)) + '%'
      if chance is not None and not pd.isna(chance) and chance >= 10 else '&nbsp;'}</div>
</div>
""",
                unsafe_allow_html=True,
            )
