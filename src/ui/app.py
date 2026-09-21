"""Streamlit interface.

    streamlit run src/ui/app.py

Four tabs, because the app serves two different people: someone who wants an
answer, and an examiner who wants to see how the answer was produced.

    Chat       conversational Q&A, with the evidence one click away
    Map        live weather on and around a place
    Dashboard  current conditions and charts at a glance
    How it works   architecture, agents, measured results

Runs the orchestrator in-process rather than calling the FastAPI service, so
the UI works standalone. The API exists for programmatic clients.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file directly, so the project root is not yet on
# sys.path when the imports below run.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from config.settings import settings
from src.agents.orchestrator import get_orchestrator
from src.agents.retrieval_agent import get_retriever
from src.analysis.trends import rolling
from src.core.errors import LocationNotFound, WeatherIQError
from src.ingestion.geocoding import detect_location_from_ip, geocode
from src.ingestion.open_meteo import (
    fetch_current,
    fetch_forecast,
    fetch_hourly,
    fetch_rich_forecast,
)
from src.ui import dashboard as dash
from src.ui import theme
from src.ui.weather_map import (
    build_map,
    fetch_points,
    legend_items,
    nearby_places,
    temperature_style,
    zoom_for,
)

st.set_page_config(
    page_title="Weather Insight Engine",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# All visual design lives in src/ui/theme.py - one place for tokens,
# typography and the Plotly styling, so charts and chrome cannot drift
# apart.
theme.inject()

# (button label, question actually sent) - a full question does not fit on a
# chip, and a truncated label tells the reader nothing.
SAMPLES = [
    ("Weather now", "What's the weather like in Mumbai right now?"),
    ("Rain this week?", "Will it rain in Pune this week?"),
    ("vs previous years",
     "How has Delhi's temperature changed vs previous years?"),
    ("Anything unusual?", "Why is the weather in Chennai unusual?"),
    ("What's a heat wave?", "What counts as a heat wave in India?"),
    ("Why so humid?",
     "Why does humid heat feel worse than dry heat at the same temperature?"),
]

CONFIDENCE_PILL = {
    "high": ("pill-green", "Well supported by the data"),
    "medium": ("pill-amber", "Supported, with gaps in the data"),
    "low": ("pill-red", "Weakly supported - indicative only"),
}

# Folium marker colour names -> hex, for the HTML legend.
SWATCH = {
    "darkred": "#8b0000", "red": "#d63e2a", "orange": "#f69730",
    "beige": "#ffcb92", "green": "#72b026", "lightblue": "#8adaff",
    "blue": "#38aadd", "darkblue": "#0067a3", "gray": "#a3a3a3",
}


# --------------------------------------------------------------- resources
@st.cache_resource
def orchestrator():
    """Cached so the RAG index loads once per session, not per question."""
    return get_orchestrator()


@st.cache_data(ttl=900, show_spinner=False)
def cached_geocode(place: str):
    return geocode(place)


@st.cache_data(ttl=900, show_spinner=False)
def cached_conditions(lat: float, lon: float, name: str):
    """Everything the dashboard needs, in one cached call."""
    from src.core.models import Location

    loc = Location(name=name, latitude=lat, longitude=lon)
    return (
        fetch_current(loc),
        fetch_rich_forecast(loc, 10),
        fetch_hourly(loc, 48),
        fetch_forecast(loc, 10),      # analysis-safe columns, for the charts
    )


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_departure(lat: float, lon: float, name: str):
    """Today's reading against its ten-year normal.

    Cached for six hours: the baseline is a decade of archive data that
    changes once a day at most, and it costs a ~1.5 s fetch on a cold cache.
    """
    from src.core.models import Location

    return dash.compute_departure(
        Location(name=name, latitude=lat, longitude=lon)
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_map_points(lat: float, lon: float, name: str, count: int):
    from src.core.models import Location

    centre = Location(name=name, latitude=lat, longitude=lon)
    return fetch_points([centre] + nearby_places(centre, count))


def active_location():
    """Whatever place the user has selected, resolved once."""
    if "location" not in st.session_state:
        st.session_state["location"] = cached_geocode("Mumbai")
    return st.session_state["location"]


# ------------------------------------------------------------------ charts
def plot_series(df: pd.DataFrame, title: str, key: str) -> None:
    """Temperature range plus rainfall on a secondary axis.

    `key` must be unique across the whole page. Streamlit derives an element
    ID from the widget type and its parameters, so two charts built from
    similar data collide and raise DuplicateElementId - and every chat
    message renders its own charts.
    """
    fig = go.Figure()

    if "temperature_2m_max" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["temperature_2m_max"],
            name="max", line=dict(color=theme.HOT, width=1.6),
        ))
    if "temperature_2m_min" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["temperature_2m_min"],
            name="min", line=dict(color=theme.COLD, width=1.6),
            fill="tonexty", fillcolor="rgba(74,143,212,.12)",
        ))
    # A 7-day mean is what makes a multi-year series readable at all.
    if "temperature_2m_mean" in df.columns and len(df) >= 14:
        fig.add_trace(go.Scatter(
            x=df.index, y=rolling(df, "temperature_2m_mean", 7),
            name="7-day mean",
            line=dict(color=theme.TEXT, width=1.6, dash="dot"),
        ))
    if "precipitation_sum" in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df["precipitation_sum"],
            name="rain", marker_color="rgba(91,155,213,.42)", yaxis="y2",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=theme.MUTED)),
        yaxis=dict(title="°C"),
        yaxis2=dict(title="rain mm", overlaying="y", side="right",
                    showgrid=False),
        hovermode="x unified",
    )
    st.plotly_chart(theme.style_figure(fig, 340), width="stretch", key=key)


def plot_year_comparison(yoy: dict, key_prefix: str) -> None:
    """Current value against each historical year, per variable."""
    for variable, block in yoy.items():
        if not isinstance(block, dict) or "by_year" not in block:
            continue
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(block["by_year"].keys()), y=list(block["by_year"].values()),
            name="previous years", marker_color=theme.MUTED,
        ))
        fig.add_hline(
            y=block["historical_mean"], line_dash="dash",
            line_color=theme.TEXT,
            annotation_text=f"mean {block['historical_mean']}",
        )
        fig.add_trace(go.Bar(
            x=["now"], y=[block["current"]], name="current",
            marker_color=theme.ACCENT,
        ))
        fig.update_layout(
            title=dict(
                text=f"{variable} · {block['verdict']}",
                font=dict(size=13, color=theme.MUTED),
            ),
            showlegend=False,
        )
        st.plotly_chart(theme.style_figure(fig, 290), width="stretch",
                        key=f"{key_prefix}_yoy_{variable}")


def render_evidence(result, key_prefix: str) -> None:
    """Charts, sources and the agent trace - the 'show your working' panel."""
    bundle = result.bundle

    if bundle is not None:
        if bundle.history is not None and not bundle.history.empty:
            plot_series(bundle.history, "Observed history",
                        key=f"{key_prefix}_hist")
        if bundle.forecast is not None and not bundle.forecast.empty:
            plot_series(bundle.forecast, "Forecast", key=f"{key_prefix}_fc")

    yoy = result.analysis.get("year_over_year") or {}
    if yoy:
        plot_year_comparison(
            {k: v for k, v in yoy.items() if k != "years_compared"},
            key_prefix,
        )

    left, right = st.columns(2)

    with left:
        st.markdown(f"**Retrieved knowledge** ({len(result.documents)} chunks)")
        if result.documents:
            for doc in result.documents:
                st.markdown(
                    f"<div class='src-card'><b>{doc.title}</b><br>"
                    f"<small><code>{doc.source}</code> &middot; "
                    f"similarity {doc.score:.3f}</small><br>"
                    f"<small>{doc.text[:220]}...</small></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Nothing was retrieved for this question.")

    with right:
        st.markdown("**Agent trace**")
        st.dataframe(
            pd.DataFrame([s.to_dict() for s in result.trace]),
            width="stretch", hide_index=True, key=f"{key_prefix}_trace",
        )
        anomalies = result.anomalies
        if anomalies.get("anomalies"):
            st.markdown(f"**Anomalies** ({anomalies.get('total', 0)})")
            st.caption(f"Compared against: {anomalies.get('baseline_used', '')}")
            st.dataframe(
                pd.DataFrame(anomalies["anomalies"]).head(10),
                width="stretch", hide_index=True, key=f"{key_prefix}_anom",
            )


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("<span class='eyebrow'>Station</span>", unsafe_allow_html=True)

    typed = st.text_input(
        "Search a place", value="",
        placeholder="Mumbai, Nagaur, Tokyo…",
        label_visibility="collapsed",
    )
    col_a, col_b = st.columns(2)
    if col_a.button("Set", width="stretch") and typed.strip():
        try:
            st.session_state["location"] = cached_geocode(typed)
        except LocationNotFound as exc:
            st.error(str(exc))
    if col_b.button("Near me", width="stretch"):
        try:
            st.session_state["location"] = detect_location_from_ip()
        except WeatherIQError as exc:
            st.warning(f"{exc} Search for a place instead.")

    place = active_location()
    st.markdown(
        f"<div class='card' style='margin-top:10px'>"
        f"<div class='hero-place'>{place.name}</div>"
        f"<div class='hero-coord'>{place.admin1 or place.country}</div>"
        f"<div class='hero-coord' style='margin-top:6px'>"
        f"{place.latitude:.3f}, {place.longitude:.3f}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<span class='eyebrow'>System</span>", unsafe_allow_html=True)

    # Status reads as four equal rows, not a warning banner. Whether a
    # language model is attached is a configuration fact, not an error - the
    # analysis, anomaly detection and retrieval all run either way.
    llm_on = settings.llm_enabled
    dot = theme.ACCENT if llm_on else theme.MUTED
    st.markdown(
        f"<div style='margin-top:8px'>"
        f"<div class='metric-row'><span>model</span>"
        f"<b style='color:{dot}'>"
        f"{settings.llm_label if llm_on else 'not connected'}</b></div>"
        f"<div class='metric-row'><span>retrieval</span>"
        f"<b>{settings.embedding_backend}</b></div>"
        f"<div class='metric-row'><span>indexed</span>"
        f"<b>{len(get_retriever())} chunks</b></div>"
        f"<div class='metric-row'><span>source</span>"
        f"<b>Open-Meteo</b></div></div>",
        unsafe_allow_html=True,
    )

    if not llm_on:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.expander("Turn on conversational answers"):
            st.markdown(
                "Answers currently come from a deterministic template. To get "
                "natural language and follow-up questions, add a **free** key:\n\n"
                "1. Open **console.groq.com/keys** and sign in\n"
                "2. Create a key and copy it\n"
                "3. Put it in `.env`:\n"
            )
            st.code("GROQ_API_KEY=gsk_your_key_here", language="bash")
            st.caption("No card required. Restart the app after saving.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.button("Clear chat", width="stretch"):
        st.session_state["messages"] = []
        st.session_state["results"] = {}
        st.rerun()


# -------------------------------------------------------------------- main
st.markdown(
    """
<div class="masthead">
  <h1>Weather Insight Engine</h1>
  <span class="tagline">is this normal? &mdash; readings measured against
    a ten-year baseline</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_chat, tab_map, tab_dash, tab_about = st.tabs(
    ["Ask", "Map", "Dashboard", "How it works"]
)


# ------------------------------------------------------------------- chat
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "results" not in st.session_state:
        st.session_state["results"] = {}

    if not st.session_state["messages"]:
        st.markdown(
            "<span class='eyebrow'>Start with</span>", unsafe_allow_html=True
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        # The container key scopes the prompt-chip styling in theme.py, so
        # these read as suggestions rather than a row of submit buttons.
        with st.container(key="samples"):
            for row_start in range(0, len(SAMPLES), 3):
                for column, (label, sample) in zip(
                    st.columns(3), SAMPLES[row_start:row_start + 3]
                ):
                    if column.button(label, width="stretch", key=f"s_{label}",
                                     help=sample):
                        st.session_state["pending_text"] = sample
                        st.rerun()

    # The transcript scrolls inside a fixed-height box so the input stays put.
    # Rendered inline, st.chat_input sits *after* the messages and is pushed
    # further down with every exchange - and because the page inside a tab
    # does not grow a scrollbar, it became unreachable after the first
    # answer. That read as "you cannot ask a second question".
    transcript = (
        st.container(height=460) if st.session_state["messages"]
        else st.container()
    )

    for index, message in enumerate(st.session_state["messages"]):
        with transcript.chat_message(message["role"]):
            st.markdown(message["content"])
            result = st.session_state["results"].get(index)
            if result is not None:
                insight = result.insight
                if insight:
                    style, note = CONFIDENCE_PILL.get(
                        insight.confidence, ("pill-amber", "")
                    )
                    # Name the place actually used, not the one requested -
                    # they differ when a guessed name could not be geocoded,
                    # and quietly answering about somewhere else is worse
                    # than saying so.
                    where = (result.bundle.location.label
                             if result.bundle else
                             (result.plan.location if result.plan else ""))
                    bits = [
                        f"<span class='pill {style}'>"
                        f"{insight.confidence.title()} confidence</span>"
                    ]
                    if where:
                        bits.append(f"<span class='meta'>📍 {where}</span>")
                    if result.plan:
                        bits.append(
                            f"<span class='meta'>🎯 {result.plan.intent}</span>"
                        )
                    bits.append(f"<span class='meta'>{note}</span>")
                    st.markdown(" ".join(bits), unsafe_allow_html=True)

                    if not settings.llm_enabled:
                        st.caption(
                            "Template mode - add a free GROQ_API_KEY to .env "
                            "for conversational answers."
                        )
                with st.expander("🔍 Show the working - data, sources, agents"):
                    render_evidence(result, key_prefix=f"m{index}")

    # A sample-button click parks its text in session state and reruns; the
    # chat box itself takes priority when the user has typed something.
    question = st.chat_input("Ask about the weather anywhere...")
    if not question:
        question = st.session_state.pop("pending_text", None)

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with transcript.chat_message("user"):
            st.markdown(question)

        with transcript.chat_message("assistant"):
            with st.spinner("Planning, fetching, analysing..."):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["messages"][:-1]
                ]
                # The sidebar selection is the fallback when the question
                # names no place, or names one the geocoder cannot find.
                reply, result = orchestrator().chat(
                    question, history,
                    default_location=active_location().name,
                )
            st.markdown(reply)

        st.session_state["messages"].append(
            {"role": "assistant", "content": reply}
        )
        st.session_state["results"][len(st.session_state["messages"]) - 1] = result
        st.rerun()


# -------------------------------------------------------------------- map
with tab_map:
    place = active_location()
    top, controls = st.columns([3, 1])
    top.markdown(
        f"<span class='eyebrow'>Conditions around</span>"
        f"<div class='hero-place' style='font-size:1.25rem;margin-top:3px'>"
        f"{place.label}</div>",
        unsafe_allow_html=True,
    )
    count = controls.slider("Nearby stations", 4, 12, 8, key="map_count")

    with st.spinner("Fetching conditions nearby..."):
        points = cached_map_points(
            place.latitude, place.longitude, place.name, count
        )

    map_col, side_col = st.columns([3, 1])

    with map_col:
        st_folium(
            build_map(place, points, zoom=zoom_for(points)),
            height=520, width=None,
            returned_objects=[],       # no reruns on pan/zoom
        )

    with side_col:
        st.markdown("**Nearby now**")
        # Warmest first: on a weather map the extremes are what people look
        # for, and alphabetical order buries them.
        ordered = sorted(
            points,
            key=lambda pair: pair[1].get("temperature_2m") or -999,
            reverse=True,
        )
        for loc, current in ordered:
            temp = current.get("temperature_2m")
            colour, _ = temperature_style(temp)
            rain = current.get("precipitation") or 0
            st.markdown(
                f"<div class='nearby-row'>"
                f"<span class='nearby-temp' "
                f"style='color:{theme.temp_colour(temp)}'>{temp}°</span>"
                f"<span class='legend-dot' style='background:"
                f"{SWATCH.get(colour, theme.MUTED)}'></span>"
                f"<b>{loc.name}</b>"
                f"<div style='font-size:.74rem;color:{theme.MUTED};"
                f"margin-left:17px'>{current.get('condition', '')}"
                + (f" · {rain} mm" if rain else "")
                + "</div></div>",
                unsafe_allow_html=True,
            )

    with st.expander("Legend"):
        for colour, label, span in legend_items():
            st.markdown(
                f"<span class='legend-dot' style='background:"
                f"{SWATCH.get(colour, '#888')}'></span> {label} "
                f"<small>({span})</small>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Blue circles show current rainfall; a bigger circle means more "
            "rain. Map tiles: OpenStreetMap. Weather: Open-Meteo. Both free."
        )


# -------------------------------------------------------------- dashboard
with tab_dash:
    place = active_location()

    try:
        current, rich, hourly, plain = cached_conditions(
            place.latitude, place.longitude, place.name
        )
    except WeatherIQError as exc:
        st.error(f"Could not fetch conditions: {exc}")
        current, rich, hourly, plain = {}, None, None, None

    if current:
        with st.spinner("Loading baseline..."):
            departure = cached_departure(
                place.latitude, place.longitude, place.name
            )
        dash.render_hero(place, current, rich, departure)

    view_today, view_hourly, view_days, view_trend = st.tabs(
        ["Today", "Next 48 hours", "10 days", "Is this normal?"]
    )

    with view_today:
        main, rail = st.columns([2, 1])
        with main:
            dash.render_today_summary(rich)
            if hourly is not None and not hourly.empty:
                st.markdown("**Through the day**")
                dash.hourly_chart(hourly.head(24), key="dash_today_hourly")
        with rail:
            dash.render_current_details(current, rich)
            dash.render_outlook(rich)

    with view_hourly:
        dash.hourly_chart(hourly, key="dash_hourly_full")
        if hourly is not None and not hourly.empty:
            table = hourly.reset_index().rename(columns={
                "time": "Time", "temperature_2m": "Temp °C",
                "apparent_temperature": "Feels °C",
                "relative_humidity_2m": "Humidity %",
                "precipitation": "Rain mm",
                "precipitation_probability": "Rain chance %",
                "wind_speed_10m": "Wind km/h", "condition": "Condition",
            })
            st.dataframe(
                table.drop(columns=["weather_code"], errors="ignore"),
                width="stretch", hide_index=True, height=300,
                key="dash_hourly_table",
            )

    with view_days:
        dash.day_cards(rich)
        st.markdown("")
        dash.range_chart(rich, key="dash_range")

    with view_trend:
        # The point of difference: not what the weather is, but whether it is
        # unusual. Same agent pipeline the chat tab uses.
        st.caption(
            "Runs the full agent pipeline on this location - fetches a "
            "10-year baseline, then reports how the current period compares."
        )
        if st.button("Analyse this location", type="primary",
                     key="dash_analyse"):
            with st.spinner("Fetching baseline and analysing..."):
                st.session_state["dash_result"] = orchestrator().answer(
                    f"Is the weather in {place.name} unusual right now?",
                    default_location=place.name,
                )

        result = st.session_state.get("dash_result")
        if result is not None:
            insight = result.insight
            st.markdown(f"**{insight.answer}**")
            for item in insight.key_findings:
                st.markdown(f"- {item}")
            st.markdown("")
            render_evidence(result, key_prefix="dashtrend")


# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(
        "<span class='eyebrow'>The claim</span>"
        "<div style='font-family:\"IBM Plex Sans Condensed\",sans-serif;"
        "font-size:1.5rem;font-weight:700;margin:6px 0 4px;max-width:640px;"
        "line-height:1.3'>Every weather app tells you it is 31 degrees. "
        "This one tells you whether 31 is unusual here.</div>"
        "<div style='color:" + theme.MUTED + ";max-width:640px;"
        "font-size:.92rem;line-height:1.6'>Seven agents plan the query, fetch "
        "only the data that question needs, analyse it against a ten-year "
        "climatological baseline, retrieve the relevant meteorology, and "
        "explain the result in plain language.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)

    # ---- the pipeline, as a row of stages ----
    st.markdown("<span class='eyebrow'>Pipeline</span>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    STAGES = [
        ("Planner", "question → structured plan",
         "Decides intent, place, date range and variables. Nothing "
         "downstream reads the raw question."),
        ("Data", "fetches only what the plan needs",
         "Current, forecast or archive back to 1940. A definition question "
         "makes zero network calls."),
        ("Analysis", "statistics, trends, comparison",
         "Trend claims are gated on R² — a slope without a fit is noise."),
        ("Anomaly", "departure from the normal",
         "Z-score against the same calendar window pooled across ten years."),
        ("Retrieval", "the R in RAG",
         "Query expanded with what the analysis found, not the question "
         "alone."),
        ("Chat / Insight", "the G in RAG",
         "Given the numbers and the retrieved text, and told to use nothing "
         "else."),
    ]
    for start in range(0, len(STAGES), 3):
        for column, (name, role, detail) in zip(
            st.columns(3), STAGES[start:start + 3]
        ):
            column.markdown(
                f"<div class='card' style='height:150px'>"
                f"<div class='card-head'>{name}</div>"
                f"<div style='font-size:.82rem;color:{theme.ACCENT};"
                f"margin-bottom:7px'>{role}</div>"
                f"<div style='font-size:.82rem;color:{theme.MUTED};"
                f"line-height:1.55'>{detail}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown(
            "<span class='eyebrow'>Measured, not claimed</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        # Numbers from eval/results/report.md. The held-out split is quoted
        # because the fitted one flatters the rule-based planner.
        st.markdown(
            "<div class='card'>"
            "<div class='metric-row'><span>planner · rule-based, held-out"
            "</span>"
            f"<b style='color:{theme.WARM}'>33.3%</b></div>"
            "<div class='metric-row'><span>planner · LLM, held-out</span>"
            f"<b style='color:{theme.ACCENT}'>91.7%</b></div>"
            "<div class='metric-row'><span>retrieval recall@4 · hybrid</span>"
            "<b>81.0%</b></div>"
            "<div class='metric-row'><span>groundedness</span>"
            "<b>100%</b></div>"
            "<div class='metric-row'><span>labelled questions</span>"
            "<b>46</b></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Both planners scored on the same held-out questions, before any "
            "defaults are applied. The rule-based planner reaches 100% on the "
            "set it was debugged against and collapses on unseen phrasing — "
            "that gap is what an LLM planner buys. Groundedness is a single "
            "run of a non-deterministic system, so treat it as a check that "
            "passed, not a guarantee."
        )

    with right:
        st.markdown(
            "<span class='eyebrow'>Nothing here is paid</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card'>"
            "<div class='metric-row'><span>weather + archive</span>"
            "<b>Open-Meteo</b></div>"
            "<div class='metric-row'><span>map tiles</span>"
            "<b>OpenStreetMap</b></div>"
            "<div class='metric-row'><span>location</span>"
            "<b>ip-api.com</b></div>"
            "<div class='metric-row'><span>language model</span>"
            "<b>Groq free tier</b></div>"
            "<div class='metric-row'><span>embeddings</span>"
            "<b>local</b></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "No credit card anywhere. Only the language model needs a key, "
            "and its free tier is rate-limited rather than billed."
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    with st.expander("Nothing in this project is trained"):
        st.markdown(
            "There is no model training and no dataset being learned.\n\n"
            "- **MiniLM** and **Llama** are pre-trained by others and used "
            "as-is.\n"
            "- `python -m src.cli index` converts text to vectors and saves "
            "them. No weights change; run it twice and get identical "
            "output.\n\n"
            "The meteorology lives in five markdown files that are "
            "**retrieved at question time** and pasted into the prompt. Edit "
            "a file, re-index, and the answer changes in seconds — which is "
            "the entire point of retrieval-augmented generation."
        )
