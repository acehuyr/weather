"""Streamlit interface.

    streamlit run src/ui/app.py

The app serves two people at once: someone who wants to know about the
weather, and an examiner who wants to see how the answer was produced. The
first one comes first. Nothing on the way to an answer mentions agents,
retrieval or embeddings; all of that lives one deliberate click away.

    Now            conditions, and whether they are unusual here
    Map            live weather on and around a place
    Ask            conversational Q&A, evidence attached to each answer
    How it works   architecture, measured results, system status

One level of navigation. The previous version nested four sub-tabs inside
the dashboard tab, which at phone width collapsed into two rows of
horizontal scrollers and hid half the app. Sections stack on one scroll
instead.

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
    # There is no sidebar any more. The location control used to live in it,
    # which meant that on a phone - where Streamlit collapses the sidebar by
    # default - the app's primary control was hidden behind a chevron.
    initial_sidebar_state="collapsed",
)

# All visual design lives in src/ui/theme.py - one place for tokens,
# typography and the Plotly styling, so charts and chrome cannot drift
# apart.
theme.inject()

# Label and payload are the same string. They used to differ - a chip read
# "vs previous years" and sent a full sentence that appeared only in a hover
# tooltip, which is invisible on touch - so you could not tell what you had
# just asked.
SAMPLES = [
    "What's the weather in Mumbai right now?",
    "Will it rain in Pune this week?",
    "How has Delhi's temperature changed vs previous years?",
    "Why is the weather in Chennai unusual?",
    "What counts as a heat wave in India?",
    "Why does humid heat feel worse than dry heat?",
]

CONFIDENCE = {
    "high": ("pill-green", "Well supported by the data"),
    "medium": ("pill-amber", "Supported, with gaps in the data"),
    "low": ("pill-red", "Weakly supported — indicative only"),
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
    """Everything the Now page needs, in one cached call."""
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


def remember(place) -> None:
    """Keep the last few places, most recent first, for one-tap return."""
    recents = [p for p in st.session_state.get("recents", [])
               if p.label != place.label]
    st.session_state["recents"] = ([place] + recents)[:4]


# ------------------------------------------------------------ small pieces
def section(title: str, note: str = "") -> None:
    """A labelled rule. This is what replaced the second row of tabs."""
    st.markdown(
        f"<div class='section'><h2>{title}</h2>"
        f"<span class='rule'></span>"
        + (f"<span class='note'>{note}</span>" if note else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def render_place_bar() -> None:
    """The location control, on every page and at every width.

    A form, so Enter submits. The old control was a text field plus a
    separate **Set** button where Enter did nothing at all.
    """
    place = active_location()

    st.markdown(
        f"<div class='placebar'><span class='pin'>◉</span>"
        f"<span class='where'>{dash.pretty_label(place)}</span>"
        f"<span class='coord'>{place.latitude:.3f}, "
        f"{place.longitude:.3f}</span></div>",
        unsafe_allow_html=True,
    )

    with st.form("place_form", clear_on_submit=True, border=False):
        field, go_btn, here_btn = st.columns([6, 1.1, 1.3])
        typed = field.text_input(
            "Change location", value="",
            placeholder="Search any place — Mumbai, Nagaur, Tokyo…",
            label_visibility="collapsed",
        )
        submitted = go_btn.form_submit_button("Go", width="stretch")
        near = here_btn.form_submit_button("Near me", width="stretch")

    if submitted and typed.strip():
        try:
            found = cached_geocode(typed)
            st.session_state["location"] = found
            remember(found)
            st.rerun()
        except LocationNotFound:
            st.warning(
                f"Could not find “{typed}”. Try adding the country — "
                f"“{typed}, India”."
            )
    elif near:
        try:
            found = detect_location_from_ip()
            st.session_state["location"] = found
            remember(found)
            st.rerun()
        except WeatherIQError as exc:
            st.warning(f"{exc} Search for a place instead.")

    recents = st.session_state.get("recents", [])
    if len(recents) > 1:
        chips = st.columns(len(recents))
        for column, past in zip(chips, recents):
            if column.button(past.name, key=f"recent_{past.label}",
                             width="stretch"):
                st.session_state["location"] = past
                st.rerun()


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


# ---------------------------------------------------------------- evidence
def render_plan(plan) -> None:
    """The planner's decision as a sentence, not a dict dump."""
    if plan is None:
        return
    window = (f"{plan.start_date} to {plan.end_date}"
              if plan.start_date and plan.end_date else "a default window")
    st.markdown(
        f"<div class='src-body'>Read as a "
        f"<b style='color:{theme.TEXT}'>{plan.intent}</b> question about "
        f"<b style='color:{theme.TEXT}'>{plan.location or 'the selected place'}</b>, "
        f"over {window}, needing "
        f"{', '.join(plan.variables) if plan.variables else 'no specific variables'}."
        f"<br><i>{plan.rationale}</i></div>",
        unsafe_allow_html=True,
    )


def render_timeline(trace) -> None:
    """The pipeline as a readable sequence.

    This used to be `st.dataframe(trace)`. A table is the least persuasive
    way to show an examiner that seven agents ran and where the time went.
    """
    rows = []
    for step in trace:
        colour = theme.ACCENT if step.ok else theme.HOT
        note = step.note or step.error or ("ok" if step.ok else "failed")
        rows.append(
            f"<div class='stage'>"
            f"<span class='stage-dot' style='background:{colour}'></span>"
            f"<span class='stage-name'>{step.agent}</span>"
            f"<span class='stage-note'>{note}</span>"
            f"<span class='stage-ms'>{step.duration_ms:.0f} ms</span></div>"
        )
    total = sum(s.duration_ms for s in trace)
    st.markdown(
        "<div class='card'>" + "".join(rows)
        + f"<div class='stage' style='border-top:1px solid var(--line)'>"
        f"<span class='stage-dot'></span>"
        f"<span class='stage-name'>total</span>"
        f"<span class='stage-note'></span>"
        f"<span class='stage-ms'>{total:.0f} ms</span></div></div>",
        unsafe_allow_html=True,
    )


def render_sources(documents) -> None:
    """Retrieved passages, ranked.

    The score is a reciprocal-rank-fusion value (`1 / (k + rank)`, summed
    over the lists that returned the chunk), not a similarity. It has no
    meaningful absolute scale - real values land around 0.02 - so drawing it
    as a percentage bar would say "weak match" about every result the
    retriever was confident in. Rank is what RRF actually encodes, so rank
    is what this shows; the bar is scaled against the best hit in the set.
    """
    if not documents:
        st.caption("No background knowledge was needed for this question.")
        return

    best = max((float(d.score) for d in documents), default=0.0) or 1.0
    for rank, doc in enumerate(documents, start=1):
        share = max(6.0, min(100.0, float(doc.score) / best * 100))
        place = "closest match" if rank == 1 else f"#{rank} of {len(documents)}"
        st.markdown(
            f"<div class='src-card'><b>{doc.title}</b>"
            f"<div class='src-strength'>"
            f"<i style='width:{share:.0f}%'></i></div>"
            f"<div class='src-body' style='color:var(--faint)'>"
            f"{place} · <code>{doc.source}</code> · "
            f"fusion score {doc.score:.3f}</div>"
            f"<div class='src-body'>{doc.text[:300]}…</div></div>",
            unsafe_allow_html=True,
        )


def render_evidence(result, key_prefix: str) -> None:
    """The 'show your working' panel - the examiner's surface."""
    bundle = result.bundle

    section("How this answer was reached")
    render_plan(result.plan)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    render_timeline(result.trace)

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

    section(f"Knowledge used ({len(result.documents)})")
    render_sources(result.documents)

    anomalies = result.anomalies
    if anomalies.get("anomalies"):
        section(f"Anomalies ({anomalies.get('total', 0)})",
                f"against {anomalies.get('baseline_used', '')}")
        st.dataframe(
            pd.DataFrame(anomalies["anomalies"]).head(10),
            width="stretch", hide_index=True, key=f"{key_prefix}_anom",
        )


# -------------------------------------------------------------------- page
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

render_place_bar()

tab_now, tab_map, tab_ask, tab_about = st.tabs(
    ["Now", "Map", "Ask", "How it works"]
)


# --------------------------------------------------------------------- now
with tab_now:
    place = active_location()

    try:
        current, rich, hourly, plain = cached_conditions(
            place.latitude, place.longitude, place.name
        )
    except WeatherIQError as exc:
        st.error(f"Could not fetch conditions: {exc}")
        current, rich, hourly, plain = {}, None, None, None

    if current:
        departure = cached_departure(
            place.latitude, place.longitude, place.name
        )
        # The verdict is inside the hero and renders straight from the
        # six-hour cache, with no button press. It used to sit behind
        # "Analyse this location" inside a sub-tab of a tab - three
        # interactions away from the thing the project exists to say.
        dash.render_hero(place, current, rich, departure)

        section("Today")
        main, rail = st.columns([2, 1])
        with main:
            dash.render_today_summary(rich)
            if hourly is not None and not hourly.empty:
                dash.hourly_chart(hourly.head(24), key="now_today_hourly")
        with rail:
            dash.render_current_details(current, rich)
            dash.render_outlook(rich)

        section("Next 10 days")
        dash.day_cards(rich)
        st.markdown("")
        dash.range_chart(rich, key="now_range")

        section("Next 48 hours")
        dash.hourly_chart(hourly, key="now_hourly_full")
        with st.expander("Hour by hour, as a table"):
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
                    key="now_hourly_table",
                )

        section("The full analysis", "runs all seven agents on this place")
        st.caption(
            "The headline above comes from the cached ten-year baseline. "
            "This runs the complete pipeline — fetching history, detecting "
            "anomalies and retrieving the relevant meteorology."
        )
        if st.button("Run the full analysis", type="primary",
                     key="now_analyse"):
            with st.status("Running the pipeline…", expanded=False) as status:
                st.session_state["now_result"] = orchestrator().answer(
                    f"Is the weather in {place.name} unusual right now?",
                    default_location=place.name,
                )
                status.update(label="Analysis complete", state="complete")

        result = st.session_state.get("now_result")
        if result is not None:
            insight = result.insight
            st.markdown(f"**{insight.answer}**")
            for item in insight.key_findings:
                st.markdown(f"- {item}")
            render_evidence(result, key_prefix="nowtrend")


# -------------------------------------------------------------------- map
with tab_map:
    place = active_location()
    top, controls = st.columns([3, 1])
    top.markdown(
        f"<span class='eyebrow'>Conditions around</span>"
        f"<div class='hero-place' style='font-size:1.25rem;margin-top:3px'>"
        f"{dash.pretty_label(place)}</div>",
        unsafe_allow_html=True,
    )
    count = controls.slider("Nearby stations", 4, 12, 8, key="map_count")

    with st.spinner("Fetching conditions nearby…"):
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
                f"<div style='font-size:.78rem;color:{theme.MUTED};"
                f"margin-left:17px'>{current.get('condition', '')}"
                + (f" · {rain} mm" if rain else "")
                + "</div></div>",
                unsafe_allow_html=True,
            )

    with st.expander("What the colours mean"):
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


# -------------------------------------------------------------------- ask
with tab_ask:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "results" not in st.session_state:
        st.session_state["results"] = {}

    messages = st.session_state["messages"]

    if not messages:
        st.markdown(
            "<div class='verdict'>Ask about the weather anywhere.</div>"
            "<div class='verdict-sub'>Answers are built from live readings "
            "and a ten-year archive, and every one of them shows its "
            "working.</div>",
            unsafe_allow_html=True,
        )
        section("Try one of these")
        # Pills, not panels - see the note in theme.py. The container key
        # scopes that styling.
        with st.container(key="samples"):
            for row_start in range(0, len(SAMPLES), 2):
                for column, sample in zip(
                    st.columns(2), SAMPLES[row_start:row_start + 2]
                ):
                    if column.button(sample, width="stretch",
                                     key=f"s_{sample[:24]}"):
                        st.session_state["pending_text"] = sample
                        st.rerun()
    else:
        asked = len([m for m in messages if m["role"] == "user"])
        head, clear = st.columns([5, 1])
        head.markdown(
            f"<span class='eyebrow'>Conversation · {asked} "
            f"question{'' if asked == 1 else 's'}</span>",
            unsafe_allow_html=True,
        )
        if clear.button("Clear", key="clear_chat", width="stretch"):
            st.session_state["messages"] = []
            st.session_state["results"] = {}
            st.rerun()

    # The transcript renders inline and the composer is sticky, so the input
    # stays reachable however long the thread gets. The previous version
    # pinned the transcript inside a fixed-height 460px scroller to keep the
    # input on screen, which meant your own question scrolled out of sight
    # the moment an answer arrived.
    for index, message in enumerate(messages):
        if message["role"] == "user":
            st.markdown(
                f"<div class='turn-you'><div>"
                f"<span class='turn-label'>You asked</span>"
                f"<div class='bubble'>{message['content']}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            continue

        result = st.session_state["results"].get(index)
        meta = ""
        if result is not None and result.insight:
            style, note = CONFIDENCE.get(
                result.insight.confidence, ("pill-amber", "")
            )
            # Name the place actually used, not the one requested - they
            # differ when a guessed name could not be geocoded, and quietly
            # answering about somewhere else is worse than saying so.
            where = (dash.pretty_label(result.bundle.location) if result.bundle
                     else (result.plan.location if result.plan else ""))
            bits = [f"<span class='pill {style}'>{note}</span>"]
            if where:
                bits.append(f"<span class='meta'>◉ {where}</span>")
            meta = f"<div class='answer-meta'>{''.join(bits)}</div>"

        # The answer body has to be a separate st.markdown call so its
        # markdown actually renders, so the turn is a keyed container and
        # theme.py styles it by key. Writing the wrapper as raw HTML would
        # close the div before the body was inside it.
        with st.container(key=f"answer_{index}"):
            st.markdown(
                f"<span class='turn-label'>Answer</span>{meta}",
                unsafe_allow_html=True,
            )
            st.markdown(message["content"])

            if result is not None:
                if not settings.llm_enabled:
                    st.caption(
                        "Template mode — add a free GROQ_API_KEY to .env "
                        "for conversational answers."
                    )
                with st.expander("Show the working — data, sources, agents"):
                    render_evidence(result, key_prefix=f"m{index}")

        st.markdown("<hr class='turn-rule'>", unsafe_allow_html=True)

    # A sample click parks its text in session state and reruns; the chat box
    # itself takes priority when the user has typed something.
    with st.container(key="composer"):
        st.markdown(
            "<div class='composer-hint'>Ask a question</div>",
            unsafe_allow_html=True,
        )
        question = st.chat_input(
            "Ask a follow-up, or about anywhere else…"
            if messages else "Ask about the weather anywhere…"
        )
    if not question:
        question = st.session_state.pop("pending_text", None)

    if question:
        st.session_state["messages"].append(
            {"role": "user", "content": question}
        )
        with st.status("Working on it…", expanded=True) as status:
            st.write("Planning the query…")
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"][:-1]
            ]
            st.write("Fetching readings and analysing…")
            # The header selection is the fallback when the question names no
            # place, or names one the geocoder cannot find.
            reply, result = orchestrator().chat(
                question, history,
                default_location=active_location().name,
            )
            status.update(label="Answered", state="complete", expanded=False)

        st.session_state["messages"].append(
            {"role": "assistant", "content": reply}
        )
        st.session_state["results"][len(st.session_state["messages"]) - 1] = result
        st.rerun()


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

    section("Pipeline")

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
                f"<div class='card' style='height:158px'>"
                f"<div class='card-head'>{name}</div>"
                f"<div style='font-size:.84rem;color:{theme.ACCENT};"
                f"margin-bottom:7px'>{role}</div>"
                f"<div style='font-size:.84rem;color:{theme.MUTED};"
                f"line-height:1.55'>{detail}</div></div>",
                unsafe_allow_html=True,
            )

    left, right = st.columns(2)

    with left:
        section("Measured, not claimed")
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
        section("Nothing here is paid")
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

    # System status lives here rather than in permanent chrome. Whether a
    # language model is attached is a configuration fact, not something a
    # person checking the weather needs on screen at all times.
    section("System status")
    llm_on = settings.llm_enabled
    st.markdown(
        f"<div class='card'>"
        f"<div class='metric-row'><span>language model</span>"
        f"<b style='color:{theme.ACCENT if llm_on else theme.MUTED}'>"
        f"{settings.llm_label if llm_on else 'not connected'}</b></div>"
        f"<div class='metric-row'><span>retrieval backend</span>"
        f"<b>{settings.embedding_backend}</b></div>"
        f"<div class='metric-row'><span>indexed knowledge</span>"
        f"<b>{len(get_retriever())} chunks</b></div>"
        f"<div class='metric-row'><span>weather source</span>"
        f"<b>Open-Meteo</b></div></div>",
        unsafe_allow_html=True,
    )

    if not llm_on:
        with st.expander("Turn on conversational answers"):
            st.markdown(
                "Answers currently come from a deterministic template. To get "
                "natural language and follow-up questions, add a **free** "
                "key:\n\n"
                "1. Open **console.groq.com/keys** and sign in\n"
                "2. Create a key and copy it\n"
                "3. Put it in `.env`:\n"
            )
            st.code("GROQ_API_KEY=gsk_your_key_here", language="bash")
            st.caption("No card required. Restart the app after saving.")

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
