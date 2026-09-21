"""Visual theme.

Design direction, so later edits stay coherent:

**The thesis is departure, not temperature.** Every weather app shows you
31 degrees. This one exists to say whether 31 is unusual here, today. So the
interface is built around a departure scale - a track with the
climatological normal at its centre - and everything else stays quiet.

**Palette comes from synoptic charts, not dashboards.** Deep slate ground,
hairline rules, and a cold-to-hot ramp that is simultaneously the colour
system *and* the data encoding. One chrome accent (instrument teal) sits
deliberately off that ramp, so interface furniture can never be mistaken for
a reading.

**Numbers are set in mono.** Weather data is instrument output, so every
numeral - readings, coordinates, timestamps, axis labels - is IBM Plex Mono
with tabular figures. Plex Sans Condensed carries headings, Plex Sans the
prose. This is the one place the design spends its boldness.

**No gradients, small radii, no pillowy cards.** Instruments are flat.
"""
from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

INK = "#0C1119"        # page
PANEL = "#141B26"      # card
PANEL_2 = "#1B2431"    # raised
LINE = "#24303F"       # hairline
TEXT = "#E4EBF3"
MUTED = "#7C8CA3"

# The anomaly ramp. Doubles as the palette, because on this page colour
# always means temperature.
COLD = "#4A8FD4"
COOL = "#79A9CE"
NORM = "#7C8CA3"
WARM = "#E0913F"
HOT = "#D35450"
EXTREME = "#B33E3A"

# Chrome only - deliberately off the temperature ramp.
ACCENT = "#35C4B5"
RAIN = "#5B9BD5"

TEMP_RAMP = [
    (40, EXTREME), (35, HOT), (30, WARM),
    (25, "#C9A227"), (20, "#6FA86B"), (15, COOL),
    (5, COLD), (-100, "#3A6FA8"),
]


def temp_colour(value) -> str:
    """Colour for a temperature, from the shared ramp."""
    if value is None:
        return NORM
    try:
        value = float(value)
    except (TypeError, ValueError):
        return NORM
    if value != value:          # NaN
        return NORM
    for threshold, colour in TEMP_RAMP:
        if value >= threshold:
            return colour
    return NORM


def departure_colour(delta) -> str:
    """Colour for a departure from normal, in degrees."""
    if delta is None:
        return NORM
    if delta >= 4:
        return EXTREME
    if delta >= 2:
        return HOT
    if delta >= 0.75:
        return WARM
    if delta <= -4:
        return "#3A6FA8"
    if delta <= -2:
        return COLD
    if delta <= -0.75:
        return COOL
    return NORM


# --------------------------------------------------------------------------
# Plotly
# --------------------------------------------------------------------------

def style_figure(fig, height: int = 340):
    """Apply the theme to a Plotly figure.

    Charts default to Plotly's own palette, which fights everything else on
    the page. Routing every figure through here keeps colour meaning
    consistent: warm is warm, rain is rain.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, system-ui", size=12, color=MUTED),
        hoverlabel=dict(
            bgcolor=PANEL_2, bordercolor=LINE,
            font=dict(family="IBM Plex Mono, monospace", size=12, color=TEXT),
        ),
        legend=dict(
            orientation="h", y=1.14, x=0,
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
            tickfont=dict(family="IBM Plex Mono, monospace", size=11),
        ),
        yaxis=dict(
            gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
            tickfont=dict(family="IBM Plex Mono, monospace", size=11),
        ),
    )
    # Secondary axes are created after layout, so style them if present.
    if "yaxis2" in fig.layout:
        fig.layout.yaxis2.update(
            gridcolor=LINE, linecolor=LINE, showgrid=False,
            tickfont=dict(family="IBM Plex Mono, monospace", size=11),
        )
    return fig


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {{
  --ink:{INK}; --panel:{PANEL}; --panel2:{PANEL_2}; --line:{LINE};
  --text:{TEXT}; --muted:{MUTED}; --accent:{ACCENT}; --rain:{RAIN};
  --cold:{COLD}; --warm:{WARM}; --hot:{HOT};
}}

/* ---- ground ---- */
.stApp, [data-testid="stAppViewContainer"] {{ background:var(--ink); }}
html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, li, span, div {{
  font-family:'IBM Plex Sans', system-ui, sans-serif;
  color:var(--text);
}}
.block-container {{ padding:1.4rem 2rem 3rem; max-width:1340px; }}

h1, h2, h3, h4 {{
  font-family:'IBM Plex Sans Condensed', system-ui, sans-serif;
  font-weight:700; letter-spacing:-.01em; color:var(--text);
}}

/* Tabular mono figures everywhere a number is a reading. */
.num, .hero-temp, .metric-row b, .day-hi, .day-lo, .dep-value b,
.stMetric [data-testid="stMetricValue"] {{
  font-family:'IBM Plex Mono', monospace;
  font-variant-numeric:tabular-nums; font-feature-settings:"tnum";
}}

/* ---- masthead ---- */
.masthead {{
  display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
  padding-bottom:14px; margin-bottom:10px;
}}
.masthead h1 {{ font-size:1.5rem; margin:0; white-space:nowrap; }}
.masthead .tagline {{
  font-size:.82rem; color:var(--muted);
  /* Wraps rather than clipping when the pane is narrow. */
  min-width:0; flex:1 1 220px;
}}
/* Below tablet the tagline is the first thing to go - the product name and
   the navigation matter more than the strapline. */
@media (max-width:700px) {{ .masthead .tagline {{ display:none; }} }}

.eyebrow {{
  font-family:'IBM Plex Mono', monospace;
  font-size:.66rem; font-weight:500; letter-spacing:.16em;
  text-transform:uppercase; color:var(--muted);
}}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{
  background:var(--panel); border-right:1px solid var(--line);
}}
[data-testid="stSidebar"] .block-container {{ padding-top:1.4rem; }}

/* ---- primary navigation ----
   A contained segmented control, not a row of underlined words. Navigation
   should read as a physical control you operate, and the enclosure is what
   makes it look built rather than left at its defaults.

   Selector note: Streamlit 1.64 dropped the `data-baseweb="tab"` attributes
   these rules used to target, so everything is keyed off ARIA roles, which
   are part of the accessibility contract and far less likely to churn. The
   legacy attributes are kept alongside for older Streamlit versions. */
.stTabs [role="tablist"],
.stTabs [data-baseweb="tab-list"] {{
  gap:3px; background:var(--panel); border:1px solid var(--line);
  border-radius:9px; padding:4px; margin-bottom:22px;
  width:fit-content; max-width:100%;
}}
.stTabs [role="tab"],
.stTabs [data-baseweb="tab"] {{
  height:36px; padding:0 20px; border-radius:6px; border-bottom:none;
  background:transparent;
  display:flex; align-items:center; justify-content:center;
  font-family:'IBM Plex Sans Condensed', sans-serif;
  font-weight:600; font-size:.92rem; letter-spacing:.015em;
  color:var(--muted); cursor:pointer;
  transition:background .14s ease, color .14s ease;
}}
.stTabs [role="tab"] p, .stTabs [data-baseweb="tab"] p {{
  font-family:inherit; font-weight:inherit; font-size:inherit;
  color:inherit; margin:0;
}}
.stTabs [role="tab"]:hover, .stTabs [data-baseweb="tab"]:hover {{
  color:var(--text); background:rgba(127,127,127,.09);
}}
.stTabs [role="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
  background:var(--panel2); color:var(--accent);
  box-shadow:inset 0 0 0 1px rgba(53,196,181,.30);
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ background:transparent; display:none; }}
.stTabs [role="tabpanel"] {{ padding-top:2px; }}

/* ---- secondary navigation ----
   The dashboard's time ranges sit inside a panel that already has its own
   enclosure, so a second box would nest two frames. Underlines instead:
   clearly subordinate to the control above. */
.stTabs .stTabs [role="tablist"],
.stTabs .stTabs [data-baseweb="tab-list"] {{
  background:transparent; border:none; border-bottom:1px solid var(--line);
  border-radius:0; padding:0; gap:2px; margin-bottom:16px; width:100%;
}}
.stTabs .stTabs [role="tab"],
.stTabs .stTabs [data-baseweb="tab"] {{
  height:32px; padding:0 14px; border-radius:0;
  border-bottom:2px solid transparent; background:transparent;
  font-size:.83rem; font-weight:500; letter-spacing:.01em;
}}
.stTabs .stTabs [role="tab"]:hover,
.stTabs .stTabs [data-baseweb="tab"]:hover {{
  background:transparent; color:var(--text);
}}
.stTabs .stTabs [role="tab"][aria-selected="true"],
.stTabs .stTabs [data-baseweb="tab"][aria-selected="true"] {{
  background:transparent; color:var(--accent);
  border-bottom:2px solid var(--accent); box-shadow:none;
}}

/* ---- panels ---- */
.card {{
  background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:14px 16px; margin-bottom:12px;
}}
.card-head {{
  font-family:'IBM Plex Mono', monospace;
  font-size:.66rem; letter-spacing:.16em; text-transform:uppercase;
  color:var(--muted); margin-bottom:10px;
  display:flex; justify-content:space-between; align-items:baseline;
}}
.card-date {{ letter-spacing:.04em; }}

/* ---- hero: flat, left-aligned, no gradient ---- */
.hero {{
  background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:18px 20px; margin-bottom:12px;
}}
.hero-top {{
  display:flex; justify-content:space-between; align-items:flex-start;
  flex-wrap:wrap; gap:16px;
}}
.hero-read {{ display:flex; align-items:center; gap:16px; }}
.hero-icon {{ font-size:2.6rem; line-height:1; }}
.hero-temp {{ font-size:3rem; font-weight:500; line-height:1; letter-spacing:-.03em; }}
.hero-unit {{ font-size:1.1rem; color:var(--muted); margin-left:2px; }}
.hero-cond {{ font-size:.86rem; color:var(--muted); margin-top:5px; }}
.hero-meta {{ text-align:right; }}
.hero-place {{
  font-family:'IBM Plex Sans Condensed', sans-serif;
  font-size:1.05rem; font-weight:600;
}}
.hero-coord {{
  font-family:'IBM Plex Mono', monospace;
  font-size:.7rem; color:var(--muted); margin-top:3px;
}}
.hero-hilo {{ font-size:.82rem; color:var(--muted); margin-top:6px; }}

/* ---- signature: the departure scale ---- */
.dep {{ margin-top:16px; border-top:1px solid var(--line); padding-top:13px; }}
.dep-head {{
  display:flex; justify-content:space-between; align-items:baseline;
  margin-bottom:11px;
}}
.dep-note {{
  font-family:'IBM Plex Mono', monospace;
  font-size:.64rem; color:var(--muted);
}}
.dep-track {{
  position:relative; height:12px; border-radius:2px;
  background:linear-gradient(90deg,
    #3A6FA8 0%, {COLD} 18%, {COOL} 34%,
    {NORM}44 50%, {WARM} 66%, {HOT} 82%, {EXTREME} 100%);
  opacity:.9;
}}
.dep-band {{
  position:absolute; top:-3px; bottom:-3px;
  border-left:1px dashed rgba(228,235,243,.35);
  border-right:1px dashed rgba(228,235,243,.35);
}}
.dep-zero {{
  position:absolute; left:50%; top:-5px; bottom:-5px; width:1px;
  background:var(--text);
}}
.dep-mark {{
  position:absolute; top:-8px; width:4px; height:28px; border-radius:2px;
  background:var(--text);
  box-shadow:0 0 0 2px var(--panel), 0 0 10px rgba(255,255,255,.35);
}}
/* Caret above the marker so the eye lands on it before the ramp. */
.dep-mark::after {{
  content:''; position:absolute; left:50%; top:-7px; transform:translateX(-50%);
  border-left:5px solid transparent; border-right:5px solid transparent;
  border-top:6px solid var(--text);
}}
.dep-scale {{
  display:flex; justify-content:space-between; margin-top:7px;
  font-family:'IBM Plex Mono', monospace;
  font-size:.64rem; color:var(--muted);
}}
.dep-value {{ font-size:.88rem; margin-top:9px; }}
.dep-value b {{ font-weight:600; }}

/* ---- metric rows ---- */
.metric-row {{
  display:flex; justify-content:space-between; align-items:baseline;
  padding:7px 0; border-bottom:1px solid var(--line); font-size:.86rem;
}}
.metric-row:last-child {{ border-bottom:none; }}
.metric-row span {{ color:var(--muted); }}
.metric-row b {{ font-weight:500; font-size:.92rem; }}

.summary-row {{
  display:flex; gap:11px; align-items:flex-start;
  padding:6px 0; font-size:.9rem; line-height:1.5;
}}
.summary-icon {{ font-size:1.15rem; line-height:1.3; }}
.outlook {{ font-size:.88rem; color:var(--text); line-height:1.55; }}

/* ---- day strip ---- */
.day-card {{
  text-align:center; padding:11px 4px; border-radius:6px;
  background:var(--panel); border:1px solid var(--line);
}}
.day-name {{
  font-family:'IBM Plex Sans Condensed', sans-serif;
  font-weight:700; font-size:.82rem; letter-spacing:.02em;
}}
.day-date {{
  font-family:'IBM Plex Mono', monospace;
  font-size:.64rem; color:var(--muted); margin-bottom:5px;
}}
.day-icon {{ font-size:1.5rem; line-height:1.4; }}
.day-hi {{ font-size:1.1rem; font-weight:500; }}
.day-lo {{ font-size:.8rem; color:var(--muted); }}
.day-rain {{
  font-family:'IBM Plex Mono', monospace;
  font-size:.66rem; color:var(--rain); margin-top:4px;
}}

/* ---- pills & meta ---- */
.pill {{
  display:inline-block; padding:2px 9px; border-radius:3px;
  font-family:'IBM Plex Mono', monospace;
  font-size:.66rem; font-weight:500; letter-spacing:.05em;
  text-transform:uppercase; margin:2px 5px 2px 0;
}}
.pill-green {{ background:#2e7d3225; color:#5cc46a; border:1px solid #5cc46a40; }}
.pill-amber {{ background:#e6510025; color:{WARM}; border:1px solid {WARM}40; }}
.pill-red   {{ background:#c6282825; color:{HOT}; border:1px solid {HOT}40; }}
.meta {{
  display:inline-block; font-family:'IBM Plex Mono', monospace;
  font-size:.68rem; color:var(--muted); margin:2px 9px 2px 0;
}}

/* ---- chat ---- */
.stChatMessage {{
  background:var(--panel); border:1px solid var(--line);
  border-radius:6px; padding:12px 14px;
}}
[data-testid="stChatInput"] {{ border-color:var(--line); }}

/* ---- sources ---- */
.src-card {{
  border-left:2px solid var(--accent); padding:7px 12px; margin:8px 0;
  background:var(--panel2); border-radius:0 4px 4px 0;
}}
.src-card small {{ color:var(--muted); }}

/* ---- nearby list ---- */
.nearby-row {{
  padding:7px 10px; margin-bottom:5px; border-radius:4px;
  background:var(--panel); border:1px solid var(--line);
  font-size:.84rem;
}}
.nearby-temp {{
  font-family:'IBM Plex Mono', monospace; float:right; font-weight:500;
}}
.legend-dot {{
  display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:8px; vertical-align:middle;
}}

/* ---- buttons ---- */
.stButton button {{
  border-radius:4px; border:1px solid var(--line);
  background:var(--panel2); color:var(--text);
  font-family:'IBM Plex Sans', sans-serif; font-size:.84rem; font-weight:500;
  padding:.42rem .9rem; min-height:36px;
  transition:border-color .12s ease, background .12s ease, color .12s ease;
}}
.stButton button:hover {{
  border-color:var(--accent); color:var(--accent);
  background:var(--panel);
}}
.stButton button:active {{ transform:translateY(1px); }}
.stButton button[kind="primary"] {{
  background:var(--accent); border-color:var(--accent); color:{INK};
  font-weight:600;
}}
.stButton button[kind="primary"]:hover {{
  background:#2FB0A3; border-color:#2FB0A3; color:{INK};
}}

/* Sample prompts are suggestions, not commands: left-aligned, lighter, and
   they lift on hover rather than looking like a row of submit buttons.
   Two notes: a `help=` tooltip wraps the button in a span, so these use
   descendant selectors; and Streamlit's own rules win on specificity for
   background and alignment, so those two are forced. */
.st-key-samples .stButton button {{
  background:transparent !important;
  justify-content:flex-start !important;
  border:1px solid var(--line);
  color:var(--muted); font-weight:400; min-height:44px;
  padding-left:14px;
}}
.st-key-samples .stButton button > div {{
  width:100%; justify-content:flex-start !important; text-align:left;
}}
.st-key-samples .stButton button:hover {{
  background:var(--panel) !important; color:var(--text);
  border-color:var(--accent);
}}
.st-key-samples .stButton button p {{
  font-size:.85rem; text-align:left; width:100%;
}}

/* ---- inputs ---- */
.stTextInput input, .stTextArea textarea {{
  background:var(--ink); border:1px solid var(--line); border-radius:4px;
  color:var(--text); font-family:'IBM Plex Sans', sans-serif;
  font-size:.86rem;
}}
.stTextInput input::placeholder {{ color:var(--muted); opacity:.75; }}
.stTextInput input:focus, .stTextArea textarea:focus {{
  border-color:var(--accent); box-shadow:none;
}}
[data-testid="stChatInput"] {{
  background:var(--panel); border:1px solid var(--line); border-radius:6px;
}}
[data-testid="stChatInput"]:focus-within {{ border-color:var(--accent); }}
[data-testid="stChatInput"] textarea {{ font-size:.9rem; }}

/* ---- slider ---- */
.stSlider [data-baseweb="slider"] [role="slider"] {{
  border:2px solid var(--accent); background:var(--ink);
}}
.stSlider [data-testid="stTickBar"] {{ display:none; }}
.stSlider label {{ font-size:.78rem; color:var(--muted); }}

/* ---- expanders ---- */
[data-testid="stExpander"] {{
  border:1px solid var(--line); border-radius:6px; background:var(--panel);
}}
[data-testid="stExpander"] summary {{
  font-family:'IBM Plex Sans Condensed', sans-serif;
  font-weight:600; font-size:.88rem; color:var(--muted);
  padding:9px 14px;
}}
[data-testid="stExpander"] summary:hover {{ color:var(--accent); }}

/* ---- tables ---- */
[data-testid="stDataFrame"] {{
  border:1px solid var(--line); border-radius:6px; overflow:hidden;
}}

/* ---- misc chrome ---- */
[data-testid="stSpinner"] > div {{ border-top-color:var(--accent); }}
hr, [data-testid="stDivider"] {{ border-color:var(--line); }}
[data-testid="stCaptionContainer"], .stCaption {{
  color:var(--muted); font-size:.78rem;
}}
[data-testid="stAlert"] {{
  border-radius:5px; border:1px solid var(--line); background:var(--panel);
}}
/* Streamlit's deploy button is noise in a demo. */
[data-testid="stToolbar"] {{ opacity:.25; }}
[data-testid="stToolbar"]:hover {{ opacity:1; }}

/* ---- accessibility floor ---- */
*:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
@media (prefers-reduced-motion:reduce) {{
  * {{ animation:none !important; transition:none !important; }}
}}
@media (max-width:820px) {{
  .hero-meta {{ text-align:left; }}
  .hero-temp {{ font-size:2.4rem; }}
  .block-container {{ padding:1rem 1rem 2rem; }}
}}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
