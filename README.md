# Intelligent Weather Data Analysis and Insight Generation

Agentic AI + Retrieval-Augmented Generation over live and historical weather
data. The system does not just display weather: it **plans** the query,
**fetches** the right data, **analyses** it, **retrieves** supporting
meteorological knowledge, and **explains** the result in plain English.

---

## Everything here is free

| Component | Source | Key needed? |
|---|---|---|
| Weather data (live + 1940 archive) | Open-Meteo | **no** |
| Map tiles | OpenStreetMap | **no** |
| Location detection | ip-api.com | **no** |
| Language model | Groq free tier (open-weight) | free key, no card |
| Embeddings | hashed TF-IDF / MiniLM | **no**, runs locally |

No credit card, no trial, nothing metered to a bill.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run src/ui/app.py
```

It runs immediately with **no key at all** — in offline mode, using a
rule-based planner and template answers, with the full data pipeline, anomaly
detection, RAG and map all working.

### Turning on the chatbot (2 minutes, free)

Answers are stiff without a language model. To fix that:

1. Go to **console.groq.com/keys** and sign in (Google/GitHub, no card).
2. Create an API key and copy it.
3. Paste it into `.env`:

```bash
GROQ_API_KEY=gsk_your_key_here
```

Restart the app. The sidebar's **model** row switches from `not connected` to
the model name, and answers become conversational and follow-up aware.

**If you get a 404 saying the model does not exist**, Groq has retired that
model ID. This is *not* an authentication problem — a bad key returns 401, not
404. List what your key can actually use, and put one in `.env` as
`GROQ_MODEL`:

```bash
python scripts/list_models.py
```

### The three interfaces

```bash
streamlit run src/ui/app.py                 # Web UI: chat, map, dashboard
python -m src.cli demo                      # CLI, runs sample questions
uvicorn src.api.main:app --reload           # REST API, docs at /docs
```

The web UI has four tabs:

- **Chat** — conversational Q&A that remembers context, with a *Show the
  working* panel under every answer exposing the data, sources and agent trace
- **Map** — live weather on and around your location, colour-coded by
  temperature with rainfall circles
- **Dashboard** — hero current conditions, then four sub-views: *Today*
  (day/night summary, hourly chart, secondary metrics, outlook), *Next 48
  hours*, *10 days* (day cards + a low-to-high range chart), and **Is this
  normal?**
- **How it works** — architecture and measured results, for demos

The *Is this normal?* sub-view is the one a normal weather app doesn't have:
it runs the full agent pipeline on the displayed location, pulls a ten-year
baseline, and reports whether current conditions are actually unusual — with
the evidence panel attached.

---

## Architecture

```
                         "Why is it so humid in Mumbai?"
                                      |
                        +-------------v-------------+
                        |      PLANNER AGENT        |
                        |  NL -> structured plan    |
                        |  intent, place, dates,    |
                        |  variables, what is needed|
                        +-------------+-------------+
                                      |  QueryPlan
        +-----------------------------+------------------------------+
        |                             |                              |
+-------v--------+          +---------v---------+         +----------v---------+
|  DATA AGENT    |          |  ANALYSIS AGENT   |         |   ANOMALY AGENT    |
| Open-Meteo:    |--------->| statistics        |-------->| z-score vs. the    |
| current /      |  frames  | linear trends     | numbers | climatological     |
| forecast /     |          | year-over-year    |         | normal + IMD       |
| 1940+ archive  |          | rainfall profile  |         | thresholds         |
+----------------+          +-------------------+         +----------+---------+
                                                                     |
                                                          findings   |
                                                    +----------------v---------+
                                                    |    RETRIEVAL AGENT       |
                                                    |  RAG over the knowledge  |
                                                    |  base. Query is expanded |
                                                    |  with what was found.    |
                                                    +----------------+---------+
                                                                     | chunks
                                          +--------------------------------+
                                          |  CHAT AGENT  /  INSIGHT AGENT  |
                                          |  Same grounded facts, two      |
                                          |  shapes: conversational prose  |
                                          |  for the UI, a structured      |
                                          |  report for the API.           |
                                          +--------------------------------+
```

Seven agents in total. The Chat and Insight agents are alternatives to each
other — the orchestrator picks one depending on whether the caller wants a
conversation or a structured response.

Every agent returns the same `AgentResult` envelope, so the orchestrator stays
simple and the **execution trace is displayable** — the UI shows which agents
ran, how long each took, and what it produced. For an agentic system, "why did
it answer that?" has to be answerable.

### Why plan first?

Nothing downstream reads the raw question; every agent reads the `QueryPlan`.
That makes behaviour inspectable and keeps work proportional: "what is
humidity?" triggers zero network calls, while "why is this unusual?" pulls a
ten-year baseline.

---

## Project layout

```
config/settings.py          every tunable, one place
data/knowledge_base/*.md    the RAG corpus (edit these, then reindex)
data/cache/                 TTL'd HTTP cache
data/vectorstore/           persisted embeddings

src/core/                   models, errors, logging
src/ingestion/              Open-Meteo provider, geocoding, HTTP cache
src/analysis/               statistics, trends, anomaly detection
src/rag/                    chunker, embeddings, vector store, retriever
src/llm/                    provider abstraction (Groq / Claude) + prompts
src/agents/                 the seven agents + orchestrator
src/api/                    FastAPI service
src/ui/app.py               Streamlit interface (chat, map, dashboard)
src/ui/theme.py             design tokens, typography, Plotly styling
src/ui/dashboard.py         hero, departure scale, forecast views
src/ui/weather_map.py       Folium map over OpenStreetMap tiles
src/cli.py                  command-line interface

eval/dataset.py             46 labelled questions (dev + held-out test)
eval/metrics.py             accuracy, F1, Recall@k, MRR, groundedness
eval/results/report.md      latest measured results

tests/                      63 offline tests + 1 live test
scripts/run_eval.py         the evaluation harness
scripts/check_ui.py         headless UI smoke check
scripts/fetch_model.py      downloads MiniLM (PyTorch files only)
```

---

## Interface design

Everything visual lives in [src/ui/theme.py](src/ui/theme.py), so chart colour
and interface colour cannot drift apart. The direction:

**The thesis is departure, not temperature.** Every weather app shows you 31
degrees. This one exists to say whether 31 is unusual *here, today*. So the
hero's signature element is a **departure scale**: a track with the ten-year
normal for this calendar date at its centre, one standard deviation shaded,
and a marker where today actually sits. It reads the project's whole argument
in one graphic, and it's computed by the same `_doy_normals` the anomaly agent
uses — so the headline and the analysis tab can never disagree.

**Colour is data.** The palette is a cold-to-hot ramp borrowed from synoptic
charts, and it doubles as the temperature encoding. One chrome accent
(instrument teal) sits deliberately *off* that ramp, so interface furniture is
never mistaken for a reading.

**Numbers are set in mono.** Weather data is instrument output, so every
numeral — readings, coordinates, axis labels, metric rows — is IBM Plex Mono
with tabular figures. Plex Sans Condensed carries headings, Plex Sans the
prose. Flat panels, hairline rules, 6px radii: instruments aren't pillowy.

## Design decisions worth defending in a viva

**Open-Meteo over OpenWeatherMap.** No API key, and the same request shape
serves current conditions, a 16-day forecast, and a reanalysis archive back to
1940. Questions like "compared with previous years" are answerable without a
second integration.

**Anomalies are measured against a climatological normal, not a rolling mean.**
A plain z-score over the last 30 days flags every winter day as cold, which is
not an anomaly — it is January. The detector pools each calendar date's values
from a ±7-day window across all baseline years, so "unusual" means unusual
*for this time of year*.

**Daily rainfall is excluded from distribution-based outlier detection when no
baseline exists.** Rainfall is zero-inflated: most days are exactly 0, so
Q1 = Q3 = 0 and every wet day becomes an "outlier". On a 3-year Mumbai series
this produced 195 false anomalies; routing rainfall to the published IMD
category thresholds instead brought it to 41 real ones.

**Trend claims are gated on R².** Daily weather is mostly noise, so a slope
alone is meaningless. A fit with R² < 0.1 is reported as `stable` with `low`
confidence rather than as a trend, and fits are skipped entirely below 30 days.

**Official thresholds, not invented ones.** IMD rainfall categories, heat-wave
criteria and cyclone classes are in the knowledge base, so the system can say
"very heavy rainfall" and cite where the threshold comes from.

**The system states what it compared against.** "Unusual for this time of year"
and "unusual compared with the last few weeks" are different claims, and the
output always says which one it is making.

---

## RAG configuration

Three retrieval backends, set by `EMBEDDING_BACKEND` in `.env`:

| Value | What it does | Install |
|---|---|---|
| `hash` *(default)* | Lexical: hashed TF-IDF over numpy | nothing |
| `sentence-transformers` | Semantic: MiniLM embeddings | `pip install sentence-transformers` |
| `hybrid` | Both, fused with Reciprocal Rank Fusion | as above |

Rebuild after switching:

```bash
python -m src.cli index
```

Indexes are namespaced per backend (`data/vectorstore/hash/`, `.../st/`), so
they coexist and `hybrid` queries both.

### Measured results

From `eval/results/report.md`, Recall@4 on the held-out test split — the
metric that matters most for RAG, because the generator reads *every*
retrieved chunk, not just the first:

| Backend | Recall@4 (dev) | Recall@4 (test) | MRR (test) |
|---|---|---|---|
| hashed TF-IDF | 76.2% | 37.5% | 0.281 |
| sentence-transformers | 61.9% | 50.0% | 0.500 |
| **hybrid (RRF)** | **81.0%** | **50.0%** | **0.500** |

The interesting finding is that neither backend dominates. **Lexical retrieval
wins when the question reuses the corpus's vocabulary; semantic retrieval wins
when it paraphrases.** On dev, where questions are phrased close to the source
text, TF-IDF beats MiniLM outright. On the held-out split, where the phrasing
is deliberately different ("downpours", "sticky", "out of line with the norm"),
it collapses from 76% to 38% while MiniLM holds at 50%.

Fusing by *rank* rather than score is what makes combining them safe — a
hashed TF-IDF cosine and a MiniLM cosine are not on comparable scales, so
averaging them would just let whichever produces bigger numbers win.

**Caveat worth stating in your report:** the test split is 8 retrieval cases.
The 37.5% → 50.0% gap is a one-question difference. The *direction* is
consistent with the mechanism, but the magnitude is not precise. Widening the
labelled set is the single highest-value thing you can still do.

### Editing the knowledge base

Drop markdown files into `data/knowledge_base/` and run `python -m src.cli
index`. Chunking splits on headings first so a concept stays with its
explanation.

---

## LLM configuration

Set `LLM_BACKEND` in `.env` to `groq` (default, free), `anthropic` (paid), or
`none` (offline). Nothing outside [src/llm/](src/llm/) imports a vendor SDK, so
adding a provider — a local Ollama, say — means one new class in
`providers.py` and no changes anywhere else.

Relevant choices:

- **Structured output for planning.** The planner must return a valid
  `QueryPlan`. Providers differ in how strictly they can enforce a schema
  (Claude has schema-enforced outputs; Groq's JSON mode only guarantees valid
  JSON, not the right shape), so `client.parse()` describes the schema in the
  prompt, validates with Pydantic, and gives the model **one repair attempt**
  with its own error before failing.
- **Two temperatures.** Planning runs at 0.1 — a plan should be reproducible.
  Conversation runs at 0.6, because a deterministic explanation reads like a
  form letter.
- **Prompt caching** (Claude path) on the frozen system prompt. System prompts
  contain no dates or per-request values; a single changing byte would
  invalidate the cache on every call.

Grounding is enforced in the prompt: the model is given the computed numbers
and the retrieved text and told to use nothing else, to say so when the data is
thin, and never to present a short record as evidence of climate change. The
`groundedness` metric in the eval harness checks it actually obeyed.

---

## Evaluation

```bash
python scripts/run_eval.py                      # planning + retrieval, offline
python scripts/run_eval.py --full               # adds groundedness (live data)
python scripts/run_eval.py --llm                # also scores the LLM planner
python scripts/run_eval.py --backend hashed     # one embedding backend only
```

Writes `eval/results/report.md` and `report.json`.

`eval/dataset.py` holds 46 hand-labelled questions with expected intent,
location, variables and — where the answer genuinely depends on the knowledge
base — the document that should be retrieved.

### The dev/test split matters

The set is split deliberately:

- **dev (34 cases)** debugged the rule-based patterns. Scores on it are
  **fitted** and read high.
- **test (12 cases)** was written *afterwards*, using phrasing absent from the
  keyword lists ("outlook", "at present", "out of line with the norm",
  "downpours"). Nothing was tuned against it.

**Quote the test column.** Reporting the dev number as if it were a
generalisation result is the most common way a project like this overstates
itself.

### Three things are measured

| | What it answers | Method |
|---|---|---|
| **Planning** | Did the question reach the right agent path? | intent accuracy, location accuracy, variable F1, confusion matrix |
| **Retrieval** | Did the right document come back, and at what rank? | Recall@1, Recall@4, Precision@1, MRR |
| **Groundedness** | Does every number in the answer trace back to computed data? | numeric extraction vs. the pipeline's own outputs |

The groundedness check is the one that catches a hallucinated statistic: it
pulls every number out of the generated answer and verifies it appears
somewhere in what the pipeline actually computed. Years, ISO dates and
published IMD thresholds are excluded as system vocabulary rather than data
claims.

**Read the two groundedness scores differently.** In offline mode it is 100%,
and it *has* to be — the template can only emit numbers it just computed, so
the metric is measuring itself. That run is the **control**: it establishes
the checker works end to end. The number that carries information is the one
measured with a language model connected, where the model is free to write a
figure nothing produced. Run `--full` in both modes and report both.

**One clean run is not proof.** Generation is non-deterministic, so a 100%
score is one sample, not a guarantee. An earlier LLM-connected run scored
96.8% with six flagged answers. Four were a bug in this checker — it stripped
ISO dates but not `23 Sept`, so a day-of-month read as a measurement — and
that is fixed and tested. The other two were **real**: the model wrote a
trend slope of `3.44` where the pipeline computed `3.35`, and `9.0` where it
computed `9.53`. Small numeric drift, exactly the failure this metric exists
to catch. Run `--full` several times before quoting a figure, and quote the
worst run, not the best.

### The planner comparison must be like for like

Both planners are scored through `PlannerAgent.plan_raw()`, which returns the
plan *before* `_apply_defaults` fills in a fallback location and date window.
An earlier version measured the LLM through `execute()` and the rule-based
planner through `_rule_based()` — post-defaults against pre-defaults — and
marked the LLM wrong on every definitional question for a substitution the
planner never made. Two tests now pin this down.

## Testing

```bash
pytest                       # 63 offline tests, ~0.6s
pytest -m network            # adds the live end-to-end test
python scripts/check_ui.py   # headless Streamlit smoke check
```

Offline tests use synthetic series with known properties — a deliberate
+0.02 C/day slope the trend fitter must recover, an injected 40 C spike the
anomaly detector must catch — so they are deterministic and need no network.
The metric implementations in `eval/metrics.py` are unit-tested against
hand-worked examples: an untested metric is a number you cannot defend.

---

## Known limitations

- Reanalysis archive lags real time by about 5 days; history requests are
  clamped accordingly.
- Daily aggregates only. Urban flooding depends on mm/hour, which daily totals
  cannot show — the system says so rather than implying otherwise.
- The hashing embedder does lexical matching, not semantic.
- Rule-based planning handles common phrasings; unusual ones need the LLM.
- Geocoding takes the first match, so ambiguous place names may resolve to the
  wrong country.
