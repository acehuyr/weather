# Evaluation Report

Generated: 2026-09-21 10:26

## Dataset

- Questions: **46** (34 dev / 12 test)
- With a labelled gold document: **29**
- Paraphrase cases: **8**
- By intent: {'anomaly': 8, 'comparison': 7, 'current': 8, 'forecast': 7, 'general': 9, 'trend': 7}

> **dev** cases were used to debug the rule-based patterns, so those
> scores are fitted and read high. **test** cases were written
> afterwards with phrasing absent from the keyword lists, and nothing
> was tuned against them. Quote the test column.

## 1. Planning

| Mode | Split | n | Intent accuracy | Location accuracy | Variable F1 | Fell back |
|---|---|---|---|---|---|---|
| rule-based | **dev** | 34 | 100.0% | 100.0% | 0.882 | — |
| rule-based | **test** | 12 | 33.3% | 100.0% | 0.750 | — |
| llm | **dev** | 34 | 100.0% | 100.0% | 0.912 | 32 |
| llm | **test** | 12 | 33.3% | 100.0% | 0.750 | 12 |

> **The LLM row is not a clean measurement.** *Fell back* counts questions where the API call failed - almost always a free-tier rate limit - and the rule-based planner answered instead. Those rows drag the LLM score toward the rule-based one. Re-run when the limit resets, or pace the run with `--sleep`.

### Planning failures (rule-based, test)

- `Give me the outlook for Jaipur over the next ten days` - intent current != forecast
- `Has Mumbai become rainier across the last decade?` - intent current != trend
- `Is today's heat in Nagpur out of line with the norm?` - intent current != anomaly
- `Set that against what Chennai usually sees in October` - intent current != comparison
- `Put simply, what makes air feel sticky?` - intent current != general
- `Does Delhi see more downpours now than it used to?` - intent current != trend
- `Anything freakish in the Pune numbers lately?` - intent current != anomaly
- `Line up this year's Mumbai monsoon beside the usual` - intent trend != comparison

### Planning failures (llm, test)

- `Give me the outlook for Jaipur over the next ten days` - intent current != forecast
- `Has Mumbai become rainier across the last decade?` - intent current != trend
- `Is today's heat in Nagpur out of line with the norm?` - intent current != anomaly
- `Set that against what Chennai usually sees in October` - intent current != comparison
- `Put simply, what makes air feel sticky?` - intent current != general
- `Does Delhi see more downpours now than it used to?` - intent current != trend
- `Anything freakish in the Pune numbers lately?` - intent current != anomaly
- `Line up this year's Mumbai monsoon beside the usual` - intent trend != comparison

## 2. Retrieval

Top-k = 4. Scored on the 29 questions with a labelled gold document. Retrieval was never tuned against either split, so dev and test are both honest here; they are shown separately only for consistency.

| Backend | Split | n | Dim | Recall@1 | Recall@4 | Precision@1 | MRR |
|---|---|---|---|---|---|---|---|
| hashed TF-IDF | **dev** | 21 | 8192 | 47.6% | 76.2% | 47.6% | 0.603 |
| hashed TF-IDF | **test** | 8 | 8192 | 25.0% | 37.5% | 25.0% | 0.281 |
| sentence-transformers | **dev** | 21 | 384 | 47.6% | 61.9% | 47.6% | 0.520 |
| sentence-transformers | **test** | 8 | 384 | 50.0% | 50.0% | 50.0% | 0.500 |
| hybrid (RRF) | **dev** | 21 | 8576 | 42.9% | 81.0% | 42.9% | 0.563 |
| hybrid (RRF) | **test** | 8 | 8576 | 50.0% | 50.0% | 50.0% | 0.500 |

### Retrieval misses (hashed TF-IDF, dev)

- `How muggy is it in Chennai today?` - wanted ['01_weather_variables.md'], got ['05_interpreting_trends.md', '02_climatology_and_anomalies.md']
- `How has the temperature in Delhi been changing?` - wanted ['05_interpreting_trends.md'], got ['02_climatology_and_anomalies.md', '01_weather_variables.md']
- `How has rainfall in Pune compared with previous years?` - wanted ['02_climatology_and_anomalies.md'], got ['05_interpreting_trends.md', '03_indian_monsoon.md']
- `Has there been any extreme rainfall in Mumbai recently?` - wanted ['04_extreme_weather_criteria.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Did Pune record anything out of the ordinary this month?` - wanted ['02_climatology_and_anomalies.md'], got ['01_weather_variables.md', '04_extreme_weather_criteria.md']

### Retrieval misses (hashed TF-IDF, test)

- `Is today's heat in Nagpur out of line with the norm?` - wanted ['02_climatology_and_anomalies.md'], got ['04_extreme_weather_criteria.md', '01_weather_variables.md']
- `Set that against what Chennai usually sees in October` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']
- `Put simply, what makes air feel sticky?` - wanted ['01_weather_variables.md'], got ['04_extreme_weather_criteria.md', '04_extreme_weather_criteria.md']
- `Does Delhi see more downpours now than it used to?` - wanted ['05_interpreting_trends.md'], got ['02_climatology_and_anomalies.md', '03_indian_monsoon.md']
- `Anything freakish in the Pune numbers lately?` - wanted ['02_climatology_and_anomalies.md'], got ['01_weather_variables.md', '05_interpreting_trends.md']

### Retrieval misses (sentence-transformers, dev)

- `How muggy is it in Chennai today?` - wanted ['01_weather_variables.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Show me the long-term rainfall pattern in Pune` - wanted ['05_interpreting_trends.md'], got ['01_weather_variables.md', '03_indian_monsoon.md']
- `How has rainfall in Pune compared with previous years?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '01_weather_variables.md']
- `Is this month wetter in Mumbai than last year?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `How does this season in Chennai stack up against earlier years?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Why is the current weather in Chennai unusual?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Is anything abnormal about the weather in Mumbai?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Is the weather in Nagpur behaving strangely?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']

### Retrieval misses (sentence-transformers, test)

- `Has Mumbai become rainier across the last decade?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '01_weather_variables.md']
- `Is today's heat in Nagpur out of line with the norm?` - wanted ['02_climatology_and_anomalies.md'], got ['04_extreme_weather_criteria.md', '04_extreme_weather_criteria.md']
- `Set that against what Chennai usually sees in October` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']
- `Does Delhi see more downpours now than it used to?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '01_weather_variables.md']

### Retrieval misses (hybrid (RRF), dev)

- `How muggy is it in Chennai today?` - wanted ['01_weather_variables.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `How has rainfall in Pune compared with previous years?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Is anything abnormal about the weather in Mumbai?` - wanted ['02_climatology_and_anomalies.md'], got ['04_extreme_weather_criteria.md', '05_interpreting_trends.md']
- `Did Pune record anything out of the ordinary this month?` - wanted ['02_climatology_and_anomalies.md'], got ['01_weather_variables.md', '03_indian_monsoon.md']

### Retrieval misses (hybrid (RRF), test)

- `Has Mumbai become rainier across the last decade?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']
- `Is today's heat in Nagpur out of line with the norm?` - wanted ['02_climatology_and_anomalies.md'], got ['04_extreme_weather_criteria.md', '04_extreme_weather_criteria.md']
- `Set that against what Chennai usually sees in October` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']
- `Does Delhi see more downpours now than it used to?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '02_climatology_and_anomalies.md']

## 3. Groundedness

- Mean groundedness: **100.0%**
- Fully grounded answers: **46/46**

Every number in the answer is checked against the numbers the pipeline actually computed. IMD thresholds and years are excluded as system vocabulary rather than data claims.

---

Reproduce with `python scripts/run_eval.py --full`.