# Evaluation Report

Generated: 2026-09-21 20:55

## Dataset

- Questions: **70** (46 dev / 24 test)
- With a labelled gold document: **45**
- Paraphrase cases: **40**
- By intent: {'anomaly': 12, 'comparison': 11, 'current': 12, 'forecast': 11, 'general': 13, 'trend': 11}

> **dev** cases are fair game for tuning, so those scores are
> fitted and read high. They include the 12 questions that were the
> v1 held-out split until their failures were used to broaden the
> intent patterns - spending a split is allowed, pretending you
> did not is not. **test** is a fresh 24-question split written
> before that change and not consulted during it.
>
> **Quote the test column.**

## 1. Planning

| Mode | Split | n | Intent accuracy | Location accuracy | Variable F1 | Fell back |
|---|---|---|---|---|---|---|
| rule-based | **dev** | 46 | 100.0% | 100.0% | 0.848 | — |
| rule-based | **test** | 24 | 79.2% | 100.0% | 0.667 | — |

### Planning failures (rule-based, test)

- `What kind of week is Bengaluru in for, heat-wise?` - intent current != forecast
- `Is Shimla losing its winters?` - intent current != trend
- `What direction has Lucknow's climate been moving?` - intent current != trend
- `Has Shimla thrown up any surprises this week?` - intent forecast != anomaly
- `Is Cherrapunji's rainfall off the charts right now?` - intent current != anomaly

## 2. Retrieval

Top-k = 4. Scored on the 45 questions with a labelled gold document. Retrieval was never tuned against either split, so dev and test are both honest here; they are shown separately only for consistency.

| Backend | Split | n | Dim | Recall@1 | Recall@4 | Precision@1 | MRR |
|---|---|---|---|---|---|---|---|
| hashed TF-IDF | **dev** | 29 | 8192 | 41.4% | 65.5% | 41.4% | 0.514 |
| hashed TF-IDF | **test** | 16 | 8192 | 25.0% | 68.8% | 25.0% | 0.411 |
| sentence-transformers | **dev** | 29 | 384 | 48.3% | 58.6% | 48.3% | 0.514 |
| sentence-transformers | **test** | 16 | 384 | 25.0% | 37.5% | 25.0% | 0.312 |
| hybrid (RRF) | **dev** | 29 | 8576 | 44.8% | 72.4% | 44.8% | 0.546 |
| hybrid (RRF) | **test** | 16 | 8576 | 31.2% | 68.8% | 31.2% | 0.411 |

### Retrieval misses (hashed TF-IDF, dev)

- `How muggy is it in Chennai today?` - wanted ['01_weather_variables.md'], got ['05_interpreting_trends.md', '02_climatology_and_anomalies.md']
- `How has the temperature in Delhi been changing?` - wanted ['05_interpreting_trends.md'], got ['02_climatology_and_anomalies.md', '01_weather_variables.md']
- `How has rainfall in Pune compared with previous years?` - wanted ['02_climatology_and_anomalies.md'], got ['05_interpreting_trends.md', '03_indian_monsoon.md']
- `Has there been any extreme rainfall in Mumbai recently?` - wanted ['04_extreme_weather_criteria.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Did Pune record anything out of the ordinary this month?` - wanted ['02_climatology_and_anomalies.md'], got ['01_weather_variables.md', '04_extreme_weather_criteria.md']
- `Is today's heat in Nagpur out of line with the norm?` - wanted ['02_climatology_and_anomalies.md'], got ['04_extreme_weather_criteria.md', '01_weather_variables.md']
- `Set that against what Chennai usually sees in October` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']
- `Put simply, what makes air feel sticky?` - wanted ['01_weather_variables.md'], got ['04_extreme_weather_criteria.md', '04_extreme_weather_criteria.md']
- `Does Delhi see more downpours now than it used to?` - wanted ['05_interpreting_trends.md'], got ['02_climatology_and_anomalies.md', '03_indian_monsoon.md']
- `Anything freakish in the Pune numbers lately?` - wanted ['02_climatology_and_anomalies.md'], got ['01_weather_variables.md', '05_interpreting_trends.md']

### Retrieval misses (hashed TF-IDF, test)

- `Is Shimla losing its winters?` - wanted ['05_interpreting_trends.md'], got ['02_climatology_and_anomalies.md', '04_extreme_weather_criteria.md']
- `Has the rain in Cherrapunji thinned out over time?` - wanted ['05_interpreting_trends.md'], got ['04_extreme_weather_criteria.md', '04_extreme_weather_criteria.md']
- `What direction has Lucknow's climate been moving?` - wanted ['05_interpreting_trends.md'], got ['02_climatology_and_anomalies.md', '02_climatology_and_anomalies.md']
- `Weigh this week's Jaipur heat against a typical year` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Does anything in the Ahmedabad data look off?` - wanted ['02_climatology_and_anomalies.md'], got ['05_interpreting_trends.md', '05_interpreting_trends.md']

### Retrieval misses (sentence-transformers, dev)

- `How muggy is it in Chennai today?` - wanted ['01_weather_variables.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Show me the long-term rainfall pattern in Pune` - wanted ['05_interpreting_trends.md'], got ['01_weather_variables.md', '03_indian_monsoon.md']
- `How has rainfall in Pune compared with previous years?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '01_weather_variables.md']
- `Is this month wetter in Mumbai than last year?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `How does this season in Chennai stack up against earlier years?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Why is the current weather in Chennai unusual?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Is anything abnormal about the weather in Mumbai?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Is the weather in Nagpur behaving strangely?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']
- `Has Mumbai become rainier across the last decade?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '01_weather_variables.md']
- `Is today's heat in Nagpur out of line with the norm?` - wanted ['02_climatology_and_anomalies.md'], got ['04_extreme_weather_criteria.md', '04_extreme_weather_criteria.md']
- `Set that against what Chennai usually sees in October` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']
- `Does Delhi see more downpours now than it used to?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '01_weather_variables.md']

### Retrieval misses (sentence-transformers, test)

- `Have Ahmedabad's summers drifted since the nineties?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Has the rain in Cherrapunji thinned out over time?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `What direction has Lucknow's climate been moving?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `How does Hyderabad this month measure against its usual September?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Put Nagpur's current spell next to what it saw twelve months ago` - wanted ['02_climatology_and_anomalies.md'], got ['04_extreme_weather_criteria.md', '04_extreme_weather_criteria.md']
- `Is Kolkata wetter or drier than it was three years back?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '01_weather_variables.md']
- `Weigh this week's Jaipur heat against a typical year` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Does anything in the Ahmedabad data look off?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']
- `Is Cherrapunji's rainfall off the charts right now?` - wanted ['04_extreme_weather_criteria.md'], got ['03_indian_monsoon.md', '02_climatology_and_anomalies.md']
- `Something looks wrong with Lucknow's temperatures lately` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']

### Retrieval misses (hybrid (RRF), dev)

- `How muggy is it in Chennai today?` - wanted ['01_weather_variables.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `How has rainfall in Pune compared with previous years?` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Is anything abnormal about the weather in Mumbai?` - wanted ['02_climatology_and_anomalies.md'], got ['04_extreme_weather_criteria.md', '05_interpreting_trends.md']
- `Did Pune record anything out of the ordinary this month?` - wanted ['02_climatology_and_anomalies.md'], got ['01_weather_variables.md', '03_indian_monsoon.md']
- `Has Mumbai become rainier across the last decade?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']
- `Is today's heat in Nagpur out of line with the norm?` - wanted ['02_climatology_and_anomalies.md'], got ['04_extreme_weather_criteria.md', '04_extreme_weather_criteria.md']
- `Set that against what Chennai usually sees in October` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']
- `Does Delhi see more downpours now than it used to?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '02_climatology_and_anomalies.md']

### Retrieval misses (hybrid (RRF), test)

- `Has the rain in Cherrapunji thinned out over time?` - wanted ['05_interpreting_trends.md'], got ['03_indian_monsoon.md', '04_extreme_weather_criteria.md']
- `Is Kolkata wetter or drier than it was three years back?` - wanted ['02_climatology_and_anomalies.md'], got ['01_weather_variables.md', '05_interpreting_trends.md']
- `Weigh this week's Jaipur heat against a typical year` - wanted ['02_climatology_and_anomalies.md'], got ['03_indian_monsoon.md', '03_indian_monsoon.md']
- `Does anything in the Ahmedabad data look off?` - wanted ['02_climatology_and_anomalies.md'], got ['04_extreme_weather_criteria.md', '05_interpreting_trends.md']
- `Is Cherrapunji's rainfall off the charts right now?` - wanted ['04_extreme_weather_criteria.md'], got ['03_indian_monsoon.md', '02_climatology_and_anomalies.md']

---

Reproduce with `python scripts/run_eval.py --full`.