# Climatology, Normals and Anomalies

## Weather versus Climate

Weather is the state of the atmosphere at a moment. Climate is the statistical
distribution of weather at a place over decades. A single hot day is weather; a
shift in the average number of hot days over thirty years is climate. Any
statement about whether a day is unusual is implicitly a comparison against a
climate distribution.

## Climate Normals

A climate normal is the average of a weather variable over a standard
reference period. The World Meteorological Organization defines normals over
30-year periods, updated every decade. The current standard reference period is
1991 to 2020.

Thirty years is a deliberate compromise. A shorter window is dominated by
year-to-year noise such as El Nino and La Nina cycles; a much longer window
blends together genuinely different climate states and hides recent change.

## Anomalies

An anomaly is the departure of an observation from the normal for that place
and that time of year:

    anomaly = observed value - climatological normal

Anomalies are far more informative than raw values because they remove the
seasonal cycle. A 20 C day in Delhi is a cold anomaly in May and a warm anomaly
in January. Comparing raw values across the calendar is nearly meaningless;
comparing anomalies is meaningful.

The key requirement is that the normal must be computed for the same time of
year. The standard approach is to pool observations from a window of plus or
minus 5 to 7 days around the target date, across all baseline years, so that
each calendar date gets its own mean and spread.

## The Z-Score

A z-score expresses an anomaly in units of the historical spread:

    z = (observed value - mean) / standard deviation

This normalises across variables and seasons. A z-score of +2 means the value
sits two standard deviations above the seasonal normal.

For an approximately normal distribution:

- about 68 percent of observations fall within z of plus or minus 1,
- about 95 percent within plus or minus 2,
- about 99.7 percent within plus or minus 3.

So a magnitude of 2 or more is a reasonable threshold for notable, and 3 or
more for extreme. Temperature is close enough to normally distributed for this
to work well.

Rainfall is not normally distributed. It is zero-inflated and strongly
right-skewed, because most days have no rain at all, so z-scores on daily
rainfall are unreliable and will over-flag. For rainfall, prefer percentile
ranks, fixed category thresholds, or z-scores computed on accumulated totals
over a week or a month rather than on single days.

## Interquartile Range as a Fallback

When no multi-year baseline is available, the interquartile range gives a
distribution-free alternative. With Q1 and Q3 as the 25th and 75th percentiles
and IQR equal to Q3 minus Q1, a value is an outlier if it falls below
Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR.

This flags outliers within the observed window only, so it answers the question
unusual compared with the last few weeks, not unusual for this time of year.
The distinction should always be stated in the output, because the two claims
have very different strength.

## Why an Anomaly is Not Automatically Alarming

Anomalies are expected. By construction, roughly 5 percent of all days exceed a
z-score magnitude of 2. Detecting an anomaly is the beginning of an
explanation, not the conclusion. A responsible report states the anomaly, its
size, its rarity, and where possible a plausible mechanism such as a passing
weather system, a heat wave, a cyclone, or an unusually early monsoon onset,
rather than asserting significance from the number alone.
