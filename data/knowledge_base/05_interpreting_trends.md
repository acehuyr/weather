# Interpreting Weather Trends Responsibly

## What a Linear Trend Actually Claims

Fitting a straight line to a weather series estimates the average rate of
change over the fitted window. It claims nothing about causation and nothing
about what happens outside that window. Extrapolating a linear fit forward is
one of the most common errors in weather data analysis.

Three numbers should always be reported together:

- the slope, expressed in useful units such as degrees per decade rather than
  degrees per day,
- the R-squared value, which is the fraction of variance the line explains,
- and the length of the record.

A slope without an R-squared is not interpretable. Daily weather is dominated
by noise, so a trend line through 30 days of temperature typically has an
R-squared near zero. Reporting that as warming would be wrong.

## How Long a Record Is Needed

Detecting a genuine climate trend in temperature generally requires at least 30
years of data, because natural variability on multi-year timescales is large
compared to the trend itself. Rainfall needs even longer records, since its
variance is far higher.

A useful discipline: if the analysis window is shorter than about 10 years,
describe what happened rather than claiming a trend. Say that this September
was warmer than the previous five Septembers, not that the climate is warming.

## Seasonality Must Be Removed First

A trend fitted across a window that begins in winter and ends in summer will
always show warming, purely because of the seasonal cycle. Legitimate options:

- compare the same calendar window across different years,
- fit the trend on anomalies rather than raw values,
- or use annual aggregates.

Comparing like with like is what makes a comparison meaningful.

## The Urban Heat Island Effect

Cities are warmer than the countryside around them, often by 2 to 5 C at night.
Concrete and asphalt store heat during the day and release it slowly, vegetation
that would cool the air by transpiration is absent, and vehicles, air
conditioners and industry add waste heat directly.

The effect is strongest on calm, clear nights and weakest on windy or cloudy
ones. It shows up most clearly in minimum temperatures.

This matters for trend analysis: part of a warming trend measured at a city
station can be growth of the city around the station rather than regional
climate change. A responsible analysis mentions this possibility when the
station is urban.

## Correlation Is Not Explanation

Weather variables are strongly correlated with each other for physical reasons.
Humidity rises when it rains; maximum temperature falls on cloudy days;
pressure falls when a storm approaches. Reporting these correlations as
discoveries adds nothing. The value is in the mechanism, not the coefficient.

## Communicating Uncertainty

Every number produced from weather data carries uncertainty from three sources:
measurement error at the station, interpolation if the data is gridded or
reanalysed, and sampling error from a finite record.

Reanalysis products such as ERA5 are models constrained by observations, not
direct measurements. They are excellent for regional and long-term analysis and
less reliable for a specific hour at a specific point, particularly for
rainfall and in complex terrain.

State the limitation once, plainly, and carry on. Hedging every sentence makes
a report unreadable; hiding the limitation entirely makes it misleading.
