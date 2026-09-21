# Weather Variables and How to Read Them

Weather is described by a small set of measured quantities. Understanding what
each one means - and what a normal value looks like - is what turns a number
into an insight.

## Air Temperature

Air temperature is measured at 2 metres above ground in a shaded, ventilated
screen (a Stevenson screen), which is why datasets label it temperature_2m.
Measuring in direct sunlight would report the temperature of the thermometer,
not of the air.

Three daily values matter:

- Maximum temperature usually occurs in mid-afternoon, roughly two to three
  hours after solar noon, because the ground keeps releasing stored heat after
  peak sunlight.
- Minimum temperature occurs just before sunrise, at the end of a full night
  of radiative cooling.
- Mean temperature is the average across the day and is the value most often
  used for trend analysis, because it is less noisy than the extremes.

The diurnal range is the gap between maximum and minimum. A wide range of 15 C
or more indicates dry, clear air over land. A narrow range indicates humidity,
cloud cover, or a maritime location, because water vapour and clouds trap
outgoing heat at night.

## Relative Humidity

Relative humidity is the amount of water vapour in the air expressed as a
percentage of the maximum the air could hold at that temperature. It is a
ratio, not an absolute amount, which leads to a common misreading: relative
humidity rises overnight and falls in the afternoon even when the actual
moisture content has not changed at all, purely because warm air can hold more
vapour than cool air.

Rules of thumb:

- Below 30 percent is dry; skin, crops and soil lose moisture quickly.
- 40 to 60 percent is the comfort band for most people.
- Above 70 percent begins to suppress the evaporation of sweat, which is why
  humid heat feels far worse than dry heat at the same temperature.
- Above 90 percent with falling temperature commonly produces fog or dew.

## Apparent Temperature and Heat Index

Apparent temperature, also called feels-like temperature, combines air
temperature with humidity and wind to estimate the physiological effect on a
human body. In humid conditions the apparent temperature can exceed the
measured temperature by 5 to 10 C, because sweat cannot evaporate efficiently.
In windy, dry cold it falls below the measured temperature, which is wind
chill. When explaining risk to people, apparent temperature is more relevant
than the raw reading.

## Precipitation

Precipitation is reported as a depth in millimetres: the height that fallen
water would reach on a flat, non-absorbing surface. 1 mm of rain equals one
litre of water per square metre.

Rainfall is the most skewed of all weather variables. A monthly mean is often
misleading because a single day can carry most of the total for that month.
For rainfall, always report:

- the total over the period,
- the number of rainy days, where the India Meteorological Department counts a
  day with 2.5 mm or more as a rainy day,
- and the single heaviest day.

A month with 200 mm spread over 20 days and a month with 200 mm in two days are
the same by mean and radically different in consequence.

## Wind Speed and Direction

Wind speed is measured at 10 metres above ground and reported in km/h or m/s.
Wind direction names where the wind comes from: a westerly blows from west to
east. Direction matters for interpretation because it tells you the origin of
the air mass. Onshore winds carry moisture inland, while continental winds are
dry.

Gusts are short peaks above the sustained average and are what cause structural
damage, so a forecast quoting only mean wind speed understates the risk.

## Atmospheric Pressure

Surface pressure is the weight of the air column above, measured in
hectopascals. The global average at sea level is about 1013 hPa. The absolute
value matters less than the change:

- Falling pressure indicates an approaching low-pressure system, bringing
  clouds, wind and rain.
- Rising pressure indicates a high-pressure system and settled, clear
  conditions.
- A rapid fall of more than 3 hPa in three hours signals a fast-developing
  storm.

Tropical cyclones are defined partly by their central pressure; a deeper and
therefore lower centre means a more intense storm.
