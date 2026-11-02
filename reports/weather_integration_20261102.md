# Weather integration and coordinate-derived location names

The original v4 artifact remains unchanged. A separate `weather_calendar_feedback_v1`
candidate adds six lagged weather fields to the same 599-tree XGBoost/calendar
recursion. Model selection remains independent of the June diagnostic comparison.

## Real API inputs

Open-Meteo's Historical Weather API supplied 4,416 complete hourly ERA5 records
for December 31, 2015 through July 1, 2016 at the NYC reference point
(40.7128, -74.0060). Fields include air/apparent temperature, precipitation,
snowfall, wind, humidity, cloud cover and weather code. Raw responses, request
URLs and SHA-256 hashes are retained. UTC timestamps avoid DST ambiguity.

Each forecast uses the weather at the origin's UTC hour minus three hours.
These six values are held fixed through the entire recursive trajectory; future
observed weather is never supplied to the forecast. The three-hour timestamp lag
is conservative but does not certify historical publication availability:
ERA5 is retrospective reanalysis, not an archived forecast issued in 2016.
Weather represents one NYC reference location, not separate observations at
all 30 neighborhoods. Cached historical API data permits reliable inference
without making visitor requests depend on the external provider.

## Bounded comparison

The May comparison uses labels strictly before April 30, 2016 22:00 New York time
and non-overlapping origins spaced 13 hours apart. Both controls have the same
capacity, calendar construction and recursion; only the weather features differ.
May is reused development data, not an untouched holdout.

| Model | May cumulative 12-hour WAPE | June diagnostic cumulative 12-hour WAPE |
|---|---:|---:|
| Matched no-weather development control | 12.5081% | Not used as the June reference |
| Frozen v4 reference | Not substituted for the matched May control | 11.6086% |
| Lagged-weather candidate | 12.4792% | 11.4395% |

The May gain is 0.0289 WAPE point, below the predeclared 0.1-point default-promotion
threshold. Therefore v4 remains the default, while the weather candidate is
available through an explicit experimental mode. June was examined only after
saving that decision and is spent diagnostic evidence. Its apparent improvement
does not establish superiority on a new independent test.

Weather mode's June one-step WAPE is 9.6992%; twelve-hour MAE is 14.3474,
RMSE 23.5565 and R-squared 0.961763. These are model-specific diagnostic results,
not replacements for v4's original frozen evidence. No calibrated prediction
bands or weather-driven operational alerts are claimed.

## Coordinate-derived names

Each of the 30 inverse-scaled fitted centroids was located in exactly one
polygon from NYC Department of City Planning's official 2020 Neighborhood
Tabulation Areas dataset. Polygon holes are respected. Labels include Harlem
(South), Midtown-Times Square, Chelsea-Hudson Yards, Greenwich Village,
Bushwick (East) and JFK Airport. Original NTA names and source codes are retained.
The names label centroid locations and do not define model cluster boundaries.
2020 statistical area names are presentational metadata, not a claim that the
2016 model regions equal those polygons. Multiple centroids can share an area
name; the original region IDs remain visible for disambiguation.

## Reproduction and serving

Run `prepare_weather_sources.py` to fetch or verify cached inputs, then
`weather_experiment.py` to rebuild causal features and run the fixed experiment.
An optional `--feature-cache` path can reuse the original verified training rows.
No weather hyperparameter search was performed. The serving repository copies
only the inference class, integrity-checked bundle and label lookup.

`POST /predict` accepts `model_variant: "baseline"` (default) or `"weather"`.
The batch endpoint and forecast map support the same option. `/weather` exposes
lagged historical context; `/model/info` and `/model/metrics` report both models.
The frontend model selector updates predictions, forecast maps and displayed
model metrics. Weather panels and CSV provenance identify the chosen model.

Sources: [Open-Meteo ERA5 API](https://open-meteo.com/en/docs/historical-weather-api)
and [NYC 2020 NTAs](https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd).
