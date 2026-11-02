"""Weather-aware causal calendar recursion; original v4 remains immutable."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ml.advanced.core import HISTORY, histories, make_features

WEATHER_COLUMNS = ['temperature_2m', 'apparent_temperature', 'precipitation', 'snowfall', 'wind_speed_10m', 'relative_humidity_2m']
WEATHER_FEATURES = ['observed_weather_' + name for name in ['temperature', 'apparent_temperature', 'precipitation', 'snowfall', 'wind', 'humidity']]


def weather_times(origins):
    return pd.DatetimeIndex(origins).tz_convert('UTC').floor('h') - pd.Timedelta(hours=3)


def weather_values(weather, origins):
    values = weather.reindex(weather_times(origins))[WEATHER_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Complete lagged historical weather is required; missing data is not fabricated')
    return values


@dataclass
class WeatherCalendarRecursive:
    base: object
    calendar: object
    weather: object
    weight: float = .6
    decay: float = 12.

    def forecast(self, y, times, positions, horizon=48):
        positions = np.asarray(positions, dtype=int)
        origins = pd.DatetimeIndex(times[positions])
        known_weather = weather_values(self.weather, origins)
        h = histories(y, positions)
        b, _, r = h.shape
        h = np.concatenate([h, np.zeros((b, horizon, r), dtype=np.float32)], axis=1)
        calendar = self.calendar.forecast(y, times, positions, horizon)
        result = []
        for k in range(horizon):
            targets = origins + pd.Timedelta(minutes=15*k)
            x = make_features(h[:, :HISTORY+k], targets, list(range(r)), self.base.config,
                              self.base.scales, self.base.neighbors, weather_values=known_weather)
            p = self.base.predict(x).reshape(b, r)
            if self.base.anchor_weight:
                anchor_weight = self.base.anchor_weight*k/47
                p = (1-anchor_weight)*p + anchor_weight*x.seasonal_anchor.to_numpy().reshape(b, r)
            if self.base.bias is not None:
                p += self.base.bias[min(k//12, 3)]
            weight = self.weight*(1-np.exp(-k/self.decay))
            p = np.maximum(0, (1-weight)*p+weight*calendar[:, k])
            if not np.isfinite(p).all():
                raise RuntimeError('Nonfinite weather-aware forecast')
            result.append(p)
            h[:, HISTORY+k] = p
        return np.stack(result, axis=1)
