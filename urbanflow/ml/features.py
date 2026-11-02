"""Causal 15-minute features. Counts must be complete; unknown != zero.

V2 intentionally changes the legacy schema and must only be used with V2 models.
All numeric features use float64; region is a numeric identifier shared by models.
"""
import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 4, 24, 96, 168, 336, 672, 720)
WINDOWS = (24, 96, 168, 336, 672, 720)
TIME_COLUMNS = ['region', 'hour', 'quarter', 'day_of_week', 'month', 'is_weekend',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
COLUMNS = TIME_COLUMNS + [f'lag_{k}' for k in LAGS] + [f'rolling_{s}_{k}' for k in WINDOWS for s in ('mean', 'std')] + ['ewma']


def validate_panel(df):
    required = {'datetime', 'region', 'total_pickups'}
    if not required.issubset(df):
        raise ValueError(f'Missing required columns: {required - set(df)}')
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('America/New_York', ambiguous='raise', nonexistent='raise')
    if df.duplicated(['datetime', 'region']).any():
        raise ValueError('Duplicate region/timestamp observations')
    if not df.region.between(0, 29).all() or not np.equal(df.region, np.floor(df.region)).all():
        raise ValueError('Invalid region')
    if not np.isfinite(df.total_pickups).all() or (df.total_pickups < 0).any():
        raise ValueError('Missing, negative or nonfinite observations')
    for region, g in df.groupby('region'):
        ts = g.datetime.sort_values()
        if len(ts) > 1 and not ts.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
            raise ValueError(f'Noncontinuous 15-minute observations in region {region}')
    return df.sort_values(['datetime', 'region']).reset_index(drop=True)


def time_features(index, regions):
    t = pd.DatetimeIndex(index)
    h = t.hour + t.minute / 60
    d = t.day_of_week
    return pd.DataFrame(dict(region=regions, hour=t.hour, quarter=t.minute // 15,
        day_of_week=d, month=t.month, is_weekend=(d >= 5).astype(int),
        hour_sin=np.sin(2*np.pi*h/24), hour_cos=np.cos(2*np.pi*h/24),
        dow_sin=np.sin(2*np.pi*d/7), dow_cos=np.cos(2*np.pi*d/7)))


def training_features(panel):
    panel = validate_panel(panel)
    result = time_features(panel.datetime, panel.region.to_numpy())
    for _, g in panel.groupby('region', sort=False):
        y = g.total_pickups
        base = y.shift(1)
        for k in LAGS:
            result.loc[g.index, f'lag_{k}'] = y.shift(k)
        for k in WINDOWS:
            result.loc[g.index, f'rolling_mean_{k}'] = base.rolling(k, min_periods=k).mean()
            result.loc[g.index, f'rolling_std_{k}'] = base.rolling(k, min_periods=k).std(ddof=1)
        result.loc[g.index, 'ewma'] = base.ewm(alpha=.4, adjust=False).mean()
    return panel, result[COLUMNS].astype('float64')


def next_features(history, timestamp, regions, ewma):
    """history shape (time, region); strictly before timestamp, on a complete grid."""
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or len(history) < max(LAGS):
        raise ValueError('At least 720 consecutive observations per region required')
    if not np.isfinite(history).all():
        raise ValueError('Historical context contains missing/nonfinite demand')
    x = time_features([timestamp] * len(regions), regions)
    for k in LAGS:
        x[f'lag_{k}'] = history[-k]
    for k in WINDOWS:
        x[f'rolling_mean_{k}'] = history[-k:].mean(axis=0)
        x[f'rolling_std_{k}'] = history[-k:].std(axis=0, ddof=1)
    x['ewma'] = ewma
    return x[COLUMNS].astype('float64')


def recursive(model, history, origin, regions, horizon):
    """Never receives future targets. Model None is the weekly seasonal naive."""
    h = np.asarray(history, dtype=float).copy()
    ewma = h[0].copy()
    for row in h[1:]:
        ewma = .4*row + .6*ewma
    out = []
    for step in range(horizon):
        x = next_features(h[-720:], origin + pd.Timedelta(minutes=15*step), regions, ewma)
        p = h[-672].copy() if model is None else np.asarray(model.predict(x), dtype=float).reshape(-1)
        if p.shape != (len(regions),) or not np.isfinite(p).all():
            raise RuntimeError('Invalid model prediction shape or values')
        p = np.maximum(p, 0)
        out.append(p)
        h = np.vstack([h, p])
        ewma = .4*p + .6*ewma
    return np.asarray(out)


def chronological_split(panel, train_end='2016-02-15', validation_end='2016-03-01', gap='2h'):
    """Gap excludes training labels; observed gap demand remains origin context."""
    ts = pd.to_datetime(panel.datetime)
    a, b = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    if ts.dt.tz is not None:
        a=a.tz_localize(ts.dt.tz); b=b.tz_localize(ts.dt.tz)
    masks = (ts < a-pd.Timedelta(gap), (ts >= a) & (ts < b-pd.Timedelta(gap)), ts >= b)
    for left, right in zip(masks, masks[1:]):
        if ts[left].max() >= ts[right].min():
            raise AssertionError('Temporal leakage')
    return masks
