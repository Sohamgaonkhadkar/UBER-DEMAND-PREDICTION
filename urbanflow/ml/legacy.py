"""Explicit repair of the original schema. Semantic changes require evaluation.

Original global lags mixed regions and cannot be faithfully reconstructed from a
single-region API context. This builder gives them causal region-local meanings;
it does not assert equivalence to the old training process.
"""
import numpy as np
import pandas as pd

HOLIDAYS = dict(zip(pd.to_datetime(['2016-01-01','2016-01-18','2016-02-15','2016-02-14','2016-03-17','2016-03-25','2016-03-27']),
    ['new_years_day','mlk_day','presidents_day','valentines_day','st_patricks_day','good_friday','easter_sunday']))
EVENTS = {'restaurant_week':('2016-01-18','2016-01-29'),'fashion_week':('2016-02-11','2016-02-18'),'nyc_half_marathon':('2016-03-20','2016-03-20')}
WEATHER = ['temperature_2m','apparent_temperature','precipitation','snowfall','snow_depth','windspeed_10m','cloudcover','relativehumidity_2m','is_raining','is_snowing','is_heavy_weather','is_freezing','is_storm_jonas','temp_bin','weather_condition']

def build_legacy(context, ts, weather, columns):
    df=context.copy().sort_index()
    if df.region.nunique()!=1 or df.index.has_duplicates: raise ValueError('Expected one region without duplicates')
    if not df.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all(): raise ValueError('Historical gaps')
    if len(df)<721 or not df.index[-1]==ts: raise ValueError('Insufficient or misaligned context')
    if not np.isfinite(df.total_pickups.iloc[:-1]).all(): raise ValueError('Unknown historical observations')
    for c,v in dict(hour=df.index.hour,day=df.index.day,month=df.index.month,day_of_week=df.index.dayofweek).items(): df[c]=v
    for name,period in [('hour',24),('day',31),('month',12)]:
        df[name+'_sin']=np.sin(2*np.pi*df[name]/period); df[name+'_cos']=np.cos(2*np.pi*df[name]/period)
    df['dow_sin']=np.sin(2*np.pi*df.day_of_week/7);df['dow_cos']=np.cos(2*np.pi*df.day_of_week/7)
    df['is_weekend']=(df.day_of_week>=5).astype(int)
    df['is_peak_hour']=df.hour.isin([7,8,9,17,18,19]).astype(int)
    df['is_night']=(df.hour<=5).astype(int)
    for k in [1,2,3,4,24,168,336,720]:
        df[f'lag_{k}']=df.total_pickups.shift(k)
        df[f'region_lag_{k}']=df[f'lag_{k}']
        if k>=24:
            df[f'lag_{k}_available']=df[f'lag_{k}'].notna().astype(int)
            df[f'region_lag_{k}_available']=df[f'lag_{k}_available']
    b=df.total_pickups.shift(1)
    df['avg_pickups']=b.ewm(alpha=.4).mean().round()
    for k in [24,168,336,720]:
        df[f'rolling_mean_{k}']=b.rolling(k,min_periods=1).mean()
        df[f'rolling_std_{k}']=b.rolling(k,min_periods=1).std().fillna(0)
        df[f'region_rolling_mean_{k}']=df[f'rolling_mean_{k}']
        df[f'region_rolling_std_{k}']=df[f'rolling_std_{k}']
    dates=df.index.normalize()
    df['holiday_name']=[HOLIDAYS.get(t,'none') for t in dates]
    df['is_holiday']=dates.isin(HOLIDAYS).astype(int)
    df['days_to_holiday']=[min([(h-t).days for h in HOLIDAYS if h>=t],default=99) for t in dates]
    df['days_after_holiday']=[min([(t-h).days for h in HOLIDAYS if h<=t],default=99) for t in dates]
    for name,(a,z) in EVENTS.items(): df['is_'+name]=((dates>=a)&(dates<=z)).astype(int)
    for name,mask in [('weekend',df.is_weekend==1),('weekday',df.is_weekend==0),('holiday',df.is_holiday==1)]:
        df[f'rolling_mean_{name}_24']=b.where(mask).rolling(24,min_periods=1).mean().fillna(df.rolling_mean_24)
    missing=set(WEATHER)-set(weather)
    if missing: raise ValueError(f'Missing weather columns: {missing}')
    row=weather.reindex([ts.floor('h')])[WEATHER]
    if row.isna().any().any(): raise ValueError('Weather missing at forecast timestamp')
    target=df.iloc[[-1]].copy()
    for c in WEATHER: target[c]=row[c].iloc[0]
    for name,a,b in [('weekend_hour_interaction','hour','is_weekend'),('rain_x_peak_hour','is_raining','is_peak_hour'),('snow_x_night','is_snowing','is_night'),('holiday_x_hour','is_holiday','hour'),('temp_x_weekend','temperature_2m','is_weekend'),('storm_x_peak','is_storm_jonas','is_peak_hour'),('wind_x_rain','windspeed_10m','is_raining')]: target[name]=target[a]*target[b]
    missing=set(columns)-set(target)
    if missing: raise ValueError(f'Unsupported required model features: {sorted(missing)}')
    target=target[columns].copy()
    for c in columns:
        if c in ['region','holiday_name','temp_bin','weather_condition']: target[c]=target[c].astype('category')
        else: target[c]=pd.to_numeric(target[c],errors='raise').astype(float)
    if target.isna().any().any(): raise ValueError('Required feature values are missing')
    return target
