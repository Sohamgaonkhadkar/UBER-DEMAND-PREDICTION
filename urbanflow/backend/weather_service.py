"""Weather with explicit provenance, full-horizon coverage and expiring cache.

API references: https://open-meteo.com/en/docs and /en/docs/historical-weather-api
Historical archive is reanalysis, NOT an archived forecast available in 2016.
"""
import asyncio, time
import httpx
import numpy as np
import pandas as pd
from errors import WeatherUnavailable

CACHE = {}; TTL = 1800; MAX_CACHE = 32
FIELDS=['temperature_2m','apparent_temperature','precipitation','snowfall','snow_depth','wind_speed_10m','cloud_cover','relative_humidity_2m','weather_code']

def derive_weather(df):
    df=df.rename(columns={'wind_speed_10m':'windspeed_10m','cloud_cover':'cloudcover','relative_humidity_2m':'relativehumidity_2m','weather_code':'weathercode'}).copy()
    cols=['temperature_2m','apparent_temperature','precipitation','snowfall','snow_depth','windspeed_10m','cloudcover','relativehumidity_2m','weathercode']
    if set(cols)-set(df): raise WeatherUnavailable('Weather provider omitted required fields')
    if not np.isfinite(df[cols].to_numpy(dtype=float)).all(): raise WeatherUnavailable('Weather provider returned missing values')
    df['is_raining']=(df.precipitation>0).astype(int)
    df['is_snowing']=(df.snowfall>0).astype(int)
    df['is_heavy_weather']=((df.precipitation>5)|(df.snowfall>2)).astype(int)
    df['is_freezing']=(df.temperature_2m<0).astype(int)
    df['temp_bin']=pd.cut(df.temperature_2m,[-np.inf,0,10,np.inf],labels=['cold','mild','warm']).astype(str)
    code=df.weathercode.astype(int)
    df['weather_condition']=np.select([code.isin([0,1]),code.isin([2,3]),code.between(51,67)|code.between(80,82),code.between(71,77)|code.between(85,86),code.isin([95,96,99])],['clear','cloudy','rain','snow','storm'],default='other')
    df['is_storm_jonas']=((df.index>=pd.Timestamp('2016-01-22'))&(df.index<=pd.Timestamp('2016-01-25'))).astype(int)
    return df

async def fetch_weather_context(start, end=None, historical=False):
    start,end=pd.Timestamp(start),pd.Timestamp(end if end is not None else start)
    key=(str(start.date()),str(end.date()),historical)
    now=time.monotonic()
    if key in CACHE and now-CACHE[key][0]<TTL: return CACHE[key][1].copy()
    url='https://archive-api.open-meteo.com/v1/archive' if historical else 'https://api.open-meteo.com/v1/forecast'
    params=dict(latitude=40.7128,longitude=-74.0060,start_date=key[0],end_date=key[1],hourly=','.join(FIELDS),timezone='America/New_York')
    last=None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response=await client.get(url,params=params)
                response.raise_for_status(); payload=response.json()
            df=pd.DataFrame(payload['hourly']); df.index=pd.to_datetime(df.pop('time'))
            df=derive_weather(df)
            required=pd.date_range(start.floor('h'),end.floor('h'),freq='h')
            if df.index.has_duplicates or not required.isin(df.index).all(): raise WeatherUnavailable('Weather horizon is incomplete')
            df.attrs['source']='historical_reanalysis_not_origin_available' if historical else 'weather_forecast'
            if len(CACHE)>=MAX_CACHE: CACHE.pop(next(iter(CACHE)))
            CACHE[key]=(now,df)
            return df.copy()
        except (httpx.HTTPError,KeyError,ValueError,WeatherUnavailable) as exc:
            last=exc
            if attempt<2: await asyncio.sleep(2**attempt)
    raise WeatherUnavailable('Weather unavailable or invalid after 3 attempts') from last
