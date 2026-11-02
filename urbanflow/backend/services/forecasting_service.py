import pandas as pd
import numpy as np
from errors import HistoryUnavailable, InferenceFailure
from weather_service import fetch_weather_context
from ml.features import COLUMNS, recursive
from ml.legacy import build_legacy

def local_timestamp(value):
    ts=pd.Timestamp(value)
    if pd.isna(ts): raise HistoryUnavailable('Datetime is required')
    if ts.tzinfo is not None: ts=ts.tz_convert('America/New_York')
    # Source trip times are NY wall clock. Ambiguous/nonexistent origins are rejected.
    try:
        if ts.tzinfo is None: ts=ts.tz_localize('America/New_York',ambiguous='raise',nonexistent='raise')
    except Exception as exc: raise HistoryUnavailable('Ambiguous or nonexistent New York local time') from exc
    if ts.minute%15 or ts.second or ts.microsecond: raise HistoryUnavailable('Origin must align to a 15-minute interval')
    return ts

async def generate_recursive_forecast(region,start_datetime,horizon,seed_data,pipeline,mode='historical'):
    ts=local_timestamp(start_datetime)
    if seed_data is None or seed_data.empty: raise HistoryUnavailable('Historical observations unavailable')
    g=seed_data.loc[seed_data.region.astype(int)==region,['region','total_pickups']].copy().sort_index()
    if g.index.tz is None:
        try: g.index=g.index.tz_localize('America/New_York',ambiguous='raise',nonexistent='raise')
        except Exception as exc: raise HistoryUnavailable('Historical clock timestamps are invalid; use validated observations') from exc
    if g.empty: raise HistoryUnavailable('No observations for this region')
    if g.index.has_duplicates: raise HistoryUnavailable('Duplicate historical timestamps')
    if mode=='realtime' and ts!=g.index.max()+pd.Timedelta(minutes=15):
        raise HistoryUnavailable('Realtime origin must immediately follow the last actual observation')
    if mode=='realtime' and abs(pd.Timestamp.now(tz='America/New_York')-ts)>pd.Timedelta(hours=1):
        raise HistoryUnavailable('Recent genuine taxi observations are unavailable; use historical mode')
    if ts>g.index.max()+pd.Timedelta(minutes=15): raise HistoryUnavailable('Origin is beyond the available observations; historical dates have not been shifted')
    # Remove ALL future targets before computing features, including original target row.
    g=g.loc[g.index<ts]
    if len(g)<720 or g.index[-1]!=ts-pd.Timedelta(minutes=15): raise HistoryUnavailable('At least 720 consecutive preceding 15-minute observations required')
    if not g.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all() or not np.isfinite(g.total_pickups).all():
        raise HistoryUnavailable('Historical context has gaps or missing observations; no implicit imputation')
    weather_context={'temperature':None,'condition':'Not used by this model','source':'not_used'}
    if pipeline.get('schema_version')==2:
        if pipeline['feature_columns']!=COLUMNS or pipeline.get('categorical_columns')!=[]: raise InferenceFailure('Candidate feature schema mismatch')
        try: values=recursive(pipeline['model'],g.total_pickups.to_numpy()[:,None],ts,[region],horizon)[:,0]
        except Exception as exc: raise InferenceFailure('Model could not produce valid predictions') from exc
    else:
        wall=g.index.tz_localize(None)
        end=ts+pd.Timedelta(minutes=15*(horizon-1))
        if not wall.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all() or ts.utcoffset()!=end.utcoffset():
            raise HistoryUnavailable('Legacy model cannot use context or horizons across DST; select a timezone-aware V2 candidate')
        ts=ts.tz_localize(None)
        g.index=g.index.tz_localize(None)
        weather=await fetch_weather_context(ts,ts+pd.Timedelta(minutes=15*(horizon-1)),historical=mode=='historical')
        values=[]
        for step in range(horizon):
            at=ts+pd.Timedelta(minutes=15*step)
            g.loc[at]=[region,np.nan]
            try:
                x=build_legacy(g,at,weather,pipeline['feature_columns'])
                value=float(pipeline['model'].predict(x)[0])
                if not np.isfinite(value): raise ValueError('Nonfinite prediction')
            except Exception as exc: raise InferenceFailure('Original model feature reconstruction or inference failed') from exc
            value=max(0,value); values.append(value); g.at[at,'total_pickups']=value
        row=weather.loc[ts.floor('h')]
        weather_context=dict(temperature=float(row.temperature_2m),condition=str(row.weather_condition),source=weather.attrs.get('source','unknown'))
    forecasts=[]
    for i,p in enumerate(values):
        at=ts+pd.Timedelta(minutes=15*i)
        if at.tzinfo is None: at=at.tz_localize('America/New_York',ambiguous='raise',nonexistent='raise')
        forecasts.append({'datetime':at.isoformat(),'prediction':round(float(p),2)})
    return forecasts,weather_context
