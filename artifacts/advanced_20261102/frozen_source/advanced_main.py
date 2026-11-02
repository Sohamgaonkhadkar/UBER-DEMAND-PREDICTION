"""Opt-in advanced local engine. Existing main.py and default artifacts are intact.

Run uvicorn advanced_main:app --app-dir urbanflow/backend with URBANFLOW_MODEL
and URBANFLOW_SEED explicitly set to the separate advanced artifacts.
"""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
import main as base
from errors import HistoryUnavailable,InferenceFailure
from services.forecasting_service import local_timestamp
from ml.advanced.core import HISTORY

original_forecast=base.generate_recursive_forecast
original_predict=base.predict_demand

async def advanced_forecast(region,start_datetime,horizon,seed_data,pipeline,mode='historical'):
    if pipeline.get('schema_version')!=3:
        return await original_forecast(region,start_datetime,horizon,seed_data,pipeline,mode)
    ts=local_timestamp(start_datetime)
    if pipeline.get('training_end') and ts<pd.Timestamp(pipeline['training_end']):
        raise HistoryUnavailable('Origin precedes this candidate training cutoff; use the validation-era artifact for earlier historical backtests')
    if seed_data is None or seed_data.empty:raise HistoryUnavailable('Historical observations unavailable')
    source=seed_data[['region','total_pickups']].copy()
    if source.index.tz is None:
        try:source.index=source.index.tz_localize('America/New_York',ambiguous='raise',nonexistent='raise')
        except Exception as exc:raise HistoryUnavailable('Invalid historical timezone context') from exc
    if ts>source.index.max()+pd.Timedelta('15min'):raise HistoryUnavailable('Origin exceeds genuine observed demand; historical timestamps are not shifted')
    if mode=='realtime' and (ts!=source.index.max()+pd.Timedelta('15min') or abs(pd.Timestamp.now(tz='America/New_York')-ts)>pd.Timedelta('1h')):
        raise HistoryUnavailable('Recent genuine observations required for realtime forecasts')
    # Future truth is excluded before any pivot, feature computation or model call.
    context=source.loc[(source.index<ts)&(source.index>=ts-pd.Timedelta(minutes=15*HISTORY))]
    if context.reset_index().duplicated([context.index.name or 'index','region']).any():raise HistoryUnavailable('Duplicate historical observations')
    wide=context.reset_index().pivot(index=context.index.name or 'index',columns='region',values='total_pickups')
    required=pd.date_range(ts-pd.Timedelta(minutes=15*HISTORY),periods=HISTORY,freq='15min')
    if set(wide.columns)!=set(range(30)):raise HistoryUnavailable('Advanced joint inference requires all 30 regions')
    wide=wide.reindex(index=required,columns=range(30))
    if not np.isfinite(wide.to_numpy(dtype=float)).all() or (wide.to_numpy()<0).any():raise HistoryUnavailable('Advanced forecasts require 1536 consecutive genuine observations in every region')
    if pipeline.get('horizon')!=48 or pipeline.get('timezone')!='America/New_York':raise InferenceFailure('Advanced artifact schema mismatch')
    try:
        times=pd.date_range(required[0],periods=HISTORY+horizon,freq='15min')
        p=pipeline['model'].forecast(wide.to_numpy(dtype=np.float32),times,np.array([HISTORY]),horizon)
        if p.shape!=(1,horizon,30) or not np.isfinite(p).all() or np.any(p<0):raise ValueError('Invalid advanced prediction')
    except Exception as exc:raise InferenceFailure('Advanced model inference failed') from exc
    result=[dict(datetime=(ts+pd.Timedelta(minutes=15*k)).isoformat(),prediction=round(float(p[0,k,region]),2)) for k in range(horizon)]
    return result,dict(temperature=None,condition='Not used by this model',source='not_used')

async def enhanced_predict(request:base.PredictRequest):
    result=await original_predict(request)
    if base.PIPELINE.get('schema_version')==3:
        result['metadata']['forecast_strategy']=base.PIPELINE.get('forecast_strategy')
        result['metadata']['candidate_id']=base.PIPELINE.get('candidate_id')
    return result

base.generate_recursive_forecast=advanced_forecast
for route in base.app.routes:
    if getattr(route,'path',None)=='/predict':
        route.endpoint=enhanced_predict;route.dependant.call=enhanced_predict
app=base.app
