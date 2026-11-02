import datetime as dt
import logging, os, pathlib, sys, time
from contextlib import asynccontextmanager
from typing import Literal
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from services.forecasting_service import generate_recursive_forecast, local_timestamp
from errors import HistoryUnavailable,WeatherUnavailable,InferenceFailure

logger=logging.getLogger('urbanflow')
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')
BASE=pathlib.Path(__file__).resolve().parent
PIPELINE={}; SEED_DATA=None
MODEL_PATH=pathlib.Path(os.environ.get('URBANFLOW_MODEL',str(BASE/'data/lgbm_taxi_pipeline.pkl')))
SEED_PATH=pathlib.Path(os.environ.get('URBANFLOW_SEED',str(BASE/'data/historical_seed_data.csv')))

@asynccontextmanager
async def lifespan(app):
    global PIPELINE,SEED_DATA
    PIPELINE=joblib.load(MODEL_PATH)
    names=PIPELINE.get('feature_columns',[])
    if not names or len(names)!=len(set(names)):
        raise RuntimeError('Invalid model feature schema')
    model=PIPELINE['model']
    trained_names=getattr(model,'feature_name_',getattr(model,'feature_names_in_',None))
    if trained_names is not None and list(trained_names)!=names:
        raise RuntimeError('Serialized feature order differs from trained model')
    if SEED_PATH.suffix=='.parquet':
        SEED_DATA=pd.read_parquet(SEED_PATH).set_index('datetime')
    else: SEED_DATA=pd.read_csv(SEED_PATH,index_col=0,parse_dates=True)
    if SEED_DATA.index.tz is None:
        SEED_DATA.index=SEED_DATA.index.tz_localize('America/New_York',ambiguous='raise',nonexistent='raise')
    logger.info('loaded model=%s rows=%d source_end=%s',MODEL_PATH,len(SEED_DATA),SEED_DATA.index.max())
    yield

app=FastAPI(title='UrbanFlow AI Engine',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['GET','POST'],allow_headers=['Content-Type'])

class PredictRequest(BaseModel):
    region:int=Field(ge=0,le=29,strict=True)
    datetime:dt.datetime
    horizon:int=Field(default=12,ge=1,le=48,strict=True)
    mode:Literal['historical','realtime']='historical'

@app.get('/health')
def health():
    ready=bool(PIPELINE) and SEED_DATA is not None and not SEED_DATA.empty
    return {'status':'ok' if ready else 'unavailable','model_loaded':bool(PIPELINE),'observations_end':str(SEED_DATA.index.max()) if ready else None,'timezone':'America/New_York','default_mode':'historical'}

@app.post('/predict')
async def predict_demand(request:PredictRequest):
    started=time.perf_counter()
    try:
        if not PIPELINE: raise HTTPException(503,'Model unavailable')
        ts=local_timestamp(request.datetime)
        for step in range(request.horizon):
            local_timestamp(ts+pd.Timedelta(minutes=15*step))
        # Each request has isolated context. Execute CPU-bound inference off the event loop.
        def compute():
            import asyncio
            return asyncio.run(generate_recursive_forecast(request.region,ts,request.horizon,SEED_DATA,PIPELINE,request.mode))
        forecast,weather=await run_in_threadpool(compute)
        source=SEED_DATA.copy(deep=False)
        if source.index.tz is None: source.index=source.index.tz_localize('America/New_York',ambiguous='raise',nonexistent='raise')
        past=source.loc[source.index<ts]
        g=past[past.region.astype(int)==request.region].sort_index().tail(192)
        history=[dict(datetime=t.isoformat(),actual=float(v)) for t,v in g.total_pickups.items()]
        heatmap=[]
        for r,group in past.groupby('region'):
            demand=float(group.sort_index().total_pickups.tail(24).mean())
            heatmap.append(dict(id=int(r),name=f'NYC Sector {int(r):02d}',demand=round(demand),level='High' if demand>750 else 'Medium' if demand>400 else 'Low'))
        model=PIPELINE['model']; importance=[]
        if hasattr(model,'feature_importances_'):
            scores=model.booster_.feature_importance(importance_type='gain') if hasattr(model,'booster_') else model.feature_importances_
            importance=sorted([dict(name=n,score=float(v)) for n,v in zip(PIPELINE['feature_columns'],scores)],key=lambda f:f['score'],reverse=True)[:10]
        return dict(forecast=forecast,history=history,weather=weather,heatmap=heatmap,feature_importance=importance,alerts=[],
            metadata=dict(mode=request.mode,timezone='America/New_York',schema_version=PIPELINE.get('schema_version','legacy_repaired_semantics'),
                heatmap_source='historical_observations_before_origin',confidence_intervals='unavailable',latency_ms=round((time.perf_counter()-started)*1000,2)))
    except HTTPException: raise
    except HistoryUnavailable as exc: raise HTTPException(422,detail=str(exc)) from exc
    except WeatherUnavailable as exc:
        logger.warning('weather failure: %s',exc); raise HTTPException(503,detail='Required weather is unavailable') from exc
    except InferenceFailure as exc:
        logger.exception('inference failure'); raise HTTPException(503,detail='Model inference failed; consult server logs') from exc
    except Exception as exc:
        logger.exception('unexpected prediction failure'); raise HTTPException(500,detail='Unexpected internal error') from exc
