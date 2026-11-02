import asyncio, contextlib, hashlib, io, json, os, pathlib, sys, traceback
import joblib, numpy as np, pandas as pd, psutil

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT/'reports'/'baseline'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/'urbanflow'/'backend'))

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()

async def main():
    if (OUT/'original.json').exists():
        print('Preserved baseline already exists; leaving it unchanged. See reports/baseline/original.json.')
        return
    print('baseline: importing service',flush=True)
    import services.forecasting_service as service
    artifact = ROOT/'urbanflow/backend/data/lgbm_taxi_pipeline.pkl'
    pipeline = joblib.load(artifact)
    print('baseline: loaded model',flush=True)
    original_params=pipeline['model'].get_params().copy()
    pipeline['model'].set_params(n_jobs=4)
    seed = pd.read_csv(ROOT/'urbanflow/backend/data/historical_seed_data.csv', index_col=0, parse_dates=True)
    print('baseline: loaded seed',flush=True)
    weather_cols = ['temperature_2m','apparent_temperature','precipitation','snowfall','snow_depth','windspeed_10m','cloudcover','relativehumidity_2m','is_raining','is_snowing','is_heavy_weather','is_freezing','is_storm_jonas','temp_bin','weather_condition']
    weather = seed[weather_cols].groupby(level=0).first()
    async def cached_weather(*args): return weather
    service.fetch_weather_context = cached_weather
    result = dict(artifact_sha256=digest(artifact), experiment_artifact_sha256=digest(ROOT/'lgbm_taxi_pipeline.pkl'),
        keys=list(pipeline), features=pipeline['feature_columns'], categorical=pipeline['categorical_columns'],
        model_params=original_params, baseline_execution_threads=4, model_categories=pipeline['model'].booster_.pandas_categorical,
        seed_min=str(seed.index.min()), seed_max=str(seed.index.max()), seed_rows=len(seed),
        dtypes=seed.dtypes.astype(str).to_dict(), cpu=os.cpu_count(), ram=dict(psutil.virtual_memory()._asdict()), forecasts={})
    for r in [0, 15, 29]:
        print('baseline: forecasting',r,flush=True)
        ts=seed.index.max()+pd.Timedelta(minutes=15)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                f,w=await service.generate_recursive_forecast(r,ts,12,seed,pipeline)
            result['forecasts'][str(r)] = {'origin':str(ts),'forecast':f,'weather':w}
        except Exception:
            result['forecasts'][str(r)]={'error':traceback.format_exc()}
    result['weather_note']='Baseline uses weather embedded in seed; no claim of live weather or deployment-realistic accuracy.'
    (OUT/'original.json').write_text(json.dumps(result,indent=2,default=str))
    joblib.dump(pipeline, OUT/'original_pipeline.joblib')
    print(json.dumps({k:v for k,v in result.items() if k not in ['dtypes','model_params','ram','forecasts','model_categories']},indent=2))
    print(result['forecasts'])

asyncio.run(main())
