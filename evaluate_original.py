"""Replay original notebook one-step evaluation, plus original API recursion.

Uses historical observed/reanalysis weather as an oracle sensitivity experiment.
Legacy target/features/preprocessing differ: NEVER interpret as a fair improvement.
"""
import ast, asyncio, contextlib, importlib.util, io, json, pathlib, sys, time
import joblib,numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'urbanflow'),str(ROOT/'urbanflow/backend')]
from ml.train import metrics
from weather_service import fetch_weather_context

def original_builder():
    nb=json.loads((ROOT/'lightgbm.ipynb').read_text(encoding='utf-8'))
    source=''.join(nb['cells'][9]['source'])
    tree=ast.parse(source)
    names={'impute_lag','impute_lag_grouped','build_holiday_features','build_features'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'NYC_HOLIDAYS_2016','NYC_EVENTS_2016'} for t in n.targets)]
    env=dict(pd=pd,np=np)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_notebook_functions','exec'),env)
    return env['build_features']

async def main():
    cache=ROOT/'artifacts/weather_2016.parquet'
    try:
        if cache.exists(): w=pd.read_parquet(cache)
        else:
            w=await fetch_weather_context(pd.Timestamp('2016-01-01'),pd.Timestamp('2016-03-31 23:00'),historical=True)
            w.to_parquet(cache)
    except Exception as exc:
        (ROOT/'reports/baseline/original_evaluation_blocker.txt').write_text(repr(exc))
        raise
    pipeline=joblib.load(ROOT/'urbanflow/backend/data/lgbm_taxi_pipeline.pkl')
    model=pipeline['model'];model.set_params(n_jobs=4)
    test=pd.read_csv(ROOT/'data/test_new.csv',index_col=0,parse_dates=True)
    with contextlib.redirect_stdout(io.StringIO()): df=original_builder()(test,w)
    x=df[pipeline['feature_columns']].copy()
    for c in pipeline['categorical_columns']: x[c]=x[c].astype('category')
    p=model.predict(x)
    result=dict(original_notebook_one_step=metrics(df.total_pickups,p),
        rows=len(df),start=str(df.index.min()),end=str(df.index.max()),
        disclaimer='Exact stored notebook feature function replay on existing test_new.csv; historical weather reanalysis downloaded now, source API revisions and notebook state may change numbers. One-step test truth enters lag features. Not a recursive deployment metric.')
    (ROOT/'artifacts/original_metrics.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    # Original deployed feature logic, without overwriting historical targets or using future demand.
    spec=importlib.util.spec_from_file_location('baseline_features',ROOT/'reports/baseline/feature_engineering_original.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    panel=pd.read_parquet(ROOT/'artifacts/counts.parquet')
    origins=pd.to_datetime(['2016-03-01 00:00','2016-03-05 23:00','2016-03-10 06:00','2016-03-18 17:00','2016-03-26 23:00','2016-03-30 12:00'])
    rows=[];timings=[]
    for origin in origins:
        print('original recursive',origin,flush=True)
        started=time.perf_counter()
        contexts={r:g.set_index('datetime')[['region','total_pickups']].loc[lambda z:z.index<origin].tail(721).copy() for r,g in panel.groupby('region')}
        actual=panel.pivot(index='datetime',columns='region',values='total_pickups')
        for step in range(48):
            ts=origin+pd.Timedelta(minutes=15*step); xs=[]
            for r,g in contexts.items():
                g.loc[ts]=[r,np.nan]
                with contextlib.redirect_stdout(io.StringIO()): xx=module.build_features(g,ts,w,pipeline['feature_columns'])
                xs.append(xx)
            x=pd.concat(xs)
            for c in pipeline['categorical_columns']: x[c]=x[c].astype('category')
            pred=np.maximum(0,model.predict(x))
            for r,v in enumerate(pred):
                contexts[r].at[ts,'total_pickups']=v
                rows.append(('original_api',str(origin),step+1,r,float(actual.loc[ts,r]),float(v)))
        timings.append(time.perf_counter()-started)
        pd.DataFrame(rows,columns=['model','origin','step','region','actual','prediction']).to_parquet(ROOT/'artifacts/original_recursive_predictions.parquet',index=False)
    d=pd.DataFrame(rows,columns=['model','origin','step','region','actual','prediction'])
    result['recursive']=metrics(d.actual,d.prediction)
    result['horizons']={str(h):metrics(d[d.step<=h].actual,d[d.step<=h].prediction) for h in [1,4,12,24,48]}
    result['seconds_per_48step_30region']=float(np.mean(timings))
    result['recursive_disclaimer']='Original deployed features, region-local inference, missing feature zero fallback retained. Context restricted to last 721 observed intervals then predictions. Weather oracle from historical reanalysis. Original model trained through February; retrained candidates through February 14. Descriptive comparison only.'
    (ROOT/'artifacts/original_metrics.json').write_text(json.dumps(result,indent=2))

asyncio.run(main())
