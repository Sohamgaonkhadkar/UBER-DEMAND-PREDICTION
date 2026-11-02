"""Run from workspace: .venv/Scripts/python urbanflow/ml/train.py.

Bounded local experiment: full regional aggregate, 3 Optuna trials, 4 threads.
Regions reuse original upstream clustering; clustering provenance is not re-fitted.
"""
import hashlib, json, os, pathlib, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ml.features import COLUMNS, training_features, recursive, chronological_split

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT/'artifacts'; REP = ROOT/'reports'
ART.mkdir(exist_ok=True); (ART/'candidates').mkdir(exist_ok=True)

def metrics(y, p):
    y,p=np.asarray(y).ravel(),np.asarray(p).ravel()
    return dict(MAE=float(mean_absolute_error(y,p)), RMSE=float(np.sqrt(mean_squared_error(y,p))),
        WAPE=float(100*np.abs(y-p).sum()/np.abs(y).sum()) if np.abs(y).sum() else None,
        R2=float(r2_score(y,p)) if len(y)>1 and np.var(y)>0 else None)

def rebuild():
    dest=ART/'counts_ny.parquet'
    if dest.exists(): return pd.read_parquet(dest)
    old=ART/'counts.parquet'
    if old.exists():
        panel=pd.read_parquet(old)
        local=panel.datetime.dt.tz_localize('America/New_York',ambiguous='raise',nonexistent='NaT')
        invalid=local.isna()
        panel.loc[invalid].to_csv(ART/'quarantined_dst_intervals.csv',index=False)
        metadata=json.loads((ART/'dataset_metadata.json').read_text())
        metadata['quarantined_nonexistent_clock_trips']=int(panel.loc[invalid,'total_pickups'].sum())
        metadata['quarantined_intervals']=int(invalid.sum())
        metadata['timezone']='America/New_York aware; nonexistent source clock hour quarantined, not reassigned. Timeline advances by true 15-minute elapsed intervals.'
        panel=panel.loc[~invalid].copy(); panel['datetime']=local[~invalid]
        panel.to_parquet(dest,index=False)
        metadata['rows']=len(panel);metadata['zero_intervals']=int(panel.total_pickups.eq(0).sum())
        (ART/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2))
        return panel
    counts=None; n=0; sha=hashlib.sha256()
    source=ROOT/'data/time_series.csv'
    with open(source,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): sha.update(b)
    for chunk in pd.read_csv(source,chunksize=1_000_000):
        chunk['datetime']=pd.to_datetime(chunk.tpep_pickup_datetime).dt.floor('15min')
        c=chunk.groupby(['datetime','region']).size()
        counts=c if counts is None else counts.add(c,fill_value=0)
        n+=len(chunk)
        print('aggregated trips',n,flush=True)
    grid=pd.MultiIndex.from_product([pd.date_range('2016-01-01','2016-04-01',freq='15min',inclusive='left'),range(30)],names=['datetime','region'])
    outside=counts.index.difference(grid)
    if len(outside): raise ValueError(f'{len(outside)} out-of-period keys; investigate before training')
    panel=counts.reindex(grid,fill_value=0).astype('int64').rename('total_pickups').reset_index()
    panel.to_parquet(dest,index=False)
    (ART/'dataset_metadata.json').write_text(json.dumps(dict(source=str(source),sha256=sha.hexdigest(),trips=n,
        rows=len(panel),zero_intervals=int(panel.total_pickups.eq(0).sum()),
        policy='Zero means no retained trips in a complete source interval. No arbitrary epsilon. Source completeness is assumed, not independently certified.',
        timezone='NYC source wall clock; DST ambiguity retained as a known source limitation. Do not claim UTC elapsed continuity.'),indent=2))
    return rebuild()

def backtest(model,panel,origins,name):
    wide=panel.pivot(index='datetime',columns='region',values='total_pickups')
    records=[]; elapsed=[]
    for origin in origins:
        t=time.perf_counter()
        p=recursive(model,wide.loc[wide.index<origin].to_numpy(),origin,list(wide.columns),48)
        elapsed.append(time.perf_counter()-t)
        actual=wide.reindex(pd.date_range(origin,periods=48,freq='15min')).to_numpy()
        if not np.isfinite(actual).all(): raise ValueError('Missing evaluation targets')
        for step in range(48):
            for r in range(30):
                records.append((name,str(origin),step+1,r,float(actual[step,r]),float(p[step,r])))
    df=pd.DataFrame(records,columns=['model','origin','step','region','actual','prediction'])
    return df,dict(seconds_per_48step_30region=float(np.mean(elapsed)))

def main():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    import optuna, mlflow
    panel=rebuild()
    panel,x=training_features(panel); y=panel.total_pickups
    train,val,test=chronological_split(panel)
    valid=x.notna().all(axis=1)
    train &= valid; val &= valid; test &= valid
    splits={k:dict(start=str(panel.loc[m,'datetime'].min()),end=str(panel.loc[m,'datetime'].max()),rows=int(m.sum())) for k,m in zip(['train','validation','test'],[train,val,test])}
    (ART/'splits.json').write_text(json.dumps(splits,indent=2))
    val_origins=pd.to_datetime(['2016-02-16 00:00','2016-02-20 12:00','2016-02-27 23:00']).tz_localize('America/New_York')
    test_origins=pd.to_datetime(['2016-03-01 00:00','2016-03-05 23:00','2016-03-10 06:00','2016-03-18 17:00','2016-03-26 23:00','2016-03-30 12:00']).tz_localize('America/New_York')
    def objective(trial):
        m=LGBMRegressor(n_estimators=trial.suggest_int('n_estimators',100,250,step=75),
            num_leaves=trial.suggest_categorical('num_leaves',[15,31,63]),learning_rate=.05,
            n_jobs=4,random_state=42,verbosity=-1)
        m.fit(x[train],y[train])
        d,_=backtest(m,panel,val_origins,'tuning')
        return metrics(d.actual,d.prediction)['WAPE']
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=3)
    study.trials_dataframe().to_csv(ART/'optuna_trials.csv',index=False)
    models=dict(seasonal_naive=None,
        lightgbm=LGBMRegressor(**study.best_params,learning_rate=.05,n_jobs=4,random_state=42,verbosity=-1),
        xgboost=XGBRegressor(n_estimators=200,max_depth=6,learning_rate=.05,n_jobs=4,random_state=42,tree_method='hist'),
        catboost=CatBoostRegressor(iterations=200,depth=6,learning_rate=.05,thread_count=4,random_seed=42,verbose=False,allow_writing_files=False),
        random_forest=RandomForestRegressor(n_estimators=60,max_depth=18,min_samples_leaf=5,n_jobs=4,random_state=42))
    mlflow.set_tracking_uri('sqlite:///'+str(ROOT/'local_tracking.db'))
    mlflow.set_experiment('urbanflow_local_repair')
    results={}; rec=[]; validation=[]
    for name,model in models.items():
        print('training',name,flush=True)
        with mlflow.start_run(run_name=name):
            started=time.perf_counter()
            if model is not None: model.fit(x[train],y[train])
            duration=time.perf_counter()-started
            vd,vl=backtest(model,panel,val_origins,name)
            validation.append(vd)
            td,tl=backtest(model,panel,test_origins,name); rec.append(td)
            one=panel.loc[test,['datetime','region','total_pickups']].copy()
            pred=x.loc[test,'lag_672'].to_numpy() if model is None else np.maximum(0,model.predict(x[test]))
            result=dict(one_step=metrics(y[test],pred), validation_recursive=metrics(vd.actual,vd.prediction),
                test_recursive=metrics(td.actual,td.prediction),latency=tl,training_seconds=duration,
                horizons={str(h):metrics(td[td.step<=h].actual,td[td.step<=h].prediction) for h in [1,4,12,24,48]})
            target=ART/'candidates'/f'{name}_v2.joblib'
            if model is not None:
                joblib.dump(dict(model=model,feature_columns=COLUMNS,categorical_columns=[],schema_version=2,
                    target='unaltered 15-minute retained trip counts',weather_policy='not used',splits=splits),target)
                result['model_bytes']=target.stat().st_size
            mlflow.log_params({'features':','.join(COLUMNS),'schema_version':2,'dataset':'counts_ny.parquet',
                'dataset_sha256':json.loads((ART/'dataset_metadata.json').read_text())['sha256'],
                'hyperparameters':str(model.get_params() if model is not None else {'period':672})})
            mlflow.log_metrics({f'{group}_{k}':v for group in ['one_step','validation_recursive','test_recursive'] for k,v in result[group].items() if v is not None})
            mlflow.log_metric('training_seconds',duration)
            mlflow.log_dict(json.loads((ART/'dataset_metadata.json').read_text()),'dataset_metadata.json')
            mlflow.log_dict(splits,'splits.json')
            if model is not None: mlflow.log_artifact(str(target))
            results[name]=result
            (ART/'results.json').write_text(json.dumps(results,indent=2))
            print(name,result,flush=True)
    pd.concat(rec).to_parquet(ART/'recursive_predictions.parquet',index=False)
    pd.concat(validation).to_parquet(ART/'validation_recursive_predictions.parquet',index=False)
    (ART/'candidates/feature_schema.json').write_text(json.dumps(dict(version=2,columns=COLUMNS,dtypes={c:'float64' for c in COLUMNS},categorical=[],missing='Reject incomplete context; exclude first 720 training intervals per region'),indent=2))
    print('completed',flush=True)

if __name__=='__main__': main()
