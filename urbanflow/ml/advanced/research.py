"""Staged, validation-only experiments. April data is never imported here."""
import dataclasses,gc,hashlib,json,os,pathlib,time
os.environ.setdefault('OMP_NUM_THREADS','4')
import joblib,numpy as np,pandas as pd
from lightgbm import LGBMRegressor,early_stopping,log_evaluation
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.base import clone
import mlflow,optuna
from .protocol import *
from .core import *

class Research:
    def __init__(self):
        initialize()
        self.panel=pd.read_parquet(ROOT/'artifacts/counts_ny.parquet')
        self.wide=self.panel.pivot(index='datetime',columns='region',values='total_pickups')
        self.times=self.wide.index;self.y=self.wide.to_numpy(dtype=np.float32)
        self.cut=pd.Timestamp('2016-03-01',tz=TZ)-pd.Timedelta('2h')
        self.end=int(self.times.searchsorted(self.cut));self.scales=np.maximum(1,self.y[:self.end].mean(axis=0))
        geo=json.loads((OUT/'geography.json').read_text());self.neighbors=neighbors_from_centers(geo)
        self.origins=origins('2016-03-01','2016-04-01');self.pos=self.times.get_indexer(self.origins)
        self.truth=np.stack([self.y[o:o+48] for o in self.pos])
        self.records=json.loads((OUT/'leaderboard.json').read_text()) if (OUT/'leaderboard.json').exists() else []
        self.weather=pd.read_parquet(ROOT/'artifacts/weather_2016.parquet')[['temperature_2m','apparent_temperature','precipitation','snowfall','windspeed_10m','relativehumidity_2m']]
        self.cache=None;self.cache_key=None
        mlflow.set_tracking_uri('sqlite:///'+str(OUT/'tracking.db'))
        mlflow.set_experiment('urbanflow_advanced_validation')

    def rows(self,config,strategy='recursive'):
        key=hashlib.sha256(json.dumps(dict(config=asdict(config),strategy=strategy,end=str(self.cut)),sort_keys=True).encode()).hexdigest()[:14]
        if self.cache_key==key:return self.cache
        self.cache=None;gc.collect()
        folder=OUT/'features';folder.mkdir(exist_ok=True)
        file=folder/(key+'.parquet')
        if file.exists():
            df=pd.read_parquet(file);y=df.pop('__target').to_numpy();ts=pd.DatetimeIndex(pd.to_datetime(df.pop('__timestamp'),utc=True)).tz_convert(TZ);x=df
        else:
            if strategy=='direct':positions=np.arange(HISTORY,self.end-47,29);leads=tuple(range(1,49))
            else:positions=np.arange(HISTORY,self.end);leads=(1,)
            x,y,ts=training_rows(self.y,self.times,positions,config,self.scales,self.neighbors,leads=leads,weather=self.weather)
            ts=pd.DatetimeIndex(ts)
            data=x.copy();data['__target']=y;data['__timestamp']=ts
            data.to_parquet(file,index=False);del data
        self.cache_key=key;self.cache=(x,y,ts)
        return self.cache

    def train(self,architecture,config,transform='raw',strategy='recursive',params=None,region_specific=False):
        if (OUT/'final_test_results.json').exists():raise RuntimeError('This test period is spent; create a new protocol before further training')
        x,y,ts=self.rows(config,strategy)
        target=y.copy()
        if transform=='residual':target=target-x.seasonal_anchor.to_numpy()
        elif transform=='log1p':target=np.log1p(target)
        elif transform=='normalized':target=target/np.tile(self.scales,len(x)//30)
        if architecture=='lightgbm':
            p=dict(n_estimators=400,learning_rate=.04,num_leaves=31,min_child_samples=40,reg_lambda=1.,verbosity=-1,n_jobs=4,random_state=42)
            p.update(params or {});model=LGBMRegressor(**p)
        elif architecture=='xgboost':
            p=dict(n_estimators=400,learning_rate=.04,max_depth=6,min_child_weight=10,subsample=.9,colsample_bytree=.9,reg_lambda=3,tree_method='hist',n_jobs=4,random_state=42)
            p.update(params or {});model=XGBRegressor(**p)
        elif architecture=='catboost':
            p=dict(iterations=500,depth=7,learning_rate=.04,l2_leaf_reg=5,thread_count=4,random_seed=42,verbose=False,allow_writing_files=False)
            p.update(params or {});model=CatBoostRegressor(**p)
        else:raise ValueError(architecture)
        # Early stopping sees only an inner February holdout, never March or April.
        inner=ts<(self.cut-pd.Timedelta(days=7))
        if architecture=='lightgbm':
            model.fit(x.loc[inner],target[inner],eval_set=[(x.loc[~inner],target[~inner])],callbacks=[early_stopping(35,verbose=False),log_evaluation(0)])
            best=max(30,model.best_iteration_);model=clone(model).set_params(n_estimators=best)
        elif architecture=='xgboost':
            model.set_params(early_stopping_rounds=35)
            model.fit(x.loc[inner],target[inner],eval_set=[(x.loc[~inner],target[~inner])],verbose=False)
            best=max(30,model.best_iteration+1);model=clone(model).set_params(n_estimators=best,early_stopping_rounds=None)
        model.fit(x,target)
        bundle=ForecastModel(model,config,self.scales,strategy,transform,self.neighbors,weather=self.weather if config.weather else None)
        if region_specific:
            bundle.region_models=[]
            for r in range(30):
                m=clone(model);mask=x.region.to_numpy()==r;m.fit(x.loc[mask],target[mask]);bundle.region_models.append(m)
        return bundle

    def evaluate(self,bundle,positions=None):
        pos=self.pos if positions is None else positions
        started=time.perf_counter();pred=bundle.forecast(self.y,self.times,pos,48);elapsed=time.perf_counter()-started
        actual=np.stack([self.y[o:o+48] for o in pos])
        scores=metrics(actual,pred)
        scores['cumulative']={str(h):metrics(actual[:,:h],pred[:,:h]) for h in [1,4,12,24,48]}
        scores['at_lead']={str(h):metrics(actual[:,h-1],pred[:,h-1]) for h in [1,4,12,24,48]}
        scores['seconds_per_origin_30regions']=elapsed/len(pos)
        return scores,pred

    def run(self,name,architecture='lightgbm',config=None,transform='raw',strategy='recursive',params=None,region_specific=False,trial=None):
        if any(r['id']==name for r in self.records):return next(r for r in self.records if r['id']==name)
        if (OUT/'selection_freeze.json').exists():raise RuntimeError('Selection is frozen; new experiments are forbidden in this protocol')
        config=config or FeatureConfig();start=time.perf_counter()
        print('EXPERIMENT',name,architecture,transform,strategy,asdict(config),flush=True)
        bundle=self.train(architecture,config,transform,strategy,params,region_specific)
        train_seconds=time.perf_counter()-start
        if trial is not None:
            subset=self.pos[::4];score,_=self.evaluate(bundle,subset)
            trial.report(score['WAPE'],0)
            if trial.should_prune():raise optuna.TrialPruned()
        scores,pred=self.evaluate(bundle)
        record=dict(id=name,architecture=architecture,features=asdict(config),transform=transform,strategy=strategy,
            region_specific=region_specific,requested_params=params or {},fitted_params=bundle.model.get_params(),
            training_end=str(self.cut),validation_origins=len(self.pos),validation=scores,training_seconds=train_seconds,
            runtime_seconds=time.perf_counter()-start,eligible=not config.weather,status='completed')
        current=[r['validation']['WAPE'] for r in self.records if r.get('eligible') and r['architecture']==architecture and r['strategy']==strategy]
        worthwhile=not current or scores['WAPE']<min(current) or name.startswith('feature_') and scores['WAPE']<15
        if worthwhile:
            path=OUT/'models';path.mkdir(exist_ok=True)
            dest=path/(name+'.joblib');joblib.dump(bundle,dest)
            record['artifact']=str(dest.relative_to(ROOT));record['model_bytes']=dest.stat().st_size
        predpath=OUT/'validation_predictions';predpath.mkdir(exist_ok=True)
        np.savez_compressed(predpath/(name+'.npz'),prediction=pred)
        self.records.append(record)
        (OUT/'leaderboard.json').write_text(json.dumps(self.records,indent=2,default=str))
        pd.DataFrame([dict(id=r['id'],architecture=r['architecture'],strategy=r['strategy'],transform=r['transform'],eligible=r['eligible'],WAPE=r['validation']['WAPE'],runtime_seconds=r['runtime_seconds'],artifact=r.get('artifact')) for r in self.records]).sort_values('WAPE').to_csv(REPORT/'leaderboard.csv',index=False)
        with mlflow.start_run(run_name=name):
            mlflow.log_params(dict(architecture=architecture,features=json.dumps(asdict(config)),transform=transform,strategy=strategy,
                hyperparameters=json.dumps(record['fitted_params'],default=str),training_end=str(self.cut),validation_period='March 2016 development',origins=len(self.pos),horizon=48,dataset_sha256=sha(ROOT/'artifacts/counts_ny.parquet')))
            mlflow.log_metrics({f'validation_{k}':v for k,v in scores.items() if isinstance(v,(float,int)) and v is not None})
            mlflow.log_metric('training_seconds',train_seconds)
            if 'artifact' in record:mlflow.log_artifact(str(ROOT/record['artifact']))
        print('RESULT',name,round(scores['WAPE'],5),'bias',round(scores['bias_pct'],3),'seconds',round(record['runtime_seconds'],2),flush=True)
        del bundle;gc.collect()
        return record

def feature_stage():
    r=Research()
    configs={
      'core':FeatureConfig(seasonal=False,extended_lags=False,calendar=False),
      'extended_lags':FeatureConfig(seasonal=False,calendar=False),
      'seasonal':FeatureConfig(calendar=False),
      'calendar':FeatureConfig(),
      'statistics':FeatureConfig(statistics=True),
      'interactions':FeatureConfig(interactions=True),
      'spatial':FeatureConfig(spatial=True),
      'all':FeatureConfig(statistics=True,interactions=True,spatial=True),
    }
    for name,cfg in configs.items():r.run('feature_'+name,config=cfg)
    best=min([a for a in r.records if a['id'].startswith('feature_')],key=lambda z:z['validation']['WAPE'])
    cfg=FeatureConfig(**best['features'])
    for transform in ['log1p','normalized','residual']:r.run('target_'+transform,config=cfg,transform=transform)
    for objective in ['regression_l1','huber','poisson','tweedie']:
        extra={'objective':objective}
        if objective=='tweedie':extra['tweedie_variance_power']=1.3
        r.run('objective_'+objective,config=cfg,params=extra)
    r.run('weather_lagged_reanalysis_sensitivity',config=dataclasses.replace(cfg,weather=True))

def tune_stage():
    r=Research()
    best=min([a for a in r.records if a['eligible'] and a['architecture']=='lightgbm'],key=lambda z:z['validation']['WAPE'])
    cfg=FeatureConfig(**best['features']);transform=best['transform'];base=best['requested_params']
    def lgb(trial):
        params=dict(base,n_estimators=trial.suggest_int('n_estimators',300,1000,step=100),
            num_leaves=trial.suggest_categorical('num_leaves',[15,31,63,95]),max_depth=trial.suggest_categorical('max_depth',[-1,6,10]),
            learning_rate=trial.suggest_float('learning_rate',.015,.09,log=True),min_child_samples=trial.suggest_int('min_child_samples',25,160),
            min_split_gain=trial.suggest_float('min_split_gain',0,.15),subsample=trial.suggest_float('subsample',.7,1),subsample_freq=1,
            colsample_bytree=trial.suggest_float('colsample_bytree',.7,1),reg_alpha=trial.suggest_float('reg_alpha',1e-3,5,log=True),
            reg_lambda=trial.suggest_float('reg_lambda',.05,15,log=True),max_bin=trial.suggest_categorical('max_bin',[127,255]))
        return r.run(f'tune_lgb_{trial.number:02}',config=cfg,transform=transform,params=params,trial=trial)['validation']['WAPE']
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42),pruner=optuna.pruners.MedianPruner(n_startup_trials=4),storage='sqlite:///'+str(OUT/'optuna.db'),study_name='lgb_recursive',load_if_exists=True)
    if len(study.trials)<12:study.optimize(lgb,n_trials=12-len(study.trials))
    study.trials_dataframe().to_csv(REPORT/'optuna_lgb.csv',index=False)
    def xgb(trial):
        obj=trial.suggest_categorical('objective',['reg:squarederror','reg:absoluteerror','count:poisson'])
        params=dict(objective=obj,n_estimators=trial.suggest_int('n_estimators',300,800,step=100),max_depth=trial.suggest_int('max_depth',3,8),
            learning_rate=trial.suggest_float('learning_rate',.02,.09,log=True),min_child_weight=trial.suggest_float('min_child_weight',2,40,log=True),
            subsample=trial.suggest_float('subsample',.7,1),colsample_bytree=trial.suggest_float('colsample_bytree',.7,1),
            gamma=trial.suggest_float('gamma',0,2),reg_alpha=trial.suggest_float('reg_alpha',.001,5,log=True),
            reg_lambda=trial.suggest_float('reg_lambda',.1,20,log=True),max_delta_step=trial.suggest_categorical('max_delta_step',[0,1,3]),tree_method='hist')
        return r.run(f'tune_xgb_{trial.number:02}',architecture='xgboost',config=cfg,params=params,trial=trial)['validation']['WAPE']
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=43),pruner=optuna.pruners.MedianPruner(n_startup_trials=4),storage='sqlite:///'+str(OUT/'optuna.db'),study_name='xgb_recursive',load_if_exists=True)
    if len(study.trials)<8:study.optimize(xgb,n_trials=8-len(study.trials))
    study.trials_dataframe().to_csv(REPORT/'optuna_xgb.csv',index=False)

def strategy_stage():
    r=Research();eligible=[a for a in r.records if a['eligible'] and a.get('artifact')]
    best=min(eligible,key=lambda z:z['validation']['WAPE']);cfg=FeatureConfig(**best['features'])
    # Direct and recursive strategies are scored on exactly the same full trajectories.
    for transform in ['raw','residual','normalized']:
        r.run('direct_'+transform,config=cfg,transform=transform,strategy='direct',params={'n_estimators':700,'learning_rate':.04,'num_leaves':63,'min_child_samples':60})
    r.run('direct_l1',config=cfg,strategy='direct',params={'n_estimators':700,'learning_rate':.04,'num_leaves':63,'objective':'regression_l1','min_child_samples':60})
    r.run('catboost_rmse',architecture='catboost',config=cfg,params={'iterations':700,'depth':7})
    r.run('catboost_mae',architecture='catboost',config=cfg,params={'iterations':700,'depth':7,'loss_function':'MAE'})
    lgb=min([a for a in r.records if a['eligible'] and a['architecture']=='lightgbm' and a['strategy']=='recursive'],key=lambda z:z['validation']['WAPE'])
    r.run('region_specific',config=FeatureConfig(**lgb['features']),transform=lgb['transform'],params=lgb['requested_params'],region_specific=True)

def direct_tune_stage():
    r=Research()
    best=min([a for a in r.records if a['eligible'] and a['strategy']=='direct'],key=lambda a:a['validation']['WAPE'])
    cfg=FeatureConfig(**best['features'])
    def objective(trial):
        transform=trial.suggest_categorical('transform',['raw','residual'])
        params=dict(objective='regression_l1',n_estimators=trial.suggest_int('n_estimators',600,1200,step=200),
            learning_rate=trial.suggest_float('learning_rate',.02,.08,log=True),num_leaves=trial.suggest_categorical('num_leaves',[31,63,95,127]),
            max_depth=trial.suggest_categorical('max_depth',[-1,8,12]),min_child_samples=trial.suggest_int('min_child_samples',30,150),
            subsample=trial.suggest_float('subsample',.75,1),subsample_freq=1,colsample_bytree=trial.suggest_float('colsample_bytree',.8,1),
            reg_alpha=trial.suggest_float('reg_alpha',.001,3,log=True),reg_lambda=trial.suggest_float('reg_lambda',.05,10,log=True))
        return r.run(f'tune_direct_{trial.number:02}',config=cfg,transform=transform,strategy='direct',params=params,trial=trial)['validation']['WAPE']
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=44),pruner=optuna.pruners.MedianPruner(n_startup_trials=4),storage='sqlite:///'+str(OUT/'optuna.db'),study_name='direct_l1',load_if_exists=True)
    if len(study.trials)<8:study.optimize(objective,n_trials=8-len(study.trials))
    study.trials_dataframe().to_csv(REPORT/'optuna_direct.csv',index=False)

if __name__=='__main__':
    import sys
    {'features':feature_stage,'tune':tune_stage,'strategies':strategy_stage,'direct_tune':direct_tune_stage}[sys.argv[1]]()
