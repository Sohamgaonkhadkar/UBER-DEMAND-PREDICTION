"""Freeze selection, refit on Q1, then open April exactly once.

The resulting test JSON is write-once. Running again reports its location.
No search function in this package imports the April holdout.
"""
import copy,json,time
import joblib,numpy as np,pandas as pd
import mlflow
from .protocol import *
from .core import *
from .research import Research
from .alternatives import SeasonalForecast,CompositeForecast,fit_calendar
from .baselines import fit_baseline
from .exposure import fit_exposure
from .calendar_anchoring import CalendarAnchoredRecursive

def bootstrap_summaries(actual,predictions):
    n=len(actual);rng=np.random.default_rng(42)
    starts=rng.integers(0,n,size=(1000,(n+2)//3))
    idx=((starts[:,:,None]+np.arange(3))%n).reshape(1000,-1)[:,:n]
    denom=np.abs(actual).sum(axis=(1,2));results={};draws={}
    for name,p in predictions.items():
        error=np.abs(actual-p).sum(axis=(1,2))
        values=100*error[idx].sum(axis=1)/denom[idx].sum(axis=1)
        draws[name]=values;results[name]=dict(WAPE_95CI=np.quantile(values,[.025,.975]).tolist())
    return results,draws

def main():
    dest=OUT/'final_test_results.json'
    if dest.exists():print('Final test already evaluated; preserving',dest);return
    r=Research();registry={a['id']:a for a in r.records}
    eligible=[a for a in r.records if a.get('eligible') and a.get('artifact') and a.get('status')=='completed']
    overall=min(eligible,key=lambda a:a['validation']['WAPE'])
    rec=min([a for a in eligible if a['strategy']=='recursive'],key=lambda a:a['validation']['WAPE'])
    # The user's primary goal is genuine recursion. Prefer it when its <=0.1pp
    # difference from the overall leader is unresolved by paired origin blocks.
    vp={a['id']:np.load(OUT/'validation_predictions'/f"{a['id']}.npz")['prediction'] for a in [overall,rec]}
    _,validation_draws=bootstrap_summaries(r.truth,vp)
    delta=validation_draws[rec['id']]-validation_draws[overall['id']]
    delta_ci=np.quantile(delta,[.025,.975]).tolist()
    prefer_rec=(rec['validation']['WAPE']-overall['validation']['WAPE']<=.1 and delta_ci[0]<=0<=delta_ci[1])
    winner=rec if prefer_rec else overall
    ids=list(dict.fromkeys([winner['id'],overall['id'],rec['id'],'comparable_v2_xgboost','comparable_v2_lightgbm','seasonal_weekly_24']))
    if any(n not in registry for n in ids):raise RuntimeError('Complete baseline/alternative stages before final evaluation')
    freeze=OUT/'selection_freeze.json'
    specification=dict(selected_id=winner['id'],best_recursive_id=rec['id'],overall_validation_leader=overall['id'],predeclared_test_comparators=ids,
        selected_validation_WAPE=winner['validation']['WAPE'],best_recursive_validation_WAPE=rec['validation']['WAPE'],
        frozen_at=pd.Timestamp.now(tz='UTC').isoformat(),leaderboard_sha256=sha(OUT/'leaderboard.json'),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in pathlib.Path(__file__).resolve().parent.glob('*.py')},
        validation_recursive_minus_overall_WAPE_95CI=delta_ci,
        selection_reason='Prefer genuine recursive strategy when within 0.1 WAPE point and paired block uncertainty includes equality; otherwise select overall validation leader. Rule fixed before April evaluation.',
        test_policy='No further model selection, tuning or alternative creation after this freeze. All comparators are fixed before April labels are opened.')
    if freeze.exists():
        old=json.loads(freeze.read_text())
        if old['leaderboard_sha256']!=specification['leaderboard_sha256']:raise RuntimeError('Selection changed after freeze')
    else:freeze.write_text(json.dumps(specification,indent=2))
    # All fitting below reads Q1 only. The fresh data file is opened later.
    r.cut=pd.Timestamp('2016-04-01',tz=TZ)-pd.Timedelta('2h')
    r.end=int(r.times.searchsorted(r.cut));r.scales=np.maximum(1,r.y[:r.end].mean(axis=0));r.cache=None;r.cache_key=None
    cache={};refit_durations={}
    def refit(name):
        if name in cache:return cache[name]
        record=registry[name];started=time.perf_counter();arch=record['architecture'];details=record['requested_params']
        if name.startswith('comparable_v2_'):bundle=fit_baseline(name.removeprefix('comparable_v2_'),r.panel,r.cut)
        elif arch in ['lightgbm','xgboost','catboost']:
            bundle=r.train(arch,FeatureConfig(**record['features']),record['transform'],record['strategy'],details,record.get('region_specific',False))
        elif arch=='seasonal':bundle=SeasonalForecast(r.scales,details['kind'],details['decay'])
        elif arch=='calendar_profile':bundle=fit_calendar(r,details['decay'])
        elif arch=='hybrid':bundle=CompositeForecast(refit(details['recursive']),refit(details['direct']),switch=details['switch'])
        elif arch=='ensemble':bundle=CompositeForecast(refit(details['first']),refit(details['second']),weight=details['weight'])
        elif arch.startswith('anchored_'):
            bundle=copy.deepcopy(refit(details['base']));bundle.anchor_weight=details['anchor_weight']
        elif arch=='exposure_lightgbm':bundle,_=fit_exposure(r,registry[details['base']],details['synthetic_weight'])
        elif arch=='calendar_anchored_recursive':bundle=CalendarAnchoredRecursive(refit(details['base']),refit(details['calendar']),details['weight'],details['decay'])
        elif arch.startswith('oof_corrected_'):
            base=registry[details['base']];bundle=copy.deepcopy(refit(details['base']))
            oof_start=(r.cut-pd.Timedelta(days=14)).normalize()
            early=Research();early.cut=oof_start-pd.Timedelta('2h');early.end=int(early.times.searchsorted(early.cut));early.scales=np.maximum(1,early.y[:early.end].mean(axis=0))
            fitted=early.train(base['architecture'],FeatureConfig(**base['features']),base['transform'],base['strategy'],base['requested_params'],base.get('region_specific',False))
            oo=pd.date_range(oof_start,r.cut-pd.Timedelta(minutes=47*15),freq='13h');pos=r.times.get_indexer(oo)
            p=fitted.forecast(r.y,r.times,pos,48);truth=np.stack([r.y[o:o+48] for o in pos])
            bias=np.stack([(truth[:,12*k:12*(k+1)]-p[:,12*k:12*(k+1)]).mean(axis=(0,1)) for k in range(4)])
            bundle.bias=np.clip(bias,-.15*r.scales[None,:],.15*r.scales[None,:])
        else:raise ValueError(f'No frozen refit recipe for {name} ({arch})')
        cache[name]=bundle;refit_durations[name]=time.perf_counter()-started
        return bundle
    selected={name:refit(name) for name in ids}
    finaldir=OUT/'final_models';finaldir.mkdir(exist_ok=True)
    for name,bundle in selected.items():joblib.dump(bundle,finaldir/f'{name}.joblib')
    refit_metadata={}
    for name,bundle in cache.items():
        model=getattr(bundle,'model',None)
        refit_metadata[name]=dict(class_name=type(bundle).__module__+'.'+type(bundle).__name__,
            fitted_params=model.get_params() if model is not None and hasattr(model,'get_params') else None,
            refit_seconds=refit_durations.get(name),training_end=str(r.cut),
            validation_record=registry[name],final_fit_policy='Frozen algorithm refit on permitted Q1 labels; early stopping uses an inner final-training week, never April.')
    (OUT/'refit_metadata.json').write_text(json.dumps(refit_metadata,indent=2,default=str))
    chosen=selected[winner['id']]
    def feature_names(bundle):
        if isinstance(bundle,CompositeForecast):return sorted(set(feature_names(bundle.first)+feature_names(bundle.second)))
        if isinstance(bundle,CalendarAnchoredRecursive):return sorted(set(feature_names(bundle.base)+feature_names(bundle.calendar)))
        if hasattr(bundle,'feature_name_'):return list(bundle.feature_name_)
        if hasattr(bundle,'model'):
            model=bundle.model
            return list(getattr(model,'feature_name_',getattr(model,'feature_names_in_',[])))
        return ['region','weekday','quarter','historical_seasonal_profile']
    schema=dict(version=3,feature_columns=feature_names(chosen),categorical_columns=[],numeric_dtype='float32',
        timezone=TZ,history_steps=HISTORY,horizon=48,strategy=winner['strategy'],target='Original-scale retained taxi pickup counts; zero preserved',
        weather='not used',region_order=list(range(30)),origin_policy='Strictly preceding genuine observations only')
    pipeline=dict(model=chosen,feature_columns=schema['feature_columns'],categorical_columns=[],schema_version=3,
        candidate_id=winner['id'],forecast_strategy=winner['strategy'],horizon=48,timezone=TZ,schema=schema,
        training_period='Q1 2016, labels before March 31 22:00 NY; 16-day warmup',training_end=r.cut.isoformat(),selection_protocol=str(freeze.relative_to(ROOT)))
    candidate=OUT/'advanced_candidate_v3.joblib';joblib.dump(pipeline,candidate)
    (OUT/'feature_schema_v3.json').write_text(json.dumps(schema,indent=2))
    validation_demo=dict(pipeline,model=joblib.load(ROOT/winner['artifact']),candidate_id=winner['id']+'_validation',
        training_end=(pd.Timestamp('2016-03-01',tz=TZ)-pd.Timedelta('2h')).isoformat(),training_period='Jan-Feb validation-era model; supports causal March/DST demonstrations')
    joblib.dump(validation_demo,OUT/'validation_demo_v3.joblib')
    print('Selection and fitted models frozen:',ids,'opening fresh April holdout once',flush=True)
    # FIRST predictive read of the new chronological test labels.
    fresh=pd.read_parquet(OUT/'april_holdout.parquet')
    if sha(OUT/'april_holdout.parquet')!=json.loads((OUT/'fresh_holdout_provenance.json').read_text())['sha256']:raise ValueError('Fresh holdout fingerprint changed')
    full=pd.concat([r.panel,fresh],ignore_index=True)
    if full.duplicated(['datetime','region']).any():raise ValueError('Duplicate evaluation observations')
    wide=full.pivot(index='datetime',columns='region',values='total_pickups').sort_index()
    if not wide.index.to_series().diff().iloc[1:].eq(pd.Timedelta('15min')).all():raise ValueError('Evaluation timeline gaps')
    times=wide.index;y=wide.to_numpy(dtype=np.float32)
    oo=origins('2016-04-01','2016-05-01');positions=times.get_indexer(oo)
    actual=np.stack([y[o:o+48] for o in positions])
    results={};predictions={}
    for name,bundle in selected.items():
        started=time.perf_counter();p=bundle.forecast(y,times,positions,48);duration=time.perf_counter()-started
        predictions[name]=p
        results[name]=dict(strategy=registry[name]['strategy'],validation_WAPE=registry[name]['validation']['WAPE'],
            primary=metrics(actual,p),cumulative={str(h):metrics(actual[:,:h],p[:,:h]) for h in [1,4,12,24,48]},
            at_lead={str(h):metrics(actual[:,h-1],p[:,h-1]) for h in [1,4,12,24,48]},
            seconds_per_origin_30regions=duration/len(oo),refit_seconds=refit_durations[name],model_bytes=(finaldir/f'{name}.joblib').stat().st_size)
        print('FINAL HELDOUT',name,results[name]['primary'],flush=True)
    uncertainty,draws=bootstrap_summaries(actual,predictions)
    for name in ids:results[name].update(uncertainty[name])
    gains={}
    for base in ['comparable_v2_xgboost','comparable_v2_lightgbm']:
        bw=results[base]['primary']['WAPE'];nw=results[winner['id']]['primary']['WAPE']
        gains[base]=dict(WAPE_points=bw-nw,relative_error_reduction_pct=100*(bw-nw)/bw,
            paired_WAPE_point_gain_95CI=np.quantile(draws[base]-draws[winner['id']],[.025,.975]).tolist())
    final=dict(selected_id=winner['id'],best_recursive_id=rec['id'],origin_count=len(oo),regions=30,predictions_per_model=actual.size,
        test_origins=[str(t) for t in oo],results=results,gains=gains,
        recursive_below_10_verified=results[rec['id']]['primary']['WAPE']<10,
        selected_below_10=results[winner['id']]['primary']['WAPE']<10,
        frozen_selection=str(freeze.relative_to(ROOT)),candidate_path=str(candidate.relative_to(ROOT)))
    dest.write_text(json.dumps(final,indent=2))
    np.savez_compressed(OUT/'final_test_predictions.npz',actual=actual,**predictions)
    full.to_parquet(OUT/'inference_observations.parquet',index=False)
    for name,result in results.items():
        with mlflow.start_run(run_name='FINAL_'+name):
            mlflow.log_params(dict(candidate_id=name,strategy=result['strategy'],frozen_before_test=True,test_period='April 2016',test_origins=len(oo),horizon=48,refit_training_end=str(r.cut)))
            mlflow.log_metrics({f'heldout_{k}':v for k,v in result['primary'].items() if v is not None})
            mlflow.log_artifact(str(finaldir/f'{name}.joblib'))
    print('Final evaluation complete; no subsequent tuning permitted.',flush=True)

if __name__=='__main__':main()
