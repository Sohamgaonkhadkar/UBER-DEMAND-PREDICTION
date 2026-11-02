"""Freeze March selection, refit permitted history, open May once."""
import copy,json,time
import joblib,numpy as np,pandas as pd,mlflow
from ml.advanced.baselines import fit_baseline
from ml.advanced.alternatives import SeasonalForecast
from .protocol import *
from .models import *
from .fit import as_controlled,fit_base,fit_calibrator

def uncertainty(actual,predictions):
    n=len(actual);rng=np.random.default_rng(42);s=rng.integers(0,n,(1000,(n+2)//3));idx=((s[:,:,None]+np.arange(3))%n).reshape(1000,-1)[:,:n]
    den=actual.sum(axis=(1,2));out={};draws={}
    for name,p in predictions.items():
        error=np.abs(actual-p).sum(axis=(1,2));v=100*error[idx].sum(1)/den[idx].sum(1)
        out[name]=np.quantile(v,[.025,.975]).tolist();draws[name]=v
    return out,draws

def main():
    initialize();dest=OUT/'final_test_results.json'
    if dest.exists():print('May already evaluated. No repeat tuning/evaluation.');return
    if not (OUT/'may_sealed.parquet').exists():raise RuntimeError('Complete May acquisition first')
    dev=json.loads((OUT/'validation_results.json').read_text());selected=dev['selected_id']
    ids=['frozen_Q1_current','matched_refit_current','matched_refit_tuned_xgboost','matched_refit_tuned_lightgbm','matched_refit_v2_xgboost','matched_refit_v2_lightgbm','clock_weekly_naive']
    if selected!='current':ids.append('selected_'+selected)
    freeze=dict(selected_recipe=selected,selected_id='matched_refit_current' if selected=='current' else 'selected_'+selected,
        selection_reason='Prespecified March materiality/paired-block/region gates; April forecast errors never fitted/selected correction weights.',
        comparators=ids,frozen_at=pd.Timestamp.now(tz='UTC').isoformat(),validation_sha256=sha(OUT/'validation_results.json'),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in pathlib.Path(__file__).resolve().parent.glob('*.py')},
        added_history_comparison='Frozen existing Q1 candidate vs identical configuration fitted Jan-April; no new hyperparameter search.',
        tuned_baseline_note='Strongest previously validated pure XGBoost/LightGBM and V2 references, all with matched Jan-April labels and warmup; recipes fixed before May outcomes.')
    file=OUT/'selection_freeze.json'
    if file.exists():
        if json.loads(file.read_text())['validation_sha256']!=freeze['validation_sha256']:raise RuntimeError('Validation changed after freeze')
    else:file.write_text(json.dumps(freeze,indent=2))
    # No May targets are read during any fit or calibration below.
    panel=read_development();wide=panel.pivot(index='datetime',columns='region',values='total_pickups');y=wide.to_numpy(dtype=np.float32);times=wide.index
    old_pipeline=joblib.load(OLD/'advanced_candidate_v3.joblib');original=old_pipeline['model']
    cutoff=pd.Timestamp('2016-05-01',tz=TZ)-pd.Timedelta('2h')
    started=time.perf_counter();refit=fit_base(y,times,cutoff,original,'final_jan_april');training_seconds={'matched_refit_current':time.perf_counter()-started}
    models={'frozen_Q1_current':original,'matched_refit_current':refit}
    old_records=json.loads((OLD/'leaderboard.json').read_text())
    for architecture in ['xgboost','lightgbm']:
        record=min([r for r in old_records if r['eligible'] and r['architecture']==architecture and r['strategy']=='recursive'],key=lambda r:r['validation']['WAPE'])
        template=copy.deepcopy(original);template.base=joblib.load(ROOT/record['artifact'])
        started=time.perf_counter();fitted=fit_base(y,times,cutoff,template,'final_jan_april')
        name='matched_refit_tuned_'+architecture;models[name]=fitted.base;models[name].anchor_weight=0.
        training_seconds[name]=time.perf_counter()-started
    for architecture in ['xgboost','lightgbm']:
        started=time.perf_counter();name='matched_refit_v2_'+architecture;models[name]=fit_baseline(architecture,panel,cutoff);training_seconds[name]=time.perf_counter()-started
    models['clock_weekly_naive']=SeasonalForecast(refit.scales,'weekly')
    if selected!='current':
        candidate=copy.deepcopy(refit)
        if selected in ['weekly_boundary_only','recent_shrunk_weighted_median']:
            candidate.calendar=profile_from_data(y,times,int(times.searchsorted(cutoff)),selected)
        else:
            # March OOF base is already Jan-Feb fitted; April residuals are unused.
            early=joblib.load(OLD/'validation_demo_v3.joblib')['model'];early=as_controlled(early,y,times,pd.Timestamp('2016-03-01',tz=TZ)-pd.Timedelta('2h'))
            oo=origins('2016-03-01','2016-04-01');pos=times.get_indexer(oo);p,tr=early.generate(y,times,pos,48,trace=True);tr['prediction']=p
            actual=np.stack([y[o:o+48] for o in pos]);kind='gate' if selected=='learned_context_gate' else 'post_path'
            calibrator,meta=fit_calibrator(tr['features'],actual,tr,early.scales,kind)
            if kind=='gate':candidate.gate=calibrator
            else:candidate.correction=calibrator
            (OUT/'final_calibration_metadata.json').write_text(json.dumps(meta,indent=2))
        models['selected_'+selected]=candidate
    candidate=models[freeze['selected_id']]
    metadata=dict(training_end=cutoff.isoformat(),history_steps=HISTORY,geography_sha256=sha(OLD/'geography.json'),
        exact_original_params=original.base.model.get_params(),refit_params=refit.base.model.get_params(),
        current_recipe_unchanged=selected=='current',development=dev,training_seconds=training_seconds,
        target='Same retained-trip counts; all zeros unchanged',April_forecast_residuals_used=False)
    (OUT/'training_metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
    folder=OUT/'final_models';folder.mkdir(exist_ok=True)
    for name,model in models.items():joblib.dump(model,folder/f'{name}.joblib')
    pipeline=dict(old_pipeline,model=candidate,candidate_id='nextgen_'+freeze['selected_id'],training_end=cutoff.isoformat(),
        training_period='Jan-April labels before April 30 22:00 NY; unchanged 16-day warmup',forecast_strategy='postprocessed_recursive' if selected=='post_path_linear' else 'recursive',
        selection_protocol=str(file.relative_to(ROOT)))
    pipeline['schema']=dict(pipeline['schema'],strategy=pipeline['forecast_strategy'])
    artifact=OUT/'nextgen_candidate.joblib';joblib.dump(pipeline,artifact)
    (OUT/'feature_schema.json').write_text(json.dumps(pipeline['schema'],indent=2))
    print('Selection/models frozen; opening May targets once. Recipe:',selected,flush=True)
    fresh=pd.read_parquet(OUT/'may_sealed.parquet')
    if sha(OUT/'may_sealed.parquet')!=json.loads((OUT/'may_provenance.json').read_text())['sha256']:raise ValueError('May fingerprint changed')
    full=pd.concat([panel,fresh],ignore_index=True)
    if full.duplicated(['datetime','region']).any():raise ValueError('Duplicate observations')
    allwide=full.pivot(index='datetime',columns='region',values='total_pickups').sort_index();alltimes=allwide.index;ally=allwide.to_numpy(dtype=np.float32)
    if not alltimes.to_series().diff().iloc[1:].eq(pd.Timedelta('15min')).all():raise ValueError('Gaps in full evaluation index')
    oo=origins('2016-05-01','2016-06-01');pos=alltimes.get_indexer(oo);actual=np.stack([ally[o:o+48] for o in pos])
    results={};predictions={}
    for name,model in models.items():
        started=time.perf_counter();p=model.forecast(ally,alltimes,pos,48);seconds=time.perf_counter()-started
        predictions[name]=p
        results[name]=dict(primary=metrics(actual,p),cumulative={str(h):metrics(actual[:,:h],p[:,:h]) for h in [1,4,12,24,48]},
            at_lead={str(h):metrics(actual[:,h-1],p[:,h-1]) for h in [1,4,12,24,48]},seconds_per_origin=seconds/len(pos),model_bytes=(folder/f'{name}.joblib').stat().st_size)
        print('MAY FINAL',name,results[name]['primary'],flush=True)
    ci,draws=uncertainty(actual,predictions)
    for name in results:results[name]['WAPE_95CI']=ci[name]
    gain={}
    for reference in ['frozen_Q1_current','matched_refit_current','matched_refit_tuned_xgboost','matched_refit_tuned_lightgbm']:
        chosen=freeze['selected_id'];old=results[reference]['primary']['WAPE'];new=results[chosen]['primary']['WAPE']
        gain[reference]=dict(WAPE_point_gain=old-new,relative_error_reduction_pct=100*(old-new)/old,paired_gain_95CI=np.quantile(draws[reference]-draws[chosen],[.025,.975]).tolist())
    final=dict(selected_id=freeze['selected_id'],selected_recipe=selected,selected_strategy=pipeline['forecast_strategy'],results=results,gains=gain,
        origins=[str(o) for o in oo],origin_count=len(oo),regions=30,predictions_per_model=actual.size,
        below_10_genuine_recursive=results[freeze['selected_id']]['primary']['WAPE']<10 and pipeline['forecast_strategy']=='recursive',
        no_further_tuning=True,candidate=str(artifact.relative_to(ROOT)))
    dest.write_text(json.dumps(final,indent=2));np.savez_compressed(OUT/'final_test_predictions.npz',actual=actual,**predictions)
    full.to_parquet(OUT/'inference_observations.parquet',index=False)
    mlflow.set_tracking_uri('sqlite:///'+str(OUT/'tracking.db'));mlflow.set_experiment('urbanflow_next_generation')
    for name,record in results.items():
        with mlflow.start_run(run_name='MAY_FINAL_'+name):
            mlflow.log_params(dict(frozen_before_May=True,horizon=48,origins=len(oo),training_end='Q1 frozen' if name=='frozen_Q1_current' else cutoff.isoformat(),recipe=selected if name==freeze['selected_id'] else name))
            mlflow.log_metrics({f'May_{k}':v for k,v in record['primary'].items() if v is not None});mlflow.log_artifact(str(folder/f'{name}.joblib'))
    print('May evaluation complete. It is now spent; no subsequent tuning.',flush=True)

if __name__=='__main__':main()
