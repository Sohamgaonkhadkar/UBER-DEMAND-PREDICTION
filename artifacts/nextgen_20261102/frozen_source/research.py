"""Four bounded configurations plus a matched existing baseline; no April tuning."""
import copy,json,time
import joblib,numpy as np,pandas as pd,mlflow
from .protocol import *
from .models import *
from .fit import as_controlled,fit_base,fit_calibrator

def paired_gain(actual,base,candidate):
    den=actual.sum(axis=(1,2));old=np.abs(actual-base).sum(axis=(1,2));new=np.abs(actual-candidate).sum(axis=(1,2))
    n=len(actual);rng=np.random.default_rng(42);starts=rng.integers(0,n,(1000,(n+2)//3));idx=((starts[:,:,None]+np.arange(3))%n).reshape(1000,-1)[:,:n]
    gains=100*(old[idx].sum(axis=1)-new[idx].sum(axis=1))/den[idx].sum(axis=1)
    oldr=100*np.abs(actual-base).sum(axis=(0,1))/actual.sum(axis=(0,1));newr=100*np.abs(actual-candidate).sum(axis=(0,1))/actual.sum(axis=(0,1))
    return dict(point_gain=metrics(actual,base)['WAPE']-metrics(actual,candidate)['WAPE'],CI=np.quantile(gains,[.025,.975]).tolist(),regions_improved=int(np.sum(newr<oldr)),worst_region_worsening=float(np.max(newr-oldr)))

def main():
    initialize()
    if (OUT/'selection_freeze.json').exists():print('Selection already frozen; no new research allowed');return
    if (OUT/'validation_results.json').exists():print('Bounded validation completed; preserving existing results');return
    panel=read_development();wide=panel.pivot(index='datetime',columns='region',values='total_pickups');y=wide.to_numpy(dtype=np.float32);times=wide.index
    prototype=joblib.load(OLD/'validation_demo_v3.joblib')['model'];cut=pd.Timestamp('2016-03-01',tz=TZ)-pd.Timedelta('2h')
    base=as_controlled(prototype,y,times,cut);oo=origins('2016-03-01','2016-04-01');positions=times.get_indexer(oo)
    actual=np.stack([y[p:p+48] for p in positions]);baseline=base.forecast(y,times,positions,48)
    old=np.load(OLD/'validation_predictions/calendar_feedback_0.6.npz')['prediction']
    np.testing.assert_allclose(baseline,old,rtol=1e-10,atol=1e-10)
    print('Baseline replication exact; validation WAPE',metrics(actual,baseline)['WAPE'],flush=True)
    # One preceding model fit supplies genuinely held-out February contexts.
    started=time.perf_counter();oofcut=pd.Timestamp('2016-02-15',tz=TZ)-pd.Timedelta('2h')
    oof=fit_base(y,times,oofcut,prototype,'calibration_february')
    cal_origins=origins('2016-02-15','2016-03-01');calpos=times.get_indexer(cal_origins)
    cp,trace=oof.generate(y,times,calpos,48,trace=True);trace['prediction']=cp
    truth=np.stack([y[p:p+48] for p in calpos])
    gate,gatemeta=fit_calibrator(trace['features'],truth,trace,oof.scales,'gate')
    correction,corrmeta=fit_calibrator(trace['features'],truth,trace,oof.scales,'post_path')
    calibration_seconds=time.perf_counter()-started
    (OUT/'calibration_metadata.json').write_text(json.dumps(dict(fit_end=str(oofcut),origins=[str(t) for t in cal_origins],gate=gatemeta,correction=corrmeta,runtime_seconds=calibration_seconds),indent=2))
    joblib.dump(dict(gate=gate,correction=correction),OUT/'development_calibrators.joblib')
    cases={'current':(base,'recursive',dict(hypothesis='Matched existing architecture; exact old validation predictions'))}
    end=int(times.searchsorted(cut))
    for recipe in ['weekly_boundary_only','recent_shrunk_weighted_median']:
        m=copy.deepcopy(base);m.calendar=profile_from_data(y,times,end,recipe)
        cases[recipe]=(m,'recursive',dict(hypothesis='Calendar adjacency and seasonal recency',recipe=recipe,half_life_days=28,prior_effective_weeks=4))
    m=copy.deepcopy(base);m.gate=gate
    cases['learned_context_gate']=(m,'recursive',dict(hypothesis='Origin-context reliability changes calendar trust',coefficient_source='Feb OOF',optimizer=gatemeta))
    m=copy.deepcopy(base);m.correction=correction
    cases['post_path_linear']=(m,'postprocessed_recursive',dict(hypothesis='Correct residuals outside the state update, avoiding amplification',coefficient_source='Feb OOF',optimizer=corrmeta))
    mlflow.set_tracking_uri('sqlite:///'+str(OUT/'tracking.db'));mlflow.set_experiment('urbanflow_next_generation')
    records=[]
    for name,(model,strategy,details) in cases.items():
        started=time.perf_counter();pred=model.forecast(y,times,positions,48);runtime=time.perf_counter()-started
        scores=metrics(actual,pred);gain=paired_gain(actual,baseline,pred)
        passes=(name=='current' or gain['point_gain']>=.1 and gain['CI'][0]>0 and gain['regions_improved']>=20 and gain['worst_region_worsening']<=2)
        if name=='learned_context_gate' and not gatemeta['optimizer_success']:passes=False
        if name=='post_path_linear' and not corrmeta['optimizer_success']:passes=False
        path=OUT/'models';path.mkdir(exist_ok=True);artifact=path/f'{name}_development.joblib';joblib.dump(model,artifact)
        np.savez_compressed(OUT/f'validation_{name}.npz',prediction=pred)
        record=dict(id=name,strategy=strategy,details=details,validation=scores,gain=gain,promotion_gate_passes=bool(passes),seconds_per_origin=runtime/len(positions),model_bytes=artifact.stat().st_size,artifact=str(artifact.relative_to(ROOT)),training_cutoff=str(cut))
        records.append(record)
        with mlflow.start_run(run_name='DEV_'+name):
            mlflow.log_params(dict(strategy=strategy,details=json.dumps(details),calibration_period='Feb OOF',validation_period='March',origins=len(oo),horizon=48,April_errors_used_for_weights=False))
            mlflow.log_metrics({f'validation_{k}':v for k,v in scores.items() if v is not None});mlflow.log_metric('inference_seconds',runtime);mlflow.log_artifact(str(artifact))
        print('VALIDATION',name,scores['WAPE'],gain,'passes',passes,flush=True)
    eligible=[r for r in records if r['promotion_gate_passes']]
    winner=min(eligible,key=lambda r:r['validation']['WAPE'])
    recursive=min([r for r in eligible if r['strategy']=='recursive'],key=lambda r:r['validation']['WAPE'])
    if winner['strategy']!='recursive' and recursive['validation']['WAPE']-winner['validation']['WAPE']<=.1:winner=recursive
    result=dict(selected_id=winner['id'],selected_strategy=winner['strategy'],records=records,validation_origins=[str(t) for t in oo],baseline_replication_max_error=float(np.max(np.abs(baseline-old))),calibration_seconds=calibration_seconds,
        selection_note='No April errors fitted coefficients or selected their weights. Selection uses March only and the prespecified block/region gate.')
    (OUT/'validation_results.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame([dict(id=r['id'],strategy=r['strategy'],WAPE=r['validation']['WAPE'],gain=r['gain']['point_gain'],CI_lower=r['gain']['CI'][0],CI_upper=r['gain']['CI'][1],regions_improved=r['gain']['regions_improved'],passes=r['promotion_gate_passes']) for r in records]).to_csv(REPORT/'validation_leaderboard.csv',index=False)
    print('SELECTED',winner['id'],flush=True)

if __name__=='__main__':main()
