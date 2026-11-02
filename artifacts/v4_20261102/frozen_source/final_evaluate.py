"""One frozen primary June evaluation plus a separate July holiday window."""
import copy,json,time
import joblib,mlflow,numpy as np,pandas as pd
from .protocol import *
from .fit import combined,templates,rows
from .research import build,score_uncertainty,setup_tracking

def describe(model):
    from .models import FineForecast,CoarseForecast,ReconciledForecast
    from ml.advanced.calendar_anchoring import CalendarAnchoredRecursive
    if isinstance(model,CalendarAnchoredRecursive):return dict(type='calendar_feedback_recursive',feature_columns=model.base.feature_name_,calendar_profile=type(model.calendar).__name__,maximum_calendar_weight=model.weight,decay=model.decay)
    if isinstance(model,FineForecast):
        return dict(type='fine_recursive',feature_columns=model.base.feature_name_,extra_calendar=model.extra,calendar_profile=type(model.profile).__name__ if model.profile is not None else None,holiday_safety=model.holiday_safety,sparse_regions=model.specialist.regions.tolist() if model.specialist is not None else [])
    if isinstance(model,CoarseForecast):
        return dict(type='coarse_recursive',factor=model.factor,feature_columns=list(model.model.feature_name_),distribution_shape=list(model.shares.shape),city=model.city)
    if isinstance(model,ReconciledForecast):
        return dict(type='post_path_coherent_alignment',fine=describe(model.fine),half=describe(model.half),hourly=describe(model.hourly),city=describe(model.city) if model.city is not None else None)
    return dict(type=type(model).__name__,feature_columns=model.feature_name_)

def main(prepare_only=False):
    initialize();freeze=json.loads((OUT/'selection_freeze.json').read_text())
    if (OUT/'final_test_results.json').exists():print('Final tests are spent; preserving results.');return
    for path,digest in freeze['code_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Source changed after selection: '+path)
    if sha(OUT/'development_results.json')!=freeze['development_sha256']:raise ValueError('Development selection changed')
    # No final-period labels are read during fitting.
    wide=combined();tt=templates();x,target,ts=rows(wide,tt['xgboost'])
    models,details=build('2016-06',wide,x,target,ts,tt,stage='final',selected=freeze['selected_id'])
    references=['pure_xgboost','pure_lightgbm','calendar_feedback','seasonal_weekly'];selected=freeze['selected_id']
    if freeze['matched_window']:references+=['matched_window_xgboost','matched_window_lightgbm','matched_window_calendar']
    keep=references+([selected] if selected not in references else []);models={n:models[n] for n in keep}
    # Data refits are written before opening any heldout target arrays.
    template=joblib.load(PREVIOUS/'nextgen_candidate.joblib');pipeline=dict(template,model=models[selected],candidate_id='v4_'+selected,training_end=pd.Timestamp('2016-05-31 22:00',tz=TZ).isoformat(),training_period='Frozen v4 recipe; final label cutoff May 31 22:00 New York; exact label windows/historical roles in training_metadata.json',forecast_strategy=freeze['strategy'],selection_protocol=str(OUT/'selection_freeze.json'))
    description=describe(models[selected])
    pipeline['schema']=dict(pipeline['schema'],strategy=freeze['strategy'],feature_columns=description.get('feature_columns',description.get('fine',{}).get('feature_columns',[])),components=description)
    if not (OUT/'candidate_v4.joblib').exists():
        joblib.dump(pipeline,OUT/'candidate_v4.joblib')
        save(OUT/'feature_schema.json',pipeline['schema']);save(OUT/'training_metadata.json',dict(training_end=pipeline['training_end'],selected_id=selected,models=details,source_panel_sha256=sha(PREVIOUS/'inference_observations.parquet'),history_sha256=sha(OUT/'history_2015.parquet'),test_data_used_in_fit=False))
    if prepare_only:print('Final refits prepared; no final targets read.',flush=True);return
    plan=json.loads((OUT/'protocol.json').read_text());result={}
    setup_tracking()
    windows=[('june_primary','june_sealed','primary_test')]
    if freeze['secondary_availability']['eligible']:windows.append(('july_event','july_event_sealed','secondary_event_test'))
    for label,sealed,key in windows:
        future=pd.read_parquet(OUT/(sealed+'.parquet'))
        if sha(OUT/(sealed+'.parquet'))!=json.loads((OUT/(sealed+'_provenance.json')).read_text())['sha256']:raise ValueError('Sealed source changed')
        if label=='june_primary':full=pd.concat([panel(),future],ignore_index=True)
        else:full=pd.concat([panel(),pd.read_parquet(OUT/'june_sealed.parquet'),future],ignore_index=True)
        allwide=full.pivot(index='datetime',columns='region',values='total_pickups').sort_index();yy=allwide.to_numpy(dtype=np.float32);times=allwide.index
        if not times.to_series().diff().iloc[1:].eq(pd.Timedelta('15min')).all():raise ValueError('Missing heldout context')
        oo=pd.to_datetime(plan[key]['origins'],utc=True).tz_convert(TZ);pos=times.get_indexer(oo);actual=np.stack([yy[o:o+48] for o in pos]);pred={};records={}
        for name,model in models.items():
            begun=time.perf_counter();p=model.forecast(yy,times,pos,48);seconds=time.perf_counter()-begun
            if p.shape!=actual.shape or not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid final forecast')
            pred[name]=p;ci=score_uncertainty(actual,p,np.zeros(len(actual),int))
            records[name]=dict(primary=metrics(actual,p),WAPE_95CI=np.quantile(ci,[.025,.975]).tolist(),cumulative={str(h):metrics(actual[:,:h],p[:,:h]) for h in [1,4,12,24,48]},at_lead={str(h):metrics(actual[:,h-1],p[:,h-1]) for h in [1,4,12,24,48]},seconds_per_origin=seconds/len(pos),model_bytes=(OUT/'models/2016-06'/(name+'.joblib')).stat().st_size)
            with mlflow.start_run(run_name=label+'_'+name):
                mlflow.log_params(dict(model=name,selection_frozen=True,window=label,origins=len(pos)));mlflow.log_metrics(records[name]['primary'])
                mlflow.log_artifact(str(OUT/'models/2016-06'/(name+'.joblib')))
                if name==selected:
                    for file in ['feature_schema.json','training_metadata.json','protocol.json','selection_freeze.json']:
                        mlflow.log_artifact(str(OUT/file))
            print('FINAL',label,name,records[name]['primary'],flush=True)
        draws={n:score_uncertainty(actual,p,np.zeros(len(actual),int)) for n,p in pred.items()}
        gains={n:dict(point_gain=records[n]['primary']['WAPE']-records[selected]['primary']['WAPE'],paired_gain_95CI=np.quantile(draws[n]-draws[selected],[.025,.975]).tolist()) for n in references}
        result[label]=dict(records=records,gains=gains,origins=[str(o) for o in oo],origin_count=len(oo),points=actual.size)
        np.savez_compressed(OUT/(label+'_predictions.npz'),actual=actual,**pred)
        # Actual observations remain legitimate history for later origins, but do
        # not update fitted parameters or select a different July model.
        if label==windows[-1][0]:full.to_parquet(OUT/'inference_observations.parquet',index=False)
    save(OUT/'final_test_results.json',dict(selected_id=selected,strategy=freeze['strategy'],windows=result,secondary_availability=freeze['secondary_availability'],below10_primary=result['june_primary']['records'][selected]['primary']['WAPE']<10,no_posttest_selection=True))

if __name__=='__main__':
    import sys
    main(prepare_only=sys.argv[-1]=='prepare')
