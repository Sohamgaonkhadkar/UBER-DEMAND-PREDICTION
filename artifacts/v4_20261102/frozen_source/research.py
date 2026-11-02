"""Bounded three-month development. Final targets are never read here."""
import copy,gc,json,time
import joblib,mlflow,numpy as np,pandas as pd
from .protocol import *
from .fit import *
from ml.advanced.alternatives import SeasonalForecast
DATA_RECIPES={'history_expanding','history_rolling60','history_decay90','history_lightgbm','history_calendar','holiday_boost','holiday_profile','holiday_safety'}

def matching_mode(selected):return {'history_rolling60':'rolling60','history_decay90':'decay90'}.get(selected,'expanding') if selected in DATA_RECIPES else None

def setup_tracking():
    mlflow.set_tracking_uri('sqlite:///'+str(OUT/'tracking.db'))
    if mlflow.get_experiment_by_name('urbanflow_research_v4') is None:
        mlflow.create_experiment('urbanflow_research_v4',artifact_location=(OUT/'mlflow_artifacts').as_uri())
    mlflow.set_experiment('urbanflow_research_v4')

def score_uncertainty(actual,pred,months):
    rng=np.random.default_rng(42);indices=[]
    for month in np.unique(months):
        where=np.flatnonzero(months==month);n=len(where);starts=rng.integers(0,n,size=(1000,(n+2)//3))
        indices.append(where[((starts[:,:,None]+np.arange(3))%n).reshape(1000,-1)[:,:n]])
    idx=np.concatenate(indices,axis=1);den=np.asarray(actual,dtype=float).sum(axis=(1,2));error=np.abs(np.asarray(actual,dtype=float)-pred).sum(axis=(1,2))
    return 100*error[idx].sum(axis=1)/den[idx].sum(axis=1)

def existing(month,name,template):
    if name=='calendar':
        file={'2016-03':OLD/'validation_demo_v3.joblib','2016-04':OLD/'advanced_candidate_v3.joblib','2016-05':PREVIOUS/'nextgen_candidate.joblib'}[month]
        return joblib.load(file)['model']
    if month=='2016-03':return copy.deepcopy(template)
    if month=='2016-05':return joblib.load(PREVIOUS/'final_models'/('matched_refit_tuned_'+name+'.joblib'))
    return None

def build(month,wide,x,target,ts,templates_,stage='development',selected=None):
    cut=pd.Timestamp(month+'-01',tz=TZ)-pd.Timedelta('2h');folder=OUT/'models'/month;folder.mkdir(parents=True,exist_ok=True)
    y=wide.to_numpy(dtype=np.float32);times=wide.index;end=int(times.searchsorted(cut));start=int(times.searchsorted(pd.Timestamp('2016-01-01',tz=TZ)))
    models={};details={}
    def get(name,fun):
        file=folder/(name+'.joblib');meta=file.with_suffix('.json')
        if file.exists():obj=joblib.load(file);record=json.loads(meta.read_text())
        else:
            begun=time.perf_counter();obj,record=fun();record['training_seconds']=time.perf_counter()-begun
            joblib.dump(obj,file);save(meta,record)
        models[name]=obj;details[name]=record;print('Ready',month,name,flush=True);return obj
    def baseline(name):
        if month!='2016-06':
            item=existing(month,name,templates_[name])
            if item is not None:return item,dict(reused_prior_frozen_model=True)
        return fit_fine(templates_[name],x,target,ts,cut,'jan')
    xb=get('pure_xgboost',lambda:baseline('xgboost'));lg=get('pure_lightgbm',lambda:baseline('lightgbm'))
    def calendar():
        if month!='2016-06':return existing(month,'calendar',None),dict(reused_prior_frozen_model=True)
        # Exact current configuration (599 trees), not the pure comparator's 600.
        prototype=copy.deepcopy(joblib.load(OLD/'advanced_candidate_v3.joblib')['model'].base)
        fine,record=fit_fine(prototype,x,target,ts,cut,'jan');p=profile_from_data(y[start:end],times[start:end],end-start)
        from ml.advanced.calendar_anchoring import CalendarAnchoredRecursive
        fine.base.anchor_weight=prototype.anchor_weight
        return CalendarAnchoredRecursive(fine.base,p,.6,12.),record
    cal=get('calendar_feedback',calendar)
    if stage=='preliminary':needed=set(['sparse_nb','hourly_disaggregated','half_hour_disaggregated','temporal_reconciled','cross_temporal','city_hour'])
    elif stage=='final':
        # All references remain, but only the development-selected challenger is
        # fitted again. The recipes below are fixed and share cached dependencies.
        needed={selected}
        if selected in ['holiday_boost','holiday_profile','holiday_safety']:needed|={'history_expanding'}
        if selected in ['holiday_profile','holiday_safety']:needed|={'holiday_boost'}
        if selected in ['temporal_reconciled','cross_temporal']:needed|={'hourly_disaggregated','half_hour_disaggregated'}
        if selected=='cross_temporal':needed|={'city_hour'}
    else:needed=set(['history_expanding','history_rolling60','history_decay90','history_lightgbm','history_calendar','holiday_boost','holiday_profile','holiday_safety','sparse_nb','hourly_disaggregated','half_hour_disaggregated','temporal_reconciled','cross_temporal','city_hour'])
    for name,mode in [('history_expanding','expanding'),('history_rolling60','rolling60'),('history_decay90','decay90')]:
        if name in needed:get(name,lambda mode=mode:fit_fine(templates_['xgboost'],x,target,ts,cut,mode))
    if 'history_lightgbm' in needed:get('history_lightgbm',lambda:fit_fine(templates_['lightgbm'],x,target,ts,cut,'expanding'))
    if 'history_calendar' in needed:
        prototype=copy.deepcopy(cal.base)
        get('history_calendar',lambda:fit_calendar_control(prototype,x,target,ts,wide,cut,'expanding'))
    if 'holiday_boost' in needed:get('holiday_boost',lambda:fit_fine(templates_['xgboost'],x,target,ts,cut,'expanding',True))
    if 'holiday_profile' in needed or 'holiday_safety' in needed:
        p=profile_from_data(y[start:end],times[start:end],end-start);holiday=fit_holiday_profile(wide,cut,p)
        def holiday_model(safety):
            model=copy.deepcopy(models['holiday_boost']);model.profile=p if safety else holiday;model.holiday_safety=safety
            return model,dict(rule='Suppress both seasonal anchors within ±1 day' if safety else 'Pooled train-only event/adjacent-day normalized ratios, shrink 8 days; minimum4 events',support=holiday.support.tolist(),ratios=holiday.ratios.tolist())
        for name,safety in [('holiday_profile',False),('holiday_safety',True)]:
            if name in needed:get(name,lambda safety=safety:holiday_model(safety))
    if 'sparse_nb' in needed:
        def specialist():
            nb,record=fit_sparse(x,target,ts,cut);fine=FineForecast(copy.deepcopy(xb.base if isinstance(xb,FineForecast) else xb));fine.specialist=nb;return fine,record
        get('sparse_nb',specialist)
    for name,factor,city in [('hourly_disaggregated',4,False),('half_hour_disaggregated',2,False),('city_hour',4,True)]:
        if name in needed:get(name,lambda factor=factor,city=city:fit_coarse(wide,cut,factor,city))
    if 'temporal_reconciled' in needed:get('temporal_reconciled',lambda:(ReconciledForecast(xb,models['half_hour_disaggregated'],models['hourly_disaggregated']),dict(weights='Fixed equal-strength nonnegative top-down alignment; not fitted MinT')))
    if 'cross_temporal' in needed:get('cross_temporal',lambda:(ReconciledForecast(xb,models['half_hour_disaggregated'],models['hourly_disaggregated'],models['city_hour']),dict(weights='Temporal alignment then equal city-hour target alignment; original regional errors remain primary')))
    if stage=='final' and matching_mode(selected):
        mode=matching_mode(selected)
        for architecture in ['xgboost','lightgbm']:
            existing_key=selected if selected in ['history_expanding','history_rolling60','history_decay90'] and architecture=='xgboost' else ('history_lightgbm' if selected=='history_lightgbm' and architecture=='lightgbm' else None)
            if architecture=='xgboost' and selected.startswith('holiday'):existing_key='history_expanding'
            if existing_key in models:get('matched_window_'+architecture,lambda existing_key=existing_key:(models[existing_key],dict(details[existing_key],matched_window=mode,reused_same_fit=True)))
            else:get('matched_window_'+architecture,lambda architecture=architecture:fit_fine(templates_[architecture],x,target,ts,cut,mode))
        if selected=='history_calendar':get('matched_window_calendar',lambda:(models[selected],dict(details[selected],matched_window=mode,reused_same_fit=True)))
        else:
            prototype=copy.deepcopy(joblib.load(OLD/'advanced_candidate_v3.joblib')['model'].base)
            get('matched_window_calendar',lambda:fit_calendar_control(prototype,x,target,ts,wide,cut,mode))
    get('seasonal_weekly',lambda:(SeasonalForecast(np.ones(30),'weekly'),dict(reference=True)))
    # City-only outputs have one series; its valid regional counterpart is the
    # explicitly reconciled cross-temporal arm, not a fake city WAPE comparison.
    models.pop('city_hour',None)
    return models,details

def main():
    initialize();development_only()
    if not (OUT/'history_2015.parquet').exists():raise RuntimeError('Historical acquisition must complete before fitting')
    wide=combined();templates_=templates();x,target,ts=rows(wide,templates_['xgboost']);months=['2016-03','2016-04','2016-05'];records=[]
    setup_tracking()
    for month in months:
        dest=OUT/('development_complete_'+month+'.npz')
        if dest.exists():print('Preserving completed month',month,flush=True);continue
        models,details=build(month,wide,x,target,ts,templates_)
        nextmonth=str((pd.Timestamp(month+'-01')+pd.offsets.MonthBegin(1)).date());oo=origins(month+'-01',nextmonth);pos=wide.index.get_indexer(oo)
        truth=np.stack([wide.to_numpy(dtype=np.float32)[o:o+48] for o in pos]);predictions={};reproduction={}
        prior_file=OUT/('development_'+month+'.npz');prior=np.load(prior_file) if prior_file.exists() else None
        if prior is not None:np.testing.assert_array_equal(truth,prior['actual'])
        for name,model in models.items():
            begun=time.perf_counter();p=model.forecast(wide.to_numpy(dtype=np.float32),wide.index,pos,48);seconds=time.perf_counter()-begun
            if p.shape!=truth.shape or not np.isfinite(p).all() or (p<0).any():raise ValueError(name+' invalid forecast')
            if prior is not None and name in prior.files:
                np.testing.assert_array_equal(p,prior[name]);reproduction[name]=float(np.max(np.abs(p-prior[name])));predictions[name]=prior[name]
                print('REPRODUCED',month,name,'max difference 0',flush=True);continue
            predictions[name]=p;record=dict(month=month,model=name,metrics=metrics(truth,p),seconds_per_origin=seconds/len(pos),details=details[name]);records.append(record)
            save(REPORT/('development_'+month+'_'+name+'.json'),record)
            with mlflow.start_run(run_name='DEV_'+month+'_'+name):
                mlflow.log_params(dict(month=month,model=name,horizon=48,origins=len(pos),final_targets_used=False));mlflow.log_metrics(record['metrics'])
                mlflow.log_artifact(str(OUT/'models'/month/(name+'.joblib')))
                mlflow.log_artifact(str(OUT/'models'/month/(name+'.json')))
            print('DEV',month,name,record['metrics']['WAPE'],flush=True)
        np.savez_compressed(dest,actual=truth,**predictions);save(OUT/('reproduction_'+month+'.json'),reproduction)
        origin_file=OUT/('origins_'+month+'.json')
        if not origin_file.exists():save(origin_file,[str(t) for t in oo])
        del models;gc.collect()
    arrays=[np.load(OUT/('development_complete_'+m+'.npz')) for m in months];truth=np.concatenate([a['actual'] for a in arrays]);groups=np.concatenate([np.full(len(a['actual']),i) for i,a in enumerate(arrays)])
    names=[n for n in arrays[0].files if n!='actual'];pred={n:np.concatenate([a[n] for a in arrays]) for n in names}
    baseline_names=['pure_xgboost','pure_lightgbm','calendar_feedback'];reference=min(baseline_names,key=lambda n:metrics(truth,pred[n])['WAPE'])
    draws={n:score_uncertainty(truth,p,groups) for n,p in pred.items()};summary=[]
    base_region=100*np.abs(truth-pred[reference]).sum(axis=(0,1))/truth.sum(axis=(0,1))
    for name,p in pred.items():
        score=metrics(truth,p);gain=metrics(truth,pred[reference])['WAPE']-score['WAPE'];ci=np.quantile(draws[reference]-draws[name],[.025,.975]).tolist()
        region=100*np.abs(truth-p).sum(axis=(0,1))/truth.sum(axis=(0,1));improved=int((region<base_region).sum());month_gain=[metrics(a['actual'],a[reference])['WAPE']-metrics(a['actual'],a[name])['WAPE'] for a in arrays]
        regional_gate=improved>=20
        if name=='sparse_nb':
            specialist=joblib.load(OUT/'models/2016-05/sparse_nb.joblib').specialist
            regional_gate=bool(np.all(region[specialist.regions]<base_region[specialist.regions]))
        passes=name in baseline_names or (gain>=.1 and ci[0]>0 and sum(g>0 for g in month_gain)>=2 and regional_gate and np.max(region-base_region)<=2)
        summary.append(dict(model=name,**score,gain_vs_strongest_baseline=gain,paired_gain_95CI=ci,regions_improved=improved,month_gains=month_gain,passes=bool(passes)))
    eligible=[r for r in summary if r['passes']];best=min(eligible,key=lambda r:r['WAPE'])
    complexity={'pure_xgboost':0,'pure_lightgbm':0,'history_expanding':0,'history_rolling60':0,'history_lightgbm':0,'history_calendar':1,'history_decay90':1,'holiday_boost':1,'calendar_feedback':1,'holiday_safety':2,'holiday_profile':3,'sparse_nb':2,'hourly_disaggregated':2,'half_hour_disaggregated':2,'temporal_reconciled':4,'cross_temporal':5,'seasonal_weekly':0}
    for candidate in sorted(eligible,key=lambda r:(complexity[r['model']],r['WAPE'])):
        if complexity[candidate['model']]<complexity[best['model']] and candidate['WAPE']-best['WAPE']<=.1:
            interval=np.quantile(draws[candidate['model']]-draws[best['model']],[.025,.975])
            if interval[0]<=0<=interval[1]:best=candidate
    selected=best['model']
    if selected in baseline_names and selected!=reference:selected=reference
    result=dict(selected_id=selected,strongest_baseline=reference,records=summary,months=months,selection='Frozen multi-month/materiality/paired-block/regional gate',no_final_outcomes_read=True)
    save(OUT/'development_results.json',result);pd.DataFrame(summary).to_csv(REPORT/'development_leaderboard.csv',index=False)
    code={str(p.relative_to(ROOT)):sha(p) for p in pathlib.Path(__file__).parent.glob('*.py')}
    save(OUT/'selection_freeze.json',dict(selected_id=selected,primary_period='June 2016',secondary_period='July 1-7, 2016; same model/cutoff if compatible',secondary_availability=json.loads((OUT/'secondary_availability.json').read_text()),frozen_at=pd.Timestamp.now(tz='UTC').isoformat(),development_sha256=sha(OUT/'development_results.json'),code_sha256=code,matched_window=matching_mode(selected),comparators=baseline_names+([selected] if selected not in baseline_names else [])+(['matched_window_xgboost','matched_window_lightgbm','matched_window_calendar'] if matching_mode(selected) else [])+['seasonal_weekly'],strategy='regional 15-minute recursive' if selected in baseline_names or selected.startswith('holiday') or selected.startswith('history') or selected=='sparse_nb' else 'coarse-recursive/disaggregated or reconciled; not 48-feedback recursion'))
    print('SELECTED',selected,'strongest baseline',reference,flush=True)

def preliminary():
    """Run independent Jan-only arms while bounded historical queries finish."""
    initialize();development_only();data=panel();wide=data.pivot(index='datetime',columns='region',values='total_pickups').sort_index()
    tt=templates();x,y,ts=rows(wide,tt['xgboost'])
    for month in ['2016-03','2016-04','2016-05']:
        dest=OUT/('preliminary_'+month+'.npz')
        if dest.exists():continue
        models,details=build(month,wide,x,y,ts,tt,stage='preliminary')
        nextmonth=str((pd.Timestamp(month+'-01')+pd.offsets.MonthBegin(1)).date());oo=origins(month+'-01',nextmonth);pos=wide.index.get_indexer(oo)
        actual=np.stack([wide.to_numpy(dtype=np.float32)[o:o+48] for o in pos]);pred={}
        for name,model in models.items():
            begun=time.perf_counter();p=model.forecast(wide.to_numpy(dtype=np.float32),wide.index,pos,48);pred[name]=p
            record=dict(month=month,model=name,metrics=metrics(actual,p),seconds_per_origin=(time.perf_counter()-begun)/len(pos),details=details[name])
            save(REPORT/('development_'+month+'_'+name+'.json'),record)
            print('PRELIM',month,name,record['metrics']['WAPE'],flush=True)
        np.savez_compressed(dest,actual=actual,**pred);del models;gc.collect()

if __name__=='__main__':
    import sys
    preliminary() if sys.argv[-1]=='preliminary' else main()
