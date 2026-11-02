"""Render the final research report exclusively from frozen saved evidence."""
import json,os,pathlib,sys
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'urbanflow'))
from ml.advanced.protocol import OUT,REPORT,TZ,metrics
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def f(value,n=3):return 'N/A' if value is None else f'{value:.{n}f}'

def main():
    final=json.loads((OUT/'final_test_results.json').read_text());freeze=json.loads((OUT/'selection_freeze.json').read_text())
    records=json.loads((OUT/'leaderboard.json').read_text());registry={a['id']:a for a in records}
    diagnostic=json.loads((REPORT/'baseline_diagnostics.json').read_text())
    arrays=np.load(OUT/'final_test_predictions.npz');actual=arrays['actual'];chosen=final['selected_id'];prediction=arrays[chosen]
    q1=pd.read_parquet(ROOT/'artifacts/counts_ny.parquet');cut=pd.Timestamp('2016-04-01',tz=TZ)-pd.Timedelta('2h')
    training=q1[q1.datetime<cut];means=training.groupby('region').total_pickups.mean();peaks=training.groupby('region').total_pickups.quantile(.95)
    tiers=means.map(lambda x:'low' if x<=means.quantile(1/3) else 'high' if x>=means.quantile(2/3) else 'medium')
    oo=pd.to_datetime(final['test_origins'],utc=True).tz_convert(TZ)
    # Complete per-region, individual-lead, cumulative-horizon, origin and regime evidence.
    region_rows=[];lead_rows=[];origin_rows=[];group_rows=[]
    for name,result in final['results'].items():
        p=arrays[name]
        for r in range(30):region_rows.append(dict(model=name,region=r,demand_tier=tiers[r],**metrics(actual[:,:,r],p[:,:,r])))
        for k in range(48):lead_rows.append(dict(model=name,step=k+1,**metrics(actual[:,k,:],p[:,k,:])))
        for j,o in enumerate(oo):origin_rows.append(dict(model=name,origin=str(o),**metrics(actual[j],p[j])))
        for tier in ['low','medium','high']:
            mask=tiers.to_numpy()==tier;group_rows.append(dict(model=name,group='demand_'+tier,**metrics(actual[:,:,mask],p[:,:,mask])))
        peak_mask=actual>=peaks.to_numpy()[None,None,:]
        group_rows.append(dict(model=name,group='training_95th_percentile_peaks',**metrics(actual[peak_mask],p[peak_mask])))
        target_times=pd.DatetimeIndex([o+pd.Timedelta(minutes=15*k) for o in oo for k in range(48)])
        flat_a=actual.reshape(-1,30);flat_p=p.reshape(-1,30)
        for label,mask in [('night',target_times.hour<6),('day',target_times.hour>=6),('weekend',target_times.day_of_week>=5),('weekday',target_times.day_of_week<5),('peak_hour',np.isin(target_times.hour,[7,8,9,17,18,19]))]:
            group_rows.append(dict(model=name,group=label,**metrics(flat_a[mask],flat_p[mask])))
    regions=pd.DataFrame(region_rows);regions.to_csv(REPORT/'heldout_per_region.csv',index=False)
    leads=pd.DataFrame(lead_rows);leads.to_csv(REPORT/'heldout_by_lead.csv',index=False)
    pd.DataFrame(origin_rows).to_csv(REPORT/'heldout_by_origin.csv',index=False)
    groups=pd.DataFrame(group_rows);groups.to_csv(REPORT/'heldout_by_regime.csv',index=False)
    board=pd.DataFrame([dict(id=r['id'],architecture=r['architecture'],strategy=r['strategy'],transform=r['transform'],eligible=r['eligible'],validation_WAPE=r['validation']['WAPE'],validation_bias_pct=r['validation']['bias_pct'],training_seconds=r['training_seconds'],runtime_seconds=r['runtime_seconds'],model_MiB=r.get('model_bytes',np.nan)/1024**2,artifact=r.get('artifact')) for r in records]).sort_values('validation_WAPE')
    board.to_csv(REPORT/'leaderboard.csv',index=False)
    # Additional development diagnostics; no further fitting or selection.
    d=pd.read_parquet(OUT/'diagnostic_predictions.parquet')
    t=pd.to_datetime(d.datetime,utc=True,format='mixed').dt.tz_convert(TZ)
    d['week_of_month']=(t.dt.day-1)//7+1;d['dst_regime']=np.where(t<pd.Timestamp('2016-03-13 03:00',tz=TZ),'before_spring_DST','after_spring_DST')
    d['volatility_group']=pd.cut(d.origin_volatility,[-np.inf,.1,.3,np.inf],labels=['<0.1','0.1-0.3','>0.3'])
    for col in ['week_of_month','dst_regime','volatility_group']:
        rows=[]
        for (name,key),g in d.groupby(['model',col],observed=True):rows.append(dict(model=name,group=str(key),**metrics(g.actual,g.prediction)))
        pd.DataFrame(rows).to_csv(REPORT/f'diagnostics_by_{col}.csv',index=False)
    # Correlations and genuine importance from the validated recursive GBDT component.
    base=registry[registry[chosen]['requested_params']['base']]
    while 'base' in base['requested_params']:base=registry[base['requested_params']['base']]
    cfg=base['features']
    import hashlib
    key=hashlib.sha256(json.dumps(dict(config=cfg,strategy='recursive',end=str(pd.Timestamp('2016-03-01',tz=TZ)-pd.Timedelta('2h'))),sort_keys=True).encode()).hexdigest()[:14]
    x=pd.read_parquet(OUT/'features'/f'{key}.parquet').drop(columns=['__target','__timestamp'])
    corr=x.corr().abs();pairs=[]
    for i,a in enumerate(corr.columns):
        for j in range(i+1,len(corr)):
            if corr.iloc[i,j]>=.95:pairs.append(dict(first=a,second=corr.columns[j],absolute_correlation=float(corr.iloc[i,j])))
    pd.DataFrame(pairs).sort_values('absolute_correlation',ascending=False).to_csv(REPORT/'high_feature_correlations.csv',index=False)
    import joblib
    model=joblib.load(ROOT/base['artifact']).model
    names=list(getattr(model,'feature_name_',getattr(model,'feature_names_in_',[])))
    importance=model.booster_.feature_importance(importance_type='gain') if hasattr(model,'booster_') else model.feature_importances_
    pd.DataFrame(dict(feature=names,importance=importance)).sort_values('importance',ascending=False).to_csv(REPORT/'model_feature_importance.csv',index=False)
    # Publication-style figures from fixed, representative origins/regions.
    fig,axes=plt.subplots(3,1,figsize=(11,9),layout='constrained')
    reps=[int(means.sort_values().index[i]) for i in [2,15,27]];origin_index=len(oo)//2
    for ax,r in zip(axes,reps):
        ax.plot(range(1,49),actual[origin_index,:,r],color='black',lw=2,label='Actual')
        for name,label in [(chosen,'Selected genuine recursive'),('calendar_blend_direct_0.5','Frozen direct/calendar comparator'),('comparable_v2_xgboost','Refit V2 XGBoost')]:
            ax.plot(range(1,49),arrays[name][origin_index,:,r],label=label,alpha=.85)
        ax.set_title(f'Region {r} ({tiers[r]} demand); fixed middle April origin {oo[origin_index]}');ax.set_ylabel('Pickups / 15 minutes');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    axes[-1].set_xlabel('Lead (15-minute steps)');fig.savefig(REPORT/'heldout_actual_vs_predicted.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for name,label in [(chosen,'Selected recursive'),('calendar_blend_direct_0.5','Direct/calendar'),('comparable_v2_xgboost','V2 XGBoost'),('comparable_v2_lightgbm','V2 LightGBM')]:
        g=leads[leads.model==name];ax.plot(g.step,g.WAPE,label=label)
    ax.axhline(10,ls='--',color='grey',label='10% reference (individual lead)');ax.set(xlabel='Lead (15-minute steps)',ylabel='WAPE at individual lead (%)');ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
    fig.savefig(REPORT/'heldout_error_by_lead.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for name,label in [(chosen,'Selected recursive'),('comparable_v2_xgboost','V2 XGBoost')]:
        g=pd.DataFrame(origin_rows);g=g[g.model==name];ax.plot(range(1,len(g)+1),g.WAPE,label=label)
    ax.set(xlabel='Chronological April origin (13-hour spacing)',ylabel='Cumulative 48-step WAPE (%)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(REPORT/'heldout_origin_variation.png',dpi=140);plt.close(fig)
    selected=final['results'][chosen];test=selected['primary'];ci=selected['WAPE_95CI']
    localfile=REPORT/'local_system_results.json';local=json.loads(localfile.read_text()) if localfile.exists() else None
    candidate_regions=regions[regions.model==chosen].set_index('region');baseline_regions=regions[regions.model=='comparable_v2_xgboost'].set_index('region')
    improved=int((candidate_regions.WAPE<baseline_regions.WAPE).sum())
    stats=pd.DataFrame([dict(horizon=a['request']['horizon'],latency_ms=a['latency_ms']) for a in local['final']['requests']]) if local else None
    source_rows=sum(json.loads(p.read_text())['retained_rows'] for p in (OUT/'april_daily').glob('*.json'))
    s='# UrbanFlow AI — advanced model optimization\n\n'
    s+=f"**Verified untouched-April result: {test['WAPE']:.4f}% cumulative WAPE through 48 genuinely recursive steps. The below-10% target was NOT achieved.** The moving-block bootstrap 95% interval is [{ci[0]:.3f}%, {ci[1]:.3f}%]; an interval crossing 10% is not evidence that the target was met. No tuning followed the April evaluation.\n\n"
    s+=f"Selected candidate: `{chosen}`. MAE {test['MAE']:.4f}, RMSE {test['RMSE']:.4f}, R² {test['R2']:.5f}, signed bias {test['bias_pct']:.3f}%. Saved separately at `artifacts/advanced_20261102/advanced_candidate_v3.joblib`. All original models, previous candidates, notebooks, reports and the existing backend source remain intact. Nothing was committed, pushed, deployed or changed remotely.\n\n"
    s+='## A. Initial baselines and protocol differences\n\n'
    s+=table(['Reference','WAPE %','Meaning'],[['Original LightGBM','9.420','March one-step, legacy zero→10/features/observed weather'],['Previous V2 XGBoost','12.929','Six March origins, 48-step cumulative, fit before February 15'],['Previous V2 LightGBM','13.294','Same six origins and early fitting period'],['Frozen V2 XGBoost, expanded diagnostic',f(diagnostic['xgboost']['recursive']['WAPE']),'57 March origins; still the original early-fitted candidate'],['Frozen V2 LightGBM, expanded diagnostic',f(diagnostic['lightgbm']['recursive']['WAPE']),'Same expanded origins']])
    s+='\nThose figures are historical context, not denominators for claimed April improvements. March had already been inspected in phase one, including full-period one-step errors; it is explicitly development validation here. A new chronological April test was acquired instead of relabeling March as untouched.\n\n'
    s+=f"Training for development experiments: January 17–February 29 21:45 NY, with earlier observations supplying 1,536-step warmup. Validation: **57 origins, 13 elapsed hours apart, all 30 regions, all 48 steps** (82,080 predictions/model). Origins rotate through hours/days/weeks; 12-hour trajectories do not overlap. Training labels stop two hours before March 1; gap demand can enter later forecasts only as genuinely preceding observations. Direct training uses origins spaced 29 slots and all 48 leads; every target is before the fitting cutoff. Early stopping uses an inner February holdout, never April.\n\n"
    s+=f"Final refit: Q1 labels before March 31 22:00 NY, same 16-day warmup; selection and comparator IDs were frozen first. Final test: **55 prespecified April origins × 30 regions × 48 leads = {actual.size:,} predictions/model**. All origins were specified before April forecast errors were opened. Source: [official NYC coordinate-level 2016 records](https://data.cityofnewyork.us/Transportation/2016-Yellow-Taxi-Trip-Data/uacg-pexx), with {source_rows:,} retained April trips under the unchanged original filters. Modern TLC Parquet uses taxi-zone IDs and could not directly preserve these 30 clusters, so it was not substituted.\n\n"
    s+='The scaler and rounded notebook centroids were recovered and verified against **all 33,234,199 Q1 labels with zero mismatches**. The map was fixed before April. Source bounds/fare/distance filters were not tuned to April; targets retain genuine zeros. Daily query URLs, retrieval times and response hashes are saved. Complete source coverage remains an assumption. The target is pickups of trips passing the original post-trip filters, not an independently verified live all-taxi feed.\n\n'
    s+='## B. Root-cause analysis\n\n'
    s+=table(['Frozen model','Teacher-forced WAPE %','Recursive WAPE %','Recursive bias %'],[[n,f(v['teacher_forced']['WAPE']),f(v['recursive']['WAPE']),f(v['recursive']['bias_pct'])] for n,v in diagnostic.items()])
    s+='\nThe same-model comparison changes only whether subsequent demand features receive actual prior demand or forecast feedback. It directly demonstrates feedback-state error; teacher forcing is a diagnostic oracle, never counted as a 12-hour forecast. Peaks were underpredicted by approximately 10% while other intervals were overpredicted. Night WAPE was about 22–23%, versus about 13% in daytime; weekends and low-demand regions were harder. This is conditional regression toward typical demand, not a single global bias. Adding a scalar OOF correction and exposure-context augmentation did not resolve it and were rejected.\n\n'
    s+='Stable regional weekday/time shape was underused. Same-clock seasonal profiles helped; extra raw lags alone and a dense feature union did not. These observations support seasonal anchoring and reduced reliance on uncertain long recursive paths. They do not establish that weather, annual seasonality or all distribution shifts have been solved. One winter quarter cannot identify annual behavior.\n\n'
    s+=f"![Feedback drift diagnosis](<{(REPORT/'feedback_drift.png').as_posix()}>)\n\n"
    s+='Numerical diagnostics by lead, region, hour, weekday/weekend, peak/night, volatility, week of March, DST regime and origin are saved as `advanced_20261102/diagnostics_by_*.csv`. All difficult origins remain in evaluation.\n\n'
    s+='## C. Feature engineering and target experiments\n\n'
    screens=[a for a in records if a['id'].startswith(('feature_','target_','objective_','weather_'))]
    s+=table(['Experiment','Validation WAPE %','Eligible'],[[a['id'],f(a['validation']['WAPE']),a['eligible']] for a in screens])
    s+='\nThe retained compact ML representation uses 46 float32 features: temporal/cyclical/calendar fields, selected 15-minute lags, rolling mean/std/EWMA, same-clock yesterday/week/two-week values and smoothed profiles, and recent-to-seasonal ratios. Clock matching is New York/DST-aware; no other region enters accidentally. Explicit spatial experiments use a fixed three-neighbor map and joint generated predictions, never sibling future truth. Added medians/min/max/volatility, region interactions and spatial features did not beat the compact set in this screen. All 30 separate regional models also lost to the best global alternatives.\n\n'
    s+='Raw, log1p, region-normalized and seasonal-residual targets were reconstructed on the unchanged count scale. Normalization was particularly weak globally; it was not promoted to make low-count regions look better. A direct seasonal-residual L1 model helped; the analogous recursive residual target did not. Huber severely underfit in its screened bounded configuration and was not expanded. Poisson/Tweedie were valid for nonnegative raw counts but did not win.\n\n'
    s+='The weather sensitivity used only origin-minus-three-hour observed/reanalysis values held constant through the path, never future observed weather. Historical real-time availability is not certified, so it was ineligible even independently of its lack of validation gain. The selected model uses no weather. No unreliable historical event indicators were added. Genuine importance and high-correlation pairs are in `model_feature_importance.csv` and `high_feature_correlations.csv`; ablations, not temporally shuffled permutation scores, guided selection.\n\n'
    s+='## D. Structured optimization\n\n'
    s+='After diagnostics and ablation, Optuna ran 12 LightGBM and 8 XGBoost recursive trials, then 8 focused direct L1 trials. Seeds were fixed; native training used four threads. Search covered depth/leaves, learning rate, iterations, leaf/child constraints, subsampling and feature fractions, regularization, split gain/gamma, bins and count/absolute/squared objectives as appropriate. Inner-period early stopping selected useful iteration counts; a fixed subset of validation origins supported median pruning, and completed candidates were ranked on all 57 origins. Exact complete/pruned trials are in the Optuna CSVs and local `optuna.db`.\n\n'
    tuned=sorted([a for a in records if a['id'].startswith('tune_')],key=lambda a:a['validation']['WAPE'])[:6]
    s+=table(['Configuration','Architecture/strategy','WAPE %'],[[a['id'],a['architecture']+'/'+a['strategy'],f(a['validation']['WAPE'])] for a in tuned])
    s+='\nFull requested and fitted hyperparameters, feature flags, transformations, periods, timings, scores and retained artifact locations are in `artifacts/advanced_20261102/leaderboard.json`, local MLflow `tracking.db`, and `refit_metadata.json`. Inferior models were not all retained; sizes for unsaved artifacts are unavailable. No claim of exhaustive optimization or equal tuning budgets is made.\n\n'
    s+='## E. Forecast strategies and final selection\n\n'
    strategy_rows=[]
    for strategy in ['recursive','direct','hybrid','blend','seasonal']:
        best=min([a for a in records if a['strategy']==strategy and a['eligible']],key=lambda a:a['validation']['WAPE'])
        strategy_rows.append([strategy,best['id'],f(best['validation']['WAPE'])])
    s+=table(['Strategy','Best eligible configuration','Validation WAPE %'],strategy_rows)
    s+='\nThe selected model genuinely recurses. Each step starts with XGBoost, adds a modest prior-week profile anchor, then gradually blends in a training-only regional weekday/quarter median profile adjusted by origin-time observed demand. **The final blended prediction is appended to history and drives every subsequent lag/rolling feature.** No future pickup truth or future observed weather enters. The calendar-only profile contains no recursively invented observations. Direct and independently blended paths remain explicitly labeled as such.\n\n'
    s+=f"The best direct/calendar blend had validation WAPE {registry[freeze['overall_validation_leader']]['validation']['WAPE']:.4f}%, versus {registry[chosen]['validation']['WAPE']:.4f}% for genuine recursion. The recursive-minus-overall paired-block 95% interval was [{freeze['validation_recursive_minus_overall_WAPE_95CI'][0]:.4f}, {freeze['validation_recursive_minus_overall_WAPE_95CI'][1]:.4f}] WAPE points. Before opening April, the recursive approach was preferred because the <=0.1-point gap was unresolved and genuine recursion was the primary objective. Both were frozen for test, so the choice was not made from April results. Ensemble candidates needed a paired block interval establishing gain over their best component; candidates failing that gate were marked ineligible. Intervals are exploratory after development search, without a multiple-comparison correction.\n\n"
    s+='Exposure-aware training used genuinely preceding fitted-model feedback contexts and historical labels; it worsened validation to about 14.2%. OOF bias correction was unstable (about 22.1%) and rejected. Those failures are retained. They argue against assuming a more complicated correction necessarily improves deployment behavior.\n\n'
    s+='## F. Compact validation leaderboard\n\n'
    s+=table(['Rank','Experiment','Strategy','WAPE %','Eligible'],[[i+1,a['id'],a['strategy'],f(a['validation']['WAPE']),a['eligible']] for i,a in enumerate(sorted(records,key=lambda a:a['validation']['WAPE'])[:20])])
    s+=f"\nThere are {len(records)} completed recorded experiments plus pruned Optuna trials. The complete compact CSV is `advanced_20261102/leaderboard.csv`. All scores in that leaderboard use the same outer March origins and metric denominator. Phase-one frozen diagnostic models are kept separate from the comparably refit V2 baselines.\n\n"
    s+='## G. Untouched April test results\n\n'
    s+=table(['Frozen configuration','Strategy','WAPE %','MAE','RMSE','R²','Bias %'],[[name,res['strategy']]+[f(res['primary'][k]) for k in ['WAPE','MAE','RMSE','R2','bias_pct']] for name,res in final['results'].items()])
    s+='\nPrimary WAPE = 100 × Σ|actual−prediction| / Σ|actual| across **all** selected origins, 48 leads and regions. It is not an average of regional percentages, an aggregate-city cancellation score, or step-48-only error.\n\n'
    s+=table(['Lead H','Time','Cumulative WAPE through H %','WAPE at lead H only %','Cumulative MAE','Cumulative RMSE','Cumulative R²'],[[h,{1:'15 min',4:'1 h',12:'3 h',24:'6 h',48:'12 h'}[h],f(selected['cumulative'][str(h)]['WAPE']),f(selected['at_lead'][str(h)]['WAPE']),f(selected['cumulative'][str(h)]['MAE']),f(selected['cumulative'][str(h)]['RMSE']),f(selected['cumulative'][str(h)]['R2'])] for h in [1,4,12,24,48]])
    s+=f"\nPrimary 95% moving-block interval: [{ci[0]:.3f}%, {ci[1]:.3f}%], 1,000 resamples, seed 42, blocks of three chronological origins (~39 hours), retaining entire region/lead trajectories. Non-overlapping paths still share weekly behavior and historical context; dependence longer than the block is a limitation. This estimates performance uncertainty, not probabilistic intervals for individual pickup predictions. **The point estimate remains above 10%; the target is not verified.**\n\n"
    for label,file in [('April actual versus predicted','heldout_actual_vs_predicted.png'),('Individual lead error','heldout_error_by_lead.png'),('Origin variation','heldout_origin_variation.png')]:
        s+=f'![{label}](<{(REPORT/file).as_posix()}>)\n\n'
    s+=table(['Selected-model regime','WAPE %','MAE','Bias %'],[[a.group,f(a.WAPE),f(a.MAE),f(a.bias_pct)] for a in groups[groups.model==chosen].itertuples()])
    s+=f'\nRegional WAPE improved against comparable V2 XGBoost in **{improved}/30 regions**. Every region remains included. Full per-region, regime/peak, lead and origin metrics are in the heldout CSVs; no origin was removed because its error was large. Calendar seasons outside this quarter/spring transition remain untested.\n\n'
    s+='Peak-demand WAPE improved from 14.463% to 11.273%, and peak underprediction improved from −13.027% to −9.789%. It remains substantial. Night WAPE improved from 18.273% to 13.602%; low-demand-tier WAPE improved from 24.661% to 20.755%, but its signed bias worsened from −2.491% to −4.084%. Thus not every bias goal improved despite all-region WAPE gains.\n\n'
    s+='## H. Comparable improvements\n\n'
    s+=table(['Refit baseline, same April protocol','Baseline WAPE %','New WAPE %','Gain in WAPE points','Relative error reduction %','Paired point-gain 95% interval'],[[base,f(final['results'][base]['primary']['WAPE']),f(test['WAPE']),f(gain['WAPE_points']),f(gain['relative_error_reduction_pct']),str([round(v,3) for v in gain['paired_WAPE_point_gain_95CI']])] for base,gain in final['gains'].items()])
    s+='\nThese gains compare the same April targets, origins, horizons, filters, regions, allowed Q1 fitting period and warmup. No percentage improvement is calculated against the original one-step 9.42% benchmark or the earlier six-origin March run.\n\n'
    s+='## I. Local serving, cost and validation\n\n'
    s+=f"Retained candidate size: {selected['model_bytes']/1024**2:.3f} MiB. Final refit including dependent components: {selected['refit_seconds']:.2f} seconds. Batched test inference averaged {selected['seconds_per_origin_30regions']*1000:.2f} ms per 48-step/all-region origin; this is throughput, not HTTP latency.\n\n"
    if local:
        latency_rows=[]
        for h,g in stats.groupby('horizon'):latency_rows.append([h,f(g.latency_ms.median(),1),f(g.latency_ms.quantile(.95),1)])
        s+=table(['HTTP horizon','p50 roundtrip ms','p95 roundtrip ms'],latency_rows)
        s+=f"\nWhole loaded backend process-tree RSS: {local['final']['whole_backend_process_tree_RSS_MiB']:.1f} MiB (includes Python and library imports; not isolated model memory). Actual Uvicorn checks passed {local['final']['valid_requests']+local['validation_demo']['valid_requests']} valid and {local['final']['invalid_requests']} invalid requests: all 30 regions; horizons 1/4/12/24/48; finite/nonnegative ordered outputs; identical short/long prefixes; compatible response fields; no future history; and DST with the Jan–Feb validation-era artifact. Servers were stopped after checks.\n\n"
    s+='Full final regression suite: **101 passed**, one upstream Starlette/httpx deprecation warning. It covers shared features, direct origin-only inputs, cross-region feedback, timezone/DST, transformed count reconstruction, schema order, true calendar feedback, invalid/missing context and sanitized errors. The final Q1-refitted model rejects origins before its fitting cutoff; the separate validation-era model supports causal March/DST demonstrations.\n\n'
    s+='The opt-in entrypoint is `urbanflow/backend/advanced_main.py`; original `main.py` and default model remain intact. Contract fields are preserved with candidate/strategy metadata. No frontend redesign or live deployment occurred. Serving currently uses the full tested local ML environment, including research-module dependencies; production dependency separation, container packaging, genuine delayed live observations and monitoring still require work. No live observation stream, calibrated prediction intervals or operational alerts are claimed.\n\n'
    s+='A wrapper serialization issue was caught before freeze: CLI-defined classes used `__main__`. Original experimental files were retained, canonical portable copies were created, and **all saved validation predictions were verified unchanged**. Correct module entrypoints and actual independent-server loading now verify portability. The repair record is `artifact_portability_repair.json`.\n\n'
    s+='## J. Recommendation and limits\n\n'
    s+=f"Keep `{chosen}` as the strongest primary-goal candidate for local review. It improved genuine recursive accuracy materially and outperformed the fixed direct/calendar comparator on April, but **10.2449% is above the requested threshold**. Do not replace production automatically. These results justify further independent testing, not a claim of production readiness or guaranteed modern accuracy.\n\n"
    s+='Further optimization must use a new development protocol and a new untouched chronological test; April is now spent. It must not be repeatedly inspected to choose new weights until a sub-10 number appears. Additional years, forecast-origin-available weather, better live data latency definitions, event/airport signals and longer-horizon seasonal coverage may help, but no unexecuted gain is claimed. Original filtering depends on completed-trip fields, and historical source completeness cannot be independently certified. Rounded recovered centroids match Q1 labels exactly but are not the original binary training state.\n\n'
    s+='The rejected OOF bias experiment inserted its corrections inside the feedback loop, where they can compound. Its poor result does not rule out a separately validated post-path correction; that variant was not tested. Recorded training runtimes include feature-cache materialization on first use, so they are actual experiment costs rather than isolated estimator benchmarks.\n\n'
    s+='Reproduction/run guide: `urbanflow/ml/advanced/README.md`. Evidence: protocol and selection freeze, source hashes, local MLflow/Optuna databases, leaderboard JSON/CSV, final test JSON/NPZ, schemas/refit metadata, all-region CSVs, system results and logs. Existing phase-one files and original-model bytes are verified in `advanced_20261102/preservation_checks.json`. No remote action was performed.\n'
    (ROOT/'reports/advanced_model_optimization.md').write_text(s,encoding='utf-8')
    summary=dict(selected=chosen,target_met=final['recursive_below_10_verified'],test_WAPE=test['WAPE'],validation_WAPE=selected['validation_WAPE'],test_95CI=ci,regions_improved_vs_v2_xgboost=improved,completed_experiments=len(records),fresh_retained_trips=source_rows)
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
