"""Report saved, frozen experiments; never fit a model or acquire test data."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent/'urbanflow'))
from ml.nextgen.protocol import ROOT, OUT, OLD, REPORT, TZ, metrics

def table(rows, columns):
    def fmt(v):
        return f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |'] +
        ['| '+' | '.join(fmt(row.get(c,'')) for c in columns)+' |' for row in rows])

def link(path, label=None):
    return f'[{label or Path(path).name}](<{Path(path).as_posix()}>)'

def main():
    final=json.loads((OUT/'final_test_results.json').read_text())
    dev=json.loads((OUT/'validation_results.json').read_text())
    diagnosis=json.loads((REPORT/'diagnosis.json').read_text())
    system=json.loads((REPORT/'local_system_results.json').read_text())
    metadata=json.loads((OUT/'training_metadata.json').read_text())
    z=np.load(OUT/'final_test_predictions.npz'); a=z['actual']
    names=list(final['results']); chosen=final['selected_id']
    oo=pd.to_datetime(final['origins'],utc=True).tz_convert(TZ)
    ts=pd.DatetimeIndex([o+pd.Timedelta(minutes=15*k) for o in oo for k in range(48)])
    train=pd.read_parquet(OLD/'inference_observations.parquet')
    train=train[train.datetime<pd.Timestamp(metadata['training_end'])]
    q=train.groupby('region').total_pickups.quantile(.95).to_numpy()
    means=train.groupby('region').total_pickups.mean().to_numpy()
    tier=np.zeros(30,int); tier[np.argsort(means)[10:20]]=1; tier[np.argsort(means)[20:]]=2
    shape=a.shape; night=np.broadcast_to((ts.hour<6).reshape(-1,48,1),shape)
    holiday=np.broadcast_to(np.asarray(ts.date==pd.Timestamp('2016-05-30').date()).reshape(-1,48,1),shape)
    masks={'night_00_06':night,'day_06_24':~night,'realized_training_p95':a>=q[None,None,:],
        'Memorial_Day':holiday,'other_days':~holiday,
        'weekend':np.broadcast_to((ts.day_of_week>=5).reshape(-1,48,1),shape),
        'weekday':np.broadcast_to((ts.day_of_week<5).reshape(-1,48,1),shape)}
    for j,t in enumerate(['low_region_mean','medium_region_mean','high_region_mean']):masks[t]=np.broadcast_to((tier==j)[None,None,:],shape)
    regional=[]; leads=[]; regimes=[]; origins=[]; hours=[]
    for name in names:
        p=z[name]
        for r in range(30):
            y,v=a[:,:,r],p[:,:,r]; mask=y>=q[r]; nm=night[:,:,r]
            row=dict(model=name,region=r,mean_demand=float(y.mean()),CV=float(y.std()/(y.mean()+1e-9)),zero_fraction=float((y==0).mean()),**metrics(y,v))
            row.update({'peak_'+k:val for k,val in metrics(y[mask],v[mask]).items()})
            row.update({'night_'+k:val for k,val in metrics(y[nm],v[nm]).items()});regional.append(row)
        for k in range(48):leads.append(dict(model=name,step=k+1,**metrics(a[:,k],p[:,k])))
        for label,mask in masks.items():regimes.append(dict(model=name,regime=label,n=int(mask.sum()),**metrics(a[mask],p[mask])))
        for i,o in enumerate(oo):origins.append(dict(model=name,origin=str(o),**metrics(a[i],p[i])))
        for h in range(24):
            mask=ts.hour==h;hours.append(dict(model=name,hour=h,**metrics(a.reshape(-1,30)[mask],p.reshape(-1,30)[mask])))
    dfs={key:pd.DataFrame(rows) for key,rows in [('may_per_region',regional),('may_by_lead',leads),('may_regime_metrics',regimes),('may_by_origin',origins),('may_by_target_hour',hours)]}
    for key,df in dfs.items():df.to_csv(REPORT/(key+'.csv'),index=False)
    # Spent April origin-start diagnostics and aggregate leads; no coefficients fitted.
    old=json.loads((OLD/'final_test_results.json').read_text()); az=np.load(OLD/'final_test_predictions.npz')
    aa,ap=az['actual'],az[old['selected_id']]; ao=pd.to_datetime(old['test_origins'],utc=True).tz_convert(TZ)
    april_origin=[dict(origin_hour=h,**metrics(aa[ao.hour==h],ap[ao.hour==h])) for h in sorted(set(ao.hour))]
    pd.DataFrame(april_origin).to_csv(REPORT/'spent_april_by_origin_hour.csv',index=False)
    april_leads=pd.DataFrame([dict(step=k+1,**metrics(aa[:,k],ap[:,k])) for k in range(48)])
    april_leads.to_csv(REPORT/'spent_april_by_lead.csv',index=False)
    plt.rcParams.update({'figure.dpi':130,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    ar=pd.read_csv(REPORT/'spent_april_region_diagnosis.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    axes[0].bar(ar.region,ar.WAPE,color=['#c44e52' if r in [3,6] else '#3279a8' for r in ar.region]);axes[0].set(xlabel='Fixed cluster ID',ylabel='April WAPE (%)',title='Sparse and airport-centered clusters are harder')
    sc=axes[1].scatter(ar.centroid_longitude,ar.centroid_latitude,c=ar.WAPE,cmap='viridis',s=65)
    for row in ar.itertuples():axes[1].annotate(str(row.region),(row.centroid_longitude,row.centroid_latitude),xytext=(3,3),textcoords='offset points',fontsize=7)
    axes[1].set(xlabel='Longitude',ylabel='Latitude',title='Verified centroids; land use unspecified');fig.colorbar(sc,ax=axes[1],label='WAPE (%)')
    fig.savefig(REPORT/'april_region_diagnosis.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    keys=['realized_top95','calendar_expected_top95','forecast_top95'];axes[0].bar(['Future actual peak','Origin calendar peak','Forecast peak'],[diagnosis[k]['bias_pct'] for k in keys],color=['#c44e52','#3279a8','#3279a8'])
    axes[0].set(ylabel='Signed bias (%)',title='Peak conditioning changes the interpretation');axes[0].axhline(0,color='black',lw=.7)
    at=pd.read_csv(REPORT/'spent_april_time_diagnosis.csv');at=at[at.field=='hour'].copy();at['value']=at.value.astype(int);at=at.sort_values('value')
    axes[1].plot(at.value,at.WAPE,marker='o');axes[1].set(xlabel='Target hour (New York)',ylabel='April WAPE (%)',title='Higher relative error overnight');fig.savefig(REPORT/'april_peak_and_time.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    labels={'frozen_Q1_current':'Current, frozen Q1','matched_refit_current':'Current, Jan-Apr refit','matched_refit_tuned_xgboost':'Prior tuned XGBoost','matched_refit_tuned_lightgbm':'Prior tuned LightGBM'}
    for name,label in labels.items():
        rows=dfs['may_by_lead'];rows=rows[rows.model==name];axes[0].plot(rows.step,rows.WAPE,label=label)
        rows=dfs['may_by_origin'];rows=rows[rows.model==name];axes[1].plot(range(len(rows)),rows.WAPE,label=label,alpha=.8)
    axes[0].set(xlabel='Individual 15-minute lead',ylabel='May WAPE (%)',title='Fresh May test: individual leads');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Chronological forecast origin (all 57)',ylabel='12-hour cumulative WAPE (%)',title='Origin variability; holiday outcomes retained');fig.savefig(REPORT/'may_horizon_and_origin.png');plt.close(fig)
    i=len(oo)//2;fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,r in zip(axes.ravel(),[2,3,6,18]):
        ax.plot(np.arange(48)/4,a[i,:,r],label='Actual',color='black');ax.plot(np.arange(48)/4,z[chosen][i,:,r],label='Jan-Apr refit');ax.plot(np.arange(48)/4,z['matched_refit_tuned_xgboost'][i,:,r],label='Prior tuned XGBoost',alpha=.8)
        ax.set(title=f'Region {r}; origin {oo[i].strftime("%b %d %H:%M %Z")}',xlabel='Hours from origin',ylabel='Retained trips / 15 minutes');ax.legend(fontsize=8)
    fig.savefig(REPORT/'may_fixed_example_paths.png');plt.close(fig)
    # HTTP measurements, distinct from batched offline evaluation throughput.
    requests=pd.DataFrame([dict(horizon=r['request']['horizon'],latency_ms=r['latency_ms']) for r in system['requests']])
    latency=[]
    for h,g in requests.groupby('horizon'):latency.append(dict(horizon=int(h),n=len(g),p50_ms=float(g.latency_ms.median()),p95_ms=float(g.latency_ms.quantile(.95))))
    pd.DataFrame(latency).to_csv(REPORT/'local_http_latency.csv',index=False)
    comparison=[]
    for name,r in final['results'].items():
        comparison.append(dict(model=name,**r['primary'],CI=str([round(v,3) for v in r['WAPE_95CI']]),offline_ms_per_origin=1000*r['seconds_per_origin'],MiB=r['model_bytes']/1024**2))
    valid=[dict(model=r['id'],WAPE=r['validation']['WAPE'],gain=r['gain']['point_gain'],gain_CI=str([round(v,5) for v in r['gain']['CI']]),regions_improved=r['gain']['regions_improved'],promote=r['promotion_gate_passes'] and r['id']!='current') for r in dev['records']]
    horizons=[]
    for name in labels:
        for h in ['1','4','12','24','48']:horizons.append(dict(model=name,step=h,individual_WAPE=final['results'][name]['at_lead'][h]['WAPE'],cumulative_WAPE=final['results'][name]['cumulative'][h]['WAPE']))
    region_comparison=dfs['may_per_region'].pivot(index='region',columns='model',values='WAPE')
    region_rows=[dict(region=int(r),frozen_Q1=row['frozen_Q1_current'],refit_current=row[chosen],tuned_XGB=row['matched_refit_tuned_xgboost'],tuned_LGB=row['matched_refit_tuned_lightgbm']) for r,row in region_comparison.iterrows()]
    selected_regimes=[r for r in regimes if r['model']==chosen]
    priorities=(REPORT/'research_priorities.md').read_text()
    params=metadata['refit_params']; baseparams={k:params.get(k) for k in ['n_estimators','max_depth','learning_rate','min_child_weight','subsample','colsample_bytree','reg_alpha','reg_lambda','objective','random_state','n_jobs']}
    sections=[f'''# UrbanFlow AI: next-generation model research

## Executive answers

1. **What limits accuracy?** Seasonal/calendar distribution shift, unpredictable regional demand innovations, sparse overnight counts, airport-centered behavior and a restricted completed-trip target. Observed peak underprediction partly reflects selecting on future actual demand; it is not evidence for an arbitrary inflation factor.
2. **What was tested?** Correct weekly profile adjacency, recency with shrinkage, a regularized origin-context calendar gate, post-path residual calibration, and an unchanged-recipe refit with genuine April history. Four research families; four novel configurations; no random search.
3. **What helped?** Adding April training history reduced matched May WAPE from **12.7634% to 12.5081%**, a **2.0007% relative error reduction** (0.2554 WAPE points; paired 95% gain interval [0.1643, 0.3342]). Weekly adjacency gave only 0.0186 point on March, below the frozen materiality threshold.
4. **What failed?** No novel recipe passed the March promotion gate. The contextual gate and post-path correction worsened validation; recency gains were small and uncertain.
5. **Below 10% on unseen recursive forecasts? No.** Selected May WAPE was **12.5081%**; the best predeclared pure XGBoost comparator was **12.0881%**. April's 10.2449% is a different period and is not a paired improvement baseline for May.
6. **What should be retained?** Retain the current recursive architecture and preserve the existing candidate/default. A separate Jan-April refit is available for review because added history helped on the matched test. Do not promote a novel architecture or switch to XGBoost based solely on its best May point estimate; the paired interval does not establish superiority.
7. **Next focus?** More representative chronological history and fresh June confirmation, holiday/calendar reliability, airport and event signals with verifiable availability at forecast origin, and explicit business agreement on the demand target. Prefer this over additional architecture complexity.

## Scope, integrity and evaluation

All work is local. No commits, pushes, PRs, deployments or production/default replacements were performed. Earlier experiments, candidates and user notebooks are protected by SHA-256 snapshots; see the preservation evidence linked below. Prior reports and the 75-experiment search were reviewed before the new analysis. New work lives in `ml/nextgen`, separate from the prior implementation.

April is **spent diagnostic data**, not an untouched test. No April forecast residuals fitted or selected new correction/gate weights. February out-of-fold calibration used a model fitted before February 14 22:00 New York; calibration origins span February 15–29. March validation used the frozen Jan-Feb base (exclusive cutoff February 29 22:00), with 57 chronological origins 13 hours apart. The current model's March predictions were reproduced exactly (maximum absolute difference 0).

Before opening May outcomes, the protocol, March selection, source hashes, comparator list and validation hash were frozen in {link(OUT/'selection_freeze.json')}. Novel promotion required at least 0.1 WAPE point gain, a strictly positive paired 95% gain interval, at least 20/30 regional improvements and no regional worsening over 2 points. No novel variant qualified; the current architecture was selected. Final fits used labels strictly before **April 30 22:00 New York**, with the unchanged 16-day feature warmup. The two-hour cutoff guard is intentional.

The fresh test covers May 2016: **57 origins**, every 13 elapsed hours from May 1 00:00, **30 regions × 48 steps = 82,080 predictions per model**. Each path receives only observations strictly before its origin; subsequent true demand cannot enter its recursive state. The complete test origin list is saved. Primary WAPE is `100 × sum(abs(actual − prediction)) / sum(abs(actual))` across every origin, region and all 48 leads. This is neither mean regional WAPE nor one-step accuracy. One step is 15 minutes; four is one hour; 48 is 12 hours. Cumulative metrics pool every step up to that lead; individual-lead metrics use only that step.

Uncertainty uses 1,000 paired circular moving-block resamples of three adjacent forecast origins, seed 42; complete 30-region/48-lead trajectories stay together. This respects within-path dependence and some adjacent-origin dependence, but one month and one spring holiday cannot establish seasonal robustness. These are score confidence intervals, not predictive intervals for individual counts.

## Deep diagnosis of the spent April test

The previous current model scored 10.2449% cumulative WAPE, MAE 13.5508, RMSE 21.7774 and R² 0.97215 on its 55 April origins. All 30 regional WAPE, MAE, RMSE, signed bias, demand mean/CV, zero fraction, peak and nighttime errors are in {link(REPORT/'spent_april_region_diagnosis.csv')}; all 48 regional leads are in {link(REPORT/'spent_april_region_by_lead.csv')}.

**Priority 1: calendar reliability and distribution changes.** Late-step bias is negative; standalone calendar bias at step 48 is −3.0593%. The short origin-level adjustment decays well before 12 hours, so persistent level shifts can be lost. Current midnight profile smoothing treats a weekday's 23:45 and the same weekday's 00:00 as adjacent; correct weekly adjacency is a plausible small repair, not a promise of meaningful forecasting gain.

**Priority 2: sparse relative error and airports.** April low/medium/high training-demand tiers have WAPE about 20.755/11.024/8.648%. Sparse region 2 averages about 4.31 retained pickups per interval: WAPE 36.16% but MAE only 1.56. Relative errors rise as the denominator shrinks; neither regional WAPE nor whole-system WAPE alone captures the operational tradeoff. Airports 3 and 6 have approximately 17.50/18.57% whole-period WAPE and 35.92/65.24% nighttime WAPE. April nighttime (00:00–05:59) WAPE is 13.602%, versus 9.808% during the rest of the day. Higher relative variability and different operating patterns are plausible; unobserved flight/event effects remain hypotheses.

Centroids were reconstructed using the saved coordinate scaler and fixed centers. Region 3 is approximately (−73.78474, 40.64664), within 0.90 km of the [FAA JFK reference](https://www.faa.gov/air_traffic/publications/atpubs/aip_html/part3_ad_2.0_new_york.html); region 6 is approximately (−73.86891, 40.77073), within 0.78 km of the [FAA LaGuardia reference](https://www.faa.gov/documentLibrary/media/Order/2022-08-19_FAA_Order_JO_7400.11G.pdf). These establish airport-centered proximity, not exact polygons. Residential/commercial/entertainment land use was not independently verified and no causal group claim is made from display names. The original Q1 mapping was checked on 33,234,199 retained trips with zero assignment mismatches.

![April region and centroid errors](<{(REPORT/'april_region_diagnosis.png').as_posix()}>)

**Priority 3: distinguish forecastable peaks from future innovations.** For outcomes above each region's training 95th percentile, April WAPE is 11.2729% and bias −9.7895%. Selecting on actual future peaks preferentially includes unexpectedly high realizations. Peaks defined at origin by calendar expectation instead have WAPE 8.0969%, bias −0.4741%; forecast-defined peaks have WAPE 7.9212%, bias −0.4722%. These different subsets do not prove calibration, but they undermine the case for a blanket peak multiplier. A correction must use origin-known features and improve out-of-sample absolute error, not merely remove conditional bias selected after the fact.

![April peak conditioning and target hour](<{(REPORT/'april_peak_and_time.png').as_posix()}>)

Hour-of-day, weekday, weekend and public-holiday breakdowns are in {link(REPORT/'spent_april_time_diagnosis.csv')}; forecast starting-hour breakdowns are in {link(REPORT/'spent_april_by_origin_hour.csv')}. April has no US federal holiday in this definition, so its holiday slice cannot test holiday generalization. Error is 16.04% at target hour 04:00 and 15.99% at 05:00, versus about 9.33–9.51% at 08:00–09:00 and 9.68–9.79% at 16:00–19:00. Weekend WAPE is 11.138%, weekday 9.856%; Sunday is 11.315% with −3.223% bias, while Wednesday is 8.501%. These are descriptive slices with unequal counts, not independently established causal effects. Forecasts starting at 19:00 score 13.460%, versus 7.934% at 10:00, but each hour has only a few origins and different dates/target-hour coverage; do not select a starting-hour correction from these outcomes.

All 48 aggregate leads, with signed drift, are in {link(REPORT/'spent_april_by_lead.csv')}. Error rises immediately from lead 1 (8.293%) to lead 4 (10.175%), then fluctuates rather than increasing monotonically; lead 48 is 10.618% with −2.994% bias. This supports a recursion/horizon reliability issue but does not isolate recursion from changing target-time mix. Regional lead tables permit checking whether drift is broad or concentrated rather than inferring it from one example path. On the fresh May test the analogous curves rise into the middle/late path before easing; the pattern is not a simple monotonic blow-up.

**Flattening is not itself an error target.** Forecast quarter-to-quarter absolute changes are 45.14% of actual changes, but at lead 48 cross-sectional forecast standard deviation retains 95.71% of actual standard deviation. Predictions should omit unpredictable count noise. These measures do not establish an avoidable flattening defect or a mathematical irreducible error floor.

## Exact architecture and controlled changes

The name `calendar_feedback_0.6` describes a **maximum**, not a constant weight. At zero-based lead k, inner output combines XGBoost with a previous-week anchor of weight `0.3*k/47`. Outer calendar weight is `0.6*(1-exp(-k/12))`, increasing from 0 to 0.5881. The effective raw learned-model weight at the last lead is approximately 0.2884. The calendar uses train-only region × weekday × quarter-hour medians with within-weekday smoothing. Its recent-origin ratio `(last-eight observed sum + 100)/(expected sum + 100)` is clipped to [0.6,1.6], with exponential decay over 12 steps. The **final blended prediction feeds back** to update subsequent lag/rolling features for all regions. Learned components and profiles cannot see future labels.

Horizon dependence already exists. A globally smaller/larger constant would not address regional reliability, calendar transitions or unseen holidays. We tested a low-capacity context gate instead of selecting weights from April errors. No arbitrary peak inflation or future-actual demand feature was introduced.

The unchanged-recipe final XGBoost configuration is `{json.dumps(baseparams, default=str)}`. Exact configurations and feature schema are saved in {link(OUT/'training_metadata.json')} and {link(OUT/'feature_schema.json')}. Existing lag/rolling, region, cyclical calendar and origin-safe spatial feature construction was preserved. The strongest prior pure XGBoost/LightGBM configurations were selected from the **prior March leaderboard**, then refitted on the same permitted Jan-April labels; V2 models are additional references, not substitutes for those stronger tuned baselines. No new tree search occurred.

## Research priorities fixed before validation

{priorities.split('# Priorities fixed before novel model validation',1)[-1]}

## Executed experiments and March selection

- **E1, weekly adjacency:** one ablation changes smoothing adjacency to a continuous 672-quarter week. A second adds a weighted median with 28-day half-life and shrinkage equivalent to four prior weeks, retaining train-only region profiles. Both preserve final-output recursive feedback.
- **E2, contextual calendar gate:** ten origin-safe features: lead fraction, target sine/cosine, weekend, night, origin gain, origin volatility, component disagreement, train region CV and relative calendar level. Standardized features are clipped to ±5; ten coefficients perturb the existing schedule through a bounded sigmoid. First-lead calendar weight stays zero. Fit uses February OOF original-count smooth MAE `sqrt(error²+1)` plus L2 0.2, L-BFGS-B capped at 200 iterations. Calibration uses baseline-trace features; March evaluation reruns the complete changed recursive path. This is a lightweight surrogate fit, not end-to-end recursive optimization.
- **E3, post-path correction:** the same regularized feature basis fits February OOF residuals, scaled by train regional means for parameterization and bounded by train OOF regional 95th-percentile absolute residuals. Corrected outputs are **not fed back**. It is explicitly labeled `postprocessed_recursive`, separate from genuine corrected-output recursion. The hypothesis was to avoid previously observed in-loop error amplification; it still failed validation.
- **E4, added history:** unchanged current architecture and fixed hyperparameters, Q1 frozen versus Jan-April refit on the same untouched May test. April observed labels train the base/profile, but April forecast residuals do not fit any new gate/correction. This is a data-history comparison, not a new architecture selected on May.

{table(valid,['model','WAPE','gain','gain_CI','regions_improved','promote'])}

Gains are current minus challenger WAPE points. Weekly adjacency improved 28 regions and has a barely positive interval, but its 0.0186-point gain is below the prespecified 0.1 materiality requirement. Recent shrinkage improved 23 regions but uncertainty includes zero. Context gating improved only 11 regions; post-path correction only five. Optimizers converged; lack of validation gain is not hidden as an optimizer failure. No failed direction was retuned after May. Separate MLflow experiment `urbanflow_next_generation` uses {link(OUT/'tracking.db')}; prior experiment stores were preserved.

## Fresh May test: matched comparisons

{table(comparison,['model','WAPE','MAE','RMSE','R2','bias_pct','CI','offline_ms_per_origin','MiB'])}

The selected recipe was the current architecture before May was opened. Adding April history produced a 0.2554-point improvement, paired 95% interval [0.1643, 0.3342], or 2.0007% relative error reduction. This is useful but modest; it does not achieve the target or prove robustness across seasons. Frozen Q1 versus refit is the valid paired improvement comparison.

The strongest prior pure XGBoost has the best May point estimate (12.0881%), reversing the earlier March ranking. The paired gain of selected refit relative to that XGBoost is **−0.4200 points**, interval **[−1.0998, 0.1509]**; the interval includes zero. Relative to tuned LightGBM, gain is −0.0388 points, interval [−0.4314, 0.3221]. Do not interpret the observed ranking as a conclusive architecture win or change model selection retrospectively. Keep pure XGBoost as a prespecified challenger for the next genuinely fresh period.

### Individual lead versus cumulative recursive WAPE

{table(horizons,['model','step','individual_WAPE','cumulative_WAPE'])}

All 48 individual leads are in {link(REPORT/'may_by_lead.csv')}; all origin-level scores are in {link(REPORT/'may_by_origin.csv')}. Overnight, peak, tier, weekend and holiday definitions below are identical across models. Regional tiers and peak thresholds are fitted only on permitted Jan-April history; realized peak selection retains the interpretation caveat above.

![May lead and origin comparisons](<{(REPORT/'may_horizon_and_origin.png').as_posix()}>)

### Selected refit regime results

{table(selected_regimes,['regime','n','WAPE','MAE','RMSE','bias_pct'])}

The May 30 Memorial Day slice is a **post-test diagnosis**, not a reason to exclude observations or select a new model. Refit WAPE is 44.1099% on that slice, versus 11.7862% on other days; bias is +21.0587% versus +0.5276%. The tuned pure XGBoost holiday WAPE is 32.5152%, versus 11.6215% otherwise. All holiday predictions remain in the primary score. The architecture's weekday profile does not condition on holidays, while the learned tree has a calendar holiday feature; strong seasonal anchoring can plausibly suppress that distinction. A single event cannot establish causation, and non-holiday error still exceeds 10%. March has no federal holiday and Q1 contains few winter holiday examples, so gate validation offers little evidence for spring holiday reliability. No May-specific holiday weights were fitted.

### Every region, identical May forecasts

{table(region_rows,['region','frozen_Q1','refit_current','tuned_XGB','tuned_LGB'])}

Full regional MAE, RMSE, R², bias, demand mean/CV, zeros, peak and night scores for **every comparator** are in {link(REPORT/'may_per_region.csv')}; corresponding regime tables are in {link(REPORT/'may_regime_metrics.csv')}. All target hours are in {link(REPORT/'may_by_target_hour.csv')}.

![Predetermined middle-origin path examples](<{(REPORT/'may_fixed_example_paths.png').as_posix()}>)

These examples use the middle chronological origin and fixed sparse/airport/other IDs [2,3,6,18]. They are illustrations; no example was selected to optimize a claim, and headline metrics include all origins.

## Data quality, target validity and feasible additional data

January raw audit scanned 10,906,858 records and exactly reproduced 10,505,157 retained trips. Marginal pass counts: pickup geography 10,725,121; dropoff geography 10,648,641; fare 10,887,046; distance 10,760,160. Reasons overlap and must not be summed. In airport-centered region 3, 232,807 of 254,836 pickups in geography survive all filters (8.65% excluded); region 6 retains 268,915 of 278,951 (3.60% excluded). Exact overlapping reason counts are in {link(REPORT/'data_quality_and_target_audit.json')}.

The target is **retained completed taxi trips**, not all taxi demand, requests, rejected rides or even all pickups. Both pickup and dropoff must fall in latitude [40.60,40.85], longitude [−74.05,−73.70], fare [0.50,81] and distance [0.25,24.43]. Post-trip fare/distance/dropoff filters systematically exclude genuine journeys, including airport trips. They were kept for fair comparison; changing them requires a new explicitly named target and evaluation protocol. Monthly historical TLC availability is not a live observation feed. A production service forecasting actual pickups needs an operational source and target agreement.

The Jan-April panel has no duplicate region/timestamps, nonfinite counts or elapsed-time grid gaps and retains all 30 regions. New May was checked for the same contiguous 15-minute elapsed grid after joining history. Zeros are preserved, not interpolated, but aggregation alone cannot certify that a source outage did not appear as zero. New York timezone-aware elapsed indexing handles DST; the earlier 34-record DST quarantine is preserved. The fixed geography is not refitted for May. Outlier filters are unchanged, so comparisons do not gain by deleting hard targets.

May coordinate records were available from the official [NYC 2016 Yellow Taxi dataset](https://data.cityofnewyork.us/Transportation/2016-Yellow-Taxi-Trip-Data/uacg-pexx). Modern location-ID-only files cannot simply reproduce the fixed coordinate clusters. Before acquisition, feasibility estimated 1.90 GiB available RAM, 69.45 GiB disk, roughly 12 million monthly trips, less than 1 GiB extra persisted data and 31 bounded daily requests. Two source requests ran concurrently; four native training threads were used. Queries selected only pickup timestamp and coordinates **after the same original filters**, daily responses stayed below the one-million-record limit, and records were aggregated then discarded. All 31 daily aggregate files/request metadata and response hashes are saved, with final panel and geography hashes in {link(OUT/'may_provenance.json')}. This is a reproducible bounded download, not an unnecessary full raw archive. Public-source completeness beyond the recorded response/grid checks is not proven.

Additional April examples helped, but Jan-April still lacks much seasonal and event coverage. No extra June outcomes were examined or used to choose these experiments. For future external data, realized weather or realized flights after origin would leak information; archived weather forecasts, published schedules and preannounced events need documented as-of availability. Their gains remain unmeasured.

## Local artifact, serving and validation

Review artifact: {link(OUT/'nextgen_candidate.joblib')}. It contains the selected unchanged-recipe **Jan-April data refit**, not a proven new architecture. The prior `advanced_candidate_v3.joblib` remains preserved. Schema, training metadata, protocol, selected-source hashes and observation seed are separate files in the new artifact directory. Reproduction/opt-in commands are in {link(ROOT/'urbanflow/ml/nextgen/README.md')}. Existing API defaults and deployed application were not switched.

**106 tests passed**, with one existing FastAPI/httpx deprecation warning. Local real HTTP validation passed **45 valid requests** and **9 invalid requests**, including all 30 regions, horizons 1/4/12/24/48, horizon-prefix consistency, midnight/weekend/holiday timestamps, history/training cutoff guards and invalid inputs. The process tree measured {system['process_tree_RSS_MiB']:.2f} MiB RSS. HTTP latency below includes request parsing and history loading on this local machine; small sample sizes and cold starts limit generalization. Offline batched all-region origin throughput in the comparison table is a different measurement and should not be presented as HTTP service latency.

{table(latency,['horizon','n','p50_ms','p95_ms'])}

Evidence: {link(REPORT/'local_system_results.json')}, {link(ROOT/'reports/nextgen_full_regression_tests.log')}, {link(REPORT/'preservation_checks.json')}. Prediction uncertainty bands are not supplied by the API; bootstrap score intervals should not be reused as forecast bands.

## Recommendation and next phase

Retain the **current recursive architecture** and existing default pending review. Preserve the separate data refit as the strongest selected local recipe for this phase; added history has positive paired evidence on May, while every novel recipe failed the frozen March gate. No new architectural winner is justified. Keep the stronger prior pure XGBoost as a serious challenger; predeclare both before fresh June confirmation instead of treating May as a reusable selection set.

Prioritize representative historical months and holiday/event coverage, verify operational target and feed availability, and investigate airport-specific origin-known signals. Weekly adjacency is a maintainability improvement with a very small measured effect; it was not bundled into the selected model. Do not spend further budget on the tested gate, shrinkage or residual correction without a new falsifiable hypothesis and additional development data. There is no measured guarantee that the present target and inputs can reach below 10%, and no proven lower bound saying they cannot.

May is now spent. Future experiments must document it as development/diagnostic history and reserve a new chronological test. All new evidence and source are local and reviewable; no automatic adoption or remote action was taken.
''']
    target=ROOT/'reports/next_generation_model_research.md'
    target.write_text('\n'.join(sections),encoding='utf-8')
    print('Report:',target)
    print('Selected May regimes:',json.dumps(selected_regimes,indent=2))

if __name__=='__main__':main()
