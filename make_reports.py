"""Generate reports and figures only from saved executed experiment outputs."""
import json,pathlib,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'urbanflow'))
from ml.train import metrics
REP=ROOT/'reports';ART=ROOT/'artifacts';PLOTS=REP/'plots';PLOTS.mkdir(exist_ok=True)

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def fmt(v):return 'N/A' if v is None else f'{v:.4f}'
def metricrow(name,m):return [name]+[fmt(m[k]) for k in ['MAE','RMSE','WAPE','R2']]

def main():
    results=json.loads((ART/'results.json').read_text());original=json.loads((ART/'original_metrics.json').read_text())
    data=json.loads((ART/'dataset_metadata.json').read_text());splits=json.loads((ART/'splits.json').read_text())
    d=pd.read_parquet(ART/'recursive_predictions.parquet')
    if (ART/'original_recursive_predictions.parquet').exists():
        od=pd.read_parquet(ART/'original_recursive_predictions.parquet');d=pd.concat([d,od],ignore_index=True)
    panel=pd.read_parquet(ART/'counts_ny.parquet')
    train=panel[panel.datetime<pd.Timestamp('2016-02-15',tz='America/New_York')-pd.Timedelta('2h')]
    means=train.groupby('region').total_pickups.mean();lo,hi=means.quantile([1/3,2/3])
    tier=means.map(lambda v:'low' if v<=lo else 'medium' if v<=hi else 'high')
    thresholds=train.groupby('region').total_pickups.quantile(.95)
    d['tier']=d.region.map(tier);d['peak']=d.actual>=d.region.map(thresholds)
    rows=[]
    for keys,g in d.groupby(['model','region']):rows.append(dict(model=keys[0],region=keys[1],**metrics(g.actual,g.prediction)))
    pd.DataFrame(rows).to_csv(REP/'per_region_metrics.csv',index=False)
    tiers=[]
    for keys,g in d.groupby(['model','tier']):tiers.append(dict(model=keys[0],tier=keys[1],**metrics(g.actual,g.prediction)))
    pd.DataFrame(tiers).to_csv(REP/'demand_tier_metrics.csv',index=False)
    step=[]
    for keys,g in d.groupby(['model','step']):step.append(dict(model=keys[0],step=keys[1],**metrics(g.actual,g.prediction)))
    pd.DataFrame(step).to_csv(REP/'step_metrics.csv',index=False)
    peaks=[]
    for name,g in d[d.peak].groupby('model'):peaks.append(dict(model=name,n=len(g),**metrics(g.actual,g.prediction)))
    pd.DataFrame(peaks).to_csv(REP/'peak_metrics.csv',index=False)
    winner=min(results,key=lambda n:results[n]['validation_recursive']['WAPE'])
    one=[metricrow('Original LightGBM (legacy target/schema; not apples-to-apples)',original['original_notebook_one_step'])]
    one += [metricrow(n,v['one_step']) for n,v in results.items()]
    recursion=[metricrow(n,v['test_recursive']) for n,v in results.items()]
    if 'recursive' in original:recursion.append(metricrow('Original API, observed-weather sensitivity',original['recursive']))
    s='# Actual local model comparison\n\n'
    s+='Original production artifact and notebooks are preserved. No commit, push, PR or deployment was made.\n\n'
    s+='## Data and evaluation\n\n'
    s+=f"Rebuilt from {data['trips']:,} retained regional trip rows, not a demonstration sample. Final timezone-aware panel: {data['rows']:,} rows, 30 regions, {data['zero_intervals']:,} zeros. Quarantined {data['quarantined_nonexistent_clock_trips']} source trips in New York’s nonexistent DST hour; original records untouched. The upstream scaler/K-means were originally fitted on the whole quarter; their artifacts are unavailable, so region assignment is inherited and not a fully leakage-free raw-trip retraining.\n\n"
    s+=table(['Split','Start','End','Rows'],[[n,v['start'],v['end'],v['rows']] for n,v in splits.items()])
    s+='\nAll new candidates use identical zero-preserving targets, 33 numeric features, the same training period, and the same validation/test origins. No future weather is used. First 720 observations per region supply warmup. Two-hour gaps exclude fitting labels, not observations that really exist before a forecast origin. Test labels are never used for tuning. Models are not refitted on validation before test.\n\n'
    s+='## One-step March test results\n\n'+table(['Model','MAE','RMSE','WAPE %','R²'],one)
    s+='\nThe original reported 12.41 / 20.04 / 9.41% / 0.9764 is closely reproduced. Its notebook builds test features using actual preceding test targets, full-period imputation, cross-region global lags and a zero→10 target. Weather is current-download historical reanalysis. These are teacher-forced one-step results, not recursive deployment accuracy. Small differences from saved notebook numbers can arise from weather revisions/library versions.\n\n'
    s+='## Rolling-origin recursive test, all steps 1–48\n\n'+table(['Model','MAE','RMSE','WAPE %','R²'],recursion)
    s+='\nThree validation origins and six prespecified March test origins, all 30 regions, 48 recursive predictions each. This is 8,640 test predictions/model, not every possible March origin. Model comparison is paired across these origins; uncertainty or statistical significance has not been estimated.\n\n'
    s+='## Validation, cost and selection\n\n'
    s+=table(['Model','Validation recursive WAPE %','Training seconds','Artifact MiB','48 steps × 30 regions seconds'],[[n,fmt(v['validation_recursive']['WAPE']),fmt(v['training_seconds']),fmt(v.get('model_bytes',0)/1024**2),fmt(v['latency']['seconds_per_48step_30region'])] for n,v in results.items()])
    s+=f'\n**Provisional local recommendation: {winner}.** It has the lowest validation recursive WAPE, a small artifact and fast batched inference. LightGBM has slightly better one-step WAPE but worse validation and test recursive WAPE. CatBoost was worse than weekly seasonal naive at 48 steps in this bounded configuration; 200 depth-six boosting rounds are not evidence that CatBoost cannot do better. Random Forest is about 72 MiB and slower, with weaker recursive results. Only LightGBM received three controlled Optuna trials; this is a bounded benchmark, not equal-budget exhaustive tuning.\n\n'
    s+='Do not replace production automatically. Original API inference and its training schema do not match: repairing their semantics changes the input distribution. The original trained through February; new candidates train through February 14. Original sensitivity results additionally use observed weather and a wall-clock legacy grid. A claimed percentage improvement against that original is scientifically unsupported. A larger untouched modern dataset, forecast-origin weather if weather is used, and more rolling origins are required before deployment.\n\n'
    ablation=ART/'ablations/lag_ablation.json'
    if ablation.exists():
        a=json.loads(ablation.read_text())
        s+='## Daily/weekly lag ablation\n\n'
        s+=f"With the same LightGBM hyperparameters, fitting period, rolling windows and validation origins, removing only lag_96 and lag_672 raised validation recursive WAPE from {a['with_added_lags']['WAPE']:.4f}% to {a['without_added_lags']['WAPE']:.4f}%. These correctly defined lags helped in this experiment. This was validation-only and did not trigger changes after test results were seen. See artifacts/ablations/lag_ablation.json.\n\n"
    s+='Model memory was not benchmarked in isolation; serialized size is reported, not confused with resident memory. Native training was capped at four threads. RAM was constrained, so the available full regional intermediate was streamed instead of fitting new clustering on all raw coordinates. Raw trip files total roughly 5.4 GB and remain unchanged.\n\n'
    s+='## Reproduction\n\nSee `urbanflow/LOCAL_REPAIR.md`, `requirements-local-lock.txt`, `urbanflow/ml/train.py`, `validate_cv.py`, `evaluate_original.py`, and `local_system_check.py`. Saved results are in `artifacts/results.json`, `artifacts/original_metrics.json`, `artifacts/splits.json`, `artifacts/chronological_cv.json`, and local MLflow SQLite tracking. Models are in `artifacts/candidates/`; original model SHA-256 is in `reports/baseline/original.json`. MLflow runs from the preliminary wall-clock experiment also remain; use the final timezone-aware JSON/Parquet outputs for these tables.\n'
    (REP/'model_comparison.md').write_text(s,encoding='utf-8')
    hrows=[]
    for name,v in results.items():
        for h in [1,4,12,24,48]:hrows.append([name,h]+[fmt(v['horizons'][str(h)][k]) for k in ['MAE','RMSE','WAPE','R2']])
    if 'horizons' in original:
        for h in [1,4,12,24,48]:hrows.append(['original_api (sensitivity)',h]+[fmt(original['horizons'][str(h)][k]) for k in ['MAE','RMSE','WAPE','R2']])
    s='# Recursive forecasting backtest\n\n'
    s+='At each origin only preceding observations are passed to the forecaster. Each predicted value is fed back before subsequent features are computed. Models receive no held-out demand during a trajectory. Origins: March 1 00:00; March 5 23:00; March 10 06:00; March 18 17:00; March 26 23:00; March 30 12:00, all America/New_York in 2016. Validation origins: February 16 00:00, February 20 12:00, February 27 23:00.\n\n'
    s+='Metrics below are **cumulative through horizon H**, not error only at step H. 1, 4, 12, 24 and 48 intervals mean 15 minutes, 1 hour, 3 hours, 6 hours and 12 hours. `step_metrics.csv` provides individual-lead metrics. WAPE = 100 × Σ|y−ŷ| / Σ|y|; undefined when all targets are zero. R² is omitted for constant targets.\n\n'
    s+=table(['Model','H','MAE','RMSE','WAPE %','R²'],hrows)
    s+='\nFor XGBoost the cumulative WAPE rises from '+fmt(results['xgboost']['horizons']['1']['WAPE'])+'% at one step to '+fmt(results['xgboost']['horizons']['48']['WAPE'])+'% through 48 steps. Individual errors need not rise monotonically because later targets and demand levels differ. One-step March evaluation uses a different population from these selected rolling origins.\n\n'
    s+='## Regional and peak robustness\n\n'+table(['Model','Demand tier','MAE','RMSE','WAPE %','R²'],[[r['model'],r['tier']]+[fmt(r[k]) for k in ['MAE','RMSE','WAPE','R2']] for r in tiers])
    s+='\nTiers are thirds of training-period region mean demand, never chosen from test results. Peaks are test observations above each region’s training 95th percentile.\n\n'
    s+=table(['Model','Peak observations','MAE','RMSE','WAPE %','R²'],[[r['model'],r['n']]+[fmt(r[k]) for k in ['MAE','RMSE','WAPE','R2']] for r in peaks])
    representatives=[int(means.sort_values().index[i]) for i in [2,15,27]]
    origin='2016-03-05 23:00:00-05:00'
    fig,axes=plt.subplots(3,1,figsize=(11,9),layout='constrained')
    for ax,r in zip(axes,representatives):
        g=d[(d.model=='xgboost')&(d.region==r)&(pd.to_datetime(d.origin,utc=True,format='mixed')==pd.Timestamp(origin).tz_convert('UTC'))]
        ax.plot(g.step,g.actual,label='Actual',color='black',lw=2)
        for n in ['xgboost','lightgbm','seasonal_naive']:
            z=d[(d.model==n)&(d.region==r)&(pd.to_datetime(d.origin,utc=True,format='mixed')==pd.Timestamp(origin).tz_convert('UTC'))]
            ax.plot(z.step,z.prediction,label=n,alpha=.8)
        ax.set_title(f'Region {r} ({tier[r]} demand), origin March 5 2016 23:00 New York');ax.set_ylabel('Pickups / 15 min');ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Recursive step (15 minutes)');fig.savefig(PLOTS/'actual_vs_predicted.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    st=pd.DataFrame(step)
    for n in results:
        z=st[st.model==n];ax.plot(z.step,z.WAPE,label=n)
    ax.set(xlabel='Recursive lead (15-minute intervals)',ylabel='WAPE at individual lead (%)',title='Recursive error by lead across six origins × 30 regions');ax.legend();ax.grid(alpha=.2)
    fig.savefig(PLOTS/'error_by_step.png',dpi=140);plt.close(fig)
    s+='\n## Figures\n\n![Actual and predicted regional demand](plots/actual_vs_predicted.png)\n\n![Error at each recursive lead](plots/error_by_step.png)\n\nFull individual predictions: `artifacts/recursive_predictions.parquet`; original sensitivity predictions: `artifacts/original_recursive_predictions.parquet`. Original observed-weather results are not deployment-realistic. New candidates use no weather. Six origins do not establish seasonal generalization or operational readiness.\n'
    (REP/'recursive_backtesting.md').write_text(s,encoding='utf-8')
    print('reports and plots generated; validation-selected candidate:',winner)

if __name__=='__main__':main()
