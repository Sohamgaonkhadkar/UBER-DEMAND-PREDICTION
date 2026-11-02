"""Fixed weather ablation; reused development windows, no claim of a fresh test.
Run from the repository root with Python 3.14. --feature-cache can reuse the
original causal feature cache; without it, equivalent rows are rebuilt.
"""
from pathlib import Path
import argparse,copy,dataclasses,gc,hashlib,json,sys,time
import joblib,numpy as np,pandas as pd
from sklearn.base import clone
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'urbanflow'))
from ml.advanced.core import HISTORY,training_rows
from ml.nextgen.models import profile_from_data
from ml.advanced.calendar_anchoring import CalendarAnchoredRecursive
from ml.weather_20261102 import WeatherCalendarRecursive,weather_values,WEATHER_FEATURES
OUT=ROOT/'artifacts/weather_20261102'

def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def metrics(y,p):
    error=p-y;den=np.abs(y).sum();total=((y-y.mean())**2).sum()
    return dict(WAPE=float(100*np.abs(error).sum()/den),MAE=float(np.abs(error).mean()),RMSE=float(np.sqrt(np.mean(error**2))),R2=float(1-(error**2).sum()/total),bias_percent=float(100*error.sum()/den))

def report(y,p):
    return dict(primary=metrics(y,p),cumulative={str(h):metrics(y[:,:h],p[:,:h]) for h in [1,4,12,24,48]},at_lead={str(h):metrics(y[:,h-1],p[:,h-1]) for h in [1,4,12,24,48]})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--feature-cache',type=Path);args=parser.parse_args()
    frozen_path=ROOT/'artifacts/v4_20261102/candidate_v4.joblib'
    frozen_hash=hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    assert frozen_hash=='7b96eaf0f9c9bec866e29677f9df9cb7647a99619c1c4589c35632a8f845c2c4'
    frozen=joblib.load(frozen_path);template=frozen['model'].base
    weather=pd.read_parquet(OUT/'weather_hourly.parquet')
    panel=pd.read_parquet(ROOT/'artifacts/v4_20261102/inference_observations.parquet')
    wide=panel.pivot(index='datetime',columns='region',values='total_pickups').sort_index();times=wide.index;y=wide.to_numpy(dtype=np.float32)
    if args.feature_cache:
        frame=pd.read_parquet(args.feature_cache);target=frame.pop('__target').to_numpy();ts=pd.DatetimeIndex(frame.pop('__timestamp'));x=frame
    else:
        positions=np.arange(HISTORY,times.searchsorted(pd.Timestamp('2016-06-01',tz='America/New_York')))
        x,target,ts=training_rows(y,times,positions,template.config,template.scales,template.neighbors);ts=pd.DatetimeIndex(ts)
    assert list(x.columns)==template.feature_name_
    wx=x.copy();values=weather_values(weather,ts)
    for j,name in enumerate(WEATHER_FEATURES):wx[name]=values[:,j].astype(np.float32)
    records={};models={}
    rounds=template.model.get_booster().num_boosted_rounds()
    def fit(cut,weather_enabled):
        mask=np.asarray(ts<cut);features=wx if weather_enabled else x
        model=clone(template.model).set_params(n_jobs=4,n_estimators=rounds,early_stopping_rounds=None)
        begun=time.perf_counter();model.fit(features.loc[mask],target[mask])
        base=copy.deepcopy(template);base.model=model;base.config=dataclasses.replace(template.config,weather=weather_enabled);base.bias=None
        end=times.searchsorted(cut);calendar=profile_from_data(y,times,end)
        result=WeatherCalendarRecursive(base,calendar,weather) if weather_enabled else CalendarAnchoredRecursive(base,calendar,.6,12.)
        record=dict(cutoff=cut.isoformat(),labels=int(mask.sum()),feature_count=features.shape[1],trees=rounds,training_seconds=time.perf_counter()-begun)
        print('FIT',weather_enabled,record,flush=True);return result,record
    cut=pd.Timestamp('2016-04-30T22:00:00',tz='America/New_York')
    origins=pd.date_range('2016-05-01T00:00:00','2016-05-31T12:00:00',freq='13h',tz='America/New_York')
    positions=times.get_indexer(origins);assert (positions>=HISTORY).all();truth=np.stack([y[p:p+48] for p in positions])
    for enabled,name in [(False,'matched_no_weather'),(True,'lagged_weather')]:
        model,record=fit(cut,enabled);p=model.forecast(y,times,positions,48);records[name]=dict(**report(truth,p),training=record)
        print('MAY',name,records[name]['primary'],flush=True)
        models[name]=model
    gain=records['matched_no_weather']['primary']['WAPE']-records['lagged_weather']['primary']['WAPE']
    selection={'may_gain_wape_points':gain,'weather_default_eligible':gain>=.1,'threshold_wape_points':.1,'selection_used_june':False,'may_is_reused_development':True}
    save('selection.json',selection);save('may_comparison.json',{'origins':[o.isoformat() for o in origins],'records':records})
    final,training=fit(pd.Timestamp('2016-05-31T22:00:00',tz='America/New_York'),True)
    bundle=dict(frozen);bundle.update(model=final,candidate_id='weather_calendar_feedback_v1',weather_source='Open-Meteo ERA5 historical reanalysis',weather_lag_hours=3,feature_columns=list(final.base.feature_name_))
    joblib.dump(bundle,OUT/'candidate_weather.joblib',compress=3)
    save('feature_schema.json',{'version':1,'candidate_id':bundle['candidate_id'],'feature_columns':final.base.feature_name_,'weather_feature_columns':WEATHER_FEATURES,'weather_lag_hours':3})
    save('training_metadata.json',training)
    # Selection is persisted before June is read as diagnostic evidence.
    plan=json.loads((ROOT/'artifacts/v4_20261102/protocol.json').read_text())
    origins=pd.to_datetime(plan['primary_test']['origins'],utc=True).tz_convert('America/New_York');positions=times.get_indexer(origins)
    actual=np.stack([y[p:p+48] for p in positions]);pred=final.forecast(y,times,positions,48)
    baseline=frozen['model'].forecast(y,times,positions,48)
    record={'period':'June 2016; reused spent diagnostic evidence','origins':[o.isoformat() for o in origins],'origin_count':len(origins),'points':int(actual.size),'selection':selection,'records':{'weather':report(actual,pred),'frozen_v4':report(actual,baseline)}}
    save('june_diagnostics.json',record);np.savez_compressed(OUT/'june_predictions.npz',actual=actual,weather=pred,frozen_v4=baseline)
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()==frozen_hash
    print('JUNE',record['records']['weather']['primary'],'selection',selection,flush=True)
if __name__=='__main__':main()
