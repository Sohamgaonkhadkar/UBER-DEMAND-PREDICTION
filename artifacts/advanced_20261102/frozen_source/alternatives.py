"""Validation-only seasonal, ensemble, hybrid and OOF correction experiments."""
from dataclasses import dataclass
import copy,json,time
import joblib,numpy as np,pandas as pd
import mlflow
from .core import *
from .protocol import *
from .research import Research

@dataclass
class SeasonalForecast:
    scales:object
    kind:str='weekly'
    decay:float=24.
    def forecast(self,y,times,positions,horizon=48):
        h=histories(y,positions);b,_,r=h.shape
        result=[]
        for k in range(horizon):
            target=pd.DatetimeIndex(times[positions])+pd.Timedelta(minutes=15*k)
            x=make_features(h,target,list(range(r)),FeatureConfig(calendar=False),self.scales,age=k)
            if self.kind=='weekly':p=x.clock_day_7.to_numpy()
            else:p=x.seasonal_anchor.to_numpy()
            if self.kind=='adaptive':
                ratio=np.clip(x.recent_week_ratio.to_numpy(),.5,1.5)
                p=p*(1+(ratio-1)*np.exp(-k/self.decay))
            result.append(np.maximum(0,p.reshape(b,r)))
        return np.stack(result,axis=1)

@dataclass
class CompositeForecast:
    first:object
    second:object
    weight:float=.5
    switch:int=0
    def forecast(self,y,times,positions,horizon=48):
        a=self.first.forecast(y,times,positions,horizon);b=self.second.forecast(y,times,positions,horizon)
        if self.switch:
            out=b.copy();out[:,:self.switch]=a[:,:self.switch];return out
        return self.weight*a+(1-self.weight)*b

@dataclass
class CalendarProfile:
    profile:object
    decay:float=48.
    adaptive:bool=True
    def forecast(self,y,times,positions,horizon=48):
        positions=np.asarray(positions,dtype=int)
        expected=[]
        for offset in range(-8,0):
            t=pd.DatetimeIndex(times[positions+offset])
            expected.append(self.profile[np.asarray(t.day_of_week),np.asarray(t.hour*4+t.minute//15)])
        expected=np.stack(expected,axis=1).sum(axis=1)
        recent=y[positions[:,None]-8+np.arange(8)[None,:]].sum(axis=1)
        gain=np.clip((recent+100)/(expected+100),.6,1.6)
        out=[]
        for k in range(horizon):
            t=pd.DatetimeIndex(times[positions])+pd.Timedelta(minutes=15*k)
            p=self.profile[np.asarray(t.day_of_week),np.asarray(t.hour*4+t.minute//15)]
            if self.adaptive:p=p*(1+(gain-1)*np.exp(-k/self.decay))
            out.append(np.maximum(0,p))
        return np.stack(out,axis=1)

def fit_calendar(r,decay=48.,adaptive=True):
    values=pd.DataFrame(r.y[HISTORY:r.end],index=r.times[HISTORY:r.end])
    profile=values.groupby([values.index.day_of_week,values.index.hour*4+values.index.minute//15]).median().reindex(pd.MultiIndex.from_product([range(7),range(96)]))
    if profile.isna().any().any():raise ValueError('Insufficient calendar profile training coverage')
    p=profile.to_numpy().reshape(7,96,30)
    p=(np.roll(p,1,axis=1)+2*p+np.roll(p,-1,axis=1))/4
    return CalendarProfile(p,decay,adaptive)

def register(r,name,bundle,architecture,strategy,details,eligible=True,training_seconds=0.):
    if any(a['id']==name for a in r.records):return next(a for a in r.records if a['id']==name)
    if (OUT/'selection_freeze.json').exists():raise RuntimeError('Selection is frozen; new alternatives are forbidden')
    started=time.perf_counter();scores,pred=r.evaluate(bundle)
    if strategy=='blend':
        a=bundle.first.forecast(r.y,r.times,r.pos,48);b=bundle.second.forecast(r.y,r.times,r.pos,48)
        base=a if metrics(r.truth,a)['WAPE']<=metrics(r.truth,b)['WAPE'] else b
        rng=np.random.default_rng(42);differences=[];n=len(pred)
        for _ in range(1000):
            starts=rng.integers(0,n,size=(n+2)//3);idx=np.concatenate([(s+np.arange(3))%n for s in starts])[:n]
            differences.append(metrics(r.truth[idx],pred[idx])['WAPE']-metrics(r.truth[idx],base[idx])['WAPE'])
        ci=np.quantile(differences,[.025,.975]).tolist();details=dict(details,paired_block_gain_ci=ci)
        eligible=eligible and ci[1]<0
    record=dict(id=name,architecture=architecture,strategy=strategy,transform='composite',features=details,
        requested_params=details,fitted_params=details,training_end=str(r.cut),validation_origins=len(r.pos),
        validation=scores,training_seconds=training_seconds,runtime_seconds=time.perf_counter()-started+training_seconds,eligible=eligible,status='completed')
    dest=OUT/'models'/f'{name}.joblib';dest.parent.mkdir(exist_ok=True)
    joblib.dump(bundle,dest);record['artifact']=str(dest.relative_to(ROOT));record['model_bytes']=dest.stat().st_size
    np.savez_compressed(OUT/'validation_predictions'/f'{name}.npz',prediction=pred)
    r.records.append(record);(OUT/'leaderboard.json').write_text(json.dumps(r.records,indent=2,default=str))
    with mlflow.start_run(run_name=name):
        mlflow.log_params(dict(architecture=architecture,strategy=strategy,details=json.dumps(details),validation_origins=len(r.pos),horizon=48))
        mlflow.log_metrics({f'validation_{k}':v for k,v in scores.items() if isinstance(v,(float,int)) and v is not None})
        mlflow.log_metric('training_seconds',training_seconds)
        mlflow.log_artifact(str(dest))
    print('RESULT',name,scores['WAPE'],flush=True);return record

def main():
    r=Research()
    for kind in ['weekly','profile','adaptive']:
        for decay in ([8.,24.,48.] if kind=='adaptive' else [24.]):
            register(r,f'seasonal_{kind}_{int(decay)}',SeasonalForecast(r.scales,kind,decay),'seasonal','seasonal',dict(kind=kind,decay=decay))
    for decay in [12.,48.,96.]:
        started=time.perf_counter();model=fit_calendar(r,decay);duration=time.perf_counter()-started
        register(r,f'calendar_profile_{int(decay)}',model,'calendar_profile','seasonal',dict(decay=decay,fit_end=str(r.cut),method='training-only weekday/quarter median profile; smoothed adjacent quarters, observed-origin gain'),training_seconds=duration)
    records=[a for a in r.records if a.get('eligible') and a.get('artifact')]
    rec=min([a for a in records if a['strategy']=='recursive'],key=lambda a:a['validation']['WAPE'])
    direct=min([a for a in records if a['strategy']=='direct'],key=lambda a:a['validation']['WAPE'])
    rb=joblib.load(ROOT/rec['artifact']);db=joblib.load(ROOT/direct['artifact'])
    for switch in [4,12,24]:
        register(r,f'hybrid_switch_{switch}',CompositeForecast(rb,db,switch=switch),'hybrid','hybrid',dict(recursive=rec['id'],direct=direct['id'],switch=switch))
    for weight in [.25,.5,.75]:
        register(r,f'blend_recursive_direct_{weight}',CompositeForecast(rb,db,weight=weight),'ensemble','blend',dict(first=rec['id'],second=direct['id'],weight=weight))
    # Recursive model with seasonal anchoring. Feedback uses the blended values.
    if isinstance(rb,ForecastModel) and rb.config.seasonal:
        for weight in [.15,.3,.5]:
            anchored=copy.deepcopy(rb);anchored.anchor_weight=weight
            register(r,f'recursive_anchor_{weight}',anchored,'anchored_'+rec['architecture'],'recursive',dict(base=rec['id'],anchor_weight=weight))
    # A correction learned from predictions strictly out of training time, not March errors.
    correction_started=time.perf_counter()
    early=Research();early.cut=pd.Timestamp('2016-02-15',tz=TZ)-pd.Timedelta('2h')
    early.end=int(early.times.searchsorted(early.cut));early.scales=np.maximum(1,early.y[:early.end].mean(axis=0))
    if isinstance(rb,ForecastModel):
        fit=early.train(rec['architecture'],rb.config,rb.transform,rb.strategy,rec['requested_params'],rec.get('region_specific',False))
        oo=origins('2016-02-15','2016-03-01');pos=early.times.get_indexer(oo)
        p=fit.forecast(early.y,early.times,pos,48);truth=np.stack([early.y[o:o+48] for o in pos])
        bias=np.stack([(truth[:,i*12:(i+1)*12]-p[:,i*12:(i+1)*12]).mean(axis=(0,1)) for i in range(4)])
        bias=np.clip(bias,-.15*r.scales[None,:],.15*r.scales[None,:])
        corrected=copy.deepcopy(rb);corrected.bias=bias
        register(r,'oof_bias_correction',corrected,'oof_corrected_'+rec['architecture'],'recursive',dict(base=rec['id'],fit_end=str(early.cut),correction_origins=[str(t) for t in oo],bias=bias.tolist()),training_seconds=time.perf_counter()-correction_started)

if __name__=='__main__':
    # Import the canonical module so persisted wrapper classes never use __main__.
    from ml.advanced.alternatives import main as entry
    entry()
