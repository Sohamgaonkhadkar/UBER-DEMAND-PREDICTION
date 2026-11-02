"""Train on causal OOF model-feedback contexts to address exposure bias."""
import gc,json,time
import numpy as np,pandas as pd
from sklearn.base import clone
from .core import *
from .protocol import *
from .research import Research
from .alternatives import register

def fit_exposure(r,base_record,weight):
    cfg=FeatureConfig(**base_record['features'])
    oof_start=(r.cut-pd.Timedelta(days=14)).normalize()
    early=Research();early.cut=oof_start-pd.Timedelta('2h');early.end=int(early.times.searchsorted(early.cut))
    early.scales=np.maximum(1,early.y[:early.end].mean(axis=0))
    base=early.train(base_record['architecture'],cfg,base_record['transform'],'recursive',base_record['requested_params'])
    oo=pd.date_range(oof_start,r.cut-pd.Timedelta(minutes=15*47),freq='13h')
    positions=r.times.get_indexer(oo)
    b=len(positions);h=np.zeros((b,HISTORY+48,30),dtype=np.float32);h[:,:HISTORY]=histories(r.y,positions)
    xx=[];yy=[]
    for k in range(48):
        # Features used by the OOF base use scales fitted before the OOF origins.
        times=r.times[positions]+pd.Timedelta(minutes=15*k)
        bx=make_features(h[:,:HISTORY+k],times,list(range(30)),cfg,early.scales,r.neighbors)
        pred=base.predict(bx).reshape(b,30)
        # Final learner's scale is an allowed full-training statistic.
        x=make_features(h[:,:HISTORY+k],times,list(range(30)),cfg,r.scales,r.neighbors,forecast_age=k)
        xx.append(x);yy.append(r.y[positions+k].reshape(-1));h[:,HISTORY+k]=pred
    synthetic=pd.concat(xx,ignore_index=True);truth=np.concatenate(yy)
    realx,realy,_=r.rows(cfg,'recursive')
    x=pd.concat([realx,synthetic],ignore_index=True);y=np.concatenate([realy,truth])
    target=y.copy()
    if base_record['transform']=='residual':target-=x.seasonal_anchor.to_numpy()
    elif base_record['transform']=='log1p':target=np.log1p(target)
    elif base_record['transform']=='normalized':target/=np.tile(r.scales,len(x)//30)
    model=clone(base.model)
    # Use frozen full-stage fitted iteration count, not the shorter OOF count.
    if 'n_estimators' in base_record['fitted_params']:model.set_params(n_estimators=base_record['fitted_params']['n_estimators'])
    weights=np.r_[np.ones(len(realx)),np.full(len(synthetic),weight)]
    model.fit(x,target,sample_weight=weights)
    bundle=ForecastModel(model,cfg,r.scales,'recursive',base_record['transform'],r.neighbors,exposure_aware=True)
    meta=dict(base=base_record['id'],synthetic_weight=weight,oof_fit_end=str(early.cut),
        oof_origins=[str(t) for t in oo],real_rows=len(realx),synthetic_rows=len(synthetic),
        causal_note='OOF base is fitted strictly before every synthetic origin. Truth enters only labels; predicted feedback generates all synthetic lag/rolling inputs.')
    return bundle,meta

def main():
    r=Research()
    base=min([a for a in r.records if a['eligible'] and a['architecture']=='lightgbm' and a['strategy']=='recursive'],key=lambda a:a['validation']['WAPE'])
    for weight in [.3,1.]:
        started=time.perf_counter()
        bundle,meta=fit_exposure(r,base,weight)
        register(r,f'exposure_weight_{weight}',bundle,'exposure_lightgbm','recursive',meta,training_seconds=time.perf_counter()-started)

if __name__=='__main__':main()
