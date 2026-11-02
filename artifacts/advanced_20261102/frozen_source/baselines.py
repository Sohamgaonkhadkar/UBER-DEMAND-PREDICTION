"""Exact phase-one feature definition, batched over independent origins."""
from dataclasses import dataclass
import time,joblib,numpy as np,pandas as pd
from sklearn.base import clone
from ml.features import COLUMNS,WINDOWS,LAGS,time_features,training_features
from .core import histories,HISTORY
from .protocol import *

@dataclass
class V2Forecast:
    model:object
    def forecast(self,y,times,positions,horizon=48):
        h=histories(y,positions).astype(float);b,_,r=h.shape
        ewma=pd.DataFrame(y[:max(positions)]).ewm(alpha=.4,adjust=False).mean().to_numpy()[positions-1]
        values=[]
        for k in range(horizon):
            targets=times[positions]+pd.Timedelta(minutes=15*k)
            x=time_features(targets.repeat(r),np.tile(np.arange(r),b))
            for lag in LAGS:x[f'lag_{lag}']=h[:,-lag,:].reshape(-1)
            for w in WINDOWS:
                x[f'rolling_mean_{w}']=h[:,-w:,:].mean(axis=1).reshape(-1)
                x[f'rolling_std_{w}']=h[:,-w:,:].std(axis=1,ddof=1).reshape(-1)
            x['ewma']=ewma.reshape(-1)
            p=np.maximum(0,np.asarray(self.model.predict(x[COLUMNS].astype(float))).reshape(b,r))
            values.append(p);h=np.concatenate([h,p[:,None,:]],axis=1);ewma=.4*p+.6*ewma
        return np.stack(values,axis=1)

def fit_baseline(name,panel,cut):
    p,x=training_features(panel)
    times=pd.DatetimeIndex(p.datetime.unique()).sort_values()
    mask=(p.datetime>=times[HISTORY])&(p.datetime<cut)
    model=clone(joblib.load(ROOT/f'artifacts/candidates/{name}_v2.joblib')['model'])
    model.fit(x[mask],p.loc[mask,'total_pickups'])
    return V2Forecast(model)

def main():
    from .research import Research
    from .alternatives import register
    r=Research()
    for name in ['lightgbm','xgboost']:
        started=time.perf_counter()
        bundle=fit_baseline(name,r.panel,r.cut)
        register(r,'comparable_v2_'+name,bundle,'v2_'+name,'recursive',dict(phase_one_hyperparameters=True,features=COLUMNS,warmup=HISTORY,training_end=str(r.cut)),training_seconds=time.perf_counter()-started)

if __name__=='__main__':
    from ml.advanced.baselines import main as entry
    entry()
