"""Focused follow-up after the training-only calendar profile won validation."""
from dataclasses import dataclass
import copy
import joblib,numpy as np,pandas as pd
from .core import *
from .protocol import *
from .research import Research
from .alternatives import register,CompositeForecast

@dataclass
class CalendarAnchoredRecursive:
    base:object
    calendar:object
    weight:float=.6
    decay:float=12.
    def forecast(self,y,times,positions,horizon=48):
        positions=np.asarray(positions,dtype=int);h=histories(y,positions)
        b,_,r=h.shape;h=np.concatenate([h,np.zeros((b,horizon,r),dtype=np.float32)],axis=1)
        calendar=self.calendar.forecast(y,times,positions,horizon)
        result=[]
        for k in range(horizon):
            targets=pd.DatetimeIndex(times[positions])+pd.Timedelta(minutes=15*k)
            x=make_features(h[:,:HISTORY+k],targets,list(range(r)),self.base.config,self.base.scales,self.base.neighbors,
                forecast_age=k if self.base.exposure_aware else None)
            p=self.base.predict(x).reshape(b,r)
            if self.base.anchor_weight:
                w=self.base.anchor_weight*k/47;p=(1-w)*p+w*x.seasonal_anchor.to_numpy().reshape(b,r)
            if self.base.bias is not None:p+=self.base.bias[min(k//12,3)]
            w=self.weight*(1-np.exp(-k/self.decay))
            p=np.maximum(0,(1-w)*p+w*calendar[:,k])
            if not np.isfinite(p).all():raise RuntimeError('Invalid calendar-anchored recursion')
            result.append(p);h[:,HISTORY+k]=p
        return np.stack(result,axis=1)

def main():
    r=Research();available=[a for a in r.records if a.get('eligible') and a.get('artifact')]
    rec=min([a for a in available if a['strategy']=='recursive'],key=lambda a:a['validation']['WAPE'])
    direct=min([a for a in available if a['strategy']=='direct'],key=lambda a:a['validation']['WAPE'])
    cal=min([a for a in available if a['architecture']=='calendar_profile'],key=lambda a:a['validation']['WAPE'])
    rb=joblib.load(ROOT/rec['artifact']);db=joblib.load(ROOT/direct['artifact']);cb=joblib.load(ROOT/cal['artifact'])
    for source,bundle in [(rec,rb),(direct,db)]:
        for weight in [.25,.5,.75]:
            register(r,f"calendar_blend_{source['strategy']}_{weight}",CompositeForecast(bundle,cb,weight=weight),'ensemble','blend',dict(first=source['id'],second=cal['id'],weight=weight))
    for switch in [4,12]:
        register(r,f'calendar_hybrid_{switch}',CompositeForecast(rb,cb,switch=switch),'hybrid','hybrid',dict(recursive=rec['id'],direct=cal['id'],switch=switch))
    for weight in [.3,.6,.9]:
        bundle=CalendarAnchoredRecursive(rb,cb,weight,12.)
        register(r,f'calendar_feedback_{weight}',bundle,'calendar_anchored_recursive','recursive',dict(base=rec['id'],calendar=cal['id'],weight=weight,decay=12.,causal_note='Blended predictions, never future truth, feed every subsequent recursive step'))

if __name__=='__main__':
    from ml.advanced.calendar_anchoring import main as entry
    entry()
