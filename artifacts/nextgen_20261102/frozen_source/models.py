"""Inference-only classes: original-scale causal recursion and small calibrators."""
from dataclasses import dataclass
import numpy as np,pandas as pd
from ml.advanced.core import HISTORY,histories,make_features

FEATURES=['lead_fraction','target_sin','target_cos','weekend','night','origin_gain','origin_volatility','component_disagreement','training_region_cv','relative_calendar_level']

@dataclass
class Profile:
    profile:object
    decay:float=12.
    adaptive:bool=True
    def origin_context(self,y,times,positions):
        expected=[]
        for offset in range(-8,0):
            t=pd.DatetimeIndex(times[positions+offset]);expected.append(self.profile[np.asarray(t.day_of_week),np.asarray(t.hour*4+t.minute//15)])
        expected=np.stack(expected,axis=1).sum(axis=1)
        recent=y[positions[:,None]-8+np.arange(8)[None,:]]
        gain=np.clip((recent.sum(axis=1)+100)/(expected+100),.6,1.6)
        vol=recent.std(axis=1)/(recent.mean(axis=1)+5)
        return gain,vol
    def forecast(self,y,times,positions,horizon=48):
        gain,_=self.origin_context(y,times,positions);out=[]
        for k in range(horizon):
            t=pd.DatetimeIndex(times[positions])+pd.Timedelta(minutes=15*k)
            p=self.profile[np.asarray(t.day_of_week),np.asarray(t.hour*4+t.minute//15)]
            if self.adaptive:p=p*(1+(gain-1)*np.exp(-k/self.decay))
            out.append(np.maximum(0,p))
        return np.stack(out,axis=1)

def context_features(targets,k,m,c,gain,vol,scales,region_cv):
    b,r=m.shape;rows=[]
    def bcast(v):return np.broadcast_to(np.asarray(v)[:,None],(b,r))
    quarter=np.asarray(targets.hour*4+targets.minute//15)
    rows=[np.full((b,r),k/47),bcast(np.sin(2*np.pi*quarter/96)),bcast(np.cos(2*np.pi*quarter/96)),
        bcast(targets.day_of_week>=5),bcast(targets.hour<6),np.log(np.maximum(gain,.01)),vol,
        (c-m)/(scales[None,:]+5),np.broadcast_to(region_cv,(b,r)),c/(scales[None,:]+5)]
    return np.stack(rows,axis=-1)

@dataclass
class Calibrator:
    mean:object
    std:object
    coefficients:object
    kind:str='gate'
    support:object=None
    def normalized(self,x):return np.clip((x-self.mean)/self.std,-5,5)
    def predict(self,x):return self.normalized(x)@self.coefficients

@dataclass
class ControlledForecast:
    base:object
    calendar:Profile
    scales:object
    region_cv:object
    weight:float=.6
    decay:float=12.
    gate:object=None
    correction:object=None
    def generate(self,y,times,positions,horizon=48,trace=False):
        y=np.asarray(y,dtype=np.float32);positions=np.asarray(positions,dtype=int)
        h=histories(y,positions);b,_,r=h.shape
        h=np.concatenate([h,np.zeros((b,horizon,r),dtype=np.float32)],axis=1)
        calendar=self.calendar.forecast(y,times,positions,horizon)
        gain,vol=self.calendar.origin_context(y,times,positions)
        out=[];xx=[];mm=[];ww=[]
        for k in range(horizon):
            targets=pd.DatetimeIndex(times[positions])+pd.Timedelta(minutes=15*k)
            x=make_features(h[:,:HISTORY+k],targets,list(range(r)),self.base.config,self.base.scales,self.base.neighbors,
                forecast_age=k if self.base.exposure_aware else None)
            m=self.base.predict(x).reshape(b,r)
            if self.base.anchor_weight:
                w=self.base.anchor_weight*k/47;m=(1-w)*m+w*x.seasonal_anchor.to_numpy().reshape(b,r)
            if self.base.bias is not None:m+=self.base.bias[min(k//12,3)]
            z=context_features(targets,k,m,calendar[:,k],gain,vol,np.asarray(self.scales),np.asarray(self.region_cv))
            w=self.weight*(1-np.exp(-k/self.decay))
            if self.gate is not None and k>0:
                prior=np.log(w/(1-w));score=np.clip(prior+self.gate.predict(z),-8,8);w=1/(1+np.exp(-score))
            p=np.maximum(0,(1-w)*m+w*calendar[:,k])
            if not np.isfinite(p).all():raise RuntimeError('Invalid next-generation recursive forecast')
            out.append(p);h[:,HISTORY+k]=p
            if trace or self.correction is not None:xx.append(z)
            if trace:mm.append(m);ww.append(np.broadcast_to(w,(b,r)))
        output=np.stack(out,axis=1)
        if self.correction is not None:
            features=np.stack(xx,axis=1)
            correction=self.correction.predict(features)*np.asarray(self.scales)[None,None,:]
            correction=np.clip(correction,-np.asarray(self.correction.support)[None,None,:],np.asarray(self.correction.support)[None,None,:])
            output=np.maximum(0,output+correction)
        if trace:return output,dict(features=np.stack(xx,axis=1),model_component=np.stack(mm,axis=1),calendar=calendar,weights=np.stack(ww,axis=1))
        return output
    def forecast(self,y,times,positions,horizon=48):return self.generate(y,times,positions,horizon)

def profile_from_data(y,times,end,recipe='baseline',half_life=28.,prior_weeks=4.):
    values=np.asarray(y[HISTORY:end],dtype=np.float32);t=pd.DatetimeIndex(times[HISTORY:end]);profile=np.zeros((7,96,30),dtype=np.float32)
    for dow in range(7):
        for slot in range(96):
            idx=np.flatnonzero((np.asarray(t.day_of_week)==dow)&(np.asarray(t.hour*4+t.minute//15)==slot))
            if not len(idx):raise ValueError('Calendar profile lacks training coverage')
            data=values[idx];long=np.median(data,axis=0)
            if recipe=='recent_shrunk_weighted_median':
                age=np.asarray((times[end-1]-t[idx])/pd.Timedelta(days=1),dtype=float)
                weights=np.exp2(-age/half_life);order=np.argsort(data,axis=0);sorted_data=np.take_along_axis(data,order,axis=0)
                sorted_weights=np.take_along_axis(np.broadcast_to(weights[:,None],data.shape),order,axis=0)
                cumulative=sorted_weights.cumsum(axis=0);middle=np.argmax(cumulative>=weights.sum()/2,axis=0)
                recent=sorted_data[middle,np.arange(30)];effective=weights.sum()**2/(weights@weights)
                shrink=effective/(effective+prior_weeks);long=(1-shrink)*long+shrink*recent
            profile[dow,slot]=long
    if recipe=='baseline':profile=(np.roll(profile,1,axis=1)+2*profile+np.roll(profile,-1,axis=1))/4
    else:
        flat=profile.reshape(672,30);flat=(np.roll(flat,1,axis=0)+2*flat+np.roll(flat,-1,axis=0))/4;profile=flat.reshape(7,96,30)
    return Profile(profile)
