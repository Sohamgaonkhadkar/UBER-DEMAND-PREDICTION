"""Portable causal models; no fitting or public-source access at inference."""
from dataclasses import dataclass
import holidays,numpy as np,pandas as pd
from scipy.stats import nbinom
from ml.advanced.core import HISTORY,histories,make_features,seasonal_indices
from .protocol import TZ

def holiday_offsets(times):
    times=pd.DatetimeIndex(times).tz_convert(TZ)
    dates=times.tz_localize(None).normalize();hs=holidays.US(years=range(times.year.min()-1,times.year.max()+2))
    hd=pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in hs))
    gap=(dates.to_numpy()[:,None]-hd.to_numpy()[None,:])/np.timedelta64(1,'D')
    before=np.where(gap<0,-gap,999).min(axis=1);after=np.where(gap>=0,gap,999).min(axis=1)
    event=np.any(gap==0,axis=1);prev=np.any(gap==1,axis=1);nxt=np.any(gap==-1,axis=1)
    return event,prev,nxt,np.minimum(before,14),np.minimum(after,14)

def extra_calendar(x,targets,r):
    x=x.copy();event,prev,nxt,to,since=holiday_offsets(targets)
    for key,value in [('holiday_before',nxt),('holiday_after',prev),('days_to_holiday',to),('days_since_holiday',since),('month_sin',np.sin(2*np.pi*np.asarray(targets.month)/12)),('month_cos',np.cos(2*np.pi*np.asarray(targets.month)/12))]:
        x[key]=np.repeat(value,r).astype('float32')
    return x

@dataclass
class HolidayProfile:
    ordinary:object
    ratios:object
    support:object
    def forecast(self,y,times,positions,horizon=48):
        p=self.ordinary.forecast(y,times,positions,horizon)
        for k in range(horizon):
            t=pd.DatetimeIndex(times[positions])+pd.Timedelta(minutes=15*k);event,prev,nxt,_,_=holiday_offsets(t)
            for j,mask in enumerate([nxt,event,prev]):
                if mask.any():p[mask,k]*=self.ratios[j,np.asarray(t.hour)[mask]]
        return p

@dataclass
class SparseNB:
    regions:object
    mean:object
    std:object
    beta:object
    alpha:float
    @staticmethod
    def design(x,regions):
        names=['lag_1','lag_4','mean_4','mean_24','clock_day_7','profile_7']
        raw=np.column_stack([np.log1p(np.maximum(0,x[n].to_numpy())) for n in names]+[x[n].to_numpy() for n in ['tod_sin','tod_cos','dow_sin','dow_cos','public_holiday','holiday_adjacent']])
        one=np.column_stack([x.region.to_numpy()==r for r in regions])
        return raw,one
    def predict(self,x):
        raw,one=self.design(x,self.regions);z=np.column_stack([one,np.clip((raw-self.mean)/self.std,-6,6)])
        mu=np.exp(np.clip(z@self.beta,-10,10));n=1/self.alpha
        return nbinom.ppf(.5,n,n/(n+mu))

@dataclass
class FineForecast:
    base:object
    extra:bool=False
    profile:object=None
    holiday_safety:bool=False
    specialist:object=None
    def forecast(self,y,times,positions,horizon=48):
        positions=np.asarray(positions,dtype=int);h=histories(y,positions);b,_,r=h.shape
        h=np.concatenate([h,np.zeros((b,horizon,r),dtype=np.float32)],axis=1);out=[]
        seasonal=self.profile.forecast(y,times,positions,horizon) if self.profile is not None else None
        for k in range(horizon):
            targets=pd.DatetimeIndex(times[positions])+pd.Timedelta(minutes=15*k)
            x=make_features(h[:,:HISTORY+k],targets,list(range(r)),self.base.config,self.base.scales,self.base.neighbors)
            if self.extra:x=extra_calendar(x,targets,r)
            p=self.base.predict(x).reshape(b,r)
            if self.specialist is not None:
                mask=np.isin(x.region.to_numpy(),self.specialist.regions)
                flat=p.reshape(-1);flat[mask]=self.specialist.predict(x.loc[mask]);p=flat.reshape(b,r)
            if seasonal is not None:
                raw=p.copy();anchor=.3*k/47;w=.6*(1-np.exp(-k/12))
                p=(1-anchor)*p+anchor*x.seasonal_anchor.to_numpy().reshape(b,r)
                p=(1-w)*p+w*seasonal[:,k]
                if self.holiday_safety:
                    event,prev,nxt,_,_=holiday_offsets(targets);unsafe=event|prev|nxt;p[unsafe]=raw[unsafe]
            p=np.maximum(0,p)
            if not np.isfinite(p).all():raise ValueError('Invalid fine path')
            h[:,HISTORY+k]=p;out.append(p)
        return np.stack(out,axis=1)

def coarse_features(h,targets,factor):
    b,length,r=h.shape;targets=pd.DatetimeIndex(targets).tz_convert(TZ);v={}
    def add(name,a):
        a=np.asarray(a,dtype=np.float32)
        if a.shape==(b,):a=np.broadcast_to(a[:,None],(b,r))
        v[name]=np.broadcast_to(a,(b,r)).reshape(-1)
    add('region',np.broadcast_to(np.arange(r),(b,r)))
    slot=np.asarray(targets.hour*4+targets.minute//15);dow=np.asarray(targets.day_of_week)
    for name,a in [('slot',slot),('dow',dow),('phase',slot%factor),('sin',np.sin(2*np.pi*slot/96)),('cos',np.cos(2*np.pi*slot/96)),('dow_sin',np.sin(2*np.pi*dow/7)),('dow_cos',np.cos(2*np.pi*dow/7))]:add(name,a)
    for lag in sorted(set([1,2,3,4,8,24//factor,96//factor,192//factor,672//factor,1344//factor])):add('lag_'+str(lag),h[:,-lag])
    for w in [4//factor,24//factor,96//factor,672//factor]:
        add('mean_'+str(w),h[:,-w:].mean(axis=1));add('std_'+str(w),h[:,-w:].std(axis=1))
    # Aggregates end at the target start; same local-clock seasonal blocks.
    for days in [1,7,14]:
        desired=(targets.tz_localize(None)-pd.Timedelta(days=days)).tz_localize(TZ,ambiguous='NaT',nonexistent='NaT')
        delta=np.asarray((targets-desired)/pd.Timedelta(minutes=15*factor));delta=np.where(np.isfinite(delta),delta,days*96/factor).astype(int)
        idx=length-delta
        if np.any(idx<0) or np.any(idx>=length):raise ValueError('Coarse seasonal context outside history')
        add('clock_'+str(days),h[np.arange(b),idx])
    event,prev,nxt,to,since=holiday_offsets(targets)
    for name,a in [('holiday',event),('before',nxt),('after',prev),('to',to),('since',since)]:add(name,a)
    return pd.DataFrame(v,dtype='float32')

@dataclass
class CoarseForecast:
    model:object
    factor:int
    shares:object
    city:bool=False
    def totals(self,y,times,positions):
        positions=np.asarray(positions,dtype=int);h=histories(y,positions)
        if self.city:h=h.sum(axis=2,keepdims=True)
        b,_,r=h.shape;h=h.reshape(b,HISTORY//self.factor,self.factor,r).sum(axis=2)
        values=[]
        for k in range(48//self.factor):
            targets=pd.DatetimeIndex(times[positions])+pd.Timedelta(minutes=15*self.factor*k)
            p=np.maximum(0,self.model.predict(coarse_features(h,targets,self.factor)).reshape(b,r))
            h=np.concatenate([h,p[:,None].astype(np.float32)],axis=1);values.append(p)
        return np.stack(values,axis=1)
    def forecast(self,y,times,positions,horizon=48):
        totals=self.totals(y,times,positions);out=[]
        for k in range(48//self.factor):
            t=pd.DatetimeIndex(times[positions])+pd.Timedelta(minutes=15*self.factor*k)
            shares=self.shares[np.asarray(t.day_of_week),np.asarray(t.hour*4+t.minute//15)]
            out.append(np.moveaxis(shares,2,1)*totals[:,k,None,:])
        return np.concatenate(out,axis=1)[:,:horizon]

def normalized(v,axis):
    v=np.asarray(v,dtype=float)
    den=v.sum(axis=axis,keepdims=True)
    return np.divide(v,den,out=np.full_like(v,1/v.shape[axis]),where=den>0)

@dataclass
class ReconciledForecast:
    fine:object
    half:object
    hourly:object
    city:object=None
    def generate(self,y,times,positions):
        f=self.fine.forecast(y,times,positions,48).reshape(-1,12,2,2,30)
        hp=self.half.forecast(y,times,positions,48).reshape(-1,12,2,2,30)
        hour=self.hourly.totals(y,times,positions)
        target_hour=(f.sum(axis=(2,3))+hour)/2
        target_half=(f.sum(axis=3)+hp.sum(axis=3))/2
        target_half=normalized(target_half,axis=2)*target_hour[:,:,None,:]
        distribution=(normalized(f,axis=3)+normalized(hp,axis=3))/2
        p=distribution*target_half[:,:,:,None,:]
        if self.city is not None:
            city_total=self.city.totals(y,times,positions)[:,:,0]
            desired=(target_hour.sum(axis=2)+city_total)/2
            denominator=p.sum(axis=(2,3,4))
            ratio=np.divide(desired,denominator,out=np.ones_like(desired),where=denominator>0)
            p*=ratio[:,:,None,None,None]
            p=np.where((denominator==0)[:,:,None,None,None],desired[:,:,None,None,None]/120,p)
        p=p.reshape(-1,48,30)
        return p,dict(hourly=p.reshape(-1,12,4,30).sum(axis=2),half_hourly=p.reshape(-1,24,2,30).sum(axis=2),city_hourly=p.reshape(-1,12,4,30).sum(axis=(2,3)))
    def forecast(self,y,times,positions,horizon=48):return self.generate(y,times,positions)[0][:,:horizon]
