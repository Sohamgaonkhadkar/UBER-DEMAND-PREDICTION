"""Shared causal batch features and joint/direct forecast execution.

Each batch owns its history. Historical seasonal indices are checked against
the forecast origin, including direct multi-horizon rows and DST transitions.
"""
from dataclasses import dataclass,asdict
import numpy as np,pandas as pd
import holidays
from .protocol import TZ

HISTORY=1536
LAGS=(1,2,3,4,8,12,24,48,96,192,288,672,1344)

@dataclass(frozen=True)
class FeatureConfig:
    seasonal:bool=True
    extended_lags:bool=True
    statistics:bool=False
    interactions:bool=False
    spatial:bool=False
    calendar:bool=True
    weather:bool=False

def seasonal_indices(targets,days,length,age):
    # Same local-clock quarter, rather than 96/672 elapsed slots across DST.
    targets=pd.DatetimeIndex(targets)
    desired=(targets.tz_convert(TZ).tz_localize(None)-pd.Timedelta(days=days)).tz_localize(TZ,ambiguous='NaT',nonexistent='NaT')
    delta=np.asarray((targets-desired)/pd.Timedelta(minutes=15),dtype=float)
    bad=np.isnan(delta);delta=np.where(bad,96*days,delta).astype(int)
    idx=length+np.asarray(age)-delta
    if np.any(idx>=length):raise ValueError('Seasonal feature reaches beyond forecast-origin observations')
    return idx,bad.astype(float)

def make_features(history,targets,regions,config,scales,neighbors=None,age=0,weather_values=None,forecast_age=None):
    h=np.asarray(history,dtype=np.float32)
    if h.ndim!=3 or h.shape[1]<HISTORY or not np.isfinite(h).all():raise ValueError('Incomplete or nonfinite 16-day joint history')
    b,length,r=h.shape
    if list(regions)!=list(range(r)):raise ValueError('Joint region order must be consecutive and explicit')
    targets=pd.DatetimeIndex(targets).tz_convert(TZ)
    if len(targets)!=b:raise ValueError('Timestamp/history batch mismatch')
    scales=np.asarray(scales,dtype=np.float32)
    if len(scales)!=r or np.any(scales<=0):raise ValueError('Invalid training-only regional scale')
    vals={};age=np.broadcast_to(np.asarray(age),b)
    def add(name,v):
        a=np.asarray(v,dtype=np.float32)
        if a.ndim==0:a=np.full((b,r),a)
        elif a.shape==(b,):a=np.broadcast_to(a[:,None],(b,r))
        elif a.shape==(r,):a=np.broadcast_to(a[None,:],(b,r))
        if a.shape!=(b,r):raise ValueError((name,a.shape,b,r))
        vals[name]=a.reshape(-1)
    add('region',np.broadcast_to(np.array(regions)[None,:],(b,r)))
    tod=np.asarray(targets.hour*4+targets.minute//15);dow=np.asarray(targets.day_of_week)
    add('slot_of_day',tod);add('day_of_week',dow);add('weekend',dow>=5)
    add('tod_sin',np.sin(2*np.pi*tod/96));add('tod_cos',np.cos(2*np.pi*tod/96))
    add('dow_sin',np.sin(2*np.pi*dow/7));add('dow_cos',np.cos(2*np.pi*dow/7))
    add('lead',age+1 if forecast_age is None else np.broadcast_to(np.asarray(forecast_age),b)+1)
    lags=LAGS if config.extended_lags else (1,2,3,4,24,96,168,336,672,720)
    for k in lags:add(f'lag_{k}',h[:,-k,:])
    for k in [4,24,96,672]:
        add(f'mean_{k}',h[:,-k:,:].mean(axis=1));add(f'std_{k}',h[:,-k:,:].std(axis=1,ddof=1))
    # Region-local exponentially weighted mean; weights are strictly historical.
    weights=(.6**np.arange(32))[::-1];weights=weights/weights.sum()
    add('ewma',np.einsum('blr,l->br',h[:,-32:,:],weights))
    if config.seasonal:
        profiles=[]
        for days in [1,7,14]:
            idx,bad=seasonal_indices(targets,days,length,age)
            if np.any(idx-4<0):raise ValueError('Insufficient seasonal history')
            a=h[np.arange(b),idx,:]
            smooth=np.mean([h[np.arange(b),idx+offset,:] for offset in [-4,-2,0,2,4]],axis=0)
            add(f'clock_day_{days}',a);add(f'profile_{days}',smooth)
            profiles.append(smooth)
            if days==1:add('clock_fallback',bad)
        anchor=.25*profiles[0]+.5*profiles[1]+.25*profiles[2]
        add('seasonal_anchor',anchor)
        add('week_profile_mean',(profiles[1]+profiles[2])/2)
        add('recent_week_ratio',(h[:,-8:,:].mean(axis=1)+10)/(h[:,-680:-672,:].mean(axis=1)+10))
        add('recent_day_ratio',(h[:,-8:,:].mean(axis=1)+10)/(h[:,-104:-96,:].mean(axis=1)+10))
    if config.statistics:
        for k in [12,96]:
            add(f'median_{k}',np.median(h[:,-k:,:],axis=1));add(f'min_{k}',h[:,-k:,:].min(axis=1));add(f'max_{k}',h[:,-k:,:].max(axis=1))
        add('growth_4',(h[:,-4:,:].mean(axis=1)+5)/(h[:,-8:-4,:].mean(axis=1)+5))
        add('volatility_24',h[:,-24:,:].std(axis=1)/(h[:,-24:,:].mean(axis=1)+5))
    if config.interactions:
        add('region_scale',np.broadcast_to(scales,(b,r)))
        for k in [1,4,96,672]:add(f'relative_lag_{k}',h[:,-k,:]/scales)
        add('relative_momentum',(h[:,-1,:]-h[:,-4,:])/scales)
    if config.spatial:
        if neighbors is None:raise ValueError('Fixed neighbor map required')
        network=h[:,-1,:].sum(axis=1)
        add('network_lag_1',network);add('network_share',h[:,-1,:]/(network[:,None]+1))
        near=np.array(neighbors)
        add('neighbor_lag_1',h[:,-1,:][:,near].mean(axis=2))
        add('neighbor_momentum',(h[:,-1,:][:,near]-h[:,-4,:][:,near]).mean(axis=2))
    if config.calendar:
        hol=holidays.US(years=sorted(set(targets.year)))
        dates=targets.date
        add('public_holiday',[d in hol for d in dates])
        add('holiday_adjacent',[(d+pd.Timedelta(days=1)).date() in hol or (d-pd.Timedelta(days=1)).date() in hol for d in targets.normalize()])
        add('peak_hour',np.isin(targets.hour,[7,8,9,17,18,19]));add('night',targets.hour<6)
    if config.weather:
        if weather_values is None:raise ValueError('Lagged weather required, never silently zero-filled')
        for j,name in enumerate(['temperature','apparent_temperature','precipitation','snowfall','wind','humidity']):add('observed_weather_'+name,weather_values[:,j])
    frame=pd.DataFrame(vals,dtype='float32')
    if not np.isfinite(frame.to_numpy()).all():raise ValueError('Invalid engineered features')
    return frame

def histories(y,positions):
    positions=np.asarray(positions,dtype=int)
    if np.any(positions<HISTORY):raise ValueError('Insufficient history')
    return y[positions[:,None]-HISTORY+np.arange(HISTORY)[None,:]]

def neighbors_from_centers(geography):
    centers=np.array(geography['centers'])
    d=((centers[:,None,:]-centers[None,:,:])**2).sum(axis=2)
    return np.argsort(d,axis=1)[:,1:4].tolist()

@dataclass
class ForecastModel:
    model:object
    config:FeatureConfig
    scales:np.ndarray
    strategy:str='recursive'
    transform:str='raw'
    neighbors:object=None
    anchor_weight:float=0.
    region_models:object=None
    bias:object=None
    weather:object=None
    exposure_aware:bool=False

    @property
    def feature_name_(self):
        names=getattr(self.model,'feature_name_',None)
        if names is None:names=getattr(self.model,'feature_names_in_',getattr(self.model,'feature_names_',None))
        if names is None:raise AttributeError('Feature names unavailable')
        return list(names)

    @property
    def feature_importances_(self):
        if hasattr(self.model,'booster_'):return self.model.booster_.feature_importance(importance_type='gain')
        return self.model.feature_importances_

    def reconstruct(self,p,x):
        p=np.asarray(p).reshape(-1)
        if self.transform=='log1p':p=np.expm1(p)
        elif self.transform=='residual':p=p+x.seasonal_anchor.to_numpy()
        elif self.transform=='normalized':p=p*np.tile(self.scales,len(x)//len(self.scales))
        return np.maximum(0,p)

    def predict(self,x):
        expected=getattr(self,'feature_name_',None)
        if expected is not None and list(x.columns)!=list(expected):
            raise ValueError('Feature schema differs from fitted model; missing/reordered columns are not filled')
        if self.region_models is None:p=self.model.predict(x)
        else:
            p=np.zeros(len(x))
            for r,model in enumerate(self.region_models):
                mask=x.region.to_numpy()==r;p[mask]=model.predict(x.loc[mask])
        return self.reconstruct(p,x)

    def forecast(self,y,times,positions,horizon=48):
        y=np.asarray(y,dtype=np.float32);positions=np.asarray(positions,dtype=int)
        base=histories(y,positions);b,_,r=base.shape
        h=np.zeros((b,HISTORY+horizon,r),dtype=np.float32);h[:,:HISTORY]=base
        output=np.empty((b,horizon,r),dtype=float)
        for k in range(horizon):
            direct=self.strategy=='direct'
            length=HISTORY if direct else HISTORY+k
            targets=pd.DatetimeIndex(times[positions])+pd.Timedelta(minutes=15*k)
            w=None
            if self.config.weather:
                if self.weather is None:raise ValueError('Historical weather context unavailable')
                known=pd.DatetimeIndex(times[positions]).tz_convert(TZ).tz_localize(None).floor('h')-pd.Timedelta(hours=3)
                w=self.weather.reindex(known).to_numpy(dtype=float)
            x=make_features(h[:,:length],targets,list(range(r)),self.config,self.scales,self.neighbors,age=k if direct else 0,weather_values=w,forecast_age=k if self.exposure_aware else None)
            p=self.predict(x).reshape(b,r)
            if self.anchor_weight:
                p=(1-self.anchor_weight*(k/47))*p+self.anchor_weight*(k/47)*x.seasonal_anchor.to_numpy().reshape(b,r)
            if self.bias is not None:
                p=p+np.asarray(self.bias)[min(k//12,3)]
            p=np.maximum(0,p)
            if not np.isfinite(p).all():raise RuntimeError('Nonfinite forecast')
            output[:,k]=p;h[:,HISTORY+k]=p
        return output

def training_rows(y,times,positions,config,scales,neighbors=None,leads=(1,),weather=None):
    allx=[];ally=[];targettimes=[]
    for lead in leads:
        for start in range(0,len(positions),96):
            pos=np.asarray(positions[start:start+96]);h=histories(y,pos)
            w=None
            if config.weather:
                if weather is None:raise ValueError('Historical weather context unavailable')
                known=pd.DatetimeIndex(times[pos]).tz_convert(TZ).tz_localize(None).floor('h')-pd.Timedelta(hours=3)
                w=weather.reindex(known).to_numpy(dtype=float)
            x=make_features(h,times[pos+lead-1],list(range(y.shape[1])),config,scales,neighbors,age=lead-1,weather_values=w)
            allx.append(x);ally.append(y[pos+lead-1].reshape(-1));targettimes.extend(np.repeat(np.asarray(times[pos+lead-1]),y.shape[1]))
    return pd.concat(allx,ignore_index=True),np.concatenate(ally),np.asarray(targettimes)
