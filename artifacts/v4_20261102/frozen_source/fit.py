"""Fixed-capacity fits on chronological labels. No final-test imports."""
import copy,json,gc
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
from scipy.optimize import minimize
from lightgbm import LGBMRegressor
from ml.advanced.core import HISTORY,training_rows,histories
from ml.nextgen.models import profile_from_data,Profile
from .models import *
from .protocol import *

def combined():
    old=pd.read_parquet(OUT/'history_2015.parquet');data=pd.concat([old,panel()],ignore_index=True)
    if data.duplicated(['datetime','region']).any():raise ValueError('Duplicate observation keys')
    wide=data.pivot(index='datetime',columns='region',values='total_pickups').sort_index()
    if not np.isfinite(wide.to_numpy()).all():raise ValueError('Missing regional context')
    return wide

def templates():
    records=json.loads((OLD/'leaderboard.json').read_text());out={}
    for name in ['xgboost','lightgbm']:
        best=min([r for r in records if r['eligible'] and r['architecture']==name and r['strategy']=='recursive'],key=lambda r:r['validation']['WAPE'])
        out[name]=joblib.load(ROOT/best['artifact'])
    return out

def valid_training_positions(times):
    gap=pd.Series(pd.DatetimeIndex(times)).diff().ne(pd.Timedelta('15min')).to_numpy()
    index=np.arange(len(times));last_start=np.maximum.accumulate(np.where(gap,index,0))
    return index[index-last_start>=HISTORY]

def rows(wide,template):
    dest=OUT/('fine_feature_cache_'+wide.index[0].strftime('%Y%m%d')+'.parquet')
    if dest.exists():
        frame=pd.read_parquet(dest);y=frame.pop('__target').to_numpy();ts=pd.DatetimeIndex(pd.to_datetime(frame.pop('__timestamp'),utc=True)).tz_convert(TZ)
        return frame,y,ts
    if template.config.interactions or template.config.spatial:raise ValueError('Cache requires scale-independent baseline features')
    values=wide.to_numpy(dtype=np.float32);pos=valid_training_positions(wide.index)
    jan_cache=OUT/'fine_feature_cache_20160101.parquet'
    if wide.index[0].year<2016 and jan_cache.exists():
        # All permitted 2016 label histories begin Jan17, so they have exactly
        # the same preceding observations as the already-verified Jan-only cache.
        jan=pd.read_parquet(jan_cache);jy=jan.pop('__target').to_numpy();jt=pd.DatetimeIndex(pd.to_datetime(jan.pop('__timestamp'),utc=True)).tz_convert(TZ)
        older=pos[wide.index[pos]<pd.Timestamp('2016-01-01',tz=TZ)]
        hx,hy,ht=training_rows(values,wide.index,older,template.config,np.ones(30),template.neighbors)
        x=pd.concat([hx,jan],ignore_index=True);y=np.concatenate([hy,jy]);ts=pd.DatetimeIndex(ht).append(jt)
    else:x,y,ts=training_rows(values,wide.index,pos,template.config,np.ones(30),template.neighbors)
    ts=pd.DatetimeIndex(ts);frame=x.copy();frame['__target']=y;frame['__timestamp']=ts;frame.to_parquet(dest,index=False);del frame;gc.collect()
    return x,y,ts

def fit_fine(template,x,y,ts,cut,mode='jan',extra=False):
    first=pd.Timestamp('2016-01-17',tz=TZ) if mode=='jan' else ts.min()
    if mode=='rolling60':first=max(first,cut-pd.Timedelta('60d'))
    mask=np.asarray((ts>=first)&(ts<cut));xx=x.loc[mask].copy();yy=y[mask]
    if extra:
        unique=pd.DatetimeIndex(ts[mask][::30]);xx=extra_calendar(xx,unique,30)
    model=clone(template.model).set_params(n_jobs=4)
    if type(model).__module__.startswith('xgboost'):model.set_params(early_stopping_rounds=None)
    weights=None
    if mode=='decay90':weights=np.exp2(-np.asarray((cut-ts[mask])/pd.Timedelta('1d'))/90)
    model.fit(xx,yy,sample_weight=weights)
    base=copy.deepcopy(template);base.model=model;base.anchor_weight=0;base.bias=None
    result=FineForecast(base,extra=extra)
    label_times=pd.DatetimeIndex(ts[mask][::30]);event=holiday_offsets(label_times)[0]
    return result,dict(labels=int(mask.sum()),start=str(ts[mask].min()),end=str(ts[mask].max()),mode=mode,extra_calendar=extra,holiday_training_dates=[str(d) for d in sorted(set(label_times[event].date))],params=model.get_params())

def fit_holiday_profile(wide,cut,ordinary):
    historical=wide[wide.index<cut];t=historical.index;values=historical.to_numpy(dtype=float)
    event,prev,nxt,_,_=holiday_offsets(t);profile=copy.deepcopy(ordinary);profile.adaptive=True
    # Ordinary 2016 profile excludes known holidays and adjacent dates.
    normal=(t>=pd.Timestamp('2016-01-17',tz=TZ))&~(event|prev|nxt)
    for dow in range(7):
        for slot in range(96):
            use=normal&(t.day_of_week==dow)&(t.hour*4+t.minute//15==slot)
            if use.any():profile.profile[dow,slot]=np.median(values[use],axis=0)
    profile.profile=(np.roll(profile.profile,1,axis=1)+2*profile.profile+np.roll(profile.profile,-1,axis=1))/4
    ratios=np.ones((3,24,30));support=np.zeros((3,24),int);hs=holidays.US(years=sorted(set(t.year)))
    for j,offset in enumerate([-1,0,1]):
        buckets=[[] for _ in range(24)]
        for holiday in sorted(hs):
            date=pd.Timestamp(holiday)+pd.Timedelta(days=offset)
            for hour in range(24):
                target=date+pd.Timedelta(hours=hour);times=pd.date_range(target,periods=4,freq='15min').tz_localize(TZ,ambiguous='NaT',nonexistent='NaT')
                if times.isna().any() or times[-1]>=cut:continue
                observed=historical.reindex(times)
                if observed.isna().any().any():continue
                reference=[]
                for shift in [-14,-7,7,14]:
                    other=(times.tz_localize(None)+pd.Timedelta(days=shift)).tz_localize(TZ,ambiguous='NaT',nonexistent='NaT')
                    if other.isna().any() or other[-1]>=cut:continue
                    if any(d in hs for d in other.date):continue
                    baseline=historical.reindex(other)
                    if not baseline.isna().any().any():reference.append(baseline.sum(axis=0).to_numpy())
                if len(reference)<2:continue
                expected=np.median(reference,axis=0);rate=(observed.sum(axis=0).to_numpy()+20)/(expected+20)
                buckets[hour].append(np.clip(rate,.25,2.5))
        for hour,samples in enumerate(buckets):
            support[j,hour]=len(samples)
            if len(samples)>=4:
                raw=np.median(samples,axis=0);weight=len(samples)/(len(samples)+8);ratios[j,hour]=(1-weight)+weight*raw
    return HolidayProfile(profile,ratios,support)

def fit_calendar_control(template,x,target,ts,wide,cut,mode):
    from ml.advanced.calendar_anchoring import CalendarAnchoredRecursive
    fine,record=fit_fine(template,x,target,ts,cut,mode)
    positions=valid_training_positions(wide.index);tt=wide.index[positions]
    use=(tt>=pd.Timestamp(record['start']))&(tt<cut);tt=tt[use];values=wide.to_numpy(dtype=np.float32)[positions[use]]
    p=np.empty((7,96,30),dtype=np.float32)
    for dow in range(7):
        for slot in range(96):
            mask=(tt.day_of_week==dow)&(tt.hour*4+tt.minute//15==slot)
            if not mask.any():raise ValueError('Matched calendar lacks weekday/time support')
            p[dow,slot]=np.median(values[mask],axis=0)
    p=(np.roll(p,1,axis=1)+2*p+np.roll(p,-1,axis=1))/4
    fine.base.anchor_weight=template.anchor_weight
    record['profile_rows']='Same valid label timestamps; unweighted median, original weekday smoothing. Case weights affect boosted fit only.'
    return CalendarAnchoredRecursive(fine.base,Profile(p),.6,12.),record

def fit_sparse(x,y,ts,cut):
    mask=(ts>=pd.Timestamp('2016-01-17',tz=TZ))&(ts<cut);means=pd.DataFrame(dict(r=x.loc[mask,'region'].to_numpy(),y=y[mask])).groupby('r').y.mean()
    regions=means[means<=10].index.astype(int).to_numpy();mask=mask&np.isin(x.region.to_numpy(),regions)
    xx=x.loc[mask];yy=y[mask].astype(float);raw,one=SparseNB.design(xx,regions)
    mean=raw.mean(axis=0);std=np.maximum(raw.std(axis=0),.1);z=np.column_stack([one,np.clip((raw-mean)/std,-6,6)])
    beta=np.zeros(z.shape[1]);beta[:len(regions)]=np.log(np.maximum(means.loc[regions].to_numpy(),.1))
    penalty=np.r_[np.zeros(len(regions)),np.full(z.shape[1]-len(regions),.01)]
    def objective(v,alpha):
        eta=np.clip(z@v,-10,10);mu=np.exp(eta)
        if alpha==0:loss=mu-yy*eta;grad=mu-yy
        else:
            n=1/alpha;loss=(yy+n)*np.log(n+mu)-yy*eta;grad=(mu-yy)/(1+alpha*mu)
        return float(loss.mean()+np.sum(penalty*v*v)),z.T@grad/len(yy)+2*penalty*v
    first=minimize(lambda v:objective(v,0),beta,jac=True,method='L-BFGS-B',options={'maxiter':150})
    mu=np.exp(np.clip(z@first.x,-10,10));alpha=float(np.clip(np.sum((yy-mu)**2-yy)/np.sum(mu**2),.01,5))
    fitted=minimize(lambda v:objective(v,alpha),first.x,jac=True,method='L-BFGS-B',options={'maxiter':150})
    if not np.isfinite(fitted.fun):raise ValueError('NB fit invalid')
    return SparseNB(regions,mean,std,fitted.x,alpha),dict(regions=regions.tolist(),alpha=alpha,optimizer_success=bool(fitted.success),iterations=int(fitted.nit),objective='regularized NB2 likelihood; output conditional median for absolute-error scoring')

def fit_coarse(wide,cut,factor,city=False):
    # Match Jan-only baselines' label start; origin-relative blocks cover every
    # quarter phase because factor+1 is coprime to factor.
    times=wide.index;y=wide.to_numpy(dtype=np.float32);end=int(times.searchsorted(cut));start=int(times.searchsorted(pd.Timestamp('2016-01-17',tz=TZ)))
    pos=np.arange(start,end-factor+1,factor+1);pos=pos[np.isin(pos,valid_training_positions(times))];allx=[];targets=[]
    for start_batch in range(0,len(pos),96):
        pp=pos[start_batch:start_batch+96];h=histories(y,pp);future=y[pp[:,None]+np.arange(factor)[None,:]]
        if city:h=h.sum(axis=2,keepdims=True);future=future.sum(axis=2,keepdims=True)
        b,_,r=h.shape;h=h.reshape(b,HISTORY//factor,factor,r).sum(axis=2)
        allx.append(coarse_features(h,times[pp],factor));targets.append(future.sum(axis=1).reshape(-1))
    x=pd.concat(allx,ignore_index=True);target=np.concatenate(targets)
    model=LGBMRegressor(n_estimators=400,learning_rate=.04,num_leaves=31,min_child_samples=40,reg_lambda=3,objective='regression_l1',n_jobs=4,random_state=42,verbosity=-1)
    model.fit(x,target);r=1 if city else 30
    shares=np.full((7,96,r,factor),1/factor,dtype=float);count=np.zeros((7,96),int)
    sums=np.full_like(shares,8/factor)
    # Distribution statistics use observed full blocks, never a partial future hour.
    for p in range(start,end-factor+1):
        v=y[p:p+factor]
        if city:v=v.sum(axis=1,keepdims=True)
        den=v.sum(axis=0);fraction=np.divide(v,den[None,:],out=np.full_like(v,1/factor),where=den[None,:]>0).T
        dow=times[p].day_of_week;slot=times[p].hour*4+times[p].minute//15;sums[dow,slot]+=fraction;count[dow,slot]+=1
    shares=sums/(count[:,:,None,None]+8)
    return CoarseForecast(model,factor,shares,city),dict(labels=len(target),factor=factor,city=city,params=model.get_params(),origin_phase_counts=np.bincount(np.asarray(times[pos].minute)//15,minlength=4).tolist())
