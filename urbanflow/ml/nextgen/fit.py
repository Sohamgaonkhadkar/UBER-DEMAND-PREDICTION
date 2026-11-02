"""Fixed-specification fits; no search and no fresh-test reads."""
import copy,json,time
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
from scipy.optimize import minimize
from ml.advanced.core import HISTORY,training_rows
from .protocol import *
from .models import *

def as_controlled(old_model,y,times,cut):
    end=int(times.searchsorted(cut));scales=np.maximum(1,y[:end].mean(axis=0));cv=y[:end].std(axis=0)/(scales+5)
    return ControlledForecast(copy.deepcopy(old_model.base),Profile(old_model.calendar.profile.copy(),old_model.calendar.decay),scales,cv,old_model.weight,old_model.decay)

def fit_base(y,times,cut,template,tag):
    if (OUT/'final_test_results.json').exists():raise RuntimeError('Fresh test is spent; do not retrain within this protocol')
    end=int(times.searchsorted(cut));scales=np.maximum(1,y[:end].mean(axis=0));cv=y[:end].std(axis=0)/(scales+5)
    cache=OUT/f'features_{tag}.parquet'
    if cache.exists():
        data=pd.read_parquet(cache);target=data.pop('__target').to_numpy();x=data
    else:
        x,target,_=training_rows(y,times,np.arange(HISTORY,end),template.base.config,scales,template.base.neighbors)
        data=x.copy();data['__target']=target;data.to_parquet(cache,index=False);del data
    model=clone(template.base.model);model.set_params(n_jobs=4)
    if type(model).__module__.startswith('xgboost'):model.set_params(early_stopping_rounds=None)
    model.fit(x,target)
    base=copy.deepcopy(template.base);base.model=model;base.scales=scales;base.bias=None
    return ControlledForecast(base,profile_from_data(y,times,end),scales,cv)

def fit_calibrator(features,actual,trace,scales,kind):
    x=features.reshape(-1,features.shape[-1]).astype(float)
    mean=x.mean(axis=0);std=np.maximum(x.std(axis=0),.05);z=np.clip((x-mean)/std,-5,5)
    y=actual.reshape(-1).astype(float);pred=trace['prediction'].reshape(-1).astype(float)
    s=np.broadcast_to(np.asarray(scales),(actual.shape)).reshape(-1)
    if kind=='gate':
        m=trace['model_component'].reshape(-1).astype(float);c=trace['calendar'].reshape(-1).astype(float)
        prior=trace['weights'].reshape(-1).astype(float);first=prior==0;logit=np.log(np.clip(prior,1e-5,1-1e-5)/(1-np.clip(prior,1e-5,1-1e-5)))
        def objective(beta):
            raw=logit+z@beta;score=np.clip(raw,-8,8);w=1/(1+np.exp(-score));w[first]=0
            error=m+w*(c-m)-y;root=np.sqrt(error**2+1)
            derivative=w*(1-w)*(c-m);derivative[first]=0
            derivative[(raw<=-8)|(raw>=8)]=0
            return float(root.mean()+.2*(beta@beta)),z.T@((error/root)*derivative)/len(y)+.4*beta
    else:
        def objective(beta):
            error=pred+s*(z@beta)-y;root=np.sqrt(error**2+1)
            return float(root.mean()+.2*(beta@beta)),z.T@((error/root)*s)/len(y)+.4*beta
    result=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-9})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():raise RuntimeError('Calibration optimizer failed')
    support=np.quantile(np.abs(actual-trace['prediction']),.95,axis=(0,1)) if kind!='gate' else None
    model=Calibrator(mean,std,result.x,kind,support)
    meta=dict(kind=kind,features=FEATURES,coefficients=result.x.tolist(),optimizer_success=bool(result.success),message=str(result.message),
        objective=float(result.fun),regularization=.2,iterations=int(result.nit),loss='Original-count smooth MAE sqrt(error^2+1), plus L2 penalty; no target replacement')
    return model,meta
