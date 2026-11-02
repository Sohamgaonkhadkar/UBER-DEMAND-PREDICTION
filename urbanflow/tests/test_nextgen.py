import copy
import numpy as np,pandas as pd
from ml.advanced.core import ForecastModel,FeatureConfig,HISTORY
from ml.nextgen.models import Profile,ControlledForecast,Calibrator,profile_from_data

class Increment:
    def predict(self,x):return x.lag_1.to_numpy()+1

def example():
    times=pd.date_range('2016-01-01',periods=6000,freq='15min',tz='America/New_York')
    y=np.arange(len(times),dtype=np.float32)[:,None]+np.arange(30)[None,:]
    base=ForecastModel(Increment(),FeatureConfig(),np.ones(30),anchor_weight=0.)
    return y.astype(np.float32),times,ControlledForecast(base,Profile(np.full((7,96,30),10,dtype=np.float32)),np.ones(30)*2,np.ones(30))

def test_weekly_boundary_uses_adjacent_weekday():
    y,t,_=example();signal=np.asarray(t.day_of_week)*100+np.asarray(t.hour*4+t.minute//15)
    y[:]=signal[:,None]
    old=profile_from_data(y,t,5500,'baseline');new=profile_from_data(y,t,5500,'weekly_boundary_only')
    assert old.profile[0,0,0]==24
    assert new.profile[0,0,0]==174

def test_profiles_never_read_future_labels():
    y,t,_=example();end=5000
    before=profile_from_data(y,t,end,'recent_shrunk_weighted_median').profile
    y[end:]=999999
    after=profile_from_data(y,t,end,'recent_shrunk_weighted_median').profile
    np.testing.assert_equal(before,after)

def test_future_target_poison_and_prefix_consistency():
    y,t,m=example();p=m.forecast(y,t,np.array([2000]),48);q=m.forecast(y,t,np.array([2000]),4)
    np.testing.assert_equal(q,p[:,:4])
    poisoned=y.copy();poisoned[2000:]=99999
    np.testing.assert_equal(p,m.forecast(poisoned,t,np.array([2000]),48))

def test_zero_gate_reproduces_prior_and_feedback_is_real():
    y,t,m=example();p=m.forecast(y,t,np.array([2000]),4)
    m.gate=Calibrator(np.zeros(10),np.ones(10),np.zeros(10))
    q,trace=m.generate(y,t,np.array([2000]),4,trace=True)
    np.testing.assert_allclose(p,q,rtol=1e-7,atol=1e-4)
    assert (trace['weights'][:,0]==0).all()
    assert (trace['weights'][:,1:]>=0).all() and (trace['weights'][:,1:]<=1).all()
    np.testing.assert_allclose(trace['model_component'][0,2],q[0,1].astype(np.float32)+1,rtol=1e-6)

def test_post_path_correction_does_not_compound_in_state():
    y,t,m=example();original=m.forecast(y,t,np.array([2000]),12)
    coef=np.zeros(10);coef[0]=1
    m.correction=Calibrator(np.zeros(10),np.ones(10),coef,'post_path',np.full(30,100.))
    corrected=m.forecast(y,t,np.array([2000]),12)
    expected=np.broadcast_to(2*np.arange(12)[None,:,None]/47,original.shape)
    np.testing.assert_allclose(corrected-original,expected,atol=1e-10)
