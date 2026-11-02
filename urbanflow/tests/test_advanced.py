import numpy as np,pandas as pd,pytest
from ml.advanced.core import *

def setup():
    times=pd.date_range('2016-02-15',periods=3500,freq='15min',tz=TZ)
    y=np.arange(len(times),dtype=np.float32)[:,None]+np.arange(30,dtype=np.float32)[None,:]*10
    return y,times

def test_direct_features_have_only_origin_information():
    y,t=setup();pos=np.array([1800,2200]);cfg=FeatureConfig(statistics=True,interactions=True)
    x=make_features(histories(y,pos),t[pos+47],list(range(30)),cfg,np.ones(30),age=47)
    assert (x.lead==48).all()
    assert x.iloc[0].lag_1==y[1799,0]
    assert x.iloc[0].clock_day_7==y[1800+47-672,0]
    assert x.iloc[0].clock_day_7!=y[1800+47,0]
    with pytest.raises(ValueError):seasonal_indices(t[pos],1,HISTORY,100)

def test_batch_training_inference_feature_agreement():
    y,t=setup();positions=np.array([1800,1801]);cfg=FeatureConfig(statistics=True,interactions=True)
    x,target,ts=training_rows(y,t,positions,cfg,np.ones(30),leads=(12,))
    online=make_features(histories(y,positions),t[positions+11],list(range(30)),cfg,np.ones(30),age=11)
    np.testing.assert_allclose(x,online,rtol=1e-6,atol=1e-4)
    np.testing.assert_equal(target,y[positions+11].reshape(-1))

def test_joint_feedback_and_future_poison():
    y,t=setup();pos=np.array([1800]);neighbors=[[(r+1)%30,(r+2)%30,(r+3)%30] for r in range(30)]
    class Model:
        def predict(self,x):return x.neighbor_lag_1.to_numpy()+1
    m=ForecastModel(Model(),FeatureConfig(spatial=True),np.ones(30),neighbors=neighbors)
    before=y.copy();p=m.forecast(y,t,pos,4)
    poisoned=y.copy();poisoned[1800:]=999999
    np.testing.assert_equal(p,m.forecast(poisoned,t,pos,4));np.testing.assert_equal(y,before)
    np.testing.assert_allclose(p[0,1,0],p[0,0,[1,2,3]].mean()+1,rtol=1e-6)

def test_same_clock_seasonal_lag_across_dst():
    target=pd.DatetimeIndex(['2016-03-13T03:00:00-04:00']).tz_convert(TZ)
    idx,bad=seasonal_indices(target,1,HISTORY,0)
    assert idx[0]==HISTORY-92 and bad[0]==0

def test_direct_path_not_recursive_feedback():
    y,t=setup()
    class Model:
        def predict(self,x):return x.lag_1.to_numpy()+x.lead.to_numpy()
    m=ForecastModel(Model(),FeatureConfig(),np.ones(30),strategy='direct')
    p=m.forecast(y,t,np.array([1800]),4)
    np.testing.assert_allclose(p[0,:,0],y[1799,0]+np.arange(1,5))

def test_incomplete_context_and_missing_weather_rejected():
    y,t=setup()
    with pytest.raises(ValueError):histories(y,[1000])
    with pytest.raises(ValueError):make_features(histories(y,[1800]),t[[1800]],list(range(30)),FeatureConfig(weather=True),np.ones(30))

def test_transformed_predictions_return_original_count_scale():
    class Model:pass
    x=pd.DataFrame({'seasonal_anchor':np.arange(30,dtype=float)})
    m=ForecastModel(Model(),FeatureConfig(),np.arange(1,31),transform='normalized')
    np.testing.assert_equal(m.reconstruct(np.ones(30),x),np.arange(1,31))
    m.transform='log1p';np.testing.assert_allclose(m.reconstruct(np.log1p(np.arange(30)),x),np.arange(30))
    m.transform='residual';np.testing.assert_equal(m.reconstruct(np.ones(30),x),np.arange(1,31))

def test_model_schema_cannot_silently_change():
    class Model:
        feature_names_in_=np.array(['a','b'])
        def predict(self,x):return np.zeros(len(x))
    m=ForecastModel(Model(),FeatureConfig(),np.ones(30))
    with pytest.raises(ValueError):m.predict(pd.DataFrame({'b':[1],'a':[2]}))

def test_calendar_anchor_is_fed_into_later_recursive_steps():
    from ml.advanced.calendar_anchoring import CalendarAnchoredRecursive
    y,t=setup()
    class Model:
        def predict(self,x):return x.lag_1.to_numpy()+1
    class Calendar:
        def forecast(self,y,times,positions,horizon):return np.full((len(positions),horizon,30),100.)
    base=ForecastModel(Model(),FeatureConfig(),np.ones(30))
    m=CalendarAnchoredRecursive(base,Calendar(),.6,12)
    p=m.forecast(y,t,np.array([1800]),3)
    w=.6*(1-np.exp(-1/12))
    expected=(1-w)*1801+w*100
    assert p[0,1,0]==pytest.approx(expected,abs=1e-4)
    w=.6*(1-np.exp(-2/12))
    assert p[0,2,0]==pytest.approx((1-w)*(np.float32(expected)+1)+w*100,abs=1e-4)
