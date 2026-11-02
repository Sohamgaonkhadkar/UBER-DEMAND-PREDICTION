import numpy as np,pandas as pd
from ml.advanced.core import FeatureConfig,ForecastModel,HISTORY
from ml.v4.models import CoarseForecast,FineForecast,ReconciledForecast,holiday_offsets,coarse_features

class LastPlusOne:
    def predict(self,x):return x.lag_1.to_numpy()+1

class Constant:
    def __init__(self,n):self.n=n
    def predict(self,x):return np.full(len(x),self.n)

def fixture():
    times=pd.date_range('2016-02-15',periods=4000,freq='15min',tz='America/New_York')
    y=np.ones((4000,30),dtype=np.float32)*3
    return y,times,np.array([HISTORY+1,HISTORY+2,HISTORY+3])

def coarse(factor,value=12,city=False):
    return CoarseForecast(Constant(value),factor,np.full((7,96,1 if city else 30,factor),1/factor),city)

def test_holiday_offsets_distinguish_before_after_and_unknown_summer():
    t=pd.DatetimeIndex(['2016-07-03','2016-07-04','2016-07-05'],tz='America/New_York')
    event,prev,nxt,_,_=holiday_offsets(t)
    np.testing.assert_array_equal(event,[False,True,False]);np.testing.assert_array_equal(prev,[False,False,True]);np.testing.assert_array_equal(nxt,[True,False,False])

def test_arbitrary_quarter_origins_conserve_predicted_hourly_totals():
    y,t,pos=fixture();model=coarse(4);pred=model.forecast(y,t,pos,48)
    np.testing.assert_allclose(pred.reshape(3,12,4,30).sum(axis=2),model.totals(y,t,pos))
    assert set(t[pos].minute)=={15,30,45}

def test_coarse_future_poison_and_prefix_consistency():
    y,t,pos=fixture();model=coarse(4);pred=model.forecast(y,t,pos[:1],48)
    y[pos[0]:]=99999
    np.testing.assert_equal(pred,model.forecast(y,t,pos[:1],48))
    for h in [1,4,12,24]:np.testing.assert_equal(pred[:,:h],model.forecast(y,t,pos[:1],h))

def test_reconciliation_nonnegative_and_all_levels_are_coherent():
    y,t,pos=fixture();fine=FineForecast(ForecastModel(Constant(7),FeatureConfig(),np.ones(30)))
    model=ReconciledForecast(fine,coarse(2,8),coarse(4,24),coarse(4,900,True));p,trace=model.generate(y,t,pos)
    assert np.isfinite(p).all() and (p>=0).all()
    np.testing.assert_allclose(p.reshape(3,12,4,30).sum(axis=2),trace['hourly'])
    np.testing.assert_allclose(p.reshape(3,24,2,30).sum(axis=2),trace['half_hourly'])
    np.testing.assert_allclose(p.reshape(3,12,4,30).sum(axis=(2,3)),trace['city_hourly'])
    np.testing.assert_allclose(trace['city_hourly'],(np.full((3,12),(7*4+24)/2*30)+900)/2)

def test_fine_outputs_really_feed_back():
    y,t,pos=fixture();model=FineForecast(ForecastModel(LastPlusOne(),FeatureConfig(),np.ones(30)))
    pred=model.forecast(y,t,pos[:1],4)
    np.testing.assert_allclose(pred[0,:,0],[4,5,6,7])

def test_dst_coarse_clock_features_use_only_origin_history():
    y,t,pos=fixture();target=pd.DatetimeIndex(['2016-03-13 04:15'],tz='America/New_York')
    history=np.arange(HISTORY//4,dtype=np.float32)[None,:,None]
    x=coarse_features(history,target,4)
    # Yesterday at same clock spans 23 elapsed hours across spring DST.
    assert x.clock_1.iloc[0]==HISTORY//4-23

def test_temporal_distribution_uses_no_future_observations():
    y,t,pos=fixture();fine=FineForecast(ForecastModel(Constant(7),FeatureConfig(),np.ones(30)))
    model=ReconciledForecast(fine,coarse(2),coarse(4));pred=model.forecast(y,t,pos[:1],48)
    y[pos[0]:]=1e8
    np.testing.assert_equal(pred,model.forecast(y,t,pos[:1],48))

def test_zero_lower_forecasts_can_reconcile_positive_city_total():
    y,t,pos=fixture();fine=FineForecast(ForecastModel(Constant(0),FeatureConfig(),np.ones(30)))
    model=ReconciledForecast(fine,coarse(2,0),coarse(4,0),coarse(4,120,True))
    p,trace=model.generate(y,t,pos)
    np.testing.assert_allclose(trace['city_hourly'],60)
    assert np.isfinite(p).all() and (p>=0).all()

def test_training_excludes_every_context_crossing_an_unacquired_gap():
    from ml.v4.fit import valid_training_positions
    first=pd.date_range('2015-11-02',periods=HISTORY+20,freq='15min',tz='America/New_York')
    second=pd.date_range('2016-01-01',periods=HISTORY+30,freq='15min',tz='America/New_York')
    times=first.append(second);positions=valid_training_positions(times)
    np.testing.assert_array_equal(positions,np.r_[np.arange(HISTORY,len(first)),np.arange(len(first)+HISTORY,len(times))])

def test_ordinary_profile_feedback_preserves_canonical_arithmetic_exactly():
    from ml.advanced.calendar_anchoring import CalendarAnchoredRecursive
    from ml.nextgen.models import Profile
    y,t,pos=fixture();base=ForecastModel(LastPlusOne(),FeatureConfig(),np.ones(30),anchor_weight=.3)
    profile=Profile(np.full((7,96,30),10,dtype=np.float32))
    canonical=CalendarAnchoredRecursive(base,profile,.6,12)
    new=FineForecast(base,profile=profile)
    np.testing.assert_array_equal(new.forecast(y,t,pos,48),canonical.forecast(y,t,pos,48))
