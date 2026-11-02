import numpy as np
import pandas as pd
import pytest
from ml.features import *

def panel():
    t=pd.date_range('2016-02-01',periods=800,freq='15min')
    return pd.DataFrame([(ts,r,float(i+r*1000)) for i,ts in enumerate(t) for r in [0,1]],columns=['datetime','region','total_pickups'])

def test_training_inference_parity():
    p,x=training_features(panel())
    t=p.datetime.max(); wide=p.pivot(index='datetime',columns='region',values='total_pickups')
    h=wide.loc[wide.index<t].to_numpy()
    ewma=h[0].copy()
    for row in h[1:]: ewma=.4*row+.6*ewma
    online=next_features(h,t,[0,1],ewma)
    np.testing.assert_allclose(online,x.loc[p.datetime==t],rtol=1e-10,atol=1e-8)

def test_future_targets_never_change_past_features():
    p=panel();_,before=training_features(p)
    cut=p.datetime.unique()[750]
    p.loc[p.datetime>=cut,'total_pickups']=99999
    _,after=training_features(p)
    np.testing.assert_allclose(before[p.datetime<=cut],after[p.datetime<=cut],equal_nan=True)

def test_daily_weekly_region_and_zero():
    p,x=training_features(panel()); row=x.iloc[-1]
    assert row.lag_96==1703 and row.lag_672==1127
    assert x.loc[1440,'lag_720']==0
    assert p.total_pickups.iloc[0]==0

@pytest.mark.parametrize('kind',['gap','duplicate','nan'])
def test_reject_invalid_history(kind):
    p=panel()
    if kind=='gap': p=p.drop(index=20)
    if kind=='duplicate': p=pd.concat([p,p.iloc[[0]]])
    if kind=='nan': p.loc[20,'total_pickups']=np.nan
    with pytest.raises(ValueError): validate_panel(p)

def test_recursive_feedback_and_no_mutation():
    class Model:
        def predict(self,x): return x.lag_1.to_numpy()+1
    h=np.zeros((720,2)); p=recursive(Model(),h,pd.Timestamp('2016-02-01'),[0,1],4)
    np.testing.assert_equal(p[:,0],[1,2,3,4]);assert (h==0).all()

def test_timestamp_gap_split_on_region_major_rows():
    p=panel().sort_values(['region','datetime'])
    a,b,c=chronological_split(p,'2016-02-05','2016-02-07')
    assert p.loc[a,'datetime'].max()<p.loc[b,'datetime'].min()-pd.Timedelta('2h')
    assert set(p[a].region)==set(p[b].region)==set(p[c].region)
