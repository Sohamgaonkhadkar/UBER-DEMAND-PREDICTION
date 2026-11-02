import asyncio
import numpy as np,pandas as pd,pytest
from errors import HistoryUnavailable,InferenceFailure
from ml.advanced.core import HISTORY
from advanced_main import advanced_forecast

class JointModel:
    def forecast(self,y,times,positions,horizon):
        return np.repeat(y[-1:][None,:,:],horizon,axis=1)+1

def seed():
    times=pd.date_range('2016-02-01',periods=HISTORY+100,freq='15min',tz='America/New_York')
    return pd.DataFrame([(t,r,float(r+1)) for t in times for r in range(30)],columns=['datetime','region','total_pickups']).set_index('datetime')

def artifact():return dict(schema_version=3,model=JointModel(),horizon=48,timezone='America/New_York')

@pytest.mark.parametrize('region',range(30))
def test_all_regions_and_future_truth_exclusion(region):
    df=seed();origin=df.index.unique()[HISTORY]
    before=df.copy(deep=True)
    p,_=asyncio.run(advanced_forecast(region,origin,12,df,artifact()))
    assert len(p)==12 and p[0]['prediction']==region+2
    df.loc[df.index>=origin,'total_pickups']=999999
    q,_=asyncio.run(advanced_forecast(region,origin,12,df,artifact()))
    assert p==q
    pd.testing.assert_frame_equal(before.loc[before.index<origin],df.loc[df.index<origin])

def test_joint_context_and_schema_failures():
    df=seed();origin=df.index.unique()[HISTORY]
    with pytest.raises(HistoryUnavailable):asyncio.run(advanced_forecast(0,origin,12,df[df.region!=1],artifact()))
    with pytest.raises(HistoryUnavailable):asyncio.run(advanced_forecast(0,origin,12,df.drop(df.index[300]),artifact()))
    with pytest.raises(HistoryUnavailable):asyncio.run(advanced_forecast(0,origin,12,pd.concat([df,df.iloc[[0]]]),artifact()))
    bad=artifact();bad['horizon']=12
    with pytest.raises(InferenceFailure):asyncio.run(advanced_forecast(0,origin,12,df,bad))

def test_refitted_model_cannot_backtest_its_own_training_period():
    df=seed();origin=df.index.unique()[HISTORY]
    p=artifact();p['training_end']='2016-04-01T00:00:00-04:00'
    with pytest.raises(HistoryUnavailable,match='training cutoff'):
        asyncio.run(advanced_forecast(0,origin,12,df,p))
