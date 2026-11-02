import asyncio
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import main
from ml.features import COLUMNS
from services.forecasting_service import generate_recursive_forecast
from errors import HistoryUnavailable,WeatherUnavailable

class Model:
    feature_importances_=np.arange(len(COLUMNS))
    def predict(self,x): return x.lag_1.to_numpy()+1

@pytest.fixture
def client(monkeypatch):
    t=pd.date_range('2016-01-28',periods=1600,freq='15min')
    df=pd.DataFrame([(ts,r,float(r)) for ts in t for r in range(30)],columns=['datetime','region','total_pickups']).set_index('datetime')
    monkeypatch.setattr(main,'SEED_DATA',df)
    monkeypatch.setattr(main,'PIPELINE',dict(model=Model(),feature_columns=COLUMNS,categorical_columns=[],schema_version=2))
    return TestClient(main.app)

@pytest.mark.parametrize('region',range(30))
def test_all_regions(client,region):
    response=client.post('/predict',json={'region':region,'datetime':'2016-02-10T00:00:00','horizon':4})
    assert response.status_code==200,response.text
    data=response.json();assert len(data['forecast'])==4
    assert data['forecast'][0]['prediction']==region+1
    assert set(['forecast','history','weather','heatmap','feature_importance','alerts'])<=data.keys()
    assert data['alerts']==[] and data['metadata']['confidence_intervals']=='unavailable'
    assert len(data['heatmap'])==30

@pytest.mark.parametrize('change',[{'region':-1},{'region':30},{'region':1.1},{'horizon':0},{'horizon':49},{'horizon':1.5},{'datetime':'bad'},{'datetime':'2026-11-02'},{'datetime':'2016-02-01T00:00:00'},{'datetime':'2016-02-10T00:01:00'},{'mode':'realtime'}])
def test_invalid(client,change):
    req={'region':0,'datetime':'2016-02-10T00:00:00'};req.update(change)
    assert client.post('/predict',json=req).status_code==422

def test_midnight_weekend_and_default(client):
    data=client.post('/predict',json={'region':0,'datetime':'2016-02-06T04:00:00Z','horizon':48}).json()
    ts=pd.to_datetime([p['datetime'] for p in data['forecast']],utc=True)
    assert len(ts)==48 and (ts.to_series().diff().iloc[1:]==pd.Timedelta('15min')).all()
    assert np.isfinite([p['prediction'] for p in data['forecast']]).all()
    assert len(client.post('/predict',json={'region':0,'datetime':'2016-02-10T00:00:00'}).json()['forecast'])==12

def test_no_observation_overwrite_or_future_leak(client):
    before=main.SEED_DATA.copy(deep=True)
    a=client.post('/predict',json={'region':0,'datetime':'2016-02-10T00:00:00'}).json()
    main.SEED_DATA.loc[main.SEED_DATA.index>=pd.Timestamp('2016-02-10'),'total_pickups']=999
    b=client.post('/predict',json={'region':0,'datetime':'2016-02-10T00:00:00'}).json()
    assert a['forecast']==b['forecast']
    main.SEED_DATA=before

def test_health_and_failures(client,monkeypatch):
    assert client.get('/health').json()['status']=='ok'
    async def fail(*a,**k): raise WeatherUnavailable('secret upstream detail')
    monkeypatch.setattr(main,'generate_recursive_forecast',fail)
    r=client.post('/predict',json={'region':0,'datetime':'2016-02-10T00:00:00'})
    assert r.status_code==503 and 'secret' not in r.text

def test_gapped_context(client):
    main.SEED_DATA=main.SEED_DATA.drop(pd.Timestamp('2016-02-05'))
    r=client.post('/predict',json={'region':0,'datetime':'2016-02-10T00:00:00'})
    assert r.status_code==422

def test_inference_failure_does_not_expose_internals(client,monkeypatch):
    def fail(x): raise RuntimeError('secret path and traceback')
    monkeypatch.setattr(main.PIPELINE['model'],'predict',fail)
    r=client.post('/predict',json={'region':0,'datetime':'2016-02-10T00:00:00'})
    assert r.status_code==503 and 'secret' not in r.text

def test_missing_context_and_unexpected_failure(client,monkeypatch):
    main.SEED_DATA=main.SEED_DATA.iloc[:0]
    assert client.post('/predict',json={'region':0,'datetime':'2016-02-10T00:00:00'}).status_code==422
    async def fail(*a,**k): raise RuntimeError('secret')
    monkeypatch.setattr(main,'generate_recursive_forecast',fail)
    r=client.post('/predict',json={'region':0,'datetime':'2016-02-10T00:00:00'})
    assert r.status_code==500 and 'secret' not in r.text
