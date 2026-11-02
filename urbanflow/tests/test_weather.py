import asyncio
import numpy as np
import pandas as pd
import pytest
import weather_service as w
from errors import WeatherUnavailable

def frame():
    t=pd.date_range('2016-03-05',periods=48,freq='h')
    d=pd.DataFrame(0.,index=t,columns=w.FIELDS)
    d['temperature_2m']=5.; d['relative_humidity_2m']=50
    return d

def test_original_training_weather_bins_and_codes():
    d=frame();d.loc[d.index[0],'weather_code']=3
    result=w.derive_weather(d)
    assert result.iloc[0].weather_condition=='cloudy'
    assert result.iloc[0].temp_bin=='mild'
    assert not result.is_freezing.any()

@pytest.mark.parametrize('kind',['missing','nan'])
def test_missing_weather_is_not_zero(kind):
    d=frame()
    if kind=='missing': d=d.drop(columns='snow_depth')
    else: d.iloc[0,0]=np.nan
    with pytest.raises(WeatherUnavailable): w.derive_weather(d)

def test_full_horizon_cache_expiry_retry(monkeypatch):
    calls=[];now=[100.]
    d=frame();payload={'hourly':dict(time=d.index.astype(str).tolist(),**{c:d[c].tolist() for c in d})}
    class Response:
        def raise_for_status(self): pass
        def json(self): return payload
    class Client:
        def __init__(self,**kw): assert kw['timeout']==15
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def get(self,url,params):
            calls.append((url,params));return Response()
    monkeypatch.setattr(w.httpx,'AsyncClient',Client)
    monkeypatch.setattr(w.time,'monotonic',lambda:now[0]);w.CACHE.clear()
    async def run():
        a=await w.fetch_weather_context('2016-03-05 23:00','2016-03-06 03:00',True)
        assert a.attrs['source']=='historical_reanalysis_not_origin_available'
        await w.fetch_weather_context('2016-03-05 23:00','2016-03-06 03:00',True)
        assert len(calls)==1
        assert calls[0][1]['end_date']=='2016-03-06'
        assert 'wind_speed_10m' in calls[0][1]['hourly']
        now[0]+=w.TTL+1
        await w.fetch_weather_context('2016-03-05 23:00','2016-03-06 03:00',True)
        assert len(calls)==2
    asyncio.run(run())

def test_network_failures_have_bounded_retries(monkeypatch):
    calls=[]
    class Client:
        def __init__(self,**kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def get(self,*args,**kw):
            calls.append(1);raise w.httpx.ConnectError('unavailable')
    async def sleep(*args): pass
    monkeypatch.setattr(w.httpx,'AsyncClient',Client);monkeypatch.setattr(w.asyncio,'sleep',sleep);w.CACHE.clear()
    with pytest.raises(WeatherUnavailable): asyncio.run(w.fetch_weather_context('2016-03-05','2016-03-06',True))
    assert len(calls)==3
