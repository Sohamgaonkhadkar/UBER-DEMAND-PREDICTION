import datetime,json
from collections import Counter
from types import SimpleNamespace
import httpx,pandas as pd
from ml.v4 import acquire

def test_retry_reuses_completed_intervals_and_keeps_coincident_trips(tmp_path,monkeypatch):
    calls=Counter();failed=False
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,url,params):
            nonlocal failed
            begin=params['$where'].split("'")[1];hour=pd.Timestamp(begin).hour;calls[hour]+=1
            if hour==12 and not failed:failed=True;raise httpx.ReadTimeout('simulated incomplete response')
            text='tpep_pickup_datetime,pickup_longitude,pickup_latitude\n'+f'{begin},-73.9,40.7\n'*2
            return SimpleNamespace(raise_for_status=lambda:None,text=text,content=text.encode(),elapsed=datetime.timedelta(seconds=1),url=url)
    monkeypatch.setattr(acquire.httpx,'Client',Client);monkeypatch.setattr(acquire.time,'sleep',lambda n:None)
    geo=dict(mean=[-74,40],scale=[1,1],centers=[[0,0],[1,1]])
    result=acquire.day(pd.Timestamp('2015-11-22'),geo,tmp_path)
    assert result.total_pickups.sum()==8
    assert calls==Counter({0:1,6:1,12:2,18:1})
    meta=json.loads((tmp_path/'2015-11-22.json').read_text());assert meta['retained_trip_rows']==8
    again=acquire.day(pd.Timestamp('2015-11-22'),geo,tmp_path)
    pd.testing.assert_frame_equal(result,again)
    assert sum(calls.values())==5
