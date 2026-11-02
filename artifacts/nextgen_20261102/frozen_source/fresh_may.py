"""Stream one compatible unseen month; do not inspect forecast outcomes."""
import concurrent.futures,hashlib,json,time
import httpx,numpy as np,pandas as pd
from sklearn.metrics import pairwise_distances_argmin
from threadpoolctl import threadpool_limits
from .protocol import *

def day(date,geo):
    dest=OUT/'may_daily'/f'{date:%Y-%m-%d}.parquet'
    if dest.exists():return pd.read_parquet(dest)
    end=date+pd.Timedelta(days=1)
    where=(f"tpep_pickup_datetime >= '{date:%Y-%m-%d}T00:00:00' AND tpep_pickup_datetime < '{end:%Y-%m-%d}T00:00:00' "
        "AND pickup_latitude between 40.60 and 40.85 AND pickup_longitude between -74.05 and -73.70 "
        "AND dropoff_latitude between 40.60 and 40.85 AND dropoff_longitude between -74.05 and -73.70 "
        "AND fare_amount between 0.50 and 81.0 AND trip_distance between 0.25 and 24.43")
    params={'$select':'tpep_pickup_datetime,pickup_longitude,pickup_latitude','$where':where,'$limit':1_000_000}
    last=None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=httpx.Timeout(180,connect=20)) as c:
                response=c.get('https://data.cityofnewyork.us/resource/uacg-pexx.json',params=params);response.raise_for_status();raw=response.json()
            if not isinstance(raw,list) or not raw or len(raw)>=1_000_000:raise ValueError('Invalid or potentially truncated source day')
            data=pd.DataFrame(raw);coords=data[['pickup_longitude','pickup_latitude']].astype(float).to_numpy()
            with threadpool_limits(limits=2):labels=pairwise_distances_argmin((coords-geo['mean'])/geo['scale'],np.array(geo['centers']))
            data['datetime']=pd.to_datetime(data.tpep_pickup_datetime).dt.floor('15min').dt.tz_localize(TZ,ambiguous='raise',nonexistent='raise');data['region']=labels
            counts=data.groupby(['datetime','region']).size().rename('total_pickups').reset_index()
            dest.parent.mkdir(exist_ok=True);counts.to_parquet(dest,index=False)
            dest.with_suffix('.json').write_text(json.dumps(dict(date=str(date.date()),retained_rows=len(data),returned_below_limit=True,request=str(response.url),response_sha256=hashlib.sha256(response.content).hexdigest(),retrieved_at=pd.Timestamp.now(tz='UTC').isoformat()),indent=2))
            print('May source staged',date.date(),len(data),flush=True);return counts
        except (httpx.HTTPError,ValueError,KeyError) as exc:
            last=exc
            if attempt<2:time.sleep(2**attempt)
    raise RuntimeError(f'Complete day unavailable: {date}') from last

def main():
    initialize();dest=OUT/'may_sealed.parquet'
    if dest.exists():print('May already staged; no forecast outcomes examined');return
    geo=json.loads((OLD/'geography.json').read_text())
    if geo['mismatches']!=0:raise ValueError('Mapping verification failed')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        chunks=list(pool.map(lambda date:day(date,geo),pd.date_range('2016-05-01','2016-05-31',freq='D')))
    s=pd.concat(chunks).set_index(['datetime','region']).total_pickups
    index=pd.MultiIndex.from_product([pd.date_range('2016-05-01','2016-06-01',freq='15min',inclusive='left',tz=TZ),range(30)],names=['datetime','region'])
    panel=s.reindex(index,fill_value=0).astype('int64').rename('total_pickups').reset_index();panel.to_parquet(dest,index=False)
    (OUT/'may_provenance.json').write_text(json.dumps(dict(source='https://data.cityofnewyork.us/Transportation/2016-Yellow-Taxi-Trip-Data/uacg-pexx',sha256=sha(dest),mapping_sha256=sha(OLD/'geography.json'),policy='Original filters/map; all 31 days, all regions; no outcome-based selection. Test targets sealed from research functions.'),indent=2))
    print('May sealed. Predictive access requires frozen selection.',flush=True)

if __name__=='__main__':main()
