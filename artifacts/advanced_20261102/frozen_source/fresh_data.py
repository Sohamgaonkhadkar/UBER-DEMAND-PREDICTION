"""Recover fixed historical geography, then stream fresh official April records.

No model tuning/evaluation reads the April aggregate until selection is frozen.
"""
import concurrent.futures,hashlib,json,time
import httpx,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin
from threadpoolctl import threadpool_limits
from .protocol import *

def recover_geography():
    path=OUT/'geography.json'
    if path.exists():return json.loads(path.read_text())
    initialize()
    nb=json.loads((ROOT/'Creating-Historical-Data.ipynb').read_text(encoding='utf-8'))
    source=''.join(nb['cells'][20]['outputs'][0]['data']['text/plain'])
    import ast
    centers=np.array(ast.literal_eval(source[source.index('(')+1:source.rindex(')')]))
    scaler=StandardScaler()
    for c in pd.read_csv(ROOT/'data/processing_data.csv',usecols=['pickup_longitude','pickup_latitude'],chunksize=200_000):scaler.partial_fit(c)
    mismatches=0;n=0
    with threadpool_limits(limits=4):
        for c,labels in zip(pd.read_csv(ROOT/'data/processing_data.csv',chunksize=200_000),pd.read_csv(ROOT/'data/time_series.csv',chunksize=200_000)):
            if not c.tpep_pickup_datetime.equals(labels.tpep_pickup_datetime):raise ValueError('Source rows do not align')
            pred=pairwise_distances_argmin(scaler.transform(c[['pickup_longitude','pickup_latitude']]),centers)
            mismatches+=int(np.sum(pred!=labels.region.to_numpy()));n+=len(c)
    result=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),centers=centers.tolist(),
        original_region_rows_verified=n,mismatches=mismatches,centroid_precision='8 decimal digits recovered from stored notebook output; scaler refit exactly with source row order',
        fit_period='January-March 2016; fixed before April',columns=['pickup_longitude','pickup_latitude'])
    path.write_text(json.dumps(result,indent=2));print('geography verification',n,mismatches,flush=True)
    if mismatches:raise ValueError('Recovered geography does not exactly reproduce historical labels; investigate before fresh test')
    return result

def fetch_day(date,geo):
    dest=OUT/'april_daily'/f'{date:%Y-%m-%d}.parquet'
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
            with httpx.Client(timeout=httpx.Timeout(180,connect=20)) as client:
                response=client.get('https://data.cityofnewyork.us/resource/uacg-pexx.json',params=params)
                response.raise_for_status();raw=response.json()
            if not isinstance(raw,list) or not raw or len(raw)>=1_000_000:raise ValueError('Unusable or possibly truncated source response')
            c=pd.DataFrame(raw)
            coords=c[['pickup_longitude','pickup_latitude']].astype(float).to_numpy()
            with threadpool_limits(limits=2):labels=pairwise_distances_argmin((coords-np.array(geo['mean']))/geo['scale'],np.array(geo['centers']))
            c['datetime']=pd.to_datetime(c.tpep_pickup_datetime).dt.floor('15min').dt.tz_localize(TZ,ambiguous='raise',nonexistent='raise')
            c['region']=labels
            counts=c.groupby(['datetime','region']).size().rename('total_pickups').reset_index()
            dest.parent.mkdir(exist_ok=True);counts.to_parquet(dest,index=False)
            meta=dict(date=str(date.date()),retained_rows=len(c),response_sha256=hashlib.sha256(response.content).hexdigest(),
                request_url=str(response.url),retrieved_at=pd.Timestamp.now(tz='UTC').isoformat(),limit=1_000_000,returned_below_limit=True)
            dest.with_suffix('.json').write_text(json.dumps(meta,indent=2))
            print('fresh source aggregated',date.date(),'rows',len(c),flush=True)
            return counts
        except (httpx.HTTPError,ValueError,KeyError) as exc:
            last=exc
            if attempt<2:time.sleep(2**attempt)
    raise RuntimeError(f'Could not retrieve complete day {date}') from last

def main():
    geo=recover_geography()
    dest=OUT/'april_holdout.parquet'
    if dest.exists():print('Fresh holdout already staged; not inspected');return
    days=pd.date_range('2016-04-01','2016-04-30',freq='D')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pieces=list(pool.map(lambda day:fetch_day(day,geo),days))
    counts=pd.concat(pieces).set_index(['datetime','region']).total_pickups
    grid=pd.MultiIndex.from_product([pd.date_range('2016-04-01','2016-05-01',inclusive='left',freq='15min',tz=TZ),range(30)],names=['datetime','region'])
    panel=counts.reindex(grid,fill_value=0).astype('int64').rename('total_pickups').reset_index()
    panel.to_parquet(dest,index=False)
    (OUT/'fresh_holdout_provenance.json').write_text(json.dumps(dict(source='https://data.cityofnewyork.us/Transportation/2016-Yellow-Taxi-Trip-Data/uacg-pexx',
        fixed_filter='Identical to Creating-Historical-Data.ipynb cell 5-7; bounds/fare/distance frozen before April requests',
        sha256=sha(dest),region_recovery_mismatches=geo['mismatches'],policy='All April days and all 30 fixed regions; no difficult origins removed. No target summary or predictive error inspected during acquisition.'),indent=2))
    print('fresh holdout staged; target values sealed from search',flush=True)

if __name__=='__main__':main()
