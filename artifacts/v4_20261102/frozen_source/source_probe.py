"""Read source metadata and test exact server-side fixed-region aggregation."""
import json,time
import httpx,numpy as np,pandas as pd
from scipy.spatial import Voronoi
from .protocol import *

FILTERS='pickup_latitude between 40.60 and 40.85 AND pickup_longitude between -74.05 and -73.70 AND dropoff_latitude between 40.60 and 40.85 AND dropoff_longitude between -74.05 and -73.70 AND fare_amount between 0.50 and 81.0 AND trip_distance between 0.25 and 24.43'

def cell_condition(geo,r):
    centers=np.asarray(geo['centers']);scale=np.asarray(geo['scale']);mean=np.asarray(geo['mean'])
    # Only Voronoi-neighbor halfspaces are necessary, not all 29 inequalities.
    pairs=Voronoi(centers).ridge_points;others=[int(b if a==r else a) for a,b in pairs if r in [a,b]]
    clauses=[]
    for j in sorted(others):
        d=2*(centers[j]-centers[r])/scale;c=float(centers[j]@centers[j]-centers[r]@centers[r]+d@mean)
        # Lower ID wins exact distance ties, consistent with argmin.
        op='<' if j<r else '<='
        clauses.append(f'(({d[0]:.17g}) * pickup_longitude + ({d[1]:.17g}) * pickup_latitude {op} ({c:.17g}))')
    return ' AND '.join(clauses)

def query(client,dataset,start,end,r,geo):
    slot='date_extract_hh(tpep_pickup_datetime) * 4 + floor(date_extract_mm(tpep_pickup_datetime) / 15)'
    params={'$select':f'date_trunc_ymd(tpep_pickup_datetime) as day, {slot} as slot, count(*) as n',
        '$where':f"tpep_pickup_datetime >= '{start}T00:00:00' AND tpep_pickup_datetime < '{end}T00:00:00' AND {FILTERS} AND {cell_condition(geo,r)}",
        '$group':f'date_trunc_ymd(tpep_pickup_datetime), {slot}','$limit':100000,'$order':'day, slot'}
    t=time.perf_counter();response=client.get(f'https://data.cityofnewyork.us/resource/{dataset}.json',params=params);response.raise_for_status();raw=response.json()
    return raw,dict(seconds=time.perf_counter()-t,request=str(response.url),response_sha256=__import__('hashlib').sha256(response.content).hexdigest(),rows=len(raw))

def main():
    initialize();geo=json.loads((OLD/'geography.json').read_text());out={}
    with httpx.Client(timeout=120) as client:
        for dataset in ['2yzn-sicd','uacg-pexx']:
            response=client.get(f'https://data.cityofnewyork.us/api/views/{dataset}.json');response.raise_for_status();meta=response.json()
            out[dataset]=dict(name=meta.get('name'),columns=[c.get('fieldName') for c in meta['columns']])
        raw,record=query(client,'uacg-pexx','2016-05-16','2016-05-17',2,geo)
        existing=panel();expected=existing[(existing.region==2)&(existing.datetime>=pd.Timestamp('2016-05-16',tz=TZ))&(existing.datetime<pd.Timestamp('2016-05-17',tz=TZ))].sort_values('datetime').total_pickups.to_numpy()
        result=np.zeros(96,dtype=int)
        for row in raw:result[int(row['slot'])]=int(row['n'])
        record['exact_match_existing_day']=bool(np.array_equal(expected,result));out['known_day_probe']=record
        if not record['exact_match_existing_day']:raise ValueError('Server assignment differs; cannot use historical data')
        raw,record=query(client,'2yzn-sicd','2015-01-01','2015-02-01',2,geo);out['historical_probe']=record
    save(REPORT/'source_feasibility.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
