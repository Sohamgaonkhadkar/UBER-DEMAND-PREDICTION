"""A bounded second feasibility probe with exact cell bounding-box pushdown."""
import json,time
import httpx,numpy as np,pandas as pd
from scipy.spatial import HalfspaceIntersection
from .protocol import *
from .source_probe import FILTERS,cell_condition

def bounds(geo,r):
    c=np.asarray(geo['centers']);mean=np.asarray(geo['mean']);scale=np.asarray(geo['scale']);a=2*(c-c[r]);b=np.sum(c*c,axis=1)-c[r]@c[r]
    hs=[list(a[j])+[-b[j]] for j in range(30) if j!=r]
    for k,(lo,hi) in enumerate([(-74.05,-73.70),(40.60,40.85)]):
        left=np.zeros(2);left[k]=-1;right=-left
        hs.extend([list(left)+[(lo-mean[k])/scale[k]],list(right)+[-(hi-mean[k])/scale[k]]])
    vertices=HalfspaceIntersection(np.asarray(hs),c[r]).intersections*scale+mean
    return vertices.min(axis=0)-1e-9,vertices.max(axis=0)+1e-9

def grouped(client,dataset,start,end,r,geo):
    field='pickup_datetime' if dataset=='2yzn-sicd' else 'tpep_pickup_datetime';lo,hi=bounds(geo,r)
    slot=f'date_extract_hh({field}) * 4 + floor(date_extract_mm({field}) / 15)';day=f'date_trunc_ymd({field})'
    where=f"{field} >= '{start}T00:00:00' AND {field} < '{end}T00:00:00' AND pickup_longitude between {lo[0]:.17g} and {hi[0]:.17g} AND pickup_latitude between {lo[1]:.17g} and {hi[1]:.17g} AND {FILTERS} AND {cell_condition(geo,r)}"
    params={'$select':f'{day} as day,{slot} as slot,count(*) as n','$where':where,'$group':day+','+slot,'$limit':100000}
    begun=time.perf_counter();response=client.get(f'https://data.cityofnewyork.us/resource/{dataset}.json',params=params);response.raise_for_status()
    return response.json(),dict(seconds=time.perf_counter()-begun,request=str(response.url),sha256=__import__('hashlib').sha256(response.content).hexdigest())

def main():
    geo=json.loads((OLD/'geography.json').read_text());out={}
    try:
        with httpx.Client(timeout=180) as client:
            raw,record=grouped(client,'2yzn-sicd','2015-11-09','2015-11-10',2,geo)
            expected=pd.read_parquet(OUT/'daily/history_2015/2015-11-09.parquet');expected=expected[expected.region==2].set_index('datetime').total_pickups
            index=pd.date_range('2015-11-09','2015-11-10',freq='15min',inclusive='left',tz=TZ);a=expected.reindex(index,fill_value=0).to_numpy();p=np.zeros(96,dtype=int)
            for row in raw:p[int(row['slot'])]=int(row['n'])
            out['known_day']=dict(record,match=bool(np.array_equal(a,p)));print(out,flush=True)
            if not np.array_equal(a,p):raise ValueError('Exact assignment mismatch')
            raw,record=grouped(client,'2yzn-sicd','2015-12-01','2016-01-01',2,geo);out['month']=dict(record,rows=len(raw));print(out,flush=True)
    except Exception as exc:out['error']=str(exc);print(type(exc).__name__,str(exc),flush=True)
    save(REPORT/'aggregate_probe.json',out)

if __name__=='__main__':main()
