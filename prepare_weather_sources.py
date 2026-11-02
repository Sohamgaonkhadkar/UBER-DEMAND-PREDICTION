"""Cache official API inputs, validate UTC weather and derive centroid area names.
No runtime geocoder is called by website visitors. NYC polygons label centroids;
they never change fitted model regions or claim exact cluster boundaries.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import httpx,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/weather_20261102'
FIELDS=['temperature_2m','apparent_temperature','precipitation','snowfall','wind_speed_10m','relative_humidity_2m','weather_code','cloud_cover']

def fetch(filename,url,params,source_name):
    path=OUT/filename;source=OUT/source_name
    if path.exists() and source.exists():
        info=json.loads(source.read_text(encoding='utf-8'))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Cached source hash mismatch')
        return json.loads(path.read_text(encoding='utf-8'))
    with httpx.Client(timeout=120,follow_redirects=True) as client:
        r=client.get(url,params=params);r.raise_for_status();data=r.json()
    path.write_bytes(r.content)
    info=dict(url=str(r.url),sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content),retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    source.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8');return data

def ring_contains(point,ring):
    x,y=point;inside=False
    for a,b in zip(ring,ring[1:]+ring[:1]):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside

def contains(point,geometry):
    polygons=geometry['coordinates'] if geometry['type']=='MultiPolygon' else [geometry['coordinates']]
    for rings in polygons:
        ring=rings[0];xs=[p[0] for p in ring];ys=[p[1] for p in ring]
        if not(min(xs)<=point[0]<=max(xs) and min(ys)<=point[1]<=max(ys)):continue
        if ring_contains(point,ring) and not any(ring_contains(point,hole) for hole in rings[1:]):return True
    return False

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    payload=fetch('provider_response.json','https://archive-api.open-meteo.com/v1/archive',dict(latitude=40.7128,longitude=-74.0060,start_date='2015-12-31',end_date='2016-07-01',hourly=','.join(FIELDS),timezone='UTC',timeformat='unixtime',models='era5',temperature_unit='celsius',wind_speed_unit='kmh',precipitation_unit='mm'),'source.json')
    frame=pd.DataFrame(payload['hourly']);frame.index=pd.to_datetime(frame.pop('time'),unit='s',utc=True)
    if frame.index.has_duplicates or not np.isfinite(frame.to_numpy(dtype=float)).all():raise ValueError('Incomplete weather response')
    if not frame.index.equals(pd.date_range(frame.index[0],frame.index[-1],freq='h')):raise ValueError('Weather grid is incomplete')
    frame.to_parquet(OUT/'weather_hourly.parquet')
    polygons=fetch('nta_geometry.geojson','https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson',{'$limit':500},'nta_source.json')
    geo=json.loads((ROOT/'artifacts/advanced_20261102/geography.json').read_text())
    points=np.asarray(geo['centers'])*geo['scale']+geo['mean'];labels=[]
    for i,p in enumerate(points):
        hits=[f for f in polygons['features'] if contains(p,f['geometry'])]
        if len(hits)!=1:raise ValueError(f'Centroid {i} lacks a unique area label')
        props=hits[0]['properties'];name=props['ntaname']
        item=dict(id=i,name='JFK Airport' if name=='John F. Kennedy International Airport' else name,official_nta_name=name,borough=props['boroname'],nta_code=props['nta2020'],lat=float(p[1]),lon=float(p[0]),label_kind='2020 NYC NTA containing the model centroid; approximate area label, not model boundaries')
        labels.append(item)
    result=dict(source=json.loads((OUT/'nta_source.json').read_text()),method='Point-in-polygon using inverse-scaled frozen centroids and official 2020 NYC NTA geometry; polygon holes respected.',labels=labels)
    (OUT/'region_labels.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Validated',len(frame),'UTC weather hours and',len(labels),'coordinate-derived area names')
if __name__=='__main__':main()
