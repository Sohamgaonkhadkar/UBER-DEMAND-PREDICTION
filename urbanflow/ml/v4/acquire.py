"""Bounded historical/final acquisition, fixed original filters and geography."""
import concurrent.futures,hashlib,time,json,io
import httpx,numpy as np,pandas as pd
from sklearn.metrics import pairwise_distances_argmin
from threadpoolctl import threadpool_limits
from .protocol import *
from .source_probe import FILTERS

def day(date,geo,folder):
    dest=folder/f'{date:%Y-%m-%d}.parquet'
    if dest.exists():
        counts=pd.read_parquet(dest);meta=json.loads(dest.with_suffix('.json').read_text())
        if int(counts.total_pickups.sum())!=meta['retained_trip_rows']:raise ValueError('Cached day total differs from source provenance')
        return counts
    end=date+pd.Timedelta(days=1);dataset='2yzn-sicd' if date.year==2015 else 'uacg-pexx'
    timestamp='pickup_datetime' if date.year==2015 else 'tpep_pickup_datetime'
    part_folder=folder/'parts';part_folder.mkdir(parents=True,exist_ok=True)
    def interval(begin,stop):
        file=part_folder/f'{begin:%Y-%m-%d_%H}-{stop:%Y-%m-%d_%H}.parquet'
        if file.exists():
            counts=pd.read_parquet(file);meta=json.loads(file.with_suffix('.json').read_text())
            if int(counts.total_pickups.sum())!=meta['rows']:raise ValueError('Partial-interval provenance mismatch')
            return counts,meta
        params={'$select':f'{timestamp} as tpep_pickup_datetime,pickup_longitude,pickup_latitude','$where':f"{timestamp} >= '{begin:%Y-%m-%dT%H:%M:%S}' AND {timestamp} < '{stop:%Y-%m-%dT%H:%M:%S}' AND {FILTERS}",'$limit':1000000}
        with httpx.Client(timeout=httpx.Timeout(300,connect=20)) as client:
            response=client.get(f'https://data.cityofnewyork.us/resource/{dataset}.csv',params=params);response.raise_for_status()
        frame=pd.read_csv(io.StringIO(response.text),float_precision='round_trip')
        if frame.empty or len(frame)>=1000000:raise ValueError('Missing/potentially truncated interval')
        coords=frame[['pickup_longitude','pickup_latitude']].astype(float).to_numpy()
        with threadpool_limits(limits=2):regions=pairwise_distances_argmin((coords-np.asarray(geo['mean']))/np.asarray(geo['scale']),np.asarray(geo['centers']))
        dt=pd.to_datetime(frame.tpep_pickup_datetime).dt.floor('15min').dt.tz_localize(TZ,ambiguous='raise',nonexistent='raise')
        if dt.min()<begin.tz_localize(TZ) or dt.max()>=stop.tz_localize(TZ):raise ValueError('Source timestamps outside requested interval')
        counts=pd.DataFrame(dict(datetime=dt,region=regions)).groupby(['datetime','region']).size().rename('total_pickups').reset_index()
        meta=dict(request=str(response.url),response_bytes=len(response.content),seconds=response.elapsed.total_seconds(),response_sha256=hashlib.sha256(response.content).hexdigest(),rows=len(frame),aggregate_sha256=None)
        temporary=file.with_suffix('.partial');counts.to_parquet(temporary,index=False);temporary.replace(file);meta['aggregate_sha256']=sha(file);save(file.with_suffix('.json'),meta)
        print('Interval cached',begin,stop,len(frame),flush=True);return counts,meta
    for attempt in range(3):
        try:
            parts=[];frames=[]
            width=6 if date==pd.Timestamp('2015-11-22') or attempt==2 else (12 if attempt==1 else 24)
            intervals=[(date+pd.Timedelta(hours=k),date+pd.Timedelta(hours=k+width)) for k in range(0,24,width)]
            for begin,stop in intervals:
                counts,part=interval(begin,stop);frames.append(counts);parts.append(part)
            counts=pd.concat(frames).groupby(['datetime','region']).total_pickups.sum().reset_index();n=sum(p['rows'] for p in parts)
            dest.parent.mkdir(exist_ok=True,parents=True);counts.to_parquet(dest,index=False)
            save(dest.with_suffix('.json'),dict(date=str(date.date()),dataset=dataset,retained_trip_rows=n,parts=parts,response_format='CSV, round-trip float parsing',response_bytes=sum(p['response_bytes'] for p in parts),seconds=sum(p['seconds'] for p in parts),retrieved_at=pd.Timestamp.now(tz='UTC').isoformat(),potential_truncation=False))
            print('Aggregated',date.date(),n,flush=True);return counts
        except (httpx.HTTPError,ValueError,KeyError) as exc:
            print('Source retry',date.date(),attempt,type(exc).__name__,flush=True)
            if attempt==2:raise
            time.sleep(2**attempt)

def period(start,end,name):
    initialize();dest=OUT/(name+'.parquet')
    if dest.exists():print('Preserving existing',name,flush=True);return
    geo=json.loads((OLD/'geography.json').read_text());folder=OUT/'daily'/name
    dates=pd.date_range(start,pd.Timestamp(end)-pd.Timedelta('1d'),freq='D')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        chunks=list(pool.map(lambda date:day(date,geo,folder),dates))
    s=pd.concat(chunks).set_index(['datetime','region']).total_pickups
    if s.index.duplicated().any():raise ValueError('Overlapping/duplicate source aggregation')
    index=pd.MultiIndex.from_product([pd.date_range(start,end,freq='15min',inclusive='left',tz=TZ),range(30)],names=['datetime','region'])
    result=s.reindex(index,fill_value=0).astype('int64').rename('total_pickups').reset_index()
    result.to_parquet(dest,index=False)
    save(OUT/(name+'_provenance.json'),dict(start=start,end_exclusive=end,days=len(dates),rows=len(result),sha256=sha(dest),geography_sha256=sha(OLD/'geography.json'),
        source_datasets=['2yzn-sicd'] if start.startswith('2015') else ['uacg-pexx'],filters=FILTERS,zeros='Preserved under source completeness assumption; raw outages cannot be certified',duplicate_policy='No overlapping day requests; no artificial deletion of coincident trips. Aggregated region/time rows are unique.',raw_trips_persisted=False))

def history():
    initialize();dest=OUT/'history_2015.parquet'
    if dest.exists():return
    dates=list(pd.date_range('2015-11-02','2015-11-30',freq='D'))+[pd.Timestamp('2015-12-'+str(d)) for d in [10,11,12,17,18,19,24,25,26]]
    geo=json.loads((OLD/'geography.json').read_text());folder=OUT/'daily/history_2015'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:chunks=list(pool.map(lambda d:day(d,geo,folder),dates))
    s=pd.concat(chunks).set_index(['datetime','region']).total_pickups
    if s.index.duplicated().any():raise ValueError('Duplicate source-day keys')
    times=pd.DatetimeIndex(np.concatenate([pd.date_range(d,d+pd.Timedelta('1D'),freq='15min',inclusive='left',tz=TZ).to_numpy() for d in dates]))
    index=pd.MultiIndex.from_product([times,range(30)],names=['datetime','region'])
    result=s.reindex(index,fill_value=0).astype('int64').rename('total_pickups').reset_index();result.to_parquet(dest,index=False)
    save(OUT/'history_2015_provenance.json',dict(dates=[str(d.date()) for d in dates],days=len(dates),rows=len(result),sha256=sha(dest),geography_sha256=sha(OLD/'geography.json'),filters=FILTERS,source='2yzn-sicd',gaps='Unacquired days remain absent, never zero-filled. All feature training excludes contexts crossing a gap.',raw_trips_persisted=False))

if __name__=='__main__':
    import sys
    if sys.argv[-1]=='history':history()
    elif sys.argv[-1]=='final':
        if not (OUT/'selection_freeze.json').exists():raise RuntimeError('Select and freeze models before final acquisition')
        period('2016-06-01','2016-07-01','june_sealed')
        if json.loads((OUT/'selection_freeze.json').read_text())['secondary_availability']['eligible']:period('2016-07-01','2016-07-08','july_event_sealed')
        else:print('Planned July event check unavailable under fixed mapping/target; June primary unchanged.',flush=True)
    else:raise SystemExit('Choose history or final')
