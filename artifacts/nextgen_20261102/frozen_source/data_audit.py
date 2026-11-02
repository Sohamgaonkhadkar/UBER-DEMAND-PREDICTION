"""Quantify target exclusions on old January data, not fresh test targets."""
import json
import numpy as np,pandas as pd
from sklearn.metrics import pairwise_distances_argmin
from threadpoolctl import threadpool_limits
from .protocol import *

def main():
    initialize();geo=json.loads((OLD/'geography.json').read_text())
    total=0;counts={k:0 for k in ['pickup_bounds','dropoff_bounds','fare','distance','retained']}
    airport={r:dict(pickups_in_geography=0,retained=0,excluded_dropoff=0,excluded_fare=0,excluded_distance=0) for r in [3,6]}
    cols=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','fare_amount','trip_distance']
    for c in pd.read_csv(ROOT/'data/yellow_tripdata_2016-01.csv',usecols=cols,chunksize=150_000):
        pickup=c.pickup_latitude.between(40.60,40.85)&c.pickup_longitude.between(-74.05,-73.70)
        drop=c.dropoff_latitude.between(40.60,40.85)&c.dropoff_longitude.between(-74.05,-73.70)
        fare=c.fare_amount.between(.5,81);distance=c.trip_distance.between(.25,24.43)
        retained=pickup&drop&fare&distance;total+=len(c)
        for key,mask in [('pickup_bounds',pickup),('dropoff_bounds',drop),('fare',fare),('distance',distance),('retained',retained)]:counts[key]+=int(mask.sum())
        part=c.loc[pickup];xy=part[['pickup_longitude','pickup_latitude']].to_numpy()
        with threadpool_limits(limits=2):labels=pairwise_distances_argmin((xy-geo['mean'])/geo['scale'],np.array(geo['centers']))
        for r in [3,6]:
            mask=labels==r;idx=part.index[mask]
            airport[r]['pickups_in_geography']+=int(mask.sum());airport[r]['retained']+=int(retained.loc[idx].sum())
            airport[r]['excluded_dropoff']+=int((~drop.loc[idx]).sum());airport[r]['excluded_fare']+=int((~fare.loc[idx]).sum());airport[r]['excluded_distance']+=int((~distance.loc[idx]).sum())
    quality={}
    panel=read_development()
    for month,g in panel.groupby(panel.datetime.dt.month):
        wide=g.pivot(index='datetime',columns='region',values='total_pickups')
        quality[str(month)]=dict(rows=len(g),duplicates=int(g.duplicated(['datetime','region']).sum()),zero_intervals=int(g.total_pickups.eq(0).sum()),nonfinite=int((~np.isfinite(g.total_pickups)).sum()),elapsed_15min_continuity=bool(wide.index.to_series().diff().iloc[1:].eq(pd.Timedelta('15min')).all()),regions=len(wide.columns),mean_demand=float(g.total_pickups.mean()))
    result=dict(January_raw_rows=total,marginal_filter_pass_counts=counts,airport_filter_impact=airport,existing_panel_quality=quality,
        caveats=['Filter exclusion reasons overlap; do not add marginal counts.','Fare/distance/dropoff filters exclude genuine trips, including airport journeys; target is retained trips, not all pickup requests.','Zero counts cannot certify absence of source outages.','Original March DST quarantine remains preserved; no target changes in this phase.'])
    (REPORT/'data_quality_and_target_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
