"""Source-compatibility probes only; no candidate forecast outcomes."""
import httpx,json
from .protocol import *
from .source_probe import FILTERS

def main():
    initialize();out={}
    with httpx.Client(timeout=90) as client:
        params={'$select':'tpep_pickup_datetime,pickup_longitude,pickup_latitude','$where':"tpep_pickup_datetime >= '2016-07-01T00:00:00' AND tpep_pickup_datetime < '2016-07-08T00:00:00' AND "+FILTERS,'$limit':1}
        response=client.get('https://data.cityofnewyork.us/resource/uacg-pexx.json',params=params);response.raise_for_status()
        out['july_coordinate_sample']=response.json();out['request']=str(response.url)
        out['note']='Availability/schema only, not forecasting outcomes; a nonempty row alone does not establish complete July coverage.'
    save(REPORT/'july_coordinate_feasibility.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
