"""Read only the official July Parquet footer, never trip rows."""
import io,struct,json
import httpx,pyarrow.parquet as pq
from .protocol import *

def main():
    url='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2016-07.parquet'
    def ranged(spec):
        with httpx.Client(timeout=45) as client:
            with client.stream('GET',url,headers={'Range':spec}) as response:
                if response.status_code!=206:raise ValueError('Server did not honor bounded metadata Range request')
                content=b''
                for chunk in response.iter_bytes():
                    content+=chunk
                    if len(content)>2*1024**2:raise ValueError('Metadata response exceeds bound')
                return content
    footer=ranged('bytes=-8')
    if footer[-4:]!=b'PAR1':raise ValueError('Invalid Parquet footer')
    size=struct.unpack('<I',footer[:4])[0];payload=ranged('bytes=-'+str(size+8))
    fields=pq.read_schema(io.BytesIO(b'PAR1'+payload)).names
    out=dict(url=url,fields=fields,bytes_read=len(payload)+8,trip_rows_read=0,has_coordinates='pickup_longitude' in fields or 'Pickup_longitude' in fields,has_taxi_zone_ids='PULocationID' in fields,
        scope='Schema-only feasibility; no June/July target outcomes inspected. Zone IDs alone cannot exactly reproduce fixed point-level clusters or original pickup/dropoff geographic filters.')
    save(REPORT/'official_july_schema.json',out);print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
