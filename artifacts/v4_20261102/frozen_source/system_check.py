"""Actual localhost API validation of the separate selected review artifact."""
import json,os,pathlib,socket,subprocess,sys,time
import httpx,numpy as np,pandas as pd,psutil
from .protocol import *

def main():
    if not (OUT/'final_test_results.json').exists():raise RuntimeError('Complete frozen final evaluation first')
    env=os.environ.copy();env['URBANFLOW_MODEL']=str(OUT/'candidate_v4.joblib');env['URBANFLOW_SEED']=str(OUT/'inference_observations.parquet')
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    with open(REPORT/'uvicorn.log','w') as log:
        server=subprocess.Popen([sys.executable,'-m','uvicorn','advanced_main:app','--app-dir',str(ROOT/'urbanflow/backend'),'--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=90) as client:
                for _ in range(150):
                    try:
                        if client.get('/health').status_code==200:break
                    except httpx.ConnectError:pass
                    if server.poll() is not None:raise RuntimeError('Server exited; inspect log')
                    time.sleep(.2)
                else:raise RuntimeError('Server startup timeout')
                requests=[dict(region=r,datetime='2016-06-15T23:00:00-04:00',horizon=12) for r in range(30)]
                requests += [dict(region=15,datetime=t,horizon=h) for t in ['2016-06-15T23:15:00-04:00','2016-06-04T23:30:00-04:00','2016-06-30T23:45:00-04:00'] for h in [1,4,12,24,48]]
                records=[]
                for request in requests:
                    begun=time.perf_counter();response=client.post('/predict',json=request);ms=1000*(time.perf_counter()-begun)
                    assert response.status_code==200,(request,response.text)
                    data=response.json();p=data['forecast'];assert len(p)==request['horizon']
                    times=pd.to_datetime([v['datetime'] for v in p],utc=True)
                    assert times[0]==pd.Timestamp(request['datetime']).tz_convert('UTC')
                    assert times.to_series().diff().iloc[1:].eq(pd.Timedelta('15min')).all()
                    assert all(np.isfinite(v['prediction']) and v['prediction']>=0 for v in p)
                    assert pd.to_datetime([v['datetime'] for v in data['history']],utc=True).max()<times[0]
                    assert data['metadata']['schema_version']==3
                    records.append(dict(request=request,latency_ms=ms,forecast=p))
                for origin in ['2016-06-15T23:15:00-04:00','2016-06-04T23:30:00-04:00','2016-06-30T23:45:00-04:00']:
                    group=[r for r in records if r['request']['datetime']==origin];long=next(r['forecast'] for r in group if r['request']['horizon']==48)
                    for row in group:assert row['forecast']==long[:row['request']['horizon']]
                invalid=[dict(region=-1),dict(region=30),dict(horizon=0),dict(horizon=49),dict(datetime='bad'),dict(datetime='2026-11-02T00:00:00'),dict(datetime='2016-05-15T23:00:00'),dict(datetime='2016-06-15T23:01:00'),dict(datetime='2016-07-04T06:45:00-04:00'),dict(mode='realtime')]
                for change in invalid:
                    request=dict(region=0,datetime='2016-06-15T23:00:00-04:00',horizon=12);request.update(change)
                    assert client.post('/predict',json=request).status_code==422
                processes=[psutil.Process(server.pid)]+psutil.Process(server.pid).children(recursive=True)
                save(REPORT/'local_system_results.json',dict(valid_requests=len(records),invalid_requests=len(invalid),requests=records,health=client.get('/health').json(),sample=data,process_tree_RSS_MiB=sum(p.memory_info().rss for p in processes)/1024**2))
                print('Local HTTP:',len(records),'valid;',len(invalid),'invalid.',flush=True)
        finally:
            try:
                children=psutil.Process(server.pid).children(recursive=True)
                for child in reversed(children):
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
                _,alive=psutil.wait_procs(children,timeout=5)
                for child in alive:child.kill()
            except psutil.NoSuchProcess:pass
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()

if __name__=='__main__':main()
