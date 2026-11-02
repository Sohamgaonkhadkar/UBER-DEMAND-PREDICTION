"""Serve the separate data-refit checkpoint locally, verify contract, stop it."""
import json,os,pathlib,socket,subprocess,sys,time
import httpx,numpy as np,pandas as pd,psutil
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'artifacts/nextgen_20261102';REPORT=ROOT/'reports/nextgen_20261102'

def main():
    if not (OUT/'final_test_results.json').exists():raise RuntimeError('Complete frozen May evaluation first')
    env=os.environ.copy();env['URBANFLOW_MODEL']=str(OUT/'nextgen_candidate.joblib');env['URBANFLOW_SEED']=str(OUT/'inference_observations.parquet')
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    with open(REPORT/'uvicorn.log','w') as log:
        server=subprocess.Popen([sys.executable,'-m','uvicorn','advanced_main:app','--app-dir',str(ROOT/'urbanflow/backend'),'--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=60) as client:
                for _ in range(100):
                    try:
                        if client.get('/health').status_code==200:break
                    except httpx.ConnectError:pass
                    if server.poll() is not None:raise RuntimeError('Local server failed; see log')
                    time.sleep(.2)
                else:raise RuntimeError('Server startup timeout')
                cases=[dict(region=r,datetime='2016-05-15T23:00:00-04:00',horizon=12) for r in range(30)]
                cases += [dict(region=15,datetime=t,horizon=h) for t in ['2016-05-15T23:00:00-04:00','2016-05-07T23:00:00-04:00','2016-05-30T06:00:00-04:00'] for h in [1,4,12,24,48]]
                records=[]
                for request in cases:
                    started=time.perf_counter();response=client.post('/predict',json=request);ms=1000*(time.perf_counter()-started)
                    assert response.status_code==200,(request,response.text)
                    d=response.json();p=d['forecast'];assert len(p)==request['horizon']
                    ts=pd.to_datetime([v['datetime'] for v in p],utc=True)
                    assert (ts.to_series().diff().iloc[1:]==pd.Timedelta('15min')).all()
                    assert all(np.isfinite(v['prediction']) and v['prediction']>=0 for v in p)
                    assert pd.to_datetime([v['datetime'] for v in d['history']],utc=True).max()<ts[0]
                    assert set(['forecast','history','weather','heatmap','feature_importance','alerts'])<=d.keys()
                    assert d['metadata']['schema_version']==3
                    records.append(dict(request=request,latency_ms=ms,forecast=p))
                prefix=[v for v in records if v['request']['region']==15 and v['request']['datetime']=='2016-05-15T23:00:00-04:00']
                long=next(v['forecast'] for v in prefix if v['request']['horizon']==48)
                for case in prefix:assert case['forecast']==long[:case['request']['horizon']]
                invalid=[dict(region=-1),dict(region=30),dict(horizon=0),dict(horizon=49),dict(datetime='bad'),dict(datetime='2026-11-02T00:00:00'),dict(datetime='2016-04-15T23:00:00'),dict(datetime='2016-05-15T23:01:00'),dict(mode='realtime')]
                for change in invalid:
                    request=dict(region=0,datetime='2016-05-15T23:00:00-04:00',horizon=12);request.update(change)
                    assert client.post('/predict',json=request).status_code==422,request
                processes=[psutil.Process(server.pid)]+psutil.Process(server.pid).children(recursive=True)
                result=dict(valid_requests=len(records),invalid_requests=len(invalid),requests=records,health=client.get('/health').json(),sample=d,
                    process_tree_RSS_MiB=sum(p.memory_info().rss for p in processes)/1024**2)
                (REPORT/'local_system_results.json').write_text(json.dumps(result,indent=2))
                print('Local HTTP passed:',len(records),'valid;',len(invalid),'invalid.')
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
