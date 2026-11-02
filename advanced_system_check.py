"""Actual localhost HTTP checks for the frozen advanced candidate and demo."""
import json,os,pathlib,socket,subprocess,sys,time
import httpx,numpy as np,pandas as pd,psutil
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'artifacts/advanced_20261102';REPORT=ROOT/'reports/advanced_20261102'

def run_server(model,requests,name):
    env=os.environ.copy();env['URBANFLOW_MODEL']=str(model);env['URBANFLOW_SEED']=str(OUT/'inference_observations.parquet')
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    records=[];sample=None
    with open(REPORT/f'uvicorn_{name}.log','w') as log:
        server=subprocess.Popen([sys.executable,'-m','uvicorn','advanced_main:app','--app-dir',str(ROOT/'urbanflow/backend'),'--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=60) as client:
                for _ in range(100):
                    try:
                        if client.get('/health').status_code==200:break
                    except httpx.ConnectError:pass
                    if server.poll() is not None:raise RuntimeError('Uvicorn failed; see log')
                    time.sleep(.2)
                else:raise RuntimeError('Uvicorn startup timeout')
                for req in requests:
                    before=time.perf_counter();response=client.post('/predict',json=req);duration=1000*(time.perf_counter()-before)
                    assert response.status_code==200,(req,response.text)
                    d=response.json();sample=d;p=d['forecast'];assert len(p)==req['horizon']
                    ts=pd.to_datetime([v['datetime'] for v in p],utc=True)
                    assert (ts.to_series().diff().iloc[1:]==pd.Timedelta('15min')).all()
                    assert all(np.isfinite(v['prediction']) and v['prediction']>=0 for v in p)
                    assert set(['forecast','history','weather','heatmap','feature_importance','alerts'])<=d.keys()
                    assert d['metadata']['schema_version']==3
                    assert d['metadata']['forecast_strategy'] in ['recursive','direct','blend','hybrid','seasonal']
                    assert pd.to_datetime([v['datetime'] for v in d['history']],utc=True).max()<ts[0]
                    records.append(dict(request=req,latency_ms=duration,predictions=p))
                if name=='final':
                    invalid=[dict(region=30),dict(region=-1),dict(horizon=0),dict(horizon=49),dict(datetime='bad'),dict(datetime='2026-11-02T00:00:00'),dict(datetime='2016-03-13T02:15:00'),dict(datetime='2016-03-01T00:00:00'),dict(mode='realtime')]
                    for change in invalid:
                        req=dict(region=0,datetime='2016-04-15T23:00:00-04:00',horizon=12);req.update(change)
                        assert client.post('/predict',json=req).status_code==422,req
                    # Identical prefixes across short/long horizon calls.
                    cases=[a for a in records if a['request']['region']==15 and a['request']['datetime']=='2016-04-15T23:00:00-04:00']
                    long=next(a['predictions'] for a in cases if a['request']['horizon']==48)
                    for case in cases:assert case['predictions']==long[:case['request']['horizon']]
                else:
                    long=next(a['predictions'] for a in records if a['request']['horizon']==48)
                    assert any('-04:00' in a['datetime'] for a in long)
                    assert all('2016-03-13T02:' not in a['datetime'] for a in long)
                rss=sum(p.memory_info().rss for p in [psutil.Process(server.pid)]+psutil.Process(server.pid).children(recursive=True))
                return dict(server_model=str(model),health=client.get('/health').json(),valid_requests=len(records),invalid_requests=9 if name=='final' else 0,
                    requests=records,sample=sample,whole_backend_process_tree_RSS_MiB=rss/1024**2)
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

def main():
    if not (OUT/'final_test_results.json').exists():raise RuntimeError('Finish frozen final evaluation first')
    requests=[dict(region=r,datetime='2016-04-15T23:00:00-04:00',horizon=12) for r in range(30)]
    requests += [dict(region=15,datetime=ts,horizon=h) for ts in ['2016-04-15T23:00:00-04:00','2016-04-02T23:00:00-04:00','2016-04-24T06:00:00-04:00'] for h in [1,4,12,24,48]]
    final=run_server(OUT/'advanced_candidate_v3.joblib',requests,'final')
    demo=run_server(OUT/'validation_demo_v3.joblib',[dict(region=r,datetime='2016-03-12T23:00:00-05:00',horizon=48) for r in [0,15,29]],'validation_demo')
    (REPORT/'local_system_results.json').write_text(json.dumps(dict(final=final,validation_demo=demo),indent=2))
    print('Actual HTTP checks passed:',final['valid_requests']+demo['valid_requests'],'valid;',final['invalid_requests'],'invalid')

if __name__=='__main__':main()
