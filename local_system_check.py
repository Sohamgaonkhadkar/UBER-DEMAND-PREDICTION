"""Start an actual loopback Uvicorn process, test trained candidates, stop it."""
import json, os, pathlib, socket, subprocess, sys, time
import httpx,numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent

def main():
    env=os.environ.copy()
    env['URBANFLOW_MODEL']=str(ROOT/'artifacts/candidates/xgboost_v2.joblib')
    env['URBANFLOW_SEED']=str(ROOT/'artifacts/counts_ny.parquet')
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    with open(ROOT/'reports/uvicorn.log','w') as log:
        server=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--app-dir',str(ROOT/'urbanflow/backend'),'--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            base=f'http://127.0.0.1:{port}'
            with httpx.Client(base_url=base,timeout=60) as client:
                for _ in range(100):
                    try:
                        if client.get('/health').status_code==200: break
                    except httpx.ConnectError: pass
                    time.sleep(.2)
                else: raise RuntimeError('Local Uvicorn did not start')
                results=[]
                cases=[dict(region=r,datetime='2016-03-05T23:00:00-05:00',horizon=12) for r in range(30)]
                cases += [dict(region=15,datetime=ts,horizon=h) for ts in ['2016-03-01T00:00:00-05:00','2016-03-12T23:00:00-05:00','2016-03-18T23:00:00-04:00'] for h in [1,4,24,48]]
                for req in cases:
                    started=time.perf_counter();response=client.post('/predict',json=req);latency=1000*(time.perf_counter()-started)
                    assert response.status_code==200,(req,response.text)
                    d=response.json();f=d['forecast']
                    assert len(f)==req['horizon']
                    assert all(np.isfinite(p['prediction']) and p['prediction']>=0 for p in f)
                    ts=pd.to_datetime([p['datetime'] for p in f],utc=True)
                    assert (ts.to_series().diff().iloc[1:]==pd.Timedelta('15min')).all()
                    assert pd.to_datetime([p['datetime'] for p in d['history']],utc=True).max()<ts[0]
                    results.append(dict(request=req,status=response.status_code,latency_ms=latency))
                invalid=[dict(region=30),dict(region=-1),dict(horizon=0),dict(horizon=49),dict(datetime='bad'),dict(datetime='2026-11-02T00:00:00'),dict(datetime='2016-01-01T00:00:00'),dict(datetime='2016-03-13T02:15:00')]
                for change in invalid:
                    req=dict(region=0,datetime='2016-03-05T23:00:00',horizon=12);req.update(change)
                    response=client.post('/predict',json=req);assert response.status_code==422,(req,response.text)
                (ROOT/'reports/local_system_results.json').write_text(json.dumps(dict(health=client.get('/health').json(),successful_requests=results,invalid_requests_passed=len(invalid),sample=d),indent=2))
                print('HTTP checks passed:',len(results),'valid;',len(invalid),'invalid',flush=True)
        finally:
            server.terminate()
            try: server.wait(timeout=10)
            except subprocess.TimeoutExpired: server.kill();server.wait()

if __name__=='__main__': main()
