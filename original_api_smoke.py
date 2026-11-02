"""Exercise preserved original artifact through the repaired default backend."""
import json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'urbanflow'),str(ROOT/'urbanflow/backend')]
from fastapi.testclient import TestClient
import main
with TestClient(main.app) as client:
    response=client.post('/predict',json=dict(region=15,datetime='2016-02-20T23:00:00-05:00',horizon=12))
    (ROOT/'reports/original_api_smoke.json').write_text(json.dumps(dict(status=response.status_code,body=response.json()),indent=2))
    assert response.status_code==200,response.text
    assert len(response.json()['forecast'])==12
    print('Preserved original model: repaired API request passed')
