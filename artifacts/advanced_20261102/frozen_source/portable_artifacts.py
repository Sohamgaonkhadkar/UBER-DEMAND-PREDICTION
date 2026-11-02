"""Preserve initial wrapper artifacts; create importable canonical copies."""
import json,sys
import joblib,numpy as np,pandas as pd
from .protocol import *
from .alternatives import CalendarProfile,SeasonalForecast,CompositeForecast
from .baselines import V2Forecast

def main():
    for cls in [CalendarProfile,SeasonalForecast,CompositeForecast,V2Forecast]:setattr(sys.modules['__main__'],cls.__name__,cls)
    records=json.loads((OUT/'leaderboard.json').read_text());changes=[]
    panel=pd.read_parquet(ROOT/'artifacts/counts_ny.parquet')
    wide=panel.pivot(index='datetime',columns='region',values='total_pickups')
    oo=origins('2016-03-01','2016-04-01');pos=wide.index.get_indexer(oo)
    for record in records:
        if not record.get('artifact') or record['artifact'].endswith('_portable.joblib'):continue
        path=ROOT/record['artifact'];bundle=joblib.load(path)
        if not isinstance(bundle,(CalendarProfile,SeasonalForecast,CompositeForecast,V2Forecast)):continue
        dest=path.with_name(path.stem+'_portable.joblib')
        if dest.exists():raise RuntimeError('Refusing to overwrite a portable candidate')
        joblib.dump(bundle,dest)
        # Every prediction is checked against the already saved validation result.
        p=bundle.forecast(wide.to_numpy(dtype=np.float32),wide.index,pos,48)
        old=np.load(OUT/'validation_predictions'/f"{record['id']}.npz")['prediction']
        np.testing.assert_allclose(p,old,rtol=1e-10,atol=1e-10)
        previous=record['artifact'];record['initial_artifact_preserved']=previous
        record['artifact']=str(dest.relative_to(ROOT));record['model_bytes']=dest.stat().st_size
        if isinstance(bundle,V2Forecast):record['fitted_params']=bundle.model.get_params()
        changes.append(dict(id=record['id'],preserved=previous,portable=record['artifact'],original_sha256=sha(path),portable_sha256=sha(dest),all_validation_predictions_unchanged=True))
    (OUT/'leaderboard.json').write_text(json.dumps(records,indent=2,default=str))
    (REPORT/'artifact_portability_repair.json').write_text(json.dumps(changes,indent=2))
    print('Canonical artifact copies verified:',len(changes))

if __name__=='__main__':main()
