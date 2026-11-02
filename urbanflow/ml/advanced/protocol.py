import hashlib,json,pathlib
import numpy as np,pandas as pd

ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/advanced_20261102'
REPORT=ROOT/'reports/advanced_20261102'
TZ='America/New_York'
for p in [OUT,REPORT]:p.mkdir(exist_ok=True,parents=True)

def origins(start,end):
    """13h spacing: non-overlapping 12h trajectories, rotating local hours."""
    return pd.date_range(pd.Timestamp(start,tz=TZ),pd.Timestamp(end,tz=TZ)-pd.Timedelta(minutes=15*47),freq='13h')

def metrics(y,p):
    y,p=np.asarray(y,dtype=float).ravel(),np.asarray(p,dtype=float).ravel()
    if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Nonfinite evaluation values')
    e=p-y;den=np.abs(y).sum()
    return dict(WAPE=float(100*np.abs(e).sum()/den) if den else None,
        MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt(np.mean(e**2))),
        R2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)) if np.var(y)>0 else None,
        bias=float(e.mean()),bias_pct=float(100*e.sum()/den) if den else None)

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def initialize():
    target=OUT/'protocol.json'
    if target.exists():return json.loads(target.read_text())
    plan=dict(train_start='2016-01-01',train_end_exclusive='2016-03-01 00:00 America/New_York minus 2h label gap',
        validation_start='2016-03-01',validation_end_exclusive='2016-04-01',
        fresh_test_start='2016-04-01',fresh_test_end_exclusive='2016-05-01',
        validation_origins=[str(t) for t in origins('2016-03-01','2016-04-01')],
        test_origins=[str(t) for t in origins('2016-04-01','2016-05-01')],
        region_ids=list(range(30)),horizon=48,origin_spacing='13 elapsed hours; trajectories do not overlap',
        primary='100 * sum(abs(y-p)) / sum(abs(y)), all origins, regions and 48 leads',
        test_policy='April targets must not be accessed by search/diagnostics. Refit frozen selected specification on January-March and evaluate once with prespecified baselines.',
        integrity_note='March was already examined in phase one; it is DEVELOPMENT VALIDATION, never a new untouched chronological test.',
        bootstrap='Paired moving blocks of 3 chronological origins (~39 hours); 1000 draws, seed 42; dependencies longer than block remain a limit.',
        compute_budget='Four native threads; staged feature/objective screening, 12 LightGBM and 8 XGBoost Optuna trials initially; expand only with validation evidence.',
        old_hashes={str(p.relative_to(ROOT)):sha(p) for folder in [ROOT/'reports',ROOT/'artifacts/candidates',ROOT/'urbanflow/backend'] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'advanced_20261102' not in p.parts and not p.name.startswith('advanced')},
        source_panel_sha256=sha(ROOT/'artifacts/counts_ny.parquet'))
    target.write_text(json.dumps(plan,indent=2))
    return plan
