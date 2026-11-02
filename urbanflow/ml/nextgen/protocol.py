import hashlib,json,pathlib
import numpy as np,pandas as pd
from ml.advanced.protocol import metrics,TZ

ROOT=pathlib.Path(__file__).resolve().parents[3]
OLD=ROOT/'artifacts/advanced_20261102'
OUT=ROOT/'artifacts/nextgen_20261102'
REPORT=ROOT/'reports/nextgen_20261102'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def origins(start,end):
    return pd.date_range(pd.Timestamp(start,tz=TZ),pd.Timestamp(end,tz=TZ)-pd.Timedelta(minutes=47*15),freq='13h')

def initialize():
    for p in [OUT,REPORT]:p.mkdir(exist_ok=True,parents=True)
    file=OUT/'protocol.json'
    if file.exists():return json.loads(file.read_text())
    protected=[]
    for folder in [ROOT/'urbanflow/backend',ROOT/'urbanflow/ml/advanced',ROOT/'reports',ROOT/'artifacts/candidates',OLD]:
        protected.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'nextgen_20261102' not in p.parts and not p.name.startswith('nextgen_'))
    protected.extend(ROOT.glob('*.ipynb'));protected.extend(ROOT.glob('*.py'))
    plan=dict(primary='100 * sum(abs(actual-prediction)) / sum(abs(actual)), all origins/30 regions/all 48 leads',
        old_April_status='Previously held out, now diagnostic-only. No gate/correction coefficients or weight selection use April forecast errors.',
        calibration=dict(OOF_training_end='2016-02-14 22:00 NY exclusive',origins=[str(t) for t in origins('2016-02-15','2016-03-01')]),
        development_validation=dict(model_fit_end='2016-02-29 22:00 NY exclusive',period='March 2016',origins=[str(t) for t in origins('2016-03-01','2016-04-01')]),
        final_refit=dict(base_and_profiles='Jan-April labels before April 30 22:00 NY',gate_coefficients='March OOF forecasts from a Jan-Feb fitted base, never April forecast residuals'),
        final_test=dict(period='May 2016, fresh chronological holdout',origins=[str(t) for t in origins('2016-05-01','2016-06-01')],region_ids=list(range(30)),horizon=48),
        experiment_families=[
            dict(id='E1',hypothesis='Stale calendar levels and incorrect within-weekday midnight smoothing weaken seasonal anchoring',variants=['weekly_boundary_only','recent_shrunk_weighted_median'],half_life_days=28,prior_effective_weeks=4),
            dict(id='E2',hypothesis='A strongly regularized origin-context gate can trust calendar profiles less in unreliable conditions',variants=['learned_context_gate'],coefficient_fit='Feb OOF only for development; March OOF only for final refit'),
            dict(id='E3',hypothesis='A learned residual correction applied after the path avoids the previous in-loop amplification',variants=['post_path_ridge'],strategy_label='recursive base with post-path calibration; corrected outputs are NOT fed back'),
            dict(id='E4',hypothesis='Adding April training history may help more than architecture complexity',comparison='Frozen Q1 current candidate versus identical recipe refit Jan-April on the same May test')],
        candidate_selection='Novel recipe must gain >=0.1 WAPE point on March, paired 3-origin block 95% gain interval strictly positive, improve >=20/30 regional WAPEs, and worsen no region by >2 WAPE points. Otherwise retain current architecture. Prefer genuine recursion if within 0.1 point and paired uncertainty unresolved.',
        final_comparators=['frozen_Q1_current','matched_refit_current','matched_refit_v2_xgboost','matched_refit_v2_lightgbm','clock_weekly_naive','selected_new_if_March_gate_passes'],
        bootstrap='1000 paired circular moving blocks of 3 chronological origins, seed 42; entire 30-region/48-lead trajectories retained',
        compute_budget='Four native threads, two public-source requests in flight; four research families, four novel configurations total; no Optuna/random search.',
        resource_estimate=dict(available_RAM_GiB=1.90,free_disk_GiB=69.45,source_expected_trips_M=12,retained_aggregate_rows=31*96*30,expected_request_count=31,raw_download_saved=False,expected_extra_disk_GiB_lt=1),
        data_policy='Same fixed 30-cluster map, source geography/fare/distance filters and genuine zero target. May schema was probed for availability only, not forecasting outcomes.',
        created_at=pd.Timestamp.now(tz='UTC').isoformat(),protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in set(protected)})
    file.write_text(json.dumps(plan,indent=2));return plan

def read_development():
    # May never enters model fitting, calibration or selection.
    return pd.read_parquet(OLD/'inference_observations.parquet')
