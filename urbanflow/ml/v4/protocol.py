import hashlib,json,pathlib
import numpy as np,pandas as pd
from ml.advanced.protocol import metrics,TZ,origins
ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/v4_20261102'
REPORT=ROOT/'reports/v4_20261102'
PREVIOUS=ROOT/'artifacts/nextgen_20261102'
OLD=ROOT/'artifacts/advanced_20261102'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(path,data):path.write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')

def initialize():
    OUT.mkdir(exist_ok=True,parents=True);REPORT.mkdir(exist_ok=True,parents=True)
    dest=OUT/'protocol.json'
    if dest.exists():return json.loads(dest.read_text())
    protected=[]
    for folder in [ROOT/'reports',ROOT/'urbanflow/backend',ROOT/'urbanflow/ml',ROOT/'urbanflow/tests',OLD,PREVIOUS,ROOT/'artifacts/candidates']:
        protected.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'v4_20261102' not in p.parts and 'v4' not in p.parts)
    protected.extend(p for p in ROOT.iterdir() if p.is_file())
    plan=dict(created_at=pd.Timestamp.now(tz='UTC').isoformat(),
        status='March/April/May are spent development evidence; June and July event window are unexamined at protocol creation.',
        development=[dict(month=m,cutoff=str(pd.Timestamp(m+'-01',tz=TZ)-pd.Timedelta('2h'))) for m in ['2016-03','2016-04','2016-05']],
        selection='Pool absolute errors and denominators across March/April/May, equal observations not equal month weights; all 13h origins, all 30 regions and 48 leads. Retain simplest strongest baseline unless novel gain >=0.1 WAPE point, paired 3-origin within-month block 95% gain lower>0, improves >=2/3 months and >=20 regions; no region worsens >2 points.',
        final_train_end='2016-05-31 22:00 America/New_York exclusive',
        primary_test=dict(period='2016-06-01 through 2016-06-30',origins=[str(t) for t in origins('2016-06-01','2016-07-01')],reason='First untouched month. No US federal holiday in June 2016; cannot validate holiday gains here.'),
        secondary_event_test=dict(period='2016-07-01 through 2016-07-07',origins=[str(t) for t in origins('2016-07-01','2016-07-08')],reason='Prespecified July 4 confirmation; same May-cutoff model, no refit/selection using June. Separate from primary; not a replacement if June fails.'),
        primary='100 * sum(abs(actual-prediction)) / sum(abs(actual)), original 15-minute regional counts, every frozen origin/region/48 lead',
        feedback='Recursive strategies append their own final outputs; hourly models recurse in coarse count space. Direct disaggregation/reconciliation explicitly distinguished; never call it 48-step feedback recursion.',
        data='Unchanged filters and original fixed scaler/centers. No future weather, flights or event attendance. Source feasibility first; bounded acquisition, reuse existing Jan-May panels.',
        families=['holiday representation with extra historical events and fixed calendar safety rule','30/60-minute origin-relative temporal totals and coherent reconciliation, optional city constraint','expanding versus rolling versus decayed training under identical fixed tree capacity and disjoint historical coverage','sparse-region negative-binomial conditional-median specialist, global aggregate acceptance gate'],
        budget='Four native threads; no Optuna, no neural/GNN until independent added information demonstrated; cache features, fixed variants, two source requests at once.',
        historical_feasibility='Try exact server-side Voronoi aggregation to avoid bulk trip downloads. If supported, acquire Jan-Feb and Nov-Dec 2015, enough chronological segments for window comparisons and more distinct holiday events. Verify against an existing day before use. Otherwise bound raw acquisition and record feasibility limitation.',
        uncertainty='1000 paired circular moving blocks of 3 origins, seed42; development resampled within month; entire regional paths retained. Intervals exploratory after bounded selection.',
        no_final_tuning=True,protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in set(protected)})
    save(dest,plan);return plan

def panel():return pd.read_parquet(PREVIOUS/'inference_observations.parquet')

def development_only():
    if (OUT/'selection_freeze.json').exists():raise RuntimeError('Selection frozen; no more development fitting')

if __name__=='__main__':print('Protocol frozen:',initialize()['created_at'])
