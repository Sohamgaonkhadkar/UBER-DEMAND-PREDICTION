# Advanced local research

All phase-one models, reports, source files and notebooks remain intact. This
package writes only to `artifacts/advanced_20261102/` and new advanced reports.
The original production model is never replaced.

Set `PYTHONPATH` to the `urbanflow` directory when running modules from the parent
workspace. The completed stages, in order, are:

```powershell
$env:PYTHONPATH = (Resolve-Path urbanflow).Path
.\.venv\Scripts\python -m ml.advanced.diagnose
.\.venv\Scripts\python -m ml.advanced.fresh_data
.\.venv\Scripts\python -m ml.advanced.research features
.\.venv\Scripts\python -m ml.advanced.research tune
.\.venv\Scripts\python -m ml.advanced.research strategies
.\.venv\Scripts\python -m ml.advanced.exposure
.\.venv\Scripts\python -m ml.advanced.baselines
.\.venv\Scripts\python -m ml.advanced.research direct_tune
.\.venv\Scripts\python -m ml.advanced.alternatives
.\.venv\Scripts\python -m ml.advanced.calendar_anchoring
.\.venv\Scripts\python -m ml.advanced.final_evaluate
```

Completed experiments are reused. Selection is frozen before opening April's
forecast errors. Further experiments are rejected after the freeze; final test
evaluation is write-once. A new optimization phase needs a new output directory,
new protocol and a genuinely fresh test period. Do not tune using these April
results and then call April untouched again.

The historical geographic map was recovered from stored notebook centroids and
a streaming scaler refit from the original coordinates. It reproduces all 33,234,199 original labels.
April is fetched from the official NYC Open Data coordinate-level source, using
the original fixed filters. Daily response hashes and queries are retained;
no full duplicate raw trip dataset is stored. Search reads Q1 only.

Run the separate opt-in backend from the parent workspace:

```powershell
$env:URBANFLOW_MODEL = (Resolve-Path artifacts/advanced_20261102/advanced_candidate_v3.joblib).Path
$env:URBANFLOW_SEED = (Resolve-Path artifacts/advanced_20261102/inference_observations.parquet).Path
.\.venv\Scripts\python -m uvicorn advanced_main:app --app-dir urbanflow/backend --host 127.0.0.1 --port 8000
```

Use an April 2016 historical origin, e.g.:

```json
{"region":15,"datetime":"2016-04-15T23:00:00-04:00","horizon":48,"mode":"historical"}
```

`validation_demo_v3.joblib` is the earlier, Jan–Feb-fitted version for March/DST
demonstrations. The Q1-refitted candidate rejects origins before its training
cutoff, avoiding an in-sample retrospective forecast being treated as a backtest.
All 30 regions and 1,536 preceding consecutive observations are required by the
joint adapter. Time is America/New_York and horizons count elapsed 15-minute steps.

The API response fields remain compatible. Added metadata identifies the
candidate and forecast strategy. No live taxi feed, forecast probability bands,
or operational alerting is claimed. `main.py` continues to serve the previous
engine; its source/default artifacts were not edited. No deployment is performed.

The environment is the existing `.venv`; versions are in the phase-one lockfile.
This is a local research/serving layout. Container packaging and production data
ingestion are not validated by this phase.
