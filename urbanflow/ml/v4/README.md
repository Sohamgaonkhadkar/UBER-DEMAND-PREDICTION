# Research-driven forecasting v4 (local only)

New code and tests are isolated here. Data/models/tracking are under
`artifacts/v4_20261102`; diagnostic evidence is under `reports/v4_20261102`.
Earlier packages, checkpoints, reports, default backend and frontend remain
unchanged. The user-requested final report is
`reports/research_driven_forecasting_v4.md`.

From the workspace root, use the existing virtual environment:

```powershell
$env:PYTHONPATH = (Resolve-Path urbanflow).Path
.\.venv\Scripts\python -m ml.v4.protocol
.\.venv\Scripts\python -m ml.v4.diagnose
.\.venv\Scripts\python -m ml.v4.acquire history
.\.venv\Scripts\python -m ml.v4.research preliminary
.\.venv\Scripts\python -m ml.v4.research
.\.venv\Scripts\python -m ml.v4.acquire final
.\.venv\Scripts\python -m ml.v4.final_evaluate
.\.venv\Scripts\python -m ml.v4.system_check
.\.venv\Scripts\python -m pytest urbanflow/tests urbanflow/ml/v4/tests -q
```

The preliminary command runs only independent Jan-only arms while historical
acquisition completes; full development reuses those checkpoints. It cannot
select a model. Model selection uses March/April/May, all already inspected.
The full development command freezes code/selection before final acquisition.
Do not tune or rerun model selection after opening final outcomes. June is the
primary test; July 1–7 is a separate prespecified holiday confirmation window,
using the same May-cutoff fits. The July check was not executed: verified
official July schema has zone IDs instead of coordinates, and complete
compatible coordinate coverage was not established. June remains the primary
test; no replacement period was chosen. Results cannot be relabeled untouched in a
later phase. Completed checkpoints/daily downloads are reused.

Optional localhost review after reading results:

```powershell
$env:URBANFLOW_MODEL = (Resolve-Path artifacts/v4_20261102/candidate_v4.joblib).Path
$env:URBANFLOW_SEED = (Resolve-Path artifacts/v4_20261102/inference_observations.parquet).Path
.\.venv\Scripts\python -m uvicorn advanced_main:app --app-dir urbanflow/backend --host 127.0.0.1 --port 8000
```

This is an opt-in review artifact, not a production replacement. Requests need
all 30 regions and 1,536 genuine preceding quarter-hour observations. No live
feed, invented future weather, future arrivals or prediction intervals are
provided. Quarterly count conservation is tested before API rounding; rounded
two-decimal API values may differ slightly from exact aggregate constraints.
