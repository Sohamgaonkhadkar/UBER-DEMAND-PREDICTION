# Bounded next-generation research

Earlier code, models and reports are unchanged. Artifacts are isolated under
`artifacts/nextgen_20261102/`; diagnostics are in `reports/nextgen_20261102/`.

The read-only diagnosis came first. Four families were selected: seasonal
profile construction (two fixed ablations), a contextual calendar gate, residual
calibration outside the recursive loop, and one extra month of training data.
No Optuna search was performed. All novel architecture changes failed the
prespecified March promotion gate; the selected recipe is the existing one.

April forecast errors were used for diagnosis only, never to fit or select new
gate/correction weights. Development coefficients use February OOF predictions.
The May evaluation was frozen before predictive access to May. Models were fit
on permitted January–April data. May is now spent; do not tune this run against
it or call it untouched again.

Modules run from the parent workspace with `PYTHONPATH` set to `urbanflow`:

```powershell
$env:PYTHONPATH = (Resolve-Path urbanflow).Path
.\.venv\Scripts\python -m ml.nextgen.diagnose
.\.venv\Scripts\python -m ml.nextgen.data_audit
.\.venv\Scripts\python -m ml.nextgen.fresh_may
.\.venv\Scripts\python -m ml.nextgen.research
.\.venv\Scripts\python -m ml.nextgen.final_evaluate
```

Completed results are preserved. The final evaluator refuses to repeat a spent
test; training and new research are guarded after selection/test freeze.

For explicit local review of the separate January–April data-refit checkpoint:

```powershell
$env:URBANFLOW_MODEL = (Resolve-Path artifacts/nextgen_20261102/nextgen_candidate.joblib).Path
$env:URBANFLOW_SEED = (Resolve-Path artifacts/nextgen_20261102/inference_observations.parquet).Path
.\.venv\Scripts\python -m uvicorn advanced_main:app --app-dir urbanflow/backend --host 127.0.0.1 --port 8000
```

Example historical origin: `2016-05-15T23:00:00-04:00`. Horizon counts 15-minute
steps, default 12, maximum 48. Earlier origins before the fitting cutoff are
rejected. No live observations are invented. All 30 regions and 16 days of
preceding context are required. The existing default application/model is not
switched, and nothing is deployed.

`nextgen_candidate.joblib` is a data refit of the current architecture, not a
successful novel-architecture experiment. Consult the report and paired May
comparisons before adoption. Old checkpoints are all preserved.
