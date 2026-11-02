# UrbanFlow backend and ML repair audit

All changes are local. No commits, pushes, pull requests, remote-branch changes,
deployments, production-model replacement, or original-data deletion occurred.

## Repositories and preserved work

The existing experimentation repository is the workspace root, branch `main`,
HEAD `4a72ca8bd22cc5e1ea58ad8cd98173e1425dc920`. Its pre-existing modified
`Breaking_NYC_to_Regions.ipynb`, `final model.ipynb`, `lightgbm.ipynb`, and untracked
`output.png` were preserved without edits. Other original notebooks and CSVs
were also left intact. The `.gitignore` changes append output exclusions.

The requested public production repository was cloned into `urbanflow/`, branch
`main`, baseline HEAD `76b1840c070e1f8f9589e7c96c18e6f9a1acc258`. An initially found
clean sibling checkout pointed to a differently named remote; it was copied to
`production/` but was not used as the authoritative repair target. Neither the
sibling checkout nor remote repositories were changed. The workspace root is
already the requested experimentation repository, so a second copy was unnecessary.

## Baseline evidence

`reports/baseline/original.json` records original outputs for regions 0, 15 and
29, seed range, feature order, categorical columns/labels, source dtypes,
parameters, resources and SHA-256 hashes. The original inference source files
are copied alongside it. `original_pipeline.joblib` is a separate baseline copy.
The actual production and experimentation artifacts have identical SHA-256:
`1cbff5cf3be0d51053540400b90212ef856ae3df9a84ad074c3c240f183c5a45`.

The artifact has **77 features**, including four categorical columns: `region`,
`holiday_name`, `weather_condition`, `temp_bin`. All category label lists are
saved. The binary does not retain every original Pandas numeric dtype, so CSV
read dtypes must not be represented as recovered training dtypes. New schemas
explicitly define float64 inputs and reject missing required inputs.

The original seed is January 31–February 29, 2016: 86,400 rows, 30 regions.
Its timestamps have not been shifted in the checked-in artifact. The shifting
script was dangerous, but evidence does not show it had been applied to this
copy. Baseline forecasts used embedded historical weather; missing future
weather became zeros and the API returned an invented 15°C fallback. These
outputs reproduce the old behavior, not a live-weather validation.

The original one-step benchmark was reproduced with the actual stored notebook
feature function and original `data/test_new.csv`: MAE 12.4143, RMSE 20.0520,
WAPE 9.4200%, R² 0.97644. Training was January–February; test was March. The
original notebook metrics are teacher-forced one-step predictions. Original
recursive sensitivity outputs and limitations are in the comparison reports.
The matching executable training path is `lightgbm.ipynb` cell 0 loading
`data/train_new.csv` / `data/test_new.csv`, cell 9’s final feature builder and
LightGBM training/tuning/evaluation, cell 10 serializing the deployment dictionary,
and cell 11 saving the final 30 days of engineered training data as seed memory.

## Findings, repairs and verification

| Issue / affected original file | Root cause | Implemented repair | Verification / limit |
| --- | --- | --- | --- |
| Missing features, `backend/feature_engineering.py` | Required columns silently filled with zero, including holiday/rolling/event/interactions | Explicit 77-column legacy reconstruction in `ml/legacy.py`; unknown columns and missing values raise; new pipeline shares `ml/features.py` | Exact trained order checked at startup; parity test at rtol 1e-10, atol 1e-8 for new schema |
| Legacy semantic mismatch, LightGBM notebook | Global lags/rolling use region-major rows; grouped columns only partly correct; early imputation uses whole-period means | Retrained V2 computes every demand feature per region, only from preceding observations; legacy adapter marked changed semantics | Synthetic isolation/causality tests pass; original trained weights cannot be repaired in place |
| Historical EWMA, `Creating-Historical-Data.ipynb` | Smoothing runs across region boundaries and alpha was assessed with same-step truth | New grouped EWMA uses alpha .4, adjust=False, target shifted before feature creation | Online/batch agreement and perturbation tests pass; original notebooks retained as evidence |
| Zero→10 targets, historical notebook | Epsilon workaround changes real observations | Reaggregate original regional trips, preserve zero counts; WAPE/MAE/RMSE/R² | Final panel retains 3,570 zeros after DST quarantine |
| Lag descriptions, notebooks/README | 15-minute steps called hours/days/weeks | Explicit durations documented; add lag_96 and lag_672 only in retrained schema | Numeric synthetic tests prove daily/weekly offsets; no rename under old weights |
| Forecast chronology, `services/forecasting_service.py` | Retains rows at/after origin; `.loc` overwrites observations; gaps ignored | Isolate request context strictly before origin; require continuous index and previous observation; recursively append predictions in order | Future-target perturbation and no-mutation tests; actual HTTP sequence checks |
| Date shifting, `shift_data.py` | Import/execution rewrites original seed into present | Disable shifting; historical/realtime modes; recent genuine observations required for realtime | Unsupported modern dates and empty contexts return 422 |
| DST and timezone, data/API | Naive timestamps lose offset semantics; source has 34 impossible local-clock trips | Separate timezone-aware derived panel, quarantine 120 region/interval rows containing those 34 trips; no source reassignment | Real HTTP forecast crosses spring DST with 15-minute UTC spacing; ambiguous/nonexistent naive origins rejected; legacy contexts/horizons crossing DST rejected |
| Time-series CV, model notebooks | Row-based split mixes simultaneous regions; gap=8 rows is not two hours | Timestamp masks for final train/validation/test, plus three expanding unique-timestamp CV folds for LightGBM/XGBoost; eight unique timestamp slots as gap | All 30 regions in folds, chronology assertions, date ranges in `chronological_cv.json`; fixed validation origins drive final selection |
| Fold preprocessing leakage | Whole-frame imputation and medians/modes see future; test preprocessing resets context | New models need no fitted imputation/encoding/scaling; warmup dropped, full preceding origin context retained | Feature causality tests pass; upstream inherited quarter-fitted clustering remains a limitation |
| Weather coverage, `weather_service.py` | Start-day-only fetch, stale cache, obsolete aliases, zero-filling, categorization drift | Full-horizon date requests; current provider fields with explicit aliases; 15-second timeout, three attempts, bounded TTL cache; validate all fields; original-compatible weather bins/WMO categories | Mock tests cover midnight, expiry, missing/null fields and failures; actual historical archive downloaded successfully |
| Weather information leakage | Observed historical weather used as though available at origin | Original results labeled reanalysis/oracle sensitivity; all new models omit weather | No future observed weather in candidate evaluation; no claim of forecast-origin weather availability in 2016 |
| API inputs/errors, `main.py` | Unbounded region/horizon, string datetime, all failures become 500 with exception details | Pydantic region 0–29, horizon 1–48/default 12, parsed datetime, clear NY handling; 422 history/input, 503 weather/model, generic 500; server logs retain exceptions | Test matrix covers invalid values, missing context, inference/weather/unexpected failure sanitization |
| API operational behavior | CPU inference blocks event loop; no health endpoint | Threadpool execution with isolated context; `/health`; structured log format; localhost CORS | Actual Uvicorn HTTP tests and startup model schema validation |
| Fake analytics, `main.py`, `frontend/src/App.jsx` | Hardcoded importance, arbitrary confidence bands, simulator fallback, random GPU/latency/alerts, unsupported scenarios | Genuine model importance, empty operational alerts, no uncalibrated confidence; real request timing; explicit unavailable errors; scenarios disabled; heatmap only pre-origin observations | Frontend production build passes; response fields preserved and HTTP contract checked; browser interaction was not manually exercised |
| Frontend horizon | UI says hours while backend counts steps | Display steps and corresponding hours | Build passes; API remains step-based |
| Shared module packaging | Backend-only Docker context cannot import sibling `ml` | Added root-context `Dockerfile.local` and local run guide | Docker runtime/image not executed; no deployment attempted |
| Production `.gitignore` encoding | Existing UTF-16 ignore file is not reliably interpreted by Git | Converted the cloned file to UTF-8 while retaining its rules and adding generated-output exclusions | `git status` excludes frontend build output, model/data paths and caches |

## Execution and outputs

- Streamed 33,234,199 genuine retained-trip region assignments from
  `data/time_series.csv`; source SHA-256 and policy in `artifacts/dataset_metadata.json`.
- Raw January–March CSVs, location data and processing data remain unchanged.
  They were inspected but not all reprocessed into new clusters because the
  original scaler/cluster artifacts are missing and RAM was constrained.
- Final panel has 261,960 observations. Original wall-clock count aggregate is
  retained separately; timezone-aware `artifacts/counts_ny.parquet` is authoritative.
- Trained LightGBM, XGBoost, CatBoost and Random Forest; computed weekly seasonal
  naive. Three Optuna LightGBM trials, four native threads, no cloud resources.
- Validation selects XGBoost; no production artifact is replaced. All candidate
  models, schemas, splits, metrics, timings, individual forecasts and quarantine
  records are saved under `artifacts/`.
- A fixed-parameter validation-only LightGBM lag ablation reduced recursive
  WAPE from 20.9634% without lag_96/lag_672 to 15.6364% with them. Rolling windows
  remained identical. It did not change the final candidate selection.
- Local MLflow SQLite tracking retains model parameters, metrics, training time,
  dataset fingerprint, feature configuration and artifacts, including the
  preliminary wall-clock experiment. Final derived-data provenance is attached
  as run artifacts/tags; reports identify the final authoritative result files.
- Final automated suite: 60 passed (one upstream Starlette/httpx deprecation warning).
- Actual localhost Uvicorn checks: 42 successful forecasts, all 30 regions,
  horizons 1/4/12/24/48, midnight/weekend/DST; 8 invalid requests rejected.
  Finite nonnegative values, output shape/order and pre-origin history verified.
- Frontend production build passes. It reports existing bundle-size and
  Browserslist-age warnings; no redesign was performed.
- The unused `SystemAlerts.tsx` component also had fabricated operational events;
  it now accepts supplied events and defaults to an empty list.
- Forecast plots were rendered and visually inspected; detailed region, demand
  tier, peak and lead metrics are CSVs in `reports/`.

## Remaining limits and review decision

This is a reproducible full regional-demand retraining, not a newly refitted raw
trip cleaning/geographic clustering pipeline. The original upstream transforms
were fitted across the full quarter and are not recoverable from provided
artifacts; fixed geographic assignments carry that provenance risk. Complete
source coverage is assumed; an outage cannot be distinguished from genuine
zero demand using retained trips alone. Quarantined DST trips need source-level
investigation if a lossless reconstruction is required.

Candidate validation/test compare like-for-like new schemas. Original-vs-new
metrics are descriptive because labels, features, fitting periods and weather
availability differ. Six test origins and a single winter quarter do not prove
modern or annual generalization. No probabilistic intervals, operational
alerting, modern live taxi feed, browser-driven UI validation, isolated model
RSS benchmark, or tested Docker runtime is claimed. Hyperparameter budgets are
small and not equal across model families. More untouched rolling origins and
modern observations are needed before choosing to replace production.

Reports: `model_comparison.md`, `recursive_backtesting.md`, CSV breakdowns,
`local_system_results.json`, `pytest.log`, `frontend_build.log`, `cv.log`,
`original_evaluation.log`, and `changed_files.json`. Run commands are in
`urbanflow/LOCAL_REPAIR.md`. Nothing is automatically promoted.
