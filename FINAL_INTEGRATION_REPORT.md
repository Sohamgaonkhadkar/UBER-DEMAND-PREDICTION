# UrbanFlow AI — Final Integration Report

Verified on November 2, 2026. The finalized v4 artifact is integrated without retraining. The exact supplied cinematic ZIP supplies the frontend design. The application is a tested **historical 2016 forecasting platform**; it does not claim current live demand, weather, calibrated prediction bands or operational alerts.

## Final local organization

Two independent sibling Git repositories are available at:

```text
C:\Users\Soham\Documents\UrbanFlow-AI\
  Model-Experimentation\
  Deployment-Platform\
```

Each has its own `.git` directory and correct remote, with no nested Git repository. The original workspace at `C:\Users\Soham\Documents\UBER DEMAND PREDICATION` remains in place, including its prior Git history and uncommitted changes. The integration copied its improved source and artifacts; it did not move or overwrite them. All 41 explicitly snapshotted original notebooks and v4 top-level artifact files retain their pre-integration SHA-256 hashes. The original research/deployment Git status snapshots are retained in research provenance.

## Model and data

| Item | Verified deployment location/value |
|---|---|
| Selected candidate | `v4_calendar_feedback` |
| Artifact | `Deployment-Platform/backend/bundle/candidate_v4.joblib` |
| Artifact bytes | 5,654,410 |
| Artifact SHA-256 | `7b96eaf0f9c9bec866e29677f9df9cb7647a99619c1c4589c35632a8f845c2c4` |
| Fitted schema | `backend/bundle/feature_schema.json`, 46 ordered columns |
| Inference observations | `backend/bundle/inference_observations.parquet`, 847,912 bytes |
| Geography | `backend/bundle/geography.json` |
| Evaluation | `backend/bundle/final_test_results.json` |
| Selection freeze | `backend/bundle/selection_freeze.json` |
| Integrity manifest | `backend/bundle/manifest.json` |
| Sealed numerical test evidence | `backend/tests/evidence/june_primary_predictions.npz` |

The model contains a 599-tree XGBoost component and a training-only calendar profile, with final blended outputs feeding each subsequent recursive step. The original core module is byte-identical; selected compatibility class ASTs match their research originals. No new fitting or selection was performed. Research and deployment candidate hashes match. Git attributes preserve source/data bytes across Windows/Linux checkouts; staged bundle blobs were checked against the integrity manifest before committing.

The minimal serving requirements pin FastAPI 0.162.0, Uvicorn 0.54.0, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, scikit-learn 1.9.1, XGBoost 3.4.1, holidays 0.107, joblib 1.6.0, PyArrow 25.0.1 and tzdata 2026.6. Python 3.14 was used. `backend/requirements-resolved.txt` constrains the complete independently installed serving/test environment. The research repository retains its complete original environment lock.

### Verified June 2016 scores

| Cumulative horizon | WAPE |
|---|---:|
| 15 minutes / 1 step | 9.750698% |
| 1 hour / 4 steps | 10.125974% |
| 3 hours / 12 steps | 11.176572% |
| 6 hours / 24 steps | 11.455725% |
| 12 hours / 48 steps | 11.608564% |

At 12 hours, MAE is **14.559471**, RMSE **23.945833**, R² **0.960489**, and bias **+2.566655%**. There are **55 origins × 48 leads × 30 regions = 79,200 predictions**. Individual lead-48 WAPE is **13.586195%**. The 95% bootstrap interval for cumulative WAPE is **10.755556–12.532959%**. It is not an individual prediction band. WAPE is an error measure, not classification accuracy. Acceptance tests independently recomputed the five cumulative score sets from sealed arrays and verified portable-runtime parity at three archived origins.

### Historical coverage

All times below are America/New_York:

- Observations: **January 1, 2016 00:00 through June 30, 2016 23:45**.
- Panel size: **524,040 region/time rows**, representing **66,874,467 retained completed trips**.
- Training labels: January 17 through May 31, 2016 21:45; cutoff exclusive at **May 31, 22:00**.
- Valid origins: **May 31, 2016 22:00 through July 1, 2016 00:00**.
- Full 12-hour observed comparison availability: origins through **June 30, 2016 12:00**.
- Default origin: **June 15, 2016 23:00**; derived from available coverage rather than today's date.
- Context: **1,536** consecutive quarter-hour bins across all 30 regions, strictly before origin.
- Forecast horizon: integer **1–48** quarter-hour steps; first target timestamp equals origin.

The target preserves the original completed Yellow Taxi filtering: both pickup/dropoff coordinates within latitude 40.60–40.85 and longitude −74.05–−73.70, fare $0.50–81.00 and trip distance 0.25–24.43 miles. It is not all requests or unmet demand. The June source is NYC Open Data dataset `uacg-pexx`; exact acquisition provenance is included. Post-trip filters mean immediate completed-bin availability remains an idealized as-of assumption. Zero bins inherit the documented source-completeness assumption.

## Backend implementation

The backend validates trusted bundle hashes and the fitted schema during startup. It rejects out-of-range origins, unsupported/unaligned timestamps, ambiguous/nonexistent civil times, unsupported horizons, invalid regions, incomplete contexts and real-time mode. Model calls receive only the strictly preceding history. Worker threads execute CPU work; a bounded locked cache stores one joint 48-step forecast per origin, shared by all regions and shorter prefixes. Public failures return safe errors without internal filesystem paths or synthetic substitutes.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Loaded model and historical status |
| `GET /coverage` | Genuine observation range, origin bounds, default and capabilities |
| `GET /regions` | 30 verified inverse-transformed fitted centroids |
| `GET /model/info` | Actual architecture, ordered features and limitations |
| `GET /model/metrics` | Verified June evaluation ledger |
| `POST /predict` | Single-region model forecast with real preceding observations |
| `POST /predict/batch` | All 30 regional trajectories at one origin |
| `GET /activity` | Explicitly observed previous-bin or forecast first-bin regional activity |
| `GET /historical` | Real regional observations with explicit interval bounds |
| `GET /insights` | Real hourly/regional summaries of the preceding 16 days |

The original scaler mean/scale and fitted centers determine coordinates exactly. Sector IDs 00–29 retain model identity without inventing neighborhood labels or polygons.

## Frontend completion and design preservation

The untouched ZIP was extracted and run before edits. Actual original browser screenshots cover the homepage and five internal pages at desktop size, plus the mobile homepage. The completed application has desktop/mobile screenshots for all six pages and an additional compiled-production Forecast Studio screenshot.

The supplied assets, CSS and decorative grid source have matching hashes. The original scene order, imagery, typography system, layout components, chart renderers, animation behavior, reduced-motion handling and responsive styles are retained. Data-driven chart shapes, corrected geographic marker positions, honest labels, loading/error panels and observed/forecast layer controls are intentional functional changes. The visual comparison was inspected after the loader finished and the real data loaded.

- Homepage: genuine forecast/history, observed regional ticker/ranking and actual retained-trip total; no synthetic demand series or fabricated engine-online claim.
- Overview: real history, selected-region forecast, observed previous-bin regional activity and honest weather unavailability.
- Forecast Studio: real region/date/time/horizon controls, genuine history, loading/errors, and CSV timestamps/values matching the API exactly.
- City Explorer: all 30 verified centroids, search/list/marker/keyboard selection, pan/zoom/reset, observed and forecast overlays with explicit timestamps, and a labeled schematic fallback when external tiles fail.
- Demand Insights: two genuine regional trajectories at the same origin/horizon and derived summaries from real observations/predictions.
- Model Intelligence: backend evaluation metrics and fitted schema metadata; unavailable whole-model attribution, prediction intervals, weather and operational alerts are identified honestly.

Review `docs/visual-comparison.png`, `docs/visual-baseline/`, `docs/screenshots/`, `docs/template-design-manifest.json`, `docs/runtime-source-parity.json`, and the professional README in the deployment repository.

## Validation actually executed

| Check | Result |
|---|---|
| Preserved core research suite | **106 passed** |
| Preserved v4 suite | **12 passed** |
| Clean-environment deployment suite | **22 passed** |
| Expanded Playwright suite | **37 passed**, final run approximately 2.9 minutes |
| Actual HTTP single-region predictions | **150 passed**: all 30 regions × 1/4/12/24/48 steps |
| Actual HTTP invalid requests | **8 correctly rejected** with 422 |
| TypeScript `tsc --noEmit` | Passed |
| Vite production build | Passed, approximately 3.456 MB single-file HTML output |
| Production preview with real API proxy | Passed, 12 genuine forecast table rows and no browser runtime errors |
| Clean environment `pip check` | Passed |
| Backend byte-compilation | Passed |
| `docker compose config --quiet` | Passed |
| Docker container build/execution | **Not executed: Docker engine unavailable** |

Research commands ran from the new research checkout using the existing full research environment:

```powershell
python -m pytest urbanflow/tests -q
python -m pytest urbanflow/ml/v4/tests -q
```

Deployment validation used its independently created `.venv` installed from `backend/requirements-dev.txt`:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m compileall -q backend
cd frontend
npm ci
npm run typecheck
npm run build
npm run test:e2e
npm run preview -- --host 127.0.0.1 --port 4176 --strictPort
```

The existing installed Chromium binary was selected through `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` for this machine. A fresh checkout can use `npx playwright install chromium` and omit that override. Tests exercised actual model/API values, every page, first-click navigation, browser back/forward/reload, mobile menus, CSV equality, verified marker interactions, real/schematic map pixels, scroll reversal, small portrait/landscape layouts, reduced motion, and deliberate backend/tile outages. Backend tests cover future poisoning, schema integrity, prefix consistency, DST/input validation, safe failure behavior and sealed numeric evidence.

Resolved findings included premature map bounds computation before region bootstrap, the original demo/calendar date paths, the real observation column mapping, missing-value display guards, preserving map state during regional refresh, Windows text encoding/newline effects, and legacy template assertions tied to demo labels. No final test failures remain. Non-failing warnings were the preserved pandas acquisition deprecation and FastAPI/Starlette TestClient deprecation; browser runs also emitted a Node color-environment warning.

Measured on this local Windows machine: one uncached joint origin HTTP request took **848.64 ms**; 20 cached single-region requests had median **12.02 ms**, maximum **17.19 ms**. These are local observations, not production service-level guarantees. `docs/http-validation.json` and `docs/browser-validation-summary.json` preserve the evidence.

## Repository cleanup and preservation

The research branch retains original/improved notebooks, experiment source, tests, canonical historical candidates, schemas, panels, sealed numeric evidence, protocols and reports. Generated intermediate checkpoints, feature caches, tracking databases, raw downloads and a duplicated 36 MB CSV panel are excluded from the new checkout; their original workspace copies remain preserved and exclusions are recorded with hashes. The largest research file is the approximately 14 MB original LightGBM notebook. Source content is approximately 97 MB before Git compression.

The deployment branch removes previously tracked `node_modules`, old Python caches, stale frontend/backend outputs and obsolete serving artifacts from the new branch's index. The installed local dependencies remain available but are ignored by Git. It includes only serving code, the compact frozen runtime/data bundle, frontend, tests, documentation and container definitions. No raw multi-million-trip downloads, environments, research search checkpoints or likely credentials were staged. Git history is retained; no force pushes or branch deletions were used.

## GitHub push verification

Both validated implementation commits were pushed successfully and verified with `git ls-remote`. This report is appended in a subsequent documentation-only verification commit; the implementation SHAs below identify the tested source changes.

| Repository | Branch | Validated implementation commit | Initial push |
|---|---|---|---|
| `https://github.com/Sohamgaonkhadkar/UBER-DEMAND-PREDICTION.git` | `codex/final-v4-integration-20261102` | `7a3ec022be5c22ff704c668b66eb2e024cd7ba42` | Verified |
| `https://github.com/Sohamgaonkhadkar/UrbanFlow-AI-Spatiotemporal-Demand-Forecasting-Platform.git` | `codex/final-v4-integration-20261102` | `3bc966f15e92ffe00fb4fe4628d44109b3bd2735` | Verified |

The research default `main` remains `8595244f4974c653598a04627e7fcc6fad40d519`; deployment `main` remains `76b1840c070e1f8f9589e7c96c18e6f9a1acc258`. Neither default branch was updated. No live cloud resources or deployment settings were changed.

## Exact startup commands

Backend, PowerShell:

```powershell
cd 'C:\Users\Soham\Documents\UrbanFlow-AI\Deployment-Platform'
# First install on a fresh clone:
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Frontend, a second PowerShell terminal:

```powershell
cd 'C:\Users\Soham\Documents\UrbanFlow-AI\Deployment-Platform\frontend'
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`; API documentation is at `http://127.0.0.1:8000/docs`. The already installed environments can be reused without recreating them. No research checkout is needed to run deployment inference.

## Remaining limitations

Docker CLI 29.7.2 is installed, but its `desktop-linux` engine pipe was absent. Compose syntax was validated; container build/run is explicitly untested. The equivalent clean Python install, actual HTTP service, TypeScript/production build, production preview and browser suite were verified.

This is historical 2016 inference for the retained completed-trip target. Current operations require a compatible as-of observation feed and new validation. Weather, calibrated prediction bands, whole-hybrid attribution, operational alerts and verified cluster polygons are unavailable. The July holiday confirmation remains unexecuted because compatible complete coordinate-bearing data was not established. These limitations are preserved in both README files and the model/API capability metadata.
