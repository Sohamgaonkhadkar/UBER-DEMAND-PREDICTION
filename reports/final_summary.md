# Final local outcome

1. **Original problems:** 77-feature model/inference mismatch; zero-filling
   missing features/weather; cross-region lags, rolling and EWMA; future-aware
   imputation; zero demand replaced by ten; row-based CV/gaps; unsafe recursive
   chronology; timestamp shifting; missing API validation; fabricated analytics.
2. **Repairs:** strict feature/schema checks, shared causal feature computation,
   timestamp-aware continuity and splits, real zero targets, DST quarantine,
   historical/realtime distinction, full-horizon validated weather with TTL and
   bounded retries, input/error handling, health endpoint, genuine importance,
   honest frontend states. Original data/model/notebooks preserved.
3. **Executed training:** full 33,234,199-row regional trip intermediate streamed
   into 261,960 valid timezone-aware region/time observations. LightGBM (three
   Optuna trials), XGBoost, CatBoost, Random Forest, and weekly seasonal naive;
   local MLflow tracking. Additional three-fold expanding timestamp CV for
   LightGBM/XGBoost and a daily/weekly lag ablation.
4. **Metrics:** original notebook one-step MAE 12.4143, RMSE 20.0520, WAPE 9.4200%,
   R² 0.97644 closely reproduce the reported benchmark. New one-step WAPE:
   LightGBM 9.1672%, XGBoost 9.2483%, CatBoost 9.6383%, Random Forest 9.5347%,
   seasonal naive 15.5980%. Full MAE/RMSE/R² tables are in model_comparison.md.
5. **Accuracy conclusion:** new candidates outperform seasonal naive on selected
   recursive origins except CatBoost at 48 steps. Original-vs-new improvement
   cannot be cleanly attributed: targets, schema, fitting periods and weather
   differ. The old API’s original feature logic drifts severely: cumulative WAPE
   57.2872% through 12 steps and 162.3996% through 48 steps in its documented
   observed-weather sensitivity experiment. These are not universal accuracy estimates.
6. **Recursive accuracy:** XGBoost cumulative WAPE through 1/4/12/24/48 steps:
   9.7199% / 10.6467% / 11.7173% / 12.2105% / 12.9293% (rounded; authoritative
   values are in artifacts/results.json). LightGBM reaches 13.2936% through 48.
   There are six March origins × 30 regions × 48 predictions = 8,640/model.
7. **Recommendation:** XGBoost for further local review, selected by validation
   recursive WAPE 14.9206%, versus LightGBM 15.6364%. About 0.92 MiB serialized;
   about 0.74 seconds for 48 steps batched across 30 regions in this run. Do not
   replace production automatically. Candidate files remain separate.
8. **Local system:** 60 automated tests pass. A genuine loopback Uvicorn server
   passed 42 valid requests across all 30 regions and 8 invalid requests,
   including DST/midnight/weekend transitions and finite ordered predictions.
   The repaired default backend also served the preserved original artifact
   with actual historical weather successfully. Frontend build passes.
9. **Limits:** inherited original quarter-fitted geographic clusters, missing
   scaler/K-means artifacts, assumed source completeness, 34 quarantined DST
   trips, winter-2016 data only, six test origins, bounded unequal tuning,
   no live taxi feed/calibrated intervals/operational alerts, no manual browser
   exercise or tested Docker runtime. Weather-free candidates avoid using future
   observed weather; original weather results are explicitly reanalysis sensitivity.
10. **Files:** repaired application in `urbanflow/`; training in
    `urbanflow/ml/train.py`; tests in `urbanflow/tests/`; candidates/data/results
    in `artifacts/`; reports/plots and machine-readable changed-file inventory in
    `reports/`; environment versions in `requirements-local-lock.txt`. Start
    commands are in `urbanflow/LOCAL_REPAIR.md`.

No commit, push, PR, remote change, deployment or production-model overwrite was
performed. Existing user notebook modifications remain intact.
