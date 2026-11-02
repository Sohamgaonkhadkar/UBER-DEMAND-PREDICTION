# Actual local model comparison

Original production artifact and notebooks are preserved. No commit, push, PR or deployment was made.

## Data and evaluation

Rebuilt from 33,234,199 retained regional trip rows, not a demonstration sample. Final timezone-aware panel: 261,960 rows, 30 regions, 3,570 zeros. Quarantined 34 source trips in New York’s nonexistent DST hour; original records untouched. The upstream scaler/K-means were originally fitted on the whole quarter; their artifacts are unavailable, so region assignment is inherited and not a fully leakage-free raw-trip retraining.

| Split | Start | End | Rows |
| --- | --- | --- | --- |
| train | 2016-01-08 12:00:00-05:00 | 2016-02-14 21:45:00-05:00 | 107760 |
| validation | 2016-02-15 00:00:00-05:00 | 2016-02-29 21:45:00-05:00 | 42960 |
| test | 2016-03-01 00:00:00-05:00 | 2016-03-31 23:45:00-04:00 | 89160 |

All new candidates use identical zero-preserving targets, 33 numeric features, the same training period, and the same validation/test origins. No future weather is used. First 720 observations per region supply warmup. Two-hour gaps exclude fitting labels, not observations that really exist before a forecast origin. Test labels are never used for tuning. Models are not refitted on validation before test.

## One-step March test results

| Model | MAE | RMSE | WAPE % | R² |
| --- | --- | --- | --- | --- |
| Original LightGBM (legacy target/schema; not apples-to-apples) | 12.4143 | 20.0520 | 9.4200 | 0.9764 |
| seasonal_naive | 20.5837 | 33.9892 | 15.5980 | 0.9323 |
| lightgbm | 12.0973 | 18.4986 | 9.1672 | 0.9799 |
| xgboost | 12.2044 | 18.8265 | 9.2483 | 0.9792 |
| catboost | 12.7191 | 19.5382 | 9.6383 | 0.9776 |
| random_forest | 12.5824 | 19.5262 | 9.5347 | 0.9777 |

The original reported 12.41 / 20.04 / 9.41% / 0.9764 is closely reproduced. Its notebook builds test features using actual preceding test targets, full-period imputation, cross-region global lags and a zero→10 target. Weather is current-download historical reanalysis. These are teacher-forced one-step results, not recursive deployment accuracy. Small differences from saved notebook numbers can arise from weather revisions/library versions.

## Rolling-origin recursive test, all steps 1–48

| Model | MAE | RMSE | WAPE % | R² |
| --- | --- | --- | --- | --- |
| seasonal_naive | 20.8034 | 34.5163 | 16.0035 | 0.9366 |
| lightgbm | 17.2807 | 28.5337 | 13.2936 | 0.9567 |
| xgboost | 16.8071 | 26.6408 | 12.9293 | 0.9622 |
| catboost | 21.5274 | 31.6401 | 16.5605 | 0.9467 |
| random_forest | 18.5695 | 33.8366 | 14.2850 | 0.9391 |
| Original API, observed-weather sensitivity | 211.1076 | 250.5393 | 162.3996 | -2.3399 |

Three validation origins and six prespecified March test origins, all 30 regions, 48 recursive predictions each. This is 8,640 test predictions/model, not every possible March origin. Model comparison is paired across these origins; uncertainty or statistical significance has not been estimated.

## Validation, cost and selection

| Model | Validation recursive WAPE % | Training seconds | Artifact MiB | 48 steps × 30 regions seconds |
| --- | --- | --- | --- | --- |
| seasonal_naive | 16.5653 | 0.0000 | 0.0000 | 0.6936 |
| lightgbm | 15.6364 | 4.1940 | 0.9318 | 0.6733 |
| xgboost | 14.9206 | 2.3407 | 0.9169 | 0.7426 |
| catboost | 17.3487 | 4.1574 | 0.2229 | 0.7684 |
| random_forest | 15.9575 | 38.1333 | 72.1075 | 1.9069 |

**Provisional local recommendation: xgboost.** It has the lowest validation recursive WAPE, a small artifact and fast batched inference. LightGBM has slightly better one-step WAPE but worse validation and test recursive WAPE. CatBoost was worse than weekly seasonal naive at 48 steps in this bounded configuration; 200 depth-six boosting rounds are not evidence that CatBoost cannot do better. Random Forest is about 72 MiB and slower, with weaker recursive results. Only LightGBM received three controlled Optuna trials; this is a bounded benchmark, not equal-budget exhaustive tuning.

Do not replace production automatically. Original API inference and its training schema do not match: repairing their semantics changes the input distribution. The original trained through February; new candidates train through February 14. Original sensitivity results additionally use observed weather and a wall-clock legacy grid. A claimed percentage improvement against that original is scientifically unsupported. A larger untouched modern dataset, forecast-origin weather if weather is used, and more rolling origins are required before deployment.

## Daily/weekly lag ablation

With the same LightGBM hyperparameters, fitting period, rolling windows and validation origins, removing only lag_96 and lag_672 raised validation recursive WAPE from 15.6364% to 20.9634%. These correctly defined lags helped in this experiment. This was validation-only and did not trigger changes after test results were seen. See artifacts/ablations/lag_ablation.json.

Model memory was not benchmarked in isolation; serialized size is reported, not confused with resident memory. Native training was capped at four threads. RAM was constrained, so the available full regional intermediate was streamed instead of fitting new clustering on all raw coordinates. Raw trip files total roughly 5.4 GB and remain unchanged.

## Reproduction

See `urbanflow/LOCAL_REPAIR.md`, `requirements-local-lock.txt`, `urbanflow/ml/train.py`, `validate_cv.py`, `evaluate_original.py`, and `local_system_check.py`. Saved results are in `artifacts/results.json`, `artifacts/original_metrics.json`, `artifacts/splits.json`, `artifacts/chronological_cv.json`, and local MLflow SQLite tracking. Models are in `artifacts/candidates/`; original model SHA-256 is in `reports/baseline/original.json`. MLflow runs from the preliminary wall-clock experiment also remain; use the final timezone-aware JSON/Parquet outputs for these tables.
