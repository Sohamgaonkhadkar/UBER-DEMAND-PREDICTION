# UrbanFlow AI — advanced model optimization

**Verified untouched-April result: 10.2449% cumulative WAPE through 48 genuinely recursive steps. The below-10% target was NOT achieved.** The moving-block bootstrap 95% interval is [9.637%, 10.943%]; an interval crossing 10% is not evidence that the target was met. No tuning followed the April evaluation.

Selected candidate: `calendar_feedback_0.6`. MAE 13.5508, RMSE 21.7774, R² 0.97215, signed bias -0.908%. Saved separately at `artifacts/advanced_20261102/advanced_candidate_v3.joblib`. All original models, previous candidates, notebooks, reports and the existing backend source remain intact. Nothing was committed, pushed, deployed or changed remotely.

## A. Initial baselines and protocol differences

| Reference | WAPE % | Meaning |
| --- | --- | --- |
| Original LightGBM | 9.420 | March one-step, legacy zero→10/features/observed weather |
| Previous V2 XGBoost | 12.929 | Six March origins, 48-step cumulative, fit before February 15 |
| Previous V2 LightGBM | 13.294 | Same six origins and early fitting period |
| Frozen V2 XGBoost, expanded diagnostic | 13.794 | 57 March origins; still the original early-fitted candidate |
| Frozen V2 LightGBM, expanded diagnostic | 13.921 | Same expanded origins |

Those figures are historical context, not denominators for claimed April improvements. March had already been inspected in phase one, including full-period one-step errors; it is explicitly development validation here. A new chronological April test was acquired instead of relabeling March as untouched.

Training for development experiments: January 17–February 29 21:45 NY, with earlier observations supplying 1,536-step warmup. Validation: **57 origins, 13 elapsed hours apart, all 30 regions, all 48 steps** (82,080 predictions/model). Origins rotate through hours/days/weeks; 12-hour trajectories do not overlap. Training labels stop two hours before March 1; gap demand can enter later forecasts only as genuinely preceding observations. Direct training uses origins spaced 29 slots and all 48 leads; every target is before the fitting cutoff. Early stopping uses an inner February holdout, never April.

Final refit: Q1 labels before March 31 22:00 NY, same 16-day warmup; selection and comparator IDs were frozen first. Final test: **55 prespecified April origins × 30 regions × 48 leads = 79,200 predictions/model**. All origins were specified before April forecast errors were opened. Source: [official NYC coordinate-level 2016 records](https://data.cityofnewyork.us/Transportation/2016-Yellow-Taxi-Trip-Data/uacg-pexx), with 11,492,826 retained April trips under the unchanged original filters. Modern TLC Parquet uses taxi-zone IDs and could not directly preserve these 30 clusters, so it was not substituted.

The scaler and rounded notebook centroids were recovered and verified against **all 33,234,199 Q1 labels with zero mismatches**. The map was fixed before April. Source bounds/fare/distance filters were not tuned to April; targets retain genuine zeros. Daily query URLs, retrieval times and response hashes are saved. Complete source coverage remains an assumption. The target is pickups of trips passing the original post-trip filters, not an independently verified live all-taxi feed.

## B. Root-cause analysis

| Frozen model | Teacher-forced WAPE % | Recursive WAPE % | Recursive bias % |
| --- | --- | --- | --- |
| lightgbm | 9.166 | 13.921 | 2.984 |
| xgboost | 9.249 | 13.794 | 2.499 |

The same-model comparison changes only whether subsequent demand features receive actual prior demand or forecast feedback. It directly demonstrates feedback-state error; teacher forcing is a diagnostic oracle, never counted as a 12-hour forecast. Peaks were underpredicted by approximately 10% while other intervals were overpredicted. Night WAPE was about 22–23%, versus about 13% in daytime; weekends and low-demand regions were harder. This is conditional regression toward typical demand, not a single global bias. Adding a scalar OOF correction and exposure-context augmentation did not resolve it and were rejected.

Stable regional weekday/time shape was underused. Same-clock seasonal profiles helped; extra raw lags alone and a dense feature union did not. These observations support seasonal anchoring and reduced reliance on uncertain long recursive paths. They do not establish that weather, annual seasonality or all distribution shifts have been solved. One winter quarter cannot identify annual behavior.

![Feedback drift diagnosis](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/advanced_20261102/feedback_drift.png>)

Numerical diagnostics by lead, region, hour, weekday/weekend, peak/night, volatility, week of March, DST regime and origin are saved as `advanced_20261102/diagnostics_by_*.csv`. All difficult origins remain in evaluation.

## C. Feature engineering and target experiments

| Experiment | Validation WAPE % | Eligible |
| --- | --- | --- |
| feature_core | 13.596 | True |
| feature_extended_lags | 13.859 | True |
| feature_seasonal | 12.942 | True |
| feature_calendar | 12.784 | True |
| feature_statistics | 12.870 | True |
| feature_interactions | 12.819 | True |
| feature_spatial | 12.949 | True |
| feature_all | 13.170 | True |
| target_log1p | 13.642 | True |
| target_normalized | 18.387 | True |
| target_residual | 13.384 | True |
| objective_regression_l1 | 12.898 | True |
| objective_huber | 72.455 | True |
| objective_poisson | 13.248 | True |
| objective_tweedie | 12.888 | True |
| weather_lagged_reanalysis_sensitivity | 12.858 | False |

The retained compact ML representation uses 46 float32 features: temporal/cyclical/calendar fields, selected 15-minute lags, rolling mean/std/EWMA, same-clock yesterday/week/two-week values and smoothed profiles, and recent-to-seasonal ratios. Clock matching is New York/DST-aware; no other region enters accidentally. Explicit spatial experiments use a fixed three-neighbor map and joint generated predictions, never sibling future truth. Added medians/min/max/volatility, region interactions and spatial features did not beat the compact set in this screen. All 30 separate regional models also lost to the best global alternatives.

Raw, log1p, region-normalized and seasonal-residual targets were reconstructed on the unchanged count scale. Normalization was particularly weak globally; it was not promoted to make low-count regions look better. A direct seasonal-residual L1 model helped; the analogous recursive residual target did not. Huber severely underfit in its screened bounded configuration and was not expanded. Poisson/Tweedie were valid for nonnegative raw counts but did not win.

The weather sensitivity used only origin-minus-three-hour observed/reanalysis values held constant through the path, never future observed weather. Historical real-time availability is not certified, so it was ineligible even independently of its lack of validation gain. The selected model uses no weather. No unreliable historical event indicators were added. Genuine importance and high-correlation pairs are in `model_feature_importance.csv` and `high_feature_correlations.csv`; ablations, not temporally shuffled permutation scores, guided selection.

## D. Structured optimization

After diagnostics and ablation, Optuna ran 12 LightGBM and 8 XGBoost recursive trials, then 8 focused direct L1 trials. Seeds were fixed; native training used four threads. Search covered depth/leaves, learning rate, iterations, leaf/child constraints, subsampling and feature fractions, regularization, split gain/gamma, bins and count/absolute/squared objectives as appropriate. Inner-period early stopping selected useful iteration counts; a fixed subset of validation origins supported median pruning, and completed candidates were ranked on all 57 origins. Exact complete/pruned trials are in the Optuna CSVs and local `optuna.db`.

| Configuration | Architecture/strategy | WAPE % |
| --- | --- | --- |
| tune_direct_01 | lightgbm/direct | 11.437 |
| tune_direct_03 | lightgbm/direct | 11.471 |
| tune_direct_06 | lightgbm/direct | 11.498 |
| tune_direct_00 | lightgbm/direct | 11.533 |
| tune_direct_02 | lightgbm/direct | 11.624 |
| tune_xgb_02 | xgboost/recursive | 11.722 |

Full requested and fitted hyperparameters, feature flags, transformations, periods, timings, scores and retained artifact locations are in `artifacts/advanced_20261102/leaderboard.json`, local MLflow `tracking.db`, and `refit_metadata.json`. Inferior models were not all retained; sizes for unsaved artifacts are unavailable. No claim of exhaustive optimization or equal tuning budgets is made.

## E. Forecast strategies and final selection

| Strategy | Best eligible configuration | Validation WAPE % |
| --- | --- | --- |
| recursive | calendar_feedback_0.6 | 10.612 |
| direct | tune_direct_01 | 11.437 |
| hybrid | calendar_hybrid_4 | 11.037 |
| blend | calendar_blend_direct_0.5 | 10.559 |
| seasonal | calendar_profile_12 | 11.064 |

The selected model genuinely recurses. Each step starts with XGBoost, adds a modest prior-week profile anchor, then gradually blends in a training-only regional weekday/quarter median profile adjusted by origin-time observed demand. **The final blended prediction is appended to history and drives every subsequent lag/rolling feature.** No future pickup truth or future observed weather enters. The calendar-only profile contains no recursively invented observations. Direct and independently blended paths remain explicitly labeled as such.

The best direct/calendar blend had validation WAPE 10.5594%, versus 10.6121% for genuine recursion. The recursive-minus-overall paired-block 95% interval was [-0.0530, 0.1697] WAPE points. Before opening April, the recursive approach was preferred because the <=0.1-point gap was unresolved and genuine recursion was the primary objective. Both were frozen for test, so the choice was not made from April results. Ensemble candidates needed a paired block interval establishing gain over their best component; candidates failing that gate were marked ineligible. Intervals are exploratory after development search, without a multiple-comparison correction.

Exposure-aware training used genuinely preceding fitted-model feedback contexts and historical labels; it worsened validation to about 14.2%. OOF bias correction was unstable (about 22.1%) and rejected. Those failures are retained. They argue against assuming a more complicated correction necessarily improves deployment behavior.

## F. Compact validation leaderboard

| Rank | Experiment | Strategy | WAPE % | Eligible |
| --- | --- | --- | --- | --- |
| 1 | calendar_blend_direct_0.5 | blend | 10.559 | True |
| 2 | calendar_blend_recursive_0.5 | blend | 10.567 | True |
| 3 | calendar_feedback_0.6 | recursive | 10.612 | True |
| 4 | calendar_blend_recursive_0.25 | blend | 10.626 | True |
| 5 | calendar_blend_direct_0.25 | blend | 10.629 | True |
| 6 | calendar_feedback_0.3 | recursive | 10.707 | True |
| 7 | calendar_feedback_0.9 | recursive | 10.756 | True |
| 8 | calendar_blend_direct_0.75 | blend | 10.840 | False |
| 9 | calendar_blend_recursive_0.75 | blend | 10.852 | False |
| 10 | calendar_hybrid_4 | hybrid | 11.037 | True |
| 11 | calendar_profile_12 | seasonal | 11.064 | True |
| 12 | blend_recursive_direct_0.5 | blend | 11.074 | True |
| 13 | blend_recursive_direct_0.25 | blend | 11.116 | True |
| 14 | calendar_hybrid_12 | hybrid | 11.150 | True |
| 15 | blend_recursive_direct_0.75 | blend | 11.285 | False |
| 16 | calendar_profile_48 | seasonal | 11.340 | True |
| 17 | hybrid_switch_12 | hybrid | 11.358 | True |
| 18 | hybrid_switch_4 | hybrid | 11.370 | True |
| 19 | tune_direct_01 | direct | 11.437 | True |
| 20 | recursive_anchor_0.3 | recursive | 11.447 | True |

There are 75 completed recorded experiments plus pruned Optuna trials. The complete compact CSV is `advanced_20261102/leaderboard.csv`. All scores in that leaderboard use the same outer March origins and metric denominator. Phase-one frozen diagnostic models are kept separate from the comparably refit V2 baselines.

## G. Untouched April test results

| Frozen configuration | Strategy | WAPE % | MAE | RMSE | R² | Bias % |
| --- | --- | --- | --- | --- | --- | --- |
| calendar_feedback_0.6 | recursive | 10.245 | 13.551 | 21.777 | 0.972 | -0.908 |
| calendar_blend_direct_0.5 | blend | 10.290 | 13.611 | 21.951 | 0.972 | -0.819 |
| comparable_v2_xgboost | recursive | 12.113 | 16.021 | 26.048 | 0.960 | -0.879 |
| comparable_v2_lightgbm | recursive | 12.153 | 16.075 | 26.436 | 0.959 | -0.396 |
| seasonal_weekly_24 | seasonal | 14.237 | 18.832 | 30.569 | 0.945 | -0.108 |

Primary WAPE = 100 × Σ|actual−prediction| / Σ|actual| across **all** selected origins, 48 leads and regions. It is not an average of regional percentages, an aggregate-city cancellation score, or step-48-only error.

| Lead H | Time | Cumulative WAPE through H % | WAPE at lead H only % | Cumulative MAE | Cumulative RMSE | Cumulative R² |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 min | 8.293 | 8.293 | 10.836 | 16.451 | 0.983 |
| 4 | 1 h | 9.274 | 10.175 | 11.739 | 18.441 | 0.978 |
| 12 | 3 h | 10.194 | 10.296 | 12.143 | 19.367 | 0.973 |
| 24 | 6 h | 10.566 | 11.172 | 12.573 | 20.371 | 0.971 |
| 48 | 12 h | 10.245 | 10.618 | 13.551 | 21.777 | 0.972 |

Primary 95% moving-block interval: [9.637%, 10.943%], 1,000 resamples, seed 42, blocks of three chronological origins (~39 hours), retaining entire region/lead trajectories. Non-overlapping paths still share weekly behavior and historical context; dependence longer than the block is a limitation. This estimates performance uncertainty, not probabilistic intervals for individual pickup predictions. **The point estimate remains above 10%; the target is not verified.**

![April actual versus predicted](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/advanced_20261102/heldout_actual_vs_predicted.png>)

![Individual lead error](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/advanced_20261102/heldout_error_by_lead.png>)

![Origin variation](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/advanced_20261102/heldout_origin_variation.png>)

| Selected-model regime | WAPE % | MAE | Bias % |
| --- | --- | --- | --- |
| demand_low | 20.755 | 5.330 | -4.084 |
| demand_medium | 11.024 | 14.971 | -1.443 |
| demand_high | 8.648 | 20.351 | -0.253 |
| training_95th_percentile_peaks | 11.273 | 30.414 | -9.789 |
| night | 13.602 | 8.286 | -0.749 |
| day | 9.808 | 15.306 | -0.929 |
| weekend | 11.138 | 15.280 | -2.226 |
| weekday | 9.856 | 12.836 | -0.334 |
| peak_hour | 9.654 | 15.696 | -0.433 |

Regional WAPE improved against comparable V2 XGBoost in **30/30 regions**. Every region remains included. Full per-region, regime/peak, lead and origin metrics are in the heldout CSVs; no origin was removed because its error was large. Calendar seasons outside this quarter/spring transition remain untested.

Peak-demand WAPE improved from 14.463% to 11.273%, and peak underprediction improved from −13.027% to −9.789%. It remains substantial. Night WAPE improved from 18.273% to 13.602%; low-demand-tier WAPE improved from 24.661% to 20.755%, but its signed bias worsened from −2.491% to −4.084%. Thus not every bias goal improved despite all-region WAPE gains.

## H. Comparable improvements

| Refit baseline, same April protocol | Baseline WAPE % | New WAPE % | Gain in WAPE points | Relative error reduction % | Paired point-gain 95% interval |
| --- | --- | --- | --- | --- | --- |
| comparable_v2_xgboost | 12.113 | 10.245 | 1.868 | 15.420 | [1.408, 2.398] |
| comparable_v2_lightgbm | 12.153 | 10.245 | 1.908 | 15.702 | [1.471, 2.393] |

These gains compare the same April targets, origins, horizons, filters, regions, allowed Q1 fitting period and warmup. No percentage improvement is calculated against the original one-step 9.42% benchmark or the earlier six-origin March run.

## I. Local serving, cost and validation

Retained candidate size: 4.590 MiB. Final refit including dependent components: 43.77 seconds. Batched test inference averaged 42.49 ms per 48-step/all-region origin; this is throughput, not HTTP latency.

| HTTP horizon | p50 roundtrip ms | p95 roundtrip ms |
| --- | --- | --- |
| 1 | 81.3 | 84.7 |
| 4 | 110.0 | 119.2 |
| 12 | 224.3 | 348.9 |
| 24 | 328.7 | 337.5 |
| 48 | 579.3 | 602.3 |

Whole loaded backend process-tree RSS: 420.9 MiB (includes Python and library imports; not isolated model memory). Actual Uvicorn checks passed 48 valid and 9 invalid requests: all 30 regions; horizons 1/4/12/24/48; finite/nonnegative ordered outputs; identical short/long prefixes; compatible response fields; no future history; and DST with the Jan–Feb validation-era artifact. Servers were stopped after checks.

Full final regression suite: **101 passed**, one upstream Starlette/httpx deprecation warning. It covers shared features, direct origin-only inputs, cross-region feedback, timezone/DST, transformed count reconstruction, schema order, true calendar feedback, invalid/missing context and sanitized errors. The final Q1-refitted model rejects origins before its fitting cutoff; the separate validation-era model supports causal March/DST demonstrations.

The opt-in entrypoint is `urbanflow/backend/advanced_main.py`; original `main.py` and default model remain intact. Contract fields are preserved with candidate/strategy metadata. No frontend redesign or live deployment occurred. Serving currently uses the full tested local ML environment, including research-module dependencies; production dependency separation, container packaging, genuine delayed live observations and monitoring still require work. No live observation stream, calibrated prediction intervals or operational alerts are claimed.

A wrapper serialization issue was caught before freeze: CLI-defined classes used `__main__`. Original experimental files were retained, canonical portable copies were created, and **all saved validation predictions were verified unchanged**. Correct module entrypoints and actual independent-server loading now verify portability. The repair record is `artifact_portability_repair.json`.

## J. Recommendation and limits

Keep `calendar_feedback_0.6` as the strongest primary-goal candidate for local review. It improved genuine recursive accuracy materially and outperformed the fixed direct/calendar comparator on April, but **10.2449% is above the requested threshold**. Do not replace production automatically. These results justify further independent testing, not a claim of production readiness or guaranteed modern accuracy.

Further optimization must use a new development protocol and a new untouched chronological test; April is now spent. It must not be repeatedly inspected to choose new weights until a sub-10 number appears. Additional years, forecast-origin-available weather, better live data latency definitions, event/airport signals and longer-horizon seasonal coverage may help, but no unexecuted gain is claimed. Original filtering depends on completed-trip fields, and historical source completeness cannot be independently certified. Rounded recovered centroids match Q1 labels exactly but are not the original binary training state.

The rejected OOF bias experiment inserted its corrections inside the feedback loop, where they can compound. Its poor result does not rule out a separately validated post-path correction; that variant was not tested. Recorded training runtimes include feature-cache materialization on first use, so they are actual experiment costs rather than isolated estimator benchmarks.

Reproduction/run guide: `urbanflow/ml/advanced/README.md`. Evidence: protocol and selection freeze, source hashes, local MLflow/Optuna databases, leaderboard JSON/CSV, final test JSON/NPZ, schemas/refit metadata, all-region CSVs, system results and logs. Existing phase-one files and original-model bytes are verified in `advanced_20261102/preservation_checks.json`. No remote action was performed.
