# UrbanFlow AI: next-generation model research

## Executive answers

1. **What limits accuracy?** Seasonal/calendar distribution shift, unpredictable regional demand innovations, sparse overnight counts, airport-centered behavior and a restricted completed-trip target. Observed peak underprediction partly reflects selecting on future actual demand; it is not evidence for an arbitrary inflation factor.
2. **What was tested?** Correct weekly profile adjacency, recency with shrinkage, a regularized origin-context calendar gate, post-path residual calibration, and an unchanged-recipe refit with genuine April history. Four research families; four novel configurations; no random search.
3. **What helped?** Adding April training history reduced matched May WAPE from **12.7634% to 12.5081%**, a **2.0007% relative error reduction** (0.2554 WAPE points; paired 95% gain interval [0.1643, 0.3342]). Weekly adjacency gave only 0.0186 point on March, below the frozen materiality threshold.
4. **What failed?** No novel recipe passed the March promotion gate. The contextual gate and post-path correction worsened validation; recency gains were small and uncertain.
5. **Below 10% on unseen recursive forecasts? No.** Selected May WAPE was **12.5081%**; the best predeclared pure XGBoost comparator was **12.0881%**. April's 10.2449% is a different period and is not a paired improvement baseline for May.
6. **What should be retained?** Retain the current recursive architecture and preserve the existing candidate/default. A separate Jan-April refit is available for review because added history helped on the matched test. Do not promote a novel architecture or switch to XGBoost based solely on its best May point estimate; the paired interval does not establish superiority.
7. **Next focus?** More representative chronological history and fresh June confirmation, holiday/calendar reliability, airport and event signals with verifiable availability at forecast origin, and explicit business agreement on the demand target. Prefer this over additional architecture complexity.

## Scope, integrity and evaluation

All work is local. No commits, pushes, PRs, deployments or production/default replacements were performed. Earlier experiments, candidates and user notebooks are protected by SHA-256 snapshots; see the preservation evidence linked below. Prior reports and the 75-experiment search were reviewed before the new analysis. New work lives in `ml/nextgen`, separate from the prior implementation.

April is **spent diagnostic data**, not an untouched test. No April forecast residuals fitted or selected new correction/gate weights. February out-of-fold calibration used a model fitted before February 14 22:00 New York; calibration origins span February 15–29. March validation used the frozen Jan-Feb base (exclusive cutoff February 29 22:00), with 57 chronological origins 13 hours apart. The current model's March predictions were reproduced exactly (maximum absolute difference 0).

Before opening May outcomes, the protocol, March selection, source hashes, comparator list and validation hash were frozen in [selection_freeze.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/nextgen_20261102/selection_freeze.json>). Novel promotion required at least 0.1 WAPE point gain, a strictly positive paired 95% gain interval, at least 20/30 regional improvements and no regional worsening over 2 points. No novel variant qualified; the current architecture was selected. Final fits used labels strictly before **April 30 22:00 New York**, with the unchanged 16-day feature warmup. The two-hour cutoff guard is intentional.

The fresh test covers May 2016: **57 origins**, every 13 elapsed hours from May 1 00:00, **30 regions × 48 steps = 82,080 predictions per model**. Each path receives only observations strictly before its origin; subsequent true demand cannot enter its recursive state. The complete test origin list is saved. Primary WAPE is `100 × sum(abs(actual − prediction)) / sum(abs(actual))` across every origin, region and all 48 leads. This is neither mean regional WAPE nor one-step accuracy. One step is 15 minutes; four is one hour; 48 is 12 hours. Cumulative metrics pool every step up to that lead; individual-lead metrics use only that step.

Uncertainty uses 1,000 paired circular moving-block resamples of three adjacent forecast origins, seed 42; complete 30-region/48-lead trajectories stay together. This respects within-path dependence and some adjacent-origin dependence, but one month and one spring holiday cannot establish seasonal robustness. These are score confidence intervals, not predictive intervals for individual counts.

## Deep diagnosis of the spent April test

The previous current model scored 10.2449% cumulative WAPE, MAE 13.5508, RMSE 21.7774 and R² 0.97215 on its 55 April origins. All 30 regional WAPE, MAE, RMSE, signed bias, demand mean/CV, zero fraction, peak and nighttime errors are in [spent_april_region_diagnosis.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/spent_april_region_diagnosis.csv>); all 48 regional leads are in [spent_april_region_by_lead.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/spent_april_region_by_lead.csv>).

**Priority 1: calendar reliability and distribution changes.** Late-step bias is negative; standalone calendar bias at step 48 is −3.0593%. The short origin-level adjustment decays well before 12 hours, so persistent level shifts can be lost. Current midnight profile smoothing treats a weekday's 23:45 and the same weekday's 00:00 as adjacent; correct weekly adjacency is a plausible small repair, not a promise of meaningful forecasting gain.

**Priority 2: sparse relative error and airports.** April low/medium/high training-demand tiers have WAPE about 20.755/11.024/8.648%. Sparse region 2 averages about 4.31 retained pickups per interval: WAPE 36.16% but MAE only 1.56. Relative errors rise as the denominator shrinks; neither regional WAPE nor whole-system WAPE alone captures the operational tradeoff. Airports 3 and 6 have approximately 17.50/18.57% whole-period WAPE and 35.92/65.24% nighttime WAPE. April nighttime (00:00–05:59) WAPE is 13.602%, versus 9.808% during the rest of the day. Higher relative variability and different operating patterns are plausible; unobserved flight/event effects remain hypotheses.

Centroids were reconstructed using the saved coordinate scaler and fixed centers. Region 3 is approximately (−73.78474, 40.64664), within 0.90 km of the [FAA JFK reference](https://www.faa.gov/air_traffic/publications/atpubs/aip_html/part3_ad_2.0_new_york.html); region 6 is approximately (−73.86891, 40.77073), within 0.78 km of the [FAA LaGuardia reference](https://www.faa.gov/documentLibrary/media/Order/2022-08-19_FAA_Order_JO_7400.11G.pdf). These establish airport-centered proximity, not exact polygons. Residential/commercial/entertainment land use was not independently verified and no causal group claim is made from display names. The original Q1 mapping was checked on 33,234,199 retained trips with zero assignment mismatches.

![April region and centroid errors](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/april_region_diagnosis.png>)

**Priority 3: distinguish forecastable peaks from future innovations.** For outcomes above each region's training 95th percentile, April WAPE is 11.2729% and bias −9.7895%. Selecting on actual future peaks preferentially includes unexpectedly high realizations. Peaks defined at origin by calendar expectation instead have WAPE 8.0969%, bias −0.4741%; forecast-defined peaks have WAPE 7.9212%, bias −0.4722%. These different subsets do not prove calibration, but they undermine the case for a blanket peak multiplier. A correction must use origin-known features and improve out-of-sample absolute error, not merely remove conditional bias selected after the fact.

![April peak conditioning and target hour](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/april_peak_and_time.png>)

Hour-of-day, weekday, weekend and public-holiday breakdowns are in [spent_april_time_diagnosis.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/spent_april_time_diagnosis.csv>); forecast starting-hour breakdowns are in [spent_april_by_origin_hour.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/spent_april_by_origin_hour.csv>). April has no US federal holiday in this definition, so its holiday slice cannot test holiday generalization. Error is 16.04% at target hour 04:00 and 15.99% at 05:00, versus about 9.33–9.51% at 08:00–09:00 and 9.68–9.79% at 16:00–19:00. Weekend WAPE is 11.138%, weekday 9.856%; Sunday is 11.315% with −3.223% bias, while Wednesday is 8.501%. These are descriptive slices with unequal counts, not independently established causal effects. Forecasts starting at 19:00 score 13.460%, versus 7.934% at 10:00, but each hour has only a few origins and different dates/target-hour coverage; do not select a starting-hour correction from these outcomes.

All 48 aggregate leads, with signed drift, are in [spent_april_by_lead.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/spent_april_by_lead.csv>). Error rises immediately from lead 1 (8.293%) to lead 4 (10.175%), then fluctuates rather than increasing monotonically; lead 48 is 10.618% with −2.994% bias. This supports a recursion/horizon reliability issue but does not isolate recursion from changing target-time mix. Regional lead tables permit checking whether drift is broad or concentrated rather than inferring it from one example path. On the fresh May test the analogous curves rise into the middle/late path before easing; the pattern is not a simple monotonic blow-up.

**Flattening is not itself an error target.** Forecast quarter-to-quarter absolute changes are 45.14% of actual changes, but at lead 48 cross-sectional forecast standard deviation retains 95.71% of actual standard deviation. Predictions should omit unpredictable count noise. These measures do not establish an avoidable flattening defect or a mathematical irreducible error floor.

## Exact architecture and controlled changes

The name `calendar_feedback_0.6` describes a **maximum**, not a constant weight. At zero-based lead k, inner output combines XGBoost with a previous-week anchor of weight `0.3*k/47`. Outer calendar weight is `0.6*(1-exp(-k/12))`, increasing from 0 to 0.5881. The effective raw learned-model weight at the last lead is approximately 0.2884. The calendar uses train-only region × weekday × quarter-hour medians with within-weekday smoothing. Its recent-origin ratio `(last-eight observed sum + 100)/(expected sum + 100)` is clipped to [0.6,1.6], with exponential decay over 12 steps. The **final blended prediction feeds back** to update subsequent lag/rolling features for all regions. Learned components and profiles cannot see future labels.

Horizon dependence already exists. A globally smaller/larger constant would not address regional reliability, calendar transitions or unseen holidays. We tested a low-capacity context gate instead of selecting weights from April errors. No arbitrary peak inflation or future-actual demand feature was introduced.

The unchanged-recipe final XGBoost configuration is `{"n_estimators": 599, "max_depth": 8, "learning_rate": 0.07844763945313377, "min_child_weight": 30.560921680599062, "subsample": 0.8575767013321436, "colsample_bytree": 0.7312056849564288, "reg_alpha": 3.351715828588323, "reg_lambda": 0.8869847878637795, "objective": "reg:absoluteerror", "random_state": 42, "n_jobs": 4}`. Exact configurations and feature schema are saved in [training_metadata.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/nextgen_20261102/training_metadata.json>) and [feature_schema.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/nextgen_20261102/feature_schema.json>). Existing lag/rolling, region, cyclical calendar and origin-safe spatial feature construction was preserved. The strongest prior pure XGBoost/LightGBM configurations were selected from the **prior March leaderboard**, then refitted on the same permitted Jan-April labels; V2 models are additional references, not substitutes for those stronger tuned baselines. No new tree search occurred.

## Research priorities fixed before validation



April is diagnostic-only. All new correction/gate coefficients use earlier OOF
data; April forecast errors will not tune or select their weights. March is a
spent development-validation period. May is the new sealed chronological test.

| Rank / family | Evidence and hypothesis | Likely benefit | Cost | Complexity | Overfit / leakage risk | Compatibility |
| --- | --- | --- | --- | --- | --- | --- |
| 1 / E1: seasonal profile reliability | Late calendar bias is about âˆ’3%; regional levels changed. Current smoothing wraps 23:45 to 00:00 of the same weekday instead of the next weekday. Test correct weekly adjacency, then a shrinkage/recency profile. | Modest, plausible; no numerical gain assumed | Seconds; reuse trained GBDT | Low | Low-medium; profiles fit only before cutoff | Same recursive API; final predictions feed back |
| 2 / E4: more genuine history | Q1 has limited seasonal coverage. April supplies genuine spring examples with compatible fixed regions. | Potentially broader than more model capacity; unknown | One bounded May source acquisition; one matched refit | Low-medium | Fresh-period integrity and source completeness require checks | Same target and map |
| 3 / E2: regularized contextual gate | Airports and nights have different reliability; a single 0.6 maximum cannot reflect all origin contexts. | Uncertain, possibly regional | One OOF fit and a small regularized optimization | Medium | Medium-high with few independent origins; no April coefficient fitting | Joint recursive feedback preserved |
| 4 / E3: post-path residual calibration | Previous residual additions amplified state inside the loop; that does not test correction outside it. | Uncertain; may reduce bias or worsen WAPE | One OOF ridge calibration; no GBDT search | Low-medium | Distribution shift; independent May test needed | Must be labeled postprocessed recursion, not feedback of corrected outputs |
| Deferred: flight/event/weather signals | Airport-center errors are large, but future realized arrivals/weather would leak information. | Potentially valuable, unmeasured | Historical as-of source availability uncertain | High | High until origin availability proven | New ingestion requirements |
| Deferred: neural / more random tuning | 75 previous experiments already tested broad GBDT/direct alternatives; no evidence capacity is the main bottleneck. | Unsupported at present | Higher | Higher | High selection/compute cost for this data volume | Additional serving burden |
| Separate task: all-pickup target | Existing filters use completed-trip fare/distance/dropoff and exclude real pickups. | Better business alignment, not a comparable metric gain | Raw-data rebuild | Medium | Target-definition change needs its own protocol | Cannot silently substitute for current target |

Four families are authorized for this run. E1 has two fixed ablations; E2 and E3
have one specification each. E4 isolates added history on May. No Optuna search.
Novel promotion requires >=0.1 WAPE point gain, a positive paired block interval,
improvement in >=20 regions, and no regional worsening exceeding 2 WAPE points.


## Executed experiments and March selection

- **E1, weekly adjacency:** one ablation changes smoothing adjacency to a continuous 672-quarter week. A second adds a weighted median with 28-day half-life and shrinkage equivalent to four prior weeks, retaining train-only region profiles. Both preserve final-output recursive feedback.
- **E2, contextual calendar gate:** ten origin-safe features: lead fraction, target sine/cosine, weekend, night, origin gain, origin volatility, component disagreement, train region CV and relative calendar level. Standardized features are clipped to ±5; ten coefficients perturb the existing schedule through a bounded sigmoid. First-lead calendar weight stays zero. Fit uses February OOF original-count smooth MAE `sqrt(error²+1)` plus L2 0.2, L-BFGS-B capped at 200 iterations. Calibration uses baseline-trace features; March evaluation reruns the complete changed recursive path. This is a lightweight surrogate fit, not end-to-end recursive optimization.
- **E3, post-path correction:** the same regularized feature basis fits February OOF residuals, scaled by train regional means for parameterization and bounded by train OOF regional 95th-percentile absolute residuals. Corrected outputs are **not fed back**. It is explicitly labeled `postprocessed_recursive`, separate from genuine corrected-output recursion. The hypothesis was to avoid previously observed in-loop error amplification; it still failed validation.
- **E4, added history:** unchanged current architecture and fixed hyperparameters, Q1 frozen versus Jan-April refit on the same untouched May test. April observed labels train the base/profile, but April forecast residuals do not fit any new gate/correction. This is a data-history comparison, not a new architecture selected on May.

| model | WAPE | gain | gain_CI | regions_improved | promote |
| --- | --- | --- | --- | --- | --- |
| current | 10.6121 | 0.0000 | [0.0, 0.0] | 0 | False |
| weekly_boundary_only | 10.5935 | 0.0186 | [0.00035, 0.03998] | 28 | False |
| recent_shrunk_weighted_median | 10.5920 | 0.0201 | [-0.02894, 0.07774] | 23 | False |
| learned_context_gate | 10.6269 | -0.0148 | [-0.06651, 0.02814] | 11 | False |
| post_path_linear | 10.8179 | -0.2058 | [-0.38199, -0.03776] | 5 | False |

Gains are current minus challenger WAPE points. Weekly adjacency improved 28 regions and has a barely positive interval, but its 0.0186-point gain is below the prespecified 0.1 materiality requirement. Recent shrinkage improved 23 regions but uncertainty includes zero. Context gating improved only 11 regions; post-path correction only five. Optimizers converged; lack of validation gain is not hidden as an optimizer failure. No failed direction was retuned after May. Separate MLflow experiment `urbanflow_next_generation` uses [tracking.db](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/nextgen_20261102/tracking.db>); prior experiment stores were preserved.

## Fresh May test: matched comparisons

| model | WAPE | MAE | RMSE | R2 | bias_pct | CI | offline_ms_per_origin | MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_Q1_current | 12.7634 | 16.1730 | 27.9981 | 0.9486 | 0.9235 | [10.877, 15.066] | 30.5314 | 4.5901 |
| matched_refit_current | 12.5081 | 15.8494 | 27.5001 | 0.9504 | 0.9862 | [10.571, 14.889] | 37.4463 | 5.0914 |
| matched_refit_tuned_xgboost | 12.0881 | 15.3173 | 24.9692 | 0.9591 | 0.4175 | [10.602, 13.914] | 55.0184 | 5.0186 |
| matched_refit_tuned_lightgbm | 12.4692 | 15.8002 | 25.8291 | 0.9563 | 1.9020 | [10.708, 14.678] | 45.8016 | 2.8583 |
| matched_refit_v2_xgboost | 13.4903 | 17.0941 | 27.2721 | 0.9512 | 0.7471 | [12.021, 15.294] | 74.8861 | 0.9400 |
| matched_refit_v2_lightgbm | 13.5222 | 17.1344 | 27.5779 | 0.9501 | 1.3119 | [12.044, 15.307] | 72.4741 | 0.9517 |
| clock_weekly_naive | 16.4166 | 20.8020 | 34.7418 | 0.9209 | 2.4183 | [14.14, 19.52] | 12.0105 | 0.0004 |

The selected recipe was the current architecture before May was opened. Adding April history produced a 0.2554-point improvement, paired 95% interval [0.1643, 0.3342], or 2.0007% relative error reduction. This is useful but modest; it does not achieve the target or prove robustness across seasons. Frozen Q1 versus refit is the valid paired improvement comparison.

The strongest prior pure XGBoost has the best May point estimate (12.0881%), reversing the earlier March ranking. The paired gain of selected refit relative to that XGBoost is **−0.4200 points**, interval **[−1.0998, 0.1509]**; the interval includes zero. Relative to tuned LightGBM, gain is −0.0388 points, interval [−0.4314, 0.3221]. Do not interpret the observed ranking as a conclusive architecture win or change model selection retrospectively. Keep pure XGBoost as a prespecified challenger for the next genuinely fresh period.

### Individual lead versus cumulative recursive WAPE

| model | step | individual_WAPE | cumulative_WAPE |
| --- | --- | --- | --- |
| frozen_Q1_current | 1 | 7.9343 | 7.9343 |
| frozen_Q1_current | 4 | 9.9547 | 8.9794 |
| frozen_Q1_current | 12 | 12.0853 | 10.4109 |
| frozen_Q1_current | 24 | 14.0350 | 11.6517 |
| frozen_Q1_current | 48 | 13.0577 | 12.7634 |
| matched_refit_current | 1 | 7.7953 | 7.7953 |
| matched_refit_current | 4 | 9.7845 | 8.8353 |
| matched_refit_current | 12 | 11.8375 | 10.2395 |
| matched_refit_current | 24 | 13.6624 | 11.4125 |
| matched_refit_current | 48 | 12.7700 | 12.5081 |
| matched_refit_tuned_xgboost | 1 | 7.7948 | 7.7948 |
| matched_refit_tuned_xgboost | 4 | 9.8250 | 8.8677 |
| matched_refit_tuned_xgboost | 12 | 11.5577 | 10.1069 |
| matched_refit_tuned_xgboost | 24 | 13.2461 | 11.2550 |
| matched_refit_tuned_xgboost | 48 | 11.9719 | 12.0881 |
| matched_refit_tuned_lightgbm | 1 | 7.9462 | 7.9462 |
| matched_refit_tuned_lightgbm | 4 | 10.1367 | 9.0468 |
| matched_refit_tuned_lightgbm | 12 | 11.9491 | 10.4257 |
| matched_refit_tuned_lightgbm | 24 | 13.6542 | 11.5819 |
| matched_refit_tuned_lightgbm | 48 | 12.3085 | 12.4692 |

All 48 individual leads are in [may_by_lead.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/may_by_lead.csv>); all origin-level scores are in [may_by_origin.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/may_by_origin.csv>). Overnight, peak, tier, weekend and holiday definitions below are identical across models. Regional tiers and peak thresholds are fitted only on permitted Jan-April history; realized peak selection retains the interpretation caveat above.

![May lead and origin comparisons](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/may_horizon_and_origin.png>)

### Selected refit regime results

| regime | n | WAPE | MAE | RMSE | bias_pct |
| --- | --- | --- | --- | --- | --- |
| night_00_06 | 20520 | 17.4991 | 10.4291 | 22.3437 | -3.1940 |
| day_06_24 | 61560 | 11.8430 | 17.6562 | 29.0160 | 1.5432 |
| realized_training_p95 | 4003 | 13.1168 | 29.7282 | 42.6086 | -11.5319 |
| Memorial_Day | 2640 | 44.1099 | 38.8096 | 61.1044 | 21.0587 |
| other_days | 79440 | 11.7862 | 15.0864 | 25.6380 | 0.5276 |
| weekend | 24120 | 13.5578 | 17.1868 | 29.3380 | 2.1604 |
| weekday | 57960 | 12.0710 | 15.2929 | 26.6980 | 0.4972 |
| low_region_mean | 27360 | 22.1923 | 5.8416 | 10.0111 | -5.4834 |
| medium_region_mean | 27360 | 12.5409 | 16.3476 | 24.6294 | 0.3956 |
| high_region_mean | 27360 | 11.3482 | 25.3591 | 39.5213 | 2.0927 |

The May 30 Memorial Day slice is a **post-test diagnosis**, not a reason to exclude observations or select a new model. Refit WAPE is 44.1099% on that slice, versus 11.7862% on other days; bias is +21.0587% versus +0.5276%. The tuned pure XGBoost holiday WAPE is 32.5152%, versus 11.6215% otherwise. All holiday predictions remain in the primary score. The architecture's weekday profile does not condition on holidays, while the learned tree has a calendar holiday feature; strong seasonal anchoring can plausibly suppress that distinction. A single event cannot establish causation, and non-holiday error still exceeds 10%. March has no federal holiday and Q1 contains few winter holiday examples, so gate validation offers little evidence for spring holiday reliability. No May-specific holiday weights were fitted.

### Every region, identical May forecasts

| region | frozen_Q1 | refit_current | tuned_XGB | tuned_LGB |
| --- | --- | --- | --- | --- |
| 0 | 18.0668 | 17.8788 | 19.1790 | 20.0340 |
| 1 | 12.3352 | 12.1092 | 11.7035 | 12.0571 |
| 2 | 39.8927 | 39.5315 | 40.9549 | 49.7100 |
| 3 | 18.1210 | 17.6845 | 17.9554 | 18.1077 |
| 4 | 12.8465 | 12.7014 | 12.2508 | 12.8527 |
| 5 | 11.9136 | 11.6161 | 10.7395 | 11.1498 |
| 6 | 18.4600 | 17.8193 | 18.3316 | 18.1881 |
| 7 | 12.4721 | 12.3268 | 10.7943 | 11.1597 |
| 8 | 10.3361 | 10.2008 | 10.0020 | 10.4464 |
| 9 | 25.9933 | 25.6370 | 25.8736 | 27.7025 |
| 10 | 13.3230 | 12.8018 | 10.9176 | 11.6478 |
| 11 | 11.3386 | 11.2681 | 11.5603 | 11.9226 |
| 12 | 12.0417 | 11.8008 | 11.3402 | 11.5253 |
| 13 | 23.5135 | 23.3726 | 24.4839 | 25.6744 |
| 14 | 12.3196 | 11.9918 | 11.2823 | 11.4038 |
| 15 | 12.6391 | 12.0996 | 11.2213 | 11.4810 |
| 16 | 26.2726 | 25.9507 | 27.1709 | 29.7219 |
| 17 | 26.8253 | 26.7370 | 27.3735 | 28.5376 |
| 18 | 10.2945 | 10.1580 | 9.7265 | 9.9307 |
| 19 | 11.9413 | 11.8039 | 11.5802 | 12.0410 |
| 20 | 11.0851 | 10.8817 | 11.3957 | 11.7253 |
| 21 | 10.3757 | 10.1479 | 9.8791 | 9.9631 |
| 22 | 12.1271 | 11.9110 | 11.0276 | 11.3479 |
| 23 | 11.3409 | 11.1740 | 11.3390 | 11.9819 |
| 24 | 36.5879 | 36.4386 | 36.2469 | 38.9705 |
| 25 | 14.5383 | 14.1248 | 14.2887 | 14.6466 |
| 26 | 13.6225 | 13.5638 | 13.1303 | 13.3724 |
| 27 | 12.0995 | 11.7649 | 11.1402 | 11.3735 |
| 28 | 33.3532 | 33.1448 | 33.5443 | 36.6836 |
| 29 | 24.9963 | 25.0735 | 25.8939 | 27.0138 |

Full regional MAE, RMSE, R², bias, demand mean/CV, zeros, peak and night scores for **every comparator** are in [may_per_region.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/may_per_region.csv>); corresponding regime tables are in [may_regime_metrics.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/may_regime_metrics.csv>). All target hours are in [may_by_target_hour.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/may_by_target_hour.csv>).

![Predetermined middle-origin path examples](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/may_fixed_example_paths.png>)

These examples use the middle chronological origin and fixed sparse/airport/other IDs [2,3,6,18]. They are illustrations; no example was selected to optimize a claim, and headline metrics include all origins.

## Data quality, target validity and feasible additional data

January raw audit scanned 10,906,858 records and exactly reproduced 10,505,157 retained trips. Marginal pass counts: pickup geography 10,725,121; dropoff geography 10,648,641; fare 10,887,046; distance 10,760,160. Reasons overlap and must not be summed. In airport-centered region 3, 232,807 of 254,836 pickups in geography survive all filters (8.65% excluded); region 6 retains 268,915 of 278,951 (3.60% excluded). Exact overlapping reason counts are in [data_quality_and_target_audit.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/data_quality_and_target_audit.json>).

The target is **retained completed taxi trips**, not all taxi demand, requests, rejected rides or even all pickups. Both pickup and dropoff must fall in latitude [40.60,40.85], longitude [−74.05,−73.70], fare [0.50,81] and distance [0.25,24.43]. Post-trip fare/distance/dropoff filters systematically exclude genuine journeys, including airport trips. They were kept for fair comparison; changing them requires a new explicitly named target and evaluation protocol. Monthly historical TLC availability is not a live observation feed. A production service forecasting actual pickups needs an operational source and target agreement.

The Jan-April panel has no duplicate region/timestamps, nonfinite counts or elapsed-time grid gaps and retains all 30 regions. New May was checked for the same contiguous 15-minute elapsed grid after joining history. Zeros are preserved, not interpolated, but aggregation alone cannot certify that a source outage did not appear as zero. New York timezone-aware elapsed indexing handles DST; the earlier 34-record DST quarantine is preserved. The fixed geography is not refitted for May. Outlier filters are unchanged, so comparisons do not gain by deleting hard targets.

May coordinate records were available from the official [NYC 2016 Yellow Taxi dataset](https://data.cityofnewyork.us/Transportation/2016-Yellow-Taxi-Trip-Data/uacg-pexx). Modern location-ID-only files cannot simply reproduce the fixed coordinate clusters. Before acquisition, feasibility estimated 1.90 GiB available RAM, 69.45 GiB disk, roughly 12 million monthly trips, less than 1 GiB extra persisted data and 31 bounded daily requests. Two source requests ran concurrently; four native training threads were used. Queries selected only pickup timestamp and coordinates **after the same original filters**, daily responses stayed below the one-million-record limit, and records were aggregated then discarded. All 31 daily aggregate files/request metadata and response hashes are saved, with final panel and geography hashes in [may_provenance.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/nextgen_20261102/may_provenance.json>). This is a reproducible bounded download, not an unnecessary full raw archive. Public-source completeness beyond the recorded response/grid checks is not proven.

Additional April examples helped, but Jan-April still lacks much seasonal and event coverage. No extra June outcomes were examined or used to choose these experiments. For future external data, realized weather or realized flights after origin would leak information; archived weather forecasts, published schedules and preannounced events need documented as-of availability. Their gains remain unmeasured.

## Local artifact, serving and validation

Review artifact: [nextgen_candidate.joblib](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/nextgen_20261102/nextgen_candidate.joblib>). It contains the selected unchanged-recipe **Jan-April data refit**, not a proven new architecture. The prior `advanced_candidate_v3.joblib` remains preserved. Schema, training metadata, protocol, selected-source hashes and observation seed are separate files in the new artifact directory. Reproduction/opt-in commands are in [README.md](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/urbanflow/ml/nextgen/README.md>). Existing API defaults and deployed application were not switched.

**106 tests passed**, with one existing FastAPI/httpx deprecation warning. Local real HTTP validation passed **45 valid requests** and **9 invalid requests**, including all 30 regions, horizons 1/4/12/24/48, horizon-prefix consistency, midnight/weekend/holiday timestamps, history/training cutoff guards and invalid inputs. The process tree measured 375.94 MiB RSS. HTTP latency below includes request parsing and history loading on this local machine; small sample sizes and cold starts limit generalization. Offline batched all-region origin throughput in the comparison table is a different measurement and should not be presented as HTTP service latency.

| horizon | n | p50_ms | p95_ms |
| --- | --- | --- | --- |
| 1 | 3 | 124.2219 | 1140.0596 |
| 4 | 3 | 195.8189 | 349.7103 |
| 12 | 33 | 339.5809 | 738.1856 |
| 24 | 3 | 631.3322 | 998.7656 |
| 48 | 3 | 1549.4340 | 1838.4618 |

Evidence: [local_system_results.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/local_system_results.json>), [nextgen_full_regression_tests.log](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_full_regression_tests.log>), [preservation_checks.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/nextgen_20261102/preservation_checks.json>). Prediction uncertainty bands are not supplied by the API; bootstrap score intervals should not be reused as forecast bands.

## Recommendation and next phase

Retain the **current recursive architecture** and existing default pending review. Preserve the separate data refit as the strongest selected local recipe for this phase; added history has positive paired evidence on May, while every novel recipe failed the frozen March gate. No new architectural winner is justified. Keep the stronger prior pure XGBoost as a serious challenger; predeclare both before fresh June confirmation instead of treating May as a reusable selection set.

Prioritize representative historical months and holiday/event coverage, verify operational target and feed availability, and investigate airport-specific origin-known signals. Weekly adjacency is a maintainability improvement with a very small measured effect; it was not bundled into the selected model. Do not spend further budget on the tested gate, shrinkage or residual correction without a new falsifiable hypothesis and additional development data. There is no measured guarantee that the present target and inputs can reach below 10%, and no proven lower bound saying they cannot.

May is now spent. Future experiments must document it as development/diagnostic history and reserve a new chronological test. All new evidence and source are local and reviewable; no automatic adoption or remote action was taken.
