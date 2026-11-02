# UrbanFlow AI — research-driven forecasting v4

## Executive result

The frozen development-selected system is **calendar_feedback**. Its chronological June benchmark score is **11.6086% cumulative 48-step WAPE**. Below 10% on the primary point estimate: **FALSE**. Strategy: **regional 15-minute recursive**. Coarse/disaggregated/reconciled variants are never mislabeled as 48 updates of quarter-hour feedback. The planned July 1–7 holiday check was **not executed** because a complete compatible coordinate source could not be verified.

No prespecified comparator reached a below-10% point estimate. A confidence interval crossing10% is not proof that the expected error is below10%.

No novel recipe displaced the selected strong baseline under the frozen gate. Selection was frozen before June targets were acquired. June remains the primary test even if a comparator looks better; no replacement event period was chosen. Because June contains no federal holiday, **holiday generalization remains unconfirmed on an independent holdout**. These are historical benchmark forecasts conditional on an idealized finalized-count history at each origin; they do not establish an operational, zero-latency observation feed.

The phase investigated four families: holiday representation, temporal/cross-temporal reconciliation, historical coverage/training windows, and sparse conditional counts. Eleven fixed novel variants plus two matched-history controls were compared with three strong model baselines and a seasonal reference over three already-inspected development months; no Optuna, tree-parameter search, contextual gate retraining or neural/GNN search was performed. All work is local; production/default models and frontend remain untouched.

## 1. Literature and strategy review



These are external findings, not reproduced UrbanFlow gains. Sources were
checked online during this phase; no paper's reported benchmark score is reused
as evidence for the taxi target.

| Approach/source | Relevant finding | UrbanFlow implication and boundary |
|---|---|---|
| [M5 winning method: In and Jung, DRFAM](https://www.sciencedirect.com/science/article/pii/S0169207021001813) | Partial pooling with direct and recursive LightGBM forecasts, followed by averaging, exploited complementary errors in retail sales. | Supports global models and testing combination rather than assuming recursion wins. M5 uses retail hierarchies and a different scoring task; the previous 75 UrbanFlow experiments already tested direct/recursive blends, so another unconstrained blend search is not justified. |
| [M5 global bottom-up robustness study](https://www.sciencedirect.com/science/article/pii/S0169207021001400) | Rolling-period experiments found that global bottom-up model rankings varied over time. | Motivates three chronological development months and fresh-period comparison, not selection from a single May ranking. |
| [Athanasopoulos et al., Forecasting with temporal hierarchies](https://robjhyndman.com/papers/temporalhierarchies.pdf) | Nonoverlapping aggregation and reconciliation combine temporal levels into consistent forecasts. | Test genuine 30/60-minute totals and original-quarter errors. Smoothing or hourly score improvement alone does not establish a win. Fixed nonnegative top-down alignment is a transparent limited implementation, not a reproduction of every optimal reconciliation method. |
| [Di Fonzo and Girolimetto, Cross-temporal reconciliation](https://arxiv.org/abs/2006.08570) | Temporal and contemporaneous constraints can be considered jointly; their study includes least-squares optimal formulations. | Test a city-hour constraint in addition to regional temporal totals. This phase does not estimate a large covariance matrix or claim MinT optimality; the score remains absolute regional quarter errors. |
| [ST-MGCN, Geng et al.](https://ojs.aaai.org/index.php/AAAI/article/view/4247) | The indexed paper describes multiple spatial relationship graphs for ride-hailing demand. The publisher blocked full-text access in this session. | Geographic adjacency, functional similarity and transport connectivity are distinct. Verified centroids supply proximity, not a road or airport-flow graph. Existing neighbor-feature ablation failed, so no demonstrated advantage warrants GNN implementation here. Full-paper reproduction was not performed. |
| [DCRNN, Li et al.](https://arxiv.org/abs/1707.01926) | Directed diffusion convolution and recurrent sequence forecasting model traffic dependence. | Road-sensor traffic results do not transfer automatically to taxi pickup clusters or a 12-hour count horizon. A reliable directed connectivity signal is currently missing. |
| [Graph WaveNet, Wu et al.](https://arxiv.org/abs/1906.00121) | Learns adaptive spatial dependencies along with temporal convolutions. | Learned correlations may be useful but can also fit seasonal co-movement without adding origin information. No graph model is implemented without a positive low-cost spatial ablation. |
| [Haslett et al., Modelling excess zeros](https://onlinelibrary.wiley.com/doi/full/10.1111/insr.12479) | Excess zeros can be modeled through different distributional and mixture constructions; overdispersion and zero mechanisms should be distinguished. | An unconditional zero excess is not proof of structural zero inflation: varying time-of-day intensity can explain part of it. Test a small overdispersed NB conditional-median specialist before adding a hurdle classifier. Prior global Poisson objective failure is retained rather than repeated. |
| [N-HiTS](https://arxiv.org/abs/2201.12886) | Multi-rate sampling and hierarchical interpolation support direct multi-horizon neural forecasts. | Its interpolation is not the same as count-conserving temporal reconciliation. It is a possible challenger after more representative development evidence; no UrbanFlow neural gain is claimed. |
| [TiDE](https://arxiv.org/abs/2304.08424) | A dense encoder-decoder provides direct forecasts with covariates at lower complexity than large attention models. | Future-known calendars could be valid inputs. Prior strong direct GBDT experiments and limited independent holiday examples make a new neural search lower priority than the selected bounded tests. |
| [Hyndman and Athanasopoulos, time-series regression](https://otexts.com/fpp3/regression.html) | Calendar/intervention predictors and the distinction between ex-ante and ex-post forecasts make information timing explicit. | Use deterministic holiday dates and directionally separate adjacent days; treat post-event attendance, future realized flights and reanalysis as unavailable in the primary experiment. This motivates representation, not any particular fitted correction weight. |

Direct models avoid repeatedly feeding predictions into features but may learn
each horizon with less effective data. Recursive models reuse a one-step model
but expose it to generated states. Multiple-output/encoder-decoder models learn
a path jointly, adding capacity and a different training objective. A blend can
reduce variance only when its errors are usefully complementary out of sample;
coherence alone is an algebraic property, not an accuracy guarantee.

Calendar dates, federal holiday identities, directional offsets and published
event schedules can be known at origin. Realized weather, flight delays,
attendance and even historical reanalysis do not become issued forecasts merely
because they are available now. No verified 2016 as-of flight/event archive was
established, so those features remain outside the primary comparison. The
holiday experiment uses deterministic calendar data and observed **past**
events only, pooled and shrunk rather than a separate model for one summer day.


## 2. Existing limitations and causal hypotheses

The prior calendar-feedback model scored 10.2449% on April and 12.5081% on May after an April-history refit. These different periods are not a paired deterioration experiment. The May frozen-Q1 versus Jan-April refit comparison did show a modest 0.2554-point benefit from additional data. Pure XGBoost was numerically best in May (12.0881%) but its paired interval against calendar feedback included zero. Calendar feedback was stronger in March and April. There was no basis for declaring one architecture universally superior.

May Memorial Day calendar-feedback WAPE was 44.1099%, versus 11.7862% on other days. The preceding Saturday also scored 26.2024% with +22.6872% bias; a holiday-date-only explanation is incomplete. The ordinary weekday profile ignores event regimes; the tree has `public_holiday` and a single undirected `holiday_adjacent` flag. New Year, MLK Day and Washington's Birthday appear in the Jan-Feb panel, but the 16-day label warmup excludes New Year: only **two distinct holiday dates supply original base-model training labels**. March and April contain no federal holiday, so successful ordinary-weekday validation cannot establish summer holiday reliability. Region/hour/offset details, including airports, are in [spent_may_holiday_analysis.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/spent_may_holiday_analysis.csv>).

![Spent May holiday and adjacent-day diagnosis](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/spent_may_holiday_offsets.png>)

Sparse regions 2/24/28 have Jan-Feb mean counts about 4.40/6.50/7.77 and zero rates 15.66/3.94/4.41%. Their unconditional variance-to-mean ratios are about 11.89/6.69/10.56. However, time-varying conditional intensity itself generates overdispersion and zeros; this does not prove a separate structural-zero process. A conditional count specialist was tested instead of assuming a hurdle classifier must help. Airports 3/6 remain verified centroid-proximity labels, not exact polygons. No residential/commercial/entertainment label was assumed geographically exact.

Regression toward typical demand, missing event information, source/target restrictions, seasonal drift and recursive state errors are plausible interacting mechanisms. None is proven to be the sole cause of the April–May difference. The prior realized-peak underprediction diagnostic conditions on future actual peaks; it does not justify arbitrary forecast inflation.

## 3. Ranked experimental plan and bounded scope



March, April and May are development evidence. June is the primary new test;
July 1â€“7 is a separate, prespecified holiday confirmation window using the same
May-cutoff model. June cannot test a US federal holiday. Neither window will be
replaced or used to select the winner after results are opened.

| Rank/family | Hypothesis and mechanism | Evidence | Difficulty/cost | Data/evaluation | Reject when |
|---|---|---|---|---|---|
| 1: Calendar representation | Directional holiday offsets and pooled historical holiday profiles can distinguish exceptional days from ordinary weekdays; a fixed safety rule can avoid enforcing an unreliable ordinary-week profile | May holiday and adjacent-day failures; existing profile ignores holidays; only three Jan-Feb holiday dates | Moderate; one fixed-capacity boosted refit and train-only shrinkage statistics, no fitted May gate | Compatible 2015 holiday examples if exact assignment feasible; March/April/May identical paths, report each event | Pooled gain <0.1 point, uncertainty unresolved, fewer than two months improve or regional gate fails; no complex per-holiday model with scarce examples |
| 2: Multi-resolution | Recursive 30/60-minute totals may be less noisy and require fewer feedback updates; disaggregate using training-only shares, reconcile with regional 15-minute forecasts | Direct and recursive alternatives have complementary errors; new methodology not previously tested | Moderate; two small fixed LightGBM models plus a city-hour model; no large covariance estimate | Origin-relative nonoverlapping blocks, including :15/:30/:45 starts; original 15-minute regional WAPE | Coarse accuracy gains fail to transfer to original counts or coherent combination fails the same promotion gate |
| 3: Historical coverage/window | Older examples may improve exceptional calendar coverage, but stale levels may hurt; compare identical tree capacity with expanded history, rolling 60-day labels and fixed 90-day-decay case weights | Added April labels helped May; prior failed profile recency is different from training sample weighting over extra historical segments | Moderate; three fixed refits; feature cache reused | Reuse Jan-May and bound 2015 aggregation; never interpolate missing months or use cross-gap histories | No robust multi-month gain; source mismatch/incompleteness prevents comparable data |
| 4: Sparse conditional counts | A negative-binomial conditional median may improve count-scale absolute error in genuinely sparse clusters without changing dense regions | Small denominators, excess zeros/overdispersion; global Poisson objective already failed, so not repeated | Low; one regularized NB regression with a pooled dispersion estimate and explicit median, only train-mean <=10 regions | Compare unchanged global recursion versus specialist feedback; original global WAPE and low-count slices | Aggregate gain fails materiality/uncertainty/regional gate even if sparse WAPE alone improves |

Use one fixed configuration per meaningful component. Evaluate hourly-only,
30-minute-only, equal-strength coherent temporal reconciliation, and optional
city-hour alignment separately. Cross-temporal results must retain regional
absolute errors; cancellation at city level is never the primary metric.

No learned contextual gate or post-path residual correction is repeated. A
fixed holiday safety rule is a structural omission test, not a May-fitted weight.
No flight/event/weather features enter without independently verified as-of
availability. Prior failed spatial features do not justify a graph neural net;
literature accuracy on other datasets is not an UrbanFlow result. Neural models
are deferred unless these bounded tests establish a specific unmet advantage.

Budget: four native threads, two source queries concurrently, cached derived
panels, no Optuna. Save each failure, preserve previous code/artifacts, and stop
changing model recipes once selection is frozen.


Historical scope was reduced for observed API cost **before historical-model outcomes**: 29 November dates plus nine December snapshot dates, **38 days total**. Calendar-snapshot dates include Christmas/adjacent days and matched prior-week references. Missing December dates remain absent; only full 16-day contiguous histories can generate ML training rows. Consequently older boosted-model labels come from the valid November segment, while isolated December snapshots contribute holiday ratios only. This is limited additional coverage, not a full year. See [resource_scope_amendment.md](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/resource_scope_amendment.md>) and [selection_clarification.md](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/selection_clarification.md>) for the pre-outcome sparse-region gate clarification.

## 4. Implementation and configurations

**Strong references.** Pure tuned XGBoost (600 trees) and tuned LightGBM (534 trees) retain their strongest earlier fixed parameters and causal feature configurations. Existing calendar feedback retains each prior fold's exact base (600 trees in March,599 in April/May) and horizon schedules; its final refit uses599 trees. March/May pure models and all three prior calendar folds were reused; the missing matched April pure baselines were fitted with the same allowed Jan–March labels. Final references refit on labels before **May 31 22:00 NY**, preserving the 16-day warmup and two-hour label guard. The primary Jan-only capacity and labels are identical for hierarchy/sparse comparisons. Additional-history arms intentionally change data; component contrasts isolate that effect.

**Holiday-aware boosted arm.** Fixed XGBoost capacity, with separate before/after flags, capped days-to/ since-holiday, and month sine/cosine added to existing calendar features. These are known before the origin. The arm uses the same expanded historical labels as its expanding-history control. November adds Thanksgiving labels; isolated Christmas snapshots cannot supply a 16-day ML context and therefore enter profiles only. Before March, this gives the boosted arm only three independent holiday label dates, while pooled profiles can use more historical event dates. Final fitting also includes Memorial Day. Month features and older data can affect ordinary days, so gains cannot automatically be attributed entirely to holidays. This remains limited evidence; no separate complex per-event model is trained on one summer date.

**Holiday profile arm.** Ordinary region×weekday×quarter medians exclude holidays and adjacent days. Each historical holiday/adjacent day is normalized against at least two available same-weekday references ±7/14 days within the fitting cutoff. Pooled region×hour ratios require at least four distinct dates and shrink toward 1 with eight prior dates; ratios are bounded [0.25,2.5] before shrinkage. Support arrays and actual ratios are in model metadata. This pools scarce event evidence rather than memorizing one holiday. The independent fixed safety arm suppresses both ordinary-week seasonal anchors within ±1 day, using the same boosted base. Its 0/1 structural rule is not a weight optimized on May residuals.

**Temporal models.** Two global LightGBM L1 models (400 estimators, learning rate .04, 31 leaves, minimum40 leaf samples, L2=3, four threads, seed42) forecast origin-relative 30/60-minute count blocks recursively. Histories contain only complete blocks preceding the origin; future hour fragments are never observed inputs. Training origins stride `factor+1`, covering every quarter phase. Features include region, local cyclic/weekday/quarter phase, historical coarse lags/means/std, same-local-clock day/week/two-week blocks and deterministic holiday offsets. Train-only region×weekday×starting-quarter proportions, shrunk with eight uniform pseudo-blocks, distribute counts back to quarters. The hourly arm performs 12 coarse feedback updates, not 48 fine feedback updates; the half-hour arm performs24.

**Reconciliation.** A fixed equal-strength regional hour target averages the fine path's hourly sum and the hourly model. Half-hour targets average fine/half-hour totals and are renormalized to that hour. Nonnegative normalized quarter shares combine fine/half-hour shape. A separate city-hour arm averages that regional sum with a learned aggregate-hour forecast and scales consistently. Here “city” means the **30-cluster retained-trip study area**, not every NYC taxi pickup. All constraints hold before API rounding, with an explicit all-zero fallback. This is transparent top-down alignment, not estimated-covariance MinT or proof of optimality. Reconciled quarters are post-path outputs; they do not feed back into the fine model. The primary score always uses regional quarter absolute errors, preventing aggregate cancellation from masquerading as improvement.

**Windows.** Same 600-tree capacity: Jan-only control; expanded valid history; rolling 60-day labels; expanded labels with fixed 90-day half-life case weights. The latter changes boosted-model training weights over older data, unlike the previously failed28-day profile-reweighting experiment. No window choice is selected on June or July.

**Sparse specialist.** Regions with train mean<=10 use a regularized log-linear NB2 model with sparse-region intercepts, selected log lag/rolling/weekly features and cyclic/calendar fields. Dispersion is estimated from training residuals after a Poisson initialization; NB likelihood is then optimized. The conditional median, rather than the mean or a probability-times-magnitude heuristic, targets absolute error. Only these regions' **final outputs** replace baseline values and feed back. Global WAPE and paired uncertainty still decide promotion. This is a specific small specialist, not a claim that all hurdle/count models fail.

Exact per-fold parameters, label bounds, support and fit times are saved under [models](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/v4_20261102/models>); selected schema/metadata are [feature_schema.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/v4_20261102/feature_schema.json>) and [training_metadata.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/v4_20261102/training_metadata.json>). MLflow experiment `urbanflow_research_v4` and artifacts are isolated in [tracking.db](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/v4_20261102/tracking.db>) and its own artifact folder. Earlier tracking stores are unchanged.

## 5. Development results and frozen selection

March, April and May are **known development evidence**: 57/55/57 nonoverlapping 12-hour paths, 13 elapsed hours apart, all 30 regions, all 48 leads. Pool absolute errors/denominators across months, rather than averaging regional or monthly percentages. The strongest pooled baseline was **calendar_feedback** at **11.1122%**. New recipes needed at least0.1 WAPE point gain, strictly positive paired95% gain interval, improvement in at least2/3 months and20 regions, and no region worsening>2 points. For a specialist changing only sparse regions, all affected sparse regions must improve instead of the impossible20-region breadth requirement; every aggregate requirement remains. Confidence intervals resample three-origin circular blocks separately within each month,1,000 draws,seed42, retaining full regional trajectories. They are exploratory after bounded selection, not multiplicity-adjusted proof.

| model | WAPE | gain | gain_CI | regions | monthly_gains | passes |
| --- | --- | --- | --- | --- | --- | --- |
| pure_xgboost | 11.5677 | -0.4555 | [-0.7562, -0.1538] | 2 | [-1.1099, -0.6463, 0.42] | True |
| pure_lightgbm | 11.8582 | -0.7460 | [-1.0058, -0.4983] | 0 | [-1.3021, -0.9487, 0.0388] | True |
| calendar_feedback | 11.1122 | 0.0000 | [0.0, 0.0] | 0 | [0.0, 0.0, 0.0] | True |
| history_expanding | 11.5582 | -0.4460 | [-0.7767, -0.0867] | 3 | [-1.1131, -0.6778, 0.4841] | False |
| history_rolling60 | 11.7453 | -0.6331 | [-0.9329, -0.3281] | 0 | [-1.1099, -0.8035, 0.0363] | False |
| history_decay90 | 11.5143 | -0.4021 | [-0.7278, -0.0467] | 2 | [-1.0821, -0.6704, 0.5782] | False |
| history_lightgbm | 11.7393 | -0.6271 | [-0.9158, -0.334] | 0 | [-1.2073, -0.8891, 0.2428] | False |
| history_calendar | 11.0337 | 0.0785 | [0.0368, 0.1175] | 20 | [0.0871, 0.0418, 0.1063] | False |
| holiday_boost | 11.6893 | -0.5771 | [-0.8816, -0.2701] | 0 | [-1.052, -0.7753, 0.1183] | False |
| holiday_profile | 11.0304 | 0.0818 | [0.0011, 0.199] | 26 | [0.0545, 0.0202, 0.1724] | False |
| holiday_safety | 11.0667 | 0.0455 | [-0.0775, 0.2367] | 22 | [-0.005, -0.0096, 0.1538] | False |
| sparse_nb | 11.6202 | -0.5081 | [-0.8071, -0.2109] | 2 | [-1.1798, -0.6912, 0.378] | False |
| hourly_disaggregated | 13.4921 | -2.3799 | [-2.887, -1.8985] | 0 | [-3.9069, -1.6873, -1.4827] | False |
| half_hour_disaggregated | 13.4103 | -2.2981 | [-2.8265, -1.7794] | 0 | [-3.7885, -1.6696, -1.3748] | False |
| temporal_reconciled | 12.0808 | -0.9686 | [-1.3235, -0.6306] | 0 | [-1.9457, -0.7807, -0.1374] | False |
| cross_temporal | 12.1451 | -1.0329 | [-1.4326, -0.6351] | 0 | [-2.1267, -0.8422, -0.0828] | False |
| seasonal_weekly | 14.7288 | -3.6166 | [-4.0417, -3.2458] | 0 | [-2.9747, -3.9923, -3.9085] | False |

`passes` marks baseline eligibility as well as challenger gates; a baseline has not thereby shown a novel gain. Monthly gains are relative to the strongest **pooled** baseline, in March/April/May order. The selected model is **calendar_feedback**, gain **0.0000** points versus that reference. Model IDs, source and validation hashes were frozen in [selection_freeze.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/v4_20261102/selection_freeze.json>) before new target acquisition. All individual monthly metrics and isolated data/feature contrasts are in [development_month_metrics.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/development_month_metrics.csv>) and [component_contrasts.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/component_contrasts.csv>).

![Development comparison](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/development_results.png>)

The tested coarse-only arms fail on original-quarter error despite smoother aggregate predictions. For example March hourly total WAPE is12.1016%, but disaggregated quarter WAPE is14.5190%. April hourly total WAPE is9.4153%, while quarter WAPE is11.9322%. These are different targets; the lower coarse score is not a valid improvement claim. Giving the distribution model the **actual future** hourly totals yields March quarter WAPE7.4883% and April7.0678%; that is an explicitly labeled diagnostic oracle, not a forecast, lower bound or eligible candidate. It shows both total-prediction and intra-block shape error matter. See [coarse_method_diagnostics.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/coarse_method_diagnostics.csv>). Rejection applies to these fixed implementations; it does not establish that every hierarchical model must fail.

Strategy-by-condition results are in [development_strategy_conditions.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/development_strategy_conditions.csv>). March/April have no federal holiday and only May supplies a summer event, so a complicated learned expert selector lacks repeated event validation. The phase tests a fixed calendar safety omission and rejects building an unconstrained selector from a few difficult dates. No selector uses future actual intensity.

All17 recipes' regional, nighttime, airport, peak and holiday development metrics are in [development_regimes.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/development_regimes.csv>) and [development_per_region.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/development_per_region.csv>); all 48 development leads are in [development_by_lead.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/development_by_lead.csv>). Gate failures, including worst regional deterioration, are in [development_gate_details.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/development_gate_details.csv>). Per-fold training time, artifact size and measured inference throughput are in [development_cost.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/development_cost.csv>).

May holiday/adjacent-day results below are **development diagnostics, not an independent holiday holdout**. A conditional gain can coexist with an insufficient pooled gain; it is not silently promoted because one difficult date improves.

Older-data pure XGBoost lowered Memorial Day WAPE from32.5152% to27.9239%, while adding richer holiday features scored30.0580%. The pooled holiday profile scored36.1930%, and fixed seasonal-anchor suppression scored30.0580%, versus original calendar feedback44.1099%. Thus the ordinary seasonal anchor can dilute an event-sensitive boosted forecast, but richer calendar features did not beat their same-history pure control on this date. None of these known-event observations establishes repeated-holiday or unseen-event superiority; the global promotion gate still failed.

| model | regime | n | WAPE | bias_pct |
| --- | --- | --- | --- | --- |
| pure_xgboost | holiday | 2640 | 32.5152 | 14.9497 |
| pure_xgboost | before_holiday | 2640 | 24.6207 | 19.0188 |
| pure_xgboost | after_holiday | 2280 | 15.0857 | -1.8415 |
| pure_lightgbm | holiday | 2640 | 36.6527 | 20.4935 |
| pure_lightgbm | before_holiday | 2640 | 27.7574 | 23.2343 |
| pure_lightgbm | after_holiday | 2280 | 15.1288 | 0.8874 |
| calendar_feedback | holiday | 2640 | 44.1099 | 21.0587 |
| calendar_feedback | before_holiday | 2640 | 26.8325 | 18.5953 |
| calendar_feedback | after_holiday | 2280 | 11.1097 | 3.6812 |
| history_expanding | holiday | 2640 | 27.9239 | 8.5099 |
| history_expanding | before_holiday | 2640 | 23.9320 | 16.8219 |
| history_expanding | after_holiday | 2280 | 15.9843 | -3.3775 |
| history_lightgbm | holiday | 2640 | 30.6164 | 13.3192 |
| history_lightgbm | before_holiday | 2640 | 26.1254 | 21.0017 |
| history_lightgbm | after_holiday | 2280 | 15.8104 | -0.1165 |
| history_calendar | holiday | 2640 | 43.6139 | 20.3155 |
| history_calendar | before_holiday | 2640 | 26.4510 | 17.5217 |
| history_calendar | after_holiday | 2280 | 11.1705 | 3.6546 |
| holiday_boost | holiday | 2640 | 30.0580 | 12.8998 |
| holiday_boost | before_holiday | 2640 | 24.2634 | 18.4412 |
| holiday_boost | after_holiday | 2280 | 15.9286 | -2.7315 |
| holiday_profile | holiday | 2640 | 36.1930 | 13.5994 |
| holiday_profile | before_holiday | 2640 | 26.8775 | 19.7758 |
| holiday_profile | after_holiday | 2280 | 10.5643 | 0.3645 |
| holiday_safety | holiday | 2640 | 30.0580 | 12.8998 |
| holiday_safety | before_holiday | 2640 | 26.9724 | 21.4762 |
| holiday_safety | after_holiday | 2280 | 15.9286 | -2.7315 |

## 6. Genuinely unseen tests

The protocol was written before acquiring final targets. June is the primary test: **55 origins ×30×48 = 79,200 points/model**. The separately prespecified July 1–7 event confirmation was not executed. A schema-only HTTP Range probe read10,292 bytes and zero trip rows from the [official July file](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2016-07.parquet): it contains PULocationID/DOLocationID and no coordinates. The coordinate API probe timed out and did not establish complete July coverage. Zone centroids or a crosswalk cannot exactly reproduce fixed point labels and original geographic filters. See [official_july_schema.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/official_july_schema.json>) and [secondary_availability.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/v4_20261102/secondary_availability.json>). This is a feasibility failure, not a zero-demand month, and no alternative event period was substituted. June and its origins were unchanged; no new target definition was introduced.

Primary WAPE is `100*sum(abs(actual-prediction))/sum(abs(actual))` across all original regional quarter counts. One-step, individual-lead and cumulative metrics are kept separate. Main origins rotate hours/days but begin on whole hours, matching prior protocols; :15/:30/:45 correctness is tested separately at the unit/API level, not represented as a full heldout quarter-origin robustness study.

| window | model | WAPE | MAE | RMSE | R2 | bias_pct | WAPE_CI | offline_ms | MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| june_primary | pure_xgboost | 11.7512 | 14.7384 | 25.4960 | 0.9552 | -1.4891 | [10.711, 12.9021] | 38.9869 | 5.3170 |
| june_primary | pure_lightgbm | 11.8268 | 14.8332 | 24.7645 | 0.9577 | -0.1317 | [10.7799, 13.0011] | 32.7275 | 2.8717 |
| june_primary | calendar_feedback | 11.6086 | 14.5595 | 23.9458 | 0.9605 | 2.5667 | [10.7556, 12.533] | 27.1773 | 5.3850 |
| june_primary | seasonal_weekly | 15.5710 | 19.5292 | 33.6046 | 0.9222 | -1.4633 | [13.3411, 18.613] | 12.8432 | 0.0005 |

Paired gains below are reference-minus-selected WAPE points; positive favors the frozen selection. Resampling uses1,000 three-origin blocks. One month cannot establish seasonal robustness, and no unseen holiday score is available. An overlapping interval or a slightly better comparator point estimate is not conclusive superiority and does not trigger retrospective reselection.

June's observed advantage for the selected model is small: reference-minus-selected gain is0.1427 points against XGBoost, paired95% interval[−0.6843,1.0425], and0.2182 against LightGBM, interval[−0.5970,1.0971]. **Neither establishes a conclusive ranking among the three strong systems.** Blocks preserve within-path and nearby-origin dependence; longer weekly dependencies and fitted-model/source uncertainty are not fully represented.

| window | reference | gain | CI |
| --- | --- | --- | --- |
| june_primary | pure_xgboost | 0.1427 | [-0.6843, 1.0425] |
| june_primary | pure_lightgbm | 0.2182 | [-0.597, 1.0971] |
| june_primary | calendar_feedback | 0.0000 | [0.0, 0.0] |
| june_primary | seasonal_weekly | 3.9624 | [1.8095, 6.9489] |

![June lead and origin comparison](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/final_horizon_and_event.png>)

### Cumulative versus individual-lead WAPE

| window | model | step | cumulative_WAPE | individual_WAPE |
| --- | --- | --- | --- | --- |
| june_primary | pure_xgboost | 1 | 9.7494 | 9.7494 |
| june_primary | pure_xgboost | 4 | 10.1529 | 10.7498 |
| june_primary | pure_xgboost | 12 | 11.0866 | 10.8631 |
| june_primary | pure_xgboost | 24 | 11.2196 | 10.8508 |
| june_primary | pure_xgboost | 48 | 11.7512 | 13.0932 |
| june_primary | pure_lightgbm | 1 | 9.7089 | 9.7089 |
| june_primary | pure_lightgbm | 4 | 10.2805 | 10.9778 |
| june_primary | pure_lightgbm | 12 | 11.3419 | 11.3881 |
| june_primary | pure_lightgbm | 24 | 11.4777 | 11.1150 |
| june_primary | pure_lightgbm | 48 | 11.8268 | 12.9758 |
| june_primary | calendar_feedback | 1 | 9.7507 | 9.7507 |
| june_primary | calendar_feedback | 4 | 10.1260 | 10.6897 |
| june_primary | calendar_feedback | 12 | 11.1766 | 11.3721 |
| june_primary | calendar_feedback | 24 | 11.4557 | 10.8189 |
| june_primary | calendar_feedback | 48 | 11.6086 | 13.5862 |
| june_primary | seasonal_weekly | 1 | 18.6495 | 18.6495 |
| june_primary | seasonal_weekly | 4 | 17.0539 | 16.1201 |
| june_primary | seasonal_weekly | 12 | 16.4304 | 15.9630 |
| june_primary | seasonal_weekly | 24 | 15.6584 | 13.9184 |
| june_primary | seasonal_weekly | 48 | 15.5710 | 18.4952 |

All48 individual leads, signed bias and every origin are in [final_by_lead.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/final_by_lead.csv>) and [final_by_origin.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/final_by_origin.csv>). Score intervals above are not probabilistic bands for individual future counts.

## 7. Regional and temporal behavior

Selected-system slices use identical final points and common Jan-May training-only thresholds. Night is00:00–05:59 local; sparse means train average<=10; demand tiers are thirds of train regional means; peak thresholds are regional training95th percentiles. Realized peaks condition on future outcomes and cannot be used as a forecast-time selector.

| window | regime | n | WAPE | MAE | RMSE | bias_pct |
| --- | --- | --- | --- | --- | --- | --- |
| june_primary | night | 19800 | 15.2334 | 9.3278 | 17.9909 | -1.9930 |
| june_primary | day | 59400 | 11.1046 | 16.3034 | 25.6251 | 3.2006 |
| june_primary | weekday | 58080 | 10.6184 | 13.4049 | 21.9779 | 1.5286 |
| june_primary | weekend | 21120 | 14.3996 | 17.7346 | 28.6694 | 5.4927 |
| june_primary | holiday | 0 | NA | NA | NA | NA |
| june_primary | day_before_holiday | 0 | NA | NA | NA | NA |
| june_primary | day_after_holiday | 0 | NA | NA | NA | NA |
| june_primary | realized_train_p95_peak | 2826 | 13.0474 | 28.6179 | 43.2417 | -9.7165 |
| june_primary | sparse_regions | 7920 | 34.1265 | 2.2317 | 3.3253 | -0.9208 |
| june_primary | airports_3_6 | 5280 | 17.7493 | 16.8374 | 24.6785 | -2.8117 |
| june_primary | low_tier | 26400 | 21.0270 | 5.3011 | 9.3491 | -1.7848 |
| june_primary | medium_tier | 26400 | 11.8028 | 15.2744 | 22.2298 | 1.6537 |
| june_primary | high_tier | 26400 | 10.4238 | 23.1029 | 33.7437 | 3.5947 |

On June, nighttime WAPE remains **15.2334%**, versus **11.1046%** by day; weekends score **14.3996%**, versus **10.6184%** on weekdays. Airport-centered regions together score **17.7493%** and the sparse group **34.1265%**. Region2 has38.4291% WAPE but MAE1.5538, illustrating the small-denominator issue. Realized peaks score **13.0474%**, with **-9.7165%** conditional bias; future-actual peak conditioning retains the selection caveat. Overall positive bias2.5667% and negative realized-peak bias coexist, so a uniform correction is not justified by these slices. No June residual was used to fit a correction.

June contains **no US federal holiday**; its holiday score is NA, not zero error. July was not evaluated with a substituted target. No June date, region or origin was removed for high error. All models' regime and hourly tables are [final_regimes.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/final_regimes.csv>) and [final_by_hour.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/final_by_hour.csv>).

June WAPE for every region:

| region | XGBoost | LightGBM | calendar | selected |
| --- | --- | --- | --- | --- |
| 0 | 18.3830 | 19.5301 | 18.8479 | 18.8479 |
| 1 | 10.8416 | 10.9580 | 10.5152 | 10.5152 |
| 2 | 41.0645 | 50.2686 | 38.4291 | 38.4291 |
| 3 | 17.7656 | 17.9289 | 17.1391 | 17.1391 |
| 4 | 11.3103 | 11.3178 | 11.3210 | 11.3210 |
| 5 | 11.4594 | 11.4007 | 11.3641 | 11.3641 |
| 6 | 19.5013 | 19.1931 | 18.2351 | 18.2351 |
| 7 | 10.2006 | 10.0699 | 11.0331 | 11.0331 |
| 8 | 9.4930 | 9.9110 | 10.7613 | 10.7613 |
| 9 | 25.4511 | 27.2682 | 22.6499 | 22.6499 |
| 10 | 14.1850 | 13.7713 | 10.9820 | 10.9820 |
| 11 | 9.9677 | 10.1611 | 10.6563 | 10.6563 |
| 12 | 11.2436 | 10.9832 | 10.4272 | 10.4272 |
| 13 | 22.3277 | 22.9473 | 21.4106 | 21.4106 |
| 14 | 11.0862 | 11.0158 | 10.7992 | 10.7992 |
| 15 | 10.7674 | 10.6900 | 11.0907 | 11.0907 |
| 16 | 25.5600 | 28.2623 | 21.8644 | 21.8644 |
| 17 | 26.1671 | 26.8424 | 25.4875 | 25.4875 |
| 18 | 9.2941 | 8.7789 | 8.5867 | 8.5867 |
| 19 | 10.6458 | 10.9848 | 11.2621 | 11.2621 |
| 20 | 10.7079 | 10.6801 | 9.8832 | 9.8832 |
| 21 | 8.9592 | 8.8256 | 9.7172 | 9.7172 |
| 22 | 10.5038 | 10.8293 | 11.0883 | 11.0883 |
| 23 | 11.1523 | 11.6870 | 12.1811 | 12.1811 |
| 24 | 35.6488 | 39.0127 | 34.6981 | 34.6981 |
| 25 | 13.8609 | 14.4473 | 13.1761 | 13.1761 |
| 26 | 12.6735 | 12.6673 | 12.3617 | 12.3617 |
| 27 | 9.9687 | 10.2990 | 10.7609 | 10.7609 |
| 28 | 33.4892 | 35.9272 | 31.4087 | 31.4087 |
| 29 | 24.5375 | 25.1801 | 23.4533 | 23.4533 |

Full per-region MAE/RMSE/R²/bias, mean/CV/zeros and nighttime/peak/holiday behavior for both windows are in [final_per_region.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/final_per_region.csv>). R² pooled across heterogeneous regions is not a guarantee of accurate low-count or holiday forecasts.

![Fixed middle-June-origin path examples](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/fixed_june_examples.png>)

The example uses the predetermined middle June origin and IDs2/3/6/18. All origins determine scores; examples were not selected for favorable errors.

## 8. Provenance, data quality and target constraints

The original scaler/30 centroids and retained-trip filters are unchanged. Official coordinate records: [2015 NYC Yellow Taxi](https://data.cityofnewyork.us/Transportation/2015-Yellow-Taxi-Trip-Data/2yzn-sicd), [2016 NYC Yellow Taxi](https://data.cityofnewyork.us/Transportation/2016-Yellow-Taxi-Trip-Data/uacg-pexx). The 2015 timestamp is `pickup_datetime`; 2016 uses `tpep_pickup_datetime`. All are interpreted as New York local calendar times. Acquisition starts after the ambiguous2015 fall-DST day; existing2016 spring-DST quarantine is preserved. Training windows crossing absent historical days are excluded and explicitly tested. Modern taxi-zone-ID files are not substituted for coordinate clusters.

Only three projected fields are transferred after applying the original pickup/dropoff latitude40.60–40.85, longitude−74.05–−73.70, fare0.50–81 and distance0.25–24.43 filters. Daily responses are verified nonempty and below the one-million-record cap, assigned locally with the fixed mapping, aggregated and discarded. Requests, retrieval times and response hashes remain saved. Coincident timestamp/coordinate triples may represent different trips and are not falsely deduplicated. Day requests do not overlap; aggregate region/time keys must be unique. Within acquired complete days, all 30 regions and96 quarters are reindexed with genuine zero counts under source-completeness assumptions. Unacquired days are not filled. Raw source outages cannot be excluded just by a regular panel.

The bounded source inventory is [source_daily_inventory.csv](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/source_daily_inventory.csv>); per-period provenance/hashes are in the new artifact folder. Initial server-side aggregation timed out. A later exact region-2 day probe matched existing counts but took134 seconds, and its month query timed out; it was not used to relabel source data. CSV round-trip parsing reduces overhead. Failed schema/DNS/timeout attempts and scope amendments are preserved; no malformed/partial response becomes a dataset. Historical context and final data are isolated from fitting/selection functions.

Post-acquisition checks in [data_quality_checks.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/data_quality_checks.json>) verify all expected acquired day/region/quarter rows, unique keys, finite nonnegative counts, source-count totals, panel fingerprints and the original geography fingerprint. History has38 acquired days with intentional gaps; June has30 complete acquired days. These checks do not certify the absence of upstream raw-data omissions.

The target is **filtered completed trips**, not all pickups/requests/unserved demand. The prior full January audit found10,505,157 retained out of10,906,858 raw records, excluding about8.65% of in-geography region3 and3.60% of region6 pickups. Fare/distance/dropoff exclusion reasons overlap. No target definition changed in this phase. A separate all-pickup target needs separate data, baseline and metric protocol.

There is also an information-latency limitation: the [official TLC source](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) is historical, published with delay, and completed-trip filters depend on post-pickup information. These experiments assume the finalized preceding target counts are already available at each simulated origin. Strict chronological exclusion of future rows is tested, but it does **not** certify a2016 issued live feed or finalized counts immediately after pickup. An operational as-of evaluation needs source vintages/latency or an explicit observation-delay/target track. The reported benchmark must not be presented as verified live production accuracy.

No future observed weather/reanalysis, realized flight delays or attendance enters any primary forecast. [BTS metadata](https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ) distinguishes scheduled arrival time from realized delay, but an archived schedule field alone does not establish which schedule version was known at origin. No such as-of archive was verified, so no unnecessary external flight/event pipeline was built. Airport behavior is measured, not assigned an invented explanation.

## 9. Engineering, latency and preservation

The isolated review artifact is [candidate_v4.joblib](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/artifacts/v4_20261102/candidate_v4.joblib>); use [README.md](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/urbanflow/ml/v4/README.md>) for reproduction and explicit localhost opt-in. Prior candidate artifacts remain available; the default production model was not switched. The selected schema lists actual component feature columns rather than silently reusing an older feature union. Data, models, MLflow and reports live in new phase directories.

Final regression: **118 passed, 5 warnings in 80.72s (0:01:20)**, recorded in [v4_final_regression_tests.log](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_final_regression_tests.log>). New checks cover missing-day history exclusion, exact feature-cache equivalence, retry reuse without fabricated counts, coincident-trip preservation, scalar arithmetic equivalence, future poisoning and reconciliation. Actual localhost API validation passed **45 valid and 10 invalid requests**, all 30 regions, required horizons1/4/12/24/48, :15/:30/:45 origins, weekday/weekend/end-of-month, exact horizon prefixes, strict past history, invalid timestamps and training-cutoff checks. July4 is rejected for missing genuine history; its deterministic holiday features are unit-tested without claiming a measured holiday forecast. Loaded process-tree RSS was **444.77 MiB**; this includes libraries, not just the artifact.

| horizon | n | p50_ms | p95_ms |
| --- | --- | --- | --- |
| 1 | 3 | 112.6952 | 122.2562 |
| 4 | 3 | 138.7648 | 173.1506 |
| 12 | 33 | 258.0176 | 303.2592 |
| 24 | 3 | 378.4861 | 428.5961 |
| 48 | 3 | 704.9535 | 719.0122 |

Offline per-origin times in the comparison table are batched all-region throughput, not HTTP latency. HTTP samples are small and include cold-start/history-loading effects. Conservation holds internally before two-decimal API rounding. No live data, calibrated prediction intervals, production monitoring or operational readiness is claimed. Temporary API processes were stopped.

Preservation hashes/source-freeze checks and file inventory are in [preservation_checks.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/preservation_checks.json>) and [changed_files.json](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/changed_files.json>). Earlier code, models, notebooks, reports and user changes were not overwritten. No commit, push, PR, remote modification, deployment or frontend change was performed.

A pre-freeze orchestration sequencing error was caught while matched-data controls were being finalized. The run was stopped before selection or June acquisition. Earlier per-month predictions/logs were retained; every reused path was reproduced with **zero maximum difference** under the complete final source, and merged `development_complete_*` files determined the sole final selection. See [pre_freeze_resumption.md](<C:/Users/Soham/Documents/UBER DEMAND PREDICATION/reports/v4_20261102/pre_freeze_resumption.md>). This did not change a final-test result or period.

## 10. Recommendation

The defensible local selection for this phase is **calendar_feedback**, selected from three spent development months before final access. Its June result is **11.6086%**; independent holiday performance remains unconfirmed because the planned July source was not compatible. Preserve the earlier candidate/default pending review; a newer engineering effort is not by itself a reason to adopt. Use paired intervals, regional behavior and serving cost above to judge whether any point-estimate difference warrants adoption. A better heldout comparator point estimate remains evidence for the next hypothesis, not authority to rewrite this frozen selection.

Reject the tested coarse/disaggregated/alignment and sparse variants when they fail the frozen gate; do not reinterpret an aggregated-only score as success. Limited historical examples can test modest coverage and pooled calendar effects but cannot establish annual or repeated-summer-holiday robustness. The next priority is representative historical/event coverage and operational target/latency agreement, followed by fresh chronological confirmation of any retained challenger. Neural or graph complexity remains unjustified without a positive added-information ablation and a matched strong-baseline comparison.

June is now spent. July was not evaluated as a compatible target and must not be described as a passed holiday test. Further optimization requires a genuinely available compatible chronological source, or a separately named zone-based/all-pickup target track with its own baselines and protocol. It cannot silently substitute zone assignments for these30 coordinate clusters. Nothing in this phase guarantees below10% across seasons, and no mathematical irreducible error floor was established.
