# UrbanFlow AI · Model Experimentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)

**A reproducible research pipeline for regional urban-demand forecasting:
66.9 million retained historical trips, causal multi-step inference, and a frozen
June 2016 evaluation.**

This repository preserves the research behind UrbanFlow AI's 30-region, 15-minute NYC taxi demand forecasting model: original notebooks, causal feature code, bounded model comparisons, protocols, frozen artifacts and evaluation evidence. The [deployment platform](https://github.com/Sohamgaonkhadkar/UrbanFlow-AI-Spatiotemporal-Demand-Forecasting-Platform/tree/main) serves the selected model through FastAPI and the finalized cinematic frontend.

The selected **v4 calendar-feedback** model combines gradient boosting,
training-only calendar structure and causal recursive feedback. This repository
provides the evidence needed to inspect the result: ordered feature schemas,
selection protocols, acquisition provenance, frozen artifacts, sealed prediction
arrays and executable validation tests.

**Maps:** [Original pickup-cluster image](https://raw.githubusercontent.com/Sohamgaonkhadkar/UBER-DEMAND-PREDICTION/codex/final-v4-integration-20261102/docs/legacy/pickup-clusters-original.png) · [Verified geographic centroids](https://raw.githubusercontent.com/Sohamgaonkhadkar/UBER-DEMAND-PREDICTION/codex/final-v4-integration-20261102/docs/figures/verified-region-centroids.png) · [Map gallery](#maps-and-research-figures)

## Project at a glance

| Dimension | Verified scope |
|---|---|
| Research question | Forecast retained completed Yellow Taxi trips across regional time series |
| Observation panel | 66,874,467 retained trips; 524,040 region/time rows; January–June 2016 |
| Spatial / temporal resolution | 30 fixed K-means regions / 15-minute bins |
| Forecast trajectory | Up to 48 steps, covering 12 hours |
| Selected model | XGBoost with recursive calendar feedback; 46 fitted features |
| Primary result | 9.7507% WAPE at one step; 11.6086% cumulative WAPE over 48 steps |
| Primary evaluation | 55 non-overlapping June origins; 79,200 region/lead predictions |
| Validation | 118 research tests passed in the recorded integration run |

The project demonstrates chronological evaluation, training/inference feature
agreement, original-scale count forecasting, regional consistency, and a clear
boundary between experimental evidence and application serving. The benchmark
is historical; it does not establish current operational performance.

## Contents

- [Final result](#final-result)
- [Maps and research figures](#maps-and-research-figures)
- [Mathematical formulation](#mathematical-formulation)
- [Selected architecture](#selected-architecture)
- [Research sequence and evidence](#research-sequence-and-evidence)
- [Dataset and target definition](#dataset-and-target-definition)
- [Layout](#layout)
- [Reproduce validation](#reproduce-validation)
- [Research directions](#research-directions)
- [Operational relationship](#operational-relationship)
- [License and third-party materials](#license-and-third-party-materials)


## Final result

| Horizon, cumulative | WAPE |
|---|---:|
| 15 minutes / 1 step | 9.7507% |
| 1 hour / 4 steps | 10.1260% |
| 3 hours / 12 steps | 11.1766% |
| 6 hours / 24 steps | 11.4557% |
| 12 hours / 48 steps | 11.6086% |

At 12 hours: **MAE 14.5595**, **RMSE 23.9458**, **R² 0.960489**, **bias +2.5667%**. The June 2016 primary evaluation comprises **55** non-overlapping origins spaced 13 elapsed hours apart, **30** regions and **48** leads: **79,200 predictions**. The individual lead-48 WAPE is **13.5862%**. Cumulative 12-hour WAPE has a 3-origin moving-block bootstrap 95% interval of **10.7556–12.5330%** (1,000 draws, seed 42). That interval quantifies metric uncertainty, not prediction uncertainty for individual counts.

WAPE is `100 × Σ|prediction − actual| / Σ|actual|`. Lower is better; these values are error rates, not a classification-accuracy percentage. No model achieved a sub-10% cumulative 12-hour WAPE. Results for April, May and June use different cutoffs and periods; differences between those months are not paired model improvements.

## Maps and research figures

### Original pickup-cluster visualization

![Original standardized-coordinate pickup-cluster figure](https://raw.githubusercontent.com/Sohamgaonkhadkar/UBER-DEMAND-PREDICTION/codex/final-v4-integration-20261102/docs/legacy/pickup-clusters-original.png)

The original spatial figure from the earlier README is preserved byte-for-byte.
Its axes show transformed coordinates, not geographic degrees; the original
notebook labels are retained as historical material. The current model's physical
centroid coordinates are shown in the verified figure below.

### Verified geographic centroids

![All 30 fitted regional centroids and central NYC detail](https://raw.githubusercontent.com/Sohamgaonkhadkar/UBER-DEMAND-PREDICTION/codex/final-v4-integration-20261102/docs/figures/verified-region-centroids.png)

This figure inverse-transforms the frozen scaler and K-means centers. Numbers are
the actual model region IDs; the dashed box selects the detail panel. These are
point centers, not authoritative cluster polygons or demand-intensity values.
The reproducible [figure script](docs/figures/render_centroids.py) reads
`artifacts/advanced_20261102/geography.json` without fitting or changing a model.

### Final June evaluation

![June error by individual lead and variability across prespecified origins](reports/v4_20261102/final_horizon_and_event.png)

The left panel shows **individual-lead WAPE**, while the right panel shows each
origin's **cumulative 48-step WAPE**. Neither curve should be confused with the
single pooled cumulative 12-hour score of 11.6086%. The figure's original
"untouched June" title describes its status at the selection freeze; June is now
spent final-evaluation evidence. The prespecified July check was unavailable.

![Observed versus forecast demand for four fixed June regional paths](reports/v4_20261102/fixed_june_examples.png)

Fixed June examples compare actual retained trips with calendar-feedback and
pure-XGBoost predictions. They illustrate regional trajectories; the complete
55-origin evaluation determines the reported aggregate scores.

<details>
<summary>View the March–May development comparison</summary>

![Development WAPE across the three spent chronological windows](reports/v4_20261102/development_results.png)

These are development comparisons with their respective training cutoffs,
not replacements for the frozen June evaluation. Their different observation
windows do not support an improvement claim based on comparing one month with
another.

</details>

Original image locations, README revisions and SHA-256 hashes are recorded in
[historical visual provenance](docs/legacy/provenance.json).

## Selected architecture

The canonical artifact is `artifacts/v4_20261102/candidate_v4.joblib`, candidate ID `v4_calendar_feedback`. The selected specification combines:

1. A global XGBoost absolute-error component, 599 fitted boosted trees and 46 ordered fitted features.
2. A training-only weekday/quarter-hour calendar profile with origin-local gain adaptation.
3. Causal 15-minute recursion across all 30 regions, feeding the final blended result into subsequent lags.
4. A gradual seasonal anchor and calendar weight growing with lead, retaining the exact frozen implementation.

The fitted schema is `feature_schema.json`; the artifact's legacy top-level feature union is not the fitted 46-column schema. Runtime compatibility classes live under `urbanflow/ml`. The deployment repository extracts only the selected inference classes and preserves their module names and forecast arithmetic, avoiding training-library imports in the serving application. Archived June prediction parity verifies that extraction.

Training labels end at **May 31, 2016 21:45 America/New_York**. The exclusive training cutoff is **22:00**, retaining the protocol's 2-hour gap. Warmup needs 1,536 bins (16 days). Training labels therefore begin January 17. Feature calculations and origin adaptation use observations strictly before each origin. Later June observations can legitimately become history for later June origins; they never alter fitted weights or selection.


```mermaid
flowchart LR
  Source[Historical coordinate-bearing taxi records] --> Panel[Fixed regions and 15-minute counts]
  Panel --> Features[Causal features and calendar context]
  Features --> Development[Chronological development comparisons]
  Development --> Freeze[Frozen selection and May cutoff]
  Freeze --> June[Sealed June evaluation]
  June --> Artifact[Versioned artifact and evidence]
  Artifact --> Serving[Separate deployment platform]
```

This diagram describes the completed research sequence. June outcomes are now
spent evaluation evidence and are not an untouched selection window for future
experiments.

## Mathematical formulation

The equations below restore the mathematical explanation from the original
README and align it with the selected v4 implementation. They describe the
current target, fitted geography, causal features and recursion; earlier
LightGBM-specific feature lists and scores are not substituted for v4 behavior.

### 1. Regional count target

For retained trip records indexed by $`i`$, let $`p_i`$ be the pickup coordinate,
$`t_i`$ the pickup timestamp, and $`a(p_i)`$ the fixed region assignment. With
$`\Delta=15`$ minutes and $`r\in\{0,\ldots,29\}`$:

```math
y_{r,t}=\sum_i \mathbf{1}[a(p_i)=r]\,\mathbf{1}[t\leq t_i<t+\Delta].
```

The count is formed after the original fare, distance and **both pickup and
dropoff** geographic filters. The post-trip availability caveat in the dataset
section still applies. The 48-step forecast begins at origin $`\tau`$ and covers
the bins starting at $`\tau,\tau+\Delta,\ldots,\tau+47\Delta`$.

### 2. Standardization, clustering and geographic reconstruction

Coordinate order is explicitly longitude then latitude:

```math
p_i=\left[\mathrm{lon}_i,\mathrm{lat}_i\right]^{\mathsf{T}},
\qquad z_i=(p_i-\mu)\oslash\sigma.
```

Here $`\mu`$ and $`\sigma`$ are the frozen scaler mean and scale, and $`\oslash`$
denotes elementwise division. The K-means objective underlying the original
spatial clustering is:

```math
\min_{S_0,\ldots,S_{29}}\;\sum_{r=0}^{29}\sum_{z_i\in S_r}
\Vert z_i-c_r\Vert_2^2.
```

The fitted mapping used for regional aggregation and centroid display is:

```math
a(p_i)=\underset{r\in\{0,\ldots,29\}}{\mathrm{argmin}}
\Vert z_i-c_r\Vert_2^2,
\qquad p_r^{\mathrm{center}}=\mu+\sigma\odot c_r.
```

The centers are fixed for v4; inference does not recluster the city. Mini-batch
updates approximate this objective and do not guarantee the identical solution
of an unrelated full-batch K-means run. The map above uses the inverse transform,
not the standardized axes from the earlier visualization.

### 3. Causal lags, rolling statistics and EWMA

Let $`\widetilde y`$ denote the available history: observations strictly before
the origin and final model predictions at subsequent recursive steps. Then:

```math
\mathrm{lag}_{\ell}(r,t)=\widetilde y_{r,t-\ell\Delta},
\qquad
m_{r,t}^{(w)}=\frac{1}{w}\sum_{j=1}^{w}\widetilde y_{r,t-j\Delta}.
```

```math
s_{r,t}^{(w)}=
\sqrt{\frac{1}{w-1}\sum_{j=1}^{w}
\left(\widetilde y_{r,t-j\Delta}-m_{r,t}^{(w)}\right)^2}.
```

The fitted extended lag set is
$`\ell\in\{1,2,3,4,8,12,24,48,96,192,288,672,1344\}`$.
Rolling mean/sample-standard-deviation windows are $`w\in\{4,24,96,672\}`$.
At this resolution **24 steps are six hours**; 96 and 672 elapsed steps are
24 hours and seven days. Separate seasonal features use New York civil-clock
lookups so daylight-saving transitions are handled explicitly.

The original README's standard EWMA recurrence was
$`m_t=\alpha y_{t-1}+(1-\alpha)m_{t-1}`$.
The actual v4 `ewma` feature uses a normalized, finite 32-bin history:

```math
\mathrm{ewma}_{r,t}=
\frac{\sum_{j=0}^{31}0.6^j\,\widetilde y_{r,t-(j+1)\Delta}}
{\sum_{j=0}^{31}0.6^j}.
```

This is the implemented weighting, rather than either conflicting alpha value
from the earlier description. No missing history is silently filled to satisfy
the 1,536-bin warmup requirement.

### 4. Cyclic time encoding

For the New York quarter-hour slot $`q=4h+\lfloor m/15\rfloor`$ and weekday $`d`$:

```math
q_{\sin}=\sin\left(\frac{2\pi q}{96}\right),\quad
q_{\cos}=\cos\left(\frac{2\pi q}{96}\right),\qquad
d_{\sin}=\sin\left(\frac{2\pi d}{7}\right),\quad
d_{\cos}=\cos\left(\frac{2\pi d}{7}\right).
```

The ordered [feature schema](artifacts/v4_20261102/feature_schema.json) remains
authoritative. The selected configuration enables seasonal/calendar features;
optional neighbor-feature, interaction, extra-statistic and weather families are
disabled in this frozen candidate.

### 5. Calendar adaptation and final recursive feedback

Let $`P_r(t)`$ be the fixed training-only weekday/quarter-hour calendar profile.
Its origin-local gain uses only the eight bins preceding $`\tau`$:

```math
g_{r,\tau}=\mathrm{clip}\left(
\frac{\sum_{j=1}^{8}y_{r,\tau-j\Delta}+100}
{\sum_{j=1}^{8}P_r(\tau-j\Delta)+100},\;0.6,\;1.6\right).
```

For zero-based lead $`k\in\{0,\ldots,47\}`$, the adapted calendar component is:

```math
c_{r,k}=P_r(\tau+k\Delta)
\left[1+(g_{r,\tau}-1)e^{-k/12}\right].
```

Let $`b_{r,k}`$ be `base.predict` on the current causal features, expressed on the
original count scale. Let $`s_{r,k}`$ be the seasonal anchor, combining smoothed
same-clock profiles one, seven and fourteen days earlier with weights
$`0.25,0.50,0.25`$. The stored v4 coefficients give:

```math
\lambda_k=0.3\frac{k}{47},\qquad
a_{r,k}=(1-\lambda_k)b_{r,k}+\lambda_k s_{r,k},\qquad
w_k=0.6\left(1-e^{-k/12}\right).
```

```math
\widehat y_{r,k}=\max\left(0,
(1-w_k)a_{r,k}+w_k c_{r,k}\right),\qquad
\widetilde y_{r,\tau+k\Delta}\leftarrow\widehat y_{r,k}.
```

The selected base has no additional bias correction. Every **final blended
prediction** is appended before computing the next step. At $`k=0`$ the two blend
weights are zero. This is causal feedback, not a blend applied only after an
independent 48-step trajectory has already been generated.

### 6. Evaluation equations

For $`N`$ evaluated origin/region/lead observations and $`e_i=\widehat y_i-y_i`$:

```math
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|e_i|,\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}e_i^2}.
```

```math
\mathrm{WAPE}(\%)=100\,
\frac{\sum_{i=1}^{N}|e_i|}{\sum_{i=1}^{N}|y_i|},\qquad
R^2=1-\frac{\sum_{i=1}^{N}e_i^2}
{\sum_{i=1}^{N}(y_i-\overline{y})^2}.
```

The primary June aggregate uses $`N=55\times30\times48=79{,}200`$.
Cumulative horizon-$`h`$ metrics pool leads $`0`$ through $`h-1`$; individual-lead
metrics restrict evaluation to a single lead. WAPE requires positive aggregate
actual demand; $`R^2`$ requires nonzero actual variance. A zero denominator does
not become a fabricated zero-error score. WAPE is an error percentage, not a
classification accuracy, and its bootstrap interval is not a prediction band.

## Research sequence and evidence

The experiments retain chronological development, causal feature construction and explicit final-test sealing. March, April and May were spent development windows. June was the primary final month after freezing selection; it is now spent evaluation evidence and must not be described as an untouched test for future experiments.

The v4 study examined 17 bounded recipes across three development windows, including extra historical holiday profiles, temporal aggregation/disaggregation and reconciliation, expanding/rolling/decayed history, sparse-count objectives and calendar alternatives. Novel recipes did not clear the prespecified material improvement threshold of 0.1 WAPE point with the required paired confidence and regional/monthly checks. The strongest existing calendar-feedback specification was retained and fit at the frozen May cutoff.

The frozen June comparison reports:

| Model | Cumulative 12-hour WAPE |
|---|---:|
| Selected calendar-feedback model | **11.6086%** |
| Pure XGBoost | 11.7512% |
| Pure LightGBM | 11.8268% |

Paired differences from the selected model have confidence intervals crossing
zero; the small ranking differences are not conclusive statistical superiority.
Sparse regions, airports, weekends and nights remain more difficult.
Actual-peak-conditioned diagnostics are retrospective subgroup summaries,
not features available at an origin.

The prespecified July 1–7 holiday confirmation was **not executed**: compatible complete coordinate-bearing source coverage could not be established. The official July schema inspected in the study used taxi-zone IDs, which cannot exactly reproduce the original point mapping and pickup/dropoff filters. No replacement period was substituted. See the availability ledger and frozen report for the bounded probes and their limitations.

Start with these records:

- [Complete v4 report](reports/research_driven_forecasting_v4.md)
- [artifacts/v4_20261102/protocol.json](artifacts/v4_20261102/protocol.json) — test and selection protocol, original protected hashes.
- [artifacts/v4_20261102/selection_freeze.json](artifacts/v4_20261102/selection_freeze.json) — exact frozen selection and secondary-test availability.
- [artifacts/v4_20261102/training_metadata.json](artifacts/v4_20261102/training_metadata.json) — cutoffs and fitted component metadata.
- [artifacts/v4_20261102/final_test_results.json](artifacts/v4_20261102/final_test_results.json) — complete primary/secondary evaluation ledger.
- [artifacts/v4_20261102/june_primary_predictions.npz](artifacts/v4_20261102/june_primary_predictions.npz) — sealed truth and forecast arrays.
- [artifacts/v4_20261102/frozen_source/](artifacts/v4_20261102/frozen_source/) — original frozen study source snapshots.
- [reports/v4_20261102/](reports/v4_20261102/) — acquisition, environment, sensitivity and system evidence.

## Dataset and target definition

The target is **retained completed Yellow Taxi trips per fitted region per 15-minute bin**, not all requests, all NYC pickups or latent unmet demand. Original coordinate-bearing records use fixed pickup/dropoff geographic bounds (latitude 40.60–40.85, longitude −74.05–−73.70), fare $0.50–81.00 and trip-distance 0.25–24.43 mile filters. Both pickup and dropoff must satisfy the geographic bounds. June was acquired from NYC Open Data dataset `uacg-pexx` using bounded, reusable day intervals; compatible historical sources and exact probes are recorded in the acquisition provenance.

The original fitted scaler and 30 K-means centers remain fixed. `artifacts/advanced_20261102/geography.json` preserves their mean, scale and centers. Longitude is the first coordinate. Inverse-transforming gives the verified centroid for each model region ID. Neighborhood labels from earlier interfaces were not authoritative cluster identities; the serving application uses generic sector names and does not invent cluster polygons.

The genuine dense panel spans January 1 through June 30, 2016, with zero bins retained under source-completeness assumptions and historical DST handling documented in the report. The canonical deployment observation panel is included. Research raw trip files, daily downloads and feature caches are deliberately omitted from Git. Intermediate exclusions and original hashes are recorded in `provenance/excluded_generated_artifacts.json` and `provenance/original_workspace.json`.

Fare/distance/completion filters require post-trip information. Immediate availability of a completed-bin count is an idealized benchmark assumption. No independent production as-of feed certification, current-domain generalization, weather contribution, calibrated prediction intervals, validated whole-hybrid attribution or operational alerts is established by this work. July holiday robustness remains unvalidated.

## Layout

```text
notebooks/                       Original research notebooks, including local improvements
urbanflow/ml/advanced/           Causal features, comparisons and calendar-feedback model
urbanflow/ml/nextgen/            Profile/adaptation/calibration research
urbanflow/ml/v4/                 Acquisition, alternatives, frozen protocol and tests
urbanflow/backend/               Preserved historical research/API compatibility source
urbanflow/tests/                 Preserved core research and backend tests
artifacts/                       Canonical candidates, panels, schemas and numeric evidence
reports/                         Detailed research reports and diagnostics
provenance/                      Original hashes/status and generated-output exclusion ledger
requirements-local-lock.txt      Exact tested research environment
pytest.ini                      Portable package/test discovery
LICENSE                         MIT license for original software/documentation
THIRD_PARTY_NOTICES.md           Dataset, dependency and embedded-material notices
```

The `urbanflow/ml` package depth is intentional: root-relative research paths resolve within this repository. Experimental source, serving compatibility code and versioned evidence remain separate from generated caches and raw downloads. Earlier notebook versions remain available in Git history.

## Reproduce validation

Use Python **3.14**. Clone the repository's `main` branch for the completed v4 research and frozen evidence:

```powershell
git clone --branch main https://github.com/Sohamgaonkhadkar/UBER-DEMAND-PREDICTION.git Model-Experimentation
cd Model-Experimentation
```

Create the research environment and execute validation from the repository root:


```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-local-lock.txt
.venv\Scripts\python.exe -m pytest -q
```

The suite covers causal lags, batch/training feature agreement, recursive joint feedback, future poisoning, original-scale transforms, calendar behavior, DST handling, API validation, coarse temporal coherence, acquisition retries and cached feature identity. The executed integration validation passed **106 core tests plus 12 v4 tests**. Detailed commands, environment and warnings are in `FINAL_INTEGRATION_REPORT.md`.

For application inference and the minimal dependency set, use the separate deployment repository; it requires neither the full experiment stack nor raw data. Its acceptance tests independently recompute June metrics and compare the portable frozen artifact with sealed predictions.

To review or reproduce a research procedure, inspect `urbanflow/ml/v4/protocol.py` and the corresponding module before execution. Run modules with the original package root:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'urbanflow'
# Read the frozen protocol and source first. Acquisition can be expensive.
.venv\Scripts\python.exe -m ml.v4.availability
```

Acquisition/research entry points may contact public data services, create caches, consume substantial resources, refit models or require excluded raw input. They are preserved for reproducibility and are **not** automatically run by the validation command. Network availability and exact historical upstream content cannot be guaranteed. Do not rerun model selection against spent June outcomes. A future study needs a new frozen protocol and genuinely unexamined compatible evaluation data.

## Research directions

The following are future research objectives, not capabilities already validated
by the v4 benchmark:

1. Establish compatible, complete coordinate-bearing evaluation data beyond
   June, including an independently frozen holiday/event window.
2. Audit the availability time of completed-trip aggregates before testing an
   operational as-of forecast feed.
3. Evaluate calibrated count-prediction intervals on new chronological cohorts,
   with explicit regional and horizon coverage checks.
4. Assess domain shift and difficult sparse/night/weekend segments without
   changing selection rules in response to spent final-test outcomes.

Contributions should include a frozen evaluation protocol, causal feature
checks, reproducible environment information and evidence beyond a single
aggregate score. New candidates should be published under a new artifact ID;
the selected v4 artifact and its sealed evaluation evidence remain immutable.

## Operational relationship

This research checkout owns experiments and evidence. The deployment checkout owns a compact checked-in runtime bundle, genuine observation seed, stable API contract, cinematic frontend, Compose definition and integration tests. Neither depends on the other's local filesystem path. No secrets, environment directories, raw trip exports or generated model-search checkpoints should be committed. Hashes identify the selected artifact in both repositories; no model was retrained during final integration.

## License and third-party materials

Original UrbanFlow AI software and documentation are available under the
[MIT License](LICENSE), copyright © 2026 Soham Mahesh Gaonkhadkar.
Third-party photographs, source datasets, map services, fonts and dependencies
retain their applicable terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

## Author

**Soham Mahesh Gaonkhadkar** · [GitHub](https://github.com/Sohamgaonkhadkar)
