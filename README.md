# UrbanFlow AI: Distributed Spatiotemporal Forecasting for NYC Mobility

## 1. Executive Summary & System Design
UrbanFlow AI is an enterprise-grade spatiotemporal forecasting platform designed to solve the fleet optimization problem for high-density urban transit systems. Utilizing over **33 million raw records** from the NYC Yellow Taxi historical dataset (spanning January to March 2016), the architecture transforms discrete, non-stationary geospatial transactions into a regularized, high-dimensional time-series matrix. 

The processing layer scales via **Dask distributed dataframes** to bypass native hardware bottlenecks, segments the urban grid into **30 optimized regional clusters**, applies strict context-aware imputation to retain structural temporal integrity (completely avoiding the loss of historical rows), and deploys a fine-tuned, out-of-core **LightGBM Regressor** optimized via multi-stage **Optuna studies** and tracked dynamically through **MLflow**.

```text
                       [ 8GB Raw CSVs (Jan-Mar 2016) ]
                                      │
                                      ▼
                       [ Dask Distributed Ingestion ]
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
     [ Percentile-Based Filtering ]           [ MiniBatch K-Means (k=30) ]
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                         [ 15-Min Resampling Grid ]
                                      │
                                      ▼
                       [ Feature Engineering Pipeline ]
                    (Cyclic, Multi-Resolution Lags, EWMA)
                                      │
                                      ▼
                     [ Strict TimeSeriesSplit Validation ]
                                      │
                                      ▼
                      [ Optuna Hyperparameter Tuning ]
                                      │
                                      ▼
                        [ Final LightGBM Inference ]
```

---

## 2. Distributed Ingestion and Robust Outlier Elimination
Processing an 8GB text corpus natively using traditional Pandas data structures causes immediate catastrophic memory degradation and system kernel panics due to the high-dimensional allocation overhead. To overcome this limitation, the system implements `dask.dataframe` to construct a lazy evaluation graph, processing data in parallel blocks.

### 2.1 Multi-Dimensional Truncation and Data Cleaning
Raw transportation data is highly prone to recording noise, including sensor malfunctions, coordinate jumps, and manual logging faults. Rather than relying on simple arbitrary truncations, the data boundaries were rigorously established by computing the 99.9th percentile ($P_{99.9}$) of the operational empirical distributions.

The filtering boundary rules are mathematically enforced as follows:

* $0.25 \le \text{trip distance} \le 24.43$ miles
* $0.50 \le \text{fare amount} \le 81.00$ dollars
* $40.60 \le \text{pickup latitude} \le 40.85$
* $-74.05 \le \text{pickup longitude} \le -73.70$

This strict spatial bounding box encapsulates the geographic boundaries of Manhattan, Brooklyn, Queens, and the major airport hubs (JFK and LaGuardia), dropping non-viable coordinates.

---

## 3. Spatial Intelligence: 30-Region Density Clustering
Direct coordinate regression on raw latitude and longitude variables is mathematically unstable due to high localized variance. To transform a continuous geospatial coordinate domain into discrete, trackable demand nodes, the spatial envelope was quantized into 30 distinct functional regions using `MiniBatchKMeans`.

### 3.1 Algorithmic Formulation
The clustering model partitions the coordinates into $k=30$ clusters by minimizing the within-cluster sum of squares (inertia):

$$
\arg\min_{\mathbf{S}} \sum_{i=1}^{30} \sum_{\mathbf{x} \in S_i} \| \mathbf{x} - \boldsymbol{\mu}_i \|^2
$$

Where $\mathbf{x} = [\text{latitude}, \text{longitude}]^T$ represents the location vector of an isolated pickup event, and $\boldsymbol{\mu}_i$ represents the centroid vector of the spatial cluster $S_i$.

Unlike traditional K-Means, which computes distances across the entire dataset globally per iteration, `MiniBatchKMeans` uses random sub-samples to update cluster centers step-by-step. This dramatically reduces computation time on 33 million rows while converging cleanly to identical cluster centroids.

## Why 30 Regions?

Instead of selecting K using only the Elbow Method,
a geographic criterion based on Haversine distance was used.

A good operational zone should have nearby neighboring regions
within roughly 1–1.5 miles.

K=30 maximized the percentage of clusters satisfying this condition,
while avoiding:

- Oversized regions (K too small)
- Sparse unstable regions (K too large)

This produced geographically meaningful dispatch zones.
> 

---

## 4. Advanced Feature Engineering and Temporal Continuity
The raw transactions are aggregated into regularized **15-minute time intervals**. This yields a highly uniform structural shape: **262,080 rows** across the operational timeline.

### 4.1 The Destruction of dropna() and the Continuity Strategy
Standard data science approaches often execute a blanket `.dropna()` command following the creation of time-lagged variables. However, doing so in a spatiotemporal time-series model deletes critical training context, creating artificial temporal discontinuities.

To preserve the true cyclical nature of real-world demand patterns, **no rows were dropped.** Instead, an automated boolean feature flagging network was implemented to denote historical lag data availability. For instance, when constructing a 720-hour lag (one month backward looking), if the target interval lies in the first month of the dataset, the value defaults to an imputation value while the structural flag feature `lag_720_available` is set to `0`. This informs the decision trees exactly when a value is true vs. when it is imputed.

### 4.2 Rolling Mean vs EWMA

Taxi demand data is inherently noisy, making smoothing an important step before creating temporal features.

Two approaches were evaluated:

- Rolling Mean
- Exponentially Weighted Moving Average (EWMA)

Rolling Mean treats all observations within a window equally, producing stable trend estimates. EWMA assigns higher weight to recent observations, making it more responsive to short-term fluctuations.

Although EWMA achieved lower smoothing errors at high alpha values, this behavior was largely driven by short-term autocorrelation. At α values close to 1, EWMA effectively behaves like a "last value" predictor, which improves smoothing metrics but does not necessarily create better forecasting features.

To determine which approach generalized better, both feature sets were evaluated using an identical XGBoost pipeline with the same train-test split and hyperparameter optimization strategy.

| Feature Set | Test MAPE | Test RMSE |
|------------|-----------|-----------|
| Rolling Mean Features | 28.11% | 20.41 |
| EWMA Features | 30.91% | 21.40 |

#### Why Rolling Mean Was Selected Over EWMA

Although EWMA achieved lower smoothing errors at high alpha values, these improvements were largely driven by strong dependence on the most recent observation.

As alpha approaches 1, EWMA increasingly behaves like a "last-value predictor", exploiting short-term autocorrelation rather than capturing broader temporal demand patterns.

Rolling Mean, in contrast, aggregates information across a wider historical window and produces more stable trend estimates. This makes the resulting features less sensitive to temporary fluctuations and better suited for forecasting future demand.

A controlled XGBoost experiment confirmed this observation. Using identical train-test splits and model configurations, Rolling Mean features achieved lower MAPE and RMSE than EWMA features on unseen March data.

This indicated that Rolling Mean captured more generalizable temporal structure and therefore served as the primary smoothing strategy in the final production pipeline.

Rolling Mean consistently achieved lower forecasting error and stronger generalization on unseen demand data.

For this reason, Rolling Mean windows were selected as the primary smoothing features of the production pipeline, while EWMA (α = 0.9) was retained as an additional supporting feature (`avg_pickups`).

### 4.3 Mathematical Formalization of the Feature Matrix
As verified by the serialized pipeline artifact `lgbm_taxi_pipeline.pkl`, the structural feature matrix is segmented into five core analytical layers:

#### A. Multi-Resolution Historical Lags
* **Micro Lags:** `lag_1`, `lag_2`, `lag_3`, `lag_4` (capturing immediate past 15, 30, 45, and 60 minutes).
* **Macro Lags:** `lag_24` (daily cycle), `lag_168` (weekly cycle), `lag_336` (bi-weekly cycle), `lag_720` (monthly baseline).
* **Cross-Spatial Lags:** `region_lag_1`, `region_lag_24` (tracking localized regional momentum shifts).

#### B. Dynamic Statistical Rolling Windows
* **Moving Averages:** `rolling_mean_24`, `rolling_mean_168`, `rolling_mean_336`, `rolling_mean_720`.
* **Volatility Trackers:** `rolling_std_24`, `rolling_std_168`.
* **Operational Segmentation:** `rolling_mean_weekend_24` and `rolling_mean_weekday_24` (preventing weekend leisure demand from corrupting business-day baseline trends).

#### C. Smooth Demand Momentum (EWMA)
To accurately isolate system momentum without introducing severe phase delays or overfitting to immediate random fluctuations, an Exponentially Weighted Moving Average ($\alpha = 0.4$) was computed. To prevent future-to-past data leakage, the feature was strictly shifted forward by one step:

$$
\hat{\mu}_{t} = \alpha y_{t-1} + (1-\alpha)\hat{\mu}_{t-1}
$$

This is explicitly tracked in the dataset as the `avg_pickups` feature column.

#### D. Topological Cyclic Encodings
Representing temporal variables like `hour` (0–23) or `day_of_week` (0–6) as simple linear integer variables creates a false mathematical distance between the final hour of one day and the first hour of the next. To preserve continuous time topology, these variables were mapped onto a 2D coordinate space via trigonometric transformations:

$$
H_{\text{sin}} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad H_{\text{cos}} = \cos\left(\frac{2\pi \cdot h}{24}\right)
$$

$$
D_{\text{sin}} = \sin\left(\frac{2\pi \cdot d}{7}\right), \quad D_{\text{cos}} = \cos\left(\frac{2\pi \cdot d}{7}\right)
$$

#### E. Spatiotemporal Exogenous Overlays
* **Calendar Tracking:** `is_weekend`, `is_peak_hour`, `is_night`, `is_holiday`.
* **Proximity Features:** `days_to_holiday`, `days_after_holiday`, `holiday_name`.
* **High-Volatility Urban Events:** `is_restaurant_week`, `is_fashion_week`, `is_nyc_half_marathon`.

---

## 5. Comparative Modeling: Baseline vs. Advanced Ensembles
The development pipeline moved systematically from baseline tracking to complex gradient boosted architectures.

### 5.1 Baseline Benchmark Evaluation
Initial model exploration evaluated standard foundational configurations:
* **Linear Regression & Ridge Regression:** Suffered from high bias, failing to capture complex non-linear spatial interactions.
* **Support Vector Regression (SVR):** Computationally untenable, exhibiting catastrophic $O(n^3)$ training complexity on large datasets.
* **Random Forest Regressor:** Formed the initial performance baseline, yielding a **Test MAPE of 30.69%**.

### 5.2 The Gradient Boosting Showdown
Three advanced gradient-boosting architectures were directly pitted against each other over identical time-series splits:

* **XGBoost Regressor:** Evaluated extensively across both RMSE and MAE optimization metrics. While exceptionally accurate, its level-wise tree splitting mechanism required significant memory allocations to hold dense, high-dimensional matrices. This resulted in long training times and high resource utilization on large datasets.
* **CatBoost Regressor:** Delivered robust native handling for the categorical region IDs. However, its symmetric (oblivious) tree architecture increased production inference latency, conflicting with real-time prediction constraints.
* **LightGBM Regressor (Selected Architecture):** LightGBM utilizes a leaf-wise (best-first) tree growth strategy rather than level-wise growth. This choice proved highly optimal for several key reasons:
  * **Leaf-wise Optimization:** Splitting the single leaf with max delta loss drastically reduces training error compared to traditional level-wise algorithms.
  * **Native Missing Value Handling:** LightGBM automatically isolates missing values into the optimal split direction based on training loss. This perfectly suited our strategy of avoiding `dropna()`, keeping historical continuity completely intact.

---

## 6. Hyperparameter Tuning & Leakage-Proof Validation
To guarantee that the model generalizes robustly to unobserved future demand patterns, the traditional random K-Fold cross-validation method was abandoned. Random data partitioning introduces massive future-to-past data leakage by shuffling chronological data points. Instead, a strict forward-chaining `TimeSeriesSplit` was enforced.

### 6.1 Production Gap Engineering
A major real-world challenge in deploying forecasting systems is downstream data pipeline latency. If a model requires immediate past-hour data to make a prediction, but the live pipeline takes two hours to collect and process data from vehicles, the model will fail in production.

To mitigate this, a strict operational gap was implemented: `TimeSeriesSplit(n_splits=5, gap=8)`. The `gap=8` skips exactly eight 15-minute intervals (2 hours) between the training fold and the validation fold. This ensures the model learns to forecast without relying on immediate real-time data that would not yet be available in a live production environment.

### 6.2 Multi-Stage Optuna Optimization
Hyperparameter tuning was handled via Optuna across distinct structural zones to prevent coordinate-descent traps. The optimized parameters stored within the final production artifact `lgbm_taxi_pipeline.pkl` converged to the following values:

```python
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
    }
    }
```
### LightGBM Final Hyperparameters

```yaml
boosting_type: gbdt
num_leaves: 55
max_depth: 10
learning_rate: 0.0412
n_estimators: 1200
min_child_samples: 35
reg_alpha: 0.145
reg_lambda: 1.821
```

# 7. Metrics & Enterprise Performance Evaluation

The choice of optimization objectives during tuning heavily impacts real-world performance. The engine was systematically evaluated against multiple loss criteria.

## 7.1 Optimization Objectives: RMSE vs. MAE

During the tuning phase, optimization runs were evaluated using both Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

### The MAE Blindspot

Tuning purely on MAE creates a model that accurately predicts median behavior but underpredicts high-volatility demand spikes.

### The RMSE Mandate

The final model was optimized against RMSE. Because RMSE quadratically penalizes large errors, it forces the model to heavily penalize large misses.

In urban transport, missing a major demand surge by 100 cars at an airport causes systemic vehicle shortages and lost revenue. Under-predicting 100 scattered pickups by 1 car each is minor by comparison. RMSE optimization forces the tree splits to capture these high-volume demand peaks.

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

## 7.2 The WAPE Metric Framework

While standard Mean Absolute Percentage Error (MAPE) is common in data science, it breaks down in real-world spatial forecasting. When local demand drops to zero (such as quiet residential areas during early morning hours), the actual value (y_i) enters the denominator, causing the error metric to divide by zero and explode toward infinity.

To prevent this distortion, the final system performance is evaluated using WAPE (Weighted Absolute Percentage Error):

$$
WAPE = \frac{\sum_{i=1}^{n}|y_i - \hat{y}*i|}{\sum*{i=1}^{n}y_i}
$$

By calculating the total absolute deviation over the total aggregate demand volume, WAPE provides a stable, volume-weighted percentage metric that accurately reflects true operational performance across all 30 regions.

## 7.3 Final Model Performance Analytics

The production-grade LightGBM model logged via MLflow achieved the following metrics on unseen test data:

| Metric Framework                          | Empirical Score Value | Operational Insight                                                       |
| ----------------------------------------- | --------------------- | ------------------------------------------------------------------------- |
| Mean Absolute Error (MAE)                 | 12.41                 | Average error is limited to approximately 12 vehicle allocations per zone |
| Root Mean Square Error (RMSE)             | 20.04                 | Demonstrates excellent control over extreme variance and surge errors     |
| Coefficient of Determination (R²)         | 0.9764                | The feature matrix accounts for 97.64% of true urban demand variance      |
| Weighted Absolute Percentage Error (WAPE) | 9.41%                 | The final forecasting accuracy sits at an elite 90.59%                    |

# 8. Repository Topology & Production Deployment

```text
urbanflow-ai/
├── data/
│   ├── train_new.csv                  # Regularized training time-series split
│   └── test_new.csv                   # Holdout test time-series split
├── notebooks/
│   ├── Removing Outliers.ipynb         # Bounding-box logic & outlier truncation
│   ├── Breaking_NYC_to_Regions.ipynb   # MiniBatch K-Means clustering (30 zones)
│   ├── EDA-Demand-Prediction.ipynb     # Dask ingestion & initial distribution profiles
│   ├── Creating-Historical-Data.ipynb  # 15-min resampling, cyclic & EWMA feature mapping
│   ├── Training-Baseline-Model.ipynb   # Foundational model tracking (Random Forest baseline)
│   ├── Model-Selection.ipynb           # Cross-framework modeling & MLflow experimentation
│   ├── catboost.ipynb                  # CatBoost tuning & metrics logging
│   └── final model.ipynb               # Final production model tuning pipeline
├── production/
│   └── lgbm_taxi_pipeline.pkl          # Serialized, hyper-optimized LightGBM pipeline
└── README.md                           # Repository Documentation Architecture
```

## 8.1 Production Initialization Code

To use the serialized model pipeline for inference, load the saved artifact directly using `joblib` or `pickle`.

```python
import pandas as pd
import joblib

# Load the production pipeline artifact
pipeline = joblib.load("production/lgbm_taxi_pipeline.pkl")

# Input feature vectors must align with the 76 columns established in the model
# Predictions output direct vehicle demand metrics per region
predicted_demand = pipeline.predict(inference_feature_matrix)
```

## Academic Context

Developed by **Soham Mahesh Gaonkhadkar**
Department of Chemical Engineering
Indian Institute of Technology Kharagpur

This repository serves as a comprehensive demonstration of large-scale data engineering, rigorous statistical modeling, and full-stack deployment within complex urban systems.
