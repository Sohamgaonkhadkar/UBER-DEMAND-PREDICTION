# Literature and strategy review

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
