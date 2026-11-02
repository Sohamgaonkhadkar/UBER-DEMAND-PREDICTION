# Research plan fixed before new model outcomes

March, April and May are development evidence. June is the primary new test;
July 1–7 is a separate, prespecified holiday confirmation window using the same
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
