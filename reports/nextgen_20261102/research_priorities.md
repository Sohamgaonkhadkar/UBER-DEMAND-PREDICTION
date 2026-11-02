# Priorities fixed before novel model validation

April is diagnostic-only. All new correction/gate coefficients use earlier OOF
data; April forecast errors will not tune or select their weights. March is a
spent development-validation period. May is the new sealed chronological test.

| Rank / family | Evidence and hypothesis | Likely benefit | Cost | Complexity | Overfit / leakage risk | Compatibility |
| --- | --- | --- | --- | --- | --- | --- |
| 1 / E1: seasonal profile reliability | Late calendar bias is about −3%; regional levels changed. Current smoothing wraps 23:45 to 00:00 of the same weekday instead of the next weekday. Test correct weekly adjacency, then a shrinkage/recency profile. | Modest, plausible; no numerical gain assumed | Seconds; reuse trained GBDT | Low | Low-medium; profiles fit only before cutoff | Same recursive API; final predictions feed back |
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
