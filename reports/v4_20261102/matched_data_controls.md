# Matched-data controls fixed before final selection and holdout access

Add two expanding-history controls to development: the strongest prior pure
LightGBM and the original calendar-feedback architecture, using the same valid
historical label timestamps as expanding XGBoost and unchanged tree capacity.
They are data controls, not additional architecture/hyperparameter trials.
They face the same promotion gate as other changed data recipes.

If a changed-window/history/holiday recipe is selected, predeclare additional
final XGBoost, LightGBM and calendar references fitted with that same selected
label window and boosted-fit case weights. Calendar profiles retain the
original unweighted weekday median/smoothing over valid label timestamps;
case weights apply to the boosted fit only. Preserve the Jan-only references
too. This separates a data-window advantage from a forecasting-method claim.
No final target or historical-arm model outcome was inspected to add these
fairness controls. The final primary June dates/metric are unchanged.
