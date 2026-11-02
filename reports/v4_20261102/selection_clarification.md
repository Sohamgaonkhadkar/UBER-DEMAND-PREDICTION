# Regional gate clarification before any development model outcomes

The general promotion gate requires improvements in at least 20 regions. A
specialist intentionally changes only training-defined sparse regions and
cannot satisfy that breadth criterion. For the sparse-NB arm only, replace the
breadth criterion with improvement in **every** training-defined sparse region.
The aggregate gain >=0.1 point, positive paired uncertainty interval, improvement
in at least two months and <=2-point worst regional deterioration still apply.
Sparse-only improvement never overrides the aggregate gate. No development
outcome or final target was inspected to make this logical clarification.

Among already-qualified recipes, prefer fewer inference components when within
0.1 WAPE point and the pairwise paired95% gain interval includes zero. Ordering:
single booster/window, then calendar-feature/training-weight changes, then
specialist/fixed seasonal omission or one coarse-disaggregation model, then
pooled holiday profile, then temporal/cross-temporal combinations. This tie
rule implements the user's simplicity preference before any historical-arm
development outcome or final target is examined.
