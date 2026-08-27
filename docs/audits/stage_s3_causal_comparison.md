# Stage S3 causal-comparison contract

This report is intentionally a pre-result contract until the frozen CUDA gate
and six runs complete. It prevents result-dependent tuning.

## Metrics and thresholds

All runs use one fixed probability threshold (`0.3`). Segmentation evaluation
reports road precision, recall, F1, IoU and AUPRC plus junction precision,
recall, F1, IoU and AUPRC. Anchor evaluation reports four per-step recalls,
top-K recall, localization error, false-positive count, missed-branch count,
and channel diversity. No run receives a separately optimized threshold.

The best checkpoint rule is the frozen-validation segmentation composite:
mean road F1, road IoU, and junction F1. Validation batches come from the
disjoint validation extent and never enter the training stream.

## Predeclared interpretation

C1 versus C0 is the indirect causal screen. It is `PROMISING` only if at least
two of road F1, road IoU, and junction F1 improve, the mean C1 gain exceeds
both C2 and C3, and precision/recall does not collapse. C2 controls for the
extra raster parameters. C3 controls for trajectory coverage statistics
without correct spatial registration.

C1 anchor evidence is `PROMISING` only if top-K recall loses no more than 0.5
percentage points and at least one major anchor metric improves. If
segmentation improves but anchor does not, the required conclusion is:
“轨迹提高了 segmentation，但尚未证明可传递到 anchor。”

J1 versus J0 estimates the joint-optimization ceiling. A J1-only improvement
cannot be described as proof that segmentation indirectly helped anchor.

Closed-loop graph exploration for C0/C1/J0/J1 is conditional: it is not run
unless an aligned raster group beats its image-only segmentation control. If
the gate does not pass, the result is `NOT_EXECUTED_BY_GATE`.

This stage uses one seed. Results may be labeled only `PROMISING`,
`INCONCLUSIVE`, `NO_EVIDENCE`, or `REGRESSION`; no statistical-significance
claim is permitted. A separately authorized three-seed replication is proposed
only when the C1 causal screen or J1 joint screen is `PROMISING`.
